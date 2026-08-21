from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from lld.main import create_app


@pytest.mark.asyncio
async def test_health_ok():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body


@pytest.mark.asyncio
async def test_settings_roundtrip(tmp_path, monkeypatch):
    from lld import settings as settings_mod

    state = tmp_path / "llamadeck"
    state.mkdir()
    monkeypatch.setattr(settings_mod, "STATE_DIR", state)
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", state / "settings.json")
    monkeypatch.setattr(settings_mod, "LOGS_DIR", state / "logs")

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert data["controller_bind_port"] == 8770
        assert data["llama_server_url"] == "http://localhost:8080"


# --- crash detection without an exit watcher --------------------------------

def test_a_zombie_is_not_running():
    """`psutil.pid_exists()` says yes for a process that has already exited.
    On 2026-08-20 the 5090 locked up (NVRM Xid 8), llama-server became a
    zombie, and the preset kept reporting "running" while its port refused
    every connection — so nothing restarted it."""
    import psutil

    from lld.supervisor import _pid_alive

    class _Zombie:
        def __init__(self, pid): pass
        def status(self): return psutil.STATUS_ZOMBIE

    real = psutil.Process
    psutil.Process = _Zombie
    try:
        assert _pid_alive(1234) is False
    finally:
        psutil.Process = real


def test_liveness_poll_notices_and_restarts_an_adopted_crash(monkeypatch):
    """An adopted handle has no `_watch_exit` task. The poll is what turns a
    silent death into a restart — but only for a process we parented."""
    from lld.settings import LlamaServerConfig
    from lld.supervisor import MultiSupervisor, ProcessHandle

    sup = MultiSupervisor("llama-server")
    h = ProcessHandle("p", LlamaServerConfig(name="p", model_path="/x.gguf"), "llama-server")
    h._adopted = True
    h._pid = 4242
    sup._handles["p"] = h

    monkeypatch.setattr("lld.supervisor._pid_alive", lambda _pid: False)
    monkeypatch.setattr(ProcessHandle, "parented_by_us", lambda self: True)
    scheduled: list[str] = []
    monkeypatch.setattr(ProcessHandle, "_schedule_autorestart",
                        lambda self: scheduled.append(self.preset_name))

    assert sup.poll_liveness() == ["p"]
    assert scheduled == ["p"]
    assert h.is_running() is False
    # Already noticed — a second tick must not schedule another restart.
    assert sup.poll_liveness() == []
    assert scheduled == ["p"]


def test_an_externally_owned_process_is_noticed_but_not_restarted(monkeypatch):
    """A llama-server someone else started is theirs to restart; LlamaDeck
    still has to stop calling it running."""
    from lld.settings import LlamaServerConfig
    from lld.supervisor import MultiSupervisor, ProcessHandle

    sup = MultiSupervisor("llama-server")
    h = ProcessHandle("ext", LlamaServerConfig(name="ext"), "llama-server")
    h._adopted = True
    h._pid = 999
    sup._handles["ext"] = h

    monkeypatch.setattr("lld.supervisor._pid_alive", lambda _pid: False)
    monkeypatch.setattr(ProcessHandle, "parented_by_us", lambda self: False)
    scheduled: list[str] = []
    monkeypatch.setattr(ProcessHandle, "_schedule_autorestart",
                        lambda self: scheduled.append(self.preset_name))

    assert sup.poll_liveness() == ["ext"]
    assert scheduled == []
    assert h.is_running() is False


def test_the_poll_defers_to_an_existing_exit_watcher(monkeypatch):
    """A process this backend spawned already has `_watch_exit` awaiting it.
    Letting the poll act as well spent two of the three restarts the 5-minute
    window allows, so one crash loop hit the cap in half the time."""
    from lld.settings import LlamaServerConfig
    from lld.supervisor import MultiSupervisor, ProcessHandle

    sup = MultiSupervisor("llama-server")
    h = ProcessHandle("own", LlamaServerConfig(name="own"), "llama-server")
    h._pid = 4242
    h._proc = object()          # spawned here → _watch_exit owns the transition
    sup._handles["own"] = h

    monkeypatch.setattr("lld.supervisor._pid_alive", lambda _pid: False)
    scheduled: list[str] = []
    monkeypatch.setattr(ProcessHandle, "_schedule_autorestart",
                        lambda self: scheduled.append(self.preset_name))

    assert sup.poll_liveness() == []
    assert scheduled == []


def test_the_crash_loop_note_is_recorded_once(monkeypatch):
    """The suppression note was appended on every suppressed attempt, so the
    error read '... (crash-looped 3× in 5min) (crash-looped 3× in 5min)'."""
    from lld.settings import LlamaServerConfig
    from lld.supervisor import ProcessHandle

    h = ProcessHandle("p", LlamaServerConfig(name="p"), "llama-server")
    h._last_error = "llama-server failed: CUDA error"
    h._autorestart_attempts_window = [1e9, 1e9, 1e9]
    monkeypatch.setattr("lld.supervisor.time.time", lambda: 1e9 + 1)

    h._schedule_autorestart()
    h._schedule_autorestart()
    assert h._last_error.count("crash-looped") == 1


# --- alive but wedged --------------------------------------------------------

async def test_loading_is_not_treated_as_a_hang(monkeypatch):
    """llama.cpp answers /health with 503 while it loads the model. Counting
    that as a failure would restart a preset in the middle of loading — over
    and over, since the restart puts it right back into loading."""
    import httpx

    from lld import health_watchdog as hw

    class _Client:
        def __init__(self, **_kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_a): return False
        async def get(self, _url): return httpx.Response(503)

    monkeypatch.setattr(hw.httpx, "AsyncClient", _Client)
    ok, detail = await hw._probe_alive("http://127.0.0.1:8080")
    assert ok is True and detail == "loading"


async def test_an_unreachable_port_is_a_failure(monkeypatch):
    """The shape a GPU fault leaves behind: the process is alive, so the
    liveness poll is happy, and the port answers nothing."""
    import httpx

    from lld import health_watchdog as hw

    class _Client:
        def __init__(self, **_kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_a): return False
        async def get(self, _url): raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(hw.httpx, "AsyncClient", _Client)
    ok, detail = await hw._probe_alive("http://127.0.0.1:8080")
    assert ok is False and "ConnectError" in detail
