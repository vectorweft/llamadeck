"""`llama_bin` is the one field that decides *what* LlamaDeck executes.

`to_argv` forces every preset's program to the configured binary and discards
whatever the raw command box names (argv.py:command_argv); spawning goes
through create_subprocess_exec, so no shell parses anything. That leaves
`argv_override` supplying arguments only — and makes `llama_bin` the single
pivot in the chain the 2026-09-07 audit walked: set it to /bin/sh, then a
preset's raw command is `-c '<anything>'`.

This is depth, not a boundary — LlamaDeck runs as the user and its job is to
start processes. What these tests pin is that the documented chain fails
closed at both ends, that the sink check cannot be walked around by editing
settings.json, and that a legitimate wrapper script is still allowed.
"""
from __future__ import annotations

import os
import stat

import pytest
from httpx import ASGITransport, AsyncClient

from lld.procutil import llama_bin_rejection


@pytest.fixture()
def app(tmp_path, monkeypatch):
    from lld import net_guard
    from lld import settings as settings_mod

    state = tmp_path / "llamadeck"
    state.mkdir()
    monkeypatch.setattr(settings_mod, "STATE_DIR", state)
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", state / "settings.json")
    monkeypatch.setattr(settings_mod, "LOGS_DIR", state / "logs")
    monkeypatch.setattr(net_guard, "SETTINGS_PATH", state / "settings.json")
    monkeypatch.setattr(net_guard, "_extra_cache", None)

    from lld.main import create_app
    return create_app()


# --- the check itself --------------------------------------------------------

@pytest.mark.parametrize("path", ["/bin/sh", "/bin/bash", "/usr/bin/python3",
                                  "/usr/bin/env", "/usr/bin/sudo"])
def test_an_interpreter_is_refused(path):
    """These turn arguments into code, which is exactly what the preset's raw
    command box supplies."""
    assert "not llama-server" in (llama_bin_rejection(path) or "")


def test_a_symlink_to_a_shell_is_refused(tmp_path):
    """Judging the name the user typed is not enough — /usr/local/bin/llama-server
    can be a symlink to bash, and only the target says so."""
    link = tmp_path / "llama-server"
    link.symlink_to("/bin/bash")
    assert llama_bin_rejection(str(link)) is not None


def test_a_wrapper_script_is_still_allowed(tmp_path):
    """Pointing llama_bin at your own shell script is a normal thing to do —
    it is how people set env vars llama.cpp has no flag for. The shebang execs
    fine, and the blocklist must never reach it."""
    wrapper = tmp_path / "llama-wrapper.sh"
    wrapper.write_text("#!/bin/sh\nexec /opt/llama.cpp/build/bin/llama-server \"$@\"\n")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    assert llama_bin_rejection(str(wrapper)) is None


def test_a_path_that_is_not_there_yet_is_not_an_error():
    """A fresh install has no binary and the setup wizard has to be reachable
    without one — see the boot log's 'run the setup wizard' branch."""
    assert llama_bin_rejection("/opt/not-built-yet/llama-server") is None
    assert llama_bin_rejection("") is None
    assert llama_bin_rejection(None) is None


def test_present_but_unusable_is_named(tmp_path):
    """Not security, just a better error than llama-server exiting 1 three
    times and being declared crash-looped."""
    d = tmp_path / "build"
    d.mkdir()
    assert "directory" in (llama_bin_rejection(str(d)) or "")
    f = tmp_path / "llama-server"
    f.write_text("")
    os.chmod(f, 0o644)
    assert "not executable" in (llama_bin_rejection(str(f)) or "")


# --- the door ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_settings_refuses_a_shell(app):
    """The audit's step one: PUT /api/settings with llama_bin=/bin/sh returned
    200 and saved it. The setup wizard's /use-binary always vetted its
    candidate; this endpoint took anything."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1") as c:
        current = (await c.get("/api/settings")).json()
        current["llama_bin"] = "/bin/sh"
        resp = await c.put("/api/settings", json=current)
    assert resp.status_code == 400
    assert "not llama-server" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_put_settings_still_saves_a_normal_binary(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1") as c:
        current = (await c.get("/api/settings")).json()
        current["llama_bin"] = "/opt/llama.cpp/build/bin/llama-server"
        resp = await c.put("/api/settings", json=current)
    assert resp.status_code == 200
    assert resp.json()["llama_bin"] == "/opt/llama.cpp/build/bin/llama-server"


# --- the sink ----------------------------------------------------------------

def test_the_supervisor_refuses_to_spawn_a_shell():
    """settings.json is a plain file. Rejecting the API call alone would be a
    check on the polite route only — this is the one that has to hold."""
    from lld.settings import LlamaServerConfig
    from lld.supervisor import ProcessHandle

    h = ProcessHandle("p", LlamaServerConfig(name="p", argv_override="-c 'id > /tmp/pwned'"),
                      "/bin/sh")
    assert "not llama-server" in (h._missing_paths() or "")
