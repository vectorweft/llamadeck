"""Stop has to mean stopped.

A Stop from the UI terminates the process with SIGTERM, which is a non-zero
exit — indistinguishable, to the exit watcher, from the crash the auto-restart
exists for. So the preset came back thirty seconds later, on its own, and the
app looked like it was refusing to stay stopped. From a real session log:

    11:14:36  [router-8085] process 25379 died unnoticed
    11:14:36  POST /api/server/stop/router-8085 200 OK
    11:15:06  [router-8085] auto-restart attempt after unexpected exit

Thirty seconds is exactly the cooldown. These tests pin the guard, and the
third one pins the thing the guard must not break: a genuine crash still
restarts.
"""
from __future__ import annotations

import asyncio
import stat

from lld.settings import LlamaServerConfig, ensure_state_dirs
from lld.supervisor import ProcessHandle


def _fake_server(tmp_path, script: str, name: str = "fake-llama-server") -> str:
    """`--help` returns at once: the flag catalog probes the binary before every
    start, and a script that sleeps through that just times the probe out."""
    p = tmp_path / name
    p.write_text('#!/bin/sh\ncase "$1" in --help) exit 0;; esac\n' + script)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def _handle(binary: str, port: int) -> ProcessHandle:
    ensure_state_dirs()   # start() opens a log file under LOGS_DIR
    cfg = LlamaServerConfig(name="t", model_path=None, host="127.0.0.1", port=port)
    return ProcessHandle("t", cfg, binary)


async def _settle() -> None:
    """Let the exit watcher run — it races stop() and decides the restart."""
    for _ in range(200):
        await asyncio.sleep(0.01)


async def test_the_liveness_poll_does_not_revive_a_stopped_preset(tmp_path):
    """The half stop() does not cover on its own.

    When the process has an exit watcher, stop() cancels it along with the log
    readers and the restart never gets queued. After the backend re-execs there
    is no watcher — execv cannot re-attach an asyncio child — so the liveness
    poll owns death detection and calls _schedule_autorestart() itself, with no
    idea that a Stop is what killed the process. That is the call the flag has
    to stop, and it is the one the router came back through.
    """
    h = _handle(_fake_server(tmp_path, "sleep 30\n"), 18781)
    await h.start()
    await h.stop()
    await _settle()
    assert not h.is_running()
    assert h._autorestart_task is None, "a Stop queued an auto-restart"

    h._schedule_autorestart()   # what poll_liveness() would call
    assert h._autorestart_task is None, "the liveness poll revived a stopped preset"


async def test_a_stop_during_the_cooldown_cancels_the_pending_restart(tmp_path):
    """The crash comes first, the Stop lands while the restart is still queued.
    Cancelling is the only thing that makes that Stop stick."""
    h = _handle(_fake_server(tmp_path, "exit 1\n"), 18782)
    await h.start()
    await _settle()
    assert h._autorestart_task is not None, "a crash should have queued a restart"

    await h.stop()
    await asyncio.sleep(0.05)
    assert h._autorestart_task.cancelled() or h._autorestart_task.done()
    assert not h.is_running()


async def test_a_crash_still_restarts(tmp_path):
    """The guard must not disable the feature it is narrowing."""
    h = _handle(_fake_server(tmp_path, "exit 1\n"), 18783)
    await h.start()
    await _settle()

    assert h._autorestart_task is not None
    assert not h._stop_requested
    h._autorestart_task.cancel()  # don't leave a 30s timer behind


async def test_start_clears_the_stop_flag(tmp_path):
    """Otherwise the first Stop would suppress every later crash-restart."""
    binary = _fake_server(tmp_path, "sleep 30\n")
    h = _handle(binary, 18784)
    await h.start()
    await h.stop()
    assert h._stop_requested

    await h.start()
    assert not h._stop_requested
    await h.stop()
