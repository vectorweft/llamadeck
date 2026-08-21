"""Cross-platform process spawning and killing.

The sidecar services (ComfyUI, TTS) start a Python that spawns its own
children, so "stop" has to take down a whole tree, not one PID. The POSIX way
(setsid + killpg) has no Windows equivalent, and `signal.SIGKILL` does not
even exist there — referencing it crashes with AttributeError. Everything that
kills something goes through this module so there is exactly one place that
knows the difference.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
from typing import NamedTuple, Sequence

import psutil

log = logging.getLogger(__name__)

IS_WINDOWS = os.name == "nt"


class CommandResult(NamedTuple):
    """Outcome of a short, captured command.

    `rc` is None whenever there is no exit status to report — the command timed
    out, or never started at all. Which of the two is in `timed_out` /
    `error`, so a caller that wants to say *why* it has no answer can.
    """

    rc: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.rc == 0

    @property
    def text(self) -> str:
        """stdout+stderr, for tools that print their banner to either."""
        return (self.stdout + self.stderr).strip()


async def _kill_and_reap(proc: asyncio.subprocess.Process) -> None:
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=2.0)
    except (asyncio.TimeoutError, ProcessLookupError, OSError):
        pass


async def run_capture(
    argv: Sequence[str],
    *,
    timeout: float,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> CommandResult:
    """Run a short command, capture its output, and never leave it behind.

    `asyncio.wait_for` cancels the *await*, not the process. Every probe in
    this codebase used it directly, so a vendor tool that hangs — and they do:
    a half-upgraded NVIDIA driver makes `nvidia-smi` block in the kernel for
    the better part of a minute — left its process running with nobody waiting
    on it. The power poller runs at 2 Hz, so that is a new stuck process every
    half second for as long as the condition lasts, each holding a pipe pair.

    Killing on timeout is the whole reason this helper exists; capturing text
    instead of bytes is convenience on top.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=cwd,
        )
    except (FileNotFoundError, PermissionError, OSError) as e:
        return CommandResult(None, "", "", error=str(e))
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        await _kill_and_reap(proc)
        log.warning("command timed out after %.1fs and was killed: %s", timeout, argv[0])
        return CommandResult(None, "", "", timed_out=True)
    except asyncio.CancelledError:
        await _kill_and_reap(proc)
        raise
    return CommandResult(
        proc.returncode,
        out.decode("utf-8", errors="replace"),
        err.decode("utf-8", errors="replace"),
    )


def new_process_group_kwargs() -> dict:
    """Popen/create_subprocess kwargs that put the child in its own group.

    `start_new_session=True` is preferred over `preexec_fn=os.setsid`: it does
    the same setsid(2) call, but inside the C fork handler, which is the only
    async-signal-safe option in a threaded process.
    """
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _group_signal(pid: int, sig: int) -> bool:
    """Signal the whole process group on POSIX; the tree on Windows.
    Returns False when the process is already gone."""
    try:
        if IS_WINDOWS:
            proc = psutil.Process(pid)
            targets = proc.children(recursive=True) + [proc]
            for t in targets:
                try:
                    t.kill() if sig == getattr(signal, "SIGKILL", 9) else t.terminate()
                except psutil.NoSuchProcess:
                    pass
            return True
        os.killpg(os.getpgid(pid), sig)
        return True
    except (ProcessLookupError, psutil.NoSuchProcess):
        return False
    except PermissionError:
        # Adopted process owned by another user — nothing we can do.
        log.warning("no permission to signal process group of pid %s", pid)
        return False


async def terminate_tree(pid: int, timeout: float = 15.0, label: str = "process") -> None:
    """Graceful stop of a process and its children: TERM, wait, then KILL."""
    if not psutil.pid_exists(pid):
        return
    _group_signal(pid, signal.SIGTERM)
    if await _wait_gone(pid, timeout):
        return
    log.warning("%s did not exit in %.0fs — killing", label, timeout)
    _group_signal(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
    await _wait_gone(pid, 5.0)


async def terminate_pid(pid: int, timeout: float = 10.0) -> None:
    """Stop a single (adopted) process — TERM, wait, KILL. No group involved:
    an adopted llama-server may share its group with the shell that started it.
    """
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    try:
        proc.terminate()
    except psutil.NoSuchProcess:
        return
    except psutil.AccessDenied:
        log.warning("no permission to terminate pid %s", pid)
        return
    if await _wait_gone(pid, timeout):
        return
    try:
        proc.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    await _wait_gone(pid, 5.0)


async def _wait_gone(pid: int, timeout: float) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if not psutil.pid_exists(pid):
            return True
        await asyncio.sleep(0.1)
    return not psutil.pid_exists(pid)
