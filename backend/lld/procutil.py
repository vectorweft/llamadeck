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
from pathlib import Path
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


# --- what may be spawned as "the llama-server binary" ------------------------
#
# `to_argv` forces every preset's program to the configured `llama_bin` and
# drops whatever the raw command box names (argv.py:command_argv), and spawning
# goes through create_subprocess_exec — no shell. That makes `llama_bin` the one
# lever that decides *what* runs; `argv_override` only decides the flags. So the
# published exploit chain — PUT /api/settings sets llama_bin=/bin/sh, then a
# preset's raw command becomes `-c '<anything>'` — hinges entirely on this one
# string, and checking it is the cheapest place to break the chain.
#
# Be honest about what this is: a speed bump, not a boundary. LlamaDeck runs as
# the user and exists to launch processes, so anything that can reach the API
# can also edit settings.json directly. What it buys is that the documented
# chain fails closed, at the sink as well as the door, and that a shell in this
# field is a named error instead of a baffling one.
#
# What it deliberately does NOT do is run the candidate to see what it is.
# `--version` probing is self-defeating as a security check: the probe executes
# the attacker's chosen binary, which is the thing being prevented. Every check
# below is metadata only.

#: Programs whose entire job is to turn arguments into more code. A wrapper
#: script is not in here and never will be — that is a path like
#: ~/bin/llama-wrapper.sh with a shebang, which execs perfectly well.
_INTERPRETERS = frozenset({
    "sh", "bash", "zsh", "dash", "ash", "ksh", "csh", "tcsh", "fish", "busybox",
    "env", "perl", "ruby", "node", "php", "lua", "awk", "gawk", "tclsh",
    "xargs", "sudo", "doas", "nohup", "setsid", "stdbuf", "timeout", "strace",
})


def _program_stem(p: Path) -> str:
    name = p.name.lower()
    return name[:-4] if name.endswith(".exe") else name


def _is_interpreter(stem: str) -> bool:
    # python, python3, python3.13 — all one thing.
    return stem in _INTERPRETERS or stem.startswith("python")


def llama_bin_rejection(path: str | None) -> str | None:
    """Why `path` must not be used as the llama-server binary, or None.

    An empty value and a path that simply is not there both return None: a
    fresh install has no binary yet and the setup wizard has to be reachable
    without one. This rejects what is present and wrong, plus interpreters
    whether or not they exist.
    """
    if not path or not path.strip():
        return None
    p = Path(os.path.expanduser(path.strip()))

    # Follow the link before judging the name: /usr/local/bin/llama-server can
    # be a symlink to /bin/bash, and only the target says so. resolve() touches
    # no content and spawns nothing.
    try:
        resolved = p.resolve()
    except (OSError, RuntimeError):
        resolved = p
    if _is_interpreter(_program_stem(p)) or _is_interpreter(_program_stem(resolved)):
        return (
            f"{p} is a shell or interpreter, not llama-server. Running it would "
            "turn a preset's arguments into arbitrary commands, so LlamaDeck "
            "refuses it. Point this at the llama-server executable "
            "(…/build/bin/llama-server), or at your own wrapper script."
        )

    if p.is_dir():
        return f"{p} is a directory, not the llama-server binary"
    if p.exists() and not p.is_file():
        return f"{p} is not a regular file"
    if p.is_file() and not os.access(p, os.X_OK):
        return f"{p} is not executable — chmod +x it, or point at the real binary"
    return None
