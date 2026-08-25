"""Multi-preset supervisor: one llama-server process per preset, each on its
own port. At any time, zero or more may be active (VRAM permitting). Switching
= stop one, start another. Preserves adoption of externally-started processes.
"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

import psutil

from .argv import cuda_hidden_env, from_argv, mmproj_backend_env, to_argv
from .devices import probe_devices, unknown_device_ids
from .flag_catalog import flags_missing_values, get_flag_catalog
from .rpc_server import RpcServerError, get_rpc_manager, needs_rpc_for
from .exit_diagnosis import diagnose_exit
from .presets import PresetError, PresetRegistry
from .procutil import terminate_pid
from .router_ini import router_env
from .settings import LOGS_DIR, LlamaServerConfig
from .vram_estimate import estimate_vram

log = logging.getLogger(__name__)

LOG_RING_SIZE = 10000
# How much of a line that blew past StreamReader's 64 KiB limit we keep. Enough
# to identify what it was; the rest of a dump that size is noise in a log view.
OVERLONG_LINE_KEEP = 4096


class SupervisorError(RuntimeError):
    pass


def _pid_alive(pid: int) -> bool:
    """True only for a process that can still do work.

    `psutil.pid_exists()` says yes for a ZOMBIE — a process that has already
    exited and is merely waiting to be reaped. Treating that as "running" is
    how a GPU lockup left a dead llama-server listed as healthy while its port
    refused every connection.
    """
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True  # cannot inspect it; do not declare it dead on a guess


class ProcessHandle:
    """Lifecycle of a single llama-server process, keyed by preset name."""

    def __init__(self, preset_name: str, cfg: LlamaServerConfig, binary: str):
        self.preset_name = preset_name
        self.cfg = cfg
        self.binary = binary
        self._proc: asyncio.subprocess.Process | None = None
        self._pid: int | None = None
        self._started_at: float | None = None
        self._adopted = False
        self._log_ring: deque[str] = deque(maxlen=LOG_RING_SIZE)
        self._log_file: Path | None = None
        self._log_fh = None
        self._reader_tasks: list[asyncio.Task] = []
        self._subscribers: set[asyncio.Queue[str]] = set()
        self._last_error: str | None = None
        # Auto-restart bookkeeping (added 2026-05-16 after a CUDA-timeout
        # crash left router-8085 zombie for 3.5h with no recovery).
        self._autorestart_attempts_window: list[float] = []
        self._autorestart_task: asyncio.Task | None = None

    # --- public state ---

    def is_running(self) -> bool:
        if self._pid is None:
            return False
        if self._adopted:
            return _pid_alive(self._pid)
        if self._proc and self._proc.returncode is None:
            return _pid_alive(self._pid)
        return False

    def parented_by_us(self) -> bool:
        """Whether this process is our own child — including one we re-adopted
        after `/api/system/restart` re-exec'd the backend. An externally
        started llama-server is somebody else's to restart; one we spawned is
        ours no matter how the handle got rebuilt."""
        if self._pid is None:
            return False
        if not self._adopted:
            return True
        try:
            return psutil.Process(self._pid).ppid() == os.getpid()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def note_death_if_gone(self) -> bool:
        """Detect a process that died without `_watch_exit` seeing it.

        `_watch_exit` only exists for a child spawned by THIS backend instance.
        Everything else — every preset, after the backend re-execs itself — had
        no crash detection at all: on 2026-08-20 the 5090 locked up (NVRM Xid
        8), llama-server turned into a zombie, `pid_exists` still said yes, and
        the preset sat there reporting "running" and serving nothing.

        Returns True when this call is what noticed the death.
        """
        if self._pid is None or _pid_alive(self._pid):
            return False
        if self._proc is not None:
            # `_watch_exit` owns this one: it is awaiting the same process and
            # will diagnose the exit and schedule the restart itself. Acting
            # here too spent two of the three restarts the 5-minute window
            # allows, so a single crash loop hit the cap in half the time.
            return False
        pid, parented = self._pid, self.parented_by_us()
        self._reap()
        log.error("[%s] process %s died unnoticed (no exit watcher)", self.preset_name, pid)
        msg = "[LlamaDeck] process disappeared (crash detected by liveness poll)"
        self._log_ring.append(msg)
        if not self._last_error:
            self._last_error = diagnose_exit(None, list(self._log_ring))
        self._proc = None
        self._pid = None
        self._started_at = None
        if parented:
            self._schedule_autorestart()
        return True

    def _reap(self) -> None:
        """Clear the zombie if it is ours to clear. Harmless otherwise."""
        if self._pid is None:
            return
        try:
            os.waitpid(self._pid, os.WNOHANG)
        except (ChildProcessError, OSError):
            pass

    def status(self) -> dict:
        running = self.is_running()
        rss_mb: float | None = None
        cpu_pct: float | None = None
        if running and self._pid is not None:
            try:
                p = psutil.Process(self._pid)
                rss_mb = p.memory_info().rss / 1024 / 1024
                cpu_pct = p.cpu_percent(interval=None)
            except psutil.NoSuchProcess:
                running = False

        uptime = time.time() - self._started_at if (running and self._started_at) else None

        est = estimate_vram(self.cfg)
        return {
            "name": self.preset_name,
            "running": running,
            "adopted": self._adopted and running,
            "pid": self._pid if running else None,
            "port": self.cfg.port,
            "started_at": self._started_at if running else None,
            # Router mode only: llama-server reads --models-preset once, at
            # startup, and there is no reload endpoint. Editing a preset
            # rewrites the INI but the running router keeps serving its
            # original copy, so a device pin (or any other change) silently
            # does nothing until a restart. Surface that rather than let the
            # user watch a model land on the wrong GPU.
            "config_stale": self._ini_is_newer_than_start() if running else False,
            "uptime_seconds": uptime,
            "rss_mb": round(rss_mb, 1) if rss_mb is not None else None,
            "cpu_percent": round(cpu_pct, 1) if cpu_pct is not None else None,
            "config": asdict(self.cfg),
            "log_file": str(self._log_file) if self._log_file else None,
            "returncode": self._proc.returncode if self._proc else None,
            "last_error": self._last_error,
            "vram_estimate": est.to_dict() if est else None,
        }

    # --- lifecycle (caller holds outer lock) ---

    def _missing_paths(self) -> str | None:
        """Why this preset cannot possibly start, checked before spawning.

        llama-server exits 1 on a path that isn't there, which the supervisor
        then treats as a crash worth retrying — three restarts, three identical
        failures, and a "manual restart needed" verdict for something no restart
        fixes. These are cheap to check and name the field to fix.
        """
        if self.cfg.mode == "router":
            md = self.cfg.models_dir
            if md and not Path(md).is_dir():
                return (
                    f"models_dir does not exist: {md} — point the preset's "
                    "models directory at your GGUF root (Presets → this preset), "
                    "or fix it in Settings → models root"
                )
        elif self.cfg.model_path and not Path(self.cfg.model_path).exists():
            return (
                f"model file not found: {self.cfg.model_path} — it was moved, "
                "renamed, or lives on a drive that is not mounted"
            )
        for label, p in (("mmproj", self.cfg.mmproj_path),
                         ("draft model", self.cfg.model_path_draft)):
            if p and not Path(p).exists():
                return f"{label} file not found: {p}"
        return None

    def _ini_is_newer_than_start(self) -> bool:
        if getattr(self.cfg, "mode", "single") != "router":
            return False
        path = getattr(self.cfg, "models_preset_path", None)
        if not path or self._started_at is None:
            return False
        try:
            return Path(path).stat().st_mtime > self._started_at
        except OSError:
            return False

    async def _ensure_rpc(self) -> None:
        """Bring up the RPC servers a pinned preset depends on.

        Must run before `_check_devices`: an RPC id is invisible to
        `--list-devices` until its endpoint is listening, so validating first
        would reject a perfectly good preset whose server simply is not up yet.
        A failure here is raised rather than swallowed — starting anyway would
        drop the model on whatever device llama.cpp picks instead, which is the
        silent misplacement this whole feature exists to prevent.
        """
        if not needs_rpc_for(self.cfg):
            return
        mgr = get_rpc_manager()
        for proc in mgr.all():
            if proc.is_running():
                continue
            try:
                await proc.start()
            except RpcServerError as e:
                raise SupervisorError(f"{self.preset_name}: {e}") from e

    async def _check_devices(self) -> None:
        """Reject a device selection this binary cannot honour.

        `-dev` names are backend-specific, so a preset pinned to Vulkan1 is
        meaningless to a CUDA-only build. llama-server would fail on its own,
        but only after the log file is open and the supervisor has counted a
        start attempt — and the message would not say which ids *are* valid.
        Checking here turns it into one actionable error.

        A binary that cannot be queried at all yields no devices; that is not
        evidence the selection is wrong, so it is allowed through.
        """
        unknown = await unknown_device_ids(self.binary, getattr(self.cfg, "devices", None) or [])
        if unknown:
            devices = await probe_devices(self.binary)
            raise SupervisorError(
                f"{self.preset_name}: device(s) {', '.join(unknown)} not available "
                f"in this llama-server build — it exposes "
                f"{', '.join(d.id for d in devices)}. Rebuild with the matching "
                f"backend, or change the preset's GPU selection."
            )

    async def _check_flag_values(self, argv: list[str]) -> None:
        """Reject a flag that needs a value and was given none.

        llama-server refuses the whole command line for this and exits before
        it has written anything a user would recognise as a reason:

            error while handling argument "--tools": expected value for argument

        Reported here instead, so a flag typed into extra_flags fails with the
        flag's own name and its usage rather than as a dead preset. The
        catalog knows this without running anything, and says nothing at all
        when the binary could not be queried.
        """
        catalog = await get_flag_catalog(self.binary)
        missing = flags_missing_values(catalog, argv)
        if not missing:
            return
        parts = ", ".join(
            f"{m['flag']} (expects {m['placeholder']})" if m["placeholder"] else m["flag"]
            for m in missing
        )
        raise SupervisorError(
            f"{self.preset_name}: {parts} — llama-server needs a value for "
            f"these and refuses the whole command line without one. Add the "
            f"value in extra_flags, or remove the flag."
        )

    async def start(self) -> None:
        if self.is_running():
            raise SupervisorError(f"{self.preset_name}: already running")
        if _port_in_use(self.cfg.host, self.cfg.port):
            raise SupervisorError(port_conflict_message(self.preset_name, self.cfg.port))
        missing = self._missing_paths()
        if missing:
            raise SupervisorError(f"{self.preset_name}: {missing}")
        await self._ensure_rpc()
        await self._check_devices()

        argv = to_argv(self.cfg, self.binary)
        await self._check_flag_values(argv)
        spawn_env = self._spawn_env()
        env_note = "".join(
            f"{k}={v} " for k, v in sorted((getattr(self.cfg, "env", None) or {}).items())
        )
        log.info("[%s] starting: %s%s", self.preset_name, env_note, " ".join(argv))

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._log_file = LOGS_DIR / f"{self.preset_name}-{stamp}.log"
        self._log_fh = open(self._log_file, "w", buffering=1)
        self._log_ring.clear()
        self._last_error = None

        try:
            self._proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=spawn_env,
            )
        except FileNotFoundError as e:
            self._log_fh.close()
            self._log_fh = None
            self._last_error = (
                f"binary not found: {self.binary} — set the llama-server path "
                "in Settings → llama.cpp paths"
            )
            raise SupervisorError(self._last_error) from e

        self._pid = self._proc.pid
        self._started_at = time.time()
        self._adopted = False

        self._reader_tasks = [
            asyncio.create_task(self._consume_stream(self._proc.stdout), name=f"{self.preset_name}-stdout"),
            asyncio.create_task(self._watch_exit(), name=f"{self.preset_name}-watch"),
        ]

    def _spawn_env(self) -> dict[str, str] | None:
        """The child's environment: ours, with the preset's overrides on top.

        None when the preset sets nothing — the overwhelmingly common case then
        execs with the inherited environment exactly as it did before this
        existed. Some llama.cpp behaviour has no flag at all
        (`GGML_CUDA_DISABLE_GRAPHS`), so this is the only way to reach it
        without wrapping the binary in a shell script.

        A router carries more than its own: the models it serves are loaded
        inside its process and inherit this environment, so a served preset's
        variables have nowhere else to land. router_env() merges them here.
        """
        overrides = {str(k): str(v) for k, v in (getattr(self.cfg, "env", None) or {}).items()}
        if getattr(self.cfg, "mode", "single") == "router" and self.cfg.models_dir:
            merged, warnings = router_env(
                self.cfg.models_dir, router_preset=self.cfg
            )
            for w in warnings:
                log.warning("[%s] %s", self.preset_name, w)
            overrides = merged
        else:
            # The mmproj (vision encoder) ignores -dev: llama.cpp pins it via
            # MTMD_BACKEND_DEVICE or drops it on the first GPU backend (CUDA0
            # on a CUDA+Vulkan build). Inject it so a single-device preset
            # keeps every part of the model on that device. setdefault → the
            # preset's own explicit env wins.
            for k, v in mmproj_backend_env(self.cfg).items():
                overrides.setdefault(k, v)
            # A preset pinned to a non-CUDA GPU still pays for a CUDA context
            # (~500 MiB on the other card) the moment llama.cpp weighs whether
            # the model fits. Hide the CUDA devices from it. Same setdefault
            # rule: an explicit CUDA_VISIBLE_DEVICES in the preset wins.
            for k, v in cuda_hidden_env(self.cfg).items():
                overrides.setdefault(k, v)
        if not overrides:
            return None
        return {**os.environ, **overrides}

    async def stop(self, timeout: float = 10.0) -> None:
        if self._adopted and self._pid is not None:
            await terminate_pid(self._pid, timeout=timeout)
            self._reset()
            return

        if not self._proc or self._proc.returncode is not None:
            self._reset()
            return

        try:
            self._proc.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass
            await self._proc.wait()

        await self._cancel_readers()
        self._reset()

    def adopt(self, pid: int, started_at: float) -> None:
        """Attach to an externally-started process. Caller has validated that
        cmdline matches this preset."""
        if self.is_running():
            raise SupervisorError(f"{self.preset_name}: already running")
        self._proc = None
        self._pid = pid
        self._started_at = started_at
        self._adopted = True
        self._log_ring.clear()
        self._log_ring.append(
            f"[LlamaDeck] Adopted externally-started process (PID {pid}) — stdout unavailable for earlier output."
        )

    def release(self) -> None:
        """Stop tracking without killing."""
        if not self._adopted:
            raise SupervisorError(f"{self.preset_name}: not adopted")
        self._reset()

    # --- logs ---

    def log_tail(self, n: int = 500) -> list[str]:
        return list(self._log_ring)[-n:]

    def subscribe(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=1000)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        self._subscribers.discard(q)

    async def stream_logs(self) -> AsyncIterator[str]:
        for line in list(self._log_ring)[-200:]:
            yield line
        q = self.subscribe()
        try:
            while True:
                yield await q.get()
        finally:
            self.unsubscribe(q)

    # --- internal ---

    async def _consume_stream(self, stream: asyncio.StreamReader | None) -> None:
        if stream is None:
            return
        try:
            while True:
                raw = await self._read_line(stream)
                if raw is None:
                    return
                line = raw.decode("utf-8", errors="replace").rstrip()
                self._log_ring.append(line)
                if self._log_fh:
                    try:
                        self._log_fh.write(line + "\n")
                    except OSError as e:
                        log.warning("[%s] log write failed: %s", self.preset_name, e)
                for q in list(self._subscribers):
                    try:
                        q.put_nowait(line)
                    except asyncio.QueueFull:
                        pass
        except Exception as e:
            log.exception("[%s] log consumer failed: %s", self.preset_name, e)

    async def _read_line(self, stream: asyncio.StreamReader) -> bytes | None:
        """One line, tolerating lines longer than StreamReader's 64 KiB limit.

        Not readline(): past the limit that raises ValueError *and discards the
        line*, so the reader task dies and every later line is lost with it —
        the log file stops mid-run while the server keeps serving, and the exit
        diagnosis ends up quoting whatever came before the reader went blind.
        Nothing caps how long a line llama-server may print (a verbose prompt
        echo, a template or tokenizer dump), so the input is unbounded even
        though no log on this machine has yet crossed the limit.

        readuntil() leaves the buffer untouched when it overruns, which lets us
        keep the head of the line, drop only the excess, and stay in sync for
        the next one. Returns None at EOF.
        """
        head = b""
        dropped = False
        while True:
            try:
                chunk = await stream.readuntil(b"\n")
            except asyncio.LimitOverrunError as e:
                # `consumed` is how far it scanned without completing a line.
                # Taking exactly that leaves the newline in place, so the line
                # after this one still arrives intact.
                try:
                    chunk = await stream.readexactly(max(e.consumed, 1))
                except asyncio.IncompleteReadError as ie:
                    chunk = ie.partial
            except asyncio.IncompleteReadError as e:
                chunk = e.partial  # EOF with no trailing newline
            if not chunk:
                return head or None
            room = OVERLONG_LINE_KEEP - len(head)
            if len(chunk) > room:
                dropped = True
            if room > 0:
                head += chunk[:room]
            if chunk.endswith(b"\n"):
                return head + (b" ...[line truncated]" if dropped else b"")

    async def _watch_exit(self) -> None:
        if not self._proc:
            return
        rc = await self._proc.wait()
        log.info("[%s] exited with code %s", self.preset_name, rc)
        msg = f"[LlamaDeck] process exited (code={rc})"
        self._log_ring.append(msg)
        for q in list(self._subscribers):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass
        if rc != 0:
            # The reason is in the server's own output, a few lines up. Report
            # that instead of the exit code — the code alone tells nobody
            # whether the model didn't fit, a part was missing, or the build is
            # too old.
            self._last_error = diagnose_exit(rc, list(self._log_ring))
        if self._log_fh:
            try:
                self._log_fh.flush()
            except OSError as e:
                log.warning("[%s] log flush failed: %s", self.preset_name, e)
        # Auto-restart on unexpected exit. Without this, a CUDA-timeout
        # crash leaves the supervisor in a "dead but tracked" state and
        # nobody re-launches the service. Cooldown + window cap prevent
        # crash loops.
        if rc is not None and rc != 0 and not self._adopted:
            self._schedule_autorestart()

    def _schedule_autorestart(self) -> None:
        # Window-based cap: at most 3 restarts per 5 minutes.
        now = time.time()
        self._autorestart_attempts_window = [
            t for t in self._autorestart_attempts_window if now - t < 300
        ]
        if len(self._autorestart_attempts_window) >= 3:
            log.error(
                "[%s] auto-restart suppressed: %d crashes in last 5min — operator intervention required",
                self.preset_name, len(self._autorestart_attempts_window),
            )
            # Keep the diagnosis from _watch_exit. Replacing it with the loop
            # count told the user to restart manually and hid the one thing
            # that mattered — a config error restarting can never fix.
            loop_note = (
                f"crash-looped {len(self._autorestart_attempts_window)}× in 5min, "
                "auto-restart stopped"
            )
            if not self._last_error:
                self._last_error = loop_note
            elif loop_note not in self._last_error:
                self._last_error = f"{self._last_error} ({loop_note})"
            return
        if self._autorestart_task and not self._autorestart_task.done():
            return  # already pending
        self._autorestart_attempts_window.append(now)
        self._autorestart_task = asyncio.create_task(
            self._autorestart_after_cooldown(),
            name=f"{self.preset_name}-autorestart",
        )

    async def _autorestart_after_cooldown(self, cooldown_s: float = 30.0) -> None:
        try:
            await asyncio.sleep(cooldown_s)
            if self.is_running():
                return  # someone else already started it
            # Reset transient state so start() preflight checks pass cleanly.
            self._proc = None
            self._pid = None
            self._started_at = None
            if self._log_fh:
                try:
                    self._log_fh.close()
                except OSError:
                    pass
                self._log_fh = None
            log.info("[%s] auto-restart attempt after unexpected exit", self.preset_name)
            await self.start()
            log.info("[%s] auto-restart succeeded", self.preset_name)
        except Exception as e:  # noqa: BLE001
            log.error("[%s] auto-restart failed: %s", self.preset_name, e)
            self._last_error = f"auto-restart failed: {e}"

    async def _cancel_readers(self) -> None:
        for t in self._reader_tasks:
            if not t.done():
                t.cancel()
        for t in self._reader_tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._reader_tasks = []

    def _reset(self) -> None:
        self._proc = None
        self._pid = None
        self._started_at = None
        self._adopted = False
        if self._log_fh:
            try:
                self._log_fh.close()
            except OSError as e:
                log.warning("[%s] log close failed: %s", self.preset_name, e)
            self._log_fh = None


async def liveness_loop(get_supervisor_fn, interval_s: float = 10.0) -> None:
    """Watch for processes that die without an exit watcher.

    Separate from `health_watchdog.watchdog_loop`, which probes the router over
    HTTP once a minute and needs that slow cadence for its failure threshold.
    This one only reads /proc, so it can tick often enough that a crash is
    noticed in seconds instead of a minute.
    """
    log.info("liveness poll: started (interval=%.0fs)", interval_s)
    while True:
        try:
            await asyncio.sleep(interval_s)
            for dead in get_supervisor_fn().poll_liveness():
                log.error("liveness poll: %s died with no exit watcher", dead)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — never let the loop die
            log.exception("liveness poll tick failed")


class MultiSupervisor:
    def __init__(self, binary: str):
        self.binary = binary
        self._handles: dict[str, ProcessHandle] = {}
        self._lock = asyncio.Lock()

    def poll_liveness(self) -> list[str]:
        """Notice processes that died without an exit watcher. Returns the
        preset names whose death this call detected."""
        died: list[str] = []
        for name, h in list(self._handles.items()):
            try:
                if h.note_death_if_gone():
                    died.append(name)
            except Exception:  # noqa: BLE001 — a poller must never take the loop down
                log.exception("liveness poll failed for %s", name)
        return died

    def rebind(self, binary: str) -> None:
        """Point at a different llama-server without restarting the backend.
        Idle handles are dropped so they get rebuilt with the new binary;
        running ones keep theirs — the process is already exec'd, and its
        handle must stay alive to supervise it."""
        if binary == self.binary:
            return
        self.binary = binary
        for name, h in list(self._handles.items()):
            if not h.is_running():
                del self._handles[name]

    def _handle(self, preset_name: str) -> ProcessHandle:
        cfg = PresetRegistry().get(preset_name)
        h = self._handles.get(preset_name)
        if h is None:
            h = ProcessHandle(preset_name, cfg, self.binary)
            self._handles[preset_name] = h
        elif not h.is_running():
            h.cfg = cfg  # pick up preset edits while idle
        return h

    async def start(self, preset_name: str) -> dict:
        async with self._lock:
            h = self._handle(preset_name)
            await h.start()
            if h.cfg.model_path:
                from .models import mark_used  # lazy: avoids import cycle at module load
                try:
                    await mark_used(h.cfg.model_path)
                except Exception:
                    log.debug("mark_used failed for %s", h.cfg.model_path, exc_info=True)
            return h.status()

    async def stop(self, preset_name: str) -> dict:
        async with self._lock:
            if preset_name not in self._handles:
                raise SupervisorError(f"{preset_name}: not tracked")
            await self._handles[preset_name].stop()
            return self._handles[preset_name].status()

    async def restart(self, preset_name: str) -> dict:
        async with self._lock:
            h = self._handle(preset_name)
            if h.is_running():
                await h.stop()
            # Refresh cfg from registry after stop so the next start picks up
            # any edits made while the process was running. Without this, the
            # handle's cfg is frozen at the value it had when start() was first
            # called, and PUT /api/presets/<name> wouldn't take effect on
            # restart.
            h.cfg = PresetRegistry().get(preset_name)
            await h.start()
            return h.status()

    async def switch(self, from_name: str, to_name: str) -> dict:
        async with self._lock:
            out: dict[str, dict] = {}
            if from_name in self._handles and self._handles[from_name].is_running():
                await self._handles[from_name].stop()
                out[from_name] = self._handles[from_name].status()
            h = self._handle(to_name)
            await h.start()
            out[to_name] = h.status()
            return {"switched": out}

    def statuses(self, *, vram_estimates: bool = True) -> dict[str, dict]:
        """Return a status row for every preset, active or not.

        `vram_estimates=False` leaves `vram_estimate` None on the idle rows.
        Producing one means reading each preset's GGUF header, which is by far
        the most expensive thing in here — callers that only want to know what
        is running (the metrics poller at 2 Hz, the watchdog, boot) should not
        pay for it.
        """
        out: dict[str, dict] = {}
        all_presets = {p.name: p for p in PresetRegistry().list()}
        # Active / tracked
        for name, h in self._handles.items():
            out[name] = h.status()
        # Idle presets (never started or released)
        for name, cfg in all_presets.items():
            if name not in out:
                est = estimate_vram(cfg) if vram_estimates else None
                out[name] = {
                    "name": name,
                    "running": False,
                    "adopted": False,
                    "pid": None,
                    "port": cfg.port,
                    "started_at": None,
                    "uptime_seconds": None,
                    "rss_mb": None,
                    "cpu_percent": None,
                    "config": asdict(cfg),
                    "log_file": None,
                    "returncode": None,
                    "last_error": None,
                    "vram_estimate": est.to_dict() if est else None,
                }
        return out

    def reap_orphan_children(self) -> int:
        """Reap dead children left over from a previous incarnation.

        `/api/system/restart` re-execs the backend in place, so its PID — and
        therefore its parenthood of every llama-server it spawned — survives.
        When one of those dies afterwards, nobody is left to wait() on it and
        it stays a <defunct> entry that `pid_exists()` happily calls alive.
        Called once at startup, before anything new is spawned, so it cannot
        race asyncio's own child watcher.
        """
        reaped = 0
        while True:
            try:
                pid, _status = os.waitpid(-1, os.WNOHANG)
            except (ChildProcessError, OSError):
                break
            if pid == 0:
                break
            log.info("reaped orphaned child pid=%s from a previous instance", pid)
            reaped += 1
        return reaped

    def scan_existing(self) -> list[dict]:
        """Find any llama-server procs on the host, and suggest a matching preset by port.

        Zombies are skipped: a <defunct> llama-server is a corpse, and adopting
        one hands the UI a preset that reports "running" on a port nothing is
        listening to.
        """
        found = []
        presets_by_port = {p.port: p.name for p in PresetRegistry().list()}
        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                cmdline = proc.info["cmdline"] or []
                if not cmdline:
                    continue
                exe = cmdline[0]
                if not (exe.endswith("llama-server") or proc.info["name"] == "llama-server"):
                    continue
                if proc.status() == psutil.STATUS_ZOMBIE:
                    continue
                # Router-mode children are llama-server processes spawned BY a
                # llama-server parent. They are managed by their router — offering
                # them for adoption is confusing and adopting one would fight the
                # router's own lifecycle. Skip anything whose parent is llama-server.
                try:
                    parent = proc.parent()
                    if parent is not None:
                        pcmd = parent.cmdline()
                        if (pcmd and pcmd[0].endswith("llama-server")) or parent.name() == "llama-server":
                            continue
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                cfg = from_argv(cmdline, name=f"adopted-{proc.info['pid']}")
                matching = presets_by_port.get(cfg.port)
                # Skip if already tracked
                if matching and matching in self._handles and self._handles[matching].is_running():
                    continue
                found.append({
                    "pid": proc.info["pid"],
                    "cmdline": cmdline,
                    "started_at": proc.info.get("create_time"),
                    "config": asdict(cfg),
                    "suggested_preset": matching,
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return found

    async def adopt(self, pid: int, preset_name: str | None = None) -> dict:
        async with self._lock:
            try:
                proc = psutil.Process(pid)
                cmdline = proc.cmdline()
            except psutil.NoSuchProcess as e:
                raise SupervisorError(f"pid {pid} does not exist") from e
            except psutil.AccessDenied as e:
                raise SupervisorError(f"access denied reading pid {pid}") from e
            if not cmdline or not cmdline[0].endswith("llama-server"):
                raise SupervisorError(f"pid {pid} is not a llama-server process")

            parsed = from_argv(cmdline, name=preset_name or f"adopted-{pid}")

            if preset_name is None:
                # Match by port
                for p in PresetRegistry().list():
                    if p.port == parsed.port:
                        preset_name = p.name
                        break

            if preset_name is None:
                raise SupervisorError(
                    f"No preset matches port {parsed.port}. Create a preset with that port first or pass preset_name explicitly."
                )

            try:
                cfg = PresetRegistry().get(preset_name)
            except PresetError as e:
                raise SupervisorError(str(e)) from e

            # Sanity: the running process's port must match the preset's port,
            # otherwise the caller is adopting under the wrong identity. This
            # also blocks mis-adoption like "claim router-8085's PID under
            # qwen3.6-27b" — a class of bug we hit when adopt() was called
            # blindly with a hard-coded preset name.
            if parsed.port and cfg.port and parsed.port != cfg.port:
                raise SupervisorError(
                    f"port mismatch: process on :{parsed.port} cannot be adopted "
                    f"as preset '{preset_name}' (preset declares :{cfg.port}). "
                    f"Either fix the preset's port or omit `preset` to auto-match."
                )

            h = self._handles.get(preset_name)
            if h is None:
                h = ProcessHandle(preset_name, cfg, self.binary)
                self._handles[preset_name] = h
            if h.is_running():
                raise SupervisorError(f"{preset_name}: already running — stop first")
            # Show the running argv values (ctx, ngl, etc.) but preserve preset-only
            # metadata (name, notes, estimated_vram_mb, mode + router fields)
            # from the registered preset. from_argv doesn't recognise router
            # flags, so without this overlay an adopted router process would
            # show up as mode="single" and break /api/router/active.
            parsed.name = cfg.name
            parsed.notes = cfg.notes
            parsed.estimated_vram_mb = cfg.estimated_vram_mb
            parsed.mode = cfg.mode
            parsed.models_dir = cfg.models_dir
            parsed.models_max = cfg.models_max
            parsed.models_autoload = cfg.models_autoload
            parsed.models_preset_path = cfg.models_preset_path
            parsed.sleep_idle_seconds = cfg.sleep_idle_seconds
            h.cfg = parsed
            h.adopt(pid, proc.create_time())
            if parsed.model_path:
                from .models import mark_used  # lazy: avoids import cycle at module load
                try:
                    await mark_used(parsed.model_path)
                except Exception:
                    log.debug("mark_used failed for %s", parsed.model_path, exc_info=True)
            return h.status()

    async def release(self, preset_name: str) -> dict:
        async with self._lock:
            if preset_name not in self._handles:
                raise SupervisorError(f"{preset_name}: not tracked")
            self._handles[preset_name].release()
            return self._handles[preset_name].status()

    def log_tail(self, preset_name: str, n: int = 500) -> list[str]:
        if preset_name not in self._handles:
            return []
        return self._handles[preset_name].log_tail(n)

    async def stream_logs(self, preset_name: str) -> AsyncIterator[str]:
        # Create handle if it doesn't exist (so user can subscribe before starting)
        if preset_name not in self._handles:
            cfg = PresetRegistry().get(preset_name)
            self._handles[preset_name] = ProcessHandle(preset_name, cfg, self.binary)
        async for line in self._handles[preset_name].stream_logs():
            yield line


# Singleton
_instance: MultiSupervisor | None = None


def get_supervisor(llama_bin: str | None = None) -> MultiSupervisor:
    global _instance
    if _instance is None:
        if llama_bin is None:
            raise RuntimeError("supervisor not initialized and no binary path provided")
        _instance = MultiSupervisor(llama_bin)
    return _instance


def port_owner(port: int) -> psutil.Process | None:
    """The process listening on `port`, or None if it cannot be identified.

    psutil needs elevated rights to name the owner of *another user's* socket;
    ours are our own, so this answers in the case that matters. A failure to
    look it up is not an error — the caller falls back to a vaguer message.
    """
    try:
        for c in psutil.net_connections(kind="inet"):
            if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN and c.pid:
                return psutil.Process(c.pid)
    except (psutil.Error, PermissionError, OSError):
        pass
    return None


def port_conflict_message(preset_name: str, port: int, expect_binary: str | None = None) -> str:
    """Say *who* holds the port, not just that somebody does.

    The old message was "port 8085 already in use by something else". On
    2026-08-22 the "something else" was that preset's own llama-server, still
    serving, merely dropped from tracking by a Release click — and the message
    sent the user hunting for a phantom second install instead of at the Adopt
    button two sections up the page.
    """
    proc = port_owner(port)
    if proc is None:
        return (
            f"{preset_name}: port {port} is already in use, and the owning process could not "
            f"be identified. Free the port or give this preset a different one."
        )
    try:
        cmdline = proc.cmdline()
        name = proc.name()
    except (psutil.Error, OSError):
        cmdline, name = [], "?"
    ours = bool(cmdline) and "llama-server" in (cmdline[0] if cmdline else "")
    if ours:
        return (
            f"{preset_name}: port {port} is held by PID {proc.pid}, an llama-server that "
            f"LlamaDeck is not tracking — usually one that was released, or started outside "
            f"the app. Adopt it on the Server page to take it back over, or stop that process "
            f"first. It is still serving requests in the meantime."
        )
    return (
        f"{preset_name}: port {port} is held by PID {proc.pid} ({name}), which is not a "
        f"llama-server. Stop it, or give this preset a different port."
    )


def _port_in_use(host: str, port: int) -> bool:
    check_host = "127.0.0.1" if host == "0.0.0.0" else host
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(0.2)
        try:
            s.connect((check_host, port))
            return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            return False
    finally:
        s.close()
