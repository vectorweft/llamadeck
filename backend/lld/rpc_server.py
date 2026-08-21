"""Lifecycle for `ggml-rpc-server` processes.

An RPC server exports one machine's GPUs to a llama-server running elsewhere —
or, on this kind of box, to a llama-server in the *same* machine that was built
with a different backend. That second case is the common one: ggml-hip compiles
the ggml-cuda sources with hipcc, so CUDA and HIP cannot coexist in one binary,
and RPC is the only way to drive an NVIDIA and an AMD card natively at once.

Kept deliberately small. These processes hold no state worth recovering: if one
dies, the devices it exported disappear from the next probe and any preset
pinned to them fails to start with a clear message, which is the right outcome.
"""
from __future__ import annotations

import re

import asyncio
import contextlib
import logging
import socket
from dataclasses import dataclass, field

from .settings import RpcServerConfig, _default_rpc_bin, load_settings

log = logging.getLogger("lld.rpc")

# Keep the tail of stderr so a failed start can say *why* rather than just
# "exited". Small on purpose — these servers are not chatty.
_LOG_LINES = 60


def endpoint_of(cfg: RpcServerConfig) -> str:
    return f"{cfg.host}:{cfg.port}"


def _port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    with contextlib.suppress(OSError):
        with socket.create_connection((host, port), timeout=timeout):
            return True
    return False


@dataclass
class RpcProcess:
    cfg: RpcServerConfig
    proc: asyncio.subprocess.Process | None = None
    log_ring: list[str] = field(default_factory=list)
    last_error: str | None = None

    @property
    def endpoint(self) -> str:
        return endpoint_of(self.cfg)

    def is_running(self) -> bool:
        """Ours and alive, or someone else's already on the port.

        The port check matters: a server started by hand outside LlamaDeck is
        just as usable as one we spawned, and refusing to see it would make the
        device list lie.
        """
        if self.proc is not None and self.proc.returncode is None:
            return True
        return _port_open(self.cfg.host, self.cfg.port)

    def owned(self) -> bool:
        """Whether *we* spawned the process currently serving this endpoint."""
        return self.proc is not None and self.proc.returncode is None

    def resolve_binary(self) -> str:
        return self.cfg.binary or _default_rpc_bin()

    def argv(self) -> list[str]:
        argv = [self.resolve_binary(), "-H", self.cfg.host, "-p", str(self.cfg.port)]
        devices = [d for d in (self.cfg.devices or []) if d]
        if devices:
            argv += ["-d", ",".join(devices)]
        return argv

    async def start(self) -> None:
        if self.is_running():
            return
        binary = self.resolve_binary()
        if not binary:
            raise RpcServerError(
                f"{self.cfg.name}: no ggml-rpc-server binary found — build one "
                f"with -DGGML_RPC=ON and set its path in Settings."
            )
        argv = self.argv()
        log.info("[rpc:%s] starting: %s", self.cfg.name, " ".join(argv))
        self.log_ring.clear()
        self.last_error = None
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except (OSError, PermissionError) as e:
            self.proc = None
            raise RpcServerError(f"{self.cfg.name}: cannot exec {binary}: {e}") from e
        asyncio.create_task(self._drain())
        # The device list is only correct once the port answers, so wait for it
        # rather than returning into a race the caller cannot see.
        for _ in range(100):
            if self.proc.returncode is not None:
                tail = "\n".join(self.log_ring[-6:]) or "(no output)"
                self.last_error = tail
                raise RpcServerError(
                    f"{self.cfg.name}: exited immediately (code "
                    f"{self.proc.returncode}):\n{tail}"
                )
            if _port_open(self.cfg.host, self.cfg.port):
                return
            await asyncio.sleep(0.1)
        raise RpcServerError(
            f"{self.cfg.name}: did not open {self.endpoint} within 10s"
        )

    async def _drain(self) -> None:
        if self.proc is None or self.proc.stdout is None:
            return
        with contextlib.suppress(Exception):
            async for raw in self.proc.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip()
                self.log_ring.append(line)
                del self.log_ring[:-_LOG_LINES]

    async def stop(self) -> None:
        """Only ever stops a process we started — an externally launched server
        on the same port is not ours to kill."""
        if not self.owned():
            self.proc = None
            return
        assert self.proc is not None
        self.proc.terminate()
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            self.proc.kill()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.proc.wait(), timeout=5.0)
        self.proc = None

    def to_dict(self) -> dict:
        return {
            "name": self.cfg.name,
            "endpoint": self.endpoint,
            "binary": self.resolve_binary(),
            "devices": list(self.cfg.devices or []),
            "autostart": self.cfg.autostart,
            "running": self.is_running(),
            "owned": self.owned(),
            "last_error": self.last_error,
            "log_tail": self.log_ring[-10:],
        }


class RpcServerError(RuntimeError):
    pass


class RpcServerManager:
    """All configured RPC servers, keyed by name."""

    def __init__(self) -> None:
        self._procs: dict[str, RpcProcess] = {}

    def _sync_with_settings(self) -> None:
        """Rebuild the table from settings, keeping live processes.

        Settings are edited while LlamaDeck runs, so the manager reads them on
        every call rather than caching a snapshot that would go stale.
        """
        configured = {c.name: c for c in (load_settings().rpc_servers or [])}
        for name, cfg in configured.items():
            existing = self._procs.get(name)
            if existing is None:
                self._procs[name] = RpcProcess(cfg=cfg)
            else:
                existing.cfg = cfg
        for name in list(self._procs):
            if name not in configured and not self._procs[name].owned():
                del self._procs[name]

    def all(self) -> list[RpcProcess]:
        self._sync_with_settings()
        return list(self._procs.values())

    def get(self, name: str) -> RpcProcess:
        self._sync_with_settings()
        proc = self._procs.get(name)
        if proc is None:
            raise RpcServerError(f"no RPC server named '{name}'")
        return proc

    def running_endpoints(self) -> list[str]:
        """Endpoints to hand llama-server as `--rpc`, in configured order."""
        return [p.endpoint for p in self.all() if p.is_running()]

    async def start_autostart(self) -> None:
        """Bring up the servers marked autostart. Failures are logged, never
        fatal: LlamaDeck must still come up so the user can fix the setting."""
        for proc in self.all():
            if not proc.cfg.autostart or proc.is_running():
                continue
            try:
                await proc.start()
            except RpcServerError as e:
                log.warning("rpc autostart failed: %s", e)

    async def stop_all(self) -> None:
        for proc in self.all():
            with contextlib.suppress(Exception):
                await proc.stop()


def needs_rpc(device_ids: list[str] | None) -> bool:
    """Whether this device selection can only resolve with `--rpc`."""
    return any(str(d).upper().startswith("RPC") for d in (device_ids or []))


_RPC_ID_RE = re.compile(r"\bRPC\d", re.I)


def needs_rpc_for(cfg) -> bool:
    """Whether this preset needs `--rpc` at all — devices OR raw flags.

    A remote device can be named two ways: in the preset's `devices` pin, or
    inside a hand-written `-ot exps=RPC0` in extra_flags. Only the first was
    checked, so the tensor-override spelling emitted no `--rpc`, `RPC0`
    resolved to nothing, and llama.cpp quietly placed the tensors on a local
    device instead — the exact silent misplacement `-dev` validation exists to
    stop.
    """
    if needs_rpc(getattr(cfg, "devices", None)):
        return True
    return any(_RPC_ID_RE.search(str(t)) for t in (getattr(cfg, "extra_flags", None) or []))


def rpc_flag_value() -> str:
    """The `--rpc` value to launch with: every running endpoint, in configured
    order.

    Order is load-bearing. llama-server numbers RPC devices by the position of
    their endpoint in this list, so `RPC0` means the first one here. The device
    probe passes the same list from the same source, which is what keeps the id
    the user picked pointing at the machine they picked.
    """
    return ",".join(get_rpc_manager().running_endpoints())


_manager: RpcServerManager | None = None


def get_rpc_manager() -> RpcServerManager:
    global _manager
    if _manager is None:
        _manager = RpcServerManager()
    return _manager
