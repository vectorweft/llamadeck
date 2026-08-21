"""TTS subprocess wrapper for the GPU broker.

When the broker grants the `tts` slot, we either:
  - adopt an already-running TTS server on `managed_url` (someone started
    it by hand or via systemd), or
  - spawn one with `python tts_http_server.py --port <port>` in the
    configured TTS checkout (e.g. /home/user/IndexTTS-2).

Mirrors ComfyService exactly except for the spawn argv and health path
(/health vs /system_stats).

When the slot is released and keepalive expires, we stop only the
process WE spawned. Adopted processes are left untouched.

Health gate: GET <managed_url>/health -> 200.
"""
from __future__ import annotations

import asyncio
import logging
import shlex
import time
from pathlib import Path

import httpx

from ..procutil import new_process_group_kwargs, terminate_tree

log = logging.getLogger("lld.processes.tts")


class TtsService:
    """Concrete Service for the tts slot."""

    name = "tts"

    def __init__(
        self,
        *,
        managed_url: str,
        est_vram_mb: int,
        tts_path: str | None = None,
        python_bin: str | None = None,
        server_script: str = "tts_http_server.py",
        extra_args: str = "",
        log_dir: Path | None = None,
        startup_timeout_s: float = 120.0,
        health_path: str = "/health",
    ):
        self.managed_url = managed_url.rstrip("/")
        self.est_vram_mb = int(est_vram_mb)
        self._tts_path = Path(tts_path).expanduser() if tts_path else None
        self._python_bin = python_bin or "python"
        self._server_script = server_script
        self._extra_args = shlex.split(extra_args) if extra_args else []
        self._log_dir = log_dir
        self._startup_timeout_s = float(startup_timeout_s)
        self._health_url = self.managed_url + health_path

        self._proc: asyncio.subprocess.Process | None = None
        self._adopted = False
        self._log_fh = None

    # -- Service protocol ----------------------------------------------------

    async def start(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            return
        if await self._reachable():
            self._adopted = True
            log.info("tts already running at %s — adopting", self.managed_url)
            return
        if not self._tts_path:
            raise RuntimeError(
                f"TTS not reachable at {self.managed_url} and no "
                f"tts_path configured to spawn one. Start the TTS server "
                f"manually or set settings.tts_path."
            )
        script = self._tts_path / self._server_script
        if not script.is_file():
            raise RuntimeError(
                f"tts_path={self._tts_path} doesn't contain {self._server_script}"
            )

        host, port = _split_url(self.managed_url)
        argv = [
            self._python_bin, str(script),
            "--host", host,
            "--port", str(port),
        ] + self._extra_args
        log.info("spawning TTS: cwd=%s argv=%s", self._tts_path, " ".join(argv))

        log_file: Path | None = None
        if self._log_dir is not None:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            log_file = self._log_dir / f"tts-{stamp}.log"
            self._log_fh = open(log_file, "ab", buffering=0)

        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(self._tts_path),
            stdout=self._log_fh or asyncio.subprocess.DEVNULL,
            stderr=self._log_fh or asyncio.subprocess.DEVNULL,
            **new_process_group_kwargs(),
        )
        self._adopted = False
        log.info("TTS spawned pid=%s log=%s", self._proc.pid, log_file)

    async def stop(self) -> None:
        if self._adopted:
            self._adopted = False
            return
        proc = self._proc
        if proc is None or proc.returncode is not None:
            self._proc = None
            self._close_log()
            return
        await terminate_tree(proc.pid, timeout=15.0, label="TTS")
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except Exception:
            pass
        self._proc = None
        self._close_log()
        log.info("TTS stopped")

    async def wait_ready(self, timeout_s: float | None = None) -> bool:
        deadline = asyncio.get_event_loop().time() + (timeout_s or self._startup_timeout_s)
        while asyncio.get_event_loop().time() < deadline:
            if await self._reachable():
                return True
            if self._proc is not None and self._proc.returncode is not None and not self._adopted:
                log.warning("TTS exited during startup rc=%s", self._proc.returncode)
                return False
            await asyncio.sleep(0.7)
        return False

    def is_running(self) -> bool:
        if self._adopted:
            return True
        proc = self._proc
        return proc is not None and proc.returncode is None

    # -- internals -----------------------------------------------------------

    async def _reachable(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(2.0, connect=1.5)) as c:
                r = await c.get(self._health_url)
                return r.status_code < 500
        except Exception:
            return False

    def _close_log(self) -> None:
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None


def _split_url(url: str) -> tuple[str, int]:
    rest = url.split("://", 1)[-1]
    host, _, port = rest.partition(":")
    if not port:
        return host, 80
    port = port.split("/", 1)[0]
    return host, int(port)


__all__ = ["TtsService"]
