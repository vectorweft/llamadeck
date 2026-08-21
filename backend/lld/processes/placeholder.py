"""Placeholder service — for slots whose real subprocess wrapper is not
yet implemented (ComfyUI, XTTS until faz-1.7). Reports running=False
unless an external service is already up at managed_url.

Behavior:
  - start(): if a service is reachable at managed_url, treat it as adopted.
             Otherwise raise — broker will surface a 503 to the caller so
             clients see a clean "service unavailable" error rather than
             a silent no-op.
  - stop(): no-op (we don't kill processes we didn't start).
  - wait_ready(): reachability check.

Replace this with a real subprocess service in faz-1.7.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Literal

import httpx

log = logging.getLogger("lld.processes.placeholder")


class PlaceholderService:
    def __init__(
        self,
        *,
        slot_name: Literal["comfy", "tts"],
        managed_url: str,
        est_vram_mb: int,
        health_path: str = "/",
        health_method: str = "GET",
    ):
        self.name = slot_name
        self.managed_url = managed_url
        self.est_vram_mb = int(est_vram_mb)
        self._health_url = managed_url.rstrip("/") + health_path
        self._health_method = health_method.upper()
        self._adopted = False

    async def start(self) -> None:
        if await self._reachable():
            self._adopted = True
            log.info("placeholder %s: external service at %s adopted",
                     self.name, self.managed_url)
            return
        raise RuntimeError(
            f"placeholder service '{self.name}' has no real start implementation; "
            f"start the external service at {self.managed_url} manually, or "
            f"upgrade to a concrete subprocess service in faz-1.7."
        )

    async def stop(self) -> None:
        # We didn't spawn it, so don't kill it.
        self._adopted = False

    async def wait_ready(self, timeout_s: float = 60.0) -> bool:
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            if await self._reachable():
                return True
            await asyncio.sleep(0.5)
        return False

    def is_running(self) -> bool:
        return self._adopted

    async def _reachable(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=1.5) as c:
                r = await c.request(self._health_method, self._health_url)
                return r.status_code < 500
        except Exception:
            return False


__all__ = ["PlaceholderService"]
