"""LLM service — broker-facing facade over the supervisor.

Design: the broker doesn't own a fixed preset. It asks the supervisor at
every check "which LLM preset is currently running?" and routes there.
This keeps three things consistent:

  - `default_llm_preset` setting (optional pin)
  - whichever preset the user (or a startup auto-adopt) actually started
  - whichever preset llama-server is bound to right now

A preset is considered the "active LLM" when:
  1. settings.default_llm_preset is set AND that preset is running, OR
  2. any single-mode preset under the supervisor is running (deterministic
     order by preset name)

If nothing is running, `start()` launches the configured default. The
broker tracks ownership: only presets the broker itself launched are
candidates for `stop()` — user-started or adopted presets are never
killed by keepalive expiry. The user can still kill them via the UI's
adopt-toggle (a deliberate manual action).
"""
from __future__ import annotations

import asyncio
import logging

import httpx

from ..presets import PresetRegistry
from ..settings import Settings
from ..supervisor import MultiSupervisor

log = logging.getLogger("lld.processes.llm")


class LlmService:
    name = "llm"

    def __init__(self, settings: Settings, supervisor: MultiSupervisor):
        self._settings = settings
        self._supervisor = supervisor
        # Preset name the broker started itself. Only this preset is eligible
        # for broker-driven stop (keepalive expiry / slot transition). User /
        # adopted presets are externally owned.
        self._auto_managed_preset: str | None = None

    # -- preset resolution ---------------------------------------------------

    def _resolve_active_preset(self) -> str | None:
        """Return the running preset's name, or None if no LLM is running."""
        pin = getattr(self._settings, "default_llm_preset", None)
        if pin and self._supervisor_running(pin):
            return pin
        # Deterministic fallback: alphabetical order over running single-mode
        # handles. With multiple LLMs running concurrently (rare), pinning is
        # the supported way to disambiguate.
        for name in sorted(self._supervisor._handles.keys()):  # type: ignore[attr-defined]
            h = self._supervisor._handles[name]                # type: ignore[attr-defined]
            if not h.is_running():
                continue
            if getattr(h.cfg, "mode", "single") == "single":
                return name
        return None

    def _supervisor_running(self, preset_name: str) -> bool:
        h = self._supervisor._handles.get(preset_name)  # type: ignore[attr-defined]
        return bool(h and h.is_running())

    def _default_preset_name(self) -> str:
        """Preset to launch when nothing is running. Pin > first single-mode
        preset > first preset in registry."""
        pin = getattr(self._settings, "default_llm_preset", None)
        if pin:
            return pin
        for p in PresetRegistry().list():
            if getattr(p, "mode", "single") == "single":
                return p.name
        names = [p.name for p in PresetRegistry().list()]
        if not names:
            raise RuntimeError("LlamaDeck has no presets registered; cannot pick default LLM preset")
        return names[0]

    @property
    def preset_name(self) -> str:
        return self._resolve_active_preset() or self._default_preset_name()

    @property
    def managed_url(self) -> str:
        active = self._resolve_active_preset() or self._default_preset_name()
        cfg = PresetRegistry().get(active)
        host = "127.0.0.1" if cfg.host == "0.0.0.0" else cfg.host
        return f"http://{host}:{cfg.port}"

    @property
    def est_vram_mb(self) -> int:
        active = self._resolve_active_preset() or self._default_preset_name()
        try:
            cfg = PresetRegistry().get(active)
        except Exception:
            return 12000
        return int(getattr(cfg, "estimated_vram_mb", None) or 12000)

    @property
    def target_devices(self) -> list[str]:
        """llama.cpp device ids the preset about to run is pinned to, or [].

        The broker's pre-spawn VRAM check needs this: "is there room" has no
        single answer on a two-GPU box. A preset pinned to the R9700 has to be
        measured against the R9700, not against whichever card happens to be
        emptiest.
        """
        active = self._resolve_active_preset() or self._default_preset_name()
        try:
            cfg = PresetRegistry().get(active)
        except Exception:  # noqa: BLE001 — an unreadable preset just means "unknown"
            return []
        return [d for d in (getattr(cfg, "devices", None) or []) if d]

    # -- ownership-tracking hooks (called by broker / server_api) ------------

    def forget_ownership(self) -> None:
        """Drop the broker's auto-managed flag.

        Called by `GpuBroker.notify_supervisor_changed` when the user takes
        manual control (start/stop/switch via /api/server/*). After this,
        the broker will treat any currently-running LLM preset as externally
        owned and won't auto-stop it.
        """
        self._auto_managed_preset = None

    # -- Service protocol ----------------------------------------------------

    async def start(self) -> None:
        """Launch the default preset only if no LLM is currently running.

        This is idempotent: if any LLM preset is already running (user-started
        or adopted), we no-op and route to it. The broker calls `wait_ready`
        afterwards to confirm health regardless.
        """
        existing = self._resolve_active_preset()
        if existing is not None:
            log.info("LlmService.start: '%s' already running, routing there", existing)
            return
        target = self._default_preset_name()
        log.info("LlmService.start: launching default preset '%s'", target)
        await self._supervisor.start(target)
        self._auto_managed_preset = target

    async def stop(self) -> None:
        """Stop only a preset the broker itself launched.

        User/adopted presets are externally owned — keepalive expiry and slot
        transitions never kill them. The user can still stop them via
        /api/server/stop or the UI's adopt-toggle.
        """
        target = self._auto_managed_preset
        if target is None:
            log.debug("LlmService.stop: no auto-managed preset, skipping (externally owned)")
            return
        try:
            await self._supervisor.stop(target)
        except Exception as e:  # noqa: BLE001
            log.warning("LlmService.stop(%s) failed: %s", target, e)
        finally:
            self._auto_managed_preset = None

    async def wait_ready(self, timeout_s: float = 120.0) -> bool:
        """Poll whichever preset is the active LLM until /health = 200."""
        active = self._resolve_active_preset() or self._auto_managed_preset
        if active is None:
            return False
        try:
            cfg = PresetRegistry().get(active)
        except Exception:
            return False
        host = "127.0.0.1" if cfg.host == "0.0.0.0" else cfg.host
        url = f"http://{host}:{cfg.port}/health"
        deadline = asyncio.get_event_loop().time() + timeout_s
        async with httpx.AsyncClient(timeout=2.0) as client:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    r = await client.get(url)
                    if r.status_code == 200:
                        return True
                except Exception:
                    pass
                await asyncio.sleep(0.5)
        return False

    def is_running(self) -> bool:
        return self._resolve_active_preset() is not None


__all__ = ["LlmService"]
