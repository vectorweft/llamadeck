"""Preflight fit-check guard on /api/server start/switch/restart.

_preflight_fit turns check_fit's "won't fit" verdicts into a 409 *before* the
llama-server is spawned, so the user gets an actionable headline instead of a
bare "process exited (code=1)" after an OOM. These tests patch the hardware
probes and check_fit so no GPU / GGUF is needed; they pin the behaviour that
matters: which levels block, that force bypasses, and that switch/restart add
the freed preset's VRAM back to the budget.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from lld.api import server_api
from lld.vram import GpuInfo, GpuProcess


class _FakeSup:
    def __init__(self, statuses: dict):
        self._statuses = statuses

    def statuses(self) -> dict:
        return self._statuses


def _row(name: str, port: int, running: bool = False, pid: int | None = None,
         mode: str = "single") -> dict:
    return {
        "name": name, "running": running, "pid": pid,
        "config": {"name": name, "model_path": "/x.gguf", "port": port, "mode": mode},
    }


def _patch_hw(monkeypatch, *, gpu_total=24000, gpu_free=2000, procs=None,
              level="too_big"):
    async def _gpus():
        return [GpuInfo(index=0, name="x", total_mb=gpu_total,
                        used_mb=gpu_total - gpu_free, free_mb=gpu_free)]

    async def _procs():
        return procs or []

    captured = {}

    async def _fit(cfg, **kw):
        captured.update(kw)
        return {"level": level, "headline": "won't fit", "messages": []}

    monkeypatch.setattr(server_api, "probe_gpus", _gpus)
    monkeypatch.setattr(server_api, "probe_processes", _procs)
    monkeypatch.setattr(server_api, "get_ram_snapshot",
                        lambda: {"total_mb": 64000, "free_mb": 40000})
    monkeypatch.setattr(server_api, "check_fit_async", _fit)
    monkeypatch.setattr(server_api, "load_settings",
                        lambda: type("S", (), {"ui_language": "en"})())
    return captured


@pytest.mark.asyncio
async def test_blocks_on_too_big(monkeypatch):
    monkeypatch.setattr(server_api, "get_supervisor",
                        lambda: _FakeSup({"big": _row("big", 8081)}))
    _patch_hw(monkeypatch, level="too_big")
    with pytest.raises(HTTPException) as ei:
        await server_api._preflight_fit("big", [], force=False)
    assert ei.value.status_code == 409
    assert ei.value.detail["code"] == "fit_block"
    assert ei.value.detail["level"] == "too_big"


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ["fits", "fits_hybrid", "fits_cpu", "unknown"])
async def test_allows_fitting_levels(monkeypatch, level):
    monkeypatch.setattr(server_api, "get_supervisor",
                        lambda: _FakeSup({"ok": _row("ok", 8081)}))
    _patch_hw(monkeypatch, level=level)
    await server_api._preflight_fit("ok", [], force=False)  # no raise


@pytest.mark.asyncio
async def test_force_bypasses(monkeypatch):
    # force short-circuits before any probing, so even a too_big verdict passes.
    monkeypatch.setattr(server_api, "get_supervisor",
                        lambda: _FakeSup({"big": _row("big", 8081)}))
    _patch_hw(monkeypatch, level="too_big")
    await server_api._preflight_fit("big", [], force=True)  # no raise


@pytest.mark.asyncio
async def test_router_preset_skipped(monkeypatch):
    monkeypatch.setattr(server_api, "get_supervisor",
                        lambda: _FakeSup({"r": _row("r", 8085, mode="router")}))
    _patch_hw(monkeypatch, level="too_big")
    await server_api._preflight_fit("r", [], force=False)  # routers not planned


@pytest.mark.asyncio
async def test_unknown_preset_skipped(monkeypatch):
    monkeypatch.setattr(server_api, "get_supervisor", lambda: _FakeSup({}))
    _patch_hw(monkeypatch, level="too_big")
    await server_api._preflight_fit("ghost", [], force=False)  # not tracked → skip


@pytest.mark.asyncio
async def test_switch_adds_freed_vram_to_budget(monkeypatch):
    """The from-preset's live VRAM must be added back before judging the
    to-preset — otherwise a swap that clearly fits gets falsely blocked."""
    statuses = {
        "old": _row("old", 8081, running=True, pid=111),
        "new": _row("new", 8082),
    }
    monkeypatch.setattr(server_api, "get_supervisor", lambda: _FakeSup(statuses))
    procs = [GpuProcess(pid=111, process_name="llama-server", used_mb=18000)]
    captured = _patch_hw(monkeypatch, gpu_free=2000, procs=procs, level="fits")
    await server_api._preflight_fit("new", ["old"], force=False)
    # free 2000 + freed 18000 from stopping "old" = 20000 effective free.
    assert captured["gpu_free_mb"] == 20000


@pytest.mark.asyncio
async def test_probe_failure_never_blocks(monkeypatch):
    monkeypatch.setattr(server_api, "get_supervisor",
                        lambda: _FakeSup({"big": _row("big", 8081)}))

    async def _boom():
        raise RuntimeError("nvidia-smi gone")

    monkeypatch.setattr(server_api, "probe_gpus", _boom)
    monkeypatch.setattr(server_api, "probe_processes", _boom)
    await server_api._preflight_fit("big", [], force=False)  # swallowed → no raise
