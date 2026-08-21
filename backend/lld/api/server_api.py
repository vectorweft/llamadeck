from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

import time

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..accel import platform_info
from ..flag_catalog import get_flag_catalog
from ..gpu_broker import get_broker
from ..fit_check import check_fit_async
from ..settings import LlamaServerConfig, load_settings
from ..supervisor import SupervisorError, get_supervisor
from ..power import (
    get_cpu_percent,
    get_cpu_power_status,
    get_cpu_temp_c,
    get_energy_tracker,
    get_ram_snapshot,
    probe_cpu_power_w,
    probe_gpu_power_w,
)
from .. import vram_calib
from ..devices import (
    invalidate_cache as invalidate_device_cache,
    offload_targets,
    probe_devices,
    selectable_devices,
)
from ..fit_budget import plan_budget
from ..rpc_server import RpcServerError, get_rpc_manager
from ..vram import (
    GpuProcess,
    nvidia_probe_failure,
    offload_gpus,
    probe_gpus,
    probe_processes,
)

router = APIRouter(prefix="/api/server", tags=["server"])

# Fit-check levels that mean "starting this now would very likely OOM". A
# preflight blocks these (409) so the user gets check_fit's actionable headline
# instead of a bare "process exited (code=1)" after the server dies loading.
# "fits"/"fits_hybrid"/"fits_cpu" and "unknown" (no GGUF geometry) never block.
_FIT_BLOCK_LEVELS = {"broken", "too_big", "fits_if_alone", "needs_offload"}


async def _preflight_fit(to_preset: str, freeing: list[str], force: bool) -> None:
    """Raise 409 if starting `to_preset` would very likely fail to fit.

    `freeing` names presets that will be stopped first (switch/restart) — their
    live VRAM is added back to the free budget before judging. Any probe/parse
    failure is swallowed: a preflight must never block a start it can't assess.
    """
    if force:
        return
    sup = get_supervisor()
    statuses = sup.statuses()
    row = statuses.get(to_preset)
    if row is None:
        return  # unknown preset — let the supervisor raise the real error
    cfg_dict = row.get("config") or {}
    # Router presets pin no single model_path; check_fit can't plan them.
    if cfg_dict.get("mode") == "router":
        return
    try:
        cfg = LlamaServerConfig(**cfg_dict)
    except TypeError:
        return
    try:
        gpus, procs = await asyncio.gather(probe_gpus(), probe_processes())
        # Plan against the cards this preset is pinned to, handing back the
        # VRAM of the presets we stop on the way in. See fit_budget.
        budget = await plan_budget(
            cfg, gpus=gpus, procs=procs, statuses=statuses, freeing=freeing
        )
    except Exception:
        return
    ram = get_ram_snapshot()
    result = await check_fit_async(
        cfg,
        gpu_total_mb=budget.total_mb,
        gpu_free_mb=budget.free_mb,
        ram_total_mb=ram["total_mb"],
        ram_available_mb=ram["free_mb"],
        gpu_budget_mb=budget.budget_mb,
        lang=load_settings().ui_language,
        unified=budget.unified,
    )
    if result.get("level") in _FIT_BLOCK_LEVELS:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "fit_block",
                "level": result["level"],
                "headline": result.get("headline")
                or f"{to_preset}: won't fit with current settings",
                "messages": result.get("messages", []),
            },
        )


async def _notify_broker() -> None:
    """Best-effort broker reconciliation after a supervisor mutation.

    Called after start/stop/restart/switch/adopt/release so the broker can
    drop stale "llm" leases when the user kills the active LLM, and clear
    its auto-managed ownership flag (so a future keepalive doesn't try to
    touch a now-externally-owned preset).

    Failures here must NOT mask the supervisor operation's success.
    """
    try:
        await get_broker().notify_supervisor_changed()
    except RuntimeError:
        # Broker not initialised (e.g. unit-test harness without lifespan)
        pass
    except Exception:
        # Reconciliation is advisory; never bubble up to the API caller.
        pass


class AdoptBody(BaseModel):
    pid: int
    preset: str | None = None


class SwitchBody(BaseModel):
    from_preset: str | None = None
    to_preset: str
    force: bool = False


@router.get("/statuses")
async def statuses() -> dict:
    return {"presets": get_supervisor().statuses()}


@router.get("/status/{preset}")
async def status_one(preset: str) -> dict:
    all_ = get_supervisor().statuses()
    if preset not in all_:
        raise HTTPException(status_code=404, detail=f"preset not found: {preset}")
    return all_[preset]


@router.get("/scan")
async def scan() -> dict:
    return {"found": get_supervisor().scan_existing()}


@router.get("/devices")
async def devices() -> dict:
    """Offload targets of the configured llama-server binary.

    Distinct from /vram: that reports the hardware the OS sees, this reports
    what *this build* can offload to. A CUDA-only binary on a box with a
    Radeon returns one device — which is the honest answer for `-dev`.

    Rows the user must not pin a preset to are returned too, flagged rather
    than hidden, so the editor can explain the omission (a second alias of one
    physical card, the iGPU, a software rasterizer).

    The CPU row is appended on top of what the binary reports, because
    `--list-devices` prints GPUs only and "offload nothing" is a target the
    editor has to be able to offer. It is appended only when the probe
    succeeded: a lone CPU row next to an empty GPU list would read as "this
    box has no GPUs" when the truth is that the binary could not be queried,
    and the editor's existing message says that far better.
    """
    s = load_settings()
    binary = s.llama_bin or ""
    devs = await probe_devices(binary)
    if devs:
        devs = offload_targets(devs)
    return {
        "binary": binary,
        "devices": [d.to_dict() for d in devs],
        "selectable_ids": [d.id for d in selectable_devices(devs)],
    }


@router.get("/flags")
async def flags() -> dict:
    """Every flag the configured llama-server binary accepts.

    Read from `llama-server --help`, so it tracks the user's own build: the
    command box can then tell "this build has no such flag" apart from
    "LlamaDeck has not heard of it", and offer a spelling correction instead
    of letting the process fail at launch.

    `available: false` means the binary could not be run at all — callers must
    then validate nothing, because an unqueryable binary is not evidence
    against the user's command.
    """
    binary = load_settings().llama_bin or ""
    catalog = await get_flag_catalog(binary)
    return {"binary": binary, **catalog.to_dict()}


@router.get("/rpc")
async def rpc_servers() -> dict:
    """Configured RPC offload servers and whether each is listening."""
    return {"servers": [p.to_dict() for p in get_rpc_manager().all()]}


@router.post("/rpc/{name}/start")
async def rpc_start(name: str) -> dict:
    mgr = get_rpc_manager()
    try:
        proc = mgr.get(name)
        await proc.start()
    except RpcServerError as e:
        raise HTTPException(status_code=409, detail=str(e))
    # A new endpoint changes which devices exist, and the probe memoizes them.
    invalidate_device_cache()
    return proc.to_dict()


@router.post("/rpc/{name}/stop")
async def rpc_stop(name: str) -> dict:
    mgr = get_rpc_manager()
    try:
        proc = mgr.get(name)
        await proc.stop()
    except RpcServerError as e:
        raise HTTPException(status_code=404, detail=str(e))
    invalidate_device_cache()
    return proc.to_dict()


@router.get("/vram")
async def vram() -> dict:
    gpus, procs, gpu_power_list, cpu_w = await asyncio.gather(
        probe_gpus(), probe_processes(), probe_gpu_power_w(), probe_cpu_power_w()
    )
    gpu_w = sum(gpu_power_list) if gpu_power_list else None
    total_w: float | None
    if gpu_w is not None and cpu_w is not None:
        total_w = gpu_w + cpu_w
    elif gpu_w is not None:
        total_w = gpu_w
    elif cpu_w is not None:
        total_w = cpu_w
    else:
        total_w = None
    # Sum only the cards a model can actually be planned onto. `gpus` still
    # carries the integrated one so the UI can show it, but adding a desktop
    # Ryzen iGPU's ~46 GB GTT aperture here would report system RAM as VRAM —
    # on this box that inflated the total from 65 GB to 113 GB.
    budget = offload_gpus(gpus)
    total = sum(g.total_mb for g in budget)
    used = sum(g.used_mb for g in budget)
    free = sum(g.free_mb for g in budget)
    # Build PID → preset/model label map from supervisor so the UI can show
    # *what* is occupying VRAM (e.g. "qwen3.6-coding" vs "embed-bge"), not
    # just a process name.
    statuses = get_supervisor().statuses()
    pid_to_label: dict[int, dict] = {}
    for name, row in statuses.items():
        pid = row.get("pid")
        if not pid or not row.get("running"):
            continue
        cfg = row.get("config", {}) or {}
        model_path = cfg.get("model_path") or cfg.get("hf_file") or cfg.get("hf_repo") or ""
        # Last path segment without .gguf, falls back to preset name.
        label = model_path.rsplit("/", 1)[-1].removesuffix(".gguf") if model_path else name
        pid_to_label[int(pid)] = {
            "preset": name,
            "model": label,
            "adopted": bool(row.get("adopted")),
        }
    process_rows = []
    for p in procs:
        info = pid_to_label.get(p.pid)
        process_rows.append({
            "pid": p.pid,
            "process_name": p.process_name,
            "used_mb": p.used_mb,
            "preset": info["preset"] if info else None,
            "model": info["model"] if info else None,
            "adopted": info["adopted"] if info else False,
        })
    process_rows.sort(key=lambda r: r["used_mb"], reverse=True)
    # Learn from what the card actually reports. A preset that has been up long
    # enough for its allocations to settle is ground truth for its model — far
    # better than the formula, which runs high on some architectures and would
    # otherwise keep refusing to start a preset the user has already run.
    # One process can hold memory on two cards at once: a Vulkan-pinned server
    # still opens a CUDA context on the other one, worth a few hundred MB. The
    # card it actually runs on is the row holding the most, and that row's
    # vendor is what the measurement gets keyed by — recording the context as
    # if it were the model would poison the other card's calibration.
    largest: dict[int, GpuProcess] = {}
    for p in procs:
        if p.used_mb > (largest[p.pid].used_mb if p.pid in largest else 0):
            largest[p.pid] = p
    for p in largest.values():
        info = pid_to_label.get(p.pid)
        if not info or info["adopted"] or p.used_mb <= 0:
            continue
        row = statuses.get(info["preset"]) or {}
        if (row.get("uptime_seconds") or 0) < vram_calib.MIN_UPTIME_S:
            continue
        est = row.get("vram_estimate") or {}
        model_path = (row.get("config") or {}).get("model_path")
        if est.get("gpu_mb") and model_path:
            await vram_calib.record(
                model_path,
                measured_mb=p.used_mb,
                estimated_mb=int(est["gpu_mb"]) + vram_calib.offset_mb(
                    model_path, (row.get("config") or {}).get("devices")
                ),
                vendor=p.vendor,
            )
    # Sum of estimated VRAM for currently active presets. Prefer the dynamic
    # estimate (computed from live ctx/np/kv-quant + GGUF geometry); fall back
    # to the static preset value when GGUF metadata isn't readable.
    active_estimate = 0
    for row in statuses.values():
        if not row.get("running"):
            continue
        est = row.get("vram_estimate")
        if est and est.get("total_mb"):
            # gpu_mb excludes weights parked in host RAM (--n-cpu-moe / low ngl)
            active_estimate += int(est.get("gpu_mb") or est["total_mb"])
        elif row.get("config", {}).get("estimated_vram_mb"):
            active_estimate += row["config"]["estimated_vram_mb"]
    return {
        "gpus": [asdict(g) for g in gpus],
        "total_mb": total,
        "used_mb": used,
        "free_mb": free,
        # Apple Silicon / AMD APU: these "VRAM" bytes are system RAM. The UI
        # labels the panel accordingly instead of showing a second, fake pool.
        # Same rule as the totals above: only the cards a model can be planned
        # onto. `gpus` still carries the iGPU for display, and letting it vote
        # here labelled a discrete 5090/R9700 box "shared memory".
        "unified_memory": any(g.unified for g in budget),
        "platform": platform_info().to_dict(),
        "active_estimate_mb": active_estimate,
        "processes": process_rows,
        "power": {
            "gpu_w": gpu_w,
            "cpu_w": cpu_w,
            "cpu_status": get_cpu_power_status(),
            "total_w": total_w,
            **get_energy_tracker().snapshot(),
        },
        "cpu_percent": get_cpu_percent(),
        "cpu_temp_c": get_cpu_temp_c(),
        "ram": get_ram_snapshot(),
        # Why a card the machine has is missing from `gpus`. Without this a
        # driver in a bad state reads as "you have no NVIDIA GPU" — the panel
        # simply shows one card fewer, and nothing on the page says so.
        "probe_warning": nvidia_probe_failure(),
    }


@router.post("/start/{preset}")
async def start(preset: str, force: bool = False) -> dict:
    await _preflight_fit(preset, [], force)
    try:
        result = await get_supervisor().start(preset)
    except SupervisorError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await _notify_broker()
    return result


@router.post("/stop/{preset}")
async def stop(preset: str) -> dict:
    try:
        result = await get_supervisor().stop(preset)
    except SupervisorError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await _notify_broker()
    return result


@router.post("/restart/{preset}")
async def restart(preset: str, force: bool = False) -> dict:
    # The preset's own live VRAM returns when it stops, so add it to the budget.
    await _preflight_fit(preset, [preset], force)
    try:
        result = await get_supervisor().restart(preset)
    except SupervisorError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await _notify_broker()
    return result


@router.post("/switch")
async def switch(body: SwitchBody) -> dict:
    await _preflight_fit(
        body.to_preset, [body.from_preset] if body.from_preset else [], body.force
    )
    try:
        result = await get_supervisor().switch(body.from_preset or "", body.to_preset)
    except SupervisorError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await _notify_broker()
    return result


@router.post("/adopt")
async def adopt(body: AdoptBody) -> dict:
    try:
        result = await get_supervisor().adopt(body.pid, body.preset)
    except SupervisorError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await _notify_broker()
    return result


@router.post("/release/{preset}")
async def release(preset: str) -> dict:
    try:
        result = await get_supervisor().release(preset)
    except SupervisorError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await _notify_broker()
    return result


@router.get("/wait_ready/{preset}")
async def wait_ready(preset: str, timeout: float = 120.0, poll_ms: int = 500) -> dict:
    """Poll the preset's llama-server /health until it responds 200 OK or
    `timeout` seconds elapse. Use this after start_preset to know when the
    model has finished loading and the server can accept completions.

    Designed for VRAM-swap workflows (e.g. hermes-agent yields VRAM to
    ComfyUI, then reclaims the LLM): stop_preset → run-other-workload →
    start_preset → wait_ready → resume.
    """
    sup = get_supervisor()
    statuses = sup.statuses()
    if preset not in statuses:
        raise HTTPException(status_code=404, detail=f"preset not found: {preset}")
    cfg = statuses[preset]["config"]
    host = cfg.get("host") or "127.0.0.1"
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = cfg.get("port")
    if not port:
        raise HTTPException(status_code=400, detail=f"{preset}: no port configured")

    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + max(timeout, 1.0)
    last_err: str | None = None
    attempts = 0
    async with httpx.AsyncClient(timeout=2.0) as c:
        while time.monotonic() < deadline:
            attempts += 1
            # If the supervisor knows the process exited, fail fast — no point
            # polling a dead port.
            row = sup.statuses().get(preset, {})
            if not row.get("running") and row.get("returncode") is not None:
                raise HTTPException(
                    status_code=409,
                    detail=f"{preset}: process exited (returncode={row['returncode']}) before becoming ready",
                )
            try:
                r = await c.get(url)
                if r.status_code == 200:
                    return {
                        "preset": preset,
                        "ready": True,
                        "elapsed_seconds": round(timeout - max(deadline - time.monotonic(), 0), 2),
                        "attempts": attempts,
                    }
                last_err = f"HTTP {r.status_code}"
            except httpx.HTTPError as e:
                last_err = type(e).__name__
            await asyncio.sleep(max(poll_ms, 100) / 1000.0)
    return {
        "preset": preset,
        "ready": False,
        "timeout_seconds": timeout,
        "attempts": attempts,
        "last_error": last_err,
    }


@router.get("/logs/tail/{preset}")
async def logs_tail(preset: str, n: int = 500) -> dict:
    return {"lines": get_supervisor().log_tail(preset, n)}


@router.get("/logs/stream/{preset}")
async def logs_stream(preset: str):
    sup = get_supervisor()

    async def event_generator():
        try:
            async for line in sup.stream_logs(preset):
                yield f"data: {json.dumps({'line': line})}\n\n"
        except asyncio.CancelledError:
            return

    return StreamingResponse(event_generator(), media_type="text/event-stream")
