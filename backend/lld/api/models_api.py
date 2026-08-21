from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Query

from ..fit_check import check_fit_async
from ..model_defaults import get_model_defaults, get_recommended_sampling
from ..model_info import info_for_path
from ..model_profiles import get_model_profile
from ..models import full_rescan, list_models
from ..power import get_ram_snapshot
from ..presets import PresetRegistry
from ..settings import LlamaServerConfig, load_settings
from ..supervisor import get_supervisor
from ..verify import get_verify_registry
from ..fit_budget import plan_budget
from ..vram import probe_gpus, probe_processes

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
async def list_all(family: str | None = None) -> list[dict]:
    return await list_models(family)


@router.get("/families")
async def families() -> list[str]:
    rows = await list_models()
    fams = sorted({r["family"] for r in rows if r["family"]})
    return fams


@router.post("/scan")
async def scan() -> dict:
    s = load_settings()
    return await full_rescan(s.scan_roots)


@router.get("/defaults")
async def model_defaults(
    path: str = Query(..., description="Absolute path to a .gguf file"),
    preset: str | None = Query(None, description="If set and running, merge live /props"),
) -> dict:
    """Return the recommended sampling defaults + arch metadata for a GGUF.

    Priority: GGUF `general.sampling.*` → live llama-server `/props` (if preset
    is running) → curated family table → empty. The `source` field reports
    which path won.
    """
    status = None
    if preset:
        all_statuses = get_supervisor().statuses()
        status = all_statuses.get(preset)
    d = await get_model_defaults(path, preset_status=status)
    return d.to_dict()


@router.get("/profile")
async def model_profile(
    path: str = Query(..., description="Absolute path to a .gguf file"),
    preset: str | None = Query(None, description="If set and running, merge live /props"),
    mmproj: str | None = Query(None, description="Projector next to the model, if any"),
) -> dict:
    """What this model can do, and the one-click recipes that configure it.

    Unlike `/defaults` (numbers) and `/info` (prose), this answers "which
    buttons should the preset editor show for THIS model" — thinking on/off
    with the sampling each mode wants, plus anything the user added to
    `~/.config/llamadeck/model-profiles.json`.

    Capabilities are read from the GGUF's own chat template, so a model
    downloaded today works without waiting for a table entry.
    """
    status = None
    if preset:
        status = get_supervisor().statuses().get(preset)
    prof = await get_model_profile(
        path,
        preset_status=status,
        mmproj_path=mmproj,
        lang=load_settings().ui_language,
    )
    return prof.to_dict()


@router.get("/recommended-sampling")
async def recommended_sampling_by_path(
    path: str = Query(..., description="Absolute path to a .gguf file"),
    preset: str | None = Query(None),
) -> dict:
    """Mode-aware base sampling for a single model.

    Returns BOTH `thinking` and `non_thinking` variants — Qwen3.x and similar
    families publish distinct values per mode. Callers (HFD-2) pick the right
    one based on whether the request enables thinking.

    Layer your use-case tuning on top: presence_penalty, repeat_penalty,
    per-expert temperature override, etc.
    """
    status = None
    if preset:
        all_statuses = get_supervisor().statuses()
        status = all_statuses.get(preset)
    rs = await get_recommended_sampling(path, preset_status=status)
    return rs.to_dict()


@router.get("/recommended-sampling/active")
async def recommended_sampling_active(
    port: int | None = Query(None, description="Match by port (e.g. 8080 for HFD-2). If omitted, returns the first running preset with a model_path."),
) -> dict:
    """Mode-aware base sampling for whichever model is currently running.

    HFD-2's primary entry point: hit this on startup (or per-request if you
    want to re-resolve after a router-mode swap). The `port` filter pins to
    a specific bind so HFD-2 always gets the model on :8080 even if other
    presets are also running.
    """
    statuses = get_supervisor().statuses()
    chosen = None
    for s in statuses.values():
        if not s.get("running"):
            continue
        cfg = s.get("config") or {}
        if port is not None and cfg.get("port") != port:
            continue
        if not cfg.get("model_path"):
            continue  # router-mode preset has no single model_path
        chosen = s
        break

    if chosen is None:
        return {
            "model_id": None,
            "source": "none",
            "architecture": None,
            "fallback_family": None,
            "thinking": {},
            "non_thinking": {},
            "notes": (
                "no running preset matches"
                + (f" port={port}" if port is not None else "")
                + " (router-mode presets are skipped — they don't pin a single model_path)"
            ),
        }

    rs = await get_recommended_sampling(
        chosen["config"]["model_path"], preset_status=chosen
    )
    out = rs.to_dict()
    out["preset"] = chosen.get("name")
    out["port"] = chosen["config"].get("port")
    out["model_path"] = chosen["config"].get("model_path")
    return out


@router.post("/fit-check")
async def fit_check(body: dict) -> dict:
    """Fit-check: takes a draft preset config (possibly unsaved), compares it
    against live GPU/RAM state and returns a level + messages + one-click
    applicable suggestions. The preset editor's fit panel calls this.
    """
    sanitized = PresetRegistry._sanitize({**body, "name": body.get("name") or "fit-check"})
    try:
        cfg = LlamaServerConfig(**sanitized)
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"invalid config: {e}")
    gpus, procs = await probe_gpus(), await probe_processes()
    ram = get_ram_snapshot()
    # Which cards this preset is planned against — and what stopping our own
    # running models would give it back. See fit_budget.
    budget = await plan_budget(
        cfg, gpus=gpus, procs=procs, statuses=get_supervisor().statuses()
    )
    return await check_fit_async(
        cfg,
        gpu_total_mb=budget.total_mb,
        gpu_free_mb=budget.free_mb,
        ram_total_mb=ram["total_mb"],
        ram_available_mb=ram["free_mb"],
        gpu_budget_mb=budget.budget_mb,
        lang=load_settings().ui_language,
        # Apple Silicon / AMD APU: "VRAM" is system RAM, so the two budgets
        # are one pool and must not be double-spent.
        unified=budget.unified,
    )


@router.get("/info")
async def model_info(
    path: str = Query(..., description="Absolute path to a .gguf file"),
    preset: str | None = Query(None, description="If set and running, merge live /props"),
) -> dict:
    """Return curated narrative documentation + sampling defaults + arch metadata.

    Combines `model_defaults` (numbers — sampling, ctx, arch from GGUF/props/family)
    with `model_info` (prose — prompt format, behavior, deployment notes, caveats,
    references). Powers the per-model info panel in the Models UI.
    """
    status = None
    if preset:
        all_statuses = get_supervisor().statuses()
        status = all_statuses.get(preset)
    defaults = await get_model_defaults(path, preset_status=status)
    rec = await get_recommended_sampling(path, preset_status=status)
    info = info_for_path(path, gguf_name=defaults.name, base_model=defaults.base_model)
    return {
        "defaults": defaults.to_dict(),
        "recommended": rec.to_dict(),
        "info": info.to_dict() if info else None,
    }


@router.post("/verify")
async def verify_start(
    path: str = Query(..., description="Absolute path to a .gguf file (part 1 of a split)"),
) -> dict:
    """Checksum a model against the sha256 huggingface_hub recorded for it.

    Corruption that survives a copy keeps the exact file size, so nothing else
    in the stack notices it — llama.cpp loads the damaged weights and the model
    answers with fluent nonsense. Hashing is the only way to tell."""
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"not a file: {path}")
    return get_verify_registry().start(path).to_dict()


@router.get("/verify")
async def verify_status(
    path: str = Query(..., description="Absolute path to a .gguf file"),
) -> dict:
    job = get_verify_registry().get(path)
    if job is None:
        raise HTTPException(status_code=404, detail="no verification run for this model")
    return job.to_dict()
