"""Router-mode API: controls a running router preset's loaded sub-models via
POST /models/load and POST /models/unload, lists available models from
GET /models, and (re)generates the per-model INI overrides file."""
from __future__ import annotations

from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..presets import PresetRegistry
from ..router_ini import render_ini, write_ini
from ..settings import STATE_DIR
from ..supervisor import get_supervisor


def _first_router_preset():
    for cfg in PresetRegistry().list():
        if getattr(cfg, "mode", "single") == "router":
            return cfg
    return None

router = APIRouter(prefix="/api/router", tags=["router"])

INI_PATH = STATE_DIR / "router-models.ini"


class LoadBody(BaseModel):
    model: str
    autoload: bool | None = None


class IniBody(BaseModel):
    models_dir: str | None = None


def _running_router() -> dict:
    """Return the status dict of the first running router preset, else 409."""
    sup = get_supervisor()
    for status in sup.statuses(vram_estimates=False).values():
        cfg = status.get("config") or {}
        if cfg.get("mode") == "router" and status.get("running"):
            return status
    raise HTTPException(
        status_code=409,
        detail="no router preset is running — start a preset with mode=router first",
    )


def _base_url(status: dict) -> str:
    cfg = status["config"]
    host = cfg.get("host") or "127.0.0.1"
    if host == "0.0.0.0":
        host = "127.0.0.1"
    return f"http://{host}:{cfg['port']}"


@router.get("/active")
async def active() -> dict:
    """Identifies which preset (if any) is currently running as router."""
    sup = get_supervisor()
    for name, status in sup.statuses(vram_estimates=False).items():
        cfg = status.get("config") or {}
        if cfg.get("mode") == "router" and status.get("running"):
            return {"running": True, "preset": name, "status": status}
    return {"running": False, "preset": None, "status": None}


_VOCAB_PREFIX = "ggml-vocab-"


def _filter_models(payload: dict) -> dict:
    """Strip llama.cpp's vocab-test stub files (ggml-vocab-*) from any
    list-shaped fields. They live in models-dir as side effects of llama.cpp's
    test corpus, never load as real models, and clutter the UI."""
    for key in ("data", "models"):
        items = payload.get(key)
        if isinstance(items, list):
            payload[key] = [
                m for m in items
                if not (
                    isinstance(m, dict)
                    and (m.get("id", "") or m.get("name", "")).startswith(_VOCAB_PREFIX)
                )
            ]
    return payload


@router.get("/models")
async def list_models() -> dict:
    """Proxy GET /models on the running router. Returns
    `{"data": [], "running": false}` when no router is up so callers can poll
    without try/except gymnastics."""
    sup = get_supervisor()
    chosen = None
    for status in sup.statuses(vram_estimates=False).values():
        cfg = status.get("config") or {}
        if cfg.get("mode") == "router" and status.get("running"):
            chosen = status
            break
    if chosen is None:
        return {"data": [], "models": [], "running": False, "router_preset": None}
    url = f"{_base_url(chosen)}/models"
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(url)
            r.raise_for_status()
            payload = r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"router /models failed: {e}")
    payload = _filter_models(payload)
    payload["running"] = True
    payload["router_preset"] = chosen.get("name")
    return payload


def _id_candidates(model: str) -> list[str]:
    """llama.cpp's router accepts model ids in two flavours depending on
    source: INI custom-section names use the bare id (`Qwen3.6-27B-UD-Q4_K_XL`),
    while files discovered via --models-dir use the basename including the
    .gguf extension. Try both so callers don't have to guess."""
    seen: list[str] = []
    if model:
        seen.append(model)
        if model.endswith(".gguf"):
            seen.append(model[:-5])
        else:
            seen.append(model + ".gguf")
    return seen


@router.post("/load")
async def load(body: LoadBody) -> dict:
    status = _running_router()
    base = _base_url(status)
    params = {}
    if body.autoload is not None:
        params["autoload"] = "true" if body.autoload else "false"
    last_err: tuple[int, str] | None = None
    try:
        async with httpx.AsyncClient(timeout=300) as c:
            for cand in _id_candidates(body.model):
                r = await c.post(
                    f"{base}/models/load",
                    json={"model": cand},
                    params=params,
                )
                if r.status_code < 400:
                    return r.json()
                last_err = (r.status_code, r.text)
                if r.status_code != 404:
                    break
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"router /models/load failed: {e}")
    if last_err:
        raise HTTPException(
            status_code=last_err[0],
            detail=f"all id candidates ({_id_candidates(body.model)}) failed: {last_err[1]}",
        )
    raise HTTPException(status_code=400, detail="empty model id")


@router.post("/unload")
async def unload(body: LoadBody) -> dict:
    status = _running_router()
    base = _base_url(status)
    last_err: tuple[int, str] | None = None
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            for cand in _id_candidates(body.model):
                r = await c.post(f"{base}/models/unload", json={"model": cand})
                if r.status_code < 400:
                    return r.json()
                last_err = (r.status_code, r.text)
                if r.status_code != 404:
                    break
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"router /models/unload failed: {e}")
    if last_err:
        raise HTTPException(
            status_code=last_err[0],
            detail=f"all id candidates ({_id_candidates(body.model)}) failed: {last_err[1]}",
        )
    raise HTTPException(status_code=400, detail="empty model id")


@router.get("/ini/preview")
async def ini_preview(models_dir: str | None = None) -> dict:
    """Render the INI without writing. Defaults to the active router's models_dir."""
    md = models_dir
    if md is None:
        try:
            status = _running_router()
            md = status["config"].get("models_dir")
        except HTTPException:
            pass
    if not md:
        raise HTTPException(
            status_code=400,
            detail="models_dir required (no router running and not provided)",
        )
    return {"path": str(INI_PATH), "ini": render_ini(md, router_preset=_first_router_preset())}


@router.post("/ini/write")
async def ini_write(body: IniBody) -> dict:
    md = body.models_dir
    if md is None:
        try:
            status = _running_router()
            md = status["config"].get("models_dir")
        except HTTPException:
            pass
    if not md:
        raise HTTPException(
            status_code=400,
            detail="models_dir required (no router running and not provided)",
        )
    text = write_ini(Path(INI_PATH), md, router_preset=_first_router_preset())
    return {"path": str(INI_PATH), "ini": text, "bytes": len(text)}
