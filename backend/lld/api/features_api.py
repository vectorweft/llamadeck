"""What's New (llama.cpp feature tracker) endpoints."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..features import (
    _probe_cache,
    FeatureError,
    get_feature_tracker,
    probe_endpoint,
    select_model,
    summary_status,
)
from ..flag_catalog import classify_flags
from ..presets import PresetError, PresetRegistry
from ..settings import LlamaServerConfig, load_settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/features", tags=["features"])


class ProbeBody(BaseModel):
    """`base_url` / `api_key` let the Settings page probe what the user is
    typing before saving it; both fall back to the stored settings."""
    base_url: str | None = None
    api_key: str | None = None


class TryBody(BaseModel):
    preset_name: str
    start: bool = False


class HintsBody(BaseModel):
    """The preset being edited, so hints can be judged against the command it
    would actually run rather than against its extra_flags alone."""
    config: dict
    architecture: str | None = None
    limit: int = 3


class AbBody(BaseModel):
    model_path: str
    n_prompts: int = 512
    n_gens: int = 128
    repetitions: int = 2


@router.get("")
async def list_cards(
    unseen_only: bool = False,
    arch: str | None = None,
    scan_to: str | None = None,
    limit: int = 100,
) -> list[dict]:
    return await get_feature_tracker().list_cards(
        unseen_only=unseen_only, arch=arch, scan_to=scan_to, limit=limit,
    )


def _arch_matches(card_archs: list[str], arch: str | None) -> bool:
    """The card names an architecture that looks like this model's.

    An EMPTY list is "the summarizer did not scope this card" — unknown, not
    universal. Treating it as a match is why a Qwen3-TTS card and a router
    scheduler card turned up on a dense text preset.
    """
    if not card_archs or not arch:
        return False
    a = arch.lower()
    return any(c and (c.lower() == a or a in c.lower() or c.lower() in a) for c in card_archs)


@router.post("/hints")
async def feature_hints(body: HintsBody) -> dict:
    """New llama.cpp features worth showing next to THIS preset.

    This panel exists to ADD a flag, so a card earns its place only by
    carrying one that can actually be applied here: real in this build, absent
    from the preset's command, and not something LlamaDeck already renders
    from a field. A card naming only `--model` / `--ctx-size` has nothing to
    offer — it belongs on the What's New page, not in the editor.

    An architecture match no longer buys inclusion (that is what put a
    Qwen3-TTS card on a dense text preset); it decides ORDER and the label,
    so the genuinely model-specific ones come first.
    """
    try:
        cfg = LlamaServerConfig(**PresetRegistry._sanitize(dict(body.config)))
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"invalid config: {e}")
    binary = load_settings().llama_bin or ""
    cards = await get_feature_tracker().list_cards(limit=40)

    hints: list[dict] = []
    for card in cards:
        flags = card.get("flags") or []
        if not flags:
            continue
        buckets = await classify_flags(cfg, binary, flags)
        if not buckets["actionable"]:
            continue
        arch_match = _arch_matches(card.get("architectures") or [], body.architecture)
        hints.append({
            "card": card,
            "match": "architecture" if arch_match else "flags",
            "add_flags": buckets["actionable"],
            **{k: v for k, v in buckets.items() if k != "actionable"},
        })
    # Architecture matches first: "this is about your model" beats "this flag
    # happens to be addable".
    hints.sort(key=lambda h: 0 if h["match"] == "architecture" else 1)
    return {"hints": hints[: max(1, body.limit)]}


@router.get("/unseen-count")
async def unseen_count() -> dict:
    return {"count": await get_feature_tracker().unseen_count()}


@router.get("/auth-status")
async def auth_status() -> dict:
    """Summarization backend status. `provider`: claude | openai. `mode`:
    api_key | env | claude_cli | profile | openai | none. For openai, `model`
    and `base_url` describe the configured endpoint."""
    return summary_status()


@router.post("/llm-endpoint")
async def probe_llm_endpoint(body: ProbeBody) -> dict:
    """Ask an OpenAI-compatible endpoint what it serves, so the Settings page
    can offer the model list instead of asking the user to type an id.

    `resolved` is the model that would actually be used right now — that is
    what an empty `llm_model` resolves to."""
    from ..settings import load_settings

    s = load_settings()
    base = (body.base_url or s.llm_base_url or "").strip().rstrip("/")
    if not base:
        raise HTTPException(status_code=400, detail="base URL is empty")
    key = body.api_key if body.api_key is not None else s.llm_api_key
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    # This endpoint IS the refresh button — never answer from the cache.
    _probe_cache.pop(base, None)
    info = await probe_endpoint(base, headers)
    out = {
        "base_url": base,
        "reachable": info["reachable"],
        "models": info["models"],
        "n_ctx": info["n_ctx"],
        "native": info["native"],
        "resolved": None,
        "detail": None,
    }
    if not info["reachable"]:
        out["detail"] = f"nothing answered at {base} — is the server running?"
        return out
    try:
        out["resolved"] = select_model((s.llm_model or "").strip(), info)
    except FeatureError as e:
        out["detail"] = str(e)
    return out


@router.get("/guide")
async def get_guide() -> dict:
    """En son rehber (yoksa {status: none})."""
    g = await get_feature_tracker().latest_guide()
    return g or {"status": "none"}


@router.post("/guide")
async def start_guide() -> dict:
    try:
        return await get_feature_tracker().start_guide()
    except FeatureError as e:
        code = 409 if "zaten" in str(e) else 400
        raise HTTPException(status_code=code, detail=str(e))


@router.get("/scans")
async def list_scans(limit: int = 20) -> list[dict]:
    return await get_feature_tracker().list_scans(limit=limit)


@router.post("/scan")
async def scan_now() -> dict:
    ft = get_feature_tracker()
    if ft.scanning():
        raise HTTPException(status_code=409, detail="a scan is already running")
    try:
        scan = await ft.run_scan(force=True)
    except FeatureError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return scan or {"status": "empty"}


@router.post("/scans/{scan_id}/retry")
async def retry_scan(scan_id: int) -> dict:
    try:
        return await get_feature_tracker().summarize_scan(scan_id)
    except FeatureError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/scans/{scan_id}")
async def delete_scan(scan_id: int) -> dict:
    """Delete a scan and its cards, so a pile of failed/pending scans (the
    usual aftermath of a local model that was too small, or an endpoint that
    was mid-change) can actually be cleared instead of accumulating forever."""
    ok = await get_feature_tracker().delete_scan(scan_id)
    if not ok:
        raise HTTPException(status_code=404, detail="scan not found")
    return {"ok": True}


@router.delete("/scans")
async def delete_scans(status: str | None = None) -> dict:
    """Bulk-delete scans, optionally only those in a given `status`
    (e.g. `?status=failed`). The frontend's "clear failed" uses this."""
    n = await get_feature_tracker().delete_scans(status=status)
    return {"deleted": n}


@router.post("/seen-all")
async def seen_all() -> dict:
    await get_feature_tracker().mark_all_seen()
    return {"ok": True}


@router.post("/{feature_id}/seen")
async def mark_seen(feature_id: int) -> dict:
    await get_feature_tracker().mark_seen(feature_id)
    return {"ok": True}


@router.post("/{feature_id}/try")
async def try_feature(feature_id: int, body: TryBody) -> dict:
    try:
        return await get_feature_tracker().try_feature(
            feature_id, body.preset_name, body.start,
        )
    except PresetError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FeatureError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # return supervisor.start errors (port conflicts etc.) readably
        raise HTTPException(status_code=500, detail=f"start error: {e}")


@router.post("/{feature_id}/ab")
async def start_ab(feature_id: int, body: AbBody) -> dict:
    try:
        return await get_feature_tracker().start_ab(
            feature_id, body.model_path,
            n_prompts=body.n_prompts, n_gens=body.n_gens,
            repetitions=body.repetitions,
        )
    except FeatureError as e:
        code = 409 if "busy" in str(e) else 400
        raise HTTPException(status_code=code, detail=str(e))


@router.get("/ab-runs")
async def list_ab_runs(feature_id: int | None = None, limit: int = 20) -> list[dict]:
    return await get_feature_tracker().list_ab_runs(feature_id=feature_id, limit=limit)
