from __future__ import annotations

import logging
from dataclasses import asdict, replace
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..argv import (
    _is_number,
    command_argv,
    config_diff,
    flag_conflicts,
    merge_command,
    shadowed_flags,
    to_argv,
    to_command,
)
from ..devices import CPU_DEVICE_ID, probe_devices, unknown_device_ids
from ..flag_catalog import flags_missing_values, get_flag_catalog
from ..presets import PresetError, PresetRegistry
from ..router_ini import write_ini
from ..settings import STATE_DIR, LlamaServerConfig, load_settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/presets", tags=["presets"])


def _build_config(body: dict, name_override: str | None = None) -> LlamaServerConfig:
    """Construct LlamaServerConfig from a request body, defensively:
    - Drops keys that aren't fields (forward-compat with frontend ahead of backend)
    - Coerces None for fields with non-None defaults via PresetRegistry._sanitize
    - Raises HTTPException(400) with a readable message on TypeError
    """
    if name_override is not None:
        body = {**body, "name": name_override}
    sanitized = PresetRegistry._sanitize(dict(body))
    try:
        cfg = LlamaServerConfig(**sanitized)
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"invalid config: {e}")
    return _sync_from_override(cfg)


_ENV_KEY_RE = __import__("re").compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _reject_bad_env(cfg: LlamaServerConfig) -> None:
    """A key the shell could not set is a typo, not a variable.

    Caught here rather than at spawn: `execve` rejects the whole environment,
    so one malformed key would turn into "preset won't start" with no clue
    which field caused it.
    """
    for k in (getattr(cfg, "env", None) or {}):
        if not _ENV_KEY_RE.match(str(k)):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'{k}' is not a usable environment variable name — use "
                    f"letters, digits and underscore, not starting with a digit "
                    f"(e.g. GGML_CUDA_DISABLE_GRAPHS)."
                ),
            )


def _sync_from_override(cfg: LlamaServerConfig) -> LlamaServerConfig:
    """Make the fields mirror the raw command whenever one is stored.

    With `argv_override` set, the command is what runs — but the rest of
    LlamaDeck still reads the FIELDS to decide where to health-check, how much
    VRAM to budget and what to write into the router INI. Leaving them at
    whatever the form last held is how a panel ends up reporting 32K context
    for a process running 128K. So the command is parsed back over the fields
    on every save: one direction, always, no partial magic.
    """
    raw = (getattr(cfg, "argv_override", None) or "").strip()
    if not raw:
        return cfg
    try:
        parsed, _ = merge_command(cfg, raw)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"the command does not parse as a shell command line: {e}",
        )
    return parsed


def _registry() -> PresetRegistry:
    return PresetRegistry()


def _refresh_router_ini() -> None:
    """Re-emit router-models.ini for every router preset after a sibling
    single-mode preset is upserted/deleted. Best-effort: log and continue."""
    try:
        for cfg in _registry().list():
            if getattr(cfg, "mode", "single") != "router":
                continue
            ini_path = Path(cfg.models_preset_path) if cfg.models_preset_path else (STATE_DIR / "router-models.ini")
            md = cfg.models_dir or load_settings().hf_models_root
            write_ini(ini_path, md, router_preset=cfg)
            log.info("router INI refreshed (preset edit): %s", ini_path)
    except Exception as e:
        log.warning("router INI refresh after preset edit failed: %s", e)


def _binary() -> str:
    return load_settings().llama_bin or "llama-server"


async def _missing_flag_values(argv: list[str]) -> list[dict]:
    """Flags in `argv` that need a value and have none — see flag_catalog.

    Advisory here rather than fatal: the editor shows it while typing, and the
    supervisor refuses the start if it is still there when Start is pressed.
    """
    return flags_missing_values(await get_flag_catalog(_binary()), argv)


async def _unknown_flags(argv: list[str]) -> list[dict]:
    """Flags in `argv` the configured binary does not accept, with the closest
    spellings it does. Empty when the binary cannot be queried — a build we
    failed to run says nothing about whether the user's command is valid."""
    catalog = await get_flag_catalog(_binary())
    if not catalog.available:
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for tok in argv[1:]:
        if not tok.startswith("-") or tok == "-" or _is_number(tok):
            continue
        name = tok.split("=", 1)[0]
        if catalog.known(name) or name in seen:
            continue
        seen.add(name)
        out.append({"flag": name, "suggestions": catalog.suggest(name)})
    return out


@router.post("/command")
async def render_command(body: dict) -> dict:
    """The exact command line a draft preset would execute.

    `command` is what runs — the raw override when one is set, the fields
    otherwise. `fields_command` always renders the fields, so the editor can
    seed the box when the user switches the override on without losing what
    the form was already saying.
    """
    body = {"name": "draft", **body}
    cfg = _build_config(body)
    binary = _binary()
    override = (cfg.argv_override or "").strip()
    fields_only = replace(cfg, argv_override=None)
    try:
        command = to_command(cfg, binary)
        argv = to_argv(cfg, binary)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"the command does not parse: {e}")
    return {
        "binary": binary,
        "source": "override" if override else "fields",
        "command": command,
        "fields_command": to_command(fields_only, binary),
        "argv": argv,
        "unknown_flags": await _unknown_flags(argv),
        "missing_values": await _missing_flag_values(argv),
        "shadowed": shadowed_flags(argv),
        "conflicts": flag_conflicts(argv),
    }


@router.post("/command/parse")
async def parse_command(body: dict) -> dict:
    """Read a hand-typed command back into preset fields.

    The other half of the round trip: `POST /command` renders, this parses.
    Both go through `lld.argv`, so the editor cannot drift from what the
    supervisor will actually run.

    Body: `{"command": "<text>", "base": {<the preset being edited>}}`.
    Returns the resulting config plus the field-by-field diff, so the editor
    can show what applying it would change before anything is touched.
    """
    command = str(body.get("command") or "")
    base = _build_config({"name": "draft", **(body.get("base") or {})})
    try:
        cfg, warnings = merge_command(base, command, name=base.name)
        argv = command_argv(command, _binary())
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"the command does not parse as a shell command line: {e}",
        )
    return {
        "config": asdict(cfg),
        "diff": config_diff(base, cfg),
        "warnings": warnings,
        "unknown_flags": await _unknown_flags(argv),
        "missing_values": await _missing_flag_values(argv),
        "shadowed": shadowed_flags(argv),
        "conflicts": flag_conflicts(argv),
    }


@router.get("")
async def list_presets() -> list[dict]:
    return [asdict(cfg) for cfg in _registry().list()]


@router.get("/{name}")
async def get_preset(name: str) -> dict:
    try:
        return asdict(_registry().get(name))
    except PresetError as e:
        raise HTTPException(status_code=404, detail=str(e))


async def _reject_unknown_devices(cfg: LlamaServerConfig) -> None:
    """Refuse a GPU pin the configured binary cannot honour.

    Save time is the only gate that covers every path. A router preset's model
    is loaded from the generated INI and never goes through the supervisor's
    start check, so validating there alone would let an impossible selection
    sit in the file and be silently ignored at load — which is exactly how a
    preset pinned to Vulkan1 ended up running on the 5090.
    """
    binary = load_settings().llama_bin or ""
    wanted = getattr(cfg, "devices", None) or []
    unknown = await unknown_device_ids(binary, wanted)
    if unknown:
        available = ", ".join(d.id for d in await probe_devices(binary)) or "none"
        raise HTTPException(
            status_code=400,
            detail=(
                f"device(s) {', '.join(unknown)} are not available in this "
                f"llama-server build (it exposes {available}). Point Settings → "
                f"llama_bin at a build with the matching backend, or pick a "
                f"different GPU."
            ),
        )
    _reject_mixed_cpu_pin(wanted)
    await _reject_aliased_devices(binary, wanted)


def _reject_mixed_cpu_pin(wanted: list[str]) -> None:
    """Refuse "CPU *and* a GPU" — llama.cpp has no such thing.

    `-dev` takes one list, and `none` in it means "offload nothing"; there is
    no reading of `-dev none,CUDA0` that does half of each. The editor makes
    the CPU row exclusive, so this only catches a hand-edited preset or a
    pasted command line — but silently honouring one half of the request is
    how a preset ends up running somewhere the user never chose.
    """
    picked = [d for d in (wanted or []) if d]
    if CPU_DEVICE_ID in picked and len(picked) > 1:
        others = ", ".join(d for d in picked if d != CPU_DEVICE_ID)
        raise HTTPException(
            status_code=400,
            detail=(
                f"`-dev {CPU_DEVICE_ID}` means offload nothing, so it cannot be "
                f"combined with {others}. Pick the CPU on its own to keep the "
                f"whole model in RAM, or drop it and choose the GPUs to use."
            ),
        )


async def _reject_aliased_devices(binary: str, wanted: list[str]) -> None:
    """Refuse a pin that names the SAME physical card under two backends.

    A CUDA+Vulkan build lists one RTX 5090 as both `CUDA0` and `Vulkan2`.
    Selecting both passes the existence check — each id is real — and then
    plans two cards' worth of VRAM onto one 32 GB card, which loads happily
    until it aborts out of memory halfway through.
    """
    if len(wanted) < 2:
        return
    picked = {d.id: d for d in await probe_devices(binary) if d.id in wanted}
    for dev in picked.values():
        twin = dev.duplicate_of
        if twin and twin in picked:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{dev.id} and {twin} are the same physical card "
                    f"({dev.name}) seen through two backends. Selecting both "
                    f"would budget its memory twice — pick one (prefer "
                    f"{twin}, the vendor-native backend)."
                ),
            )


def _reject_bad_speculation(cfg: LlamaServerConfig) -> None:
    """A draft floor above the ceiling silently disables speculative decoding.

    llama.cpp discards any draft shorter than `n_min` (common/speculative.cpp),
    so `draft_min > draft_max` throws *every* draft away. The result is worse
    than turning speculation off — the drafting cost is still paid — while the
    UI happily reports it enabled. Measured here on qwen3.8-27b-vision:
    n_min=70 against n_max=3 gave 17.9 tok/s and zero accepted drafts; n_min=0
    gave 32.2 tok/s at 33% acceptance.

    Both values on this box came from confusing `--spec-draft-n-min` (a token
    count) with `--spec-draft-p-min` (a 0..1 probability floor), which is why a
    fractional count is rejected too rather than silently truncated to 0.
    """
    lo, hi = cfg.draft_min, cfg.draft_max
    if lo is None:
        return
    if isinstance(lo, float) and lo != int(lo):
        raise HTTPException(
            status_code=400,
            detail=(
                f"draft_min must be a whole number of tokens (got {lo}). "
                f"For a probability floor use --spec-draft-p-min in extra_flags."
            ),
        )
    if hi is not None and lo > hi:
        raise HTTPException(
            status_code=400,
            detail=(
                f"draft_min ({lo}) is above draft_max ({hi}) — llama.cpp drops "
                f"every draft shorter than draft_min, which disables "
                f"speculative decoding and makes generation slower than with it "
                f"off. Use 0 unless you have a reason not to."
            ),
        )


@router.put("/{name}")
async def put_preset(name: str, body: dict) -> dict:
    cfg = _build_config(body, name_override=name)
    _reject_bad_env(cfg)
    _reject_bad_speculation(cfg)
    await _reject_unknown_devices(cfg)
    try:
        out = asdict(_registry().upsert(cfg))
        _refresh_router_ini()
        return out
    except PresetError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("")
async def create_preset(body: dict) -> dict:
    cfg = _build_config(body)
    _reject_bad_env(cfg)
    _reject_bad_speculation(cfg)
    await _reject_unknown_devices(cfg)
    try:
        out = asdict(_registry().upsert(cfg))
        _refresh_router_ini()
        return out
    except PresetError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{name}")
async def delete_preset(name: str) -> dict:
    try:
        _registry().delete(name)
        _refresh_router_ini()
        return {"deleted": name}
    except PresetError as e:
        raise HTTPException(status_code=404, detail=str(e))
