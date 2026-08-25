"""Generate a `--models-preset` INI from sibling single-mode presets.

llama-server's router resolves model ids by basename (relative to --models-dir
or the cache). For our setup the model id is the GGUF basename minus '.gguf'.
We emit one `[<basename>]` section per single-mode preset whose model_path lives
under the router's models_dir, mapping LlamaDeck fields to llama-server CLI keys.
"""
from __future__ import annotations

from pathlib import Path

from .argv import mmproj_backend_env
from .models import scan_roots
from .presets import PresetRegistry
from .rpc_server import needs_rpc_for, rpc_flag_value
from .settings import LlamaServerConfig, atomic_write_text

# (cfg attr, INI key). Per-model SAMPLING is intentionally skipped — defaults
# come from GGUF / family — with ONE opt-in exception: presence_penalty. It
# defaults to None, so _emit_ini_field skips it for every preset that leaves it
# unset (i.e. all of them by default → behaviour unchanged). A preset that sets
# it explicitly (e.g. a vision/OCR model at 0.0) gets it pinned per-model in the
# router section, overriding the family default (Qwen3.6 → 1.5) for that model
# alone — without touching the text model that shares the same router.
_INI_FIELDS: list[tuple[str, str]] = [
    ("ctx_size", "ctx-size"),
    ("n_gpu_layers", "n-gpu-layers"),
    ("parallel", "parallel"),
    ("batch_size", "batch-size"),
    ("ubatch_size", "ubatch-size"),
    ("threads", "threads"),
    ("flash_attn", "flash-attn"),
    ("cache_type_k", "cache-type-k"),
    ("cache_type_v", "cache-type-v"),
    # --cache-reuse N. None means "llama.cpp's own default", which
    # _emit_ini_field already drops, so a preset that never sets it renders
    # byte-identical to before.
    ("cache_reuse", "cache-reuse"),
    ("presence_penalty", "presence-penalty"),
    # Thinking on/off, same opt-in shape as presence_penalty: the default
    # "auto" emits nothing, so a router whose presets never touch it renders
    # byte-identical to before. A dual-mode model (Qwen3.x, DeepSeek-R1) can
    # otherwise only be pinned to one mode for the whole router process.
    ("reasoning", "reasoning"),
    # Speculative decoding (only emitted when the preset actually enables it):
    # spec_type="none" stays out so the router section doesn't carry a no-op.
    ("spec_type", "spec-type"),
    ("draft_max", "spec-draft-n-max"),
    ("draft_min", "spec-draft-n-min"),
    # `-ts`. A scalar string, so the generic emitter handles it; `devices` is a
    # list and is emitted separately below.
    ("tensor_split", "tensor-split"),
]


# Tri-state toggles (True / False / None). They cannot ride _INI_FIELDS: that
# path renders the value with str(), and Python's "True"/"False" are neither
# truthy nor falsey to llama.cpp — common_arg_utils::is_truthy/is_falsey match
# "true"/"false"/"on"/"off"/"1"/"0" exactly, case-sensitively. A capitalised
# "False" would sail past is_falsey(), so to_args() would keep the POSITIVE
# flag and apply_to_params() would then read it back as false — the setting
# inverted between the child's command line and the router's own params.
_INI_TOGGLES: list[tuple[str, str]] = [
    ("cache_idle_slots", "cache-idle-slots"),
    ("context_shift", "context-shift"),
    ("kv_offload", "kv-offload"),
]


def _emit_toggles(cfg, out: list[str]) -> None:
    """Append `key = true|false` for every toggle the preset actually sets.

    llama.cpp maps the INI key back onto the option and picks the negative
    spelling itself (`context-shift = false` -> `--no-context-shift`), so we
    only ever write the positive key.
    """
    for attr, key in _INI_TOGGLES:
        val = getattr(cfg, attr, None)
        if val is None:
            continue
        out.append(f"{key} = {'true' if val else 'false'}")


def _emit_ini_field(attr: str, val) -> bool:
    """Whether (attr, val) should produce an INI line."""
    if val is None or val == -1:
        return False
    if attr == "spec_type" and val in ("", "none"):
        return False
    if attr == "reasoning" and val in ("", "auto"):
        return False
    return True


def _model_id(model_path: str, models_dir: Path | None = None) -> str:
    """The id llama-server assigns when scanning --models-dir.

    Empirically (cross-checked against /api/router/models on a running
    router): if the GGUF's parent directory is a *direct* child of
    models_dir, llama-server uses that directory's name as the id (e.g.
    `/models/Gemma/foo.gguf` → id `Gemma`). Otherwise it falls back to
    the file stem, except mmproj/-GGUF layouts collapse to the parent
    directory name so the projector + weights share one id."""
    p = Path(model_path)
    if models_dir is not None:
        try:
            if p.parent.parent.resolve() == Path(models_dir).resolve():
                return p.parent.name
        except (OSError, ValueError):
            pass
    if p.parent.name and (p.parent.name.endswith("-GGUF") or any(
        sib.name.startswith("mmproj") for sib in p.parent.glob("mmproj*")
    )):
        return p.parent.name
    return p.stem


def _serves(cfg: LlamaServerConfig, md: Path) -> bool:
    """Whether the router built from `md` serves this preset's model.

    Same test render_ini applies when deciding which presets get a section:
    single-mode, has a model, and that model lives under models_dir.
    """
    if getattr(cfg, "mode", "single") != "single":
        return False
    if not cfg.model_path:
        return False
    try:
        Path(cfg.model_path).resolve().relative_to(md)
    except (ValueError, OSError):
        return False
    return True


def router_env(
    models_dir: str,
    presets: list[LlamaServerConfig] | None = None,
    router_preset: LlamaServerConfig | None = None,
) -> tuple[dict[str, str], list[str]]:
    """The environment a router process needs: every served preset's env, merged.

    A preset's `env` reaches its model by being in that model's *process*
    environment. Router models are loaded inside the router's process, so the
    router is the only place these can take effect. Rendering the INI without
    merging them is how `GGML_CUDA_DISABLE_GRAPHS=1` — set on the 5090 preset
    precisely because CUDA graphs lock that GPU up — stopped applying the day
    the model moved from its own port to the router, and the card answered
    with five Xid 8 lockups in ninety minutes.

    Process-wide means shared: two presets that disagree on a key cannot both
    win. The router preset's own value wins outright; otherwise the first
    preset by name takes the key, and the loser is named in the returned
    warnings for the caller to log.

    A served vision preset also contributes its *derived* MTMD_BACKEND_DEVICE
    (see argv.mmproj_backend_env): the encoder ignores the INI's `device =`
    line exactly like it ignores -dev, and the router loads its models inside
    its own process, so this is the only place that pin can land.
    """
    if presets is None:
        presets = PresetRegistry().list()
    md = Path(models_dir).resolve()

    merged: dict[str, str] = {}
    owner: dict[str, str] = {}
    warnings: list[str] = []
    for cfg in sorted(presets, key=lambda c: c.name):
        if not _serves(cfg, md):
            continue
        contribution = {
            **mmproj_backend_env(cfg),
            **{str(k): str(v) for k, v in (getattr(cfg, "env", None) or {}).items()},
        }
        for k, v in contribution.items():
            k, v = str(k), str(v)
            if k in merged and merged[k] != v:
                warnings.append(
                    f"{k} is set to '{merged[k]}' by preset '{owner[k]}' and to "
                    f"'{v}' by preset '{cfg.name}'; the router process can only "
                    f"have one, keeping '{merged[k]}'"
                )
                continue
            merged.setdefault(k, v)
            owner.setdefault(k, cfg.name)

    for k, v in (getattr(router_preset, "env", None) or {}).items():
        k, v = str(k), str(v)
        if k in merged and merged[k] != v:
            warnings.append(
                f"{k} is set to '{merged[k]}' by preset '{owner[k]}'; the router "
                f"preset's own '{v}' wins"
            )
        merged[k] = v

    return merged, warnings


def render_ini(
    models_dir: str,
    presets: list[LlamaServerConfig] | None = None,
    router_preset: LlamaServerConfig | None = None,
) -> str:
    """Render an INI string suitable for --models-preset.

    Layout:
      [*]                      ← global defaults from router_preset (ctx, ngl,
                                 parallel, batch, fa, ckv, jinja). Inherited by
                                 every loaded model unless overridden below.
      [<model_id>]             ← per-model overrides from sibling single-mode
                                 presets whose model_path lives in models_dir.

    Why router_preset feeds [*] (not CLI): llama-server precedence is
    CLI > per-model section > [*]. If we passed ctx/ngl/parallel via CLI on the
    router, those would silently override per-model INI sections — defeating
    the whole point. Putting them in [*] lets per-model sections win.
    """
    if presets is None:
        presets = PresetRegistry().list()

    md = Path(models_dir).resolve()
    out: list[str] = ["version = 1", ""]

    # [*] global defaults from the router preset (or first router preset found)
    if router_preset is None:
        for p in presets:
            if getattr(p, "mode", "single") == "router":
                router_preset = p
                break
    if router_preset is not None:
        out.append(f"; global defaults from router preset '{router_preset.name}'")
        out.append("[*]")
        for attr, key in _INI_FIELDS:
            val = getattr(router_preset, attr, None)
            if _emit_ini_field(attr, val):
                out.append(f"{key} = {val}")
        router_devices = [d for d in (getattr(router_preset, "devices", None) or []) if d]
        if router_devices:
            out.append(f"device = {','.join(router_devices)}")
        if router_preset.jinja:
            out.append("jinja = true")
        if not router_preset.cont_batching:
            out.append("cont-batching = false")
        _emit_toggles(router_preset, out)
        out.append("")

    emitted: set[str] = set()

    for cfg in presets:
        if not _serves(cfg, md):
            continue
        mp = Path(cfg.model_path).resolve()

        # If the basename-derived id collides with an already-emitted section
        # (two presets pointing at the same GGUF — e.g. one for batch with
        # parallel=8, one for chat with parallel=1+MTP), fall back to the preset
        # name. Result: first preset gets the canonical id, later ones get a
        # preset-name id so clients can address both via `model=` in requests.
        section_id = _model_id(cfg.model_path, models_dir=md)
        if section_id in emitted:
            section_id = cfg.name
        out.append(f"; from preset '{cfg.name}'")
        # A raw command overrides the *process* llama-server starts. Router
        # models are loaded by the router's own process from this file, so the
        # override cannot apply here — say so in the file rather than letting
        # the user wonder why their command had no effect on the router.
        if getattr(cfg, "env", None):
            out.append(
                "; NOTE: this preset sets environment variables ("
                + ", ".join(sorted(cfg.env))
                + "). The router loads models inside its OWN process, so these "
                "are applied to the whole router — every model it serves sees "
                "them, not just this one. See router_env()."
            )
        if (getattr(cfg, "argv_override", None) or "").strip():
            out.append(
                "; NOTE: this preset has a raw command override; it applies only "
                "when the preset runs on its own port, not through the router. "
                "The keys below come from its fields."
            )
        out.append(f"[{section_id}]")
        out.append(f"model = {mp}")
        if cfg.mmproj_path:
            out.append(f"mmproj = {Path(cfg.mmproj_path).resolve()}")
        for attr, key in _INI_FIELDS:
            val = getattr(cfg, attr, None)
            if _emit_ini_field(attr, val):
                out.append(f"{key} = {val}")
        # Device pin. Not an _INI_FIELDS row because the value is a list here
        # and one comma-joined token to llama-server. Without this the router
        # path silently ignored a preset's GPU selection: the picker saved it,
        # to_argv emitted it, and the model still loaded wherever llama.cpp
        # felt like — because router models are loaded from this INI, not argv.
        devices = [d for d in (getattr(cfg, "devices", None) or []) if d]
        # Same rule as to_argv: an RPC id named anywhere — the pin or a raw
        # `-ot …=RPC0` — needs the endpoint list, or it resolves to nothing.
        if needs_rpc_for(cfg):
            endpoints = rpc_flag_value()
            if endpoints:
                out.append(f"rpc = {endpoints}")
        if devices:
            out.append(f"device = {','.join(devices)}")
        if cfg.jinja:
            out.append("jinja = true")
        if not cfg.cont_batching:
            out.append("cont-batching = false")
        _emit_toggles(cfg, out)
        for tok in _extra_pairs(cfg.extra_flags):
            out.append(tok)
        out.append("")
        emitted.add(section_id)

    # Drop-in fallback: any GGUF under models_dir without a preset still gets
    # a bare section so the user can dump a file in the folder and have the
    # router pick it up. [*] supplies sane defaults; per-model tuning is what
    # presets are for.
    try:
        registry_entries = scan_roots([str(md)])
    except Exception:
        registry_entries = []
    for entry in registry_entries:
        try:
            mp = Path(entry.path).resolve()
            mp.relative_to(md)
        except (ValueError, OSError):
            continue
        section_id = _model_id(entry.path, models_dir=md)
        if section_id in emitted:
            continue
        out.append("; from registry (no preset — inherits [*] defaults)")
        out.append(f"[{section_id}]")
        out.append(f"model = {mp}")
        if entry.mmproj_path:
            out.append(f"mmproj = {Path(entry.mmproj_path).resolve()}")
        out.append("")
        emitted.add(section_id)

    return "\n".join(out) + "\n"


def _extra_pairs(extra: list[str]) -> list[str]:
    """Best-effort: pair `--flag value` and `--bool` from extra_flags into
    INI lines. Skips spec-default and other server-only knobs that the router
    rejects (host/port/api-key/hf-repo/model-alias)."""
    SKIP = {
        "--host", "--port", "--api-key", "-hf", "--hf-repo", "-hff", "--hf-file",
        "--alias", "--model-alias", "--spec-default",
    }
    out: list[str] = []
    i = 0
    while i < len(extra):
        tok = extra[i]
        if tok in SKIP:
            # SKIP one-arg flags consume the next token; bool ones won't.
            i += 1
            if tok in {"--host", "--port", "--api-key", "-hf", "--hf-repo",
                       "-hff", "--hf-file", "--alias", "--model-alias"}:
                if i < len(extra) and not extra[i].startswith("-"):
                    i += 1
            continue
        if not tok.startswith("-"):
            i += 1
            continue
        key = tok.lstrip("-")
        if i + 1 < len(extra) and not extra[i + 1].startswith("-"):
            out.append(f"{key} = {extra[i + 1]}")
            i += 2
        else:
            out.append(f"{key} = true")
            i += 1
    return out


def write_ini(
    path: Path,
    models_dir: str,
    router_preset: LlamaServerConfig | None = None,
) -> str:
    text = render_ini(models_dir, router_preset=router_preset)
    atomic_write_text(path, text)
    return text
