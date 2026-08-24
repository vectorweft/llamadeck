"""Translate LlamaServerConfig <-> argv list for llama-server.

Only flags covered by LlamaServerConfig fields are round-tripped natively.
Anything unknown during `from_argv` goes into `extra_flags` and is re-emitted
verbatim by `to_argv`.

Three layers decide what actually gets executed, last one winning:

  1. the preset's fields              -> `to_argv`
  2. `extra_flags`, appended verbatim -> `to_argv`
  3. `argv_override`, a raw command   -> `command_argv`, and nothing else runs

Layer 3 is the pro escape hatch: whatever the user typed in the command box is
the process, verbatim. `merge_command` goes the other way — it parses a typed
command back into fields so the form and the command stay one thing rather
than two that can disagree.
"""
from __future__ import annotations

import re
import shlex

from .rpc_server import needs_rpc_for, rpc_flag_value
from .settings import LlamaServerConfig

# Maps dataclass field -> (flag_name_or_names, kind)
# kind: 'value' (flag then value), 'bool_if_true' (emit only when True),
#       'bool_no_prefix' (emit --no-<name> when False, otherwise nothing),
#       'str_value' (flag then value, but only if non-empty/non-None)
_FIELD_FLAGS: list[tuple[str, str, str]] = [
    ("model_path", "--model", "str_value"),
    ("hf_repo", "-hf", "str_value"),
    ("hf_file", "-hff", "str_value"),
    ("mmproj_path", "--mmproj", "str_value"),
    ("host", "--host", "value"),
    ("port", "--port", "value"),
    ("api_key", "--api-key", "str_value"),
    ("ctx_size", "--ctx-size", "value"),
    ("n_gpu_layers", "--n-gpu-layers", "value"),
    ("parallel", "--parallel", "value"),
    ("batch_size", "--batch-size", "value"),
    ("ubatch_size", "--ubatch-size", "value"),
    ("threads", "--threads", "value"),
    ("flash_attn", "--flash-attn", "value"),
    ("cache_type_k", "--cache-type-k", "value"),
    ("cache_type_v", "--cache-type-v", "value"),
    ("cache_reuse", "--cache-reuse", "value"),
    ("temperature", "--temp", "value"),
    ("top_k", "--top-k", "value"),
    ("top_p", "--top-p", "value"),
    ("min_p", "--min-p", "value"),
    ("repeat_penalty", "--repeat-penalty", "value"),
    # presence_penalty defaults to None → "value" kind skips it (None), so it's
    # emitted only when a preset sets it explicitly. Lets a vision/OCR preset
    # pin --presence-penalty 0 without touching any other preset's defaults.
    ("presence_penalty", "--presence-penalty", "value"),
    # Thinking / reasoning. "auto" means "whatever the chat template decides",
    # which is llama-server's own default, so it emits nothing — a preset that
    # never touches the setting produces the exact command it did before.
    ("reasoning", "--reasoning", "tri_state"),
    ("spec_type", "--spec-type", "enum_value"),
    ("model_path_draft", "--model-draft", "str_value"),
    ("n_gpu_layers_draft", "-ngld", "value"),
    # Upstream renamed --draft-max → --spec-draft-n-max (and --draft-min → --spec-draft-n-min)
    # somewhere around the MTP merge; the old names are now hard errors, not aliases.
    ("draft_max", "--spec-draft-n-max", "value"),
    ("draft_min", "--spec-draft-n-min", "value"),
]


def to_argv(cfg: LlamaServerConfig, binary: str) -> list[str]:
    # A raw command wins over every field, in every mode. This is the whole
    # promise of the command box: what you read there is what gets executed.
    override = (getattr(cfg, "argv_override", None) or "").strip()
    if override:
        return command_argv(override, binary)
    if getattr(cfg, "mode", "single") == "router":
        return _router_argv(cfg, binary)

    argv: list[str] = [binary]
    spec_active = getattr(cfg, "spec_type", "none") not in (None, "", "none")
    drafter_set = bool(getattr(cfg, "model_path_draft", None)) or spec_active
    # n_gpu_layers_draft only matters when a separate drafter GGUF is loaded;
    # MTP heads live inside the main model, so -ngld has no effect for draft-mtp.
    drafter_model_helpers = {"n_gpu_layers_draft"}
    drafter_helpers = {"draft_max", "draft_min"}
    for field, flag, kind in _FIELD_FLAGS:
        val = getattr(cfg, field)
        if field in drafter_model_helpers and not bool(getattr(cfg, "model_path_draft", None)):
            continue
        if field in drafter_helpers and not drafter_set:
            continue
        if kind == "str_value":
            if val:
                argv.extend([flag, str(val)])
        elif kind == "value":
            if val is not None and val != -1:
                argv.extend([flag, str(val)])
        elif kind == "tri_state":
            # on|off are explicit; auto (and empty) defer to the model.
            if val in ("on", "off"):
                argv.extend([flag, str(val)])
        elif kind == "enum_value":
            if val and val != "none":
                # Legacy name "draft-model" → the "draft-simple" that
                # llama-server expects (sent raw it fails at startup with
                # "unknown speculative type"). Old persisted presets get
                # fixed up this way.
                if field == "spec_type" and val == "draft-model":
                    val = "draft-simple"
                argv.extend([flag, str(val)])

    # Device pinning. Emitted from a list rather than a _FIELD_FLAGS row
    # because llama-server takes one comma-joined value, not repeated flags.
    devices = [d for d in (getattr(cfg, "devices", None) or []) if d]
    # An RPC id resolves to nothing unless --rpc names the endpoints, and it
    # must come first so the device is registered before -dev (or a trailing
    # `-ot …=RPC0` in extra_flags) is parsed.
    if needs_rpc_for(cfg):
        endpoints = rpc_flag_value()
        if endpoints:
            argv.extend(["--rpc", endpoints])
    if devices:
        argv.extend(["-dev", ",".join(devices)])
    tensor_split = getattr(cfg, "tensor_split", None)
    if tensor_split:
        argv.extend(["-ts", str(tensor_split)])

    # Boolean flags
    if not cfg.cont_batching:
        argv.append("--no-cont-batching")
    if cfg.jinja:
        argv.append("--jinja")
    if cfg.metrics:
        argv.append("--metrics")
    if cfg.slots:
        argv.append("--slots")

    # Prompt-processing / KV-cache reuse toggles: True → positive flag,
    # False → --no-<flag>, None → leave llama.cpp's default (emit nothing).
    # `cache_reuse` is a value flag (`--cache-reuse N`) and comes from
    # _FIELD_FLAGS above.
    for _field, _pos, _neg in (
        ("cache_idle_slots", "--cache-idle-slots", "--no-cache-idle-slots"),
        ("context_shift", "--context-shift", "--no-context-shift"),
        ("kv_offload", "--kv-offload", "--no-kv-offload"),
    ):
        _v = getattr(cfg, _field)
        if _v is True:
            argv.append(_pos)
        elif _v is False:
            argv.append(_neg)

    argv.extend(cfg.extra_flags)
    return argv


# Router-mode CLI: ONLY bind + router-control flags. ctx/ngl/parallel/batch/...
# would otherwise win over per-model INI overrides (llama-server precedence:
# CLI > per-model section > [*] global), defeating the point of the INI. Those
# defaults are emitted into the INI's [*] global section instead — see
# router_ini.render_ini(router_preset=...).
def _router_argv(cfg: LlamaServerConfig, binary: str) -> list[str]:
    argv: list[str] = [binary]

    # Bind
    if cfg.host:
        argv.extend(["--host", cfg.host])
    if cfg.port:
        argv.extend(["--port", str(cfg.port)])
    if cfg.api_key:
        argv.extend(["--api-key", cfg.api_key])

    # Router-specific
    if cfg.models_dir:
        argv.extend(["--models-dir", cfg.models_dir])
    if cfg.models_preset_path:
        argv.extend(["--models-preset", cfg.models_preset_path])
    if cfg.models_max is not None:
        argv.extend(["--models-max", str(cfg.models_max)])
    if cfg.models_autoload is False:
        argv.append("--no-models-autoload")
    if cfg.sleep_idle_seconds is not None:
        argv.extend(["--sleep-idle-seconds", str(cfg.sleep_idle_seconds)])

    # Process-level only (per-model loaded instances don't expose these).
    if cfg.metrics:
        argv.append("--metrics")
    if cfg.slots:
        argv.append("--slots")

    # extra_flags pass-through, but only if the user really meant them at
    # router level (e.g. --log-verbose). Per-model overrides should live in
    # the preset (single-mode) and propagate to INI per-section.
    argv.extend(cfg.extra_flags)
    return argv


# Flag aliases llama-server accepts. Key = canonical long name used in _FIELD_FLAGS.
_FLAG_ALIASES = {
    "--model": {"-m"},
    "--mmproj": set(),
    "-hf": {"--hf-repo"},
    "-hff": {"--hf-file"},
    "--host": set(),
    "--port": set(),
    "--api-key": set(),
    "--ctx-size": {"-c"},
    "--n-gpu-layers": {"-ngl", "--gpu-layers"},
    "--parallel": {"-np"},
    "--batch-size": {"-b"},
    "--ubatch-size": {"-ub"},
    "--threads": {"-t"},
    "--flash-attn": {"-fa"},
    "--cache-type-k": {"-ctk"},
    "--cache-type-v": {"-ctv"},
    "--cache-reuse": set(),
    "--temp": {"--temperature"},
    "--top-k": set(),
    "--top-p": set(),
    "--min-p": set(),
    "--repeat-penalty": set(),
    "--presence-penalty": set(),
    "--model-draft": {"-md"},
    "-ngld": {"--gpu-layers-draft", "--n-gpu-layers-draft"},
    # Canonical = the current upstream name. Old names kept as aliases so we can
    # still parse a legacy adopted process that was started with the removed flags.
    "--spec-draft-n-max": {"--draft", "--draft-n", "--draft-max"},
    "--spec-draft-n-min": {"--draft-min", "--draft-n-min"},
    "--spec-type": set(),
    # Not _FIELD_FLAGS rows — nothing emits them from a field — but they must
    # canonicalize for the conflict rules (and for shadowed-flag detection) to
    # see `-sm none` and `-ot ...` as the flags they are.
    "--split-mode": {"-sm"},
    "--override-tensor": {"-ot"},
    "--main-gpu": {"-mg"},
    "--reasoning": {"-rea"},
}


def _canonical(flag: str) -> str | None:
    for canon, aliases in _FLAG_ALIASES.items():
        if flag == canon or flag in aliases:
            return canon
    return None


_FIELD_BY_CANON = {flag: (field, kind) for field, flag, kind in _FIELD_FLAGS}

# Router-mode flags, kept out of _FIELD_FLAGS because `to_argv` must NOT emit
# them in single mode (see _router_argv for why the two argv shapes differ).
_ROUTER_VALUE_FLAGS: dict[str, tuple[str, type]] = {
    "--models-dir": ("models_dir", str),
    "--models-preset": ("models_preset_path", str),
    "--models-max": ("models_max", int),
    "--sleep-idle-seconds": ("sleep_idle_seconds", int),
}


def from_argv(cmdline: list[str], name: str = "adopted") -> LlamaServerConfig:
    """Parse a llama-server argv list back into a LlamaServerConfig. First element
    is the binary. Unknown flags are preserved in extra_flags verbatim."""
    cfg = LlamaServerConfig(name=name)
    # `--jinja` / `--metrics` / `--slots` are emitted only when true, so their
    # ABSENCE from a command line means off. Leaving the dataclass defaults
    # (metrics/slots start on) would make parsing lie: delete --metrics in the
    # command box, apply, and it reappears in the very next render.
    cfg.jinja = False
    cfg.metrics = False
    cfg.slots = False
    extra: list[str] = []
    i = 1  # skip binary
    while i < len(cmdline):
        tok = cmdline[i]
        canon = _canonical(tok)
        if canon and canon in _FIELD_BY_CANON:
            field, kind = _FIELD_BY_CANON[canon]
            if kind == "tri_state":
                # `--reasoning` takes an OPTIONAL value: "--reasoning off" and
                # a bare "--reasoning" (meaning on) are both legal.
                nxt = cmdline[i + 1] if i + 1 < len(cmdline) else None
                if nxt in ("on", "off", "auto"):
                    setattr(cfg, field, nxt)
                    i += 2
                else:
                    setattr(cfg, field, "on")
                    i += 1
                continue
            if kind in ("value", "str_value", "enum_value"):
                if i + 1 >= len(cmdline):
                    extra.append(tok)
                    i += 1
                    continue
                raw = cmdline[i + 1]
                _assign(cfg, field, raw)
                i += 2
                continue
        # -dev / -ts: value-taking, but not _FIELD_FLAGS rows (see to_argv).
        # Parsed here so adopting a running process — or a preset that carried
        # these in extra_flags before the fields existed — fills the fields
        # instead of dumping them back into extra_flags.
        # --rpc is derived from the RPC server settings at launch, so parsing
        # it back into extra_flags would duplicate it on the next start.
        if tok == "--rpc" and i + 1 < len(cmdline):
            i += 2
            continue
        if tok in ("-dev", "--device") and i + 1 < len(cmdline):
            cfg.devices = [p for p in cmdline[i + 1].split(",") if p]
            i += 2
            continue
        if tok in ("-ts", "--tensor-split") and i + 1 < len(cmdline):
            cfg.tensor_split = cmdline[i + 1]
            i += 2
            continue

        # Router-mode flags. Without these a router command pasted into the
        # command box would land entirely in extra_flags and come back out as
        # a single-mode preset with a dead models_dir.
        if tok in _ROUTER_VALUE_FLAGS and i + 1 < len(cmdline):
            field, caster = _ROUTER_VALUE_FLAGS[tok]
            try:
                setattr(cfg, field, caster(cmdline[i + 1]))
            except ValueError:
                extra.extend(cmdline[i:i + 2])
            cfg.mode = "router"
            i += 2
            continue
        if tok == "--no-models-autoload":
            cfg.models_autoload = False
            cfg.mode = "router"
            i += 1
            continue
        if tok == "--models-autoload":
            cfg.models_autoload = True
            cfg.mode = "router"
            i += 1
            continue

        # Boolean / special flags
        if tok in ("-cb", "--cont-batching"):
            cfg.cont_batching = True
            i += 1
            continue
        if tok == "--no-cont-batching":
            cfg.cont_batching = False
            i += 1
            continue
        if tok == "--jinja":
            cfg.jinja = True
            i += 1
            continue
        if tok == "--metrics":
            cfg.metrics = True
            i += 1
            continue
        if tok == "--slots":
            cfg.slots = True
            i += 1
            continue
        # Prompt-processing / KV-cache reuse toggles (see to_argv's emission).
        _toggle_seen = False
        for _field, _pos, _neg in (
            ("cache_idle_slots", "--cache-idle-slots", "--no-cache-idle-slots"),
            ("context_shift", "--context-shift", "--no-context-shift"),
            ("kv_offload", "--kv-offload", "--no-kv-offload"),
        ):
            if tok == _pos:
                setattr(cfg, _field, True)
                _toggle_seen = True
                break
            if tok == _neg:
                setattr(cfg, _field, False)
                _toggle_seen = True
                break
        if _toggle_seen:
            i += 1
            continue
        # Unknown — keep verbatim (plus its value if the next token doesn't look like a flag)
        extra.append(tok)
        i += 1
        if i < len(cmdline) and not cmdline[i].startswith("-"):
            extra.append(cmdline[i])
            i += 1
    cfg.extra_flags = extra
    return cfg


# Optional-typed fields default to None at runtime, so isinstance(current, ...)
# can't tell us their intended type. Drive coercion off the dataclass annotations.
_INT_FIELDS = {"draft_max", "draft_min", "cache_reuse"}
# presence_penalty defaults to None, so `_assign` cannot infer "float" from the
# current value and would store the raw string "1.5" — which then renders as a
# perfectly valid-looking flag and only fails on the next numeric comparison.
_FLOAT_FIELDS: set[str] = {"presence_penalty"}


def _assign(cfg: LlamaServerConfig, field: str, raw: str) -> None:
    current = getattr(cfg, field)
    if isinstance(current, bool):
        setattr(cfg, field, raw.lower() in ("1", "true", "yes", "on"))
        return
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            setattr(cfg, field, int(raw))
        except ValueError:
            setattr(cfg, field, current)
        return
    if isinstance(current, float):
        try:
            setattr(cfg, field, float(raw))
        except ValueError:
            setattr(cfg, field, current)
        return
    if field in _INT_FIELDS:
        try:
            setattr(cfg, field, int(raw))
        except ValueError:
            pass
        return
    if field in _FLOAT_FIELDS:
        try:
            setattr(cfg, field, float(raw))
        except ValueError:
            pass
        return
    setattr(cfg, field, raw)


# --- command box: one raw command line, authoritative ------------------------
# Fields LlamaDeck owns rather than llama-server. They have no flag, so a
# command line cannot carry them and a parse must not invent values for them.
_METADATA_FIELDS = ("name", "notes", "ui_hidden", "estimated_vram_mb", "argv_override")


def split_command(text: str) -> list[str]:
    """Tokenize a shell-style command line.

    Handles quoting, `#` comments and trailing-backslash continuations, so a
    command pasted straight out of a terminal — or out of `to_command`'s
    wrapped output — parses. Raises ValueError on an unbalanced quote; the
    caller turns that into a 400 rather than launching something truncated.
    """
    return shlex.split(text.replace("\\\n", " "), comments=True)


_ENV_ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.DOTALL)


def split_env_prefix(tokens: list[str]) -> tuple[dict[str, str], list[str]]:
    """Peel `KEY=VALUE` assignments off the front, exactly as a shell does.

    `FOO=1 llama-server --model x` sets FOO for the process; a `FOO=1` further
    along is an ordinary argument (`--api-key k=v` must survive untouched), so
    only the leading run is taken.
    """
    env: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        m = _ENV_ASSIGN_RE.match(tokens[i])
        if not m:
            break
        env[m.group(1)] = m.group(2)
        i += 1
    return env, tokens[i:]


def looks_like_binary(token: str) -> bool:
    """Whether a leading token is a program path rather than a flag."""
    return bool(token) and not token.startswith("-")


def command_argv(text: str, binary: str) -> list[str]:
    """argv for a raw command, with the program forced to `binary`.

    The user's own first token is deliberately dropped: presets must keep
    following Settings -> llama_bin (a rebuild moves it, and a stale absolute
    path in a saved command would then start yesterday's binary). Everything
    after it is passed through verbatim — that is the point of the override.
    """
    _env, tokens = split_env_prefix(split_command(text))
    if not tokens:
        return [binary]
    if looks_like_binary(tokens[0]):
        return [binary, *tokens[1:]]
    return [binary, *tokens]


def command_env(text: str) -> dict[str, str]:
    """The `KEY=VALUE` prefix of a raw command, as an environment mapping."""
    env, _rest = split_env_prefix(split_command(text))
    return env


# llama.cpp's multimodal projector ignores `--device`. tools/mtmd/clip.cpp
# picks the encoder's GPU from the MTMD_BACKEND_DEVICE env var, defaulting to
# the *first* GPU backend in ggml's registry — on a CUDA+Vulkan build that is
# CUDA0. So a vision preset pinned to `Vulkan1` (the R9700) still drops its
# vision encoder on the NVIDIA card, and the 5090 OOMs the moment it is busy.
# There is no flag for this, only the env var — the same class of knob as
# GGML_CUDA_DISABLE_GRAPHS, which is why it is injected at spawn rather than
# rendered into argv.
_MTMD_BACKEND_DEVICE = "MTMD_BACKEND_DEVICE"


def mmproj_backend_env(cfg: LlamaServerConfig) -> dict[str, str]:
    """The MTMD_BACKEND_DEVICE a preset's process needs, or {} when it doesn't.

    Injected only when the preset loads a multimodal projector (`--mmproj`)
    AND pins at least one real device. The encoder is a single model, so it
    must live on exactly one device; the first entry of the pin is the one the
    user named first. A preset that leaves `devices` empty keeps llama.cpp's
    default ("let llama.cpp choose" stays untouched), and an explicit
    MTMD_BACKEND_DEVICE in the preset's own `env` wins — callers merge this
    mapping *under* the preset's env.
    """
    if not getattr(cfg, "mmproj_path", None):
        return {}
    devices = [d for d in (getattr(cfg, "devices", None) or []) if d]
    if not devices or "none" in devices:
        return {}
    return {_MTMD_BACKEND_DEVICE: devices[0]}


def _is_number(tok: str) -> bool:
    """`-1` is a value, not a flag — it belongs on its flag's line."""
    try:
        float(tok)
    except ValueError:
        return False
    return True


def to_command(cfg: LlamaServerConfig, binary: str, multiline: bool = True) -> str:
    """Render the exact command `to_argv` would execute, shell-quoted.

    A *rendering* of to_argv, never a second implementation of it: the command
    box stops being trustworthy the moment the two can disagree.
    """
    argv = to_argv(cfg, binary)
    quoted = [shlex.quote(tok) for tok in argv]
    # Environment first, the way it is typed at a shell prompt. An override
    # already carries its own assignments verbatim, so they are not added
    # twice.
    env_tokens: list[str] = []
    if not (getattr(cfg, "argv_override", None) or "").strip():
        env_tokens = [
            f"{k}={shlex.quote(v)}" for k, v in sorted((getattr(cfg, "env", None) or {}).items())
        ]
    if not multiline:
        return " ".join([*env_tokens, *quoted])
    lines: list[str] = [*env_tokens, quoted[0]]
    for tok in quoted[1:]:
        if tok.startswith("-") and not _is_number(tok):
            lines.append(tok)
        else:
            lines[-1] += " " + tok
    return " \\\n  ".join(lines)


def merge_command(
    base: LlamaServerConfig,
    text: str,
    name: str | None = None,
) -> tuple[LlamaServerConfig, list[str]]:
    """Parse a typed command back into a full config, plus advisory warnings.

    Everything llama-server can express comes from the command — including the
    ABSENCE of a flag, which is why `from_argv` resets the emit-when-true
    booleans. Fields llama-server knows nothing about (name, notes, ...) are
    carried over from `base` so applying a command never wipes the preset's
    identity.
    """
    env, tokens = split_env_prefix(split_command(text))
    argv = tokens if (tokens and looks_like_binary(tokens[0])) else ["llama-server", *tokens]
    cfg = from_argv(argv, name=name or base.name)
    for f in _METADATA_FIELDS:
        setattr(cfg, f, getattr(base, f))
    # Typed environment is part of the command, so it round-trips with it.
    cfg.env = env

    warnings: list[str] = []
    # A drafter GGUF with no --spec-type is the pre-spec_type spelling of
    # classic speculative decoding; keep the preset runnable instead of
    # dropping the drafter on the floor.
    if cfg.model_path_draft and cfg.spec_type in (None, "", "none"):
        cfg.spec_type = "draft-simple"
        warnings.append(
            "--model-draft without --spec-type: read as speculative decoding "
            "(spec_type=draft-simple)."
        )
    if cfg.mode == "router" and not cfg.models_dir:
        warnings.append(
            "router flags present but no --models-dir: llama-server exits "
            "immediately without one."
        )
    if not cfg.model_path and not cfg.hf_repo and cfg.mode != "router":
        warnings.append("no --model / -hf in the command: nothing to load.")
    return cfg, warnings


def config_diff(before: LlamaServerConfig, after: LlamaServerConfig) -> list[dict]:
    """Which fields a parsed command would change, for a confirm-before-apply
    view. Metadata fields are skipped — they are carried over by definition."""
    from dataclasses import fields as _fields

    out: list[dict] = []
    for f in _fields(LlamaServerConfig):
        if f.name in _METADATA_FIELDS:
            continue
        old, new = getattr(before, f.name), getattr(after, f.name)
        if old != new:
            out.append({"field": f.name, "from": old, "to": new})
    return out


def shadowed_flags(argv: list[str]) -> list[dict]:
    """Flags a command line passes more than once, with different values.

    llama.cpp keeps the LAST occurrence, so a preset that also carries
    `--temp 0.7` in extra_flags silently overrules its own temperature field:
    the editor shows 0.8 and the process samples at 0.7. That is how sampling
    hand-tuned on a terminal ends up invisible to the app that launched it.

    Reported, never auto-fixed — "the field is stale" and "the raw flag is the
    real intent" are both legitimate readings, and only the user knows which.
    Aliases are collapsed (`--temp` and `--temperature` are one flag), so a
    duplicate spelled two ways is still caught.
    """
    seen: dict[str, list[str | None]] = {}
    spellings: dict[str, list[str]] = {}
    i = 1
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("-") or tok == "-" or _is_number(tok):
            i += 1
            continue
        name, sep, inline = tok.partition("=")
        key = _canonical(name) or name
        value: str | None = inline if sep else None
        if value is None and i + 1 < len(argv):
            nxt = argv[i + 1]
            if not nxt.startswith("-") or _is_number(nxt):
                value = nxt
                i += 1
        seen.setdefault(key, []).append(value)
        if name not in spellings.setdefault(key, []):
            spellings[key].append(name)
        i += 1

    out: list[dict] = []
    for key, values in seen.items():
        if len(values) < 2:
            continue
        # Repeating a bare boolean, or the same value twice, changes nothing.
        if len({v for v in values if v is not None}) < 2:
            continue
        out.append({
            "flag": key,
            "spellings": spellings[key],
            "wins": values[-1],
            "shadowed": values[:-1],
        })
    return out


# Flag pairs that cancel each other out without llama-server saying a word.
# Data-driven so the next one found on a real box is one row, not a branch.
_CONFLICT_RULES: list[dict] = [
    {
        "id": "sm-none-vs-ot",
        "when": {"--split-mode": "none"},
        "with_any": ["--override-tensor"],
        "message": (
            "`--split-mode none` loads the model on ONE GPU, so the tensors "
            "`-ot` places on another device never get there — llama-server "
            "reports no error, the second card simply stays empty. Drop "
            "`-sm none` and steer the split with `-ts` instead (e.g. `-ts 1,0` "
            "keeps every non-overridden layer on the first device)."
        ),
    },
]


def flag_conflicts(argv: list[str]) -> list[dict]:
    """Flag combinations in `argv` that silently defeat each other.

    Separate from `shadowed_flags`: that one catches the same flag passed
    twice, this one catches two different flags whose meanings collide. Both
    are reported rather than fixed — the user may have meant either half.
    """
    present: dict[str, str | None] = {}
    i = 1
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("-") or tok == "-" or _is_number(tok):
            i += 1
            continue
        name, sep, inline = tok.partition("=")
        key = _canonical(name) or name
        value: str | None = inline if sep else None
        if value is None and i + 1 < len(argv):
            nxt = argv[i + 1]
            if not nxt.startswith("-") or _is_number(nxt):
                value = nxt
                i += 1
        present[key] = value
        i += 1

    out: list[dict] = []
    for rule in _CONFLICT_RULES:
        if not all(present.get(f) == v for f, v in rule["when"].items()):
            continue
        hit = [f for f in rule["with_any"] if f in present]
        if not hit:
            continue
        out.append({
            "id": rule["id"],
            "flags": [*rule["when"], *hit],
            "message": rule["message"],
        })
    return out
