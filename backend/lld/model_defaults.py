"""Read recommended sampling defaults + architecture metadata from a GGUF file,
with optional live merge from llama-server `/props` if the model is running.

Priority (best to worst):
  1. GGUF `general.sampling.*` keys (producer-embedded — Unsloth does this)
  2. llama-server `/props` `default_generation_settings` (live-computed)
  3. Curated family table below (known-good Qwen/Llama/Gemma/... recs)
  4. Empty (UI shows nothing)
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .gguf_header import read_kv

log = logging.getLogger(__name__)


# --- curated family-level recommendations (fallback when GGUF lacks sampling) ---
# Keys mirror what llama.cpp accepts. Values reflect the model family's public
# guidance (HF cards, Unsloth docs, first-party model cards).
_FAMILY_DEFAULTS: list[tuple[re.Pattern[str], dict[str, float]]] = [
    # Qwen3.6 — Unsloth model card "Best Practices" (instruct/non-thinking preset
    # is the mode-agnostic fallback used by callers who don't know the mode):
    # temp=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5,
    # repetition_penalty=1.0 (Unsloth note: presence_penalty between 0–2
    # reduces endless repetitions; higher may cause language mixing).
    (re.compile(r"Qwen3\.6",         re.I), {"temperature": 0.7, "top_k": 20, "top_p": 0.80, "min_p": 0.00, "repeat_penalty": 1.00, "presence_penalty": 1.50}),
    (re.compile(r"Qwen3\.5",         re.I), {"temperature": 0.6, "top_k": 20, "top_p": 0.80, "min_p": 0.00, "repeat_penalty": 1.05}),
    (re.compile(r"Qwen[23]\.?5",     re.I), {"temperature": 0.7, "top_k": 20, "top_p": 0.80, "min_p": 0.00, "repeat_penalty": 1.05}),
    (re.compile(r"QwQ",              re.I), {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "min_p": 0.00, "repeat_penalty": 1.05}),
    (re.compile(r"Qwen",             re.I), {"temperature": 0.7, "top_k": 20, "top_p": 0.80}),
    (re.compile(r"Llama-?3\.?[12]",  re.I), {"temperature": 0.6, "top_k": 40, "top_p": 0.90, "min_p": 0.00}),
    (re.compile(r"Llama",            re.I), {"temperature": 0.7, "top_k": 40, "top_p": 0.90}),
    (re.compile(r"Gemma-?4",         re.I), {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "min_p": 0.00}),
    (re.compile(r"Gemma-?3",         re.I), {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "min_p": 0.00}),
    (re.compile(r"Gemma",            re.I), {"temperature": 1.0, "top_k": 64, "top_p": 0.95}),
    (re.compile(r"Phi-?\d",          re.I), {"temperature": 0.7, "top_k": 40, "top_p": 0.95}),
    (re.compile(r"Mistral|Mixtral",  re.I), {"temperature": 0.7, "top_k": 40, "top_p": 0.90}),
    (re.compile(r"DeepSeek",         re.I), {"temperature": 0.6, "top_k": 40, "top_p": 0.95}),
]


# --- thinking-vs-non-thinking variants ---------------------------------------
# Some families (Qwen3.x, QwQ, DeepSeek-R1) publish DIFFERENT recommended
# sampling for thinking-mode vs non-thinking-mode. Where a family appears here,
# get_recommended_sampling() returns BOTH variants; otherwise both keys map to
# the same (mode-agnostic) recommendation. Source: Unsloth model cards + Qwen
# README (https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507).
_FAMILY_VARIANTS: list[tuple[re.Pattern[str], dict[str, dict[str, float]]]] = [
    # Qwen3 family — official "Best Practices" section
    # Qwen3.6: per Unsloth model card (https://huggingface.co/unsloth/Qwen3.6-27B-GGUF)
    #   thinking ON:  temp=1.0, top_p=0.95, top_k=20, min_p=0.0,
    #                 presence_penalty=0.0, repetition_penalty=1.0
    #   instruct OFF: temp=0.7, top_p=0.80, top_k=20, min_p=0.0,
    #                 presence_penalty=1.5, repetition_penalty=1.0
    # Verified against the dense Qwen3.6-27B-UD-Q4_K_XL card 2026-05-02.
    (re.compile(r"Qwen3\.6", re.I), {
        "thinking":     {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.00, "repeat_penalty": 1.00, "presence_penalty": 0.00},
        "non_thinking": {"temperature": 0.7, "top_k": 20, "top_p": 0.80, "min_p": 0.00, "repeat_penalty": 1.00, "presence_penalty": 1.50},
    }),
    (re.compile(r"Qwen3\.5", re.I), {
        "thinking":     {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "min_p": 0.00},
        "non_thinking": {"temperature": 0.7, "top_k": 20, "top_p": 0.80, "min_p": 0.00},
    }),
    # QwQ is thinking-only, but include both to keep the contract uniform.
    (re.compile(r"QwQ", re.I), {
        "thinking":     {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "min_p": 0.00},
        "non_thinking": {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "min_p": 0.00},
    }),
    # DeepSeek-R1 distills: thinking on by default; non-thinking generally not used.
    (re.compile(r"DeepSeek.*R1|deepseek-r1", re.I), {
        "thinking":     {"temperature": 0.6, "top_k": 40, "top_p": 0.95, "min_p": 0.00},
        "non_thinking": {"temperature": 0.6, "top_k": 40, "top_p": 0.95, "min_p": 0.00},
    }),
]


def _family_variants(name_hint: str) -> dict[str, dict[str, float]] | None:
    for pat, variants in _FAMILY_VARIANTS:
        if pat.search(name_hint):
            return {k: dict(v) for k, v in variants.items()}
    return None

# Keys we care about on the GGUF side. Values stored in GGUF are (often) named
# inconsistently across producers — map them to canonical llama.cpp field names.
_GGUF_SAMPLING_KEYS: dict[str, str] = {
    "general.sampling.temp": "temperature",
    "general.sampling.temperature": "temperature",
    "general.sampling.top_k": "top_k",
    "general.sampling.top_p": "top_p",
    "general.sampling.min_p": "min_p",
    "general.sampling.typical_p": "typical_p",
    "general.sampling.repetition_penalty": "repeat_penalty",
    "general.sampling.repeat_penalty": "repeat_penalty",
    "general.sampling.presence_penalty": "presence_penalty",
    "general.sampling.frequency_penalty": "frequency_penalty",
}


# GGUF stores these as f32, so `top_p` reads back as 0.949999988079071 and
# `top_k` as 20.0. Handed to a preset unchanged, the first turns a clean number
# into noise in the editor and the second renders as `--top-k 20.0`, which
# llama-server rejects. Cleaned once here, at the only place they enter the
# system, so every consumer — the editor, the recipes, an API client pulling
# recommended sampling — sees the same tidy values.
_SAMPLING_INT_KEYS = {"top_k"}


def _clean_sampling(key: str, value: float) -> float | int:
    if key in _SAMPLING_INT_KEYS:
        return int(round(value))
    return round(value, 4)


@dataclass
class ModelDefaults:
    source: str                  # "gguf" | "props" | "family" | "none"
    architecture: str | None
    name: str | None
    base_model: str | None
    quantized_by: str | None
    context_length: int | None
    chat_template_preview: str | None
    sampling: dict[str, float]
    fallback_family: str | None  # which family pattern matched, if we fell back

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "architecture": self.architecture,
            "name": self.name,
            "base_model": self.base_model,
            "quantized_by": self.quantized_by,
            "context_length": self.context_length,
            "chat_template_preview": self.chat_template_preview,
            "sampling": self.sampling,
            "fallback_family": self.fallback_family,
        }


def _read_gguf_sync(path: Path) -> dict[str, Any]:
    """Synchronous GGUF header read — returns a dict of selected keys. Does
    NOT load tensors; only parses the metadata block at the start of the file.

    Goes through gguf_header rather than `gguf.GGUFReader` because the reader
    decodes every field it walks past, and the biggest fields in a modern GGUF
    are the tokenizer tables — ~7 s per model here, spent entirely on a
    vocabulary none of this cares about. The filter below is applied *during*
    the walk instead of after it.
    """
    want_prefixes = ("general.", "tokenizer.chat_template")
    # We also want <arch>.context_length but we don't know the arch yet;
    # pull any *.context_length plus the geometry keys vram_estimate.py needs
    # (block_count, attention.head_count{,_kv}, attention.{key,value}_length,
    # embedding_length).
    want_suffixes = (
        ".context_length",
        ".block_count",
        ".attention.head_count",
        ".attention.head_count_kv",
        ".attention.key_length",
        ".attention.value_length",
        ".attention.key_length_swa",
        ".attention.value_length_swa",
        ".attention.sliding_window",
        ".attention.sliding_window_pattern",
        ".embedding_length",
        ".full_attention_interval",
        ".ssm.state_size",
        ".ssm.inner_size",
        ".ssm.conv_kernel",
        ".ssm.group_count",
    )
    def want(k: str) -> bool:
        # Token tables can match a wanted prefix (tokenizer.*); never decode one.
        if "tokens" in k or "merges" in k or "scores" in k:
            return False
        return k.startswith(want_prefixes) or k.endswith(want_suffixes)

    out: dict[str, Any] = {}
    for k, val in read_kv(path, want).items():
        if val is None:
            continue  # an array too large to be one of ours (see MAX_ARRAY_ELEMS)
        if isinstance(val, bytes):
            val = val.decode("utf-8", errors="replace")
        out[k] = val
    return out


# Module-level cache for parsed GGUF metadata. Key includes (size, mtime_ns)
# so the cache invalidates automatically when a file is replaced. Bounded
# manually since lru_cache can't take Path keys with side-effects on staleness.
_GGUF_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}
_GGUF_CACHE_MAX = 64


def _read_gguf_cached(path: Path) -> dict[str, Any]:
    try:
        st = path.stat()
    except OSError:
        return _read_gguf_sync(path)
    key = (str(path), st.st_size, st.st_mtime_ns)
    cached = _GGUF_CACHE.get(key)
    if cached is not None:
        return cached
    out = _read_gguf_sync(path)
    if len(_GGUF_CACHE) >= _GGUF_CACHE_MAX:
        # FIFO eviction is fine; preset editor accesses ~10 unique GGUFs.
        _GGUF_CACHE.pop(next(iter(_GGUF_CACHE)))
    _GGUF_CACHE[key] = out
    return out


async def read_gguf(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    return await asyncio.to_thread(_read_gguf_cached, p)


def _family_fallback(name_hint: str) -> tuple[str | None, dict[str, float]]:
    """Match a name hint (filename, general.name, general.base_model, ...) against
    the curated family table. Returns (matched_family_regex_or_None, sampling_dict)."""
    for pat, defaults in _FAMILY_DEFAULTS:
        if pat.search(name_hint):
            return pat.pattern, dict(defaults)
    return None, {}


async def _try_props(preset_status: dict | None) -> dict | None:
    """If the preset is running, query llama-server /props. Returns the parsed
    JSON or None on any failure / if not running."""
    if not preset_status or not preset_status.get("running"):
        return None
    cfg = preset_status.get("config") or {}
    port = cfg.get("port")
    host = cfg.get("host") or "127.0.0.1"
    if host == "0.0.0.0":
        host = "127.0.0.1"
    if not port:
        return None
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            r = await c.get(f"http://{host}:{port}/props")
            if r.status_code == 200:
                return r.json()
    except Exception as e:
        log.debug("props probe failed: %s", e)
    return None


async def get_model_defaults(
    model_path: str,
    preset_status: dict | None = None,
) -> ModelDefaults:
    """Return recommended sampling + architecture metadata for a GGUF file.

    If the matching preset is running, the live /props values override both
    the GGUF and the family fallback (since they reflect what llama.cpp
    actually computed for this binary + model). The `source` field reports
    which path won.
    """
    gguf = await read_gguf(model_path)

    sampling_from_gguf: dict[str, float] = {}
    for gguf_key, canonical in _GGUF_SAMPLING_KEYS.items():
        if gguf_key in gguf:
            try:
                sampling_from_gguf[canonical] = _clean_sampling(canonical, float(gguf[gguf_key]))
            except (TypeError, ValueError):
                continue

    arch = str(gguf.get("general.architecture")) if gguf.get("general.architecture") is not None else None
    name = str(gguf.get("general.name")) if gguf.get("general.name") is not None else None
    base_model = None
    if gguf.get("general.base_model.0.name"):
        base_model = str(gguf["general.base_model.0.name"])
    quantized_by = str(gguf.get("general.quantized_by")) if gguf.get("general.quantized_by") else None
    # Arch-scoped context_length — e.g. qwen35.context_length
    ctx_length = None
    for k, v in gguf.items():
        if k.endswith(".context_length"):
            try:
                ctx_length = int(v)
            except (TypeError, ValueError):
                pass
            break
    tmpl = gguf.get("tokenizer.chat_template")
    tmpl_preview = None
    if isinstance(tmpl, str):
        tmpl_preview = tmpl[:240].replace("\n", "\\n")

    # Try live /props — highest authority
    props = await _try_props(preset_status)
    sampling_from_props: dict[str, float] = {}
    if props and isinstance(props, dict):
        dg = props.get("default_generation_settings") or {}
        for key in ("temperature", "top_k", "top_p", "min_p", "typical_p", "repeat_penalty", "presence_penalty", "frequency_penalty"):
            if key in dg:
                try:
                    sampling_from_props[key] = _clean_sampling(key, float(dg[key]))
                except (TypeError, ValueError):
                    continue

    # Family fallback — used when neither GGUF nor /props gave us anything
    family = None
    family_sampling: dict[str, float] = {}
    hint = " ".join(filter(None, [name, base_model, Path(model_path).stem]))
    family_pattern, family_sampling = _family_fallback(hint)
    if family_pattern:
        family = family_pattern

    # Decide winner & source
    if sampling_from_props:
        source = "props"
        chosen = sampling_from_props
    elif sampling_from_gguf:
        source = "gguf"
        chosen = sampling_from_gguf
    elif family_sampling:
        source = "family"
        chosen = family_sampling
    else:
        source = "none"
        chosen = {}

    return ModelDefaults(
        source=source,
        architecture=arch,
        name=name,
        base_model=base_model,
        quantized_by=quantized_by,
        context_length=ctx_length,
        chat_template_preview=tmpl_preview,
        sampling=chosen,
        fallback_family=family,
    )


@dataclass
class RecommendedSampling:
    """Mode-aware sampling for a single model — what callers (HFD-2) should use
    as the BASE before layering use-case tuning (presence_penalty etc.) and
    per-call overrides (per-expert YAML temperature)."""

    model_id: str
    source: str                     # "props" | "gguf" | "family-variants" | "family" | "none"
    architecture: str | None
    fallback_family: str | None
    thinking: dict[str, float]      # temp/top_k/top_p/min_p
    non_thinking: dict[str, float]
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "source": self.source,
            "architecture": self.architecture,
            "fallback_family": self.fallback_family,
            "thinking": self.thinking,
            "non_thinking": self.non_thinking,
            "notes": self.notes,
        }


_BASE_KEYS = ("temperature", "top_k", "top_p", "min_p")


def _project_base(d: dict[str, float]) -> dict[str, float]:
    return {k: d[k] for k in _BASE_KEYS if k in d}


async def get_recommended_sampling(
    model_path: str,
    preset_status: dict | None = None,
) -> RecommendedSampling:
    """Return base sampling for a model, split into thinking vs non-thinking
    variants. Priority:

      1. Family variants table (Qwen3.x, QwQ, DeepSeek-R1) — these publish
         distinct values per mode, which a single /props snapshot cannot
         distinguish (server only knows the current default).
      2. Live /props (assumed to apply equally to both modes — caller knows
         which mode it's in).
      3. GGUF general.sampling.* (single value, applied to both).
      4. Family fallback table (single value, applied to both).
    """
    md = await get_model_defaults(model_path, preset_status)
    name_hint = " ".join(filter(None, [md.name, md.base_model, Path(model_path).stem]))
    model_id = md.name or Path(model_path).stem

    variants = _family_variants(name_hint)
    if variants:
        # If /props is live, prefer its values for whichever mode the live
        # server is presumably configured for, but keep both variants visible.
        notes = "thinking/non_thinking from family table (Qwen3 best-practices)"
        return RecommendedSampling(
            model_id=model_id,
            source="family-variants",
            architecture=md.architecture,
            fallback_family=md.fallback_family,
            thinking=variants["thinking"],
            non_thinking=variants["non_thinking"],
            notes=notes,
        )

    base = _project_base(md.sampling)
    if not base:
        return RecommendedSampling(
            model_id=model_id,
            source="none",
            architecture=md.architecture,
            fallback_family=md.fallback_family,
            thinking={},
            non_thinking={},
            notes="no recommendation found (no GGUF metadata, no /props, no family match)",
        )
    return RecommendedSampling(
        model_id=model_id,
        source=md.source,
        architecture=md.architecture,
        fallback_family=md.fallback_family,
        thinking=dict(base),
        non_thinking=dict(base),
        notes="single recommendation applied to both modes (no mode-split known for this family)",
    )
