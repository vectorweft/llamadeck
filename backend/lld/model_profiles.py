"""What a specific model wants from a preset, as one-click recipes.

The three modules around this one each answer a narrower question:

  * `model_defaults.py`  -> which sampling numbers (GGUF / /props / family table)
  * `model_info.py`      -> the human-readable family briefing
  * `flag_catalog.py`    -> which flags this llama-server build accepts

This one turns those into ACTIONS: "Thinking on", "Instruct (no thinking)",
"OCR" — each a set of preset fields and flags the editor can apply in one
click. That is the part that has to keep working for a model nobody has
written a table entry for yet, so capability detection reads the GGUF's own
chat template instead of matching a name:

    enable_thinking / <think> in the template  ->  the model can think
    tools / tool_calls in the template         ->  it can call tools

A downloaded model therefore arrives with working recipes on day one, and the
curated tables only sharpen the numbers.

Users extend this without touching Python by dropping recipes into
`~/.config/llamadeck/model-profiles.json`:

    {"profiles": [
       {"match": "Qwen3\\\\.6",
        "notes": ["presence_penalty 0 for transcription, 1.5 for prose"],
        "recipes": [
          {"id": "ocr", "label": "OCR / transcription",
           "why": "pp=1.5 makes the model drift on hard pages",
           "set": {"temperature": 0.2, "presence_penalty": 0.0, "reasoning": "off"},
           "add_flags": ["--no-context-shift"]}
        ]}
    ]}
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .model_defaults import get_model_defaults, get_recommended_sampling, read_gguf
from .settings import STATE_DIR, LlamaServerConfig

log = logging.getLogger(__name__)

OVERLAY_PATH = STATE_DIR / "model-profiles.json"

# Chat-template markers. A template that branches on `enable_thinking`, or that
# emits a `<think>` block, is a dual-mode model whatever it calls itself.
_THINKING_MARKERS = ("enable_thinking", "<think>", "reasoning_effort", "thinking")
_TOOL_MARKERS = ("tool_calls", "tools", "function_call")

# Recipe labels are UI content, not error details, so they follow
# settings.ui_language the same way fit-check messages do. Diagnostics about a
# malformed overlay file stay English, per the house rule in fit_check.py.
_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "thinking_label": "Thinking on",
        "instruct_label": "Thinking off (instruct)",
        "recommended_label": "Recommended sampling",
        "thinking_why_split": (
            "Reasoning models want their own sampling — the instruct numbers "
            "make a thinking model loop."
        ),
        "instruct_why_split": (
            "Straight answers, no <think> block. Note the different sampling: "
            "this is not the thinking preset with reasoning switched off."
        ),
        "thinking_why_same": "Recommended sampling ({source}), thinking forced on.",
        "instruct_why_same": "Recommended sampling ({source}), thinking forced off.",
        "recommended_why": "From {source}.",
        "same_numbers_note": (
            "This model can think, but no per-mode sampling is known for it — "
            "both recipes use the same numbers."
        ),
    },
    "tr": {
        "thinking_label": "Düşünme açık",
        "instruct_label": "Düşünme kapalı (instruct)",
        "recommended_label": "Önerilen örnekleme",
        "thinking_why_split": (
            "Düşünen modeller kendi örnekleme değerlerini ister — instruct "
            "değerleriyle çalıştırılan bir düşünme modeli döngüye girer."
        ),
        "instruct_why_split": (
            "Doğrudan cevap, <think> bloğu yok. Örnekleme değerleri farklı: "
            "bu, düşünme preset'inin sadece kapatılmış hâli değil."
        ),
        "thinking_why_same": "Önerilen örnekleme ({source}), düşünme açık.",
        "instruct_why_same": "Önerilen örnekleme ({source}), düşünme kapalı.",
        "recommended_why": "Kaynak: {source}.",
        "same_numbers_note": (
            "Bu model düşünebiliyor ama moda göre ayrı örnekleme değeri "
            "bilinmiyor — iki tarif de aynı sayıları kullanıyor."
        ),
    },
}


def _strings(lang: str) -> dict[str, str]:
    return _STRINGS["tr" if lang == "tr" else "en"]


@dataclass
class Recipe:
    """One click: fields to set, flags to add, flags to drop."""

    id: str
    label: str
    why: str = ""
    set: dict[str, Any] = field(default_factory=dict)
    add_flags: list[str] = field(default_factory=list)
    remove_flags: list[str] = field(default_factory=list)
    source: str = "builtin"      # "builtin" | "user"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelProfile:
    model_id: str
    architecture: str | None
    family: str | None
    context_length: int | None
    capabilities: dict[str, bool]
    detected_by: dict[str, str]
    sampling: dict[str, dict[str, float]]     # {"thinking": {...}, "non_thinking": {...}}
    sampling_source: str
    recipes: list[Recipe]
    notes: list[str]
    overlay_path: str
    overlay_loaded: bool

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["recipes"] = [r.to_dict() for r in self.recipes]
        return d


_CONFIG_FIELDS = {f.name for f in __import__("dataclasses").fields(LlamaServerConfig)}

# GGUF metadata is f32: `top_k` comes back as 20.0 and `top_p` as
# 0.949999988079071. Written to the preset unchanged, the first produces
# `--top-k 20.0` — which llama-server rejects — and the second turns a clean
# number into noise in the editor. Coerce on the way out, once, here.
_INT_FIELDS = {
    "top_k", "ctx_size", "n_gpu_layers", "parallel", "batch_size", "ubatch_size",
    "threads", "port", "draft_max", "draft_min", "n_gpu_layers_draft",
    "models_max", "sleep_idle_seconds", "estimated_vram_mb",
}


def _coerce(field_name: str, value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if field_name in _INT_FIELDS:
        return int(value)
    if isinstance(value, float):
        return round(value, 4)
    return value


def _coerce_set(values: dict[str, Any]) -> dict[str, Any]:
    return {k: _coerce(k, v) for k, v in values.items()}


def _load_overlay() -> tuple[list[dict], bool]:
    """User-supplied profiles. A broken file is logged and ignored — a typo in
    a hand-edited JSON must not take the preset editor down with it."""
    if not OVERLAY_PATH.exists():
        return [], False
    try:
        data = json.loads(OVERLAY_PATH.read_text())
    except (OSError, json.JSONDecodeError) as e:
        log.warning("model-profiles.json ignored (%s)", e)
        return [], False
    profiles = data.get("profiles") if isinstance(data, dict) else data
    if not isinstance(profiles, list):
        log.warning("model-profiles.json: expected a list of profiles")
        return [], False
    return profiles, True


def _overlay_for(name_hint: str, arch: str | None) -> tuple[list[Recipe], list[str], bool]:
    """Recipes and notes whose `match` regex hits this model."""
    profiles, loaded = _load_overlay()
    recipes: list[Recipe] = []
    notes: list[str] = []
    haystack = " ".join(filter(None, [name_hint, arch]))
    for prof in profiles:
        if not isinstance(prof, dict):
            continue
        pattern = prof.get("match") or ""
        try:
            if pattern and not re.search(pattern, haystack, re.I):
                continue
        except re.error as e:
            log.warning("model-profiles.json: bad regex %r (%s)", pattern, e)
            continue
        for n in prof.get("notes") or []:
            if isinstance(n, str):
                notes.append(n)
        for raw in prof.get("recipes") or []:
            if not isinstance(raw, dict) or not raw.get("id"):
                continue
            unknown = [k for k in (raw.get("set") or {}) if k not in _CONFIG_FIELDS]
            if unknown:
                notes.append(
                    f"recipe '{raw['id']}' sets unknown field(s) {', '.join(unknown)} — ignored"
                )
            recipes.append(Recipe(
                id=str(raw["id"]),
                label=str(raw.get("label") or raw["id"]),
                why=str(raw.get("why") or ""),
                set=_coerce_set({k: v for k, v in (raw.get("set") or {}).items() if k in _CONFIG_FIELDS}),
                add_flags=[str(f) for f in (raw.get("add_flags") or [])],
                remove_flags=[str(f) for f in (raw.get("remove_flags") or [])],
                source="user",
            ))
    return recipes, notes, loaded


def detect_capabilities(chat_template: str | None, mmproj_path: str | None = None) -> tuple[dict[str, bool], dict[str, str]]:
    """What the model can do, read off its own chat template.

    Returns (capabilities, detected_by) — the second maps each true capability
    to the evidence, so the UI can say *why* it is offering a thinking toggle
    instead of asking the user to trust it.
    """
    caps = {"thinking": False, "tools": False, "vision": bool(mmproj_path)}
    why: dict[str, str] = {}
    tmpl = (chat_template or "")
    lowered = tmpl.lower()
    for marker in _THINKING_MARKERS:
        if marker in lowered:
            caps["thinking"] = True
            why["thinking"] = f"chat template mentions `{marker}`"
            break
    for marker in _TOOL_MARKERS:
        if marker in lowered:
            caps["tools"] = True
            why["tools"] = f"chat template mentions `{marker}`"
            break
    if mmproj_path:
        why["vision"] = "an mmproj projector sits next to the model"
    return caps, why


def _sampling_recipes(
    caps: dict[str, bool],
    sampling: dict[str, dict[str, float]],
    source: str,
    S: dict[str, str],
) -> list[Recipe]:
    """Built-in recipes: the sampling the model's own authors recommend,
    split by thinking mode where the family publishes two sets."""
    thinking = dict(sampling.get("thinking") or {})
    instruct = dict(sampling.get("non_thinking") or {})
    if not thinking and not instruct:
        return []

    out: list[Recipe] = []
    if caps.get("thinking"):
        differs = thinking != instruct
        out.append(Recipe(
            id="thinking",
            label=S["thinking_label"],
            why=(S["thinking_why_split"] if differs
                 else S["thinking_why_same"].format(source=source)),
            set=_coerce_set({"reasoning": "on", **thinking}),
        ))
        out.append(Recipe(
            id="instruct",
            label=S["instruct_label"],
            why=(S["instruct_why_split"] if differs
                 else S["instruct_why_same"].format(source=source)),
            set=_coerce_set({"reasoning": "off", **instruct}),
        ))
    else:
        base = thinking or instruct
        out.append(Recipe(
            id="recommended",
            label=S["recommended_label"],
            why=S["recommended_why"].format(source=source),
            set=_coerce_set(base),
        ))
    return out


async def get_model_profile(
    model_path: str,
    preset_status: dict | None = None,
    mmproj_path: str | None = None,
    lang: str = "en",
) -> ModelProfile:
    """Capabilities + one-click recipes for one GGUF."""
    S = _strings(lang)
    gguf = await read_gguf(model_path)
    tmpl = gguf.get("tokenizer.chat_template")
    tmpl = tmpl if isinstance(tmpl, str) else None

    defaults = await get_model_defaults(model_path, preset_status=preset_status)
    rec = await get_recommended_sampling(model_path, preset_status=preset_status)

    caps, why = detect_capabilities(tmpl, mmproj_path)
    # A family table that publishes two sampling modes is itself evidence of a
    # dual-mode model, even when the GGUF ships a template we cannot read.
    if not caps["thinking"] and rec.source == "family-variants":
        caps["thinking"] = True
        why["thinking"] = f"known dual-mode family ({rec.fallback_family})"

    name_hint = " ".join(filter(None, [defaults.name, defaults.base_model, Path(model_path).stem]))
    family = None
    try:
        from .model_info import lookup_info  # local: avoids a cycle at import time

        info = lookup_info(name_hint)
        family = info.family if info else None
    except Exception:  # noqa: BLE001 — narrative docs are decoration here
        family = None

    sampling = {"thinking": _coerce_set(rec.thinking), "non_thinking": _coerce_set(rec.non_thinking)}
    recipes = _sampling_recipes(caps, sampling, rec.source, S)
    user_recipes, notes, overlay_loaded = _overlay_for(name_hint, defaults.architecture)
    recipes.extend(user_recipes)

    if caps["thinking"] and rec.thinking == rec.non_thinking and rec.source != "none":
        notes.append(S["same_numbers_note"])

    return ModelProfile(
        model_id=rec.model_id,
        architecture=defaults.architecture,
        family=family,
        context_length=defaults.context_length,
        capabilities=caps,
        detected_by=why,
        sampling=sampling,
        sampling_source=rec.source,
        recipes=recipes,
        notes=notes,
        overlay_path=str(OVERLAY_PATH),
        overlay_loaded=overlay_loaded,
    )


def apply_recipe(cfg: LlamaServerConfig, recipe: Recipe) -> LlamaServerConfig:
    """Apply a recipe to a config in place and return it.

    Flag removal takes the flag's value with it, so re-applying a recipe that
    swaps `--n-cpu-moe 24` for `--n-cpu-moe 32` cannot leave both behind.
    """
    for k, v in recipe.set.items():
        if k in _CONFIG_FIELDS:
            setattr(cfg, k, _coerce(k, v))
    if recipe.remove_flags:
        drop = set(recipe.remove_flags)
        out: list[str] = []
        flags = list(cfg.extra_flags)
        i = 0
        while i < len(flags):
            if flags[i] in drop:
                i += 1
                if i < len(flags) and not flags[i].startswith("-"):
                    i += 1
                continue
            out.append(flags[i])
            i += 1
        cfg.extra_flags = out
    if recipe.add_flags:
        present = set(cfg.extra_flags)
        cfg.extra_flags = [*cfg.extra_flags, *(f for f in recipe.add_flags if f not in present)]
    return cfg
