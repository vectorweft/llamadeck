"""PresetRegistry — named LlamaServerConfig presets stored in presets.json."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .settings import PRESETS_PATH, LlamaServerConfig, atomic_write_text, ensure_state_dirs


class PresetError(Exception):
    pass


class PresetRegistry:
    def __init__(self, path: Path = PRESETS_PATH):
        self.path = path

    def _load_raw(self) -> list[dict[str, Any]]:
        """Load presets.json. On corrupt JSON, preserve the broken file as
        `.broken` and start with an empty list — better than crashing at boot.
        The user's seeds will re-populate via `add_missing()` next startup."""
        import logging
        log = logging.getLogger(__name__)
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text())
        except json.JSONDecodeError as e:
            backup = self.path.with_suffix(".json.broken")
            try:
                self.path.rename(backup)
                log.error("presets.json is corrupt (%s); moved to %s, starting empty", e, backup)
            except OSError:
                log.error("presets.json is corrupt (%s) and could not be moved aside", e)
            return []
        if not isinstance(data, list):
            raise PresetError(f"{self.path} must contain a JSON array")
        return data

    def _save_raw(self, items: list[dict[str, Any]]) -> None:
        ensure_state_dirs()
        atomic_write_text(self.path, json.dumps(items, indent=2) + "\n")

    @staticmethod
    def _sanitize(item: dict) -> dict:
        """Coerce JSON null/missing values back to dataclass defaults, AND
        strip unknown keys so adding/removing dataclass fields doesn't break
        load. New `_FIELD_NAMES`-based filter prevents TypeError on
        `LlamaServerConfig(**item)` when JSON contains extras (e.g. a key
        renamed in code but still present in user's persisted file).

        Without this filter, e.g. removing a field from LlamaServerConfig
        would crash on every preset load until the user hand-edited the JSON."""
        from dataclasses import fields as _fields

        # Coerce None/"" → safe defaults for fields with non-None dataclass defaults
        # (so `mode: null` in legacy JSON doesn't override "single").
        none_to_default = {
            "mode": "single",
            "models_max": 1,
            "models_autoload": True,
            "n_gpu_layers_draft": 999,
            "extra_flags": [],
            "env": {},
            "reasoning": "auto",
            "devices": [],
            "ui_hidden": False,
        }
        for k, v in none_to_default.items():
            if item.get(k) in (None, ""):
                item[k] = v

        # Drop keys that aren't valid dataclass fields (forward-compat).
        valid = {f.name for f in _fields(LlamaServerConfig)}
        return {k: v for k, v in item.items() if k in valid}

    def list(self) -> list[LlamaServerConfig]:
        return [LlamaServerConfig(**self._sanitize(item)) for item in self._load_raw()]

    def get(self, name: str) -> LlamaServerConfig:
        for item in self._load_raw():
            if item.get("name") == name:
                return LlamaServerConfig(**self._sanitize(item))
        raise PresetError(f"preset not found: {name}")

    def upsert(self, cfg: LlamaServerConfig) -> LlamaServerConfig:
        if not cfg.name or not cfg.name.strip():
            raise PresetError("preset name is required")
        items = self._load_raw()
        for i, item in enumerate(items):
            if item.get("name") == cfg.name:
                items[i] = asdict(cfg)
                self._save_raw(items)
                return cfg
        items.append(asdict(cfg))
        self._save_raw(items)
        return cfg

    def delete(self, name: str) -> None:
        items = self._load_raw()
        new = [i for i in items if i.get("name") != name]
        if len(new) == len(items):
            raise PresetError(f"preset not found: {name}")
        self._save_raw(new)

    def seed_if_empty(self, defaults: list[LlamaServerConfig]) -> None:
        if self.path.exists() and self._load_raw():
            return
        self._save_raw([asdict(c) for c in defaults])

    def add_missing(self, defaults: list[LlamaServerConfig]) -> list[str]:
        """Add seed presets that aren't already in the file. Existing entries
        are left untouched so the user's edits are preserved. Returns the names
        that were added."""
        existing = {item.get("name") for item in self._load_raw()}
        added: list[str] = []
        for cfg in defaults:
            if cfg.name not in existing:
                self.upsert(cfg)
                added.append(cfg.name)
        return added


def default_seeds() -> list[LlamaServerConfig]:
    """Canonical seed presets for a fresh install. Kept deliberately minimal:
    one router-mode preset that exposes every GGUF under the default models
    root. Users create their own single-mode presets from the Presets page
    (model paths are machine-specific, so we never seed them)."""
    # llama-server's --models-dir only scans one level deep; deeper model
    # trees rely on the INI exclusively. The INI is auto-(re)generated from
    # sibling single-mode presets on startup and via POST /api/router/ini/write.
    from .settings import STATE_DIR as _STATE_DIR, load_settings as _load_settings
    # The models root the user actually configured — NOT the factory default.
    # Baking Path.home()/"llama.cpp"/"models" in here seeded a router preset
    # pointing at a directory that does not exist on any box whose models live
    # elsewhere, and llama-server exits 1 on a missing --models-dir.
    try:
        _models_root = _load_settings().hf_models_root
    except Exception:  # settings unreadable on a truly fresh box
        _models_root = str(Path.home() / "llama.cpp" / "models")
    router_default = LlamaServerConfig(
        name="router-8085",
        mode="router",
        model_path=None,
        mmproj_path=None,
        # Localhost only — the router serves models with no auth. Users who
        # want it on the LAN change this one field in the preset editor.
        host="127.0.0.1",
        port=8085,
        # llama-server's router needs models_dir (even with INI) to enumerate
        # available model ids for /models. INI overrides per-model settings;
        # vocab-test stubs are filtered out by /api/router/models.
        models_dir=_models_root,
        models_preset_path=str(_STATE_DIR / "router-models.ini"),
        models_max=1,
        models_autoload=True,
        ctx_size=32768,
        n_gpu_layers=999,
        parallel=2,
        batch_size=2048,
        ubatch_size=512,
        threads=8,
        flash_attn="on",
        cache_type_k="q8_0",
        cache_type_v="q8_0",
        cont_batching=True,
        jinja=True,
        metrics=True,
        slots=True,
        extra_flags=[],
        notes="Router on :8085. Models exposed via INI overrides "
              "auto-generated from sibling single-mode presets. Restart-free "
              "swap via /models/load and /models/unload.",
        estimated_vram_mb=None,
    )
    return [router_default]
