"""What the router INI does — and does not — carry over from a preset.

The router loads models from this file, not from `to_argv`, so anything a
preset field does that never reaches an INI key does nothing at all in router
mode. That gap has bitten twice already (the GPU pin, then the sampler
fields); these tests pin the shape of the newer prompt-cache fields so it
cannot bite a third time silently.
"""
from __future__ import annotations

from lld.router_ini import render_ini
from lld.settings import LlamaServerConfig


def _preset(tmp_path, **kw) -> LlamaServerConfig:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    return LlamaServerConfig(name="p", model_path=str(model), **kw)


def _section(ini: str, name: str = "model") -> dict[str, str]:
    """The `key = value` lines of one section, in order."""
    out: dict[str, str] = {}
    inside = False
    for raw in ini.splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            inside = line == f"[{name}]"
            continue
        if inside and "=" in line and not line.startswith(";"):
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def test_prompt_cache_fields_reach_the_ini(tmp_path):
    """The whole point of making them preset fields: in router mode the INI is
    the only thing the model is loaded from."""
    cfg = _preset(
        tmp_path,
        cache_reuse=256,
        cache_idle_slots=False,
        context_shift=False,
        kv_offload=True,
    )
    keys = _section(render_ini(str(tmp_path), presets=[cfg]))
    assert keys["cache-reuse"] == "256"
    assert keys["cache-idle-slots"] == "false"
    assert keys["context-shift"] == "false"
    assert keys["kv-offload"] == "true"


def test_booleans_are_written_lowercase(tmp_path):
    """llama.cpp's is_truthy/is_falsey match "true"/"false" exactly and
    case-sensitively (common/arg.cpp). Python's str(False) is "False", which
    is *neither* — common_preset::to_args would then keep the positive flag
    while apply_to_params read the same value back as false, inverting the
    setting between the child's command line and the router's own params."""
    ini = render_ini(str(tmp_path), presets=[_preset(tmp_path, context_shift=False)])
    assert "context-shift = false" in ini
    assert "False" not in ini


def test_unset_toggles_emit_nothing(tmp_path):
    """None means "llama.cpp's own default". A preset that never touched these
    must render exactly what it rendered before the fields existed."""
    before = render_ini(str(tmp_path), presets=[_preset(tmp_path)])
    for key in ("cache-reuse", "cache-idle-slots", "context-shift", "kv-offload"):
        assert key not in before


def test_the_router_preset_carries_them_into_the_global_section(tmp_path):
    """[*] is where the router preset's own defaults live; a toggle set there
    should apply to every model that does not override it."""
    router = LlamaServerConfig(name="router-8085", mode="router", context_shift=False)
    ini = render_ini(str(tmp_path), presets=[], router_preset=router)
    assert _section(ini, "*")["context-shift"] == "false"
