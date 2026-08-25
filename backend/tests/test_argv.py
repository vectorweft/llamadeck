from __future__ import annotations
import pytest


from lld.argv import from_argv, to_argv


def test_roundtrip_qwen36_coding():
    argv = [
        "/home/user/llama.cpp/build/bin/llama-server",
        "--model", "/home/user/llama.cpp/models/Qwen3.6/Qwen3.6-35B-A3B-MXFP4_MOE.gguf",
        "--mmproj", "/home/user/llama.cpp/models/Qwen3.6/mmproj-F16.gguf",
        "--host", "0.0.0.0",
        "--port", "8080",
        "--n-gpu-layers", "999",
        "--ctx-size", "147456",
        "--threads", "8",
        "--batch-size", "8192",
        "--ubatch-size", "2048",
        "--flash-attn", "on",
        "--cont-batching",
        "--metrics",
        "--jinja",
        "--reasoning-format", "deepseek",
        "--no-context-shift",
        "--parallel", "6",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "-n", "16384",
    ]
    cfg = from_argv(argv, name="qwen3.6-coding")
    assert cfg.model_path == "/home/user/llama.cpp/models/Qwen3.6/Qwen3.6-35B-A3B-MXFP4_MOE.gguf"
    assert cfg.mmproj_path == "/home/user/llama.cpp/models/Qwen3.6/mmproj-F16.gguf"
    assert cfg.port == 8080
    assert cfg.ctx_size == 147456
    assert cfg.parallel == 6
    assert cfg.flash_attn == "on"
    assert cfg.cache_type_k == "q8_0"
    assert cfg.cache_type_v == "q8_0"
    assert cfg.threads == 8
    assert cfg.jinja is True
    assert cfg.metrics is True
    assert cfg.cont_batching is True
    assert "--reasoning-format" in cfg.extra_flags
    assert "deepseek" in cfg.extra_flags
    # --no-context-shift is now a first-class field, not an unknown extra flag.
    assert cfg.context_shift is False
    assert "--no-context-shift" not in cfg.extra_flags
    assert "-n" in cfg.extra_flags

    rebuilt = to_argv(cfg, binary="/home/user/llama.cpp/build/bin/llama-server")
    for needle in ["--model", "--mmproj", "--port", "8080", "--jinja", "--metrics", "--flash-attn", "on", "q8_0", "--parallel", "6"]:
        assert needle in rebuilt
    assert "--no-context-shift" in rebuilt


def test_short_flag_aliases():
    argv = ["llama-server", "-m", "/x.gguf", "-c", "4096", "-ngl", "50", "-np", "2", "-fa", "auto"]
    cfg = from_argv(argv)
    assert cfg.model_path == "/x.gguf"
    assert cfg.ctx_size == 4096
    assert cfg.n_gpu_layers == 50
    assert cfg.parallel == 2
    assert cfg.flash_attn == "auto"


def test_device_pin_roundtrips_through_dev_and_ts():
    """`-dev` takes one comma-joined value, so the field is a list on our side
    and a single token on llama-server's."""
    from lld.settings import LlamaServerConfig

    cfg = LlamaServerConfig(
        name="pinned", model_path="/m.gguf",
        devices=["CUDA0", "Vulkan1"], tensor_split="1,0",
    )
    argv = to_argv(cfg, "llama-server")
    assert argv[argv.index("-dev") + 1] == "CUDA0,Vulkan1"
    assert argv[argv.index("-ts") + 1] == "1,0"

    back = from_argv(argv)
    assert back.devices == ["CUDA0", "Vulkan1"]
    assert back.tensor_split == "1,0"
    assert back.extra_flags == []


def test_cpu_pin_needs_no_translation_layer():
    """The CPU row's id IS llama-server's value for "offload nothing", so it
    reaches the command line untouched and comes back the same. Storing
    something friendlier (a "CPU" that became `none` on the way out) would make
    the command box show one thing and the preset hold another."""
    from lld.settings import LlamaServerConfig

    cfg = LlamaServerConfig(
        name="lfm-cpu", model_path="/m.gguf", devices=["none"], n_gpu_layers=0,
    )
    argv = to_argv(cfg, "llama-server")
    assert argv[argv.index("-dev") + 1] == "none"
    assert from_argv(argv).devices == ["none"]


def test_unpinned_preset_emits_no_device_flags():
    """The default must stay exactly what it was before pinning existed."""
    from lld.settings import LlamaServerConfig

    argv = to_argv(LlamaServerConfig(name="plain", model_path="/m.gguf"), "llama-server")
    assert "-dev" not in argv
    assert "-ts" not in argv


def test_long_device_flags_migrate_out_of_extra_flags():
    """A preset that carried `--device` by hand before the field existed, or a
    running process being adopted, must fill the field rather than round-trip
    the flag as an opaque extra."""
    cfg = from_argv([
        "llama-server", "-m", "/m.gguf",
        "--device", "CUDA0,Vulkan1", "--tensor-split", "1,0",
    ])
    assert cfg.devices == ["CUDA0", "Vulkan1"]
    assert cfg.tensor_split == "1,0"
    assert cfg.extra_flags == []


def test_draft_floor_above_ceiling_is_rejected():
    """llama.cpp discards any draft shorter than n_min, so n_min > n_max throws
    every draft away — slower than no speculation, while still reporting it on.
    Measured on qwen3.8-27b-vision: 17.9 tok/s at n_min=70 vs 32.2 at n_min=0."""
    from fastapi import HTTPException

    from lld.api.presets_api import _reject_bad_speculation
    from lld.settings import LlamaServerConfig

    bad = LlamaServerConfig(name="x", spec_type="draft-mtp", draft_max=3, draft_min=70)
    with pytest.raises(HTTPException) as e:
        _reject_bad_speculation(bad)
    assert "draft_min" in str(e.value.detail)

    ok = LlamaServerConfig(name="x", spec_type="draft-mtp", draft_max=3, draft_min=0)
    _reject_bad_speculation(ok)  # must not raise


def test_fractional_draft_min_is_rejected_not_truncated():
    """A fractional token count means the user reached for --spec-draft-p-min
    (a 0..1 probability). Truncating it to 0 would hide the mistake."""
    from fastapi import HTTPException

    from lld.api.presets_api import _reject_bad_speculation
    from lld.settings import LlamaServerConfig

    with pytest.raises(HTTPException):
        _reject_bad_speculation(
            LlamaServerConfig(name="x", spec_type="draft-mtp", draft_max=2, draft_min=0.7)
        )


def test_cache_reuse_fields_roundtrip():
    """The prompt-processing / KV-cache reuse knobs as first-class fields."""
    from lld.settings import LlamaServerConfig

    cfg = LlamaServerConfig(
        name="cached", model_path="/m.gguf",
        cache_reuse=256, cache_idle_slots=True, context_shift=False, kv_offload=True,
    )
    argv = to_argv(cfg, "llama-server")
    assert "--cache-reuse" in argv and argv[argv.index("--cache-reuse") + 1] == "256"
    assert "--cache-idle-slots" in argv
    assert "--no-context-shift" in argv
    assert "--kv-offload" in argv

    back = from_argv(argv)
    assert back.cache_reuse == 256
    assert back.cache_idle_slots is True
    assert back.context_shift is False
    assert back.kv_offload is True


def test_cache_toggles_default_to_nothing():
    """None on all four means emit nothing — untouched presets stay untouched."""
    from lld.settings import LlamaServerConfig

    argv = to_argv(LlamaServerConfig(name="plain", model_path="/m.gguf"), "llama-server")
    for flag in ("--cache-reuse", "--cache-idle-slots", "--no-cache-idle-slots",
                 "--context-shift", "--no-context-shift", "--kv-offload", "--no-kv-offload"):
        assert flag not in argv, flag


def test_short_spellings_of_the_kv_offload_toggle_are_adopted():
    """A server adopted from a hand-written command line is as likely to say
    -nkvo as --no-kv-offload; llama-server treats them as one option. Missing
    the short form dropped it into extra_flags, where the preset editor cannot
    show it and the field reads as unset."""
    assert from_argv(["llama-server", "-nkvo"]).kv_offload is False
    assert from_argv(["llama-server", "-kvo"]).kv_offload is True
    assert from_argv(["llama-server", "-nkvo"]).extra_flags == []


def test_an_adopted_short_toggle_re_emits_as_the_long_flag():
    cfg = from_argv(["llama-server", "-nkvo"])
    assert "--no-kv-offload" in to_argv(cfg, "llama-server")
