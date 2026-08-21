"""KV-cache sizing for sliding-window-attention models.

Gemma 3/4 keep a full ctx-sized cache on only a few layers; the rest hold
`sliding_window` tokens *per slot*. Estimating them as all-full-attention
overshoots by an order of magnitude and makes fit_check block presets that
run fine (a real Gemma-4-31B preset was reported as "~112 GB needed" on a
32 GB card that in fact loads it at 27 GB). The GGUF is faked here — only
the geometry maths is under test.
"""
from __future__ import annotations

from lld import vram_estimate
from lld.settings import LlamaServerConfig
from lld.vram_estimate import estimate_vram

# Real gemma-4-31B-it geometry: 60 layers on a 5:1 SWA:full pattern, SWA
# layers with 16 KV heads at head_dim 256, full layers with 4 at 512.
GEMMA4_FIELDS = {
    "gemma4.block_count": 60,
    "gemma4.attention.head_count": 32,
    "gemma4.attention.head_count_kv": [
        4 if (i + 1) % 6 == 0 else 16 for i in range(60)
    ],
    "gemma4.attention.key_length": 512,
    "gemma4.attention.value_length": 512,
    "gemma4.attention.key_length_swa": 256,
    "gemma4.attention.value_length_swa": 256,
    "gemma4.attention.sliding_window": 1024,
    "gemma4.attention.sliding_window_pattern": [
        (i + 1) % 6 != 0 for i in range(60)
    ],
    "gemma4.embedding_length": 5376,
}

# A plain dense model with no SWA keys, to guard the non-SWA branch.
LLAMA_FIELDS = {
    "llama.block_count": 32,
    "llama.attention.head_count": 32,
    "llama.attention.head_count_kv": 8,
    "llama.attention.key_length": 128,
    "llama.attention.value_length": 128,
    "llama.embedding_length": 4096,
}

MODEL_MB = 17950


def _patch(monkeypatch, tmp_path, fields):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setattr(vram_estimate, "_read_gguf_sync", lambda path: fields)
    monkeypatch.setattr(vram_estimate, "_file_mb", lambda p: MODEL_MB if p else 0)
    monkeypatch.setattr(vram_estimate, "read_model_profile", lambda p: {"n_layers": 60})
    vram_estimate._GEOMETRY_CACHE.clear()
    return str(model)


def _cfg(path, ctx, parallel):
    return LlamaServerConfig(
        name="t", model_path=path, ctx_size=ctx, parallel=parallel,
        n_gpu_layers=999, batch_size=2048, ubatch_size=512,
        flash_attn="on", cache_type_k="q8_0", cache_type_v="q8_0",
    )


def test_swa_geometry_parsed(monkeypatch, tmp_path):
    path = _patch(monkeypatch, tmp_path, GEMMA4_FIELDS)
    geo = vram_estimate._read_geometry(path)
    assert sum(geo["swa_mask"]) == 50
    assert geo["sliding_window"] == 1024
    assert geo["k_dim_swa"] == 256
    assert sorted(set(geo["kv_heads_per_layer"])) == [4, 16]


def test_swa_128k_ctx_fits_a_32gb_card(monkeypatch, tmp_path):
    """ctx 131072 / np 4 measures 27.6 GB on a 5090; the pre-fix estimator
    claimed ~112 GB and fit_check refused to start it."""
    path = _patch(monkeypatch, tmp_path, GEMMA4_FIELDS)
    est = estimate_vram(_cfg(path, 131072, 4))
    assert 7000 < est.kv_cache_mb < 9000
    assert est.gpu_mb < 32607


def test_swa_kv_driven_by_parallel_not_ctx(monkeypatch, tmp_path):
    """The window layers are per-slot, so np is the expensive knob: a 48k ctx
    across 20 slots costs more KV than 128k across 4."""
    path = _patch(monkeypatch, tmp_path, GEMMA4_FIELDS)
    wide = estimate_vram(_cfg(path, 48000, 20))
    deep = estimate_vram(_cfg(path, 131072, 4))
    assert wide.kv_cache_mb > deep.kv_cache_mb
    assert wide.gpu_mb > 32607  # OOMs in practice, must be blocked


def test_dense_model_keeps_full_ctx_scaling(monkeypatch, tmp_path):
    path = _patch(monkeypatch, tmp_path, LLAMA_FIELDS)
    small = estimate_vram(_cfg(path, 16384, 4))
    big = estimate_vram(_cfg(path, 65536, 4))
    assert big.kv_cache_mb == 4 * small.kv_cache_mb
    assert small.details["bytes_per_token_kv"] == 32 * 8 * 256 * (17 / 16)
