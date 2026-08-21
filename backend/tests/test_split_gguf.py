"""Multi-part (split) GGUF handling.

A split model is `<prefix>-00001-of-000NN.gguf` plus its siblings, and
llama.cpp finds the siblings by re-expanding the first file's name — it never
lists the directory. Two things used to go wrong because only the path we were
handed got looked at:

  * size. Some publishers put the whole metadata blob in shard 1 and no
    tensors at all, so a 96 GB DeepSeek-V4 measured 5 MB and fit_check
    cheerfully answered "fits" against a 32 GB card that then OOM'd.
  * tensors. The expert tensors live in the later shards, so the MoE scan
    found none, is_moe came back False, and the --n-cpu-moe advice that would
    have made the model runnable never appeared.

The third case is the shard that isn't there at all: llama.cpp aborts during
load and the user is left with "exited with code 1".
"""
from __future__ import annotations

from lld import fit_check
from lld.settings import LlamaServerConfig
from lld.vram_estimate import (
    VramEstimate,
    estimate_vram,
    missing_shards,
    rival_split_prefix,
    split_shards,
)
from lld import vram_estimate


def _write(tmp_path, name: str, size: int):
    p = tmp_path / name
    p.write_bytes(b"\0" * size)
    return p


def _shards(tmp_path, sizes: dict[int, int], stem="M-UD-IQ3_XXS"):
    """Create shards `sizes` maps index -> bytes; indices left out are absent."""
    for idx, size in sizes.items():
        _write(tmp_path, f"{stem}-{idx:05d}-of-00004.gguf", size)
    return str(tmp_path / f"{stem}-00001-of-00004.gguf")


# ---- name derivation --------------------------------------------------------

def test_split_shards_expands_all_parts():
    got = split_shards("/m/Model-UD-IQ3_XXS-00001-of-00004.gguf")
    assert got == [
        f"/m/Model-UD-IQ3_XXS-{i:05d}-of-00004.gguf" for i in range(1, 5)
    ]


def test_split_shards_derives_from_any_index():
    """llama.cpp accepts being pointed at any shard; the prefix is what counts."""
    assert (
        split_shards("/m/M-00003-of-00004.gguf")[0] == "/m/M-00001-of-00004.gguf"
    )


def test_single_file_model_is_its_own_only_shard():
    assert split_shards("/m/model.gguf") == ["/m/model.gguf"]
    assert split_shards("/m/model-Q4_K_M.gguf") == ["/m/model-Q4_K_M.gguf"]


def test_of_00001_is_not_treated_as_a_split():
    assert split_shards("/m/M-00001-of-00001.gguf") == ["/m/M-00001-of-00001.gguf"]


# ---- size ------------------------------------------------------------------

MB = 1024 * 1024


def test_model_mb_sums_every_shard(monkeypatch, tmp_path):
    """The real regression: shard 1 holds metadata only, the weights are in
    2-4. Sizing shard 1 alone reported 5 MB for a 96 GB model."""
    path = _shards(tmp_path, {1: 5 * MB, 2: 40 * MB, 3: 40 * MB, 4: 11 * MB})
    monkeypatch.setattr(vram_estimate, "_read_gguf_sync", lambda p: {
        "deepseek4.block_count": 43,
        "deepseek4.attention.head_count_kv": 1,
        "deepseek4.attention.key_length": 512,
        "deepseek4.attention.value_length": 512,
        "deepseek4.embedding_length": 4096,
    })
    monkeypatch.setattr(vram_estimate, "read_model_profile", lambda p: {"n_layers": 43})
    vram_estimate._GEOMETRY_CACHE.clear()
    est = estimate_vram(LlamaServerConfig(
        name="t", model_path=path, ctx_size=4096, parallel=1, n_gpu_layers=999,
        cache_type_k="q8_0", cache_type_v="q8_0",
    ))
    assert est.details["model_path_mb"] == 96


def test_missing_shards_named(tmp_path):
    path = _shards(tmp_path, {1: MB, 2: MB, 4: MB})  # 3 absent
    absent = missing_shards(path)
    assert len(absent) == 1
    assert absent[0].endswith("-00003-of-00004.gguf")


def test_no_missing_shards_when_complete(tmp_path):
    path = _shards(tmp_path, {1: MB, 2: MB, 3: MB, 4: MB})
    assert missing_shards(path) == []


# ---- fit_check reports the missing shard instead of a memory verdict --------

def _est() -> VramEstimate:
    return VramEstimate(
        total_mb=8192, model_mb=6144, kv_cache_mb=1024, compute_mb=1024,
        source="computed", details={"n_layers": 43},
    )


def _check(path, lang="en"):
    return fit_check.check_fit(
        LlamaServerConfig(name="t", model_path=path, ctx_size=4096, n_gpu_layers=999),
        gpu_total_mb=32607, gpu_free_mb=32000,
        ram_total_mb=91895, ram_available_mb=86000, lang=lang,
    )


def test_incomplete_split_is_broken_not_fits(monkeypatch, tmp_path):
    """Plenty of VRAM, but a shard is gone — the answer must be the missing
    file, not 'fits'. Without this the server dies with 'exited with code 1'."""
    monkeypatch.setattr(fit_check, "estimate_vram", lambda cfg: _est())
    path = _shards(tmp_path, {1: MB, 2: MB, 4: MB})
    res = _check(path)
    assert res["level"] == "broken"
    assert "00003-of-00004" in res["headline"]
    assert res["messages"][0]["severity"] == "error"


def test_incomplete_split_message_is_translated(monkeypatch, tmp_path):
    monkeypatch.setattr(fit_check, "estimate_vram", lambda cfg: _est())
    path = _shards(tmp_path, {1: MB, 2: MB, 4: MB})
    assert "Model eksik" in _check(path, lang="tr")["headline"]


# ---- parts from two different releases in one folder -----------------------

def test_rival_prefix_detected(tmp_path):
    """The real shape of it: part 1 from the 0731 release, parts 2-4 from the
    plain one. llama.cpp calls part 2 missing while it is sitting right there
    under a different name."""
    _write(tmp_path, "M-0731-UD-IQ3_XXS-00001-of-00004.gguf", MB)
    for i in (2, 3, 4):
        _write(tmp_path, f"M-UD-IQ3_XXS-{i:05d}-of-00004.gguf", MB)
    path = str(tmp_path / "M-0731-UD-IQ3_XXS-00001-of-00004.gguf")
    assert missing_shards(path)  # llama.cpp's view
    assert rival_split_prefix(path) == "M-UD-IQ3_XXS"


def test_no_rival_when_the_set_is_consistent(tmp_path):
    path = _shards(tmp_path, {1: MB, 2: MB, 3: MB, 4: MB})
    assert rival_split_prefix(path) is None


def test_no_rival_for_a_single_file_model(tmp_path):
    p = _write(tmp_path, "model.gguf", MB)
    assert rival_split_prefix(str(p)) is None


def test_rival_named_in_the_fit_check_messages(monkeypatch, tmp_path):
    monkeypatch.setattr(fit_check, "estimate_vram", lambda cfg: _est())
    _write(tmp_path, "M-0731-UD-IQ3_XXS-00001-of-00004.gguf", MB)
    for i in (2, 3, 4):
        _write(tmp_path, f"M-UD-IQ3_XXS-{i:05d}-of-00004.gguf", MB)
    res = _check(str(tmp_path / "M-0731-UD-IQ3_XXS-00001-of-00004.gguf"))
    assert res["level"] == "broken"
    assert any("M-UD-IQ3_XXS" in m["text"] for m in res["messages"])


def test_complete_split_reaches_the_normal_verdict(monkeypatch, tmp_path):
    monkeypatch.setattr(fit_check, "estimate_vram", lambda cfg: _est())
    monkeypatch.setattr(fit_check, "read_model_profile", lambda p: {"n_layers": 43})
    path = _shards(tmp_path, {1: MB, 2: MB, 3: MB, 4: MB})
    assert _check(path)["level"] == "fits"
