"""What made a cold launch take 41 seconds.

`supervisor.statuses()` produced a VRAM estimate for every idle preset, and
each estimate parsed a GGUF header. Boot called it, the metrics poller called
it twice a second, and the whole walk is synchronous — so the health endpoint
did not answer until every model on the machine had been read.

Two things keep it down, and both are load-bearing rather than nice-to-have:
callers that only want "what is running" can ask for that, and the parse
result outlives the process.
"""
from __future__ import annotations

import json

import pytest

from lld import vram_estimate
from lld.presets import PresetRegistry
from lld.settings import LlamaServerConfig
from lld.supervisor import MultiSupervisor


@pytest.fixture
def registry(tmp_path, monkeypatch):
    from lld import presets as presets_mod

    monkeypatch.setattr(presets_mod, "PRESETS_PATH", tmp_path / "presets.json")
    reg = PresetRegistry(tmp_path / "presets.json")
    reg.upsert(LlamaServerConfig(name="a", model_path=str(tmp_path / "a.gguf"), port=18801))
    reg.upsert(LlamaServerConfig(name="b", model_path=str(tmp_path / "b.gguf"), port=18802))
    monkeypatch.setattr("lld.supervisor.PresetRegistry", lambda *a, **k: reg)
    return reg


def test_statuses_can_skip_the_gguf_reads(registry, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        "lld.supervisor.estimate_vram",
        lambda cfg: calls.append(cfg.name),
    )
    sup = MultiSupervisor("/nonexistent/llama-server")

    rows = sup.statuses(vram_estimates=False)

    assert set(rows) == {"a", "b"}
    assert all(r["vram_estimate"] is None for r in rows.values())
    assert calls == [], "no preset should have been estimated"


def test_statuses_still_estimates_by_default(registry, monkeypatch):
    calls: list[str] = []

    class _Est:
        def to_dict(self):
            return {"total_mb": 1}

    def _fake(cfg):
        calls.append(cfg.name)
        return _Est()

    monkeypatch.setattr("lld.supervisor.estimate_vram", _fake)
    sup = MultiSupervisor("/nonexistent/llama-server")

    rows = sup.statuses()

    assert sorted(calls) == ["a", "b"]
    assert rows["a"]["vram_estimate"] == {"total_mb": 1}


# --------------------------------------------------------------------------
# The parsed profile has to survive a restart, not just a request
# --------------------------------------------------------------------------

@pytest.fixture
def profile_cache(tmp_path, monkeypatch):
    from lld import settings as settings_mod

    monkeypatch.setattr(settings_mod, "STATE_DIR", tmp_path)
    vram_estimate._SPLIT_CACHE.clear()
    monkeypatch.setattr(vram_estimate, "_disk_cache", None)
    yield tmp_path / "gguf-profile-cache.json"
    vram_estimate._SPLIT_CACHE.clear()


def _write_model(tmp_path, name="m.gguf"):
    from tests.test_gguf_header import _kv_str, _kv_u32, _tensor, write_gguf

    p = tmp_path / name
    write_gguf(
        p,
        kvs=[
            _kv_str("general.architecture", "qwen3"),
            _kv_u32("qwen3.block_count", 4),
            _kv_u32("qwen3.expert_count", 128),
        ],
        tensors=[
            _tensor("blk.0.ffn_gate_exps.weight", [4096, 1536, 128], 12, 0),
            _tensor("output.weight", [4096, 152064], 1, 8),
        ],
    )
    return p


def test_a_parsed_profile_is_written_to_disk(profile_cache, tmp_path):
    model = _write_model(tmp_path)

    first = vram_estimate.read_model_profile(str(model))

    assert first["expert_count"] == 128
    assert first["n_exp_layers"] == 1
    assert profile_cache.exists()
    assert json.loads(profile_cache.read_text())


def test_a_second_process_reads_the_profile_without_reopening_the_gguf(
    profile_cache, tmp_path, monkeypatch,
):
    model = _write_model(tmp_path)
    expected = vram_estimate.read_model_profile(str(model))

    # Simulate a restart: in-memory caches gone, the file still there. If the
    # GGUF is opened again the boom below fires.
    vram_estimate._SPLIT_CACHE.clear()
    monkeypatch.setattr(vram_estimate, "_disk_cache", None)
    monkeypatch.setattr(
        vram_estimate, "read_tensor_index",
        lambda *_a, **_k: pytest.fail("the GGUF was re-read after a restart"),
    )

    assert vram_estimate.read_model_profile(str(model)) == expected


def test_a_rewritten_model_is_not_served_from_a_stale_entry(profile_cache, tmp_path):
    """The key is path + mtime, so replacing the file invalidates it. A resumed
    or re-quantised download must not keep the old tensor sizes."""
    import os
    import time

    model = _write_model(tmp_path)
    before = vram_estimate.read_model_profile(str(model))

    from tests.test_gguf_header import _kv_str, _kv_u32, _tensor, write_gguf

    write_gguf(
        model,
        kvs=[_kv_str("general.architecture", "qwen3"), _kv_u32("qwen3.expert_count", 8)],
        tensors=[_tensor("blk.0.ffn_gate_exps.weight", [4096, 1536, 8], 12, 0)],
    )
    os.utime(model, (time.time() + 10, time.time() + 10))
    vram_estimate._SPLIT_CACHE.clear()

    after = vram_estimate.read_model_profile(str(model))
    assert after["expert_count"] == 8
    assert after["exps_mb"] != before["exps_mb"]


def test_a_corrupt_cache_file_is_ignored_rather_than_fatal(profile_cache, tmp_path):
    profile_cache.write_text("{not json")
    model = _write_model(tmp_path)
    assert vram_estimate.read_model_profile(str(model))["expert_count"] == 128
