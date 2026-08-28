"""-ot / --override-tensor accounting in the VRAM estimate.

--n-cpu-moe only ever matches `ffn_*_exps`, so a model whose bulk sits in some
other tensor is judged unfittable without reading the overrides. The case that
forced this: Qwen3.8-Flash-Next keeps 26.8 GiB in `per_layer_token_embd`, and
the panel reported "core too big" for a single-5090 preset that in fact loads
and runs at 26.6 of 31.9 GiB.
"""
from __future__ import annotations

from lld import vram_estimate
from lld.settings import LlamaServerConfig
from lld.vram_estimate import override_cpu_mb, parse_tensor_overrides

MB = 1024 * 1024

# Two expert layers plus one large non-expert lookup tensor.
FAKE_INDEX = [
    ("blk.0.ffn_gate_exps.weight", 100 * MB),
    ("blk.1.ffn_gate_exps.weight", 100 * MB),
    ("per_layer_token_embd.weight", 700 * MB),
    ("blk.0.attn_q.weight", 10 * MB),
]


def _cfg(extra):
    return LlamaServerConfig(name="t", model_path="/fake/model.gguf", extra_flags=extra)


def _patch(monkeypatch):
    monkeypatch.setattr(vram_estimate, "split_shards", lambda p: [p])
    monkeypatch.setattr(vram_estimate, "read_tensor_index", lambda p: FAKE_INDEX)
    monkeypatch.setattr(vram_estimate.os, "stat", lambda p: type("S", (), {"st_mtime_ns": 1})())
    vram_estimate._OT_CACHE.clear()


def test_parses_long_and_short_spellings():
    assert parse_tensor_overrides(_cfg(["-ot", r"foo\.weight=CPU"])) == [
        (r"foo\.weight", "CPU")
    ]
    assert parse_tensor_overrides(
        _cfg(["--override-tensor", "a=CPU", "--override-tensor", "b=CUDA0"])
    ) == [("a", "CPU"), ("b", "CUDA0")]


def test_comma_separated_rules_are_flattened():
    assert parse_tensor_overrides(_cfg(["-ot", "a=CPU,b=Vulkan2"])) == [
        ("a", "CPU"), ("b", "Vulkan2"),
    ]


def test_only_cpu_bound_overrides_count(monkeypatch):
    _patch(monkeypatch)
    ov = parse_tensor_overrides(_cfg(["-ot", r"per_layer_token_embd\.weight=Vulkan2"]))
    assert override_cpu_mb("/fake/model.gguf", ov) == 0


def test_non_expert_tensor_is_counted(monkeypatch):
    _patch(monkeypatch)
    ov = parse_tensor_overrides(_cfg(["-ot", r"per_layer_token_embd\.weight=CPU"]))
    assert override_cpu_mb("/fake/model.gguf", ov) == 700


def test_experts_already_parked_by_n_cpu_moe_are_not_double_counted(monkeypatch):
    _patch(monkeypatch)
    ov = parse_tensor_overrides(_cfg(["-ot", "exps=CPU"]))
    # both expert layers on CPU already -> the override adds nothing
    assert override_cpu_mb("/fake/model.gguf", ov, cpu_moe_layers=2) == 0
    # only layer 0 parked -> layer 1 is new
    assert override_cpu_mb("/fake/model.gguf", ov, cpu_moe_layers=1) == 100


def test_regex_is_a_search_not_a_full_match(monkeypatch):
    _patch(monkeypatch)
    ov = parse_tensor_overrides(_cfg(["-ot", "token_embd=CPU"]))
    assert override_cpu_mb("/fake/model.gguf", ov) == 700


def test_bad_regex_is_ignored_rather_than_raising(monkeypatch):
    _patch(monkeypatch)
    assert override_cpu_mb("/fake/model.gguf", [("[unclosed", "CPU")]) == 0
