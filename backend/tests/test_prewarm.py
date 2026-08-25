"""Page-cache pre-warming for CPU-offloaded expert tensors.

The regression this protects against: a MoE model served with `--n-cpu-moe N`
keeps the expert weights of the first N layers mmap'd from disk, and after a
cold page cache every decode step faults expert pages in from storage — decode
drops from ~19 t/s to ~1 t/s. prewarm.py must locate exactly those tensors'
byte ranges (shard by shard, block by block) and warm only them.
"""
from __future__ import annotations

import os
import struct

from lld import prewarm
from lld.prewarm import ALL_CPU_MOE


# ---- synthetic GGUF helpers -------------------------------------------------

_F32 = 0
_U32 = 4
_STR = 8
_U64 = 10


def _wstr(buf: bytearray, s: str) -> None:
    b = s.encode()
    buf += struct.pack("<Q", len(b)) + b


def _wval(buf: bytearray, vtype: int, val) -> None:
    if vtype == _STR:
        _wstr(buf, val)
    elif vtype == _U32:
        buf += struct.pack("<I", val)
    elif vtype == _U64:
        buf += struct.pack("<Q", val)
    else:
        raise ValueError(vtype)


def _tensor_entry_size(name: str, dims: list[int]) -> int:
    return 8 + len(name.encode()) + 4 + 8 * len(dims) + 4 + 8


def write_gguf(
    path,
    kv: list[tuple[str, int, object]],
    tensors: list[tuple[str, list[int], int]],
    data_size: int = 0,
) -> None:
    """Minimal GGUF v3: metadata + tensor index + (zeros) data.

    `tensors` is (name, dims, ggml_type); F32 is used so sizes are
    n_elements * 4. Data bytes are written as zeros after the header.
    """
    hdr = bytearray()
    hdr += b"GGUF"
    hdr += struct.pack("<I", 3)
    hdr += struct.pack("<Q", len(tensors))
    hdr += struct.pack("<Q", len(kv))
    for k, vt, v in kv:
        _wstr(hdr, k)
        hdr += struct.pack("<I", vt)
        _wval(hdr, vt, v)
    entry_offsets: list[int] = []
    off = len(hdr) + sum(_tensor_entry_size(n, d) for n, d, _ in tensors)
    for name, dims, _ in tensors:
        entry_offsets.append(off)
        n_elements = 1
        for d in dims:
            n_elements *= d
        off += n_elements * 4  # F32
    for (name, dims, gtype), off in zip(tensors, entry_offsets):
        _wstr(hdr, name)
        hdr += struct.pack("<I", len(dims))
        for d in dims:
            hdr += struct.pack("<Q", d)
        hdr += struct.pack("<I", gtype)
        hdr += struct.pack("<Q", off)
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(b"\0" * (entry_offsets[-1] - len(hdr) + 1 if entry_offsets else data_size))


def _moe_tensors() -> list[tuple[str, list[int], int]]:
    return [
        ("blk.0.ffn_gate_exps.weight", [16, 16, 4], _F32),
        ("blk.0.ffn_up_exps.weight", [16, 16, 4], _F32),
        ("blk.0.ffn_gate_shexp.weight", [16, 16], _F32),   # shared expert: NOT offloaded
        ("blk.0.attn_norm.weight", [16], _F32),
        ("blk.1.ffn_down_exps.weight", [16, 16, 4], _F32),
        ("blk.5.ffn_gate_exps.weight", [16, 16, 4], _F32),  # beyond n-cpu-moe: stays on GPU
        ("output.weight", [16, 16], _F32),
    ]


# ---- tensor_ranges ----------------------------------------------------------

def test_tensor_ranges_parses_names_sizes_and_offsets(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, [("general.architecture", _STR, "deepseek4")], _moe_tensors())
    table = prewarm.tensor_ranges(p)
    assert "blk.0.ffn_gate_exps.weight" in table
    # F32, 16*16*4 = 1024 elements * 4 bytes
    off, size = table["blk.0.ffn_gate_exps.weight"]
    assert size == 1024 * 4
    assert off >= 0
    # offsets are distinct and ordered by file layout
    offsets = [table[n][0] for n in table]
    assert offsets == sorted(offsets)


def test_tensor_ranges_returns_empty_for_garbage(tmp_path):
    p = tmp_path / "junk.gguf"
    p.write_bytes(b"not a gguf at all")
    assert prewarm.tensor_ranges(p) == {}


# ---- cpu_expert_ranges ------------------------------------------------------

def test_cpu_expert_ranges_keeps_only_offloaded_blocks(tmp_path):
    p = tmp_path / "M-UD-IQ3_XXS-00001-of-00002.gguf"
    write_gguf(p, [("general.architecture", _STR, "deepseek4")], [])  # metadata-only shard 1
    p2 = tmp_path / "M-UD-IQ3_XXS-00002-of-00002.gguf"
    write_gguf(p2, [], _moe_tensors())
    ranges = prewarm.cpu_expert_ranges(str(p), 2)
    names = {r[0].rsplit("/", 1)[-1] for r in ranges}
    # only shard 2, only expert tensors of blocks 0..1
    assert names == {
        "M-UD-IQ3_XXS-00002-of-00002.gguf",
    }
    paths = {r[0] for r in ranges}
    assert len(paths) == 1
    # per-tensor entries: blk.0 gate+up, blk.1 down = 3 ranges
    assert len(ranges) == 3
    for path, off, size in ranges:
        assert off >= 0 and size > 0
        assert os.path.isfile(path)


def test_cpu_expert_ranges_all_cpu_moe_includes_every_block(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, [("general.architecture", _STR, "deepseek4")], _moe_tensors())
    ranges = prewarm.cpu_expert_ranges(str(p), ALL_CPU_MOE)
    # blk.0 gate+up, blk.1 down, blk.5 gate = 4 expert tensors; shexp excluded
    assert len(ranges) == 4


def test_cpu_expert_ranges_zero_is_noop(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, [("general.architecture", _STR, "deepseek4")], _moe_tensors())
    assert prewarm.cpu_expert_ranges(str(p), 0) == []


# ---- cpu_moe_from_args ------------------------------------------------------

def test_cpu_moe_from_args_parses_spawn_argv():
    args = [
        "--alias", "DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004",
        "--device", "CUDA0",
        "--model", "/m/DeepSeek.gguf",
        "--n-cpu-moe", "32",
        "--n-gpu-layers", "999",
    ]
    assert prewarm.cpu_moe_from_args(args) == 32


def test_cpu_moe_from_args_short_flag_and_all():
    assert prewarm.cpu_moe_from_args(["-ncmoe", "16"]) == 16
    assert prewarm.cpu_moe_from_args(["--cpu-moe"]) == ALL_CPU_MOE
    assert prewarm.cpu_moe_from_args(["--flash-attn", "on"]) == 0


# ---- warm_ranges ------------------------------------------------------------

def test_warm_ranges_reads_every_byte(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, [("general.architecture", _STR, "deepseek4")], _moe_tensors())
    ranges = prewarm.cpu_expert_ranges(str(p), 2)
    expected = sum(size for _, _, size in ranges)
    assert expected > 0
    assert prewarm.warm_ranges(ranges) == expected


def test_warm_ranges_empty():
    assert prewarm.warm_ranges([]) == 0


# ---- prewarm_model / prewarm_from_args --------------------------------------

def test_prewarm_model_reports_stats(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, [("general.architecture", _STR, "deepseek4")], _moe_tensors())
    stats = prewarm.prewarm_model(str(p), 2)
    assert stats["ranges"] == 3
    assert stats["bytes"] > 0
    assert stats["n_cpu_moe"] == 2


def test_prewarm_model_skips_non_moe(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, [("general.architecture", _STR, "deepseek4")],
               [("blk.0.attn_norm.weight", [16], _F32)])
    stats = prewarm.prewarm_model(str(p), 32)
    assert stats["ranges"] == 0
    assert stats["bytes"] == 0


def test_prewarm_from_args_uses_spawn_argv(tmp_path):
    p = tmp_path / "DeepSeek-00001-of-00002.gguf"
    write_gguf(p, [("general.architecture", _STR, "deepseek4")], [])
    p2 = tmp_path / "DeepSeek-00002-of-00002.gguf"
    write_gguf(p2, [], _moe_tensors())
    args = [
        "--port", "51267", "--alias", "DeepSeek",
        "--model", str(p), "--n-cpu-moe", "2", "--n-gpu-layers", "999",
    ]
    stats = prewarm.prewarm_from_args(args)
    assert stats["ranges"] == 3
    assert stats["bytes"] > 0
