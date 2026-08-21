"""The lightweight GGUF header reader.

These build real GGUF bytes rather than mock the parser, because the point of
the module is that it agrees with the format down to the byte — it exists to
replace `gguf.GGUFReader`, which decodes the tokenizer vocabulary on the way
past and cost ~7 seconds per model.
"""
from __future__ import annotations

import struct

import pytest

from lld.gguf_header import (
    MAX_ARRAY_ELEMS,
    prefix_suffix_filter,
    read_kv,
    read_tensor_index,
)

# GGUF value type ids
U32, F32, STRING, ARRAY, U64 = 4, 6, 8, 9, 10

F16 = 1     # ggml type: 1 element per block, 2 bytes
Q4_K = 12   # ggml type: 256 elements per block, 144 bytes


def _str(s: str) -> bytes:
    raw = s.encode()
    return struct.pack("<Q", len(raw)) + raw


def _kv_u32(key: str, value: int) -> bytes:
    return _str(key) + struct.pack("<II", U32, value)


def _kv_str(key: str, value: str) -> bytes:
    return _str(key) + struct.pack("<I", STRING) + _str(value)


def _kv_str_array(key: str, values: list[str]) -> bytes:
    body = b"".join(_str(v) for v in values)
    return _str(key) + struct.pack("<IIQ", ARRAY, STRING, len(values)) + body


def _kv_u32_array(key: str, values: list[int]) -> bytes:
    body = struct.pack(f"<{len(values)}I", *values)
    return _str(key) + struct.pack("<IIQ", ARRAY, U32, len(values)) + body


def _tensor(name: str, dims: list[int], ggml_type: int, offset: int) -> bytes:
    return (
        _str(name)
        + struct.pack("<I", len(dims))
        + struct.pack(f"<{len(dims)}Q", *dims)
        + struct.pack("<IQ", ggml_type, offset)
    )


def write_gguf(path, kvs: list[bytes], tensors: list[bytes] = ()) -> None:
    tensors = list(tensors)
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs))
    path.write_bytes(header + b"".join(kvs) + b"".join(tensors) + b"\x00" * 64)


@pytest.fixture
def model(tmp_path):
    """A file shaped like a real GGUF: geometry keys buried behind a vocabulary."""
    p = tmp_path / "model.gguf"
    write_gguf(
        p,
        kvs=[
            _kv_str("general.architecture", "qwen3"),
            _kv_str("general.name", "Test 7B"),
            # 50k tokens ahead of the geometry, exactly as a real file has it:
            # anything that decodes this to reach what follows is doing the
            # expensive thing this module was written to avoid.
            _kv_str_array("tokenizer.ggml.tokens", [f"tok{i}" for i in range(50_000)]),
            _kv_u32_array("tokenizer.ggml.token_type", [1] * 50_000),
            _kv_u32("qwen3.block_count", 28),
            _kv_u32("qwen3.attention.head_count_kv", 4),
            _kv_u32("qwen3.context_length", 131072),
            _kv_u32_array("qwen3.attention.head_count_kv_per_layer", [4, 8, 4]),
        ],
        tensors=[
            _tensor("token_embd.weight", [4096, 152064], F16, 0),
            _tensor("blk.0.ffn_gate_exps.weight", [4096, 1536, 128], Q4_K, 1024),
            _tensor("blk.1.ffn_gate_exps.weight", [4096, 1536, 128], Q4_K, 2048),
        ],
    )
    return p


def test_reads_the_keys_it_was_asked_for(model):
    kv = read_kv(model, prefix_suffix_filter(
        prefixes=("general.",), suffixes=(".block_count", ".context_length"),
    ))
    assert kv["general.architecture"] == "qwen3"
    assert kv["general.name"] == "Test 7B"
    assert kv["qwen3.block_count"] == 28
    assert kv["qwen3.context_length"] == 131072


def test_skips_everything_else_including_the_vocabulary(model):
    kv = read_kv(model, lambda k: k.endswith(".block_count"))
    assert list(kv) == ["qwen3.block_count"]


def test_a_wanted_array_comes_back_as_a_list(model):
    kv = read_kv(model, lambda k: k.endswith(".head_count_kv_per_layer"))
    assert kv["qwen3.attention.head_count_kv_per_layer"] == [4, 8, 4]


def test_an_oversized_array_is_skipped_rather_than_materialised(model):
    """A filter that matches something vocabulary-sized must not build 50k
    Python strings; the key is reported with no value instead."""
    kv = read_kv(model, lambda k: k == "tokenizer.ggml.tokens")
    assert kv == {"tokenizer.ggml.tokens": None}
    assert MAX_ARRAY_ELEMS < 50_000


def test_tensor_index_sizes_match_the_quant_block_math(model):
    index = dict(read_tensor_index(model))
    # F16: one element per block, two bytes each.
    assert index["token_embd.weight"] == 4096 * 152064 * 2
    # Q4_K: 256 elements per 144-byte block.
    expected = (4096 * 1536 * 128 // 256) * 144
    assert index["blk.0.ffn_gate_exps.weight"] == expected
    assert index["blk.1.ffn_gate_exps.weight"] == expected


def test_a_file_that_is_not_gguf_reads_as_empty(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("this is not a model")
    assert read_kv(p) == {}
    assert read_tensor_index(p) == []


def test_a_truncated_download_reads_as_empty(model):
    """Half a GGUF is what an interrupted download leaves behind. It must come
    back as "no metadata", not as an exception out of a status endpoint."""
    data = model.read_bytes()
    model.write_bytes(data[: len(data) // 3])
    assert read_kv(model, lambda k: True) == {}
    assert read_tensor_index(model) == []


def test_a_missing_file_reads_as_empty(tmp_path):
    assert read_kv(tmp_path / "gone.gguf") == {}
    assert read_tensor_index(tmp_path / "gone.gguf") == []


def test_prefix_suffix_filter_matches_either_end():
    want = prefix_suffix_filter(prefixes=("general.",), suffixes=(".block_count",))
    assert want("general.name")
    assert want("llama.block_count")
    assert not want("tokenizer.ggml.tokens")


def test_prefix_only_filter_does_not_match_everything():
    want = prefix_suffix_filter(prefixes=("general.",))
    assert want("general.name")
    assert not want("llama.block_count")
