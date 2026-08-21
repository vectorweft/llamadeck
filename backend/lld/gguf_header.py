"""Minimal GGUF header reader: the KV metadata and the tensor index, nothing else.

`gguf.GGUFReader` is the obvious way to do this and the wrong one here. It
materialises *every* field eagerly, and a modern GGUF's largest fields are the
tokenizer tables — `tokenizer.ggml.tokens`, `.merges`, `.scores` — which run to
hundreds of thousands of entries each. Measured on this box, opening one 20 GB
Qwen3.8 GGUF that way costs ~7.4 seconds of pure CPU, essentially all of it in
`_get_str` over the vocabulary. Nothing in LlamaDeck ever looks at the
vocabulary: the callers want `block_count`, `attention.head_count_kv`,
`context_length`, and the size of the expert tensors.

Six presets × 7.4 s is where a ~40 second startup came from, because
`supervisor.statuses()` estimates VRAM for every idle preset and the whole walk
is synchronous. This module reads the same numbers by streaming the header and
*seeking past* any value it was not asked for. The vocabulary is skipped
without ever being decoded.

Both entry points fall back to returning nothing on a malformed file — callers
already treat "no metadata" as a normal outcome (a partial download, a file on
a disconnected mount), and a truncated GGUF must not raise out of a status
endpoint.

Format reference (GGUF v2/v3), little-endian:

    magic "GGUF" | version u32 | tensor_count u64 | kv_count u64
    kv_count  × [ key: str | value_type u32 | value ]
    tensor_count × [ name: str | n_dims u32 | dims u64[n_dims] | type u32 | offset u64 ]

where `str` is a u64 byte length followed by that many UTF-8 bytes.
"""
from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable

log = logging.getLogger(__name__)

MAGIC = b"GGUF"

# GGUFValueType, inlined so this module does not pay an import of the gguf
# package just to read four integers.
_UINT8, _INT8, _UINT16, _INT16 = 0, 1, 2, 3
_UINT32, _INT32, _FLOAT32, _BOOL = 4, 5, 6, 7
_STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 8, 9, 10, 11, 12

#: value type -> (struct code, byte width)
_SCALARS: dict[int, tuple[str, int]] = {
    _UINT8: ("B", 1), _INT8: ("b", 1),
    _UINT16: ("H", 2), _INT16: ("h", 2),
    _UINT32: ("I", 4), _INT32: ("i", 4),
    _FLOAT32: ("f", 4), _BOOL: ("?", 1),
    _UINT64: ("Q", 8), _INT64: ("q", 8),
    _FLOAT64: ("d", 8),
}

#: Ceiling on how many elements of a wanted array are actually decoded. The
#: arrays LlamaDeck reads are per-layer (a few hundred entries at most); a
#: match on something vocabulary-sized is a mistake in the caller's filter, and
#: should not turn into hundreds of thousands of Python objects.
MAX_ARRAY_ELEMS = 8192

#: Guards against a corrupt length field turning into a multi-gigabyte read.
_MAX_STRING_BYTES = 64 * 1024 * 1024


class GgufFormatError(ValueError):
    pass


def _read_exact(f: BinaryIO, n: int) -> bytes:
    if n < 0:
        raise GgufFormatError(f"negative length {n}")
    buf = f.read(n)
    if len(buf) != n:
        raise GgufFormatError(f"truncated: wanted {n} bytes, got {len(buf)}")
    return buf


def _u32(f: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _read_str(f: BinaryIO) -> str:
    n = _u64(f)
    if n > _MAX_STRING_BYTES:
        raise GgufFormatError(f"implausible string length {n}")
    return _read_exact(f, n).decode("utf-8", errors="replace")


def _skip_str(f: BinaryIO) -> None:
    n = _u64(f)
    if n > _MAX_STRING_BYTES:
        raise GgufFormatError(f"implausible string length {n}")
    f.seek(n, 1)


def _read_value(f: BinaryIO, vtype: int, keep: bool) -> Any:
    """Consume one value. Returns it when `keep`, else None — but always leaves
    the stream positioned immediately after it, which is the whole point."""
    scalar = _SCALARS.get(vtype)
    if scalar is not None:
        code, width = scalar
        raw = _read_exact(f, width)
        return struct.unpack("<" + code, raw)[0] if keep else None
    if vtype == _STRING:
        if keep:
            return _read_str(f)
        _skip_str(f)
        return None
    if vtype == _ARRAY:
        etype = _u32(f)
        count = _u64(f)
        escalar = _SCALARS.get(etype)
        if escalar is not None:
            code, width = escalar
            nbytes = count * width
            if keep and count <= MAX_ARRAY_ELEMS:
                return list(struct.unpack(f"<{count}{code}", _read_exact(f, nbytes)))
            f.seek(nbytes, 1)
            return None
        if etype == _STRING:
            if keep and count <= MAX_ARRAY_ELEMS:
                return [_read_str(f) for _ in range(count)]
            for _ in range(count):
                _skip_str(f)
            return None
        if etype == _ARRAY:
            for _ in range(count):
                _read_value(f, _ARRAY, False)
            return None
        raise GgufFormatError(f"unknown array element type {etype}")
    raise GgufFormatError(f"unknown value type {vtype}")


def _open_header(f: BinaryIO) -> tuple[int, int, int]:
    magic = _read_exact(f, 4)
    if magic != MAGIC:
        raise GgufFormatError(f"not a GGUF file (magic {magic!r})")
    version = _u32(f)
    if version < 2:
        # v1 stored counts as u32. No such file has been in circulation for
        # years; say so rather than mis-parse one.
        raise GgufFormatError(f"unsupported GGUF version {version}")
    tensor_count = _u64(f)
    kv_count = _u64(f)
    return version, tensor_count, kv_count


def read_kv(
    path: str | Path,
    want: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """KV metadata from a GGUF header.

    `want(key)` decides which values are decoded; everything else is skipped
    over without being read. With no filter every value is decoded, vocabulary
    included — so pass one.

    Returns {} for anything that is not a readable GGUF.
    """
    out: dict[str, Any] = {}
    try:
        with open(path, "rb") as f:
            _, _, kv_count = _open_header(f)
            for _ in range(kv_count):
                key = _read_str(f)
                vtype = _u32(f)
                keep = want is None or want(key)
                value = _read_value(f, vtype, keep)
                if keep:
                    out[key] = value
    except (OSError, GgufFormatError, struct.error) as e:
        log.warning("could not read GGUF metadata from %s: %s", path, e)
        return {}
    return out


def _quant_sizes() -> dict[int, tuple[int, int]]:
    """ggml type id -> (block size in elements, bytes per block)."""
    from gguf.constants import GGML_QUANT_SIZES

    return {int(k): v for k, v in GGML_QUANT_SIZES.items()}


def read_tensor_index(path: str | Path) -> list[tuple[str, int]]:
    """[(tensor name, size in bytes)] from a GGUF header.

    The KV block has to be walked to reach the tensor index, but none of it is
    decoded — this is the cheap way to ask "how many bytes are the expert
    tensors", which is what the MoE offload advice is built on.

    Returns [] for anything that is not a readable GGUF.
    """
    sizes = _quant_sizes()
    out: list[tuple[str, int]] = []
    try:
        with open(path, "rb") as f:
            _, tensor_count, kv_count = _open_header(f)
            for _ in range(kv_count):
                _skip_str(f)
                _read_value(f, _u32(f), False)
            for _ in range(tensor_count):
                name = _read_str(f)
                n_dims = _u32(f)
                if n_dims > 8:
                    raise GgufFormatError(f"implausible tensor rank {n_dims}")
                n_elements = 1
                for _ in range(n_dims):
                    n_elements *= _u64(f)
                ggml_type = _u32(f)
                _u64(f)  # data offset — not needed, the file is never read past here
                block_size, type_size = sizes.get(ggml_type, (1, 0))
                if not type_size or not block_size:
                    log.debug("unknown ggml type %s for tensor %s", ggml_type, name)
                    continue
                out.append((name, (n_elements // block_size) * type_size))
    except (OSError, GgufFormatError, struct.error) as e:
        log.warning("could not read GGUF tensor index from %s: %s", path, e)
        return []
    return out


def prefix_suffix_filter(
    prefixes: Iterable[str] = (), suffixes: Iterable[str] = (),
) -> Callable[[str], bool]:
    """`want` predicate matching keys by prefix or suffix.

    Architecture-scoped keys are named `<arch>.block_count`, and the
    architecture is itself one of the values, so the geometry keys can only be
    matched by suffix.
    """
    pre, suf = tuple(prefixes), tuple(suffixes)

    def want(key: str) -> bool:
        return (bool(pre) and key.startswith(pre)) or (bool(suf) and key.endswith(suf))

    return want
