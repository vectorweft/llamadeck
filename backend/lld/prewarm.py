"""Page-cache pre-warming for CPU-offloaded model tensors.

llama.cpp keeps the MoE expert weights of the first `--n-cpu-moe N` layers in
system memory via mmap, faulting pages in lazily from the GGUF files on first
touch. After a (re)load — or after the page cache has been churned by other
big model loads — every generated token can stall on disk reads instead of
computing: measured here as ~1 t/s decode vs ~19 t/s once the pages are
resident. The same warning shows up in llama-server's own output:

    tensor overrides to CPU are used with mmap enabled - consider using
    --load-mode none for better performance

Pre-warming the *exact byte ranges* of the CPU-held expert tensors right after
the model reports "loaded" brings decode back to full speed without changing
how the model is served.

The warm is deliberately targeted, not whole-file: a 96 GiB model with
--n-cpu-moe 32 keeps ~66 GiB of expert tensors in system memory. That fits
this machine's 89 GiB RAM and stays resident once faulted in. Reading the
whole 103 GiB file instead would evict its own head (LRU) and leave the first
blocks cold again — the exact tensors the CPU needs.
"""
from __future__ import annotations

import logging
import os
import re
import struct
from pathlib import Path

from .gguf_header import (
    GgufFormatError,
    _open_header,
    _quant_sizes,
    _read_str,
    _read_value,
    _skip_str,
    _u32,
    _u64,
)
from .model_defaults import _read_gguf_cached
from .vram_estimate import split_shards

log = logging.getLogger(__name__)

# Matches the MoE expert tensors llama.cpp's --n-cpu-moe override regex covers:
# ffn_(up|down|gate_up|gate)_(ch)?exps — but NOT the shared-expert tensors
# (ffn_*_shexp), which stay on the GPU.
_EXPERT_TENSOR_RE = re.compile(r"blk\.(\d+)\.ffn_(?:up|down|gate_up|gate)_(?:ch)?exps\.weight$")

#: Sentinel returned by cpu_moe_from_args() for --cpu-moe (all layers).
ALL_CPU_MOE = -1


def tensor_ranges(path: str | Path) -> dict[str, tuple[int, int]]:
    """GGUF tensor table of one shard: tensor name -> (file offset, byte size).

    Sizes come from ggml's quant tables (same math as gguf_header's
    read_tensor_index); offsets are the per-tensor data offsets the index
    records. Returns {} for anything that is not a readable GGUF.
    """
    sizes = _quant_sizes()
    out: dict[str, tuple[int, int]] = {}
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
                offset = _u64(f)
                block_size, type_size = sizes.get(ggml_type, (1, 0))
                if not type_size or not block_size:
                    continue
                size = (n_elements // block_size) * type_size
                if size > 0:
                    out[name] = (offset, size)
    except (OSError, GgufFormatError, struct.error) as e:
        log.warning("prewarm: could not read GGUF tensor index from %s: %s", path, e)
        return {}
    return out


def cpu_expert_ranges(model_path: str, n_cpu_moe: int) -> list[tuple[str, int, int]]:
    """(shard path, offset, length) for every expert tensor of the first
    `n_cpu_moe` layers — the tensors `--n-cpu-moe N` keeps in system memory.

    `n_cpu_moe` may be ALL_CPU_MOE (--cpu-moe): then every expert tensor of
    the model is included.
    """
    if n_cpu_moe == 0:
        return []
    ranges: list[tuple[str, int, int]] = []
    for shard in split_shards(model_path):
        if not os.path.isfile(shard):
            continue
        for name, (offset, size) in tensor_ranges(shard).items():
            m = _EXPERT_TENSOR_RE.match(name)
            if not m:
                continue
            block = int(m.group(1))
            if n_cpu_moe == ALL_CPU_MOE or block < n_cpu_moe:
                ranges.append((shard, offset, size))
    return ranges


def warm_ranges(ranges: list[tuple[str, int, int]], chunk: int = 1 << 20) -> int:
    """Bring the given byte ranges into the page cache.

    Async WILLNEED kicks the kernel's readahead; a synchronous pread then
    guarantees the pages are resident (and their contents valid) by the time
    we return. Returns the number of bytes warmed.
    """
    if not ranges:
        return 0
    total = 0
    by_file: dict[str, list[tuple[int, int]]] = {}
    for path, offset, size in ranges:
        by_file.setdefault(path, []).append((offset, size))
    for path, rs in by_file.items():
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError as e:
            log.warning("prewarm: cannot open %s: %s", path, e)
            continue
        try:
            for offset, size in rs:
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(fd, offset, size, os.POSIX_FADV_WILLNEED)
                pos, end = offset, offset + size
                while pos < end:
                    got = os.pread(fd, min(chunk, end - pos), pos)
                    if not got:
                        break
                    pos += len(got)
                total += size
        finally:
            os.close(fd)
    return total


def cpu_moe_from_args(args: list[str]) -> int:
    """The --n-cpu-moe / --cpu-moe value from a llama-server argv, as spawned.
    0 when the model does no MoE CPU offload; ALL_CPU_MOE for --cpu-moe."""
    n = 0
    for i, tok in enumerate(args):
        if tok in ("--cpu-moe", "-cmoe"):
            return ALL_CPU_MOE
        if tok in ("--n-cpu-moe", "-ncmoe") and i + 1 < len(args):
            try:
                n = max(n, int(args[i + 1]))
            except ValueError:
                pass
    return n


def _model_path_from_args(args: list[str]) -> str | None:
    for i, tok in enumerate(args):
        if tok in ("--model", "-m") and i + 1 < len(args):
            return args[i + 1]
    return None


def _uses_no_mmap(args: list[str]) -> bool:
    """True when the server was told to skip mmap (--no-mmap / --load-mode
    none|dio): weights then live in anonymous memory and warming the page
    cache would only waste I/O."""
    for i, tok in enumerate(args):
        if tok in ("--no-mmap", "--direct-io", "-dio"):
            return True
        if tok in ("-lm", "--load-mode") and i + 1 < len(args):
            if args[i + 1] in ("none", "dio"):
                return True
    return False


def _block_count(model_path: str) -> int:
    """Number of layers the model declares (0 when the header is unreadable)."""
    try:
        meta = _read_gguf_cached(Path(model_path))
    except Exception:  # noqa: BLE001 - never fail a warm over metadata
        return 0
    arch = meta.get("general.architecture", "")
    return int(meta.get(f"{arch}.block_count") or 0)


def prewarm_model(model_path: str, n_cpu_moe: int) -> dict:
    """Warm the CPU-offloaded expert tensors of `model_path`.

    Safe to call on any model: non-MoE models and n_cpu_moe <= 0 warm nothing.
    Returns a stats dict for logging/tests.
    """
    if n_cpu_moe == 0 or not model_path:
        return {"n_cpu_moe": n_cpu_moe, "ranges": 0, "bytes": 0}
    ranges = cpu_expert_ranges(model_path, n_cpu_moe)
    if not ranges:
        return {"n_cpu_moe": n_cpu_moe, "ranges": 0, "bytes": 0}
    bytes_warmed = warm_ranges(ranges)
    log.info(
        "prewarm: %d ranges, %.1f GiB of CPU-offloaded experts for %s",
        len(ranges), bytes_warmed / 2**30, Path(model_path).name,
    )
    return {"n_cpu_moe": n_cpu_moe, "ranges": len(ranges), "bytes": bytes_warmed}


def prewarm_from_args(args: list[str]) -> dict:
    """Warm using a spawned llama-server argv (router /models status.args).

    No-op when the server was launched with --no-mmap / --load-mode none:
    those load the weights into anonymous memory, which the page cache does
    not govern.
    """
    if _uses_no_mmap(args):
        return {"n_cpu_moe": 0, "ranges": 0, "bytes": 0, "skipped": "no-mmap"}
    model_path = _model_path_from_args(args)
    n_cpu_moe = cpu_moe_from_args(args)
    if n_cpu_moe == ALL_CPU_MOE:
        n_cpu_moe = _block_count(model_path) if model_path else 0
    return prewarm_model(model_path or "", n_cpu_moe)


def prewarm_from_config(config: dict) -> dict:
    """Warm using a preset config dict (supervisor status["config"])."""
    model_path = config.get("model_path")
    if not model_path:
        return {"n_cpu_moe": 0, "ranges": 0, "bytes": 0}
    flags = config.get("extra_flags") or []
    if _uses_no_mmap(flags):
        return {"n_cpu_moe": 0, "ranges": 0, "bytes": 0, "skipped": "no-mmap"}
    n = 0
    for i, tok in enumerate(flags):
        if tok in ("--cpu-moe", "-cmoe"):
            n = _block_count(model_path)
        elif tok in ("--n-cpu-moe", "-ncmoe") and i + 1 < len(flags):
            try:
                n = max(n, int(flags[i + 1]))
            except ValueError:
                pass
    return prewarm_model(model_path, min(n, _block_count(model_path)) or 0)
