"""Dynamic VRAM estimator for a LlamaServerConfig.

The static `estimated_vram_mb` baked into a preset is fine while ctx/np/kv
quant match the value the author measured against, but it drifts the moment
those are tuned (e.g. ctx 65536 -> 156000 multiplies KV cache by ~2.4x).
This module computes a live estimate from:

  model weights : GGUF file size on disk (+ mmproj if present)
  KV cache      : ctx_size * n_layers * n_kv_heads * (k_dim + v_dim)
                  * bytes_per_element(cache_type)
  compute buf   : flat 1024 MB headroom (graph + logits + scratch)

GGUF geometry (n_layers, n_kv_heads, head_dim) is read once per (path, mtime)
and cached in process. If the GGUF can't be parsed, KV size falls back to a
coarse heuristic from the file size, and the result is flagged source="approx".

Notes:
  * llama.cpp -c is the *total* KV span shared across --parallel slots, so
    KV size scales with ctx_size only — np doesn't multiply KV. The exception
    is sliding-window layers (Gemma 3/4): those hold `sliding_window` tokens
    per *slot*, so there np is the multiplier and ctx_size is not.
  * flash_attn changes working memory, not KV size.
  * Quantized KV uses 17/16 bytes/element for q8_0, 18/32 for q4_0, etc.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import re

from . import vram_calib
from .gguf_header import read_kv, read_tensor_index
from .model_defaults import _read_gguf_sync
from .settings import LlamaServerConfig

log = logging.getLogger(__name__)


# bytes per scalar in a llama.cpp KV cache, per cache_type_{k,v} value.
# q*_0 / q*_1 are block-quantized with a small per-block overhead (scale,
# optional min) — values below are the effective bytes per element.
_KV_BYTES_PER_ELEM: dict[str, float] = {
    "f32":   4.0,
    "f16":   2.0,
    "bf16":  2.0,
    "q8_0":  17 / 16,    # 1.0625
    "q8_1":  18 / 16,    # 1.125
    "q5_0":  22 / 32,    # 0.6875
    "q5_1":  24 / 32,    # 0.75
    "q4_0":  18 / 32,    # 0.5625
    "q4_1":  20 / 32,    # 0.625
    "iq4_nl": 18 / 32,
}


@dataclass
class VramEstimate:
    total_mb: int
    model_mb: int
    kv_cache_mb: int
    compute_mb: int
    source: str          # "computed" | "approx" | "unavailable"
    details: dict[str, Any]
    # GPU/RAM split of total_mb given the config's offload flags
    # (--cpu-moe / --n-cpu-moe, or a partial/zero -ngl). gpu_mb is what the
    # card/VRAM budget should show; ram_mb is the host-RAM share.
    gpu_mb: int = 0
    ram_mb: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# (path, mtime_ns) -> geometry dict
_GEOMETRY_CACHE: dict[tuple[str, int], dict[str, int] | None] = {}


def _kv_bytes_per_elem(cache_type: str) -> float:
    return _KV_BYTES_PER_ELEM.get((cache_type or "f16").lower(), 2.0)


def _read_geometry(model_path: str) -> dict[str, int] | None:
    """Read attention + SSM geometry from a GGUF, cached by mtime.

    Returned keys (when available):
      n_layers, n_kv_heads, k_dim, v_dim       — attention geometry
      kv_heads_per_layer                       — per-layer head_count_kv when
                                                 the GGUF publishes an array
      swa_mask, sliding_window,
      k_dim_swa, v_dim_swa                     — sliding-window attention
                                                 (Gemma 3/4): which layers use
                                                 a window-sized cache instead
                                                 of the full ctx
      full_attention_interval                  — present on hybrid models
                                                 (Qwen3-Next, Mamba2-style):
                                                 only every Nth layer keeps a
                                                 full KV cache
      ssm_state_size, ssm_inner_size,
      ssm_conv_kernel                          — SSM recurrent state (per seq)
    """
    try:
        st = os.stat(model_path)
    except OSError:
        return None
    key = (model_path, st.st_mtime_ns)
    if key in _GEOMETRY_CACHE:
        return _GEOMETRY_CACHE[key]

    fields = _read_gguf_sync(Path(model_path))
    if not fields:
        _GEOMETRY_CACHE[key] = None
        return None

    n_layers: int | None = None
    n_kv_heads: int | None = None
    n_heads: int | None = None
    k_dim: int | None = None
    v_dim: int | None = None
    k_dim_swa: int | None = None
    v_dim_swa: int | None = None
    sliding_window: int | None = None
    embedding_length: int | None = None
    full_attention_interval: int | None = None
    ssm_state_size: int | None = None
    ssm_inner_size: int | None = None
    ssm_conv_kernel: int | None = None
    kv_heads_per_layer: list[int] | None = None
    swa_pattern: Any = None

    for k, v in fields.items():
        # Gemma-style models publish per-layer arrays (head_count_kv varies
        # between SWA and full-attention layers). Keep those before the
        # int() coercion below drops them.
        if isinstance(v, (list, tuple)):
            if k.endswith(".attention.head_count_kv"):
                kv_heads_per_layer = [int(x) for x in v]
            elif k.endswith(".attention.sliding_window_pattern"):
                swa_pattern = list(v)
            continue
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if k.endswith(".block_count"):
            n_layers = iv
        elif k.endswith(".attention.head_count_kv"):
            n_kv_heads = iv
        elif k.endswith(".attention.head_count"):
            n_heads = iv
        elif k.endswith(".attention.key_length"):
            k_dim = iv
        elif k.endswith(".attention.value_length"):
            v_dim = iv
        elif k.endswith(".attention.key_length_swa"):
            k_dim_swa = iv
        elif k.endswith(".attention.value_length_swa"):
            v_dim_swa = iv
        elif k.endswith(".attention.sliding_window"):
            sliding_window = iv
        elif k.endswith(".attention.sliding_window_pattern"):
            swa_pattern = iv
        elif k.endswith(".embedding_length"):
            embedding_length = iv
        elif k.endswith(".full_attention_interval"):
            full_attention_interval = iv
        elif k.endswith(".ssm.state_size"):
            ssm_state_size = iv
        elif k.endswith(".ssm.inner_size"):
            ssm_inner_size = iv
        elif k.endswith(".ssm.conv_kernel"):
            ssm_conv_kernel = iv

    if (k_dim is None or v_dim is None) and embedding_length and n_heads:
        derived = embedding_length // n_heads
        k_dim = k_dim or derived
        v_dim = v_dim or derived

    if n_kv_heads is None and kv_heads_per_layer:
        n_kv_heads = max(kv_heads_per_layer)
    if n_kv_heads is None and n_heads is not None:
        n_kv_heads = n_heads

    if not (n_layers and n_kv_heads and k_dim and v_dim):
        _GEOMETRY_CACHE[key] = None
        return None

    geo: dict[str, Any] = {
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "k_dim": k_dim,
        "v_dim": v_dim,
    }
    if full_attention_interval and full_attention_interval > 1:
        geo["full_attention_interval"] = full_attention_interval

    # Sliding-window attention (Gemma 3/4, Ministral, ...): most layers keep
    # only a `sliding_window`-sized cache per sequence instead of the full ctx,
    # often with their own head count and head dim. Normalise whatever the GGUF
    # published into a per-layer boolean mask so the KV maths below is exact.
    swa_mask: list[bool] | None = None
    if isinstance(swa_pattern, (list, tuple)) and len(swa_pattern) == n_layers:
        swa_mask = [bool(x) for x in swa_pattern]
    elif isinstance(swa_pattern, int) and swa_pattern > 1:
        # Gemma 3 style: every Nth layer is full attention, the rest are SWA.
        swa_mask = [(i + 1) % swa_pattern != 0 for i in range(n_layers)]
    if swa_mask and sliding_window and any(swa_mask) and not all(swa_mask):
        geo["swa_mask"] = swa_mask
        geo["sliding_window"] = sliding_window
        geo["k_dim_swa"] = k_dim_swa or k_dim
        geo["v_dim_swa"] = v_dim_swa or v_dim
    if kv_heads_per_layer and len(kv_heads_per_layer) == n_layers:
        geo["kv_heads_per_layer"] = kv_heads_per_layer
    if ssm_state_size:
        geo["ssm_state_size"] = ssm_state_size
    if ssm_inner_size:
        geo["ssm_inner_size"] = ssm_inner_size
    if ssm_conv_kernel:
        geo["ssm_conv_kernel"] = ssm_conv_kernel
    _GEOMETRY_CACHE[key] = geo
    return geo


# ---- Split (multi-part) GGUF handling --------------------------------------
# llama.cpp splits are named `<prefix>-00001-of-000NN.gguf`, and the loader
# derives its siblings from whatever path it was given: it strips this suffix
# and re-expands it for every index. So the filename — not the directory
# listing — decides which files get opened.
_SPLIT_PATH_RE = re.compile(r"^(?P<prefix>.*)-(?P<idx>\d{5})-of-(?P<count>\d{5})\.gguf$", re.I)


def split_shards(path: str) -> list[str]:
    """Every shard of a split GGUF, in load order, derived the way llama.cpp
    derives them. A single-file model returns just `[path]`.

    Paths are returned whether or not they exist — that a shard is missing is
    a diagnosis of its own, see `missing_shards()`."""
    m = _SPLIT_PATH_RE.match(path)
    if not m:
        return [path]
    count = int(m.group("count"))
    if not 1 < count <= 999:
        return [path]
    prefix = m.group("prefix")
    return [f"{prefix}-{i:05d}-of-{count:05d}.gguf" for i in range(1, count + 1)]


def missing_shards(path: str) -> list[str]:
    """Shards llama.cpp will look for and not find. Non-empty means the server
    dies during load with a bare `exited with code 1`, so callers should say
    which file is missing before anything is started."""
    return [p for p in split_shards(path) if not os.path.isfile(p)]


def rival_split_prefix(path: str) -> str | None:
    """A *different* split prefix sitting in the same folder, if one is more
    complete than ours.

    The nastiest way to hold a split model is with the parts named
    inconsistently — part 1 from one release (`Model-0731-…-00001-of-00004`)
    next to parts 2-4 from another (`Model-…-00002-of-00004`). llama.cpp then
    reports part 2 as missing while it is plainly there, and the two releases
    are not interchangeable even when the tensor counts line up. Returns the
    basename prefix of the rival set so callers can name it."""
    m = _SPLIT_PATH_RE.match(path)
    if not m:
        return None
    folder = os.path.dirname(path) or "."
    ours = os.path.basename(m.group("prefix"))
    try:
        entries = os.listdir(folder)
    except OSError:
        return None
    counts: dict[str, int] = {}
    for name in entries:
        em = _SPLIT_PATH_RE.match(name)
        if em:
            counts[em.group("prefix")] = counts.get(em.group("prefix"), 0) + 1
    mine = counts.get(ours, 0)
    rivals = [(n, p) for p, n in counts.items() if p != ours and n > mine]
    if not rivals:
        return None
    return max(rivals)[1]


# ---- Expert/core tensor profile (shared with fit_check) --------------------
_EXPS_RE = re.compile(r"blk\.(\d+)\.ffn_(?:gate|down|up)_exps\.")

# ((path, mtime_ns), ...) over all shards -> tensor/metadata summary
_SPLIT_CACHE: dict[tuple[tuple[str, int], ...], dict[str, Any] | None] = {}
_SPLIT_CACHE_MAX = 32

# The same map, persisted. Parsing a GGUF tensor table costs ~7s per model here
# — GGUFReader walks every tensor entry — and an in-memory cache alone starts
# cold on every launch. supervisor.statuses() estimates VRAM for each idle
# preset, so a box with six presets paid ~40s of it at boot, with the health
# endpoint unreachable for the duration because the walk is synchronous.
#
# Keyed by path + mtime of every shard, so a re-downloaded or truncated file is
# re-read rather than trusted. Negative results are cached too: a GGUF that
# cannot be opened must not be re-opened on every restart.
_DISK_CACHE_NAME = "gguf-profile-cache.json"
_DISK_CACHE_MAX = 256
_disk_cache: dict[str, dict[str, Any] | None] | None = None


def _disk_cache_path() -> Path:
    from .settings import STATE_DIR

    return STATE_DIR / _DISK_CACHE_NAME


def _cache_key_str(key: tuple[tuple[str, int], ...]) -> str:
    return "\u0000".join(f"{p}:{m}" for p, m in key)


def _load_disk_cache() -> dict[str, dict[str, Any] | None]:
    global _disk_cache
    if _disk_cache is not None:
        return _disk_cache
    _disk_cache = {}
    try:
        raw = _disk_cache_path().read_text()
    except OSError:
        return _disk_cache
    try:
        data = json.loads(raw)
    except ValueError:
        log.warning("gguf profile cache is corrupt; ignoring it")
        return _disk_cache
    if isinstance(data, dict):
        _disk_cache = {k: v for k, v in data.items() if v is None or isinstance(v, dict)}
    return _disk_cache


def _store_disk_cache(key: str, value: dict[str, Any] | None) -> None:
    from .settings import atomic_write_text

    cache = _load_disk_cache()
    cache.pop(key, None)
    cache[key] = value
    while len(cache) > _DISK_CACHE_MAX:
        cache.pop(next(iter(cache)))
    try:
        atomic_write_text(_disk_cache_path(), json.dumps(cache))
    except OSError as e:
        log.debug("could not persist gguf profile cache: %s", e)


def read_model_profile(model_path: str) -> dict[str, Any] | None:
    """Reads expert/core tensor sizes and MoE metadata from the GGUF header.
    Even for an 80+ GB file only the header is parsed (~1-3 s); the result
    is cached by mtime.

    A split model is read shard by shard: the tensor table is spread across
    all of them and shard 1 may carry none of it, so scanning only the path we
    were handed reports a 256-expert MoE as having no experts at all — and the
    --n-cpu-moe advice that depends on it never appears."""
    shards = split_shards(model_path)
    try:
        key = tuple((p, os.stat(p).st_mtime_ns) for p in shards)
    except OSError:
        return None
    if key in _SPLIT_CACHE:
        return _SPLIT_CACHE[key]
    key_str = _cache_key_str(key)
    disk = _load_disk_cache()
    if key_str in disk:
        cached = disk[key_str]
        _SPLIT_CACHE[key] = cached
        return cached

    _WANTED_KV = (
        ".expert_count", ".expert_used_count", ".block_count", ".context_length",
    )
    fields: dict[str, Any] = {}
    exps_bytes = 0
    other_bytes = 0
    exp_layers: set[int] = set()
    try:
        for i, shard in enumerate(shards):
            if i == 0:
                # Only shard 1 carries the model's KV metadata; the rest hold
                # nothing but split.* bookkeeping.
                fields = read_kv(shard, lambda k: k.endswith(_WANTED_KV))
            for name, nb in read_tensor_index(shard):
                m = _EXPS_RE.match(name)
                if m:
                    exps_bytes += nb
                    exp_layers.add(int(m.group(1)))
                else:
                    other_bytes += nb
    except Exception as e:
        log.warning("vram-estimate: could not open GGUF %s: %s", model_path, e)
        _SPLIT_CACHE[key] = None
        _store_disk_cache(key_str, None)
        return None
    if not exps_bytes and not other_bytes:
        # No readable tensor index at all — a partial download, or a path that
        # is not a GGUF. Cache the miss so it is not re-read every restart.
        _SPLIT_CACHE[key] = None
        _store_disk_cache(key_str, None)
        return None

    def _kv_int(suffix: str) -> int | None:
        for k, v in fields.items():
            if k.endswith(suffix):
                try:
                    return int(v)
                except (TypeError, ValueError):
                    return None
        return None

    n_exp_layers = len(exp_layers)
    out: dict[str, Any] = {
        "exps_mb": exps_bytes // (1024 * 1024),
        "other_mb": other_bytes // (1024 * 1024),
        "n_exp_layers": n_exp_layers,
        "exps_per_layer_mb": (exps_bytes / n_exp_layers / (1024 * 1024)) if n_exp_layers else 0.0,
        "expert_count": _kv_int(".expert_count"),
        "expert_used_count": _kv_int(".expert_used_count"),
        "n_layers": _kv_int(".block_count"),
        "context_length": _kv_int(".context_length"),
    }
    if len(_SPLIT_CACHE) >= _SPLIT_CACHE_MAX:
        _SPLIT_CACHE.pop(next(iter(_SPLIT_CACHE)))
    _SPLIT_CACHE[key] = out
    _store_disk_cache(key_str, out)
    return out


def parse_cpu_moe_offload(cfg: LlamaServerConfig, n_exp_layers: int) -> int:
    """Reads the --cpu-moe / --n-cpu-moe N value from extra_flags.
    Returns the number of expert layers moved to CPU (0 = no offload)."""
    flags = cfg.extra_flags or []
    n = 0
    for i, tok in enumerate(flags):
        if tok in ("--cpu-moe", "-cmoe"):
            n = n_exp_layers
        elif tok in ("--n-cpu-moe", "-ncmoe") and i + 1 < len(flags):
            try:
                n = min(int(flags[i + 1]), n_exp_layers)
            except ValueError:
                pass
    return n


def _file_mb(path: str | None) -> int:
    if not path:
        return 0
    try:
        return int(os.path.getsize(path) / (1024 * 1024))
    except OSError:
        return 0


def _model_mb(path: str | None) -> int:
    """On-disk size of a model, summed over every shard of a split GGUF.

    Sizing only the path we were handed under-reports catastrophically: some
    publishers put the entire metadata blob in shard 1 and no tensors at all,
    so a 96 GB model measures 5 MB and every downstream check says "fits"."""
    if not path:
        return 0
    return sum(_file_mb(p) for p in split_shards(path))


def estimate_vram(cfg: LlamaServerConfig) -> VramEstimate | None:
    """Compute a VRAM estimate for a single-mode preset. Returns None for
    router presets (per-model loading is dynamic) or when no model_path is
    available (HF-only presets that haven't downloaded yet)."""
    if cfg.mode == "router":
        return None
    model_path = cfg.model_path
    if not model_path or not os.path.isfile(model_path):
        return None

    model_mb = _model_mb(model_path)
    mmproj_mb = _file_mb(cfg.mmproj_path)

    geo = _read_geometry(model_path)
    bytes_k = _kv_bytes_per_elem(cfg.cache_type_k)
    bytes_v = _kv_bytes_per_elem(cfg.cache_type_v)

    ssm_mb = 0
    if geo:
        # Hybrid SSM/attention models (Qwen3.6 / Qwen3-Next / Mamba2-style)
        # keep a full KV cache only on every Nth layer (full_attention_interval).
        # The other layers use a small recurrent SSM state that is per-sequence
        # rather than per-token, so it doesn't scale with ctx.
        n_layers_total = geo["n_layers"]
        fai = geo.get("full_attention_interval", 0) or 0
        if fai > 1:
            n_attn_layers = max(1, n_layers_total // fai)
            n_ssm_layers = n_layers_total - n_attn_layers
        else:
            n_attn_layers = n_layers_total
            n_ssm_layers = 0

        swa_mask = geo.get("swa_mask")
        if swa_mask:
            # SWA models: only the full-attention layers scale with ctx. The
            # sliding-window layers keep `window` tokens *per sequence*, so
            # they scale with --parallel instead — which is why bumping np on
            # a Gemma-class model costs far more VRAM than bumping ctx.
            kvh = geo.get("kv_heads_per_layer") or [geo["n_kv_heads"]] * n_layers_total
            full_per_token = sum(
                kvh[i] * (geo["k_dim"] * bytes_k + geo["v_dim"] * bytes_v)
                for i in range(n_layers_total)
                if not swa_mask[i]
            )
            swa_per_token = sum(
                kvh[i] * (geo["k_dim_swa"] * bytes_k + geo["v_dim_swa"] * bytes_v)
                for i in range(n_layers_total)
                if swa_mask[i]
            )
            n_seq = max(cfg.parallel, 1)
            # llama.cpp sizes the SWA cache at window + one ubatch per slot,
            # capped by the slot's own ctx span.
            span = min(
                geo["sliding_window"] + (cfg.ubatch_size or 512),
                max(cfg.ctx_size // n_seq, 1),
            )
            kv_bytes = cfg.ctx_size * full_per_token + n_seq * span * swa_per_token
            per_token_bytes = kv_bytes / max(cfg.ctx_size, 1)
        else:
            per_token_bytes = n_attn_layers * geo["n_kv_heads"] * (
                geo["k_dim"] * bytes_k + geo["v_dim"] * bytes_v
            )
            kv_bytes = cfg.ctx_size * per_token_bytes
        kv_mb = int(kv_bytes / (1024 * 1024))

        # SSM state: per layer, per sequence. Recurrent state is f32; keep
        # bytes_per=4 to stay conservative.
        if n_ssm_layers and geo.get("ssm_state_size") and geo.get("ssm_inner_size"):
            state = geo["ssm_state_size"] * geo["ssm_inner_size"]
            conv = (geo.get("ssm_conv_kernel") or 0) * geo["ssm_inner_size"]
            ssm_bytes = n_ssm_layers * (state + conv) * 4 * max(cfg.parallel, 1)
            ssm_mb = int(ssm_bytes / (1024 * 1024))
        source = "computed"
    else:
        # Coarse fallback: scale a per-token KV size from the model's file size
        # class. Tuned roughly for 4-bit dense models (Q4_K_XL etc):
        # ~30 bytes/token/GB-of-model at f16, halved for q8_0, etc.
        avg_kv_bytes_per_elem = (bytes_k + bytes_v) / 2
        gb = max(model_mb / 1024, 1.0)
        per_token_bytes = 30 * gb * (avg_kv_bytes_per_elem / 2.0)
        kv_bytes = cfg.ctx_size * per_token_bytes
        kv_mb = int(kv_bytes / (1024 * 1024))
        source = "approx"

    # Compute / scratch buffers — graph workspace, logits, sampling.
    # Empirically ~0.7-1.2 GB for dense 7B-70B; round to 1024.
    compute_mb = 1024

    total_mb = model_mb + mmproj_mb + kv_mb + ssm_mb + compute_mb

    # GPU/RAM split. Offload flags move weights to host RAM: --cpu-moe /
    # --n-cpu-moe park expert tensors, and a partial/zero -ngl parks whole
    # layers. Without this, MoE presets like a 90 GB model with 66 expert
    # layers in RAM show a terrifying (and wrong) "~90 GB VRAM" on the card.
    profile = read_model_profile(model_path) or {}
    n_exp_layers = int(profile.get("n_exp_layers") or 0)
    per_layer_mb = float(profile.get("exps_per_layer_mb") or 0.0)
    cpu_moe_layers = parse_cpu_moe_offload(cfg, n_exp_layers) if n_exp_layers else 0
    n_layers_meta = int(profile.get("n_layers") or (geo or {}).get("n_layers") or 0)
    ngl = cfg.n_gpu_layers
    if ngl == 0:
        ram_mb = model_mb
    elif cpu_moe_layers > 0:
        ram_mb = int(cpu_moe_layers * per_layer_mb)
    elif n_layers_meta and 0 < ngl < n_layers_meta:
        ram_mb = int(model_mb * (n_layers_meta - ngl) / n_layers_meta)
    else:
        ram_mb = 0
    ram_mb = min(ram_mb, model_mb)
    gpu_mb = total_mb - ram_mb
    # Correction learned from what the card actually reported for this model
    # last time it ran. Zero until there has been a measurement.
    calibration_mb = vram_calib.offset_mb(model_path, getattr(cfg, "devices", None))
    if calibration_mb:
        gpu_mb = max(0, gpu_mb - calibration_mb)
    # Whether that number came from a measurement rather than the formula.
    # A zero offset is not the same as "never measured" — a model whose
    # estimate happened to land exactly right still counts as measured — so
    # this is asked separately instead of inferred from calibration_mb.
    measured = vram_calib.is_measured(model_path, getattr(cfg, "devices", None))

    details: dict[str, Any] = {
        "model_path_mb": model_mb,
        "mmproj_mb": mmproj_mb,
        "ssm_state_mb": ssm_mb,
        "ctx_size": cfg.ctx_size,
        "parallel": cfg.parallel,
        "cache_type_k": cfg.cache_type_k,
        "cache_type_v": cfg.cache_type_v,
        "bytes_per_token_kv": round(per_token_bytes, 1),
        "calibration_mb": calibration_mb,
        "measured": measured,
    }
    if geo:
        details.update(geo)

    return VramEstimate(
        total_mb=total_mb,
        model_mb=model_mb + mmproj_mb,
        kv_cache_mb=kv_mb + ssm_mb,
        compute_mb=compute_mb,
        source=source,
        details=details,
        gpu_mb=gpu_mb,
        ram_mb=ram_mb,
    )
