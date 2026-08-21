"""ModelRegistry: scan configured roots for .gguf files, classify family and
quant from the filename, detect sibling mmproj projectors, persist to SQLite.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from .db import connect
from .vram_estimate import split_shards

log = logging.getLogger(__name__)

# Files we never want to surface — these are llama.cpp test vocabularies,
# or they're mmproj projectors (surfaced via the has_mmproj flag on their sibling instead).
EXCLUDE_PREFIXES = ("ggml-vocab-",)

# Directory names to skip recursively — caches, in-flight downloads, OS junk.
SKIP_DIRS = {".cache", ".git", "__pycache__", ".ipynb_checkpoints", "node_modules"}


# Family detection. Order matters — first match wins.
_FAMILY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"Qwen3\.6", re.I), "Qwen3.6"),
    (re.compile(r"Qwen3\.5", re.I), "Qwen3.5"),
    (re.compile(r"Qwen2\.5[-_.]?Coder", re.I), "Qwen2.5-Coder"),
    (re.compile(r"Qwen2\.5", re.I), "Qwen2.5"),
    (re.compile(r"Qwen3", re.I), "Qwen3"),
    (re.compile(r"Qwen2", re.I), "Qwen2"),
    (re.compile(r"QwQ", re.I), "QwQ"),
    (re.compile(r"Gemma-?4", re.I), "Gemma-4"),
    (re.compile(r"Gemma-?3", re.I), "Gemma-3"),
    (re.compile(r"Gemma-?2", re.I), "Gemma-2"),
    (re.compile(r"Gemma", re.I), "Gemma"),
    (re.compile(r"Llama-?3\.?1", re.I), "Llama-3.1"),
    (re.compile(r"Llama-?3", re.I), "Llama-3"),
    (re.compile(r"Llama-?2", re.I), "Llama-2"),
    (re.compile(r"Ministral", re.I), "Ministral"),
    (re.compile(r"Mixtral", re.I), "Mixtral"),
    (re.compile(r"Mistral", re.I), "Mistral"),
    (re.compile(r"DeepSeek-?R1", re.I), "DeepSeek-R1"),
    (re.compile(r"DeepSeek", re.I), "DeepSeek"),
    (re.compile(r"OLMo-?3", re.I), "OLMo-3"),
    (re.compile(r"OLMo", re.I), "OLMo"),
    (re.compile(r"Phi-?4", re.I), "Phi-4"),
    (re.compile(r"Phi-?3", re.I), "Phi-3"),
    (re.compile(r"Phi", re.I), "Phi"),
    (re.compile(r"gpt-oss", re.I), "gpt-oss"),
    (re.compile(r"Hy[-_]?3|Hy[-_]?V3|Hunyuan", re.I), "Hunyuan-V3"),
]

# Match common GGUF quant suffixes. Checks from more-specific to less.
_QUANT_PATTERNS = [
    re.compile(r"(UD-?Q[0-9]+_K_[A-Z]+)", re.I),   # UD-Q4_K_XL
    re.compile(r"(MXFP4(?:[-_][A-Z]+)?)", re.I),    # MXFP4_MOE or bare MXFP4
    re.compile(r"(IQ[0-9]+[-_][A-Z]+(?:_[A-Z]+)?)", re.I),  # IQ4_XS
    re.compile(r"(Q[0-9]+_[A-Z]+(?:_[A-Z]+)?)", re.I),       # Q4_K_M, Q5_K_S, Q6_K
    re.compile(r"(Q[0-9]+_[0-9])", re.I),           # Q8_0, Q4_0
    re.compile(r"(BF16|F16|F32)", re.I),
]


@dataclass
class ModelEntry:
    path: str
    family: str | None
    quant: str | None
    size_bytes: int
    mtime: float
    has_mmproj: bool
    mmproj_path: str | None
    last_used: float | None = None


def _is_mmproj(name: str) -> bool:
    lower = name.lower()
    return lower.startswith("mmproj") or "mmproj" in lower.split(".")[0]


def _excluded(name: str) -> bool:
    return any(name.startswith(p) for p in EXCLUDE_PREFIXES) or _is_mmproj(name)


def _detect_family(name: str) -> str | None:
    for pat, family in _FAMILY_PATTERNS:
        if pat.search(name):
            return family
    return None


def _detect_quant(name: str) -> str | None:
    for pat in _QUANT_PATTERNS:
        m = pat.search(name)
        if m:
            return m.group(1).upper()
    return None


# `<prefix>-00002-of-00004.gguf` — a continuation of a split model, not a
# model. Only part 1 can be handed to llama.cpp; it finds the rest itself.
_SPLIT_CONTINUATION = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)


def _split_continuation(name: str) -> bool:
    m = _SPLIT_CONTINUATION.search(name)
    return bool(m) and int(m.group(1)) > 1 and int(m.group(2)) > 1


def _split_set_bytes(path: Path) -> int | None:
    """Total size of a split set, for the part-1 entry that stands in for it.

    Part 1 on its own can be a few MB of metadata, so listing its file size
    makes a 96 GB model read as "0.0 GB" in the picker."""
    shards = split_shards(str(path))
    if len(shards) == 1:
        return None
    # Sum whatever is actually on disk. An incomplete set then reads as the
    # bytes it really has rather than as part 1's few MB of metadata; that it
    # is incomplete is fit_check's job to say, and it does, by name.
    total = 0
    for s in shards:
        try:
            total += Path(s).stat().st_size
        except OSError:
            continue
    return total or None


def _find_sibling_mmproj(gguf_path: Path) -> Path | None:
    try:
        for sibling in gguf_path.parent.iterdir():
            if sibling.is_file() and sibling.suffix == ".gguf" and _is_mmproj(sibling.name):
                return sibling
    except (OSError, PermissionError):
        pass
    return None


def _walk_ggufs(root: Path):
    """Iterator over .gguf files under root, pruning SKIP_DIRS and warning on
    malformed names like 'foo.gguf?download=true'."""
    import os
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".cache")]
        for fn in filenames:
            lower = fn.lower()
            if ".gguf?" in lower or ".gguf%" in lower:
                log.warning(
                    "skipping malformed GGUF filename (URL suffix detected): %s — rename to plain .gguf to include it",
                    Path(dirpath) / fn,
                )
                continue
            if fn.endswith(".gguf"):
                yield Path(dirpath) / fn


def scan_roots(roots: list[str]) -> list[ModelEntry]:
    """Walk filesystem, return a list of ModelEntry (not yet persisted)."""
    seen: set[Path] = set()
    out: list[ModelEntry] = []
    for root_str in roots:
        root = Path(root_str)
        if not root.exists():
            log.warning("scan root does not exist: %s", root)
            continue
        for p in _walk_ggufs(root):
            try:
                p = p.resolve()
            except (OSError, RuntimeError):
                continue
            if p in seen:
                continue
            if _excluded(p.name) or _split_continuation(p.name):
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            seen.add(p)
            mmproj = _find_sibling_mmproj(p)
            out.append(ModelEntry(
                path=str(p),
                family=_detect_family(p.name),
                quant=_detect_quant(p.name),
                size_bytes=_split_set_bytes(p) or st.st_size,
                mtime=st.st_mtime,
                has_mmproj=mmproj is not None,
                mmproj_path=str(mmproj) if mmproj else None,
            ))
    return out


async def persist(entries: list[ModelEntry]) -> dict:
    """Upsert entries into SQLite `models`, preserving last_used timestamps."""
    added = 0
    updated = 0
    async with connect() as db:
        for e in entries:
            cur = await db.execute(
                "SELECT size_bytes, mtime, family, quant, has_mmproj FROM models WHERE path=?",
                (e.path,),
            )
            existing = await cur.fetchone()
            if existing is None:
                await db.execute(
                    """INSERT INTO models
                       (path, family, quant, size_bytes, mtime, has_mmproj, mmproj_path, last_used)
                       VALUES (?, ?, ?, ?, ?, ?, ?, NULL)""",
                    (e.path, e.family, e.quant, e.size_bytes, e.mtime, int(e.has_mmproj), e.mmproj_path),
                )
                added += 1
            elif (
                existing["size_bytes"] != e.size_bytes
                or existing["mtime"] != e.mtime
                or existing["family"] != e.family
                or existing["quant"] != e.quant
                or bool(existing["has_mmproj"]) != e.has_mmproj
            ):
                # Re-classify on every metadata change — including family/quant
                # regex tweaks that don't move the file. Without this clause,
                # adding a new family pattern (e.g. Gemma-4) wouldn't fix
                # already-scanned files until the user manually deleted them.
                await db.execute(
                    """UPDATE models SET family=?, quant=?, size_bytes=?, mtime=?, has_mmproj=?, mmproj_path=?
                       WHERE path=?""",
                    (e.family, e.quant, e.size_bytes, e.mtime, int(e.has_mmproj), e.mmproj_path, e.path),
                )
                updated += 1
        # Prune entries that were previously scanned but no longer exist on disk.
        current_paths = {e.path for e in entries}
        cur = await db.execute("SELECT path FROM models")
        existing_paths = {row["path"] for row in await cur.fetchall()}
        stale = existing_paths - current_paths
        for p in stale:
            await db.execute("DELETE FROM models WHERE path=?", (p,))
        await db.commit()
    return {"added": added, "updated": updated, "removed": len(stale), "total": len(entries)}


async def list_models(family: str | None = None) -> list[dict]:
    async with connect() as db:
        if family:
            cur = await db.execute(
                "SELECT * FROM models WHERE family=? ORDER BY family, size_bytes DESC",
                (family,),
            )
        else:
            cur = await db.execute("SELECT * FROM models ORDER BY family, size_bytes DESC")
        rows = await cur.fetchall()
    return [_row_to_dict(r) for r in rows]


async def mark_used(path: str) -> None:
    async with connect() as db:
        await db.execute("UPDATE models SET last_used=? WHERE path=?", (time.time(), path))
        await db.commit()


def _row_to_dict(row: aiosqlite.Row) -> dict:
    return {
        "path": row["path"],
        "family": row["family"],
        "quant": row["quant"],
        "size_bytes": row["size_bytes"],
        "size_gb": round(row["size_bytes"] / 1024**3, 2),
        "mtime": row["mtime"],
        "has_mmproj": bool(row["has_mmproj"]),
        "mmproj_path": row["mmproj_path"],
        "last_used": row["last_used"],
    }


async def full_rescan(roots: list[str]) -> dict:
    """Walk the roots and persist the result.

    The walk runs in a worker thread. It is ordinary blocking filesystem work —
    os.walk plus a stat() per file, and a stat() per shard for split sets — and
    on a root with a few hundred GGUFs, or one that lives on a network mount,
    it stalls the whole event loop: this runs at boot, so the health endpoint
    does not answer and the launcher's readiness poll sits there waiting.
    """
    import asyncio

    entries = await asyncio.to_thread(scan_roots, roots)
    return await persist(entries)
