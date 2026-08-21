"""Corrects the VRAM estimate against what the card actually reported.

The estimator adds up file bytes, a KV-cache formula and a flat compute
allowance. That is close enough to steer by, but it is systematically high for
some architectures — a DeepSeek-V4 MLA model came out ~2.1 GB over on every
measurement, because the KV formula assumes a full k/v cache where llama.cpp
keeps a compressed latent one, and the flat 1 GB compute allowance is generous
for a model whose experts live in RAM.

Being 2 GB high is not a rounding error: it makes fit_check block a preset the
user has already run successfully, which teaches them to ignore it.

So: once a preset has been up long enough for its allocations to settle, store
`measured - estimated` for that model. Later estimates for the same model are
shifted by it. The correction is per model_path, not per config, because the
error is a constant offset — it does not scale with how the experts are split
(verified across two offload settings on the same model).

Measurements are kept per (model, vendor). The same GGUF runs on the 5090
through CUDA and on the R9700 through Vulkan, and the two allocators do not
make the same error — applying one card's correction to the other is how a
"won't fit" turns into an out-of-memory abort at load.

Nothing here invents data: with no measurement the offset is 0 and the
estimate is used unchanged.
"""
from __future__ import annotations

import logging
import time

from .db import connect

log = logging.getLogger(__name__)

# Don't trust a measurement taken before the model has finished loading and
# the allocator has settled.
MIN_UPTIME_S = 90.0
# A correction larger than this means the estimate is wrong in kind, not by a
# constant — better to leave it visible than to paper over it.
MAX_ABS_OFFSET_MB = 8192

# (model_path, vendor) -> offset_mb, read once per process and updated on write.
_cache: dict[tuple[str, str], int] = {}
_loaded = False

_CREATE = """
    CREATE TABLE IF NOT EXISTS vram_calibration (
        model_path TEXT NOT NULL,
        vendor TEXT NOT NULL DEFAULT 'nvidia',
        offset_mb INTEGER NOT NULL,
        measured_mb INTEGER NOT NULL,
        estimated_mb INTEGER NOT NULL,
        measured_at REAL NOT NULL,
        PRIMARY KEY (model_path, vendor)
    )
"""


async def _has_table(db, name: str) -> bool:
    async with db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ) as cur:
        return await cur.fetchone() is not None


async def ensure_table() -> None:
    """Create the table, and fold in rows from the pre-vendor schema.

    Rows written before AMD could be measured are keyed by model_path alone
    and are all nvidia-smi readings by construction. SQLite cannot widen a
    primary key in place, so the table is rebuilt around them.

    The copy is a separate step from the rename on purpose: SQLite runs DDL
    outside the surrounding transaction, so a process killed mid-migration can
    leave the old rows sitting in `_v1`. Folding in whatever is there on every
    startup makes that recoverable instead of a silent loss of every
    measurement the user has accumulated.
    """
    async with connect() as db:
        await db.execute(_CREATE)
        async with db.execute("PRAGMA table_info(vram_calibration)") as cur:
            columns = {row[1] for row in await cur.fetchall()}
        if "vendor" not in columns:
            await db.execute("ALTER TABLE vram_calibration RENAME TO vram_calibration_v1")
            await db.execute(_CREATE)
        if await _has_table(db, "vram_calibration_v1"):
            await db.execute(
                """
                INSERT OR IGNORE INTO vram_calibration
                    (model_path, vendor, offset_mb, measured_mb, estimated_mb, measured_at)
                SELECT model_path, 'nvidia', offset_mb, measured_mb, estimated_mb, measured_at
                FROM vram_calibration_v1
                """
            )
            await db.execute("DROP TABLE vram_calibration_v1")
            log.info("vram calibration: folded the pre-vendor rows into the new table")
        await db.commit()


async def load() -> None:
    global _loaded
    try:
        await ensure_table()
        async with connect() as db:
            async with db.execute(
                "SELECT model_path, vendor, offset_mb FROM vram_calibration"
            ) as cur:
                async for row in cur:
                    _cache[(row[0], row[1])] = int(row[2])
    except Exception as e:  # noqa: BLE001 — calibration is an optimisation, never fatal
        log.warning("vram calibration: could not load: %s", e)
    _loaded = True


def _entry(model_path: str | None, device_ids: list[str] | None) -> int | None:
    """The measurement that applies to this plan, or None when there is none.

    An unresolvable vendor ("" — no pin, nothing probed yet, or a pin spanning
    two cards) may still use a measurement, but only while the model has been
    measured on exactly one vendor: on a single-GPU box that is the ordinary
    case, and where two vendors have both measured it there is no way to tell
    which one this plan will hit.
    """
    if not model_path:
        return None
    from .devices import pin_vendor

    vendor = pin_vendor(device_ids)
    if vendor:
        return _cache.get((model_path, vendor))
    rows = [v for (path, _vendor), v in _cache.items() if path == model_path]
    return rows[0] if len(rows) == 1 else None


def offset_mb(model_path: str | None, device_ids: list[str] | None = None) -> int:
    """How much to subtract from a GPU estimate for this model. 0 when unknown."""
    return _entry(model_path, device_ids) or 0


def is_measured(model_path: str | None, device_ids: list[str] | None = None) -> bool:
    """Whether this plan's memory number is backed by a real measurement.

    The difference matters to the safety margin the planner adds: a formula
    that has never been checked against this model can be ~2 GB out in either
    direction, while a measured one is only exposed to allocator drift.
    """
    return _entry(model_path, device_ids) is not None


async def record(
    model_path: str, measured_mb: int, estimated_mb: int, vendor: str = "nvidia"
) -> None:
    """Note what the card actually used for a model we estimated."""
    if not model_path or measured_mb <= 0 or estimated_mb <= 0:
        return
    vendor = vendor or "nvidia"
    offset = estimated_mb - measured_mb
    if abs(offset) > MAX_ABS_OFFSET_MB:
        log.warning(
            "vram calibration: ignoring %s — estimate off by %d MB, too large to be an offset",
            model_path, offset,
        )
        return
    if _cache.get((model_path, vendor)) == offset:
        return
    _cache[(model_path, vendor)] = offset
    try:
        await ensure_table()
        async with connect() as db:
            await db.execute(
                """
                INSERT INTO vram_calibration
                    (model_path, vendor, offset_mb, measured_mb, estimated_mb, measured_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_path, vendor) DO UPDATE SET
                    offset_mb=excluded.offset_mb,
                    measured_mb=excluded.measured_mb,
                    estimated_mb=excluded.estimated_mb,
                    measured_at=excluded.measured_at
                """,
                (model_path, vendor, offset, measured_mb, estimated_mb, time.time()),
            )
            await db.commit()
        log.info(
            "vram calibration: %s on %s estimated %d MB, measured %d MB (offset %+d MB)",
            model_path, vendor, estimated_mb, measured_mb, -offset,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("vram calibration: could not persist: %s", e)
