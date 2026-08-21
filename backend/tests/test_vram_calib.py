"""Correcting the estimate against what the card actually reported.

The estimator was ~2.1 GB high on a DeepSeek-V4 MLA model — enough that
fit_check refused to start a preset the user had already been running. It was
high by the same amount at two different expert-offload settings, which is why
a single per-model offset is the right shape for the correction.
"""
from __future__ import annotations

import pytest

from lld import vram_calib

# Captured before the autouse fixture below swaps it for a no-op.
_REAL_ENSURE_TABLE = vram_calib.ensure_table


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Keep the in-process cache per-test and never touch the real database."""
    monkeypatch.setattr(vram_calib, "_cache", {})

    async def _noop():
        return None

    monkeypatch.setattr(vram_calib, "ensure_table", _noop)

    class _FakeDb:
        async def execute(self, *a, **kw):
            return None

        async def commit(self):
            return None

    class _Conn:
        async def __aenter__(self):
            return _FakeDb()

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(vram_calib, "connect", lambda: _Conn())


def test_unknown_model_has_no_correction():
    assert vram_calib.offset_mb("/m/never-run.gguf") == 0
    assert vram_calib.offset_mb(None) == 0


@pytest.mark.asyncio
async def test_offset_is_estimate_minus_measured():
    await vram_calib.record("/m/a.gguf", measured_mb=27930, estimated_mb=30076)
    assert vram_calib.offset_mb("/m/a.gguf") == 2146


@pytest.mark.asyncio
async def test_correction_carries_across_offload_settings():
    """The error is a constant, so one measurement fixes the whole slider.

    Measured 27930 at --n-cpu-moe 34 (estimate 30076) and 32280 at 32
    (estimate 34318): the same ~2.1 GB offset explains both."""
    await vram_calib.record("/m/a.gguf", measured_mb=27930, estimated_mb=30076)
    corrected_at_32 = 34318 - vram_calib.offset_mb("/m/a.gguf")
    assert abs(corrected_at_32 - 32280) < 200


@pytest.mark.asyncio
async def test_absurd_offset_is_refused():
    """An estimate off by more than a few GB is wrong in kind — recording it
    would hide a real bug behind a fudge factor."""
    await vram_calib.record("/m/a.gguf", measured_mb=1000, estimated_mb=99000)
    assert vram_calib.offset_mb("/m/a.gguf") == 0


@pytest.mark.asyncio
async def test_garbage_measurements_ignored():
    await vram_calib.record("/m/a.gguf", measured_mb=0, estimated_mb=30000)
    await vram_calib.record("", measured_mb=100, estimated_mb=200)
    assert vram_calib.offset_mb("/m/a.gguf") == 0


@pytest.mark.asyncio
async def test_later_measurement_replaces_earlier():
    await vram_calib.record("/m/a.gguf", measured_mb=27930, estimated_mb=30076)
    await vram_calib.record("/m/a.gguf", measured_mb=28500, estimated_mb=30076)
    assert vram_calib.offset_mb("/m/a.gguf") == 1576


def _devices():
    from lld import devices

    return [
        devices.LlamaDevice(id="CUDA0", name="NVIDIA GeForce RTX 5090",
                            total_mb=32000, free_mb=32000, backend="CUDA"),
        devices.LlamaDevice(id="Vulkan1", name="AMD Radeon AI PRO R9700 (RADV GFX1201)",
                            total_mb=32000, free_mb=32000, backend="Vulkan"),
    ]


def test_a_cuda_measurement_is_not_applied_to_the_radeon(monkeypatch):
    """The same GGUF runs on both cards through two different allocators, and
    they do not make the same error. Shifting a Vulkan estimate by a
    CUDA-learned ~2 GB correction turns "won't fit" into "fits" and the user
    gets an OOM abort at load instead of a warning."""
    from lld import devices, vram_calib

    monkeypatch.setitem(vram_calib._cache, ("/ml/model.gguf", "nvidia"), 2100)
    monkeypatch.setattr(devices, "cached_devices", _devices)

    assert vram_calib.offset_mb("/ml/model.gguf", ["CUDA0"]) == 2100
    assert vram_calib.offset_mb("/ml/model.gguf", ["Vulkan1"]) == 0
    assert vram_calib.is_measured("/ml/model.gguf", ["Vulkan1"]) is False


def test_each_card_keeps_its_own_measurement(monkeypatch):
    """Measured on both: each pin gets the number learned on its own silicon."""
    from lld import devices, vram_calib

    monkeypatch.setitem(vram_calib._cache, ("/ml/model.gguf", "nvidia"), 2100)
    monkeypatch.setitem(vram_calib._cache, ("/ml/model.gguf", "amd"), -400)
    monkeypatch.setattr(devices, "cached_devices", _devices)

    assert vram_calib.offset_mb("/ml/model.gguf", ["CUDA0"]) == 2100
    assert vram_calib.offset_mb("/ml/model.gguf", ["Vulkan1"]) == -400
    # Which card an unpinned plan lands on is not knowable, and the two
    # corrections are 2.5 GB apart — guessing either way is worse than none.
    assert vram_calib.offset_mb("/ml/model.gguf") == 0


def test_unknown_pin_keeps_a_lone_measurement(monkeypatch):
    """Nothing probed yet: a bare `Vulkan0` could be either card. With one
    measurement on record there is nothing to confuse it with, and dropping a
    real correction is worse than keeping it — the estimate stays as it was
    before device pinning existed. `ROCm0` is unambiguously not NVIDIA, so the
    CUDA measurement stays out of it."""
    from lld import devices, vram_calib

    monkeypatch.setitem(vram_calib._cache, ("/ml/model.gguf", "nvidia"), 2100)
    monkeypatch.setattr(devices, "cached_devices", list)
    assert vram_calib.offset_mb("/ml/model.gguf", ["Vulkan0"]) == 2100
    assert vram_calib.offset_mb("/ml/model.gguf", ["ROCm0"]) == 0


def test_measured_is_not_the_same_as_a_nonzero_offset(monkeypatch):
    """A model whose estimate happened to land exactly right still counts as
    measured — the planner's safety margin depends on that distinction."""
    from lld import devices, vram_calib

    monkeypatch.setitem(vram_calib._cache, ("/ml/model.gguf", "nvidia"), 0)
    monkeypatch.setattr(devices, "cached_devices", _devices)
    assert vram_calib.offset_mb("/ml/model.gguf", ["CUDA0"]) == 0
    assert vram_calib.is_measured("/ml/model.gguf", ["CUDA0"]) is True
    assert vram_calib.is_measured("/ml/other.gguf", ["CUDA0"]) is False


# ---- schema migration ------------------------------------------------------
# The table predates AMD being measurable at all: one row per model, every one
# of them an nvidia-smi reading. Those rows are the user's accumulated ground
# truth and must survive the widening of the key.

@pytest.mark.asyncio
async def test_pre_vendor_rows_are_carried_over(tmp_path, monkeypatch):
    import contextlib

    import aiosqlite

    db_path = tmp_path / "llamadeck.db"
    with __import__("sqlite3").connect(db_path) as raw:
        raw.execute(
            """CREATE TABLE vram_calibration (
                   model_path TEXT PRIMARY KEY, offset_mb INTEGER NOT NULL,
                   measured_mb INTEGER NOT NULL, estimated_mb INTEGER NOT NULL,
                   measured_at REAL NOT NULL)"""
        )
        raw.execute("INSERT INTO vram_calibration VALUES ('/ml/a.gguf', 2378, 31178, 33556, 1.0)")
        raw.commit()

    @contextlib.asynccontextmanager
    async def _connect(*a, **kw):
        async with aiosqlite.connect(db_path) as conn:
            conn.row_factory = aiosqlite.Row
            yield conn

    monkeypatch.setattr(vram_calib, "connect", _connect)
    monkeypatch.setattr(vram_calib, "ensure_table", _REAL_ENSURE_TABLE)
    monkeypatch.setattr(vram_calib, "_cache", {})
    await vram_calib.load()
    assert vram_calib._cache == {("/ml/a.gguf", "nvidia"): 2378}

    # Idempotent: a second startup neither duplicates nor loses the row.
    monkeypatch.setattr(vram_calib, "_cache", {})
    await vram_calib.load()
    assert vram_calib._cache == {("/ml/a.gguf", "nvidia"): 2378}


@pytest.mark.asyncio
async def test_rows_left_in_a_half_migrated_table_are_recovered(tmp_path, monkeypatch):
    """SQLite runs DDL outside the transaction, so a process killed between
    the rename and the copy leaves every measurement in `_v1`. Startup has to
    pick them up rather than quietly start from nothing."""
    import contextlib

    import aiosqlite

    db_path = tmp_path / "llamadeck.db"
    with __import__("sqlite3").connect(db_path) as raw:
        raw.execute(
            """CREATE TABLE vram_calibration_v1 (
                   model_path TEXT PRIMARY KEY, offset_mb INTEGER NOT NULL,
                   measured_mb INTEGER NOT NULL, estimated_mb INTEGER NOT NULL,
                   measured_at REAL NOT NULL)"""
        )
        raw.execute("INSERT INTO vram_calibration_v1 VALUES ('/ml/b.gguf', 648, 31168, 31816, 1.0)")
        raw.commit()

    @contextlib.asynccontextmanager
    async def _connect(*a, **kw):
        async with aiosqlite.connect(db_path) as conn:
            conn.row_factory = aiosqlite.Row
            yield conn

    monkeypatch.setattr(vram_calib, "connect", _connect)
    monkeypatch.setattr(vram_calib, "ensure_table", _REAL_ENSURE_TABLE)
    monkeypatch.setattr(vram_calib, "_cache", {})
    await vram_calib.load()
    assert vram_calib._cache == {("/ml/b.gguf", "nvidia"): 648}
