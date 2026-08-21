from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

from .settings import DB_PATH, ensure_state_dirs

SCHEMA_VERSION = 1

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS models (
        path TEXT PRIMARY KEY,
        family TEXT,
        quant TEXT,
        size_bytes INTEGER NOT NULL,
        mtime REAL NOT NULL,
        has_mmproj INTEGER NOT NULL DEFAULT 0,
        mmproj_path TEXT,
        last_used REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS request_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,
        consumer_id TEXT,
        endpoint TEXT NOT NULL,
        model TEXT,
        prompt_tokens INTEGER,
        completion_tokens INTEGER,
        duration_ms INTEGER,
        status INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS metrics_downsampled (
        ts REAL PRIMARY KEY,
        prompt_tps REAL,
        decode_tps REAL,
        kv_cache_used INTEGER,
        kv_cache_total INTEGER,
        busy_slots INTEGER,
        total_slots INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS builds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at REAL NOT NULL,
        finished_at REAL,
        from_commit TEXT,
        to_commit TEXT,
        status TEXT NOT NULL,
        log_path TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS downloads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        repo_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        revision TEXT NOT NULL DEFAULT 'main',
        brand TEXT,
        series TEXT,
        target_path TEXT,
        status TEXT NOT NULL,
        bytes_total INTEGER,
        bytes_done INTEGER,
        started_at REAL,
        finished_at REAL,
        error TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS hf_jobs (
        job_id TEXT PRIMARY KEY,
        repo_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        revision TEXT NOT NULL DEFAULT 'main',
        brand TEXT,
        series TEXT,
        base_model TEXT,
        target_dir TEXT,
        target_path TEXT,
        status TEXT NOT NULL,
        bytes_downloaded INTEGER NOT NULL DEFAULT 0,
        total_bytes INTEGER NOT NULL DEFAULT 0,
        error TEXT,
        created_at REAL NOT NULL,
        finished_at REAL,
        xfer TEXT NOT NULL DEFAULT ''
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS benchmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_path TEXT NOT NULL,
        model_name TEXT,
        build_number INTEGER,
        build_commit TEXT,
        started_at REAL NOT NULL,
        finished_at REAL,
        status TEXT NOT NULL,
        params_json TEXT,
        results_json TEXT,
        log_path TEXT,
        error TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS help_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        commit_sha TEXT NOT NULL UNIQUE,
        build_number INTEGER,
        captured_at REAL NOT NULL,
        help_text TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS feature_scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at REAL NOT NULL,
        from_commit TEXT,
        to_commit TEXT,
        build_number INTEGER,
        new_flags_json TEXT,
        removed_flags_json TEXT,
        commits_json TEXT,
        releases_json TEXT,
        status TEXT NOT NULL,
        error TEXT,
        seen INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS release_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_id INTEGER NOT NULL REFERENCES feature_scans(id),
        created_at REAL NOT NULL,
        title_tr TEXT NOT NULL,
        what_tr TEXT NOT NULL,
        how_tr TEXT NOT NULL,
        why_tr TEXT NOT NULL,
        flags_json TEXT NOT NULL,
        architectures_json TEXT NOT NULL,
        source_urls_json TEXT NOT NULL,
        confidence TEXT NOT NULL,
        seen INTEGER NOT NULL DEFAULT 0,
        seen_at REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS guides (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at REAL NOT NULL,
        build_number INTEGER,
        commit_sha TEXT,
        status TEXT NOT NULL,
        content_md TEXT,
        error TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS feature_ab_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        feature_id INTEGER REFERENCES release_features(id),
        model_path TEXT NOT NULL,
        flags_json TEXT NOT NULL,
        created_at REAL NOT NULL,
        status TEXT NOT NULL,
        bench_off_id INTEGER,
        bench_on_id INTEGER,
        error TEXT
    )
    """,
]


async def init_db(db_path: Path = DB_PATH) -> None:
    ensure_state_dirs()
    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        for stmt in SCHEMA:
            await db.execute(stmt)
        cur = await db.execute("SELECT version FROM schema_version LIMIT 1")
        row = await cur.fetchone()
        if row is None:
            await db.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        await db.commit()


@asynccontextmanager
async def connect(db_path: Path = DB_PATH):
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        yield db
