"""HuggingFace download manager.

classify()  — repo_id → (brand, series) using filename/repo-name regex
search()    — huggingface_hub HfApi search filtered to GGUF repos
download()  — hf_hub_download into <hf_models_root>/<brand>/<series>/
              runs in asyncio.to_thread; progress published via asyncio.Queue

Job lifecycle: QUEUED → IN_PROGRESS → DONE / FAILED
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import AsyncIterator

log = logging.getLogger("lld.hf")

# Fixed sink that swallows tqdm output (keeps the progress bar out of stderr/journal)
_DEVNULL = open(os.devnull, "w")

# Xet disabled: the xet path silences progress callbacks for minutes
# (→ speed/progress/pause don't work) and it doesn't write .incomplete sequentially,
# so it's incompatible with HTTP Range-resume. The classic CDN path makes both reliable.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


class _DownloadPaused(Exception):
    """User paused — not an error; .incomplete stays on disk and is resumed."""

# ---------------------------------------------------------------------------
# Brand / series classifier
# ---------------------------------------------------------------------------

# (pattern, brand, series_template)
# series_template may reference capture group \1 via \\1
_CLASSIFIERS: list[tuple[re.Pattern[str], str, str]] = [
    # Qwen — match "Qwen<major>.<minor>" or "Qwen<major>" (handles Qwen3.6, Qwen2.5, Qwen2, ...)
    (re.compile(r"QwQ",                             re.I), "Qwen",       "QwQ"),
    (re.compile(r"Qwen(\d+(?:\.\d+)?)",             re.I), "Qwen",       r"Qwen\1"),
    # Meta
    (re.compile(r"Llama[-_]?(\d+(?:\.\d+)?)",       re.I), "Meta",       r"Llama-\1"),
    # Google
    (re.compile(r"Gemma[-_]?(\d+(?:\.\d+)?)",       re.I), "Google",     r"Gemma-\1"),
    (re.compile(r"Gemma",                           re.I), "Google",     "Gemma"),
    # Mistral
    (re.compile(r"Ministral[-_](\d+(?:\.\d+)?)",    re.I), "Mistral",    r"Ministral-\1"),
    (re.compile(r"Mixtral[-_](\d+(?:\.\d+)?)",      re.I), "Mistral",    r"Mixtral-\1"),
    (re.compile(r"Mistral[-_](\d+(?:\.\d+)?)",      re.I), "Mistral",    r"Mistral-\1"),
    (re.compile(r"Mistral",                         re.I), "Mistral",    "Mistral"),
    # DeepSeek
    (re.compile(r"DeepSeek[-_]R(\d+(?:\.\d+)?)",    re.I), "DeepSeek",   r"DeepSeek-R\1"),
    (re.compile(r"DeepSeek[-_]V(\d+(?:\.\d+)?)",    re.I), "DeepSeek",   r"DeepSeek-V\1"),
    (re.compile(r"DeepSeek",                        re.I), "DeepSeek",   "DeepSeek"),
    # Microsoft
    (re.compile(r"Phi[-_]?(\d+(?:\.\d+)?)",         re.I), "Microsoft",  r"Phi-\1"),
    # OpenAI
    (re.compile(r"gpt[-_]?oss",                     re.I), "OpenAI",     "gpt-oss"),
    # AllenAI
    (re.compile(r"OLMo[-_](\d+(?:\.\d+)?)",         re.I), "AllenAI",    r"OLMo-\1"),
    (re.compile(r"OLMo",                            re.I), "AllenAI",    "OLMo"),
    # Others
    (re.compile(r"Falcon[-_](\d+(?:\.\d+)?)",       re.I), "TII",        r"Falcon-\1"),
    (re.compile(r"Yi[-_](\d+(?:\.\d+)?)",           re.I), "01-AI",      r"Yi-\1"),
    (re.compile(r"SmolLM[-_](\d+(?:\.\d+)?)",       re.I), "HuggingFace", r"SmolLM-\1"),
    (re.compile(r"TinyLlama",                       re.I), "TinyLlama",  "TinyLlama"),
    (re.compile(r"Granite[-_](\d+(?:\.\d+)?)",     re.I), "IBM",         r"Granite-\1"),
]


def classify(repo_id: str, tags: list[str] | None = None) -> tuple[str, str]:
    """Return (brand, series) for a HuggingFace repo_id."""
    name = repo_id.split("/")[-1]  # strip owner prefix
    for pat, brand, series_tpl in _CLASSIFIERS:
        m = pat.search(name)
        if m:
            series = _apply_template(series_tpl, m)
            return brand, series
    # fallback: owner / repo-base
    parts = repo_id.split("/")
    owner = parts[0] if len(parts) >= 2 else "Unknown"
    base = re.sub(r"[-_]GGUF$", "", parts[-1], flags=re.I)
    return owner, base


def _apply_template(tpl: str, m: re.Match[str]) -> str:
    """Replace \\1, \\2 etc. in template with match groups."""
    def repl(g: re.Match[str]) -> str:
        idx = int(g.group(1))
        try:
            return m.group(idx) or ""
        except IndexError:
            return ""
    return re.sub(r"\\(\d)", repl, tpl)


# Trailing quant suffix (and any modifiers) in a GGUF filename stem.
# Order matters: most-specific first. Matches include the leading separator.
_QUANT_SUFFIX = re.compile(
    r"(?:[-_.]?(?:UD[-_])?"                         # optional UD/unsloth prefix with sep
    r"(?:Q\d+_[A-Z0-9_]+|"                          # Q4_K_M / Q5_K_XL / Q8_0
    r"IQ\d+[A-Z0-9_]*|"                             # IQ3_XXS / IQ4_NL
    r"MXFP\d+(?:_MOE|_[A-Z0-9_]+)?|"                # MXFP4_MOE
    r"BF16|F16|F32|FP16|FP32|FP8)"
    r"(?:_[A-Z0-9]+)*)$",
    re.I,
)


def derive_base_model(repo_id: str, filename: str) -> str:
    """Third-level folder name identifying the *base model* (architecture-identical
    unit that shares an mmproj). Different quants of the same base model go in
    the same subfolder; different base models (even within the same series) get
    their own folder.

    Strategy:
      - For mmproj-*.gguf, derive from repo_id (mmproj belongs to the repo's
        base model, not to itself).
      - For regular weights, strip the trailing quant suffix from the filename
        stem.
      - Fallback: strip -GGUF suffix from repo_id.
    """
    stem = re.sub(r"\.gguf$", "", filename.split("/")[-1], flags=re.I)
    # mmproj → use repo-derived base (mmproj itself has no quant)
    if stem.lower().startswith("mmproj"):
        repo_name = repo_id.split("/")[-1]
        return re.sub(r"[-_]GGUF$", "", repo_name, flags=re.I)
    stripped = _QUANT_SUFFIX.sub("", stem)
    if stripped and stripped != stem:
        return stripped
    # No recognizable quant suffix — fall back to repo-derived
    repo_name = repo_id.split("/")[-1]
    return re.sub(r"[-_]GGUF$", "", repo_name, flags=re.I)


def target_segments(brand: str, series: str, base_model: str) -> list[str]:
    """The <brand>/<series>/<base_model> path segments, with a segment dropped
    when it only repeats its parent.

    For a repo no classifier matches, classify() falls back to
    (owner, repo-name-without-GGUF) and derive_base_model() strips the quant
    suffix down to that same repo name — which used to nest
    unsloth/LFM2.5-1.2B-Thinking/LFM2.5-1.2B-Thinking. One level is enough.
    """
    segs: list[str] = []
    for seg in (brand, series, base_model):
        seg = (seg or "").strip("/ ")
        if not seg:
            continue
        if segs and segs[-1].casefold() == seg.casefold():
            continue
        segs.append(seg)
    return segs


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class HFFile:
    name: str
    size: int | None  # bytes; None if unknown


@dataclass
class HFModel:
    repo_id: str
    likes: int
    downloads: int
    tags: list[str]
    files: list[HFFile]
    brand: str = ""
    series: str = ""

    def __post_init__(self) -> None:
        if not self.brand:
            self.brand, self.series = classify(self.repo_id, self.tags)


class JobStatus(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"


# States that end the stream / close SSE
_SETTLED = (JobStatus.PAUSED, JobStatus.DONE, JobStatus.FAILED)


@dataclass
class DownloadJob:
    job_id: str
    repo_id: str
    filename: str
    brand: str
    series: str
    base_model: str
    target_dir: str
    revision: str = "main"
    target_path: str = ""
    status: JobStatus = JobStatus.QUEUED
    bytes_downloaded: int = 0
    total_bytes: int = 0
    speed_bps: float = 0.0
    eta_seconds: float | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    # Download-method marker: "http" = Range-resume safe. If empty/other
    # (an old xet-era job) .incomplete may not be sequential → deleted on resume.
    xfer: str = ""
    # Pause signal; read by the tqdm hook in the worker thread. Not persisted.
    cancel_event: threading.Event = field(
        default_factory=threading.Event, repr=False, compare=False
    )

    def to_dict(self) -> dict:
        pct = (self.bytes_downloaded / self.total_bytes * 100) if self.total_bytes else 0
        return {
            "job_id": self.job_id,
            "repo_id": self.repo_id,
            "filename": self.filename,
            "brand": self.brand,
            "series": self.series,
            "base_model": self.base_model,
            "target_dir": self.target_dir,
            "target_path": self.target_path,
            "status": self.status.value,
            "bytes_downloaded": self.bytes_downloaded,
            "total_bytes": self.total_bytes,
            "pct": round(pct, 1),
            "speed_bps": round(self.speed_bps, 0),
            "eta_seconds": self.eta_seconds,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


# ---------------------------------------------------------------------------
# Downloader service
# ---------------------------------------------------------------------------

class HFDownloader:
    #: Maximum number of jobs kept in the DB (older ones are pruned)
    MAX_PERSISTED_JOBS = 100

    def __init__(self, models_root: str, hf_token: str | None = None) -> None:
        self.models_root = Path(models_root)
        self.hf_token = hf_token
        self._jobs: dict[str, DownloadJob] = {}
        self._queues: dict[str, asyncio.Queue[DownloadJob]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._db_lock = threading.Lock()
        try:
            self._load_persisted_jobs()
        except Exception as e:  # noqa: BLE001 — a persistence error must not take down downloads
            log.warning("could not load hf_jobs: %s", e)

    # ------------------------------------------------------------------
    # Persistence (sqlite3, short-lived connection — safe from any thread)
    # ------------------------------------------------------------------

    @staticmethod
    def _db_path() -> Path:
        from .settings import DB_PATH, ensure_state_dirs
        ensure_state_dirs()
        return DB_PATH

    _JOB_COLUMNS = (
        "job_id, repo_id, filename, revision, brand, series, base_model, "
        "target_dir, target_path, status, bytes_downloaded, total_bytes, "
        "error, created_at, finished_at, xfer"
    )

    def _load_persisted_jobs(self) -> None:
        """Restore history at startup. Unfinished jobs (queued/in_progress) are
        marked 'paused' — since .incomplete stays on disk the user can
        resume from where it left off."""
        with self._db_lock, sqlite3.connect(self._db_path(), timeout=5) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""
                CREATE TABLE IF NOT EXISTS hf_jobs (
                    job_id TEXT PRIMARY KEY, repo_id TEXT NOT NULL,
                    filename TEXT NOT NULL, revision TEXT NOT NULL DEFAULT 'main',
                    brand TEXT, series TEXT, base_model TEXT,
                    target_dir TEXT, target_path TEXT, status TEXT NOT NULL,
                    bytes_downloaded INTEGER NOT NULL DEFAULT 0,
                    total_bytes INTEGER NOT NULL DEFAULT 0,
                    error TEXT, created_at REAL NOT NULL, finished_at REAL,
                    xfer TEXT NOT NULL DEFAULT ''
                )
            """)
            try:  # add the column if the table was created with the old schema
                db.execute("ALTER TABLE hf_jobs ADD COLUMN xfer TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass  # kolon zaten var
            db.execute(
                "DELETE FROM hf_jobs WHERE job_id NOT IN "
                "(SELECT job_id FROM hf_jobs ORDER BY created_at DESC LIMIT ?)",
                (self.MAX_PERSISTED_JOBS,),
            )
            db.execute(
                "UPDATE hf_jobs SET status = ?, error = NULL "
                "WHERE status IN (?, ?)",
                (JobStatus.PAUSED.value, JobStatus.QUEUED.value, JobStatus.IN_PROGRESS.value),
            )
            rows = db.execute(
                f"SELECT {self._JOB_COLUMNS} FROM hf_jobs ORDER BY created_at"
            ).fetchall()
            db.commit()
        for r in rows:
            job = DownloadJob(
                job_id=r[0], repo_id=r[1], filename=r[2], revision=r[3] or "main",
                brand=r[4] or "", series=r[5] or "", base_model=r[6] or "",
                target_dir=r[7] or "", target_path=r[8] or "",
                status=JobStatus(r[9]),
                bytes_downloaded=r[10] or 0, total_bytes=r[11] or 0,
                error=r[12], created_at=r[13], finished_at=r[14],
                xfer=r[15] or "",
            )
            self._jobs[job.job_id] = job
        if rows:
            log.info("hf_jobs: loaded %d past jobs", len(rows))

    def _persist(self, job: DownloadJob) -> None:
        try:
            with self._db_lock, sqlite3.connect(self._db_path(), timeout=5) as db:
                db.execute(
                    f"INSERT OR REPLACE INTO hf_jobs ({self._JOB_COLUMNS}) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        job.job_id, job.repo_id, job.filename, job.revision,
                        job.brand, job.series, job.base_model,
                        job.target_dir, job.target_path, job.status.value,
                        job.bytes_downloaded, job.total_bytes,
                        job.error, job.created_at, job.finished_at, job.xfer,
                    ),
                )
                db.commit()
        except Exception as e:  # noqa: BLE001 — a DB write must never take down a download
            log.debug("hf_jobs persist %s: %s", job.job_id, e)

    def _delete_persisted(self, job_id: str) -> None:
        try:
            with self._db_lock, sqlite3.connect(self._db_path(), timeout=5) as db:
                db.execute("DELETE FROM hf_jobs WHERE job_id = ?", (job_id,))
                db.commit()
        except Exception as e:  # noqa: BLE001
            log.debug("hf_jobs delete %s: %s", job_id, e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(self, query: str, limit: int = 20) -> list[HFModel]:
        return await asyncio.to_thread(self._search_sync, query, limit)

    async def list_files(self, repo_id: str) -> list[HFFile]:
        return await asyncio.to_thread(self._list_files_sync, repo_id)

    def enqueue(
        self,
        repo_id: str,
        filename: str,
        brand: str | None = None,
        series: str | None = None,
        base_model: str | None = None,
        revision: str = "main",
    ) -> DownloadJob:
        if brand is None or series is None:
            b, s = classify(repo_id)
            brand = brand or b
            series = series or s
        if base_model is None:
            base_model = derive_base_model(repo_id, filename)

        target_dir = str(self.models_root.joinpath(*target_segments(brand, series, base_model)))

        # Dedupe: if an active job exists for the same file, return it (a repeat
        # click must not open a new download); resume a paused/failed job in place.
        for existing in self._jobs.values():
            if existing.repo_id != repo_id or existing.filename != filename:
                continue
            if existing.status in (JobStatus.QUEUED, JobStatus.IN_PROGRESS):
                return existing
            if (
                existing.status in (JobStatus.PAUSED, JobStatus.FAILED)
                and existing.target_dir == target_dir
            ):
                return self.resume(existing.job_id) or existing

        job = DownloadJob(
            job_id=str(uuid.uuid4()),
            repo_id=repo_id,
            filename=filename,
            brand=brand,
            series=series,
            base_model=base_model,
            target_dir=target_dir,
            revision=revision,
        )
        self._jobs[job.job_id] = job
        self._persist(job)
        asyncio.ensure_future(self._run_job(job))
        return job

    def pause(self, job_id: str) -> DownloadJob | None:
        """Pause an active job. The worker thread sees the signal on its next
        progress callback; .incomplete stays on disk and can be resumed."""
        job = self._jobs.get(job_id)
        if job is None or job.status not in (JobStatus.QUEUED, JobStatus.IN_PROGRESS):
            return None
        job.cancel_event.set()
        return job

    def resume(self, job_id: str) -> DownloadJob | None:
        """Restart a paused/failed job from where it left off
        (hf_hub_download continues from the .incomplete file)."""
        job = self._jobs.get(job_id)
        if job is None or job.status not in (JobStatus.PAUSED, JobStatus.FAILED):
            return None
        job.cancel_event = threading.Event()  # reset a previously set signal
        job.status = JobStatus.QUEUED
        job.error = None
        job.finished_at = None
        job.speed_bps = 0.0
        job.eta_seconds = None
        self._persist(job)
        asyncio.ensure_future(self._run_job(job))
        return job

    def remove(self, job_id: str) -> bool:
        """Remove a finished/paused/failed job from the list and the DB.
        An active job cannot be removed (pause it first)."""
        job = self._jobs.get(job_id)
        if job is None or job.status in (JobStatus.QUEUED, JobStatus.IN_PROGRESS):
            return False
        del self._jobs[job_id]
        self._delete_persisted(job_id)
        return True

    def get_job(self, job_id: str) -> DownloadJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[DownloadJob]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    async def stream_job(self, job_id: str) -> AsyncIterator[DownloadJob]:
        """Yield job updates until done/failed."""
        job = self._jobs.get(job_id)
        if job is None:
            return
        if job.status in _SETTLED:
            yield job
            return
        q: asyncio.Queue[DownloadJob] = asyncio.Queue()
        self._queues[job_id] = q
        try:
            yield job  # initial state
            while True:
                try:
                    updated = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield updated
                    if updated.status in _SETTLED:
                        break
                except asyncio.TimeoutError:
                    yield job  # heartbeat
        finally:
            self._queues.pop(job_id, None)

    # ------------------------------------------------------------------
    # Internal sync helpers (run in thread)
    # ------------------------------------------------------------------

    def _search_sync(self, query: str, limit: int) -> list[HFModel]:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            log.warning("huggingface_hub not installed")
            return []

        api = HfApi()
        # Unsloth repos rarely land in the global last_modified window, so query
        # them separately and pin them on top (each block stays newest-first).
        ordered: list = []
        seen: set[str] = set()
        for kwargs in (
            {"author": "unsloth", "filter": "gguf", "search": query,
             "limit": limit, "sort": "last_modified"},
            {"filter": "gguf", "search": query,
             "limit": limit, "sort": "last_modified", "cardData": True},
        ):
            try:
                for m in api.list_models(**kwargs):
                    if m.modelId not in seen:
                        seen.add(m.modelId)
                        ordered.append(m)
            except Exception as e:
                log.warning("HF search failed (%s): %s", kwargs.get("author", "all"), e)
        ordered = ordered[:limit]

        results: list[HFModel] = []
        for m in ordered:
            try:
                files = self._list_files_sync(m.modelId)
                gguf_files = [f for f in files if f.name.endswith(".gguf")]
                results.append(HFModel(
                    repo_id=m.modelId,
                    likes=m.likes or 0,
                    downloads=m.downloads or 0,
                    tags=list(m.tags or []),
                    files=gguf_files,
                ))
            except Exception as e:
                log.debug("skip %s: %s", m.modelId, e)
        return results

    def _list_files_sync(self, repo_id: str) -> list[HFFile]:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            return []
        api = HfApi()
        try:
            # recursive=True is required: big repos like unsloth put quants in
            # subfolders; a flat listing shows 0 .gguf files.
            info = api.list_repo_tree(repo_id, recursive=True, token=self.hf_token)
            files: list[HFFile] = []
            for entry in info:
                if hasattr(entry, "path"):
                    size = getattr(entry, "size", None)
                    files.append(HFFile(name=entry.path, size=size))
            return files
        except Exception as e:
            log.debug("list_files %s: %s", repo_id, e)
            return []

    # ------------------------------------------------------------------
    # Download runner
    # ------------------------------------------------------------------

    async def _run_job(self, job: DownloadJob) -> None:
        self._loop = asyncio.get_running_loop()
        if job.cancel_event.is_set():  # paused before it even started
            self._mark_paused(job)
            return
        job.status = JobStatus.IN_PROGRESS
        self._persist(job)
        self._publish(job)
        try:
            await asyncio.to_thread(self._download_sync, job)
            job.status = JobStatus.DONE
            job.finished_at = time.time()
            job.speed_bps = 0.0
            job.eta_seconds = None
            if job.total_bytes:  # if the file is already on disk no progress flows — show 100%
                job.bytes_downloaded = job.total_bytes
            self._persist(job)
            self._publish(job)
            log.info("Download done: %s → %s", job.filename, job.target_path)
            # trigger rescan
            try:
                from .models import full_rescan
                from .settings import load_settings
                s = load_settings()
                asyncio.ensure_future(full_rescan(s.scan_roots))
            except Exception as e:
                log.debug("rescan after download: %s", e)
        except _DownloadPaused:
            self._mark_paused(job)
        except Exception as e:
            # The pause signal can get wrapped in another exception by layers
            # like hf_xet — if the signal is set, treat this as a pause too.
            if job.cancel_event.is_set():
                self._mark_paused(job)
                return
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.finished_at = time.time()
            job.speed_bps = 0.0
            job.eta_seconds = None
            self._persist(job)
            self._publish(job)
            log.error("Download failed %s: %s", job.filename, e)

    def _mark_paused(self, job: DownloadJob) -> None:
        job.status = JobStatus.PAUSED
        job.error = None
        job.finished_at = None
        job.speed_bps = 0.0
        job.eta_seconds = None
        self._persist(job)
        self._publish(job)
        log.info("Download paused: %s (%d/%d bytes)",
                 job.filename, job.bytes_downloaded, job.total_bytes)

    def _download_sync(self, job: DownloadJob) -> None:
        try:
            import huggingface_hub
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import tqdm as hf_tqdm
        except ImportError as e:
            raise RuntimeError("huggingface_hub not installed") from e

        # The env var may have been set after huggingface_hub was imported;
        # also override the constant at call time (is_xet_available reads it).
        huggingface_hub.constants.HF_HUB_DISABLE_XET = True
        # The default 10 MB chunk reduces progress/pause responsiveness to one
        # event per chunk (~10 s lag on slow links). 1 MB is smooth and free.
        huggingface_hub.constants.DOWNLOAD_CHUNK_SIZE = 1024 * 1024

        target = Path(job.target_dir)

        # Migration guard: if this job wasn't previously run over HTTP (old xet
        # era), .incomplete may not be a sequential prefix — Range-resume would
        # silently corrupt the file. Safe path: delete the partial, start clean.
        if job.xfer != "http":
            dl_cache = target / ".cache" / "huggingface" / "download"
            for p in dl_cache.glob("*.incomplete"):
                try:
                    log.warning("deleting xet-era partial file (resume not safe): %s", p)
                    p.unlink()
                except OSError as e:
                    log.debug("incomplete silinemedi %s: %s", p, e)
            job.bytes_downloaded = 0
            job.xfer = "http"
            self._persist(job)

        # Resolve the file size up front so the disk-space guard below can
        # actually fire. Otherwise total_bytes is only learned from tqdm mid-
        # download — too late to refuse, so a too-big file fills the disk and
        # fails halfway.
        if not job.total_bytes:
            try:
                for f in self._list_files_sync(job.repo_id):
                    if f.name == job.filename and f.size:
                        job.total_bytes = f.size
                        break
            except Exception as e:
                log.debug("size probe for %s failed: %s", job.filename, e)

        # disk space check — probe the nearest existing ancestor dir
        probe = target
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        usage = shutil.disk_usage(probe)
        # On resume the downloaded part is already on disk — request only the rest
        need = max(job.total_bytes - job.bytes_downloaded, 0)
        if need and usage.free < need * 1.1:
            raise RuntimeError(
                f"Insufficient disk space: need {need/1e9:.1f} GB, "
                f"free {usage.free/1e9:.1f} GB"
            )

        target.mkdir(parents=True, exist_ok=True)

        # We wrap tqdm to get progress callbacks. huggingface_hub uses tqdm internally.
        # We monkey-patch via a custom tqdm class.
        job_ref = job
        downloader = self

        class _ProgressTqdm(hf_tqdm):
            def __init__(self, *args, **kwargs):
                # The service runs without a TTY; with disable=None tqdm turns itself
                # off and update() never increments self.n → progress stays 0.
                kwargs["disable"] = False
                kwargs["file"] = _DEVNULL
                super().__init__(*args, **kwargs)
                self._last_publish = 0.0
                self._last_persist = 0.0
                if self.total:
                    job_ref.total_bytes = self.total

            def update(self, n: int = 1) -> bool | None:
                if job_ref.cancel_event.is_set():
                    raise _DownloadPaused()
                result = super().update(n)
                job_ref.bytes_downloaded = self.n
                if self.total:
                    job_ref.total_bytes = self.total
                rate = self.format_dict.get("rate") or 0
                job_ref.speed_bps = rate or 0  # bytes/s
                remaining = self.total - self.n if self.total else None
                job_ref.eta_seconds = (remaining / rate) if (rate and remaining) else None
                # xet calls update per chunk; throttle so SSE isn't flooded
                now = time.monotonic()
                if now - self._last_publish >= 0.5:
                    self._last_publish = now
                    downloader._publish(job_ref)
                # persist periodically so the "where it left off" info is fresh after restart
                if now - self._last_persist >= 3.0:
                    self._last_persist = now
                    downloader._persist(job_ref)
                return result

        local_path = hf_hub_download(
            repo_id=job.repo_id,
            filename=job.filename,
            revision=job.revision,
            local_dir=str(target),
            token=self.hf_token,
            tqdm_class=_ProgressTqdm,
        )
        job.target_path = local_path

    def _publish(self, job: DownloadJob) -> None:
        # Runs in the download worker thread; asyncio.Queue is not thread-safe,
        # so route the put through the event loop.
        q = self._queues.get(job.job_id)
        loop = self._loop
        if q is None or loop is None:
            return
        try:
            loop.call_soon_threadsafe(q.put_nowait, job)
        except RuntimeError:
            pass  # loop already closed


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_downloader: HFDownloader | None = None


def get_downloader(models_root: str | None = None, hf_token: str | None = None) -> HFDownloader:
    global _downloader
    if _downloader is None:
        if models_root is None:
            from .settings import load_settings
            s = load_settings()
            models_root = s.hf_models_root
            hf_token = s.hf_token
        _downloader = HFDownloader(models_root, hf_token)
    return _downloader
