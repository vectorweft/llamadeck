"""BenchManager — run llama-bench against a model with configurable params.

One concurrent bench allowed (like BuildManager). Streams stdout line-by-line
via pub/sub, parses the final JSON array (llama-bench -o json), persists to
the `benchmarks` SQLite table.

Benchmarks cannot coexist with a loaded llama-server on the same GPU: before
running, the caller should stop any running preset whose VRAM overlaps with
the bench target. For simplicity we do NOT auto-stop presets — the UI surfaces
a warning if VRAM looks tight.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from .build import get_build_manager
from .db import connect
from .settings import LOGS_DIR, ensure_state_dirs, load_settings

log = logging.getLogger(__name__)


class BenchError(RuntimeError):
    pass


class BenchStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BenchParams:
    """User-facing params for one bench invocation."""
    model_path: str
    n_prompts: list[int] = field(default_factory=lambda: [512])   # -p values
    n_gens: list[int] = field(default_factory=lambda: [128])      # -n values
    pg_pairs: list[tuple[int, int]] = field(default_factory=list) # -pg pp,tg (optional)
    n_gpu_layers: int = 999
    batch_size: int = 2048
    ubatch_size: int = 512
    threads: int | None = None
    flash_attn: bool = True
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    n_depth: int = 0
    repetitions: int = 3
    # llama.cpp device ids to pin the run to ("Vulkan1"). Empty = llama.cpp
    # chooses. Without this a bench cannot answer "how fast is *this* card",
    # which is the whole question when the box has two.
    devices: list[str] = field(default_factory=list)
    extra_flags: list[str] = field(default_factory=list)


def build_argv(binary: str, params: BenchParams) -> list[str]:
    argv = [binary, "-m", params.model_path, "-o", "json",
            "--progress", "--no-warmup"]
    if params.n_prompts:
        argv += ["-p", ",".join(str(v) for v in params.n_prompts)]
    if params.n_gens:
        argv += ["-n", ",".join(str(v) for v in params.n_gens)]
    for pp, tg in params.pg_pairs:
        argv += ["-pg", f"{pp},{tg}"]
    if params.n_depth:
        argv += ["-d", str(params.n_depth)]
    argv += ["-ngl", str(params.n_gpu_layers)]
    argv += ["-b", str(params.batch_size)]
    argv += ["-ub", str(params.ubatch_size)]
    if params.threads is not None:
        argv += ["-t", str(params.threads)]
    argv += ["-ctk", params.cache_type_k, "-ctv", params.cache_type_v]
    argv += ["-fa", "1" if params.flash_attn else "0"]
    argv += ["-r", str(params.repetitions)]
    devices = [d for d in (params.devices or []) if d]
    if devices:
        argv += ["-dev", ",".join(devices)]
    argv += list(params.extra_flags)
    return argv


@dataclass
class BenchJob:
    id: int
    model_path: str
    started_at: float
    finished_at: float | None = None
    status: BenchStatus = BenchStatus.RUNNING
    params: BenchParams | None = None
    results: list[dict] = field(default_factory=list)
    log_path: str | None = None
    error: str | None = None
    build_number: int | None = None
    build_commit: str | None = None
    # live only
    lines: list[str] = field(default_factory=list)
    _json_buf: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "model_path": self.model_path,
            "model_name": Path(self.model_path).stem,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status.value,
            "params": self.params.__dict__ if self.params else None,
            "results": self.results,
            "log_path": self.log_path,
            "error": self.error,
            "build_number": self.build_number,
            "build_commit": self.build_commit,
            "duration_seconds": (self.finished_at - self.started_at) if self.finished_at else (time.time() - self.started_at),
        }


class BenchManager:
    def __init__(self, llama_bin: str):
        # llama-bench lives next to llama-server
        bin_path = Path(llama_bin).with_name("llama-bench")
        self.binary = bin_path
        self._active: BenchJob | None = None
        self._subscribers: set[asyncio.Queue[str]] = set()
        self._proc: asyncio.subprocess.Process | None = None
        self._cancel_requested = False

    def rebind(self, llama_bin: str) -> None:
        """Follow a llama_bin change made after boot (setup wizard, Settings)
        so benchmarks don't keep pointing at a path that never existed."""
        self.binary = Path(llama_bin).with_name("llama-bench")

    # ---------- execution ----------

    def active(self) -> BenchJob | None:
        return self._active

    async def cancel(self) -> BenchJob:
        job = self._active
        if not job or job.status != BenchStatus.RUNNING:
            raise BenchError("no running benchmark to cancel")
        self._cancel_requested = True
        proc = self._proc
        if proc and proc.returncode is None:
            proc.terminate()

            async def _hard_kill() -> None:
                await asyncio.sleep(5)
                if proc.returncode is None:
                    proc.kill()

            asyncio.create_task(_hard_kill(), name=f"bench-{job.id}-kill")
        return job

    async def run(self, params: BenchParams) -> BenchJob:
        if self._active and self._active.status == BenchStatus.RUNNING:
            raise BenchError(f"a benchmark is already running (id={self._active.id})")
        if not self.binary.exists():
            raise BenchError(f"llama-bench not found at {self.binary} — build llama.cpp first")
        if not Path(params.model_path).exists():
            raise BenchError(f"model not found: {params.model_path}")

        ensure_state_dirs()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = LOGS_DIR / f"llama-bench-{stamp}.log"

        # Capture current llama-server build metadata so history rows are comparable
        build_mgr = get_build_manager()
        v = await build_mgr.current_version()

        async with connect() as db:
            cur = await db.execute(
                "INSERT INTO benchmarks (model_path, model_name, build_number, build_commit, started_at, status, params_json, log_path) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    params.model_path, Path(params.model_path).stem,
                    v.build_number, v.commit, time.time(),
                    BenchStatus.RUNNING.value,
                    json.dumps(params.__dict__),
                    str(log_path),
                ),
            )
            await db.commit()
            bench_id = cur.lastrowid

        job = BenchJob(
            id=bench_id,
            model_path=params.model_path,
            started_at=time.time(),
            params=params,
            log_path=str(log_path),
            build_number=v.build_number,
            build_commit=v.commit,
        )
        self._active = job
        asyncio.create_task(self._run(job), name=f"bench-{bench_id}")
        return job

    async def _run(self, job: BenchJob) -> None:
        assert job.params is not None
        log_fh = open(job.log_path, "a", buffering=1)

        def emit(line: str) -> None:
            stamped = f"[{datetime.now().strftime('%H:%M:%S')}] {line}"
            log_fh.write(stamped + "\n")
            job.lines.append(stamped)
            if len(job.lines) > 5000:
                del job.lines[:-5000]
            for q in list(self._subscribers):
                try:
                    q.put_nowait(stamped)
                except asyncio.QueueFull:
                    pass

        try:
            argv = build_argv(str(self.binary), job.params)
            emit(f"[LlamaDeck] bench {job.id} starting")
            emit("$ " + " ".join(argv))

            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._proc = proc
            self._cancel_requested = False
            # Stream stderr live (llama-bench progress goes there);
            # collect stdout to parse JSON at the end.
            stdout_buf: list[str] = []

            async def pump_stdout(stream: asyncio.StreamReader) -> None:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    s = line.decode(errors="replace").rstrip()
                    stdout_buf.append(s)

            async def pump_stderr(stream: asyncio.StreamReader) -> None:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    emit(line.decode(errors="replace").rstrip())

            assert proc.stdout is not None and proc.stderr is not None
            await asyncio.gather(
                pump_stdout(proc.stdout),
                pump_stderr(proc.stderr),
            )
            rc = await proc.wait()
            raw = "\n".join(stdout_buf).strip()
            emit(f"[LlamaDeck] llama-bench exited with code {rc}")

            if self._cancel_requested:
                job.status = BenchStatus.CANCELLED
                job.finished_at = time.time()
                emit(f"[LlamaDeck] bench {job.id} cancelled by user")
                return

            if rc != 0:
                raise BenchError(f"llama-bench exit {rc}")

            # Parse JSON array from stdout
            try:
                results = json.loads(raw)
                if not isinstance(results, list):
                    raise ValueError("expected JSON array")
                job.results = results
                emit(f"[LlamaDeck] parsed {len(results)} result rows")
            except Exception as e:
                raise BenchError(f"failed to parse bench output as JSON: {e}") from e

            job.status = BenchStatus.SUCCESS
            job.finished_at = time.time()
            emit(
                f"[LlamaDeck] bench {job.id} SUCCESS in "
                f"{(job.finished_at - job.started_at):.1f}s — {len(results)} runs"
            )
        except Exception as e:
            job.status = BenchStatus.FAILED
            job.finished_at = time.time()
            job.error = str(e)
            err = f"[LlamaDeck] bench {job.id} FAILED: {e}"
            log_fh.write(err + "\n")
            job.lines.append(err)
            for q in list(self._subscribers):
                try:
                    q.put_nowait(err)
                except asyncio.QueueFull:
                    pass
            log.exception("bench %s failed", job.id)
        finally:
            self._proc = None
            log_fh.close()
            async with connect() as db:
                await db.execute(
                    "UPDATE benchmarks SET finished_at=?, status=?, results_json=?, error=? WHERE id=?",
                    (
                        job.finished_at,
                        job.status.value,
                        json.dumps(job.results),
                        job.error,
                        job.id,
                    ),
                )
                await db.commit()

    # ---------- pub/sub ----------

    def subscribe(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=500)
        if self._active:
            for line in list(self._active.lines)[-200:]:
                try:
                    q.put_nowait(line)
                except asyncio.QueueFull:
                    break
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        self._subscribers.discard(q)

    # ---------- history ----------

    async def history(self, limit: int = 50, model_path: str | None = None) -> list[dict]:
        async with connect() as db:
            if model_path:
                cur = await db.execute(
                    "SELECT * FROM benchmarks WHERE model_path=? ORDER BY started_at DESC LIMIT ?",
                    (model_path, limit),
                )
            else:
                cur = await db.execute(
                    "SELECT * FROM benchmarks ORDER BY started_at DESC LIMIT ?", (limit,),
                )
            rows = await cur.fetchall()
        out: list[dict] = []
        for r in rows:
            d = dict(r)
            if d.get("params_json"):
                try:
                    d["params"] = json.loads(d["params_json"])
                except Exception:
                    d["params"] = None
            if d.get("results_json"):
                try:
                    d["results"] = json.loads(d["results_json"])
                except Exception:
                    d["results"] = []
            d.pop("params_json", None)
            d.pop("results_json", None)
            out.append(d)
        return out


_instance: BenchManager | None = None


def get_bench_manager(llama_bin: str | None = None) -> BenchManager:
    global _instance
    if _instance is None:
        s = load_settings()
        _instance = BenchManager(llama_bin or s.llama_bin)
    return _instance
