"""Checksum verification for local GGUF files.

Nothing in the normal path notices a corrupted model. `huggingface_hub` only
compares sizes when resuming, GGUF carries no internal checksum, and llama.cpp
loads damaged weights without a word — the model then produces fluent nonsense
or collapses to one repeated token, which reads like a sampling or
quantization problem and sends people tuning temperature for an afternoon.

Copies between disks are where this bites: a 96 GB DeepSeek-V4 copied off an
external SSD arrived with two of four shards silently altered, every file the
exact expected size.

The expected hashes come from `huggingface_hub` itself. When a model is fetched
into a `local_dir`, it writes `<dir>/.cache/huggingface/download/<rel>.metadata`
whose second line is the file's sha256 (the LFS OID). That file travels with the
model when the folder is copied, so it can vouch for the copy afterwards.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from .vram_estimate import split_shards

log = logging.getLogger(__name__)

_CHUNK = 8 * 1024 * 1024


@dataclass
class ShardResult:
    path: str
    size_bytes: int
    expected_sha256: str | None = None
    actual_sha256: str | None = None
    status: str = "pending"  # pending | ok | corrupt | unverifiable | missing

    def to_dict(self) -> dict:
        return {
            "name": os.path.basename(self.path),
            "path": self.path,
            "size_bytes": self.size_bytes,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
            "status": self.status,
        }


@dataclass
class VerifyJob:
    model_path: str
    shards: list[ShardResult] = field(default_factory=list)
    state: str = "running"          # running | done | error
    bytes_total: int = 0
    bytes_done: int = 0
    error: str | None = None

    @property
    def verdict(self) -> str:
        """Worst outcome across the shards — what the UI should react to."""
        st = {s.status for s in self.shards}
        if "corrupt" in st:
            return "corrupt"
        # Nothing is damaged, a file is simply absent — a different problem
        # with a different fix, so don't file it under "corrupt".
        if "missing" in st:
            return "incomplete"
        if "pending" in st:
            return "running"
        if st == {"ok"}:
            return "ok"
        return "unverifiable"

    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "state": self.state,
            "verdict": self.verdict,
            "bytes_total": self.bytes_total,
            "bytes_done": self.bytes_done,
            "percent": round(self.bytes_done / self.bytes_total * 100, 1)
            if self.bytes_total
            else 0.0,
            "error": self.error,
            "shards": [s.to_dict() for s in self.shards],
        }


def expected_sha256(model_path: str) -> str | None:
    """The sha256 huggingface_hub recorded for this file, if the model was
    downloaded into a local_dir and its .cache sidecar survived the copy.

    The sidecar lives at `<root>/.cache/huggingface/download/<rel>.metadata`,
    where <root> is the local_dir the download targeted — walk up from the file
    until a .cache/huggingface/download tree containing it turns up."""
    p = Path(model_path).resolve()
    rel_parts: list[str] = [p.name]
    for parent in p.parents:
        sidecar = parent / ".cache" / "huggingface" / "download" / Path(*rel_parts)
        sidecar = sidecar.with_name(sidecar.name + ".metadata")
        if sidecar.is_file():
            try:
                lines = sidecar.read_text().splitlines()
            except OSError:
                return None
            # commit hash, then etag (sha256 for LFS blobs), then timestamp.
            if len(lines) >= 2 and len(lines[1].strip()) == 64:
                return lines[1].strip()
            return None
        rel_parts.insert(0, parent.name)
        if len(rel_parts) > 8:
            break
    return None


def _sha256_file(path: str, on_chunk) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
            on_chunk(len(chunk))
    return h.hexdigest()


class VerifyRegistry:
    """One verification at a time per model path; results kept for the session."""

    def __init__(self) -> None:
        self._jobs: dict[str, VerifyJob] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def get(self, model_path: str) -> VerifyJob | None:
        return self._jobs.get(model_path)

    def start(self, model_path: str) -> VerifyJob:
        running = self._tasks.get(model_path)
        if running and not running.done():
            return self._jobs[model_path]

        job = VerifyJob(model_path=model_path)
        for shard in split_shards(model_path):
            try:
                size = os.path.getsize(shard)
            except OSError:
                job.shards.append(ShardResult(path=shard, size_bytes=0, status="missing"))
                continue
            job.shards.append(
                ShardResult(
                    path=shard, size_bytes=size, expected_sha256=expected_sha256(shard)
                )
            )
            job.bytes_total += size
        self._jobs[model_path] = job
        self._tasks[model_path] = asyncio.create_task(self._run(job))
        return job

    async def _run(self, job: VerifyJob) -> None:
        loop = asyncio.get_running_loop()
        try:
            for shard in job.shards:
                if shard.status == "missing":
                    continue
                if not shard.expected_sha256:
                    # Nothing to compare against — say so rather than implying
                    # the file is good.
                    shard.status = "unverifiable"
                    job.bytes_done += shard.size_bytes
                    continue

                def on_chunk(n: int) -> None:
                    job.bytes_done += n

                digest = await loop.run_in_executor(
                    None, _sha256_file, shard.path, on_chunk
                )
                shard.actual_sha256 = digest
                shard.status = "ok" if digest == shard.expected_sha256 else "corrupt"
                if shard.status == "corrupt":
                    log.error(
                        "verify: %s is corrupt (expected %s, got %s)",
                        shard.path, shard.expected_sha256, digest,
                    )
            job.state = "done"
        except Exception as e:  # noqa: BLE001 — surfaced to the caller as job.error
            log.exception("verify failed for %s", job.model_path)
            job.state = "error"
            job.error = str(e)


_registry: VerifyRegistry | None = None


def get_verify_registry() -> VerifyRegistry:
    global _registry
    if _registry is None:
        _registry = VerifyRegistry()
    return _registry
