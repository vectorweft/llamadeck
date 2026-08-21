"""Checksum verification against huggingface_hub's recorded sha256.

The failure this exists for: a copied model whose shards keep their exact
sizes but not their contents. Size checks pass, GGUF parses, llama.cpp loads
it, and the model answers with garbage.
"""
from __future__ import annotations

import asyncio
import hashlib

import pytest

from lld.verify import VerifyRegistry, expected_sha256


def _model(tmp_path, name: str, body: bytes, sidecar_sha: str | None):
    """A model file plus, optionally, the .cache sidecar hf_hub leaves behind."""
    f = tmp_path / name
    f.write_bytes(body)
    if sidecar_sha is not None:
        cache = tmp_path / ".cache" / "huggingface" / "download"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / f"{name}.metadata").write_text(
            f"e3aa0d6a5fa4f820d9e132ac1fd1d01e1b2b49e0\n{sidecar_sha}\n1785579149.58\n"
        )
    return str(f)


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


async def _run(path: str):
    reg = VerifyRegistry()
    reg.start(path)
    for _ in range(200):
        job = reg.get(path)
        if job.state != "running":
            return job
        await asyncio.sleep(0.01)
    raise AssertionError("verification did not finish")


def test_expected_sha_read_from_sidecar(tmp_path):
    body = b"weights"
    path = _model(tmp_path, "m.gguf", body, _sha(body))
    assert expected_sha256(path) == _sha(body)


def test_expected_sha_found_from_a_subdirectory(tmp_path):
    """hf_hub mirrors the repo layout, so the sidecar sits under the same
    relative path — the model itself is usually a folder or two down."""
    sub = tmp_path / "UD-IQ3_XXS"
    sub.mkdir()
    body = b"weights"
    f = sub / "m.gguf"
    f.write_bytes(body)
    cache = tmp_path / ".cache" / "huggingface" / "download" / "UD-IQ3_XXS"
    cache.mkdir(parents=True)
    (cache / "m.gguf.metadata").write_text(f"commit\n{_sha(body)}\n1.0\n")
    assert expected_sha256(str(f)) == _sha(body)


def test_no_sidecar_means_no_expectation(tmp_path):
    assert expected_sha256(_model(tmp_path, "m.gguf", b"x", None)) is None


@pytest.mark.asyncio
async def test_intact_file_verifies_ok(tmp_path):
    body = b"a" * 4096
    job = await _run(_model(tmp_path, "m.gguf", body, _sha(body)))
    assert job.verdict == "ok"
    assert job.shards[0].status == "ok"


@pytest.mark.asyncio
async def test_same_size_different_bytes_is_caught(tmp_path):
    """The exact regression: size matches, contents do not."""
    good = b"a" * 4096
    bad = b"a" * 4095 + b"b"
    assert len(good) == len(bad)
    job = await _run(_model(tmp_path, "m.gguf", bad, _sha(good)))
    assert job.verdict == "corrupt"
    assert job.shards[0].status == "corrupt"
    assert job.shards[0].actual_sha256 == _sha(bad)


@pytest.mark.asyncio
async def test_unverifiable_is_not_reported_as_ok(tmp_path):
    job = await _run(_model(tmp_path, "m.gguf", b"x" * 32, None))
    assert job.verdict == "unverifiable"


@pytest.mark.asyncio
async def test_every_shard_of_a_split_is_checked(tmp_path):
    bodies = {i: bytes([i]) * 1024 for i in range(1, 5)}
    for i, body in bodies.items():
        _model(tmp_path, f"M-{i:05d}-of-00004.gguf", body, _sha(body))
    # Corrupt part 3 only, keeping its length.
    p3 = tmp_path / "M-00003-of-00004.gguf"
    p3.write_bytes(b"\xff" * 1024)

    job = await _run(str(tmp_path / "M-00001-of-00004.gguf"))
    assert len(job.shards) == 4
    assert [s.status for s in job.shards] == ["ok", "ok", "corrupt", "ok"]
    assert job.verdict == "corrupt"


@pytest.mark.asyncio
async def test_missing_shard_reported(tmp_path):
    for i in (1, 2, 4):
        body = bytes([i]) * 512
        _model(tmp_path, f"M-{i:05d}-of-00004.gguf", body, _sha(body))
    job = await _run(str(tmp_path / "M-00001-of-00004.gguf"))
    assert job.shards[2].status == "missing"
    # Absent is not damaged — the fix is a download, not a re-copy.
    assert job.verdict == "incomplete"
