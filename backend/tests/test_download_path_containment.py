"""A download may only ever land under the models root.

Every part of a download's destination arrives from the request body:
`brand`, `series` and `base_model` become directory segments, and `filename`
is joined on by huggingface_hub, whose only `..` guard is Windows-gated
(`_local_folder.py`), so nothing upstream stops a traversal on Linux. The
write that follows is a plain file write — it lands wherever the path points.

Verified on 2026-09-07 to be reachable end to end. Two corrections the audit
report got wrong are worth keeping next to the tests, because they set what
this is actually defending: huggingface_hub's `_chmod_and_move` writes in
umask mode (0664 — **no exec bit**) and unlinks the destination first, so
overwriting a binary is a corruption and denial-of-service story, not the
"next start runs my code" one. The path that does reach execution is writing
over a build input like CMakeLists.txt, which cmake reads and which needs no
exec bit.
"""
from __future__ import annotations

import pytest

from lld.hf import (
    HFDownloader,
    UnsafePathError,
    assert_within,
    safe_relative_filename,
    target_segments,
)


# --- the segments -----------------------------------------------------------

@pytest.mark.parametrize("bad", ["..", ".", "../etc", "a/b", "a\\b", "sub/../.."])
def test_a_directory_segment_may_not_walk_out(bad):
    with pytest.raises(UnsafePathError):
        target_segments(bad, "series", "model")
    with pytest.raises(UnsafePathError):
        target_segments("brand", bad, "model")
    with pytest.raises(UnsafePathError):
        target_segments("brand", "series", bad)


def test_ordinary_segments_are_untouched():
    """The guard must not change the layout — see test_hf_layout for the rules
    it is protecting."""
    assert target_segments("Qwen", "Qwen3.8", "Qwen3.8-27B") == ["Qwen", "Qwen3.8", "Qwen3.8-27B"]
    assert target_segments("unsloth", "LFM2.5", "LFM2.5") == ["unsloth", "LFM2.5"]
    assert target_segments("", None, "solo") == ["solo"]


# --- the filename -----------------------------------------------------------

@pytest.mark.parametrize("bad", [
    "../../etc/cron.d/x",
    "..\\..\\windows",
    "/etc/passwd",
    "C:\\windows\\system32\\x",
    "sub/../../../out.gguf",
    "",
    "   ",
])
def test_a_filename_may_not_walk_out(bad):
    with pytest.raises(UnsafePathError):
        safe_relative_filename(bad)


def test_a_quant_subdirectory_is_still_a_valid_filename():
    """Split GGUFs live in a per-quant folder inside the repo; rejecting the
    slash outright would break every multi-part download."""
    name = "UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00002.gguf"
    assert safe_relative_filename(name) == name


# --- the assembled path -----------------------------------------------------

def test_assert_within_is_the_belt(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    assert assert_within(root, root / "a" / "b") == (root / "a" / "b").resolve()
    assert assert_within(root, root) == root.resolve()
    with pytest.raises(UnsafePathError):
        assert_within(root, tmp_path / "elsewhere")
    with pytest.raises(UnsafePathError):
        assert_within(root, root / ".." / "elsewhere")


# --- the two call sites -----------------------------------------------------

def test_enqueue_refuses_to_create_a_job_outside_the_root(tmp_path):
    dl = HFDownloader(str(tmp_path / "models"), None)
    with pytest.raises(UnsafePathError):
        dl.enqueue(repo_id="x/y", filename="../../../../tmp/pwned.txt")
    with pytest.raises(UnsafePathError):
        dl.enqueue(repo_id="x/y", filename="ok.gguf", brand="..", series="..", base_model="..")


def test_a_job_persisted_before_the_guard_is_not_resumed_into_it(tmp_path):
    """enqueue() is not the only way a job reaches the downloader: rows are
    persisted and replayed at boot, so one written by an older build (or by
    hand into the SQLite file) must be refused at the point of writing bytes,
    not only at the point of asking."""
    from lld.hf import DownloadJob

    root = tmp_path / "models"
    root.mkdir()
    dl = HFDownloader(str(root), None)
    job = DownloadJob(
        job_id="j1", repo_id="x/y", filename="pwned.txt",
        brand="x", series="y", base_model="z",
        target_dir=str(tmp_path / "elsewhere"),
    )
    with pytest.raises(UnsafePathError):
        dl._download_sync(job)


@pytest.mark.asyncio
async def test_the_api_answers_400_rather_than_500(tmp_path, monkeypatch):
    """A rejected path is a bad request, not a crash — the download page shows
    the detail, and the traversal never reaches the job list."""
    from httpx import ASGITransport, AsyncClient

    from lld import hf as hf_mod
    from lld.main import create_app

    monkeypatch.setattr(hf_mod, "_downloader", None, raising=False)
    hf_mod.get_downloader(str(tmp_path / "models"), None)

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1") as c:
        resp = await c.post("/api/hf/download", json={
            "repo_id": "x/y", "filename": "../../../../tmp/pwned.gguf",
        })
    assert resp.status_code == 400
    assert "repo-relative" in resp.json()["detail"] or ".." in resp.json()["detail"]
