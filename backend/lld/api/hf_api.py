from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..hf import classify, derive_base_model, get_downloader

router = APIRouter(prefix="/api/hf", tags=["hf"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class DownloadRequest(BaseModel):
    repo_id: str
    filename: str
    brand: str | None = None
    series: str | None = None
    base_model: str | None = None
    revision: str = "main"


class ClassifyRequest(BaseModel):
    repo_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/classify")
async def classify_repo(repo_id: str = Query(...), filename: str | None = Query(None)):
    brand, series = classify(repo_id)
    base_model = derive_base_model(repo_id, filename or "")
    return {"brand": brand, "series": series, "base_model": base_model}


@router.get("/search")
async def search(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=50)):
    dl = get_downloader()
    results = await dl.search(q, limit=limit)
    return {
        "results": [
            {
                "repo_id": m.repo_id,
                "likes": m.likes,
                "downloads": m.downloads,
                "tags": m.tags,
                "brand": m.brand,
                "series": m.series,
                "files": [
                    {"name": f.name, "size": f.size}
                    for f in m.files
                ],
            }
            for m in results
        ]
    }


@router.get("/files")
async def list_files(repo_id: str = Query(...)):
    dl = get_downloader()
    files = await dl.list_files(repo_id)
    brand, series = classify(repo_id)
    return {
        "repo_id": repo_id,
        "brand": brand,
        "series": series,
        "files": [{"name": f.name, "size": f.size} for f in files if f.name.endswith(".gguf")],
    }


@router.post("/download")
async def start_download(req: DownloadRequest):
    dl = get_downloader()
    job = dl.enqueue(
        repo_id=req.repo_id,
        filename=req.filename,
        brand=req.brand,
        series=req.series,
        base_model=req.base_model,
        revision=req.revision,
    )
    return job.to_dict()


@router.get("/jobs")
async def list_jobs():
    dl = get_downloader()
    return {"jobs": [j.to_dict() for j in dl.list_jobs()]}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    dl = get_downloader()
    job = dl.get_job(job_id)
    if job is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_dict()


@router.post("/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    from fastapi import HTTPException
    dl = get_downloader()
    if dl.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    job = dl.pause(job_id)
    if job is None:
        raise HTTPException(status_code=409, detail="job is not active")
    return job.to_dict()


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    from fastapi import HTTPException
    dl = get_downloader()
    if dl.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    job = dl.resume(job_id)
    if job is None:
        raise HTTPException(status_code=409, detail="job is not resumable")
    return job.to_dict()


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    from fastapi import HTTPException
    dl = get_downloader()
    if dl.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    if not dl.remove(job_id):
        raise HTTPException(status_code=409, detail="job is active; pause it first")
    return {"ok": True}


@router.get("/stream/{job_id}")
async def stream_job(job_id: str):
    dl = get_downloader()

    async def gen():
        try:
            async for state in dl.stream_job(job_id):
                yield f"data: {json.dumps(state.to_dict())}\n\n"
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            return

    return StreamingResponse(gen(), media_type="text/event-stream")
