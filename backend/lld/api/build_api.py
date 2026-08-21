from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import accel
from ..build import BuildError, get_build_manager

router = APIRouter(prefix="/api/build", tags=["build"])


class RebuildBody(BaseModel):
    # accel backend id: auto | cuda | metal | hip | vulkan | cpu
    backend: str | None = None
    # Pre-0.2 clients sent only this. Ignored when `backend` is set.
    cuda: bool | None = None
    jobs: int | None = None


@router.get("/version")
async def version() -> dict:
    v = await get_build_manager().current_version()
    return {"build_number": v.build_number, "commit": v.commit, "raw": v.raw}


@router.get("/check")
async def check() -> dict:
    try:
        return await get_build_manager().check_updates()
    except BuildError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active")
async def active() -> dict:
    j = get_build_manager().active()
    return j.to_dict() if j else {"status": "idle"}


@router.get("/history")
async def history(limit: int = 20) -> list[dict]:
    return await get_build_manager().history(limit)


@router.get("/backends")
async def backends(refresh: bool = False) -> dict:
    """Compute backends buildable on this machine + what the current build
    directory was configured with. Drives the Build page's backend picker."""
    if refresh:
        accel.refresh()
    mgr = get_build_manager()
    return {
        "platform": accel.platform_info().to_dict(),
        "backends": [b.to_dict() for b in accel.detect_backends()],
        "preferred": accel.preferred_backend(),
        "current": mgr.cached_backend(mgr.llama_repo / "build"),
    }


@router.post("/rebuild")
async def rebuild(body: RebuildBody) -> dict:
    try:
        job = await get_build_manager().rebuild(
            backend=body.backend, cuda=body.cuda, jobs=body.jobs,
        )
        return job.to_dict()
    except BuildError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/stream")
async def stream():
    mgr = get_build_manager()
    q = mgr.subscribe()

    async def gen():
        try:
            while True:
                line = await q.get()
                yield f"data: {json.dumps({'line': line})}\n\n"
        except asyncio.CancelledError:
            return
        finally:
            mgr.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream")
