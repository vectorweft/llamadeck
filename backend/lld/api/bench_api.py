from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..bench import BenchError, BenchParams, get_bench_manager

router = APIRouter(prefix="/api/bench", tags=["bench"])


class RunBody(BaseModel):
    model_path: str
    n_prompts: list[int] = [512]
    n_gens: list[int] = [128]
    pg_pairs: list[tuple[int, int]] = []
    n_gpu_layers: int = 999
    batch_size: int = 2048
    ubatch_size: int = 512
    threads: int | None = None
    flash_attn: bool = True
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    n_depth: int = 0
    repetitions: int = 3
    extra_flags: list[str] = []


@router.get("/active")
async def active() -> dict:
    j = get_bench_manager().active()
    return j.to_dict() if j else {"status": "idle"}


@router.get("/history")
async def history(limit: int = 50, model_path: str | None = None) -> list[dict]:
    return await get_bench_manager().history(limit=limit, model_path=model_path)


@router.post("/cancel")
async def cancel() -> dict:
    try:
        j = await get_bench_manager().cancel()
        return j.to_dict()
    except BenchError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/run")
async def run(body: RunBody) -> dict:
    try:
        params = BenchParams(**body.model_dump())
        job = await get_bench_manager().run(params)
        return job.to_dict()
    except BenchError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/stream")
async def stream():
    mgr = get_bench_manager()
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
