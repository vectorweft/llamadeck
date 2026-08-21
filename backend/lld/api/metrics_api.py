from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..metrics import get_metrics_service

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/snapshot")
async def snapshot(history_n: int = 120) -> dict:
    return get_metrics_service().snapshot(history_n)


@router.get("/stream/{preset}")
async def stream(preset: str):
    svc = get_metrics_service()

    async def gen():
        try:
            async for frame in svc.stream(preset):
                yield f"data: {json.dumps(frame.to_dict())}\n\n"
        except asyncio.CancelledError:
            return

    return StreamingResponse(gen(), media_type="text/event-stream")
