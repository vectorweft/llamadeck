"""GPU broker HTTP API.

Endpoints (mounted at /api/gpu by main.py):
  POST /api/gpu/lease                    acquire a slot lease (blocks)
  POST /api/gpu/lease/{lease_id}/heartbeat
  POST /api/gpu/lease/{lease_id}/release
  GET  /api/gpu/lease/{lease_id}         introspect a lease
  GET  /api/gpu/state                    current slot, leases, queue
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..gpu_broker import (
    ALL_SLOTS,
    BrokerError,
    get_broker,
)

log = logging.getLogger("lld.api.gpu")
router = APIRouter(prefix="/api/gpu", tags=["gpu"])


# ---------------------------------------------------------------------------
# Pydantic shapes
# ---------------------------------------------------------------------------

class LeaseRequest(BaseModel):
    slot: Literal["llm", "comfy", "tts"]
    holder: str = Field(..., min_length=1, max_length=120)
    est_duration_s: float = Field(120.0, ge=1.0, le=24 * 3600.0)
    est_vram_mb: int = Field(0, ge=0, le=200000)
    wait_timeout_s: float = Field(300.0, ge=1.0, le=3600.0)


class LeaseResponse(BaseModel):
    lease_id: str
    slot: str
    holder: str
    granted_at: float
    expires_at: float
    ttl_s: float
    managed_url: str


class HeartbeatRequest(BaseModel):
    ttl_s: float | None = Field(None, ge=1.0, le=3600.0)


class HeartbeatResponse(BaseModel):
    lease_id: str
    expires_at: float
    ttl_s: float


class ReleaseResponse(BaseModel):
    ok: bool


class GpuStateResponse(BaseModel):
    active_slot: str | None
    starting_slot: str | None = None
    active_leases: list[dict[str, Any]]
    queue: list[dict[str, Any]]
    keepalive_s: float
    default_ttl_s: float
    recent_start_failures: dict[str, list[float]] = {}
    services: dict[str, dict[str, Any]]


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------

def _raise_broker(err: BrokerError) -> None:
    raise HTTPException(status_code=err.status, detail={"code": err.code, "message": err.message})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/lease", response_model=LeaseResponse)
async def acquire_lease(req: LeaseRequest) -> LeaseResponse:
    broker = get_broker()
    try:
        lease = await broker.acquire(
            slot=req.slot,             # type: ignore[arg-type]
            holder=req.holder,
            est_duration_s=req.est_duration_s,
            est_vram_mb=req.est_vram_mb,
            wait_timeout_s=req.wait_timeout_s,
        )
    except BrokerError as e:
        _raise_broker(e)
        raise  # unreachable
    return LeaseResponse(
        lease_id=lease.lease_id,
        slot=lease.slot,
        holder=lease.holder,
        granted_at=lease.granted_at,
        expires_at=lease.expires_at,
        ttl_s=round(lease.remaining_s(), 1),
        managed_url=broker.service_url(lease.slot),
    )


@router.post("/lease/{lease_id}/heartbeat", response_model=HeartbeatResponse)
async def heartbeat(lease_id: str, req: HeartbeatRequest | None = None) -> HeartbeatResponse:
    broker = get_broker()
    ttl = req.ttl_s if req else None
    try:
        lease = await broker.heartbeat(lease_id, ttl_s=ttl)
    except BrokerError as e:
        _raise_broker(e)
        raise
    return HeartbeatResponse(
        lease_id=lease.lease_id,
        expires_at=lease.expires_at,
        ttl_s=round(lease.remaining_s(), 1),
    )


@router.post("/lease/{lease_id}/release", response_model=ReleaseResponse)
async def release(lease_id: str) -> ReleaseResponse:
    broker = get_broker()
    try:
        await broker.release(lease_id)
    except BrokerError as e:
        _raise_broker(e)
        raise
    return ReleaseResponse(ok=True)


@router.get("/lease/{lease_id}")
async def get_lease(lease_id: str) -> dict[str, Any]:
    broker = get_broker()
    lease = broker.get_lease(lease_id)
    if lease is None:
        raise HTTPException(status_code=404, detail={
            "code": "lease.not_found",
            "message": f"unknown lease {lease_id}",
        })
    return {
        **lease.to_dict(),
        "managed_url": broker.service_url(lease.slot),
    }


@router.get("/state", response_model=GpuStateResponse)
async def state() -> GpuStateResponse:
    broker = get_broker()
    return GpuStateResponse(**broker.state())


@router.get("/slots")
async def slots() -> dict[str, Any]:
    """Static info about supported slots — useful for client introspection."""
    return {"slots": list(ALL_SLOTS)}


__all__ = ["router"]
