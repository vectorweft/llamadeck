"""OpenAI-compatible LLM proxy.

Clients (HFD-2, SCont, hermes-agent, plain curl) point their
OPENAI_BASE_URL at LlamaDeck's /v1 prefix. LlamaDeck handles:

  - llm slot acquire (auto, ephemeral lease) if caller didn't bring a lease
  - heartbeat for the duration of the request (chat completions can be long)
  - proxying request/response to the underlying llama-server (managed_url)
  - lease release on completion / failure
  - streaming pass-through (SSE) for stream=True

The auto-lease is keyed by `X-LlamaDeck-Holder` header; if absent we mint a
holder string from client IP — but every well-behaved client should
send the header so logs are legible.

Header contract:
  X-LlamaDeck-Holder: <project>.<surface>.<purpose>     (recommended)
  X-LlamaDeck-Lease:  <lease_id>                        (optional; reuse a manually-acquired lease)

The pre-rename spellings `X-LSC-Holder` / `X-LSC-Lease` are still accepted, so
a client written against the old name keeps working; the new names win when
both are present. They are the only piece of the old name kept on purpose —
this is a wire contract other projects already send.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..gpu_broker import BrokerError, get_broker

log = logging.getLogger("lld.api.openai_proxy")
router = APIRouter(prefix="/v1", tags=["openai"])


# Default proxy timeouts. LLM requests can be long; we set a generous
# read timeout but a short connect timeout.
_PROXY_CONNECT_S = 5.0
_PROXY_READ_S = 600.0
_HEARTBEAT_INTERVAL_S = 60.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _holder_from_request(provided: str | None, request: Request, fallback_purpose: str) -> str:
    if provided and provided.strip():
        return provided.strip()[:120]
    client = request.client.host if request.client else "unknown"
    return f"anonymous.{fallback_purpose}.{client}"[:120]


def _vram_for_request() -> int:
    """Rough VRAM reservation for an LLM request. Use the active LLM
    service's est_vram_mb if possible — otherwise a safe default."""
    broker = get_broker()
    svcs = broker.state().get("services", {})
    llm = svcs.get("llm") or {}
    return int(llm.get("est_vram_mb") or 12000)


async def _heartbeat_loop(lease_id: str) -> None:
    """Keep an ephemeral lease alive while a long request is in flight."""
    broker = get_broker()
    try:
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
            try:
                await broker.heartbeat(lease_id)
            except BrokerError:
                return
    except asyncio.CancelledError:
        return


# ---------------------------------------------------------------------------
# Core proxy entry — used by every endpoint below
# ---------------------------------------------------------------------------

async def _proxy_to_llama(
    *,
    upstream_path: str,
    body: dict[str, Any],
    holder: str,
    explicit_lease: str | None,
    purpose: str,
) -> StreamingResponse | dict[str, Any]:
    broker = get_broker()
    stream_requested = bool(body.get("stream"))

    # Acquire (ephemeral) or use the provided lease
    ephemeral_lease_id: str | None = None
    if explicit_lease:
        if broker.get_lease(explicit_lease) is None:
            raise HTTPException(status_code=404, detail={
                "code": "lease.not_found",
                "message": f"X-LlamaDeck-Lease {explicit_lease} not active",
            })
    else:
        try:
            lease = await broker.acquire(
                slot="llm",
                holder=holder,
                est_duration_s=120.0,
                est_vram_mb=_vram_for_request(),
                wait_timeout_s=300.0,
                ephemeral=True,
            )
        except BrokerError as e:
            raise HTTPException(status_code=e.status, detail={"code": e.code, "message": e.message})
        ephemeral_lease_id = lease.lease_id

    upstream_url = broker.service_url("llm").rstrip("/") + upstream_path

    async def _release_ephemeral() -> None:
        if ephemeral_lease_id:
            try:
                await broker.release(ephemeral_lease_id)
            except BrokerError:
                pass

    timeout = httpx.Timeout(connect=_PROXY_CONNECT_S, read=_PROXY_READ_S, write=_PROXY_READ_S, pool=_PROXY_READ_S)

    if stream_requested:
        # Streaming: open upstream connection, yield bytes, hold lease until done
        client = httpx.AsyncClient(timeout=timeout)
        hb_task: asyncio.Task | None = None
        if ephemeral_lease_id:
            hb_task = asyncio.create_task(_heartbeat_loop(ephemeral_lease_id),
                                          name=f"openai_proxy.hb.{purpose}")

        async def _gen():
            try:
                async with client.stream("POST", upstream_url, json=body) as resp:
                    if resp.status_code >= 400:
                        text = await resp.aread()
                        raise HTTPException(
                            status_code=resp.status_code,
                            detail={"code": "upstream.error",
                                    "message": text.decode("utf-8", "ignore")[:600]},
                        )
                    async for chunk in resp.aiter_raw():
                        if chunk:
                            yield chunk
            finally:
                if hb_task and not hb_task.done():
                    hb_task.cancel()
                await client.aclose()
                await _release_ephemeral()

        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming: simple request/response
    hb_task: asyncio.Task | None = None
    if ephemeral_lease_id:
        hb_task = asyncio.create_task(_heartbeat_loop(ephemeral_lease_id),
                                      name=f"openai_proxy.hb.{purpose}")
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(upstream_url, json=body)
        if r.status_code >= 400:
            raise HTTPException(
                status_code=r.status_code,
                detail={"code": "upstream.error",
                        "message": r.text[:600]},
            )
        return r.json()
    finally:
        if hb_task and not hb_task.done():
            hb_task.cancel()
        await _release_ephemeral()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    x_llamadeck_holder: str | None = Header(default=None, alias="X-LlamaDeck-Holder"),
    x_llamadeck_lease: str | None = Header(default=None, alias="X-LlamaDeck-Lease"),
    x_lsc_holder: str | None = Header(default=None, alias="X-LSC-Holder"),
    x_lsc_lease: str | None = Header(default=None, alias="X-LSC-Lease"),
):
    body = await request.json()
    holder = _holder_from_request(x_llamadeck_holder or x_lsc_holder, request, "chat")
    return await _proxy_to_llama(
        upstream_path="/v1/chat/completions",
        body=body, holder=holder,
        explicit_lease=x_llamadeck_lease or x_lsc_lease, purpose="chat",
    )


@router.post("/completions")
async def completions(
    request: Request,
    x_llamadeck_holder: str | None = Header(default=None, alias="X-LlamaDeck-Holder"),
    x_llamadeck_lease: str | None = Header(default=None, alias="X-LlamaDeck-Lease"),
    x_lsc_holder: str | None = Header(default=None, alias="X-LSC-Holder"),
    x_lsc_lease: str | None = Header(default=None, alias="X-LSC-Lease"),
):
    body = await request.json()
    holder = _holder_from_request(x_llamadeck_holder or x_lsc_holder, request, "completions")
    return await _proxy_to_llama(
        upstream_path="/v1/completions",
        body=body, holder=holder,
        explicit_lease=x_llamadeck_lease or x_lsc_lease, purpose="completions",
    )


@router.post("/embeddings")
async def embeddings(
    request: Request,
    x_llamadeck_holder: str | None = Header(default=None, alias="X-LlamaDeck-Holder"),
    x_llamadeck_lease: str | None = Header(default=None, alias="X-LlamaDeck-Lease"),
    x_lsc_holder: str | None = Header(default=None, alias="X-LSC-Holder"),
    x_lsc_lease: str | None = Header(default=None, alias="X-LSC-Lease"),
):
    body = await request.json()
    holder = _holder_from_request(x_llamadeck_holder or x_lsc_holder, request, "embeddings")
    return await _proxy_to_llama(
        upstream_path="/v1/embeddings",
        body=body, holder=holder,
        explicit_lease=x_llamadeck_lease or x_lsc_lease, purpose="embeddings",
    )


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """OpenAI-compatible model listing. Reports the active LLM preset
    (and other registered presets as available)."""
    from ..presets import PresetRegistry
    broker = get_broker()
    services = broker.state().get("services", {})
    active_running = bool(services.get("llm", {}).get("running"))
    data = []
    for p in PresetRegistry().list():
        data.append({
            "id": p.name,
            "object": "model",
            "owned_by": "llamadeck",
            "active": active_running and broker.active_slot() == "llm",
            "port": p.port,
        })
    return {"object": "list", "data": data}


__all__ = ["router"]
