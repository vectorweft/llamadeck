"""System-level controls: process self-restart, etc.

Exposes POST /api/system/restart which re-execs the LlamaDeck backend in place
(via os.execv) so code changes take effect without dropping the bash launcher.
The HTTP response is flushed before exec runs, so the client sees a 202.

Supervised llama-server children survive: execv preserves PID and child
processes. The new supervisor instance re-adopts them via its scan-on-start
flow (or the Server page's adopt suggestions).
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

from fastapi import APIRouter

router = APIRouter(prefix="/api/system", tags=["system"])
log = logging.getLogger("lld.system")


async def _delayed_exec() -> None:
    # Small delay so the HTTP 202 response actually flushes to the client
    # before we vanish from under uvicorn.
    await asyncio.sleep(0.4)
    log.warning("Restarting LlamaDeck backend via os.execv (argv=%s)", sys.argv)
    try:
        os.execv(sys.argv[0], sys.argv)
    except OSError as e:
        log.error("execv failed: %s — falling back to sys.exit(0)", e)
        os._exit(0)


@router.post("/restart")
async def restart() -> dict:
    asyncio.create_task(_delayed_exec())
    return {"restarting": True, "pid": os.getpid()}
