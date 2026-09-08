from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException

from ..procutil import llama_bin_rejection
from ..settings import Settings, load_settings, save_settings

log = logging.getLogger("lld")

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("")
async def get_settings() -> Settings:
    return load_settings()


@router.put("")
async def put_settings(s: Settings) -> Settings:
    # `llama_bin` decides what every preset executes — the raw command box only
    # supplies its arguments (argv.py:command_argv forces the program). The
    # setup wizard's /use-binary has always vetted its candidate; this endpoint
    # took anything, which is the asymmetry the audit walked through. Check the
    # same way the wizard does, minus the version probe (see procutil).
    rejection = llama_bin_rejection(s.llama_bin)
    if rejection:
        raise HTTPException(status_code=400, detail=rejection)

    previous = load_settings()
    save_settings(s)
    # Never silent: this one field is the pivot in every local-privilege story
    # about LlamaDeck, so a change to it leaves a line behind.
    if previous.llama_bin != s.llama_bin:
        log.warning("llama_bin changed: %s -> %s", previous.llama_bin, s.llama_bin)
    return s


def _probe(path: str | None, want: str) -> dict[str, Any]:
    """Existence probe for one configured path. `want` is "file" or "dir";
    a path that exists but is the wrong kind is reported as a mismatch, so
    the UI can say "that's a directory, not the binary"."""
    if not path or not path.strip():
        return {"exists": False, "kind": None, "ok": False}
    expanded = os.path.expanduser(path.strip())
    if not os.path.exists(expanded):
        return {"exists": False, "kind": None, "ok": False}
    kind = "dir" if os.path.isdir(expanded) else "file"
    return {"exists": True, "kind": kind, "ok": kind == want}


@router.post("/check")
async def check_paths(body: dict) -> dict:
    """Validate the paths of a DRAFT settings object (not yet saved) so the
    Settings page can flag a typo before the user hits Save — the difference
    between "nothing works and I don't know why" and "that path is wrong"."""
    scan_roots = body.get("scan_roots") or []
    return {
        "llama_bin": _probe(body.get("llama_bin"), "file"),
        "llama_repo": _probe(body.get("llama_repo"), "dir"),
        "hf_models_root": _probe(body.get("hf_models_root"), "dir"),
        "scan_roots": [_probe(p, "dir") for p in scan_roots],
    }
