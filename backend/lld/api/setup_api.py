"""First-run setup wizard.

A fresh install has no llama-server, no models and no presets, and the rest of
the UI assumes all three. These endpoints answer the two questions the wizard
needs — "what does this machine already have?" and "install the missing
piece" — in a form the frontend can render as steps.

Everything here is a thin composition of machinery that already exists
(accel probes, BuildManager, models table, PresetRegistry); the wizard's own
contribution is deciding *which step the user is on* in one round trip, and
writing the results back into settings so the rest of the app picks them up.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import accel, models
from ..build import LLAMA_CPP_URL, BuildError, get_build_manager
from ..presets import PresetRegistry
from ..procutil import run_capture
from ..settings import load_settings, save_settings

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/setup", tags=["setup"])

#: Where a llama-server may already be sitting on a machine that never ran
#: LlamaDeck. Ordered by how likely it is to be the one the user means.
_BINARY_HINTS = (
    ("~/llama.cpp/build/bin/llama-server", "source build"),
    ("~/src/llama.cpp/build/bin/llama-server", "source build"),
    ("/opt/homebrew/bin/llama-server", "homebrew"),
    ("/usr/local/bin/llama-server", "local install"),
    ("/usr/bin/llama-server", "system package"),
)


def _expand(p: str | None) -> Path | None:
    if not p or not p.strip():
        return None
    return Path(os.path.expanduser(p.strip()))


#: Memoized `--version` results, keyed by (path, mtime, size). The wizard
#: polls /state every few seconds and the dashboard card every 15 s; without
#: this, each poll spawns a process just to re-learn something that only
#: changes when the binary itself is replaced — which the key catches.
_version_cache: dict[tuple[str, float, int], str | None] = {}


async def _binary_version(path: Path) -> str | None:
    """`llama-server --version` banner, or None if it doesn't run. Doubles as
    the "is this actually a llama-server?" check — an existing file at the
    right path can still be the wrong binary, or built for another arch."""
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime, stat.st_size)
    except OSError:
        return None
    if key in _version_cache:
        return _version_cache[key]
    result = await _probe_version(path)
    # One entry per distinct binary; a rebuild adds a new key. Bounded so a
    # pathological caller can't grow it without limit.
    if len(_version_cache) > 32:
        _version_cache.clear()
    _version_cache[key] = result
    return result


async def _probe_version(path: Path) -> str | None:
    res = await run_capture([str(path), "--version"], timeout=10.0)
    text = res.text
    return text.splitlines()[0] if text else None


def _discover_binaries(configured: Path | None) -> list[dict]:
    """llama-server binaries already on this machine, most likely first.
    Deduplicated by resolved path so PATH and a hint dir don't show twice."""
    found: list[dict] = []
    seen: set[Path] = set()

    def add(path: Path, source: str) -> None:
        try:
            real = path.resolve()
        except OSError:
            return
        if real in seen or not path.is_file() or not os.access(path, os.X_OK):
            return
        seen.add(real)
        found.append({"path": str(path), "source": source})

    if configured:
        add(configured, "configured")
    on_path = shutil.which("llama-server")
    if on_path:
        add(Path(on_path), "PATH")
    for hint, source in _BINARY_HINTS:
        add(Path(os.path.expanduser(hint)), source)
    return found


def _toolchain() -> dict:
    """What a source build needs, and whether it is here. Reported per-tool so
    the wizard can name the missing package instead of failing at cmake."""
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    return {
        "git": shutil.which("git"),
        "cmake": shutil.which("cmake"),
        "compiler": cc,
        "make_jobs": os.cpu_count() or 4,
    }


@router.get("/state")
async def state() -> dict:
    """Everything the wizard needs, in one request.

    `step` is the wizard's verdict on where the user is; the frontend renders
    it rather than re-deriving the rule, so the two can't disagree. It is
    computed fresh every call — deleting your models sends you back a step,
    which is the honest answer.
    """
    s = load_settings()
    mgr = get_build_manager()

    bin_path = _expand(s.llama_bin)
    bin_exists = bool(bin_path and bin_path.is_file())
    version = await _binary_version(bin_path) if bin_exists else None
    # A file that exists but won't report a version is not a usable server;
    # treating it as "done" would strand the user at a start button that fails.
    bin_ok = bool(version)

    repo_path = _expand(s.llama_repo)
    repo_ok = bool(repo_path and (repo_path / ".git").exists())

    models_root = _expand(s.hf_models_root)
    model_rows = await models.list_models()

    registry = PresetRegistry()
    own_presets = [c for c in registry.list() if (c.mode or "single") == "single"]

    root_ok = bool(models_root and models_root.is_dir())
    if not bin_ok:
        step = "llama"
    elif not root_ok:
        step = "models_dir"
    elif not model_rows:
        step = "model"
    elif not own_presets:
        step = "preset"
    else:
        step = "done"

    return {
        "step": step,
        # Only the first two steps are prerequisites. Getting a model is the
        # user's business — plenty of people already have GGUFs on an external
        # disk and will copy them in — so the wizard must be able to finish
        # without one, and nothing may nag once these two are satisfied.
        "required_done": bin_ok and root_ok,
        "platform": accel.platform_info().to_dict(),
        "llama": {
            "bin_path": str(bin_path) if bin_path else "",
            "bin_exists": bin_exists,
            "bin_ok": bin_ok,
            "version": version,
            "repo_path": str(repo_path) if repo_path else "",
            "repo_ok": repo_ok,
            "default_repo_path": str(Path.home() / "llama.cpp"),
            "clone_url": LLAMA_CPP_URL,
            "candidates": _discover_binaries(bin_path if bin_exists else None),
        },
        "toolchain": _toolchain(),
        "backends": [b.to_dict() for b in accel.detect_backends()],
        "preferred_backend": accel.preferred_backend(),
        "models": {
            "root": str(models_root) if models_root else "",
            "root_ok": bool(models_root and models_root.is_dir()),
            # What to prefill when the configured root does not exist. The
            # settings default points inside the llama.cpp checkout, which is a
            # poor suggestion on a fresh machine — models outlive the checkout,
            # and a `git clean` there should not eat 200 GB of weights.
            "default_root": str(
                models_root if (models_root and models_root.is_dir()) else Path.home() / "models"
            ),
            "count": len(model_rows),
        },
        "presets": {"count": len(own_presets)},
        "build_active": (mgr.active().to_dict() if mgr.active() else None),
    }


class UseBinaryBody(BaseModel):
    path: str


@router.post("/use-binary")
async def use_binary(body: UseBinaryBody) -> dict:
    """Adopt an existing llama-server. Verifies it runs before saving, so a
    wrong path is rejected here rather than surfacing later as a preset that
    won't start."""
    path = _expand(body.path)
    if not path or not path.is_file():
        raise HTTPException(status_code=400, detail=f"no file at {body.path}")
    if not os.access(path, os.X_OK):
        raise HTTPException(status_code=400, detail=f"{path} is not executable")
    # Uncached: this is an explicit user action, and a chmod +x since the last
    # probe does not change mtime — the cache would keep answering "no".
    version = await _probe_version(path)
    if not version:
        raise HTTPException(
            status_code=400,
            detail=f"{path} did not respond to --version — is it really llama-server?",
        )

    s = load_settings()
    s.llama_bin = str(path)
    # A binary under <repo>/build/bin implies the checkout it came from, which
    # is what the Build page needs. Only filled in when we can prove it.
    repo_guess = path.parent.parent.parent
    if (repo_guess / ".git").exists():
        s.llama_repo = str(repo_guess)
    save_settings(s)

    get_build_manager().set_paths(llama_repo=s.llama_repo, llama_bin=s.llama_bin)
    _rebind_singletons(str(path))
    return {"llama_bin": s.llama_bin, "llama_repo": s.llama_repo, "version": version}


class BuildBody(BaseModel):
    """Install llama.cpp from source: clone if needed, then build."""
    repo_path: str | None = None
    backend: str | None = None       # accel id; None → auto
    jobs: int | None = None


@router.post("/build")
async def build_from_source(body: BuildBody) -> dict:
    """Kick off clone+build and return immediately — the job streams over the
    existing /api/build/stream SSE channel, so the wizard and the Build page
    show the same log."""
    tc = _toolchain()
    missing = [k for k in ("git", "cmake", "compiler") if not tc[k]]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                "missing build tools: " + ", ".join(missing) +
                " — on Debian/Ubuntu: apt install git cmake build-essential"
            ),
        )

    s = load_settings()
    if body.repo_path:
        repo = _expand(body.repo_path)
        if repo is None:
            raise HTTPException(status_code=400, detail="repo_path is empty")
        s.llama_repo = str(repo)
        save_settings(s)

    mgr = get_build_manager()
    mgr.set_paths(llama_repo=s.llama_repo)
    cloning = not mgr.has_repo()   # read before the job starts and creates it
    try:
        job = await mgr.rebuild(
            backend=body.backend, jobs=body.jobs, clone_url=LLAMA_CPP_URL,
        )
    except BuildError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"job": job.to_dict(), "repo_path": s.llama_repo, "cloning": cloning}


class ModelsRootBody(BaseModel):
    path: str
    create: bool = False


@router.post("/models-root")
async def set_models_root(body: ModelsRootBody) -> dict:
    """Set where GGUFs live. Also becomes a scan root, because a models
    directory the scanner doesn't look at is the same as no directory."""
    path = _expand(body.path)
    if path is None:
        raise HTTPException(status_code=400, detail="path is empty")
    if not path.exists():
        if not body.create:
            raise HTTPException(status_code=400, detail=f"{path} does not exist")
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise HTTPException(status_code=400, detail=f"could not create {path}: {e}")
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"{path} is not a directory")

    s = load_settings()
    s.hf_models_root = str(path)
    if str(path) not in s.scan_roots:
        s.scan_roots.append(str(path))
    # Drop the untouched factory default if it never existed. It is provably
    # not a user choice, and leaving it behind means a warning on every scan
    # forever. Any other missing root is left alone — it may be a USB disk
    # that is simply not plugged in right now.
    factory_default = str(Path.home() / "llama.cpp" / "models")
    if factory_default != str(path) and not Path(factory_default).exists():
        s.scan_roots = [r for r in s.scan_roots if r != factory_default]
    save_settings(s)

    scan = await models.full_rescan(s.scan_roots)
    return {"hf_models_root": s.hf_models_root, "scan_roots": s.scan_roots, "scan": scan}


@router.post("/rescan")
async def rescan() -> dict:
    """Re-index the scan roots. The wizard's answer to "I copied my GGUFs in
    from a USB disk" — without it the only way to notice files that appeared
    on disk is to restart the backend."""
    s = load_settings()
    result = await models.full_rescan(s.scan_roots)
    rows = await models.list_models()
    return {"scan": result, "count": len(rows), "roots": s.scan_roots}


def _rebind_singletons(llama_bin: str) -> None:
    """Make a llama_bin change visible to the components that captured it at
    boot. Best-effort: a failure here costs a backend restart, not the setting."""
    try:
        from ..supervisor import get_supervisor
        get_supervisor().rebind(llama_bin)
    except Exception as e:
        log.warning("could not rebind supervisor to %s: %s", llama_bin, e)
    try:
        from ..bench import get_bench_manager
        get_bench_manager().rebind(llama_bin)
    except Exception as e:
        log.warning("could not rebind bench manager to %s: %s", llama_bin, e)
