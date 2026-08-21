"""BuildManager — obtain and rebuild the local llama.cpp checkout.

Pipeline: `git clone` (first run) or `git pull` -> `cmake -B build
-DGGML_CUDA=... -DCMAKE_BUILD_TYPE=Release` -> `cmake --build build -j<N>`.
Each step streams stdout+stderr line-by-line to SSE subscribers and to a log
file, and persists progress to the `builds` SQLite table.

Only one concurrent build is allowed; starting a second while one is running
raises BuildError.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


from . import accel
from .db import connect
from .procutil import run_capture
from .settings import LOGS_DIR, ensure_state_dirs, load_settings, save_settings

log = logging.getLogger(__name__)

#: `llama-server --version` prints its banner on stderr, and the shape of that
#: banner changed upstream. Both are still in the wild:
#:
#:   version: 4589 (a1b2c3d4)                          — through mid-2025
#:   version: 0.1.0-dev (build 10449, commit 0d9ceae1e) — current
#:
#: Only the first form was matched, so on any recent binary `commit` came back
#: empty and everything keyed on it went quiet: the What's New baseline was
#: never captured (capture_snapshot bails without a commit), so no rebuild ever
#: produced a feature diff, and the build page showed no version at all.
_VERSION_LINES = (
    re.compile(r"version:\s*\S+\s*\(\s*build\s+(\d+)\s*,\s*commit\s+([0-9a-f]{6,40})\s*\)", re.I),
    re.compile(r"version:\s*(\d+)\s*\(([0-9a-f]{6,40})\)", re.I),
)


def parse_version_banner(text: str) -> tuple[int | None, str | None]:
    """(build_number, commit) from a --version banner, or (None, None)."""
    for pat in _VERSION_LINES:
        m = pat.search(text)
        if m:
            return int(m.group(1)), m.group(2)
    return None, None

#: Build tools redraw progress with a bare \r. Treat it as a line break so
#: each redraw becomes its own log line instead of accumulating into one.
_LINE_SPLIT = re.compile(r"\r\n|\r|\n")

#: Upstream llama.cpp. The setup wizard clones this when a machine has no
#: checkout; kept here (not in the API layer) so the clone and the build that
#: follows it agree on what "the repo" is.
LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp"


def _node_major(node_path: str) -> int:
    """Major version of the Node binary at `node_path` (0 if it can't be read)."""
    try:
        out = subprocess.run(
            [node_path, "-v"], capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return 0
    m = re.match(r"v(\d+)\.", out.stdout.strip())
    return int(m.group(1)) if m else 0


def _node_bin_dir() -> str | None:
    """A directory to prepend to PATH so the build sees Node >=20, or None if
    the current PATH already provides one.

    llama.cpp's web UI build shells out to `npm`, which needs Node >=20. The
    systemd unit's PATH only carries the distro Node 18, so fall back to the
    newest nvm-installed Node. Resolved dynamically (newest >=20) rather than a
    hardcoded version path, so it keeps working after `nvm install <newer>`.
    """
    on_path = shutil.which("node")
    if on_path and _node_major(on_path) >= 20:
        return None
    nvm_root = Path.home() / ".nvm" / "versions" / "node"
    best: tuple[tuple[int, int, int], Path] | None = None
    if nvm_root.is_dir():
        for d in nvm_root.iterdir():
            m = re.match(r"v(\d+)\.(\d+)\.(\d+)$", d.name)
            if not m:
                continue
            ver = (int(m[1]), int(m[2]), int(m[3]))
            if ver[0] < 20:
                continue
            if (d / "bin" / "node").exists() and (best is None or ver > best[0]):
                best = (ver, d / "bin")
    return str(best[1]) if best else None


class BuildError(RuntimeError):
    pass


class BuildStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class VersionInfo:
    build_number: int | None
    commit: str | None
    raw: str


@dataclass
class Commit:
    sha: str
    subject: str


@dataclass
class BuildJob:
    id: int
    started_at: float
    finished_at: float | None = None
    from_commit: str | None = None
    to_commit: str | None = None
    status: BuildStatus = BuildStatus.RUNNING
    log_path: str | None = None
    current_step: str = ""
    # live-only (not persisted)
    lines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "from_commit": self.from_commit,
            "to_commit": self.to_commit,
            "status": self.status.value,
            "log_path": self.log_path,
            "current_step": self.current_step,
            "duration_seconds": (self.finished_at - self.started_at) if self.finished_at else (time.time() - self.started_at),
        }


class BuildManager:
    """Singleton. `get_build_manager()` initialises with llama_repo from settings."""

    def __init__(self, llama_repo: str, llama_bin: str):
        self.llama_repo = Path(llama_repo)
        self.llama_bin = Path(llama_bin)
        self._active: BuildJob | None = None
        self._lock = asyncio.Lock()
        self._subscribers: set[asyncio.Queue[str]] = set()

    def set_paths(self, llama_repo: str | None = None, llama_bin: str | None = None) -> None:
        """Repoint the manager without a process restart. The singleton is
        constructed from settings at boot, but the setup wizard chooses these
        paths afterwards — without this, a fresh install would build into the
        default `~/llama.cpp` no matter what the user picked."""
        if llama_repo:
            self.llama_repo = Path(os.path.expanduser(llama_repo))
        if llama_bin:
            self.llama_bin = Path(os.path.expanduser(llama_bin))

    @property
    def built_binary(self) -> Path:
        """Where a successful cmake build leaves llama-server."""
        return self.llama_repo / "build" / "bin" / "llama-server"

    # ---------- version / git inspection (read-only, fast) ----------

    async def current_version(self) -> VersionInfo:
        """Run `llama-server --version` and parse its stderr banner."""
        if not self.llama_bin.exists():
            return VersionInfo(build_number=None, commit=None, raw=f"{self.llama_bin} not found")
        res = await run_capture([str(self.llama_bin), "--version"], timeout=10.0)
        if res.error:
            return VersionInfo(build_number=None, commit=None, raw=f"error: {res.error}")
        if res.timed_out:
            return VersionInfo(
                build_number=None, commit=None,
                raw=f"error: {self.llama_bin} did not answer --version in 10s",
            )
        text = res.text
        build_number, commit = parse_version_banner(text)
        if commit:
            return VersionInfo(build_number=build_number, commit=commit, raw=text)
        return VersionInfo(build_number=None, commit=None, raw=text)

    async def head_commit(self) -> str | None:
        """`git -C <repo> rev-parse HEAD` → short sha."""
        res = await run_capture(
            ["git", "-C", str(self.llama_repo), "rev-parse", "--short", "HEAD"],
            timeout=5.0,
        )
        return res.stdout.strip() or None if res.ok else None

    async def default_branch(self) -> str:
        """Return the tracked remote branch (origin/HEAD target). Falls back to master."""
        res = await run_capture(
            ["git", "-C", str(self.llama_repo), "symbolic-ref", "--quiet",
             "--short", "refs/remotes/origin/HEAD"],
            timeout=5.0,
        )
        ref = res.stdout.strip()
        if ref.startswith("origin/"):
            return ref.split("/", 1)[1]
        return "master"

    def has_repo(self) -> bool:
        return (self.llama_repo / ".git").exists()

    def _ensure_repo(self) -> None:
        """Raise an actionable BuildError instead of a raw git failure (or a
        FileNotFoundError → HTTP 500) on machines that have no source
        checkout — fresh installs, or brew/prebuilt-binary setups."""
        if not self.has_repo():
            raise BuildError(
                f"llama.cpp source checkout not found at {self.llama_repo} — "
                "run the setup wizard to clone it, or point Settings → llama.cpp "
                "paths at an existing checkout. Building from source is optional; "
                "a prebuilt llama-server binary works without it."
            )

    def _check_clone_target(self) -> None:
        """Validate the clone destination before a job is created, so a bad
        path fails as a 4xx on the request instead of as a dead build job the
        user has to go read a log to understand."""
        dest = self.llama_repo
        if dest.exists() and any(dest.iterdir()):
            raise BuildError(
                f"{dest} already exists and is not empty, but has no .git — "
                "pick an empty directory, or point at the existing checkout."
            )
        parent = dest.parent
        if not parent.exists():
            raise BuildError(f"parent directory {parent} does not exist")
        if not os.access(parent, os.W_OK):
            raise BuildError(f"no write permission on {parent}")
        if not shutil.which("git"):
            raise BuildError("git not found on PATH — install git first (Debian/Ubuntu: apt install git)")

    async def check_updates(self) -> dict:
        """`git fetch` then list commits on origin/<branch> that are ahead of HEAD."""
        self._ensure_repo()
        branch = await self.default_branch()
        try:
            fetch = await asyncio.create_subprocess_exec(
                "git", "-C", str(self.llama_repo), "fetch", "--quiet",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            raise BuildError("git not found on PATH — install git to use the Build page")
        try:
            _, err = await asyncio.wait_for(fetch.communicate(), timeout=60.0)
        except asyncio.TimeoutError:
            fetch.kill()
            raise BuildError("git fetch timed out")
        if fetch.returncode != 0:
            raise BuildError(f"git fetch failed: {err.decode(errors='replace').strip()}")

        proc = await asyncio.create_subprocess_exec(
            "git", "-C", str(self.llama_repo), "log", f"HEAD..origin/{branch}",
            "--pretty=format:%h\t%s", "--no-merges",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            raise BuildError(f"git log failed: {err.decode(errors='replace').strip()}")
        commits: list[Commit] = []
        for line in out.decode(errors="replace").splitlines():
            if not line.strip():
                continue
            sha, _, subject = line.partition("\t")
            commits.append(Commit(sha=sha.strip(), subject=subject.strip()))
        head = await self.head_commit()
        return {
            "branch": branch,
            "head_commit": head,
            "ahead": len(commits),
            "commits": [{"sha": c.sha, "subject": c.subject} for c in commits],
        }

    # ---------- rebuild (long-running, one at a time) ----------

    def active(self) -> BuildJob | None:
        return self._active

    async def rebuild(
        self,
        cuda: bool | None = None,
        jobs: int | None = None,
        backend: str | None = None,
        clone_url: str | None = None,
    ) -> BuildJob:
        """Start a rebuild. `backend` is an accel.py id (auto/cuda/metal/hip/
        vulkan/cpu); `cuda` is the pre-0.2 boolean kept for older clients and
        only consulted when `backend` is absent.

        `clone_url` turns the first pipeline step from `git pull` into
        `git clone` when there is no checkout yet — the setup wizard's
        "install llama.cpp for me" path. Without it a missing repo is still an
        error, so an ordinary rebuild can never silently re-clone."""
        if backend is None:
            backend = accel.AUTO if cuda is None else (accel.CUDA if cuda else accel.CPU)
        if self._active and self._active.status == BuildStatus.RUNNING:
            raise BuildError(f"a build is already running (id={self._active.id})")
        cloning = bool(clone_url) and not self.has_repo()
        if cloning:
            self._check_clone_target()
        else:
            self._ensure_repo()

        ensure_state_dirs()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = LOGS_DIR / f"llama-cpp-build-{stamp}.log"
        from_commit = await self.head_commit()

        async with connect() as db:
            cur = await db.execute(
                "INSERT INTO builds (started_at, from_commit, status, log_path) VALUES (?, ?, ?, ?)",
                (time.time(), from_commit, BuildStatus.RUNNING.value, str(log_path)),
            )
            await db.commit()
            build_id = cur.lastrowid

        job = BuildJob(
            id=build_id,
            started_at=time.time(),
            from_commit=from_commit,
            status=BuildStatus.RUNNING,
            log_path=str(log_path),
        )
        self._active = job

        asyncio.create_task(
            self._run(job, backend=backend, jobs=jobs, clone_url=clone_url if cloning else None),
            name=f"build-{build_id}",
        )
        return job

    @staticmethod
    def cached_backend(build_dir: Path) -> str | None:
        """Which backend the existing cmake cache was configured with, or None
        if there is no cache to read."""
        cache = build_dir / "CMakeCache.txt"
        try:
            text = cache.read_text(errors="replace")
        except OSError:
            return None
        for flag, backend in (
            ("GGML_CUDA:BOOL=ON", accel.CUDA),
            ("GGML_HIP:BOOL=ON", accel.HIP),
            ("GGML_VULKAN:BOOL=ON", accel.VULKAN),
            ("GGML_METAL:BOOL=ON", accel.METAL),
        ):
            if flag in text:
                return backend
        return accel.CPU

    @staticmethod
    def cached_generator(build_dir: Path) -> str | None:
        """Which cmake generator the existing build directory was configured
        with, or None if there is no cache to read."""
        try:
            text = (build_dir / "CMakeCache.txt").read_text(errors="replace")
        except OSError:
            return None
        for line in text.splitlines():
            if line.startswith("CMAKE_GENERATOR:INTERNAL="):
                return line.split("=", 1)[1].strip() or None
        return None

    @staticmethod
    def preferred_generator() -> str:
        """Ninja when it is installed, else cmake's platform default.

        Ninja schedules a full dependency graph instead of recursing per
        directory, so incremental rebuilds — the common case here, since the
        Build page exists to pull upstream and rebuild — start compiling
        immediately instead of walking the tree first. It also pairs better
        with ccache, because nothing is serialised behind a directory.
        """
        return "Ninja" if shutil.which("ninja") else ""

    def _stale_cache_reason(self, build_dir: Path, backend: str, generator: str) -> str | None:
        """Why the existing cmake cache cannot be reused, or None if it can.

        Two things poison a cache: a different backend (it pins the compiler —
        nvcc vs hipcc vs clang — and cmake will happily reuse it and then fail),
        and a different generator (cmake refuses outright, with an error most
        people read as "the build is broken").
        """
        prev_backend = self.cached_backend(build_dir)
        if prev_backend is not None and prev_backend != backend:
            return f"backend changed ({prev_backend} -> {backend})"
        prev_gen = self.cached_generator(build_dir)
        # An empty `generator` means "cmake's default", which is what a cache
        # holding "Unix Makefiles" already is — not a change.
        if generator and prev_gen is not None and prev_gen != generator:
            return f"generator changed ({prev_gen} -> {generator})"
        # `generator` is empty only when ninja is absent. If the cache was
        # configured with Ninja, cmake would reuse it and fail on a missing
        # binary — a build broken by uninstalling an unrelated package.
        if not generator and prev_gen == "Ninja":
            return "generator Ninja is no longer installed"
        return None

    async def _run(self, job: BuildJob, backend: str, jobs: int | None, clone_url: str | None = None) -> None:
        j = jobs or max(1, os.cpu_count() or 4)
        build_dir = self.llama_repo / "build"
        log_fh = open(job.log_path, "a", buffering=1)  # line-buffered
        try:
            def emit(line: str) -> None:
                stamped = f"[{datetime.now().strftime('%H:%M:%S')}] {line}"
                log_fh.write(stamped + "\n")
                job.lines.append(stamped)
                if len(job.lines) > 5000:
                    del job.lines[:-5000]
                for q in list(self._subscribers):
                    try:
                        q.put_nowait(stamped)
                    except asyncio.QueueFull:
                        pass

            resolved, accel_flags, accel_path = accel.resolve(backend)
            p = accel.platform_info()
            emit(
                f"[LlamaDeck] build {job.id} starting on {p.os}/{p.arch} "
                f"(backend={resolved}{' via auto' if backend == accel.AUTO else ''}, -j{j})"
            )

            if clone_url:
                job.current_step = "git clone"
                emit(f"[LlamaDeck] ── {job.current_step}")
                self.llama_repo.parent.mkdir(parents=True, exist_ok=True)
                # Full history, not --depth 1: the What's New feed diffs commit
                # ranges between builds, and `git pull` on a shallow clone
                # degrades badly. ~250 MB once is worth it.
                # --progress forces git to emit counters even though stdout is
                # a pipe, so the UI shows movement during the long download.
                rc = await self._stream_cmd(
                    ["git", "clone", "--progress", clone_url, str(self.llama_repo)],
                    emit, cwd=str(self.llama_repo.parent),
                )
                if rc != 0:
                    raise BuildError(f"git clone failed (exit {rc})")
            else:
                job.current_step = "git pull"
                emit(f"[LlamaDeck] ── {job.current_step}")
                # --autostash: tolerate auto-regenerated files (e.g. tools/ui/package-lock.json
                # touched by npm) by stashing the dirty tree, fast-forwarding, then popping.
                rc = await self._stream_cmd(
                    ["git", "-C", str(self.llama_repo), "pull", "--ff-only", "--autostash"], emit,
                )
                if rc != 0:
                    raise BuildError(f"git pull failed (exit {rc})")

            job.current_step = "cmake configure"
            emit(f"[LlamaDeck] ── {job.current_step}")
            # The web UI build step (tools/ui) runs `npm`, which needs Node >=20.
            # Make sure the build subprocesses get one regardless of the unit's
            # PATH (which only carries the distro Node 18).
            build_env = self._build_env(emit, accel_path)
            generator = self.preferred_generator()
            cfg = [
                "cmake", "-B", str(build_dir), "-S", str(self.llama_repo),
                "-DCMAKE_BUILD_TYPE=Release",
                *(["-G", generator] if generator else []),
                *accel_flags,
            ]
            if generator:
                emit(f"[LlamaDeck] generator: {generator}")
            # A cache pinned to another backend or another generator cannot be
            # reused — see _stale_cache_reason. Dropping it costs one full
            # rebuild; keeping it costs a confusing failure.
            stale_reason = self._stale_cache_reason(build_dir, resolved, generator)
            if stale_reason:
                if stale_reason.startswith("generator"):
                    # Object files from another generator are not reusable, and
                    # its driver files (Makefile / build.ninja) would survive as
                    # litter pointing at a cache that no longer exists. Start
                    # the directory over — nothing of value is lost.
                    emit(f"[LlamaDeck] {stale_reason} -> removing {build_dir}")
                    shutil.rmtree(build_dir, ignore_errors=True)
                else:
                    emit(f"[LlamaDeck] {stale_reason} -> wiping {build_dir / 'CMakeCache.txt'}")
                    for stale in (build_dir / "CMakeCache.txt", build_dir / "CMakeFiles"):
                        if stale.is_dir():
                            shutil.rmtree(stale, ignore_errors=True)
                        elif stale.exists():
                            stale.unlink()
            rc = await self._stream_cmd(cfg, emit, env=build_env)
            if rc != 0:
                raise BuildError(f"cmake configure failed (exit {rc})")

            job.current_step = f"cmake --build -j{j}"
            emit(f"[LlamaDeck] ── {job.current_step}")
            rc = await self._stream_cmd(
                ["cmake", "--build", str(build_dir), "-j", str(j)],
                emit, env=build_env,
            )
            if rc != 0:
                raise BuildError(f"cmake build failed (exit {rc})")

            job.to_commit = await self.head_commit()
            job.status = BuildStatus.SUCCESS
            job.finished_at = time.time()
            job.current_step = "done"
            emit(
                f"[LlamaDeck] build {job.id} SUCCESS in "
                f"{(job.finished_at - job.started_at):.1f}s — {job.from_commit} → {job.to_commit}"
            )
            self._adopt_binary(emit)
            # What's New: compare the new binary against the old one and
            # produce feature cards. Lazy import (breaks features -> build cycle).
            from .features import get_feature_tracker

            async def _feature_scan() -> None:
                try:
                    await get_feature_tracker().run_scan()
                except Exception as e:
                    log.warning("feature scan after build %s failed: %s", job.id, e)

            asyncio.create_task(_feature_scan(), name=f"feature-scan-{job.id}")
        except Exception as e:
            job.status = BuildStatus.FAILED
            job.finished_at = time.time()
            job.current_step = "failed"
            err = f"[LlamaDeck] build {job.id} FAILED: {e}"
            log_fh.write(err + "\n")
            job.lines.append(err)
            for q in list(self._subscribers):
                try:
                    q.put_nowait(err)
                except asyncio.QueueFull:
                    pass
            log.exception("build %s failed", job.id)
        finally:
            log_fh.close()
            async with connect() as db:
                await db.execute(
                    "UPDATE builds SET finished_at=?, to_commit=?, status=? WHERE id=?",
                    (job.finished_at, job.to_commit, job.status.value, job.id),
                )
                await db.commit()
            # Leave job as `self._active` so UI can display the final state;
            # overwritten when the next rebuild starts.

    def _adopt_binary(self, emit) -> bool:
        """Point settings at the binary this build just produced, when the
        configured one does not exist.

        Without this a fresh install sits through a 20-minute build and still
        cannot start a model: `llama_bin` holds the conventional default path,
        which was never more than a hint. A configured binary that *does*
        exist is left alone — the user may be deliberately building a second
        checkout they are not ready to switch to.
        """
        built = self.built_binary
        if not built.exists():
            return False
        s = load_settings()
        current = Path(os.path.expanduser(s.llama_bin)) if s.llama_bin else None
        if current == built or (current and current.exists()):
            return False
        s.llama_bin = str(built)
        save_settings(s)
        self.llama_bin = built
        emit(f"[LlamaDeck] llama_bin was not set to an existing binary -> now {built}")

        # Rebind the singletons that captured llama_bin at boot, so the user
        # can start a preset straight away instead of restarting the backend.
        try:
            from .supervisor import get_supervisor
            get_supervisor().rebind(str(built))
        except Exception as e:          # never fail a successful build on this
            log.warning("could not rebind supervisor to %s: %s", built, e)
        try:
            from .bench import get_bench_manager
            get_bench_manager().rebind(str(built))
        except Exception as e:
            log.warning("could not rebind bench manager to %s: %s", built, e)
        return True

    def _build_env(self, emit, extra_path: list[str] | None = None) -> dict[str, str] | None:
        """Environment for build subprocesses: a Node >=20 on PATH when the
        inherited environment lacks one, plus the selected backend's toolchain
        directory (nvcc/hipcc often live outside a service's PATH). Returns
        None to inherit the environment unchanged."""
        node_dir = _node_bin_dir()
        prepend = list(extra_path or [])
        if node_dir:
            prepend.insert(0, node_dir)
        if not prepend:
            return None
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join(prepend) + os.pathsep + env.get("PATH", "")
        if node_dir:
            ver = _node_major(str(Path(node_dir) / "node"))
            emit(f"[LlamaDeck] node: using {node_dir} (v{ver}.x) for web UI build")
        for d in (extra_path or []):
            emit(f"[LlamaDeck] toolchain: added {d} to PATH")
        return env

    async def _stream_cmd(
        self, cmd: list[str], emit, env: dict[str, str] | None = None, cwd: str | None = None,
    ) -> int:
        """Run `cmd`, forwarding stdout+stderr line-by-line to emit(). Returns exit code."""
        emit(f"$ {' '.join(cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd or str(self.llama_repo),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )

        async def pump(stream: asyncio.StreamReader) -> None:
            # Read raw chunks rather than readline(): git and cmake redraw
            # progress in place with a bare \r and no newline until the phase
            # ends. readline() would buffer an entire "Receiving objects
            # 0%…100%" sequence into one 200 kB line that arrives only once the
            # download is already over — no movement during the slow part, and
            # a log panel with one unreadable line in it.
            buf = ""
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                buf += chunk.decode(errors="replace")
                parts = _LINE_SPLIT.split(buf)
                buf = parts.pop()          # trailing partial line, keep buffering
                for p in parts:
                    if p.strip():
                        emit(p.rstrip())
            if buf.strip():
                emit(buf.rstrip())

        assert proc.stdout is not None
        await pump(proc.stdout)
        return await proc.wait()

    # ---------- pub/sub for live UI ----------

    def subscribe(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=500)
        # Prime with existing lines so late-joining UIs see scrollback
        if self._active:
            for line in list(self._active.lines)[-200:]:
                try:
                    q.put_nowait(line)
                except asyncio.QueueFull:
                    break
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        self._subscribers.discard(q)

    # ---------- history from DB ----------

    async def history(self, limit: int = 20) -> list[dict]:
        async with connect() as db:
            cur = await db.execute(
                "SELECT * FROM builds ORDER BY started_at DESC LIMIT ?", (limit,),
            )
            rows = await cur.fetchall()
        return [dict(r) for r in rows]


_instance: BuildManager | None = None


def get_build_manager(llama_repo: str | None = None, llama_bin: str | None = None) -> BuildManager:
    global _instance
    if _instance is None:
        s = load_settings()
        _instance = BuildManager(
            llama_repo=llama_repo or s.llama_repo,
            llama_bin=llama_bin or s.llama_bin,
        )
    return _instance
