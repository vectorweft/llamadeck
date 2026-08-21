from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api.bench_api import router as bench_router
from .api.build_api import router as build_router
from .api.features_api import router as features_router
from .api.gpu_api import router as gpu_router
from .api.hf_api import router as hf_router
from .api.metrics_api import router as metrics_router
from .api.models_api import router as models_router
from .api.openai_proxy import router as openai_router
from .api.presets_api import router as presets_router
from .api.router_api import router as router_router
from .api.server_api import router as server_router
from .api.settings_api import router as settings_router
from .api.setup_api import router as setup_router
from .api.system_api import router as system_router
from .bench import get_bench_manager
from .accel import platform_info
from .build import get_build_manager
from .db import init_db
from . import vram_calib
from .gpu_broker import init_broker
from .hf import get_downloader
from .mcp_server import build_http_app, build_mcp
from .metrics import get_metrics_service
from .power import power_loop
from .models import full_rescan
from .presets import PresetRegistry, default_seeds  # noqa: F401
from .processes import LlmService
from .processes.comfyui import ComfyService
from .processes.tts import TtsService
from .router_ini import write_ini
from .settings import LOGS_DIR, STATE_DIR, load_settings
from .supervisor import get_supervisor
from .rpc_server import get_rpc_manager
from .vram import offload_gpus, probe_gpus

log = logging.getLogger("lld")

STATIC_DIR = Path(__file__).parent / "static"

# Regenerated on every interpreter start — including after os.execv, which
# keeps the pid. /health exposes it so clients can detect a real restart.
BOOT_ID = uuid.uuid4().hex


mcp_instance = build_mcp()
mcp_http_app = build_http_app(mcp_instance)


class BootTimer:
    """Wall-clock per startup phase, so a slow launch names its own cause.

    Boot is not one thing: it walks the model roots, adopts stray processes,
    probes GPUs and shells out to the llama.cpp binary more than once. When
    someone says "the first launch takes ages" the only useful question is
    which of those spent the time — and without this the log shows a gap
    between two unrelated INFO lines and nothing else. The summary is one
    line, logged always, ordered slowest first.
    """

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._mark = self._t0
        self._phases: list[tuple[str, float]] = []

    def phase(self, name: str) -> None:
        now = time.perf_counter()
        self._phases.append((name, now - self._mark))
        self._mark = now

    @property
    def total_s(self) -> float:
        return time.perf_counter() - self._t0

    def log_summary(self) -> None:
        slowest = sorted(self._phases, key=lambda kv: kv[1], reverse=True)
        detail = " ".join(f"{n}={d:.2f}s" for n, d in slowest if d >= 0.01)
        log.info("Startup finished in %.2fs — %s", self.total_s, detail or "all phases <10ms")


async def _probe_vram_budget(broker) -> None:
    """Detect the GPU VRAM budget and hand it to the broker, off the boot path.

    Returning without setting anything is a valid outcome: a CPU-only box, or
    one whose GPU tooling is not answering, simply runs with the per-request
    admission check disabled — which is what it did before this was detected at
    all. What must not happen is boot blocking on it.
    """
    try:
        gpus = await probe_gpus()
    except Exception as e:
        log.warning("VRAM probe failed: %s", e)
        return
    targets = offload_gpus(gpus)
    if not targets:
        log.info("no GPU detected — the broker's VRAM admission check stays off")
        return
    total_vram_mb = max(int(getattr(g, "total_mb", 0) or 0) for g in targets)
    broker.set_gpu_total_vram_mb(total_vram_mb)
    log.info(
        "GPU detected: %s | budget total_vram_mb=%d",
        ", ".join(
            f"{g.name} {g.total_mb}MB"
            + ("" if not g.integrated else " (integrated, excluded)")
            for g in gpus
        ),
        total_vram_mb,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    boot = BootTimer()
    await init_db()
    await vram_calib.load()
    s = load_settings()
    boot.phase("db+settings")
    # `llamadeck serve --port X` overrides the settings value; the CLI puts the
    # effective bind here so the banner shows the URL that actually works.
    bind_host = os.environ.get("LLAMADECK_BIND_HOST") or s.controller_bind_host
    bind_port = os.environ.get("LLAMADECK_BIND_PORT") or s.controller_bind_port
    log.info("LlamaDeck %s — http://%s:%s", __version__,
             "127.0.0.1" if bind_host == "0.0.0.0" else bind_host, bind_port)
    p = platform_info()
    log.info("Host: %s · %s/%s%s", p.cpu_name, p.os, p.arch,
             f" · {p.detail}" if p.detail else "")
    log.info("State dir: %s", STATE_DIR)
    log.info("Logs dir: %s", LOGS_DIR)
    added = PresetRegistry().add_missing(default_seeds())
    if added:
        log.info("Seeded missing presets: %s", added)
    # Auto-(re)generate router-models.ini for any router preset that points at
    # a path under STATE_DIR — keeps INI in sync with sibling preset edits on
    # every restart. Use the first router preset's effective models_dir.
    try:
        from pathlib import Path as _P
        for cfg in PresetRegistry().list():
            if getattr(cfg, "mode", "single") != "router":
                continue
            ini_path = _P(cfg.models_preset_path) if cfg.models_preset_path else (STATE_DIR / "router-models.ini")
            md = cfg.models_dir or s.hf_models_root
            write_ini(ini_path, md, router_preset=cfg)
            log.info("router INI refreshed: %s (models_dir=%s, [*] from %s)", ini_path, md, cfg.name)
            break
    except Exception as e:
        log.warning("router INI refresh failed: %s", e)
    boot.phase("presets+router-ini")
    sup = get_supervisor(s.llama_bin)
    # Auto-adopt any llama-server already running on the host
    # A restart re-execs this process, so children it spawned before are still
    # ours — and any that died meanwhile are <defunct> with nobody to wait on
    # them. Clear those before adopting, or a corpse gets adopted as "running".
    reaped = sup.reap_orphan_children()
    if reaped:
        log.info("Reaped %d orphaned llama-server child(ren) from a previous run", reaped)
    for found in sup.scan_existing():
        suggested = found.get("suggested_preset")
        if suggested:
            try:
                await sup.adopt(found["pid"], suggested)
                log.info("Auto-adopted PID %s as preset '%s'", found["pid"], suggested)
            except Exception as e:
                log.warning("Auto-adopt PID %s failed: %s", found["pid"], e)
    boot.phase("adopt-existing")
    try:
        scan_result = await full_rescan(s.scan_roots)
        log.info(
            "Model registry: %d total, %d added, %d updated, %d removed",
            scan_result["total"], scan_result["added"], scan_result["updated"], scan_result["removed"],
        )
    except Exception as e:
        log.warning("initial model scan failed: %s", e)
    boot.phase("model-scan")
    get_downloader(s.hf_models_root, s.hf_token)  # initialise singleton
    get_build_manager(s.llama_repo, s.llama_bin)  # initialise singleton
    get_bench_manager(s.llama_bin)               # initialise singleton

    # ---- What's New (feature tracker) baseline -------------------------
    # Capture the current binary's --help snapshot in the background if
    # missing; the first real diff happens after the next rebuild.
    from .features import get_feature_tracker
    import asyncio as _asyncio_ft
    baseline_task = _asyncio_ft.create_task(
        get_feature_tracker().ensure_baseline(), name="features_baseline",
    )
    metrics_svc = get_metrics_service(sup)
    await metrics_svc.start()
    log.info("Metrics poller started (%.1f Hz)", 1.0 / 0.5)
    boot.phase("singletons+metrics")

    # ---- Power / energy tracker -----------------------------------------
    # Background task that polls GPU+CPU power at 2 Hz and integrates total
    # energy ONLY while at least one LLM slot is busy (idle time excluded).
    import asyncio as _asyncio

    def _llm_busy() -> bool:
        for st in metrics_svc.states.values():
            latest = st.latest()
            if latest and latest.busy_slots > 0:
                return True
        return False

    power_task = _asyncio.create_task(power_loop(_llm_busy))

    # ---- Router health watchdog -----------------------------------------
    # External end-to-end probe loop for router-mode presets. Detects the
    # case where supervisor sees the router process as alive but its
    # internal child llama-server died/zombied and proxy calls return
    # "Could not establish connection". On sustained failure, restarts the
    # preset. See health_watchdog.py for the full rationale.
    from .supervisor import liveness_loop
    liveness_task = _asyncio.create_task(
        liveness_loop(lambda: get_supervisor()),
        name="liveness-poll",
    )
    from .health_watchdog import watchdog_loop
    watchdog_task = _asyncio.create_task(
        watchdog_loop(lambda: get_supervisor()),
        name="health_watchdog",
    )

    # ---- GPU broker ------------------------------------------------------
    # The VRAM budget is auto-detected when it is not pinned in settings, but
    # NOT here: detection shells out to vendor tools, and those are precisely
    # the calls that hang on a machine whose driver is mid-upgrade. On this box
    # a mismatched NVIDIA kernel module made `nvidia-smi` block for 41 seconds,
    # and because the probe sat on the critical path the whole app — health
    # endpoint included — was unreachable for all of it while the launcher
    # waited on /health. It runs in the background now (see _probe_vram_budget)
    # and hands its answer to the broker when it has one; until then the broker
    # behaves exactly as it does on a box with no GPU telemetry.
    total_vram_mb = s.gpu_total_vram_mb

    # LlmService resolves the active preset dynamically at every check
    # (against supervisor state), so no boot-time pin is needed. Users who
    # want a sticky preference can set settings.default_llm_preset.

    services = {
        "llm": LlmService(s, sup),
        "comfy": ComfyService(
            managed_url=s.comfy_managed_url,
            est_vram_mb=s.comfy_est_vram_mb,
            comfy_path=s.comfy_path,
            python_bin=s.comfy_python_bin,
            extra_args=s.comfy_extra_args,
            startup_timeout_s=s.comfy_startup_timeout_s,
            health_path=s.comfy_health_path,
            log_dir=LOGS_DIR,
        ),
        "tts": TtsService(
            managed_url=s.xtts_managed_url,
            est_vram_mb=s.xtts_est_vram_mb,
            tts_path=s.xtts_path,
            python_bin=s.xtts_python_bin,
            server_script=s.xtts_server_script,
            extra_args=s.xtts_extra_args,
            startup_timeout_s=s.xtts_startup_timeout_s,
            health_path=s.xtts_health_path,
            log_dir=LOGS_DIR,
        ),
    }
    broker = init_broker(
        services,
        keepalive_s=s.llm_keepalive_s,
        default_ttl_s=s.gpu_default_ttl_s,
        gpu_total_vram_mb=total_vram_mb,
        external_reserved_vram_mb=s.external_reserved_vram_mb,
        vram_spawn_headroom_mb=s.vram_spawn_headroom_mb,
    )
    await broker.start()
    log.info(
        "GPU broker started (slots=llm/comfy/tts, keepalive=%.0fs, gpu_total=%s, "
        "external_reserved=%d MB, spawn_headroom=%d MB)",
        s.llm_keepalive_s,
        f"{total_vram_mb} MB" if total_vram_mb is not None else "probing…",
        s.external_reserved_vram_mb, s.vram_spawn_headroom_mb,
    )
    vram_probe_task = (
        None if total_vram_mb is not None
        else _asyncio.create_task(_probe_vram_budget(broker), name="vram-budget-probe")
    )
    boot.phase("gpu-broker")

    # ---- Auto-start always-on router presets ---------------------------
    # Without this, `systemctl restart llamadeck` leaves router-mode
    # llama-server (port 8085) dead, and downstream consumers (HFD-2
    # canon ingest, deepen) silently fail because httpx requests time
    # out or return empty. The supervisor only auto-restarts on crashes
    # *during* its own lifetime; restart of the FastAPI host loses that
    # RPC offload servers come up first: a preset pinned to an RPC device
    # cannot resolve that id unless the server behind it is already listening,
    # and the router preset below may be exactly such a preset.
    await get_rpc_manager().start_autostart()
    boot.phase("rpc-autostart")

    # state. Spawn router presets explicitly at boot. (2026-05-27)
    # Only 'is it already running' is needed here; skip the GGUF reads.
    boot_statuses = sup.statuses(vram_estimates=False)
    if not Path(s.llama_bin).exists():
        # Fresh install: llama_bin isn't configured yet. Starting would just
        # fail with "binary not found" — an ERROR on every boot before the
        # user ever opened Settings. Say what to do instead.
        log.info(
            "skipping router preset auto-start: llama-server binary not found "
            "at %s — open the dashboard and run the setup wizard", s.llama_bin,
        )
    else:
        for cfg in PresetRegistry().list():
            if getattr(cfg, "mode", "single") != "router":
                continue
            if boot_statuses.get(cfg.name, {}).get("running"):
                continue  # adopted from a prior session
            try:
                await sup.start(cfg.name)
                log.info("auto-started router preset '%s' at boot", cfg.name)
            except Exception as e:
                log.error("auto-start of router preset '%s' failed: %s",
                          cfg.name, e)
    boot.phase("router-autostart")

    async with mcp_instance.session_manager.run():
        log.info("MCP streamable-http transport mounted at /mcp")
        boot.phase("mcp-mount")
        boot.log_summary()
        yield
    power_task.cancel()
    try:
        await power_task
    except (BaseException,):
        pass
    watchdog_task.cancel()
    try:
        await watchdog_task
    except (BaseException,):
        pass
    liveness_task.cancel()
    try:
        await liveness_task
    except (BaseException,):
        pass
    baseline_task.cancel()
    try:
        await baseline_task
    except (BaseException,):
        pass
    if vram_probe_task is not None:
        vram_probe_task.cancel()
        try:
            await vram_probe_task
        except (BaseException,):
            pass
    await broker.stop()
    await metrics_svc.stop()
    await get_rpc_manager().stop_all()
    log.info("LlamaDeck stopping")


def create_app() -> FastAPI:
    app = FastAPI(title="LlamaDeck", version=__version__, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str | int]:
        # boot_id changes on every process start and is how the UI's restart
        # flow tells the NEW process apart from the old one. It cannot use the
        # pid: os.execv REPLACES the image but KEEPS the pid (that's why child
        # llama-servers survive), so the pid is identical across a restart.
        import os as _os
        return {
            "status": "ok",
            "version": __version__,
            "pid": _os.getpid(),
            "boot_id": BOOT_ID,
        }

    app.include_router(settings_router)
    app.include_router(setup_router)
    app.include_router(presets_router)
    app.include_router(server_router)
    app.include_router(models_router)
    app.include_router(metrics_router)
    app.include_router(hf_router)
    app.include_router(build_router)
    app.include_router(bench_router)
    app.include_router(features_router)
    app.include_router(router_router)
    # GPU broker + OpenAI-compatible proxy (faz-1.5)
    app.include_router(gpu_router)
    app.include_router(openai_router)
    app.include_router(system_router)

    # MCP streamable-http transport — all tools live under /mcp
    #
    # A Starlette Mount only matches "/mcp/…", so a client configured with the
    # obvious `http://host:port/mcp` got a 405 and no explanation. Rewrite that
    # exact path to "/mcp/" before routing so both spellings work.
    class _McpPathAlias:
        def __init__(self, inner):
            self.inner = inner

        async def __call__(self, scope, receive, send):
            if scope.get("type") == "http" and scope.get("path") == "/mcp":
                scope = {**scope, "path": "/mcp/", "raw_path": b"/mcp/"}
            await self.inner(scope, receive, send)

    app.mount("/mcp", mcp_http_app)
    app.add_middleware(_McpPathAlias)

    if STATIC_DIR.exists() and any(STATIC_DIR.iterdir()):
        index = STATIC_DIR / "index.html"

        # Cache policy, and why it matters: everything under /_app is
        # content-hashed by the bundler, so it can be cached forever. The SPA
        # shell that *names* those hashes must not be — without an explicit
        # header browsers apply heuristic freshness to it, keep serving the
        # previous index.html after a UI rebuild, and the app then asks for
        # chunk filenames that no longer exist. The symptom is a half-dead page
        # ("Failed to fetch dynamically imported module") that a normal reload
        # does not fix, which is a miserable thing to hand someone who just
        # rebuilt the frontend.
        IMMUTABLE = "public, max-age=31536000, immutable"
        NO_CACHE = "no-cache"

        class _ImmutableAssets(StaticFiles):
            def file_response(self, *args, **kwargs):
                resp = super().file_response(*args, **kwargs)
                resp.headers["cache-control"] = IMMUTABLE
                return resp

        def _shell() -> FileResponse:
            return FileResponse(index, headers={"cache-control": NO_CACHE})

        # Serve built assets (JS/CSS/images) under /_app and any other files.
        app.mount("/_app", _ImmutableAssets(directory=STATIC_DIR / "_app"), name="app_assets")

        @app.get("/")
        async def root():
            return _shell()

        @app.get("/{full_path:path}")
        async def spa_fallback(full_path: str):
            # Any non-API path returns the SPA shell so client-side routing works.
            # API routes are registered above so they take precedence.
            # Unknown /api paths must get a proper JSON 404, NOT the SPA HTML —
            # otherwise an old backend + new UI window shows the client a
            # meaningless parse error like "Unexpected token '<'".
            if full_path.startswith("api/"):
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=404,
                    content={"detail": f"unknown API path: /{full_path} (is the service running old code?)"},
                )
            target = STATIC_DIR / full_path
            if target.is_file():
                # Unhashed extras (favicon, manifest…) — revalidate rather than
                # pin, they are replaced in place by a rebuild.
                return FileResponse(target, headers={"cache-control": NO_CACHE})
            return _shell()
    else:
        @app.get("/")
        async def root_placeholder() -> dict[str, str]:
            return {
                "status": "ok",
                "ui": "not-built",
                "hint": "Run `cd frontend && npm install && npm run build` to build the UI",
            }

    return app


app = create_app()
