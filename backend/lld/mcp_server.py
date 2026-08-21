"""MCP server for LlamaDeck — exposes supervisor, registry, metrics, and HF download
as MCP tools.

All tools hit the LlamaDeck REST API on 127.0.0.1:<controller_bind_port>; the FastAPI
app is the single source of truth for state. This lets the same tool code
serve both the stdio transport (spawned by Claude Desktop / Code) and the
HTTP+SSE transport mounted under /mcp in the main app.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from .settings import load_settings

# mcp 2.0 renamed the high-level server: `mcp.server.fastmcp.FastMCP` became
# `mcp.server.mcpserver.MCPServer`, and `streamable_http_path` moved from the
# constructor to `streamable_http_app()`. The decorator/tool surface we use is
# identical in both, so a two-line shim keeps one code path working on 1.x and
# 2.x — important because `uvx llamadeck` always resolves the newest mcp.
try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as _McpServer

    MCP_MAJOR = 2
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _McpServer

    MCP_MAJOR = 1

log = logging.getLogger(__name__)


def _base_url() -> str:
    s = load_settings()
    host = s.controller_bind_host
    if host == "0.0.0.0":
        host = "127.0.0.1"
    return f"http://{host}:{s.controller_bind_port}"


async def _get(path: str, params: dict | None = None) -> Any:
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.get(_base_url() + path, params=params)
        r.raise_for_status()
        return r.json()


async def _post(path: str, json: dict | None = None) -> Any:
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.post(_base_url() + path, json=json)
        r.raise_for_status()
        return r.json()


def build_mcp() -> Any:
    # When mounted under /mcp in the parent FastAPI app, the transport must
    # serve at the mount root so the public path is /mcp (not /mcp/mcp). On
    # 1.x that is a constructor setting; on 2.x it is an argument to
    # streamable_http_app() — see build_http_app below.
    kwargs: dict[str, Any] = {} if MCP_MAJOR >= 2 else {"streamable_http_path": "/"}
    mcp = _McpServer(
        "llamadeck",
        instructions=(
            "LlamaDeck exposes control over llama.cpp's llama-server "
            "processes: list/start/stop presets, inspect live metrics, browse "
            "the model registry, and download GGUF files from HuggingFace."
        ),
        **kwargs,
    )

    # --- Presets / models (read-only inventory) ---

    @mcp.tool()
    async def list_presets() -> list[dict]:
        """List every configured preset (llama-server launch config)."""
        return await _get("/api/presets")

    @mcp.tool()
    async def list_models(family: str | None = None) -> list[dict]:
        """List scanned GGUF models. Optional `family` filter (Qwen3.6, Gemma, ...)."""
        params = {"family": family} if family else None
        return await _get("/api/models", params=params)

    @mcp.tool()
    async def model_families() -> list[str]:
        """Return distinct families present in the model registry."""
        return await _get("/api/models/families")

    @mcp.tool()
    async def rescan_models() -> dict:
        """Rescan `scan_roots` from disk and refresh the model registry."""
        return await _post("/api/models/scan")

    @mcp.tool()
    async def get_model_defaults(path: str, preset: str | None = None) -> dict:
        """Return the recommended sampling defaults + arch metadata for a GGUF file.

        Reads `general.sampling.*` keys embedded by the producer (Unsloth does this
        for Qwen3.6), merges live `/props` if the preset is running, and falls back
        to a curated family table otherwise. Use this before creating a preset or
        deciding sampling params — don't guess."""
        params = {"path": path}
        if preset:
            params["preset"] = preset
        return await _get("/api/models/defaults", params=params)

    # --- Server control ---

    @mcp.tool()
    async def get_server_statuses() -> dict:
        """Return per-preset status: running/adopted, PID, uptime, port, config."""
        return await _get("/api/server/statuses")

    @mcp.tool()
    async def get_vram() -> dict:
        """Return GPU VRAM report (total/used/free) + sum of active preset estimates."""
        return await _get("/api/server/vram")

    @mcp.tool()
    async def start_preset(preset: str) -> dict:
        """Start the llama-server process for the named preset."""
        return await _post(f"/api/server/start/{preset}")

    @mcp.tool()
    async def stop_preset(preset: str) -> dict:
        """Stop (SIGTERM then SIGKILL) the llama-server process for the preset."""
        return await _post(f"/api/server/stop/{preset}")

    @mcp.tool()
    async def restart_preset(preset: str) -> dict:
        """Restart the llama-server process for the preset (stop + start)."""
        return await _post(f"/api/server/restart/{preset}")

    @mcp.tool()
    async def switch_preset(to_preset: str, from_preset: str | None = None) -> dict:
        """Stop `from_preset` (or whichever is running) and start `to_preset`.

        Use this to free VRAM for a different model without a UI round-trip.
        """
        return await _post(
            "/api/server/switch",
            json={"to_preset": to_preset, "from_preset": from_preset},
        )

    @mcp.tool()
    async def wait_until_ready(preset: str, timeout: float = 120.0) -> dict:
        """Block until the preset's llama-server /health returns 200 OK,
        or `timeout` seconds elapse. Returns {ready: bool, elapsed_seconds, ...}.

        Pair with start_preset for VRAM-swap orchestration:
          1. stop_preset(llm)  — frees VRAM
          2. (run ComfyUI / other GPU job)
          3. start_preset(llm) — spawns llama-server (returns before model loads)
          4. wait_until_ready(llm) — blocks until the model is actually serving
          5. send completion requests
        """
        # Bypass the 30 s _post timeout cap — wait_ready may legitimately take
        # longer than that to load a 27B at long context.
        async with httpx.AsyncClient(timeout=max(timeout + 10, 30)) as c:
            r = await c.get(_base_url() + f"/api/server/wait_ready/{preset}",
                            params={"timeout": timeout})
            r.raise_for_status()
            return r.json()

    @mcp.tool()
    async def tail_logs(preset: str, n: int = 200) -> dict:
        """Return the last `n` stdout/stderr lines of a preset's llama-server."""
        return await _get(f"/api/server/logs/tail/{preset}", params={"n": n})

    # --- Metrics ---

    @mcp.tool()
    async def get_metrics_snapshot(history_n: int = 30) -> dict:
        """Snapshot of per-preset metrics: latest frame + last `history_n` frames
        (2 Hz). Includes instant decode tok/s, KV-cache, per-slot state, queue depth."""
        return await _get("/api/metrics/snapshot", params={"history_n": history_n})

    # --- HuggingFace ---

    @mcp.tool()
    async def hf_search(q: str, limit: int = 10) -> dict:
        """Search HuggingFace for GGUF models matching `q` (name / author)."""
        return await _get("/api/hf/search", params={"q": q, "limit": limit})

    @mcp.tool()
    async def hf_files(repo_id: str) -> dict:
        """List GGUF files in a HuggingFace repo with size and inferred brand/series."""
        return await _get("/api/hf/files", params={"repo_id": repo_id})

    @mcp.tool()
    async def hf_classify(repo_id: str, filename: str | None = None) -> dict:
        """Return `(brand, series, base_model)` LlamaDeck would use for the 3-level
        download target: models/<brand>/<series>/<base_model>/<file>."""
        params = {"repo_id": repo_id}
        if filename:
            params["filename"] = filename
        return await _get("/api/hf/classify", params=params)

    @mcp.tool()
    async def hf_download(
        repo_id: str,
        filename: str,
        brand: str | None = None,
        series: str | None = None,
        base_model: str | None = None,
        revision: str = "main",
    ) -> dict:
        """Enqueue a download into models/<brand>/<series>/<base_model>/.
        Returns a DownloadJob (poll via hf_jobs)."""
        return await _post(
            "/api/hf/download",
            json={
                "repo_id": repo_id,
                "filename": filename,
                "brand": brand,
                "series": series,
                "base_model": base_model,
                "revision": revision,
            },
        )

    @mcp.tool()
    async def hf_jobs() -> dict:
        """List all HuggingFace download jobs (queued/in_progress/done/failed)."""
        return await _get("/api/hf/jobs")

    @mcp.tool()
    async def hf_job(job_id: str) -> dict:
        """Get a single download job by id."""
        return await _get(f"/api/hf/jobs/{job_id}")

    # --- llama.cpp build manager ---

    @mcp.tool()
    async def llama_version() -> dict:
        """Return the current llama-server build number, commit, and banner."""
        return await _get("/api/build/version")

    @mcp.tool()
    async def llama_check_updates() -> dict:
        """`git fetch` and list commits on origin that are ahead of local HEAD."""
        return await _get("/api/build/check")

    @mcp.tool()
    async def llama_backends() -> dict:
        """Compute backends this machine can build (cuda/metal/hip/vulkan/cpu),
        which one `auto` picks, and what the current build dir uses. Call this
        before llama_rebuild instead of assuming NVIDIA."""
        return await _get("/api/build/backends")

    @mcp.tool()
    async def llama_rebuild(backend: str = "auto", jobs: int | None = None) -> dict:
        """Rebuild llama.cpp (git pull + cmake). Streams live log via /api/build/stream.
        `backend` is auto | cuda | metal | hip | vulkan | cpu — `auto` picks the
        best toolchain present. Only one concurrent build is allowed."""
        return await _post("/api/build/rebuild", json={"backend": backend, "jobs": jobs})

    @mcp.tool()
    async def llama_build_active() -> dict:
        """Return the currently running (or most recently finished) build job."""
        return await _get("/api/build/active")

    @mcp.tool()
    async def llama_build_history(limit: int = 20) -> list[dict]:
        """Return the last `limit` build records from the DB."""
        return await _get("/api/build/history", params={"limit": limit})

    # --- llama-bench ---

    @mcp.tool()
    async def bench_run(
        model_path: str,
        n_prompts: list[int] | None = None,
        n_gens: list[int] | None = None,
        pg_pairs: list[list[int]] | None = None,
        n_gpu_layers: int = 999,
        batch_size: int = 2048,
        ubatch_size: int = 512,
        threads: int | None = None,
        flash_attn: bool = True,
        cache_type_k: str = "f16",
        cache_type_v: str = "f16",
        n_depth: int = 0,
        repetitions: int = 3,
        extra_flags: list[str] | None = None,
    ) -> dict:
        """Run llama-bench against `model_path`. Returns immediately with a
        BenchJob; poll bench_active / stream stdout via /api/bench/stream."""
        return await _post("/api/bench/run", json={
            "model_path": model_path,
            "n_prompts": n_prompts or [512],
            "n_gens": n_gens or [128],
            "pg_pairs": [tuple(p) for p in (pg_pairs or [])],
            "n_gpu_layers": n_gpu_layers,
            "batch_size": batch_size,
            "ubatch_size": ubatch_size,
            "threads": threads,
            "flash_attn": flash_attn,
            "cache_type_k": cache_type_k,
            "cache_type_v": cache_type_v,
            "n_depth": n_depth,
            "repetitions": repetitions,
            "extra_flags": extra_flags or [],
        })

    @mcp.tool()
    async def bench_active() -> dict:
        """Return the currently running (or most recently finished) bench job."""
        return await _get("/api/bench/active")

    @mcp.tool()
    async def bench_history(limit: int = 50, model_path: str | None = None) -> list[dict]:
        """Return benchmark history. Filter by `model_path` to get runs for one model."""
        params: dict = {"limit": limit}
        if model_path:
            params["model_path"] = model_path
        return await _get("/api/bench/history", params=params)

    # --- Router mode (Faz 4: restart-free model swap on a single port) ---

    @mcp.tool()
    async def router_active() -> dict:
        """Identifies which router preset is currently running, if any."""
        return await _get("/api/router/active")

    @mcp.tool()
    async def router_models() -> dict:
        """List models known to the running router (cache + --models-dir + INI)."""
        return await _get("/api/router/models")

    @mcp.tool()
    async def router_load(model: str, autoload: bool | None = None) -> dict:
        """Load a model into the running router by id (e.g. the GGUF basename
        without extension). Uses POST /models/load — no llama-server restart."""
        body: dict = {"model": model}
        if autoload is not None:
            body["autoload"] = autoload
        return await _post("/api/router/load", json=body)

    @mcp.tool()
    async def router_unload(model: str) -> dict:
        """Unload a model from the running router (frees VRAM)."""
        return await _post("/api/router/unload", json={"model": model})

    @mcp.tool()
    async def router_ini_write(models_dir: str | None = None) -> dict:
        """(Re)generate router-models.ini from sibling single-mode presets.
        Restart the router for changes to take effect."""
        body: dict = {}
        if models_dir:
            body["models_dir"] = models_dir
        return await _post("/api/router/ini/write", json=body)

    return mcp


def build_http_app(mcp: Any) -> Any:
    """Starlette app for the streamable-http transport, served at the mount root.

    mcp 2.x takes the path here; 1.x took it in the constructor (build_mcp).
    """
    if MCP_MAJOR >= 2:
        return mcp.streamable_http_app(streamable_http_path="/")
    return mcp.streamable_http_app()


def main() -> None:
    """Entry point for `llamadeck-mcp` (stdio transport). Claude Desktop / Code
    spawns this via:
        "llamadeck": {"command": "uvx", "args": ["--from", "llamadeck", "llamadeck-mcp"]}
    """
    logging.basicConfig(level=logging.INFO)
    mcp = build_mcp()
    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
