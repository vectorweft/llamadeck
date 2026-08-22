from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically: write a temp file in the same
    directory, fsync it, then os.replace() into place. A crash or power loss
    mid-write leaves the original file intact instead of a truncated/corrupt
    one — readers never observe a half-written state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def xdg_config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


def xdg_data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")


def _state_dir() -> Path:
    """Where settings, presets, the model DB and per-preset logs live.

    `LLAMADECK_STATE_DIR` overrides it outright, which is the only way to run a
    second LlamaDeck against its own state without touching the real one.
    """
    override = os.environ.get("LLAMADECK_STATE_DIR")
    if override:
        return Path(override).expanduser()
    return xdg_config_home() / "llamadeck"


STATE_DIR = _state_dir()
DB_PATH = STATE_DIR / "llamadeck.db"
SETTINGS_PATH = STATE_DIR / "settings.json"
PRESETS_PATH = STATE_DIR / "presets.json"
LOGS_DIR = STATE_DIR / "logs"

# Pre-rename home. The project shipped as "LSC" and kept everything in
# ~/.config/lsc; migrate_legacy_state() moves that across on first run so an
# existing install does not silently come up as a fresh one.
LEGACY_STATE_DIR = xdg_config_home() / "lsc"
LEGACY_DB_NAME = "lsc.db"

_migrated = False


def migrate_legacy_state() -> None:
    """Move ~/.config/lsc to ~/.config/llamadeck, once, if that is what exists.

    Three things have to survive the move, or the rename costs the user their
    setup:

      * the directory itself — renamed, never copied, so nothing is duplicated
        and a crash mid-way cannot leave two half-states;
      * absolute paths already written INTO that state (a preset's
        `models_preset_path`, a router INI a llama-server is running against
        right now). A symlink left behind at the old location keeps every one
        of them resolving, including in processes that are already running;
      * the SQLite file, renamed with its -wal/-shm siblings so the database
        is not resurrected from a stale write-ahead log.

    Every step is best-effort: a box where the migration cannot happen must
    still start, just with empty state, rather than refuse to boot.
    """
    global _migrated
    if _migrated:
        return
    _migrated = True
    import logging
    log = logging.getLogger(__name__)
    if STATE_DIR != _state_dir():
        # A test (or anything else) repointed STATE_DIR. Migrating the real
        # ~/.config/lsc into a temporary directory would destroy the user's
        # install; only ever migrate into the location this module computes.
        return
    if STATE_DIR.exists() or not LEGACY_STATE_DIR.is_dir():
        return
    if LEGACY_STATE_DIR.is_symlink():
        return  # already pointing at the new home (a previous migration)
    try:
        STATE_DIR.parent.mkdir(parents=True, exist_ok=True)
        os.rename(LEGACY_STATE_DIR, STATE_DIR)
    except OSError as e:
        log.warning("could not migrate %s to %s: %s", LEGACY_STATE_DIR, STATE_DIR, e)
        return
    log.info("migrated state directory %s -> %s", LEGACY_STATE_DIR, STATE_DIR)
    try:
        os.symlink(STATE_DIR, LEGACY_STATE_DIR)
    except OSError as e:
        # Not fatal, but say so: a preset that stored an absolute path under
        # the old directory will fail to find its file and the reason would
        # otherwise be invisible.
        log.warning(
            "state migrated, but the compatibility symlink %s could not be created (%s) — "
            "presets holding absolute paths under the old directory need updating",
            LEGACY_STATE_DIR, e,
        )
    # Paths that were saved INTO the state (a router preset's
    # `models_preset_path`, for one) still spell the old directory. The symlink
    # keeps them working, but leaving them there means the old name reappears
    # in the UI forever — rewrite the ones that point at our own state dir.
    for f in (STATE_DIR / "settings.json", STATE_DIR / "presets.json"):
        try:
            text = f.read_text()
        except OSError:
            continue
        rewritten = text.replace(str(LEGACY_STATE_DIR) + "/", str(STATE_DIR) + "/")
        if rewritten != text:
            try:
                atomic_write_text(f, rewritten)
                log.info("rewrote legacy state paths inside %s", f.name)
            except OSError as e:
                log.warning("could not rewrite legacy paths in %s: %s", f, e)
    legacy_db = STATE_DIR / LEGACY_DB_NAME
    if legacy_db.exists() and not DB_PATH.exists():
        for suffix in ("", "-wal", "-shm"):
            src = STATE_DIR / (LEGACY_DB_NAME + suffix)
            if src.exists():
                try:
                    os.rename(src, STATE_DIR / (DB_PATH.name + suffix))
                except OSError as e:
                    log.warning("could not rename %s: %s", src, e)
        log.info("migrated model database %s -> %s", legacy_db.name, DB_PATH.name)


def _default_llama_bin() -> str:
    """First-run default for llama_bin. The source-build convention
    (~/llama.cpp/build/bin/llama-server) wins when present; otherwise a
    llama-server already on PATH (brew install llama.cpp, distro packages)
    is used. Only affects fresh installs — an existing settings.json always
    carries its own value. Falls back to the conventional path as a visible
    hint of where LlamaDeck expects a source build."""
    conventional = Path.home() / "llama.cpp" / "build" / "bin" / "llama-server"
    if conventional.exists():
        return str(conventional)
    import shutil
    on_path = shutil.which("llama-server")
    if on_path:
        return on_path
    return str(conventional)


def _default_rpc_bin() -> str:
    """Best guess at a `ggml-rpc-server` path, or "" when there is none.

    An RPC server is normally built from the same checkout as llama_bin but
    with a different backend — you cannot have CUDA and HIP in one binary
    (ggml-hip compiles the ggml-cuda sources with hipcc), which is the whole
    reason RPC exists. So look next to llama_bin first, then across sibling
    build trees of the same repo, then PATH.
    """
    import shutil

    try:
        from_settings = Path(SETTINGS_PATH).read_text()
    except OSError:
        from_settings = ""
    candidates: list[Path] = []
    llama_bin = ""
    if from_settings:
        import json
        try:
            llama_bin = json.loads(from_settings).get("llama_bin") or ""
        except ValueError:
            llama_bin = ""
    if llama_bin:
        bin_dir = Path(llama_bin).parent
        candidates.append(bin_dir / "ggml-rpc-server")
        repo = bin_dir.parent.parent          # …/<repo>/<build>/bin → <repo>
        if repo.is_dir():
            try:
                for build in sorted(repo.iterdir()):
                    if build.is_dir():
                        candidates.append(build / "bin" / "ggml-rpc-server")
            except OSError:
                pass
    for c in candidates:
        if c.exists():
            return str(c)
    return shutil.which("ggml-rpc-server") or ""


class RpcServerConfig(BaseModel):
    """One `ggml-rpc-server` LlamaDeck starts and stops.

    Exists so a backend the main binary cannot host — ROCm alongside CUDA, or a
    GPU in another machine — can still be an offload target. The devices it
    exports show up as `RPC0[host:port]` in the device list once it is running.
    """
    name: str = "rpc0"
    # Empty = resolve with _default_rpc_bin() at start time, so a rebuild that
    # creates the binary later does not need the setting to be re-saved.
    binary: str = ""
    host: str = "127.0.0.1"
    port: int = 50052
    # Device ids *in the RPC server's own binary* (e.g. "ROCm0"). Empty exports
    # everything that binary can see — including its CPU, which is rarely what
    # anyone wants, so the UI pre-fills this.
    devices: list[str] = Field(default_factory=list)
    # Start together with LlamaDeck rather than on demand.
    autostart: bool = False


class GatewaySettings(BaseModel):
    enabled: bool = False


class Settings(BaseModel):
    controller_bind_host: str = "127.0.0.1"
    controller_bind_port: int = 8770
    mcp_bind_host: str = "127.0.0.1"
    mcp_bind_port: int = 8765
    lan_token: str | None = None

    llama_repo: str = Field(default_factory=lambda: str(Path.home() / "llama.cpp"))
    llama_bin: str = Field(default_factory=lambda: _default_llama_bin())
    llama_server_url: str = "http://localhost:8080"

    # RPC offload servers. Empty by default: this is opt-in extra machinery,
    # and a box whose single binary already covers its GPUs needs none of it.
    rpc_servers: list[RpcServerConfig] = Field(default_factory=list)

    scan_roots: list[str] = Field(
        default_factory=lambda: [str(Path.home() / "llama.cpp" / "models")]
    )
    hf_models_root: str = Field(default_factory=lambda: str(Path.home() / "llama.cpp" / "models"))
    hf_token: str | None = None
    # UI language ("en" | "tr"). Mirrors the header toggle (localStorage is the
    # frontend's source of truth); the backend uses it for generated content —
    # fit-check messages, What's New cards and the guide.
    ui_language: str = "en"
    # What's New (release feature tracker) — Claude API summary cards
    anthropic_api_key: str | None = None
    # Which LLM generates What's New cards and the build guide:
    #   "claude" — Anthropic chain (settings key → env → Claude Code CLI → profile)
    #   "openai" — any OpenAI-compatible chat-completions endpoint: OpenRouter,
    #              OpenAI, Groq, DeepSeek, a local llama-server (/v1), Ollama…
    llm_provider: str = "claude"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str | None = None
    llm_model: str = ""

    gateway: GatewaySettings = Field(default_factory=GatewaySettings)

    # ----- GPU broker (faz-1.5) ---------------------------------------------
    # Default LLM preset the broker starts when slot=llm is acquired and no
    # other preset has been pinned. If null, broker picks the first
    # single-mode preset in the registry.
    default_llm_preset: str | None = None
    # Time the broker keeps an idle LLM slot warm before stopping it to free
    # VRAM for other slots. 0 = stop immediately on release. Default 5 min.
    llm_keepalive_s: float = 300.0
    # Default lease TTL when caller does not specify a duration. The broker
    # inflates this to max(default_ttl_s, est_duration_s * 1.2).
    gpu_default_ttl_s: float = 600.0
    # Hard cap on a single lease's vram budget; rejects 503 if exceeded.
    # Auto-detected at startup (NVIDIA/AMD/Apple probe) if null.
    gpu_total_vram_mb: int | None = None
    # VRAM reserved for non-LlamaDeck-managed GPU residents (e.g. an embedding
    # model or desktop compositor living in its own process — LlamaDeck's slot
    # accounting does NOT see it; we subtract it from the available budget
    # before spawning a new llama-server child). Set this if other GPU
    # workloads run alongside LlamaDeck.
    external_reserved_vram_mb: int = 0
    # Safety margin added to a service's est_vram_mb when comparing
    # against actually-free VRAM at spawn time. Catches estimate drift
    # and CUDA allocator slack so we don't OOM mid-load.
    vram_spawn_headroom_mb: int = 2048

    # ----- ComfyUI -----------------------------------------------------------
    # If `comfy_path` is set, LlamaDeck will spawn `python main.py --listen ... --port ...`
    # in that directory the first time the comfy slot is acquired. If it's
    # null, LlamaDeck only ADOPTS an already-running ComfyUI at `comfy_managed_url`
    # (started manually / via systemd). Either path is fine.
    comfy_managed_url: str = "http://127.0.0.1:8188"
    comfy_est_vram_mb: int = 11000
    comfy_health_path: str = "/system_stats"
    comfy_path: str | None = None         # absolute path to ComfyUI checkout
    comfy_python_bin: str | None = None   # null → use "python" on PATH
    comfy_extra_args: str = ""            # e.g. "--lowvram --preview-method auto"
    comfy_startup_timeout_s: float = 120.0

    # ----- TTS (IndexTTS-2 / XTTS) ------------------------------------------
    # If `xtts_path` is set, LlamaDeck spawns `python <xtts_server_script> --host ... --port ...`
    # in that directory the first time the tts slot is acquired. If it's
    # null, LlamaDeck only ADOPTS an already-running TTS server at xtts_managed_url
    # (started manually / via systemd). Either path is fine.
    xtts_managed_url: str = "http://127.0.0.1:8020"
    xtts_est_vram_mb: int = 4000
    xtts_health_path: str = "/health"
    xtts_path: str | None = None              # absolute path to TTS checkout
    xtts_python_bin: str | None = None        # null → use "python" on PATH
    xtts_server_script: str = "tts_http_server.py"  # script name inside xtts_path
    xtts_extra_args: str = ""
    xtts_startup_timeout_s: float = 120.0


def ensure_state_dirs() -> None:
    migrate_legacy_state()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """Load settings from JSON. On corrupt/invalid file, fall back to defaults
    after preserving the broken file so the user can recover it manually.
    Otherwise a hand-edit typo would crash LlamaDeck at boot with a stack trace."""
    import logging
    log = logging.getLogger(__name__)
    ensure_state_dirs()
    if SETTINGS_PATH.exists():
        try:
            raw = SETTINGS_PATH.read_text()
            data: dict[str, Any] = json.loads(raw)
            return Settings.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            backup = SETTINGS_PATH.with_suffix(".json.broken")
            try:
                SETTINGS_PATH.rename(backup)
                log.error("settings.json is corrupt (%s); moved to %s, regenerating defaults", e, backup)
            except OSError:
                log.error("settings.json is corrupt (%s) and could not be moved aside", e)
        except OSError as e:
            log.error("could not read settings.json (%s); using defaults", e)
    s = Settings()
    save_settings(s)
    return s


def save_settings(s: Settings) -> None:
    ensure_state_dirs()
    atomic_write_text(SETTINGS_PATH, json.dumps(s.model_dump(), indent=2) + "\n")


@dataclass
class LlamaServerConfig:
    """Full configuration for one llama-server invocation. Serialized into presets.json
    and rendered into argv for ProcessSupervisor.start() in Faz 1."""

    name: str
    model_path: str | None = None
    hf_repo: str | None = None
    hf_file: str | None = None
    mmproj_path: str | None = None

    # Localhost by default: a llama-server has no authentication, so a fresh
    # install must not put an inference endpoint on the network without the
    # user asking for it. Set 0.0.0.0 (or a specific interface) per preset to
    # expose it on the LAN.
    host: str = "127.0.0.1"
    port: int = 8080
    api_key: str | None = None

    ctx_size: int = 8192
    n_gpu_layers: int = 999

    # Which GPUs this preset loads onto, as llama.cpp device ids ("CUDA0",
    # "Vulkan1") — emitted as `-dev`. Empty means "let llama.cpp choose", which
    # is the historical behaviour and stays the default.
    #
    # The ids are backend-specific and come from `llama-server --list-devices`
    # (see devices.py), NOT from LlamaDeck's own GPU probe: a CUDA+Vulkan build
    # lists one RTX 5090 twice (CUDA0 and Vulkan2) and only that binary knows
    # the names. Selecting a device the current binary does not expose is
    # rejected at start rather than silently ignored.
    devices: list[str] = field(default_factory=list)
    # `-ts`: proportions for spreading layers over the selected devices. The
    # useful non-uniform case is "1,0" — keep every non-overridden layer on the
    # first device while a second one still holds tensors placed by `-ot`.
    tensor_split: str | None = None
    parallel: int = 1
    batch_size: int = 2048
    ubatch_size: int = 512
    threads: int = -1

    flash_attn: str = "auto"  # on | off | auto
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    cont_batching: bool = True

    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    repeat_penalty: float = 1.0
    # presence_penalty: opt-in only (None → emit nothing, defer to the model's
    # GGUF / family default — e.g. Qwen3.6 family default is 1.5, see
    # model_defaults.py). Set it explicitly on a preset when that preset's job
    # needs a non-default value — e.g. 0.0 for a vision/OCR preset, where the
    # task is faithful TRANSCRIPTION and pp=1.5 (prose diversity) makes the
    # model drift / run away on hard pages instead of emitting EOS. Unlike the
    # fields above it defaults to None so existing presets are unchanged.
    presence_penalty: float | None = None

    # Thinking / reasoning, emitted as `--reasoning on|off`:
    #   "auto" — let the chat template decide (llama-server's own default, and
    #            what every preset written before this field did). Emits nothing.
    #   "on"   — force thinking on for a model whose template supports it.
    #   "off"  — force it off: the instruct path of a dual-mode model
    #            (Qwen3.x, DeepSeek-R1, …). Note that the RIGHT SAMPLING
    #            DIFFERS between the two modes — see model_defaults.py, where
    #            Qwen3.6 wants temp 1.0/top_p 0.95 thinking and
    #            temp 0.7/top_p 0.80/presence 1.5 instruct.
    # There is deliberately no reasoning-EFFORT field beside this one. How hard
    # a model thinks is a per-request preference and belongs to the caller, which
    # sends `reasoning_effort` itself; pinning a level here only sets a floor the
    # caller cannot see. Whether a dual-mode model thinks AT ALL is different —
    # that is a deployment decision, and for callers that cannot send
    # `chat_template_kwargs` it is the only lever, so it stays.
    reasoning: str = "auto"

    jinja: bool = False
    metrics: bool = True
    slots: bool = True

    # Speculative decoding (single-mode only).
    #
    # `spec_type` picks the strategy and is emitted as `--spec-type <value>`.
    #   "none"        — disabled (no --spec-type emitted).
    #   "draft-mtp"   — self-speculate via the main model's MTP (Multi-Token
    #                   Prediction) head. No separate drafter GGUF needed; the
    #                   head must be baked into the main model file. Only
    #                   Qwen3.5 / Qwen3.6 architectures actually execute MTP
    #                   today (others load the tensors but ignore them at
    #                   inference). `draft_max` / `draft_min` still apply.
    #   "draft-model" — classic vocab-matched speculative decoding with a
    #                   separate small GGUF in `model_path_draft`. Drafter must
    #                   share the target's tokenizer (same family — e.g.
    #                   Gemma-4-E2B for Gemma-4-31B).
    #
    # Backward compat: presets persisted before this field existed will load
    # with spec_type="none"; if `model_path_draft` is set on such a preset, the
    # UI shows it as "draft-model" automatically.
    spec_type: str = "none"
    model_path_draft: str | None = None
    n_gpu_layers_draft: int = 999
    draft_max: int | None = None
    draft_min: int | None = None

    # Environment variables for the llama-server process, e.g.
    # {"GGML_CUDA_DISABLE_GRAPHS": "1"}. Some llama.cpp behaviour has no flag
    # at all and can only be reached this way — disabling CUDA graph capture
    # being the one that matters on this box, where the MTP + quantized-KV
    # combination hangs a kernel and the driver kills the channel (Xid 8).
    # Merged over the backend's own environment at spawn; per preset, so a
    # workaround for one card does not follow every other preset around.
    env: dict[str, str] = field(default_factory=dict)

    extra_flags: list[str] = field(default_factory=list)
    # The pro escape hatch: a full command line that REPLACES everything
    # rendered from the fields above. When set, `to_argv` returns exactly these
    # tokens (with the binary swapped for the configured one) — no field, no
    # extra flag, no router shaping is applied on top. Empty/None = normal
    # field-driven rendering, which is what every preset does by default.
    #
    # Host/port/model_path are read back out of the command on save, because
    # the rest of LlamaDeck (health checks, port-conflict detection, the VRAM
    # estimate) still has to know where the process listens and what it loads.
    argv_override: str | None = None
    notes: str = ""
    estimated_vram_mb: int | None = None
    # Cosmetic only: when true the preset moves into the collapsed "hidden"
    # section in lists. No effect on the router INI or the supervisor —
    # external projects' alias presets keep working even when hidden.
    ui_hidden: bool = False

    # Router mode (Faz 4): when "router", llama-server runs as a multi-model
    # router. model_path / mmproj_path / sampling are ignored; loaded models are
    # picked at runtime via POST /models/load. ctx/ngl/parallel/batch become
    # global defaults inherited by every loaded model. The INI at
    # ~/.config/llamadeck/router-models.ini (auto-generated from sibling presets)
    # supplies per-model overrides.
    mode: str = "single"  # "single" | "router"
    models_dir: str | None = None
    models_max: int = 1
    models_autoload: bool = True
    models_preset_path: str | None = None
    sleep_idle_seconds: int | None = None
