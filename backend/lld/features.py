"""FeatureTracker — pipeline that tracks and teaches llama.cpp updates.

Flow (hybrid detection):
1. After every successful rebuild (build.py hook) or on manual trigger:
   the new binary's `--help` output is captured into `help_snapshots`.
2. A line-based flag diff against the previous snapshot (deterministic,
   no LLM — proof the flag really exists in this build).
3. Commit subjects / release notes are pulled from the GitHub compare +
   releases APIs as context (model-architecture news comes from here).
4. Raw data is written to a `feature_scans` row; if an LLM backend is
   configured (Claude credentials, or any OpenAI-compatible endpoint chosen
   in Settings — OpenRouter / local llama-server / …) it is summarized into
   "what landed / how to use / why" cards (`release_features`). Without
   credentials (or on a failed call) the scan stays pending/failed and the
   retry endpoint re-runs the summary from the stored raw data.
   Against a local endpoint the prompt is fitted to the server's actual
   context window and stepped down through _DETAIL_LEVELS on overflow — see
   the "endpoint robustness" section below.

Additionally: one-click "try in a preset" from a card (clone + add flags +
start) and background A/B comparison via two sequential llama-bench runs
(`feature_ab_runs`).

Note: the *_tr column/field names (title_tr, what_tr, …) are historical —
their content is English.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import replace
from pathlib import Path

import httpx

from .build import get_build_manager
from .db import connect
from .procutil import run_capture
from .settings import load_settings

log = logging.getLogger(__name__)

GITHUB_REPO = "ggml-org/llama.cpp"
CLAUDE_MODEL = "claude-opus-4-8"

# Commit subjects that never carry user-facing model/server features.
_NOISE_PREFIXES = (
    "ci", "docs", "doc", "nix", "readme", "sync", "chore", "devops",
    "editorconfig", "github", "gitignore", "license", "scripts", "tests", "test",
)

_FLAG_RE = re.compile(r"--[a-zA-Z0-9][A-Za-z0-9-]*")
_RELEASE_TAG_RE = re.compile(r"^b(\d+)$")

FEATURES_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["features"],
    "properties": {
        "features": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "title_tr", "what_tr", "how_tr", "why_tr",
                    "flags", "architectures", "source_urls", "confidence",
                ],
                "properties": {
                    "title_tr": {"type": "string"},
                    "what_tr": {"type": "string"},
                    "how_tr": {"type": "string"},
                    "why_tr": {"type": "string"},
                    "flags": {"type": "array", "items": {"type": "string"}},
                    "architectures": {"type": "array", "items": {"type": "string"}},
                    "source_urls": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
            },
        },
    },
}


class FeatureError(RuntimeError):
    pass


# ---------- OpenAI-compatible endpoint robustness ----------
#
# Local endpoints (llama-server & friends) fail in ways hosted APIs do not:
# the served model id is a full .gguf path rather than the name the user typed,
# the context window is whatever the preset was started with, and small models
# happily stop mid-JSON. Everything below exists to make those three survivable
# instead of turning into a failed scan.

_CHARS_PER_TOKEN = 3.0      # conservative fallback when /tokenize is absent
_MIN_COMPLETION = 1200      # a summary below this is not worth asking for
_MAX_COMPLETION = 8000
_CTX_MARGIN = 256           # chat-template wrapping the token count misses

# Prompt sizes, largest first. Each retry after a context overflow steps down
# one level; `release_chars` 0 drops release notes entirely.
_DETAIL_LEVELS = (
    {"commits": 200, "release_chars": 6000, "flag_names": None, "new_flags": 40, "max_cards": 8},
    {"commits": 120, "release_chars": 3000, "flag_names": 400, "new_flags": 30, "max_cards": 6},
    {"commits": 60, "release_chars": 1200, "flag_names": 200, "new_flags": 20, "max_cards": 5},
    {"commits": 25, "release_chars": 0, "flag_names": 80, "new_flags": 12, "max_cards": 4},
)


class ContextTooSmall(FeatureError):
    """The prompt does not fit the endpoint's context window."""

    def __init__(self, message: str, *, n_ctx: int | None = None,
                 n_prompt: int | None = None) -> None:
        super().__init__(message)
        self.n_ctx = n_ctx
        self.n_prompt = n_prompt


class TruncatedOutput(FeatureError):
    """The endpoint stopped mid-answer (token cap or context exhausted)."""


def _api_root(base: str) -> str:
    """llama.cpp's native endpoints (/props, /tokenize) sit at the server root,
    while the OpenAI-compatible ones may be under /v1."""
    base = base.rstrip("/")
    return base[:-3].rstrip("/") if base.endswith("/v1") else base


def _model_key(model_id: str) -> str:
    """Comparable form of a model id: llama-server reports the full
    `/ml/…/Foo-Q6_K.gguf` path where the user typed `Foo-Q6_K`."""
    name = model_id.replace("\\", "/").rsplit("/", 1)[-1]
    if name.lower().endswith(".gguf"):
        name = name[:-5]
    return name.strip().lower()


def resolve_model_id(configured: str, served: list[str]) -> str:
    """Pick the id to send as `model`.

    An empty `configured` means "auto": the local case, where the endpoint
    already knows which model is loaded and asking the user to retype its name
    only creates a way to get it wrong. A stale name is tolerated the same way
    when the endpoint serves exactly one model."""
    configured = (configured or "").strip()
    if not configured:
        if len(served) == 1:
            return served[0]
        if len(served) > 1:
            raise FeatureError(
                f"this endpoint serves {len(served)} models — pick one in "
                "Settings → AI provider"
            )
        raise FeatureError(
            "could not read the model list from this endpoint — check the base "
            "URL, or name the model in Settings → AI provider"
        )
    if not served or configured in served:
        return configured
    want = _model_key(configured)
    exact = [m for m in served if _model_key(m) == want]
    if len(exact) == 1:
        return exact[0]
    partial = [m for m in served if want and want in _model_key(m)]
    if len(partial) == 1:
        return partial[0]
    if len(served) == 1:
        log.warning(
            "features: endpoint serves %r, not the configured %r — using the served model",
            served[0], configured,
        )
        return served[0]
    if exact or partial:
        raise FeatureError(
            f"model '{configured}' is ambiguous at this endpoint: "
            + ", ".join((exact or partial)[:5])
        )
    raise FeatureError(
        f"model '{configured}' is not served by this endpoint — available: "
        + ", ".join(served[:5]) + ("…" if len(served) > 5 else "")
    )


_probe_cache: dict[str, tuple[float, dict]] = {}
_PROBE_TTL = 30.0


async def probe_endpoint(base: str, headers: dict | None = None,
                         timeout: float = 8.0) -> dict:
    """Best-effort discovery of an OpenAI-compatible endpoint:
    `{"models": [id…], "n_ctx": int|None, "native": bool}`.

    Never raises — an endpoint that answers none of these is simply used blind,
    exactly as before this probe existed."""
    base = base.rstrip("/")
    hit = _probe_cache.get(base)
    if hit and time.monotonic() - hit[0] < _PROBE_TTL:
        return hit[1]
    info: dict = {"models": [], "n_ctx": None, "native": False, "reachable": False}
    root = _api_root(base)
    async with httpx.AsyncClient(timeout=timeout, headers=headers or {}) as client:
        for url in (f"{base}/models", f"{root}/v1/models"):
            try:
                r = await client.get(url)
                if r.status_code != 200:
                    continue
                data = r.json()
            except Exception:
                continue
            ids = [
                str(m["id"]) for m in (data.get("data") or [])
                if isinstance(m, dict) and m.get("id")
            ] or [
                str(m["name"]) for m in (data.get("models") or [])
                if isinstance(m, dict) and m.get("name")
            ]
            info["reachable"] = True
            if ids:
                info["models"] = ids
                break
        try:
            r = await client.get(f"{root}/props")
            if r.status_code == 200:
                props = r.json()
                n_ctx = props.get("n_ctx") or (
                    props.get("default_generation_settings") or {}).get("n_ctx")
                info["native"] = True
                info["reachable"] = True
                if isinstance(n_ctx, int) and n_ctx > 0:
                    info["n_ctx"] = n_ctx
        except Exception:
            pass
    _probe_cache[base] = (time.monotonic(), info)
    return info


async def count_tokens(text: str, base: str, headers: dict | None = None,
                       native: bool = False) -> int:
    """Exact count via llama.cpp's /tokenize when available, else a
    deliberately pessimistic character estimate."""
    if native:
        try:
            async with httpx.AsyncClient(timeout=30.0, headers=headers or {}) as client:
                r = await client.post(f"{_api_root(base)}/tokenize", json={"content": text})
            if r.status_code == 200:
                toks = r.json().get("tokens")
                if isinstance(toks, list) and toks:
                    return len(toks)
        except Exception:
            pass
    return int(len(text) / _CHARS_PER_TOKEN) + 1


def strip_reasoning(text: str) -> str:
    """Remove <think> blocks, including the half-open ones a truncated or
    template-mangled response leaves behind."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    if "</think>" in text:               # opener eaten by the chat template
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:                # never closed: the answer never came
        text = text.split("<think>")[0]
    return text.strip()


def _strip_fences(text: str) -> str:
    """Drop ``` fence lines wherever they appear — models put prose before the
    fence often enough that an anchored regex misses them."""
    if "```" not in text:
        return text.strip()
    return "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("```")
    ).strip()


def extract_json_object(text: str) -> str:
    """The first brace-balanced object in `text`, string- and escape-aware.
    Returns the tail from the opening brace when it is never closed, so the
    salvage path still has something to chew on."""
    start = text.find("{")
    if start < 0:
        return text.strip()
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
        elif ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return text[start:]


def _salvage_feature_objects(text: str) -> list[dict]:
    """Pull the complete card objects out of a truncated `features` array.
    Six good cards plus a half-written seventh is a usable answer; throwing all
    seven away because of the seventh is not."""
    anchor = text.find('"features"')
    start = text.find("[", anchor) if anchor >= 0 else text.find("[")
    if start < 0:
        return []
    decoder = json.JSONDecoder()
    pos = start + 1
    out: list[dict] = []
    while pos < len(text):
        while pos < len(text) and text[pos] in " \t\r\n,":
            pos += 1
        if pos >= len(text) or text[pos] != "{":
            break
        try:
            obj, pos = decoder.raw_decode(text, pos)
        except ValueError:
            break
        if isinstance(obj, dict):
            out.append(obj)
    return out


def parse_feature_cards(text: str) -> tuple[list, bool]:
    """→ (raw cards, salvaged). Raises FeatureError only when nothing at all
    could be read out of the response."""
    cleaned = _strip_fences(strip_reasoning(text))
    blob = extract_json_object(cleaned)
    try:
        data = json.loads(blob)
        if isinstance(data, dict) and isinstance(data.get("features"), list):
            return data["features"], False
        if isinstance(data, list):
            return data, False
    except ValueError:
        pass
    salvaged = _salvage_feature_objects(cleaned)
    if salvaged:
        return salvaged, True
    raise FeatureError("no feature cards could be read from the LLM response")


def extract_flags(help_text: str) -> dict[str, str]:
    """Map of `--flag` → usage line.

    llama-server --help format: flag lines start with an unindented `-`
    (`-h,    --help, --usage        print usage...`), continuation/description
    lines are indented. Since description text may reference other flags
    ("Complements --cpu-mask"), each line is split on 2+ spaces and only the
    leading chunks that start with `-` (the flag column) are scanned."""
    flags: dict[str, str] = {}
    for line in help_text.splitlines():
        if not line.startswith("-"):
            continue
        stripped = line.strip()
        flag_col: list[str] = []
        for chunk in re.split(r"\s{2,}", stripped):
            if not chunk.startswith("-"):
                break
            flag_col.append(chunk)
        for flag in _FLAG_RE.findall(" ".join(flag_col)):
            flags.setdefault(flag, stripped)
    return flags


def diff_help(old_text: str, new_text: str) -> tuple[list[dict], list[str]]:
    """(new flags [{flag, usage}], removed flag names)."""
    old_flags = extract_flags(old_text)
    new_flags = extract_flags(new_text)
    added = [
        {"flag": f, "usage": usage}
        for f, usage in new_flags.items()
        if f not in old_flags
    ]
    removed = [f for f in old_flags if f not in new_flags]
    return added, removed


def _is_noise(subject: str) -> bool:
    head = subject.split(":", 1)[0].strip().lower()
    return any(head == p or head.startswith(p + " ") for p in _NOISE_PREFIXES)


def find_claude_cli() -> str | None:
    """Find the Claude Code binary. Pro/Max subscriptions work through this
    CLI (no API credits needed). The ~/.local/bin/claude symlink can break on
    snap updates, so version directories are scanned too."""
    import shutil

    p = shutil.which("claude")
    if p and Path(p).resolve().exists():
        return p
    link = Path.home() / ".local" / "bin" / "claude"
    if link.exists() and link.resolve().exists():
        return str(link)
    candidates: list[tuple[tuple[int, ...], Path]] = []
    for pattern in (
        Path.home() / "snap" / "code" / "current" / ".local" / "share" / "claude" / "versions",
        Path.home() / ".local" / "share" / "claude" / "versions",
    ):
        if not pattern.is_dir():
            continue
        for d in pattern.iterdir():
            m = re.match(r"^(\d+)\.(\d+)\.(\d+)$", d.name)
            if m and d.is_file() and os.access(d, os.X_OK):
                candidates.append((tuple(int(x) for x in m.groups()), d))
    if candidates:
        return str(max(candidates)[1])
    return None


def anthropic_auth_mode() -> str:
    """Which credential source can be used for summarization? Priority order:

    api_key    → key from settings (API, pay-as-you-go)
    env        → ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN environment variable
    claude_cli → Claude Code session on this machine (Pro/Max SUBSCRIPTION —
                 no credits needed; summary produced via `claude -p`)
    profile    → `ant auth login` OAuth profile (REQUIRES API credits)
    none       → none of the above
    """
    if load_settings().anthropic_api_key:
        return "api_key"
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "env"
    if find_claude_cli() and (Path.home() / ".claude" / ".credentials.json").exists():
        return "claude_cli"
    cfg = Path(
        os.environ.get("ANTHROPIC_CONFIG_DIR")
        or Path.home() / ".config" / "anthropic"
    )
    creds = cfg / "credentials"
    if creds.is_dir() and any(creds.glob("*.json")):
        return "profile"
    return "none"


_NO_AUTH_MSG = (
    "No credentials found — Claude Code must be installed and signed in (Pro subscription), "
    "enter an Anthropic API key on the What's New page, or pick another provider "
    "(OpenRouter / local model / any OpenAI-compatible API) in Settings"
)

_OPENAI_UNCONFIGURED_MSG = (
    "OpenAI-compatible provider selected but not configured — "
    "set the base URL in Settings → AI provider"
)


def summary_status() -> dict:
    """Which LLM backend will produce summaries, and is it ready?

    provider "claude" → mode is the anthropic_auth_mode() value
    provider "openai" → mode "openai" when base URL + model are set, else "none"
    """
    s = load_settings()
    if (s.llm_provider or "claude") == "openai":
        # No model name needed: an endpoint that serves one model is asked what
        # it has loaded at call time.
        ok = bool((s.llm_base_url or "").strip())
        return {
            "provider": "openai",
            "mode": "openai" if ok else "none",
            "model": (s.llm_model or "").strip() or None,
            "base_url": s.llm_base_url or None,
            "detail": None if ok else _OPENAI_UNCONFIGURED_MSG,
        }
    mode = anthropic_auth_mode()
    return {
        "provider": "claude",
        "mode": mode,
        "model": None,
        "base_url": None,
        "detail": None if mode != "none" else _NO_AUTH_MSG,
    }

# Summaries go through the CLI with Sonnet to protect the Pro quota.
CLAUDE_CLI_MODEL = "sonnet"


class FeatureTracker:
    """Singleton (get_feature_tracker). One scan at a time; A/B runs do not
    share this lock (bench has its own single-job lock)."""

    def __init__(self) -> None:
        self._scan_lock = asyncio.Lock()
        self._scanning = False
        self._ab_running = False
        self._guide_running = False

    # ---------- snapshot ----------

    async def capture_snapshot(self) -> dict | None:
        """Store the current binary's --help output against its commit.
        No second row is created for the same commit (INSERT OR IGNORE)."""
        bm = get_build_manager()
        v = await bm.current_version()
        if not v.commit:
            # No binary yet is the normal state of a fresh install, not a
            # problem to report: boot has already said so, in a line that names
            # the setup wizard. Repeating it as a WARNING made the first launch
            # of a clean clone end on a yellow line about a file the user had
            # not installed yet. A binary that *is* there and still will not
            # answer --version is a real fault and still warns.
            level = logging.DEBUG if not bm.llama_bin.exists() else logging.WARNING
            log.log(level, "features: could not read binary version: %s", v.raw)
            return None
        res = await run_capture([str(bm.llama_bin), "--help"], timeout=15.0)
        if res.error or res.timed_out:
            log.warning("features: could not run --help: %s",
                        res.error or "timed out after 15s")
            return None
        help_text = res.text
        if not help_text:
            return None
        async with connect() as db:
            await db.execute(
                "INSERT OR IGNORE INTO help_snapshots (commit_sha, build_number, captured_at, help_text) "
                "VALUES (?, ?, ?, ?)",
                (v.commit, v.build_number, time.time(), help_text),
            )
            await db.commit()
            cur = await db.execute(
                "SELECT id, commit_sha, build_number FROM help_snapshots WHERE commit_sha=?",
                (v.commit,),
            )
            row = await cur.fetchone()
        return dict(row) if row else None

    async def ensure_baseline(self) -> None:
        """Boot task: capture a snapshot for the current binary if missing.
        Prepares the baseline on first install; the first real diff happens
        on the next rebuild."""
        try:
            snap = await self.capture_snapshot()
            if snap:
                log.info("features: help snapshot ready (commit=%s)", snap["commit_sha"])
        except Exception:
            log.exception("features: could not capture baseline snapshot")

    # ---------- GitHub context ----------

    async def _github_context(
        self, from_commit: str, to_commit: str,
        from_build: int | None, to_build: int | None,
    ) -> tuple[list[dict], list[dict]]:
        """(commit subjects, release notes). Returns empty on network/rate
        errors — the flag diff alone still produces a valid scan."""
        commits: list[dict] = []
        releases: list[dict] = []
        headers = {"Accept": "application/vnd.github+json"}
        try:
            async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
                r = await client.get(
                    f"https://api.github.com/repos/{GITHUB_REPO}/compare/"
                    f"{from_commit}...{to_commit}",
                    params={"per_page": 250},
                )
                if r.status_code == 200:
                    for c in r.json().get("commits", []):
                        subject = (c.get("commit", {}).get("message") or "").split("\n", 1)[0]
                        if subject and not _is_noise(subject):
                            commits.append({"sha": c.get("sha", "")[:9], "subject": subject})
                else:
                    log.warning("features: GitHub compare %s: %s", r.status_code, r.text[:200])

                r = await client.get(
                    f"https://api.github.com/repos/{GITHUB_REPO}/releases",
                    params={"per_page": 15},
                )
                if r.status_code == 200:
                    for rel in r.json():
                        m = _RELEASE_TAG_RE.match(rel.get("tag_name") or "")
                        if not m:
                            continue
                        num = int(m.group(1))
                        if from_build and to_build and not (from_build < num <= to_build):
                            continue
                        body = (rel.get("body") or "").strip()
                        if body:
                            releases.append({
                                "tag": rel["tag_name"],
                                "name": rel.get("name") or rel["tag_name"],
                                "body": body[:4000],
                            })
                else:
                    log.warning("features: GitHub releases %s", r.status_code)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError) as e:
            log.warning("features: GitHub unreachable (%s) — continuing with flag diff", e)
        return commits[:200], releases

    # ---------- scan ----------

    def scanning(self) -> bool:
        return self._scanning

    async def run_scan(self, force: bool = False) -> dict | None:
        """Capture snapshot → diff against the previous distinct snapshot →
        GitHub context → scan row → (with credentials) Claude summary.
        Returns the scan dict."""
        if self._scan_lock.locked():
            raise FeatureError("a scan is already running")
        async with self._scan_lock:
            self._scanning = True
            try:
                return await self._run_scan_inner(force)
            finally:
                self._scanning = False

    async def _run_scan_inner(self, force: bool) -> dict | None:
        current = await self.capture_snapshot()
        if not current:
            raise FeatureError(
                "could not read --help/--version from the llama-server binary — "
                "check Settings → llama.cpp paths"
            )

        async with connect() as db:
            cur = await db.execute(
                "SELECT * FROM help_snapshots WHERE commit_sha != ? "
                "ORDER BY captured_at DESC LIMIT 1",
                (current["commit_sha"],),
            )
            baseline = await cur.fetchone()
            cur = await db.execute(
                "SELECT help_text FROM help_snapshots WHERE commit_sha=?",
                (current["commit_sha"],),
            )
            current_text = (await cur.fetchone())["help_text"]

        if baseline is None:
            if not force:
                log.info("features: no previous snapshot to compare — waiting for the first rebuild")
                return None
            raise FeatureError(
                "no previous snapshot to compare against; scanning becomes possible after the first rebuild"
            )

        # A repeat scan for the same commit pair (e.g. CUDA-only toggle) is skipped.
        async with connect() as db:
            cur = await db.execute(
                "SELECT id FROM feature_scans WHERE from_commit=? AND to_commit=?",
                (baseline["commit_sha"], current["commit_sha"]),
            )
            existing = await cur.fetchone()
        if existing and not force:
            log.info("features: %s→%s already scanned (scan=%s)",
                     baseline["commit_sha"], current["commit_sha"], existing["id"])
            return None

        new_flags, removed_flags = diff_help(baseline["help_text"], current_text)
        commits, releases = await self._github_context(
            baseline["commit_sha"], current["commit_sha"],
            baseline["build_number"], current["build_number"],
        )

        status = "empty" if not new_flags and not commits and not releases else "pending"
        async with connect() as db:
            cur = await db.execute(
                "INSERT INTO feature_scans (created_at, from_commit, to_commit, build_number, "
                "new_flags_json, removed_flags_json, commits_json, releases_json, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(), baseline["commit_sha"], current["commit_sha"],
                    current["build_number"],
                    json.dumps(new_flags), json.dumps(removed_flags),
                    json.dumps(commits), json.dumps(releases), status,
                ),
            )
            await db.commit()
            scan_id = cur.lastrowid

        log.info(
            "features: scan %s (%s→%s): %d new flags, %d commits, %d releases",
            scan_id, baseline["commit_sha"], current["commit_sha"],
            len(new_flags), len(commits), len(releases),
        )
        if status == "pending":
            await self.summarize_scan(scan_id)
        return await self.get_scan(scan_id)

    async def get_scan(self, scan_id: int) -> dict | None:
        async with connect() as db:
            cur = await db.execute("SELECT * FROM feature_scans WHERE id=?", (scan_id,))
            row = await cur.fetchone()
        return self._scan_to_dict(row) if row else None

    @staticmethod
    def _scan_to_dict(row) -> dict:
        d = dict(row)
        for col, key in (
            ("new_flags_json", "new_flags"), ("removed_flags_json", "removed_flags"),
            ("commits_json", "commits"), ("releases_json", "releases"),
        ):
            try:
                d[key] = json.loads(d.get(col) or "[]")
            except Exception:
                d[key] = []
            d.pop(col, None)
        return d

    # ---------- Claude summarization ----------

    async def _model_inventory(self) -> list[str]:
        """The user's local model families (fed to the prompt as context)."""
        async with connect() as db:
            cur = await db.execute(
                "SELECT DISTINCT family FROM models WHERE family IS NOT NULL ORDER BY family"
            )
            rows = await cur.fetchall()
        return [r["family"] for r in rows]

    # Key flags whose full usage goes into the card prompt — so the model
    # doesn't invent names/values when writing spec/draft/perf commands
    # (e.g. the real `--spec-type draft-mtp` instead of a nonexistent
    # `--draft-model`).
    _KEY_FLAGS = (
        "--spec-type", "--spec-draft-model", "--spec-default",
        "--spec-draft-n-max", "--flash-attn",
        "--cache-type-k", "--cache-type-v", "--cpu-moe", "--n-cpu-moe",
    )

    @staticmethod
    def _output_language() -> str:
        """Language for generated content (cards/guide), from settings."""
        return "Turkish" if load_settings().ui_language == "tr" else "English"

    def _build_prompt(self, scan: dict, families: list[str],
                      build_flags: dict[str, str], level: int = 0) -> str:
        """`level` indexes _DETAIL_LEVELS: 0 is the full prompt, higher levels
        shrink it to fit a smaller context window."""
        lim = _DETAIL_LEVELS[min(level, len(_DETAIL_LEVELS) - 1)]
        lang = self._output_language()
        flags_block = "\n".join(
            f"  {f['flag']}: {f['usage']}" for f in scan["new_flags"][:lim["new_flags"]]
        ) or "  (none)"
        removed_block = ", ".join(scan["removed_flags"][:20]) or "(none)"
        commits_block = "\n".join(
            f"  {c['sha']} {c['subject']}" for c in scan["commits"][:lim["commits"]]
        ) or "  (unavailable)"
        releases_block = "\n\n".join(
            f"### {r['tag']} — {r['name']}\n{r['body']}" for r in scan["releases"]
        ) or "(none)"
        if not lim["release_chars"]:
            releases_block = "(omitted — the endpoint's context is limited)"
        elif len(releases_block) > lim["release_chars"]:
            releases_block = releases_block[:lim["release_chars"]] + "\n…(truncated)"
        fam_block = ", ".join(families) or "(no models scanned)"
        # The flag universe that ACTUALLY exists in this build: full usage of
        # the key ones + a list of all names. The model may only pick from these.
        key_usage_block = "\n".join(
            f"  {build_flags[f]}" for f in self._KEY_FLAGS if f in build_flags
        ) or "  (not in this build)"
        names = sorted(build_flags)
        if lim["flag_names"] and len(names) > lim["flag_names"]:
            names = names[:lim["flag_names"]]
        available_block = ", ".join(names) or "(unknown)"
        return f"""You are an assistant with deep llama.cpp expertise. Below are the changes detected in the user's local llama.cpp build update {scan['from_commit']} → {scan['to_commit']}. Turn them into {lang} feature cards that TEACH the user.

## New llama-server flags (added for the FIRST time in this update):
{flags_block}

## Removed flags:
{removed_block}

## FULL usage of key flags PRESENT in this build (when writing spec/draft/perf commands pick values EXACTLY from here; e.g. only the listed enum values for --spec-type):
{key_usage_block}

## ALL flag names that exist in this build (when writing commands/examples use ONLY these names, do NOT invent flags):
{available_block}

## Commit subjects (filtered):
{commits_block}

## Release notes:
{releases_block}

## The user's local model families (derived from file names):
{fam_block}

## Task
Cluster related flags and commits into meaningful FEATURES and produce one card per feature. Rules (note: the *_tr field names are historical — write all values in {lang}):
1. `title_tr`: short, clear {lang} title (e.g. "MTP support landed for Gemma models", in {lang}).
2. `what_tr`: WHAT LANDED — what the feature is, 2-4 sentences, technical but accessible {lang}.
3. `how_tr`: HOW TO USE IT — with concrete llama-server commands. Use ONLY the REAL flags listed above under "ALL flag names" and "full usage of key flags"; do NOT invent flags or enum values. If a flag takes a value, give a real example value (e.g. `--spec-type draft-mtp` for MTP).
4. `why_tr`: WHY USE IT — the expected benefit (speed, VRAM, quality) and when it helps; caveats if any.
5. `flags`: the flags needed to TRY this feature that ACTUALLY EXIST in this build — whether new in this update or pre-existing (from the flag universe above). If a flag takes a value, write it with the value (e.g. `--spec-type draft-mtp`). Do NOT write nonexistent flags. Empty list for news-only cards with nothing concrete to try.
6. `architectures`: llama.cpp lowercase arch ids if the feature is specific to certain model architectures (e.g. "gemma3", "qwen3", "llama"); empty list for general features.
7. `source_urls`: related commit/PR/release URLs (the https://github.com/{GITHUB_REPO}/commit/<sha> form is fine).
8. `confidence`: evidence strength — "high" with flag + commit, "medium" if inferred only from a commit subject, "low" if speculative.
9. Prioritize updates that affect the user's model families, but don't skip important general updates.
10. Do NOT turn trivial internal changes (refactors, backend maintenance, CI) into cards. Produce AT MOST {lim["max_cards"]} cards; return an empty list if nothing noteworthy landed.
"""

    async def summarize_scan(self, scan_id: int) -> dict:
        """Produce cards with Claude from the stored raw scan data (idempotent:
        the scan's old cards are deleted first)."""
        scan = await self.get_scan(scan_id)
        if not scan:
            raise FeatureError(f"scan not found: {scan_id}")
        status = summary_status()
        if status["mode"] == "none":
            async with connect() as db:
                await db.execute(
                    "UPDATE feature_scans SET status='pending', error=? WHERE id=?",
                    (status["detail"], scan_id),
                )
                await db.commit()
            return (await self.get_scan(scan_id)) or scan

        families = await self._model_inventory()

        # The build's real flag universe (from the to_commit snapshot): grounds
        # the prompt (so the model can't invent flags) and validates card
        # flags. This also lets new-model cards carry existing spec flags
        # (like "--spec-type draft-mtp") into the "try it" action.
        async with connect() as db:
            cur = await db.execute(
                "SELECT help_text FROM help_snapshots WHERE commit_sha=? "
                "ORDER BY captured_at DESC LIMIT 1", (scan["to_commit"],),
            )
            hrow = await cur.fetchone()
        build_flags = extract_flags(hrow["help_text"]) if hrow else {}

        try:
            if status["provider"] == "openai":
                cards = await self._summarize_openai_adaptive(scan, families, build_flags)
            elif status["mode"] == "claude_cli":
                cards = await self._summarize_via_cli(
                    self._build_prompt(scan, families, build_flags))
            else:
                cards = await self._summarize_via_sdk(
                    self._build_prompt(scan, families, build_flags), status["mode"])
        except FeatureError as e:
            return await self._mark_scan_failed(scan_id, str(e))
        except Exception as e:
            return await self._mark_scan_failed(scan_id, f"summarization failed: {e}")

        # Validate card flags against what ACTUALLY exists in the build (new +
        # existing). If help couldn't be read, fall back to the old behavior
        # (new flags only).
        verified = set(build_flags) or {f["flag"] for f in scan["new_flags"]}
        now = time.time()
        async with connect() as db:
            await db.execute("DELETE FROM release_features WHERE scan_id=?", (scan_id,))
            for card in cards:
                flags = [
                    fl for fl in card["flags"]
                    if fl.split()[0].split("=")[0] in verified
                ]
                await db.execute(
                    "INSERT INTO release_features (scan_id, created_at, title_tr, what_tr, how_tr, why_tr, "
                    "flags_json, architectures_json, source_urls_json, confidence) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        scan_id, now,
                        card["title_tr"], card["what_tr"], card["how_tr"], card["why_tr"],
                        json.dumps(flags),
                        json.dumps([a.lower() for a in card["architectures"]]),
                        json.dumps(card["source_urls"]),
                        card["confidence"],
                    ),
                )
            await db.execute(
                "UPDATE feature_scans SET status='summarized', error=NULL WHERE id=?", (scan_id,)
            )
            await db.commit()
        log.info("features: scan %s summarized → %d cards", scan_id, len(cards))
        return (await self.get_scan(scan_id)) or scan

    async def _summarize_via_sdk(self, prompt: str, auth_mode: str) -> list[dict]:
        """Summary via the Anthropic API (settings key / env / profile).
        Note: the profile path needs API credits — a Pro subscription is not enough."""
        import anthropic  # lazy: the app should still boot if env is out of sync

        try:
            if auth_mode == "api_key":
                client = anthropic.AsyncAnthropic(api_key=load_settings().anthropic_api_key)
            else:
                client = anthropic.AsyncAnthropic()
            resp = await client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=8000,
                thinking={"type": "adaptive"},
                output_config={"format": {"type": "json_schema", "schema": FEATURES_SCHEMA}},
                messages=[{"role": "user", "content": prompt}],
            )
            text = next(b.text for b in resp.content if b.type == "text")
            return self._normalize_cards(json.loads(text)["features"])
        except anthropic.APIStatusError as e:
            raise FeatureError(f"Claude API error ({e.status_code}): {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise FeatureError(f"could not connect to the Claude API: {e}") from e

    async def _run_claude_cli(self, prompt: str, timeout: float = 600.0) -> str:
        """`claude -p` (Claude Code headless) call — works under a Pro/Max
        subscription, no API credits needed. The model is pinned to Sonnet to
        keep quota usage low. Returns the raw text result."""
        binary = find_claude_cli()
        if not binary:
            raise FeatureError("claude CLI not found — is Claude Code installed?")
        from .settings import STATE_DIR
        proc = await asyncio.create_subprocess_exec(
            binary, "-p", "--output-format", "json", "--model", CLAUDE_CLI_MODEL,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(STATE_DIR),  # avoid loading any project CLAUDE.md context
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(prompt.encode()), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise FeatureError("claude CLI timed out (10 min)")
        if proc.returncode != 0:
            raise FeatureError(
                f"claude CLI failed (exit {proc.returncode}): {err.decode(errors='replace')[:300]}"
            )
        try:
            envelope = json.loads(out.decode(errors="replace"))
        except Exception as e:
            raise FeatureError(f"could not parse claude CLI output: {e}") from e
        if envelope.get("is_error"):
            raise FeatureError(f"claude CLI: {str(envelope.get('result'))[:300]}")
        return str(envelope.get("result") or "").strip()

    # Appended on paths with no server-side schema enforcement (claude CLI,
    # OpenAI-compatible endpoints) — _normalize_cards validates the result.
    _JSON_SUFFIX = (
        "\n\n## Output format (MANDATORY)\n"
        "Do not use tools. Reply with ONLY a valid JSON object; "
        "no code fences, commentary or other text. Schema:\n"
        '{"features": [{"title_tr": str, "what_tr": str, "how_tr": str, '
        '"why_tr": str, "flags": [str], "architectures": [str], '
        '"source_urls": [str], "confidence": "high"|"medium"|"low"}]}'
    )

    async def _summarize_via_cli(self, prompt: str) -> list[dict]:
        text = re.sub(
            r"^```(?:json)?\s*|\s*```$", "",
            await self._run_claude_cli(prompt + self._JSON_SUFFIX),
        )
        try:
            return self._normalize_cards(json.loads(text)["features"])
        except FeatureError:
            raise
        except Exception as e:
            raise FeatureError(f"claude CLI JSON did not match the schema: {e}") from e

    def _openai_conf(self) -> tuple[str, str, dict[str, str]]:
        s = load_settings()
        base = (s.llm_base_url or "").strip().rstrip("/")
        model = (s.llm_model or "").strip()   # "" = whatever the endpoint serves
        if not base:
            raise FeatureError(_OPENAI_UNCONFIGURED_MSG)
        headers = {"Authorization": f"Bearer {s.llm_api_key}"} if s.llm_api_key else {}
        return base, model, headers

    # A local server that is still loading a model answers 503, and a busy one
    # can drop the connection; neither means the summary is impossible.
    _RETRY_STATUS = (429, 500, 502, 503, 504)
    _RETRY_BACKOFF = 2.0

    async def _openai_completion(
        self, prompt: str, *, timeout: float = 600.0,
        min_completion: int = _MIN_COMPLETION, attempts: int = 3,
    ) -> tuple[str, str]:
        """One chat-completions round against the configured OpenAI-compatible
        endpoint (OpenRouter, OpenAI, a local llama-server /v1, …).

        Returns (text, finish_reason). The model id is resolved against what
        the endpoint actually serves and `max_tokens` is fitted to its context
        window, so the two failure modes local servers hit most — a mismatched
        model name and a prompt larger than n_ctx — are handled before the call
        rather than surfacing as a raw 400."""
        base, configured, headers = self._openai_conf()
        info = await probe_endpoint(base, headers)
        model = resolve_model_id(configured, info["models"])

        max_tokens = _MAX_COMPLETION
        n_ctx = info["n_ctx"]
        if n_ctx:
            n_prompt = await count_tokens(prompt, base, headers, info["native"])
            room = n_ctx - n_prompt - _CTX_MARGIN
            if room < min_completion:
                raise ContextTooSmall(
                    f"the prompt ({n_prompt} tokens) does not fit the model's context "
                    f"({n_ctx} tokens) with room to answer — serve this model with "
                    f"--ctx-size {n_prompt + min_completion + _CTX_MARGIN} or more",
                    n_ctx=n_ctx, n_prompt=n_prompt,
                )
            max_tokens = max(min_completion, min(_MAX_COMPLETION, room))

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
        }
        last: Exception | None = None
        reprobed = False
        for attempt in range(attempts):
            try:
                async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
                    r = await client.post(f"{base}/chat/completions", json=payload)
            except httpx.HTTPError as e:
                last = FeatureError(f"could not reach the LLM API at {base}: {e}")
            else:
                if r.status_code == 200:
                    return self._read_completion(r.json())
                if r.status_code in (400, 404) and "not found" in r.text and not reprobed:
                    # The model behind the endpoint was swapped after the probe
                    # was cached: re-read what it serves and send that instead.
                    reprobed = True
                    _probe_cache.pop(base, None)
                    info = await probe_endpoint(base, headers)
                    payload["model"] = resolve_model_id(configured, info["models"])
                    log.warning("features: endpoint rejected the model id, re-probed")
                    continue
                if r.status_code == 400:
                    raise self._openai_400(r.text, n_ctx)
                if r.status_code not in self._RETRY_STATUS:
                    raise FeatureError(f"LLM API error ({r.status_code}): {r.text[:300]}")
                last = FeatureError(f"LLM API error ({r.status_code}): {r.text[:300]}")
            if attempt < attempts - 1:
                await asyncio.sleep(self._RETRY_BACKOFF * (attempt + 1))
                log.warning("features: LLM call failed (%s), retrying", last)
        raise last or FeatureError("the LLM call failed")

    @staticmethod
    def _openai_400(body: str, n_ctx: int | None) -> FeatureError:
        """A 400 carrying llama.cpp's context error becomes a typed
        ContextTooSmall so the caller can shrink the prompt and retry."""
        try:
            err = json.loads(body).get("error") or {}
        except Exception:
            err = {}
        if err.get("type") == "exceed_context_size_error" or "context size" in body:
            return ContextTooSmall(
                f"LLM API error (400): {err.get('message') or body[:300]}",
                n_ctx=err.get("n_ctx") or n_ctx,
                n_prompt=err.get("n_prompt_tokens"),
            )
        return FeatureError(f"LLM API error (400): {body[:300]}")

    @staticmethod
    def _read_completion(data: dict) -> tuple[str, str]:
        """Text + finish_reason out of a chat-completions body, tolerating the
        shapes local servers use: content as a list of parts, the answer under
        `reasoning_content`, or a bare completions-style `text`."""
        try:
            choice = data["choices"][0]
        except Exception as e:
            raise FeatureError(f"unexpected LLM API response shape: {e}") from e
        msg = choice.get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        text = strip_reasoning(content or "")
        if not text:
            text = strip_reasoning(msg.get("reasoning_content") or choice.get("text") or "")
        finish = str(choice.get("finish_reason") or "")
        if not text:
            if finish == "length":
                raise TruncatedOutput(
                    "the LLM used its whole token budget on reasoning without answering"
                )
            raise FeatureError("the LLM returned an empty response")
        return text, finish

    async def _openai_chat(self, prompt: str, timeout: float = 600.0,
                           min_completion: int = _MIN_COMPLETION) -> str:
        """Text-only wrapper (guide path)."""
        text, _ = await self._openai_completion(
            prompt, timeout=timeout, min_completion=min_completion,
        )
        return text

    async def _summarize_via_openai(self, prompt: str) -> list[dict]:
        text, finish = await self._openai_completion(prompt + self._JSON_SUFFIX)
        try:
            cards, salvaged = parse_feature_cards(text)
        except FeatureError:
            if finish == "length":
                raise TruncatedOutput(
                    "the LLM answer was cut off before any complete card"
                ) from None
            raise
        if salvaged:
            log.warning(
                "features: LLM JSON was truncated — salvaged %d complete card(s)",
                len(cards),
            )
        return self._normalize_cards(cards)

    async def _summarize_openai_adaptive(
        self, scan: dict, families: list[str], build_flags: dict[str, str],
    ) -> list[dict]:
        """Summarize against an OpenAI-compatible endpoint, stepping the prompt
        down through _DETAIL_LEVELS until it fits the context window and the
        answer arrives intact. Local models are served with whatever --ctx-size
        the preset happened to use, so the first level is a guess, not a given."""
        base, _, headers = self._openai_conf()
        info = await probe_endpoint(base, headers)

        level = 0
        if info["n_ctx"]:
            # Skip the levels that provably cannot fit, instead of burning a
            # slow local generation to discover it.
            for lvl in range(len(_DETAIL_LEVELS)):
                prompt = self._build_prompt(scan, families, build_flags, lvl)
                n_prompt = await count_tokens(
                    prompt + self._JSON_SUFFIX, base, headers, info["native"],
                )
                if info["n_ctx"] - n_prompt - _CTX_MARGIN >= _MIN_COMPLETION:
                    level = lvl
                    log.info(
                        "features: prompt detail level %d (%d of %d context tokens)",
                        lvl, n_prompt, info["n_ctx"],
                    )
                    break
            else:
                raise ContextTooSmall(
                    f"this model's context ({info['n_ctx']} tokens) is too small to "
                    f"summarize the update even at minimum detail — serve it with a "
                    f"larger --ctx-size (8192+) and retry",
                    n_ctx=info["n_ctx"],
                )

        last: FeatureError | None = None
        for lvl in range(level, len(_DETAIL_LEVELS)):
            prompt = self._build_prompt(scan, families, build_flags, lvl)
            try:
                cards = await self._summarize_via_openai(prompt)
            except (ContextTooSmall, TruncatedOutput) as e:
                last = e
                log.warning("features: detail level %d failed (%s) — shrinking", lvl, e)
                continue
            if lvl > level:
                log.info("features: summarized at reduced detail level %d", lvl)
            return cards
        raise last or FeatureError("the summary could not be produced")

    @staticmethod
    def _normalize_cards(cards: list) -> list[dict]:
        """Validate/default card fields — neither the CLI nor an
        OpenAI-compatible endpoint guarantees the schema, so every path goes
        through here.

        A malformed card is dropped rather than fatal: a small local model that
        writes six good cards and one without a `why_tr` should still leave the
        user with six cards."""
        out: list[dict] = []
        rejected = 0
        for c in cards:
            if not isinstance(c, dict):
                rejected += 1
                continue
            for k in ("title_tr", "what_tr", "how_tr", "why_tr"):
                v = c.get(k)
                if not isinstance(v, str) or not v.strip():
                    break
                # The model likes to wrap commands in markdown fences; cards
                # render as plain text, so strip the ``` lines at the source.
                c[k] = "\n".join(
                    ln for ln in v.splitlines() if not ln.lstrip().startswith("```")
                ).strip()
            else:
                for k in ("flags", "architectures", "source_urls"):
                    v = c.get(k)
                    # Empty strings would later blow up flag verification.
                    c[k] = (
                        [str(x).strip() for x in v if str(x).strip()]
                        if isinstance(v, list) else []
                    )
                if c.get("confidence") not in ("high", "medium", "low"):
                    c["confidence"] = "medium"
                out.append(c)
                continue
            rejected += 1
        if rejected:
            log.warning("features: dropped %d malformed card(s)", rejected)
        if cards and not out:
            raise FeatureError("no usable cards in the LLM response")
        return out

    async def _mark_scan_failed(self, scan_id: int, error: str) -> dict:
        log.warning("features: scan %s: %s", scan_id, error)
        async with connect() as db:
            await db.execute(
                "UPDATE feature_scans SET status='failed', error=? WHERE id=?", (error, scan_id)
            )
            await db.commit()
        return (await self.get_scan(scan_id)) or {"id": scan_id, "status": "failed", "error": error}

    # ---------- queries ----------

    async def list_cards(
        self, unseen_only: bool = False, arch: str | None = None,
        scan_to: str | None = None, limit: int = 100,
    ) -> list[dict]:
        q = (
            "SELECT rf.*, fs.from_commit, fs.to_commit, fs.build_number "
            "FROM release_features rf JOIN feature_scans fs ON fs.id = rf.scan_id"
        )
        conds, params = [], []
        if unseen_only:
            conds.append("rf.seen = 0")
        if scan_to:
            conds.append("fs.to_commit = ?")
            params.append(scan_to)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY rf.created_at DESC, rf.id DESC LIMIT ?"
        params.append(limit)
        async with connect() as db:
            cur = await db.execute(q, params)
            rows = await cur.fetchall()
        cards = [self._card_to_dict(r) for r in rows]
        if arch:
            a = arch.lower()
            cards = [
                c for c in cards
                if not c["architectures"] or any(x in a or a in x for x in c["architectures"])
            ]
        return cards

    @staticmethod
    def _card_to_dict(row) -> dict:
        d = dict(row)
        for col, key in (
            ("flags_json", "flags"), ("architectures_json", "architectures"),
            ("source_urls_json", "source_urls"),
        ):
            try:
                d[key] = json.loads(d.get(col) or "[]")
            except Exception:
                d[key] = []
            d.pop(col, None)
        return d

    async def list_scans(self, limit: int = 20) -> list[dict]:
        async with connect() as db:
            cur = await db.execute(
                "SELECT * FROM feature_scans ORDER BY created_at DESC LIMIT ?", (limit,)
            )
            rows = await cur.fetchall()
        return [self._scan_to_dict(r) for r in rows]

    async def unseen_count(self) -> int:
        async with connect() as db:
            cur = await db.execute("SELECT COUNT(*) AS n FROM release_features WHERE seen=0")
            cards = (await cur.fetchone())["n"]
            cur = await db.execute(
                "SELECT COUNT(*) AS n FROM feature_scans "
                "WHERE seen=0 AND status IN ('pending','failed')"
            )
            scans = (await cur.fetchone())["n"]
        return cards + scans

    async def mark_seen(self, feature_id: int) -> None:
        async with connect() as db:
            await db.execute(
                "UPDATE release_features SET seen=1, seen_at=? WHERE id=?",
                (time.time(), feature_id),
            )
            await db.commit()

    async def mark_all_seen(self) -> None:
        async with connect() as db:
            await db.execute(
                "UPDATE release_features SET seen=1, seen_at=? WHERE seen=0", (time.time(),)
            )
            await db.execute("UPDATE feature_scans SET seen=1 WHERE seen=0")
            await db.commit()

    async def get_card(self, feature_id: int) -> dict | None:
        async with connect() as db:
            cur = await db.execute("SELECT * FROM release_features WHERE id=?", (feature_id,))
            row = await cur.fetchone()
        return self._card_to_dict(row) if row else None

    # ---------- Guide (what the current build can do) ----------

    def guide_running(self) -> bool:
        return self._guide_running

    async def latest_guide(self) -> dict | None:
        async with connect() as db:
            cur = await db.execute("SELECT * FROM guides ORDER BY created_at DESC LIMIT 1")
            row = await cur.fetchone()
        return dict(row) if row else None

    async def start_guide(self) -> dict:
        """Start guide generation in the background (one at a time)."""
        if self._guide_running:
            raise FeatureError("a guide is already being generated")
        status = summary_status()
        if status["mode"] == "none":
            raise FeatureError(status["detail"])
        v = await get_build_manager().current_version()
        async with connect() as db:
            cur = await db.execute(
                "INSERT INTO guides (created_at, build_number, commit_sha, status) "
                "VALUES (?, ?, ?, 'running')",
                (time.time(), v.build_number, v.commit),
            )
            await db.commit()
            guide_id = cur.lastrowid
        self._guide_running = True
        asyncio.create_task(self._generate_guide(guide_id), name=f"guide-{guide_id}")
        return {"id": guide_id, "status": "running"}

    async def _guide_context(self) -> str:
        """Collect live system data for the guide prompt."""
        from .presets import PresetRegistry

        bm = get_build_manager()
        v = await bm.current_version()

        # --help flags (the ones that actually exist in this build)
        async with connect() as db:
            cur = await db.execute(
                "SELECT help_text FROM help_snapshots WHERE commit_sha=? "
                "ORDER BY captured_at DESC LIMIT 1", (v.commit,),
            )
            row = await cur.fetchone()
        flags = extract_flags(row["help_text"]) if row else {}
        flags_block = "\n".join(sorted(flags.keys()))
        # Full usage lines of the key enums — so the model doesn't invent
        # values (e.g. the real "draft-simple" instead of an invalid "draft-model").
        key_usage = "\n".join(
            flags[f] for f in ("--spec-type", "--flash-attn", "--cache-type-k") if f in flags
        )

        # MTP-capable architectures: deterministic scan of the source tree
        mtp_archs: list[str] = []
        models_src = Path(load_settings().llama_repo) / "src" / "models"
        try:
            for f in sorted(models_src.glob("*.cpp")):
                try:
                    if re.search(r"mtp", f.read_text(errors="ignore"), re.IGNORECASE):
                        mtp_archs.append(f.stem)
                except OSError:
                    continue
        except OSError:
            pass

        # Model inventory
        async with connect() as db:
            cur = await db.execute(
                "SELECT path, family, quant, size_bytes, has_mmproj FROM models "
                "ORDER BY family, size_bytes DESC"
            )
            models = await cur.fetchall()
        inv_lines = [
            f"  {m['family'] or '?':14s} {m['size_bytes']/1e9:6.1f} GB  "
            f"{m['path'].split('/')[-1]}{'  [has mmproj]' if m['has_mmproj'] else ''}"
            for m in models
        ]

        # Preset summary
        try:
            preset_lines = [
                f"  {c.name}: {(c.model_path or c.models_dir or '?').split('/')[-1]} "
                f"| port {c.port} | ctx {c.ctx_size} | spec {getattr(c, 'spec_type', 'none') or 'none'} | mode {getattr(c, 'mode', 'single')}"
                for c in PresetRegistry().list()
            ]
        except Exception:
            preset_lines = []

        # GPU
        gpu_line = "unknown"
        try:
            from .vram import probe_gpus
            gpus = await probe_gpus()
            if gpus:
                gpu_line = ", ".join(f"{g.name} ({g.total_mb} MB)" for g in gpus)
        except Exception:
            pass

        # Comparison with previous builds: scan history (which flags landed
        # in which update) + generated feature cards (with what/why summaries).
        async with connect() as db:
            cur = await db.execute(
                "SELECT from_commit, to_commit, build_number, new_flags_json "
                "FROM feature_scans WHERE status != 'empty' "
                "ORDER BY created_at DESC LIMIT 8"
            )
            scan_rows = await cur.fetchall()
            cur = await db.execute(
                "SELECT rf.title_tr, rf.what_tr, rf.why_tr, rf.flags_json, fs.build_number "
                "FROM release_features rf JOIN feature_scans fs ON fs.id = rf.scan_id "
                "ORDER BY rf.created_at DESC LIMIT 15"
            )
            card_rows = await cur.fetchall()
        history_lines: list[str] = []
        for s in scan_rows:
            try:
                nf = [f["flag"] for f in json.loads(s["new_flags_json"] or "[]")]
            except Exception:
                nf = []
            history_lines.append(
                f"  {s['from_commit']} → {s['to_commit']} (build {s['build_number']}): "
                f"new flags: {', '.join(nf) or '(none)'}"
            )
        card_lines: list[str] = []
        for c in card_rows:
            try:
                cf = json.loads(c["flags_json"] or "[]")
            except Exception:
                cf = []
            card_lines.append(
                f"  [build {c['build_number']}] {c['title_tr']} ({', '.join(cf) or 'no flags'})\n"
                f"    What: {c['what_tr'][:200]}\n    Why: {c['why_tr'][:200]}"
            )

        return f"""## System data (live, verified)
llama.cpp: build {v.build_number}, commit {v.commit}
GPU: {gpu_line}

### llama-server flags (present in this binary, {len(flags)} total):
{flags_block}

### Full usage of key flags (pick values EXACTLY from these):
{key_usage}

### Architectures compiled with MTP support (src/models scan):
{', '.join(mtp_archs) or 'not detected'}

### Local model inventory:
{chr(10).join(inv_lines) or '  (empty)'}

### Existing presets:
{chr(10).join(preset_lines) or '  (empty)'}

### Previous build updates (flag diff history):
{chr(10).join(history_lines) or '  (no records yet — fills after the first rebuild)'}

### Feature cards generated for recent updates:
{chr(10).join(card_lines) or '  (none yet)'}
"""

    async def _generate_guide(self, guide_id: int) -> None:
        try:
            context = await self._guide_context()
            lang = self._output_language()
            prompt = context + f"""
## Task
Based on the LIVE system data above, write a "current build guide" for this user in {lang}. The user manages llama.cpp through their own UI called LlamaDeck (it has Presets/Server/Bench/What's New pages).

Rules:
- Return ONLY Markdown; no extra commentary, greeting or code-fence wrapper. Do not use tools.
- State only what is VERIFIABLE from the data above; where unsure say "most likely", never invent.
- BE COMPREHENSIVE: target 1200-2000 words. This is a reference guide — the user wants a thorough answer to "what is this build's full potential, what can I do". Give concrete examples in every section (flag combination, expected effect, which LlamaDeck field it goes into). Don't just skim.
- Sections:
  ## Summary (what this build promises and for whom, 3-4 sentences)
  ## Since previous builds (from the diff history + feature cards above: which capabilities are new, what each gains in practice, how to try it with your models — 2-3 sentences per card)
  ## Speed: speculative decoding (for each `--spec-type` value: when, which model, expected gain, which fields in the LlamaDeck preset editor; match MTP-capable architectures against the inventory)
  ## What your models are good for (2-3 sentences for EVERY model in the inventory: strengths, ideal task, suggested ctx/kv settings, special features if any)
  ## VRAM in practice (concrete math for the GPU: which model + which ctx = how many GB; which pairs fit side by side; the effect of KV cache q8_0)
  ## Worth trying (5-7 concrete steps; each as clear as "do X on this LlamaDeck page → you'll see Y")
  ## Watch out (known limitations, orphaned files, incompatibilities)
- Write flag/command names as `code`; favor tables; no marketing language.
"""
            status = summary_status()
            if status["provider"] == "openai":
                # The guide asks for 1200-2000 words, so it needs more room to
                # answer than a card summary before the context is worth using.
                md = await self._openai_chat(prompt, min_completion=2048)
            elif status["mode"] == "claude_cli":
                md = await self._run_claude_cli(prompt)
            else:
                md = await self._generate_text_via_sdk(prompt)
            md = re.sub(r"^```(?:markdown|md)?\s*|\s*```$", "", md.strip())
            if len(md) < 200:
                raise FeatureError(f"guide came back unexpectedly short: {md[:120]!r}")
            async with connect() as db:
                await db.execute(
                    "UPDATE guides SET status='success', content_md=?, error=NULL WHERE id=?",
                    (md, guide_id),
                )
                await db.commit()
            log.info("guide %s generated (%d characters)", guide_id, len(md))
        except Exception as e:
            log.exception("guide %s generation failed", guide_id)
            async with connect() as db:
                await db.execute(
                    "UPDATE guides SET status='failed', error=? WHERE id=?",
                    (str(e), guide_id),
                )
                await db.commit()
        finally:
            self._guide_running = False

    async def _generate_text_via_sdk(self, prompt: str) -> str:
        """Plain-text SDK path for the guide (key/env credentials)."""
        import anthropic

        auth_mode = anthropic_auth_mode()
        try:
            if auth_mode == "api_key":
                client = anthropic.AsyncAnthropic(api_key=load_settings().anthropic_api_key)
            else:
                client = anthropic.AsyncAnthropic()
            resp = await client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=8000,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            )
            return next(b.text for b in resp.content if b.type == "text")
        except anthropic.APIStatusError as e:
            raise FeatureError(f"Claude API error ({e.status_code}): {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise FeatureError(f"could not connect to the Claude API: {e}") from e

    # ---------- try-it ----------

    async def try_feature(self, feature_id: int, preset_name: str, start: bool) -> dict:
        """Clone the preset, add the card's missing flags to extra_flags,
        optionally start it. The original preset is untouched."""
        from .presets import PresetRegistry
        from .supervisor import get_supervisor

        card = await self.get_card(feature_id)
        if not card:
            raise FeatureError(f"feature not found: {feature_id}")
        if not card["flags"]:
            raise FeatureError("this card has no flags to try (news-type update)")

        reg = PresetRegistry()
        base = reg.get(preset_name)

        # Only flags that can actually be appended. A card lists the flags a
        # change *touches*, so its list routinely includes `--model` and
        # `--ctx-size`; adding those verbatim yields `--model` with no value,
        # or a duplicate that shadows the preset's own field.
        from .flag_catalog import classify_flags
        from .settings import load_settings as _load_settings

        buckets = await classify_flags(base, _load_settings().llama_bin or "", card["flags"])
        added = buckets["actionable"]
        if not added:
            reason = "already in the preset" if buckets["present"] else (
                "settings LlamaDeck manages as fields — change them in the preset editor"
                if buckets["managed"] else
                "flags that need a value this card does not give"
                if buckets["needs_value"] else
                "flags this llama-server build does not have"
            )
            raise FeatureError(f"nothing to try here: this card only names {reason}.")
        merged = list(base.extra_flags)
        for flag in added:
            merged.extend(flag.split())

        clone_name = f"{base.name}-trial-{feature_id}"
        names = {c.name for c in reg.list()}
        n = 2
        while clone_name in names:
            clone_name = f"{base.name}-trial-{feature_id}-{n}"
            n += 1

        clone = replace(base, name=clone_name, extra_flags=merged)
        reg.upsert(clone)
        log.info("features: '%s' → '%s' cloned (+%s)", preset_name, clone_name, added)

        started = False
        if start:
            await get_supervisor().start(clone_name)
            started = True
        return {"preset": clone_name, "added_flags": added, "started": started}

    # ---------- A/B bench ----------

    async def start_ab(self, feature_id: int, model_path: str,
                       n_prompts: int = 512, n_gens: int = 128,
                       repetitions: int = 2) -> dict:
        """Two sequential llama-bench runs (flags off → on) as a background
        task. FeatureError if bench is busy."""
        from .bench import BenchStatus, get_bench_manager

        card = await self.get_card(feature_id)
        if not card:
            raise FeatureError(f"feature not found: {feature_id}")
        if not card["flags"]:
            raise FeatureError("this card has no flags to try in A/B")
        bm = get_bench_manager()
        active = bm.active()
        if self._ab_running or (active and active.status == BenchStatus.RUNNING):
            raise FeatureError("bench is busy — wait for the current run to finish")

        async with connect() as db:
            cur = await db.execute(
                "INSERT INTO feature_ab_runs (feature_id, model_path, flags_json, created_at, status) "
                "VALUES (?, ?, ?, ?, 'running')",
                (feature_id, model_path, json.dumps(card["flags"]), time.time()),
            )
            await db.commit()
            run_id = cur.lastrowid

        self._ab_running = True
        asyncio.create_task(
            self._run_ab(run_id, card, model_path, n_prompts, n_gens, repetitions),
            name=f"feature-ab-{run_id}",
        )
        return {"id": run_id, "status": "running"}

    async def _run_ab(self, run_id: int, card: dict, model_path: str,
                      n_prompts: int, n_gens: int, repetitions: int) -> None:
        from .bench import BenchParams, BenchStatus, get_bench_manager

        bm = get_bench_manager()
        flag_tokens: list[str] = []
        for flag in card["flags"]:
            flag_tokens.extend(flag.split())

        async def one_run(extra: list[str]) -> tuple[int, bool]:
            params = BenchParams(
                model_path=model_path,
                n_prompts=[n_prompts], n_gens=[n_gens],
                repetitions=repetitions, extra_flags=extra,
            )
            job = await bm.run(params)
            while job.status == BenchStatus.RUNNING:
                await asyncio.sleep(2)
            return job.id, job.status == BenchStatus.SUCCESS

        try:
            off_id, off_ok = await one_run([])
            async with connect() as db:
                await db.execute(
                    "UPDATE feature_ab_runs SET bench_off_id=? WHERE id=?", (off_id, run_id)
                )
                await db.commit()
            if not off_ok:
                raise FeatureError("off (baseline) bench run failed")

            on_id, on_ok = await one_run(flag_tokens)
            async with connect() as db:
                await db.execute(
                    "UPDATE feature_ab_runs SET bench_on_id=?, status=?, error=? WHERE id=?",
                    (on_id, "success" if on_ok else "failed",
                     None if on_ok else "on (feature) bench run failed", run_id),
                )
                await db.commit()
            log.info("features: A/B %s finished (off=%s on=%s)", run_id, off_id, on_id)
        except Exception as e:
            log.exception("features: A/B %s failed", run_id)
            async with connect() as db:
                await db.execute(
                    "UPDATE feature_ab_runs SET status='failed', error=? WHERE id=?",
                    (str(e), run_id),
                )
                await db.commit()
        finally:
            self._ab_running = False

    async def list_ab_runs(self, feature_id: int | None = None, limit: int = 20) -> list[dict]:
        """A/B runs joined with the results of their two benchmark rows."""
        async with connect() as db:
            if feature_id is not None:
                cur = await db.execute(
                    "SELECT * FROM feature_ab_runs WHERE feature_id=? ORDER BY created_at DESC LIMIT ?",
                    (feature_id, limit),
                )
            else:
                cur = await db.execute(
                    "SELECT * FROM feature_ab_runs ORDER BY created_at DESC LIMIT ?", (limit,)
                )
            rows = await cur.fetchall()
            out: list[dict] = []
            for r in rows:
                d = dict(r)
                try:
                    d["flags"] = json.loads(d.pop("flags_json") or "[]")
                except Exception:
                    d["flags"] = []
                for col, key in (("bench_off_id", "off"), ("bench_on_id", "on")):
                    d[key] = None
                    if d.get(col):
                        cur2 = await db.execute(
                            "SELECT id, status, results_json FROM benchmarks WHERE id=?",
                            (d[col],),
                        )
                        b = await cur2.fetchone()
                        if b:
                            try:
                                results = json.loads(b["results_json"] or "[]")
                            except Exception:
                                results = []
                            d[key] = {"id": b["id"], "status": b["status"], "results": results}
                out.append(d)
        return out


_instance: FeatureTracker | None = None


def get_feature_tracker() -> FeatureTracker:
    global _instance
    if _instance is None:
        _instance = FeatureTracker()
    return _instance
