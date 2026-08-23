"""End-to-end health watchdog for router-mode llama-server presets.

Why this exists
---------------
The supervisor (`supervisor.py`) only knows about the processes IT
launched. In router mode (`llama-server --models-preset ...`), the
spawned process itself spawns CHILD llama-servers per loaded model;
those grandchildren are invisible to LlamaDeck. When a grandchild dies
(OOM, CUDA fault, segfault) the router does not always reap it or
update its internal state — the model stays listed as ``loaded`` but
proxy calls fail with ``Could not establish connection``. Supervisor
sees the router process as alive and never triggers auto-restart.

The other half of that blind spot lives in `supervisor.liveness_loop`: an
ADOPTED process (every preset, once the backend has re-exec'd) has no exit
watcher, so nothing noticed when one died.

The watchdog closes that loop with an external observation:

  1. Pick the active router-mode preset(s).
  2. Probe every model the router reports as ``loaded`` with a real
     ``chat/completions`` round-trip.
  3. If a model probe fails for ``FAIL_THRESHOLD`` consecutive ticks,
     restart the router preset. Subsequent requests trigger a fresh
     lazy spawn against a clean process.

A window cap (``MAX_RESTARTS_PER_WINDOW`` in ``RESTART_WINDOW_S``)
prevents restart storms when something deeper is broken (kernel,
driver, model file) — at that point the operator needs to look.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Any

import httpx

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# How often to probe (seconds). 60s keeps the per-tick overhead negligible
# (~1 short chat round-trip per loaded model) while still catching a stuck
# router within ~3 minutes.
PROBE_INTERVAL_S: float = 60.0

# Consecutive failures required before triggering a restart. Raised from 3
# to 8 (2026-05-26) after observing watchdog kill a healthy router whose
# all 8 slots were busy with long canon-ingest extractions; ping queued >
# 8s timeout for 3 ticks → false-positive restart. 8 ticks = ~8 min of
# consistent breakage before kill, which still catches genuine hangs but
# tolerates slot-saturated heavy workloads.
FAIL_THRESHOLD: int = 8

# Per-probe timeout. Raised from 8s to 45s (2026-05-26) for the same
# reason — under heavy load a 1-token ping may wait for a slot to free,
# and 8s often isn't enough. 45s is still well under the 60s probe
# interval so probes never overlap.
PROBE_TIMEOUT_S: float = 45.0

# Timeout for the single-mode /health probe. Unlike the router ping this does
# not queue behind a slot — llama.cpp answers /health straight away — so it can
# be short. A process wedged by a GPU fault simply never answers.
HANG_PROBE_TIMEOUT_S: float = 15.0

# Restart rate cap.
MAX_RESTARTS_PER_WINDOW: int = 2
RESTART_WINDOW_S: float = 900.0  # 15 minutes


async def _probe_model(router_base: str, model_id: str) -> tuple[bool, str]:
    """End-to-end probe via /v1/chat/completions. Returns (ok, detail).

    autoload=false is mandatory: this probe exists to CHECK a model that the
    router reports as loaded. Without the flag the router's default autoload
    (true) makes the probe itself load the model — so a stale "loaded" entry
    (child killed by a GPU fault, or a state desync) gets silently reloaded,
    and with several models the wrong one can come back.
    """
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0.0,
    }
    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_S) as client:
            r = await client.post(
                f"{router_base}/v1/chat/completions",
                params={"autoload": "false"},
                json=payload,
            )
            if r.is_success:
                return True, "ok"
            body = (r.text or "")[:120]
            return False, f"http_{r.status_code}:{body}"
    except httpx.TimeoutException:
        return False, "timeout"
    except httpx.ConnectError as e:
        return False, f"connect:{e}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}:{e}"


async def _list_loaded_models(router_base: str) -> list[str]:
    """Ask the router which models it currently has loaded."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{router_base}/models")
            if not r.is_success:
                return []
            data = r.json().get("data", [])
            return [
                m["id"] for m in data
                if (m.get("status") or {}).get("value") == "loaded"
            ]
    except Exception as e:  # noqa: BLE001
        log.debug("watchdog: list_loaded failed: %s", e)
        return []


async def _probe_alive(base: str) -> tuple[bool, str]:
    """Is this llama-server still answering at all?

    A crash is caught by the supervisor's liveness poll — the process is gone.
    This catches the other shape of the same failure: the process is alive but
    wedged (a GPU fault can leave it that way), so the port accepts nothing and
    nothing exits. 503 is NOT a failure: that is llama.cpp saying "still
    loading the model", which is a healthy state to be in.
    """
    try:
        async with httpx.AsyncClient(timeout=HANG_PROBE_TIMEOUT_S) as c:
            r = await c.get(f"{base}/health")
    except Exception as e:  # noqa: BLE001 — any transport failure is a failure
        return False, f"{type(e).__name__}: {e}"
    if r.status_code == 503:
        return True, "loading"
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}"
    return True, "ok"


async def watchdog_loop(get_supervisor_fn) -> None:
    """Run forever. Cancelled on app shutdown via the lifespan task.

    `get_supervisor_fn` is passed in (not imported) so tests can inject a
    fake supervisor without monkeypatching module state.
    """
    fail_counts: dict[tuple[str, str], int] = defaultdict(int)  # (preset, model) → fails
    restart_history: dict[str, deque] = defaultdict(deque)       # preset → restart ts

    log.info(
        "health_watchdog: started (interval=%.0fs, threshold=%d, "
        "max_restarts=%d/%.0fs)",
        PROBE_INTERVAL_S, FAIL_THRESHOLD,
        MAX_RESTARTS_PER_WINDOW, RESTART_WINDOW_S,
    )

    while True:
        try:
            await asyncio.sleep(PROBE_INTERVAL_S)
            sup = get_supervisor_fn()
            statuses: dict[str, dict[str, Any]] = sup.statuses(vram_estimates=False)

            # Router presets need a per-MODEL probe (the router answers on
            # its own port while a loaded model behind it is wedged). Single
            # presets only need "does the port still answer" — handled inline
            # below, because "the process is alive" was never the same thing
            # as "the model still works".
            for preset_name, st in statuses.items():
                if not st.get("running"):
                    continue
                cfg = st.get("config") or {}
                port = st.get("port") or cfg.get("port")
                if cfg.get("mode") != "router":
                    # Single-mode: the process is alive (the liveness poll
                    # would have caught it otherwise), so the only question
                    # left is whether it still answers.
                    if not port:
                        continue
                    key = (preset_name, "__process__")
                    ok, detail = await _probe_alive(f"http://127.0.0.1:{port}")
                    if ok:
                        if fail_counts.get(key, 0) > 0:
                            log.info("health_watchdog: %s answering again after %d fails",
                                     preset_name, fail_counts[key])
                        fail_counts.pop(key, None)
                        continue
                    fail_counts[key] += 1
                    log.warning("health_watchdog: %s unresponsive %d/%d (%s)",
                                preset_name, fail_counts[key], FAIL_THRESHOLD, detail)
                    if fail_counts[key] < FAIL_THRESHOLD:
                        continue
                    now = time.time()
                    hist = restart_history[preset_name]
                    while hist and now - hist[0] > RESTART_WINDOW_S:
                        hist.popleft()
                    if len(hist) >= MAX_RESTARTS_PER_WINDOW:
                        log.error(
                            "health_watchdog: %s restart suppressed — %d restarts in "
                            "last %.0fs; OPERATOR INTERVENTION REQUIRED (unresponsive: %s)",
                            preset_name, len(hist), RESTART_WINDOW_S, detail,
                        )
                        continue
                    log.error("health_watchdog: %s alive but unresponsive for %d ticks "
                              "→ restarting (%s)", preset_name, fail_counts[key], detail)
                    hist.append(now)
                    try:
                        await sup.restart(preset_name)
                        fail_counts.pop(key, None)
                        log.info("health_watchdog: %s restart issued", preset_name)
                    except Exception as e:  # noqa: BLE001
                        log.exception("health_watchdog: %s restart failed: %s", preset_name, e)
                    continue
                if not port:
                    continue
                router_base = f"http://127.0.0.1:{port}"

                models = await _list_loaded_models(router_base)
                if not models:
                    # Nothing loaded → nothing to fail. Reset any stale fails.
                    for key in list(fail_counts.keys()):
                        if key[0] == preset_name:
                            fail_counts.pop(key, None)
                    continue

                broken: list[tuple[str, str]] = []
                for mid in models:
                    ok, detail = await _probe_model(router_base, mid)
                    key = (preset_name, mid)
                    if ok:
                        if fail_counts.get(key, 0) > 0:
                            log.info(
                                "health_watchdog: %s/%s recovered after %d fails",
                                preset_name, mid, fail_counts[key],
                            )
                        fail_counts.pop(key, None)
                    else:
                        fail_counts[key] += 1
                        log.warning(
                            "health_watchdog: %s/%s fail %d/%d (%s)",
                            preset_name, mid,
                            fail_counts[key], FAIL_THRESHOLD, detail,
                        )
                        if fail_counts[key] >= FAIL_THRESHOLD:
                            broken.append((mid, detail))

                if not broken:
                    continue

                # Restart-rate guard.
                now = time.time()
                hist = restart_history[preset_name]
                while hist and now - hist[0] > RESTART_WINDOW_S:
                    hist.popleft()
                if len(hist) >= MAX_RESTARTS_PER_WINDOW:
                    log.error(
                        "health_watchdog: %s restart suppressed — %d restarts "
                        "in last %.0fs; OPERATOR INTERVENTION REQUIRED. "
                        "Broken models: %s",
                        preset_name, len(hist), RESTART_WINDOW_S,
                        [m for m, _ in broken],
                    )
                    continue

                log.error(
                    "health_watchdog: %s router probe failed for %s "
                    "→ restarting (broken: %s)",
                    preset_name, [m for m, _ in broken], broken,
                )
                hist.append(now)
                try:
                    await sup.restart(preset_name)
                    log.info("health_watchdog: %s restart issued", preset_name)
                    # Clear all fail counters for this preset so the next
                    # tick starts from zero against the fresh process.
                    for key in list(fail_counts.keys()):
                        if key[0] == preset_name:
                            fail_counts.pop(key, None)
                except Exception as e:  # noqa: BLE001
                    log.exception("health_watchdog: %s restart failed: %s",
                                  preset_name, e)
        except asyncio.CancelledError:
            log.info("health_watchdog: cancelled, exiting")
            return
        except Exception:  # noqa: BLE001
            log.exception("health_watchdog: tick failed; loop continues")
