"""MetricsService: polls /slots and /metrics on each running preset at 2 Hz,
keeps a ring buffer, computes instantaneous tok/s from counter deltas, and
broadcasts frames to SSE subscribers.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from .prewarm import prewarm_from_args, prewarm_from_config
from .supervisor import MultiSupervisor

log = logging.getLogger(__name__)

POLL_HZ = 2.0
POLL_INTERVAL = 1.0 / POLL_HZ
RING_SIZE = 600  # 5 min at 2 Hz
HTTP_TIMEOUT = 4.0
# Surface frame.error in the UI only after this many consecutive failures, so
# a single ReadTimeout caused by llama-server being briefly busy with an
# inference batch doesn't flash a red "metric error" chip on the dashboard.
ERROR_GRACE_FRAMES = 2


_PROM_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9eE+\-.]+)\s*$")


def parse_prometheus(text: str) -> dict[str, float]:
    """Return metric_name -> value. Labels ignored (llama-server doesn't use them)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _PROM_LINE.match(line)
        if not m:
            continue
        try:
            out[m.group(1)] = float(m.group(3))
        except ValueError:
            continue
    return out


@dataclass
class SlotState:
    id: int
    is_processing: bool
    n_ctx: int
    task_id: int
    n_prompt_tokens: int = 0  # prompt length held in this slot's KV cache
    # Of that prompt, how much this request had to actually run through the
    # model, and how much came back from the KV cache. On a long chat the
    # second number is nearly all of it — which is why a 160k-token prompt can
    # come back instantly and the pp rate has to be read from the first.
    n_prompt_tokens_processed: int = 0
    n_prompt_tokens_cache: int = 0
    n_decoded: int = 0
    n_remain: int = 0
    n_predict: int = 0
    has_next_token: bool = False
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    speculative: bool = False


@dataclass
class JobRecord:
    """Aggregate metrics for a single inference task (start → end).

    Identified by (slot_id, task_id). The avg tok/s anchor is the first
    frame in which we observed `n_decoded > 0` (not the task's first
    sighting), so the prompt-processing phase — during which n_decoded
    stays at 0 but wall time advances — doesn't drag the rate down.
    `tokens_decoded` is the latest n_decoded; `decode_start_at` and
    `decode_baseline_tokens` define the anchor.
    """
    slot_id: int
    task_id: int
    started_at: float          # first sighting (may include prompt processing)
    last_seen_at: float
    tokens_decoded: int
    n_predict: int = 0
    ended_at: float | None = None
    decode_start_at: float | None = None  # first frame with n_decoded > 0
    decode_baseline_tokens: int = 0       # n_decoded at decode_start_at

    @property
    def duration(self) -> float:
        end = self.ended_at if self.ended_at is not None else self.last_seen_at
        return max(0.0, end - self.started_at)

    @property
    def avg_decode_tps(self) -> float:
        if self.decode_start_at is None:
            return 0.0
        end = self.ended_at if self.ended_at is not None else self.last_seen_at
        dt = end - self.decode_start_at
        if dt <= 0:
            return 0.0
        return max(0, self.tokens_decoded - self.decode_baseline_tokens) / dt

    def to_dict(self) -> dict:
        return {
            "slot_id": self.slot_id,
            "task_id": self.task_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "last_seen_at": self.last_seen_at,
            "duration_s": self.duration,
            "tokens_decoded": self.tokens_decoded,
            "n_predict": self.n_predict,
            "avg_decode_tps": self.avg_decode_tps,
            "active": self.ended_at is None,
            "decode_start_at": self.decode_start_at,
            "decode_baseline_tokens": self.decode_baseline_tokens,
        }


@dataclass
class MetricsFrame:
    ts: float
    preset: str
    port: int
    slots: list[SlotState]
    prom: dict[str, float]
    instant_prompt_tps: float | None = None
    instant_decode_tps: float | None = None
    requests_processing: int = 0
    requests_deferred: int = 0
    kv_cache_used_tokens: int = 0
    kv_cache_max_tokens: int = 0
    busy_slots: int = 0
    total_slots: int = 0
    error: str | None = None
    loaded_model_id: str | None = None  # router-mode: which model these stats are for
    active_jobs: list[JobRecord] = field(default_factory=list)
    recent_jobs: list[JobRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ts": self.ts,
            "preset": self.preset,
            "port": self.port,
            "slots": [
                {
                    "id": s.id,
                    "is_processing": s.is_processing,
                    "n_ctx": s.n_ctx,
                    "task_id": s.task_id,
                    "n_prompt_tokens": s.n_prompt_tokens,
                    "n_prompt_tokens_processed": s.n_prompt_tokens_processed,
                    "n_prompt_tokens_cache": s.n_prompt_tokens_cache,
                    "n_decoded": s.n_decoded,
                    "n_remain": s.n_remain,
                    "n_predict": s.n_predict,
                    "has_next_token": s.has_next_token,
                    "temperature": s.temperature,
                    "top_k": s.top_k,
                    "top_p": s.top_p,
                    "max_tokens": s.max_tokens,
                    "speculative": s.speculative,
                }
                for s in self.slots
            ],
            "prom": self.prom,
            "instant_prompt_tps": self.instant_prompt_tps,
            "instant_decode_tps": self.instant_decode_tps,
            "lifetime_prompt_tps": self.prom.get("llamacpp:prompt_tokens_seconds"),
            "lifetime_decode_tps": self.prom.get("llamacpp:predicted_tokens_seconds"),
            "requests_processing": self.requests_processing,
            "requests_deferred": self.requests_deferred,
            "kv_cache_used_tokens": self.kv_cache_used_tokens,
            "kv_cache_max_tokens": self.kv_cache_max_tokens,
            "busy_slots": self.busy_slots,
            "total_slots": self.total_slots,
            "error": self.error,
            "loaded_model_id": self.loaded_model_id,
            "active_jobs": [j.to_dict() for j in self.active_jobs],
            "recent_jobs": [j.to_dict() for j in self.recent_jobs],
        }


@dataclass
class PresetMetricsState:
    frames: deque[MetricsFrame] = field(default_factory=lambda: deque(maxlen=RING_SIZE))
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    consecutive_errors: int = 0
    # Active jobs keyed by (slot_id, task_id). When a key disappears from the
    # current slot snapshot we finalize the job into `recent_jobs`.
    active_jobs: dict[tuple[int, int], JobRecord] = field(default_factory=dict)
    recent_jobs: deque[JobRecord] = field(default_factory=lambda: deque(maxlen=20))

    def latest(self) -> MetricsFrame | None:
        return self.frames[-1] if self.frames else None

    def append(self, frame: MetricsFrame) -> None:
        self.frames.append(frame)
        for q in list(self.subscribers):
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                pass


class MetricsService:
    def __init__(self, supervisor: MultiSupervisor):
        self.supervisor = supervisor
        self.states: dict[str, PresetMetricsState] = {}
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._client: httpx.AsyncClient | None = None
        # (kind, id, port) tuples whose page-cache pre-warm has already been
        # fired. Router children get a fresh port on every reload, so a reload
        # warms again; single-mode keys on the pid so a restart does too.
        self._prewarmed: set[tuple[str, str, int]] = set()
        # asyncio only holds a weak reference to a running task, so a warm
        # fired and forgotten can be collected mid-flight. Hold it until it
        # finishes.
        self._prewarm_tasks: set[asyncio.Task] = set()

    def state(self, preset: str) -> PresetMetricsState:
        if preset not in self.states:
            self.states[preset] = PresetMetricsState()
        return self.states[preset]

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._client = httpx.AsyncClient(timeout=HTTP_TIMEOUT)
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name="metrics-poller")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                await self._poll_once()
            except Exception as e:
                log.exception("metrics poll failed: %s", e)
            elapsed = time.monotonic() - t0
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=max(0, POLL_INTERVAL - elapsed))
            except asyncio.TimeoutError:
                pass

    async def _poll_once(self) -> None:
        statuses = self.supervisor.statuses(vram_estimates=False)
        running = [(name, s) for name, s in statuses.items() if s.get("running")]
        if not running:
            return
        await asyncio.gather(*(self._poll_preset(name, s) for name, s in running), return_exceptions=True)

    async def _poll_preset(self, preset: str, status: dict) -> None:
        cfg = status.get("config") or {}
        port = cfg.get("port")
        host = cfg.get("host") or "127.0.0.1"
        if host == "0.0.0.0":
            host = "127.0.0.1"
        if not port or not self._client:
            return

        # Router-mode needs ?model=<id> on /slots and /metrics (which model's
        # stats?). Resolve the currently-loaded model via /models once, then
        # build an endpoint-specific query params dict. Single-mode: no param.
        is_router = cfg.get("mode") == "router"
        query_params: dict[str, str] = {}
        loaded_model_id: str | None = None
        loaded_model_args: list[str] | None = None
        if is_router:
            try:
                models_resp = await self._client.get(f"http://{host}:{port}/models")
                if models_resp.status_code == 200:
                    for m in (models_resp.json() or {}).get("data", []):
                        if isinstance(m, dict) and (m.get("status") or {}).get("value") == "loaded":
                            loaded_model_id = m.get("id")
                            loaded_model_args = (m.get("status") or {}).get("args") or []
                            break
            except Exception as e:
                log.debug("%s: router /models probe failed: %s", preset, e)
            if loaded_model_id is None:
                # No model loaded → no slots/metrics to fetch; record an empty
                # frame so UI knows the router is idle but alive.
                frame = MetricsFrame(
                    ts=time.time(),
                    preset=preset,
                    port=port,
                    slots=[],
                    prom={},
                    error="router idle (no model loaded)",
                )
                self.state(preset).append(frame)
                return
            # autoload=false so the poller never accidentally wakes a sleeping
            # model — /props is documented as non-load-triggering, but /slots
            # and /metrics aren't explicitly, and we saw them hang on unloaded.
            query_params = {"model": loaded_model_id, "autoload": "false"}
            child_port = port
            if loaded_model_args:
                for i, tok in enumerate(loaded_model_args):
                    if tok == "--port" and i + 1 < len(loaded_model_args):
                        try:
                            child_port = int(loaded_model_args[i + 1])
                        except ValueError:
                            pass
                        break
            self._maybe_prewarm("router", loaded_model_id, child_port, loaded_model_args or [])
        else:
            # Single-mode: the model is loaded by construction. Warm its
            # CPU-offloaded expert pages once per (preset, pid).
            self._maybe_prewarm("single", preset, int(status.get("pid") or 0), config=cfg)

        slots_raw: list = []
        prom_text: str = ""
        frame_error: str | None = None
        try:
            slots_resp, metrics_resp = await asyncio.gather(
                self._client.get(f"http://{host}:{port}/slots", params=query_params or None),
                self._client.get(f"http://{host}:{port}/metrics", params=query_params or None),
                return_exceptions=True,
            )
            if isinstance(slots_resp, Exception):
                frame_error = f"slots: {slots_resp}"
            elif slots_resp.status_code == 200:
                slots_raw = slots_resp.json()
            elif slots_resp.status_code == 501:
                frame_error = "slots disabled (--slots not set)"
            else:
                frame_error = f"slots HTTP {slots_resp.status_code}"

            if isinstance(metrics_resp, Exception):
                frame_error = (frame_error or "") + f" metrics: {metrics_resp}"
            elif metrics_resp.status_code == 200:
                prom_text = metrics_resp.text
            elif metrics_resp.status_code == 501:
                frame_error = (frame_error or "") + " metrics disabled (--metrics not set)"
        except Exception as e:
            frame_error = str(e)

        prom = parse_prometheus(prom_text) if prom_text else {}

        slots: list[SlotState] = []
        for s in slots_raw:
            try:
                next_token = s.get("next_token") or []
                nt = next_token[0] if isinstance(next_token, list) and next_token else (next_token if isinstance(next_token, dict) else {})
                params = s.get("params") or {}
                slots.append(SlotState(
                    id=s.get("id", 0),
                    is_processing=bool(s.get("is_processing")),
                    n_ctx=int(s.get("n_ctx", 0)),
                    task_id=int(s.get("id_task", -1)),
                    n_prompt_tokens=int(s.get("n_prompt_tokens", 0) or 0),
                    n_prompt_tokens_processed=int(s.get("n_prompt_tokens_processed", 0) or 0),
                    n_prompt_tokens_cache=int(s.get("n_prompt_tokens_cache", 0) or 0),
                    n_decoded=int(nt.get("n_decoded", 0) or 0),
                    n_remain=int(nt.get("n_remain", 0) or 0),
                    n_predict=int(params.get("n_predict", 0) or 0),
                    has_next_token=bool(nt.get("has_next_token", False)),
                    temperature=params.get("temperature"),
                    top_k=params.get("top_k"),
                    top_p=params.get("top_p"),
                    max_tokens=params.get("max_tokens"),
                    speculative=bool(s.get("speculative", False)),
                ))
            except (TypeError, ValueError):
                continue
        busy = sum(1 for s in slots if s.is_processing)

        # KV-cache occupancy. Max = total context = sum of every slot's n_ctx
        # (a stable capacity). Used = the tokens actually held in each busy
        # slot's cache (prompt + generated so far), clamped to that slot's
        # capacity. Do NOT use n_ctx as "used" (that's the slot's *capacity*,
        # not its fill) and do NOT use the llamacpp:n_tokens_max prom counter as
        # the max — it's a running high-water mark, not a fixed size, so pairing
        # it with slot capacity produced >100% readings.
        kv_max = sum(s.n_ctx for s in slots)
        kv_used = sum(
            min(s.n_ctx, s.n_prompt_tokens + s.n_decoded)
            for s in slots if s.is_processing
        )

        ts = time.time()
        instant_prompt_tps: float | None = None
        instant_decode_tps: float | None = None
        st = self.state(preset)
        prev = st.latest()

        # Prefer per-slot n_decoded delta for decode tok/s — this counter ticks
        # every batched step, giving a truly live signal. Prometheus
        # tokens_predicted_total only flushes at request completion in router
        # mode, so it stays at zero during a long inference and then spikes at
        # the end.
        #
        # Delta is only valid when the same slot is working on the SAME task as
        # the previous frame (matching task_id). When a task ends or a new
        # task starts, n_decoded resets — counting the new task's initial
        # n_decoded as a "delta" would fabricate a massive tok/s spike the
        # moment a new prompt lands. Treat the cross-task frame as "no signal"
        # (instant_decode_tps = None) so the rolling window simply drops it.
        # 250ms minimum dt: if consecutive polls landed too close (e.g. one poll
        # took 480ms so the next fired 20ms later), the delta is fine but the
        # divisor is so small that any single batch decode produces a 5-figure
        # tok/s spike. Below this floor we treat the frame as too noisy and
        # leave instant_decode_tps None (rolling window skips it).
        MIN_DT_FOR_RATE = 0.25
        if prev and prev.loaded_model_id == loaded_model_id:
            dt = ts - prev.ts
            if dt >= MIN_DT_FOR_RATE:
                prev_by_id = {
                    s.id: (s.task_id, s.n_decoded)
                    for s in prev.slots if s.is_processing
                }
                delta_tokens = 0
                counted_slots = 0
                cross_task = False
                for s in slots:
                    if not s.is_processing:
                        continue
                    prev_state = prev_by_id.get(s.id)
                    if prev_state is None:
                        cross_task = True  # slot just started a task this frame
                        continue
                    prev_task, prev_n = prev_state
                    if prev_task != s.task_id:
                        cross_task = True  # task transition on this slot
                        continue
                    if s.n_decoded >= prev_n:
                        delta_tokens += s.n_decoded - prev_n
                        counted_slots += 1
                prev_busy_any = any(s.is_processing for s in prev.slots)
                now_busy_any = any(s.is_processing for s in slots)
                if counted_slots > 0:
                    instant_decode_tps = delta_tokens / dt
                elif cross_task or prev_busy_any or now_busy_any:
                    # Boundary frame (task starting, ending, or transitioning):
                    # the Prometheus counter would double-count tokens already
                    # credited via slot deltas in earlier frames. Leave None
                    # and let the rolling average skip this point.
                    instant_decode_tps = None
                elif prom:
                    # Truly idle-to-idle window with a counter bump (rare —
                    # only happens if a task started and finished entirely
                    # between our 0.5s polls).
                    dd = prom.get("llamacpp:tokens_predicted_total", 0) - prev.prom.get("llamacpp:tokens_predicted_total", 0)
                    if dd > 0:
                        instant_decode_tps = dd / dt
                # Prompt rate, the same shape as the decode path above and for
                # the same reason: llamacpp:prompt_tokens_total only moves when
                # a request FINISHES, so a 160k-token prefill read 0 tok/s for
                # the whole minute it was running and then spiked once. The
                # slot's own n_prompt_tokens_processed ticks while the prefill
                # is happening, which is when someone is looking at the dial.
                prev_prompt = {
                    s.id: (s.task_id, s.n_prompt_tokens_processed)
                    for s in prev.slots if s.is_processing
                }
                delta_prompt = 0
                counted_prompt = 0
                for s in slots:
                    if not s.is_processing:
                        continue
                    prev_state = prev_prompt.get(s.id)
                    if prev_state is None or prev_state[0] != s.task_id:
                        continue  # new task: the counter restarted from 0
                    if s.n_prompt_tokens_processed >= prev_state[1]:
                        delta_prompt += s.n_prompt_tokens_processed - prev_state[1]
                        counted_prompt += 1
                if counted_prompt > 0:
                    # 0 is a real answer here — the slot is past its prompt and
                    # into generation, so nothing is being ingested right now.
                    instant_prompt_tps = delta_prompt / dt
                elif cross_task or prev_busy_any or now_busy_any:
                    # Boundary frame: the counter total is about to absorb
                    # tokens already credited above. Report no signal instead
                    # of double-counting them into a spike.
                    instant_prompt_tps = None
                elif prom:
                    dp = prom.get("llamacpp:prompt_tokens_total", 0) - prev.prom.get("llamacpp:prompt_tokens_total", 0)
                    if dp > 0:
                        instant_prompt_tps = dp / dt

        # Suppress single-frame transient errors (e.g. one /slots ReadTimeout
        # while llama-server was busy with an inference batch) so the dashboard
        # doesn't flash red on every blip. Surface the error only after it
        # persists across ERROR_GRACE_FRAMES consecutive polls.
        if frame_error:
            st.consecutive_errors += 1
        else:
            st.consecutive_errors = 0
        surfaced_error = frame_error if st.consecutive_errors >= ERROR_GRACE_FRAMES else None

        # Per-task job tracking: maintain a record per (slot_id, task_id) so we
        # can surface end-to-end total tokens and avg tok/s for each inference
        # — the 3-second rolling min/avg/max spans too narrow a slice to be
        # informative on its own.
        current_keys: set[tuple[int, int]] = set()
        for s in slots:
            if not s.is_processing or s.task_id < 0:
                continue
            key = (s.id, s.task_id)
            current_keys.add(key)
            job = st.active_jobs.get(key)
            if job is None:
                job = JobRecord(
                    slot_id=s.id,
                    task_id=s.task_id,
                    started_at=ts,
                    last_seen_at=ts,
                    tokens_decoded=s.n_decoded,
                    n_predict=s.n_predict,
                )
                if s.n_decoded > 0:
                    job.decode_start_at = ts
                    job.decode_baseline_tokens = s.n_decoded
                st.active_jobs[key] = job
            else:
                job.last_seen_at = ts
                if s.n_decoded > job.tokens_decoded:
                    job.tokens_decoded = s.n_decoded
                if s.n_predict:
                    job.n_predict = s.n_predict
                if job.decode_start_at is None and s.n_decoded > 0:
                    job.decode_start_at = ts
                    job.decode_baseline_tokens = s.n_decoded
        for key in list(st.active_jobs.keys()):
            if key not in current_keys:
                finished = st.active_jobs.pop(key)
                finished.ended_at = finished.last_seen_at
                st.recent_jobs.append(finished)

        frame = MetricsFrame(
            ts=ts,
            preset=preset,
            port=port,
            slots=slots,
            prom=prom,
            instant_prompt_tps=instant_prompt_tps,
            instant_decode_tps=instant_decode_tps,
            active_jobs=list(st.active_jobs.values()),
            recent_jobs=list(st.recent_jobs),
            requests_processing=int(prom.get("llamacpp:requests_processing", 0)),
            requests_deferred=int(prom.get("llamacpp:requests_deferred", 0)),
            kv_cache_used_tokens=kv_used,
            kv_cache_max_tokens=kv_max,
            busy_slots=busy,
            total_slots=len(slots),
            error=surfaced_error,
            loaded_model_id=loaded_model_id,
        )
        st.append(frame)

    # --- page-cache pre-warm ------------------------------------------------

    def _maybe_prewarm(
        self,
        kind: str,
        model_id: str,
        port: int,
        args: list[str] | None = None,
        config: dict | None = None,
    ) -> None:
        """Fire a one-shot page-cache warm for a freshly loaded model's
        CPU-offloaded expert tensors.

        The warm runs in a worker thread so the poll loop is never blocked;
        failures are logged, never fatal. Warming is idempotent per
        (kind, model_id, port): a router reload gets a fresh child port and
        warms again, single-mode keys on the server's pid.
        """
        key = (kind, model_id, int(port))
        if key in self._prewarmed:
            return
        self._prewarmed.add(key)

        def _work() -> None:
            try:
                if args is not None:
                    prewarm_from_args(args)
                elif config is not None:
                    prewarm_from_config(config)
            except Exception:
                log.exception("prewarm failed for %s %s", kind, model_id)

        try:
            task = asyncio.get_running_loop().create_task(asyncio.to_thread(_work))
        except RuntimeError:
            _work()  # no loop (tests, sync callers): warm inline
        else:
            self._prewarm_tasks.add(task)
            task.add_done_callback(self._prewarm_tasks.discard)

    # --- subscriptions ---

    def subscribe(self, preset: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=32)
        self.state(preset).subscribers.add(q)
        return q

    def unsubscribe(self, preset: str, q: asyncio.Queue) -> None:
        if preset in self.states:
            self.states[preset].subscribers.discard(q)

    async def stream(self, preset: str) -> AsyncIterator[MetricsFrame]:
        st = self.state(preset)
        if st.frames:
            yield st.frames[-1]
        q = self.subscribe(preset)
        try:
            while True:
                yield await q.get()
        finally:
            self.unsubscribe(preset, q)

    def snapshot(self, history_n: int = 120) -> dict:
        out: dict = {"presets": {}}
        for name, st in self.states.items():
            frames = list(st.frames)[-history_n:]
            out["presets"][name] = {
                "latest": st.latest().to_dict() if st.latest() else None,
                "history": [f.to_dict() for f in frames],
            }
        return out


_instance: MetricsService | None = None


def get_metrics_service(supervisor: MultiSupervisor | None = None) -> MetricsService:
    global _instance
    if _instance is None:
        if supervisor is None:
            raise RuntimeError("metrics service not initialized")
        _instance = MetricsService(supervisor)
    return _instance
