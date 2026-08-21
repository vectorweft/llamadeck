"""GPU broker — single-GPU slot lease registry.

LlamaDeck owns the GPU. Every client (HFD-2, SCont, hermes-agent, curl) that
wants to do GPU work acquires a slot lease here. Slots are mutually
exclusive on a single GPU; the broker enforces that exactly ONE slot
is "active" at any time and queues callers waiting for a different slot.

Slot semantics:
    "llm"   — llama-server (managed url :8080 etc.)
    "comfy" — ComfyUI       (managed url :8188)
    "tts"   — XTTS server   (managed url :8020)

State machine:
    idle → starting → active → releasing → idle

Concurrency rules (lock discipline):
    - `self._lock` protects state transitions ONLY (active/starting slot,
      lease/queue dicts, failure counters).
    - Service start/stop calls (`Service.start`, `wait_ready`, `stop`) are
      ALWAYS run outside the lock. Long startups (e.g. llama-server cold
      load takes 30-120s) must not block other broker ops like `state()`,
      `release()`, or unrelated-slot acquires.
    - When a slot needs to be started or transitioned, the originating
      `acquire()` enqueues a waiter and a background task does the work,
      then re-takes the lock to grant queued waiters atomically.

Caller flow:
    1. POST /api/gpu/lease {slot, holder, est_duration_s, est_vram_mb}
    2. broker either:
       a) granted immediately (current slot matches AND ready), OR
       b) queues caller, transitions slot when current slot is released
    3. POST /api/gpu/lease/{id}/heartbeat to extend ttl
    4. POST /api/gpu/lease/{id}/release when done

The broker also runs an autonomous keepalive timer: when a slot is
released and the queue is empty, the active service is kept warm for
`settings.llm_keepalive_s` (default 300s); after that it is stopped to
free VRAM — but `LlmService.stop` is a no-op for user/adopted presets,
so externally-owned llama-servers stay alive.

Failure backoff: if a slot's start fails 3 times in 60s, subsequent
acquires for that slot return 503 "service.unhealthy" with a 10s
cooldown. Designed to align with HFD-2's 5s/10s 503-retry windows
(shared/llm_client.py) so client retries naturally pick up after the
cooldown clears.

This module deliberately does NOT touch process management directly —
it goes through the `Service` interface so llm/comfy/tts can each be
swapped out independently. Concrete services live in lld.processes.*
"""
from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal, Protocol

log = logging.getLogger("lld.gpu_broker")


SlotName = Literal["llm", "comfy", "tts"]
ALL_SLOTS: tuple[SlotName, ...] = ("llm", "comfy", "tts")

# Failure backoff tuning — kept in sync with HFD-2 503 retry policy.
_FAILURE_WINDOW_S = 60.0
_FAILURE_THRESHOLD = 3
_COOLDOWN_S = 10.0


# ---------------------------------------------------------------------------
# Service interface — what a slot manager must implement
# ---------------------------------------------------------------------------

class Service(Protocol):
    """Concrete services (LlmService, ComfyService, XttsService) implement
    this protocol. The broker only calls these four methods plus reads
    the three attributes."""

    name: SlotName
    managed_url: str            # http://127.0.0.1:NNNN  (may be a property)
    est_vram_mb: int            # rough budget for VRAM rejection check (may be a property)
    # Optional: llama.cpp device ids this service will load onto ("Vulkan1").
    # Absent or empty means "wherever the runtime chooses". The pre-spawn VRAM
    # check reads it so a pinned service is measured against ITS card.

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def wait_ready(self, timeout_s: float) -> bool: ...
    def is_running(self) -> bool: ...


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

@dataclass
class Lease:
    lease_id: str
    slot: SlotName
    holder: str
    granted_at: float
    expires_at: float           # absolute epoch time
    est_duration_s: float
    est_vram_mb: int
    ephemeral: bool = False     # True for OpenAI proxy auto-leases

    def remaining_s(self) -> float:
        return max(0.0, self.expires_at - time.time())

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "slot": self.slot,
            "holder": self.holder,
            "granted_at": self.granted_at,
            "expires_at": self.expires_at,
            "ttl_s": round(self.remaining_s(), 1),
            "est_duration_s": self.est_duration_s,
            "est_vram_mb": self.est_vram_mb,
            "ephemeral": self.ephemeral,
        }


@dataclass
class _QueuedRequest:
    slot: SlotName
    holder: str
    est_duration_s: float
    est_vram_mb: int
    queued_at: float
    waiter: asyncio.Future        # resolves to a Lease or BrokerError
    deadline_at: float            # absolute epoch
    ephemeral: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "holder": self.holder,
            "queued_at": self.queued_at,
            "est_duration_s": self.est_duration_s,
            "deadline_at": self.deadline_at,
        }


class BrokerError(Exception):
    """Raised when an acquire / release / heartbeat fails for a known reason."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------

class GpuBroker:
    """Single-GPU slot broker. Singleton via get_broker()."""

    def __init__(
        self,
        services: dict[SlotName, Service],
        *,
        keepalive_s: float = 300.0,
        default_ttl_s: float = 600.0,
        gpu_total_vram_mb: int | None = None,
        external_reserved_vram_mb: int = 0,
        vram_spawn_headroom_mb: int = 0,
    ):
        self._services = dict(services)
        for slot in ALL_SLOTS:
            if slot not in self._services:
                raise ValueError(f"Service for slot '{slot}' not provided")

        self._keepalive_s = float(keepalive_s)
        self._default_ttl_s = float(default_ttl_s)
        self._gpu_total_vram_mb = gpu_total_vram_mb
        self._external_reserved_vram_mb = int(external_reserved_vram_mb)
        self._vram_spawn_headroom_mb = int(vram_spawn_headroom_mb)

        self._lock = asyncio.Lock()                 # serialize state transitions
        self._active_slot: SlotName | None = None
        self._starting_slot: SlotName | None = None  # background _start_and_grant in flight
        self._active_leases: dict[str, Lease] = {}  # lease_id → Lease
        self._queue: list[_QueuedRequest] = []
        self._failed_starts: dict[SlotName, deque[float]] = {}  # slot → recent failure timestamps
        self._keepalive_task: asyncio.Task | None = None
        self._heartbeat_watchdog_task: asyncio.Task | None = None
        self._background_tasks: set[asyncio.Task] = set()       # _start_and_grant tasks
        self._stopped = False

    def set_gpu_total_vram_mb(self, total_mb: int | None) -> None:
        """Fill in the VRAM budget after construction.

        The budget comes from a hardware probe that shells out to vendor tools,
        and boot no longer waits for it — those tools hang on a machine whose
        driver is halfway through an upgrade. The answer arrives here instead.
        Until it does, `_gpu_total_vram_mb` stays None and the admission check
        is skipped, exactly as on a box with no GPU telemetry.
        """
        self._gpu_total_vram_mb = total_mb

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Launch background watchdog. Idempotent."""
        if self._heartbeat_watchdog_task is None or self._heartbeat_watchdog_task.done():
            self._heartbeat_watchdog_task = asyncio.create_task(
                self._watchdog_loop(), name="gpu_broker.watchdog",
            )
        log.info("gpu_broker started; keepalive=%.0fs default_ttl=%.0fs",
                 self._keepalive_s, self._default_ttl_s)

    async def stop(self) -> None:
        """Stop all services, cancel queue, shut watchdog."""
        self._stopped = True
        async with self._lock:
            for q in self._queue:
                if not q.waiter.done():
                    q.waiter.set_exception(BrokerError("broker.shutdown", "broker shutting down", status=503))
            self._queue.clear()
            self._active_leases.clear()
            slot = self._active_slot
            self._active_slot = None
            self._starting_slot = None
        if slot is not None:
            try:
                await self._services[slot].stop()
            except Exception as e:  # noqa: BLE001
                log.warning("stop(%s) failed: %s", slot, e)
        # Cancel any in-flight start_and_grant tasks
        for t in list(self._background_tasks):
            if not t.done():
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        self._background_tasks.clear()
        for task in (self._keepalive_task, self._heartbeat_watchdog_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    # -- public API ----------------------------------------------------------

    async def acquire(
        self,
        slot: SlotName,
        holder: str,
        *,
        est_duration_s: float = 120.0,
        est_vram_mb: int = 0,
        wait_timeout_s: float = 300.0,
        ephemeral: bool = False,
    ) -> Lease:
        """Block until a lease for `slot` is granted or wait_timeout_s passes.

        Granted immediately when:
          - active slot is `slot` AND no start is in flight (multi-lease same slot allowed)

        Otherwise queued. Either:
          - we are the first waiter for an idle slot → spawn a background
            _start_and_grant task that starts the service then grants all
            same-slot waiters atomically;
          - the active slot is held → wait until release triggers transition.

        Returns 503 service.unhealthy if the slot is in a failure-cooldown.
        Returns 408 acquire.timeout if no lease arrives within wait_timeout_s.
        """
        if self._stopped:
            raise BrokerError("broker.stopped", "broker is not running", status=503)
        if slot not in ALL_SLOTS:
            raise BrokerError("invalid_slot", f"unknown slot '{slot}'", status=400)
        if not holder or not holder.strip():
            raise BrokerError("invalid_holder", "holder is required", status=400)

        if (
            self._gpu_total_vram_mb is not None
            and est_vram_mb > 0
            and est_vram_mb > self._gpu_total_vram_mb
        ):
            raise BrokerError(
                "vram.exceeds_capacity",
                f"requested {est_vram_mb} MB > GPU capacity {self._gpu_total_vram_mb} MB",
                status=503,
            )

        we_will_start = False
        prev_slot_for_transition: SlotName | None = None

        async with self._lock:
            # Cooldown gate — stay aligned with HFD-2's 5s/10s retry budget.
            cooldown = self._cooldown_remaining_s_locked(slot)
            if cooldown > 0:
                raise BrokerError(
                    "service.unhealthy",
                    f"slot '{slot}' cooling down {cooldown:.1f}s after recent start failures",
                    status=503,
                )

            # Fast path: same slot active and ready, no start in flight.
            if self._active_slot == slot and self._starting_slot is None:
                if self._keepalive_task and not self._keepalive_task.done():
                    self._keepalive_task.cancel()
                    self._keepalive_task = None
                lease = self._make_lease(
                    slot=slot, holder=holder,
                    est_duration_s=est_duration_s,
                    est_vram_mb=est_vram_mb,
                    ephemeral=ephemeral,
                )
                self._active_leases[lease.lease_id] = lease
                log.info("granted-fast lease=%s slot=%s holder=%s ttl=%.0fs",
                         lease.lease_id, slot, holder, lease.remaining_s())
                return lease

            # Slow path: queue. Then decide whether we trigger the start.
            now = time.time()
            waiter: asyncio.Future = asyncio.get_event_loop().create_future()
            req = _QueuedRequest(
                slot=slot, holder=holder,
                est_duration_s=est_duration_s,
                est_vram_mb=est_vram_mb,
                queued_at=now,
                waiter=waiter,
                deadline_at=now + max(1.0, wait_timeout_s),
                ephemeral=ephemeral,
            )
            self._queue.append(req)
            log.info("queued slot=%s holder=%s qlen=%d", slot, holder, len(self._queue))

            # Trigger a start/transition if nothing is in progress.
            if self._starting_slot is None:
                if self._active_slot is None:
                    we_will_start = True
                    prev_slot_for_transition = None
                    self._starting_slot = slot
                elif self._active_slot != slot and not self._active_leases:
                    # Active slot is idle (no leases) → safe to transition now,
                    # not waiting for an explicit release.
                    we_will_start = True
                    prev_slot_for_transition = self._active_slot
                    self._active_slot = None
                    self._starting_slot = slot
                # else: active_slot has leases → wait for release_to_transition

        if we_will_start:
            self._spawn_start_and_grant(slot, prev_slot_for_transition)

        try:
            return await asyncio.wait_for(waiter, timeout=wait_timeout_s)
        except asyncio.TimeoutError:
            async with self._lock:
                self._queue = [q for q in self._queue if q is not req]
            raise BrokerError(
                "acquire.timeout",
                f"no lease for slot={slot} within {wait_timeout_s:.0f}s",
                status=408,
            )

    async def heartbeat(self, lease_id: str, *, ttl_s: float | None = None) -> Lease:
        async with self._lock:
            lease = self._active_leases.get(lease_id)
            if lease is None:
                raise BrokerError("lease.not_found", f"unknown lease {lease_id}", status=404)
            new_ttl = float(ttl_s if ttl_s is not None else self._default_ttl_s)
            lease.expires_at = time.time() + new_ttl
            return lease

    async def release(self, lease_id: str) -> None:
        target_slot: SlotName | None = None
        prev_for_transition: SlotName | None = None
        schedule_keepalive = False
        async with self._lock:
            lease = self._active_leases.pop(lease_id, None)
            if lease is None:
                raise BrokerError("lease.not_found", f"unknown lease {lease_id}", status=404)
            log.info("released lease=%s slot=%s holder=%s leases_left=%d",
                     lease_id, lease.slot, lease.holder, len(self._active_leases))
            if self._active_leases:
                return
            if self._starting_slot is not None:
                # A start is already in flight — it'll grant queue waiters when done.
                return
            if self._queue:
                target_slot = self._queue[0].slot
                if target_slot == self._active_slot:
                    # Same slot in queue (rare race): grant immediately, no transition.
                    self._grant_same_slot_waiters_locked(target_slot)
                    return
                prev_for_transition = self._active_slot
                self._active_slot = None
                self._starting_slot = target_slot
            else:
                schedule_keepalive = True

        if target_slot is not None:
            self._spawn_start_and_grant(target_slot, prev_for_transition)
        elif schedule_keepalive:
            async with self._lock:
                self._schedule_keepalive_locked()

    async def notify_supervisor_changed(self) -> None:
        """Reconcile broker state after the supervisor changed under us
        (start/stop/restart/switch/adopt/release via /api/server/*).

        Invariants we restore:
          - If active_slot=="llm" but the LLM service is no longer running
            (user stopped it), expire all llm leases and mark slot idle.
          - Drop the LlmService auto-managed flag so a future keepalive
            doesn't try to stop a now-externally-owned preset.
          - Trigger any queued transition waiters.
        """
        target_slot: SlotName | None = None
        prev_for_transition: SlotName | None = None
        async with self._lock:
            llm_svc = self._services.get("llm")
            # LlmService manages its own ownership flag. Always drop it: any
            # /api/server/* mutation means the user is steering, not us.
            forget = getattr(llm_svc, "forget_ownership", None)
            if callable(forget):
                forget()
            if self._active_slot == "llm" and llm_svc is not None and not llm_svc.is_running():
                expired = [lid for lid, lease in self._active_leases.items() if lease.slot == "llm"]
                for lid in expired:
                    self._active_leases.pop(lid, None)
                if expired:
                    log.warning("supervisor change: invalidated %d llm leases (service stopped externally)",
                                len(expired))
                self._active_slot = None
                if self._starting_slot is None and self._queue:
                    target_slot = self._queue[0].slot
                    prev_for_transition = None
                    self._starting_slot = target_slot
        if target_slot is not None:
            self._spawn_start_and_grant(target_slot, prev_for_transition)

    def state(self) -> dict[str, Any]:
        return {
            "active_slot": self._active_slot,
            "starting_slot": self._starting_slot,
            "active_leases": [lease.to_dict() for lease in self._active_leases.values()],
            "queue": [q.to_dict() for q in self._queue],
            "keepalive_s": self._keepalive_s,
            "default_ttl_s": self._default_ttl_s,
            "recent_start_failures": {
                slot: list(times) for slot, times in self._failed_starts.items() if times
            },
            "services": {
                slot: {
                    "running": svc.is_running(),
                    "managed_url": svc.managed_url,
                    "est_vram_mb": svc.est_vram_mb,
                }
                for slot, svc in self._services.items()
            },
        }

    def get_lease(self, lease_id: str) -> Lease | None:
        return self._active_leases.get(lease_id)

    def active_slot(self) -> SlotName | None:
        return self._active_slot

    def service_url(self, slot: SlotName) -> str:
        # `managed_url` may be a property on the concrete service (e.g.
        # LlmService resolves dynamically), so we read it fresh every call.
        return self._services[slot].managed_url

    # -- internal helpers ----------------------------------------------------

    def _make_lease(
        self,
        *,
        slot: SlotName,
        holder: str,
        est_duration_s: float,
        est_vram_mb: int,
        ephemeral: bool,
    ) -> Lease:
        now = time.time()
        ttl = max(self._default_ttl_s, est_duration_s * 1.2)
        return Lease(
            lease_id=secrets.token_urlsafe(12),
            slot=slot,
            holder=holder,
            granted_at=now,
            expires_at=now + ttl,
            est_duration_s=est_duration_s,
            est_vram_mb=est_vram_mb,
            ephemeral=ephemeral,
        )

    def _spawn_start_and_grant(self, slot: SlotName, prev_slot: SlotName | None) -> None:
        task = asyncio.create_task(
            self._start_and_grant(slot, prev_slot),
            name=f"gpu_broker.start_and_grant.{slot}",
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _start_and_grant(self, slot: SlotName, prev_slot: SlotName | None) -> None:
        """Stop previous slot (if transitioning), start target, grant queued waiters.

        Runs entirely outside `self._lock` for the slow steps; only re-takes
        the lock to mutate state (set active_slot, hand out leases, record
        failures). This is what keeps unrelated broker ops responsive
        during a long llama-server cold load.
        """
        if prev_slot is not None and prev_slot != slot:
            log.info("transition %s → %s", prev_slot, slot)
            try:
                await self._services[prev_slot].stop()
            except Exception as e:  # noqa: BLE001
                log.warning("stop(%s) failed during transition: %s", prev_slot, e)

        try:
            await self._start_service(slot)
        except Exception as e:  # noqa: BLE001
            log.warning("start(%s) failed: %s", slot, e)
            async with self._lock:
                self._record_failure_locked(slot)
                self._starting_slot = None
                err_msg = str(e) or type(e).__name__
                err = BrokerError(
                    "service.start_failed",
                    f"start({slot}) failed: {err_msg}",
                    status=503,
                )
                failed_qs = [q for q in self._queue if q.slot == slot]
                self._queue = [q for q in self._queue if q.slot != slot]
                for q in failed_qs:
                    if not q.waiter.done():
                        q.waiter.set_exception(err)
                # If queue still has different-slot requests, kick a transition.
                if not self._active_leases and self._active_slot is None and self._queue:
                    next_slot = self._queue[0].slot
                    self._starting_slot = next_slot
                    asyncio.get_event_loop().call_soon(
                        lambda: self._spawn_start_and_grant(next_slot, None)
                    )
            return

        async with self._lock:
            self._active_slot = slot
            self._starting_slot = None
            self._grant_same_slot_waiters_locked(slot)

    def _grant_same_slot_waiters_locked(self, slot: SlotName) -> None:
        """Pop all queued waiters for `slot` and grant them leases. Caller holds lock."""
        same = [q for q in self._queue if q.slot == slot]
        self._queue = [q for q in self._queue if q.slot != slot]
        for q in same:
            if q.waiter.done():
                continue
            lease = self._make_lease(
                slot=q.slot, holder=q.holder,
                est_duration_s=q.est_duration_s,
                est_vram_mb=q.est_vram_mb,
                ephemeral=q.ephemeral,
            )
            self._active_leases[lease.lease_id] = lease
            q.waiter.set_result(lease)
            log.info("granted-from-queue lease=%s slot=%s holder=%s",
                     lease.lease_id, lease.slot, lease.holder)

    def _record_failure_locked(self, slot: SlotName) -> None:
        now = time.time()
        dq = self._failed_starts.setdefault(slot, deque())
        dq.append(now)
        cutoff = now - _FAILURE_WINDOW_S
        while dq and dq[0] < cutoff:
            dq.popleft()

    def _cooldown_remaining_s_locked(self, slot: SlotName) -> float:
        """How many seconds until acquires for `slot` are allowed again. 0 if healthy."""
        dq = self._failed_starts.get(slot)
        if not dq:
            return 0.0
        # Drop stale failures
        cutoff = time.time() - _FAILURE_WINDOW_S
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) < _FAILURE_THRESHOLD:
            return 0.0
        last_failure = dq[-1]
        remaining = _COOLDOWN_S - (time.time() - last_failure)
        if remaining <= 0:
            # Cooldown expired; clear so we don't keep gating after a long quiet period
            dq.clear()
            return 0.0
        return remaining

    def _schedule_keepalive_locked(self) -> None:
        """Active slot empty + queue empty: keep service warm for keepalive_s,
        then call svc.stop() (which is a no-op for externally-owned LLM presets)."""
        if self._active_slot is None:
            return
        if self._keepalive_s <= 0:
            asyncio.create_task(self._stop_active_now(), name="gpu_broker.stop_immediate")
            return

        slot = self._active_slot
        log.info("scheduling keepalive stop slot=%s in %.0fs",
                 slot, self._keepalive_s)

        async def _runner() -> None:
            try:
                await asyncio.sleep(self._keepalive_s)
            except asyncio.CancelledError:
                return
            async with self._lock:
                if self._active_slot != slot or self._active_leases or self._queue:
                    return
                log.info("keepalive expired, stopping slot=%s", slot)
                try:
                    await self._services[slot].stop()
                except Exception as e:  # noqa: BLE001
                    log.warning("keepalive stop(%s) failed: %s", slot, e)
                # Only release the slot if the service is actually down.
                # LlmService.stop is a no-op for externally-owned presets,
                # so the LLM may still be running; in that case keep the slot
                # pinned so the next acquire takes the fast path.
                if not self._services[slot].is_running():
                    self._active_slot = None

        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
        self._keepalive_task = asyncio.create_task(_runner(), name="gpu_broker.keepalive")

    async def _stop_active_now(self) -> None:
        async with self._lock:
            slot = self._active_slot
            if slot is None or self._active_leases or self._queue:
                return
            try:
                await self._services[slot].stop()
            except Exception as e:  # noqa: BLE001
                log.warning("immediate stop(%s) failed: %s", slot, e)
            if not self._services[slot].is_running():
                self._active_slot = None

    async def _start_service(self, slot: SlotName) -> None:
        svc = self._services[slot]
        if not svc.is_running():
            # Pre-spawn cumulative VRAM check. Reads ACTUAL free VRAM
            # from the GPU probe (not LlamaDeck's slot accounting, which is blind
            # to external processes like HFD-2's embedding model). If
            # the service estimate + headroom doesn't fit in real free
            # VRAM minus the external-reserved budget, refuse the spawn
            # and bubble a 503 to the queue waiters instead of OOMing
            # mid-load and crashing the child + corrupting the GPU.
            need_mb = int(getattr(svc, "est_vram_mb", 0) or 0)
            if need_mb > 0:
                try:
                    from .vram import offload_gpus as _offload_gpus
                    from .vram import probe_gpus as _probe_gpus
                    gpus = _offload_gpus(await _probe_gpus())
                except Exception as e:  # noqa: BLE001
                    log.warning("vram pre-spawn probe failed: %s", e)
                    gpus = []
                if gpus:
                    # probe_gpus() returns list[GpuInfo] (a dataclass) — use
                    # attribute access. Dict-style .get() raised AttributeError
                    # here, and since it was caught as "start(slot) failed",
                    # every pre-spawn VRAM check was effectively dead.
                    #
                    # "Free VRAM" is per card. Taking the emptiest card's free
                    # memory let a service pinned to the R9700 through while
                    # the R9700 had 0.5 GB left and the 5090 had 31 — the check
                    # passed and the spawn died out of memory anyway.
                    free_mb: int | None = None
                    pinned = [d for d in (getattr(svc, "target_devices", None) or []) if d]
                    if pinned:
                        try:
                            from .devices import device_budget_mb as _device_budget_mb
                            from .settings import load_settings as _load_settings
                            pair = await _device_budget_mb(
                                _load_settings().llama_bin or "", pinned
                            )
                        except Exception as e:  # noqa: BLE001
                            log.warning("vram pre-spawn device probe failed: %s", e)
                            pair = None
                        if pair is not None:
                            free_mb = pair[1]
                    if free_mb is None:
                        free_mb = max(int(getattr(g, "free_mb", 0) or 0) for g in gpus)
                    effective_free = free_mb - self._external_reserved_vram_mb
                    budget = need_mb + self._vram_spawn_headroom_mb
                    if budget > effective_free:
                        log.error(
                            "vram pre-spawn reject slot=%s need_mb=%d "
                            "headroom_mb=%d external_reserved_mb=%d "
                            "actual_free_mb=%d effective_free=%d",
                            slot, need_mb, self._vram_spawn_headroom_mb,
                            self._external_reserved_vram_mb, free_mb,
                            effective_free,
                        )
                        raise BrokerError(
                            "vram.insufficient_free",
                            f"slot '{slot}' needs ~{budget} MB "
                            f"(est {need_mb} + {self._vram_spawn_headroom_mb} headroom); "
                            f"only {effective_free} MB free "
                            f"(actual {free_mb} - {self._external_reserved_vram_mb} reserved). "
                            f"Release other GPU residents or wait for unload.",
                            status=503,
                        )
                    log.info(
                        "vram pre-spawn ok slot=%s need_mb=%d effective_free=%d",
                        slot, need_mb, effective_free,
                    )
            log.info("starting service slot=%s", slot)
            await svc.start()
        ok = await svc.wait_ready(timeout_s=120.0)
        if not ok:
            raise BrokerError("service.not_ready",
                              f"{slot} did not become ready in time", status=503)

    async def _watchdog_loop(self) -> None:
        """Expire leases past their TTL; expire queue waiters past deadline."""
        try:
            while not self._stopped:
                await asyncio.sleep(2.0)
                expired_leases: list[str] = []
                triggered_transition = False
                trans_target: SlotName | None = None
                trans_prev: SlotName | None = None
                async with self._lock:
                    now = time.time()
                    for lid, lease in self._active_leases.items():
                        if lease.expires_at < now:
                            expired_leases.append(lid)
                    for lid in expired_leases:
                        lease = self._active_leases.pop(lid, None)
                        if lease is not None:
                            log.warning("lease expired (no heartbeat) id=%s slot=%s holder=%s",
                                        lid, lease.slot, lease.holder)
                    survivors: list[_QueuedRequest] = []
                    for q in self._queue:
                        if q.deadline_at < now:
                            if not q.waiter.done():
                                q.waiter.set_exception(
                                    BrokerError("acquire.timeout",
                                                f"deadline elapsed for slot={q.slot}",
                                                status=408))
                        else:
                            survivors.append(q)
                    self._queue = survivors
                    if expired_leases and not self._active_leases and self._starting_slot is None:
                        if self._queue:
                            trans_target = self._queue[0].slot
                            if trans_target == self._active_slot:
                                self._grant_same_slot_waiters_locked(trans_target)
                            else:
                                trans_prev = self._active_slot
                                self._active_slot = None
                                self._starting_slot = trans_target
                                triggered_transition = True
                        else:
                            self._schedule_keepalive_locked()
                if triggered_transition and trans_target is not None:
                    self._spawn_start_and_grant(trans_target, trans_prev)
        except asyncio.CancelledError:
            return


# ---------------------------------------------------------------------------
# Singleton accessor (initialised at app startup)
# ---------------------------------------------------------------------------

_instance: GpuBroker | None = None


def init_broker(
    services: dict[SlotName, Service],
    *,
    keepalive_s: float = 300.0,
    default_ttl_s: float = 600.0,
    gpu_total_vram_mb: int | None = None,
    external_reserved_vram_mb: int = 0,
    vram_spawn_headroom_mb: int = 0,
) -> GpuBroker:
    global _instance
    if _instance is not None:
        raise RuntimeError("broker already initialised")
    _instance = GpuBroker(
        services,
        keepalive_s=keepalive_s,
        default_ttl_s=default_ttl_s,
        gpu_total_vram_mb=gpu_total_vram_mb,
        external_reserved_vram_mb=external_reserved_vram_mb,
        vram_spawn_headroom_mb=vram_spawn_headroom_mb,
    )
    return _instance


def get_broker() -> GpuBroker:
    if _instance is None:
        raise RuntimeError("broker not initialised — call init_broker() at startup")
    return _instance


__all__ = [
    "ALL_SLOTS", "SlotName",
    "Service", "Lease", "BrokerError", "GpuBroker",
    "init_broker", "get_broker",
]
