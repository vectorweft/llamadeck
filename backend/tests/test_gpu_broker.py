"""Unit tests for GpuBroker — service behavior is mocked."""
from __future__ import annotations

import asyncio
import logging

import pytest

from lld.gpu_broker import BrokerError, GpuBroker

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture(autouse=True)
def fake_gpu(monkeypatch):
    """The broker's pre-spawn check probes real hardware via nvidia-smi. These
    tests assert scheduling behavior, not this machine's free VRAM, so pin a
    generous fake GPU — otherwise results depend on whatever happens to be
    resident on the developer's card."""
    from dataclasses import dataclass

    @dataclass
    class _FakeGpu:
        index: int = 0
        name: str = "FakeGPU"
        total_mb: int = 48000
        used_mb: int = 0
        free_mb: int = 48000
        # `offload_gpus()` reads these. Without them the probe raised
        # AttributeError, the broker logged "pre-spawn probe failed" and
        # skipped the check — so the very thing this fixture exists to pin
        # was never exercised.
        vendor: str = "nvidia"
        unified: bool = False
        integrated: bool = False

    async def _fake_probe():
        return [_FakeGpu()]

    monkeypatch.setattr("lld.vram.probe_gpus", _fake_probe)


class FakeService:
    def __init__(self, name: str, *, est_vram_mb: int = 1000, ready_delay_s: float = 0.0):
        self.name = name
        self.managed_url = f"http://127.0.0.1:9000/{name}"
        self.est_vram_mb = est_vram_mb
        self._ready_delay = ready_delay_s
        self._running = False
        self.start_calls = 0
        self.stop_calls = 0

    async def start(self) -> None:
        self.start_calls += 1
        await asyncio.sleep(0)
        self._running = True

    async def stop(self) -> None:
        self.stop_calls += 1
        await asyncio.sleep(0)
        self._running = False

    async def wait_ready(self, timeout_s: float = 10.0) -> bool:
        if self._ready_delay:
            await asyncio.sleep(self._ready_delay)
        return self._running

    def is_running(self) -> bool:
        return self._running


def _services():
    return {
        "llm": FakeService("llm", est_vram_mb=12000),
        "comfy": FakeService("comfy", est_vram_mb=11000),
        "tts": FakeService("tts", est_vram_mb=4000),
    }


@pytest.mark.asyncio
async def test_acquire_idle_grants_immediately():
    services = _services()
    broker = GpuBroker(services, keepalive_s=0.1, default_ttl_s=10.0)
    await broker.start()
    try:
        lease = await broker.acquire(slot="llm", holder="t.test")
        assert lease.slot == "llm"
        assert services["llm"].start_calls == 1
        assert broker.active_slot() == "llm"
        await broker.release(lease.lease_id)
    finally:
        await broker.stop()


@pytest.mark.asyncio
async def test_same_slot_multiple_leases():
    services = _services()
    broker = GpuBroker(services, keepalive_s=0.1, default_ttl_s=10.0)
    await broker.start()
    try:
        l1 = await broker.acquire(slot="llm", holder="a")
        l2 = await broker.acquire(slot="llm", holder="b")
        assert l1.slot == l2.slot == "llm"
        # same slot, second start NOT called
        assert services["llm"].start_calls == 1
        await broker.release(l1.lease_id)
        await broker.release(l2.lease_id)
    finally:
        await broker.stop()


@pytest.mark.asyncio
async def test_different_slot_queues_then_swaps():
    services = _services()
    broker = GpuBroker(services, keepalive_s=0.0, default_ttl_s=10.0)
    await broker.start()
    try:
        l1 = await broker.acquire(slot="llm", holder="a")

        async def acquire_comfy():
            return await broker.acquire(slot="comfy", holder="b", wait_timeout_s=5.0)

        task = asyncio.create_task(acquire_comfy())
        await asyncio.sleep(0.05)
        # comfy is queued; not yet started
        assert services["comfy"].start_calls == 0
        await broker.release(l1.lease_id)
        l2 = await asyncio.wait_for(task, timeout=2.0)
        assert l2.slot == "comfy"
        assert services["llm"].stop_calls == 1
        assert services["comfy"].start_calls == 1
        await broker.release(l2.lease_id)
    finally:
        await broker.stop()


@pytest.mark.asyncio
async def test_acquire_timeout_when_blocked():
    services = _services()
    broker = GpuBroker(services, keepalive_s=600.0, default_ttl_s=10.0)
    await broker.start()
    try:
        l1 = await broker.acquire(slot="llm", holder="a")
        with pytest.raises(BrokerError) as ei:
            await broker.acquire(slot="comfy", holder="b", wait_timeout_s=0.5)
        assert ei.value.code == "acquire.timeout"
        await broker.release(l1.lease_id)
    finally:
        await broker.stop()


@pytest.mark.asyncio
async def test_keepalive_stops_idle_service():
    services = _services()
    broker = GpuBroker(services, keepalive_s=0.2, default_ttl_s=10.0)
    await broker.start()
    try:
        lease = await broker.acquire(slot="llm", holder="a")
        await broker.release(lease.lease_id)
        # Wait past keepalive
        await asyncio.sleep(0.5)
        assert services["llm"].stop_calls == 1
        assert broker.active_slot() is None
    finally:
        await broker.stop()


@pytest.mark.asyncio
async def test_keepalive_zero_stops_immediately():
    services = _services()
    broker = GpuBroker(services, keepalive_s=0.0, default_ttl_s=10.0)
    await broker.start()
    try:
        lease = await broker.acquire(slot="llm", holder="a")
        await broker.release(lease.lease_id)
        await asyncio.sleep(0.1)
        assert services["llm"].stop_calls == 1
        assert broker.active_slot() is None
    finally:
        await broker.stop()


@pytest.mark.asyncio
async def test_heartbeat_extends_ttl():
    services = _services()
    broker = GpuBroker(services, keepalive_s=0.0, default_ttl_s=1.0)
    await broker.start()
    try:
        lease = await broker.acquire(slot="llm", holder="a", est_duration_s=0.5)
        before = lease.expires_at
        await asyncio.sleep(0.05)
        new_lease = await broker.heartbeat(lease.lease_id, ttl_s=10.0)
        assert new_lease.expires_at > before
        await broker.release(lease.lease_id)
    finally:
        await broker.stop()


@pytest.mark.asyncio
async def test_vram_overflow_rejected():
    services = _services()
    broker = GpuBroker(services, keepalive_s=0.0, default_ttl_s=10.0,
                       gpu_total_vram_mb=8000)
    await broker.start()
    try:
        with pytest.raises(BrokerError) as ei:
            await broker.acquire(slot="llm", holder="a", est_vram_mb=20000)
        assert ei.value.code == "vram.exceeds_capacity"
    finally:
        await broker.stop()


@pytest.mark.asyncio
async def test_release_unknown_lease_404():
    services = _services()
    broker = GpuBroker(services)
    await broker.start()
    try:
        with pytest.raises(BrokerError) as ei:
            await broker.release("ghost")
        assert ei.value.code == "lease.not_found"
        assert ei.value.status == 404
    finally:
        await broker.stop()


@pytest.mark.asyncio
async def test_lease_expires_without_heartbeat():
    services = _services()
    # Watchdog runs every 2s; default_ttl floor wins over est_duration*1.2
    broker = GpuBroker(services, keepalive_s=0.0, default_ttl_s=0.5)
    await broker.start()
    try:
        lease = await broker.acquire(slot="llm", holder="a", est_duration_s=0.1)
        # default_ttl 0.5 wins; wait > 0.5 + watchdog 2s
        await asyncio.sleep(3.0)
        # Lease expired and removed
        with pytest.raises(BrokerError):
            await broker.release(lease.lease_id)
    finally:
        await broker.stop()


# ---------------------------------------------------------------------------
# Regressions for the faz-1.5 robustness sweep
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_slow_start_does_not_block_unrelated_state_or_release():
    """B2 regression: a slot whose start takes 2s must NOT block state()
    reads or release() calls for other leases. Prior implementation held
    `self._lock` across the entire `wait_ready`, freezing the broker."""
    services = _services()
    services["llm"] = FakeService("llm", est_vram_mb=12000, ready_delay_s=2.0)
    broker = GpuBroker(services, keepalive_s=0.0, default_ttl_s=10.0)
    await broker.start()
    try:
        acquire_task = asyncio.create_task(
            broker.acquire(slot="llm", holder="slow", wait_timeout_s=10.0)
        )
        # Give the background _start_and_grant a chance to begin.
        await asyncio.sleep(0.05)
        # state() must respond immediately even mid-start.
        snap_before = broker.state()
        assert snap_before["starting_slot"] == "llm"
        assert snap_before["active_slot"] is None
        # Even unrelated release on a non-existent lease must not block.
        with pytest.raises(BrokerError):
            await asyncio.wait_for(broker.release("ghost"), timeout=0.2)
        lease = await asyncio.wait_for(acquire_task, timeout=5.0)
        assert lease.slot == "llm"
        await broker.release(lease.lease_id)
    finally:
        await broker.stop()


class FlakyService(FakeService):
    """Service whose start() raises N times before succeeding (or forever)."""

    def __init__(self, name: str, *, fail_count: int, **kw):
        super().__init__(name, **kw)
        self._fail_remaining = fail_count

    async def start(self) -> None:
        self.start_calls += 1
        await asyncio.sleep(0)
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise RuntimeError(f"simulated start failure ({self._fail_remaining} left)")
        self._running = True


@pytest.mark.asyncio
async def test_failed_start_propagates_to_waiter():
    """First-line of B4 regression: a start failure raises BrokerError to
    the acquiring caller (no infinite hang, no silent grant)."""
    services = _services()
    services["llm"] = FlakyService("llm", est_vram_mb=12000, fail_count=99)
    broker = GpuBroker(services, keepalive_s=0.0, default_ttl_s=10.0)
    await broker.start()
    try:
        with pytest.raises(BrokerError) as ei:
            await broker.acquire(slot="llm", holder="x", wait_timeout_s=2.0)
        assert ei.value.code == "service.start_failed"
        assert ei.value.status == 503
    finally:
        await broker.stop()


@pytest.mark.asyncio
async def test_repeated_failures_trigger_cooldown():
    """B4 regression: 3 failures in 60s → next acquire returns 503
    service.unhealthy with a cooldown, aligned with HFD-2's 5s/10s 503-retry."""
    services = _services()
    services["llm"] = FlakyService("llm", est_vram_mb=12000, fail_count=99)
    broker = GpuBroker(services, keepalive_s=0.0, default_ttl_s=10.0)
    await broker.start()
    try:
        for _ in range(3):
            with pytest.raises(BrokerError):
                await broker.acquire(slot="llm", holder="x", wait_timeout_s=1.0)
        with pytest.raises(BrokerError) as ei:
            await broker.acquire(slot="llm", holder="x", wait_timeout_s=1.0)
        assert ei.value.code == "service.unhealthy"
        assert ei.value.status == 503
        # Other slots must remain unaffected.
        lease = await broker.acquire(slot="comfy", holder="y", wait_timeout_s=1.0)
        await broker.release(lease.lease_id)
    finally:
        await broker.stop()


@pytest.mark.asyncio
async def test_supervisor_change_invalidates_llm_leases():
    """B5 regression: notify_supervisor_changed() expires existing 'llm'
    leases when the underlying service has been stopped externally."""
    services = _services()
    broker = GpuBroker(services, keepalive_s=600.0, default_ttl_s=10.0)
    await broker.start()
    try:
        lease = await broker.acquire(slot="llm", holder="a")
        # Simulate user stopping the LLM externally (e.g. UI Stop button).
        services["llm"]._running = False
        await broker.notify_supervisor_changed()
        # Lease is gone, slot is idle.
        with pytest.raises(BrokerError):
            await broker.release(lease.lease_id)
        assert broker.active_slot() is None
    finally:
        await broker.stop()


@pytest.mark.asyncio
async def test_state_exposes_starting_slot_and_failures():
    """B-introspection: state() reports starting_slot during a slow start
    and recent_start_failures after a flaky start."""
    services = _services()
    services["llm"] = FlakyService("llm", est_vram_mb=12000, fail_count=2,
                                    ready_delay_s=0.0)
    broker = GpuBroker(services, keepalive_s=0.0, default_ttl_s=10.0)
    await broker.start()
    try:
        with pytest.raises(BrokerError):
            await broker.acquire(slot="llm", holder="x", wait_timeout_s=1.0)
        snap = broker.state()
        assert "llm" in snap["recent_start_failures"]
        assert len(snap["recent_start_failures"]["llm"]) >= 1
    finally:
        await broker.stop()


@pytest.mark.asyncio
async def test_pre_spawn_checks_the_pinned_card_not_the_emptiest(monkeypatch):
    """Two GPUs, and only one of them is the target.

    The check used the emptiest card in the box, so a service pinned to the
    R9700 (0.5 GB free) sailed past it on the strength of the 5090's 31 GB —
    and then died out of memory during load, which is exactly the failure this
    check exists to prevent.
    """
    services = _services()
    services["llm"] = FakeService("llm", est_vram_mb=20000)
    services["llm"].target_devices = ["Vulkan1"]

    async def fake_budget(_binary, ids):
        assert ids == ["Vulkan1"]
        return (32000, 500)          # the R9700: nearly full

    monkeypatch.setattr("lld.devices.device_budget_mb", fake_budget)
    broker = GpuBroker(services, keepalive_s=0.0, default_ttl_s=10.0,
                       vram_spawn_headroom_mb=2048)
    await broker.start()
    try:
        with pytest.raises(BrokerError) as e:
            await broker.acquire(slot="llm", holder="t.test")
        # The spawn is refused against the PINNED card's 500 MB, not the
        # fixture GPU's 48 GB. (The broker re-wraps the refusal as
        # service.start_failed, so the numbers are what identify it.)
        assert "only 500 MB free" in str(e.value)
        assert services["llm"].start_calls == 0
    finally:
        await broker.stop()


@pytest.mark.asyncio
async def test_unpinned_service_still_uses_the_emptiest_card(monkeypatch):
    """No pin means llama.cpp chooses, and the roomiest card is the honest
    guess — behaviour unchanged for every single-GPU box."""
    services = _services()
    services["llm"] = FakeService("llm", est_vram_mb=20000)

    async def fail_if_called(_binary, _ids):  # pragma: no cover - must not run
        raise AssertionError("device budget probed for an unpinned service")

    monkeypatch.setattr("lld.devices.device_budget_mb", fail_if_called)
    broker = GpuBroker(services, keepalive_s=0.0, default_ttl_s=10.0)
    await broker.start()
    try:
        lease = await broker.acquire(slot="llm", holder="t.test")
        assert lease.slot == "llm"
    finally:
        await broker.stop()
