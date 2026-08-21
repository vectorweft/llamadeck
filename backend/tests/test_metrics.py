from __future__ import annotations

import pytest

from lld import metrics as metrics_mod
from lld.metrics import MetricsService, parse_prometheus


def test_parse_prometheus_llamacpp():
    sample = """
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 2.66759e+06
llamacpp:tokens_predicted_total 783284
llamacpp:prompt_tokens_seconds 10221.5
llamacpp:requests_processing 0
llamacpp:n_busy_slots_per_decode 1.00123
"""
    m = parse_prometheus(sample)
    assert m["llamacpp:prompt_tokens_total"] == 2667590.0
    assert m["llamacpp:tokens_predicted_total"] == 783284.0
    assert m["llamacpp:prompt_tokens_seconds"] == 10221.5
    assert m["llamacpp:requests_processing"] == 0.0
    assert round(m["llamacpp:n_busy_slots_per_decode"], 5) == 1.00123


def test_parse_prometheus_ignores_comments_and_empty():
    sample = "\n# comment\n# HELP foo bar\n\nfoo_metric 42\n"
    m = parse_prometheus(sample)
    assert m == {"foo_metric": 42.0}


# --------------------------------------------------------------------------
# Live prompt-processing rate
#
# `llamacpp:prompt_tokens_total` only moves when a request FINISHES, so a
# dashboard reading it alone shows 0 tok/s for the whole minute a 160k-token
# prefill is running and then a one-frame spike. The rate is taken from the
# slot's own progress counter instead; these pin that down.
# --------------------------------------------------------------------------

class _Resp:
    def __init__(self, payload, text=""):
        self.status_code = 200
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def _service(monkeypatch, frames_of_slots, clock_step=0.5):
    """A MetricsService whose /slots answers come from a scripted list."""
    svc = MetricsService(supervisor=None)  # type: ignore[arg-type]
    queue = list(frames_of_slots)
    now = [1000.0]

    class _Client:
        async def get(self, url, params=None):
            if url.endswith("/slots"):
                return _Resp(queue.pop(0))
            return _Resp({}, text="llamacpp:prompt_tokens_total 0\n")

    svc._client = _Client()  # type: ignore[assignment]
    monkeypatch.setattr(metrics_mod.time, "time", lambda: now[0])
    return svc, now, clock_step


def _slot(*, task=1, processed=0, decoded=0, processing=True):
    return {
        "id": 0, "n_ctx": 4096, "is_processing": processing, "id_task": task,
        "n_prompt_tokens": 8192, "n_prompt_tokens_processed": processed,
        "n_prompt_tokens_cache": 0,
        "next_token": [{"n_decoded": decoded, "n_remain": 0, "has_next_token": True}],
        "params": {},
    }


STATUS = {"config": {"port": 9999, "host": "127.0.0.1"}}


async def test_prompt_rate_ticks_while_the_prefill_is_running(monkeypatch):
    """2048 more prompt tokens in half a second is 4096 tok/s, live."""
    svc, now, step = _service(monkeypatch, [
        [_slot(processed=0)],
        [_slot(processed=2048)],
    ])
    await svc._poll_preset("p", STATUS)
    now[0] += step
    await svc._poll_preset("p", STATUS)
    assert svc.state("p").latest().instant_prompt_tps == pytest.approx(4096.0)


async def test_prompt_rate_is_zero_once_the_slot_is_generating(monkeypatch):
    """Past the prompt, nothing is being ingested — 0 is the honest reading."""
    svc, now, step = _service(monkeypatch, [
        [_slot(processed=2048, decoded=10)],
        [_slot(processed=2048, decoded=40)],
    ])
    await svc._poll_preset("p", STATUS)
    now[0] += step
    await svc._poll_preset("p", STATUS)
    assert svc.state("p").latest().instant_prompt_tps == 0.0


async def test_a_new_task_does_not_fabricate_a_spike(monkeypatch):
    """The counter restarts per request; the reset is not throughput.

    Without the task_id guard, the second frame's 3000 would read as a delta
    against the previous task's 2048 (or worse, against zero) and paint a
    rate that never happened.
    """
    svc, now, step = _service(monkeypatch, [
        [_slot(task=1, processed=2048, decoded=99)],
        [_slot(task=2, processed=3000, decoded=0)],
    ])
    await svc._poll_preset("p", STATUS)
    now[0] += step
    await svc._poll_preset("p", STATUS)
    # No same-task slot to diff, and a busy boundary frame → no signal at all,
    # which the rolling average skips rather than averaging in.
    assert svc.state("p").latest().instant_prompt_tps is None


async def test_idle_to_idle_falls_back_to_the_counter(monkeypatch):
    """A request that began and ended between two polls still shows up."""
    svc, now, step = _service(monkeypatch, [])

    texts = iter([
        "llamacpp:prompt_tokens_total 0\n",
        "llamacpp:prompt_tokens_total 512\n",
    ])

    class _Client:
        def __init__(self, slots):
            self._slots = list(slots)

        async def get(self, url, params=None):
            if url.endswith("/slots"):
                return _Resp(self._slots.pop(0))
            return _Resp({}, text=next(texts))

    svc._client = _Client([[_slot(processing=False)], [_slot(processing=False)]])  # type: ignore[assignment]
    await svc._poll_preset("p", STATUS)
    now[0] += step
    await svc._poll_preset("p", STATUS)
    assert svc.state("p").latest().instant_prompt_tps == pytest.approx(1024.0)


async def test_slot_prompt_fields_reach_the_frame(monkeypatch):
    """The UI shows cache reuse as a percentage; it needs both numbers."""
    svc, now, _ = _service(monkeypatch, [[
        {**_slot(processed=90), "n_prompt_tokens": 160711, "n_prompt_tokens_cache": 160621},
    ]])
    await svc._poll_preset("p", STATUS)
    slot = svc.state("p").latest().to_dict()["slots"][0]
    assert slot["n_prompt_tokens"] == 160711
    assert slot["n_prompt_tokens_cache"] == 160621
    assert slot["n_prompt_tokens_processed"] == 90
