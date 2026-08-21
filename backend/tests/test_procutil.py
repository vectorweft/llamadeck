"""`run_capture` — the one place a short command is spawned and waited on.

The behaviour under test is the reason it exists: `asyncio.wait_for` cancels
the await, not the process, so every probe that used it directly leaked a live
child on timeout. The power poller runs at 2 Hz, which turns one hanging
`nvidia-smi` into a new orphan every half second.
"""
from __future__ import annotations

import asyncio
import sys

import psutil
import pytest

from lld.procutil import run_capture


async def test_captures_stdout_and_the_exit_code():
    res = await run_capture([sys.executable, "-c", "print('hello')"], timeout=10)
    assert res.rc == 0
    assert res.ok
    assert res.stdout.strip() == "hello"
    assert res.timed_out is False
    assert res.error is None


async def test_stderr_is_kept_separate_but_text_has_both():
    res = await run_capture(
        [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"],
        timeout=10,
    )
    assert res.stdout.strip() == "out"
    assert res.stderr.strip() == "err"
    assert set(res.text.split()) == {"out", "err"}


async def test_a_nonzero_exit_is_not_ok_but_still_reports_output():
    res = await run_capture(
        [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"],
        timeout=10,
    )
    assert res.rc == 3
    assert not res.ok
    assert res.stderr == "boom"


async def test_a_missing_binary_reports_why_instead_of_raising():
    res = await run_capture(["definitely-not-a-real-binary-9f3a"], timeout=5)
    assert res.rc is None
    assert res.timed_out is False
    assert res.error


async def test_a_timeout_kills_the_child_rather_than_leaving_it_running():
    code = "import time; time.sleep(120)"
    before = {p.pid for p in psutil.process_iter(["pid"])}
    res = await run_capture([sys.executable, "-c", code], timeout=0.4)
    assert res.timed_out is True
    assert res.rc is None
    # Give the kill a moment to be reaped, then assert nothing new is sleeping.
    await asyncio.sleep(0.3)
    leaked = []
    for p in psutil.process_iter(["pid", "cmdline", "status"]):
        if p.pid in before:
            continue
        try:
            if code in (p.info["cmdline"] or []) and p.info["status"] != psutil.STATUS_ZOMBIE:
                leaked.append(p.pid)
        except psutil.Error:
            continue
    assert leaked == [], f"run_capture left {leaked} behind after its timeout"


async def test_cancelling_the_caller_also_kills_the_child():
    """A shutdown cancels whatever the poller was awaiting. The process it
    spawned must not outlive that."""
    code = "import time; time.sleep(120)"
    task = asyncio.create_task(run_capture([sys.executable, "-c", code], timeout=60))
    await asyncio.sleep(0.4)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.3)
    alive = [
        p.pid
        for p in psutil.process_iter(["pid", "cmdline", "status"])
        if code in (p.info["cmdline"] or []) and p.info["status"] != psutil.STATUS_ZOMBIE
    ]
    assert alive == []
