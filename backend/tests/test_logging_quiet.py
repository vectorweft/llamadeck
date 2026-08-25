"""First-run terminal must not flood with httpx transport logs.

`main.configure_logging()` raises the ROOT logger to INFO so LlamaDeck's own
startup phases are visible. httpx's 'httpx'/'httpcore' loggers default to
NOTSET and therefore inherit that INFO level, so every internal HTTP request —
the 2 Hz metrics poller (GET /models + /slots + /metrics per running preset),
the health watchdog, the GPU probe — gets logged as
"HTTP Request: GET http://127.0.0.1:8085/models". With a preset up (and on a
fresh install where the seeded router auto-starts at boot with no model loaded)
the terminal scrolls the same line forever, reading as an infinite loop hunting
for a loaded model. This test pins the fix: httpx stays at WARNING while `lld`
keeps INFO.
"""
from __future__ import annotations

import logging

from lld import main as main_mod


def test_configure_logging_quiets_httpx_but_keeps_lld():
    # Under pytest the root logger already has handlers, so basicConfig(INFO)
    # is a no-op and root stays at WARNING. In the real app basicConfig sets
    # root to INFO, so replicate that to test the actual runtime condition.
    logging.getLogger().setLevel(logging.INFO)
    main_mod.configure_logging()

    # httpx transports must not inherit the root INFO level and log every
    # request; WARNING still surfaces real transport failures.
    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING
    assert logging.getLogger("httpcore").getEffectiveLevel() >= logging.WARNING

    # LlamaDeck's own logger stays at INFO so boot phases remain visible.
    assert logging.getLogger("lld").getEffectiveLevel() <= logging.INFO


def _access_record(path: str, status: int) -> logging.LogRecord:
    """A record shaped the way uvicorn.access emits one."""
    return logging.LogRecord(
        name="uvicorn.access", level=logging.INFO, pathname=__file__, lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:1234", "GET", path, "1.1", status),
        exc_info=None,
    )


def test_heartbeat_access_lines_are_filtered_but_real_traffic_is_not():
    f = main_mod._HeartbeatAccessFilter()

    # The dashboard's own timer polls: ~1 line/s forever, from the app talking
    # to itself. Dropped.
    assert not f.filter(_access_record("/api/server/vram", 200))
    assert not f.filter(_access_record("/api/server/statuses", 200))
    assert not f.filter(_access_record("/health", 200))
    assert not f.filter(_access_record("/api/server/status/my-preset", 200))

    # Anything a person did stays in the log.
    assert f.filter(_access_record("/api/presets", 200))
    assert f.filter(_access_record("/api/setup/build", 200))

    # A heartbeat that starts failing is exactly when its line matters.
    assert f.filter(_access_record("/api/server/vram", 500))
    assert f.filter(_access_record("/health", 404))


def test_heartbeat_filter_passes_records_it_does_not_understand():
    """A record that isn't uvicorn's 5-tuple shape must not be swallowed."""
    f = main_mod._HeartbeatAccessFilter()
    odd = logging.LogRecord(
        name="uvicorn.access", level=logging.INFO, pathname=__file__, lineno=1,
        msg="something else entirely", args=None, exc_info=None,
    )
    assert f.filter(odd)
