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
