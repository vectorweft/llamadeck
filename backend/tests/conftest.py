"""Test-suite-wide guard: the real user state directory is off limits.

`lld.settings` resolves STATE_DIR at import time, so a fixture cannot move it
after the fact — by then the module constants are already bound. conftest.py is
imported before any test module, which makes this the one place early enough to
redirect it. Without it, any test that reaches load_settings() or
ensure_state_dirs() without patching writes into ~/.config/llamadeck: a fresh
settings.json with defaults, on the developer's own machine, silently replacing
the state a running install is about to migrate into.
"""
from __future__ import annotations

import os
import tempfile

_TEST_STATE_DIR = tempfile.mkdtemp(prefix="llamadeck-test-state-")
os.environ["LLAMADECK_STATE_DIR"] = _TEST_STATE_DIR
# The launcher and a few helpers read XDG directly; point those at the same
# throwaway root so nothing escapes into the real config either.
os.environ.setdefault("XDG_CONFIG_HOME", os.path.join(_TEST_STATE_DIR, "config"))
