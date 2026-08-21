"""Parsing the `llama-server --version` banner.

Getting this wrong is silent: `capture_snapshot` bails when there is no commit,
so What's New never gets a baseline and no rebuild ever produces a feature
diff. The banner changed shape upstream and the old pattern only matched the
retired one, which is exactly how that happened.
"""
from __future__ import annotations

from lld.build import parse_version_banner

# What a current build prints (llama.cpp b10449, August 2026).
CURRENT = """WARNING: radv is not a conformant Vulkan implementation, testing use only.
version: 0.1.0-dev (build 10449, commit 0d9ceae1e)
built with GNU 15.2.0 for Linux x86_64
"""

# What builds through mid-2025 printed.
LEGACY = """version: 4589 (a1b2c3d4)
built with cc (Ubuntu 13.2.0) for x86_64-linux-gnu
"""


def test_current_banner():
    assert parse_version_banner(CURRENT) == (10449, "0d9ceae1e")


def test_legacy_banner():
    assert parse_version_banner(LEGACY) == (4589, "a1b2c3d4")


def test_a_driver_warning_ahead_of_the_banner_does_not_hide_it():
    assert parse_version_banner("noise\nmore noise\n" + CURRENT)[1] == "0d9ceae1e"


def test_a_full_length_sha_is_accepted():
    banner = "version: 0.1.0-dev (build 10449, commit 0d9ceae1e2f4b6c8d0a1e3f5a7b9c1d3e5f7a9b1)"
    assert parse_version_banner(banner) == (
        10449, "0d9ceae1e2f4b6c8d0a1e3f5a7b9c1d3e5f7a9b1",
    )


def test_nothing_recognisable_yields_nothing():
    assert parse_version_banner("error while loading shared libraries") == (None, None)
    assert parse_version_banner("") == (None, None)
