"""What the update check actually asks git for.

The Build page's "what has upstream landed since" runs one `git fetch` inside
one HTTP request. A plain fetch also pulls every new tag, and llama.cpp tags
every CI build — the checkout on a developer box carries ~7k of them and gains
dozens a day. That transfer is the whole cost of the check and none of it is
read here: on a checkout a few days stale it ran past the timeout, so the page
reported the check as unavailable on a machine whose network was fine, and left
killed-mid-download tmp_pack files in .git/objects/pack.

These tests run against two real local repositories, so they assert the
behaviour (a tag-only update is not transferred, a new commit is) rather than
the shape of an argv.
"""
from __future__ import annotations

import asyncio
import subprocess

import pytest

from lld.build import BuildManager


def git(repo, *args) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


@pytest.fixture()
def repos(tmp_path):
    """(upstream, clone) — a bare-ish origin and a checkout tracking its master."""
    up = tmp_path / "upstream"
    up.mkdir()
    git(up, "init", "--quiet", "--initial-branch=master")
    git(up, "config", "user.email", "t@t")
    git(up, "config", "user.name", "t")
    (up / "README").write_text("one\n")
    git(up, "add", "README")
    git(up, "commit", "--quiet", "-m", "one")

    clone = tmp_path / "clone"
    git(tmp_path, "clone", "--quiet", str(up), str(clone))
    git(clone, "config", "user.email", "t@t")
    git(clone, "config", "user.name", "t")
    return up, clone


def commits_ahead(mgr) -> dict:
    return asyncio.run(mgr.check_updates())


def test_a_new_upstream_commit_is_reported(repos):
    up, clone = repos
    (up / "README").write_text("two\n")
    git(up, "commit", "--quiet", "-am", "two: a real change")

    mgr = BuildManager(str(clone), str(clone / "build" / "bin" / "llama-server"))
    res = commits_ahead(mgr)

    assert res["branch"] == "master"
    assert res["ahead"] == 1
    assert res["commits"][0]["subject"] == "two: a real change"


def test_up_to_date_is_zero_not_an_error(repos):
    _, clone = repos
    mgr = BuildManager(str(clone), str(clone / "build" / "bin" / "llama-server"))
    res = commits_ahead(mgr)
    assert res["ahead"] == 0
    assert res["commits"] == []


def test_upstream_tags_are_not_fetched(repos):
    """The expensive half of a bare `git fetch`, and the check never reads it."""
    up, clone = repos
    for n in range(5):
        git(up, "tag", f"b{10600 + n}")

    mgr = BuildManager(str(clone), str(clone / "build" / "bin" / "llama-server"))
    commits_ahead(mgr)

    assert git(clone, "tag", "--list") == ""


def test_a_missing_checkout_says_so_instead_of_failing_as_git(tmp_path):
    from lld.build import BuildError

    mgr = BuildManager(str(tmp_path / "nope"), str(tmp_path / "nope" / "llama-server"))
    with pytest.raises(BuildError, match="source checkout not found"):
        commits_ahead(mgr)
