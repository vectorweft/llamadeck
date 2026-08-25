"""A second LlamaDeck must not adopt the first one's llama-servers.

Boot adopts every llama-server `scan_existing()` returns, unconditionally. A
separate state directory does not isolate you from another instance's processes,
so a second LlamaDeck on the same host took over the first one's router: both
polled it at 2 Hz, both showed it as theirs, and either watchdog could restart a
process the other was serving from. It happened to a sandbox instance run beside
a live one.

The rule pinned here: a llama-server whose parent is a live LlamaDeck belongs to
that instance. Our own children, and children orphaned by a restart (reparented
to init), stay adoptable — that is what adoption is for.
"""
from __future__ import annotations

import os

import psutil
import pytest

from lld.supervisor import _looks_like_llamadeck, owning_llamadeck


class _FakeProc:
    """Minimal psutil.Process stand-in: only what owning_llamadeck touches."""

    def __init__(self, pid: int, cmdline: list[str], parent: "_FakeProc | None" = None,
                 running: bool = True, raises: type[Exception] | None = None):
        self.pid = pid
        self._cmdline = cmdline
        self._parent = parent
        self._running = running
        self._raises = raises

    def parent(self):
        if self._raises:
            raise self._raises(self.pid)
        return self._parent

    def is_running(self):
        return self._running

    def cmdline(self):
        return self._cmdline


SERVER = ["/opt/llama.cpp/build/bin/llama-server", "--port", "8085"]


@pytest.mark.parametrize("cmdline", [
    ["/srv/app/.venv/bin/python", "/srv/app/.venv/bin/llamadeck", "serve"],
    ["/usr/bin/uv", "run", "llamadeck", "serve", "--port", "8770"],
    ["/usr/bin/python3", "-m", "lld.cli", "serve"],
    ["/srv/app/.venv/bin/uvicorn", "lld.main:app"],
    ["/srv/app/.venv/bin/lld", "serve"],
])
def test_recognises_a_llamadeck_command_line(cmdline):
    assert _looks_like_llamadeck(cmdline)


@pytest.mark.parametrize("cmdline", [
    ["/opt/llama.cpp/build/bin/llama-server", "--port", "8085"],
    ["/bin/bash"],
    # A path that merely contains the name must not count — the state dir is
    # ~/.config/llamadeck on every install, and it shows up as a flag value.
    ["/opt/llama.cpp/bin/llama-server", "--models-preset",
     "/home/u/.config/llamadeck/router-models.ini"],
    [],
])
def test_does_not_mistake_other_processes_for_llamadeck(cmdline):
    assert not _looks_like_llamadeck(cmdline)


def test_another_instances_child_has_an_owner():
    other = _FakeProc(4242, ["/srv/app/.venv/bin/python", "/srv/app/.venv/bin/llamadeck", "serve"])
    server = _FakeProc(5000, SERVER, parent=other)
    owner = owning_llamadeck(server)
    assert owner is not None and owner.pid == 4242


def test_our_own_child_is_not_someone_elses():
    """Its parent is this process. Adopting it back is the whole point."""
    us = _FakeProc(os.getpid(), ["/srv/app/.venv/bin/llamadeck", "serve"])
    assert owning_llamadeck(_FakeProc(5001, SERVER, parent=us)) is None


def test_an_orphan_is_free_to_adopt():
    """A restart reparents children to init. Nobody owns them any more."""
    init = _FakeProc(1, ["/sbin/init"])
    assert owning_llamadeck(_FakeProc(5002, SERVER, parent=init)) is None


def test_a_hand_started_server_is_free_to_adopt():
    shell = _FakeProc(900, ["/bin/bash"])
    assert owning_llamadeck(_FakeProc(5003, SERVER, parent=shell)) is None


def test_a_dead_parent_does_not_own_anything():
    stale = _FakeProc(4243, ["/srv/app/.venv/bin/llamadeck", "serve"], running=False)
    assert owning_llamadeck(_FakeProc(5004, SERVER, parent=stale)) is None


@pytest.mark.parametrize("exc", [psutil.NoSuchProcess, psutil.AccessDenied])
def test_an_unreadable_parent_leaves_the_process_adoptable(exc):
    """Erring toward "no owner" keeps the pre-existing behaviour for processes we
    cannot inspect — a permissions quirk must not make adoption stop working."""
    assert owning_llamadeck(_FakeProc(5005, SERVER, raises=exc)) is None
