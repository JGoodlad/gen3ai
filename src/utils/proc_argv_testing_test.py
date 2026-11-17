"""``utils.proc_argv_testing``: the stand-in shows its argv the moment the helper returns, and a broken fixture is LOUD.

``integration``: spawns (and kills, by its own handle) real ``sleep`` processes.
"""
from __future__ import annotations

import subprocess

import pytest

from utils import procfs
from utils.proc_argv_testing import FixtureError, spawn_sleeper, wait_argv

pytestmark = pytest.mark.integration


def _reap(p: subprocess.Popen) -> None:
    if p.poll() is None:
        p.kill()
    p.wait(timeout=10)


def test_a_sleeper_shows_its_argv_the_instant_the_helper_returns():
    """Every call: the cmdline is the full argv, never '' (the exec window) and never a shell's."""
    for _ in range(200):
        p = spawn_sleeper(30, name="python -m main.launcher --run-name x")
        try:
            assert procfs.cmdline(p.pid) == "python -m main.launcher --run-name x 30"
        finally:
            _reap(p)


def test_an_unnamed_sleeper_is_a_plain_sleep():
    p = spawn_sleeper(37)
    try:
        assert procfs.cmdline(p.pid) == "sleep 37"
    finally:
        _reap(p)


def test_an_argv_that_never_appears_fails_the_fixture_loudly_and_reaps_the_child():
    p = subprocess.Popen(["sleep", "30"])
    with pytest.raises(FixtureError, match=r"did not show its argv within 0\.3 s.*last /proc cmdline 'sleep 30'"):
        wait_argv(p, equals="never this", timeout_s=0.3)
    assert p.poll() is not None                      # killed and reaped, never leaked


def test_a_child_that_already_exited_fails_the_fixture_at_once():
    p = subprocess.Popen(["true"])
    p.wait()
    with pytest.raises(FixtureError, match="exited"):
        wait_argv(p, contains="true", timeout_s=30)
