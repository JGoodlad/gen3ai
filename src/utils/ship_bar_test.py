"""THE 8 s BAR (``utils.ship_bar``): the decision, and the root conftest's wiring in a real child session.

Load-bearing pins, each FAILS on revert of what it guards:
* the decision bands — under the bar ok, the margin band WARN, >= 2x the stretched bar FAIL, an
  unreadable window never FAIL, a busier box stretches the bar and a quiet one never shrinks it;
* the WIRING: a child pytest session over the real root conftest, with the bar on, FAILS (exit 1) on a
  routine test far over a (shrunk) bar, names a near one as WARN, and exempts ``slow`` / ``linchpin``;
  the same session with the bar off exits 0.

``integration``: one child pytest session per arm, ~2 s each.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

from utils import ship_bar as SB
from utils.contention import scale_timeout
from utils.paths import repo_root, src_root


@pytest.mark.parametrize("total,factor,want", [
    (7.9, 1.0, "ok"),            # under the bar: never read
    (7.9, None, "ok"),
    (8.0, 1.0, "warn"),          # at the bar, quiet box: the margin band
    (15.9, 2.7, "warn"),         # just under 2x at the reference contention: still the margin
    (16.0, 2.7, "fail"),         # 2x the bar at the reference contention
    (16.0, 1.0, "fail"),         # a quiet box never SHRINKS the bar below 8 s ...
    (15.9, 5.4, "ok"),           # ... and a box 2x busier than the reference doubles it
    (31.9, 5.4, "warn"),
    (32.0, 5.4, "fail"),
    (500.0, None, "warn"),       # an unreadable window is never a verdict
])
def test_the_decision_bands(total, factor, want, monkeypatch):
    monkeypatch.delenv(SB.BAR_ENV, raising=False)
    assert SB.judge(total, factor) == want


_PLANTED = '''\
import time, pytest

def test_fast():
    pass

def test_drifted():
    time.sleep(0.6)

def test_near():
    time.sleep(0.3)

@pytest.mark.slow
def test_marked_slow():
    time.sleep(0.6)

@pytest.mark.linchpin
def test_a_linchpin():
    time.sleep(0.6)
'''


def _session(tmp_path, *, bar_on: bool):
    shutil.copy(repo_root() / "conftest.py", tmp_path / "conftest.py")
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n    slow: slow\n    linchpin: linchpin\n")
    (tmp_path / "planted_test.py").write_text(_PLANTED)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTEST_", "GEN3AI_SHIP_BAR"))}
    env.update(GEN3AI_SKIP_DEPS_GUARD="1", PYTHONPATH=str(src_root()), GEN3AI_COST_SCHEDULE="0",
               GEN3AI_SLOW_STATUS_FILE=str(tmp_path / "status.json"), GEN3AI_SKIP_TIER_BUDGET="1",
               GEN3AI_TIMEOUT_SCALE="1",            # the contention reading is pinned at x1: a fixed factor
               GEN3AI_SHIP_BAR_S="0.25", GEN3AI_SHIP_BAR_REPORT=str(tmp_path / "bar.json"))
    if bar_on:
        env["GEN3AI_SHIP_BAR"] = "1"
    return subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:xdist", "-q",
                           "planted_test.py"], cwd=str(tmp_path), env=env, capture_output=True, text=True,
                          timeout=scale_timeout(120))


@pytest.mark.integration
def test_the_conftest_fails_a_drifted_routine_test_and_names_a_near_one(tmp_path):
    r = _session(tmp_path, bar_on=True)
    out = r.stdout + r.stderr
    assert r.returncode == 1, f"a routine test at 2.4x the bar did not fail the session:\n{out}"
    assert "SHIP GATE 8 s BAR" in out
    lines = [ln for ln in out.splitlines() if "planted_test.py::" in ln and ("FAIL" in ln or "WARN" in ln)]
    assert any("FAIL" in ln and "test_drifted" in ln for ln in lines), lines
    assert any("WARN" in ln and "test_near" in ln for ln in lines), lines
    assert not any("test_marked_slow" in ln or "test_a_linchpin" in ln for ln in lines), \
        f"a cost-marked / linchpin test was judged: {lines}"
    assert "5 passed" in out, "every planted test itself passed: the FAIL is the bar's"
    assert (tmp_path / "bar.json").exists()


@pytest.mark.integration
def test_the_bar_off_judges_nothing(tmp_path):
    r = _session(tmp_path, bar_on=False)
    assert r.returncode == 0 and "8 s BAR" not in r.stdout, r.stdout + r.stderr
