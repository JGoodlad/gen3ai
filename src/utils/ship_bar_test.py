"""THE 8 s BAR (``utils.ship_bar``): the decision, and the root conftest's wiring in a real child session.

Load-bearing pins, each FAILS on revert of what it guards:
* the decision bands — under the bar ok, the margin band WARN (through the nominal 2x line and its 25 %
  exclusion band), >= 2.5x the stretched bar FAIL, an
  unreadable window never FAIL, a busier box stretches the bar and a quiet one never shrinks it;
* the SCOPE attribution (``SCOPE`` in ``utils.ship_bar``): a module-scoped fixture's setup AND teardown are
  charged to the MODULE, once, never to the first / last test that happens to pay them — so the same
  tree gives the same verdict in a different test order and under xdist's ``--dist load`` split; a
  function-scoped fixture still counts toward its test;
* the WIRING: a child pytest session over the real root conftest, with the bar on, FAILS (exit 1) on a
  routine test far over a (shrunk) bar, names a near one as WARN, and exempts ``slow`` / ``linchpin``;
  the same session with the bar off exits 0.

``integration``: one child pytest session per arm, ~2-4 s each.
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
    (16.0, 2.7, "warn"),         # the NOMINAL 2x line is inside the exclusion band: never a verdict
    (16.7, 2.32, "warn"),        # THE INCIDENT (2026-10-19): compile_cache_test[passing] 16.7 s vs 16.0 s
    (19.9, 2.7, "warn"),         # the top of the band (FAIL_BAND = 25 %: 20 s)
    (20.0, 2.7, "fail"),         # clearly above: FAIL
    (20.0, 1.0, "fail"),         # a quiet box never SHRINKS the bar below 8 s ...
    (15.9, 5.4, "ok"),           # ... and a box 2x busier than the reference doubles it
    (39.9, 5.4, "warn"),
    (40.0, 5.4, "fail"),
    (500.0, None, "warn"),       # an unreadable window is never a verdict
])
def test_the_decision_bands(total, factor, want, monkeypatch):
    monkeypatch.delenv(SB.BAR_ENV, raising=False)
    assert SB.judge(total, factor) == want


def test_a_read_just_over_the_nominal_fail_line_is_warn_and_a_clear_one_is_fail(monkeypatch):
    """Rule 8: the 2x line has an exclusion band (``FAIL_BAND``). Fails on revert of the band: a read at 1.04x the
    nominal line FAILED (the 2026-10-19 incident), and so did anything at 1.0x."""
    monkeypatch.delenv(SB.BAR_ENV, raising=False)
    for factor in (1.0, 2.7, 3.5, 4.9):
        line = SB.FAIL_MARGIN * SB.threshold(factor)
        assert SB.judge(1.04 * line, factor) == "warn", factor
        assert SB.judge(line, factor) == "warn", factor
        assert SB.judge(SB.FAIL_LINE * SB.threshold(factor) * 1.02, factor) == "fail", factor
        assert SB.judge(1.5 * line, factor) == "fail", factor
    assert SB.FAIL_BAND >= 0.25, "the band is at least the contention meter's documented 25 % under-read"


_PLANTED = '''\
import time, pytest

def test_fast():
    pass

def test_drifted():
    time.sleep(0.9)

def test_near():
    time.sleep(0.3)

@pytest.mark.slow
def test_marked_slow():
    time.sleep(0.6)

@pytest.mark.linchpin
def test_a_linchpin():
    time.sleep(0.6)
'''


def _session(tmp_path, *, bar_on: bool, source: str = _PLANTED, args=("-p", "no:xdist"), targets=("planted_test.py",)):
    shutil.copy(repo_root() / "conftest.py", tmp_path / "conftest.py")
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n    slow: slow\n    linchpin: linchpin\n")
    (tmp_path / "planted_test.py").write_text(source)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTEST_", "GEN3AI_SHIP_"))}
    env.update(GEN3AI_SKIP_DEPS_GUARD="1", PYTHONPATH=str(src_root()), GEN3AI_COST_SCHEDULE="0",
               GEN3AI_SLOW_STATUS_FILE=str(tmp_path / "status.json"), GEN3AI_SKIP_TIER_BUDGET="1",
               GEN3AI_TIMEOUT_SCALE="1",            # the contention reading is pinned at x1: a fixed factor
               GEN3AI_SHIP_BAR_S="0.25", GEN3AI_SHIP_BAR_REPORT=str(tmp_path / "bar.json"))
    if bar_on:
        env["GEN3AI_SHIP_BAR"] = "1"
    return subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args, "-q",
                           *targets], cwd=str(tmp_path), env=env, capture_output=True, text=True,
                          timeout=scale_timeout(120))


@pytest.mark.integration
def test_the_conftest_fails_a_drifted_routine_test_and_names_a_near_one(tmp_path):
    r = _session(tmp_path, bar_on=True)
    out = r.stdout + r.stderr
    assert r.returncode == 1, f"a routine test at 3.6x the bar did not fail the session:\n{out}"
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


# A module fixture whose setup (0.5 s) AND teardown (0.5 s) cost 1.0 s against a 0.25 s bar (4x: clear of the
# fail line's 2.5x), four tests that are individually instant, and one test whose own FUNCTION-scoped fixture
# costs 0.9 s.
_SCOPED = '''\
import time, pytest

@pytest.fixture(scope="module")
def heavy():
    time.sleep(0.5)
    yield
    time.sleep(0.5)

@pytest.fixture
def slow_fn():
    time.sleep(0.9)

def test_a(heavy): pass
def test_b(heavy): pass
def test_c(heavy): pass
def test_d(heavy): pass
def test_fn_fixture_counts_toward_the_test(slow_fn): pass
'''


def _judged_rows(out: str):
    """The (verdict, who) rows of the bar's section, `who` normalised to the module / test it names."""
    rows = []
    for ln in out.splitlines():
        parts = ln.split()
        if len(parts) >= 4 and parts[1] in ("FAIL", "WARN") and parts[2].startswith("x") and "planted_test.py" in ln:
            rows.append((parts[1], ln.split(parts[2], 1)[1].strip().split(" [", 1)[0]))
    return sorted(rows)


_ORDERINGS = {
    "definition order": (("-p", "no:xdist"), ("planted_test.py",)),
    "reversed order": (("-p", "no:xdist"), tuple(f"planted_test.py::{t}" for t in (
        "test_fn_fixture_counts_toward_the_test", "test_d", "test_c", "test_b", "test_a"))),
    "xdist load split": (("-n", "2", "--dist", "load"), ("planted_test.py",)),
}


#: The ONE verdict every arm must give (so all arms agree): the module row, and the test whose own
#: function-scoped fixture cost 0.9 s — never a test that merely asked for the module fixture first / last.
_EXPECTED_ROWS = [("FAIL", "planted_test.py"), ("FAIL", "planted_test.py::test_fn_fixture_counts_toward_the_test")]


@pytest.mark.integration
@pytest.mark.parametrize("label", sorted(_ORDERINGS))
def test_a_module_fixtures_cost_is_the_modules_in_every_order_and_split(tmp_path, label):
    """THE DETERMINISM PROOF (rule 8). Before the scope attribution the 0.5 s setup landed on whichever test
    ran first and the 0.5 s teardown on whichever ran last, so the NAMED test changed with the order and the
    split (the revert names ``test_a`` / ``test_d`` here). Every arm — definition order, reversed order, and
    ``-n 2 --dist load`` — must name exactly :data:`_EXPECTED_ROWS`, a constant, so the arms agree."""
    args, targets = _ORDERINGS[label]
    r = _session(tmp_path, bar_on=True, source=_SCOPED, args=args, targets=targets)
    out = r.stdout + r.stderr
    assert r.returncode == 1, f"[{label}] the over-bar module must FAIL the session:\n{out}"
    assert "5 passed" in out, f"[{label}]\n{out}"
    assert _judged_rows(out) == _EXPECTED_ROWS, f"[{label}] the verdict depends on the order / the split:\n{out}"


@pytest.mark.integration
def test_a_cheap_module_fixture_judges_nothing(tmp_path):
    src = _SCOPED.replace("time.sleep(0.5)", "time.sleep(0.02)").replace("time.sleep(0.9)", "pass")
    r = _session(tmp_path, bar_on=True, source=src)
    assert r.returncode == 0 and "8 s BAR" not in r.stdout, r.stdout + r.stderr
