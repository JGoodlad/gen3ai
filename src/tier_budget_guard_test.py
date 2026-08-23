"""The tier-budget guard must not fail a run on a measurement it cannot trust.

WHY THIS EXISTS. The guard's job is to stop a slow test hiding in the routine gate, and the obvious
implementation — "fail if any unmarked test overran a contention-SCALED budget" — is the exact
shape this tree has been burned by repeatedly: a starved `bridge_impl_parity` run once reported
39/40 timeouts as a clean PASS, and three separate investigations were voided by wall-clock bounds
measured beside a live trainer. Scaling alone is NOT sufficient here, and the numbers say why: the
factor is `loadavg / cpus` (~1.4 at load 22), while a compile-heavy test competing for every core
slowed **12.3s -> 65.9s (5.4x)** in a real run today. A scaled budget still false-fails.

So the guard is a real verdict on an idle box and ADVISORY on a busy one. These tests pin both
halves, using the documented `GEN3AI_TIMEOUT_SCALE` override so they assert the DECISION rather
than waiting for the box to be in a particular state.
"""
import importlib.util

import pytest

from utils.paths import repo_path

_CONFTEST = repo_path("conftest.py")


def _load_conftest():
    """Import the ROOT conftest as a module so its pure decision helpers are callable.

    pytest has already imported it as a plugin, but not under an importable name; loading a second
    copy is fine because the only state involved is a module-level list this test never populates.
    """
    spec = importlib.util.spec_from_file_location("_gen3_root_conftest", _CONFTEST)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def conf():
    return _load_conftest()


def test_an_idle_box_gives_a_real_verdict(conf, monkeypatch):
    monkeypatch.setenv("GEN3AI_TIMEOUT_SCALE", "1")
    assert conf._box_is_idle() is True


def test_a_contended_box_makes_the_guard_ADVISORY(conf, monkeypatch):
    """THE regression. At scale 6 the box is badly starved, so an overrun says nothing about the
    test and must not fail the run."""
    monkeypatch.setenv("GEN3AI_TIMEOUT_SCALE", "6")
    assert conf._box_is_idle() is False


def test_a_box_carrying_the_TRAINER_is_not_idle(conf, monkeypatch):
    """THE regression, and it is a real one this guard shipped with.

    The first cut reused `contention.py`'s 1.25 "looks idle" wording as its fail threshold. This
    box runs a `--nice 10` trainer essentially always, which parks the load average around 16-25
    on 16 cpus — straddling that line. A real gate run then FAILED at **factor 1.24**, on two
    tests whose measured idle cost is a third of what they showed, while printing "box looks idle"
    beside a load average of 19.9. Anything in that band must read as CONTENDED.
    """
    monkeypatch.setenv("GEN3AI_TIMEOUT_SCALE", "1.24")
    assert conf._box_is_idle() is False, (
        "factor 1.24 is a box under a live training run, not an idle one — failing a duration "
        "there is the knife-edge flake this guard exists to avoid")


def test_the_budget_scales_with_contention(conf, monkeypatch):
    """The budget still stretches — the idle/busy split is on top of that, not instead of it."""
    monkeypatch.setenv("GEN3AI_TIMEOUT_SCALE", "1")
    idle = conf._tier_budget_seconds()
    monkeypatch.setenv("GEN3AI_TIMEOUT_SCALE", "6")
    busy = conf._tier_budget_seconds()
    assert busy > idle, f"budget did not scale with contention ({idle}s -> {busy}s)"


def test_only_COST_markers_exempt_a_test(conf):
    """A `sim` test is not excused for being slow — the 6-battle obs-golden linchpin is `sim` and
    runs in ~4s, and it BELONGS in the routine gate. Exempting capability markers would put it back
    behind the wall that hid three obs regressions."""
    assert set(conf._COST_MARKERS) == {"slow", "e2e", "benchmark"}
    for capability in ("sim", "browser", "integration"):
        assert capability not in conf._COST_MARKERS, (
            f"`{capability}` says what a test NEEDS, not what it COSTS — exempting it would let a "
            "slow test of that kind sit in the routine gate unnoticed.")


class _Session:
    """Minimal stand-in for pytest's Session — the hook only ever sets `exitstatus`."""
    def __init__(self):
        self.exitstatus = 0


def test_an_overrun_FAILS_the_run_on_an_idle_box(conf, monkeypatch):
    monkeypatch.setenv("GEN3AI_TIMEOUT_SCALE", "1")
    conf._over_budget.clear()
    conf._over_budget.append(("slowpoke::t", 999.0))
    s = _Session()
    conf.pytest_sessionfinish(s, 0)
    conf._over_budget.clear()
    assert s.exitstatus == 1, "an overrun on an idle box must be a real verdict"


def test_an_overrun_does_NOT_fail_the_run_on_a_busy_box(conf, monkeypatch):
    """THE regression this guard's design turns on. Measured today: the same compile test ran 12.3s
    idle and 65.9s at load 22 — a 5.4x slowdown against a ~1.4x scaling factor. Failing on that
    would make the routine gate red whenever a training run is live, which is most of the time."""
    monkeypatch.setenv("GEN3AI_TIMEOUT_SCALE", "6")
    conf._over_budget.clear()
    conf._over_budget.append(("slowpoke::t", 999.0))
    s = _Session()
    conf.pytest_sessionfinish(s, 0)
    conf._over_budget.clear()
    assert s.exitstatus == 0, (
        "a contended box must make the guard ADVISORY — a duration measured under starvation is "
        "not a measurement of the test.")


def test_a_real_failure_is_never_masked_by_the_advisory_path(conf, monkeypatch):
    """The guard may only ever ADD a failure, never clear one."""
    monkeypatch.setenv("GEN3AI_TIMEOUT_SCALE", "6")
    conf._over_budget.clear()
    conf._over_budget.append(("slowpoke::t", 999.0))
    s = _Session()
    s.exitstatus = 1                      # a genuine test failure already happened
    conf.pytest_sessionfinish(s, 1)
    conf._over_budget.clear()
    assert s.exitstatus == 1


def test_the_guard_can_be_switched_off(conf, monkeypatch):
    """An escape hatch that does not work is not an escape hatch."""
    monkeypatch.setenv("GEN3AI_SKIP_TIER_BUDGET", "1")

    class _Report:
        when, skipped, duration, keywords, nodeid = "call", False, 10_000.0, {}, "x::y"

    before = len(conf._over_budget)
    conf.pytest_runtest_logreport(_Report())
    assert len(conf._over_budget) == before, "GEN3AI_SKIP_TIER_BUDGET did not disable recording"


def test_an_unmarked_overrun_is_recorded_and_a_marked_one_is_not(conf, monkeypatch):
    monkeypatch.delenv("GEN3AI_SKIP_TIER_BUDGET", raising=False)
    monkeypatch.setenv("GEN3AI_TIMEOUT_SCALE", "1")
    conf._over_budget.clear()

    class _Rep:
        when, skipped = "call", False

        def __init__(self, nodeid, duration, keywords):
            self.nodeid, self.duration, self.keywords = nodeid, duration, keywords

    conf.pytest_runtest_logreport(_Rep("fast::t", 1.0, {}))
    conf.pytest_runtest_logreport(_Rep("slow_marked::t", 10_000.0, {"slow": 1}))
    conf.pytest_runtest_logreport(_Rep("sim_unmarked::t", 10_000.0, {"sim": 1}))
    recorded = [n for n, _ in conf._over_budget]
    assert recorded == ["sim_unmarked::t"], (
        f"expected only the unmarked overrun to be recorded, got {recorded}")
    conf._over_budget.clear()
