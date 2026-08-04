"""Unit tests for the contention-robust timeout helpers (`gen3_contention_robust_timeouts_v1`).

Pure + fast: the load average is monkeypatched, so these assert the DECISION LOGIC (does a
slow-but-progressing wait survive; does a wedged one still fail) without depending on the
box's actual load — which would make the tests for contention-robustness themselves
contention-sensitive.
"""
import os
import time

import pytest

from utils import contention
from utils.contention import (
    ProgressDeadline,
    ProgressTimeout,
    cpu_contention_factor,
    describe_contention,
    scale_timeout,
    warn_if_contended,
)


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch):
    """Each test starts from a cold measurement and no override."""
    monkeypatch.delenv("GEN3AI_TIMEOUT_SCALE", raising=False)
    contention._cached = None
    yield
    contention._cached = None


def _fake_load(monkeypatch, load: float, ncpu: int = 16):
    monkeypatch.setattr(contention.os, "getloadavg", lambda: (load, load, load))
    monkeypatch.setattr(contention, "_cpu_count", lambda: ncpu)
    contention._cached = None


# --- the factor -------------------------------------------------------------------------


def test_idle_box_does_not_scale(monkeypatch):
    """An idle box must be byte-identical to the old fixed timeouts — this whole change is a
    no-op when nothing else is running, which is what makes it safe to apply broadly."""
    _fake_load(monkeypatch, 0.1)
    assert cpu_contention_factor() == 1.0
    assert scale_timeout(20.0) == 20.0


def test_load_below_cpu_count_still_does_not_scale(monkeypatch):
    """Load 8 on 16 cpus means half the box is free; a single-threaded waiter is not slowed."""
    _fake_load(monkeypatch, 8.0)
    assert cpu_contention_factor() == 1.0


def test_production_training_run_scales(monkeypatch):
    """The regime that voided three investigations: ~35 load on 16 cpus."""
    _fake_load(monkeypatch, 35.0)
    assert cpu_contention_factor() == pytest.approx(35.0 / 16.0)
    assert scale_timeout(20.0) == pytest.approx(43.75)


def test_factor_is_clamped(monkeypatch):
    """A runaway load must not turn a bounded wait into an effectively unbounded one."""
    _fake_load(monkeypatch, 100_000.0)
    assert cpu_contention_factor() == contention._MAX_FACTOR


def test_affinity_mask_is_the_denominator(monkeypatch):
    """A process pinned to 2 cpus is starved at a load the 16-cpu box shrugs off, so the
    denominator must be the affinity mask, not the machine's core count."""
    _fake_load(monkeypatch, 8.0, ncpu=2)
    assert cpu_contention_factor() == pytest.approx(4.0)


def test_env_override_wins(monkeypatch):
    _fake_load(monkeypatch, 0.1)
    monkeypatch.setenv("GEN3AI_TIMEOUT_SCALE", "4")
    assert cpu_contention_factor() == 4.0


def test_malformed_override_falls_back_to_measurement(monkeypatch):
    """A typo in an env var must not break every timeout in the process."""
    _fake_load(monkeypatch, 35.0)
    monkeypatch.setenv("GEN3AI_TIMEOUT_SCALE", "not-a-number")
    assert cpu_contention_factor() == pytest.approx(35.0 / 16.0)


def test_factor_is_cached_then_refreshes(monkeypatch):
    """Cheap enough for a hot path, but must still ADAPT when a trainer starts mid-wait."""
    _fake_load(monkeypatch, 0.1)
    assert cpu_contention_factor() == 1.0
    _fake_load(monkeypatch, 35.0)
    contention._cached = (time.monotonic(), 1.0)  # pretend we sampled just now
    assert cpu_contention_factor() == 1.0, "should still be serving the cached value"
    assert cpu_contention_factor(refresh=True) == pytest.approx(35.0 / 16.0)


# --- ProgressDeadline: the core claim ----------------------------------------------------


def test_slow_but_progressing_never_expires(monkeypatch):
    """THE POINT OF THE MODULE. A wait that takes 100x its idle budget in TOTAL is fine as
    long as something keeps happening — that is exactly the starved-but-healthy bridge
    battle a total-duration cap kills."""
    _fake_load(monkeypatch, 0.1)  # not even scaled — progress alone must carry it
    d = ProgressDeadline(idle_budget_s=0.05, what="slow battle")
    for _ in range(20):
        time.sleep(0.01)
        d.progress()
        d.check()  # must not raise
    assert d.elapsed_seconds > 0.15
    assert d.progress_count == 20


def test_wedged_wait_still_fails(monkeypatch):
    """The complement: robustness must not cost us the detection. No progress at all still
    raises — otherwise this module would just be hiding the deadlocks it was built beside."""
    _fake_load(monkeypatch, 0.1)
    d = ProgressDeadline(idle_budget_s=0.02, what="wedged battle")
    time.sleep(0.05)
    assert d.expired()
    with pytest.raises(ProgressTimeout, match="no progress"):
        d.check()


def test_contention_widens_the_idle_budget(monkeypatch):
    """Same elapsed idle time: fails on an idle box, tolerated on a starved one."""
    _fake_load(monkeypatch, 0.1)
    d = ProgressDeadline(idle_budget_s=0.02)
    time.sleep(0.05)
    assert d.expired()

    _fake_load(monkeypatch, 160.0)  # factor 10
    assert not d.expired(), "the same idle gap must be tolerated under 10x contention"


def test_timeout_message_self_diagnoses(monkeypatch):
    """The three void investigations cost hours because the failure did not mention the box.
    A ProgressTimeout must name the load average so the reader stops immediately."""
    _fake_load(monkeypatch, 35.0)
    d = ProgressDeadline(idle_budget_s=1e-9, what="parity battle")
    time.sleep(0.001)
    with pytest.raises(ProgressTimeout) as ei:
        d.check()
    msg = str(ei.value)
    assert "parity battle" in msg
    assert "load average" in msg
    assert "CPU-STARVED" in msg
    assert "ps -eo pcpu" in msg, "must name the exact diagnostic command"


def test_total_budget_catches_livelock(monkeypatch):
    """Progress that never converges resets the idle bound forever; the optional total
    backstop is what still stops it."""
    _fake_load(monkeypatch, 0.1)
    d = ProgressDeadline(idle_budget_s=10.0, total_budget_s=0.05, what="chattering child")
    for _ in range(10):
        time.sleep(0.01)
        d.progress()
    with pytest.raises(ProgressTimeout, match="livelock"):
        d.check()


def test_total_budget_is_off_by_default(monkeypatch):
    """Opting in matters: a default total bound would re-introduce the duration sensitivity
    the module exists to remove."""
    _fake_load(monkeypatch, 0.1)
    d = ProgressDeadline(idle_budget_s=10.0)
    assert d.total_budget_s is None
    time.sleep(0.02)
    d.progress()
    d.check()


def test_progress_timeout_is_a_timeout_error():
    """Callers with an existing `except TimeoutError` keep working; the distinct type is for
    the ones that must NOT fold a timeout into a semantic bucket."""
    assert issubclass(ProgressTimeout, TimeoutError)


def test_rejects_nonpositive_budget():
    with pytest.raises(ValueError):
        ProgressDeadline(idle_budget_s=0)


def test_describe_contention_is_honest_about_an_idle_box(monkeypatch):
    _fake_load(monkeypatch, 0.1)
    assert "box looks idle" in describe_contention()


# --- benchmarks get the OPPOSITE treatment: warn, never stretch ---------------------------


def test_benchmark_guard_is_silent_on_an_idle_box(monkeypatch, capsys):
    _fake_load(monkeypatch, 0.1)
    assert warn_if_contended("obs-build") is False
    assert capsys.readouterr().err == ""


def test_benchmark_guard_warns_on_a_busy_box(monkeypatch, capsys):
    """A benchmark's output IS the measurement, so a busy box must be stated, not absorbed —
    the node-vs-rust throughput comparison that had to be superseded was reported with nothing
    in its output saying the box had been saturated."""
    _fake_load(monkeypatch, 35.0)
    assert warn_if_contended("bridge-impl throughput") is True
    err = capsys.readouterr().err
    assert "THE BOX IS BUSY" in err
    assert "bridge-impl throughput" in err
    assert "not comparable to an idle-box baseline" in err


def test_benchmark_guard_warns_rather_than_refuses(monkeypatch):
    """A same-load back-to-back A/B is still informative, so this must not raise — the caller
    judges that, and the only unacceptable behaviour is silence."""
    _fake_load(monkeypatch, 35.0)
    warn_if_contended("throughput")  # must not raise
