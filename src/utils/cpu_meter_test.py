"""Tests for the windowed contention meter (`gen3_contention_meter_v2`, `utils.cpu_meter`).

Two layers. The PURE tests feed synthetic counter samples, so they pin the MAPPING — including the
2026-09-30 incident shape, where load1/cpus read 1.00 on a box whose busy threads were sharing cores
— without depending on the box's load. The one REAL test spawns busy processes and reads the kernel
counters, because a meter that is only ever tested against numbers it was handed has never been
shown to read the machine.
"""
import os
import subprocess
import sys
import time

import pytest

from utils import cpu_meter
from utils.cpu_meter import (
    SMT_SIBLING_SLOWDOWN,
    ContentionReading,
    CpuSample,
    Topology,
    reading_between,
    smt_slowdown,
    take_sample,
)

# The box the meter was calibrated on: 8 cores, 16 hardware threads.
_TOPO = Topology(tuple(range(16)), 8)
_TICKS = 100          # per cpu per second


@pytest.fixture(autouse=True)
def _no_override(monkeypatch):
    monkeypatch.delenv("GEN3AI_TIMEOUT_SCALE", raising=False)


def _pair(*, busy_cpus, seconds=10.0, wait_per_run=0.0, psi_share=0.0, load1=0.0,
          self_cpus=None, schedstat=True, procstat=True):
    """Two samples ``seconds`` apart describing a window with the given averages."""
    total = int(16 * _TICKS * seconds)
    busy = int(busy_cpus * _TICKS * seconds)
    run = int(busy_cpus * seconds * 1e9)
    self_a = {} if self_cpus is None else {4242: 100.0}
    self_b = {} if self_cpus is None else {4242: 100.0 + self_cpus * seconds}
    a = CpuSample(mono=1000.0, wall=5000.0,
                  busy_ticks=0 if procstat else None, total_ticks=0 if procstat else None,
                  run_ns=0 if schedstat else None, wait_ns=0 if schedstat else None,
                  psi_some_us=0, load1=load1, self_cpu_s=self_a)
    b = CpuSample(mono=1000.0 + seconds, wall=5000.0 + seconds,
                  busy_ticks=busy if procstat else None, total_ticks=total if procstat else None,
                  run_ns=run if schedstat else None,
                  wait_ns=int(run * wait_per_run) if schedstat else None,
                  psi_some_us=int(psi_share * seconds * 1e6), load1=load1, self_cpu_s=self_b)
    return a, b


# --- the mapping ---------------------------------------------------------------------------------


def test_an_idle_window_reads_quiet():
    a, b = _pair(busy_cpus=1.0, load1=1.0)
    r = reading_between(a, b, topo=_TOPO)
    assert r.factor == 1.0, r.describe()


def test_THE_INCIDENT_smt_sharing_reads_contended_where_load1_read_quiet():
    """2026-09-30: ~10-12 busy threads on 8 cores / 16 threads, load1 ~10, PSI some ~0.5 %, no
    run-queue wait — and tests ran 1.3-2.7x slower. The OLD meter (load1 / cpus) read 1.00 and
    ENFORCED the budget. Revert the occupancy term and this fails."""
    a, b = _pair(busy_cpus=12.0, psi_share=0.005, load1=10.0)
    r = reading_between(a, b, topo=_TOPO)
    assert r.load1_factor == 1.0, "precondition: the old meter reads this window as quiet"
    assert r.factor >= 1.4, r.describe()
    assert r.factor == pytest.approx(smt_slowdown(12.0, _TOPO))


def test_the_session_s_OWN_load_is_not_contention():
    """A test that saturates the box by itself is exactly what the budget must catch."""
    a, b = _pair(busy_cpus=12.0, load1=12.0, self_cpus=12.0)
    r = reading_between(a, b, self_root=4242, topo=_TOPO)
    assert r.factor == pytest.approx(1.0), r.describe()
    # ...and the same window with nothing attributed to self is contended.
    assert reading_between(a, b, topo=_TOPO).factor > 1.4


def _with_own_tasks(a, b, *, run_s, wait_s, root=4242):
    """Attach the self subtree's own-task counters (one long-lived task) to a sample pair."""
    import dataclasses
    a = dataclasses.replace(a, self_tasks={root: {7: (0, 0)}},
                            self_cpu_s=a.self_cpu_s or {root: 0.0})
    b = dataclasses.replace(b, self_tasks={root: {7: (int(run_s * 1e9), int(wait_s * 1e9))}},
                            self_cpu_s=b.self_cpu_s or {root: 0.0})
    return a, b


def test_ANOTHER_process_queueing_on_itself_is_not_our_contention():
    """MEASURED 2026-09-30: the box-wide /proc/schedstat read x11.1 over a 29 s window with 8 of 16
    cpus busy and PSI at 1.7 % — another agent's pinned threads queueing on their own cpus. Our
    tasks did not wait, so the queue term must come from OUR tasks. Revert to the box-wide counter
    and this fails."""
    a, b = _pair(busy_cpus=8.0, wait_per_run=10.1, psi_share=0.017, load1=8.0, self_cpus=1.0)
    assert reading_between(a, b, topo=_TOPO).factor > 5, "precondition: box-wide reads the anomaly"
    a, b = _with_own_tasks(a, b, run_s=10.0, wait_s=0.05)
    r = reading_between(a, b, self_root=4242, topo=_TOPO)
    assert r.factor < 1.05 and "OWN tasks" in r.source, r.describe()


def test_OUR_tasks_waiting_behind_others_is_contention():
    a, b = _pair(busy_cpus=16.0, wait_per_run=0.0, load1=24.0, self_cpus=1.0)
    a, b = _with_own_tasks(a, b, run_s=10.0, wait_s=10.0)
    r = reading_between(a, b, self_root=4242, topo=_TOPO)
    assert r.queue == pytest.approx(1.0 + 1.0 * 15.0 / 16.0)
    assert r.factor > 3.0, r.describe()


def test_run_queue_wait_multiplies_on_top_of_smt():
    a, b = _pair(busy_cpus=16.0, wait_per_run=1.0, psi_share=0.7, load1=32.0)
    r = reading_between(a, b, topo=_TOPO)
    assert r.smt == pytest.approx(SMT_SIBLING_SLOWDOWN)
    assert r.queue == pytest.approx(2.0)
    assert r.factor == pytest.approx(2.0 * SMT_SIBLING_SLOWDOWN)


def test_no_smt_box_has_no_occupancy_term():
    flat = Topology(tuple(range(16)), 16)
    a, b = _pair(busy_cpus=15.0)
    assert reading_between(a, b, topo=flat).factor == 1.0


def test_fallback_chain_names_its_source():
    a, b = _pair(busy_cpus=16.0, psi_share=0.5, schedstat=False)
    r = reading_between(a, b, topo=_TOPO)
    assert "PSI" in r.source and r.queue == pytest.approx(2.0)
    a, b = _pair(busy_cpus=16.0, procstat=False, schedstat=False)
    assert "load1" in reading_between(a, b, topo=_TOPO).source


def test_the_override_wins(monkeypatch):
    monkeypatch.setenv("GEN3AI_TIMEOUT_SCALE", "3")
    a, b = _pair(busy_cpus=1.0)
    r = reading_between(a, b, topo=_TOPO)
    assert r.factor == 3.0 and "GEN3AI_TIMEOUT_SCALE" in r.source


def test_window_brackets_the_test_not_the_session():
    """A burst in the middle of a session must be charged to the test it overlapped."""
    m = cpu_meter.WindowMeter(lambda: [])
    quiet_a, quiet_b = _pair(busy_cpus=1.0)
    s0 = quiet_a
    s1 = CpuSample(mono=s0.mono + 10, wall=s0.wall + 10, busy_ticks=16 * _TICKS * 10 // 16,
                   total_ticks=16 * _TICKS * 10, run_ns=10 ** 10, wait_ns=0, psi_some_us=0,
                   load1=1.0)
    # 10 s later, fully busy for 10 s (16 cpus), then 100 s quiet.
    s2 = CpuSample(mono=s1.mono + 10, wall=s1.wall + 10,
                   busy_ticks=s1.busy_ticks + 16 * _TICKS * 10, total_ticks=s1.total_ticks
                   + 16 * _TICKS * 10, run_ns=s1.run_ns + 16 * 10 ** 10, wait_ns=0,
                   psi_some_us=0, load1=1.0)
    s3 = CpuSample(mono=s2.mono + 100, wall=s2.wall + 100,
                   busy_ticks=s2.busy_ticks + 16 * _TICKS * 100 // 16,
                   total_ticks=s2.total_ticks + 16 * _TICKS * 100,
                   run_ns=s2.run_ns + 10 ** 11, wait_ns=0, psi_some_us=0, load1=1.0)
    m.topo = _TOPO
    m.samples = [s0, s1, s2, s3]
    burst = m.window(s1.wall + 0.1, s2.wall - 0.1, None)
    assert burst.busy_cpus == pytest.approx(16.0) and burst.factor > 1.5, burst.describe()
    whole = reading_between(s0, s3, topo=_TOPO)
    assert whole.factor < 1.05, "precondition: the SESSION average hides the burst"


def test_describe_carries_the_source_and_the_old_reading():
    a, b = _pair(busy_cpus=12.0, load1=10.0)
    text = reading_between(a, b, topo=_TOPO).describe()
    assert "/proc/schedstat" in text and "old load1/cpus meter reads 1.00" in text
    assert isinstance(ContentionReading(1.0, "x").describe(), str)


# --- the real machine ----------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not os.path.exists("/proc/stat"), reason="Linux /proc counters only")
def test_REAL_busy_workers_read_contended_and_self_attribution_subtracts_them():
    """Spawn 12 CPU-bound processes for ~3 s — the incident shape: 12 busy threads cannot raise
    load1 past 16 in 3 s (it moves ~0.8), so the old meter reads this window as quiet unless the box
    was already loaded. The new meter must read it contended when the workers are NOT ours, and must
    attribute them to us when they are."""
    k = 12
    workers = [subprocess.Popen([sys.executable, "-c", "while 1: pass"]) for _ in range(k)]
    try:
        time.sleep(0.5)                                  # let them start burning
        a = take_sample([os.getpid()])
        time.sleep(3.0)
        b = take_sample([os.getpid()])
    finally:
        for w in workers:
            w.kill()
        for w in workers:
            w.wait()
    external = reading_between(a, b)                     # nothing attributed to self
    ours = reading_between(a, b, self_root=os.getpid())
    assert external.busy_cpus is not None and external.busy_cpus >= 0.5 * k, external.describe()
    assert external.factor >= 1.3, (
        f"{k} busy workers did not read as contention: {external.describe()}")
    assert ours.self_cpus is not None and ours.self_cpus >= 0.5 * k, ours.describe()
    assert ours.factor < external.factor, (
        f"our own workers were not subtracted: ours {ours.describe()} vs {external.describe()}")
