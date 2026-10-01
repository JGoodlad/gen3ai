"""`cuda_memory_trend` — the K6 CUDA memory LEAK DETECTOR's rules (pure, no GPU; routine tier).

Each rule test was checked to FAIL with its rule reverted (2026-09-30): the odd-group MEDIAN (a mean
-> the step-up tests read SUSTAINED), the PERSISTENCE count (1 -> the short-ramp tests STOP), the
every-group RISE (any -> the short-ramp tests STOP), the HORIZON comparison (dropped -> the slow-leak
and bounded-growth tests STOP), the STOP itself (-> the leak tests), and the noise bar + minimum slope
(0 -> the measured units read SUSTAINED; with the horizon dropped too, the magnified noise STOPs).
The measured fixture is `designs/research_state/measurements/k6_k8/memory/healthy_trace.json`.
"""
from __future__ import annotations

import json
from typing import Iterable, List, Optional

import pytest

from agents.training import cuda_memory_trend as M
from agents.training.cuda_memory_trend import GiB, MiB, MemorySample, MemoryTrend
from utils.paths import repo_path

TOTAL = 12 * GiB
FIXTURE = repo_path("designs", "research_state", "measurements", "k6_k8", "memory", "healthy_trace.json")


def _s(update: int, allocated: float, *, phase: str = "post_update", reserved: Optional[float] = None,
       peak_reserved: Optional[float] = None, segments: int = 40, retries: int = 0,
       other: float = 0.5 * GiB) -> MemorySample:
    """One sample on a 12 GiB card: the process holds ``reserved``; ``other`` bytes belong to
    other processes / the context; everything else is free."""
    res = int(reserved if reserved is not None else allocated + 3 * GiB)
    pk = int(peak_reserved if peak_reserved is not None else res)
    return MemorySample(update=update, phase=phase, allocated=int(allocated), reserved=res,
                        active=int(allocated), inactive_split=int(res - allocated), segments=segments,
                        segments_allocated_total=segments, peak_allocated=int(allocated + 2 * GiB),
                        peak_reserved=pk, alloc_retries=retries, ooms=0,
                        device_free=int(TOTAL - res - other), device_total=TOTAL)


def _run(samples: Iterable[MemorySample], trend: Optional[MemoryTrend] = None) -> List[M.TrendVerdict]:
    t = trend or MemoryTrend()
    return [t.observe(s) for s in samples]


def _levels(vs: List[M.TrendVerdict]) -> set:
    return {v.level for v in vs}


# ------------------------------------------------------------------------------------------------
# the measured healthy learner never stops
# ------------------------------------------------------------------------------------------------

def _fixture_units() -> List[List[MemorySample]]:
    data = json.loads(FIXTURE.read_text())
    units = []
    for u in data["units"]:
        units.append([MemorySample(**{k: r[k] for k in M.MemorySample.__dataclass_fields__})
                      for r in u["samples"]])
    return units


def test_measured_healthy_units_never_stop_or_read_sustained():
    units = _fixture_units()
    assert len(units) >= 2 and all(len(u) >= 30 for u in units), "the committed healthy trace is thin"
    for u in units:
        vs = _run(u)
        assert "STOP" not in _levels(vs)
        assert not any(v.sustained for v in vs)


def test_measured_noise_scaled_up_never_stops():
    """The measured unit-to-unit floor wobble, magnified 8x around its own median, over a 300-update
    process: still no STOP (the noise bar is not the only line of defence)."""
    units = _fixture_units()
    for u in units:
        post = [s for s in u if s.update >= M.WARMUP_UPDATES]
        med = sorted(s.allocated for s in post)[len(post) // 2]
        seq = []
        for k in range(300):
            s = post[k % len(post)]
            seq.append(MemorySample(**{**s.as_row(), "update": M.WARMUP_UPDATES + k // 2,
                                        "allocated": int(med + 8 * (s.allocated - med))}))
        assert "STOP" not in _levels(_run(seq))


# ------------------------------------------------------------------------------------------------
# step-ups, plateaus, ramps
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("at", list(range(3, 40)))
def test_single_step_up_warns_and_never_stops(at):
    """A one-off +1.5 GiB step of live memory (a lazily-built cache that then stops), at every
    position: WARNs with the step, never STOPs. The headroom after it is small (projection-prone)."""
    base = 4 * GiB
    seq = [_s(u, base + (1.5 * GiB if u >= at else 0), reserved=base + 2 * GiB + (1.5 * GiB if u >= at else 0))
           for u in range(80)]
    vs = _run(seq)
    assert "STOP" not in _levels(vs)
    assert not any(v.sustained for v in vs), "a single step must never line up three odd-group medians"
    assert any(any(r.startswith("step_up") for r in v.reasons) for v in vs)


def test_fragmentation_growth_then_plateau_never_stops():
    """Reserved (and segments) climb for 30 updates then plateau, the live floor flat: never STOP,
    never sustained — fragmentation is not a leak."""
    base = 3 * GiB
    seq = [_s(u, base, reserved=base + 1 * GiB + min(u, 30) * 150 * MiB, segments=40 + min(u, 30))
           for u in range(120)]
    t = MemoryTrend()
    vs = _run(seq, t)
    assert "STOP" not in _levels(vs)
    assert not any(v.sustained for v in vs)
    assert any(any(r.startswith("reserved_step") for r in v.reasons) for v in vs)
    assert vs[-1].level == "OK"
    assert t.segments_after_freeze() == 30 - M.WARMUP_UPDATES
    assert t.tb_scalars()[M.TB_PREFIX + "segments_after_freeze"] == 30 - M.WARMUP_UPDATES


@pytest.mark.parametrize("start", list(range(2, 30, 3)))
def test_short_fast_ramp_then_plateau_never_stops(start):
    """Live memory ramps 2 GiB over 4 updates (2 windows) right up against the ceiling, then
    plateaus: the group medians can line up for at most the ramp's windows, under the persistence."""
    base = 3 * GiB

    def alloc(u: int) -> float:
        return base + 2 * GiB * min(max(u - start, 0), 4) / 4

    seq = [_s(u, alloc(u), reserved=alloc(u) + 2.5 * GiB, other=2 * GiB) for u in range(100)]
    vs = _run(seq)
    assert "STOP" not in _levels(vs)


def test_bounded_growth_far_from_the_ceiling_warns_then_clears():
    """30 MiB/update of live growth for 40 updates then flat, on a card with ~7 GiB free: sustained
    while it lasts (WARN with a projection beyond the horizon), never STOP, OK again after."""
    base = 2 * GiB
    seq = [_s(u, base + min(u, 40) * 30 * MiB, reserved=base + 1 * GiB + min(u, 40) * 30 * MiB)
           for u in range(120)]
    vs = _run(seq)
    assert "STOP" not in _levels(vs)
    assert any(v.sustained and v.level == "WARN" for v in vs)
    assert vs[-1].level == "OK"


# ------------------------------------------------------------------------------------------------
# leaks
# ------------------------------------------------------------------------------------------------

def test_linear_leak_crossing_inside_the_horizon_stops():
    """60 MiB/update of live memory with ~6 GiB of headroom: it reaches the ceiling ~100 updates in,
    so by the time the span and the persistence are full it projects inside 25 updates -> STOP,
    with the projection numbers and the stop message."""
    base = 2 * GiB
    t = MemoryTrend()
    vs = _run([_s(u, base + u * 60 * MiB, reserved=base + 2 * GiB + u * 60 * MiB) for u in range(120)], t)
    stops = [v for v in vs if v.level == "STOP"]
    assert stops, [v.message for v in vs[-3:]]
    v = stops[0]
    assert v.sustained and v.updates_to_ceiling is not None and v.updates_to_ceiling <= M.HORIZON_UPDATES
    assert v.slope_bytes_per_update == pytest.approx(60 * MiB, rel=0.01)
    assert v.projected_demand_at_horizon > v.ceiling_bytes
    assert "SUSTAINED" in v.message and "projected OOM in" in v.message
    assert t.stop_message() is not None or vs[-1].level != "STOP"
    # the STOP came only after the persistence: the PERSIST-1 closes before it were WARN
    i = vs.index(v)
    closes = [w for w in vs[:i] if w.window_closed]
    assert all(w.level == "WARN" for w in closes[-(M.PERSIST_WINDOWS - 1):])


def test_slow_leak_beyond_the_horizon_warns_with_its_projection():
    """12 MiB/update with ~7 GiB free: sustained, projected to the ceiling ~1,700 updates out — a WARN
    that carries the slope and the updates-to-ceiling, never a STOP."""
    base = 2 * GiB
    vs = _run([_s(u, base + u * 12 * MiB, reserved=base + 1 * GiB + u * 12 * MiB) for u in range(150)])
    assert "STOP" not in _levels(vs)
    warns = [v for v in vs if v.sustained]
    assert warns and all(v.level == "WARN" for v in warns)
    v = warns[-1]
    assert v.updates_to_ceiling is not None and v.updates_to_ceiling > M.HORIZON_UPDATES
    assert v.slope_bytes_per_update == pytest.approx(12 * MiB, rel=0.01)
    assert "projected OOM in" in v.message and "sustained_growth_beyond_horizon" in v.reasons


def test_integrator_horizon_overrides_the_default():
    """The same slow leak STOPs when the integrator declares a horizon past its crossing."""
    base = 2 * GiB
    t = MemoryTrend()
    vs = [t.observe(_s(u, base + u * 12 * MiB, reserved=base + 1 * GiB + u * 12 * MiB), horizon_updates=10_000)
          for u in range(150)]
    assert "STOP" in _levels(vs)


# ------------------------------------------------------------------------------------------------
# outputs and inputs
# ------------------------------------------------------------------------------------------------

def test_tb_scalars_and_window_cadence():
    t = MemoryTrend()
    closes = 0
    for u in range(40):
        v = t.observe(_s(u, 3 * GiB, phase="post_rollout"))
        v2 = t.observe(_s(u, 3 * GiB))
        closes += int(v.window_closed) + int(v2.window_closed)
    assert closes == (40 - M.WARMUP_UPDATES) // M.WINDOW_UPDATES - 1
    sc = t.tb_scalars()
    assert all(k.startswith(M.TB_PREFIX) for k in sc)
    assert sc[M.TB_PREFIX + "level"] == 0.0
    assert sc[M.TB_PREFIX + "floor_mib"] == pytest.approx(3 * 1024)
    for k in ("ceiling_mib", "demand_mib", "slope_mib_per_update", "updates_to_ceiling", "segments",
              "alloc_retries", "reserved_mib", "device_free_mib", "segments_after_freeze", "sustained"):
        assert M.TB_PREFIX + k in sc


def test_alloc_retries_warn_never_stop():
    seq = [_s(u, 3 * GiB, retries=0 if u < 20 else 3) for u in range(60)]
    vs = _run(seq)
    assert "STOP" not in _levels(vs)
    assert any(any(r.startswith("alloc_retries") for r in v.reasons) for v in vs)


def test_sample_from_stats_maps_the_allocator_keys():
    stats = {"allocated_bytes.all.current": 5, "reserved_bytes.all.current": 9,
             "active_bytes.all.current": 6, "inactive_split_bytes.all.current": 2,
             "segment.all.current": 3, "segment.all.allocated": 7, "allocated_bytes.all.peak": 8,
             "reserved_bytes.all.peak": 10, "num_alloc_retries": 1, "num_ooms": 0}
    s = M.sample_from_stats(stats, 100, 200, update=4, phase="post_update")
    assert (s.allocated, s.reserved, s.active, s.inactive_split, s.segments, s.segments_allocated_total,
            s.peak_allocated, s.peak_reserved, s.alloc_retries, s.ooms, s.device_free, s.device_total) == \
        (5, 9, 6, 2, 3, 7, 8, 10, 1, 0, 100, 200)
    assert M.sample_from_stats({}, 1, 2, update=0, phase="x", peaks_valid=False).peak_reserved == -1


def test_an_even_group_size_is_refused():
    with pytest.raises(ValueError, match="ODD"):
        MemoryTrend(fit_windows=6, segments=3)


def test_module_imports_no_torch():
    import subprocess
    import sys
    code = ("import sys; import agents.training.cuda_memory_trend as m; "
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
