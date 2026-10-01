"""CUDA MEMORY LEAK DETECTOR — the learner's DECLARED LIFECYCLE, memory half (M5 Lane K6,
`gen3_cuda_memory_trend_v1`; `designs/training/learner_gates.md` "K6 — the CUDA memory trend").

**What it is for.** A clean EARLY STOP, not a correctness gate. A slow leak of live CUDA tensors in a
training process ends, hours later, in an out-of-memory error in the middle of an update — no
checkpoint, and a launcher restart that replays the same leak (a crash loop). This module watches
the trainer's own card once per PPO update and says OK / WARN / STOP. It says STOP only when a
SUSTAINED growth trend PROJECTS the out-of-memory inside the declared HORIZON, so the integrator can
checkpoint and exit with a typed, non-restartable status. Everything else — a one-off step-up, a
fragmentation plateau, a bounded growth that levels off — is at most a WARN and never stops a run.

**The two halves.**

* ``sample_cuda(device, update=, phase=)`` reads ``torch.cuda.memory_stats(device)`` (allocated,
  reserved, active and inactive-split bytes; the segment count; the cumulative cudaMalloc count;
  ``num_alloc_retries``; ``num_ooms``; the peaks since the previous sample, which it then resets)
  and ``torch.cuda.mem_get_info`` (device-wide free / total — other processes share the card) into
  a plain ``MemorySample``. The only torch-touching code here; it imports torch lazily.
* ``MemoryTrend`` is PURE (no torch): ``observe(sample)`` folds one sample and returns a
  ``TrendVerdict`` carrying the level, a one-line human message and the projection numbers.

**Which statistic, and why.** Two quantities are easy to confuse:

* RESERVED bytes (what the caching allocator holds from the driver) move in STEPS and never come
  back down without ``empty_cache``: a new block size, a fragmented free list or a rare large
  transient grows it once and it stays. A trend fit on reserved reads every fragmentation step-up
  as growth. Reserved is therefore NOT the leak signal; it enters only as the run's real DEMAND
  (the peak it needed, fragmentation included) and through the device-wide CAPACITY.
* ALLOCATED bytes at a QUIESCENT point (after ``train()`` returned, or after a rollout block — no
  update's temporaries alive) are the live tensors the process keeps between updates: weights,
  optimizer state, grads, buffers, caches. Fragmentation does not move them. A leak of live tensors
  moves them by the leaked bytes every update, forever.

So the leak signal is the per-WINDOW FLOOR = the minimum quiescent allocated bytes over a window of
``WINDOW_UPDATES`` updates (a min, so a sample that happens to catch a lingering transient cannot
raise it). The rule over the last ``FIT_WINDOWS`` floors:

1. SUSTAINED = the span splits into ``SEGMENTS`` consecutive groups of ``FIT_WINDOWS // SEGMENTS``
   (odd) windows; each group's MEDIAN floor exceeds the previous group's by more than
   ``NOISE_BAR_BYTES``, in EVERY consecutive pair, AND the Theil-Sen slope of floor vs update over
   the span is at least ``MIN_SLOPE_BYTES_PER_UPDATE``. A single step-up anywhere raises at most
   one group median (a median of an odd group is the majority side of a step), so it can never be
   sustained; growth that levels off leaves the last group flat. Theil-Sen (the median of pairwise
   slopes) because a single outlier window cannot move it.
2. PROJECTION = demand now (the max PEAK reserved over the span — what the process actually needed,
   fragmentation included) plus slope x updates, against the CEILING = reserved now + device free
   now (from ``mem_get_info``, so another process on the card shrinks it) minus ``CEILING_MARGIN``.
   ``updates_to_ceiling = (ceiling - demand) / slope``.
3. STOP iff SUSTAINED and ``updates_to_ceiling <= horizon`` (``HORIZON_UPDATES`` unless the
   integrator passes its own) on ``PERSIST_WINDOWS`` CONSECUTIVE window closes — a ramp of a few
   windows (a lazily-built cache that then stops) can line the three medians up for as many
   closes as it has ramp windows; a leak holds them on every close. Until then, and for a
   sustained trend beyond the horizon, it is a WARN carrying the projection, so a slow leak is
   visible hours before it matters.

WARN (never STOP) also for: a one-off floor step-up larger than ``STEP_WARN_BYTES`` between
consecutive windows; a RESERVED step-up above ``RESERVED_STEP_WARN_BYTES`` (a fragmentation step;
smaller ones, measured at 4-120 MiB on a process's first diagnostics updates, are only counted in
``segments_after_freeze`` — the lifecycle's after-freeze counter, a TB scalar, never enforced
here; it reads 0 under ``expandable_segments:True``, whose segments ``memory_stats`` does not
count); a new ``num_alloc_retries`` (the allocator hit the ceiling and flushed its cache to satisfy
a request — the OOM precursor); a new ``num_ooms``; and headroom (ceiling - demand) under
``CEILING_MARGIN_BYTES`` with no trend.

The first ``WARMUP_UPDATES`` updates of a PROCESS are never windowed: iteration one acquires state
lazily (Adam's moment buffers at the first optimizer step, per-process probe state) — the step-up a
declared lifecycle expects at startup.

**Calibration.** Every constant below is from a MEASURED healthy learner (production shape, the
caching allocator's default and ``expandable_segments:True``), with the false-trip estimate and its
derivation in ``designs/research_state/measurements/k6_k8/memory/README.md``.

**Integration** (the K6 worker wires it; this module never stops anything itself): build one
``MemoryTrend`` per process after the startup freeze; after every ``train()`` call
``v = trend.observe(sample_cuda(device, update=n_updates, phase="post_update"))`` (optionally also
after each rollout block with ``phase="post_rollout"`` and the same update count); when
``v.window_closed`` log ``trend.tb_scalars()``; when ``v.level == "WARN"`` and ``v.changed`` emit
``v.message``; when ``v.level == "STOP"`` checkpoint and raise the typed stop with ``v.message``.
"""
from __future__ import annotations

import dataclasses
import math
import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "gen3_cuda_memory_trend_v1"

MiB = 1 << 20
GiB = 1 << 30

# ------------------------------------------------------------------------------------------------
# DECLARED constants (calibration: designs/research_state/measurements/k6_k8/memory/README.md)
# ------------------------------------------------------------------------------------------------

#: Updates per window. A window's floor is the min quiescent allocated over its samples.
WINDOW_UPDATES = 2
#: Windows in the fit span (a multiple of SEGMENTS, with an ODD group size).
FIT_WINDOWS = 9
#: Consecutive groups the span splits into; each must rise over the previous by NOISE_BAR_BYTES.
SEGMENTS = 3
#: Updates of a process that are never windowed (iteration one's lazy acquisitions).
WARMUP_UPDATES = 2
#: A group-median rise at or under this is noise.
NOISE_BAR_BYTES = 8 * MiB
#: The span's Theil-Sen slope must reach this for a trend to count as sustained.
MIN_SLOPE_BYTES_PER_UPDATE = 1 * MiB
#: STOP needs the stop condition on this many CONSECUTIVE window closes. A short ramp (a lazily
#: built cache, a one-off block) can line up the three group medians for as many closes as it has
#: ramp windows; a leak holds the condition on every close.
PERSIST_WINDOWS = 3
#: A floor rise between consecutive windows above this is a one-off STEP-UP (WARN).
STEP_WARN_BYTES = 64 * MiB
#: A RESERVED rise between consecutive windows above this is a fragmentation step (WARN). Measured
#: healthy steps after the warm-up: +4 / +16 MiB (default allocator), +36 / +120 MiB
#: (`expandable_segments:True`), each on a diagnostics update; 256 MiB is 2.1x the largest.
RESERVED_STEP_WARN_BYTES = 256 * MiB
#: Projected OOM inside this many updates => STOP. 25 = the default periodic-checkpoint cadence at
#: production shape (50,000 vec calls x 48 envs = 2.4M env steps = 24.4 rollouts of 98,304): a
#: crossing projected before the NEXT checkpoint is the one a clean stop must pre-empt.
HORIZON_UPDATES = 25
#: Kept free below the device ceiling (the CUDA context, cuBLAS workspaces, a late large block).
CEILING_MARGIN_BYTES = 512 * MiB

LEVELS = ("OK", "WARN", "STOP")
TB_PREFIX = "lifecycle/cuda_"


# ------------------------------------------------------------------------------------------------
# The sample
# ------------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MemorySample:
    """One reading of the trainer's card. ``update`` = updates COMPLETED when it was taken. Peaks
    are since the previous sample (-1 when unknown, e.g. the caller does not reset them)."""

    update: int
    phase: str
    allocated: int
    reserved: int
    active: int
    inactive_split: int
    segments: int
    segments_allocated_total: int
    peak_allocated: int
    peak_reserved: int
    alloc_retries: int
    ooms: int
    device_free: int
    device_total: int

    def as_row(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def sample_from_stats(stats: Mapping[str, Any], free: int, total: int, *, update: int, phase: str,
                      peaks_valid: bool = True) -> MemorySample:
    """``torch.cuda.memory_stats()`` + ``mem_get_info()`` -> ``MemorySample`` (pure)."""

    def g(k: str) -> int:
        return int(stats.get(k, 0) or 0)

    return MemorySample(
        update=int(update), phase=str(phase),
        allocated=g("allocated_bytes.all.current"), reserved=g("reserved_bytes.all.current"),
        active=g("active_bytes.all.current"), inactive_split=g("inactive_split_bytes.all.current"),
        segments=g("segment.all.current"), segments_allocated_total=g("segment.all.allocated"),
        peak_allocated=g("allocated_bytes.all.peak") if peaks_valid else -1,
        peak_reserved=g("reserved_bytes.all.peak") if peaks_valid else -1,
        alloc_retries=g("num_alloc_retries"), ooms=g("num_ooms"),
        device_free=int(free), device_total=int(total))


def sample_cuda(device: Any, *, update: int, phase: str, reset_peak: bool = True) -> MemorySample:
    """Read the card now. With ``reset_peak`` the peak counters are reset AFTER the read, so the next
    sample's peaks cover exactly the interval between the two (a production call site resets; a
    process that never sampled before reports peaks since its start). No synchronize: the caching
    allocator's counters are host-side and current as of the last call that allocated or freed."""
    import torch

    dev = torch.device(device)
    stats = torch.cuda.memory_stats(dev)
    free, total = torch.cuda.mem_get_info(dev)
    s = sample_from_stats(stats, free, total, update=update, phase=phase)
    if reset_peak:
        torch.cuda.reset_peak_memory_stats(dev)
    return s


# ------------------------------------------------------------------------------------------------
# The verdict
# ------------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TrendVerdict:
    level: str
    message: str
    update: int
    windows: int
    window_closed: bool
    changed: bool
    sustained: bool
    slope_bytes_per_update: Optional[float]
    floor_bytes: Optional[int]
    demand_bytes: Optional[int]
    ceiling_bytes: Optional[int]
    projected_demand_at_horizon: Optional[float]
    updates_to_ceiling: Optional[float]
    horizon_updates: int
    reasons: Tuple[str, ...] = ()


@dataclasses.dataclass
class _Window:
    index: int
    floor: int
    peak_demand: int
    reserved_last: int
    device_free_last: int
    segments_max: int
    retries_last: int
    ooms_last: int
    n: int

    @property
    def center(self) -> float:
        return float(self.index)


def theil_sen(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Median of pairwise slopes (0.0 for fewer than two distinct x)."""
    sl = [(ys[j] - ys[i]) / (xs[j] - xs[i])
          for i in range(len(xs)) for j in range(i + 1, len(xs)) if xs[j] != xs[i]]
    return float(statistics.median(sl)) if sl else 0.0


def _fmt_b(b: Optional[float]) -> str:
    if b is None:
        return "n/a"
    return f"{b / GiB:.2f} GiB" if abs(b) >= GiB else f"{b / MiB:.1f} MiB"


class MemoryTrend:
    """Folds per-update ``MemorySample``s into window floors and judges the trend (module docs)."""

    def __init__(self, *, window_updates: int = WINDOW_UPDATES, fit_windows: int = FIT_WINDOWS,
                 segments: int = SEGMENTS, warmup_updates: int = WARMUP_UPDATES,
                 noise_bar_bytes: int = NOISE_BAR_BYTES,
                 min_slope_bytes_per_update: float = MIN_SLOPE_BYTES_PER_UPDATE,
                 persist_windows: int = PERSIST_WINDOWS,
                 step_warn_bytes: int = STEP_WARN_BYTES,
                 reserved_step_warn_bytes: int = RESERVED_STEP_WARN_BYTES,
                 horizon_updates: int = HORIZON_UPDATES,
                 ceiling_margin_bytes: int = CEILING_MARGIN_BYTES):
        if window_updates < 1 or fit_windows < segments or segments < 2:
            raise ValueError("need window_updates >= 1 and fit_windows >= segments >= 2")
        if fit_windows % segments or (fit_windows // segments) % 2 == 0:
            raise ValueError(f"fit_windows ({fit_windows}) must split into {segments} groups of an ODD "
                             "size: a median of an even group averages the two sides of a step-up")
        self.window_updates = int(window_updates)
        self.fit_windows = int(fit_windows)
        self.segments = int(segments)
        self.warmup_updates = int(warmup_updates)
        self.noise_bar = int(noise_bar_bytes)
        self.min_slope = float(min_slope_bytes_per_update)
        self.persist = max(1, int(persist_windows))
        self._stop_run = 0
        self.step_warn = int(step_warn_bytes)
        self.reserved_step_warn = int(reserved_step_warn_bytes)
        self.horizon = int(horizon_updates)
        self.margin = int(ceiling_margin_bytes)
        self.windows: List[_Window] = []
        self._cur: Optional[_Window] = None
        self._first: Optional[MemorySample] = None
        self._segments_at_freeze: Optional[int] = None
        self._last: Optional[TrendVerdict] = None
        self.samples = 0
        self.stops = 0
        self.warns = 0

    # ---------------------------------------------------------------- folding
    def _window_of(self, update: int) -> Optional[int]:
        if update < self.warmup_updates:
            return None
        return (update - self.warmup_updates) // self.window_updates

    def observe(self, s: MemorySample, *, horizon_updates: Optional[int] = None) -> TrendVerdict:
        """Fold one sample; recompute the verdict when a window closes (else repeat the last one,
        with ``window_closed=False``)."""
        self.samples += 1
        if self._first is None:
            self._first = s
        w = self._window_of(int(s.update))
        closed = False
        if w is not None:
            if self._segments_at_freeze is None:
                self._segments_at_freeze = int(s.segments)
            if self._cur is not None and w != self._cur.index:
                self.windows.append(self._cur)
                self._cur = None
                closed = True
            demand = s.peak_reserved if s.peak_reserved >= 0 else s.reserved
            if self._cur is None:
                self._cur = _Window(index=w, floor=s.allocated, peak_demand=demand,
                                    reserved_last=s.reserved, device_free_last=s.device_free,
                                    segments_max=s.segments, retries_last=s.alloc_retries,
                                    ooms_last=s.ooms, n=1)
            else:
                c = self._cur
                c.floor = min(c.floor, s.allocated)
                c.peak_demand = max(c.peak_demand, demand)
                c.reserved_last, c.device_free_last = s.reserved, s.device_free
                c.segments_max = max(c.segments_max, s.segments)
                c.retries_last, c.ooms_last = s.alloc_retries, s.ooms
                c.n += 1
        if closed or self._last is None:
            v = self._judge(int(s.update), closed,
                            self.horizon if horizon_updates is None else int(horizon_updates))
        else:
            v = dataclasses.replace(self._last, update=int(s.update), window_closed=False, changed=False)
        self._last = v
        return v

    # ---------------------------------------------------------------- judging
    def _judge(self, update: int, closed: bool, horizon: int) -> TrendVerdict:
        prev = self._last.level if self._last is not None else None
        ws = self.windows
        reasons: List[str] = []
        base = dict(update=update, windows=len(ws), window_closed=closed, horizon_updates=horizon)
        if not ws:
            return TrendVerdict(level="OK", message="[CudaMemTrend] warming up (no closed window yet)",
                                changed=prev not in (None, "OK"), sustained=False,
                                slope_bytes_per_update=None, floor_bytes=None, demand_bytes=None,
                                ceiling_bytes=None, projected_demand_at_horizon=None,
                                updates_to_ceiling=None, **base)
        last = ws[-1]
        span = ws[-self.fit_windows:]
        demand = max(x.peak_demand for x in span)
        ceiling = last.reserved_last + last.device_free_last - self.margin
        slope: Optional[float] = None
        sustained = False
        if len(span) == self.fit_windows:
            xs = [x.center * self.window_updates for x in span]          # in updates
            ys = [float(x.floor) for x in span]
            slope = theil_sen(xs, ys)
            g = self.fit_windows // self.segments
            meds = [statistics.median(ys[i * g:(i + 1) * g]) for i in range(self.segments)]
            rises = [b - a for a, b in zip(meds, meds[1:])]
            sustained = all(r > self.noise_bar for r in rises) and slope >= self.min_slope
        to_ceiling: Optional[float] = None
        projected: Optional[float] = None
        if slope is not None and slope > 0:
            to_ceiling = max(0.0, (ceiling - demand) / slope)
            projected = demand + slope * horizon
        level = "OK"
        inside = sustained and to_ceiling is not None and to_ceiling <= horizon
        if closed:
            self._stop_run = self._stop_run + 1 if inside else 0
        if sustained:
            assert slope is not None and to_ceiling is not None
            if inside and self._stop_run >= self.persist:
                level = "STOP"
                reasons.append("sustained_growth_projects_oom_inside_horizon")
            elif inside:
                level = "WARN"
                reasons.append(f"sustained_growth_projects_oom_inside_horizon "
                               f"({self._stop_run}/{self.persist} windows)")
            else:
                level = "WARN"
                reasons.append("sustained_growth_beyond_horizon")
        if len(ws) >= 2 and ws[-1].floor - ws[-2].floor > self.step_warn:
            reasons.append(f"step_up {_fmt_b(ws[-1].floor - ws[-2].floor)}")
        if len(ws) >= 2 and last.retries_last > ws[-2].retries_last:
            reasons.append(f"alloc_retries +{last.retries_last - ws[-2].retries_last}")
        if len(ws) >= 2 and last.ooms_last > ws[-2].ooms_last:
            reasons.append(f"ooms +{last.ooms_last - ws[-2].ooms_last}")
        if len(ws) >= 2 and last.reserved_last - ws[-2].reserved_last > self.reserved_step_warn:
            reasons.append(f"reserved_step {_fmt_b(last.reserved_last - ws[-2].reserved_last)} "
                           f"(segments +{last.segments_max - ws[-2].segments_max}, "
                           f"{self.segments_after_freeze()} after freeze)")
        if ceiling - demand < self.margin:
            reasons.append(f"headroom {_fmt_b(ceiling - demand)} < margin {_fmt_b(self.margin)}")
        if level == "OK" and reasons:
            level = "WARN"
        msg = self._message(level, update, slope, last.floor, demand, ceiling, projected, to_ceiling,
                            horizon, reasons)
        if level == "STOP":
            self.stops += 1
        elif level == "WARN":
            self.warns += 1
        return TrendVerdict(level=level, message=msg, changed=level != prev, sustained=sustained,
                            slope_bytes_per_update=slope, floor_bytes=last.floor, demand_bytes=demand,
                            ceiling_bytes=ceiling, projected_demand_at_horizon=projected,
                            updates_to_ceiling=to_ceiling, reasons=tuple(reasons), **base)

    def _message(self, level: str, update: int, slope: Optional[float], floor: int, demand: int,
                 ceiling: int, projected: Optional[float], to_ceiling: Optional[float], horizon: int,
                 reasons: List[str]) -> str:
        head = f"[CudaMemTrend] {level} @ update {update}:"
        proj = (f"floor {_fmt_b(floor)}, slope {_fmt_b(slope)}/update, demand {_fmt_b(demand)}, "
                f"ceiling {_fmt_b(ceiling)}")
        if to_ceiling is not None:
            proj += f", projected OOM in {to_ceiling:.0f} updates (horizon {horizon})"
        if level == "STOP":
            return (f"{head} a SUSTAINED growth of live CUDA memory projects an out-of-memory inside the "
                    f"horizon — stopping cleanly with a checkpoint (a restart would replay the leak). "
                    f"{proj}")
        if reasons:
            return f"{head} {'; '.join(reasons)} — {proj}"
        return f"{head} {proj}"

    # ---------------------------------------------------------------- outputs
    @property
    def last_verdict(self) -> Optional[TrendVerdict]:
        return self._last

    def segments_after_freeze(self) -> int:
        """Segments acquired since the first windowed sample (the lifecycle's after-freeze counter;
        a fragmentation step-up shows here and is REPORTED, never a stop)."""
        if self._segments_at_freeze is None or not self.windows:
            return 0
        return max(0, max(w.segments_max for w in self.windows) - self._segments_at_freeze)

    def tb_scalars(self) -> Dict[str, float]:
        """The per-window TB dict (``lifecycle/cuda_*``). Absent numbers are -1 so a reader can tell
        "no projection" from zero."""
        v = self._last
        out: Dict[str, float] = {}
        if v is None:
            return out
        lw = self.windows[-1] if self.windows else None

        def f(x: Optional[float]) -> float:
            return -1.0 if x is None else float(x)

        out[TB_PREFIX + "level"] = float(LEVELS.index(v.level))
        out[TB_PREFIX + "floor_mib"] = f(None if v.floor_bytes is None else v.floor_bytes / MiB)
        out[TB_PREFIX + "demand_mib"] = f(None if v.demand_bytes is None else v.demand_bytes / MiB)
        out[TB_PREFIX + "ceiling_mib"] = f(None if v.ceiling_bytes is None else v.ceiling_bytes / MiB)
        out[TB_PREFIX + "slope_mib_per_update"] = f(None if v.slope_bytes_per_update is None
                                                    else v.slope_bytes_per_update / MiB)
        out[TB_PREFIX + "updates_to_ceiling"] = (f(v.updates_to_ceiling)
                                                 if v.updates_to_ceiling is None or
                                                 math.isfinite(v.updates_to_ceiling) else 1e9)
        out[TB_PREFIX + "sustained"] = 1.0 if v.sustained else 0.0
        out[TB_PREFIX + "segments_after_freeze"] = float(self.segments_after_freeze())
        if lw is not None:
            out[TB_PREFIX + "reserved_mib"] = lw.reserved_last / MiB
            out[TB_PREFIX + "device_free_mib"] = lw.device_free_last / MiB
            out[TB_PREFIX + "segments"] = float(lw.segments_max)
            out[TB_PREFIX + "alloc_retries"] = float(lw.retries_last)
            out[TB_PREFIX + "ooms"] = float(lw.ooms_last)
        return out

    def stop_message(self) -> Optional[str]:
        """The STOP message text, when the last verdict is STOP."""
        v = self._last
        return v.message if v is not None and v.level == "STOP" else None
