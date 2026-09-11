"""Read a live run's instruments from the TENSORBOARD EVENTS, not the child log's table.

    python -m main.ops.tb_read <run-name|run-dir> [--last N] [--tag TAG]...

**Why this exists, and what it already caught.** The child log's table is a RENDERING. It drops
the scalar's group prefix — ``grad/value_policy_logratio`` prints as a bare
``value_policy_logratio`` — and it is subject to ``--log-level``'s periodic printing, so it
UNDERSAMPLES the rollouts. Reading it produced two confident wrong readings on the live arm: an
episode length reported as "falling" that was in fact oscillating, and a metric read under the
wrong group. The events carry every rollout with its step.

**ABSENCE IS NOT A ZERO.** A tag with no scalar prints ``NO SCALAR FOUND`` and the run exits
non-zero. It never defaults to 0, and it never quietly omits the row — a missing instrument and
an instrument reading zero are different states of the world and only one of them is a
measurement.

**A REGIME MARKER IS NOT A NOISY SERIES.** ``train/selfplay_fraction`` is a step function: the
median ACROSS regimes describes neither of them (a 2-point series read 0.45, a value the run has
never been at). It is reported as the CURRENT value plus the change list, never as a central
tendency. It rides in the default tag set because a THROUGHPUT number on a self-play arm is a
FUNCTION of it — it went 0.0 -> 0.9 at the first pool promotion and the rollout cadence fell
107 s -> 163 s (919 -> 605 fps) with zero compile in the window. An fps figure quoted without it
is not interpretable.

Promoted 2026-09-07 from the Training Run session's ``tb_read.py``.
"""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from main.ops.run_ref import event_dirs, resolve_run_dir

#: The default read. `train/selfplay_fraction` leads because every throughput figure below is a
#: function of it; the rest are the live win-prob arm's registered instruments.
TAGS: List[str] = [
    "train/selfplay_fraction",
    "grad/value_policy_logratio",
    "signal/draw_rate",
    "rollout/ep_len_mean",
    "reward/untracked_abs_mean",
    "win_prob/critic_resolution",
    "win_prob/critic_reliability",
    "win_prob/critic_skill",
    "win_prob/critic_ece",
    "win_prob/critic_brier",
]

#: The registered calibration-cost bar (Orchestrator 1, 2026-09-06): a sharpening critic buys
#: resolution and pays a little reliability, and that is the EXPECTED shape. It becomes a finding
#: only when reliability exceeds this share of resolution. Encoded rather than remembered, so the
#: ratio is computed at every read.
CALIBRATION_COST_BAR = 0.10

Series = Dict[str, List[Tuple[int, float]]]


def load(run_dir: Path, tags: Sequence[str] = tuple(TAGS)) -> Series:
    """Every ``(step, value)`` for each tag, across all of the run's event directories."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    series: Series = {t: [] for t in tags}
    for d in event_dirs(run_dir):
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        avail = set(ea.Tags().get("scalars", []))
        for t in tags:
            if t in avail:
                series[t] += [(s.step, s.value) for s in ea.Scalars(t)]
    for t in series:
        series[t].sort(key=lambda x: x[0])
    return series


#: the two scalars that DEFINE the self-play crossing. `eval/pool_snapshot_count` is the count of
#: promoted snapshots the pool holds; `train/selfplay_fraction` is the share of episodes drawn
#: against one. Either moving off zero means the run has started facing itself.
CROSSING_TAGS: Tuple[str, str] = ("eval/pool_snapshot_count", "train/selfplay_fraction")


def selfplay_crossing_step(run_dir: Path) -> Dict[str, object]:
    """The first step at which a run carries a POOL opponent at all — read-only, run dir untouched.

    🚨 **WHY A READ NEEDS THIS.** The first crossing is a draw-level coin flip. It is the SEED
    path, not the promotion line (ledger 2026-09-11 CORRECTION): with an empty pool
    `win_rate_vs_pool` is 0.0, so `selfplay_callback.py:651` seeds the frozen eval snapshot as soon
    as `heuristic_fraction(win_rate_vs_bots, start=SELF_PLAY_START)` leaves zero — the deciding
    quantity is **BOTS against `SELF_PLAY_START = 0.55`** (`--self-play-start-wr`), and
    `--promote-threshold` (a different constant that coincides numerically at 0.55) governs only
    LATER promotions. 0.55 sits inside the 2M-bots replicate floor (roster 0.37-0.58), so two
    identically-configured arms cross a whole restart interval apart: six of eight 10M arms at
    **4,128,768**, two (`lambda09`, `lambda095`) at **2,162,688** — the λ-0.9 replicate pair is
    itself mismatched. ⚠️ **The mismatch is a RAMP, not a step**: the self-play fraction smoothsteps
    from `start` to `SELF_PLAY_FULL = 0.80`, so an early crosser spends its extra 2M steps at a
    fraction rising from ~0.03 (`lambda09` 0.0303), not at 0.9. The field is DESCRIPTIVE and
    selects nothing — the observed crossing is a POST-TREATMENT variable that may be a mediator of
    the lever under test, so a read states it and never picks a comparator by it.

    🚨 **TWO DEFINITIONS, AND THEY ARE NOT THE SAME STEP — BOTH ARE REPORTED.** ``step`` is the
    registered one: the first step carrying ANY ``*_pool`` scalar, i.e. the first time a pool
    opponent actually appeared in a rollout. ``promotion_step`` is the first step at which
    ``eval/pool_snapshot_count`` or ``train/selfplay_fraction`` rose above zero — the PROMOTION
    itself, which is logged at the EVAL step that decided it. The promotion lands one eval→rollout
    lag earlier: measured on the 10M ladder, 2,000,016 vs **2,162,688** and 4,000,032 vs
    **4,128,768**. Quoting one as the other moves the boundary by ~130-160k steps, so the field
    carries both and names which is which.

    Returns ``{"step": int | None, "promotion_step": int | None, "by_tag": {tag: step | None},
    "note": …}``. DESCRIPTIVE: no floor, no verdict. ``None`` means the run never logged a pool
    scalar — which is what a pure-bot arm looks like, and is reported as such rather than as a
    zero.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    by_tag: Dict[str, object] = {t: None for t in CROSSING_TAGS}
    first_pool = None
    for d in event_dirs(run_dir):
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        avail = set(ea.Tags().get("scalars", []))
        for t in CROSSING_TAGS:
            if t not in avail:
                continue
            hits = [int(sc.step) for sc in ea.Scalars(t) if sc.value > 0]
            if hits:
                cur = by_tag[t]
                by_tag[t] = min(hits) if cur is None else min(int(cur), min(hits))
        # the REGISTERED definition: any `*_pool` scalar existing at all, which is the first
        # rollout that actually contained a pool opponent. It lands AFTER the promotion scalar
        # above, by one eval->rollout lag; both are returned because they are different events.
        for t in avail:
            if not (t.endswith("_pool") or t.endswith("_vs_pool")):
                continue
            steps = [int(sc.step) for sc in ea.Scalars(t)]
            if steps:
                m = min(steps)
                first_pool = m if first_pool is None else min(first_pool, m)
    live = [int(v) for v in by_tag.values() if v is not None]
    return {"step": first_pool,
            "promotion_step": min(live) if live else None,
            "by_tag": by_tag,
            "note": ("`step` = the first step carrying any `*_pool` scalar (the first pool "
                     "opponent in a rollout); `promotion_step` = the first step at which "
                     "`eval/pool_snapshot_count` or `train/selfplay_fraction` rose above zero "
                     "(the promotion, logged at the EVAL step that decided it, one eval->rollout "
                     "lag earlier). DESCRIPTIVE: no floor and no verdict.")}


def report(run_dir: Path, series: Series, tags: Sequence[str], n: int) -> int:
    """Print the read. Returns the number of tags that had NO SCALAR — 0 means a full read."""
    missing = 0
    for t in tags:
        v = series[t]
        if not v:
            missing += 1
            print(f"{t}: NO SCALAR FOUND in {run_dir}/tb  <- absence is not a zero")
            continue
        last = [x[1] for x in v[-n:]]
        if t == "train/selfplay_fraction":
            # A STEP FUNCTION, not a noisy series: a regime marker. Report the current regime
            # and when it changed; never a central tendency.
            print(f"\n{t}   <- regime marker, read the CURRENT value")
            print(f"  CURRENT {v[-1][1]:.2f} since step "
                  f"{next(s0 for s0, x in v if x == v[-1][1]):,}")
            print("  changes: " + " -> ".join(
                f"{x:.2f}@{s0:,}" for i, (s0, x) in enumerate(v)
                if i == 0 or x != v[i - 1][1]))
            continue
        print(f"\n{t}")
        print(f"  points {len(v)}   steps {v[0][0]:,} .. {v[-1][0]:,}   (last {len(last)})")
        print(f"  median {st.median(last):+.4f}   min {min(last):+.4f}   max {max(last):+.4f}"
              f"   last {last[-1]:+.4f}")

    v = [x[1] for x in series.get("grad/value_policy_logratio", [])]
    if v:
        last = v[-n:]
        med = st.median(last)
        a = abs(med)
        print("\n=== vf_coef VERDICT (ledger cfc72ad0; log10, median of last %d) ===" % len(last))
        print(f"  median {med:+.3f} log10 = {10**med:.2f}x (value/policy grad norm)")
        print(f"  window spans {min(last):+.3f} .. {max(last):+.3f} log10"
              f"  ({10**max(last)/10**min(last):.1f}x end to end)")
        if a <= 0.5:
            print(f"  KEEP vf_coef 0.5   (|{med:+.3f}| <= 0.5)")
        elif a < 1.0:
            print(f"  KEEP + FLAG, re-read at 2nd restart   (0.5 < |{med:+.3f}| < 1.0)")
        else:
            print(f"  NEW ARM: --vf-coef {0.5*(10**-med):.4g}   (|{med:+.3f}| >= 1.0)")
            print("  (vf_coef is resume-immutable — it cannot be changed on this run)")

    # --- CALIBRATION-COST CHECK (Orchestrator 1, 2026-09-06) --------------------------------
    rel = [x[1] for x in series.get("win_prob/critic_reliability", [])]
    res = [x[1] for x in series.get("win_prob/critic_resolution", [])]
    if rel and res:
        r_med, s_med = st.median(rel[-n:]), st.median(res[-n:])
        print("\n=== CALIBRATION COST (reliability as a share of resolution) ===")
        if s_med <= 0:
            print("  resolution <= 0 -- ratio undefined, NOT read as clean")
        else:
            frac = r_med / s_med
            print(f"  reliability {r_med:.4f} / resolution {s_med:.4f} = {frac:.1%} of resolution")
            if frac > CALIBRATION_COST_BAR:
                print(f"  FLAG: {frac:.1%} > {CALIBRATION_COST_BAR:.0%} -- the head is buying "
                      "discrimination with")
                print("  miscalibration. Report it; do not act unilaterally.")
            else:
                print(f"  ok ({frac:.1%} <= {CALIBRATION_COST_BAR:.0%}) -- expected shape of a "
                      "sharpening critic")
    else:
        print("\n=== CALIBRATION COST ===\n  NO SCALAR for reliability and/or resolution"
              "  <- absence is not a zero")
    return missing


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if args else 2
    run_dir = resolve_run_dir(args[0])
    n = 20
    if "--last" in args:
        n = int(args[args.index("--last") + 1])
    tags = [args[i + 1] for i, a in enumerate(args) if a == "--tag"] or list(TAGS)
    missing = report(run_dir, load(run_dir, tags), tags, n)
    if missing:
        # A tag that is not there is a FAILED read, not a clean one. Say so in the exit status
        # so a caller in a chain cannot mistake "nothing found" for "nothing wrong".
        print(f"\n{missing} of {len(tags)} requested tag(s) had NO SCALAR — this read is INCOMPLETE.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
