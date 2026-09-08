"""IS NEGATIVE CALIBRATION SKILL REAL? — the null registered by Orchestrator 1, 2026-09-07.

    python -m main.ops.negskill_null <run-name|run-dir> --from-step N
                                     [--boot 2000] [--nulls 2000]

CLAIM UNDER TEST: on the arm this was written for, 13 of 65 opponent-cycles (20%) over 20M-28M
scored skill < 0 — the win-prob head worse than a constant base-rate predictor.

🚨 A COUNT OF NEGATIVE CELLS SAYS NOTHING WITHOUT ITS NULL. At 12-20 battles per cell, a
PERFECTLY CALIBRATED critic scores negative cells by chance. The count is only evidence if it
exceeds what calibration itself produces at these n.

TWO TESTS, both from rows that already exist:
 (i)  PER-OPPONENT pooled skill over the cycles with a BATTLE-CLUSTERED bootstrap interval.
      Report how many opponents sit entirely BELOW zero (significantly worse than a constant
      predictor), entirely above, and covering.
 (ii) PARAMETRIC NULL for the cell count: for each cell, resample outcomes from THE HEAD'S OWN
      FORECASTS (y ~ Bernoulli(p)) and recompute skill. That is the distribution of
      negative-cell counts a perfectly calibrated critic with these very forecasts would
      produce at these n.

VERDICT (registered before the numbers): observed count INSIDE the null band ⇒ the claim is
WITHDRAWN. Above it ⇒ "worse than constant on K opponents, SIGNIFICANT".

Promoted 2026-09-07 from the Training Run session's ``negskill_null.py`` (the run was
positional, the 20M step floor was a literal, and the import path was a
``sys.path.insert(0, "src")`` that only worked from the repo root).
"""
from __future__ import annotations

import sys
from typing import Sequence

import numpy as np

from agents.training.scaffolding import reliability_table
from main.ops.run_ref import refuse, resolve_run_dir
from main.scaffolding_gauge import (collect_slices, opponent_class, selection_weights,
                                    true_win_rates)


def skill(p, y, w):
    if len(set(y.tolist())) < 2:
        return None
    return float(reliability_table(p, y, weights=w)["skill"])


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if args else 2
    run = resolve_run_dir(args[0])
    boot = int(args[args.index("--boot") + 1]) if "--boot" in args else 2000
    nulls = int(args[args.index("--nulls") + 1]) if "--nulls" in args else 2000
    if "--from-step" not in args:
        refuse("REFUSING: --from-step N is required. The registered read names the window it",
               "  counts over; a default floor would silently count a different one.")
    floor = int(args[args.index("--from-step") + 1])

    slices, _ = collect_slices(str(run))
    tw = true_win_rates(str(run))
    steps = [s for s in sorted(slices) if s >= floor]
    if not steps:
        print(f"  NO steps at or above {floor:,} — nothing to count. This is not a count of zero.")
        return 2
    print(f"  run {run}")
    print(f"  steps: {', '.join(f'{s/1e6:.0f}M' for s in steps)}")

    cells = []          # (opponent, step, p, y, w, battles)
    for st_ in steps:
        s = slices[st_]
        p = np.asarray(s["win_probs"], float)
        y = np.asarray(s["outcomes"], float)
        b = np.asarray(s["battles"])
        o = np.asarray(s["opponents"])
        w, _ = selection_weights(y, b, o, tw.get(int(st_), {}))
        if w is None:
            w = np.ones_like(p)
        for nm in sorted(set(o.tolist())):
            sel = o == nm
            if sel.sum() < 2 or len(set(y[sel].tolist())) < 2:
                continue
            cells.append((nm, st_, p[sel], y[sel], w[sel], b[sel]))
    if not cells:
        print("  NO usable (opponent, step) cells — nothing is concluded.")
        return 2

    obs_neg = sum(1 for c in cells if (skill(c[2], c[3], c[4]) or 0) < 0)
    print(f"\n  OBSERVED: {obs_neg} of {len(cells)} cells have skill < 0  "
          f"({obs_neg/len(cells):.0%})")

    # ---- (ii) parametric null: y ~ Bernoulli(the head's own p) ----------------------------
    rng = np.random.default_rng(0)
    null_counts = []
    for _ in range(nulls):
        n = 0
        for nm, st_, p, y, w, b in cells:
            ys = (rng.random(p.size) < p).astype(float)
            s_ = skill(p, ys, w)
            if s_ is not None and s_ < 0:
                n += 1
        null_counts.append(n)
    lo, hi = np.percentile(null_counts, [2.5, 97.5])
    med = float(np.median(null_counts))
    print(f"  NULL (perfectly calibrated critic, SAME forecasts and n): median {med:.0f} negative"
          f" cells, 95% band [{lo:.0f}, {hi:.0f}]  over {nulls} draws")
    if obs_neg > hi:
        print(f"  => {obs_neg} EXCEEDS the null band. Negative skill is MORE COMMON than calibration")
        print("     alone produces at these sample sizes.")
    elif obs_neg < lo:
        print(f"  => {obs_neg} is BELOW the null band (fewer negatives than calibration produces).")
    else:
        print(f"  => {obs_neg} sits INSIDE the null band. THE CLAIM IS WITHDRAWN: a perfectly")
        print("     calibrated critic with these forecasts scores this many negative cells by chance.")

    # ---- (i) per-opponent pooled skill, battle-clustered bootstrap ------------------------
    print("\n  PER-OPPONENT pooled over the cycles, battle-clustered bootstrap:")
    print(f"    {'opponent':<16}{'class':>6}{'btl':>6}{'skill':>9}{'[ 95% CI ]':>22}")
    below = above = covers = 0
    for nm in sorted({c[0] for c in cells}):
        mine = [c for c in cells if c[0] == nm]
        p = np.concatenate([c[2] for c in mine])
        y = np.concatenate([c[3] for c in mine])
        w = np.concatenate([c[4] for c in mine])
        b = np.concatenate([np.asarray([f"{c[1]}:{x}" for x in c[5]]) for c in mine])
        s_obs = skill(p, y, w)
        bl = np.asarray(sorted(set(b.tolist())))
        reps = []
        for _ in range(boot):
            draw = rng.choice(len(bl), size=len(bl), replace=True)
            idx = np.concatenate([np.flatnonzero(b == bl[i]) for i in draw])
            v = skill(p[idx], y[idx], w[idx])
            if v is not None:
                reps.append(v)
        if len(reps) < 50:
            print(f"    {nm:<16}{opponent_class(nm):>6}{len(bl):>6}{s_obs:>9.3f}"
                  "   too few replicates")
            continue
        l2, h2 = np.percentile(reps, [2.5, 97.5])
        tag = ""
        if h2 < 0:
            tag = "   <- ENTIRELY BELOW ZERO"
            below += 1
        elif l2 > 0:
            above += 1
        else:
            covers += 1
        print(f"    {nm:<16}{opponent_class(nm):>6}{len(bl):>6}{s_obs:>9.3f}"
              f"   [{l2:+.3f}, {h2:+.3f}]{tag}")
    print(f"\n    entirely BELOW zero: {below}   entirely ABOVE: {above}   COVERING zero: {covers}")
    if below:
        print(f"    => worse than a constant predictor on {below} opponent(s), SIGNIFICANT.")
    else:
        print("    => NO opponent is significantly worse than a constant predictor once pooled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
