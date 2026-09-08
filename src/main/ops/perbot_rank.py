"""THE CAP-ACCOUNT ORDERING TEST — registered by Orchestrator 1 on 2026-09-07, before 28M's rows.

    python -m main.ops.perbot_rank <run-name|run-dir> [--steps A B ...] [--from-step N]
                                   [--boot 2000]

THE CAP ACCOUNT IS AN ORDERING CLAIM: calibration skill should FALL as the base rate RISES,
because the resolution cap base*(1-base) shrinks where the arm dominates. So the statistic is
SPEARMAN rho, not Pearson, and the prediction is ONE-SIDED: the cap account predicts rho < 0.

DESIGN (registered):
  unit        (opponent, step) over the selected steps — cycles x opponents, sentinels by name
  statistic   Spearman rho(base, skill)
  bootstrap   clustered by OPPONENT (resample opponents with replacement, carrying ALL of that
              opponent's step-rows) so repeated measures across steps are respected
  verdict     interval clear of zero, positive  -> "ordering points AWAY from the cap account,
                                                   SIGNIFICANT"
              covers zero                       -> "NOT DETECTED"
              clear of zero, negative           -> the cap account is SUPPORTED

🚨 WHY NOT MORE STEPS = MORE POWER (the structural point): pooling steps tightens each
opponent's skill estimate, but a correlation's n is the number of INDEPENDENT UNITS. Treating
steps as units raises n only if the clustered bootstrap keeps the repeated measures honest,
which is what clustering by opponent does. Do not read 40 points as 40 independent ones.

🚨 A CONFIRMED SIGN IS "AGAINST THE CAP ACCOUNT", NEVER "THE HEAD IS BLUR-LIMITED". Low skill
on contested opponents is equally predicted by an IRREDUCIBLE hidden-information floor. Only
the identity test's sd_true_excess separates the two.

``--from-step N`` selects every available step at or above N; the session copy had that floor
written in as ``20_000_000``. Give one or the other; the tool refuses to pick a pool for you.

Promoted 2026-09-07 from the Training Run session's ``perbot_rank.py``.
"""
from __future__ import annotations

import math
import sys
from typing import Sequence

import numpy as np

from agents.training.scaffolding import reliability_table
from main.ops.run_ref import refuse, resolve_run_dir
from main.scaffolding_gauge import (collect_slices, opponent_class, selection_weights,
                                    true_win_rates)


def spearman(xs, ys) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((c - my) ** 2 for c in ry))
    return (sum((a - mx) * (c - my) for a, c in zip(rx, ry)) / den) if den else float("nan")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if args else 2
    run = resolve_run_dir(args[0])
    boot = int(args[args.index("--boot") + 1]) if "--boot" in args else 2000

    slices, _meta = collect_slices(str(run))
    assert isinstance(slices, dict)
    tw = true_win_rates(str(run))
    avail = sorted(slices)

    if "--steps" in args:
        i = args.index("--steps") + 1
        steps = []
        while i < len(args) and not args[i].startswith("--"):
            steps.append(int(args[i]))
            i += 1
    elif "--from-step" in args:
        floor = int(args[args.index("--from-step") + 1])
        steps = [s for s in avail if s >= floor]
    else:
        refuse("REFUSING: give --steps A B ... or --from-step N.",
               "  The registered read names its pool; a default floor would silently score one",
               f"  arm's registration against another's snapshots. Available: "
               f"{[f'{a/1e6:.0f}M' for a in avail]}")
    missing = [s for s in steps if s not in slices]
    if missing:
        print(f"  MISSING steps: {[f'{m/1e6:.0f}M' for m in missing]} — NOT a reading over the")
        print(f"  requested pool. Available: {[f'{a/1e6:.0f}M' for a in avail]}")
        return 2
    print(f"  run {run}")
    print(f"  pooled steps: {', '.join(f'{s/1e6:.0f}M' for s in steps)}")

    def rows_for(klass):
        out = []
        for st_ in steps:
            s = slices[st_]
            p = np.asarray(s["win_probs"], float)
            y = np.asarray(s["outcomes"], float)
            b = np.asarray(s["battles"])
            o = np.asarray(s["opponents"])
            w, _ = selection_weights(y, b, o, tw.get(int(st_), {}))
            if w is None:
                w = np.ones_like(p)
            for nm in sorted({x for x in o.tolist() if opponent_class(x) == klass}):
                sel = o == nm
                if sel.sum() < 2 or len(set(y[sel].tolist())) < 2:
                    continue
                r = reliability_table(p[sel], y[sel], weights=w[sel])
                out.append((nm, st_, float(r["base_rate"]), float(r["skill"]),
                            len(set(b[sel].tolist()))))
        return out

    for klass in ("bot", "pool"):
        rows = rows_for(klass)
        if len(rows) < 6:
            print(f"\n  {klass}: only {len(rows)} rows — NOT COMPUTED")
            continue
        names = sorted({r[0] for r in rows})
        print(f"\n  {klass.upper()}: {len(rows)} (opponent, step) rows over {len(names)} opponents")
        print(f"    {'opponent':<16}{'steps':>6}{'base mean':>11}{'skill mean':>12}{'neg':>5}")
        for nm in names:
            rs = [r for r in rows if r[0] == nm]
            neg = sum(1 for r in rs if r[3] < 0)
            print(f"    {nm:<16}{len(rs):>6}{sum(r[2] for r in rs)/len(rs):>11.3f}"
                  f"{sum(r[3] for r in rs)/len(rs):>12.3f}{neg:>5}"
                  + ("   <- NEGATIVE SKILL in some cycles" if neg else ""))
        rho = spearman([r[2] for r in rows], [r[3] for r in rows])
        rng = np.random.default_rng(0)
        reps = []
        for _ in range(boot):
            draw = rng.choice(len(names), size=len(names), replace=True)
            rs = [r for i in draw for r in rows if r[0] == names[i]]
            v = spearman([r[2] for r in rs], [r[3] for r in rs])
            if not math.isnan(v):
                reps.append(v)
        lo, hi = np.percentile(reps, [2.5, 97.5]) if len(reps) > 50 \
            else (float("nan"), float("nan"))
        print(f"    SPEARMAN rho(base, skill) = {rho:+.3f}   95% CI clustered by OPPONENT "
              f"[{lo:+.3f}, {hi:+.3f}]   {len(reps)} replicates")
        if math.isnan(lo):
            print("    too few replicates — NOT COMPUTED")
        elif lo > 0:
            print("    => the ordering points AWAY from the cap account, SIGNIFICANT.")
            print("    NOT a claim that the head is blur-limited — an irreducible hidden-information")
            print("    floor predicts the same ordering. Only sd_true_excess separates them.")
        elif hi < 0:
            print("    => the CAP ACCOUNT IS SUPPORTED (skill falls as base rises).")
        else:
            print("    => NOT DETECTED. The ordering does not resolve; carried as suggestive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
