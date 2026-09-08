"""THE PER-BOT CALIBRATION CORRELATION — pre-registered for the 28M read (2026-09-07).

    python -m main.ops.perbot_r <run-name|run-dir> --steps A B ... [--boot 2000]

QUESTION: is low calibration skill concentrated in the bots the arm beats MOST easily?
  CAP/SATURATION predicts r(base, skill) NEGATIVE — least skill where the cap base*(1-base)
  is smallest. At 26M the observed r was +0.788 on n=8, i.e. the OPPOSITE sign: least skill
  on the most CONTESTED bots. This tool pools two or more steps to raise n and puts an interval
  on r.

🚨 WHAT A LOW SKILL ON CONTESTED OPPONENTS DOES *NOT* DISTINGUISH (Orchestrator 1, 2026-09-07):
  (a) REDUCIBLE — the head lacks resolution on contested opponents (the blur disease, rule 14).
  (b) IRREDUCIBLE — the outcome there is decided by hidden information and dice no critic can
      see, so achievable skill is low for ANY critic (the hidden-information floor; the shaped
      G0 audit found the 0.83 class was 53% luck).
  BOTH predict low skill where outcomes are contested. The instrument that separates them is
  the identity test's sd_true_excess, NOT this table. So a confirmed sign is banked as
  "AGAINST THE CAP ACCOUNT", never as "the head is blur-limited".

The bootstrap is CLUSTER-OVER-BATTLES, matching the gauge's own CIs: each replicate resamples
whole battles within each opponent, recomputes that opponent's skill, then recomputes r.
Weights are constant within a battle so the clusters survive the selection reweighting.

Promoted 2026-09-07 from the Training Run session's ``perbot_r.py`` (the run was positional and
the import path was a ``sys.path.insert(0, "src")``, which only worked from the repo root).
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


def corr(xs, ys) -> float:
    if len(xs) < 3:
        return float("nan")
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((c - my) ** 2 for c in ys))
    return (sum((a - mx) * (c - my) for a, c in zip(xs, ys)) / den) if den else float("nan")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if args else 2
    run = resolve_run_dir(args[0])
    if "--steps" not in args:
        refuse("REFUSING: --steps is required. The registered read names the snapshot steps it",
               "  pools; a default would pool whatever happens to exist. Nothing is concluded.")
    # --steps consumes values until the NEXT FLAG. Taking "every digit after --steps" swallowed
    # --boot's argument and invented a phantom step 400 ("0M").
    i = args.index("--steps") + 1
    steps = []
    while i < len(args) and not args[i].startswith("--"):
        steps.append(int(args[i]))
        i += 1
    boot = int(args[args.index("--boot") + 1]) if "--boot" in args else 2000

    # collect_slices returns a TUPLE (slices, meta). Binding the tuple made every membership
    # test false and printed "no traces yet" for traces that were present — a confident false
    # negative from an unchecked return shape (rule 16, in this tool's own code).
    slices, _slice_meta = collect_slices(str(run))
    assert isinstance(slices, dict), f"expected a dict of slices, got {type(slices)}"
    tw = true_win_rates(str(run))
    have = [s for s in steps if s in slices]
    missing = [s for s in steps if s not in slices]
    if missing:
        print(f"  MISSING steps (no traces yet): {[f'{m/1e6:.0f}M' for m in missing]}")
        print("  This is NOT a reading over the requested pool. Re-run when they exist.")
        if not have:
            return 2

    # pool the requested steps, keeping battle ids unique per step
    P, Y, B, O, W = [], [], [], [], []
    for st_ in have:
        s = slices[st_]
        p = np.asarray(s["win_probs"], float)
        y = np.asarray(s["outcomes"], float)
        b = np.asarray([f"{st_}:{x}" for x in s["battles"]])
        o = np.asarray(s["opponents"])
        w, _ = selection_weights(y, b, o, tw.get(int(st_), {}))
        P.append(p)
        Y.append(y)
        B.append(b)
        O.append(o)
        W.append(w if w is not None else np.ones_like(p))
    P, Y, B, O, W = (np.concatenate(x) for x in (P, Y, B, O, W))
    print(f"  run {run}")
    print(f"  pooled steps: {', '.join(f'{s/1e6:.0f}M' for s in have)}   "
          f"{P.size} rows over {len(set(B.tolist()))} battles")

    def skill_and_base(sel, idx=None):
        """Murphy skill and base rate for a selection, optionally on a resampled index."""
        p, y, w = (P[sel], Y[sel], W[sel]) if idx is None \
            else (P[sel][idx], Y[sel][idx], W[sel][idx])
        if y.size < 2 or len(set(y.tolist())) < 2:
            return None
        r = reliability_table(p, y, weights=w)
        return float(r["skill"]), float(r["base_rate"])

    for klass in ("bot", "pool"):
        names = sorted({o for o in O.tolist() if opponent_class(o) == klass})
        rows = []
        for nm in names:
            sel = O == nm
            sb = skill_and_base(sel)
            nb = len(set(B[sel].tolist()))
            if sb:
                rows.append((nm, sb[1], sb[0], nb))
        if len(rows) < 3:
            print(f"\n  {klass}: only {len(rows)} usable opponent(s) — r NOT COMPUTED")
            continue
        print(f"\n  {klass.upper()} strata, pooled:")
        print(f"    {'opponent':<16}{'base':>7}{'cap':>8}{'skill':>8}{'battles':>8}")
        for nm, base, sk, nb in sorted(rows, key=lambda r_: -r_[1]):
            flag = "   <- NEGATIVE SKILL" if sk < 0 else ""
            print(f"    {nm:<16}{base:>7.3f}{base*(1-base):>8.4f}{sk:>8.3f}{nb:>8}{flag}")
        r_obs = corr([x[1] for x in rows], [x[2] for x in rows])

        rng = np.random.default_rng(0)
        reps = []
        for _ in range(boot):
            xs, ys = [], []
            for nm, _, _, _ in rows:
                sel = O == nm
                bl = np.asarray(sorted(set(B[sel].tolist())))
                draw = rng.choice(len(bl), size=len(bl), replace=True)
                keep = np.concatenate([np.flatnonzero(B[sel] == bl[i]) for i in draw])
                sb = skill_and_base(sel, keep)
                if sb:
                    ys.append(sb[0])
                    xs.append(sb[1])
            if len(xs) == len(rows):
                reps.append(corr(xs, ys))
        reps = [x for x in reps if not math.isnan(x)]
        lo, hi = (np.percentile(reps, [2.5, 97.5]) if len(reps) > 50
                  else (float("nan"), float("nan")))
        print(f"    r(base, skill) = {r_obs:+.3f}   95% cluster-bootstrap-over-battles "
              f"[{lo:+.3f}, {hi:+.3f}]   n={len(rows)} opponents, {len(reps)} replicates")
        if not math.isnan(lo):
            if lo > 0:
                print("    INTERVAL CLEAR OF ZERO and POSITIVE -> the CAP account is REFUTED on this")
                print("    pool; the anti-cap observation STANDS.")
                print("    It does NOT distinguish a blur-limited head from the hidden-information")
                print("    floor — only sd_true_excess does that.")
            elif hi < 0:
                print("    INTERVAL CLEAR OF ZERO and NEGATIVE -> consistent with the CAP account.")
            else:
                print("    INTERVAL COVERS ZERO -> SUGGESTIVE ONLY, carried as such, no claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
