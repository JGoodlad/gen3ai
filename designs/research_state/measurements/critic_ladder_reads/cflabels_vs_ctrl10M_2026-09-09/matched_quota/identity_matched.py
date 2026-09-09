"""Is the IDENTITY-bias row confounded by the quota asymmetry the way the own-team row is?

STRUCTURALLY, NO — and that is the primary argument. `identity.bias.*` is a WEIGHTED MEAN of the
per-state quantity `V - MC`: no model is fit, nothing is held out, no decoder is trained. A
weighted mean's EXPECTATION does not depend on how many states entered it (given correct weights,
which rule 17's capture-rate IPW supplies); only its VARIANCE does. The own-team R² is the
opposite kind of object — an out-of-fold score of a ridge FIT on the frame, whose expectation
rises with the training set.

There is a second-order channel worth checking anyway: `cf_audit` draws its 800 labelled states
from the TRACE TREE, so the arm's draw was taken from 627 battles and the control's from 198, and
the `pop` recombination uses the tree's own (decile × outcome) mass. This script restricts the
arm's committed payload to the battles a matched subsample keeps, RECOMPUTES the frame mass and
the capture rates for that subsample, and re-reads the bias.

🚨 THE HONEST LIMIT. Restricting the payload leaves ~250 of the 800 labelled states, whereas a
true matched-quota `cf_audit` would have drawn a fresh 800 from the smaller tree (denser coverage
of the same battles). This is therefore a STABILITY check on the point estimate under a shrinking
frame — it cannot, and does not, reproduce the interval a matched read would print. New MC
rollouts would be needed for that, and this job runs no rollouts.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict, List

import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subsample as S                                              # noqa: E402

from main.ops import critic_readouts as R                          # noqa: E402

STRATA = ("ALL", "early (turn<=10)", "mid (11-24)", "late (turn>=25)", "bot", "pool")
WEIGHTINGS = ("ipw", "pop", "raw")


def rows_of(rows: List[dict], name: str) -> List[dict]:
    if name == "ALL":
        return list(rows)
    if name in ("bot", "pool"):
        return [r for r in rows if (r["opp_class"] == "bot") == (name == "bot")]
    return [r for r in rows if R.turn_bucket(r["turn"]) == name]


def bias_block(rows: List[dict], frame_cells: Dict[str, int], cap, *, boot: int,
               seed: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in STRATA:
        sub = rows_of(rows, name)
        if len(sub) < 5:
            out[name] = {"n": len(sub)}
            continue
        w_pop, _cov = R.pop_weights(sub, frame_cells)
        w_ipw, ipw_cov = R.apply_capture(sub, w_pop, cap)
        f = R.bias_stat(sub)
        entry: Dict[str, Any] = {"n": len(sub),
                                 "n_battles": len({r["battle_key"] for r in sub}),
                                 "ipw_coverage": ipw_cov}
        for wname, w in (("raw", np.ones(len(sub))), ("pop", w_pop), ("ipw", w_ipw)):
            pt, draws = R.boot_draws(sub, f, w=w, draws=boot, seed=seed)
            entry[wname] = {"point": pt, "ci": R.ci_of(pt, draws)}
        out[name] = entry
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--caps", default="8,12,5")
    args = ap.parse_args(argv)
    cw, cl, cd = (int(x) for x in args.caps.split(","))

    from agents.training import cf_audit

    payload = json.load(open(args.payload))
    rows = payload["rows"]
    arm_run = os.path.join(S.ARCHIVE, S.ARM)
    arm_tr = os.path.join(arm_run, "eval_traces", f"step_{S.STEP}")

    full_cap = R.capture_weights(arm_run, S.STEP)
    full = bias_block(rows, payload["frame_cells"], full_cap, boot=args.boot, seed=0)
    print("## arm FULL frame (recomputed from the committed payload)\n")
    print("| stratum | n states | n battles | ipw | pop | raw |")
    print("|---|---|---|---|---|---|")
    for name in STRATA:
        e = full[name]
        if "ipw" not in e:
            continue
        print(f"| {name} | {e['n']} | {e['n_battles']} | "
              + " | ".join(f"{e[w]['point']:+.4f} [{e[w]['ci'][0]:+.4f}, {e[w]['ci'][1]:+.4f}]"
                           for w in WEIGHTINGS) + " |")

    per_seed: List[Dict[str, Any]] = []
    os.makedirs(args.tmp, exist_ok=True)
    for s in range(args.seeds):
        kept, man, _t = S.plan(arm_tr, cw, cl, cd, s)
        dest_run = os.path.join(args.tmp, "idview")
        dest = S.materialize(arm_tr, kept, man, dest_run)
        keep_keys = {f"{opp}/{b}" for opp, bs in kept.items() for b in bs}
        sub_rows = [r for r in rows if r["battle_key"] in keep_keys]
        # the SUBSAMPLE's own (decile x outcome x class x turn-bucket) mass — the `pop`
        # recombination targets the frame it was drawn from, so it must be rebuilt, not reused.
        frame, _skipped = cf_audit.build_frame(dest)
        cells: Counter = Counter()
        for dd in frame:
            cells[(min(9, int(dd.win_prob * 10)), dd.outcome, dd.opp_class,
                   R.turn_bucket(dd.turn))] += 1
        frame_cells = {"|".join(map(str, k)): v for k, v in sorted(cells.items())}
        cap = R.capture_weights(dest_run, S.STEP)
        per_seed.append(bias_block(sub_rows, frame_cells, cap, boot=args.boot, seed=0))
        shutil.rmtree(dest_run)
        print(f"  seed {s}: {len(sub_rows)} labelled states kept "
              f"({len({r['battle_key'] for r in sub_rows})} battles); "
              f"bias.ALL ipw {per_seed[-1]['ALL']['ipw']['point']:+.4f}", flush=True)

    print(f"\n## arm MATCHED frame ({cw}/{cl}/{cd}), {args.seeds} subsample seeds\n")
    print("| stratum | n states (median) | ipw median [2.5,97.5] | pop | raw | FULL ipw |")
    print("|---|---|---|---|---|---|")
    summary: Dict[str, Any] = {}
    for name in STRATA:
        if "ipw" not in full.get(name, {}):
            continue
        cells = [p[name] for p in per_seed if "ipw" in p.get(name, {})]
        if not cells:
            continue
        cols = []
        for w in WEIGHTINGS:
            v = [c[w]["point"] for c in cells]
            lo, hi = np.percentile(v, [2.5, 97.5])
            cols.append(f"{np.median(v):+.4f} [{lo:+.4f}, {hi:+.4f}]")
        print(f"| {name} | {np.median([c['n'] for c in cells]):.0f} | " + " | ".join(cols)
              + f" | {full[name]['ipw']['point']:+.4f} |")
        summary[name] = {"full": {w: full[name][w] for w in WEIGHTINGS},
                         "matched_median": {w: float(np.median([c[w]["point"] for c in cells]))
                                            for w in WEIGHTINGS},
                         "matched_q": {w: [float(x) for x in np.percentile(
                             [c[w]["point"] for c in cells], [2.5, 97.5])]
                             for w in WEIGHTINGS},
                         "n_median": float(np.median([c["n"] for c in cells]))}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"caps": [cw, cl, cd], "n_seeds": args.seeds, "boot": args.boot,
               "strata": summary}, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
