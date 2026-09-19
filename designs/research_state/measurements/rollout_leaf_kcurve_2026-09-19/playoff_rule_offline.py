"""THE PLAYOFF'S OWN DECISION RULE, APPLIED OFFLINE TO THE K-CURVE'S PAIRED ROLLOUTS — the
overrule-rate dose curve, for free.

    python3 playoff_rule_offline.py --ext <rerollx dir> --banked <reroll dir> --out playoff_rule.json

WHY THIS EXISTS. The live cell measured 64 of 64 playoffs INCONCLUSIVE at R = 4 and
`n_changed = 0` — the arm never acts — and one battle of it costs ~1,200 s. Reading the same
question at R = 8 and R = 16 live would cost hours per point. But the K-curve's re-rolls ARE paired
rollouts under common random numbers: for re-roll k the three branches share ONE seed, so
`d_k = score(top1, k) - score(top2, k)` is exactly the statistic
:meth:`PlayoffRunner.adjudicate` forms. So the rule can be evaluated on 1,002 forks at every R
without playing a single further battle.

NOTHING IS RE-IMPLEMENTED. `playoff.paired_stats`, `playoff.is_conclusive`, `playoff.decide`,
`playoff.SE_MULTIPLE` and `playoff.MIN_PAIRS` are imported from the production module. If the rule
changes, this read changes with it — which is the point: a second hand-written copy of a gate is
how a gate's constant gets quietly forked.

🚨 **TWO DECLARED DIFFERENCES FROM THE LIVE CELL, and they cut in opposite directions.**

1. **The rollouts.** Here both sides are GREEDY and the opponent is the banked tree's SENTINEL;
   in the live mirror both sides are the SAME network STOCHASTIC at temperature 1. A stochastic
   opponent adds variance to every draw, so the live conclusive rate should be LOWER than this at
   matched R — this is an UPPER bound on the live rate.
2. **The candidates.** Here the pair is the POLICY's own top-2; live it is the SCREEN's top-2,
   which may be a wider-apart pair (the screen ranks by V, not by the pointer head) and therefore
   easier to resolve. That cuts the other way.

So this is a bound with a named direction on each side, not a prediction of the live number — and
it is reported as such. What it settles cleanly is the SHAPE: how the rate moves with R, which is
a property of the rule and the dice and of nothing else.

ALSO REPORTED: when the rule DOES conclude, how often it picks the branch the K'=8 LABEL says is
better — i.e. whether an override, when it happens, is right. That is the quantity an override
budget is spent on.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

BRANCHES = ("top1", "top2", "rand")
RS = (1, 2, 4, 8, 16)


def read(d, pat):
    o = {}
    for p in sorted(glob.glob(os.path.join(d, pat))):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                break
            o[(r["shard"], int(r["fork_line"]))] = r
    return o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", required=True)
    ap.add_argument("--banked", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args(argv)

    from main.search_dividend.playoff import (MIN_PAIRS, SE_MULTIPLE, decide, is_conclusive,
                                              paired_stats)

    ext, lab = read(args.ext, "rerollx_s*.jsonl"), read(args.banked, "reroll_s*.jsonl")
    forks = []
    for k in sorted(ext):
        e, b = ext[k], lab.get(k)
        if b is None or e["seed_salt"] != b["seed_salt"]:
            continue
        if any(len(e["scores"][x]) < max(RS) for x in BRANCHES) or \
           any(len(b["scores"][x]) < 8 for x in BRANCHES):
            continue
        forks.append((e, b))
    if not forks:
        raise SystemExit("REFUSED: no fork joined")

    out = {"n_forks": len(forks), "se_multiple": SE_MULTIPLE, "min_pairs": MIN_PAIRS,
           "rule_source": "main.search_dividend.playoff (imported, not re-implemented)", "R": {}}
    rng = np.random.default_rng(7)
    for R in RS:
        conc, right, act_top1 = [], [], []
        for e, b in forks:
            d = [float(e["scores"]["top1"][i]) - float(e["scores"]["top2"][i]) for i in range(R)]
            mean, se, n = paired_stats(d)
            ok = is_conclusive(mean, se, n, k=SE_MULTIPLE, min_pairs=MIN_PAIRS)
            conc.append(float(ok))
            if ok:
                # the rule's pick: a1 = top1 by construction here (the policy's own argmax), so a
                # POSITIVE mean keeps the policy's action and a negative one OVERRULES it.
                pick_top1 = mean > 0
                act_top1.append(float(pick_top1))
                lt1 = float(np.mean(b["scores"]["top1"][:8]))
                lt2 = float(np.mean(b["scores"]["top2"][:8]))
                if lt1 != lt2:
                    right.append(float(pick_top1 == (lt1 > lt2)))
        conc = np.asarray(conc)
        ids = np.arange(len(conc))
        boots = [conc[rng.choice(ids, len(ids), True)].mean() for _ in range(args.boot)]
        out["R"][str(R)] = {
            "conclusive_rate": float(conc.mean()),
            "conclusive_rate_ci": [float(np.quantile(boots, 0.025)),
                                   float(np.quantile(boots, 0.975))],
            "n_conclusive": int(conc.sum()),
            # a conclusion that KEEPS the policy's action changes nothing; only the other kind is
            # an overrule, and only overrules can move a win rate
            "overrule_rate_of_all": float(1.0 - np.mean(act_top1)) * float(conc.mean())
                                    if act_top1 else 0.0,
            "kept_policy_when_conclusive": float(np.mean(act_top1)) if act_top1 else None,
            "agrees_with_K8_label_when_conclusive": float(np.mean(right)) if right else None,
            "n_scored_against_label": len(right),
            "rollouts_per_decision": 2 * R,
        }
        print(f"R={R:2d}  conclusive {conc.mean():.4f} "
              f"[{out['R'][str(R)]['conclusive_rate_ci'][0]:.4f}, "
              f"{out['R'][str(R)]['conclusive_rate_ci'][1]:.4f}]  "
              f"n={int(conc.sum()):4d}  keeps_policy="
              f"{out['R'][str(R)]['kept_policy_when_conclusive']}  "
              f"right={out['R'][str(R)]['agrees_with_K8_label_when_conclusive']}", flush=True)

    json.dump(out, open(args.out, "w"), indent=1)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
