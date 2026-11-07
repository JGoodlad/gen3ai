"""S'-HONEST play PREREG input (OUTCOME-BLIND): the exclusion guard's numbers from stage 1b's battles (PREREG §6).

Stage 1b's rows (``sprime_honest_inc_stage1b_2026-11-06/inc_rows.jsonl.gz``: every decision of 407 battles' sequences)
are resampled BY BATTLE (exclusions cluster: a stuck battle excludes the rest of its decisions):

1. the projection's own error: per panel, the projected live share (each turn bucket's rate weighted by S'-LIVE's
   search-path decisions a game in that bucket, ``cost_model.json``) and its battle-bootstrap 95 % interval; the
   full-read BOUND per panel = max(2 x the projection, the interval's upper end, 5 %), rounded UP to a whole percent;
2. the checkpoint's limit: the POOLED share (excluded / search-path decisions over 20 battles a panel, 60 in all, as
   the checkpoint's 60 searched games) under the same resampling; the hold fires above its 99th percentile.

    python3 excl_bounds.py --rows INC_ROWS --cost-model cost_model.json --out excl_bounds.json
"""
import argparse
import collections
import gzip
import json
import math
from pathlib import Path

import numpy as np

B = ("10-19", "20-29", "30-63", "64+")
PANELS = ("H", "F", "P72")
SEED, NBOOT, CKPT_GAMES = 20261106, 4000, 20


def main(a):
    rows = [r for r in (json.loads(x) for x in gzip.open(a.rows, "rt") if x.strip())
            if "group_done" not in r and not r.get("group_error")]
    W = json.load(open(a.cost_model))["inputs"]["slive"]["per_game"]
    bat = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for x in rows:
        c = bat[(x["panel"], json.dumps(x["group"]))][x["bucket"]]
        c[0] += int(x["res"].get("why") == "excluded")
        c[1] += 1
    rng = np.random.default_rng(SEED)
    out = {"inputs": {"rows": a.rows, "cost_model": a.cost_model, "seed": SEED, "B": NBOOT}, "panels": {}}
    E, N = {}, {}
    for p in PANELS:
        bs = [v for (pp, _g), v in sorted(bat.items()) if pp == p]

        def proj(sample):
            e, n = collections.Counter(), collections.Counter()
            for v in sample:
                for b in B:
                    e[b] += v[b][0]
                    n[b] += v[b][1]
            return sum(W[p][b] * (e[b] / n[b] if n[b] else 0.0) for b in B) / sum(W[p].values())

        pt = proj(bs)
        reps = np.array([proj([bs[i] for i in rng.integers(0, len(bs), len(bs))]) for _ in range(NBOOT)])
        lo, hi = np.percentile(reps, [2.5, 97.5])
        bound = math.ceil(round(max(2 * pt, hi, 0.05), 6) * 100) / 100
        out["panels"][p] = {"battles": len(bs), "battles_reaching_64": sum(1 for v in bs if v["64+"][1]),
                            "projection": round(pt, 4), "projection_ci95": [round(lo, 4), round(hi, 4)],
                            "bound": bound}
        E[p] = np.array([sum(v[b][0] for b in B) for v in bs])
        N[p] = np.array([sum(v[b][1] for b in B) for v in bs])
    num = np.zeros(NBOOT)
    den = np.zeros(NBOOT)
    for p in PANELS:
        idx = rng.integers(0, len(E[p]), (NBOOT, CKPT_GAMES))
        num += E[p][idx].sum(1)
        den += N[p][idx].sum(1)
    s = num / den
    out["checkpoint"] = {"games_a_panel": CKPT_GAMES, "pooled_share_p50": round(float(np.percentile(s, 50)), 4),
                         "pooled_share_p99": round(float(np.percentile(s, 99)), 4),
                         "hold_above": math.ceil(round(float(np.percentile(s, 99)), 6) * 100) / 100}
    for p in PANELS:                                   # per panel at the checkpoint: reported, never a hold
        idx = rng.integers(0, len(E[p]), (NBOOT, CKPT_GAMES))
        sp = E[p][idx].sum(1) / N[p][idx].sum(1)
        out["checkpoint"][f"{p}_share_p99"] = round(float(np.percentile(sp, 99)), 4)
        out["checkpoint"][f"{p}_P_over_bound"] = round(float((sp > out["panels"][p]["bound"]).mean()), 4)
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--cost-model", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
