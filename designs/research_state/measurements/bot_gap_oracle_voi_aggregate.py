"""Aggregate oracle_voi_results.jsonl — cluster bootstrap over battles per stratum."""
import json
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "tmp/si2/oracle_voi_results.jsonl"
rows = [json.loads(l) for l in open(path)]
ok = [r for r in rows if r.get("ok") and r.get("alpha_available")]
err = [r for r in rows if not r.get("ok")]
print(f"{len(rows)} rows, {len(ok)} ok+alpha, {len(err)} errors")
from collections import Counter
print("error kinds:", Counter(e.get("error", "")[:60] for e in err).most_common(5))

rng = np.random.default_rng(0)


def boot_mean(vals, clusters, n=3000):
    """Cluster bootstrap: resample battles, mean over their decisions."""
    by = {}
    for v, c in zip(vals, clusters):
        by.setdefault(c, []).append(v)
    keys = list(by)
    if not keys:
        return None
    stats = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys))
        s = [x for i in pick for x in by[keys[i]]]
        stats.append(np.mean(s))
    return float(np.mean(vals)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def report(name, sel):
    if not sel:
        print(f"-- {name}: EMPTY")
        return {}
    cl = [r["battle"] for r in sel]
    out = {"n_decisions": len(sel), "n_battles": len(set(cl))}
    for key in ("flip", "voi_wp", "oracle_gain_over_chosen", "marginal_gain_over_chosen",
                "alpha_on_recorded", "alpha_switch_mass"):
        vals = [float(r[key]) for r in sel if r.get(key) is not None]
        cls = [r["battle"] for r in sel if r.get(key) is not None]
        if vals:
            m, lo, hi = boot_mean(np.array(vals), cls)
            out[key] = {"mean": round(m, 4), "ci": [round(lo, 4), round(hi, 4)], "n": len(vals)}
    flips = [r for r in sel if r.get("flip")]
    if flips:
        m, lo, hi = boot_mean(np.array([r["voi_wp"] for r in flips]), [r["battle"] for r in flips])
        out["voi_wp_given_flip"] = {"mean": round(m, 4), "ci": [round(lo, 4), round(hi, 4)],
                                    "n": len(flips)}
    out["opp_switch_rate"] = round(float(np.mean([r["opp_actually_switched"] for r in sel
                                                  if r.get("opp_actually_switched") is not None])), 4)
    out["mean_n_opp"] = round(float(np.mean([r["n_opp"] for r in sel])), 2)
    print(f"-- {name}: {json.dumps(out)}")
    return out


strata = {
    "loss_crater": [r for r in ok if r["outcome"] == "loss" and r["stratum"] == "crater" and r["bot"] != "random"],
    "loss_random_dec": [r for r in ok if r["outcome"] == "loss" and r["stratum"] == "random" and r["bot"] != "random"],
    "win_random_dec": [r for r in ok if r["outcome"] == "win" and r["bot"] != "random"],
    "vs_random_bot": [r for r in ok if r["bot"] == "random"],
}
agg = {k: report(k, v) for k, v in strata.items()}

# per-bot VoI at loss craters
per_bot = {}
for bot in sorted({r["bot"] for r in ok}):
    sel = [r for r in strata["loss_crater"] if r["bot"] == bot]
    if sel:
        per_bot[bot] = {"n": len(sel), "flip": round(float(np.mean([r["flip"] for r in sel])), 3),
                        "voi_mean": round(float(np.mean([r["voi_wp"] for r in sel])), 4)}
print("per-bot craters:", json.dumps(per_bot))
agg["per_bot_loss_crater"] = per_bot

json.dump(agg, open("tmp/si2/oracle_voi_agg.json", "w"), indent=1)
print("saved tmp/si2/oracle_voi_agg.json")
