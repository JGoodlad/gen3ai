"""X65 ExIt-0 by hand (PREREG §6 "Checking it"): recompute every panel fraction, bound and the class of the rule
`distill_play` from the SAME inputs (the read's `inputs.json`) and the registered parameters (its spec), with loops of its
own that share no code with `main.reads.rules_distill_play`. `by_hand_test` in `src/main/reads/` holds it to the rule on
the constructed cases; on the read, PREREG §10 VOIDs the read if the two disagree.

    python3 by_hand.py <state>/inputs.json <read_spec.json> > by_hand.json

Only the bootstrap's index DRAW is shared with the rule, by construction: the same `numpy.random.default_rng(seed)`, fresh
per estimate, panels in order, seeds ascending, `integers(0, n, size=(B, n))` (the registered draw); everything computed
from it is a Python loop over exact fractions.
"""
import json
import math
import sys
from fractions import Fraction

import numpy as np


def y2(u):
    """Twice a one-game unit's score: win 2, tie 1, loss 0."""
    assert u["n"] == 1, u["unit"]
    return 2 * u["wins"] + (u.get("ties") or 0)


def recompute(doc, p):
    out = {"panels": {}}
    look = doc["look"]
    dv = doc["delivery"]
    if dv.get("problems"):
        out["class"] = "VOID"
        return out
    if not dv["delivered"]:
        out["class"] = "NOT-DELIVERED"
        return out
    units = {}
    for u in doc["units"]:
        if u["role"] == "deal":
            units[u["unit"]] = u
    void = []
    sums = {}
    for name in ("paired", "go_imputed", "null_imputed"):
        sums[name] = {}
        for ts in doc["u_panels"]:
            sums[name][ts] = {}
    for ts in doc["u_panels"]:
        deals = drops = unfinished = 0
        for d in doc["deals"]:
            if d["panel"] != ts or d["look"] > look:
                continue
            deals += 1
            a, b = d["student"]["status"], d["raw"]["status"]
            if a not in ("counted", "exhausted") or b not in ("counted", "exhausted"):
                unfinished += 1
                continue
            s = d["unit_seed"]
            ys = y2(units[d["student"]["unit"]]) if a == "counted" else None
            yr = y2(units[d["raw"]["unit"]]) if b == "counted" else None
            if ys is not None and yr is not None:
                for name in sums:
                    acc = sums[name][ts].setdefault(s, [0, 0])
                    acc[0] += ys - yr
                    acc[1] += 1
                continue
            drops += 1
            go = (0 if ys is None else ys) - (2 if yr is None else yr)       # GO: student 0, raw 1
            nu = (2 if ys is None else ys) - (0 if yr is None else yr)       # NULL: student 1, raw 0
            for name, v in (("go_imputed", go), ("null_imputed", nu)):
                acc = sums[name][ts].setdefault(s, [0, 0])
                acc[0] += v
                acc[1] += 1
        out["panels"][ts] = {"deals": deals, "drops": drops, "unfinished": unfinished}
        if unfinished:
            void.append(f"{ts}: unfinished")
        if deals and Fraction(drops, deals) > Fraction(str(p["max_drop_frac"])):
            void.append(f"{ts}: drops")
        if len(sums["go_imputed"][ts]) < 2:
            void.append(f"{ts}: < 2 seeds")
    for d in doc["deals"]:                       # REVIEW R-m5: a counted pair whose two STARTs are not one deal
        st = d.get("start")
        if st is not None and st["match"] is not True:
            void.append("pairing")
    for u in doc["units"]:
        if not u["counted"]:
            continue
        if u["engine"] != p["engine"]:
            void.append("engine")
        if u["model_sha256"] != doc["want_sha256"][u["label"]]:
            void.append("bytes")
        if u["our_regime"] != "greedy" or u["search"] is not None or u["search_played"] is not None \
                or u["belief_prior_mix"] is not None:
            void.append("regime")
        if u["games"]:
            result = "win" if u["wins"] else ("tie" if u.get("ties") else "loss")
            if len(u["games"]) != 1 or u["games"][0][3] != result:
                void.append("results")
    for lab in doc["want_sha256"]:
        if doc["resolved_sha256"][lab] != doc["want_sha256"][lab]:
            void.append("resolved bytes")
    for ts in doc["u_panels"]:
        for key in ("regime_id", "peer_code", "policy_mode"):
            seen = set()
            for u in doc["units"]:
                if u["counted"] and u["panel"] == ts:
                    seen.add(u[key])
            if len(seen) != 1 or None in seen:
                void.append(f"{ts}: {key}")
    if doc.get("gbank") is not None:
        gb = doc["gbank"]
        same = done = 0
        for c in gb["compared"]:
            if c["identical"] is not None:
                done += 1
            if c["identical"] is True:
                same += 1
        out["gbank_identical"] = same
        if done != len(gb["compared"]) or len(gb["compared"]) != gb["n"] or same < gb["min_identical"]:
            void.append("G-BANK")
    if void:
        out["class"] = "VOID"
        out["void"] = sorted(set(void))
        return out
    est = {}
    for name in sums:
        g = {}
        for ts in doc["u_panels"]:
            top = bot = 0
            for s, (a, c) in sums[name][ts].items():
                top += a
                bot += c
            g[ts] = Fraction(top, 2 * bot)
        gg = (g[doc["u_panels"][0]] + g[doc["u_panels"][1]] + g[doc["u_panels"][2]]) / 3
        rng = np.random.default_rng(p["bootstrap"]["seed"])
        B = p["bootstrap"]["B"]
        reps = [Fraction(0)] * B
        for ts in doc["u_panels"]:
            seeds = sorted(sums[name][ts])
            idx = rng.integers(0, len(seeds), size=(B, len(seeds)))
            for b in range(B):
                top = bot = 0
                for j in idx[b]:
                    a, c = sums[name][ts][seeds[int(j)]]
                    top += a
                    bot += c
                reps[b] += Fraction(top, 2 * bot)
        mean = Fraction(0)
        for b in range(B):
            reps[b] = reps[b] / 3
            mean += reps[b]
        mean /= B
        ss = Fraction(0)
        for r in reps:
            ss += (r - mean) * (r - mean)
        se = math.sqrt(ss / (B - 1))
        est[name] = {"G": {ts: str(v) for ts, v in g.items()}, "G_all": str(gg), "se": se}
    z = Fraction(str(p["z_go"][str(look)]))
    lb = Fraction(est["go_imputed"]["G_all"]) - z * Fraction(est["go_imputed"]["se"])
    ub = Fraction(est["null_imputed"]["G_all"]) + Fraction(str(p["z_null"])) * Fraction(est["null_imputed"]["se"])
    eps = Fraction(str(p["rule8_eps"]))
    bar = Fraction(str(p["null_bar_pp"])) / 100
    near_lb, near_ub = abs(lb) <= eps, abs(ub - bar) <= eps
    if lb > 0 and not near_lb:
        cls = "GO"
    elif ub < bar and not near_ub:
        cls = "NULL"
    elif look == 1:
        cls = "CONTINUE"
    else:
        cls = "UNDECIDED"
    out.update({"class": cls, "boundary": near_lb or near_ub, "estimates": est, "lb_exact": str(lb),
                "ub_exact": str(ub)})
    return out


def main(inputs_path, spec_path):
    doc = json.load(open(inputs_path))
    p = json.load(open(spec_path))["decision"]["params"]
    print(json.dumps(recompute(doc, p), indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
