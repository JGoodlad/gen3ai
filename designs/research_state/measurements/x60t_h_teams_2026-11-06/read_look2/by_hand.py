"""FROZEN record (X60-T look 2, PREREG §12 / §13's by-hand lines; DESCRIPTIVE, deciding nothing).

(1) §13: Δ_F without the three resumed ON seeds (s3501, s3506, s3508): ON's five unresumed finals against C0's eight.
    Rule `x60`'s interval is defined for equal finals an arm, so the 5 v 8 line generalises §5's V by hand
    (V = binomial + max(1.0 pp, σ̂)² · (1/5 + 1/8), σ̂² = the df-weighted W − b); the 5 v 5 line at the same seeds is
    the rule's own `x60_interval`.
(2) §12: Newcombe 95 % intervals for the REPORTED pooled Δ_H / Δ_home72 / Δ_P72 (binomial only: no seed term).
(3) §12: U (H, F, P72 at equal weight) and the goal-(b) statistic F + P72, with D1's stratified Newcombe interval
    and AM-5's overdispersion factor (`main.x35.d1`), and Δ_H − Δ_F.
(4) the per-seed differences of every meter, and each meter against §16's registered guess.

Reads this folder's decision.json and inputs.json only; prints JSON (by_hand.json). Not a tool: never re-run as one.
"""
import json
import math
import os
import sys

from main.reads import rules_x60 as X
from main.x35 import d1 as D

HERE = os.path.dirname(os.path.abspath(__file__))
dec = json.load(open(os.path.join(HERE, "decision.json")))
inp = json.load(open(os.path.join(HERE, "inputs.json")))
RESUMED = (3501, 3506, 3508)
SEEDS = tuple(range(3501, 3509))
out = {}

# ---------------------------------------------------------------- (1) Δ_F without the resumed ON seeds
cnt = inp["primary"]["counts"]
on = {s: (cnt[f"x60t_s{s}"]["forum72"]["wins"], cnt[f"x60t_s{s}"]["forum72"]["n"]) for s in SEEDS}
off = {s: (cnt[f"c0_s{s}"]["forum72"]["wins"], cnt[f"c0_s{s}"]["forum72"]["n"]) for s in SEEDS}
keep = [s for s in SEEDS if s not in RESUMED]


def var(xs):
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def unequal(a, c):
    """§5's V generalised to unequal finals an arm (by hand; equal n reduces to the rule's)."""
    pa, pc = sum(w for w, _ in a) / sum(n for _, n in a), sum(w for w, _ in c) / sum(n for _, n in c)
    ra, rc = [w / n for w, n in a], [w / n for w, n in c]
    vbin = pa * (1 - pa) / sum(n for _, n in a) + pc * (1 - pc) / sum(n for _, n in c)
    dfa, dfc = len(a) - 1, len(c) - 1
    w = (dfa * var(ra) + dfc * var(rc)) / (dfa + dfc)
    b = (sum(r * (1 - r) / n for r, (_, n) in zip(ra, a)) + sum(r * (1 - r) / n for r, (_, n) in zip(rc, c))) / (
        len(a) + len(c))
    s2 = max(X.SIGMA_F ** 2, w - b)
    hw = X.Z * math.sqrt(vbin + s2 * (1 / len(a) + 1 / len(c)))
    d = pa - pc
    return {"on": pa, "off": pc, "delta_pp": 100 * d, "ci95_pp": [100 * (d - hw), 100 * (d + hw)],
            "sigma_hat_sq": w - b, "finals": [len(a), len(c)], "class_if_it_were_the_rule": X.classes(d - hw, d + hw)}


out["delta_f_all_8v8_check"] = unequal([on[s] for s in SEEDS], [off[s] for s in SEEDS])
out["delta_f_without_resumed_5v8"] = unequal([on[s] for s in keep], [off[s] for s in SEEDS])
r = X.x60_interval([[on[s]] for s in keep], [[off[s]] for s in keep])
out["delta_f_without_resumed_5v5_same_seeds"] = {
    "seeds": keep, "delta_pp": 100 * r["delta"], "ci95_pp": [100 * r["lo"], 100 * r["hi"]],
    "sigma_hat_sq": r["sigma_hat_sq"], "welch_used": r["welch_used"],
    "class_if_it_were_the_rule": X.classes(r["lo"], r["hi"])}
out["delta_f_per_seed_pp"] = {str(s): 100 * (on[s][0] / on[s][1] - off[s][0] / off[s][1]) for s in SEEDS}
out["f_wins"] = {"on": [sum(w for w, _ in on.values()), sum(n for _, n in on.values())],
                 "off": [sum(w for w, _ in off.values()), sum(n for _, n in off.values())]}

# ---------------------------------------------------------------- (2), (3) the REPORTED panels
pooled = dec["reported"]["pooled"]


def per_seed(panel, side, prefix):
    blk = pooled[panel][side]["per_final"]
    res = {}
    for s in SEEDS:
        w = blk[f"{prefix}_s{s}"] * 140
        assert abs(w - round(w)) < 1e-6, (panel, side, s, w)
        res[str(s)] = (int(round(w)), 140)
    assert sum(w for w, _ in res.values()) == pooled[panel][side]["wins"]
    return res


rep_on = {p: per_seed(p, "on", "x60t") for p in pooled}
rep_off = {p: per_seed(p, "off", "c0") for p in pooled}
rep_on["forum72"] = {str(s): on[s] for s in SEEDS}
rep_off["forum72"] = {str(s): off[s] for s in SEEDS}


def tot(blk):
    return (sum(w for w, _ in blk.values()), sum(n for _, n in blk.values()))


def strat(panels):
    a = D.strat_score({p: tot(rep_on[p]) for p in panels}, panels)
    c = D.strat_score({p: tot(rep_off[p]) for p in panels}, panels)
    d, lo, hi = D.newcombe_strat(a, c)
    res = {"on": a.mean, "off": c.mean, "delta_pp": 100 * d, "ci95_pp": [100 * lo, 100 * hi]}
    if len(panels) > 1:
        od = D.overdispersion({s: {p: rep_on[p][s] for p in panels} for s in map(str, SEEDS)},
                              {s: {p: rep_off[p][s] for p in panels} for s in map(str, SEEDS)}, panels)
        wlo, whi = D.widen(d, lo, hi, od.factor)
        res.update(phi_hat=od.phi, factor=od.factor, ci95_widened_pp=[100 * wlo, 100 * whi])
    return res


out["reported_newcombe"] = {pooled[p]["name"]: dict(strat((p,)), wins_on=list(tot(rep_on[p])),
                                                    wins_off=list(tot(rep_off[p]))) for p in pooled}
out["u_h_f_p72"] = strat(("heldout72_hpfix", "forum72", "p72"))
out["goal_b_f_p72"] = strat(("forum72", "p72"))
out["delta_h_minus_delta_f_pp"] = 100 * (pooled["heldout72_hpfix"]["delta"] - dec["primary"]["look2"]["delta"])
out["reported_per_seed_pp"] = {
    pooled[p]["name"]: {str(s): 100 * (rep_on[p][str(s)][0] - rep_off[p][str(s)][0]) / 140 for s in SEEDS}
    for p in pooled}

# ---------------------------------------------------------------- (4) the S meters per seed, and §16's guesses
for key in ("p", "i_f", "r72"):
    nodes = inp[key]["nodes"]
    out[f"{key}_per_seed_pp"] = {
        str(s): {"on": nodes[f"x60t_s{s}"]["s_pp"], "off": nodes[f"c0_s{s}"]["s_pp"],
                 "delta": nodes[f"x60t_s{s}"]["s_pp"] - nodes[f"c0_s{s}"]["s_pp"]} for s in SEEDS}
guess = {"delta_f": (1.5, -1.5, 4.5, 100 * dec["primary"]["look2"]["delta"]),
         "i_f": (0.8, -0.5, 2.5, dec["i_f"]["welch95"]["delta_pp"]),
         "delta_p": (-1.5, -3.0, 0.0, dec["p"]["stat"]["delta_pp"]),
         "r72": (0.8, -0.3, 2.5, dec["r72"]["welch95"]["delta_pp"]),
         "t_minus_d1": (-0.5, -3.5, 2.5, 100 * dec["reported"]["contrast_t_minus_d1"]["stat"]["delta"]),
         "delta_h": (2.5, -1.0, 6.0, 100 * pooled["heldout72_hpfix"]["delta"])}
out["against_section_16"] = {k: {"guess_pp": g, "interval80_pp": [lo, hi], "read_pp": v, "inside": lo <= v <= hi}
                             for k, (g, lo, hi, v) in guess.items()}
json.dump(out, sys.stdout, indent=1)
