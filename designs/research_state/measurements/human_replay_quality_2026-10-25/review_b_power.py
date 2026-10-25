"""REVIEW B's power recomputation for X59 and X60 (design_human_replay_lever.md §5), CPU, seeded, deterministic.

Run: python review_b_power.py  (numpy + scipy; ~1 min). Prints one JSON document; REVIEW_B_experiments.md quotes it.
Models: every Kakuna cell is binomial (X51's pairing bought nothing: P* - 161M half-widths of +-6.8 pp at n = 400 are
the unpaired binomial width); X59 uses normal intervals with plug-in variances (its registered paired bootstrap over
20-game units is close to this when units carry no extra variance); X60 reproduces main.x35.d1's classifier (equal-
weight panels, Kish n_eff, Newcombe-style half-width as sqrt(V_on + V_off) * 1.96, AM-5 widening by sqrt(max(1, phi)),
the per-seed sign rule, the [-3, +3] NULL band, look 2 only when look 1 is UNDECIDED with a positive estimate).
"""
import json

import numpy as np
from scipy.stats import chi2, norm

SEED = 20261025
Z = 1.959963985


def x59(rng, hom_home, d_u, pstar=(0.598, 0.497, 0.530, 0.433), n_hom=400, n_pstar=400, reps=200_000):
    """P(SUPPORTS), P(CONTRADICTS), P(UNCLEAR) under X59's registered rule (§5.1)."""
    ph, pH, pF, pP = pstar
    b = lambda p, n: rng.binomial(n, p, reps) / n
    Hh, HH, HF, HP = b(hom_home, n_hom), b(pH + d_u, n_hom), b(pF + d_u, n_hom), b(pP + d_u, n_hom)
    Ph, PH, PF, PP = b(ph, n_pstar), b(pH, n_pstar), b(pF, n_pstar), b(pP, n_pstar)
    hu, pu = (HH + HF + HP) / 3, (PH + PF + PP) / 3
    du, dprem = hu - pu, (Hh - hu) - (Ph - pu)
    v = lambda p, n: p * (1 - p) / n
    sdu = np.sqrt((v(HH, n_hom) + v(HF, n_hom) + v(HP, n_hom)) / 9 + (v(PH, n_pstar) + v(PF, n_pstar) + v(PP, n_pstar)) / 9)
    sdp = np.sqrt(v(Hh, n_hom) + v(Ph, n_pstar) + sdu ** 2)
    sup = (dprem + Z * sdp < 0) & (du - Z * sdu > -0.03)
    con = (dprem - Z * sdp > -0.02) & (du + Z * sdu < 0)
    return {"SUPPORTS": round(float(sup.mean()), 3), "CONTRADICTS": round(float(con.mean()), 4),
            "UNCLEAR": round(float(1 - sup.mean() - con.mean()), 3),
            "dprem_halfwidth_pp": round(float(np.median(Z * sdp)) * 100, 2),
            "dU_halfwidth_pp": round(float(np.median(Z * sdu)) * 100, 2)}


def x60(rng, delta, seed_sd=0.0, games=140, base=0.47, K=3, band=0.03, reps=100_000):
    """X60's final class probabilities in D1's frame (main.x35.d1), 3 seeds at look 1, 5 at look 2."""
    def arm(d):
        s = rng.normal(0.0, seed_sd, (reps, 5))
        p = np.clip(base + d + s[:, :, None] + np.zeros((1, 1, K)), 0.01, 0.99)
        return rng.binomial(games, p) / games                    # reps x seeds x panels
    on, off = arm(delta), arm(0.0)
    looks = []
    for look, ns in ((1, 3), (2, 5)):
        a, c = on[:, :ns, :], off[:, :ns, :]
        pa, pc = a.mean(1), c.mean(1)                            # pooled over seeds, per panel
        va = (pa * (1 - pa) / (games * ns)).sum(1) / K ** 2
        vc = (pc * (1 - pc) / (games * ns)).sum(1) / K ** 2
        def ratio(x):
            e = (x * (1 - x) / games).sum(2).mean(1) / K ** 2
            return x.mean(2).var(1, ddof=1) / e
        f = np.sqrt(np.maximum(1.0, (ratio(a) + ratio(c)) / 2))
        d = pa.mean(1) - pc.mean(1)
        hw = Z * np.sqrt(va + vc) * f
        sign_ok = ((a.mean(2) - c.mean(2)) > 0).sum(1) >= (2 if look == 1 else 3)
        lo, hi = d - hw, d + hw
        cls = np.where((lo > 0) & sign_ok, "GAIN", np.where(hi < 0, "LOSS",
                       np.where((lo > -band) & (hi < band), "NULL", "UNDECIDED")))
        looks.append((cls, d, hw))
    (c1, d1, h1), (c2, _, h2) = looks
    final = np.where((c1 == "UNDECIDED") & (d1 > 0), c2, c1)
    out = {k: round(float((final == k).mean()), 3) for k in ("GAIN", "NULL", "LOSS", "UNDECIDED")}
    out["halfwidth_look1_pp"] = round(float(np.median(h1)) * 100, 2)
    out["halfwidth_look2_pp"] = round(float(np.median(h2)) * 100, 2)
    return out


def main():
    rng = np.random.default_rng(SEED)
    r = {"seed": SEED, "x59": {}, "x60": {}, "other": {}}
    worlds = {
        "registered prediction (HOM home72 0.52, dU -1 pp; implies dprem -6.8, not -5)": (0.52, -0.01),
        "null world (HOM = P* everywhere)": (0.598, 0.0),
        "pure FORGETTING, no unfamiliar skill (HOM home72 0.52, dU 0)": (0.52, 0.0),
        "real unfamiliar skill, no forgetting (HOM home72 0.598, dU +4 pp)": (0.598, 0.04),
        "real skill AND forgetting (HOM home72 0.55, dU +3 pp)": (0.55, 0.03),
        "A-1 false, HOM worse on U (HOM home72 0.52, dU -5 pp)": (0.52, -0.05),
    }
    for k, (h, du) in worlds.items():
        r["x59"][k] = x59(rng, h, du)
    r["x59"]["registered prediction, HOM 800/cell AND P* re-played to 800/cell"] = x59(rng, 0.52, -0.01, n_hom=800, n_pstar=800)
    r["x59"]["dU-only (HOM - P*), half-width and P(LB > 0 | +3 pp) by games per cell per model"] = {
        str(n): {"hw_pp": round(Z * np.sqrt(2 * 3 * 0.25 / n / 9) * 100, 2),
                 "power_+3pp": round(float(1 - norm.cdf(Z - 0.03 / np.sqrt(2 * 3 * 0.25 / n / 9))), 3)}
        for n in (400, 800, 1600)}
    for g in (140, 420, 700):
        for sd in (0.0, 0.027):
            for d in (0.0, 0.02, 0.025, 0.04):
                r["x60"][f"games/panel/seed={g} seed_sd={sd*100:.1f}pp delta={d*100:.1f}pp"] = x60(rng, d, sd, g)
    # D1 look 2's phi-hat 0.770 on df 8: the one-sided 95 % upper bound on phi, and the seed SD it allows at 140 games
    phi, df = 0.7704053883810678, 8
    phi_ub = phi * df / chi2.ppf(0.05, df)
    binom_var = 3 * 0.48 * 0.52 / 140 / 9
    r["other"]["D1 phi-hat UB95 and implied between-seed SD of the U mean (pp)"] = {
        "phi_ub95": round(phi_ub, 3), "seed_sd_ub_pp": round(float(np.sqrt(max(0.0, phi_ub - 1) * binom_var)) * 100, 2)}
    # A-7's per-team falsifier: ~10 games per piloted team per arm at 140 games/panel/seed x 5 seeds / 72 teams
    g_team = 140 * 5 / 72
    se_slope = np.sqrt(2 * 0.25 / g_team) / np.sqrt(72)
    r["other"]["A-7 per-team regression MDE80 (pp per SD of the proximity covariate)"] = round((Z + 0.8416) * se_slope * 100, 1)
    # the matched-student contest (REVIEW B's first experiment): Kakuna U at 1,200 / 2,400 games per student
    r["other"]["student contest dU MDE80 (pp) at 1200 / 2400 U games per student"] = [
        round((Z + 0.8416) * np.sqrt(2 * 0.25 / n) * 100, 1) for n in (1200, 2400)]
    print(json.dumps(r, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
