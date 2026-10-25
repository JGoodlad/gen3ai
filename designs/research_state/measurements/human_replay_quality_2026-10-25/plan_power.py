"""The REVISED plan's power (design_human_replay_lever.md §5, revision of 2026-10-25), CPU, seeded, deterministic.

Run: python plan_power.py  (numpy + scipy; about 2 min on three cores). Prints one JSON document the plan quotes.
It reuses REVIEW B's models (review_b_power.py, beside this file): every Kakuna cell is binomial; X60 reproduces
main.x35.d1's classifier. What is new here:
- X59-S: the matched-student contrast at 1,200 / 2,400 / 3,600 U games per student, its in-house floor fallback,
  and the replication rule (a second seed pair must agree in sign and clear its own LB);
- X60 at 420 and 630 games per (seed final, panel) with the PRIMARY moved to F + P72 (two panels) and the
  heterogeneity guard (both panels' point estimates > 0), the contrast arms (H-TEAMS - D1, H-BC - P-BC) and the positive control
  (H-p, the practised half of H).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_b_power as RB  # noqa: E402

SEED = 20261026
Z = RB.Z
Z80 = 0.8416212335729143


def x59s(rng, delta, n_u, base=0.30, reps=200_000):
    """P(HUMAN-LABELS-BETTER), P(OWN-LABELS-BETTER) for Delta_S = S-H - S-P on U (three panels, equal n),
    one seed pair, and the same with a second, independent seed pair that must agree (the replication rule)."""
    def one():
        k = n_u // 3
        sh = rng.binomial(k, np.clip(base + delta, 0.01, 0.99), (reps, 3)) / k
        sp = rng.binomial(k, base, (reps, 3)) / k
        d = sh.mean(1) - sp.mean(1)
        v = ((sh * (1 - sh)).sum(1) + (sp * (1 - sp)).sum(1)) / k / 9
        hw = Z * np.sqrt(v)
        return d - hw > 0, d + hw < 0, np.median(hw)
    b1, w1, hw = one()
    b2, w2, _ = one()
    return {"HUMAN_BETTER": round(float(b1.mean()), 3), "OWN_BETTER": round(float(w1.mean()), 4),
            "HUMAN_BETTER_replicated": round(float((b1 & b2).mean()), 3), "halfwidth_pp": round(float(hw) * 100, 2)}


def h2h_halfwidth(n):
    """The in-house head-to-head (S-H vs S-P) half-width at n games, around 0.5."""
    return round(Z * np.sqrt(0.25 / n) * 100, 2)


def x60_fixed(rng, d_f, d_p, seed_sd=0.0, games=420, base=0.47, band=0.03, reps=100_000):
    """X60's revised primary: F + P72 pooled equal-weight (K = 2), AM-5's widening, the per-seed sign rule, plus the
    heterogeneity guard (each panel's pooled point estimate > 0 for a GAIN). Two looks as main.x35.d1."""
    K = 2
    d = np.array([d_f, d_p])

    def arm(eff):
        s = rng.normal(0.0, seed_sd, (reps, 5))
        p = np.clip(base + eff[None, None, :] + s[:, :, None], 0.01, 0.99)
        return rng.binomial(games, p) / games
    on, off = arm(d), arm(np.zeros(K))
    looks = []
    for look, ns in ((1, 3), (2, 5)):
        a, c = on[:, :ns, :], off[:, :ns, :]
        pa, pc = a.mean(1), c.mean(1)
        va = (pa * (1 - pa) / (games * ns)).sum(1) / K ** 2
        vc = (pc * (1 - pc) / (games * ns)).sum(1) / K ** 2

        def ratio(x):
            e = (x * (1 - x) / games).sum(2).mean(1) / K ** 2
            return x.mean(2).var(1, ddof=1) / e
        f = np.sqrt(np.maximum(1.0, (ratio(a) + ratio(c)) / 2))
        dd = pa.mean(1) - pc.mean(1)
        hw = Z * np.sqrt(va + vc) * f
        sign_ok = ((a.mean(2) - c.mean(2)) > 0).sum(1) >= (2 if look == 1 else 3)
        het_ok = ((pa - pc) > 0).all(1)
        lo, hi = dd - hw, dd + hw
        cls = np.where((lo > 0) & sign_ok & het_ok, "GAIN", np.where(hi < 0, "LOSS",
                       np.where((lo > -band) & (hi < band), "NULL", "UNDECIDED")))
        looks.append((cls, dd, hw))
    (c1, d1, h1), (c2, _, h2) = looks
    final = np.where((c1 == "UNDECIDED") & (d1 > 0), c2, c1)
    out = {k: round(float((final == k).mean()), 3) for k in ("GAIN", "NULL", "LOSS", "UNDECIDED")}
    out["halfwidth_look2_pp"] = round(float(np.median(h2)) * 100, 2)
    return out


def contrast_power(delta, games=420, seeds=5, panels=2, base=0.47):
    """A between-ARM contrast (H-TEAMS - D1, H-BC - P-BC) on the same pooled panels: both arms binomial at
    games x seeds per panel. Returns the half-width and P(LB > 0) at the true contrast ``delta``."""
    n = games * seeds
    se = np.sqrt(2 * base * (1 - base) / n / panels)
    return {"halfwidth_pp": round(Z * se * 100, 2), "P(LB>0)": round(float(1 - norm.cdf(Z - delta / se)), 3)}


def main():
    rng = np.random.default_rng(SEED)
    r = {"seed": SEED, "x59s": {}, "x60_fixed": {}, "contrasts": {}, "positive_control": {}, "first_reads": {}}
    for n_u in (1200, 2400, 3600):
        for d in (0.0, 0.03, 0.04, 0.05):
            r["x59s"][f"U games/student={n_u} delta={d*100:.0f}pp base=0.30"] = x59s(rng, d, n_u)
        r["x59s"][f"MDE80_pp at U games/student={n_u} (p=0.5, worst case)"] = round(
            (Z + Z80) * np.sqrt(2 * 0.25 / n_u) * 100, 2)
    r["x59s"]["in-house S-H vs S-P half-width (pp) at 2,000 / 4,000 / 8,000 games per panel"] = [
        h2h_halfwidth(n) for n in (2000, 4000, 8000)]
    for g in (420, 630):
        for sd in (0.0, 0.027):
            for (df_, dp) in ((0.0, 0.0), (0.02, 0.02), (0.025, 0.025), (0.04, 0.0), (0.03, 0.01)):
                r["x60_fixed"][f"games/panel/seed={g} seed_sd={sd*100:.1f}pp dF={df_*100:.1f} dP72={dp*100:.1f}"] = \
                    x60_fixed(rng, df_, dp, sd, games=g)
    # the drafted frame at 420 games (three panels, no heterogeneity guard) for comparison with REVIEW B's table
    for d in (0.0, 0.025):
        r["x60_fixed"][f"(REVIEW B frame, 3 panels, 420 games) delta={d*100:.1f}pp"] = RB.x60(rng, d, 0.0, 420)
    for d in (0.0, 0.02, 0.03, 0.04):
        r["contrasts"][f"arm contrast on F+P72, 630 x 5 seeds, delta={d*100:.0f}pp"] = contrast_power(d, games=630)
    # H-p: the practised half of H (36 teams) at 420 games per seed final; 5 seeds; one panel
    for d in (0.03, 0.05, 0.08):
        r["positive_control"][f"H-p ON-OFF, 420 x 5 seeds, delta={d*100:.0f}pp"] = contrast_power(d, panels=1)
    # the first reads: P* vs a Metamon human-only model, per cell and pooled U, at the planned n
    for n in (400, 1200):
        r["first_reads"][f"P* vs human-only model: half-width (pp) at n={n}, p=0.5 / p=0.7"] = [
            round(Z * np.sqrt(p * (1 - p) / n) * 100, 2) for p in (0.5, 0.7)]
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
