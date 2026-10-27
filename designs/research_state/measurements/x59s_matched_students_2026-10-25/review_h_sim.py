"""REVIEW H (causal): X59-S's registered rule simulated where power.py did not look (REVIEW_H_causal.md).

Run: python review_h_sim.py [reps]  (numpy + scipy, CPU, about a minute at 40,000 reps). Prints one JSON document;
the review quotes results/review_h_sim.json (40,000 reps, seed 20261027).

A vectorised copy of x59s_read.read / classify: the z interval with the 1.5 pp seed floor, sigma-hat pooled over all
three arms, the guard that hands the contrast to Welch, NO-DIFFERENCE-AT-4PP; plus the floor pilot (units 0-3, 80
games a panel: the floor fires iff EVERY arm's mean is < 0.15). A student's true rate is p_arm + N(0, sigma_arm),
shared over its three panels; 400 games a panel, the first 80 of them the pilot's. What power.py does not cover:
bases of 0.03-0.20 (the PREREG predicts 0.10-0.25), seed SDs that differ by arm, a 5 pp seed SD, the pilot's
conditioning, and contrasts between arms on either side of the floor bar."""
import json, sys
import numpy as np
from scipy.stats import chi2, t as tdist
Z = 1.959963985; SMAX = 0.015; EQ = 0.04; TIE = 1e-9; P = 3; K = 5
G_FULL, G_PILOT = 400, 80          # games per (student, panel): 20 units x 20; units 0-3
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40000

def arm(rng, p, sig, n_reps):
    """true per-student rate, then pilot games (first 80) and full games (400 = pilot + 320 more)."""
    q = np.clip(p + rng.normal(0, sig, (n_reps, K)), 0.002, 0.998)[..., None].repeat(P, -1)
    pil = rng.binomial(G_PILOT, q); rest = rng.binomial(G_FULL - G_PILOT, q)
    return pil / G_PILOT, (pil + rest) / G_FULL      # per-panel rates (reps, K, P)

def rule(A, B, C):
    """x59s_read.contrast(A, B) with sigma-hat pooled over A, B, C; returns lo, hi, guard."""
    ra, rb, rc = A.mean(-1), B.mean(-1), C.mean(-1)
    sb = lambda X: (X * (1 - X) / G_FULL).sum(-1) / P ** 2
    W = (ra.var(1, ddof=1) + rb.var(1, ddof=1) + rc.var(1, ddof=1)) / 3
    b = (sb(A).sum(1) + sb(B).sum(1) + sb(C).sum(1)) / (3 * K)
    df = 3 * (K - 1)
    s2 = np.maximum(0, W - b); lb = np.maximum(0, W * df / chi2.ppf(0.95, df) - b)
    guard = lb > SMAX ** 2
    d = ra.mean(1) - rb.mean(1)
    va, vb = ra.var(1, ddof=1) / K, rb.var(1, ddof=1) / K
    dfw = (va + vb) ** 2 / (va ** 2 / (K - 1) + vb ** 2 / (K - 1))
    hw_w = tdist.ppf(0.975, dfw) * np.sqrt(va + vb)
    hw_z = Z * np.sqrt((sb(A).sum(1) + sb(B).sum(1)) / K ** 2 + 2 * np.maximum(SMAX ** 2, s2) / K)
    hw = np.where(guard, hw_w, hw_z)
    return d - hw, d + hw, guard

def cls(lo, hi):
    better = lo > TIE; worse = hi < -TIE
    nod = (~better) & (~worse) & (lo > -EQ + TIE) & (hi < EQ - TIE)
    return better, worse, nod

def run(rng, pH, pP, pS, sH, sP, sS):
    (pilH, H), (pilP, Pp), (pilS, S) = arm(rng, pH, sH, REPS), arm(rng, pP, sP, REPS), arm(rng, pS, sS, REPS)
    lo, hi, g = rule(H, S, Pp)                     # delta_data = S-H - S-S
    llo, lhi, _ = rule(H, Pp, S)                    # delta_labels = S-H - S-P
    floor = (pilH.mean(-1).mean(1) < 0.15) & (pilP.mean(-1).mean(1) < 0.15) & (pilS.mean(-1).mean(1) < 0.15)
    go = ~floor
    b, w, n = cls(lo, hi); lb_, lw_, ln_ = cls(llo, lhi)
    f = lambda x, m=None: round(float(x.mean() if m is None else (x[m].mean() if m.any() else np.nan)), 4)
    return {"P(floor fires)": f(floor), "guard": f(g),
            "data better/worse/nodiff": [f(b), f(w), f(n)],
            "data | continue: better/worse/nodiff": [f(b, go), f(w, go), f(n, go)],
            "labels better/worse/nodiff": [f(lb_), f(lw_), f(ln_)],
            "labels | continue: better/worse/nodiff": [f(lb_, go), f(lw_, go), f(ln_, go)]}

rng = np.random.default_rng(20261027)
out = {}
S = [  # (name, pH, pP, pS, sigH, sigP, sigS)
 ("null p=.10 s=1.5", .10, .10, .10, .015, .015, .015),
 ("null p=.14 s=2 (pilot edge)", .14, .14, .14, .02, .02, .02),
 ("null p=.15 s=3 (pilot edge)", .15, .15, .15, .03, .03, .03),
 ("null p=.20 s=3", .20, .20, .20, .03, .03, .03),
 ("null p=.20 s=5", .20, .20, .20, .05, .05, .05),
 ("null p=.20 hetero sH=4 sP=sS=1", .20, .20, .20, .04, .01, .01),
 ("null p=.20 hetero sH=5 sP=sS=1", .20, .20, .20, .05, .01, .01),
 ("null p=.30 hetero sH=5 sP=sS=1", .30, .30, .30, .05, .01, .01),
 ("S-S +3 p=.20 s=2", .20, .20, .23, .02, .02, .02),
 ("S-S +4 p=.20 s=3", .20, .20, .24, .03, .03, .03),
 ("S-S +4 p=.20 s=5", .20, .20, .24, .05, .05, .05),
 ("S-H +4 p=.20 s=2", .24, .20, .20, .02, .02, .02),
 ("S-H 2x rel (.06 vs .03) s=1 [labels: S-P=.03]", .06, .03, .17, .01, .01, .01),
 ("S-H .08 vs S-P .05, S-S .18 s=1.5 (floor-compressed labels)", .08, .05, .18, .015, .015, .015),
 ("S-H .12 vs S-P .09, S-S .16 s=1.5", .12, .09, .16, .015, .015, .015),
 ("S-H .05 vs S-P .03 (x1.7), S-S .20 s=1", .05, .03, .20, .01, .01, .01),
 ("S-H .04 S-P .04 S-S .13 s=1.5 (3x gap under the bar)", .04, .04, .13, .015, .015, .015),
 ("S-H .13 S-P .13 S-S .08 s=1.5 (gap under the bar)", .13, .13, .08, .015, .015, .015),
 ("S-H +4 p=.20 s=3", .24, .20, .20, .03, .03, .03),
 ("S-H +4 p=.20 s=5", .24, .20, .20, .05, .05, .05),
 ("S-H +6 p=.20 s=3", .26, .20, .20, .03, .03, .03),
 ("S-H +6 p=.20 s=5", .26, .20, .20, .05, .05, .05),
 ("null p=.20 hetero sS=5 sH=sP=1", .20, .20, .20, .01, .01, .05),
]
for name, *a in S:
    out[name] = run(rng, *a)
print(json.dumps({"seed": 20261027, "reps": REPS, "games_per_panel": G_FULL, "pilot_games_per_panel": G_PILOT,
                  "scenarios": out}, indent=1))
