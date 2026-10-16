"""A DESCRIPTIVE rectangular-cross statistic for the toy crosses (PREREG.md §4). NOT backlog T40 (T40 is the registered
m x n extension of main.h2h.cross.cross_stat, with its own pre-registration); this is the simplest seeds-as-units
analogue, declared here before any toy-cross game and self-checked on constructed cases.

h[i][j] (pp) = toy seed i's score against EB seed j (one mirrored cell). m rows (toy seeds), n columns (EB seeds).
  Δ̂ = mean_ij h_ij − 50;  R_i = row means, C_j = column means;
  V̂ = s²_R / m + s²_C / n   (the square cross_stat's (s²_R + s²_C)/n when m = n);
  df = Welch–Satterthwaite: V̂² / ((s²_R/m)²/(m−1) + (s²_C/n)²/(n−1))   (non-integer; scipy's t takes it);
  the two-sided 90 % interval Δ̂ ± t_{0.95, df} √V̂ (the R-T reads' registered level, readiness §3.5), and the 95 % one.
Under the additive model h_ij = 50 + Δ + a_i − b_j + e_ij, E[V̂] = σ²_a/m + σ²_b/n + 2σ²_e/(mn) while Var(Δ̂) =
σ²_a/m + σ²_b/n + σ²_e/(mn): V̂ counts the cell noise twice, so the interval is CONSERVATIVE in that term.
It differs from the square cross_stat in df (Welch, not 2(n−1)), so it is NOT bit-identical to it on a square matrix
(V̂ and Δ̂ are; the self-check pins that).
Per-seed (descriptive): row i's mean − 50 with the one-sample t interval over its n cells (df n − 1); that interval
holds the EB seeds' spread + cell noise, not the toy seed's own training noise.
Rule 8: a bound within 1e-9 of 0 is reported as touching it (`near_zero`), never as excluding it.
Usage: rect_stat.py self-check   (exit 0 iff every constructed case passes)."""
import json, math, random, sys
from scipy.stats import t as student_t

EPS = 1e-9


def _mean(x): return math.fsum(x) / len(x)


def _var(x):
    m = _mean(x); return math.fsum((v - m) ** 2 for v in x) / (len(x) - 1)


def rect_stat(h, conf=0.90):
    m = len(h); n = len(h[0]) if m else 0
    if m < 2 or n < 2 or any(len(r) != n for r in h) or not all(math.isfinite(v) for r in h for v in r):
        raise ValueError(f"needs an m x n finite matrix with m, n >= 2 (got {m} rows of {[len(r) for r in h]})")
    R = [_mean(r) for r in h]; C = [_mean([h[i][j] for i in range(m)]) for j in range(n)]
    d = _mean([v for r in h for v in r]) - 50.0
    a, b = _var(R) / m, _var(C) / n; V = a + b
    if not V > 0: raise ValueError("zero variance across both sides' seed means")
    df = V * V / ((a * a / (m - 1) if m > 1 else 0) + (b * b / (n - 1) if n > 1 else 0))
    se = math.sqrt(V)
    out = {"m": m, "n": n, "delta_hat_pp": d, "row_means_pp": R, "col_means_pp": C, "s2_rows": _var(R), "s2_cols": _var(C),
           "v_hat": V, "se_pp": se, "df_welch": df}
    for lvl in (conf, 0.95):
        q = float(student_t.ppf(0.5 + lvl / 2, df)); lo, hi = d - q * se, d + q * se
        out[f"ci{int(round(lvl * 100))}_pp"] = [lo, hi]
        out[f"ci{int(round(lvl * 100))}_excludes_0"] = (hi < -EPS) or (lo > EPS)
        out[f"ci{int(round(lvl * 100))}_near_zero"] = abs(hi) <= EPS or abs(lo) <= EPS
    per = []
    for i, r in enumerate(h):
        s = math.sqrt(_var(r) / n); q = float(student_t.ppf(0.5 + conf / 2, n - 1))
        per.append({"row": i, "mean_minus_50_pp": R[i] - 50.0, f"ci{int(round(conf * 100))}_pp": [R[i] - 50 - q * s, R[i] - 50 + q * s],
                    "se_pp": s})
    out["per_seed"] = per
    return out


def _self_check():
    ok = []
    # 1. square: Δ̂ and V̂ equal main.h2h.cross.cross_stat's (df differs by declaration)
    try:
        from main.h2h.cross import cross_stat
        H = [[48.0, 51.0, 47.5], [52.0, 55.5, 49.0], [45.0, 50.0, 44.0]]
        s, r = cross_stat(H), rect_stat(H)
        ok.append(("square delta/v match cross_stat", abs(s.delta_hat - r["delta_hat_pp"]) < 1e-12 and abs(s.v_hat - r["v_hat"]) < 1e-12))
    except ImportError:
        ok.append(("square delta/v match cross_stat (main.h2h importable)", False))
    # 2. a noiseless additive 3 x 8 matrix: Δ̂ exact, row / col means recover the effects
    a = [-2.0, 0.0, 2.0]; b = [-3.0, -1.0, 0.0, 1.0, 3.0, 2.0, -2.0, 0.0]; D = -10.0
    H = [[50 + D + ai - bj for bj in b] for ai in a]; r = rect_stat(H)
    ok.append(("additive delta exact", abs(r["delta_hat_pp"] - D) < 1e-12))
    ok.append(("additive row means", all(abs(r["row_means_pp"][i] - (50 + D + a[i] - _mean(b))) < 1e-12 for i in range(3))))
    ok.append(("additive interval excludes 0", r["ci90_excludes_0"] and r["ci90_pp"][1] < 0))
    # 3. Monte-Carlo coverage of the 90 % interval at m=3, n=8 under the additive model (fixed seed: deterministic);
    #    the conservative V̂ must cover >= 0.88 (sampling error of 4,000 draws ~0.5 pp)
    rng = random.Random(20261016); hit = 0; N = 4000
    for _ in range(N):
        aa = [rng.gauss(0, 2.0) for _ in range(3)]; bb = [rng.gauss(0, 2.5) for _ in range(8)]
        H = [[50 - 8.0 + x - y + rng.gauss(0, 2.2) for y in bb] for x in aa]
        lo, hi = rect_stat(H)["ci90_pp"]; hit += lo <= -8.0 <= hi
    cov = hit / N
    ok.append((f"MC coverage 90% interval = {cov:.4f} >= 0.88", cov >= 0.88))
    # 4. refusals: 1 x 8 and a ragged matrix
    for name, H in (("refuses 1 x 8", [[50.0] * 8]), ("refuses ragged", [[50.0, 51.0], [49.0]])):
        try: rect_stat(H); ok.append((name, False))
        except ValueError: ok.append((name, True))
    # 5. rule 8: an interval whose upper end is exactly 0 is near_zero and does not exclude 0
    H = [[40.0, 60.0], [40.0, 60.0]]  # Δ̂ 0, V from columns only
    r = rect_stat(H); ok.append(("Δ̂ 0 interval does not exclude 0", not r["ci90_excludes_0"]))
    for k, v in ok: print(("PASS " if v else "FAIL ") + k)
    return all(v for _, v in ok)


if __name__ == "__main__":
    if sys.argv[1:] == ["self-check"]:
        sys.exit(0 if _self_check() else 1)
    print(json.dumps(rect_stat(json.loads(sys.argv[1])), indent=1))
