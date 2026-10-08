"""Motivated-cognition read, ANALYSIS half: the pre-registered decision rule (README §0.4–0.5) over
``rows/`` → ``result.json`` + ``result.md``. Pure numpy/scipy, no model; runs anywhere.

    python3 analyze.py [--rows rows] [--boot 1000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

SEEDS = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]
ARMS = ["blob", "fm"]
TIE_EPS = 1e-6
BOUND_EPS = 1e-9
BOOT_SEED = 20261007
MATERIAL = {"T1": 0.02, "T2": 0.02, "T3": 0.01}
TEST_COLS = {"T1": "e1", "T2": "d2", "T3": "e3"}
CALIB = {
    "presence": ["ece", "brier", "logscore", "citl"],
    "moves": ["ece", "brier", "logscore", "citl"],
    "item": ["ece_top1", "citl_top1", "brier", "logscore", "acc"],
    "nature": ["ece_top1", "citl_top1", "brier", "logscore", "acc"],
    "hptype": ["ece_top1", "citl_top1", "brier", "logscore", "acc"],
    "cb": ["ece", "citl", "brier", "mean_p", "mean_y"],
}


def tci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return {"mean": float(m), "lo": float(m - h), "hi": float(m + h), "per_seed": [float(v) for v in x]}


def load(rows: Path):
    meta = np.load(rows / "_bank_meta.npz")
    bi, free, masks = meta["battle_index"], meta["free"], meta["masks"]
    elig_base = free & masks[:, 6:11].any(1) & masks[:, 0:6].any(1)
    data = {}
    for arm in ARMS:
        for s in SEEDS:
            lab = f"{arm}_s{s}"
            R = dict(np.load(rows / f"{lab}.npz"))
            C = json.loads((rows / f"{lab}.calib.json").read_text())
            elig = elig_base & (R["margin"] >= TIE_EPS)
            stay = R["top1"] >= 6
            data[lab] = {"R": R, "C": C, "elig": elig, "stay": stay,
                         "n_tie": int((elig_base & (R["margin"] < TIE_EPS)).sum())}
    return bi, data


def per_battle(bi, nb, mask, val):
    """(Σ val, count) per battle over rows in ``mask``."""
    v = np.where(mask, val, 0.0)
    return np.bincount(bi, v, nb), np.bincount(bi, mask.astype(float), nb)


def test_parts(test, d, col, bi, nb):
    """Per-battle sufficient statistics for one (label, column): ((S_a, N_a), (S_b, N_b)) — group b is
    None for T2 (a plain mean)."""
    R = d["R"]
    v = R[f"{TEST_COLS[test]}_{col}"]
    ok = d["elig"] & np.isfinite(v)
    if test == "T2":
        return per_battle(bi, nb, ok, v), None
    return per_battle(bi, nb, ok & d["stay"], v), per_battle(bi, nb, ok & ~d["stay"], v)


def delta(parts, w):
    (Sa, Na), b = parts
    ma = (w @ Sa) / (w @ Na)
    if b is None:
        return ma
    Sb, Nb = b
    return ma - (w @ Sb) / (w @ Nb)


def verdict(arm_ci, del_ci, arm_bs, del_bs, M):
    def below(ci):
        return ci["hi"] < -BOUND_EPS

    def above(ci):
        return ci["lo"] > BOUND_EPS
    mag = abs(del_ci["mean"])
    material = mag - M > BOUND_EPS
    if (arm_ci["mean"] < 0 and del_ci["mean"] < 0 and below(arm_ci) and below(del_ci)
            and below(arm_bs) and below(del_bs) and material):
        return "BITING"
    if (arm_ci["mean"] > 0 and del_ci["mean"] > 0 and above(arm_ci) and above(del_ci)
            and above(arm_bs) and above(del_bs) and material):
        return "ANTI-self-serving"
    return "NOT DETECTED"


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--rows", default=str(here / "rows"))
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    bi, data = load(Path(a.rows))
    nb = int(bi.max()) + 1
    rng = np.random.default_rng(BOOT_SEED)
    W = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float) for _ in range(a.boot)])
    one = np.ones(nb)
    out = {"tests": {}, "calibration": {}, "arm_compare": {}, "counts": {}}
    for arm in ARMS:
        labs = [f"{arm}_s{s}" for s in SEEDS]
        out["counts"][arm] = {
            "eligible_rows": [int(data[l]["elig"].sum()) for l in labs],
            "rule8_tie_excluded": [data[l]["n_tie"] for l in labs],
            "stay_share": [float(data[l]["stay"][data[l]["elig"]].mean()) for l in labs],
        }
        for test in ("T1", "T2", "T3"):
            res = {}
            point = {c: [] for c in ("arm", "prior", "delta")}
            boot = {c: np.zeros(a.boot) for c in ("arm", "prior", "delta")}
            ns = []
            for lab in labs:
                pa = test_parts(test, data[lab], "arm", bi, nb)
                pp = test_parts(test, data[lab], "prior", bi, nb)
                da, dp = delta(pa, one), delta(pp, one)
                point["arm"].append(da); point["prior"].append(dp); point["delta"].append(da - dp)
                ba, bp = delta(pa, W), delta(pp, W)
                boot["arm"] += ba / len(labs); boot["prior"] += bp / len(labs)
                boot["delta"] += (ba - bp) / len(labs)
                if pa[1] is None:
                    ns.append([int(pa[0][1].sum())])
                else:
                    ns.append([int(pa[0][1].sum()), int(pa[1][1].sum())])
            for c in point:
                res[c] = tci(point[c])
                lo, hi = np.percentile(boot[c], [2.5, 97.5])
                res[c]["boot_lo"], res[c]["boot_hi"] = float(lo), float(hi)
            res["n_rows_per_seed"] = ns
            bs = {c: {"lo": res[c]["boot_lo"], "hi": res[c]["boot_hi"]} for c in ("arm", "delta")}
            res["verdict"] = verdict(res["arm"], res["delta"], bs["arm"], bs["delta"], MATERIAL[test])
            out["tests"].setdefault(test, {})[arm] = res
        cal = {}
        for head, mets in CALIB.items():
            cal[head] = {}
            for m in mets:
                av = [data[l]["C"][f"{head}_arm"][m] for l in labs]
                pv = [data[l]["C"][f"{head}_prior"][m] for l in labs]
                cal[head][m] = {"arm": tci(av), "prior": tci(pv), "delta": tci(np.subtract(av, pv))}
            cal[head]["n_units"] = [data[l]["C"][f"{head}_arm"]["n"] for l in labs]
            # pooled reliability (binary) / top-1 bins (categorical) across seeds
            key = "rel" if "rel" in data[labs[0]]["C"][f"{head}_arm"] else "bins"
            pooled = {}
            for c in ("arm", "prior"):
                agg = None
                for l in labs:
                    r = data[l]["C"][f"{head}_{c}"][key]
                    vals = {k: np.asarray(v, float) for k, v in r.items() if k != "edges"}
                    agg = vals if agg is None else {k: agg[k] + vals[k] for k in agg}
                pooled[c] = {k: v.tolist() for k, v in agg.items()}
            cal[head]["pooled_bins"] = pooled
        ev = [data[l]["C"]["ev_mae"] for l in labs]
        cal["ev_mae"] = {"arm": tci([e["arm"] for e in ev]), "prior": tci([e["prior"] for e in ev]),
                         "delta": tci([e["arm"] - e["prior"] for e in ev])}
        out["calibration"][arm] = cal
    # Part 3: blob vs fixed_mass (Welch, two-sided)
    for test in ("T1", "T2", "T3"):
        b = out["tests"][test]["blob"]["delta"]["per_seed"]
        f = out["tests"][test]["fm"]["delta"]["per_seed"]
        r = stats.ttest_ind(b, f, equal_var=False)
        out["arm_compare"][test] = {"blob_minus_fm": float(np.mean(b) - np.mean(f)), "t": float(r.statistic),
                                    "p": float(r.pvalue),
                                    "blob_more_exposed": bool(np.mean(b) < np.mean(f) and r.pvalue < 0.05)}
    for head, mets in CALIB.items():
        for m in mets:
            b = out["calibration"]["blob"][head][m]["delta"]["per_seed"]
            f = out["calibration"]["fm"][head][m]["delta"]["per_seed"]
            r = stats.ttest_ind(b, f, equal_var=False)
            out["arm_compare"][f"calib.{head}.{m}"] = {"blob_minus_fm": float(np.mean(b) - np.mean(f)),
                                                       "t": float(r.statistic), "p": float(r.pvalue)}
    out["overall"] = ("BITES" if any(out["tests"][t][a_]["verdict"] == "BITING"
                                     for t in out["tests"] for a_ in ARMS) else "NOT DETECTED")
    here_out = Path(a.rows).parent
    (here_out / "result.json").write_text(json.dumps(out, indent=1))
    (here_out / "result.md").write_text(render(out))
    print(render(out))


def f(x, n=4):
    return f"{x:+.{n}f}"


def render(o) -> str:
    L = ["# Motivated-cognition read: result (generated by `analyze.py`)", ""]
    L += ["## Self-serving tests (Δ < 0 = self-serving; CI: across-seed t95 | battle bootstrap 95 %)", "",
          "| test | arm | Δ_arm | Δ_prior | Δ_delta = arm − prior | rows/seed | verdict |", "|---|---|---|---|---|---|---|"]
    for t, by in o["tests"].items():
        for arm, r in by.items():
            cell = {c: f"{f(r[c]['mean'])} [{f(r[c]['lo'])}, {f(r[c]['hi'])}] \\| [{f(r[c]['boot_lo'])}, {f(r[c]['boot_hi'])}]"
                    for c in ("arm", "prior", "delta")}
            ns = r["n_rows_per_seed"]
            nr = f"{min(map(sum, ns))}–{max(map(sum, ns))}"
            L.append(f"| {t} | {arm} | {cell['arm']} | {cell['prior']} | {cell['delta']} | {nr} | **{r['verdict']}** |")
    L += ["", f"**Overall: {o['overall']}**", "", "## Blob vs fixed_mass (Welch on per-seed Δ_delta)", "",
          "| test | blob − fm | t | p | blob more exposed |", "|---|---|---|---|---|"]
    for t in ("T1", "T2", "T3"):
        c = o["arm_compare"][t]
        L.append(f"| {t} | {f(c['blob_minus_fm'])} | {c['t']:+.2f} | {c['p']:.3f} | {c['blob_more_exposed']} |")
    L += ["", "## Calibration (across-seed mean [t95]; ARM, PRIOR, ARM − PRIOR)", "",
          "| head | metric | blob ARM | blob PRIOR | blob Δ | fm ARM | fm Δ | blob−fm Δ (p) |",
          "|---|---|---|---|---|---|---|---|"]
    for head, mets in CALIB.items():
        for m in mets:
            b = o["calibration"]["blob"][head][m]
            fm = o["calibration"]["fm"][head][m]
            cmp_ = o["arm_compare"][f"calib.{head}.{m}"]

            def ci(x):
                return f"{x['mean']:.4f} [{x['lo']:.4f}, {x['hi']:.4f}]"
            L.append(f"| {head} | {m} | {ci(b['arm'])} | {ci(b['prior'])} | {ci(b['delta'])} | {ci(fm['arm'])} | "
                     f"{ci(fm['delta'])} | {f(cmp_['blob_minus_fm'])} ({cmp_['p']:.3f}) |")
    for arm in ARMS:
        e = o["calibration"][arm]["ev_mae"]
        L.append(f"| ev | MAE (EV points), {arm} | arm {e['arm']['mean']:.2f} | prior {e['prior']['mean']:.2f} | "
                 f"{e['delta']['mean']:+.2f} [{e['delta']['lo']:+.2f}, {e['delta']['hi']:+.2f}] | | | |")
    L += ["", "## Counts", ""]
    for arm, c in o["counts"].items():
        L.append(f"- {arm}: eligible rows {c['eligible_rows']}; rule-8 ties excluded {c['rule8_tie_excluded']}; "
                 f"STAY share {[round(x, 3) for x in c['stay_share']]}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
