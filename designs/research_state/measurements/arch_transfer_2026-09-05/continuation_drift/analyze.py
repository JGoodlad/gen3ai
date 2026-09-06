"""H9 analysis — exponents, bootstraps, cosines, output twin. Pure numpy over drift_{gen,v8}.json.

No models, no torch, no battles. Reads only what drift.py wrote.

Statistics, exactly as pre-registered in README.md:
  P1  b in |Delta theta| ~ t^b, per arm (v8: exact 2-point slope; gen: OLS over 3 and the matched
      2-point over its OUTER depths) and pooled; bootstrap over ARMS (n=3, crude by construction);
      the complete-separation rule and the exact 20-arrangement permutation p (floor 0.10).
  P2  replicate cosine cos(Delta_X, Delta_Y) at matched depth, per group.
  P3  c in KL(parent||arm) ~ t^c, the output-side twin; cluster bootstrap over the 24 teams for
      the LEVEL, over arms for the exponent.
  E1  EXPLORATORY, not pre-registered: cos(Delta_t1, Delta_t2) WITHIN one arm. A random walk
      predicts sqrt(t1/t2); a directed drift predicts 1. Third independent directedness estimator.

Run: nice -n 10 python analyze.py --out analysis.json
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260906
NBOOT = 20000
ARMS = ["A", "B", "C"]


def ols_slope(t, y):
    """OLS slope of log y on log t. Exact two-point slope when len(t) == 2."""
    lt, ly = np.log(np.asarray(t, float)), np.log(np.asarray(y, float))
    lt = lt - lt.mean()
    return float((lt @ (ly - ly.mean())) / (lt @ lt))


def load(era):
    with open(os.path.join(HERE, f"drift_{era}.json")) as f:
        return json.load(f)


def fit_points(J, arm):
    """[(t, rec)] for one arm's FIT depths, in increasing t."""
    out = [(r["t"], r) for k, r in J["arms"].items()
           if r["arm"] == arm and r.get("fit_point", True)]
    return sorted(out)


def arm_slopes(J, key_fn, depths=None):
    """{arm: slope} of log(key_fn(rec)) on log t over the arm's fit points (optionally a subset)."""
    out = {}
    for arm in ARMS:
        pts = fit_points(J, arm)
        if depths is not None:
            pts = [(t, r) for t, r in pts if r["depth"] in depths]
        out[arm] = ols_slope([t for t, _ in pts], [key_fn(r) for _, r in pts])
    return out


def pooled_slope(J, key_fn, depths=None):
    t, y = [], []
    for arm in ARMS:
        for tt, r in fit_points(J, arm):
            if depths is None or r["depth"] in depths:
                t.append(tt)
                y.append(key_fn(r))
    return ols_slope(t, y)


def boot_mean(vals, rng, nboot=NBOOT):
    v = np.asarray(vals, float)
    d = rng.integers(0, len(v), size=(nboot, len(v)))
    m = v[d].mean(axis=1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def perm_p(a, b):
    """Exact two-sided permutation p on the mean difference over ALL splits of a+b into |a|,|b|."""
    pool = list(a) + list(b)
    obs = abs(np.mean(a) - np.mean(b))
    n = len(a)
    hits = tot = 0
    for idx in itertools.combinations(range(len(pool)), n):
        x = [pool[i] for i in idx]
        y = [pool[i] for i in range(len(pool)) if i not in idx]
        tot += 1
        if abs(np.mean(x) - np.mean(y)) >= obs - 1e-12:
            hits += 1
    return hits / tot, tot


def cosine_pairs(J, depth):
    """{(X,Y): {group: cos}} for the three replicate pairs at one depth label."""
    keys = J["_meta"]["keys"]
    idx = {k: i for i, k in enumerate(keys)}
    out = {}
    for x, y in itertools.combinations(ARMS, 2):
        kx, ky = f"{x}@{depth}", f"{y}@{depth}"
        if kx not in idx or ky not in idx:
            continue
        out[f"{x}.{y}"] = {g: float(J["cosine"][g][idx[kx]][idx[ky]]) for g in J["cosine"]}
    return out


def within_arm_cos(J, d_early, d_late):
    """EXPLORATORY: {arm: {group: cos(Delta_early, Delta_late)}} + the diffusive sqrt(t1/t2)."""
    keys = J["_meta"]["keys"]
    idx = {k: i for i, k in enumerate(keys)}
    out = {}
    for arm in ARMS:
        ke, kl = f"{arm}@{d_early}", f"{arm}@{d_late}"
        te, tl = J["arms"][ke]["t"], J["arms"][kl]["t"]
        out[arm] = {"t_early": te, "t_late": tl,
                    "diffusive_prediction": float(np.sqrt(te / tl)),
                    **{g: float(J["cosine"][g][idx[ke]][idx[kl]]) for g in J["cosine"]}}
    return out


def team_cluster_boot(J, arm, depth, rng, nboot=NBOOT):
    r = J["arms"][f"{arm}@{depth}"]
    v = np.array(list(r["kl_per_team"].values()), float)
    d = rng.integers(0, len(v), size=(nboot, len(v)))
    m = v[d].mean(axis=1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "analysis.json"))
    a = ap.parse_args(argv)
    rng = np.random.default_rng(SEED)

    J = {e: load(e) for e in ("gen", "v8")}
    groups = J["gen"]["_meta"]["groups"]
    res = {"_meta": {"seed": SEED, "nboot": NBOOT,
                     "sources": {e: f"drift_{e}.json" for e in J},
                     "n_params": {e: J[e]["_meta"]["n_params"] for e in J},
                     "parent_steps": {e: J[e]["_meta"]["parent_steps"] for e in J}}}

    # --- depth grid + the v8 same-depth agreement check ----------------------------------------
    grid = {}
    for e in J:
        grid[e] = {k: dict(t=r["t"], steps=r["steps"], depth=r["depth"], arm=r["arm"],
                           l2=r["l2"], rel_l2=r["rel_l2"], kl_all=r["kl_all"],
                           fit_point=r.get("fit_point", True))
                   for k, r in J[e]["arms"].items()}
    res["depth_grid"] = grid
    agree = {}
    for arm in ARMS:
        k3, kc = f"{arm}@d3", f"{arm}@d3ck"
        if kc in J["v8"]["arms"]:
            r3, rc = J["v8"]["arms"][k3], J["v8"]["arms"][kc]
            agree[arm] = dict(step_gap=r3["steps"] - rc["steps"],
                              l2_final=r3["l2"], l2_ckpt=rc["l2"],
                              rel_diff=abs(r3["l2"] - rc["l2"]) / r3["l2"],
                              identical_to_1e9=abs(r3["l2"] - rc["l2"]) < 1e-9 * r3["l2"])
    res["v8_same_depth_agreement"] = agree

    # --- P1 -----------------------------------------------------------------------------------
    P1 = {}
    variants = {
        "gen_3pt": ("gen", None),
        "gen_2pt_outer": ("gen", {"d1", "d3"}),      # MATCHED lever arm — the headline
        "v8_2pt": ("v8", {"d1", "d3"}),
    }
    for name, (era, dep) in variants.items():
        blk = {"era": era, "depths": sorted(dep) if dep else "all fit points"}
        for g in ["ALL"] + groups:
            kf = ((lambda r: r["l2"]) if g == "ALL"
                  else (lambda r, g=g: r["groups"][g]["l2"]))
            per = arm_slopes(J[era], kf, dep)
            m, lo, hi = boot_mean(list(per.values()), rng)
            blk[g] = dict(per_arm=per, mean=m, ci=[lo, hi],
                          spread=float(max(per.values()) - min(per.values())),
                          pooled=pooled_slope(J[era], kf, dep))
        P1[name] = blk
    res["P1_displacement_exponent"] = P1

    # headline contrast: matched two-point windows
    contrast = {}
    for g in ["ALL"] + groups:
        av = list(P1["v8_2pt"][g]["per_arm"].values())
        ag = list(P1["gen_2pt_outer"][g]["per_arm"].values())
        p, tot = perm_p(av, ag)
        sep = (min(av) > max(ag)) or (min(ag) > max(av))
        ci_v, ci_g = P1["v8_2pt"][g]["ci"], P1["gen_2pt_outer"][g]["ci"]
        disjoint = (ci_v[0] > ci_g[1]) or (ci_g[0] > ci_v[1])
        d = float(np.mean(av) - np.mean(ag))
        maxspread = max(P1["v8_2pt"][g]["spread"], P1["gen_2pt_outer"][g]["spread"])
        verdict = ("SIGNIFICANT" if (sep and disjoint)
                   else "WITHIN FLOOR" if (not disjoint and abs(d) < maxspread)
                   else "NOT DETECTED")
        contrast[g] = dict(v8_mean=float(np.mean(av)), gen_mean=float(np.mean(ag)),
                           diff_v8_minus_gen=d, complete_separation=bool(sep),
                           ci_disjoint=bool(disjoint), perm_p=p, perm_arrangements=tot,
                           max_within_cell_spread=maxspread, verdict=verdict)
    res["P1_contrast_matched_2pt"] = contrast

    # --- P2 -----------------------------------------------------------------------------------
    P2 = {}
    for era, deps in (("gen", ["d1", "d2", "d3"]), ("v8", ["d1", "d3"])):
        P2[era] = {}
        for dep in deps:
            pairs = cosine_pairs(J[era], dep)
            blk = {"pairs": pairs, "t": J[era]["arms"][f"A@{dep}"]["t"]}
            for g in ["ALL"] + groups:
                vals = [pairs[p][g] for p in pairs]
                m, lo, hi = boot_mean(vals, rng)
                blk[g] = dict(values=vals, mean=m, ci=[lo, hi],
                              spread=float(max(vals) - min(vals)))
            P2[era][dep] = blk
    res["P2_replicate_cosine"] = P2
    p2c = {}
    for g in ["ALL"] + groups:
        av = [P2["v8"]["d3"]["pairs"][p][g] for p in P2["v8"]["d3"]["pairs"]]
        ag = [P2["gen"]["d3"]["pairs"][p][g] for p in P2["gen"]["d3"]["pairs"]]
        p, tot = perm_p(av, ag)
        d = float(np.mean(av) - np.mean(ag))
        maxspread = max(P2["v8"]["d3"][g]["spread"], P2["gen"]["d3"][g]["spread"])
        sep = (min(av) > max(ag)) or (min(ag) > max(av))
        p2c[g] = dict(v8_mean=float(np.mean(av)), gen_mean=float(np.mean(ag)),
                      diff_v8_minus_gen=d, complete_separation=bool(sep),
                      perm_p=p, perm_arrangements=tot, max_within_cell_spread=maxspread,
                      verdict=("NOT DETECTED" if abs(d) < maxspread else
                               "DIRECTIONAL, bar-uncertain"))
    res["P2_contrast_at_end"] = p2c
    res["P2_random_walk_floor"] = {e: float(1.0 / np.sqrt(J[e]["_meta"]["n_params"])) for e in J}
    res["P2_fold_reference_cosine"] = 0.56

    # --- P3 -----------------------------------------------------------------------------------
    P3 = {}
    for era, dep in (("gen", {"d1", "d3"}), ("v8", {"d1", "d3"})):
        blk = {"depths": sorted(dep)}
        for slc in ("kl_all", "kl_taught", "kl_untaught"):
            per = arm_slopes(J[era], (lambda r, s=slc: r[s]), dep)
            m, lo, hi = boot_mean(list(per.values()), rng)
            blk[slc] = dict(per_arm=per, mean=m, ci=[lo, hi],
                            spread=float(max(per.values()) - min(per.values())),
                            pooled=pooled_slope(J[era], (lambda r, s=slc: r[s]), dep))
        blk["levels"] = {}
        for d_ in ("d1", "d3"):
            blk["levels"][d_] = {arm: dict(zip(("mean", "lo", "hi"),
                                               team_cluster_boot(J[era], arm, d_, rng)))
                                 for arm in ARMS}
        P3[era] = blk
    res["P3_output_twin"] = P3
    p3c = {}
    for slc in ("kl_all", "kl_taught", "kl_untaught"):
        av = list(P3["v8"][slc]["per_arm"].values())
        ag = list(P3["gen"][slc]["per_arm"].values())
        p, tot = perm_p(av, ag)
        d = float(np.mean(av) - np.mean(ag))
        maxspread = max(P3["v8"][slc]["spread"], P3["gen"][slc]["spread"])
        sep = (min(av) > max(ag)) or (min(ag) > max(av))
        p3c[slc] = dict(v8_mean=float(np.mean(av)), gen_mean=float(np.mean(ag)),
                        diff_v8_minus_gen=d, complete_separation=bool(sep), perm_p=p,
                        perm_arrangements=tot, max_within_cell_spread=maxspread,
                        verdict=("SIGNIFICANT" if sep and abs(d) > maxspread else
                                 "WITHIN FLOOR" if abs(d) < maxspread else "NOT DETECTED"))
    res["P3_contrast"] = p3c
    # c ~ 2b consistency
    res["P3_c_over_b"] = {
        "v8": P3["v8"]["kl_all"]["mean"] / P1["v8_2pt"]["ALL"]["mean"],
        "gen": P3["gen"]["kl_all"]["mean"] / P1["gen_2pt_outer"]["ALL"]["mean"]}

    # --- E1 EXPLORATORY -----------------------------------------------------------------------
    res["E1_within_arm_direction_persistence"] = {
        "gen": within_arm_cos(J["gen"], "d1", "d3"),
        "v8": within_arm_cos(J["v8"], "d1", "d3"),
        "_note": ("cos(Delta_t1, Delta_t2) within ONE arm. Pure random walk => sqrt(t1/t2); "
                  "pure directed drift => 1. NOT pre-registered.")}

    # --- buffers, reported separately ---------------------------------------------------------
    res["buffers_changed"] = {e: {k: sorted(r["buffers_changed"]) for k, r in J[e]["arms"].items()}
                              for e in J}

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[a] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
