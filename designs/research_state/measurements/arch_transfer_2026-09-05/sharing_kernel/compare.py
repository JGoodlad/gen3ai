"""BETWEEN-ERA comparison — is the gen-era cross/within ratio higher than the v8-era one?

Each era's `kernel.py` run answers a WITHIN-era question (is cross different from within?). The
pre-registered prediction is a BETWEEN-era one, and it needs its own test.

BOTH ERAS ARE MEASURED ON THE SAME 24 TEAMS, so every comparison here is PAIRED at the team level:

  * paired cluster BOOTSTRAP -- resample the 16 taught and the 8 untaught teams with replacement
    ONCE, and recompute both eras' statistics on that one resample. The era difference is then a
    difference on shared draws, not two independent noisy numbers subtracted.
  * paired PERMUTATION -- relabel the 24 teams 16/8 ONCE and apply the SAME relabelling to both
    eras' team matrices. The null is "the taught/untaught split carries no more contrast in one era
    than in the other".

Run: python compare.py --gen kernel_gen.json --v8 kernel_v8.json --out compare.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel import stats_from_M, GROUPS  # noqa: E402  -- imported, never re-implemented


def load(path):
    K = json.load(open(path))
    out = {}
    for g, d in K["groups"].items():
        if "team_matrix" in d:
            out[g] = (np.array(d["team_matrix"]), np.array(d["team_diag"]))
    return K, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--v8", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nboot", type=int, default=4000)
    ap.add_argument("--nperm", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260905)
    a = ap.parse_args(argv)

    Kg, Mg = load(a.gen)
    Kv, Mv = load(a.v8)
    if Kg["_meta"]["teams_order"] != Kv["_meta"]["teams_order"]:
        raise SystemExit("[cmp] the two eras' team orders differ -- the pairing would be wrong")
    teams = Kg["_meta"]["teams_order"]
    taught = [teams.index(s) for s in Kg["_meta"]["taught_teams"]]
    untaught = [teams.index(s) for s in Kg["_meta"]["untaught_teams"]]
    if sorted(Kg["_meta"]["taught_teams"]) != sorted(Kv["_meta"]["taught_teams"]):
        raise SystemExit("[cmp] the two eras' taught sets differ")
    print(f"[cmp] paired on {len(teams)} teams ({len(taught)} taught / {len(untaught)} untaught)",
          flush=True)

    rng = np.random.default_rng(a.seed)
    ti, ui = np.array(taught), np.array(untaught)
    res = {"_meta": {"gen": Kg["_meta"], "v8": Kv["_meta"],
                     "pairing": "SAME 24 teams in both eras; bootstrap resamples and label "
                                "permutations are applied ONCE and shared by both eras",
                     "nboot": a.nboot, "nperm": a.nperm, "seed": a.seed},
           "groups": {}}

    for g in ["ALL"] + GROUPS:
        if g not in Mg or g not in Mv:
            res["groups"][g] = {"status": "ZERO GRADIENT in at least one era (off the log-pi path)"}
            print(f"[cmp] {g:16s} skipped (zero gradient in at least one era)", flush=True)
            continue
        sg = stats_from_M(*Mg[g], taught, untaught)
        sv = stats_from_M(*Mv[g], taught, untaught)
        d_ratio = sg["ratio"] - sv["ratio"]
        d_cross = sg["cross"] - sv["cross"]
        d_gapn = ((sg["cross"] - sg["within_pooled"]) / sg["within_pooled"]
                  - (sv["cross"] - sv["within_pooled"]) / sv["within_pooled"])

        # ROBUSTNESS: `within_pooled` is pair-count weighted, so the 120 taught-taught pairs
        # outvote the 28 untaught-untaught ones ~4:1. If the two halves differ in internal
        # homogeneity -- and they do, the untaught 8 are balance/stall-heavy by their own selection
        # note -- that asymmetry alone moves the ratio. `ratio_halves` divides by the UNWEIGHTED
        # mean of the two halves instead, and is carried through the same paired test.
        d_ratioh = sg["ratio_halves"] - sv["ratio_halves"]

        br = np.empty(a.nboot)
        bc = np.empty(a.nboot)
        bh = np.empty(a.nboot)
        for b in range(a.nboot):
            rt = rng.choice(ti, size=len(ti), replace=True)
            ru = rng.choice(ui, size=len(ui), replace=True)
            xg = stats_from_M(*Mg[g], rt, ru)
            xv = stats_from_M(*Mv[g], rt, ru)
            br[b] = xg["ratio"] - xv["ratio"]
            bc[b] = xg["cross"] - xv["cross"]
            bh[b] = xg["ratio_halves"] - xv["ratio_halves"]

        allt = np.arange(len(teams))
        pr = np.empty(a.nperm)
        ph = np.empty(a.nperm)
        for b in range(a.nperm):
            p = rng.permutation(allt)
            xg = stats_from_M(*Mg[g], p[:len(ti)], p[len(ti):])
            xv = stats_from_M(*Mv[g], p[:len(ti)], p[len(ti):])
            pr[b] = xg["ratio"] - xv["ratio"]
            ph[b] = xg["ratio_halves"] - xv["ratio_halves"]
        p_perm = float((np.abs(pr) >= abs(d_ratio)).mean())
        p_perm_h = float((np.abs(ph) >= abs(d_ratioh)).mean())
        # bootstrap two-sided p for the difference (fraction of resamples on the other side of 0)
        p_boot = float(2 * min((br <= 0).mean(), (br >= 0).mean()))

        res["groups"][g] = {
            "gen": {k: sg[k] for k in ("within_taught", "within_untaught", "within_pooled",
                                       "cross", "ratio", "ratio_halves", "within_same_team")},
            "v8": {k: sv[k] for k in ("within_taught", "within_untaught", "within_pooled",
                                      "cross", "ratio", "ratio_halves", "within_same_team")},
            "gen_norm_share": Kg["groups"][g]["norm_share_pooled"],
            "v8_norm_share": Kv["groups"][g]["norm_share_pooled"],
            "gen_n_params": Kg["groups"][g]["n_params"],
            "v8_n_params": Kv["groups"][g]["n_params"],
            "delta_ratio_gen_minus_v8": d_ratio,
            "delta_ratio_boot_ci95": [float(np.percentile(br, 2.5)),
                                      float(np.percentile(br, 97.5))],
            "delta_ratio_perm_p": p_perm,
            "delta_ratio_boot_p": min(1.0, p_boot),
            "delta_ratio_halves_gen_minus_v8": d_ratioh,
            "delta_ratio_halves_boot_ci95": [float(np.percentile(bh, 2.5)),
                                             float(np.percentile(bh, 97.5))],
            "delta_ratio_halves_perm_p": p_perm_h,
            "delta_cross_gen_minus_v8": d_cross,
            "delta_cross_boot_ci95": [float(np.percentile(bc, 2.5)),
                                      float(np.percentile(bc, 97.5))],
            "delta_relative_gap": d_gapn,
        }
        r = res["groups"][g]
        lo, hi = r["delta_ratio_boot_ci95"]
        hl, hh = r["delta_ratio_halves_boot_ci95"]
        print(f"[cmp] {g:16s} ratio gen {sg['ratio']:.4f} vs v8 {sv['ratio']:.4f}  "
              f"delta {d_ratio:+.4f} [{lo:+.4f},{hi:+.4f}]  perm_p {p_perm:.4f}  "
              f"normshare {r['gen_norm_share']:.4f}/{r['v8_norm_share']:.4f}", flush=True)
        print(f"{'':22s} halves gen {sg['ratio_halves']:.4f} vs v8 {sv['ratio_halves']:.4f}  "
              f"delta {d_ratioh:+.4f} [{hl:+.4f},{hh:+.4f}]  perm_p {p_perm_h:.4f}", flush=True)

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[cmp] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
