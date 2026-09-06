"""CORRECTION 3, demonstrated on v1's OWN artifact — the under-sampled pooled-L bootstrap.

`content_locality/v8_era_locality.py` built `own_all` with one cell per (teacher, taught team) —
10 + 3 + 10 = 23 — and resampled it with `bsT = rng.integers(0, nT, ...)` where
`nT = len(taught_union) = 22` (the DEDUPED union; one team is taught by two teachers). Index 22,
`defensive10`'s last taught team, could never be drawn, so `primary_A_era`'s CI was computed on 22
of its 23 clusters.

This recomputes that CI from v1's committed per-team KL vectors, both ways, so the size of the
defect is measured rather than asserted. Point estimates are unaffected by construction.

Run: python boot_bug_demo.py
"""
import json
import os
import sys

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
from boot import Boot   # noqa: E402

V1 = os.path.join(os.path.dirname(H), "content_locality", "v8_era_n9.json")


def main():
    d = json.load(open(V1))
    kl = {k: np.array(v) for k, v in d["per_team_kl_fwd"].items()}
    teams = d["_meta"]["teams"]
    idx = {t["sha10"]: t["i"] for t in teams}
    n_unt = sum(1 for t in teams if t["kind"] == "untaught")
    taught_of = d["_meta"]["taught_of"]

    own_all = np.array([kl[n][idx[s]] for n in taught_of for s in taught_of[n]])
    unt_all = np.array([np.mean([kl[n][i] for n in taught_of]) for i in range(n_unt)])
    n_union = len({s for n in taught_of for s in taught_of[n]})
    print(f"  own_all cells                {len(own_all)}   "
          f"(10 + 3 + 10 teacher-team pairs)")
    print(f"  len(taught_union) used by v1 {n_union}   (deduped — one team taught twice)")
    print(f"  => v1 drew indices in [0, {n_union}) over a {len(own_all)}-cell array; "
          f"cell {len(own_all) - 1} was unreachable")

    rec = d["primary_A_era"]
    print(f"\n  RECORDED (v1)              L {rec['L']:.4f}  "
          f"CI [{rec['L_ci95'][0]:.4f}, {rec['L_ci95'][1]:.4f}]   "
          f"taught CI [{rec['kl_taught_ci95'][0]:.4f}, {rec['kl_taught_ci95'][1]:.4f}]")

    # the bug, reproduced exactly: v1's draw order and its undersized taught matrix
    rng = np.random.default_rng(20260905)
    bsU = rng.integers(0, n_unt, (20000, n_unt))
    bsT = rng.integers(0, n_union, (20000, n_union))
    Lb = own_all[bsT].mean(axis=1) / unt_all[bsU].mean(axis=1)
    tb = own_all[bsT].mean(axis=1)
    print(f"  REPRODUCED under-sampled   L {own_all.mean() / unt_all.mean():.4f}  "
          f"CI [{np.percentile(Lb, 2.5):.4f}, {np.percentile(Lb, 97.5):.4f}]   "
          f"taught CI [{np.percentile(tb, 2.5):.4f}, {np.percentile(tb, 97.5):.4f}]")

    b = Boot()
    Lc = (own_all[b.idx(len(own_all))].mean(axis=1)
          / unt_all[b.idx(len(unt_all))].mean(axis=1))
    lo, hi = b.ci(own_all)
    print(f"  CORRECTLY SIZED            L {own_all.mean() / unt_all.mean():.4f}  "
          f"CI [{np.percentile(Lc, 2.5):.4f}, {np.percentile(Lc, 97.5):.4f}]   "
          f"taught CI [{lo:.4f}, {hi:.4f}]")
    print("\n  Point estimate identical; only the interval moves. The bug never touched the "
          "headline sibling-control R, whose arrays were correctly sized in v1.")


if __name__ == "__main__":
    main()
