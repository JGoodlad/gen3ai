"""DECIDE: PREREG §3's statistic and verdict, plus the S1 / S2 / S4 secondaries. Reads units.npz, s4.npz and the metas;
writes result.json. Pure numpy, deterministic (bootstrap seed 0, 2,000 resamples of GAMES per group)."""
from __future__ import annotations

import json

import numpy as np

import n2common as N

PANELS = ("home72", "F", "H", "P72")
B = 2000
MIN_CELL = 30
EPS = 0.005


def game_boot(games_by_group, rng):
    """Per group, B resamples of its games: a [B, n_games] index array into its unique games."""
    return {g: rng.integers(0, len(u), size=(B, len(u))) for g, u in games_by_group.items()}


def cell_sums(val, game, cell, ugames, ucells):
    gi = np.searchsorted(ugames, game)
    ci = np.searchsorted(ucells, cell)
    S = np.zeros((len(ugames), len(ucells)))
    Nn = np.zeros((len(ugames), len(ucells)))
    np.add.at(S, (gi, ci), val)
    np.add.at(Nn, (gi, ci), 1)
    return S, Nn


def standardised(val, grp, game, cell, idx_boot, ugames, keep_cells=None):
    """(point ratio pool/unf, [B] boot ratios, per-group point means, kept cells, weights, n_discarded)."""
    ucells = np.unique(cell)
    sums = {g: cell_sums(val[grp == g], game[grp == g], cell[grp == g], ugames[g], ucells) for g in (0, 1)}
    n_c = {g: sums[g][1].sum(0) for g in (0, 1)}
    keep = (n_c[0] >= MIN_CELL) & (n_c[1] >= MIN_CELL) if keep_cells is None else np.isin(ucells, keep_cells)
    w = (n_c[0] + n_c[1]) * keep
    w = w / w.sum()

    def gmean(S, Nn):
        with np.errstate(invalid="ignore", divide="ignore"):
            m = S[..., keep] / Nn[..., keep]
        return (m * w[keep]).sum(-1)
    pt = {g: float(gmean(sums[g][0].sum(0), sums[g][1].sum(0))) for g in (0, 1)}
    bm = {}
    for g in (0, 1):
        S, Nn = sums[g]
        bm[g] = gmean(S[idx_boot[g]].sum(1), Nn[idx_boot[g]].sum(1))
    br = bm[0] / bm[1]
    ok = np.isfinite(br)
    return pt[0] / pt[1], br[ok], pt, ucells[keep].tolist(), w[keep].tolist(), int((~ok).sum())


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def verdict(R, lo, hi, guard_lo):
    near = lambda x, t: abs(x - t) < EPS  # noqa: E731
    keys = R >= 1.5 and not near(R, 1.5) and lo > 1.0 and not near(lo, 1.0)
    comp = hi < 1.5 and not near(hi, 1.5)
    if keys:
        return "KEYS-ON-SETS", None
    if comp:
        if guard_lo is not None and guard_lo > 1.5 and not near(guard_lo, 1.5):
            return "INCONCLUSIVE", "the team-level guard: S4's identity-ablation ratio CI lower bound > 1.5"
        return "COMPOSITIONAL", None
    return "INCONCLUSIVE", "neither rule met"


def main() -> None:
    u = dict(np.load(N.OUT / "units.npz"))
    s4 = dict(np.load(N.OUT / "s4.npz"))
    pm = json.load(open(N.OUT / "probe_meta.json"))
    s4m = json.load(open(N.OUT / "s4_meta.json"))
    st = pm["stats"]
    grp, game = u["group"], u["game"] * 10 + u["panel"]          # a game id unique across panels
    cell = u["n_moves"] * 10 + u["item"]
    ugames = {g: np.unique(game[grp == g]) for g in (0, 1)}
    rng = np.random.default_rng(0)
    idx = game_boot(ugames, rng)
    res = {"n_units": {"pool": int((grp == 0).sum()), "unfamiliar": int((grp == 1).sum())},
           "n_games": {"pool": len(ugames[0]), "unfamiliar": len(ugames[1])},
           "n_rows": {"pool": int(len(np.unique(u["row"][grp == 0]))), "unfamiliar": int(len(np.unique(u["row"][grp == 1])))}}
    fails = sum(v for k, v in st.items() if k.endswith("splice_guard_fail"))
    tg = sum(v for k, v in st.items() if k.endswith("_targets"))
    res["splice_guard"] = {"fail": fails, "pass": tg, "fail_frac": fails / max(fails + tg, 1)}
    void = []
    if res["splice_guard"]["fail_frac"] > 0.05:
        void.append("splice guard > 5 %")
    if min(res["n_units"].values()) < 300:
        void.append("< 300 units in a group")
    if s4m["patched_identity_max_abs_dprob"] > 1e-5:
        void.append("S4 patched-encoder identity > 1e-5")

    R, br, pt, cells, w, nd = standardised(u["kl"], grp, game, cell, idx, ugames)
    res["primary"] = {"R_kl_standardised": R, "ci": ci(br), "mean_kl_pool": pt[0], "mean_kl_unfamiliar": pt[1],
                      "cells_kept_(moves*10+item)": cells, "weights": [round(x, 4) for x in w],
                      "boot_discarded": nd}
    cell_tab = {}
    for c in np.unique(cell):
        a, b = (grp == 0) & (cell == c), (grp == 1) & (cell == c)
        cell_tab[f"moves{c // 10}_item{c % 10}"] = {"n_pool": int(a.sum()), "n_unf": int(b.sum()),
                                                   "kl_pool": float(u["kl"][a].mean()) if a.any() else None,
                                                   "kl_unf": float(u["kl"][b].mean()) if b.any() else None}
    res["cells"] = cell_tab
    # S1
    one = np.zeros_like(cell)
    Rr, brr, ptr, _, _, _ = standardised(u["kl"], grp, game, one, idx, ugames)
    res["S1_raw_kl_ratio"] = {"R": Rr, "ci": ci(brr), "pool": ptr[0], "unfamiliar": ptr[1]}
    for name, val in (("flip", u["flip"].astype(float)), ("abs_dv", np.abs(u["dv"]))):
        r_, b_, p_, _, _, _ = standardised(val, grp, game, cell, idx, ugames, keep_cells=cells)
        res[f"S1_{name}_ratio"] = {"R": r_, "ci": ci(b_), "pool": p_[0], "unfamiliar": p_[1]}
    shared = np.intersect1d(np.unique(u["species"][grp == 0]), np.unique(u["species"][grp == 1]))
    sm = np.isin(u["species"], shared)
    ug_s = {g: np.unique(game[sm & (grp == g)]) for g in (0, 1)}
    idx_s = game_boot(ug_s, np.random.default_rng(0))
    r_, b_, p_, c_, _, _ = standardised(u["kl"][sm], grp[sm], game[sm], cell[sm], idx_s, ug_s)
    res["S1_shared_species"] = {"n_species": int(len(shared)), "n_units": {"pool": int((sm & (grp == 0)).sum()),
                                "unfamiliar": int((sm & (grp == 1)).sum())}, "R": r_, "ci": ci(b_),
                                "pool": p_[0], "unfamiliar": p_[1], "cells": c_}
    per_panel = {}
    for pi, p in enumerate(PANELS):
        mk = u["panel"] == pi
        per_panel[p] = {"n": int(mk.sum()), "kl_mean": float(u["kl"][mk].mean()), "flip": float(u["flip"][mk].mean()),
                        "abs_dv": float(np.abs(u["dv"][mk]).mean())}
        if p != "home72":                       # the pool group against this panel alone, standardised
            sel = (grp == 0) | mk
            ug_p = {g: np.unique(game[sel & (grp == g)]) for g in (0, 1)}
            idx_p = game_boot(ug_p, np.random.default_rng(0))
            r_, b_, _, _, _, _ = standardised(u["kl"][sel], grp[sel], game[sel], cell[sel], idx_p, ug_p)
            per_panel[p]["R_pool_over_this_panel"] = r_
            per_panel[p]["ci"] = ci(b_)
    res["S1_per_panel"] = per_panel
    # S2
    r_, b_, p_, _, _, _ = standardised(np.abs(u["dswitch"]), grp, game, cell, idx, ugames, keep_cells=cells)
    res["S2_abs_dswitch_to_target_ratio"] = {"R": r_, "ci": ci(b_), "pool": p_[0], "unfamiliar": p_[1]}
    # S4 (rows; unstandardised; the same game bootstrap scheme)
    g4, game4 = s4["group"], s4["game"] * 10 + s4["panel"]
    ug4 = {g: np.unique(game4[g4 == g]) for g in (0, 1)}
    idx4 = game_boot(ug4, np.random.default_rng(0))
    z4 = np.zeros_like(g4)
    s4res = {"patched_identity_max_abs_dprob": s4m["patched_identity_max_abs_dprob"], "n_rows": s4m["n_rows"]}
    for arm in ("identity", "mechanics"):
        for met in ("kl", "flip"):
            r_, b_, p_, _, _, _ = standardised(s4[f"{arm}_{met}"].astype(float), g4, game4, z4, idx4, ug4)
            s4res[f"{arm}_{met}"] = {"R": r_, "ci": ci(b_), "pool": p_[0], "unfamiliar": p_[1]}
    res["S4"] = s4res
    res["S3"] = json.load(open(N.OUT / "s3.json"))
    lo, hi = res["primary"]["ci"]
    v, why = verdict(R, lo, hi, s4res["identity_kl"]["ci"][0])
    if void:
        v, why = "VOID", "; ".join(void)
    res["verdict"] = {"verdict": v, "reason": why}
    json.dump(res, open(N.OUT / "result.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
