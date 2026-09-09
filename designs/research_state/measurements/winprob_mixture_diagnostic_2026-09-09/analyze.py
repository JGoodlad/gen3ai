"""THE MIXTURE TEST — is arm A's win-prob critic a single marginal V that fits no opponent?

Hypothesis (owner, 2026-09-08). V(s) is trained against a MIXTURE of opponents
(9 scripted bots + 5 pool sentinels). If the head never learned "who am I playing",
it emits the mixture's marginal win probability at every state: too LOW against a
weak opponent (where the true rate is ~1.0) and too HIGH against a strong one.

Prediction, stated before the reading (sign matters):
    bias := V - outcome.  Against WEAK opponents outcome ~ 1.0 > V  => bias NEGATIVE.
    Against STRONG opponents outcome ~ 0.5 < V                      => bias POSITIVE.
    So the mixture hypothesis predicts a POSITIVE slope of bias on opponent strength.
    A separating head predicts a slope indistinguishable from zero.

Second, sharper prediction. For ANY calibrated critic, E[V | opponent] = E[y | opponent]
exactly, so the BETWEEN-OPPONENT spread of V must equal the between-opponent spread of
the outcome. The mixture defect is precisely a shortfall of the first against the second,
and that shortfall is scale-free and needs no strength axis at all.

Every interval is a BATTLE-CLUSTERED bootstrap. Every statistic is Horvitz-Thompson
reweighted by the manifest's capture rates (the trace quota prefers losses). Every
DETECTED claim is on the DELTA's own interval, never on two overlapping bars.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

BOTS = ["random", "heuristic", "heuristic2", "staller", "staller_v2",
        "aggressive", "aggressive_v2", "setup_sweep", "setup_sweep_v2"]
EARLY, MID = 10, 24          # early <= 10 | mid 11-24 | late >= 25 (the registered buckets)
T_EARLY = 3                  # the "before the board has said much" window


# ──────────────────────────────────────────────────────────────────────────────
# Per-battle roll-up: every statistic below is a ratio of sums over BATTLES, so a
# cluster bootstrap is a gather over this table and nothing has to be recomputed
# from the 29k states.
# ──────────────────────────────────────────────────────────────────────────────
def rollup(arr, weighted=True):
    key = np.stack([arr["cycle"].astype("i8"),
                    np.unique(arr["opponent"], return_inverse=True)[1],
                    np.unique(arr["battle"], return_inverse=True)[1]], axis=1)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    n_b = first.size
    turn, V = arr["turn"], arr["V"]
    masks = {"all": np.ones(arr.size, bool),
             "early": turn <= EARLY, "mid": (turn > EARLY) & (turn <= MID), "late": turn > MID,
             "t1_3": turn <= T_EARLY, "t1": turn == 1}
    b = {"cycle": arr["cycle"][first], "opponent": arr["opponent"][first],
         "opp_class": arr["opp_class"][first], "battle": arr["battle"][first],
         "y": arr["y"][first], "team": arr["team"][first],
         "strength": arr["strength"][first], "true_wr": arr["true_wr"][first],
         "w": arr["w"][first] if weighted else np.ones(n_b)}
    for name, m in masks.items():
        b[f"n_{name}"] = np.bincount(inv[m], minlength=n_b).astype(float)
        b[f"sV_{name}"] = np.bincount(inv[m], weights=V[m], minlength=n_b)
        b[f"sV2_{name}"] = np.bincount(inv[m], weights=V[m] ** 2, minlength=n_b)
    return b


def cell_index(b):
    """(cycle, opponent) cells, plus the battle slots of each, sorted by cell."""
    cyc = np.unique(b["cycle"])
    opps = list(BOTS) + sorted({o for o in np.unique(b["opponent"]) if o not in BOTS})
    oidx = {o: i for i, o in enumerate(opps)}
    cid = np.array([np.searchsorted(cyc, c) * len(opps) + oidx[o]
                    for c, o in zip(b["cycle"], b["opponent"])])
    return cyc, opps, oidx, cid


# ──────────────────────────────────────────────────────────────────────────────
# Statistics, all as functions of a battle-index selection `sel` (a bootstrap draw)
# ──────────────────────────────────────────────────────────────────────────────
def cell_stats(b, sel, cid_of_sel, n_cells, bucket="all", wr_draw=None):
    """Per-cell: IPW battle-level mean V, IPW battle-level outcome, state-level means.

    🚨 ``cid_of_sel`` is the cell code of each SELECTED battle, aligned 1:1 with ``sel`` --
    never a per-battle table to be indexed here. A cluster bootstrap resamples battles WITHIN
    a cell, so a draw has exactly as many entries as the original and a size test cannot tell
    the two apart; an earlier revision used one and silently paired resampled battles with the
    ORIGINAL cell codes, scrambling battles across opponents and making every interval wrong.
    The caller now always supplies the aligned codes."""
    w = b["w"][sel]
    c = cid_of_sel
    ns = b[f"n_{bucket}"][sel]
    sV = b[f"sV_{bucket}"][sel]
    sV2 = b[f"sV2_{bucket}"][sel]
    y = b["y"][sel]
    has = ns > 0
    # battle-level (each battle one unit; V̄_b is its own mean over the bucket's states)
    Wb = np.bincount(c[has], weights=w[has], minlength=n_cells)
    Vb = np.bincount(c[has], weights=(w * sV / np.where(ns > 0, ns, 1))[has], minlength=n_cells)
    Yb = np.bincount(c[has], weights=(w * y)[has], minlength=n_cells)
    # state-level (population of traced states; each state carries its battle's weight)
    Ws = np.bincount(c, weights=w * ns, minlength=n_cells)
    Vs = np.bincount(c, weights=w * sV, minlength=n_cells)
    V2s = np.bincount(c, weights=w * sV2, minlength=n_cells)
    Ys = np.bincount(c, weights=w * ns * y, minlength=n_cells)
    ok = Wb > 0
    # squared standard error of the cell's IPW battle-level mean V, for the noise correction
    # below: a raw between-cell variance contains the sampling noise of each cell mean, and
    # subtracting it is the difference between "the opponents differ" and "the cells are small".
    with np.errstate(invalid="ignore", divide="ignore"):
        mV_tmp = np.where(ok, Vb / np.where(ok, Wb, 1), np.nan)
        vb = np.where(ns > 0, sV / np.where(ns > 0, ns, 1), np.nan)
        dev = np.where(has, (w * (vb - mV_tmp[c])) ** 2, 0.0)
        S2 = np.bincount(c[has], weights=dev[has], minlength=n_cells)
        nb_ = np.bincount(c[has], minlength=n_cells).astype(float)
        seV2 = np.where((Wb > 0) & (nb_ > 1), S2 / np.where(Wb > 0, Wb, 1) ** 2
                        * np.where(nb_ > 1, nb_ / (nb_ - 1), 1.0), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {"n_battles": np.bincount(c[has], minlength=n_cells).astype(float),
               "n_states": np.bincount(c, weights=ns, minlength=n_cells),
               "W_b": Wb, "W_s": Ws,
               "meanV_b": np.where(ok, Vb / np.where(ok, Wb, 1), np.nan),
               "out_b": np.where(ok, Yb / np.where(ok, Wb, 1), np.nan),
               "meanV_s": np.where(Ws > 0, Vs / np.where(Ws > 0, Ws, 1), np.nan),
               "meanV2_s": np.where(Ws > 0, V2s / np.where(Ws > 0, Ws, 1), np.nan),
               "out_s": np.where(Ws > 0, Ys / np.where(Ws > 0, Ws, 1), np.nan),
               "seV2_b": seV2}
    out["true_wr"] = wr_draw
    # PRIMARY: both sides battle-level, so no length bias; the outcome is the cycle's TRUE
    # win rate (100 games), not the loss-enriched traced rate.
    out["bias"] = (out["meanV_b"] - wr_draw) if wr_draw is not None else np.full(n_cells, np.nan)
    out["bias_state"] = out["meanV_s"] - out["out_s"]  # robustness: the state population
    return out


def ols_slope(bias, strength, cyc_of_cell, keep):
    """OLS of bias on strength/100 with CYCLE fixed effects. Returns slope per 100 Elo."""
    m = keep & np.isfinite(bias) & np.isfinite(strength)
    if m.sum() < 3:
        return np.nan
    x = strength[m] / 100.0
    cy = cyc_of_cell[m]
    lv = np.unique(cy)
    X = np.zeros((m.sum(), 1 + lv.size))
    X[:, 0] = x
    for j, c in enumerate(lv):
        X[:, 1 + j] = (cy == c)
    try:
        beta, *_ = np.linalg.lstsq(X, bias[m], rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    return float(beta[0])


def var_decomp(st, keep):
    """Between/within-opponent variance of the OUTCOME and of V, over the state population.

    Within-opponent variance of y is p(1-p) with p the cell's state-level win rate;
    within-opponent variance of V is E[V^2]-E[V]^2 in the cell. Both use the same IPW
    state weights, so the two decompositions are strictly comparable.
    """
    m = keep & (st["W_s"] > 0)
    W = st["W_s"][m]
    if W.sum() <= 0:
        return {k: np.nan for k in ("btw_y", "wth_y", "eta_y", "btw_V", "wth_V", "eta_V",
                                    "sd_btw_y", "sd_btw_V", "sd_ratio", "sd_delta")}
    p = W / W.sum()
    my, mV = st["out_s"][m], st["meanV_s"][m]
    gy, gV = float(p @ my), float(p @ mV)
    btw_y = float(p @ (my - gy) ** 2)
    btw_V = float(p @ (mV - gV) ** 2)
    wth_y = float(p @ (my * (1 - my)))
    wth_V = float(p @ np.clip(st["meanV2_s"][m] - mV ** 2, 0, None))
    sy, sV = np.sqrt(btw_y), np.sqrt(btw_V)
    return {"btw_y": btw_y, "wth_y": wth_y, "eta_y": btw_y / (btw_y + wth_y),
            "btw_V": btw_V, "wth_V": wth_V, "eta_V": btw_V / (btw_V + wth_V),
            "sd_btw_y": sy, "sd_btw_V": sV,
            "sd_ratio": sV / sy if sy > 0 else np.nan, "sd_delta": sV - sy}


def spread(st, keep, bucket_stats=None):
    """UNWEIGHTED spread across opponent cells: SD of the per-opponent mean."""
    s = bucket_stats if bucket_stats is not None else st
    m = keep & (s["W_s"] > 0)
    if m.sum() < 2:
        return {"sd_V": np.nan, "sd_y": np.nan, "delta": np.nan, "ratio": np.nan}
    sV = float(np.std(s["meanV_s"][m], ddof=1))
    sy = float(np.std(s["out_s"][m], ddof=1))
    return {"sd_V": sV, "sd_y": sy, "delta": sV - sy,
            "ratio": sV / sy if sy > 0 else np.nan}


def spread_corrected(st, keep, true_wr, n_games=100.0):
    """BETWEEN-OPPONENT spread of V vs of the OUTCOME, each corrected for its own sampling noise.

    For ANY calibrated critic E[V | opponent] == E[y | opponent], so these two spreads must be
    EQUAL. The mixture defect is exactly a shortfall of the first against the second, and it
    needs no strength axis at all.

    * the outcome's per-opponent mean is the cycle's TRUE win rate (100 games, so the noise it
      carries is p(1-p)/100 exactly, not an estimate);
    * V's per-opponent mean is the IPW battle-level mean, whose noise is its own squared SE.

    Cells are weighted EQUALLY: the question is how far apart the opponents are, not how much
    traffic each got.
    """
    m = keep & np.isfinite(st["meanV_b"]) & np.isfinite(true_wr) & np.isfinite(st["seV2_b"])
    if m.sum() < 3:
        return {k: np.nan for k in ("sd_V", "sd_y", "ratio", "delta",
                                    "sd_V_raw", "sd_y_raw", "noise_V", "noise_y")}
    V, Y = st["meanV_b"][m], true_wr[m]
    nV = float(np.mean(st["seV2_b"][m]))
    nY = float(np.mean(Y * (1 - Y) / n_games))
    vV = float(np.var(V, ddof=1))
    vY = float(np.var(Y, ddof=1))
    sV = np.sqrt(max(vV - nV, 0.0))
    sY = np.sqrt(max(vY - nY, 0.0))
    return {"sd_V": sV, "sd_y": sY, "ratio": (sV / sY) if sY > 0 else np.nan,
            "delta": sV - sY, "sd_V_raw": np.sqrt(vV), "sd_y_raw": np.sqrt(vY),
            "noise_V": np.sqrt(nV), "noise_y": np.sqrt(nY)}


def resid_between(b, sel, group_codes, n_groups):
    """Between-group variance of the IPW mean residual (V - y), for group = opponent or team."""
    w, ns, sV, y = b["w"][sel], b["n_all"][sel], b["sV_all"][sel], b["y"][sel]
    g = group_codes[sel]
    W = np.bincount(g, weights=w * ns, minlength=n_groups)
    R = np.bincount(g, weights=w * (sV - ns * y), minlength=n_groups)
    m = W > 0
    if m.sum() < 2:
        return np.nan
    p = W[m] / W[m].sum()
    r = R[m] / W[m]
    return float(p @ (r - float(p @ r)) ** 2)


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap driver
# ──────────────────────────────────────────────────────────────────────────────
def ci(draws, lo=2.5, hi=97.5):
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if d.size < 20:
        return (np.nan, np.nan)
    return (float(np.percentile(d, lo)), float(np.percentile(d, hi)))


def run(states_path, out_path, n_boot=4000, n_boot_head=10000, seed=20260909):
    arr = np.load(states_path, allow_pickle=False)
    b = rollup(arr)
    cyc, opps, oidx, cid = cell_index(b)
    n_cells = len(cyc) * len(opps)
    cyc_of_cell = np.repeat(cyc, len(opps))
    opp_of_cell = np.tile(np.array(opps), len(cyc))
    is_bot_cell = np.isin(opp_of_cell, BOTS)

    # per-cell fixed facts (strength, true win rate, games played)
    strength = np.full(n_cells, np.nan)
    true_wr = np.full(n_cells, np.nan)
    for c, o, s, t in zip(b["cycle"], b["opponent"], b["strength"], b["true_wr"]):
        k = np.searchsorted(cyc, c) * len(opps) + oidx[o]
        strength[k], true_wr[k] = s, t
    n_games = 100.0   # every cell: manifest battles_played == 100 (asserted below)

    # battle slots, sorted by cell, for the vectorised cluster bootstrap
    order = np.argsort(cid, kind="stable")
    cid_s = cid[order]
    starts = np.searchsorted(cid_s, np.arange(n_cells), "left")
    sizes = np.searchsorted(cid_s, np.arange(n_cells), "right") - starts
    slots_of_cell = [order[starts[k]:starts[k] + sizes[k]] for k in range(n_cells)]
    all_slots = np.concatenate([s for s in slots_of_cell if s.size])
    off_per_slot = np.concatenate([np.full(s.size, i) for i, s in enumerate(slots_of_cell) if s.size])
    size_per_slot = sizes[off_per_slot].astype(float)
    base_per_slot = np.concatenate([np.arange(s.size) * 0 for s in slots_of_cell if s.size])
    cell_start_in_all = {}
    p = 0
    for k, s in enumerate(slots_of_cell):
        if s.size:
            cell_start_in_all[k] = p
            p += s.size
    start_per_slot = np.array([cell_start_in_all[k] for k in off_per_slot])
    del base_per_slot

    rng = np.random.default_rng(seed)

    def draw_battles():
        """Cluster bootstrap: resample battles WITHIN each (cycle, opponent) cell.
        Returns (battle indices, the cell code of each) -- the codes come from the SLOT,
        which the within-cell resample leaves fixed."""
        j = start_per_slot + (rng.random(all_slots.size) * size_per_slot).astype(int)
        return all_slots[j], off_per_slot

    def draw_wr():
        return rng.binomial(n_games, np.nan_to_num(true_wr, nan=0.5)) / n_games

    # ── point estimates ──────────────────────────────────────────────────────
    sel0 = np.arange(b["y"].size)
    cid0 = cid
    st0 = {bk: cell_stats(b, sel0, cid0, n_cells, bk, true_wr)
           for bk in ("all", "early", "mid", "late", "t1_3", "t1")}
    b_unw = rollup(arr, weighted=False)
    st0_unw = cell_stats(b_unw, sel0, cid0, n_cells, "all", true_wr)

    keep_bot = is_bot_cell & (st0["all"]["n_battles"] > 0)
    keep_all = st0["all"]["n_battles"] > 0

    pt = {
        "slope_bot": ols_slope(st0["all"]["bias"], strength, cyc_of_cell, keep_bot),
        "slope_all": ols_slope(st0["all"]["bias"], strength, cyc_of_cell, keep_all),
        "slope_bot_raw": ols_slope(st0_unw["bias"], strength, cyc_of_cell, keep_bot),
        "slope_all_raw": ols_slope(st0_unw["bias"], strength, cyc_of_cell, keep_all),
        "slope_bot_state": ols_slope(st0["all"]["bias_state"], strength, cyc_of_cell, keep_bot),
    }
    for bk in ("early", "mid", "late"):
        pt[f"slope_bot_{bk}"] = ols_slope(st0[bk]["bias"], strength, cyc_of_cell, keep_bot)
        pt[f"slope_all_{bk}"] = ols_slope(st0[bk]["bias"], strength, cyc_of_cell, keep_all)
    per_cycle = {}
    for ci_, c in enumerate(cyc):
        mc = cyc_of_cell == c
        per_cycle[int(c)] = {
            "slope_bot": ols_slope(st0["all"]["bias"], strength, cyc_of_cell, keep_bot & mc),
            "slope_all": ols_slope(st0["all"]["bias"], strength, cyc_of_cell, keep_all & mc),
            "var_all": var_decomp(st0["all"], keep_all & mc),
            "spread_t1_3_all": spread(None, keep_all & mc, st0["t1_3"]),
            "spread_all": spread(st0["all"], keep_all & mc),
            "sc_all": spread_corrected(st0["all"], keep_all & mc, true_wr),
            "sc_t1_3": spread_corrected(st0["t1_3"], keep_all & mc, true_wr),
            "sc_all_bot": spread_corrected(st0["all"], keep_bot & mc, true_wr),
            "sc_t1_3_bot": spread_corrected(st0["t1_3"], keep_bot & mc, true_wr),
        }
    vd_all = var_decomp(st0["all"], keep_all)
    vd_bot = var_decomp(st0["all"], keep_bot)
    sp_all = spread(st0["all"], keep_all)
    sp_bot = spread(st0["all"], keep_bot)
    sp_t13_all = spread(None, keep_all, st0["t1_3"])
    sp_t13_bot = spread(None, keep_bot, st0["t1_3"])
    sp_t1_all = spread(None, keep_all, st0["t1"])
    SC = {}
    for bk in ("all", "t1_3", "t1", "early", "mid", "late"):
        for tag, kk in (("all", keep_all), ("bot", keep_bot)):
            SC[f"{bk}_{tag}"] = spread_corrected(st0[bk], kk, true_wr)

    # team vs opponent, on the residual V - y
    team_codes = np.unique(b["team"], return_inverse=True)[1]
    opp_codes = np.unique(b["opponent"], return_inverse=True)[1]
    n_team, n_opp = team_codes.max() + 1, opp_codes.max() + 1
    pt["btw_opp_resid"] = resid_between(b, sel0, opp_codes, n_opp)
    pt["btw_team_resid"] = resid_between(b, sel0, team_codes, n_team)
    pt["n_teams"] = int(n_team)
    # 🚨 A between-group variance is UPWARD BIASED by small groups: 216 teams over 951 battles
    # is ~4.4 battles each, against 14 opponents at ~68 each. Comparing the two raw numbers
    # would convict the team on group size alone. So each is measured as an EXCESS over its own
    # LABEL-PERMUTATION null (labels shuffled across battles, group sizes preserved), which is
    # exactly the small-group inflation and nothing else.
    rng_null = np.random.default_rng(seed + 1)
    null_opp, null_team = [], []
    for _ in range(400):
        null_opp.append(resid_between(b, sel0, rng_null.permutation(opp_codes), n_opp))
        null_team.append(resid_between(b, sel0, rng_null.permutation(team_codes), n_team))
    pt["null_opp_resid"] = float(np.nanmean(null_opp))
    pt["null_team_resid"] = float(np.nanmean(null_team))
    pt["excess_opp_resid"] = pt["btw_opp_resid"] - pt["null_opp_resid"]
    pt["excess_team_resid"] = pt["btw_team_resid"] - pt["null_team_resid"]
    pt["excess_gap_opp_minus_team"] = pt["excess_opp_resid"] - pt["excess_team_resid"]

    # ── bootstrap ────────────────────────────────────────────────────────────
    D = {k: [] for k in list(pt) + ["eta_y_all", "eta_V_all", "sd_btw_y_all", "sd_btw_V_all",
                                    "sd_delta_all", "sd_ratio_all", "eta_y_bot", "eta_V_bot",
                                    "sd_btw_y_bot", "sd_btw_V_bot", "sd_delta_bot",
                                    "sd_ratio_bot", "sp_delta_all", "sp_delta_bot",
                                    "sp_t13_delta_all", "sp_t13_delta_bot", "sp_t13_sd_V_all",
                                    "sp_t13_sd_y_all", "sp_t1_delta_all",
                                    "resid_gap_team_minus_opp", "excess_gap_opp_minus_team",
                                    "excess_opp_only", "excess_team_only"]
     + [f"sc_{bk}_{tag}_{f}" for bk in ("all", "t1_3", "t1", "early", "mid", "late")
        for tag in ("all", "bot") for f in ("sd_V", "sd_y", "ratio", "delta")]}
    cell_bias_draws = np.full((n_boot, n_cells), np.nan)
    for i in range(n_boot):
        sel, cid_sel = draw_battles()
        wr = draw_wr()
        s_all = cell_stats(b, sel, cid_sel, n_cells, "all", wr)
        cell_bias_draws[i] = s_all["bias"]
        kb = is_bot_cell & (s_all["n_battles"] > 0)
        ka = s_all["n_battles"] > 0
        D["slope_bot"].append(ols_slope(s_all["bias"], strength, cyc_of_cell, kb))
        D["slope_all"].append(ols_slope(s_all["bias"], strength, cyc_of_cell, ka))
        D["slope_bot_state"].append(ols_slope(s_all["bias_state"], strength, cyc_of_cell, kb))
        s_unw = cell_stats(b_unw, sel, cid_sel, n_cells, "all", wr)
        D["slope_bot_raw"].append(ols_slope(s_unw["bias"], strength, cyc_of_cell, kb))
        D["slope_all_raw"].append(ols_slope(s_unw["bias"], strength, cyc_of_cell, ka))
        for bk in ("early", "mid", "late"):
            s_bk = cell_stats(b, sel, cid_sel, n_cells, bk, wr)
            D[f"slope_bot_{bk}"].append(ols_slope(s_bk["bias"], strength, cyc_of_cell, kb))
            D[f"slope_all_{bk}"].append(ols_slope(s_bk["bias"], strength, cyc_of_cell, ka))
        for tag, kk in (("all", ka), ("bot", kb)):
            v = var_decomp(s_all, kk)
            for f in ("eta_y", "eta_V", "sd_btw_y", "sd_btw_V", "sd_delta", "sd_ratio"):
                D[f"{f}_{tag}"].append(v[f])
            D[f"sp_delta_{tag}"].append(spread(s_all, kk)["delta"])
        s_t13 = cell_stats(b, sel, cid_sel, n_cells, "t1_3", wr)
        for tag, kk in (("all", ka), ("bot", kb)):
            sp = spread(None, kk, s_t13)
            D[f"sp_t13_delta_{tag}"].append(sp["delta"])
            if tag == "all":
                D["sp_t13_sd_V_all"].append(sp["sd_V"])
                D["sp_t13_sd_y_all"].append(sp["sd_y"])
        s_t1 = cell_stats(b, sel, cid_sel, n_cells, "t1", wr)
        D["sp_t1_delta_all"].append(spread(None, ka, s_t1)["delta"])
        for bk in ("all", "t1_3", "t1", "early", "mid", "late"):
            s_bk2 = s_all if bk == "all" else cell_stats(b, sel, cid_sel, n_cells, bk, wr)
            for tag, kk in (("all", ka), ("bot", kb)):
                v = spread_corrected(s_bk2, kk, wr)
                for f in ("sd_V", "sd_y", "ratio", "delta"):
                    D[f"sc_{bk}_{tag}_{f}"].append(v[f])
        bo = resid_between(b, sel, opp_codes, n_opp)
        bt = resid_between(b, sel, team_codes, n_team)
        D["btw_opp_resid"].append(bo)
        D["btw_team_resid"].append(bt)
        D["resid_gap_team_minus_opp"].append(bt - bo)
        D["excess_opp_only"].append(bo - pt["null_opp_resid"])
        D["excess_team_only"].append(bt - pt["null_team_resid"])
        D["excess_gap_opp_minus_team"].append(
            (bo - pt["null_opp_resid"]) - (bt - pt["null_team_resid"]))

    # ── headline slope, with OPPONENTS resampled too (the conservative variant) ──
    def draw_with_opponents(pool_idx):
        """The conservative variant: resample the OPPONENT SET with replacement, then resample
        battles within each drawn cell. A duplicated opponent becomes two independent cells at
        the same strength, which is what makes the interval price "these 9 opponents happened to
        be the 9 we own" as well as "these battles happened to be the ones traced".

        Returns (battle indices, the cell code of each, the opponent index of each NEW cell).
        🚨 The opponent index per new cell is returned EXPLICITLY. An earlier revision recovered
        it from the cell codes and shadowed the name doing so, which silently paired each drawn
        cell with the strength of a DIFFERENT opponent."""
        picks = rng.choice(pool_idx, size=pool_idx.size, replace=True)
        sel, cid2 = [], []
        opp_of_new = np.full(len(cyc) * len(opps), -1, dtype=int)
        for ci_ in range(len(cyc)):
            for rep, o in enumerate(picks):
                k_src = ci_ * len(opps) + int(o)
                s_ = slots_of_cell[k_src]
                if s_.size == 0:
                    continue
                take = s_[(rng.random(s_.size) * s_.size).astype(int)]
                k_new = ci_ * len(opps) + rep
                sel.append(take)
                cid2.append(np.full(take.size, k_new))
                opp_of_new[k_new] = int(o)
        if not sel:
            return None, None, None
        return np.concatenate(sel), np.concatenate(cid2), opp_of_new

    bot_pool = np.array([oidx[o] for o in BOTS])
    all_pool = np.arange(len(opps))
    head = {}
    for tag, pool in (("bot", bot_pool), ("all", all_pool)):
        dr = []
        for _ in range(n_boot_head):
            sel, cid2, opp_of_new = draw_with_opponents(pool)
            if sel is None:
                continue
            n_c2 = len(cyc) * len(opps)
            st_ = cell_stats(b, sel, cid2, n_c2, "all", None)  # cid2 is aligned to sel
            s2 = np.full(n_c2, np.nan)
            t2 = np.full(n_c2, np.nan)
            cy2 = np.repeat(cyc, len(opps))
            live = opp_of_new >= 0
            src = (np.arange(n_c2) // len(opps)) * len(opps) + np.where(live, opp_of_new, 0)
            s2[live] = strength[src[live]]
            t2[live] = rng.binomial(n_games, true_wr[src[live]]) / n_games
            bias2 = st_["meanV_b"] - t2
            dr.append(ols_slope(bias2, s2, cy2, np.isfinite(bias2) & live))
        head[f"slope_{tag}_oppboot"] = (float(np.nanmean(dr)), ci(dr))

    res = {
        "n_states": int(arr.size), "n_battles": int(b["y"].size),
        "n_cells": int(keep_all.sum()), "n_bot_cells": int(keep_bot.sum()),
        "n_boot": n_boot, "n_boot_head": n_boot_head, "seed": seed,
        "point": {k: (None if not np.isfinite(v) else float(v))
                  for k, v in pt.items() if isinstance(v, float)},
        "point_int": {"n_teams": pt["n_teams"]},
        "ci": {k: ci(v) for k, v in D.items() if v},
        "head": head,
        "spread_corrected": SC,
        "per_cycle": per_cycle,
        "var_decomp": {"all": vd_all, "bot": vd_bot},
        "spread": {"all": sp_all, "bot": sp_bot,
                   "t1_3_all": sp_t13_all, "t1_3_bot": sp_t13_bot, "t1_all": sp_t1_all},
        "cells": [],
    }
    for k in range(n_cells):
        if not keep_all[k]:
            continue
        lo, hi = ci(cell_bias_draws[:, k])
        res["cells"].append({
            "cycle": int(cyc_of_cell[k]), "opponent": str(opp_of_cell[k]),
            "class": "bot" if is_bot_cell[k] else "sentinel",
            "strength": float(strength[k]), "true_wr": float(true_wr[k]),
            "n_battles": int(st0["all"]["n_battles"][k]),
            "n_states": float(st0["all"]["n_states"][k]),
            "meanV_b": float(st0["all"]["meanV_b"][k]),
            "meanV_s": float(st0["all"]["meanV_s"][k]),
            "out_traced_raw": float(st0_unw["out_b"][k]),
            "out_reweighted": float(st0["all"]["out_b"][k]),
            "bias": float(st0["all"]["bias"][k]), "bias_ci": [lo, hi],
            "meanV_t1_3": (float(st0["t1_3"]["meanV_s"][k])
                           if st0["t1_3"]["W_s"][k] > 0 else None),
            "n_states_t1_3": float(st0["t1_3"]["n_states"][k]),
        })
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1, default=lambda o: None if o != o else float(o))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--boot-head", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260909)
    a = ap.parse_args()
    r = run(a.states, a.out, a.boot, a.boot_head, a.seed)
    print(json.dumps({k: v for k, v in r.items() if k != "cells"}, indent=1, default=str)[:4000])
