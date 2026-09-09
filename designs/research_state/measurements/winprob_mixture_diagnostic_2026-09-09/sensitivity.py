"""Sensitivities the headline slope has to survive.

(1) LEAVE-ONE-OPPONENT-OUT. `random` sits at Elo 1000 with the next bot at 1511.7, so it
    carries most of the bot-only leverage. If the slope is an artefact of one point, dropping
    it kills the slope.
(2) THE GREEDY-VS-STOCHASTIC SENTINEL HANDICAP. Arm A ran the OLD regime (greedy trainee vs
    stochastic sentinel, asymmetric teambuilder) worth +8.9 pp to the trainee. The sentinels'
    recorded win rates are therefore INFLATED, which -- since bias = V - outcome -- SHRINKS a
    positive slope. The corrected slope is the sensitivity.
(3) RAW vs REWEIGHTED, computed end to end on the traced sample with NO reweighting at all
    (both V and the outcome), which is what a naive read of the trace tree would report.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from analyze import BOTS, cell_index, cell_stats, ols_slope, rollup

HANDICAP_PP = 0.089   # +8.9 pp [+7.0, +10.7], measured 2026-09-07 (critic_gate.md)


def build(states_path):
    arr = np.load(states_path, allow_pickle=False)
    b, b_unw = rollup(arr), rollup(arr, weighted=False)
    cyc, opps, oidx, cid = cell_index(b)
    n_cells = len(cyc) * len(opps)
    strength = np.full(n_cells, np.nan)
    true_wr = np.full(n_cells, np.nan)
    is_sent = np.zeros(n_cells, bool)
    for c, o, s, t, k_ in zip(b["cycle"], b["opponent"], b["strength"], b["true_wr"],
                              b["opp_class"]):
        k = np.searchsorted(cyc, c) * len(opps) + oidx[o]
        strength[k], true_wr[k], is_sent[k] = s, t, (k_ == "sentinel")
    return arr, b, b_unw, cyc, opps, oidx, cid, n_cells, strength, true_wr, is_sent


def main(states_path, out_path, n_boot=4000, seed=20260909):
    (arr, b, b_unw, cyc, opps, oidx, cid, n_cells,
     strength, true_wr, is_sent) = build(states_path)
    sel0 = np.arange(b["y"].size)
    cyc_of_cell = np.repeat(cyc, len(opps))
    opp_of_cell = np.tile(np.array(opps), len(cyc))
    is_bot_cell = np.isin(opp_of_cell, BOTS)
    st = cell_stats(b, sel0, cid, n_cells, "all", true_wr)
    st_unw = cell_stats(b_unw, sel0, cid, n_cells, "all", true_wr)
    keep_bot = is_bot_cell & (st["n_battles"] > 0)
    keep_all = st["n_battles"] > 0

    res = {"leave_one_out": {}, "handicap": {}, "raw": {}}

    # (1) leave one opponent out
    for tag, keep in (("bot", keep_bot), ("all", keep_all)):
        base = ols_slope(st["bias"], strength, cyc_of_cell, keep)
        res["leave_one_out"][tag] = {"__all__": base}
        pool = BOTS if tag == "bot" else list(opps)
        for o in pool:
            k2 = keep & (opp_of_cell != o)
            res["leave_one_out"][tag][o] = ols_slope(st["bias"], strength, cyc_of_cell, k2)

    # (2) the sentinel handicap: the recorded sentinel win rate is +8.9 pp too generous
    wr_corr = np.where(is_sent, np.clip(true_wr - HANDICAP_PP, 0.0, 1.0), true_wr)
    bias_corr = st["meanV_b"] - wr_corr
    res["handicap"] = {
        "as_recorded_all": ols_slope(st["bias"], strength, cyc_of_cell, keep_all),
        "handicap_corrected_all": ols_slope(bias_corr, strength, cyc_of_cell, keep_all),
        "bot_only_unaffected": ols_slope(st["bias"], strength, cyc_of_cell, keep_bot),
        "handicap_pp": HANDICAP_PP,
    }

    # (3) the fully RAW read: no IPW anywhere, outcome = the traced sample's own win rate
    bias_raw = st_unw["meanV_b"] - st_unw["out_b"]
    res["raw"] = {
        "slope_bot_raw_endtoend": ols_slope(bias_raw, strength, cyc_of_cell, keep_bot),
        "slope_all_raw_endtoend": ols_slope(bias_raw, strength, cyc_of_cell, keep_all),
        "slope_bot_reweighted": ols_slope(st["bias"], strength, cyc_of_cell, keep_bot),
        "slope_all_reweighted": ols_slope(st["bias"], strength, cyc_of_cell, keep_all),
        "mean_traced_win_rate": float(np.nanmean(st_unw["out_b"][keep_all])),
        "mean_true_win_rate": float(np.nanmean(true_wr[keep_all])),
    }

    # bootstrap the two quantities the writeup quotes as intervals
    rng = np.random.default_rng(seed)
    order = np.argsort(cid, kind="stable")
    cid_s = cid[order]
    starts = np.searchsorted(cid_s, np.arange(n_cells), "left")
    sizes = np.searchsorted(cid_s, np.arange(n_cells), "right") - starts
    slots = [order[starts[k]:starts[k] + sizes[k]] for k in range(n_cells)]
    all_slots = np.concatenate([s for s in slots if s.size])
    off = np.concatenate([np.full(s.size, i) for i, s in enumerate(slots) if s.size])
    st_in = []
    p = 0
    for s in slots:
        if s.size:
            st_in.append(np.full(s.size, p))
            p += s.size
    st_in = np.concatenate(st_in)
    sz = sizes[off].astype(float)
    D = {"handicap_corrected_all": [], "raw_bot_endtoend": [], "raw_all_endtoend": [],
         "loo_bot_min": [], "loo_bot_max": []}
    for _ in range(n_boot):
        sel = all_slots[st_in + (rng.random(all_slots.size) * sz).astype(int)]
        wr = rng.binomial(100, true_wr) / 100.0
        s_ = cell_stats(b, sel, off, n_cells, "all", wr)
        su = cell_stats(b_unw, sel, off, n_cells, "all", wr)
        ka = s_["n_battles"] > 0
        kb = is_bot_cell & ka
        wc = np.where(is_sent, np.clip(wr - HANDICAP_PP, 0, 1), wr)
        D["handicap_corrected_all"].append(
            ols_slope(s_["meanV_b"] - wc, strength, cyc_of_cell, ka))
        br = su["meanV_b"] - su["out_b"]
        D["raw_bot_endtoend"].append(ols_slope(br, strength, cyc_of_cell, kb))
        D["raw_all_endtoend"].append(ols_slope(br, strength, cyc_of_cell, ka))
        loo = [ols_slope(s_["bias"], strength, cyc_of_cell, kb & (opp_of_cell != o))
               for o in BOTS]
        D["loo_bot_min"].append(np.nanmin(loo))
        D["loo_bot_max"].append(np.nanmax(loo))
    res["ci"] = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
                 for k, v in D.items()}
    res["n_boot"] = n_boot
    res["seed"] = seed
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()
    main(a.states, a.out, a.boot)
