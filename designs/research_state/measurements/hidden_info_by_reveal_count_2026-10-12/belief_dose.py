"""(C) THE BELIEF-QUALITY DOSE-RESPONSE, DESCRIPTIVE NOW, from the pilot's BANKED per-world, per-action outcomes (this
folder's README; registered as a Tier-1 arm family in `design_hidden_info_binding.md` §12.13 / §6.9).

The idea (owner, 2026-10-12): a search that averages K worlds, the TRUE world with weight p and honest worlds sharing
1 - p, and a curve of its decision improvement against p. The playouts do not depend on p, so ONE set of per-world
outcomes gives the whole curve by re-weighting. The archive banks exactly those outcomes (Phase 1 = the true world,
16 leaf dice seeds per action; `det2` = 8 honest v2 worlds, one playout each, dice seed j in world j), so no playout
is run here. Everything below is a re-weighting of banked numbers, scored by the pilot's own referee (24 truth seeds,
16-39, which no scorer reads), with its tie rule (toward the policy's order).

ESTIMATORS (score of action a, per scorer z in LEAF-K8, TRUNC-4-K8, CRITIC; H = the honest worlds' per-world values):
  M8  S_p(a) = p * mean(true world, 8 seeds) + (1 - p) * mean(all 8 honest worlds).   The two ENDPOINTS are exactly the
      pilot's banked reads (p = 0: A' LEAF-K8 etc.; p = 1: the oracle LEAF-K8): the SHAPE between them is the new fact.
      The true world is known to 8 dice seeds, so M8 is the 'belief mass' curve with the true world's own noise matched
      to the honest component's (8 playouts each).
  M5  the LITERAL design: K = 5 worlds, each played ONCE per action: the true world (dice seed s0) with weight p and
      4 of the 8 honest worlds with weight (1 - p)/4 each.  IMP is averaged over all 16 x 70 (s0, 4-subset) draws (an
      exact U-statistic, no Monte Carlo); the critic's true world uses its 8 successor seeds.
  M5W the PLACEBO of M5: ONE honest world (a confident WRONG world: a random draw of the belief) at weight p on its
      single playout, 4 other honest worlds sharing 1 - p; averaged over all 8 x 35 choices.  p = 0 is identical to M5.
      M5 - M5W at the same p is the value of the CONTENT (the world being TRUE), not of concentrating weight.
Then, per cell (all states, k bin, turn bucket, material bucket; end2, prod, pooled; battle-clustered bootstrap on the
pilot's draws, every contrast paired because the worlds are shared across p):
  the IMP(p) curve, the BREAK-EVEN p* (the smallest p after which the contrast stays above the 0.0005 edge, by the point
  estimate and by the 95 % lower bound) where honest-world search (a) beats the policy, (b) beats the critic-only search
  at the SAME p, (c) beats the critic-only search at TODAY'S belief (p = 0), and the IMP-EQUIVALENT p of the world kinds
  we actually have (B' belief species, C' network move belief, CT' + the seen mons' true sets): the p at which the M8
  curve equals that world kind's own measured IMP, i.e. where today's belief sits on the curve in units of VALUE.

    scripts: run_read.sh belief_dose.py   →  belief_dose.json
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brc_data as B  # noqa: E402
from brc_data import Bt, CPS, S1, TIE  # noqa: E402

P7 = np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])           # the owner's grid
PF = np.round(np.linspace(0.0, 1.0, 21), 2)                      # the fine grid (interpolation, p*, p_eq)
SCORERS = ("LEAF-K8", "TRUNC-4-K8", "CRITIC")
EDGE = S1.EDGE
SUB4 = np.array(list(combinations(range(8), 4)))                 # 70 honest 4-subsets
WRONG = [(w, np.array(list(combinations([j for j in range(8) if j != w], 4)))) for w in range(8)]   # 8 x 35


def imp_of(S, Tp):
    """S [..., A] in POLICY order (index 0 = the policy's top1): pick = the first action within TIE of the max."""
    m = S.max(-1, keepdims=True)
    idx = (S >= m - TIE).argmax(-1)
    return Tp[idx] - Tp[0]


def components(b, ordr):
    """{scorer: (T_mean [A], T_seeds [S0, A], H [8, A])} in policy order."""
    g = b["arms"]["A"]
    return {
        "LEAF-K8": (b["WT"][ordr][:, :8].mean(1), b["WT"][ordr].T, g["W"][ordr].T),
        "TRUNC-4-K8": (b["X4T"][ordr][:, :8].mean(1), b["X4T"][ordr].T, g["X4"][ordr].T),
        "CRITIC": (b["critT"][ordr], b["critTs"][ordr].T, g["crit"][ordr].T),
    }


def state_curves(b):
    """Per scorer, the IMP of every estimator at every p: {name: array}. M8 on PF; M5 / M5W on P7."""
    ordr = np.array(b["order"])
    Tp = b["T"][ordr]
    out = {}
    for z, (Tm, Ts, H) in components(b, ordr).items():
        Hm = H.mean(0)
        S = PF[:, None] * Tm[None, :] + (1 - PF)[:, None] * Hm[None, :]                   # [P, A]
        out[f"M8|{z}"] = imp_of(S, Tp)
        out[f"scale|{z}"] = np.array([Tm.std(), Hm.std()])             # signal scale across actions: true world, honest mean
        Hsub = H[SUB4].mean(1)                                                              # [70, A]
        p = P7[:, None, None, None]
        S5 = p * Ts[None, :, None, :] + (1 - p) * Hsub[None, None, :, :]                   # [P, S0, 70, A]
        out[f"M5|{z}"] = imp_of(S5, Tp).mean((1, 2))
        heavy = np.concatenate([np.repeat(H[w][None, :], len(J), axis=0) for w, J in WRONG])          # [280, A]
        light = np.concatenate([H[J].mean(1) for _w, J in WRONG])                                     # [280, A]
        SW = P7[:, None, None] * heavy[None] + (1 - P7)[:, None, None] * light[None]                  # [P, 280, A]
        out[f"M5W|{z}"] = imp_of(SW, Tp).mean(1)
    return out


def persistent_first(mask):
    """The smallest index i such that mask[i:] is all True (None if the last entry is False)."""
    if not mask[-1]:
        return None
    i = len(mask) - 1
    while i > 0 and mask[i - 1]:
        i -= 1
    return i


def crossing(curve_mean, curve_vec, thr, grid):
    """The p* of a contrast curve: by the point estimate and by the 95 % lower bound, and the bootstrap interval of the
    per-draw point crossing (None = never within [0, 1])."""
    lo = np.nanpercentile(curve_vec, 2.5, axis=0)
    ip, il = persistent_first(curve_mean > thr), persistent_first(lo > thr)
    per_draw = []
    for v in curve_vec:
        j = persistent_first(v > thr) if np.isfinite(v).all() else None
        per_draw.append(np.inf if j is None else grid[j])
    per_draw = np.array(per_draw)
    never = np.isinf(per_draw)
    q = np.percentile(np.where(never, 9.0, per_draw), [2.5, 50, 97.5])      # 9.0 = 'never' (inf breaks the interpolation)
    return {"p_star_point": None if ip is None else float(grid[ip]), "p_star_lower95": None if il is None else float(grid[il]),
            "per_draw_interval": [None if x > 1.5 else float(x) for x in q],
            "share_of_draws_never": round(float(never.mean()), 4)}


def equivalent_p(curve_mean, curve_vec, target_mean, target_vec, grid):
    """The p at which the (monotone-enough) curve first reaches the target's IMP: linear interpolation on the grid.
    Returns the point value and the 95 % interval over draws; 0.0 = the target is at or below the curve's p = 0 value,
    None = the target lies above the curve's p = 1 value."""
    def one(c, t):
        if t <= c[0]:
            return 0.0
        for i in range(1, len(c)):
            if c[i] >= t:
                lo, hi = c[i - 1], c[i]
                return float(grid[i - 1] + (grid[i] - grid[i - 1]) * (t - lo) / (hi - lo)) if hi > lo else float(grid[i])
        return None
    pt = one(curve_mean, target_mean)
    ds = [one(c, t) for c, t in zip(curve_vec, target_vec)]
    ds = np.array([9.0 if d is None else d for d in ds])                     # 9.0 = above the curve's p = 1 value
    q = np.percentile(ds, [2.5, 50, 97.5])
    return {"p_eq": pt, "interval": [None if x > 1.5 else round(float(x), 3) for x in q]}


def cell_read(bt, X, other, sel, scales=None):
    """X: {name: [n, P]} the per-state IMP curves; other: {kind: [n]} the world kinds' own LEAF-K8 IMP."""
    out = {"n": int(np.sum(sel))}
    curves = {}
    for name, M in X.items():
        grid_is_fine = name.startswith("M8")
        mean, vec = bt.curve(M, sel)
        curves[name] = (mean, vec)
        keep = [list(PF if grid_is_fine else P7).index(p) for p in (P7 if grid_is_fine else P7)]
        out[name] = {"p": [float(x) for x in P7],
                     "imp": [round(float(mean[i]), 5) for i in keep],
                     "ci": [bt._ci(vec[:, i]) for i in keep]}
    # M5 vs placebo, paired differences at each p
    for z in SCORERS:
        m5, v5 = curves[f"M5|{z}"]
        mw, vw = curves[f"M5W|{z}"]
        out[f"M5-M5W|{z}"] = {"p": [float(x) for x in P7], "diff": [round(float(m5[i] - mw[i]), 5) for i in range(len(P7))],
                              "ci": [bt._ci(v5[:, i] - vw[:, i]) for i in range(len(P7))]}
        out[f"M5 gain over p=0|{z}"] = {"p": [float(x) for x in P7],
                                        "diff": [round(float(m5[i] - m5[0]), 5) for i in range(len(P7))],
                                        "ci": [bt._ci(v5[:, i] - v5[:, 0]) for i in range(len(P7))]}
    # break-evens on the fine M8 grid
    lm, lv = curves["M8|LEAF-K8"]
    cm, cv = curves["M8|CRITIC"]
    tm, tv = curves["M8|TRUNC-4-K8"]
    out["break_even"] = {
        "LEAF beats the policy (IMP > edge)": crossing(lm, lv, EDGE, PF),
        "TRUNC-4 beats the policy": crossing(tm, tv, EDGE, PF),
        "CRITIC beats the policy": crossing(cm, cv, EDGE, PF),
        "LEAF beats the critic search at the same p": crossing(lm - cm, lv - cv, EDGE, PF),
        "TRUNC-4 beats the critic search at the same p": crossing(tm - cm, tv - cv, EDGE, PF),
        "LEAF beats the critic search built from today's belief (p = 0)": crossing(
            lm - cm[0], lv - cv[:, [0]], EDGE, PF),
        "LEAF beats the best honest scorer at p = 0 (max of LEAF, TRUNC-4, CRITIC)": crossing(
            lm - max(lm[0], tm[0], cm[0]), lv - np.maximum(np.maximum(lv[:, [0]], tv[:, [0]]), cv[:, [0]]), EDGE, PF),
    }
    # where today's world kinds sit on the M8 LEAF curve (units of VALUE)
    out["p_equivalent_of_world_kinds"] = {}
    for kind, x in other.items():
        km, kv = bt.mean(x, sel)
        out["p_equivalent_of_world_kinds"][kind] = {
            "imp": round(km, 5), **equivalent_p(lm, lv, km, kv, PF)}
    if scales is not None:
        out["signal_scale_across_actions"] = {
            z: {"true world (8 seeds)": bt.stat(scales[f"scale|{z}"][:, 0], sel),
                "honest worlds' mean (8 worlds)": bt.stat(scales[f"scale|{z}"][:, 1], sel),
                "ratio of means": round(float(scales[f"scale|{z}"][sel][:, 0].mean() / scales[f"scale|{z}"][sel][:, 1].mean()), 3)}
            for z in SCORERS}
    # p-half: the weight on the truth at which the M8 leaf curve has covered half of its own p = 0 -> p = 1 rise
    half_t = (lm[0] + lm[-1]) / 2.0
    half_v = (lv[:, 0] + lv[:, -1]) / 2.0
    out["p_half_leaf"] = equivalent_p(lm, lv, half_t, half_v, PF)
    out["endpoints_leaf"] = {"p=0 (honest A')": round(float(lm[0]), 5), "p=1 (oracle)": round(float(lm[-1]), 5)}
    return out


def main():
    bundles, sts = [], {}
    for cp in CPS:
        st, bund = B.load_bundle(cp)
        sts[cp] = st
        bundles += [bund[d] for d in sorted(bund)]
    names = [f"{kind}|{z}" for kind in ("M8", "M5", "M5W") for z in SCORERS] + [f"scale|{z}" for z in SCORERS]
    X = {n: [] for n in names}
    for b in bundles:
        c = state_curves(b)
        for n in names:
            X[n].append(c[n])
    X = {n: np.array(v) for n, v in X.items()}
    scales = {n: X.pop(n) for n in [m for m in list(X) if m.startswith("scale|")]}
    # world kinds' own IMP, for the p-equivalent
    recs = []
    for cp in CPS:
        _st, rs = B.load_records(cp)
        recs += rs
    assert [r["id"] for r in recs] == [b["id"] for b in bundles]
    kinds = {f"{k}' LEAF-K8": np.array([r["imp"][(k, "LEAF-K8")] for r in recs]) for k in ("B", "C", "CT")}
    kinds["A' LEAF-K8"] = np.array([r["imp"][("A", "LEAF-K8")] for r in recs])
    kinds["oracle LEAF-K8"] = np.array([r["imp"][("O", "LEAF-K8")] for r in recs])

    # --- validity: the endpoints ARE the pilot's reads; p-linearity; M5 p = 0 == M5W p = 0
    cp_arr = np.array([b["cp"] for b in bundles])
    val = {}
    for cp in CPS:
        m = cp_arr == cp
        val[cp] = {"M8 LEAF p=0 vs A' LEAF-K8": [round(float(X["M8|LEAF-K8"][m][:, 0].mean()), 6), round(float(kinds["A' LEAF-K8"][m].mean()), 6)],
                   "M8 LEAF p=1 vs oracle LEAF-K8": [round(float(X["M8|LEAF-K8"][m][:, -1].mean()), 6), round(float(kinds["oracle LEAF-K8"][m].mean()), 6)],
                   "M8 CRITIC p=0": round(float(X["M8|CRITIC"][m][:, 0].mean()), 6),
                   "M5 vs M5W at p=0 (max abs per-state diff)": round(float(np.abs(X["M5|LEAF-K8"][:, 0] - X["M5W|LEAF-K8"][:, 0]).max()), 8)}
        assert abs(val[cp]["M8 LEAF p=0 vs A' LEAF-K8"][0] - val[cp]["M8 LEAF p=0 vs A' LEAF-K8"][1]) < 1e-9
        assert abs(val[cp]["M8 LEAF p=1 vs oracle LEAF-K8"][0] - val[cp]["M8 LEAF p=1 vs oracle LEAF-K8"][1]) < 1e-9
    assert np.abs(X["M5|LEAF-K8"][:, 0] - X["M5W|LEAF-K8"][:, 0]).max() < 1e-9

    # --- stage cells (the SAME buckets as by_reveal_count.py: quartiles of the 1,200 registered states)
    reg = {cp: json.loads((B.PILOT / f"states_{cp}.json").read_text())["states"] for cp in CPS}
    stage = {cp: json.loads((B.HERE / f"stage_facts_{cp}.json").read_text())["states"] for cp in CPS}
    turns = np.array([reg[cp][d]["turn"] for cp in CPS for d in reg[cp]], float)
    mats = np.array([stage[cp][d]["alive_ours"] + stage[cp][d]["alive_theirs"] for cp in CPS for d in reg[cp]], float)
    t_edges = [int(np.percentile(turns, q)) for q in (25, 50, 75)]
    m_edges = [int(np.percentile(mats, q)) for q in (25, 50, 75)]
    K = np.array([b["k"] for b in bundles])
    TB = np.array([int(sum(b["turn"] > e for e in t_edges)) for b in bundles])
    MB = np.array([int(sum(b["alive"] > e for e in m_edges)) for b in bundles])
    tlab = [f"turn<={t_edges[0]}", f"turn {t_edges[0] + 1}-{t_edges[1]}", f"turn {t_edges[1] + 1}-{t_edges[2]}", f"turn>={t_edges[2] + 1}"]
    mlab = [f"alive<={m_edges[0]}", f"alive {m_edges[0] + 1}-{m_edges[1]}", f"alive {m_edges[1] + 1}-{m_edges[2]}", f"alive>={m_edges[2] + 1}"]
    kbins = (("k1-2", (1, 2)), ("k3", (3, 3)), ("k4", (4, 4)), ("k5", (5, 5)), ("k6", (6, 6)), ("k1-3", (1, 3)), ("k4-6", (4, 6)),
             ("k5-6", (5, 6)))
    cells = {"all": np.ones(len(bundles), bool)}
    cells.update({lab: (K >= lo) & (K <= hi) for lab, (lo, hi) in kbins})
    cells.update({tlab[i]: TB == i for i in range(4)})
    cells.update({mlab[i]: MB == i for i in range(4)})
    res = {"validity": val, "grids": {"owner_P7": [float(x) for x in P7], "fine_PF": [float(x) for x in PF]},
           "edges": {"turn": t_edges, "material": m_edges}, "scopes": {}}
    for scope in ("end2", "prod", "pooled"):
        sel_scope = np.ones(len(bundles), bool) if scope == "pooled" else (cp_arr == scope)
        idx = np.where(sel_scope)[0]
        order_cp = [cp for cp in CPS if any(cp_arr[idx] == cp)]
        wts = np.concatenate([S1.reweights(sts[cp], [sts[cp]["states"][bundles[i]["id"]] for i in idx if bundles[i]["cp"] == cp])
                              for cp in order_cp])
        bt = Bt([bundles[i]["battle"] for i in idx], wts)
        Xs = {n: v[idx] for n, v in X.items()}
        scs = {n: v[idx] for n, v in scales.items()}
        os_ = {k: v[idx] for k, v in kinds.items() if k.startswith(("B'", "C'", "CT'"))}   # A' is p = 0, the oracle p = 1: trivial
        res["scopes"][scope] = {lab: cell_read(bt, Xs, os_, sel[idx], scs) for lab, sel in cells.items() if sel[idx].sum() >= 30}
    (HERE / "belief_dose.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"validity": val, "edges": res["edges"]}, indent=1))


if __name__ == "__main__":
    main()
