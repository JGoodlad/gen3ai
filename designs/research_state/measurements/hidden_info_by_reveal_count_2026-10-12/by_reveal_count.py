"""THE POST-HOC, DESCRIPTIVE per-STAGE read of the pilot's EXISTING rows (this folder's README; the amendment is
`designs/endstate/design_hidden_info_binding.md` §12.13). No new playout, no verdict: every number is descriptive and
read AFTER the original study's data existed. It may NOT steer the registered rules of the amendment (§12.13 says so).

Three STAGE axes, one definition each (all derived from the SEARCHED side's own view, no outcome):
  k          the number of distinct opponent species the searched side has seen switch or be dragged in, up to the root
             (= 6 - the pilot's `unrevealed`; `root_facts_<cp>.json` `n_seen` and `make_states.revealed_counts` agree)
  turn       the state's turn (decoded from the obs clock, gate-checked against the protocol by the pilot)
  material   the mons still unfainted on the board, ours + theirs (2..12), from `stage_facts_<cp>.json`
Per cell and checkpoint (end2, prod, pooled; battle-clustered bootstrap, the pilot's `score.Boot`, 2,000 draws, seed
20261011, every contrast paired) it computes, on the states that the oracle AND all four honest world kinds played:
  IMP (win scale, = T(pick) - T(policy top1), the pilot's decision improvement) of
    O  the oracle leaf (Phase 1, the TRUE world), A' the honest v2 Smogon-prior worlds (`det2`), B' belief-species worlds
    (`det2_b`), C' network-move-belief worlds (`det2_c`), CT' C' + the SEEN mons' TRUE sets (`det2_ct`), at LEAF-K8; and
    the one-ply CRITIC and TRUNC-4-K8 of each;
  the gap G = O - A', the headroom over the best HONEST world O - C', its split into the SEEN mons' moves (CT' - C') and
    the rest (O - CT': the never-seen mons, plus what v2 still keeps true), search-over-critic contrasts;
  the decomposition G = (change rate c) x (stakes s): c = share of states where the oracle's pick differs from A''s pick,
    s = the mean TRUE-value gain T(oracle pick) - T(A' pick) among those states (G = c * s exactly: g is 0 where the
    picks agree);
  context: the policy's top-2 logit gap, legal actions, ORACLE_UPPER (max_a T - T(top1), winner's curse), turn, material.

    scripts: run_read.sh by_reveal_count.py   →  by_reveal_count.json  (then render_tables.py → tables.md)
Pure numpy, reading the pilot archive; run through run_read.sh (nice 19, taskset 5-7,13-15, mem_cap).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brc_data as B  # noqa: E402
from brc_data import Bt, CPS, S1  # noqa: E402

PILOT = B.PILOT

#: k bins: a k with fewer than 80 states in EITHER checkpoint merges upward (k = 1: 62 / 67 states -> k = 1-2)
KBINS = (("k1-2", (1, 2)), ("k3", (3, 3)), ("k4", (4, 4)), ("k5", (5, 5)), ("k6", (6, 6)))
KSINGLE = tuple((f"k{k}", (k, k)) for k in range(1, 7))
MIN_CELL = 30            # a joint cell below this is shown by its n only (its numbers are suppressed)
SHARE_MIN_LOWER = 0.02   # a share is computed only when the denominator's lower bound exceeds this (design doc 6.3)
NAMES = B.NAMES


def col(recs, a, nme):
    return np.array([r["imp"][(a, nme)] for r in recs])


def cell(bt, recs, sel, weighted=False):
    """Every descriptive number of one cell (a boolean selection over `recs`)."""
    n = int(np.sum(sel))
    out = {"n": n, "n_battles": int(len({r["battle"] for r, s in zip(recs, sel) if s}))}
    if n < 2:
        return out
    X = {(a, nme): col(recs, a, nme) for a in ("O", "A", "B", "C", "CT") for nme in NAMES}
    L8 = "LEAF-K8"
    out["imp"] = {f"{a}|{nme}": bt.stat(X[(a, nme)], sel) for a in ("O", "A", "C", "CT") for nme in
                  ("LEAF-K8", "CRITIC", "TRUNC-4-K8")}
    out["imp"]["B|LEAF-K8"] = bt.stat(X[("B", L8)], sel)
    G = X[("O", L8)] - X[("A", L8)]
    out["contrast"] = {
        "G = O - A' (LEAF-K8)": bt.stat(G, sel),
        "headroom O - C'": bt.stat(X[("O", L8)] - X[("C", L8)], sel),
        "seen moves CT' - C'": bt.stat(X[("CT", L8)] - X[("C", L8)], sel),
        "rest O - CT'": bt.stat(X[("O", L8)] - X[("CT", L8)], sel),
        "net move belief C' - A'": bt.stat(X[("C", L8)] - X[("A", L8)], sel),
        "search over critic, honest A' (LEAF-K8 - CRITIC)": bt.stat(X[("A", L8)] - X[("A", "CRITIC")], sel),
        "search over critic, honest A' (TRUNC-4-K8 - CRITIC)": bt.stat(X[("A", "TRUNC-4-K8")] - X[("A", "CRITIC")], sel),
        "search over critic, C' (LEAF-K8 - CRITIC)": bt.stat(X[("C", L8)] - X[("C", "CRITIC")], sel),
        "search over critic, oracle (LEAF-K8 - CRITIC)": bt.stat(X[("O", L8)] - X[("O", "CRITIC")], sel),
        "G at CRITIC (oracle critic - honest critic)": bt.stat(X[("O", "CRITIC")] - X[("A", "CRITIC")], sel),
        "G at TRUNC-4-K8": bt.stat(X[("O", "TRUNC-4-K8")] - X[("A", "TRUNC-4-K8")], sel),
    }
    if weighted:
        out["reweighted"] = {"O": bt.stat(X[("O", L8)], sel, True), "A": bt.stat(X[("A", L8)], sel, True),
                             "G": bt.stat(G, sel, True)}
    # shares of the headroom over C', only when its lower bound clears the design doc's 0.02
    hr = X[("O", L8)] - X[("C", L8)]
    shares = {}
    hl = out["contrast"]["headroom O - C'"]["ci"][0]
    for lab, num in (("seen_moves", X[("CT", L8)] - X[("C", L8)]), ("rest", X[("O", L8)] - X[("CT", L8)])):
        est, vec = bt.ratio(num, hr, sel)
        shares[lab] = ({"mean": round(est, 4), "ci": bt._ci(vec)} if (hl is not None and hl > SHARE_MIN_LOWER)
                       else {"mean": None, "ci": None, "note": f"headroom lower {hl} <= {SHARE_MIN_LOWER}: no share"})
    out["share_of_headroom_over_C"] = shares
    # VOI decomposition: G = c * s
    differ = np.array([r["pick"][("O", L8)] != r["pick"][("A", L8)] for r in recs], float)
    gain = np.array([r["gain_OA"] for r in recs])
    est_s, vec_s = bt.ratio(gain, differ, sel)
    out["voi"] = {"change_rate c (oracle pick != honest pick)": bt.stat(differ, sel),
                  "stakes s (true-value gain per changed decision)": {"mean": round(est_s, 5), "ci": bt._ci(vec_s)},
                  "c * s": round(float(differ[sel].mean() * est_s), 5) if np.isfinite(est_s) else None,
                  "oracle moves off the policy's pick": bt.stat(
                      np.array([r["pick"][("O", L8)] != r["top1"] for r in recs], float), sel),
                  "honest A' moves off the policy's pick": bt.stat(
                      np.array([r["pick"][("A", L8)] != r["top1"] for r in recs], float), sel),
                  "regret of the honest pick, max_a T - T(A' pick)": bt.stat(np.array([r["regret_A"] for r in recs]), sel),
                  "action-value spread, max_a T - mean_a T": bt.stat(np.array([r["spread"] for r in recs]), sel)}
    out["context"] = {"ORACLE_UPPER (winner's curse)": bt.stat(np.array([r["upper"] for r in recs]), sel),
                      "policy top-2 logit gap": bt.stat(np.array([r["gap"] for r in recs]), sel),
                      "legal actions": bt.stat(np.array([r["n_legal"] for r in recs], float), sel),
                      "turn": bt.stat(np.array([r["turn"] for r in recs], float), sel),
                      "material (ours + theirs unfainted)": bt.stat(np.array([r["alive"] for r in recs], float), sel),
                      "k": bt.stat(np.array([r["k"] for r in recs], float), sel)}
    return out


def power_rows(bt, recs, sel):
    """SD_eff of the per-state contrasts the amendment's rules read (bootstrap SE x sqrt(n)): the power plan's inputs."""
    n = int(np.sum(sel))
    L8 = "LEAF-K8"
    X = {(a, nme): col(recs, a, nme) for a in ("O", "A", "C", "CT") for nme in NAMES}
    cons = {"G": X[("O", L8)] - X[("A", L8)], "O-C": X[("O", L8)] - X[("C", L8)],
            "CT-C": X[("CT", L8)] - X[("C", L8)], "O-CT": X[("O", L8)] - X[("CT", L8)],
            "A leaf - A crit": X[("A", L8)] - X[("A", "CRITIC")], "O leaf": X[("O", L8)]}
    out = {"n": n}
    for k, x in cons.items():
        est, vec = bt.mean(x, sel)
        se = float(np.nanstd(vec, ddof=1))
        out[k] = {"se": round(se, 5), "sd_eff": round(se * np.sqrt(n), 4), "sd_state": round(float(x[sel].std(ddof=1)), 4)}
    return out


def main():
    allrecs, sts = [], {}
    for cp in CPS:
        st, recs = B.load_records(cp)
        sts[cp] = st
        allrecs += recs
    # --- validity: k agrees with the independent protocol-derived count; the common-state means reproduce the pilot's
    #     own v2 read (`result_moves.json`: the same arms, the same states, its own scorer code) to 1e-4
    val = {"k_mismatch_vs_root_facts": int(sum(r["k"] != r["k_check"] for r in allrecs)),
           "n_common": {cp: sum(r["cp"] == cp for r in allrecs) for cp in CPS}}
    ref = {x["cp"]: x for x in json.loads((PILOT / "result_moves.json").read_text())}
    lab = {"A": "A (Smogon)", "B": "B (belief species)", "C": "C (net moves)", "CT": "CT (true seen moves)",
           "O": "oracle"}
    worst, repro = 0.0, {}
    for cp in CPS:
        rs = [r for r in allrecs if r["cp"] == cp]
        repro[cp] = {"n_common": len(rs), "pilot_n_common": ref[cp]["n_states"]}
        for who, nme in (("O", "LEAF-K8"), ("A", "LEAF-K8"), ("B", "LEAF-K8"), ("C", "LEAF-K8"), ("CT", "LEAF-K8"),
                         ("A", "CRITIC"), ("C", "CRITIC"), ("CT", "CRITIC"), ("A", "TRUNC-4-K4"), ("C", "TRUNC-4-K4"),
                         ("CT", "TRUNC-4-K4"), ("O", "TRUNC-4-K4"), ("A", "TRUNC-4-K8"), ("C", "TRUNC-4-K8"),
                         ("CT", "TRUNC-4-K8"), ("O", "TRUNC-4-K8"), ("O", "CRITIC")):
            mine = float(np.mean([r["imp"][(who, nme)] for r in rs]))
            key = f"ARGMAX|{nme}"
            if key in ref[cp]["imp"]:
                theirs = ref[cp]["imp"][key][lab[who]]["mean"]
                worst = max(worst, abs(mine - theirs))
                repro[cp][f"{who}|{nme}"] = {"mine": round(mine, 5), "pilot": theirs}
    val["reproduction"] = repro
    val["max_abs_diff_vs_pilot_result_moves"] = round(worst, 6)
    if worst > 1e-4 or any(repro[cp]["n_common"] != repro[cp]["pilot_n_common"] for cp in CPS):
        raise SystemExit(f"REPRODUCTION FAILED against the pilot's result_moves.json: {json.dumps(val)[:2000]}")

    # --- stage-axis bucket edges: quartiles of the 1,200 REGISTERED states (a proportional sample of the eligible pool)
    reg = {cp: json.loads((PILOT / f"states_{cp}.json").read_text())["states"] for cp in CPS}
    stage = {cp: json.loads((B.HERE / f"stage_facts_{cp}.json").read_text())["states"] for cp in CPS}
    turns = np.array([reg[cp][d]["turn"] for cp in CPS for d in reg[cp]], float)
    mats = np.array([stage[cp][d]["alive_ours"] + stage[cp][d]["alive_theirs"] for cp in CPS for d in reg[cp]], float)
    t_edges = [int(np.percentile(turns, q)) for q in (25, 50, 75)]
    m_edges = [int(np.percentile(mats, q)) for q in (25, 50, 75)]

    def bucket(x, edges):
        return int(sum(x > e for e in edges))          # 0..3: <= e1, <= e2, <= e3, above
    for r in allrecs:
        r["tb"] = bucket(r["turn"], t_edges)
        r["mb"] = bucket(r["alive"], m_edges)
        r["th"] = int(r["turn"] > t_edges[1])           # the median split: early / late
    tlab = [f"turn<={t_edges[0]}", f"turn {t_edges[0] + 1}-{t_edges[1]}", f"turn {t_edges[1] + 1}-{t_edges[2]}",
            f"turn>={t_edges[2] + 1}"]
    mlab = [f"alive<={m_edges[0]}", f"alive {m_edges[0] + 1}-{m_edges[1]}", f"alive {m_edges[1] + 1}-{m_edges[2]}",
            f"alive>={m_edges[2] + 1}"]
    thlab = [f"early(turn<={t_edges[1]})", f"late(turn>={t_edges[1] + 1})"]
    val["edges"] = {"turn_quartile_edges": t_edges, "material_quartile_edges": m_edges, "turn_labels": tlab,
                    "material_labels": mlab, "thalf_labels": thlab, "n_registered": int(len(turns))}

    res = {"validity": val, "scopes": {}, "power": {}, "trend": {}}
    for scope in ("end2", "prod", "pooled"):
        recs = allrecs if scope == "pooled" else [r for r in allrecs if r["cp"] == scope]
        order_cp = [cp for cp in CPS if any(r["cp"] == cp for r in recs)]
        recs = [r for cp in order_cp for r in recs if r["cp"] == cp]
        wts = np.concatenate([S1.reweights(sts[cp], [sts[cp]["states"][r["id"]] for r in recs if r["cp"] == cp])
                              for cp in order_cp])
        bt = Bt([r["battle"] for r in recs], wts)
        arr = lambda f: np.array([f(r) for r in recs])    # noqa: E731
        K, TB, MB, TH = arr(lambda r: r["k"]), arr(lambda r: r["tb"]), arr(lambda r: r["mb"]), arr(lambda r: r["th"])
        ax = {"all": {"all": cell(bt, recs, np.ones(len(recs), bool), weighted=True)}}
        ax["k"] = {lab_: cell(bt, recs, (K >= lo) & (K <= hi), weighted=True) for lab_, (lo, hi) in KBINS}
        ax["k_single"] = {lab_: cell(bt, recs, (K >= lo) & (K <= hi)) for lab_, (lo, hi) in KSINGLE}
        ax["turn"] = {tlab[i]: cell(bt, recs, TB == i) for i in range(4)}
        ax["material"] = {mlab[i]: cell(bt, recs, MB == i) for i in range(4)}
        ax["k_x_turn_half"] = {f"{kl}|{thlab[h]}": cell(bt, recs, (K >= lo) & (K <= hi) & (TH == h))
                               for kl, (lo, hi) in KBINS for h in (0, 1)}
        ax["k_x_turn4"] = {f"{kl}|{tlab[i]}": cell(bt, recs, (K >= lo) & (K <= hi) & (TB == i))
                           for kl, (lo, hi) in KBINS for i in range(4)}
        ax["k_x_material"] = {f"{kl}|{mlab[i]}": cell(bt, recs, (K >= lo) & (K <= hi) & (MB == i))
                              for kl, (lo, hi) in KBINS for i in range(4)}
        for a_name in ("k_x_turn_half", "k_x_turn4", "k_x_material"):
            for lab_, c in list(ax[a_name].items()):
                if c["n"] < MIN_CELL:
                    ax[a_name][lab_] = {"n": c["n"], "n_battles": c["n_battles"], "suppressed": f"n < {MIN_CELL}"}
        res["scopes"][scope] = ax
        # --- the trends and the stage-vs-reveal separation (OLS of the state-level gap on standardised stage axes)
        G = col(recs, "O", "LEAF-K8") - col(recs, "A", "LEAF-K8")
        OA = col(recs, "O", "LEAF-K8")
        AA = col(recs, "A", "LEAF-K8")
        z = lambda v: (v - v.mean()) / v.std()      # noqa: E731
        Tn, Mn = arr(lambda r: float(r["turn"])), arr(lambda r: float(r["alive"]))
        tr = {"corr_k_turn": round(float(np.corrcoef(K, Tn)[0, 1]), 3),
              "corr_k_material": round(float(np.corrcoef(K, Mn)[0, 1]), 3),
              "corr_turn_material": round(float(np.corrcoef(Tn, Mn)[0, 1]), 3),
              "sd": {"k": round(float(K.std()), 3), "turn": round(float(Tn.std()), 3), "material": round(float(Mn.std()), 3)}}
        for tgt, y in (("G", G), ("oracle_edge_O", OA), ("honest_A", AA)):
            tr[tgt] = {"on k (per +1 SD)": bt.ols(y, z(K)[:, None]),
                       "on turn (per +1 SD)": bt.ols(y, z(Tn)[:, None]),
                       "on material (per +1 SD)": bt.ols(y, z(Mn)[:, None]),
                       "on k + turn": bt.ols(y, np.column_stack([z(K), z(Tn)])),
                       "on k + material": bt.ols(y, np.column_stack([z(K), z(Mn)])),
                       "on k + turn + material": bt.ols(y, np.column_stack([z(K), z(Tn), z(Mn)]))}
        low, high = (K <= 3), (K >= 5)
        tr["G low(k<=3) - high(k>=5)"] = bt.diff(G, low, high)
        tr["oracle edge low - high"] = bt.diff(OA, low, high)
        tr["honest A' low - high"] = bt.diff(AA, low, high)
        # --- the MECHANISM: G = c x s, early-vs-late contrasts on each axis (extreme groups, paired on the same draws)
        differ = arr(lambda r: float(r["pick"][("O", "LEAF-K8")] != r["pick"][("A", "LEAF-K8")]))
        gain = arr(lambda r: r["gain_OA"])
        TBa, MBa = arr(lambda r: r["tb"]), arr(lambda r: r["mb"])
        groups = {"k: k<=3 vs k>=5": (K <= 3, K >= 5), "turn: <=14 vs >=15": (Tn <= t_edges[1], Tn > t_edges[1]),
                  "turn: first quartile vs last quartile": (TBa == 0, TBa == 3),
                  "material: >=10 alive vs <=7 alive": (Mn > m_edges[1], Mn <= m_edges[0]),
                  "k: k<=3 vs k>=5 within alive>=8": ((K <= 3) & (Mn >= 8), (K >= 5) & (Mn >= 8))}
        mech = {}
        for gname, (sa, sb) in groups.items():
            ea, va = bt.ratio(gain, differ, sa)
            eb, vb = bt.ratio(gain, differ, sb)
            mech[gname] = {"n": [int(sa.sum()), int(sb.sum())],
                           "G (first - second)": bt.diff(G, sa, sb), "oracle edge IMP_O": bt.diff(OA, sa, sb),
                           "honest IMP_A'": bt.diff(AA, sa, sb), "change rate c": bt.diff(differ, sa, sb),
                           "stakes s (gain per changed decision)": {"mean": round(ea - eb, 5), "ci": bt._ci(va - vb)},
                           "regret of the honest pick": bt.diff(arr(lambda r: r["regret_A"]), sa, sb),
                           "action-value spread": bt.diff(arr(lambda r: r["spread"]), sa, sb)}
        tr["mechanism_contrasts"] = mech
        res["trend"][scope] = tr
        res["power"][scope] = {"k": {lab_: power_rows(bt, recs, (K >= lo) & (K <= hi)) for lab_, (lo, hi) in KBINS},
                               "all": power_rows(bt, recs, np.ones(len(recs), bool))}
    (HERE / "by_reveal_count.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"validity": {k: v for k, v in val.items() if k != "reproduction"}}, indent=1)[:2500])


if __name__ == "__main__":
    main()
