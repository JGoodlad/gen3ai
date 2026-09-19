"""THE READ — every scorer, every pair-type column, under BOTH label regimes, plus the two
registered CEILING instruments.

    python3 score_controls.py --forks <banked dir> --snapshot <zip> \
        --hand <hand dir> --reroll <reroll dir> --fit <fitted-head dir> --out <json>

Nothing here re-implements the metric. `refit.load_forks` / `build_table` / `read_head` /
`paired_delta` / `boot_ci` are imported BY PATH from `paired_refit_discrimination_2026-09-14`,
`score_forks.branch_index` from `fork_arm_read_2026-09-16` (the alignment assertion), and
`fit_heads.forward_feats` / `head_V` / `wide_head` from `offline_leaf_fit_2026-09-18`. What is new
is (a) the HAND scorers, (b) the AVERAGED-label pair table, (c) the ceilings.

THE HAND SCORERS are built from the board captured by `hand_capture.py`, joined to the banked
successors by LOCAL index within a shard — the same shard order `refit.load_forks` concatenates in.
Every joined row carries its own `obs_match`, and a row whose re-run did not reproduce the banked
successor obs element-wise is DROPPED and counted: a hand score computed on a different state than
the head saw would not be a control.

THE AVERAGED-LABEL TABLE is enumerated here rather than by `build_table`, because `build_table`'s
label is a win/loss string and the averaged label is a mean in [0,1]. The pair rule is otherwise
identical — all pairs within a fork, "non-tied" means the two labels differ, the scorer is
`sign(V_a - V_b)` with 0.5 on an exact V tie, and every CI bootstraps over FORKS.

THE TWO CEILINGS (registered in PREDICTION.md §3.2):

1. SPLIT-HALF self-agreement — label A = mean of re-rolls {0,2,4,6}, label B = mean of {1,3,5,7}.
   The rate at which `sign(A_a - A_b)` equals `sign(B_a - B_b)` is what a K=4 label agrees with
   ITSELF about; no scorer read against a K=4 label can beat it.
2. The MOMENT-ESTIMATED true-gap distribution. For each branch the K rollouts give an unbiased
   `p_hat` and a noise variance `p_hat(1-p_hat)/K`; across pairs,
   `Var(p_a - p_b) = Var(p_hat_a - p_hat_b) - E[noise_a + noise_b]`. From a folded-normal gap
   distribution with that variance, `ACC_oracle(K)` of PREDICTION.md §2 is integrated at K=1 and
   K=8 — the accuracy a scorer that knew both true win probabilities EXACTLY would read against a
   K-rollout label. It is the number every measured accuracy is compared against.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

BRANCHES = ("top1", "top2", "rand")
_MEAS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("paired_refit_discrimination_2026-09-14", "fork_arm_read_2026-09-16",
           "offline_leaf_fit_2026-09-18"):
    sys.path.insert(0, os.path.join(_MEAS, _d))

_TYPE_CHART = None


def _mult(att_types, def_types):
    """The best gen-3 type multiplier an attacker with ``att_types`` has into ``def_types`` —
    STAB-only, no move list, no damage roll. A deliberately crude matchup term: its job is to be
    the one quantity for which OMNISCIENCE is not free (the opponent's unrevealed bench)."""
    global _TYPE_CHART
    if _TYPE_CHART is None:
        from agents.gen3_data import type_chart
        _TYPE_CHART = type_chart.chart()
    best = 0.0
    for a in att_types:
        m = 1.0
        for d in def_types:
            row = _TYPE_CHART.get(str(d).upper()) or _TYPE_CHART.get(str(d).lower()) or {}
            # poke-env's chart is {DEF: {ATT: multiplier}}
            m *= float(row.get(str(a).upper(), row.get(str(a).lower(), 1.0)))
        best = max(best, m)
    return best


def _type_edge(board, omni: bool) -> float:
    """(our active into their alive team) − (their active into our alive team), means of the
    best-STAB multipliers. ``omni`` swaps the opponent's REVEALED alive set for its TRUE one."""
    our = board["our"]
    theirs = board["opp_true"] if (omni and board.get("opp_true")) else board["opp_seen"]
    ours_at = our.get("active_types") or []
    their_at = theirs.get("active_types") or []
    their_team = theirs.get("alive_types") or []
    our_team = our.get("alive_types") or []
    off = float(np.mean([_mult(ours_at, t) for t in their_team])) if (ours_at and their_team) else 1.0
    dfn = float(np.mean([_mult(their_at, t) for t in our_team])) if (their_at and our_team) else 1.0
    return off - dfn


_MAT_HP_W, _MAT_ALIVE_W, _STATUS_W = 2.0, 1.25, 0.3


def _hand_scores(board):
    """Every hand scorer for ONE successor board, as a dict. Signs are OURS-minus-THEIRS."""
    our, seen, true = board["our"], board["opp_seen"], board.get("opp_true")
    phi = board["phi"]
    # one-sided opponent material, the production convention: unrevealed declared slots count
    # full-HP-alive (`_compute_phi_mat`). Recomputed here only for MAT; PBRS uses Φ_mat itself.
    n_unrev = max(0, seen["declared"] - seen["n_known"])
    opp_hp_1s = seen["hp_known"] + n_unrev
    opp_alive_1s = seen["alive_known"] + n_unrev
    mat_1s = (_MAT_HP_W * (our["hp_known"] - opp_hp_1s)
              + _MAT_ALIVE_W * (our["alive_known"] - opp_alive_1s))
    pbrs_1s = phi["mat"] + phi["status"] + phi["hazard"] + phi["boost"] + phi["opp_boosts"] + phi["roar"]
    out = {
        "MAT_1S": mat_1s,
        "PBRS_1S": pbrs_1s,
        "PBRS_BELIEF_1S": pbrs_1s + float(phi.get("belief", 0.0)),
        "BELIEF_1S": float(phi.get("belief", 0.0)),
        "TYPE_1S": mat_1s + 2.0 * _type_edge(board, omni=False),
    }
    if true is not None:
        mat_omni = (_MAT_HP_W * (our["hp_known"] - true["hp_known"])
                    + _MAT_ALIVE_W * (our["alive_known"] - true["alive_known"]))
        # Φ_status' omniscient twin: the public terms are unchanged, the opponent tempo count is
        # the TRUE one rather than the revealed one.
        status_omni = _STATUS_W * (true["tempo"] - our["tempo"])
        out["MAT_OMNI"] = mat_omni
        out["PBRS_OMNI"] = (mat_omni + status_omni + phi["hazard"] + phi["boost"]
                            + phi["opp_boosts"] + phi["roar"])
        out["PBRS_BELIEF_OMNI"] = out["PBRS_OMNI"] + float(phi.get("belief", 0.0))
        out["TYPE_OMNI"] = mat_omni + 2.0 * _type_edge(board, omni=True)
    return out


HAND_NAMES = ("MAT_1S", "PBRS_1S", "PBRS_BELIEF_1S", "BELIEF_1S", "TYPE_1S",
              "MAT_OMNI", "PBRS_OMNI", "PBRS_BELIEF_OMNI", "TYPE_OMNI")


def load_hand(hand_dir, forks_dir):
    """Join the captured boards to the GLOBAL successor index `refit.load_forks` assigns.

    `load_forks` concatenates the shards in `sorted(glob('forks_s*.jsonl'))` order, offsetting each
    shard's successor indices by the running length of the successor arrays. This reproduces that
    offset EXACTLY from the same sorted order, so a hand score lands on the same row as the head's
    V. Returns (scores dict of name -> float array with NaN where unjoined, match mask, stats)."""
    paths = sorted(glob.glob(os.path.join(forks_dir, "forks_s*.jsonl")))
    off, total = {}, 0
    for rp in paths:
        tag = os.path.basename(rp)[len("forks_"):-len(".jsonl")]
        S = np.load(os.path.join(forks_dir, f"succ_{tag}.npy"), mmap_mode="r")
        off[tag] = total
        total += len(S)
    vals = {n: np.full(total, np.nan) for n in HAND_NAMES}
    turn = np.full(total, np.nan)
    st = {"lines": 0, "joined": 0, "obs_mismatch_dropped": 0, "no_opp_true": 0,
          "turn_mismatch": 0, "unknown_shard": 0}
    for hp in sorted(glob.glob(os.path.join(hand_dir, "hand_s*.jsonl"))):
        for line in open(hp):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                break
            st["lines"] += 1
            if d["shard"] not in off:
                st["unknown_shard"] += 1
                continue
            if not d.get("obs_match"):
                st["obs_mismatch_dropped"] += 1
                continue
            b = d["board"]
            if b.get("opp_true") is None:
                st["no_opp_true"] += 1
            if b.get("opp_turn") is not None and b.get("opp_turn") != b.get("turn"):
                st["turn_mismatch"] += 1
            g = off[d["shard"]] + int(d["succ"])
            for k, v in _hand_scores(b).items():
                vals[k][g] = v
            turn[g] = b["turn"]
            st["joined"] += 1
    return vals, turn, st


def _acc_ci(agree, pf_nt, boot_ci, seed=7):
    def acc_for(fork_ids):
        m = np.isin(pf_nt, fork_ids)
        return float(agree[m].mean()) if m.sum() else None
    lo, hi = boot_ci(acc_for, pf_nt, seed=seed)
    return float(agree.mean()) if len(agree) else float("nan"), [lo, hi]


def acc_oracle(gap_sd, K, base_p=0.5, n=200001, seed=3):
    """ACC_oracle(K) — the accuracy a scorer knowing both true win probabilities EXACTLY reads
    against a K-rollout label, integrated over a half-normal gap distribution of sd ``gap_sd``.

    K=1: the pair is non-tied with probability p_a(1-p_b)+p_b(1-p_a) and the oracle is right
    p_a(1-p_b) of the time inside that — PREDICTION.md §2.
    K>1: the label is a mean of K Bernoulli draws; the oracle is right when the two SAMPLE means
    order the same way as the true ones, which is estimated by simulation on the same gaps."""
    rng = np.random.default_rng(seed)
    g = np.abs(rng.normal(0.0, max(gap_sd, 1e-9), n))
    pa = np.clip(base_p + g / 2, 0.0, 1.0)
    pb = np.clip(base_p - g / 2, 0.0, 1.0)
    if K == 1:
        w = pa * (1 - pb)
        l = pb * (1 - pa)
        tot = w + l
        keep = tot > 0
        return float((w[keep] / tot[keep]).mean()), float(tot.mean())
    ma = rng.binomial(K, pa) / K
    mb = rng.binomial(K, pb) / K
    nt = ma != mb
    return float((ma[nt] > mb[nt]).mean()), float(nt.mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forks", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--reroll", default="")
    ap.add_argument("--fit", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)
    import refit
    from refit import boot_ci, build_table, paired_delta, read_head
    from score_forks import branch_index
    from forks import load_model
    from fit_heads import forward_feats, head_V, wide_head
    from agents.model.aux_value_heads import WinProbHead

    out = {"inputs": {"forks": args.forks, "hand": args.hand, "reroll": args.reroll,
                      "snapshot": args.snapshot, "fit": args.fit}}

    # ---------------- the banked set, exactly as the published read loads it ------------------
    rows, succ, sm = refit.load_forks(args.forks)
    model = load_model(args.snapshot)
    pooled, prepool, V0 = forward_feats(model, succ, sm)
    tab, pairs = build_table(rows, pooled, V0)
    b = branch_index(rows)
    assert np.allclose(tab["V0"], V0[b]), "REFUSED: branch-row / successor index disagreement"
    name_of = tab["name"]
    ptype = np.array(["|".join(sorted((name_of[x], name_of[y])))
                      for x, y in zip(pairs["a"], pairs["b"])])
    out["dataset"] = {"n_rows": len(rows), "n_successors": int(len(succ)),
                      "n_branch_rows": int(len(tab["y"])), "n_pairs": int(len(pairs["y"])),
                      "n_nontied": int((pairs["y"] != 0.5).sum()),
                      "base_rate": float(tab["y"].mean())}

    # ---------------- the scorers -------------------------------------------------------------
    V = {"headW": V0[b]}
    if args.fit:
        Xp = pooled[b]
        Xpre = np.concatenate([prepool[b].reshape(len(b), -1), Xp], 1).astype(np.float32)
        for nm, mk, pre in (("fit_c_wide_pooled", lambda: wide_head(128), False),
                            ("fit_d_wide_prepool", lambda: wide_head(Xpre.shape[1]), True),
                            ("fit_a_winprob_warm", WinProbHead, False)):
            fp = os.path.join(args.fit, {"fit_c_wide_pooled": "head_c_wide_pooled.pt",
                                         "fit_d_wide_prepool": "head_d_wide_prepool.pt",
                                         "fit_a_winprob_warm": "head_a_winprob_rand_warm.pt"}[nm])
            if not os.path.exists(fp):
                continue
            h = mk()
            h.load_state_dict(torch.load(fp, weights_only=False)["win_head"])
            h.eval()
            V[nm] = head_V(h, Xpre if pre else Xp)

    hand, succ_turn, hstats = load_hand(args.hand, args.forks)
    out["hand_join"] = hstats
    for nm in HAND_NAMES:
        v = hand[nm][tab["idx"]]
        if np.isfinite(v).sum() == 0:
            continue
        V[nm] = v
    out["hand_coverage"] = {nm: float(np.isfinite(V[nm]).mean()) for nm in HAND_NAMES if nm in V}

    # 🚨 A scorer with NaNs on some rows would silently score those pairs as a V tie (0.5). Every
    # scorer is read on the SAME pairs, and that set is the pairs both branch rows of which have a
    # finite score for EVERY scorer — declared, and its size reported.
    finite = np.ones(len(tab["y"]), bool)
    for nm, v in V.items():
        finite &= np.isfinite(v)
    pair_ok = finite[pairs["a"]] & finite[pairs["b"]]
    out["common_pair_set"] = {"n_pairs_all": int(len(pairs["y"])),
                              "n_pairs_common": int(pair_ok.sum()),
                              "n_branch_rows_finite": int(finite.sum())}

    # ---------------- READ 1: the SINGLE-ROLLOUT labels, by pair type -------------------------
    single = {}
    for col in ["POOLED"] + sorted(set(ptype)):
        sub = np.flatnonzero(pair_ok if col == "POOLED" else (pair_ok & (ptype == col)))
        cell = {"n_pairs": int(len(sub)),
                "n_nontied": int((pairs["y"][sub] != 0.5).sum()), "heads": {}}
        for nm in V:
            r = read_head(V[nm], tab, pairs, sub, nm)
            cell["heads"][nm] = {"acc": r["pairwise_acc"], "ci": r["pairwise_acc_ci"],
                                 "sep_ratio": r["sep_ratio"]}
        for nm in V:
            if nm == "headW":
                continue
            cell["heads"][nm]["delta_vs_headW"] = paired_delta(V[nm], V["headW"], pairs, sub)
        # POST-HOC: 19 % of successors sit at the SAME turn as the fork (a KO forced a replacement
        # inside the fork turn), so a pair can compare states at DIFFERENT depths. Split on it.
        st_a = succ_turn[tab["idx"]][pairs["a"][sub]]
        st_b = succ_turn[tab["idx"]][pairs["b"][sub]]
        same = np.isfinite(st_a) & np.isfinite(st_b) & (st_a == st_b)
        cell["depth_split"] = {}
        for lab, m in (("same_turn", same), ("diff_turn", ~same)):
            ss = sub[m]
            if len(ss) < 50:
                continue
            cell["depth_split"][lab] = {"n_pairs": int(len(ss)), "heads": {}}
            for nm in V:
                r = read_head(V[nm], tab, pairs, ss, nm)
                cell["depth_split"][lab]["n_nontied"] = r["n_nontied"]
                cell["depth_split"][lab]["heads"][nm] = {"acc": r["pairwise_acc"],
                                                         "ci": r["pairwise_acc_ci"]}
        single[col] = cell
        print(f"[single] {col:12s} n={len(sub):6d} nt={cell['n_nontied']:5d}  " +
              "  ".join(f"{k}={v['acc']:.4f}" for k, v in cell["heads"].items()), flush=True)
    out["single_label"] = single

    if not args.reroll:
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"-> {args.out} (no --reroll; control 2 not scored)")
        return 0

    # ---------------- READ 2: the K-AVERAGED labels -------------------------------------------
    # The join is by (shard, local succ) → the same global index the single-label read uses.
    paths = sorted(glob.glob(os.path.join(args.forks, "forks_s*.jsonl")))
    off, total = {}, 0
    for rp in paths:
        tg = os.path.basename(rp)[len("forks_"):-len(".jsonl")]
        off[tg] = total
        total += len(np.load(os.path.join(args.forks, f"succ_{tg}.npy"), mmap_mode="r"))
    g_of = {int(i): k for k, i in enumerate(tab["idx"])}      # global succ index → branch row

    rr = []
    for rp in sorted(glob.glob(os.path.join(args.reroll, "reroll_s*.jsonl"))):
        for line in open(rp):
            try:
                rr.append(json.loads(line))
            except json.JSONDecodeError:
                break
    K = int(rr[0]["k"]) if rr else 0
    avg_rows, dropped = [], {"no_common_succ": 0, "short_k": 0}
    for d in rr:
        if len(set(len(d["scores"][b]) for b in BRANCHES)) != 1 or len(d["scores"]["top1"]) != K:
            dropped["short_k"] += 1
            continue
        g = {b: off[d["shard"]] + int(d["succ"][b]) for b in BRANCHES}
        avg_rows.append({"g": g, "d": d})
    out["averaged"] = {"n_forks": len(avg_rows), "K": K, "dropped": dropped,
                       "capped_rate": {
                           b: float(np.mean([r["d"]["capped_k"][b] for r in avg_rows]) / max(1, K))
                           for b in BRANCHES} if avg_rows else {}}

    # the averaged pair table: all three pairs of every sampled fork
    pa, pb, pm_a, pm_b, pf, pt, pV = [], [], [], [], [], [], []
    ps_a, ps_b = [], []       # split-half half-labels
    banked_a, banked_b = [], []
    for f, r in enumerate(avg_rows):
        d = r["d"]
        for x in range(3):
            for y in range(x + 1, 3):
                bx, by = BRANCHES[x], BRANCHES[y]
                pa.append(r["g"][bx]); pb.append(r["g"][by])
                pm_a.append(d["mean"][bx]); pm_b.append(d["mean"][by])
                sa = np.asarray(d["scores"][bx]); sb = np.asarray(d["scores"][by])
                ps_a.append(sa); ps_b.append(sb)
                pf.append(f); pt.append("|".join(sorted((bx, by))))
                # 🚨 ddof=1. `np.var(ddof=0)` estimates p(1-p)(K-1)/K, so Var(K-mean) is
                # s2_unbiased / K, NOT s2_biased / K — the biased form under-states the label
                # noise by (K-1)/K and inflates every recovered true gap by the same factor.
                pV.append((np.var(sa, ddof=1), np.var(sb, ddof=1)))
                banked_a.append(d["banked_outcome"][bx]); banked_b.append(d["banked_outcome"][by])
    pa = np.asarray(pa); pb = np.asarray(pb)
    pm_a = np.asarray(pm_a); pm_b = np.asarray(pm_b)
    pf = np.asarray(pf); pt = np.asarray(pt)
    ps_a = np.asarray(ps_a); ps_b = np.asarray(ps_b)
    banked_a = np.asarray(banked_a); banked_b = np.asarray(banked_b)

    # scorer values at those GLOBAL successor indices (the hand scorers are already global; the
    # head / fit values live on BRANCH ROWS, so they are mapped back through `tab["idx"]`)
    def vals_at(nm, gidx):
        if nm in HAND_NAMES:
            return hand[nm][gidx]
        v = np.full(len(gidx), np.nan)
        for i, g in enumerate(gidx):
            k = g_of.get(int(g))
            if k is not None:
                v[i] = V[nm][k]
        return v

    Va = {nm: vals_at(nm, pa) for nm in V}
    Vb = {nm: vals_at(nm, pb) for nm in V}
    ok = np.ones(len(pa), bool)
    for nm in V:
        ok &= np.isfinite(Va[nm]) & np.isfinite(Vb[nm])
    out["averaged"]["n_pairs"] = int(len(pa))
    out["averaged"]["n_pairs_scorable"] = int(ok.sum())

    avg = {}
    for col in ["POOLED"] + sorted(set(pt)):
        sel = ok if col == "POOLED" else (ok & (pt == col))
        nt = sel & (pm_a != pm_b)
        y = (pm_a[nt] > pm_b[nt]).astype(np.float64)
        pfnt = pf[nt]
        cell = {"n_pairs": int(sel.sum()), "n_nontied": int(nt.sum()), "heads": {}}
        for nm in V:
            d = Va[nm][nt] - Vb[nm][nt]
            agree = np.where(d > 0, y, np.where(d < 0, 1.0 - y, 0.5))
            acc, ci = _acc_ci(agree, pfnt, boot_ci)
            cell["heads"][nm] = {"acc": acc, "ci": ci}
        # --- CEILING 1: SPLIT-HALF self-agreement, at two dice budgets ------------------------
        # Agreement between two INDEPENDENT half-labels is `A = q^2 + (1-q)^2` where q is the
        # probability ONE half orders the pair the true way — so the accuracy an ORACLE (a scorer
        # that knew both true win probabilities exactly) would read against a label of that size
        # is `q = (1 + sqrt(2A - 1)) / 2`. Assumption-light: the two halves are independent given
        # the truth BY CONSTRUCTION (separate dice), which is the whole content of the step.
        def _split(ka, kb):
            da = np.asarray([a[ka].mean() - b[ka].mean() for a, b in
                             zip(ps_a[nt], ps_b[nt])])
            db = np.asarray([a[kb].mean() - b[kb].mean() for a, b in
                             zip(ps_a[nt], ps_b[nt])])
            sh = np.where((da == 0) | (db == 0), 0.5, ((da > 0) == (db > 0)).astype(float))
            acc, ci = _acc_ci(sh, pfnt, boot_ci)
            q = (1.0 + np.sqrt(max(0.0, 2.0 * acc - 1.0))) / 2.0
            qci = [(1.0 + np.sqrt(max(0.0, 2.0 * c - 1.0))) / 2.0 for c in ci]
            return {"agreement": acc, "agreement_ci": ci, "oracle_acc": float(q),
                    "oracle_acc_ci": qci}
        half = K // 2
        quarter = max(1, K // 4)
        cell["split_half_ceiling"] = _split(slice(0, half), slice(half, K))
        cell["split_quarter_ceiling"] = _split(slice(0, quarter), slice(quarter, 2 * quarter))
        cell["split_half_ceiling"]["K_per_half"] = half
        cell["split_quarter_ceiling"]["K_per_half"] = quarter

        # --- 🚨 THE MATCHED-NOISE CEILING, fully model-free ------------------------------------
        # ONE label — the mean of re-rolls [half:K] — and TWO scorers read against it on the SAME
        # pairs: the head, and `ROLLOUT_Khalf`, an INDEPENDENT mean of re-rolls [0:half] of the
        # very same states. `ROLLOUT_Khalf` is an unbiased estimate of the true successor value, so
        # whatever it reads is what a value function of THAT quality reads on THIS label. Nothing
        # is assumed about tie rates, gap distributions or the shape of anything — the comparison
        # is two scorers, one label, one pair set (memory: matched-NOISE control before any
        # peer-vs-peer agreement claim).
        lab_a = np.asarray([a[half:K].mean() for a in ps_a])
        lab_b = np.asarray([b[half:K].mean() for b in ps_b])
        hnt = sel & (lab_a != lab_b)
        yh = (lab_a[hnt] > lab_b[hnt]).astype(np.float64)
        cell["half_label"] = {"K_label": K - half, "K_scorer": half,
                              "n_nontied": int(hnt.sum()), "heads": {}}
        probe = {"ROLLOUT_Khalf": (np.asarray([a[:half].mean() for a in ps_a]),
                                   np.asarray([b[:half].mean() for b in ps_b]))}
        for nm, (Ea, Eb) in probe.items():
            dd = Ea[hnt] - Eb[hnt]
            agr = np.where(dd > 0, yh, np.where(dd < 0, 1.0 - yh, 0.5))
            a_, c_ = _acc_ci(agr, pf[hnt], boot_ci)
            cell["half_label"]["heads"][nm] = {"acc": a_, "ci": c_}
        for nm in V:
            dd = Va[nm][hnt] - Vb[nm][hnt]
            agr = np.where(dd > 0, yh, np.where(dd < 0, 1.0 - yh, 0.5))
            a_, c_ = _acc_ci(agr, pf[hnt], boot_ci)
            cell["half_label"]["heads"][nm] = {"acc": a_, "ci": c_}
        # the paired delta that decides BAR 3: the rollout probe MINUS the head, same pairs
        ra = probe["ROLLOUT_Khalf"][0][hnt] - probe["ROLLOUT_Khalf"][1][hnt]
        ha = Va["headW"][hnt] - Vb["headW"][hnt]
        g_r = np.where(ra > 0, yh, np.where(ra < 0, 1.0 - yh, 0.5))
        g_h = np.where(ha > 0, yh, np.where(ha < 0, 1.0 - yh, 0.5))

        def _dfn(fork_ids, _a=g_r, _b=g_h, _pf=pf[hnt]):
            m = np.isin(_pf, fork_ids)
            return float(_a[m].mean() - _b[m].mean()) if m.sum() else None
        lo_, hi_ = boot_ci(_dfn, pf[hnt], seed=11)
        cell["half_label"]["rollout_minus_head"] = {
            "delta": float(g_r.mean() - g_h.mean()), "ci": [lo_, hi_],
            "detected": bool(lo_ > 0 or hi_ < 0)}
        # --- the single-vs-averaged ORDER DISAGREEMENT rate ------------------------------------
        bnt = nt & (banked_a != banked_b) & np.isin(banked_a, ["win", "loss"]) \
            & np.isin(banked_b, ["win", "loss"])
        if bnt.sum():
            single_order = (banked_a[bnt] == "win").astype(float)
            avg_order = (pm_a[bnt] > pm_b[bnt]).astype(float)
            cell["order_agreement_single_vs_avg"] = {
                "n": int(bnt.sum()), "agree": float((single_order == avg_order).mean())}
        # --- CEILING 2: the moment-estimated true-gap sd + ACC_oracle --------------------------
        va = np.asarray([v[0] for v in pV])[nt]
        vb = np.asarray([v[1] for v in pV])[nt]
        obs_var = float(np.var(pm_a[nt] - pm_b[nt], ddof=0))
        noise_var = float(np.mean(va + vb) / K)     # Var(K-mean) = UNBIASED per-draw var / K
        gap_var = max(0.0, obs_var - noise_var)
        gap_sd = float(np.sqrt(gap_var))
        cell["gap"] = {"observed_var": obs_var, "label_noise_var": noise_var,
                       "true_gap_var": gap_var, "true_gap_sd": gap_sd,
                       "E_abs_gap": float(gap_sd * np.sqrt(2 / np.pi)), "acc_oracle": {}}
        for kk in sorted({1, quarter, half, K}):
            a, ntr = acc_oracle(gap_sd, kk)
            cell["gap"]["acc_oracle"][f"K{kk}"] = {"acc": a, "nontied_rate": ntr}
        # --- THE MATCHED SINGLE-LABEL READ, on the SAME forks and the SAME pairs ---------------
        # 🚨 The averaged-label population is NOT the single-label population: a pair that is TIED
        # under two single rollouts can be non-tied under two K-means and vice versa (here 26 % of
        # pairs are non-tied on single labels against ~57 % on K-means). "Averaging raised it" is
        # therefore only a statement when BOTH numbers are read on the same forks, so the banked
        # single label is re-read here on exactly this subset.
        snt = sel & (banked_a != banked_b) & np.isin(banked_a, ["win", "loss"]) \
            & np.isin(banked_b, ["win", "loss"])
        ys = (banked_a[snt] == "win").astype(np.float64)
        cell["single_on_subset"] = {"n_nontied": int(snt.sum()), "heads": {}}
        # 🚨 THE MODEL-FREE CEILING. The K-rollout MEAN, used as a SCORER, read against the BANKED
        # single-rollout label. Its dice are independent of the banked label's (the K re-roll seeds
        # never reproduce the realized stream), so there is no shared-noise inflation; and it is a
        # direct, assumption-free measurement of what an ESTIMATOR OF THE TRUE STATE VALUE reads on
        # exactly the metric the campaign publishes. It is the number arm W's 0.545 is compared
        # against — not 0.60, and not 1.0.
        extra = {"ORACLE_Kfull": (pm_a, pm_b),
                 "ORACLE_Khalf": (np.asarray([a[:half].mean() for a in ps_a]),
                                  np.asarray([b[:half].mean() for b in ps_b]))}
        for nm, (Ea, Eb) in extra.items():
            dd = Ea[snt] - Eb[snt]
            agr = np.where(dd > 0, ys, np.where(dd < 0, 1.0 - ys, 0.5))
            a_, c_ = _acc_ci(agr, pf[snt], boot_ci)
            cell["single_on_subset"]["heads"][nm] = {"acc": a_, "ci": c_}
        for nm in V:
            dd = Va[nm][snt] - Vb[nm][snt]
            agr = np.where(dd > 0, ys, np.where(dd < 0, 1.0 - ys, 0.5))
            a_, c_ = _acc_ci(agr, pf[snt], boot_ci)
            cell["single_on_subset"]["heads"][nm] = {"acc": a_, "ci": c_}
        avg[col] = cell
        print(f"[avg]    {col:12s} n={cell['n_pairs']:6d} nt={cell['n_nontied']:5d} "
              f"A{half}={cell['split_half_ceiling']['agreement']:.4f}->q={cell['split_half_ceiling']['oracle_acc']:.4f} "
              f"A{quarter}={cell['split_quarter_ceiling']['agreement']:.4f}->q={cell['split_quarter_ceiling']['oracle_acc']:.4f} "
              f"gap_sd={gap_sd:.4f} "
              f"model_q{quarter}={cell['gap']['acc_oracle'][f'K{quarter}']['acc']:.4f} "
              f"model_q{half}={cell['gap']['acc_oracle'][f'K{half}']['acc']:.4f} "
              f"model_q1={cell['gap']['acc_oracle']['K1']['acc']:.4f} "
              f"model_q{K}={cell['gap']['acc_oracle'][f'K{K}']['acc']:.4f}  " +
              "  ".join(f"{k}={v['acc']:.4f}" for k, v in cell["heads"].items()), flush=True)
    out["averaged_label"] = avg

    # --- what the top-2 gap is WORTH (P12): mean K-label of top1 minus top2 -------------------
    if avg_rows:
        m1 = np.asarray([r["d"]["mean"]["top1"] for r in avg_rows])
        m2 = np.asarray([r["d"]["mean"]["top2"] for r in avg_rows])
        mr = np.asarray([r["d"]["mean"]["rand"] for r in avg_rows])
        ids = np.arange(len(avg_rows))
        out["policy_ordering_value"] = {
            "mean_top1": float(m1.mean()), "mean_top2": float(m2.mean()),
            "mean_rand": float(mr.mean()),
            "top1_minus_top2": dict(zip(("v", "ci"),
                                        _acc_ci(m1 - m2, ids, boot_ci))),
            "top1_minus_rand": dict(zip(("v", "ci"), _acc_ci(m1 - mr, ids, boot_ci))),
        }
        print(f"[value] K-label mean top1 {m1.mean():.4f} top2 {m2.mean():.4f} "
              f"rand {mr.mean():.4f}", flush=True)

    json.dump(out, open(args.out, "w"), indent=1)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
