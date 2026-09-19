"""JOB 1's READ — the K-CURVE: K-rollout leaves against ONE K'=8 label, beside the head, beside
the label's own ceiling, beside the cost.

    python3 score_kcurve.py --forks <banked forks> --snapshot <zip> \
        --banked <reroll dir> --ext <rerollx dir> [--hand <hand dir>] --out kcurve.json

NOTHING HERE RE-IMPLEMENTS THE METRIC. `refit.load_forks` / `boot_ci` and
`fit_heads.forward_feats` are imported BY PATH from the 2026-09-14 and 2026-09-18 records,
unmodified, and `score_controls.acc_oracle` (the parametric cross-check) from the 2026-09-19 one.
The pair rule is the record's: all pairs within a fork, "non-tied" means the two LABELS differ,
the scorer is `sign(V_a - V_b)` with 0.5 on an exact V tie, every CI bootstraps over FORKS.

THE DESIGN, in one line: **one label, six scorers, one pair set.**

* **THE LABEL** is the BANKED mean of re-rolls 0-7 — K' = 8, the same label for every K, so the
  curve's levels are comparable to each other and to the head (whose numbers on this label are
  published in `leaf_ceiling_controls_2026-09-19` §5.1).
* **THE LEAVES** are means of the FRESH re-rolls 8..8+K-1 for K in {1,2,4,8,16}. The dice are
  independent of the label's, so `ROLLOUT_K` is an unbiased estimate of the same successor's true
  value and whatever it reads is what a value function OF THAT QUALITY reads on THIS label.
* **NESTED across K**, declared: a smaller K is a strict sub-sample of a larger one's dice, so the
  levels are positively CORRELATED. That is the right design for a dose curve (the K-to-K step is
  paired and carries no between-K dice noise) and the wrong one for treating two levels as
  independent samples.

THE HEAD is read at the GLOBAL SUCCESSOR index (`V0`), not through `build_table`'s branch rows:
`build_table` drops a branch whose BANKED single-rollout outcome was capped, and this read does not
use the banked outcome at all. The consequence is a slightly LARGER pair set than §5.1's, which is
declared and printed; every delta is on identical pairs regardless.

THREE CEILINGS, ranked by what they assume (PREDICTION.md §2.2):
 1. MEASURED, model-free — the K = 16 level itself. A LOWER BOUND on what a perfect scorer reads
    against this label, because 16 rollouts are still noisy.
 2. SPLIT-HALF of the LABEL (re-rolls 0-3 vs 4-7) -> `q = (1+sqrt(2A-1))/2`. CROSS-CHECK ONLY:
    biased UP by Jensen over heterogeneous gaps, DOWN by scoring an exact half-label tie as 0.5.
 3. PARAMETRIC `ACC_oracle(gap_sd, K'=8)` from the moment-recovered gap (ddof=1). CROSS-CHECK ONLY.

THE COST COLUMN is sim TURNS, not seconds: a rollout runs from the fork to a terminal, so its cost
is `turns_at_terminal - fork_turn` and that number is load-free. Wall-clock seconds per rollout are
reported beside it WITH the box's load, because they are not.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

BRANCHES = ("top1", "top2", "rand")
KS = (1, 2, 4, 8, 16)
_MEAS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("paired_refit_discrimination_2026-09-14", "offline_leaf_fit_2026-09-18",
           "leaf_ceiling_controls_2026-09-19"):
    sys.path.insert(0, os.path.join(_MEAS, _d))


def _acc_ci(agree, pf_nt, boot_ci, seed=7):
    def acc_for(fork_ids):
        m = np.isin(pf_nt, fork_ids)
        return float(agree[m].mean()) if m.sum() else None
    lo, hi = boot_ci(acc_for, pf_nt, seed=seed)
    return float(agree.mean()) if len(agree) else float("nan"), [lo, hi]


def _agree(dv, y):
    return np.where(dv > 0, y, np.where(dv < 0, 1.0 - y, 0.5))


def _paired(ga, gb, pf, boot_ci, seed=11):
    def dfn(fork_ids):
        m = np.isin(pf, fork_ids)
        return float(ga[m].mean() - gb[m].mean()) if m.sum() else None
    lo, hi = boot_ci(dfn, pf, seed=seed)
    return {"delta": float(ga.mean() - gb.mean()), "ci": [lo, hi],
            "detected": bool(lo > 0 or hi < 0)}


def load_turn_by_succ(hand_dir, forks_dir):
    """POST-HOC (hazard 2 of the control): the successor's TURN, so the `top1|top2` column can be
    split by DEPTH. Re-used from the control's already-captured boards — no new battle."""
    paths = sorted(glob.glob(os.path.join(forks_dir, "forks_s*.jsonl")))
    off, total = {}, 0
    for rp in paths:
        tg = os.path.basename(rp)[len("forks_"):-len(".jsonl")]
        off[tg] = total
        total += len(np.load(os.path.join(forks_dir, f"succ_{tg}.npy"), mmap_mode="r"))
    turn = np.full(total, np.nan)
    for hp in sorted(glob.glob(os.path.join(hand_dir, "hand_s*.jsonl"))):
        for line in open(hp):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                break
            if d["shard"] not in off or not d.get("obs_match"):
                continue
            turn[off[d["shard"]] + int(d["succ"])] = d["board"]["turn"]
    return turn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forks", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--banked", required=True, help="reroll_s*.jsonl — the K'=8 LABEL")
    ap.add_argument("--ext", required=True, help="rerollx_s*.jsonl — the FRESH leaf dice")
    ap.add_argument("--hand", default="", help="the control's captured boards, for the depth split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)
    import refit
    from refit import boot_ci
    from forks import load_model
    from fit_heads import forward_feats
    from score_controls import acc_oracle

    out = {"inputs": vars(args)}

    rows, succ, sm = refit.load_forks(args.forks)
    model = load_model(args.snapshot)
    _pooled, _pre, V0 = forward_feats(model, succ, sm, want_prepool=False)
    out["dataset"] = {"n_rows": len(rows), "n_successors": int(len(succ))}

    # the GLOBAL successor offset, the same one `load_forks` applies
    paths = sorted(glob.glob(os.path.join(args.forks, "forks_s*.jsonl")))
    off, total = {}, 0
    for rp in paths:
        tg = os.path.basename(rp)[len("forks_"):-len(".jsonl")]
        off[tg] = total
        total += len(np.load(os.path.join(args.forks, f"succ_{tg}.npy"), mmap_mode="r"))
    assert total == len(succ), f"successor count {total} != {len(succ)}"

    def read(d, pat, key):
        o = {}
        for p in sorted(glob.glob(os.path.join(d, pat))):
            for line in open(p):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    break
                o[(r["shard"], int(r["fork_line"]))] = r
        return o

    lab = read(args.banked, "reroll_s*.jsonl", "banked")
    ext = read(args.ext, "rerollx_s*.jsonl", "ext")
    join = {"label_forks": len(lab), "ext_forks": len(ext), "matched": 0,
            "salt_mismatch": 0, "succ_mismatch": 0, "short_label": 0, "short_ext": 0,
            "verify_ok": 0, "verify_bad": 0, "kmax": 0}
    K_LAB, K_EXT = 8, max(KS)
    forks = []
    for key in sorted(ext):
        e, b = ext[key], lab.get(key)
        if b is None:
            continue
        if e["seed_salt"] != b["seed_salt"]:
            join["salt_mismatch"] += 1
            continue
        if any(int(e["succ"][x]) != int(b["succ"][x]) for x in BRANCHES):
            join["succ_mismatch"] += 1
            continue
        if any(len(b["scores"][x]) < K_LAB for x in BRANCHES):
            join["short_label"] += 1
            continue
        if any(len(e["scores"][x]) < K_EXT for x in BRANCHES):
            join["short_ext"] += 1
            continue
        if e.get("verify"):
            good = all(abs(v[0] - v[1]) < 1e-9 for v in e["verify"].values())
            join["verify_ok" if good else "verify_bad"] += 1
        forks.append((key, b, e))
        join["matched"] += 1
    if not forks:
        raise SystemExit("REFUSED: no fork joined the label to the fresh dice")
    out["join"] = join

    # --------------------------------------------------------------- the pair table
    pf, pt, pa_g, pb_g = [], [], [], []
    LA, LB = [], []                     # the K'=8 LABEL means
    HA, HB = [], []                     # the label's two independent HALVES (ceiling 2)
    RA = {k: [] for k in KS}            # the leaf at each K
    RB = {k: [] for k in KS}
    VA_lab, VB_lab = [], []             # per-draw label variances (ceiling 3, ddof=1)
    cost_turns, cost_turns_ext = [], []
    for f, (key, b, e) in enumerate(forks):
        gl = {x: off[key[0]] + int(b["succ"][x]) for x in BRANCHES}
        sb = {x: np.asarray(b["scores"][x][:K_LAB], float) for x in BRANCHES}
        se = {x: np.asarray(e["scores"][x][:K_EXT], float) for x in BRANCHES}
        for x in BRANCHES:
            cost_turns.append(float(np.mean(b["mean_turns"][x])) - float(b["turn"]))
            cost_turns_ext.append(float(np.mean(e["mean_turns"][x])) - float(e["turn"]))
        for i in range(3):
            for j in range(i + 1, 3):
                xi, xj = BRANCHES[i], BRANCHES[j]
                pf.append(f)
                pt.append("|".join(sorted((xi, xj))))
                pa_g.append(gl[xi]); pb_g.append(gl[xj])
                LA.append(sb[xi].mean()); LB.append(sb[xj].mean())
                HA.append((sb[xi][:4].mean(), sb[xi][4:].mean()))
                HB.append((sb[xj][:4].mean(), sb[xj][4:].mean()))
                VA_lab.append(np.var(sb[xi], ddof=1)); VB_lab.append(np.var(sb[xj], ddof=1))
                for k in KS:
                    RA[k].append(se[xi][:k].mean()); RB[k].append(se[xj][:k].mean())
    pf = np.asarray(pf); pt = np.asarray(pt)
    pa_g = np.asarray(pa_g); pb_g = np.asarray(pb_g)
    LA = np.asarray(LA); LB = np.asarray(LB)
    HA = np.asarray(HA); HB = np.asarray(HB)
    VA_lab = np.asarray(VA_lab); VB_lab = np.asarray(VB_lab)
    RA = {k: np.asarray(v) for k, v in RA.items()}
    RB = {k: np.asarray(v) for k, v in RB.items()}
    HEAD_A, HEAD_B = V0[pa_g], V0[pb_g]

    out["cost"] = {
        "mean_remaining_turns_per_rollout_label": float(np.mean(cost_turns)),
        "mean_remaining_turns_per_rollout_fresh": float(np.mean(cost_turns_ext)),
        "median_remaining_turns_fresh": float(np.median(cost_turns_ext)),
        "n_branch_observations": int(len(cost_turns_ext)),
    }
    for d, nm in ((args.ext, "ext"), (args.banked, "label")):
        metas = []
        for mp in sorted(glob.glob(os.path.join(d, "reroll*meta_s*.json"))):
            metas.append(json.load(open(mp)))
        if metas:
            roll = sum(m["stats"]["rollouts"] for m in metas)
            wall = max(m["wall_s"] for m in metas)
            out["cost"][f"{nm}_rollouts"] = roll
            out["cost"][f"{nm}_wall_s_max_shard"] = wall
            out["cost"][f"{nm}_s_per_rollout_per_worker"] = float(
                sum(m["wall_s"] for m in metas) / max(1, roll))
            out["cost"][f"{nm}_capped"] = sum(m["stats"]["capped"] for m in metas)
            out["cost"][f"{nm}_errors"] = sum(m["stats"]["errors"] for m in metas)

    turn_of = load_turn_by_succ(args.hand, args.forks) if args.hand else None

    # --------------------------------------------------------------- the read
    curve = {}
    for col in ["POOLED"] + sorted(set(pt)):
        sel = np.ones(len(pf), bool) if col == "POOLED" else (pt == col)
        nt = sel & (LA != LB)
        y = (LA[nt] > LB[nt]).astype(np.float64)
        pfn = pf[nt]
        cell = {"n_pairs": int(sel.sum()), "n_nontied": int(nt.sum()),
                "n_forks": int(len(np.unique(pfn))), "scorers": {}}
        g_head = _agree(HEAD_A[nt] - HEAD_B[nt], y)
        a, ci = _acc_ci(g_head, pfn, boot_ci)
        cell["scorers"]["headW"] = {"acc": a, "ci": ci, "K": 0, "rollouts": 0}
        for k in KS:
            g = _agree(RA[k][nt] - RB[k][nt], y)
            a_, c_ = _acc_ci(g, pfn, boot_ci)
            cell["scorers"][f"ROLLOUT_{k}"] = {
                "acc": a_, "ci": c_, "K": k, "rollouts": k,
                "delta_vs_head": _paired(g, g_head, pfn, boot_ci),
                "clears_head_point": bool(c_[0] > cell["scorers"]["headW"]["acc"]),
                "clears_060": bool(c_[0] > 0.60),
                "sim_turns_per_leaf_eval": k * out["cost"]["mean_remaining_turns_per_rollout_fresh"],
            }
        # --- BAR K1 / K2 ---------------------------------------------------------------
        cell["bar_K1_smallest_K_clearing_head_point"] = next(
            (k for k in KS if cell["scorers"][f"ROLLOUT_{k}"]["clears_head_point"]), None)
        cell["bar_K2_smallest_K_clearing_060"] = next(
            (k for k in KS if cell["scorers"][f"ROLLOUT_{k}"]["clears_060"]), None)
        # --- BAR K3/K4: the marginal table and the knee ---------------------------------
        acc = {k: cell["scorers"][f"ROLLOUT_{k}"]["acc"] for k in KS}
        ah = cell["scorers"]["headW"]["acc"]
        cell["marginal"] = []
        knee = None
        for k in KS[:-1]:
            k2 = 2 * k
            step = acc[k2] - acc[k]
            cell["marginal"].append({
                "K": k, "to": k2, "d_acc": step, "d_acc_per_rollout": step / k,
                "avg_gain_per_rollout_so_far": (acc[k] - ah) / k,
                "marginal_below_average": bool(step < (acc[k] - ah))})
            if knee is None and step < (acc[k] - ah):
                knee = k
        cell["bar_K4_knee"] = knee
        # --- CEILING 2: SPLIT-HALF of the LABEL (two K=4 halves) ------------------------
        da = HA[nt, 0] - HB[nt, 0]
        db = HA[nt, 1] - HB[nt, 1]
        sh = np.where((da == 0) | (db == 0), 0.5, ((da > 0) == (db > 0)).astype(float))
        A, Aci = _acc_ci(sh, pfn, boot_ci)
        q = (1.0 + np.sqrt(max(0.0, 2.0 * A - 1.0))) / 2.0
        cell["ceiling_split_half_of_label"] = {
            "K_per_half": 4, "agreement": A, "agreement_ci": Aci, "implied_oracle_acc": float(q),
            "implied_oracle_acc_ci": [(1.0 + np.sqrt(max(0.0, 2.0 * c - 1.0))) / 2.0 for c in Aci],
            "note": "CROSS-CHECK ONLY (biased up by Jensen, down by tie-scoring)"}
        # --- CEILING 3: the moment-recovered gap + parametric oracle at K'=8 ------------
        obs_var = float(np.var(LA[nt] - LB[nt], ddof=0))
        noise_var = float(np.mean(VA_lab[nt] + VB_lab[nt]) / K_LAB)
        gap_var = max(0.0, obs_var - noise_var)
        gap_sd = float(np.sqrt(gap_var))
        o8, nt8 = acc_oracle(gap_sd, K_LAB)
        o1, _ = acc_oracle(gap_sd, 1)
        cell["ceiling_parametric"] = {
            "observed_var": obs_var, "label_noise_var": noise_var,
            "label_noise_share": float(noise_var / obs_var) if obs_var else None,
            "true_gap_sd": gap_sd, "E_abs_gap": float(gap_sd * np.sqrt(2 / np.pi)),
            "acc_oracle_at_K8": o8, "acc_oracle_at_K1": o1,
            "note": "CROSS-CHECK ONLY (over-read the measured 8-rollout number by ~0.06 on "
                    "2026-09-19)"}
        # --- CEILING 1 (the headline) ---------------------------------------------------
        cell["ceiling_measured_lower_bound"] = {
            "scorer": "ROLLOUT_16", "acc": acc[16],
            "ci": cell["scorers"]["ROLLOUT_16"]["ci"],
            "note": "model-free LOWER BOUND on what a perfect scorer reads against this label"}
        # --- POST-HOC: the DEPTH split (hazard 2 of the control) ------------------------
        if turn_of is not None:
            ta, tb = turn_of[pa_g], turn_of[pb_g]
            same = np.isfinite(ta) & np.isfinite(tb) & (ta == tb)
            cell["depth_split"] = {}
            for lb, m in (("same_turn", same), ("diff_turn", ~same)):
                s2 = sel & m & (LA != LB)
                if s2.sum() < 50:
                    continue
                y2 = (LA[s2] > LB[s2]).astype(np.float64)
                blk = {"n_nontied": int(s2.sum()), "scorers": {}}
                gh2 = _agree(HEAD_A[s2] - HEAD_B[s2], y2)
                a2, c2 = _acc_ci(gh2, pf[s2], boot_ci)
                blk["scorers"]["headW"] = {"acc": a2, "ci": c2}
                for k in KS:
                    g2 = _agree(RA[k][s2] - RB[k][s2], y2)
                    a3, c3 = _acc_ci(g2, pf[s2], boot_ci)
                    blk["scorers"][f"ROLLOUT_{k}"] = {
                        "acc": a3, "ci": c3,
                        "delta_vs_head": _paired(g2, gh2, pf[s2], boot_ci)}
                cell["depth_split"][lb] = blk
        curve[col] = cell
        print(f"[{col:12s}] n={cell['n_pairs']:5d} nt={cell['n_nontied']:5d} "
              f"head={cell['scorers']['headW']['acc']:.4f}  " +
              "  ".join(f"K{k}={cell['scorers'][f'ROLLOUT_{k}']['acc']:.4f}" for k in KS) +
              f"  ceilA={A:.4f}->q={q:.4f} gap_sd={gap_sd:.4f} "
              f"K1bar={cell['bar_K1_smallest_K_clearing_head_point']} "
              f"K2bar={cell['bar_K2_smallest_K_clearing_060']} knee={knee}", flush=True)
    out["curve"] = curve

    # what the policy's own preference is worth, on the K'=8 label (the control's §5.5, re-read on
    # exactly the forks this curve uses)
    m = {x: np.asarray([np.mean(b["scores"][x][:K_LAB]) for _k, b, _e in forks]) for x in BRANCHES}
    ids = np.arange(len(forks))
    out["policy_ordering_value"] = {
        f"mean_{x}": float(m[x].mean()) for x in BRANCHES}
    for nm, d in (("top1_minus_top2", m["top1"] - m["top2"]),
                  ("top1_minus_rand", m["top1"] - m["rand"])):
        v, ci = _acc_ci(d, ids, boot_ci)
        out["policy_ordering_value"][nm] = {"v": v, "ci": ci}

    json.dump(out, open(args.out, "w"), indent=1)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
