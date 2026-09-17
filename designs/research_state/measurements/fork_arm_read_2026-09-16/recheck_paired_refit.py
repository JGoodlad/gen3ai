"""RECHECK — re-score the 2026-09-14 paired refit with the ORIGINAL head correctly indexed.

    python3 recheck_paired_refit.py --forks <dir> --snapshot <zip> --heads <dir> --out <json>

THE DEFECT. `paired_refit_discrimination_2026-09-14/refit.py` builds its branch table with

    b_idx.append(br["succ"])            # an index into the SUCCESSOR array
    tab["V0"] = V0[b_idx]               # correct: one entry per BRANCH ROW

and then evaluates the heads with

    V["control_bce"] = head_V(ctrl, tab["pooled"])   # per BRANCH ROW  -- correct
    V["original"]    = V0                            # per SUCCESSOR   -- MIS-INDEXED

`read_head` addresses `V` by BRANCH-ROW index (`V[pairs["a"]]`). The two index spaces coincide only
while every branch that HAS a successor is kept; a branch that has one but is dropped (stall-capped,
or a non-binary outcome) shifts every later row. On the banked dataset that first happens at branch
row 1,880 and reaches a maximum offset of 23.

Only the ORIGINAL head is affected — which is the one the registered baseline (0.5169, "ranks
siblings at chance") comes from, and the one every delta in that record is taken against.

This script recomputes the identical held-out split (the same seed, the same battles, the same 562
non-tied test pairs) both ways and reports what the record's three headline rows become. It does
NOT edit the record: a record is what was believed at the time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REC = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                   "paired_refit_discrimination_2026-09-14")
sys.path.insert(0, os.path.abspath(REC))


def published_split(tab, pairs, seed=20260914):
    """The 2026-09-14 split, reproduced operation for operation."""
    rng = np.random.default_rng(seed)
    battles = np.unique(tab["battle"])
    rng.shuffle(battles)
    n_test = max(1, int(round(0.20 * len(battles))))
    n_val = max(1, int(round(0.15 * len(battles))))
    test_b = set(battles[:n_test])
    is_test = np.array([b in test_b for b in tab["battle"]])
    te = set(np.flatnonzero(is_test).tolist())
    p_te = np.flatnonzero(np.array([(a in te and b in te)
                                    for a, b in zip(pairs["a"], pairs["b"])]))
    return np.flatnonzero(is_test), p_te


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--forks", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--heads", required=True, help="the record's saved head_*.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)
    from refit import load_forks, build_table, read_head, paired_delta, head_V
    from forks import load_model, score_batch
    from agents.model.aux_value_heads import WinProbHead

    rows, succ, smask = load_forks(args.forks)
    model = load_model(args.snapshot)
    pooled = np.empty((len(succ), 128), np.float32)
    V0 = np.empty(len(succ))
    for i in range(0, len(succ), 256):
        j = min(len(succ), i + 256)
        pooled[i:j], V0[i:j] = score_batch(model, succ[i:j], smask[i:j])
    tab, pairs = build_table(rows, pooled, V0)

    b_idx = np.empty(len(tab["y"]), int)
    k = 0
    for r in rows:
        for nm in ("top1", "top2", "rand"):
            br = r["branches"][nm]
            if br["succ"] is None or br["capped"] or br["outcome"] not in ("win", "loss"):
                continue
            b_idx[k] = br["succ"]
            k += 1
    assert k == len(b_idx)
    ar = np.arange(len(b_idx))
    mis = b_idx != ar
    align = {"n_forks": len(rows), "n_successors": int(len(succ)), "n_branch_rows": int(len(b_idx)),
             "b_idx_equals_arange": bool(not mis.any()),
             "first_mismatch_row": int(np.argmax(mis)) if mis.any() else None,
             "max_offset": int((b_idx - ar).max()), "n_rows_misindexed": int(mis.sum())}
    print("[align]", json.dumps(align), flush=True)

    te_idx, p_te = published_split(tab, pairs)
    V_rowwise = {"original": tab["V0"]}
    V_asbug = {"original": V0}
    for name in ("control_bce", "rank_0.1"):
        p = os.path.join(args.heads, f"head_{name}.pt")
        if not os.path.exists(p):
            continue
        h = WinProbHead()
        h.load_state_dict(torch.load(p, map_location="cpu")["win_head"])
        h.eval()
        V_rowwise[name] = head_V(h, tab["pooled"])

    out = {"alignment": align, "split": {"test_rows": int(len(te_idx)), "test_pairs": int(len(p_te)),
                                         "test_nontied": int((pairs["y"][p_te] != 0.5).sum())},
           "as_published": {}, "corrected": {}, "deltas_corrected": {}, "deltas_as_published": {}}
    out["as_published"]["original"] = read_head(V_asbug["original"], tab, pairs, p_te,
                                                "original(as published, mis-indexed)")
    for k2, v in V_rowwise.items():
        out["corrected"][k2] = read_head(v, tab, pairs, p_te, k2)
    # the record's three headline deltas, recomputed against the CORRECT original
    if "control_bce" in V_rowwise:
        out["deltas_corrected"]["control_minus_original"] = paired_delta(
            V_rowwise["control_bce"], V_rowwise["original"], pairs, p_te)
        out["deltas_as_published"]["control_minus_original"] = paired_delta(
            V_rowwise["control_bce"], V_asbug["original"], pairs, p_te)
    if "rank_0.1" in V_rowwise:
        out["deltas_corrected"]["bestrank_minus_original"] = paired_delta(
            V_rowwise["rank_0.1"], V_rowwise["original"], pairs, p_te)
        out["deltas_corrected"]["bestrank_minus_control"] = paired_delta(
            V_rowwise["rank_0.1"], V_rowwise["control_bce"], pairs, p_te)
    # and the same read over EVERY pair, not just the held-out fifth
    allp = np.arange(len(pairs["y"]))
    out["corrected_all_pairs"] = {k2: read_head(v, tab, pairs, allp, k2)
                                  for k2, v in V_rowwise.items()}
    json.dump(out, open(args.out, "w"), indent=1)
    for k2 in ("as_published", "corrected"):
        for lbl, r in out[k2].items():
            print(f"  {k2:14s} {lbl:22s} pairwise {r['pairwise_acc']:.4f} "
                  f"[{r['pairwise_acc_ci'][0]:.4f}, {r['pairwise_acc_ci'][1]:.4f}] "
                  f"sep {r['sep_ratio']:.3f} ECE {r['ece']:.4f} Brier {r['brier']:.4f}", flush=True)
    for lbl, d in out["deltas_corrected"].items():
        print(f"  delta CORRECTED {lbl:28s} {d['delta']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}] "
              f"{'DETECTED' if d['detected'] else 'NOT DETECTED'}", flush=True)
    print(f"[recheck] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
