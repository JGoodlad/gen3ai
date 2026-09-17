"""SCORE — held-out PAIRWISE ACCURACY of two win-prob heads on two fresh fork datasets.

    python3 score_forks.py --arm-forks <dir> --ctrl-forks <dir> \
        --arm-snapshot <zip> --ctrl-snapshot <zip> --out <json>

Nothing is fitted here. Each dataset is read whole (no train/val/test split — there is nothing to
hold out FROM), and both heads are scored on BOTH datasets, which is what separates the two effects
the registration cares about:

    HEAD effect            two heads, the SAME forks, the SAME pairs -> a PAIRED delta with a
                           bootstrap-over-forks CI
    STATE-DISTRIBUTION     one head, two different fork sets -> two intervals, reported side by
    effect                 side and NEVER differenced with a CI (the forks are not paired)

The metric, the pairing, the exclusions and the bootstrap are
`paired_refit_discrimination_2026-09-14/refit.py`'s, REUSED BY IMPORT rather than copied.

🚨 THE ALIGNMENT ASSERTION. That record's original-head row is mis-indexed (a successor-indexed V
read at branch-row positions; see `recheck_paired_refit.py`). Every `V` this script hands to
`read_head` is built as `V_successor[b_idx]` and then ASSERTED against a freshly recomputed
`b_idx`, so the same defect cannot recur silently.
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

BRANCHES = ("top1", "top2", "rand")


def branch_index(rows):
    """b_idx, recomputed independently of build_table — the assertion's other half."""
    out = []
    for r in rows:
        for nm in BRANCHES:
            br = r["branches"][nm]
            if br["succ"] is None or br["capped"] or br["outcome"] not in ("win", "loss"):
                continue
            out.append(br["succ"])
    return np.asarray(out, int)


def dataset_stats(rows, tab):
    """Everything about the DATASET that does not involve a head."""
    n_cap = n_nosucc = 0
    for r in rows:
        for nm in BRANCHES:
            br = r["branches"][nm]
            n_cap += int(bool(br["capped"]))
            n_nosucc += int(br["succ"] is None)
    bs_n = bs_hit = 0
    for r in rows:
        o = {b: r["branches"][b] for b in BRANCHES}
        if any(o[b]["capped"] or o[b]["outcome"] not in ("win", "loss") for b in BRANCHES):
            continue
        bs_n += 1
        if (o["rand"]["outcome"] == "win" and o["top1"]["outcome"] == "loss"
                and o["top2"]["outcome"] == "loss"):
            bs_hit += 1
    lo, hi = wilson(bs_hit, bs_n)
    return {"n_forks": len(rows), "n_branch_rows": int(len(tab["y"])),
            "capped_branches": n_cap, "no_successor_branches": n_nosucc,
            "branch_win_rate": {b: (float(tab["y"][tab["name"] == b].mean())
                                    if (tab["name"] == b).sum() else None) for b in BRANCHES},
            "branch_n": {b: int((tab["name"] == b).sum()) for b in BRANCHES},
            "blind_spot": {"n_complete_forks": bs_n, "rand_beats_both": bs_hit,
                           "rate": (bs_hit / bs_n) if bs_n else None, "wilson": [lo, hi]},
            "turn_median": float(np.median([r["turn"] for r in rows])),
            "opp_mix": {o: sum(1 for r in rows if r["opp"] == o)
                        for o in sorted({r["opp"] for r in rows})}}


def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / d, (c + h) / d)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-forks", required=True)
    ap.add_argument("--ctrl-forks", required=True)
    ap.add_argument("--arm-snapshot", required=True)
    ap.add_argument("--ctrl-snapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)
    from refit import load_forks, build_table, read_head, paired_delta
    from forks import load_model, score_batch

    models = {"arm": load_model(args.arm_snapshot), "ctrl": load_model(args.ctrl_snapshot)}
    out = {"snapshots": {"arm": args.arm_snapshot, "ctrl": args.ctrl_snapshot},
           "datasets": {}, "reads": {}, "head_deltas": {}}

    for ds, fdir in (("arm_states", args.arm_forks), ("ctrl_states", args.ctrl_forks)):
        print(f"=== dataset {ds} from {fdir}", flush=True)
        rows, succ, smask = load_forks(fdir)
        V_succ = {}
        for who, m in models.items():
            V = np.empty(len(succ))
            for i in range(0, len(succ), 256):
                j = min(len(succ), i + 256)
                _, V[i:j] = score_batch(m, succ[i:j], smask[i:j])
            V_succ[who] = V
        tab, pairs = build_table(rows, np.zeros((len(succ), 128), np.float32), V_succ["arm"])
        # 🚨 THE ALIGNMENT ASSERTION. `b_idx` is recomputed here from the raw rows, entirely
        # independently of `build_table`; `tab["V0"]` is what build_table produced by indexing the
        # arm's successor-space V with ITS OWN b_idx. If the two disagree by even one row, the
        # 2026-09-14 defect is present and the read is refused rather than reported.
        b_idx = branch_index(rows)
        if len(b_idx) != len(tab["y"]):
            raise SystemExit(f"REFUSED: branch table length {len(tab['y'])} != {len(b_idx)}")
        if not np.allclose(tab["V0"], V_succ["arm"][b_idx]):
            raise SystemExit("REFUSED: branch-row / successor index disagreement")
        n_shift = int((b_idx != np.arange(len(b_idx))).sum())
        out.setdefault("alignment", {})[ds] = {
            "n_branch_rows": int(len(b_idx)), "n_successors": int(len(succ)),
            "rows_where_branch_index_differs_from_position": n_shift,
            "max_offset": int((b_idx - np.arange(len(b_idx))).max()),
            "note": "a non-zero count is exactly the condition under which the 2026-09-14 "
                    "record's original-head row is mis-scored"}
        V_row = {who: V[b_idx] for who, V in V_succ.items()}
        out["datasets"][ds] = dataset_stats(rows, tab)
        out["datasets"][ds]["n_pairs"] = int(len(pairs["y"]))
        out["datasets"][ds]["n_nontied_pairs"] = int((pairs["y"] != 0.5).sum())
        out["datasets"][ds]["tie_rate"] = float((pairs["y"] == 0.5).mean())
        allp = np.arange(len(pairs["y"]))
        out["reads"][ds] = {who: read_head(V_row[who], tab, pairs, allp, who) for who in models}
        out["head_deltas"][ds] = paired_delta(V_row["arm"], V_row["ctrl"], pairs, allp)
        for who in models:
            r = out["reads"][ds][who]
            print(f"  {ds:11s} head={who:5s} pairwise {r['pairwise_acc']:.4f} "
                  f"[{r['pairwise_acc_ci'][0]:.4f}, {r['pairwise_acc_ci'][1]:.4f}] "
                  f"n_nontied {r['n_nontied']} sep {r['sep_ratio']:.3f} "
                  f"ECE {r['ece']:.4f} Brier {r['brier']:.4f}", flush=True)
        d = out["head_deltas"][ds]
        print(f"  {ds:11s} PAIRED delta (arm - ctrl) {d['delta']:+.4f} "
              f"[{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}] "
              f"{'DETECTED' if d['detected'] else 'NOT DETECTED'}", flush=True)

    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[score] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
