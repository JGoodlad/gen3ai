"""Pool the shard metas + rows of a fork dataset into the numbers the README's frame table needs.

    python3 fork_stats.py <fork_dir> [--json out.json]

Everything here is a property of the DATASET, not of any head: the contested-selection rate, the
CRN determinism checks (a single failure voids the pairing, so the count is reported whether or not
it is zero), the branch-outcome table, the tie rate the ranking term's zero weight lands on, and
the POLICY's blind-spot rate — how often a uniformly-random legal alternative beats both of the
policy's own top-2 candidates on identical dice.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
from collections import Counter

BRANCHES = ("top1", "top2", "rand")


def wilson(k, n, z=1.96):
    if n <= 0:
        return (None, None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fork_dir")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    metas = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.fork_dir, "meta_*.json")))]
    rows = []
    for p in sorted(glob.glob(os.path.join(args.fork_dir, "forks_s*.jsonl"))):
        for line in open(p):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break

    out = {"n_shards_finished": len(metas), "n_forks": len(rows)}
    if metas:
        out["selection"] = {
            "gap_quantile": metas[0]["gap_quantile"],
            "candidates": sum(m["n_candidates"] for m in metas),
            "selected": sum(m["n_selected"] for m in metas),
            "selection_rate": sum(m["n_selected"] for m in metas) / sum(m["n_candidates"] for m in metas),
            "gap_thresholds": [round(m["gap_threshold"], 4) for m in metas],
            "min_turn": metas[0]["min_turn"], "max_turn": metas[0].get("max_turn"),
            "min_legal": metas[0]["min_legal"], "forks_per_battle": metas[0]["forks_per_battle"]}
        st = Counter()
        for m in metas:
            st.update(m["stats"])
        out["shard_stats"] = dict(st)

    # per-branch outcomes, ties, caps, blind spot
    oc = {b: Counter() for b in BRANCHES}
    turns, opps, battles = Counter(), Counter(), set()
    n_pairs = n_tied = n_nontied = 0
    complete = bs = 0
    for r in rows:
        battles.add(r["base"])
        opps[r["opp"]] += 1
        turns[r["turn"]] += 1
        for b in BRANCHES:
            br = r["branches"][b]
            oc[b][("capped" if br["capped"] else (br["outcome"] or "none"))] += 1
        usable = {b: r["branches"][b] for b in BRANCHES
                  if not r["branches"][b]["capped"] and r["branches"][b]["outcome"] in ("win", "loss")
                  and r["branches"][b]["succ"] is not None}
        for a, b in itertools.combinations(sorted(usable), 2):
            n_pairs += 1
            if usable[a]["outcome"] == usable[b]["outcome"]:
                n_tied += 1
            else:
                n_nontied += 1
        if len(usable) == 3:
            complete += 1
            if (usable["rand"]["outcome"] == "win" and usable["top1"]["outcome"] == "loss"
                    and usable["top2"]["outcome"] == "loss"):
                bs += 1
    out["battles"] = len(battles)
    out["by_opponent"] = dict(opps)
    out["turn_median"] = sorted(turns.elements())[len(list(turns.elements())) // 2] if turns else None
    out["branch_outcomes"] = {b: dict(c) for b, c in oc.items()}
    out["branch_win_rate"] = {
        b: (oc[b]["win"] / (oc[b]["win"] + oc[b]["loss"]) if (oc[b]["win"] + oc[b]["loss"]) else None)
        for b in BRANCHES}
    out["pairs"] = {"total": n_pairs, "tied": n_tied, "non_tied": n_nontied,
                    "non_tied_rate": (n_nontied / n_pairs) if n_pairs else None}
    p, lo, hi = wilson(bs, complete)
    out["blind_spot"] = {"complete_forks": complete, "rand_beats_both": bs,
                         "rate": p, "wilson_ci": [lo, hi]}
    det = out.get("shard_stats")
    if not det:
        # shards still running (or killed by the clock): their metas do not exist yet, so take the
        # counters off the last progress line each shard printed. A missing count must read
        # UNKNOWN, never "VOID" — the two say opposite things about the dataset.
        det = Counter()
        for lg in sorted(glob.glob(os.path.join(args.fork_dir, "logs", "shard_*.log"))):
            last = None
            for line in open(lg, errors="replace"):
                if "det_checks" in line and "{" in line:
                    last = line[line.index("{"):].strip()
            if last:
                try:
                    det.update(json.loads(last.replace("'", '"')))
                except json.JSONDecodeError:
                    pass
        det = dict(det)
        out["shard_stats_from_logs"] = det
    f = det.get("det_fail")
    out["CRN_determinism"] = {
        "checks": det.get("det_checks"), "failures": f,
        "verdict": ("UNKNOWN — no counter found" if f is None else
                    "PAIRING HOLDS" if f == 0 else
                    "🚨 VOID — a re-run of the same branch disagreed")}
    print(json.dumps(out, indent=1))
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
