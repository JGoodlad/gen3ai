"""Score the 2026-09-11 search-dividend battery over the three current WIN-PROB heads.

Reads the shard files written by `run_battery.sh` (`<cell>__<head>__s<lo>.jsonl`), REFUSES
overlapping shard windows (the per-game seed and team draw are functions of the index alone, so
an overlap would double-count one battle and silently break the pairing), and prints/dumps:

* per (cell, head): the PAIRED mirror win rate with its CI — the null is 0.50 by CONSTRUCTION,
  so that interval IS the dividend's own CI;
* the mechanism fold (forced / raced / separated / overruled, realized widths, banked clock);
* head-vs-head and rung-vs-rung PAIRED deltas on the SHARED game indices — every head plays the
  same battles at `--games-seed 7`, so these are paired differences, not two point estimates
  compared across independent samples;
* the timeout/error hygiene check: >25% of attempted battles unfinished ⇒ the cell is
  INCONCLUSIVE, never a win rate.

Run:  python3 report.py <data_dir> [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

from main.search_dividend.summary import mirror_report

FNAME = re.compile(r"(?P<cell>[a-zA-Z0-9]+)__(?P<head>[a-zA-Z0-9_]+)__s(?P<lo>\d+)\.jsonl$")
INCONCLUSIVE_FRAC = 0.25


def load(data_dir: str):
    """{(cell, head): [rows]} — refusing an overlapping shard window rather than pooling it."""
    groups = defaultdict(list)
    windows = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
        m = FNAME.search(os.path.basename(path))
        if not m:
            continue
        key = (m["cell"], m["head"])
        rows = [json.loads(line) for line in open(path) if line.strip()]
        idx = {int(r["game"]) for r in rows}
        for other, prev in windows[key]:
            if idx & prev:
                raise SystemExit(f"OVERLAPPING SHARDS for {key}: {other} and {path} share "
                                 f"{len(idx & prev)} game indices — refusing to pool.")
        windows[key].append((path, idx))
        groups[key].extend(rows)
    return groups


def pair_scores(rows):
    """{game_index: pair score in {0, 0.5, 1}} over FINISHED side-swap pairs only."""
    by_game = defaultdict(dict)
    for r in rows:
        if not int(r.get("finished", 0)):
            continue
        s = 0.5 if int(r.get("tied", 0) or 0) else float(int(r.get("won", 0)))
        by_game[int(r["game"])][int(r.get("orientation", 0) or 0)] = s
    return {g: sum(o.values()) / 2.0 for g, o in by_game.items() if len(o) == 2}


def mean_ci(xs):
    n = len(xs)
    if n < 2:
        return (None, None, n)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    half = 1.96 * math.sqrt(var / n)
    return (m, (m - half, m + half), n)


def paired_delta(a_rows, b_rows):
    """Mean of (A's pair score − B's pair score) over the game indices BOTH played."""
    a, b = pair_scores(a_rows), pair_scores(b_rows)
    shared = sorted(set(a) & set(b))
    m, ci, n = mean_ci([a[g] - b[g] for g in shared])
    return {"n_shared_pairs": n, "delta": m, "ci95": ci,
            "detected": bool(ci and (ci[0] > 0 or ci[1] < 0))}


def verdict(paired, ci):
    if paired is None or ci is None:
        return "NO DATA"
    if ci[0] > 0.50:
        return "SEARCH PAYS"
    if ci[1] < 0.50:
        return "SEARCH HARMS"
    if paired >= 0.51:
        return "real but unresolved"
    if 0.49 <= paired <= 0.51:
        return "NO DIVIDEND DETECTED"
    return "below null, unresolved"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    groups = load(args.data_dir)
    out = {"cells": {}, "deltas": {}}

    for (cell, head), rows in sorted(groups.items()):
        mc = mirror_report(rows)["cells"]
        assert len(mc) == 1, f"{cell}/{head}: expected one mirror cell, got {len(mc)}"
        c = mc[0]
        attempted = len(rows)
        unfinished = sum(1 for r in rows if not int(r.get("finished", 0)))
        frac = unfinished / attempted if attempted else 0.0
        strategies = sorted({r.get("root_strategy") for r in rows})
        scores = sorted({r.get("score_mode") for r in rows})
        rec = {
            "cell": cell, "head": head, "attempted": attempted,
            "finished": c["finished"], "unfinished": unfinished,
            "unfinished_frac": round(frac, 4),
            "INCONCLUSIVE": frac > INCONCLUSIVE_FRAC,
            "root_strategy": strategies, "score_mode": scores,
            "n_pairs": c["n_pairs"],
            "paired_win_rate": c["paired_win_rate"], "paired_ci95": c["paired_ci95"],
            "dividend": (None if c["paired_win_rate"] is None
                         else round(c["paired_win_rate"] - 0.5, 4)),
            "dividend_ci95": (None if not c["paired_ci95"] else
                              [round(c["paired_ci95"][0] - 0.5, 4),
                               round(c["paired_ci95"][1] - 0.5, 4)]),
            "unpaired_win_rate": c["win_rate"], "unpaired_ci95": c["ci95"],
            "ties": c["ties"], "errors": c["errors"],
            "decisions": c["decisions"], "searched": c["searched"],
            "change_rate": c["change_rate"], "realized_mean": c["realized_mean"],
            "defensive": c["defensive"], "fallbacks": c["fallbacks"],
            "verdict": verdict(c["paired_win_rate"], c["paired_ci95"]),
        }
        if rec["INCONCLUSIVE"]:
            rec["verdict"] = "INCONCLUSIVE (unfinished fraction > 25%)"
        out["cells"][f"{cell}/{head}"] = rec

    heads = ["ctrl10M", "lambda09", "wp73M"]
    for cell in sorted({c for c, _ in groups}):
        for i, a in enumerate(heads):
            for b in heads[i + 1:]:
                if (cell, a) in groups and (cell, b) in groups:
                    out["deltas"][f"{cell}: {a} - {b}"] = paired_delta(groups[(cell, a)],
                                                                      groups[(cell, b)])
    for head in heads:
        for a, b in (("defB", "defA"), ("defB", "grid"), ("defA", "grid")):
            if (a, head) in groups and (b, head) in groups:
                out["deltas"][f"{head}: {a} - {b}"] = paired_delta(groups[(a, head)],
                                                                  groups[(b, head)])

    print(f"{'cell/head':26} {'pairs':>6} {'paired':>7} {'CI95':>18} {'verdict':24} "
          f"{'forced':>7} {'overrule':>9} {'chg':>6} {'unfin':>6}")
    for k, r in out["cells"].items():
        d = r["defensive"] or {}
        ci = ("-" if not r["paired_ci95"] else
              f"[{r['paired_ci95'][0]:.4f},{r['paired_ci95'][1]:.4f}]")
        print(f"{k:26} {r['n_pairs']:6d} "
              f"{(r['paired_win_rate'] if r['paired_win_rate'] is not None else float('nan')):7.4f} "
              f"{ci:>18} {r['verdict']:24} "
              f"{d.get('forced_rate', float('nan')) if d else float('nan'):7} "
              f"{d.get('overrule_rate', float('nan')) if d else float('nan'):>9} "
              f"{r['change_rate']:6} {r['unfinished']:6d}")
    print()
    for k, d in out["deltas"].items():
        if d["delta"] is None:
            print(f"{k:34} n<2")
            continue
        print(f"{k:34} {d['delta']:+.4f} [{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] "
              f"over {d['n_shared_pairs']} shared pairs "
              f"{'DETECTED' if d['detected'] else 'NOT DETECTED'}")

    if args.json:
        json.dump(out, open(args.json, "w"), indent=2, sort_keys=True)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
