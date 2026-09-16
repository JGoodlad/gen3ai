#!/usr/bin/env python3
"""THE PAIR READ — the AWAY-set external anchor, through the TOOL OF RECORD `main.anchors`.

`designs/ops/EXTERNAL_ANCHORS_SOP.md` sec 1: **report BOTH team sets, always** — they disagreed in
SIGN for `SmallRL`, and the away set (Metamon's own 20-team `competitive` gen3ou export) is the one
no dial of ours is set to. The home-set cells are arm S's harness, reproduced verbatim for arm W
(`pair_metamon_cells.py`); this row is the away set, taken on BOTH arms with `python -m main.anchors
--opponent metamon:SmallRL --regime greedy --teamset away --games 100`, which verifies the regime
PER DECISION on both sides and stamps every row of `games.jsonl` with it.

Sign convention: arm S MINUS arm W, the same orientation as every other row in the pair read.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

Z = 1.959963985
ROOT = Path("/home/goodlad/.claude/jobs/9ab51de6/tmp/pair_read/anchors/away")
ARMS = {"armS": ("ai_v13_01_flywheel_shaped", "arm S — the pair's SHAPED arm"),
        "armW": ("ai_v13_02_flywheel_winprob", "arm W — the pair's WIN-PROB arm")}


def wilson(k: int, n: int):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def newcombe(k1, n1, k2, n2):
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return (d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def main() -> int:
    out: dict = {"cell": "metamon:SmallRL, regime greedy (BOTH sides, verified per decision), "
                         "teamset away (Metamon's own 20-team competitive gen3ou export), 100 games "
                         "split evenly across the two challenge roles",
                 "tool": "python -m main.anchors", "arms": {}}
    ks = {}
    for tag, (run, human) in ARMS.items():
        s = json.loads((ROOT / f"{tag}_smallrl_greedy_away" / "summary.json").read_text())
        k, n = int(s["wins"]), int(s["n"])
        ks[tag] = (k, n)
        p, lo, hi = wilson(k, n)
        pr = s["provenance"]["peer_report"]
        out["arms"][tag] = {
            "run": run, "human_description": human,
            "model_step": s["cell"]["model_step"], "model_zip": s["cell"]["model_zip"],
            "n": n, "wins": k, "losses": s["losses"], "ties": s.get("ties"),
            "win_rate": round(p, 4), "wilson95": [round(lo, 4), round(hi, 4)],
            "by_half": {h: {"n": v["n"], "wins": v["wins"], "win_rate": v["win_rate"]}
                        for h, v in s["by_half"].items()},
            "mean_turns": s["mean_turns"], "max_turns": s["max_turns"],
            "hit_forfeit_limit": s["hit_forfeit_limit"],
            "regime_verified": {"ours": s["cell"]["our_regime"],
                                "theirs": s["cell"]["their_regime"],
                                "regime_matched": s["cell"]["regime_matched"],
                                "peer_argmax_match_rate": pr.get("argmax_match_rate")},
            "opponent": {"version": s["cell"]["opponent_version"],
                         "commit": s["cell"]["opponent_commit"]},
            "showdown_pin": s["cell"]["showdown_pin"],
        }
    d, lo, hi = newcombe(*ks["armS"], *ks["armW"])
    out["pair_delta_S_minus_W"] = {
        "delta": round(d, 4), "newcombe95": [round(lo, 4), round(hi, 4)],
        "verdict": ("NOT DETECTED — the difference interval covers zero. Rule 6: never 'equal'; "
                    "equivalence needs the delta's own CI inside a bar and no bar exists for this "
                    "row" if lo <= 0 <= hi else "DETECTED at 95% — the interval excludes zero"),
    }
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pair_away_anchor.json")
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["pair_delta_S_minus_W"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
