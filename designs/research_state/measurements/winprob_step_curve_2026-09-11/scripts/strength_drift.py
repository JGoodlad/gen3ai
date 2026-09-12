#!/usr/bin/env python3
"""THE STEP CURVE — the strength drift the curve must be read against.

🚨 WHY THIS EXISTS. The four checkpoints face an IDENTICAL opponent panel (PREDICTION.md's
amendment), so the panel does not move — but the TRAINEE does, and two of the registered rows are
bounded by quantities that move with it:

* **Murphy resolution is capped by the base-rate variance `p(1-p)`.** A trainee that wins more has
  less outcome variance to resolve, so `gate.resolution.*` can fall while the critic is unchanged.
* **The opponent-CLASS AUC is a pool-vs-bot separation.** As the trainee closes the gap on the
  fixed late-strength sentinels, pool and bot outcomes CONVERGE, and two classes whose outcomes
  converge are harder to separate — whatever the representation knows.

So this table is not context: it is the denominator. It is read straight off the trace FILENAMES
(`win_*`/`loss_*`/`draw_*`), which the recorder writes per battle, so it costs nothing and depends
on no model forward and no fitted decoder.

Usage:  strength_drift.py <cycles_root> [--json OUT.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from pathlib import Path

STEPS = [9_969_408, 18_660_864, 40_935_168, 73_121_280]
LABEL = {9_969_408: "10M", 18_660_864: "20M", 40_935_168: "40M", 73_121_280: "73M"}
DRAWS = ["draw1", "draw2"]
EXPECTED = 4800


def cycle_rates(root: Path, step: int, draw: str) -> dict | None:
    base = root / str(step) / draw / "eval_traces" / f"step_{step}"
    files = glob.glob(str(base / "*" / "*_states.npz"))
    if not files:
        return None
    tot = collections.Counter()
    bot = collections.Counter()
    pool = collections.Counter()
    for f in files:
        opp = Path(f).parent.name
        outcome = os.path.basename(f).split("_")[0]
        tot[outcome] += 1
        (pool if opp.startswith("sentinel") else bot)[outcome] += 1
    n = sum(tot.values())
    nb, npl = sum(bot.values()), sum(pool.values())
    p = tot["win"] / n
    return {
        "battles": n, "complete": n == EXPECTED,
        "win": tot["win"], "loss": tot["loss"], "draw": tot["draw"],
        "win_rate": p, "base_rate_variance": p * (1 - p),
        "bot_win_rate": bot["win"] / nb if nb else None,
        "pool_win_rate": pool["win"] / npl if npl else None,
        "class_outcome_gap": (bot["win"] / nb - pool["win"] / npl) if (nb and npl) else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="the directory holding <step>/<draw>/ cycles")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    out: dict = {}
    print("| step | draw | battles | win rate | p(1-p) | bot wr | pool wr | "
          "class outcome gap |")
    print("|---|---|---|---|---|---|---|---|")
    for step in STEPS:
        for draw in DRAWS:
            r = cycle_rates(root, step, draw)
            if r is None:
                continue
            out.setdefault(str(step), {})[draw] = r
            flag = "" if r["complete"] else f" ⚠️ INCOMPLETE ({r['battles']}/{EXPECTED})"
            print(f"| {LABEL[step]} | {draw} | {r['battles']:,}{flag} | {r['win_rate']:.4f} | "
                  f"{r['base_rate_variance']:.4f} | {r['bot_win_rate']:.4f} | "
                  f"{r['pool_win_rate']:.4f} | {r['class_outcome_gap']:.4f} |")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
