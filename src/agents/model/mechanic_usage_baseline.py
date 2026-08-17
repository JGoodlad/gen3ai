"""G2 — the pre-build behavioural BASELINE for the conditional-execution mechanics.

`design_conditional_execution.md` §6 G2: *"Measure current usage rates of each mechanic — how
often does the policy click Counter, Focus Punch, Protect, and with what confidence? If it
already never clicks Counter, that is the number the retrain has to move."* Model-free — it
reads only the eval-trace summary.json `actions` blocks (per-action prob + valid), so it works
on any run regardless of architecture drift, and it defines "did it work" for the
`--intent-threshold` / `--intent-conditional` retrains: the v84/v85 cells exist to move exactly
these numbers, and without this baseline a post-retrain reading has nothing to move FROM.

Per mechanic: `available` (decisions where the move was a valid action), `chosen` (it was the
argmax pick), `pick_rate` = chosen/available, and `mean_prob` (the policy's average probability
mass on it when available — LOW mean_prob with nonzero availability is the "never even
considers it" signature §3.7 predicts for Counter).

Usage:
    python -m agents.model.mechanic_usage_baseline <run_dir> [--out <json>]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Sequence

# The v84 threshold mechanics + the v85 conditional mechanics + Protect (step 5, for context).
MECHANICS = (
    "focuspunch", "substitute", "endure", "destinybond", "endeavor",          # p_thresh (v84)
    "counter", "mirrorcoat", "explosion", "selfdestruct", "pursuit",          # conditional (v85)
    "protect", "detect",                                                      # step 5 context
)


def measure(run_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(run_dir, "eval_traces", "**", "*_summary.json"),
                             recursive=True))
    stats = {m: {"available": 0, "chosen": 0, "prob_sum": 0.0} for m in MECHANICS}
    decisions = 0
    for f in files:
        try:
            doc = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        for inv in doc.get("invocations", []):
            actions = inv.get("actions")
            if not isinstance(actions, dict):
                continue
            decisions += 1
            chosen = inv.get("chosen")
            for m in MECHANICS:
                a = actions.get(m)
                if a is None or not a.get("valid"):
                    continue
                st = stats[m]
                st["available"] += 1
                st["prob_sum"] += float(str(a.get("prob", "0")).rstrip("%")) / 100.0
                if chosen == m:
                    st["chosen"] += 1
    out: dict[str, Any] = {"run": os.path.basename(os.path.normpath(run_dir)),
           "n_summary_files": len(files), "n_decisions": decisions, "mechanics": {}}
    for m, st in stats.items():
        av = st["available"]
        out["mechanics"][m] = {
            "available": av,
            "chosen": st["chosen"],
            "pick_rate": round(st["chosen"] / av, 4) if av else None,
            "mean_prob": round(st["prob_sum"] / av, 4) if av else None,
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help="write the JSON here (default: stdout only)")
    args = ap.parse_args(argv)
    result = measure(args.run_dir)
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
