"""Read-only refit of a run's snapshot ladder over ALL rated steps (any run).

Generalised from `winprob_mixture_diagnostic_2026-09-09/refit_ladder.py`: the same
hazard, the same refusal, one `--run` argument instead of a hardcoded arm.

Why a refit and not the committed `ladder.json`: pool grooming deletes early snapshots
and the committed fit is sliced to the survivors (the P1 defect named in UNDERSTANDING
§4.2b), so a cycle whose sentinel is an early snapshot would be unrated. `games.jsonl`
still carries the dense frozen-vs-frozen edges, and this tree's `fit_ladder` drops the
eval cycles' greedy-vs-stochastic sentinel edges.

write=False. Nothing is written under models/.
"""
from __future__ import annotations

import argparse
import json
import os

from agents.training.snapshot_ladder import fit_ladder, load_games
from utils.paths import repo_root

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    # 🚨 HAZARD (inherited, re-asserted): fit_ladder calls elo.load_bot_anchors() with its default
    # RELATIVE path ("data/gen3_bot_elo_anchors.json"), so a fit run from any other cwd silently
    # loses the bot pins and returns an UNANCHORED ladder — `anchored_to_bots: false`, ratings
    # ~1000 instead of ~2000, no error. Asserted, never assumed.
    os.chdir(repo_root())
    steps = sorted({s for pair in load_games(a.run) for s in pair})
    fit = fit_ladder(a.run, steps=steps, write=False)
    if not fit.get("anchored_to_bots"):
        raise SystemExit("REFUSED: refit is NOT anchored to the bot pins — the strength axis "
                         "would not share a scale with the bot ratings.")
    with open(a.out, "w") as f:
        json.dump(fit, f, indent=1)
    print(f"run={os.path.basename(a.run.rstrip('/'))} steps={len(steps)} "
          f"rated={len(fit['ratings'])} converged={fit.get('converged')} "
          f"sentinel_edges_dropped={fit.get('eval_sentinel_edges_dropped')}")
    r = fit["ratings"]
    for k in sorted(r, key=lambda x: int(x)):
        print(f"  {k:>10} {r[k]:8.1f}")
