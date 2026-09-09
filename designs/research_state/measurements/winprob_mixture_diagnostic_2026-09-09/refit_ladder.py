"""Read-only refit of arm A's snapshot ladder over ALL rated steps.

The COMMITTED ladder.json covers only 36M-74M: pool grooming deleted the early
snapshots and the committed fit was sliced to the survivors (the P1 tool defect
named in UNDERSTANDING §4.2b). Seven of this read's sentinel cells are snapshots
at 10M-34M, so the committed file cannot rate them.

games.jsonl still carries the dense frozen-vs-frozen edges for every step from
4.0M up, so a refit over that full step list rates them all on one bot-anchored
scale -- and this tree's fit_ladder DROPS the eval cycles' greedy-vs-stochastic
sentinel edges, which is also the fix for the committed file reading ~+73 high.

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
    ap.add_argument("--run", default="/home/goodlad/dev/gen3ai/models/ai_v12_02_winprob_critic")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    # HAZARD: fit_ladder calls elo.load_bot_anchors() with its default RELATIVE path
    # ("data/gen3_bot_elo_anchors.json"), so a fit run from any other cwd silently loses the
    # bot pins and returns an UNANCHORED ladder -- `anchored_to_bots: false`, ratings ~1000
    # instead of ~2000. Asserted, never assumed.
    os.chdir(repo_root())
    steps = sorted({s for pair in load_games(a.run) for s in pair})
    fit = fit_ladder(a.run, steps=steps, write=False)
    if not fit.get("anchored_to_bots"):
        raise SystemExit("REFUSED: refit is NOT anchored to the bot pins -- the strength axis "
                         "would not share a scale with the bot ratings.")
    with open(a.out, "w") as f:
        json.dump(fit, f, indent=1)
    print(f"steps={len(steps)} rated={len(fit['ratings'])} "
          f"converged={fit.get('converged')} "
          f"sentinel_edges_dropped={fit.get('eval_sentinel_edges_dropped')}")
    r = fit["ratings"]
    for k in sorted(r, key=lambda x: int(x)):
        print(f"  {k:>10} {r[k]:8.1f}")
