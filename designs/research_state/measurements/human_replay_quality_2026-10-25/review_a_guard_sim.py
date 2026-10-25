"""Would main.heldout_panels.contamination (exact mon sets, >= 4 shared) catch a human who piloted a held-out panel
team EXACTLY, after a C-GAME-like completion? Each mon reveals each of its moves with p (the card: ~1.9 of 4 at
>= 1500), the rest are refilled from Smogon's move marginals; the item / ability / spread do not enter the key.
All six mons are kept (an upper bound on the guard's recall)."""
from __future__ import annotations

import json
import sys

import numpy as np

from agents.gen3_data import priors
from main import heldout_panels as HP
from main.anchors.team_overlap import parse_team

P_REVEAL = float(sys.argv[1]) if len(sys.argv) > 1 else 1.9 / 4
rng = np.random.default_rng(20261025)
out = {}
for panel in ("heldout72_hpfix", "forum72", "p72"):
    teams = [parse_team(t, lab) for lab, t in HP.panel_team_texts(panel)]
    caught = []
    exact_mons = []
    for t in teams:
        for _ in range(20):
            same = 0
            for ms in t.mons:
                prior = {m: w for m, w in priors.moves(ms.species).items() if w > 0}
                keep = [m for m in ms.moves if rng.random() < P_REVEAL]
                cand = [m for m in prior if m not in keep and not (m.startswith("hiddenpower")
                                                                  and any(k.startswith("hiddenpower") for k in keep))]
                need = len(ms.moves) - len(keep)
                if need and cand:
                    w = np.asarray([prior[m] for m in cand], float)
                    pick = rng.choice(len(cand), size=min(need, len(cand)), replace=False, p=w / w.sum())
                    filled = keep + [cand[i] for i in pick]
                else:
                    filled = keep
                same += tuple(sorted(filled)) == tuple(sorted(ms.moves))
            exact_mons.append(same)
            caught.append(same >= 4)
    out[panel] = {"teams": len(teams), "draws": len(caught), "guard_recall_shared_ge4": float(np.mean(caught)),
                  "mean_exact_mons_of_6": float(np.mean(exact_mons))}
print(json.dumps({"p_reveal_per_move": P_REVEAL, **out}, indent=1))
