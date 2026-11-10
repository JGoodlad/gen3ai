"""The training pool's per-species movesets, for MEASUREMENT only (never a prior): n_sets[S], count[S, M] by
national-dex num and move num (the typed Hidden Power keeps its own num), and every distinct (species, moveset)."""
from __future__ import annotations

import sys

import numpy as np

from agents import gen3_data
from main.anchors.team_overlap import parse_team
from utils.team_sources import team_list

S, M = 400, 400
teams = [parse_team(x, f"pool{i}") for i, x in enumerate(team_list("pool"))]
n_sets = np.zeros(S, dtype=np.int64)
count = np.zeros((S, M), dtype=np.int64)
from collections import Counter
sets = Counter()
skipped = 0
for t in teams:
    for mon in t.mons:
        sd = gen3_data.species.get(mon.species)
        if sd is None or not (0 < sd.num < S):
            skipped += 1
            continue
        nums = []
        for mid in mon.moves:
            md = gen3_data.moves.get(mid)
            if md is None or not (0 < md.num < M):
                skipped += 1
                continue
            nums.append(int(md.num))
        n_sets[sd.num] += 1
        for n in set(nums):
            count[sd.num, n] += 1
        sets[(int(sd.num), *sorted(set(nums)), *([-1] * (4 - len(set(nums)))))] += 1
print(f"pool teams {len(teams)}, mons {int(n_sets.sum())}, species {int((n_sets > 0).sum())}, distinct sets {len(sets)}, skipped {skipped}")
ks = sorted(sets)
np.savez(sys.argv[1], n_sets=n_sets, count=count, sets=np.array(ks, dtype=np.int64), set_count=np.array([sets[k] for k in ks], dtype=np.int64))
