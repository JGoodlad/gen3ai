### 3.1 `grid` — the SENSITIVE leaf row, 100 pairs per head, COMPLETE

Unguarded search: score every legal action on the leaf, play the best. No gate to hide behind, and
the row that separated heads with DETECTED deltas in the phase-2/3 battery.

| head (what the lever is) | paired mirror win rate | action changed | vs the **contemporaneous** `ctrl10M` anchor, paired on the 100 shared indices |
|---|---|---|---|
| **`ctrl10M` ANCHOR** (the ladder control) | **0.2975** [0.2353, 0.3597] | 58.2 % | — |
| `vf025` (`--vf-coef 0.25`) | 0.2825 [0.2265, 0.3385] | 62.4 % | −0.0150 [−0.0934, +0.0634] NOT DETECTED |
| `ent05` (`--ent-coef 0.05`) | 0.2725 [0.2113, 0.3337] | 62.2 % | −0.0250 [−0.1127, +0.0627] NOT DETECTED |
| `rollout` (`--win-prob-rollout-target`) | **0.2450** [0.1819, 0.3081] | 59.9 % | −0.0525 [−0.1412, +0.0362] NOT DETECTED |

**All four SEARCH HARMS, decisively** — every CI upper bound is below 0.36, i.e. unguarded search
on each of these leaves loses roughly three games in four against its own unsearched self. **No arm
beats its contemporaneous anchor; all three point estimates sit BELOW it**, and the ordering puts
the rollout-target arm last.
