### 4.1 The dataset, and the two checks that had to pass before any of it was read

**5,076 forks over 957 battles**, 3 branches each — 15,228 rollouts, produced in 6 shards over
~2.3 h at load 22–55. Target was ~20,000 forks; the realized number is **25 % of that**, and the
reason is the box: a live training arm and its eval workers held the load average at 22–55 on 16
cores for the whole window, and the Part A battery was running beside it. Reported as a scale-down,
not hidden.

| | |
|---|---|
| source | `ai_v12_11_ladder_ctrl10M`'s OFFLINE full-capture eval tree @10000032 (9,600 battles) |
| opponents | `sentinel_2` (8M, recorded win rate 0.536) · `sentinel_1` (6M, 0.725) · `sentinel_0` (4M, 0.861) — the three POOL SENTINELS, reloaded from the plan's exact snapshot paths and played GREEDY, which is the regime the manifest says they were recorded in (`eval_sentinel_greedy: true`) |
| why sentinels only | a sentinel is reloadable EXACTLY and was greedy; a scripted bot is rebuilt from code whose internal RNG the pairing does not control, and against the easy bots (0.87–0.996 win rate) nearly every branch pair would be TIED and carry no ranking signal at all |
| contested selection | move round · turn ∈ [2, 40] · ≥ 3 legal actions · **policy top-2 masked-logit gap below the shard's own 40th percentile**. Realized selection rate **0.400** by construction; the per-shard gap thresholds are in `meta_s*.json`. `|V − 0.5|` is recorded and is NOT the selector |
| branches | `top1` (policy argmax) · `top2` (runner-up) · `rand` (uniform over legal ∖ {top1, top2}, decision-keyed) |
| continuation | both sides GREEDY, the record's own resolved `>start` seed, **no `post_t_seed` reseed** ⇒ one dice stream per fork |
| median fork turn | 15 |
| errors / no-`rand` | **0 / 0** |
| no successor state | 316 branches (2.1 %) — the line ended at the divergence turn |
| draw at the 250-turn cap | 13 branches (0.09 %), EXCLUDED (a capped result is decided by SEAT) |

**CHECK 1 — the ANCHOR.** 40 `divergence_turn=None` full replays (scripted on both sides, nothing
played by a policy): **40/40 reproduced the recorded winner, 0 script exhaustions.** The prefix a
fork forks from is the battle it claims to fork.

**CHECK 2 — the CRN pairing.** Every 50th fork re-ran its `top1` branch a second time and compared
outcomes: **104 checks, 0 disagreements.** With greedy play on both sides and one dice stream, a
branch is a deterministic function of its action — which is what makes the three branches of a fork
a paired comparison rather than three samples.
