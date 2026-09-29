# The Kakuna goal read — X22 (b)–(f), 2026-09-28

**Verdict: the near-term goal is NOT MET.** The bar is N0's Wilson 95% lower bound > 0.50 against
`metamon:Kakuna` on our teams, at Kakuna's strongest temperature. Kakuna is strongest at
**T = 1.0**, where N0 reads **0.380 [0.316, 0.449]** (n = 200). The whole interval is below 0.50,
so N0 is **WORSE** there. At greedy (the SOP's standard regime) N0 reads **0.460 [0.392, 0.529]**
(n = 200): NOT DETECTED, and the lower bound does not clear 0.50.

**(f), the owner's question "does our model play better greedy?": YES.** N0 sampling at T = 1.0 is
worse than N0 greedy against both Kakuna settings: −0.070 [−0.161, +0.023] against Kakuna T = 1.0
(not detected) and **−0.230 [−0.354, −0.095]** against Kakuna greedy (detected). At T = 1.0 N0 plays
its own argmax on only **62.8%** of decisions (counted per decision), so T = 1.0 moves about 37% of
its choices. Greedy is the right way to deploy it (§ (f)).

## Stamps shared by every number below

| field | value |
|---|---|
| our side | `models/ai_v14_01_base/final_model.zip`, **step 75,005,952** (`explicit_zip`), `--temperature 0.0` (greedy), `--device cpu`, `main.play`'s 250-turn forfeit |
| Kakuna | `metamon:Kakuna` **ckpt 34** (142,832,563 params), metamon checkout `0a00a759`, CPU, windowed attention (H20), `OMP_NUM_THREADS=1` |
| teams | **home72**: the N0 read's sanitized 72 (`n0_endofrun_2026-09-27/teams_home72`), byte-identical to Metamon's `gen3ai_home72` copy (72/72 `cmp` checked) and to Foul Play's `--team-name` dir. `team_source_asymmetry = false` on every unit |
| transport | `--server rust` (the websocket front end), `sim_bridge` from the N0 read's pinned tree `dcddac0b` (`src/rust_sim` and `data/` are identical to this commit's), Showdown pin `e0551883f` |
| tool | `main.anchors` at this commit's tree (base `678112b4` + the `--opponent-temperature` flag, §4) |
| units | 20-game units ((b)/(c)), 50-game units ((d)), 10-game units ((e)). Each unit uses `--team-seed = --seed-base = s`, role-balanced halves, and is pooled by summing. **Paired seeds:** Kakuna's greedy, T = 0.5 and T = 1.0 cells use seeds 0–4 (the 100-game reads). Greedy and T = 1.0 then continue with seeds 5–9 |
| box | CPU only, no GPU. A training arm and other CPU work were live throughout. 1-min load average **24–29 on 16 cores**, recorded per unit (`unit_load1_mean` / `unit_load1_max` on every row) |
| unit counts iff | `status == OK`, `n ==` unit games, `regime_verified_decisions == true`, `team_source_asymmetry == false` (the N0 read's rule). No unit was excluded |

## (b) + (c) — N0 vs Kakuna, by Kakuna's temperature

| Kakuna regime | n | W–L–T | N0 win rate [Wilson 95%] | Kakuna `argmax_match_rate` per half | vs 0.50 |
|---|---:|---|---|---|---|
| greedy (`sample=False`) | 200 | 92–108–0 | **0.460 [0.392, 0.529]** | 1.0000 (all) | NOT DETECTED |
| sample, T = 0.5 | 100 | 44–56–0 | 0.440 [0.347, 0.538] | 0.943–0.976 | NOT DETECTED |
| **sample, T = 1.0 (strongest)** | 200 | 76–123–1 | **0.380 [0.316, 0.449]** | 0.847–0.928 | **WORSE** (upper bound < 0.50) |

`regime_matched = false` on the T = 0.5 and T = 1.0 rows, and `their_regime = sample:T=<T>`. Our
side is greedy in all three. Every peer exited cleanly (`peer_clean` true on all 25 units); no H17.
Median game length is 32 turns in all three cells.

**Picking the strongest temperature (rule set in `scripts/kq.py` before any top-up):** the lowest
N0 win rate on the paired seeds 0–4 (100 games each) gets topped up to 200. Those rates were
greedy 0.51, T = 0.5 0.44, **T = 1.0 0.36**, so T = 1.0 was topped up (`topup.json` in the state
dir, copied into `cells.json`). 🚨 **The ranking is a point-estimate ordering and the differences are
not detected** (Newcombe, independent-sample):

- greedy − T1.0 = **+0.080 [−0.017, +0.175]**
- greedy − T0.5 = +0.020 [−0.099, +0.136]
- T0.5 − T1.0 = +0.060 [−0.056, +0.177]

"Kakuna is strongest at T = 1.0" is the best current reading, not a resolved fact. Both readings
fail the goal either way: neither lower bound is near 0.50.

Context: at T = 1.0 Kakuna plays its own argmax **85–93%** of the time. That sits between
`SyntheticRLV2` (88.0%) and `SmallRL` (64.7%) on the SOP's RULE 1 table, and is the same
direction as `SyntheticRLV2`: sampling helps this larger model.

## (d) — Kakuna vs SyntheticRLV2, greedy vs greedy (pair cell)

`--opponent-a metamon:Kakuna --opponent-b metamon:SyntheticRLV2 --regime greedy --teamset home`
(home72, both sides). Reported **from Kakuna's side**. Metamon's battle CSV is the record, and it
books a tie as a loss.

| n | Kakuna W–L | Kakuna win rate [Wilson 95%] | argmax (both peers) | turns |
|---:|---|---|---|---|
| 200 | 131–69 | **0.655 [0.587, 0.717]** | 1.0000 | median 40, max 454. 2 games > 250 turns |

**Kakuna beats SyntheticRLV2 (BETTER, lower bound > 0.50).** 🚨 **A pair cell has no 250-turn
forfeit** (SOP §2): the sim's 1000-turn tie is the only cap. So this edge is under a different
stall rule from the N0 cells. Two games went past 250 turns.
SyntheticRLV2 ckpt 48. Load 28.6 mean.

## (e) — Kakuna (greedy) vs Foul Play @ 1,000 ms (pair cell)

`--opponent-a metamon:Kakuna --opponent-b foulplay --regime greedy --teamset home --search-time-ms
1000 --search-parallelism 1` (Foul Play `--search-threads 1`), home72 on both sides. Reported from
**Kakuna's side**.

| n | Kakuna W–L | Kakuna win rate [Wilson 95%] | FP realized visits/decision | turns | box |
|---:|---|---|---|---|---|
| **200** (20 × 10-game units, seeds 200–219) | 98–102 | **0.490 [0.422, 0.559]** | mean **601 k**, per-row range 388 k–885 k | median 46, max 387, 1 game > 250 | load1 mean 26.5 over units (per-unit 23.1–29.9) |

**NOT DETECTED**: Kakuna at greedy and Foul Play at 1,000 ms read as even. Units e_00–e_12 ran on
2026-09-28 (`--nice 15` on the peer); e_13–e_19 were resumed 2026-09-28 22:48–00:08 PT with the whole
unit at nice 19, peers included (verified with `ps -o ni` on the Metamon and Foul Play children).
🚨 Foul Play's strength depends on CPU. Its budget is wall clock, so the realized width is part of
the result (SOP rule 23). Every row carries `fp_search_time_ms = 1000`, `their_visits_mean`,
`unit_load1_mean` and `unit_load1_max`. The width here (601 k) is below the N0 read's 728 k smoke
and the 09-16 read's 780 k. Kakuna's greedy regime is verified per decision
(`regime_verified_decisions` true on all 20 units). All peers exited cleanly. 🚨 A pair cell has
no 250-turn forfeit (SOP §2); one game ran to 387 turns.

## (f) — does N0 play better greedy? (owner question, 2026-09-28)

The question: we train at T = 1.0, so is greedy out of distribution? Two cells with **N0 sampling
at T = 1.0** (`--our-temperature 1.0`, new flag, §5), each paired with a banked greedy cell on the
same seeds. Same model, teams (home72), transport and tool as above. Our sampling generator is
seeded per unit (`GEN3AI_POLICY_SEED` = the unit seed). Every game in each pair has the same team
for our side (200/200 and 100/100 checked).

| cell | Kakuna | n | W–L–T | N0 win rate [Wilson 95%] | paired greedy cell (same seeds) | greedy − sampled [95%] |
|---|---|---:|---|---|---|---|
| **f1** | sample T = 1.0 | 200 | 62–137–1 | **0.310 [0.250, 0.377]** | (c) 0.380 [0.316, 0.449], seeds 0–9 | **+0.070 [−0.023, +0.161]**, NOT DETECTED |
| **f2** | greedy | 100 | 28–72–0 | **0.280 [0.201, 0.375]** | (b) seeds 0–4: 0.510 (51/100) | **+0.230 [+0.095, +0.354]**, DETECTED |
| | | | | | (b) all 200: 0.460 | +0.180 [+0.063, +0.285] |

**Method:** the difference of two independent proportions, Newcombe's hybrid score interval from
the two Wilson intervals (`main.anchors.results.newcombe`). It ignores the pairing, so it is
conservative. Paired game by game (same seed and slot), the discordant counts are greedy-only wins
44 vs sampled-only wins 30 (f1) and **35 vs 12** (f2).

`regime_matched = true` on f1 (both sides at nominal T = 1.0) and false on f2
(`our_regime = sample:T=1`, `their_regime = greedy`). `regime_verified_decisions` true on all 15
units: every one of our decisions received `stochastic=True`, and Kakuna's argmax rate is
0.875–0.922 per half in f1 and 1.0000 in f2. All peers exited cleanly. Median 31 turns. Load1 mean
19.7 (f1) and 25.3 (f2), per unit on every row.

### (f3) — OUR own-argmax rate at T = 1.0

Counted **per decision**: the sampled action against the argmax of the same decision's masked
logits (`our_argmax_matches` / `our_argmax_decisions` on every row; 12,526 decisions, which equals
the play client's own decision counter in all 30 halves).

| cell | our decisions | = our argmax | rate | per-unit range |
|---|---:|---:|---:|---|
| f1 (vs Kakuna T = 1.0) | 8,418 | 5,287 | **62.8%** | 59.8–68.1% |
| f2 (vs Kakuna greedy) | 4,108 | 2,605 | **63.4%** | 61.0–66.9% |

On the SOP §0 table N0 sits below `SmallRL` (64.7%), with `SyntheticRLV2` (88.0%) and Kakuna
(85–93%) far above. At T = 1.0 more than a third of N0's choices are not its top action.

### The answer to the owner's two questions

1. **Does N0 play better greedy? Yes.** Against greedy Kakuna, sampling costs 23 pp on paired seeds
   (CI clear of zero). Against sampling Kakuna the point estimate is −7 pp, not detected. Neither
   cell shows sampling helping. This is the opposite of the larger Metamon models, which gain from
   sampling.
2. **Is greedy out of distribution because we train at T = 1? Not in any way that hurts.** Greedy
   does put N0 in states its T = 1 rollouts reach less often, because ~37% of its training-time
   actions are not its argmax. But the measured effect of removing that noise is a gain, not a loss.
   Sampling changes ~37% of N0's decisions away from its top action, and on this evidence those
   changes cost games. **UNVERIFIED:** whether a temperature between 0 and 1 beats greedy
   was not measured.

## The round robin on home72 so far (greedy Metamon, descriptors)

| edge | reading | source |
|---|---|---|
| N0 – Kakuna | N0 **0.460** [0.392, 0.529], n = 200 | (c), this read |
| Kakuna – SyntheticRLV2 | Kakuna **0.655** [0.587, 0.717], n = 200 | (d), this read |
| Kakuna – Foul Play @1000 ms | Kakuna **0.490** [0.422, 0.559], n = 200 | (e), this read |
| N0 – SyntheticRLV2, N0 – Foul Play, SyntheticRLV2 – Foul Play | not restated here | `n0_endofrun_2026-09-27` A4 / A3 / §3 (their own README owns them) |

No fit is made here. Each edge is under its own stall rule (N0 forfeits at 250; pair cells do not).

## P2 side-peek (V's within-game discrimination vs Kakuna)

**Not available cheaply, so not done.** The anchors rows carry per-GAME outcomes and per-decision
REGIME instruments (`n_decisions`, the sample kwargs, the argmax rate). They do **not** carry our
value head's per-decision V. `main.play` does not log it, and building a logger was out of this
brief's scope.

## 4. The tool changes this read needed — `--opponent-temperature`

`main.anchors` refused the "us greedy, Metamon sampling" cell outright: the H17 `mixed` refusal on
`--allow-unmatched-regime`, and `--regime` names both sides. X22(b) needs exactly that cell. The
new flag `--opponent-temperature T` does three things:

- It runs the Metamon peer at `--regime t1 --temperature T`.
- It keeps our side at `--temperature 0.0`.
- It stamps `regime_matched = false` and `their_regime = sample:T=<T>`.

It is refused for any other opponent, our-side or regime, and for T ≤ 0. The flag is tested in
`src/main/anchors/cli_test.py` and documented in `designs/ops/EXTERNAL_ANCHORS_SOP.md` §2. H17 did
not fire in any of the 15 mixed units.

## 5. `--our-temperature` (X22 (f))

The mirror of §4: `--our-temperature T` makes OUR checkpoint sample (`main.play --temperature T`)
while the Metamon peer follows `--regime greedy` or `--opponent-temperature`. Every row is stamped
`our_regime = sample:T=<T>`; `regime_matched` is true only when both nominal temperatures are equal.
Our half is verified per decision (every decision must receive `stochastic=True`), and each row
carries `our_argmax_matches` / `our_argmax_decisions`, read from the masked logits that
`RLPlayer._predict_best_action` now keeps as `_last_masked_logits` (a tensor reference; no copy, no
second forward). Tests: `src/main/anchors/cli_test.py`, `session_test.py`, `results_test.py`. SOP:
`designs/ops/EXTERNAL_ANCHORS_SOP.md` §2. Smoke (2 games, `state/smoke/f1_smoke`): 69 decisions,
counted and verified, before the queue ran.

## Files

- `rows.jsonl`: every game row of every counted unit, the anchors tool's full row plus these fields:
  - `x22_cell`, `unit`, `unit_seed`, `lane`
  - `unit_load1_mean`, `unit_load1_max`
  - `fp_search_time_ms`
- `cells.json`: pooled per cell, with the argmax rates (theirs, and ours as `our_argmax_*`), dirty exits,
  load and FP visits, plus the top-up decision and the (f) greedy − sampled deltas (`f_deltas`).
- `scripts/kq.py`: the detached supervisor. 2 lanes, ports 9561/9562. Units run at `--nice 19` and
  the supervisor under `nice -n 19` (from the e_13 resume on), so peers inherit 19. `run | status`. A unit is done
  iff its own `summary.json` says so. A partial unit is moved aside and re-run. `--stop-at` stops
  new starts and kills a straggler by its own process group.
- `scripts/aggregate.py`: pools the units into `rows.jsonl` and `cells.json`.
- Raw per-unit cells (logs, peer reports, Metamon CSVs), `units.jsonl` (every start and end with
  rc, wall and load) and `topup.json` are in the STATE dir
  `/home/goodlad/dev/gen3ai-reads/kakuna_goal_read_2026-09-28/`.

