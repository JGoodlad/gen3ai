# The Kakuna goal read — X22 (b)–(e), 2026-09-28

**Verdict: the near-term goal is NOT MET.** The bar is N0's Wilson 95% lower bound > 0.50 against
`metamon:Kakuna` on our teams, at Kakuna's strongest temperature. Kakuna is strongest at
**T = 1.0**, where N0 reads **0.380 [0.316, 0.449]** (n = 200). The whole interval is below 0.50,
so N0 is **WORSE** there. At greedy (the SOP's standard regime) N0 reads **0.460 [0.392, 0.529]**
(n = 200): NOT DETECTED, and the lower bound does not clear 0.50.

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

| n (of 200 planned) | Kakuna W–L | Kakuna win rate [Wilson 95%] | FP realized visits/decision | turns | box |
|---:|---|---|---|---|---|
| **130** (13 × 10-game units, seeds 200–212) | 64–66 | **0.492 [0.408, 0.577]** | mean **601 k**, per-unit range 481 k–783 k | median 48, max 190, none > 250 | load1 mean 25.8 per unit, max 38–44 |

**NOT DETECTED** at n = 130: Kakuna at greedy and Foul Play at 1,000 ms read as roughly even. The
cell is **INCOMPLETE**: units e_13–e_19 (70 games) are pending at the hard stop. 🚨 Foul Play's
strength depends on CPU. Its budget is wall clock, so the realized width is part of the result
(SOP rule 23). Every row carries `fp_search_time_ms = 1000`, `their_visits_mean`,
`unit_load1_mean` and `unit_load1_max`. The width here (601 k) is below the N0 read's 728 k
smoke and the 09-16 read's 780 k. Kakuna's greedy regime is verified per decision
(`regime_verified_decisions` true on all 13 units). All peers exited cleanly.

## The round robin on home72 so far (greedy Metamon, descriptors)

| edge | reading | source |
|---|---|---|
| N0 – Kakuna | N0 **0.460** [0.392, 0.529], n = 200 | (c), this read |
| Kakuna – SyntheticRLV2 | Kakuna **0.655** [0.587, 0.717], n = 200 | (d), this read |
| Kakuna – Foul Play @1000 ms | Kakuna **0.492** [0.408, 0.577], n = 130 (incomplete) | (e), this read |
| N0 – SyntheticRLV2, N0 – Foul Play, SyntheticRLV2 – Foul Play | not restated here | `n0_endofrun_2026-09-27` A4 / A3 / §3 (their own README owns them) |

No fit is made here. Each edge is under its own stall rule (N0 forfeits at 250; pair cells do not).

## P2 side-peek (V's within-game discrimination vs Kakuna)

**Not available cheaply, so not done.** The anchors rows carry per-GAME outcomes and per-decision
REGIME instruments (`n_decisions`, the sample kwargs, the argmax rate). They do **not** carry our
value head's per-decision V. `main.play` does not log it, and building a logger was out of this
brief's scope.

## 4. A tool change this read needed — `--opponent-temperature`

`main.anchors` refused the "us greedy, Metamon sampling" cell outright: the H17 `mixed` refusal on
`--allow-unmatched-regime`, and `--regime` names both sides. X22(b) needs exactly that cell. The
new flag `--opponent-temperature T` does three things:

- It runs the Metamon peer at `--regime t1 --temperature T`.
- It keeps our side at `--temperature 0.0`.
- It stamps `regime_matched = false` and `their_regime = sample:T=<T>`.

It is refused for any other opponent, our-side or regime, and for T ≤ 0. The flag is tested in
`src/main/anchors/cli_test.py` and documented in `designs/ops/EXTERNAL_ANCHORS_SOP.md` §2. H17 did
not fire in any of the 15 mixed units.

## Files

- `rows.jsonl`: every game row of every counted unit, the anchors tool's full row plus these fields:
  - `x22_cell`, `unit`, `unit_seed`, `lane`
  - `unit_load1_mean`, `unit_load1_max`
  - `fp_search_time_ms`
- `cells.json`: pooled per cell, with the argmax rates, dirty exits, load and FP visits, plus the top-up decision.
- `scripts/kq.py`: the detached supervisor. 2 lanes, ports 9561/9562. `run | status`. A unit is done
  iff its own `summary.json` says so. A partial unit is moved aside and re-run. `--stop-at` stops
  new starts and kills a straggler by its own process group.
- `scripts/aggregate.py`: pools the units into `rows.jsonl` and `cells.json`.
- Raw per-unit cells (logs, peer reports, Metamon CSVs), `units.jsonl` (every start and end with
  rc, wall and load) and `topup.json` are in the STATE dir
  `/home/goodlad/dev/gen3ai-reads/kakuna_goal_read_2026-09-28/`.

RESUM`--opponent-a metamon:Kakuna --opponent-b foulplay --regime greedy --teamset home --search-time-ms
1000 --search-parallelism 1` (Foul Play `--search-threads 1`), home72 on both sides. Reported from
**Kakuna's side**.

| n (of 200 planned) | Kakuna W–L | Kakuna win rate [Wilson 95%] | FP realized visits/decision | turns | box |
|---:|---|---|---|---|---|
| **130** (13 × 10-game units, seeds 200–212) | 64–66 | **0.492 [0.408, 0.577]** | mean **601 k**, per-unit range 481 k–783 k | median 48, max 190, none > 250 | load1 mean 25.8 per unit, max 38–44 |

**NOT DETECTED** at n = 130: Kakuna at greedy and Foul Play at 1,000 ms read as roughly even. The
cell is **INCOMPLETE**: units e_13–e_19 (70 games) are pending at the hard stop. 🚨 Foul Play's
strength depends on CPU. Its budget is wall clock, so the realized width is part of the result
(SOP rule 23). Every row carries `fp_search_time_ms = 1000`, `their_visits_mean`,
`unit_load1_mean` and `unit_load1_max`. The width here (601 k) is below the N0 read's 728 k
smoke and the 09-16 read's 780 k. Kakuna's greedy regime is verified per decision
(`regime_verified_decisions` true on all 13 units). All peers exited cleanly.
