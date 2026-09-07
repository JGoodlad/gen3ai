# Does `eval/elo` include the self-play sentinels? — VERIFIED YES

**Direct answer:** Yes. Sentinel edges are in the fit, unweighted and unfiltered, and they are
doing most of the work in the number: removing them from the latest cycle moves the trainee
**2088.74 → 1978.45 (−110.30 Elo)** and widens the CI **±30.05 → ±40.58**.

## 1. Edge inventory (source)

`eval/elo` is written by `record_elo` (`src/agents/training/eval_callback.py:871`), which appends
the cycle row then refits the WHOLE accumulated ladder (`:900` `fit_from_run(model_dir,
source="log")`) and records `eval/elo` + `eval/elo_ci` (`:906-909`). The fit's edges come from
`_rows_to_results` (`src/agents/training/elo.py:374-386`) — per row, subject = `snap:<step>`:

| edge family | present? | file:line |
|---|---|---|
| trainee × bot (all 9: random, heuristic, heuristic2, staller, staller_v2, aggressive, aggressive_v2, setup_sweep, setup_sweep_v2) | **YES** | `elo.py:382-383` |
| trainee × sentinel (`snap:<sentinel_step>`) | **YES — same loop, same `n`, no weight** | `elo.py:384-385` |
| sentinel × sentinel | **NO** — every edge is incident to the row's trainee | `elo.py:380` (`subj` fixed per row) |
| sentinel × bot | **only historically** — a sentinel was itself a trainee in its own earlier row, so its bot edges enter through that row | `elo.py:382-383` + `elo.py:37-39` (a snapshot is the SAME player as trainee or sentinel, unified by step) |

That last row is what links the ladder: bots anchor absolutely (9 pinned from
`data/gen3_bot_elo_anchors.json`, `elo.py:654-657`), and the trainee×sentinel edges chain each
new snapshot to the anchored past.

Both callbacks share `record_elo`. The **self-play** path passes real sentinels
(`selfplay_callback.py:719-724`, from `kept_sentinels`); the **bot-only** path passes `[]`
(`eval_callback.py:1525`, commented "no sentinels on the bot-only path").

**Not muted anywhere:** no per-family weight, no filter. The only drop is a sentinel whose worker
died (`selfplay_callback.py:539-548`, `v is not None`). Games per edge are equal by construction —
bots and sentinels are built as `EvalItem(..., n_games)` from one variable
(`selfplay_callback.py:419-421`), and the fit uses that single `row.n_games` for every edge
(`elo.py:381`).

*(Aside, verified: the two `eval/hodge_*` scalars deliberately EXCLUDE sentinel edges —
`hodge.py:785-787`, "Sentinel edges enter the FIT (real spine information) but lie on no
triangle". That is the width metric only, not `eval/elo`.)*

## 2. Sentinel selection rule

`self._pool.sentinel_entries(n=self._n_sentinels)` (`selfplay_callback.py:407`) →
`snapshot_pool.py:312-323`: up to `n` **evenly spaced entries, newest first**
(`step_size = (len(entries)-1)/(n-1)`; all entries returned while the pool is smaller than `n`).
`--n-sentinels` default **5** (`selfplay_callback.py:173`, `:285`). Each gets the same `n_games`
as each bot — **100** on this run.

## 3. Live-run artifacts — `models/ai_v12_02_winprob_critic` (read-only)

Sentinels first appear at 6M and reach the full 5 at 14M. Latest 3 completed cycles
(`eval_results.jsonl`, `n_games=100` for every edge):

| step | bots (win rate) | sentinels (step → wr) |
|---|---|---|
| 26000016 | rnd 1.00, heur .93, heur2 .95, stall .90, stall2 .95, aggr .89, aggr2 .89, setup .82, setup2 .92 | 24000000→.660, 20000016→.640, 14000016→.670, 8000016→.810, 4000032→.910 |
| 28000032 | rnd 1.00, heur .92, heur2 .90, stall .94, stall2 .96, aggr .89, aggr2 .92, setup .93, setup2 .89 | 26000016→.630, 20000016→.730, 14000016→.710, 10000032→.760, 4000032→.910 |
| 30000000 | rnd 1.00, heur .93, heur2 .84, stall .92, stall2 .95, aggr .92, aggr2 .87, setup .92, setup2 .88 | 28000032→.620, 22000032→.670, 16000032→.720, 10000032→.720, 4000032→.920 |

The evenly-spaced-newest-first rule is visible (newest, then a spread back to the oldest 4000032).
The bot column is the saturation the owner describes: 0.82–1.00. The sentinel column sits at
0.62–0.92 — that is where the resolution is.

**TensorBoard** (`tb/`, 5 event files): `eval/elo` and `eval/elo_ci` each have **15/15 points, one
at every cycle** including all 10 cycles since sentinels appeared at 6M — no gaps:
2000016→1589.9, 4000032→1859.5, 6000000→1938.1, 8000016→1977.6, 10000032→2009.9, 12000000→2033.1,
14000016→2048.8, 16000032→2071.2, 18000000→2085.6, 20000016→2082.9, 22000032→2073.2,
24000000→2060.3, 26000016→2082.5, 28000032→2107.1, 30000000→2088.7. `eval/hodge_width_elo` and
`eval/hodge_cyclic_fraction` also 15/15; `eval/ladder_elo`(+`_ci`) 11 points.

## 4. Refit (a) vs (b) — the direct test

Project functions only (`elo.load_rows(source="log")`, `elo.fit_elo`, 9 anchors pinned,
base 1000). Arm (a) = the same rows with `sentinels=[]` (`dataclasses.replace`), so only the
edge set differs.

| fit at step 30000000 | Elo | CI95 |
|---|---|---|
| (b) bot + sentinel edges | **2088.74** | ±30.05 |
| (a) bot edges only | **1978.45** | ±40.58 |
| **difference** | **+110.30** | −10.53 (tighter) |

Arm (b) **reproduces the recorded TensorBoard value bit-for-bit** (2088.74 / 30.05), confirming
the refit is the live code path. (A 32M cycle landed mid-analysis; on the full 16-row log the same
contrast is 2077.10 ±29.58 vs 1964.12 ±39.28, **+112.98**.)

## 5. Contrast with the dense ladder

`eval/elo` is a **per-cycle, sparse star** — one trainee against 9 saturated bots plus 5 ~100-game
sentinel edges, refit globally each cycle (±30 here), so the newest node is systematically
inflated and retro-adjusts as later cycles land. `snapshot_ladder/ladder.json` is the **dense**
counterpart: 91/91 frozen snapshot-vs-snapshot pairs measured once and permanently, plus each
snapshot's historical bot edges for the anchor, giving se ≈ 8–10 (30000000 → **2063.5 se 9.7** vs
the live 2088.7 ±30.1). Project rules say to quote **the ladder, at run end, at matched snapshot
COUNT** — not the live `eval/elo`.

## 6. Anything muted or missing? — No

Nothing to report as a defect. The two structural absences are by design and documented in code:
sentinel-vs-sentinel edges are absent from `eval/elo` (they are exactly what the dense ladder
supplies), and sentinel edges are excluded from the `hodge` width scope only (`hodge.py:785-787`).

*Everything above is verified from source and artifacts; nothing is inferred except the noted
"only historically" mechanism for sentinel×bot edges, which follows from the `snap:<step>` player
identity at `elo.py:37-39` and is consistent with the 25-player fit.*
