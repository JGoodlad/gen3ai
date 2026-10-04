# Rustboro-era bot anchor base ratings (2026-10-03)

**What this is.** A new base rating for the nine scripted eval bots, the bots the snapshot ladder pins, measured
with the FIXED bots: the F-LF-1 setup step and Curse-as-setup, both 2026-09-29. Every bot-anchored number before
this one, including `data/gen3_bot_elo_anchors.json` (2026-06-06, `1081a8e0`), measured the BROKEN bots and
belongs to the old era. This directory is the measurement only. **The new anchor is NOT installed into `data/`.**
Whether and when to swap it in is a separate decision, because swapping it moves every ladder's bot-anchored
headline.

Scope (owner re-scope, 2026-10-03): the bot base ratings only. No old ladder, Hodge baseline or end-of-run
battery was re-baked, and `models/` was not touched.

## Result: 259,200 games, 3,600 mirrored pairs per edge, all 36 edges

Anchor: `random` pinned at 1000, every other bot free. This is the convention of the existing anchor
(`bot_elo_calibration._fit_and_write`); the ladder then pins all nine bots to these numbers. The CI is the
95 % pair-bootstrap percentile interval over 1,000 replicates. `se_hess` is the independent-games Hessian SE,
and it agrees with the bootstrap.

| bot | rating | 95 % CI | se (boot) | old anchor | Δ vs old |
|---|---|---|---|---|---|
| heuristic2 | **1664.2** | [1656.2, 1673.2] | 4.4 | 1638.8 | +25.4 |
| setup_sweep_v2 | **1656.4** | [1648.1, 1665.6] | 4.5 | 1618.7 | +37.7 |
| aggressive_v2 | **1656.0** | [1648.1, 1665.1] | 4.4 | 1630.1 | +25.9 |
| setup_sweep | **1635.0** | [1627.1, 1644.3] | 4.4 | 1597.8 | +37.2 |
| heuristic | **1619.2** | [1611.1, 1628.4] | 4.4 | 1577.5 | +41.7 |
| staller | **1577.2** | [1569.0, 1586.1] | 4.5 | 1570.9 | +6.3 |
| staller_v2 | **1558.8** | [1550.2, 1567.9] | 4.6 | 1554.9 | +3.9 |
| aggressive | **1549.9** | [1541.8, 1559.2] | 4.5 | 1511.7 | +38.2 |
| random | 1000 (pinned) | — | — | 1000 | 0 |

- **Fit quality:** mean |predicted − observed| score 0.0069, max 0.0274 over the 36 edges.
- **HodgeRank** (triangle scope, 84 triangles, 300-replicate parametric bootstrap): spine 655.8 Elo. Width is
  12.64 raw against a 4.05 null, so the excess is **11.97 Elo, p = 0.0033**. That p is the bootstrap's floor,
  1/301: no null replicate reached the observed width. The cyclic energy is 1.0 % raw and 0.9 % excess.
  **No individually significant 3-cycle.** So the bot ensemble is about 99 % transitive. The small cyclic
  residue is real, not noise, but it is diffuse.
- **Games:** 6,592 draws in all, 264 of them turn-limit timeouts. No unfinished game and no voided pair. Each
  bot sat p1 in exactly half of every edge's games.

## Sizing: chosen from a precision target, not a habit number

- **Target:** a 95 % CI half-width of at most ±10 Elo for every free bot, under the ladder's `random` = 1000 pin.
- **Pilot:** 200 pairs/edge (14,400 games). It gave a worst half-width of 38.7, all bots alike, so the error is
  the common offset to `random`. Scaling as 1/√n, that implied **3,001 pairs/edge**.
- **Chosen:** 3,600 pairs/edge. The extra 20 % margin covers the pilot's own SE noise: the binding edges are
  the ones against `random`, where `random` wins only 1–7 % of games.
- **Achieved:** a worst half-width of **8.98**. On these data the target needs 2,906 pairs/edge.

The SE is almost entirely the common offset to `random`. The CONTRASTS among the eight non-random bots are
tighter than ±9, which is what a ladder that pins all nine actually uses.

## Protocol (`bot_rr.py`)

- **Roster:** `eval_opponent_names()`, the nine eval bots. BaitBot was a training-only bot, never on this
  roster, and is deleted (P11d).
- **Teams:** the trainee's eval team distribution (`rust_eval.build.eval_builders` trainee builder, the default
  pool with a 10 % sample-team bias) for BOTH seats. Teams are drawn by the Rust eval core's per-game seed rule
  (`rust_eval.seeds.draw_team`).
- **Mirrored pairs:** `gen3_mirrored_pairs_v1`. Both games of a pair share one key, so they get the same two
  teams, the same battle seed and the same bot streams, and the second game hands the teams over. **The seat
  alternates by pair:** bot A sits p1 in even pairs and bot B in odd pairs. The mirror cancels team luck, and
  the alternation cancels the seat.
- **Bot streams:** the p2 bot is re-seeded per game exactly as the eval core re-seeds a p2 bot route
  (`seeds.bot_stream_seeds`). The p1 bot gets the same rule on route seed `bot_route_seed(name) + 0x100`.
- **Turn limit:** the core's stall rule. p1 forfeits at its first decision with `turn >= stall_threshold()`, and
  `classify_result` scores that game as a DRAW (timeout).
- **Scoring:** a draw counts as half a win.
- **Sim:** the Rust Showdown port over the in-process bridge (`run_local_battles(impl="rust")`, the worktree's
  own release `sim_bridge`). Each game is played alone at concurrency 1 with its own battle seed, which is the
  Lane H gate's Python path. CPU only, no GPU.
- **Reproducible:** every game is a pure function of `(schedule seed 0, edge, batch, game index)` and the code.
  Replaying one 50-pair unit twice gave identical counts, pentanomial and team counters. Re-fitting from the
  gzipped shards reproduces `result.json` exactly.
- **Code version:** pinned worktree at **`aecccb23`**, the deletion pass's DONE commit, which has the fixed bots.
  The bots never read the damage op, so the concurrent fixed-damage fix (`f0310ee7`) does not bear on this.

### Why the Python bots and not the Rust env core

- **The limit:** the env core's route table seats a scripted bot only at p2. In `rust_env/src/opponents.rs`, p1
  is always answered by the caller, and there is no p1 bot route. A bot-vs-bot game therefore cannot run inside
  the core without a core change.
- **What was used instead:** the PYTHON bots, the reference implementation that the Rust bot port is gated
  against at 0 mismatches (`bots_gate_test`, bank re-recorded with the Curse batches). They were driven by the
  eval core's own per-game seed rule, on the Rust sim. This is the Lane H gate's oracle path.

## Files

| file | what |
|---|---|
| `bot_rr.py` | the round-robin player (resumable; one §0b row per edge × 100-pair batch) |
| `fit.py` | the BT fit with a pair bootstrap and Hodge. `--check` resolves the inputs and is registered in `src/measurements_readout_gate_test.py` |
| `ledger/ledger.*.jsonl.gz` | 1,296 §0b COUNT rows (`gen3_eval_count_row_v1`, purpose `anchor`, regime `c93b27af04745db9`), gzipped closed shards (`eval_ledger.read_rows` reads them) |
| `result.json` | every number above, plus per-edge W/L/D, pentanomial, seats and timeouts |
| `gen3_bot_elo_anchors.rustboro.json` | the new anchor in `data/gen3_bot_elo_anchors.json`'s format. **Not installed** |
| `play.log`, `fit.log` | the run's logs |

Reproduce: `export PYTHONPATH=<checkout>/src; python bot_rr.py play --pairs 3600 --batch-pairs 100 --workers 12
--out ledger` from the repo root (the team loader reads `data/teams` relative to cwd), then `python fit.py`.

### Installing the anchor: a later decision, not done here

```bash
cp designs/research_state/measurements/bot_base_ratings_2026-10-03/gen3_bot_elo_anchors.rustboro.json \
   data/gen3_bot_elo_anchors.json
```

Do not install it while a pinned run is live: a pin isolates code, not `data/`. The swap re-scales every ladder's
bot-anchored headline, and every reader of the old scale then reads a different number.

Two format notes on the new file:

- Its `win_matrix` is the SCORE, with draws as half. The old file's was the raw win rate.
- Its `se` is the pair-bootstrap SD. `load_bot_anchors` reads only `ratings` and `base`.

## The Δ column is NOT a measurement of the bot fix alone

Old and new differ in more than the bots:

| | old anchor | this measurement |
|---|---|---|
| sim | Node bridge | Rust sim |
| seeding | unseeded | seeded |
| pairs | unmirrored | mirrored, seat alternating |
| team builder | each bot's own builder | the trainee builder, per-game seeded |
| turn limit | none (only the 1000-turn runaway cap) | the stall rule |
| draws | counted as wins for the SECOND-named bot of each `(a, b)` result; `fit_pairwise` gives `b` the `g − wins_a` remainder | half a win |

The bots whose code did NOT change (`staller` +6, `staller_v2` +4, `aggressive` +38, `aggressive_v2` +26) moved
too, so part of every Δ is protocol and offset. The four fixed bots moved by +25 to +42.
