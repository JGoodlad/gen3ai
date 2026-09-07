# Reusing eval-cycle edges for the DENSE ladder — investigation

**Headline:** the ladder is already dense (105/105 pairs), and `fit_ladder` **already reuses**
the eval sentinel edges — that is the problem, not the opportunity. The eval edge and the ladder
edge for the *same pair* differ by **+8.9 pp [+7.0, +10.7] in the newer snapshot's favour**
(60 paired pairs, live run), because eval plays greedy-trainee vs **stochastic** sentinel while
the ladder plays greedy-vs-greedy. Mixing them today inflates the top of `ladder.json` by
**+21 to +29 Elo**. Reuse is cheap; it is not yet *legitimate*.

## 1. Overlap table — `models/ai_v12_02_winprob_critic` (15 snapshots, 4M–32M, read-only)

| quantity | count |
|---|---|
| eval cycles in `eval_results.jsonl` | 16 (2M–32M), `n_games=100`/edge |
| distinct (snapshot, snapshot) pairs from trainee×sentinel rows | **60** (6,000 battles) |
| pairs possible among the 15 pool snapshots | **105** |
| covered by eval alone | **60 / 105 (57%)** |
| measured by the ladder updater (`games.jsonl`) | **105 / 105**, 10,600 battles |
| **pairs measured TWICE** | **60** — 6,100 ladder battles spent re-measuring |
| eval pairs with an endpoint outside the pool | 0 |

Per promotion (`update_for_promotion`, `snapshot_ladder.py:288-293`): the new node plays every
other pool member; **5 of those pairs (500 battles) were just played by that cycle's own eval**
(the 5 sentinels), 100% of them while the pool is < 5. Steady state at pool 15: 5 of 14 pairs
= **36% of the per-promotion tax is duplicated effort**.

## 2. Are the two edge kinds the same edge?

**(a) Same weights — YES, bit-exact, zero step gap.** The eval cycle freezes the live model to
`.eval_runs/step_<s>/snapshot.zip` (`selfplay_callback.py:402-404`) and promotion copies **that same
file** into the pool (`selfplay_callback.py:784` → `snapshot_pool.py:229-242`, `shutil.copy2` to
`snapshots/snapshot_<s>.zip`). Sentinels are pool entries (`selfplay_callback.py:407` →
`snapshot_pool.py:312-323`), so the ladder's `_snapshot_zip` (`snapshot_ladder.py:67-68`) loads the
identical bytes. **This half of the premise is sound.**

**(b) Teams / seating / seeds / game count — three differences.**
- *Teams*: ladder gives **both** players `Gen3Teambuilder(all_teams, bias_teams=sample_teams,
  bias_prob=0.1)` (`snapshot_ladder.py:246-247`). Eval gives the **trainee** that biased builder
  (`eval_worker.py:75`) but the **sentinel** the unbiased `opp_tb = Gen3Teambuilder(all_teams)`
  (`eval_worker.py:211`, `:149`) — an asymmetric 10% tilt toward the curated sample teams,
  favouring the trainee.
- *Seating*: both put the newer snapshot on sim side p1 (`snapshot_ladder.py:251-254`;
  `eval_worker.py:184` via `_play`, `local_battle_runner.py:167-182`). Consistent, but
  counterbalanced in neither.
- *Seeds*: **neither is seeded** — `run_local_battles(..., seed=None)` on both paths. Independent
  IID samples, not paired. Concurrency differs (ladder 4, `snapshot_ladder.py:288`; eval 1 here,
  `--eval-concurrency-per-worker`), which matters only for reproducibility, which neither has.
- *Game count*: **100 both** on this run (`snapshot_ladder_games=100`; eval `n_games=100`).

**(c) Greedy vs sampled — the decisive mismatch.** Ladder: both players `stochastic=False`
(`snapshot_ladder.py:249`, "greedy = a stable frozen yardstick"). Eval: trainee greedy, sentinel
`stochastic=not sentinel_greedy, temperature=self_play_temp` (`eval_worker.py:153`); this run has
`eval_sentinel_greedy=False`, `self_play_temp=1.0`, so **every eval sentinel edge is greedy-vs-
temperature-1.0**. The flag's own help calls it "the greedy-trainee-vs-stochastic-sentinel
handicap" (`parser/eval_subprocess.py:129-136`), and `elo.py:44-46` documents it as a KNOWN CAVEAT.

**Measured, paired over the 60 shared pairs:** eval win rate − ladder win rate for the newer
snapshot = **+0.0887, sd 0.0735, 95% CI [+0.0701, +0.1073]**. Systematic, not noise.

**Does `fit_ladder` already mix them? YES** — `snapshot_ladder.py:168-174` folds
`_rows_to_results(load_rows(...))` in, and that generator yields **both** bot edges *and* sentinel
edges (`elo.py:382-385`). Refit of the live run, same code path, only the sentinel edges toggled:

| fit | 26M | 28M | 30M | 32M |
|---|---|---|---|---|
| as shipped (games + bot + eval-sentinel) | 2046.2 ±9.1 | 2059.9 | 2051.5 | **2061.3** se 9.4 |
| games + bot edges only | 2024.8 ±10.0 | 2037.2 | 2029.0 | **2032.0** se 10.0 |

**+21 to +29 Elo, largest on the newest node** — it compounds the known newest-node inflation the
matched-snapshot-COUNT rule exists to control (`snapshot_ladder.py:136-143`).

## 3. Options

| | change | saves | costs / risks | tests |
|---|---|---|---|---|
| **A. Treat eval-covered pairs as covered** | `_measure_missing` (`snapshot_ladder.py:258-265`) or the target-pair build at `:291-292` unions `load_games` with eval-derived pairs | 5 pairs = **500 battles/promotion**; 6,000 on this run | makes the **biased** eval edge the *only* measurement of 57% of pairs; the +8.9pp mismatch stops being averaged against a clean edge and becomes the estimate | `test_measure_once_contract` (`snapshot_ladder_test.py:31`) must learn the second source |
| **B. Seed `games.jsonl` from eval rows at promotion** | one `_append_game` loop in `_spawn_snapshot_ladder_update`/`update_for_promotion` | same 500/promotion | same bias, plus it **double-counts**: `load_games` sums duplicates (`:86-88`) *and* source (2) re-adds the same rows, so each eval edge lands twice; also breaks the "a measured pair is NEVER replayed" invariant's meaning (a pair could be "measured" without a ladder game) | as above, plus `test_games_log_accumulates_and_is_symmetric` |
| **C. Rotate sentinels so evals alone go dense** | `snapshot_pool.sentinel_entries` (`snapshot_pool.py:312-323`) | could reach 105/105 from evals with zero extra battles | changes what `win_rate_vs_pool` / the promotion gate measure (they average the sentinels) → curriculum + promotion side-effects; still the biased measurement; loses the evenly-spaced monotonicity read | `selfplay_callback` + pool tests |
| **D. Make the two measurements the SAME first** | launch with `--eval-sentinel-greedy` (+ `--promote-threshold 0.55`, already auto-lowered) and give the eval sentinel the trainee's biased teambuilder (`eval_worker.py:211`) | nothing yet — it *unlocks* A | metric discontinuity vs prior cycles (the flag's help says so); a run-config change, not a code change, for the greedy half | none change |

## Recommendation — **D then A, in that order; A alone is not safe.**
The eval edge and the ladder edge are provably different measurements of the same frozen pair
(+8.9pp, CI excludes zero), so reusing eval games today would replace 57% of the dense matrix with
a systematically trainee-favouring estimate — and the fit already carries that bias, worth +29 Elo
on the newest node. Flipping `--eval-sentinel-greedy` on the next run (plus the one-line
teambuilder symmetry in `eval_worker.py:211`) makes the two edge kinds *the same experiment*, after
which A is a ~5-line change in `_measure_missing` that saves 500 battles per promotion for free.
Independently and immediately: `fit_ladder` should stop folding **sentinel** edges from source (2)
while the mismatch exists — keep the bot edges (the anchor) and drop `snap:`-vs-`snap:` rows — which
costs nothing, removes a known bias from every `ladder.json`, and is a 2-line filter at `:170-172`.

## 4. Premise check
- **Right:** the pool snapshot and the eval-cycle model *are* the same weights at the same step —
  no drift, no gap. And dense ranking **is** already supported and complete on this run (105/105).
- **Right:** the evals really do already play 57% of the pairs, and 6,000 ladder battles on this
  run were spent re-measuring them.
- **Wrong (the important part):** "we already know it from its last evaluation" assumes the eval
  edge is the same quantity. It is not — greedy-vs-stochastic plus an asymmetric teambuilder make
  it +8.9pp optimistic for the newer snapshot. What is **not** reused is the *battles*; what **is**
  already (silently) reused is the *edge*, in the fit, at a +21…+29 Elo cost.
