---
name: project_eval_item_workstealing
description: "Battle-level (item-granularity) eval work-stealing — eval_sharding/ package, exact aggregation, Glicko-ready rating seam; SHIPPED 2026-06-08 as 2158784 (rebased onto main incl. the replay-HTML feature; reward_factory re-applied)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1b8ca591-c6f8-4578-89a9-f0eebe07034e
---

Built 2026-06-08 (worktree, NOT shipped — no /gen3ai-ship yet): replaced opponent-level eval
work-stealing (one worker claims a whole opponent's 100 games → long tail when #opponents <
#workers) with **battle/chunk-level units** so any idle worker drains a straggler's remaining games.

**Package `src/agents/training/eval_sharding/`** (deep interface, 4 small files): `units.py`
(EvalItem + ShardUnit + pure `plan_units`, LPT round-robin order), `results.py` (ShardResult raw
additive metrics + `aggregate` + `td_tail` MOVED here, re-exported by eval_callback — one-way dep
eval_callback→eval_sharding), `pool.py` (`ShardedEvalPool`: parent `write_plan`/`collect`, worker
`from_plan`/`claim_next`/`publish` — hides all lock/shard-file mechanics), `__init__.py`. Worker
rewritten: `eval_worker._play_unit` (fresh trainee+opponent per unit, per-worker model cache for
sentinel/fixed by path). Flag `--eval-shard-games` (default 25 → ~4 shards/opp; ≥ EVAL_GAMES =
old behavior). `n_workers` now capped by UNIT count not opponent count. Both callbacks build
EvalItems → write `plan.json` (single source of truth, no two-site reconstruction) → spawn.
`merge_eval_results` delegates to `pool.collect`, returns the SAME `merged` shape (+ additive
`counts`/`coverage`) so ALL downstream (record_per_opponent/build_bot_eval_block/record_elo/pool+
externals) is UNTOUCHED.

**Exactness:** win_rate/reward/ep_len pool bit-exact vs unsharded. KEY subtlety: reward pools over
`n_episodes` (reward-tracked count) but ep_len/win_rate over `n_finished` — DIFFERENT denominators,
matching the two unsharded helpers (`mean_episode_reward` vs `_mean_episode_length`); exact because
each numerator sums against its own denominator (a battle can finish w/o a finalized reward → n_ep <
n_fin). ShardResult carries BOTH. `td_resid_tail`: pool raw δ then ONE td_tail (CVaR can't be
averaged) — aggregation exact, but the captured SAMPLE shifts with shard count (forensic quota is
per-unit, scaled max(1,⌈q/shards⌉)) so it's a sampled diagnostic, not per-game exact. Forensic files
namespaced by per-unit trace_tag (`{outcome}_s{shard}_{idx}`) to avoid collision. Per-cycle run_dir
wiped at cleanup + cleared at launch → no lock/shard/plan leak across cycles.

**Rating seam `rating.py`** (the "extend to Glicko" ask): MatchRecord/RatingResult/RatingModel
(batch protocol) + `BradleyTerryRating` (delegates to elo.fit_pairwise, byte-identical to live fit,
TESTED) + `eval_rows_to_match_records`. DELIBERATELY not wired into live record_elo (zero risk;
Glicko-2 is sequential-period so needs a SequentialRatingModel sibling — documented, not built).
`eval_results.jsonl` enriched with exact per-opponent `counts` (additive, the real Glicko-enabler +
partial-coverage fidelity). See [[project_throughput_profile]] (eval-latency context),
[[feedback_prober_self_improvement]].

Process: built via ultracode workflows (understand → 4-critic design red-team → implement →
adversarial review). Review confirmed the n_ep/n_fin denominator divergence is a FALSE-positive on
correctness (matches unsharded) but a real doc/coverage gap → added
`test_sharded_aggregation_diverging_denominators`. Tests: `eval_sharding_test.py`,
`rating_test.py`, `eval_sharding_fuzz_test.py` (real bridge battles through the real worker). Full
suite 2086 green. Also shipped earlier same day: `--eval-concurrency-per-worker` (intra-worker
latency-hiding, ~2× on spare cores; commit 2651d9c).

PARITY REVIEW (2026-06-09, after 2158784 shipped) — a systematic old-vs-new sweep by two bug CLASSES
caught 3 real divergences, ALL latent in shipped 2158784, now fixed+tested (uncommitted, pending ship):
(1) **forensic "first N" quota inflated** with shard count — `max(1,ceil(quota/n_shards))` captured every
battle at fine granularity → `quota_share(total,n_parts,index)` distributes the cap (Σ==cap); (2) **prober
couldn't parse sharded trace filenames** — `_FNAME_RE` didn't match the `s<shard>_` infix so every sharded
trace read outcome="?", breaking win/loss filter + `scan --outcome loss` (the loss-investigation tool) →
regex + `BattleTrace.shard` + `_short_id` now carry the shard (old traces shard=None, unchanged); (3) **live
ELO ignored the stored `counts`** — `_rows_from_log` reconstructed wins from `win_rate×nominal_n_games`, so
partial shard coverage overstated BT certainty → `EvalRow.bot_counts` populated from the jsonl `counts`,
`_rows_to_results` uses exact (n_won,n_finished) (byte-identical at full coverage). The TWO bug CLASSES to
watch for any future sharding: **(A) a per-opponent global quantity recomputed/replicated per-shard** (quota),
**(B) a filename/format consumer not taught the sharded format** (prober). Principle: workers emit RAW
composable per-shard data; every per-opponent decision happens ONCE at the aggregation boundary.
DELIBERATELY NOT changed (parity-preserving): `win_rate_vs_bots`/`bot_mean` stays an UNWEIGHTED mean of
per-opponent rates (identical to pre-sharding at full coverage; games-weighting would DIVERGE). Ruled out:
result__ has no readers, duration_sec aggregate semantic preserved, stall HTML is battle_tag-unique, teams
use independent per-process RNG (same distribution). Architecture decision: keep the decentralized static-
plan + O_EXCL pull-claim + central-reduce design (NOT a live orchestrator — the parent is the training process
and must stay non-blocking; neither bug was caused by decentralization).

Dead-code pruned (eval_one_matchup/run_eval/claim_next_opponent/_mean_episode_length removed; superseded
by _play_unit/pool.claim_next/episode_length_sum). SHIPPED as a single commit `2158784` after rebasing
onto main's 5 newer commits — KEY reconciliation: main's cf043dc made eval_worker thread a
**reward_factory** (from model_config.json, so eval scores with the TRAINED reward incl.
bias_redesign/draw_penalty); my full eval_worker rewrite had DROPPED it, so I re-applied it to
`_play_unit`/`_run` (build_eval_players now REQUIRES reward_fn_factory). main's 37f741b (replay-HTML:
write_battle_record also emits `<prefix>_replay.html`) lives in battle_recorder.py (untouched) → composes
with sharding for free; verified end-to-end that sharded eval writes shard-namespaced
`{outcome}_s{shard}_NNN_replay.html`. main's 6334574 (artifact_retention keep_stalls/keep_crashes) auto-merged.
Full suite 2105 green post-rebase.
