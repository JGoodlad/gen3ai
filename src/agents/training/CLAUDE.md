# CLAUDE.md — Training (`src/agents/training/`)

Callbacks, reward manager, episode/turn tracking, stall detection, the belief/value auxiliaries and
the bot-eval pipeline. **How to launch training** (commands, flags) lives in the root `CLAUDE.md`
→ Training / Launcher and in [`designs/ops/training_runbook.md`](../../../designs/ops/training_runbook.md);
this file documents the subsystems' internal design. The `TurnDelta` fold and the
LiveView/TurnView/LegalActions read-models it consumes are in `src/agents/battle/CLAUDE.md`; the
obs-build performance gate is in `src/agents/observation/CLAUDE.md`.

## Where the detail is — the topic map

**This leaf keeps each subsystem's heading, the rules and hazards an agent must know BEFORE
touching it, and a pointer. `designs/training/` owns the detail and carries the same
always-current obligation as this file — update the topic doc in the same pass as the code.**

| I am about to touch… | Read |
|---|---|
| the reward registry, PBRS (deleted), the no-progress clock | [`designs/training/reward.md`](../../../designs/training/reward.md) |
| the PPO package's module map or a source-level pin on `train()` | [`designs/training/ppo_step.md`](../../../designs/training/ppo_step.md) |
| the learner's GIGO gates (K9): the LEARNER GOLDEN, behaviour-policy consistency, fail-closed non-finite, and the non-finite AUDIT; K6's CUDA memory TREND (`cuda_memory_trend.py`, a leak detector for a clean early stop, never a gate) | [`designs/training/learner_gates.md`](../../../designs/training/learner_gates.md) |
| the learner's DECLARED LIFECYCLE (K6): the FREEZE GUARD, optimizer state declared at startup, `@startup_builder` | [`designs/training/learner_lifecycle.md`](../../../designs/training/learner_lifecycle.md) |
| bot eval, the untaught meter, the critic gate, ELO / the ladder / Hodge, the baseline registry | [`designs/training/eval_and_rating.md`](../../../designs/training/eval_and_rating.md) |
| self-play, the snapshot pool, stable opponents | [`designs/training/self_play_and_pool.md`](../../../designs/training/self_play_and_pool.md) |
| exploiter mode, the warm start, distillation + the off-slice anchor | [`designs/training/exploiter_and_distillation.md`](../../../designs/training/exploiter_and_distillation.md) |
| team-side PFSP, per-team win-rate tracking | [`designs/training/team_curriculum.md`](../../../designs/training/team_curriculum.md) |
| `--critic`, TD-aux, the 250-turn cap (PopArt, the value-tail weight and the value-dist head are DELETED, config v131) | [`designs/training/critic_and_value_losses.md`](../../../designs/training/critic_and_value_losses.md) |
| the win-prob head (its PBRS routes were DELETED, config v131) | [`designs/training/winprob_head_and_pbrs.md`](../../../designs/training/winprob_head_and_pbrs.md) |
| the training-side value sidecar, its two flags, its cost, `main.ops.value_sidecar_read` | [`designs/training/value_sidecar.md`](../../../designs/training/value_sidecar.md) |
| any supervised belief loss, or the opponent-class label weight | [`designs/training/belief_losses.md`](../../../designs/training/belief_losses.md) |
| gradient accumulation, the noise scale, the DOSE, `--fork-lr`, `--adaptive-batch` | [`designs/training/step_size_and_batch.md`](../../../designs/training/step_size_and_batch.md) |
| the MatchupSpec, run-spec resolution provenance, LINEAGE, TB inheritance | [`designs/training/matchup_and_lineage.md`](../../../designs/training/matchup_and_lineage.md) |
| the TB census detail, capacity telemetry, grad balance, `signal/`, the scaffolding gauge | [`designs/training/telemetry_scalars.md`](../../../designs/training/telemetry_scalars.md) |
| the counterfactual audit, the cf label plumbing, the prefix-sharing materializer | [`designs/training/cf_grounding.md`](../../../designs/training/cf_grounding.md) |
| a SUPPLY GUARD — any live lever that must deliver (self-play pool, PFSP, team-PFSP, fork arm, search teacher), `--supply-starve-cycles`, `FatalConfigError` | [`designs/training/supply_guards.md`](../../../designs/training/supply_guards.md) |
| search-as-teacher, OPD, the win-prob one-ply teacher | [`designs/training/search_teacher.md`](../../../designs/training/search_teacher.md) |
| the FORK ARM — contested-state forks, the branch rows, the CRN, the `fork/` family | [`designs/training/forks.md`](../../../designs/training/forks.md) |
| the stall-tail harvest / head-repair pipeline | [`designs/training/stall_tail_harvest.md`](../../../designs/training/stall_tail_harvest.md) |
| either `--compile-*` flag, BLAS pinning | [`designs/training/compile_flags.md`](../../../designs/training/compile_flags.md) |
| `stats.py`, the replay-imputation probe | [`designs/training/offline_meters.md`](../../../designs/training/offline_meters.md) |
| the RUST COLLECTOR — the rollout on the M5 Rust env core, the complete-game buffer, staleness, the keyed draw (`rust_rollout/`, `keyed_draw.py`, `rust_vec_env.py`) | [`designs/training/rust_collector.md`](../../../designs/training/rust_collector.md) |

Closed history — **do not update it, and do not re-derive a plan from it**:
`designs/research_state/claude_md_archive/training_leaf_faint_attribution_history.md` and
`designs/research_state/claude_md_archive/training_leaf_deleted_subsystems_history.md`.

## TensorBoard export census — every scalar, and its CURRENCY

> 🔒 **Two anchors here are PINNED to this file** by
> `tb_relevance_test.py::test_the_census_table_carries_an_era_column`: the string
> `gen3_tb_relevance_v1` and the heading *What to watch on a WIN-PROB run*. Keep both. Everything
> beneath them — the site/tag counts, the group table, the NOISE / REDUNDANT / CONDITIONAL
> classification, the five per-group tag tables and the annotated dashboard — is in
> [`designs/training/telemetry_scalars.md`](../../../designs/training/telemetry_scalars.md).

**THE FIRST QUESTION ABOUT ANY SCALAR HERE IS WHAT UNIT IT IS IN**, because this trainer runs
value quantities in **three different currencies at once** (a fourth, PopArt-normalized, left with PopArt) and two of them look like floats:

| currency | is | who is in it |
|---|---|---|
| **RAW REWARD** | the units `--victory-value` is in, undiscounted | every `reward/*` term, `--draw-penalty` |
| **RAW SHAPED RETURN** | `Σγᵏr` in raw-reward units | `train/return_*`, `rollout_buffer.{values,returns}`, `train/explained_variance` |
| **PROBABILITY** | `[0, 1]`, outcome units, undiscounted | every `win_prob/*`, `cf/*` labels, `eval/win_rate_*` |
| ⚠️ **PROBABILITY, under `--critic winprob`** | the same `[0,1]`, but it is now ALSO what `rollout_buffer.values` / `returns` / `train/explained_variance` are in | the row above **plus** `train/return_*`, `train/explained_variance`, `train/value_loss` (raw) |

⚠️ **A number is only comparable to another number in the SAME currency.** PopArt (and with it the
`popart/*` tags and the normalized-return currency) was DELETED (deletion pass L1, config v131), so
`train/value_loss` is raw in every run; `designs/learning/popart_value_scale_and_currencies.md` is the
historical background for old traces.

🚨 **`--critic winprob` COLLAPSES the currencies into one, which changes what several tags
MEAN without changing their names** (`gen3_winprob_critic_mode_v1`). The reward is the terminal WIN
INDICATOR and `V(s) = sigmoid(win_head logit)` — so `train/return_mean` reads a
win RATE, `train/value_loss` is an MSE in probability units (a diagnostic; its term is
dropped from the loss) and `train/explained_variance` is EV in the P(win) currency. **A `winprob`
run's `train/*` value tags are not comparable with a `shaped` run's**, and nothing in the tag names
says so — read the run's `🎯 [CRITIC]` startup line first. The one tag that IS comparable across
the two is the `win_prob/` family, which was in probability units all along.

**ERA RELEVANCE — a tag whose SOURCE is absent is not emitted (`gen3_tb_relevance_v1`).** The
`--critic winprob` era changes no tag NAME but removes the SOURCE behind several of them, and the
recorders kept publishing — flat constants and byte-identical duplicates a reader cannot tell from
a measurement. Classified from the live arm's own tfevents (216 tags): **166 LIVE · 31 NOISE, all
now GATED · 19 REDUNDANT · 0 DEAD**, plus ~48 correctly-silent CONDITIONAL ones. 🚨 **The gate is
on the SOURCE, never on the value**, so a shaped run's tag set stays byte-identical and a dead
source leaves a GAP rather than a confident number. ⚠️ **A run pinned before
`gen3_obs_margin_unconditional_v1` carries a degenerate `win_margin`** — read its whole
`win_prob/*contested*` family as absent.

### What to watch on a WIN-PROB run — the 28-tag dashboard

The first two blocks are in PROBABILITY units, which is the era's whole point. Read the run's
`🎯 [CRITIC]` startup line first — a `shaped` run's `train/*` value tags are not comparable with
these. Each tag's currency and reading rule: the topic doc.

**Is it getting stronger?** `eval/elo` + `eval/elo_ci` (the only cross-run number) · `eval/win_rate_vs_bots` (saturates — read a FALL as an alarm) · `eval/win_rate_vs_pool` (pinned near 0.50 by the gate; leaving it is the news) · `signal/outcome_win_rate_bots` · `_pool` · `rollout/ep_rew_mean`.

**Is the critic HONEST?** **`win_prob/critic_resolution`** (the G1 PRIMARY meter, HIGHER is better) · `critic_reliability` · `critic_brier` · `critic_skill` · `critic_uncertainty` · `critic_base_rate` · `critic_decomp_residual` (must sit at ≈0) · `win_prob/ece` · `mce` · `rel_gap_b0…b9` (a NaN bin is a HOLE, never a zero error) · **`win_prob/start_gap`** (the PAIRED episode-start read; **positive = optimistic at the opening board**) · `train/explained_variance` · `win_prob/coverage` (a fall here invalidates every line above it).

**Is the OPTIMIZATION healthy?** `train/approx_kl` · `train/learning_rate` + **`train/dose_rate`** · `clip_fraction` · `grad_norm` · **`grad/value_policy_logratio`** · `grad/{policy,value,aux}_share` · `train/noise_scale_ratio_policy` (read BESIDE `train/noise_scale_ratio`, never alone) · `signal/adv_raw_mean` **as a ratio to** `adv_raw_std`.

**The G7 KILL condition and the two GIGO guards.** **`rollout/ep_len_mean`** and **`signal/draw_rate`** are PRIMARY endpoints, not monitored ones — a `[0,1]` critic cannot represent "a timeout is worse than a loss", so stalling is this era's registered failure mode. **`reward/untracked_abs_mean` must read exactly 0.0** (the startup composition census and the reward folds agree) and **`train/return_abs_max` must read exactly 1.0** (the reward stream IS the win indicator the `V = P(win)` identity rests on). Plus `time/fps`.

**What to read when the contested split is ABSENT:** the family is gated off rather than duplicated
on a spread-free margin, so the contested-vs-blowout question is answered offline by
`python -m main.scaffolding_gauge --reliability --reliability-reweight` and by `main.critic_gate`.

## The PPO step (`instrumented_ppo/`) — and the FOLD ORDER contract

`instrumented_ppo` is a PACKAGE whose `__init__.py` is a pure re-export hub; the module map and the
two source-pin rules (a pin that says "in `train()`" should read `ppo.train_step_source()`; a
monkeypatch follows the SYMBOL) are in
[`designs/training/ppo_step.md`](../../../designs/training/ppo_step.md). **The contract below stays
here, because it is the thing a fold edit must not get wrong.**

**THE FOLD SEQUENCE is TWO straight lines** (K8, `gen3_learner_micro_step_v1`): steps 1 to 3a below
are the body of ONE function, `instrumented_ppo/micro_step.micro_step` — the compile REGION R1 under
`--compile-trainer` (`fullgraph=True`), eager otherwise — and `train()` folds the steps after 3a onto
R1's loss as the DECLARED EAGER TAIL, in order. Each part is straight-line source and
`instrumented_ppo_hub_contract_test.py` pins both orders. 🚨 **R1 is a static-shape program**: no
host read (`.item()`, `float(t)`, `bool(t)`), no boolean-mask indexing / `nonzero` / `bincount`, no
Python branch on a tensor value, no numpy — a diagnostic is a `(value, weight)` pair of 0-d tensors
(weight 1.0 exactly where the old fold appended to its list), and `train()` reads ALL of a micro-batch's
diagnostics in ONE host read (`micro_step.pack`). The belief losses' static twins are
`belief_bank_static.py`; the opponent-intent block's is `instrumented_ppo/intent_fold.py` — the legacy
`belief_bank` / `opp_intent` functions stay as the REFERENCE they are pinned equal to (float64 to
1e-12). A new term on the production surface belongs in R1, written to these rules; anything else
joins the tail in contract order. What moved out of `train()` before K8 is everything AROUND the
sequence: the pre-loop setup (`train_setup`, incl. R1's static flags `_micro_static` and its declared levers `_r1_levers`), the metrics
export (`metrics_export`) and the per-rollout probes (`rollout_probes`).

🚨 **THE LOOP AROUND `train()` IS OURS TOO** (`gen3_owned_ppo_loop_v1`, `src/agents/training/instrumented_ppo/loop.py`;
[`designs/endstate/design_own_ppo_loop.md`](../../../designs/endstate/design_own_ppo_loop.md)): `learn()`
is the declared `LOOP_PHASES` table, and `_setup_learn` / `dump_logs` / the Python core's collect are
vendored from sb3 operation for operation (hash-pinned). Three things an edit must not break: the
**dump stays BEFORE the update** (update k's `train/*` is stamped after rollout k+1 — the archive's TB
convention and the KL controller's logger read); **`learn` > `collect` | `update` are the DECLARED
HOOK POINTS** (`agents/training/loop_hooks.py`, `gen3_declared_loop_hooks_v1`: K6's freeze guard and
the compile sentinel REGISTER there, outermost first by `HOOK_OWNERS`; the table freezes at training
start and a late, duplicate or undeclared hook is FATAL_CONFIG — never reassign a learner's bound
method to hook it); the **Python collect keeps sb3's local names and
fires `on_step` before `rollout_buffer.add`** (`self.locals`, `buf.pos`). `own_ppo_loop_test.py` holds
it EXACT against upstream; `GEN3AI_PPO_LOOP=sb3_reference` is the A/B test seam. 🚨 **REGIME BOUNDARY
(`gen3_eval_dump_isolation_v1`, 2026-10-01):** an eval cycle's mid-rollout `logger.dump(step)` used to
CLEAR the previous update's `train/*`, so the KL→LR controller (and RankTripwire, DistillStop, the
DistillAnchor dual) skipped one reading per eval cycle — 5% of N0's updates. Both eval callbacks'
`_collect_pending` now run under `logger_scope.isolated_dump`, so the cycle dumps only its own scalars.
A live-controller run from that commit onward is not comparable with an earlier one on its LR / dose
trajectory (`designs/training/step_size_and_batch.md`). A new callback that dumps the logger
mid-rollout MUST use the same decorator. **Seeding is OURS too** (`gen3_owned_seeding_v1`, `OwnedLoop.set_random_seed`):
sb3's draws in sb3's order and NO cuDNN flag — sb3 set the process-wide `cudnn.deterministic=True` on
every CUDA construction and load (a nominal regime boundary: 0 cuDNN kernels run in a production update).
🚨 **No global RNG is SEEDED after the freeze** (`gen3_no_global_reseed_v1`, `global_rng_guard.py`): `LearnerFreeze` arms a guard
on `random.seed` / `numpy.random.seed` / `torch.manual_seed` and kin, so a seed is `GlobalReseedError` (FATAL_CONFIG) naming its
site. An opponent / reader load (`InferenceMaskablePPO`) never seeds and builds inside `isolated_global_rng()`. Until 2026-10-02
every load re-seeded to the snapshot's seed, which replayed the minibatch permutation (rust core) and every worker's team draws
(python core). A stream that must repeat owns a generator. Static twin: `src/global_rng_seed_gate_test.py`. Detail:
`designs/training/learner_lifecycle.md` "No global reseed after the freeze".

**K9 — the learner's GIGO gates** ([`designs/training/learner_gates.md`](../../../designs/training/learner_gates.md)).
🚨 **`learner_golden_test.py` pins what ONE update computes** — exact post-update parameter bytes and
every loss, per torch build — so ANY change to the fold, a term, a coefficient default or the step
fails the routine gate until someone re-records deliberately: `python -m agents.training.learner_golden
record --reason "..."` under EVERY interpreter with an entry (never a routine step). Every non-finite
loss / gradient / buffer value / KL is `main.exit_codes.NonFiniteLearnerError` (tagged `[Learner]
FATAL`; exit 4, the launcher does NOT restart) BEFORE the optimizer moves anything
(`instrumented_ppo/learner_gates.py`) — never a `nan_to_num`, a NaN-mask on a trained
quantity or a skipped step; a new term must reach the assembled loss (or carry its own check), and a
`where(isfinite)` on a label is a NaN hide unless it means `-inf` (use `isneginf`).

🚨 **K6 — THE LEARNER FREEZES at the first rollout of `learn()`** (`learner_lifecycle.py`, [`designs/training/learner_lifecycle.md`](../../../designs/training/learner_lifecycle.md)): after it, a NEW optimizer, `nn.Parameter`, module, buffer or optimizer-state entry anywhere in the learner's graph is `LazyAcquisitionError` (`[LearnerLifecycle] FATAL`, exit 3, not restarted) naming the object and its construction site. Build anything the steady state uses at STARTUP — in `_build` / `_setup_model` / an `__init__`, or a function marked `@lifecycle_decl.startup_builder` that the startup path runs — never on the first update. Adam/AdamW state is declared at startup (`declare_optimizer_state`, bit-identical to torch's lazy init). Every run logs a CUDA memory LEDGER by startup step (`cuda_ledger.py`, `<run_dir>/cuda_ledger.json`) and per-update peaks (`lifecycle/cuda_*_peak_*`, `lifecycle/device_batch_mib`); a rust-core pool refresh is a DECLARED LOAD (pool on the CPU; `checked_slot_load` refuses a load that allocates). On CUDA the MEMORY half rides the same attach (`CudaMemoryWatch` over `cuda_memory_trend.py`): a sample after every rollout and update, the OOM projection logged at every window (`[CudaMemTrend]`, TB `lifecycle/cuda_*`), and only a SUSTAINED leak projecting an OOM inside 25 updates stops — `CudaMemoryLeakError`, checkpoint (`final_model_exception.zip`) then exit 6 (`FATAL_CUDA_LEAK`), which the launcher restarts from that checkpoint at most twice per session; a step-up or fragmentation never does. The STATIC twin is `src/learner_lifecycle_gate_test.py` (routine, EMPTY allowlist): a
construction in a training-step path outside a `@startup_builder` / `__init__` / `_build` / `_setup_model`
fails the gate before it can fail a run. 🚨 **A CUDA STREAM / GRAPH / GRAPH POOL is a startup acquisition
too** (the gate's `cuda_resource` kind, `gen3_staged_compute_stream_v1`), and for it a bare `__init__` is
NOT an exemption — only `@startup_builder` / `_setup_model` / `_build`: the caching allocator keeps a
cache per stream, so the staged batch's per-update side stream (built in a helper's `__init__`) stranded
+1.86 GiB of reserved over 31 updates of sizing arm B, invisible to every segment counter.

**`train()` carries BENCHMARK-ONLY phase marks** (`gen3_learner_phase_hook_v1`): ~14 lines of
`if _ph is not None: _ph("<phase>")`, `_ph` read ONCE per call from `instrumented_ppo/phase_hook.py`
(None in production). A new mark must use exactly that guarded one-line form and a name in
`phase_hook.PHASES`; `learner_benchmark_test.py` pins the guard, the names, and that installing a
hook changes no parameter. The one consumer is `learner_benchmark.py` (where an update's wall time
goes — `designs/ops/testing.md` → Benchmarks).

Per minibatch (1 to 3a inside R1):

1. the upstream PPO loss (`policy_grad_coef·policy_loss + ent_coef·entropy + vf_term` — `--policy-grad-coef`
   scales ONLY the clipped surrogate, never entropy/value/aux; at the 1.0 default the UNSCALED
   `policy_loss` tensor is used, byte-identical to upstream, and 0.0 removes the policy-gradient
   term alone — the arm-F pure-distill/aux phase. Training-only, the `td_aux_coef` provenance
   class: recorded, `_resolve`-inherited on a flagless resume, never gated)
2. the belief bank — species/moves aux, opponent intent (+ set-valued β), move / spread /
   nature-EV / HP-type / item belief, move-latent
3. (3a) the win-prob BCE — the last R1 term; then (3b, the tail's first) the
   CF-twin on-policy mirror
4. (retired — the value-dist HL-Gauss CE was deleted with the dist head; the numbering below is unchanged)
5. the distill family — the policy term (full KL, or the top-K/action-CE form with the optional
   advantage gate under `--distill-target action` — gen3_distill_target_gate_v1), value MSE, the
   value-feature hint
6. search-teacher AWR, then OPD
7. **TD-AUX**
8. **the counterfactual block** — cf-winprob, cf-evidential, cf-twin, cf-shadow, **q-winprob**

**No flag combination reorders these.** Each term is guarded by its own `if <x>_on:`; a term that
is off contributes nothing and moves no one. **Steps 7 and 8 are last because they each run their
OWN extractor forward, which CLOBBERS the minibatch's stashes** (`last_win_prob_logits`,
`last_spread_belief`, …) that steps 2-4 read. Moving a stash-reading fold below step 7 does not
crash — it silently scores the wrong states. `instrumented_ppo_hub_contract_test.py` pins the
7-before-8 half by reading the source, along with the mixin base list (a dropped mixin removes a
whole family of loss terms without breaking an import) and `MaskablePPO` staying LAST in the MRO
(or `_excluded_save_params`'s `super()` stops reaching upstream and checkpoints start pickling a
`threading.Lock`). It also walks the package's own import graph TRANSITIVELY from the hub, so a
module reached only through `train_setup` still counts as reachable — requiring a direct edge from
`__init__`/`ppo` would forbid a decomposition rather than check one.

## The reward — the TERMINAL alone (`reward_manager.py`); the no-progress clock (`progress_clock.py`)

**The shaped reward path is DELETED** (`gen3_shaped_reward_deletion_v1`, config v122, 2026-09-26):
the PBRS potentials, the BIAS terms, the bias refund, the no-progress TAX and their 14 flags
(`designs/deleted_flags.md`). The reward is the terminal — production's win indicator
(`--terminal-indicator --victory-value 1.0`), or the signed ±`victory_value` / `--draw-penalty`
terminal. `reward_golden_test.py` was recorded at the last pre-deletion commit and passes unchanged,
which is the proof production's reward did not move. 🚨 **A resume or fork of a checkpoint trained
WITH shaping REFUSES** (`agents.model.model_version.shaped_reward`, enforced in `resolve_config`
and `main.checkargs`) — run it pinned to ≤ `029cee83`; never continue it silently on the terminal
alone. `material_margin.py` is the `win_margin` training-only OBS key, not a reward term, and
`ProgressClock` is now an OBS-only counter (no `last_penalty`).

**Full detail — the terminal table, the parity proof, the refusal, the clock — is in [`designs/training/reward.md`](../../../designs/training/reward.md).**

## MatchupSpec — the declared matchup (`matchup_spec.py`)

**The ONE explicit declaration of what a run's battles look like** — built ONCE in `train_rl_agent`
(`MatchupSpec.from_args(args)`), then CONSUMED, never re-derived (the `plan.json` pattern). It
exists because one week produced four independent failures with a shared root: *the matchup a run
plays was assembled implicitly across seams that nothing forced to agree*.

🚨 **`spec_hash()` is a MEASUREMENT-REGIME TAG: two runs or eras with different hashes are NOT
metric-comparable.** It is stamped into `metadata.json`, every `eval_results.jsonl` row,
`eval_manifest.json`, each checkpoint sidecar and `snapshot_history`; a `--model` launch whose
declared matchup differs from the checkpoint's recorded one prints `⚠️ [MATCHUP DRIFT]` with the
field-level diff, worded **RESTART** (a mid-run change) or **FORK of <parent>** (the expected case for an
exploiter fork of a self-play parent). **The two sides are
independent BY CONSTRUCTION** (`trainee_teams` / `opponent_teams`), so the mirror-bug class is
structurally closed.
**Full detail — in [`designs/training/matchup_and_lineage.md`](../../../designs/training/matchup_and_lineage.md).**

## Faint attribution in the trace (`gen3_faint_attribution_v1`)

`BattleRecorder` names the newly-fainted species by a SET DIFFERENCE over the two snapshots'
`*_fainted_species`, never by labelling a count change with the mon that was active at the
DECISION — and it emits one event per species, because **one side can lose two mons in a turn**.
Gate: `poke_env_gaps/faint_attribution_fuzz_test.py`, validated against the sim's own protocol log.
⚠️ **A protocol identifier carries the NICKNAME, not the species** — this pool holds teams with
localized nicknames (`Triopikeur` = Dugtrio), so any protocol-vs-our-data comparison must resolve
identifiers through poke-env's `battle.team` map. ⚠️ **A forensic recorder must never take down a
run**, which is why a mis-read falls back to a slightly-wrong label rather than raising. The
measured defect and its revert numbers are CLOSED history:
`designs/research_state/claude_md_archive/training_leaf_faint_attribution_history.md`.

## THE BASELINE REGISTRY (`baselines.py` · `designs/baselines.json` · `python -m main.baselines`)

**A baseline is the thing a result is read AGAINST, and this module is the ONE accessor over the
named set** (`gen3_baselines_registry_v1`). 🚨 **Read a baseline BY NAME, never by copying a path**
— `production`, `v9_long_baseline`, `v9_fold_parent`, `famine_comparator`,
`untaught_meter_opponent` and friends. Torch-free and offline; `resolve()` is the only call that
touches `models/`.

```python
from agents.training import baselines
baselines.get("v9_fold_parent").spec        # "ai_v9_59_R2ACTION_0827/final_model.zip"
baselines.resolve("famine_comparator")      # through fixed_opponent_pool.resolve_model_ref
baselines.protected_files()                 # {run: [rel path]} — the grooming keep-list
```

🚨 **Every entry is EXPLICIT** (a `.zip`, a `.json` or an `@step`, never a bare run dir), so the
last-snapshot rule below cannot move what a name points at while its run keeps training.
🚨 **A NEW OPPONENT IS A RE-MEASUREMENT, NOT A RENAME** — untaught-meter levels are not comparable
across opponents, so `python -m main.baselines set <name> <file> --reason "<ledger title>"` is the
only legal edit, and it PRINTS the ledger line to append rather than writing one.
🚨 **LOAD a baseline with `baselines.load(name)`, NEVER a bare `MaskablePPO.load`** — the bare path
rebuilds the extractor from the zip's own pickled kwargs and (measured 2026-09-22) raises
`unexpected keyword argument 'threat_prob_outspeed'` on **all five** current-generation entries,
which is how a 2026-09-14 read concluded "arch drift" and silently substituted a stand-in
checkpoint. `load()` uses the sanitizing `load_foreign_opponent` and either returns a model or
raises **`BaselineLoadError`** with `.reason` (`pre_generation` · `arch_drift` · `unresolvable` ·
`not_a_model`) and a message naming the FIX. `era_checkout_only` is VALIDATED against the entry's
recorded generation, so an unmarked pre-generation node is an error. `python -m main.baselines
check --load` runs it for real.
**Full detail — in [`designs/training/eval_and_rating.md`](../../../designs/training/eval_and_rating.md).**

## Bot evaluation (subprocess, non-blocking)

**Flat schedule, full roster, non-blocking.** Eval fires every `EVAL_FREQ_STEPS` (2M) for
`EVAL_GAMES` (100) games per opponent (`--eval-games N` overrides), uniformly over the eight
archetype bots plus `random` and every self-play sentinel — no maturity tiers, no per-opponent
caps, no roster flag. It **skips a cycle while the previous one is still running**, so a heavier
roster self-throttles instead of needing tuned ceilings.

🚨 **THE FORENSIC TRACE'S RESULT VOCABULARY is `WIN` | `LOSS` | `DRAW`** (`gen3_trace_result_v2`,
`trace_result.py`); a `DRAW` carries `meta.draw_kind` (`timeout` vs `tie`), and an unknown result is
REFUSED rather than coerced. A timeout arrives wearing a LOSS's flags, which is how it went
unnoticed for the whole archive. 🚨 **The capture quota PREFERS LOSSES and draws have their OWN
bucket** — a trace tree is a loss-enriched sample by design, each cycle's manifest states the rule
in words, and a tree that records none is SELECTION UNKNOWN, never uniform.
🚨 **MIRRORED TEAM PAIRS are a REGIME (`--eval-mirrored-pairs`, T17, config v128, DEFAULT OFF).** Each
team pairing is played from both sides on one battle seed (`rust_eval.seeds.pair_game`, both eval
cores), counts are even, and every interval is the PAIR-level pentanomial one (`mirrored_pairs.py`) —
never per-game. Recorded in `model_config.json` and `_resolve`-inherited like `eval_sentinel_greedy`;
the row's `mirrored_pairs` block is the stamp, and `elo.load_rows` REFUSES a run whose rows span it.
Do not flip the default: the M5 sizing arms are compared across it; the orchestrator flips it at the
X26 baseline. 🚨 **`--promotion-sprt` (T6, config v129, DEFAULT OFF) is its promotion twin**: each
eval-cycle snapshot is a candidate decided by a pentanomial GSPRT on fresh mirrored pairs vs the pool
frozen at its launch (H0 0.50 / H1 0.55, α = β = 0.05, cap 1,680 pairs = reject; `sprt.py`,
`sprt_promotion.py`), the cycle's own pool games never enter it, and `sprt_promotion.jsonl` makes a
failed or interrupted test un-rerunnable. Eval-only fields like this one are RECORDED, `_resolve`-inherited and never compared by
`check_compatible` — they are not `flag_registry` rows (that registry declares extractor toggles).
**Full detail — in [`designs/training/eval_and_rating.md`](../../../designs/training/eval_and_rating.md).**

## 🚨 Every live lever's SUPPLY is a declared resource (`gen3_supply_guard_v2`, `lever_supply.py`)

A flag that is ON while the mechanism behind it delivers nothing used to train a whole run that then
read as a result about the lever. Now each such lever is judged once per cycle by a
`lever_supply.DryStreakGuard`: LIVE and delivering ZERO for its declared floor of consecutive cycles
→ `LeverStarvedError` → exit **`FATAL_SUPPLY` (5)**; a mis-wiring a restart would repeat →
`LeverConfigError` → **`FATAL_CONFIG` (3)**. Neither is restarted. Floors (`lever_supply.LEVERS`,
override `--supply-starve-cycles key=N`, `key=0` = off and ANNOUNCED): `self_play_pool` 3 eval
cycles (`--self-play` with the pool still EMPTY — failed cycles count; `ai_v12_27` trained 10M
against bots), `pfsp` 3, `team_pfsp` 5 updates, `fork` 5 rollouts
(its four DISABLE-with-a-print paths are now FATAL_CONFIG), `search_teacher` 3 cycles with no
candidate. The eval/teacher-cycle streaks persist per RUN (`snapshots/summary.json`
`supply_guard`, `teacher_cycle/supply_state.json`), not per launcher segment. Every supply line goes
through `lever_supply.loud` (the run's own log AND the launcher event stream — `emit` alone never
reaches the child log). End of every segment: LOUD at zero. Also `FATAL_CONFIG`, never CRASH: a
`--bot-weights` typo, a failed `--warmstart-consensus`, and a `--distill-teacher` team that fails
gen3ou validation (the teambuilder used to DROP it silently). **Detail: [`designs/training/supply_guards.md`](../../../designs/training/supply_guards.md).**

## Self-play opponents (`--self-play`, gated behind pathology hunting)

`SelfPlayCallback` replaces `PerOpponentEvalCallback` and the training opponents become frozen
snapshots of the agent itself, drawn from a directory-backed `SnapshotPool` (`snapshot_pool.py`;
state reconstructed from `<run_dir>/snapshots/` on every restart — no manifest). Design:
`designs/ai_v5/`. 🚨 **A FORK starts with an EMPTY pool, and an empty pool does not disable
`--self-play` — it falls back to the BOT pool**; a genuine fork auto-seeds its parent's and exits
`FATAL_CONFIG` if it still has none (`pool_seed.py`), and ANY run whose pool is still empty after 3
eval cycles exits `FATAL_SUPPLY` (§ above). 🚨 **`max_snapshots` holds on EVERY path that
populates the pool, a directory SCAN included** (`gen3_pool_cap_every_path_v1`): a scan applies the add
path's eviction order, only the trainer's `owns_dir` pool deletes what it evicts, and a pool still over
its cap raises `PoolOverCapError`.
**Full detail — in [`designs/training/self_play_and_pool.md`](../../../designs/training/self_play_and_pool.md).**

## WHICH FILE a run spec names — the ONE resolution rule (`gen3_last_snapshot_resolution_v1`)

**A bare run directory resolves to the run's LAST SNAPSHOT, not to `best_model/best_model.zip`.**
Owner ruling, 2026-09-06. `best_model` is exported on **BOT win rate** — an opponent set with
nothing to do with what a teacher is being distilled FOR — and probe H8 measured the consequence:
for 2 of 8 unfunded R5F teachers the exported file was a ~0.93M-step exploiter rather than the
~2.93M final, and **nothing recorded which file was used**. Every meter this programme banks scores
a run at its END, so the last snapshot is what the metrics already measure.

### The rungs, for a BARE run dir (no `@step`)

| # | rung | file |
|---|---|---|
| 1 | `latest_txt` | `<run>/latest.txt` — a run-RELATIVE path (root CLAUDE.md); resolves both forms it can hold (`checkpoints/checkpoint_<N>_steps.zip` and the bare `final_model.zip`) |
| 2 | `highest_checkpoint` | the highest-step `checkpoints/checkpoint_<N>_steps.zip`, **including** the SIGUSR1 `checkpoint_forced_<N>_<HHMMSS>.zip`; legacy run-root copies too |
| 3 | `final_model` | `final_model.zip` / `final_model_interrupted.zip` (the higher of the two) |
| 4 | `best_model_fallback` | `best_model/best_model.zip`, then the legacy `<run>/best_model.zip` — **LAST**, only for a run that has nothing else, and it says so on **stderr** when it fires |

Two more rungs are not ladder steps at all — they are the ways a caller names a file outright, and
both **bypass the ladder entirely**: `explicit_step` (`<run>@<step>` → that checkpoint) and
`explicit_zip` (a path ending `.zip`, **`best_model/best_model.zip` included**, used verbatim).
**Naming the file is how you pin it.** Each rung also reports a coarse `rule` — `explicit_step` /
`explicit_zip` / `last_snapshot` (rungs 1-3) / `best_model_fallback`.

### 🚨 DISAGREEMENT: the higher `num_timesteps` wins, not the earlier rung

Rungs 1-3 are three names for "the end of this run", and they disagree in **both** directions:

* a **COMPLETED** run writes `latest.txt → final_model.zip` *after* its last periodic checkpoint, so
  `latest.txt` is AHEAD of `checkpoints/`. Measured on the eight R5F runs (2026-09-06):
  `final_model.zip` @**28,115,184** vs the highest checkpoint @**28,067,760** — **47,424 steps
  apart**, and rung 1 fires for every one of them;
* an **INTERRUPTED** / crashed run can leave `latest.txt` naming a file a later
  `final_model_interrupted.zip` has since passed.

Taking the earlier rung is right in the first case and wrong in the second, so neither ordering is
the rule. The rule is **the file that trained furthest**, with the rung order used only to break a
tie — or to decide when NO candidate declares a step at all (an unreadable zip). `num_timesteps` is
read from the SB3 zip's plain-JSON `data` member (`lineage.checkpoint_num_timesteps` — no torch, no
model load), falling back to the `checkpoint_<N>_steps.zip` filename. `best_model` is not on that
tier at all: it is a different SELECTION rule, so it never competes on steps and loses to every
other rung even when it trained further.

**Every consumer goes through ONE choke point** —
`agents.training.fixed_opponent_pool.resolve_model_ref(path, step=None)` → a `ResolvedModel`
carrying the rung, the rule and `num_timesteps`. It serves `--distill-teacher`,
`--stable-opponents`, `--exploiter`, `--exploiter-ladder`,
`--warmstart-consensus` and `--distill-anchor-parent`; `run_spec_test.py` holds the census that
fails, naming the file and its flags, when one of them stops. 🚨 **EVERY TEACHER LOADED BEFORE
2026-09-06 WENT THROUGH THE OLD RULE and recorded nothing about it** — `main.lineage` says so
rather than re-resolving under today's rule, because a current answer presented as history is worse
than no answer. **NOT VERSIONED**: this changes which FILE a run loads, never a weight shape.
**Full detail — in [`designs/training/matchup_and_lineage.md`](../../../designs/training/matchup_and_lineage.md).**

## Stable (cross-run) opponents (`--stable-opponents`, `fixed_opponent_pool.py`)

Load a frozen model from **another, already-finished run** as a **fixed opponent** — measured
against in eval AND (under `--self-play`) played against in training. Which FILE a run dir resolves
to is the ONE rule above. Design: `designs/ai_v5/design_stable_opponents.md`.
**Full detail — in [`designs/training/self_play_and_pool.md`](../../../designs/training/self_play_and_pool.md).**

## Exploiter mode (`--exploiter`, `MaskableAgentWrapper._exploiter_player`)

A clean opponent-mix front-end for the league **exploiter** role: train a dedicated agent against
ONE fixed foreign model as the **sole opponent every episode**, to surface (and then patch, by
folding the exploiter back as a stable opponent / pool member) the non-robustness a *self-play* Nash
cannot see. It needs **no `--self-play` / `--stable-opponents` / share fiddling**.

🚨 **AN EXPLOITER'S GAIN IS TARGET-SPECIFIC AND DOES NOT ARRIVE AS PILOTING SKILL** (ledger
2026-09-21). All three 5-team teachers were LEVEL with the plateau parent on their own five teams
against a fixed third party (−1.00 / +0.58 / −1.12 pp, every CI straddling zero) while the offense
teacher beat that same parent **0.657 [0.610, 0.702] head-to-head**; the redesign against a
DISTRIBUTION of opponents failed identically (−1.37 pp). So an exploiter is read as an OPPONENT for
the pool, not as a teacher — and the meter for the loop that consumes it that way is the
**BEST-RESPONSE GAP**, `python -m main.best_response_gap` (offline; see the root `CLAUDE.md`'s
offline-meters block and `designs/training/exploiter_and_distillation.md`). It REFUSES a comparison
at unmatched budget / dose / regime, because that week's era-2/era-1 read was confounded by a 4.5×
dose gap nobody registered. 🚨 **Two exploiters of one archetype against one target file are
REPLICATES**: each keeps its row, a `POOLED` row is the archetype's one pairing unit, and the delta
prints every replicate's own row beside it — until 2026-09-23 the later-sorted one silently
replaced the other (finding F3). One archetype + one round but a different TARGET or teamset size
is `ReplicateCollisionError`, naming both runs.

**Full detail — in [`designs/training/exploiter_and_distillation.md`](../../../designs/training/exploiter_and_distillation.md).**

## Team curriculum — team-side PFSP (`--team-pfsp`) and per-team win-rate tracking (`--team-wr-tracking`)

Two independent instruments on the TEAM axis. **`--team-pfsp {off,measure,var,onesided}`** (default
`off`, byte-identical) biases the TRAINEE's team sampling toward the pool teams it is weakest on,
with a floor, a cap and `--team-block-episodes` for per-team gradient density.
**`--team-wr-tracking`** (DEFAULT ON) is instrumentation only — a running per-`team_sha` record of
wins/games stratified by opponent class, riding `metadata.json`'s `team_win_rates` block.

⚠️ **A raw per-team win rate conflates PILOT COMPETENCE with TEAM STRENGTH** (the ai_v8 team-PFSP
finding). Anything spending budget on this signal must normalize against a team-strength baseline
first; the artifact carries that sentence in its own `notes` field. 🚨 **NO TensorBoard emission**
(owner rule: per-team series are noisy spam), pinned by a test that fails on "just one scalar".
🚨 **Same pool SIZE is not the same pool ORDER** — both aggregators verify per-index team identity
across workers and the tracker RAISES on disagreement. Both are training-only, not version-locked,
and both take an `env_method` PULL rather than an info-dict thread, because that is the seam that
works identically under `--async-rollout`.
**Full detail — in [`designs/training/team_curriculum.md`](../../../designs/training/team_curriculum.md).**

## ELO / skill rating (`elo.py`, `bot_elo_calibration.py`, `main.elo`)

Once training is mostly self-play pool play, win rate stops being legible — `win_rate_vs_pool` is a
treadmill pinned near 0.50 **by construction** and `win_rate_vs_bots` saturates. The ELO subsystem
gives a single **absolute** number, anchored to the fixed bots.

🚨 **THE EVAL OPPONENT REGIME IS A RECORDED, INHERITED PROPERTY OF A RUN** (2026-09-07,
`gen3_eval_sentinel_greedy_default_v1`). Pool sentinels are **GREEDY by default** and draw the
**trainee's own teams**; `--no-eval-sentinel-greedy` restores the old greedy-trainee-vs-stochastic-
sentinel regime, whose asymmetry read **+8.9 pp [+7.0, +10.7]** in the trainee's favour on the same
frozen pair the dense ladder plays symmetrically. `--promote-threshold` follows the regime (0.55
greedy / 0.65 stochastic) and an explicit value still wins. Both are `ModelVersion` fields (config
**v112**) with argparse default `None`, so **a flagless resume or launcher restart INHERITS the
checkpoint's regime** rather than silently crossing an opponent-regime boundary (rule of evidence
15); every launch prints `⚖️  [EVAL REGIME] …` naming both resolved values and their source.

🚨 **THE GAMES THAT SELECTED A SNAPSHOT NEVER RATE IT** (ladder recipe **v3**, owner decision
2026-09-27). v2 reused the promoting eval cycle's sentinel games as ladder edges (`source:
"eval_cycle"`) — a winner's curse of ~+15..+40 Elo at n = 100. Now each promotion plays **200 FRESH
games vs each sentinel the eval used** (`source: "promotion_baseline"`) plus the usual 100 vs every
other frozen node, logs `FRESH GAMES THIS PROMOTION: N`, and `load_games` ignores any `eval_cycle`
row; `snapshot_ladder --backfill-fresh` replaces a v2 run's. `ladder.json` also carries
**`ratings_relative`** — Elo above a pinned frozen reference node (default the
`untaught_meter_opponent_v14` baseline when the ladder holds it, else the first snapshot), frozen
edges only, a second column beside the bot-anchored headline. Detail:
`designs/training/eval_and_rating.md` (recipe v3, the relative column).

🚨 **EVERY `snapshot_ladder/ladder.json` CARRIES A `recipe` STAMP, AND A CROSS-RUN READER REFUSES
OR REFITS WITHOUT IT** (`snapshot_ladder.recipe_status` → `current`/`absent`/`differs`;
`check_recipe` raises `LadderRecipeError`). A rating is only comparable to one fitted the same
way: a file fitted before `3e6875a5` folded the eval-cycle sentinel edges in and read **+73.1
Elo** above the current fit of the same 20 nodes, flipping the sign of a cross-run delta
(2026-09-14). `main.critic_gate` refuses a stale committed file on its FALLBACK path (it refits
otherwise); `--exploiter-ladder auto:` refits in memory or refuses; `latest_promoted_elo`
deliberately does NOT check it (a within-run trend scalar). **Bump `LADDER_FITTER_VERSION`
whenever the fit changes what a rating MEANS.**

**Converting a stale file: `python -m main.elo refit [--apply] <run>`** — refits from the
append-only `games.jsonl` (plays nothing) over the COMMITTED file's node set, and with `--apply`
writes the stamped fit while keeping the old one as `snapshot_ladder/ladder.pre_recipe.json`
(`ladder.pre_recipe_vN.json` for a file stamped vN).
Every file on disk before `0f230405` is stale; **68 of 93 move** (median max |Δ| 53.4 Elo, 65 of
68 newest nodes DOWN) and the v9 generation ladder reverses 21 of 153 orderings — the whole audit,
one JSON per run, is `designs/research_state/measurements/ladder_refit_audit_2026-09-22/`.

**Full detail — in [`designs/training/eval_and_rating.md`](../../../designs/training/eval_and_rating.md)
and [`designs/training/self_play_and_pool.md`](../../../designs/training/self_play_and_pool.md).**

## Rollout collection: sync barrier vs `--async-rollout` (`async_vec_env.py`)

The default `SubprocVecEnv.step()` is a **per-step barrier**. `--async-rollout` swaps in
**`AsyncSubprocVecEnv`** (per-env `send_step`/`poll_ready`/`recv_step` + **drain-safe
`env_method`**, which stashes in-flight step results before any barrier RPC) and
`collect_rollouts_async`. It keeps every worker in flight, batch-forwards whichever envs are READY,
and writes each env's transition into **its own buffer column**. It is **exactly on-policy** — a
scheduling change, not an APPO-style algorithm change — and the per-decision mask rides in the Dict
obs, so no wrapper changes. Off by default; ignored under `--debug`. Measured **+14% FPS at the
production `--n-envs 64`** (1489→1695, heuristic opponents); design + benchmark table:
`designs/ai_v5/design_async_rollout.md`.

🚨 **This is why several callbacks are `env_method` PULLS rather than info-dict threads** (reward
terms, team PFSP, per-team win rates): the async collector wave-batches, so callback locals cannot
recover which buffer ROW a step landed on. A capture that needs the row is INLINED into
`collect_rollouts_async` instead (`WinProbLabelCallback`'s terminal capture).

🚨 **A STEP-COUNTED CADENCE IS TOTAL ENV STEPS, never vec calls or rollouts** (F-SZ-3, 2026-10-01).
The periodic checkpoint was SB3's `n_calls % save_freq` at 50,000 calls — 2.4M env steps at N = 48,
~102M at N = 2048, early under `--async-rollout` waves; it now saves when `num_timesteps` crosses each
multiple of 2.4M (`main.train.constants.checkpoint_due`), like eval, the pool add/refresh it drives,
the search teacher and the plasticity canary. A callback CALL is not a fixed number of env steps (N
sync, N decisions on the Rust collector, one WAVE < N under async), so a new cadence compares
`num_timesteps` against a boundary — never `n_calls`. Everything counted in UPDATES or rollouts
(`--diagnostics-every`, the compile canary, the team pulls, the CUDA memory-trend horizon, …) moves
with `n_steps × n_envs`. The table and its tests: `designs/ops/training_runbook.md` → "Cadences and
N", `src/main/train/cadence_n_independence_test.py`.

**`--policy-gae-lambda` (default 0.80) is the POLICY's GAE λ** (`gen3_policy_gae_lambda_v1`, config v123; the critic's BCE target is the terminal outcome, with no λ-return).
The former was a literal 0.80 at both `model_build` sites until 2026-09-26; it is now recorded on
`ModelVersion` and INHERITED on a flagless resume (name it to change it). **`train()` logs one
`train/approx_kl_epoch_<k>` / `train/clip_fraction_epoch_<k>` pair per epoch that ran**, folded
from the numbers the loop already computes — ⚠️ stock `train/approx_kl` is the LAST epoch's mean
while `train/clip_fraction` pools every epoch. Detail:
[`designs/training/ppo_step.md`](../../../designs/training/ppo_step.md).

**`--diagnostics-every N` (fresh default 10; `gen3_diagnostics_cadence_v1`, config v124) runs the
OPTIONAL probes — per-term noise scale, `grad/*`, `rank/*`, `edge/*`, `cell/*` — on every Nth update
only**, and a skipped update writes NONE of their tags (a gap, never a stale value). The first update
of every process always runs them (the compile lock follows it), `--rank-tripwire` keeps `rank/*`
every update and `--adaptive-batch policy` keeps the per-term probe every update. Learning is
BIT-IDENTICAL at any N (`diagnostics_cadence_test.py`). Recorded + inherited; a pre-v124 run inherits
1. ⚠️ A reader that windows by reading COUNT (the vf_coef "last 20") now spans N× the updates.
Detail: [`designs/training/ppo_step.md`](../../../designs/training/ppo_step.md).

## Where the trainee's observation comes from (`--obs-source {python,core}`, DEFAULT `core` on the rust bridge)

`gen3_core_obs_source_v1` — the Rust core program's M6: **the production default since the cutover
(2026-09-25)**; `python` is the explicit opt-out (byte-identical by construction; removed by the
deletion pass) and the default on `--use-bridge node|off`. `core` takes the trainee's observation row (2761-dim) and 11-bit mask from the rust
`sim_bridge` child (`__OBS__` frames; needs `--use-bridge rust`); `Gen3Env` REFUSES a frame of
another battle, decision (`n`) or turn, a NaN cell or a mask that disagrees with the reading. Labels,
reward, the tracker fold and the action mapping stay Python; terminal and non-decision embeds are
still encoded here and counted (`Gen3Env.core_obs_counts`). The env-level parity gate is slice N
(`main/rust_core_cutover/slice_n_test.py`), zero differences, no allowlist. 🚨 **A DECISION is recorded only when the env
asks the trainee to move** (`gen3_no_phantom_decision_v1`, a TRAINING-INPUT change): poke-env embeds
`battle1` on every step, including a `wait` request or its re-embed of an answered request, and the
trackers used to take a decision there (5.0% of steps); a `wait` request reaching the record RAISES. 🚨 **The OPPONENT is polled only when its order will be SENT** (`gen3_no_phantom_opponent_poll_v1`, M5 Lane E, the opponent twin): `SingleAgentWrapper.step` asked `choose_move` on steps whose p2 order was dropped, so a self-play `RLPlayer` recorded a phantom decision (progress clock one step high) and drew a sample, and a bot drew from its RNG.
Detail: `designs/rust_sim/encoder.md`, `designs/endstate/program_rust_core.md` §3.

## The env core — `--env-core {python,rust}` (PRODUCTION `rust` — the M5 switch; M5 Lane G)

🚨 **THE M5 SWITCH (`gen3_env_core_switch_v1`): `rust` is the PRODUCTION env core**, declared with every
run SIZE in ONE block, `designs/production_config.json` `recipe.sizing` (the sizing verdict fills it).
An UNTYPED `--env-core`: fresh `--arch production` → rust; `--model` (a restart or a fork) → INHERITED,
the core the checkpoint was produced on when that is rust — a PYTHON-ERA checkpoint (produced on python,
or before `--env-core` existed) moves onto rust, announced as a CORE SWITCH, and one that trained the
SHAPED critic is REFUSED whatever the core (`FATAL_CONFIG`: run it pinned) — deletion pass D4; a bare non-production fresh argv → rust too (the deletion pass's bare-argv flip, D2 2026-10-02: the
bare argv is `--critic winprob` + its three reward values, which the Rust core serves). `--env-core python`
(typed) opts out until the deletion pass removes the Python core.
One resolver: `main.train.rust_env_setup.resolve_env_core_default`, called by `resolve_config` and
`checkargs`; pinned by `main/train/env_core_switch_test.py`. Runbook: `designs/ops/training_runbook.md`.

`--env-core rust` runs the rollout on the M5 Rust env core: N envs in ONE core (process front end by
default, `--rust-env-front`), the trainee and every policy opponent forwarded through the inference
service in ONE flush, the scripted bots played inside the core, and the COMPLETE-GAME collector
(`--rollout-trigger complete_game`, the default there): a game's rows are buffered until it ends, GAE
and the win label run on the complete game (every row `win_mask` 1), and an update fires at
`--rollout-target-samples` completed-game rows (default `n_steps × n_envs`), consuming exactly that many
— no row is dropped or down-weighted for age. `python` is untouched and stays the default; the cutover is
a separate decision. Hazards an agent must know before touching it:

- 🚨 **Startup runs BEFORE `--compile-trainer`** (`model_build._start_rust_env`): the inference service
  deep-copies the policy as its slot templates, and a copy taken after the compile would carry the
  patched `forward` bound to the LEARNER's extractor.
- 🚨 **Every flag whose path the Rust core does not serve is REFUSED at startup, by name**
  (`combination_checks`' `env_core_rust_*`): distillation, `--cf-records`, the search teacher,
  `--team-pfsp`, `--exploiter-ladder`, `--async-rollout`. The
  collector flags typed on the python core are refused too (they would be silently inert).
- **`WinProbLabelCallback` is not registered** under `rust`: the collector fills `win_target` /
  `win_mask` (the window fill calls the callback's own `backfill_terminal_labels`).
- **K9(b) `--behaviour-check`** (default `fatal` on BOTH cores): before any optimizer step of every
  update, the learner's log π on rows played at the CURRENT version must equal the stored behaviour
  log-prob (ONE gate, at fp32 matmul precision `highest` — the only precision, TF32 was retired: DETERMINISTIC — a row whose forward has a declared selection / threshold within a relative margin 2e-4 of its cutoff is EXCLUDED (3.7 % of healthy rows; `agents/model/selection_sites.py`, `rust_rollout/tie_margins.py`), every other row's |Δ| < 1e-4 or FATAL at once, the excluded share < 0.15; a process at any other precision is refused). Under `rust` Lane G's pre-loop probe runs its own forward and logs
  `staleness/*` (ratio, clip fraction, KL by row AGE) and `behaviour/*`; under `python` (no per-row
  versions) the first micro-batch's own forward is compared in-loop instead (`learner_gates.md`). Per-game version pinning (`--version-pinning per_game`) is the first
  staleness remedy, OFF unless those measurements call for it.
- **Every micro-batch is FULL — no padding, no drop.** `--rollout-target-samples` must be a multiple of
  lcm(`--batch-size`, `--n-envs`) (refused at parse, at the trigger, at every adaptive move, and by the
  collector before a fill); the one game straddling the target is split and its tail trained next
  update. Only the last ACCUMULATION group can be short, and the learner flushes it as a FULL-weight step
  normalised by its real rows — which the dose now counts as such (K10(c)).
- **Stochastic actions are the KEYED DRAW** (`gen3_keyed_draw_v1`; the trainee always, policy opponents
  by default — `--opponent-sampling keyed`): replayable from the decision's key, which is what keeps the
  parity gates exact. It is NOT a speed lever (~0.2 ms a step: F-LE-8's "5.1 ms of sampling" was the host
  waiting for the forward). A core RESPAWN (up to `--rust-env-respawn-budget`) cuts the live games,
  derives a new segment seed and re-stages; past the budget it is fatal.
- `rollout/collect_ms` + `rollout/collect_decisions` are logged on BOTH cores (the A/B reads them);
  `rust_env/*` is the collector's per-phase read. `metadata.json` records `env_core` on every save.
- ⚠️ A launcher RESUME pins to the checkpoint's commit. A commit before Lane G has no `--env-core`
  and is refused by name. The trainer builds its checkout's env core at startup
  (`utils.rust_env.build`), because a pin worktree has no `target/`. An untyped `--env-core` on a
  `--model` launch INHERITS the checkpoint's recorded core (an `--arch production` restart's from
  `cli_args`); a TYPED switch is announced (`⚠️ [ENV CORE]`). Detail: `src/main/launcher/CLAUDE.md` → "A `--env-core rust` run under the
  launcher".
- **EVAL runs on the core too** (M5 Lane H, `rust_eval/`): both eval callbacks write the same plan and
  manifest, then play the cycle IN PROCESS and BLOCKING on a declared eval core (`--rust-eval-envs`) and
  declared eval T2 slots, publish the workers' own shard records, and collect them with the unchanged
  code. Games are seeded by the GAME (`gen3_eval_game_seed_v1`); traces are CORE traces (records +
  reconstruction + states; the prober expands them). A missing eval core under `rust` is FATAL, never a
  fall-back to Python workers. Detail: `designs/training/eval_and_rating.md` → "Eval on the Rust env core".

Detail: [`designs/training/rust_collector.md`](../../../designs/training/rust_collector.md).

## The two compile flags (`--compile-opponents` · `--compile-trainer`, both DEFAULT ON)

**Split by WHO and WHERE** (renamed 2026-08-14 from the single `--compile-extractor`, which said
neither): **`--compile-opponents`** is the CPU/ROLLOUT half — the frozen opponents in the env
workers, plus BLAS thread pinning. **`--compile-trainer`** is the GPU/LEARNER half (auto-on for
cuda) — the CUDA forward **and backward** the PPO step runs, and the larger of the two. They are
orthogonal; a run can take either, both or neither.
**fp32 matmul precision `highest` is the ONLY precision** (TF32 retired, deletion pass K2; `--matmul-precision`
is DELETED — `designs/deleted_flags.md`). Nothing in the trainer sets it; `metadata.json` still records the
realized value as `matmul_precision`, and every parity gate (the region gate, the canary, T2's judge, K9(b))
refuses a process at any other precision. A resume or fork of a run whose `metadata.json` recorded
`matmul_precision: high` is refused `FATAL_CONFIG` naming the pin (`model_version.retired_levers`,
`LAST_COMMIT_K2`). K9(b)'s behaviour gate (`rust_rollout/consistency.BEHAVIOUR_GATE`): `max` |Δ log π| < 1e-4,
DETERMINISTIC (`gen3_behaviour_tie_exclusion_v1`: the policy forward selects with topk / argmax, and a selection within a rounding error of its cutoff can resolve differently in T2 and in the learner — such rows, 3.7 % at a relative margin 2e-4, are EXCLUDED and every other row is judged single-shot; the offending rows are dumped to `<run_dir>/behaviour_violations.jsonl`;
`designs/training/learner_gates.md`).
🚨 **`--compile-trainer`'s startup gate (the REGION gate, `compile_regions.gate_regions`) runs on REAL
obs rows**, never zeros. The rows are the committed fixture `src/agents/model/compile_parity_obs.npz`
(R1 uses the K9 golden's labelled buffer on the production surface); regenerate the fixture with
`python -m agents.model.compile_parity_fixture --write` after an obs-layout change, and a stale
fixture REFUSES. It holds R0's decision readout (MASKED legal log-probs, V) and R1's loss and every
policy gradient (cosine ≥ 0.9999 plus the per-parameter rule) to eager. **On FRESH weights** (a fresh
launch: the zero-init pointer head makes every legal log-prob `-log(n_legal)`, so that check cannot
fail) it ALSO runs on a seeded, bit-exactly-restored perturbation of the policy
(`agents.model.parity_probe`, `gen3_fresh_parity_probe_v1`), and a vacuous comparison REFUSES rather
than passes. The CPU `--compile-opponents` path runs a decision-level parity check once per distinct
weights (`agents.model.opponent_parity`), which RAISES on a mismatch. 🚨 **HEAD runs torch >= 2.8 ONLY** (deletion pass K1, 2026-10-02): `utils/torch_floor.py`
exits the trainer `FATAL_CONFIG` on an older torch, because every 2.5.1 path is gone — the
extractor-only compile and its gate, and the CUDA trunk split (`gen3_inductor_trunk_split_v1`) that
2.5.1's single Inductor graph needed (it miscompiled on real rows: every default cuda run from
`28eaef29`, 2026-08-17, to 2026-09-28 — argmax agreement 70.9%, gradient cosine 0.778). A 2.5.1 run
resumes PINNED to its own commit through the launcher, which selects `gen3ai_stable`
(`designs/training/compile_flags.md` "Lane K1").
🚨 **THE COMPILE SENTINEL (`gen3_compile_sentinel_v1`) makes a silent recompile / eager fallback
FATAL.** `src/agents/model/compile_control.py` is the ONLY runtime module that touches
`torch._dynamo` — add nothing that does elsewhere. Phases: `torch._dynamo.reset()` → install + gate
the DECLARED REGIONS → prewarm every DECLARED signature (`compile_regions.prewarm_calls` is the
declaration) → LOCK at the END OF STARTUP, before the first real
iteration (K6; `🧊 [COMPILE LOCK]`) → RELEASE when `learn()` returns (the final eval runs in-process).
While locked, any recompile, late first compile or cache-limit hit exits `[CompileSentinel] FATAL` /
`FATAL_CONFIG` (not restarted), and the message NAMES the failing guard (`UNDECLARED SIGNATURE — the
failing guard(s): …`). At update 10 and then every 100 updates the IN-RUN PARITY CANARY (`agents/model/compile_canary.py`) holds
the compiled learner to eager on the real-obs fixture at the startup gate's bars (decision readout AND
the train graph's gradient). A disagreement is CONFIRMED in the same update (the same rows + an
independent slice, eager recomputed): confirmed ⇒ checkpoint + `[CompileCanary] FATAL` naming the safe
rollback point (`<run_dir>/canary_verdicts.jsonl`); unconfirmed ⇒ counted, and two consecutive ⇒ FATAL. The rank probe is HOOK-FREE (a forward
hook on a compiled module is a guard — the iteration-1 signature `8fc297a2` used to absorb). 🚨 **The learner compiles ONLY as DECLARED REGIONS** (K8, `agents/model/compile_regions.py`, `designs/training/compile_flags.md` "K8 — DECLARED COMPILE REGIONS"): R0 the rollout core and R1 the micro-step, each `fullgraph=True` at ONE declared signature (a ragged micro-batch and batch 1 take the declared eager route); the rank probe reads R1's stashes (no second forward, the spectra on the device); on a CUDA buffer every micro-batch is STAGED to the device by a prefetch thread (`--device-batch staged`, the default since 2026-10-01; `resident` = one device copy of the whole flattened buffer per update, +~1.1 GB of update peak; `instrumented_ppo/device_batches.py`, bit-identical batches in every mode, the same permutation). 🚨 **Startup RUNS one dry update** (`agents/training/update_fit.py`, `gen3_update_fit_v1`): one real `train()` epoch on a fixture rollout of the buffer's full shape, the learner restored bit-identically, and a first update that would not leave 1,024 MiB of device headroom (or OOMs) is `UpdateWontFit` (FATAL_CONFIG) — `designs/training/learner_lifecycle.md` "The update fit check". ONE startup gate per region (`gen3_one_gate_per_region_v1`); the trainer's compile step (`compile_trainer.preflight_compile_trainer`) compiles nothing and refuses a learner without the micro-step. 🚨 **No region runs eager silently** (`gen3_no_silent_eager_v1`): a compiled-route call whose Python body executes, a ragged tail past one per epoch, or dynamo disabled / errors suppressed / the stance moved under the lock is a typed FATAL; every update logs `lifecycle/compiled_region_calls`, `lifecycle/eager_fallback_calls`, `lifecycle/eager_share`, `lifecycle/update_wall_s` (`designs/training/compile_flags.md`). R1's startup gate and the canary judge its per-parameter gradient by WEIGHT REGIME (`compile_regions.weights_regime`: fresh vs trained, measured bars — `designs/training/compile_flags.md`). Code inside R1 must stay a static-shape program (the fold contract above). 🚨 **Every lever R1 reads is DECLARED at startup from the resolved config** (`gen3_r1_declared_levers_v1`): `TrainSetup._r1_levers` is the ONE predicate for the strata and rollout-weight levers (never a rollout's data — an idle strata lever gets neutral ones, bit-identical), and every compiled update is held to the declaration (`compile_regions.check_r1_declared`, a typed FATAL naming the field or key). A new R1 lever is declared there, never by relaxing the lock (`compile_flags.md` "R1's DECLARED LEVERS"). A NEW compiled signature is added to the region table (`compile_regions.REGIONS`)
and its prewarm, never absorbed after the lock; a learner-process caller OUTSIDE the regions runs the
extractor eager already (the regions compile functions over the module and never patch `fe.forward`,
so `compile_trainer.eager_extractor(fe)` is now a no-op on the learner); batch 1 is ALWAYS
eager (`compile_trainer.EAGER_BATCHES`, `gen3_batch1_eager_v1` — torch 2.8 cannot lower a batch-1 CUDA graph) — the late-shape
table in the doc lists every caller. An unknown torch or a drifted torch internal
(`_SOURCE_HASHES`) REFUSES. TB: `compile/recompiles_after_lock` must stay 0.
🚨 **THE COMPILE CACHE IS THE RUN'S OWN (K3, `gen3_hermetic_compile_cache_v1`).** Every process of a
run's tree compiles into `<run>/compile_cache/` (Inductor, Triton, T2's AOT packages), declared by
`lifecycle._declare_compile_cache` the moment the run dir exists — EMPTY at a fresh launch or fork,
reused only by the run's own restart under a matching stamp (commit, clean tree, torch, interpreter,
`compile_control.config_row_hash()`), wiped otherwise; printed as `🧊 [CompileCache]`. Tests and
every other compiling process get a fresh private dir on the REAL DISK (`$GEN3AI_SCRATCH`, else
`~/.cache/gen3ai/tmp`; never tmpfs `/tmp`), deleted at exit, swept by PID if its owner died
(`agents.model.compile_cache`). **A new
`torch.compile` site must call `ensure_hermetic_cache()` first** — `compile_cache_test` fails a
production module that does not. The box-wide `/tmp/torchinductor_<user>` and
`/tmp/gen3ai_inductor_cache` are no longer used. Cost: a fresh launch always compiles cold (see the doc).
**Full detail — in [`designs/training/compile_flags.md`](../../../designs/training/compile_flags.md).**

## Gradient-balance + value-scale diagnostics (`grad_balance.py`)

The dual-head extractor shares ONE transformer trunk between policy, value and a dozen auxiliaries,
and all of their gradients compete there. A read-only `autograd.grad(retain_graph=True)` probe on
ONE minibatch per `train()` measures each head's pull **on one common denominator**, so every
`grad/*_share` sums to ~1 and any term crowding out the rest is read directly. "Shared" is the
DECLARED `SHARED_TRUNK_PHASES` allow-list — it excludes `cls_pool` and both projection heads, so
only truly contested params count.

**`grad/value_policy_logratio` is the gauge to read** (`log10(‖g_value‖/‖g_policy‖)`, 0 = balanced)
— it is AUX-INDEPENDENT, where `grad/value_share` moves with how many auxiliaries are on. It is
also what `--vf-coef`'s startup announcement quotes rather than recomputing.
⚠️ **`edge/<fam>_*` and `cell/<name>_*` liveness are NOT effect sizes** — every family and cell
enters ZERO-INIT, so a dead one is bit-identical in the logs to a working one; read `weight_norm`
and `grad_norm` as a PAIR, and only an ablation measures importance.
**Full detail — in [`designs/training/telemetry_scalars.md`](../../../designs/training/telemetry_scalars.md).**

## Live capacity telemetry (`--capacity-telemetry`, `capacity_telemetry.py`)

**Three continuous saturation early-warnings that ride the train loop** — the plasticity canary, the
half-batch trunk-gradient cosine, and feature velocity. They exist because every previous answer to
*"is the network out of capacity?"* was an expensive one-shot probe returning a NUMBER at a MOMENT,
and saturation is a trend.
**Full detail — in [`designs/training/telemetry_scalars.md`](../../../designs/training/telemetry_scalars.md).**

## THE VALUE LOSS has a MODE — `--critic {shaped,winprob}` (`gen3_winprob_critic_mode_v1`)

**Default `winprob` (the bare-argv flip, deletion pass D2); `shaped` is the historical critic, `--env-core python` only.** Design of record:
`designs/ai_v12/design_winprob_only_critic.md`; the model-side half is `src/agents/model/CLAUDE.md`
→ *The CRITIC MODE*.

| | `shaped` | `winprob` |
|---|---|---|
| the value TERM | `vf_coef · mean((returns − values)²)` (clipped under `--clip-range-vf`) | `vf_coef · _win_prob_loss(...)` — the head's **BCE against the terminal outcome** |
| the scalar `value_loss` | the loss | a DIAGNOSTIC only (its term is dropped), computed UNCLIPPED |
| PopArt · `--value-dist-*` · `--value-from-dist` · `--value-tail-weight` · `--win-prob-coef` · every `--win-prob-pbrs-*` | **DELETED** (L1, config v131: `designs/deleted_flags.md`; a checkpoint that recorded one ON is refused, `model_version/retired_levers.py`) | **DELETED** |

🚨 **`winprob` REQUIRES all three of `--terminal-indicator --victory-value 1.0 --draw-penalty 0`**,
each named by its own `combination_checks` refusal, so the undiscounted return
is exactly `1{win}` and at `--gamma 1.0` `V(s) = P(win|s)` with no approximation term.
🚨 **THE COST IS STATED, NOT BURIED: a critic bounded in [0,1] cannot represent "a timeout is worse
than a loss."** The anti-stall pressure is the obs deadline clock (the reward has no anti-stall term), and
**stall rate and mean episode length are PRIMARY, kill-condition-bearing endpoints on a `winprob`
arm.** 🚨 **A `winprob` run's `train/*` value tags are not comparable with a `shaped` run's.**

🚨 **`critic_resolution` IS the meter; `critic_reliability` is not.** A base-rate forecaster scores
a perfect 0 reliability and a useless 0 resolution — the committed baseline measured this head at
reliability ~0.002 against a resolution of 0.062 out of an available 0.182, so a promotion that
improves ECE and leaves `critic_resolution` flat has moved the meter that was never the disease.
The `win_prob/critic_*` family comes from `scaffolding.reliability_table`, **imported, never
re-implemented**, and an unmeasurable rollout publishes **`{}`, never zeros**.

🚨 **A DRAW IS SCORED AS A NOT-WIN BY DECISION** (`y = 0`), never dropped and never 0.5 — that is
what makes "P(win)" literally P(win). ⚠️ `signal/draw_rate` counts **ties and not timeouts**, so the
series that watches the 250-turn cap is `signal/stall_rate` / mean episode length.
🚨 **THE 250-TURN CAP IS A TERMINAL, NOT A TRUNCATION** — this env never truncates in the SB3 sense;
a cap forfeit used to arrive as `truncated`, so SB3 bootstrapped `V(s_last)` onto a 0 reward at
γ=1 and the timeout left the loss entirely with **a TD error of identically zero**. Fixed in
`wrappers.resolve_episode_end`, under `winprob` only. **`--gamma` is a flag now and is INERT ON A
RESUME like `--lr`.**

### `--win-prob-strata-weight` — the BCE's opponent MIX (`gen3_winprob_strata_weight_v1`, v115)

**Default `0.0` = OFF and the loss is BIT-identical; `--critic winprob` is REQUIRED** (refused
otherwise — under `shaped` that BCE is an auxiliary diagnostic, not the value loss). Each state's
BCE term is multiplied by its opponent CLASS's weight `w_c ∝ freq_c ** (−s)`, capped at **8×** and
renormalised so the **mean weight over the rollout buffer is exactly 1** — it re-prices the MIX
without moving the loss SCALE, so an arm cannot confound "re-weighted the classes" with "raised the
critic's learning rate". At `s = 1` every class contributes equally; `s` interpolates.

🚨 **WHY, and it is arithmetic rather than a hunch.** Only **10.2 % / 14.4 %** of the terminal 0/1
label's variance lies BETWEEN (cycle, opponent) cells, so a head minimising BCE buys its resolution
from the board and its own team — which is cheaper — and never conditions on the opponent. The head
refit proved the fault is the TARGET, not the head, on both substrates
([`winprob_head_refit_2026-09-09`](../../../designs/research_state/measurements/winprob_head_refit_2026-09-09/README.md)
§6/§11). This raises that share directly, with **no new labels and no rollout cost**.

🚨 **THE VOCABULARY IS THE FOUR `opp_class` CODES** — `bot` / `pool` / `stable` / `exploiter` — and
per-BOT identity is **not available**: the archetype is drawn per EPISODE in
`MaskableAgentWrapper._select_episode_opponent` and never reaches the observation. So the lever
balances the between-CLASS share and leaves within-class heterogeneity in episode proportion.
⚠️ **THE CAP BINDS AT THE PRODUCTION MIX, deliberately**: at ~10 % bots / ~90 % self-play, `s = 1`
asks for 10× and gets 8×, so the objective splits **44/56, not 50/50** — a 4.4× re-pricing with a
bounded per-row weight. Read `win_prob/strata_share_*`, `strata_w_entropy` (1.0 = balanced) and
`loss` vs `loss_unweighted`. 🚨 **`strata_active` 0 vs an ABSENT family are different facts**: the
family is published whenever the flag is on, so 0 means "on, but one class present / no labels yet"
(every `--debug` run and any run before the pool seeds) and ABSENT means the flag is off.

### `--fork-fraction` — THE FORK ARM, contested-state EXPLORING STARTS (`gen3_fork_v1`, v120)

**Default `0.0` = OFF and BIT-identical** — no module imported, no obs key declared, no callback
attached, no buffer installed, no row injected. **`--critic winprob` AND `--cf-records` are BOTH
REQUIRED** (the second on the Python core only), and `--win-prob-strata-weight` is REFUSED alongside it. Detail:
[`designs/training/forks.md`](../../../designs/training/forks.md).

🚨 **TWO IMPLEMENTATIONS, and the Python one is LEGACY until the deletion pass.** Under
`--env-core rust` the arm is the COLLECTOR's fork phase (`rust_rollout/fork.py`,
`gen3_fork_rust_v1`, forks.md §14) — DECLARED and OFF, deferred by the owner's one-ply scope
(2026-10-01). It replays the core's finished input log on Lane I playout handles, keys every branch
draw on the PARENT's keyed-draw key (a parent-action branch IS the parent — gate
`rust_rollout/fork_crn_integration_test.py`) and puts each branch game into the complete-game FIFO
after its parent, so **branch rows COMPETE for the update's D** rather than doubling the buffer, and
a branch plays the parent's REAL policy opponent where its slot still serves it
(`fork/opp_substituted` is the rest). Both are DEPARTURES from the arm that read NOT DETECTED on
2026-09-16 — that read does not transfer unchanged. Requires `--opponent-sampling keyed` and
`--rollout-trigger complete_game` (refused by name otherwise).

🚨 **WHY — the head ranks siblings at CHANCE.** `paired_refit_discrimination_2026-09-14` measured
the promoted win-prob critic's pairwise accuracy on successors ONE MOVE APART at **0.5169
[0.4800, 0.5524]**, while a FROZEN trunk with only the head's four tensors refit on counterfactual
successors reaches **0.6032** (+0.0863, DETECTED) and a pairwise RANKING term buys **nothing**
(−0.0107, NOT DETECTED). Pairwise accuracy is a RANK statistic, so the ordering was in
`value_pooled` all along and the on-policy stream never asked for it: **the DATA is the lever, not
the loss form.** A rollout visits exactly ONE successor per decision; this manufactures the
siblings. At a contested decision the battle is forked, three branches (the policy's top-2 + ONE
uniformly random legal action) are played to a terminal by the CURRENT policy, and their
transitions enter the SAME PPO buffer. **Plain BCE, NO ranking term — closed as a lever.**

🚨 **THE MASK RULE IS UNIFORM: the FORK STEP is out of the policy term for EVERY branch**, the
top-2 included, and the term is RENORMALISED over the kept rows (not just zeroed — a masked
`.mean()` would silently lower the effective policy LR by the fork rate). Masking only the random
branch would re-weight the policy gradient by the branch MIX. The fork step stays fully in the
VALUE terms. Carrier: the `fork_pg_m` obs key.

🚨 **THE PREFIX IS COUNTED ONCE** — a branch's rows begin AT the fork step. The fork STATE appears
once per branch with a DIFFERENT action; that is the exploring start, not a duplicate.

🚨 **`--fork-crn dice_and_draws` (default) pairs the DICE *and* the policy draws.** `cf_q_labels`
paired only the dice — a concrete, testable account of its null — and `fork_crn_sim_test` proves
byte-identical protocol on identical actions through the real bridge.

⚠️ **The ecology approximation is the arm's largest caveat:** a `__RECON__` record carries no
opponent identity, so a branch is played against a SELF-LIKE opponent. Injected rows are labelled
`opp_class = POOL` for that reason; `fork/branch_share` and `fork/bot_share` price it.

⚠️ **RAISE `--cf-records-keep`** (the ring is pruned globally to the newest N while a rollout
finishes ~2,400 episodes, so at 512 the forks that resolve are the LATE ones — a selection bias).
🚨 **A fork dropped at the row budget has ALREADY BEEN PLAYED**, so the ask is bounded by the
previous rollout's MEASURED `fork/rows_per_fork`. Read **`fork/rate`**, **`fork/branch_share`**,
**`fork/tie_rate`**, **`fork/random_wins`**, **`fork/pairwise_acc`** (IN-SAMPLE; the endpoint is a
held-out read) and **`fork/sim_steps_share`** (the cost).

**Full detail — the currency argument, the cap-terminal measurement, the `--vf-coef` BCE
announcement and every value-side flag below — is in
[`designs/training/critic_and_value_losses.md`](../../../designs/training/critic_and_value_losses.md).**

### The other value-side flags, in one place

| flag | default | what it does, and the one thing to know |
|---|---|---|
| `--vf-coef` | `0.5` | multiplies a BCE under `winprob`, an MSE on a shaped return under `shaped` — **the 0.5 default carries no information about the first**. The startup announcement prints the raw BCE and the value/policy shared-trunk gradient RATIO (`10 ** grad/value_policy_logratio`); it never divides by `|policy loss|`, which is ≈0 by construction on epoch 1. Fixed for a run's lifetime |
| `--td-aux-coef` | `0.0` | the Bellman identity as an explicit loss over CONTIGUOUS pairs the PPO permutation destroys. 🚨 **Pre-registered band 1.0–3.0; `λ ≤ 0.1` measured significantly WORSE than control** — the small-coef regime is to be avoided, not treated as "a bit of the effect". Episode boundaries DROP the pair, never zero it |

## The DETACHED RIDE-ALONG heads (`--ridealong-ensemble` · `--ridealong-rnd` · `--ridealong-adv` · `--ridealong-opp` · `--ridealong-rnd-variants`)

`gen3_ridealong_heads_v1` (v126; `instrumented_ppo/ridealong_terms.py`, heads in
`agents/model/ridealong_heads.py`). Baselines that OBSERVE: a V ensemble (epistemic uncertainty),
RND novelty on the raw observation, and the Q = V + A + B main effects on PPO's own labels.
`_ridealong_update` runs right after each minibatch's `evaluate_actions` and BEFORE PPO's loss is
built: detached stashes → the heads' losses → their own backward, clip and Adam step → their grads
back to None. So it sits OUTSIDE the fold order above, and nothing in the fold can see it.
`ridealong_update_test` pins one real update ON vs OFF as bit-identical (params, PPO optimizer
state, PPO scalars, RNG). Two things to know when reading `ridealong/*`. **The heads TRAIN and READ
on epoch 0 only** (`RIDEALONG_EPOCHS` = 1: each rollout row seen once, scored before the heads train
on it). All 10 epochs cost +13 % of a GPU update on the learner benchmark; one pass MEASURED +0.59 s = 0.88 % of a 67 s update (`ridealong_step_benchmark.py`; X26's
`PREREGISTRATION.md` "Overhead").
And **the heads' Adam state is not checkpointed** (a restart resumes their weights with a fresh Adam).
**Every ride-along optimizer is ACQUIRED AT STARTUP** (`RideAlongTerms._setup_model` →
`_ridealong_acquire`, Adam state pre-allocated, bit-identical to lazy init). There is NO lazy build:
a step that finds an optimizer missing, or bound to other heads, raises
`RideAlongLifecycleViolation`, as K6.1's freeze guard would. Tooling that swaps `policy.ridealong`
must call `_ridealong_acquire()` again (both benchmarks do). 🚨 **Only the TRAINEE acquires: an
OPPONENT load acquires nothing** (`gen3_opponent_inference_load_v1`). The self-play pool and the eval
sentinels load through `snapshot.load_opponent_snapshot`, and every `load_foreign_opponent` (stable
opponents, exploiter targets, distill teachers and anchors, `main.anchors`, the offline readers) is an
`InferenceMaskablePPO` by default: policy weights only, with no optimizer, no ride-along optimizer and
no rollout buffer. It refuses `learn` / `train` / `save`. Before this fix a pool load after the freeze
pre-stepped a ride-along Adam, and K6 FATALed the X26 launch at its first pool seeding. An opponent may
differ from the trainee ONLY in the declared ride-along keys (`RIDEALONG_FLAGS`), in either direction
(F-MEM); the trainee's own resume stays strict. Detail: `designs/training/learner_lifecycle.md`. The step is
K8's candidate compile region R-ride; it stays eager. **The RND variants**
(`--ridealong-rnd-variants all`, v127: `fast` / `decay` / `small` / `feat`, beside the unchanged base
RND) each step on their own Adam after the four heads. A non-finite variant disables ITSELF
(`ridealong/rndv_<name>_disabled`). `decay`'s pull toward init runs once per update, at the first step
(the accumulator marks the boundary). Their series are `ridealong/rndv_<name>_*`: loss, error mean /
median / IQR / `rel_spread` (saturation), z by class, the V-error meters, and `*_ident_ratio` (block
chimera ÷ real-row error, the identification monitor). Base logs the same `rnd_err_median` /
`rnd_err_rel_spread` / `rnd_ident_ratio`. With every variant on, the heads add 0.97 s
(1.45 %) to a 67 s update (`designs/research_state/measurements/ridealong_baseline/overhead_variants_2026-09-30.json`). B trains
only where `--opp-intent-coef > 0`, because that is what aligns the one-ahead opponent labels. The
meters are disagreement / novelty vs |V − z| (`*_auroc_err`, `*_spearman_err`, top vs bottom decile),
`rnd_z_<class>`, `adv_corr_logit`, `adv_std_starved` vs `adv_std_fed`, `q_out_of_range`. These are
MONITORING (rows the heads just trained on). The verdicts are the offline reader's,
`python -m main.ridealong_read` (`src/main/ridealong_read/`: CPU forwards on the Lane S bank and the
X4 pre-read truth turns; `--fresh-heads` reads the same checkpoint's untrained floor; every RND variant
and the amended comparisons (a)–(e); `--run-checkpoints <run>` reads a run's retained checkpoints as a
series), plus
`python -m main.ridealong_read.rnd_states` for the state-level RND reads. The pre-registered run is
EXPERIMENT_BACKLOG X26.

## The win-probability head (`--win-prob-mode`); its PBRS routes were DELETED

A calibrated **P(win|state)** supervised by the Monte-Carlo episode OUTCOME, back-filled onto every
step of an episode by `WinProbLabelCallback`; the trailing in-progress episode gets `win_mask=0` and
is **never trained toward a fabricated label**. `win_target`/`win_mask` are TRAINING-ONLY obs keys
read only by the loss, so the outcome cannot leak into the forward.

🚨 **`--win-prob-mode shaping` carries NO behavioral force, and the word has misled readers.** It is
**REPRESENTATION** shaping — the BCE gradient reaches the shared trunk; there is no gradient path
anywhere from *predicting wins* to *choosing winning actions*, because the logit is a SIDE readout
never concatenated into pi/vf. **The head is a BAROMETER, not a coach.** It is also
self-referential: its labels are outcomes under the CURRENT policy, so a habitual whiff that still
wins 55% teaches it "55%", never "the whiff was the mistake".

**The routes that were pointed at behaviour — `--win-prob-pbrs-coef` (self-φ), `--win-prob-pbrs-source` (a frozen foreign φ) and `--win-prob-pbrs-frozen` (the actor-only frozen potential) — were DELETED** (deletion pass L1, config v131; `designs/deleted_flags.md`, ledger L18659: shaped vs winprob NOT DETECTED).
**Full detail — in [`designs/training/winprob_head_and_pbrs.md`](../../../designs/training/winprob_head_and_pbrs.md).**

## The training-side VALUE SIDECAR (`--value-sidecar`, `gen3_value_sidecar_v1`)

**Detail: [`designs/training/value_sidecar.md`](../../../designs/training/value_sidecar.md).**

Every other instrument that reads the critic reads **eval** battles. This one reads the **training
buffer** — the value PPO actually used, against `win_target`, the label the BCE actually minimises.
Once per rollout at `_on_rollout_end`, a seeded 1/64 of buffer states is appended to
`<run>/value_sidecar/rows.jsonl`. Read it with `python -m main.ops.value_sidecar_read <run>`.

- **`--value-sidecar {auto,on,off}` (default `auto` = ON under `--critic winprob`)**, plus
  `--value-sidecar-fraction` (1/64) and `--value-sidecar-seed` (0). None of the three reaches
  `model_config.json`, so there is no `MODEL_CONFIG_VERSION` implication.
- 🚨 **CALLBACK ORDER IS LOAD-BEARING AND SILENT IF WRONG.** It MUST be appended after
  `WinProbLabelCallback` — that callback's `_on_rollout_end` is what replaces the `win_target` /
  `win_mask` placeholders with the Monte-Carlo label. Registered earlier it reads ZEROS and writes a
  file that looks exactly like a critic scoring an unbroken run of losses. `main.train.callbacks`
  appends them in that order, `value_sidecar_test.py` pins it, and at runtime an all-zero mask over
  a whole rollout is REPORTED (`labels_unfilled`) rather than written as data.
- 🚨 **It cannot be reconstructed after the fact.** It reads the rollout buffer, which is gone the
  moment `train()` returns. A run launched without it has no training-side read, ever.
- 🚨 **`target` is the terminal 0/1 OUTCOME, and the `outcome` column always equals it.** An OLD
  file (written while the λ-return / rollout targets existed) can carry a soft `target` and the
  `win_prob_lambda` / `win_prob_rollout_*` header fields; `value_sidecar_read` still reads them, and
  the header says which quantity a file holds. **Matching row counts are not evidence of a matching
  quantity**, and **`win_margin` is a per-turn MATERIAL margin, not an outcome**.
- 🚨 **ONE HEADER PER WRITER SESSION, not per file** — a resume used to append its rows under the
  first process's header, hiding a mid-file change of meaning. `value_sidecar_read` reads the
  header FIRST, labels every table with the quantity it scored, and REFUSES a mid-file change by
  ROW INDEX and a `--compare` across a quantity boundary. A schema-1 file and a schema-2 file at
  λ = 1.0 compare EQUAL on purpose.
- 🚨 **`critic_read` (eval) and `value_sidecar_read` (training) answer DIFFERENT questions.**
  Neither supersedes the other; a disagreement is a finding about GENERALISATION.
- ⚠️ **`opp_class` now rides the win-prob gate too**, not just the intent labels — a win-prob arm
  normally has no intent loss, so the by-opponent-class slice was otherwise empty on exactly the
  runs the sidecar exists for. It stays a label key the network never reads
  (`designs/ARCHITECTURE.md` §7). There is no opponent NAME, snapshot STEP or ladder RATING: the
  first two never reach the observation and the third does not exist at training time.
- **Cost, measured 2026-09-08 at production shape: 19.3 ms median per rollout, 0.57 MB — 0.016% of
  a hostile 120 s rollout.** 🚨 A `--debug` smoke A/B CANNOT measure this (the ON arm came out 6.5 s
  *faster*); use `src/agents/training/value_sidecar_benchmark.py`, which measures the numerator
  directly and warns on contention rather than rescaling.

## The recipe surface — `--arch production`'s TRAINING-RECIPE half (`main.train.recipe_surface`, K10(a))

The production TRAINING RECIPE is mirrored in `designs/production_config.json`'s `recipe` block:
- **`recipe.fresh`** is N0's MEASURED fresh recipe (`models/ai_v14_01_base`): every training knob it
  launched with — the rollout/update shape, `n_epochs` 10, `lr` 3e-4 with the KL controller as it ran
  (`max_lr` unset, no cosine, and `kl_controller`'s `target_kl` / `kl_factor` / `lr_factor`, which
  are constructor constants pinned against the callbacks), clip, entropy, `gamma`, self-play, the
  critic with its reward values, `vf_coef` and every supervision dose. A key that is also a
  recorded top-level mirror field must EQUAL it.
- **`recipe.fork`** is what a generalist fork changes: E5 (`n_epochs` 5 at a FROZEN `fork_lr`
  5.6e-5). Never applied by `--arch` (`--fork-lr` is refused on a fresh run); a fork's argv is
  compared with it as INFO. 🚨 5 epochs at a FRESH LR was never measured — never pair them.
- `baselines.production_config()` STRIPS the block (every arch consumer sees config fields only);
  `baselines.production_recipe_block()` reads it; `compare_production` exempts it; `--sync-config`
  carries it over.
- **`--arch production`** writes every `recipe.fresh` knob the argv did not TYPE, as if typed (after
  the ARCH surface, before `resolve_critic_mode`). "Typed" is recorded by the parser
  (`_recipe_typed`). `recipe_source` lands in `metadata.json`'s `cli_args`.
- 🚨 **`--gamma` is PAIRED with the critic, not a free recipe row** (`critic_mode.critic_gamma`:
  winprob 1.0, shaped 0.9999 — no run ever trained a shaped critic at 1.0). Under a TYPED `--critic
  shaped` the umbrella applies 0.9999 (reported `paired`); `--critic winprob` refuses a typed gamma
  other than 1.0; an untyped gamma that is not its critic's is a launch `FATAL_CONFIG`.
- **Refusal.** `checkargs`, `--dry-run` and the launcher REFUSE a FRESH argv that differs on an
  UNTYPED knob; a TYPED difference is the arm's lever (INFO); `--allow-nonproduction-recipe`
  consents.
- 🚨 **A same-run RESTART strips `--arch`.** For a run whose immutable `original_command` carried
  `--arch production`, `inherit_on_restart` resolves each untyped knob by exactly one route
  (`restart_route`), announced as `[Recipe] … from <source>`:
  - `--lr` / `--batch-size` / `--n-steps` / `--gamma` are INERT on a resume (SB3 restores them) —
    never re-applied;
  - a recorded tri-state field (critic, doses incl. `opp_intent_coef`, `policy_gae_lambda`) is `_resolve`'s to inherit;
  - a recorded value-checked field (`terminal_indicator`, `victory_value`, `draw_penalty`,
    `vf_coef`) comes from the checkpoint's `model_config.json`;
  - a knob recorded nowhere else comes from the run's `metadata.json:cli_args`.

  A value MISSING from its route REFUSES by name (`RecipeRestartError` → `FATAL_CONFIG`; `checkargs`
  and `--dry-run` report it) — never a parser or registry default. The restart also KEEPS the
  provenance tags the stripped `--arch` would have stamped — `recipe_source` from `cli_args`, and (any
  same-run restart, `arch_surface.inherit_arch_source_on_restart`) `arch_source` from the
  checkpoint's `model_config.json` — which the first restart used to null. It sits ON TOP of the general
  rule (`68850f27`: a restart inherits the surface from `model_config.json`, `opp_intent_coef`
  recorded from config v125, a pre-v125 dose migrated from `cli_args` or refused) and covers only
  what that cannot supply.
- **The cutover harness's "production" IS a launch:** `rust_core_cutover.envs.production_args()`
  runs `resolve_config` on a fresh `--arch production` argv (the trainer's own resolver), so the
  recipe arrives with the arch; `production_args_test.py` holds it to a real fresh launch on every
  mirror key, the recipe included (it used to `hasattr`-copy the top-level keys and skip `recipe`).
- Values, sources, what was left out: `designs/endstate/design_learner_recipe.md` §3.22;
  `src/recipe_doc_gate_test.py` holds the doc and the block together.

## Step size, batch size and THE DOSE (`--grad-accum-steps` · `--fork-lr` · `--adaptive-batch`)

**`--grad-accum-steps K`** runs K `batch_size` micro-batches per optimizer step, so the accumulated
gradient is the **exact** gradient of a `batch_size·K` batch at one micro-batch's activation peak;
`K=1` is byte-identical to upstream. It is also what makes the McCandlish **gradient noise scale**
free (`train/noise_scale_ratio` ≫1 ⇒ noise-limited, ≪1 ⇒ over-batched).

🚨 **`train/noise_scale` is measured on the TOTAL gradient, and on this tree the total gradient is
mostly not PPO.** A dozen dense supervised auxiliaries have agreeing per-example gradients, which
DEFLATES `B_simple` — the live runs read the total at 0.001–0.05 ("over-batched 16–1000×") on runs
whose **`train/noise_scale_ratio_policy`** read noise-limited. **Read the policy term; never size a
batch on the total.** `--adaptive-batch policy` closes that loop, moving K only — never
`--batch-size`, which would be the unbounded shape set `--compile-trainer` refuses.

🚨 **`--lr`, `--batch-size` and `--n-steps` are INERT on a resume** — SB3 restores the checkpoint's
own values, so a FORK inherits whatever the parent's KL controller had annealed to. **`--fork-lr`
pins it** and fires ONLY on a genuine fork (a checkpoint outside the run dir); `--fork-lr-freeze`
makes it constant and persists across every restart. 🚨 **The quantity that predicts a fold's
collateral is the DOSE**, `lr × n_epochs × optimizer steps per epoch / rollout rows` (=
`lr × n_epochs / (batch_size × grad_accum_steps)` when the rollout divides evenly; the learner's short
last accumulation group is a FULL-weight step, so 98,304 rows at 2,048 × 32 take 2 steps an epoch, not
1.5 — K10(c), `agents/training/dose.py`) — three folds launched
at the same `--lr` ran at 1.00× / 6.62× / 3.19× v8's rate and nothing in any of them said so. Read
it with `python -m main.dose <run>` or the live `train/dose_rate`.
🚨 **A FORK INHERITS THE PARENT'S LR BUT NOT ITS FREEZE**, so forking a `--fork-lr-freeze` run
without naming a `--fork-lr` of your own is a startup `[ForkLR] FATAL`
(`gen3_fork_lr_inherit_guard_v1`; `--allow-inherited-fork-lr` is the deliberate opt-in). The three
era-2 exploiters did exactly that — the parent's frozen 2.80e-05 annealed to 8.36e-05, median
5.5e-05, **0.39× v8 against era-1's 1.78×**. The evidence is the PARENT's `metadata.json` (its
`original_command`, else its `dose` block): the optimisation block is **not in
`model_config.json`**, which is why this is not a `combination_checks` rule. ⚠️ **The dose is a product of TWO
controllers** (KL-lr and adaptive-batch), so watch `train/dose_rate`, never either loop's own series.
**Full detail — in [`designs/training/step_size_and_batch.md`](../../../designs/training/step_size_and_batch.md).**

## LINEAGE — who forked whom (`lineage.py`, `python -m main.lineage`)

`metadata.json`'s **`lineage`** block states the fork graph instead of implying it: `role`
(`fresh`/`fork`/`fold`/`exploiter`), `fork_parent`, `teachers`, `exploiter_target`, a walked
`ancestry` and an `ancestry_stop` saying where the chain went dark and why. 🚨 **IMMUTABILITY is the
whole feature** — the existing value always wins, because a launcher restart re-derives the block
and would silently re-point the recorded parent at the DRIFTED student. 🚨 **RECORDED and DERIVED
are INDEPENDENT**: a block REGEXed out of `original_command` is a recorded GUESS, and the CLI's
header says both. **The FRESH form is explicit** (`fork_parent: null`) — "no block" and "no parent"
are different facts. A fork also inherits its parent's TB curves as a TRUNCATED prefix
(`tb_inherit.py`, `--no-tb-inherit` opts out); truncation is not optional, or parent-only progress
draws inside the fork's own step range.
**Full detail — in [`designs/training/matchup_and_lineage.md`](../../../designs/training/matchup_and_lineage.md).**

## The `signal/` group — advantage density × outcome entropy (`gen3_signal_rate_metrics_v1`)

**How much action-attributable learning signal is PPO actually receiving?** Two always-on, flagless
scalar families answer it live. Pure observability: no gradient path, no extra battle, no env call —
a handful of numpy means per rollout.

**Full detail — every flag, gate, measurement and hazard — is in [`designs/training/telemetry_scalars.md`](../../../designs/training/telemetry_scalars.md).**

## The SCAFFOLDING GAUGE — `train/scaffolding_gauge` + `python -m main.scaffolding_gauge`

🚨 **THIS GAUGE IS A `--critic shaped` INSTRUMENT AND IS DEGENERATE ON THE PRODUCTION RUN.** It
measures the divergence between TWO readouts; under `--critic winprob` there is one — the win-prob
head IS the critic, so the gauge compares a head with itself and its rank correlation is 1 by
construction. Read it on a shaped run, or on an archived one; do not read it as a scaffolding
measurement of a terminal-only run, which has no scaffolding to measure.

**Full detail — every flag, gate, measurement and hazard — is in [`designs/training/telemetry_scalars.md`](../../../designs/training/telemetry_scalars.md).**

## The POLICY DRIFT meter — refining or adopting a new strategy? (`policy_drift.py`, `python -m main.policy_drift`)

🚨 **A DESCRIPTOR, not a test.** It extends the churn probe (`churn_probe.py`: masked KL between two
checkpoints on a FROZEN probe-state set) into a per-snapshot SERIES. Each snapshot is compared with
(a) the PREVIOUS one, (b) the latest one at least `--back-steps` (10M) earlier, (c) a fixed ANCHOR (the
first snapshot recorded — pool snapshots exist only once self-play is seeded — or `--anchor <zip>`).
Per reference it records the masked KL (mean and median), the greedy FLIP RATE bucketed by the OLDER
policy's top-1 − top-2 margin (`<0.1 | 0.1-0.3 | >0.3`; single-legal states excluded), and the
ACTION-MIX shares on "choice" states (both a switch and a move legal): switch / attack / status / setup
/ hazard / recovery / phazing / self_ko, classed from the dex (`agents.gen3_data.moves`) through the
REQUEST-order `active_req_moves` block. Self-KO is a curated id set (Explosion / Selfdestruct / Memento),
because the dex carries no self-faint flag. **Cycling?** means the current policy is closer to an older
reference than the previous snapshot was. The verdict thresholds (`SHIFT_ABS`, `CONF_FLIP_RATE`,
`CYCLE_*`) are reading aids, not calibrated against a null.

**CONDITIONAL class rates are the primary action read** (`policy_drift_cond.py`). The overall share
("hazard 2%") confounds LIKING a class with how often it is AVAILABLE, so each class is also reported
among the probe states (≥ 2 legal actions) where it was LEGAL: the greedy rate, its n, a Wilson 95%
interval, the mean probability MASS on the class (a lean before the argmax flips), and the delta vs
prev / back / anchor with a PAIRED bootstrap 95% interval (same states, seeded). For hazard / recovery
/ setup a USEFUL grain is added from obs facts read through the layout, each rule verified in
deps/pokemon-showdown: hazard = opp Spikes < 3; recovery = HP < 100% (Wish: none pending — it is useful
at full HP; Rest: not asleep; Swallow: stockpiled); setup = a raised stat < +6 (Belly Drum: HP > 50%;
non-Ghost Curse assumed). The verdict names a class by its conditional change (useful grain where
defined) **only when that interval excludes 0** and |Δ| ≥ `SHIFT_ABS`; the overall shares stay as a
secondary line. Rows written before the block existed get it from `backfill <run>` (from the cached
`probs/`, no model run) into a `cond.jsonl` sidecar — `rows.jsonl` is never rewritten; `watch` runs
the backfill at startup.

- `collect <run>` freezes a probe set once per lineage, from the latest pool snapshot (or the latest
  checkpoint, with a warning to RECOLLECT once the pool seeds). `watch <run>` is DETACHABLE and
  RESUMABLE. It writes one fsynced row per snapshot to `~/gen3ai_archive/policy_drift/<run>/rows.jsonl`
  (`$GEN3AI_ARCHIVE_DIR` overrides) and never writes under `models/`, which it REFUSES. It also caches
  each snapshot's action probabilities in `probs/`, so a snapshot the sliding pool window has pruned
  still serves as a 10M-back reference. `report <run>` prints the KL/flip table, then the conditional
  tables (rates, plain-legal rates for the gated classes, Δ vs the long reference with intervals).
- 🚨 **The probe set is pinned by sha256 in `meta.json`**, and a watch on a different one is REFUSED.
  To recollect, start a NEW out dir. Comparisons across two probe sets are not comparable.
- `--source snapshots` (the default) follows the PROMOTION-gated pool, so a run that stops being
  promoted stops producing rows. `--source both` adds the periodic checkpoints. Note that the default
  anchor is then the first CHECKPOINT, which may predate self-play.

## ⚠️ Reading a belief target: `belief_supervision(...)`, never `last_*`

Cross-cutting rule for **every** belief loss below (`gen3_belief_label_only_v1`). Under
`--belief-grad-mode label_only` the extractor's `last_move_belief_logits` / `last_spread_belief` /
`last_hp_type_logits` / `last_spread_nature_logits` / `last_spread_ev` / `last_alpha_logits` stashes
are **stop-grad publications** — that is how the mode stops the policy/value gradient reaching a
belief head through any of its forward consumers. A supervised loss must therefore read its target
through **`self.policy.features_extractor.belief_supervision("<key>")`**, which returns the LIVE
tensor (and the identical object under `shaping`/`detached`).

A loss that reads the `last_*` attribute instead trains **nothing** under `label_only`, and does so
**silently** — the loss value, its gradient norm and every `belief/*` metric look completely normal,
because the loss is still computed; only the graph behind it is gone. The accessor raises a
`KeyError` on an unknown key so a typo cannot degrade into that, and
`agents/model/belief_label_only_gate_test.py::test_every_belief_loss_still_trains_its_head` is the
guard that each key still deposits gradient on its own head. The full four-route table is in
`src/agents/model/CLAUDE.md` → `--belief-grad-mode`.

## The supervised belief losses

Six supervised belief heads, all folded through **`belief_bank.py`** — one declarative ROW per head
(stash/attr/obs spec · coef key · metric prefix) and `compute(site=…)` at the THREE original
`train()` positions, where the site tag is what preserves the float-addition sequence exactly. A
seventh belief is a row, not a slice. Every head emits **`mask_rate`** under its own prefix — the
uniform per-head coverage key, comparable across heads and batch sizes where the older `n_slots`
counts are not. ⚠️ **The mask conventions TILE**: hidden-team masks HIDDEN slots, the
spread/nature-EV/hp-type heads mask REVEALED ones.

| flag | default | supervises | label |
|---|---|---|---|
| `--opp-belief-aux-coef` (+ `--opp-belief-moves-weight`) | `0.0` | the opponent's still-hidden mons — species CE + moves BCE, **order-invariant (Hungarian)** over the anonymous believed slots | `belief_species`/`belief_moves`, from agent2's own team |
| `--move-belief-mode` / `--move-belief-coef` | `off` / `0.0` | the reinjected moveset, over two DISJOINT slot populations (revealed = direct BCE, unrevealed = Hungarian) | `known_moves` / `belief_moves` |
| `--spread-belief-coef` (+ `--spread-belief-nature`) | `0.0` | the hidden derived stats the `DamageOperator` consumes; the nature⊕EV decomposition supervises it structurally | true `mon.stats`, and agent2's TRUE declared nature/EVs, guarded against them (`gen3_true_spread_labels_v1`) |
| `--hp-type-belief-coef` | `0.05` | the discrete Hidden-Power type posterior | `hp_type_label`/`hp_type_mask` |
| `--intent-label-bot-weight` | `1.0` (OFF) | a per-sample weight on the opponent-intent (α/β) LABELS produced against a **bot** | the existing `opp_class` obs key |

🚨 **Every belief label is a TRAINING-ONLY Dict-obs key read only by the loss** — the model forward
reads only `obs["observation"]`, so privileged truth cannot leak — and each builder is **fail-loud**
(a non-contiguous `species_known`, an out-of-vocab num) rather than mis-slotting supervision.
🚨 **`--intent-label-bot-weight` is confined to α/β and that is a design claim**: the other beliefs
are TEAM TRUTH, which does not depend on who is piloting, so discounting a bot's rows there would
throw away valid labels. Only INTENT is behaviour; at 1.0 the loss is **bit-identical**. Every
structural toggle is version-checked and fresh-only; every `*_coef` is training-only and **read back
on a flagless resume**.
**Full detail — in [`designs/training/belief_losses.md`](../../../designs/training/belief_losses.md).**

## `stats.py` — the package's SHARED small-sample statistics

**Where a stateless estimator lives once the second consumer exists** — `wilson_ci`, `spearman`,
`cluster_bootstrap_ci` / `cluster_bootstrap_diff_ci`, `sd_true_excess` and the `MIN_CELL_N` /
`MIN_SUBCELL_N` floors. Pure NumPy in, floats out: no labels, no battles, no checkpoints, no
filesystem, no torch, no RNG except an explicitly seeded bootstrap. That is the admission rule; a
helper that has to know what a *decision* or a *bias map* is belongs beside the instrument that owns
the concept. ⚠️ **Three near-siblings elsewhere are deliberately NOT merged into it** (the
NaN-refusal pair in `scaffolding.py`, `winprob_finetune.label_noise_variance`,
`main/q_amortization.spearman`) — the reasons are in the module docstring and in
[`designs/training/offline_meters.md`](../../../designs/training/offline_meters.md), so nobody
"de-duplicates" a shipped instrument's output by accident.

## `cf_audit` — the counterfactual audit instrument (`cf_audit.py`)

**Three modules, one instrument**: `cf_audit.py` owns the frame, the sampler, the label schema, the
bias map and the CLI; `cf_audit_render.py` turns a finished bias map into markdown (its two
formatting rules are the ABSENT-vs-ZERO ones — a row of zeros makes "no head" and "no uncertainty"
read identically); `cf_audit_twin.py` holds the twin-head paired read and the shadow read.
`cf_audit` re-imports every name, so the historic import paths still resolve.
**Full detail — in [`designs/training/cf_grounding.md`](../../../designs/training/cf_grounding.md).**

## Prefix-sharing materialization (`obs_materializer.materialize_branches`)

K counterfactual arms of one decision share an identical prefix; the materializer replays it once,
snapshots the player's whole battle/tracker state at the branch decision, and restores it per arm —
**exactly equivalent to per-arm `materialize_decisions`, bit-for-bit** (59/59 arms byte-identical,
15.4 → 5.3 ms per arm). The per-arm restore is serialized ONCE and rebuilt per arm rather than
deep-copied (1.98 → 0.22 ms). ⚠️ A graph that will not pickle **falls back to deepcopy and says so
once on stderr** — a 9× regression nothing mentions is the failure shape this tree keeps eating.

🚨 **AN OFFLINE REPLAY'S BATTLE TAG IS ALWAYS UNIQUE, AND THAT IS A CORRECTNESS PROPERTY**
(`gen3_recon_tag_collision_v1`, 2026-09-19). `_next_tag` used to hand the caller's `battle_tag`
back verbatim, and the search passes the LIVE record's tag — so an offline replay ran in the LIVE
BATTLE'S ROOM, and `search_dividend.record.install_choice_tap` (a process-wide patch whose only
discriminator is that room) recorded every `/choose default` the replay player emits when its
action list runs out as a choice the LIVE player had made. Measured: ~1 per materialized ARM, so
the `playoff` arm's nested rollouts replayed a prefix that was not the battle and 64 of 66
playoffs died. It was filed as a rust defect and was not one — node's bridge re-requests where
rust fails loud, so the node cell ran the same wrong rollouts and reported a clean number. Pinned
by `main/search_dividend/recon_tag_isolation_test.py`.
**Full detail — in [`designs/training/cf_grounding.md`](../../../designs/training/cf_grounding.md).**

`materialize_branches` (the prober's counterfactual lookahead) replays the shared prefix ONCE
(`open_branch_fork`, which freezes a `_PlayerSnapshot`) and runs every arm off it
(`materialize_branches_from`); `clone_pins.py` is the ONE definition of WHICH objects a per-arm
clone must SHARE rather than copy (a `logging.Logger`, a `MappingProxyType`, the append-only
immutable records, the `GenData` singleton) and of the pinned-pickle freeze/thaw that makes a
clone ~9× cheaper than a `deepcopy`. **The SEARCH no longer materializes anything in Python** — its
successors are Rust-core versions whose rows the driver encodes (`gen3_core_search_v1`); the
search's protocol road, its one-sided VIEW road (`view_successor.py`, the M1 event folder, the
per-decision fork caches) and `core_successor.py` are DELETED (Rust Core deletion pass, program
§4 M2). `materialize_branches`, `_PlayerSnapshot` and `clone_pins.py` leave at M7 with the prober.

🚨 **`EpisodeTracker.record` and `update_progress_clock` are SPLIT, not copied.** `record_context`
and `advance_window` are their bodies once the context and the event windows exist; `record` /
`update_progress_clock` are the poke-env-battle wrappers.

## Counterfactual win-prob grounding (`--cf-records` / `--cf-winprob-coef`, `gen3_cf_label_plumbing_v1`)

The **trainer-side plumbing** for `designs/ai_v10/design_counterfactual_value_grounding.md` — its gate
**G3**, which is explicitly "tap + buffer + flags at coefficient zero, byte-identity gated". Rung **R1**
only: tight Monte-Carlo P(win) labels, delivered to the **win-prob head**. The label PRODUCER is a
separate, out-of-process program (`cf_producer.py`, § *The label PRODUCER DRIVER* below);
**nothing in this section produces a label**, and the two halves share only a file format.

🚨 **THE LABEL SUPPLY IS A DECLARED STARTUP RESOURCE** (`gen3_supply_guard_v1`, `cf_supply.py` +
`cf_supply_callback.py`, 2026-09-30). `ai_v12_12_ladder_cflabels` trained 10M steps at
`--cf-winprob-coef 0.5` with ZERO labels because nobody started the producer. Now any live
cf-buffer coefficient (`cf_supply.CF_CONSUMER_COEFS`: cf_winprob / cf_evidential / cf_twin /
cf_shadow / q_winprob / q_winprob_onpolicy) makes the supply mandatory: under
`--cf-label-supply producer` (default) **the trainer spawns `cf_producer` itself** (bound to it by
`--parent-pid`, one per run by `<run>/cf_producer.lock`, `--q-labels` added when a Q coefficient
is live, `--cf-producer-args` for the rest) and REFUSES without `--cf-records`; under `external` it
REFUSES unless a producer holds the lock, printing the command. In flight, a stream that accepts
nothing for `--cf-supply-starve-cycles` (5) cycles AND `--cf-supply-starve-minutes` (30) once a
checkpoint exists — or a spawned producer that exits — raises `CfLabelSupplyError` → exit
**`FATAL_SUPPLY` (5)**, which the launcher does not restart; the end-of-run summary is LOUD at zero.

**Full detail — every flag, gate, measurement and hazard — is in [`designs/training/cf_grounding.md`](../../../designs/training/cf_grounding.md).**

## The STALL-TAIL HARVEST + head-repair pipeline (`main.harvest` → `winprob_finetune` → `main.harvest_meter`)

An **offline, three-stage pipeline** that manufactures win-probability labels for the population
probe O convicted, fits the win-prob head on them with the trunk frozen, and measures the result
against a battle-level holdout. It is the ai_v12 head-repair backbone and it writes nothing into
`models/` — artifacts land under `utils.paths.harvest_dir()` (repo-root `harvest/`, gitignored,
`$GEN3AI_HARVEST_DIR` overrides).

**Full detail — every flag, gate, measurement and hazard — is in [`designs/training/stall_tail_harvest.md`](../../../designs/training/stall_tail_harvest.md).**

## `replay_imputation_probe` — the own-side imputation meter (`replay_imputation_probe.py`)

A **meter, not a lever**: how far would our observation move if the only thing we knew about our OWN
side were what a public Showdown replay had shown by now? It plays reproducible bridge battles,
overwrites our not-yet-revealed moves / item / spread with the top Smogon-prior candidate, and
re-encodes — with truth re-encoded a THIRD time after the restore and required to be bit-identical,
because a leaked restore would make every later "truth" a previous decision's imputation. There is
**no transcoder and no `|request|` synthesis here**, deliberately.
**Full detail — in [`designs/training/offline_meters.md`](../../../designs/training/offline_meters.md).**

## Two DELETED subsystems

The **latent-belief loss** (`--opp-belief-latent-coef`, the SimSiam predictor, v75) and the
**public-replay value aux V_pub** (v88 `gen3_dead_flag_purge_v1`) are **gone** — the first was a side
readout never fed forward that cost ~13% of the train step, the second measured NULL and was never
ON in a production generation. A checkpoint recording either is REFUSED by its migration. The
reasoning generalises to every aux head on this trunk and is kept as closed history:
`designs/research_state/claude_md_archive/training_leaf_deleted_subsystems_history.md`.

## Exploiter distillation (`--distill-teacher` / `--distill-coef` / `--distill-value-coef` / `--distill-value-feat-coef`)

`gen3_exploiter_distill_v1` — pour a frozen per-team SPECIALIST into the generalist so it learns to
PILOT that team, closing the amortization gap the self-play average cannot. `--distill-teacher`
takes `TEACHER:TEAM` colon pairs (a bare run dir → that run's **LAST SNAPSHOT**; the `🧪 [DISTILL]`
line states the resolved file, its step and the rung per teacher); the env emits a training-only
`distill_mask` key, and per teacher `distill_coef · KL(π_teacher ‖ π_student)` is folded masked to
that teacher's states, the per-teacher mean-KLs AVERAGED so a small-coverage teacher still
contributes comparable gradient. `--distill-team-bias` (0.4) keeps the rest as pool rehearsal. OFF
is byte-identical; training-only, NOT version-locked, inherited on a flagless resume.
**Full detail — the off-slice anchor, the stop rule, the advantage-gated/action-form target and the
rank tripwire — is in [`designs/training/exploiter_and_distillation.md`](../../../designs/training/exploiter_and_distillation.md).**

## Search-as-teacher (`--search-teacher`, `teacher/` package)

Selective **Expert Iteration**: each cycle, search + rollout-confirm the worst loss craters of
recent eval traces and distil the VERIFIED-better action into the policy via an advantage-weighted
CE aux loss (AWR), or the full improved distribution via `--opd-coef` KL. Off by default and
byte-identical. The "expert" is the prober's `better_line` beam plus the rollout-confirm tiers.

🚨 **The distilled advantage is the CONFIRMED win-rate improvement, never the critic's optimistic
backed-up value** (the Spore 95%-vs-62% lesson), and an opponent that cannot be resolved exactly is
**SKIPPED, never approximated** — distilling "A\* beats a proxy" is a soundness failure, not a
degrade. 🚨 **BOTH PHASES OF A CYCLE RUN IN CHILDREN**; selection used to run INLINE on the training
step (measured 48.1 s over 9 traces, 350.2 s over 60) and "non-blocking" was half true for a year —
`teacher/step_block_ms` now records what the teacher costs the training step as a series.
🚨 **A FAILED SELECTION IS A REPORTED CYCLE, not a silence** (`teacher/selection_failures_total`).
⚠️ **`--search-teacher-mode winprob_oneply`** (ai_v12 routes 2+3, nothing has run it) swaps only the
selection and production halves, and its **CONFIRMATION step is a REQUIREMENT, not a refinement** —
the WINNER'S CURSE: a separation procedure certifies the leaf's residual differential bias as much
as signal, and unlike PBRS a distillation target has **no invariance shield**.
**Full detail — in [`designs/training/search_teacher.md`](../../../designs/training/search_teacher.md).**

## Process liveness guards (`watchdog.py`)

Two daemon-thread watchdogs keep a hung/abandoned run from lingering:

- **`start_subprocess_watchdog`** — for the `SubprocVecEnv` path. A crashed worker leaves the
  parent blocked on a pipe `recv` forever; this thread polls `processes` and `os._exit(1)`s the
  moment a worker dies with a nonzero exitcode. Started *after* env construction (and, in
  self-play, after `_maybe_engage_self_play` rebuilds the env), right before `learn()`, and
  **stood down the moment `learn()` returns** (`model_build` sets its shutdown event before the
  final save/eval; the thread checks the event before the workers). Without that, a worker
  SIGTERM'd in teardown (exitcode −15) turned every clean finish into a launcher "crash #1"
  (`watchdog_teardown_test.py`). It is a **no-op on the `--debug` DummyVecEnv path** (no worker
  processes to watch).
- **`start_orphan_watchdog`** — for the `--debug` smoke path, which has no worker watchdog. A
  smoke run is a child of the launching shell/agent; if that parent dies the run is orphaned
  (PPID changes) and a hung smoke (e.g. a vanished `9XXX` server) would otherwise sit as a
  multi-GB zombie indefinitely. This thread captures the launching PPID up front and `os._exit`s
  when `os.getppid()` *changes* (by-change, not `== 1`, so PID-namespace subreapers count).
  Started early in `main()` inside the `if args.debug:` block — before team/env/server setup —
  so a startup hang is covered too. **Real launcher-managed runs keep a live parent and never
  arm it.** Regression test: `watchdog_test.py` (subprocess-driven orphan + no-false-fire).

## Showdown port threading (the `server_config` seam)

`train_rl_agent.py --showdown-port <port>` builds **one** `ServerConfiguration` in `main()`
via the single constructor `localhost_server_configuration(port)` (in
`poke_env.ps_client.server_configuration`) and threads it to **every** Showdown client —
the training-env players (carried into the `SubprocVecEnv` spawn workers via the env-factory
closures), eval, and self-play. Every player-creating callback takes a `server_config` param
(defaulting to port 8000 for standalone use) and builds its players from it — **never** from a
bare `LocalhostServerConfiguration` constant. `server_port_threading_test.py` is the
regression guard: it fails if any of these callbacks hardcodes the default port instead of
threading the configured one (the original bug had the now-retired replay recorder connecting
to :8000 while training ran on :8001; eval forensic traces inherit the same guard).
There is no environment variable; `train_rl_agent.py`'s own default is 8000, but the **launcher**
overrides it to 8001 before forwarding (see `src/main/launcher/CLAUDE.md`). The launcher
forwards `--showdown-port` verbatim (it strips only launcher-owned flags).
