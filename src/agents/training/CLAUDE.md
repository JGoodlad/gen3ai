# CLAUDE.md — Training (`src/agents/training/`)

Callbacks, the reward's declarations, the belief / value auxiliaries, the owned PPO loop and the eval pipeline.
Training and eval run in process on the Rust core; every fold, label, reward and trace record is the core's
(`designs/rust_sim/trackers.md`). How to launch: `designs/ops/training_runbook.md`. This leaf holds the hazards
to know before touching a subsystem; **`designs/training/<topic>.md` owns the detail** and changes in the same
commit as the code.

## Where the detail is

| I am about to touch… | Read |
|---|---|
| a trainer / launcher flag (add, keep, delete) | `designs/ops/flag_census.md` (`src/main/flag_census_test.py`); a deleted flag gets a `designs/deleted_flags.md` row |
| the PPO step, the fold order, the owned loop, `--diagnostics-every` | `designs/training/ppo_step.md` |
| the K9 gates (learner golden, behaviour check, non-finite), the CUDA memory trend | `designs/training/learner_gates.md` |
| the declared lifecycle (K6): the freeze, `@startup_builder`, the update-fit check, reseeding | `designs/training/learner_lifecycle.md` |
| the reward (the terminal alone), the no-progress clock | `designs/training/reward.md` |
| the in-loop eval, the eval count ledger, `main.h2h`, `main.plateau`, `main.cycle_monitor` (the plateau's Tier 2), baselines, ELO, traces | `designs/training/eval_rules.md`, then `designs/training/eval_and_rating.md` |
| self-play, the snapshot pool, stable opponents | `designs/training/self_play_and_pool.md` |
| exploiters, the best-response gap, the rank tripwire | `designs/training/exploiter_and_distillation.md` |
| `--team-mix`, per-team win rates, team blocking | `designs/training/team_curriculum.md` |
| the win-prob critic, the 250-turn cap, `--vf-coef` | `designs/training/critic_and_value_losses.md` |
| the win-prob head (`--win-prob-mode`) | `designs/training/winprob_head_and_pbrs.md` |
| the ride-along heads, strict checkpoint loads, opponent loads | `designs/training/ridealong_heads.md` |
| the value sidecar | `designs/training/value_sidecar.md` |
| a supervised belief loss | `designs/training/belief_losses.md` |
| the recipe surface, gradient accumulation, the noise scale, the dose, `--fork-lr` | `designs/training/step_size_and_batch.md` |
| a fork that GROWS the network (`--grow trunk_layers=N`: identity-init growth, the optimizer rule, the fork-time gate) | `designs/model/versioning.md` "Declared growth at a fork" |
| the `MatchupSpec`, which file a run spec names, lineage, TB inheritance | `designs/training/matchup_and_lineage.md` |
| TensorBoard: the census, the currencies, the WIN-PROB dashboard, capacity telemetry | `designs/training/telemetry_scalars.md` |
| a supply guard (`FATAL_SUPPLY`) | `designs/training/supply_guards.md` |
| the fork arm (`--fork-fraction`, OFF) | `designs/training/forks.md` |
| the Rust collector, the env core, the keyed draw, cadences | `designs/training/rust_collector.md` |
| `--compile-trainer`, compile regions, the sentinel / canary / cache, `--trunk-precision` | `designs/training/compile_flags.md` |
| `stats.py`, `main.policy_drift`, other offline meters | `designs/training/offline_meters.md` |
| the stall-tail harvest | `designs/training/stall_tail_harvest.md` |

Deleted subsystems (the Python env core and collectors, the shaped reward, PopArt, distillation,
search-as-teacher, the counterfactual stack): a typed flag is refused with its `designs/deleted_flags.md`
reason, and a checkpoint that recorded one is refused by `model_version/retired_levers.py`.

## Metrics

- **Read a scalar's unit first.** Under the win-prob critic, returns, `train/value_loss` and explained
  variance are in probability units, so they do not compare with a pre-era shaped run's tags of the same name.
- **A tag whose source is absent is not emitted**, so a gap means no source, never zero. `edge/*`, `cell/*`
  liveness and `grad/*_share` are not effect sizes.
- **A metric key has one producer, written from one call site**, and stays distinct in every output. Startup
  refuses a colliding key set, and a second producer raises under pytest. Test a new key through the real
  logger, not a dict.
- A callback that publishes a quantity another callback's decision changes registers after it (`build_callbacks`).

## The PPO step (`instrumented_ppo/`)

- 🚨 **The fold order is fixed** (`instrumented_ppo_hub_contract_test.py`): R1 (`micro_step`) then the eager
  tail. A tail fold that runs its own extractor forward clobbers the stashes earlier steps read, so a stash
  reader moved below one scores the wrong states.
- **R1 is a static-shape program**: no host read (`.item()`, `float(t)`, `bool(t)`), no boolean-mask indexing,
  no Python branch on a tensor. A new per-micro-batch host read goes through `instrumented_ppo/host_reads.py`
  (`host_sync_guard_test.py`).
- **The loop is ours** (`instrumented_ppo/loop.py`): the dump stays before the update, hooks register only at
  the declared points (`loop_hooks.py`), and a callback that dumps mid-rollout uses
  `logger_scope.isolated_dump`, or the KL→LR controller skips readings.
- **Build everything at startup** (K6, `learner_lifecycle.py`): an optimizer, parameter, module, buffer or CUDA
  stream acquired after the first rollout is a typed fatal. No global RNG is seeded after the freeze; a stream
  that must repeat owns its generator.
- **Played bits:** `played_bits_sentinel_test.py` (routine) pins 17 seeded self-play decisions' actions + fp32
  log-probs and the played teams' pack; a change to model numerics or engine-facing data moves it — re-record
  by its docstring's procedure (cause + log line) and run the slow reveal file too (`designs/ops/testing.md`).
- **K9:** `learner_golden_test.py` pins one update's bytes; re-record it (`python -m
  agents.training.learner_golden record --reason "..."`) only for a deliberate change. A non-finite loss,
  gradient or KL is `NonFiniteLearnerError`: never `nan_to_num` it or skip the step.

## Opponents, eval, ratings

- **Resolve every model reference through `fixed_opponent_pool.resolve_model_ref`**, the one choke point (a bare
  run dir means its last snapshot). `MatchupSpec.spec_hash()` is a measurement-regime tag: built once, never
  re-derived; two hashes are not metric-comparable.
- **Eval regimes are recorded and inherited** (`--eval-mirrored-pairs`, `--promotion-sprt`, greedy sentinels):
  read the run's `[EVAL REGIME]` line, never assume, and never flip a default on your own.
- A protocol identifier carries the nickname, not the species. A forensic recorder must never take down a run.
- `main.h2h` keeps the player in seat p1, so a checkpoint against itself reads 0.5 plus a seat effect.
- **Load a baseline with `baselines.load(name)`**, never a bare `MaskablePPO.load`.
- **The games that selected a snapshot never rate it**; bump `LADDER_FITTER_VERSION` whenever the fit changes
  what a rating means.
- Snapshot, `best_model` and pool files may be hard links to one checkpoint: store with
  `weights_link.link_or_copy` and `unshare` before an in-place save; never write through one.
- A live lever's supply line goes through `lever_supply.loud`. Team corpora live in `src/utils/team_corpora/`,
  never `data/` (a pinned run reads `data/` from main).

## The collector, compile, value

- **A step-counted cadence counts total env steps** (`num_timesteps`), never `n_calls`.
- 🚨 **Under `--overlap-collect` (X14) the host loop runs on a WORKER THREAD during `train()`**: code reached from
  `RustCollector.host_step` must never touch the learner, the logger, a callback or `num_timesteps` (the
  main thread replays those after the join). `designs/training/rust_collector.md` "Overlap (X14)".
- A fresh model is built under `utils.torch_state_guard.single_thread_build`; the spaces have one builder,
  `agents.training.trainee_spaces`.
- 🚨 **`src/agents/model/compile_control.py` is the only runtime module that touches `torch._dynamo`.** A new
  `torch.compile` site calls `ensure_hermetic_cache()` first; a trainer flag that shapes a compile joins
  `COMPILE_ARG_KEYS`.
- After an obs-layout change, regenerate the compile parity fixture (`python -m
  agents.model.compile_parity_fixture --write`). A gate or canary forward that is never backpropagated keeps
  nothing for a backward.
- R1 alone runs bf16; a new forward of the live learner policy that must agree with the rollout (a behaviour
  log-prob in a test or tool) runs under `trunk_precision.fp32_scope`.
- Do not launch `--learner-cudagraphs on`: it fails on the GPU (`designs/training/compile_flags.md` "Lever C").
  Nothing in a loss, an optimizer or a parity bar may read `--canary-shift-max`.
- **R1's margin rule judges an arm by its NAME**: only `compile_regions.YARDSTICK_ARMS` (`eager`, `fp32 reference`)
  are yardsticks (out of margin = a recorded event and one re-run at M′); any other name is judged as COMPILED. A new
  yardstick arm joins that tuple, never a new spelling (`compile_flags.md` "The yardstick is not the compiled arm").
- **A draw is a not-win, and the 250-turn cap is a terminal**, not a truncation.
- 🚨 **A belief loss reads `features_extractor.belief_supervision("<key>")`, never a `last_*` stash**: under
  `--belief-grad-mode label_only` the stash is stop-grad and the loss trains nothing while every metric looks
  normal (`agents/model/belief_label_only_gate_test.py`).
- The value sidecar reads the rollout buffer, so it cannot be reconstructed after the fact.
