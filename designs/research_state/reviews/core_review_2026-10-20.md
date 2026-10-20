# Independent core review, 2026-10-20

TECH_DEBT_BACKLOG P2 "ONE INDEPENDENT CODE REVIEW of the newest core modules", widened by the brief to the
training-hot-path modules landed 2026-10-19. Reviewed at `origin/main` `ba366dc4b` by reviewers that wrote none of
the code (one coordinator, four `opus-high` readers, one unit each). Scope: **correctness only** — wrong results,
silent data loss, races, state leaks across updates / restarts, wrong device or dtype, resume / fork inheritance,
vacuous checks. No style.

**Method.** Each finding was attacked for a refutation (a caller-side guard, a production config that cannot reach
it, an existing test) and then demonstrated with a CPU repro. CONFIRMED = a repro showed the wrong behaviour;
PLAUSIBLE = read from the code, not demonstrated. **Nothing was run on a GPU** (no lease): every CUDA-only path
(staged copies, streams, cudagraph routes, the bf16 trunk on the card, the allocator counters) is as its commit left
it. The repro scripts lived in `/tmp/core_review/{A,B,C,D}/` (scratch, not kept); each finding states its repro.

## Findings, by severity

| id | sev | verdict | where | the failure | disposition |
|---|---|---|---|---|---|
| A1 | MED | CONFIRMED | `agents/training/update_fit.py` `dry_update` (+ `instrumented_ppo/train_setup.py` `_overlap_prepare`) | With `--overlap-collect` on CUDA, `model_build` attaches the collector's overlap runner (`_start_rust_env`) BEFORE `check_update_fits`. The dry update's `train()` reaches `proximal_transform`, which needs `model._overlap_prox` — set only by a real FILL — and raises `OverlapError` (only an OOM is caught): **every X14 launch dies at startup.** Repro: `dry_update` on the learner golden with a runner attached → `OverlapError … got None, want (16, 4)`. X14 had never run on a card. | **FIXED** in this unit: the dry update hands `train()` an identity rewrite (`_overlap_prox := log μ`, w = 1); `update_fit_test::test_a_dry_update_runs_with_the_overlap_runner_attached` fails on revert. Residual: the dry update measures memory WITHOUT the overlapped worker running beside it (row). |
| A2 | MED | CONFIRMED (mechanism); size unmeasured | `agents/training/rust_rollout/overlap.py:395` with `instrumented_ppo/micro_step.py:172-173` | Decoupled PPO folds w into the buffer's advantage (`A := w·A`), and `normalize_advantage` then normalises w·A per micro-batch. The docs called this "a small rescale and baseline shift"; it is a **bias**: for an advantage independent of the action the expected gradient is (E[A]/s)·Σₐ(π_prox(a) − μ(a))·∇log π(a) ≠ 0, a steady push along μ → π_prox. Normalise-then-weight gives 0. Repro (real `proximal_transform` + `micro_step`, 3 actions, E[A] = s, TV(π_prox, μ) = 0.1): logit grad as coded `[0.0968 -0.0489 -0.0479]`, closed form `[0.0964 -0.0482 -0.0482]`, normalise-then-weight `[0.0004 -0.0007 0.0002]`. X14 is default OFF. | row (P1: gates any X14 arm's verdict) |
| C-3 | LOW | CONFIRMED (code order) | `instrumented_ppo/ppo.py:291-294` before `train_setup.py:198`; `ridealong_terms.py:285` | Under overlap `_overlap_prepare` weights the advantages BEFORE `advantage_density_metrics` and the ride-along `adv` head read them, though both document "RAW GAE advantages": the `signal/*` tags and that head's TARGET see w·A. Value targets untouched. | row (folds into A2's fix: keep A raw, pass w separately) |
| C-2 | LOW | CONFIRMED | `main/train/recipe_surface.py` (`ALL_ROWS`) | `--overlap-collect` changes the objective but is not a recorded-recipe row: a manual `--model` resume or a fork of an X14 arm that omits the flag silently trains serial PPO; nothing compares against `metadata.json` `env_core.overlap_collect`. Launcher restarts are safe (`checkpoint._set_arg` keeps the typed argv). Repro: `"overlap_collect" in recipe rows` → False. | row |
| B-F1 | LOW | mechanism CONFIRMED; production reach PLAUSIBLE | `agents/training/snapshot_pool.py:302-305` (`add_from_path`), `:545-554` (`_write`), cache `:385-403`; `agents/training/rust_env_opponents.py:293` (`SlotFamily.admit`) | Re-promoting a pool snapshot at a step the pool already holds overwrites `snapshot_<S>.zip` but neither drops `_model_cache[path]` nor reloads the resident T2 slot `pool:<S>` (`admit` skips resident ids). Reach: a crash-restart from checkpoint C < S replays to S and promotes again. Self-play then plays the OLD weights under a name whose file (sentinels, PFSP, snapshot ladder) holds the NEW ones. Repro (stub loader hashing the file): disk `f1f0595b`, served `83a02a7e`; T2 slot loads for `pool:1000` across two refreshes: 1. | row |
| D-1 | LOW | CONFIRMED | `utils/gate_lock.py:255` (`gate_slot` → `HS.heavy_slot` without `timeout_s`) | `gate_lock.sh --timeout-s S` did not bound the new heavy-slot wait, and the gate stayed HELD while it waited: with a run live and both heavy slots held every other gate queued behind an idle gate, and exit 4 never came. Repro (temp dirs): `--timeout-s 2` killed by an outer `timeout 15` (rc 124). | **FIXED** in this unit: the heavy-slot wait gets what remains of `timeout_s`; `gate_lock_test::test_the_timeout_bounds_the_heavy_slot_wait_too_and_frees_the_gate` fails on revert. |
| D-2 | LOW | CONFIRMED | `utils/heavy_slots.py:226-266` with `utils/mem_cap.py` `run_capped` | Only the `mem_cap` wrapper holds the slot's flock; the job runs in its own systemd scope and survives the wrapper. A SIGKILL / OOM kill of the wrapper (`timeout -s KILL mem_cap.sh …`) frees the slot while the job runs, so a third heavy job starts. Repro (temp dir, N = 1): wrapper killed → job alive, slot free, a second job takes slot 0. | row |
| C-1 | LOW | PLAUSIBLE (needs a GPU) | `agents/inference/service/service.py` `_frozen_guard` (~161-185); `main/launcher/child.py:50` | The rust_collector doc already records that T2's guard reads process-wide counters and that RESERVED growth during an overlapped flush is warn-only. Not recorded: `cuda_segments_after_freeze` is FATAL today, and without `expandable_segments:True` (the launcher only `setdefault`s it; a direct trainer launch or a user env that sets `PYTORCH_CUDA_ALLOC_CONF` lacks it) a learner segment allocated during a worker flush is booked to T2 → `LifecycleViolation`. | row (with A2's X14 rows) |
| D-3 | LOW | PLAUSIBLE | `main/ops/cuda_build_validate.py:149-160`, `scripts/ops/diag_bf16_canary.sh:74` | The documented order `gpu_lock.sh mem_cap.sh <gpu job>` holds the GPU lock idle while the job queues for a CPU heavy slot (the inner `timeout` does not cover that wait), unless the orchestrator sets `GEN3AI_BLOCKING`. Not a deadlock (gpu_lock refuses at once), but the starvation the slots were built to avoid. | row |

## Notes (no row)

- **H2 (latent).** `opponent_parity.weights_fingerprint`'s process cache key omits the trunk's precision class, so an
  fp32 pass would be reused for the same weights under a bf16 trunk. No path reaches it today: every opponent and
  reader builds a fresh fp32-class policy from a state dict.
- **reserved_growth.** Once `RESERVED_GROWTH_POISONS` is armed, the collector's and eval executor's raise cannot fire
  first: T2's own `_frozen_guard` fails on the same growth and shuts the service down. In today's warn mode the
  downstream checks warn as intended; their tests drive the counters directly, so they pass either way. Worth knowing
  when the switch is armed; not a bug.
- **Restart RNG.** Every restart re-seeds the global numpy / torch RNGs to `--seed` (sb3's `set_random_seed`), before
  the freeze guard is armed, so each restart replays the same PPO minibatch-permutation sequence. The permutation is
  data-independent; judged harmless. Other global-RNG users were not audited. The collector uses a per-restart seed.

## Reviewed and found clean

- **PPO loop** (`instrumented_ppo/loop.py`, `ppo.py`, `micro_step.py`, `device_batches.py`, `train_setup.py` bar A1,
  `hparams.py`, `host_reads.py`, `phase_hook.py`, `value_terms.py`, `aux_terms.py`, `learner_gates.py`,
  `diagnostics_cadence.py`; `loop_hooks.py`): loop order and `_setup_learn`'s remaining-steps target on resume,
  dump-before-update plus the final dump; the clipped surrogate, KL early stop, gradient-accumulation scaling and the
  trailing partial-group rescale; deferred host reads and K9(c) verdicts before every optimizer step; per-micro-batch
  advantage normalisation with the one-row guard; identical permutation draws across device-batch modes; the
  first-update latch re-armed after the dry update; LR / clip schedules constant across restarts.
- **Learner lifecycle** (`learner_lifecycle.py`, `lifecycle_decl.py`, `global_rng_guard.py`, `update_fit.py`'s
  `preserved_learner`: modules, optimizers, RNG, attributes and logger restored exactly).
- **Compile regions** (`compile_regions.py`, `compile_trainer.py`, `compile_control.py`, `compile_config.py`,
  `compile_cache_shared.py` incl. X45's key): arm independence, clip relocation on the micro-step's own normalised
  advantages, golden-row loss re-evaluation (fork mask included), the bf16 drift / envelope checks, the unmoved-
  parameter rule, the fp32 reference arm's class swap restored before any compiled call; NaN fails every verdict.
- **Inference-only opponent load** (`compile_opponents.py`: 12 extractor instances → one cache entry, no eager
  fallback; `instrumented_ppo/inference.py`, `strict_load.py`: no non-strict path; `snapshot.py` opponent / foreign
  loads on CPU behind the arch gates; `fixed_opponent_pool.py`; `snapshot_pool.py` bar B-F1).
- **X14 overlap** (`overlap.py` bar A2, `collector.py`'s replay, `consistency.py`, loop / probe placement): runner
  protocol ordering, behaviour snapshots swapped and restored bit-exactly (sb3's aliased extractor included), K9(b)'s
  version arithmetic, the rewrite before any optimizer step, the replayed `on_step` events / infos identical to a
  serial collection, `TorchFunctionMode` thread-local (no leak into the worker).
- **X45 grouped forward** (`grouped.py`, `engine.py`, `service.py`, `spec.py`): slot → stacked-weight row by view (an
  in-place load is read on the next call), padding rows masked legal, index bounds, lane / stream order, only real rows
  scattered back, every load parity-gated through the grouped path.
- **B-GROW** (`growth.py`, `grow_fork.py`, `fork_arch.py`, versioning v159, `model_build`, checkargs / dry run):
  production-shape 3 → 4 after two AdamW steps is bit-identical (logits and values, fp32 and bf16 trunk, real and random
  legal masks); a save → strict reload → by-name optimizer realignment has 0 state mismatches; a same-run restart of a
  grown fork does not re-grow; a pool interrupted mid-growth resumes correctly; `check_compatible_grown` bypasses only
  the declared fields.
- **heavy_slots** bar D-1 / D-2 / D-3: non-blocking flock only (no check-then-lock race), `O_CLOEXEC` fds, the holder
  marker verified against the ancestor chain and `/proc/locks`, gate-before-slot lock order with no deadlock.

## Hazards met during the review

- The heavy-slot queue was **12-13 jobs deep** beside the live run (`rb_toy_ent_s4002_deep`): two reviewers' pytest
  runs of the scope's test files (`overlap_test`, `grouped_test`, `lifecycle_reserved_test`, the compile-region files)
  never got a slot and were stopped by their own PIDs. Those existing test files were therefore NOT re-run by the
  reviewers; the coordinator ran `overlap_test`, `update_fit_test`, `gate_lock_test`, `heavy_slots_test` and
  `mem_cap_test` directly (light, nice 19 on the agent cores): 50 passed.
- No GPU: the CUDA-only behaviour of X14 (streams, allocator attribution), X45 and the compile regions on the card is
  unverified here.
