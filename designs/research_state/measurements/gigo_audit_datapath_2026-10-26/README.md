# GIGO audit of the training data path, Rust self-play to the PPO update (2026-10-26)

An adversarial audit of the path from the Rust collector's rows to the PPO update. It covers the trajectory
buffers, GAE across game ends, the 250-turn cap, restarts and forks, the behaviour log-prob, the opponent-pool
and team draws and their seeding, the minibatch shuffle, the value targets, the entropy and KL terms, the LR
schedule and `--min-lr`, and what a resume restores. Method: read every stage critically, then check what can be
checked by EXECUTING the real collector on the CPU (`harness.py`, below). A suspicion that was not reproduced is
listed as PLAUSIBLE. Code read at `13c704e6`; fixes land in this directory's commit.

**Verdict: no GIGO bug in what the network trains on.** On the CPU harness, every buffer row is its played row:
the observation, mask, labels, action, behaviour log-prob, value, outcome label and GAE all match. Three bugs
were CONFIRMED, and none changes a loss. One makes per-class telemetry wrong. Two are seeding hygiene: they
answer the seed-independence question (§ Seeds), and neither undermines a past A/B.

## The harness (`harness.py`, the evidence for the CLEARED rows)

`harness.py` runs the real `RustCollector` (self-check build, CPU, T2 eager, a perturbed fresh production
policy; p2 is a seeded random policy on an external route, via `rust_rollout.testkit`). A collector hook records
every trainee decision AT PLAY TIME, keyed by (env, episode, dec_n), never by the reused arena slot. The
callbacks' infos record every ended game. After each fill, the harness re-derives every buffer cell
INDEPENDENTLY from that record:

| check | what | result |
|---|---|---|
| A | obs, mask, every label column, action, behaviour log-prob and play-time value equal the play record | 0 mismatches |
| B | `episode_starts == (dec_n == 0)`; `win_target == 1{reward > 0}`; `win_mask == 1` | 0 |
| C | advantage and return equal an independent float64 GAE(γ = 1, λ = 0.8) over the WHOLE game, rows from several updates included; the terminal reward sits on the last row; no bootstrap past it | 0 (bar 1e-5) |
| D | inside a buffer column, neighbouring cells are dec_n then dec_n + 1 of one game, or one game's LAST row then another game's dec_n 0 (the premise of the opponent-intent shift) | 0 |
| E | the learner's own forward on the CURRENT-version rows reproduces the stored log-prob and VALUE | max \|Δ log π\| 4.8e-7, max \|Δ V\| 4.5e-7 |
| F | no row is consumed twice; no key is played twice | 0 / 0 |
| G | (6-turn limit) a game cut at the turn limit is a terminal not-win (reward 0), reported as the stall forfeit | 120 / 120 cut games: reward 0, terminal, forfeit flag set |

Runs: `run_default.json` (ffi front end, 4 envs, 5 fills, 640 cells, 11 games): every check passed, exit 0.
`run_turnlimit6.json` is the same with `--turn-limit 6 --updates 6` (768 cells, 120 games, every one cut at
the limit): every check passed, exit 0, max \|Δ log π\| 7.2e-7. Rerun one with
`PYTHONPATH=src python designs/research_state/measurements/gigo_audit_datapath_2026-10-26/harness.py
[--turn-limit N] [--updates K] [--front ffi|proc]` under `scripts/ops/mem_cap.sh` (≈ 8 min on 3 contended
cores, most of it the self-check build). Exit 0 means every check passed.

## CONFIRMED

### C1. A dropped intent row is re-labelled a BOT row (telemetry only; fixed)

`train()` shifts every one-ahead intent label back one row (`TrainSetup._align_opp_intent_labels`), and
`opp_class` rides the same shift. The rows the shift DROPS are each game's last row and each column's last row.
They took the scalar fill 0, and 0 is `OPP_CLASS_BOT`. So about 1/44 + 1/384 ≈ 2.5 % of every update's rows were
counted as bot rows. The 44 is the live D3 run's `rollout/ep_len_mean`, ≈ 44 decisions per game, and 384 is
`n_steps`. Bot games are about 10 % of games (`train/nonbot_fraction` 0.90), so roughly a fifth of the "bot"
subset was really self-play and pool rows.

- **Contaminated:** `ridealong/rnd*_z_bot` (and so `_z_pool` is read against the wrong reference), and the
  intent fold's per-class "other" rates `opp_intent/other_*_rate_bot` (these are taken over ALL rows of a class).
- **Not affected:**
  - The LOSS. A dropped row's intent label is `KIND_UNKNOWN`, so it is unsupervised and `bot_label_weight`
    multiplies a zero.
  - `opp_intent/flat_*_bot` and `label_bot_frac`, which are taken over supervised rows only.
  - The rollout-time per-class reads (`signal/*_bots`, `win_prob/start_*_bots`), which run before the shift.
- No research claim cites an affected tag (searched `UNDERSTANDING.md` and the endstate docs).
- **Fix:** `align_labels_to_predictions` takes a per-row fill, so a dropped row keeps its OWN class.
- **Guard:** `check_constant_within_episodes` raises `EpisodeKeyNotConstant` when `opp_class` changes inside
  one game in a buffer column. That fail-closed check of the shift's premise runs every update.
- **Tests:** `opp_class_plumbing_test.py::test_a_dropped_row_keeps_its_own_opp_class_not_the_bot_code` fails on
  revert (`[2, 2, 0, 2, 2, 0]`), and `::test_a_class_change_inside_one_game_is_refused_before_the_shift`.

### C2. `--seed` never reached a forked learner's global streams (fixed)

A loaded learner's `_setup_model` seeds Python `random`, NumPy and torch from the checkpoint's SAVED seed. On a
fork, that is the parent's seed. Every X35 checkpoint read for this audit (`rb_x35_c0_s3501`, `_s3502`, `_s3505`,
latest zips, read-only) saves `seed 2601`, the seed of `rb_x26_s2601`. So every fork of one parent, and every
restart segment of one run, shuffled with ONE global stream. `learner_lifecycle.md` measures the PPO minibatch
permutation as that stream's only steady-state consumer.

- **Impact:** the replicates shuffled their buffers in the same permutation ORDER, but the buffers themselves
  are independent (§ Seeds), so no shared randomness reaches WHAT is trained. Not a validity problem for any
  past A/B; a provenance lie in every fork's checkpoint.
- **Fix:** `main.train.learner_seed.seed_loaded_learner`, called on the resume path after the hparams, records
  `--seed` as `model.seed` and seeds the streams from `learner_segment_seed(--seed, num_timesteps)`. That is a
  keyed hash, stream 13, so a restart no longer replays its first segment's permutations.
- **Guard:** `check_declared_seed` sits before both `learn()` calls and raises `UndeclaredLearnerSeed`
  (FATAL_CONFIG).
- **Tests:** `learner_seed_test.py`. Its first test fails on revert: the parent's 2601 is refused.
- **Executed:** a CPU `--debug` smoke, fresh at `--seed 5`, then a fork of its `final_model.zip` at `--seed 9`. Both
  exit 0; the fork logs `[LearnerSeed] --seed 9 ... seeded 1619140111 for the segment at step 10,260`, and its
  `final_model.zip` saves `seed 9`. Before the fix, it would have saved the parent's 5.

### C3. The scripted bots' streams were one constant across every run (fixed)

The training plan's `bot_seed` was the constant 0 (`rust_env_setup`, `OpponentPlan.from_args(..., bot_seed=0)`),
and a bot route's streams are `stream_seed(0 + route, env, k)`. So every training run, every `--seed` replicate
and every restart segment replayed the same bot RNG streams.

- **Impact:** the streams are consumed in different game states, because the battle seeds and teams differ, so
  outcomes are not correlated in any way we can name. Same-index uniforms are still shared, though, so
  "independent seeds" was not literally true.
- **Fix:** each segment's spec carries `bot_route_seed(run_seed)`, stream 14, recorded as `env_core.bot_seed`.
- **Test:** `rust_env_setup_test.py::test_the_training_bots_are_keyed_to_the_segment_not_a_constant` covers 10
  distinct seed sets and pins the wiring.
- **No throwing guard:** the seed comes from one call site, and the test pins it.
- **Residual:** a RESPAWNED core reuses its spec, so after a respawn the bots replay that segment's bot stream.

**Regime note for C2 and C3:** the minibatch order and the bot draws change for runs launched on this code.
Their distribution is unchanged, and a "replicate" (same pin and seed) still reproduces bit for bit.

## Seeds 3501–3505: are they independent?

| stream | keyed by | independent across `--seed`? |
|---|---|---|
| teams (per-env builder RNG), battle seeds, the trainee's keyed draw, a policy opponent's keyed draw | `segment_seed(--seed, num_timesteps)` (hash) | **yes** |
| opponent route and pool-snapshot draws (`EpisodeOpponentSampler`) | `random.Random(run_seed·1000 + i)` / `+ 5000 + i`, run_seed hashed | **yes** (no overlap for n_envs < 1,000) |
| team-mix corpus draws (`--team-mix`) | the builder's seeded `_rng` | **yes** |
| PPO minibatch permutation (global NumPy) | the checkpoint's saved seed (2601 for every X35 arm) | **no**, the same order (C2; fixed) |
| scripted bots | constant 0 | **no**, the same streams (C3; fixed) |
| initial weights | the shared parent checkpoint (every X35 arm is a fork of `rb_x26_s2601` @ 280,000,170) | shared by design |

Arms that share a seed (`rb_x35_c0_s3501` and `rb_x35_d3mut_s3501`) share the run seed, so they play the same
teams, battle seeds and draws at the start. These are common random numbers, by design. **The data each
replicate trains on are independent across seeds; no past A/B read is invalidated.**

## CLEARED (read, and where possible executed)

- **Buffer construction and complete-game GAE.** Harness A–F. A game that straddles an update is split, never
  dropped: its tail stays at the FIFO head and its GAE was computed once, at its end. A CUT game (a quarantine
  or a respawn) is released and counted, never trained on.
- **The 250-turn cap.** It is a terminal not-win, reward 0, with no bootstrap. That is the declared semantics
  (`training/CLAUDE.md`: "the 250-turn cap is a terminal"), and it is what `game_gae` does. A tie is a not-win.
- **The behaviour log-prob.** It is recorded as `logp_all[action]` from the SAME served row the keyed draw used,
  and K9(b) recomputes it under fp32 (harness E: 4.8e-7 on the CPU). On the live D3 run (`rb_x35_d3mut_s3501`,
  `--behaviour-check warn`), `behaviour_violations.jsonl` holds three violations since the fork (updates 28,500, 29,080 and 29,190). Each is
  ONE row over the 1e-4 bar out of a 4,096-row sample, with \|Δ\| 1.0–1.7e-4 and a p99 of 2e-5. That is
  numerics, not a fault; the bar is a whole order of magnitude below anything the ratio clip (0.15) sees.
- **The value.** T2 serves the policy's `_critic_value`, sigmoid(win head), which is the same function the
  learner evaluates (harness E: 4.5e-7). Under the win-prob critic the value loss is the BCE against the game's
  outcome (`win_target`). GAE reads the play-time sigmoid values. The units agree.
- **The minibatch.** One permutation per epoch indexes every field. The device-batch modes are bit-identical
  to the host path (`device_batches_test`). Full micro-batches are enforced. Advantages are normalised per
  2,048-row micro-batch under 32× accumulation: a declared recipe choice
  (`step_size_and_batch.md` "Subset-normalised terms under accumulation"), not exact full-batch equivalence.
- **The loss terms.** The entropy term is `ent_coef · (−mean H)`, which maximises H, so the sign is right. The
  clip uses `old_log_prob` = the behaviour log-prob. approx-KL is `E[(r − 1) − log r]`. The vf term is 0 under
  the win-prob critic, whose BCE carries `vf_coef`.
- **The LR on a resume.**
  - Phase 1 restores the optimizer's LR, clamped to [`--min-lr`, max].
  - Phase 2 recomputes the cosine at `num_timesteps` from the persisted `handoff_lr`.
  - Forks run under `--fork-lr-freeze` (`recipe.fork`).
  - The KL EMA is not persisted, but the controller's cooldown is re-seeded at every start, so a restart cannot
    act on one raw reading. That is deliberate (`adaptive_lr_callback.py`).
- **A resume, row by row.** The FIFO's carried rows and the games in progress are lost at a restart, never
  duplicated. The new segment's run seed differs, so nothing replays. The pool is rebuilt from its directory and
  `summary.json`. `gamma` and `gae_lambda` reach the collector after the resume overrides them, because
  `_start_rust_env` runs after `apply_training_hparams`.
- **The pool slots.** A slot is reloaded only when no env's current or staged episode names it
  (`SlotFamily`), so no game switches opponents mid-play.

## PLAUSIBLE / NOT VERIFIED (FINDINGS)

- **NOT VERIFIED, GPU numerics.** The bf16 R1 against T2's graph backend was not exercised: the audit was CPU
  only. The live run's K9(b) record is the only GPU evidence, and it is clean (above).
- **NOT VERIFIED, X14 `--overlap-collect` and the fork arm (`--fork-fraction`).** Both are OFF in production and
  in every live and queued arm. Not audited.
- **Partly verified, the harness's updates.** The harness steps the weights with a perturbation plus
  `after_update`, not a real `train()`. The real update's own alignment is covered by the K9 learner golden and
  by the C1 tests, not by this harness.
- **PLAUSIBLE, low: forfeit-vs-tie telemetry.** `p1_forfeit = truncated and last_turn >= turn_limit` reads the
  trainee's LAST decision turn. A game cut while only p2 acts could report `win_draw` instead of the forfeit.
  The reward and the label are identical (0) either way. Check G found every cut game flagged as the forfeit
  (120 / 120 at a 6-turn limit); a p2-only final stretch was not constructed.
- **Residual of C3:** a respawned core keeps its segment's bot seed.
