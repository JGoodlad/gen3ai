# Training — the Rust collector (M5 Lane G)

The PPO trainer's rollout on the M5 Rust env core. **Status and resume point:**
[`../research_state/measurements/m5_laneG/PROGRESS.md`](../research_state/measurements/m5_laneG/PROGRESS.md);
the program's lane row, paragraph and Decision record rows:
[`../endstate/program_rust_core.md`](../endstate/program_rust_core.md) §2 M5 (Lane G, order
constraints 5 and 6). This file OWNS the detail and is always-current (the `designs/training/` rule).

Code: `src/agents/training/rust_rollout/` (the package; its `__init__.py` is the module map),
`src/agents/training/keyed_draw.py` (the keyed draw), `src/agents/training/rust_vec_env.py` (the
VecEnv the model holds).

---

## What it is

ONE env core (`src/rust_env/`) steps N battles on its own worker threads. The host loop
(`collector.RustCollector.host_step`) does, per host step:

1. submits p2's POLICY rows to their T2 slots (Lane E's `PolicyOpponentServer.submit`), and the
   trainee's rows (every env whose p1 needs a decision) to the trainee's T2 slot;
2. ONE T2 `flush` serves both (lanes replay the slots concurrently);
3. p2's actions (greedy, or the route's sample); the trainee's action is the KEYED DRAW from the served
   log-probs (below). The row goes into the arena with its behaviour log-prob μ(a|s), V(s) and the
   POLICY VERSION T2 served;
4. `core.step()`;
5. every env whose game ended closes it (reward, the raw flags, the outcome, a tie vs the stall forfeit);
   every env whose `episode` column MOVED is re-staged: its next route (Lane E `after_op`, F-LE-4),
   teams and battle seed (`teams.TeamStager`).

Scripted bots are played INSIDE the core (Lane F); p2 decisions on a bot route are never exposed.

## The complete-game buffer (order constraint 6 — the owner's collector, 2026-09-29)

`n_steps` windows are retired as the schedule. Every row of a game waits in the arena
(`store.RowStore`, preallocated at startup) until the game ENDS; then:

- its GAE is computed over the COMPLETE game (`store.game_gae` — sb3's
  `compute_returns_and_advantage` arithmetic operation for operation, float32, so a game that lies
  wholly inside an sb3 window gets the same BITS: `store_test.py` pins it at three (γ, λ) pairs);
- every row is labelled with its own game's outcome (`win_target` = 1 on a win, 0 otherwise; `win_mask`
  = 1 on every row — the λ = 1 win-prob critic no longer drops the rows of games unfinished at a
  window edge, which is K10(b), SUBSUMED);
- the game joins the completed-game FIFO.

The update fires when the FIFO holds at least the TARGET rows (`trigger.SampleTrigger`) and consumes
EXACTLY the target, FIFO by completion; the one game straddling the D-th row is split and its tail
stays at the FIFO head for the next update. The rows are laid column-major into the model's own
`[D / N, N]` buffer (`rollout_buffer.RolloutBuffer` — owned since deletion pass U4, the layout of sb3-contrib's `MaskableDictRolloutBuffer`), so `train()` and every buffer reader are unchanged.

**No row is ever dropped or down-weighted for AGE, and there is no truncation by default** (owner):
PPO's per-row ratio π/μ corrects a stale row. The only rows that leave without being trained on are a
CUT game's (a quarantine, a core respawn): it has no outcome and no next state, so its rows are
released and COUNTED (`staleness/rows_cut_total`), never fabricated into a loss.

**The target** is a declared parameter — default `n_steps × n_envs` (today's rollout size, so the switch
changes WHEN rows are trained on, not how many) — and a multiple of lcm(micro-batch, N): no ragged
micro-batch reaches the one compiled learner graph, and the buffer keeps its shape. A target that is not
also a multiple of micro-batch × K takes a smaller last optimizer step each epoch (the recipe review's
§3.3 "ragged step"; PROPOSED there as Stage 0.2, reported by `trigger.ragged_accumulation`, not imposed).

**The ORACLE REVEAL** (`--oracle-reveal {off,species,full}`, `designs/endstate/design_x5_belief_tokens.md` §7.6; a DIAGNOSTIC observation mode): `RustEnvDecl.oracle_reveal` is written into the core's spec (`oracle_reveal`, a required `SPEC_KEYS` entry) and read by `build_eval_core` from the same declaration, so the training pool and the in-loop eval core always share the run's mode. The core builds each side's chain with the OTHER side's team (`encoder/oracle.rs`), so the trainee and a T2 policy opponent are symmetric; scripted bots read the view and are unaffected; the labels follow the row (`env_labels.md` §2). The spec key also takes a PER-SEAT pair `[p1, p2]` (`build_eval_core(oracle_reveal=(p1, p2))`), which only `main.h2h`'s per-side oracle reveal passes (`designs/training/eval_and_rating.md`); the trainer never does.

The row arena is sized by `--rollout-target-samples` (or `n_steps x n_envs`) alone. The complete-game trigger is the only trigger (`SampleTrigger(target, quantum)`); `store.fill_complete` is the one fill, and it labels every row with its game's outcome (`win_mask` 1).

## The FORK phase (declared, OFF — `gen3_fork_rust_v1`)

With `--fork-fraction > 0` the collector's `collect` is `COLLECT_PHASES` = play → **fork** → fill: after
the trigger fires, `rust_rollout/fork.py` branches contested decisions of the games that ended since its
last pass (their core input logs come from `core.finished()`), plays the branches to the end on declared
Lane I playout handles with T2 serving every decision, and inserts each branch game into the completed-game
FIFO right after its parent. The arena then also tracks each row's TURN and which live rows are branch
rows (the fork row budget, `ROW_BUDGET_MULTIPLE × target`, added to the declared capacity). OFF builds
none of it. Detail:
[`forks.md`](forks.md) §14.

## Staleness — measured, not pre-empted

Envs keep playing across an update, so a game in progress when the learner steps has rows of two
versions, and a completed game carried to the next update is one version old when trained on. Every
row records the version that played it; every fill records the AGE histogram (`FillReport`). Before any
optimizer step of every update, `consistency.behaviour_probe` runs the learner's own forward on one
micro-batch and logs, per age bucket (0, 1, 2, 3–4, 5–8, 9+): rows, the ratio π/μ (mean and mean
|r − 1|), the share outside the clip band, and sb3's approx-KL — the `staleness/*` tags.

The measurements above are the whole staleness read: no remedy is built (per-game version pinning was deleted), so a row is trained on at whatever age the `staleness/*` tags show, corrected by PPO's own per-row ratio.

## K9(b) — behaviour-policy consistency

The same probe's first job: on the rows played at the CURRENT version, the learner's recomputed
log π(a|s) must equal the stored μ(a|s) — max |Δ| < 1e-4 at fp32 `highest`, the only precision, over the rows
not at a selection tie (one table, `consistency.BEHAVIOUR_GATE`, measured — `learner_gates.md`) — before any optimizer step — else a typed
`BehaviourMismatch` (`--behaviour-check fatal`, the default on both env cores; on the python core, whose
buffer carries no versions, the check reads the first micro-batch's own forward instead of running this
probe — `learner_gates.md`). It catches stale served weights, an eval-vs-train-mode difference and a rollout/learner
observation mismatch; it does not catch a miscompile shared by both sides (K6's eager canary does).

## The keyed draw (`gen3_keyed_draw_v1`)

A stochastic action is a pure function of the decision's key and the served log-probs:
key = (run seed, stream, env, episode, `dec_n`) → splitmix64 chain → a uniform u in [0, 1) →
inverse CDF over `softmax(logp / T)` of the legal actions (`keyed_draw.py` states it bit for bit).
No generator, no state: the Python replay recomputes it from ITS OWN log-probs, so the parity gates stay
EXACT, and the margin `min |c_i − u·c_last| / c_last` names the near-boundary rows two paths whose
log-probs differ in the last bits could disagree on (counted, never a silent pass). The trainee always
uses it (stream 0); a POLICY OPPONENT's stochastic action uses it too, always (stream 1;
`PolicyOpponentServer`'s own `sampling` argument keeps the generator stream only as the keyed-draw benchmark's A/B baseline and its parity test). The reason is REPLAYABILITY — Lane E's opponent gate then replays a sampled game
EXACTLY from its key (`rust_env_opponents_parity.py` mode `keyed`) — not speed: the draws cost
~0.13 / 0.25 / 0.38 ms at 8 / 40 / 48 rows either way (F-LE-8's "5.1 ms of sampling" was the host
waiting for the forward, corrected 2026-09-30), so the keyed draw saves ~0.2 ms a step at most. The
RUN SEED is a hash of `--seed` and `num_timesteps` at the process's startup, so a launcher restart
never replays the first segment's teams, battle seeds or draws; a core RESPAWN re-derives it too.

## Every micro-batch is full — no padding, no drop

The update takes EXACTLY the target, and the target is a multiple of lcm(`--batch-size`, N) (refused
off the quantum at parse — `rollout_target_on_the_quantum` — at the trigger's construction, and by
`_ensure_buffer` if the learner's micro-batch ever moved under a built
collector). The one game straddling the D-th row is SPLIT, its tail trained next update (F-LG-1), so
the row count never varies and no rows are dropped. Every micro-batch the learner sees is therefore
full-shaped: one compiled learner graph, no pad rows, no masked means. What CAN be short is the last
ACCUMULATION group (98,304 = 1.5 × 65,536 at the live shape: 32 micros, then 16): the learner flushes
it as one FULL-weight step, rescaling its summed gradient from 1/K to 1/(its micro count) — i.e.
normalised by the step's real rows (`instrumented_ppo/ppo.py`, the trailing-group flush; pinned equal
to the unaccumulated step over the same rows by `instrumented_ppo_test.test_grad_accum_matches_full_batch`'s
`(4, 3, 12)` case). The dose now counts that step as the full step it is (K10(c),
`agents/training/dose.py`: 2 steps an epoch, not 1.5).

## Where a host step's time goes (the timers, and what is NOT a lever)

`CollectorStats.seconds`: `submit` (gathering + submits), `flush` (the T2 flush CALL — it only
launches the replays), `gpu_wait` (the host waiting on `ticket.host()`: the forward's completion),
`opp_draw` / `draw` (the opponents' / the trainee's draws), `write` (the arena), `core`, `post`,
`fill`. At the production mix (95 % self-play, a 20-snapshot pool, N = 48; the owner's registered A/B,
2026-09-30) a step is 11.7 ms. The T2 forward is 8.7 ms of it (launch 1.6 + wait 7.1), and that is the
opponents' FAN-OUT: ~19.5 distinct slot replays a step, each latency-bound. The core is 1.9 ms, the
draws 0.25 ms, the glue 0.9 ms. Neither the keyed draw nor an env-step / inference overlap is a speed
lever at this N:

- overlap measured 0.71× of serial — splitting the envs doubles the replays;
- the keyed draw is replayability, not speed.

A grouped forward across slots is capped at 4.7 of 8.4 ms a flush. Both are re-measured by the SIZING
study at larger N; the rules are in the program doc's Decision record.

## The declared lifecycle

Startup (`RustVecEnv.startup(model)`, called by `model_build` right after the model is built or loaded
and BEFORE `--compile-trainer` — T2 deep-copies the policy as its slot templates, and a copy taken after
the compile would carry the patched `forward` bound to the learner's extractor) acquires the core
(`Core::new` validates every team of the table by use), T2 (compiles, captures, parity), the arena and
the opponents; then nothing. After every update `check_lifecycle` reads every `*_after_freeze` counter
(core + T2) and raises on a non-zero one. A process-front-end RESPAWN is the one counted steady-state
event, allowed up to a declared budget (F-LB-1): its games are cut and the core is RESET (F-LB-2).
The recovery (`RustCollector.recover_respawn`) checks the budget, cuts every live game (rows released,
counted), derives a NEW segment seed (so the reset never replays a draw), forgets every env's staged
opponent episode, re-stages opponents and teams, RESETs and re-stages the next episodes — pinned by
`collector_integration_test`'s SIGKILL-the-child test (training continues, the lifecycle stays clean).

## Teams, seeds, the opponent route

`teams.TeamStager` holds, per env, a SEEDED copy of the run's trainee and opponent teambuilders (and of
each pinned stable / exploiter route's builder, F-LE-5); a draw is the builder's own `yield_team`, mapped
into the startup team table. The per-team win-rate record follows the team the episode PLAYED (the
builder's own `_last_pool_idx` already names the next, staged one). Today's builders draw from each
worker's unseeded global `random`: a declared change of STREAM, not distribution (like F-LE-3). On the python
core that global stream was also RE-SEEDED by every pool load until 2026-10-02, which skewed its team curriculum
(ledger 2026-10-02, `gen3_no_global_reseed_v1`); the seeded per-env copies here never had it.

## The env surface

`RustVecEnv.SURFACE` maps every `env_method` a production callback calls (self-play pushes → Lane E's
`RustEnvOpponents`; `drain_team_wr_counts`, `drain_reward_terms`, `opponent_default_stats`) and nothing else — a method outside
`SURFACE` is a typed `RustEnvSurfaceError`; `rust_vec_env_test.py` walks every `env_method` call in
the training sources and fails on one that is not in it. (The refused-method table
`REFUSED_WITH_FLAG` is deleted with the flags it named, deletion pass L4.)

## T2 slot groups and buckets

One slot group per ARCHITECTURE in the route table's slot order (pool, stables, exploiter — ONE slot;
the two-slot ladder plan of F-LE-6 was deleted with the exploiter ladder, deletion pass L4), then the trainee's slot(s); consecutive same-architecture routes share a group
(F-LE-7). "Same architecture" is the state-dict signature plus the forward fingerprint over the CANONICAL extractor
kwargs — a kwarg recorded at the value its absence means at load (the extractor constructor's signature default) does
not split a group (`slots.canonical_extractor_kwargs`, X5 look 3 FINDING 1; `eval_and_rating.md`). Default buckets `(8, N)`, plus 64 when N > 64 (`build.py` states why; decision: the program doc's
Decision record). PER-SLOT CAPS (`gen3_slot_bucket_caps_v1`, `--t2-opponent-bucket-cap`, default 64): only the
trainee's slot(s) capture the N-row bucket; every other slot (opponents, eval's) captures the buckets <= 64, rows
beyond chunked — each lane's CUDA-graph pool is sized by its largest capture (8 lanes x 232 MiB uncapped at
N = 256; `learner_lifecycle.md` "The update fit check").
EVAL's slots (M5 Lane H) are appended after the trainee's (`build_collector(extra_slots=)`: the trainee's
eval slot, the sentinel slots, fixed opponents the plan does not already serve) — in the trainee's group
when the architecture matches, so they reuse its compiled buckets; `col.extra_slots` names them and
`col.evaluator` is the eval core built over the same service (`designs/training/eval_and_rating.md`).

**Every slot serves the SAME full forward — an opponent slot computes a V nothing reads, and that is measured to be
free.** A policy-only forward for the non-trainee slots (`ServiceSpec.value_slots`, a module without the critic tower,
the value arena NaN for those slots) was built, proven bit-identical on the policy outputs (CPU, CUDA eager, captured
graph) and MEASURED (deletion pass P2, 2026-10-02, 31 slots x 8 lanes, buckets 8/64/256): graph replay −0.3 % /
−0.6 % at bucket 8 / 64, `max_memory_allocated` identical, a COLD T2 startup +103 s (a second Inductor entry per
bucket) — NOT adopted. There is no critic tower at all since the version break's part 2 (audit F1 deleted it; before
that, Inductor already dropped its MLP under `winprob`), and the value branch is a thin readout
on the trunk pi shares (cutting the extractor's value routes + win head too saves only 3.1–3.8 %). A new consumer of
an opponent slot's V needs nothing declared today; if the mode is ever revived (`git cherry-pick e9d17f06`), the
consumer list is `rust_rollout.build.value_slot_ids`. Evidence: `designs/ops/deletion_pass_manifest.md` §6 finding 11,
`designs/research_state/measurements/p2_policy_only_slots/`.

## The gate — what holds the collector now

The rollout-level slice N and the learner-level check (`rust_rollout/parity.py`: RECORD in Rust, REPLAY through the Python `collect_rollouts` over production-surface `Gen3Env`s) were retired with the Python env core in the deletion pass (U3); no Python path remains to replay against. The collector is held by its own unit and integration tests (`rust_rollout/store_test.py`, `trigger_test.py`, `collector_integration_test.py`, `fork_test.py`), by K9(b)'s behaviour-consistency gate at every update (`consistency.py`, above), and by the Rust-side differential gates (`designs/rust_sim/`). The measured numbers of the retired gate (exact observation / action / reward equality, values and log-probs within 1e-5, one optimizer step within 1e-5 of the Python learner; F-LG-8's amplification over several steps) are history in `designs/research_state/` PROGRESS files.

## From the training leaf (moved 2026-10-10)

> These sections headed `src/agents/training/CLAUDE.md` until its 2026-10-10 cleanup; moved here as they
> stood (minus statements verified FALSE). Where an earlier section of this doc says the same in more
> detail, both are current; fix both in the same pass.

### Rollout collection — the Rust complete-game collector is the ONLY collector

There is no Python rollout collector, no per-step `SubprocVecEnv` barrier and no async-wave collector (all deleted in the deletion pass, U3; the async-rollout flag with them). The collector is `rust_rollout/` (see "The env core" below and `designs/training/rust_collector.md`).

🚨 **A STEP-COUNTED CADENCE IS TOTAL ENV STEPS, never vec calls or rollouts** (F-SZ-3, 2026-10-01).
The periodic checkpoint was SB3's `n_calls % save_freq` at 50,000 calls — 2.4M env steps at N = 48,
~102M at N = 2048; it now saves when `num_timesteps` crosses each
multiple of 2.4M (`main.train.constants.checkpoint_due`), like eval, the pool add/refresh it drives,
and the plasticity canary. A callback CALL is not a fixed number of env steps (N decisions on the Rust collector), so a new cadence compares
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
[`designs/training/ppo_step.md`](ppo_step.md).

**`--diagnostics-every N` (fresh default 10; `gen3_diagnostics_cadence_v1`, config v124) runs the
OPTIONAL probes — per-term noise scale, `grad/*`, `rank/*`, `edge/*`, `cell/*` — on every Nth update
only**, and a skipped update writes NONE of their tags (a gap, never a stale value). The first update
of every process always runs them (the compile lock follows it), `--rank-tripwire` keeps `rank/*`
every update and `--adaptive-batch policy` keeps the per-term probe every update. Learning is
BIT-IDENTICAL at any N (`diagnostics_cadence_test.py`). Recorded + inherited; a pre-v124 run inherits
1. ⚠️ A reader that windows by reading COUNT (the vf_coef "last 20") now spans N× the updates.
Detail: [`designs/training/ppo_step.md`](ppo_step.md).

### Where the trainee's observation comes from — the Rust core, the ONLY source

`gen3_core_obs_source_v1`: the trainee's observation row (2845-dim) and 11-bit mask are produced by the Rust core (`__OBS__` frames
from the `sim_bridge` / env core) and read by the collector; the obs-source flag (a Python-vs-core choice) was deleted in the deletion pass (U3) with the Python env core, so there is no second source to
diff against. Labels, reward and the action mapping are served by the core too (`rust_rollout/`). **The trainee's `(observation_space, action_space)` has ONE builder,
`agents.training.trainee_spaces`** (`trainee_spaces()`, `trainee_env_kwargs()` — the per-run label switches as a pure function of the args; no env is built to read them). The Rust env core records a decision only at the env's decision rows (no phantom decision on a `wait` request, no phantom opponent poll; the Python-road fixes `gen3_no_phantom_decision_v1` / `gen3_no_phantom_opponent_poll_v1` were retired with that road, T27 P6).
Detail: `designs/rust_sim/encoder.md`, `designs/endstate/program_rust_core.md` §3.

### The env core — the Rust core (the ONLY core; the M5 switch, M5 Lane G)

🚨 **`rust` is the ONLY env core** (`gen3_env_core_switch_v1`; the Python core was deleted in the deletion pass, U3, and the one-valued `--env-core` flag with it, P11b: a typed one is refused at parse time with the reason, and no `env_core` attribute is left on the namespace). Every run SIZE is declared in ONE block, `designs/production_config.json` `recipe.sizing`.
Every launch runs on rust (a fresh `--arch production` or bare argv — the bare argv is the win-prob critic (the only critic) + its three reward values, which the Rust core serves). What stays keyed on the RECORD: a `--model` checkpoint produced on rust runs on it silently, and a PYTHON-ERA checkpoint (produced on python, or before the core was stamped) that trained the WINPROB critic moves onto rust, announced as a CORE SWITCH (`rust_env_setup.env_core_switch_line`, read off `metadata.json`'s `env_core` stamp). A checkpoint that trained the SHAPED critic is REFUSED on a resume or fork whatever the core (`FATAL_CONFIG`, D4 `PythonEraShapedCheckpoint`: run it pinned to its own commit); it still LOADS as an opponent, in the meters and in the prober.
One refusal: `main.train.rust_env_setup.refuse_python_era_checkpoint`, called by `resolve_config` and
`checkargs` (read off the checkpoint's recorded critic); pinned by `main/train/env_core_switch_test.py`. Runbook: `designs/ops/training_runbook.md`.

The trainer runs the rollout on the M5 Rust env core: N envs in ONE core (process front end by
default, `--rust-env-front`), the trainee and every policy opponent forwarded through the inference
service in ONE flush, the scripted bots played inside the core, and the COMPLETE-GAME collector
(the only trigger): a game's rows are buffered until it ends, GAE
and the win label run on the complete game (every row `win_mask` 1), and an update fires at
`--rollout-target-samples` completed-game rows (default `n_steps × n_envs`), consuming exactly that many
— no row is dropped or down-weighted for age. Hazards an agent must know before touching it:

- 🚨 **A fresh model is BUILT AT ONE TORCH THREAD** (`model_build.construct_fresh_learner`, `utils.torch_state_guard.single_thread_build`, `gen3_single_thread_init_v1`): SB3's orthogonal re-init is a LAPACK QR whose rounding follows the thread count, so an unpinned build gave a different start for the same `--seed` at a different core count / `OMP_NUM_THREADS`. The caller's count is restored on return; a resume / fork loads strictly and needs none. It is RECORDED: `construct_fresh_learner` sets `model.init_num_threads` to the count `single_thread_build` reported from INSIDE the block (the helper yields it; never the process default), the int rides every checkpoint, `run_io._model_hparams` hands it to `save_model_snapshot`, and `metadata.json`'s `init_num_threads` is IMMUTABLE once written (existing value wins, like `original_command`): a resume keeps the original, a fork (new run dir) records the build its loaded weights came from, and a checkpoint that predates the record leaves the key ABSENT (unknown, never a guessed 1). Pinned by `src/main/train/init_num_threads_test.py`. Any new site that creates and initialises fresh parameters uses the same helper (`fresh_build_threads_test.py`; `designs/training/learner_gates.md`).
- 🚨 **`--oracle-reveal` is the RUN's observation mode, declared once** (`RustEnvDecl.oracle_reveal`, resume-immutable; a DIAGNOSTIC, never production — `designs/endstate/design_x5_belief_tokens.md` §7.6): the collector's spec and `build_eval_core` (which reads its `collector_decl`) take the same value, so the in-loop eval plays at the run's recorded mode; the Rust core gives each side's chain the other side's team (symmetric; scripted bots unaffected). It is refused with the fork arm (`oracle_reveal_vs_fork_arm`: search chains build `off` rows). `main.h2h` plays an oracle checkpoint only through its PER-SIDE reveal (`--oracle-reveal-mode one_sided|both_sided`, `main.h2h.reveal`: `build_eval_core(oracle_reveal=(p1, p2))`, the spec's `[p1, p2]` form) and refuses one under its default `off`.
- 🚨 **Startup runs BEFORE `--compile-trainer`** (`model_build._start_rust_env`): the inference service
  deep-copies the policy as its slot templates, and a copy taken after the compile would carry the
  patched `forward` bound to the LEARNER's extractor.
- **The collector fills `win_target` / `win_mask`** (`store.fill_complete`; the `WinProbLabelCallback` is deleted).
- **K9(b) `--behaviour-check`** (default `fatal`): before any optimizer step of every
  update, the learner's log π on rows played at the CURRENT version must equal the stored behaviour
  log-prob (ONE gate, at fp32 matmul precision `highest` — the only precision, TF32 was retired: DETERMINISTIC — a row whose forward has a declared selection / threshold within a relative margin 2e-4 of its cutoff is EXCLUDED (3.7 % of healthy rows; `agents/model/selection_sites.py`, `rust_rollout/tie_margins.py`) — unless the tied candidates gather bit-identical PAYLOADS, and no row while the action scorers are still zero-init (`gen3_behaviour_tie_identity_v1`: such a tie cannot move log π), and an X5 sort pair counts only where its caller reads the ORDER (`gen3_behaviour_tie_consumed_v1`: the op's per-mon move orders are a SET before each cut), and a row whose ONLY near tie is one element of one call (an isolated sort pair, a threshold) is JUDGED under BOTH resolutions instead — passes if either is within the bar, FATAL if neither (`gen3_behaviour_tie_flip_judge_v1`, one extra probe forward only when needed) — every other row's |Δ| < 1e-4 or FATAL at once, the excluded share < 0.15; a process at any other precision is refused; under `warn` a violation's diagnostic scan is a 4,096-row sample). Lane G's pre-loop probe runs its own forward and logs
  `staleness/*` (ratio, clip fraction, KL by row AGE) and `behaviour/*` (`learner_gates.md`). No staleness remedy is built (per-game version pinning was deleted); those tags are the whole read.
- **Every micro-batch is FULL — no padding, no drop.** `--rollout-target-samples` must be a multiple of
  lcm(`--batch-size`, `--n-envs`) (refused at parse, at the trigger, and by the
  collector before a fill); the one game straddling the target is split and its tail trained next
  update. Only the last ACCUMULATION group can be short, and the learner flushes it as a FULL-weight step
  normalised by its real rows — which the dose now counts as such (K10(c)).
- **Stochastic actions are the KEYED DRAW** (`gen3_keyed_draw_v1`; the trainee always, policy opponents
  always): replayable from the decision's key, which is what keeps the
  parity gates exact. It is NOT a speed lever (~0.2 ms a step: F-LE-8's "5.1 ms of sampling" was the host
  waiting for the forward). A core RESPAWN (up to `--rust-env-respawn-budget`) cuts the live games,
  derives a new segment seed and re-stages; past the budget it is fatal.
- `rollout/collect_ms` + `rollout/collect_decisions` are logged (the A/B reads them);
  `rust_env/*` is the collector's per-phase read. `metadata.json` records `env_core` on every save.
- ⚠️ A launcher RESUME pins to the checkpoint's commit. A commit before Lane G has no env core flag
  and is refused by name (`NOT IN PINNED TREE`, for the collector flags the argv types). The trainer builds its checkout's env core at startup
  (`utils.rust_env.build`), because a pin worktree has no `target/`. A `--model` launch is judged off
  the checkpoint's RECORDED core (`metadata.json` / the sidecar's `env_core` stamp). Detail:
  `src/main/launcher/CLAUDE.md` → "A Rust-core run under the launcher".
- **EVAL runs on the core too** (M5 Lane H, `rust_eval/`): both eval callbacks write the same plan and
  manifest, then play the cycle IN PROCESS and BLOCKING on a declared eval core (`--rust-eval-envs`) and
  declared eval T2 slots, publish the workers' own shard records, and collect them with the unchanged
  code. Games are seeded by the GAME (`gen3_eval_game_seed_v1`); traces are CORE traces (records +
  reconstruction + states; the prober expands them). A missing eval core is FATAL, never a
  fall-back to Python workers. Detail: `designs/training/eval_and_rating.md` → "Eval on the Rust env core".

Detail: [`designs/training/rust_collector.md`](rust_collector.md).
