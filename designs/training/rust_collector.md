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
`[D / N, N]` `MaskableDictRolloutBuffer`, so `train()` and every buffer reader are unchanged.

**No row is ever dropped or down-weighted for AGE, and there is no truncation by default** (owner):
PPO's per-row ratio π/μ corrects a stale row. The only rows that leave without being trained on are a
CUT game's (a quarantine, a core respawn): it has no outcome and no next state, so its rows are
released and COUNTED (`staleness/rows_cut_total`), never fabricated into a loss.

**The target** is a declared parameter — default `n_steps × n_envs` (today's rollout size, so the switch
changes WHEN rows are trained on, not how many) — and a multiple of lcm(micro-batch, N): no ragged
micro-batch reaches the one compiled learner graph, and the buffer keeps its shape. A target that is not
also a multiple of micro-batch × K takes a smaller last optimizer step each epoch (the recipe review's
§3.3 "ragged step"; PROPOSED there as Stage 0.2, reported by `trigger.ragged_accumulation`, not imposed).

**The adaptive-batch hook:** `SampleTrigger.set_target(rows)` inside a declared band `[lo, hi]` (the
band's `hi` sizes the arena at startup, so a move allocates nothing); off-quantum or out-of-band is a
typed refusal; accepted moves are counted. The SIZING study (order constraint 5) chooses the numbers.

**The window fill** (`--rollout-trigger window`, `store.fill_window`) reproduces TODAY'S schedule —
column `i` = env `i`'s next `n_steps` rows, sb3's own GAE with the bootstrap V of each env's next row,
and the win labels by `WinProbLabelCallback`'s own back-fill (`backfill_terminal_labels`, one function
both call). It exists to prove the collector against today's path before the schedule changes (the
rollout-level slice N). Rows beyond a column carry to the next window.

## Staleness — measured, not pre-empted

Envs keep playing across an update, so a game in progress when the learner steps has rows of two
versions, and a completed game carried to the next update is one version old when trained on. Every
row records the version that played it; every fill records the AGE histogram (`FillReport`). Before any
optimizer step of every update, `consistency.behaviour_probe` runs the learner's own forward on one
micro-batch and logs, per age bucket (0, 1, 2, 3–4, 5–8, 9+): rows, the ratio π/μ (mean and mean
|r − 1|), the share outside the clip band, and sb3's approx-KL — the `staleness/*` tags.

**The first remedy, if staleness proves harmful, is PER-GAME VERSION PINNING** — a declared option,
OFF by default: each game is played start-to-finish by the version it began with, held in its own T2
slot (K trainee slots); an update loads into a FREE slot (one no game in play still uses); no free slot
is a typed `SlotCapacityExceeded`.

## K9(b) — behaviour-policy consistency

The same probe's first job: on the rows played at the CURRENT version, the learner's recomputed
log π(a|s) must equal the stored μ(a|s), max |Δ| < 1e-4, before any optimizer step — else a typed
`BehaviourMismatch` (`--behaviour-check fatal`, the default under `--env-core rust`; opt-in on the
python core). It catches stale served weights, an eval-vs-train-mode difference and a rollout/learner
observation mismatch; it does not catch a miscompile shared by both sides (K6's eager canary does).

## The keyed draw (`gen3_keyed_draw_v1`)

A stochastic action is a pure function of the decision's key and the served log-probs:
key = (run seed, stream, env, episode, `dec_n`) → splitmix64 chain → a uniform u in [0, 1) →
inverse CDF over `softmax(logp / T)` of the legal actions (`keyed_draw.py` states it bit for bit).
No generator, no state: the Python replay recomputes it from ITS OWN log-probs, so the parity gates stay
EXACT, and the margin `min |c_i − u·c_last| / c_last` names the near-boundary rows two paths whose
log-probs differ in the last bits could disagree on (counted, never a silent pass). The trainee always
uses it (stream 0); a POLICY OPPONENT's stochastic action uses it too by default (stream 1,
`--opponent-sampling keyed`; `generator` keeps today's per-env `torch.Generator` stream, bit for bit
with `RLPlayer`). The reason is REPLAYABILITY — Lane E's opponent gate then replays a sampled game
EXACTLY from its key (`rust_env_opponents_parity.py` mode `keyed`) — not speed: the draws cost
~0.13 / 0.25 / 0.38 ms at 8 / 40 / 48 rows either way (F-LE-8's "5.1 ms of sampling" was the host
waiting for the forward, corrected 2026-09-30), so the keyed draw saves ~0.2 ms a step at most. The
RUN SEED is a hash of `--seed` and `num_timesteps` at the process's startup, so a launcher restart
never replays the first segment's teams, battle seeds or draws; a core RESPAWN re-derives it too.

## Every micro-batch is full — no padding, no drop

The update takes EXACTLY the target, and the target is a multiple of lcm(`--batch-size`, N) (refused
off the quantum at parse — `rollout_target_on_the_quantum` — at the trigger's construction, at every
adaptive move, and by `_ensure_buffer` if the learner's micro-batch ever moved under a built
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
worker's unseeded global `random`: a declared change of STREAM, not distribution (like F-LE-3).

## The env surface

`RustVecEnv.SURFACE` maps every `env_method` a production callback calls (self-play pushes → Lane E's
`RustEnvOpponents`; `drain_team_wr_counts`, `drain_reward_terms`, `opponent_default_stats`,
`exploiter_winrate_totals`, `set_exploiter_temperature`). `REFUSED_WITH_FLAG` names the ones reachable
only under a flag the Rust core refuses at startup; `rust_vec_env_test.py` walks every `env_method`
call in the training sources and fails on one in neither table.

## T2 slot groups and buckets

One slot group per ARCHITECTURE in the route table's slot order (pool, stables, exploiter — two slots
under a ladder, F-LE-6), then the trainee's slot(s); consecutive same-architecture routes share a group
(F-LE-7). Default buckets `(8, N)` (`build.py` states why; decision: the program doc's Decision record).

## The gate — slice N at the ROLLOUT level + the learner-level check (`rust_rollout/parity.py`)

RECORD in Rust (the collector in WINDOW mode — today's schedule, so the collector is proven before the
schedule changes — a fixed policy, p2 an external seeded-random route whose actions are recorded), REPLAY
in Python through today's path itself: `InstrumentedMaskablePPO.collect_rollouts` over a `DummyVecEnv` of
production-surface `Gen3Env`s wrapped as `env_factory` wraps a worker (`Monitor(MaskableAgentWrapper)`),
`WinProbLabelCallback` registered. The one substitution into sb3's loop is the policy forward's SAMPLE:
the replay draws the trainee's action with the keyed draw from its OWN log-probs, so reproducing every
recorded action is itself a check. Compared per window: every observation key, actions, masks, rewards,
episode starts EXACT; values / log-probs within 1e-5 and advantages / returns within 1e-4 (the paths
forward different batch compositions, and CPU matmul rounds by batch shape — measured ≤ 7.2e-7). The
learner check runs ONE optimizer step (the buffer in 4 accumulated micro-batches) from identical
production learners on each buffer and compares the weights (≤ 1e-5) and every logged scalar
(≤ 1e-6 + 1e-4 × |value|); over several steps PPO amplifies the rounding (F-LG-8). COMMIT in the routine
gate, MILESTONE (`slow`) on pool and ladder teams. Numbers: PROGRESS.
