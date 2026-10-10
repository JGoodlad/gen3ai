# Training — self play and pool

Moved verbatim out of `src/agents/training/CLAUDE.md` on **2026-09-07**, when that leaf was
split by topic (it was 8,219 lines / 676 KB, loaded in full for any session touching the
training package). Each section below is unchanged, including its dated measurements.

`src/agents/training/CLAUDE.md` keeps the heading and the opening paragraph of each, and
points here. **This file is the owner of the detail.**

---

## Self-play opponents (`--self-play`, gated behind pathology hunting)

When `--self-play` is set, `SelfPlayCallback` replaces `PerOpponentEvalCallback` and the
training opponents become frozen snapshots of the agent itself, drawn from a directory-backed
`SnapshotPool` (`snapshot_pool.py`; state reconstructed from `<run_dir>/snapshots/` on every
restart — no manifest). Design lives in `designs/ai_v5/`. Key behaviors:

- **Eval + promotion are BLOCKING and IN PROCESS (the Rust eval core), mirroring
  `PerOpponentEvalCallback`.** On a trigger step `SelfPlayCallback` freezes the live weights to disk
  (`model.save`), writes the plan — the bot roster AND up to `--n-sentinels` pool sentinels (default 5;
  all split into shard units) — and `rust_eval.launch` plays it between two host steps of the collector
  (the SENTINEL units play the frozen trainee greedy vs each sentinel, sampled at `--self-play-temp`
  unless `eval_sentinel_greedy`); the cycle is collected in the SAME `_on_step`. It costs training its wall
  time (~1.5% of wall at N=256, `designs/research_state/measurements/m5_sizing/PROGRESS.md` O9). The
  collect merges per-opponent + per-sentinel results → `win_rate_vs_bots` /
  `win_rate_vs_pool` / `sentinel_monotonicity`, records to TensorBoard + the TUI + metadata.json
  (with the `pool` block), persists `win_rate_vs_bots` (feeds `heuristic_fraction` next run),
  saves best by **copying** the frozen snapshot, and — if `win_rate_vs_pool > --promote-threshold`
  — **promotes the FROZEN snapshot into the pool by file-copy** (`SnapshotPool.add_from_path`):
  the file the cycle actually played, never a re-save of `self.model`. Sentinels load via
  `load_opponent_snapshot` (inference-only, the ride-along keys ignored —
  `gen3_opponent_inference_load_v1`) against the pool's shared `model_config.json` (a WRITE-ONCE record,
  `gen3_pool_arch_record_v1`: below)
  using `current_model_version(mappings)` — a stale-arch snapshot fails with `ModelVersionError`,
  never loads silently. A failed cycle is logged-and-continued (and counts as a failed supply cycle),
  resume-republish behaves exactly as the bot path above, and there is no drain, no skip-while-running
  rule and no hung-cycle watchdog (all were the Python worker pool's, deleted in P10-F2). The
  launch→collect mechanics are the **shared** `eval_launch.launch_rust_eval_cycle` /
  `eval_collect.merge_eval_results` / `persist_eval_snapshot` / `prune_eval_*` / `eval_record.replay_last_eval_to_tui`
  helpers, so the two paths can't drift. `--debug --self-play --debug-eval` uses a
  fast eval cadence (every 4k steps, 3 games) so a short CPU smoke exercises seed → pool eval →
  promotion (a plain `--debug` smoke skips all eval by default — see `--debug-eval`).
- **The promotion's snapshot-ladder update is a DETACHED subprocess on the Rust bridge**
  (`SelfPlayCallback._spawn_snapshot_ladder_update` → `python -m agents.training.snapshot_ladder … --impl rust`,
  CPU only). 🚨 From P11 B4 (`9a92a491`) until P10-F2 it passed no `--impl`, so it fell back to the tool's
  own default `node` and every live promotion's round-robin ran on the NODE bridge; it is `rust` by name now
  (`selfplay_callback_test.test_the_detached_ladder_updater_plays_on_the_rust_bridge`).
- **Curriculum: thresholded ramp + LIVE per-episode fraction.** `heuristic_fraction`
  (`snapshot_pool.py`) is **0% self-play below `SELF_PLAY_START` (0.55)** — a weak model trains
  100% vs bots, no cycles wasted on a useless self-opponent — then smoothsteps `0.55→0.80` up to
  **90% self-play** (`HEURISTIC_FLOOR`=0.10 keeps a few % vs real bots for anti-forgetting). The
  three anchors are **configurable** — `--heuristic-floor` / `--self-play-start-wr` /
  `--self-play-full-wr` (defaults = the constants) thread through both the startup fraction and the
  live push, so a run can keep the coverage-punishing bots in the mix longer (raise `full` to ramp
  slower, raise `floor` for a bigger permanent bot slice). `--bot-weights name=w,…` additionally
  biases WHICH heuristic each episode draws (e.g. `aggressive_v2=3,heuristic2=3` → ~3× emphasis on
  the loss-analysis-flagged coverage bots; unlisted bots stay 1.0, omitted → uniform) — the weighted
  pick lives in `rust_env_opponents.EpisodeOpponentSampler` (the rule-for-rule successor of the deleted Python `MaskableAgentWrapper._select_episode_opponent`), an O(1) in-memory `rng.choices`
  with zero per-step cost. All three default to the original behavior, so an unset run is unchanged.
  Crucially the heuristic-vs-pool split is **no longer fixed per process**: every training env
  picks its opponent **per episode** (the sampler, at each episode start) from a live
  `self_play_fraction`, and `SelfPlayCallback` pushes the fresh fraction (+ a `pool_generation`)
  to all envs via `training_env.env_method("set_self_play_target", …)` **after every eval**, so
  the ratio tracks measured strength mid-run with no restart. The opponent is a pure decision
  function over `env.battle2` (env.agent1/agent2 do the networking), so swapping it between
  episodes is free and safe — built `start_listening=False` (no idle connections), and the
  in-episode stale-decision path is untouched. The pool-vs-heuristic **coin flip is per-episode**
  (so the live fraction is honored exactly), but the pool **snapshot is (re)sampled+loaded only
  once per `pool_generation`**, NOT per episode: `load_model` deserializes a ~27MB MaskablePPO,
  and doing it every episode against an N-deep pool (LRU `lru_cache_size`=3) thrashed the workers
  — they blocked in `reset()` on the deserialize, dropping CPU to ~40% and FPS from ~1400 to ~500
  (regression fixed in `_select_episode_opponent`). A `pool_generation` bump (after a seed/promote)
  makes the worker re-scan + re-sample, so promotions become training opponents within a
  generation; diversity comes from 48 envs sampling independently + rotating each generation, not
  from per-episode churn. (`_n_pool_envs` / the `_maybe_engage_self_play` env-rebuild are gone.)
- **Opponent-mix reporting (`train/selfplay_fraction` / `train/stable_fraction` /
  `train/nonbot_fraction`).** The curriculum coin `sf` (`1 − heuristic_fraction(win_rate)`) pushed to
  the envs and persisted to `summary.json` is the **challenge-ENTRY** probability (= pool +
  un-mastered stable, *when* the challenge pick returns non-None) — NOT the pool share. So the
  reported metrics are derived separately by `SelfPlayCallback._opponent_mix_fractions(sf, pool_ready)`,
  a pure mirror of the sampler's selection (it does **not** change selection).
  The four mutually-exclusive opponent types (bot / pool / un-mastered-stable / mastered-stable) sum
  to 1; the metrics report **`train/selfplay_fraction` = P(pool)** (REPOINTED — it used to log `sf`),
  **`train/stable_fraction` = P(any stable)** (un-mastered in the challenge **+** mastered in the
  weighted floor — a mastered stable "becomes a bot" so it's NOT in `sf`), and **`train/nonbot_fraction`
  = pool + stable** (= 1 − bot; bot is left implicit). `nonbot` is independent of the stable challenge
  share (it cancels); the per-bucket split needs three **reporting-only** inputs threaded into the
  callback from `train_rl_agent` (the capped `stable_challenge_share`, the `--bot-weights` vector, and
  `len(OPPONENT_NAMES)` — the floor roster, `matchup_setup.TRAIN_BOT_NAMES`, which excludes eval-only `random`).
  With no stable opponents these reduce to `selfplay_fraction = nonbot = sf·P`, `stable = 0`.
  `_opponent_mix_fractions` is a hand-written **mirror** of the wrapper's selection, so the anti-drift
  guard was `wrappers_test.py::test_mix_fractions_match_actual_sampling` (deleted with the wrapper, U3; it ran the REAL
  selection thousands of times against the analytic fractions — the sampler's own tests in `rust_env_opponents_test.py` now hold the draw, and the per-case `selfplay_callback_test.py::test_opponent_mix_*` pin the math
  itself). A future selection change that isn't mirrored fails that cross-check.
- **Seeding is GATED on competence; the pool is a SLIDING WINDOW (nothing pinned) by default.** The
  pool is seeded only once win rate clears `SELF_PLAY_START` (at startup via `_maybe_seed_pool`, or the
  moment it crosses mid-run in `_collect_pending`), so the first self-play opponent is a
  *competent* model — never the random/weak step-0 seed of old. By default nothing is pinned: the
  oldest snapshot (incl. the seed) ages out as the window slides past `max_snapshots`, so the floor
  stays a recent self; anti-forgetting is the heuristic floor, not a pinned seed.
- 🚨 **THE WINDOW HOLDS ON EVERY PATH THAT POPULATES THE POOL** (`gen3_pool_cap_every_path_v1`,
  2026-09-30). `max_snapshots` used to be applied only when a snapshot was ADDED, so a pool built by a
  directory SCAN — every env worker's, every resume's, the Rust env core's (`rust_env_setup`), any
  harness's — held whatever the directory held. Now `_scan`, `seed`, `add` and `add_from_path` all end
  in `_enforce_cap`: the add path's own eviction order (oldest first, or spread retention under
  `--pool-spread`), then a hard `PoolOverCapError` if the pool is still over (all pinned); `sample()`
  re-checks. Only the pool that WRITES the directory — the trainer's, built with `owns_dir=True` in
  `train_rl_agent` before any worker exists — deletes what its scan evicts; every reader trims IN
  MEMORY, so a harness reading a real run's `snapshots/` never touches it. A trimmed scan prints a
  `⚠️  [SELFPLAY] Pool dir … over its declared max_snapshots` line naming the dropped steps.
  **Audit (2026-09-30):** production never had a `max_snapshots` flag (20 everywhere) and has one
  writer whose add path trims, so the only way a production pool went over is a directory written
  behind the trainer's back. `eval/pool_snapshot_count` (logged every eval, BEFORE that eval's
  promotion, since the pool existed) exceeds 20 in exactly ONE of 734 tagged event files under
  `models/`: `ai_v6_13_outgoing_dmg_0620_exp_v1`, a pre-auto-seed fork whose parent's 20-snapshot pool
  was hand-copied into its 2-snapshot pool at 2026-06-24 16:47 — 22 selves (2 over, spread retention)
  from the 16:53 restart to the 122M promotion at 19:02, i.e. ~118M→122M steps and the 120M eval's
  sentinels. Ledger 2026-09-30. M5 Lane G's first fewer-snapshots benchmark was void for the same
  reason (a harness declaring K < the directory).

### 🚨 A FORK starts POOLLESS — auto-seed, and REFUSE the silent bot fallback (`pool_seed.py`)

**`SnapshotPool` derives its whole state from a directory, so a FORK begins in a new run dir whose
`snapshots/` is EMPTY — and an empty pool does NOT disable `--self-play`. It falls back to the BOT
pool.** A fold launched to replicate a parent that trained at 90% self-play therefore trains at ~0%
self-play, silently, with the argv, the startup banner and every metric still saying self-play. It
has now cost or nearly cost two cells:

* **2026-08-18** — three 3M-step `ai_v9_17_tdaux_*` forks off a 25M base ended with 1, 2 and 1
  snapshots against the base's 12. Essentially all 9M fork-steps were bot games, voiding a three-arm
  A/B whose gates were defined on the self-play regime.
* **2026-09-02** — the three-dose cell. Caught at launch by reading the startup line, before a
  GPU-hour was spent, and fixed by hand.

**THE MANUAL FIX WAS HALF A FIX THE FIRST TIME, and that half is the point.** Copying the parent's
`snapshot_*.zip` files alone still printed `self_play_fraction=0%`: the STARTING fraction comes from
`SnapshotPool.load_persisted_win_rate()`, which reads the pool's **METADATA**, not its zips. A pool
with 14 snapshots and no metadata reads as a competent-model pool the ramp has not opened yet —
exactly as wrong as an empty one, and it looks healthier.

**THE FILE SET the pool's loader reads out of its own directory** (audited 2026-09-02; `pool_seed.py`
copies exactly this, and `pool_seed_test.py` pins the two names against `SnapshotPool`'s own class
attributes so a rename there breaks the test rather than un-copying a file):

| path | read by |
|---|---|
| `snapshot_*.zip` | `_scan()` — they ARE the pool's entries |
| `summary.json` | `_SUMMARY_FILE` — `load_summary` / `load_persisted_win_rate`; carries `win_rate_vs_bots` (the ramp input) plus `self_play_fraction` / `last_eval_step` / `seeded` / `pool_generation` |
| `win_rate_vs_bots.txt` | `_WIN_RATE_FILE` — the legacy single-float fallback |
| `model_config.json` | the pool's ARCH RECORD, WRITE-ONCE (`SnapshotPool._record_arch`, `gen3_pool_arch_record_v1`). `load_opponent_snapshot` looks beside the `.zip` and then one dir up, so without it every pool opponent arch-checks against the RUN ROOT's config instead of the pool's own. Written when absent; the live version is checked against it BEFORE every add (`ModelVersionError`, nothing written) and it is NEVER rewritten — it used to be rewritten with the live version on every add, so after the first promotion every load compared the run with itself. A fork's seeded pool carries the PARENT's record |

Nothing else in the directory is read — there is no manifest, which is the whole reason the class is
directory-derived.

**THE SEEDING RULE** (`agents.training.pool_seed.prepare_pool`, called once from `train_rl_agent`
**before** the run's `SnapshotPool` is constructed — the starting fraction is read at construction,
so seeding afterwards would land the files and still announce 0%):

> `--self-play` ON **and** a genuine FORK **and** this run's pool is EMPTY ⇒ copy the fork parent's
> pool — every `snapshot_*.zip` plus every metadata file above — and print one line:
> `🌱 [SELFPLAY] [pool] seeded N snapshots + metadata from <parent run> (win_rate_vs_bots=…) [files: …]`

FORK-vs-RESTART is `main.train.fork_lr.is_same_run_checkpoint`, **IMPORTED, never re-derived** (a
second predicate for the same question is a second answer waiting to disagree), so a launcher restart
never re-seeds — re-seeding there would overwrite the run's own grown pool with the parent's stale one
every few hours. A **non-empty pool is never touched**, so the hand-seeded arms of a running cell keep
exactly the pool they were given when a later arm syncs this code. A **FRESH** run (no `--model`) is
unchanged: it legitimately starts poolless and grows one, and the win-rate gate is what stops it
seeding a random-weights opponent. The parent run dir comes from `lineage.fork_parent(run_dir)` when
the run already records one (immutable, so it names the ORIGINAL parent even after the launcher swaps
`--model`), else the `--model` path's own run dir — which is the only answer available on a fork's
first process, before any save has written a lineage block.

**THE REFUSAL.** If `--self-play` is on, the run is a FORK, and the pool is STILL empty after that
step — the parent has no pool, or `--no-fork-pool-seed` was passed — the launch exits
**`FATAL_CONFIG`** naming the three ways out (seed by hand, `--allow-empty-pool`, drop `--self-play`)
rather than quietly training against bots. `FATAL_CONFIG` and not `parser.error` because a restart
would hit the identical config, so the launcher must give up instead of looping.

**THE IN-FLIGHT FLOOR (`gen3_supply_guard_v2`).** The refusal above covers a FORK at startup. ANY
`--self-play` run — fresh ones included — whose pool is still EMPTY after **3 consecutive eval
cycles** (failed cycles count: a cycle with no results cannot seed) exits **`FATAL_SUPPLY` (5)**,
naming the seeding gate (`win_rate_vs_bots` vs `--self-play-start-wr`) and the remedies;
`--pfsp-scale` with sentinels launched but no win-rate measured for 3 cycles is the same FATAL.
`ai_v12_27_ladder_ctrl10M_shaped_dense` is the case: bots win rate 0.148 → 0.531 over five cycles,
never the 0.55 gate, so its whole 10M was bot games under a `--self-play` argv. The streak is
persisted in this pool's `summary.json` (`supply_guard`, keyed to the run dir) so it survives a
launcher restart; `--supply-starve-cycles self_play_pool=0` turns it off (announced). Detail:
[`supply_guards.md`](supply_guards.md).

| flag | default | |
|---|---|---|
| `--fork-pool-seed` / **`--no-fork-pool-seed`** | ON | opt out of the auto-seed. Declared POSITIVELY so `BoolFlag` generates the `--no-` form — declaring `--no-fork-pool-seed` would have generated `--no-no-fork-pool-seed` |
| **`--allow-empty-pool`** | OFF | explicit consent to the bot fallback on a fork. Never needed by a fresh run |

Both are **training-runtime** flags: they reach no extractor, scale no loss and change no weight
shape, so they are not in `agents/model/flag_registry.py`, not on `ModelVersion`, and not in
`check_compatible`; they land in `metadata.json`'s `cli_args` like every train-loop knob and the
launcher forwards them verbatim (`launcher/pool_seed_flag_forwarding_test.py`).

**PROVENANCE.** `seed_pool` writes `<pool_dir>/pool_seed.json` (parent run dir + name, pool dir, N,
the snapshot names, the metadata files copied, and the resulting `win_rate_vs_bots`), and
`run_io._run_lineage` attaches it to the run's lineage block as a **SIBLING key
`lineage.pool_seeded_from`** — never an edit to `fork_parent`. The block is written ONCE at fork
creation and frozen thereafter (`save_model_snapshot`: the existing value always wins), and the pool
is seeded earlier in that same process, so the fact is available exactly when the block is built and
no later restart can add or change it. ⚠️ The record is deliberately NOT written into
`metadata.json` at seed time: `run_io._resolve_fresh_model_dir` treats an existing `metadata.json` as
*"this name is already a run"*, so writing one before the first checkpoint would make a crashed
pre-save fork un-relaunchable under its own name.

**Verified end to end** (2026-09-02, CPU `--debug`, a scratch run root): a hand-built parent pool at
`win_rate_vs_bots=0.901250` seeded into a fork reproduced the parent's own startup line exactly —
`Pool has 1 snapshots, win_rate_vs_bots=90.12% → self_play_fraction=90%` — and the same fork with the
parent's pool hidden exited **3** with the three-way message. Gates:
`agents/training/pool_seed_test.py` (34), including the negative control that the **zips alone** read
`self_play_fraction=0%`, and the idempotence check that a hand-seeded pool comes back byte-identical.

- **PFSP / league-lite (`--pfsp-scale`, `--pool-spread`; both OFF → byte-identical).** A pure recency
  window is a near-50% echo chamber (recent selves beat each other ~evenly), so it never up-weights the
  *kind* of self the trainee is actually losing to. Two opt-in knobs turn it into a prioritised
  curriculum:
    - **`--pfsp-scale S` (default 0.0)** — `SnapshotPool.sample()` blends a per-snapshot HARDNESS factor
      into the weight: `weight = recency × (1 + S·(1 − p))`, where `p` is the trainee's measured win-rate
      vs that snapshot. A self it loses to (`p→0`) is sampled up to `1+S`× more; one it dominates (`p→1`)
      keeps factor 1 — never starved, so coverage is preserved. An unmeasured snapshot uses the mean of the
      known rates (average difficulty); with **no** rates yet (cold start) every factor is 1 ⇒ pure recency.
      The per-snapshot win-rates are exactly the sentinel win-rates the eval already measures: each cycle
      `SelfPlayCallback._update_pfsp_ema` EMA-smooths them (`_PFSP_WR_EMA_BETA`=0.5, to damp ~100-game eval
      noise) and `_prune_and_push_pfsp` prunes the map to the live pool and pushes it to every env via
      `env_method("set_opponent_win_rates", {step: p})` (mirrors the `set_self_play_target` push;
      `RustEnvOpponents.set_opponent_win_rates` → `SnapshotPool.set_win_rates`). The map survives resume
      in `summary.json` (`pfsp_win_rates`). Headline signals: `eval/pfsp_hardest_win_rate` (the most
      up-weighted self) + `eval/pfsp_tracked_snapshots`. Try `1.0–2.0`.
    - **`--pool-spread` (default off)** — replaces the oldest-evicted window with **spread retention**
      (`SnapshotPool._evict_spread`): always keep the newest + the oldest (a weak early self = a forgetting
      tripwire PFSP can up-weight) and thin the most-redundant interior snapshot (smallest neighbour
      step-gap) to an even ladder. So PFSP weights over a genuinely diverse range of selves, not a
      recent-selves cluster. Pairs with `--pfsp-scale`; alone it just diversifies the window.

  Both are threaded into the `SnapshotPool` at **both** construction sites (the per-env-worker pool that
  samples, and the trainer-side pool used for honest sentinel-weight telemetry); off → no extra IPC and the
  legacy sampling/eviction byte-for-byte.

  **REVIVAL VERIFICATION (2026-08-18) — it SURVIVED; nothing needed repair.** PFSP was built
  ai_v8-era and never production-enabled, so gen-16 wanting it ON required checking whether code
  that no test-suite failure would have protected still worked across the fresh-generation reset,
  the frame deletion and two signature bumps. It did: **70/70 existing tests green unmodified**, and
  every call site is intact — both `SnapshotPool` constructions (env-worker + trainer-side), the
  `_update_pfsp_ema` fold in `_collect_pending`, the `_prune_and_push_pfsp` env push, the
  `summary.json` `pfsp_win_rates` resume-load, and `RustEnvOpponents.set_opponent_win_rates`.
  A `--debug --self-play --debug-eval --pfsp-scale 2.0 --pool-spread` CPU smoke ran to
  `Training complete` (exit 0). **What that smoke does NOT show, and why it can't:** pool seeding is
  gated on `win_rate_vs_bots >= SELF_PLAY_START` (0.55) and a fresh debug model sits at ~4%, so the
  pool stays empty and PFSP never weights anything — the smoke proves the flags launch and thread,
  not that they skew. (Since `gen3_supply_guard_v2` that smoke exits `FATAL_SUPPLY` after its third
  eval cycle with the pool still empty — repeat it with `--supply-starve-cycles self_play_pool=0`.)
  **The gap the revival actually closed was in the TESTS, not the code.** Every pre-existing test
  exercised ONE link with the other side mocked (pool math / callback EMA / wrapper forwarding), so
  a green suite said nothing about the composition — the thing a revival has to prove. Two
  end-to-end tests now run measured win-rates through callback → `env_method` → wrapper →
  `SnapshotPool` → `sample()`: `test_measured_winrates_skew_real_sampling_end_to_end` asserts the
  empirical 40k-draw distribution matches the analytic weights (at `pfsp_scale=2.0`, win-rates
  0.1/0.5/0.9 and recency off ⇒ factors 2.8/2.0/1.2 ⇒ shares **0.467 / 0.333 / 0.200**, and the
  self we lose to is drawn **2.33×** as often as the one we dominate), and
  `test_pfsp_off_makes_no_push_and_no_skew_end_to_end` asserts the same composition at
  `pfsp_scale=0` makes **no IPC call at all** and leaves the draw uniform under the same win-rates.

  **Honest caveats (it's a partial-coverage curriculum, not a full PFSP league):** (1) only the
  **`--n-sentinels` (default 5) evenly-spaced sentinels** the eval measures per cycle get a fresh win-rate —
  the other snapshots fall back to the cohort
  mean (treated as average difficulty), so on a 20-deep pool the default PFSP actively re-prioritises ≈¼ of the pool per
  cycle and an un-remeasured snapshot keeps its **last** EMA (a staleness bias toward selves you *used* to lose
  to — watch `eval/pfsp_hardest_win_rate` is tracking a moving target, not a fossil). (2) The `1 +` floor in the
  weight keeps coverage but makes the tilt mild: a self at `p=0.1` vs one at `p=0.5` differ only `(1+S·0.9)/(1+S·0.5)`
  (≈1.4× at `S=2`), and in a healthy gate-pinned pool the sentinel win-rates cluster near 50% so the realised
  prioritisation is modest — lean toward the high end of `S` (or beyond) if you want it to bite. PFSP touches
  **only which frozen opponent is sampled** — never the rollout, GAE, value target, promotion gate, or the
  `win_rate_vs_bots` curriculum ramp — so it cannot corrupt training; the worst case is "does little." A denser
  sentinel count under PFSP + a decay-toward-neutral for stale entries are the obvious follow-ups (deferred).
- **Full roster (v1 + v2 of every archetype).** Training (`TRAIN_BOT_NAMES`, display names) and eval
  (`eval_opponent_names()` / `_EVAL_OPPONENT_SPECS`) both use all eight archetype bots —
  `{Heuristic, Heuristic2, Staller, StallerV2, Aggressive, AggressiveV2, SetupSweep,
  SetupSweepV2}` — because they play differently and the extra playstyle diversity is the
  point. There is no roster flag; the same nine names (eight bots + `random`) feed every
  path. `Random` is eval-only (a cheap "is the model broken" floor, excluded from
  `win_rate_vs_bots`); it is never a training opponent. ⚠️ The setup step of `Heuristic`,
  `Heuristic2`, `SetupSweep` and `SetupSweepV2` was DEAD until 2026-09-29 (F-LF-1 — a `Target`
  enum compared to a str); every training mix before then played the broken bots
  (`eval_and_rating.md`, BOT ERA BOUNDARY). Since then a non-Ghost's Curse is also one of
  their setup moves (+1 Atk / +1 Def / −1 Spe); a Ghost's never is.
- **Resume state in `summary.json`.** `SelfPlayCallback` writes
  `<snapshot_dir>/summary.json` each eval (`win_rate_vs_bots`, `self_play_fraction`,
  `last_eval_step`, `seeded`, `pool_generation`) — `SnapshotPool.persist_summary`/`load_summary`.
  Read at `train_rl_agent` setup → the initial `self_play_fraction` (so a strong resumed model
  starts at the right ramp level, not the 0% cold-start) and the seed-gate decision. Distinct
  from the prober's `eval_traces/*/summary.json`; the legacy `win_rate_vs_bots.txt` is still read
  as a fallback.
- **Opponents sample, they don't argmax.** Training opponents are built with `stochastic=True`
  (formerly the `RLPlayer` default; `RLPlayer` is deleted, T27 P6 — the Rust env core's opponents sample) so the learner trains against the policy's full action
  distribution — a richer, less-exploitable signal than the greedy move. Temperature is
  `--self-play-temp` (default `1.0` = the policy's own distribution; >1 flatter). **The measured
  trainee is always greedy** (`stochastic=False`) — that's what gives `win_rate_vs_bots`
  (curriculum) and `win_rate_vs_pool` (promotion) a stable, comparable control signal. The bots
  are deterministic rule-based players. The **pool sentinels are GREEDY by default since 2026-09-07**
  (`gen3_eval_sentinel_greedy_default_v1`) — `_play_unit` builds the sentinel `stochastic=False` AND
  hands it the **trainee's own teambuilder** (`_sentinel_tb`), so a sentinel matchup is best-vs-best
  on a symmetric team draw and `win_rate_vs_pool` / the snapshot ELO reflect real skill (≈50% vs a
  recent self, ramping with sentinel age). It is eval-only — TRAINING opponents stay stochastic —
  and `--promote-threshold` follows the regime (`0.55` greedy / `0.65` stochastic; else the
  handicap-free pool win rate never clears the gate and the pool freezes). Pass
  **`--no-eval-sentinel-greedy`** for the old regime: sentinel stochastic@`--self-play-temp` drawing
  from the flat pool builder while the trainee draws sample-biased and plays greedy.

  🚨 **THE OLD REGIME'S HANDICAP WAS MEASURED, AND IT IS NOT SMALL.** Against the dense snapshot
  ladder's own edge for the SAME frozen pair (greedy-vs-greedy, both sides sample-biased), the
  asymmetric eval edge favoured the newer snapshot by **+8.9 pp [+7.0, +10.7]** — 60 paired pairs on
  `ai_v12_02_winprob_critic`, 2026-09-07 — which inflated every pre-fix `ladder.json`'s newest nodes
  by +21..+29 Elo. `--eval-sentinel-greedy` was ON for 49 runs (v5.5 through v8) and was dropped,
  **unrecorded**, at the v9 launch; 164 runs since were compared across that boundary with nothing
  on disk naming it. The default flip RESTORES the v8 convention rather than inventing one.

  🚨 **THE REGIME IS RECORDED AND INHERITED — a resume never crosses the boundary silently.**
  `eval_sentinel_greedy` and `promote_threshold` are `ModelVersion` fields (config **v112**) with
  argparse defaults of `None`, so a **flagless resume or launcher restart INHERITS what the
  checkpoint recorded** (`main.train.config.resolve_config` → `inherit_saved_flag`) and a v9-era run
  resumed on this code stays stochastic. That is rule of evidence 15 applied to a flag: a windowed
  statistic never crosses an opponent-regime boundary. Resolution order for the gate: an explicit
  `--promote-threshold` wins; a regime **typed on this argv** re-derives it (inheriting 0.65 into a
  freshly-greedy run would freeze the pool); otherwise the checkpoint's own gate is inherited. Every
  launch prints one line naming both resolved values and their source (`argv` / `inherited` /
  `default`): `⚖️  [EVAL REGIME] …`. A pre-v112 config migrates to stochastic + 0.65 — a record, not
  a guess, for everything above `MIGRATION_FLOOR`; the 49 v5.5–v8 greedy runs sit below the floor
  and their regime survives only in `metadata.json:cli_args`.

  Each eval row in `eval_results.jsonl` carries its own **`sentinel_regime`** stamp
  (`{"greedy", "symmetric_teams"}`) plus exact per-sentinel `counts`. `symmetric_teams` is the
  stricter claim — *both players drew from the LADDER's own builder* — so a **specialist** run
  (`--trainee-team`) records `greedy: true, symmetric_teams: false`: both its players draw from the
  taught team, which is right for eval and is not the ladder's draw. Tests:
  `eval_sentinel_greedy_test.py` (both halves of the switch), `selfplay_callback_test.py` (the
  stamp, incl. the specialist case), `main/train/eval_sentinel_regime_test.py` (the default, resume
  inheritance both ways, the gate's three branches, the round-trip and the migration).
- **Opponent snapshots are version-checked.** They load via `load_opponent_snapshot` (not a raw
  `MaskablePPO.load`), and `SnapshotPool` records a shared `model_config.json` next to its
  snapshots ONCE (write-once, checked before every add — `gen3_pool_arch_record_v1`), so an
  arch-mismatched snapshot fails with a clean `ModelVersionError` instead of
  loading mismatched weights. The check ignores the ride-along keys alone (F-MEM). The load is
  INFERENCE-ONLY: no optimizer and no ride-along acquisition, because it runs after the learner froze
  (`gen3_opponent_inference_load_v1`, `designs/training/learner_lifecycle.md`).
- **HISTORY — the Python road's stale-decision race (the whole bullet, and every module and test it names — `RLPlayer`, `SingleAgentWrapper`, `EpisodeTracker`, `agents/action/mapper.py`, `redecide_rollback_fuzz_test`, `episode_tracker_test`, `single_agent_wrapper_test`, `order_to_action_race_fuzz_test` — is DELETED, T27 P6; the Rust env core has no Python decision loop).** The opponent RE-DECIDED on a stale decision; the trainee crashed — split by who *owned* the
  decision. `SingleAgentWrapper` polls the opponent's `choose_move` on the *training* thread while
  POKE_LOOP mutates its battle, so by serialize time the captured snapshot (`ctx.legal`) can diverge
  from the live battle: POKE_LOOP parses an **in-flight turn-resolution during the model forward**,
  advancing `battle.turn` one ahead of `ctx.turn` (proven by the race trace — mutual Arena-Trap
  Dugtrios, the turn resolves mid-decision). `assert_decision_current` / `action_to_order` raise
  `StaleDecisionError`; handling then splits:
  - **Opponent** — its decision is *internal* to `step` (SB3 never sees it), so `RLPlayer.choose_move`
    catches the error and **re-decides on the now-current request**, bounded (`_OPP_REDECIDE_MAX`),
    with a valid default fallback only if the battle never settles. It must always return a valid
    order: SB3 has **no failed-step path** (a raise kills the `SubprocVecEnv` worker → parent hangs →
    worker-watchdog `os._exit`s → launcher restart). Each attempt's `embed_battle()` records its
    would-be decision into the rolling turn-history, so `choose_move` snapshots the tracker before
    the loop and `EpisodeTracker.restore()`s on a stale attempt — the superseded decision leaves
    **no phantom turn** in the opponent's turn-history obs (only the committed one survives; guarded
    by `redecide_rollback_fuzz_test.py` + `episode_tracker_test.py`). The re-decide guards only up to
    the order `choose_move` RETURNS; `SingleAgentWrapper.step` then re-serializes it via
    `self.env.order_to_action`, re-reading the battle **one more time** — a second, narrower window
    where it can finish/flip-to-wait under us (`ValueError ... not in valid orders ['/choose
    default']`). On that the wrapper falls back to the default order rather than crash (guarded by
    `single_agent_wrapper_test.py` + `order_to_action_race_fuzz_test.py`).
  - **Trainee** — its action is *SB3's*, computed outside `step` and not re-runnable mid-step, so a
    stale trainee decision **crashes** (the Python `gen3_env`, deleted in U3; no fallback): acting on it would corrupt its
    `(obs, action) → (reward, next_obs)` transition. Empirically it doesn't hit this — gated by the
    env's `race_get` request-wait (17 h vs-bots + self-play, zero trainee staleness).
  `_settle_opponent_battle` is a **pre-drain** that only trims how often the opponent re-decides — it
  can't drain *in-flight* messages, which is why re-decide (not settle) is the fix. The comprehensive
  `assert_decision_current` (every axis: moves+disabled, switches+species,
  force_switch/trapped/maybe_trapped/wait/struggle) is the detector; `train/selfplay_opp_redecide_rate`
  surfaces the resolved-race rate. **Full context — mechanism, the race trace, why it was hard, and the
  verification tiers — is in `race_fuzz_README.md`.** (`GEN3_FORCE_SELFPLAY` forces 100% self-play for
  the stress; `GEN3_RACE_TRACE=1` dumps the per-battle cross-thread interleaving into the
  `StaleDecisionError` **and** into the `race_get` silent-stall crash — see below. `StaleDecisionError`
  lives in `agents/action/mapper.py`.)
  - **Force-switch request-delivery deadlock (`_AsyncQueue.race_get`, `env.py`) — FIXED.** A
    *different* failure from the stale-decision race, and a latent bug **inherited verbatim from
    upstream poke-env 0.15.0**: `race_get` races a per-agent `queue.get()` against the
    `_waiting`/`_trying_again` coordination events, and can drop a request the server already
    delivered into the `battle_queue`. Two ways: **(1) stranding** — `asyncio.wait(FIRST_COMPLETED)`
    returns the instant any waiter completes, so an already-set **stale** event wins before the
    equally-ready `queue.get()` runs → `race_get` returns `None`, the agent is marked not-to-move,
    and its request sits unread; **(2) orphan theft** — `race_get` `cancel()`s the pending
    `queue.get()`, which a later `put` can resurrect to dequeue-and-discard the request.
    `_trying_again` goes stale because `env.step` cleared it only on the `None` path, and a
    re-request makes the battle non-`None`, skipping that clear. The trigger is the mutual
    Arena-Trap Dugtrio self-play mirror (trapped-switch `[Unavailable choice]` → stale
    `_trying_again`, then a faint → a `wait`+`forceSwitch` pair whose force-switch is stranded);
    rare (~1/8600 battles), so it only surfaced once self-play was on. **Fix:** `race_get` now
    `cancel()`s **and `await`s** the get to settle it (recovering its item, never orphaning it) and
    **prefers a queued battle over a stale event**, and `env.step` clears `_trying_again` the moment
    its agent receives a battle. Repro + regression guard: `forceswitch_deadlock_fuzz_e2e_test.py`
    (needs a `9XXX` server; `--widen` surfaces the timing race); unit coverage of both failure modes
    in `async_queue_disconnect_test.py`.
  - **Silent-stall watchdog (now a should-never-fire backstop).** Independently of the fix above,
    `race_get` bounds its wait by `_RACE_GET_TIMEOUT_S` (120 s, ~100× a normal step; override with
    `GEN3_RACE_GET_TIMEOUT_S`) and on a silent stall **raises `ShowdownException`** — a hard crash
    that propagates uncaught through the wrapper step chain to the SubprocVecEnv worker, so SB3
    discards the in-flight rollout (no fabricated transition reaches backprop) and the launcher
    restarts from the last checkpoint. It **crashes, never recovers in place** (recovering would feed
    PPO a stale `(obs, action) → (reward, next_obs)`). With `GEN3_RACE_TRACE=1` the wedged battle's
    interleaving is appended to the crash message via `race_trace.dump_recent()` (wedged battle
    ordered last so its newest events survive the launcher's last-100-line crash-file tail; the full
    trace is in `launcher_child.log`). `env.step` also emits `ENVSTEP` enter/race trace lines under
    `GEN3_RACE_TRACE` for debugging this handshake. Kept as defense-in-depth against any future
    request-delivery regression.
- **Self-play engages in the first process, not only after a restart.** The env is built before
  the model exists (the model needs the env's spaces), so on the first self-play process
  `_maybe_engage_self_play` seeds the pool from the loaded weights and rebuilds the env with
  pool opponents (then `set_env`). The worker watchdog is started *after* this, just before
  `learn()`. Later restarts find the pool already populated and skip the rebuild.
- **`--debug --self-play --debug-eval` exercises the real path** (seed → pool eval → promotion)
  on a fast eval cadence (add a short `--eval-freq`), in process and serverless (`--debug` skips
  all eval by default — `--debug-eval` opts in). (`selfplay_opponent_fuzz_test.py`, which drove Python battles, was deleted in T27 P6 slice 6d-2,
  `cbfc2a31`.)

## Stable (cross-run) opponents (`--stable-opponents`, `fixed_opponent_pool.py`)

Load a frozen model from **another, already-finished run** as a **fixed opponent** — measured
against in eval AND (under `--self-play`) played against in training. Design:
`designs/ai_v5/design_stable_opponents.md`. Which FILE a run dir resolves to is the ONE rule above
— the run's **LAST SNAPSHOT**, with `best_model/best_model.zip` as the last-resort fallback.

**Training-mix participation (Stage 2) — "tossed in like a sentinel, becomes a bot when mastered":**
a stable opponent rides the *existing* pool-vs-heuristic split in the opponent sampler
(`rust_env_opponents.py`), no new source-model abstraction:
- **CHALLENGE bucket** (the self-play pool branch, competence-gated by `self_play_fraction`): the
  pool gets the BULK; un-mastered stable opponents share a **capped minority slice**
  (`STABLE_CHALLENGE_SHARE` = 0.20 in `opponent_classes.py`), so a single fixed opponent can never dominate
  training (multiple un-mastered ones SHARE the 20%, so the total stays bounded). It only enters the
  mix once the model clears `SELF_PLAY_START` (a weak model trains on bots first), and only under
  `--self-play` (without it, stable opponents are eval-only — a startup NOTE says so).
- **FLOOR bucket** (the heuristic-bot branch): once the trainee **masters** it
  (`win_rate_vs_ext_<run>` ≥ `--stable-opponent-mastered-wr`, default `0.80`, for
  `_MASTERY_CONFIRM_CYCLES`=2 consecutive cycles — a noise guard since the irreversible flip is
  one-way), it "becomes another bot" — moved to the always-on coverage floor (weighted like an
  unlisted bot). The eval callback tracks a **monotonic** mastered set + a per-label streak counter,
  recomputed each cycle (→ resume-safe), and pushes it via `env_method("set_stable_mastered", …)`,
  exactly like `set_self_play_target`. The recompute+push runs **early** in `_collect_pending` (with
  the training-mix telemetry below), so this cycle's challenge↔floor flips show up in both the pushed
  env state and the reported fractions. **Resume note:** the mastered set lives only in callback
  memory, so after a launcher restart a previously-mastered opponent reverts to the challenge bucket
  until the first post-restart eval re-confirms it (self-healing; bounded by the eval cadence).
- **Training-mix share is reported, not just eval win rate.** The stable opponents' actual slice of
  the training mix shows up in `train/stable_fraction` (challenge un-mastered + floor mastered), with
  `train/selfplay_fraction` (pool) and `train/nonbot_fraction` (their sum); see the Curriculum
  subsection's **Opponent-mix reporting** bullet above for the exact decomposition.
- **Dynamic within-slice selection (`--stable-opponent-pfsp`, default off).** A FLAT capped share
  splits the stable slice UNIFORMLY over the un-mastered opponents — so a generalist hardening against
  several exploiters at once spends equal budget on the axis it already handles and the one it's
  failing. Under `--stable-opponent-pfsp`, the sampler's `_pick_stable` weights the
  un-mastered-stable pick by **`1 − win_rate`** (floored 0.05) — the exploiter it's LOSING to worst
  gets most of the slice, and each fades as mastered (win_rate→1 ⇒ weight→0), then the mastery flip
  retires it to the floor. Win-rates are the same `win_rate_vs_ext_<label>` eval already computes,
  EMA-smoothed (`_PFSP_WR_EMA_BETA`) and pushed each cycle via `SelfPlayCallback._push_stable_mastered`
  → `env_method("set_stable_win_rates", …)` (mirrors the pool PFSP `set_opponent_win_rates`). **The
  TOTAL pool-vs-stable share is unchanged** (still `--stable-opponent-selfplay-share`), so the
  opponent-mix telemetry is
  unaffected — only WHICH un-mastered stable opponent is picked shifts. Training-only (not
  version-locked, forwarded on resume like `--pfsp-scale`); OFF = uniform, byte-identical. **Pairs
  with a raised `--stable-opponent-selfplay-share`.** Motivation: a flat 0.35 share (≈12% exposure
  each of 3 exploiters) left ai_v7_14's hardening flattening at ~0.30 vs the exploiters; the dynamic
  focus + a raised share is the fix. Tests: the sampler's stable-PFSP cases in `rust_env_opponents_test.py` (the old `wrappers_test.py::test_stable_pfsp_*` went with the wrapper).
- The stable-opponent players are **built once per worker** (`load_foreign_opponent` in the env
  factory), so no per-episode reload; each plays **stochastic** at `--stable-opponent-temp` in
  TRAINING but **greedy (temp 0)** in EVAL (a clean yardstick).
- **Surfaced in the launcher Events panel** (via `emit`, like the `[SELFPLAY]` startup lines): a
  `🐴 [STABLE] N cross-run opponent(s): ext_<run> — eval greedy; training ≤<share> of self-play until
  mastered (win_rate ≥ <wr>)` line at startup (and a `🏇 [SELFPLAY] Mastered stable opponent(s) …`
  line on the challenge→floor flip), and each eval-summary event gains a `stable <pct>%` field. (Per-opponent `eval/win_rate_vs_ext_<run>` also rides the normal eval Metrics table.)

- **CLI:** simplest form is just the run dir — `--stable-opponents models/ai_v5_5_popart_N_0607`,
  which resolves to that run's **LAST SNAPSHOT** (the rung table above; it was `best_model` until
  2026-09-06);
  the opponent is **labelled by the run-dir name** (`ext_ai_v5_5_popart_N_0607`, derived
  `best_model`/`snapshots`-aware so a direct `…/best_model/best_model.zip` path still yields the run
  name, not `best_model`). Optional per-entry suffixes: `@<step>` (a specific checkpoint, which
  BYPASSES the ladder), `:<name>` (rename). The resolved zip, its `num_timesteps` and the rung that
  picked it ride the entry (`FixedOpponentEntry.{resolution_rung, resolution_rule, num_timesteps}`)
  and are printed on the `🐴 [STABLE]` / `🥊 [EXPLOITER]` startup lines. **Per-opponent weights (`=<weight>`) are rejected** with a clear
  message (not supported). Knobs: `--stable-opponent-temp` (default 1.0 — the *training* play
  temperature; eval is always greedy) and `--stable-opponent-mastered-wr` (default 0.80 — the
  challenge→floor flip). Parsed + resolved at startup by `fixed_opponent_pool.resolve_stable_opponents`.
- **Compatibility = the OBSERVATION FAMILY only** (two axes: obs family vs model family — see the
  design §3). The gate is **same `arch_signature`** (`ModelVersion.check_opponent_compatible`,
  the obs-family proxy); a mismatch is a **startup FATAL** (`[StableOpponent] FATAL` →
  `TrainExitCode.FATAL_CONFIG`, surfaced to the TUI, no restart). Loaded inference-only via
  `snapshot.load_foreign_opponent` (`env=None`), which **skips `check_compatible`** — so
  `vf_coef`/reward differences (irrelevant to an opponent's forward, which never reads
  the value head) don't block it. ⚠️ A checkpoint that recorded PopArt / the value-dist head /
  `value_from_dist` ON (the `ai_v5_5_popart_*` runs, say) is now REFUSED by `_migrate_config` — those levers
  were deleted (L1, config v131; `model_version/retired_levers.py`) and their parameters cannot be rebuilt.
- **Label namespace `ext_<run>`** — underscore separator (NOT `ext:`) so the emitted metric tags are
  **uniform** with the rest (`eval/win_rate_vs_ext_<run>`, like `eval/win_rate_vs_sentinel_0`), no
  colons in TensorBoard. `is_external` (`startswith("ext_")`) keeps them out of the bot aggregates.
  Both eval callbacks (`PerOpponentEvalCallback` + `SelfPlayCallback`) add the `ext_` labels as
  `FIXED` `EvalItem`s (so they shard + ride the same plan); the Rust eval executor's fixed route plays the
  **greedy trainee vs the greedy stable opponent** (a clean yardstick).
- **Metric set (deliberate, uniform across both callbacks):** per opponent —
  `eval/win_rate_vs_ext_<run>`, `eval/mean_reward_vs_ext_<run>`, `eval/mean_ep_len_vs_ext_<run>`;
  plus `eval/win_rate_vs_external` ONLY for a mini-league (2+ — with one it duplicates its row; it's
  an `_EVAL_SUMMARY` "vs External" row, not a fake per-opponent row); plus a `metadata.json:latest_eval`
  `externals` block. Kept **OUT of** `win_rate_vs_bots` (`bot_mean` excludes them), `win_rate_vs_pool`,
  the best-model aggregate, the `td_resid_tail_mean` headline, and **the ELO FIT itself** (no ladder
  distortion). **NOT emitted for ext:** `td_resid_tail` (a bot/sentinel critic-coverage diagnostic).
  The TUI renders each by its run name with an `(ext)` tag.
- **ELO shown in the eval table** (`record_external_elos`): the elo column for an `ext_` row PREFERS
  the opponent's **own recorded ELO** — read at startup from its `best_model.json` sidecar (or run
  `metadata.json`) `latest_eval.elo` into `FixedOpponentEntry.source_elo` (`_read_source_elo`). It's a
  well-fit, bot-anchored rating (cross-run-comparable since the bot anchors are stable) — e.g. 1902 for
  `ai_v5_5_popart_50m_0607`. **Fallback** (`external_elo`) when the opponent carries no recorded ELO:
  invert the BT win prob from the trainee's live rating + win rate (`R_opp = R_trainee −
  (400/ln10)·logit(wr)`, clamped ≈±676) — a rough single-edge estimate. Recorded as
  `eval/elo_vs_ext_<run>`; the opponent is NEVER a player in the fit itself (no ladder distortion).
- **`best_model/` is self-contained.** Saving the best model copies the run's `model_config.json` AND
  writes a `best_model.json` sidecar (`copy_run_config_to_best_model` + `write_best_model_sidecar`,
  both called from both eval callbacks' best-save). `best_model.json` reuses
  `snapshot.write_checkpoint_metadata` (the per-checkpoint sidecar code) so it carries the
  `latest_eval` block **incl. the run's ELO** —
  `best_model/{best_model.zip,model_config.json,best_model.json}` co-located (arch gate + carried ELO,
  no parent search). Backfilled for existing `models/*/best_model/` dirs.
- **Per-opponent pinned teams (the league FOLD-BACK contract).** A SPECIALIST stable opponent —
  one whose run pinned `--trainee-team` — pilots **ITS OWN team** here, not the shared pool
  (otherwise a trapper exploiter folds back piloting random teams and the pressure it was trained
  to apply evaporates — the realized-matchup lesson applied to the opponent side).
  `resolve_stable_opponents` reads the pin from the opponent run's `metadata.json:
  cli_args.trainee_team` (`_read_trainee_pin`) into `FixedOpponentEntry.team_str` — **fail-loud**:
  a recorded pin whose file is missing raises, and a pin that no longer matches the run's recorded
  MatchupSpec `pin_sha` raises (never a silent pool fallback). TRAINING: the Rust env core's opponent team plumbing (`rust_rollout/teams.py`; the deleted Python wrapper's `_apply_opponent_team` switched `env.agent2._team` **per episode** — the mirror lesson: the opponent's real team is the one its agent holds) plays each pinned entry on its own pin; unpinned episodes use the pool teams. EVAL:
  `team_str` rides `to_cfg()` → the `EvalItem` → `rust_eval.build.eval_builders` (the fixed opponent's pinned
  builder), so the FIXED route measures the opponent piloting its pin (eval matches training, same rule as the trainee's
  own pin). The `[STABLE]`/`[EXPLOITER]` startup lines annotate `[pilots ITS OWN pin: <file>]`.
  (The guard `poke_env_gaps/opponent_pin_fuzz_test.py` — bridge, real battles — was deleted with the Python env core in U3.)
- **Tests:** `fixed_opponent_pool_test.py` (parse + resolve + the arch FATAL gate + the pin
  resolve/fail-loud/sha cases + `register_exploiter_for_eval` dedup),
  `snapshot_test.py::*opponent*/*foreign*` (the loader + `check_opponent_compatible`), and the
  end-to-end `stable_opponent_fuzz_test.py` (bridge, no server — resolve + arch FATAL + foreign
  load + legal stochastic play) (the fold-back realized-team guard `opponent_pin_fuzz_test.py` was deleted in U3).

## From the training leaf (moved 2026-10-10)

> These sections headed `src/agents/training/CLAUDE.md` until its 2026-10-10 cleanup; moved here as they
> stood (minus statements verified FALSE). Where an earlier section of this doc says the same in more
> detail, both are current; fix both in the same pass.

### Self-play opponents (`--self-play`, gated behind pathology hunting)

`SelfPlayCallback` replaces `PerOpponentEvalCallback` and the training opponents become frozen
snapshots of the agent itself, drawn from a directory-backed `SnapshotPool` (`snapshot_pool.py`;
state reconstructed from `<run_dir>/snapshots/` on every restart — no manifest). Design:
`designs/ai_v5/`. 🚨 **A FORK starts with an EMPTY pool, and an empty pool does not disable
`--self-play` — it falls back to the BOT pool**; a genuine fork auto-seeds its parent's and exits
`FATAL_CONFIG` if it still has none (`pool_seed.py`), and ANY run whose pool is still empty after 3
eval cycles exits `FATAL_SUPPLY` (§ above). 🚨 **`max_snapshots` holds on EVERY path that
populates the pool, a directory SCAN included** (`gen3_pool_cap_every_path_v1`): a scan applies the add
path's eviction order, only the trainer's `owns_dir` pool deletes what it evicts, and a pool still over
its cap raises `PoolOverCapError`. 🚨 **The pool's `model_config.json` is a WRITE-ONCE ARCH RECORD**
(`SnapshotPool._record_arch`, `gen3_pool_arch_record_v1`): a snapshot zip carries no `ModelVersion`, so
the record is the only witness of what the pool was written under. It used to be REWRITTEN with the live
version on every add, so after the first promotion every `load_opponent_snapshot` check compared the run
with itself. Now it is written when absent, checked against the live version BEFORE every add (a
`ModelVersionError`, nothing written) and never rewritten.
**Full detail — in [`designs/training/self_play_and_pool.md`](self_play_and_pool.md).**
