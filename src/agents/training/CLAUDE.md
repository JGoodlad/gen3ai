# CLAUDE.md — Training (`src/agents/training/`)

Callbacks, reward manager, episode/turn tracking, stall detection, and the bot-eval pipeline.
**How to launch training** (commands, flags) lives in the root `CLAUDE.md` → Training /
Launcher; this file documents the subsystems' internal design. The `TurnDelta` fold and the
LiveView/TurnView/LegalActions read-models it consumes are documented in
`src/agents/battle/CLAUDE.md`. The obs-build performance gate is in
`src/agents/observation/CLAUDE.md`.

## Reward redesign — registry + PBRS + the no-progress clock (`reward_manager.py`, `progress_clock.py`)

The reward (`Gen3RewardManager`) is organised as a **registry of class-tagged terms**
(design `designs/ai_v5/design_markovian_reward_and_features.md`). Every `RewardBreakdown` field is one
entry in `RewardBreakdown._REGISTRY` mapping name → `RewardClass`. The **BIAS class is folded
generically** off the registry (`_fold_bias_refund` sums `registry_fields(BIAS)`); TERMINAL and the
three PBRS terms are **explicit named folds** (`_fold_material_pbrs` / `_fold_belief_pbrs` /
`_fold_status_pbrs`) because each PBRS term carries its own `_prev_phi_*` telescoping state a generic
loop can't hold — `process_turn_reward` reads as a short phase sequence over these helpers:

- **TERMINAL** (`win_loss`, the ±30) — emitted as-is; never shaped/flag-affected. Out of scope.
- **PBRS** (always telescoping, objective-neutral; `Φ(terminal)=0`): `pbrs_material` (the material
  potential **Φ_mat**, design §2), `pbrs_belief` (the shipped incoming-KO belief PBRS — RENAMED from
  the mis-named `pbrs_material`), and `pbrs_status` (the non-damaging-tempo status potential **Φ_status**,
  design §2.7 — `bias_redesign`-gated, see below). The field holds `γ·Φ(s′)−Φ(s)`; `PBRS_GAMMA` MUST ==
  the PPO gamma (asserted in `train_rl_agent.py` after the model is built — the manager is built first,
  in the env factory, so it can't assert in `__init__`).
- **BIAS** (everything else) — additive shaping whose additive↔telescoping mix is set by
  `--bias-additivity` λ∈[0,1] (`RewardConfig.bias_additivity`, default 1.0). Implemented as
  **accumulate-and-refund**: each BIAS term emits its current per-turn value; the manager accumulates
  `_bias_acc` and emits `bias_refund = −(1−λ)·Δacc` (the low-variance accumulator-potential spread). At
  **λ=1 the refund is identically 0** → byte-identical to the old additive biases (the no-op the
  registry-coverage / no-op-equivalence tests pin).

**Φ_mat** (`_compute_phi_mat`) = `MAT_HP_WEIGHT·(Σ our_hp − Σ opp_hp) + MAT_ALIVE_WEIGHT·(n_alive_ours
− n_alive_opp)`, over the **declared team size** (unrevealed opp mons = full-HP-alive → `Φ_mat(s_0)≈0`,
no opp-reveal jumps, no start-state variance). It REPLACES the old unconditional `hp_ours/hp_opp/
faint_ours/faint_opp` base spine — material no longer banks the lead, so every win returns +30 / loss
−30 (the clutch-vs-dominant fix). The old asymmetric `−0.75 FAINT_MATERIAL_PENALTY` is REMOVED (folded
into `MAT_ALIVE_WEIGHT=1.25`, a state potential, not a bias). The `+2.0` explosion literal is deleted
(survive-Explosion credit rides Φ_mat); `explosion_block` is kept.

**Φ_status** (`_compute_phi_status` / `_fold_status_pbrs`, `pbrs_status`) = `STATUS_TEMPO_WEIGHT·(opp_tempo
_statused − our_tempo_statused)` over **non-fainted par/slp/frz mons only** (`_TEMPO_STATUSES`). It
restores the *standing* value of a held non-damaging status that the event-form `status` reframe drops —
sleep/freeze/para "lose the opponent turns", value `Φ_mat` can't see (Toxic/burn/poison value is the chip
→ already in `Φ_mat`, so they're excluded to avoid a double-bridge). Nobody is statused at `s_0` →
`Φ_status(s_0)=0`, `Φ_status(terminal)=0` → it telescopes to **zero net** (policy-invariant dense signal,
not a net bias). **Gated on `bias_redesign`** (the default count-diff `status` BIAS already pays the
standing value → folding `Φ_status` there double-counts; OFF → `pbrs_status≡0`, `_prev_phi_status` stays
None, byte-identical default). It adds **no** resume-immutable field — it rides the existing
`bias_redesign` flag (design §2.7 / §7.4 hedge).

**The no-progress clock** (`ProgressClock`, `progress_clock.py`) is an episode-scoped
`turns_since_progress` counter **owned by `EpisodeTracker`** (NOT LiveView — it is cross-turn state;
precedent = `HiddenPowerTracker`). It is updated at `record()`/`embed_battle` time (so the obs is fresh
— poke-env runs `embed_battle` before `calc_reward`), and read by BOTH the obs encoder (`value()` →
the `vec[14]` scalar) and the reward (`last_penalty` → `no_progress_tax`), so **obs and reward key on
one value**. The ternary predicate per decision window: PROGRESS (our-attributed damage ≥3% / status
landed / hazard layer / forced opp commit / **an our-owned residual — Toxic/poison/burn or Leech
Seed/Curse/Nightmare — chipping the opp NET-down** → reset), DENIED (freeze), NO_OP (deliberate
wheel-spin → increment + charge, gated off on forced-switch windows and when no switch is legal).
DENIED splits two ways (`_denial_kind`): **exogenous** (miss / Protect-block / cant) is ALWAYS frozen;
a **productive heal** is frozen only for `HEAL_FREEZE_GRACE`=2 consecutive windows — a SUSTAINED heal
with no progress (the self-play mirror heal-war) then falls through to NO_OP and CHARGES, so the
250-turn stall finally registers. The residual-PROGRESS branch is what keeps a *winning* Toxic/Leech
defensive stall from being taxed (the discriminator is the opp net-losing HP; a heal-war where they
out-heal the tick still charges) — validated end-to-end by `progress_clock_fuzz_test.py` (bridge, real
battles: a winning-residual window is never charged). The env (`gen3_env.py`) folds the delta once at
embed time, updates the clock, caches it for `calc_reward` (no double fold), and wires
`reward_manager.progress_clock = tracker.progress_clock`.

**Anti-stall terminal (`--draw-penalty`, default −30.0 = byte-unchanged).** The trainee FORFEITS a
stalled battle at the turn cap (`gen3_env` `ForfeitBattleOrder` at turn ≥ `StallConfig.threshold`), so
a 250-turn stall ends as a forfeit-**loss** (`lost=True`), NOT a tie. The terminal therefore detects a
timeout by **`live.turn >= _TIMEOUT_TURN_CAP`** (synced to `StallConfig.threshold`), not by won/lost:
`if won: +30; elif finished: draw_penalty if timed_out else −30`. Set `--draw-penalty -35` to make a
stall-to-cap strictly worse than a clean loss (cancels the γ=0.9999 discount pull of delaying an
inevitable −30). Resume-immutable, value-checked (`MODEL_CONFIG_VERSION 6→7`, `check_reward_config`).

**Staged rollout (`RewardConfig.bias_redesign`, `--bias-redesign`, default OFF).** OFF = the
**single-variable default run**: today's anti-spam taxes + roar/status/spikes, so the ONLY reward
change vs the live baseline is the material clutch-fix (clean attribution). ON = the no-progress clock
SUBSUMES the escalating anti-spam family (repetition/bouncing/dead-matchup/struggle suppressed) and the
clock charge is active. The `turns_since_progress` OBS scalar is present EITHER way (the clock always
tracks it), so both arms share one architecture and can A/B by resume. `--bias-additivity` /
`--mat-alive-weight` / `--bias-redesign` are resume-immutable, value-checked by
`ModelVersion.check_reward_config` (the same machinery as `--vf-coef`). Tests: `reward_redesign_test.py`
(registry coverage, Φ_mat telescoping + terminal-zeroing, **Φ_status non-damaging-only + gated-off-default
+ telescopes-to-zero**, bias no-op + parameterized blend, the bias_redesign reframes, the full
ProgressClock predicate), plus the updated `reward_manager_test.py`.

**Belief-risk-scaled switch BIAS lever (`--switch-bias-weight`, default 0.0 = OFF).** The shipped
`pbrs_belief` is policy-INVARIANT (a telescoping potential) so it can't move a *converged* under-switch
preference — verified on `run_20260607_102632`: switch-mass still inverts vs P(KO), stay-and-die ≈ 61%
== the V1 control. The fix (`design_reward_switching.md §7`, `impl_step6`) adds two **BIAS-class** terms
that *do* tilt the objective: `stay_risk_tax = max(−w·risk, −2.0)` for STAYING into a high imminent-KO
spot a safe pivot could escape, and `escape_risk_bonus = w·0.5·risk` for escaping it (asymmetric < the
tax → no farm). `risk = max(phys_pko,spec_pko)·(1−P(outspeed))` from the incoming belief. Hardened gates
(red-teamed): never tax a **trapped** stay (`_cur_can_switch` from the decision-time `ctx.mask`), an RNG
fizzle (`our_failed_to_move`), a KO'ing stay (`opp_fainted`), or a forced stay (a `_prev_safe_pivot`
bench mon with raw P(KO) ≤ `SAFE_PIVOT_PKO_MAX`=0.35 must exist; the escape bonus needs it too). Snapshots
are decision-time (set end of last turn / in `record_action`), read before `_fold_belief_pbrs` overwrites
them. **Reward-only — no obs/arch change** (ARCH unchanged; `MODEL_CONFIG_VERSION 4→5`), resume-immutable
(`check_reward_config`). Being BIAS-class it rides `--bias-additivity`, so a fixed weight at **λ=1 vs λ=0**
is the causal A/B for "is it the objective tilt that helps." Tests: `reward_redesign_test.py::TestSwitchBias`.

## Bot evaluation (subprocess, non-blocking)

**Flat schedule, full roster.** Eval fires every `EVAL_FREQ_STEPS` (2M steps) and plays
`EVAL_GAMES` (100) games per opponent — one cadence, one game count, applied uniformly to
every bot *and* every self-play sentinel (no maturity tiers, no per-opponent caps). The
roster is the full set of eight archetype bots — both the v1 and v2 of each
(`heuristic`/`heuristic2`, `staller`/`staller_v2`, `aggressive`/`aggressive_v2`,
`setup_sweep`/`setup_sweep_v2`) — plus `random` as the eval-only "is-the-model-broken"
floor (excluded from `win_rate_vs_bots`). All nine are the single source of truth in
`_EVAL_OPPONENT_SPECS` / `eval_opponent_names()`, shared by the bot path, the self-play
path, and the worker. There is no roster flag — every bot always plays, because they play
differently and the playstyle diversity is the point. The flat numbers are safe precisely
because eval is non-blocking and **skips a cycle while the previous one is still running**
(below): a heavier roster self-throttles to a sparser cadence instead of needing tuned
ceilings.

`PerOpponentEvalCallback` (non-self-play path) does **not** eval in-process. On each
scheduled step it snapshots the live weights (`model.save`) and spawns `--eval-workers`
(default 3) `main.eval_worker` subprocesses that **work-steal at battle granularity** from a
shared pool, load the **frozen** snapshot, and play against the shared Showdown server (or the
in-process bridge) **without pausing training**. Each opponent's `EVAL_GAMES` are split into
**shard units** of `--eval-shard-games` (default 25 → 4 shards/opponent); a worker claims units
(atomic `O_EXCL` lock per `unit_id`), plays them, and publishes one `shard__<unit_id>.json` of
**raw** counts; the parent pools an opponent's shards back into one **exact** result. This is the
long-tail fix — when fewer opponents remain than workers, the straggler's remaining games spread
across idle workers instead of one worker grinding a whole opponent alone (workers are capped by
unit count, not opponent count). The whole mechanism lives in the **`eval_sharding/` package**
(below); when all workers finish the parent merges → TensorBoard + TUI + best-model (the winning
snapshot is promoted by copy, not re-saved). Forensic traces land under
`<run_dir>/eval_traces/step_<N>/<opponent>/` as a per-captured-battle triple (`write_battle_record`,
`battle_recorder.py`): `<outcome>_s<shard>_NNN_summary.json` (the human-readable per-decision dump) +
`<outcome>_s<shard>_NNN_states.npz` (raw obs/logits/values for the prober) +
**`<outcome>_s<shard>_NNN_replay.html`** — a self-contained, **browser-watchable** Showdown replay of
that battle (poke-env `save_replay` over the accumulated protocol stream). The first two are
prober-only; the HTML lets a human just open the game in a browser (no checkout, no prober) — the
only watchable replay for *non-stall* eval battles (stall games still get their own `stalls/*.html`).
The `s<shard>_` prefix namespaces the files so concurrent shards of one opponent never collide. All
three sit alongside a per-cycle
**`eval_manifest.json`** (`write_eval_manifest`) recording exactly which model produced them
— `num_timesteps`, `git_hash` + `arch_signature` (read from the run's `metadata.json` /
`model_config.json`), and a `snapshot` pointer. The eval snapshot is normally ephemeral
(`model.save` → workers load → deleted in `_cleanup`) and the eval `step` rarely lines up with
a persisted `checkpoint_<N>_steps.zip`, so the prober can't reload the *exact* weights unless
they're retained: `--keep-eval-snapshots N` copies the snapshot into
`eval_traces/step_<N>/snapshot.zip` (keeping the N most-recent) and points the manifest at it.
The prober consumes the manifest to load the exact model, falling back to the nearest
checkpoint. **The trainer grooms the traces it writes**: after each cycle
`_prune_eval_traces` keeps only the `--keep-eval-trace-steps` (default 20) most-recent eval
step dirs, and `_prune_eval_snapshots` keeps the `--keep-eval-snapshots` (default 10)
most-recent snapshots — so `eval_traces/` stays bounded without any external task
(`python -m main.prober.groom` is the manual fallback). **The same cycle also bounds the run's
two append-only debug dirs** via `_prune_run_artifacts` (`artifact_retention.py`, a dedicated
module — not bolted onto this busy callback): keep the `--keep-stalls` (default 50) most-recent
`stalls/stall_*.html` replays and the `--keep-crashes` (default 10) most-recent
`crashes/restart_err_*.txt` launcher diagnostics, newest-by-mtime, `0` = keep all. Same
producer-grooms-its-own-data contract; `python -m agents.training.artifact_retention <run_dir |
models_dir> [--apply]` is the manual fallback (dry-run by default; sweeps every run under a
`models/` tree). The eval summary itself is
written to `metadata.json` as a **top-level `latest_eval`** block (step-labeled, NOT
nested under a checkpoint) — robust to the async timing (an eval can finish after a
newer checkpoint, or before any checkpoint exists); `save_model_snapshot` carries it
forward so a later checkpoint never erases it. That top-level block is the canonical,
timing-robust record; **additionally, `record_checkpoint` stamps a point-in-time copy
of the then-current `latest_eval` into each checkpoint's entry** (both the per-checkpoint
sidecar `.json` and the run-level `snapshot_history` entry, under a `latest_eval` key) so
each checkpoint carries the most-recent eval+pool stats as of when it was saved. The
embedded block keeps its own `step`, so storing it under a possibly-newer checkpoint never
mislabels which weights were measured (`snapshot._read_latest_eval` reads it; the union
builder `_build_snapshot_entry` keeps sidecar + history in lockstep).

The frozen snapshot makes parallel eval correct (a worker can't read mutating in-memory
weights), and the fresh process returns all eval memory to the OS on exit (no fragmentation
in the trainer). Behaviors:
- A trigger that fires while the previous cycle still runs is **skipped** (logged) — on CPU
  an eval can outlast its interval; cadence just goes sparser.
- A worker crash is **logged-and-continued**, never fatal (its opponents are just missing
  for that cycle).
- **Graceful shutdown waits for eval to finish**: a scheduled restart is self-initiated by
  `GracefulRestartCallback` at a rollout boundary and the launcher won't force-kill until the
  child overruns the deadline by `--restart-grace-minutes` (20 min), so the drain budget is a
  full `_ABORT_EVAL_DRAIN_SEC` (10 min) AFTER the checkpoint is saved — long enough for a CPU
  eval to complete. Even the pathological forced-SIGTERM case (already overran → ~90s SIGKILL)
  is safe: the checkpoint is saved first, only the in-flight eval can be lost.
- **On resume the last eval is re-published to the TUI** from the resumed checkpoint's
  `metadata.json` (`replay_last_eval_to_tui`), so the eval panel isn't blank until the next
  cycle. This covers the **self-play `pool` block too** — the aggregate (`win_rate_vs_pool`,
  `mean_reward_vs_pool`, monotonicity, snapshot count) and every per-sentinel row are
  re-published from the saved block, with the saved step tags, so Pool/sentinel rows survive
  a restart exactly like the bot rows (no waiting a full cadence for fresh numbers). Safe
  because the pool only changes at an eval-collect — the same moment the block is persisted —
  so the saved rows match the pool reconstructed from `snapshots/`. A pre-seed eval persists an
  empty `sentinels` list, which isn't re-published (nothing to show yet).

| Flag | Default | Notes |
|------|---------|-------|
| `--eval-workers` | `5` | Eval subprocesses per cycle; work-steal **shard units** from a shared pool. Capped at the unit count (≈ opponents × shards-per-opponent, so sharding lets the full pool help). Self-play doubles this (→ `10`) since sentinel matchups run the model for both players. |
| `--eval-shard-games` | `25` | Games per work-steal **shard unit** (battle-level work-stealing). Each opponent's `EVAL_GAMES` split into chunks any idle worker drains → the long tail collapses to one shard (≈4-shards-per-opponent default = ~4× shorter tail). Smaller = finer tail collapse but more player builds / (on websocket) more connection churn — the bridge is preferred for fine shards. `>= EVAL_GAMES` ⇒ one shard/opponent = the original opponent-level behaviour. Aggregation is exact (Σwon/Σfinished etc.); see the package below. |
| `--eval-device` | `cpu` | Device for eval-worker inference. `cpu` decouples eval from the training GPU. |
| `--eval-concurrency-per-worker` | `1` | Battles each worker overlaps **within** its claimed opponent (single-thread asyncio latency-hiding — NOT multi-core). `1` = today's sequential play. Threaded to the constructor's `eval_concurrency` → `cfg["concurrency"]` → `run_local_battles(concurrency=)` (bridge) / the player's `max_concurrent_battles` (websocket). See the concurrency note below. |
| `--keep-eval-snapshots` | `10` | Retain the N most-recent eval weight snapshots in `eval_traces/step_<N>/snapshot.zip` (~27MB each; default ≈270MB) for bit-exact prober replay. `0` writes the identity manifest only; the prober then loads the nearest persisted checkpoint. The trainer auto-prunes to this cap each cycle. |
| `--keep-eval-trace-steps` | `20` | The trainer keeps only the N most-recent eval **step dirs** under `eval_traces/` after each cycle (`0` = keep all), so forensic data stays bounded. `python -m main.prober.groom` is the manual fallback. |
| `--keep-stalls` | `50` | Each cycle keep only the N most-recent `stalls/stall_*.html` replays (`0` = keep all). A self-play run writes thousands (~80 KB each); this caps the dir. `artifact_retention.py`; CLI fallback `python -m agents.training.artifact_retention`. |
| `--keep-crashes` | `10` | Each cycle keep only the N most-recent `crashes/restart_err_*.txt` launcher diagnostics (`0` = keep all). Same module/CLI as `--keep-stalls`. |

**TD-residual tail metric (`eval/td_resid_tail_*`).** Each cycle also folds a **left-tail
statistic of the per-decision critic surprise** δ(t) = r(t) + γ·V(s_{t+1}) − V(s_t) — the same
formula the prober uses (`main/prober/session.py::_td`, the single source of truth). `BattleRecorder`
accumulates δ live (one-step delayed backfill, closing each transition at the next `record()` when
the reward is finalized and V(s′) is known; the last decision has no δ). It costs **zero extra GPU**:
δ is computed only over the battles eval already captures forensically (where `need_aux=True` already
paid for V(s)), pooled per opponent (one `EvalRLPlayer` per matchup → `td_tail()`), and folded as a
**CVaR@5%** (mean of the worst 5%, `TD_TAIL_FRAC`; single min below `TD_TAIL_MIN_SAMPLES`=20). It
rides the exact win-rate plumbing — worker `shard__<unit_id>.json` (raw δ pooled across shards) → `merge_eval_results` →
`eval/td_resid_tail_vs_<opponent>` + `eval/td_resid_tail_mean` (TB + TUI), the `metadata.json`
`latest_eval` block (per-opponent + pool aggregate), and the append-only `eval_results.jsonl`. The
run's `model.gamma` is threaded into the worker (`base_cfg["gamma"]`) so the live δ matches the
prober's offline recompute (guarded by `td_residual_parity_fuzz_test.py`). More-negative = the critic
got blindsided more often — the **leading indicator for the critic-coverage obs work** (it moves in a
cycle or two, where saturated win-rate / gate-pinned `win_rate_vs_pool` / wide-CI ELO don't).

**Intra-worker concurrency (`--eval-concurrency-per-worker`, default `1` = sequential).** Each
worker overlaps up to N battles **within** its claimed opponent. This is **single-thread asyncio
latency-hiding, NOT multi-core** — everything (the obs build + PyTorch forward in `choose_move`, the
bridge/server I/O) runs on the one `POKE_LOOP` thread with BLAS pinned (`OMP/MKL=1`), so concurrency
only overlaps the time a worker is *blocked* on the bridge subprocess / websocket round-trip with
another battle's forward. The ceiling is **one core of compute**: a single-core bridge benchmark
(`/tmp/eval_concurrency_bench.py`, NN trainee vs bot and vs NN sentinel) measured ~**2.0× decisions/sec
at conc=3** on spare cores (plateau ~3; bot eval ≈2.0×, the heavier NN-vs-NN ≈1.8×) — i.e. about half
the per-decision wall-time at conc=1 was bridge I/O wait. **The old `_EVAL_SUBPROCESS_CONCURRENCY` = 1
default and its "measured slower" note were the *saturated* regime** (eval contending with training's
64 env workers for already-full cores — there the extra event-loop overhead nets negative); on **spare
cores (idle box / the cycle tail)** it's a clean ~2×. So the live gain runs between 1× and 2×
depending on how saturated the box is during the eval window; default stays `1` (opt-in). It does
**not** use idle cores at the tail — that needs *process-level* sharding (chunk one opponent across
workers); concurrency stacks multiplicatively on top of that (≈`2 × #shards`). Cross-opponent
parallelism is still the `--eval-workers` (5) subprocesses work-stealing the pool.

### Battle-level work-stealing (`eval_sharding/` package)

The *process-level* tail fix above is the `eval_sharding/` package — a small, deeply-encapsulated
unit with a narrow interface (4 focused files, no mega-file):

- **`units.py`** — `EvalItem` (one opponent the parent declares) + `ShardUnit` (a chunk of its
  games) + `plan_units(items, shard_games)`, a **pure** partition: split each item's games into
  ≤`shard_games` chunks (Σshards == n_games exactly), ordered LPT-ish (cost-descending items, shards
  round-robined) so every opponent starts early and the expensive ones lead.
- **`results.py`** — `ShardResult` (raw additive metrics: won/finished, reward+turn sums, the raw δ
  list — never a reduced ratio) + `aggregate`, which pools an opponent's shards back **exactly**:
  win_rate=Σwon/Σfinished, reward/ep_len count-weighted, and the TD tail by **pooling raw δ then one
  `td_tail`** (a CVaR can't be averaged). `td_tail` + its constants live here (the single source of
  truth; `eval_callback` re-exports them, so the dependency is one-way `eval_callback → eval_sharding`).
- **`pool.py`** — `ShardedEvalPool`, the deep coordinator. Parent: `write_plan(run_dir)` →
  `collect(result_dir)`. Worker: `from_plan(run_dir)` → `claim_next(claim_dir)` / `publish(...)`. It
  hides every filesystem mechanic; the worker never touches a lock file, the parent never touches a
  shard file. The plan (`plan.json`, items + shard_games) is the **single source of truth** both
  sides read — neither reconstructs the universe independently, so they can't drift.
- **`merge_eval_results`** is now a thin delegate to `ShardedEvalPool.collect` returning the same
  `merged` shape every downstream consumer already reads (record_per_opponent / build_bot_eval_block
  / record_elo / pool & externals blocks are **untouched**), plus additive `counts` (exact W/L) and
  `coverage` siblings.

**Exactness caveat (documented, by design):** win_rate / reward / ep_len are exact regardless of
`shard_games`. `td_resid_tail`'s *aggregation* is exact (pool the raw δ, compute the CVaR once), but
the *captured-battle sample* it's computed over shifts slightly with the shard count — the forensic
capture quota is per-unit (scaled `max(1, ⌈quota/shards⌉)`), so which battles contribute δ depends on
the split. It's a sampled diagnostic either way. Forensic trace files are namespaced by a per-unit
`trace_tag` (`{outcome}_s{shard}_{idx}`) so concurrent shards of one opponent don't collide in the
shared `eval_traces/step_<N>/<opponent>/` dir. Per-cycle `run_dir` is wiped at cleanup (and cleared
at launch), so no lock/shard/plan ever leaks across cycles. Sentinel/fixed opponent models are cached
per worker by path (immutable within a cycle → safe; the version check rides the first load) so a
fine split doesn't pay an N× 27MB deserialize. Worker rewrite: `eval_worker._play_unit` (one fresh
trainee + opponent per unit → independent measurement) + a per-worker model cache; tests:
`eval_sharding_test.py` (partition + aggregation-exactness property + claim-once + coverage),
`eval_sharding_fuzz_test.py` (real bridge battles through the real worker → exact pooled result).

### Rating-model seam (`rating.py`) — extensibility for Glicko-2 / TrueSkill

The live skill rating is anchored Bradley-Terry (`elo.py`), a *global batch* fit. `rating.py` is the
**ready drop-in point** for a different model without re-plumbing: `MatchRecord` (exact counts +
draws + `period_id` + optional opponent priors — the union BT, Glicko-2 and TrueSkill all need),
`RatingResult`, a `RatingModel` **batch** protocol, and `BradleyTerryRating` — a thin adapter over
`elo.fit_pairwise` whose ratings+SE are **byte-identical** to the live fit (pinned by `rating_test.py`).
`eval_rows_to_match_records` bridges the existing `EvalRow` history. The live `record_elo` path is
**deliberately unchanged** (zero risk): the seam exists and is tested, but routing through it buys
nothing until a new model is actually wanted — and Glicko-2 is *sequential* (period-by-period RD
carry-forward), so it needs the `SequentialRatingModel` sibling sketched in the module footer, not the
batch `fit`. Data fidelity is already in place: `eval_results.jsonl` now carries exact per-opponent
`counts` (additive, backward-compatible), so a future Glicko backfill has an exact ladder even under
partial shard coverage (where `win_rate × n_games` would be ambiguous).

## Self-play opponents (`--self-play`, gated behind pathology hunting)

When `--self-play` is set, `SelfPlayCallback` replaces `PerOpponentEvalCallback` and the
training opponents become frozen snapshots of the agent itself, drawn from a directory-backed
`SnapshotPool` (`snapshot_pool.py`; state reconstructed from `<run_dir>/snapshots/` on every
restart — no manifest). Design lives in `designs/ai_v5/`. Key behaviors:

- **Eval + promotion are NON-BLOCKING (frozen-snapshot subprocess), mirroring
  `PerOpponentEvalCallback`.** Self-play eval no longer runs in-process on the training thread.
  On a trigger step `SelfPlayCallback` freezes the live weights to disk (`model.save`) and
  spawns `--eval-workers`×2 (default 10) `main.eval_worker` subprocesses that **work-steal BOTH
  the bot roster AND up to 5 pool sentinels** (all split into shard units) from one shared pool (the
  worker's `_play_unit` SENTINEL branch plays the frozen trainee greedy vs each sentinel stochastic);
  training continues immediately. On a later
  `_on_step` poll the parent merges per-opponent + per-sentinel results → `win_rate_vs_bots` /
  `win_rate_vs_pool` / `sentinel_monotonicity`, records to TensorBoard + the TUI + metadata.json
  (with the `pool` block), persists `win_rate_vs_bots` (feeds `heuristic_fraction` next run),
  saves best by **copying** the frozen snapshot, and — if `win_rate_vs_pool > --promote-threshold`
  — **promotes the FROZEN snapshot into the pool by file-copy** (`SnapshotPool.add_from_path`):
  the live model has advanced since launch, so re-saving `self.model` would promote the wrong
  weights. Sentinels load via `load_model_snapshot` against the pool's shared `model_config.json`
  using `current_model_version(mappings)` — a stale-arch snapshot fails with `ModelVersionError`,
  never loads silently. The **only** training-thread work per cycle is the `model.save` freeze +
  one cheap `opponent_default_stats` IPC at collect; all battles / model loads / inference run in
  the worker processes, and the trainer holds no live eval connections (the worker rebuilds
  opponents/teambuilders/mappings itself). Skip-while-running, worker-crash-logged-and-continued,
  graceful-shutdown `drain()`, and resume-republish all behave exactly as the bot path above. The
  launch→poll→collect→drain mechanics are the **shared** `eval_callback.spawn_eval_workers` /
  `merge_eval_results` / `persist_eval_snapshot` / `prune_eval_*` / `replay_last_eval_to_tui`
  helpers, so the two non-blocking paths can't drift. `--debug --self-play` uses a fast eval
  cadence (every 4k steps, 3 games) so a short CPU smoke exercises seed → pool eval → promotion.
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
  pick lives in `MaskableAgentWrapper._select_episode_opponent`, an O(1) in-memory `rng.choices`
  with zero per-step cost. All three default to the original behavior, so an unset run is unchanged.
  Crucially the heuristic-vs-pool split is **no longer fixed per process**: every training env
  picks its opponent **per episode** in `MaskableAgentWrapper.reset()` from a live
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
  a pure mirror of `MaskableAgentWrapper._select_episode_opponent` (it does **not** change selection).
  The four mutually-exclusive opponent types (bot / pool / un-mastered-stable / mastered-stable) sum
  to 1; the metrics report **`train/selfplay_fraction` = P(pool)** (REPOINTED — it used to log `sf`),
  **`train/stable_fraction` = P(any stable)** (un-mastered in the challenge **+** mastered in the
  weighted floor — a mastered stable "becomes a bot" so it's NOT in `sf`), and **`train/nonbot_fraction`
  = pool + stable** (= 1 − bot; bot is left implicit). `nonbot` is independent of the stable challenge
  share (it cancels); the per-bucket split needs three **reporting-only** inputs threaded into the
  callback from `train_rl_agent` (the capped `stable_challenge_share`, the `--bot-weights` vector, and
  `len(OPPONENT_CLASSES)` — the floor roster, which excludes eval-only `random`). Distillation drops
  both stable buckets (`_distill_deployed`, last-reconcile state → a same-cycle flip lags ≤1 cycle).
  With no stable opponents these reduce to `selfplay_fraction = nonbot = sf·P`, `stable = 0`.
  `_opponent_mix_fractions` is a hand-written **mirror** of the wrapper's selection, so the anti-drift
  guard is `wrappers_test.py::test_mix_fractions_match_actual_sampling`: it runs the REAL
  `_select_episode_opponent` thousands of times and asserts the empirical pool/stable shares match
  the analytic fractions (the per-case `selfplay_callback_test.py::test_opponent_mix_*` pin the math
  itself). A future selection change that isn't mirrored fails that cross-check.
- **Seeding is GATED on competence; the pool is a SLIDING WINDOW (nothing pinned).** The pool is
  seeded only once win rate clears `SELF_PLAY_START` (at startup via `_maybe_seed_pool`, or the
  moment it crosses mid-run in `_collect_pending`), so the first self-play opponent is a
  *competent* model — never the random/weak step-0 seed of old. Nothing is pinned: the oldest
  snapshot (incl. the seed) ages out as the window slides past `max_snapshots`, so the floor
  stays a recent self; anti-forgetting is the heuristic floor, not a pinned seed.
- **Full roster (v1 + v2 of every archetype).** Training (`OPPONENT_CLASSES`) and eval
  (`eval_opponent_names()` / `_EVAL_OPPONENT_SPECS`) both use all eight archetype bots —
  `{Heuristic, Heuristic2, Staller, StallerV2, Aggressive, AggressiveV2, SetupSweep,
  SetupSweepV2}` — because they play differently and the extra playstyle diversity is the
  point. There is no roster flag; the same nine names (eight bots + `random`) feed every
  path. `Random` is eval-only (a cheap "is the model broken" floor, excluded from
  `win_rate_vs_bots`); it is never a training opponent.
- **Resume state in `summary.json`.** `SelfPlayCallback` writes
  `<snapshot_dir>/summary.json` each eval (`win_rate_vs_bots`, `self_play_fraction`,
  `last_eval_step`, `seeded`, `pool_generation`) — `SnapshotPool.persist_summary`/`load_summary`.
  Read at `train_rl_agent` setup → the initial `self_play_fraction` (so a strong resumed model
  starts at the right ramp level, not the 0% cold-start) and the seed-gate decision. Distinct
  from the prober's `eval_traces/*/summary.json`; the legacy `win_rate_vs_bots.txt` is still read
  as a fallback.
- **Opponents sample, they don't argmax.** Training opponents are built with `stochastic=True`
  (now the `RLPlayer` default) so the learner trains against the policy's full action
  distribution — a richer, less-exploitable signal than the greedy move. Temperature is
  `--self-play-temp` (default `1.0` = the policy's own distribution; >1 flatter). **The measured
  trainee is always greedy** (`stochastic=False`) — that's what gives `win_rate_vs_bots`
  (curriculum) and `win_rate_vs_pool` (promotion) a stable, comparable control signal. The bots
  are deterministic rule-based players. The **pool sentinels default to stochastic@`--self-play-temp`**
  (mirroring how they act as training opponents) — so a sentinel matchup is greedy-trainee vs
  stochastic-sentinel, a deliberate asymmetry that inflates `win_rate_vs_pool` by a ~constant
  temperature handicap (≈15–20 pts; the [ELO caveat](#elo--skill-rating) below). **`--eval-sentinel-greedy`
  makes the sentinels greedy too** (`_play_unit` builds the sentinel opponent `stochastic=False`), so the
  matchup is best-vs-best and `win_rate_vs_pool` / the snapshot ELO reflect real skill (≈50% vs a
  recent self, ramping with sentinel age). It's eval-only — TRAINING opponents stay stochastic — and
  it auto-lowers `--promote-threshold` to `0.55` (else the handicap-free pool win rate never clears
  the 0.65 gate and the pool freezes). Default off so the live metric stays continuous until opted in.
- **Opponent snapshots are version-checked.** They load via `load_model_snapshot` (not a raw
  `MaskablePPO.load`), and `SnapshotPool` writes a shared `model_config.json` next to its
  snapshots, so an arch-mismatched snapshot fails with a clean `ModelVersionError` instead of
  loading mismatched weights.
- **The opponent RE-DECIDES on a stale decision; the trainee crashes** — split by who *owns* the
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
    stale trainee decision **crashes** (`gen3_env`, no fallback): acting on it would corrupt its
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
- **`--debug --self-play` exercises the real path** (seed → pool eval → promotion) on a fast
  eval cadence, so a CPU smoke against a `9XXX` server validates the wiring without disrupting
  the `:8001` training server. `selfplay_opponent_fuzz_test.py` covers the opponent load + legal
  play (both modes) + version check in-process via the local bridge (no server).

## Stable (cross-run) opponents (`--stable-opponents`, `fixed_opponent_pool.py`)

Load a frozen model from **another, already-finished run** as a **fixed opponent** — measured
against in eval AND (under `--self-play`) played against in training. Design:
`designs/ai_v5/design_stable_opponents.md`.

**Training-mix participation (Stage 2) — "tossed in like a sentinel, becomes a bot when mastered":**
a stable opponent rides the *existing* pool-vs-heuristic split in `MaskableAgentWrapper`
(`wrappers.py`), no new source-model abstraction:
- **CHALLENGE bucket** (the self-play pool branch, competence-gated by `self_play_fraction`): the
  pool gets the BULK; un-mastered stable opponents share a **capped minority slice**
  (`STABLE_CHALLENGE_SHARE` = 0.20 in `wrappers.py`), so a single fixed opponent can never dominate
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
- The stable-opponent players are **built once per worker** (`load_foreign_opponent` in the env
  factory), so no per-episode reload; each plays **stochastic** at `--stable-opponent-temp` in
  TRAINING but **greedy (temp 0)** in EVAL (a clean yardstick).
- **Surfaced in the launcher Events panel** (via `emit`, like the `[SELFPLAY]` startup lines): a
  `🐴 [STABLE] N cross-run opponent(s): ext_<run> — eval greedy; training ≤<share> of self-play until
  mastered (win_rate ≥ <wr>)` line at startup (and a `🏇 [SELFPLAY] Mastered stable opponent(s) …`
  line on the challenge→floor flip), and each eval-summary event gains a `stable <pct>%` field. (Per-opponent `eval/win_rate_vs_ext_<run>` also rides the normal eval Metrics table.)
- **Distillation interaction:** under `--distill-opponents` the pool flips to 100% cheap distilled
  models (all-or-nothing — one full-model worker straggles and gates the per-step barrier). A full
  foreign stable opponent would re-introduce that straggler, so stable opponents drop OUT of the
  training mix entirely while distill is active (eval-only that period); they re-enter when distill
  is off. (`_pick_challenge_opponent` / `_pick_floor_opponent` gate on `self._distill_active`.)

- **CLI:** simplest form is just the run dir — `--stable-opponents models/ai_v5_5_popart_N_0607`;
  the opponent is **labelled by the run-dir name** (`ext_ai_v5_5_popart_N_0607`, derived
  `best_model`/`snapshots`-aware so a direct `…/best_model/best_model.zip` path still yields the run
  name, not `best_model`). Optional per-entry suffixes: `@<step>` (a specific checkpoint; default
  `best_model`), `:<name>` (rename). **Per-opponent weights (`=<weight>`) are rejected** with a clear
  message (not supported). Knobs: `--stable-opponent-temp` (default 1.0 — the *training* play
  temperature; eval is always greedy) and `--stable-opponent-mastered-wr` (default 0.80 — the
  challenge→floor flip). Parsed + resolved at startup by `fixed_opponent_pool.resolve_stable_opponents`.
- **Compatibility = the OBSERVATION FAMILY only** (two axes: obs family vs model family — see the
  design §3). The gate is **same `arch_signature`** (`ModelVersion.check_opponent_compatible`,
  the obs-family proxy); a mismatch is a **startup FATAL** (`[StableOpponent] FATAL` →
  `TrainExitCode.FATAL_CONFIG`, surfaced to the TUI, no restart). Loaded inference-only via
  `snapshot.load_foreign_opponent` (`env=None`), which **skips `check_compatible`** — so
  `use_popart`/`vf_coef`/reward differences (irrelevant to an opponent's forward, which never reads
  the value head) don't block it. The example `models/ai_v5_5_popart_N_0607` shares HEAD's arch, so
  it loads despite being PopArt-on.
- **Label namespace `ext_<run>`** — underscore separator (NOT `ext:`) so the emitted metric tags are
  **uniform** with the rest (`eval/win_rate_vs_ext_<run>`, like `eval/win_rate_vs_sentinel_0`), no
  colons in TensorBoard. `is_external` (`startswith("ext_")`) keeps them out of the bot aggregates.
  Both eval callbacks (`PerOpponentEvalCallback` + `SelfPlayCallback`) add the `ext_` labels as
  `FIXED` `EvalItem`s (so they shard + ride the same plan); the worker's `_play_unit` FIXED branch
  (`eval_worker.py`) plays the **greedy trainee vs the greedy stable opponent** (a clean yardstick).
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
- **Tests:** `fixed_opponent_pool_test.py` (parse + resolve + the arch FATAL gate),
  `snapshot_test.py::*opponent*/*foreign*` (the loader + `check_opponent_compatible`), and the
  end-to-end `stable_opponent_fuzz_test.py` (bridge, no server — resolve + arch FATAL + foreign
  load + legal stochastic play).

## ELO / skill rating (`elo.py`, `bot_elo_calibration.py`, `main.elo`)

Once training is mostly self-play **pool play**, win-rate stops being legible: the promotion
gate only promotes when `win_rate_vs_pool > promote_threshold` and the pool is a *sliding window
of recent selves*, so `win_rate_vs_pool` is a treadmill pinned near 50-65% **by construction** —
it cannot trend up however much the model improves; `win_rate_vs_bots` saturates near 100%. The
ELO subsystem gives a single **absolute** number that genuinely rises with skill, anchored to the
fixed bots.

- **No new battles.** Every eval cycle already plays the trainee (greedy) vs all 9 bots and vs
  up to 5 pool sentinels, `EVAL_GAMES` each — a full tournament-matrix row. `record_elo`
  (`eval_callback.py`, shared by BOTH callbacks) appends that row to an **append-only
  `<run>/eval_results.jsonl`** (`snapshot.append_eval_result_row`) — the canonical, restart-safe
  source of truth, distinct from the overwritten `metadata.json:latest_eval`.
- **The model = anchored Bradley-Terry** (`elo.fit_elo`): `P(i beats j)=σ((Rᵢ−Rⱼ)·ln10/400)`,
  fit in **batch** by penalized MLE (weak Gaussian prior keeps 100-0 records finite), SE from the
  inverse Hessian. Each bot is a player `bot:<name>`, each snapshot `snap:<step>` — a snapshot is
  the SAME player whether it appears as a cycle's trainee or later as a sentinel (unified by
  step), which links the whole ladder. Batch-BT (not online K-factor Elo) is drift-free and
  re-runnable; the fit is a few Newton steps over ~tens of players. **Not Glicko-2**: its
  volatility models skill drift, but snapshots are *frozen* — the drift is the *sequence* of
  snapshots (the ELO-vs-step curve); the per-player uncertainty (Glicko's valuable part) is the
  Hessian SE.
- **Anchor = a precomputed bot-vs-bot round-robin.** `python -m agents.training.bot_elo_calibration`
  plays all 36 bot pairs toward `--target-games` (default 5000) **in-process via the bridge — no
  server** (safe alongside a live run; it does use CPU — throttle with `--concurrency`), fits BT
  (`elo.fit_pairwise`, `random` pinned at `base`=1000), and writes the anchor. **Artifact split:**
  the immutable bot anchor (ratings, SEs, the 9×9 win-matrix, a non-transitivity `fit_quality`) is
  the only runtime input, so it lives in **`data/gen3_bot_elo_anchors.json`**; the raw game-count
  **store** (resume state) and the **heatmap** PNG are calibration provenance/viz, so they live with
  the ELO design work under **`designs/ai_v5/elo_calibration/`** (override with `--games-store` /
  `--heatmap`). The
  live/offline fits then **pin all 9 bots** to those high-confidence ratings and fit only
  snapshots — so a snapshot is well-grounded from its first cycle, and because the anchor is
  identical across runs, **snapshot ELOs are comparable run-to-run**. **Regenerate when bot logic
  changes** (the json records `git_hash` + date). Graceful fallback when the file is absent:
  `random` pinned at `base`, other bots float (rank/trend preserved, scale not cross-run-stable).
  Bots build once and are reused across pairs (`reset_battles` between) — building warms the data
  singletons (~4.5 s each), so per-pair rebuilds dominated cost; the full 5000-game job is a
  many-hour, run-overnight one-time cost.
- **Live (each eval cycle).** `record_elo` refits and records `eval/elo` + `eval/elo_ci` (95% CI
  half-width) to TensorBoard + the TUI dict, and stamps `elo`/`elo_ci` into `metadata.json:
  latest_eval` (so the resume-republish path shows ELO immediately after a restart — the saved
  headline is authoritative; and if a resumed checkpoint predates the `elo` field,
  `replay_last_eval_to_tui` **fits** the saved block's win rates via `elo.fit_from_block` to recover
  both the headline and each opponent's ELO, so the badge never blanks for a full cadence). The
  launcher
  surfaces a `🏅 ELO 1532 ±40` badge (`app.py::_elo_badge`) + an `elo` column in the eval panel:
  the model's rating on the `all` row, and each opponent's anchored ELO on its row
  (`_record_opponent_elos` records `eval/elo_vs_<bot>` + positional `eval/elo_vs_sentinel_<i>` to
  the TUI). The live number is the best estimate from data SO FAR (batch-BT is global → early
  points retro-adjust; the single-cycle per-sentinel ELO is rough — only the trainee is
  bot-anchored each cycle); the offline CLI re-fits canonically over the full per-snapshot history.
- **Offline (`python -m main.elo <run_dir>`).** Loads results (`--source auto|log|tb|meta` —
  `tb` **backfills an already-running run straight from TensorBoard, zero training change**), fits,
  and prints a ranked ladder + writes `elo_ratings.json` + an Elo-vs-step `elo_curve.png` (CI band
  + bot anchor lines). `--out` defaults to `<run>/elo/`; point elsewhere to analyze a LIVE run
  without writing into it.
- **Caveat (acceptable, noted in code):** by default the trainee is greedy but the sentinels are
  stochastic@temp, so a snapshot's rating blends greedy strength (when it's the cycle's trainee)
  with stochastic strength (when it's a later sentinel) — a roughly uniform shift that preserves the
  trend, but it does mean the same snapshot is scored in two regimes. **`--eval-sentinel-greedy`
  removes this** — sentinels play greedy too, so every snapshot is scored greedy in both roles and
  the ELO ladder is internally consistent (at the cost of a one-time scale shift vs prior cycles;
  the bot-anchored scale is preserved since trainee-vs-bot records are unchanged). Tests:
  `elo_test.py` (synthetic-ladder recovery, anchoring, perfect-score, loaders, `fit_pairwise`).

## Opponent distillation (`--distill-opponents`, off by default)

Distils the frozen self-play opponents into a **cheaper network** (the opponent forward is ~70% of
worker CPU) for faster rollouts — implemented in **`distill/` (has its own CLAUDE.md)**. The governing
constraint is the per-step barrier: distillation is **all-or-nothing** (one full-opponent worker
straggles and gates the batch), so the pool is only ever 100% distilled or 100% full. A single
idempotent **reconcile loop** (`DistilledOpponentManager`, run by `SelfPlayCallback` each eval + on a
throttle) keeps the on-disk distilled set in sync with the pool — **backfill on enable ≡ steady-state**,
no-op when nothing's missing — spawning the `distill/worker.py` subprocess per snapshot (gate =
fidelity + head-to-head). Distilled artifacts + their gate manifests live in `models/<run>/distilled/`
(the manifest is the per-snapshot source of truth; `summary.json` gets only a re-publish block);
cleanup is automatic via the reconcile's window-eviction. The env's `MaskableAgentWrapper` does the
atomic full↔distilled opponent switch (`set_distill_active`). **Observability:** `_reconcile_distill`
records five `distill/*` scalars (frac/all_distilled/ready/running/exhausted) to TensorBoard + the
launcher dashboard, and emits launcher **Events** for each gate result (deployed/escalated/exhausted
with h2h + speedup), the atomic full↔100%-distilled switch, and backfill spawns — surfaced in the TUI
as a `⚗ distilled 100%`/`⚗ distilling N%` badge + a `distill/*` metrics block + Events lines (zero
footprint when off). **Full design: `designs/ai_v5/distill_integration.md`
(§8 all-or-nothing, §7 restart resilience); module map: `src/agents/training/distill/CLAUDE.md`.**

## Rollout collection: sync barrier vs `--async-rollout` (`async_vec_env.py`)

The default `SubprocVecEnv.step()` is a **per-step barrier** — the trainer waits for the slowest of
N env workers every step, so a slow battle turn / heavy opponent forward / oversubscription jitter
stalls the whole batch and the GPU policy-forward never overlaps CPU env-stepping. `--async-rollout`
swaps in **`AsyncSubprocVecEnv`** (per-env `send_step`/`poll_ready`/`recv_step` over the pipes +
**drain-safe `env_method`** — the eval callback's `set_self_play_target`/`set_distill_active`/
`opponent_default_stats` fire mid-collection, so the override stashes in-flight step results before
any barrier RPC to avoid a pipe desync) and **`collect_rollouts_async`**, dispatched by
`InstrumentedMaskablePPO.collect_rollouts` when `model._async_rollout` is set.

The collector keeps every worker continuously in-flight, batch-forwards whichever envs are READY
(dynamic batch), and writes each env's transition into **its own buffer column**
(`MaskableDictRolloutBuffer`); collection ends when every column has `n_steps`. It is **exactly
on-policy** — PPO freezes the policy during collection, so this is a *scheduling* change (overlap
forward with stepping, drop the max-latency barrier), NOT an APPO-style algorithm change. Bookkeeping
(`num_timesteps`, GH-#633 timeout bootstrap, `_update_info_buffer`, `_last_*` carry-over, per-column
GAE) mirrors the stock loop exactly. The per-decision **mask rides in the Dict obs**
(`obs["action_mask"]`, = `last_ctx.mask`), so no per-env `env_method` and no wrapper change.

**Measured FPS (bridge, GPU forward, steady-state, heuristic opponents):** +20% at `--n-envs 16`;
**+14% at the production `--n-envs 64` (1489→1695)**; `--async-rollout --n-envs 32` matches `sync@64`
FPS with half the envs (≈half the env/bridge RAM). Off by default (stock `SubprocVecEnv`), ignored
under `--debug`. Compounds with distillation (async attacks the barrier; distill attacks the per-step
opponent CPU). Caveat: benchmarked with heuristic opponents — re-bench under `--self-play` for the
production-regime number. Full design + benchmark table: `designs/ai_v5/design_async_rollout.md`.

## Gradient-balance + value-scale diagnostics (`grad_balance.py`)

The dual-head extractor shares ONE transformer trunk between the policy and value heads
(`src/agents/model/CLAUDE.md`); both losses' gradients compete there. When the value loss
dominates (large / unclipped, big-return scale) it **swamps the trunk** and the policy barely
updates — visible before only *indirectly* as suppressed `train/approx_kl` + `train/clip_fraction`
while `train/explained_variance` races ahead. `InstrumentedMaskablePPO.train()` now measures it
**directly** via the pure helpers in `grad_balance.py` (no SB3 / logging coupling → unit-tested in
`grad_balance_test.py`), recorded once per `train()` call through the standard logger → TensorBoard
**and** the launcher TUI (the new scalars ride the generic `MetricsExporterCallback` →
`ipc.send_metrics` path with zero extra wiring; ordering/labels live in `launcher/format.py`).

- **Gradient balance — the value-vs-policy *pull* on the shared trunk.** Sampled on the first
  minibatch (graph alive) by two **read-only** `autograd.grad` probes (`retain_graph=True`, so the
  real `loss.backward()` is unaffected) against the shared-trunk params. "Shared" =
  `SHARED_TRUNK_PHASES = {embeddings, pokemon_encoder, team_transformer, assembler}` (the allow-list
  is the single source of truth), which **excludes** `cls_pool` (head-private `our_cls`/`their_cls`/
  `value_cls` queries) and both projection heads — only *truly contested* params count.
  - `grad/value_share` = `‖g_value‖ / (‖g_value‖+‖g_policy‖)` (~0.5 balanced, →1 value swamps).
    Weighted by the live `vf_coef`/`ent_coef`, so it is the **tuning target**: dial `vf_coef`
    (`--vf-coef`, default 0.5) so it sits near ~0.5. `vf_coef` is **fixed per run** — recorded in
    `model_config.json` and FATAL to change on resume (it rescales this very gradient mid-run); tune
    it on a fresh run. See `src/agents/model/CLAUDE.md` → resume-immutable training hparams.
  - `grad/value_policy_logratio` = `log10(‖g_value‖/‖g_policy‖)` — the **same** imbalance as
    `value_share` but on a *linear, non-saturating* scale (0 = balanced, >0 = value dominates, e.g.
    ≈+1.8 at a 66:1 swamp). Prefer it for **watching a fix land**: `value_share` is pinned near 1 in
    the swamped regime (0.985 / 0.99 / 0.995 are 66× / 99× / 199× but look alike), so PopArt /
    a `vf_coef` change crawls there while the log-ratio moves linearly toward 0.
  - `grad/policy_value_cosine` — scale-invariant (hence `vf_coef`-independent) structural-conflict
    signal: <0 ⟹ the heads pull the trunk in opposing directions.
  - `grad/policy_norm_shared` / `grad/value_norm_shared` — the weighted norms, for absolute context.
- **Value scale — PopArt prep.** From the full rollout buffer: `train/return_mean` / `train/return_std`
  / `train/return_abs_max` (exactly the `(μ, σ)` + tail an adaptive return normalizer / PopArt's ART
  half tracks) and `train/value_pred_std` (the value head's actual output spread). Watch these to SEE
  the non-stationary value-scale drift (reward annealing / policy improvement) that a static `vf_coef`
  cannot follow. Plus `train/grad_norm` (pre-clip total grad norm, mean over minibatches → grad-clip
  activity).

Cost: **2 extra partial backward passes on ONE minibatch per `train()` call** (negligible vs the
`n_epochs × n_minibatches` the loop already runs) + trivial NumPy stats. The probe is a **no-op**
(records nothing) when `shared_trunk_parameters` finds no matching modules (a non-Gen3 policy). **Why
it exists:** to prepare for **reducing `vf_coef`** and **adding return normalization (PopArt)** — both
target the value→trunk pressure, which can now be tuned to a number instead of inferred. (The
`+INSTRUMENTATION` markers in `instrumented_ppo.py` flag the added lines; the upstream-drift hash check
is unaffected since it hashes only `sb3_contrib.MaskablePPO.train`.)

## PopArt value-target normalization (`--use-popart`)

The fix for the swamping the diagnostics above reveal. `train()` reads `self.popart =
getattr(self.policy, "popart", None)` (built by the policy when `--use-popart`; see
`src/agents/model/CLAUDE.md` → PopArt for the math + version-checking). When present: once per
`train()` (before the epochs) `popart.update(self.rollout_buffer.returns, self.policy.value_net)`
advances the running `(mu, sigma)` **and** POP-rescales `value_net`; the value loss then becomes
`MSE(popart.normalize(returns), popart.normalize(values))` — the **normalized**-space loss, so the
value gradient into the shared trunk drops by ≈`sigma²` and stops swamping the policy. The policy's
value sites de-normalize, so `rollout_buffer.values` / GAE / advantages stay real-unit — the policy
path is untouched. **`--use-popart` requires an explicit `--clip-range-vf none`** (errors otherwise —
self-documenting config; clipping is unnecessary with value normalization, and would clip in
un-normalized units). New diagnostics ride the same generic metrics path:
`popart/mu`, `popart/sigma` (watch them track `train/return_mean`/`return_std`),
`popart/value_weight_norm` (POP keeps it bounded). Under PopArt `train/value_loss` is the normalized
loss (≈O(1)) and `grad/value_share` should fall from ≈1.0 toward ~0.4 — the live confirmation it
worked.

## Process liveness guards (`watchdog.py`)

Two daemon-thread watchdogs keep a hung/abandoned run from lingering:

- **`start_subprocess_watchdog`** — for the `SubprocVecEnv` path. A crashed worker leaves the
  parent blocked on a pipe `recv` forever; this thread polls `processes` and `os._exit(1)`s the
  moment a worker dies with a nonzero exitcode. Started *after* env construction (and, in
  self-play, after `_maybe_engage_self_play` rebuilds the env), right before `learn()`. It is a
  **no-op on the `--debug` DummyVecEnv path** (no worker processes to watch).
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
