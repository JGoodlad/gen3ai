# Training — team curriculum — team-side PFSP and per-team win-rate tracking

Moved verbatim out of `src/agents/training/CLAUDE.md` on **2026-09-08**, in the second pass of the
topic split (that leaf was 3,183 lines / 264 KB, loaded in full for any session touching the
training package). Each section below is unchanged, including its dated measurements.

`src/agents/training/CLAUDE.md` keeps the heading and a short summary of each, and points here.
**This file is the owner of the detail.**

---

## Team-side PFSP (`--team-pfsp`, `team_pfsp_callback.py`)

The TEAM-axis complement to the opponent-side `--pfsp-scale`: bias the TRAINEE's team sampling toward
the pool teams it is weakest on, so training spends gradient where the win-rate says there's headroom
instead of uniformly over ~700 pool teams (the documented "uniform team sampling = headroom" gap).
Four modes: **`off`** (default → byte-identical), **`measure`** (TRACK + persist the per-team
self-play win-rate WITHOUT biasing sampling — pure observability), **`var`** (measure + bias,
symmetric variance), **`onesided`** (measure + bias, losing side held at MAX).

- **Variance weighting + cap + floor.** For pool team `i` the weight is `raw_i = --team-pfsp-floor +
  w(p_i)` where `p_i` is the team's self-play win-rate EMA (seed 0.5 → an unmeasured team gets the
  MAX weight → explored), then capped `w_i = min(raw_i, --team-pfsp-cap·mean(raw))` (no team is
  sampled more than `cap`× the uniform share — the over-representation bound). **`var`**: `w(p) =
  p·(1−p)` — peaks at 50% and decays to the floor at BOTH extremes, so it self-ignores both the teams
  we crush AND the truly-lost teams. **`onesided`** (owner-requested, the z_arch/FiLM companion):
  `w(p) = 0.25 for p < 0.5, else p·(1−p)` (continuous at 0.5) — every sub-50% team stays MAXIMALLY
  sampled and only mastery retires a team, because under the conditioning hypothesis the weak-team
  tail is exactly the learnable headroom (the amortization gap): "truly lost" is the claim under
  test, not a sampling prior to bake in. The floor keeps nothing fully starved either way.
  `compute_team_pfsp_weights` is the pure, unit-tested math.
- **Team-blocked episodes (`--team-block-episodes`, default 1 = off, byte-identical).** Each env
  holds its drawn TRAINEE team for N consecutive episodes before redrawing
  (`Gen3Teambuilder.set_block_episodes`; the WHOLE draw is held — bias branch, PFSP weights,
  tracking index — so weights apply at redraw and outcomes attribute to the blocked team for the
  whole block; each SubprocVecEnv worker unpickles its own builder copy ⇒ blocks are per-env). The
  per-team gradient-DENSITY counter to the sample starvation the retired FiLM group measured
  (`film/noise_scale` ran ≈ 8–9× the batch before the v78 zarch deletion took that metric with it;
  the DENSITY argument stands on its own): per-episode redraw gives ~700 teams × ~4 episodes
  (~140 decisions) per rollout;
  at ~64 (≈ `n_steps`/ep_len — the phase-transition value) each env carries ONE team per rollout at
  ~2k decisions (~15× density) AND the block spans an update boundary, so the env replays the team
  right after its gradient landed (the mini-exploiter learn-and-retest loop — the piece of the
  exploiter regime per-episode redraw never provides). Acceptance: the fixed-matchup ablation
  probe's intact-vs-ablated gap widening. Trainee side
  only (opponent draws stay per-episode); training-only, NOT version-locked, resume-forwarded.
- **Self-play only, pool teams only.** The per-team win-rate is measured ONLY on self-play POOL battles
  (bots wash the signal out — we win ~0.99 vs bots): `MaskableAgentWrapper.step` records the outcome to
  the trainee's `Gen3Teambuilder` (`self.env.agent1._team`) only when `self.opponent is
  self._pool_player`. A bias/distill-pinned team (the `--distill-team-bias` branch) yields
  `_last_pool_idx=None` → its battle is never tracked (those teams get fixed exposure via the bias, not
  the win-rate weighting).
- **Centralized aggregation (NOT per-worker — ~700 teams makes a single worker's counts too sparse; NOT
  info-dict threading — that breaks under `--async-rollout`).** Each worker's teambuilder accumulates
  LOCAL windowed `(wins, games)` per pool team; `TeamPFSPCallback` every `update_every` (3) rollouts
  PULLs them from all workers via `env_method("drain_team_pfsp_counts")` (drain-zeroes each window), SUMs
  by pool index, EMA-smooths a global per-team win-rate, computes the capped weights, and PUSHes them
  back via `env_method("set_team_pfsp_weights", w)` → the teambuilder samples with
  `random.choices(weights=…)`.
- **Auditability + GIGO guard.** Each pool team carries a `team_sha` fingerprint
  (`sha1(team_str.strip())[:10]` — the SAME convention as `matchup_spec.pin_sha` / the archetype
  artifact, so a key JOINS every provenance record). The callback pulls them ONCE
  (`env_method("get_team_pfsp_keys")`) and verifies the per-INDEX team identity is IDENTICAL across
  every worker (**same pool SIZE ≠ same pool ORDER** — a diverged order would silently mis-attribute
  win-rates, which the cheap per-cycle size-only belt can't catch), then logs the weakest measured
  teams by `sha@win-rate` so the weighting is inspectable (which teams/archetypes the budget
  concentrates on), not an anonymous min/max scalar. Metrics
  `team_pfsp/{min_wr,max_wr,n_measured,weight_spread}`.
- **Persisted artifacts (both `measure` and `var`) → offline "which exploiter next".** Each update the
  callback writes to the run dir: `team_winrates.json` (the latest snapshot — per-team `{sha, win_rate,
  games, archetype}` sorted WEAKEST-FIRST, atomic-replaced; the weakest teams = candidate exploiter
  targets, and `archetype` is joined from `gen3_team_archetypes.json` via `team_sha` so it reads
  "weakest = stall-class") and an appended `team_winrates_history.jsonl` row `{step, wr:{sha:wr}}` (so
  the per-team win-rate is trackable OVER TIME offline — trends + noise, not just the latest). `measure`
  gives this signal on ANY self-play run without changing the team distribution.
- **Training-only, not version-locked.** Threaded into the TRAINEE teambuilder only (both the
  `matchup.trainee_teams.build` and the distill `Gen3Teambuilder` paths); the opponent builder is
  untouched. Registered ONLY when `--team-pfsp != off` (off → no callback, no `env_method`, exact-legacy
  `random.choice` → byte-identical); `var` pushes weights, `measure` never does. Forward it like
  `--pfsp-scale` on resume; no `model_config`/`ModelVersion` entry.
- **Tests.** `utils/teambuilder_test.py` (off==uniform RNG-identical, weighted sampling, record/drain,
  the cap+floor weight math), `team_pfsp_callback_test.py` (cross-worker aggregation, the pool-size GIGO
  guard, the `update_every` throttle, None-worker filtering).

## Per-team win-rate tracking (`--team-wr-tracking`, DEFAULT ON, `team_winrate_callback.py`)

A first-class running record of how the trainee does **piloting each team**, keyed by `team_sha`.
The training loop always knew which team an episode piloted and how it ended; nothing kept the
record, so the three flywheel consumers that need it — the deficit thermostat, **headroom
capture's denominator**, and slice-curation evidence — each had to be a scratch script. This is
**instrumentation only: no prioritization consumer ships with it**, by design.

⚠️ **THE CONFOUND, and it is written into the artifact rather than only into this file.** A raw
per-team win rate conflates **PILOT COMPETENCE with TEAM STRENGTH** (the ai_v8 team-PFSP finding:
team-PFSP win rate was confounded by team strength). "Our win rate with team T is low" does not
mean "we pilot T badly". Anything that spends budget on this signal must first normalize against a
**team-strength baseline** — e.g. T's pool-average win rate under a reference pilot. The artifact
carries that sentence in its `notes` field so it travels with the numbers, plus the reminder to
read `by_class`: a pre-self-play curriculum phase is ~all `bot` episodes, where every team reads
~0.99.

- **The seam is an `env_method` PULL, not an info-dict thread — and that is the async decision.**
  Each worker's `Gen3Teambuilder` accumulates a windowed per-team, per-opponent-class count
  (`record_team_wr_outcome`), fed by `MaskableAgentWrapper._maybe_record_team_wr` at the terminal
  step beside the existing `win_outcome` capture; `TeamWinRateCallback._on_rollout_end` drains
  every worker (`drain_team_wr_counts`) at a rollout boundary. **This works identically under
  `SubprocVecEnv` and `--async-rollout`** because `AsyncSubprocVecEnv.env_method` is drain-safe (it
  stashes in-flight step results before the barrier RPC), whereas an info-dict route would have to
  know which buffer ROW a terminal landed on — knowledge only the async collector has, which is why
  the team-PFSP precedent avoided that route for the same reason.
  `test_aggregation_reads_env_method_and_never_the_info_dicts` pins it by feeding the callback a
  deliberately contradictory `self.locals["infos"]` and asserting the result ignores it.
- **The default uniform draw stays RNG-identical.** With `--team-pfsp off` (the default)
  `_draw_team` is `random.choice(self.packed_teams)`, which returns the team and not its index. The
  index is recovered by a **reverse dict lookup** (`_pool_index_by_packed`, built at construction),
  never by re-drawing it — so the byte-identity baseline is untouched
  (`test_default_uniform_draw_is_rng_identical_with_tracking`). Side effect worth knowing:
  `--team-block-episodes` caches `_last_pool_idx` for the block, which on the default path used to
  be `None`, so a blocked default run can now attribute its whole block to the team it held.
- **Stratified by opponent class** (`MaskableAgentWrapper.OPP_CLASS_*` / `OPP_CLASS_NAMES`), so a
  rate can always be split back out by who it was measured against. A bias/distill-pinned yield
  (`_last_pool_idx is None`) is never attributed to a pool team.
- **NO TensorBoard emission — owner rule** (design_flywheel_tick_tock.md §6b: per-team series
  would be noisy spam; "let's not spam it if the data won't be nice"). Pinned by
  `test_NOTHING_is_emitted_to_tensorboard` — a future "just one scalar" regression fails there.
- **The table rides `metadata.json`** as the top-level `team_win_rates` block (written via
  `snapshot.record_team_win_rates`, carried forward across checkpoints by `save_model_snapshot`
  exactly like `latest_eval` — one artifact per run holding per-team AND per-opponent records
  side by side):
  `{step, updated_at, n_teams_seen, n_games, opp_classes, notes, teams: {sha: {n, wins, wr,
  archetype, by_class}}}`. **RAW COUNTS, not a smoothed rate** — headroom capture needs a
  denominator, which is exactly what team-PFSP's EMA throws away. `archetype` is joined via
  `load_team_archetypes` on the same `team_sha`. **Restart-safe by load-and-continue**, and keyed
  by sha rather than pool index so a pool that was reordered or resized between runs still joins
  (`test_reload_is_keyed_by_sha_so_a_REORDERED_pool_still_joins`). A corrupt file starts fresh.
- **GIGO guard, throwing.** Counts arrive per pool INDEX and are keyed to a sha by the worker's own
  key list; if any worker's list disagrees the callback **raises**. Same pool SIZE is not the same
  pool ORDER, and a diverged order would attribute every per-team number to the wrong team.
- **Deliberately NOT coupled to `--team-pfsp`, and the overlap is real enough to state.**
  `--team-pfsp measure` also tracks a per-team win rate and also writes an archetype-joined
  `team_winrates.json`. Four differences make it unusable as this instrument: it is **off by
  default**; it measures **self-play POOL battles only** (bots wash out its weighting signal), so a
  pre-self-play generation records nothing; it keys per pool **INDEX** with the sha only for an
  audit line; and it stores an **EMA rate**, not counts. The two share the builder's "which team
  did I just yield" draw index (`_last_pool_idx`) and **nothing else** — separate counter tables,
  separate accessors, separate artifacts, deliberately differently-named files
  (`team_win_rates.json` vs `team_winrates.json`). If the owner later wants one tracker,
  consolidating team-PFSP's `measure` mode onto this table is the direction, not the reverse.
- **Flag class: training-runtime, like `--team-pfsp`.** Never reaches the extractor, scales no loss,
  changes no weight shape ⇒ **no `ARCH_SIGNATURE` bump, not in `model_config.json`/`ModelVersion`,
  not in `check_compatible`, and deliberately not in `agents/model/flag_registry.py`** (that
  registry's scope is extractor architecture toggles — the `--td-aux-coef` /
  `--intent-label-bot-weight` precedent, which are recorded on `ModelVersion` only because they
  scale a loss and want flagless-resume inheritance; this one does neither). Forwarded verbatim by
  the launcher like any non-launcher flag. `--no-team-wr-tracking` opts out (no callback, no
  `env_method`, the wrapper hook returns immediately).
- **Verified end to end** by a `--debug --steps 4000` CPU smoke: **96 teams / 103 games** recorded,
  archetypes joined (`semi_stall`, `balance`, `hyper_offense`), `by_class` correctly all-`bot` on a
  fresh run, the `notes` caveat present, and `teams/n_teams_seen` / `teams/n_games` on the TB
  event file. The four `wr_*` scalars need a team past the 10-game floor, which a 719-team uniform
  pool does not reach in 4000 steps — a `--trainee-team`-pinned smoke exercises them, and all six
  keys are pinned numerically by `test_sparse_tb_keys_are_summaries_not_one_series_per_team`.
- **Tests.** `team_winrate_callback_test.py` (29): the `team_sha` convention agreement with
  `team_archetypes.team_sha` incl. strip-normalization, the RNG-identity claim, the builder
  accumulator + drain-zeroing + bias-yield exclusion + PFSP-table independence, the wrapper hook and
  its off path, the callback's running math across workers AND windows, per-class restriction,
  `min_games`, the `update_every` throttle, None-worker filtering, the throwing order guard, the TB
  key set with hand-computed values, the artifact shape + confound note, the archetype join and its
  missing-artifact fallback, restart reload incl. the reordered-pool case and a corrupt file, and
  the `env_method`-not-infos seam claim. Plus `utils/teambuilder_test.py` (the off-path RNG identity
  now also asserts the index resolves while PFSP still ignores it).

