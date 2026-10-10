# Training — team curriculum — per-team win-rate tracking and team-blocked episodes

Moved verbatim out of `src/agents/training/CLAUDE.md` on **2026-09-08**, in the second pass of the
topic split (that leaf was 3,183 lines / 264 KB, loaded in full for any session touching the
training package). The team-PFSP half was deleted in deletion pass L4 (2026-10-02) and is a history note
below; the other sections keep their dated measurements.

`src/agents/training/CLAUDE.md` keeps the heading and a short summary of each, and points here.
**This file is the owner of the detail.**

---

## Team-side PFSP (DELETED — history)

**Deleted 2026-10-02, deletion pass L4** (owner-approved, decision D1 "delete, port none"; config v134):
`--team-pfsp {off,measure,var,onesided}`, its cap / floor knobs, `TeamPFSPCallback`, the
`team_pfsp` supply lever (`lever_supply.LEVERS` is now `{self_play_pool, pfsp, fork}`) and the
weighting / accumulator methods of `Gen3Teambuilder`. It was the TEAM-axis complement to the
opponent-side `--pfsp-scale`: every update it pulled windowed per-pool-team self-play win rates from
all workers, EMA-smoothed them and pushed variance-style weights (`p(1-p)`, or a one-sided form that kept
every sub-50% team at the maximum) back to the trainee's teambuilder, capped at a multiple of the uniform
share, with a `team_winrates.json` / `team_winrates_history.jsonl` artifact and `team_pfsp/*` scalars.

**Why it is gone.** It was OFF in the production recipe, its arm was probe P (ledger 2026-08-30,
*PROBE P DISPATCHED*), and the ladder campaign replaces team-side sampling with ONE PLR (prioritized
level replay) sampler, which is a new build, not this. Flag-by-flag citations: `designs/deleted_flags.md`;
the old text: `git show cbd20111:designs/training/team_curriculum.md`.

**What stays.** The two instruments below: per-team win-rate TRACKING (default on) and team-blocked
episodes. The team-PFSP finding that survives is the CONFOUND they both carry: a per-team win rate
conflates pilot competence with team strength (see the tracking section).

## Team-blocked episodes (`--team-block-episodes`)

**`--team-block-episodes` (default 1 = off, byte-identical).** Each env
holds its drawn TRAINEE team for N consecutive episodes before redrawing
(`Gen3Teambuilder.set_block_episodes`; the WHOLE draw is held — bias branch and
tracking index — so outcomes attribute to the blocked team for the
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

## Per-team win-rate tracking (`--team-wr-tracking`, DEFAULT ON, `team_winrate_callback.py`)

A first-class running record of how the trainee does **piloting each team**, keyed by `team_sha`.
The training loop always knew which team an episode piloted and how it ended; nothing kept the
record, so the three flywheel consumers that need it — the deficit thermostat, **headroom
capture's denominator**, and slice-curation evidence — each had to be a scratch script. This is
**instrumentation only: no prioritization consumer ships with it**, by design.

⚠️ **THE CONFOUND, and it is written into the artifact rather than only into this file.** A raw
per-team win rate conflates **PILOT COMPETENCE with TEAM STRENGTH** (the ai_v8 team-PFSP finding:
team-PFSP win rate was confounded by team strength; team-PFSP itself is deleted, see above). "Our win rate with team T is low" does not
mean "we pilot T badly". Anything that spends budget on this signal must first normalize against a
**team-strength baseline** — e.g. T's pool-average win rate under a reference pilot. The artifact
carries that sentence in its `notes` field so it travels with the numbers, plus the reminder to
read `by_class`: a pre-self-play curriculum phase is ~all `bot` episodes, where every team reads
~0.99.

- **The seam is an `env_method` PULL, not an info-dict thread — and that is the async decision.**
  Each worker's `Gen3Teambuilder` accumulates a windowed per-team, per-opponent-class count
  (`record_team_wr_outcome`), fed by the env core's terminal step (the deleted Python `MaskableAgentWrapper._maybe_record_team_wr`'s job; `RustVecEnv.SURFACE` serves `drain_team_wr_counts`); `TeamWinRateCallback._on_rollout_end` drains
  every worker (`drain_team_wr_counts`) at a rollout boundary. An info-dict route would have to
  know which buffer ROW a terminal landed on — knowledge the callback does not have, which is why
  the (deleted) team-PFSP callback avoided that route for the same reason.
  `test_aggregation_reads_env_method_and_never_the_info_dicts` pins it by feeding the callback a
  deliberately contradictory `self.locals["infos"]` and asserting the result ignores it.
- **The uniform draw stays RNG-identical.** `_draw_team` is `random.choice(self.packed_teams)`, which
  returns the team and not its index. The index is recovered by a **reverse dict lookup**
  (`_pool_index_by_packed`, built at construction), never by re-drawing it — so the byte-identity
  baseline is untouched
  (`test_default_uniform_draw_is_rng_identical_with_tracking`). Side effect worth knowing:
  `--team-block-episodes` caches `_last_pool_idx` for the block, which on the default path used to
  be `None`, so a blocked default run can now attribute its whole block to the team it held.
- **Stratified by opponent class** (`agents.training.opponent_classes.OPP_CLASS_*` / `OPP_CLASS_NAMES`), so a
  rate can always be split back out by who it was measured against. A bias-pinned yield
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
  denominator, which an EMA rate (what the deleted team-PFSP kept) throws away. `archetype` is joined via
  `load_team_archetypes` on the same `team_sha`. **Restart-safe by load-and-continue**, and keyed
  by sha rather than pool index so a pool that was reordered or resized between runs still joins
  (`test_reload_is_keyed_by_sha_so_a_REORDERED_pool_still_joins`). A corrupt file starts fresh.
- **GIGO guard, throwing.** Counts arrive per pool INDEX and are keyed to a sha by the worker's own
  key list; if any worker's list disagrees the callback **raises**. Same pool SIZE is not the same
  pool ORDER, and a diverged order would attribute every per-team number to the wrong team.
- **History: it was deliberately NOT coupled to team-PFSP's `measure` mode** (off by default, self-play pool battles only, keyed by pool INDEX, an EMA rate rather than counts, different artifact name). Team-PFSP is deleted; this table is the one per-team tracker.
- **Flag class: training-runtime.** Never reaches the extractor, scales no loss,
  changes no weight shape ⇒ **no `ARCH_SIGNATURE` bump, not in `model_config.json`/`ModelVersion`,
  not in `check_compatible`, and deliberately not in `agents/model/flag_registry.py`** (that
  registry's scope is extractor architecture toggles — the `--intent-label-bot-weight`
  precedent, which is recorded on `ModelVersion` only because it
  scales a loss and want flagless-resume inheritance; this one does neither). Forwarded verbatim by
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
  accumulator + drain-zeroing + bias-yield exclusion, the wrapper hook and
  its off path, the callback's running math across workers AND windows, per-class restriction,
  `min_games`, the `update_every` throttle, None-worker filtering, the throwing order guard, the TB
  key set with hand-computed values, the artifact shape + confound note, the archetype join and its
  missing-artifact fallback, restart reload incl. the reordered-pool case and a corrupt file, and
  the `env_method`-not-infos seam claim. Plus `utils/teambuilder_test.py` (the off-path RNG identity
  also asserts the index resolves).

## From the training leaf (moved 2026-10-10)

> These sections headed `src/agents/training/CLAUDE.md` until its 2026-10-10 cleanup; moved here as they
> stood (minus statements verified FALSE). Where an earlier section of this doc says the same in more
> detail, both are current; fix both in the same pass.

### Team curriculum — per-team win-rate tracking (`--team-wr-tracking`) and team blocking (`--team-block-episodes`)

**`--team-wr-tracking`** (DEFAULT ON) is instrumentation only — a running per-`team_sha` record of
wins/games stratified by opponent class, riding `metadata.json`'s `team_win_rates` block.
`--team-block-episodes N` holds each drawn trainee team for N consecutive episodes (per-team gradient
density; 1 = off). **Team-side PFSP (`team-pfsp`, which biased the trainee's team draw toward the pool
teams it was weakest on) was DELETED in deletion pass L4** — `designs/deleted_flags.md` has the flags and
citation.

⚠️ **A raw per-team win rate conflates PILOT COMPETENCE with TEAM STRENGTH** (the ai_v8 team-PFSP
finding). Anything spending budget on this signal must normalize against a team-strength baseline
first; the artifact carries that sentence in its own `notes` field. 🚨 **NO TensorBoard emission**
(owner rule: per-team series are noisy spam), pinned by a test that fails on "just one scalar".
🚨 **Same pool SIZE is not the same pool ORDER** — the tracker verifies per-index team identity
across workers and RAISES on disagreement. Both are training-only, not version-locked, and the
tracker takes an `env_method` PULL rather than an info-dict thread.
**Full detail — in [`designs/training/team_curriculum.md`](team_curriculum.md).**
