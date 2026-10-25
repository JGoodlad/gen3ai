# Training — eval and rating

The detail owner for the eval and rating machinery: the baseline registry, the in-loop eval, the offline meters (untaught,
head-to-head, plateau), the snapshot ladder, and how an ELO is read. `src/agents/training/CLAUDE.md` points here;
[`eval_rules.md`](eval_rules.md) is the rule-sized companion. Dated incidents, per-run numbers and superseded designs were
lifted on 2026-10-25 into `designs/research_state/claude_md_archive/training_eval_and_rating_history.md` (history, not a claim
about the present).

---

## THE BASELINE REGISTRY (`baselines.py` · `designs/baselines.json` · `python -m main.baselines`)

**A baseline is the thing a result is read AGAINST, and `agents.training.baselines` is the ONE accessor over the named
set** (`designs/baselines.json`, `gen3_baselines_registry_v1`): no consumer keeps a hand-copied path or a string literal
for one. Torch-free and offline: it reads JSON, and only `resolve()` touches `models/`.

**Every entry is EXPLICIT** — a `.zip`, a `.json`, or an `@step`, never a bare run directory — so
`gen3_last_snapshot_resolution_v1`'s last-snapshot rule cannot move what a name points at while its run keeps training;
`resolve()` always lands on the `explicit_zip` / `explicit_step` rung (asserted).

```python
from agents.training import baselines
baselines.get("v9_fold_parent").spec        # "ai_v9_59_R2ACTION_0827/final_model.zip"
baselines.resolve("v9_fold_parent")         # through fixed_opponent_pool.resolve_model_ref
baselines.describe("famine_comparator")     # the line every consumer prints
baselines.get("famine_comparator").floor_elo   # 38.0 — the bar travels with its comparator
baselines.protected_files()                 # {run: [rel path]} — the grooming keep-list
```

**Consumers accept a NAME wherever they accept a ref:** `main.untaught_meter`'s `--opponent` / `--config` / `--baseline` /
`--control` (the meter's opponent literal is gone: the engine exposes `default_opponent()`), `main.elo`'s positional run
dir, and `main.tb_curate` (which unions the registry's `tb_curated` list into the curated logdir). **Each prints
`baseline <name> = <run>@<step> (set <date>, <ledger title>)`**: a reader must never have to recognise a path.

🚨 **A NEW OPPONENT IS A RE-MEASUREMENT, NOT A RENAME.** Untaught-meter levels are not comparable across opponents, so
re-pointing a baseline invalidates every banked level measured against the old one. `python -m main.baselines set <name>
<run>/<file>.zip --reason "<ledger entry title>"` rewrites the entry with a freshly computed sha/commit/version and PRINTS
the ledger line to append; it never edits the ledger (append-only, and the WHY is the one field no tool can author).

**Validation is a routine-suite test** (`src/main/baselines_test.py`): every named file exists, every sha matches,
`config_version` / `arch_signature` are re-read from the run's own `model_config.json`, and the `production` entry's
declared CONSTRUCTION matches `designs/production_config.json`. `designs/research_state/measurements/archive_grooming_tiers.py`
reads `protected_files()` so a registry-named checkpoint survives every retention tier.

### A name either LOADS or fails LOUD — `baselines.load` and `BaselineLoadError`

🚨 **Load a baseline with `baselines.load(name)`, never a bare `MaskablePPO.load`.** It goes through
`agents.model.snapshot.load_foreign_opponent` (verifies the `arch_signature`, runs the deleted-kwarg sanitizer); a bare
load rebuilds the extractor from the zip's own pickled kwargs and dies on any constructor flag deleted since the
checkpoint was written. Re-pointing a name would move that failure, not remove it.

A by-name load either returns a model or raises **`BaselineLoadError`** (a `BaselineError` subclass) with `.reason` in
`pre_generation` · `arch_drift` · `unresolvable` · `not_a_model` and a message that always names the FIX.
`pre_generation`: the entry records a generation this tree no longer contains (`config_version < MIGRATION_FLOOR`, or
another `arch_signature`); the message names the commit the weights ARE readable from, and the registry marks such an
entry `era_checkout_only`. `arch_drift`: a current-generation entry the loader still refuses is a defect, **not a licence
to substitute a stand-in checkpoint**. `is_pre_generation()` answers from registry data alone (no archive, no torch), and
`validate()` makes an unmarked pre-generation entry an ERROR, as it does a stale mark.

```bash
python -m main.baselines check --load     # + actually load every checkpoint entry (~2.5 s each)
```

`src/agents/training/baselines_loadability_test.py` iterates the registry (each entry loads or raises the typed error) and
pins that the bare path still fails where `baselines.load` succeeds. History (the 2026-09-22 measurement, the 2026-09-14
stand-in checkpoint): `training_eval_and_rating_history.md` → "A name either LOADS".

## Bot evaluation (in process on the Rust eval core, BLOCKING)

**Flat schedule, full roster.** Eval fires every `EVAL_FREQ_STEPS` (2M steps; `--eval-freq`) and plays `EVAL_GAMES` (100;
`--eval-games N`) games per opponent (n=100 → ±0.098 per-cell 95% CI, n=200 → ±0.069; the recorded `n_games` is the
actual cycle size): one cadence and one game count for every bot and every self-play sentinel. The roster is the nine bots
of `eval_schedule._EVAL_ROSTER` / `eval_opponent_names()` (v1 and v2 of `heuristic`, `staller`, `aggressive`, `setup_sweep`,
plus `random` as the eval-only "is-the-model-broken" floor, excluded from `win_rate_vs_bots`), shared by the bot path, the
self-play path, the offline harness and the Rust eval core's bot routes; there is no roster flag.

🚨 **The cycle is BLOCKING**: it plays in the trainer's own process between two host
steps of the collector and is collected in the same step (`eval_callback._launch_eval` →
`eval_launch.launch_rust_eval_cycle` → `_collect_pending`), so it costs training its wall time (about 1.5 % of wall at
N=256 on the sizing toy, about 3.5 % on the end-state production net; `design_x26_readiness.md` §2.6 F15). No cycle is
ever "still running" when the next step begins, so nothing is skipped. Measurements: `training_eval_and_rating_history.md`
→ "Bot evaluation".

🚨 **BOT ERA BOUNDARY — F-LF-1, fixed 2026-09-29.** Before it, the SETUP step of `heuristic`, `heuristic2`, `setup_sweep`
and `setup_sweep_v2` never fired, so every `eval/win_rate_vs_{heuristic,heuristic2,setup_sweep,setup_sweep_v2}` row, every
`win_rate_vs_bots` aggregate, every training bot mix and every `bot:<name>` anchor / prober replay measured the BROKEN
bots and is not comparable with a later one (ledger `2026-09-29 · GIGO FIX · F-LF-1`). The same day a NON-Ghost's Curse
became a setup move for all four (owner: "allow Curse"): the same boundary.

**The anchor after the boundary.** `data/gen3_bot_elo_anchors.json` holds the Rustboro-era bot base ratings (installed
2026-10-04, `era: Rustboro`): a mirrored, per-game-seeded round robin of the nine FIXED bots (259,200 games, every bot ±9
Elo at 95 %, `random` pinned at 1000; provenance `designs/research_state/measurements/bot_base_ratings_2026-10-03/`). It
re-scaled every bot-anchored ladder headline: a ladder fitted before that install is on the older scale.

### ⚠️ GLOBAL-RANDOM COUPLING — the five seeds a paired-arm design must set

A drawer that reaches into a **process-wide** RNG couples itself to every other drawer in the process: two paired **arms** that
interleave their draws differently play different games **with no treatment involved**, which a paired design claims cannot
happen, and a fixed sim seed alone buys nothing (found by a failed integrity check, 2026-08-29).

**On the Rust eval core the problem does not arise**: every game is keyed (`rust_eval.seeds`, `gen3_eval_game_seed_v1`) and
a scripted bot's streams are re-seeded per game, so a game is a function of the game and the weights. The process-wide
streams that remain are OPT-IN seams: with no seed the RNG *is* the `random` module (or torch's default generator), so
default behaviour is byte-identical, and an unparseable env seed RAISES rather than silently running unseeded.

| what draws | seed kwarg | env hook |
|---|---|---|
| the **team draw** (`utils.teambuilder`, per battle) | `rng_seed=` | **`$GEN3AI_TEAM_SEED`** |
| the **policy's action sample** (`main.anchors.core_side`: torch's default generator when stochastic) | n/a | **`$GEN3AI_POLICY_SEED`** |
| the **self-play pool draw** (`SnapshotPool.sample`, per episode) | `rng_seed=` | **`$GEN3AI_POOL_SEED`** |

`$GEN3AI_PLAYER_SEED` / `$GEN3AI_STALLER_SEED` belonged to the deleted Python bot classes. **A paired-arm design over battles
seeds all live seams**, with distinct `rng_seed=` values where two sides must be independent as well as reproducible (a flat
seed gives two instances the same sequence). The five-seed census and the falsifier: `training_eval_and_rating_history.md` →
"GLOBAL-RANDOM COUPLING".

### The untaught meter (`untaught_meter.py` · CLI `python -m main.untaught_meter`)

**The meter every fold verdict in the ledger rests on.** It plays a checkpoint PILOTING a fixed team slice against ONE fixed
opponent and reports a cluster-bootstrapped win rate. Offline: no training, no launcher, no server, nothing written under
`models/`.

**THE RECIPE** (one declaration instead of five copies):

| | |
|---|---|
| teams | `--teams` a manifest JSON, **in order: the order IS the seed** (index = team seed offset). Default: the untaught 8 (`reuse_batch_2026-09-03/offline_collateral_kl/untaught_teams.json`); `--taught` swaps in the taught 16 (`teacher_content_2x2_2026-09-04/taught_teams.json`). `pin_sha` (raw bytes) and `team_sha` (strip-normalized) are recorded per team |
| refs | resolved through `fixed_opponent_pool.resolve_model_ref`, the call every checkpoint-reference flag goes through: a bare run dir means the run's **LAST SNAPSHOT**, and the resolved file + `rung` + `rule` + `num_timesteps` are printed per ref and stamped in the JSON |
| opponent | one fixed model piloting the **paired** pool draw, **BY NAME** from the baseline registry: **`untaught_meter_opponent_v14`** (N0's 24M snapshot, config v121), an **INTERIM** default until the Rustboro opponent (legacy manifest D-L3 / R0). **It is `era_checkout_only` (behind the X5 version break's floor, v144): a bare launch refuses at the default (`main.untaught_meter.apply_baseline_defaults`), naming `--opponent` and the pin `f7567a9f`, because no post-break opponent is registered.** `--opponent` also takes any other registry name or a raw ref |
| module tree | **`--config auto` is the default**: each model's OWN `model_config.json`, the only value that loads a current checkpoint (the shared v101 `untaught_meter_config` is selectable by name but unloadable at HEAD) |
| play | `stochastic=True` both sides (keyed draw at T = 1.0) on the Rust EVAL CORE with T2 on the CPU and the training encoder's rows (`agents.training.untaught_rust`); `--n-envs` (32) games in flight; `--concurrency` other than 1 and `--impl node` are refused. Run it from the repo root |
| aggregation | equal-weight cluster mean over TEAMS, and **ONE fixed resampling index set shared by every ref and every contrast**, so a ref-vs-ref difference is paired on the same team draws |

🚨 **THE SERIES BOUNDARY IS RECORDED AND ENFORCED.** Every played artifact stamps `_meta.series` (`opponent_run`,
`opponent_file`, `opponent_num_timesteps`, `opponent_sha256`, `key`, `config_mode`); `--from-rows` **REFUSES a mix of
different opponents** (`MeterError`, exit 1; a sha256 decides when both sides carry one, else `run/file`;
`--allow-opponent-mix` consents) and warns on an artifact that records none. The v101 (rev-1) and v121 (`ai_v14_01_base`)
series are two series.

**THE SEEDS.** A cell = one (ref, team) = one eval cycle at cycle seed `blake2b('gen3_untaught_cycle_seed_v1', seed,
team_index)`, never the ref; each game is keyed (cycle seed, `untaught`, game index) through `rust_eval.seeds`, so every ref
plays the SAME games (CRN) and the schedule is prefix-consistent (a 20-game cell is the first 20 games of a 60-game one). A
cell replays bit for bit at one compute; a decision within a rounding error of a tie can flip with `--n-envs`, so the
compute is stamped. SHARDING is over TEAMS (`--workers N`) and cannot move a number
(`src/main/untaught_meter_reproducibility_integration_test.py`). Artifacts from before 2026-10-06 (the poke-env bridge path)
are the record only: `training_eval_and_rating_history.md` → "The untaught meter".

🚨 **THE TRANSPORT BOUNDARY (P2, 2026-10-06).** Every cell / shard / artifact stamps `_meta.transport = "rust_eval"` (+
`encoder`, `core_stamp`, `eval_regime`, `compute`); an artifact without it is `python_bridge`. `aggregate`, `merge_cells` and
`--from-rows` REFUSE a mix (`TransportMixError`); `--allow-transport-mix` reads them side by side and labels the read
`MIXED:…`. The paired read at the switch was −4.38pp [−8.75, +0.42], NOT DETECTED
(`measurements/pokeenv_p2_ladder_2026-10-06/untaught/`). `main.best_response_gap --play` plays through the same `play_cells`.

🚨 **THE CONTINUATION CONTROL IS THE SECOND COLUMN, not optional bookkeeping.** A plain +1.08M-step continuation of a parent
(no teacher, no stable opponents) moved this meter **+3.45pp [+0.46, +6.48]** on its own (ledger 2026-09-06, cell 2), so a
delta against a **frozen** parent credits a fold with progress the parent would have made anyway. `--control <arms…>` pools
the continuation arms equal-weight and computes their **max-pairwise replicate floor** (`|Δ|` is a magnitude; **one**
control arm gives a re-based delta but NO floor, and the meter says so rather than inventing a zero). `--floor PP` supplies
the externally-ruled floor for the *baseline* column, regime-specific (1.66pp frozen K=3 · 4.27pp controller-live), **never
pooled across regimes**. Verdicts, in order: `WITHIN FLOOR` (|Δ| below the floor; the CI may still exclude zero, which says
the games are consistent, not that the arm differs) → `NOT DETECTED` (CI spans zero) → `SIGNIFICANT`.

**Timeouts:** a game at the turn limit is a DRAW (`Cell.turn_limit_draws`); a read whose timeouts exceed **25%** of attempted
battles reports `INCONCLUSIVE` with no verdict (CLI exit 3).

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.untaught_meter FOLD_A=<run> FOLD_B=<run> \
  --baseline <frozen parent> --control <cont A> <cont B> <cont C> \
  --games-per-team 200 --workers 6 --floor 1.66 --json out.json --md out.md
python -m main.untaught_meter <refs…> --baseline <ref> --check     # resolve only, plays nothing
python -m main.untaught_meter --from-rows A=<untaught_X_end.json> --baseline B=<untaught_Y_end.json>
```

`--from-rows` re-reads committed per-team artifacts (`untaught_<TAG>_<depth>.json`-shaped) with no models and no battles (the
`POOLED` row is a summary, never a ninth cluster); `untaught_meter_test.py` pins the aggregation to the banked 2×2 endpoint
legs.

### The critic gate: RETIRED 2026-10-25 (what outlives it)

`main.critic_gate` (the win-prob-critic arm's pre-registered read, `designs/ai_v12/design_winprob_only_critic.md` §5.5), its
renderer, `main.ops.g7_report`, `main.ops.calib_trend` and `scripts/ops/famine_read.sh` were retired by the owner's standing
grant (the arm's read is decided and closed; `designs/deleted_flags.md`). Three things outlive it:

- **The matched-COUNT ladder read** is `python -m main.elo compare <run_a> <run_b> […]` (`main/elo_compare.py`; "Reading an
  ELO" rules 2-5 below, enforced). Library layer: `snapshot_ladder.own_rated_steps_ever` (the counting basis) and
  `fit_ladder(run_dir, first_n=n, write=False, steps=…)` (the strict-prefix refit); `main.elo refit` is a same-run recipe
  refit.
- **The calibration numbers**: `main.scaffolding_gauge --reliability --reliability-reweight` (selection-reweighted, `bot` and
  `pool` never pooled) and the training-side `main.ops.value_sidecar_read`.
- **The episode-length lesson.** `eval_results.jsonl` carries NO episode-length field (a full-cycle `mean_ep_len_vs_{bots,pool}`
  is a TensorBoard scalar and sits in `metadata.json`'s `latest_eval` / `snapshot_history[*].latest_eval`, keyed by
  CHECKPOINT); a stall rate is a CAPTURE-QUOTA statistic (loss-enriched traces, `meta.turns >= MAX_TURNS`); and a threshold
  defined on EVAL episode length is never applied to the train series `rollout/ep_len_mean` (same name, different population).

### The in-loop eval callback (`PerOpponentEvalCallback`) and what its cycle writes

`PerOpponentEvalCallback` (non-self-play) and `SelfPlayCallback` (self-play) eval IN PROCESS on the Rust eval core, blocking
(*Eval on the Rust env core* below). On each scheduled step the callback snapshots the live weights (`model.save`) and the
cycle plays against that **frozen** snapshot, which is also what the prober reloads and what the best-model copy and a pool
promotion take. **The trainee's eval teambuilder follows the run's `--trainee-team` pin** (`trainee_team_str` →
`agents.training.eval_teams.build_trainee_tb`; both callbacks): a specialist run is measured piloting ITS OWN team, no pin
gives the default pool builder. Each opponent's `EVAL_GAMES` are split into **shard units** of `--eval-shard-games` that the
Rust executor plays in plan order on one env; the parent pools them into one exact result (*Battle-level work-stealing*),
and the collect merges → TensorBoard + TUI + best-model (promoted by copy). The Python-worker era (`main.eval_worker`,
`--eval-workers` / `--eval-device` / `--eval-concurrency-per-worker`) is history: `training_eval_and_rating_history.md` →
"The in-loop eval callback".

**Forensic traces** land under `<run_dir>/eval_traces/step_<N>/<opponent>/` as **core traces** (`gen3_core_trace_v1`,
`rust_eval/traces.py`; files listed in *Eval on the Rust env core*). The filename stem comes from ONE helper,
`eval_quota.trace_filename_stem(outcome, trace_tag, idx)` (`<outcome>_<trace_tag><idx:03d>`, `trace_tag` = `s<shard>_` per
shard unit), the contract the prober's `discovery._FNAME_RE` inverts; `eval_callback_test.test_trace_naming_contract` pins the
pair (a drifted regex once made the whole prober read every sharded trace as outcome `?`).

🚨 **THE RESULT VOCABULARY IS `WIN` | `LOSS` | `DRAW`** (`gen3_trace_result_v2`, `agents/training/trace_result.py`, pure stdlib:
the recorder writes it, the prober, harvesters and gauge read it); `<outcome>` in a filename is the lowercase form. A
`DRAW` carries `meta.draw_kind`: `timeout` (the trainee FORFEITED at `turn >= MAX_TURNS` (250), which the battle layer
reports as a loss: a stall) or `tie` (the sim emitted `|tie|`). `classify_result` tests the 250-turn cap BEFORE the loss, on
the constant the reward uses (`reward_weights._TIMEOUT_TURN_CAP`), so the label and the terminal fold cannot disagree. A
trace with no `meta.result_vocabulary` is the old `gen3_trace_result_v1` era (`trace_result.result_era`): a stall is separable
only by `meta.turns >= MAX_TURNS` and a TIE rate is NOT MEASURABLE (`trace_result.era_note`). `to_summary` and the prober REFUSE
a result outside the vocabulary (`UnknownTraceResult`); nothing back-fills.

🚨 **THE CAPTURE QUOTA IS OUTCOME-CONDITIONAL, and the manifest RECORDS it** (`gen3_trace_selection_manifest_v1`,
`trace_selection.py`). Per opponent per cycle the executor persists at most `_FORENSIC_WIN_QUOTA` (5) wins,
`_FORENSIC_LOSS_QUOTA` (10) losses and `_FORENSIC_DRAW_QUOTA` (5) draws (`--forensic-win-quota` / `--forensic-loss-quota` /
`--forensic-draw-quota`; scaled per shard unit), so traces are a LOSS-ENRICHED sample by design and every consumer that
averages over them (`calibration`, `falsify_scan`, `main.scaffolding_gauge`) must reweight by the recorded capture rates.
Draws have their OWN bucket, so a stall storm cannot evict the decisive losses. `record_eval_selection` patches each cycle's
manifest at COLLECT with, per opponent, `battles_played` / `battles_won` / `battles_drawn` / `traces_written` /
`traces_won` / `traces_drawn`, the derived `capture_rate_{win,loss,draw}` and the rule in words
(`forensic_selection_rule`). 🚨 **A DRAW IS SUBTRACTED FROM THE LOSSES, NOT ADDED TO THE PLAYED COUNT** (`lost = played −
won − drawn` keeps `capture_rate_loss` about DECISIVE losses; the drawn denominator is the unit's `draws_seen`,
`rust_eval/capture_quota_test.py`). The `selection` block is schema 2, additive (schema 1 stays readable, its draw keys
ABSENT rather than zero). **Absent reads as SELECTION UNKNOWN, never as uniform** (a legacy tree, or a cycle that crashed
before collecting, where the block is `null`).

Each cycle also writes `eval_manifest.json` (`write_eval_manifest`): which model produced the traces (`num_timesteps`,
`git_hash` + `arch_signature` from the run's `metadata.json` / `model_config.json`) and a `snapshot` pointer.
`--keep-eval-snapshots N` keeps the N newest cycle snapshots at `eval_traces/step_<N>/snapshot.zip`, so the prober reloads
the exact weights (else the nearest checkpoint). 🚨 **`snapshot.zip` is a HARD LINK to `checkpoints/checkpoint_<N>_steps.zip`**
(`eval_collect.store_eval_snapshot`) when that file is on the same filesystem and BYTE-IDENTICAL (sha256), else a copy; the
manifest's `snapshot_storage` is `{mode: hardlink|copy, note, sha256, checkpoint}` and a copy made although a checkpoint
exists prints `⚠️ [EVAL] … COPIED although a checkpoint exists` with the reason. A link survives the checkpoint's deletion,
`main.prober.groom` counts it NOT reclaimed, and a `cp -r` or cross-filesystem move of a run dir un-links it. The same rule
stores `best_model/best_model.zip` (`eval_collect.store_best_model`) and the self-play pool entry
(`SnapshotPool.add_from_path(..., twin=<checkpoint_at_step>)`) through ONE implementation, `weights_link.link_or_copy`.
🚨 The hazard is an IN-PLACE writer opening a name that shares an inode; there is none, because the periodic checkpointer,
the pool's `add` and the eval snapshots all save through `utils.atomic_io.atomic_save_model` (temp file, then `os.replace`;
`designs/training/checkpoint_writes.md`). Pinned by `eval_snapshot_dedup_test.py` and `weights_link_test.py`.

**The trainer grooms what it writes**, after each cycle: `_prune_eval_traces` keeps the `--keep-eval-trace-steps` newest step
dirs (**default `0` = KEEP ALL**: a cap of 20 once deleted a registered matched-step comparator's traces), `_prune_eval_snapshots`
the `--keep-eval-snapshots` newest snapshots, and `_prune_run_artifacts` (`artifact_retention.py`) the `--keep-stalls` newest
`stalls/stall_*.html` and `--keep-crashes` newest `crashes/restart_err_*.txt`. Manual fallbacks: `python -m
main.prober.groom`, `python -m agents.training.artifact_retention <run_dir | models_dir> [--apply]` (dry run by default).
The eval summary is written to `metadata.json` as a **top-level `latest_eval`** block (step-labeled, robust to async timing;
`save_model_snapshot` carries it forward), and `record_checkpoint` stamps a point-in-time copy into each checkpoint's sidecar
and `snapshot_history` entry (`snapshot._read_latest_eval` reads it).

Behaviours:
- **A failed cycle is logged-and-continued**: an `EvalCoreError` (e.g. a QUARANTINED battle) leaves the opponents missing and
  records nothing (`eval_callback_rust_test`); a lifecycle violation and a missing eval core RAISE (`RustEvalUnavailable`).
- **No hung-cycle watchdog and no skip-while-running rule.** A stop signal is honoured INSIDE the cycle at its safe points
  (`safe_point_fn`, `gen3_deferred_abort_v1`), the abort's 120 s deadline bounds a cycle that never reaches one, and the
  partial cycle is abandoned, never collected.
- **An operator can force an off-cadence eval** from the launcher's `f` button (confirm → SIGUSR2): the handler only flags
  `request_forced_eval()`, and the active callback consumes it on its next `_on_step` (`_ForcedEvalMixin._maybe_force_eval`,
  shared by BOTH callbacks), launching a cycle and consuming the current cadence bucket (`test_force_eval_*`).
- **On resume the last eval is re-published to the TUI** from the checkpoint's `metadata.json` (`replay_last_eval_to_tui`),
  the self-play `pool` block and per-sentinel rows included.
- **The cadence ANCHOR is restored on resume, CLAMPED to the current step** (`_ForcedEvalMixin._restore_last_eval_step`).
  `_last_eval_step` is restored from `metadata.json`; on a FORK that block is the SOURCE run's, whose `latest_eval.step` can be
  far AHEAD of the forked checkpoint, and `(now // freq) > (anchor // freq)` would launch ZERO cycles until the fork caught
  up (no `win_rate_vs_*`, no `eval_results.jsonl` row, no ELO). The clamp announces itself with an
  `anchor is AHEAD` event. ⚠️ It must read **`self.model.num_timesteps`**, not the callback's own `num_timesteps` (still `0`
  at `_init_callback`). Test: `eval_fork_cadence_test.py`.

| Flag | Default | Notes |
|------|---------|-------|
| `--eval-freq` | `None` (= `EVAL_FREQ_STEPS`, 2M) | total env steps between cycles; lower it for short arms |
| `--eval-games` | `None` (= `EVAL_GAMES`, 100) | games per **opponent** per cycle; 200 → ±0.069 at proportionally more BLOCKING wall time |
| `--eval-shard-games` | `25` | games per shard unit; `>= EVAL_GAMES` ⇒ one shard per opponent; aggregation is exact |
| `--keep-eval-snapshots` | `10` | newest N `eval_traces/step_<N>/snapshot.zip` (~62 MB each, ~0 extra when hard-linked); `0` writes the identity manifest only |
| `--keep-eval-trace-steps` | `0` (= **KEEP ALL**) | a positive N caps the step dirs (~55 MB/cycle, ~3 GB for a 75M run) |
| `--keep-stalls` / `--keep-crashes` | `50` / `10` | newest `stalls/stall_*.html` / `crashes/restart_err_*.txt` kept; `0` = keep all |

**TD-residual tail metric (`eval/td_resid_tail_*`).** Each cycle folds the **left tail of the per-decision critic surprise** δ(t) =
r(t) + γ·V(s_{t+1}) − V(s_t) (the prober's formula, `main/prober/session/core.py::ProbeSession._td`) over the forensically
captured battles only (zero extra GPU), pooled per opponent as a **CVaR@5%** (`TD_TAIL_FRAC`; a single min below
`TD_TAIL_MIN_SAMPLES` = 20), into `eval/td_resid_tail_vs_<opponent>` + `eval/td_resid_tail_mean` (TB + TUI), `latest_eval`
and `eval_results.jsonl`; `model.gamma` is threaded into the cycle (`run_rust_eval_cycle(gamma=…)`) so the live δ matches the
prober's recompute. More negative = the critic was blindsided more often: the leading indicator for critic-coverage work (it
moves in a cycle or two, where saturated win rates, the gate-pinned `win_rate_vs_pool` and wide-CI ELO don't).

### Battle-level work-stealing (`eval_sharding/` package)

The `eval_sharding/` package (`units.py`, `results.py`, `pool.py`, and `merge_eval_results` in `eval_collect.py`) holds the
PLAN and the EXACT AGGREGATION of an eval cycle. The Rust executor plays its units in plan order on one env and publishes
the same `ShardResult` the deleted Python workers did (their lock-claim and cross-process work-stealing died with them; the
section keeps its name).

- **`units.py`** — `EvalItem` (one opponent) + `ShardUnit` (a chunk of its games) + `plan_units(items, shard_games)`, a **pure**
  partition: ≤`shard_games` games per chunk (Σshards == n_games exactly), cost-descending items, shards round-robined.
- **`results.py`** — `ShardResult` (raw additive metrics: won/finished, reward+turn sums, the raw δ list, never a reduced
  ratio) + `aggregate`, which pools an opponent's shards back **exactly**: win_rate = Σwon/Σfinished, reward/ep_len
  count-weighted, the TD tail by **pooling the raw δ then one `td_tail`** (a CVaR can't be averaged).
- **`pool.py`** — `ShardedEvalPool`: `write_plan(run_dir)` → `collect(result_dir)`; `plan.json` is the single source of truth.
  `merge_eval_results` delegates to `collect` and returns the `merged` shape every downstream consumer reads, plus additive
  `counts` (exact W/L) and `coverage`.

win_rate / reward / ep_len are exact regardless of `shard_games`; `td_resid_tail`'s aggregation is exact, but the
captured-battle sample it covers shifts slightly with the shard count (the capture quota is per unit, `max(1, ⌈quota/shards⌉)`).
Tests: `eval_sharding_test.py`.

### Mirrored team pairs (`--eval-mirrored-pairs`, T17 — `gen3_mirrored_pairs_v1`, DEFAULT OFF)

**What it is.** Every team pairing of the in-loop eval (the bot roster, the pool sentinels and the fixed / stable opponents)
is played from BOTH sides on ONE battle seed: game `2k` the trainee pilots team `A` (its own builder's draw) against the
opponent on `B`; game `2k+1` the teams are HANDED OVER (trainee on `B`, opponent on `A`) with the same battle seed, scripted-bot
streams and sampled-opponent seed. Team-draw luck cancels inside the pair instead of riding the win rate as noise; the
trainee keeps seat p1 in both games. Training games are untouched. **One rule** (`rust_eval.seeds.pair_game`, called by
`executor._make_game`): unmirrored, a game's key is its own `(cycle seed, opponent, game)`; mirrored, games `2k` and `2k+1`
both take the key of game `2k` and the second is `swapped`.

**Counts are even by construction:** an odd `--eval-games` is rounded UP (`mirrored_eval_games`) and the shard split is in PAIRS
(`units._split_games(paired=True)`), so a pair's two games sit in ONE unit; the plan carries `"mirrored": true` (absent when off).

🚨 **THE PAIR IS THE STATISTICAL UNIT** (`agents.training.mirrored_pairs`). A game is worth 2 / 1 / 0 HALF-POINTS to the
trainee (win / draw / loss; a timeout and a `|tie|` are DRAWs), so a pair is one draw from a five-category PENTANOMIAL. Each
shard publishes its `pair_counts`; the collect pools them exactly (`merged["pairs"]`); a pair with an unfinished game is VOIDED
whole. Every interval recorded on mirrored games is the pentanomial one over PAIRS, never a per-game binomial one:
`eval/pair_score_vs_{bots,pool}` (the mean per-game score, a draw = ½, Random excluded from bots), `eval/pair_score_ci_vs_*`
(the 95% half-width) and `eval/pairs_vs_*` (the pair count). `win_rate_*` keep their meaning (`n_won / n_finished`) and the
promotion read stays `win_rate_vs_pool > --promote-threshold`. ⚠️ **`eval/elo`'s CI is still the BT fit's per-game one**: the
live BT fit consumes game counts; the per-opponent pentanomials ride on the row for a pair-level refit.

**THE REGIME, recorded and inherited like `eval_sentinel_greedy`.** `model_config.json`'s `eval_mirrored_pairs` (config v128) is
read back by `_resolve` on a flagless resume, so a live run never crosses the boundary unless the flag is NAMED; the launch
prints `⚖️  [EVAL REGIME] in-loop eval pairing: …` with its source. Every artifact states it (the cycle manifest's
`mirrored_pairs`, the `eval_results.jsonl` row's `mirrored_pairs` block, ABSENT on an unmirrored row, and
`latest_eval.mirrored_pairs` / `latest_eval.pool.pairs`).

🚨 **A REGIME BOUNDARY — and the readers refuse to cross it.** Half the mirrored games put the trainee on the OPPONENT
builder's team (the flat pool for a bot; a specialist's untaught team), so `win_rate_vs_pool` / `eval/elo` / the promotion
read are a different population on either side of it. `elo.load_rows` REFUSES (`MixedEvalRegimeError`, naming the first
mirrored step) a run whose rows span it unless the caller picks one regime (`mirrored=True/False`); the live callbacks pass
THEIR regime. `snapshot_ladder.eval_measured_pairs` skips mirrored rows (not the ladder's protocol), and
`best_response_gap`'s regime tuple carries `eval_mirrored_pairs`, so two exploiters across it are an UNMATCHED comparison.

**The default is OFF on purpose.** The parser default is `None`: a fresh run resolves OFF, a flagless resume INHERITS the
recorded value, and `--arch production` does not set it. It is flipped at an era start together with `--promotion-sprt` (T6)
so it is one regime change, never mid-comparison. Read a run's regime from its `model_config.json` `eval_mirrored_pairs`.

🚨 **The PINNED-TEAM meters do NOT mirror: `--mirrored-pairs` is REFUSED there** (owner 2026-10-02). `main.untaught_meter` and
`main.best_response_gap --play` measure the pilot ON its pinned team; a mirrored pair hands that team to the opponent for its
second game, so a pair would pool PILOTING the pinned team with the RESPONSE to it. Both CLIs and the one library path they play
through (`untaught_meter.play_cells(mirrored=True)`) refuse before anything is read (typed `MirroredPinnedTeamError`, text
`mirrored_pairs.pinned_team_refusal`). Mirroring belongs to the SYMMETRIC evals: the in-loop `--eval-mirrored-pairs` and
`main.anchors --mirrored-pairs` ([`EXTERNAL_ANCHORS_SOP.md`](../ops/EXTERNAL_ANCHORS_SOP.md)). The readers of a mirrored cell stay,
for shards written while the meters could mirror (2026-10-01 to 10-02).

Tests: `agents/training/mirrored_pairs_test.py`, `main/train/eval_mirrored_pairs_regime_test.py` (resolve / inherit / migrate),
`main/anchors/mirrored_test.py`. The CPU evidence of 2026-10-01 and one open item (a GPU-run confirmation):
`training_eval_and_rating_history.md` → "Mirrored team pairs".

### The eval COUNT ledger (`agents/training/eval_ledger/`, `gen3_eval_count_row_v2`, eval U1)

**What it is.** `design_evaluation.md` §0b, built: ONE append-only, archive-level ledger of COUNTS, one row per (batch ×
matchup), at `<archive>/_ledger/` (`utils.paths.run_archive_dir()`; `$GEN3AI_MODELS_DIR` is authoritative; pytest seals it, so
a test uses the `run_archive` fixture or a temp root). Beside the rows: `requests/` (the claim queue, a deterministic fold of
`open` / `family` / `claim` / `void` / `row` / `done` / `cancel` events), `decisions/` and `references/`. One writer per file,
every append fsynced; under `models/` a writer may write ONLY to `<archive>/_ledger/` itself.

🚨 **EVERY READ IS DECLARED.** `eval_ledger.read(ReaderDecl(name, purposes, regime, requests, selection, flags_ok, inference[,
decision_kind]), ...)`, every field spelled out. It validates every row (a v1 row is upgraded ON READ, never rewritten),
applies corrections, refuses ANY duplicate batch key or seed block anywhere in the ledger (`DuplicateBatchError`), filters by
the declaration, returns ONE regime (`MixedRegimeError`) and honours `as_of`. `src/eval_ledger_reader_gate_test.py` (static,
EMPTY allowlist) fails a module that reads the ledger any other way, or whose declaration is not spelled out.

🚨 **EVERY BATCH IS WRITTEN UNDER A CLAIM, FOR A REQUEST.** `LedgerWriter.claim` takes the unit `(request, batch, player,
opponent, regime)` under one `flock`; a dead writer's claim (pid dead on this host, or expired) is voided deterministically
and the batch replayed on the same seeds; a writer whose claim was voided DROPS its batch. A request pins its purpose, its
spec and (once a row lands) its regime; a family pins its protocol.

**Per-cell INCONCLUSIVE and family reads.** `cells.cells(read)` gives each (request, player, opponent) cell its verdict:
INCONCLUSIVE when `4 × aborted > attempted` (integers: exactly 25 % is not) or below a registered pair minimum; `cells.looks(read)`
reads a group-sequential FAMILY across its looks; `cells.pooled_pairs` refuses a read declared `across_runs`.

**Operate it:** `python -m main.eval_ledger audit | show | verify <decision> | void-dead | close-stale [--apply] |
family-register | request-close`. `audit` exits 1 on any broken cross-record invariant (duplicates, unresolved requests /
families / supersedes, a requested row without its claim, a decision whose rows or digest no longer match).

Tests: `agents/training/eval_ledger/{schema,writer,reader,index,producers_contract}_test.py`.

### The in-loop eval on the ledger (`agents/training/cycle_ledger.py`, eval U2 — DUAL WRITE)

🚨 **The trainer's in-loop eval cycle and the SPRT promotion write ledger rows BESIDE their old files.** The trainer builds ONE
`CycleLedger.for_run(model_dir)` at startup (`main/train/callbacks.py`, only when eval runs; the `📒 [EVAL LEDGER]` line names
the root) and hands it to the eval callback; rows land in `run_archive_dir()/_ledger/rows/inloop/`, protocol
`gen3_eval_protocol_v1_inloop`. `eval_results.jsonl`, `metadata.json`'s eval block and `sprt_promotion.jsonl` are written
EXACTLY as before and every existing reader (TensorBoard / TUI, `main.elo`, `best_response_gap`, the supply guards) still
reads them. Nothing about a game changed (storage only; the before / after outcome-digest proof is
`designs/research_state/measurements/eval_ledger_u2_2026-10-04/`).

- **The cycle:** purpose `cycle`, ONE request per (cycle step × regime) (`<run>:cycle:<step>:<regime_id>`: bots, sentinels and
  each fixed opponent are different regimes), one row per opponent: exact W / L / D (a draw is COUNTED), per-team counters,
  the pentanomial under `--eval-mirrored-pairs`, the outcome digest. `seat_rule = fixed_p1`; the trainee greedy.
- **The SPRT** (`--promotion-sprt`): ONE request per candidate (`<run>:sprt:<step>`, kind `sprt`, purpose `promotion`), one
  row per (batch × sentinel), and ONE decision row at the verdict (`promotion`, subject = the candidate's sha256, verdict
  `accept` / `reject` / `abandoned`).
- **How it is kept honest:** claims are taken BEFORE the cycle plays; after it, every row is checked against the executor's
  published shard results (a mismatch RAISES `CycleLedgerError`) before any row is appended. A stop at a safe point writes
  nothing; a failed cycle writes nothing and CANCELS its requests; a re-eval at the same step replays only what is missing.
- **Cost is flat in the archive's size** (`design_evaluation.md` §0b.4 / §0b.7): the writer folds only the requests-stream
  bytes beyond each file's cursor (a persisted index under `<ledger>/.ledger_index/`, a CACHE rebuilt when missing, corrupt or
  stale; `python -m main.eval_ledger audit --rebuild-index`). `GEN3AI_LEDGER_INDEX=0` runs the old full fold.

Tests: `agents/training/cycle_ledger_test.py` (a fake executor) and `cycle_ledger_integration_test.py` (`sim` + `slow`, the REAL
core).

### The checkpoint-vs-checkpoint head-to-head (`main.h2h`, X5 §7.3 / U0 — ledger v2 since eval U1)

**What it is.** The offline, SYMMETRIC read of one checkpoint against another: `python -m main.h2h play --player <ckpt.zip |
run dir | run@step> --opponent <…> --pairs N [--out <root>]` plays N MIRRORED team pairs on the Rust eval core and prints the
player's win rate with its PAIR-clustered 95 % interval; `python -m main.h2h read [<root>]` pools a ledger's h2h rows per edge,
one regime at a time. It declares ONE T2 service with two slots (the player's eval slot, one SENTINEL slot for the opponent)
and ONE eval core (two architectures: one slot group per architecture, below) and plays the plan through
`RustEvalCore.run_cycle`, the executor the in-loop eval and the SPRT promotion use, in BATCHES. It writes the COUNT ledger
(protocol `gen3_eval_protocol_v1_h2h`) to the archive's `<archive>/_ledger/` or to a root `--out` names outside `models/`,
every batch under a claim for a REQUEST: by default one derived from the players, the regime and the schedule, so a re-run
resumes it; `--request` / `--family` / `--request-kind` / `--purpose ab` put the edge into a caller's request (an X5 look).

**The regime — on every row, and readers refuse to mix it.** BOTH sides GREEDY (the eval regime); MIRRORED pairs
(`rust_eval.seeds.pair_game`); the turn limit is `StallConfig().threshold` (a timeout is a DRAW); BOTH sides draw their teams
from the PLAYER's eval team builder (the default pool with its 10 % sample-team bias, what `eval_builders(None, [])` builds),
UNLESS `--panel <name>` names a REGISTERED frozen team panel (below). REFUSED: a player whose run pinned its eval trainee team
(`metadata.json` `matchup_history`) and one whose `model_config.json` records `progress_decision_tense` /
`progress_switch_freeze` ON (the env core no longer has those variants). Both players load through `load_checkpoint_strict`
with `historical_load_kwargs` (a deleted kwarg is stripped, an ON one refused) and are gated against the live architecture
(`check_opponent_snapshot_compatible`); a slot group's version and toggles are the checkpoint's RECORDED ones (read from its
policy kwargs, never off the built modules; `gen3_recorded_version_gate_v1`, `main/h2h/recorded_version_cross_test.py`).

🚨 **THE MIRROR CANCELS TEAM LUCK, NOT SEAT LUCK, AND IT IS NOT AN EXACT MIRROR IMAGE.** A pair hands the teams over and KEEPS
the player in seat p1 on one battle seed. So (1) a SEAT advantage `u` adds to every edge: a checkpoint against ITSELF reads
`0.5 + u`, and the measurement carries it (`runfloor.seat_effect`, or the two directions of an edge: `d_ij + d_ji = 2u`); (2)
a player against itself is NOT exactly 0.5 per pair: the second game is the first with the seats relabelled, a mirror image
only while the engine is seat-symmetric, and a speed tie is broken by the RNG in a seat-dependent order, so some self-play
pairs score WW or LL (5.2 % on the X5 P0 finals) and the pair-clustered interval is the honest one. `play_test.py` pins the
exact cancellation of the SCORING on a seat-symmetric toy engine; `play_mirror_integration_test.py` asserts it on the real
engine only for the pairs that stayed mirror images decision for decision. A 4-game block (teams × seats) would cancel seat
luck too but needs the core to seat the sentinel at p1, which it does not (a FINDING, not built).

**The rows.** One row per (batch × matchup), appended and fsynced when the batch ends: the player and opponent (`id` =
`<run>@<step>`, sha256, path), the regime (+ `regime_id`), a `compute` block (device, backend, env count, games per second,
load, a NEAR-TIE CENSUS and the OUTCOME DIGEST: sha256 over the `(game, W/L/D, turns)` vector of the games with no decision
inside 2e-3), `purpose` (default `audit`; `ab` for a pre-registered A/B read), the `request` block, W / L / D / aborted, the
pentanomial, per-team counters and the seed block. `eval_ledger.validate_row` checks the shapes, the closed vocabularies and
the arithmetic (W + L + D = 2 × pairs; half-points = 2W + D). `--out` under `models/` is REFUSED except the archive's own
`_ledger/`.

**Reproducible by construction — and the one thing a re-run can change.** Every game's teams and battle seed are a pure
function of `(schedule seed, schedule key, batch, game index)` (`play.cycle_seed` + `rust_eval.seeds`), never of the env, the
env count, the thread count or the batch schedule. The default schedule key is a digest of the two checkpoints' content
hashes, ORDER-INDEPENDENT (`A vs B` and `B vs A` draw the same team pairs). What can differ across devices, backends or batch
shapes is the FORWARD: a greedy decision within a rounding error of a tie can flip. Every row therefore counts the decisions
inside `NEAR_TIE` (2e-5, twice the CPU eager-vs-eager bar) and inside the GPU-bar margin (2e-3), the games that contain one,
and the pairs with none (`clean_pairs`, `clean_pairs_off_center`): games whose outcome rests on a rounding error are COUNTED,
never left to chance (standing rule 8). A re-run of one configuration reproduces every game bit for bit.

**Resumable.** A re-run with the same arguments resumes the same request (skips recorded batches; a longer `--pairs` plays only
the new ones); a different `--batch-pairs` over an existing request is REFUSED, and the ledger refuses a second row on one
seed block under ANY request. **Reading:** `stats.edge_summary` pools an edge's batches (mean per-game score over PAIRS, a
draw = ½, its pair-clustered normal interval `mirrored_pairs.pair_score_ci`); a reader refuses rows of two regimes or two
edges; `runfloor.sigma_h` turns a complete round-robin into the run SD σ_h with the METER's variance subtracted exactly.

**Hardware.** On CUDA the tool takes the GPU lock itself (re-entrant under `scripts/ops/gpu_lock.sh`); a heavy job runs under
`scripts/ops/mem_cap.sh`. The engine START (the T2 graphs compile again for every engine) dominates a small cell, which
`play-many` removes from every cell after the first.

**On a REGISTERED team panel (`--panel <name>`, `main/h2h/panel.py`).** `main.h2h play | play-many --panel home72` makes BOTH
sides of every mirrored pair draw from ONE panel of `designs/research_state/heldout_panels.json` (`main.heldout_panels`)
instead of the pool: a plain `Gen3Teambuilder` over the panel's teams, no bias, re-seeded per (game key, side). The regime's
`team_set` is `ts:` over `{registered_panel, panel name, its gen3_team_set_digest_v1 content digest, the registry schema, the
draw rule}`: the panel's CONTENT is identity, so its rows never pool with the pool's or another panel's (protocol unchanged,
`gen3_eval_protocol_v1_h2h`). REFUSED before any request is opened: an unregistered name, a panel whose files no longer match
the registry (count, content digest), and a panel the gen3ou validator does not pack WHOLE. Tests: `panel_test.py`,
`play_panel_integration_test.py`. D1's P reads it as `python -m main.x35.d1 p-input` (one registered `--purpose ab` family
look; a look whose rows are on another team set is INCONCLUSIVE).

**Many cells on ONE engine (`main.h2h play-many`, `main/h2h/many.py`).** `python -m main.h2h play-many --cells <cells.json> |
--players P1 P2 … --opponents O1 O2 … --pairs N [the play flags]` plays every cell (`--players`/`--opponents` = the CROSS,
players outer) on ONE engine built once; a cell only changes which checkpoints the next cycles LOAD into the slots
(`H2HEngine.set_cell`). Each cell is planned and played by the single-cell code (`play.plan_edge` / `play.play_planned`), so
this is STORAGE AND ENGINE REUSE ONLY: the games are byte-identical to single-cell `play` on the same seeds
(`play_many_integration_test.py`). `--request R` puts EVERY cell into one request (an X5 LOOK: `--purpose ab --family F
--request <look>`); without it each cell keeps its own default request, so `play` and `play-many` resume each other's rows.
- 🚨 **UP TO TWO ARCHITECTURES PER ENGINE (`main/h2h/arch.py`).** A PRE-FLIGHT sorts every side of every cell into at most TWO
  architectures (`arch.declare_engine`: architecture toggles, model version, served state-dict signature, forward
  fingerprint) BEFORE any engine is built: one T2 slot group per architecture with ONLY the slots its cells need, and one eval
  core per (player group, opponent group). A THIRD architecture or an undeclared combination is `CellArchMismatch`. A
  same-architecture cell plays on the two-group engine exactly the games it plays on the single-group one
  (`play_cross_integration_test.py`). GPU memory for a cross is ESTIMATED, UNMEASURED.
- 🚨 **A KWARG RECORDED AT ITS DEFAULT DOES NOT SPLIT AN ARCHITECTURE.** The forward fingerprint hashes the CANONICAL
  extractor kwargs (`agents.inference.service.slots.canonical_extractor_kwargs`): a key recorded at EXACTLY (same type, equal,
  same repr) the value its ABSENCE means at load is dropped. "What absence means" is the extractor class's own
  constructor-signature default (what sb3's load applies), NOT the flag registry's CLI default and not a `ModelVersion`
  migration default; any non-default value still splits. Tests: `src/agents/inference/service/slots_fingerprint_test.py`,
  `src/main/h2h/arch_default_kwargs_test.py`. The same fingerprint groups the TRAINER's T2 slots
  (`rust_rollout.build._arch_key`; `designs/training/rust_collector.md`).
- 🚨 **THE TEAM POOL IS READ FROM THE WORKING DIRECTORY** (`utils.team_loader.TeamLoader`: `./data/teams`), so `main.h2h`
  REFUSES any cwd whose `data/teams` is not this checkout's (`play.check_team_pool`). Run it from the repo root.
- **A SCHEDULED team source** (`H2HEngine(team_builder=...)`, opt-in; its one caller is
  `main.team_strength.scheduled.ScheduledTeams`): BOTH sides draw from it, and an unplanned draw is a typed `ScheduleMiss`.
- 🚨 **HOT-SWAP SAFETY.** Every cycle loads the cell's two slots through `InferenceService.load` (a bit-exact copy check and a
  parity gate; a failure POISONS the service), and after every cycle `H2HEngine.verify_slots` checks both slots bit-exact
  against THIS cell's checkpoints: a skipped load or a no-op copy is caught before the batch is scored, and no row is written.
- 🚨 **THE PER-SIDE ORACLE REVEAL (`--oracle-reveal-mode {off,one_sided,both_sided,withheld}`, `main/h2h/reveal.py`: its
  docstring is the spec).** `main.h2h` is the one offline tool that PLAYS an oracle checkpoint (a run that recorded a non-`off`
  `--oracle-reveal` spec). A plan declares ONE mode and each cell's `(p1, p2)` levels follow from it and the checkpoints'
  RECORDED levels, never typed: `off` (an oracle checkpoint is REFUSED, `RevealModeError`), `one_sided` (clairvoyance, the
  primary read), `both_sided` (DESCRIPTIVE, out of distribution for the non-oracle side) and `withheld` (the HONEST-PLAY read,
  `design_hidden_info_binding.md` §8.4: the oracle-trained checkpoint played at `off` BY DECLARATION, the one explicit
  exception to `off`'s refusal). A mode needs exactly ONE oracle side per cell; a side at a level other than its
  checkpoint's or the mode's is `RevealLevelMismatch`. Each mode is its own eval PROTOCOL
  (`gen3_eval_protocol_v1_h2h_oracle_*`: its own regime, never pooled), and every revealed row carries
  `compute.oracle_reveal` with the oracle checkpoint's PROVENANCE, read from its RECORDED label in `model_config.json`,
  NEVER from `metadata.json`'s per-checkpoint `oracle_reveal_record` (diagnostic forks written before
  `gen3_reveal_provenance_v1` took it from their PARENT: the 11 `rb_hib_*` knock-ins, KNOWN-BAD in `main.lineage` /
  `main.sidecar_audit`) and the NEAREST `--diagnostic-fork-consent` stamp (`RevealProvenanceMismatch` on a contradiction).
  **`off` is byte-identical** to the engine before the reveal. Tests: `reveal_test.py`, `play_reveal_integration_test.py`,
  the Rust `oracle_reveal_test.rs`.

**The A/B's registered read (`main/h2h/cross.py`, X5 §7.4, committed before any look-1 cell).** Pure functions over the seed ×
seed matrix h_ij (pp, draws ½): Δ̂ = mean h − 50, V̂ = (s²_R + s²_C)/n, t = (Δ̂ + δ)/√V̂ on 2(n − 1) df, δ = 3.5; look boundaries
5.761 / 2.683 / 1.874; the outcome table (INCONCLUSIVE / NON-INFERIOR / FUTILITY STOP / CONTINUE / NOT DETECTED); rule 8 at
EVERY boundary. The matrix comes from a declared FAMILY read (`inference="across_runs"`) per protocol, and a missing / short
(< 1,000 pairs) / aborted cell is an INCONCLUSIVE reason. The oracle reads (C, C_full − C_species, F, H) are in the module.

Tests: `agents/training/eval_ledger/` (above); `main/h2h/*_test.py` (scoring and the toy-engine mirror, seeds, the digest, resume,
the multi-cell engine, the two-architecture engine, the reveal, the panel, `cross`) and the `*_integration_test.py` real-engine
halves (CPU, tiny, the in-process core).

### SPRT promotion (`--promotion-sprt`, T6 — `gen3_sprt_promotion_v1`, DEFAULT OFF)

**What it replaces.** Without it, a snapshot is promoted the first cycle its `win_rate_vs_pool` crosses `--promote-threshold`
(optional stopping: a winner's curse). With it, **every eval-cycle snapshot is a CANDIDATE** once the pool holds a snapshot,
and its own sequential test decides it (`agents/training/sprt.py`, `sprt_promotion.py`); `--promote-threshold` is then not
read and typing it is REFUSED. It needs `--self-play`.

**The test is Fishtest's GSPRT.** The likelihood is the PENTANOMIAL over MIRRORED PAIRS (*Mirrored team pairs*; a 250-turn timeout
and a `|tie|` are DRAWs, ½); the maximum-likelihood pentanomial under each hypothesis's mean constraint is solved by
deterministic bisection on its 1-D Lagrange multiplier (Van den Bergh; 200 fixed halvings, pinned to a residual < 1e-12), and
the LLR is `N · Σ p̂ ln(q1/q0)` with zero counts regularized to 1e-3.

**The hypotheses are in SCORE space: H0 ≤ 0.50 vs H1 ≥ 0.55**, the mean per-game score with a draw worth ½; α = β = 0.05,
Wald bounds ±2.944. Score space because the owner's hypotheses and the gate being replaced are stated as win rates, draws
are rare (and unmeasurable in old traces), and the GSPRT already uses the empirical pentanomial variance.

**The schedule** (from the Monte Carlo, [`measurements/sprt_promotion/`](../research_state/measurements/sprt_promotion/README.md);
pinned by `sprt_test.py`): a check every **40** pairs, split round-robin over the sentinels; the first decision at **40**; a
cap of **1,680 pairs** (the untruncated worst-cell p95), reaching it is a REJECT; plain **Wald** bounds. At 20k runs per
cell: false promotion at a true 0.50 ≈ 2.7 %, false rejection at a true 0.55 ≈ 2.7 %, expected pairs at 0.50 225 to 346.

**The three discipline rules, made true rather than intended:**
1. **The pool is FROZEN at the test's start.** The test plays the sentinels the candidate's own cycle launched against, in
   every batch, and runs to its verdict inside the candidate's own collect, so no other cycle can launch, promote or seed.
2. **Selection games never pool into the decision.** The cycle's own pool games are telemetry; the test plays its OWN fresh
   pairs on a seed namespace disjoint from every cycle's (`sprt_seed`, pinned by a test), with no forensic trace.
3. **A failed test is never re-run.** `<run>/sprt_promotion.jsonl` holds the start and verdict lines, append-only; a candidate
   step with ANY line is never retested. A test the process died in (or a batch that played nothing) is recorded `abandoned`
   at the next start, which means not promoted.

**Where it runs.** The whole test runs inside the candidate's own collect on the Rust eval core, blocking like the cycle, so
the live weights ARE the candidate's. ⚠️ **A long test blocks training until its verdict.**

**What it records.** On ACCEPT the candidate's frozen snapshot is promoted exactly as before (pool, generation,
`train/selfplay_promoted_steps`, the detached ladder update); TensorBoard gets `eval/sprt_llr`, `eval/sprt_pairs` and
`eval/sprt_promoted`. 🚨 **The test's own score is a SELECTED number** (`test_score_selected` in the log): a promoted
snapshot's strength is read from later, unselected games (`ladder.json`), never from its passing test. **The regime:**
recorded as `promotion_sprt` (config v129), `_resolve`-inherited, announced as `⚖️  [EVAL REGIME] promotion: …`; the parser
default is `None` (a fresh run resolves OFF, a flagless resume inherits), flipped at an era start together with
`--eval-mirrored-pairs`. The CPU evidence of 2026-10-01 and the GPU confirmation that was owed (**UNVERIFIED** whether done):
`training_eval_and_rating_history.md` → "SPRT promotion".

### The plateau meter, TIER 1 (`python -m main.plateau`, eval U9a — `gen3_plateau_t1_v1`)

**What it answers:** "does Tier 1 say PLATEAU, or not yet?" for a live or finished run (`designs/endstate/design_evaluation.md` §8.1,
"As built"). Every Δ = 10M steps, the newest node (the first periodic checkpoint at or above the grid step) plays the node W =
50M steps back on mirrored pairs, and a pentanomial GSPRT (`agents/training/plateau_t1.py`: H0 0.50 / H1 0.52, α = β = 0.05,
Wald bounds, 40-pair batches, cap 6,000 pairs) decides GAIN / FLAT / UNDECIDED (INCONCLUSIVE above 25 % aborted games).

```bash
python -m main.plateau status <run>                  # every check's state + the run's Tier-1 status; plays nothing
python -m main.plateau tick <run> [--max-checks N]    # play every DUE check to its decision (CPU by default)
python -m main.plateau tick <run> --dry-run           # list the due checks
python -m main.plateau tick <fork> --lineage-run <parent>   # the parent's checkpoints are nodes too
python -m main.eval_ledger verify <decision_id>       # re-derives a check's verdict from its rows
```

- 🚨 **Tier 1 alone is a CANDIDATE.** `TIER1_PLATEAU` (FLAT at two consecutive checks) is not a plateau: §8.3's plateau also
  needs the cycle monitor NONE and the panel FLAT at the same two checks. `status` prints TIER 2 below Tier 1 and the run's
  **TWO-TIER STATUS** (next section); `--tier1-only` skips it.
- **Statuses:** `NOT_YET` · `CLIMBING` (newest decided check GAIN) · `FLAT_ONCE` · `TIER1_PLATEAU` · `CONTINUE` (newest decided
  check UNDECIDED / INCONCLUSIVE: re-check next interval). A later check still running is listed as pending.
- **Ledger:** one `plateau_t1` request per (run, check step, W), purpose `plateau`, the h2h protocol and regime; its spec
  freezes both nodes' sha256 (a replaced node file is REFUSED); one decision row (kind `plateau`) per request, whose `rule`
  string carries every constant. `status` exits 2 on a CONFLICT. **Resumable at any batch**: a re-run never replays a banked
  batch and records a decided test without playing.
- **Parallel batches (`--inflight-batches K`, default 4).** One WAVE of K batches is ONE eval cycle on the one engine in ONE
  process (so rule 5b's ≤ 2 heavy jobs holds at any K); every batch keeps its own seed block (the executor's `seed_blocks`)
  and its own row and claim. The decision is unchanged (the rule folds batches in order and stops at the first crossing; an
  overshoot of ≤ K − 1 batches costs games, never the verdict). Proof: `design_evaluation.md` §8.1 "Parallel batches".
- 🚨 **Where to run it.** Offline, by the operator after each 10M checkpoint (the deep run's training agent can call `tick` from
  its cron). CPU by default, under `scripts/ops/mem_cap.sh` and `nice`; **never on CPU beside an X5 A/B arm**
  (`design_evaluation.md` §4.2); `--device cuda` takes the GPU lease through the engine (lease holder only). The in-trainer
  window is not built (U4).

### The cycle monitor = the plateau's TIER 2 (`python -m main.cycle_monitor`, eval U5 + U9 offline — `gen3_cycle_monitor_v1`, `gen3_plateau_t2_v2`)

**What it answers:** "is the run CYCLING — beating its recent past while holes open against older selves?" at every Δ = 10M
check, and, with Tier 1 and the panel, which §8.3 signature the run is in (`designs/endstate/design_evaluation.md` §2.5 / §8.2
"As built"). Each check's ROW plays the newest node against every older node at the hybrid lags {1..8, 16, 32, 64} × Δ, 500
mirrored pairs per edge; the accumulated matrix is read by four signals (Hodge χ², the lag curve, triangles through the newest
node, the Nash support), each at α = 0.05 against a parametric bootstrap or its Bonferroni / Holm bound. Verdicts: NONE ·
SUSPECTED (a signal flags) · CONFIRMED (it flags again on 500 fresh top-up pairs) · INCONCLUSIVE (a signal in its rounding band,
or > 25 % aborts).

```bash
python -m main.cycle_monitor status <run> [--panel NAME]       # per check: monitor + panel + the TWO-TIER STATUS
python -m main.cycle_monitor tick <run> [--panel NAME] [--grid G ...] [--max-checks N] [--dry-run]
python -m main.cycle_monitor tick <fork> --lineage-run <parent>  # a fork re-plays the parent's rows in its own family
python -m main.cycle_monitor ladder <run> [--tier1] [--json]   # DESCRIPTIVE: the in-run snapshot ladder, no games
python -m main.plateau status <run> [--panel NAME]             # Tier 1 + Tier 2 + the two-tier status
```

- **The TWO-TIER STATUS** (`agents/training/plateau_t2.py`): `TRANSITIVE_IMPROVEMENT` (Tier 1 GAIN, monitor counted NONE) ·
  `TRUE_PLATEAU` (Tier 1 FLAT, monitor counted NONE, panel FLAT at two consecutive checks) · `PLATEAU_BUT_PANEL` (the same
  without a FLAT panel: NOT a plateau) · `TREADMILL` (the monitor CONFIRMED and MATERIAL) · `CONTINUE`.
- 🚨 **The SIZE BAR (F-ED-27):** a SUSPECTED / CONFIRMED check COUNTS only when the per-edge cyclic SD on its RECENT WINDOW (the
  last 8 checks' nodes) is MATERIAL (the one-sided 95 % lower bound above 1.7 pp); NEGLIGIBLE (upper bound below 1.7 pp) counts
  as NONE; the bar inside the interval counts as INCONCLUSIVE (`cycle_monitor.cyclic_size`; `status` prints `size` and
  `counted`). One check resolves sizes below ≈ 1 pp or above ≈ 2.5 pp (F-ED-28).
- **Ledger:** one `monitor_row` request per check in ONE family per run (`monitor:<run>:d<Δ>`, decision kind `cycle_flag`),
  purpose `monitor`, the node set frozen in its spec; a check is decided only when every row up to it is complete (`WAITING`
  otherwise); a SUSPECTED check plays a `monitor_topup` first; `python -m main.eval_ledger verify` re-derives each decision.
- **The panel:** a REGISTERED frozen checkpoint set (`designs/research_state/plateau_panels.json`: `outside_v2` = six end-state
  seeds, its one-member predecessor `outside_v1` stays registered), one `panel` request per (node, panel), purpose `plateau`, 500
  pairs per member (1,000 at a candidate: Tier 1 FLAT and the counted monitor NONE); Δ_panel in Elo, FLAT iff its upper
  one-sided 95 % bound < 14 Elo. Bots cannot be members (`main.h2h` plays checkpoints only); Metamon / Kakuna stay REPORTED
  milestones on `main.anchors`.
- 🚨 **Cost.** The toy ≈ 15–20 games/s on rule 5b's cores (≈ 10 min a row); the production net ≈ 1.2–2 games/s (≈ 1.5–2.5 h a
  row). Run it like Tier 1: offline, `nice` + `taskset -c 5-7,13-15` + `scripts/ops/mem_cap.sh` beside a live run, never on CPU
  beside an X5 A/B arm. The in-trainer window is not built (U4).
- **`ladder`** applies the same signals to `snapshot_ladder/games.jsonl` as of each check, plus the windowed size; `--tier1` adds a
  DESCRIPTIVE two-tier status: a free FIRST LOOK at a pinned run, never a decision, never the registered row.

### Eval on the Rust env core (the only trainer core, M5 Lane H — `agents/training/rust_eval/`)

An eval cycle is played on the M5 Rust env core (`rust_eval/`); the poke-env `main.eval_worker` processes it replaced were
deleted in poke-env retirement P6 slice 6c. **It keeps, exactly:** the plan (`ShardedEvalPool`: the nine roster bots, the
pool sentinels, the fixed / stable opponents, the same shard units), the regime (the trainee GREEDY; sentinels greedy on the
trainee's own teams under `eval_sentinel_greedy`, else a sample at `--self-play-temp`; fixed opponents greedy on their own
pinned teams), the forensic quota per unit with its LOSS-preferring selection, `eval_manifest.json` with its `selection`
record, the promotion thresholds and every metric, because the Rust cycle publishes the SAME per-unit `ShardResult` and the
callback's collect / record / promote code is unchanged. **What it changes:**

- **In process and BLOCKING.** The callback writes the plan and manifest, then `rust_eval.launch` plays the cycle between two
  host steps of the collector and collects it in the same `_on_step`. A failed cycle (`EvalCoreError`, e.g. a quarantined
  battle) is logged and records nothing; a lifecycle violation and a missing eval core (`RustEvalUnavailable`) are FATAL,
  never a silent fall-back to Python workers.
- **Everything is declared at startup** (`rust_eval.build`, from `main.train.rust_env_setup`): the eval CORE
  (`--rust-eval-envs`, default 64), the eval SLOTS in the trainer's ONE T2 service (the trainee's eval slot and one per
  `--n-sentinels` under `--self-play`, in the trainee's slot group, so eval rides the rollout's compiled buckets; a fixed
  opponent reuses its training slot or gets its own, loaded once by `rust_eval.build.load_fixed_slots`) and the eval ROUTE
  TABLE. A cycle only LOADS weights into declared slots, and every core and T2 `*_after_freeze` counter is checked after it.
- **Units play in order on one env**, so capture and the quota are decided game by game in plan order; an env without a unit
  plays FILLER games (nothing recorded) until the cycle ends.
- **Every game is seeded by the game** (`gen3_eval_game_seed_v1`, `rust_eval/seeds.py`): key = (cycle seed, opponent, game
  index in plan order) gives the two teams (one `yield_team` of the eval builders, `rust_eval.build.eval_builders`,
  re-seeded per game), the battle seed, and a scripted bot's streams (re-seeded by the core at the game's start; the bot
  route's `"streams": "episode"`). A game's result is a function of the game and the weights, not of the env, N, the
  threads or the schedule. The cycle seed is a hash of the collector's run seed and the step.
- **Traces are core traces** (`gen3_core_trace_v1`, `rust_eval/traces.py`): per kept game, both sides' `gen3_core_event_v1`
  records (`.p1/.p2.jsonl.gz`), the reconstruction record, the states npz (the core's obs rows, T2's legal log-probs as
  `logits`, V, the actions and masks) and a meta-only `_summary.json` that the prober EXPANDS on read
  (`main.prober.core_trace`). There is no `_replay.html` and the auxiliary heads are absent (`win_probs` NaN, belief / intent):
  `analyze` re-runs the model on the stored obs.
- `rust_eval/*` TensorBoard tags: the cycle's wall-clock, games, trainee decisions/s, traces, near-ties.

**The parity gate against the Python worker** (`rust_eval/parity.py`) was deleted with the worker; the lane's PROGRESS
(`measurements/m5_laneH/PROGRESS.md`) has the numbers and what they do NOT prove (F-LH-10), and `rust_eval/offline.py` keeps
the seeded-checkpoint and Rust-cycle helpers. Detail: `training_eval_and_rating_history.md` → "Eval on the Rust env core".

**Three OPT-IN overrides of `RustEvalCore.run_cycle`** (all `None` by default = the cycle above, byte for byte): `trainee_temp`
(`gen3_eval_trainee_sample_v1`: the trainee SAMPLES by the keyed draw on `STREAM_TRAINEE`) and `trainee_builder` (this
cycle's trainee team builder; its teams must be in the core's declared team table), both used only by the offline untaught
meter (`agents.training.untaught_rust`); and `seed_blocks` `(seeds, B)` (`gen3_eval_seed_blocks_v1`), which keys plan game g as
game g mod B of cycle seed `seeds[g // B]` (B even under a mirrored pool), so ONE cycle plays several head-to-head batches
whose games are each batch's own (the plateau meter's parallel batches: `main.h2h.play.H2HEngine.play_batches`).

### OFFLINE generation of an eval cycle (`main.ops.eval_trace_gen`) — and the PROVENANCE marker

A live eval cycle is sized for a training run: `--eval-games` games per opponent and a per-opponent OUTCOME QUOTA that
persists only ~200 traced battles a side. That is right for a run (the traces are a loss-forensics sample) and wrong for a
MEASUREMENT, because every conditioning and identity row an offline read computes is a statistic of that frame. When a read's
binding constraint is POWER rather than effect size, `python -m main.ops.eval_trace_gen <run>@<step> --games N --sentinels K
--out DIR` plays the cycle again from the saved checkpoint, offline and on CPU, at whatever size is worth paying for.

**It reuses this chapter's machinery**: the same `EvalItem` list and `ShardedEvalPool` plan, played on the same executor
(`RustEvalCore.run_cycle`) over an eval core and T2 slots declared the way the trainer declares them
(`rust_eval.offline.run_rust`), so the sentinel regime and teams, the forensic quota, the CORE traces and the
`record_eval_selection` collect are the code documented above. It supplies the WHAT: which checkpoint, how many games, which
pool snapshots, the compute (`--n-envs`, `--threads`, `--torch-threads`, `--front`, `--profile`; CPU, eager) and where the
output goes. A cycle generated by the deleted poke-env worker path carries no `transport` in
`generated_by`; the spec keeps it apart from a Rust-core one (`spec_of`'s `transport`).

**Five contracts make a generated cycle safe to read.**

1. **The REGIME is read, never assumed.** `eval_sentinel_greedy` is taken from the run's `model_config.json` (a run that
   recorded none is REFUSED); the mirrored-pair regime (`eval_mirrored_pairs`; an odd `--games` is then refused) and the
   observation mode (`oracle_reveal`) are read the same way. The wrong sentinel regime is worth **+8.9 pp** to the trainee.
2. **Capture defaults to ALL.** `--quota` restores a live-shaped outcome quota; the manifest's `selection_rule` says which.
3. **Nothing is written under `models/`.** The output is a self-contained SHADOW RUN DIR: the run's `model_config.json` and
   `metadata.json` copied (with `latest_eval.pool.sentinels` REWRITTEN to this cycle's sentinels), `snapshots/` symlinked back
   read-only, and the checkpoint copied to `eval_traces/step_<N>/snapshot.zip` (where the offline readers and the prober's
   `resolve_model_for_step` look). `--out` inside the run archive is refused.
4. 🚨 **The manifest carries a `generated_by` block, and a reader may not ignore it** (tool, schema, source run, checkpoint sha,
   games, bot list, `sentinels_requested` beside `sentinels_used`, capture rule, regime fields, `transport`, compute flags,
   the seed and whether it was given or drawn, the near-tie count, whether the cycle is REPRODUCIBLE, the POPULATION in
   words). A reader must REFUSE to form a delta between a generated frame and a live one (a quota match corrects a
   difference in capture RATE, not in the shape of the frame); two generated frames must agree on games / opponent set /
   capture rule / sentinel regime (a differing SEED is deliberately NOT checked). It records `battles_expected` beside
   `battles_played` and a `complete` flag, and a reader must REFUSE a side that is incomplete **or** whose completeness is
   UNRECORDED. Run a long generation from the MAIN checkout; a cycle cut short (`--timeout-min`) keeps its finished shard
   units and is recorded INCOMPLETE.
5. 🚨 **A generated cycle is CORE traces, so every reader that scores by the RECORDED win-prob head REFUSES it** (`win_probs`
   NaN, F-LH-4), as on every live Rust-core cycle; the calibration gauge (`scaffolding_gauge`) is such a reader. The prober's
   views and the shadow run's layout read it (`main/ops/eval_trace_gen_integration_test.py`).

**Sentinels are CLAMPED, never padded.** `--sentinels K` draws from the run's own `snapshots/`, evenly spaced across the step
range (both endpoints kept), excluding any snapshot at or above the read step (`--include-current-snapshot` opts in); a run
with fewer than K keeps what it has and records the gap (a duplicated cell would inflate the between-opponent SPREAD). The
chosen snapshot steps ARE recorded.

**Reproducibility, and its honest limit.** Every game is seeded by the GAME (`gen3_eval_game_seed_v1`): its teams, battle
seed, the bots' streams and a sampled sentinel's draws are a pure function of (`--seed`, opponent, game index). The same
seed, checkpoint, plan and compute flags therefore give the IDENTICAL cycle, file for file
(`eval_trace_gen_integration_test.py`); a run with no `--seed` draws one and RECORDS it (`seed_source: drawn`). What the seed
does NOT pin is the float rounding of the batched forward (its batch composition follows `--n-envs` and the schedule): a
greedy decision within a rounding error of a tie can resolve the other way under other compute flags, another env-core build
or another torch. The manifest records the compute flags and the near-tie count and states this in `reproducibility_note`.

### Rating-model seam (`rating.py`) — extensibility for Glicko-2 / TrueSkill

The live skill rating is anchored Bradley-Terry (`elo.py`), a *global batch* fit. `rating.py` is the drop-in point for a
different model: `MatchRecord` (exact counts + draws + `period_id` + optional opponent priors), `RatingResult`, a
`RatingModel` **batch** protocol, and `BradleyTerryRating`, a thin adapter over `elo.fit_pairwise` whose ratings+SE are
**byte-identical** to the live fit (`rating_test.py`); `eval_rows_to_match_records` bridges the `EvalRow` history. The live
`record_elo` path is **deliberately unchanged**: the seam exists and is tested, but routing through it buys nothing until a
new model is wanted, and Glicko-2 is *sequential*, so it needs a `SequentialRatingModel` sibling rather than the batch `fit`.
`eval_results.jsonl` carries exact per-opponent `counts`, so a future backfill has an exact ladder even under partial shard
coverage.


## ELO / skill rating (`elo.py`, `main.elo`; the bot anchor `data/gen3_bot_elo_anchors.json`)

Once training is mostly self-play **pool play**, win rate stops being legible: the promotion gate only promotes when
`win_rate_vs_pool > promote_threshold` and the pool is a *sliding window of recent selves*, so `win_rate_vs_pool` is a treadmill
pinned near 50-65% **by construction**, and `win_rate_vs_bots` saturates near 100%. The ELO subsystem gives a single
**absolute** number that genuinely rises with skill, anchored to the fixed bots.

- **No new battles.** Every eval cycle already plays the trainee (greedy) vs all 9 bots and vs up to `--n-sentinels` (default
  5) pool sentinels, `EVAL_GAMES` each: a full tournament-matrix row. `record_elo` (`eval_record.py`, shared by BOTH callbacks)
  appends it to an **append-only `<run>/eval_results.jsonl`** (`snapshot.append_eval_result_row`), the canonical, restart-safe
  source of truth, distinct from the overwritten `metadata.json:latest_eval`.
- **The model = anchored Bradley-Terry** (`elo.fit_elo`): `P(i beats j)=σ((Rᵢ−Rⱼ)·ln10/400)`, fit in **batch** by penalized MLE
  (a weak Gaussian prior keeps 100-0 records finite), SE from the inverse Hessian. Each bot is a player `bot:<name>`, each
  snapshot `snap:<step>`; a snapshot is the SAME player whether it appears as a cycle's trainee or later as a sentinel
  (unified by step), which links the whole ladder. Batch-BT (not online K-factor Elo) is drift-free and re-runnable. **Not
  Glicko-2**: snapshots are *frozen*, so the drift is the *sequence* of snapshots (the ELO-vs-step curve), and the
  per-player uncertainty is the Hessian SE.
- **Anchor = a precomputed bot-vs-bot round-robin**, `data/gen3_bot_elo_anchors.json`: the ratings, SEs, the 9×9 win matrix and
  a non-transitivity `fit_quality`, with the `git_hash` it was computed at (the Rustboro-era file; see *Bot evaluation*). It is
  the only runtime input. The live/offline fits **pin all 9 bots** to it and fit only snapshots, so a snapshot is
  well-grounded from its first cycle and, because the anchor is identical across runs, **snapshot ELOs are comparable
  run-to-run**. **Regenerate when bot logic changes.** Graceful fallback when the file is absent: `random` pinned at `base`,
  other bots float (rank/trend preserved, scale not cross-run-stable). The Python calibration that produced the first anchor
  (`bot_elo_calibration`, and the raw-matrix tool `bot_matchup_matrix`, whose `data/gen3_bot_matchups.json` stays) was DELETED in
  poke-env retirement P6 (2026-10-08); the Rustboro anchor came from `measurements/bot_base_ratings_2026-10-03/`, and
  regenerating it is an owner decision because it re-scales every bot-anchored headline. The original calibration's
  mechanics: `training_eval_and_rating_history.md` → "ELO / skill rating".
- **Live (each eval cycle).** `record_elo` refits and records `eval/elo` + `eval/elo_ci` (95% CI half-width) to TensorBoard + the
  TUI dict and stamps `elo`/`elo_ci` into `metadata.json:latest_eval` (a checkpoint that predates the field is re-fitted by
  `replay_last_eval_to_tui` via `elo.fit_from_block`). The launcher shows a `🏅 ELO 1532 ±40` badge (`app.py::_elo_badge`) and
  an `elo` column in the eval panel. The live number is the best estimate from data SO FAR (batch-BT is global, so early
  points retro-adjust; the single-cycle per-sentinel ELO is rough, only the trainee is bot-anchored each cycle); the offline
  CLI re-fits canonically over the full per-snapshot history.
- **Offline (`python -m main.elo <run_dir>`).** Loads results (`--source auto|log|tb|meta`; `tb` **backfills an already-running
  run straight from TensorBoard**), fits, prints a ranked ladder and writes `elo_ratings.json` + `elo_curve.png` under `--out`
  (default `<run>/elo/`). 🚨 With no `--out` and the run LIVE (`utils.procfs.live_training_runs`) it REFUSES (`LiveRunDirError`,
  exit 3, naming `--out <dir>`); `--allow-live-run-dir` overrides (`src/main/elo_live_guard_test.py`).
- **The two-regime caveat is CLOSED for new runs** (`gen3_eval_sentinel_greedy_default_v1`): **sentinels are GREEDY and draw
  the trainee's own teams by default** (`--no-eval-sentinel-greedy` restores the old greedy-trainee-vs-sampling-sentinel
  regime), so every snapshot is scored in one regime. 🚨 **The cost is a ONE-TIME DISCONTINUITY in `eval/elo` and
  `win_rate_vs_pool` against every v9-era run** (the asymmetric edge read **+8.9 pp [+7.0, +10.7]** in the trainee's favour,
  in the SENTINEL edges only: the bot-anchored scale is preserved). A run's regime is recorded (`model_config.json` config
  v112) and INHERITED on a flagless resume; see [`self_play_and_pool.md`](self_play_and_pool.md) for the resolution rule and
  the startup line. Tests: `elo_test.py`, `main/train/eval_sentinel_regime_test.py`.

### Frozen-snapshot ELO ladder — the dense, pay-once resolution (`snapshot_ladder.py`)

The live ELO above is RESOLUTION-limited at the frontier: the fixed bots have SATURATED (a 10-Elo trainee move shifts its
bot-WR by ~0.5% against a 1.9%/200-game noise floor), so the bots pin the absolute LEVEL but the fine ordering rides on the
sparse, near-50% sentinel edges (±15 Elo CIs). Fix from the other side: a promoted snapshot is FROZEN, so snapshot-A-vs-snapshot-B
is a STATIONARY Bernoulli, measured ONCE (dense round-robin) and permanent. On each promotion,
`SelfPlayCallback._spawn_snapshot_ladder_update` fires a **DETACHED** `python -m agents.training.snapshot_ladder <run> --promote
<step>` subprocess (CPU, off the training path; every game on the **Rust eval engine** since 2026-10-06, see "The TRANSPORT
boundary" below) that plays the new frozen node vs the current frozen pool: **200 FRESH games vs each sentinel the promoting
eval cycle used** (`source: "promotion_baseline"`, recipe v3) and `--snapshot-ladder-games` (default 100/pair; 0 disables the
whole update) vs every other frozen node, and appends to `<run>/snapshot_ladder/games.jsonl` (**forever, race-safe line
appends; a measured pair is NEVER replayed**). `fit_ladder` combines that dense frozen-vs-frozen matrix with each snapshot's
historical BOT edges from `eval_results.jsonl` (the anchor connection, **bot edges only**: see the sentinel exclusion below)
into an anchored BT fit (`fit_pairwise`, bots pinned) written to `<run>/snapshot_ladder/ladder.json`; `_record_ladder_elo`
surfaces the latest promoted node's rating as `eval/ladder_elo` (+`_ci`) on TB/TUI, the high-resolution counterpart to the
saturated `eval/elo`. Snapshots load through the strict loader with each zip's historical kwargs sanitized
(`main.h2h.arch._load_host`). `--backfill` pays the one-time back tax over the whole current pool (idempotent);
`--fit-only` refits without playing. `ladder.json.fit_quality.mean_abs_err` QUANTIFIES non-transitivity (see *Hodge*). Tests:
`snapshot_ladder_test.py`, `snapshot_ladder_transport_test.py` (the transport boundary) and
`snapshot_ladder_rust_integration_test.py` (an edge played end to end on the Rust engine, `sim` + `integration`).

#### 🚨 The TRANSPORT boundary — the ladder plays on the Rust eval engine (poke-env retirement P2, 2026-10-06)

**What changed.** Until 2026-10-06 every ladder edge was played by two poke-env `RLPlayer`s over the in-process bridge on the
PYTHON encoder, while every trainee since the M5 cutover trains on RUST rows. Every NEW edge is now played by
`agents/training/snapshot_ladder_play.py` on the head-to-head meter's engine (`main.h2h.play.H2HEngine`: one T2 service, one
Rust eval core, CPU eager, every slot load byte-checked), protocol `gen3_ladder_rust_v1`: the Rust encoder; both sides greedy
on the default pool with `n_games // 2` MIRRORED team pairs; SEAT-BALANCED (`ceil(P/2)` pairs with `a` as the player,
`floor(P/2)` with `b`); a draw is recorded (`draws`) and EXCLUDED from the edge (`games` = decisive games); seeds a pure
function of the two content hashes and the edge's batch index (a deliberate re-measurement takes a NEW batch); a pinned-team
run plays the same default pool (the imported checkout's: inside a pinned run the pin's own `data/teams`,
`gen3_pinned_data_root_v1`). The Python ladder (`python_bridge`, every row before 2026-10-06) played the Python encoder with
`a` (the NEW node) on p1 in every game, independent draws, a draw folded into `b`'s column and the unseeded global `random`
stream (side-by-side table: archive).

**THE DECISION: a REGIME BOUNDARY, like 2026-09-07's.** Even a zero transport shift leaves the protocol changed (the seat effect
no longer credited to the newest node; draws no longer `b`'s), so nothing may fit or compare the two as one:

- every `games.jsonl` row carries `transport` (absent = `python_bridge`) plus the play stamp (`encoder`, `protocol`, `draws`,
  `by_seat`, both seats' pentanomials, the schedule key and cycle seed, `regime_id`, `core_stamp`, both content hashes);
- `fit_ladder` REFUSES a fit whose kept pairs span both transports (`LadderTransportError`, naming the fix): `--transport
  rust_eval` fits only the Rust rows, `--allow-mixed-transport` fits both, stamped `mixed`;
- `ladder.json`'s `recipe` block carries `transport` (`snapshot_ladder.ladder_transport`) as PROVENANCE only
  (`LADDER_FITTER_VERSION` stays 3);
- a reader that compares two ladders calls `snapshot_ladder.check_same_transport`, which REFUSES a pair played on different
  transports (or a side whose log mixes them) unless told; `main.elo` and `main.ops.plateau_signal` print the transport
  beside the number, and `hodge.edges_from_run` states a mixed log as a caveat;
- `missing_pairs` counts a pair measured only by the Python ladder as MISSING, so `--backfill` on a pre-boundary run
  re-measures every pair on Rust (then fit with `--transport rust_eval`).

A run pinned before the boundary keeps writing Python rows; only a run launched (or re-pinned) at or after it writes Rust
rows. The engine's shape is `--n-envs` (32) / `--threads` (2) / `--torch-threads` (2); the old `--impl` / `--concurrency` are
accepted and ignored.

**The measured shift** (N0 `ai_v14_01_base`, 16 frozen pairs × 100 games per stack, CPU,
`measurements/pokeenv_p2_ladder_2026-10-06/`): the edge as the Rust ladder fits it moves the mean edge **+0.99 pp [−2.46,
+4.43]** against a same-stack floor of **−4.00 pp [−8.58, +0.58]**: **NOT DETECTED**, and not equivalence (no registered bar),
so a mean shift of a few pp (≈ ±25 to 35 Elo per edge) is not excluded. The seat effect, which the Python ladder credited to
the newest node on every edge, is **u = +0.56 pp** (not detected).

🚨 **THE EVAL CYCLES' SENTINEL EDGES ARE EXCLUDED FROM THE LADDER FIT** (2026-09-07). `elo._rows_to_results` yields TWO
families off one eval row, trainee-vs-bot (`bot:`) and trainee-vs-**sentinel** (`snap:` vs `snap:`); `fit_ladder` folds in
only the first. The second is a **different measurement of the same frozen pair** the dense matrix already holds (in the
regime of that date a GREEDY trainee played a STOCHASTIC sentinel with an asymmetric teambuilder, while the ladder plays
greedy-vs-greedy), and mixing them favoured the NEWER snapshot by **+8.9 pp [+7.0, +10.7]** and inflated the newest nodes by
**+21 to +29 Elo**. `games.jsonl` is therefore the ONLY snapshot-vs-snapshot source, and the excluded count is written as
**`eval_sentinel_edges_dropped`**. Every ladder written before 2026-09-07 folded those edges in.

##### 🚨 THE RECIPE STAMP — a rating is only comparable to one fitted the same way

The excluded COUNT is not enough (`0` is indistinguishable from "fitted by a tree that did not drop them"), so `fit_ladder`
writes a **`recipe` block**:

```json
"recipe": {"name": "gen3_ladder_recipe_v1", "fitter_version": 3,
           "eval_sentinel_edges_dropped": true, "eval_sentinel_edges_dropped_count": 37,
           "eval_cycle_pair_edges_used": false, "eval_cycle_pair_edges_dropped_count": 0,
           "promotion_baseline_games": 200,
           "commit": "<40-char HEAD of the tree that fit it>"}
```

`eval_sentinel_edges_dropped` is the **policy** (a boolean); the count beside it describes the run's eval history.
`LADDER_FITTER_VERSION` is **bumped whenever the fit changes what a rating MEANS** (a new edge family, a dropped one, a
different anchor set); a bump makes every previously committed file read `differs`, loudly.

`snapshot_ladder.recipe_status(doc)` returns `current` / `absent` / `differs` (a pure function, so every reader asks the same
question), and `check_recipe` raises **`LadderRecipeError`** carrying the refit command and the size of the error it
prevents. **Why a refusal and not a label:** `ai_v12_02_winprob_critic`'s committed file reads **2057.3** at its newest node
where the current recipe refits the same 20 nodes to **1984.2** (+73.1 Elo, which FLIPPED THE SIGN of a cross-run delta;
`flywheel_armS_reads_2026-09-14/` §2.1). A label on a number nobody can convert is not a safeguard.

| reader | what it does with a stale stamp |
|---|---|
| `main.ops.plateau_signal` | names a stale file as stale and points at `--fixed-fit` |
| `snapshot_ladder.latest_promoted_elo` | **deliberately does NOT check it**: a WITHIN-RUN trend scalar (`eval/ladder_elo`) written by the run's own pinned code; no cross-run comparison is made |
| `agents.training.snapshot_ladder` CLI | prints `[ladder] recipe: …` beside every table |
| `main.elo <run>` | prints a `[ladder]` HEADLINE line and **WITHHOLDS the dense number** on a stale stamp, printing the refusal in its place |

Pinned by `src/agents/training/ladder_recipe_test.py` and `src/main/elo_refit_test.py`.

###### Putting a committed ladder BACK on the current recipe — `main.elo refit`

```bash
python -m main.elo refit <run_dir>            # read-only: committed vs refit, node by node
python -m main.elo refit --apply <run_dir>    # write the stamped fit, KEEP the old file
```

Both **refit and play NOTHING** (`games.jsonl` is append-only and never stale). The fit is over **the committed file's own node
set** (`--pool` uses the snapshots on disk instead): BT re-solves every node on every add and the newest node is
systematically inflated, so a fit over a different node set would conflate the recipe change with a node-set change.
`--apply` keeps the file it replaces as **`snapshot_ladder/ladder.pre_recipe.json`** (`ladder.pre_recipe_vN.json` for an older
fitter version N), the only surviving evidence of what a banked number was quoted from, and REFUSES rather than overwrite one
that is already there; on an already-current identical file it is a no-op. `--fit-only` on the `snapshot_ladder` CLI does the
same fit but rewrites in place, keeping nothing.

**The whole archive was audited this way on 2026-09-22** (`designs/research_state/measurements/ladder_refit_audit_2026-09-22/`):
of 93 runs, **68 move and 25 reproduce to 0.0**, the split exactly the `3e6875a5` boundary; the median max |Δ| across a
mover's nodes is **53.4 Elo** (largest **255.6**), **65 of 68 move downward**, and the argmax node moves on 33 runs. Early nodes
RISE and late nodes FALL. Per-run shifts: `training_eval_and_rating_history.md` → "Putting a committed ladder BACK".
**The per-cycle `eval/elo` star fit (`elo.fit_from_run` / `record_elo`) is UNCHANGED and still uses the sentinel edges**: it
has no dense matrix to prefer, and those near-50% edges are the only resolution the saturated bots cannot give (removing them
from one cycle moves the trainee −110 Elo and WIDENS the CI ±30 → ±41). The exclusion is the LADDER's alone, and
`snapshot_ladder_test.py` pins that separation with a test on `fit_from_run`.

#### The games that SELECTED a snapshot never RATE it — recipe v3 (owner decision 2026-09-27)

**What v2 did (2026-09-07 → 2026-09-27) and why v3 stops.** v2 REUSED the eval cycle's sentinel pairs as ladder edges
(tagged `"source": "eval_cycle"`). Promotion is DECIDED by those very games (win rate vs the
pool ≥ the threshold), so rating a snapshot on the sample that selected it is a **winner's curse**, about **+15 to +40 Elo at
n = 100** (analytic): a new node reads high and drifts down.
So under `gen3_ladder_recipe_v1` **v3**:

- **Each promotion plays a FRESH 200-game pair vs every sentinel the promoting eval used** (`PROMOTION_BASELINE_GAMES`; the
  sentinels are read from that step's `eval_results.jsonl` rows in ANY regime: selection, not protocol, causes the curse),
  tagged `"source": "promotion_baseline"`. A sentinel groomed out of the pool is named and skipped; a hand `--promote` with no
  eval row plays everything as round-robin.
- **Every other pair is the ordinary round-robin** at `--snapshot-ladder-games` (100), never replayed. The promotion decision
  itself is unchanged.
- **The eval games stay in `eval_results.jsonl` only.** `load_games` skips every `eval_cycle` row a v2 tree left behind, so
  such a pair reads as UNMEASURED; `eval_measured_pairs` survives as the read of what the eval measured. Every new
  `games.jsonl` row carries `"recipe_version": 3`.
- `update_for_promotion` prints `FRESH GAMES THIS PROMOTION: N` (`--promote <step> --dry-run` prints the plan without playing).

**Converting a v2 ladder.** A v2 file reads `differs` everywhere the recipe is checked. `python -m main.elo refit <run>`
already DROPS the eval-cycle pairs (it plays nothing) and names how many pairs carried ONLY such rows and what their fresh
replacement costs; `python -m agents.training.snapshot_ladder <run> --backfill-fresh [--dry-run]` plays exactly those pairs
at 200 games (idempotent; `games.jsonl` is append-only, the old rows are never deleted). A v2 run whose log holds no
`eval_cycle` rows refits to IDENTICAL ratings. **Scope:** runs LAUNCHED on the v3 code; a run pinned to an older tree keeps
writing v2 stamps, which every main-tree cross-run reader flags as stale. Tests: `snapshot_ladder_test.py`,
`ladder_recipe_test.py`.

#### The RELATIVE column — Elo above a pinned frozen reference node (2026-09-27)

`ladder.json` carries a SECOND rating column beside the bot-anchored headline: **`ratings_relative`** (+ `se_relative`) = each
node's Elo **above the reference node** (`reference.step`, pinned at 0), fitted by BT over the **frozen-vs-frozen `games.jsonl`
edges ALONE** (no bot edges, no eval edges, a weak prior of SD 2000 that only keeps a sweep finite). The reference defaults to
the registry baseline **`untaught_meter_opponent_v14`** when the run's ladder holds THAT FILE (its own run, or a matching
sha256), else the run's **first** snapshot; `--reference <step>` overrides; nodes with no frozen path to it are omitted.
**Why:** a frozen pair is measured once and never changes, so this column moves only when a new edge says something about an
existing node (adding a stronger node leaves the others' relative Elo unchanged on a transitive matrix), whereas the
bot-anchored level re-solves against bot edges. **The headline is still the bot-anchored `ratings`**; the relative column
answers "how far above the pinned reference", within a run or across runs sharing that reference file. The `snapshot_ladder`
CLI table, the `main.elo` headline line and the `main.elo refit` table print it.

### Hodge decomposition — the SPINE and the WIDTH (`hodge.py`)

A scalar rating is a **transitive** model by construction, so a BT fit cannot see a cycle: two snapshots with identical ELO
can have a lopsided head-to-head. `ladder.json`'s `fit_quality.mean_abs_err` NOTICES the residue but has **no noise floor**,
so it cannot say whether the non-transitivity is real or binomial noise on 100-game edges. HodgeRank answers it by splitting
the measured flows

```
Y_ij = logit(p_ij)   =   (r_i − r_j)   +   R_ij        w_ij = n_ij·p_ij·(1−p_ij)
                          ───────────       ────
                          TRANSITIVE        CYCLIC     (Fisher info of a logit = the weight)
```

where `r` is the weighted-least-squares (graph-Laplacian) solve, reported BESIDE the BT ratings so the estimators'
disagreement is visible. The split is **exactly w-orthogonal** (`Σw·Y² = Σw·(rᵢ−rⱼ)² + Σw·R²`, pinned by a test). Units: 1 logit
= 400/ln10 ≈ **173.72 ELO**.

**The noise floor is the whole instrument.** Two nulls, both reported: an exact-mean analytic one (`E[Σw·R²] = Σ(1 −
w_e·Reff_e)`) and a **parametric bootstrap** that simulates games from the fitted transitive model and re-runs the whole
pipeline. `width_rms_excess = √(raw² − null²)` is the width that survives, with a p-value for "width > noise" (floored at
`1/(B+1)`). **Width SCOPE:** a pendant edge's residual is identically zero, so width statistics default to the
**triangle-supported subgraph** (the spine is always fit over every edge); `n_triangles` + `n_width_edges` ride with every
read.

- **Offline — THE instrument.** `python -m main.elo <run>` prints the block and writes it into `elo/elo_ratings.json` under
  `hodge` (flags: `--no-hodge`, `--hodge-bootstrap N`, `--hodge-seed`, `--hodge-with-bot-rr`). The graph is `fit_ladder`'s: the
  dense frozen matrix + every cycle's bot/sentinel edges; the static bot round-robin is **excluded by default** (its 36 edges
  carry ~2700 games each against a ladder edge's 100, so the "width" would describe the immutable anchor, not this run).
  `main.endofrun`'s §1 block carries the same read for the run and its `--ref`.
- **Live — two scalars beside `eval/elo`** from `record_elo`: `eval/hodge_width_elo` (excess width, ELO) and
  `eval/hodge_cyclic_fraction` (null-adjusted). Both ride in the `eval_results.jsonl` row's `hodge` block, and a cycle whose
  graph had **no testable triangle** writes `recorded: false` + a reason and records NOTHING to TB (never 0-as-a-stand-in,
  never NaN).

⚠️ **THE STAR-GRAPH SUBTLETY — read this before quoting a live width.** A cycle's own new games are a **star** (trainee vs each
opponent); a star is a tree, so a width computed on the cycle's games alone is *identically zero*, a fake instrument. The
triangles come from joining the trainee's edges to the **static bot-vs-bot round-robin** in `data/gen3_bot_elo_anchors.json`
(which ships the raw 9×9 `win_matrix` + per-pair `pair_games`; an anchor carrying only `ratings` falls back to reconstructed,
transitive edges flagged in `caveats`). So the live metric means exactly **"the trainee's matchup deviation from its own
rating, over trainee×bot×bot triangles"**, nothing about the pool's width; sentinel edges are in the FIT but on no triangle.
And the live read is **weak by construction** (~100-game edges put the noise floor around 35-60 ELO). **The offline
dense-ladder read is the real instrument; the ELO-reading rules below apply unchanged: never narrate a mid-run width.**

**First reading** (gen-15 and gen-14 dense ladders, 300 bootstrap reps): both are 96 to 98 % spine and both carry cycle content
that is **not sampling noise** (excess width 46 ELO, p = 0.005, and 26 ELO, p = 0.0033), the first evidence that the BT gate
is a lossy projection by a *measured* amount. Detail: `training_eval_and_rating_history.md` → "Hodge decomposition". Tests:
`hodge_test.py`.

#### 🚨 Reading an ELO: `ladder.json`, at matched SNAPSHOT COUNT, never mid-run

**Read `<run>/snapshot_ladder/ladder.json` — not `eval/elo`, not the per-cycle TB scalar.** On
gen-10's completed ladder the two agree at the end (24M: dense 2079 vs sparse 2102) but the dense
CI is **±10 vs ±29**. Precision is the reason to prefer it; it is *not* immune to the drift below.

**A snapshot's rating keeps moving until it stops gaining opponents.** Anchored BT is a GLOBAL BATCH fit (every added player
re-solves every rating) and the movement is a **systematic downward bias on the newest node**, not noise: over gen-10's 12
successive refits a node fell 44 to 101 Elo between its first and final fit.

Mechanism: a fresh snapshot's only edges are ~90% wins over the bots, and a saturated edge says *"≥380 Elo above"* with a
likelihood that is **flat upward**, so the MLE is inflated; the near-50% frozen-vs-frozen edges are sharply informative and,
as they accumulate, pull the chain down onto the anchor. Dense measurement buys resolution, not an early answer.

**Therefore, five rules** (2-5 are the matched-count read):

1. **Never narrate a mid-run ELO or a mid-run delta.** (gen-10's 12M delta read +108, +82, +73, +64 before settling at +11.)
   Wait for the run to end.
2. **Cross-run comparison must be at matched snapshot count `n`, not matched step.** Both runs' node at n=k carries the same
   inflation, so it cancels; a live run's newest node against a finished run's *final* value does not (a +38 read that was
   the drift and nothing else; the matched n=7 read was −6, tied).

`n_frozen_pairs_measured` / `n_pairs_possible` in `ladder.json` is the completeness check — a fit
at 21/21 pairs is internally complete but only 7 nodes deep, and depth is what the bias tracks.

3. **Matched COUNT means matched FIT SIZE: REFIT, never read the comparator's n-th node off its FINAL fit.**
   `snapshot_ladder.fit_ladder(run_dir, first_n=n, write=False, steps=…)` is the refit: every frozen pair AND every
   bot/sentinel edge whose snapshot endpoints all lie in the prefix, **strict** (an edge to a node outside the prefix
   re-imports exactly the future the rule excludes). The first measured case read a **30-Elo TRAIL as a 64-Elo LEAD**. A
   ladder that cannot be refit (no `games.jsonl`, or a FORK's ladder whose early nodes are rated only through edges to its
   own late selves) is reported at **UNMATCHED FIT SIZE** with that label on the number, and a read built on this must REFUSE
   it rather than print a lead.

4. **A committed `ladder.json` is not evidence: REFIT BOTH SIDES from `games.jsonl`.** A run's `ladder.json` was written by the
   code that run is PINNED to, and every ladder written before 2026-09-07 folded the eval cycles' greedy-vs-stochastic
   **sentinel** edges into the fit (+21..+29 Elo on the newest nodes; see the ladder section above). A cross-run read
   therefore refits **both** sides with THIS tree's `fit_ladder` whenever `games.jsonl` exists, even at already-matched node
   counts, LABELS any side that had to fall back to its committed ladder, and **REFUSES that fallback outright when the
   committed file's `recipe` stamp is absent or differs** (see the recipe-stamp section above; +73.1 Elo on one measured
   run, enough to flip a delta's sign).

5. **The COUNT is over every OWN snapshot EVER rated, from `games.jsonl`, never the committed file's nodes.** The pool is
   GROOMED to `SnapshotPool.DEFAULT_MAX_SNAPSHOTS` (20; no flag sets it) and every refit rates only the SURVIVORS, so a closed
   run's `ladder.json` is its last 20 and "the arm's first 12" would read a different slice at run end; a FORK's ladder also
   rates the parent snapshots its pool was seeded with (at or below its `fork_step`). `snapshot_ladder.own_rated_steps_ever`
   (the append-only log's endpoints + the committed nodes, above the lineage record's `fork_step`; pinned by
   `snapshot_ladder_test.py`) is the counting basis, and `fit_ladder(first_n=n, steps=<that list>)` the strict-prefix refit.
   A read must REFUSE a side with NO `games.jsonl` whose eval record names a pool snapshot its committed file no longer rates
   (the first n cannot be known). 🚨 A hand read off `ladder.json`'s n-th node is NOT a matched-count read on a groomed or
   forked run: `python -m main.elo compare <run_a> <run_b> […]` composes those two calls and applies rules 2-5 (first run =
   reference; both sides refit on the first N own-rated snapshots; a log-less side is read from a CURRENT committed file only
   when it rates exactly N, else refused; a prefix with no inner edge, a non-converged fit, a transport mix and an unanchored
   fit are refused; a cross-era pair and an unfinished run are labelled). It plays and writes nothing.
