# Training — the eval-side rules (forensic traces, the baseline registry, the in-loop eval, ELO)

> Lifted out of `src/agents/training/CLAUDE.md` on 2026-10-10 (the leaf keeps the rules an agent must know before
> touching these subsystems and points here). Always-current, like the leaf: update it in the same pass as the code.
> The longer narrative is [`eval_and_rating.md`](eval_and_rating.md).

## Faint attribution in the trace (`gen3_faint_attribution_v1`)

The trace recorder names the newly-fainted species by a SET DIFFERENCE over the two snapshots'
`*_fainted_species`, never by labelling a count change with the mon that was active at the
DECISION — and it emits one event per species, because **one side can lose two mons in a turn**.
The recorder is the prober's `main.prober.core_recorder` over the Rust core's walk (the Python `BattleRecorder` and
its protocol-validated fuzz gate were deleted in T27 P6 slice 6d-2); its labels are `trace_labels.py`, shared.
⚠️ **A protocol identifier carries the NICKNAME, not the species** — this pool holds teams with
localized nicknames (`Triopikeur` = Dugtrio), so any protocol-vs-our-data comparison must resolve
identifiers through the side reader's own-team map (formerly poke-env's `battle.team`). ⚠️ **A forensic recorder must never take down a
run**, which is why a mis-read falls back to a slightly-wrong label rather than raising. The
measured defect and its revert numbers are CLOSED history:
`designs/research_state/claude_md_archive/training_leaf_faint_attribution_history.md`.

## THE BASELINE REGISTRY (`baselines.py` · `designs/baselines.json` · `python -m main.baselines`)

**A baseline is the thing a result is read AGAINST, and this module is the ONE accessor over the
named set** (`gen3_baselines_registry_v1`). 🚨 **Read a baseline BY NAME, never by copying a path**
— `production`, `v9_long_baseline`, `v9_fold_parent`, `famine_comparator`,
`untaught_meter_opponent` and friends. Torch-free and offline; `resolve()` is the only call that
touches `models/`.

```python
from agents.training import baselines
baselines.get("v9_fold_parent").spec        # "ai_v9_59_R2ACTION_0827/final_model.zip"
baselines.resolve("famine_comparator")      # through fixed_opponent_pool.resolve_model_ref
baselines.protected_files()                 # {run: [rel path]} — the grooming keep-list
```

🚨 **Every entry is EXPLICIT** (a `.zip`, a `.json` or an `@step`, never a bare run dir), so the
last-snapshot rule below cannot move what a name points at while its run keeps training.
🚨 **A NEW OPPONENT IS A RE-MEASUREMENT, NOT A RENAME** — untaught-meter levels are not comparable
across opponents, so `python -m main.baselines set <name> <file> --reason "<ledger title>"` is the
only legal edit, and it PRINTS the ledger line to append rather than writing one.
🚨 **The untaught meter's default opponent is `untaught_meter_opponent_v14` (INTERIM until the Rustboro opponent, legacy manifest D-L3) and its default config is `auto`** (B3, 2026-10-04: the old v101 defaults did not load at HEAD). 🚨 **Since the X5 version break (v144) that default is `era_checkout_only` (below `MIGRATION_FLOOR`), so a bare launch REFUSES at the default, naming `--opponent <a v144+ checkpoint>` (a NEW series) or a run PINNED to ≤ `f7567a9f` for the v14 series** — never a bare floor error at the first model load. `main.critic_gate`'s endpoint 3 passes one through `--meter-opponent` (its `--check` reports the meter's refusal without it). A played artifact stamps `_meta.series`; `--from-rows` REFUSES artifacts that recorded different opponents (`--allow-opponent-mix` consents). Its games run on the Rust eval core since P2 (2026-10-06, `untaught_rust.py`; the executor's opt-in `trainee_temp` / per-cycle `trainee_builder`); every artifact stamps `_meta.transport`, and the readers refuse a `rust_eval` / `python_bridge` mix (`--allow-transport-mix` consents). Detail: `designs/training/eval_and_rating.md`.
🚨 **LOAD a baseline with `baselines.load(name)`, NEVER a bare `MaskablePPO.load`** — the bare path
rebuilds the extractor from the zip's own pickled kwargs and (measured 2026-09-22) raises
`unexpected keyword argument 'threat_prob_outspeed'` on **all five** current-generation entries,
which is how a 2026-09-14 read concluded "arch drift" and silently substituted a stand-in
checkpoint. `load()` uses the sanitizing `load_foreign_opponent` and either returns a model or
raises **`BaselineLoadError`** with `.reason` (`pre_generation` · `arch_drift` · `unresolvable` ·
`not_a_model`) and a message naming the FIX. `era_checkout_only` is VALIDATED against the entry's
recorded generation, so an unmarked pre-generation node is an error. `python -m main.baselines
check --load` runs it for real.
**Full detail — in [`designs/training/eval_and_rating.md`](eval_and_rating.md).**

## Bot evaluation (in process on the Rust eval core, BLOCKING)

**Flat schedule, full roster, blocking.** Eval fires every `EVAL_FREQ_STEPS` (2M) for
`EVAL_GAMES` (100) games per opponent (`--eval-games N` overrides), uniformly over the eight
archetype bots plus `random` and every self-play sentinel — no maturity tiers, no per-opponent
caps, no roster flag. 🚨 **The cycle plays in the trainer's own process between two host steps of the
collector and is collected in the same step** (`eval_callback._launch_eval` →
`eval_launch.launch_rust_eval_cycle` → `_collect_pending`): ~1.5% of wall at N=256
(`designs/research_state/measurements/m5_sizing/PROGRESS.md` O9). There is no worker pool, no skipped
cycle, no hung-cycle watchdog and no drain (the Python worker branch, `--eval-workers` and
`PerOpponentEvalCallback.drain` were deleted in P10-F2); a stop signal is honoured INSIDE the cycle at its
safe points (`safe_point_fn`). The standalone Python eval oracle that outlived it (`main.eval_worker`,
`eval_launch.spawn_eval_workers`, `EvalRLPlayer`, `rust_eval.parity`, `eval_benchmark`) was deleted in poke-env
retirement P6 slice 6c: every eval cycle, the offline ones included, plays on the Rust eval executor. An OFFLINE
caller declares its own eval core + T2 slots through `rust_eval.offline` (`run_rust`; `build_models` for seeded
perturbed checkpoints) — `main.ops.eval_trace_gen` generates a cycle for a saved checkpoint that way.
🚨 **A cycle's retained `eval_traces/step_<N>/snapshot.zip` is a HARD LINK to the same-step checkpoint** when that is
byte-identical (sha256) on the same filesystem, else a copy (`eval_collect.store_eval_snapshot`; the manifest's
`snapshot_storage` says which): the weights are stored once, the link survives the checkpoint's deletion, and the file
is replaced atomically so a re-persist never writes THROUGH a link into a checkpoint. Detail:
`designs/training/eval_and_rating.md`; pinned by `eval_snapshot_dedup_test.py`.

🚨 **THE FORENSIC TRACE'S RESULT VOCABULARY is `WIN` | `LOSS` | `DRAW`** (`gen3_trace_result_v2`,
`trace_result.py`); a `DRAW` carries `meta.draw_kind` (`timeout` vs `tie`), and an unknown result is
REFUSED rather than coerced. A timeout arrives wearing a LOSS's flags, which is how it went
unnoticed for the whole archive. 🚨 **The capture quota PREFERS LOSSES and draws have their OWN
bucket** — a trace tree is a loss-enriched sample by design, each cycle's manifest states the rule
in words, and a tree that records none is SELECTION UNKNOWN, never uniform.
🚨 **MIRRORED TEAM PAIRS are a REGIME (`--eval-mirrored-pairs`, T17, config v128, DEFAULT OFF).** Each
team pairing is played from both sides on one battle seed (`rust_eval.seeds.pair_game`, both eval
cores), counts are even, and every interval is the PAIR-level pentanomial one (`mirrored_pairs.py`) —
never per-game. Recorded in `model_config.json` and `_resolve`-inherited like `eval_sentinel_greedy`;
the row's `mirrored_pairs` block is the stamp, and `elo.load_rows` REFUSES a run whose rows span it.
Do not flip the default: the M5 sizing arms are compared across it; the orchestrator flips it at the
X26 baseline. 🚨 **`--promotion-sprt` (T6, config v129, DEFAULT OFF) is its promotion twin**: each
eval-cycle snapshot is a candidate decided by a pentanomial GSPRT on fresh mirrored pairs vs the pool
frozen at its launch (H0 0.50 / H1 0.55, α = β = 0.05, cap 1,680 pairs = reject; `sprt.py`,
`sprt_promotion.py`), the cycle's own pool games never enter it, and `sprt_promotion.jsonl` makes a
failed or interrupted test un-rerunnable. Eval-only fields like this one are RECORDED, `_resolve`-inherited and never compared by
`check_compatible` — they are not `flag_registry` rows (that registry declares extractor toggles).
🚨 **THE EVAL COUNT LEDGER is `eval_ledger/` (v2, `gen3_eval_count_row_v2`; eval U1).** Archive-level at
`<archive>/_ledger/`; every batch row written under a CLAIM for a REQUEST (`LedgerWriter`); every consumer reads through
`eval_ledger.read` with a spelled-out `ReaderDecl` (purposes, regime, requests, selection, flags_ok, inference) — the static
gate `src/eval_ledger_reader_gate_test.py` (EMPTY allowlist) fails anything else, and every read refuses a duplicate batch
or seed block. v1 rows are upgraded on read, never rewritten. `python -m main.eval_ledger audit` checks it.
🚨 **THE LEDGER'S COST IS FLAT IN THE ARCHIVE'S SIZE (F-ED-22 fixed; `event_index.py`, `row_index.py`, `incremental.py`).** A
claim / append folds only the requests-stream bytes beyond each file's cursor, through the same `queue.apply_event` the full
`queue.fold` is made of, under the one file lock; a request- / family-scoped `read` costs that request's rows. Both ride
persisted SQLite indexes in `<ledger>/.ledger_index/` — CACHES, git-ignored, safe to delete: rebuilt from the append-only
streams when missing, corrupt, stale (a file shrank / was rewritten at its tail / vanished) or out of `seq` order; an
in-place edit of an old region is caught only by `audit` (re-folds against the index; `audit --rebuild-index` rebuilds). The
row index answers only when it can prove the scan would agree, else the reader scans (typed errors, full messages), and it
is built only for the archive's own ledger or a root that already has `.ledger_index/` — a measurement directory is never
littered. **Never trust a speed claim here without the counts:** `store.IO` counts every line / byte parsed and
`index_test.py` asserts the SHAPE (a cycle at 2 and 12 earlier cycles parses the same lines; the teeth case shows the
pre-index fold growing). `GEN3AI_LEDGER_INDEX=0` runs the old full fold / scan (a debugging hatch, the tests' oracle).
🚨 **THE IN-LOOP EVAL WRITES THE LEDGER TOO — a DUAL WRITE (`cycle_ledger.py`, eval U2).** The trainer builds ONE
`CycleLedger` at startup (`main/train/callbacks.py`) and the eval cycle (`eval_launch.launch_rust_eval_cycle`) and the
SPRT promotion append rows (protocol `gen3_eval_protocol_v1_inloop`; purposes `cycle` / `promotion`; one request per
(cycle × regime), one per SPRT candidate + a decision row) to the archive's `_ledger/rows/inloop/` — BESIDE
`eval_results.jsonl` / `sprt_promotion.jsonl`, which every existing reader still reads until eval U3c. Storage only:
the games are unchanged (digest-proved). A row that disagrees with the published shard results RAISES.
🚨 **`python -m main.h2h` is the OFFLINE checkpoint-vs-checkpoint mirrored head-to-head** (X5 §7 / P0; writes the ledger,
protocol `gen3_eval_protocol_v1_h2h`, by default to the archive's `_ledger/`). It plays two checkpoints on the Rust
eval core through the same `run_cycle`, GREEDY both sides, MIRRORED, teams from the player's eval builder; one valid row per
batch under a claim, resumable per request, any other root under `models/` refused. 🚨 **The mirror hands the TEAMS over and keeps the player in seat p1** — a seat effect
`u` rides every edge (a checkpoint against itself reads 0.5 + u) and a self-play pair is NOT exactly 0.5 (the speed-tie RNG
order is seat-dependent); read a pair-clustered interval, never a per-game one. `python -m main.h2h play-many` plays MANY
cells on ONE engine (eval U6 / X5 U0): the same games and rows as single-cell `play`, one engine start instead of one per
cell; 🚨 **up to TWO architectures per engine** (`main/h2h/arch.py`, F-U6-1 closed): one T2 slot group per architecture
with only the slots its cells need and one eval core per (player group, opponent group), so a two-architecture cross (the
X5 A/B's was `fixed_mass` × `blob`, played at its own pin) plays on one engine — a THIRD architecture is refused before
anything plays; an extractor kwarg RECORDED AT ITS DEFAULT (the extractor constructor's signature default — what an
absent key builds) does not split an architecture, any non-default value does (`slots.canonical_extractor_kwargs`, X5
look 3 FINDING 1). Run it from the repo root (the team
pool is read cwd-relative; any other cwd is refused).
🚨 **`python -m main.plateau` is the PLATEAU meter's TIER 1** (eval U9a; the rule is `plateau_t1.py`): `tick <run>` plays each
due check (the newest 10M-grid node vs the node 50M back) as the registered GSPRT on `main.h2h`'s engine, one 40-pair batch
at a time, and writes ONE `plateau` decision row per check (GAIN / FLAT / UNDECIDED / INCONCLUSIVE; `verify` re-derives it);
`status <run>` prints the run's Tier-1 status. 🚨 **Tier 1 only**: `TIER1_PLATEAU` is a candidate until Tier 2 (the cycle
monitor) and the panel exist. Offline, never on CPU beside an X5 A/B arm (`designs/training/eval_and_rating.md`).
🚨 **`python -m main.belief_roles` is the X5 PURPOSE-METRIC reader** (U7; `designs/endstate/design_x5_belief_tokens.md` §4 / §7.4) — it plays
NOTHING: any checkpoint `.zip` at HEAD's architecture (X5) on the Lane S bank, CPU forwards only, output refused under
`models/`; a blob or other pre-break checkpoint is REFUSED with the pinned commit (`[belief_roles] REFUSED`; the A/B's blob
reads are banked JSON + `.erow.npz`, which `infer` and `--reference` still read). Its
`per_run` block (on-pool primary) is the ONE value per run §7.4's adoption gate infers on, ACROSS SEEDS
(`python -m main.belief_roles infer --treat … --control … --boundary <the stopping look's t>`); its battle-clustered intervals
are descriptive only. 🚨 **Purpose metric (1)'s ADOPTION-GATE form is `intent_logloss_conditional`** (Amendment 3(b), §7.7(b)):
the log loss renormalised over E_row = BLOB's named set, on the rows whose event is in it, for both arms — a `fixed_mass`
read is scored on EVERY blob run of the look and its value is the MEAN (`read --reference <fm label>=<blob>.erow.npz[,…]`,
banked `.erow.npz` files only since the version break; without any its value is None, and `infer` refuses a fixed_mass
read whose references are not EXACTLY the control group's blob runs).
The as-built `intent_logloss` (each arm over the rows its OWN candidates cover, the MISS rate its own column) favours blob
twice and is DESCRIPTIVE only; the coverage (`fixed_mass`'s mass outside E_row vs the realised outside frequency) is reported. The role set is derived from Smogon data at read time and stamped (`role_set_sha256`); `infer` refuses mixed sets.
**Full detail — in [`designs/training/eval_and_rating.md`](eval_and_rating.md)** (its
"The eval COUNT ledger" and "The checkpoint-vs-checkpoint head-to-head" sections).

## ELO / skill rating (`elo.py`, `main.elo`; the bot anchor `data/gen3_bot_elo_anchors.json`)

Once training is mostly self-play pool play, win rate stops being legible — `win_rate_vs_pool` is a
treadmill pinned near 0.50 **by construction** and `win_rate_vs_bots` saturates. The ELO subsystem
gives a single **absolute** number, anchored to the fixed bots.

🚨 **THE EVAL OPPONENT REGIME IS A RECORDED, INHERITED PROPERTY OF A RUN** (2026-09-07,
`gen3_eval_sentinel_greedy_default_v1`). Pool sentinels are **GREEDY by default** and draw the
**trainee's own teams**; `--no-eval-sentinel-greedy` restores the old greedy-trainee-vs-stochastic-
sentinel regime, whose asymmetry read **+8.9 pp [+7.0, +10.7]** in the trainee's favour on the same
frozen pair the dense ladder plays symmetrically. `--promote-threshold` follows the regime (0.55
greedy / 0.65 stochastic) and an explicit value still wins. Both are `ModelVersion` fields (config
**v112**) with argparse default `None`, so **a flagless resume or launcher restart INHERITS the
checkpoint's regime** rather than silently crossing an opponent-regime boundary (rule of evidence
15); every launch prints `⚖️  [EVAL REGIME] …` naming both resolved values and their source.

🚨 **THE GAMES THAT SELECTED A SNAPSHOT NEVER RATE IT** (ladder recipe **v3**, owner decision
2026-09-27). v2 reused the promoting eval cycle's sentinel games as ladder edges (`source:
"eval_cycle"`) — a winner's curse of ~+15..+40 Elo at n = 100. Now each promotion plays **200 FRESH
games vs each sentinel the eval used** (`source: "promotion_baseline"`) plus the usual 100 vs every
other frozen node, logs `FRESH GAMES THIS PROMOTION: N`, and `load_games` ignores any `eval_cycle`
row; `snapshot_ladder --backfill-fresh` replaces a v2 run's. `ladder.json` also carries
**`ratings_relative`** — Elo above a pinned frozen reference node (default the
`untaught_meter_opponent_v14` baseline when the ladder holds it, else the first snapshot), frozen
edges only, a second column beside the bot-anchored headline. Detail:
`designs/training/eval_and_rating.md` (recipe v3, the relative column).

🚨 **EVERY `snapshot_ladder/ladder.json` CARRIES A `recipe` STAMP, AND A CROSS-RUN READER REFUSES
OR REFITS WITHOUT IT** (`snapshot_ladder.recipe_status` → `current`/`absent`/`differs`;
`check_recipe` raises `LadderRecipeError`). A rating is only comparable to one fitted the same
way: a file fitted before `3e6875a5` folded the eval-cycle sentinel edges in and read **+73.1
Elo** above the current fit of the same 20 nodes, flipping the sign of a cross-run delta
(2026-09-14). `main.critic_gate` refuses a stale committed file on its FALLBACK path (it refits
otherwise); `latest_promoted_elo`
deliberately does NOT check it (a within-run trend scalar). **Bump `LADDER_FITTER_VERSION`
whenever the fit changes what a rating MEANS.**

🚨 **THE LADDER PLAYS ON THE RUST EVAL ENGINE, AND THE TRANSPORT IS A REGIME BOUNDARY** (poke-env retirement P2,
2026-10-06; `snapshot_ladder_play.py`, `snapshot_ladder_transport.py`). Every new edge is seat-balanced mirrored
pairs on `main.h2h`'s engine (Rust rows, CPU), draws recorded and excluded from the edge; every row before it was
the poke-env `RLPlayer` + Python-encoder ladder (`python_bridge`, `a` always on p1). Each row carries `transport`
(absent = `python_bridge`), `ladder.json`'s recipe block names it, `fit_ladder` REFUSES a fit over both
(`--transport` / `--allow-mixed-transport`), and `main.critic_gate` refuses two ladders of different transports
(`--allow-transport-mix`). The transport is NOT a recipe change (no fitter bump). Detail and the measured shift:
`designs/training/eval_and_rating.md` "The TRANSPORT boundary".

**Converting a stale file: `python -m main.elo refit [--apply] <run>`** — refits from the
append-only `games.jsonl` (plays nothing) over the COMMITTED file's node set, and with `--apply`
writes the stamped fit while keeping the old one as `snapshot_ladder/ladder.pre_recipe.json`
(`ladder.pre_recipe_vN.json` for a file stamped vN).
Every file on disk before `0f230405` is stale; **68 of 93 move** (median max |Δ| 53.4 Elo, 65 of
68 newest nodes DOWN) and the v9 generation ladder reverses 21 of 153 orderings — the whole audit,
one JSON per run, is `designs/research_state/measurements/ladder_refit_audit_2026-09-22/`.

**Full detail — in [`designs/training/eval_and_rating.md`](eval_and_rating.md)
and [`designs/training/self_play_and_pool.md`](self_play_and_pool.md).**
