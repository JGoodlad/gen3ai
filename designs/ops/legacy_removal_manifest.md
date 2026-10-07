# The LEGACY-SUPPORT REMOVAL — manifest (Rustboro era boundary)

**Status: PROPOSED 2026-10-03, pending the owner's decisions in §0. A document only — nothing is deleted by it.**
Scoped at `82b583b1`. Line counts are `wc -l` at that commit; `≈` is an estimate from a line range, and each unit
re-measures at its own commit. Format follows `deletion_pass_manifest.md` (rows, risks, units, exit criterion).

**The owner's direction (2026-10-03).** *"I would like us to start a new era … I more or less am ready to leave almost
all of the legacy stuff behind, and if it can't be eval'd on the GPU, if it can't be run efficiently, if it isn't part
of a modern lineage … I don't think we need it."* The new era is **Rustboro** (`src/utils/era.py`). It starts at the
X5 A/B and X26. A pre-era run is HISTORY: quotable as history, never a comparator. `models/` FILES stay on disk
(retention is the owner's call); CODE stops supporting them.

**Not legacy, and staying:** the external anchors (Metamon SmallRL / Kakuna, Foul Play) and `main.anchors`; the team
pool and the fixed bots; the Rustboro bot anchors (`data/gen3_bot_elo_anchors.json`, `30ff42f3`); the ledger and its
lessons; everything the X5 build, the X5 A/B, X26 and the eval system need.

**The clean break.** On X5 adoption the losing belief path is deleted (`design_x5_belief_tokens.md` §3.8) and the
version floor rises past every pre-era checkpoint. After that, a pre-era checkpoint does not load at HEAD; it is read
from its own commit (the `era_checkout_only` rule `designs/baselines.json` already applies to seven entries). **§0
D-L1 is the decision that has to be right first:** the floor must rise past pre-era runs WITHOUT also refusing the
Rustboro runs that the A/B and X26 produce (FINDING F-LR-1).

---

## 0. The owner's decisions

| # | decision | recommendation |
|---|---|---|
| D-L1 | **What the "bump" refuses.** Today `MIGRATION_FLOOR == SIGNATURE_FIRST_VERSION[ARCH_SIGNATURE]` (`migration_floor_test.py`), and §3.8 bumps the signature when the loser is DELETED, after the A/B. Under that rule, the floor lands on the deletion commit's version, which also refuses the A/B runs (built one or more versions earlier, at the X5 flag's version) and the X26 baseline, which CONTINUES the winning seed-1001 A/B run (F-LR-1) | **Decouple the floor from the signature.** Set `MIGRATION_FLOOR` to the **first Rustboro config version**: the version the X5 build stamps (`--belief-tokens`, §3.8). Declare it once as `RUSTBORO_FLOOR`, and assert it in `migration_floor_test`. Bump `ARCH_SIGNATURE` only if the winner's weight shapes differ from what an A/B run recorded. For the blob path's removal itself, one `_DEAD_FEK_JUDGED` row is enough: `("belief_tokens", <winner>)`. Every pre-era checkpoint is then refused, and every Rustboro checkpoint loads. **SUPERSEDED 2026-10-07 by the X5 VERSION BREAK, part 1** (config v144, `gen3_x5_version_break_v1`; orchestrator decision): the floor RISES WITH the signature to the break's own version (`MIGRATION_FLOOR` = 144 = `SIGNATURE_FIRST_VERSION[ARCH_SIGNATURE]`, `migration_floor_test`'s rule unchanged), because the break's later parts reshape weights (F1, F16b, the op's `out_gain` slot-tie) and change behaviour (the intent_conditional pre-gain) — no pre-break checkpoint, A/B runs included, is reproducible at HEAD, so every one runs PINNED (`model_version/version_break.py`, `LAST_BLOB_COMMIT`). The `("belief_tokens", "fixed_mass")` `_DEAD_FEK_JUDGED` row landed as specified |
| D-L2 | **A′ (`ai_v14_09_r0_offense_a`, paused at 92.1M on torch 2.5.1)** is the last reason HEAD keeps the 2.5.1 env selection (deletion pass R7's KEEP) | **Abandon A′ as a run; keep its files.** It is pre-era: it trained under the CUDA compile miscompile and the old bots. Then L5 deletes the legacy-torch selection EARLY (no bump needed) |
| D-L3 | **The Rustboro untaught-meter opponent and ladder reference node** (§2.1) | **The winning arm's seed-1001 A/B FINAL snapshot.** That is the exact file the X26 baseline continues from, so it exists the day of adoption, and it is a node in X26's own ladder. Register it by sha256 at adoption. The alternative is the X26 baseline's 24M snapshot, which repeats N0's convention but cannot be chosen until X26 reaches 24M |
| D-L4 | **The pre-era eval BACKFILL** (`design_evaluation.md` §0b.6, U3) | **Drop the general backfill** (`eval_results.jsonl` × 265 runs, `snapshot_ladder/games.jsonl` × 104 files). **Keep** the verbatim copies (P0's 120 rows, the 1,296-row bot round robin) and the committed `measurements/anchors_*/` rows. N0's ladder is backfilled ONLY if the plateau-test unit asks for its back-test (§3) |
| D-L5 | **The offline cf stack** (deletion pass D6, left to the flag census, which ruled on flags, not these modules) | **DELETE** `cf_producer*`, `cf_audit*`, `cf_mc_return`, `cf_q_labels`, `main/ops/critic_read`. Each reads a `cf_records` ring, and only the deleted Python core wrote one, so no Rustboro run ever will. `main.harvest` + `winprob_finetune`: verify their input at the unit, and delete them unless they run on Rust-core traces (L10b) **ANSWERED (owner 2026-10-05):** "I like the counterfactual stack, I think it is where a lot of our future work would go. Happy to drop if we want to just rewrite it when we need." The orchestrator's reading: the stack reads a `cf_records` ring that only the deleted Python core wrote, so it cannot run on any Rustboro run as it stands, and a future version would branch from Rust core clones (`design_q_head.md`). So: DELETE L10a in B5, but FIRST write `designs/endstate/design_counterfactual_labels.md`, a short preservation note (what each piece computed: the per-action labels, MC returns, Q labels, the audit and twin methods, the lock and sampler, what worked and what each was used to learn, with the `git log` pointer), as the seed for the rewrite on the Rust core. L10b is unchanged: check it at the unit. |
| D-L6 | **T5** (`TASK_BACKLOG.md`: the ladder recipe v3 backfill for N0, `--backfill-fresh`) | **DROP T5.** It cleans a pre-era ladder. Rustboro runs write recipe v3 natively. Whether `--backfill-fresh` itself goes is decided at the unit (it lives in `snapshot_ladder.py` + `main/elo.py`; delete it only if nothing in the eval system's U7 reuses it) |

---

## 1. INVENTORY — one row per legacy-support item

Paths are under `src/`. "Live dependent" was grepped at `82b583b1`. Verdicts:
- **DELETE**: no live dependent and no bump needed, so it is a Monday-burn candidate.
- **DELETE-AT-BUMP**: it must wait for D-L1's floor raise.
- **KEEP**: there is a live reason.
- **NEEDS-OWNER**: §0 decides.

| # | item (files ≈ lines, module / tests) | the pre-era thing it supports | live dependent in the Rustboro plan | verdict | risk |
|---|---|---|---|---|---|
| L1 | **The v121–v135 migration branches and their history.** `agents/model/model_version/migrations.py` 656 (the post-floor `if version < N` branches + the version-independent POP sanitizers ≈ 300; the "v97–v120 MIGRATION HISTORY" comment ≈ 230); `constants.py` 466, mostly the version-history comment block above `MODEL_CONFIG_VERSION` (≈ 380; the CHANGELOG already holds each entry). **≈ 0.9k → ≈ 0.1k** | configs recorded at v121–v135: N0 `ai_v14_01_base` and its forks (A′, C_fix, K2, K3) — every `gen3_event_record_v2` run before the X5 build | none once the floor is the first Rustboro version. The X5 build's own `_migrate_config` default (`blob`) is the one branch that survives | DELETE-AT-BUMP | `migration_floor_test`'s rule changes under D-L1. Without that change, a mechanical raise refuses X26 (F-LR-1) |
| L2 | **The retired-lever and shaped-reward refusal tables.** `model_version/retired_levers.py` 395, `model_version/shaped_reward.py` 165 + their callers (`check_no_retired_levers` in `main/train/config.py` and `main/checkargs.py:471`; `saved_config_path`). Tests: `retired_levers_test` 604, `shaped_reward_test` 200. **≈ 0.6k / ≈ 0.8k** | a v121–v135 config that recorded a deleted lever ON. The module calls itself belt-and-braces: "every v121+ run on record recorded every lever below OFF" | none: a Rustboro config never carried those fields | DELETE-AT-BUMP | `recorded_critic` (L4) borrows `shaped_reward.saved_config_path`, so they go together |
| L3 | **The dead-kwarg SANITIZERS.** `agents/model/snapshot.py` L1020–1270: `_DEAD_FEK_INERT`, `_DEAD_FEK_JUDGED`, `_DEAD_POLICY_KWARGS_JUDGED`, `sanitize_dead_extractor_kwargs`, `_patch_historical_floor`, `historical_load_kwargs` (5 call sites: `main/play.py`, `main/h2h/play.py`, `main/eval_worker.py`, `agents/training/rust_eval/parity.py`, the prober). Plus `belief_tables.sanitize_historical_move_floor` (a pre-v65 floor patch, re-exported by `damage_tables`, called by `delivery_graph.py:257`). Test: `dead_kwargs_sanitize_test` 162. **≈ 0.3k / ≈ 0.2k** | zips whose pickled `policy_kwargs` carry kwargs deleted since (PopArt, the value-dist head, zarch, pubval…). The move-floor patch is for v64-and-older zips | `delivery_graph` sanitizes the production MIRROR, which is still built from a v97 config (L12). It goes when the mirror is re-synced | DELETE-AT-BUMP. **KEEP the RULE and `ctor_kwarg_snapshot_test`**: the blob deletion adds the first row of the new lists (D-L1) | the call sites need `historical_load_kwargs` gone or reduced to `{}` in one commit. The h2h and eval paths load Rustboro snapshots daily |
| L4 | **The critic-record fallbacks for python-era checkpoints.** `agents/model/critic_mode.CRITIC_UNRECORDED` (an absent `critic` reads as shaped) and its reader in `value_sidecar.py:220`. In `main/train/rust_env_setup.py` ≈ L40–150: `recorded_critic`, `PythonEraShapedCheckpoint`, `python_era_refusal`, `refuse_python_era_checkpoint`, `D4_CORE_SWITCH`, `env_core_switch_line`, and the recorded-core reader. Call sites: `main/checkargs.py:371`, `main/train/config.py:400`. **≈ 0.15k / ≈ 0.2k** | deletion pass D4: a python-core checkpoint. If it trained shaped it is REFUSED; if winprob it moves onto Rust with an announcement | until the bump, an N0-lineage FORK still needs D4's announcement. After it, nothing pre-era reaches these paths | DELETE-AT-BUMP | none beyond L1 |
| L5 | **The legacy torch selection.** `main/launcher/torch_runtime.py` 153: `LEGACY_TORCH`, `KNOWN_ENVS["2.5.1"]`, "an unrecorded run is 2.5.1" (≈ 40). Also the `gen3ai_stable` env, `environment.yml` 132 (FROZEN by rule), `scripts/bootstrap.sh`'s stable-env handling (3 sites), and their test rows in `torch_runtime_test`, `interpreter_test`, `bootstrap_env_stamp_test`. **≈ 0.2k / ≈ 0.15k** | resuming a run that recorded torch 2.5.1 (or none) PINNED, on its own torch. In practice that means A′ | none. **KEEP** `utils/torch_floor.py` (HEAD on < 2.8 is refused, a general guard), the `torch_version` record, the mismatch refusal and `--allow-torch-switch`. The next torch upgrade reuses all four | NEEDS-OWNER (D-L2) → DELETE early | a pre-era run without a torch record would then be refused rather than routed. That is the intended outcome, and the message must say "pre-era: run it from its own commit + env by hand" |
| L6 | **The prober's lenient load.** `main/prober/model.py` `sanitized_load_custom_objects`, `_dropped_extractor_kwargs`, `_accepted_extractor_kwargs` (≈ 80). It drops unknown kwargs so that an archived run still loads read-only | forensic model-loading views on v121–v135 runs whose zips carry since-deleted kwargs | none after the bump: a pre-era load fails the strict path anyway. **KEEP `ArchDriftError`.** The diagnosis that names the commit to check out is how history stays readable | DELETE-AT-BUMP | a forensic read of an N0-lineage run then needs that run's commit. The prober already prints the hash |
| L7 | **The prober's legacy checkpoint LAYOUT.** `main/prober/discovery.py` `list_checkpoints` searches `<run>/*.zip` (the pre-`checkpoints/` layout) plus the `legacy_latest` / `legacy_best_model` rungs (≈ 30) | runs from before checkpoints moved under `checkpoints/`; every one of them is long pre-era | none (no Rustboro run writes the old layout) | DELETE (early) | the model-free commands do not need checkpoints, so nothing they read is lost |
| L8 | **The prober's "absent on older traces" tolerance.** About 15 `None when absent` branches across `engine/`, `session/trace_io.py`, `awareness.py`, `timeline.py` | traces that predate a field | **the same branches serve Rust-core traces and websocket traces**, which also lack those fields (`rust_eval/traces.py:38` stores obs / logits / values only) | KEEP | — |
| L9 | **The SCAFFOLDING GAUGE.** `main/scaffolding_gauge.py` 882 / `scaffolding_gauge_test` 609. Its subject is the gap between a shaped critic's PopArt-normalised V and the win-prob head (the RANK + AFFINE gauges) | shaped-critic runs only. Since P11d no run has two value readouts to compare | `main/critic_gate.py` (580, 610–611, 1083) and `main/ops/critic_readouts.py:473` IMPORT its shared helpers: `collect_slices`, `opponent_class`, `build_reliability`, `true_win_rates`, `SelectionWeightError`. **KEEP** `agents/training/scaffolding.py`'s `reliability_table` / `cluster_bootstrap_ci` / `spearman_rho` (read live by `instrumented_ppo/calibration.py` and `critic_gate`) | DELETE (early) after moving the five helpers to a `critic_reliability` module | `critic_gate_test`, `core_trace_readers_test` and `trace_summary_reader_gate_test` reference the module. They are re-pointed in the same commit |
| L10a | **The offline cf stack, the `cf_records` half.** `agents/training/cf_producer.py` 1,535 + `_labels` 145 + `_lock` 76 + `_sampler` 108 + `_snapshot` 257; `cf_audit.py` 959 + `_render` 161 + `_twin` 251; `cf_mc_return.py` 213; `cf_q_labels.py` 231; `main/ops/critic_read.py` 1,372. **≈ 5.3k / ≈ 4.4k** | reading a run's `cf_records` reconstruction ring. Only the deleted Python core wrote one (deletion pass R2), so every source is pre-era | `main/search_dividend/playoff.py:150` imports `cf_producer.rollout_outcome_score` (move it). `ops/critic_readouts.py` and `stats.py` import `cf_audit` pieces (move or drop). `design_q_head.md` cites the `cf_audit` command as precedent only; a Q head would branch from Rust core clones | NEEDS-OWNER (D-L5) → DELETE (early) | `harvest_schema.py` imports `cf_audit` (L10b) |
| L10b | **Harvest + the win-prob fine-tune.** `main/harvest.py` 1,327, `main/harvest_meter.py` 530, `agents/training/harvest_schema.py` 254, `agents/training/winprob_finetune.py` 958. Test: `harvest_test` (15 pre-era run references). **≈ 3.1k** | the 2026-09 "is the information limit in V?" probe, run on pre-era runs | **UNVERIFIED** whether `main.harvest` can mine a Rust-core trace. Its rollouts and schema predate the core traces | NEEDS-OWNER (D-L5): verify at the unit, then DELETE unless it runs on a Rustboro run | deleting a working probe. The check costs one run of `main.harvest` on a Rust-core trace tree |
| L11 | **The pre-era BASELINES.** `designs/baselines.json`: seven of nine entries are `era_checkout_only` pre-v121 (`production` = gen-17, `v9_long_baseline`, `v9_fold_parent`, `v8_line`, `v8_parent`, `famine_comparator`, `untaught_meter_opponent` + `untaught_meter_config`). `untaught_meter_opponent_v14` (v121) becomes one AT the bump. The `tb_curated` list names `v8_line` + `v9_long_baseline` | comparators for the gen / v8 / v9 eras | `production` is LIVE (L12, F-LR-3). `untaught_meter_opponent_v14` is LIVE (L13, L14) | DELETE-AT-BUMP as RE-POINT: the replacements in §2 go in, and the old entries move to a `history` block. **KEEP** the `era_checkout_only` machinery (`agents/training/baselines.py`, `main/baselines.py`), which is how history stays quotable | `main.baselines check` must stay green with history entries, which needs their warn level (today: `pre_generation` + `era_checkout_only` ⇒ warn) |
| L12 | **The production MIRROR's pre-era construction.** `designs/production_config.json` was built from gen-17's v97 config, migrated to v109, with a critic override block. Its `config_overrides` `arch_signature` / `total_dim` / `active_context_dim` are the "SIGNATURE-BUMP WINDOW" (`production_config.README.md`), there "until the first new-lineage run exists". `delivery_graph.py:248–257` patches its pre-v65 move floor | `--arch production`, `checkargs`' ARCH / RECIPE SURFACE, `ARCHITECTURE.md` generation, the compile gate, the arch viewer — the PRODUCTION SURFACE | **LIVE: every Rustboro launch reads it** | KEEP, and REPLACE the construction at the bump (§2.3) | 🚨 a GIGO class: on 2026-09-06 a mirror built by omission dropped 60 of 120 graph nodes. Re-sync by the existing procedure, and diff the ARCH SURFACE before and after (it must be empty except for X5's own flag) |
| L13 | **The UNTAUGHT METER and its pinned opponent.** `agents/training/untaught_meter.py` 949, `main/untaught_meter.py` 495. `DEFAULT_OPPONENT_BASELINE` was `untaught_meter_opponent` (rev-1's 24M, pre-v121, unloadable at HEAD: F-LR-2) and is `untaught_meter_opponent_v14` since B3 (2026-10-04), with `DEFAULT_CONFIG = "auto"`. The taught / untaught team manifests come from the 2×2 era | the fold-era measurement series | **LIVE: X5's reported secondary** (`design_x5_belief_tokens.md` §7.2, §7.4). It is also named by `design_evaluation.md` (U3b), `design_ladder_campaign.md` and `design_learner_recipe.md` | KEEP the meter; RE-POINT the default (§2.1) | 🚨 the A/B's untaught reads use the v14 opponent, which the floor refuses, so **every X5 untaught read must finish before the bump commit** |
| L14 | **The ladder's REFERENCE NODE.** `agents/training/snapshot_ladder.py:467` `DEFAULT_REFERENCE_BASELINE = "untaught_meter_opponent_v14"` | `ratings_relative` for N0-lineage ladders | a fresh Rustboro run's ladder never holds that node, so it falls back to its FIRST snapshot (no break). Cross-run `ratings_relative` then has no common zero | KEEP the mechanism; RE-POINT (§2.1) | none (the fallback is already a declared rule) |
| L15 | **Lane S, `policy_spectrum`.** `designs/research_state/measurements/m5_laneS/bank_v1` (2.7 MB, about 20.7k re-encodable turns recorded from N0-lineage eval traces), `baseline_2026-09-29` (13 pre-era readings), and the reproduction "teeth" test (`main/policy_spectrum/policy_spectrum_integration_test.py`, reads `N0@74M`) | bank_v1's RECORDING logits are N0's. The turns themselves are input logs, so any checkpoint at HEAD can read them | **LIVE: X5's ADOPTION GATE** (purpose metrics 1–4, `main.belief_roles`) is read on bank_v1. Both arms are read on the SAME turns at one commit, so the turns must not change during the A/B | KEEP bank_v1; REPLACE the teeth (§2.2) | the teeth have NOT asserted reproduction since `f0310ee7`. They skip on the op-semantics mismatch while determinism still passes (F-LR-4). After the bump `N0@74M` does not load at all |
| L16 | **`main.elo refit` and `ladder.pre_recipe.json`** | ladders fitted under an older recipe | generic: the fitter will change again within Rustboro | KEEP | — |
| L17 | **The pre-boundary INFERENCE branches in eval readers.** Three of them: the asymmetric-sentinel regime inferred for rows before 2026-09-07; the ladder recipe inferred from an absent `recipe_version`; and F-LH-13's multi-team protocol on 20 runs' `ext_` rows | reading and backfilling pre-era eval history | only §0b.6's backfill needs them in the eval design | NEEDS-OWNER (D-L4) → DELETE with the backfill if it is dropped. Sizes measured at the unit | a reader that meets an old file without its branch must REFUSE it, not misread it. Keep the refusal |
| L18 | **`main.lineage --backfill`** (`main/lineage.py` `backfill()` L265+, ≈ 0.1k) and the `"derived": true` reading | writing a derived lineage block into a legacy run's `metadata.json` | none: every run since the lineage record writes it at creation | DELETE (early). KEEP the reader of an existing `derived` block (history) | none |
| L19 | **Tests that name pre-era runs.** About 30 `*_test.py` files (e.g. `main/best_response_gap_integration_test.py` 10, `main/harvest_test.py` 15, `main/baselines_test.py` 8, `main/train/arch_surface_test.py` 6) | fixtures and archive loads | most are STRINGS (stay); the loads skip without the archive today and refuse after the bump | DELETE-AT-BUMP (rewire per file) | 🚨 a load that turns from PASS into SKIP at the bump hides lost coverage. The bump unit lists each one that changes outcome |
| L20 | **The `production` load in the parity fixture recorder.** `agents/battle/rust_core_parity.py:602–620` (`production_checkpoint`, `load_production_policy`, used by `record_commit_fixture`) | records 2 production-policy battles with gen-17 | the COMMIT-tier fixture's re-record path | KEEP the path; RE-POINT it through §2.3 | it probably does not run at HEAD today (F-LR-3) |
| — | **Not legacy support (out of scope)** | `main/search_dividend/` (7.7k; it searches around any checkpoint), the prober's counterfactual view (bridge replays, no `cf_records`), the `eval_results.jsonl` readers (new runs still write that file), `ArchDriftError`, `main.anchors`' floor refusal, launcher pinning, `--sync-to-main` / `--no-pin`, `torch_floor` | | | — | |

**Counts by verdict** (code / tests, ±25%):

| verdict | rows | ≈ lines code / tests |
|---|---|---|
| DELETE (early) | L7, L9, L18 | ≈ 1.0k / ≈ 0.6k (helpers move, not counted) |
| DELETE-AT-BUMP | L1, L2, L3, L4, L6, L19 (+ L11 re-point) | ≈ 2.0k / ≈ 1.4k + the L19 rewires |
| NEEDS-OWNER → DELETE | L5 (D-L2), L10a + L10b (D-L5), L17 (D-L4), T5 (D-L6) | ≈ 8.6k / ≈ 5.0k (L10b alone ≈ 3.1k / ≈ 0.6k) |
| KEEP (with re-point or replacement) | L8, L12, L13, L14, L15, L16, L20 | — |

If every recommendation is taken: **≈ 11.6k non-test lines and ≈ 7k test lines** (about 5% of non-test `src/` Python).
The 8.6k from the cf stack and harvest (L10a + L10b) is most of it.

---

## 2. Replacements needed BEFORE deletion

### 2.1 The Rustboro untaught-meter opponent and ladder reference node (D-L3)
- **Which file.** The winning arm's seed-1001 A/B final snapshot, the file the X26 baseline CONTINUES from. It exists
  on adoption day. It is Rustboro-era. Because X26 continues that run, the node is in X26's own ladder (the
  `_baseline_reference_step` rule: same run, or same sha256).
- **When.** Register it in the adoption commit, before the floor raise: `python -m main.baselines set
  untaught_meter_opponent_rb <run>/snapshots/<final>.zip --reason "<ledger title>"`. One entry carries both roles, as
  `untaught_meter_opponent_v14` does today.
- **Config.** `untaught_meter_config_rb` is that snapshot's `model_config.json`.
- **Re-point** `untaught_meter.DEFAULT_OPPONENT_BASELINE` (the config default is already `auto`, B3) and
  `snapshot_ladder.DEFAULT_REFERENCE_BASELINE` to the new names in the same commit.
- **A new opponent is a RE-MEASUREMENT.** Untaught levels from the v14 opponent and from the Rustboro one are never on
  one scale (F-X5-20's lesson). The ledger entry says so.
- **Risk: saturation.** A 15M opponent may be too weak for X26 at 75M+, and the meter compresses at the top. The
  first X26 untaught read at about 50M checks it. If the level is above about 0.8, the owner may add a second,
  later opponent (a new series, never a replacement).

### 2.2 Lane S
- **The X5 purpose metrics need NO rebuild.** They read both arms on bank_v1's fixed turns at one commit, and changing
  the bank mid-A/B would break the pre-registered comparison.
- **What needs replacing is the teeth.** Build `bank_v2` from the X26 baseline's own eval traces (Rust-core traces at
  current op semantics) after X26 has a few eval cycles, using the existing builder. Re-point the reproduction test to
  a Rustboro snapshot, so that "a read reproduces its recording" is ASSERTED again rather than skipped.
- bank_v1 stays committed as the X5 substrate. Its `baseline_2026-09-29` readings become history.

### 2.3 The baselines registry and the production mirror
- `production`: re-point to the X26 baseline at a declared step. Recommendation: the same snapshot as §2.1, because
  that is the first Rustboro checkpoint with the adopted architecture.
- Re-sync `designs/production_config.json` from that run's `model_config.json` by the README's procedure. Remove the
  signature-window `config_overrides` and `delivery_graph`'s historical floor patch.
- Gates: `checkargs`' ARCH SURFACE diff before vs after must be empty apart from X5's flag. `mode_flag_doc_gate`,
  `recipe_doc_gate` and the delivery graph must stay green. This also repairs L20 (F-LR-3).
- `tb_curated`: replace `v8_line` / `v9_long_baseline` with the X26 baseline.
- The other pre-era entries move to a `history` block (still quotable, never a comparator).

---

## 3. The eval backfill under the owner's direction (D-L4)

`design_evaluation.md` §0b.6 plans a backfill:
- 265 runs' `eval_results.jsonl` (2,071 rows);
- 104 `snapshot_ladder/games.jsonl` (5,766 rows);
- the anchor measurements;
- P0's rows and the bot round robin.

Every one of the first two sources is pre-era, and its rows carry several defects at once:
- the old bots (pre-`f885ad8f`, pre-Curse-as-setup);
- an asymmetric sentinel regime on part of them;
- no pentanomial pairs;
- draws folded on most of them;
- teams and seeds unrecorded.

The design's own rules already forbid pooling them with new rows (bot eras never pool; sentinel regimes never pool).
Under the owner's direction they would be HISTORY that no Rustboro decision reads. A decision rule reads only rows of
its own request (§0c rule 1), and pre-era rows belong to no Rustboro request.

**Recommendation.**
- **Drop the general backfill.** It saves most of U3 and lets L17's inference branches go. The source files stay on
  disk, quotable as history in their own terms.
- **Keep the cheap verbatim copies:** P0's 120 v1 rows and the 1,296-row bot round robin (the Rustboro bot anchors'
  own evidence), both "copied verbatim, upgraded on read".
- **Keep the committed `measurements/anchors_*/` rows.** The external anchors are standing numbers; they carry W/L/D
  and the regime, so they need almost no flags.
- **N0's ladder is the one exception.** Backfill it only if the plateau-test unit decides it needs §8's back-test.
  It reads COUNTS and loads no model, so it survives the bump. §8 already notes that a real back-test needs about 70k
  more games on a checkpoint that goes unloadable at the bump. If the back-test is wanted, run those games before the
  bump.

---

## 4. The order

| unit | content | size (agent-days) | tier | when |
|---|---|---|---|---|
| **B1** | L9: move the five helpers to `critic_reliability`, delete the scaffolding gauge | 0.5 | opus-medium | **Monday-night burn** |
| **B2** | L7 + L18: the prober's legacy layout, `main.lineage --backfill` | 0.25 | opus-medium | **Monday-night burn** |
| **B3** | **DONE 2026-10-04.** F-LR-2 fix: `untaught_meter.DEFAULT_OPPONENT_BASELINE` = `untaught_meter_opponent_v14`, the config default = `auto` (the v101 shared config cannot load at HEAD), the series boundary recorded in `_meta.series` and enforced by `--from-rows` (a mixed-opponent read is refused). INTERIM until §2.1 / R0 | 0.1 | opus-medium | **Monday-night burn** (a fix, not a deletion) |
| **B4** | L5 after D-L2: the legacy torch selection, `environment.yml`, the bootstrap's stable path | 0.5 | opus-medium | Monday burn **if D-L2 is answered** |
| **B5** | L10a (+ L10b per its check) after D-L5; move `rollout_outcome_score` | 1.0 (+0.5 for the harvest check) | opus-medium | Monday burn **if D-L5 is answered** |
| **R0** | §2.1 + §2.3 registrations at X5 ADOPTION: the Rustboro opponent / reference, the `production` re-point, the mirror re-sync, `tb_curated` | 1.0 | **opus-high** (production surface, GIGO class) | with the adoption commit, BEFORE the floor raise; after every X5 untaught read |
| **R1** | THE BUMP, in the blob-deletion unit: D-L1's floor; L1, L2, L3, L4, L6; `_DEAD_FEK_JUDGED`'s blob row; the pre-era baselines to `history` | 1.5 | **opus-high** (every load path) | right after R0 |
| **R2** | L19 test rewires (the list of outcome changes) + docs: `ARCHITECTURE.md`, CHANGELOG, the leaf `CLAUDE.md`s, `deleted_flags.md`, `designs/ops/TRAINING_RUN_SOP.md` if it names a removed path | 1.0 | opus-medium | with R1 |
| **R3** | §2.2: Lane S `bank_v2` from X26's eval traces, the teeth re-pointed | 1.0 | opus-medium | after X26 has written about 3 eval cycles |
| **(D-L4)** | the eval design's U3 shrunk to verbatim copies (+ the N0 ladder only on request); L17 with it | (saves most of U3) | — | in the eval system's own plan |

**Total ≈ 7.3 agent-days**: early ≈ 2.9 (B1–B5), at the bump ≈ 3.5 (R0–R2), later ≈ 1.0 (R3).

**Monday-night burn candidates (no live dependent, no bump needed):** B1, B2, B3 unconditionally, and B4 and B5 once
D-L2 / D-L5 are answered.

---

## 5. EXIT CRITERION
1. `MIGRATION_FLOOR` equals the first Rustboro config version, and `migration_floor_test` asserts the new rule (D-L1).
   A pre-era checkpoint at HEAD is a typed refusal naming its commit. Every Rustboro checkpoint (the A/B finals, the
   X26 baseline) loads.
2. `grep` finds no `retired_levers`, `shaped_reward`, `CRITIC_UNRECORDED`, `refuse_python_era_checkpoint`,
   `historical_load_kwargs`, `sanitize_historical_move_floor`, `LEGACY_TORCH` or `gen3ai_stable` in `src/`. A
   decision recorded otherwise is the exception.
3. `designs/baselines.json` has no live entry pointing at a pre-era run. `main.baselines check` is green.
   `production_config.json` carries no `config_overrides` signature window.
4. The untaught meter's default opponent and the ladder reference node are Rustboro snapshots, each loadable at HEAD.
5. The Lane S teeth ASSERT reproduction on a Rustboro source (no op-semantics skip).
6. The routine gate is green. L19's list of tests whose outcome changed is in the R2 commit body. One `--debug` smoke
   and one `--arch production` dry-run (`python -m main.launcher --dry-run`) are clean.
7. The ledger records the boundary: what stopped loading at HEAD, and the commit to read it from.

---

## 6. FINDINGS raised while scoping

| id | finding | owner unit |
|---|---|---|
| **F-LR-1** | Under today's rule (`MIGRATION_FLOOR == SIGNATURE_FIRST_VERSION[ARCH_SIGNATURE]`), the bump that §3.8 places at the loser's DELETION sets the floor to the deletion commit's version. That REFUSES the A/B runs and the X26 baseline (a continuation of the winning seed-1001 A/B run, which by then is a v-floor−1 checkpoint), so X26 could only run pinned. Not built yet (`--belief-tokens` does not exist at `82b583b1`), so it costs nothing to fix now. **RESOLVED BY DECISION 2026-10-07** (the X5 version break): the floor rose to 144 deliberately — the break's later parts make every pre-break checkpoint irreproducible at HEAD — so an A/B checkpoint (and anything that continues one) runs pinned at or before `26131c0c`; a post-break baseline starts fresh | D-L1, R1. **The X5 build unit must know this before stamping its version** |
| **F-LR-2** | `untaught_meter.DEFAULT_OPPONENT_BASELINE` is `untaught_meter_opponent`, rev-1's 24M at v101 (`era_checkout_only`). It does not load at HEAD, so a bare `python -m main.untaught_meter` fails, and the live opponent (v14) must be passed by hand. REPRODUCED 2026-10-04 (`ModelVersionError: config_version 101 is a PRE-GENERATION checkpoint`, at the opponent load; passing `--opponent` alone does NOT fix it because the default CONFIG was also v101). **FIXED in B3** | B3 (done) |
| **F-LR-3** | `rust_core_parity.record_commit_fixture` → `load_production_policy` → `baselines.load("production")`, which is gen-17, v97, `era_checkout_only`. `baselines.load` runs `check_era`, so re-recording the COMMIT-tier fixture very likely raises at HEAD. UNVERIFIED by execution (re-recording is rare; the committed fixture still replays) | R0 (the `production` re-point) |
| **F-LR-4** | The Lane S reproduction teeth have NOT asserted reproduction since `f0310ee7`. bank_v1's manifest has no `op_semantics` stamp, HEAD's is `gen3_beatup_exact_v1`, so the test asserts only that the mismatch is visible (`max_abs_dp ≥ DP_BAR`) and then SKIPs. Determinism still passes | R3 |
| **F-LR-5** | `production_config.json` is still constructed from gen-17's v97 config, with the "signature-bump window" overrides that were meant to last only until the first new-lineage run existed. N0 existed from 2026-09-27 and was never used to re-sync it. The X26 baseline is the natural point | R0 |
| **F-LR-6** | `TASK_BACKLOG.md` T5 (the N0 ladder v3 backfill) and `design_evaluation.md` §0b.6 / §8's N0 back-test both spend CPU on a pre-era lineage. Under D-L4 / D-L6 they shrink or drop | owner |
| **F-LR-7** | The A/B's untaught secondary reads the v14 opponent, so every one of those reads must complete before R1's floor raise. Nothing enforces the order except this manifest | R0's checklist |
