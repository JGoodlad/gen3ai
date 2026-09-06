# Flag census — 2026-09-06 (`gen3_dead_flag_purge_v2`)

**Why this exists.** An end-state paydown before the win-prob-only generation and its multi-day runs.
The question asked of every flag was the v88 deletion question: *does this flag's OFF value describe
code that has never run and will never run again?* The census was produced FIRST, and the deletion
set was chosen from it — not the other way round.

**Headline: 1 deleted, and that is the honest result.** Exactly **one** flag met all four deletion
conditions. Every other OFF flag is an armed lever, an umbrella-desugar target, a documented next
step, or a switch the production config actively depends on being where it is. Rounds v75/v78/v88
already took the easy ones; a purge that finds little is what a healthy surface looks like.

## The four conditions

A flag is deleted only if ALL of these hold. Any one failing keeps it, and the reason is recorded.

1. **Never ON in any gen-9+ run's `model_config.json`** (measured over the archive, not assumed).
2. **Not in the production config** (`ARCHITECTURE.md` §6).
3. **No `state_dict` key depends on it being OFF** — deleting the flag deletes only code that never
   ran, and `ARCH_SIGNATURE` therefore does not move.
4. **No live design doc names it as the next lever** (`designs/research_state/`).

## Method and provenance

* **Registry surface** — `agents.model.flag_registry.REGISTRY`, **49 entries**, the single
  declaration behind the five binding sites. Read from the module, never grepped out of
  `extractor_arch.py`, which is GENERATED.
* **Argparse surface** — `main.train_rl_agent.build_parser()`, **256 options**, inspected via
  `parser._actions` rather than scraped `--help` text.
* **Production values** — `designs/ARCHITECTURE.md` §6, including its `INERT` markings.
* **Archive usage** — every `models/*/model_config.json` in the MAIN checkout, read-only.
  **214 runs carry a config; 123 are gen-9+**, config_version spanning 2–107.
* **Ledger verdicts** — `designs/research_state/ledger.md`, grepped per flag in both the
  `--dashed-form` and the underscored config-field form.

The gen-9 line is drawn at `config_version >= 69`, the first version whose `arch_signature` is in the
current lineage. The pre-gen-9 side is entirely below `MIGRATION_FLOOR` 96 and so cannot be loaded
into the current architecture by any route — which is what makes *"never ON since gen-9"* the right
question rather than the unanswerable *"never ON ever"*.

⚠️ **Counts are measurements with a date on them.** The two surface counts above were re-measured
directly from the live modules for this census, and disagreed with a first pass; recount before
quoting them rather than inheriting a number from this file.

## (b)(c) Classification

### DELETED — 1

| flag | evidence |
|---|---|
| `threat_prob_outspeed` | **0 of 123** gen-9+ runs ON. OFF in production. Builds **no parameters** — it only chose the divisor in `DamageOperator._p_outspeed` (believed speed STD vs a fixed scale), so no `state_dict` key moves. **Zero** mentions under `designs/research_state/`. 61 archive runs recorded it ON; the **highest config_version among them is 46**, far below the migration floor of 96. |

The migration REFUSES a recorded `True` rather than popping it, and the reason inverts the usual one.
Every other member of `_DEAD_FEK_JUDGED` is there because its ON value named parameters, so popping
would hand SB3 an unplaceable `state_dict`. This flag is the opposite: a `True` and a `False`
checkpoint are **byte-identical in every key**, so a silent pop would load cleanly, pass every shape
gate, and run a checkpoint under physics it was never trained on — forever, with nothing able to
notice. *Loading cleanly is the reason to refuse it, not a reason to allow it.*

There is **no replacement flag**: the surviving behaviour is the fixed-scale logistic that every
gen-9+ run already used, so a stale command drops the flag and launches unchanged.

### KEPT — the categories, and why

**PRODUCTION-ON (keep, no argument).** Every flag at an ACTIVE value in §6. Not candidates.

**OFF-but-live-lever (keep).** The large majority of the OFF surface. Representative cases:

* **The reward-shaping family** — `drop_redundant_bias`, `drop_switch_bias`, `stall_pbrs`,
  `bias_redesign`. Never typed in a recorded command except `bias_redesign` (7 runs), but these are
  *armed* levers in `reward_manager.py` gating real term families, and they are the documented path
  toward the zero-BIAS destination. `bias_redesign` additionally gates `no_progress_tax` alongside
  `all_shaping_pbrs`, so the `--no-all-shaping-pbrs` fallback depends on it. They are also the wrong
  MECHANISM for this purge: these are `check_reward_config` resume-immutable value hparams, not
  extractor toggles, and deleting one silently narrows the reward-immutability check rather than
  removing dead code. **Fails conditions 3 and 4.**
* **`progress_decision_tense`, `progress_switch_freeze`** — v106 opt-in fixes from 2026-08-29, named
  in `measurements/ai_v12_adversarial_review_2026-08-30.md`. Recent and deliberately opt-in.
  **Fails condition 4 outright.**
* **Umbrella-desugar targets** — the `damage_matrices_*` and `--unified-moves` families. OFF in
  isolation but written by a desugar the production config uses; deleting a target breaks the
  umbrella. **Fails condition 2 in substance if not in letter.**
* **The counterfactual / distillation instrument family** — armed by an active campaign.
  **Fails condition 4.**

**Measured-NULL but still KEPT — the sub-case worth naming.** A ledger NULL verdict is *not*
sufficient on its own. A flag whose ON value put parameters in a `state_dict` cannot be deleted
without refusing checkpoints still in the current lineage, however dead the lever measured. Those
stay, and the ledger verdict is a reason not to *enable* them rather than a licence to remove them.
Conflating "this lever did nothing" with "this code never ran" is the mistake this column exists to
prevent.

## (e) Second census — the 1,000–2,000 line band

No deletions here; this is the pool the next decomposition comes from. **19 files**, 0 over the
2,000 hard bound. Ranked by line count, with the seam I would cut.

1. **`agents/training/reward_manager.py` (1966)** — closest to the bound and the most urgent. Three
   separable things: the `RewardConfig`/`RewardClass`/`RewardBreakdown` *schema*; the *composition
   announcer* (`reward_class_composition`, `reward_config_digest`, `format_reward_composition`,
   ~130 lines of pure config-to-text touching no battle state); and the `Gen3RewardManager` engine
   from line 565 on. Cut the announcer out first as `reward_composition.py` — it is already fenced
   by its own comment banner, holds no state, and is the one piece with zero coupling to the
   per-turn hot path.

2. **`agents/training/cf_producer.py` (1899)** — already banner-sectioned. The cleanest seam is
   **checkpoint resolution + snapshot loading** (`step_from_checkpoint_name`,
   `resolve_latest_checkpoint`, `Snapshot`, `load_snapshot`, `_warm_the_compiled_graph`, ~230 lines)
   into `cf_snapshot.py`: it is about *models on disk*, not about counterfactual labels, and
   `cf_audit.py` plausibly wants it too.

3. **`agents/training/instrumented_ppo/ppo.py` (1826)** — one class, `InstrumentedMaskablePPO`, so
   the package's existing base-class-chain convention is the answer rather than a new module: lift
   the fold sequence inside `train()` into a `PpoFoldSequence` base, leaving `ppo.py` the
   orchestration. Follows `features_extractor`'s precedent exactly, and keeps every `state_dict` key
   and `inspect.signature` reader byte-identical.

4. **`agents/training/eval_callback.py` (1559)** — the most seam-rich file in the band: ~20
   module-level functions before any class. **Trace and snapshot housekeeping** (`prune_eval_traces`,
   `prune_eval_snapshots`, `persist_eval_snapshot`, `trace_filename_stem`) is pure filesystem
   retention with no eval-cycle state — lift to `eval_retention.py`. The worker-lifecycle trio
   (`spawn_eval_workers`, `kill_eval_workers`, `merge_eval_results`) is a second, independent cut.

5. **`agents/model/snapshot.py` (1450)** — two subsystems sharing a file. The **pin-history /
   git-hash provenance** group (`GitHashMismatchError`, `_hashes_agree`, `resolve_git_hash`,
   `_update_pin_history`, `_read_pin_history`) is self-contained, torch-free, and is exactly what
   `main.sidecar_audit` reasons about — lift to `snapshot_provenance.py`. The
   `record_*` / `append_eval_result_row` metadata-writing group is a second cut.

6. **`agents/training/cf_audit.py` (1439)** — the statistics are the seam. `wilson_ci`, `_ranks`,
   `_is_flat`, `spearman`, `cluster_bootstrap_ci`, `_cluster_pools`, `cluster_bootstrap_diff_ci`,
   `sd_true_excess` (~180 lines) are pure NumPy with no audit concepts, and the cluster-bootstrap
   pair is conceptually re-implemented elsewhere in the tree. Lift to `agents/training/stats.py` —
   the highest reuse payoff in the band, and the one cut that would pay for itself twice.

7. **`agents/model/damage_tables.py` (1433)** — a flat list of ~14 independent `build_*` buffer
   constructors. Split by *what the buffer is about*: the **belief priors** (`build_opp_spread_prior`,
   `build_species_nature_prior`, `build_species_ev_prior`, `build_hp_type_prior`, `build_item_prior`,
   `invert_nature_evs`) into `belief_tables.py`, leaving the damage/type/stat buffers. Each function
   is independently testable, so this is the lowest-risk cut in the band.

8. **`main/prober/web/app.py` (1352)** — `create_app` spans lines 152–1087: one ~935-line factory
   holding every route. Cut by **view group** into routers (`routes_scan.py`, `routes_battle.py`,
   `routes_analyze.py`, `routes_jobs.py`) mounted by a thin `create_app`. Move the `_form_int` /
   `_form_float` / `_safe_next` / `_is_https` / `_client` helpers to `web_util.py` first as the cheap
   opening move. The committed `openapi.json` contract is the gate that makes this safe.

9. **`main/harvest.py` (1319)** — the pipeline is already linear: build candidates → score → select.
   **Scoring** (`score_candidates`, `_win_prob_batch`, `_worker_init`) is the only part with a torch
   dependency; lifting it to `harvest_score.py` makes the candidate/selection half model-free and
   testable without a checkpoint.

10. **`main/search_dividend/search.py` (1196)** — `SearchEngine` is lines 304–1139, ~835 lines. The
    module-level predicates below it (`branchable`, `_selectable_across_worlds`, `_terminal_label`,
    `_no_arm_reason`) are pure and belong in `search_predicates.py`; the real cut is separating **ply
    expansion** from **budget/width bookkeeping** inside the class, mirroring how `deepen.py` and
    `budget.py` already split those concerns beside it.

11. **`agents/model/delivery_graph.py` (1189)** — `build_graph` is lines 242–999, a ~757-line single
    function, and *that*, not the file length, is the actual defect. Cut it per EDGE FAMILY into
    `_edges_<family>(...)` helpers appended to one node/edge accumulator; `to_dot`, `_check` and
    `main` are separately liftable as `delivery_graph_render.py`.

12. **`agents/model/damage_op_blocks.py` (1182)** — one class, `DamageOperatorBlocks`, mixed into
    `DamageOperator`. Split by DIRECTION — incoming blocks vs outgoing blocks — as two mixins, which
    is the axis the op's own layout contract already uses.

13. **`agents/training/distill_anchor_test.py` (1136)** — a test file, **exempt** (single subject:
    `distill_anchor.py`). Listed for completeness; splitting it would scatter one module's spec,
    which is precisely what the exemption exists to prevent.

14. **`main/prober/model.py` (1091)** — the view-model layer. Seam: per-view builders (`battle`,
    `scan`, `analyze`) into a `model/` package, one module per view, mirroring the `engine/` package
    split that already happened next door.

15. **`agents/training/selfplay_callback.py` (1072)** — the promotion/gating logic and the sentinel
    pool management are independent; lift the `SnapshotPool` interaction into `selfplay_pool.py`.

16. **`agents/model/damage_op_pairwise.py` (1065)** — the pairwise benches (`pairwise_bench_*`) are
    parallel, independent kernels over the same context; group them by bench family. Note this file
    is one of the three still computing the now-vestigial `opp_spe_std` (see below).

17. **`agents/model/damage_op.py` (1026)** — the hub the two above mix into. Leave it; it shrinks
    when they do.

18. **`main/train/config.py` (1014)** — `_resolve` inheritance, the desugars, and `validate` are
    three phases already named in the module docstring. The desugars (`desugar_umbrella_flags` and
    friends) are the cleanest lift, and `main.checkargs` already imports them as a unit — an existing
    consumer boundary is the best evidence a seam is real.

19. **`agents/training/obs_materializer.py` (1007)** — just over the line. `scan_record` and
    `materialize_from_record` are the two public entry points over a shared replay; the seam is the
    `_ReplayObsPlayer` internals into `obs_replay_player.py`.

## Banked follow-up from this round

`DamageOperator._p_outspeed` still ACCEPTS `opp_spe_std`, and ~12 call sites across `damage_op.py`,
`damage_op_blocks.py` and `damage_op_pairwise.py` still compute it from
`SPECIES_SPREAD_PRIOR[..., _SB_SPE, 1]` and pass it. It now has no consumer.

This was deliberately NOT removed in the same pass, and the distinction is the v88 rule's own: the
deleted BRANCH is code that never ran, whereas those lookups **ran every forward and were
discarded**. Removing them is behaviour-preserving and a small hot-path win, but it is a live-physics
edit across three files rather than a provably-inert deletion, and it wants the authoritative
`damage_op_probe_fuzz_test.py` gate rather than a purge's test set. Smuggling it in would have made
one commit half-provable.

`bidir_threat_test.py::test_p_outspeed_is_the_fixed_scale_logistic_and_ignores_the_std` asserts the
std cannot reach the divisor, so a re-introduction has to change a test rather than silently change
the physics.

**Separate finding, PRE-EXISTING, not produced by this round.** `python -m main.checkargs <run>`
resolves a FORK PARENT's `model_config.json` relative to the current checkout, so inside a linked
worktree — where `models/` does not exist — it always falls back to ARGV-ONLY checking and prints a
warning naming the paths it tried. It is loud rather than silent, so it is not the C1 defect class,
but it means the inherited-value half of the check is inert for every agent working in a worktree,
which is most of them, on the exact tool built to catch inherited-value defects.
`utils.paths.main_models_dir()` exists for precisely this and is the fix.
