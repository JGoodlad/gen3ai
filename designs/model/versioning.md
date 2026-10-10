# Model versioning — the package, the sanitizers, and the resume-immutable families

**Lifted out of `src/agents/model/CLAUDE.md` on 2026-09-08.** This doc OWNS its subject and carries
the same ALWAYS-CURRENT obligation as that leaf — update it in the same pass as the code.
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) is the doc of record for what the model IS; where the
two disagree, ARCHITECTURE.md wins.

## The `model_version` package, and what a save writes

**`model_version` is a PACKAGE** (2026-08-23; it was a single 2,000-line file, exactly at the size
gate's hard bound). `agents/model/model_version/__init__.py` is a pure re-export hub, so every
`from agents.model.model_version import <name>` across the ~48 import sites resolves unchanged:

| module | holds |
|---|---|
| `constants.py` | `MODEL_CONFIG_VERSION` · `ARCH_SIGNATURE` · **`OBS_SEMANTICS_VERSION`** (+ `OBS_SEMANTICS_REASON`: the first config version whose observation VALUES mean something different, shape change or not — a reader's marker for the weights-fit-but-inputs-changed class, compared with a checkpoint's recorded `config_version` by `main/prober/arch_status.py`; gates no resume; pinned beside the obs golden by `obs_semantics_test.py`; `designs/prober/arch_drift.md`) · `ModelVersionError` · the reward-immutable field table + `_reward_flag_repr` |
| `migrations.py` | `MIGRATION_FLOOR` · `SIGNATURE_FIRST_VERSION` · `_migrate_config`, **including the PRE-FLOOR HISTORY archive** (a deliberate record of what every deleted branch did — do not trim it) |
| `fields.py` | `ModelVersionFields` — the dataclass field block alone. Declaration ORDER is the constructor's positional order and `asdict()`'s key order |
| `construct.py` | `from_layout_and_policy_kwargs` |
| `compat.py` | `check_compatible` — the gate that runs on **every** load |
| `resume_checks.py` | `check_opponent_compatible` + the six resume-immutable hparam gates |
| `spec.py` | `ModelVersion` = the fields plus one mixin per family, and the JSON IO |

`ModelVersion` is assembled from MIXINS, which trades a file-size problem for a **base-list**
problem: a family can drop out of the bases without any import failing, and the class would still
construct, still round-trip through JSON, and simply stop gating. `model_version_hub_contract_test.py`
pins the base list, every gate method by name, the hub's pre-split export list (recovered by AST),
the no-submodule-imports-its-own-hub cycle guard, and that the migration archive survived.

Every model save writes the **run-level** `model_config.json` + `metadata.json` at the run root via `save_model_snapshot()`, plus a **per-checkpoint** `.json` sidecar beside each checkpoint `.zip` (`write_checkpoint_metadata`, derived from the zip path). Periodic + forced checkpoints `.zip` live in `<run>/checkpoints/` (so their sidecar lands there too); the run-level config/metadata stay one level up at the run root. Loading goes through `load_model_snapshot()`, which resolves the zip then searches **its dir AND its parent** for `model_config.json` (so the run-root config is found even when the zip is in `checkpoints/`; `load_foreign_opponent` does the same) and runs `check_compatible()` before `MaskablePPO.load()` — a mismatch fails fast with a clear error rather than silently loading bad weights. (`snapshot_history` keys + the `worktree.py` resume lookup stay BARE basenames, e.g. `checkpoint_123_steps.zip`, regardless of the subdir.) Both files carry a top-level **`num_timesteps`** — how far the run had trained at that write, "latest" at the run level (overwritten every save, unlike the immutable `original_command` / `lineage` / `pin_history`) and per-checkpoint in each sidecar, so the JSON-only offline tools (`main.lineage`, `main.sidecar_audit`, `main.dose`) can read a run's step count without opening a `.zip`; a run that predates the key is ABSENT, which reads as unknown and never as 0.

## Judging a deleted extractor kwarg, and the two sanitizers

1. **Judge it**, into exactly one of `snapshot._DEAD_FEK_*`:
   - **`_DEAD_FEK_INERT`** — no value of it selected anything in the surviving forward (it only
     SIZED or INITIALISED a deleted module, was training-only, or its branch was unreachable in
     production). Popped unconditionally.
   - **`_DEAD_FEK_JUDGED`** — some value fed a forward this code can no longer reproduce. Record the
     one value that IS still reproducible; every other value is REFUSED loudly. Two things put a
     flag here: a byte-identical state_dict across its values (nothing shape-based can catch the
     swap), or an ON value that named PARAMETERS (popping it hands SB3 an unplaceable state_dict).
2. **`_migrate_config`** needs a matching entry only if the name could still appear in a config at
   or above `MIGRATION_FLOOR`; below the floor the blanket PRE-GENERATION refusal already owns the
   config half. That asymmetry is pinned by `dead_kwargs_sanitize_test.py`.
3. **Update `ctor_kwarg_snapshot_test.CTOR_KWARGS_V96`** — last, and only after steps 1–2.

That snapshot is the tripwire, and it exists because the machinery cannot detect this failure about
itself: the sanitizer is a CURATED list, so a forgotten name produces no error at deletion time and
no failing test — only a bare `TypeError` months later. **Measured 2026-08-17 over the 89 archived
runs carrying a checkpoint:** 23 distinct rejected kwarg names, **five in neither list**
(`mask_incoming_damage_obs` / `mask_active_move_scalars_obs` / `mask_move_effects_obs` at v48,
`hp_type_belief_mode` at v52, `spread_belief_nature_marginalize` at v66), present on **70 of the 89
runs**; 7 of them (generations `ai_v9_01`–`ai_v9_07`) reached the TypeError rather than a judgment.
All five are JUDGED now and the archive is at 0 TypeErrors.

**Do not confuse this with the prober's sanitizer.** `main.prober.model.sanitized_load_custom_objects`
is pure set math over the live signature and NEVER refuses — reading an archived model may be
approximate as long as it SAYS SO (`dropped_kwargs` rides the drift banner). The curated one serves
the paths where being wrong corrupts something, so it refuses (69 of 89 runs). Both are needed;
neither delegates to the other.

## The resume-immutable hparams, and the reward-config family

**Resume-immutable training hparams (value-meaning, NOT weight-shape).** A hyperparameter can
be wrong-to-change-mid-run without changing any weight shape — `vf_coef` (`--vf-coef`) is the
first: it rescales the value head's gradient on the shared trunk, so a forgotten/typo'd flag on
resume would silently drift training. These are recorded on `ModelVersion` (→ `model_config.json`)
but **deliberately excluded from `check_compatible`** — that gates EVERY load, including the frozen
eval / self-play-pool opponents, where the forward is identical regardless of the value
and a false rejection would break league play. Instead they get a dedicated check
(`ModelVersion.check_vf_coef`) invoked **only on the training-resume path** via
`load_model_snapshot(..., enforce_vf_coef=…)`; `train_rl_agent.py` FATALs on mismatch exactly like
an arch error. To add another such hparam, follow the optional-feature playbook in
`src/agents/model/CLAUDE.md` (field +
`MODEL_CONFIG_VERSION` bump + `_migrate_config` default) **plus** a dedicated `check_*` + an
`enforce_*` opt-in on `load_model_snapshot`, and leave it out of `_WEIGHT_FIELDS`.

The **reward-config** hparams are the same kind, bundled into one check: `victory_value`, `terminal_indicator`, `draw_penalty` (their flags `--victory-value` / `--terminal-indicator` / `--draw-penalty` are DELETED, P11b batch (c) — constants of the trainer namespace, still RECORDED;
the DRAW/250-turn-timeout score of the SIGNED terminal — −35.0 is what an UNRECORDED field means; the
production value is 0.0 with the indicator terminal / victory 1.0), and the no-progress
clock's two OBS switches `progress_decision_tense` / `progress_switch_freeze` (their flags and both clocks' branches are
DELETED, P11d — constants of the trainer namespace, False, still RECORDED; a recorded True is refused). All are recorded on
`ModelVersion` and enforced on resume by **`check_reward_config`** (FATAL on drift; there is no flag to re-pass, so the error says: run it PINNED to its own commit, or start a fresh run), excluded from `check_compatible` because a frozen eval / pool
forward never reads the reward. They are reward-VALUE changes — **no `ARCH_SIGNATURE` bump**.

🚨 **The 14 SHAPED-reward fields LEFT the config at v122** (`gen3_shaped_reward_deletion_v1`,
2026-09-26): `bias_additivity`, `mat_alive_weight`, `bias_redesign`, `switch_bias_weight`,
`self_ko_hp_penalty`, `drop_redundant_bias`, `drop_switch_bias`, `all_shaping_pbrs`, `stall_pbrs`,
`no_progress_penalty`, `hand_shaping`, `pbrs_material`, `pbrs_belief`, `no_progress_tax_armed`.
`_migrate_config` POPs them version-independently (so a frozen load of any vintage works), and a
RESUME or FORK of a config that recorded a shaped reward is REFUSED from the raw file before the
migration runs — `agents.model.model_version.shaped_reward` (`ShapedRewardCheckpointError`,
enforced in `main.train.config.resolve_config` and `main.checkargs`), never silently continued on
the terminal alone. The fix it names is a pin to ≤ `029cee83`. Detail:
`designs/training/reward.md`.

🚨 **The RETIRED LEVERS left the config at v131** (deletion pass L1, `gen3_retired_levers_l1_v1`, owner-approved
2026-10-02): `use_popart`, `value_dist_mode` (+ `value_dist_bins` / `vmin` / `vmax` / `coef`), `value_from_dist`,
`value_tail_weight`, `win_prob_coef`, `win_prob_pbrs_coef`, `win_prob_pbrs_source`, `win_prob_pbrs_frozen`.
`_migrate_config` POPs them version-independently (`cls(**data)` TypeErrors on a stale key). They are judged in
ONE table, `model_version/retired_levers.py` (`RETIRED`; **each later deletion unit APPENDS its levers**), in the
shape `shaped_reward.py` set:

* a **STRUCTURAL** lever (`use_popart`, `value_dist_mode != 'none'`, `value_from_dist` — an ON value named
  PARAMETERS or a critic route the surviving code cannot rebuild) recorded ON is REFUSED on EVERY load
  (`refuse_structural`, called from `_migrate_config`): popping it would hand SB3 an unplaceable state_dict;
* a **TRAINING-ONLY** lever (the tail weight, the aux-BCE coefficient, both PBRS rungs) pops silently on a frozen
  load (a forward never reads it), and a RESUME or FORK of a run that recorded one ON is refused from the RAW
  file (`check_no_retired_levers`, called by `main.train.config.enforce_not_shaped_parent` and by
  `main.checkargs.retired_levers_finding`), naming the flag and the pin (`RetiredLever.last_commit`);
* the PICKLED half — a zip's `policy_kwargs` `use_popart` / `value_from_dist` and `features_extractor_kwargs`
  `value_dist_*` — is `snapshot._DEAD_POLICY_KWARGS_JUDGED` / `_DEAD_FEK_JUDGED` / `_DEAD_FEK_INERT`, applied by
  every sanitizing loader (`_patch_historical_floor`; `snapshot.historical_load_kwargs` for `play.py`; the prober's
  `sanitized_load_custom_objects`). A BARE `MaskablePPO.load` of a pre-deletion zip TypeErrors — that is the
  failure the lists exist for, pinned by `agents/model/retired_levers_test.py`.

Today this is belt-and-braces: every v121+ run on record recorded every lever OFF. No `ARCH_SIGNATURE` bump, no
`MIGRATION_FLOOR` change.

🚨 **L2 (v132, `gen3_retired_levers_l2_v1`) appended** `win_prob_lambda` (+ `win_prob_lambda_truncated`),
`win_prob_rollout_target` (+ `_r` / `_mode` / `_weight`), `win_prob_dense_aux` and the STRUCTURAL bools `dense_aux` /
`value_true_team` (a `DenseAuxHead` / `TrueTeamValueReadout` in the state_dict has no home, so an ON record is refused on
every load; `snapshot._DEAD_FEK_JUDGED` carries the pickled extractor kwargs). Pin: `LAST_COMMIT_L2`.

🚨 **The DISTILLATION and SEARCH-TEACHER levers left the config at v133** (deletion pass L3,
`gen3_retired_levers_l3_v1`, owner-approved 2026-10-02: "delete all, port none"; stamp-only, no `ARCH_SIGNATURE`
bump). Removed `ModelVersion` fields: `distill_target`, `distill_topk`, `distill_gate`, `distill_gate_tau`,
`distill_beta`, `teacher_scan_limit`. They ride the SAME `retired_levers.py` table: `distill_target != 'kl'`,
`distill_gate != 'none'` and `teacher_scan_limit != 60` are TRAINING-ONLY retired levers (a resume or fork of a run
that recorded one is refused, naming the flag and the pin); `distill_topk` / `distill_gate_tau` / `distill_beta` are
INERT retired fields (popped silently, any value). The pin is `LAST_COMMIT_L3`
(`615a764fdb7e05abfcc1575797e2c83eb61c3ea1`). 🚨 **`--distill-coef`, `--distill-teacher`, `--search-teacher` and
the rest were never recorded fields**, so a run that used them with the default knobs cannot be recognised from
`model_config.json`: its recorded argv fails argparse on an UNPINNED resume instead. Flag list:
`designs/deleted_flags.md`; history: [`../training/exploiter_and_distillation.md`](../training/exploiter_and_distillation.md).

🚨 **The COUNTERFACTUAL TRAINING HALF left the config at v134** (deletion pass L4, `gen3_retired_levers_l4_v1`,
owner-approved 2026-10-02: "delete all, port none"; stamp-only, no `ARCH_SIGNATURE` bump). Removed `ModelVersion`
fields (16): `cf_records`, `cf_records_keep`, `cf_winprob_coef`, `cf_head_only`, `cf_label_lag_steps`,
`cf_label_likelihood`, `cf_evidential_coef`, `cf_evidential_reg`, `cf_twin_coef`, `cf_shadow_coef`,
`q_winprob_coef`, `q_winprob_onpolicy_coef`, and the FOUR STRUCTURAL head toggles `cf_evidential`, `cf_twin_heads`,
`cf_shadow_critic`, `q_winprob_mode`. The toggles are the first STRUCTURAL levers since L2: an ON value built modules
(`CfEvidentialHead`, two extra `WinProbHead`s, `ShadowValueHead`, a `QWinProbHead`) that the surviving extractor has no
home for, so `retired_levers.refuse_structural` refuses a config recording one ON on EVERY load and
`snapshot._DEAD_FEK_JUDGED` carries the four pickled extractor kwargs (OFF pops, ON is refused). The coefficients and
`cf_records` are TRAINING-ONLY retired levers (a resume or fork of a run that recorded one live is refused, pin
`LAST_COMMIT_L4 = cbd20111…`); `cf_records_keep` / `cf_head_only` / `cf_label_lag_steps` / `cf_label_likelihood` /
`cf_evidential_reg` are INERT retired fields. `--team-pfsp` / `--exploiter-ladder` / `--cf-label-supply` /
`--cf-producer-args` / `--cf-supply-starve-*` were never recorded fields. Every v121+ run on record recorded all of it
OFF (2026-10-02 archive scan). `QWinProbHead` survives as the ride-along A head's scorer class only. Flag list:
`designs/deleted_flags.md`; the producer / audit stack that reads old runs' rings is kept (manifest D6).

🚨 **THE X5 VERSION BREAK RAISED THE FLOOR (config v144, `gen3_x5_version_break_v1`, 2026-10-07)** — the ONE planned
checkpoint break after X5's adoption, and ONE bump for all of its parts (later parts append to v144's comment and
CHANGELOG entry, never a second bump). `ARCH_SIGNATURE` → `gen3_x5_version_break_v1`, `MIGRATION_FLOOR` → 144,
`SIGNATURE_FIRST_VERSION["gen3_x5_version_break_v1"] = 144` (`migration_floor_test` keeps its rule). The floor
RISES with the signature (contra the legacy manifest's provisional D-L1 "decouple") because the break's later parts
reshape weights and change behaviour, so NO pre-break checkpoint — blob or fixed_mass — is reproducible at HEAD;
every one runs PINNED. `model_version/version_break.py` is the one home of the consequences:

* **`LAST_BLOB_COMMIT`** (`f7567a9f…`) — the last commit that builds `belief_tokens='blob'` and the last pre-break
  commit; every refusal names it, nothing else spells the hash.
* **The pre-floor diagnosis** — `_migrate_config`'s PRE-GENERATION refusal appends `pre_break_diagnosis`: a
  v121–v143 config recording `belief_tokens: "blob"` (or lacking the key below v136, where blob was the only past) is
  told the blob path is DELETED; one recording `fixed_mass` that it is a pre-break X5 checkpoint whose weights the
  break reshaped. Both name the pinned fix.
* **The resume / fork refusal** — `check_post_break` reads the RAW parent config BEFORE any loader, so
  `main.train.config.enforce_not_shaped_parent` exits `FATAL_CONFIG` with that text (a flagless resume would
  otherwise fall back to OFF defaults and fail somewhere unrelated), and `main.checkargs` prints it as the
  "pre-break parent" block (`version_break_finding`; ADVISORY when the child runs a pin at or before the commit).
* **The pickled kwarg** — `belief_tokens` is `_DEAD_FEK_JUDGED` with `fixed_mass` the reproducible value; any other
  value (`blob`) is refused by `refuse_pickled_belief_tokens` with the same deletion reason, and
  `snapshot.load_checkpoint_strict` runs `refuse_deleted_pickled_kwargs` on the zip itself, so a caller whose
  sanitizer never refuses (the prober's set math) or passes none cannot hand SB3 a blob kwarg.

* **Part 2 (the EXACT-refactor bundle: audit F1 / F6a / F7a / F16b + the blob leftovers)** adds NO bump and judges
  no extractor kwarg (none was removed, so nothing new under `_DEAD_FEK_*`); the policy kwarg `critic` survives,
  defaulting to `winprob` and accepting only it. A pre-break `fixed_mass` checkpoint's state_dict now holds keys this
  code refuses (the deleted value tower; `flat_intent_head.out.bias`), which `pre_break_fixed_mass_reason` names.
  Retired modules leave a plain `None` (`extractor_api.drop_child`), so a strict load REPORTS a retired module's
  keys as unexpected instead of swallowing them.

* **Part 3 (the OBS-FACTS append, `gen3_obs_facts_v1`)** adds NO bump: the observation grows 2761 → 2845 (the
  OBS-FACTS block, its last block) and the new STRUCTURAL field `obs_facts` (`--obs-facts {off,v1}`) is born AT
  v144, so it has no migration branch (every pre-144 config is refused at the floor; `total_dim`, a weight field,
  would refuse a 2761-wide one anyway). `check_compatible` compares `obs_facts` (`v1` builds `ObsFactsInject`, the
  state_dict delta); `pre_break_fixed_mass_reason` names the observation change.

Every post-floor `if version < N` branch in `_migrate_config` (v122–v143) is now UNREACHABLE and is left in place;
moving their history into the archive block and deleting them is legacy-manifest unit R1 / L1, not the break's.

🚨 **v145 RAISED THE FLOOR AGAIN (`gen3_mon_tied_gain_v1`, 2026-10-07; owner: "fix those non-equivariant
knobs")** — the op's learned `out_gain` tied across our TEAM SLOTS and their mons (the incoming per-mon rows 72 → 12,
the Choice-Band tail 12 → 2, the render matrices' per-mon cells; production `damage_op.out_gain` 99 → 29), landed
before any v144 checkpoint was trained. `ARCH_SIGNATURE` → `gen3_mon_tied_gain_v1`, `MIGRATION_FLOOR` → 145,
`SIGNATURE_FIRST_VERSION["gen3_mon_tied_gain_v1"] = 145`. The same `version_break.py` machinery carries it:
`pre_break_diagnosis` gives a v144 config `v144_reason()` (the tie, the shape change, the pinned fix), and
`check_post_break` reports **`LAST_V144_COMMIT`** (main's last v144 commit) as the pin a v144 resume must run at or
before; a v121–v143 config keeps its belief-specific reason and `LAST_BLOB_COMMIT`. No migration branch sits between
144 and 145. The archive held NO config at or above v144 when it landed (scan 2026-10-07: max v138). Tests:
`mon_tied_gain_test.py`.

**v147 (`gen3_static_port_v1`, 2026-10-09)** reshaped only `--token-encoding static` (its type pair summed, its op
content's outgoing route a set function) and added two STRUCTURAL fields, `mon_hazard_cost` / `move_actor_state`
{off,on}. A v145 / v146 `static` record is REFUSED in `_migrate_config` (its state_dict has no home; the archive held
none: every static run is the pinned screen's v143); `legacy` stamps through and both fields default `off`. No
`ARCH_SIGNATURE` or floor change (production is `legacy`, byte-identical).

**v148 (`gen3_spikes_entry_base_types_v1`, 2026-10-09)** — a forward fix with NO field and no weight shape:
`DamageOperator.spikes_entry` reads the mon's BASE (species) types for the Flying check, not the obs type columns
(current types). `_migrate_config` stamps every past config through (`if version < 148`); no `ARCH_SIGNATURE` or floor
change. Production's `x` cell differs for an ACTIVE mon whose current types differ from its species'. Test:
`static_port_test.py`.

**v149 (`gen3_spikes_entry_species_levitate_v1`, 2026-10-09)** — the same fix's ability half, again NO field and no
weight shape: `spikes_entry` reads Levitate from the SPECIES (`SPECIES_TRAP_PRIOR[:, 3]`, exactly 0 or 1 in gen 3) instead
of the current-ability column. `_migrate_config` stamps every past config through (`if version < 149`); no
`ARCH_SIGNATURE` or floor change.

**v150 (`gen3_static_recovery_v1`, 2026-10-09)** added three STRUCTURAL fields, each gated in `check_compatible`:
`trunk_layers` (int, 2 = production; 3 / 4 append identity-init trunk rounds), `switch_hazard_cost` and `eot_residual`
({off,on}). A pre-v150 config migrates to 2 / off / off (the only possible past); no `ARCH_SIGNATURE` or floor change
(production builds byte-identically). The NAMED ARM `--arch static_recovery` writes all three (and the v147 facts) as
recorded fields, so a launcher restart inherits them from `model_config.json`.

**v151 (`gen3_toxic_stage_scale_v1` + `gen3_wish_flag_truth_v1`, 2026-10-09)** — two OBSERVATION-VALUE fixes with NO field and
no weight shape. (1) The per-mon toxic counter cell is `min(stage, 15) / 15` (the engine's `tox` stage over its cap), where
it was `min(stage, 8) / 8` (every tick from the 9th on read as the 8th); every toxic mon's cell value changes (`n/15`, not
`n/8`), so a checkpoint trained on the old scale reads a re-scaled input from here on. (2) The board's Wish flag means "a
Wish lands at the NEXT end-of-turn residual" (it read "cast last turn", the wrong phase at a replacement decision after the
end-of-turn faint). `_migrate_config` stamps every past config through (`if version < 151`); no `ARCH_SIGNATURE` or floor
change (a signature bump would refuse every existing checkpoint, eval opponent and anchor to re-scale one cell and move
one flag). The obs golden moved on 107 of 991 decisions, all on those two cells. Tests: `toxic_stage_core_test.py`,
`obs_stage_truth_test.rs`.

**v152 (`gen3_endstate_facts_v1`, 2026-10-09)** added five STRUCTURAL fields, each gated in `check_compatible`:
`move_resolution_facts` ({off,full}), `status_facts` ({off,exact}), `ko_ramp` ({ramp,exact}), `drop_progress_clock`
({off,on}) and `g_ledger` ({coarse,eot}). A pre-v152 config migrates to off / off / ramp / off / coarse (the only
possible past); no `ARCH_SIGNATURE` or floor change (production builds byte-identically). Three of them build no
parameter (`ko_ramp`, `drop_progress_clock`, `g_ledger`: the string compare is the only gate, as for `speed_physics`);
the NAMED ARM `--arch endstate` writes all five (and the static-recovery / bundle levers) as recorded fields.

**v153 (`gen3_probe_facts_v1`, 2026-10-09)** added two STRUCTURAL fields, each gated in `check_compatible`:
`effective_stats` and `move_target_state` ({off,on}; each builds one zero-init projection). A pre-v153 config migrates
to off / off; no `ARCH_SIGNATURE` or floor change. `--arch endstate` records both.

**v154 (`gen3_move_set_closure_v1`, 2026-10-10)** added one STRUCTURAL field gated in `check_compatible`:
`move_set_closure` ({off,on}; no parameter — `on` changes what every non-active opponent slot's MoveBelief reinjection
reads, so the string compare is the only gate). A pre-v154 config migrates to off; no `ARCH_SIGNATURE` or floor change.
The NAMED ARM `--arch endstate` records it `on` (owner 2026-10-10, a pre-data amendment of the closing test); no
config bump came with that: an arm's overlay is a launch surface, and the field already existed.

**`ko_ramp = 'exact'` is computed in CLOSED FORM (`gen3_ko_exact_closed_v1`, 2026-10-10), with NO config, obs-semantics
or signature bump.** The value `exact_closed` that carried the closed form beside the 16-roll sum was legal for one
commit (`6e0a1a7a`, no bump: a new VALUE of an existing field, nothing to migrate, no parameter); the owner's "remove
the old one" made the closed form what `exact` computes and RETIRED the value. The judgment, rule by rule:

* **`MODEL_CONFIG_VERSION`: no bump.** No field is added, removed or defaulted differently; every recorded `ko_ramp`
  keeps its meaning (`exact` = THE exact P(KO) over the 16 rolls + crit, resolved over the observed HP interval — the
  same real function, now in closed form). The retired value is refused VERSION-INDEPENDENTLY, like
  `refuse_structural`: `_migrate_config` → `retired_levers.refuse_retired_values` (config), `snapshot.
  sanitize_dead_extractor_kwargs` (zip), the extractor build / `DamageOperator` (constructor) and the parser — each with
  the reason (`retired_levers.RETIRED_VALUES`, mirrored in `designs/deleted_flags.md` §4). No run recorded it.
* **`OBS_SEMANTICS_VERSION`: no bump.** No observation cell moved.
* **`ARCH_SIGNATURE`: no bump — a judgment against the letter of "any forward-math change bumps it".** The real
  function the `exact` forward computes is unchanged; only its floating-point evaluation is (the closed form and the
  sum differ by ≤ 8 eps (1 + (|top| + |hp_lo|) / w), measured ≤ 1.2e-6 on the op's fp32 KO column over the
  compile-parity rows, the class of difference a compile or a device change makes). A signature bump would refuse
  every checkpoint, production's included, whose forward did not change at all. Production (`ramp`) is
  byte-identical. A checkpoint trained under the 16-roll `exact` (the closing test's end-state arm, pinned at
  `b132b099`) loads here and evaluates its KO cells to that rounding bound.

## Where the per-version entries went

**The per-version entries that used to live here have moved to `designs/CHANGELOG.md` §4**
(verbatim). They described what each of v6–v57 added, in parallel with the root `CLAUDE.md`'s own
version narrative — two records of the same history that had drifted out of agreement with each
other and with the code.

- **What the architecture IS right now** — obs layout, the phase chain under the production config,
  what each head consumes, the `DamageOperator` block, the edge families, the flag table with
  `INERT` markings: **`designs/ARCHITECTURE.md`**.
- **What each version changed**: `designs/CHANGELOG.md` (history — do not quote as current).
- **The live values**: `MODEL_CONFIG_VERSION` and `ARCH_SIGNATURE` in `model_version/constants.py`. Read them
  there. This file deliberately no longer states them: a version number written into prose is stale
  the moment the next one lands, and quoting a stale one is how a v30 description got applied to a
  v59 model.

## The `critic` route has no fallback, its version gate, and how the mode is threaded

**The `winprob` route has NO FALLBACK, for the deleted `value_from_dist` route's exact reason** (the v89
orphaned-route class): there is nothing to fall back to — the scalar `value_net` and the whole SB3 value
tower are DELETED (version break part 2, audit F1; `policy.value_net` is a raising stub). A missing head, an
un-stashed `last_win_prob_logits`, or a batch-size disagreement with `latent_vf` (= `value_pooled`) all RAISE.

🚨 **The version gate matters more here than for a typical structural flag, and the reason is
worth internalising: BOTH routes return a `[B,1]` float tensor.** A flipped `critic` produces no
shape error anywhere, no load failure, and no metric that changes name — the run simply predicts a
different quantity for the rest of its life. So the string compare in `check_compatible` is the
ONLY thing standing between a resume and that, which is the same argument `win_prob_mode` and
`q_winprob_mode` make and the reason all three are gated identically.

**NO `ARCH_SIGNATURE` bump — at v109 (the mode) nor at v130 (the default flip, deletion pass D2,
2026-10-02).** The flip moved only what an UNTYPED critic meant on a FRESH argv (the `--critic` flag itself is deleted since P11b batch (b); `critic` is now the recorded field alone)
(`critic_mode.CRITIC_DEFAULT` = `winprob`); what an ABSENT RECORD means is a separate constant that did
not move (`CRITIC_UNRECORDED` = `shaped`: `ModelVersion.critic`'s field default and every
`getattr(policy, "_critic_mode", …)` read — it is what a RECORD without the key means. Since the version break's
part 2 the POLICY constructor's default is `winprob`, and any other value is refused before anything is built:
a `shaped` checkpoint is below `MIGRATION_FLOOR` and runs PINNED). So no checkpoint loads differently, a
flagless resume inherits its recorded critic, and `check_compatible`'s string compare still refuses a
critic mismatch — the warm-start hazard the old plan bumped the signature for cannot arise. (The earlier
plan here — "the signature bump belongs to the DEFAULT FLIP" — assumed the two meanings were one constant.)

`critic` is threaded as a POLICY kwarg (the class `use_popart` / `value_from_dist` were), which is why
it is absent from `agents/model/flag_registry.py`: that registry's declared scope is EXTRACTOR
architecture toggles, and this one reaches no extractor — the heads it selects between were already
built by their own flags. It rides `snapshot.current_model_version(critic=…)` and
`arch_toggles_from_model` so a frozen eval / pool / sentinel opponent's load gate sees it.

---

## Moved from the leaf (2026-10-10)

The full text of the model leaf's sections on this topic, moved here when `src/agents/model/CLAUDE.md` was cut to rules, commands, map and hazards (the leaf keeps a one-line pointer to each). Always-current like the rest of this doc; where it overlaps an earlier section, the earlier section is the fuller statement.

### Model versioning (`model_version/`, `snapshot.py`)

**`model_version` is a PACKAGE**; `__init__.py` is a pure re-export hub. What each module holds, what
a save writes, the two sanitizers and the recorded-`critic` version gate:
[`designs/model/versioning.md`](versioning.md). The playbooks are here.

**When you change an architecture constant:**
- `check_compatible()` catches the mismatch automatically — no extra steps needed
- Old models can't be loaded, which is correct (rapid iteration project)

**When you add an optional new feature** (new field with a sensible default):
1. Add the field to `ModelVersionFields` in `model_version/fields.py`
2. Bump `MODEL_CONFIG_VERSION`
3. Add one `if version < N:` block in `_migrate_config()` with `data.setdefault(...)`

🚨 **Every value `--arch production` writes must be a `ModelVersion` field**, including a DERIVED
row's enabling coefficient. A launcher restart strips `--arch` and inherits only what
`model_config.json` records. `opp_intent_coef` was not recorded until v125, and every such restart
FATALed. When the past value is unknown, the migration leaves the field `None` rather than invent it
(the v125 branch). Gate: `main/train/derived_toggle_resume_test.py`.

🚨 **When you change what an observation cell MEANS without changing a shape** (a re-scale, a re-defined flag — v151's
Toxic counter n/8 → n/15), set `OBS_SEMANTICS_VERSION` (`model_version/constants.py`) to the config version the commit
stamps, add its clause to `OBS_SEMANTICS_REASON`, and re-pin `obs_semantics_test.py`'s golden hash. `check_compatible`
cannot see this (shapes fit, so the weights load); the marker is what lets a READER — the prober's model views, via
`main/prober/arch_status.py` — refuse a checkpoint recorded below it. It gates no resume. The test fails when the obs
golden moves without it; a model-INTERNAL input fix the golden cannot see (an op edge cell) must be raised by hand.

**When you make a structural change** (different forward pass, new layer type):
1. Change `ARCH_SIGNATURE` in `model_version/constants.py` (e.g. `"gen3_attn_v1"` → `"gen3_lstm_v1"`)
2. Old models get a clear arch-family error on load

🚨 **The X5 VERSION BREAK (config v144, `gen3_x5_version_break_v1`) RAISED `MIGRATION_FLOOR` to 144**: every
pre-break checkpoint (blob or fixed_mass) is refused at the floor with the belief-specific reason and runs PINNED
(`model_version/version_break.py` — `LAST_BLOB_COMMIT`, `check_post_break`, the pickled `belief_tokens` judgment).
The post-floor `if version < N` branches (v122–v143) are unreachable and left in place (legacy manifest R1 / L1).
**v145 (`gen3_mon_tied_gain_v1`) raised the floor to 145** (the op's `out_gain` tied across team slots): a v144 config
is refused with `version_break.v144_reason()` and runs pinned at or before `LAST_V144_COMMIT`.

**When you DELETE an extractor kwarg** — the case with no automatic gate, and the one this project
has silently got wrong five times. Every archived checkpoint keeps the deleted name pickled in
`policy_kwargs["features_extractor_kwargs"]`, and **SB3 rebuilds the extractor from the ZIP, not
from `model_config.json`** — so a deleted kwarg `TypeError`s every training resume, frozen pool
opponent and eval worker that touches such a checkpoint. `_migrate_config`'s `MIGRATION_FLOOR` does
NOT cover it: the pickled kwargs carry no `config_version` for a floor to apply to.

The judging procedure — `_DEAD_FEK_INERT` vs `_DEAD_FEK_JUDGED`, when `_migrate_config` needs a
matching entry, and `ctor_kwarg_snapshot_test.CTOR_KWARGS_V96` LAST — is in
[`designs/model/versioning.md`](versioning.md). Do all three, in that order.

**⚠️ REORDERING a module's parameters silently breaks the optimizer on resume.** SB3/torch save+load
the Adam optimizer state **by parameter POSITION, not name**. So if a refactor changes the *order*
`named_parameters()` yields (e.g. building submodules in `__init__` in a different sequence — the
`gen3_nature_ev_belief_v1` bug, where `SpreadBelief.__init__` moved `reinject`/`norm` before
`stat_head`), a resume's **weights** still load fine (name-keyed `load_state_dict` → arch check PASSES)
but the **momentum** (`exp_avg`/`exp_avg_sq`) gets assigned to the WRONG params. It then crashes in
`AdamW.step()` ("size of tensor a (128) must match b (5)") the moment a misassigned param of a
different shape first gets a gradient — **data-dependently, so it can survive many steps**, and (until
the guard) the broad `except` in the trainer masked it as a clean completion. Guard:
`main/train/checkpoint_state._validate_or_reset_optimizer_state(model, checkpoint_path)` runs on every resume and
**REMAPS the momentum to the current params BY NAME** — it reads the saved optimizer state + the saved
parameter NAME ORDER straight from the checkpoint zip (`policy.optimizer.pth` + `policy.pth`) and
rebuilds `opt.state` so each current param receives the momentum saved for its name, regardless of
registration order. So a reorder is **corrected**, not just caught: a **same-shape** reorder (which a
shape check CANNOT see and would silently scramble) now follows the name, and a name reused at a
different shape (or a genuinely new param) cleanly drops to fresh zero-init. **This means "append new
params LAST" is no longer load-bearing for optimizer correctness** — though still good hygiene. Falls
back to the legacy shape-only drop-all-momentum reset only if the zip can't be read (never crashes a
resume); no-op (momentum carried verbatim) on an aligned resume. Pinned by
`src/main/resume_optimizer_realign_test.py` (incl. the same-shape-reorder + zip-read cases).

**Resume-immutable training hparams (value-meaning, NOT weight-shape)** — `vf_coef`, the reward
fields — are recorded on `ModelVersion` but **deliberately excluded from `check_compatible`**, which
gates EVERY load including the frozen eval / pool opponents whose forward is identical
regardless. They get a dedicated `check_*` on the training-resume path only, and the trainer
FATALs on a mismatch exactly like an arch error. To add one: field + `MODEL_CONFIG_VERSION` bump +
`_migrate_config` default, **plus** a dedicated `check_*` and an `enforce_*` opt-in on
`load_model_snapshot`, and leave it out of `_WEIGHT_FIELDS`. The family list and the 2026-08-18
default flip: [`designs/model/versioning.md`](versioning.md).

The live `MODEL_CONFIG_VERSION` is in `model_version/constants.py`; per-version entries are in `designs/CHANGELOG.md`.

- **What the architecture IS right now**: [`designs/ARCHITECTURE.md`](../ARCHITECTURE.md).
- **What each version changed**: [`designs/CHANGELOG.md`](../CHANGELOG.md) — history, do not quote as current.
- **The live values**: `MODEL_CONFIG_VERSION` and `ARCH_SIGNATURE` in `model_version/constants.py`. Read
  them there. This file deliberately states neither: a version number in prose is stale the moment the
  next one lands, and quoting a stale one is how a v30 description got applied to a v59 model.

The mechanics above (what to bump when, the optimizer-reorder guard, the resume-immutable-hparam
playbook) are the durable part and stay here. When you add a toggle, follow those rules, then record
the entry in `CHANGELOG.md` and state the new truth in `ARCHITECTURE.md` — never narrate it here.

A startup smoke test (`_run_roundtrip_test` in `main/train/lifecycle.py`) saves to a temp dir and reloads before every `model.learn()` call — catches serialization issues immediately.

### The CRITIC MODE (`critic_mode.py`, the recorded `critic` field {shaped,winprob}, v109)

`gen3_winprob_critic_mode_v1`. `critic` is a RECORDED field with two historical values, and
`agents/model/critic_mode.py` is the ONE declaration of the legal set; since the X5 version break's part 2
(architecture audit F1) the policy BUILDS only `winprob` — its `critic` kwarg defaults to `winprob` and any other
value is refused before anything is built (the scalar `value_net` a `shaped` critic read is deleted with the
whole SB3 value tower). That module is
deliberately **torch-free and import-light**: `main.checkargs` promises not to import torch and
needs the legal set to validate an argv offline. There is NO `--critic` flag any more (deletion pass P11b batch (b)): the win-prob critic is a CONSTANT of every trainer namespace, and `critic` survives as the FIELD `model_config.json` records, string-compared by `check_compatible` (an absent record still means `shaped`).

| recorded `critic` | `_critic_value` returns | at HEAD |
|---|---|---|
| `shaped` (what an ABSENT RECORD means — `CRITIC_UNRECORDED`, for a `ModelVersion` / `model_config.json` read) | `value_net(latent_vf)` (historical) | NOT constructible: the policy refuses it, and every such checkpoint is below `MIGRATION_FLOOR` 144 — run it PINNED to its own commit |
| **`winprob`** (the only critic; the policy kwarg's DEFAULT) | `sigmoid(fe.last_win_prob_logits)` in **[0,1]**, `[B,1]` | the only one built; no value tower exists |

**Read the mode through `is_winprob`, never a bare `== "winprob"`** — one spelling, one answer, and
a `getattr(obj, "critic", "shaped")` read answers correctly through it.

🚨 **The version gate matters more here than for a typical structural flag: BOTH routes return a
`[B,1]` float tensor**, so a flipped `critic` produces no shape error, no load failure and no metric
that changes name — the run simply predicts a different quantity for the rest of its life. The string
compare in `check_compatible` is the only thing standing between a resume and that. (No
`ARCH_SIGNATURE` bump at v109 nor at the default flip; the version break's floor now refuses every
`shaped` checkpoint anyway.)
[`designs/model/versioning.md`](versioning.md) has both in full, and how the
kwarg is threaded.

### DELETED: PopArt, the distributional value head, `value_from_dist` (deletion pass L1, v131)

`popart.py`, `ValueDistHead` (`--value-dist-*`), the `value_from_dist` critic route, the CVaR value-tail
weight and the win-prob aux-BCE coefficient were levers of the shaped critic and of the Python env core, and
every v121+ checkpoint recorded them OFF. They are DELETED (`designs/deleted_flags.md`). The recorded fields
are popped by `_migrate_config`; a checkpoint that recorded PopArt / the dist head / `value_from_dist` ON is
refused on every load, and a resume or fork of a run that recorded ANY retired lever ON is refused
(`model_version/retired_levers.py` — each later deletion unit appends its levers to that one table). A
checkpoint's PICKLED `use_popart` / `value_from_dist` (policy kwargs) and `value_dist_*` (extractor kwargs) are
stripped on load by `snapshot._DEAD_POLICY_KWARGS_JUDGED` / `_DEAD_FEK_*` — a BARE `MaskablePPO.load` of a
pre-deletion zip TypeErrors (`play.py` and the prober's loaders sanitize; `snapshot.historical_load_kwargs`
is the helper). Detail: [`designs/model/versioning.md`](versioning.md).
