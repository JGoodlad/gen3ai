# `production_config.json` — where it came from

`production_config.json` is JSON with no comment syntax and is normally written **verbatim** by
`--sync-config` (sorted keys, never hand-edited), so its provenance cannot live in the file. It
lives here.

| | |
|---|---|
| **ADOPTED 2026-10-12: THE END STATE** | the closing test PASSED (`designs/endstate/design_endstate_closing_test.md` §9; ledger 2026-10-12 MEASUREMENT (MAJOR)), so the mirror took the 19 keys of the `--arch endstate` overlay (`src/main/train/arch_arms.py`) at their arm values: `token_encoding` static, `mon_hazard_cost` / `move_actor_state` / `switch_hazard_cost` / `eot_residual` on, `trunk_layers` 3, `move_resolution` on, `speed_physics` on, `value_threat_inject` false, `op_reduction` principled, `obs_facts` v1, `move_resolution_facts` full, `status_facts` exact, `ko_ramp` exact, `drop_progress_clock` on, `g_ledger` eot, `effective_stats` / `move_target_state` / `move_set_closure` on. **Every other key, and the `recipe` block, is unchanged** (both closing-test arms trained on that one recipe). See § "The end-state adoption (2026-10-12)" below. **Eight of those keys have since LEFT the mirror** with their fields (`switch_hazard_cost`, `eot_residual`, `speed_physics`, `op_reduction`, `status_facts`, `ko_ramp`, `g_ledger`, `effective_stats`; legacy deletion U1, 2026-10-15): their adopted behaviour is unconditional wherever the damage op is built — § "The legacy deletion's U1 (2026-10-15)". **Three more left with legacy deletion U3** (`move_resolution`, `value_threat_inject`, `move_resolution_facts`), with the seven per-action block keys the pre-adoption mirror already carried (`intent_move_cell`, `intent_threshold`, `intent_conditional`, `pair_outcome_cell`, `pair_outcome_switch`, `switch_branch_cell`, `conditional_threat_cell`, all `true` and INERT) — § "The legacy deletion's U3 (2026-10-15)" |
| **2026-10-20: `trunk_ffn_by_kind` = `off`** | B-KFFN (`gen3_trunk_ffn_by_kind_v1`, config v161; `designs/endstate/design_growth_builds_kffn_int.md` §1) added a STRUCTURAL field whose production value is OFF (its registry default: nothing built, byte-identical); the mirror records it so the arch surface covers it (`arch_surface_test`). X48 reaches `on` only through a fork's `--grow trunk_ffn_by_kind=on`. No other key changed |
| **2026-10-27: `trunk_ffn_dim` = `256`** | B-FFNW (`gen3_trunk_ffn_width_v1`, config v164; `designs/endstate/design_n4_capacity_screen.md` §2) added a STRUCTURAL field whose production value is the base width 256 (its registry default: nothing built, byte-identical); the mirror records it so the arch surface covers it. N4 reaches 1024 only through a fork's `--grow trunk_ffn_dim=1024`. No other key changed |
| **Run** | `models/ai_v12_02_winprob_critic/` — the WIN-PROB CRITIC era (the base the pre-adoption surface was built from) |
| **Built as** | `models/ai_v9_21_gen17_pfspoff_0820/model_config.json` (gen-17, v97) migrated to the v109 schema, with the **critic family alone** overridden |
| **Critic rows changed from gen-17** | 13 (listed below) · **non-critic rows changed: 0** |
| **Mirror updated** | 2026-09-06 |
| **`config_version` / `arch_signature`** | **157 / `gen3_mon_tied_gain_v1`** — 157 since 2026-10-15 (legacy deletion U3: ten fields left the config, a stamp-only migration with no signature or floor change; `designs/baselines.json`'s `production` entry declares `config_mirror_version: 157`); 156 earlier that day (legacy deletion U1: eight fields left). 154 from 2026-10-14: the end-state adoption moved `token_encoding` to `static` but left 145, and the static port (v147) REFUSES a pre-v147 config recording `static`, so the mirror was no loadable config (caught by `diagnostics_cadence_test`; pinned by `baselines_loadability_test::test_the_production_MIRROR_is_a_loadable_config_at_HEAD`). 154 is what the tested end-state runs record (`rb_es_end2_s2001`–`s2008`). Before that, v145 (2026-10-07, the op's `out_gain` tied across our team slots; `MIGRATION_FLOOR` 145) moved only these two keys (144 → 145, `gen3_x5_version_break_v1` → `gen3_mon_tied_gain_v1`), in the same follow-the-code window the break opened. The window: the X5 VERSION BREAK (2026-10-07, `designs/endstate/design_x5_belief_tokens.md` Decision record): `MIGRATION_FLOOR` rose to 144, so the mirror follows the CODE until the first run at the new signature exists (a mirror stamped below the floor is not a loadable config). Keys moved at the break: `arch_signature` (`gen3_event_record_v2` → `gen3_x5_version_break_v1`), `config_version` (122 → 144), `belief_tokens` LEFT the file with the field (X5's hypothesis tokens — the old `fixed_mass` — are the only belief representation; the mirror recorded `blob`), and part 3 (the OBS-FACTS append) moved `total_dim` 2761 → 2845 (the `production` entry's `total_dim` override with it) and added `obs_facts` (`"off"`: production reads none of the block). Every other key is unchanged. The previous window (2026-09-26, `gen3_event_record_v2`: 109 → 121, `total_dim` 2501 → 2761, `active_context_dim` 58 → 60) is history. When the first new-lineage run lands, re-sync from ITS `model_config.json` (`--sync-config`) and delete this note |
| **Before the window** | 109 / `gen3_critic_route_wave_v1` (the run's own schema) |
| **Previously** | gen-17's config verbatim (v97), from 2026-08-22 |

## The `recipe` block — the production TRAINING RECIPE (K10(a), 2026-09-30)

One key of this file is NOT a `model_config.json` field: the nested `recipe` block
(`main.train.recipe_surface`).

- **`recipe.fresh`** is N0's measured fresh recipe (`models/ai_v14_01_base`): every training knob it
  launched with, including the critic, its reward values, the supervision doses and
  `kl_controller` (the KL controller's constructor constants). `--arch production` applies it. A key
  that is also a recorded field of this file must EQUAL that field (the reader refuses otherwise).
- **`recipe.fork`** is what a generalist fork changes: E5's 5 epochs at a frozen `fork_lr` 5.6e-5
  (`models/ai_v14_03_lbat_e5`). Never applied on a fresh run.
- **It is HAND-EDITED**, unlike the rest of the file: no run config carries it. Change it only
  together with `designs/endstate/design_learner_recipe.md` §3.22 (the value, its SOURCE, and a
  Decision-record row) — `src/recipe_doc_gate_test.py` fails when either moves alone. `n_envs` 48 and
  `n_steps` 2048 are the SIZING study's to change; so is the fresh epoch count.
- **Invisible to every architecture consumer:** `agents.training.baselines.production_config()`
  strips it, `compare_production` exempts it (never a mirror-vs-run key delta), and the ARCH surface
  never reads it.
- **`--sync-config` CARRIES IT OVER** from the mirror it replaces (`delivery_graph.py`). After a
  re-sync, a `recipe.fresh` value that disagrees with the newly synced field makes the reader REFUSE
  — reconcile the two in the same commit.
- It changes the file's blob hash, so a recipe edit changes the `production_config@<hash>` tag that
  `arch_source` / `recipe_source` record — the tag names the file's CONTENT, recipe included.

## The bf16 trunk adoption (2026-10-14)

**One `recipe.fresh` row was added: `trunk_precision` `"bf16"`** (the hand-edited block; no run config carries it, and no
mirror field changed, so the `production_config@<hash>` tag moved only because the file's content did). The bf16 strength
A/B PASSED (`designs/endstate/design_bf16_strength_ab.md` §7: 8 × 8 at 15M, Δ̂ +1.06 pp, 90 % interval [−1.19, +3.31]), so
`--arch production` now trains the trunk rounds under bf16 autocast (`--trunk-precision fp32` is the explicit opt-out).
`recipe.fork` has no row for it on purpose: a fork inherits its parent's recorded precision. The row is not N0's (N0
predates the flag and trained fp32); its source is the A/B, cited in `design_learner_recipe.md` §3.22's table.

## The end-state adoption (2026-10-12)

**CONSTRUCTED, then VERIFIED against the tested runs.** The mirror was the pre-adoption mirror with the 19 overlay keys
of `--arch endstate` applied (eight of them left with U1 and three with U3, below) (the arm's own declaration, so the mirror and the arm cannot disagree on a value).
Verification against the closing test's eight END-STATE runs' recorded `model_config.json`
(`models/rb_es_end2_s2001`–`s2008`, trained at `62085750`): **every shared key is equal except** `config_version`
(the runs record 154; the mirror kept 145 until 2026-10-14, when it moved to 154, 156 since U1 and 157 since U3 — see the table above) and the five `ridealong_*` keys
(the runs typed `--ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 --ridealong-opp 5
--ridealong-rnd-variants all`; the ride-along heads are DETACHED critic READOUTS outside the ARCH surface, so production
keeps them at 0 / off, as the PRODUCTION arm's runs differed from the old mirror in exactly the same five keys). Keys
only the runs carry (`arch_source`, the eval / fork / promotion knobs, `policy_gae_lambda`, `opp_intent_coef`, …) are
run-level records, not mirror fields.

**What HEAD's end state differs from the TESTED `62085750` build** is classified in the adoption's ledger entry
(2026-10-12 BUILT "THE END STATE IS PRODUCTION"): one math-equal change (the closed-form exact KO, `5eef9934`,
max |Δ| ≈ 1e-6 relative on the same weights, greedy agreement 136 / 136), every other file behaviour-identical under
the production argv. `--arch endstate` stays as an ALIAS (its overlay equals the mirror; 8 keys since U3).
`--arch static_recovery`, redeclared against the new mirror at the adoption, was DELETED by the legacy deletion's U0
(2026-10-15). The pre-adoption production is reproducible by pinning `95d014fa`
(`--pin-commit 95d014fa --arch production`), and its baseline is kept under a legacy name in `baselines.json`.

## The legacy deletion's U1 (2026-10-15)

**Eight keys LEFT the mirror, and `config_version` moved 154 → 156** (`designs/ops/legacy_deletion_manifest_2026-10-12.md`
lane OP, D1–D6): `switch_hazard_cost` / `eot_residual` (`on`), `speed_physics` (`on`), `op_reduction` (`principled`),
`status_facts` (`exact`), `ko_ramp` (`exact`), `g_ledger` (`eot`), `effective_stats` (`on`). Their fields are gone from
`ModelVersion`; the values the mirror held are now the damage op's only physics, built whenever `damage_op` is (D-LD2
"DERIVED"), so the network the mirror describes did not change (the K9 learner golden is byte-identical; the delivery
graph moved only its meta `config_version`). The tested end-state runs still record the eight keys at v154 and still
load — they recorded the adopted values; a run that recorded another value with the op is refused, read pinned at the
tag `pre-legacy-deletion` (`model_version/adopted_levers.py`). An ABSENT key in a config written at v156+ (this mirror)
reads as the adopted value. Every other key, and the `recipe` block, is unchanged.

## The legacy deletion's U3 (2026-10-15)

**Ten keys LEFT the mirror, and `config_version` moved 156 → 157** (`designs/ops/legacy_deletion_manifest_2026-10-12.md`
lanes MR + VTI, rows B1 / B2 / C1): `move_resolution` (`on`), `move_resolution_facts` (`full`), `value_threat_inject`
(`false`) and the seven per-action block keys (`true`, INERT under move resolution). Move resolution and its restored
facts are now DERIVED — built whenever `opp_intent`, `damage_op`, `damage_outgoing`, both per-move matrices and
`damage_topk_k` > 0 are (D-LD2) — and the critic's threat projection and the seven blocks are gone, so the TRAINED
network the mirror describes is unchanged in shape (280 parameters, 2,609,183 elements). 🚨 The FRESH-INIT stream is
not: the deleted modules were constructed and then retired, so their RNG draws are gone and a fresh seed builds a
different initial network than before U3 (D-LD7; the ledger's U3 entry and
`research_state/measurements/legacy_deletion_u3_init_2026-10-15/`). The tested end-state runs still record the ten
keys and still load (adopted values, or blocks inert beside `move_resolution on`); a run that recorded
`move_resolution off` (the legacy family) or `value_threat_inject true` with the op is refused, read pinned at the tag
`pre-legacy-deletion`. Every other key, and the `recipe` block, is unchanged.

## Why it was CONSTRUCTED rather than copied from a run

The first arm of this era, `ai_v12_01_winprob_critic`, launched with a 38-flag argv that carried
only the critic block and the hyperparameters — so it recorded a config **44 shared fields** away
from gen-17, of which only 13 were the critic. The other 31 were the whole architecture surface
silently reverting to its OFF defaults: `edge_bias_families` `off`, `entity_topk_seats` 0,
`opp_belief_slots` false, `history_events` false, both intent heads off, every pointer cell off,
four beliefs off, both value routes off. That arm was killed and relaunched as `_02` with gen-17's
full surface plus the critic block.

**Mirroring `_01` would have redefined the production architecture by omission.** So this file is
built the other way round — gen-17's surface, migrated forward, with an explicit override list —
and then VERIFIED against the relaunched arm.

## The override set (13 rows differed from gen-17 when this file was written; the table shows what is LEFT)

⚠️ **Deletion pass L1 (config v131, 2026-10-02) removed the seven critic-implied KEYS from the mirror** — `use_popart`, `win_prob_coef`, `value_dist_mode`/`_bins`/`_vmin`/`_vmax`, `value_from_dist` and `value_tail_weight` (and the three `win_prob_pbrs_*` fields) no longer exist in `ModelVersion`, so there is nothing for the mirror to override; `designs/deleted_flags.md` has each with its citation. The rows below that name them are struck.

| key | gen-17 | production | why |
|---|---|---|---|
| `critic` | `shaped` | `winprob` | the mode itself |
| `hand_shaping` | true | *(deleted)* | the override LEFT the mirror with the field (config v122, `gen3_shaped_reward_deletion_v1`, 2026-09-26): the reward is terminal-only by construction, and the 14 shaped fields are gone from the mirror |
| `terminal_indicator` | false | **true** | REQUIRED |
| `victory_value` | 30.0 | **1.0** | REQUIRED — at 1.0 the undiscounted return IS `1{win}` |
| `draw_penalty` | −35.0 | **0.0** | REQUIRED — a [0,1] critic cannot rank a timeout below a loss |

`win_prob_mode` is `shaping` in both, so the mode's third implication changes nothing.
**`gamma` is not a `model_config.json` key at all** — the mode implies 1.0 and the value in force is
recorded in `metadata.json`'s `cli_args`, so it cannot be mirrored here.

**Every other key is gen-17's, byte-for-byte** — including `value_entity_pool`,
`value_entity_pool_full` and `value_threat_inject` (the last since deleted, legacy deletion U3), which SURVIVED the
swap: they inject additively into `value_pooled` / the value pool, which is exactly what the win head reads.

The v97 → v109 migration added 39 keys that did not exist in gen-17's config (`policy_grad_coef`, `rank_tripwire`, … — and a distillation block that deletion pass L3 has since removed; the `cf_*` family (13 keys) and the `q_winprob_*` family (3 keys) were removed from the mirror with their levers by deletion pass L4, which took 16 keys out of it). All take the
migration's defaults, and all read OFF or INERT in §6's generated table.

### One flag a hand-built relaunch argv will drop

`--intent-label-bot-weight` has an argparse default of `None` and a `_resolve` default of **1.0**,
while gen-17 ran it at **0.25**. It is part of gen-17's architecture surface and is NOT critic
family, so the relaunch argv must pass `--intent-label-bot-weight 0.25` explicitly or the run will
record 1.0 and diverge from this mirror. (`_01` recorded 1.0 for exactly this reason.)

## What derives from this file

Four things, which would otherwise each carry their own idea of the architecture:

- `designs/ARCHITECTURE.md`'s generated sections (`python -m agents.model.arch_tables`)
- `designs/architecture_graph.dot` + `src/agents/model/delivery_graph_snapshot.json`
- `designs/architecture_viewer.html` (`python -m agents.model.build_arch_viewer`)
- `extractor_compiles_test`, which compiles "the production arch" from it

## Which run is production is a JUDGEMENT

Nothing derivable answers it. `arch_tables_test.test_production_config_matches_newest_run` only
enforces that the mirror agrees with the **newest run in `models/` on every shared field** — it
cannot tell an experiment arm from a generation, and, as `_01` showed, the newest run is not always
the answer. The call is made by a human (or the orchestrator on the owner's direction) and recorded
here.

## Refreshing it

Top-down, in one commit, never partially:

```bash
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
python -m agents.model.delivery_graph \
    --sync-config models/<run>/model_config.json \
    --dot designs/architecture_graph.dot \
    --json src/agents/model/delivery_graph_snapshot.json
python -m agents.model.build_arch_viewer
python -m agents.model.arch_tables            # ARCHITECTURE.md's generated sections
```

Then update this file's table, re-derive the hand-written prose the new config contradicts, and
append a `designs/CHANGELOG.md` entry naming the switch. Both `--check` gates
(`delivery_graph --check`, `build_arch_viewer --check`) must pass afterwards.

⚠️ **Prefer `--sync-config` — a CONSTRUCTED mirror like this one owes a verification step.** A
hand-built config describes a run nobody has launched until it is checked against one, and every
artifact above then describes a model that may not exist.

## Verification (2026-09-06)

`ai_v12_02_winprob_critic` did not exist when this mirror was built, so it was verified against the
**relaunch argv** instead: gen-17's recorded `original_command`, minus the launcher-owned flags,
minus every flag the win-prob critic refuses, plus the critic block — parsed by the live
`build_parser()` and resolved through the launch path's own
`resolve_critic_mode` → `desugar_umbrella_flags` → `resolve_config`.

**Result: all 105 argv-settable fields of this mirror equal what that argv resolves to.** The other
23 keys come from the observation LAYOUT or are derived, so no argv can move them: the six embedding
dims, the five `max_*` capacities, `net_arch` / `move_net_hidden` / `role_encoder_hidden` /
`role_token_size` / `projection_dim`, `active_context_dim`, `total_dim`, `arch_signature`,
`config_version`, the config-only `attend_unrevealed_opponents`, and the three `_DERIVED` toggles
(`opp_belief_slots`, `opp_intent`, `opp_intent_grad_mode`, computed from their coefficients).

**Re-verify against the run itself** once `ai_v12_02` writes a `model_config.json` — that is the
authoritative check, and a difference there is a defect in this file, not in the run.

### The argv finding that verification produced

**gen-17's recorded command no longer launches on HEAD**, for a reason unrelated to the critic:
`--distill-team-bias 0.4` was refused with no `--distill-teacher` (2026-09-06) and is now a DELETED flag
(deletion pass L3, `designs/deleted_flags.md`), so argparse rejects it outright. It is not a
`model_config.json` key, so it cannot move this mirror — but a relaunch argv built by copying gen-17's
command will be rejected until it is dropped (or the run is launched pinned to its own commit). `python -m main.checkargs` reports it offline.
