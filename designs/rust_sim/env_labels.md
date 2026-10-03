# The Rust env's TRAINING LABELS — inventory and derivations (M5 Lane C)

**Always current.** Owner of the plan: `designs/endstate/program_rust_core.md` §2 M5, lane C
("training labels"). Resume point and findings: `designs/research_state/measurements/m5_laneC/PROGRESS.md`.

> **Deletion pass U3:** the Python env this lane was built against (`Gen3Env`, its `_belief_labels` / `_spread_labels` / `_hp_type_labels` / `_item_labels` / `_opp_intent_labels` / `_merge_training_keys` methods, `env_factory`) and the label parity gates that compared the core to it (`rust_env_labels_parity_test.py`, the `slice_n` harness, `nature_ev_label_test.py`) are DELETED. The Rust label columns are the only implementation and are pinned by `src/rust_env/tests` and `src/utils/rust_env/label_columns_test.py`. Where this doc says "Python", "`Gen3Env`" or "parity gate", it records how the columns were derived and proven when they were built; the `producer` column names the (deleted) method each key's DEFINITION came from.

The Rust env core (`src/rust_env/`, package `pokesim_env`) writes the trainee's `obs` row and
`mask` (Lane 0), and the trainee's Dict obs also carries TRAINING-ONLY LABEL keys. They are
read by a loss, a callback or a diagnostic, never by the policy forward (`designs/ARCHITECTURE.md` §7).
Every key the production surface emits comes from the core or from the host. This doc is the human-readable view of that table.

## 1. The inventory

**The table of record is `src/utils/rust_env/label_inventory.py` (`LABELS`).** The routine test
`src/agents/training/rust_env_label_inventory_test.py` reads the trainee's declared spaces (`agents.training.trainee_spaces`; no battle, no server)
and FAILS when:

- `trainee_spaces` can declare a key the table does not list (or the reverse);
- a dtype or shape drifts;
- a row's emit gate does not emit it;
- the PRODUCTION surface emits a different set than the rows marked production. The production
  surface is `--arch production` + `designs/production_config.json`, resolved by
  `main.train.production_args.production_args` → `trainee_spaces.trainee_env_kwargs`, exactly as a
  production launch resolves it;
- a named consumer stops naming its key;
- ARCHITECTURE.md §7's ✅/❌, or this doc's table, disagrees with the table.

**Headline (deletion pass L3): the trainee's spaces can declare 22 label keys. Production emits 21.** (Seven keys — `win_row_w`, `opp_true_team`, `aux_target` / `aux_mask` / `aux_turn`, `defensive_opportunity`, `bait_opportunity` — went with their deleted levers in L2, and `distill_mask`, with distillation, in L3; its `refused` label kind went with it, so every row is now `core`, `host_const` or `host_episode`.) The production keys fall into
eight families:

- belief: 3 keys
- spread / nature / EV: 6
- Hidden Power type: 2
- item: 2
- win-prob placeholders: 2
- material margin: 1
- opponent class: 1
- opponent intent (α/β): 4

**18 of the 21 are per-decision values the core must compute. 2 are host constants. 1 is a
per-episode host value.** The remaining 1 key (`fork_pg_m`) is off the production surface.

The `rust` column says where the Rust env gets each key:

- `core`: a label column the env core writes per decision (it was gated against `Gen3Env` by
  `rust_env_labels_parity_test.py` until both were deleted in U3).
- `host_const`: a placeholder the host writes as a constant. A rollout callback overwrites it after
  collection.
- `host_episode`: a per-episode value the host owns, keyed by the core's `episode` column. `opp_class` is the class of the episode's opponent ROUTE: `rust_env_opponents.RustEnvOpponents.opp_class(cols)` reads it off the core's `opp_route` column (M5 Lane E), so it can never describe the previous opponent.

| key | dtype `shape` | family | prod | producer (Python) | consumer | rust |
|---|---|---|---|---|---|---|
| `belief_species` | i64 `[6]` | belief | yes | `Gen3Env._belief_labels` | `belief_bank.py` | `core` |
| `belief_moves` | i64 `[6,4]` | belief | yes | `Gen3Env._belief_labels` | `belief_bank.py` | `core` |
| `known_moves` | i64 `[6,4]` | belief | yes | `Gen3Env._belief_labels` | `belief_bank.py` | `core` |
| `belief_spread` | f32 `[6,5]` | spread | yes | `Gen3Env._spread_labels` | `belief_bank.py` | `core` |
| `belief_spread_mask` | f32 `[6]` | spread | yes | `Gen3Env._spread_labels` | `belief_bank.py` | `core` |
| `belief_nature` | i64 `[6]` | spread | yes | `Gen3Env._spread_labels` | `belief_bank.py` | `core` |
| `belief_nature_mask` | f32 `[6]` | spread | yes | `Gen3Env._spread_labels` | `belief_bank.py` | `core` |
| `belief_ev` | f32 `[6,5]` | spread | yes | `Gen3Env._spread_labels` | `belief_bank.py` | `core` |
| `belief_ev_mask` | f32 `[6]` | spread | yes | `Gen3Env._spread_labels` | `belief_bank.py` | `core` |
| `hp_type_label` | i64 `[6]` | hp_type | yes | `Gen3Env._hp_type_labels` | `belief_bank.py` | `core` |
| `hp_type_mask` | f32 `[6]` | hp_type | yes | `Gen3Env._hp_type_labels` | `belief_bank.py` | `core` |
| `item_label` | i64 `[6]` | item | yes | `Gen3Env._item_labels` | `belief_bank.py` | `core` |
| `item_mask` | f32 `[6]` | item | yes | `Gen3Env._item_labels` | `belief_bank.py` | `core` |
| `win_target` | f32 `[1]` | winprob | yes | `Gen3Env._merge_training_keys` (0.0) | `rust_rollout/store.py`, `instrumented_ppo/ppo.py` | `host_const` |
| `win_mask` | f32 `[1]` | winprob | yes | `Gen3Env._merge_training_keys` (0.0) | `rust_rollout/store.py`, `instrumented_ppo/ppo.py` | `host_const` |
| `win_margin` | f32 `[1]` | margin | yes | `Gen3RewardManager.process_turn_reward` → `material_margin` | `instrumented_ppo/ppo.py`, `value_sidecar.py` | `core` |
| `opp_class` | i64 `[1]` | opp_class | yes | `Gen3Env._merge_training_keys` (the wrapper's `_opponent_class`) | `instrumented_ppo/ppo.py`, `value_sidecar.py` | `host_episode` |
| `opp_action_kind` | i64 `[1]` | intent | yes | `Gen3Env._opp_intent_labels` | `instrumented_ppo/ppo.py`, `train_setup.py` | `core` |
| `opp_action_num` | i64 `[1]` | intent | yes | `Gen3Env._opp_intent_labels` | `instrumented_ppo/ppo.py`, `train_setup.py` | `core` |
| `opp_switch_slot` | i64 `[1]` | intent | yes | `Gen3Env._opp_intent_labels` | `instrumented_ppo/ppo.py`, `train_setup.py` | `core` |
| `opp_switch_species` | i64 `[1]` | intent | yes | `Gen3Env._opp_intent_labels` | `instrumented_ppo/ppo.py`, `train_setup.py` | `core` |
| `fork_pg_m` | f32 `[1]` | fork | no | `Gen3Env._merge_training_keys` (1.0) | `fork_arm.py` | `host_const` |

`fork_buffer.py`'s FILL table and SB3's rollout buffer CARRY every key, but neither is a consumer.

## 2. Where each `core` family comes from in the Rust env

Every `core` label reads two things the env already holds for the trainee side `s`.

**The TRUTH is the OTHER side's own reading of its team.** `BattleVersion::stream(1-s).board_reading.team`
mirrors poke-env's `battle2.team`:

- species;
- moves, as poke-env keys an OWN mon's moves. The request spells a typed Hidden Power without its
  power in gen 3 (`sim/pokemon.ts` `getSwitchRequestData`, `hiddenpower<type>`). It spells
  `return<bp>` / `frustration<bp>` too — see §3;
- the CURRENT `item`;
- `stats`, the request's `baseStoredStats`.

The Python env reads `battle2` for every privileged label. Reading the other side's parse chain is
therefore the like-for-like source; the engine board is not. The truth side is folded in the same
op, before the trainee's decision opens (`pool.rs` `advance` folds both sides per write).

**The READING is the side's own row and its `opp` list.** `species_known` is read from the row the
core just encoded, as the Python env reads it from `obs["observation"]`. The revealed mons come from
`board_reading.opp` in encoder slot order: the encoder's `get_team_list(is_opponent=True)` order is
reveal order, and the Rust encoder indexes `reading.opp` directly.

| family | keys | the Rust derivation |
|---|---|---|
| belief | `belief_species`, `belief_moves`, `known_moves` | `assign_hidden_to_slots`: hidden = truth team minus the revealed species, MATCHED BY DEX NUM (a forme shares its base species' num), sorted by species num; the j-th fills the j-th believed slot (`species_known < 0.5`). Moves are num-mapped in set order, max 4. `known_moves` gives each revealed slot its species' full truth moveset. Nums come from `mappings.json`'s `species` / `moves` `num`, the same tables the encoder's embeddings index |
| spread | the 6 spread keys | the truth `stats` in (atk, def, spa, spd, spe) order. Nature / EVs are the truth mon's DECLARED spread (`nature`, `evs`, `ivs`, which the reading backfills from the side's packed team exactly as poke-env's `backfill_spread_from_teambuilder` does): the nature's num and the EVs at `4·⌊ev/4⌋` — `belief_tables.true_nature_ev_label`, rule for rule (`gen3_true_spread_labels_v1`). A THROWING guard: the declared spread at L100 with its true IVs must reproduce the five stats, or the label write is an `Err` (a FAULT; Python raises `SpreadLabelError`). Read per decision, never cached |
| hp_type | `hp_type_label`, `hp_type_mask` | the first `hiddenpower<type>` move of the truth mon → the index in `belief_labels.HP_TYPE_NAMES` |
| item | `item_label`, `item_mask` | the truth mon's CURRENT item (`None` / "" ⇒ 0) → the item num (`gen3_data.items`); an unknown id is an `Err` |
| margin | `win_margin` | `material_margin(live)` on the side's `present()` view. **Timing:** Python computes it in `calc_reward` on the same board the obs describes, and 0.0 at reset |
| intent | the 4 intent keys | the port's `trackers::IntentLabel` (slice T already gates it against the Python label), num-mapped. Hidden Power resolves to the attacker's truth typed num (`_intent_move_num_resolver`). `opp_switch_slot` uses the PREVIOUS decision's revealed-slot map (`_opp_slot_map_prev`); `SWITCH_SLOT_NONE` and the zero label come from `opp_intent_labels.py`. A move or switch-in with no num, or a Hidden Power attacker off the truth team, is an `Err` |

**🚨 No lookup ever SKIPS (F-X5-3, `gen3_label_lookup_guard_v1`, 2026-10-03).** Every `core` family's
writer returns an `Err` — a FAULT, which fails the batch and poisons the pool — when a lookup cannot
be made: a species, move or item with no num; a revealed species that is not on the truth team
(matched by DEX NUM in every family, so a forme the two readings spell differently still matches);
a truth mon with an unknown stat; more hidden truth mons than believed slots. Until then the
writers SKIPPED each of these in silence (a shorter hidden list, a mask 0, a silent UNKNOWN / 237 /
species 0), which would undercount a label with nothing to show for it. The only absences left are
legitimate ones: a revealed mon that runs no Hidden Power, and believed slots beyond a SHORT team.
**Measured dormant before the guard:** 0 skips over 4,001 pool + bridge-corpus episodes (572,451
decisions) and 30,000 ladder-corpus episodes (5,253,432 decisions), random policy, every family
declared; the guarded writers then ran 4,001 + 30,000 fresh-seed episodes with 0 FAULTs. No
production label row was ever dropped. Pinned by `src/rust_env/tests/label_lookup_guard_test.rs`
(seven of its eight cases fail on a revert of the guard). The Python reference builders
(`agents/observation/belief_labels.py`, `agents/training/opp_intent_labels.py`) raise
`BeliefLabelError` / `IntentLabelError` the same way.

## 3. Hazards the (since-deleted) parity gate had to decide

- **The truth side's timing (DECIDED by the item gate).** Python reads `battle2` when the trainee's
  step returns; the core reads the truth chain after the same write. `item` is the only dynamic
  truth field today. It changed 65 times over the milestone's 400 episodes, and every decision is
  equal on both paths.
- **`return` / `frustration` / typed-HP ids (CLOSED by the belief gate).** Showdown's request
  spells `return102`. Both readings key an own mon's moves by `Move.retrieve_id`, which folds it
  to `return`, while a Hidden Power keeps its type (`Move._id`). The belief milestone runs 400
  episodes with 0 divergences.
- **Episode edges.** At a decision where `battle2` does not exist yet, Python emits the all-PAD /
  all-zero form. Whether any trainee decision is ever labelled that way is measured.

## 4. Build order (Lane C units)

1. The inventory and its routine test.
2. **(BUILT)** The GENERATED label columns: `columns.py` builds one column per `core` row,
   18 in all, named by the key. The spec gains a `labels` declaration: the families the caller
   wants. `src/rust_env/src/labels/mod.rs` refuses, at STARTUP and by name, a `host_*` family, a
   an unknown family, and a `core` family not built yet (`labels::BUILT`).
3. One unit per `core` family, each with its slice-N parity gate against `Gen3Env` on recorded
   battles (the gates were deleted with `Gen3Env` in U3; the milestone numbers below are their last read): COMMIT tier in the routine gate, MILESTONE tier `slow`. The order is belief, hp_type +
   item, spread, intent, then margin. **`belief` BUILT**
   (`src/rust_env/src/labels/belief.rs`, gated by `src/agents/training/rust_env_labels_parity_test.py`:
   400 milestone episodes, 0 divergences). **`hp_type` and `item` BUILT**
   (`src/rust_env/src/labels/per_slot.rs`, the same gate: 0 divergences). **`spread` BUILT**
   (`src/rust_env/src/labels/spread.rs`, the nature table loaded at startup from the stamp's data
   dir; the same gate: 0 divergences — first as a port of the IV-31 stat inversion, since
   2026-09-29 as the declared-spread read, `gen3_true_spread_labels_v1`). **`intent` BUILT**
   (`src/rust_env/src/labels/intent.rs`, over `trackers::IntentLabel`; the same gate: 0 divergences).
   **`margin` BUILT** (`src/rust_env/src/labels/margin.rs`; 0.0 at the RESET decision). **Every
   `core` family is BUILT:** the milestone covers 500 episodes (pool, ladder, procedural),
   714,978 key compares, 0 divergences.
4. **Label COVERAGE is pinned by the same gate** (`gen3_true_spread_labels_v1`, 2026-09-29): every
   revealed slot with a `belief_spread` label carries its nature / EV label too. Milestone:
   83,905 / 83,905 pool, 91,316 / 91,316 ladder, 42,539 / 42,539 procedural slot-decisions (the
   IV-31 inversion it replaced left 54.6 % of the pool's short and 1.2 % of the ladder's). Over all
   719 pool teams, `nature_ev_label_test.py` (deleted in U3) pinned 4,314 / 4,314 mons labelled
   with their declared set.
