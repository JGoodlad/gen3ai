# The Rust env's TRAINING LABELS — inventory and derivations (M5 Lane C)

**Always current.** Owner of the plan: `designs/endstate/program_rust_core.md` §2 M5, lane C
("training labels"). Resume point and findings: `designs/research_state/measurements/m5_laneC/PROGRESS.md`.

The Rust env core (`src/rust_env/`, package `pokesim_env`) writes the trainee's `obs` row and
`mask` (Lane 0). `Gen3Env` also puts TRAINING-ONLY LABEL keys into the trainee's Dict obs. They are
read by a loss, a callback or a diagnostic, never by the policy forward (`designs/ARCHITECTURE.md` §7).
For Python to leave the per-decision loop, every key the production surface emits must come from
the core, from the host, or be refused. This doc is the human-readable view of that table.

## 1. The inventory

**The table of record is `src/utils/rust_env/label_inventory.py` (`LABELS`).** The routine test
`src/agents/training/rust_env_label_inventory_test.py` constructs `Gen3Env` (no battle, no server)
and FAILS when:

- `Gen3Env` can declare a key the table does not list (or the reverse);
- a dtype or shape drifts;
- a row's emit gate does not emit it;
- the PRODUCTION surface emits a different set than the rows marked production. The production
  surface is `--arch production` + `designs/production_config.json`, resolved by
  `main.rust_core_cutover.envs.production_args` → `env_factory.trainee_env_kwargs`, exactly as a
  production launch resolves it;
- a named consumer stops naming its key;
- ARCHITECTURE.md §7's ✅/❌, or this doc's table, disagrees with the table.

**Headline (2026-09-29): `Gen3Env` can emit 30 label keys. Production emits 21.** They fall into
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
per-episode host value.** The remaining 9 keys are off the production surface.

The `rust` column says where the Rust env gets each key:

- `core`: a label column the env core writes per decision. It is gated against `Gen3Env` by
  `rust_env_labels_parity_test.py`.
- `host_const`: a placeholder the host writes as a constant. A rollout callback overwrites it after
  collection.
- `host_episode`: a per-episode value the host owns, keyed by the core's `episode` column.
- `refused`: off the production surface. A spec that asks for it is refused at startup.

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
| `win_target` | f32 `[1]` | winprob | yes | `Gen3Env._merge_training_keys` (0.0) | `win_prob_callback.py`, `instrumented_ppo/ppo.py` | `host_const` |
| `win_mask` | f32 `[1]` | winprob | yes | `Gen3Env._merge_training_keys` (0.0) | `win_prob_callback.py`, `instrumented_ppo/ppo.py` | `host_const` |
| `win_margin` | f32 `[1]` | margin | yes | `Gen3RewardManager.process_turn_reward` → `material_margin` | `instrumented_ppo/ppo.py`, `value_sidecar.py` | `core` |
| `opp_class` | i64 `[1]` | opp_class | yes | `Gen3Env._merge_training_keys` (the wrapper's `_opponent_class`) | `instrumented_ppo/ppo.py`, `value_sidecar.py` | `host_episode` |
| `opp_action_kind` | i64 `[1]` | intent | yes | `Gen3Env._opp_intent_labels` | `instrumented_ppo/ppo.py`, `train_setup.py` | `core` |
| `opp_action_num` | i64 `[1]` | intent | yes | `Gen3Env._opp_intent_labels` | `instrumented_ppo/ppo.py`, `train_setup.py` | `core` |
| `opp_switch_slot` | i64 `[1]` | intent | yes | `Gen3Env._opp_intent_labels` | `instrumented_ppo/ppo.py`, `train_setup.py` | `core` |
| `opp_switch_species` | i64 `[1]` | intent | yes | `Gen3Env._opp_intent_labels` | `instrumented_ppo/ppo.py`, `train_setup.py` | `core` |
| `win_row_w` | f32 `[1]` | winprob_weight | no | `Gen3Env._merge_training_keys` (1.0) | `win_prob_rollout.py`, `instrumented_ppo/ppo.py` | `host_const` |
| `fork_pg_m` | f32 `[1]` | fork | no | `Gen3Env._merge_training_keys` (1.0) | `fork_arm.py` | `host_const` |
| `opp_true_team` | f32 `[6,122]` | true_team | no | `Gen3Env._true_team_block` | `model/extractor_forward.py` | `refused` |
| `aux_target` | f32 `[25]` | dense_aux | no | `Gen3Env._merge_training_keys` (0.0) | `dense_aux.py`, `instrumented_ppo/ppo.py` | `refused` |
| `aux_mask` | f32 `[25]` | dense_aux | no | `dense_aux.state_visibility` | `dense_aux.py`, `instrumented_ppo/ppo.py` | `refused` |
| `aux_turn` | f32 `[1]` | dense_aux | no | `Gen3Env._merge_training_keys` | `dense_aux.py` | `refused` |
| `defensive_opportunity` | f32 `[1]` | defensive | no | `Gen3Env._defensive_opportunity` | `instrumented_ppo/ppo.py` | `refused` |
| `bait_opportunity` | f32 `[1]` | bait | no | `Gen3Env._bait_opportunity` | `instrumented_ppo/ppo.py` | `refused` |
| `distill_mask` | f32 `[1]` | distill | no | `Gen3Env._distill_mask` | `instrumented_ppo/ppo.py`, `distill_anchor.py` | `refused` |

`fork_buffer.py` and SB3's rollout buffer CARRY every key, but neither is a consumer.

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
| belief | `belief_species`, `belief_moves`, `known_moves` | `assign_hidden_to_slots`: hidden = truth team minus the revealed species (skipping a species with no num), sorted by species num; the j-th fills the j-th believed slot (`species_known < 0.5`). Moves are num-mapped in set order, max 4, unknown ids skipped. `known_moves` gives each revealed slot its species' full truth moveset. Nums come from `mappings.json`'s `species` / `moves` `num`, the same tables the encoder's embeddings index |
| spread | the 6 spread keys | the truth `stats` in (atk, def, spa, spd, spe) order. Nature / EVs come from `damage_tables.invert_nature_evs(derived, base, species_id)` — a PORT of the inversion, including its tie-breaking and its "no valid spread ⇒ mask 0". It is cached per episode, because the truth team is fixed |
| hp_type | `hp_type_label`, `hp_type_mask` | the first `hiddenpower<type>` move of the truth mon → the index in `belief_labels.HP_TYPE_NAMES` |
| item | `item_label`, `item_mask` | the truth mon's CURRENT item (`None` / "" ⇒ 0) → the item num (`gen3_data.items`); an unknown id is absent (mask 0) |
| margin | `win_margin` | `material_margin(live)` on the side's `present()` view. **Timing:** Python computes it in `calc_reward` on the same board the obs describes, and 0.0 at reset |
| intent | the 4 intent keys | the port's `trackers::IntentLabel` (slice T already gates it against the Python label), num-mapped. Hidden Power resolves to the attacker's truth typed num (`_intent_move_num_resolver`). `opp_switch_slot` uses the PREVIOUS decision's revealed-slot map (`_opp_slot_map_prev`); `SWITCH_SLOT_NONE` and the zero label come from `opp_intent_labels.py` |

## 3. Hazards the parity gate must decide (no claim made here)

- **The truth side's timing.** Python reads `battle2` when the trainee's step returns, whatever
  poke-env has processed for agent2 by then. The core reads the truth chain after the same write.
  They should agree, but it is **UNVERIFIED** until the gate runs. `item` is the only dynamic truth
  field today.
- **`return` / `frustration` / typed-HP ids.** Showdown's request appends the base power to
  `return` / `frustration` (`return102`), and poke-env's `Move.retrieve_id` folds it. The label maps
  a mon's moves through `to_id_str(mv.id)`. What id the Python label actually looks up for these is
  measured by the gate, not assumed.
- **Episode edges.** At a decision where `battle2` does not exist yet, Python emits the all-PAD /
  all-zero form. Whether any trainee decision is ever labelled that way is measured.

## 4. Build order (Lane C units)

1. The inventory and its routine test.
2. The GENERATED label columns (`columns.py` rows owned by lane C). The spec also gains a
   `labels` declaration: the families the caller wants. A family marked `refused` is refused at
   startup.
3. One unit per `core` family, each with its slice-N parity gate against `Gen3Env` on recorded
   battles: COMMIT tier in the routine gate, MILESTONE tier `slow`. The order is belief, hp_type +
   item, spread (the inversion port), intent, then margin.
