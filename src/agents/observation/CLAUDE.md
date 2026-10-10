# CLAUDE.md — the Observation LAYOUT (`src/agents/observation/`)

This directory is the observation **as the MODEL reads it**: the row's layout, the vocabularies the model's tables are
sized from, the read-back decoders the prober uses, and a few pure helpers the model and prober share. It does NOT
write the row. What the observation IS (the block table, the per-mon slot, the event-record schema) is
[`designs/ARCHITECTURE.md`](../../../designs/ARCHITECTURE.md) §1; what each field MEANS and where it comes from is
[`designs/observation/`](../../../designs/observation/README.md) (detail lifted out of this leaf on 2026-10-10; the
pre-cleanup leaf is frozen at `designs/research_state/claude_md_archive/src_agents_observation_CLAUDE_2026-10-10.md`).

🚨 **The RUST encoder is the ONLY encoder** (`src/rust_sim/src/encoder/`, served by the env core to training, eval,
live play and the prober; [`designs/rust_sim/encoder.md`](../../../designs/rust_sim/encoder.md)). The Python encoder's
encode path (`Gen3ObservationEncoder.encode` / `get_observation`, every sub-encoder's battle-reading `encode`, the
incremental assembler, the Wish fold, the obs-facts writer, the incoming-damage obs encoder) and the Python trackers
are DELETED (T27 P6 slice 6d-2, 2026-10-08; `designs/ops/deletion_pass_manifest.md` §8.5). The obs golden passed
byte-identically across the deletion. The classes that remain here are block DESCRIPTORS: width, layout, read-back.

🚨 **NEVER hardcode an obs index.** Read `Gen3ObservationEncoder(load_mappings()).get_layout()` and `.dimension`
(`state_encoder.py`); every offset is a named constant in `constants.py`. Event-window columns are
`constants.EventCol` members, never bare integers (see below).

> 🚨 **This package imports NO poke-env.** The trainer imports it (`damage_tables` → `types`), so an import here fails
> `src/poke_env_free_entry_points_test.py` and `src/poke_env_import_gate_test.py`.

---

## Two things can regress, with different gates

1. **Observation *values*** — a change to what a cell contains is a change to the Rust encoder, and it is
   **retrain-class**: bump `ARCH_SIGNATURE` (`src/agents/model/model_version/`). The value gates are the obs GOLDEN
   (`python -m agents.training.golden_obs_core --check`, `golden_obs_core_test.py`) and the four checks of
   `designs/rust_sim/encoder.md` §6a.
2. **The LAYOUT** — 🚨 `src/rust_sim/src/encoder/layout.rs` is **Rust-OWNED source** (since slice 6d-2): edit it
   directly, and change `constants.py` in the SAME edit. `rust_core_obs_layout_test.py` PARSES `layout.rs` and holds
   every value both sides carry equal — every shared `usize` constant (the block offsets and `OBS_DIM` among them),
   `EventCol`, `EVENT_STATUS_IDS`, `TOXIC_STAGE_MAX`, the obs-facts tables and the model-read vocabularies — so a
   one-sided change FAILS.

A layout move also breaks the prober's pinned offsets: `src/main/prober/engine_test.py::test_offsets_resolve_matches_layout`
(the prober resolves offsets at runtime from `get_layout()` via `ObsOffsets`, `0 = absent` for a deleted block) —
update the pins there.

---

## MANDATORY on every observation change: the ENCODER benchmark + the golden

Any change here, in `constants.py`, or in the Rust encoder:

```bash
python -m agents.observation.rust_encoder_benchmark --decisions 40 --reps 300 | tee /tmp/enc_before.txt
# … the change …
python -m agents.observation.rust_encoder_benchmark --decisions 40 --reps 300 | tee /tmp/enc_after.txt
python -m agents.training.golden_obs_core --check
```

It times the release `core_events --obs-bench` encoder at a seeded sample of the obs golden's banked decisions. Compare
a SAME-SESSION before / after on the same load; it warns on a busy box and never rescales. Detail:
`designs/rust_sim/encoder.md` §8. (The Python benchmarks `obs_build_benchmark.py` & co. are deleted with the path they
profiled — `designs/deleted_flags.md`; dated figures citing them are measurements, not contracts.)

---

## The map

| File | What it is |
|---|---|
| `constants.py` | every offset / `*_DIM` / `OFFSET_*`; `EventCol` (the event-window column contract), `EVENT_OUTCOME_GROUP` / `EVENT_EFF_GROUP`, `MAX_TURNS`, `TOXIC_STAGE_MAX`, the `FACTS_*` layout. Twin of `layout.rs` |
| `state_encoder.py` | `load_mappings()`, `Gen3ObservationEncoder` — `get_layout()`, `.dimension`, `describe_vector` over the whole row |
| `base.py` | `ObservationEncoder`, the block-descriptor base (`dimension`, layout, `describe_vector`) |
| `species.py` `items.py` `abilities.py` `types.py` `moves.py` `pokemon.py` `active_context.py` `global_env.py` `reactive.py` | per-block descriptors; the model-read vocabularies live here (`types.TypeEncoder.TYPE_TO_IDX`, `pokemon._STATUS_STR_IDX`, `moves.HIDDEN_POWER_MOVE_NUM`) |
| `gen3_effects.py` | the volatile + cant vocabularies (`VOLATILE_SLOTS`, `GEN3_VOLATILE_TO_SLOT`, `NOT_A_VOLATILE`, `CANT_REASONS` / `CANT_REASONS_LIVE`); `encode_volatiles` RAISES on an unclassified id |
| `gen3_effect_sources.py` | derives every effect line the gen3 sim can announce from the Showdown source (off the hot path) |
| `obs_facts.py` | the OBS-FACTS block's layout + `describe` (the Rust twin `encoder/facts.rs` writes it) |
| `turn_delta_encoder.py` | `TurnDeltaEncoder` + `TURN_DELTA_EMBEDDED_IDS` — the ARCHIVED lag-frame slot (deleted from the live row); survives as the prober's decoder for archived runs and the feature-coverage probes' encoder |
| `sleep_belief.py` | the sleep-wake belief layout + `expected_free_turns` |
| `incoming_damage.py` | pure damage / KO / outspeed math the prober engine imports (its obs-encoder half is deleted) |
| `belief_labels.py` | the reference definition of the belief-aux labels the Rust env core writes |
| `schema.py` | the declarative SCHEMA view over `get_layout()`: names every contiguous block and proves the blocks tile the row |
| `rust_encoder_benchmark.py` | the mandatory benchmark above |
| `rust_core_obs_layout_test.py` | the layout gate above |
| `toxic_stage_core_test.py` | the toxic stage end to end through the real core |

---

## Rules and hazards

- 🚨 **`EventCol` is the ONE column contract for the event window** — the model's consumers
  (`team_transformer.EventSeats`, `feature_coverage/_support.py::obs_with_event_row`) import it; never write a bare
  column index (a producer/consumer pair bound by POSITION is the class the 2026-08-18 sweep convicted five times). The
  two one-hot groups are written by INDEXING, so their ORDER is load-bearing — reordering a member relabels every
  historical row with no shape change. `src/agents/training/event_window_test.py` pins it.
- 🚨 **The volatile vocabulary is SOURCE-DERIVED and crash-don't-drop.** A new effect id gets classified in
  `gen3_effects.py` (a slot, or a `NOT_A_VOLATILE` row naming where its consequence arrives) — never a catch-all; `unknown`
  still raises. Gates: `gen3_effects_test.py::test_every_effect_the_gen3_sim_announces_is_classified` (+ siblings),
  `src/main/live/effect_scan_test.py`, and `src/main/ladder_drift_scan.py` against Showdown master. Detail and the
  closed findings: [`designs/observation/volatile_vocabulary.md`](../../../designs/observation/volatile_vocabulary.md).
- 🚨 **Two move orders are live at once.** The per-mon move block is sorted by `Move.id` (it feeds the order-sensitive
  role token); `active_req_moves` is REQUEST order (slot *k* ↔ action logit 6+*k*). The model crosses them ONLY by
  move-num IDENTITY (`extractor_ctx.active_request_sorted_match`); a positional crossing is silent whenever every move
  is legal. `agents/action/ordering_integrity.check_obs_move_order` is the throwing guard.
- 🚨 **The ACTIVE flag is the LAST dim of the per-mon slot** (`POKEMON_ACTIVE_OFFSET`) — the model's
  `hp_and_active[:, :, -1]` convention is load-bearing.
- **`turns_since_progress` (board block) is NOT the deadline.** It is a resettable stall counter; how much game is LEFT
  is the global block's `clock` group (`MAX_TURNS` = the forfeit turn, pinned by
  `global_env_test.py::test_max_turns_is_the_forfeit_deadline`).
- **Embedded ids never reach a Linear raw** — move / species / type ids in the row are embedding ids, routed by the
  model (`slice_pokemon_categoricals`); a new embedded column is declared, never left as a scalar.

Per-block semantics, sources and their gates (recency, protect odds, last action, pair history, event window,
toxic stage, sleep-wake belief, move slot, spread, OBS-FACTS, the board block, Wish, the deleted move-effect /
incoming-damage blocks, the archived TurnDelta slot):
[`designs/observation/per_block_reference.md`](../../../designs/observation/per_block_reference.md).

---

## Static typing (mypy)

This package is type-checked at **ZERO errors**, sharing `mypy.ini` (`files = src/agents/model, src/agents/observation`)
with the model package; the gate is `src/agents/model/mypy_gate_test.py`. **A loosening to clear something here
silently de-tiers the model package — narrow with a targeted `# type: ignore[code]` + reason instead.** mypy types the
ARRAY, not the INDEX: it cannot catch a wrong offset. Detail and the idioms:
[`designs/observation/typing.md`](../../../designs/observation/typing.md).
