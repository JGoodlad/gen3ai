# CLAUDE.md — Battle read-models (`src/agents/battle/`)

**The Python event-sourced battle layer is DELETED** (T27 P6 slice 6d-2, 2026-10-08): `Gen3Battle`, the
`BattleEvent` log and its completeness registry (`battle_event.py`), `TurnView`, `StrictBattleView`, the offline
feed, and with them the strict-API lock (`src/agents/strict_api_lock_test.py`). Nothing trains, evaluates, plays or
probes on them any more — every reading, row, view, legality surface, tracker and reward is the RUST core's
(`src/rust_sim/`: `present/` the view, `core_events/` the event record, `trackers/` the folds, `encoder/` the row).
The deleted modules are listed in `designs/ops/deletion_pass_manifest.md` §8.5; their design record is
`designs/ai_v4/impl_step8_strict_battle_api_and_turndelta_fold.md` (history).

What this package holds now:

- **`live_view.py` — the read-models as frozen DATA CLASSES.** `LiveView` / `LiveSide` / `LivePokemon` /
  `LiveMove` / `LiveWeather` (the current board — "what is true now", primitives only, NO past-turn state) and
  `LegalActions` / `LegalMove` / `LegalSwitch` (the server-authoritative legality surface: wire-truth
  `move_slots`, the `switches` / `force_switch` / `trapped` / `maybe_trapped` / `wait` / `struggle` flags, a
  read-only `last_request` mirror, `own_hp_typed_id` and the `display_move_ids` label accessor). They compute
  NOTHING: every field is copied in by `core_view`. Struggle stays single-sourced (a lone `struggle` request entry
  is the `struggle` flag, never a move slot — the core's rule). 🚨 **What a PROTOCOL line revealed is current
  knowledge and is on the view** (`gen3_obs_facts_v1`): `LiveMove.seen`, `LivePokemon.item_public` /
  `ability_public`, `LiveView.residual_done` — the core's `present()` carries them (rules V18 / V19, pinned by
  `src/rust_sim/src/present/tests.rs` and `tests/obs_facts_truth_test.rs`). `live_view_test.py` pins the shape:
  `LivePokemon`'s field set, and that no past-turn field (`last_move`, …) exists on it.
- **`core_view.py` — a `LiveView` / `LegalActions` from the core's `present()` / `legal_actions()` JSON**
  (`gen3_core_present_v1`). A pure TRANSPORT: every presentation rule is applied in Rust (`src/rust_sim/src/present/`),
  so nothing here derives a field. Its consumer is the prober's core walk (`main/prober/core_walk.py`).
  🚨 **The poke-env data the core's reading consults is FROZEN, Rust-owned source** (P1):
  `src/rust_sim/src/present/tables.rs` is edited directly. Contract:
  [`designs/rust_sim/present.md`](../../../designs/rust_sim/present.md).
- **`core_obs.py` — the core's observation FRAMES** (`wrap_row`, `check_row`, `frame_for_decision`, `decode_frame`):
  a `<f4` row the core emitted, checked (shape, fully written, no NaN) and wrapped as the observation.
- **`core_replay.py` + `core_corpus_test.py` — the core's recorded corpus replayed through `core_events`.**
  `core_replay` (`RecordedBattle`, `run_core`, `core_chunks`, `chunks_sha`) imports no poke-env;
  `core_corpus_test.py` (`sim`, routine gate, ~4 s) replays the 12 recorded battles of
  `rust_core_parity_fixtures/commit_tier.json.gz` and holds what reads ONLY the core: every battle replays `ok`
  (`parse == step`, the parse-chain encode gate, the engine BOARD audit at every decision), every row is fully written
  and every decision's tokens are its legal actions, the information boundary holds on the native record, and the
  golden RECORDS (`rust_core_parity_fixtures/records`) read, re-write byte-identically and re-parse. Contract:
  [`designs/rust_sim/core_events.md`](../../../designs/rust_sim/core_events.md).
  `core_present_golden_test.py` freezes the core's whole `present()` answer on the 44 constructed scenarios the deleted
  present-parity tests used (recorded where core and poke-env still agreed).
- **`faint_causes.py` — the faint-cause vocabulary.** `FAINT_CAUSE_VOCAB` / `FAINT_CAUSE_DIM` (8, the ARCHIVE
  vocabulary that sizes the frozen `TurnDelta` lag frame the prober decodes) and `FAINT_CAUSE_VOCAB_LIVE` /
  `FAINT_CAUSE_DIM_LIVE` (10, the event window's, which `team_transformer.EventSeats` sizes its table from).
  `src/agents/observation/rust_core_obs_layout_test.py` holds the live vocabulary equal to `layout.rs`'s.

🚨 **The core's event SCHEMA is Rust-owned** (`src/rust_sim/src/core_events/schema.rs`, P1): there is no Python
`battle_event.py` to mirror any more — edit `schema.rs` (and `reading.rs`) directly; the cargo tests
(`tests/core_events_test.rs`, `tracker_semantics_test.rs`, `window_record_test.rs`) are its gates.

**Verification:** `live_view_test.py` (shape), `core_obs_test.py`, `core_corpus_test.py`, `core_present_golden_test.py`.
