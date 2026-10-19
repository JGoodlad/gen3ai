# CLAUDE.md — the Observation layout (`src/agents/observation/`)

This package is the observation as the model reads it: the row's layout, the vocabularies the model's tables
are sized from, read-back decoders for the prober, and pure helpers. It does not write the row: the Rust
encoder (`src/rust_sim/src/encoder/`, `designs/rust_sim/encoder.md`) is the only encoder. What the row is:
`designs/ARCHITECTURE.md` §1. What each field means and where it comes from: `designs/observation/`
(per block: `designs/observation/per_block_reference.md`).

## Changing the observation

- **Never hardcode an obs index.** Read `Gen3ObservationEncoder(load_mappings()).get_layout()`; every offset
  is a named constant in `constants.py`.
- 🚨 **`src/rust_sim/src/encoder/layout.rs` is Rust-owned source: change `constants.py` in the same edit.**
  `rust_core_obs_layout_test.py` parses `layout.rs` and fails a one-sided change. A layout move also moves the
  prober's pinned offsets (`src/main/prober/engine_test.py::test_offsets_resolve_matches_layout`).
- **A change in what a cell MEANS** (a value change with no shape change) sets `OBS_SEMANTICS_VERSION`, not
  `ARCH_SIGNATURE`: `src/agents/model/CLAUDE.md` "Versioning — what to bump". The value gates are the obs golden
  (`python -m agents.training.golden_obs_core --check`) and `designs/rust_sim/encoder.md` §6a. The benchmark
  every obs change runs is in the root `CLAUDE.md`.
- The archived `TurnDelta` slot and the incoming-damage obs encoder stay deleted; `turn_delta_encoder.py` and
  `incoming_damage.py` survive only as the prober's decoder and pure math.

## Hazards

- 🚨 **The event window's columns are `constants.EventCol`; compiled model code reads the plain-int mirror
  `constants.EVENT_COL`.** An `IntEnum` member inside a traced forward breaks `torch.fx` code generation
  (`SyntaxError` far from the cause), so `team_transformer` uses `EVENT_COL`; use `EventCol` for the declaration
  (tests, enumeration, the eager probe helper `feature_coverage/_support.py::obs_with_event`). Never a bare
  column index. The one-hot groups are written by indexing, so member order is load-bearing: a reorder
  relabels every historical row with no shape change (`src/agents/training/event_window_test.py`).
- **The volatile vocabulary is source-derived and crash-don't-drop**: classify a new effect id in
  `gen3_effects.py` (a slot, or a `NOT_A_VOLATILE` row), never a catch-all
  (`designs/observation/volatile_vocabulary.md`).
- **The ACTIVE flag is the last dim of the per-mon slot** (`POKEMON_ACTIVE_OFFSET`); the model's
  `hp_and_active[:, :, -1]` relies on it.
- `turns_since_progress` is a resettable stall counter, not the deadline; the game left is the global block's
  `clock` group.
- Move, species and type ids in the row are embedding ids: a new embedded column is declared, never fed to a
  Linear raw.
- The two move orders and the mypy rule (this package shares the model's `mypy.ini`) are in
  `src/agents/model/CLAUDE.md`.
