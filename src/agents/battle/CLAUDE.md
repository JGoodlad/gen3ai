# CLAUDE.md — Battle read-models (`src/agents/battle/`)

There is no Python battle layer: every reading, row, view, legality surface, tracker and reward is the
Rust core's (`src/rust_sim/`: `present/` the view, `core_events/` the event record, `trackers/`, `encoder/`).
This package is a thin Python side of it.

- **`live_view.py`: `LiveView` / `LegalActions` and their parts are frozen data classes that compute
  nothing.** `core_view.py` copies every field in from the core's `present()` / `legal_actions()` JSON; a
  presentation rule belongs in `src/rust_sim/src/present/` (contract: `designs/rust_sim/present.md`). The
  view is the current board only, with no past-turn field (`live_view_test.py` pins the shape).
- **Struggle is a flag, never a move slot**: a lone `struggle` request entry sets `LegalActions.struggle`.
- **What a protocol line revealed is current knowledge and lives on the view** (`LiveMove.seen`,
  `LivePokemon.item_public` / `ability_public`, `LiveView.residual_done`), carried by the core's
  `present()`.
- **Two faint-cause vocabularies** (`faint_causes.py`): `FAINT_CAUSE_VOCAB` (8) sizes the frozen archive
  `TurnDelta` frame the prober decodes; `FAINT_CAUSE_VOCAB_LIVE` (10) sizes the event window
  (`team_transformer.EventSeats`), held equal to `layout.rs` by
  `src/agents/observation/rust_core_obs_layout_test.py`.
- **`core_corpus_test.py`** (`sim`, routine gate) replays the 12 recorded battles of
  `rust_core_parity_fixtures/commit_tier.json.gz` through `core_replay`: every battle replays `ok`, every
  row is fully written, every decision's tokens are its legal actions, the information boundary holds, and
  the golden records re-write byte-identically. Contract: `designs/rust_sim/core_events.md`.
