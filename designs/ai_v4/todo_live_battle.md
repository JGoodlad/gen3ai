# TODO: Lock down battle access behind the strict read-model API

**Status: Phases 1–5a DONE.** The strict battle-API is implemented and locked:

- **Phases 1–2** (widened `LiveView`, new `LegalActions`, the `StrictBattleView` boundary) —
  `impl_step8_strict_battle_api_and_turndelta_fold.md`.
- **Phases 3–4** (every `observation/` / `action/` / `training/` / `inference/` consumer migrated
  to read through `StrictBattleView` / `LiveView` / `LegalActions`, value-neutral; the
  `agents.enums` seam + the static `strict_api_lock_test.py` no-raw-read guard as **the lock**)
  and **Phase 5a** (`TurnDelta` folds entirely from the event log + `LiveView`;
  `battle_context.py` deleted) — `impl_step9_strict_api_perf_and_trapping.md`.

This file is kept only as the background pointer (referenced by `strict_api_lock_test.py`) and the
tracker for the one remaining item.

---

## Phase 5b — true `LiveView` current-board event-fold (FUTURE, not scheduled)

Replace the per-field copy in `LiveView.from_battle` with an **event-fold**, so poke-env's tracker
is no longer the source of current-board state and we are genuinely decoupled. Most per-mon fields
are foldable from the event log; the hard residue (each effectively a poke-env re-implementation, so
do them deliberately, behind the now-strict boundary):

- **Side-condition counter semantics** — Spikes layers, screen turn counts, Toxic Spikes, with the
  exact gen3 decrement/clear rules.
- **Ability / item inference** — the reveal-gating + uniquely-inferable-from-species logic.
- **Type-change edge cases** — Conversion, Camouflage, forme/transform interactions.
- **Action legality** stays server-authoritative via `LegalActions` even after a `LiveView` fold
  (not derivable from the event log alone).

Because Phases 1–4 put the strict boundary in place, this happens **entirely behind the API** (no
consumer changes) and is validated by the existing live e2e fuzz (event-fold `LiveView` vs
poke-env's tracker, per field, per turn). Lower priority than the open ai_v4 tail (pathology
hunting, see `todo.md`).
