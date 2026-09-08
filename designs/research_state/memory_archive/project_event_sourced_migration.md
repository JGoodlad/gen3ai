---
name: project_event_sourced_migration
description: "ai_v4 event-sourced + strict battle-API migration — COMPLETE & shipped (ai_v4 closed out); the durable standard, how it's enforced, and the hard-won lessons"
metadata:
  type: project
  originSessionId: 7d8e97b7-8408-4667-86ce-a49dff1433e3
---

> **Archived 2026-09-08** — ai_v4 closed out and shipped; src/agents/battle/CLAUDE.md is the always-current doc of record. Preserved verbatim; nothing below is current.

# ai_v4 — event-sourced battle layer + strict battle-API (DONE)

**Status (as of 2026-05-31): COMPLETE and SHIPPED on main — `ai_v4` is closed out
(`a775765` "close out ai_v4").** ARCH_SIGNATURE = `gen3_trapping_signals_v1`, obs dim
**3321**, TURN_DELTA_DIM **159**, MODEL_CONFIG_VERSION 2. The full architecture is documented
in the root `CLAUDE.md` — don't restate it from memory; read CLAUDE.md (always in context).
Only **Phase 5b** (`LiveView` current-board *event-fold* — making the current-board read
independent of poke-env's mutable state) is parked; tracked in
`designs/ai_v4/todo_live_battle.md` and `designs/ai_v4/todo.md`.

## The durable standard (the long-term rule the user drove toward — now achieved)
**Our code never touches poke-env directly.** All battle state flows through vetted, fuzzed
read-models we own, reached via `battle.strict_view()`:
- `LiveView` — current board ("what is true now"), no past-turn state.
- `TurnView` — history fold ("what happened, in order").
- `LegalActions` — server-authoritative legality (mask/mapper read this).
poke-env is the engine underneath but an implementation detail. Motivation: poke-env's
temporal fields (`last_move`, `last_cant_reason`, `protect_counter`, mutable `effects`) cause
hard-to-find bugs (Focus-Punch / Sleep-Talk class).

**Enforcement (the lock):** `src/agents/strict_api_lock_test.py` AST-walks `observation/` /
`action/` / `training/` / `inference/` and FAILS on any raw stateful `battle.<attr>` read or
direct poke-env value-enum import. Value-enums flow through the `src/agents/enums.py`
re-export seam (`PokemonType`/`Status`/`MoveCategory`/`Weather`; `Effect` deliberately
excluded — replaced by `observation/gen3_effects.py`). Two documented seams remain (allowlisted):
`action/serialize.py` (the lone Choice→poke-env-order touch) and `training/battle_snapshot.py`
(the per-decision current-board snapshot; reads poke-env for the HP tracker). `battle_context.py`
was DELETED — split into `training/turn_delta.py` (fold-only) + `training/battle_snapshot.py`.

## Hard-won lessons worth keeping (hard to re-derive)
- **How to find gen3 effects (volatiles):** Showdown `moves.ts` `volatileStatus`/`addVolatile`
  ∩ `data/pokemon/gen3_moves.json`, **AND** `abilities.ts` ∩ `gen3_abilities.json` —
  volatiles come from BOTH moves and abilities (flashfire, ability-activation cures were
  missed by a moves-only scan). Filter to poke-env's `Effect` enum (only enum ids reach
  `LiveView.volatiles`). `encode_volatiles`/`encode_cant_reason` use **crash-don't-drop**:
  RAISE on anything unclassified rather than silently dropping. Fuzzing (not unit tests)
  found the real gaps — that tripwire loop is the design working.
- **HP-delta value-identity:** fold per-slot HP from each `DAMAGE/HEAL/SETHP` `hp_after` +
  `FAINT`→0 (bit-identical to `curr_hp − prev_hp`). An amount-SUM is NOT value-identical —
  float accumulation flips the `opp_hp_delta.sum() >= 0` futile-attack threshold (~6e-8).
  Self-KO Explosion/Selfdestruct emits NO `-damage` on the user (only `|faint|`) → HP→0 comes
  from the FAINT. Boost-stage delta is NOT event-foldable (`SETBOOST`/`clearboost`/`invert`/
  `copy`/`swap` carry only an `op`, no realized amount) → stays a LiveView snapshot read.
- **gap=0 action↔outcome desync:** a PRESSED switch can fail to realize as a switch (a move
  fires instead) on rare turns, so the event-fold says "no switch" while `record_action`
  already credited the switch from the pressed action. Documented/accepted in
  `reward_tracker.py` ("log, don't crash"); a hard invariant coupling the two will flake.
  Lesson from diagnosing it: **instrument (ring-buffer the event window), don't theorize** —
  two plausible hypotheses were both wrong before the instrumentation showed the truth.

Verified throughout by bridge-backed fuzz harnesses ([[feedback_fuzz_tests]],
[[project_local_sim_bridge]]) and a 15-min `turn_delta_fold_equivalence_fuzz_test.py`
(158k+ decisions, 0 field/reward diffs vs the retired diff-detective). Related:
[[project_training_versions]], [[project_reward_shaping_verification]].
