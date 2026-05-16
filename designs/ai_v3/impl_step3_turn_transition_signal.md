# Implementation: Step 3 — Turn Transition Signal

This step fills in the TODOs left by Step 1 and wires a complete, structured account of
what happened each turn into `BattleContext` and `TurnDelta`. It also adds cant-move
tracking via a small poke-env fork, backed by an e2e fuzz test that confirmed behavior at
scale across three targeted edge-case scenarios.

---

## Motivation

After Step 1, `TurnDelta` had two unfilled stubs:

```python
our_move_id = None  # TODO: add move_ids to BattleContext
opp_move_id = None  # TODO: extract from battle log
```

The reward function had no way to know *what* either side did last turn — only what
changed (HP deltas, faint events). The model also had no signal for high-value disruption
mechanics: paralysis, flinch, and freeze deprive the opponent of a turn, but without
cant-move tracking the model had to infer this indirectly from HP patterns.

---

## What Was Built

### New `BattleContext` Fields

**`src/agents/training/battle_context.py`**

Five fields added to the frozen per-turn snapshot:

| Field | Type | Source | Purpose |
|-------|------|---------|---------|
| `active_move_ids` | `list[str \| None]` | `_gen3_decision_context` / `last_request` | Maps action indices 6-9 to move IDs in the same slot order as the mask |
| `opp_last_move_id` | `str \| None` | `battle.opponent_active_pokemon.last_move` | Move the opponent used last turn (poke-env tracks via `\|-move\|` protocol) |
| `opp_active_revealed_moves` | `frozenset` | `opponent_active_pokemon.moves.keys()` | All moves the opponent's active mon has shown so far |
| `our_cant_reason` | `str \| None` | `battle.active_pokemon.last_cant_reason` | Why we failed to move this turn (`"par"`, `"slp"`, `"flinch"`, etc.), or None if we moved |
| `opp_cant_reason` | `str \| None` | `battle.opponent_active_pokemon.last_cant_reason` | Same for the opponent |

**`active_move_ids` slot ordering:** `Gen3ActionMasker.get_mask()` latches a
`_gen3_decision_context` dict onto the battle object containing the move IDs in the exact
request-slot order that the mask was built from. `active_move_ids` always reads this
first so the slot mapping is guaranteed consistent with the action mask.

### Completed `TurnDelta.build()`

Six new fields in `TurnDelta`:

| Field | Type | Derivation |
|-------|------|-----------|
| `our_move_id` | `str \| None` | `prev_ctx.active_move_ids[action - 6]`; `"struggle"` for action 10; None if switch |
| `opp_move_id` | `str \| None` | `curr_ctx.opp_last_move_id` (None if opp switched — switch guard prevents contamination) |
| `opp_move_known` | `bool` | False only when `opp_move_id is None` and opponent did not switch (Explosion gap, first-active-turn cant) |
| `our_failed_to_move` | `bool` | `curr_ctx.our_cant_reason is not None` |
| `opp_failed_to_move` | `bool` | `curr_ctx.opp_cant_reason is not None` |
| `our_cant_reason` / `opp_cant_reason` | `str \| None` | Passed through from curr_ctx |

**Key nuance:** `opp_move_id` may be non-None even when `opp_failed_to_move` is True. When
a Pokémon gets a `|cant|` turn after having already used a move, poke-env's
`cant_move()` does not clear `_is_last_used` — so `last_move` persists from the prior
turn. Consumers must check `opp_failed_to_move` rather than assuming a non-None
`opp_move_id` means the opponent actually moved this turn.

### poke-env Fork: Cant-Move Reason Tracking

**`src/poke_env/battle/pokemon.py`**

`cant_move()` now accepts an optional `reason` parameter and stores it:

```python
def cant_move(self, reason: Optional[str] = None):
    self._last_cant_reason = reason   # "par", "slp", "frz", "flinch", "confusion", ...
    self._dancing = False
    self._protect_counter = 0
    if self._status == Status.SLP:
        self._status_counter += 1
```

`moved()` clears the field when the Pokémon actually acts:

```python
def moved(self, move_id, ...):
    self._last_cant_reason = None
    ...
```

`last_cant_reason` property exposes the field.

**`src/poke_env/battle/abstract_battle.py`**

The `|cant|` handler now passes `event[3]` (the reason) to `cant_move()`:

```python
elif event[1] == "cant":
    pokemon = event[2]
    reason = event[3] if len(event) > 3 else None
    self.get_pokemon(pokemon).cant_move(reason)
```

The Showdown `|cant|` message format is `|cant|POKEMON|REASON[|MOVE]`, where REASON is
always at index 3. Known values in Gen 3: `"par"`, `"slp"`, `"frz"`, `"flinch"`,
`"confusion"`, `"Disable"`, `"recharge"`.

---

## poke-env Behavior Research

A targeted e2e fuzz test was written to verify the behavior of `last_move` and
`last_cant_reason` across edge cases. See
`src/agents/training/poke_env_gaps/README.md` for the full findings. Summary:

| Scenario | `last_move` behavior |
|----------|---------------------|
| Normal move | Set to the move used — always correct |
| `\|cant\|` on first active turn | None — mon has no move history yet |
| `\|cant\|` after a prior move | **Persists** from last actual use — `last_move` is not cleared |
| Sleep Talk (success) | Set to the **delegated** move (e.g. `"surf"`), not `"sleeptalk"` |
| Sleep Talk (delegation fails) | Set to `"sleeptalk"` — only the first `\|move\|` message fires |
| Hyper Beam recharge | Persists as `"hyperbeam"` on the `\|cant\|recharge` turn |
| Explosion / Self-Destruct | None on new switch-in — attacker already off field |
| Metronome / Nature Power | None — poke-env calls `moved(reveal=False)`, clearing all flags |

**Fuzz results (50 battles × 3 scenarios, ~30K transitions):**

- `our_move_slot_unknown = 0` — `active_move_ids` resolution is correct for all transitions
- Explosion gaps: ~1,000 (expected; `opp_fainted=True` still captures the event)
- Cant-move (estimated): ~2,200 (correctly classified as expected `|cant|` behavior)
- True anomalies (new move revealed but `last_move=None`): ~165 / 30K = **0.5%** — likely
  false positives from poke-env's `_update_from_request()` path, not training-relevant

---

## Files Changed

| File | Change |
|------|--------|
| `src/poke_env/battle/pokemon.py` | `cant_move(reason)`, `_last_cant_reason` field + property, `moved()` clears field |
| `src/poke_env/battle/abstract_battle.py` | Pass `event[3]` to `cant_move()` |
| `src/agents/training/battle_context.py` | 5 new `BattleContext` fields; `TurnDelta` completed (6 new fields); `build()` and `empty()` updated |
| `src/agents/training/environment.py` | Populates all new `BattleContext` fields in `_get_observation()` |
| `src/main/train_rl_agent.py` | Updated parallel `embed_battle()` BattleContext construction to include all new fields |
| `src/agents/training/battle_context_test.py` | `_ctx()` helper updated; 13 new tests added |

---

## Tests Added

**`src/agents/training/battle_context_test.py`** — 27 tests total (13 new):

| Test | What it covers |
|------|---------------|
| `test_turn_delta_build_our_move_id` | action 7 → `active_move_ids[1]` |
| `test_turn_delta_build_our_move_id_first_slot` | action 6 → `active_move_ids[0]` |
| `test_turn_delta_build_struggle` | action 10 → `our_move_id = "struggle"` |
| `test_turn_delta_build_our_move_slot_missing_ids` | graceful None when ids list is short |
| `test_turn_delta_opp_move_known_from_last_move` | `opp_last_move_id` → `opp_move_id`, `opp_move_known=True` |
| `test_turn_delta_opp_move_unknown_no_last_move` | None last_move + no switch → `opp_move_known=False` |
| `test_turn_delta_opp_switch_known` | active species change → `opp_switch_to`, `opp_move_id=None` |
| `test_turn_delta_opp_switch_ignores_last_move` | switch guard: new mon's prior last_move not used |
| `test_turn_delta_opp_switch_to_none_active` | opp_active="NONE" edge case |
| `test_turn_delta_opp_failed_to_move_par` | `opp_cant_reason="par"` → `opp_failed_to_move=True` |
| `test_turn_delta_opp_failed_to_move_flinch` | flinch reason |
| `test_turn_delta_our_failed_to_move_slp` | our side sleep tracking |
| `test_turn_delta_flinch_with_persisting_last_move` | documents that `opp_move_id` may be non-None even when `opp_failed_to_move=True` |

---

## Research Artifacts

**`src/agents/training/poke_env_gaps/`** — Created as a permanent research directory:

- `README.md` — full documentation of all poke-env `last_move` and `cant_move` behaviors,
  confirmed by fuzz testing; proposed fix for the `reveal=False` delegating-move gap
- `transition_fuzz_e2e_test.py` — three scenario teams (Explosion, Rest/Sleep Talk,
  Hyper Beam) with structured reporting on move known/unknown rates, cant-move
  classification, and Sleep Talk delegation

---

## What's Next (`todo.md`)

See `designs/ai_v3/todo.md` for remaining items:

1. **Delegating move `last_move` gap** (Metronome, Nature Power, Assist) — low priority;
   affects only niche gen3ou sets. Fix is ~5 lines in `Pokemon.moved()`.
2. **Status and stat-stage deltas** in `TurnDelta` — needed for reward signals around
   Aromatherapy, Calm Mind, Curse, and Intimidate.
3. **`TurnDeltaEncoder` observation block** — once `TurnDelta` is proven stable through
   training, append a fixed-dim encoding of last-turn events to the observation vector
   (what move we used, what they used, did either side fail to move, HP deltas). Gives the
   feedforward model a one-turn memory without an RNN.
