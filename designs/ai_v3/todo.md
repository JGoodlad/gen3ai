# AI v3 — Future Work

---

## 1. poke-env: Delegating Move `last_move` Gap

**Where:** `src/poke_env/battle/pokemon.py` → `Pokemon.moved()`
**Research:** `src/agents/training/poke_env_gaps/README.md`

### Problem

`TurnDelta.opp_move_id` is sourced from `battle.opponent_active_pokemon.last_move`, which
poke-env tracks via `Move._is_last_used`. When a delegating move (Metronome, Nature Power,
Assist, Mirror Move) fires its delegated action, poke-env calls `moved(delegated_id,
reveal=False)`, which sets `move=None` and clears all `_is_last_used` flags. Result:
`last_move = None` after the turn, even though the opponent clearly acted.

Affected moves in Gen 3:
- **Metronome** — picks a random move from the entire move pool
- **Nature Power** — becomes Swift in standard Showdown terrain
- **Assist** — picks a random teammate move (Delcatty, Persian)
- **Mirror Move** — copies the opponent's last move (Pidgeot, Swellow)

Sleep Talk is **not** affected (poke-env uses `pass` for it, not `reveal=False`), except
when delegation fails (0 PP) — in that case `last_move = "sleeptalk"`, which is correct.

### Current Impact

Low. None of these moves are standard in serious gen3ou teams. The model handles the gap
gracefully via `opp_move_known = False`. Confirmed by fuzz testing: `our_move_slot_unknown`
is 0 across 30K transitions, meaning our side is never affected.

### Proposed Fix (5-line poke-env change)

In `Pokemon.moved()`, always track the move ID regardless of `reveal`:

```python
if use:
    self._last_used_move_id = to_id_str(move_id)   # new field, always set
    for m in self.moves.values():
        m._is_last_used = m is move
```

Then expose `last_move_id: str | None` as a property that returns `_last_used_move_id`
directly, bypassing the `moves` dict scan. `BattleContext.opp_last_move_id` in
`gen3_env.py` would read this instead of `last_move.id`.

### Also: Explosion / Self-Destruct faint gap

When the opponent uses Explosion and faints, `opponent_active_pokemon` is already the
new switch-in by the time `_get_observation()` runs, so `last_move = None` for the
switch-in. Captured by `TurnDelta.opp_fainted = True` and `opp_prev_active`. Acceptable.

Fix requires intercepting `|faint|` in `AbstractBattle._parse_message()` to snapshot
`last_move` before the active slot changes. Slightly more invasive than the `moved()` fix.

---

## 2. Status and Stat-Stage Deltas in `TurnDelta`

**Where:** `src/agents/training/battle_context.py` → `TurnDelta`

### Problem

`TurnDelta` currently tracks HP deltas and faint events but not:
- **Status conditions** — burn, paralysis, sleep, freeze, poison applied or cured this turn
- **Stat stages** — Calm Mind boosts, Intimidate drops, etc.

These matter for reward signal (e.g., reward for inflicting burn) and future observation
encoding (knowing the opponent is now paralyzed changes move selection).

### Complexity

- **Aromatherapy / Heal Bell** clears the entire team's status at once — needs per-slot
  before/after snapshots, not a single delta.
- **Stat stages reset on switch** — need to track stage values per active slot across turns.
- **Intimidate** applies on switch-in, before the first move — needs careful turn ordering.

Add `our_status_delta: dict[str, str | None]` and `opp_status_delta` (slot → new status),
plus `our_stage_delta: np.ndarray` (6-stat vector per slot) when the architecture is ready
to consume them.

---

## 3. `TurnDeltaEncoder` — One-Turn Memory in the Observation Vector ✓ DONE

**Where:** `src/agents/observation/turn_delta_encoder.py`
**Implemented in Step 3 (same session as TurnDelta).** 29-dim block appended to obs by `gen3_env.embed_battle()`. See CLAUDE.md for layout.

### Problem

The current observation vector encodes the raw battle state (HP, status, moves, matchups)
but nothing about what happened *last turn*. The model must infer momentum signals
(the opponent just used Rock Slide, we should expect flinch pressure) from patterns in
consecutive obs frames, which is slow to learn for a feedforward network.

### Proposed Design

Append a fixed-dim block to the observation vector encoding the previous turn's
`TurnDelta`. Keeps the feedforward architecture (no LSTM required for basic one-turn
memory):

```
TurnDeltaEncoder output (~32 dims):
  our_move_id_embed       (16,)  — move embedding, zeros if we switched
  our_switched            (1,)   — bool
  our_failed_to_move      (1,)   — bool
  our_cant_reason_onehot  (5,)   — [par, slp, frz, flinch, confusion]
  opp_move_id_embed       (16,)  — zeros if opp switched or move unknown
  opp_switched            (1,)   — bool
  opp_failed_to_move      (1,)   — bool
  opp_cant_reason_onehot  (5,)   — same categories
  our_hp_delta_sum        (1,)   — scalar damage we took
  opp_hp_delta_sum        (1,)   — scalar damage we dealt
  we_fainted              (1,)
  opp_fainted             (1,)
  opp_move_known          (1,)   — False signals Explosion gap or new active mon
```

The `TurnDelta.empty()` sentinel (first turn of episode) maps to an all-zeros block.

### Trade-offs

- **Pro:** Gives the model clear signal for Rock Slide flinch value, paralysis disruption,
  Sleep Talk sequencing — things the current obs can only imply.
- **Pro:** No architecture change required; just widens the projection layer input.
- **Con:** Adds ~32 dims permanently. Verify `Gen3FeaturesExtractor` projection input
  updates correctly (it auto-discovers dim via dummy forward pass, so no hardcoding needed).

---

## 4. Observation / Encoding: Volatile Count Encoding

**Where:** `src/agents/observation/active_context.py`

`active_context.py` encodes volatiles as binary flags only. Two cases where count matters:
- **Perish Song**: 0–3 turns remaining, encoded as a single bit regardless.
- **Sleep**: 1–7 turns remaining per Gen 3 mechanics; you cannot reset the counter by
  switching. The network cannot learn sleep-turn-aware switch timing without this signal.

Not critical for early training, but affects late-game decision quality.
