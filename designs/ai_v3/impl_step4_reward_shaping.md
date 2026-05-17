# Implementation: Step 4 — Strategic Reward Shaping

This step adds a suite of reward signals to address the core behavioural failures
observed after Step 3 training: the model learned to trade Pokémon 1-for-1 but
could not close wins, rarely switched strategically, and ignored status as a
resource. It also fixes two bugs in `TurnDelta` / observation that were discovered
during this analysis period.

---

## Motivation

After Step 3, TensorBoard showed `explained_variance ≈ 0.69` and a clear
draw-seeking local optimum: the model learned symmetric trading (stay in, deal
damage, faint roughly even) but never developed the ability to press an advantage
or avoid obviously bad matchups. A replay showed Tyranitar staying in vs Suicune
and taking −58% HP from Surf on turn 1 while Zapdos (0.6% switch probability)
sat on the bench.

Three root causes were identified:

1. **No incentive to switch into a good matchup.** The switch subsidy rewarded
   *any* voluntary switch, but not specifically bringing in a Pokémon that counters
   the opponent. The model had to discover this purely from HP delta signals, which
   are too delayed for good credit assignment.

2. **Status as a resource was invisible.** The model saw HP drain from burn/toxic
   but had no explicit signal that inflicting sleep/para/burn is valuable or that
   allowing them is costly.

3. **Sleep management was unlearned.** A sleeping mon sitting in the active slot
   wastes every turn; rotating it to the bench while it wakes is free tempo, but
   the model had no reward for this.

---

## What Was Built

### 1. Destiny Bond Volatile Tracking (`e0b82ee`)

**`src/agents/observation/active_context.py`**, **`src/agents/observation/constants.py`**

Added `Effect.DESTINY_BOND` as index 8 of the volatile condition block (both sides).

- `VOLATILES_DIM`: 8 → 9
- `ACTIVE_CONTEXT_DIM`: 22 → 23
- Total obs dim: 1105 → **1107**

The features extractor auto-discovers the new projection input dim via its dummy
forward pass — no architecture change required.

Three new unit tests added: presence, absence, and `describe_vector` for Destiny Bond.

---

### 2. Phaze Move Recovery in `TurnDelta` (`c13232e`)

**`src/agents/training/battle_context.py`**

When we used Roar or Whirlwind, `TurnDelta.build()` discarded the opponent's move:
it detected an `opp_active` species change and treated the turn as a voluntary switch,
setting `opp_move_id = None`. The root cause: `BattleContext.opp_last_move_id` read
from the newly-dragged-in mon, which had no move history yet.

**Fix — `opp_all_last_move_ids`:** a new `BattleContext` field that snapshots
`last_move` for *every revealed opponent mon* (not just the active one). After a
phaze, the dragged-out mon still carries its `last_move` in `battle.opponent_team`.

`TurnDelta.build()` now checks: if `opp_switched` and `our_move_id in {"roar", "whirlwind"}`,
recover the phazed mon's move from `curr_ctx.opp_all_last_move_ids[opp_prev_active]`.

No obs dimension change — `opp_move_id` was already encoded; it now has a non-zero
value on phaze turns instead of being all-zeros.

Four new unit tests: Roar captures move, Whirlwind captures move, phazed mon
couldn't move (None correctly), voluntary switch not contaminated.

**Logging fix (`5e4f0f5`):**  
`battle_recorder.py` and `_action_str` in the trace display were updated to format
phaze turns correctly: `"surf → phazed_to:metagross"` instead of discarding the move.

**E2E fuzz Scenario D:** `ROAR_TEAM` (Skarmory / Suicune) added to
`transition_fuzz_e2e_test.py`. Tracks `phaze_opp_move_missing`; must be 0 to PASS.

---

### 3. Strategic Reward Shaping

**`src/agents/training/reward_manager.py`**

#### Switch subsidy — flat bonus replacing pool system

The original pool-based subsidy (`remaining_switch_pool = 7.5`, halved per switch,
decaying with turn and `reactive_mult`) was replaced with a simple flat constant:

```python
SWITCH_BASE_BONUS = 0.5   # per voluntary switch
```

Payout: `SWITCH_BASE_BONUS × spam_mult`, where `spam_mult = 0` for consecutive
switches (same turn immediately after the last). All other multipliers (`ratio_mult`,
`turn_decay`, `reactive_mult`, `payout`) were removed. The pool state
(`remaining_switch_pool`, `_opp_turns_active`, `_last_opp_active_name`) was
deleted. The signal is now consistent throughout the game rather than front-loaded
and decaying.

Bouncing tax (`−0.15`) and repetition tax (`−0.02`) are unchanged.

#### Roar/Whirlwind exclusion (`_last_switch_was_roared`)

Forced switches via Roar/Whirlwind are now distinguished from faint-forced switches.
`record_action()` sets `self._last_switch_was_roared = True` when a forced switch
occurs and the outgoing mon was real (not `NONE`). This flag gates several bonuses:

- **SE switch-in bonus**: skipped if roared — the incoming mon wasn't our choice.
- **Pivot bonus**: skipped if roared.
- **Sleep-swap bonus** (out): skipped if roared — no preservation value when
  opponent phazes the sleeping mon.
- **Sleep-swap penalty** (in): *not* skipped — we still end up with a sleeping
  active mon regardless of cause.

#### Status reward — one-time on change, not per-turn

The original per-turn formula (`(opp_statused − our_statused) × STATUS_BONUS` every
turn) was changed to fire **only when the count changes**:

```python
d_our = our_statused - self._prev_our_statused
d_opp = opp_statused - self._prev_opp_statused
self._prev_our_statused = our_statused
self._prev_opp_statused = opp_statused
reward = (d_opp - d_our) * STATUS_BONUS
```

Motivation: the per-turn version accumulated unbounded reward for a 3-statused
opponent — misleading when a match was already decided. The one-time signal still
fires the same `±0.3` per infliction/cure, which is the meaningful event.

Bench mons and opponent-team coverage are unchanged.

#### Sleep-swap signals (folded into `_compute_status_reward`)

- `+SLEEP_SWAP_BONUS` when we voluntarily rotate a sleeping mon *out* — not on
  Roar/Whirlwind and not on faint (no preservation value in those cases).
- `−SLEEP_SWAP_BONUS` when the mon we switch *in* is sleeping, **unless** we were
  phazing (i.e., `_last_switch_was_roared`).

These teach the model to manage sleep: bench the sleeping mon while it wakes
rather than wasting active turns.

Reward constants:

| Constant | Value | Signal |
|----------|-------|--------|
| `SWITCH_BASE_BONUS` | 0.5 | Flat per-voluntary-switch (× spam_mult) |
| `STATUS_BONUS` | 0.3 | One-time on status infliction/cure |
| `ROAR_BONUS` | 0.2 | Roar that forces a switch under spikes or vs boosted mon |
| `SE_SWITCH_BONUS` | 0.2 | Switch-in with a SE damaging move vs opponent's active |
| `SLEEP_SWAP_BONUS` | 0.25 | Rotating a sleeping mon out (+) / in (−) |

#### `_compute_roar_bonus(delta, battle)`

Fires `+ROAR_BONUS` when:
- `delta.our_move_id == "roar"` AND `delta.opp_switch_to is not None` (Roar succeeded)
- AND: opponent side has Spikes (`battle.opponent_side_conditions`) **or** opponent's
  active had positive stat boosts last turn (`self._prev_opp_boosts`)

`_prev_opp_boosts` is updated at the end of each turn from `battle.opponent_active_pokemon.boosts`.

#### `_compute_se_switch_bonus(delta, battle)`

Fires `+SE_SWITCH_BONUS` when `delta.our_switch_to is not None` (and not a roared
switch) and the newly active mon has at least one move with `base_power > 0` that
hits `≥ 2×` vs the opponent's active types. Uses the shared
`self._type_chart = GenData.from_gen(3).type_chart`.

This directly targets the failure mode in the Tyranitar/Suicune replay: switching
Zapdos in (Thunderbolt is 2× vs Suicune) would have fired the bonus.

#### `_compute_spikes_bonus(delta, battle)`

Bridges the credit assignment gap between setting hazards and the delayed entry chip:

```python
curr = battle.opponent_side_conditions.get(SideCondition.SPIKES, 0)
new_layers = curr - self._prev_opp_spikes
self._prev_opp_spikes = curr
if new_layers > 0:
    return new_layers * SPIKES_LAYER_BONUS      # +0.5 per layer
if delta.our_move_id == "spikes" and curr == 3:
    return SPIKES_WASTE_PENALTY                  # −0.2 wasted turn
return 0.0
```

- Detection is state-diff based: if the move fails (e.g., Magic Coat), the count
  doesn't increase and no bonus fires.
- The penalty fires specifically when the agent chose Spikes but the field was
  already at the 3-layer cap — a guaranteed wasted turn.
- The HP reward still fires when entry chip lands on switch-in; the setup bonus is
  additive, not a replacement.

| Constant | Value | Signal |
|----------|-------|--------|
| `SPIKES_LAYER_BONUS` | +0.5 | Per layer successfully added to opponent's side |
| `SPIKES_WASTE_PENALTY` | −0.2 | Spikes used when 3 layers already up |

---

## Reward Signal Summary

All shaping signals remain small relative to the base reward scale:

| Signal | Scale | Notes |
|--------|-------|-------|
| Faint | ±2.0 | base |
| Victory | ±30.0 | terminal |
| Switch subsidy | +0.5 | one-time per voluntary switch (× spam_mult); 0 if consecutive |
| Status infliction/cure | ±0.3 | one-time on the event turn |
| Sleep swap out | +0.25 | one-time, voluntary only |
| Sleep swap in | −0.25 | one-time, unless phazing |
| SE switch-in | +0.2 | one-time; skipped if phazing |
| Roar bonus | +0.2 | one-time on Roar turn |
| Pivot bonus | +0.1–0.15 | one-time; skipped if phazing |
| Spikes layer set | +0.5 | per layer added (max +1.5 for full spikes) |
| Spikes waste | −0.2 | used Spikes at 3-layer cap |

---

## Files Changed

| File | Change |
|------|--------|
| `src/agents/observation/active_context.py` | Destiny Bond at volatile index 8 |
| `src/agents/observation/constants.py` | `VOLATILES_DIM` 8→9, `ACTIVE_CONTEXT_DIM` 22→23 |
| `src/agents/observation/active_context_test.py` | 3 new Destiny Bond tests |
| `src/agents/observation/state_encoder_test.py` | `EXPECTED_OBS_DIM` updated 1105→1107 |
| `src/agents/training/battle_context.py` | `opp_all_last_move_ids` field; phaze recovery in `TurnDelta.build()` |
| `src/agents/training/battle_context_test.py` | 4 new phaze unit tests |
| `src/agents/training/poke_env_gaps/transition_fuzz_e2e_test.py` | Scenario D (ROAR_TEAM) |
| `src/agents/training/reward_manager.py` | `SWITCH_BASE_BONUS` (flat subsidy replaces pool); `STATUS_BONUS`, `ROAR_BONUS`, `SE_SWITCH_BONUS`, `SLEEP_SWAP_BONUS`, `SPIKES_LAYER_BONUS`, `SPIKES_WASTE_PENALTY`; `_compute_roar_bonus`, `_compute_se_switch_bonus`, `_compute_status_reward`, `_compute_spikes_bonus`; `_prev_opp_boosts`, `_prev_opp_spikes`, `_prev_our_statused`, `_prev_opp_statused`, `_last_switch_was_roared` tracking; roar-exclusion gates on SE/pivot/sleep-swap bonuses |
| `src/agents/model/features_extractor.py` | Phaze turn display in `_action_str` |
| `src/agents/training/battle_recorder.py` | Phaze turn format: `"move → phazed_to:species"` |
| `CLAUDE.md` | Obs dim table updated to 1107 |

---

## What's Next

See `designs/ai_v3/todo.md`. Priority items after this step:

1. **Stat-stage deltas in `TurnDelta`** — Calm Mind / Dragon Dance boosts reset on
   switch; Intimidate fires on switch-in. Needed to reward setup moves properly and
   to penalise letting the opponent boost unchecked. Tracking requires per-slot
   before/after snapshots since Aromatherapy clears the whole team at once.
2. **Turn-history memory** — Sliding window of K `TurnDelta` blocks, or a GRU over
   the turn sequence, once the current reward signals are stable.
3. **Delegating move gap** — Metronome/Nature Power/Assist produce `opp_move_known=False`;
   low priority since these moves rarely appear in gen3ou.
