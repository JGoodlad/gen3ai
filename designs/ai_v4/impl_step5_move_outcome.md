# Implementation: Step 5 — Move-Outcome Reporting (crit / miss / fail / cant)

This step reports the **fate of each side's move** into every turn-history slot:
did it *hit*, *miss*, or *fail*, did it *crit*, and — when a Pokémon was
prevented from acting — the full `|cant|` reason. Until now three protocol facts
were silently dropped: `|-miss|`, `|-fail|`/`|-notarget|`/`|-nothing|`, and
`|-crit|` all sat in `AbstractBattle.MESSAGES_TO_IGNORE`, and the `|cant|` reason
one-hot only covered 5 of the ~11 reasons Gen 3 emits. A missed Hydro Pump looked
identical to a 0-damage hit; a KO-by-crit looked like a routine KO; a Protect that
failed on repeat was invisible.

The data flows through the same four layers as effectiveness/move-order:
protocol → poke-env turn-gated property → `BattleContext` snapshot → `TurnDelta`
→ encoded slot. No learned embedding is added — the outcome is a 3-way categorical
and crit is one bit, so one-hots fed straight into the projection are the right
representation (embeddings earn their keep only on high-cardinality vocabularies
like species/move/type). `ARCH_SIGNATURE` bumps to `gen3_move_outcome_v1`.

Primary themes: attribute-by-current-mover (sidestepping the attacker-vs-defender
ambiguity in the raw lines), a `hit`/`miss`/`fail` derivation that respects the
`|cant|` "no move resolved" case, and a layout refactor that replaces every magic
base-block index with a computed `OFFSET_*` constant so widening the cant one-hot
doesn't silently corrupt downstream decoders.

---

## Motivation

### The gap

Three message families were ignored before any state update:

```python
MESSAGES_TO_IGNORE = { ..., "-crit", "-fail", "-miss", "-notarget", "-nothing", ... }
```

So the only "the move didn't connect" signal reaching the model was an indirect
one — a damaging `|move|` that produced no `|-damage|` and no effectiveness
emission, which the pending → promote mechanic lets *silently expire*. That is
indistinguishable from a status move, a fully-resisted hit, or a Wonder-Guard
no-op. For a **value head** the cost is concrete: losing a Pokémon to a crit is
variance, not a bad decision, and a missed 70%-accuracy move is bad luck, not a
weak play. Without the signal the critic has to absorb that variance as noise.

Separately, the cant one-hot was `["par", "slp", "frz", "flinch", "confusion"]`.
Gen 3 also emits `recharge` (Hyper Beam), `move: Taunt`, `move: Disable`,
`move: Imprison`, `ability: Truant` (Slaking), and `nopp`. Those set
`failed_to_move = True` but left the reason vector all-zeros — the model knew the
mon lost its turn but not why.

### What the protocol tells us, and the attribution trap

The raw lines disagree on whose name they carry:

| Line | Names | Fires during |
|---|---|---|
| `\|-crit\|DEFENDER` | the **defender** | attacker's move resolution |
| `\|-miss\|SOURCE\|TARGET` | the **source** (attacker) | attacker's move resolution |
| `\|-fail\|POKEMON` | usually the user | attacker's move resolution |
| `\|-notarget\|` / `\|-nothing\|` | sometimes nobody | attacker's move resolution |

Attributing off the named Pokémon means inverting the side for `-crit` but not
for `-miss`, and handling lines with no Pokémon at all. The clean invariant is
that **all of these fire while one side's move is resolving** — so the owner is
always the currently-resolving attacker, regardless of which Pokémon the line
happens to name.

---

## What Changed

### Attribute-by-current-mover (`AbstractBattle`)

A new `_current_move_user_side` (`"ours"` / `"opp"` / `None`) is set on every
`|move|` line and reset each turn in `end_turn`. Every `|-crit|`/`|-miss|`/
`|-fail|`/`|-notarget|`/`|-nothing|` is attributed to it via a single helper:

```python
def _mark_move_outcome(self, which):  # which ∈ {"crit","missed","failed"}
    side = self._current_move_user_side
    if side is None:
        return  # event fired outside any |move| context — ignore
    stamp = (self._turn, True)
    ...  # set _{our,opp}_move_{crit,missed,failed}
```

The five message types are removed from `MESSAGES_TO_IGNORE` and given explicit
handlers (`-fail`/`-notarget`/`-nothing` all fold into `"failed"`). The older
protocol variant where miss/notarget rides as a `[miss]`/`[notarget]` suffix on
the `|move|` line itself is also mirrored into the trackers, so detection is
robust to both formats.

### Six new turn-gated properties

`our/opp_move_crit`, `our/opp_move_missed`, `our/opp_move_failed` — each stored as
a `(turn_set, True)` tuple and read through `_turn_gated_bool` (gates on
`self._turn - 1`, exactly like `our_last_effectiveness` and `we_moved_first`). A
stale flag from an earlier turn never leaks forward. These are deliberately kept
**independent of `DamagingMoveEvent`**: that event only promotes on an explicit
effectiveness emission, so a neutral-effectiveness crit/miss would be lost if the
flag hung off it.

### `BattleContext` snapshot + `TurnDelta` derivation

`BattleContext` snapshots the six flags. `TurnDelta.build` collapses them into one
per-side category via `_derive_move_outcome(move_used, missed, failed, suppressed)`:

```
suppressed (|cant|, no move resolved) → None
missed                                → "miss"
failed                                → "fail"
move_used                             → "hit"
otherwise                             → None   (switch / no identifiable move)
```

`suppressed = our_failed_to_move` (the cant case) guards the `hit` fallback: a
move *selected* via the action but then prevented by paralysis/sleep must not read
as a hit. crit passes through as `delta.our/opp_move_crit` (orthogonal — a `hit`
slot may also have crit set). The cant reason itself stays in the existing
`our/opp_cant_reason` field and is reported through the (now wider) cant one-hot.

### Encoder: outcome one-hot, crit bit, full cant set

`turn_delta_encoder.py`:

- **Outcome one-hot (3)** per side — `[hit, miss, fail]`; all-zeros for
  switch / cant / no-move.
- **Crit bit (1)** per side.
- **Cant one-hot widened 5 → 11** — adds `recharge, taunt, disable, imprison,
  truant, nopp`. `_normalize_cant_reason` strips Showdown's `move: ` / `ability: `
  / `item: ` qualifier and lowercases, so `"move: Taunt"` → `taunt`. Unknown
  reasons → all-zeros (no crash).

The outcome/crit fields are placed in the extended block **immediately before the
six species IDs**, keeping the species block the contiguous slot tail that
`features_extractor._embed_delta_slot` slices for embedding. Because they land in
the extractor's existing `slot[:, 10:OFFSET_OUR_ACTOR_SPECIES]` pass-through
range, **no extractor change was needed** — `_td_embed_dim` (`TURN_DELTA_DIM - 10`)
auto-derives.

### Base-block offset refactor

Widening the cant one-hot shifts every base-block field after it. To make that
safe, all base-block positions referenced from `describe_vector` or the e2e fuzz
tests are now computed `OFFSET_*` constants (`OFFSET_OUR_CANT`, `OFFSET_OUR_EFF`,
`OFFSET_ORDER`, …) derived from the dims — no magic indices survive. The
effectiveness fuzz test's Layer 4 (previously `enc[29:33]`) was updated to read
the same named offsets.

---

## Implementation Details

### Layout (TURN_DELTA_DIM 88 → 108)

| Block | Was | Now | Δ |
|---|---|---|---|
| Base block | 39 | 51 | +12 (cant one-hot 5→11, ×2 sides) |
| Extended block | 49 | 57 | +8 (outcome 3×2 + crit 1×2) |
| **TurnDelta slot** | **88** | **108** | **+20** |

Extended-block order (unchanged prefix, then the new fields before the species
tail): boost deltas (14), phase flag (1), target_hp_delta (2), HP levels (12),
target-status onehots (14), **our/opp move-outcome onehots (6)**, **our/opp crit
(2)**, six species IDs (6).

### Observation dimension

`N_HISTORY_TURNS = 10`, so the obs grows by `10 × 20 = 200`:

| | base | + prev_mask | + history | total |
|---|---|---|---|---|
| was | 1547 | 11 | 10 × 88 = 880 | 2438 |
| now | 1547 | 11 | 10 × 108 = 1080 | **2638** |

### Version bump

`ARCH_SIGNATURE = "gen3_move_outcome_v1"` (`model_version.py`). The history
projection input width changed, so v4 (`gen3_abilities_v2`) checkpoints are not
weight-compatible — `check_compatible()` rejects them at load with a clear error,
which is correct for this rapid-iteration project.

---

## Edge Cases

- **Cant with a selected move.** Action 6–9 sets `our_move_id` even when the mon
  is then fully paralyzed; `suppressed = our_failed_to_move` forces outcome →
  None so it doesn't read as a hit. The cant reason is reported separately.
- **Faint-forced opp switch.** `opp_move_id` is recovered (not None) when the opp
  moved before dying, so its outcome/crit are still reported; a *voluntary* opp
  switch leaves `opp_move_id` None → outcome None. No switch suppression needed on
  the opp side.
- **Neutral-effectiveness crit/miss.** Captured because the flags are independent
  of `DamagingMoveEvent` promotion (which only fires on an explicit effectiveness
  emission).
- **Event outside a move context.** `_mark_move_outcome` no-ops when
  `_current_move_user_side is None`, so a stray line can't mis-attribute.
- **Forced-switch timing.** The properties are turn-gated to `_turn - 1`; read
  mid-turn on a `force_switch` before `|turn|N+1` they return False, identical to
  the effectiveness fields. The fuzz test skips Layer 1 there (Layers 2–4 stay
  internally consistent).
- **`confusion` cant reason.** Kept in the one-hot for safety even though modern
  Showdown reports a confusion self-hit as `|-activate|…|confusion` + `|-damage|`
  rather than a `|cant|` — costs one dim, avoids silently dropping it if emitted.

---

## Test Suite

### Unit tests

- `turn_delta_encoder_test.py` — failing magic-index assertions migrated to the
  named `OFFSET_*` constants; new `test_move_outcome_onehot`, `test_crit_bits`,
  and `test_cant_reason_prefix_normalized` (covers `move: Taunt` → `taunt`,
  `ability: Truant` → `truant`, bare `recharge`, and unknown → all-zeros).
- `battle_context_test` / `battle_recorder_test` — mock battle builder extended
  with the six new `our/opp_move_{crit,missed,failed}` attributes.
- `state_encoder_test` — `EXPECTED_OBS_DIM` 2438 → 2638.
- Full unit suite: **858 passed**, plus **18 integration** passing.

### E2E move-outcome fuzz (`poke_env_gaps/move_outcome_fuzz_e2e_test.py`)

New four-layer fuzz, modeled on `effectiveness_fuzz_e2e_test`. A
`MoveOutcomeFuzzPlayer` intercepts the raw `|-crit|`/`|-miss|`/`|-fail|`/
`|-notarget|`/`|-nothing|`/`|cant|` lines, replicates poke-env's
current-mover attribution, and cross-checks all four layers:

1. raw vs poke-env properties,
2. properties vs `BattleContext`,
3. `BattleContext` vs `TurnDelta` outcome derivation,
4. `TurnDelta` vs encoded one-hots/bit (at the named offsets, with cant
   normalization).

Coverage is asserted (the run fails if crit / miss / fail / ≥2 distinct cant
reasons are never observed). The `VARIANCE_TEAM` is engineered for it: Slaking
(Truant), Lovely Kiss/Hypnosis (slp), Thunder Wave (par), Ice Beam (frz), Rock
Slide (flinch), Hyper Beam (recharge), Taunt, low-accuracy Blizzard/Hydro
Pump/Cross Chop (miss), Protect/Substitute (fail), Slash (high crit).

**Result (120 battles, 6876 decision turns): 0 mismatches across all four
layers.** Coverage: hit 5653, miss 461, fail 480, crit 222, cant reasons
`{flinch, frz, par, recharge, slp, taunt, truant}`. The pre-existing
effectiveness and transition fuzz tests still pass.

---

## What This Enables

The model can now distinguish, in the turn-history window:

- a **missed** attack from a 0-damage neutral hit (variance vs weak move),
- a **failed** Protect/Substitute/Taunt from a successful one (wasted turn),
- a **crit** that explains an outsized HP swing or an unexpected KO — the single
  most useful signal for a value head separating luck from policy,
- *why* a Pokémon lost its turn (Truant loafing vs a recharge vs sleep), not just
  *that* it did.

---

## Files Changed

| File | Change |
|---|---|
| `src/poke_env/battle/abstract_battle.py` | Un-ignore 5 message types; `_current_move_user_side`; `_mark_move_outcome` + 6 turn-gated properties; `[miss]`/`[notarget]` suffix mirroring; reset in `end_turn` |
| `src/agents/training/battle_context.py` | 6 new `BattleContext` fields + snapshot; `_derive_move_outcome`; `TurnDelta` outcome/crit fields wired in `build`/`empty` |
| `src/agents/observation/turn_delta_encoder.py` | Outcome one-hot + crit bit; cant set 5→11 with `_normalize_cant_reason`; computed base-block `OFFSET_*`; `describe_vector` rewrite |
| `src/agents/model/model_version.py` | `ARCH_SIGNATURE` → `gen3_move_outcome_v1` + changelog |
| `src/agents/observation/turn_delta_encoder_test.py` | Named-offset migration + new outcome/crit/cant tests |
| `src/agents/observation/state_encoder_test.py` | `EXPECTED_OBS_DIM` 2438 → 2638 |
| `src/agents/training/battle_recorder_test.py` | Mock battle gains the 6 outcome attrs |
| `src/agents/training/poke_env_gaps/move_outcome_fuzz_e2e_test.py` | **New** four-layer e2e fuzz |
| `src/agents/training/poke_env_gaps/effectiveness_fuzz_e2e_test.py` | Layer 4 reads named eff/order offsets (shift-proof) |
