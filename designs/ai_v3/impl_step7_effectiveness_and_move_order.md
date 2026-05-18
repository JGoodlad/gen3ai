# Implementation: Step 7 — Move Effectiveness + Who-Moved-First Tracking

This step added two high-value one-turn-memory signals to the TurnDelta observation
block: the effectiveness multiplier of each side's last damaging move, and which side
executed their action first.

Primary themes: parsing previously-ignored Showdown protocol messages, flowing new
values through the full pipeline (poke-env → BattleContext → TurnDelta →
TurnDeltaEncoder), and building three independent test layers to verify correctness.

---

## Motivation

After Step 6, the TurnDelta block gave the model one-turn memory of move IDs, power,
HP deltas, and cant-reasons — but was missing two signals that are critical for
reasoning about the current matchup:

1. **Move effectiveness.** Whether our last attack hit for 0× (immune), 0.5× (resisted),
   1× (normal), or 2× (super-effective) directly tells the model whether the current
   type matchup is favourable. Without it, the model must re-derive this from type
   embeddings every turn, which is expensive for the attention heads to learn.

2. **Who moved first.** Move order reveals relative speed tiers and priority usage.
   Knowing we moved first is the difference between "we can outspeed and KO" vs.
   "we will take a hit before we can act." This is one of the most important pieces of
   information for predicting damage exchanges.

Both signals were present in the raw Showdown protocol but were either in
`MESSAGES_TO_IGNORE` or not tracked at all.

---

## What Changed

### TurnDelta block: 29 → 39 dims

The TurnDelta block (appended at offset 1102 in the full observation) grew from 29 to
39 dimensions:

| Dims | Content |
|---|---|
| 29–32 | `our_effectiveness` one-hot: [immune, resisted, normal, super-effective] |
| 33–36 | `opp_effectiveness` one-hot |
| 37–38 | `move_order`: [we_first, opp_first]; all-zero = na (switch turn / turn 0) |

All zeros when the side switched or used a non-damaging move. All zeros on the first
turn of each episode (no previous turn exists).

The `ORDER_DIM` is 2 (not 3) — "na" is represented by all-zero, consistent with how
the rest of the block handles unknown/absent values.

---

## Implementation Details

### poke-env: `abstract_battle.py`

**New slots (5):**
```python
"_our_last_effectiveness",   # Optional[tuple[int, float]] — (turn_set, multiplier)
"_opp_last_effectiveness",   # Optional[tuple[int, float]]
"_we_moved_first",           # Optional[tuple[int, bool]]
"_this_turn_move_sides",     # list[str] — deduped "ours"/"opp" per turn
```

**`MESSAGES_TO_IGNORE` change:** removed `"-supereffective"` and `"-resisted"` from the
ignore set (the no-dash variants `"supereffective"` and `"resisted"` remain for legacy
client-side messages that poke-env never sees in practice).

**`|move|` handler additions:**

```python
move_side = "ours" if event[2][:2] == self._player_role else "opp"
if move_side not in self._this_turn_move_sides:
    self._this_turn_move_sides.append(move_side)
if len(self._this_turn_move_sides) == 2:
    we_first = self._this_turn_move_sides[0] == "ours"
    self._we_moved_first = (self._turn, we_first)

# Tentative neutral effectiveness for any damaging move
_move_entry = GenData.from_gen(self._gen).moves.get(to_id_str(move), {})
if _move_entry.get("basePower", 0) > 0:
    if move_side == "ours":
        self._our_last_effectiveness = (self._turn, 1.0)
    else:
        self._opp_last_effectiveness = (self._turn, 1.0)
```

The tentative-1.0 logic is intentional: for moves with no explicit
`|-supereffective|`/`|-resisted|`/`|-immune|` message (i.e. normal-effectiveness), the
model still gets signal (1.0) rather than None. The explicit messages override it.

**New elif branches for `-supereffective`, `-resisted`, `-immune`:**

```python
elif event[1] == "-supereffective":
    if len(event) >= 3:
        defender_side = event[2][:2]
        if defender_side == self._player_role:
            self._opp_last_effectiveness = (self._turn, 2.0)
        else:
            self._our_last_effectiveness = (self._turn, 2.0)
```

(Identical pattern for `-resisted` → 0.5, and `-immune` → 0.0.)

Note: the existing no-dash `"immune"` handler was extended to also write to
`_opp_last_effectiveness` / `_our_last_effectiveness` in addition to its existing
ability-reveal logic.

**`end_turn()` change:** resets `_this_turn_move_sides = []` (move-order tracking is
per-turn). The `(turn, value)` tuples in the effectiveness fields are NOT reset — they
persist and gate via `turn_set == self._turn - 1` in the properties.

**Properties:**

```python
@property
def our_last_effectiveness(self) -> Optional[float]:
    if self._our_last_effectiveness is not None:
        turn_set, mult = self._our_last_effectiveness
        if turn_set == self._turn - 1:
            return mult
    return None
```

The `turn - 1` gate ensures the property is only readable on the turn immediately
after the move was used, and returns None at all other times (including after skipped
turns, first turn, or when the side switched).

### BattleContext: 3 new fields

```python
our_last_effectiveness: Optional[float]
opp_last_effectiveness: Optional[float]
we_moved_first: Optional[bool]
```

Set in `from_battle()` by directly reading the new properties:
```python
our_last_effectiveness=battle.our_last_effectiveness,
opp_last_effectiveness=battle.opp_last_effectiveness,
we_moved_first=battle.we_moved_first,
```

### TurnDelta: 3 new fields + TurnDeltaEncoder: 10 new dims

`TurnDelta.build()` passes through from `curr_ctx`:
```python
our_effectiveness=curr_ctx.our_last_effectiveness,
opp_effectiveness=curr_ctx.opp_last_effectiveness,
we_moved_first=curr_ctx.we_moved_first,
```

`TurnDeltaEncoder.encode()` appends:
```python
self._effectiveness_onehot(delta.our_effectiveness),   # 4
self._effectiveness_onehot(delta.opp_effectiveness),   # 4
self._order_onehot(delta.we_moved_first),              # 2
```

---

## Edge Cases Encountered

These came up during implementation and testing and are documented here because they
will recur in future work touching the poke-env parser or effectiveness logic.

### 1. Fixed-damage moves report `basePower = 0`

Seismic Toss, Dragon Rage, and Night Shade all have `basePower: 0` in `gen3_moves.json`
(despite dealing real damage). The tentative-1.0 logic correctly skips them — their
effectiveness is non-standard (damage is fixed, not type-based), so `None` is the right
value. Verified in `effectiveness_test.py::test_fixed_damage_move`.

### 2. Quad-resisted / quad-SE collapse to 2.0 / 0.5

Showdown sends exactly one `|-supereffective|` message even for 4× matchups (e.g.
Fire vs Ice/Steel). The same for `|-resisted|` on 4× resistances. We store 2.0 and
0.5 respectively. The encoder bins ≥2.0 as super-effective and ≤0.5 as resisted, so
quad and single collapses correctly into the same category.

### 3. Multi-hit moves: last write wins

Each hit of a multi-hit move (Double Kick, Fury Attack, etc.) generates its own
effectiveness message. All hits have the same type, so all fire the same multiplier.
The last write wins and the result is consistent. No special handling needed.

### 4. Missed moves: tentative 1.0 persists

When a damaging move misses (`|-miss|`), the tentative 1.0 set in the `|move|` handler
is NOT cleared. The model sees 1.0 (the type interaction exists even if damage wasn't
dealt). This is intentional — we don't want a separate "missed" bucket; the miss is
already captured by the `our_hp_delta` being zero.

### 5. Sleep Talk fires two `|move|` messages from the same side

When Sleep Talk is used, Showdown sends:
```
|move|p1a: Snorlax|Sleep Talk|p2a: ...
|move|p1a: Snorlax|Body Slam|p2a: ...   ← the randomly chosen move
```

Both entries are p1. Without deduplication, the raw-interpreter test would see
`['p1', 'p1']` and infer `we_moved_first = True`, while poke-env correctly returns None
(only one side moved). The fix: `if move_side not in self._this_turn_move_sides` in the
`|move|` handler prevents double-counting the same side. The fuzz test's raw interpreter
uses the same deduplication logic.

### 6. Status moves fire `-immune` but have `basePower = 0`

Thunder Wave against a Ground-type fires:
```
|move|p1a: Jolteon|Thunder Wave|p2a: Steelix
|-immune|p2a: Steelix
```

Thunder Wave has `basePower: 0`, so the tentative-1.0 logic does NOT fire in the
`|move|` handler. But the `-immune` handler DOES fire and sets
`_our_last_effectiveness = (turn, 0.0)`. Result: 0.0 for a non-damaging move is
technically correct (the move was immune and had no effect), but it's a different
semantic from "damaged but resisted". The encoder bins it as immune regardless.

### 7. `MESSAGES_TO_IGNORE` has two levels

`Player._handle_battle_message()` has its own small ignore set (`{"t:", "expire",
"uhtmlchange"}`). `AbstractBattle.parse_message()` has the larger set. Crucially,
`win`, `tie`, `request`, `showteam`, and `error` are handled at the Player level and
**never forwarded** to `parse_message()`. Calling `parse_message(['', 'win', '...'])`
directly (e.g. in replay-based tests) hits the catch-all `else: raise
NotImplementedError`. The replay test filters these with `_PLAYER_LEVEL` before any
`parse_message()` call.

### 8. Mid-turn forced-switch timing

When our Pokémon faints mid-turn, poke-env calls `choose_move()` for the forced switch
BEFORE `|turn|N+1|` fires. At this point, `battle._turn = N` (unchanged), and
`_opp_last_effectiveness` may be `(N, ...)` from the move that just KO'd us — but the
property checks `turn_set == N - 1`, so it returns None.

This is **correct behaviour for training** (the model always decides after a full turn
in the RL env), but means the fuzz test must skip Layer 1 effectiveness validation on
`battle.force_switch = True` turns and not count them as mismatches. The comment in
`effectiveness_fuzz_e2e_test.py` documents this skip.

### 9. `BattleContext.from_battle()` requires `SlotRegistry` instances

`from_battle()` has the signature:
```python
def from_battle(cls, battle, mask, obs, our_slots: SlotRegistry, opp_slots: SlotRegistry)
```

The `SlotRegistry` instances must persist for the lifetime of each battle (they assign
stable slot indices to Pokémon as they're revealed). Any test or player that calls
`from_battle()` directly must create and track these per-battle, initialising them on
first encounter and discarding on battle end. The fuzz test stores them in
`self._our_slots[battle_tag]` / `self._opp_slots[battle_tag]`.

### 10. Replay HTML extraction must filter player-level messages

`effectiveness_replay_test.py` extracts raw `|`-prefixed lines from replay HTML and
feeds them into `Battle.parse_message()`. The `|win|`, `|tie|`, `|request|`,
`|showteam|` messages must be filtered before calling `parse_message()` — they raise
`NotImplementedError` because they're normally consumed at the Player level before
reaching the Battle object.

### 11. Replay tests: tentative-1.0 vs raw-interpreter disagreement

The replay test's independent raw interpreter returns `None` for turns where no
explicit effectiveness message fired (normal-effectiveness moves). poke-env returns
`1.0` (tentative). These are NOT mismatches — they're by design. The test only
validates turns where `expected_our is not None or expected_opp is not None` (i.e.,
at least one explicit SE/resisted/immune message fired). Coverage of the 1.0 case is
still confirmed by using `actual` values (which include tentative-1.0) in the coverage
check rather than `expected`.

### 12. `_find_repo_root()` for replay file discovery

The replay test locates replay files by walking up the directory tree until it finds a
directory containing `models/`. A fixed relative path like
`../../../../../` will silently produce the wrong root if the file moves or the test
runs from a different worktree depth. The walk-up approach is robust to this.

---

## Test Suite

Three independent test layers verify correctness:

| File | Type | What it validates |
|---|---|---|
| `effectiveness_test.py` | Unit | 25 deterministic tests: all 6 combinations of (our/opp) × (SE/resisted/immune), fixed-damage moves, status moves, move order (4 cases), staleness, multi-hit, tentative override, p2 role, OHKO immune format |
| `effectiveness_replay_test.py` | Unit (replay) | Parses real battle HTML replays; validates poke-env matches independent raw interpretation on turns with explicit effectiveness messages; coverage check across all categories |
| `effectiveness_fuzz_e2e_test.py` | E2E | 4-layer pipeline validation (raw → poke-env → BattleContext → TurnDelta → encoder) across 3 targeted scenarios; 50 battles each; requires live Showdown server |

**Fuzz test results:** 50 battles × 3 scenarios, 18,014 decision turns, **0 mismatches
at all 4 layers**. Coverage: immune=818, resisted=2000, normal=10992, SE=922,
we_first=3966, opp_first=3394, switches=10654.

---

## What's Next

- **Hidden Power type inference** (`todo_hidden_power_inference.md`): The current
  implementation stores `type_id=0` (unknown) for Hidden Power. Inferring the type
  from damage dealt against known-type opponents would let the model reason about HP
  type matchups.

- **Turn history (N-frame)**: `EpisodeTracker._history` already stores the full
  episode. Exposing a window of past TurnDeltas (e.g. last 3 turns) would let the
  model track patterns like "they've switched three times this game" or "their mon is
  getting progressively weaker from sand/poison".
