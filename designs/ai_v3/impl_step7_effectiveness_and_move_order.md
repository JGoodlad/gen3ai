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

### Move effectiveness

Whether our last attack hit for 0× (immune), 0.5× (resisted), 1× (normal), or 2×
(super-effective) directly communicates the type matchup quality. Without it, the model
has to re-derive this every turn from type embeddings and species IDs — an indirect
inference that requires the model to have learned the Gen 3 type chart and know the
opponent's type from a partially-observed team. The effectiveness one-hot is a direct
shortcut: "this matchup is bad, switch out" vs. "this matchup is great, stay in."

The signal is especially valuable when the opponent has unrevealed or mixed types (e.g.
Gyarados is Water/Flying — Electric is 2× because Flying, not 2× because Water). The
type chart calculation is non-trivial; a one-hot read directly from Showdown's damage
resolution is authoritative.

### Who moved first

Move order encodes relative speed tier, which is one of the most important strategic
facts in Gen 3 OU. Knowing we moved first last turn tells the model:

- Our active Pokémon is likely faster than theirs (if neither used priority)
- We got to deal damage before taking it this turn
- If we stay in, we probably move first next turn too (barring switches, paralysis, or
  speed boosts)

Knowing we moved SECOND is equally important: it means the opponent will act before us
again, we need to plan around taking a hit, and switching in a faster mon may flip the
matchup.

Both signals were present in the raw Showdown protocol. `|-supereffective|` and
`|-resisted|` were in `MESSAGES_TO_IGNORE`. Move order was tracked nowhere.

---

## What Changed

### TurnDelta block: 29 → 39 dims

The TurnDelta block (appended at offset 1102 in the full observation vector) grew from
29 to 39 dimensions:

| Dims | Content |
|---|---|
| 0–4 | our_move features (move_id, power_norm, has_secondary, has_recoil, type_id) |
| 5–9 | opp_move features (same layout) |
| 10–13 | our_switched, opp_switched, our_failed_to_move, opp_failed_to_move |
| 14–18 | our_cant_onehot: [par, slp, frz, flinch, confusion] |
| 19–23 | opp_cant_onehot |
| 24–28 | our_hp_delta, opp_hp_delta, we_fainted, opp_fainted, opp_move_known |
| **29–32** | **our_effectiveness one-hot: [immune, resisted, normal, super-effective]** |
| **33–36** | **opp_effectiveness one-hot** |
| **37–38** | **move_order: [we_first, opp_first]; all-zero = na** |

Full layout included for reference — the pre-existing dims (0–28) are unchanged.

All-zero effectiveness means the side switched, used a non-damaging move (status/OHKO
status/fixed-damage), or it is turn 0 of the episode (no previous turn). All-zero
move_order means one or both sides did not use a move this turn (switch, cant, first
turn).

`ORDER_DIM` is 2, not 3. "Na" is all-zero, consistent with every other absent-value
encoding in the block. A 3-dim one-hot with an explicit "na" slot would waste a
dimension since the model can detect all-zero.

---

## What the Model Can Learn from These Signals

### From effectiveness

The model can now directly identify:
- **"Bad matchup, I should switch"** — opp_effectiveness has immune/SE lit, we're
  taking unreciprocated damage
- **"Great matchup, stay in"** — our_effectiveness is SE, theirs is resisted/immune
- **"Neutral exchange"** — both sides at 1.0, decision is pure speed/damage math

Previously the model had to triangulate this from species IDs, type embeddings, and the
attention heads — an indirect multi-step inference. The one-hot makes it a first-class
feature the policy head can read directly.

The feature is particularly valuable for **type-coverage decisions**: when debating
which move to use, knowing that our last Earthquake was immune (Levitate) is an
explicit signal to choose a different coverage move this turn.

### From move order

The model can now learn:
- **Speed tier estimation** — after several turns with the same matchup, the model
  accumulates evidence about relative speed. If we_first is consistently True with no
  priority, we are faster.
- **Priority detection** — when we_first flips unexpectedly (our normally-faster mon
  went second), the opp likely used a priority move. The move_id block already carries
  that move; the model can correlate.
- **Switch timing** — order=na (all-zero) signals a switch happened. Combined with
  opp_switched=1 in scalars, the model knows the opponent switched and should
  reconsider the type matchup.
- **Paralysis interaction** — paralysis reduces speed to 25% in Gen 3. A paralyzed
  fast mon may move after a normally-slower one. The model has paralysis status in
  the per-Pokémon vector (condition one-hot) and `we_moved_first` in the delta; it can
  learn this interaction.

### Limits of what we_moved_first tells you

`we_moved_first` is a **binary observation of the outcome**, not a full speed model.
It conflates multiple causes:

| Cause of we_first=True | Notes |
|---|---|
| Our speed > opp speed, same priority | Most common; implies speed advantage |
| We used priority move (Quick Attack +1, ExtremeSpeed +2) | Visible in move_id block |
| Opp used negative priority (Vital Throw -1, Roar -6) | Visible in opp move_id block |
| Opp was fully paralyzed (25% speed) | Visible in condition bits |

| Cause of we_first=False | Notes |
|---|---|
| Opp speed > our speed, same priority | Most common; implies speed disadvantage |
| Opp used priority move | Visible in opp move_id block |
| We used negative priority move | Visible in our move_id block |
| We were fully paralyzed | Visible in condition bits |

| Cause of we_first=None | Notes |
|---|---|
| Either side switched (switch has implicit priority) | opp_switched/our_switched bits |
| Only one side used a move (other canted) | cant_onehot bits lit |
| First turn of episode | turn 0, no previous data |
| Speed tie (randomly resolved either way) | Looks like we_first=True/False; indistinguishable |

The model cannot infer exact speed stats from `we_moved_first` alone — it can only
accumulate probabilistic evidence. Future explicit speed features (Step 8+) should
extend this.

**Important for Gen 3 specifically:** there is no Choice Scarf in Gen 3. Speed is
determined by base stat × EVs × nature, plus Agility/speed-stage boosts. The model
can learn approximate speed tiers from repeated observation of `we_moved_first` across
different matchups.

---

## Implementation Details

### poke-env: `abstract_battle.py`

**New slots (4 used, 1 named but unused):**
```python
"_our_last_effectiveness",   # Optional[tuple[int, float]] — (turn_set, multiplier)
"_opp_last_effectiveness",   # Optional[tuple[int, float]]
"_we_moved_first",           # Optional[tuple[int, bool]]
"_this_turn_move_sides",     # list[str] — "ours"/"opp" in move order, deduped
```

**Why `(turn, value)` tuples instead of resetting in `end_turn()`:**

The naive approach would be to reset `_our_last_effectiveness = None` in `end_turn()`
and then re-set it during the next turn's move events. This creates a race condition:
if `BattleContext.from_battle()` is called at any point during `end_turn()` processing,
it would see `None` when the previous turn's value is actually valid and needed.

The tuple approach stores `(turn_N, value)` persistently. The property gates on
`turn_set == self._turn - 1`, which is only True when read from the immediately
following turn. This is safe regardless of call order — snapshot before or after
`end_turn()` reads the right thing. Old values from two or more turns ago silently
fail the gate and return None.

**`MESSAGES_TO_IGNORE` change:**

Removed `"-supereffective"` and `"-resisted"` (leading-dash variants).

The no-dash variants `"supereffective"` and `"resisted"` remain in the set. These are
legacy client-side display messages from the old PS client protocol; the server-side
damage resolution messages that reach our parser always have the leading dash. The two
variants are distinct message types.

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

The `if move_side not in self._this_turn_move_sides` guard is load-bearing — it
handles Sleep Talk (which fires two `|move|` messages for the same side; see edge
case 5) and any other multi-message move mechanic.

The tentative-1.0 logic: for normal-effectiveness moves, Showdown emits no explicit
`|-supereffective|`/`|-resisted|`/`|-immune|` message. Setting 1.0 at move-time and
overriding on explicit messages means normal effectiveness gets a signal (1.0) rather
than being indistinguishable from "switched/non-damaging" (None). This is intentional.

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

Identical pattern for `-resisted` → 0.5 and `-immune` → 0.0.

The `if len(event) >= 3` guard handles malformed messages gracefully. In practice
the defender Pokémon identifier is always present in Showdown's output.

Note: the existing no-dash `"immune"` handler (ability-reveal path) was extended to
also write effectiveness in addition to its existing ability-reveal logic. The
leading-dash `"-immune"` is a new separate handler.

**`end_turn()` change:**
```python
def end_turn(self, turn: int):
    self._this_turn_move_sides = []   # reset per-turn move-order accumulator
    self.turn = turn                  # advance counter; tuple gates recalibrate automatically
    for mon in self.all_active_pokemons:
        if mon:
            mon.end_turn()
```

Only `_this_turn_move_sides` is reset. The effectiveness tuples and `_we_moved_first`
persist — they become stale naturally when `self._turn` advances past `turn_set + 1`.

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

Returns None in all cases except "the stored value was written during the immediately
preceding turn." Stale values, first-turn accesses, switch turns, and non-damaging
moves all return None cleanly.

### BattleContext: 3 new fields

```python
our_last_effectiveness: Optional[float]   # 0.0/0.5/1.0/2.0 or None
opp_last_effectiveness: Optional[float]
we_moved_first: Optional[bool]            # True/False or None
```

Set in `from_battle()` by directly reading the properties:
```python
our_last_effectiveness=battle.our_last_effectiveness,
opp_last_effectiveness=battle.opp_last_effectiveness,
we_moved_first=battle.we_moved_first,
```

The snapshot happens after `end_turn(N)` advances `battle._turn` to N, so the
properties correctly gate on `turn_set == N-1` and return the previous turn's values.

### TurnDelta: 3 new fields

```python
our_effectiveness: Optional[float]   # passed through from curr_ctx
opp_effectiveness: Optional[float]
we_moved_first: Optional[bool]
```

`TurnDelta.build()` passes these directly from `curr_ctx`:
```python
our_effectiveness=curr_ctx.our_last_effectiveness,
opp_effectiveness=curr_ctx.opp_last_effectiveness,
we_moved_first=curr_ctx.we_moved_first,
```

These are "last-turn" values (what happened on the turn that just ended), which is
exactly what `curr_ctx` represents. No derivation needed.

### TurnDeltaEncoder: 10 new dims

```python
def _effectiveness_onehot(self, mult: Optional[float]) -> np.ndarray:
    vec = np.zeros(EFF_DIM, dtype=np.float32)   # EFF_DIM = 4
    if mult is None: return vec
    if mult == 0.0:    vec[0] = 1.0  # immune
    elif mult <= 0.5:  vec[1] = 1.0  # resisted (catches 0.25 quad-resist too)
    elif mult == 1.0:  vec[2] = 1.0  # normal
    else:              vec[3] = 1.0  # SE (catches 4x too)
    return vec

def _order_onehot(self, we_first: Optional[bool]) -> np.ndarray:
    vec = np.zeros(ORDER_DIM, dtype=np.float32)   # ORDER_DIM = 2
    if we_first is True:   vec[0] = 1.0
    elif we_first is False: vec[1] = 1.0
    return vec
```

Appended in `encode()`:
```python
self._effectiveness_onehot(delta.our_effectiveness),   # 4 dims
self._effectiveness_onehot(delta.opp_effectiveness),   # 4 dims
self._order_onehot(delta.we_moved_first),              # 2 dims
```

The `<=` threshold in `_effectiveness_onehot` means a 0.25× quad-resist maps to
`resisted` rather than requiring an exact match. The `else` branch catches any value
>1.0 (including 4× quad-SE) as super-effective. Both are correct: we only know the
category from Showdown's message, not the exact multiplier.

---

## Edge Cases Encountered

These are documented in full because they will recur in any future work touching the
poke-env parser, effectiveness logic, or direct Battle object testing.

### 1. Fixed-damage moves report `basePower = 0`

Seismic Toss, Dragon Rage, and Night Shade all have `basePower: 0` in `gen3_moves.json`
despite dealing real damage. The tentative-1.0 logic correctly skips them (the
`basePower > 0` guard fails). Their damage is not type-based, so `None` is the right
effectiveness value — the model should not conclude the matchup is neutral from a
Seismic Toss. The HP delta still reflects the damage taken.

Verified: `effectiveness_test.py::test_fixed_damage_move_no_effectiveness`

### 2. Quad-SE and quad-resist collapse to 2.0 and 0.5

Showdown emits exactly one `|-supereffective|` message regardless of whether the
matchup is 2× (e.g. Water vs Fire) or 4× (e.g. Water vs Fire/Rock Tyranitar). Same
for `|-resisted|` on 0.5× vs 0.25×. We store 2.0/0.5; the encoder bins ≥2.0 as SE
and ≤0.5 as resisted. Single and quad collapse to the same category.

This is the correct design given the protocol — we faithfully represent what Showdown
tells us. The HP delta is the signal that distinguishes 2× from 4× damage in practice.

### 3. Multi-hit moves: last write wins, consistent

Each hit of a multi-hit move (Double Kick, Fury Attack, Bullet Seed, etc.) generates
its own `|-supereffective|`/`|-resisted|` message. All hits are the same type, so all
generate the same multiplier. The last write wins, and the result is consistent.
No special handling needed.

Verified: `effectiveness_test.py::test_multihit_supereffective_last_write_wins`

### 4. Missed moves: tentative 1.0 persists

When a damaging move misses (`|-miss|`), the tentative 1.0 set in the `|move|` handler
is NOT cleared. The model sees 1.0 even though no damage was dealt. This is
intentional: the type interaction exists regardless of whether the move connected. The
miss is already captured by `our_hp_delta` being zero. We do not want a separate
"missed" bucket in the effectiveness one-hot — that would conflate two unrelated pieces
of information.

### 5. Sleep Talk fires two `|move|` messages from the same side

When Sleep Talk is used, Showdown sends:
```
|move|p1a: Snorlax|Sleep Talk|p2a: ...
|move|p1a: Snorlax|Body Slam|p2a: ...   ← the randomly chosen move
```

Both entries are p1. Without deduplication, `_this_turn_move_sides` would become
`['ours', 'ours']`, triggering `len == 2` and setting `we_moved_first = True` even
though the opponent never moved. The fix: the `if move_side not in self._this_turn_move_sides`
guard in the `|move|` handler prevents double-counting the same side.

With deduplication, `_this_turn_move_sides = ['ours']` (length 1), so `_we_moved_first`
is not updated. The property returns None (previous stale value fails the turn gate),
which is correct — only one side used a move.

The fuzz test's raw interpreter uses the same deduplication logic. Without it, the
Layer 1 comparison would see 17 false mismatches per 50-battle scenario C run.

Other moves that generate multiple `|move|` messages for the same mon: none confirmed
in Gen 3, but the guard is generic enough to handle any future case.

### 6. Status moves can fire `-immune` with `basePower = 0`

Thunder Wave against a Ground-type:
```
|move|p1a: Jolteon|Thunder Wave|p2a: Steelix
|-immune|p2a: Steelix
```

Thunder Wave has `basePower: 0`, so the tentative-1.0 guard does NOT fire in the
`|move|` handler. But the `-immune` handler DOES fire and sets
`_our_last_effectiveness = (turn, 0.0)`. This is intentional: the move had no effect;
0.0 is the right signal. The encoder bins it as immune.

The semantic difference (immune to a damaging move vs. immune to a status move) is
not distinguished in the one-hot. The model can correlate with the move_id block
(Thunder Wave has `basePower=0`, so power_norm=0) to learn the distinction if it
matters.

### 7. Thick Fat: ability-based damage reduction, NO effectiveness message

When Snorlax (Thick Fat) is hit by Fire or Ice moves, Showdown applies 0.5× damage
reduction silently. No `|-resisted|` message fires. Our tracker records tentative 1.0
(Normal type, neutral type chart result). The actual HP delta reflects the reduced
damage, but the effectiveness one-hot shows `normal`, not `resisted`.

This is a **known limitation, by design**. We track type effectiveness, not
ability-modified damage. The model sees:
- `opp_effectiveness = 1.0` (normal type)
- `opp_hp_delta` = less damage than a full 1× hit

It must learn from the HP delta discrepancy that Thick Fat is at play. This is
acceptable — the type chart itself is neutral; the ability is a separate modifier.

Abilities that work this way in Gen 3: Thick Fat (Snorlax, Piloswine, Marill line).
Future work: Flash Fire *does* fire `|-immune|` correctly (tested), but Thick Fat does
not — these are not symmetric.

### 8. `MESSAGES_TO_IGNORE` has two independent filter levels

`Player._handle_battle_message()` has its own ignore set: `{"t:", "expire",
"uhtmlchange"}`. `AbstractBattle.parse_message()` has the larger set. These are
independent — a message filtered at the Player level never reaches `parse_message()`.

Crucially, `win`, `tie`, `request`, `showteam`, and `error` are handled at the Player
level and **never forwarded** to `parse_message()`. Calling `parse_message(['', 'win',
'...'])` directly in tests (e.g. from a replay HTML) hits the catch-all
`else: raise NotImplementedError`. The replay test pre-filters these with:

```python
_PLAYER_LEVEL = {"win", "tie", "request", "showteam", "error", "uhtml", "html"}
for line in turn_lines:
    parts = line.split("|")
    if parts[1] in _PLAYER_LEVEL:
        continue
    b.parse_message(parts)
```

Any future test that feeds raw protocol lines into `Battle.parse_message()` directly
must apply the same filter.

### 9. Mid-turn forced-switch timing: effectiveness unavailable

When our Pokémon faints mid-turn, poke-env calls `choose_move()` for the forced
replacement BEFORE `|turn|N+1` fires. At that point:
- `battle._turn = N` (set when `|turn|N` fired at the start of this turn)
- This turn's move events have set `_opp_last_effectiveness = (N, ...)` (tentative or
  explicit, from the move that just KO'd us)
- The property checks `turn_set == self._turn - 1 == N-1`
- `(N, ...) != N-1` → property returns None

So on forced-switch decisions, `our/opp_last_effectiveness` and `we_moved_first` are
all None. **This is correct for training**: in the RL environment (`gen3_env.py`),
`embed_battle()` is always called after a full turn (after `|turn|N+1` fires), not
mid-turn. The BattleContext snapshot that the model sees for a forced-switch decision
correctly shows None for these fields — the model learns that forced-switch turns have
no effectiveness context.

The fuzz test skips Layer 1 validation on `battle.force_switch = True` turns for this
reason.

### 10. `BattleContext.from_battle()` requires persistent `SlotRegistry` instances

```python
def from_battle(cls, battle, mask, obs, our_slots: SlotRegistry, opp_slots: SlotRegistry)
```

`SlotRegistry` assigns stable integer slot indices to Pokémon as they are revealed
across a battle. It must **persist for the full battle lifetime** — creating a fresh
one each turn would re-assign slots and break delta calculations. The fuzz test stores
registries in `self._our_slots[battle_tag]` / `self._opp_slots[battle_tag]`,
initialised on first encounter and discarded on battle end. Any future code that calls
`from_battle()` outside of `gen3_env.py` must follow the same pattern.

### 11. `MESSAGES_TO_IGNORE` two-level filter in replay tests

`effectiveness_replay_test.py` extracts raw `|`-prefixed lines from replay HTML and
feeds them into `Battle.parse_message()`. The `|win|`, `|tie|`, `|request|`,
`|showteam|` messages must be filtered — they raise `NotImplementedError` because they
are consumed at the Player level in live battles. See edge case 8 for the filter code.

### 12. Replay: tentative-1.0 vs raw-interpreter disagreement is not a bug

The replay test's independent raw interpreter returns `None` for turns where no
explicit effectiveness message fired (normal-effectiveness moves). poke-env returns
`1.0` (tentative). The test only validates turns where the raw interpreter has an
explicit signal (`expected_our is not None or expected_opp is not None`). Coverage of
the 1.0 path is confirmed separately by using `actual` values (which include
tentative-1.0) in the coverage counter rather than `expected` values.

### 13. `_find_repo_root()` walk-up for replay file discovery

The replay test locates replay files relative to the repo root by walking up the
directory tree until finding a directory that contains a `models/` subdirectory. A
fixed relative path (`../../../../../`) silently produces the wrong result if the test
file moves or if a worktree is nested at a different depth. The walk-up is robust
to both.

### 14. Gen 3-specific type interactions (all verified)

The following Gen 3 type chart interactions were explicitly tested in
`test_known_matchups()` in `effectiveness_test.py`:

| Matchup | Result | Notes |
|---|---|---|
| Normal → Ghost | 0.0 immune | e.g. Tackle vs Gengar |
| Ghost → Normal | 0.0 immune | e.g. Shadow Ball vs Blissey — Gen 3 type chart |
| Psychic → Dark | 0.0 immune | Dark types are immune to Psychic — Gen 3 |
| Ground → Flying | 0.0 immune | e.g. Earthquake vs Zapdos |
| Electric → Water/Flying | 2.0 SE | e.g. Thunderbolt vs Gyarados |
| Water → Fire/Rock | 2.0 SE | e.g. Surf vs Charizard |
| Steel → Steel | 0.5 resisted | e.g. Meteor Mash vs Skarmory |
| Ghost → Steel | 0.5 resisted | Gen 3 specific — Steel's Ghost resistance removed in Gen 6 |
| Dark → Steel | 0.5 resisted | Gen 3 specific — Steel's Dark resistance removed in Gen 6 |
| Fire → Flash Fire | 0.0 immune | Ability-based; same protocol message as type immunity |
| Fire → Neutral Normal | 1.0 tentative | No explicit message; tentative-1.0 logic |

The two Steel resistances (Ghost/Dark) are **Gen 3 specific** and were removed in
Gen 6. Any future work that imports type-chart logic from a generic source must use
the Gen 3 chart, not the current one.

---

## Test Suite

Three independent test layers verify correctness at every level of the pipeline:

| File | Type | Marker | What it validates |
|---|---|---|---|
| `src/poke_env/battle/effectiveness_test.py` | Unit | none | 25 deterministic tests: all 6 (our/opp)×(SE/resisted/immune) combos, fixed-damage, status, move order (4 cases), staleness, multi-hit, tentative override, p2 role, OHKO format, ability-reveal format, both-sides-independent, Gen 3 type chart regressions |
| `src/poke_env/battle/effectiveness_replay_test.py` | Unit (replay) | none | Parses real gen3ou HTML replays; validates poke-env agrees with independent raw interpretation on all turns with explicit effectiveness messages; coverage check confirms all four categories appear across the corpus |
| `src/agents/training/poke_env_gaps/effectiveness_fuzz_e2e_test.py` | E2E | e2e | 4-layer pipeline validation across 3 targeted scenarios; 50 battles each; requires live Showdown server (`npm run showdown`) |

### Fuzz test: 4 validation layers

```
Raw Showdown messages (intercepted in _handle_battle_message override)
    ↓ Layer 1: poke-env parser correctness
battle.our/opp_last_effectiveness, battle.we_moved_first
    ↓ Layer 2: BattleContext snapshot correctness
curr_ctx.our/opp_last_effectiveness, curr_ctx.we_moved_first
    ↓ Layer 3: TurnDelta.build() correctness
delta.our/opp_effectiveness, delta.we_moved_first
    ↓ Layer 4: TurnDeltaEncoder correctness
encoded_vec[29:33], encoded_vec[33:37], encoded_vec[37:39]
```

### Fuzz test scenarios

| Scenario | Key teams | What it exercises |
|---|---|---|
| A — Immunity | Gengar/Flygon (Levitate), Jolteon (Volt Absorb), Vaporeon (Water Absorb) vs diverse attackers | All four effectiveness categories including multiple immunity types |
| B — SE/Resisted | Charizard/Blastoise/Jolteon/Steelix/Heracross/Skarmory | Fire/Water/Electric/Ground/Fighting/Rock matchups; type-based SE and resisted from both sides |
| C — Priority + speed | Arcanine (ExtremeSpeed +2), Jolteon (Quick Attack +1), Snorlax (Sleep Talk), Blissey, Forretress | we_moved_first correctness for priority users; Sleep Talk deduplication edge case; switch-turn None coverage |

### Fuzz test results (50 battles per scenario)

```
Total decision turns: 18,014
Layer 1 mismatches: 0   (raw messages vs poke-env properties)
Layer 2 mismatches: 0   (poke-env properties vs BattleContext)
Layer 3 mismatches: 0   (BattleContext vs TurnDelta)
Layer 4 mismatches: 0   (TurnDelta vs encoded vector)

Coverage:
  immune:     818  resisted:  2,000  normal: 10,992  SE:    922
  we_first: 3,966  opp_first: 3,394  na:  10,654
```

### Layer 1 validation design decisions

Layer 1 compares the raw interceptor's interpretation against poke-env's properties.
Two categories of turns are intentionally NOT validated:

1. **Tentative-1.0 turns** (`expected=None, actual=1.0`): the raw interceptor returns
   None when no explicit effectiveness message fires; poke-env returns 1.0 (tentative
   for normal-effectiveness damaging moves). This is by design. Flagging it as a
   mismatch would produce ~half of all turns as false failures.

2. **Forced-switch turns** (`battle.force_switch=True`): mid-turn decision point where
   the effectiveness properties are unavailable (see edge case 9). Skipped in
   validation; documented in fuzz test code.

Layer 1 only flags: raw has explicit signal (SE/resisted/immune fired) AND poke-env
disagrees. This is the definition of a real parsing bug.

---

## Observation Dimension

Full observation: **1141 dims** (unchanged total; the +10 dims from this step were
already reflected in the CLAUDE.md update made earlier in Step 7 planning).

| Block | Dims | Notes |
|---|---|---|
| Our team (6 × 61) | 366 | Unchanged |
| Opp team (6 × 61) | 366 | Unchanged |
| Active context ×2 | 46 | Unchanged |
| Global env | 13 | Unchanged |
| Reactive + matchups | 300 | Unchanged |
| Prev-turn action mask | 11 | Unchanged |
| **TurnDelta block** | **39** | **Was 29; +10 in this step** |

Total: 1141.

---

## What's Next: Speed Inference (Step 8)

`we_moved_first` is the foundation for speed reasoning but is not sufficient alone.
The model gets a binary signal per turn but cannot directly read speed stats. The gap
between "what we have now" and "full speed model" is:

### What we have

- `we_moved_first` (True/False/None) in the TurnDelta block
- The move_id for both sides (dims 0 and 5 of TurnDelta) — the model can learn which
  moves have priority from the move embedding
- Per-Pokémon HP fraction and condition bits (paralysis in condition one-hot)

### What a speed inference step would add

Options, roughly in order of implementation complexity:

**Option A — Speed tier one-hot per active Pokémon**
Derive approximate speed tier (very slow / slow / medium / fast / very fast) from
the base speed stat available in `gen3_species.json`. Append to the active-context
encoder or per-Pokémon vector. This gives the model an absolute reference point
rather than purely relative (who-moved-first).

Caveat: EVs and nature modify actual speed by up to ±10%, so tiers will overlap.
The model sees EVs nowhere — this is a known blind spot.

**Option B — Inferred speed comparison from we_moved_first + priority**

Add a 3-way signal per turn: [we_faster, opp_faster, unknown].

Derivation:
1. Look up both sides' move priorities from `gen3_moves.json`
2. If priorities differ → unknown (priority-determined, not speed-determined)
3. If priorities equal AND we_moved_first=True → we_faster=1
4. If priorities equal AND we_moved_first=False → opp_faster=1
5. If we_moved_first=None → unknown

This is a deterministic inference from signals already available. It would give the
model a direct speed-comparison bit rather than requiring it to learn the priority
system implicitly.

Caveat: speed ties (equal speed) are randomly resolved and would appear as noisy
we_faster/opp_faster signals.

**Option C — Explicit speed score per active Pokémon**
Look up base speed from `gen3_species.json` and normalise to [0, 1]. Inject into the
active-context encoder alongside the boost vector. The model then sees absolute speed
context for both active mons.

This is the most informative but requires adding a field to `ActiveContextEncoder` and
rebuilding the projection dimension.

### Priority system in Gen 3 (reference)

Priorities that appear in gen3ou:

| Move | Priority |
|---|---|
| Helping Hand | +5 |
| Protect, Detect, Endure | +3 |
| ExtremeSpeed | +2 |
| Quick Attack, Fake Out, Mach Punch, Bullet Punch | +1 |
| All normal moves | 0 |
| Vital Throw | −1 |
| Roar, Whirlwind | −6 |

Note: Counter and Mirror Coat have −1 priority in Gen 3 (changed in Gen 4).
Note: Trick Room does NOT exist in Gen 3. Speed order is always faster-first within
a priority tier.

### Interaction with paralysis

Paralysis reduces speed to 25% in Gen 3 (not 50% as in Gen 4+). A Jolteon
(base 130 Spe) with full paralysis has effective speed ~32, slower than Blissey
(base 55 Spe). The condition one-hot already tracks paralysis. Speed inference logic
must account for this when building a speed-comparison signal.

### What NOT to do

Do not add a "did we use a priority move this turn" feature separately — the move_id
embedding already encodes the move, and priority is a property of that move. Adding
a redundant priority bit would just be a second encoding of information already
present.
