# Implementation: Step 3 — Foundational Damaging-Event Attribution

This step moves "who fired, who got hit, what effectiveness" out of inference-from-
snapshot-diffs and into a first-class structure produced by the protocol parser.
A new `DamagingMoveEvent` is built at `|move|` parse time and finalized when the
matching `|-supereffective|` / `|-resisted|` / `|-immune|` / Flash Fire `|-start|`
fires. `BattleContext` snapshots it; the Hidden Power tracker reads it directly.
The non-BP target resolver, the multi-faint disambiguation, the `prev`-snapshot
lookup, and the Roar/phazing workarounds all delete — total HP-attribution code
shrinks from ~70 lines to ~5. A silent gap in Flash Fire effectiveness reporting
was found and fixed along the way.

A follow-on pass then **threads the protocol-truth data into the model's
observation**: the per-turn `TurnDelta` slot in the turn-history block grows from
39 dims → 88 dims, picking up actor/target/switch_to species IDs (routed through
the existing `species_embedding` table), boost deltas, a forced-switch phase
flag, per-side target HP deltas, full per-slot HP-level vectors, and target-
status onehots at move-fire time. `ARCH_SIGNATURE` bumps to `gen3_unified_v2`.
After this pass the model can finally learn cross-turn inferences like "Hidden
Power was SE against the mon we switched in 3 turns ago" — the original
motivation for step 3's protocol-truth attribution.

Primary themes: protocol-truth over snapshot-diff inference, the pending → promote
pattern that distinguishes "tentative neutral" from "confirmed effectiveness", an
intent-correct `our_switch_to` semantic that survives force-replacement chains,
a fuzz validator restructured around protocol cross-checks, and a "this-side-
this-turn" convention for the model-facing slot that keeps `our_*` fields
semantically consistent across every column.

---

## Motivation

### The gap

Step 2 built a `_resolve_hp_target` resolver that reconstructed the HP target from
before/after `BattleContext` snapshots: voluntary switch, Baton Pass, "prev_active
fainted from HP", and "switch-in fainted" each had their own branch keyed off
`delta.our_switch_to`, `delta.our_move_id`, and `newly_fainted` set arithmetic.
The Step 2 fuzz duplicated the same logic in `HiddenPowerTrackerFuzzPlayer._resolve_target`
to keep the test independent.

Two production crashes during smoke testing exposed the design's limits:

1. **Multi-faint chain misattribution.** Voluntary switch (Arcanine → Swampert)
   triggers Spikes, Swampert dies to HP, the force-replacement chain (Lanturn →
   Arcanine → Snorlax) burns through multiple Spikes-vulnerable mons before turn
   N+1. Case A (`prev.our_active in newly_fainted`) fires because Arcanine ended
   up dead — but the actual HP target was Swampert. `delta.our_switch_to` was
   computed from `curr_ctx.our_active` (the end-of-chain Snorlax), not from the
   slot we actually picked, so the resolver couldn't recover the right target.

2. **Stale-last_move with a different firer.** Opp's Celebi voluntary-switches
   to Forretress (no damaging move). Forretress's `last_move` is a stale
   `"hiddenpower"` from an earlier turn. At the next snapshot, `ctx.opp_last_move_id
   == "hiddenpower"` (Forretress's stale value) and `opp_last_effectiveness` is
   set from the just-ended turn — but the just-ended turn's damaging move was
   Celebi's *non-HP* attack. The HP gate passes wrongly, the effectiveness
   reflects the wrong move, and the observation gets attributed to whichever opp
   mon was active at the previous move-selection.

Both failures share a root cause: **the protocol stated user / target / move
explicitly, but we threw that information away and tried to infer it from
state.** Each new edge case (Roar phazing, BP, force-replacement chains,
|cant| under spikes, stale last_move on the new active) needed another inference
branch. The resolver had become an inference engine pretending to be a fact.

### What the protocol actually tells us

A Showdown `|move|p2a: Zapdos|Hidden Power|p1a: Weezing|` line, followed by
`|-supereffective|p1a: Weezing|` for HP Grass on a Water/Ground target, fully
specifies the firer (`Zapdos`), the target (`Weezing`), the move (`Hidden Power`),
and the effectiveness (`2.0`). poke-env's parser already reads all four —
`_opp_last_effectiveness` is set in the existing `-supereffective`/`-resisted`/
`-immune` handlers, but user / target / move identity were discarded. Fixing
attribution by extending poke-env to *keep* what it parses is a smaller change
than re-deriving it.

### Why a pending → promote design (and not tentative-fill)

A first attempt set the `DamagingMoveEvent` to a tentative `effectiveness=1.0`
at `|move|` time, mirroring how `_opp_last_effectiveness` defaults to neutral.
The fuzz caught the problem inside 100 battles: **Thick Fat silently halves
Fire and Ice damage** — Showdown applies the multiplier in `onBasePower` and
emits no effectiveness message, so the tentative 1.0 leaks through as the
attributed effectiveness, even though the true bucket is 0.5×. The HP tracker
then narrows on a wrong observation.

Pending → promote sidesteps this: the partial event waits in
`_pending_*_damaging_move` until an explicit effectiveness emission lands,
and only then is promoted into `_*_last_damaging_move`. Silent damage reducers
leave the property at `None`, which the HP tracker treats as "no observation"
— the same skip-on-uncertainty behavior the old code had via
`opp_last_effectiveness is None`. Scalar `opp_last_effectiveness` keeps its
legacy tentative-1.0 semantics for non-HP callers (basePower>0 only) so
existing reward/recorder code is unaffected.

### Why Flash Fire needed its own branch

Flash Fire is the lone ability immunity that uses `|-start|` instead of `|-immune|`:
Showdown packages "0× damage" and "Fire boost activated" as a single
`|-start|p1a: Arcanine|ability: Flash Fire|` line. The pre-existing `-immune`
handler missed this entirely — every Flash Fire absorption was treated as
unknown-effectiveness (skipped) by the HP tracker, even though the protocol
told us exactly what happened. Wiring `-start ability: Flash Fire` into
`_set_effectiveness` recovers ~45 observations per 100 battles in the new fuzz
(immunity-trigger counter).

---

## What Changed

### New type: `DamagingMoveEvent`

`poke_env.battle.abstract_battle.DamagingMoveEvent` — a `NamedTuple` exposing
the protocol facts:

| Field | Type | Source |
|-------|------|--------|
| `user_species` | `str` | `|move|<user-side>: <UserName>|` |
| `target_species` | `str` | `|move|...|<move>|<target-side>: <TargName>|` |
| `target_status` | `Optional[Status]` | sampled at `|move|` time, BEFORE the move resolves |
| `move_id` | `str` | `to_id_str(<move>)` — `"hiddenpower"`, not `"hiddenpowerice"` |
| `effectiveness` | `float` | filled in when an effectiveness emission fires |

`target_status` is sampled at `|move|` parse time (pre-resolution) so the Gen 3
Fire-thaws-Frozen quirk stays observable: a frozen Flash Fire holder hit by
HP Fire takes 0.5× damage AND thaws AND often faints from the same hit. By
the next snapshot, live status is `None` (thawed) or `FNT` (dead) — neither
the FRZ that mattered.

### Two new turn-gated properties on `AbstractBattle`

```python
@property
def our_last_damaging_move(self) -> Optional[DamagingMoveEvent]:
    """Our last damaging move resolved last turn. Set only when an explicit
    effectiveness emission fires — Thick Fat-style silent halving leaves
    this None (use opp_last_effectiveness for tentative-neutral signal)."""

@property
def opp_last_damaging_move(self) -> Optional[DamagingMoveEvent]:
    """Mirror of the above for the opponent."""
```

Both gate on `turn_set == self._turn - 1` exactly like `our/opp_last_effectiveness`,
so a stale event from earlier in the battle can never leak.

### Pending → promote mechanic

Three new private slots on `AbstractBattle`:

| Slot | Set at | Promoted at | Read via |
|------|--------|-------------|----------|
| `_pending_our_damaging_move` | `|move|` (category Physical/Special) | `_set_effectiveness` (defender_side != us) | — |
| `_pending_opp_damaging_move` | `|move|` (category Physical/Special) | `_set_effectiveness` (defender_side == us) | — |
| `_our_last_damaging_move` | promotion only | — | `our_last_damaging_move` property |
| `_opp_last_damaging_move` | promotion only | — | `opp_last_damaging_move` property |

`_set_effectiveness(defender_side, mult)` is a new helper that DRYs the previously
duplicated `-supereffective` / `-resisted` / `-immune` handlers and also runs
the pending-event promotion. It's the single sink for "effectiveness just got
confirmed for this turn".

### Flash Fire wired into the effectiveness pipeline

`-start` handler now matches `effect == "ability: Flash Fire"` and routes
through `_set_effectiveness(defender_side, 0.0)`. The defender side is parsed
from `event[2][:2]` exactly like the existing handlers.

### `our_switch_to` reflects intent, not end-state

`TurnDelta.build` previously computed `our_switch_to` from `curr_ctx.our_active`.
When a switch-in died and a force-replacement chain cycled several more mons
through the field, `curr_ctx.our_active` ended up being the final replacement —
not the species we actually picked. The HP target resolver fell back to
`newly_fainted` set arithmetic to recover the right mon, but multi-faint
chains broke that disambiguation.

`BattleContext.our_team_order: tuple[str, ...]` now snapshots
`tuple(m.species for m in battle.team.values())` at every turn —
the exact iteration order the action mask was built against. `TurnDelta.build`
for action 0-5 looks up `prev_ctx.our_team_order[action]` to get the species
we *picked*. End-of-chain replacements stop polluting the field. This is an
independent improvement over Step 2 even though HP attribution no longer
needs it; reward/recorder callers (`battle_recorder.py`, `reward_manager.py`)
also benefit from the corrected semantic.

### HP attribution: ~70 lines deleted, ~5 lines added

The non-BP target resolver and all its branches:

```diff
-def _resolve_hp_target(battle, prev, curr, delta):
-    voluntary_switch = delta.our_switch_to is not None
-    visible_side_change = prev.our_active != curr.our_active
-    baton_pass = delta.our_move_id == BATON_PASS and visible_side_change
-    if voluntary_switch:
-        target_species = delta.our_switch_to
-    elif baton_pass and delta.we_moved_first is True:
-        newly_fainted = curr.our_fainted_species - prev.our_fainted_species
-        switch_in_fainted = newly_fainted - {prev.our_active}
-        if not switch_in_fainted:
-            target_species = curr.our_active
-        elif len(switch_in_fainted) == 1:
-            target_species = next(iter(switch_in_fainted))
-        else:
-            raise RuntimeError("BP target ambiguous: ...")
-    else:
-        target_species = prev.our_active
-    return _snapshot_target(battle, target_species, prev)
```

…is replaced by a direct read:

```python
def _maybe_observe_hidden_power(self, battle, ctx):
    if ctx.phase != "move_selection":
        return
    event = ctx.opp_last_damaging_event
    if event is None or event.move_id != "hiddenpower":
        return
    target = _wrap_hp_target(battle, event)
    if target is None:
        return
    self._hidden_power_tracker.observe(
        event.user_species, event.effectiveness, target
    )
```

`_wrap_hp_target` is a small adapter that looks up the live mon by species and
overlays `event.target_status`. `_resolve_hp_target`, `_snapshot_target`,
`_find_prev_move_selection_index`, the `BATON_PASS` constant, the entire
non-trivial `_HpTargetMon` resolution path, and the `TurnDelta` import in
`episode_tracker.py` all delete.

### Fuzz validator: independent resolver replaced by protocol cross-check

The Step 2 fuzz validator duplicated `_resolve_hp_target` so a bug in either
piece wouldn't slip through. With the resolver gone from production, that
parallel logic has no counterpart. The fuzz now validates three things:

| Layer | What it checks |
|---|---|
| **Per-observation invariant** | After every `observe()`, every surviving candidate type satisfies `effective_multiplier(type, target) == observed_effectiveness` |
| **End-of-battle ground truth** | The true HP type for every opp species that used HP must still be a non-zero candidate |
| **Protocol cross-check** | Every `DamagingMoveEvent` is corroborated by a `|move|...|Hidden Power|<target>|` line in the same turn's archived protocol log |

The protocol cross-check is new — it's the test that would catch a regression
in poke-env's protocol parser (e.g. a future refactor that broke the `pending →
promote` link).

The fuzz no longer tracks `_prev_our_active`, `_prev_opp_active`,
`_prev_our_fainted`, `_prev_action_was_switch`, `_prev_switch_to`,
`_prev_move_id`, or `_prev_our_team_status`. The full `_resolve_target` /
`_snapshot_target` methods delete. `_is_switch_order` deletes. The turn_log
diagnostic dump deletes. Net: 525 → 326 lines (~38% reduction).

---

## Implementation Details

### `_set_effectiveness` (`src/poke_env/battle/abstract_battle.py`)

```python
def _set_effectiveness(self, defender_side: str, mult: float) -> None:
    if defender_side == self._player_role:
        self._opp_last_effectiveness = (self._turn, mult)
        pending = self._pending_opp_damaging_move
        if pending is not None and pending[0] == self._turn:
            self._opp_last_damaging_move = (
                self._turn,
                pending[1]._replace(effectiveness=mult),
            )
    else:
        self._our_last_effectiveness = (self._turn, mult)
        pending = self._pending_our_damaging_move
        if pending is not None and pending[0] == self._turn:
            self._our_last_damaging_move = (
                self._turn,
                pending[1]._replace(effectiveness=mult),
            )
```

The four call sites — `-supereffective`, `-resisted`, `-immune`, and the new
Flash Fire `-start` branch — now collapse to single-line dispatches:

```python
elif event[1] == "-supereffective":
    if len(event) >= 3:
        self._set_effectiveness(event[2][:2], 2.0)
elif event[1] == "-resisted":
    if len(event) >= 3:
        self._set_effectiveness(event[2][:2], 0.5)
elif event[1] == "-immune":
    if len(event) >= 3:
        self._set_effectiveness(event[2][:2], 0.0)
    ...
elif split_message[1] == "-start":
    pokemon, effect = event[2:4]
    ...
    if effect == "ability: Flash Fire":
        self._set_effectiveness(pokemon[:2], 0.0)
```

### Pending capture at `|move|`

```python
# --- pending damaging-move event ---
# Captured at |move| time for ANY damaging move (Physical/Special category —
# includes HP and other callback-power moves). The target's status is sampled
# BEFORE the move resolves (Flash Fire-vs-frozen). Promoted into
# _*_last_damaging_move only when an effectiveness emission lands in the same
# turn — silently expires otherwise.
if _move_entry.get("category") in ("Physical", "Special"):
    user_species = self.get_pokemon(pokemon).species
    target_mon = (
        self.get_pokemon(presumed_target)
        if presumed_target and presumed_target not in ("", None)
        else None
    )
    target_species = target_mon.species if target_mon else user_species
    target_status = target_mon.status if target_mon else None
    partial_event = DamagingMoveEvent(
        user_species=user_species,
        target_species=target_species,
        target_status=target_status,
        move_id=move_id_str,
        effectiveness=1.0,  # placeholder, overwritten on promotion
    )
    if move_side == "ours":
        self._pending_our_damaging_move = (self._turn, partial_event)
    else:
        self._pending_opp_damaging_move = (self._turn, partial_event)
```

Gating on `category` rather than `basePower > 0` matters because Gen 3's
`hiddenpower` entry has `basePower=0` (resolved at runtime via callback) —
the old gate would never have captured HP at all. Other callback-power moves
(Magnitude, Reversal, Flail, Dragon Rage) get the same correct treatment for
free.

### `BattleContext` extensions

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `our_team_order` | `tuple[str, ...]` | `()` | `battle.team.values()` iteration order at snapshot — intent-correct switch attribution |
| `our_last_damaging_event` | `Optional[DamagingMoveEvent]` | `None` | Mirrors `battle.our_last_damaging_move` |
| `opp_last_damaging_event` | `Optional[DamagingMoveEvent]` | `None` | Mirrors `battle.opp_last_damaging_move` |

All three sit at the bottom of the dataclass with defaults so the existing
non-defaulted field order stays intact (`@dataclass` requires defaults after
non-defaults).

### `_wrap_hp_target` adapter

```python
def _wrap_hp_target(battle, event: DamagingMoveEvent) -> Optional[_HpTargetMon]:
    """Look up the target's current type/ability from battle.team and overlay
    the status captured at the moment HP fired."""
    live_mon = _find_mon(battle, event.target_species)
    if live_mon is None:
        return None
    return _HpTargetMon(
        species=live_mon.species,
        type_1=live_mon.type_1,
        type_2=live_mon.type_2,
        ability=live_mon.ability,
        status=event.target_status,
    )
```

Type / ability come from the live mon (these don't change mid-battle in Gen 3 — no
Skill Swap shenanigans relevant to HP). Status comes from the event's snapshot,
which is the only field that can drift before observation time.

### Fuzz protocol cross-check

```python
def _protocol_says_hp(self, tag, turn, user_species, target_species):
    """Confirm the DamagingMoveEvent matches a |move|...|Hidden Power|...|
    line in the named turn. Walks back to |turn|N|, then forward to the next
    |turn|N+1| marker, scanning |move| lines."""
    plog = self._proto_log.get(tag, [])
    turn_str = str(turn)
    for i, sm in enumerate(plog):
        if len(sm) >= 3 and sm[1] == "turn" and sm[2] == turn_str:
            for sm2 in plog[i + 1:]:
                if len(sm2) >= 3 and sm2[1] == "turn":
                    return False
                if len(sm2) >= 5 and sm2[1] == "move":
                    if sm2[3].lower().replace(" ", "") == "hiddenpower":
                        mover = sm2[2].split(": ", 1)[-1].lower()
                        target = sm2[4].split(": ", 1)[-1].lower()
                        if mover == user_species and target == target_species:
                            return True
            return False
    return False
```

Fails the fuzz with `os._exit(1)` and a 60-line protocol dump if the event
disagrees with the raw lines. This is the regression net for future poke-env
parser changes.

### Dedupe on consecutive choose_move calls

The same turn-gated event can surface on multiple consecutive choose_move calls
(force-switch sub-calls before the next move_selection). The fuzz dedupes via
`_observed_keys: dict[tag, set[(turn, user_species, target_species)]]` — observe
once per distinct event. Production doesn't need this because `record()` only
fires on `phase == "move_selection"`.

---

## Edge Cases

### Thick Fat silently halving Fire / Ice

No effectiveness emission → pending event expires → property returns `None` →
HP tracker skips. Matches Step 2's "skip when `opp_last_effectiveness is None`"
behavior. Loses the observation but never records a wrong one.

### Flash Fire absorbing Fire (non-frozen holder)

`-start ability: Flash Fire` routes through `_set_effectiveness(0.0)` → pending
event promotes with `effectiveness=0.0` → HP tracker correctly narrows to
"only Fire types" survive. New behavior — Step 2 silently dropped these.

### Flash Fire-vs-frozen quirk preserved

`target_status` is sampled at `|move|` time, before the Fire move thaws the
target. `_wrap_hp_target` passes `status=event.target_status` (= FRZ) to
`effective_multiplier`, which honors the Gen 3 "Flash Fire suppressed while
frozen" branch from Step 2. Without the snapshot, live status would already
be `None` and Fire would be wrongly eliminated.

### Opp forced out by our Roar / Whirlwind (HP fired this turn)

Opp HP resolves first (priority 0), then our Roar phazes. At the next snapshot,
opp_active is the phazed-in mon. Pending event was set when opp fired HP —
keyed by user_species, not by who's currently active — so the promotion still
works against the original firer. Step 2 had a known gap here (the resolver
read `ctx.opp_last_move_id` from the new active's stale last_move); this step
closes it because attribution is identity-explicit at capture time.

### Opp's HP user fainted at end of turn

Identical to the Roar case. The mon's `species` doesn't change when they faint,
and `_find_mon` resolves dead mons via `battle.team.values()` (which includes
fainted entries). HP attribution to the now-dead user is correct.

### Voluntary switch with multi-faint force-replacement chain

The case that crashed Step 2 in production. `delta.our_switch_to` is now derived
from `prev.our_team_order[action]` (intent-correct), but more importantly:
**the HP attribution path doesn't touch `delta` at all**. The event's
`target_species` came straight from the `|move|` line — switch-in dies, chain
cycles, snapshot inconsistency, none of it matters.

### Same-name mons (mirror match)

Showdown identifiers like `p1a: Salamence` and `p2a: Salamence` both resolve
to species `salamence` via `get_pokemon().species`. The side prefix (`p1a:`
vs `p2a:`) determines which side's pending slot receives the event — no
ambiguity.

### Forced-switch ctx skipping

`_maybe_observe_hidden_power` early-returns on `phase != "move_selection"`,
unchanged from Step 2. A forced-switch context's `opp_last_damaging_event` is
turn-gated and would be `None` anyway in most cases — but the gate stays
defensive against future changes to when force-switch ctxs are taken.

### Baton Pass (BP)

BP is no longer special-cased in production. If opp HP fires and a BP-recipient
ends up being the actual target, the `|move|...|Hidden Power|p1a: <Recipient>|`
line names them directly. The pending event captures the recipient at parse
time; promotion at `-resisted` / etc. preserves the identity. No timing
heuristics, no `we_moved_first` branching. (BP is rare and the protocol's
target identity is authoritative regardless of timing.)

---

## Test Suite

### `BattleContext` unit tests (`battle_context_test.py`)

| Test | What changed |
|------|--------------|
| `test_turn_delta_build_switch_action` | Now sets `our_team_order` on prev so action 1 maps to `"skarmory"` by intent |
| `test_turn_delta_build_switch_action_uses_intent_not_end_state` | **New** — multi-faint regression test (action 3 → Swampert even when curr.our_active is Snorlax) |
| `test_turn_delta_boost_delta_zeroed_on_our_switch` | `our_team_order` added so action 1 resolves correctly |

### `BattleRecorder` test (`battle_recorder_test.py`)

`_battle` helper now sets `b.our_last_damaging_move = None` and
`b.opp_last_damaging_move = None` so `BattleContext.from_battle` can read the
new attributes off the namespace stub.

### HP tracker fuzz (`hidden_power_tracker_fuzz_e2e_test.py`)

Stress runs:

| Battles | HP observations | Invariant pass | Ground truth pass | Notes |
|---|---|---|---|---|
| 100 | 969 | 969 / 969 | 349 / 349 | Includes 45 Flash Fire 0× triggers |
| 500 × 3 | ~4,500 each | 100% | 100% | Forretress (HP Bug) consistently 49–61 obs per run |
| 2,000 | 16,146 | 100% | 6,026 / 6,026 | Coverage breakdown unchanged from Step 2 |

Production smoke (`train_rl_agent.py --debug --steps 10000`): exit 0, training
completes, 70% vs Random / 9-15% vs heuristic bots in eval (expected for 10K
steps), zero tracker errors across the run.

### Full unit suite

719 passing (+1 over Step 2 baseline from the new
`test_turn_delta_build_switch_action_uses_intent_not_end_state`). Nine
pre-existing `launcher_ui_test.py` failures unrelated to this work.

### TurnDelta history expansion — added tests

`turn_delta_encoder_test.py` grew from 11 to 22 tests covering the extended
slot:

| Test | What it validates |
|------|-------------------|
| `test_dimension` | `TURN_DELTA_DIM == 88` |
| `test_boost_deltas_encoded` | Raw 7-dim signed boost deltas pass through |
| `test_phase_forced_switch_flag` | 0/1 flag from `delta.phase_is_forced_switch` |
| `test_target_hp_delta_signed` / `_none_is_zero` | Signed scalar + None→0.0 coercion |
| `test_hp_levels_vector` | 6-dim per-side `hp_after` round-trips |
| `test_actor_species_falls_back_to_prev_active` | No event → `prev_active` species ID |
| `test_actor_species_prefers_damaging_event_user` | Event `user_species` wins (mirror match correctness) |
| `test_target_species_from_damaging_event` | "This-side-this-turn" convention: `our_target_species_id` ← `opp_damaging_event.target_species` |
| `test_switch_to_species_encoded` / `_zero_when_no_switch` | Sentinel handling for switch-to slot |
| `test_target_status_onehot_encoded` / `_zero_when_no_status` | 7-state status onehot with sentinel zero |
| `test_unknown_species_raises` | `ValueError` on non-sentinel species not in mapping |
| `test_sentinel_species_encode_to_zero` | `None`/`"NONE"`/`"NULL"` stay at id 0 without raising |
| `test_describe_vector_decodes_extended_fields` | Full round-trip on every new field with both sides' events |

`state_encoder_test.EXPECTED_OBS_DIM` 1924 → 2414. `episode_tracker_test`'s
encoder factory now passes the species mapping, and its `OUR_HP_DELTA_IDX`
is driven from `OFFSET_OUR_HP_DELTA_SUM` rather than a magic number.
`player_test._make_battle` mock fixed to stub `our_last_damaging_move` /
`opp_last_damaging_move` to `None` (latent test bug surfaced by the new
strict species lookup).

Post-expansion: **779 passing**, two-step `train_rl_agent.py --debug`
smoke (5000 + 10000 steps): exit 0, `[ModelVersion] Round-trip smoke test
PASSED`, evaluation completes against all five fixed bots.

---

## Exposure to the Model: TurnDelta History Expansion

The damaging event sat on `BattleContext` and was consumed by the reward
function and HP tracker, but the model's observation tail (the
`N_HISTORY_TURNS` × 39-dim turn-history block) didn't carry it. The history
block only encoded *what move ID fired* and *summed HP deltas* — actor,
target, switch_to identities, target status, and boost magnitudes were all
dropped at the encoder boundary. The model could see "Hidden Power was SE
last turn" but not "SE against the mon we just switched in."

This sub-step routes the protocol-truth attribution data directly into the
slot, expanding it from 39 → 88 dims.

### The "this-side-this-turn" convention

Every `our_*` field in the slot describes the mon ON our side this turn,
mirrored for `opp_*`:

| Field | Source | Meaning |
|-------|--------|---------|
| `our_actor_species_id` | `delta.our_damaging_event.user_species` ∪ `delta.our_prev_active` | The mon on our side that acted |
| `our_target_species_id` | `delta.opp_damaging_event.target_species` | The mon on our side that opp hit |
| `our_target_status` | `delta.opp_damaging_event.target_status` | Its status at move-fire time |
| `our_target_hp_delta` | `delta.our_hp_delta[slot_of(opp_event.target_species)]` | HP loss on the named target |
| `our_switch_to_species_id` | `delta.our_switch_to` | The mon we sent in |
| `our_boost_delta` | `delta.our_boost_delta` | Stat-stage change on our active |
| `our_hp_levels` (6) | `delta.our_hp_after` | End-of-turn HP for every team slot |

A semantic mismatch was caught and fixed in code review: an early
implementation had `our_target_species_id` sourced from `our_damaging_event`
(the mon we hit on opp's side), which would have silently miscoded training
data — the species ID and the hp_delta scalar at the same `our_target_*`
prefix described Pokémon on opposite sides. Now everything under the `our_*`
prefix is consistently "what happened to / on our side this turn."

### Slot layout (88 dims, indices 0–87)

Indices 0–38 are the legacy base block (unchanged positions for layout
stability — old encoders' assumptions about the move/type/cant/effectiveness
positions still hold). Indices 39–87 are the new extended block, grouped by
purpose:

| Offset | Dims | Field |
|--------|------|-------|
| 39 | 7 | `our_boost_delta` (BOOST_STATS order) |
| 46 | 7 | `opp_boost_delta` |
| 53 | 1 | `phase_is_forced_switch` |
| 54–55 | 2 | `our_target_hp_delta`, `opp_target_hp_delta` |
| 56 | 6 | `our_hp_levels` (end-of-turn HP for all team slots) |
| 62 | 6 | `opp_hp_levels` |
| 68 | 7 | `our_target_status_onehot` (BRN, FNT, FRZ, PAR, PSN, SLP, TOX) |
| 75 | 7 | `opp_target_status_onehot` |
| 82–87 | 6 | Six raw species IDs — our/opp × actor/target/switch_to |

Named offset constants `OFFSET_OUR_BOOST_DELTA` … `OFFSET_OPP_SWITCH_TO_SPEC`
are exported from `turn_delta_encoder.py` so external code (tests, debug
tooling) never magic-number-indexes into the slot. Two named constants for
the base block (`OFFSET_OUR_HP_DELTA_SUM`, `OFFSET_OPP_HP_DELTA_SUM`) cover
the two positions that are referenced from outside the encoder.

### Phase flag — what is "one slot of history"?

`EpisodeTracker.record()` appends a `BattleContext` *every time the agent is
asked for input*, not every Showdown turn. A real turn that includes a
faint produces two slots: `move_selection_N → forced_switch` (captures opp's
killing move + our faint) and `forced_switch → move_selection_{N+1}`
(captures our replacement choice). This is the natural human framing —
"they KO'd me, I picked a replacement" reads as two events — and the
actor/target binding for the killing move stays intact in one slot.

But the model needed a way to distinguish a half-turn replacement slot from
a full action-pair slot. `phase_is_forced_switch` (1 dim, sourced from
`BattleContext.phase`) flips the bit, preventing the model from reading "no
opp move this slot" as "opp voluntarily passed."

### Per-slot HP levels, not deltas

The slot stores `our_hp_after` / `opp_hp_after` (6 floats each, end-of-turn
HP for every team slot) rather than per-slot deltas. Levels strictly
dominate deltas at the same dim cost:

- Adjacent-position deltas are recoverable by attention subtracting consecutive slots
- Levels additionally give the model absolute reference points ("this mon has been at 5% for 3 turns and isn't dead → Recover pending or Sitrus held"), which deltas can't express

The single-scalar `our_hp_delta_sum` at offset 24 also stays in the base
block for backwards-compatible coarse signal — the level vector is the
richer alternative, not a replacement.

### Strict species lookup

`_species_id(species)` previously fell back to ID 0 for any unknown name.
That collided with the "no actor / target / switch_to / unrevealed"
sentinel (also ID 0), so a typo or a stale `gen3_species.json` would
silently train on the wrong embedding. It now:

- Returns 0 for `None` / `"NONE"` / `"NULL"` — the legitimate empty case
- **Raises `ValueError`** for any other species name not in the mapping

This matches `SpeciesEncoder.encode()`'s existing convention. A latent bug
in `player_test._make_battle` surfaced immediately — the mock didn't stub
`our_last_damaging_move` / `opp_last_damaging_move`, so `MagicMock`
auto-created a tree whose `.user_species` (another `MagicMock`) leaked into
the encoder. Previously masked by the silent fallback; now an explicit
`= None` in the mock.

### `_embed_delta_slot` extension

`Gen3FeaturesExtractor._embed_delta_slot` now threads `species_embedding`
alongside `move_embedding` and `type_embedding`. Each history slot's 4 raw
move/type IDs (positions 0, 4, 5, 9) plus 6 raw species IDs (contiguous at
slot tail, offsets 82–87) get looked up and concatenated with the 78
pass-through scalars:

```
_td_embed_dim = 2 * move_emb + 2 * type_emb + 6 * species_emb + (TURN_DELTA_DIM - 10)
              = 2*16 + 2*16 + 6*32 + 78
              = 334
```

`history_proj` Linear's input dim auto-adjusts in `__init__` because the
projection input dim is discovered via dummy forward pass — no manual
plumbing change. Layer reuse means no new embedding table.

### Observation dimension growth

| Block | Pre-expansion | Post-expansion |
|-------|---------------|----------------|
| Base (teams + active ctx + global + reactive) | 1523 | 1523 |
| Prev-turn action mask | 11 | 11 |
| Turn history (N × slot_dim) | 10 × 39 = 390 | 10 × 88 = 880 |
| **Total** | **1924** | **2414** |

### Architecture version bump

`ARCH_SIGNATURE` changed from `"gen3_unified_v1"` to `"gen3_unified_v2"`.
The total_dim mismatch alone would catch v1 checkpoints via
`check_compatible()`, but the signature bump produces an explicit
"architecture family mismatch" error — and the slot extension introduces a
new wire (the species_embedding table is now reached from the history
block, which it never was before in v1), so the structural break is real.
`MODEL_CONFIG_VERSION` stays at 2; no migration needed because every
weight-shape difference is covered by existing fields plus the signature.

### TurnDelta extensions (`battle_context.py`)

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `phase_is_forced_switch` | `bool` | `False` | From `curr_ctx.phase == "forced_switch"` |
| `our_hp_after` | `np.ndarray` (6,) float32 | zeros | Copied from `curr_ctx.our_hp` |
| `opp_hp_after` | `np.ndarray` (6,) float32 | zeros | Copied from `curr_ctx.opp_hp` |
| `our_target_hp_delta` | `Optional[float]` | `None` | HP delta on opp's damaging-event target on our side |
| `opp_target_hp_delta` | `Optional[float]` | `None` | Symmetric |

`_resolve_target_hp_delta(event, hp_delta, slot_map)` is a small helper that
looks up the named target species in the slot map and returns the delta at
that slot — None when no event fired or the target isn't in the map.

### `Optional[float]` → 0.0 ambiguity in encoded slot

`target_hp_delta` is coerced `None → 0.0` in the encoded vector. "No
damaging event this turn" and "event with exactly zero damage" both encode
as 0.0 — but the paired actor / target species IDs collapse to 0 in the
no-event case, and `power_norm` / effectiveness onehots are also zero, so
the model has multiple signals to disambiguate. Not worth a separate
validity bit.

---

## What This Enables

The protocol's `|move|<user>|<move>|<target>|` line is now first-class data on
`BattleContext`, **and** the protocol-truth attribution data is now in the
model's observation through the expanded turn-history block:

- **Reward shaping** can attribute pressure correctly. "Opp Pursuit'd our
  switch" is now a single field check — previously required cross-referencing
  prev/curr active mons against `delta.our_switch_to` and move IDs.
- **Turn-history features** record real events rather than inferred ones —
  shipped in this same step. The 88-dim TurnDelta slot now carries actor,
  target, target_status, switch_to species, boost magnitudes, phase, and
  per-slot HP levels in addition to the legacy move/effectiveness scalars.
  The model can learn cross-turn inferences like Hidden Power type narrowing
  by target species, Toxic-on-Steel immunity confirmation, CM-magnitude
  reads, post-KO forced-in identity, and mirror-match attribution.
- **Replay analysis** in tooling can show ground-truth attribution. The Step 2
  `_resolve_hp_target` was production-only — no replay viewer could call into
  it without dragging in the full episode tracker. The damaging event is
  serialisable.
- **Future BC / MCTS work** (ai_v5) needs deterministic attribution to
  reproduce battles from replay; the protocol-truth path is the right
  primitive.

---

## Files Changed

| File | Change |
|------|--------|
| `src/poke_env/battle/abstract_battle.py` | `DamagingMoveEvent` NamedTuple; `_our/opp_last_damaging_move` + `_pending_*` slots and init; `_set_effectiveness` helper; pending capture at `|move|`; promotion at `-supereffective` / `-resisted` / `-immune`; Flash Fire `-start` → `_set_effectiveness(..., 0.0)`; `our/opp_last_damaging_move` turn-gated properties |
| `src/agents/training/battle_context.py` | Import `DamagingMoveEvent`; add `our_team_order`, `our_last_damaging_event`, `opp_last_damaging_event` fields (with defaults); populate all three in `from_battle()`; intent-correct `our_switch_to` derivation in `TurnDelta.build` from `prev_ctx.our_team_order[action]` |
| `src/agents/training/episode_tracker.py` | **Delete** `_resolve_hp_target`, `_snapshot_target`, `_find_prev_move_selection_index`, `BATON_PASS` constant, `TurnDelta` import; replace with `_wrap_hp_target` adapter; `_maybe_observe_hidden_power` reduces to a direct event read |
| `src/agents/training/hidden_power_tracker_fuzz_e2e_test.py` | **Delete** `_resolve_target`, `_snapshot_target`, `_is_switch_order`, all `_prev_*` tracking, the turn_log diagnostic; **add** `_protocol_says_hp` cross-check, `_observed_keys` dedupe, `_observe_event` helper; choose_move shrinks to a thin event router; ~38% net line reduction |
| `src/agents/training/battle_context_test.py` | Set `our_team_order` on prev in switch-action tests; new `test_turn_delta_build_switch_action_uses_intent_not_end_state` covering the multi-faint regression |
| `src/agents/training/battle_recorder_test.py` | `_battle` helper sets `our_last_damaging_move = None`, `opp_last_damaging_move = None` so the namespace stub satisfies `BattleContext.from_battle` |
| `src/agents/training/battle_context.py` (TurnDelta expansion) | Add `phase_is_forced_switch`, `our/opp_hp_after`, `our/opp_target_hp_delta` fields with defaults; `_resolve_target_hp_delta` helper; `build()` populates them via `curr_ctx.phase`, `curr_ctx.{our,opp}_hp.copy()`, and slot-map lookup |
| `src/agents/observation/turn_delta_encoder.py` | `TURN_DELTA_DIM` 39 → 88; new constants `STATUS_DIM`, `HP_LEVEL_DIM`, `SPECIES_ID_COUNT`, `OFFSET_OUR_HP_DELTA_SUM`/`OFFSET_OPP_HP_DELTA_SUM` (base block) and `OFFSET_OUR_BOOST_DELTA` … `OFFSET_OPP_SWITCH_TO_SPEC` (extended block); accept `gen3_species` mapping; `_species_id` (raises on unknown non-sentinel); `_status_onehot`; `_actor_species` / `_target_species` helpers using "this-side-this-turn" convention; `encode()` emits extended block; `describe_vector` decodes every new field |
| `src/agents/model/features_extractor.py` | `_embed_delta_slot` extended to look up 6 species IDs via `species_embedding`; `_td_embed_dim` formula updated (`+ 6 * species_emb`, `- 10` raw IDs); `history_proj` input auto-adjusts |
| `src/agents/model/model_version.py` | `ARCH_SIGNATURE` `"gen3_unified_v1"` → `"gen3_unified_v2"` with changelog comment listing the slot additions |
| `src/agents/observation/state_encoder.py`, `src/agents/training/gen3_env.py`, `src/agents/inference/player.py`, `src/agents/training/poke_env_gaps/effectiveness_fuzz_e2e_test.py` | TurnDeltaEncoder constructor call sites pass `mappings.get("species", {})` |
| `src/agents/observation/turn_delta_encoder_test.py` | Rewritten: minimal `_SPECIES` fixture, `TURN_DELTA_DIM == 88` assertion, +11 new tests covering every extended field including the strict-species raise and sentinel paths |
| `src/agents/observation/state_encoder_test.py` | `EXPECTED_OBS_DIM` 1924 → 2414 (and 2274 in the in-between commit) |
| `src/agents/training/episode_tracker_test.py` | Encoder factory passes species mapping; `OUR_HP_DELTA_IDX` imported from `OFFSET_OUR_HP_DELTA_SUM` |
| `src/agents/inference/player_test.py` | Mock battle explicitly stubs `our/opp_last_damaging_move = None` (surfaced by strict species lookup) |
| `CLAUDE.md` | Observation Vector table updated: 1525 → 2414 total dim, turn-history slot 39 → 88 dims, prev_mask offset 1319 → 1523, history offset 1330 → 1534; turn-history slot section rewritten to describe base + extended layout |
