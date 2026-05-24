# Implementation: Step 3 — Full Gen3 Mechanics

Extend `src/sim/src/turn.rs` (and supporting modules) until the fuzz harness passes
10k random battles with zero state divergences. All work is driven by fuzz failures —
no speculative implementation. Run the harness, read the first divergence, fix it,
re-run.

---

## Workflow

```bash
# Run with the 32 sample teams (covers most Gen 3 OU mechanics)
export PYTHONPATH=$PYTHONPATH:src
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 \
  src/agents/mcts/rust_sim_fuzz_e2e_test.py --battles 200

# A divergence looks like:
#   turn 7 | p1.team[2].hp: showdown=183  rust=185
#   turn 7 | p1.team[2].statusState.duration: showdown=2  rust=3
```

Read the divergence, identify which mechanic is wrong, find the canonical
implementation in the Showdown source files below, fix it in Rust, re-run.
Increase `--battles` as mechanics stabilise.

---

## Source File Reference

Before implementing any mechanic, read the relevant Showdown source. The Gen 3
engine inherits from Gen 4 with overrides — **always check the Gen 3 mod file
first**; if a mechanic is not overridden there, read the base implementation.

| File | Contains |
|------|---------|
| `deps/pokemon-showdown/data/mods/gen3/scripts.ts` | Damage formula ordering, physical/special split, accuracy calc, multi-hit, recoil |
| `deps/pokemon-showdown/data/mods/gen3/conditions.ts` | Sleep, freeze, sandstorm overrides |
| `deps/pokemon-showdown/data/mods/gen3/items.ts` | All Gen 3 item overrides (berry triggers, type boosters) |
| `deps/pokemon-showdown/data/mods/gen3/moves.ts` | All Gen 3 move overrides |
| `deps/pokemon-showdown/data/mods/gen3/abilities.ts` | Gen 3 ability overrides |
| `deps/pokemon-showdown/sim/battle-actions.ts` | Base damage calc, move execution, switch handling |
| `deps/pokemon-showdown/data/conditions.ts` | Base status conditions (burn, par, psn, tox, frz) |
| `deps/pokemon-showdown/data/items.ts` | Base item implementations (Leftovers, Choice Band, etc.) |

---

## Mechanic Groups

Ordered by frequency in the 32 sample teams. Work through them in this order — the
fuzz harness will surface failures roughly in this priority.

---

### Group 1 — Physical/Special Split

**Source:** `data/mods/gen3/scripts.ts` lines 5–14

Gen 3 determines move category from type, not from the move's declared `category`
field. This is applied at `init()` time by Showdown and must be replicated in the
Rust sim's data loader.

**Special types:** Fire, Water, Grass, Ice, Electric, Dark, Psychic, Dragon.
Everything else is Physical.

```rust
// data.rs — when loading move data from gen3_moves.json
fn is_special_type(type_id: &str) -> bool {
    matches!(type_id,
        "Fire" | "Water" | "Grass" | "Ice" |
        "Electric" | "Dark" | "Psychic" | "Dragon"
    )
}
// Override move.category based on type, ignoring the stored category field
```

**This must be correct before any damage calc is meaningful.** A physical move being
treated as special (or vice versa) will cause every damage value to diverge.

---

### Group 2 — Damage Formula (exact ordering)

**Source:** `data/mods/gen3/scripts.ts` `modifyDamage()` lines 33–118

The Gen 3 damage formula has a specific ordering that differs from later gens.
The `+2` base damage and the crit modifier position are both critical.

```
Step 1: Apply burn (0.5× if burned + Physical move + no Guts ability)
Step 2: ModifyDamagePhase1 (Reflect/Light Screen — see Group 5)
Step 3: Spread move modifier (0.5× in Gen 3, not 0.75× like later gens)
Step 4: Weather modifier (WeatherModifyDamage event)
Step 5: If Physical and damage is 0, set to 1  ← floor before +2
Step 6: ADD 2  ← this happens after burn/screens, before crit
Step 7: Crit modifier (2×; Gen 3 crits are always 2× — no 1.5× like Gen 6+)
Step 8: ModifyDamagePhase2 (floor after all ×modifiers)
Step 9: STAB (1.5× if move type matches user type)
Step 10: Type effectiveness (each resist: floor(damage/2); each SE: damage*2)
Step 11: Random roll: damage = floor(damage × roll / 255)
         where roll = random(39) + 217  → values in [217..255], ~16 discrete steps
Step 12: Final ModifyDamage event
Step 13: Minimum 1 HP
```

**Critical Gen 3 specifics:**
- **Crits ignore ALL stat stage modifiers.** Both attacker's boosts and defender's
  drops are ignored entirely. Recalculate atk/def using base stats + EVs/IVs only.
  This differs from Gen 4+ where only the unfavourable modifiers are ignored.
- **Each resist applies a separate `floor(÷2)`.** A 4× resist is
  `floor(floor(damage / 2) / 2)`, not `floor(damage / 4)`. Observable.
- **`+2` is added after burn and screens, before crit.** If you add it at the end
  the result is wrong for low-damage moves.
- **Random roll is last** — applied after STAB and type effectiveness.

---

### Group 3 — Status Conditions

**Source:** `data/mods/gen3/conditions.ts`, `data/conditions.ts`

#### Sleep (`slp`)

```
onStart:
  effectState.time = random(2, 6)   → 1–4 turns remaining
  effectState.skippedTime = 0

onSwitchIn:
  effectState.time += effectState.skippedTime   ← sleep counter carries over on switch-in
  effectState.skippedTime = 0

onBeforeMove:
  time--
  if time <= 0: cure status, allow move this turn
  else: add 'cant' message, block move
        if move.sleepUsable (Sleep Talk / Snore): skippedTime++
```

Key: `time` starts at 1–4, decrements BEFORE checking. Pokemon wakes when `time`
reaches 0 (i.e. after using their last sleep turn). The sleep counter is
`statusState.duration` in the serialised state.

`skippedTime` is **not** in `serializeBattle()` output — it resets on switch-in so
only matters within a turn. The `time` field IS serialised and must match exactly.

#### Burn (`brn`)

- Halves Attack for Physical moves (applied in damage formula, Group 2)
- End-of-turn residual: `damage(max_hp / 8)` — minimum 1
- Applied after Leftovers in the residual chain

#### Paralysis (`par`)

- Speed halved (applied when computing action speed / `getStat('spe')`)
- 25% chance to be fully paralysed: `onBeforeMove` → `randomChance(1, 4)` → can't move

#### Poison (`psn`)

- End-of-turn residual: `damage(max_hp / 8)`

#### Toxic (`tox`)

- `statusState.duration` is the toxic counter (starts at 1, increments after each
  residual damage application)
- End-of-turn residual: `damage(max_hp × counter / 16)`, then counter++
- Counter resets to 1 on switch-out/switch-in

#### Freeze (`frz`)

**Source:** `data/mods/gen3/conditions.ts` lines 41–46

Gen 3 override: freeze is permanent unless broken by a Fire-type damaging move
hitting the frozen Pokemon. The base thaw chance (`randomChance(1, 5)` per turn)
does NOT apply in Gen 3 — `onModifySpD` is undefined (no SpD boost from Sandstorm
either, see Group 7).

```rust
// frz: no natural thaw in Gen 3
// Only cured by: Fire-type damaging move hitting the frozen Pokemon
// (handled in the onDamagingHit callback in gen3/conditions.ts)
```

---

### Group 4 — Items

**Source:** `data/mods/gen3/items.ts`, `data/items.ts`

#### Leftovers

- End-of-turn residual: `heal(max_hp / 16)`
- Applied before burn/poison in the residual chain (residual order 5)

#### Sitrus Berry

**Gen 3 difference:** heals flat **30 HP**, not 25% of max HP (which came in Gen 4).

```rust
// items.ts Gen 3 override:
// onResidual: if hp <= max_hp / 2: eat item
// onEat: heal(30)   ← flat 30, NOT percentage
```

Trigger: end-of-turn (`onResidual`), not immediately on taking damage.
Berry activates when HP drops to or below 50% during the residual phase.

#### Half-HP Berries (Figy, Wiki, Mago, Aguav, Iapapa)

- Trigger at ≤ max_hp / 2 HP (end of turn)
- Restore max_hp / 8 HP

#### Quarter-HP Berries (Liechi, Ganlon, Salac, Petaya, Apicot, Lansat, Starf)

- Trigger at ≤ max_hp / 4 HP (end of turn)
- Apply a +1 stat boost (type varies by berry)

#### Lum Berry

- Cures any status condition on activation
- Triggers at end of turn (`onResidual`)

#### Type-Boosting Items (Black Belt, Charcoal, Mystic Water, etc.)

**Gen 3 difference:** these boost the *stat* (`onModifyAtk` or `onModifySpA`) by
1.1×, **not** the base power. This means they interact differently with the damage
formula — they inflate the stat used in the `atk/def` ratio, not the power term.

All use `chainModify(1.1)` on the appropriate stat.

#### Choice Band

- Boosts Attack by 1.5× (`onModifyAtk`)
- In Gen 3, Choice Band does **not** lock the user into one move — that mechanic
  came in Gen 4. It is purely a stat boost item.

#### Quick Claw

**Source:** `data/mods/gen3/scripts.ts` `getActionSpeed()` lines 18–28

Quick Claw in Gen 3 sets speed to 65535 within the speed ordering (not a priority
bracket change). `quickClawRoll` is resolved at the start of turn selection.
Implement as: if Quick Claw activates (`randomChance(1, 5)`), this Pokemon's action
speed is treated as 65535 for ordering purposes this turn.

---

### Group 5 — Entry Hazards and Screens

#### Spikes

- 1 layer: 1/8 max HP on switch-in
- 2 layers: 1/6 max HP on switch-in
- 3 layers: 1/4 max HP on switch-in
- Flying types and Levitate ability holders are immune
- Rapid Spin removes them (see Group 8)

#### Reflect / Light Screen

- Duration: 5 turns (base), reduced to 3 by Light Clay (not in Gen 3 OU standard)
- Reflect: halves Physical damage; Light Screen: halves Special damage
- Screens are applied in **ModifyDamagePhase1** (before the `+2` is added)
- Screens are **bypassed by crits** — if isCrit, skip the screen modifier
- Brick Break removes the opponent's screens before hitting (even through Substitute)
  — see `data/mods/gen3/moves.ts` lines 108–114

---

### Group 6 — Volatiles

#### Confusion

- Duration: 1–4 turns (`random(2, 6)`)
- `onBeforeMove`: 50% chance to hit self for a typeless Physical 40-power move
  (uses `confusionDamage` which ignores type effectiveness and STAB)
- Counter decrements each turn; clears on switch-out

#### Flinch

- Applied by moves with secondary flinch chance
- `onBeforeMove`: Pokemon cannot move this turn
- Always clears after the turn (duration 1)

#### Substitute

- Created at 25% HP cost; absorbs damage until broken
- Blocks status moves from the opponent (but not field moves like Spikes)
- HP tracked in `volatiles['substitute'].hp`

#### Encore

**Source:** `data/mods/gen3/moves.ts` lines 251–259

- Duration: 2–6 turns (`random(3, 7)`)
- Forces the target to repeat the last used move
- Ends early if PP of the encored move runs out

#### Leech Seed

- End-of-turn: drain 1/8 max HP from seeded Pokemon, heal to opponent
- Grass types are immune
- Clears on switch-out

#### Taunt

**Source:** `data/mods/gen3/moves.ts` lines 593–603

Gen 3 Taunt lasts exactly **2 turns** (hard `duration: 2`, no durationCallback).
This differs from later gens. Taunt prevents Status-category moves.

#### Disable

**Source:** `data/mods/gen3/moves.ts` lines 201–212

Duration: 1–5 turns (`random(2, 6)`). Disables the last move used.

---

### Group 7 — Weather

#### Sandstorm

**Source:** `data/mods/gen3/conditions.ts` lines 47–49

Gen 3 override: `onModifySpD: undefined` — **Rock types do NOT get a SpD boost in
Gen 3.** The Sandstorm SpD boost is a Gen 4 addition.

End-of-turn: deal 1/16 max HP to all non-Rock, non-Steel, non-Ground types.
Duration: 5 turns.

#### Rain Dance

- Water moves: ×1.5; Fire moves: ×0.5 (applied in WeatherModifyDamage event)
- Thunder: accuracy becomes `true` (always hits)
- Duration: 5 turns

#### Sunny Day

- Fire moves: ×1.5; Water moves: ×0.5
- Solar Beam fires without charge turn
- Duration: 5 turns

---

### Group 8 — Complex Moves

These have non-trivial interactions that will surface as fuzz failures.

#### Rapid Spin

- Deals 20 damage, then removes the user's side's Spikes and the user's Leech Seed
  (`onAfterHit` or `onHit` callback)

#### Brick Break

**Source:** `data/mods/gen3/moves.ts` lines 108–114

Removes the opponent's Reflect AND Light Screen **before the hit resolves**, even
if hitting a Substitute. The screens are gone for the damage calculation of the
same Brick Break hit.

#### Pursuit

**Source:** `data/mods/gen3/moves.ts` lines 479–490

Complex interaction: if the target switches out on the same turn, Pursuit executes
before the switch with doubled base power (100) and ignores the switch. Showdown
implements this via `beforeTurnCallback` adding a volatile to the switching target.

#### Counter / Mirror Coat

**Source:** `data/mods/gen3/moves.ts` lines 163–182, 407–426

**Gen 3 Counter:** tracks Physical damage only. **Exception:** Hidden Power also
triggers Counter in Gen 3 (because HP is computed as Physical or Special by type,
but Counter always tracks it — see the `effect.id === 'hiddenpower'` check in
`onDamage`). Counter deals 2× the last Physical (or HP) hit taken.

**Gen 3 Mirror Coat:** tracks Special damage only. **Exception:** Hidden Power does
NOT trigger Mirror Coat in Gen 3 (unlike Counter). Mirror Coat deals 2× the last
Special hit taken.

#### Beat Up

**Source:** `data/mods/gen3/moves.ts` lines 33–55

Each party member that is alive and has no status contributes one hit. The hit uses
that party member's base Attack stat as the attacking stat (not current stat with
boosts), and the target's base Defense. Move type is `???` (no STAB, no type
effectiveness), category is Special.

#### Hidden Power

**Source:** `data/mods/gen3/moves.ts` lines 348–355

Type is determined by IVs (already in `gen3_moves.json` or computed via
`gen3_hidden_power`). Category is Physical or Special based on the type (same
physical/special split as Group 1).

#### Struggle

**Source:** `data/mods/gen3/moves.ts` lines 581–586

Recoil is 25% of damage dealt (`recoil: [1, 4]`), not a percentage of max HP.
This is a Gen 3 override — later gens use `struggleRecoil` (HP-based).

---

### Group 9 — Abilities

**Source:** `data/mods/gen3/abilities.ts`

Gen 3 has a limited ability set. Most relevant for Gen 3 OU:

- **Intimidate**: on switch-in, lower opponent's Attack by 1
- **Levitate**: immune to Ground moves and Spikes
- **Flash Fire**: immune to Fire moves; boosts Fire moves by 1.5× after activation
- **Synchronize**: if inflicted with burn/paralysis/poison, opponent gets the same
- **Early Bird**: halves sleep duration (implemented via Early Bird halving in the
  `onBeforeMove` sleep check)
- **Truant**: every other turn, Pokemon uses "loafing around" and cannot move
- **Guts**: if statused, 1.5× Attack; also negates burn's Attack drop for physical
  moves in the damage formula
- **Natural Cure**: cures status on switch-out

These will appear as fuzz failures when teams with these abilities are used.
The `data/mods/gen3/abilities.ts` file has all Gen 3 overrides.

---

## Milestones

### M1 — Physical/Special split + damage formula exact (500 battles)
Every move category assigned correctly. All damage formula steps in the right order
including +2 placement, crit ignoring all boosts, individual resist floors, and
random roll being last. 500 battles, zero divergences.

*Verify:* pick a turn from a fuzz run, extract the matchup, compute expected damage
by hand using the formula above, confirm Rust and Showdown agree.

### M2 — Status conditions (1k battles)
Sleep counter (including skippedTime reset on switch-in), burn residual, paralysis
speed halve + 25% skip, poison and toxic residuals (toxic counter increments), freeze
only thaws from Fire-type hits. 1k battles, zero status divergences.

*Verify:* in a run with a sleeping Pokemon, trace `statusState.duration` turn-by-turn
across both Rust and Showdown output — they must match every turn.

### M3 — Items (1k battles)
Leftovers (+1/16 per turn), Sitrus Berry (flat 30, triggers at ≤50% end of turn),
type-boosting items (modify stat not base power), Choice Band (stat boost only, no
lock), Quick Claw (action speed 65535). 1k battles using teams with common items.

*Verify:* a battle involving Leftovers recovery must produce identical HP values
on every turn.

### M4 — Entry hazards + Screens (500 battles)
Spikes damage on switch-in (1/2/3 layers → 1/8, 1/6, 1/4), flying/levitate
immunity, Rapid Spin clears. Reflect/Light Screen turn counters, half-damage for
correct category, bypassed by crits, removed by Brick Break before the hit.
500 battles with Spikes and screens teams.

### M5 — Volatiles (2k battles)
Confusion self-hit (typeless Physical 40-power), flinch, Substitute HP tracking,
Encore duration (2–6 turns), Leech Seed drain, Taunt (exactly 2 turns).
2k battles, zero volatile divergences.

### M6 — Weather (1k battles)
Sandstorm: no Rock SpD boost, correct residual targets. Rain: ×1.5 Water / ×0.5
Fire, Thunder always hits. Sun: ×1.5 Fire / ×0.5 Water, Solar Beam fires instantly.
1k battles with weather teams.

### M7 — Complex moves (2k battles)
Pursuit doubling on switch, Counter (physical + HP, not special), Mirror Coat
(special only, not HP), Beat Up (base stats, `???` type), Struggle recoil (25% of
damage not max HP), Brick Break removing screens before damage, Hidden Power
category from type. 2k battles, zero complex-move divergences.

### M8 — Abilities (2k battles)
Intimidate, Levitate (Spikes immunity), Flash Fire, Synchronize, Early Bird,
Truant, Guts (burn damage negation), Natural Cure. 2k battles with ability-heavy
teams.

### M9 — 10k battles (final gate)
Run `--battles 10000` with the full 32 sample team pool. Zero divergences.
This is the Step 3 exit gate.

---

## Parallelisation Note

Once M1 (damage formula) is confirmed correct, Groups 3–9 are largely independent
and can be implemented by parallel agents:

- **Agent A**: Status conditions (Group 3) + Weather (Group 7)
- **Agent B**: Items (Group 4) + Abilities (Group 9)
- **Agent C**: Entry hazards + Screens (Group 5) + Volatiles (Group 6)
- **Agent D**: Complex moves (Group 8)

Each agent runs the fuzz harness targeting their specific mechanic area (using teams
that exercise those mechanics), fixes failures, and the final M9 run integrates all
of them.

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/sim/src/turn.rs` | All mechanic implementations |
| `src/sim/src/state.rs` | Add volatile fields as fuzz failures reveal them |
| `src/sim/src/data.rs` | Physical/special split in move loader; item effect tables |
| `src/sim/src/damage.rs` | Formula refinements (burn position, crit behaviour, floor order) |
| `src/agents/mcts/rust_sim_fuzz_e2e_test.py` | Increase `--battles` gate; add per-mechanic targeted runs |

---

## Final State

Step 3 is complete when:
- `--battles 10000` passes with zero divergences using the 32 sample teams
- All milestones M1–M9 individually verified
- No known divergence classes remaining

**Ready for Step 4:** the Rust sim is a faithful oracle of Showdown. Step 4 introduces
`EncoderState` and fuzz-tests that both the poke-env adapter and the Rust adapter
produce identical observation vectors for the same game state.
