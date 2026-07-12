//! Turn — the FULL per-turn cycle + a multi-turn driver for turns where BOTH sides
//! use a damaging move (gen 3 OU singles). The **RNG-consumption-order** layer: it
//! orders the two move actions (priority → effective speed, with the speed-tie
//! Fisher-Yates shuffle that DRAWS — wiring [`crate::event::speed_sort`] onto a
//! production path), resolves each move, runs the per-action `eachEvent` shuffles +
//! the end-of-turn residuals, and loops the cycle across turns — consuming the PRNG
//! in the EXACT order + count Showdown does, sustained across multiple turns.
//!
//! # Scope (read first — this is a BOUNDED step)
//!
//! Implemented:
//! - action ordering (the action-order speed-tie shuffle draw);
//! - the per-action `eachEvent('BeforeTurn'/'Update'/'Weather')` speed-tie shuffles
//!   (the draws the single-turn step deferred — so a TIE turn AND cross-turn seed
//!   parity now hold);
//! - per move, in resolved order: accuracy → crit → damage
//!   ([`crate::damage::calc_damage`]) → apply HP → the deferred-faint protocol;
//! - the END-OF-TURN RESIDUALS (gen-3 residualOrder): the weather chip
//!   (Sandstorm/Hail `maxhp/16` to non-Rock/Ground/Steel), Leftovers (`+maxhp/16`),
//!   and the major-status DoT (burn `maxhp/8`, poison `maxhp/8`, Toxic `n/16` ramp),
//!   all draw-free arithmetic except the handler-sort + nested-Weather tie-shuffles;
//! - the gen-3 end-of-turn Quick Claw roll;
//! - [`BattleState::run_battle`] — a scripted multi-turn loop that STOPS at the first
//!   faint (no switching);
//! - [`BattleState::run_full_battle`] — the FULL battle-to-WIN/LOSS driver:
//!   voluntary SWITCHES (sort before moves; the two-switch action-order tie-shuffle),
//!   the gen-3 draw-FREE switch-in (the entrant's ability `Start`, reusing
//!   [`crate::event::single_event_ability_start`]), POST-FAINT replacement (single +
//!   DOUBLE, with the double's `insertChoice` order-101 splice draw), the pause/resume
//!   of the saved turn tail, and win/loss (`pokemon_left == 0` loses; both → a gen-3
//!   tie). It consumes a [`ScriptDecision`] per REQUEST boundary (a `move` request =
//!   both sides; a forced `switch` = the flagged side(s)) and returns a
//!   [`BattleOutcome`] (winner + per-decision records). See its doc for the exact
//!   switch-phase draw model.
//!
//! DEFERRED (NOT modeled here): secondary effects (the per-move `random(100)`),
//! status MOVES, entry hazards (Spikes), Pursuit (the switch-trap move), Baton Pass,
//! Leech Seed / Wish / items beyond Leftovers, status `onBeforeMove` draws
//! (para/sleep/freeze), and protocol-string emission. This step asserts STATE+SEED+
//! winner, not protocol bytes.
//!
//! # The EXACT per-turn PRNG draw order (verified — `harness/trace_multiturn_rng.js`)
//!
//! Maximal (SPEED-TIE, no weather) one turn cycle = 16 draws, in this order:
//!
//! 1. **action-order speed-tie shuffle** — `commitChoices → BattleQueue.sort →
//!    speedSort`, BEFORE either move. Tie-only.
//! 2. **eachEvent('BeforeTurn')** shuffle — `beforeTurn` runAction body. Tie-only.
//! 3. **eachEvent('Update')** shuffle — end of the `beforeTurn` runAction. Tie-only.
//! 4–6. **mover 1**: accuracy `randomChance(acc,100)` (skip iff `never_miss`) → crit
//!    `randomChance(1,critMult[critRatio])` → damage `random(16)`.
//! 7. **eachEvent('Update')** shuffle INSIDE `tryMoveHit` (gen3 `scripts.ts:470`) —
//!    fires ONLY when the move LANDED (a miss/immune returns before it). Tie-only.
//! 8. **eachEvent('Update')** shuffle — end of mover-1's runAction. Tie-only.
//! 9–11. **mover 2**: accuracy → crit → damage.
//! 12. in-`tryMoveHit` Update shuffle (mover 2). Tie-only.
//! 13. end-of-mover-2-runAction Update shuffle. Tie-only.
//! 14. **residual** `fieldEvent('Residual')` handler-sort shuffle — draws on a full
//!    comparePriority tie (e.g. both sides' Leftovers at equal speed). With
//!    sandstorm present its `onFieldResidual` also nests an `eachEvent('Weather')`
//!    shuffle (a 17th draw in sand-tie). The HP effects themselves draw NOTHING.
//! 15. end-of-residual-runAction Update shuffle. Tie-only.
//! 16. **Quick Claw** `randomChance(1,5)` — UNCONDITIONAL in gen3 at `endTurn`, but
//!    reached only if `endTurn()` completes (no faint this turn).
//!
//! At DISTINCT speed every shuffle vanishes (the 2 actives never tie), so the turn
//! is exactly `acc/crit/dmg ×2 + QuickClaw` (≤7 draws) — the single-turn closure
//! looped.
//!
//! # The faint-turn draw-COUNT crux (the deferred-faint protocol)
//!
//! When a move KOs the defender, Showdown's `pokemon.damage` only ZEROES the HP and
//! enqueues the mon — the `fainted` flag is set later by `faintMessages()` at the
//! END of the runAction. So the in-`tryMoveHit` `eachEvent('Update')` shuffle runs
//! while the KO'd mon is at 0 HP but STILL counted by `getAllActive()` (it fires on
//! a tie turn), and only THEN is the mon excluded. The KO then reaches
//! `makeRequest('switch')` and returns BEFORE the trailing end-of-runAction Update —
//! so a faint turn draws the in-tryMoveHit shuffle but NOT the trailing Update, the
//! second mover (faint-skips-rest), the residual, or the Quick Claw. This crate
//! mirrors that by splitting [`BattleState::apply_damage`] (zero HP) from
//! [`BattleState::process_faints`] (set `fainted`, run AFTER the in-tryMoveHit
//! shuffle). A first-mover KO truncates the second move's [acc, crit, dmg] entirely.
//!
//! # Effective speed (gen 3) for the tie key
//!
//! `action.speed = pokemon.getActionSpeed()` — the gen-3 OVERRIDE
//! (`data/mods/gen3/scripts.ts:18`) returns the RAW boosted + ModifySpe speed,
//! **NOT** `trunc(spe, 13)` (that is the base-sim path gen3 replaces). So the tie
//! key is `getStat('spe', false, false)` = `storedStats.spe` through the boost
//! table (floor) then ModifySpe (paralysis ×0.5 = `floor(spe*50/100)` in gen 3 —
//! NOT ×0.25; Choice Scarf is gen-4+, absent here), capped at 10000. For the clean
//! both-attack tie scenarios (no para, no boosts) this is just `stats[5]`.

use crate::damage::{
    calc_damage, AtkStatMod, BpMod, Combatant, DamageContext, MoveInput, Weather as DmgWeather,
};
use crate::dex::{to_id, Dex, DmgFold, MoveCategory, Type, TypeBoostFold};
use crate::event::{single_event_ability_start, speed_sort, EventHandler, NO_ORDER};
use crate::prng::PrngSeed;
use crate::protocol::{Cause, HpStatus, MonRef, ProtocolLine};
use crate::state::{BattleState, Status, Weather, BOOST_LEN};

/// The denominators `critMult[critRatio]` for gen ≤ 5 (`battle-actions.ts:1631`):
/// index by the (clamped 0..5) crit ratio. A normal damaging move resolves to
/// `critRatio == 1` ⇒ `randomChance(1, 16)`; high-crit moves use 8/4/3/2.
const CRIT_MULT: [u32; 6] = [0, 16, 8, 4, 3, 2];

/// The gen-3 PROTECT/DETECT `stall` volatile's `counterMax` (`data/mods/gen4/
/// conditions.ts`: `stall: { inherit: true, counterMax: 8 }`, inherited by gen3 —
/// "In gen 3-4, the chance of protect succeeding does not fall below 1/8"). The
/// `onRestart` `counter *= 2` stops doubling once `counter >= counterMax`, so the
/// consecutive-success denominator sequence floors at 8: `2 → 4 → 8 → 8 → …`
/// (success 100%/50%/25%/12.5%/12.5%). VERIFIED vs the sim's resolved condition +
/// the PRNG probe (`harness/probe_protect_rng.js`).
///
/// PINNED to the COMPILED `deps/pokemon-showdown/dist/` build's RESOLVED gen3 `stall`
/// (gen5-base `onStart 2` / `onStallMove randomChance(1,counter)` / `onRestart *=2`,
/// NO delete-on-fail, with the gen4 `counterMax: 8` override) — the same `dist/` the
/// goldens are generated against. If `dist/` is ever rebuilt and upstream re-resolves
/// `stall` differently, regenerate the protect + e2e goldens.
const PROTECT_COUNTER_MAX: u8 = 8;

/// The gen-3 PROTECT/DETECT `stall` volatile's `duration` (the base `stall` condition's
/// `duration: 2`, reset to 2 by `onStart`/`onRestart` on every successful protect). The
/// volatile is decremented at each RESIDUAL and EXPIRES (→ `protect_counter` 0) at 0 —
/// so the stall counter survives exactly one non-protect turn before it resets (VERIFIED
/// vs the sim).
const STALL_DURATION: u8 = 2;

/// The gen-3 TAUNT volatile's FIXED `duration: 2` (`gen3_taunt_disable_v1`) — no
/// `durationCallback`, so applying Taunt draws NO duration dice (VERIFIED vs the sim). The
/// volatile is decremented at each RESIDUAL (a duration-only handler at onResidualOrder 10,
/// onResidualSubOrder 15) and EXPIRES → `None` at 0.
const TAUNT_DURATION: u8 = 2;

/// The TAUNT volatile's residual duration-handler `onResidualSubOrder` (15,
/// `gen3_taunt_disable_v1`) — the gen-3 `taunt` condition carries `onResidualOrder: 10,
/// onResidualSubOrder: 15`, so at the shared order 10 it sorts AFTER Leftovers (sub 4) /
/// Leech Seed (sub 5) / the status DoT (sub 6) *at equal speed* (speed breaks before
/// subOrder within an order group). Its handler decrements the duration + ENDs the volatile
/// at 0. NOTE the mod-chain subtlety: the BASE `taunt` condition has `onResidualOrder: 15`
/// (order 15, no subOrder), but gen3 inherits through the **gen4 mod**, whose condition
/// override sets `onResidualOrder: 10, onResidualSubOrder: 15` and SHADOWS the base — a
/// base-source reading would mis-place the handler. VERIFIED vs the sim's residual handler
/// dump AND behaviorally (`harness/probe_taunt_disable_onbeforemove_rng.js` scenario 6: a
/// FAST taunted mon's `-end` precedes a SLOW foe's brn `-damage` in the same residual —
/// possible only at order 10 with the speed tiebreak; base order-15 would reverse them).
const TAUNT_RESIDUAL_SUBORDER: i32 = 15;

// === Gen-3 RESIDUAL `comparePriority` keys (the gen4-mod overrides gen3 inherits —
//     NOT the base-data values; the base burn/Leftovers orders are wrong for gen3).
//     Smaller `order` resolves FIRST. ===

/// Sandstorm/Hail `onFieldResidualOrder` (`data/mods/gen4/conditions.ts:150/154`).
/// Smaller than the mon-held order 10, so the weather chip resolves FIRST.
const WEATHER_RESIDUAL_ORDER: u64 = 8;
/// The weather field handler's `subOrder` is `resolvePriority`'s `effectTypeOrder`
/// fallback for a Weather effect = 5 (`battle.ts:962`). Unobservable today (order 8
/// already sorts it before every order-10 mon handler, and it's the only order-8
/// handler so it never ties), but kept exact for when a 2nd field handler is added.
const WEATHER_RESIDUAL_SUBORDER: i32 = 5;
/// Leftovers + the major-status DoT both sit at `onResidualOrder` 10
/// (`data/mods/gen4/items.ts:233` for Leftovers; `…/conditions.ts:4/57/62` for
/// brn/psn/tox). The subOrder below splits them.
const STATUS_RESIDUAL_ORDER: u64 = 10;
/// Leftovers `onResidualSubOrder` 4 (`data/mods/gen4/items.ts:234`) — SMALLER than
/// the status DoT's 6, so Leftovers HEALS before the status chips.
const LEFTOVERS_SUBORDER: i32 = 4;
/// The RESIDUAL ABILITY class (`gen3_ability_batch1_v1`, Speed Boost / Rain Dish) —
/// `onResidualOrder` 10, `onResidualSubOrder` **3** (VERIFIED vs the resolved dist
/// `harness/probe_residual_abilities.js`). So at order 10 the ladder is **ability sub 3 →
/// Leftovers sub 4 → leech sub 5 → status DoT sub 6**: Speed Boost's +1 spe / Rain Dish's
/// heal fire FIRST among the mon handlers. Both DRAW-FREE.
const RESIDUAL_ABILITY_SUBORDER: i32 = 3;
/// The **Leech Seed** volatile's `onResidualSubOrder` 5 (`data/mods/gen4/moves.ts:716`,
/// gen3-inherited — the base move-data has subOrder undefined / order 8; the gen4 mod
/// OVERRIDES it to order 10, subOrder 5). So at order 10 the residual ladder is
/// **Leftovers sub 4 → LEECH SEED sub 5 → status DoT sub 6**: Leftovers heals the seeded
/// mon, THEN Leech Seed drains it (and heals the seeder), THEN the burn/poison/Toxic
/// chips. VERIFIED bit-for-bit vs the omniscient sim (`harness/probe_leechseed_rng.js`):
/// `sandstorm[o=8,s=5] → leftovers[o=10,s=4] → leechseed[o=10,s=5] → brn[o=10,s=6]`.
const LEECH_SEED_SUBORDER: i32 = 5;
/// brn/psn/tox `onResidualSubOrder` 6 (`data/mods/gen4/conditions.ts:5/58/63`).
const STATUS_DOT_SUBORDER: i32 = 6;
/// A duration-only volatile's residual handler subOrder. `resolvePriority` falls back to
/// the effect's `effectTypeOrder` when no `onResidualSubOrder` is set; for a Condition
/// (the `protect`/`stall` volatiles) that is **2** (`battle.ts` effectTypeOrder), VERIFIED
/// vs the sim's residual speed-sort dump (`protect[sub=2]` / `stall[sub=2]`). Both protect
/// and stall use it, so they tie (with each other) at order NO_ORDER — the tie-group
/// shuffle the protect golden's seed parity pins.
const VOLATILE_RESIDUAL_SUBORDER: i32 = 2;
/// TRUANT's residual toggle (`gen3_ability_batch4_v1`) — the resolved
/// `truant.onResidualOrder: 27` (base data, gen3-inherited): its OWN order group,
/// AFTER every order-10 mon handler, BEFORE the NO_ORDER duration-only volatiles.
/// The toggle is DRAW-FREE; the ONLY possible order-27 tie is the other side's
/// Truant at equal speed (a Slaking mirror), which adds exactly ONE tie-shuffle
/// draw (probe_truant_rng.js Q4: tied mirror 9 shuffles vs the no-truant control's
/// 8). Its subOrder is unobservable within the group (both members share it) —
/// the ability effectTypeOrder slot is reused.
const TRUANT_RESIDUAL_ORDER: u64 = 27;

/// A resolved per-move action, in the order it will be run. `slot` is the team
/// slot of the actor; `target_slot` the opposing active.
#[derive(Debug, Clone, Copy)]
struct MoveAction {
    side: usize,
    slot: usize,
    move_index: usize,
    /// Whether this action is a FORCED STRUGGLE (`gen3_pp_tracking_v1`): the mon had no
    /// usable move (every slot at 0 PP), so `side.choose` substituted `moveid:'struggle'`.
    /// When set, `run_move` resolves the synthetic `struggle` move (typeless '???', BP 50,
    /// physical, accuracy 100 → draws accuracy, crit + damage like a normal move, + the
    /// gen-3 `recoil:[1,4]` = `max(floor(dmg/4),1)` recoil), IGNORING `move_index` (which
    /// slot the stale script named). Struggle consumes NO PP (it is not a slot).
    struggle: bool,
}

/// The outcome of one [`BattleState::run_move`] call. `landed` gates the
/// in-`tryMoveHit` `eachEvent('Update')` shuffle: it is true ONLY when the move
/// actually hit (acc-hit AND non-immune AND a real damaging move) — a miss / immune
/// / no-op returns from `tryMoveHit` before that shuffle.
#[derive(Debug, Clone, Copy)]
struct MoveResolution {
    missed: bool,
    crit: bool,
    landed: bool,
    /// A SUCCESSFUL **phaze** (Roar / Whirlwind) — the foe's `forceSwitchFlag` was set
    /// (the foe has an eligible switch-in). The runAction tail consumes it via `dragIn`
    /// (the random-target `sample` draw + the forced switch + the runSwitch enqueue),
    /// mirroring `runAction`'s `if (pokemon.forceSwitchFlag) this.actions.dragIn(...)`
    /// (battle.ts:2350). `force_switch_foe` is the side whose active is dragged out.
    /// `None` = no phaze (or a phaze that FAILED — foe's last mon, draw-free except its
    /// accuracy roll). Set ONLY by the phaze arm; every other path leaves it `None`.
    force_switch_foe: Option<usize>,
}

impl MoveResolution {
    /// An out-of-scope (unknown / status / 0-BP) move: no draws, never landed.
    fn no_op() -> MoveResolution {
        MoveResolution { missed: false, crit: false, landed: false, force_switch_foe: None }
    }
    /// The common terminal: a resolved move that drew its taxes but is not landed and
    /// triggers no forced switch (a miss / immune / status-applied / boost / heal).
    fn done(missed: bool, crit: bool, landed: bool) -> MoveResolution {
        MoveResolution { missed, crit, landed, force_switch_foe: None }
    }
}

/// The outcome of a foe hit hitting a Substitute (the `absorb_into_sub` result), used
/// to emit the right protocol line: `Held` → `-activate …|Substitute|[damage]` (the sub
/// survived), `Broke` → `-end …|Substitute` (the sub was destroyed), `NoSub` → the
/// caller applies the damage to the mon (`-damage`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubAbsorb {
    NoSub,
    Held,
    Broke,
}

/// One residual handler's typed action (what its `onResidual`/`onFieldResidual`
/// callback does). Sorted by [`speed_sort`] (the comparePriority key + tie-shuffle)
/// then dispatched in resolved order; every effect is draw-free arithmetic.
#[derive(Debug, Clone, Copy)]
enum ResidualAction {
    /// Sandstorm/Hail chip (the field handler; nests an `eachEvent('Weather')`
    /// shuffle then chips each non-immune active).
    WeatherChip(Weather),
    /// Leftovers heal `maxhp/16` on `side`'s `slot` mon.
    Leftovers { side: usize, slot: usize },
    /// SPEED BOOST (`gen3_ability_batch1_v1`, order 10 subOrder 3): +1 spe stage (clamped +6)
    /// at end of turn, ONLY if the mon `activeTurns` (it was active the whole turn — a mon that
    /// switched in this turn does NOT boost). DRAW-FREE (`boost()` consumes no PRNG). Feeds
    /// NEXT turn's cached speed (like a Dragon Dance — `cached_speed` stays stale until the next
    /// re-cache site). Fires FIRST among the order-10 mon handlers.
    SpeedBoost { side: usize, slot: usize },
    /// RAIN DISH (`gen3_ability_batch1_v1`, order 10 subOrder 3): heal `floor(maxhp/16)` in
    /// EFFECTIVE rain. DRAW-FREE (`heal()` consumes no PRNG; a full-HP heal fails draw-free).
    RainDish { side: usize, slot: usize },
    /// The major-status DoT (burn/poison/Toxic) on `side`'s `slot` mon.
    StatusDot { side: usize, slot: usize },
    /// The **LEECH SEED** drain on `side`'s `slot` mon (the SEEDED holder): it loses
    /// `floor(maxhp/8)` (clamped to its HP) and the seeder's CURRENT active heals the
    /// drained amount. order 10, subOrder 5 (between Leftovers and the status DoT).
    /// `seeder_side` is the side whose active is healed (read at apply time so the heal
    /// follows the seeder's current active, like `getAtSlot(sourceSlot)` in singles).
    /// DRAW-FREE. Gathered with the VOLATILES (after the status DoT, before Leftovers),
    /// mirroring `findPokemonEventHandlers`'s status→volatiles→item order.
    LeechSeed { side: usize, slot: usize, seeder_side: usize },
    /// A duration-only volatile's residual handler (`protect` / `stall`) — gathered by
    /// `findPokemonEventHandlers(..., 'duration')` SOLELY to decrement the volatile's
    /// `duration` (no `onResidual` callback → NO HP effect, DRAW-FREE apply). It MUST be
    /// in the residual handler list because the speed-sort's tie-group Fisher-Yates
    /// shuffle reads the full handler set (a protecting mon adds 2 such handlers — protect
    /// + stall — which tie at order=NO_ORDER/subOrder 2, changing the shuffle COUNT).
    /// `is_stall` distinguishes the longer-lived `stall` volatile (whose `duration: 2`
    /// expiry zeros `protect_counter` — modeled here) from the `protect` volatile (`duration:
    /// 1`, whose expiry is the turn-top `protected` clear — a no-op here). DRAW-FREE.
    VolatileDuration { side: usize, slot: usize, is_stall: bool },
    /// The **TAUNT** volatile's residual duration handler (`gen3_taunt_disable_v1`, order 10,
    /// subOrder 15): decrement `MonState::taunt` and, on reaching 0, CLEAR it (→ `None`, freeing
    /// the mon's Status moves). NO HP effect (`taunt.onResidual` only ticks the duration + emits
    /// `-end`), DRAW-FREE. It participates in the residual speed-sort (a taunted mon adds ONE
    /// tied handler at order 10 subOrder 15 — its only tie is the OTHER mon's taunt at equal
    /// speed).
    TauntDuration { side: usize, slot: usize },
    /// The **DISABLE** volatile's residual duration handler (`gen3_taunt_disable_v1`, order
    /// NO_ORDER, subOrder 2 — the SAME tie-group as protect/stall/flinch): decrement
    /// `MonState::disable`'s turn counter and, on reaching 0, CLEAR it (→ `None`, freeing the
    /// disabled move). NO HP effect, DRAW-FREE. It ties with a same-mon protect/stall/flinch
    /// handler (all NO_ORDER subOrder 2) — but a disabled mon rarely also protects — and with the
    /// other mon's NO_ORDER handlers at equal speed.
    DisableDuration { side: usize, slot: usize },
    /// SHED SKIN (`gen3_berry_trace_shedskin_v1`, order 10 subOrder **3** — the ability
    /// slot, with Speed Boost / Rain Dish): while the holder has a MAJOR status, draw ONE
    /// `randomChance(33,100)`; on a pass CURE it (`-activate` + `-curestatus [msg]`).
    /// Because subOrder 3 < the status DoT's 6, a cure turn takes NO chip. An UNSTATUSED
    /// holder draws NOTHING — but the handler is still GATHERED (the ability registers its
    /// onResidual unconditionally, so it participates in the residual tie-shuffle).
    /// Confusion is NOT cured. Probe `probe_trace_shedskin_rng.js` (S1/S2/S3).
    ShedSkin { side: usize, slot: usize },
    /// TRUANT's end-of-turn toggle (`gen3_ability_batch4_v1`, order **27** — its own
    /// group, AFTER every order-10 handler): `truant_turn = !truant_turn`, DRAW-FREE.
    /// Gathered UNCONDITIONALLY for an active Truant holder (the parity clock ticks on
    /// move AND loaf turns alike — probe_truant_rng.js Q1/Q2); its only tie is a Truant
    /// mirror at equal speed (ONE extra shuffle draw, probe Q4). A holder that fainted
    /// earlier this residual is skipped (the effectHolder.fainted guard), which is what
    /// leaves a POST-residual replacement's armed `true` un-toggled (probe edge E1).
    TruantToggle { side: usize, slot: usize },
    /// A HEAL/PINCH **BERRY**'s residual trigger (`gen3_berry_trace_shedskin_v1`, order 10
    /// subOrder **4** — the ITEM slot, the SAME sort key as Leftovers, gathered in the item
    /// position): at apply time, if the holder's CURRENT hp is at/below the class threshold
    /// (`2*hp <= maxhp` heal / `4*hp <= maxhp` pinch — exact, probe-settled boundaries), the
    /// holder EATS the berry (item → NONE for the battle) and the onEat effect applies
    /// (heal / +1 boost / Starf's `sample` +2 / Lansat's focus-energy volatile / the Figy
    /// family's nature-gated confusion `random(2,6)`). The handler is gathered whenever the
    /// item is HELD (no hp gate — like Leftovers, the threshold check lives in the apply),
    /// so a full-HP berry holder still ties the residual shuffle exactly like a Leftovers
    /// holder (probe `probe_berry_sub_tie_rng.js` (B): identical draw sequence).
    /// CURE/PP berries register NO residual handler (their gen3 trigger is `onUpdate`).
    BerryResidual { side: usize, slot: usize },
}

/// The per-mon outcome of a turn (what THIS step validates against the sim).
///
/// `crit`/`missed` are surfaced so the differential can assert WHICH RNG outcomes
/// fired (the sim records them via `|-crit|` / `|-miss|`); `acted` is whether the
/// mover got to move at all (false if it was KO'd before its turn — the faint-skip,
/// so its draws are absent).
#[derive(Debug, Clone, Copy, Default)]
pub struct MoveOutcome {
    /// Whether this mover actually moved (false ⇒ KO'd before acting ⇒ no draws).
    pub acted: bool,
    /// Whether this move MISSED (accuracy roll failed). False if it never_miss or hit.
    pub missed: bool,
    /// Whether this move was a critical hit.
    pub crit: bool,
}

/// The result of running one turn: per-side, the active mover's outcome. Index
/// `[0]` is p1's move outcome, `[1]` is p2's (keyed by side, NOT by move order).
#[derive(Debug, Clone, Copy, Default)]
pub struct TurnResult {
    /// Per-side (p1, p2) move outcome.
    pub outcome: [MoveOutcome; 2],
    /// Whether the end-of-turn Quick Claw roll was drawn (i.e. endTurn completed —
    /// no faint this turn). Surfaced so a test can assert the conditional draw.
    pub quick_claw_drawn: bool,
    /// The side (0 = p1, 1 = p2) whose move resolved FIRST — the action-order
    /// outcome (on a speed tie, the Fisher-Yates shuffle's decision). `None` if
    /// neither side had a runnable move.
    pub first_mover: Option<usize>,
}

/// A post-turn snapshot of one side's active mon — the STATE the multi-turn
/// differential asserts each turn (hp/maxhp/fainted/status, incl. the Toxic stage
/// carried on `Status::Toxic`).
#[derive(Debug, Clone, Copy)]
pub struct MonSnapshot {
    pub hp: u16,
    pub maxhp: u16,
    pub fainted: bool,
    pub status: Option<Status>,
    /// The stat-stage boosts `[atk, def, spa, spd, spe, accuracy, evasion]` — so the
    /// per-decision differential asserts a secondary's boost STAGE (Crunch −1 SpD,
    /// Meteor Mash +1 Atk self, Intimidate −1 Atk on entry), not just the seed.
    pub boosts: [i8; BOOST_LEN],
    /// The CONFUSION counter (`None` = not confused; `Some(t)` = `t` turns left) — so
    /// the differential asserts a Water-Pulse confusion was inflicted AND its
    /// `random(2,6)` duration matches (a missing draw would also desync the seed).
    pub confusion: Option<u8>,
    /// The gen-3 PROTECT/DETECT **stall counter** (`MonState::protect_counter`; the
    /// sim's `volatiles.stall.counter`, 0 = no stall volatile) at the decision boundary
    /// — so the protect differential asserts the consecutive-use denominator escalation
    /// (`0→2→4→8→8`) + the reset (a non-protect/switch turn → 0), not just the seed. A
    /// wrong stall draw model also desyncs the SEED; this pins the STATE too.
    pub protect_counter: u8,
    /// Whether this active mon is **LEECH-SEEDED** (`MonState::leech_seed.is_some()`) at the
    /// decision boundary — so the Leech Seed differential asserts the seed LANDS (and
    /// PERSISTS), is cleared on switch-out, and that a Grass-immune / already-seeded /
    /// missed Leech Seed leaves the field correctly. The end-of-turn drain/heal HP is
    /// already asserted via `hp`; this pins the volatile STATE too. (We expose only the
    /// boolean presence — the seeder side is an internal residual detail.)
    pub leech_seeded: bool,
    /// The **SUBSTITUTE** HP (`MonState::substitute`; the sim's `volatiles.substitute.hp`,
    /// `None` = no sub) at the decision boundary — so the Substitute differential asserts
    /// the sub is CREATED at `floor(maxhp/4)`, ABSORBS the exact damage (sub HP drops), and
    /// BREAKS (→ `None`) on a hit ≥ its HP, plus the create FAIL / already-subbed / blocked
    /// status / confusion-self-hit-hits-the-mon cases. The mon's own HP is asserted via `hp`;
    /// this pins the decoy's HP too.
    pub substitute: Option<u16>,
    /// The 4 move slots' CURRENT PP (`gen3_pp_tracking_v1`), padded with `-1` for a mon
    /// with fewer than 4 moves — so the PP/Struggle differential asserts the exact −1 (−2
    /// under Pressure) decrement cadence, that PP PERSISTS across a switch, and that a mon
    /// hits 0 PP right before it is forced to Struggle. A `Vec` would break the `Copy`
    /// derive `TurnRecord` relies on, so this is a fixed `[i16; 4]` (PP fits in i16). Struggle
    /// is not a slot, so it never shows here.
    pub move_pp: [i16; 4],
    /// Whether this active mon is **TAUNTED** (`MonState::taunt.is_some()`, `gen3_taunt_disable_v1`)
    /// at the decision boundary — so the Taunt differential asserts the volatile LANDS, PERSISTS
    /// its FIXED 2-turn duration (residual tick), EXPIRES on schedule, and clears on switch-out.
    /// (We expose only the boolean presence; the exact remaining-turn count is an internal residual
    /// detail whose effect — the Status-move selection restriction — is asserted via the request
    /// legality + the seed.)
    pub taunted: bool,
    /// The **DISABLED** move slot (`MonState::disable` mapped to its slot index, or `-1` = not
    /// disabled, `gen3_taunt_disable_v1`) at the decision boundary — so the Disable differential
    /// asserts WHICH move was disabled (the target's lastMove slot), that it PERSISTS the
    /// random(2,6) duration + `+1`, EXPIRES on schedule, and clears on switch-out. A `-1` = no
    /// disable; a `0..=3` = the disabled slot. (The exact remaining-turn count is internal; its
    /// effect is asserted via the request legality + the seed.)
    pub disabled_slot: i8,
    /// Whether this active mon still HOLDS an item (`MonState::item` non-empty,
    /// `gen3_berry_trace_shedskin_v1`) at the decision boundary — so the berry
    /// differential asserts the EAT timeline (held → eaten-at-the-right-decision →
    /// stays gone; the WHICH-item is fixed per scenario/set, so the boolean is the
    /// full item state). `bool` keeps the `Copy` derive `TurnRecord` relies on.
    pub item_held: bool,
}

/// The record of one turn in a [`BattleState::run_battle`] sequence: the turn
/// number, each side's post-turn active-mon snapshot, the [`TurnResult`], and
/// whether the loop ENDED on this turn because a mon fainted (the bounded step
/// stops at the first faint — no switching modeled).
#[derive(Debug, Clone, Copy)]
pub struct TurnRecord {
    pub turn: u32,
    pub p1: MonSnapshot,
    pub p2: MonSnapshot,
    pub result: TurnResult,
    /// True iff a mon fainted this turn (so the loop stopped here).
    pub ended_on_faint: bool,
}

impl BattleState {
    /// Execute ONE FULL turn cycle where both sides use a damaging move from their
    /// active mon's move slot (`p1_move_slot` / `p2_move_slot`, 0-based into the
    /// set's `moves`): the action-order shuffle, the per-action `eachEvent` shuffles,
    /// each move (accuracy → crit → damage → HP → deferred faint), the end-of-turn
    /// residuals (weather chip / Leftovers / status DoT), and the Quick Claw roll —
    /// all consuming the PRNG in Showdown's EXACT order + count (see the module docs
    /// for the full draw sequence). Increments `turn` by 1.
    ///
    /// # Scope
    ///
    /// Both moves MUST be damaging (base power > 0). DEFERRED: secondaries, status
    /// MOVES, switching (a faint ends the bounded step — no switch), Leech Seed /
    /// Wish / items beyond Leftovers. A non-damaging / unknown move in either slot is
    /// a programming error (the caller picks the slots); it is resolved as a no-op
    /// move that draws nothing for that side.
    pub fn run_turn(&mut self, p1_move_slot: usize, p2_move_slot: usize, dex: &Dex) -> TurnResult {
        self.turn += 1;
        // FLINCH is a `duration:1` volatile that expires at the end of the turn it was
        // set; clearing both actives' flinch at the TOP of each turn guarantees it
        // never blocks a move in a LATER turn (a flinch is only ever checked within
        // the turn it was inflicted, when the flinchee moves second). DRAW-FREE.
        self.clear_flinch();
        // `commitChoices` refreshes the cached `pokemon.speed` at turn start
        // (`battle.js:2494`) — so the per-action `eachEvent` shuffles read the
        // para/boost-aware speed (the residual re-refreshes it post-move).
        self.update_speed(dex);
        let mut result = TurnResult::default();

        // ===================================================================
        // The FULL gen-3 per-turn cycle, consuming the PRNG in Showdown's EXACT
        // order. The queue is `[beforeTurn, move1, move2, residual]`; each action
        // ends (gen<5) with `eachEvent('Update')`. The per-action `eachEvent`
        // speed-tie shuffles (over the 2 active mons) are the draws the single-turn
        // step deferred — they fire ONLY on a speed tie, but must be REQUESTED in
        // the exact place/count so a tie turn AND cross-turn seed parity hold.
        //
        // Verified draw order (harness/trace_multiturn_rng.js), SPEED-TIE case:
        //   [1] action-order shuffle (commitChoices→queue.sort, BEFORE turnLoop)
        //   [2] eachEvent('BeforeTurn') shuffle      (beforeTurn runAction body)
        //   [3] eachEvent('Update') shuffle          (end of beforeTurn runAction)
        //   [4..6] move1 acc/crit/dmg
        //   [7] eachEvent('Update') shuffle          (INSIDE tryMoveHit, gen3 scripts)
        //   [8] eachEvent('Update') shuffle          (end of move1 runAction)
        //   [9..11] move2 acc/crit/dmg
        //   [12] eachEvent('Update') shuffle         (INSIDE tryMoveHit)
        //   [13] eachEvent('Update') shuffle         (end of move2 runAction)
        //   [14] residual fieldEvent handler-sort shuffle (+ nested Weather shuffle
        //        for sandstorm)  [15] eachEvent('Update') (end of residual runAction)
        //   [16] Quick Claw randomChance(1,5)  (endTurn)
        // At DISTINCT speed every shuffle [1][2][3][7][8][12][13][14(eachEvent
        // parts)][15] draws ZERO (the 2 actives never tie), so a distinct-speed turn
        // is exactly `acc/crit/dmg ×2 + QuickClaw` (= the single-turn step's 7).
        // ===================================================================

        // --- [1] ACTION-ORDER shuffle: order the two move actions (priority →
        //     effective speed) via speed_sort — draws on a priority+speed tie. ---
        let order = self.order_actions(p1_move_slot, p2_move_slot, dex);
        result.first_mover = order.first().map(|a| a.side);

        // --- beforeTurn action: [2] eachEvent('BeforeTurn') then [3] the
        //     end-of-runAction eachEvent('Update') (gen<5 tail). ---
        self.each_event_shuffle(); // [2] BeforeTurn (no item onUpdate here — a different event)
        let upd = self.each_event_shuffle(); // [3] end-of-beforeTurn-runAction Update
        self.run_update_items(&upd, dex); // the cure-berry/Leppa onUpdate site (draw-free)

        // --- Run each move action in resolved order. Each landed move is bracketed
        //     by [in-tryMoveHit Update] then [end-of-runAction Update]. ---
        let mut any_faint = false;
        for (idx, action) in order.iter().enumerate() {
            let action = *action;
            // The faint-skip: a mon KO'd before its turn does not move (and draws
            // nothing) — `runAction`'s fainted guard + gen3 cancelAction-all. A
            // fainted actor's runAction is cancelled, so NO trailing Update either.
            if self.sides[action.side].pokemon[action.slot].fainted {
                continue;
            }
            // `willAct()`: a later move action remains in this turn's order (this is the
            // no-switch single-turn path, so the only pending actions are moves). A
            // first-moving Protect has its foe's move still pending → succeeds; a
            // second-moving one would have none — but Protect is priority 3, so it always
            // sorts FIRST vs a normal foe move (this path has no foe switch).
            let will_act = idx + 1 < order.len();
            // `willMove(target)` for Disable's duration `+1` (`gen3_taunt_disable_v1`): does the
            // foe still have a LATER move action this turn? In this no-switch single-turn path the
            // only pending actions are moves, so it is true iff the foe's move comes after `idx`.
            let foe = 1 - action.side;
            let foe_will_move = order[idx + 1..].iter().any(|a| a.side == foe);
            let outcome = self.run_move(action, will_act, foe_will_move, dex);
            let oc = &mut result.outcome[action.side];
            oc.acted = true;
            oc.missed = outcome.missed;
            oc.crit = outcome.crit;

            // The in-tryMoveHit eachEvent('Update') (gen3 scripts.ts:470) fires only
            // when the move actually LANDED (acc-hit + non-immune) — a miss/immune
            // returns from tryMoveHit BEFORE it. (Verified: a miss drops this shuffle
            // AND the crit/dmg draws.) CRUX: this shuffle runs while a KO'd defender
            // is at 0 HP but NOT yet `fainted` (faintMessages hasn't run), so it is
            // still counted by getAllActive() → the 2-active speed-tie shuffle FIRES
            // on a KO turn (verified: the faint turn draws 7, incl. this shuffle).
            if outcome.landed {
                let upd = self.each_event_shuffle(); // [7]/[12] in-tryMoveHit Update
                self.run_update_items(&upd, dex); // cure-berry/Leppa onUpdate (draw-free)
            }

            // faintMessages() at the END of runAction sets the `fainted` flag (now the
            // 0-HP mon is excluded from getAllActive). A faint then reaches
            // makeRequest('switch') and RETURNS *before* the line-2424 trailing Update
            // — so the KO move's end-of-runAction Update does NOT fire, residuals /
            // QuickClaw are DEFERRED, and (no switching modeled) the turn ENDS here.
            if self.process_faints(dex) {
                any_faint = true;
                break;
            }

            // [8]/[13] end-of-move-runAction eachEvent('Update').
            let upd = self.each_event_shuffle();
            self.run_update_items(&upd, dex); // cure-berry/Leppa onUpdate (draw-free)
        }

        if any_faint {
            // A faint ends the bounded step: residuals, the trailing Update, and the
            // endTurn Quick Claw are all deferred behind the (unmodeled) switch
            // request. `quick_claw_drawn` stays false; `ended_on_faint` is the caller
            // signal.
            return result;
        }

        // --- residual action: fieldEvent('Residual') handler-sort shuffle [14] +
        //     the draw-free HP effects (weather chip / Leftovers / status DoT), with
        //     the nested eachEvent('Weather') shuffle for sandstorm, then (no faint)
        //     [15] the end-of-residual-runAction eachEvent('Update'). The residual HP
        //     effects zero HP but (like a move) defer the `fainted` flag to
        //     faintMessages — which here runs AFTER the residual's HP effects but
        //     BEFORE the trailing [15] Update; a faint there requests a switch and
        //     SKIPS [15] (so the 0-HP mon is never seen by [15], and on a tie the [15]
        //     shuffle is not drawn at all). ---
        self.run_residuals(dex);

        // faintMessages() runs at the END of the residual runAction (battle.ts:2856),
        // BEFORE the trailing eachEvent('Update') (battle.ts:2938) — and a faint there
        // makes `makeRequest('switch'); return` SKIP that trailing Update (and the
        // endTurn Quick Claw). `run_residuals` now runs faintMessages PER HANDLER
        // (the e2e-capstone fix), so a residual faint already has `fainted` set; the
        // post-residual `process_faints()` would then return FALSE (nothing NEWLY
        // fainted) — so gate on the STATE (`any_active_fainted`), not the newly-fainted
        // return, or we'd wrongly draw the trailing Update [15] + Quick Claw on a
        // residual-faint turn (the seed desync). A residual (Toxic / burn / poison /
        // sand chip) KO pauses for a switch, deferring BOTH the trailing Update shuffle
        // AND the Quick Claw, and ends the step.
        self.process_faints(dex); // idempotent — already set inside run_residuals
        if self.any_active_fainted() {
            return result;
        }

        // [15] end-of-residual-runAction eachEvent('Update') — only reached when the
        // residual did NOT faint a mon (else the switch request skipped it).
        let upd = self.each_event_shuffle();
        self.run_update_items(&upd, dex); // cure-berry/Leppa onUpdate (draw-free)

        // The endTurn `runEvent('DisableMove')` handler-sort shuffle (a taunt+disable /
        // Choice-lock+disable mon draws one size-2 shuffle, `gen3_taunt_disable_v1`) — BEFORE
        // the Quick Claw. Draw-free for a mon with <2 move-disabling volatiles (the common case
        // in this no-switch single-turn path), so seed-neutral there.
        self.disable_move_event_shuffle();

        // `activeTurns++` per active mon (`endTurn`, battle.ts:1762) — DRAW-FREE, AFTER the
        // DisableMove event + BEFORE the Quick Claw. Feeds NEXT turn's Speed Boost residual
        // gate (`gen3_ability_batch1_v1`). A mon that switched in this turn was reset to 0 in
        // `execute_switch`, so it becomes 1 here (its first boost comes NEXT turn's residual).
        self.bump_active_turns();

        // --- [16] END-OF-TURN Quick Claw randomChance(1,5): UNCONDITIONAL in gen3
        //     once endTurn completes (no faint this turn). ---
        let _ = self.prng.random_chance(1, 5);
        result.quick_claw_drawn = true;

        result
    }

    /// `endTurn`'s per-active `pokemon.activeTurns++` (battle.ts:1762) — DRAW-FREE, run after
    /// the residual/DisableMove and before the Quick Claw. Feeds the Speed Boost residual gate
    /// (`gen3_ability_batch1_v1`). A fainted mon is skipped (`if (pokemon.fainted) continue`).
    fn bump_active_turns(&mut self) {
        for side in 0..2 {
            let slot = self.sides[side].active;
            let mon = &mut self.sides[side].pokemon[slot];
            if !mon.fainted {
                mon.active_turns = mon.active_turns.saturating_add(1);
            }
        }
    }

    /// Play a SCRIPTED sequence of `(p1_move_slot, p2_move_slot)` turns through the
    /// full per-turn cycle ([`BattleState::run_turn`]) + residuals, STOPPING at the
    /// first faint (no switching this step). Returns a per-turn [`TurnRecord`]
    /// (post-turn hp/maxhp/status/fainted both sides + an `ended_on_faint` flag).
    ///
    /// The running PRNG carries across turns — so the post-turn seed must match
    /// Showdown's after EVERY turn, the cross-turn draw-order proof. The loop ends
    /// the turn a mon faints (recording that turn), since the bounded step does not
    /// model the switch a faint requests.
    pub fn run_battle(&mut self, scripted: &[(usize, usize)], dex: &Dex) -> Vec<TurnRecord> {
        let mut records = Vec::with_capacity(scripted.len());
        for &(p1_slot, p2_slot) in scripted {
            let result = self.run_turn(p1_slot, p2_slot, dex);
            let ended_on_faint = self.any_active_fainted();
            records.push(TurnRecord {
                turn: self.turn,
                p1: self.side_snapshot(0),
                p2: self.side_snapshot(1),
                result,
                ended_on_faint,
            });
            if ended_on_faint {
                break; // a faint requests a switch — out of this step's scope
            }
        }
        records
    }

    /// Snapshot one side's active mon (post-turn STATE the differential asserts).
    fn side_snapshot(&self, side: usize) -> MonSnapshot {
        let mon = self.sides[side].active();
        MonSnapshot {
            hp: mon.hp,
            maxhp: mon.maxhp,
            fainted: mon.fainted,
            status: mon.status,
            boosts: mon.boosts,
            confusion: mon.confusion,
            protect_counter: mon.protect_counter,
            leech_seeded: mon.leech_seed.is_some(),
            substitute: mon.substitute,
            move_pp: mon.pp_array(),
            taunted: mon.taunt.is_some(),
            disabled_slot: mon.disable.map(|(k, _)| k as i8).unwrap_or(-1),
            item_held: !mon.item.is_empty(),
        }
    }

    /// `eachEvent('Update'/'BeforeTurn'/'Weather')` (`battle.ts:465-475`): a
    /// `speedSort(getAllActive(), (a,b)=>b.speed-a.speed)` over the NON-fainted
    /// active mons — which draws ONE `random(0,2)` Fisher-Yates shuffle iff the two
    /// actives TIE on the CACHED speed, and NOTHING otherwise. gen3's `Update`/
    /// `BeforeTurn`/`Weather` events have no DRAWING handlers beyond this shuffle —
    /// the CURE-berry / Leppa `onUpdate` handlers (`gen3_berry_trace_shedskin_v1`)
    /// are DRAW-FREE and applied by the caller via [`Self::run_update_items`] at the
    /// Update sites only (never BeforeTurn / Weather / WeatherChange).
    ///
    /// **It reads `cached_speed` (`pokemon.speed`), NOT the live `effective_speed`**
    /// (`battle.js:296` — `speedSort(actives, (a,b)=>b.speed-a.speed)`). The two differ
    /// for a mon paralyzed/boosted WHILE active SINCE the last `updateSpeed` site (turn
    /// start / residual): its cached speed stays at the turn-start value through the
    /// move-phase shuffles and only drops at the residual — so a mon that gets paralyzed
    /// mid-turn still ties the shuffles on its FULL speed until the residual (the e2e-
    /// capstone seed-desync the cached-speed model fixes). A mon that SWITCHES IN gets
    /// its current para/boost-aware speed established immediately (see `execute_switch`),
    /// so a just-switched-in PARALYZED mon ties on its PARA speed.
    ///
    /// The PRNG consumption is what matters for seed parity (Showdown sorts a throwaway
    /// list in place and discards it) — but the RESOLVED order IS observable in the
    /// emitted `|-damage|` line order of the weather chip (a same-species speed TIE chips
    /// in the shuffled side order, e.g. a Snorlax-vs-Snorlax mirror). So this RETURNS the
    /// speed-sorted side order (the `handler: side` payloads after `speed_sort`), which
    /// the weather-chip emitter reads to order its per-active `|-damage|` lines. Callers
    /// that only need the DRAW ignore the return. A side with a fainted active
    /// (`getAllActive` excludes it) collapses to <2 handlers ⇒ no draw — matching the sim.
    fn each_event_shuffle(&mut self) -> Vec<usize> {
        let mut handlers: Vec<EventHandler<usize>> = (0..2)
            .filter(|&side| {
                let a = self.sides[side].active;
                !self.sides[side].pokemon[a].fainted
            })
            .map(|side| {
                let slot = self.sides[side].active;
                EventHandler {
                    order: NO_ORDER,
                    priority: 0,
                    speed: self.sides[side].pokemon[slot].cached_speed as f64,
                    sub_order: 0,
                    effect_order: 0,
                    handler: side,
                }
            })
            .collect();
        // speed_sort early-returns for <2 handlers (no draw), and draws exactly one
        // size-2 shuffle on a speed tie — bit-identical to eachEvent's speedSort.
        speed_sort(&mut handlers, &mut self.prng);
        handlers.into_iter().map(|h| h.handler).collect()
    }

    /// The `runEvent('SetStatus')` HANDLER-SORT SHUFFLE — the gen3ou-only draw fired
    /// every time a status APPLICATION reaches the SetStatus event (passed hp /
    /// already-statused / type-immunity). The 2 `Standard` format clauses (Sleep Clause
    /// Mod + Freeze Clause Mod) register `onSetStatus` handlers at EQUAL order/priority/
    /// **speed** (Format effects carry no speed → speed 0), so they ALWAYS tie → a
    /// size-2 Fisher-Yates speed-sort shuffle draws EXACTLY one `random(0,2)` (modeled
    /// by `shuffle_range` over 2 sentinels, bit-identical to `each_event_shuffle`'s tie
    /// shuffle). Only the caller-gated `sleep_clause` formats fire this; gen3customgame
    /// (0 handlers) draws nothing.
    fn set_status_event_shuffle(&mut self) {
        // Two tied handlers (the 2 Standard clauses) at equal everything → speed_sort
        // draws one size-2 shuffle. The handler payload is irrelevant (we only need the
        // DRAW); reuse the generic EventHandler with equal keys.
        let mut handlers: Vec<EventHandler<usize>> = (0..2)
            .map(|i| EventHandler {
                order: NO_ORDER,
                priority: 0,
                speed: 0.0,
                sub_order: 0,
                effect_order: 0,
                handler: i,
            })
            .collect();
        speed_sort(&mut handlers, &mut self.prng);
    }

    /// The `endTurn` `runEvent('DisableMove', pokemon)` HANDLER-SORT SHUFFLE
    /// (`gen3_taunt_disable_v1`). `endTurn` (battle.ts:1683) runs, for EACH active mon in
    /// ARRAY order (p1 active then p2 active), `runEvent('DisableMove', pokemon)` — the
    /// per-turn recompute that re-applies the mon's move-disabling volatiles. That event
    /// gathers the mon's `onDisableMove` handlers (**taunt** disables every Status move,
    /// **disable** disables the one slot, **choicelock** [Choice Band] disables the non-locked
    /// slots). If a mon carries ≥2 of these volatiles at once, its handlers all TIE (same
    /// Condition order=false / priority 0 / SAME mon's speed / subOrder) → `speedSort` shuffles
    /// the tie group, drawing `n-1` `random(range)` calls (a size-2 group = 1 draw). Fires per
    /// mon INDEPENDENTLY, in array order. It runs in `endTurn` AFTER the residual, BEFORE the
    /// Quick Claw roll (`if gen===3 quickClawRoll = randomChance(1,5)`), so the draw order is
    /// `… residual … → [DisableMove shuffle per mon] → Quick Claw`. A mon with 0 or 1 such
    /// volatile draws NOTHING. VERIFIED vs the sim (`harness/probe_taunt_disable_rng.js` +
    /// `probe_disable_full_lifecycle.js`): a taunt+disable mon draws one size-2 shuffle at
    /// endTurn; a taunt-only / disable-only mon draws none. This is the ONLY `DisableMove`-event
    /// draw (`singleEvent('DisableMove', …)` per moveslot is a no-sort/no-RNG `singleEvent`).
    fn disable_move_event_shuffle(&mut self) {
        // Per active mon in ARRAY order (p1 then p2), mirroring `for pokemon of getAllActive()`.
        // The sim's endTurn per-mon loop runs `runEvent('DisableMove')` THEN the TRAPPING
        // events (`runEvent('TrapPokemon')` + `runEvent('MaybeTrapPokemon')`,
        // `gen3_trapping_v1`) for EACH mon before moving to the next — so the two shuffle
        // families INTERLEAVE per mon (battle.ts:1689/1723-1726), all BEFORE the gen3
        // quickClawRoll draw (battle.ts:1795).
        for side in 0..2 {
            let slot = self.sides[side].active;
            let mon = &self.sides[side].pokemon[slot];
            if mon.fainted {
                continue;
            }
            // Count the mon's move-disabling volatiles carrying an `onDisableMove` handler.
            let n = (mon.taunt.is_some() as usize)
                + (mon.disable.is_some() as usize)
                + (mon.choice_locked_move.is_some() as usize);
            if n >= 2 {
                // The n handlers all tie (same mon's speed + Condition sort key) → speed_sort
                // shuffles a size-n tie group, drawing n-1 `random(range)` calls. Model it with
                // n equal-keyed sentinel handlers so the DRAW count is bit-identical.
                let mut handlers: Vec<EventHandler<usize>> = (0..n)
                    .map(|i| EventHandler {
                        order: NO_ORDER,
                        priority: 0,
                        speed: self.sides[side].pokemon[slot].cached_speed as f64,
                        sub_order: 0,
                        effect_order: 0,
                        handler: i,
                    })
                    .collect();
                speed_sort(&mut handlers, &mut self.prng);
            }
            // [gen3_trapping_v1] The TRAPPING runEvents for THIS mon (right after its
            // DisableMove event, mirroring the sim's per-mon endTurn loop).
            self.trap_event_shuffles(side);
        }
    }

    /// The endTurn `runEvent('TrapPokemon')` + `runEvent('MaybeTrapPokemon')` handler-sort
    /// shuffles for ONE active mon (`gen3_trapping_v1`) — the ONLY PRNG the trapping layer
    /// can consume (the trapped COMPUTATION itself and the switch-choice REJECTION are
    /// draw-free; VERIFIED vs the omniscient sim `harness/probe_trapping_rng.js`).
    ///
    /// The handler matrix per event on mon X (singles), from `findEventHandlers`:
    ///   - X's OWN ability contributes iff it has an `onAny<Event>` handler — gen-3
    ///     **Magnet Pull** is overridden to `onAnyTrapPokemon`/`onAnyMaybeTrapPokemon`
    ///     (data/mods/gen3/abilities.ts), gathered via `alliesAndSelf()` (its body
    ///     no-ops on `isAdjacent(self, self) === false`, but the handler still SORTS);
    ///   - the FOE active's ability contributes via `onFoe<Event>` (**Arena Trap**:
    ///     `onFoeTrapPokemon`/`onFoeMaybeTrapPokemon`, base data) or `onAny<Event>`
    ///     (Magnet Pull again).
    /// Both events carry the IDENTICAL matrix (each ability registers both its Trap and
    /// its MaybeTrap callback), and gen-3 has NO `trapped` type-immunity (the gen3 dex
    /// resolves Ghost `damageTaken.trapped` = undefined — a grounded Ghost IS trapped,
    /// probe-verified) and no Illusion (`knownType` always true), so `MaybeTrapPokemon`
    /// ALWAYS runs after `TrapPokemon` (battle.ts:1725 gate passes).
    ///
    /// With >= 2 handlers the sort ties iff the holders' cached `pokemon.speed` are equal
    /// (abilities share order/priority/subOrder; `effectOrder` is only resolved for
    /// SwitchIn/RedirectTarget callbacks) → ONE Fisher-Yates draw per tied event. The
    /// MAGNETON MIRROR (both actives Magnet Pull, equal speeds) therefore draws
    /// **4 per endTurn** (2 events × 2 mons, 1 each — probed 11 vs the Sturdy-control's
    /// 7 draws/turn, seed-verified); the DUGTRIO MIRROR (Arena Trap is `onFoe`-only → 1
    /// handler per event) draws ZERO (probed byte-identical seeds to a Sand Veil control).
    fn trap_event_shuffles(&mut self, side: usize) {
        let foe_side = 1 - side;
        let me = &self.sides[side].pokemon[self.sides[side].active];
        let foe = &self.sides[foe_side].pokemon[self.sides[foe_side].active];
        // Handler speeds for ONE trap event on `me` (same matrix for both events).
        let mut speeds: Vec<f64> = Vec::with_capacity(2);
        // alliesAndSelf: own onAny* — gen-3 Magnet Pull only.
        if to_id(&me.ability) == "magnetpull" {
            speeds.push(me.cached_speed as f64);
        }
        // foes(): onFoe* (Arena Trap) + onAny* (Magnet Pull) — `foes()` filters fainted.
        if !foe.fainted {
            let foe_ability = to_id(&foe.ability);
            if foe_ability == "arenatrap" || foe_ability == "magnetpull" {
                speeds.push(foe.cached_speed as f64);
            }
        }
        if speeds.len() < 2 {
            return; // 0/1 handlers → no sort tie possible → draw-free (the common case)
        }
        // TrapPokemon then MaybeTrapPokemon — each a speed_sort over the same handlers.
        for _ in 0..2 {
            let mut handlers: Vec<EventHandler<usize>> = speeds
                .iter()
                .enumerate()
                .map(|(i, &speed)| EventHandler {
                    order: NO_ORDER,
                    priority: 0,
                    speed,
                    sub_order: 0,
                    effect_order: 0,
                    handler: i,
                })
                .collect();
            speed_sort(&mut handlers, &mut self.prng);
        }
    }

    /// Whether `side`'s active mon is TRAPPED (`gen3_trapping_v1`) — barred from a
    /// VOLUNTARY switch-out by the FOE active's trapping ability. The port's equivalent
    /// of the sim's `pokemon.trapped` truthiness (both abilities call `tryTrap(true)` →
    /// `trapped = 'hidden'`), which `side.choose` reads to REJECT a `switch N` at a
    /// `move` request ("Can't switch: The active Pokémon is trapped") — DRAW-FREE.
    ///
    /// Gen-3 semantics, VERIFIED vs the omniscient sim (`harness/probe_trapping_rng.js`):
    ///   - **Arena Trap** (foe): traps a GROUNDED mon — a Flying-type or Levitate holder
    ///     escapes ('isGrounded()`; gen-3 has no Gravity/Iron Ball/Magnet Rise/roost, so
    ///     grounded == not-Flying && not-Levitate, the spikes rule). A grounded GHOST
    ///     **IS** trapped in Showdown-gen3 (the gen3 dex has NO `trapped` type-immunity —
    ///     Ghost `damageTaken.trapped` resolves undefined; the cartridge gen6+ immunity
    ///     does not exist here; probed: Sableye's switch is rejected).
    ///   - **Magnet Pull** (foe): traps a STEEL-type mon — groundedness IRRELEVANT
    ///     (Skarmory, Steel/Flying, is trapped; probed).
    ///   - MIRRORS are MUTUAL: two Dugtrio trap each other (both grounded); two Magneton
    ///     trap each other (both Steel). A mon does NOT trap itself (`isAdjacent(self)`
    ///     is false in singles), so a lone Magnet Pull holder is only trapped if the FOE
    ///     traps it.
    ///   - The flag gates ONLY the voluntary switch: a PHAZE drag (Roar/Whirlwind) still
    ///     moves a trapped mon, a FAINTED mon's forced replacement is accepted (the sim's
    ///     check is `requestState === 'move'`-only), and the TRAPPING mon itself switches
    ///     freely (probed).
    ///
    /// Computed LIVE from the current actives — identical to the sim's endTurn-cached
    /// value at every MOVE-request boundary (nothing changes between endTurn and the
    /// request; `foes()` never yields a fainted foe there). The `dex` resolves the mon's
    /// types (Flying/Steel) from its species.
    pub fn is_trapped(&self, side: usize, dex: &Dex) -> bool {
        let me = &self.sides[side].pokemon[self.sides[side].active];
        let foe = &self.sides[1 - side].pokemon[self.sides[1 - side].active];
        if me.fainted || foe.fainted {
            return false; // isAdjacent() is false for a fainted mon — no trapping
        }
        match to_id(&foe.ability).as_str() {
            "arenatrap" => {
                // Grounded-only: Flying-type / Levitate escape (the spikes grounded rule).
                let types = mon_types(me, dex);
                let is_flying = types.contains(&Type::Flying);
                let is_levitate = to_id(&me.ability) == "levitate";
                !(is_flying || is_levitate)
            }
            "magnetpull" => mon_types(me, dex).contains(&Type::Steel),
            // SHADOW TAG (`gen3_ability_batch4_v1`): traps UNCONDITIONALLY — no grounded
            // gate (a Flying-type / Levitate holder IS trapped) and no type gate. A
            // Shadow-Tag MIRROR is MUTUALLY trapped (`onFoeTrapPokemon` has NO
            // fellow-holder exemption in gen3 — only the display-only
            // `onFoeMaybeTrapPokemon` skips ST holders). DRAW-FREE (onFoe* handlers add
            // ZERO draws — a Wobbuffet mirror's per-turn draw count is IDENTICAL to a
            // no-trap control; probe `probe_shadowtag_rng.js`, vs the Arena-Trap 0-draw
            // and Magnet-Pull onAny* precedents).
            "shadowtag" => true,
            _ => false,
        }
    }

    /// Whether this side's active is trapped by a **FIRM** (`trapped === true`) trap
    /// — as opposed to the `'hidden'` (`maybeTrapped`) traps. This drives the bridge
    /// `|request|` flag: the sim's gen3 **Shadow Tag** override sets `pokemon.trapped =
    /// true` DIRECTLY in `onFoeTrapPokemon` (`data/mods/gen3/abilities.ts` — NOT the
    /// base `tryTrap(true)` → `'hidden'`), so `getMoveRequestData`'s `trapped === true`
    /// branch fires and the FIRST `move` request already carries `trapped:true` (no
    /// `maybeTrapped` phase, no rejection round). Arena Trap / Magnet Pull call
    /// `tryTrap(true)` → `trapped = 'hidden'`, so they show `maybeTrapped` until a
    /// rejected switch firms them. Probe-settled: a Shadow-Tag foe's first p2 request is
    /// `trapped`, an Arena-Trap / Magnet-Pull foe's is `maybe` (the request/per-side A/B
    /// fuzzer's #1 find — `bridge_ab_fuzz.js`). Draw-neutral (display-only, like
    /// `is_trapped`).
    pub fn trap_is_firm(&self, side: usize, dex: &Dex) -> bool {
        if !self.is_trapped(side, dex) {
            return false;
        }
        let foe = &self.sides[1 - side].pokemon[self.sides[1 - side].active];
        to_id(&foe.ability) == "shadowtag"
    }

    /// `updateSpeed()` (`battle.js:241` / `pokemon.js:283`): refresh BOTH actives'
    /// cached `pokemon.speed` to the live `getActionSpeed()` (para/boost/ModifySpe-
    /// aware). Showdown refreshes the cached speed at `commitChoices` (turn start,
    /// `battle.js:2494`) and the start of the `residual` action (`battle.js:2342`); the
    /// entrant's speed is also (re)established on SWITCH-IN (see `execute_switch`).
    /// Between those sites the cached value is STALE — a mon paralyzed mid-turn keeps its
    /// turn-start (full) speed for the rest of the move phase, which is what the
    /// `eachEvent` shuffles read. DRAW-FREE.
    fn update_speed(&mut self, dex: &Dex) {
        for side in 0..2 {
            let slot = self.sides[side].active;
            self.sides[side].pokemon[slot].cached_speed = self.effective_speed(side, slot, dex);
        }
    }

    /// The gen-3 end-of-turn RESIDUAL phase (`residual` action → `fieldEvent('Residual')`,
    /// `battle.ts:2832-2839`). Builds the residual handler list, sorts it by the full
    /// `comparePriority` key with the Fisher-Yates tie-shuffle DRAW (the only residual
    /// draw besides the nested weather shuffle), then applies each handler's HP effect
    /// in resolved order — all DRAW-FREE arithmetic.
    ///
    /// Gen-3 residualOrder ladder (smaller `order` first, via the gen4-mod overrides
    /// gen3 inherits): **sandstorm/hail field-residual order 8** → **Leftovers order
    /// 10, subOrder 4** → **burn/poison/Toxic order 10, subOrder 6**. So Leftovers
    /// heals BEFORE the status DoT; the weather chip resolves first. Equal-key
    /// handlers (e.g. both sides' Leftovers at equal speed) tie → the speed-tie
    /// shuffle draws.
    ///
    /// Verified values (gen6/gen4 mod overrides gen3 inherits): burn = `maxhp/8`,
    /// poison = `maxhp/8`, Toxic = `max(1, floor(maxhp/16)) * stage` (stage ramps
    /// 1..15), Leftovers = `floor(maxhp/16)`, sand chip = `max(1, floor(maxhp/16))`
    /// to non-Rock/Ground/Steel. `clampIntRange(x,1) = max(1, floor(x))` for damage;
    /// heal floors; HP capped at `[0, maxhp]`.
    fn run_residuals(&mut self, dex: &Dex) {
        // The `residual` action refreshes the cached `pokemon.speed` FIRST
        // (`battle.js:2342` `this.updateSpeed()` before `fieldEvent("Residual")`), so
        // the handler-sort below + the nested `eachEvent('Weather')` shuffle tie on the
        // CURRENT (para/boost-aware) speed — even for a mon that switched in mid-turn
        // with a stale raw cached speed.
        self.update_speed(dex);

        // --- Gather residual handlers (one per active per applicable effect), each
        //     with its resolved comparePriority key + a typed action. ---
        let mut handlers: Vec<EventHandler<ResidualAction>> = Vec::new();

        // The weather FIELD handler (`onFieldResidual`, order 8) sorts FIRST. It has no
        // holder speed (a Field handler), so its `speed` key is 0 — it never ties a
        // mon-held handler (those are order 10). One handler for the whole field. It fires
        // the nested `eachEvent('Weather')` speed-tie shuffle (`apply_weather_chip` calls
        // `each_event_shuffle` → one `random(0,2)` on a speed TIE, zero otherwise), then chips
        // the non-immune actives.
        //
        // **Sun/rain fire the shuffle UNDER ALL weather — even a WEATHER_NEGATE mon — while
        // sand/hail are SUPPRESSED by a negater. The exact split, PROBE-VERIFIED against the
        // resolved `Dex.mod('gen3')` (`gen3_ability_batch1_v1`, the STEP-1 fix):** the gen-3
        // `onFieldResidual` bodies are
        //   - sunnyday/raindance: `this.add('-weather',…,'[upkeep]'); this.eachEvent('Weather');`
        //     — the `eachEvent` is UNCONDITIONAL (probe_weather_eachevent_sunrain.js);
        //   - sandstorm/hail: `…; if (this.field.isWeather('<w>')) this.eachEvent('Weather');`
        //     — the `isWeather` guard reads `effectiveWeather()`, so a Cloud Nine / Air Lock mon
        //     SUPPRESSES the shuffle (AND the chip). The guard is NOT redundant: it is exactly
        //     what silences sand/hail under a negater.
        // So the shuffle-scheduling weather source DIFFERS by weather: RAW `field.weather` for
        // sun/rain (fires even when suppressed), `effective_weather()` for sand/hail (suppressed
        // by a negater). VERIFIED full matrix (Cloud-Nine Δdraw on a tie): rain +1, sun +1,
        // sand 0, hail 0 (== the sim). The prior `Sand | Hail`-only gate MISSED the sun/rain
        // shuffle entirely (a 1-draw desync on a sun/rain weather-turn tie).
        //
        // `apply_weather_chip` ALWAYS fires the shuffle (the draw), then chips only the
        // non-immune actives — `weather_immune` returns `true` for EVERY mon under Rain/Sun (they
        // have no chip), so the sun/rain path draws the shuffle + emits the `-weather …|[upkeep]`
        // tick but applies ZERO chip HP, matching the sim. So ONE `WeatherChip(w)` handler serves
        // both weather families; only its SCHEDULING gate differs.
        //
        // WEATHER_NEGATE draw-safety (probe_weather_negate_residual.js): under a negater a
        // sand/hail handler is NOT scheduled (no chip, no shuffle) — bit-for-bit with the sim's
        // `isWeather`-false skip — and the field-order-8 handler is in its OWN sort group anyway
        // (never ties an order-10 mon handler), so a sun/rain handler that IS scheduled under a
        // negater still can't perturb any mon-held tie-shuffle.
        let raw_weather = self.field.weather;
        let sched_weather = match raw_weather {
            // Sun/rain: eachEvent('Weather') is unconditional → schedule off RAW weather
            // (fires even under a WEATHER_NEGATE mon).
            Some(w @ (Weather::Sun | Weather::Rain)) => Some(w),
            // Sand/hail: gated on `isWeather` == `effectiveWeather()` → schedule only when NOT
            // suppressed (a negater silences both the chip AND the shuffle).
            Some(Weather::Sand) | Some(Weather::Hail) => self.effective_weather(dex),
            None => None,
        };
        if let Some(w) = sched_weather {
            handlers.push(EventHandler {
                order: WEATHER_RESIDUAL_ORDER,
                priority: 0,
                speed: 0.0,
                sub_order: WEATHER_RESIDUAL_SUBORDER,
                effect_order: 0,
                handler: ResidualAction::WeatherChip(w),
            });
        }

        // Per-active mon-held handlers: the status DoT (order 10 sub 6) and Leftovers
        // (order 10 sub 4). The handler `speed` is `pokemon.speed` (the CACHED value,
        // `battle.js:767`) — fresh here because the `residual` action ran `updateSpeed()`
        // first (the caller above), so it equals the live speed at residual time (faster
        // sorts first; an equal-speed tie draws the shuffle).
        //
        // **GATHER ORDER MATTERS (the e2e_190 fix):** the speed-sort is a SELECTION SORT
        // whose tie-group Fisher-Yates shuffle PERMUTES the tied handlers IN THEIR
        // PRE-SORT ORDER — so for a SAME shuffle draw, a different gather order yields a
        // DIFFERENT permutation. `findPokemonEventHandlers` (battle.ts) gathers a mon's
        // handlers as **STATUS first, then volatiles, then ability, then ITEM** — so the
        // status DoT MUST be pushed BEFORE Leftovers for each mon (the subOrder still
        // sorts Leftovers ahead in the final order, but the SHUFFLE of a 2-mon DoT tie
        // [or a 2-mon Leftovers tie] reads the gather order). Mirror it: DoT then
        // Leftovers, per mon, side 0 then side 1.
        for side in 0..2 {
            let slot = self.sides[side].active;
            let mon = &self.sides[side].pokemon[slot];
            if mon.fainted {
                continue; // getAllActive excludes a fainted mon
            }
            let speed = mon.cached_speed as f64;

            // The major-status DoT. Order 10, subOrder 6. (Gathered FIRST, mirroring
            // `findPokemonEventHandlers`'s status-before-item order — the shuffle reads it.)
            match mon.status {
                Some(Status::Burn) | Some(Status::Poison) | Some(Status::Toxic(_)) => {
                    handlers.push(EventHandler {
                        order: STATUS_RESIDUAL_ORDER,
                        priority: 0,
                        speed,
                        sub_order: STATUS_DOT_SUBORDER,
                        effect_order: 0,
                        handler: ResidualAction::StatusDot { side, slot },
                    });
                }
                _ => {}
            }

            // --- The LEECH SEED volatile's drain handler. Order 10, subOrder 5 (gen4-mod
            //     override gen3 inherits) — so the final order is Leftovers(4) → LEECH(5) →
            //     status DoT(6) at order 10. It is a VOLATILE (`leechseed`), so it is
            //     gathered HERE — AFTER the status DoT but BEFORE the item (Leftovers) — to
            //     mirror `findPokemonEventHandlers`'s status→volatiles→ability→item gather
            //     order (the tie-shuffle permutes handlers in pre-sort/gather order). The
            //     ONLY tie a leech handler can take is the OTHER mon's leech at equal speed
            //     (both sides seeded, equal cached speed), so the relative gather order vs
            //     the same-mon protect/stall/flinch volatiles below is unobservable (those
            //     are NO_ORDER; leech is order 10 → never ties them). The seeder side is
            //     stored; the apply reads the seeder's CURRENT active (`getAtSlot`). ---
            if let Some(seeder_side) = mon.leech_seed {
                handlers.push(EventHandler {
                    order: STATUS_RESIDUAL_ORDER,
                    priority: 0,
                    speed,
                    sub_order: LEECH_SEED_SUBORDER,
                    effect_order: 0,
                    handler: ResidualAction::LeechSeed { side, slot, seeder_side },
                });
            }

            // --- The TAUNT volatile's duration handler (`gen3_taunt_disable_v1`). Order 10,
            //     subOrder 15 (the gen-3 `taunt` condition's `onResidualOrder: 10,
            //     onResidualSubOrder: 15`) — so at order 10 it sorts AFTER Leftovers (4) / Leech
            //     Seed (5) / status DoT (6). It is a VOLATILE, gathered in the volatiles group
            //     (after leech, before the NO_ORDER protect/stall/flinch + before the item). It
            //     has no `onResidual` HP effect (only ticks the duration + `-end`), so the apply
            //     just decrements + expires. Its ONLY tie is the OTHER mon's taunt at equal
            //     speed (subOrder 15 is unique among order-10 handlers), so its gather position is
            //     otherwise unobservable. VERIFIED order via the sim's residual handler dump. ---
            if mon.taunt.is_some() {
                handlers.push(EventHandler {
                    order: STATUS_RESIDUAL_ORDER,
                    priority: 0,
                    speed,
                    sub_order: TAUNT_RESIDUAL_SUBORDER,
                    effect_order: 0,
                    handler: ResidualAction::TauntDuration { side, slot },
                });
            }

            // --- DURATION-ONLY VOLATILE handlers (`protect` / `stall`) — gathered AFTER
            //     the status DoT but BEFORE the item (the status→volatiles→ability→item
            //     order of `findPokemonEventHandlers`). They have NO `onResidual` callback;
            //     `findPokemonEventHandlers(..., 'duration')` adds them solely to decrement
            //     the volatile's `duration`. They participate in the speed-sort (order
            //     NO_ORDER — sorts AFTER the order-10 Leftovers/DoT — priority 0, speed =
            //     cached, subOrder = the effect's `effectTypeOrder` = 2 for a Condition), so
            //     a protecting mon ADDS 2 tied handlers (protect + stall) that change the
            //     tie-group shuffle COUNT (the e2e/golden seed-desync this models). The
            //     APPLY is a no-op (the duration countdown is the turn-top `clear_flinch`
            //     reset). Gathered in the volatile INSERTION order: `protect` (the move's
            //     `volatileStatus`, added first) then `stall` (added by `onHit`). Order
            //     within the gather is load-bearing — the shuffle permutes in pre-sort order.
            if mon.protected {
                handlers.push(EventHandler {
                    order: NO_ORDER,
                    priority: 0,
                    speed,
                    sub_order: VOLATILE_RESIDUAL_SUBORDER,
                    effect_order: 0,
                    handler: ResidualAction::VolatileDuration { side, slot, is_stall: false },
                });
            }
            if mon.protect_counter > 0 {
                handlers.push(EventHandler {
                    order: NO_ORDER,
                    priority: 0,
                    speed,
                    sub_order: VOLATILE_RESIDUAL_SUBORDER,
                    effect_order: 0,
                    handler: ResidualAction::VolatileDuration { side, slot, is_stall: true },
                });
            }
            // The FLINCH volatile (`duration: 1`) ALSO registers a residual duration
            // handler (gathered by `findPokemonEventHandlers(..., 'duration')`, same as
            // protect/stall — VERIFIED vs the sim: a flinched mon's residual list carries a
            // `flinch` handler). order NO_ORDER, subOrder 2 — so it TIES with a protect/
            // stall handler at the same speed (the new tie a protecting+flinching board
            // creates). The flinch is cleared at the next turn-top (`clear_flinch`), so the
            // apply here is a no-op (a `VolatileDuration` with `is_stall: false`). A
            // protected mon blocked the hit so can't ALSO flinch — flinch is alone among a
            // mon's NO_ORDER volatiles, so its relative gather order is unobservable. NOTE:
            // confusion has NO `duration` (its counter is `effectState.time`), so it does
            // NOT register a residual handler — only flinch/protect/stall do.
            if mon.flinch {
                handlers.push(EventHandler {
                    order: NO_ORDER,
                    priority: 0,
                    speed,
                    sub_order: VOLATILE_RESIDUAL_SUBORDER,
                    effect_order: 0,
                    handler: ResidualAction::VolatileDuration { side, slot, is_stall: false },
                });
            }
            // --- The DISABLE volatile's duration handler (`gen3_taunt_disable_v1`). Order
            //     NO_ORDER, subOrder 2 (the gen-3 `disable` condition has NO `onResidualOrder`/
            //     `onResidualSubOrder`, so `resolvePriority` falls back to order=false and the
            //     "Condition" effectType subOrder 2 — VERIFIED via the source). So it ties with
            //     the same-mon protect/stall/flinch handlers AND the other mon's NO_ORDER handlers
            //     at equal speed (the same tie-group + shuffle-count machinery). It has no
            //     `onResidual` HP effect (only ticks the duration + `-end`), so the apply just
            //     decrements + frees the disabled move. A mon disabled by a FOE rarely also
            //     protects, so the same-mon collision is rare; gathered here in the volatiles
            //     group (after flinch, before the item). ---
            if mon.disable.is_some() {
                handlers.push(EventHandler {
                    order: NO_ORDER,
                    priority: 0,
                    speed,
                    sub_order: VOLATILE_RESIDUAL_SUBORDER,
                    effect_order: 0,
                    handler: ResidualAction::DisableDuration { side, slot },
                });
            }

            // --- RESIDUAL ABILITY handlers (`gen3_ability_batch1_v1`, Speed Boost / Rain
            //     Dish). Order 10, subOrder **3** — so BEFORE Leftovers (4) / leech (5) /
            //     status DoT (6) at order 10. They are ABILITY handlers, gathered in the
            //     ability group (after volatiles, before the item) per
            //     `findPokemonEventHandlers`'s status→volatiles→ability→item order. DRAW-FREE.
            //     Only ONE ability member can be present per mon (an ability is singular), so
            //     the gather is a single push. ---
            if let Some(ab) = dex.ability(&mon.ability) {
                if to_id(&mon.ability) == "speedboost" {
                    handlers.push(EventHandler {
                        order: STATUS_RESIDUAL_ORDER,
                        priority: 0,
                        speed,
                        sub_order: RESIDUAL_ABILITY_SUBORDER,
                        effect_order: 0,
                        handler: ResidualAction::SpeedBoost { side, slot },
                    });
                } else if to_id(&mon.ability) == "raindish" {
                    handlers.push(EventHandler {
                        order: STATUS_RESIDUAL_ORDER,
                        priority: 0,
                        speed,
                        sub_order: RESIDUAL_ABILITY_SUBORDER,
                        effect_order: 0,
                        handler: ResidualAction::RainDish { side, slot },
                    });
                } else if ab.shed_skin {
                    // SHED SKIN (`gen3_berry_trace_shedskin_v1`): order 10 subOrder 3, the
                    // SAME ability slot as Speed Boost / Rain Dish. Gathered UNCONDITIONALLY
                    // (the ability's onResidual registers regardless of status — the status
                    // gate lives in the apply), so an unstatused holder still participates
                    // in the residual tie-shuffle.
                    handlers.push(EventHandler {
                        order: STATUS_RESIDUAL_ORDER,
                        priority: 0,
                        speed,
                        sub_order: RESIDUAL_ABILITY_SUBORDER,
                        effect_order: 0,
                        handler: ResidualAction::ShedSkin { side, slot },
                    });
                } else if to_id(&mon.ability) == "truant" {
                    // TRUANT (`gen3_ability_batch4_v1`): order **27** — its own group,
                    // AFTER every order-10 handler, BEFORE the NO_ORDER duration
                    // volatiles. Gathered UNCONDITIONALLY for an active holder (the
                    // parity clock ticks on move and loaf turns alike). Only a Truant
                    // MIRROR can tie it (one extra shuffle draw — probe Q4).
                    handlers.push(EventHandler {
                        order: TRUANT_RESIDUAL_ORDER,
                        priority: 0,
                        speed,
                        sub_order: RESIDUAL_ABILITY_SUBORDER,
                        effect_order: 0,
                        handler: ResidualAction::TruantToggle { side, slot },
                    });
                }
            }

            // Leftovers (heal). Order 10, subOrder 4 (gen4-mod override gen3 inherits).
            // (Gathered AFTER the status DoT — the ITEM-after-status order.)
            if to_id(&mon.item) == "leftovers" {
                handlers.push(EventHandler {
                    order: STATUS_RESIDUAL_ORDER,
                    priority: 0,
                    speed,
                    sub_order: LEFTOVERS_SUBORDER,
                    effect_order: 0,
                    handler: ResidualAction::Leftovers { side, slot },
                });
            } else if matches!(
                dex.item(&to_id(&mon.item)).and_then(|i| i.berry_effect.as_ref()),
                Some(
                    crate::dex::BerryEffect::HealFixed { .. }
                        | crate::dex::BerryEffect::HealFrac { .. }
                        | crate::dex::BerryEffect::PinchBoost { .. }
                        | crate::dex::BerryEffect::StarfRandom2
                        | crate::dex::BerryEffect::LansatFocusEnergy
                )
            ) {
                // A held HEAL/PINCH berry (`gen3_berry_trace_shedskin_v1`): order 10
                // subOrder 4 — the SAME sort key as Leftovers, gathered in the SAME item
                // position (a mon holds ONE item, so Leftovers-vs-berry same-mon is
                // impossible; a 2-mon equal-speed berry-vs-Leftovers board tie-shuffles
                // exactly like a Leftovers mirror — probe_berry_sub_tie_rng.js (B)).
                // Gathered whenever HELD (the hp-threshold gate lives in the apply).
                // CURE/PP berries have NO residual handler (their trigger is onUpdate).
                handlers.push(EventHandler {
                    order: STATUS_RESIDUAL_ORDER,
                    priority: 0,
                    speed,
                    sub_order: LEFTOVERS_SUBORDER,
                    effect_order: 0,
                    handler: ResidualAction::BerryResidual { side, slot },
                });
            }
        }

        // --- Sort by comparePriority WITH the Fisher-Yates tie-shuffle (the only
        //     residual-phase NUMERIC draw besides the nested weather shuffle). ---
        speed_sort(&mut handlers, &mut self.prng);

        // --- Apply each handler in resolved order, mirroring `fieldEvent('Residual')`'s
        //     `while (handlers.length)` loop (battle.js:334-389) EXACTLY:
        //
        //       const handler = handlers.shift();
        //       if (handler.effectHolder.fainted) continue;   // skip a fainted holder
        //       singleEvent(...);                             // the HP effect (the chip
        //                                                     //   handler nests the
        //                                                     //   eachEvent('Weather') draw)
        //       this.faintMessages();                         // set `fainted` + checkWin
        //       if (this.ended) return;                       // game over → ABORT the rest
        //
        //     The two subtleties this encodes (both surfaced by the e2e capstone's
        //     residual-vs-faint-under-weather divergences):
        //
        //     (1) PER-HANDLER faintMessages — `fainted` is set BETWEEN handlers, not
        //         once at the end. So a handler that KOs its holder makes that holder's
        //         LATER handlers (the `fainted` `continue`) AND, on a game-ending KO,
        //         ALL remaining handlers no-op. Concretely: a fast burned mon's
        //         status-DoT (sub 6) sorts ahead of a slow foe's Leftovers (sub 4) when
        //         the burned mon is FASTER (speed is a higher-precedence key than
        //         subOrder), so its self-KO ends the battle BEFORE the foe heals — the
        //         survivor does NOT get its Leftovers tick that turn (e2e_0 etc.).
        //
        //     (2) NON-game-ending faints do NOT abort the residual — the loop CONTINUES
        //         for the OTHER mons (only the fainted holder's own remaining handlers
        //         are `continue`d). Only `this.ended` (a side out of mons) returns. So a
        //         residual that KOs ONE active still ticks the surviving active's
        //         Leftovers/DoT this turn (the abort is win-gated, NOT faint-gated — the
        //         prior "break on any 0-HP" doc was WRONG).
        //
        //     Draw-safety: every residual PRNG draw (the handler-sort shuffle above +
        //     the nested `eachEvent('Weather')` INSIDE the weather chip) fires
        //     before/inside its handler; `process_faints` (faintMessages) is draw-free,
        //     as are the skipped HP applications. The deferred-faint protocol is
        //     preserved: a residual that left an active at 0 HP has its `fainted` flag
        //     already set here, so `run_turn`/`turn_loop`'s post-residual handling
        //     pauses for the switch and skips the trailing [15] Update / Quick Claw. ---
        for h in handlers {
            // `if (handler.effectHolder.fainted) continue;` — a holder fainted by an
            // earlier residual handler THIS phase is skipped (no HP effect, no draw).
            // The weather chip's holder is the FIELD (never fainted); the apply path
            // below additionally re-guards per-active for the both-actives chip.
            match h.handler {
                ResidualAction::WeatherChip(w) => self.apply_weather_chip(w, dex),
                ResidualAction::Leftovers { side, slot } => {
                    if self.sides[side].pokemon[slot].fainted {
                        continue;
                    }
                    self.apply_leftovers(side, slot, dex);
                }
                ResidualAction::SpeedBoost { side, slot } => {
                    if self.sides[side].pokemon[slot].fainted {
                        continue;
                    }
                    self.apply_speed_boost(side, slot, dex);
                }
                ResidualAction::RainDish { side, slot } => {
                    if self.sides[side].pokemon[slot].fainted {
                        continue;
                    }
                    self.apply_rain_dish(side, slot, dex);
                }
                ResidualAction::TruantToggle { side, slot } => {
                    // `truant.onResidual`: `truantTurn = !truantTurn`. DRAW-FREE. The
                    // fainted guard mirrors `if (handler.effectHolder.fainted) continue`
                    // (a DoT-KO'd Slaking's toggle is skipped — probe edge E1's arming
                    // depends on it for the REPLACEMENT, which isn't in this list at all).
                    if self.sides[side].pokemon[slot].fainted {
                        continue;
                    }
                    let m = &mut self.sides[side].pokemon[slot];
                    m.truant_turn = !m.truant_turn;
                }
                ResidualAction::StatusDot { side, slot } => {
                    if self.sides[side].pokemon[slot].fainted {
                        continue;
                    }
                    self.apply_status_dot(side, slot, dex);
                }
                ResidualAction::LeechSeed { side, slot, seeder_side } => {
                    if self.sides[side].pokemon[slot].fainted {
                        continue;
                    }
                    self.apply_leech_seed(side, slot, seeder_side, dex);
                }
                // A duration-only volatile (protect/stall): NO HP effect (no `onResidual`),
                // DRAW-FREE. Mirror `fieldEvent('Residual')`'s `handler.state.duration--; if
                // (!duration) end()`: for the STALL volatile, decrement `stall_duration` and,
                // on reaching 0, REMOVE it → zero `protect_counter` (the expiry that makes a
                // Protect after one non-protect turn a fresh first-protect). The `protect`
                // volatile (`duration: 1`) expires the same way but is the turn-top
                // `protected` clear, so its handler is a no-op here. A fainted holder was
                // already `continue`d (the protect/stall mon can't faint behind protect, but
                // a residual could KO it — the guard keeps parity).
                ResidualAction::VolatileDuration { side, slot, is_stall } => {
                    if self.sides[side].pokemon[slot].fainted {
                        continue;
                    }
                    if is_stall {
                        let mon = &mut self.sides[side].pokemon[slot];
                        mon.stall_duration = mon.stall_duration.saturating_sub(1);
                        if mon.stall_duration == 0 {
                            mon.protect_counter = 0; // the volatile ended → reset
                        }
                    }
                }
                // TAUNT duration tick (`gen3_taunt_disable_v1`): decrement + expire at 0. NO HP
                // effect, DRAW-FREE. On expiry the mon's Status moves are usable again.
                ResidualAction::TauntDuration { side, slot } => {
                    if self.sides[side].pokemon[slot].fainted {
                        continue;
                    }
                    let mon = &mut self.sides[side].pokemon[slot];
                    if let Some(turns) = mon.taunt {
                        let next = turns.saturating_sub(1);
                        if next == 0 {
                            mon.taunt = None; // the volatile expired
                            // [EMIT] `|-end|<mon>|move: Taunt|[silent]` — the residual expiry
                            // (`taunt.onEnd`, Phase 3 `gen3_protocol_phase3_v1`; byte-verified
                            // vs the taunt_lifecycle capture). Observation-only.
                            if self.logging() {
                                let m = self.mon_ref(side, slot, dex);
                                self.log.volatile_end_silent(&m, "move: Taunt");
                            }
                        } else {
                            mon.taunt = Some(next);
                        }
                    }
                }
                // DISABLE duration tick (`gen3_taunt_disable_v1`): decrement + expire at 0. NO HP
                // effect, DRAW-FREE. On expiry the disabled move is usable again.
                ResidualAction::DisableDuration { side, slot } => {
                    if self.sides[side].pokemon[slot].fainted {
                        continue;
                    }
                    let mon = &mut self.sides[side].pokemon[slot];
                    if let Some((dslot, turns)) = mon.disable {
                        let next = turns.saturating_sub(1);
                        if next == 0 {
                            mon.disable = None; // the volatile expired
                            // [EMIT] `|-end|<mon>|Disable` — the residual expiry
                            // (`disable.onEnd`, Phase 3 `gen3_protocol_phase3_v1`; byte-verified
                            // vs the disable_lifecycle capture). Observation-only.
                            if self.logging() {
                                let m = self.mon_ref(side, slot, dex);
                                self.log.volatile_end(&m, "Disable");
                            }
                        } else {
                            mon.disable = Some((dslot, next));
                        }
                    }
                }
                // SHED SKIN (`gen3_berry_trace_shedskin_v1`): while STATUSED, ONE
                // randomChance(33,100); a pass cures (before the same-mon DoT at sub 6 —
                // a cure turn takes no chip). Unstatused → NO draw (`pokemon.hp &&
                // pokemon.status &&` short-circuits BEFORE the randomChance).
                ResidualAction::ShedSkin { side, slot } => {
                    if self.sides[side].pokemon[slot].fainted {
                        continue;
                    }
                    self.apply_shed_skin(side, slot, dex);
                }
                // A HEAL/PINCH berry's residual trigger (`gen3_berry_trace_shedskin_v1`):
                // threshold check at APPLY time (an earlier handler — sand chip / Shed
                // Skin — may have moved hp), then EAT + the onEat effect.
                ResidualAction::BerryResidual { side, slot } => {
                    if self.sides[side].pokemon[slot].fainted {
                        continue;
                    }
                    self.apply_berry_residual(side, slot, dex);
                }
            }

            // `this.faintMessages();` — set `fainted` on any active now at 0 HP (and
            // decrement pokemonLeft). `if (this.ended) return;` — if that emptied a
            // side, the battle is over and the REMAINING residual handlers do NOT run.
            self.process_faints(dex);
            if self.check_win().is_some() {
                return;
            }
        }
    }


    /// Sandstorm/Hail chip: `onFieldResidual` adds the upkeep then calls
    /// `eachEvent('Weather')` (a nested 2-active speed-tie shuffle DRAW) → per active
    /// `onWeather` `this.damage(maxhp/16)`, skipping the type-immune mons. The
    /// shuffle fires BEFORE any chip is applied (it sorts the actives), so we draw it
    /// first, then chip both actives in side order (the chip itself is draw-free; its
    /// order is unobservable as it touches distinct mons).
    fn apply_weather_chip(&mut self, weather: Weather, dex: &Dex) {
        // [EMIT] `|-weather|<Weather>|[upkeep]` — the end-of-turn weather TICK, emitted at
        // the TOP of the field-residual (gen-3 `onFieldResidual` adds `-weather …
        // [upkeep]` BEFORE the `eachEvent('Weather')` chip loop, verified vs the golden:
        // it precedes every per-active `|-damage|…|[from] Sandstorm`, and is emitted even
        // when both actives are chip-immune — protect_block turn 1). Observation-only.
        if self.logging() {
            self.log.weather(weather_display(weather), None, None, true);
        }

        // [14b] the nested eachEvent('Weather') active-sort shuffle (tie-only). The
        // returned side order IS the `onWeather` handler run order — so the per-active
        // chip's `|-damage|` PROTOCOL line is emitted in the SPEED-SORTED order (faster
        // first; a same-species / same-speed TIE uses the shuffle's exact permutation, a
        // Snorlax-vs-Snorlax mirror the golden's `-damage` order reveals). The chip itself
        // is state-/seed-INVARIANT (distinct mons, saturating, draw-free — the shuffle
        // already drew), so reading the permutation for the emit changes nothing the seed
        // suite asserts; it only fixes the emitted-line order. A collapsed <2 order (a
        // fainted active) falls back to a natural side walk for the surviving side.
        // Fired for ALL weather (sun/rain INCLUDED — `gen3_ability_batch1_v1`): the
        // `eachEvent('Weather')` shuffle draws on a tie regardless of whether the weather chips.
        let shuffled = self.each_event_shuffle();

        // Sun / Rain have NO chip (`onWeather` has no damage handler) — the shuffle above IS the
        // whole field-residual for them. Return before the per-active chip walk (so the
        // Sand/Hail-only chip code below is unreachable for sun/rain by CONSTRUCTION, not by the
        // per-mon `weather_immune` short-circuit).
        if matches!(weather, Weather::Rain | Weather::Sun) {
            return;
        }

        let order: Vec<usize> = if shuffled.len() == 2 { shuffled } else { vec![0, 1] };

        for &side in &order {
            let slot = self.sides[side].active;
            if self.sides[side].pokemon[slot].fainted {
                continue;
            }
            if self.weather_immune(side, slot, weather, dex) {
                continue;
            }
            let maxhp = self.sides[side].pokemon[slot].maxhp;
            let chip = (maxhp / 16).max(1); // clampIntRange(maxhp/16, 1)
            // FOCUS BAND: the onDamage roll draws on the chip too (a lethal chip still
            // faints — effectType Weather, not Move). `gen3_ability_batch4_v1`.
            let chip = self.focus_band_damage(side, slot, chip, false, dex);
            self.apply_damage(side, slot, chip);
            // [EMIT] `|-damage|<mon>|<HP>|[from] Sandstorm` (or `Hail`) — the residual
            // weather chip, post-chip HP (`0 fnt` if it KO'd). Observation-only.
            if chip > 0 && self.logging() {
                let mon_ref = self.mon_ref(side, slot, dex);
                let hp = self.hp_status(side, slot);
                let name = match weather {
                    Weather::Sand => "Sandstorm",
                    Weather::Hail => "Hail",
                    Weather::Rain | Weather::Sun => unreachable!("rain/sun have no chip"),
                };
                self.log.damage(&mon_ref, &hp, Some(&Cause::Bare(name.to_string())));
            }
        }
    }

    /// Whether `side`'s active is IMMUNE to `weather`'s chip. Sand: Rock/Ground/Steel
    /// (`typechart damageTaken['sandstorm']===3`); Hail: Ice. We read the dex
    /// species types (the calc's type source) and check membership.
    fn weather_immune(&self, side: usize, slot: usize, weather: Weather, dex: &Dex) -> bool {
        let mon = &self.sides[side].pokemon[slot];
        let types = mon_types(mon, dex);
        match weather {
            Weather::Sand => {
                // Rock/Ground/Steel types + the SAND VEIL ability (`gen3_accuracy_pipeline_v1`
                // — its `onImmunity('sandstorm')` returns false, so a Sand Veil holder takes
                // NO sand chip; the ONLY gen3 ability with a weather-chip onImmunity, verified
                // vs the resolved dist). Modeled alongside Sand Veil's ×0.8 evasion so the two
                // ship together: a Sand-Veil mon in sand neither chips nor is hit at full acc.
                types
                    .iter()
                    .any(|t| matches!(t, Type::Rock | Type::Ground | Type::Steel))
                    || to_id(&mon.ability) == "sandveil"
            }
            Weather::Hail => types.iter().any(|t| matches!(t, Type::Ice)),
            // Rain/Sun have no chip.
            Weather::Rain | Weather::Sun => true,
        }
    }

    /// Leftovers heal: `this.heal(maxhp/16)` (floor), capped at maxhp. Draw-free.
    /// `this.heal` (`battle.ts`) returns early `if (!target?.hp)` — a 0-HP mon is NOT
    /// revived. This matters when an EARLIER residual handler (the sand/hail field
    /// chip, order 8 < Leftovers order 10) KO'd this Leftovers holder THIS residual:
    /// `faintMessages` hasn't run yet so `fainted` is still false, but the sim still
    /// won't heal it — so guard on `hp == 0` too, or we'd revive a dead mon to
    /// `maxhp/16` (a STATE divergence).
    fn apply_leftovers(&mut self, side: usize, slot: usize, dex: &Dex) {
        let mon = &mut self.sides[side].pokemon[slot];
        if mon.fainted || mon.hp == 0 {
            return;
        }
        let heal = mon.maxhp / 16; // floor; heal nothing if 0
        let before = mon.hp;
        mon.hp = (mon.hp + heal).min(mon.maxhp);
        // [EMIT] `|-heal|<mon>|<HP>|[from] item: Leftovers` (only when HP actually
        // rose — a full-HP mon's Leftovers is a silent no-op the sim skips too).
        if self.logging() && self.sides[side].pokemon[slot].hp != before {
            let mon_ref = self.mon_ref(side, slot, dex);
            let hp = self.hp_status(side, slot);
            self.log.heal(&mon_ref, &hp, Some(&Cause::Item("Leftovers".to_string())));
        }
    }

    /// SHED SKIN residual (`gen3_berry_trace_shedskin_v1`, `shedskin.onResidual` — order 10
    /// subOrder 3): `if (pokemon.hp && pokemon.status && this.randomChance(33, 100)) cure`.
    /// The status gate SHORT-CIRCUITS BEFORE the randomChance — an unstatused holder draws
    /// NOTHING; a statused one draws EXACTLY ONE randomChance per residual until the cure
    /// lands (probe `probe_trace_shedskin_rng.js` S2). A pass cures the WHOLE major status
    /// (tox stage / sleep counter vanish with the variant); confusion is NOT cured (S3).
    /// Emits `|-activate|<mon>|ability: Shed Skin` + `|-curestatus|<mon>|<tok>|[msg]`.
    fn apply_shed_skin(&mut self, side: usize, slot: usize, dex: &Dex) {
        let mon = &self.sides[side].pokemon[slot];
        if mon.fainted || mon.hp == 0 || mon.status.is_none() {
            return; // no status → NO draw (the sim's short-circuit)
        }
        if !self.prng.random_chance(33, 100) {
            return; // the roll failed — statused, one draw, no cure
        }
        let tok = status_token(self.sides[side].pokemon[slot].status).unwrap_or("");
        self.sides[side].pokemon[slot].status = None;
        if self.logging() {
            let m = self.mon_ref(side, slot, dex);
            self.log.activate(&m, "ability: Shed Skin", None);
            self.log.curestatus(&m, tok, true);
        }
    }

    /// EAT the held berry (`pokemon.eatItem()`, `gen3_berry_trace_shedskin_v1`): the item
    /// becomes NONE for the battle (no later item mods / second eat; it does NOT come back
    /// on switch-out) + the `|-enditem|<mon>|<Item>|[eat]` reveal. The eat itself is
    /// DRAW-FREE (the UseItem/TryEatItem/EatItem events have no other gen-3 handlers —
    /// probe `probe_berry_rng.js`); only an onEat EFFECT may draw (Starf's sample, the
    /// Figy family's confusion). Returns the eaten item's display name.
    fn eat_item(&mut self, side: usize, slot: usize, dex: &Dex) -> String {
        let item_id = to_id(&self.sides[side].pokemon[slot].item);
        let display = dex.item(&item_id).map(|i| i.name.clone()).unwrap_or(item_id);
        self.sides[side].pokemon[slot].item = String::new();
        if self.logging() {
            let m = self.mon_ref(side, slot, dex);
            self.log.push_raw(format!("|-enditem|{m}|{display}|[eat]"));
        }
        display
    }

    /// A HEAL/PINCH berry's RESIDUAL apply (`gen3_berry_trace_shedskin_v1`, order 10
    /// subOrder 4 — the Leftovers slot). The threshold reads the holder's hp AT APPLY
    /// TIME (`hp <= maxhp/2` == `2*hp <= maxhp` exactly; pinch `4*hp <= maxhp` — the
    /// probe-settled float boundaries), then eats + applies the onEat effect:
    ///   - HealFixed: `this.heal(N)` (Oran 10 / Sitrus 30), capped at maxhp. Draw-free.
    ///   - HealFrac: `floor(maxhp/8)` (the RESOLVED gen3 Figy family) + a CONFUSION
    ///     volatile iff the holder's nature LOWERS the flavor stat — the addVolatile
    ///     draws `random(2,6)` (probe: figy + Modest → `-start confusion` + one draw).
    ///   - PinchBoost: +1 stage (clamped +6), draw-free.
    ///   - StarfRandom2: ONE `sample` over the non-capped [atk,def,spa,spd,spe] (in that
    ///     order — the boosts-iteration order; the sample DRAWS even for one candidate;
    ///     an all-capped board draws nothing) then +2 on the drawn stat (clamped).
    ///   - LansatFocusEnergy: the `focusenergy` volatile (crit stage +2), draw-free.
    fn apply_berry_residual(&mut self, side: usize, slot: usize, dex: &Dex) {
        let (hp, maxhp) = {
            let mon = &self.sides[side].pokemon[slot];
            if mon.fainted || mon.hp == 0 {
                return; // eatItem gates `!this.hp`
            }
            (mon.hp as u32, mon.maxhp as u32)
        };
        let item_id = to_id(&self.sides[side].pokemon[slot].item);
        let Some(be) = dex.item(&item_id).and_then(|i| i.berry_effect.clone()) else {
            return;
        };
        use crate::dex::BerryEffect as BE;
        match be {
            BE::HealFixed { amount } => {
                if 2 * hp <= maxhp {
                    let name = self.eat_item(side, slot, dex);
                    self.berry_heal(side, slot, amount, &name, dex);
                }
            }
            BE::HealFrac { frac, confuse_if_minus } => {
                if 2 * hp <= maxhp {
                    let name = self.eat_item(side, slot, dex);
                    let amount = (maxhp / frac as u32) as u16; // floor(maxhp/8)
                    self.berry_heal(side, slot, amount, &name, dex);
                    // `if (pokemon.getNature().minus === "<stat>") addVolatile('confusion')`
                    // — the nature that LOWERS the berry's flavor stat confuses. An EMPTY /
                    // neutral nature has no minus. The addVolatile draws random(2,6) via the
                    // shared confusion-add gates (already-confused / Own Tempo → no draw).
                    let minus = nature_minus_stat(&self.sides[side].pokemon[slot].set.nature, dex);
                    if minus.as_deref() == Some(confuse_if_minus.as_str()) {
                        let before = self.sides[side].pokemon[slot].confusion.is_some();
                        self.add_confusion(side, slot, dex);
                        if self.logging() && !before && self.sides[side].pokemon[slot].confusion.is_some() {
                            let m = self.mon_ref(side, slot, dex);
                            self.log.volatile_start(&m, "confusion");
                        }
                    }
                }
            }
            BE::PinchBoost { stat } => {
                if 4 * hp <= maxhp {
                    let name = self.eat_item(side, slot, dex);
                    self.berry_boost(side, slot, stat, 1, &name, dex);
                }
            }
            BE::StarfRandom2 => {
                if 4 * hp <= maxhp {
                    let name = self.eat_item(side, slot, dex);
                    // The candidate pool: the 5 battle stats (boosts order atk,def,spa,
                    // spd,spe — the sim's `for (stat in pokemon.boosts)` skipping acc/eva)
                    // with stage < +6. `this.sample(stats)` draws `random(len)` even for a
                    // single candidate (probe: all-but-spe capped → `random(1)` + `sample
                    // ([spe])`); an EMPTY pool draws nothing (`if (stats.length)`).
                    let pool: Vec<usize> = (0..5)
                        .filter(|&i| self.sides[side].pokemon[slot].boosts[i] < 6)
                        .collect();
                    if !pool.is_empty() {
                        let idx = self.prng.random_below(pool.len() as u32) as usize;
                        self.berry_boost(side, slot, pool[idx], 2, &name, dex);
                    }
                }
            }
            BE::LansatFocusEnergy => {
                if 4 * hp <= maxhp {
                    let _ = self.eat_item(side, slot, dex);
                    // `addVolatile('focusenergy')` — draw-free; the crit roll reads it
                    // (`onModifyCritRatio critRatio + 2`). Emits the sim's
                    // `|-start|<mon>|move: Focus Energy`.
                    self.sides[side].pokemon[slot].focus_energy = true;
                    if self.logging() {
                        let m = self.mon_ref(side, slot, dex);
                        self.log.volatile_start(&m, "move: Focus Energy");
                    }
                }
            }
            // CURE/PP classes never register a residual handler (their gen-3 trigger is
            // the eachEvent('Update') site / the setStatus tail — `berry_on_update`).
            BE::Cure { .. } | BE::PpRestore { .. } => {}
        }
    }

    /// A berry heal (`this.heal(N)` from onEat): floor'd, capped at maxhp, with the
    /// `|-heal|<mon>|<HP>|[from] item: <Berry>` reveal (probe line form).
    fn berry_heal(&mut self, side: usize, slot: usize, amount: u16, item_name: &str, dex: &Dex) {
        let mon = &mut self.sides[side].pokemon[slot];
        let before = mon.hp;
        mon.hp = (mon.hp + amount).min(mon.maxhp);
        if self.logging() && self.sides[side].pokemon[slot].hp != before {
            let m = self.mon_ref(side, slot, dex);
            let hp = self.hp_status(side, slot);
            self.log.heal(&m, &hp, Some(&Cause::Item(item_name.to_string())));
        }
    }

    /// A berry stat boost (`this.boost({stat: n})` from onEat): clamped to +6, with the
    /// `|-boost|<mon>|<stat>|<n>|[from] item: <Berry>` reveal. A no-op at the cap emits
    /// nothing (the eat still happened — the caller consumed the item first).
    fn berry_boost(&mut self, side: usize, slot: usize, stat: usize, stages: i8, item_name: &str, dex: &Dex) {
        let s = &mut self.sides[side].pokemon[slot].boosts[stat];
        let before = *s;
        *s = (*s + stages).min(6);
        let delta = self.sides[side].pokemon[slot].boosts[stat] - before;
        if self.logging() && delta > 0 {
            let m = self.mon_ref(side, slot, dex);
            self.log.push_raw(format!(
                "|-boost|{m}|{}|{delta}|[from] item: {item_name}",
                crate::protocol::STAT_TOKENS[stat]
            ));
        }
    }

    /// The `eachEvent('Update')` ITEM handlers (`gen3_berry_trace_shedskin_v1`) — the CURE
    /// berries + Leppa, run at EVERY Update site right after its tie-shuffle, per active in
    /// the shuffled speed order (the per-mon effects are independent + DRAW-FREE, so the
    /// order only shapes protocol line order). The HEAL/PINCH berries do NOT fire here
    /// (their gen3 onUpdate is mod-DELETED — residual-only). Probe: a cure lands at the
    /// FIRST Update after the status (before the holder's own move — it never rolls
    /// full-para that turn); an eaten berry never fires again.
    fn run_update_items(&mut self, order: &[usize], dex: &Dex) {
        for &side in order {
            // STATUS_IMMUNE `onUpdate` CURE (`gen3_statusimmune_onupdate_cure_v1`, the A/B
            // Trace-Porygon2 status cluster): each of the 6 resolved STATUS_IMMUNE members
            // (Insomnia / Vital Spirit slp, Limber par, Immunity psn+tox, Water Veil brn,
            // Magma Armor frz) carries an `onUpdate` that CURES the holder's matching
            // status. Unreachable with the mon's OWN ability (the apply is blocked), but a
            // STATUSED mon that TRACES one (the only gen3 route — a slept Porygon2 re-enters
            // vs an Insomnia Hypno) is cured at the FIRST `Update` after the copy. DRAW-FREE
            // (`cureStatus` draws nothing; the `-activate` line is protocol-only). The
            // ability handler is gathered BEFORE the item's (`findPokemonEventHandlers`
            // ability-then-item), so it runs ahead of the berry check per mon.
            self.status_immune_on_update(side, dex);
            self.berry_on_update(side, dex);
        }
    }

    /// One active's Update-site STATUS_IMMUNE-ability cure (see [`Self::run_update_items`]).
    fn status_immune_on_update(&mut self, side: usize, dex: &Dex) {
        let slot = self.sides[side].active;
        let mon = &self.sides[side].pokemon[slot];
        if mon.fainted || mon.hp == 0 {
            return;
        }
        let Some(tok) = status_token(mon.status) else { return };
        let cures = dex
            .ability(&to_id(&mon.ability))
            .and_then(|a| a.status_immune.as_ref())
            .map(|si| si.blocks(tok))
            .unwrap_or(false);
        if cures {
            self.sides[side].pokemon[slot].status = None;
        }
    }

    /// One active's Update-site berry check (see [`Self::run_update_items`]).
    fn berry_on_update(&mut self, side: usize, dex: &Dex) {
        let slot = self.sides[side].active;
        {
            let mon = &self.sides[side].pokemon[slot];
            if mon.fainted || mon.hp == 0 {
                return;
            }
        }
        let item_id = to_id(&self.sides[side].pokemon[slot].item);
        let Some(be) = dex.item(&item_id).and_then(|i| i.berry_effect.clone()) else {
            return;
        };
        use crate::dex::BerryEffect as BE;
        match be {
            BE::Cure { statuses, cures_confusion, .. } => {
                let mon = &self.sides[side].pokemon[slot];
                let status_hit = status_token(mon.status)
                    .map(|t| statuses.iter().any(|s| s == t))
                    .unwrap_or(false);
                let conf_hit = cures_confusion && mon.confusion.is_some();
                if status_hit || conf_hit {
                    let _ = self.eat_item(side, slot, dex);
                    self.berry_cure(side, slot, status_hit, conf_hit, dex);
                }
            }
            BE::PpRestore { amount } => {
                // Leppa: `moveSlots.some(pp === 0)` → eat; onEat restores the FIRST 0-PP
                // slot `min(pp+10, maxpp)` + `|-activate|<mon>|item: Leppa Berry|<Move>|
                // [consumed]` (probe: a 5-pp Blizzard maxpp 8 restored 0→8).
                let k = self.sides[side].pokemon[slot].move_pp.iter().position(|&p| p == 0);
                if let Some(k) = k {
                    let name = self.eat_item(side, slot, dex);
                    let mon = &mut self.sides[side].pokemon[slot];
                    mon.move_pp[k] = (mon.move_pp[k] + amount).min(mon.move_maxpp[k]);
                    if self.logging() {
                        let move_name = dex
                            .moves(&self.sides[side].pokemon[slot].set.moves[k])
                            .map(|mv| mv.name.clone())
                            .unwrap_or_default();
                        let m = self.mon_ref(side, slot, dex);
                        self.log.push_raw(format!("|-activate|{m}|item: {name}|{move_name}|[consumed]"));
                    }
                }
            }
            // HEAL/PINCH fire at the RESIDUAL only (the gen3 mod deletes their onUpdate).
            _ => {}
        }
    }

    /// A cure berry's onEat: cure the matching major status (`-curestatus [msg]`) and/or
    /// remove confusion (Lum cures BOTH; Persim confusion-only; the single-status six cure
    /// exactly their status). Clearing `status` to None drops the tox stage / sleep
    /// counter with the variant. DRAW-FREE.
    fn berry_cure(&mut self, side: usize, slot: usize, cure_status: bool, cure_confusion: bool, dex: &Dex) {
        if cure_status {
            let tok = status_token(self.sides[side].pokemon[slot].status).unwrap_or("");
            self.sides[side].pokemon[slot].status = None;
            if self.logging() {
                let m = self.mon_ref(side, slot, dex);
                self.log.curestatus(&m, tok, true);
            }
        }
        if cure_confusion {
            let had = self.sides[side].pokemon[slot].confusion.is_some();
            self.sides[side].pokemon[slot].confusion = None;
            if self.logging() && had {
                let m = self.mon_ref(side, slot, dex);
                self.log.volatile_end(&m, "confusion");
            }
        }
    }

    /// LUM's IMMEDIATE eat (`lum.onAfterSetStatus`, priority -1 — AFTER a Synchronize
    /// reflect at priority 0; probe: `|-status|holder| → |-status|source|…Synchronize| →
    /// |-enditem|holder|Lum Berry|[eat]| → |-curestatus|`): fired from the two status-SET
    /// sites (`try_set_status`'s tail + Rest's self-sleep, BEFORE Rest's heal) when the
    /// holder's CURRENT item is an `immediate` cure berry covering the just-set status.
    /// DRAW-FREE. (The non-immediate cure berries wait for the next Update site.)
    fn berry_after_set_status(&mut self, side: usize, slot: usize, dex: &Dex) {
        let mon = &self.sides[side].pokemon[slot];
        if mon.fainted || mon.hp == 0 || mon.status.is_none() {
            return;
        }
        let item_id = to_id(&mon.item);
        let Some(crate::dex::BerryEffect::Cure { statuses, cures_confusion, immediate }) =
            dex.item(&item_id).and_then(|i| i.berry_effect.clone())
        else {
            return;
        };
        if !immediate {
            return;
        }
        let status_hit = status_token(self.sides[side].pokemon[slot].status)
            .map(|t| statuses.iter().any(|s| s == t))
            .unwrap_or(false);
        if status_hit {
            let cure_conf = cures_confusion && self.sides[side].pokemon[slot].confusion.is_some();
            let _ = self.eat_item(side, slot, dex);
            self.berry_cure(side, slot, true, cure_conf, dex);
        }
    }

    /// SPEED BOOST residual (`gen3_ability_batch1_v1`, `speedboost.onResidual`): `+1 spe`
    /// stage on `side`'s `slot` mon (clamped +6), ONLY if `pokemon.activeTurns` (it was
    /// active the whole turn — a mon that switched in this turn has `active_turns == 0` and
    /// skips its entry turn). DRAW-FREE (`boost()` consumes no PRNG). The boost updates the
    /// stage IMMEDIATELY but does NOT touch `cached_speed` (like a Dragon Dance — the cached
    /// speed re-establishes at the next turn-start/residual/switch-in, so the boosted speed
    /// takes effect NEXT turn). VERIFIED vs the sim (`harness/probe_speedboost_activeturns.js`).
    fn apply_speed_boost(&mut self, side: usize, slot: usize, dex: &Dex) {
        let mon = &mut self.sides[side].pokemon[slot];
        if mon.fainted || mon.hp == 0 {
            return;
        }
        // `if (pokemon.activeTurns)` — a switch-in (active_turns 0) skips its entry turn.
        if mon.active_turns == 0 {
            return;
        }
        // `this.boost({spe: 1})` — clamp the stage to +6 (a boost at +6 is a no-op-success).
        let s = &mut mon.boosts[4];
        let before = *s;
        *s = (*s + 1).min(6);
        // [EMIT] `|-boost|<mon>|spe|1` (only when the stage actually rose; a +6-capped
        // boost emits nothing — the clamped-delta-sign convention used elsewhere).
        if self.logging() && self.sides[side].pokemon[slot].boosts[4] != before {
            let mon_ref = self.mon_ref(side, slot, dex);
            self.log.boost(&mon_ref, 4, 1); // stat_idx 4 == spe (STAT_TOKENS order)
        }
    }

    /// RAIN DISH residual (`gen3_ability_batch1_v1`, `raindish.onResidual`): heal
    /// `floor(maxhp/16)` on `side`'s `slot` mon in EFFECTIVE rain (suppressed by a Cloud
    /// Nine / Air Lock mon on the field via `effective_weather`). DRAW-FREE (`heal()`
    /// consumes no PRNG; a full-HP heal fails). VERIFIED vs the sim.
    fn apply_rain_dish(&mut self, side: usize, slot: usize, dex: &Dex) {
        if self.effective_weather(dex) != Some(Weather::Rain) {
            return;
        }
        let mon = &mut self.sides[side].pokemon[slot];
        if mon.fainted || mon.hp == 0 {
            return;
        }
        let heal = mon.maxhp / 16; // floor
        let before = mon.hp;
        mon.hp = (mon.hp + heal).min(mon.maxhp);
        // [EMIT] `|-heal|<mon>|<HP>|[from] ability: Rain Dish` (only when HP rose).
        if self.logging() && self.sides[side].pokemon[slot].hp != before {
            let mon_ref = self.mon_ref(side, slot, dex);
            let hp = self.hp_status(side, slot);
            self.log.heal(&mon_ref, &hp, Some(&Cause::Ability("Rain Dish".to_string())));
        }
    }

    /// A RECOVERY-MOVE self-heal: add `amount` HP to `side`'s active `slot`, clamped to
    /// `maxhp` (Showdown's `this.heal` — `pokemon.ts:1641`). Returns `true` iff the heal
    /// did anything (mirrors `heal`'s truthy return). DRAW-FREE — `heal` consumes NO PRNG.
    ///
    /// `heal` is a NO-OP (returns false) when: hp == 0 (a fainted mon is NOT revived),
    /// `amount <= 0`, OR `hp >= maxhp` (already full). The caller uses the `false` return
    /// for the FULL-HP / heal-0 FAIL path (a Recover at full HP fails — `runMoveEffects`
    /// emits `-fail` and ends the move; still draw-free). The clamp mirrors the sim's
    /// `if (this.hp > this.maxhp) this.hp = this.maxhp`.
    ///
    /// NOTE: Showdown's `heal()` also bumps a positive sub-1 heal to 1
    /// (`if (damage && damage <= 1) damage = 1`). We intentionally OMIT it: the caller
    /// pre-floors `amount`, and every modeled gen-3-OU heal (≥ ~25 = floor(maxhp/4) for a
    /// ~100+ maxhp mon) is well above 1, so the bump is unreachable here. It would only
    /// matter for a tiny-maxhp construct (maxhp < 4) outside the modeled scope.
    fn apply_heal(&mut self, side: usize, slot: usize, amount: u16) -> bool {
        let mon = &mut self.sides[side].pokemon[slot];
        // `if (!this.hp) return false;` — a 0-HP / fainted mon is never healed.
        if mon.fainted || mon.hp == 0 {
            return false;
        }
        // `if (d <= 0) return false;` — a 0-amount heal does nothing.
        if amount == 0 {
            return false;
        }
        // `if (this.hp >= this.maxhp) return false;` — already full → FAIL (the
        // Recover-at-full-HP path; draw-free either way).
        if mon.hp >= mon.maxhp {
            return false;
        }
        mon.hp = (mon.hp + amount).min(mon.maxhp);
        true
    }

    /// Water Absorb / Volt Absorb `onTryHit` heal (`floor(maxhp/4)`, capped at
    /// maxhp, no-op at full HP — gen3 `this.heal(target.baseMaxhp/4)`). Called on the
    /// immune short-circuit when the defender's ability is the matching absorb for
    /// the move type. DRAW-FREE (heal consumes no PRNG). A faint-guard: a 0-HP mon
    /// never reaches here (the move that KO'd it already resolved), but match
    /// `apply_leftovers`'s guard for safety.
    fn apply_absorb_heal(&mut self, side: usize, slot: usize, move_type: Option<Type>) {
        let mon = &mut self.sides[side].pokemon[slot];
        if mon.fainted || mon.hp == 0 {
            return;
        }
        let ability = to_id(&mon.ability);
        let heals = matches!(
            (ability.as_str(), move_type),
            ("waterabsorb", Some(Type::Water)) | ("voltabsorb", Some(Type::Electric))
        );
        if !heals {
            return;
        }
        let heal = mon.maxhp / 4; // floor
        mon.hp = (mon.hp + heal).min(mon.maxhp);
    }

    /// FLASH FIRE activation (gen3 `flashfire.onTryHit`): a Fire-type move that LANDS on a
    /// Flash Fire holder ARMS its `flash_fire` volatile so its OWN Fire moves become ×1.5.
    /// Called on the immune short-circuit (the same `acc_hit`-gated site as the absorb heal)
    /// so a MISSED Fire move does not activate it. DRAW-FREE. VERIFIED vs the omniscient sim
    /// (`harness/probe_flashfire_rng.js`).
    ///
    /// The resolved `onTryHit` gates: the target's ability is `flashfire`, the move is
    /// `Fire`-type, `target != source` (always true here — a self-target damaging move never
    /// hits the immune path). It SKIPS activation when the holder's status is `frz` (the
    /// `status === 'frz'` guard — a frozen FF mon absorbs the hit's damage but does NOT arm).
    /// The Will-O-Wisp Fire-type/statused/subbed skip is N/A on this DAMAGING path — WoW is a
    /// status move, routed through `run_status_move`, whose OWN Flash-Fire absorb arm
    /// (`gen3_ff_wisp_absorb_v1`, after the sub block) models the WoW special-case; any OTHER
    /// (non-`frz`) status is irrelevant here.
    fn apply_flash_fire_activation(&mut self, side: usize, slot: usize, move_type: Option<Type>) {
        if move_type != Some(Type::Fire) {
            return;
        }
        let mon = &mut self.sides[side].pokemon[slot];
        if mon.fainted {
            return;
        }
        if to_id(&mon.ability) != "flashfire" {
            return;
        }
        // The gen3 `onTryHit` `status === 'frz'` guard: a frozen holder does NOT arm.
        if mon.status == Some(Status::Freeze) {
            return;
        }
        mon.flash_fire = true;
    }

    /// Apply the major-status DoT for `side`'s active (burn `maxhp/8`, poison
    /// `maxhp/8`, Toxic `max(1, floor(maxhp/16)) * stage` with the per-mon stage
    /// ramping 1..15). The Toxic stage counter lives ON `Status::Toxic(stage)` and is
    /// incremented HERE (gen3 `onResidual` does `if (stage<15) stage++` BEFORE the
    /// damage). All draw-free; faints at 0.
    fn apply_status_dot(&mut self, side: usize, slot: usize, dex: &Dex) {
        let mon = &mut self.sides[side].pokemon[slot];
        if mon.fainted {
            return;
        }
        let maxhp = mon.maxhp;
        // The bare-status `[from]` token for the emitted `|-damage|` (`brn`/`psn`/`tox`).
        let (dmg, from_tok): (u16, &'static str) = match mon.status {
            Some(Status::Burn) => ((maxhp / 8).max(1), "brn"),
            Some(Status::Poison) => ((maxhp / 8).max(1), "psn"),
            Some(Status::Toxic(stage)) => {
                // gen3: ramp the stage (cap 15) BEFORE computing the damage.
                let next = if stage < 15 { stage + 1 } else { 15 };
                mon.status = Some(Status::Toxic(next));
                ((maxhp / 16).max(1) * next as u16, "tox")
            }
            _ => return,
        };
        // FOCUS BAND: the onDamage roll draws on the DoT chip (no survive — not a Move).
        let dmg = self.focus_band_damage(side, slot, dmg, false, dex);
        self.apply_damage(side, slot, dmg);
        // [EMIT] `|-damage|<mon>|<HP>|[from] <brn|psn|tox>` (the residual DoT chip;
        // the HP is post-chip, `0 fnt` if the DoT KO'd). Observation-only.
        if dmg > 0 && self.logging() {
            let mon_ref = self.mon_ref(side, slot, dex);
            let hp = self.hp_status(side, slot);
            self.log.damage(&mon_ref, &hp, Some(&Cause::Bare(from_tok.to_string())));
        }
    }

    /// The **LEECH SEED** residual drain (`leechseed.onResidual`, gen-3 — verified
    /// bit-for-bit vs the omniscient sim `harness/probe_leechseed_rng.js`). The seeded
    /// mon (`side`/`slot`) loses `floor(baseMaxhp/8)` and the SEEDER's CURRENT active
    /// (`seeder_side`'s active) HEALS the drained amount. The gen-3 `onResidual`:
    ///
    /// ```text
    /// const target = getAtSlot(volatiles.leechseed.sourceSlot);   // the seeder's active
    /// if (!target || target.fainted || target.hp <= 0) return;    // "Nothing to leech into"
    /// const damage = this.damage(pokemon.baseMaxhp / 8, pokemon, target);
    /// if (damage) this.heal(damage, target, pokemon);
    /// ```
    ///
    /// In gen-3 singles the `sourceSlot` is always the seeder's `a` slot, so `getAtSlot`
    /// returns whatever is CURRENTLY active on the seeder's side — the heal FOLLOWS a
    /// seeder that switched (we store only `seeder_side`, reading `side.active` here).
    ///
    /// Key behaviours (all verified):
    ///   * **SEEDER-FAINTED gate** — if the seeder's active is fainted (or 0 HP), the
    ///     WHOLE residual returns EARLY: NO drain on the seeded mon, NO heal. (The drain
    ///     is conditional on a live recipient, unlike burn/poison which always chip.)
    ///   * **DRAIN amount** `floor(maxhp/8)` (gen-3 `baseMaxhp == maxhp`), clamped to the
    ///     seeded mon's current HP via `apply_damage` (a KO sets HP 0).
    ///   * **HEAL amount** = the ACTUAL drained amount (`min(floor(maxhp/8), hp_before)`,
    ///     the sim's `damage()` return), clamped to the seeder's maxhp via `apply_heal`.
    ///     The heal is applied even when the drain KOs the seeded mon (the heal is inside
    ///     the same `onResidual`, before `faintMessages`).
    ///   * **LIQUID OOZE** (the seeded mon's ability) REVERSES it — the seeder takes the
    ///     damage instead of healing. Rare in gen-3 OU; FAIL-LOUD (the e2e filter keeps a
    ///     Liquid Ooze team out, and no modeled scenario uses it) so it can never silently
    ///     mis-resolve. DRAW-FREE either way.
    fn apply_leech_seed(&mut self, side: usize, slot: usize, seeder_side: usize, dex: &Dex) {
        // LIQUID OOZE fail-loud: the seeded mon's ability reverses the drain into damage
        // on the SEEDER. Not modeled (rare in gen-3 OU; excluded from the e2e filter).
        let seeded_ability = to_id(&self.sides[side].pokemon[slot].ability);
        assert!(
            seeded_ability != "liquidooze",
            "Leech Seed into a Liquid Ooze holder reverses the drain (seeder takes damage) \
             — NOT modeled (rare in gen-3 OU). Exclude it or model it before a battle uses it."
        );
        let _ = dex; // reserved (no dex read needed for the gen-3 drain/heal math)

        // The seeder's CURRENT active (the heal recipient / `getAtSlot(sourceSlot)`).
        let seeder_slot = self.sides[seeder_side].active;
        let seeder = &self.sides[seeder_side].pokemon[seeder_slot];
        // SEEDER-FAINTED gate: `if (!target || target.fainted || target.hp <= 0) return;`
        // — no drain, no heal, the whole onResidual returns.
        if seeder.fainted || seeder.hp == 0 {
            return;
        }

        // DRAIN the seeded mon `floor(maxhp/8)`, clamped to its HP (apply_damage saturates
        // at 0 + returns whether it KO'd; the dealt amount is min(drain, hp_before)).
        let seeded = &self.sides[side].pokemon[slot];
        let hp_before = seeded.hp;
        let drain = seeded.maxhp / 8; // floor; gen-3 baseMaxhp == maxhp
        if drain == 0 {
            return; // a sub-8-maxhp construct: damage 0 → no heal (the `if (damage)` gate)
        }
        // FOCUS BAND: the onDamage roll draws on the drain (no survive — not a Move).
        let drain = self.focus_band_damage(side, slot, drain, false, dex);
        let dealt = drain.min(hp_before); // the sim's `damage()` return (actual dealt)
        self.apply_damage(side, slot, drain);
        // [EMIT] `|-damage|<seeded>|<HP>|[from] Leech Seed|[of] <seeder-active>` — the
        // `[of]` is the seeder side's CURRENT active (the heal recipient), matching the
        // sim's getAtSlot resolution. Byte-verified vs the leechseed_splash_payday
        // capture (Phase 3). Observation-only.
        if self.logging() && dealt > 0 {
            let seeded_ref = self.mon_ref(side, slot, dex);
            let hp = self.hp_status(side, slot);
            let of = self.mon_ref(seeder_side, seeder_slot, dex);
            self.log.damage_of(&seeded_ref, &hp, &Cause::Bare("Leech Seed".to_string()), &of);
        }

        // HEAL the seeder's active by the DEALT amount (clamped to its maxhp). The
        // `if (damage)` gate: a 0-deal (the seeded mon was already at 0) heals nothing —
        // but apply_damage already no-ops at hp 0 and dealt would be 0, so this is safe.
        if dealt > 0 {
            let healed = self.apply_heal(seeder_side, seeder_slot, dealt);
            // [EMIT] `|-heal|<seeder-active>|<HP>|[silent]` — the drain's heal half
            // (byte-verified vs the capture: the `[silent]` bare tag, no `[from]`).
            if self.logging() && healed {
                let m = self.mon_ref(seeder_side, seeder_slot, dex);
                let hp = self.hp_status(seeder_side, seeder_slot);
                self.log.push_raw(format!("|-heal|{m}|{hp}|[silent]"));
            }
        }
    }

    /// Clear the FLINCH volatile on BOTH actives (the `duration:1` end-of-turn
    /// expiry). DRAW-FREE — flinch carries no PRNG.
    fn clear_flinch(&mut self) {
        for side in 0..2 {
            let a = self.sides[side].active;
            self.sides[side].pokemon[a].flinch = false;
            // --- The `protect` this-turn volatile (`duration: 1`) expires at the next
            //     turn-top, exactly like `flinch` — clear it. The longer-lived `stall`
            //     volatile (`duration: 2`) is NOT touched here: its expiry happens at the
            //     RESIDUAL (`run_residuals` decrements `stall_duration` and zeros
            //     `protect_counter` at 0), faithful to `fieldEvent('Residual')`'s
            //     duration-handler `end`. (A SUCCESSFUL consecutive protect re-set
            //     `protected` true + reset `stall_duration` to 2 in `run_protect`, so it
            //     survives; a non-protect turn lets the residual run it down.) ---
            self.sides[side].pokemon[a].protected = false;
        }
    }

    /// Whether either side's active mon is fainted (the trailing-Quick-Claw gate).
    fn any_active_fainted(&self) -> bool {
        (0..2).any(|s| {
            let a = self.sides[s].active;
            self.sides[s].pokemon[a].fainted
        })
    }

    /// Build the two move actions and order them by (priority → effective speed),
    /// wiring [`speed_sort`] so a priority+speed TIE draws the action-order
    /// Fisher-Yates shuffle from the PRNG (the production-path wiring). Returns the
    /// actions in resolved run order.
    fn order_actions(
        &mut self,
        p1_move_slot: usize,
        p2_move_slot: usize,
        dex: &Dex,
    ) -> Vec<MoveAction> {
        let slots = [p1_move_slot, p2_move_slot];
        // Build an EventHandler per side keyed by move priority + effective speed.
        // The payload is the MoveAction. `order` is the move default (both equal so
        // it never separates); `sub_order`/`effect_order` are 0 (no SwitchIn
        // fractional offset on a move action — a true speed tie is a FULL tie that
        // triggers the shuffle, exactly the sim's behaviour).
        let mut handlers: Vec<EventHandler<MoveAction>> = (0..2)
            .map(|side| {
                let slot = self.sides[side].active;
                let priority = self.move_priority(side, slot, slots[side], dex);
                let speed = self.effective_speed(side, slot, dex);
                EventHandler {
                    order: NO_ORDER,
                    priority: priority as i32,
                    speed: speed as f64,
                    // A real Showdown MoveAction carries no subOrder/effectOrder, so
                    // comparePriority reads them as 0 — match that so the key is
                    // correct if move actions are ever ordered against other actions.
                    sub_order: 0,
                    effect_order: 0,
                    handler: MoveAction { side, slot, move_index: slots[side], struggle: false },
                }
            })
            .collect();

        // speed_sort orders DESCENDING (priority high-first, speed high-first) and
        // draws the size-2 shuffle ONLY on an exact (priority, speed) tie — the one
        // production-path action-order draw.
        speed_sort(&mut handlers, &mut self.prng);
        handlers.iter().map(|h| h.handler).collect()
    }

    /// The move's resolved gen-3 priority (the raw dex priority; gen3 has no
    /// `onModifyPriority` handlers and `fractionalPriority` is always 0, so this is
    /// the dex value, drawing nothing).
    fn move_priority(&self, side: usize, slot: usize, move_index: usize, dex: &Dex) -> i8 {
        match self.move_at(side, slot, move_index, dex) {
            Some(m) => m.priority,
            None => 0,
        }
    }

    /// `getActionSpeed` (gen-3 override): the RAW boosted + ModifySpe speed (NOT
    /// `trunc(spe, 13)`), capped at 10000. `storedStats.spe` (`stats[5]`) through
    /// the boost table (floor) then paralysis ×0.25 (`floor(spe*25/100)`). Gen-3
    /// INHERITS gen4's `par.onModifySpe` `chainModify(0.25)` (data/mods/gen4/
    /// conditions.ts:9-14; gen3/conditions.ts does NOT redefine `par`), VERIFIED
    /// against the sim (a raw-350 Tauros reports 87 = floor(350·0.25) under para,
    /// not 175). For a clean (no-para, no-boost) mon this is just `stats[5]`.
    ///
    /// **Deferred `onModifySpe`/`getActionSpeed` inputs (none on our golden teams,
    /// so seed-parity holds; wire them with the full `runEvent` gather in the
    /// turn-loop step):** the gen-3 weather-speed abilities **Swift Swim** /
    /// **Chlorophyll** (×2 in rain/sun) and the **Quick Claw** `speed = 65535`
    /// override (a QC holder whose end-of-prev-turn `randomChance(1,5)` hit moves
    /// first). A Swift-Swim mon under rain or a Quick-Claw proc would mis-order
    /// here and could desync — out of this bounded step's scope.
    fn effective_speed(&self, side: usize, slot: usize, dex: &Dex) -> u32 {
        let mon = &self.sides[side].pokemon[slot];
        let base = mon.stats[5] as u32;
        // boost-table floor (boosts index 4 == spe) — applied BEFORE the ModifySpe chain
        // (VERIFIED vs the sim's getStat: +1 Swift-Swim mon in rain = boost then ×2).
        let boosted = apply_boost(base, mon.boosts[4]);
        // ModifySpe (`runEvent('ModifySpe')` — gen3 `getStat('spe')`): every onModifySpe
        // handler is a `chainModify` that ACCUMULATES into ONE 4096 modifier applied once
        // (NOT sequential `modify`s). The gen-3 onModifySpe handlers on our modeled mons:
        //   - PARALYSIS ×0.25 (gen4-inherited `chainModify(0.25)`, conditions.ts:9-14).
        //   - WEATHER_SPEED ×2 (`gen3_ability_batch1_v1`, Chlorophyll in sun / Swift Swim in
        //     rain — `onModifySpe chainModify(2)` gated on `effectiveWeather()`).
        // The `+2047`-rounded fixed-point form matters (raw 359 → 90 not 89); composing
        // ×2 ⊗ ×0.25 as ONE chain (=×0.5) matches the sim (206→103), sequential would too
        // here but a general chain is the faithful model. Empty ⇒ `boosted` unchanged.
        let mut mods: Vec<(u64, u64)> = Vec::new();
        // Weather-speed ×2, gated on the CURRENT effective weather (suppressed by a
        // Cloud Nine / Air Lock mon on the field — `effective_weather`) AND on the mon
        // being ALIVE (`gen3_fainted_no_ability_speed_v1`): a FAINTED mon's ability
        // handlers no longer gather (`faintMessages` sets `isActive = false`, so
        // `runEvent('ModifySpe')` skips the corpse's Swift Swim / Chlorophyll), and the
        // corpse sorts the replacement instaswitch at its PLAIN speed. Probe-verified on
        // the ab_894_12 board (rain up, Kingdra `getActionSpeed()`: alive 368 → fainted
        // 184, TYING the 184 Smeargle corpse → the instaswitch shuffle draw the port was
        // missing). The para analogue is `check_fainted`'s status-`fnt` clear
        // (`gen3_fnt_clears_status_v1`).
        if let Some(w) = dex.ability(&mon.ability).and_then(|a| a.weather_speed) {
            if !mon.fainted && self.effective_weather(dex) == Some(w) {
                mods.push((2, 1));
            }
        }
        if mon.status == Some(Status::Paralysis) {
            mods.push((1, 4));
        }
        let modified = crate::damage::chain_modify(boosted as u64, &mods) as u32;
        modified.min(10000)
    }

    /// The gen-3 `field.effectiveWeather()` — `field.weather` UNLESS a Cloud Nine / Air Lock
    /// (WEATHER_NEGATE, `gen3_ability_batch1_v1`) mon is active on EITHER side, in which case
    /// weather's effects are suppressed and this returns `None` (the sim's `suppressWeather`
    /// gate). The RAW `field.weather` still persists (for the upkeep/counter); only its
    /// EFFECTS (speed ×2, the chip, damage mods) are suppressed while a negater is up.
    fn effective_weather(&self, dex: &Dex) -> Option<crate::state::Weather> {
        let w = self.field.weather?;
        for side in 0..2 {
            let slot = self.sides[side].active;
            let mon = &self.sides[side].pokemon[slot];
            if mon.fainted {
                continue;
            }
            if dex.ability(&mon.ability).map(|a| a.weather_negate).unwrap_or(false) {
                return None;
            }
        }
        Some(w)
    }

    /// Resolve the [`MoveData`]-like move for a side's active move slot. `None` for
    /// an out-of-range slot or unknown move id.
    fn move_at<'a>(
        &self,
        side: usize,
        slot: usize,
        move_index: usize,
        dex: &'a Dex,
    ) -> Option<&'a crate::dex::MoveData> {
        let mon = &self.sides[side].pokemon[slot];
        let move_id = mon.set.moves.get(move_index)?;
        dex.moves(move_id)
    }

    /// The EFFECTIVE to-hit accuracy the single `random(100) < effAcc` roll compares
    /// against (`gen3_accuracy_pipeline_v1`) — the exact gen3 `tryMoveHit` computation:
    ///
    /// ```text
    /// effAcc = move.accuracy                                   (integer)
    ///          × acc/eva STAGE TABLE  [3/3,4/3,5/3,6/3,7/3,8/3,9/3]  (inline float ops)
    ///          × the accMod handlers  (runEvent('ModifyAccuracy'))
    /// ```
    ///
    /// Stage math (gen3 `scripts.ts::tryMoveHit`): the ATTACKER's accuracy stage
    /// (`boosts[5]`) → `acc *= table[+s]` / `acc /= table[-s]` (s==0 divides by
    /// `table[0]=1`, a no-op); the DEFENDER's evasion stage (`boosts[6]`) → `acc /=
    /// table[+s]` for a positive stage, `acc *= table[-s]` for a negative one (s==0 does
    /// nothing — the `else if boost < 0` branch). The result stays a RAW f64 (NOT
    /// floored) into the comparison — `random(100)` is an integer 0..99, so `int <
    /// effAcc_f64` matches JS bit-for-bit.
    ///
    /// accMod (`runEvent('ModifyAccuracy')`, handlers priority-DESC): the DIRECT
    /// multiplies (Bright Powder ×0.9 / Lax Incense ×0.95 — `AccOp::Multiply`) mutate the
    /// running accuracy unconditionally; the CHAIN members (Compound Eyes ×1.3 / Sand Veil
    /// ×0.8-in-sand / Hustle ×3277/4096 physical — `AccOp::Chain`) accumulate into ONE
    /// 4096 modifier that is applied at the END via `modify(acc, modifier)` — BUT ONLY
    /// when `acc` is a NON-NEGATIVE INTEGER (the runEvent guard `relayVar ===
    /// Math.abs(Math.floor(relayVar))`, battle.ts). So a stage or a direct multiply that
    /// made `acc` a non-integer float SKIPS every chain member — the exact integer-guard
    /// (probe `harness/probe_accuracy_intguard.js`). `Accuracy` runEvent has no gen3
    /// handlers (identity).
    ///
    /// The attacker is `(side, slot)`; the evasion target + defender item/ability is
    /// `(foe, foe_slot)` (every DRAWN move targets the foe — a self-target move is
    /// never_miss and never reaches the roll). `move_type` gates Hustle. DRAW-FREE — this
    /// consumes NO PRNG; the caller does the one `random(100)` roll.
    ///
    /// EMPTY-PATH IDENTITY: with no acc/eva stage and no accMod, `acc` stays the integer
    /// `move.accuracy` and the chain is empty → returns `move.accuracy as f64`, so the
    /// caller's `random_below(100) < effAcc` is byte-identical to the pre-existing
    /// `random_chance(accuracy, 100)` (= `random_below(100) < accuracy`).
    fn effective_accuracy(
        &self,
        side: usize,
        slot: usize,
        foe: usize,
        foe_slot: usize,
        base_accuracy: u16,
        move_type: Option<Type>,
        dex: &Dex,
    ) -> f64 {
        // gen3 boostTable [1, 4/3, 5/3, 2, 7/3, 8/3, 3] — the 3/3-base ±6 acc/eva form.
        const BOOST_TABLE: [f64; 7] =
            [1.0, 4.0 / 3.0, 5.0 / 3.0, 2.0, 7.0 / 3.0, 8.0 / 3.0, 3.0];
        let mut acc = base_accuracy as f64;

        // --- inline stages (BEFORE ModifyAccuracy) ---
        // Attacker accuracy stage (gen3 has no ignoreAccuracy move in scope).
        let acc_stage = self.sides[side].pokemon[slot].boosts[5].clamp(-6, 6);
        if acc_stage > 0 {
            acc *= BOOST_TABLE[acc_stage as usize];
        } else {
            // s==0 divides by table[0]=1 (a no-op float divide, matching the sim's `else`).
            acc /= BOOST_TABLE[(-acc_stage) as usize];
        }
        // Defender evasion stage.
        let eva_stage = self.sides[foe].pokemon[foe_slot].boosts[6].clamp(-6, 6);
        if eva_stage > 0 {
            acc /= BOOST_TABLE[eva_stage as usize];
        } else if eva_stage < 0 {
            acc *= BOOST_TABLE[(-eva_stage) as usize];
        }

        // --- runEvent('ModifyAccuracy') ---
        // Gather the accMod handlers: attacker item+ability (side=Attacker) + defender
        // item+ability (side=Defender). Direct multiplies mutate `acc`; chain members
        // accumulate into `chain` (applied at the END, integer-guarded). Handler ORDER
        // between chain and direct is irrelevant (they touch disjoint state: event.modifier
        // vs relayVar), and the chain-vs-chain accumulate is order-independent for the gen3
        // members (proven) — so a single deterministic pass is bit-faithful.
        let mut chain: Vec<(u64, u64)> = Vec::new();
        // WEATHER_NEGATE (`gen3_ability_batch1_v1`): Sand Veil's ×0.8 acc drop needs EFFECTIVE
        // sand — a Cloud Nine / Air Lock mon on the field suppresses it (`effective_weather`).
        let sand = self.effective_weather(dex) == Some(Weather::Sand);
        let mut apply = |am: Option<&crate::dex::accmod::AccMod>| {
            if let Some(am) = am {
                if am.weather_sand && !sand {
                    return;
                }
                if am.physical_types_only
                    && !move_type.map_or(false, hustle_boosts_accuracy_type)
                {
                    return;
                }
                match am.op {
                    crate::dex::accmod::AccOp::Multiply(f) => acc *= f,
                    crate::dex::accmod::AccOp::Chain(n, d) => chain.push((n, d)),
                }
            }
        };
        let atk_item = to_id(&self.sides[side].pokemon[slot].item);
        let atk_abil = to_id(&self.sides[side].pokemon[slot].ability);
        let def_item = to_id(&self.sides[foe].pokemon[foe_slot].item);
        let def_abil = to_id(&self.sides[foe].pokemon[foe_slot].ability);
        // Attacker-owned (onSourceModifyAccuracy): Compound Eyes / Hustle.
        apply(dex.item(&atk_item).and_then(|i| i.acc_mod.as_ref()).filter(|a| a.side == crate::dex::accmod::AccSide::Attacker));
        apply(dex.ability(&atk_abil).and_then(|a| a.acc_mod.as_ref()).filter(|a| a.side == crate::dex::accmod::AccSide::Attacker));
        // Defender-owned (onModifyAccuracy): Bright Powder / Lax Incense / Sand Veil.
        apply(dex.item(&def_item).and_then(|i| i.acc_mod.as_ref()).filter(|a| a.side == crate::dex::accmod::AccSide::Defender));
        apply(dex.ability(&def_abil).and_then(|a| a.acc_mod.as_ref()).filter(|a| a.side == crate::dex::accmod::AccSide::Defender));

        // The chain modifier applies at the END of runEvent ONLY when `acc` is a
        // non-negative integer (`relayVar === Math.abs(Math.floor(relayVar))`).
        if !chain.is_empty() && acc >= 0.0 && acc.fract() == 0.0 {
            acc = accuracy_chain_modify(acc as u64, &chain) as f64;
        }
        acc
    }

    /// Draw the single gen3 to-hit roll: `random(100) < effAcc` (never_miss ⇒ auto-hit,
    /// NO draw). Consumes exactly ONE `random(100)` (byte-identical to the pre-pipeline
    /// `random_chance(accuracy, 100)` when there is no stage/item/ability). See
    /// [`Self::effective_accuracy`] for the effAcc math.
    fn roll_accuracy(
        &mut self,
        side: usize,
        slot: usize,
        foe: usize,
        foe_slot: usize,
        accuracy: u16,
        never_miss: bool,
        move_type: Option<Type>,
        dex: &Dex,
    ) -> bool {
        if never_miss {
            return true;
        }
        let eff = self.effective_accuracy(side, slot, foe, foe_slot, accuracy, move_type, dex);
        // random(100) is an integer in [0,99]; `int < effAcc_f64` matches JS's
        // `this.random(100) < accuracy` bit-for-bit (both promote the int to f64).
        (self.prng.random_below(100) as f64) < eff
    }

    /// Resolve and APPLY one damaging move (accuracy → crit → damage → HP → faint).
    /// Consumes the PRNG in the exact verified order. Returns a [`MoveResolution`]
    /// whose `landed` flag gates the in-tryMoveHit `eachEvent('Update')` shuffle (it
    /// fires only when the move actually hits — a miss/immune returns from
    /// `tryMoveHit` before it).
    fn run_move(&mut self, action: MoveAction, will_act: bool, foe_will_move: bool, dex: &Dex) -> MoveResolution {
        let MoveAction { side, slot, move_index, struggle } = action;
        let foe = 1 - side;
        let foe_slot = self.sides[foe].active;

        // Resolve the move's static facts (clone the few fields we need so the
        // immutable dex borrow doesn't conflict with the &mut self mutations).
        //
        // A FORCED STRUGGLE (`gen3_pp_tracking_v1`) resolves the synthetic `struggle`
        // dex entry (bp 50, physical, accuracy 100, gen-3 `recoil:[1,4]`) but with its
        // TYPE overridden to typeless '???' (`move.type = '???'` in Showdown's
        // `onModifyMove`), so it has NO STAB and hits everything (incl. Ghosts — a
        // typeless move has no type-chart row → effectiveness 1×). `move_index` is
        // IGNORED for a Struggle (it is not a slot). Struggle is fail-loud-required in
        // the data (a missing `struggle` entry PANICS rather than silently no-op).
        let (
            accuracy,
            never_miss,
            base_power,
            move_type,
            category,
            halves_def,
            crit_ratio,
            is_fire,
            move_id,
            status_inflicted,
            is_protect_move,
            targets_self,
            move_name,
            is_contact,
        ) = if struggle {
            let m = dex
                .moves("struggle")
                .expect("gen3_pp_tracking_v1: the `struggle` move MUST exist in gen3_moves.json");
            (
                m.accuracy,   // 100 (gen-3 Struggle is NOT never-miss → it DRAWS accuracy)
                m.never_miss, // false
                m.base_power, // 50
                None,         // typeless '???' (the onModifyMove override) — no STAB, hits Ghosts
                MoveCategory::Physical, // gen-3 Struggle is physical
                false,        // not explosion/selfdestruct
                m.crit_ratio, // 1 (normal crit)
                false,        // not a Fire move
                "struggle".to_string(),
                None,         // no status
                false,        // not a protect move
                false,        // targets the foe, not self
                m.name.clone(), // "Struggle"
                m.contact,    // Struggle IS a contact move (gen-3 `flags.contact`) → it CAN proc a contact ability
            )
        } else {
            match self.move_at(side, slot, move_index, dex) {
                Some(m) => (
                    m.accuracy,
                    m.never_miss,
                    m.base_power,
                    m.move_type,
                    m.category,
                    to_id(&m.id) == "explosion" || to_id(&m.id) == "selfdestruct",
                    m.crit_ratio,
                    m.move_type == Some(Type::Fire) && m.category != MoveCategory::Status,
                    to_id(&m.id),
                    m.status_inflicted.clone(),
                    m.is_protect,
                    m.target == "self",
                    m.name.clone(),
                    m.contact,
                ),
                // Not a known move — resolve as a no-op (draws nothing). This is a
                // programming error in scope (the caller picks damaging slots).
                None => return MoveResolution::no_op(),
            }
        };

        // --- onBeforeMove STATUS draws (BEFORE accuracy), mirroring
        //     runEvent('BeforeMove') at runMove (battle-actions.ts:255), which
        //     precedes useMove/PP/accuracy. Handlers run priority-DESC (sleep 10,
        //     freeze 10, flinch 8, confusion 3, par 1), SHORT-CIRCUITING on the first
        //     abort (a lower-priority status then never draws). A move that aborts
        //     here draws NOTHING further — no accuracy/crit/damage/secondary. ---
        // `is_status` for the TAUNT onBeforeMove cant — whether Taunt BLOCKS this move
        // (`gen3_taunt_disable_v1`): a derived-Status move that is NOT a fixed-damage move
        // (Seismic Toss etc. are bp-0 but a non-Status Showdown category, so Taunt does not
        // block them — VERIFIED vs the sim). Struggle is physical so this is false anyway.
        let is_status_move =
            category == MoveCategory::Status && !is_fixed_damage_move(&move_id);
        if !self.on_before_move(side, slot, move_index, is_status_move, struggle, dex) {
            // The move was cancelled (full-para / still-asleep / frozen-no-thaw /
            // flinched / confusion self-hit / DISABLED move / TAUNTED status move). Like a
            // miss/immune: not landed, no tail.
            //
            // PP is NOT deducted on a cancelled move — Showdown's `deductPP` runs AFTER
            // `runEvent('BeforeMove')` PASSES (battle-actions.ts:255-287), so a full-para
            // / still-asleep / flinched / frozen / confusion-self-hit turn consumes NO PP
            // (VERIFIED vs the sim: a full-para Snorlax keeps its Body Slam PP). Struggle
            // is likewise not deducted (it has no slot).
            return MoveResolution::done(false, false, false);
        }

        // --- PP DEDUCTION (`gen3_pp_tracking_v1`), mirroring `deductPP` at
        //     battle-actions.ts:282 (right AFTER BeforeMove passes, BEFORE accuracy).
        //     DRAW-FREE. The mon's OWN move deducts 1 from its used slot; a foe holding
        //     **Pressure** deducts 2 (the `runEvent('DeductPP')` extra, battle-actions.ts:
        //     472-483 — VERIFIED −2, no RNG). Struggle deducts NOTHING (it is not a slot).
        //     A self-target move (setup/heal/protect) still deducts its slot's PP; only a
        //     Pressure FOE that is actually TARGETED adds the extra — a self-target move
        //     does not target a foe, so no Pressure extra (Showdown's `pressureTargets`).
        if !struggle {
            let pressure_extra = !targets_self
                && to_id(&self.sides[foe].pokemon[foe_slot].ability) == "pressure";
            let deduct = if pressure_extra { 2 } else { 1 };
            self.sides[side].pokemon[slot].deduct_pp(move_index, deduct);

            // --- CHOICE LOCK (`gen3_pp_tracking_v1`, `choiceband.onModifyMove` →
            //     `addVolatile('choicelock')`): a Choice-item mon (gen-3: only Choice Band)
            //     LOCKS to the FIRST slot it uses; every other slot becomes disabled. Set it
            //     HERE (when the move actually runs, after PP is deducted — matching Showdown's
            //     onModifyMove timing). Idempotent: re-using the locked move keeps the lock.
            //     This is what forces Struggle once the locked move's PP hits 0 while other
            //     slots still have PP (the CB-Tyranitar exhausting Crunch → Struggle). Cleared
            //     on switch-out (`execute_switch`). ---
            if to_id(&self.sides[side].pokemon[slot].item) == "choiceband" {
                self.sides[side].pokemon[slot].choice_locked_move = Some(move_index);
            }
        }

        // --- LAST-USED MOVE (`gen3_taunt_disable_v1`, `pokemon.moveUsed` at battle-actions.ts:260,
        //     right after PP is deducted / BeforeMove passed). Records the slot this mon just USED
        //     so a FOE's Disable can disable it. Set for a REAL move to `Some(move_index)`; a
        //     STRUGGLE sets `pokemon.lastMove = struggle` (NOT a slot) — Disable's `onTryHit`
        //     rejects a Struggle lastMove, so we store `None` (no disable-able slot). A cancelled
        //     move (full-para / sleep / flinch / frozen / confusion-self-hit) returned BEFORE this
        //     point (like PP), so it leaves `last_move` unchanged — mirroring `moveUsed` running
        //     only after BeforeMove passes. ---
        self.sides[side].pokemon[slot].last_move = if struggle { None } else { Some(move_index) };

        // --- PROTECT / DETECT (the user's OWN protect move) — a self-target,
        //     never-miss, priority-3 `stallingMove`/`volatileStatus:'protect'` move. It
        //     resolves at priority 3 (BEFORE the foe's attack, so the volatile is up when
        //     the foe's move runs), draws the gen-3 stall SUCCESS roll on a CONSECUTIVE
        //     use, and sets the `protected` volatile + (re)multiplies the stall counter.
        //     See `run_protect` for the full draw model. (We route on `is_protect_move`,
        //     the dex `isProtect` flag, which in our modeled scope is exactly Protect /
        //     Detect — Endure carries `volatileStatus:'endure'` + a different `onDamage`
        //     mechanic and is fail-loud-EXCLUDED in `run_protect`.) ---
        if is_protect_move {
            return self.run_protect(side, slot, &move_id, &move_name, will_act, dex);
        }

        // --- PROTECT BLOCK (the FOE side): a move TARGETING the protected mon is
        //     blocked. In gen-3 `tryMoveHit` the protect `onTryHit` fires at the `TryHit`
        //     event, which runs AFTER the accuracy roll (`scripts.ts` line 364 accuracy →
        //     line 369 `if (accPass) runEvent('TryHit')`) and BEFORE the immunity report
        //     — so the blocked foe move DRAWS its accuracy roll, then (if it passes) is
        //     blocked, drawing NO crit / damage / secondary / status. A move that MISSES
        //     its accuracy never reaches the protect block (reports `-miss`, not
        //     `-activate Protect`). Protect only blocks moves that TARGET the protected
        //     mon (a self-target move — the foe's own Protect / setup / recovery — is
        //     never blocked). VERIFIED vs the sim (incl. ordering: EQ into a Flying /
        //     Levitate protector shows `-activate Protect`, NOT `-immune` — protect wins
        //     the TryHit before immunity is reported). The block draw is handled per-path
        //     (damaging / status) right after each one's accuracy roll, via
        //     `protect_blocks` — see below. ---

        // --- FIXED-DAMAGE / FIXED-FORMULA moves (Seismic Toss / Night Shade / Sonic
        //     Boom / Dragon Rage / Super Fang) — a `damageCallback` / `damage:` move in
        //     Showdown that BYPASSES `getDamage`, so it deals a fixed/derived number with
        //     NO crit roll + NO 16-way damage roll. These carry `basePower: 0` in the data,
        //     so `derive_category` classifies them as **Status** (bp 0) — so we MUST route
        //     them HERE, BEFORE the `category == Status` branch (which would else send them
        //     to `run_status_move`'s fail-loud guard). We route by id, mirroring how the
        //     status/setup/recovery arms fan out inside `run_status_move`. Their draw model
        //     (VERIFIED bit-for-bit vs `harness/probe_fixeddamage_rng.js`): draw accuracy
        //     (`randomChance(acc,100)`, skip iff never_miss — Seismic Toss / Night Shade /
        //     Dragon Rage are acc-100 but NOT never-miss, so they STILL draw one roll; Sonic
        //     Boom / Super Fang are acc-90 and CAN miss), then the type-immunity short-circuit
        //     (accuracy-drawn-THEN-`-immune`, like the damaging path), then apply the fixed
        //     damage through the sub-absorb / faint machinery. NO crit, NO damage roll, NO
        //     secondary. The DEFERRED fixed-damage set (Psywave / the OHKO moves / Counter /
        //     Mirror Coat / Bide / Endeavor) is routed here too but has NO `fixed_damage_amount`
        //     entry, so `run_fixed_damage_move` PANICS fail-loud rather than silently desync. ---
        if is_fixed_damage_move(&move_id) {
            return self.run_fixed_damage_move(
                side, slot, foe, foe_slot, accuracy, never_miss, move_type, &move_id,
                &move_name, targets_self, dex,
            );
        }

        // --- STANDALONE STATUS MOVE (category Status, bp 0, `move.status` set) —
        //     the gen-3 `data/mods/gen3/scripts.ts::tryMoveHit` status path. A status
        //     move has NO crit / damage / secondary; it draws ONLY accuracy (unless
        //     never_miss) then applies `move.status` via `try_set_status`. It NEVER
        //     fires the in-`tryMoveHit` `eachEvent('Update')` shuffle (verified: a
        //     landed status `moveHit` returns `undefined` → `scripts.ts:468`'s
        //     `if (!damage && damage !== 0) return damage;` short-circuits BEFORE the
        //     line-470 `eachEvent('Update')`), so `landed` is always FALSE. ---
        if category == MoveCategory::Status {
            return self.run_status_move(
                side,
                slot,
                foe,
                foe_slot,
                accuracy,
                never_miss,
                move_type,
                &move_id,
                &move_name,
                targets_self,
                status_inflicted.as_deref(),
                foe_will_move,
                dex,
            );
        }

        // A non-damaging non-status (0-BP, no status) move is out of scope: no-op.
        if base_power == 0 {
            return MoveResolution::no_op();
        }

        // --- DAMP (`gen3_ability_batch2_v1`, `damp.onAnyTryMove`) — an active mon on EITHER
        //     side CANCELS Explosion / Self-Destruct at `runEvent('TryMove')` (battle-actions.ts:412),
        //     which precedes BOTH the self-KO faint (line 422) AND the accuracy roll. So a
        //     Damp-blocked Explosion draws NOTHING (the user does NOT self-KO, no acc/crit/dmg)
        //     and emits `|cant|<damp holder>|ability: Damp|<MoveName>|[of] <user>`. PROBE-verified
        //     (`harness/probe_block_abilities_rng.js`: only the end-of-turn Quick Claw draws).
        //     `onAnyTryMove` fires for the FOE's Explosion AND the Damp mon's OWN Explosion.
        //     PP is already deducted (the sim's DeductPP at 401 precedes TryMove) — so this
        //     matches the port's PP deduction at the on_before_move-passed site above. ---
        if halves_def {
            if let Some((damp_side, damp_slot)) = self.damp_holder(dex) {
                // [EMIT] `|cant|<damp holder>|ability: Damp|<MoveName>|[of] <user>` — the CANT is
                // on the DAMP HOLDER (the sim's `this.effectState.target`), the move name, and
                // `[of]` the move's user. Observation-only; the move draws nothing.
                if self.logging() {
                    let holder = self.mon_ref(damp_side, damp_slot, dex);
                    let user = self.mon_ref(side, slot, dex);
                    self.log.cant_of_move(&holder, "ability: Damp", &move_name, &user);
                }
                // No self-KO, no accuracy, no damage — a full no-op that draws nothing.
                return MoveResolution::done(false, false, false);
            }
        }

        // --- SELF-DESTRUCT (Explosion/Selfdestruct, gen≠4): the user faints AS PART
        //     of the move BEFORE the hit (`useMoveInner` battle-actions.ts:422 →
        //     `this.battle.faint(pokemon)`), zeroing its HP. Draw-free — it does NOT
        //     reorder the acc/crit/dmg draws below. process_faints (the runAction
        //     tail) then sets the `fainted` flag for BOTH the user and the KO'd
        //     target, so a mutual Explosion KO is a true double-faint. `halves_def`
        //     IS the selfdestruct flag (explosion/selfdestruct), reused here. ---
        if halves_def {
            self.sides[side].pokemon[slot].hp = 0;
            // [EMIT] the self-KO'd user is enqueued FIRST (Showdown `faint(user)` in
            // useMoveInner precedes the hit) so `process_faints` emits `|faint|<user>`
            // BEFORE `|faint|<target>` on a mutual Explosion. Observation-only push.
            self.record_faint_order(side, slot);
            // Flag the self-KO for the per-decision record (a coverage/diagnostic signal ONLY;
            // the faint is applied via the normal `process_faints` machinery, so this does not
            // touch any draw or state). Explosion is momentary — no persistent board state like
            // a substitute — so the e2e capstone reads THIS flag to count explosion decisions.
            self.pending_explosion_self_ko = true;
        }

        // --- 1. ACCURACY: random(100) < effAcc, drawn unless never_miss
        //     (move.accuracy === true). effAcc folds the acc/eva stage table + the
        //     accMod item/ability handlers (`gen3_accuracy_pipeline_v1`,
        //     `roll_accuracy`/`effective_accuracy`). We draw NOW (for draw-order parity)
        //     but DEFER the miss decision until after the immunity check: gen3 reports
        //     `-immune` (not `-miss`) for an immune move even when the accuracy roll
        //     would have failed. Same draw count either way. ---
        let acc_hit =
            self.roll_accuracy(side, slot, foe, foe_slot, accuracy, never_miss, move_type, dex);

        // --- PROTECT BLOCK (damaging move): the accuracy roll has been drawn. If it
        //     PASSED and the foe is `protected` (the protect volatile is up — it resolved
        //     at priority 3 before this attack) and this move TARGETS the foe (not a
        //     self/field move, which a damaging move never is), the move is BLOCKED at the
        //     `TryHit` event — drawing NO crit / damage / secondary. (A miss reached the
        //     `-miss` path before the protect block, so `!acc_hit` falls through to the
        //     genuine-miss return below — never reaching this.) The block precedes the
        //     immunity short-circuit (protect wins the TryHit before `-immune` is
        //     reported — verified vs the sim). DRAW-FREE: only the accuracy roll already
        //     happened; the block adds no draw. ---
        if acc_hit && self.protect_blocks(foe, foe_slot, targets_self) {
            // [EMIT] `|move|<attacker>|<Name>|<protector>` then `|-activate|<protector>|
            // Protect` — a blocked foe attack shows the `|move|` announce (NO `-crit`/
            // `-damage`/effectiveness — the block precedes them) + the Protect activate,
            // BEFORE the `-immune` report (protect wins TryHit). VERIFIED vs the golden.
            if self.logging() {
                let user = self.mon_ref(side, slot, dex);
                let target = self.mon_ref(foe, foe_slot, dex);
                self.log.move_used(&user, &move_name, Some(&target), false, false);
                self.log.activate(&target, "Protect", None);
            }
            // JUMP KICK / HIGH JUMP KICK crash through Protect (`gen3_jump_kick_crash_v1`):
            // the sim's `onMoveFail` fires on the Protect block too — the crash draws its
            // crit + damage roll and hurts the user (probe D: a JK into Protect is
            // protect-turn-baseline +2 draws + the user `-damage`).
            if move_id == "jumpkick" || move_id == "highjumpkick" {
                self.apply_jump_kick_crash(
                    side, slot, foe, foe_slot, base_power, move_type, category, crit_ratio, dex,
                );
            }
            return MoveResolution::done(false, false, false);
        }

        // --- Build the DamageContext (no PRNG; resolves the gen-3 ability/Levitate +
        //     type immunities + stat mods + weather). ---
        let mut ctx = self.build_damage_context(
            side, slot, foe, foe_slot, base_power, move_type, category, halves_def, dex,
        );

        // --- FACADE ×2-when-statused (`gen3_facade_v1`, probe
        //     `harness/probe_facade_gen3.js`): the move's OWN dist `onBasePower` is
        //     `if (pokemon.status && pokemon.status !== 'slp') return chainModify(2)` —
        //     a BASE-POWER-phase CHAIN member (it joins the ONE accumulated 4096
        //     modifier with the incense/pinch/Thick-Fat members), NOT a direct multiply.
        //     Probe-settled: psn/tox/par all deal ×2 (70→BP 140); brn deals ×2 AND the
        //     gen3 burn damage-halve STILL applies (gen3 Facade does NOT ignore burn —
        //     max-roll 108 == the unstatused 108); a burned GUTS user composes Atk ×1.5
        //     + halve-suppressed + BP ×2 (318); slp is EXCLUDED by the handler (and
        //     unreachable — a sleeping mon can't act; a mon that WAKES this turn is
        //     cured before the damage calc, same as a thawed one). Id-gated per the
        //     fixed-damage precedent (`gen3_moves.json` carries no onBasePower field).
        //     DRAW-FREE (probe: 4 draws with and without the boost). ---
        if move_id == "facade"
            && matches!(
                self.sides[side].pokemon[slot].status,
                Some(s) if !matches!(s, Status::Sleep(_))
            )
        {
            ctx.bp_mods.push(BpMod::Chain(2, 1));
        }

        // --- IMMUNITY short-circuit (the draw-COUNT crux): a type-chart 0× or an
        //     ability/Levitate immunity. gen3 `tryMoveHit` knows immunity up front,
        //     draws accuracy UNCONDITIONALLY, then emits `-immune` and returns — so an
        //     immune move draws ONLY accuracy (NO crit, NO damage roll) and reports
        //     IMMUNE (not missed), regardless of the accuracy roll (verified vs the
        //     sim: Earthquake into a Flying/Levitate mon draws `randomChance(100,100)`
        //     then `-immune`). Drawing crit/damage here would desync every later draw. ---
        if move_is_immune(&ctx, dex) {
            // Whether THIS immunity is a gen3 `onTryHit`-class ABILITY immunity (Flash
            // Fire / Water Absorb / Volt Absorb) — those resolve AFTER the accuracy roll,
            // so a MISS shows `[miss]`+`-miss` (F2), unlike Levitate + type-chart 0× which
            // resolve at the PRE-accuracy `runImmunity` and ALWAYS report `-immune` even on
            // a would-be miss (probe `harness/probe_levitate_miss.js`: 40/40 immune). The
            // frz/faint guards mirror the FF `onTryHit` early-return (a frozen FF holder is
            // NOT fire-immune — the move hits with full draws). Read state directly.
            let fm_ability = to_id(&self.sides[foe].pokemon[foe_slot].ability);
            let fm_frozen = self.sides[foe].pokemon[foe_slot].status == Some(Status::Freeze);
            let fm_fainted = self.sides[foe].pokemon[foe_slot].fainted;
            let is_tryhit_ff = move_type == Some(Type::Fire)
                && fm_ability == "flashfire"
                && !fm_frozen
                && !fm_fainted;
            let absorb_name = tryhit_absorb_ability(&fm_ability, move_type);
            let is_tryhit_ability = is_tryhit_ff || absorb_name.is_some();

            // F2 — a MISSED TryHit-class ability immunity emits `[miss]`+`-miss`, NOT
            // `-immune` (the accuracy roll already drew — draw-identical, emission-only).
            // Byte-verified vs the capture (`harness/probe_f1_f2_f3_lines.js` /
            // `probe_f2_ff_armed_miss.js`). The heal/arm are already `acc_hit`-gated below,
            // so on a miss no state changes — this only fixes the emitted line + return.
            if is_tryhit_ability && !acc_hit {
                if self.logging() {
                    let user = self.mon_ref(side, slot, dex);
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.move_used(&user, &move_name, Some(&target), true, false);
                    self.log.miss(&user, Some(&target));
                }
                return MoveResolution::done(true, false, false);
            }

            // [EMIT] `|move|<user>|<Name>|<foe>` then the immune line (gen3 reports
            // `-immune`, NOT `-miss`, even when the accuracy roll would have failed —
            // for the PRE-accuracy immunities; the post-accuracy miss returned above).
            // Observation-only: reads already-resolved state, draws nothing.
            //
            // FLASH FIRE split (Phase 3, byte-verified vs the flashfire_cycle capture):
            // a Fire move that HITS an un-armed FF holder shows the ARM line
            // `|-start|<foe>|ability: Flash Fire` (NO `-immune`); an ALREADY-armed
            // holder shows `|-immune|<foe>|[from] ability: Flash Fire`.
            //
            // WATER/VOLT ABSORB (F3, byte-verified vs the capture): a LANDED absorb shows
            // `|-immune|<foe>|[from] ability: Water Absorb` (resp. Volt Absorb), NOT a
            // plain `-immune`. Type-chart 0× / Levitate keep the plain `|-immune|<foe>`.
            if self.logging() {
                let user = self.mon_ref(side, slot, dex);
                let target = self.mon_ref(foe, foe_slot, dex);
                self.log.move_used(&user, &move_name, Some(&target), false, false);
                let fm = &self.sides[foe].pokemon[foe_slot];
                let ff_case = is_tryhit_ff && acc_hit;
                if ff_case {
                    if fm.flash_fire {
                        self.log.immune_from_ability(&target, "Flash Fire");
                    } else {
                        self.log.volatile_start(&target, "ability: Flash Fire");
                    }
                } else if let Some(name) = absorb_name.filter(|_| acc_hit) {
                    self.log.immune_from_ability(&target, name);
                } else {
                    self.log.immune(&target);
                }
            }
            // WATER / VOLT ABSORB heal (gen3 `onTryHit`, DRAW-FREE): an absorbed
            // Water/Electric move heals the defender `floor(maxhp/4)` (capped, no
            // heal at full HP) instead of doing damage. CRUCIALLY this is an
            // `onTryHit` ability handler that fires AFTER the accuracy check — so it
            // triggers ONLY when the move actually HITS. A Water/Electric move that
            // MISSES (e.g. Hydro Pump's 80% accuracy fails) does NOT heal the
            // Absorb holder (verified vs the sim: a missed Hydro Pump into a Water
            // Absorb Politoed reports `-miss`, no heal — only the Sandstorm chip
            // applies). So gate the heal on `acc_hit`. (The draw COUNT is identical
            // either way — accuracy only, no crit/damage — which is why a wrongly-
            // applied heal desyncs the post-hit HP STATE but not the seed.) Flash
            // Fire does NOT heal (its only effect is a Fire-boost flag — a lesser,
            // deferred gap); Levitate / type-chart 0× immunities resolve at the
            // pre-accuracy `runImmunity` and do not heal.
            if acc_hit {
                self.apply_absorb_heal(foe, foe_slot, move_type);
                // FLASH FIRE ACTIVATION (gen3 `flashfire.onTryHit`, DRAW-FREE): a Fire move
                // that LANDS on a Flash Fire holder ARMS its `flash_fire` volatile (thereafter
                // its own Fire moves are ×1.5). Like Water/Volt Absorb this is an `onTryHit`
                // handler AFTER the accuracy check — so it is gated on `acc_hit` (a MISSED Fire
                // move does NOT activate it — probe `harness/probe_flashfire_rng.js` A2). The
                // resolved `onTryHit` skips a `frz`-status holder (the `status === 'frz'`
                // guard) and Will-O-Wisp into a Fire-type/statused/subbed target; in gen-3 OU
                // every FF holder IS Fire-type, so WoW never arms it (probe A3) — but the
                // Fire-type-target skip is naturally covered because a Fire move that is
                // BLOCKED (immune) still reaches here only via the type-absorb path, and the
                // `frz` guard is honoured below. Any non-`frz` status is irrelevant (probe A6).
                // HONEST SCOPE: activation lives ONLY at this damaging-move `acc_hit` site, so a
                // Fire-type STATUS move (Will-O-Wisp routes through `run_status_move`) arming a
                // NON-Fire FF holder is not modeled — impossible in gen-3 OU (every FF holder is
                // Fire-type → the sim's own `hasType("Fire")`/brn-immunity skip), so it never diverges.
                self.apply_flash_fire_activation(foe, foe_slot, move_type);
            }
            // hit-but-immune (acc_hit) OR a genuine miss (!acc_hit): either way NOT
            // landed (no in-tryMoveHit Update), 0 further draws. The `missed` flag is
            // cosmetic here (it gates nothing downstream — only `landed` does).
            return MoveResolution::done(!acc_hit, false, false);
        }

        // --- A genuine accuracy miss (non-immune move): no crit/damage draw, and
        //     NOT landed (the in-tryMoveHit Update shuffle is skipped). ---
        if !acc_hit {
            // [EMIT] `|move|<user>|<Name>|<foe>|[miss]` then `|-miss|<user>|<foe>`.
            if self.logging() {
                let user = self.mon_ref(side, slot, dex);
                let target = self.mon_ref(foe, foe_slot, dex);
                self.log.move_used(&user, &move_name, Some(&target), true, false);
                self.log.miss(&user, Some(&target));
            }
            // JUMP KICK / HIGH JUMP KICK crash on a miss (`gen3_jump_kick_crash_v1`):
            // the resolved gen3 `onMoveFail` — the user takes half the damage the move
            // would have dealt (its `getDamage` DRAWS crit + the 16-way roll), clamped
            // to [1, floor(TARGET.maxhp/2)]. A type-immune target never reaches this
            // return (the `-immune` short-circuit above) — matching the handler's
            // `runImmunity("Fighting")` gate. Probe: `probe_jumpkick_crash_rng.js`.
            if move_id == "jumpkick" || move_id == "highjumpkick" {
                self.apply_jump_kick_crash(
                    side, slot, foe, foe_slot, base_power, move_type, category, crit_ratio, dex,
                );
            }
            return MoveResolution::done(true, false, false);
        }

        // [EMIT] `|move|<user>|<Name>|<foe>` (a landed damaging move — the effectiveness
        // / crit / damage lines follow). Emitted BEFORE the crit roll so the line order
        // matches the sim; the emit reads already-resolved state (draws nothing).
        if self.logging() {
            let user = self.mon_ref(side, slot, dex);
            let target = self.mon_ref(foe, foe_slot, dex);
            self.log.move_used(&user, &move_name, Some(&target), false, false);
            // Effectiveness (the DEFENDER's ident) — the type-chart product of the
            // move type vs the defender's species types. `-supereffective` (>1) /
            // `-resisted` (<1, non-zero); a 0× is the immune path above. Emitted
            // BEFORE `-crit` (the sim's order, verified vs the golden).
            if let Some(mt) = move_type {
                let def_types = mon_types(&self.sides[foe].pokemon[foe_slot], dex);
                let eff = dex.type_chart().effectiveness(mt, &def_types);
                if eff > 1.0 {
                    self.log.supereffective(&target);
                } else if eff < 1.0 && eff > 0.0 {
                    self.log.resisted(&target);
                }
            }
        }

        // --- 2. CRIT: randomChance(1, critMult[critRatio]) — UNCONDITIONAL for a
        //     damaging, non-immune move (critRatio >= 1). After accuracy, before
        //     damage. FOCUS ENERGY (`gen3_berry_trace_shedskin_v1` — in gen3 reachable
        //     only via a Lansat Berry eat): the volatile's `onModifyCritRatio` adds +2
        //     inside `runEvent('ModifyCritRatio')`, then `clampIntRange(critRatio, 0, 5)`
        //     — so the DENOMINATOR shifts (1→3 ⇒ 1/4; a high-crit 2→4 ⇒ 1/3) while the
        //     draw COUNT is unchanged. ---
        let eff_crit_ratio = if self.sides[side].pokemon[slot].focus_energy {
            (crit_ratio as u32 + 2).min(5)
        } else {
            crit_ratio as u32
        };
        let mut crit = self.prng.random_chance(1, CRIT_MULT[eff_crit_ratio as usize]);
        // CRIT_IMMUNE (`gen3_ability_batch1_v1`, Battle Armor / Shell Armor): the roll above
        // is DRAWN normally (draw-count unchanged), THEN `runEvent('CriticalHit')` reads the
        // defender's `onCriticalHit=false` and OVERRIDES the crit to false — so a crit-immune
        // DEFENDER is never crit-hit while the RNG draw is identical to a normal target
        // (PROBE-settled `harness/probe_critimmune_rng.js`: draw-free override). A confusion
        // self-hit passes `crit:false` and doesn't reach here.
        if crit
            && dex
                .ability(&self.sides[foe].pokemon[foe_slot].ability)
                .map(|a| a.crit_immune)
                .unwrap_or(false)
        {
            crit = false;
        }

        // [EMIT] `|-crit|<foe>` (after effectiveness, before `|-damage|`).
        if crit && self.logging() {
            let target = self.mon_ref(foe, foe_slot, dex);
            self.log.crit(&target);
        }

        // Re-resolve damage WITH the crit flag (crit zeros boosts/screens).
        ctx.crit = crit;
        let dmg = crate::damage::calc_damage(&ctx, dex);

        // --- 3. DAMAGE: random(16) selects rolls[r] (gen-3 randomizer). ---
        let r = self.prng.random_below(16) as usize;
        let realized = dmg.rolls[r];

        // --- SUBSTITUTE ABSORB (the gen-3 sub-intercept, `substitute.onTryPrimaryHit`):
        //     if the foe TARGET has a substitute, the DAMAGE hits the SUB's HP (not the
        //     mon). The sub BREAKS (→ None) when its HP reaches 0; the excess does NOT
        //     carry to the mon (gen-3). The acc/crit/damage draws above are UNCHANGED
        //     (the sub-block runs INSIDE moveHit, AFTER getDamage already drew crit +
        //     the random(16) — VERIFIED via the PRNG probe: a damaging move into a sub
        //     draws the same acc+crit+dmg as a bare hit). `absorbed` gates the SECONDARY
        //     EFFECT below (the secondary random(100) STILL draws — gen-3 quirk — but its
        //     status/stat-drop/flinch/confusion effect does NOT apply when the sub took
        //     the hit). A self-target damaging move never has a sub here (foe != self). ---
        // Capture the pre-hit HP of what the damage lands on (the mon, or the sub's HP)
        // so a recoil move (Struggle) can base its recoil on the ACTUAL damage dealt
        // (`move.totalDamage` — the sim's `damage()` return is clamped to the target's
        // remaining HP), not the raw computed roll. For a sub the base is the sub's HP.
        let target_hp_before = match self.sides[foe].pokemon[foe_slot].substitute {
            Some(sub_hp) => sub_hp,
            None => self.sides[foe].pokemon[foe_slot].hp,
        };
        let dealt = realized.min(target_hp_before);
        let sub = self.absorb_into_sub(foe, foe_slot, realized);
        let absorbed = sub != SubAbsorb::NoSub;

        // --- APPLY HP + faint at 0 (only when the sub did NOT absorb it). ---
        if !absorbed {
            // FOCUS BAND (`gen3_ability_batch4_v1`): the onDamage roll draws AFTER the
            // damage rolls, BEFORE the apply; a lethal MOVE hit that passes survives at
            // 1 HP (probe seed 8: crit path included). A sub-absorbed hit never draws.
            let realized = self.focus_band_damage(foe, foe_slot, realized, true, dex);
            self.apply_damage(foe, foe_slot, realized);
            // [EMIT] `|-damage|<foe>|<HP>` with the POST-damage HP (`x/y`, `x/y
            // <status>`, or `0 fnt` when the hit KO'd). Observation-only. A 0-damage
            // move that reaches here (none modeled) would still emit — matching the
            // sim's `-damage` on a 0 hit — but the modeled damaging moves always deal
            // >=1 when not absorbed.
            if realized > 0 && self.logging() {
                let target = self.mon_ref(foe, foe_slot, dex);
                let hp = self.hp_status(foe, foe_slot);
                self.log.damage(&target, &hp, None);
            }
        } else if self.logging() {
            // [EMIT] the SUBSTITUTE result (INSTEAD of `-damage`): a sub that SURVIVED
            // shows `|-activate|<foe>|Substitute|[damage]`; a sub that BROKE shows
            // `|-end|<foe>|Substitute`. VERIFIED vs the golden (both follow the `|move|`
            // + any effectiveness line, replacing the `-damage`). The effectiveness /
            // `-crit` lines emitted above are UNCHANGED (they still show for a sub hit).
            let target = self.mon_ref(foe, foe_slot, dex);
            match sub {
                SubAbsorb::Held => self.log.activate(&target, "Substitute", Some("[damage]")),
                SubAbsorb::Broke => self.log.volatile_end(&target, "Substitute"),
                SubAbsorb::NoSub => unreachable!(),
            }
        }

        // [EMIT] PAY DAY's `|-fieldactivate|move: Pay Day` — the coin-scatter onHit
        // marker (display-only; the handler-audit `move:payday:onHit` row). Emitted
        // AFTER the `-damage` line, on a landed DIRECT hit (Phase 3, byte-verified vs
        // the leechseed_splash_payday capture). A sub-absorbed Pay Day's form is
        // uncaptured → stays un-emitted behind a sub (the honesty discipline).
        if self.logging() && !struggle && !absorbed {
            let is_payday = self
                .move_at(side, slot, move_index, dex)
                .map(|m| m.id == "payday")
                .unwrap_or(false);
            if is_payday {
                self.log.fieldactivate_move("Pay Day");
            }
        }

        // --- 4. SECONDARY effects (the per-move random(100) AFTER the hit lands +
        //     HP applied), mirroring spreadMoveHit step 5 (battle-actions.ts:1120,
        //     after spreadDamage/runMoveEffects/selfDrops). For each surviving
        //     secondary: one random_below(100); apply if `< chance`. A KO'd target
        //     STILL draws the secondary (the mon is not `false`); the status no-ops
        //     on hp==0. A miss/immune returned earlier so never reaches here.
        //     `absorbed` (the sub took the hit): the secondary random(100) STILL DRAWS
        //     (the gen-3 draw is unconditional), but the EFFECT is SUPPRESSED (the sub
        //     blocked the foe-targeting secondary — incl. its confusion random(2,6)). ---
        //     Struggle has NO secondary — and `move_index` is the STALE scripted slot (the
        //     Choice-locked move, e.g. Crunch's −1 SpD), so it MUST NOT read that slot's
        //     secondaries. Skip the whole step for a forced Struggle (draw-free: Struggle
        //     itself has no secondary `random(100)`).
        if !struggle {
            self.apply_secondaries(side, slot, foe, foe_slot, move_index, absorbed, dex);
        }

        // --- KING'S ROCK appended flinch secondary (`gen3_ability_batch4_v1`) — an
        //     ORDINARY trailing secondary the holder's onModifyMove pushed for a LISTED
        //     move: rolled AFTER the move's own secondary (list order), BEFORE the foe's
        //     contact proc (probe_kingsrock_order_rng.js O1/O2/O3). Struggle IS listed
        //     (the synthetic id is passed explicitly — `move_index` is stale for it).
        //     Behind a sub the roll draws, the flinch is suppressed. ---
        {
            let kr_move_id: String = if struggle {
                "struggle".to_string()
            } else {
                match self.move_at(side, slot, move_index, dex) {
                    Some(m) => m.id.clone(),
                    None => String::new(),
                }
            };
            if !kr_move_id.is_empty() {
                self.apply_kings_rock_secondary(side, slot, foe, foe_slot, &kr_move_id, absorbed, dex);
            }
        }

        // --- CONTACT_PROC + CONTACT recoil (`gen3_ability_batch2_v1`) — the DEFENDER's
        //     reactive `onDamagingHit` ability. Fires INSIDE `runEvent('DamagingHit')` (gen<5,
        //     battle-actions.ts:982) which is AFTER `secondaries()` (line 957) — so the
        //     contact-proc `randomChance` draws AFTER the move's own secondary `random(100)`
        //     (VERIFIED vs the sim, `harness/probe_contact_proc_rng.js`). It fires on a CONTACT
        //     move that dealt damage DIRECTLY TO THE MON — NOT behind a SUBSTITUTE (a sub-absorbed
        //     hit never reaches the mon, so the mon's `onDamagingHit` does not fire — the SAME
        //     `!absorbed` gate as the fire-thaw below; VERIFIED by the adversarial review's behind-sub
        //     probe: a Static holder behind a SURVIVING sub leaves the attacker un-statused and draws
        //     BYTE-IDENTICAL to a no-ability control). It DOES fire on a KO (the DamagingHit event
        //     fires on the damaged/KO'd target). `dealt` is the damage dealt (mon or sub); the
        //     `!absorbed` gate excludes the sub case. The status lands on the ATTACKER (`side`/`slot`);
        //     Rough Skin deals recoil to the attacker. Struggle IS a contact move so it CAN proc.
        //     (Placed AFTER the fire-thaw's SIBLING position is fine — both are draw-free-except-
        //     this, on different mons; this is the only DamagingHit draw.) ---
        if is_contact && !absorbed && dealt > 0 {
            self.apply_contact_proc(side, slot, foe, foe_slot, dex);
        }

        // --- FIRE-MOVE THAW (gen3 frz.onDamagingHit, conditions.ts:45): a Fire
        //     damaging move cures the DEFENDER's freeze — DRAW-FREE, runs via
        //     runEvent('DamagingHit') AFTER secondaries (gen<5). Ice Beam (Ice) can't
        //     thaw; a Fire move (e.g. Fire Blast) on a frozen target does. A sub ABSORBED
        //     hit does NOT reach the mon's status, so a frozen-behind-a-sub mon is NOT
        //     thawed (the thaw is `onDamagingHit` on the MON; the sub took the damage). ---
        if is_fire && !absorbed && self.sides[foe].pokemon[foe_slot].status == Some(Status::Freeze) {
            self.sides[foe].pokemon[foe_slot].status = None;
        }

        // --- COLOR CHANGE (`gen3_ability_batch4_v1`) — the DEFENDER's onDamagingHit
        //     type override, the SAME DamagingHit region as the contact procs / thaw:
        //     NOT behind a sub (the mon's event never fires — probe t2), not on the KO
        //     hit, never for typeless ??? (Struggle). DRAW-FREE. ---
        if !absorbed && dealt > 0 {
            let cc_type = if struggle { None } else { move_type };
            self.apply_color_change(foe, foe_slot, cc_type, dex);
        }

        // --- STRUGGLE RECOIL (`gen3_pp_tracking_v1`) — the gen-3 `recoil:[1,4]` path
        //     (`data/mods/gen3/scripts.ts::calcRecoilDamage`), NOT the gen4+ `struggleRecoil
        //     = maxhp/4` path (gen-3 sets `struggleRecoil: false` + `recoil:[1,4]` — VERIFIED
        //     vs the sim). The user takes `max(floor(damageDealt / 4), 1)` HP, where
        //     `damageDealt` is `move.totalDamage` = the damage the hit dealt (to the mon, or
        //     to the sub when absorbed — `moveHit`'s return). Applied via the SAME faint
        //     machinery (a recoil KO faints the user), DRAW-FREE (the sim's `this.battle.damage`
        //     consumes no PRNG — it fires AT scripts.ts:461, AFTER the hit + BEFORE the
        //     in-tryMoveHit Update, so it does NOT reorder any draw). The recoil is `>=1` only
        //     when `move.totalDamage` is truthy (a >0 hit) — a 0-damage Struggle (into a mon at
        //     0 HP, unreachable here) applies none. Emitted as `|-damage|<user>|<HP>|[from]
        //     Recoil|[of] <target>`. Struggle deals damage only when it hit (this path is the
        //     landed-hit path), so recoil applies whenever the realized damage was >0. ---
        if struggle && dealt > 0 {
            let user_hp_before = self.sides[side].pokemon[slot].hp;
            let recoil = (dealt / 4).max(1).min(user_hp_before);
            // FOCUS BAND: the recoil is a Damage event into the user (effect 'recoil',
            // not a Move) — the roll draws, no survive. `gen3_ability_batch4_v1`.
            let recoil = self.focus_band_damage(side, slot, recoil, false, dex);
            self.apply_damage(side, slot, recoil);
            if self.logging() {
                let user = self.mon_ref(side, slot, dex);
                let target = self.mon_ref(foe, foe_slot, dex);
                let hp = self.hp_status(side, slot);
                self.log.damage_of(&user, &hp, &Cause::Bare("Recoil".into()), &target);
            }
        }

        // The move landed (acc-hit + non-immune; a sub hit STILL fires the in-tryMoveHit
        // Update — gen-3 `moveHit` returns a truthy `true` for a HIT_SUBSTITUTE so
        // `tryMoveHit`'s `if (!damage && damage !== 0) return` does NOT short-circuit).
        MoveResolution::done(false, crit, true)
    }

    /// Resolve + APPLY a FIXED-DAMAGE / FIXED-FORMULA move (Seismic Toss / Night Shade /
    /// Sonic Boom / Dragon Rage / Super Fang) — Showdown's `damage:` / `damageCallback`
    /// path that BYPASSES `getDamage`. The draw model, VERIFIED bit-for-bit vs the
    /// omniscient sim's PRNG probe (`harness/probe_fixeddamage_rng.js`):
    ///
    ///   1. **ACCURACY** — `randomChance(acc, 100)`, drawn unless `never_miss`. Seismic
    ///      Toss / Night Shade / Dragon Rage are acc-100 but `never_miss == false`, so
    ///      they STILL draw one accuracy roll (the phaze acc-100 precedent); Sonic Boom /
    ///      Super Fang are acc-90 and CAN genuinely miss. This is the ONLY per-move draw —
    ///      NO crit roll, NO 16-way damage roll, NO secondary.
    ///   2. **PROTECT BLOCK** — a fixed-damage move TARGETING a protected foe draws its
    ///      accuracy roll then is blocked at the `TryHit` event (before any damage), exactly
    ///      like the damaging path.
    ///   3. **TYPE IMMUNITY** — accuracy-drawn-THEN-`-immune` (the same short-circuit as a
    ///      normal damaging move): Seismic Toss (Fighting) into a GHOST, Night Shade (Ghost)
    ///      into a NORMAL, Sonic Boom / Super Fang (Normal) into a GHOST all report `-immune`
    ///      (NOT `-miss`) with the SAME draw count as a landed hit. Resolved via
    ///      `move_is_immune` over a `DamageContext` (0× type-chart OR ability immunity).
    ///   4. **APPLY** the FIXED amount (`fixed_damage_amount` — user's level / 20 / 40 /
    ///      max(floor(target.hp/2),1)) through the EXISTING `absorb_into_sub` (a sub takes
    ///      the number, breaks with no carry) / `apply_damage` / deferred-faint machinery —
    ///      identical to the damaging path, so a fixed-damage KO goes through the normal
    ///      faint/win/Quick-Claw protocol (no Quick Claw on a deciding faint).
    ///
    /// `landed` is TRUE on a hit (the in-`tryMoveHit` Update fires — a `damage:` move
    /// returns a truthy damage number, so `tryMoveHit`'s `if (!damage && damage !== 0)
    /// return` does NOT short-circuit); FALSE on a miss / immune / block.
    ///
    /// FAIL-LOUD: the DEFERRED fixed-damage family (Psywave / Fissure / Horn Drill /
    /// Guillotine / Counter / Mirror Coat / Bide / Endeavor) is routed here by
    /// `is_fixed_damage_move` but has NO `fixed_damage_amount` entry, so this PANICS
    /// rather than silently no-op (they need extra RNG or reactive/OHKO machinery).
    #[allow(clippy::too_many_arguments)]
    fn run_fixed_damage_move(
        &mut self,
        side: usize,
        slot: usize,
        foe: usize,
        foe_slot: usize,
        accuracy: u16,
        never_miss: bool,
        move_type: Option<Type>,
        move_id: &str,
        move_name: &str,
        targets_self: bool,
        dex: &Dex,
    ) -> MoveResolution {
        // FAIL-LOUD: only the modeled fixed-damage set has a formula. A DEFERRED
        // fixed-damage move (Psywave / OHKO / Counter / Mirror Coat / Bide / Endeavor)
        // reaches here via `is_fixed_damage_move` but has no amount → PANIC (never a
        // silent desync — these draw an unmodeled number/order or need reactive state).
        let amount = match fixed_damage_amount(
            move_id,
            &self.sides[side].pokemon[slot],
            &self.sides[foe].pokemon[foe_slot],
        ) {
            Some(a) => a,
            None => panic!(
                "unmodeled FIXED-DAMAGE move {move_id:?} routed to run_fixed_damage_move — \
                 Psywave (variable RNG) / the OHKO moves (Fissure/Horn Drill/Guillotine) / \
                 Counter / Mirror Coat / Bide / Endeavor are DEFERRED (they draw an unmodeled \
                 number/order or need reactive/OHKO machinery). Model it bit-for-bit or keep it \
                 off the pickable set — do NOT let it silently no-op."
            ),
        };

        // --- 1. ACCURACY: random(100) < effAcc (acc/eva stages + accMod folded via
        //     `roll_accuracy`), drawn unless never_miss. Drawn NOW (draw-order parity) but
        //     the miss decision is DEFERRED past the immunity check (gen3 reports `-immune`,
        //     not `-miss`, for an immune move even when the accuracy roll would have failed —
        //     the same draw count either way). ---
        let acc_hit =
            self.roll_accuracy(side, slot, foe, foe_slot, accuracy, never_miss, move_type, dex);

        // --- PROTECT BLOCK (foe side): if the accuracy roll PASSED and the foe is
        //     `protected` and this move TARGETS the foe, the move is BLOCKED at the
        //     `TryHit` event — drawing NO damage. DRAW-FREE (the accuracy roll already
        //     happened). Precedes the immunity report (protect wins TryHit). A miss falls
        //     through to the genuine-miss return below (never reaching this). ---
        if acc_hit && self.protect_blocks(foe, foe_slot, targets_self) {
            // [EMIT] `|move|<attacker>|<Name>|<protector>` then `|-activate|…Protect`.
            if self.logging() {
                let user = self.mon_ref(side, slot, dex);
                let target = self.mon_ref(foe, foe_slot, dex);
                self.log.move_used(&user, move_name, Some(&target), false, false);
                self.log.activate(&target, "Protect", None);
            }
            return MoveResolution::done(false, false, false);
        }

        // --- IMMUNITY short-circuit: a type-chart 0× (Fighting→Ghost, Ghost→Normal,
        //     Normal→Ghost) or an ability immunity. Build a DamageContext (no PRNG) purely
        //     to reuse `move_is_immune` — the base_power/category are placeholders (a
        //     fixed-damage move's amount is `fixed_damage_amount`, not the calc), only the
        //     move TYPE + the defender types/ability matter for the immunity read. ---
        let ctx = self.build_damage_context(
            side, slot, foe, foe_slot, /*base_power*/ 1, move_type, MoveCategory::Physical,
            /*halves_def*/ false, dex,
        );
        if move_is_immune(&ctx, dex) {
            // [EMIT] `|move|<user>|<Name>|<foe>` then `|-immune|<foe>` (gen3 reports
            // `-immune`, NOT `-miss`, even when the accuracy roll would have failed).
            if self.logging() {
                let user = self.mon_ref(side, slot, dex);
                let target = self.mon_ref(foe, foe_slot, dex);
                self.log.move_used(&user, move_name, Some(&target), false, false);
                self.log.immune(&target);
            }
            // A fixed-damage move deals TYPELESS raw damage (no Water/Volt Absorb heal —
            // those trigger only on Water/Electric moves, and no modeled fixed-damage move
            // is Water/Electric). `move_is_immune` covers the type-chart 0× + Levitate;
            // nothing to heal. Not landed → no in-tryMoveHit Update, 0 further draws.
            return MoveResolution::done(!acc_hit, false, false);
        }

        // --- A genuine accuracy miss (non-immune move): no damage, NOT landed. ---
        if !acc_hit {
            // [EMIT] `|move|<user>|<Name>|<foe>|[miss]` then `|-miss|<user>|<foe>`.
            if self.logging() {
                let user = self.mon_ref(side, slot, dex);
                let target = self.mon_ref(foe, foe_slot, dex);
                self.log.move_used(&user, move_name, Some(&target), true, false);
                self.log.miss(&user, Some(&target));
            }
            return MoveResolution::done(true, false, false);
        }

        // [EMIT] `|move|<user>|<Name>|<foe>` (a landed fixed-damage move). NO effectiveness
        // line (`-supereffective`/`-resisted`): a `damage:` move deals TYPELESS raw damage —
        // the sim shows no effectiveness marker for a landed Seismic Toss (VERIFIED: the
        // probe's landed rows emit only `|move|` + `|-damage|`, no `-supereffective`).
        if self.logging() {
            let user = self.mon_ref(side, slot, dex);
            let target = self.mon_ref(foe, foe_slot, dex);
            self.log.move_used(&user, move_name, Some(&target), false, false);
        }

        // --- APPLY the FIXED amount through the sub-absorb / faint machinery (identical
        //     to the damaging path). NO crit, NO damage roll — `amount` is the exact fixed
        //     value. A sub takes the number (breaks with no carry); else the mon takes it
        //     (fainting at 0 via the deferred-faint protocol). A fixed-damage move has NO
        //     secondary, so nothing follows the apply. ---
        let sub = self.absorb_into_sub(foe, foe_slot, amount);
        let absorbed = sub != SubAbsorb::NoSub;
        if !absorbed {
            // FOCUS BAND: a fixed-damage move is a MOVE-effect Damage event — the roll
            // draws and a lethal Seismic Toss can be survived at 1. `gen3_ability_batch4_v1`.
            let amount = self.focus_band_damage(foe, foe_slot, amount, true, dex);
            self.apply_damage(foe, foe_slot, amount);
            // [EMIT] `|-damage|<foe>|<HP>` with the POST-damage HP. The modeled fixed-
            // damage moves always deal >= 1 to a non-immune target, so the emit fires.
            if amount > 0 && self.logging() {
                let target = self.mon_ref(foe, foe_slot, dex);
                let hp = self.hp_status(foe, foe_slot);
                self.log.damage(&target, &hp, None);
            }
        } else if self.logging() {
            // [EMIT] the SUBSTITUTE result (INSTEAD of `-damage`): a survived absorb shows
            // `|-activate|<foe>|Substitute|[damage]`, a broken sub `|-end|<foe>|Substitute`.
            let target = self.mon_ref(foe, foe_slot, dex);
            match sub {
                SubAbsorb::Held => self.log.activate(&target, "Substitute", Some("[damage]")),
                SubAbsorb::Broke => self.log.volatile_end(&target, "Substitute"),
                SubAbsorb::NoSub => unreachable!(),
            }
        }

        // --- KING'S ROCK on a fixed-damage LISTED move (Seismic Toss — probed: the
        //     appended secondary rolls right after the damage apply, `random(100,)` with
        //     no crit/randomizer preceding it) + COLOR CHANGE (a fixed-damage move has a
        //     real move TYPE for the override). Same DamagingHit-region ordering as the
        //     damaging path. `gen3_ability_batch4_v1`. ---
        self.apply_kings_rock_secondary(side, slot, foe, foe_slot, move_id, absorbed, dex);
        if !absorbed && amount > 0 {
            self.apply_color_change(foe, foe_slot, move_type, dex);
        }

        // The move landed (a sub hit STILL fires the in-tryMoveHit Update). NO crit.
        MoveResolution::done(false, false, true)
    }

    /// Resolve + APPLY a STANDALONE STATUS MOVE (category Status, bp 0, whose PURPOSE
    /// is a major status — Thunder Wave / Stun Spore / Glare [par], Toxic [tox], Poison
    /// Powder / Poison Gas [psn], Will-O-Wisp [brn], Spore / Sleep Powder / Hypnosis /
    /// Sing / Lovely Kiss / Grass Whistle [slp]). The gen-3 `tryMoveHit` status path
    /// (`data/mods/gen3/scripts.ts`), VERIFIED bit-for-bit vs the omniscient sim:
    ///
    ///   1. **MOVE-TYPE IMMUNITY** (`runImmunity`, DRAW-FREE) — for a Status move,
    ///      `move.ignoreImmunity` defaults to `true` so type immunity is IGNORED,
    ///      EXCEPT the two gen-3 status moves that set `ignoreImmunity: false`:
    ///      **Thunder Wave** (Electric → a GROUND target is immune) and **Glare**
    ///      (Normal → a GHOST target is immune). The resolved `natural_immunity`
    ///      short-circuits AFTER accuracy (so the draw count is identical to a damaging
    ///      immune move: accuracy-only, no status). No OTHER modeled status move is
    ///      type-blocked here (Will-O-Wisp's Fire-immunity, Toxic's Poison/Steel etc.
    ///      live in `try_set_status`).
    ///   2. **ACCURACY** `random_chance(accuracy, 100)` — ALWAYS drawn (unless
    ///      `never_miss`), EVEN when type-immune (gen3 draws accuracy then reports
    ///      `-immune`). A genuine miss (acc fail, non-immune) ends here.
    ///   3. **APPLY** via `try_set_status` (the onTrySetStatus gates: already-statused
    ///      / gen-3 status-type immunity / ability immunity [Insomnia/Vital Spirit slp,
    ///      etc.] / Sleep Clause) — sleep draws the `random(2,6)` duration onStart,
    ///      Toxic begins at stage 0 (draw-free; the residual ramps it to 1 before the
    ///      first chip, mirroring the sim's `statusState.stage`).
    ///
    /// NO crit, NO damage, NO secondary, and `landed` is ALWAYS FALSE — a status move
    /// returns `undefined` from `moveHit`, so the in-`tryMoveHit` `eachEvent('Update')`
    /// shuffle (`scripts.ts:470`) is SKIPPED by the `scripts.ts:468` guard.
    ///
    /// FAIL-LOUD: any status move NOT in the modeled set PANICS (mirroring the >1-col
    /// secondary guard) so a future status move can never silently desync.
    #[allow(clippy::too_many_arguments)]
    fn run_status_move(
        &mut self,
        // The caster side/slot (the USER — the self-boost / recovery / substitute /
        // self-Rest arms mutate it, and the `|move|` announce reads it).
        _side: usize,
        _slot: usize,
        foe: usize,
        foe_slot: usize,
        accuracy: u16,
        never_miss: bool,
        move_type: Option<Type>,
        move_id: &str,
        move_name: &str,
        targets_self: bool,
        status_inflicted: Option<&str>,
        // Whether the TARGET (foe) still has a PENDING move action this turn (`willMove(target)`,
        // `gen3_taunt_disable_v1`) — the Disable duration `+1` fires when this is FALSE (the
        // target has ALREADY moved, i.e. the disabler is slower / moved 2nd). Unused by every
        // other status-move arm.
        foe_will_move: bool,
        dex: &Dex,
    ) -> MoveResolution {
        // [EMIT] `|move|<user>|<Name>|<target>` — the status-move ANNOUNCE, emitted for
        // EVERY status move that routes here (setup / recovery / Rest / Spikes / Roar /
        // Thunder Wave / Substitute / Leech Seed / …), at the very top of the status path
        // (BEFORE any branch), so a fail / block / immune line follows it. Two forms:
        //  - the `[still]` empty-target form (`|move|<user>|<Name>||[still]`) — Showdown
        //    uses it when the move did NOTHING observable. Among the moves routing here,
        //    the only top-level `[still]` case is **Spikes at the 3-layer cap** (a 4th
        //    Spikes fails → `[still]` + `-fail`, verified vs the golden). Every other fail
        //    (Substitute already-up / can't-afford, Rest at full HP) still shows the
        //    TARGET form + `-fail` (verified). (Protect's success/fail `[still]` split is
        //    handled in `run_protect`, which returns before this fn.)
        //  - otherwise the TARGET form: a SELF-target move (`target: self` — Rest /
        //    Recover / Substitute / setup) renders the USER; a foe / foeSide move
        //    (Thunder Wave / Toxic / Spikes-success / Roar / Leech Seed) renders the FOE
        //    ACTIVE (`|move|p1a: Skarmory|Spikes|p2a: Tyranitar`).
        // Observation-only: reads already-resolved state, draws nothing.
        // An ALREADY-STATUSED foe-status-move FAIL classification (see
        // `foe_status_move_fail`) — drives BOTH the move-announce form here (a
        // different-status fail shows the `[still]` empty-target form + a fail on the
        // USER, mirroring `attrLastMove('[still]')`) and the `-fail` line in the
        // standalone-status arm below. `None` = not this case (announce normally).
        let status_fail = self.foe_status_move_fail(foe, foe_slot, move_id, targets_self, status_inflicted, dex);
        if self.logging() {
            let user = self.mon_ref(_side, _slot, dex);
            // Spikes at the cap is a top-level `[still]` (did-nothing) case; a
            // DIFFERENT-status status-move fail is the other (`|move|…||[still]` — the
            // move's `trySetStatus(this.status)` re-passes the foe's OWN status, so
            // `sourceEffect.status != this.status` → `add('-fail', source)` + `[still]`).
            // (Phase 3 fix: the DIFFERENT-status fail's `[still]` form is a RETRO-EDIT at
            // the fail site (3a), NOT an up-front announce form — the accuracy roll comes
            // FIRST, and a MISSED move keeps its target-form announce + gains `[miss]`
            // (byte-verified: `|move|…|Hypnosis|<target>|[miss]` into a paralyzed Hypno).
            // Spikes-at-cap stays up-front (never-miss, no roll before the fail).)
            let still_form = move_id == "spikes" && self.sides[foe].spikes >= 3;
            if still_form {
                self.log.move_used(&user, move_name, None, false, true);
            } else {
                let target = if targets_self {
                    user.clone()
                } else {
                    self.mon_ref(foe, foe_slot, dex)
                };
                self.log.move_used(&user, move_name, Some(&target), false, false);
            }
        }
        // --- PURE SELF-BOOST SETUP MOVE (Swords Dance / Dragon Dance / Calm Mind /
        //     Agility / Bulk Up / …) — a `target: self` Status move whose ENTIRE effect
        //     is raising the USER'S stat stages. The gen-3 `tryMoveHit` self-boost path:
        //
        //       1. ACCURACY: every modeled setup move is NEVER-MISS (`accuracy: true`),
        //          so NO accuracy draw. (Handled generally: draw `random_chance(acc,100)`
        //          iff NOT never_miss — defensive; the modeled set is all never-miss, so
        //          this never draws. A non-never-miss setup move is excluded.)
        //       2. APPLY `boost()` on the USER (battle-actions.ts → `this.boost(boosts,
        //          source)`), each (stat, stages) clamped to ±6. **DRAW-FREE** — `boost()`
        //          consumes NO PRNG (like `apply_secondary_boost`). Our OWN Clear Body /
        //          White Smoke / Hyper Cutter / Keen Eye do NOT block our OWN self-boost
        //          (the `onTryBoost` immunity is for FOE-inflicted drops only). A boost
        //          into the +6 cap is a no-op but still "succeeds" (draws nothing).
        //       3. `landed` is ALWAYS FALSE — a status `moveHit` returns `undefined`, so
        //          the in-`tryMoveHit` `eachEvent('Update')` shuffle is SKIPPED (identical
        //          to the status-inflicting path).
        //
        //     **Speed interaction (the cached-speed crux):** a +Spe boost (Dragon Dance /
        //     Agility) updates `boosts[4]` IMMEDIATELY, but `MonState::cached_speed`
        //     (`pokemon.speed`) is NOT refreshed here — Showdown re-establishes it only at
        //     the next re-cache site (the residual's `updateSpeed`, the next turn's
        //     `commitChoices`, or a switch-in). So THIS turn's later `eachEvent` tie-
        //     shuffles still read the PRE-boost cached speed, and the NEXT turn's action
        //     order picks up the boosted speed at turn-start. We deliberately do NOT call
        //     `update_speed()` (matching the stale-between-sites model the residual fix
        //     established). ---
        if let Some(self_boosts) = self_boost_spec(move_id, dex) {
            // ACCURACY (defensive — the modeled set is never-miss, so this is skipped).
            if !never_miss {
                let acc_hit = self.roll_accuracy(_side, _slot, foe, foe_slot, accuracy, never_miss, move_type, dex);
                if !acc_hit {
                    return MoveResolution::done(true, false, false);
                }
            }
            // APPLY the self-boost on the USER (`_side`/`_slot` is the caster). DRAW-FREE,
            // ±6 clamp, no immunity gate (self-boosts are never blocked). Emit `|-boost|`
            // per stat with the CLAMPED delta actually applied — a boost into the +6 cap is
            // a no-op-but-success that emits NOTHING (the delta is 0), matching the golden.
            for &(idx, stages) in &self_boosts {
                let cur = self.sides[_side].pokemon[_slot].boosts[idx] as i32;
                let next = (cur + stages as i32).clamp(-6, 6);
                self.sides[_side].pokemon[_slot].boosts[idx] = next as i8;
                if self.logging() {
                    let delta = (next - cur) as i8; // the applied (post-clamp) magnitude
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.boost(&user, idx, delta);
                }
            }
            // A status move never fires the in-tryMoveHit Update shuffle: not missed,
            // NOT landed. cached_speed is deliberately left STALE (see above).
            return MoveResolution::done(false, false, false);
        }

        // --- SELF-HEAL / RECOVERY MOVE (category Status / bp 0 / target self / isHeal) —
        //     Recover / Soft-Boiled / Slack Off / Milk Drink (heal floor(maxhp/2)),
        //     Moonlight / Synthesis / Morning Sun (WEATHER-conditional heal), and Rest
        //     (full heal + self-sleep + status cure). The gen-3 (gen4-inherited)
        //     `tryMoveHit` self-heal path — verified bit-for-bit vs the omniscient sim:
        //
        //       1. ACCURACY: every recovery move is NEVER-MISS (`accuracy: true`) → NO
        //          accuracy draw. (Defensive: draw `random_chance(acc,100)` iff NOT
        //          never_miss; the modeled set is all never-miss, so this never draws.)
        //       2. HEAL (`this.heal(amount)` on the USER) — DRAW-FREE (`heal` consumes
        //          no PRNG). Amounts are INTEGER truncations (gen3 `maxhp == baseMaxhp`):
        //            * Recover/Soft-Boiled/Slack Off/Milk Drink: `floor(maxhp/2)` (the
        //              `move.heal:[1,2]` path, `Math.floor(baseMaxhp*1/2)`).
        //            * Moonlight/Synthesis/Morning Sun (gen4-inherited `onHit`): NONE →
        //              `floor(maxhp/2)`; SUN → `floor(maxhp*2/3)`; SAND/RAIN/HAIL →
        //              `floor(maxhp/4)`. (NOT the 4096-`modify` — the gen4 override is
        //              plain integer arithmetic, VERIFIED: Espeon maxhp 271 in sun heals
        //              `floor(271*2/3)=180`, not `modify(271,0.667)=181`.)
        //          The HEAL-AT-FULL-HP / heal-0 case FAILS (`heal` returns false →
        //          `-fail`), draw-free either way.
        //       3. `landed` is ALWAYS FALSE — a status `moveHit` returns `undefined`, so
        //          the in-`tryMoveHit` `eachEvent('Update')` shuffle is SKIPPED.
        //
        //     REST is the one move with a draw subtlety (see `run_rest`): it DRAWS one
        //     `random(2,6)` (gen-3 `slp.onStart` ALWAYS rolls the sleep duration) but
        //     Rest's `onHit` then OVERWRITES the stored time to a FIXED `Sleep(3)` — so
        //     the draw is consumed-then-DISCARDED (verified vs a sim PRNG probe; it is
        //     NOT draw-free, unlike the task's original assumption). Plus a full heal, a
        //     prior-status cure (Rest overrides), and the gen3ou-only SetStatus
        //     handler-sort shuffle BEFORE that `random(2,6)` on the self-`setStatus('slp')`
        //     (gated by `sleep_clause`; gen3customgame draws only the `random(2,6)`). A
        //     self-Rest sleep is EXEMPT from the Sleep Clause cap (it never blocks). ---
        if move_id == "rest" {
            return self.run_rest(_side, _slot, dex);
        }
        // WEATHER_NEGATE (`gen3_ability_batch1_v1`): Moonlight / Synthesis / Morning Sun read
        // the EFFECTIVE weather for their heal fraction — a Cloud Nine / Air Lock mon on the
        // field makes them heal the no-weather `maxhp/2` (`effective_weather`).
        if let Some(amount) = recovery_heal_amount(move_id, &self.sides[_side].pokemon[_slot], self.effective_weather(dex)) {
            // HEAL the USER (DRAW-FREE). A full-HP / heal-0 heal FAILS (`-fail`) but draws
            // nothing extra — the seed is unchanged either way; only the user's HP changes.
            let healed = self.apply_heal(_side, _slot, amount);
            // [EMIT] a Recover-family heal has NO `[from]` tag (`|-heal|<user>|<HP>`); a
            // full-HP / heal-0 fail emits `|-fail|<user>` (the `runMoveEffects` `-fail`).
            if self.logging() {
                let user = self.mon_ref(_side, _slot, dex);
                if healed {
                    let hp = self.hp_status(_side, _slot);
                    self.log.heal(&user, &hp, None);
                } else {
                    self.log.fail(&user, None, false);
                }
            }
            // A status move never fires the in-tryMoveHit Update shuffle: not missed,
            // NOT landed. No accuracy draw (never-miss). cached_speed is untouched.
            return MoveResolution::done(false, false, false);
        }

        // --- SPLASH (a true NO-OP status move, never-miss) — gen-3 `splash` does literally
        //     nothing (`onTry` → `-nothing`, then the move ends). DRAW-FREE: no accuracy
        //     (never-miss), no effect, `landed` false (no in-tryMoveHit Update). It is the
        //     "do nothing" filler the recovery golden uses to take chip without attacking;
        //     modeling it as a draw-free no-op keeps the engine faithful (it never changes
        //     HP/status/boosts and never draws). ---
        if move_id == "splash" {
            // [EMIT] `|-nothing` — Splash's onTryHit marker (Phase 3, byte-verified vs
            // the leechseed_splash_payday capture: `|move|<user>|Splash|<user>` then
            // `|-nothing`). Observation-only.
            if self.logging() {
                self.log.nothing();
            }
            return MoveResolution::done(false, false, false);
        }

        // --- SPIKES (the gen-3 ENTRY HAZARD — the first SIDE CONDITION, `sideCondition:
        //     "spikes"`, `target: "foeSide"`). NEVER-MISS (`accuracy: true`) so NO accuracy
        //     draw. It is the FOE-side-targeting hazard: a Spikes cast INCREMENTS the
        //     CASTER's FOE side's `spikes` layer count by 1, CAPPED at 3 (the
        //     `onSideRestart` `if (layers >= 3) return false` → a Spikes at 3 FAILS,
        //     `-fail`). DRAW-FREE both ways (the side condition's `onSideStart`/
        //     `onSideRestart` consume NO PRNG — the only effect is bumping `effectState.
        //     layers`); `landed` is FALSE (a status `moveHit` returns `undefined` → the
        //     in-`tryMoveHit` Update shuffle is skipped). The actual switch-in damage is
        //     applied LATER, when a grounded mon switches into the spiked side (see
        //     `apply_entry_hazards` on the gen-3 `runSwitch`'s `runEvent('EntryHazard')`).
        //     VERIFIED bit-for-bit vs the omniscient sim (`harness/probe_spikes_rng.js`): a
        //     Spikes-vs-move turn draws ONLY the existing action-order/eachEvent shuffles
        //     (the move itself adds nothing), and a Spikes-at-max draws nothing extra. The
        //     `foe` here is the move's foe side (Spikes targets `foeSide`). DEFERRED
        //     (fail-loud below): Toxic Spikes / Stealth Rock (NOT gen3), Rapid Spin (the
        //     hazard-clear move) — Spikes is the only gen-3 entry hazard. ---
        if move_id == "spikes" {
            // gen-3 caps at 3 layers; at 3 the (re)start FAILS (draw-free, `-fail`).
            if self.sides[foe].spikes < 3 {
                self.sides[foe].spikes += 1;
                // [EMIT] `|-sidestart|<foe-side>|Spikes` — the side condition begins /
                // stacks. `<side>` is `p<N>: <PlayerName>` (no position letter).
                if self.logging() {
                    let side_ref =
                        crate::protocol::ProtocolBuilder::side_ref(foe, &self.sides[foe].name);
                    self.log.sidestart(&side_ref, "Spikes");
                }
            } else {
                // [EMIT] `|-fail|<caster>` — a 4th Spikes at the 3-layer cap fails. The
                // caster is the mon that USED the move (`_side`/`_slot`).
                if self.logging() {
                    let caster = self.mon_ref(_side, _slot, dex);
                    self.log.fail(&caster, None, false);
                }
            }
            return MoveResolution::done(false, false, false);
        }

        // --- PHAZE (Roar / Whirlwind — the gen-3 `forceSwitch: true` moves) — force the
        //     FOE to switch to a RANDOM team member. VERIFIED bit-for-bit vs the omniscient
        //     sim's PRNG probe (`harness/probe_phaze_rng.js`). The gen-3 draw model:
        //
        //       1. PRIORITY −6 → the phazer almost always moves LAST (handled by
        //          `sort_actions` reading the dex priority; not this fn's concern).
        //       2. ACCURACY — gen-3 Roar/Whirlwind have **`accuracy: 100`** (NOT `true`!),
        //          so they DO draw `randomChance(100, 100)` (which ALWAYS passes but
        //          CONSUMES a draw). This is the surprise the probe surfaced: the base
        //          Showdown data shows `accuracy: true`, but the resolved gen-3 dex value
        //          is 100, so a phaze is NOT never-miss — it draws the accuracy roll just
        //          like a 100-acc status move. A category-Status move routes here BEFORE
        //          `run_move`'s damaging-path accuracy draw, so we draw it in THIS arm.
        //       3. THE RANDOM TARGET DRAW — `forceSwitch` (battle-actions.ts:1167) sets
        //          `target.forceSwitchFlag = true` IFF `canSwitch(target.side)` (the foe
        //          has >= 1 eligible non-active, non-fainted bench mon). The ACTUAL drag
        //          (`getRandomSwitchable` → `sample(canSwitchIn)` → `this.random(n)` — ONE
        //          draw, EVEN when n == 1, since `random(1)` returns 0 but still calls
        //          `rng.next()`) happens LATER, in the runAction tail (battle.ts:2350),
        //          AFTER this whole move's body. So here we only DECIDE eligibility +
        //          signal the pending drag; the sample draw + the swap + the runSwitch
        //          (EntryHazard/Spikes → ability Start) are done by `turn_loop` via
        //          `drag_in` (mirroring the source's runAction → dragIn order). A phaze
        //          with NO eligible target (the foe's last mon alive) FAILS: NO
        //          forceSwitchFlag, NO drag, NO `sample` draw (the accuracy roll already
        //          drew — that's the only draw this turn for the phaze move). ---
        if modeled_phaze_move(move_id) {
            // GIGO guard: the resolved gen-3 dex must agree (accuracy 100, not never-miss).
            // If a data drift makes a phaze never-miss, the accuracy-draw model is wrong.
            debug_assert!(
                !never_miss && accuracy == 100,
                "phaze {move_id:?} expected gen-3 accuracy 100 + not never_miss, got \
                 accuracy={accuracy} never_miss={never_miss}"
            );
            let _ = move_type; // phaze type is irrelevant (forceSwitch ignores type immunity)
            // (2) ACCURACY — `randomChance(accuracy, 100)`, drawn unless never-miss. Gen-3
            //     Roar/Whirlwind are accuracy 100 → this ALWAYS passes but CONSUMES the
            //     draw. A genuine miss (defensive — never happens at 100) ends the move.
            if !never_miss {
                let acc_hit = self.roll_accuracy(_side, _slot, foe, foe_slot, accuracy, never_miss, move_type, dex);
                if !acc_hit {
                    return MoveResolution::done(true, false, false);
                }
            }
            // (2a) SOUNDPROOF — Roar IS a `flags.sound` move (Whirlwind is NOT), so a Roar into a
            //      Soundproof holder is IMMUNE at `onTryHit` (after the accuracy roll): `-immune|
            //      [from] ability: Soundproof`, NO `forceSwitchFlag`, NO drag → **NO `sample`
            //      draw**. VERIFIED vs the sim (probe above): identical to the natural-immune
            //      status path (accuracy drawn, then immune, no further roll). This precedes the
            //      Protect block (a Soundproof holder is immune before any TryHit drag signal).
            if self.move_is_sound(move_id, dex)
                && dex.ability(&to_id(&self.sides[foe].pokemon[foe_slot].ability)).map(|a| a.blocks_sound).unwrap_or(false)
            {
                // [EMIT] `|-immune|<target>|[from] ability: Soundproof`. No drag, no `-fail`.
                if self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.immune_from_ability(&target, "Soundproof");
                }
                return MoveResolution::done(false, false, false);
            }
            // (2b) PROTECT BLOCK — gen-3 Roar / Whirlwind BOTH carry the `protect: 1` flag, so
            //      a Protect / Detect on the target BLOCKS the phaze at the `TryHit` event
            //      (`runEvent('TryHit')`, `scripts.ts:369`, AFTER the accuracy roll above). A
            //      blocked phaze draws its accuracy then is blocked → `-activate Protect`, and
            //      crucially NO `forceSwitchFlag` is set → NO drag → **NO `sample` draw** (the
            //      runAction-tail `dragIn` never fires). Missing this was the multi-phaze
            //      `sample` desync: the port dragged (an extra `sample`) into a protected mon
            //      the sim left in place, shifting every later phaze's `sample` PRNG position.
            //      VERIFIED vs the sim (`harness/probe_phaze_regression_rng.js` PHAZE-PROTECT):
            //      a Protect-blocked Roar draws NO `sample` and leaves the target active. This
            //      mirrors the leechseed / standalone-status arms' `protect_blocks` check.
            //      (Substitute does NOT block a phaze — Roar/Whirlwind carry `bypasssub: 1`, so
            //      there is intentionally no substitute check here; only Protect blocks.)
            if self.protect_blocks(foe, foe_slot, false) {
                // [EMIT] `|-activate|<protector>|Protect` — the blocked phaze (the `|move|`
                // announce already showed). No drag, no `-fail`. Observation-only.
                if self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.activate(&target, "Protect", None);
                }
                return MoveResolution::done(false, false, false);
            }
            // (2c) SUCTION CUPS (`gen3_ability_batch2_v1`, `suctioncups.onDragOut`) — a phaze
            //      into a Suction Cups holder does NOT drag it. In the sim, `forceSwitch`
            //      (battle-actions.ts:1166) runs `runEvent('DragOut')` INSIDE the move body: if
            //      it returns falsy (Suction Cups' `onDragOut` returns `null`), `forceSwitchFlag`
            //      is NOT set → the runAction tail's `if (forceSwitchFlag)` is false → **no
            //      `dragIn` → NO `sample` draw** (the accuracy roll already drew — that's all).
            //      Since `onDragOut` returns `null` (not `false`), the `hitResult === false`
            //      `-fail` branch does NOT fire — only the ability's own `-activate Suction Cups`.
            //      VERIFIED vs the sim (`harness/probe_block_abilities_rng.js`: a Roar into a
            //      Suction Cups mon draws its accuracy then `-activate`, drawing NO sample; the
            //      mon stays active). Note this is BEFORE `canSwitch` matters (the DragOut gate is
            //      independent of whether the foe has a bench — but we only reach here on a
            //      non-Protect-blocked, non-Soundproof phaze). ---
            if dex.ability(&to_id(&self.sides[foe].pokemon[foe_slot].ability)).map(|a| a.blocks_phaze_drag).unwrap_or(false) {
                // [EMIT] `|-activate|<holder>|ability: Suction Cups` — the drag blocked. No drag,
                // no `sample`, no `-fail`. The holder stays active.
                if self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.activate(&target, "ability: Suction Cups", None);
                }
                return MoveResolution::done(false, false, false);
            }
            // (3) `canSwitch(foe.side)`: the foe has an eligible (non-active, non-fainted)
            //     bench mon. If so, signal the pending drag (consumed at the runAction
            //     tail — the `sample` draw + the swap + the runSwitch); else the phaze
            //     FAILS draw-free (the accuracy roll already drew — that's all).
            let eligible = self.eligible_switch_ins(foe);
            let force = if eligible.is_empty() { None } else { Some(foe) };
            // [EMIT] a phaze with NO eligible foe bench mon FAILS → `|-fail|<caster>`. The
            // `|drag|` line for a SUCCESS is emitted later at the runAction tail (Phase 1's
            // `drag_in`). (Never realized in the modeled spikes_and_phaze scenarios — the
            // foe always has bench — but emitted for completeness.)
            if force.is_none() && self.logging() {
                let caster = self.mon_ref(_side, _slot, dex);
                self.log.fail(&caster, None, false);
            }
            return MoveResolution { missed: false, crit: false, landed: false, force_switch_foe: force };
        }

        // --- TAUNT (`taunt` — a foe-targeting `volatileStatus:'taunt'` Status move, type Dark,
        //     accuracy 100). The gen-3 `tryMoveHit` path, VERIFIED bit-for-bit vs the omniscient
        //     sim (`harness/probe_taunt_disable_rng.js` + `probe_taunt_disable_duration.js`):
        //
        //       1. ACCURACY — gen-3 Taunt is `accuracy: 100` (NOT never-miss), so it DRAWS
        //          `randomChance(100,100)` (which ALWAYS passes but CONSUMES a draw) — the phaze
        //          acc-100 precedent. This is the ONLY per-move draw (NO duration draw — the
        //          `taunt` volatile is `duration: 2` FIXED, no `durationCallback`).
        //       2. PROTECT BLOCK — Taunt carries `protect: 1`, so a Protect / Detect on the target
        //          BLOCKS it at TryHit (after the accuracy roll) → `-activate Protect`, NO volatile.
        //          Substitute does NOT block Taunt (`bypasssub: 1` → NO substitute check).
        //       3. APPLY the `taunt` volatile on the FOE with duration 2 (DRAW-FREE). A re-Taunt
        //          into an already-taunted foe FAILS (`addVolatile` false; accuracy drawn, no
        //          change) — draw-free past accuracy. `landed` is FALSE (a status `moveHit`
        //          returns `undefined` → the in-`tryMoveHit` Update is skipped). The end-of-turn
        //          residual duration handler (order 10, subOrder 15) ticks it down + expires it.
        //     While taunted the foe cannot SELECT any Status move (`move_usable` folds this in),
        //     forcing Struggle if only Status moves remain. ---
        if move_id == "taunt" {
            // GIGO guard: the resolved gen-3 dex must agree (Dark, accuracy 100, not never-miss).
            debug_assert!(
                !never_miss && accuracy == 100 && move_type == Some(Type::Dark),
                "taunt expected gen-3 Dark / accuracy 100 / not never_miss, got \
                 accuracy={accuracy} never_miss={never_miss} type={move_type:?}"
            );
            // (1) ACCURACY — randomChance(100,100), drawn unless never-miss (it isn't).
            if !never_miss {
                let acc_hit = self.roll_accuracy(_side, _slot, foe, foe_slot, accuracy, never_miss, move_type, dex);
                if !acc_hit {
                    // [EMIT] a genuine miss (never at 100, but defensive).
                    if self.logging() {
                        let user = self.mon_ref(_side, _slot, dex);
                        let target = self.mon_ref(foe, foe_slot, dex);
                        self.log.miss(&user, Some(&target));
                    }
                    return MoveResolution::done(true, false, false);
                }
            }
            // (2) PROTECT BLOCK (foe-targeting): blocked at TryHit after accuracy. Substitute
            //     does NOT block (bypasssub) → no substitute check.
            if self.protect_blocks(foe, foe_slot, false) {
                if self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.activate(&target, "Protect", None);
                }
                return MoveResolution::done(false, false, false);
            }
            // (3) APPLY the taunt volatile (duration 2), OR fail if already taunted (DRAW-FREE).
            if self.sides[foe].pokemon[foe_slot].taunt.is_none() {
                self.sides[foe].pokemon[foe_slot].taunt = Some(TAUNT_DURATION);
                // [EMIT] `|-start|<target>|move: Taunt` (the volatile's onStart).
                if self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.volatile_start(&target, "move: Taunt");
                }
            } else if self.logging() {
                // [EMIT] a re-Taunt into an already-taunted foe: the announce is retro-
                // edited to the `[still]` did-nothing form (`|move|<user>|Taunt||[still]`,
                // the sim's `attrLastMove('[still]')` on `addVolatile` false) and the
                // `-fail` is on the USER (`add('-fail', source)`). Byte-verified vs the
                // taunt_lifecycle capture (Phase 3 — the pre-Phase-3 target-form + fail-on-
                // target was never byte-gated and was wrong).
                let user = self.mon_ref(_side, _slot, dex);
                self.log.attr_last_move_still();
                self.log.fail(&user, None, false);
            }
            return MoveResolution::done(false, false, false);
        }

        // --- DISABLE (`disable` — a foe-targeting `volatileStatus:'disable'` Status move, type
        //     Normal, accuracy 55). Disables the FOE's LAST-USED move for a RANDOM duration. The
        //     gen-3 path, VERIFIED bit-for-bit vs the omniscient sim
        //     (`harness/probe_taunt_disable_rng.js` + the onStart/duration probes):
        //
        //       1. ACCURACY — gen-3 Disable is `accuracy: 55` (NOT 100! the task hint was wrong;
        //          NOT never-miss), so it DRAWS `randomChance(55,100)` and CAN genuinely miss.
        //       2. onTryHit FAIL (DRAW-FREE, BEFORE the duration draw): if the target has NO
        //          `last_move` (hasn't moved / just switched in) OR its lastMove was Struggle
        //          (`last_move == None`), Disable FAILS — accuracy was drawn, then `-fail`, NO
        //          `random(2,6)`. VERIFIED: a Disable into a not-yet-moved mon draws ONLY accuracy.
        //       3. PROTECT BLOCK — Disable carries `protect: 1`, so Protect / Detect BLOCKS it at
        //          TryHit (after accuracy). Substitute does NOT block (`bypasssub: 1`).
        //       4. DURATION DRAW — on a landed hit into a mon with a real lastMove, `addVolatile`
        //          draws ONE `random(2,6)` (∈ {2,3,4,5}) for the duration, then onStart does
        //          `duration += 1` iff the target has ALREADY moved this turn (`!foe_will_move` —
        //          the disabler is slower / moved 2nd). This is the ONLY extra draw. It disables
        //          the lastMove's SLOT — UNLESS that slot has 0 PP left: the gen4-inherited
        //          onStart 0-PP guard then REJECTS the volatile AFTER the draw (`-fail`, no
        //          `-start`, volatiles stay empty — see (4b) below, probe-verified).
        //       5. ALREADY-DISABLED — a re-Disable into an already-disabled foe: gen-3
        //          `addVolatile` returns false, so onStart's durationCallback is NOT reached → NO
        //          `random(2,6)`. Accuracy drawn, then `-fail`. DRAW-FREE past accuracy.
        //          (VERIFIED addVolatile-false short-circuits the durationCallback.)
        //       6. `landed` is ALWAYS FALSE (a status `moveHit` returns `undefined`). The residual
        //          duration handler (order NO_ORDER, subOrder 2 — ties with protect/stall/flinch)
        //          ticks it down + frees the move at 0. Cleared on switch-out.
        //     While disabled the foe cannot SELECT that ONE slot (`move_usable` folds this in). ---
        if move_id == "disable" {
            // GIGO guard: the resolved gen-3 dex must agree (Normal, accuracy 55, not never-miss).
            debug_assert!(
                !never_miss && accuracy == 55 && move_type == Some(Type::Normal),
                "disable expected gen-3 Normal / accuracy 55 / not never_miss, got \
                 accuracy={accuracy} never_miss={never_miss} type={move_type:?}"
            );
            // (1) ACCURACY — randomChance(55,100), drawn unless never-miss (it isn't). CAN miss.
            let acc_hit = if never_miss {
                true
            } else {
                self.roll_accuracy(_side, _slot, foe, foe_slot, accuracy, never_miss, move_type, dex)
            };
            if !acc_hit {
                // [EMIT] the announce is retro-edited with the `[miss]` attr
                // (`|move|<user>|Disable|<target>|[miss]`, the sim's `attrLastMove('[miss]')`)
                // then `|-miss|<user>|<target>`. Byte-verified vs the disable_lifecycle
                // capture (Phase 3).
                if self.logging() {
                    let user = self.mon_ref(_side, _slot, dex);
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.attr_last_move_miss();
                    self.log.miss(&user, Some(&target));
                }
                return MoveResolution::done(true, false, false);
            }
            // (3) PROTECT BLOCK (foe-targeting): blocked at TryHit after accuracy. Substitute does
            //     NOT block (bypasssub). Checked BEFORE the onTryHit lastMove guard, mirroring the
            //     event order (TryHit protect handler runs at the same event; a protected target
            //     never reaches addVolatile — no `random(2,6)`). VERIFIED: a Protect-blocked
            //     Disable draws only accuracy.
            if self.protect_blocks(foe, foe_slot, false) {
                if self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.activate(&target, "Protect", None);
                }
                return MoveResolution::done(false, false, false);
            }
            // (2) onTryHit FAIL (DRAW-FREE): no valid lastMove slot to disable (never moved /
            //     just switched in / lastMove was Struggle → `last_move == None`).
            let last = self.sides[foe].pokemon[foe_slot].last_move;
            let last_slot = match last {
                Some(k) if k < self.sides[foe].pokemon[foe_slot].move_pp.len() => k,
                _ => {
                    // -fail, draw-free (no random(2,6)). The announce is retro-edited to
                    // the `[still]` did-nothing form (`|move|<user>|Disable||[still]` +
                    // `|-fail|<user>` — byte-verified vs the disable_lifecycle capture,
                    // Phase 3; the pre-Phase-3 target-form announce was the documented
                    // "retro-edit nit").
                    if self.logging() {
                        let user = self.mon_ref(_side, _slot, dex);
                        self.log.attr_last_move_still();
                        self.log.fail(&user, None, false);
                    }
                    return MoveResolution::done(false, false, false);
                }
            };
            // (5) ALREADY-DISABLED — addVolatile returns false → NO durationCallback draw.
            //     Same `[still]` retro-edit + `-fail` on the USER as the no-lastMove fail
            //     (byte-verified, Phase 3).
            if self.sides[foe].pokemon[foe_slot].disable.is_some() {
                if self.logging() {
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.attr_last_move_still();
                    self.log.fail(&user, None, false);
                }
                return MoveResolution::done(false, false, false);
            }
            // (4) DURATION DRAW — addVolatile's `durationCallback: random(2,6)` (∈ {2,3,4,5}), then
            //     onStart `duration++` iff the target has ALREADY moved this turn (`!foe_will_move`).
            let rolled = self.prng.random_range(2, 6); // random(2,6) → 2..=5
            let duration = if foe_will_move { rolled } else { rolled + 1 };
            // (4b) THE onStart 0-PP GUARD (`gen3_disable_zero_pp_v1` — the gen4-inherited
            //     `!moveSlot.pp → return false`, which gen3's condition INHERITS; gen3 only
            //     overrides durationCallback + the residual orders). Reachable in real gen3
            //     play: the target spends its lastMove's FINAL PP (e.g. a mono-move mon now
            //     forced to Struggle — lastMove still names the exhausted slot until another
            //     move executes), then gets Disabled. The `random(2,6)` above was STILL drawn
            //     (addVolatile fires the durationCallback BEFORE onStart), so the draw is
            //     consumed — but onStart returns false, the volatile is REMOVED, and NO
            //     residual duration handler registers (recording it anyway would let the
            //     phantom handler TIE with a taunt/stall/flinch duration handler → an extra
            //     residual tie-shuffle draw → a future seed desync; and `disable` state
            //     would be wrong). Protocol: the announce is retro-edited to the `[still]`
            //     did-nothing form (`attrLastMove('[still]')`) + `|-fail|<user>`, NO `-start`.
            //     VERIFIED vs the sim (`harness/probe_disable_zero_pp_rng.js`: draws =
            //     accuracy + random(2,6), then `|move|p1a: Suicune|Disable||[still]` +
            //     `|-fail|p1a: Suicune`, target volatiles EMPTY).
            if self.sides[foe].pokemon[foe_slot].move_pp[last_slot] == 0 {
                if self.logging() {
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.attr_last_move_still();
                    self.log.fail(&user, None, false);
                }
                return MoveResolution::done(false, false, false);
            }
            // Disable the lastMove's slot for `duration` turns (DRAW-FREE apply). A
            // just-used move normally has PP — the 0-PP rejection above is the one
            // exception — so past the guard we record the volatile at the resolved slot.
            self.sides[foe].pokemon[foe_slot].disable = Some((last_slot, duration as u8));
            // [EMIT] `|-start|<target>|Disable|<MoveName>` (the volatile's onStart).
            if self.logging() {
                let target = self.mon_ref(foe, foe_slot, dex);
                let mv_name = self.sides[foe].pokemon[foe_slot]
                    .set
                    .moves
                    .get(last_slot)
                    .and_then(|mid| dex.moves(mid))
                    .map(|m| m.name.clone())
                    .unwrap_or_default();
                self.log.volatile_start(&target, &format!("Disable|{mv_name}"));
            }
            return MoveResolution::done(false, false, false);
        }

        // --- LEECH SEED (`leechseed` — a foe-targeting `volatileStatus:'leechseed'`
        //     Status move, type Grass, accuracy 90). The gen-3 `tryMoveHit` path, VERIFIED
        //     bit-for-bit vs the omniscient sim (`harness/probe_leechseed_rng.js`):
        //
        //       1. ACCURACY — gen-3 Leech Seed is `accuracy: 90` (NOT never-miss), so it
        //          DRAWS `randomChance(90,100)` — it CAN miss. The accuracy roll is drawn
        //          UNCONDITIONALLY, even into a Grass-immune or already-seeded target (the
        //          immunity/fail is reported only AFTER the accuracy roll). VERIFIED: a
        //          splash/splash baseline turn draws 1 (Quick Claw); a Leech-Seed turn —
        //          land, Grass-immune, OR already-seeded-fail — ALL draw 2 (accuracy + QC).
        //       2. GRASS IMMUNITY (`onTryImmunity` → `!target.hasType('Grass')`) — a Grass
        //          target is IMMUNE: the accuracy roll is still drawn, then `-immune`, NO
        //          volatile. DRAW-FREE past accuracy.
        //       3. ALREADY-SEEDED — a 2nd Leech Seed on a seeded target FAILS (`addVolatile`
        //          returns false): the accuracy roll is drawn, then `-fail`/"did nothing",
        //          the existing volatile is UNCHANGED. DRAW-FREE past accuracy.
        //       4. PROTECT BLOCK — a foe-targeting move into a `protected` mon is blocked at
        //          TryHit after the accuracy roll (`-activate Protect`, no volatile).
        //       5. PLANT the `leechseed` volatile on the foe (DRAW-FREE — `addVolatile`'s
        //          `onStart` only adds `-start`, no PRNG), recording the SEEDER's side. The
        //          end-of-turn drain/heal is the `LeechSeed` residual (see `apply_leech_seed`).
        //       6. `landed` is ALWAYS FALSE — a status `moveHit` returns `undefined`, so the
        //          in-`tryMoveHit` `eachEvent('Update')` shuffle is SKIPPED.
        //
        //     DEFERRED (fail-loud at the residual): a Liquid Ooze target reverses the drain
        //     (`apply_leech_seed` panics) — rare in gen-3 OU, excluded from the e2e filter.
        if move_id == "leechseed" {
            // GIGO guard: the resolved gen-3 dex must agree (Grass, accuracy 90, not never-miss).
            debug_assert!(
                !never_miss && accuracy == 90 && move_type == Some(Type::Grass),
                "leechseed expected gen-3 Grass / accuracy 90 / not never_miss, got \
                 accuracy={accuracy} never_miss={never_miss} type={move_type:?}"
            );
            // (1) ACCURACY — randomChance(90,100), drawn unless never-miss (it isn't).
            let acc_hit = if never_miss {
                true
            } else {
                self.roll_accuracy(_side, _slot, foe, foe_slot, accuracy, never_miss, move_type, dex)
            };
            // (2) GRASS IMMUNITY — resolved AFTER the accuracy draw (same draw count).
            let target_is_grass = mon_types(&self.sides[foe].pokemon[foe_slot], dex)
                .contains(&Type::Grass);
            if target_is_grass {
                // [EMIT] `|-immune|<target>` — the plain type-immune form, announce
                // UN-edited (byte-verified vs the leechseed_splash_payday capture, Phase 3).
                if self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.immune(&target);
                }
                return MoveResolution::done(false, false, false); // no volatile
            }
            // A genuine accuracy miss (non-immune): end here, no volatile.
            if !acc_hit {
                // [EMIT] the `[miss]` retro-edit + `|-miss|<user>|<target>` (byte-verified
                // vs the leechseed_splash_payday capture, Phase 3).
                if self.logging() {
                    self.log.attr_last_move_miss();
                    let user = self.mon_ref(_side, _slot, dex);
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.miss(&user, Some(&target));
                }
                return MoveResolution::done(true, false, false);
            }
            // (4) PROTECT BLOCK (foe-targeting): blocked at TryHit after accuracy.
            if self.protect_blocks(foe, foe_slot, false) {
                // [EMIT] `|-activate|<target>|Protect` (the standard block line — same
                // form as every other protected foe-targeting status move).
                if self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.activate(&target, "Protect", None);
                }
                return MoveResolution::done(false, false, false);
            }
            // (4b) SUBSTITUTE BLOCK — a Leech Seed into a substituted foe is BLOCKED by the
            //      sub (the `volatileStatus` is a foe-targeting effect → the sub's
            //      `onTryPrimaryHit` returns before `addVolatile`): accuracy drawn, then the
            //      volatile is NOT planted. VERIFIED vs the sim probe (a landed Leech Seed
            //      into a subbed mon leaves it UN-seeded). DRAW-FREE past accuracy.
            //      Protocol (F1, `harness/probe_f1_f2_f3_lines.js`): the sim's `moveHit`
            //      returns `false` for the sub-blocked primary hit, so gen3 retro-edits the
            //      announce to `|move|<user>|Leech Seed||[still]` and emits `|-fail|<user>` —
            //      IDENTICAL to the already-seeded fail form. DRAW-FREE emit (reads state only).
            if self.sides[foe].pokemon[foe_slot].substitute.is_some() {
                // [EMIT] the `[still]` retro-edit + `|-fail|<user>` (byte-verified vs the
                // leechseed_splash_payday capture's sub-block arm, F1).
                if self.logging() {
                    self.log.attr_last_move_still();
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.fail(&user, None, false);
                }
                return MoveResolution::done(false, false, false);
            }
            // (3) ALREADY-SEEDED — a re-seed FAILS, the existing volatile is unchanged.
            if self.sides[foe].pokemon[foe_slot].leech_seed.is_some() {
                // [EMIT] the `[still]` retro-edit (`|move|<user>|Leech Seed||[still]`) +
                // `|-fail|<user>` (byte-verified vs the capture, Phase 3).
                if self.logging() {
                    self.log.attr_last_move_still();
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.fail(&user, None, false);
                }
                return MoveResolution::done(false, false, false);
            }
            // (5) PLANT the leechseed volatile on the foe (DRAW-FREE), seeder = caster side.
            self.sides[foe].pokemon[foe_slot].leech_seed = Some(_side);
            // [EMIT] `|-start|<target>|move: Leech Seed` (the volatile's onStart —
            // byte-verified vs the capture, Phase 3).
            if self.logging() {
                let target = self.mon_ref(foe, foe_slot, dex);
                self.log.volatile_start(&target, "move: Leech Seed");
            }
            // (6) Status move: not missed, NOT landed (no in-tryMoveHit Update).
            return MoveResolution::done(false, false, false);
        }

        // --- SUBSTITUTE (`substitute` — a self-targeting `volatileStatus:'substitute'`
        //     Status move) — the user spends `floor(maxhp/4)` HP to create a decoy with
        //     that much HP that ABSORBS incoming foe hits until it breaks. The gen-3
        //     `tryMoveHit` path, VERIFIED bit-for-bit vs the omniscient sim's PRNG probe
        //     (`harness/probe_substitute_rng.js`):
        //
        //       1. NEVER-MISS (`accuracy: true`) → NO accuracy draw.
        //       2. FAIL (DRAW-FREE) at `onTryHit` if a `substitute` is ALREADY present
        //          (`-fail`) OR the user can't afford it: `source.hp <= source.maxhp / 4`
        //          (a FLOAT compare; the Shedinja `maxhp === 1` clause is N/A here) — VERIFIED:
        //          hp == floor(maxhp/4) FAILS, hp == floor(maxhp/4)+1 SUCCEEDS. So the gate is
        //          `hp <= floor(maxhp/4)` for an integer hp (the float `maxhp/4` rounds the
        //          boundary the same way since hp is an integer ≤ that float iff ≤ its floor).
        //       3. On SUCCESS: `onHit` → `this.directDamage(maxhp/4)` subtracts `floor(maxhp/4)`
        //          from the user's HP, and the volatile's `onStart` sets `effectState.hp =
        //          floor(maxhp/4)` (a `-start`, DRAW-FREE — no PRNG).
        //       4. `landed` is ALWAYS FALSE — a status `moveHit` returns `undefined`, so the
        //          in-`tryMoveHit` `eachEvent('Update')` shuffle is SKIPPED.
        //
        //     The ABSORB / block side (a foe move INTO a substituted mon) lives in `run_move`
        //     / `apply_secondaries` / `on_before_move`, NOT here. The cost is `floor(maxhp/4)`,
        //     matching the sub's HP (gen-3: `directDamage(maxhp/4)` floors). ---
        if move_id == "substitute" {
            // GIGO guard: the resolved gen-3 dex must agree (never-miss, self-target, Status).
            debug_assert!(
                never_miss,
                "substitute expected gen-3 never_miss (accuracy:true), got never_miss={never_miss}"
            );
            let mon = &self.sides[_side].pokemon[_slot];
            let cost = sub_cost(mon.maxhp); // floor(maxhp/4)
            // FAIL (draw-free): already-subbed OR can't afford (hp <= floor(maxhp/4)).
            if mon.substitute.is_some() || mon.hp <= cost {
                // [EMIT] `|-fail|<user>|move: Substitute` (already-subbed — the sub is
                // still up) OR `|-fail|<user>|move: Substitute|[weak]` (can't afford the
                // HP cost — the too-weak fail). VERIFIED vs the golden: already-subbed has
                // NO `[weak]`; can't-afford carries `[weak]`.
                if self.logging() {
                    let already_subbed = self.sides[_side].pokemon[_slot].substitute.is_some();
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.fail(&user, Some("move: Substitute"), !already_subbed);
                }
                return MoveResolution::done(false, false, false);
            }
            // SUCCESS: pay the cost + create the substitute with hp == cost. DRAW-FREE.
            let mon = &mut self.sides[_side].pokemon[_slot];
            mon.hp -= cost; // directDamage(floor(maxhp/4)); cost < hp guaranteed above
            mon.substitute = Some(cost);
            // [EMIT] `|-start|<user>|Substitute` (the volatile's onStart) THEN
            // `|-damage|<user>|<HP>` (the directDamage cost) — the golden order is start
            // BEFORE the cost damage. Observation-only.
            if self.logging() {
                let user = self.mon_ref(_side, _slot, dex);
                self.log.volatile_start(&user, "Substitute");
                let hp = self.hp_status(_side, _slot);
                self.log.damage(&user, &hp, None);
            }
            // A status move is never landed (no in-tryMoveHit Update).
            return MoveResolution::done(false, false, false);
        }

        // FAIL-LOUD GUARD: only the explicitly-modeled foe-targeting major-status
        // moves are allowed. Any other status move (recovery/boost/phaze/hazard/field)
        // would draw an UNMODELED number/order of PRNG calls → a silent desync. PANIC
        // instead (like the >1-secondary guard). (Substitute is modeled above.)
        let status = match modeled_status_move(move_id) {
            Some(s) => s,
            None => panic!(
                "status move {move_id:?} is not modeled — its execution path (recovery / \
                 boost / phaze / hazard / field-target) would draw an unmodeled number/order \
                 of PRNG calls and silently desync. Model it (or exclude it from \
                 isModeledMove) before a battle can use it."
            ),
        };
        // The dex `status` field MUST agree with the modeled mapping (a GIGO guard: a
        // data drift that changed e.g. Toxic's status would silently mis-apply).
        debug_assert_eq!(
            status_inflicted,
            Some(status),
            "modeled status move {move_id:?} dex status {status_inflicted:?} != {status:?}"
        );

        // --- 1. MOVE-TYPE IMMUNITY (DRAW-FREE), only for ignoreImmunity:false status
        //     moves (Thunder Wave / Glare). The type-chart 0× check mirrors gen3
        //     `runImmunity(move)` (which here uses the MOVE type, not the status). All
        //     other modeled status moves IGNORE type immunity. ---
        let natural_immunity = if status_move_checks_type_immunity(move_id) {
            match move_type {
                Some(t) => {
                    let def_types = mon_types(&self.sides[foe].pokemon[foe_slot], dex);
                    dex.type_chart().effectiveness(t, &def_types) == 0.0
                }
                None => false,
            }
        } else {
            false
        };

        // --- 2. ACCURACY: random_chance(accuracy, 100), drawn unless never_miss.
        //     Drawn UNCONDITIONALLY (even when type-immune) — gen3 draws accuracy then
        //     reports `-immune`. Same draw count either way. ---
        let acc_hit = if never_miss {
            true
        } else {
            self.roll_accuracy(_side, _slot, foe, foe_slot, accuracy, never_miss, move_type, dex)
        };

        // Type-immune (accuracy was still drawn): `-immune`, no status, no further draw.
        if natural_immunity {
            // [EMIT] `|-immune|<target>` (Thunder Wave → Ground, Glare → Ghost).
            if self.logging() {
                let target = self.mon_ref(foe, foe_slot, dex);
                self.log.immune(&target);
            }
            return MoveResolution::done(false, false, false);
        }

        // --- SOUNDPROOF (`gen3_ability_batch2_v1`, `soundproof.onTryHit`) — a SOUND status
        //     move (Sing / Grass Whistle) into a Soundproof holder is IMMUNE. Fires at
        //     `runEvent('TryHit')` (AFTER the accuracy roll — the same draw model as the
        //     type-immune path: accuracy drawn, then `-immune|[from] ability: Soundproof`, no
        //     status, no SetStatus shuffle, no sleep `random(2,6)`). VERIFIED vs the sim
        //     (`harness/probe_block_abilities_rng.js`: Sing into Soundproof draws its 55-accuracy
        //     roll then `-immune`, drawing NO further roll). ---
        if self.move_is_sound(move_id, dex)
            && dex.ability(&to_id(&self.sides[foe].pokemon[foe_slot].ability)).map(|a| a.blocks_sound).unwrap_or(false)
        {
            // [EMIT] `|-immune|<target>|[from] ability: Soundproof`.
            if self.logging() {
                let target = self.mon_ref(foe, foe_slot, dex);
                self.log.immune_from_ability(&target, "Soundproof");
            }
            return MoveResolution::done(false, false, false);
        }
        // A genuine accuracy miss (non-immune): end here, no status.
        if !acc_hit {
            // [EMIT] the `[miss]` retro-edit on the announce (`attrLastMove('[miss]')` —
            // the announce at the top already showed WITHOUT it) then `|-miss|<user>|
            // <target>` (Will-O-Wisp 75, Hypnosis 60, …). Byte-verified vs the
            // status_immune_lines capture (Phase 3).
            // (The pre-Phase-3 claim that "a status-move `|move|` never carries `[miss]`"
            // was an artifact of a corpus with no status-move miss — the Phase-3 capture
            // DISPROVED it: `|move|p1a: Gengar|Hypnosis|p2a: Hypno|[miss]` + `|-miss|`.)
            if self.logging() {
                self.log.attr_last_move_miss();
                let user = self.mon_ref(_side, _slot, dex);
                let target = self.mon_ref(foe, foe_slot, dex);
                self.log.miss(&user, Some(&target));
            }
            return MoveResolution::done(true, false, false);
        }

        // --- PROTECT BLOCK (foe-targeting status move): the accuracy roll passed; if the
        //     target is `protected`, the status move is BLOCKED at `TryHit` (after
        //     accuracy, before the status applies) — `-activate Protect`, NO status set,
        //     no further draw. This path only handles foe-targeting major-status moves
        //     (Thunder Wave / Toxic / Will-O-Wisp / Spore / …); self-target status (the
        //     setup / recovery moves) returned earlier, so the target here is always the
        //     foe → no self-target exemption needed. VERIFIED vs the sim (Thunder Wave
        //     into a protected mon draws its accuracy then `-activate Protect`, no status,
        //     no SetStatus shuffle since the status never reaches `setStatus`). ---
        if self.protect_blocks(foe, foe_slot, false) {
            // [EMIT] `|-activate|<protector>|Protect` — a foe-targeting status move blocked
            // by Protect (the `|move|` announce already showed at the top). No status set.
            if self.logging() {
                let target = self.mon_ref(foe, foe_slot, dex);
                self.log.activate(&target, "Protect", None);
            }
            return MoveResolution::done(false, false, false);
        }

        // --- SUBSTITUTE BLOCK (foe-targeting major-status move): a Thunder Wave / Toxic /
        //     Will-O-Wisp / Spore / … into a substituted foe is BLOCKED by the sub — the
        //     sub's `onTryPrimaryHit` returns (`-fail`) BEFORE `setStatus`, so the accuracy
        //     was drawn (above) but NO status applies AND the `runEvent('SetStatus')`
        //     handler-sort shuffle does NOT fire (setStatus is never reached) AND a landed
        //     sleep's `random(2,6)` is NOT drawn. VERIFIED vs the sim probe (Thunder Wave
        //     into a subbed mon draws its accuracy then `-fail`, no status). DRAW-FREE past
        //     accuracy. (A self-target status move returned earlier, so the target is always
        //     the foe here.) ---
        if self.sides[foe].pokemon[foe_slot].substitute.is_some() {
            return MoveResolution::done(false, false, false);
        }

        // --- FLASH FIRE ABSORB of a Fire-type STATUS move (`gen3_ff_wisp_absorb_v1` — gen3
        //     Will-O-Wisp is the only member): the resolved gen3 `flashfire.onTryHit`
        //     special-cases WoW — a Fire-TYPE / already-STATUSED / SUBBED target FALLS
        //     THROUGH to the normal gates (the sub block above; the already-statused 3a +
        //     brn→Fire type immunity below — all already modeled), but otherwise the holder
        //     ABSORBS the move: the `flash_fire` volatile ARMS and NO status applies.
        //     DRAW-FREE past the accuracy roll (blocked BEFORE `setStatus` → no SetStatus
        //     clause shuffle even in gen3ou, no sleep/burn follow-on). CURRENT-ability read
        //     (a TRACED Flash Fire absorbs too — the A/B cluster's Porygon2). Probe:
        //     `harness/probe_flashfire_rng.js` A3 (WoW LANDED on a NON-Fire FF Snorlax →
        //     `flashfire` volatile armed, status '-', no burn; a Fire-type FF Houndoom →
        //     NO activation, the type-immunity gate burns nothing either). The A/B fuzzer's
        //     willowisp STATE cluster: the port burned a traced-FF Porygon2 (maxhp/8 DoT
        //     desync) the sim absorbed. ---
        {
            let fm = &self.sides[foe].pokemon[foe_slot];
            if move_type == Some(Type::Fire)
                && to_id(&fm.ability) == "flashfire"
                && fm.status.is_none()
                && !mon_types(fm, dex).contains(&Type::Fire)
            {
                self.sides[foe].pokemon[foe_slot].flash_fire = true;
                return MoveResolution::done(false, false, false);
            }
        }

        // --- 3a. ALREADY-STATUSED FAIL — a foe already carrying a major status can't take
        //     another. gen-3 `trySetStatus` re-passes the foe's OWN status to `setStatus`,
        //     so `status.id === this.status` → the fail is emitted at `setStatus`
        //     (pokemon.ts:1699) BEFORE the SetStatus event → DRAW-FREE (the accuracy roll
        //     above was the only draw; the `-fail` never reaches `runEvent('SetStatus')`,
        //     so no clause handler-sort shuffle even in gen3ou). Two forms, split by whether
        //     the move's inflicted status MATCHES the foe's current status:
        //       - SAME (`sourceEffect.status === this.status`, e.g. Thunder Wave→par into
        //         par): `|-fail|<target>|<status>` — the fail on the TARGET, status token.
        //       - DIFFERENT (`sourceEffect.status` set but `!= this.status`, e.g. Thunder
        //         Wave→par into brn): `|-fail|<user>` + the move announce's `[still]` form
        //         (emitted above). The fail is on the USER, NO status token.
        //     VERIFIED vs the sim (`harness/probe_status_move_fail_lines.js` +
        //     `probe_status_fail_accuracy.js`): both forms are draw-free past accuracy, and
        //     the same-status form draws accuracy just like a clean apply. `try_set_status`
        //     is NOT called here (it would no-op silently) — we emit the fail + return.
        if let Some(kind) = status_fail {
            if self.logging() {
                match kind {
                    StatusMoveFail::Same => {
                        let target = self.mon_ref(foe, foe_slot, dex);
                        // The status token of the foe's CURRENT status (== the move's
                        // inflicted status in the Same case), e.g. `par`.
                        let tok = status_token(self.sides[foe].pokemon[foe_slot].status)
                            .unwrap_or(status);
                        self.log.fail(&target, Some(tok), false);
                    }
                    StatusMoveFail::Different => {
                        // `-fail` on the USER (`add('-fail', source)`) + the `[still]`
                        // RETRO-EDIT on the announce (`attrLastMove('[still]')` — applied
                        // HERE, after the accuracy roll passed; a missed move never
                        // reaches this and keeps its `[miss]` form instead).
                        self.log.attr_last_move_still();
                        let user = self.mon_ref(_side, _slot, dex);
                        self.log.fail(&user, None, false);
                    }
                }
            }
            return MoveResolution::done(false, false, false);
        }

        // --- 3b. APPLY the status via try_set_status (the onTrySetStatus gates +
        //     Sleep Clause + the sleep random(2,6) onStart). DRAW-FREE except a landed
        //     sleep's duration draw. (An already-statused foe was handled in 3a above,
        //     so this path always targets an un-statused foe — a re-status no-op here
        //     would silently emit no line, which is why the fail-emit lives in 3a.) The
        //     source `(_side, _slot)` is the attacker (for the Synchronize reflect). ---
        // `announce_immune_block=true`: a status MOVE blocked by a setStatus-phase
        // STATUS_IMMUNE ability announces `|-immune|…|[from] ability: <A>` (Phase 3).
        self.try_set_status_impl(foe, foe_slot, status, Some((_side, _slot)), true, None, dex);

        // A status move never fires the in-tryMoveHit Update shuffle (moveHit returns
        // undefined): not missed, NOT landed.
        MoveResolution::done(false, false, false)
    }

    /// REST (`data/moves.ts::rest`, gen-3 — the `onTry` + `onHit` path), verified
    /// bit-for-bit vs the omniscient sim. Rest:
    ///   1. **FULL-HP GUARD** (`onTry`): if the user is at full HP, Rest FAILS (`-fail
    ///      heal`) — it does NOT sleep, NOT heal, NOT cure (draw-free). (A user already
    ///      asleep also fails, but a still-asleep mon can't choose a move, so it's
    ///      unreachable here.)
    ///   2. **NEVER-MISS** — no accuracy draw (`accuracy: true`).
    ///   3. **SELF-SLEEP** (`onHit` → `target.setStatus('slp')`): Rest's `setStatus('slp')`
    ///      runs the gen-3 `slp.onStart`, which ALWAYS draws `random(2,6)` for the duration
    ///      (`effectState.time = this.random(2, 6)`) — so the DRAW HAPPENS — and Rest's
    ///      `onHit` THEN OVERWRITES it to a FIXED 3 (`statusState.time = startTime = 3`). So
    ///      the draw COUNT includes the `random(2,6)` (its VALUE is discarded), and the
    ///      stored counter is a FIXED `Sleep(3)`. This is the draw-COUNT subtlety: a Rest
    ///      DOES consume the `random(2,6)` (VERIFIED vs the sim's PRNG probe — the Rest turn
    ///      draws one MORE `random(2,6)` than a no-sleep turn), but the resulting sleep
    ///      length is always 3 (not the rolled 1-4). The user then wakes via the EXISTING
    ///      `on_before_move` sleep counter (3 → cant → cant → wake+move) — bit-identical to
    ///      the sim's 3-attempt Rest sleep.
    ///   4. **STATUS CURE** — `setStatus` overrides any prior major status (a paralyzed
    ///      Rester becomes asleep; the prior para/tox is gone). DRAW-FREE.
    ///   5. **FULL HEAL** — `this.heal(target.maxhp)` → HP to maxhp (the heal is silent /
    ///      aesthetic-only since the sleep already set in; draw-free).
    ///   6. **The gen3ou SetStatus handler-sort shuffle** — Rest's self-`setStatus('slp')`
    ///      reaches `runEvent('SetStatus')`, which in a clause format (gen3ou) draws the
    ///      size-2 Sleep/Freeze-Clause handler-sort shuffle ONE `random(0,2)`
    ///      (`set_status_event_shuffle`), gated by `sleep_clause`; gen3customgame (0
    ///      handlers) draws nothing (VERIFIED: the Rest turn's seed delta differs between
    ///      gen3ou and gen3customgame by exactly one shuffle). A self-Rest sleep is
    ///      EXEMPT from the Sleep Clause CAP (it never blocks the other side's sleep and
    ///      a 2nd self-Rest is fine), so we draw the shuffle but do NOT run the
    ///      `side_has_sleeper` block.
    ///   7. `landed` is ALWAYS FALSE — a status `moveHit` returns `undefined`, so the
    ///      in-`tryMoveHit` `eachEvent('Update')` shuffle is SKIPPED.
    fn run_rest(&mut self, side: usize, slot: usize, dex: &Dex) -> MoveResolution {
        let mon = &self.sides[side].pokemon[slot];
        // (1) FULL-HP GUARD (onTry: `if source.hp === source.maxhp` → `-fail heal`,
        //     return null). A fainted mon never reaches here. Draw-free — return before
        //     the SetStatus shuffle (the sim's `onTry` precedes `singleEvent('Try')`'s
        //     success and the moveHit → setStatus path entirely).
        if mon.hp >= mon.maxhp {
            // [EMIT] a Rest at full HP fails `|-fail|<user>|heal` — the heal-fail path
            // carries the `heal` detail token (byte-verified vs the writeline capture,
            // `gen3_writeline_stream_v1` — the Phase-2 corpus never realized a full-HP
            // Rest, so the missing token went unnoticed until the per-write gate).
            if self.logging() {
                let user = self.mon_ref(side, slot, dex);
                self.log.fail(&user, Some("heal"), false);
            }
            return MoveResolution::done(false, false, false);
        }

        // (6) The gen3ou SetStatus handler-sort shuffle (the self-`setStatus('slp')`
        //     reaches `runEvent('SetStatus')`). Gated by `sleep_clause`; gen3customgame
        //     draws nothing. A self-Rest is EXEMPT from the Sleep-Clause CAP, so we draw
        //     the shuffle (the 2 clause handlers always tie) but never run the sleeper
        //     block.
        if self.sleep_clause {
            self.set_status_event_shuffle();
        }

        // (3)+(4) SELF-SLEEP + status CURE (setStatus overrides any prior major status).
        //     `setStatus('slp')` runs `slp.onStart`, which DRAWS `random(2,6)` for the
        //     duration — so the DRAW HAPPENS (we must consume it for seed parity) — and
        //     Rest's `onHit` then OVERWRITES the time to a FIXED 3. So: draw + DISCARD the
        //     value, then store a FIXED Sleep(3). The counter is decremented draw-free by
        //     the existing on_before_move handler (3 attempts → wake).
        let _discarded = self.prng.random_range(2, 6); // slp.onStart's random(2,6) — value discarded
        self.sides[side].pokemon[slot].status = Some(Status::Sleep(3));

        // [EMIT] `|-status|<user>|slp|[from] move: Rest` — the self-sleep, carrying the
        // Rest provenance. (setStatus overriding a prior status does NOT emit a
        // `-curestatus` in gen3 — verified vs the golden; only a natural wake does.)
        if self.logging() {
            let user = self.mon_ref(side, slot, dex);
            self.log.status(&user, "slp", Some(&Cause::Move("Rest".to_string())));
        }

        // LUM's IMMEDIATE eat fires on Rest's self-setStatus too (`gen3_berry_trace_shedskin_v1`
        // — the classic LumRest: the sleep is cured RIGHT HERE, BEFORE the heal below, so the
        // user rests to FULL and is instantly awake. Probe: `|-status slp|[from] move: Rest| →
        // |-enditem|Lum Berry|[eat]| → |-curestatus|slp|[msg]| → |-heal|…|[silent]` with the
        // heal line showing NO slp token). Synchronize never fires for a self-inflict. DRAW-FREE
        // (the slp.onStart random(2,6) above already drew — the eat consumes nothing).
        self.berry_after_set_status(side, slot, dex);

        // (5) FULL HEAL to maxhp (draw-free). Set HP directly (the user just passed the
        //     full-HP guard, so this always heals; clamp for safety).
        let maxhp = self.sides[side].pokemon[slot].maxhp;
        self.sides[side].pokemon[slot].hp = maxhp;

        // [EMIT] `|-heal|<user>|<HP> slp|[silent]` — Rest's full heal, with the `[silent]`
        // tag (Showdown suppresses the client message; the HP field carries the new `slp`
        // status token, e.g. `524/524 slp`). Emitted AFTER the `-status` (the sim sets the
        // status then heals). The `[silent]` cause has no `[from]`; render it via a bare
        // tag helper (push the raw line — the HP-with-status is formatted by `hp_status`).
        if self.logging() {
            let user = self.mon_ref(side, slot, dex);
            let hp = self.hp_status(side, slot);
            self.log.push_raw(format!("|-heal|{user}|{hp}|[silent]"));
        }

        // (7) Status move → not missed, NOT landed (no in-tryMoveHit Update shuffle).
        MoveResolution::done(false, false, false)
    }

    /// PROTECT / DETECT (`data/moves.ts::protect`/`detect`, the gen-3 path) — the
    /// user's own protection move. VERIFIED bit-for-bit vs the omniscient sim's PRNG
    /// probe (`harness/probe_protect_rng.js`). The draw model:
    ///
    ///   1. **NEVER-MISS** — `accuracy: true` → NO accuracy draw.
    ///   2. **THE STALL SUCCESS ROLL** (`onPrepareHit` → `runEvent('StallMove')`): the
    ///      `stall` volatile's `onStallMove` draws `randomChance(1, counter)` — but ONLY
    ///      when the `stall` volatile is ALREADY present (`protect_counter > 0`). On the
    ///      FIRST protect (counter 0, no volatile yet) `runEvent('StallMove')` has NO
    ///      handler → returns true with **NO DRAW** → the protect SHORT-CIRCUITS to
    ///      success. (The `willAct()` gate in `onPrepareHit` is always true here: protect
    ///      is priority 3 so the foe's queued attack is still pending — verified — so it
    ///      never fails for lack of a follow-up action.)
    ///   3. **ON SUCCESS** — the protect goes up (`protected = true`) and the `stall`
    ///      volatile is (re)added (`onHit` → `addVolatile('stall')`): from counter 0 it
    ///      `onStart`s to **2**, otherwise `onRestart`s `*= 2` capped at the gen3
    ///      **counterMax 8** — so consecutive successes give the floored sequence
    ///      `0→2→4→8→8→…` (success 100%/50%/25%/12.5%/12.5%). DRAW-FREE apply.
    ///   4. **ON FAILURE** (the stall roll failed) — gen-3's resolved (gen5-base)
    ///      `onStallMove` does NOT delete the `stall` volatile (UNLIKE the gen8+ base's
    ///      `if (!success) delete pokemon.volatiles['stall']`): the counter + duration
    ///      PERSIST unchanged, so consecutive fails re-roll at the SAME denominator AND
    ///      the stall residual still fires this turn. `protected` stays false (no
    ///      protection this turn); the move "fails" but draws nothing further.
    ///   5. `landed` is ALWAYS FALSE — the protect is a status `moveHit`; the
    ///      in-`tryMoveHit` `eachEvent('Update')` shuffle is SKIPPED.
    ///
    /// FAIL-LOUD: only Protect / Detect are modeled. Any other `isProtect` move (Endure
    /// — `volatileStatus:'endure'` + a survive-at-1-HP `onDamage`; the gen4+ Quick
    /// Guard / Wide Guard / King's Shield / Spiky Shield / … which gen3 doesn't have)
    /// PANICS so a future protection variant can never silently desync.
    fn run_protect(
        &mut self,
        side: usize,
        slot: usize,
        move_id: &str,
        move_name: &str,
        will_act: bool,
        dex: &Dex,
    ) -> MoveResolution {
        if move_id != "protect" && move_id != "detect" {
            panic!(
                "protect-class move {move_id:?} is not modeled — only Protect / Detect \
                 are (identical full-turn protection). Endure (survive-at-1-HP, a \
                 different onDamage mechanic) + Quick Guard / Wide Guard / King's Shield \
                 / etc. are DEFERRED (and gen3 has none of the latter). Model it (or \
                 exclude it from isModeledMove) before a battle can use it."
            );
        }

        // (1) THE `willAct()` GATE (`onPrepareHit`: `!!this.queue.willAct() &&
        //     runEvent('StallMove')`): Protect/Detect FAIL — DRAW-FREE, no volatile, no
        //     stall roll — when NO move/switch action remains in the queue when they
        //     resolve. Because a switch (order 103) sorts before the protect's move (order
        //     200), a foe that SWITCHED has already left the queue empty of actions → the
        //     protect fails. (A foe that uses a MOVE is still pending — protect priority 3
        //     runs first — so `will_act` is true and we proceed.) VERIFIED vs the sim:
        //     Protect into a foe SWITCH leaves the protector with NO volatiles. The
        //     `&&`-short-circuit means `runEvent('StallMove')` does NOT fire → NO stall
        //     draw either; the counter is UNTOUCHED. (In gen3 NEITHER a failed-by-willAct
        //     protect NOR a failed stall ROLL deletes the stall volatile — the resolved
        //     gen5-base has no delete-on-fail; the will_act case additionally draws nothing.)
        if !will_act {
            // [EMIT] a willAct-fail Protect renders the `[still]` empty-target `|move|`
            // (`|move|<user>|Protect||[still]`, verified vs the golden — the same form as a
            // failed stall roll), with NO `-singleturn`.
            if self.logging() {
                let user = self.mon_ref(side, slot, dex);
                self.log.move_used(&user, move_name, None, false, true);
            }
            return MoveResolution::done(false, false, false);
        }

        // (2) THE STALL SUCCESS ROLL — drawn ONLY when the stall volatile is present
        //     (counter > 0). The first protect short-circuits with NO draw.
        let counter = self.sides[side].pokemon[slot].protect_counter;
        let success = if counter == 0 {
            true // first use: runEvent('StallMove') has no handler → true, no draw
        } else {
            self.prng.random_chance(1, counter as u32)
        };

        // [EMIT] the Protect `|move|` line. SUCCESS renders the SELF-target form
        // (`|move|<user>|Protect|<user>`) + `|-singleturn|<user>|Protect`; a FAILED stall
        // roll renders the `[still]` empty-target form (`|move|<user>|Protect||[still]`),
        // NO `-singleturn`. VERIFIED vs the golden. Emitted BEFORE the state mutation.
        if self.logging() {
            let user = self.mon_ref(side, slot, dex);
            if success {
                self.log.move_used(&user, move_name, Some(&user), false, false);
                self.log.singleturn(&user, "Protect");
            } else {
                // A FAILED stall roll shows the `[still]` move + `|-fail|<user>` (the
                // protect "failed" — verified vs the golden: `|move|…Protect||[still]` then
                // `|-fail|p1a: Skarmory`). The willAct-fail branch above emits only the
                // `[still]` move (no `-fail` — the foe switched, "did nothing" not "failed").
                self.log.move_used(&user, move_name, None, false, true);
                self.log.fail(&user, None, false);
            }
        }

        let mon = &mut self.sides[side].pokemon[slot];
        if success {
            // (3) Protect goes up; (re)add the stall volatile (onStart 2 / onRestart *2,
            //     capped at counterMax 8 — the gen3 floor 1/8). BOTH onStart and onRestart
            //     reset the stall volatile's `duration` to 2 (its lifetime — it expires at
            //     the residual one turn after the user stops protecting).
            mon.protected = true;
            mon.protect_counter = if counter == 0 {
                2 // onStart
            } else {
                (counter * 2).min(PROTECT_COUNTER_MAX) // onRestart *= 2, capped
            };
            mon.stall_duration = STALL_DURATION; // duration: 2 (reset on every success)
        } else {
            // (4) Stall roll FAILED: NO protection this turn — but the gen3 (resolved gen5-
            //     base) `stall` `onStallMove` does NOT delete the volatile on failure (unlike
            //     the gen8+ base condition, which `delete pokemon.volatiles['stall']`s). So
            //     the stall volatile PERSISTS with its counter + duration UNCHANGED (`onHit`
            //     did not run, so no `onRestart` re-multiply / duration-refresh). It expires
            //     naturally via the residual `duration` countdown. VERIFIED vs the sim: a
            //     failed 2nd protect leaves a `stall` residual handler (it ties with the
            //     RockSlide flinch → a residual shuffle the deletion model wrongly dropped),
            //     and consecutive fails re-roll at the SAME denominator (`2(F),2(F),2(F)`).
            //     Only `protected` is false (no protect-volatile up); `protect_counter` /
            //     `stall_duration` are LEFT INTACT.
            mon.protected = false;
        }

        // (5) Status move → not missed, NOT landed (no in-tryMoveHit Update shuffle).
        MoveResolution::done(false, false, false)
    }

    /// Whether a foe move targeting `(foe, foe_slot)` is BLOCKED by an active Protect /
    /// Detect. Returns `true` iff the target has its `protected` volatile up AND the
    /// move is NOT self-targeting (`!targets_self`). Protect only blocks moves that
    /// TARGET the protected mon — a self-target move (the attacker's own Protect / setup
    /// / recovery) is never blocked. DRAW-FREE (the caller has already drawn the foe
    /// move's accuracy; the block adds nothing). The protect volatile persists for the
    /// rest of THIS turn (cleared at the next turn-top by `clear_flinch`).
    fn protect_blocks(&self, foe: usize, foe_slot: usize, targets_self: bool) -> bool {
        !targets_self && self.sides[foe].pokemon[foe_slot].protected
    }

    /// The `onBeforeMove` STATUS draws for `side`'s active `slot`, BEFORE accuracy
    /// (mirroring `runEvent('BeforeMove')` at `runMove`, battle-actions.ts:255).
    /// Returns `true` to PROCEED with the move, `false` to ABORT it (drawing nothing
    /// further). Handlers fire in `onBeforeMovePriority`-DESC order and the FIRST
    /// abort SHORT-CIRCUITS the rest (a lower-priority status then never draws) —
    /// `battle.ts:912-920`'s break-on-falsy. At most one MAJOR status (slp/par/frz)
    /// can be present, so the priority-10 slp/frz tie never fires concurrently;
    /// flinch + confusion are volatiles that coexist with a major status.
    ///
    /// Priorities: SLEEP 10 (draw-free counter), FREEZE 10 (`randomChance(1,5)` thaw),
    /// FLINCH 8 (draw-free), CONFUSION 3 (`randomChance(1,2)` + a self-hit
    /// `random(16)`), PARALYSIS 1 (`randomChance(1,4)` full-para).
    fn on_before_move(
        &mut self,
        side: usize,
        slot: usize,
        // The move about to be used (`gen3_taunt_disable_v1`), for the DISABLE + TAUNT
        // `onBeforeMove` cants. `move_index` is the slot; `is_status` is whether the move's
        // category is Status; `struggle` marks the synthetic Struggle (never disabled/taunted —
        // it is a typeless physical move with no slot). A cancelled move (any cant) draws
        // NOTHING further and deducts NO PP (the caller returns before the PP step).
        move_index: usize,
        is_status: bool,
        struggle: bool,
        dex: &Dex,
    ) -> bool {
        // --- SLEEP (priority 10): DRAW-FREE counter decrement (gen3
        //     conditions.ts::slp.onBeforeMove: `if earlybird time--; time--; if
        //     time<=0 wake`). **EARLY BIRD** decrements the counter an EXTRA time (so a
        //     2-turn sleep wakes after 1 attempt). On wake (time<=0) cureStatus +
        //     PROCEED (the mon moves THIS turn); else 'cant' + ABORT. The duration was
        //     drawn (random(2,6)) at status-SET time, NOT here. ---
        if let Some(Status::Sleep(time)) = self.sides[side].pokemon[slot].status {
            let early_bird = to_id(&self.sides[side].pokemon[slot].ability) == "earlybird";
            let dec: u8 = if early_bird { 2 } else { 1 };
            let next = time.saturating_sub(dec);
            if next == 0 {
                // Wake: cure the status, then PROCEED (gen3 wakes and moves same turn).
                self.sides[side].pokemon[slot].status = None;
                // [EMIT] `|-curestatus|<mon>|slp|[msg]` — a natural sleep wake (the `[msg]`
                // shows the client wake message). Emitted BEFORE the mon's own `|move|`
                // (it wakes then moves the same turn). Observation-only.
                if self.logging() {
                    let mon_ref = self.mon_ref(side, slot, dex);
                    self.log.curestatus(&mon_ref, "slp", true);
                }
                // (fall through to the next-lower handlers, like the sim's `return`
                //  from the slp handler — but slp is exclusive, so none else fire.)
            } else {
                self.sides[side].pokemon[slot].status = Some(Status::Sleep(next));
                // [EMIT] `|cant|<mon>|slp` — still asleep, the move is cancelled.
                if self.logging() {
                    let mon_ref = self.mon_ref(side, slot, dex);
                    self.log.cant(&mon_ref, "slp", None);
                }
                return false; // still asleep → ABORT (no further draws).
            }
        }

        // --- FREEZE (priority 10): randomChance(1,5)=20% thaw FIRST (gen4
        //     conditions.ts:89) — the roll DRAWS even for a `flags.defrost` move (the
        //     resolved gen3 `frz.onBeforeMove` puts the roll BEFORE the defrost check).
        //     true → cureStatus + PROCEED (natural thaw, `[msg]`). false → a DEFROST
        //     move (Sacred Fire / Flame Wheel — the ONLY two gen3 `flags.defrost`
        //     carriers; id-gated per the fixed-damage precedent, gen3_moves.json has
        //     no flags field) PROCEEDS anyway (`if (move.flags['defrost']) return;`)
        //     and then thaws the USER draw-free via `frz.onModifyMove`
        //     (`|-curestatus|<mon>|frz|[from] move: <Move>`, emitted BEFORE the
        //     user's `|move|` line); any OTHER move → `|cant|frz` + ABORT.
        //     `gen3_defrost_v1`, probe `harness/probe_sacredfire_defrost.js`: a
        //     frozen Sacred Fire/Flame Wheel user moves at 25/25 seeds, ends
        //     un-frozen at 25/25, and draws EXACTLY ONE more than a healthy user
        //     (the thaw roll) — the port's old always-cant-on-a-failed-roll model
        //     was a draw-COUNT desync (the ~61-repro sacredfire A/B tail). Struggle
        //     is never a defrost move (typeless synthetic, no slot). ---
        if self.sides[side].pokemon[slot].status == Some(Status::Freeze) {
            if self.prng.random_chance(1, 5) {
                self.sides[side].pokemon[slot].status = None; // thawed → PROCEED
                // [EMIT] `|-curestatus|<mon>|frz|[msg]` — a natural thaw.
                if self.logging() {
                    let mon_ref = self.mon_ref(side, slot, dex);
                    self.log.curestatus(&mon_ref, "frz", true);
                }
            } else {
                let defrost_move: Option<String> = if struggle {
                    None
                } else {
                    self.sides[side].pokemon[slot]
                        .set
                        .moves
                        .get(move_index)
                        .filter(|mid| is_defrost_move(&to_id(mid)))
                        .and_then(|mid| dex.moves(mid))
                        .map(|m| m.name.clone())
                };
                if let Some(mv_name) = defrost_move {
                    // DEFROST: the move proceeds while frozen; `frz.onModifyMove`
                    // clears the status draw-free before the `|move|` line.
                    self.sides[side].pokemon[slot].status = None;
                    // [EMIT] `|-curestatus|<mon>|frz|[from] move: <Move>`.
                    if self.logging() {
                        let mon_ref = self.mon_ref(side, slot, dex);
                        self.log.curestatus_from_move(&mon_ref, "frz", &mv_name);
                    }
                } else {
                    // [EMIT] `|cant|<mon>|frz` — stays frozen, the move is cancelled.
                    if self.logging() {
                        let mon_ref = self.mon_ref(side, slot, dex);
                        self.log.cant(&mon_ref, "frz", None);
                    }
                    return false; // stays frozen → ABORT
                }
            }
        }

        // --- TRUANT (`gen3_ability_batch4_v1`, priority 9 — the resolved
        //     `truant.onBeforeMovePriority: 9`, AFTER sleep/freeze (10), BEFORE flinch (8)):
        //     if the holder's `truant_turn` flag is set, `|cant|<mon>|ability: Truant`,
        //     DRAW-FREE, ABORT — no PP, and NO lower handler fires (a paralyzed Slaking's
        //     loaf turn draws NO para roll — probe_truant_rng.js Q2b). An ASLEEP holder's
        //     sleep handler runs FIRST (the cant is `slp`, the counter still decrements);
        //     a holder that WAKES this turn falls through to the truant gate (probe Q2:
        //     wake+move on a truant_turn=false turn, wake→`cant Truant` never observed
        //     because the toggle parity matches — the gate is still checked). The flag is
        //     ARMED on switch-in + TOGGLED by the order-27 residual (see `MonState::
        //     truant_turn`). The ability read is LIVE (`mon.ability` — a Traced Truant
        //     would gate; a Trace holder that copied past Truant reverts on switch-out). ---
        if self.sides[side].pokemon[slot].truant_turn
            && to_id(&self.sides[side].pokemon[slot].ability) == "truant"
        {
            // [EMIT] `|cant|<mon>|ability: Truant` — the loafing turn.
            if self.logging() {
                let mon_ref = self.mon_ref(side, slot, dex);
                self.log.cant(&mon_ref, "ability: Truant", None);
            }
            return false; // loafing → ABORT (no draw, no PP)
        }

        // --- FLINCH (priority 8): DRAW-FREE volatile; if present, ABORT (the move is
        //     cancelled this turn). Set by a flinch secondary when the attacker moved
        //     first; cleared at end-of-turn (duration 1). ---
        if self.sides[side].pokemon[slot].flinch {
            // [EMIT] `|cant|<mon>|flinch` — the flinched mon can't move.
            if self.logging() {
                let mon_ref = self.mon_ref(side, slot, dex);
                self.log.cant(&mon_ref, "flinch", None);
            }
            return false; // flinched → ABORT (no draw)
        }

        // --- DISABLE onBeforeMove (`gen3_taunt_disable_v1`, priority 7 — the gen-3 `disable`
        //     condition's `onBeforeMovePriority: 7`, inherited, so it sorts AFTER sleep/freeze
        //     (10) + flinch (8) but BEFORE confusion (3) + paralysis (1)). If the mon is about
        //     to use the DISABLED slot, the move is CANCELLED here: `|cant|<mon>|Disable|<Move>`,
        //     DRAW-FREE, ABORT — so the disabled move draws NOTHING (no confusion/para roll, no
        //     accuracy/crit/damage) and deducts NO PP (the caller returns before the PP step).
        //     This is reachable when the disabled move was SELECTED the same turn a FASTER
        //     disabler disabled it (the target's choice was locked in before the volatile
        //     landed). Struggle is never disabled (typeless, no slot). VERIFIED vs the sim
        //     (`harness/probe_taunt_disable_onbeforemove_rng.js` scenarios 3+4): the queued
        //     disabled Earthquake is cant'd with ZERO draws for the blocked action and NO PP
        //     deducted — and a paralyzed+disabled mon draws NO para roll either (priority 7
        //     cants BEFORE paralysis 1: the same turn's draw count is IDENTICAL with and
        //     without the paralysis — the OPPOSITE of taunt's priority-0 block below). ---
        if !struggle {
            if let Some((dslot, _)) = self.sides[side].pokemon[slot].disable {
                if dslot == move_index {
                    // [EMIT] `|cant|<mon>|Disable|<MoveName>`.
                    if self.logging() {
                        let mon_ref = self.mon_ref(side, slot, dex);
                        let mv_name = self.sides[side].pokemon[slot]
                            .set
                            .moves
                            .get(move_index)
                            .and_then(|mid| dex.moves(mid))
                            .map(|m| m.name.clone())
                            .unwrap_or_default();
                        self.log.cant(&mon_ref, "Disable", Some(&mv_name));
                    }
                    return false; // disabled → ABORT (no draw, no PP)
                }
            }
        }

        // --- CONFUSION (priority 3): decrement the counter (DRAW-FREE); on 0 remove
        //     + PROCEED; else randomChance(1,2)=50% — true PROCEED, false a typeless
        //     40-BP self-hit (one random(16), NO crit) then ABORT. ---
        if let Some(time) = self.sides[side].pokemon[slot].confusion {
            let next = time.saturating_sub(1);
            if next == 0 {
                self.sides[side].pokemon[slot].confusion = None; // confusion ends → PROCEED
            } else {
                self.sides[side].pokemon[slot].confusion = Some(next);
                if self.prng.random_chance(1, 2) {
                    // Acts normally (no self-hit) → PROCEED.
                } else {
                    // SELF-HIT: getDamage(self,self,40), typeless 40-BP physical,
                    // willCrit=false (no crit roll) → ONE random(16). Routes through
                    // the FULL modify_damage chain (burn/screens/etc.). Then ABORT.
                    self.apply_confusion_self_hit(side, slot, dex);
                    return false;
                }
            }
        }

        // --- ATTRACT (`gen3_ability_batch4_v1`, priority 2 — AFTER confusion (3), BEFORE
        //     paralysis (1)): emit `|-activate|<mon>|move: Attract|[of] <source>` ALWAYS
        //     (even when the mon then moves — probe_cutecharm_attract_rng.js), then draw
        //     `randomChance(1,2)`; on a pass `|cant|<mon>|Attract`, ABORT (no PP, no
        //     para roll). The volatile has NO duration — it outlives turns until the
        //     source leaves the field or the holder switches out. ---
        if let Some((src_side, src_uid)) = self.sides[side].pokemon[slot].attract {
            // [EMIT] the always-on activation line, attributing the source mon.
            if self.logging() {
                let mon_ref = self.mon_ref(side, slot, dex);
                let src_slot = self.sides[src_side]
                    .pokemon
                    .iter()
                    .position(|p| p.uid == src_uid)
                    .expect("attract source uid must exist on its side");
                let src_ref = self.mon_ref(src_side, src_slot, dex);
                self.log.activate(&mon_ref, "move: Attract", Some(&format!("[of] {src_ref}")));
            }
            if self.prng.random_chance(1, 2) {
                // [EMIT] `|cant|<mon>|Attract` — immobilized by love.
                if self.logging() {
                    let mon_ref = self.mon_ref(side, slot, dex);
                    self.log.cant(&mon_ref, "Attract", None);
                }
                return false; // attracted → ABORT (no PP)
            }
        }

        // --- PARALYSIS (priority 1): randomChance(1,4)=25% full-para → ABORT;
        //     else PROCEED. ---
        if self.sides[side].pokemon[slot].status == Some(Status::Paralysis)
            && self.prng.random_chance(1, 4)
        {
            // [EMIT] `|cant|<mon>|par` — the fully-paralyzed mon can't move.
            if self.logging() {
                let mon_ref = self.mon_ref(side, slot, dex);
                self.log.cant(&mon_ref, "par", None);
            }
            return false; // full-para → ABORT
        }

        // --- TAUNT onBeforeMove (`gen3_taunt_disable_v1`, priority 0 — the gen-3 `taunt`
        //     override sets `onBeforeMovePriority: undefined`, so `resolvePriority` defaults it
        //     to 0, sorting it LAST (AFTER paralysis, priority 1)). If the mon is TAUNTED and
        //     about to use a STATUS-category move, the move is CANCELLED here:
        //     `|cant|<mon>|move: Taunt|<Move>`, DRAW-FREE, ABORT — and it deducts NO PP. Because
        //     taunt sorts AFTER paralysis, a taunted+paralyzed mon still DRAWS the para roll
        //     (`randomChance(1,4)`) FIRST; only if it is NOT fully-para'd does the taunt cant it
        //     (VERIFIED vs the sim, `harness/probe_taunt_disable_onbeforemove_rng.js` scenarios
        //     1+2: the blocked Thunder Wave drew ZERO + kept its PP; adding paralysis added
        //     EXACTLY one draw before the same cant line). This is reachable
        //     when a Status move was SELECTED the same turn a FASTER Taunter landed Taunt (the
        //     selection restriction can't retroactively block an already-committed choice).
        //     Struggle is never a Status move (typeless physical). Mirrors the base `taunt`
        //     condition's inherited `onBeforeMove` (`move.category === 'Status'` → `cant`). ---
        if !struggle
            && is_status
            && self.sides[side].pokemon[slot].taunt.is_some()
        {
            // [EMIT] `|cant|<mon>|move: Taunt|<MoveName>`.
            if self.logging() {
                let mon_ref = self.mon_ref(side, slot, dex);
                let mv_name = self.sides[side].pokemon[slot]
                    .set
                    .moves
                    .get(move_index)
                    .and_then(|mid| dex.moves(mid))
                    .map(|m| m.name.clone())
                    .unwrap_or_default();
                self.log.cant(&mon_ref, "move: Taunt", Some(&mv_name));
            }
            return false; // taunted status move → ABORT (no draw, no PP)
        }

        true // PROCEED to the move's accuracy/crit/damage.
    }

    /// The CONFUSION self-hit (`getDamage(pokemon, pokemon, 40)` →
    /// `this.damage(...)`, gen4/conditions.ts:74-83). A typeless 40-BP PHYSICAL
    /// self-hit with `willCrit:false` (so NO crit roll) routed through the FULL
    /// `getDamage` modify chain (one `random(16)` randomizer). Applies the rolled
    /// HP to the confused mon itself.
    ///
    /// **CHOICE BAND applies (the e2e_194 fix).** gen-4 confusion (which gen-3 inherits)
    /// runs `this.actions.getDamage(pokemon, pokemon, 40)` — the FULL `getDamage`, NOT the
    /// simplified `getConfusionDamage` (that's the base/gen7 path). So the attacker's
    /// `onModifyAtk` item — **Choice Band ×1.5 (physical)** — folds into the self-hit exactly
    /// as it would a normal physical move (VERIFIED vs the omniscient sim: a Choice-Band
    /// Aerodactyl's confusion self-hit uses its CB-boosted Atk 463, not the stored 309 — the
    /// rolls jump from ~71 to 90-106). We resolve the SAME `resolve_atk_stat_mods` a real move
    /// would, with `move_type: None` (typeless '???' → NO type-boost item, NO Sea Incense; only
    /// the type-agnostic Choice Band survives). No STAB, neutral effectiveness, willCrit=false;
    /// a SELF-hit so no foe screens/Thick-Fat/immunity apply.
    fn apply_confusion_self_hit(&mut self, side: usize, slot: usize, dex: &Dex) {
        let mon = &self.sides[side].pokemon[slot];
        // Same combatant on both sides (self-hit): the mon's own atk/def + boosts.
        // Typeless 40-BP physical → no STAB, neutral effectiveness, willCrit=false.
        let me = Combatant {
            level: mon.level,
            atk_stat: mon.stats[1],
            spa_stat: mon.stats[3],
            def_stat: mon.stats[2],
            spd_stat: mon.stats[4],
            types: mon_types(mon, dex),
            boosts: [mon.boosts[0], mon.boosts[1], mon.boosts[2], mon.boosts[3], mon.boosts[4]],
            burned: mon.status == Some(Status::Burn),
            has_guts: to_id(&mon.ability) == "guts",
        };
        // The attacker's stat-phase item + ability mods `getDamage` folds: Choice Band ×1.5
        // (physical, type-agnostic) — and, data-driven, any species-gated PHYSICAL stat item
        // (a Thick Club Marowak's confusion self-hit uses the ×2 Atk) PLUS the ability
        // ModifyAtk folds (Huge/Pure Power ×2, Guts ×1.5 when statused — PROBED: a burned
        // Guts mon's confusion self-hit is ×1.5 AND the burn-halve is suppressed, exactly
        // like a real physical hit). A typeless '???' move ⇒ NO type-boost item / Sea Incense
        // / BP-fold item / pinch family (all gate on the move type). The DEFENDER is the SAME
        // mon — its ModifyDef event runs too (a Metal Powder Ditto's self-hit is halved by
        // its own ×2 Def; a statused Marvel Scale mon's self-hit is reduced ×2/3). Resolved
        // from the SAME helpers a real move uses, with the mon's own status/ability on both
        // sides.
        let ability = to_id(&mon.ability);
        let self_statused = mon.status.is_some();
        let atk_stat_mods = resolve_atk_stat_mods(
            &mon.item,
            &ability,
            &mon.species_id,
            None,
            MoveCategory::Physical,
            self_statused,
            dex,
        );
        let def_stat_mods = resolve_def_stat_mods(
            &mon.item,
            &ability,
            &mon.species_id,
            MoveCategory::Physical,
            self_statused,
            dex,
        );
        // Hustle's Atk ×1.5 DIRECT applies to the confusion self-hit too (physical), same
        // `dmg_mod` read as a real physical move (`gen3_accuracy_pipeline_v1`).
        let atk_direct_modify = dex
            .ability(&ability)
            .and_then(|a| a.dmg_mod.as_ref())
            .filter(|m| m.direct && m.fold == DmgFold::Atk)
            .map(|m| (m.num, m.den));
        let ctx = DamageContext {
            attacker: me.clone(),
            defender: me,
            mv: MoveInput {
                base_power: 40,
                move_type: None, // typeless ('???') — no STAB, neutral
                category: MoveCategory::Physical,
                halves_defense: false,
            },
            crit: false, // willCrit:false → NO crit roll
            weather: None,
            reflect: false,
            light_screen: false,
            atk_stat_mods,
            atk_direct_modify,
            def_stat_mods,
            bp_mods: Vec::new(), // type-gated ('???' never matches) — none can fire
            defender_thick_fat: false,
            immune: false,
            // Flash Fire is Fire-type-gated; the typeless '???' self-hit is NOT Fire, so an
            // FF-armed mon that confusion-self-hits gets NO ×1.5 (probe-consistent). false.
            flash_fire: false,
        };
        let dmg = calc_damage(&ctx, dex);
        // ONE random(16) randomizer roll for the self-hit damage.
        let r = self.prng.random_below(16) as usize;
        let realized = dmg.rolls[r];
        // FOCUS BAND: the confusion self-hit is dealt with `effectType: 'Move'` (the
        // gen4-inherited condition) — the roll draws AFTER the randomizer (probe
        // `probe_focusband_confusion_rng.js`) and a lethal self-hit can be survived.
        let realized = self.focus_band_damage(side, slot, realized, true, dex);
        self.apply_damage(side, slot, realized);
    }

    /// Apply a damaging move's SECONDARY effects (battle-actions.ts secondaries(),
    /// 1357-1373): for each surviving secondary, draw ONE `random_below(100)` and
    /// apply the effect if `roll < chance`. The chance is the RAW move chance
    /// (Serene Grace ×2 baked in at onModifyMove; Shield Dust on the DEFENDER FILTERS
    /// the target secondaries OUT — they then draw NO random(100), a draw-COUNT
    /// effect). For our 4 test moves each is ONE secondary (par/frz/flinch). A KO'd
    /// target still draws (setStatus no-ops on hp==0). The status applies via the
    /// onTrySetStatus gates (already-statused / type-immunity — draw-free).
    fn apply_secondaries(
        &mut self,
        side: usize,
        slot: usize,
        foe: usize,
        foe_slot: usize,
        move_index: usize,
        absorbed_by_sub: bool,
        dex: &Dex,
    ) {
        // Resolve the move's secondaries + the attacker's Serene Grace, and the
        // DEFENDER's Shield Dust — clone the few facts so the dex borrow is released.
        let (secondaries, secondary_boosts, move_id): (
            Vec<(String, u16)>,
            Vec<crate::dex::moves::SecondaryBoost>,
            String,
        ) = match self.move_at(side, slot, move_index, dex) {
            Some(m) => (m.secondary_effects.clone(), m.secondary_boosts.clone(), m.id.clone()),
            None => return,
        };
        if secondaries.is_empty() {
            return;
        }
        let serene_grace = to_id(&self.sides[side].pokemon[slot].ability) == "serenegrace";
        let shield_dust = to_id(&self.sides[foe].pokemon[foe_slot].ability) == "shielddust";

        // FAIL-LOUD GUARD: Tri Attack (move id "triattack") is the ONLY gen-3 move whose
        // single secondary block SAMPLES one of brn/par/frz at onHit time — the data
        // FLATTENS it to 3 cols ({par:7,brn:7,frz:6}). Replaying those 3 cols as 3
        // separate random(100)s is WRONG (3 draws + wrong outcome distribution vs the
        // sim's ONE random(100) + ONE sample random(3)). Special-case it; any OTHER move
        // with >1 secondary col is an unmodeled multi-secondary shape → PANIC (never a
        // silent desync).
        if secondaries.len() > 1 {
            if move_id == "triattack" {
                self.apply_triattack_secondary(side, slot, foe, foe_slot, serene_grace, shield_dust, absorbed_by_sub, dex);
                return;
            }
            panic!(
                "move {move_id:?} has {} secondary cols ({secondaries:?}) — unmodeled \
                 multi-secondary shape; a naive per-col replay would mis-draw the LCG. \
                 Add a structured handler (like Tri Attack) before using it.",
                secondaries.len()
            );
        }

        for (effect, chance) in &secondaries {
            // SHIELD DUST: filters out all NON-self secondaries (our test moves'
            // secondaries all target the FOE), so the random(100) is NEVER drawn
            // (the loop body is skipped) — a DRAW-COUNT effect. NOTE: a SELF-boost
            // secondary (self_boost) targets the USER, not the foe, so Shield Dust on
            // the DEFENDER does NOT filter it — but none of our self-boost moves
            // co-occur with a Shield Dust defender in scope, and Shield Dust only
            // scales effects on the bearer, so filtering on the foe-defender's
            // Shield Dust matches the foe-targeting secondaries we model.
            //
            // BUT NOT BEHIND A SUBSTITUTE (`gen3_shielddust_sub_v1` — the A/B fuzzer's #1
            // sub×secondary SEED cluster, probe `probe_sub_break_secondary_rng.js` 2a/2b/2c):
            // Shield Dust's filter is a TARGET-gathered ModifySecondaries handler, and a
            // sub-absorbed hit's target list is `null` — so the filter never gathers and the
            // secondary `random(100)` STILL DRAWS (held AND breaking sub; the effect is then
            // suppressed by the null target like any sub-absorbed secondary). The bare-target
            // filter (2a: no draw) is unchanged.
            if shield_dust && !absorbed_by_sub && effect.as_str() != "self_boost" {
                continue;
            }
            // SERENE GRACE ×2 the chance (the threshold, NOT the draw — onModifyMove
            // pre-doubles it before the hit). Cap at 100 (a 60% → 120% never matters
            // for our moves but mirrors the in-place doubling that can exceed 100).
            let eff_chance = if serene_grace { (*chance as u32) * 2 } else { *chance as u32 };

            // The secondary random(100) — UNCONDITIONAL per surviving secondary,
            // drawn BEFORE the chance check (battle-actions.ts:1364). DRAWN EVEN when a
            // SUBSTITUTE absorbed the hit (the gen-3 quirk: `secondaries()` iterates the
            // target list whose entry is now `null`, not `false`, so the `random(100)`
            // still fires — VERIFIED vs the sim probe: a Body Slam / Crunch / Water Pulse
            // into a sub draws the same acc+crit+dmg+random(100) as a bare hit).
            let roll = self.prng.random_below(100);
            // EFFECT SUPPRESSION when the sub absorbed it: the secondary's effect runs via
            // `moveHit(target=null)`, so the FOE-TARGETING effects do NOTHING — no status, no
            // stat-drop, no flinch, AND no confusion `random(2,6)` follow-on draw (VERIFIED:
            // Water Pulse into a sub draws the random(100) but NOT the random(2,6)). BUT a
            // **SELF-boost** secondary (Meteor Mash +1 Atk to the USER) STILL APPLIES through a
            // sub — its `secondary.self.boosts` targets the SOURCE, not the null target, so the
            // sub doesn't block it (VERIFIED: a 20%-proc Meteor Mash into a Skarmory sub gives
            // the attacker +1 Atk). So suppress only when the sub absorbed it AND the effect is
            // FOE-targeting (`self_boost` is exempt). The `random(100)` was drawn above either way.
            let suppressed = absorbed_by_sub && effect.as_str() != "self_boost";
            if roll < eff_chance && !suppressed {
                self.apply_one_secondary(side, slot, foe, foe_slot, effect, &secondary_boosts, dex);
            }
        }
    }

    /// Tri Attack's `secondary {chance:20, onHit{ sample(['brn','par','frz']) }}`: ONE
    /// `random(100)` (the 20% gate) then, ON LAND, ONE `random(3)` (`this.sample`) that
    /// picks brn/par/frz → `trySetStatus`. NOT three `random(100)`s. Mirrors the verified
    /// sim draw sequence (battle-actions.ts:1364 + prng.ts sample). Shield Dust filters it
    /// out (no draw); Serene Grace ×2 the threshold.
    fn apply_triattack_secondary(
        &mut self,
        side: usize,
        slot: usize,
        foe: usize,
        foe_slot: usize,
        serene_grace: bool,
        shield_dust: bool,
        absorbed_by_sub: bool,
        dex: &Dex,
    ) {
        if shield_dust && !absorbed_by_sub {
            // a DRAW-COUNT effect: the secondary is filtered → NO random(100). BUT NOT
            // BEHIND A SUB (`gen3_shielddust_sub_v1`, probe 3a/3b): the target-gathered
            // filter never fires on a null (sub-absorbed) target, so the random(100)
            // draws (and the `random(3)` sample stays sub-suppressed below).
            return;
        }
        let eff_chance: u32 = if serene_grace { 40 } else { 20 };
        let roll = self.prng.random_below(100);
        // SUBSTITUTE ABSORB: the `random(100)` STILL draws (the gen-3 quirk), but a sub-
        // absorbed hit runs the secondary's `onHit` (the `sample`) on a `null` target, so
        // the `random(3)` does NOT draw and NO status applies (VERIFIED: Tri Attack into a
        // sub draws the random(100) but NOT the random(3)).
        if roll < eff_chance && !absorbed_by_sub {
            // sample(['brn','par','frz']) = ONE random(3) selecting the status.
            const SAMPLE: [&str; 3] = ["brn", "par", "frz"];
            let idx = self.prng.random_below(3) as usize;
            // Source = the attacker (for the Synchronize reflect of the sampled status).
            self.try_set_status(foe, foe_slot, SAMPLE[idx], Some((side, slot)), dex);
        }
    }

    /// Apply ONE landed secondary effect (DRAW-FREE for status/flinch/boost; the
    /// CONFUSION secondary alone draws ONE `random(2,6)` duration inside `addVolatile`'s
    /// onStart). `effect` is the flattened secondary col; `side`/`slot` the USER (for a
    /// self-boost / confusion-source gating), `foe`/`foe_slot` the target. `boosts` is
    /// the move's structured `secondary_boosts` (looked up by `foe_statdrop`/`self_boost`).
    ///   - status (par/brn/frz/slp/psn/tox) → `try_set_status` (the onTrySetStatus gates);
    ///   - flinch → the volatile (blocks the target's move this turn if it hasn't moved);
    ///   - confusion → `add_confusion` (already-confused / Own Tempo gates draw NOTHING;
    ///     a successful add draws `random(2,6)`);
    ///   - foe_statdrop / self_boost → `apply_secondary_boost` (the structured spec).
    /// All no-op on a KO'd (hp==0) target.
    fn apply_one_secondary(
        &mut self,
        side: usize,
        slot: usize,
        foe: usize,
        foe_slot: usize,
        effect: &str,
        boosts: &[crate::dex::moves::SecondaryBoost],
        dex: &Dex,
    ) {
        match effect {
            "par" | "brn" | "frz" | "slp" | "psn" | "tox" => {
                // Source = the attacker (`side`/`slot`) for the Synchronize reflect of a
                // secondary-inflicted status (Body Slam's par into a Synchronize holder reflects).
                self.try_set_status(foe, foe_slot, effect, Some((side, slot)), dex);
            }
            "flinch" => {
                // The flinch volatile blocks the target's move THIS turn if it hasn't
                // moved yet. A fainted target can't flinch (no-op), and a flinch on a
                // mon that already moved is harmless (cleared end-of-turn).
                //
                // INNER FOCUS (`gen3_ability_batch4_v1`): blocks at the volatile APPLY
                // (`innerfocus.onTryAddVolatile` → null for flinch) — the secondary's
                // random(100) was ALREADY drawn by the caller, so the draw count is
                // IDENTICAL to a landed flinch (probe_innerfocus_rng.js: draws match a
                // Thick Fat control bit-for-bit, cant=0). CONTRAST Shield Dust, which
                // FILTERS the secondary so the roll is never drawn. Applies to a move's
                // own flinch secondary AND the King's Rock appended one alike.
                let mon = &mut self.sides[foe].pokemon[foe_slot];
                if !mon.fainted && mon.hp > 0 && to_id(&mon.ability) != "innerfocus" {
                    mon.flinch = true;
                }
            }
            // CONFUSION secondary: the addVolatile('confusion') draws random(2,6) at
            // onStart UNLESS gated (already-confused / Own Tempo). The secondary
            // random(100) itself was already drawn by the caller (a landed-but-gated
            // confusion STILL drew the random(100) but draws NO random(2,6)).
            "confusion" => {
                self.add_confusion(foe, foe_slot, dex);
            }
            // The STRUCTURED stat-boost secondaries (foe stat-drop / self stat-raise) —
            // DRAW-FREE; apply the (stat, stages) from the move's secondary_boosts spec
            // to the foe or the user.
            "foe_statdrop" => {
                self.apply_secondary_boost(side, slot, foe, foe_slot, false, boosts, dex);
            }
            "self_boost" => {
                self.apply_secondary_boost(side, slot, foe, foe_slot, true, boosts, dex);
            }
            other => panic!(
                "unhandled secondary effect col {other:?} — add a handler before using \
                 a move that carries it (a silent no-op risks a STATE divergence)."
            ),
        }
    }

    /// `target.addVolatile('confusion')` (pokemon.ts:1969-2027): the CONFUSION
    /// secondary's apply. The gates are checked BEFORE the onStart duration draw and
    /// each makes the add a draw-free no-op:
    ///   - ALREADY CONFUSED (`this.volatiles['confusion']` set, confusion has no
    ///     onRestart) → return false, NO random(2,6);
    ///   - OWN TEMPO (`owntempo.onTryAddVolatile` returns null for confusion) → return
    ///     before onStart, NO random(2,6);
    ///   - a KO'd / 0-HP target → no add.
    /// SUBSTITUTE is a moveHit-level gate (the sub absorbs the hit so addVolatile is
    /// never reached) — but Substitute is not modeled in this engine, and the caller
    /// only reaches here on a landed damaging hit, so it is N/A here.
    /// On a SUCCESSFUL add it draws ONE `random(2,6)` (the onStart duration, min=2 for
    /// a move source → 2..5 turns) into the `confusion: Option<u8>` counter.
    fn add_confusion(&mut self, foe: usize, foe_slot: usize, _dex: &Dex) {
        let mon = &self.sides[foe].pokemon[foe_slot];
        // KO'd target: no add (and a confused-then-fainted mon can't be re-confused).
        if mon.fainted || mon.hp == 0 {
            return;
        }
        // ALREADY CONFUSED: addVolatile returns false before onStart → NO draw.
        if mon.confusion.is_some() {
            return;
        }
        // OWN TEMPO: the onTryAddVolatile immunity returns before onStart → NO draw.
        if to_id(&mon.ability) == "owntempo" {
            return;
        }
        // SUCCESSFUL add → the onStart duration draw random(2,6) (2..5 turns).
        let dur = self.prng.random_range(2, 6) as u8;
        self.sides[foe].pokemon[foe_slot].confusion = Some(dur);
    }

    /// `target.trySetStatus(status)` → `setStatus` (pokemon.ts:1670-1750) — apply a
    /// MAJOR status with the gen-3 onTrySetStatus gates, ALL DRAW-FREE:
    ///   (1) already-statused: if the mon already has a major status → FAIL (no-op);
    ///   (2) type immunity (runStatusImmunity → getImmunity, damageTaken[status]==3):
    ///       par → Electric immune; frz → Ice immune; brn → Fire immune;
    ///       psn/tox → Poison & Steel immune (slp has NO type immunity);
    ///   (3) hp==0 (a KO'd target): setStatus returns false → no-op.
    /// On success it sets the status. **SLEEP** draws ONE `random(2,6)` at onStart
    /// (the 1-4-turn gen-3 sleep counter — `data/mods/gen3/conditions.ts::slp.onStart`,
    /// `this.random(2,6)`), stored directly in `Status::Sleep(n)` (decremented per move
    /// in `on_before_move`). **TOXIC** begins at stage 0 with NO onStart draw (the
    /// residual ramps it to 1 before the first chip — mirrors `statusState.stage`). The
    /// **SLEEP CLAUSE MOD** (gen3ou) + the **status-immunity ABILITIES** fire at the
    /// `runEvent('SetStatus')` event — both DRAW-FREE and BEFORE the onStart, so a
    /// blocked sleep move draws accuracy (already done by the caller) but NO
    /// `random(2,6)`.
    ///
    /// Classify whether a foe-targeting major-status MOVE would FAIL because the foe is
    /// ALREADY statused — and, if so, whether the fail is the SAME-status or
    /// DIFFERENT-status form (they emit different `|-fail|` lines). Returns `None` when the
    /// move is not a foe-status move, the foe is un-statused, or the move is type-immune to
    /// the foe (the `-immune` path wins first). Only used for the PROTOCOL `-fail` line +
    /// the `[still]` move-announce form — it does NOT gate the DRAW model (accuracy is
    /// always drawn; the already-statused fail is draw-free past accuracy). This mirrors
    /// gen-3 `trySetStatus` → `setStatus(this.status || status)`: re-passing the foe's OWN
    /// status makes `status.id === this.status`, so `setStatus` (pokemon.ts:1699) branches
    /// on whether the MOVE's inflicted status equals the foe's current status.
    ///
    /// A SECONDARY status (Body Slam's par) has no top-level `move.status`, so its already-
    /// statused no-op emits NOTHING — this classifier is called only from the standalone
    /// foe-status-move arm, so that distinction holds by construction.
    fn foe_status_move_fail(
        &self,
        foe: usize,
        foe_slot: usize,
        move_id: &str,
        targets_self: bool,
        status_inflicted: Option<&str>,
        dex: &Dex,
    ) -> Option<StatusMoveFail> {
        // Only a foe-targeting MODELED major-status move (the `move.status` moves) can
        // produce this fail; a self-target / non-status move never does.
        if targets_self {
            return None;
        }
        let inflicts = modeled_status_move(move_id)?;
        // A move whose dex status disagrees with the modeled mapping is a data drift — let
        // the debug_assert in the standalone arm catch it; here just proceed on the modeled
        // value so the classification is consistent.
        let _ = status_inflicted;
        let cur = self.sides[foe].pokemon[foe_slot].status;
        let cur_tok = status_token(cur)?; // None → un-statused → not this case
        // TYPE IMMUNITY WINS FIRST — a Ground target vs Thunder Wave reports `-immune`, not
        // a status fail (the `-immune` branch precedes `setStatus`'s already-statused
        // check). Mirror the standalone arm's `natural_immunity` gate.
        if status_move_checks_type_immunity(move_id) {
            if let Some(t) = self.move_type_of(move_id, dex) {
                let def_types = mon_types(&self.sides[foe].pokemon[foe_slot], dex);
                if dex.type_chart().effectiveness(t, &def_types) == 0.0 {
                    return None; // `-immune` path, not an already-statused fail
                }
            }
        }
        // SAME vs DIFFERENT: does the move's inflicted status match the foe's current one?
        if cur_tok == inflicts {
            Some(StatusMoveFail::Same)
        } else {
            Some(StatusMoveFail::Different)
        }
    }

    /// The resolved gen-3 move type for a move id (a thin dex lookup — used by the
    /// status-fail classifier to reproduce the type-immunity gate outside `run_move`).
    fn move_type_of(&self, move_id: &str, dex: &Dex) -> Option<Type> {
        dex.moves(move_id).and_then(|m| m.move_type)
    }

    /// Whether `move_id` is a SOUND move (`flags.sound`, `gen3_ability_batch2_v1`) — Soundproof
    /// is immune to it. A thin dex lookup (Sing / Grass Whistle / Roar / Perish Song are sound).
    fn move_is_sound(&self, move_id: &str, dex: &Dex) -> bool {
        dex.moves(move_id).map(|m| m.is_sound).unwrap_or(false)
    }

    /// The `(side, slot)` of an ACTIVE mon with the Damp ability (`gen3_ability_batch2_v1`), if
    /// any — the holder whose `onAnyTryMove` cancels Explosion/Self-Destruct. gen-3 singles: at
    /// most 2 actives; `onAny*` fires for BOTH sides, so an Explosion is blocked if EITHER active
    /// has Damp (incl. the user's OWN side). Returns the FIRST found (either side blocks
    /// identically — the `|cant|` names the holder). `None` ⇒ no active Damp mon (Explosion runs).
    fn damp_holder(&self, dex: &Dex) -> Option<(usize, usize)> {
        for s in 0..2 {
            let slot = self.sides[s].active;
            let mon = &self.sides[s].pokemon[slot];
            if !mon.fainted
                && dex.ability(&to_id(&mon.ability)).map(|a| a.blocks_explosion).unwrap_or(false)
            {
                return Some((s, slot));
            }
        }
        None
    }

    /// CONTACT_PROC + CONTACT recoil (`gen3_ability_batch2_v1`) — the DEFENDER's reactive
    /// `onDamagingHit` ability firing against the ATTACKER after a CONTACT hit that dealt damage.
    /// Called from `run_move`'s landed-hit tail (the `runEvent('DamagingHit')` position, AFTER
    /// `apply_secondaries`). `atk_side`/`atk_slot` = the ATTACKER (who gets statused / takes
    /// recoil); `def_side`/`def_slot` = the ability HOLDER (the DEFENDER).
    ///
    /// The draw model (PROBE-settled `harness/probe_contact_proc_{rng,lands}.js` +
    /// `probe_effectspore_sample.js`):
    ///   - **Status procs** (Static par / Poison Point psn / Flame Body brn — `sample=false`):
    ///     ONE `randomChance(num, den)` (one `random(den)`); on a PASS, `try_set_status(the-one-
    ///     status, attacker)` with the ATTACKER as target and the DEFENDER as the reflect source
    ///     (so gen-3 type/ability/already-statused gates apply, and gen3ou draws the SetStatus
    ///     shuffle — draw-free in the e2e customgame). The `randomChance` draws UNCONDITIONALLY
    ///     (even if the attacker is already-statused / type-immune — the gate is inside
    ///     trySetStatus, AFTER the roll).
    ///   - **Effect Spore** (`sample=true`): `randomChance(1,10)` gate; on a PASS one
    ///     `sample(["slp","par","psn"])` (a `random(3)`) picks the status → `try_set_status`.
    ///   - **Rough Skin** (`contact_recoil`): DRAW-FREE — `baseMaxhp/16` recoil to the attacker
    ///     (`this.damage`, no PRNG). A recoil KO faints the attacker via the normal machinery.
    /// Nothing fires for a non-CONTACT_PROC / non-Rough-Skin holder (draw-free no-op).
    fn apply_contact_proc(&mut self, atk_side: usize, atk_slot: usize, def_side: usize, def_slot: usize, dex: &Dex) {
        let ability = to_id(&self.sides[def_side].pokemon[def_slot].ability);
        let (contact_proc, contact_recoil, contact_attract) = match dex.ability(&ability) {
            Some(a) => (a.contact_proc.clone(), a.contact_recoil, a.contact_attract),
            None => (None, false, None),
        };
        if let Some(cp) = contact_proc {
            // The `randomChance(num, den)` gate — DRAWN unconditionally on a contact-damaging hit.
            let passed = self.prng.random_chance(cp.chance.0, cp.chance.1);
            if passed {
                // Which status: a single (Static/PoisonPoint/FlameBody) applies directly; a
                // `sample` (Effect Spore) draws ONE `random(len)` to pick.
                let status: &str = if cp.sample {
                    let idx = self.prng.random_below(cp.statuses.len() as u32) as usize;
                    &cp.statuses[idx]
                } else {
                    &cp.statuses[0]
                };
                // Apply to the ATTACKER; the DEFENDER (the ability holder) is the Synchronize
                // source (so a Synchronize attacker would reflect it back — an ability-on-ability
                // interaction the choke point handles). The `[from] ability: <Name>|[of]` reveal:
                // Showdown emits it via the contact-proc effect context — but the plain
                // `try_set_status` emit would show a bare `|-status|`. Suppress it + emit the
                // ability form (observation-only; the seed suites are unaffected).
                let status_owned = status.to_string();
                let was_logging = self.log.is_enabled();
                if was_logging {
                    self.log.set_enabled(false);
                }
                self.try_set_status(atk_side, atk_slot, &status_owned, Some((def_side, def_slot)), dex);
                if was_logging {
                    self.log.set_enabled(true);
                }
                // [EMIT] `|-status|<attacker>|<status>|[from] ability: <Name>|[of] <holder>` —
                // only if the status actually landed (the attacker was statusable). Reads the
                // ability's display NAME from the dex.
                if self.logging()
                    && self.sides[atk_side].pokemon[atk_slot].status.is_some()
                {
                    let name = dex.ability(&ability).map(|a| a.name.clone()).unwrap_or_else(|| ability.clone());
                    let atk_ref = self.mon_ref(atk_side, atk_slot, dex);
                    let holder_ref = self.mon_ref(def_side, def_slot, dex);
                    self.log.status_from_ability(&atk_ref, &status_owned, &name, &holder_ref);
                }
            }
        } else if let Some((num, den)) = contact_attract {
            // CUTE CHARM (`gen3_ability_batch4_v1`): the SAME DamagingHit-position roll as the
            // status procs — `randomChance(1,3)` drawn UNCONDITIONALLY on a damaging contact
            // hit (the gender gate lives INSIDE attract.onStart: a same-gender / genderless
            // attacker still DRAWS the roll and the volatile fails draw-free — probe
            // `probe_cutecharm_attract_rng.js`). On a pass, add ATTRACT to the ATTACKER.
            let passed = self.prng.random_chance(num, den);
            if passed {
                self.try_add_attract(atk_side, atk_slot, def_side, def_slot, dex);
            }
        } else if contact_recoil {
            // ROUGH SKIN: DRAW-FREE recoil `baseMaxhp/16` to the ATTACKER (the sim's
            // `this.damage(source.baseMaxhp / 16, source, target)`). gen-3 `maxhp == baseMaxhp`.
            let recoil = (self.sides[atk_side].pokemon[atk_slot].maxhp / 16).max(1);
            let recoil = recoil.min(self.sides[atk_side].pokemon[atk_slot].hp);
            let recoil = self.focus_band_damage(atk_side, atk_slot, recoil, false, dex);
            if recoil > 0 {
                self.apply_damage(atk_side, atk_slot, recoil);
                // [EMIT] `|-damage|<attacker>|<HP>|[from] ability: Rough Skin|[of] <holder>`.
                if self.logging() {
                    let name = dex.ability(&ability).map(|a| a.name.clone()).unwrap_or_else(|| ability.clone());
                    let atk_ref = self.mon_ref(atk_side, atk_slot, dex);
                    let holder_ref = self.mon_ref(def_side, def_slot, dex);
                    let hp = self.hp_status(atk_side, atk_slot);
                    self.log.damage_of(&atk_ref, &hp, &Cause::Ability(name), &holder_ref);
                }
            }
        }
    }

    /// FOCUS BAND (`gen3_ability_batch4_v1`, `ItemData::survive_lethal`) — the `onDamage`
    /// (priority -40) handler: `randomChance(1,10) && damage >= target.hp && effect.effectType
    /// === 'Move'`. The JS `&&` order means the roll DRAWS FIRST — on EVERY Damage event into
    /// the holder (move hits, burn/psn/tox chips, sand/hail chips, the Leech Seed drain,
    /// Spikes, Struggle/Rough-Skin recoil, confusion self-hits) — while the survive-at-1-HP
    /// fires only when the roll passed AND the damage is lethal AND the effect is a MOVE
    /// (`is_move`: the damaging/fixed-damage hit + the confusion self-hit, whose gen4-inherited
    /// effect is `effectType: 'Move'`; a lethal chip/recoil still faints). NOT called for a
    /// sub-absorbed hit (the mon's Damage event never runs — no draw). Returns the (possibly
    /// hp-1-capped) damage; emits `|-activate|<mon>|item: Focus Band` on a survive.
    /// PROBE-settled `probe_focusband_rng.js` + `probe_focusband_confusion_rng.js`.
    fn focus_band_damage(&mut self, side: usize, slot: usize, dmg: u16, is_move: bool, dex: &Dex) -> u16 {
        if dmg == 0 {
            return 0; // no Damage event for a 0-damage apply (no modeled path reaches this)
        }
        {
            let mon = &self.sides[side].pokemon[slot];
            if mon.fainted || mon.hp == 0 {
                return dmg; // a fainted/0-HP holder runs no Damage event
            }
            match dex.item(&to_id(&mon.item)).and_then(|i| i.survive_lethal) {
                Some(_) => {}
                None => return dmg,
            }
        }
        let (num, den) = dex
            .item(&to_id(&self.sides[side].pokemon[slot].item))
            .and_then(|i| i.survive_lethal)
            .expect("checked above");
        let passed = self.prng.random_chance(num, den);
        let hp = self.sides[side].pokemon[slot].hp;
        if passed && is_move && dmg >= hp {
            // [EMIT] `|-activate|<mon>|item: Focus Band` — the survive (BEFORE the -damage).
            if self.logging() {
                let mon_ref = self.mon_ref(side, slot, dex);
                self.log.activate(&mon_ref, "item: Focus Band", None);
            }
            return hp - 1; // hang on at 1 HP
        }
        dmg
    }

    /// KING'S ROCK (`gen3_ability_batch4_v1`, `ItemData::flinch_secondary`) — the appended
    /// `{chance: 10, flinch}` trailing secondary for the holder's LISTED moves (`onModifyMove`
    /// pushes it onto `move.secondaries`, so it is an ORDINARY secondary): rolled AFTER the
    /// move's own secondary (list order — probe_kingsrock_order_rng.js O1/O3), BEFORE the
    /// foe's contact proc (O2). Serene Grace DOUBLES the chance (threshold 20 — the SG probe's
    /// 10/15-land vs 31+-miss split); Shield Dust FILTERS it (NO draw); behind a SUBSTITUTE it
    /// DRAWS but does not apply (the standard gen-3 secondary-vs-sub rule); Inner Focus blocks
    /// at the flinch APPLY (the roll still draws). The 17 typed Hidden Powers canonicalize to
    /// the bare sim id `hiddenpower` for the list lookup. A fixed-damage listed move (Seismic
    /// Toss) and Struggle both proc. Probe `probe_kingsrock_rng.js`.
    fn apply_kings_rock_secondary(
        &mut self,
        side: usize,
        slot: usize,
        foe: usize,
        foe_slot: usize,
        move_id: &str,
        absorbed_by_sub: bool,
        dex: &Dex,
    ) {
        let item = to_id(&self.sides[side].pokemon[slot].item);
        let (chance, listed) = match dex.item(&item).and_then(|i| i.flinch_secondary.as_ref()) {
            Some(fs) => {
                // Typed HP ids (hiddenpowerice, ...) collapse to the sim's one id.
                let canon = if move_id.starts_with("hiddenpower") { "hiddenpower" } else { move_id };
                (fs.chance as u32, fs.moves.iter().any(|m| m == canon))
            }
            None => return,
        };
        if !listed {
            return; // a non-listed move gains no secondary
        }
        // SHIELD DUST filters the appended secondary like any other (NO draw) — but NOT
        // behind a SUB (`gen3_shielddust_sub_v1`, probe 4a/4b: the target-gathered filter
        // never fires on a sub-absorbed hit, so the KR random(100) draws; the flinch stays
        // sub-suppressed below).
        if to_id(&self.sides[foe].pokemon[foe_slot].ability) == "shielddust" && !absorbed_by_sub {
            return;
        }
        // SERENE GRACE x2 the threshold (10 -> 20), like the move's own secondaries.
        let serene = to_id(&self.sides[side].pokemon[slot].ability) == "serenegrace";
        let eff_chance = if serene { chance * 2 } else { chance };
        let roll = self.prng.random_below(100);
        // Behind a sub: the roll drew, the flinch does NOT apply (foe-targeting suppression).
        if roll < eff_chance && !absorbed_by_sub {
            self.apply_one_secondary(side, slot, foe, foe_slot, "flinch", &[], dex);
        }
    }

    /// The ATTRACT volatile's add (`attract.onStart`, `gen3_ability_batch4_v1`) — from a Cute
    /// Charm contact proc. The gates, ALL draw-free (the CC `randomChance(1,3)` was already
    /// drawn by the caller): already-attracted (addVolatile returns false, no re-add), a
    /// fainted/0-HP target, the GENDER gate (M<->F opposite pairs only; a same-gender or
    /// genderless pair fails — probe: F-into-F / genderless draw the roll but never attract),
    /// and OBLIVIOUS (the runEvent('Attract') block — probed draw-free). A target with an
    /// UNSPECIFIED gender (the sim would have SAMPLED one at construction — an init draw the
    /// port does not model) PANICS fail-loud. On success the volatile records the SOURCE
    /// (side, uid) — cleared when the source leaves the field or the holder switches out.
    fn try_add_attract(&mut self, side: usize, slot: usize, src_side: usize, src_slot: usize, dex: &Dex) {
        {
            let mon = &self.sides[side].pokemon[slot];
            if mon.attract.is_some() || mon.fainted || mon.hp == 0 {
                return;
            }
        }
        let need = |g: Option<char>, who: &str| -> char {
            g.unwrap_or_else(|| {
                panic!(
                    "attract gender-compare reached a mon with an UNSPECIFIED gender ({who}) — \
                     the sim samples one at construction (an unmodeled init draw); pin genders \
                     explicitly in any team that can reach Cute Charm."
                )
            })
        };
        let gt = need(self.sides[side].pokemon[slot].gender, "target");
        let gs = need(self.sides[src_side].pokemon[src_slot].gender, "source");
        let opposite = (gt == 'M' && gs == 'F') || (gt == 'F' && gs == 'M');
        if !opposite {
            return; // incompatible gender -> draw-free fail
        }
        if to_id(&self.sides[side].pokemon[slot].ability) == "oblivious" {
            return; // the runEvent('Attract') block -> draw-free fail
        }
        let src_uid = self.sides[src_side].pokemon[src_slot].uid;
        self.sides[side].pokemon[slot].attract = Some((src_side, src_uid));
        // [EMIT] `|-start|<mon>|Attract|[from] ability: Cute Charm|[of] <source>`.
        if self.logging() {
            let mon_ref = self.mon_ref(side, slot, dex);
            let src_ref = self.mon_ref(src_side, src_slot, dex);
            self.log
                .push_raw(format!("|-start|{mon_ref}|Attract|[from] ability: Cute Charm|[of] {src_ref}"));
        }
    }

    /// COLOR CHANGE (`gen3_ability_batch4_v1`) — the DEFENDER's `onDamagingHit` type override:
    /// on a damaging hit that reached the MON (NOT a sub-absorbed hit — the mon's DamagingHit
    /// never fires behind a sub, same as the contact procs; probe t2: a TBolt into the sub
    /// leaves Kecleon Normal), while the holder is ALIVE (no change on the KO hit), for a
    /// non-Status move with a real type (typeless `???` — Struggle, the confusion self-hit —
    /// never changes) that the holder doesn't already have: `types_override = [move.type]`.
    /// DRAW-FREE. The override feeds every later `mon_types` read (STAB / chart / status
    /// type-immunity / sand-chip immunity). Cleared on switch-out. Probe `probe_colorchange_rng.js`.
    fn apply_color_change(&mut self, side: usize, slot: usize, move_type: Option<Type>, dex: &Dex) {
        {
            let mon = &self.sides[side].pokemon[slot];
            if to_id(&mon.ability) != "colorchange" || mon.fainted || mon.hp == 0 {
                return;
            }
            let t = match move_type {
                Some(t) => t,
                None => return, // typeless ??? never changes
            };
            if mon_types(mon, dex).contains(&t) {
                return; // already that type -> no-op (no line)
            }
        }
        let t = move_type.expect("checked above");
        self.sides[side].pokemon[slot].types_override = Some(vec![t]);
        // [EMIT] `|-start|<mon>|typechange|<Type>|[from] ability: Color Change`.
        if self.logging() {
            let mon_ref = self.mon_ref(side, slot, dex);
            self.log
                .push_raw(format!("|-start|{mon_ref}|typechange|{}|[from] ability: Color Change", t.name()));
        }
    }

    /// Try to apply a major status `effect` to `foe`/`foe_slot`, mirroring gen-3
    /// `pokemon.trySetStatus`/`setStatus` (all the onTrySetStatus gates + the gen3ou SetStatus
    /// clause shuffle + the STATUS_IMMUNE abilities). `source` is `Some((side, slot))` when a
    /// FOE effect inflicted the status (a status move / a damaging move's secondary / a contact
    /// proc), `None` for a source-less / self-inflicted apply (or when the caller does not want
    /// the Synchronize reflect). When the status successfully applies to a **Synchronize** holder
    /// and `source` is a DIFFERENT mon, gen-3 `synchronize.onAfterSetStatus` REFLECTS the status
    /// back to the source (slp/frz EXEMPT; tox→psn) — draw-free in gen3customgame (the e2e
    /// format), and drawing the reflected status's own SetStatus shuffle in gen3ou. `gen3_ability_batch2_v1`.
    fn try_set_status(&mut self, foe: usize, foe_slot: usize, effect: &str, source: Option<(usize, usize)>, dex: &Dex) {
        self.try_set_status_impl(foe, foe_slot, effect, source, false, None, dex);
    }

    /// [`Self::try_set_status`] with the two Phase-3 emission nuances:
    /// - `announce_immune_block`: a `setStatus`-phase STATUS_IMMUNE ability's `onSetStatus`
    ///   handler emits its `|-immune|<target>|[from] ability: <A>` line ONLY when the source
    ///   effect is a status MOVE (`(effect as Move)?.status` — Thunder Wave / Will-O-Wisp /
    ///   Toxic / Hypnosis; byte-verified vs the status_immune_lines capture). A SECONDARY /
    ///   contact-proc / Tri-Attack block is SILENT (same gate, no line). Only the
    ///   standalone-status-move arm passes `true`.
    /// - `sync_reveal`: `Some((holder_side, holder_slot))` when this apply IS a Synchronize
    ///   reflect — the `-status` line then carries the `[from] ability: Synchronize|[of]
    ///   <holder>` reveal INSTEAD of the plain form, at the same position, so the holder's
    ///   Lum `-enditem`/`-curestatus` tail (fired from THIS apply) lands AFTER it
    ///   (byte-verified interleave vs the synchronize_lum_rest capture).
    fn try_set_status_impl(
        &mut self,
        foe: usize,
        foe_slot: usize,
        effect: &str,
        source: Option<(usize, usize)>,
        announce_immune_block: bool,
        sync_reveal: Option<(usize, usize)>,
        dex: &Dex,
    ) {
        let mon = &self.sides[foe].pokemon[foe_slot];
        // (3) A KO'd / 0-HP target cannot be statused (setStatus: `if (!this.hp)`).
        if mon.fainted || mon.hp == 0 {
            return;
        }
        // (1) Already-statused → fail (a mon with ANY major status can't take another).
        if mon.status.is_some() {
            return;
        }
        // (2) Status-type immunity — the gen-3 rules the repo type chart lacks
        //     (multipliers-only): brn→Fire, frz→Ice, psn/tox→Poison&Steel; par/slp none.
        //     `setStatus` returns at `runStatusImmunity` BEFORE `runEvent('SetStatus')`,
        //     so a type-immune status does NOT draw the SetStatus handler-sort shuffle.
        let types = mon_types(mon, dex);
        if status_type_immune(effect, &types) {
            return;
        }
        // (2a) SUN → freeze immunity (`gen3_sun_freeze_immunity_v1`). The base `sunnyday`
        //      weather registers `onImmunity(type) { if effectiveWeather()!=='sunnyday'
        //      return; if type==='frz' return false; }` (conditions.ts) — so while the
        //      field weather is Sun (Drought / Sunny Day), a mon CANNOT be frozen.
        //      `setStatus` → `runStatusImmunity('frz')` → `runEvent('Immunity', 'frz')`
        //      returns FALSE, so the freeze is NOT applied. This is the SAME `runStatusImmunity`
        //      position as the type immunity above — checked BEFORE `runEvent('SetStatus')`
        //      — so like the type gate it SKIPS the gen3ou clause shuffle, and the sun
        //      `onImmunity` handler is itself DRAW-FREE. So the freeze SECONDARY's
        //      `random(100)` still fires (the seed matches) but the freeze must simply not
        //      land — the A/B-fuzz "ice-freeze cluster" (196 repros, expected=None
        //      got=Some(Freeze), seed matching) was this missing gate. Probe-verified
        //      (`harness/probe_sun_freeze_immunity.js`): under Drought the same seed that
        //      freezes in no-sun leaves the mon UN-frozen with an IDENTICAL draw count
        //      (customgame), and ONE FEWER draw in gen3ou (the skipped clause shuffle).
        //      An ALREADY-frozen mon PERSISTS under sun (frz has no onWeather cure) — this
        //      gates only the APPLICATION, never a thaw. (No modeled gen-3 mon carries a
        //      weather-suppressing ability — Air Lock / Cloud Nine — is now MODELED
        //      (`gen3_ability_batch1_v1`), so this gates on the EFFECTIVE weather: an Air Lock
        //      / Cloud Nine mon on the field suppresses the sun → a freeze CAN land under
        //      raw-Sun-but-negated weather (matching `sunnyday.onImmunity` reading
        //      `effectiveWeather()`).
        if effect == "frz" && self.effective_weather(dex) == Some(Weather::Sun) {
            return;
        }
        // (2d) STATUS_IMMUNE ability, `immunity` PHASE (`gen3_status_immune_v1`, data-driven
        //      via `AbilityData.status_immune`) — an ability whose block is an
        //      `onImmunity(status)` that returns false at `runStatusImmunity`, BEFORE
        //      `runEvent('SetStatus')`. The ONLY gen-3 member is **Magma Armor** (frz) —
        //      structurally the SAME position as the sun-freeze gate above (probe-settled:
        //      `harness/probe_statusimmune_magmaarmor.js` — MA blocks at the Immunity event,
        //      the SetStatus event is NEVER reached, so NO clause shuffle fires in gen3ou).
        //      So it MUST be gated HERE, before the shuffle — DRAW-FREE. (A secondary-freeze
        //      move's `random(100)` already drew upstream; MA just blocks the application.)
        let ability = to_id(&mon.ability);
        if let Some(si) = self.status_immune_of(&ability, dex) {
            if si.phase == crate::dex::StatusImmunePhase::Immunity && si.blocks(effect) {
                return;
            }
        }

        // --- The `runEvent('SetStatus')` HANDLER-SORT SHUFFLE (the gen3ou-only draw):
        //     in formats with the `Standard` clauses (Sleep Clause Mod + Freeze Clause
        //     Mod), `setStatus` calls `runEvent('SetStatus')`, which GATHERS the 2
        //     format-clause `onSetStatus` handlers. At equal order/priority/speed they
        //     TIE → a size-2 Fisher-Yates speed-sort shuffle draws ONE `random(0,2)`
        //     EVERY time the event is reached (a status that PASSED hp/already-statused/
        //     type-immunity — incl. one the clause OR a STATUS_IMMUNE ability then BLOCKS).
        //     In gen3customgame (no clauses → the ONLY handler is the ability's own, size-1
        //     → NO tie) NO shuffle is drawn.
        //
        //     THE CRUX (probe-settled — `harness/probe_statusimmune_shuffle_size.js`): a
        //     `setStatus`-phase STATUS_IMMUNE ability (Limber/Insomnia/…) DOES register an
        //     `onSetStatus` handler → in gen3ou the SetStatus event gathers **3** handlers.
        //     But `speedSort` sorts `order→priority→speed→subOrder`, and the ability handler
        //     carries a DEFINED `speed` (the mon's speed, 96 in the probe) while the two
        //     clause handlers have `speed=undefined` → the ability sorts into its OWN group
        //     (index 0, no tie), leaving the **2 clauses a size-2 tie** → the Fisher-Yates
        //     `shuffle(list, 1, 3)` draws EXACTLY ONE `random` — IDENTICAL to the control's
        //     size-2 `shuffle(list, 0, 2)`. So a STATUS_IMMUNE ability does NOT change the
        //     draw count; `set_status_event_shuffle()` (one draw) is correct for it too. Only
        //     a genuinely UNMODELED `onSetStatus` ability (one not in the `status_immune`
        //     table AND without a known size-2-preserving sort) could desync — FAIL-LOUD on
        //     that (a future scenario can never silently break). ---
        if self.sleep_clause {
            if self.ability_unmodeled_on_set_status(&ability, dex) {
                panic!(
                    "status target ability {ability:?} has an UNMODELED onSetStatus handler \
                     — its participation in the gen3ou SetStatus handler-sort shuffle is not \
                     modeled (the STATUS_IMMUNE members sort into their own speed group so \
                     the 2-clause tie stays size-2, but this ability is NOT one of them). \
                     Probe its sort position + model it before using it under a clause format."
                );
            }
            // The size-2 clause-handler tie shuffle: exactly one `random(0,2)`
            // (bit-identical to a 2-active eachEvent shuffle on a tie). A `setStatus`-phase
            // STATUS_IMMUNE ability adds a distinctly-sorted 3rd handler that does NOT change
            // this count (the crux above).
            self.set_status_event_shuffle();
        }

        // (2b) STATUS_IMMUNE ability, `setStatus` PHASE (`gen3_status_immune_v1`, data-driven)
        //      — an `onSetStatus` handler that RETURNS false INSIDE the SetStatus event the
        //      shuffle already drew: Insomnia/Vital Spirit block slp; Limber par; Immunity
        //      psn/tox; Water Veil brn. DRAW-FREE (the block is a handler return; the shuffle
        //      already fired above). (Magma Armor's frz block is the `immunity`-phase gate
        //      above, BEFORE the shuffle — handled at (2d), which stays SILENT: the sim's
        //      `runStatusImmunity(status.id)` passes no message, and no direct gen-3 freeze
        //      move exists to capture a line against.)
        //      [EMIT] `|-immune|<target>|[from] ability: <Ability>` — the handler's own
        //      announce, ONLY for a status-MOVE source (`announce_immune_block`; a blocked
        //      SECONDARY is silent — the `(effect as Move)?.status` gate). Byte-verified vs
        //      the status_immune_lines capture (Phase 3).
        if let Some(si) = self.status_immune_of(&ability, dex) {
            if si.phase == crate::dex::StatusImmunePhase::SetStatus && si.blocks(effect) {
                if announce_immune_block && self.logging() {
                    let display = dex
                        .ability(&ability)
                        .map(|a| a.name.clone())
                        .unwrap_or_else(|| ability.clone());
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.immune_from_ability(&target, &display);
                }
                return;
            }
        }
        // (2c) SLEEP CLAUSE MOD (a clause `onSetStatus` RETURN false — DRAW-FREE; the
        //      handler-sort shuffle above already drew). ONLY in formats that carry it
        //      (gen3ou via `Standard`; NOT gen3customgame). A sleep move FAILS if any
        //      LIVING mon on the TARGET's side already has a (non-self-inflicted) sleep.
        //      All sleep in this engine is foe-inflicted (Rest is out of scope), so
        //      simply: fail if any living slot is asleep. The block is BEFORE the
        //      onStart → NO random(2,6) on a clause block (the shuffle still drew).
        if effect == "slp" && self.sleep_clause && self.side_has_sleeper(foe) {
            return;
        }
        // (2c-frz) FREEZE CLAUSE MOD (`gen3_freeze_clause_v1` — the handler-completeness
        //      audit's second real miss; the resolved rule's `onSetStatus` returns false
        //      INSIDE the SetStatus event, so the handler-sort shuffle above already
        //      drew — the block itself is DRAW-FREE, probe
        //      `harness/probe_freeze_clause_rng.js`: a blocked second freeze's turn draw
        //      count == a landed freeze's). ONLY in clause formats (gen3ou etc. — the
        //      same `sleep_clause` flag; gen3's Standard ruleset carries BOTH clauses).
        //      A freeze FAILS if any LIVING mon on the TARGET's side is already frozen.
        //      A FAINTED mon never counts — the sim sets `status = 'fnt'` on faint, so
        //      the rule's `pokemon.status === "frz"` scan can only match living mons
        //      (probe B). The rule's `source?.isAlly(target)` self-exemption is moot:
        //      gen3 has no self-inflicted freeze.
        if effect == "frz" && self.sleep_clause && self.side_has_frozen(foe) {
            return;
        }
        // Apply the status. SLEEP draws the onStart random(2,6) duration (1-4 turns,
        // gen-3 `slp.onStart`); Toxic starts at stage 0 (draw-free; the residual ramps
        // it to 1 before the first chip — mirrors `statusState.stage`); all others
        // draw-free.
        let new = match effect {
            "par" => Status::Paralysis,
            "brn" => Status::Burn,
            "frz" => Status::Freeze,
            "psn" => Status::Poison,
            "slp" => {
                // gen-3 `this.random(2,6)` (= `random_range(2,6)` ∈ {2,3,4,5}).
                let dur = self.prng.random_range(2, 6) as u8;
                Status::Sleep(dur)
            }
            // gen-3 `tox.onStart` sets `effectState.stage = 0`; the first RESIDUAL
            // ramps it to 1 (and chips 1×). `Status::Toxic(stage)` mirrors the sim's
            // `statusState.stage` EXACTLY (so the differential can assert it), and
            // `apply_status_dot` does the `if stage<15 stage++` ramp BEFORE the damage.
            "tox" => Status::Toxic(0),
            _ => return,
        };
        self.sides[foe].pokemon[foe_slot].status = Some(new);
        // [EMIT] `|-status|<target>|<status>` — a foe-inflicted status (secondary /
        // standalone status move / Tri Attack). NO `[from]` (only Rest's self-inflict
        // carries `[from] move: Rest`, emitted by `run_rest` directly). A SYNCHRONIZE
        // REFLECT apply (`sync_reveal`) instead carries the `[from] ability: Synchronize|
        // [of] <holder>` reveal — emitted HERE so the holder's Lum tail (below) follows
        // it in the byte-verified order. Observation-only.
        if self.logging() {
            let target = self.mon_ref(foe, foe_slot, dex);
            match sync_reveal {
                Some((h_side, h_slot)) => {
                    let holder_ref = self.mon_ref(h_side, h_slot, dex);
                    self.log.status_from_ability(&target, effect, "Synchronize", &holder_ref);
                }
                None => self.log.status(&target, effect, None),
            }
        }

        // --- SYNCHRONIZE (`gen3_ability_batch2_v1`, `synchronize.onAfterSetStatus`) — when the
        //     holder (`foe`) is inflicted a MAJOR status by a FOE `source`, REFLECT it back to
        //     that source. The resolved gen3 handler:
        //       if (!source || source === target) return;  // no self-inflict / source-less
        //       let id = status.id; if (id==='slp'||id==='frz') return;  // slp/frz EXEMPT
        //       if (id==='tox') id='psn';                   // tox reflects as psn
        //       source.trySetStatus(id, target);
        //     The reflected `trySetStatus` runs with `source=None` (Synchronize does NOT chain a
        //     re-reflect — the reflected apply's own onAfterSetStatus has `source===target`
        //     false but the ORIGINAL holder isn't a Synchronize-reflect target of itself; a
        //     source-less reflect also avoids any infinite ping-pong). DRAW-FREE in gen3customgame
        //     (the reflected status draws no clause shuffle); in gen3ou it draws the reflected
        //     status's OWN SetStatus 2-clause shuffle (via the recursive `try_set_status`).
        //     PROBE-settled: `harness/probe_synchronize_rng.js`.
        if let Some((src_side, src_slot)) = source {
            if (src_side, src_slot) != (foe, foe_slot)
                && dex.ability(&to_id(&self.sides[foe].pokemon[foe_slot].ability)).map(|a| a.synchronize).unwrap_or(false)
            {
                // slp/frz are exempt; tox reflects as psn; the rest reflect as-is.
                let reflect = match effect {
                    "slp" | "frz" => None,
                    "tox" => Some("psn"),
                    other => Some(other),
                };
                if let Some(refl) = reflect {
                    // The `[from] ability: Synchronize|[of] <holder>` reveal is emitted by the
                    // reflected `try_set_status` as a plain `|-status|` — Showdown adds the
                    // `[from] ability: Synchronize` via the effect context. We mirror the reveal
                    // form directly for the source (observation-only; the seed suites are
                    // unaffected since the reflected apply's DRAW behaviour is what matters).
                    self.apply_synchronize_reflect(foe, foe_slot, src_side, src_slot, refl, dex);
                }
            }
        }

        // --- LUM's IMMEDIATE eat (`gen3_berry_trace_shedskin_v1`, `lum.onAfterSetStatus`
        //     priority -1 — AFTER Synchronize's priority-0 reflect, probe-verified line
        //     order `|-status|holder| → |-status|source|…Synchronize| → |-enditem| →
        //     |-curestatus|`). DRAW-FREE. The single-status cure berries do NOT fire here
        //     (no onAfterSetStatus) — they wait for the next Update site. ---
        self.berry_after_set_status(foe, foe_slot, dex);
    }

    /// Reflect a Synchronize holder's status back to its SOURCE (`gen3_ability_batch2_v1`).
    /// Applies `refl` to `(src_side, src_slot)` via the status choke point with `source=None`
    /// (no re-reflect chain) and emits the `|-status|…|[from] ability: Synchronize|[of] <holder>`
    /// reveal. Separated so the recursive `try_set_status` call is explicit + the emit form is
    /// the Synchronize one (a foe status inflicted by the ability, not a bare move status).
    fn apply_synchronize_reflect(
        &mut self,
        holder_side: usize,
        holder_slot: usize,
        src_side: usize,
        src_slot: usize,
        refl: &str,
        dex: &Dex,
    ) {
        // The recursive apply. `source=None` → no re-reflect (Synchronize doesn't ping-pong; the
        // reflected mon isn't a Synchronize target of the holder). This draws the reflected
        // status's own SetStatus shuffle in gen3ou / draw-free in gen3customgame, and applies the
        // gen-3 type/ability/already-statused gates on the SOURCE. `sync_reveal` makes the
        // apply's own `-status` emit carry the `[from] ability: Synchronize|[of] <holder>`
        // reveal (a blocked/no-op'd reflect emits nothing), and lets the SOURCE's own Lum
        // `-enditem`/`-curestatus` tail fire INSIDE the apply, AFTER the reveal — the
        // byte-verified Phase-3 interleave (`-status holder → -status source [from]
        // Synchronize → -enditem → -curestatus`).
        self.try_set_status_impl(
            src_side, src_slot, refl, None, false, Some((holder_side, holder_slot)), dex,
        );
    }

    /// The STATUS_IMMUNE params of `ability_id` (`gen3_status_immune_v1`, data-driven via
    /// `AbilityData.status_immune`) — the class of abilities that grant immunity to a
    /// specific MAJOR status (Limber par / Insomnia+Vital Spirit slp / Immunity psn,tox /
    /// Water Veil brn via `onSetStatus`; Magma Armor frz via `onImmunity`). `None` for an
    /// ability with no status immunity (or an unknown id). A cheap dex lookup — the caller
    /// checks `.phase` + `.blocks(status)`.
    fn status_immune_of<'a>(&self, ability_id: &str, dex: &'a Dex) -> Option<&'a crate::dex::StatusImmune> {
        dex.ability(ability_id).and_then(|a| a.status_immune.as_ref())
    }

    /// Whether `ability_id` has an `onSetStatus` handler the port does NOT model for the
    /// gen3ou SetStatus handler-sort shuffle. The MODELED `onSetStatus` abilities are exactly
    /// the `setStatus`-phase STATUS_IMMUNE members (Limber/Insomnia/Vital Spirit/Immunity/
    /// Water Veil) — probe-proven to sort into their OWN speed group so the 2-clause tie stays
    /// size-2 (`set_status_event_shuffle`'s one draw is unchanged). Any OTHER gen-3 ability
    /// with an `onSetStatus` (Leaf Guard — num 102, not even gen-3-legal; Synchronize's
    /// `onAfterSetStatus` is a DIFFERENT event) would need its own sort-position probe, so we
    /// FAIL-LOUD on it under a clause format. Magma Armor is NOT here (it blocks via
    /// `onImmunity`, phase=Immunity, so it registers NO SetStatus handler).
    fn ability_unmodeled_on_set_status(&self, ability_id: &str, dex: &Dex) -> bool {
        // The set of abilities whose onSetStatus participation IS modeled = the
        // setStatus-phase STATUS_IMMUNE members. Everything else with an onSetStatus is
        // unmodeled. We can't enumerate "has an onSetStatus" from the data (the JS callback
        // is invisible), so we hardcode the gen-3 abilities KNOWN to carry one and check
        // whether each is modeled. In gen-3 the onSetStatus abilities are exactly the 5
        // setStatus-phase STATUS_IMMUNE members (probe-enumerated: Limber/Insomnia/Immunity/
        // Water Veil/Vital Spirit) + Synchronize (which uses onAfterSetStatus, a DIFFERENT
        // event that does NOT add a SetStatus handler) — so the modeled check is exactly the
        // status_immune-with-setStatus-phase membership.
        let known_on_set_status = matches!(
            ability_id,
            "limber" | "insomnia" | "immunity" | "waterveil" | "vitalspirit"
        );
        if !known_on_set_status {
            return false;
        }
        // Modeled iff it is a setStatus-phase STATUS_IMMUNE member (all 5 above are).
        !matches!(
            self.status_immune_of(ability_id, dex).map(|si| si.phase),
            Some(crate::dex::StatusImmunePhase::SetStatus)
        )
    }

    /// Whether `side` has ANY living mon (active OR bench) already asleep — the
    /// gen3ou Sleep Clause check. All sleep this engine inflicts is foe-sourced (Rest
    /// is out of scope), so an asleep mon always counts toward the clause (the sim
    /// excludes only ally-sourced / self Rest sleep, which we never set).
    fn side_has_sleeper(&self, side: usize) -> bool {
        self.sides[side]
            .pokemon
            .iter()
            .any(|m| m.hp > 0 && !m.fainted && matches!(m.status, Some(Status::Sleep(_))))
    }

    /// Any LIVING mon on `side` frozen — the Freeze Clause Mod scan
    /// (`gen3_freeze_clause_v1`). A fainted mon's sim status is `'fnt'`, never `'frz'`
    /// (probe `harness/probe_freeze_clause_rng.js` B), so only living mons count.
    fn side_has_frozen(&self, side: usize) -> bool {
        self.sides[side]
            .pokemon
            .iter()
            .any(|m| m.hp > 0 && !m.fainted && matches!(m.status, Some(Status::Freeze)))
    }

    /// Apply a STRUCTURED stat-boost secondary (`boost()` — DRAW-FREE; `boostBy`
    /// clamps each stage to `[-6, 6]`, pokemon.ts:1222). The `want_self` selects which
    /// of the move's `secondary_boosts` entries to apply: `false` = the foe stat-DROP
    /// (Crunch −1 SpD), `true` = the user's self stat-RAISE (Meteor Mash +1 Atk). The
    /// target mon is the foe (drop) or the user (raise).
    ///
    /// Boost-immunity ABILITIES on the foe (DRAW-FREE `onTryBoost`) block a foe
    /// stat-DROP and the stage stays put: Clear Body / White Smoke (all stats),
    /// Hyper Cutter (atk only), Keen Eye (accuracy only). These never apply to a
    /// self-RAISE. (Showdown clamps a cap-reached boost to no-op too, which the
    /// `[-6, 6]` clamp here reproduces — the secondary random(100) STILL drew either
    /// way, so this is purely a STATE gate.)
    fn apply_secondary_boost(
        &mut self,
        side: usize,
        slot: usize,
        foe: usize,
        foe_slot: usize,
        want_self: bool,
        boosts: &[crate::dex::moves::SecondaryBoost],
        _dex: &Dex,
    ) {
        // Find the matching structured block (foe-drop vs self-raise). A gen-3 move has
        // at most one of each, so the first match is unambiguous.
        let spec = match boosts.iter().find(|b| b.target_self == want_self) {
            Some(b) => b,
            None => return, // no structured spec → nothing to apply (data gap; no-op).
        };
        let (t_side, t_slot) = if want_self { (side, slot) } else { (foe, foe_slot) };
        // A KO'd target takes no boost (the apply happens on a live mon).
        {
            let mon = &self.sides[t_side].pokemon[t_slot];
            if mon.fainted || mon.hp == 0 {
                return;
            }
        }
        // Foe stat-DROP immunity abilities (DRAW-FREE onTryBoost). N/A for a self-raise.
        let foe_ability = if want_self {
            String::new()
        } else {
            to_id(&self.sides[t_side].pokemon[t_slot].ability)
        };
        for &(idx, stages) in &spec.boosts {
            if !want_self && stat_drop_blocked(&foe_ability, idx) {
                continue; // Clear Body / White Smoke / Hyper Cutter / Keen Eye.
            }
            let cur = self.sides[t_side].pokemon[t_slot].boosts[idx] as i32;
            let next = (cur + stages as i32).clamp(-6, 6);
            self.sides[t_side].pokemon[t_slot].boosts[idx] = next as i8;
            // [EMIT] `|-boost|`/`|-unboost|<target>|<stat>|<mag>` by the SIGN of the
            // CLAMPED delta actually applied (a boost/drop into the ±6 cap has delta 0 →
            // emits nothing, matching the sim). The foe-drop targets the foe; a self-raise
            // targets the user (`_dex` reads the display name). Observation-only.
            if self.logging() {
                let delta = (next - cur) as i8;
                let target = self.mon_ref(t_side, t_slot, _dex);
                self.log.boost(&target, idx, delta);
            }
        }
    }

    /// Build the [`DamageContext`] for `side`'s active hitting `foe`'s active. The
    /// `crit` field is left `false` here (the caller sets it after the crit roll).
    /// Resolves the gen-3 ability/Levitate immunity + Thick Fat + Choice Band /
    /// type-item / Sea Incense stat mods + burn + weather + screens (screens are
    /// none in this bounded step — no side conditions are tracked yet).
    #[allow(clippy::too_many_arguments)]
    fn build_damage_context(
        &self,
        side: usize,
        slot: usize,
        foe: usize,
        foe_slot: usize,
        base_power: u16,
        move_type: Option<Type>,
        category: MoveCategory,
        halves_def: bool,
        dex: &Dex,
    ) -> DamageContext {
        let atk_mon = &self.sides[side].pokemon[slot];
        let def_mon = &self.sides[foe].pokemon[foe_slot];

        // CURRENT types via the `mon_types` choke point (the species types, unless a
        // Color Change `types_override` replaced them — `gen3_ability_batch4_v1`).
        let atk_types = mon_types(atk_mon, dex);
        let def_types = mon_types(def_mon, dex);

        let attacker = Combatant {
            level: atk_mon.level,
            atk_stat: atk_mon.stats[1],
            spa_stat: atk_mon.stats[3],
            def_stat: atk_mon.stats[2],
            spd_stat: atk_mon.stats[4],
            types: atk_types,
            // damage boosts are [atk, def, spa, spd, spe] (drop accuracy/evasion).
            boosts: [
                atk_mon.boosts[0],
                atk_mon.boosts[1],
                atk_mon.boosts[2],
                atk_mon.boosts[3],
                atk_mon.boosts[4],
            ],
            burned: atk_mon.status == Some(Status::Burn),
            has_guts: to_id(&atk_mon.ability) == "guts",
        };
        let defender = Combatant {
            level: def_mon.level,
            atk_stat: def_mon.stats[1],
            spa_stat: def_mon.stats[3],
            def_stat: def_mon.stats[2],
            spd_stat: def_mon.stats[4],
            types: def_types,
            boosts: [
                def_mon.boosts[0],
                def_mon.boosts[1],
                def_mon.boosts[2],
                def_mon.boosts[3],
                def_mon.boosts[4],
            ],
            burned: false,
            has_guts: false,
        };

        let mv = MoveInput { base_power, move_type, category, halves_defense: halves_def };

        // Runtime conditions the ability DMG_MOD folds gate on (`gen3_item_mechanics_v1`
        // ability side): a major status on either mon, and the attacker's PINCH state
        // `3*hp <= maxhp` (bit-exactly the sim's integer-hp `hp <= maxhp/3` float compare
        // — probe-verified at the maxhp=341 float boundary). A fainted (hp==0) mon never
        // attacks, so `3*0 <= maxhp` is harmless dead state.
        let atk_ability = to_id(&atk_mon.ability);
        let def_ability_id = to_id(&def_mon.ability);
        let attacker_statused = atk_mon.status.is_some();
        let defender_statused = def_mon.status.is_some();
        let attacker_in_pinch = 3 * (atk_mon.hp as u32) <= atk_mon.maxhp as u32;

        // Attacker stat mods (item/ability) — DATA-DRIVEN from the dex mechanics
        // fields (`gen3_item_mechanics_v1`): Choice Band ×1.5 (physical), the
        // stat-fold type-boost items ×1.1 / Sea Incense ×1.05, the species stat
        // items (Thick Club / Light Ball / DeepSeaTooth / Soul Dew SpA) + the ability
        // ModifyAtk folds (Huge/Pure Power ×2, Guts ×1.5 when statused). Flash Fire
        // ×1.5 (Fire while boosted) — Flash Fire-boost state is not tracked yet, so
        // omitted.
        let mut atk_stat_mods = resolve_atk_stat_mods(
            &atk_mon.item,
            &atk_ability,
            &atk_mon.species_id,
            move_type,
            category,
            attacker_statused,
            dex,
        );
        // PLUS / MINUS (`gen3_plus_minus_v1`): the gen3 RESOLVED `onModifySpA` scans
        // `getAllActive()` — FOES INCLUDED (gen5+ narrowed it to allies; the old NOOP
        // classification's "partner-less in singles → no-op" missed the cross-field
        // pairing, the A/B fuzzer's thunderbolt-vs-Plusle/Minun STATE cluster). ×1.5
        // SpA iff ANY non-fainted ACTIVE carries the PAIRED ability (plus↔minus ONLY —
        // same-ability pairs do NOT boost in gen3). SPECIAL only (a ModifySpA chain
        // member), CURRENT-ability read (`hasAbility` — a Traced pair counts),
        // DRAW-FREE. Probe: `harness/probe_plus_minus_gen3.js` (maxRoll 90 vs 60
        // control = ×1.5 both directions; plus-vs-plus / minus-vs-minus = 60 = no
        // boost; physical 35 == 35; post-turn seed identical to the control's).
        if category == MoveCategory::Special {
            let paired = match atk_ability.as_str() {
                "plus" => Some("minus"),
                "minus" => Some("plus"),
                _ => None,
            };
            if let Some(paired) = paired {
                let pair_on_field = (0..2usize).any(|s| {
                    let m = &self.sides[s].pokemon[self.sides[s].active];
                    !m.fainted && to_id(&m.ability) == paired
                });
                if pair_on_field {
                    atk_stat_mods.push(AtkStatMod::Item { num: 3, den: 2 });
                }
            }
        }
        // Hustle's Atk ×1.5 DIRECT modify (`gen3_accuracy_pipeline_v1`) — PHYSICAL only
        // (onModifyAtk touches only the Atk stat), applied separately before the chain.
        // Shipped WITH Hustle's ×0.8 accuracy side (both this phase). Read from the same
        // `dmg_mod` the DMG_MOD phase left `direct` + unwired.
        let atk_direct_modify = if category == MoveCategory::Physical {
            dex.ability(&atk_ability)
                .and_then(|a| a.dmg_mod.as_ref())
                .filter(|m| m.direct && m.fold == DmgFold::Atk)
                .map(|m| (m.num, m.den))
        } else {
            None
        };
        // Defender stat mods (DeepSeaScale / Metal Powder / Soul Dew SpD + Marvel Scale
        // Def ×1.5 when the defender is statused) + the attacker base-power mods (the
        // incenses / the bows + the pinch family Torrent/Blaze/Overgrow/Swarm).
        let def_stat_mods = resolve_def_stat_mods(
            &def_mon.item,
            &def_ability_id,
            &def_mon.species_id,
            category,
            defender_statused,
            dex,
        );
        let bp_mods =
            resolve_bp_mods(&atk_mon.item, &atk_ability, move_type, attacker_in_pinch, dex);

        // Defender ability immunity + Thick Fat (reuses `def_ability_id` from above).
        // A FROZEN Flash Fire holder is NOT fire-immune (`gen3_ff_frozen_no_absorb_v1`) —
        // the resolved `flashfire.onTryHit`'s `if (target.status === "frz") return;` lets
        // the Fire move proceed normally (full acc/crit/dmg/secondary draws; the
        // fire-move thaw then cures the freeze post-hit, and the absorb never arms —
        // `apply_flash_fire_activation` already skips frz). The pre-fix port kept the
        // frozen holder immune → accuracy-only + a phantom thaw roll on its own move
        // (the ab_1309_23 flamethrower-vs-frozen-Houndoom 3-vs-9-draw desync).
        let (immune, thick_fat) = resolve_defender_ability(
            &def_ability_id,
            move_type,
            def_mon.status == Some(Status::Freeze),
        );

        // FLASH FIRE ×1.5 (`gen3_flashfire_boost_v1`): the attacker's `flash_fire` volatile is
        // ARMED (it absorbed a Fire move earlier) AND this move is Fire-type. A ModifyDamagePhase1
        // damage fold (probe-settled — NOT a stat mod), so it is category-agnostic (both phys +
        // spec Fire moves) and folded in `modify_damage`. DRAW-FREE.
        let flash_fire = atk_mon.flash_fire && move_type == Some(Type::Fire);

        // Weather (only rain/sun touch damage). Sand/Hail have no damage multiplier
        // in gen3 OU here. WEATHER_NEGATE (`gen3_ability_batch1_v1`): a Cloud Nine / Air Lock
        // mon on the field suppresses the rain/sun damage mod (`effective_weather`).
        let weather = match self.effective_weather(dex) {
            Some(Weather::Rain) => Some(DmgWeather::Rain),
            Some(Weather::Sun) => Some(DmgWeather::Sun),
            _ => None,
        };

        DamageContext {
            attacker,
            defender,
            mv,
            crit: false, // set by the caller after the crit roll
            weather,
            reflect: false,      // side conditions not tracked this step
            light_screen: false, // side conditions not tracked this step
            atk_stat_mods,
            atk_direct_modify,
            def_stat_mods,
            bp_mods,
            defender_thick_fat: thick_fat,
            immune,
            flash_fire,
        }
    }

    /// Apply `dmg` HP to `side`'s `slot` mon, saturating at 0. Mirrors
    /// `pokemon.damage` (`hp -= d; if hp <= 0 { hp = 0 }`) — it ZEROES the HP but does
    /// **NOT** set the `fainted` flag. In Showdown, `faint()` only enqueues the mon
    /// in `faintQueue`; the `fainted` flag is set later by `faintMessages()`, which
    /// runs at the END of the `runAction` — AFTER the in-`tryMoveHit`
    /// `eachEvent('Update')` shuffle. So a mon at 0 HP is STILL counted by
    /// `getAllActive()` (hence still in the in-tryMoveHit 2-active speed-tie shuffle)
    /// until [`BattleState::process_faints`] runs. Keeping these two events separate
    /// is the draw-COUNT crux on a faint turn (the KO move's in-tryMoveHit shuffle
    /// fires; only then is the mon excluded). Returns whether the mon hit 0 HP.
    /// JUMP KICK / HIGH JUMP KICK crash damage (`gen3_jump_kick_crash_v1` — the
    /// handler-completeness audit's first real miss). The resolved gen3 `onMoveFail`:
    ///
    /// ```js
    /// onMoveFail(target, source, move) {
    ///   if (target.runImmunity("Fighting")) {
    ///     const damage = this.actions.getDamage(source, target, move, true);
    ///     this.damage(this.clampIntRange(damage / 2, 1, Math.floor(target.maxhp / 2)),
    ///                 source, source, move);
    ///   }
    /// }
    /// ```
    ///
    /// Fires when the move FAILS — an accuracy MISS or a Protect BLOCK (probe D) — but
    /// NOT against a Fighting-immune (Ghost) target (`runImmunity` gate; the type-immune
    /// miss already took the `-immune` short-circuit, and a protect-blocked JK into a
    /// Ghost is gated here, draw-free). PROBE-SETTLED
    /// (`harness/probe_jumpkick_crash_rng.js`):
    ///   * the crash's `getDamage` DRAWS the normal crit `randomChance(1, critMult)`
    ///     (Focus Energy shifts the denominator; a crit-immune DEFENDER overrides the
    ///     drawn crit draw-free) + the `random(16)` damage roll — a missed JK is exactly
    ///     +2 draws vs a missed control move;
    ///   * crash = `floor(rolled/2)` clamped to `[1, floor(TARGET.maxhp/2)]` (the
    ///     TARGET's maxhp, not the user's — probe A: 125 > user-maxhp/2 120);
    ///   * the crash is a MOVE-effect Damage event into the USER — Focus Band draws its
    ///     `randomChance(1,10)` and CAN survive-at-1 (`focus_band_damage(is_move=true)`);
    ///   * the crash can FAINT the user (probe F; the deciding-faint Quick Claw skip
    ///     then applies via the normal turn machinery).
    #[allow(clippy::too_many_arguments)]
    fn apply_jump_kick_crash(
        &mut self,
        side: usize,
        slot: usize,
        foe: usize,
        foe_slot: usize,
        base_power: u16,
        move_type: Option<Type>,
        category: MoveCategory,
        crit_ratio: u8,
        dex: &Dex,
    ) {
        let mut ctx = self.build_damage_context(
            side, slot, foe, foe_slot, base_power, move_type, category, false, dex,
        );
        // `target.runImmunity("Fighting")`: a Fighting-immune (Ghost) target → no crash,
        // NO draws (probe C: accuracy-only draw count).
        if move_is_immune(&ctx, dex) {
            return;
        }
        // The crash `getDamage` draws the SAME crit + damage-roll sequence as a landed
        // hit (Focus Energy denominator shift + the crit-immune draw-free override).
        let eff_crit_ratio = if self.sides[side].pokemon[slot].focus_energy {
            (crit_ratio as u32 + 2).min(5)
        } else {
            crit_ratio as u32
        };
        let mut crit = self.prng.random_chance(1, CRIT_MULT[eff_crit_ratio as usize]);
        if crit
            && dex
                .ability(&self.sides[foe].pokemon[foe_slot].ability)
                .map(|a| a.crit_immune)
                .unwrap_or(false)
        {
            crit = false;
        }
        ctx.crit = crit;
        let dmg = crate::damage::calc_damage(&ctx, dex);
        let r = self.prng.random_below(16) as usize;
        let rolled = dmg.rolls[r];
        // clampIntRange(damage / 2, 1, floor(target.maxhp / 2)) — integer floor, min 1,
        // ceiling = the TARGET's maxhp/2.
        let ceiling = (self.sides[foe].pokemon[foe_slot].maxhp / 2).max(1);
        let crash = (rolled / 2).clamp(1, ceiling);
        // The Damage event into the USER (effect = the MOVE): Focus Band rolls + can
        // survive-at-1 a lethal crash (`is_move = true`).
        let crash = self.focus_band_damage(side, slot, crash, true, dex);
        self.apply_damage(side, slot, crash);
        // [EMIT] `|-damage|<user>|<HP>` (after the `-miss` / Protect `-activate` line).
        if self.logging() {
            let user = self.mon_ref(side, slot, dex);
            let hp = self.hp_status(side, slot);
            self.log.damage(&user, &hp, None);
        }
    }

    fn apply_damage(&mut self, side: usize, slot: usize, dmg: u16) -> bool {
        let mon = &mut self.sides[side].pokemon[slot];
        if dmg == 0 {
            return false;
        }
        if dmg >= mon.hp {
            mon.hp = 0;
            // [EMIT] record the faint ORDER (this mon was zeroed now) so
            // `process_faints` can emit `|faint|` in Showdown `faintQueue` order.
            // Observation-only: a Vec push behind the logging gate, no PRNG. Guard
            // against a double-push if the same active is somehow zeroed twice in one
            // action (it can't in the modeled scope, but keep it idempotent).
            self.record_faint_order(side, slot);
            true
        } else {
            mon.hp -= dmg;
            false
        }
    }

    /// Record that `side`'s `slot` (an ACTIVE) mon hit 0 HP — the `faintQueue`
    /// ENQUEUE order (see [`crate::state::BattleState::faint_emit_queue`]).
    ///
    /// DRAW-BEARING, not just emission (`gen3_faint_queue_order_v1`): Showdown's
    /// `faintMessages` drains the queue in enqueue order, fully processing each
    /// corpse (`fainted = true`, `isActive = false`) BEFORE the next corpse's
    /// ability-`End` singleEvent — so on a DOUBLE faint the second corpse's Cloud
    /// Nine / Air Lock `onEnd → eachEvent('WeatherChange')` no longer gathers the
    /// first corpse, and a cached-speed tie between the two corpses draws NO
    /// shuffle. A side-order walk processed the WRONG corpse first whenever the
    /// enqueue order was side-1-first (e.g. an Explosion self-faint enqueues the
    /// USER before the KO'd target — the ab_723_13 extra draw). A pure `Vec` push
    /// (no PRNG), only for the ACTIVE slot (the only faintable mon in gen-3
    /// singles), idempotent per side.
    fn record_faint_order(&mut self, side: usize, slot: usize) {
        if slot == self.sides[side].active && !self.faint_emit_queue.contains(&side) {
            self.faint_emit_queue.push(side);
        }
    }

    /// SUBSTITUTE ABSORB (`substitute.onTryPrimaryHit`, moves.ts): if `side`'s `slot` mon
    /// has a substitute, route `dmg` to the SUB's HP instead of the mon and return `true`
    /// (the mon takes NO damage). The sub's HP is reduced by `min(dmg, sub.hp)` (the excess
    /// does NOT carry to the mon in gen-3 — `if (damage > sub.hp) damage = sub.hp`); when it
    /// reaches 0 the sub BREAKS (→ `None`). DRAW-FREE (the crit/damage rolls were already
    /// drawn by the caller before the sub-intercept). Returns `false` (and does nothing) when
    /// there is NO sub — the caller then applies the damage to the mon normally.
    ///
    /// A `dmg == 0` hit into a sub STILL "absorbs" (returns `true`, the sub took a 0 hit) —
    /// gen-3 routes a 0-damage move into the sub (`-activate Substitute [damage]`), so the
    /// mon is untouched; the sub HP is unchanged. (No modeled 0-damage damaging move reaches
    /// here, but the branch is correct.)
    fn absorb_into_sub(&mut self, side: usize, slot: usize, dmg: u16) -> SubAbsorb {
        let mon = &mut self.sides[side].pokemon[slot];
        let sub_hp = match mon.substitute {
            Some(hp) => hp,
            None => return SubAbsorb::NoSub, // no sub → the caller hits the mon
        };
        let taken = dmg.min(sub_hp); // excess does NOT carry to the mon (gen-3)
        let remaining = sub_hp - taken;
        if remaining == 0 {
            mon.substitute = None;
            SubAbsorb::Broke // the sub is gone → `-end`
        } else {
            mon.substitute = Some(remaining);
            SubAbsorb::Held // the sub survived → `-activate …|[damage]`
        }
    }

    /// Process pending faints (`faintMessages` setting `fainted = true`): mark every
    /// 0-HP active mon `fainted` and decrement its side's `pokemon_left`. Runs at the
    /// END of a `runAction` — AFTER the in-`tryMoveHit` Update shuffle — so a 0-HP mon
    /// is excluded from `getAllActive()` (the later eachEvent / residual shuffles)
    /// only from this point on. Returns whether any active was newly fainted.
    fn process_faints(&mut self, dex: &Dex) -> bool {
        // `faintQueue` ORDER — DRAW-BEARING (`gen3_faint_queue_order_v1`): Showdown's
        // `faintMessages` drains `faintQueue` in the order faints were ENQUEUED, and each
        // corpse is fully processed (`fainted = true`) BEFORE the next corpse's ability-`End`
        // fires — so on a double faint the SECOND corpse's Cloud Nine / Air Lock
        // `onEnd → eachEvent('WeatherChange')` gathers only itself (no tie, NO shuffle draw)
        // even when the two corpses tie on cached speed. `apply_damage` / the explosion
        // self-KO recorded the enqueue order into `faint_emit_queue` (unconditionally); walk
        // it FIRST (in order), then fall back to side order for any 0-HP active not in the
        // queue (a residual/state path that zeroed HP without a push — keeps the walk robust
        // even if a future faint site forgets to record). This order also drives the
        // `|faint|` emission lines.
        let ordered: Vec<usize> = {
            let mut o: Vec<usize> = std::mem::take(&mut self.faint_emit_queue);
            for side in 0..2 {
                if !o.contains(&side) {
                    o.push(side);
                }
            }
            o
        };

        let mut any = false;
        for &side in &ordered {
            let slot = self.sides[side].active;
            // WEATHER_NEGATE `onEnd` → `eachEvent('WeatherChange')` at the FAINT site
            // (`gen3_cloudnine_end_v1`, the ab_916_16 fingerprint): `faintMessages` fires the
            // faintee's ability `End` singleEvent (battle.js:2109) BEFORE `pokemon.fainted =
            // true` — so a KO'd Cloud Nine / Air Lock holder's `onEnd` runs its
            // `eachEvent("WeatherChange")` with the DYING mon STILL in `getAllActive` (hp 0,
            // fainted false), drawing ONE tie-shuffle iff the two actives tie on cached
            // speed. Mirror it here, BEFORE setting `fainted` (the port's
            // `each_event_shuffle` filters on `fainted`, matching `getAllActive`).
            {
                let m = &self.sides[side].pokemon[slot];
                if m.hp == 0
                    && !m.fainted
                    && dex.ability(&to_id(&m.ability)).map(|a| a.weather_negate).unwrap_or(false)
                {
                    self.each_event_shuffle();
                }
            }
            let mon = &mut self.sides[side].pokemon[slot];
            if mon.hp == 0 && !mon.fainted {
                mon.fainted = true;
                // `faintMessages` → `clearVolatile` ZEROES the faintee's BOOSTS
                // (pokemon.js:1080 — the first thing clearVolatile does). LOAD-BEARING for
                // the replacement instaswitch sort (`gen3_fnt_clears_status_v1`, the
                // ab_806_16 fingerprint): the corpse's gen3 `getActionSpeed()` reads the
                // boost-table-folded speed, so a +6-Agility Metagross corpse must sort at
                // its PLAIN 145 (tying a 145 Swalot corpse → the shuffle draw), not a
                // stale 580. (The alive-outgoing boost clear lives in `execute_switch`;
                // this is the FAINT-path mirror.)
                mon.boosts = [0; crate::state::BOOST_LEN];
                // `faintMessages` → `clearVolatile` drops the faintee's volatiles. The LEECH
                // SEED volatile clears here (a fainted seeded mon is no longer seeded — the
                // golden reads `p2Seed=0` on the turn the leech drain KOs it). The other
                // volatiles (confusion/flinch/protect/stall) clear on the subsequent switch-out
                // (`execute_switch`) and the differential only compares them on a LIVE mon, so
                // leech is the one whose post-faint state the per-decision golden asserts.
                mon.leech_seed = None;
                // clearVolatile also drops the SUBSTITUTE on faint (a fainted mon has no
                // sub). (A sub-owner can only faint to NON-absorbed damage — confusion
                // self-hit, residual chip, or a hit AFTER the sub broke — so this keeps the
                // state clean for a re-encode.)
                mon.substitute = None;
                // clearVolatile also drops the CHOICE LOCK on faint (`gen3_pp_tracking_v1`) —
                // a fainted Choice-Band mon's replacement enters unlocked; and if THIS mon is
                // ever re-encoded it must not carry a stale lock.
                mon.choice_locked_move = None;
                // clearVolatile also drops the TAUNT + DISABLE volatiles + resets `last_move`
                // on faint (`gen3_taunt_disable_v1`): a fainted mon carries no selection
                // restriction, and its replacement enters with no lastMove.
                mon.taunt = None;
                mon.disable = None;
                mon.last_move = None;
                // clearVolatile also drops the FLASH FIRE activation on faint — a fainted FF
                // mon carries no boost, and if re-encoded must not show a stale `flash_fire`.
                mon.flash_fire = false;
                // `faintMessages` decrements the side's live-mon count — the count `check_win`
                // reads to END the battle (a side at `pokemon_left == 0` has lost). The `mon`
                // borrow above ends before this disjoint-field write.
                let left = &mut self.sides[side].pokemon_left;
                *left = left.saturating_sub(1);
                any = true;
                // [EMIT] `|faint|<mon>` at the sim's `faintMessages` position (in queue order).
                if self.logging() {
                    let mon_ref = self.mon_ref(side, slot, dex);
                    self.log.faint(&mon_ref);
                }
            }
        }
        any
    }
}

// ===========================================================================
// PROTOCOL EMISSION (level-2, Phase 1) — the OBSERVATION-ONLY line hooks.
//
// Every method here is a pure READ of already-computed state formatted into a
// `|...|` line pushed onto `self.log`; NONE draws from the PRNG. They are no-ops
// when `self.log` is disabled (the seed suite's path), so wiring them changes NO
// seed assertion. See `protocol.rs` + `PROTOCOL_EMISSION_DESIGN.md`.
// ===========================================================================

impl BattleState {
    /// Whether protocol emission is active (gates all the emit hooks — the
    /// disabled path pushes nothing AND, crucially, skips the state reads that
    /// build the emit args, so the hot seed-suite loop is untouched).
    #[inline]
    fn logging(&self) -> bool {
        self.log.is_enabled()
    }

    /// The IDENTIFIER name of `side`'s `slot` mon — the on-field NICKNAME, which is
    /// what every `p<N>a: <name>` protocol token (`|move|` user/target, `|switch|`,
    /// `|-damage|`, `|-ability|`, `|-status|`, `[of]`, …) references. This mirrors
    /// Showdown's `Pokemon.name` = `set.name || species.name`: the packed set's
    /// nickname when present (e.g. `Electhor` for a Zapdos), falling back to the
    /// species DISPLAY name only when the set carries no nickname. poke-env tracks
    /// each mon by THIS token, so rendering the species here (instead of the
    /// nickname) makes it fail to match the mon it already knows and try to ADD a
    /// 7th — the localized/nicknamed-team crash. NOTE: the SPECIES (for `|switch|`
    /// details, `Zapdos`) comes from [`species_name`], NOT this.
    fn display_name(&self, side: usize, slot: usize, dex: &Dex) -> String {
        let nick = &self.sides[side].pokemon[slot].set.name;
        if nick.is_empty() { self.species_name(side, slot, dex) } else { nick.clone() }
    }

    /// The SPECIES display name of `side`'s `slot` mon (e.g. `Zapdos`), for the
    /// `|switch|`/`|drag|` DETAILS field — distinct from the ident nickname
    /// ([`display_name`]).
    fn species_name(&self, side: usize, slot: usize, dex: &Dex) -> String {
        let mon = &self.sides[side].pokemon[slot];
        dex.species(&mon.species_id).map_or_else(|| mon.species_id.clone(), |s| s.name.clone())
    }

    /// A `MonRef` (`p<N>a: <Name>`) for `side`'s `slot` mon.
    fn mon_ref(&self, side: usize, slot: usize, dex: &Dex) -> MonRef {
        MonRef { side, name: self.display_name(side, slot, dex) }
    }

    /// A `MonRef` for `side`'s ACTIVE mon.
    fn active_ref(&self, side: usize, dex: &Dex) -> MonRef {
        self.mon_ref(side, self.sides[side].active, dex)
    }

    /// The HP-field (`x/y` / `x/y <status>` / `0 fnt`) for `side`'s `slot` mon,
    /// reading its live hp/maxhp/status.
    fn hp_status(&self, side: usize, slot: usize) -> HpStatus {
        let mon = &self.sides[side].pokemon[slot];
        HpStatus { hp: mon.hp, maxhp: mon.maxhp, status: status_token(mon.status) }
    }

    /// The `switch`/`drag` Details string (`Pokemon.details`): the species display
    /// name, then `, <Gender>` when the mon has a real gender ('M'/'F'); L100 +
    /// genderless ('N'/none) are omitted by Showdown. Genderless capture teams show
    /// just the species (`|switch|p1a: Snorlax|Snorlax|524/524`); a gendered mon
    /// shows `|switch|p2a: Snorlax|Snorlax, M|461/461` (the bridge goldens). This is
    /// observation-only: the protocol/writeline goldens use genderless teams (so the
    /// output is unchanged there), and the e2e_fuzz gate compares SEED+STATE, not
    /// protocol lines.
    fn switch_details(&self, side: usize, slot: usize, dex: &Dex) -> String {
        let species = self.species_name(side, slot, dex);
        match self.sides[side].pokemon[slot].gender {
            Some('M') => format!("{species}, M"),
            Some('F') => format!("{species}, F"),
            _ => species,
        }
    }

    /// Emit the battle-init framing (once, at the top of a logged battle), in the
    /// sim's exact order (the golden's first ~14 lines). The switch-in ability
    /// lines (`|-ability|`/`|-weather|`/`|-unboost|`) are DEFERRED (Phase 2) — the
    /// leads' `|switch|` + the `|turn|1` marker ARE emitted here (Phase 1). The
    /// `tier`/`rule` strings are the gen3customgame Custom-Game values the capture
    /// records (the port targets that format).
    fn emit_framing(&mut self, dex: &Dex) {
        self.log.timestamp();
        self.log.gametype_singles();
        // `|player|p1|<name>|` / `|player|p2|<name>|`.
        let p1_name = self.sides[0].name.clone();
        let p2_name = self.sides[1].name.clone();
        self.log.player(0, &p1_name);
        self.log.player(1, &p2_name);
        self.log.gen(self.gen);
        self.log.tier("[Gen 3] Custom Game");
        self.log.rule("HP Percentage Mod: HP is shown in percentages");
        self.log.separator();
        self.log.timestamp();
        self.log.teamsize(0, self.sides[0].pokemon.len());
        self.log.teamsize(1, self.sides[1].pokemon.len());
        self.log.start();
        // The two leads' switch-in lines (p1 then p2 — the start action's side order).
        for side in 0..2 {
            let mon = self.active_ref(side, dex);
            let details = self.switch_details(side, self.sides[side].active, dex);
            let hp = self.hp_status(side, self.sides[side].active);
            self.log.switch(&mon, &details, &hp);
        }
        // The switch-in ability lines (Intimidate `-ability`/`-unboost`, weather-setter
        // `-weather` SET) — Phase 2. RECONSTRUCTED from the already-computed post-switch-in
        // state (observation-only; `run_start_switchins` already applied the effects during
        // construction, before logging was enabled). Emitted in the SAME faster-first lead
        // order `run_start_switchins` fires them (a tie keeps side order), so the line order
        // matches the sim (verified vs the golden: Salamence Intimidate before Tyranitar Sand
        // Stream when Salamence is faster).
        self.emit_switchin_ability_lines(dex);
        self.log.turn(1);
    }

    /// Reconstruct + emit the leads' switch-in ability lines (Phase 2). Mirrors
    /// [`crate::state::BattleState::run_start_switchins`]: order the two leads faster-Speed
    /// first (a tie keeps side order), then per lead emit — for **Intimidate** an
    /// `|-ability|<lead>|Intimidate|boost` + (if the foe's Atk drop was NOT blocked by
    /// Clear Body / White Smoke / Hyper Cutter) an `|-unboost|<foe>|atk|1`; for a
    /// **weather** ability (Sand Stream / Drizzle / Drought) whose resulting weather is
    /// STILL the current `field.weather` (the winning setter) an `|-weather|<Weather>|[from]
    /// ability: <AbilityName>|[of] <lead>` SET line. DRAW-FREE (a formatting read of state
    /// the construction already resolved).
    fn emit_switchin_ability_lines(&mut self, dex: &Dex) {
        if !self.logging() {
            return;
        }
        // The two leads with raw Speed (stats[5]); faster first, tie = side order
        // (mirrors `run_start_switchins`'s stable `sort_by(|a,b| b.spe.cmp(&a.spe))`).
        let mut leads: Vec<(usize, usize, u16)> = (0..self.sides.len())
            .map(|side| {
                let slot = self.sides[side].active;
                (side, slot, self.sides[side].pokemon[slot].stats[5])
            })
            .collect();
        leads.sort_by(|a, b| b.2.cmp(&a.2));

        for (side, slot, _spe) in leads {
            // The lead's SET ability (== its current ability for everything but a
            // Trace lead, whose current ability is already the copy the construction
            // applied). The winning-setter rule: emit the `-weather` SET line ONLY if
            // this lead's weather is the CURRENT field weather (the slower of two
            // setters wins; only the survivor's line wasn't overwritten).
            let ability = to_id(&self.sides[side].pokemon[slot].set.ability);
            let weather_line = match ability.as_str() {
                "sandstream" => self.field.weather == Some(Weather::Sand),
                "drizzle" => self.field.weather == Some(Weather::Rain),
                "drought" => self.field.weather == Some(Weather::Sun),
                _ => false,
            };
            // The framing/lead reconstruction: a battle-start foe is never below −1, so an
            // Intimidate clamp never bites → the applied delta is always −1 (`None`).
            self.emit_ability_start_lines(side, slot, weather_line, None, dex);
        }
    }

    /// Emit the switch-in ability `Start` lines for ONE just-switched-in mon (Phase 2
    /// framing + Phase 3 mid-battle — `gen3_protocol_phase3_v1`). Keyed on the mon's
    /// SET ability (its pre-Start identity; a Trace holder's CURRENT ability is already
    /// the copy). `weather_line` gates the `-weather` SET emission (framing: the
    /// winning-setter rule; mid-battle: `run_switch`'s before≠after weather change — a
    /// permanent same-weather re-set emits nothing). All byte-verified vs the capture
    /// golden (midswitch_ability_lines / trace_switchin / flashfire_cycle):
    ///   - Intimidate → `|-ability|<mon>|Intimidate|boost` + `|-unboost|<foe>|atk|<applied>`;
    ///     a Clear Body / White Smoke / Hyper Cutter foe instead shows
    ///     `|-fail|<foe>|unboost|[from] ability: <Blocker>|[of] <foe>`; a SUBSTITUTE
    ///     foe shows ONLY the gen3 `|-hint|` (no `-ability` at all).
    ///   - Sand Stream / Drizzle / Drought → `|-weather|<W>|[from] ability: <A>|[of] <mon>`.
    ///   - Pressure → `|-ability|<mon>|Pressure|[silent]` (the addSplit secret line the
    ///     omniscient stream carries).
    ///   - Trace (copy applied) → `|-ability|<mon>|<Copied>|Trace|[from] ability: Trace|
    ///     [of] <foe>`.
    ///
    /// `intim_atk_pre` is the foe active's Atk STAGE just BEFORE this Intimidate's clamped
    /// drop was applied (threaded from the boost-apply site). The emit reconstructs from the
    /// POST-drop state (this fn runs AFTER the Start), so it CANNOT infer the applied delta
    /// from the post-drop stage alone — a foe at −6 is ambiguous (was it −5 dropped, or −6
    /// a no-op?). We emit the sim's CLAMPED-APPLIED delta `new_stage − pre_stage` ∈ {−1, 0}:
    /// a foe already at −6 drops by 0 → `|-unboost|<foe>|atk|0` (the line is STILL emitted,
    /// probe-verified — NOT omitted, NOT a `-fail`), a foe at −5 → `atk|1` landing −6.
    /// `None` = a reconstruction with no pre-stage available (the framing/lead path, where a
    /// battle-start foe is never below −1 so the clamp never bites → the applied delta is
    /// always −1). DRAW-FREE / observation-only: a formatting read of state the Start already
    /// resolved.
    fn emit_ability_start_lines(
        &mut self,
        side: usize,
        slot: usize,
        weather_line: bool,
        intim_atk_pre: Option<i8>,
        dex: &Dex,
    ) {
        if !self.logging() {
            return;
        }
        let ability = to_id(&self.sides[side].pokemon[slot].set.ability);
        match ability.as_str() {
            "intimidate" => {
                let foe = 1 - side;
                let foe_slot = self.sides[foe].active;
                if self.sides[foe].pokemon[foe_slot].substitute.is_some() {
                    // The gen3 mod's sub skip: NO `-ability`, just the hint.
                    self.log.hint(
                        "In Gen 3, Intimidate does not activate if every target has a Substitute.",
                    );
                    return;
                }
                let mon = self.mon_ref(side, slot, dex);
                self.log.ability(&mon, "Intimidate", Some("boost"));
                let foe_ability = to_id(&self.sides[foe].pokemon[foe_slot].ability);
                let foe_ref = self.mon_ref(foe, foe_slot, dex);
                if matches!(foe_ability.as_str(), "clearbody" | "whitesmoke" | "hypercutter") {
                    let display = dex
                        .ability(&foe_ability)
                        .map(|a| a.name.clone())
                        .unwrap_or(foe_ability);
                    self.log.fail_unboost_from_ability(&foe_ref, &display);
                } else {
                    // The CLAMPED-APPLIED delta `new_atk − pre_atk` ∈ {−1, 0}: a foe already
                    // at the −6 floor drops by 0 → `|-unboost|…|atk|0` (the sim STILL emits
                    // the line even at a 0 applied delta — probe
                    // `harness/probe_intimidate_floor.js`; a REQUESTED boost/drop always
                    // reports its clamped result, unlike a genuine no-op `boost()` call). So
                    // we route through `unboost_atk_applied` (which emits at 0 too), NOT the
                    // generic `boost()` (which skips a zero delta). `None` (the framing/lead
                    // reconstruction) can never floor → −1.
                    let new_atk = self.sides[foe].pokemon[foe_slot].boosts[0];
                    let applied = match intim_atk_pre {
                        Some(pre) => new_atk - pre,
                        None => -1,
                    };
                    self.log.unboost_atk_applied(&foe_ref, applied.unsigned_abs());
                }
            }
            "sandstream" | "drizzle" | "drought" => {
                if weather_line {
                    if let Some(w) = self.field.weather {
                        let mon = self.mon_ref(side, slot, dex);
                        let wname = weather_display(w);
                        let aname = self.sides[side].pokemon[slot].ability.clone();
                        self.log.weather(wname, Some(&Cause::Ability(aname)), Some(&mon), false);
                    }
                }
            }
            "pressure" => {
                let mon = self.mon_ref(side, slot, dex);
                self.log.ability_silent(&mon, "Pressure");
            }
            "trace" => {
                // Emit only if the copy actually applied (a fainted/absent foe → no copy).
                let cur = to_id(&self.sides[side].pokemon[slot].ability);
                if cur != "trace" {
                    let copied_display = dex
                        .ability(&cur)
                        .map(|a| a.name.clone())
                        .unwrap_or_else(|| self.sides[side].pokemon[slot].ability.clone());
                    let foe = 1 - side;
                    let foe_ref = self.mon_ref(foe, self.sides[foe].active, dex);
                    let mon = self.mon_ref(side, slot, dex);
                    self.log.ability_traced(&mon, &copied_display, &foe_ref);
                }
            }
            _ => {}
        }
    }

    /// Play a full scripted battle to game-end WITH protocol emission (level-2, Phase 1):
    /// enable the log, emit the battle-init framing (+ `|turn|1`), run [`run_full_battle`]
    /// (which emits the per-turn/`|move|`/`|-damage|`/`|switch|`/`|faint|`/… lines at its
    /// hook points), and return the `(BattleOutcome, Vec<ProtocolLine>)` — the outcome plus
    /// the accumulated OMNISCIENT stream.
    ///
    /// **Observation-only:** this is the SAME engine path as `run_full_battle`; the ONLY
    /// difference is `self.log` is enabled, and every emit hook is a PRNG-free read of
    /// already-computed state. So the seed sequence (hence every seed assertion in the
    /// existing suite) is UNCHANGED — the byte layer rides on top of the bit-for-bit engine.
    ///
    /// The battle must be constructed via [`crate::state::BattleState::start_with_switchins`]
    /// (the framing emits the leads' `|switch|`, matching the sim's `>start` sequence).
    pub fn run_full_battle_logged(
        &mut self,
        script: &[ScriptDecision],
        dex: &Dex,
    ) -> (BattleOutcome, Vec<ProtocolLine>) {
        self.log.enable();
        self.emit_framing(dex);
        let outcome = self.run_full_battle(script, dex);
        (outcome, self.log.drain())
    }
}

// ===========================================================================
// SWITCHING + post-faint replacement + win/loss — the full battle-to-game-end
// driver (`BattleState::run_full_battle`). See the `SWITCHING` doc block below.
// ===========================================================================

/// A per-side, per-request CHOICE: use a move slot, or switch to a benched team
/// slot. This is the unit of the [`BattleScript`] the full-battle driver consumes
/// (a `move K` / `switch N` choice, exactly the two the omniscient oracle submits).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Choice {
    /// Use the active mon's 0-based move slot (`move K` ⇒ `Move(K-1)`).
    Move(usize),
    /// Switch the active mon out for the 0-based team slot (`switch N` ⇒
    /// `Switch(N-1)`). The target must be a non-active, non-fainted bench mon.
    Switch(usize),
}

/// One decision boundary's scripted choices — mirroring exactly what the oracle
/// writes to the stream at each `>p1 …` / `>p2 …` boundary. A side is `None` when
/// it has no choice THIS request (the off-side of a single-mon forced replacement,
/// or a side whose active already committed its turn-choice). The driver pulls one
/// [`ScriptDecision`] per request boundary, in order — a regular `move` request
/// consumes both sides' choices, a forced-`switch` request consumes only the
/// flagged side(s) (matching the sim's `forceSwitch` table).
#[derive(Debug, Clone, Copy, Default)]
pub struct ScriptDecision {
    pub p1: Option<Choice>,
    pub p2: Option<Choice>,
}

impl ScriptDecision {
    /// A regular turn: both sides choose.
    pub fn both(p1: Choice, p2: Choice) -> ScriptDecision {
        ScriptDecision { p1: Some(p1), p2: Some(p2) }
    }
    /// Set one side's choice in place (the per-side pending-choice accumulator —
    /// `run_full_battle`'s mirror of the sim's per-side `side.choose` acceptance).
    pub fn set_side(&mut self, side: usize, c: Choice) {
        if side == 0 {
            self.p1 = Some(c);
        } else {
            self.p2 = Some(c);
        }
    }

    /// A single-side forced replacement (the other side has no choice).
    pub fn one(side: usize, c: Choice) -> ScriptDecision {
        if side == 0 {
            ScriptDecision { p1: Some(c), p2: None }
        } else {
            ScriptDecision { p1: None, p2: Some(c) }
        }
    }
    fn for_side(&self, side: usize) -> Option<Choice> {
        if side == 0 { self.p1 } else { self.p2 }
    }
}

/// The kind of request the engine is currently waiting on — surfaced per decision
/// in [`DecisionRecord`] so a test can assert the move-vs-forceSwitch boundary
/// EXACTLY as the sim's `battle.requestState` reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestKind {
    /// A normal turn (`requestState === 'move'`): both sides choose.
    Move,
    /// A forced post-faint replacement (`requestState === 'switch'`): the
    /// `force` bools say which side(s) must replace.
    ForceSwitch { force: [bool; 2] },
}

/// A per-decision-boundary record — the STATE the full-battle differential asserts
/// at EACH request boundary (a turn OR a forced-replacement sub-step). Carries the
/// post-decision per-side active snapshot, the running PRNG seed AFTER the decision
/// committed (the draw-order proof), the first mover (for a move turn), and the
/// request kind that was just answered.
#[derive(Debug, Clone)]
pub struct DecisionRecord {
    /// The request this decision answered (move vs forced-switch).
    pub request: RequestKind,
    /// Post-decision active snapshot, per side `[p1, p2]`.
    pub active: [MonSnapshot; 2],
    /// Post-decision active-mon species id, per side `[p1, p2]` — so a switch's
    /// "which mon is now active" is verifiable (proves the array-swap correctness).
    pub active_species: [String; 2],
    /// Per-side `pokemon_left` after this decision.
    pub pokemon_left: [usize; 2],
    /// Per-side **Spikes** layer count after this decision (`side.spikes`, 0..=3) — the
    /// SIDE-CONDITION state the spikes differential asserts. A side condition (not a mon
    /// volatile), so it is reported per side regardless of which mon is active.
    pub spikes: [u8; 2],
    /// The PRNG seed AFTER the decision committed (== the sim's `seed_after`).
    pub seed_after: PrngSeed,
    /// On a move turn, the side (0/1) whose action resolved first; `None` for a
    /// forced-switch sub-step (no move ordering) or if neither side acted.
    pub first_mover: Option<usize>,
    /// Whether an Explosion / Self-Destruct SELF-KO fired during the turn this boundary
    /// closes (the user fainted as part of the move). A coverage/diagnostic signal ONLY —
    /// the self-KO is applied via the normal faint machinery, so it does not affect any
    /// asserted state or seed. Lets the e2e capstone COUNT explosion decisions (the mechanic
    /// leaves no persistent board state, unlike a substitute).
    pub explosion_self_ko: bool,
    /// Whether a PHAZE (Roar / Whirlwind) drag FIRED during the turn this boundary closes (the
    /// `sample` ran + a foe mon was dragged in). A coverage/diagnostic signal ONLY — the drag is
    /// applied via the normal switch machinery, so it does not affect any asserted state or seed.
    /// Lets the e2e capstone COUNT phaze-drag decisions (a Protect-blocked / no-bench phaze does
    /// NOT set it, so this counts only drags that genuinely exercised the `sample` path).
    pub phaze_drag: bool,
    /// Per-side TRAPPED flag at this boundary (`gen3_trapping_v1`): whether each side's
    /// active mon is switch-trapped by the FOE active's Arena Trap / Magnet Pull —
    /// [`crate::turn::BattleState::is_trapped`], the port's equivalent of the sim's
    /// `pokemon.trapped` truthiness (the sim caches it at endTurn; the port computes it
    /// live, which is identical at every MOVE-request boundary since nothing changes
    /// between endTurn and the request). MEANINGFUL only on a `RequestKind::Move`
    /// boundary — at a mid-turn forced-switch boundary the sim's flag is stale (computed
    /// at the previous endTurn, possibly for the now-fainted mon), so the differentials
    /// only assert it at move boundaries.
    pub trapped: [bool; 2],
}

/// The outcome of a full scripted battle: the winner (if any) + every decision's
/// record. `winner == None` while the script ran out before game-end OR on a gen-3
/// double-faint TIE (distinguished by `ended`).
#[derive(Debug, Clone)]
pub struct BattleOutcome {
    /// `Some(side)` when a side won (its foe's `pokemon_left` hit 0); `None` on a
    /// tie (both 0) or if the script ended before game-end.
    pub winner: Option<usize>,
    /// Whether the battle reached game-end (a side out of mons). On a tie this is
    /// true with `winner == None`.
    pub ended: bool,
    /// Per-decision-boundary records, in order.
    pub decisions: Vec<DecisionRecord>,
}

/// A queued action inside the full-turn loop. Mirrors Showdown's `BattleQueue`
/// action `choice` + the `order` field [`comparePriority`] sorts on.
#[derive(Debug, Clone, Copy)]
enum QAction {
    /// `eachEvent('BeforeTurn')` action (order 4): one trailing-Update tail.
    BeforeTurn,
    /// A voluntary `switch` (order 103) — `pokemon` (outgoing slot) switches to
    /// `target` (bench slot). Carries the OUTGOING mon's speed for the tie key.
    Switch { side: usize, target: usize },
    /// A forced post-faint `instaswitch` (order 3).
    InstaSwitch { side: usize, target: usize },
    /// The deferred `runSwitch` step (order 101) for `side`'s entering mon: fires
    /// the entrant's switch-in ability `Start` (Intimidate/weather), draw-free.
    RunSwitch { side: usize },
    /// A `move` action (order 200). Keyed by the actor's STABLE `uid` (not array
    /// slot), so the action follows the mon across a `switchIn` array swap
    /// (mirroring Showdown's `action.pokemon` object reference). `move_index` is the
    /// 0-based move slot. `struggle` is set (`gen3_pp_tracking_v1`) when the mon had NO
    /// usable move at choice-commit time (all slots 0 PP) so `side.choose` substituted
    /// `moveid:'struggle'` — the dispatch runs Struggle, ignoring `move_index`.
    Move { side: usize, uid: usize, move_index: usize, struggle: bool },
    /// The end-of-turn `residual` action (order 300).
    Residual,
}

impl QAction {
    /// The `order` field (`battle-queue.ts` orders map) — smaller resolves FIRST.
    fn order(&self) -> u64 {
        match self {
            QAction::InstaSwitch { .. } => 3,
            QAction::BeforeTurn => 4,
            QAction::RunSwitch { .. } => 101,
            QAction::Switch { .. } => 103,
            QAction::Move { .. } => 200,
            QAction::Residual => 300,
        }
    }
}

impl BattleState {
    /// Play a FULL scripted battle to game-end (or until the `script` is
    /// exhausted), handling voluntary switches, post-faint replacements (single
    /// AND double), and win/loss — consuming the PRNG in Showdown's EXACT order +
    /// count across every turn AND replacement sub-step.
    ///
    /// # SWITCHING — the verified draw model
    ///
    /// The driver builds the turn's action queue and runs it through a turnLoop
    /// that can PAUSE for a forced replacement and RESUME the saved tail (mirroring
    /// `commitChoices`/`turnLoop`). The draw sites, all the existing
    /// [`speed_sort`]/`random_range`:
    /// - **action-order shuffle** ([`speed_sort`] over the turn's actions, in
    ///   `commitChoices`): a two-switch / switch-vs-move ordering. Switches (order
    ///   103) sort before moves (200) by `order`; two same-kind switches tie on
    ///   order+priority → the speed-tie shuffle draws.
    /// - **per-action trailing `eachEvent('Update')`** ([`each_event_shuffle`]) at
    ///   the END of every runAction — UNLESS the runAction paused for a switch
    ///   request (`makeRequest('switch'); return`) or its next queued action is an
    ///   `instaswitch` (`battle.ts:2372`); both SKIP the trailing Update.
    /// - **gen3 runSwitch is DRAW-FREE** (the gen4 override: no `speedSort(allActive)`,
    ///   no `fieldEvent('SwitchIn')`) — only the entrant's ability `Start` fires
    ///   (draw-free for our abilities). Do NOT add a SwitchIn tie-shuffle.
    /// - **double-replacement `insertChoice` splice** (`battle-queue.ts:283`): when
    ///   the 2nd instaswitch's `switchIn` enqueues its `runSwitch` into a queue
    ///   already holding the 1st instaswitch's `runSwitch` (same order 101), an
    ///   order-tie window draws ONE `random(firstIndex, lastIndex+1)`. A SINGLE
    ///   replacement's runSwitch inserts with no tie window → no draw.
    /// - **Quick Claw** `randomChance(1,5)` at `endTurn` — UNLESS the turn ended on
    ///   a faint pause OR a game-ending faint (no trailing residual/Quick Claw on
    ///   the deciding turn).
    ///
    /// # Win / loss
    ///
    /// `checkWin` runs inside the faint protocol: a side with `pokemon_left == 0`
    /// loses, its foe wins; both 0 → a gen-3 TIE (`win(None)`). The deciding faint
    /// draws no trailing Quick Claw.
    ///
    /// # Script contract
    ///
    /// One [`ScriptDecision`] per REQUEST boundary, in order: a `move` request
    /// consumes the next decision's both-side choices; a forced-`switch` request
    /// consumes the next decision's flagged-side choice(s). The driver stops when
    /// the battle ends OR the script is exhausted (a missing required choice ends
    /// the run, recorded but not asserted as game-end).
    pub fn run_full_battle(&mut self, script: &[ScriptDecision], dex: &Dex) -> BattleOutcome {
        let mut decisions: Vec<DecisionRecord> = Vec::new();
        let mut script_iter = script.iter();

        // PER-SIDE pending choices at the open `move` request (Phase 3 — the sim's
        // `side.choose` is PER-SIDE: one side's valid choice is ACCEPTED and held while
        // the other side's invalid choice is rejected, and a later re-submission by the
        // already-chosen side is itself rejected ("You already made choices") — so the
        // turn can commit with choices accepted at DIFFERENT capture decisions. The old
        // whole-decision skip mis-mapped that split (midswitch_ability_lines/2: p2's
        // `move 2` accepted at the decision whose p1 `switch` was rejected; the turn ran
        // with p2's HELD choice + p1's NEXT one). Zero-draw / observation-only: only the
        // boundary MAPPING changes; every pre-Phase-3 golden script has no split-accept
        // (their rejects came with the other side empty), so their replay is unchanged.
        let mut pending = ScriptDecision::default();

        // Whether the NEXT turn was already opened EAGERLY at the previous turn's end
        // (the sim's `makeRequest('move')` emits `|turn|N+1` in the COMPLETING write's
        // flush — `gen3_writeline_stream_v1`): the increment + marker then must not
        // re-fire at the next decision pull (which still emits the separator + `|t:|`,
        // the next chunk's opening bytes). The CONCATENATED stream is byte-identical
        // either way (the protocol gate pins it); eager emission only moves the marker
        // into the right per-write chunk for the `BattleStream::write_line` surface.
        let mut turn_already_opened = false;

        loop {
            // Pull the next `move`-request decision (both sides choose).
            let dec = match script_iter.next() {
                Some(d) => *d,
                None => break, // script exhausted at a turn boundary
            };

            // [side.choose validation — the sim's request-boundary reject-and-re-request]
            // A `move` request commits ONLY when every choosing side submits a VALID
            // choice; the sim's `side.choose` rejects a `move K` whose slot the active
            // mon doesn't have ("Your <mon> doesn't have a move K"), drawing NOTHING and
            // leaving `requestState === 'move'`, so the boundary STAYS OPEN for the next
            // submission. The omniscient capture, submitting from a stale per-turn plan,
            // can therefore record a PHANTOM zero-draw `move` decision right after a
            // replacement changed the active mon to one with fewer moves (e.g. a 3-move
            // Tyranitar → a 2-move Snorlax, then a scripted `move 3`): its `seedAfter`
            // equals the prior boundary's (nothing ran). We mirror the sim EXACTLY —
            // SKIP an invalid `move` decision (run no turn, emit nothing, draw nothing,
            // record nothing) and re-pull the next decision for the SAME boundary. A
            // forced-`switch` request is handled inside the turnLoop below (this gate is
            // only for a top-of-turn `move` request); a valid script never trips it, so
            // the fullbattle / secondary / e2e goldens are unaffected. VERIFIED vs the sim
            // (`harness/probe_forced_replacement_resume.js`): the rejected decision is
            // zero-draw — an observation-only decision-boundary MAPPING fix, not a seed
            // change.
            // PER-SIDE acceptance (the sim's `side.choose`, Phase 3): a side that already
            // holds an accepted pending choice DISCARDS a re-submission ("You already made
            // choices"); a side with an ILLEGAL fresh choice is rejected (draw-free); a
            // legal fresh choice is HELD. The turn commits only when BOTH sides hold one.
            for side in 0..2 {
                if pending.for_side(side).is_none() {
                    if let Some(c) = dec.for_side(side) {
                        if self.choice_is_legal(side, c, dex) {
                            pending.set_side(side, c);
                        }
                    }
                }
            }
            if pending.p1.is_none() || pending.p2.is_none() {
                // The boundary stays OPEN (any one-side HELD choice included); re-pull the
                // next decision for the SAME turn. Draw-free / zero-state / zero-EMIT (a
                // rejected/half boundary flushes NOTHING — the sim's separator + `|t:|`
                // are part of the turn-RUN batch, not the request; probe-verified per-write
                // attribution, `gen3_writeline_stream_v1`).
                continue;
            }
            let dec = pending;
            pending = ScriptDecision::default();

            // The turn COMMITS now. [makeRequest → commitChoices framing]: increment the
            // turn (unless the previous turn's end already opened it EAGERLY — the
            // `|turn|N` marker then already emitted in the completing chunk) + emit the
            // batch separator + `|t:|` that OPEN this turn's run batch. The concatenated
            // order (`…|upkeep`, `|turn|N`, `|`, `|t:|`, `|move|…`) is byte-identical to
            // the pre-Phase-3 flow (the protocol golden pins it); only the per-WRITE chunk
            // attribution moved (the separator/timestamp belong to the completing write).
            if !turn_already_opened {
                // Only the FIRST turn reaches here un-opened (turn 0 → 1, no marker —
                // `|turn|1` is in the framing); every later turn was opened EAGERLY at
                // the previous turn's end, so no reset is needed (the flag is always
                // true again before the next commit reads it).
                self.turn += 1;
                if self.logging() && self.turn >= 2 {
                    self.log.turn(self.turn);
                }
            }
            if self.logging() {
                self.log.separator();
                self.log.timestamp();
            }

            // Clear the per-turn Explosion self-KO diagnostic flag; `run_move` re-sets it if a
            // selfdestruct move self-KOs this turn (read into every boundary record of this turn).
            self.pending_explosion_self_ko = false;
            // Clear the per-turn PHAZE-drag diagnostic flag; `drag_in` re-sets it if a Roar /
            // Whirlwind drag fires this turn (read into every boundary record of this turn).
            self.pending_phaze_drag = false;
            // Expire the `duration:1` FLINCH volatiles from the previous turn (see
            // `run_turn`). DRAW-FREE.
            self.clear_flinch();
            // `commitChoices` refreshes the cached `pokemon.speed` at turn start
            // (`battle.js:2494` `this.updateSpeed()`) — para/boost-aware — BEFORE the
            // action-order sort + the per-action `eachEvent` shuffles read it.
            self.update_speed(dex);

            // --- Build + sort the turn's action queue. ---
            let mut actions: Vec<QAction> = Vec::new();
            for side in 0..2 {
                let active = self.sides[side].active;
                let uid = self.sides[side].pokemon[active].uid;
                match dec.for_side(side) {
                    Some(Choice::Move(mi)) => {
                        // `gen3_pp_tracking_v1`: a mon with NO usable move (all slots at 0
                        // PP) has `moveid:'struggle'` substituted by `side.choose` at
                        // choice-commit time — regardless of the scripted slot `mi`. This
                        // mirrors the sim: PP is exhausted on a PRIOR turn (a move deducts
                        // its own PP as it runs), so the Struggle substitution is decided
                        // from the mon's CURRENT PP state at the TOP of this turn. (A
                        // `Move(mi)` on a 0-PP slot while ANOTHER slot still has PP is
                        // REJECTED earlier by `move_decision_is_legal`, mirroring the sim's
                        // "doesn't have PP" reject — so it never reaches here.)
                        let struggle = self.sides[side].pokemon[active].must_struggle(dex);
                        actions.push(QAction::Move { side, uid, move_index: mi, struggle });
                    }
                    Some(Choice::Switch(target)) => {
                        actions.push(QAction::Switch { side, target });
                    }
                    None => {} // no choice this side (not expected on a move request)
                }
            }
            // [commitChoices] ACTION-ORDER sort (the tie shuffle on an
            // order+priority+speed tie — two same-kind switches / two equal moves).
            self.sort_actions(&mut actions, dex);

            // First mover = the first move/switch action (post action-order sort).
            let first_mover = actions.iter().find_map(|a| match a {
                QAction::Move { side, .. } => Some(*side),
                QAction::Switch { side, .. } => Some(*side),
                _ => None,
            });

            // The turn's queue: [beforeTurn, <sorted actions>, residual].
            let mut queue: Vec<QAction> = Vec::with_capacity(actions.len() + 2);
            queue.push(QAction::BeforeTurn);
            queue.extend(actions);
            queue.push(QAction::Residual);

            // --- [turnLoop] run the queue, pausing for forced replacements. ---
            let mut request = RequestKind::Move; // the request THIS boundary answers
            let mut script_exhausted = false;
            let stop = loop {
                match self.turn_loop(&mut queue, dex) {
                    TurnLoopStop::Ended { winner } => break Some(TurnEnd::Ended(winner)),
                    TurnLoopStop::Done => {
                        // endTurn: the `runEvent('DisableMove')` handler-sort shuffle (a
                        // taunt+disable mon draws one size-2 shuffle here, `gen3_taunt_disable_v1`)
                        // fires BEFORE the unconditional Quick Claw roll (no faint pause).
                        self.disable_move_event_shuffle();
                        // `activeTurns++` per active (battle.ts:1762) — DRAW-FREE, feeds the
                        // Speed Boost residual gate (`gen3_ability_batch1_v1`).
                        self.bump_active_turns();
                        let _ = self.prng.random_chance(1, 5);
                        break Some(TurnEnd::Done);
                    }
                    TurnLoopStop::NeedSwitch { force } => {
                        // makeRequest('switch') paused the turn. Record the boundary
                        // we JUST finished (the move request, or the prior forced
                        // switch) at the PAUSE seed, then take the next replacement.
                        decisions.push(DecisionRecord {
                            request,
                            active: [self.active_snapshot(0), self.active_snapshot(1)],
                            active_species: [self.active_species_id(0), self.active_species_id(1)],
                            pokemon_left: [self.sides[0].pokemon_left, self.sides[1].pokemon_left],
                            spikes: [self.sides[0].spikes, self.sides[1].spikes],
                            seed_after: self.prng.get_seed(),
                            first_mover: if matches!(request, RequestKind::Move) {
                                first_mover
                            } else {
                                None
                            },
                            explosion_self_ko: self.pending_explosion_self_ko,
                            phaze_drag: self.pending_phaze_drag,
                            trapped: [self.is_trapped(0, dex), self.is_trapped(1, dex)],
                        });

                        // Pull the replacement decision(s); commit the flagged sides'
                        // instaswitch(es); prepend them before the saved tail. PER-SIDE
                        // accumulation (Phase 3, mirroring the top-of-turn accumulator +
                        // the sim's per-side `side.choose`): a DOUBLE replacement may
                        // arrive as ONE decision carrying both (the capture's DEC grammar
                        // — every pre-Phase-3 golden) OR as one-sided decisions across
                        // writes (the `write_line` streaming surface's `>p1 switch 3` /
                        // `>p2 switch 2`). An INVALID forced target (fainted / out-of-
                        // range / the active) or a non-switch token for a forced side is
                        // REJECTED draw-free and the boundary stays open (the sim's
                        // "can't switch to a fainted Pokémon"); a token for a NON-forced
                        // side is DISCARDED (the sim rejects choices from a side with
                        // nothing to choose).
                        let mut have: [Option<usize>; 2] = [None, None];
                        let satisfied = |have: &[Option<usize>; 2]| {
                            (0..2).all(|s| !force[s] || have[s].is_some())
                        };
                        while !satisfied(&have) {
                            let rep_dec = match script_iter.next() {
                                Some(d) => *d,
                                None => {
                                    script_exhausted = true;
                                    break;
                                }
                            };
                            for side in 0..2 {
                                if force[side] && have[side].is_none() {
                                    if let Some(Choice::Switch(t)) = rep_dec.for_side(side) {
                                        let sd = &self.sides[side];
                                        if t < sd.pokemon.len()
                                            && !sd.pokemon[t].fainted
                                            && t != sd.active
                                        {
                                            have[side] = Some(t);
                                        }
                                    }
                                }
                            }
                        }
                        if !satisfied(&have) {
                            break None; // script exhausted mid-replacement
                        }
                        let mut insta: Vec<QAction> = Vec::new();
                        for side in 0..2 {
                            if force[side] {
                                insta.push(QAction::InstaSwitch { side, target: have[side].expect("satisfied") });
                                self.sides[side].switch_flag = false;
                            }
                        }
                        // [commitChoices on a switch request] `commitChoices()` runs
                        // `this.updateSpeed()` at its TOP (battle.ts:3020) on EVERY choice
                        // commit — INCLUDING a mid-turn FORCED-replacement submit. So the
                        // cached `pokemon.speed` of BOTH actives is refreshed para/boost-
                        // aware HERE, before the instaswitch sorts + the resumed tail's
                        // `eachEvent('Update')` tie-shuffles read it. CRUX (the e2e-capstone
                        // forced-replacement fix): a foe mon paralyzed mid-turn keeps its
                        // STALE turn-start speed through the move phase, but the replacement
                        // commit's `updateSpeed` drops it to its para speed — so the
                        // resumed-turn trailing-Update shuffle no longer spuriously TIES the
                        // (now-para-slowed) foe with the fresh entrant (verified vs the sim:
                        // a Jirachi mirror where the foe Jirachi was para'd mid-turn must
                        // read 53, not its stale 212, during the post-replacement Updates —
                        // else 2 phantom tie-shuffle draws desync the seed).
                        self.update_speed(dex);
                        // [EMIT] open the REPLACEMENT batch: `|` (separator) + `|t:|`,
                        // right before the instaswitch(es) run (their `|switch|` line emits
                        // in `execute_switch`). Every forced replacement is its own batch,
                        // matching the golden (`…|faint| / | / |t:| / |switch|…`).
                        if self.logging() {
                            self.log.separator();
                            self.log.timestamp();
                        }
                        // sort ONLY the new instaswitch(es) (the double-replacement tie
                        // shuffle), then PREPEND before the saved remainder (oldQueue).
                        self.sort_actions(&mut insta, dex);
                        let old_queue = std::mem::take(&mut queue);
                        queue = insta;
                        queue.extend(old_queue);

                        // The NEXT boundary we record answers THIS forced switch.
                        request = RequestKind::ForceSwitch { force };
                    }
                }
            };

            // --- Record the final boundary of this turn (move request, or the last
            //     forced-switch request) at the post-turn seed, then handle end. ---
            match stop {
                Some(TurnEnd::Done) => {
                    decisions.push(self.boundary_record(request, first_mover, dex));
                    // [EMIT] the NEXT turn's `|turn|N+1` marker EAGERLY — the sim's
                    // `makeRequest('move')` flushes it in the COMPLETING write's chunk
                    // (`gen3_writeline_stream_v1`; probe: the real BattleStream's turn
                    // chunk ends `…|upkeep, |turn|N+1`). The next decision pull skips the
                    // increment/marker (`turn_already_opened`) and emits only the
                    // separator + `|t:|` — the next chunk's opening bytes. Concatenated
                    // order is UNCHANGED (the protocol golden pins it byte-for-byte).
                    self.turn += 1;
                    if self.logging() && self.turn >= 2 {
                        self.log.turn(self.turn);
                    }
                    turn_already_opened = true;
                }
                Some(TurnEnd::Ended(winner)) => {
                    // [EMIT] the deciding `|` separator then `|win|<PlayerName>` (or `|tie`).
                    // The deciding faint's `|faint|` line was already emitted; the game-end
                    // line follows a bare separator (no residual/upkeep on a deciding faint —
                    // `turn_loop` returned before the Residual action). Observation-only.
                    if self.logging() {
                        self.log.separator();
                        match winner {
                            Some(side) => {
                                let name = self.sides[side].name.clone();
                                self.log.win(&name);
                            }
                            None => self.log.tie(),
                        }
                    }
                    decisions.push(self.boundary_record(request, first_mover, dex));
                    return BattleOutcome { winner, ended: true, decisions };
                }
                None => {
                    // Script exhausted mid-turn (a required replacement missing). The
                    // pause boundary was already recorded; stop here, not game-end.
                    let _ = script_exhausted;
                    return BattleOutcome { winner: None, ended: false, decisions };
                }
            }
        }

        BattleOutcome { winner: None, ended: false, decisions }
    }

    /// Whether ONE side's top-of-turn `move`-request choice would be ACCEPTED by the
    /// sim's PER-SIDE `side.choose` (Phase 3 refactor of the old whole-decision
    /// `move_decision_is_legal` — same gates, per-side granularity): a `Move(K)` must
    /// name a slot the CURRENT active mon has ("Your <mon> doesn't have a move K") that
    /// is usable-or-Struggle-substituted; a VOLUNTARY `Switch(t)` must name a live,
    /// non-active bench slot and the active must not be TRAPPED (`gen3_trapping_v1`).
    /// A rejected choice draws nothing and leaves the side's half of the boundary open —
    /// see the per-side pending-choice accumulator at the top of [`run_full_battle`].
    fn choice_is_legal(&self, side: usize, choice: Choice, dex: &Dex) -> bool {
        {
            match Some(choice) {
                Some(Choice::Move(mi)) => {
                    let active = self.sides[side].active;
                    let mon = &self.sides[side].pokemon[active];
                    // (1) OUT-OF-RANGE slot (the forced-replacement resume phantom): a
                    // `Move(K)` whose slot the current active mon doesn't have.
                    if mi >= mon.set.moves.len() {
                        return false;
                    }
                    // (2) UN-USABLE slot WHILE another move is usable (`gen3_pp_tracking_v1` +
                    // `gen3_taunt_disable_v1`): the sim's `side.choose` REJECTS a `move K` that is
                    // out of PP, DISABLED, or TAUNTED ("(move) is disabled") when the mon still has
                    // a usable move — draw-free, boundary stays open (VERIFIED vs the sim: a stale
                    // `move 1` on a 0-PP Earthquake while Crunch has PP drew 0 and did not advance;
                    // the same reject applies to a taunted Status move or a disabled move). This
                    // mirrors the out-of-range case. EXCEPTION: when the mon has NO usable move at
                    // all (`must_struggle`), the `move K` is NOT rejected — `side.choose`
                    // SUBSTITUTES `moveid:'struggle'` instead (handled at queue-build time), so the
                    // decision commits and runs Struggle. So reject ONLY when this specific slot is
                    // un-usable AND a different usable move exists. `move_usable` now folds in the
                    // Disable/Taunt selection restriction (dex read for Taunt's per-slot category).
                    if !mon.move_usable(mi, dex) && !mon.must_struggle(dex) {
                        return false;
                    }
                }
                Some(Choice::Switch(t)) => {
                    // (2b) INVALID TARGET (Phase 3 — the capture plans now submit blind
                    // `switch N` tokens, so a scripted switch to a FAINTED / out-of-range /
                    // already-active slot must be SKIPPED exactly like the sim's
                    // `chooseSwitch` reject ("can't switch to a fainted Pokémon" — draw-free,
                    // boundary stays open; the capture records the rejected decision with an
                    // UNCHANGED seedAfter). The older harness plans always computed a live
                    // bench target, so this gate never fired for the pre-Phase-3 goldens.
                    let s = &self.sides[side];
                    if t >= s.pokemon.len() || s.pokemon[t].fainted || t == s.active {
                        return false;
                    }
                    // (3) TRAPPED (`gen3_trapping_v1` — the SWITCH-legality gate, mirroring
                    // the move gate above): the sim's `chooseSwitch` at a `move` request
                    // REJECTS a voluntary switch by a trapped mon ("Can't switch: The active
                    // Pokémon is trapped") — DRAW-FREE, boundary stays open (VERIFIED vs the
                    // sim `harness/probe_trapping_rng.js`: the rejected `switch 2` left the
                    // seed byte-identical + the request open). This gates ONLY the
                    // top-of-turn `move`-request path — a FORCED replacement (`requestState
                    // === 'switch'`, the `NeedSwitch` branch) never consults `trapped`, and
                    // a PHAZE drag (`drag_in`) moves a trapped mon regardless. (The target's
                    // live-bench validity is enforced elsewhere, as before.)
                    if self.is_trapped(side, dex) {
                        return false;
                    }
                }
                None => {}
            }
        }
        true
    }

    fn boundary_record(
        &self,
        request: RequestKind,
        first_mover: Option<usize>,
        dex: &Dex,
    ) -> DecisionRecord {
        DecisionRecord {
            request,
            active: [self.active_snapshot(0), self.active_snapshot(1)],
            active_species: [self.active_species_id(0), self.active_species_id(1)],
            pokemon_left: [self.sides[0].pokemon_left, self.sides[1].pokemon_left],
            spikes: [self.sides[0].spikes, self.sides[1].spikes],
            seed_after: self.prng.get_seed(),
            first_mover: if matches!(request, RequestKind::Move) { first_mover } else { None },
            explosion_self_ko: self.pending_explosion_self_ko,
            phaze_drag: self.pending_phaze_drag,
            trapped: [self.is_trapped(0, dex), self.is_trapped(1, dex)],
        }
    }

    /// Active-mon snapshot (the per-decision STATE the differential asserts).
    fn active_snapshot(&self, side: usize) -> MonSnapshot {
        let a = self.sides[side].active;
        let mon = &self.sides[side].pokemon[a];
        MonSnapshot {
            hp: mon.hp,
            maxhp: mon.maxhp,
            fainted: mon.fainted,
            status: mon.status,
            boosts: mon.boosts,
            confusion: mon.confusion,
            protect_counter: mon.protect_counter,
            leech_seeded: mon.leech_seed.is_some(),
            substitute: mon.substitute,
            move_pp: mon.pp_array(),
            taunted: mon.taunt.is_some(),
            disabled_slot: mon.disable.map(|(k, _)| k as i8).unwrap_or(-1),
            item_held: !mon.item.is_empty(),
        }
    }

    /// The active mon's resolved species id (e.g. `tyranitar`) — proves a switch
    /// brought the right mon to the active slot (the array-swap correctness).
    fn active_species_id(&self, side: usize) -> String {
        let a = self.sides[side].active;
        self.sides[side].pokemon[a].species_id.clone()
    }
}

/// How a foe-targeting major-status MOVE fails when the foe is ALREADY statused — the
/// two gen-3 `setStatus` fail branches (pokemon.ts:1699-1706), which emit DIFFERENT
/// `|-fail|` lines. `Same` = the move's inflicted status matches the foe's current one
/// (`|-fail|<target>|<status>`); `Different` = a different status (`|-fail|<user>` + the
/// `[still]` move-announce form). Both are draw-free past the move's accuracy roll.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatusMoveFail {
    Same,
    Different,
}

/// How a single turn-loop pass ended.
enum TurnEnd {
    /// The turn completed (the endTurn Quick Claw drew).
    Done,
    /// The battle ended this turn (no Quick Claw on the deciding faint).
    Ended(Option<usize>),
}

/// Why [`BattleState::turn_loop`] returned — mirroring `turnLoop`'s exit
/// conditions (`if (this.requestState || this.ended) return`).
enum TurnLoopStop {
    /// The queue drained with no pending request and no game-end — ready for the
    /// endTurn Quick Claw.
    Done,
    /// A faint paused the turn for a forced replacement (`makeRequest('switch')`).
    /// `force[side]` marks which side(s) must replace.
    NeedSwitch { force: [bool; 2] },
    /// The battle ended (a side out of mons). `winner` is `None` on a tie.
    Ended { winner: Option<usize> },
}

impl BattleState {
    /// Sort a slice of queued actions by the full `comparePriority` key
    /// (`order → priority → speed → subOrder → effectOrder`, descending) via
    /// [`speed_sort`] — wiring the action-order Fisher-Yates tie-shuffle DRAW onto
    /// the switch path (two same-kind switches / two equal moves at a full tie).
    ///
    /// The handler `speed` is the OUTGOING (acting) mon's effective speed (the
    /// gen-3 `getActionSpeed`), so a faster mon's switch/move resolves first.
    /// `order` is the action's queue order (switch 103 before move 200, etc.).
    fn sort_actions(&mut self, actions: &mut Vec<QAction>, dex: &Dex) {
        if actions.len() < 2 {
            return; // speed_sort no-ops, but skip the alloc on the common 0/1 case
        }
        let mut handlers: Vec<EventHandler<QAction>> = actions
            .iter()
            .map(|a| {
                let (side, slot, priority): (usize, usize, i32) = match a {
                    QAction::Move { side, uid, move_index, struggle } => {
                        let slot = self.slot_of_uid(*side, *uid).unwrap_or(self.sides[*side].active);
                        // A forced Struggle has priority 0 (Struggle's dex priority), NOT the
                        // stale scripted slot's — read it from the `struggle` move, else the slot.
                        let pr = if *struggle {
                            0
                        } else {
                            self.move_priority(*side, slot, *move_index, dex) as i32
                        };
                        (*side, slot, pr)
                    }
                    QAction::Switch { side, .. } | QAction::InstaSwitch { side, .. } => {
                        // A switch action's `speed` key is the OUTGOING mon's speed
                        // (the active at sort time); priority 0.
                        (*side, self.sides[*side].active, 0)
                    }
                    // beforeTurn / runSwitch / residual never co-sort with these in
                    // our queue construction, but keep a total key for safety.
                    QAction::BeforeTurn | QAction::RunSwitch { .. } | QAction::Residual => {
                        (0, self.sides[0].active, 0)
                    }
                };
                let speed = self.effective_speed(side, slot, dex) as f64;
                EventHandler {
                    order: a.order(),
                    priority,
                    speed,
                    sub_order: 0,
                    effect_order: 0,
                    handler: *a,
                }
            })
            .collect();
        speed_sort(&mut handlers, &mut self.prng);
        *actions = handlers.into_iter().map(|h| h.handler).collect();
    }

    /// Run queued actions until the queue drains, a faint requests a replacement,
    /// or the battle ends — mirroring `turnLoop` + the per-`runAction` tail
    /// (faintMessages → checkWin → checkFainted → the switch-request gate → the
    /// gen<5 trailing `eachEvent('Update')`).
    ///
    /// On `NeedSwitch`/`Ended` it RETURNS with the queue's remaining tail intact
    /// (the saved `oldQueue` the caller resumes after committing replacements).
    fn turn_loop(&mut self, queue: &mut Vec<QAction>, dex: &Dex) -> TurnLoopStop {
        while !queue.is_empty() {
            let action = queue.remove(0);

            // --- Run the action body. A move/residual may faint a mon (HP zeroed,
            //     `fainted` set by process_faints which we call as faintMessages). ---
            match action {
                QAction::BeforeTurn => {
                    self.each_event_shuffle(); // eachEvent('BeforeTurn')
                }
                QAction::Move { side, uid, move_index, struggle } => {
                    // The `case 'move'` guard (battle.ts:2239-2240): a move whose actor
                    // is no longer the active (KO'd / switched out) `return false`s from
                    // runAction IMMEDIATELY — drawing NOTHING and running NO TAIL (no
                    // faintMessages, no trailing Update). So we `continue` the loop,
                    // skipping the entire tail. (This is the just-fainted mon's
                    // surviving-but-no-op queued move; its presence in the queue does
                    // NOT add a trailing-Update draw — verified vs the sim's
                    // double-replacement trace, where `move p2:Electrode` drew nothing
                    // and got no trailing Update.) The actor is resolved by its STABLE
                    // uid (its array slot may have swapped).
                    let live = self
                        .slot_of_uid(side, uid)
                        .map(|slot| !self.sides[side].pokemon[slot].fainted && self.sides[side].active == slot)
                        .unwrap_or(false);
                    if !live {
                        continue; // runAction returned false → no tail, no draws
                    }
                    let slot = self.slot_of_uid(side, uid).unwrap();
                    // `willAct()` (battle-queue.ts:310) — does ANY move/switch/instaswitch/
                    // shift action REMAIN in the queue? Protect/Detect's `onPrepareHit`
                    // requires it (`!!this.queue.willAct() && runEvent('StallMove')`): a
                    // Protect that resolves with NO pending foe action (the foe SWITCHED — a
                    // switch is order 103 < the protect's move order 200, so it already ran)
                    // FAILS (draw-free, no volatile). The residual (the only other queued
                    // action here) is NOT a move/switch, so this is true iff a foe move/
                    // switch is still pending. VERIFIED vs the sim: Protect vs a foe SWITCH
                    // leaves the protector with NO volatiles (`|move|…Protect||[still]`).
                    let will_act = queue.iter().any(|a| {
                        matches!(
                            a,
                            QAction::Move { .. } | QAction::Switch { .. } | QAction::InstaSwitch { .. }
                        )
                    });
                    // `willMove(target)` for Disable's duration `+1` (`gen3_taunt_disable_v1`):
                    // does the FOE (the disable target) still have a PENDING move/switch action in
                    // the queue? False iff the foe has already acted this turn (disabler moved 2nd)
                    // → Disable's onStart does `duration++`.
                    let foe = 1 - side;
                    let foe_will_move = queue.iter().any(|a| match a {
                        QAction::Move { side: s, .. }
                        | QAction::Switch { side: s, .. }
                        | QAction::InstaSwitch { side: s, .. } => *s == foe,
                        _ => false,
                    });
                    let res = self.run_move(MoveAction { side, slot, move_index, struggle }, will_act, foe_will_move, dex);
                    if res.landed {
                        let upd = self.each_event_shuffle(); // in-tryMoveHit Update (landed)
                        self.run_update_items(&upd, dex); // cure-berry/Leppa onUpdate (draw-free)
                    }
                    // --- forceSwitchFlag consumption (battle.ts:2348-2353): at the END of
                    //     the move's runAction (AFTER the move body / any in-tryMoveHit
                    //     Update, BEFORE faintMessages), a SUCCESSFUL phaze (Roar /
                    //     Whirlwind) DRAGS in a random foe (the `sample` draw + the forced
                    //     switch + the runSwitch enqueue). `drag_in` does exactly that. A
                    //     status move is never `landed`, so the order is: phaze accuracy
                    //     (drawn in run_move) → drag sample (here) → faintMessages → … →
                    //     the trailing Update, mirroring the source EXACTLY. ---
                    if let Some(foe) = res.force_switch_foe {
                        self.drag_in(foe, dex, &mut *queue);
                    }
                }
                QAction::Switch { side, target } | QAction::InstaSwitch { side, target } => {
                    self.execute_switch(side, target, false, dex, &mut *queue);
                }
                QAction::RunSwitch { side } => {
                    // gen3 runSwitch: EntryHazard (Spikes damage, draw-free) → ability Start.
                    // The ability Start is draw-free EXCEPT when a Sand Stream / Drizzle /
                    // Drought entrant CHANGES the weather: `field.setWeather` then fires
                    // `eachEvent('WeatherChange')` (field.ts:87) — a 2-active speedSort that
                    // draws ONE tie-shuffle iff the actives tie on cached speed. It fires
                    // INSIDE this runSwitch runAction (before the trailing Update), so do it
                    // here, ahead of the tail. A Spikes-KO here faints the entrant; the tail
                    // forces another replacement.
                    let weather_changed = self.run_switch(side, dex);
                    if weather_changed {
                        self.each_event_shuffle(); // eachEvent('WeatherChange') tie-shuffle
                    }
                }
                QAction::Residual => {
                    // [EMIT] the `|` separator that CLOSES the action block + OPENS the
                    // residual block (always emitted when a turn reaches its residual —
                    // even with no residual HP lines, matching the golden). Then the
                    // residual `|-heal|`/`|-damage|` lines emit inside `run_residuals`.
                    if self.logging() {
                        self.log.separator();
                    }
                    self.run_residuals(dex);
                    // [EMIT] `|upkeep` — the end-of-turn marker, emitted AFTER the residual
                    // lines. SKIPPED when a residual KO ended the battle (then
                    // `run_full_battle` emits `|win|`/`|tie|` instead); `check_win` detects
                    // that. The `|turn|N+1` marker is NOT emitted here — it lands at the TOP
                    // of the NEXT turn's outer-loop iteration (after any forced replacement
                    // this residual triggered), matching the sim's `makeRequest('move')`
                    // placement. A RESIDUAL faint's replacement `|switch|` must precede
                    // `|turn|N+1`; a MOVE faint's replacement (before the residual) already
                    // did — so deferring the marker to the next-turn top is correct for both.
                    if self.logging() && self.check_win().is_none() {
                        self.log.upkeep();
                    }
                }
            }

            // --- runAction tail (battle.ts:2357-2424). ---
            // faintMessages: set `fainted`, decrement pokemonLeft, then checkWin.
            let any_faint = self.process_faints(dex);
            if any_faint {
                // gen-3 singles cancelAction-all (battle.ts:2140-2148): "in gen 3,
                // fainting skips all moves and switches" — remove the REST of this
                // turn's queued move/switch actions whose actor is a NON-fainted
                // active. The just-fainted mon's own queued move is NOT cancelled
                // (it's excluded from getAllActive) but runs as a no-op (the
                // `actor.fainted` guard above). The residual survives.
                self.cancel_active_actions(queue);
            }
            if let Some(end) = self.check_win() {
                return TurnLoopStop::Ended { winner: end };
            }

            // checkFainted (gen3 replaces after EACH action): every 0-HP active gets
            // switchFlag. (Showdown gates this on `!peek || peek in [move,residual]`;
            // in our linear gen3-singles queue the equivalent is: any fainted active
            // that CAN switch needs a replacement — set the flag unconditionally and
            // let the gate below decide.)
            self.check_fainted();

            // `else if peek === 'instaswitch' return false` (battle.ts:2372): if the
            // NEXT queued action is an instaswitch, this runAction returns IMMEDIATELY
            // — BEFORE the switch-request gate AND before the trailing Update. This is
            // the double-replacement resume: the FIRST instaswitch's tail does NOT
            // re-request a switch (the SECOND instaswitch is already queued to run),
            // and it draws no trailing Update. Must precede the gate (battle.ts:2372
            // is `else if`, ahead of the gate at 2418).
            if matches!(queue.first(), Some(QAction::InstaSwitch { .. })) {
                continue;
            }

            // The switch-request gate: a side with a flagged active that can switch
            // pauses the turn (makeRequest('switch'); return) — SKIPPING the trailing
            // Update.
            let force = self.switch_request_gate();
            if force[0] || force[1] {
                return TurnLoopStop::NeedSwitch { force };
            }

            // The gen<5 trailing eachEvent('Update') (battle.ts:2424) — reached only
            // when the runAction did NOT pause / was not pre-empted by an instaswitch.
            let upd = self.each_event_shuffle();
            self.run_update_items(&upd, dex); // cure-berry/Leppa onUpdate (draw-free)
        }
        TurnLoopStop::Done
    }

    /// `checkWin` (battle.ts:2155): both sides out → gen-3 TIE (`Ended{winner:None}`);
    /// a side whose FOE is out → that side wins. Returns `Some(winner_or_tie)` when
    /// the battle ends, `None` otherwise. (Wrapped in an `Option<Option<usize>>` via
    /// the caller; here `Some(Some(side))` = win, `Some(None)` = tie.)
    fn check_win(&self) -> Option<Option<usize>> {
        if self.sides[0].pokemon_left == 0 && self.sides[1].pokemon_left == 0 {
            return Some(None); // gen-3: both out → tie (win(null))
        }
        for side in 0..2 {
            let foe = 1 - side;
            if self.sides[foe].pokemon_left == 0 {
                return Some(Some(side)); // foe out → `side` wins
            }
        }
        None
    }

    /// gen-3 singles cancelAction-all (`faintMessages`, battle.ts:2606-2616): on ANY
    /// faint, `for (const pokemon of this.getAllActive()) this.queue.cancelAction(pokemon)`
    /// — "in gen 3, fainting skips all moves AND SWITCHES". `cancelAction(pokemon)`
    /// (battle-queue.ts:329) removes EVERY queued action whose `action.pokemon === pokemon`,
    /// for each NON-fainted active (a fainted mon is excluded from `getAllActive`). So it
    /// drops a NON-fainted active's queued `move` / voluntary `switch` — AND its pending
    /// **`runSwitch`** (a runSwitch's `action.pokemon` is the entrant, which is that side's
    /// current active). The fainted mon's own queued move is LEFT (it's not in getAllActive)
    /// and runs as a no-op. residual / beforeTurn / instaswitch are not actor-keyed and survive.
    ///
    /// # The double-faint → cascade `runSwitch` cancellation (the e2e_9 / e2e_194 fix)
    ///
    /// The `runSwitch` cancellation is load-bearing for a DOUBLE-REPLACEMENT cascade. When a
    /// mutual double faint forces BOTH sides to replace, both fresh entrants enqueue an order-101
    /// `runSwitch` (the 2nd ties the 1st → the splice draw). If the FIRST runSwitch to run FAINTS
    /// its own entrant (its own side's Spikes KO on entry — the cascade), `faintMessages` fires
    /// `cancelAction` over `getAllActive` — which REMOVES the OTHER side's still-pending
    /// `runSwitch`. So the OTHER entrant's entry-hazard (Spikes) chip is NEVER applied: it stays
    /// at full HP (e2e_9: p1's Jolteon runSwitch KO's Jolteon first → p2's Jirachi runSwitch is
    /// cancelled → Jirachi keeps its full 403, NOT chipped to 353). VERIFIED bit-for-bit vs the
    /// omniscient sim (`harness/probe_cascade_hazard_order.js`): with the FAINTING side's runSwitch
    /// FIRST the foe entrant is UNCHIPPED; with the surviving side's runSwitch first the foe is
    /// chipped ONCE (its runSwitch already ran, so there is nothing to cancel). DRAW-FREE (a queue
    /// splice, no PRNG) — the SEED is untouched; only the mis-applied hazard STATE is fixed.
    fn cancel_active_actions(&mut self, queue: &mut Vec<QAction>) {
        // Borrow-checker: resolve the keep-decision into a Vec<bool> first.
        let keep: Vec<bool> = queue
            .iter()
            .map(|a| match a {
                QAction::Move { side, uid, .. } => {
                    // Keep only if the actor is a fainted mon (→ runs as a no-op) —
                    // a fainted mon is excluded from getAllActive so cancelAction does
                    // NOT touch it. A NON-fainted active's queued move IS cancelled.
                    match self.slot_of_uid(*side, *uid) {
                        Some(slot) => self.sides[*side].pokemon[slot].fainted,
                        None => false,
                    }
                }
                QAction::Switch { side, .. } => {
                    // A voluntary switch from a NON-fainted active is cancelled; a
                    // fainted active makes a forced instaswitch, not a queued switch.
                    self.sides[*side].pokemon[self.sides[*side].active].fainted
                }
                QAction::RunSwitch { side } => {
                    // `cancelAction(entrant)` removes the entrant's pending runSwitch when
                    // the entrant is a NON-fainted active (a getAllActive member). A runSwitch
                    // for an entrant that itself fainted (its own hazard KO'd it) is NOT in
                    // getAllActive → NOT cancelled (and would no-op anyway via run_switch's
                    // fainted guard). This is the cascade fix: drop the OTHER side's stale
                    // runSwitch so its hazard isn't re-applied after a cascade replacement.
                    self.sides[*side].pokemon[self.sides[*side].active].fainted
                }
                // Non-actor-keyed actions (instaswitch forced/post-cancel, residual,
                // beforeTurn) are not removed by cancelAction.
                _ => true,
            })
            .collect();
        let mut it = keep.into_iter();
        queue.retain(|_| it.next().unwrap_or(true));
    }

    /// `checkFainted` (battle.ts:2078): flag every fainted active for replacement
    /// (`switchFlag = true`) AND set its status to `fnt` (`pokemon.status = "fnt"`,
    /// battle.js:2082 — modeled as `None`). In gen-3 singles a fainted active is the
    /// only one.
    ///
    /// The status clear is LOAD-BEARING for the replacement sort
    /// (`gen3_fnt_clears_status_v1`, the A/B switch-boundary cluster): the
    /// instaswitch action's speed key is the OUTGOING mon's gen3 `getActionSpeed()`
    /// = `getStat('spe')`, whose para ×0.25 `onModifySpe` gates on
    /// `status === 'par'` — now `'fnt'`, so a fainted formerly-PARALYZED mon sorts
    /// at its FULL speed. The ab_1182_15 double-replacement: two L84 Muks, one par —
    /// the sim TIES them (par erased by `fnt`) and draws the instaswitch shuffle;
    /// the pre-fix port kept `par` on the corpse (×0.25 → no tie → a missing draw).
    fn check_fainted(&mut self) {
        for side in 0..2 {
            let a = self.sides[side].active;
            if self.sides[side].pokemon[a].fainted {
                self.sides[side].switch_flag = true;
                self.sides[side].pokemon[a].status = None;
            }
        }
    }

    /// The switch-request gate (battle.ts:2390-2422): a side with a flagged active
    /// that CAN switch (has a non-fainted bench mon) keeps the flag; one that can't
    /// clears it. Returns the per-side force table (which sides must replace).
    fn switch_request_gate(&mut self) -> [bool; 2] {
        let mut force = [false; 2];
        for side in 0..2 {
            if self.sides[side].switch_flag {
                if self.can_switch(side) {
                    force[side] = true;
                } else {
                    // Can't switch (no live bench) — clear the flag (the side has no
                    // replacement; the game is over for it, caught by checkWin).
                    self.sides[side].switch_flag = false;
                }
            }
        }
        force
    }

    /// The CURRENT array slot of `side`'s mon with stable identity `uid` (the
    /// construction-time index), after any `switchIn` array swaps. `None` if no such
    /// mon (defensive). This is the gen3-singles equivalent of dereferencing
    /// `action.pokemon` to find where the mon now lives.
    fn slot_of_uid(&self, side: usize, uid: usize) -> Option<usize> {
        self.sides[side].pokemon.iter().position(|m| m.uid == uid)
    }

    /// Whether `side` has a non-active, non-fainted bench mon to switch to.
    fn can_switch(&self, side: usize) -> bool {
        let active = self.sides[side].active;
        self.sides[side]
            .pokemon
            .iter()
            .enumerate()
            .any(|(i, m)| i != active && !m.fainted)
    }

    /// Execute a `switch`/`instaswitch` action: SWAP `side`'s `pokemon[active]` with
    /// `pokemon[target]` (so the entering mon takes the active position and the
    /// outgoing mon takes the entrant's old bench slot — Showdown's `switchIn`
    /// position swap, battle-actions.ts:125/131-133), then enqueue the entrant's
    /// `runSwitch` (order 101) via the `insertChoice` path — which draws the splice
    /// `random(firstIndex, lastIndex+1)` ONLY when its order-101 tie window has >1
    /// slot (the double-replacement case; a single replacement inserts with no tie
    /// window → no draw).
    ///
    /// Mirrors `switchIn` (battle-actions.ts:62): for a VOLUNTARY switch the outgoing
    /// mon is alive (we run its draw-free End/clearVolatile — no-ops in our scope);
    /// for a POST-FAINT replacement the outgoing mon is fainted (that block is
    /// skipped). gen3 does NOT inline runSwitch — it ENQUEUES it. The active slot
    /// INDEX stays the same (gen-3 singles: 0); the array entries swap so the
    /// entrant lives at that index — keeping team-slot numbering identical to
    /// Showdown (`switch N` ⇒ the bench slot AFTER prior swaps).
    /// The eligible forced-switch-in targets for `side` — `possibleSwitches`
    /// (battle.ts:1297): every NON-active, NON-fainted bench mon, in ARRAY ORDER (the
    /// list `sample`/`getRandomSwitchable` indexes into). In gen-3 singles the active is
    /// always index 0 (the swap keeps it there), so this is the non-fainted slots
    /// `1..pokemon.len`. Returns an empty list when the foe's last mon is alive (=
    /// `canSwitch` is false → a phaze FAILS). The ORDER is load-bearing: `sample` draws
    /// `random(list.len)` and picks `list[idx]`, so the list must match `possibleSwitches`'
    /// array walk EXACTLY (a wrong order drags the wrong mon — a STATE desync, not a seed
    /// one, since the same single draw is consumed either way).
    fn eligible_switch_ins(&self, side: usize) -> Vec<usize> {
        let active = self.sides[side].active;
        (0..self.sides[side].pokemon.len())
            .filter(|&i| i != active && !self.sides[side].pokemon[i].fainted)
            .collect()
    }

    /// `dragIn(side, pos)` (battle-actions.ts:159) — the forced random switch a
    /// successful phaze (Roar / Whirlwind) triggers, consumed at the runAction tail
    /// (battle.ts:2350-2353, AFTER the move body, BEFORE faintMessages). VERIFIED
    /// bit-for-bit vs the sim's PRNG probe (`harness/probe_phaze_rng.js`). The exact
    /// gen-3 sequence:
    ///   1. `getRandomSwitchable(side)` → `sample(possibleSwitches(side))` → `this.random(n)`
    ///      — THE RANDOM TARGET DRAW. Drawn EVEN when `n == 1` (`random(1)` returns 0 but
    ///      still calls `rng.next()` — the n=1 draw-count gotcha). The eligibility list is
    ///      already non-empty here (the phaze arm checked `canSwitch` before flagging).
    ///   2. The forced SWITCH (a `dragIn`, `isDrag = true`): the dragged-in mon takes the
    ///      active slot; the phazed-OUT mon's boosts/volatiles are cleared (it left).
    ///      Reuses `execute_switch` (the array swap + the entrant's `updateSpeed` +
    ///      the `insert_runswitch` enqueue) — gen-3 `dragIn`'s `switchIn(... isDrag=true)`
    ///      enqueues the order-101 `runSwitch` (gen <= 4 path, battle-actions.ts:155),
    ///      so the runSwitch (EntryHazard/Spikes → ability Start) runs as a separate
    ///      queued action, IDENTICALLY to a voluntary switch's runSwitch. So a phaze-into-
    ///      Spikes chips the dragged mon, and a Spikes-KO on entry forces a NORMAL
    ///      replacement — all via the EXISTING switch machinery, no new draw beyond the
    ///      one `sample`.
    /// (gen-3 `dragIn` also runs `runEvent('DragOut', oldActive)` — DRAW-FREE for our
    /// modeled abilities; Suction Cups [`onDragOut`] is NOT on any modeled team and is
    /// guarded fail-loud in the phaze e2e filter, never here.)
    fn drag_in(&mut self, side: usize, dex: &Dex, queue: &mut Vec<QAction>) {
        let eligible = self.eligible_switch_ins(side);
        debug_assert!(
            !eligible.is_empty(),
            "drag_in called with no eligible switch-in — the phaze arm must gate on canSwitch"
        );
        // (1) THE RANDOM TARGET DRAW — `sample(eligible)` = `random(eligible.len)`. ONE
        //     draw, even for len == 1 (the n=1 gotcha).
        let idx = self.prng.random_below(eligible.len() as u32) as usize;
        let target = eligible[idx];
        // Diagnostic/coverage flag: a phaze drag ACTUALLY fired (the `sample` ran). Read into
        // the per-decision `DecisionRecord.phaze_drag` for the e2e phaze coverage floor; does
        // NOT affect any draw or state (mirrors `pending_explosion_self_ko`).
        self.pending_phaze_drag = true;
        // (2) The forced switch (reuses the voluntary-switch swap + runSwitch enqueue).
        //     `is_drag = true` → `execute_switch` emits `|drag|`, not `|switch|`.
        self.execute_switch(side, target, true, dex, queue);
    }

    fn execute_switch(&mut self, side: usize, target: usize, is_drag: bool, dex: &Dex, queue: &mut Vec<QAction>) {
        let active = self.sides[side].active;
        // WEATHER_NEGATE `onEnd` → `eachEvent('WeatherChange')` (`gen3_cloudnine_end_v1`,
        // the A/B icebeam-tail root cause): the resolved gen3 Cloud Nine / Air Lock carry
        // an `onEnd` that fires `this.eachEvent("WeatherChange", this.effect)`
        // UNCONDITIONALLY (weather present or not). `switchIn` fires the outgoing mon's
        // ability `End` singleEvent ONLY for an ALIVE outgoing mon (`unfaintedActive` —
        // battle-actions.js:87/106: voluntary pivot AND phaze-drag; a fainted mon's
        // replacement skips it), BEFORE the position swap — so the eachEvent's 2-active
        // speedSort compares the OUTGOING holder vs the foe on cached speed, drawing ONE
        // `random(0,2)` iff they TIE (the ab_1196_18 Golduck-L81 mirror: the sim drew 8,
        // the port 7). DRAW-FREE otherwise. Ground truth: the sim stack
        // `abilities.js onEnd → singleEvent(End) → switchIn` (traced 2026-07-10); pin
        // `cloud_nine_switch_out_fires_the_weatherchange_shuffle`.
        {
            let m = &self.sides[side].pokemon[active];
            if !m.fainted
                && dex.ability(&to_id(&m.ability)).map(|a| a.weather_negate).unwrap_or(false)
            {
                self.each_event_shuffle();
            }
        }
        // [EMIT] `|-end|<outgoing>|ability: Flash Fire|[silent]` — an ALIVE outgoing mon
        // with an ARMED flash_fire volatile announces the singleEvent(End) removal
        // (`flashfire.onEnd`, the `gen3_cloudnine_end_v1` End mechanism), BEFORE the
        // entrant's `|switch|` line. A FAINTED armed holder emits NOTHING (byte-verified
        // vs the flashfire_cycle capture: switch-out shows the line, the faint does not).
        // Phase 3, observation-only.
        if self.logging() {
            let m = &self.sides[side].pokemon[active];
            if !m.fainted && m.flash_fire {
                let out_ref = self.mon_ref(side, active, dex);
                self.log.volatile_end_silent(&out_ref, "ability: Flash Fire");
            }
        }
        // The outgoing (active) mon: if alive (voluntary pivot), clearVolatile resets
        // its stat stages AND volatiles (confusion / flinch) — reset them so a later
        // re-entry is pristine (no seed effect). A fainted mon's clearVolatile already
        // ran in faintMessages; reset its volatiles too so a re-encode is clean.
        {
            let m = &mut self.sides[side].pokemon[active];
            if !m.fainted {
                m.boosts = [0; crate::state::BOOST_LEN];
            }
            // clearVolatile removes confusion + flinch regardless of faint state.
            m.confusion = None;
            m.flinch = false;
            // The `protect` this-turn volatile + the `stall` counter (+ its duration) clear
            // on switch-out (clearVolatile drops them — verified vs the sim:
            // P,P,switch,switchback,P,P re-protects on a FRESH counter, drawing rc(1,2)
            // again, not rc(1,8)).
            m.protected = false;
            m.protect_counter = 0;
            m.stall_duration = 0;
            // The LEECH SEED volatile clears on switch-out (clearVolatile) — a seeded mon
            // that switches out is no longer seeded (a fresh Leech Seed is needed). VERIFIED
            // vs the sim. (We only clear the OUTGOING mon's seeded state; a seeder switching
            // out does NOT clear the foe's seed — the residual reads the seeder side's
            // CURRENT active, so the heal simply follows the replacement.)
            m.leech_seed = None;
            // The SUBSTITUTE volatile clears on switch-out (clearVolatile) — a sub does not
            // follow its owner off the field; the entrant comes in with no sub.
            m.substitute = None;
            // The CHOICE-LOCK volatile clears on switch-out (`gen3_pp_tracking_v1`,
            // `choicelock` is `noCopy` + dropped by clearVolatile / re-set by
            // `choiceband.onStart`): a Choice-Band mon that pivots out is UNLOCKED, so it can
            // pick any move on its next entry (and re-lock to the first it uses). PP itself
            // does NOT reset (VERIFIED — it persists across the switch).
            m.choice_locked_move = None;
            // The TAUNT + DISABLE selection-restriction volatiles clear on switch-out
            // (`gen3_taunt_disable_v1`, clearVolatile drops them): a taunted/disabled mon that
            // pivots out comes back UN-restricted (a fresh Taunt/Disable is needed). And
            // `last_move` clears (`pokemon.lastMove` is reset on switch-out — a switched-in mon
            // has no lastMove, so a Disable into it fails draw-free). VERIFIED vs the sim.
            m.taunt = None;
            m.disable = None;
            m.last_move = None;
            // The FLASH FIRE activation volatile clears on switch-out (`clearVolatile` →
            // `flashfire.onEnd`, verified vs the sim `harness/probe_flashfire_rng.js` A5): an
            // FF mon that pivots out loses its ×1.5 boost and must re-absorb a Fire move to
            // re-arm it. VERIFIED bit-for-bit.
            m.flash_fire = false;
            // NATURAL CURE (`gen3_natural_cure_v1`, the sole gen-3 SWITCH_OUT-cure ability) —
            // an ALIVE outgoing mon holding Natural Cure has its major `status` CURED as it
            // leaves the field (all of brn/par/psn/tox/slp/frz; the tox stage + sleep counter
            // reset since clearing `status` to None drops the whole Status variant). This is
            // the gen-3 `naturalcure.onSwitchOut` — fired by `switchIn`'s `runEvent('SwitchOut',
            // oldActive)`, which runs on BOTH a voluntary pivot AND a phaze-DRAG-out (only
            // `BeforeSwitchOut` is `!isDrag`-gated; SwitchOut fires regardless of `is_drag`), and
            // BEFORE `clearVolatile()`. It is a NO-OP on a fainted mon (the `onSwitchOut` guard
            // `if (!pokemon.status || pokemon.status==='fnt') return`), which is why this sits
            // inside the `!m.fainted` gate. **DRAW-FREE** — the cure + its `[silent]`
            // `-curestatus` reveal (`onCheckShow` is `undefined` in the resolved gen3 dist)
            // consume ZERO PRNG, so admitting Natural Cure is SEED-NEUTRAL for every
            // pre-existing suite. PROBE-SETTLED bit-for-bit vs the omniscient sim by
            // `harness/probe_naturalcure_rng.js` (D1 draw-count == a non-NC control for all 6
            // statuses, voluntary + drag; D2 which-statuses; D5 the faint no-op).
            if !m.fainted && to_id(&m.ability) == "naturalcure" {
                m.status = None;
            }
            // TRACE revert (`gen3_berry_trace_shedskin_v1`): switch-out restores the BASE
            // ability (Showdown's clearVolatile `ability = baseAbility` — probe T4: a
            // traced Gardevoir is `trace` again on the bench and RE-TRACES on re-entry).
            // A no-op for every un-traced mon (`ability` already == `set.ability`). The
            // EATEN-item state, by contrast, is PERMANENT (item does NOT revert).
            m.ability = m.set.ability.clone();
            // The FOCUS ENERGY volatile (a Lansat eat) clears on switch-out (clearVolatile).
            m.focus_energy = false;
            // The ATTRACT volatile clears on the HOLDER's switch-out (clearVolatile —
            // probe_cutecharm_attract_rng.js "holder leaves"). `gen3_ability_batch4_v1`.
            m.attract = None;
            // The COLOR CHANGE type-override clears on switch-out (a re-entering Kecleon
            // is Normal again — probe_colorchange_rng.js t4). `gen3_ability_batch4_v1`.
            m.types_override = None;
        }
        // ATTRACT source-left clear (`attract.onUpdate`, `gen3_ability_batch4_v1`): when the
        // DEPARTING mon is the SOURCE of the foe active's attraction, the volatile is removed
        // (probe: Miltank pivots out → Zangoose's Attract ends, `-end …|Attract|[silent]`).
        // DRAW-FREE; checked here at the switch (the sim's next Update — nothing between reads it).
        {
            let departing_uid = self.sides[side].pokemon[self.sides[side].active].uid;
            let foe = 1 - side;
            let foe_active = self.sides[foe].active;
            if self.sides[foe].pokemon[foe_active].attract == Some((side, departing_uid)) {
                self.sides[foe].pokemon[foe_active].attract = None;
                // [EMIT] `|-end|<mon>|Attract|[silent]` — the onUpdate removal.
                if self.logging() {
                    let mon_ref = self.mon_ref(foe, foe_active, dex);
                    self.log.volatile_end_silent(&mon_ref, "Attract");
                }
            }
        }
        // SWAP the team-array entries (the entrant → active position, outgoing →
        // the entrant's old bench position) + fix each mon's `position` field.
        self.sides[side].pokemon.swap(active, target);
        self.sides[side].pokemon[active].position = active;
        self.sides[side].pokemon[target].position = target;
        // TOXIC STAGE — the reset does NOT live here (`gen3_tox_stage_persists_v1`):
        // the resolved gen3 `tox.onSwitchIn(){ stage = 0 }` fires via the RUNSWITCH-time
        // `runEvent('SwitchIn')` (mods/gen4/scripts.js:42), NOT at this raw array swap.
        // So the stage RESETS when the queued runSwitch actually RUNS (`run_switch`
        // owns the reset — probe `harness/probe_tox_stage_switch.js`: re-entry residual
        // 22 = maxhp/16 = stage 1 again), but PERSISTS when the queued runSwitch is
        // CANCELLED by gen3's faint-cancels-all (a co-replacement Spikes-faint — the
        // ab_1166_22 Mew: the unreset stage-2 chip KO'd it where the port's swap-time
        // reset left it alive + rolled a phantom Quick Claw). Pinned BOTH ways:
        // `tox_stage_resets_when_the_runswitch_runs` (TX1) +
        // `tox_stage_persists_when_the_runswitch_is_cancelled` (TX2 — pins the
        // PLACEMENT: a swap-time reset here fails TX2, no reset anywhere fails TX1).
        // The entrant's cached `pokemon.speed` is established to its CURRENT para/boost-
        // aware action speed on switch-in (VERIFIED vs the sim via the e2e-capstone
        // eachEvent probe: a Jirachi that switches in PARALYZED appears at its para
        // speed 53, NOT its raw 212, in the first post-switch `eachEvent('Update')`
        // shuffle; an unparalyzed entrant appears at raw == live). This differs from a
        // mon that is paralyzed WHILE already active, whose cached `.speed` goes stale
        // at the turn-start value until the residual's `updateSpeed` refresh — so the
        // switch-in establishes the live value, while a mid-turn status change does not
        // (it waits for the next `update_speed` site). Computed AFTER the array swap, so
        // `active` is the entrant.
        self.sides[side].pokemon[active].cached_speed = self.effective_speed(side, active, dex);
        // The entrant's `activeTurns` RESETS to 0 on switch-in (`switchIn` sets
        // `pokemon.activeTurns = 0`, battle-actions.ts:137) — so a Speed Boost mon that
        // switches in does NOT boost on its entry turn (activeTurns 0 at that residual;
        // `endTurn` increments it to 1 afterward). `gen3_ability_batch1_v1`.
        self.sides[side].pokemon[active].active_turns = 0;
        // TRUANT arming (`truant.onSwitchIn`: `truantTurn = this.turn !== 0`,
        // `gen3_ability_batch4_v1`): every mid-battle entrant arms `true`; whether it then
        // LOAFS its first full turn depends on whether the order-27 residual still toggles
        // it this turn (a mid-turn pivot/drag/action-faint replacement → toggled → moves;
        // a post-residual DoT-KO replacement → un-toggled → loafs). Leads arm `false` at
        // construction (turn 0 — `MonState::truant_turn`'s default; the lead path doesn't
        // run execute_switch). Probes `probe_truant_rng.js` Q3/Q3b + `probe_truant_edges_rng.js`.
        if to_id(&self.sides[side].pokemon[active].ability) == "truant" {
            self.sides[side].pokemon[active].truant_turn = self.turn != 0;
        }
        // `active` index is unchanged (gen-3 singles pos 0); the entrant now lives
        // there. Clear the side's switch flag — it's been answered.
        self.sides[side].switch_flag = false;

        // [EMIT] `|switch|<entrant>|<Details>|<HP>` (a voluntary/forced-replacement
        // switch) OR `|drag|…` (a Roar/Whirlwind phaze). Emitted AFTER the array swap,
        // so `active` is the entrant — its ident/details/HP are the fresh entrant's.
        // The spikes switch-in chip (`apply_entry_hazards`) fires LATER when the queued
        // RunSwitch runs, emitting its own `|-damage|…|[from] Spikes` — so this `|switch|`
        // /`|drag|` correctly precedes the hazard chip (matching the golden). Observation-
        // only. NOTE: the switch-in ability lines (`|-ability|`/`|-weather|`) are Phase 2.
        if self.logging() {
            let entrant = self.mon_ref(side, active, dex);
            let details = self.switch_details(side, active, dex);
            let hp = self.hp_status(side, active);
            if is_drag {
                self.log.drag(&entrant, &details, &hp);
            } else {
                self.log.switch(&entrant, &details, &hp);
            }
        }

        // Enqueue the runSwitch (order 101) via insertChoice — drawing the splice
        // only on an order-101 tie window (double replacement).
        self.insert_runswitch(side, queue, dex);
    }

    /// `queue.insertChoice({choice:'runSwitch', pokemon})` (battle-actions.ts:157):
    /// insert the order-101 runSwitch into `queue` at its sorted position, drawing
    /// `random(firstIndex, lastIndex+1)` iff its tie window (same order+priority+
    /// speed) spans >1 slot. For a SINGLE replacement the runSwitch is strictly
    /// before the leftover move/residual (no tie) → no draw; for a DOUBLE
    /// replacement the 2nd runSwitch ties the 1st (both order 101, both entrants'
    /// speeds) → one splice draw.
    fn insert_runswitch(&mut self, side: usize, queue: &mut Vec<QAction>, dex: &Dex) {
        let new = QAction::RunSwitch { side };
        let new_key = self.action_sort_key(&new, dex);

        // Find the insert window [firstIndex, lastIndex) exactly like insertChoice:
        //   firstIndex = first i where compare(new, queue[i]) <= 0 (new sorts at/before)
        //   lastIndex  = first i where compare(new, queue[i]) <  0 (new strictly before)
        let mut first_index: Option<usize> = None;
        let mut last_index: Option<usize> = None;
        for (i, cur) in queue.iter().enumerate() {
            let cmp = compare_keys(&new_key, &self.action_sort_key(cur, dex));
            if cmp <= 0.0 && first_index.is_none() {
                first_index = Some(i);
            }
            if cmp < 0.0 {
                last_index = Some(i);
                break;
            }
        }
        match first_index {
            None => queue.push(new), // sorts after everything → append, no draw
            Some(fi) => {
                let li = last_index.unwrap_or(queue.len());
                let index = if fi == li {
                    fi // unique slot → no draw
                } else {
                    // The order-101 tie window spans >1 slot → the splice draw.
                    self.prng.random_range(fi as u32, (li + 1) as u32) as usize
                };
                queue.insert(index, new);
            }
        }
    }

    /// The `comparePriority` key for one queued action (order, priority, speed).
    fn action_sort_key(&self, a: &QAction, dex: &Dex) -> (u64, i32, f64) {
        match a {
            QAction::Move { side, uid, .. } => {
                // priority is read without the dex here (insert path only compares
                // order vs other actions; runSwitch[101] vs move[200] differ on order
                // so priority/speed never decide). Use 0 priority + the actor speed.
                let slot = self.slot_of_uid(*side, *uid).unwrap_or(self.sides[*side].active);
                (a.order(), 0, self.effective_speed(*side, slot, dex) as f64)
            }
            QAction::Switch { side, .. } | QAction::InstaSwitch { side, .. } => {
                (a.order(), 0, self.effective_speed(*side, self.sides[*side].active, dex) as f64)
            }
            QAction::RunSwitch { side } => {
                // The runSwitch action's speed key is the ENTERING mon's speed (it's
                // resolved at insertChoice time, after the switch swapped active).
                (a.order(), 0, self.effective_speed(*side, self.sides[*side].active, dex) as f64)
            }
            QAction::BeforeTurn | QAction::Residual => (a.order(), 0, 0.0),
        }
    }

    /// gen-3 `runSwitch` (the gen4 override, data/mods/gen4/scripts.ts:18). The exact
    /// order (gen3-inherited):
    ///   1. `runEvent('EntryHazard', pokemon)` — the **Spikes** switch-in damage (see
    ///      [`apply_entry_hazards`]). DRAW-FREE (a single side-condition handler → no
    ///      tie-shuffle; `runEvent('Damage')` has no drawing handler for our abilities).
    ///   2. `runEvent('SwitchIn', pokemon)` — no gen-3 ability has a drawing `onSwitchIn`.
    ///   3. `if (!pokemon.hp) return false;` — if a Spikes hit KO'd the entrant, STOP
    ///      (the ability `Start` is SKIPPED; the faint is set by the runAction tail's
    ///      `process_faints`, which then forces ANOTHER replacement). VERIFIED vs the sim.
    ///   4. `singleEvent('Start', ability)` — Intimidate / Sand Stream / Drizzle / Drought
    ///      via [`single_event_ability_start`]. DRAW-FREE for our abilities.
    /// No `speedSort(allActive)`, no `fieldEvent('SwitchIn')` (the base-sim path gen3
    /// REPLACES — do NOT add a SwitchIn tie-shuffle).
    ///
    /// # The `eachEvent('WeatherChange')` switch-in draw (`Field.setWeather`, field.ts:87)
    ///
    /// The ability `Start` step is draw-free for the boost/weather APPLY itself, BUT when a
    /// Sand Stream / Drizzle / Drought entrant actually SETS (changes) the weather,
    /// `field.setWeather` ends with `this.battle.eachEvent('WeatherChange', sourceEffect)`
    /// (field.ts:87, before the gen-3 `>=7` Update-nest gate so NO nested Update) — a
    /// `speedSort(getAllActive())` over the two actives that draws ONE `random(0,2)` Fisher-
    /// Yates tie-shuffle IFF they TIE on cached speed (and nothing on distinct speed). This
    /// is the SWITCH-IN-INTO-A-SPEED-TIE-UNDER-FRESHLY-SET-WEATHER draw the e2e fuzz surfaced
    /// (e2e_84 dec4: a 213-speed Tyranitar switches in under Sand Stream while a 213-speed
    /// Suicune acts → the sim draws ONE more than the port did). It fires INSIDE this
    /// runSwitch runAction (during the ability Start), BEFORE the runAction tail's
    /// faintMessages + trailing `eachEvent('Update')` — so `turn_loop` fires the
    /// `each_event_shuffle()` immediately on this returning `true`, ahead of `process_faints`.
    /// A re-set of the SAME ability-permanent weather is a `setWeather`-returns-false no-op
    /// (no eachEvent) — `set_weather_changed_on_start` returns false there, matching the sim.
    ///
    /// Returns `true` iff the ability `Start` CHANGED the field weather (⇒ the caller must
    /// fire the `eachEvent('WeatherChange')` tie-shuffle).
    fn run_switch(&mut self, side: usize, dex: &Dex) -> bool {
        let slot = self.sides[side].active;
        if self.sides[side].pokemon[slot].fainted {
            return false; // (defensive: a runSwitch never fires for a fainted mon)
        }
        // (1) EntryHazard — apply Spikes switch-in damage (grounded-only). DRAW-FREE.
        self.apply_entry_hazards(side, slot, dex);
        // (2) runEvent('SwitchIn') — the gen4-override runSwitch fires it right after
        // EntryHazard (mods/gen4/scripts.js:41-42, BEFORE the hp check). The one modeled
        // handler is `tox.onSwitchIn`: the entrant's TOXIC stage resets to 0
        // (`gen3_tox_stage_persists_v1` — the stage reset lives HERE, in the runSwitch,
        // NOT in `execute_switch`'s array swap). LOAD-BEARING for the cancellation law:
        // a replacement whose queued runSwitch is CANCELLED (gen3 faint-cancels-all —
        // e.g. its co-replacement died to Spikes, the ab_1166_22 Mew) KEEPS its prior
        // stage and its next residual keeps ramping from it; a replacement whose
        // runSwitch RUNS (the ab_403_13 Umbreon: re-entered at stage 4, the resumed
        // residual dealt 19 = stage 1) resets. Probe-settled three ways: the
        // voluntary-switch reset (`harness/probe_tox_stage_switch.js`: 22 then 44 after
        // re-entry, `statusState.stage` 1→2), the forced-replacement reset (the 403
        // protocol: heal +19 / chip −19 on the resumed residual), and the
        // cancelled-runSwitch persistence (the 1166 protocol: Mew re-enters at 13 tox,
        // heal +16 → 29, chip 32 = stage 2 → faints). DRAW-FREE.
        if let Some(Status::Toxic(_)) = self.sides[side].pokemon[slot].status {
            self.sides[side].pokemon[slot].status = Some(Status::Toxic(0));
        }
        // (3) If the entrant was KO'd by the hazard, its ability `Start` does NOT fire
        // (`if (!pokemon.hp) return false`). The faint flag + the forced replacement are
        // handled by the runAction tail (process_faints → check_fainted → the switch gate).
        if self.sides[side].pokemon[slot].hp == 0 {
            return false;
        }
        // (4) ability `Start` (Intimidate / weather). The boost/weather APPLY is draw-free;
        // a WEATHER CHANGE additionally requires the `eachEvent('WeatherChange')` shuffle,
        // which the caller fires (the shuffle reads the prng, which must advance AFTER the
        // weather is set — exactly the source order). Snapshot the weather state across the
        // Start to detect a real change (a same-weather permanent re-set is a no-op).
        let before = (self.field.weather, self.field.weather_turns);
        // Snapshot the FOE active's pre-drop Atk stage for the Intimidate emit's CLAMPED
        // applied delta (a foe already at −6 emits `atk|0`, not `atk|1`; the post-Start
        // state can't recover the pre-drop stage — a −6 is ambiguous). Read here, BEFORE
        // the Start applies + clamps the drop. Harmless for a non-Intimidate entrant (the
        // emit ignores it unless the ability is Intimidate).
        let intim_foe = 1 - side;
        let intim_atk_pre = self.sides[intim_foe].pokemon[self.sides[intim_foe].active].boosts[0];
        // `draw_trace=true`: a MID-BATTLE Trace switch-in DRAWS its `randomFoe()` sample
        // (`gen3_berry_trace_shedskin_v1`, probe T1 — `random(1)` even for the single foe),
        // INSIDE this runSwitch runAction (before the trailing Update), unlike the
        // battle-start window where the draw pre-dates the seeded start.
        single_event_ability_start(self, side, slot, true);
        let after = (self.field.weather, self.field.weather_turns);
        let weather_changed = before != after;
        // [EMIT] the MID-BATTLE switch-in ability lines (Phase 3 — the framing already
        // emits the lead versions): Intimidate / weather-setter / Pressure / Trace.
        // The weather SET line fires only on a REAL change (a permanent same-weather
        // re-set — e.g. a re-dragged Sand Stream Tyranitar under standing sand — is a
        // setWeather no-op and emits nothing). Observation-only.
        self.emit_ability_start_lines(side, slot, weather_changed, Some(intim_atk_pre), dex);
        weather_changed
    }

    /// Apply the gen-3 **Spikes** entry-hazard damage to a switching-in mon — the
    /// `runEvent('EntryHazard')` step of the gen4 `runSwitch` (gen3-inherited). DRAW-FREE
    /// (the side condition's `onEntryHazard` does deterministic `this.damage(...)`; the
    /// nested `runEvent('Damage')` has no drawing handler for our abilities, so the seed
    /// is unchanged — VERIFIED vs the omniscient sim `harness/probe_spikes_rng.js`).
    ///
    /// The gen-3 amount (the resolved `spikes.onEntryHazard`,
    /// `damageAmounts[layers] * maxhp / 24` → `damage()` → `clampIntRange(_, 1)`): with
    /// `damageAmounts = [_, 3, 4, 6]` and the `clampIntRange` floor-then-min-1,
    ///   1 layer → `max(floor(maxhp/8), 1)`,
    ///   2 layers → `max(floor(maxhp/6), 1)`,
    ///   3 layers → `max(floor(maxhp/4), 1)`.
    /// GROUNDED-ONLY: a FLYING-type or **Levitate** entrant takes ZERO (`isGrounded()` is
    /// false/null). (Iron Ball / Air Balloon / Magnet Rise etc. don't exist in gen-3 OU
    /// and never appear on a modeled team, so grounded == not-Flying && not-Levitate.)
    /// A hazard hit that zeroes HP faints the entrant (the tail processes it → forces
    /// another replacement, which ALSO takes Spikes); no Quick Claw / extra draw.
    fn apply_entry_hazards(&mut self, side: usize, slot: usize, dex: &Dex) {
        let layers = self.sides[side].spikes;
        if layers == 0 {
            return;
        }
        // GROUNDED check: Flying-type or Levitate ability → immune (ZERO damage).
        let mon = &self.sides[side].pokemon[slot];
        let types = mon_types(mon, dex);
        let is_flying = types.contains(&Type::Flying);
        let is_levitate = to_id(&mon.ability) == "levitate";
        if is_flying || is_levitate {
            return;
        }
        // gen-3 amount: damageAmounts[layers] * maxhp / 24, floored, min 1.
        let amounts = [0u32, 3, 4, 6];
        let maxhp = mon.maxhp as u32;
        let raw = amounts[layers as usize] * maxhp / 24; // integer floor (matches JS float→floor)
        let dmg = raw.max(1) as u16;
        // FOCUS BAND: the Spikes chip is a Damage event into the entrant — the roll
        // draws (probe: the switch-in turn shows the extra random(10)), no survive
        // (effect Spikes, not a Move). `gen3_ability_batch4_v1`.
        let dmg = self.focus_band_damage(side, slot, dmg, false, dex);
        // Apply (saturating at 0). The faint flag is set by the runAction tail's
        // process_faints (mirroring faintMessages), exactly like move damage.
        self.apply_damage(side, slot, dmg);
        // [EMIT] `|-damage|<entrant>|<HP>|[from] Spikes` — the grounded switch-in chip,
        // post-chip HP (`0 fnt` if it KO'd). Observation-only. Emitted here (in RunSwitch)
        // so it correctly follows the `|switch|`/`|drag|` line from `execute_switch`.
        if dmg > 0 && self.logging() {
            let mon_ref = self.mon_ref(side, slot, dex);
            let hp = self.hp_status(side, slot);
            self.log.damage(&mon_ref, &hp, Some(&Cause::Bare("Spikes".to_string())));
        }
    }
}

/// Compare two `comparePriority` keys (order asc, priority desc, speed desc).
/// Returns <0 if `a` sorts BEFORE `b`, >0 if after, 0 on a full tie — matching
/// `comparePriority`'s sign convention used by `insertChoice` (`compared <= 0`
/// ⇒ a sorts at/before b).
fn compare_keys(a: &(u64, i32, f64), b: &(u64, i32, f64)) -> f64 {
    if a.0 != b.0 {
        return a.0 as f64 - b.0 as f64; // smaller order first
    }
    if a.1 != b.1 {
        return (b.1 - a.1) as f64; // higher priority first
    }
    if a.2 != b.2 {
        return b.2 - a.2; // higher speed first
    }
    0.0
}

/// The gen-3 SUBSTITUTE cost = `floor(maxhp/4)`, which is ALSO the created sub's HP
/// (`directDamage(maxhp/4)` floors, and the volatile's `onStart` sets `effectState.hp =
/// Math.floor(maxhp/4)`). A Substitute FAILS (draw-free) if `hp <= floor(maxhp/4)` — the
/// user can't afford to make a sub it has exactly enough HP for (the gen-3 `<=` boundary;
/// VERIFIED: hp == floor(maxhp/4) FAILS, hp == that + 1 SUCCEEDS).
fn sub_cost(maxhp: u16) -> u16 {
    maxhp / 4
}

/// Whether this move is IMMUNE against the defender — an ability/Levitate
/// immunity (`ctx.immune`, caller-resolved) OR a type-chart 0× (Electric→Ground,
/// Ground→Flying, etc.). Gen-3 `tryMoveHit` resolves this AFTER the accuracy roll
/// but BEFORE the crit/damage draws, so the engine must short-circuit here to keep
/// the draw count exact. A typeless move (`move_type == None`) is never immune by
/// the chart (neutral). Mirrors the immunity short-circuit inside `calc_damage`
/// (which would also return 0) but is checked BEFORE the crit roll so no crit/damage
/// draw is consumed.
fn move_is_immune(ctx: &DamageContext, dex: &Dex) -> bool {
    if ctx.immune {
        return true;
    }
    match ctx.mv.move_type {
        Some(t) => dex.type_chart().effectiveness(t, &ctx.defender.types) == 0.0,
        None => false,
    }
}

/// The GEN-3 STATUS type-immunity gate (`runStatusImmunity` → `dex.getImmunity`,
/// reading the GEN-3 `typechart.damageTaken[status] === 3`). The repo's
/// `data/pokemon/gen3_type_chart.json` is multipliers-ONLY (no `damageTaken`
/// status rows), so these rules are ADDED here to mirror the GEN-3 sim. CRUX:
/// **gen-3 has NO Electric→paralysis immunity** (that was added in Gen 6) — a
/// VERIFIED-vs-sim fact (Jolteon/Magneton `trySetStatus('par')` returns true in
/// the gen3 mod; `dex.types.get('Electric').damageTaken.par` is `undefined`). The
/// gen-3 status type-immunities, each confirmed against the live gen3 sim:
///   - `frz` → ICE is immune;
///   - `brn` → FIRE is immune;
///   - `psn`/`tox` → POISON & STEEL are immune (`tox` checks the `psn` immunity);
///   - `par`/`slp` have NO type immunity (Electric/Steel CAN be paralyzed in gen3).
/// Returns whether ANY of `types` confers immunity.
fn status_type_immune(status: &str, types: &[Type]) -> bool {
    let immune_type = |t: Type| match status {
        "frz" => t == Type::Ice,
        "brn" => t == Type::Fire,
        "psn" | "tox" => t == Type::Poison || t == Type::Steel,
        // par / slp (and anything else): NO type immunity in gen 3.
        _ => false,
    };
    types.iter().any(|&t| immune_type(t))
}

/// The lowercase protocol status token (`brn`/`par`/`slp`/`frz`/`psn`/`tox`) for an
/// HP-line's appended status field, or `None` for no major status. Used ONLY by the
/// PROTOCOL-EMISSION HP formatter (`HpStatus`) — the omniscient stream appends the
/// status token to a live mon's HP (`116/524 slp`); a fainted mon renders `0 fnt`
/// (the token is dropped by `HpStatus`).
/// The stat a nature LOWERS (`pokemon.getNature().minus`,
/// `gen3_berry_trace_shedskin_v1` — the Figy-family confusion gate), as the boosts-order
/// token (`atk`/`def`/`spa`/`spd`/`spe`). `None` for a neutral nature (all ×1.0) or an
/// EMPTY nature field (the sim's nonexistent-nature = no plus/minus — the e2e_8 Suicune).
fn nature_minus_stat(nature: &str, dex: &Dex) -> Option<String> {
    if to_id(nature).is_empty() {
        return None;
    }
    let n = dex.nature(nature)?;
    for (name, mult) in [("atk", n.atk), ("def", n.def), ("spa", n.spa), ("spd", n.spd), ("spe", n.spe)] {
        if mult < 1.0 {
            return Some(name.to_string());
        }
    }
    None
}

fn status_token(status: Option<Status>) -> Option<&'static str> {
    match status {
        Some(Status::Burn) => Some("brn"),
        Some(Status::Paralysis) => Some("par"),
        Some(Status::Sleep(_)) => Some("slp"),
        Some(Status::Freeze) => Some("frz"),
        Some(Status::Poison) => Some("psn"),
        Some(Status::Toxic(_)) => Some("tox"),
        None => None,
    }
}

/// The Showdown protocol display name for a [`Weather`] (the `-weather` line value —
/// e.g. Sand → `Sandstorm`, Rain → `RainDance`, Sun → `SunnyDay`, Hail → `Hail`;
/// matching poke-env's parsed tokens). Used by the switch-in weather-set line + the
/// residual `[upkeep]` tick.
fn weather_display(weather: Weather) -> &'static str {
    match weather {
        Weather::Sand => "Sandstorm",
        Weather::Rain => "RainDance",
        Weather::Sun => "SunnyDay",
        Weather::Hail => "Hail",
    }
}

/// The MODELED standalone status-inflicting moves → the gen-3 status id (`par`/`tox`/
/// `psn`/`brn`/`slp`) they apply. Only the foe-targeting MAJOR-status moves the
/// engine builds bit-for-bit are listed; any other status move (recovery/boost/phaze/
/// hazard/Substitute/field) returns `None` → the caller PANICS (a future status move
/// can never silently desync). The id is `to_id`-normalized.
/// The MODELED gen-3 PHAZE moves (`forceSwitch: true`): exactly **Roar** and
/// **Whirlwind**. These force the FOE to switch to a RANDOM eligible team member. The
/// extractor's `isPhaze` (= `forceSwitch`) covers exactly these two in gen-3 (Haze
/// resets boosts via `boosts`, NOT forceSwitch; Perish Song / Roar of Time are not
/// gen-3 `forceSwitch` phaze moves) — so this id list and the dex flag agree. Routed in
/// `run_status_move`'s phaze arm; any other `forceSwitch` move (none exist in gen-3)
/// would fall through to the fail-loud status-move guard.
fn modeled_phaze_move(move_id: &str) -> bool {
    matches!(move_id, "roar" | "whirlwind")
}

fn modeled_status_move(move_id: &str) -> Option<&'static str> {
    Some(match move_id {
        // paralysis
        "thunderwave" | "stunspore" | "glare" => "par",
        // badly poisoned
        "toxic" => "tox",
        // poison
        "poisonpowder" | "poisongas" => "psn",
        // burn
        "willowisp" => "brn",
        // sleep
        "spore" | "sleeppowder" | "hypnosis" | "sing" | "lovelykiss" | "grasswhistle" => "slp",
        _ => return None,
    })
}

/// The PRIMARY self-boost spec for a PURE SETUP move (Swords Dance / Dragon Dance /
/// Calm Mind / Agility / Bulk Up / Amnesia / Barrier / Acid Armor / Iron Defense /
/// Cosmic Power / Tail Glow / Meditate / Sharpen / Howl / Harden / Withdraw / Growth)
/// — the `(boost-array index, stages)` pairs the engine applies on the USER, read
/// straight from the data file's `selfBoosts` (`MoveData::self_boosts`). Returns
/// `Some(spec)` ONLY for a move that actually carries a non-empty `selfBoosts` (the
/// ~17 pure setup moves the extractor emits — moves with an extra effect, a volatile,
/// an evasion boost, or an HP cost are EXCLUDED at extraction, so they return `None`
/// and the caller's fail-loud status-move guard catches them). A clone of the dex
/// spec, so the caller can mutate `self` while iterating it.
fn self_boost_spec(move_id: &str, dex: &Dex) -> Option<Vec<(usize, i8)>> {
    let m = dex.moves(move_id)?;
    if m.self_boosts.is_empty() {
        None
    } else {
        Some(m.self_boosts.clone())
    }
}

/// The MODELED SELF-HEAL / RECOVERY-MOVE amount for `mon` under `weather` — the integer
/// HP `this.heal()` adds to the USER. Returns `Some(amount)` ONLY for the modeled
/// non-Rest recovery moves; `None` for everything else (Rest is routed separately, and a
/// non-recovery status move falls through to the caller's fail-loud guard).
///
/// REST is NOT here (it's handled by `run_rest` — a full heal + self-sleep + cure, not a
/// plain `heal`). DEFERRED recovery-class moves stay `None` → fail-loud: **Wish** (a
/// DELAYED slot-keyed end-of-next-turn heal), **Heal Bell / Aromatherapy / Refresh**
/// (team/self STATUS cure, not HP), **Pain Split / Leech Seed / drain / Ingrain /
/// Aqua Ring**.
///
/// Amounts (gen3 `maxhp == baseMaxhp`, all `floor`/integer truncation):
///   - **Recover / Soft-Boiled / Slack Off / Milk Drink** — `floor(maxhp/2)` (the
///     `move.heal:[1,2]` path, `Math.floor(baseMaxhp*1/2)`).
///   - **Moonlight / Synthesis / Morning Sun** — the gen4-inherited weather-conditional
///     `onHit` (PLAIN integer, NOT the 4096-`modify`): NONE → `floor(maxhp/2)`; SUN →
///     `floor(maxhp*2/3)`; SAND/RAIN/HAIL → `floor(maxhp/4)`. VERIFIED vs the omniscient
///     sim (Espeon maxhp 271 in sun heals `floor(271*2/3)=180`, not `modify=181`).
fn recovery_heal_amount(move_id: &str, mon: &crate::state::MonState, weather: Option<Weather>) -> Option<u16> {
    let maxhp = mon.maxhp;
    Some(match move_id {
        // Flat half-HP recovery (the `move.heal:[1,2]` path).
        "recover" | "softboiled" | "slackoff" | "milkdrink" => maxhp / 2,
        // Weather-conditional recovery (gen4-inherited integer onHit).
        "moonlight" | "synthesis" | "morningsun" => match weather {
            Some(Weather::Sun) => (maxhp as u32 * 2 / 3) as u16,
            Some(Weather::Sand) | Some(Weather::Rain) | Some(Weather::Hail) => maxhp / 4,
            None => maxhp / 2,
        },
        _ => return None,
    })
}

/// The MODELED gen-3 FIXED-DAMAGE moves → the exact damage they deal (bypassing
/// `getDamage`, so NO crit roll + NO 16-way damage roll). Showdown implements these
/// with `damage: 'level' | <int>` or a `damageCallback` in `data/moves.ts`; our
/// `gen3_moves.json` has no such field (a fixed-damage move is recognized by its move
/// id, not a data flag — the same convention every other move layer uses), so the
/// fixed formula is pinned PER-ID here, VERIFIED bit-for-bit vs the omniscient sim's
/// PRNG probe (`harness/probe_fixeddamage_rng.js`). `attacker` is the USER (its level),
/// `defender` is the TARGET (its CURRENT hp — Super Fang halves it).
///
/// The MODELED set (all draw-identical: accuracy-only, no crit, no damage roll):
///   - **Seismic Toss / Night Shade** — `damage: 'level'` → the USER's level exactly
///     (e.g. level 100 → 100). Seismic Toss is Fighting (a GHOST is immune, 0× — the
///     immunity is resolved in `run_move` by `move_is_immune`, accuracy-drawn-then-
///     `-immune`); Night Shade is Ghost (a NORMAL is immune, 0×).
///   - **Sonic Boom** — `damage: 20` (Normal; a GHOST is immune, 0×).
///   - **Dragon Rage** — `damage: 40` (Dragon; NO gen-3 type immunity).
///   - **Super Fang** — `damageCallback` = `clampIntRange(target.getUndynamaxedHP()/2,
///     1)` = `max(floor(target.hp / 2), 1)`; gen-3 `getUndynamaxedHP() == hp`, so it
///     halves the TARGET's CURRENT HP (min 1; Normal → a GHOST is immune, 0×). Behind a
///     SUBSTITUTE it STILL halves the MON's hp — the `damageCallback` reads `target.hp`
///     BEFORE the sub-intercept redirects the resulting NUMBER onto the sub's HP
///     (VERIFIED vs the sim: SF into a full-HP-536 Blissey behind a 178-HP sub deals
///     floor(536/2)=268 → the sub BREAKS, no carry, NOT floor(178/2)=89).
///
/// DEFERRED (fail-loud — these need extra RNG or reactive/OHKO machinery the caller
/// PANICS on, never silently no-ops): **Psywave** (variable — draws RNG), the OHKO
/// moves **Fissure / Horn Drill / Guillotine** (accuracy-gated instakill + the level
/// gate), **Counter / Mirror Coat / Bide** (reactive), **Endeavor** (sets hp to the
/// user's). They are NOT in this set → the caller's fixed-damage fail-loud guard fires.
fn fixed_damage_amount(
    move_id: &str,
    attacker: &crate::state::MonState,
    defender: &crate::state::MonState,
) -> Option<u16> {
    Some(match move_id {
        // `damage: 'level'` — the user's level.
        "seismictoss" | "nightshade" => attacker.level as u16,
        // `damage: <int>` — a flat fixed number.
        "sonicboom" => 20,
        "dragonrage" => 40,
        // `damageCallback` — half the TARGET's current HP, min 1 (the sub, if any,
        // absorbs this NUMBER; the halving reads the MON's hp, not the sub's).
        "superfang" => (defender.hp / 2).max(1),
        _ => return None,
    })
}

/// Whether `move_id` (already `to_id`-normalized) is a FIXED-DAMAGE / FIXED-FORMULA
/// move — one whose damage BYPASSES `getDamage` (a `damage:` / `damageCallback` move).
/// This set is the ROUTING gate in `run_move`: a `true` sends the move to
/// `run_fixed_damage_move`, which handles the MODELED ones bit-for-bit and PANICS
/// fail-loud on the DEFERRED ones (never a silent `base_power == 0` no-op / desync).
///
/// It is the UNION of the modeled set (`fixed_damage_amount` — Seismic Toss / Night
/// Shade / Sonic Boom / Dragon Rage / Super Fang) AND the DEFERRED fixed-damage family
/// (Psywave — variable, draws RNG; the OHKO moves Fissure / Horn Drill / Guillotine —
/// accuracy-gated instakill + a level gate; Counter / Mirror Coat / Bide — reactive;
/// Endeavor — sets hp to the user's). Listing the deferred ones here (not relying on the
/// `base_power == 0` fall-through) makes their exclusion EXPLICIT + FAIL-LOUD: a real
/// team that carries one PANICS in `run_fixed_damage_move` instead of quietly desyncing.
fn is_fixed_damage_move(move_id: &str) -> bool {
    matches!(
        move_id,
        // Modeled (bit-for-bit):
        "seismictoss" | "nightshade" | "sonicboom" | "dragonrage" | "superfang"
        // Deferred (fail-loud in run_fixed_damage_move):
        | "psywave" | "fissure" | "horndrill" | "guillotine"
        | "counter" | "mirrorcoat" | "bide" | "endeavor"
    )
}

/// Whether `move_id` (already `to_id`-normalized) carries the gen3 `flags.defrost`
/// (`gen3_defrost_v1`): a FROZEN user of such a move still draws the 1/5 thaw roll,
/// but on a FAILED roll it PROCEEDS anyway and is thawed draw-free by
/// `frz.onModifyMove` (see `on_before_move`'s FREEZE arm). Sacred Fire and Flame
/// Wheel are the ONLY two gen3 defrost carriers (probe
/// `harness/probe_sacredfire_defrost.js` prints the resolved flags; Flare Blitz is
/// gen4). Id-gated per the `is_fixed_damage_move` precedent — `gen3_moves.json`
/// carries no defrost flag (`contact`/`sound` are the only extracted flags).
fn is_defrost_move(move_id: &str) -> bool {
    matches!(move_id, "sacredfire" | "flamewheel")
}

/// Whether a modeled status move performs the gen-3 MOVE-TYPE immunity check
/// (`runImmunity(move)` in `tryMoveHit`). A Status move defaults to
/// `move.ignoreImmunity = (category === 'Status')` = `true` (type immunity IGNORED),
/// so type immunity matters ONLY for the two gen-3 status moves that explicitly set
/// `ignoreImmunity: false`: **Thunder Wave** (Electric → Ground immune) and **Glare**
/// (Normal → Ghost immune; gen-3 `data/mods/gen3/moves.ts`). Every other modeled
/// status move ignores type immunity (its status-type/ability immunity is resolved in
/// `try_set_status`).
fn status_move_checks_type_immunity(move_id: &str) -> bool {
    matches!(move_id, "thunderwave" | "glare")
}

// NOTE (`gen3_status_immune_v1`): the STATUS_IMMUNE ability class (Limber/Insomnia/Vital
// Spirit/Immunity/Water Veil/Magma Armor) is now DATA-DRIVEN — read from
// `AbilityData.status_immune` via `BattleState::status_immune_of` inside `try_set_status`,
// with the `onSetStatus`-vs-`onImmunity` blocking phase settled by
// `harness/probe_statusimmune_*.js`. The old hardcoded `status_ability_immune` +
// `ability_has_on_set_status` match-arms are REMOVED (the latter's "size-3 shuffle" fail-loud
// was WRONG — probe-proven the ability handler sorts into its own speed group so the 2-clause
// tie stays size-2). The fail-loud for a genuinely UNMODELED onSetStatus ability lives in
// `BattleState::ability_unmodeled_on_set_status`.

/// Whether `ability_id` (already `to_id`-normalized) blocks a FOE stat-DROP on the
/// boost-array index `stat_idx` (`[atk, def, spa, spd, spe, accuracy, evasion]`) — the
/// gen-3 `onTryBoost` immunities (DRAW-FREE): Clear Body / White Smoke block ALL
/// stat-drops; Hyper Cutter blocks ATK only; Keen Eye blocks ACCURACY only. (Showdown
/// only blocks NEGATIVE deltas; the caller passes only foe stat-drops here, so the sign
/// is implicit.)
fn stat_drop_blocked(ability_id: &str, stat_idx: usize) -> bool {
    match ability_id {
        "clearbody" | "whitesmoke" => true,
        "hypercutter" => stat_idx == 0,  // atk
        "keeneye" => stat_idx == 5,      // accuracy
        _ => false,
    }
}

/// Whether a move of `t` is in the gen3 mod's Hustle physical-type list
/// (`gen3_accuracy_pipeline_v1`). The RESOLVED gen3 `hustle.onSourceModifyAccuracy` gates
/// its ×0.8 on `physicalTypes.includes(move.type)` — the gen1-3 TYPE-based physical/special
/// split — NOT `move.category`. Probe-confirmed vs the resolved dist
/// (`harness/probe_accuracy_tohit.js`): a Normal move (Tackle) drops to ×0.8, an Electric
/// move (Thunder) is unaffected.
fn hustle_boosts_accuracy_type(t: Type) -> bool {
    matches!(
        t,
        Type::Normal
            | Type::Fighting
            | Type::Flying
            | Type::Poison
            | Type::Ground
            | Type::Rock
            | Type::Bug
            | Type::Ghost
            | Type::Steel
    )
}

/// The accuracy CHAIN modifier applied at the END of `runEvent('ModifyAccuracy')`
/// (`gen3_accuracy_pipeline_v1`) — `modify(value, event.modifier)` where the modifier is
/// the accumulated `chainModify` product. Bit-identical to the sim's fixed-point
/// (`battle.ts::chainModify` accumulate + `modify` apply): each `chainModify([n,d])`
/// contributes `nextMod = trunc(n*4096/d)`, accumulated `acc = (acc*nextMod + 2048) >> 12`,
/// then `modify(value, acc/4096) = trunc((trunc(value*acc) + 2048 - 1) / 4096)`. The CALLER
/// applies the runEvent integer-guard (only invoked when `value` is a non-negative integer);
/// this is the same math as `damage::chain_modify` but kept local to the accuracy path (the
/// guard placement differs — a stat/BP chain always applies, the accuracy chain is guarded).
fn accuracy_chain_modify(value: u64, mods: &[(u64, u64)]) -> u64 {
    if mods.is_empty() {
        return value;
    }
    let mut acc: u64 = 4096;
    for &(num, den) in mods {
        let next = num * 4096 / den; // trunc(num*4096/den) — integer division
        acc = (acc * next + 2048) >> 12;
    }
    (value * acc + 2048 - 1) / 4096
}

/// `calculateStat`'s boost-table application (`pokemon.ts:597-640`):
/// `boost>=0 -> floor(stat * boostTable[boost])`, `boost<0 -> floor(stat /
/// boostTable[-boost])`, table `[1,1.5,2,2.5,3,3.5,4]`. Used for the speed tie key.
fn apply_boost(stat: u32, boost: i8) -> u32 {
    let b = boost.clamp(-6, 6);
    // (num, den): [1, 3/2, 2, 5/2, 3, 7/2, 4].
    const TABLE: [(u32, u32); 7] = [(1, 1), (3, 2), (2, 1), (5, 2), (3, 1), (7, 2), (4, 1)];
    if b >= 0 {
        let (n, d) = TABLE[b as usize];
        stat * n / d
    } else {
        let (n, d) = TABLE[(-b) as usize];
        stat * d / n
    }
}

/// The species' gen-3 types (from the dex). Empty if the species is unknown (a
/// data gap — the damage calc treats no types as no STAB / neutral effectiveness).
/// NOTE: in-battle CURRENT-type reads must go through [`mon_types`] (the ONE
/// type-read choke point honoring a Color Change `types_override`); this raw
/// species read is only for a mon-less context (never an active's live types).
fn species_types(species_id: &str, dex: &Dex) -> Vec<Type> {
    dex.species(species_id).map(|s| s.types.clone()).unwrap_or_default()
}

/// A mon's CURRENT in-battle types — the ONE type-read choke point
/// (`gen3_ability_batch4_v1`): a Color Change `types_override` (the sim's
/// `pokemon.setType`) REPLACES the species types for every live read (STAB, chart
/// effectiveness, status type-immunity, sand-chip immunity, Magnet Pull's Steel
/// gate, Leech Seed's Grass gate); otherwise the species types. Probe
/// `probe_colorchange_rng.js` (an EQ into an Electric-overridden Kecleon is
/// super-effective; a Rock-overridden Kecleon is sand-immune).
fn mon_types(mon: &crate::state::MonState, dex: &Dex) -> Vec<Type> {
    match &mon.types_override {
        Some(t) => t.clone(),
        None => species_types(&mon.species_id, dex),
    }
}

/// The holder-species gate for a SPECIES_STAT item (`ItemData.stat_mods.only_species`
/// — Thick Club Cubone/Marowak, Light Ball Pikachu, DeepSea* Clamperl, Metal Powder
/// Ditto, Soul Dew Lati@s). Empty ⇒ unconditional (Choice Band). `untransformed_only`
/// (Metal Powder) is always satisfied — the port has no Transform.
fn species_gate_passes(mods: &crate::dex::StatMods, species_id: &str) -> bool {
    mods.only_species.is_empty() || {
        let sid = to_id(species_id);
        mods.only_species.iter().any(|s| *s == sid)
    }
}

/// Resolve the attacker's STAT-event modifiers (`runEvent('ModifyAtk'/'ModifySpA')`)
/// from the DEX's `gen3_item_mechanics_v1` fields — the DATA-DRIVEN path that replaced
/// the hardcoded per-id match arm (the drift that let MODELED_ITEMS list bows/incenses
/// the engine never priced). Members:
///   - `stat_mods` on the category's offensive stat, species-gated (Choice Band ×1.5
///     unconditional-physical; Thick Club/Light Ball/DeepSeaTooth ×2; Soul Dew SpA ×1.5)
///   - `type_boost` at `fold=stat` when the move type matches (the ×1.1 family +
///     Sea Incense ×1.05) — the gen3-mod `onModifyAtk`/`onModifySpA` handlers
///   - the ABILITY DMG_MOD `fold=atk` members (`gen3_item_mechanics_v1` ability side):
///     Huge/Pure Power ×2 unconditional, Guts ×1.5 when the attacker is statused. These
///     are `onModifyAtk chainModify` handlers → the SAME accumulate-once ModifyAtk chain
///     as the items (probed: Guts ×1.5 + Choice Band ×1.5 = one ×2.25 chain, not two
///     rounds). They fold ONLY on a PHYSICAL move (ModifyAtk touches only the Atk stat);
///     Hustle's `direct` shape (a `this.modify`, plus an acc side) stays UNWIRED here.
/// `attacker_statused` gates Guts (any major status: brn/psn/tox/par/slp/frz). An
/// unknown/absent item+ability (or a typeless '???' move for the type gate) resolves to
/// no mods, exactly like the old `_ => {}`.
fn resolve_atk_stat_mods(
    item: &str,
    ability: &str,
    species_id: &str,
    move_type: Option<Type>,
    category: MoveCategory,
    attacker_statused: bool,
    dex: &Dex,
) -> Vec<AtkStatMod> {
    let mut mods = Vec::new();
    if let Some(it) = dex.item(item) {
        if let Some(sm) = &it.stat_mods {
            if species_gate_passes(sm, species_id) {
                let ratio = match category {
                    MoveCategory::Physical => sm.atk,
                    MoveCategory::Special => sm.spa,
                    MoveCategory::Status => None,
                };
                if let Some((num, den)) = ratio {
                    mods.push(AtkStatMod::Item { num, den });
                }
            }
        }
        if let Some(tb) = &it.type_boost {
            if tb.fold == TypeBoostFold::Stat && move_type == Some(tb.type_) {
                mods.push(AtkStatMod::Item { num: tb.num, den: tb.den });
            }
        }
    }
    // Ability DMG_MOD `fold=atk` (Huge/Pure Power / Guts) — PHYSICAL only, into the
    // same ModifyAtk chain. `direct` (Hustle) is deferred to the accuracy phase.
    if category == MoveCategory::Physical {
        if let Some(m) = dex.ability(ability).and_then(|a| a.dmg_mod.as_ref()) {
            if m.fold == DmgFold::Atk && !m.direct && (!m.when_statused || attacker_statused) {
                mods.push(AtkStatMod::Item { num: m.num, den: m.den });
            }
        }
    }
    mods
}

/// Resolve the DEFENDER's STAT-event modifiers (`runEvent('ModifyDef'/'ModifySpD')`)
/// — the defense-side SPECIES_STAT items: DeepSeaScale (SpD ×2 Clamperl), Metal Powder
/// (Def ×2 untransformed Ditto), Soul Dew (SpD ×1.5 Lati@s) — PLUS the ability DMG_MOD
/// `fold=def` member Marvel Scale (Def ×1.5 when the DEFENDER is statused;
/// `gen3_item_mechanics_v1` ability side, `onModifyDef chainModify(1.5)`). Folded into
/// the defensive stat AFTER the boost table, BEFORE the gen<=4 Explosion Def-halve (the
/// `getDamage` order). Marvel Scale touches only the physical Def (ModifyDef fires on a
/// physical hit); no gen3 item type-boosts a DEFENSE stat. `defender_statused` gates
/// Marvel Scale (any major status). (Probed: a burned Marvel Scale Milotic takes ×2/3 a
/// physical hit — Def ×1.5 — while a burned NON-Marvel mon is unchanged.)
fn resolve_def_stat_mods(
    item: &str,
    ability: &str,
    species_id: &str,
    category: MoveCategory,
    defender_statused: bool,
    dex: &Dex,
) -> Vec<AtkStatMod> {
    let mut mods = Vec::new();
    if let Some(it) = dex.item(item) {
        if let Some(sm) = &it.stat_mods {
            if species_gate_passes(sm, species_id) {
                let ratio = match category {
                    MoveCategory::Physical => sm.def,
                    MoveCategory::Special => sm.spd,
                    MoveCategory::Status => None,
                };
                if let Some((num, den)) = ratio {
                    mods.push(AtkStatMod::Item { num, den });
                }
            }
        }
    }
    // Ability DMG_MOD `fold=def` (Marvel Scale) — the physical Def only, when statused.
    if category == MoveCategory::Physical {
        if let Some(m) = dex.ability(ability).and_then(|a| a.dmg_mod.as_ref()) {
            if m.fold == DmgFold::Def && !m.direct && (!m.when_statused || defender_statused) {
                mods.push(AtkStatMod::Item { num: m.num, den: m.den });
            }
        }
    }
    mods
}

/// Resolve the attacker-item + attacker-ability BASE-POWER-phase modifiers
/// (`runEvent('BasePower')`):
///   - the ITEM `type_boost` at `fold=basePower` (the gen4-named incenses,
///     `chainModify` into the event's ONE accumulated modifier alongside defender Thick
///     Fat) or `fold=basePowerDirect` (the gen2 bows, the DIRECT ×1.1 float return — see
///     [`BpMod::Direct`] for the runEvent-tail skip semantics);
///   - the ABILITY DMG_MOD PINCH family (`gen3_item_mechanics_v1` ability side) —
///     Torrent/Blaze/Overgrow/Swarm: an `onBasePower chainModify(1.5)` for the ability's
///     type when the user is at `hp <= maxhp/3`. The pinch condition is passed as
///     `attacker_in_pinch` (the caller computes `3*hp <= maxhp`, bit-exactly the sim's
///     integer-hp `hp <= maxhp/3` float compare — probe-verified at the maxhp=341 float
///     boundary). It joins the SAME accumulated BP chain as the incenses/Thick Fat.
/// `fold=sourceBasePower` (Thick Fat) is NOT resolved here — the port models it via
/// `DamageContext::defender_thick_fat` (it is a DEFENDER handler on the ATTACKER's BP).
fn resolve_bp_mods(
    item: &str,
    ability: &str,
    move_type: Option<Type>,
    attacker_in_pinch: bool,
    dex: &Dex,
) -> Vec<BpMod> {
    let mut mods = Vec::new();
    if let Some(it) = dex.item(item) {
        if let Some(tb) = &it.type_boost {
            if move_type == Some(tb.type_) {
                match tb.fold {
                    TypeBoostFold::Stat => {}
                    TypeBoostFold::BasePower => mods.push(BpMod::Chain(tb.num, tb.den)),
                    TypeBoostFold::BasePowerDirect => mods.push(BpMod::Direct(tb.num, tb.den)),
                }
            }
        }
    }
    // Ability DMG_MOD PINCH (Torrent/Blaze/Overgrow/Swarm): BP ×1.5 for the ability's
    // type at hp<=maxhp/3. A chain member (joins the incense/Thick-Fat accumulate-once).
    if let (Some(mt), Some(m)) = (move_type, dex.ability(ability).and_then(|a| a.dmg_mod.as_ref())) {
        if m.fold == DmgFold::BasePower && m.pinch && attacker_in_pinch && m.types.contains(&mt) {
            mods.push(BpMod::Chain(m.num, m.den));
        }
    }
    mods
}

/// Resolve a DEFENDER ability immunity / Thick Fat from (ability id, move-type).
/// The type-chart 0× immunities are handled inside `calc_damage`; this covers the
/// The gen3 `onTryHit`-class ability immunities (POST-accuracy): Flash Fire (Fire),
/// Water Absorb (Water), Volt Absorb (Electric). These resolve AFTER the accuracy roll,
/// so on a MISS the sim reports `[miss]`+`-miss` (NOT `-immune`) and on a LANDED absorb
/// reports `|-immune|<t>|[from] ability: <Name>` (F2/F3, `harness/probe_f1_f2_f3_lines.js`
/// + `probe_levitate_miss.js`). Contrast Levitate + type-chart 0×, which resolve at the
/// PRE-accuracy `runImmunity` and ALWAYS report a plain `-immune` (probe: 40/40 immune on
/// a missable Ground move). Returns the ability's DISPLAY name (for the `[from] ability:`
/// attr) when the (ability, move type) pair is a TryHit-class absorb-immunity, else None.
/// Flash Fire is excluded here — it is handled by the dedicated `ff_case` arm (its landed
/// form is an ARM `-start`, not a `[from]` immune, and it needs the frz/faint guards).
fn tryhit_absorb_ability(ability_id: &str, move_type: Option<Type>) -> Option<&'static str> {
    match (ability_id, move_type) {
        ("waterabsorb", Some(Type::Water)) => Some("Water Absorb"),
        ("voltabsorb", Some(Type::Electric)) => Some("Volt Absorb"),
        _ => None,
    }
}

/// ability ones (Levitate / Flash Fire / Water&Volt Absorb) + Thick Fat (Ice/Fire
/// ×0.5). Mirrors `tests/damage_test.rs::resolve_defender`.
///
/// `defender_frozen` (`gen3_ff_frozen_no_absorb_v1`): the resolved gen3
/// `flashfire.onTryHit` returns early for a `frz` holder — a FROZEN Flash Fire mon is
/// NOT fire-immune (the move hits with full draws and its fire-move thaw then cures the
/// freeze; the absorb volatile never arms). Only Flash Fire reads the flag.
fn resolve_defender_ability(
    ability_id: &str,
    move_type: Option<Type>,
    defender_frozen: bool,
) -> (bool, bool) {
    let immune = match ability_id {
        "levitate" => move_type == Some(Type::Ground),
        "flashfire" => move_type == Some(Type::Fire) && !defender_frozen,
        "waterabsorb" => move_type == Some(Type::Water),
        "voltabsorb" => move_type == Some(Type::Electric),
        _ => false,
    };
    let thick_fat =
        ability_id == "thickfat" && (move_type == Some(Type::Ice) || move_type == Some(Type::Fire));
    (immune, thick_fat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::battle::{BattleOptions, PackedTeam, PlayerOptions};

    fn dex() -> Dex {
        Dex::for_gen(3)
    }

    fn opts(p1: &str, p2: &str, seed: &str) -> BattleOptions {
        BattleOptions {
            format_id: "gen3ou".to_string(),
            seed: Some(format!("[{seed}]")),
            p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(p1.to_string()) },
            p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(p2.to_string()) },
        }
    }

    /// gen3customgame opts (NO clauses → `sleep_clause` OFF → no SetStatus handler-sort
    /// shuffle), for the status-move unit gates that isolate the bare draw model.
    fn opts_cg(p1: &str, p2: &str, seed: &str) -> BattleOptions {
        BattleOptions { format_id: "gen3customgame".to_string(), ..opts(p1, p2, seed) }
    }

    // crit_ratio is now DATA-DRIVEN from the dex (`critRatio` in gen3_moves.json):
    // normal moves are 1 (1/16), the gen-3 high-crit set is 2 (1/8). This includes
    // aircutter/blazekick/leafblade — which the old hardcoded allowlist MISSED.
    #[test]
    fn crit_ratio_is_data_driven() {
        let d = crate::dex::Dex::for_gen(3);
        let cr = |id: &str| d.moves(id).unwrap().crit_ratio;
        assert_eq!(cr("earthquake"), 1);
        assert_eq!(cr("surf"), 1);
        assert_eq!(cr("slash"), 2);
        assert_eq!(cr("crabhammer"), 2);
        // The three the hardcoded list omitted (the bug the data-drive fixes):
        assert_eq!(cr("aircutter"), 2);
        assert_eq!(cr("blazekick"), 2);
        assert_eq!(cr("leafblade"), 2);
    }

    // effective_speed: a +1 boost floors via the table; paralysis halves (×0.5).
    #[test]
    fn effective_speed_boost_and_para() {
        let d = dex();
        // Jolteon spe stat is large; use a known packed set.
        let jolteon =
            "Jolteon||leftovers|voltabsorb|thunderbolt,shadowball,batonpass,hiddenpowerice|Timid|,,,252,4,252|||||";
        let snorlax =
            "Snorlax||leftovers|immunity|bodyslam,earthquake,rest,curse|Adamant|252,252,4,,,|||||";
        let mut state =
            BattleState::start(&opts(jolteon, snorlax, "1,2,3,4"), &d).expect("start");
        let base = state.effective_speed(0, 0, &d);
        assert!(base > 0, "jolteon has positive speed");
        // +1 boost: floor(base * 1.5).
        state.sides[0].pokemon[0].boosts[4] = 1;
        assert_eq!(state.effective_speed(0, 0, &d), base * 3 / 2, "+1 spe = floor(base*1.5)");
        // paralysis: gen3 ×0.25 via modify(base,1,4) = floor((base*1024+2047)/4096)
        // (gen4-inherited chainModify(0.25); the +2047 round, NOT base*0.5/0.25 plain).
        state.sides[0].pokemon[0].boosts[4] = 0;
        state.sides[0].pokemon[0].status = Some(Status::Paralysis);
        let expected = ((base as u64 * 1024 + 2047) / 4096) as u32;
        assert_eq!(state.effective_speed(0, 0, &d), expected, "para = modify(base,1,4) (gen3 ×0.25)");
    }

    // WEATHER_SPEED (`gen3_ability_batch1_v1`): a Swift Swim mon's speed is ×2 in EFFECTIVE
    // rain, composing with a boost (boost FIRST) then the para ×0.25 as ONE chain. Pinned to
    // the sim's getStat('spe') (harness/probe_weather_speed_stat.js: Kingdra raw 206 →
    // 412 rain, 103 rain+para, 618 rain+1boost, 154 rain+1boost+para). A Cloud Nine / Air Lock
    // mon on the field SUPPRESSES the ×2 (effective_weather = None). All draw-free.
    #[test]
    fn weather_speed_swift_swim_folds_x2_in_rain() {
        use crate::state::Weather;
        let d = dex();
        // Kingdra Swift Swim, 0 EV Serious → raw spe 206 (matches the probe). The packed
        // format is `species|nick|item|ability|moves|nature|evs|gender|ivs|shiny|level`.
        let kingdra = "Kingdra||leftovers|swiftswim|surf,icebeam,rest,raindance|Serious||||||";
        let foe = "Snorlax||leftovers|shellarmor|bodyslam,rest,curse,earthquake|Serious||||||";
        let mut state = BattleState::start(&opts(kingdra, foe, "1,2,3,4"), &d).expect("start");
        let raw = state.effective_speed(0, 0, &d);
        assert_eq!(raw, 206, "no weather → raw spe (Swift Swim inert)");
        // Rain up (no negater): ×2.
        state.field.weather = Some(Weather::Rain);
        assert_eq!(state.effective_speed(0, 0, &d), 412, "Swift Swim in rain = raw×2");
        // Rain + para: modify(206, 2×0.25=0.5) as ONE chain = 103.
        state.sides[0].pokemon[0].status = Some(Status::Paralysis);
        assert_eq!(state.effective_speed(0, 0, &d), 103, "Swift Swim rain+para = one chain ×0.5");
        // Rain + +1 boost (no para): boost FIRST floor(206*1.5)=309, then ×2 = 618.
        state.sides[0].pokemon[0].status = None;
        state.sides[0].pokemon[0].boosts[4] = 1;
        assert_eq!(state.effective_speed(0, 0, &d), 618, "Swift Swim rain +1 = boost then ×2");
        // Rain + +1 boost + para: floor(206*1.5)=309, then modify(309, 0.5) = 154.
        state.sides[0].pokemon[0].status = Some(Status::Paralysis);
        assert_eq!(state.effective_speed(0, 0, &d), 154, "Swift Swim rain +1 +para");
        // WEATHER_NEGATE: give the FOE Cloud Nine → the ×2 is suppressed (effective weather
        // None), so with +1 boost + para it drops to boost-then-para only = floor(309*0.25)…
        state.sides[0].pokemon[0].boosts[4] = 0;
        state.sides[0].pokemon[0].status = None;
        let golduck = crate::team::unpack("Golduck||leftovers|cloudnine|surf,rest,calmmind,icebeam|Serious||||||", &d).unwrap().remove(0);
        state.sides[1].pokemon[0] = crate::state::MonState::from_set(golduck, 1, &d).unwrap();
        // Rain still raw in the field, but a Cloud Nine foe suppresses it → back to raw 206.
        assert_eq!(state.effective_speed(0, 0, &d), 206, "Cloud Nine foe suppresses the weather ×2");
    }

    // effective_accuracy (gen3_accuracy_pipeline_v1): the effAcc reaching random(100)<eff,
    // pinned to the SIM-CAPTURED values from harness/probe_accuracy_tohit.js +
    // probe_accuracy_intguard.js (the ONLY oracle). Covers the stage table, each accMod
    // member, and the runEvent integer-guard (a chain member SKIPPED when acc is a float).
    #[test]
    fn effective_accuracy_matches_sim_probe() {
        let d = dex();
        // p1 attacker / p2 defender. Tackle is gen3 accuracy 95 (probe-confirmed).
        let atk = |item: &str, abil: &str| {
            format!("Tauros||{}|{}|tackle,,,|Hardy|,,,,,|||||", item, abil)
        };
        let deff = |item: &str, abil: &str| {
            format!("Snorlax||{}|{}|tackle,,,|Hardy|252,,,,,|||||", item, abil)
        };
        // eff() builds a fresh battle so boosts/weather don't leak between cases.
        let eff = |ai: &str, aa: &str, di: &str, da: &str,
                   acc_stage: i8, eva_stage: i8, sand: bool| -> f64 {
            let mut s =
                BattleState::start(&opts(&atk(ai, aa), &deff(di, da), "1,2,3,4"), &d).expect("start");
            s.sides[0].pokemon[0].boosts[5] = acc_stage; // attacker accuracy stage
            s.sides[1].pokemon[0].boosts[6] = eva_stage; // defender evasion stage
            if sand {
                s.field.weather = Some(Weather::Sand);
            }
            // Tackle is Normal; move.accuracy 95.
            s.effective_accuracy(0, 0, 1, 0, 95, Some(Type::Normal), &d)
        };
        let approx = |got: f64, want: f64| {
            assert!((got - want).abs() < 1e-9, "effAcc {got} != {want}");
        };
        // --- STAGE TABLE (no items/abilities) ---
        approx(eff("", "noability", "", "noability", 0, 0, false), 95.0); // baseline
        approx(eff("", "noability", "", "noability", -1, 0, false), 71.25); // 95/(4/3)
        approx(eff("", "noability", "", "noability", -2, 0, false), 57.0); // 95/(5/3)
        approx(eff("", "noability", "", "noability", 0, 1, false), 71.25); // eva+1: 95/(4/3)
        approx(eff("", "noability", "", "noability", 0, -1, false), 95.0 * 4.0 / 3.0); // eva-1
        approx(eff("", "noability", "", "noability", 1, 0, false), 95.0 * 4.0 / 3.0); // acc+1
        approx(eff("", "noability", "", "noability", -3, 0, false), 47.5); // 95/2
        // --- accMod members (no stage) ---
        approx(eff("", "noability", "brightpowder", "noability", 0, 0, false), 85.5); // 95*0.9
        approx(eff("", "noability", "laxincense", "noability", 0, 0, false), 90.25); // 95*0.95
        approx(eff("", "compoundeyes", "", "noability", 0, 0, false), 123.0); // chainModify(1.3)
        approx(eff("", "noability", "", "sandveil", 0, 0, true), 76.0); // chainModify(0.8) in sand
        approx(eff("", "noability", "", "sandveil", 0, 0, false), 95.0); // Sand Veil NO sand → no-op
        approx(eff("", "hustle", "", "noability", 0, 0, false), 76.0); // Normal is a physical-type
        // --- INTEGER GUARD: a chain member SKIPPED when acc is a non-integer float ---
        // eva-1 makes 95*4/3=126.66 (float) → Bright Powder DIRECT *0.9 applies (114.0).
        approx(eff("", "noability", "brightpowder", "noability", 0, -1, false), 95.0 * 4.0 / 3.0 * 0.9);
        // acc+1 makes 126.66 (float) → Hustle chain SKIPPED → stays 126.66.
        approx(eff("", "hustle", "", "noability", 1, 0, false), 95.0 * 4.0 / 3.0);
        // acc-3 → 47.5 (float) THEN Bright Powder DIRECT *0.9 = 42.75.
        approx(eff("", "noability", "brightpowder", "noability", -3, 0, false), 42.75);

        // --- Hustle does NOT touch a non-physical-type move (Thunder=Electric).
        let ampharos = "Ampharos||leftovers|hustle|thunder,,,|Hardy|,,,,,|||||";
        let s = BattleState::start(&opts(ampharos, &deff("", "noability"), "1,2,3,4"), &d)
            .expect("start");
        let thunder_eff = s.effective_accuracy(0, 0, 1, 0, 70, Some(Type::Electric), &d);
        assert!((thunder_eff - 70.0).abs() < 1e-9, "Hustle unaffected on Electric: {thunder_eff}");
    }

    // A faster mon that OHKOs the slower: the slower draws nothing (faint-skip), and
    // the trailing Quick Claw is NOT drawn (a faint defers endTurn).
    #[test]
    fn faster_ohko_skips_slower_and_quickclaw() {
        let d = dex();
        // Strong fast attacker vs frail slow target. Aerodactyl (spe 130 base) Rock
        // Slide vs Magikarp — but Rock Slide has a flinch secondary (no secondary
        // draw modeled here, fine — we don't draw it). Use a no-secondary move:
        // Aerodactyl Earthquake vs Magikarp (frail). Earthquake OHKOs Magikarp.
        let aero =
            "Aerodactyl||choiceband|rockhead|earthquake,rockslide,doubleedge,hiddenpowerflying|Adamant|,252,,,4,252|||||";
        let karp = "Magikarp||leftovers|swiftswim|tackle,splash,,|Adamant|,252,,,4,252|||||";
        let mut state = BattleState::start(&opts(aero, karp, "1,2,3,4"), &d).expect("start");
        // p1 Earthquake (slot 0), p2 Tackle (slot 0). Aerodactyl is much faster.
        let res = state.run_turn(0, 0, &d);
        // Magikarp fainted.
        assert!(state.sides[1].pokemon[0].fainted, "Magikarp KO'd by Earthquake");
        assert!(res.outcome[0].acted, "Aerodactyl acted");
        assert!(!res.outcome[1].acted, "Magikarp KO'd before acting ⇒ did not move");
        assert!(!res.quick_claw_drawn, "a faint defers the end-of-turn Quick Claw");
        // Aerodactyl untouched (Magikarp never moved).
        assert_eq!(
            state.sides[0].pokemon[0].hp, state.sides[0].pokemon[0].maxhp,
            "Aerodactyl unscathed (slower mon never acted)"
        );
    }

    // An IMMUNE move draws ONLY its accuracy roll (no crit/damage) — the draw-count
    // crux. Flygon (Ground/Dragon) Earthquake into Skarmory (Steel/Flying) is
    // Ground-immune. We isolate JUST Flygon's immune move (no foe move, no
    // quickclaw) via the same-seed prng-advance: an immune hit must advance the
    // prng by EXACTLY one `randomChance(100,100)` (accuracy), NOT three. The full
    // ordered draw-count under a real turn is the golden's job
    // (`immune_eq_vs_skarmory` seed parity in `tests/turn_test.rs`).
    #[test]
    fn immune_move_draws_only_accuracy() {
        let d = dex();
        let flygon =
            "Flygon||leftovers|levitate|earthquake,rockslide,fireblast,dragonclaw|Jolly|,252,,,4,252|||||";
        let skarmory =
            "Skarmory||leftovers|keeneye|drillpeck,spikes,roar,rest|Impish|252,,252,,4,|||||";
        let mut state =
            BattleState::start(&opts(flygon, skarmory, "1,2,3,4"), &d).expect("start");
        let before = state.prng_seed();
        // Drive ONLY Flygon's immune Earthquake (side 0 active, slot 0) through the
        // private `run_move` path via an action; assert it drew exactly accuracy.
        let action = MoveAction { side: 0, slot: 0, move_index: 0, struggle: false };
        let res = state.run_move(action, true, true, &d);
        assert!(!res.missed && !res.crit, "immune EQ neither missed nor crit (it hit-but-immune)");
        assert!(!res.landed, "an immune move did NOT land ⇒ no in-tryMoveHit Update shuffle");
        // The oracle: exactly ONE accuracy draw from `before`.
        let mut oracle = crate::prng::Prng::new(before.as_str());
        let _ = oracle.random_chance(100, 100); // EQ accuracy (100) — then immune, stop.
        assert_eq!(
            state.prng_seed(),
            oracle.get_seed(),
            "an immune move draws ONLY accuracy (1 draw), NOT accuracy+crit+damage (3)"
        );
        // Skarmory untouched by the immune Earthquake.
        assert_eq!(
            state.sides[1].pokemon[0].hp, state.sides[1].pokemon[0].maxhp,
            "Skarmory unscathed by the immune Earthquake"
        );
    }

    // A clean both-survive distinct-speed turn draws the trailing Quick Claw.
    #[test]
    fn both_survive_draws_quick_claw() {
        let d = dex();
        let suicune =
            "Suicune||leftovers|pressure|surf,icebeam,calmmind,rest|Bold|252,,252,4,,|||||";
        let blissey =
            "Blissey||leftovers|naturalcure|seismictoss,softboiled,toxic,aromatherapy|Calm|252,,,,252,4|||||";
        // Suicune Surf (slot 0) vs Blissey ... Blissey's slot 0 is Seismic Toss (a
        // fixed-damage move; out of damaging scope). Give Blissey a damaging slot:
        let blissey2 =
            "Blissey||leftovers|naturalcure|icebeam,softboiled,toxic,aromatherapy|Calm|252,,,,252,4|||||";
        let mut state =
            BattleState::start(&opts(suicune, blissey2, "1,2,3,4"), &d).expect("start");
        let res = state.run_turn(0, 0, &d);
        assert!(res.outcome[0].acted && res.outcome[1].acted, "both acted");
        assert!(!state.sides[0].pokemon[0].fainted && !state.sides[1].pokemon[0].fainted);
        assert!(res.quick_claw_drawn, "no faint ⇒ Quick Claw drawn");
        // Both took some damage.
        assert!(state.sides[1].pokemon[0].hp < state.sides[1].pokemon[0].maxhp, "Blissey hit");
        let _ = suicune; // silence unused (kept for readability)
        let _ = blissey;
    }

    // ===================================================================
    // SWITCHING + post-faint replacement + win/loss (`run_full_battle`).
    // Behavioral unit tests — the per-seed STATE+SEED differential to game-end is
    // the harness/`fullbattle_test.rs` job; these pin the mechanics deterministically.
    // ===================================================================

    // A VOLUNTARY switch resolves BEFORE the foe's move (order 103 < 200): a slow
    // mon pivots, the entrant takes the foe's hit (not the mon that switched out).
    #[test]
    fn voluntary_switch_resolves_before_move() {
        let d = dex();
        // p1 slow Snorlax → switches to Suicune; p2 fast Jolteon attacks (Surf, no
        // secondary). The switch must complete first, so Suicune (not Snorlax) is hit.
        let snorlax_team =
            "Snorlax||leftovers||earthquake,bodyslam|Adamant|252,252,,,,|||||]Suicune||leftovers||surf,icebeam|Bold|252,,252,,,|||||";
        let jolt = "Jolteon||leftovers||surf,swift|Timid|,,,252,,252|||||";
        let mut state =
            BattleState::start(&opts(snorlax_team, jolt, "1,2,3,4"), &d).expect("start");
        let script = vec![ScriptDecision::both(Choice::Switch(1), Choice::Move(0))];
        let out = state.run_full_battle(&script, &d);
        assert_eq!(out.decisions.len(), 1);
        // p1's active is now Suicune (the switch resolved), and it took Surf damage.
        assert_eq!(state.sides[0].active_species(), "suicune", "p1 switched to Suicune");
        assert!(state.sides[0].active().hp < state.sides[0].active().maxhp, "Suicune (the entrant) took the hit");
        // The switch resolved before the move (first action is the switch's side).
        assert_eq!(out.decisions[0].first_mover, Some(0), "the switch (p1) resolved first");
        assert!(!out.ended, "neither side is out of mons");
    }

    // POST-FAINT single replacement → the battle continues PAST the faint to a WIN.
    // A fast strong attacker OHKOs both of the opp's frail mons; after the first KO
    // the opp replaces, then the second KO wins.
    #[test]
    fn post_faint_replacement_continues_to_win() {
        let d = dex();
        let aero = "Aerodactyl||choiceband|rockhead|earthquake,rockslide|Adamant|,252,,,,252|||||";
        // Two frail p2 mons that Aerodactyl EQ OHKOs (Jolteon, then Gengar — both
        // take neutral/SE EQ; secondary-free).
        let p2 = "Jolteon||none||swift,surf|Timid|,,,252,,252|||||]Gengar||none||surf,icepunch|Timid|,,,252,,252|||||";
        let mut state = BattleState::start(&opts(aero, p2, "1,2,3,4"), &d).expect("start");
        let script = vec![
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // KO p2 lead
            ScriptDecision::one(1, Choice::Switch(1)),              // p2 replaces
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // KO p2's last mon → win
        ];
        let out = state.run_full_battle(&script, &d);
        assert!(out.ended, "battle reached game-end");
        assert_eq!(out.winner, Some(0), "p1 (Aerodactyl) swept to a win");
        assert_eq!(state.sides[1].pokemon_left, 0, "p2 is out of mons");
        // The middle decision was a forced replacement for p2.
        assert!(
            matches!(out.decisions[1].request, RequestKind::ForceSwitch { force: [false, true] }),
            "decision 1 was p2's forced replacement"
        );
    }

    // A last-mon mutual KO (Explosion mirror) is a gen-3 TIE: both pokemon_left hit
    // 0 the same faint protocol → `win(None)` (a tie), and the deciding faint draws
    // no trailing Quick Claw.
    #[test]
    fn last_mon_double_ko_is_a_tie() {
        let d = dex();
        let electrode = "Electrode||none||explosion,thunderbolt|Hasty|,252,,,,252|||||";
        let mut state =
            BattleState::start(&opts(electrode, electrode, "1,2,3,4"), &d).expect("start");
        let script = vec![ScriptDecision::both(Choice::Move(0), Choice::Move(0))];
        let out = state.run_full_battle(&script, &d);
        assert!(out.ended, "the mutual Explosion KO ended the battle");
        assert_eq!(out.winner, None, "both sides out ⇒ gen-3 TIE (win(None))");
        assert_eq!(state.sides[0].pokemon_left, 0);
        assert_eq!(state.sides[1].pokemon_left, 0);
    }

    // The position-swap invariant: after a switch the entrant lives at the active
    // index (0 in singles) and the outgoing mon takes the entrant's old bench slot —
    // so a `Switch(N)` choice keeps referring to the CURRENT array slot (like the
    // sim's `switch N`), and switching back works.
    #[test]
    fn switch_swaps_team_positions() {
        let d = dex();
        let team =
            "Snorlax||leftovers||earthquake,bodyslam|Adamant|252,252,,,,|||||]Suicune||leftovers||surf,icebeam|Bold|252,,252,,,|||||";
        let foe = "Blissey||leftovers||icebeam,softboiled|Calm|252,,,,252,|||||";
        let mut state = BattleState::start(&opts(team, foe, "1,2,3,4"), &d).expect("start");
        assert_eq!(state.sides[0].active_species(), "snorlax", "lead is Snorlax");
        // Switch to slot 1 (Suicune); after the swap Suicune is at index 0 (active),
        // Snorlax at index 1.
        let script = vec![ScriptDecision::both(Choice::Switch(1), Choice::Move(0))];
        let out = state.run_full_battle(&script, &d);
        assert_eq!(out.decisions.len(), 1);
        assert_eq!(state.sides[0].active, 0, "active index stays 0 (gen-3 singles)");
        assert_eq!(state.sides[0].pokemon[0].species_id, "suicune", "entrant at index 0");
        assert_eq!(state.sides[0].pokemon[1].species_id, "snorlax", "outgoing at index 1");
        // Switch BACK to Snorlax (now at slot 1) — proves `switch N` tracks the swap.
        let script2 = vec![ScriptDecision::both(Choice::Switch(1), Choice::Move(0))];
        let _ = state.run_full_battle(&script2, &d);
        assert_eq!(state.sides[0].pokemon[0].species_id, "snorlax", "switched back to Snorlax");
    }

    // ===================================================================
    // SECONDARY effects + onBeforeMove STATUS draws (this step). Behavioral unit
    // tests pinning the new draw COUNT/ORDER deterministically; the per-seed
    // STATE+SEED differential to game-end is `tests/secondary_test.rs`.
    // ===================================================================

    // A PARALYZED mon's onBeforeMove draws EXACTLY ONE randomChance(1,4) BEFORE its
    // accuracy. On a full-para (true) the move ABORTS — no accuracy/crit/dmg/secondary
    // (1 draw total). We seed-advance an oracle to find a full-para seed and assert
    // the draw count + the abort.
    #[test]
    fn paralysis_full_para_draws_only_the_para_roll() {
        let d = dex();
        // A paralyzed Snorlax Body Slam: onBeforeMove randomChance(1,4) first.
        let snorlax =
            "Snorlax||leftovers||bodyslam,earthquake|Adamant|252,252,,,,|||||";
        let blissey =
            "Blissey||leftovers||icebeam,softboiled|Calm|252,,,,252,|||||";
        // Find a seed where the para roll is TRUE (full-para). randomChance(1,4) is
        // random_below(4) < 1, i.e. == 0.
        for s in 0..200u32 {
            let seed = format!("{},{},{},{}", s + 1, s + 2, s + 3, s + 4);
            let mut probe = crate::prng::Prng::new(&seed);
            let full_para = probe.random_chance(1, 4);
            let mut state =
                BattleState::start(&opts(snorlax, blissey, &seed), &d).expect("start");
            state.sides[0].pokemon[0].status = Some(Status::Paralysis);
            let before = state.prng_seed();
            let res = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
            if full_para {
                // ABORTED: not landed, and EXACTLY one draw (the para roll) consumed.
                assert!(!res.landed && !res.missed && !res.crit, "full-para aborts the move");
                let mut oracle = crate::prng::Prng::new(before.as_str());
                let _ = oracle.random_chance(1, 4);
                assert_eq!(
                    state.prng_seed(), oracle.get_seed(),
                    "a full-para move draws ONLY the randomChance(1,4) — no acc/crit/dmg/secondary"
                );
                // Blissey untouched (the paralyzed mon never moved).
                assert_eq!(state.sides[1].pokemon[0].hp, state.sides[1].pokemon[0].maxhp);
                return;
            }
        }
        panic!("no full-para seed in 200 — the para roll never fired");
    }

    // A paralyzed mon that PASSES its para check draws the para roll THEN proceeds to
    // the normal acc/crit/dmg/secondary (the para roll is the NEW LEADING draw).
    #[test]
    fn paralysis_pass_then_full_move_plus_secondary() {
        let d = dex();
        let snorlax = "Snorlax||leftovers||bodyslam,earthquake|Adamant|252,252,,,,|||||";
        let shuckle = "Shuckle||leftovers||tackle,rest|Bold|252,,252,,,|||||";
        for s in 0..200u32 {
            let seed = format!("{},{},{},{}", s + 7, s + 3, s + 11, s + 5);
            let mut probe = crate::prng::Prng::new(&seed);
            let full_para = probe.random_chance(1, 4);
            if full_para {
                continue; // want a PASS
            }
            let mut state =
                BattleState::start(&opts(snorlax, shuckle, &seed), &d).expect("start");
            state.sides[0].pokemon[0].status = Some(Status::Paralysis);
            let before = state.prng_seed();
            let res = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
            assert!(res.landed, "a passed-para Body Slam lands");
            // Oracle: para(1,4) -> accuracy(100,100) -> crit(1,16) -> dmg random(16)
            //         -> secondary random(100). EXACTLY 5 draws.
            let mut oracle = crate::prng::Prng::new(before.as_str());
            let _ = oracle.random_chance(1, 4); // para (pass)
            let _ = oracle.random_chance(100, 100); // accuracy
            let _ = oracle.random_chance(1, 16); // crit
            let _ = oracle.random_below(16); // damage
            let _ = oracle.random_below(100); // secondary par30
            assert_eq!(
                state.prng_seed(), oracle.get_seed(),
                "passed-para Body Slam = para+acc+crit+dmg+secondary (5 draws, in order)"
            );
            return;
        }
        panic!("no para-pass seed found");
    }

    // A landed secondary move (no status on the mover) draws EXACTLY ONE random(100)
    // AFTER the acc/crit/dmg — the NEW TRAILING draw. Body Slam into Shuckle (survives).
    #[test]
    fn landed_secondary_draws_one_random_100_after_damage() {
        let d = dex();
        let tauros = "Tauros||leftovers||bodyslam,earthquake|Jolly|,252,,,,252|||||";
        let shuckle = "Shuckle||leftovers||tackle,rest|Bold|252,,252,,,|||||";
        let mut state = BattleState::start(&opts(tauros, shuckle, "5,6,7,8"), &d).expect("start");
        let before = state.prng_seed();
        let res = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        assert!(res.landed, "Body Slam lands on Shuckle");
        // Oracle: accuracy(100,100) -> crit(1,16) -> dmg(16) -> secondary(100).
        let mut oracle = crate::prng::Prng::new(before.as_str());
        let _ = oracle.random_chance(100, 100);
        let _ = oracle.random_chance(1, 16);
        let _ = oracle.random_below(16);
        let _ = oracle.random_below(100); // the secondary
        assert_eq!(
            state.prng_seed(), oracle.get_seed(),
            "a landed Body Slam = acc+crit+dmg+secondary (the secondary is the 4th draw)"
        );
    }

    // The gen-3 STATUS type-immunity rules (VERIFIED vs the sim): Electric is NOT
    // para-immune; Ice IS frz-immune; Fire IS brn-immune; Poison & Steel ARE
    // psn/tox-immune; slp has no type immunity.
    #[test]
    fn gen3_status_type_immunity_rules() {
        // par: NO type immunity in gen3 (the Gen-6 Electric rule does NOT apply).
        assert!(!status_type_immune("par", &[Type::Electric]), "gen3: Electric CAN be paralyzed");
        assert!(!status_type_immune("par", &[Type::Steel]), "gen3: Steel CAN be paralyzed");
        // frz / brn.
        assert!(status_type_immune("frz", &[Type::Ice]), "Ice is frz-immune");
        assert!(!status_type_immune("frz", &[Type::Water]), "Water is NOT frz-immune");
        assert!(status_type_immune("brn", &[Type::Fire]), "Fire is brn-immune");
        // psn / tox: Poison & Steel.
        assert!(status_type_immune("psn", &[Type::Poison]), "Poison is psn-immune");
        assert!(status_type_immune("psn", &[Type::Steel]), "Steel is psn-immune");
        assert!(status_type_immune("tox", &[Type::Poison]), "Poison is tox-immune (via psn)");
        assert!(status_type_immune("tox", &[Type::Steel]), "Steel is tox-immune");
        assert!(!status_type_immune("psn", &[Type::Grass, Type::Ground]), "Grass/Ground not psn-immune");
        // slp: never.
        assert!(!status_type_immune("slp", &[Type::Ice]), "slp has no type immunity");
    }

    // The onTrySetStatus gates applied by a secondary: an already-statused mon is NOT
    // re-statused, and a type-immune target's status no-ops — both DRAW-FREE (the
    // random(100) already drew; only the APPLY is gated).
    #[test]
    fn try_set_status_gates_already_statused_and_type_immune() {
        let d = dex();
        let gengar = "Gengar||leftovers||sludgebomb,thunderbolt|Modest|,,,252,,252|||||";
        let muk = "Muk||leftovers||sludgebomb,shadowball|Modest|252,,,,252,|||||";
        let mut state = BattleState::start(&opts(gengar, muk, "1,2,3,4"), &d).expect("start");
        // (a) Muk is pure Poison → psn-IMMUNE. try_set_status('psn') no-ops.
        state.try_set_status(1, 0, "psn", None, &d);
        assert_eq!(state.sides[1].pokemon[0].status, None, "Poison-type Muk is psn-immune");
        // (b) Already-statused: paralyze a normal mon, then a 2nd status fails.
        let snorlax = "Snorlax||leftovers||bodyslam,earthquake|Adamant|252,252,,,,|||||";
        let mut st2 = BattleState::start(&opts(snorlax, muk, "1,2,3,4"), &d).expect("start");
        st2.try_set_status(0, 0, "par", None, &d);
        assert_eq!(st2.sides[0].pokemon[0].status, Some(Status::Paralysis), "Snorlax paralyzed");
        st2.try_set_status(0, 0, "brn", None, &d); // already statused → fail
        assert_eq!(st2.sides[0].pokemon[0].status, Some(Status::Paralysis), "already-statused: no re-status");
    }

    // SHIELD DUST on the DEFENDER suppresses the secondary random(100) ENTIRELY (a
    // draw-COUNT effect): a Body Slam into a Shield Dust mon draws acc+crit+dmg but
    // NO secondary random(100).
    #[test]
    fn shield_dust_defender_suppresses_the_secondary_draw() {
        let d = dex();
        // Tauros Body Slam vs a Shield Dust Dustox (bulky enough to survive).
        let tauros = "Tauros||leftovers||bodyslam,earthquake|Jolly|,252,,,,252|||||";
        let dustox = "Dustox||leftovers|shielddust|tackle,rest|Calm|252,,,,252,|||||";
        let mut state = BattleState::start(&opts(tauros, dustox, "5,6,7,8"), &d).expect("start");
        let before = state.prng_seed();
        let res = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        assert!(res.landed, "Body Slam lands on Dustox");
        // Oracle: acc+crit+dmg ONLY — NO secondary random(100) (Shield Dust filtered).
        let mut oracle = crate::prng::Prng::new(before.as_str());
        let _ = oracle.random_chance(100, 100);
        let _ = oracle.random_chance(1, 16);
        let _ = oracle.random_below(16);
        assert_eq!(
            state.prng_seed(), oracle.get_seed(),
            "Shield Dust suppresses the secondary random(100) — acc+crit+dmg only (3 draws)"
        );
    }

    // A FLINCHED mon's onBeforeMove is DRAW-FREE: the move aborts with NO roll. We set
    // the flinch volatile and assert run_move draws nothing.
    #[test]
    fn flinch_aborts_draw_free() {
        let d = dex();
        let snorlax = "Snorlax||leftovers||bodyslam,earthquake|Adamant|252,252,,,,|||||";
        let blissey = "Blissey||leftovers||icebeam,softboiled|Calm|252,,,,252,|||||";
        let mut state = BattleState::start(&opts(snorlax, blissey, "1,2,3,4"), &d).expect("start");
        state.sides[0].pokemon[0].flinch = true;
        let before = state.prng_seed();
        let res = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        assert!(!res.landed && !res.missed, "a flinched move aborts");
        assert_eq!(state.prng_seed(), before, "flinch draws NOTHING (the seed is unchanged)");
        assert_eq!(state.sides[1].pokemon[0].hp, state.sides[1].pokemon[0].maxhp, "Blissey untouched");
    }

    // A still-ASLEEP mon's onBeforeMove is DRAW-FREE (a counter decrement); on the
    // wake turn (counter hits 0) it cures + proceeds. We set Sleep(2) and assert two
    // moves: first decrements to Sleep(1) + aborts draw-free, second wakes + moves.
    #[test]
    fn sleep_counter_is_draw_free_and_wakes() {
        let d = dex();
        let snorlax = "Snorlax||leftovers||bodyslam,earthquake|Adamant|252,252,,,,|||||";
        let shuckle = "Shuckle||leftovers||tackle,rest|Bold|252,,252,,,|||||";
        let mut state = BattleState::start(&opts(snorlax, shuckle, "9,9,9,9"), &d).expect("start");
        state.sides[0].pokemon[0].status = Some(Status::Sleep(2));
        let before = state.prng_seed();
        // First move: Sleep(2) -> Sleep(1), ABORT, DRAW-FREE.
        let r1 = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        assert!(!r1.landed, "still asleep aborts");
        assert_eq!(state.sides[0].pokemon[0].status, Some(Status::Sleep(1)), "counter decremented");
        assert_eq!(state.prng_seed(), before, "the sleep counter decrement is DRAW-FREE");
        // Second move: Sleep(1) -> 0 -> WAKE (cure) + PROCEED to a full move.
        let r2 = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        assert!(r2.landed, "the wake turn proceeds to the move");
        assert_eq!(state.sides[0].pokemon[0].status, None, "woke up (status cleared)");
    }

    // --- STANDALONE STATUS MOVES (this step) ---

    // THUNDER WAVE LANDS (gen3customgame, NO SetStatus shuffle): draws ONLY accuracy
    // (a never-miss-100 move so randomChance(100,100) = ONE random(100)), applies par,
    // NO crit/damage/secondary, `landed` FALSE (no in-tryMoveHit Update).
    #[test]
    fn thunder_wave_draws_only_accuracy_and_applies_par() {
        let d = dex();
        // gen3customgame opts → sleep_clause OFF → no SetStatus handler-sort shuffle.
        let zapdos = "Zapdos||leftovers||thunderwave,thunderbolt|Modest|,,,252,,252|||||";
        let snorlax = "Snorlax||leftovers||bodyslam,earthquake|Adamant|252,252,,,,|||||";
        let mut state = BattleState::start(&opts_cg(zapdos, snorlax, "1,2,3,4"), &d).expect("start");
        let before = state.prng_seed();
        let res = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        assert!(!res.landed, "a status move never fires the in-tryMoveHit Update (landed=false)");
        assert!(!res.missed, "Thunder Wave is 100 acc — it hits");
        assert_eq!(state.sides[1].pokemon[0].status, Some(Status::Paralysis), "Snorlax paralyzed");
        // Oracle: ONE random(100) (the accuracy randomChance(100,100)) and nothing else.
        let mut oracle = crate::prng::Prng::new(before.as_str());
        let _ = oracle.random_chance(100, 100);
        assert_eq!(state.prng_seed(), oracle.get_seed(), "TWave draws ONLY accuracy (1 draw)");
    }

    // THUNDER WAVE -> GROUND IMMUNE: the move-type immunity (Electric 0x vs Ground) →
    // NO paralysis. Accuracy is STILL drawn (gen3 reports -immune after the acc roll),
    // and there is NO SetStatus shuffle (the event is never reached). 1 draw total.
    #[test]
    fn thunder_wave_into_ground_is_immune_accuracy_only() {
        let d = dex();
        let zapdos = "Zapdos||leftovers||thunderwave,icebeam|Modest|,,,252,,252|||||";
        // Swampert is Water/Ground → Electric is 0x → TWave's type immunity blocks par.
        let swampert = "Swampert||leftovers||surf,earthquake|Bold|252,252,,,,|||||";
        let mut state = BattleState::start(&opts_cg(zapdos, swampert, "1,2,3,4"), &d).expect("start");
        let before = state.prng_seed();
        let res = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        assert!(!res.landed && !res.missed, "immune: not landed, not missed (accuracy-only)");
        assert_eq!(state.sides[1].pokemon[0].status, None, "Ground is immune to Thunder Wave");
        // Accuracy was STILL drawn (one random(100)); nothing else (no SetStatus shuffle).
        let mut oracle = crate::prng::Prng::new(before.as_str());
        let _ = oracle.random_chance(100, 100);
        assert_eq!(state.prng_seed(), oracle.get_seed(), "immune draws ONLY accuracy");
    }

    // SPORE -> SLEEP: a 100-acc sleep move draws accuracy THEN the slp onStart
    // random(2,6) duration (gen3customgame, no SetStatus shuffle). The Sleep counter
    // stored == the random(2,6) value.
    #[test]
    fn spore_draws_accuracy_then_sleep_random_2_6() {
        let d = dex();
        let breloom = "Breloom||leftovers||spore,skyuppercut|Adamant|,252,,,,252|||||";
        let snorlax = "Snorlax||leftovers||bodyslam,earthquake|Adamant|252,252,,,,|||||";
        let mut state = BattleState::start(&opts_cg(breloom, snorlax, "1,2,3,4"), &d).expect("start");
        let before = state.prng_seed();
        // Oracle the expected duration: accuracy randomChance(100,100) then random_range(2,6).
        let mut oracle = crate::prng::Prng::new(before.as_str());
        let _ = oracle.random_chance(100, 100);
        let dur = oracle.random_range(2, 6) as u8;
        let res = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        assert!(!res.landed, "a status move's landed flag is false");
        assert_eq!(
            state.sides[1].pokemon[0].status, Some(Status::Sleep(dur)),
            "Spore applies Sleep(random(2,6)); stored == the drawn duration"
        );
        assert_eq!(state.prng_seed(), oracle.get_seed(), "Spore draws accuracy + random(2,6) (2 draws)");
    }

    // SLEEP CLAUSE MOD (gen3ou): a 2nd foe-sleep FAILS — the move draws accuracy + the
    // SetStatus handler-sort shuffle (the clause handler is one of the 2 tied) but NO
    // random(2,6) (the onStart never runs). Compared head-to-head with a clause-FREE
    // gen3customgame run, which DOES draw the random(2,6).
    #[test]
    fn sleep_clause_blocks_second_sleep_no_duration_draw() {
        let d = dex();
        let breloom = "Breloom||leftovers||spore,skyuppercut|Adamant|,252,,,,252|||||";
        // Two foe mons; sleep the FIRST, then try to sleep the SECOND (active swap not
        // needed — we set the bench mon asleep and target the active).
        let foes = "Snorlax||leftovers||bodyslam,earthquake|Adamant|252,252,,,,|||||]Blissey||leftovers||icebeam,thunderbolt|Calm|252,,,,252,|||||";
        // gen3ou (clause ON): the active foe is targeted while the BENCH foe is asleep.
        let mut ou = BattleState::start(&opts(breloom, foes, "1,2,3,4"), &d).expect("start");
        ou.sides[1].pokemon[1].status = Some(Status::Sleep(3)); // a living, asleep ally
        let before = ou.prng_seed();
        ou.try_set_status(1, 0, "slp", None, &d); // target the ACTIVE foe (slot 0)
        assert_eq!(ou.sides[1].pokemon[0].status, None, "Sleep Clause blocks the 2nd sleep");
        // Drew the SetStatus shuffle (1 size-2 Fisher-Yates draw) but NO random(2,6).
        let mut oracle = crate::prng::Prng::new(before.as_str());
        let mut h: Vec<EventHandler<usize>> = (0..2)
            .map(|i| EventHandler { order: NO_ORDER, priority: 0, speed: 0.0, sub_order: 0, effect_order: 0, handler: i })
            .collect();
        speed_sort(&mut h, &mut oracle);
        assert_eq!(ou.prng_seed(), oracle.get_seed(), "clause block draws the SetStatus shuffle, NOT random(2,6)");

        // gen3customgame (clause OFF): the same 2nd sleep LANDS and draws random(2,6).
        let mut cg = BattleState::start(&opts_cg(breloom, foes, "1,2,3,4"), &d).expect("start");
        cg.sides[1].pokemon[1].status = Some(Status::Sleep(3));
        cg.try_set_status(1, 0, "slp", None, &d);
        assert!(matches!(cg.sides[1].pokemon[0].status, Some(Status::Sleep(_))), "no clause → 2nd sleep lands");
    }

    // The SetStatus handler-sort shuffle is gen3ou-ONLY: a status APPLICATION in gen3ou
    // draws one extra size-2 shuffle vs the same application in gen3customgame.
    #[test]
    fn set_status_shuffle_is_gen3ou_only() {
        let d = dex();
        let zapdos = "Zapdos||leftovers||thunderwave,thunderbolt|Modest|,,,252,,252|||||";
        let snorlax = "Snorlax||leftovers||bodyslam,earthquake|Adamant|252,252,,,,|||||";
        // gen3customgame: try_set_status('par') is DRAW-FREE (no clause handlers).
        let mut cg = BattleState::start(&opts_cg(zapdos, snorlax, "1,2,3,4"), &d).expect("start");
        let cg_before = cg.prng_seed();
        cg.try_set_status(1, 0, "par", None, &d);
        assert_eq!(cg.prng_seed(), cg_before, "gen3customgame: try_set_status is draw-free");
        assert_eq!(cg.sides[1].pokemon[0].status, Some(Status::Paralysis));
        // gen3ou: the SAME apply draws ONE size-2 SetStatus handler-sort shuffle.
        let mut ou = BattleState::start(&opts(zapdos, snorlax, "1,2,3,4"), &d).expect("start");
        let ou_before = ou.prng_seed();
        ou.try_set_status(1, 0, "par", None, &d);
        let mut oracle = crate::prng::Prng::new(ou_before.as_str());
        let mut h: Vec<EventHandler<usize>> = (0..2)
            .map(|i| EventHandler { order: NO_ORDER, priority: 0, speed: 0.0, sub_order: 0, effect_order: 0, handler: i })
            .collect();
        speed_sort(&mut h, &mut oracle);
        assert_eq!(ou.prng_seed(), oracle.get_seed(), "gen3ou: try_set_status draws ONE size-2 shuffle");
    }

    // FAIL-LOUD: an unmodeled status move (Haze/Substitute/Baton Pass/field) PANICS
    // rather than silently desyncing — mirroring the >1-secondary guard. (Pure
    // SELF-BOOST setup moves are now MODELED — see `self_boost_move_applies_the_boost_*`
    // below; SELF-HEAL recovery moves [Recover / Rest / Moonlight / …] are MODELED; and
    // as of THIS step the gen-3 PHAZE moves Roar / Whirlwind are MODELED too — so the
    // example is re-keyed to a STILL-unmodeled status move: **Haze** [boost reset — a
    // DIFFERENT mechanic from a forceSwitch phaze, explicitly deferred].)
    #[test]
    #[should_panic(expected = "is not modeled")]
    fn unmodeled_status_move_panics() {
        let d = dex();
        // Haze is a Status move (resets all boosts — NOT a phaze) NOT in the modeled
        // set → panic. (Roar/Whirlwind would NOT panic now — they are the modeled phaze.)
        let suicune = "Suicune||leftovers||haze,surf|Bold|252,,252,,,|||||";
        let snorlax = "Snorlax||leftovers||bodyslam,earthquake|Adamant|252,252,,,,|||||";
        let mut state = BattleState::start(&opts_cg(suicune, snorlax, "1,2,3,4"), &d).expect("start");
        let _ = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
    }

    // DESTINY BOND (out-of-gen-3-modeled-scope — a reactive `volatileStatus:'destinybond'`
    // move) FAIL-LOUDS: it is category Status, not in the modeled status set, so it hits the
    // same fail-loud guard as Haze. This confirms the port PANICS (never silently mishandles /
    // no-ops) if destinybond is ever routed — the belt-and-braces the e2e's `MOVE_ID_BLOCKLIST`
    // + the Status-branch exclusion keep OFF the pickable path (a team carrying it plays fine
    // because `pickMove` never picks it, but if it ever were, the port fails loud, not silently).
    #[test]
    #[should_panic(expected = "is not modeled")]
    fn destinybond_status_move_panics_fail_loud() {
        let d = dex();
        let gengar = "Gengar||leftovers|levitate|destinybond,thunderbolt|Timid|252,,,,,252|||||";
        let snorlax = "Snorlax||leftovers||bodyslam,earthquake|Adamant|252,252,,,,|||||";
        let mut state = BattleState::start(&opts_cg(gengar, snorlax, "1,2,3,4"), &d).expect("start");
        let _ = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
    }

    // RECOVERY: Recover heals EXACTLY `floor(maxhp/2)` on the USER, DRAW-FREE (never-miss
    // → no accuracy draw; `heal()` consumes no PRNG; status moveHit → not landed). The
    // seed is unchanged; only the user's HP changes.
    #[test]
    fn recover_heals_half_maxhp_draw_free() {
        let d = dex();
        let suicune = "Suicune||leftovers||recover,surf|Bold|252,,252,,,|||||";
        let snorlax = "Snorlax||leftovers||bodyslam,earthquake|Adamant|252,252,,,,|||||";
        let mut state = BattleState::start(&opts_cg(suicune, snorlax, "1,2,3,4"), &d).expect("start");
        let max = state.sides[0].pokemon[0].maxhp;
        state.sides[0].pokemon[0].hp = 1; // very low → the heal is unclamped
        let before = state.prng_seed();
        let res = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        assert_eq!(state.sides[0].pokemon[0].hp, (1 + max / 2).min(max), "Recover heals floor(maxhp/2)");
        assert!(!res.landed && !res.missed, "a recovery status move never lands and never misses");
        assert_eq!(state.prng_seed(), before, "Recover consumes no PRNG");
    }

    // RECOVERY at FULL HP: Recover FAILS (heals 0) — `heal()` returns false → `-fail`.
    // Still DRAW-FREE; the HP stays at max and the seed is unchanged.
    #[test]
    fn recover_at_full_hp_fails_draw_free() {
        let d = dex();
        let suicune = "Suicune||leftovers||recover,surf|Bold|252,,252,,,|||||";
        let snorlax = "Snorlax||leftovers||bodyslam,earthquake|Adamant|252,252,,,,|||||";
        let mut state = BattleState::start(&opts_cg(suicune, snorlax, "1,2,3,4"), &d).expect("start");
        let max = state.sides[0].pokemon[0].maxhp; // hp == maxhp at construction
        let before = state.prng_seed();
        let _ = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        assert_eq!(state.sides[0].pokemon[0].hp, max, "Recover at full HP heals 0 (fail)");
        assert_eq!(state.prng_seed(), before, "a failed full-HP Recover draws nothing");
    }

    // WEATHER-CONDITIONAL RECOVERY: the gen4-inherited integer onHit. NONE → floor(maxhp/2),
    // SUN → floor(maxhp*2/3), SAND/RAIN/HAIL → floor(maxhp/4). All DRAW-FREE.
    #[test]
    fn moonlight_weather_conditional_heal_amounts() {
        let d = dex();
        let umbreon = "Umbreon||leftovers||moonlight,toxic|Calm|252,,,,252,|||||";
        let snorlax = "Snorlax||leftovers||bodyslam,earthquake|Adamant|252,252,,,,|||||";
        let max = {
            let st = BattleState::start(&opts_cg(umbreon, snorlax, "1,2,3,4"), &d).expect("start");
            st.sides[0].pokemon[0].maxhp
        };
        let cases = [
            (None, max / 2),
            (Some(Weather::Sun), (max as u32 * 2 / 3) as u16),
            (Some(Weather::Sand), max / 4),
            (Some(Weather::Rain), max / 4),
            (Some(Weather::Hail), max / 4),
        ];
        for (weather, expected) in cases {
            let mut state = BattleState::start(&opts_cg(umbreon, snorlax, "1,2,3,4"), &d).expect("start");
            state.field.weather = weather;
            state.sides[0].pokemon[0].hp = 1; // unclamped
            let before = state.prng_seed();
            let _ = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
            assert_eq!(
                state.sides[0].pokemon[0].hp, (1 + expected).min(max),
                "Moonlight heal under {weather:?} should be {expected}"
            );
            assert_eq!(state.prng_seed(), before, "weather-heal Moonlight consumes no PRNG");
        }
    }

    // REST: a FIXED `Sleep(3)` self-sleep + a FULL heal + a prior-status CURE. In
    // gen3customgame (no clauses) Rest draws EXACTLY ONE `random(2,6)` — the gen-3
    // `slp.onStart` duration roll, whose VALUE Rest then OVERWRITES to a fixed 3 (the
    // draw-COUNT subtlety: the draw happens, the value is discarded). No clause shuffle in
    // gen3customgame. The user wakes via the existing on_before_move counter (3 attempts).
    #[test]
    fn rest_full_heal_fixed_sleep_three_and_cure_draws_one_random_2_6_customgame() {
        let d = dex();
        let snorlax = "Snorlax||leftovers||rest,bodyslam|Careful|252,,,,252,|||||";
        let tauros = "Tauros||leftovers||bodyslam,earthquake|Adamant|,252,,,,252|||||";
        let mut state = BattleState::start(&opts_cg(snorlax, tauros, "1,2,3,4"), &d).expect("start");
        let max = state.sides[0].pokemon[0].maxhp;
        state.sides[0].pokemon[0].hp = max / 3;
        // A prior status to cure. Use POISON (no onBeforeMove draw — para/sleep/freeze
        // would draw a pre-move roll that confounds the draw-count assertion); Rest
        // overrides any prior major status with sleep regardless.
        state.sides[0].pokemon[0].status = Some(Status::Poison);
        let before = state.prng_seed();
        let res = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        assert_eq!(state.sides[0].pokemon[0].hp, max, "Rest fully heals to maxhp");
        assert_eq!(
            state.sides[0].pokemon[0].status, Some(Status::Sleep(3)),
            "Rest sets a FIXED Sleep(3) (duration value discarded), curing the prior poison"
        );
        assert!(!res.landed && !res.missed, "Rest is a status move (not landed)");
        // The seed must equal an oracle that drew EXACTLY ONE random(2,6) (and nothing
        // else): the slp.onStart duration roll Rest consumes-then-overrides. (And it must
        // have CHANGED from `before` — a draw DID occur, NOT draw-free.)
        assert_ne!(state.prng_seed(), before, "Rest consumes one random(2,6) → the seed advances");
        let mut oracle = crate::prng::Prng::new("1,2,3,4");
        let _ = oracle.random_range(2, 6);
        assert_eq!(
            state.prng_seed(), oracle.get_seed(),
            "Rest in gen3customgame draws EXACTLY one random(2,6) (the slp.onStart duration, value discarded)"
        );
    }

    // REST at FULL HP: the onTry FAILS (`-fail heal`) — Rest does NOT sleep, NOT cure, NOT
    // heal. Draw-free; the user stays awake and at max HP.
    #[test]
    fn rest_at_full_hp_fails_without_sleeping() {
        let d = dex();
        let snorlax = "Snorlax||leftovers||rest,bodyslam|Careful|252,,,,252,|||||";
        let tauros = "Tauros||leftovers||bodyslam,earthquake|Adamant|,252,,,,252|||||";
        let mut state = BattleState::start(&opts_cg(snorlax, tauros, "1,2,3,4"), &d).expect("start");
        let max = state.sides[0].pokemon[0].maxhp; // full at construction
        let before = state.prng_seed();
        let _ = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        assert_eq!(state.sides[0].pokemon[0].hp, max, "full-HP Rest heals nothing");
        assert_eq!(state.sides[0].pokemon[0].status, None, "full-HP Rest does NOT put the user to sleep");
        assert_eq!(state.prng_seed(), before, "a failed full-HP Rest draws nothing");
    }

    // REST in gen3ou: the self-`setStatus('slp')` reaches `runEvent('SetStatus')`, which
    // draws the size-2 clause handler-sort shuffle (ONE random(0,2)) — gated by
    // `sleep_clause` — and THEN `slp.onStart` draws the `random(2,6)` duration (value
    // discarded → Sleep(3)). A self-Rest is EXEMPT from the Sleep-Clause CAP (it never
    // blocks). So in gen3ou Rest draws TWO things in order: the SetStatus shuffle, then the
    // random(2,6). Verify by comparing the post-Rest seed to that exact oracle sequence.
    #[test]
    fn rest_in_gen3ou_draws_the_setstatus_shuffle_then_random_2_6() {
        let d = dex();
        let snorlax = "Snorlax||leftovers||rest,bodyslam|Careful|252,,,,252,|||||";
        let tauros = "Tauros||leftovers||bodyslam,earthquake|Adamant|,252,,,,252|||||";
        // gen3ou (sleep_clause ON) — `opts` defaults to gen3ou.
        let mut state = BattleState::start(&opts(snorlax, tauros, "1,2,3,4"), &d).expect("start");
        let max = state.sides[0].pokemon[0].maxhp;
        state.sides[0].pokemon[0].hp = max / 3;
        let _ = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        assert_eq!(state.sides[0].pokemon[0].status, Some(Status::Sleep(3)), "Rest sleeps fixed-3 in gen3ou too");
        assert_eq!(state.sides[0].pokemon[0].hp, max, "Rest full-heals in gen3ou");
        // The oracle: from the same init seed, draw the size-2 SetStatus shuffle FIRST,
        // then the slp.onStart random(2,6).
        let mut oracle = crate::prng::Prng::new("1,2,3,4");
        let mut h: Vec<EventHandler<usize>> = (0..2)
            .map(|i| EventHandler { order: NO_ORDER, priority: 0, speed: 0.0, sub_order: 0, effect_order: 0, handler: i })
            .collect();
        speed_sort(&mut h, &mut oracle);
        let _ = oracle.random_range(2, 6);
        assert_eq!(state.prng_seed(), oracle.get_seed(), "gen3ou Rest draws the SetStatus shuffle THEN random(2,6)");
    }

    // A pure SELF-BOOST setup move (Calm Mind +1 SpA/+1 SpD) raises the USER'S stat
    // stages, DRAW-FREE (boost() consumes no PRNG), and does NOT panic. The seed is
    // unchanged (never-miss → no accuracy draw, no in-tryMoveHit Update), and only the
    // user's boost STATE changes.
    #[test]
    fn self_boost_move_applies_the_boost_draw_free() {
        let d = dex();
        let alakazam = "Alakazam||leftovers||calmmind,psychic|Timid|,,,252,,252|||||";
        let snorlax = "Snorlax||leftovers||bodyslam,earthquake|Adamant|252,252,,,,|||||";
        let mut state = BattleState::start(&opts_cg(alakazam, snorlax, "1,2,3,4"), &d).expect("start");
        let before = state.prng_seed();
        let res = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        // Calm Mind = +1 SpA (idx 2), +1 SpD (idx 3); never landed (status moveHit).
        assert_eq!(state.sides[0].pokemon[0].boosts[2], 1, "+1 SpA");
        assert_eq!(state.sides[0].pokemon[0].boosts[3], 1, "+1 SpD");
        assert!(!res.landed, "a status move never lands (no in-tryMoveHit Update)");
        assert!(!res.missed, "a never-miss setup move never misses");
        // DRAW-FREE: the seed is unchanged.
        assert_eq!(state.prng_seed(), before, "self-boost consumes no PRNG");
    }

    // A self-boost into the +6 cap is a no-op but still "succeeds" (draws nothing): a
    // +2 Atk Swords Dance from +5 lands at +6, not +7.
    #[test]
    fn self_boost_clamps_at_plus_six() {
        let d = dex();
        let scizor = "Scizor||leftovers||swordsdance,silverwind|Adamant|252,252,,,,|||||";
        let snorlax = "Snorlax||leftovers||bodyslam,earthquake|Adamant|252,252,,,,|||||";
        let mut state = BattleState::start(&opts_cg(scizor, snorlax, "1,2,3,4"), &d).expect("start");
        state.sides[0].pokemon[0].boosts[0] = 5; // already +5 Atk
        let before = state.prng_seed();
        let _ = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        assert_eq!(state.sides[0].pokemon[0].boosts[0], 6, "+2 from +5 caps at +6");
        assert_eq!(state.prng_seed(), before, "a capped self-boost still draws nothing");
    }

    // The +SPEED self-boost (Dragon Dance / Agility) raises `boosts[4]` IMMEDIATELY but
    // leaves `cached_speed` STALE — Showdown re-establishes it only at the next re-cache
    // site (turn start / residual / switch-in). So the boost is visible in the boost
    // array right away, while the cached speed (read by the eachEvent tie-shuffles) is
    // unchanged until `update_speed()` runs.
    #[test]
    fn dragon_dance_boosts_speed_but_leaves_cached_speed_stale() {
        let d = dex();
        let salamence = "Salamence||leftovers|intimidate|dragondance,dragonclaw|Adamant|,252,,,,252|||||";
        let snorlax = "Snorlax||leftovers||bodyslam,earthquake|Adamant|252,252,,,,|||||";
        let mut state = BattleState::start(&opts_cg(salamence, snorlax, "1,2,3,4"), &d).expect("start");
        // Establish the turn-start cached speed (commitChoices' updateSpeed).
        state.update_speed(&d);
        let cached_before = state.sides[0].pokemon[0].cached_speed;
        let _ = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        // The boost stage is applied immediately (+1 Atk, +1 Spe).
        assert_eq!(state.sides[0].pokemon[0].boosts[0], 1, "+1 Atk");
        assert_eq!(state.sides[0].pokemon[0].boosts[4], 1, "+1 Spe");
        // But cached_speed is UNCHANGED (stale until the next re-cache site).
        assert_eq!(
            state.sides[0].pokemon[0].cached_speed, cached_before,
            "Dragon Dance does not refresh the cached pokemon.speed mid-turn"
        );
        // The next update_speed() (residual / next turn) picks up the boosted speed.
        state.update_speed(&d);
        assert!(
            state.sides[0].pokemon[0].cached_speed > cached_before,
            "the boosted speed takes effect at the next updateSpeed site"
        );
    }

    // WATER ABSORB heals ONLY when the absorbed move HITS — it is an `onTryHit`
    // ability that fires AFTER the accuracy check. A MISSED Water move (e.g. Hydro
    // Pump's 80% accuracy fails) must NOT heal the Absorb holder (an e2e-capstone
    // fix: a missed Hydro Pump into a Water Absorb Politoed left it healed by maxhp/4
    // when it should only have missed). The draw count is accuracy-only either way.
    #[test]
    fn water_absorb_heals_on_hit_but_not_on_a_miss() {
        let d = dex();
        // p1 Suicune (Hydro Pump, 80% acc). p2 Politoed (Water Absorb) at half HP.
        let suicune = "Suicune||leftovers||hydropump,surf|Modest|,,,252,,252|||||";
        let politoed = "Politoed||leftovers|waterabsorb|icebeam,surf|Bold|252,,252,,,|||||";

        // --- A HIT: Water Absorb heals maxhp/4 (Hydro Pump is immune → heal, not damage). ---
        // Force the accuracy roll to PASS by seeding so randomChance(80,100) succeeds.
        let mut hit_seed = None;
        for s in 0..400u32 {
            let seed = format!("{},{},{},{}", s + 1, 7, 13, 29);
            let mut st = BattleState::start(&opts_cg(suicune, politoed, &seed), &d).expect("start");
            st.sides[1].pokemon[0].hp = st.sides[1].pokemon[0].maxhp / 2;
            let hp0 = st.sides[1].pokemon[0].hp;
            let max = st.sides[1].pokemon[0].maxhp;
            let _ = st.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
            let hp1 = st.sides[1].pokemon[0].hp;
            if hp1 == (hp0 + max / 4).min(max) {
                hit_seed = Some(seed); // a HIT → healed by maxhp/4
                break;
            }
        }
        assert!(hit_seed.is_some(), "expected a Hydro Pump HIT seed where Water Absorb heals maxhp/4");

        // --- A MISS: the holder is UNCHANGED (no heal, no damage) — only the accuracy
        //     roll is consumed. ---
        let mut miss_found = false;
        for s in 0..400u32 {
            let seed = format!("{},{},{},{}", s + 1, 7, 13, 29);
            let mut st = BattleState::start(&opts_cg(suicune, politoed, &seed), &d).expect("start");
            st.sides[1].pokemon[0].hp = st.sides[1].pokemon[0].maxhp / 2;
            let hp0 = st.sides[1].pokemon[0].hp;
            let res = st.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
            if res.missed {
                assert_eq!(
                    st.sides[1].pokemon[0].hp, hp0,
                    "a MISSED Water move must NOT heal the Water Absorb holder"
                );
                miss_found = true;
                break;
            }
        }
        assert!(miss_found, "expected a Hydro Pump MISS seed (80% accuracy)");
    }

    // STATUS_IMMUNE `setStatus`-phase (Insomnia → slp) — `gen3_status_immune_v1`. The block
    // is now DATA-DRIVEN + DOES NOT PANIC (the old "size-3 shuffle" panic was WRONG). PROBE-
    // settled draw model (`harness/probe_statusimmune_shuffle_size.js`): in gen3ou the ability
    // handler sorts into its OWN speed group, leaving the 2-clause tie a SIZE-2 shuffle → the
    // SetStatus event draws EXACTLY ONE `random(0,2)` (same as a normal status apply), then the
    // ability RETURN blocks the sleep DRAW-FREE. So the mon stays unstatused AND the draw count
    // matches a landed status's shuffle.
    #[test]
    fn insomnia_blocks_slp_in_gen3ou_drawing_the_size2_clause_shuffle() {
        let d = dex();
        // gen3ou (sleep_clause ON): the SetStatus event fires its 2-clause shuffle.
        let breloom = "Breloom||leftovers||spore,skyuppercut|Adamant|,252,,,,252|||||";
        let noctowl = "Noctowl||leftovers|insomnia|hypnosis,psychic|Modest|252,,,,252,|||||";
        let mut state = BattleState::start(&opts(breloom, noctowl, "1,2,3,4"), &d).expect("start");
        assert!(state.sleep_clause, "gen3ou must carry the sleep clause");
        let before = state.prng_seed();
        // Reference: a normal status apply's SetStatus event draws exactly one random(0,2).
        // Reconstruct a fresh PRNG at `before` and draw once (the PRNG is deterministic on its
        // seed string, so this is the bit-exact "one size-2 shuffle draw" reference).
        let mut ref_prng = crate::prng::Prng::new(&before);
        ref_prng.random_range(0, 2);
        let want = ref_prng.get_seed();
        state.try_set_status(1, 0, "slp", None, &d);
        assert_eq!(state.sides[1].pokemon[0].status, None, "Insomnia blocks the sleep");
        assert_ne!(state.prng_seed(), before, "the size-2 clause shuffle DID draw in gen3ou");
        assert_eq!(state.prng_seed(), want, "exactly ONE random(0,2) (size-2), not a size-3 draw");
    }

    // STATUS_IMMUNE `setStatus`-phase in gen3customgame (no clauses): the SetStatus event's
    // only handler is the ability's own (size-1 → no tie → NO shuffle) → the block is fully
    // DRAW-FREE. The mon stays unstatused and the seed is untouched.
    #[test]
    fn insomnia_blocks_slp_in_customgame_draw_free() {
        let d = dex();
        let breloom = "Breloom||leftovers||spore,skyuppercut|Adamant|,252,,,,252|||||";
        let noctowl = "Noctowl||leftovers|insomnia|hypnosis,psychic|Modest|252,,,,252,|||||";
        let mut state = BattleState::start(&opts_cg(breloom, noctowl, "1,2,3,4"), &d).expect("start");
        assert!(!state.sleep_clause, "gen3customgame carries no sleep clause");
        let before = state.prng_seed();
        state.try_set_status(1, 0, "slp", None, &d);
        assert_eq!(state.sides[1].pokemon[0].status, None, "Insomnia blocks the sleep");
        assert_eq!(state.prng_seed(), before, "customgame block is DRAW-FREE (size-1, no shuffle)");
    }

    // STATUS_IMMUNE `immunity`-phase (Magma Armor → frz): blocks at runStatusImmunity BEFORE
    // the SetStatus event, so NO clause shuffle fires EVEN in gen3ou (probe-settled by
    // `harness/probe_statusimmune_magmaarmor.js`) — like the sun-freeze gate. Fully DRAW-FREE.
    #[test]
    fn magma_armor_blocks_frz_before_the_setstatus_shuffle() {
        let d = dex();
        let attacker = "Blissey||leftovers||icebeam,softboiled|Calm|252,,,,252,|||||";
        // Snorlax (Normal, no Ice type-immunity) with Magma Armor.
        let snorlax = "Snorlax||leftovers|magmaarmor|bodyslam,rest|Adamant|252,252,,,,|||||";
        for fmt_cg in [false, true] {
            let o = if fmt_cg {
                opts_cg(attacker, snorlax, "1,2,3,4")
            } else {
                opts(attacker, snorlax, "1,2,3,4")
            };
            let mut state = BattleState::start(&o, &d).expect("start");
            let before = state.prng_seed();
            state.try_set_status(1, 0, "frz", None, &d);
            assert_eq!(state.sides[1].pokemon[0].status, None, "Magma Armor blocks the freeze");
            assert_eq!(
                state.prng_seed(),
                before,
                "Magma Armor blocks at runStatusImmunity (BEFORE the SetStatus event) → NO clause shuffle, DRAW-FREE"
            );
        }
    }

    // A `setStatus`-phase STATUS_IMMUNE ability does NOT block a DIFFERENT status: Limber (par)
    // does NOT block a burn, so the burn applies normally (draws the gen3ou shuffle then lands).
    #[test]
    fn limber_does_not_block_a_non_par_status() {
        let d = dex();
        let attacker = "Blissey||leftovers||willowisp,softboiled|Calm|252,,,,252,|||||";
        let snorlax = "Snorlax||leftovers|limber|bodyslam,rest|Adamant|252,252,,,,|||||";
        let mut state = BattleState::start(&opts_cg(attacker, snorlax, "1,2,3,4"), &d).expect("start");
        state.try_set_status(1, 0, "brn", None, &d);
        assert_eq!(
            state.sides[1].pokemon[0].status,
            Some(Status::Burn),
            "Limber blocks par only — a burn lands"
        );
    }

    // EARLY BIRD halves the sleep counter via a DOUBLE decrement in onBeforeMove: a
    // Sleep(3) Early-Bird mon wakes after ONE attempt (3 - 2 = 1, then 1 - 2 = -1 ->
    // wake), vs two attempts for a non-Early-Bird mon.
    #[test]
    fn early_bird_double_decrements_the_sleep_counter() {
        let d = dex();
        // Dodrio has Early Bird. Set Sleep(3); the first onBeforeMove decrements by 2.
        let dodrio = "Dodrio||leftovers|earlybird|drillpeck,doubleedge|Jolly|,252,,,,252|||||";
        let foe = "Snorlax||leftovers||bodyslam,earthquake|Adamant|252,252,,,,|||||";
        let mut state = BattleState::start(&opts_cg(dodrio, foe, "1,2,3,4"), &d).expect("start");
        state.sides[0].pokemon[0].status = Some(Status::Sleep(3));
        let before = state.prng_seed();
        // Early Bird: 3 - 2 = 1 (next != 0) → still asleep, abort, DRAW-FREE.
        let r1 = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        assert!(!r1.landed, "Sleep(3) with Early Bird: still asleep after a -2 decrement");
        assert_eq!(state.sides[0].pokemon[0].status, Some(Status::Sleep(1)), "counter 3 -> 1 (Early Bird -2)");
        assert_eq!(state.prng_seed(), before, "the sleep decrement is draw-free");
        // Next attempt: 1 - 2 = -1 (saturating 0) → WAKE + proceed.
        let r2 = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        assert!(r2.landed, "wakes on the 2nd attempt (Early Bird)");
        assert_eq!(state.sides[0].pokemon[0].status, None, "woke up");
    }

    // CONFUSION self-hit: when the 50% roll says self-hit, the mon takes a typeless
    // 40-BP self-hit (ONE random(16), NO crit) then ABORTS — exactly 2 draws (the
    // randomChance(1,2) + the random(16)).
    #[test]
    fn confusion_self_hit_draws_chance_plus_one_random16() {
        let d = dex();
        let snorlax = "Snorlax||leftovers||bodyslam,earthquake|Adamant|252,252,,,,|||||";
        let blissey = "Blissey||leftovers||icebeam,softboiled|Calm|252,,,,252,|||||";
        for s in 0..200u32 {
            let seed = format!("{},{},{},{}", s + 13, s + 17, s + 19, s + 23);
            // confusion: decrement first (we set time=2 so next=1, no removal), then
            // randomChance(1,2): self-hit when random_below(2) >= 1 (== 1).
            let mut probe = crate::prng::Prng::new(&seed);
            let acts_normally = probe.random_chance(1, 2); // true = acts (no self-hit)
            let mut state =
                BattleState::start(&opts(snorlax, blissey, &seed), &d).expect("start");
            state.sides[0].pokemon[0].confusion = Some(2);
            let before = state.prng_seed();
            let res = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
            if !acts_normally {
                // SELF-HIT: aborted, took damage, drew chance + random(16).
                assert!(!res.landed, "confusion self-hit aborts the move");
                assert!(state.sides[0].pokemon[0].hp < state.sides[0].pokemon[0].maxhp, "took self-hit damage");
                let mut oracle = crate::prng::Prng::new(before.as_str());
                let _ = oracle.random_chance(1, 2); // the self-hit gate
                let _ = oracle.random_below(16); // the self-hit damage roll (NO crit)
                assert_eq!(
                    state.prng_seed(), oracle.get_seed(),
                    "confusion self-hit = randomChance(1,2) + ONE random(16) (no crit)"
                );
                // Blissey untouched (the confused mon hit itself, didn't move).
                assert_eq!(state.sides[1].pokemon[0].hp, state.sides[1].pokemon[0].maxhp);
                // confusion counter decremented 2 -> 1.
                assert_eq!(state.sides[0].pokemon[0].confusion, Some(1));
                return;
            }
        }
        panic!("no confusion self-hit seed found");
    }

    // FREEZE onBeforeMove draws ONE randomChance(1,5); on a non-thaw the move ABORTS
    // (1 draw), on a thaw it cures + proceeds to the full move.
    #[test]
    fn freeze_draws_one_thaw_roll() {
        let d = dex();
        let snorlax = "Snorlax||leftovers||bodyslam,earthquake|Adamant|252,252,,,,|||||";
        let blissey = "Blissey||leftovers||icebeam,softboiled|Calm|252,,,,252,|||||";
        // Find a STAY-frozen seed (randomChance(1,5) false).
        for s in 0..200u32 {
            let seed = format!("{},{},{},{}", s + 2, s + 5, s + 8, s + 11);
            let mut probe = crate::prng::Prng::new(&seed);
            let thaw = probe.random_chance(1, 5);
            if thaw {
                continue;
            }
            let mut state =
                BattleState::start(&opts(snorlax, blissey, &seed), &d).expect("start");
            state.sides[0].pokemon[0].status = Some(Status::Freeze);
            let before = state.prng_seed();
            let res = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
            assert!(!res.landed, "a non-thawing frozen mon's move aborts");
            assert_eq!(state.sides[0].pokemon[0].status, Some(Status::Freeze), "stays frozen");
            let mut oracle = crate::prng::Prng::new(before.as_str());
            let _ = oracle.random_chance(1, 5);
            assert_eq!(
                state.prng_seed(), oracle.get_seed(),
                "a stay-frozen move draws ONLY the randomChance(1,5) thaw roll"
            );
            return;
        }
        panic!("no stay-frozen seed found");
    }

    // A LANDED confusion secondary (Water Pulse 20%) draws TWO sequential draws when it
    // lands: the secondary random(100) THEN the addVolatile onStart random(2,6) duration
    // (2..5). We find a seed where the secondary LANDS and assert the exact 5-draw move
    // sequence acc+crit+dmg+secondary(100)+random(2,6) AND the confusion counter set.
    #[test]
    fn confusion_secondary_draws_random_100_then_random_2_6() {
        let d = dex();
        let starmie = "Starmie||leftovers||waterpulse,thunderbolt|Timid|,,,252,,252|||||";
        let snorlax = "Snorlax||leftovers||bodyslam,earthquake|Adamant|252,252,,,,|||||";
        for s in 0..400u32 {
            let seed = format!("{},{},{},{}", s + 3, s + 7, s + 11, s + 13);
            // Probe the draw sequence: acc(100,100), crit(1,16), dmg(16), secondary(100).
            let mut probe = crate::prng::Prng::new(&seed);
            let _ = probe.random_chance(100, 100);
            let _ = probe.random_chance(1, 16);
            let _ = probe.random_below(16);
            let sec = probe.random_below(100);
            if sec >= 20 {
                continue; // the secondary whiffs this seed — try the next.
            }
            // It LANDS → expect the random(2,6) duration next.
            let dur = probe.random_range(2, 6) as u8;
            let mut state = BattleState::start(&opts(starmie, snorlax, &seed), &d).expect("start");
            let before = state.prng_seed();
            let res = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
            assert!(res.landed, "Water Pulse lands on Snorlax");
            let mut oracle = crate::prng::Prng::new(before.as_str());
            let _ = oracle.random_chance(100, 100);
            let _ = oracle.random_chance(1, 16);
            let _ = oracle.random_below(16);
            let _ = oracle.random_below(100); // secondary gate (lands)
            let _ = oracle.random_range(2, 6); // the confusion duration
            assert_eq!(
                state.prng_seed(), oracle.get_seed(),
                "a landed confusion secondary = acc+crit+dmg+secondary(100)+random(2,6)"
            );
            assert_eq!(
                state.sides[1].pokemon[0].confusion, Some(dur),
                "the confusion counter is the random(2,6) duration"
            );
            return;
        }
        panic!("no landing-confusion seed found");
    }

    // A confusion secondary on an ALREADY-CONFUSED target STILL draws the secondary
    // random(100) but draws NO random(2,6) (addVolatile returns false before onStart) —
    // the draw-COUNT gate. We pre-confuse the target and assert exactly 4 draws on a land.
    #[test]
    fn confusion_secondary_already_confused_skips_the_duration_draw() {
        let d = dex();
        let starmie = "Starmie||leftovers||waterpulse,thunderbolt|Timid|,,,252,,252|||||";
        let snorlax = "Snorlax||leftovers||bodyslam,earthquake|Adamant|252,252,,,,|||||";
        for s in 0..400u32 {
            let seed = format!("{},{},{},{}", s + 5, s + 9, s + 13, s + 17);
            let mut probe = crate::prng::Prng::new(&seed);
            let _ = probe.random_chance(100, 100);
            let _ = probe.random_chance(1, 16);
            let _ = probe.random_below(16);
            if probe.random_below(100) >= 20 {
                continue; // need a LANDED secondary to prove the gate skips random(2,6).
            }
            let mut state = BattleState::start(&opts(starmie, snorlax, &seed), &d).expect("start");
            // PRE-CONFUSE the target with a DISTINCT counter so a wrong re-draw would show.
            state.sides[1].pokemon[0].confusion = Some(4);
            let before = state.prng_seed();
            let res = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
            assert!(res.landed, "Water Pulse lands");
            // Oracle: acc+crit+dmg+secondary(100) ONLY — NO random(2,6).
            let mut oracle = crate::prng::Prng::new(before.as_str());
            let _ = oracle.random_chance(100, 100);
            let _ = oracle.random_chance(1, 16);
            let _ = oracle.random_below(16);
            let _ = oracle.random_below(100);
            assert_eq!(
                state.prng_seed(), oracle.get_seed(),
                "an already-confused target draws the secondary random(100) but NOT random(2,6)"
            );
            assert_eq!(
                state.sides[1].pokemon[0].confusion, Some(4),
                "the existing confusion counter is untouched (no re-add)"
            );
            return;
        }
        panic!("no landing-confusion seed found (already-confused case)");
    }

    // OWN TEMPO blocks the confusion add (onTryAddVolatile returns null) → the secondary
    // random(100) STILL draws but NO random(2,6), and the target is never confused.
    #[test]
    fn confusion_secondary_own_tempo_skips_the_duration_draw() {
        let d = dex();
        let starmie = "Starmie||leftovers||waterpulse,thunderbolt|Timid|,,,252,,252|||||";
        // Slaking has Truant; use a clean Own Tempo carrier: Slowbro (Own Tempo).
        let slowbro = "Slowbro||leftovers|owntempo|surf,rest|Bold|252,252,,,,|||||";
        for s in 0..400u32 {
            let seed = format!("{},{},{},{}", s + 2, s + 6, s + 10, s + 14);
            let mut probe = crate::prng::Prng::new(&seed);
            let _ = probe.random_chance(100, 100);
            let _ = probe.random_chance(1, 16);
            let _ = probe.random_below(16);
            if probe.random_below(100) >= 20 {
                continue;
            }
            let mut state = BattleState::start(&opts(starmie, slowbro, &seed), &d).expect("start");
            let before = state.prng_seed();
            let res = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
            assert!(res.landed, "Water Pulse lands on Slowbro");
            let mut oracle = crate::prng::Prng::new(before.as_str());
            let _ = oracle.random_chance(100, 100);
            let _ = oracle.random_chance(1, 16);
            let _ = oracle.random_below(16);
            let _ = oracle.random_below(100);
            assert_eq!(
                state.prng_seed(), oracle.get_seed(),
                "Own Tempo: the secondary random(100) drew but NO random(2,6)"
            );
            assert_eq!(
                state.sides[1].pokemon[0].confusion, None,
                "Own Tempo is immune to confusion (never set)"
            );
            return;
        }
        panic!("no landing-confusion seed found (Own Tempo case)");
    }

    // TRI ATTACK is the ONLY gen-3 multi-status secondary: on a LAND it draws ONE
    // random(100) (the 20% gate) THEN ONE random(3) (the sample of brn/par/frz) — NOT
    // three random(100)s. We find a landing seed and assert the exact 5-draw sequence
    // acc+crit+dmg+secondary(100)+random(3) AND that the sampled status was applied.
    #[test]
    fn tri_attack_draws_random_100_then_sample_random_3() {
        let d = dex();
        // Porygon2 learns Tri Attack; target a status-able Snorlax (Normal — not immune
        // to brn/par/frz by type).
        let porygon2 = "Porygon2||leftovers||triattack,recover|Modest|252,,,252,,|||||";
        let snorlax = "Snorlax||leftovers||bodyslam,earthquake|Adamant|252,252,,,,|||||";
        const SAMPLE: [&str; 3] = ["brn", "par", "frz"];
        for s in 0..400u32 {
            let seed = format!("{},{},{},{}", s + 4, s + 8, s + 12, s + 16);
            let mut probe = crate::prng::Prng::new(&seed);
            let _ = probe.random_chance(100, 100); // Tri Attack accuracy 100
            let _ = probe.random_chance(1, 16);
            let _ = probe.random_below(16);
            let sec = probe.random_below(100);
            if sec >= 20 {
                continue; // whiff — no sample draw — try the next seed.
            }
            let pick = probe.random_below(3) as usize; // the sample
            // gen3customgame (the secondary golden's format → NO SetStatus handler-sort
            // shuffle), so the Tri-Attack draw model is the bare acc+crit+dmg+100+sample(3).
            let mut state = BattleState::start(&opts_cg(porygon2, snorlax, &seed), &d).expect("start");
            let before = state.prng_seed();
            let res = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
            assert!(res.landed, "Tri Attack lands");
            // Oracle: acc+crit+dmg+secondary(100)+random(3) — exactly 5 draws (NOT 3×100).
            let mut oracle = crate::prng::Prng::new(before.as_str());
            let _ = oracle.random_chance(100, 100);
            let _ = oracle.random_chance(1, 16);
            let _ = oracle.random_below(16);
            let _ = oracle.random_below(100); // the 20% gate (lands)
            let _ = oracle.random_below(3); // the sample
            assert_eq!(
                state.prng_seed(), oracle.get_seed(),
                "Tri Attack = acc+crit+dmg+secondary(100)+sample(3) — NOT three random(100)s"
            );
            // The sampled status (brn/par/frz) was applied to Snorlax.
            let want = SAMPLE[pick];
            let got = state.sides[1].pokemon[0].status;
            let ok = matches!(
                (want, got),
                ("brn", Some(Status::Burn)) | ("par", Some(Status::Paralysis)) | ("frz", Some(Status::Freeze))
            );
            assert!(ok, "Tri Attack sampled {want:?} but applied {got:?}");
            return;
        }
        panic!("no landing Tri Attack seed found");
    }

    // A move whose flattened secondaryEffects has >1 col AND isn't Tri Attack must PANIC
    // (fail-loud), so a future multi-secondary shape can never silently mis-draw. We
    // build a synthetic MoveData and call apply_secondaries through a forged dex entry.
    #[test]
    #[should_panic(expected = "unmodeled")]
    fn unmodeled_multi_secondary_panics() {
        let mut d = dex();
        // Forge a 2-col secondary move (a status + a flinch in one move) on a real id.
        // (No real gen-3 move has this shape except Tri Attack, which is special-cased.)
        let m = d.moves_mut().get_mut("tackle").expect("tackle exists");
        m.secondary_effects = vec![("flinch".to_string(), 10), ("par".to_string(), 10)];
        let snorlax = "Snorlax||leftovers||tackle,earthquake|Adamant|252,252,,,,|||||";
        let blissey = "Blissey||leftovers||icebeam,softboiled|Calm|252,,,,252,|||||";
        let mut state = BattleState::start(&opts(snorlax, blissey, "1,2,3,4"), &d).expect("start");
        // Drives apply_secondaries with the forged 2-col move → must PANIC.
        let _ = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
    }

    // A FOE STAT-DROP secondary (Crunch −1 SpD, gen3 override) draws ONE random(100) and,
    // on a land, applies −1 SpD to the FOE (DRAW-FREE apply). A SELF-BOOST (Meteor Mash
    // +1 Atk) draws one random(100) and applies +1 Atk to the USER. Both verified by the
    // full-battle golden too; this pins the per-move apply directly.
    #[test]
    fn stat_drop_and_self_boost_apply_the_structured_spec() {
        let d = dex();
        // Crunch (−1 SpD foe, 20%): force a land by sweeping seeds.
        let ttar = "Tyranitar||leftovers||crunch,rockslide|Adamant|252,252,,,,|||||";
        let snorlax = "Snorlax||leftovers||bodyslam,earthquake|Careful|252,,,252,,|||||";
        let mut crunched = false;
        for s in 0..400u32 {
            let seed = format!("{},{},{},{}", s + 1, s + 3, s + 5, s + 7);
            let mut state = BattleState::start(&opts(ttar, snorlax, &seed), &d).expect("start");
            let _ = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
            // boosts index 3 = spd; a landed Crunch drops it to −1.
            if state.sides[1].pokemon[0].boosts[3] == -1 {
                crunched = true;
                break;
            }
        }
        assert!(crunched, "a landed Crunch must drop the foe's SpD to −1");

        // Meteor Mash (+1 Atk self, 20%): a landed one raises the USER's Atk to +1.
        let metagross = "Metagross||leftovers||meteormash,earthquake|Adamant|252,252,,,,|||||";
        let skarmory = "Skarmory||leftovers||drillpeck,rest|Impish|252,,252,,,|||||";
        let mut self_boosted = false;
        for s in 0..400u32 {
            let seed = format!("{},{},{},{}", s + 2, s + 4, s + 6, s + 8);
            let mut state = BattleState::start(&opts(metagross, skarmory, &seed), &d).expect("start");
            let _ = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
            // boosts index 0 = atk on the USER (side 0).
            if state.sides[0].pokemon[0].boosts[0] == 1 {
                self_boosted = true;
                break;
            }
        }
        assert!(self_boosted, "a landed Meteor Mash must raise the USER's Atk to +1");
    }

    // CLEAR BODY blocks a foe stat-DROP (DRAW-FREE onTryBoost): a landed Crunch into a
    // Clear Body Metagross draws the secondary random(100) but the SpD stage stays 0.
    #[test]
    fn clear_body_blocks_the_foe_stat_drop() {
        let d = dex();
        let ttar = "Tyranitar||leftovers||crunch,rockslide|Adamant|252,252,,,,|||||";
        // Metagross has Clear Body; bulky enough to survive several Crunches.
        let metagross = "Metagross||leftovers|clearbody|meteormash,rest|Impish|252,,252,,,|||||";
        for s in 0..400u32 {
            let seed = format!("{},{},{},{}", s + 3, s + 5, s + 7, s + 9);
            // Find a LANDED-secondary seed (the random(100) is the 4th draw).
            let mut probe = crate::prng::Prng::new(&seed);
            let _ = probe.random_chance(100, 100);
            let _ = probe.random_chance(1, 16);
            let _ = probe.random_below(16);
            if probe.random_below(100) >= 20 {
                continue;
            }
            let mut state = BattleState::start(&opts(ttar, metagross, &seed), &d).expect("start");
            let _ = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
            assert_eq!(
                state.sides[1].pokemon[0].boosts[3], 0,
                "Clear Body blocks the Crunch SpD-drop (the stage stays 0)"
            );
            return;
        }
        panic!("no landing-Crunch seed found (Clear Body case)");
    }

    // PROTECT: the FIRST protect (counter 0, no stall volatile) SHORT-CIRCUITS with NO
    // PRNG draw and ALWAYS succeeds — it sets `protected` and `onStart`s the stall counter
    // to 2 (+ duration 2). DRAW-FREE (run_protect itself draws nothing on the first use).
    #[test]
    fn first_protect_draws_nothing_and_sets_counter_two() {
        let d = dex();
        let snorlax = "Snorlax||leftovers||protect,bodyslam|Careful|252,,,,252,|||||";
        let tauros = "Tauros||leftovers||bodyslam,earthquake|Adamant|,252,,,,252|||||";
        let mut state = BattleState::start(&opts_cg(snorlax, tauros, "1,2,3,4"), &d).expect("start");
        let before = state.prng_seed();
        let res = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        assert!(state.sides[0].pokemon[0].protected, "the protect volatile is up");
        assert_eq!(state.sides[0].pokemon[0].protect_counter, 2, "onStart sets the stall counter to 2");
        assert_eq!(state.sides[0].pokemon[0].stall_duration, 2, "the stall volatile duration is 2");
        assert!(!res.landed && !res.missed, "protect is a status move (never lands, never misses)");
        assert_eq!(state.prng_seed(), before, "the FIRST protect draws no PRNG (StallMove has no handler yet)");
    }

    // PROTECT: a CONSECUTIVE protect (counter already > 0) draws exactly ONE
    // randomChance(1, counter); on SUCCESS the counter `onRestart`s `*= 2` (capped at 8); on
    // FAILURE the gen3 (resolved gen5-base) stall does NOT delete the volatile — the counter
    // + duration persist (so consecutive fails re-roll at the SAME denominator). The
    // success-escalation sequence is the gen3 floored 2→4→8→8.
    #[test]
    fn consecutive_protect_draws_one_stall_roll_and_escalates() {
        let d = dex();
        let snorlax = "Snorlax||leftovers||protect,bodyslam|Careful|252,,,,252,|||||";
        let tauros = "Tauros||leftovers||bodyslam,earthquake|Adamant|,252,,,,252|||||";
        // Pre-seed the stall counter to 2 (a successful first protect already happened).
        for &(counter, expect_denom, next_on_success) in &[(2u8, 2u32, 4u8), (4, 4, 8), (8, 8, 8)] {
            let mut state = BattleState::start(&opts_cg(snorlax, tauros, "1,2,3,4"), &d).expect("start");
            state.sides[0].pokemon[0].protect_counter = counter;
            state.sides[0].pokemon[0].stall_duration = 2;
            // The oracle: one randomChance(1, denom).
            let mut oracle = crate::prng::Prng::new("1,2,3,4");
            let success = oracle.random_chance(1, expect_denom);
            let _ = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
            assert_eq!(
                state.prng_seed(), oracle.get_seed(),
                "a consecutive protect at counter {counter} draws exactly one randomChance(1,{expect_denom})"
            );
            if success {
                assert!(state.sides[0].pokemon[0].protected, "stall success → protect up");
                assert_eq!(
                    state.sides[0].pokemon[0].protect_counter, next_on_success,
                    "stall success at {counter} → onRestart counter (cap 8)"
                );
            } else {
                assert!(!state.sides[0].pokemon[0].protected, "stall failure → no protection");
                // gen3 stall does NOT delete the volatile on a FAILED roll (unlike gen8+
                // base) — the counter + duration are LEFT INTACT (the volatile persists, so
                // the next consecutive protect re-rolls at the SAME denominator, and the
                // stall residual handler still fires this turn).
                assert_eq!(state.sides[0].pokemon[0].protect_counter, counter, "stall failure leaves the counter intact");
                assert_eq!(state.sides[0].pokemon[0].stall_duration, 2, "stall failure leaves the duration intact (no refresh, no delete)");
            }
        }
    }

    // PROTECT BLOCK: a foe move TARGETING the protected mon draws its accuracy roll then is
    // blocked (NO crit / damage / secondary). A blocked Earthquake leaves the protected mon
    // at full HP; the seed advances by exactly ONE accuracy randomChance (no crit/damage).
    #[test]
    fn protect_blocks_foe_move_after_its_accuracy_draw() {
        let d = dex();
        let snorlax = "Snorlax||leftovers||protect,bodyslam|Careful|252,,,,252,|||||";
        let dugtrio = "Dugtrio||||earthquake,bodyslam|Jolly|,252,,,,252|||||";
        let mut state = BattleState::start(&opts_cg(snorlax, dugtrio, "1,2,3,4"), &d).expect("start");
        // Put the protect up on the defending Snorlax (side 0), then run the foe's EQ.
        state.sides[0].pokemon[0].protected = true;
        let hp_before = state.sides[0].pokemon[0].hp;
        let res = state.run_move(MoveAction { side: 1, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        assert_eq!(state.sides[0].pokemon[0].hp, hp_before, "a blocked Earthquake deals NO damage");
        assert!(!res.landed, "a blocked move did not land (no in-tryMoveHit Update)");
        // The seed advanced by exactly ONE accuracy randomChance (EQ 100 acc), no crit/damage.
        let mut oracle = crate::prng::Prng::new("1,2,3,4");
        let _ = oracle.random_chance(100, 100);
        assert_eq!(state.prng_seed(), oracle.get_seed(), "a blocked foe move draws ONLY its accuracy roll");
    }

    // PROTECT BLOCK is target-specific: the protect on side 0 does NOT block the protecting
    // mon's OWN move (a self-target / outgoing move is never blocked by your own protect).
    #[test]
    fn protect_does_not_block_the_protectors_own_move() {
        let d = dex();
        let snorlax = "Snorlax||leftovers||bodyslam,protect|Careful|252,252,,,,|||||";
        let chansey = "Chansey||leftovers||softboiled,icebeam|Bold|252,,252,,,|||||";
        let mut state = BattleState::start(&opts_cg(snorlax, chansey, "1,2,3,4"), &d).expect("start");
        // Snorlax has its OWN protect up; it Body Slams the foe — the move must still hit.
        state.sides[0].pokemon[0].protected = true;
        let foe_hp_before = state.sides[1].pokemon[0].hp;
        let res = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        assert!(res.landed, "the protector's own attack lands (its protect does not block itself)");
        assert!(state.sides[1].pokemon[0].hp < foe_hp_before, "the foe took Body Slam damage");
    }

    // FAIL-LOUD: an `isProtect` move that is NOT Protect/Detect (Endure — a different
    // onDamage mechanic) PANICS rather than silently desync.
    #[test]
    #[should_panic(expected = "is not modeled")]
    fn endure_panics_fail_loud() {
        let d = dex();
        let snorlax = "Snorlax||leftovers||endure,bodyslam|Careful|252,,,,252,|||||";
        let tauros = "Tauros||leftovers||bodyslam,earthquake|Adamant|,252,,,,252|||||";
        let mut state = BattleState::start(&opts_cg(snorlax, tauros, "1,2,3,4"), &d).expect("start");
        // Endure is isProtect=true but volatileStatus:'endure' → fail-loud in run_protect.
        let _ = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
    }

    // FAIL-LOUD: a DEFERRED fixed-damage move (Counter — a reactive damageCallback) is in
    // `is_fixed_damage_move` (so it can never silently no-op via the bp==0 fall-through) but
    // has NO `fixed_damage_amount` entry, so it PANICS in `run_fixed_damage_move` rather than
    // desync. Mirrors `endure_panics_fail_loud`; pins the deferred-set guarantee so a future
    // change that drops a deferred id from `is_fixed_damage_move` (or adds a wrong amount)
    // can't silently regress it. (Psywave / the OHKO moves / Mirror Coat / Bide / Endeavor
    // share this path.)
    #[test]
    #[should_panic(expected = "unmodeled FIXED-DAMAGE move")]
    fn deferred_fixed_damage_move_panics_fail_loud() {
        let d = dex();
        let snorlax = "Snorlax||leftovers||counter,bodyslam|Careful|252,,,,252,|||||";
        let tauros = "Tauros||leftovers||bodyslam,earthquake|Adamant|,252,,,,252|||||";
        let mut state = BattleState::start(&opts_cg(snorlax, tauros, "1,2,3,4"), &d).expect("start");
        // Counter is is_fixed_damage_move=true but fixed_damage_amount=None → fail-loud.
        let _ = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
    }

    // SPIKES (the entry hazard): the Spikes MOVE is never-miss + DRAW-FREE — it increments
    // the CASTER's FOE side's `spikes` layer by 1, capped at 3, and a 4th Spikes FAILS
    // (draw-free). `landed` is FALSE (no in-tryMoveHit Update).
    #[test]
    fn spikes_move_increments_foe_side_draw_free_and_caps_at_three() {
        let d = dex();
        let skarm = "Skarmory||leftovers||spikes,drillpeck|Impish|252,,252,,,|||||";
        let blissey = "Blissey||leftovers||softboiled,icebeam|Bold|252,,252,,,|||||";
        let mut state = BattleState::start(&opts_cg(skarm, blissey, "1,2,3,4"), &d).expect("start");
        for expect in [1u8, 2, 3, 3] {
            let before = state.prng_seed();
            let res = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
            // Spikes targets the FOE side (side 1); the caster's own side stays 0.
            assert_eq!(state.sides[1].spikes, expect, "spikes layer climbs 1→2→3, then caps (a 4th FAILS)");
            assert_eq!(state.sides[0].spikes, 0, "the caster's OWN side gets no spikes");
            assert!(!res.landed && !res.missed, "spikes is a never-miss status move (no land, no miss)");
            assert_eq!(state.prng_seed(), before, "the Spikes move draws NO PRNG (the side condition is draw-free)");
        }
    }

    // SPIKES switch-in damage: a GROUNDED entrant takes the gen-3 floor per layer
    // (maxhp/8, maxhp/6, maxhp/4), DRAW-FREE. The damage is applied in `run_switch`'s
    // `apply_entry_hazards` (the gen4-inherited runSwitch's EntryHazard step).
    #[test]
    fn spikes_switch_in_damage_grounded_per_layer_draw_free() {
        let d = dex();
        // p1 lays spikes on the p2 side; p2's bench Snorlax switches in and takes the chip.
        let skarm = "Skarmory||leftovers||spikes,drillpeck|Impish|252,,252,,,|||||";
        // A grounded bulky Snorlax (Normal — not Flying, not Levitate).
        let team2 = "Blissey||leftovers||softboiled|Bold|252,,252,,,|||||]Snorlax||leftovers||bodyslam|Careful|252,,,,252,|||||";
        for (layers, denom) in [(1u8, 8u16), (2, 6), (3, 4)] {
            let mut state = BattleState::start(&opts_cg(skarm, team2, "1,2,3,4"), &d).expect("start");
            state.sides[1].spikes = layers;
            // Snorlax is bench slot index 1; switch it into the active slot (runs runSwitch).
            let snorlax_maxhp = state.sides[1].pokemon[1].maxhp;
            let expect_dmg = (snorlax_maxhp / denom).max(1);
            let before = state.prng_seed();
            let mut queue: Vec<QAction> = Vec::new();
            state.execute_switch(1, 1, false, &d, &mut queue); // swap Snorlax to active + enqueue runSwitch
            // Run the enqueued runSwitch (the EntryHazard step applies spikes).
            assert!(matches!(queue.first(), Some(QAction::RunSwitch { side: 1 })));
            state.run_switch(1, &d);
            let snorlax = &state.sides[1].pokemon[state.sides[1].active];
            assert_eq!(snorlax.species_id, "snorlax", "Snorlax is now active");
            assert_eq!(
                snorlax.hp, snorlax_maxhp - expect_dmg,
                "{layers}-layer spikes chips a grounded entrant maxhp/{denom} = {expect_dmg}"
            );
            assert_eq!(state.prng_seed(), before, "the spikes switch-in damage is DRAW-FREE");
        }
    }

    // SPIKES grounded gate: a FLYING-type or LEVITATE entrant takes ZERO spikes damage.
    #[test]
    fn spikes_switch_in_flying_and_levitate_take_zero() {
        let d = dex();
        let skarm = "Skarmory||leftovers||spikes,drillpeck|Impish|252,,252,,,|||||";
        // bench slot 1 = Salamence (Flying-type → immune), slot 2 = Claydol (Ground/Psychic
        // but LEVITATE ability → immune). The ability field (4th, after item) MUST carry
        // `levitate` for the grounded gate to read it.
        let team2 = "Blissey|||naturalcure|softboiled|Bold|252,,252,,,|||||]Salamence|||intimidate|dragonclaw|Adamant|252,252,,,,|||||]Claydol|||levitate|psychic|Bold|252,,252,,,|||||";
        for slot in [1usize, 2] {
            let mut state = BattleState::start(&opts_cg(skarm, team2, "1,2,3,4"), &d).expect("start");
            state.sides[1].spikes = 3; // max layers — still ZERO for an immune entrant
            let maxhp = state.sides[1].pokemon[slot].maxhp;
            let mut queue: Vec<QAction> = Vec::new();
            state.execute_switch(1, slot, false, &d, &mut queue);
            state.run_switch(1, &d);
            let entrant = &state.sides[1].pokemon[state.sides[1].active];
            assert_eq!(entrant.hp, maxhp, "a Flying/Levitate entrant ({}) takes ZERO spikes", entrant.species_id);
        }
    }

    // SPIKES KO on switch-in: a grounded entrant at low HP whose spikes chip zeroes its HP
    // is left at 0 HP by `run_switch` (the runAction tail then faints it + forces a
    // replacement). The ability `Start` is SKIPPED (`if (!pokemon.hp) return`). DRAW-FREE.
    #[test]
    fn spikes_ko_on_switch_in_zeroes_hp_draw_free() {
        let d = dex();
        let skarm = "Skarmory||leftovers||spikes,drillpeck|Impish|252,,252,,,|||||";
        // bench slot 1 = a tiny lvl-1 Diglett (grounded). At 3 layers, spikes = floor(maxhp/4).
        // Packed fields: name|species|item|ability|moves|nature|evs|gender|ivs|shiny|level → level=1.
        let team2 = "Blissey|||naturalcure|softboiled|Bold|252,,252,,,|||||]Diglett|||sandveil|scratch|Hardy|||||1|";
        let mut state = BattleState::start(&opts_cg(skarm, team2, "1,2,3,4"), &d).expect("start");
        state.sides[1].spikes = 3;
        // Pre-chip the Diglett to 1 HP so the 3-layer spikes (floor(maxhp/4) >= 1) KOs it on entry.
        state.sides[1].pokemon[1].hp = 1;
        let before = state.prng_seed();
        let mut queue: Vec<QAction> = Vec::new();
        state.execute_switch(1, 1, false, &d, &mut queue);
        state.run_switch(1, &d);
        let entrant = &state.sides[1].pokemon[state.sides[1].active];
        assert_eq!(entrant.species_id, "diglett");
        assert_eq!(entrant.hp, 0, "the spikes chip KO'd the low-HP entrant on switch-in");
        assert_eq!(state.prng_seed(), before, "the spikes KO is DRAW-FREE (no Quick Claw / extra draw)");
    }

    // ── TAUNT + DISABLE (gen3_taunt_disable_v1) ───────────────────────────────

    // TAUNT (Dark, acc 100): the move draws ONLY its accuracy roll (NO duration draw — the
    // taunt volatile is duration:2 FIXED). It applies the taunt volatile to the FOE.
    #[test]
    fn taunt_draws_only_accuracy_and_applies_the_volatile_draw_free_duration() {
        let d = dex();
        let gengar = "Gengar||leftovers||taunt,shadowball|Timid|,,,252,4,252|||||";
        let snorlax = "Snorlax||leftovers||bodyslam,toxic|Careful|252,,,,252,|||||";
        let mut state = BattleState::start(&opts_cg(gengar, snorlax, "1,2,3,4"), &d).expect("start");
        let before = state.prng_seed();
        // Gengar (side 0) uses Taunt (slot 0) at the foe. foe_will_move irrelevant for taunt.
        let res =
            state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        assert!(!res.landed && !res.missed, "taunt is a status move: no land, no miss (acc 100 passes)");
        assert!(state.sides[1].pokemon[0].taunt.is_some(), "the foe is taunted");
        assert_eq!(state.sides[1].pokemon[0].taunt, Some(TAUNT_DURATION), "taunt duration is the FIXED 2");
        // The oracle: EXACTLY one accuracy draw (randomChance(100,100)) — NO random(2,6).
        let mut oracle = crate::prng::Prng::new(before.as_str());
        let _ = oracle.random_chance(100, 100);
        assert_eq!(
            state.prng_seed(), oracle.get_seed(),
            "TAUNT draws ONLY accuracy (1 draw) — the volatile duration is draw-free (duration:2 fixed)"
        );
    }

    // TAUNT selection restriction: a taunted mon can't SELECT any Status move (`move_usable`
    // is false for a Status slot) but its attacking slots stay usable.
    #[test]
    fn taunt_makes_status_moves_unusable_but_leaves_attacks() {
        let d = dex();
        let gengar = "Gengar||leftovers||taunt,shadowball|Timid|,,,252,4,252|||||";
        // Snorlax: slot 0 Body Slam (Physical), slot 1 Toxic (Status), slot 2 Rest (Status).
        let snorlax = "Snorlax||leftovers||bodyslam,toxic,rest|Careful|252,,,,252,|||||";
        let mut state = BattleState::start(&opts_cg(gengar, snorlax, "1,2,3,4"), &d).expect("start");
        // Taunt the Snorlax directly (bypass the move to isolate move_usable).
        state.sides[1].pokemon[0].taunt = Some(2);
        let lax = &state.sides[1].pokemon[0];
        assert!(lax.move_usable(0, &d), "Body Slam (Physical) stays usable while taunted");
        assert!(!lax.move_usable(1, &d), "Toxic (Status) is UN-usable while taunted");
        assert!(!lax.move_usable(2, &d), "Rest (Status) is UN-usable while taunted");
        assert!(!lax.must_struggle(&d), "an attacking move remains → NOT forced Struggle");
    }

    // TAUNT forces Struggle when EVERY usable move is Status.
    #[test]
    fn taunt_forces_struggle_when_only_status_moves_remain() {
        let d = dex();
        let gengar = "Gengar||leftovers||taunt,shadowball|Timid|,,,252,4,252|||||";
        // Blissey with ONLY Status moves (Toxic + Soft-Boiled).
        let blissey = "Blissey||leftovers||toxic,softboiled|Calm|252,,,252,,|||||";
        let mut state = BattleState::start(&opts_cg(gengar, blissey, "1,2,3,4"), &d).expect("start");
        state.sides[1].pokemon[0].taunt = Some(2);
        assert!(
            state.sides[1].pokemon[0].must_struggle(&d),
            "a taunted mon whose ONLY moves are Status is forced to Struggle"
        );
    }

    // TAUNT residual tick + expiry: the volatile ticks 2→1 at the first residual and
    // 1→None at the second, DRAW-FREE. Modeled by driving two residuals.
    #[test]
    fn taunt_ticks_down_and_expires_at_the_residual_draw_free() {
        let d = dex();
        let gengar = "Gengar||leftovers||taunt,shadowball|Timid|,,,252,4,252|||||";
        let snorlax = "Snorlax||leftovers||bodyslam,toxic|Careful|252,,,,252,|||||";
        let mut state = BattleState::start(&opts_cg(gengar, snorlax, "1,2,3,4"), &d).expect("start");
        state.sides[1].pokemon[0].taunt = Some(2);
        // First residual: 2 → 1 (still taunted).
        let before = state.prng_seed();
        state.run_residuals(&d);
        assert_eq!(state.sides[1].pokemon[0].taunt, Some(1), "residual ticks taunt 2→1");
        // Second residual: 1 → None (expired).
        state.run_residuals(&d);
        assert_eq!(state.sides[1].pokemon[0].taunt, None, "residual expires taunt at 0 → None");
        // No leftovers/status/weather here except Leftovers heals (draw-free); no shuffle
        // fires (distinct speeds), so the seed is unchanged across BOTH residuals.
        assert_eq!(state.prng_seed(), before, "the taunt duration tick is DRAW-FREE (no residual shuffle here)");
    }

    // DISABLE (Normal, acc 55): on a landed hit into a mon with a lastMove, the move draws
    // accuracy + ONE random(2,6) for the duration, and disables the lastMove's slot.
    //
    // The STORED DURATION is the sim's POST-onStart value (the residual DisableDuration handler
    // then ticks it -1 each residual, so the stored value must equal the sim's post-onStart
    // `effectState.duration`). GROUND TRUTH — measured DIRECTLY from the omniscient sim's
    // `addVolatile(disable)` return, NOT self-reconstructed (`harness/probe_disable_full_lifecycle.js`
    // + `probe_disable_onstart.js`):
    //   - disabler moved FIRST  (`willMove(target)` TRUE  / `foe_will_move` TRUE)  → stored = rolled
    //   - disabler moved SECOND (`willMove(target)` FALSE / `foe_will_move` FALSE) → stored = rolled+1
    // (The differential `tests/taunt_disable_test.rs` is the AUTHORITATIVE gate — it asserts the
    // disabled slot per boundary vs the sim, so a wrong duration on EITHER branch frees the move a
    // boundary early/late and FAILS there. This unit test pins the exact per-branch draw count +
    // stored value in isolation.)
    #[test]
    fn disable_draws_accuracy_plus_one_random_2_6_and_stores_the_sim_duration_per_branch() {
        let d = dex();
        let aero = "Aerodactyl||leftovers||disable,rockslide|Jolly|,252,,,4,252|||||";
        let snorlax = "Snorlax||leftovers||bodyslam,rest|Careful|252,,,,252,|||||";
        // Find a seed where the acc-55 Disable LANDS (the first draw < 55).
        let mut landed = None;
        for s in 0u32..40 {
            let seed = format!("{},{},{},{}", s + 1, s * 7 + 3, s * 13 + 5, s * 17 + 7);
            let mut probe = crate::prng::Prng::new(&seed);
            if probe.random_chance(55, 100) {
                landed = Some(seed);
                break;
            }
        }
        let seed = landed.expect("a landing seed exists");

        // For BOTH `foe_will_move` branches: run the disable, assert the stored duration is the
        // sim's post-onStart value (rolled for FASTER, rolled+1 for SLOWER) AND the draw count
        // (accuracy + exactly one random(2,6)).
        for (foe_will_move, expected) in [(true, 0u32), (false, 1u32)] {
            let mut state = BattleState::start(&opts_cg(aero, snorlax, &seed), &d).expect("start");
            // Snorlax's lastMove = slot 0 (Body Slam); it hasn't switched.
            state.sides[1].pokemon[0].last_move = Some(0);
            let before = state.prng_seed();
            let res = state.run_move(
                MoveAction { side: 0, slot: 0, move_index: 0, struggle: false },
                true,
                foe_will_move,
                &d,
            );
            assert!(!res.landed && !res.missed, "a landed Disable is a status move: no land, no miss");
            let dis = state.sides[1].pokemon[0].disable.expect("the foe's last move is disabled");
            assert_eq!(dis.0, 0, "the DISABLED slot is the target's lastMove slot (Body Slam = slot 0)");
            // The oracle: accuracy(55) THEN random(2,6) — exactly two draws, either branch.
            let mut oracle = crate::prng::Prng::new(before.as_str());
            assert!(oracle.random_chance(55, 100), "the chosen seed lands the acc-55 Disable");
            let rolled = oracle.random_range(2, 6);
            assert_eq!(
                dis.1,
                (rolled + expected) as u8,
                "stored duration = the SIM's post-onStart value: rolled+{expected} for \
                 foe_will_move={foe_will_move} (rolled for FASTER disabler, rolled+1 for SLOWER; \
                 measured DIRECTLY from the sim's addVolatile return)"
            );
            assert_eq!(
                state.prng_seed(), oracle.get_seed(),
                "DISABLE draws accuracy + EXACTLY one random(2,6) (the duration) — either branch"
            );
        }
    }

    // DISABLE into a mon with NO lastMove (never moved / just switched in): the onTryHit
    // FAILS draw-free — accuracy is drawn, then NO random(2,6). No volatile.
    #[test]
    fn disable_into_no_last_move_draws_only_accuracy_no_duration() {
        let d = dex();
        let aero = "Aerodactyl||leftovers||disable,rockslide|Jolly|,252,,,4,252|||||";
        let snorlax = "Snorlax||leftovers||bodyslam,rest|Careful|252,,,,252,|||||";
        // A seed where the acc-55 Disable LANDS (so the fail is the no-lastMove path, not a miss).
        let mut landed = None;
        for s in 0u32..40 {
            let seed = format!("{},{},{},{}", s + 1, s * 7 + 3, s * 13 + 5, s * 17 + 7);
            let mut probe = crate::prng::Prng::new(&seed);
            if probe.random_chance(55, 100) {
                landed = Some(seed);
                break;
            }
        }
        let seed = landed.expect("a landing seed exists");
        let mut state = BattleState::start(&opts_cg(aero, snorlax, &seed), &d).expect("start");
        // Snorlax has NO lastMove (default None → never moved).
        assert!(state.sides[1].pokemon[0].last_move.is_none());
        let before = state.prng_seed();
        let res =
            state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        assert!(!res.landed && !res.missed, "a no-lastMove Disable status move: no land, no miss");
        assert!(state.sides[1].pokemon[0].disable.is_none(), "no volatile — onTryHit failed");
        // The oracle: EXACTLY one accuracy draw — NO random(2,6).
        let mut oracle = crate::prng::Prng::new(before.as_str());
        let _ = oracle.random_chance(55, 100);
        assert_eq!(
            state.prng_seed(), oracle.get_seed(),
            "a no-lastMove Disable draws ONLY accuracy (onTryHit fails BEFORE the random(2,6))"
        );
    }

    // DISABLE selection restriction: the disabled slot is UN-usable, other slots stay usable.
    #[test]
    fn disable_makes_the_one_slot_unusable_but_leaves_the_rest() {
        let d = dex();
        let aero = "Aerodactyl||leftovers||disable,rockslide|Jolly|,252,,,4,252|||||";
        let snorlax = "Snorlax||leftovers||bodyslam,earthquake,rest|Careful|252,,,,252,|||||";
        let mut state = BattleState::start(&opts_cg(aero, snorlax, "1,2,3,4"), &d).expect("start");
        // Disable Snorlax's slot 1 (Earthquake).
        state.sides[1].pokemon[0].disable = Some((1, 3));
        let lax = &state.sides[1].pokemon[0];
        assert!(lax.move_usable(0, &d), "Body Slam (slot 0) stays usable");
        assert!(!lax.move_usable(1, &d), "the DISABLED Earthquake (slot 1) is UN-usable");
        assert!(lax.move_usable(2, &d), "Rest (slot 2) stays usable");
        assert!(!lax.must_struggle(&d), "other moves remain → NOT forced Struggle");
    }

    // DISABLE residual tick + expiry: the volatile ticks down + frees the move at 0, DRAW-FREE.
    #[test]
    fn disable_ticks_down_and_expires_at_the_residual_draw_free() {
        let d = dex();
        let aero = "Aerodactyl||leftovers||disable,rockslide|Jolly|,252,,,4,252|||||";
        let snorlax = "Snorlax||leftovers||bodyslam,earthquake|Careful|252,,,,252,|||||";
        let mut state = BattleState::start(&opts_cg(aero, snorlax, "1,2,3,4"), &d).expect("start");
        state.sides[1].pokemon[0].disable = Some((1, 2));
        let before = state.prng_seed();
        state.run_residuals(&d);
        assert_eq!(state.sides[1].pokemon[0].disable, Some((1, 1)), "residual ticks disable 2→1");
        state.run_residuals(&d);
        assert_eq!(state.sides[1].pokemon[0].disable, None, "residual expires disable at 0 → the move frees up");
        assert!(state.sides[1].pokemon[0].move_usable(1, &d), "the freed move is usable again");
        assert_eq!(state.prng_seed(), before, "the disable duration tick is DRAW-FREE");
    }

    // TAUNT + DISABLE clear on switch-out + faint (clearVolatile). And last_move clears too.
    #[test]
    fn taunt_disable_last_move_clear_on_switch_out() {
        let d = dex();
        let gengar = "Gengar||leftovers||shadowball,nightshade|Timid|,,,252,4,252|||||";
        let team2 = "Snorlax||leftovers||bodyslam,earthquake|Careful|252,,,,252,|||||]Blissey||leftovers||softboiled,icebeam|Bold|252,,252,,,|||||";
        let mut state = BattleState::start(&opts_cg(gengar, team2, "1,2,3,4"), &d).expect("start");
        state.sides[1].pokemon[0].taunt = Some(2);
        state.sides[1].pokemon[0].disable = Some((0, 3));
        state.sides[1].pokemon[0].last_move = Some(0);
        let mut queue: Vec<QAction> = Vec::new();
        state.execute_switch(1, 1, false, &d, &mut queue); // Snorlax out, Blissey in
        // The outgoing Snorlax (now bench slot 1) is un-restricted.
        let out = &state.sides[1].pokemon[1];
        assert!(out.taunt.is_none() && out.disable.is_none() && out.last_move.is_none(),
            "taunt/disable/last_move all clear on switch-out (clearVolatile)");
    }
}


