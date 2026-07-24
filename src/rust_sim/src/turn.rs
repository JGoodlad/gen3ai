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

use crate::prng::PrngSeed;
use crate::state::{Status, Weather, BOOST_LEN};

mod driver;
mod helpers;
mod items;
mod moves;
mod residuals;
mod secondaries;
mod speed;
mod status;
mod status_moves;
mod switch;

/// Re-export the live per-mon TYPE read (species types ⊕ Color Change `types_override`)
/// so the bridge's `|request|` serializer can price Curse's `nonGhostTarget` off the SAME
/// runtime types the engine uses (`gen3_bridge_curse_request_target_v1`).
pub(crate) use helpers::mon_types;

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

/// The gen-3 TIMED-WEATHER MOVE duration (`gen3_move_coverage_batch2_v1`) — Rain Dance /
/// Sunny Day set `weather_turns = 5` (`raindance`/`sunnyday` `durationCallback` yields 5;
/// gen3 has no Damp/Heat Rock → always 5, VERIFIED vs the sim). Distinct from the
/// ability-source PERMANENT weather (`weather_turns = 0`). Decremented ONCE per end-of-turn
/// FIELD residual; at 0 the weather clears (`|-weather|none`).
const WEATHER_MOVE_DURATION: u8 = 5;

/// The gen-3 SCREEN duration (`gen3_move_coverage_batch2_v1`) — Light Screen / Reflect set
/// `light_screen`/`reflect = 5` (`lightscreen`/`reflect` `durationCallback` yields 5; gen3
/// has no Light Clay → always 5, VERIFIED). Decremented ONCE per end-of-turn SIDE residual;
/// at 0 the screen expires (`|-sideend|…`).
const SCREEN_DURATION: u8 = 5;

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

/// The **YAWN** delayed-sleep volatile's residual `onResidualSubOrder` (`gen3_yawn_v1`) — the
/// gen-3 `yawn` condition carries `onResidualOrder: 10, onResidualSubOrder: 19` (VERIFIED vs the
/// resolved `Dex.mod('gen3')`). At the shared order 10 it sorts AFTER Leftovers (4) / Leech (5) /
/// status DoT (6) / Curse (8) / Encore (14) / Taunt (15) at equal speed — unique among the
/// order-10 handlers, so its ONLY residual tie is the OTHER mon's yawn at equal cached speed (a
/// yawn MIRROR → one `random(0,2)` handler-sort tie-shuffle). Its handler decrements the duration
/// and, at 1 → 0, fires the `onEnd` (`-end [silent]` + `trySetStatus('slp')`).
const YAWN_RESIDUAL_SUBORDER: i32 = 19;

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
/// FAIL-LOUD runaway watchdogs (`gen3_bridge_turn_watchdog_v1`). A single turn processes a
/// BOUNDED number of queued actions (both moves + any switches + their RunSwitches + the
/// residual — a few dozen at most), and a whole battle terminates in a few hundred turns
/// (the ladder/env forfeits a stall well before). If a queued action ever re-enqueues in a
/// cycle, or a battle never reaches a terminal state, the engine would spin at 100% CPU
/// forever — a SILENT hang that deadlocks the `SubprocVecEnv` step barrier in a
/// `--use-bridge=rust` training run (the launcher auto-restarts a CRASH but not a 0-FPS
/// wedge). These caps sit astronomically above any legitimate battle and PANIC with the
/// state, converting a hang into a catchable crash + a self-documenting repro. NEVER reached
/// on any real gen-3 battle; a trip is always a bug.
const TURN_LOOP_ACTION_CAP: usize = 100_000;
const BATTLE_TURN_CAP: u32 = 1_000;
/// A duration-only volatile's residual handler subOrder. `resolvePriority` falls back to
/// the effect's `effectTypeOrder` when no `onResidualSubOrder` is set; for a Condition
/// (the `protect`/`stall` volatiles) that is **2** (`battle.ts` effectTypeOrder), VERIFIED
/// vs the sim's residual speed-sort dump (`protect[sub=2]` / `stall[sub=2]`). Both protect
/// and stall use it, so they tie (with each other) at order NO_ORDER — the tie-group
/// shuffle the protect golden's seed parity pins.
const VOLATILE_RESIDUAL_SUBORDER: i32 = 2;
/// The **WISH** slot condition's `onResidualOrder` **7** (`gen3_move_coverage_batch3_v1`,
/// the resolved gen-3 `wish` condition — NO subOrder). Smaller than the sand chip's order 8
/// and every order-10 mon handler, so the Wish heal fires FIRST among the residuals (VERIFIED
/// vs the sim: on the resolve turn `-heal Wish` precedes `-heal Leftovers` precedes `-damage
/// brn`). The handler participates in the residual speed-sort with the slot's active mon's
/// cached speed, so two Wishes resolving the same turn at EQUAL speed draw ONE tie-shuffle
/// `random(0,2)` (probe: a Blissey-mirror both-Wish resolve turn draws +1 vs a single-Wish
/// control; distinct-speed draws none). Its subOrder is unset → `resolvePriority`'s
/// effectTypeOrder default; unobservable (order 7 is unique among the modeled residuals).
const WISH_RESIDUAL_ORDER: u64 = 7;
/// The **WISH** slot condition's duration (`wish` condition `duration: 2`,
/// `gen3_move_coverage_batch3_v1`): cast turn decrements 2 → 1, next turn 1 → 0 → the
/// `onEnd` heal fires. So the heal lands at the END of the turn AFTER cast.
const WISH_DURATION: u8 = 2;
/// The **CURSE** volatile's `onResidualSubOrder` **8** (`gen3_move_coverage_batch3_v1`, the
/// resolved gen-3 `curse` condition `onResidualOrder: 10, onResidualSubOrder: 8`). So at
/// order 10 the residual ladder is ability sub 3 → Leftovers sub 4 → leech sub 5 → status DoT
/// sub 6 → **CURSE sub 8** → Taunt sub 15: Curse's floor(maxhp/4) chip on the cursed foe
/// fires AFTER the Leftovers/leech/DoT. VERIFIED vs the sim's residual handler dump. Gathered
/// with the VOLATILES (after leech, before the item / the NO_ORDER duration volatiles),
/// mirroring `findPokemonEventHandlers`'s status→volatiles→item order.
const CURSE_RESIDUAL_SUBORDER: i32 = 8;
/// The **ENCORE** volatile's residual sort key (`gen3_move_coverage_batch6_v1` — the
/// resolved gen-3 `encore` condition's `onResidualOrder: 10, onResidualSubOrder: 14`,
/// ONE below Taunt's 15). At order 10 it sorts AFTER Leftovers(4)/leech(5)/DoT(6)/
/// curse(8) and BEFORE taunt(15). The handler visit does BOTH the duration decrement
/// (the generic `effectState.duration--` — `|-end|` at 0) AND the `onResidual` pp
/// check (the encored slot at 0 PP removes the volatile EARLY, same-turn `-end` even
/// at duration 5 — probe EN5). Its only tie is the OTHER mon's encore at equal speed.
const ENCORE_RESIDUAL_SUBORDER: i32 = 14;
/// The **PERISH SONG** counter's residual order (`gen3_move_coverage_batch6_v1` — the
/// resolved gen-3 `perishsong` condition's `onResidualOrder: 12`, NO subOrder → the
/// Condition effectTypeOrder default 2). Order 12 = LAST in the modeled ladder
/// (probed: Leftovers 10.4 → brn 10.6 → futuremove 11 → **perish 12**). The handler
/// visit decrements the duration THEN prints `|-start|<mon>|perish<duration>`; the
/// 1 → 0 tick fires `onEnd` (`perish0` + faint) instead. TWO perished mons at EQUAL
/// cached speed are a size-2 tie group (ONE `random(0,2)` per residual — the
/// duration decrement and onResidual ride the SAME handler visit, probe P5).
const PERISH_RESIDUAL_ORDER: u64 = 12;
/// The **FUTUREMOVE** slot condition's `onResidualOrder` **11** (`gen3_move_coverage_
/// batch4c_v1`, the resolved gen-3 `futuremove` condition — NO subOrder). Order 11 sits
/// AFTER every order-10 mon handler (incl. Taunt's 10/15) and BEFORE Truant's 27 / the
/// NO_ORDER duration volatiles. OBSERVED one-turn
/// order (probe `harness/probe_batch4c_doomdesire.js`, the Celebi/Tyranitar pin): Wish
/// heal(7) → weather upkeep + sand chip(8) → Leftovers(10.4) → futuremove `-end`+damage
/// (11) LAST. The handler participates in the residual speed-sort with the SLOT's active
/// mon's cached speed (an equal-speed FS MIRROR draws ONE tie-shuffle per residual —
/// cast/idle/resolve alike), gathered EVERY end-of-turn while pending.
const FUTURE_RESIDUAL_ORDER: u64 = 11;
/// The **FUTUREMOVE** slot condition's duration (`futuremove` condition `duration: 3`,
/// `gen3_move_coverage_batch4c_v1`): cast-turn residual 3 → 2, idle turn 2 → 1, and the
/// 1 → 0 tick RESOLVES the strike at the end of turn N+2 (cast t1 → `-end` at end of t3).
const FUTURE_MOVE_DURATION: u8 = 3;
/// TRUANT's residual toggle (`gen3_ability_batch4_v1`) — the resolved
/// `truant.onResidualOrder: 27` (base data, gen3-inherited): its OWN order group,
/// AFTER every order-10 mon handler, BEFORE the NO_ORDER duration-only volatiles.
/// The toggle is DRAW-FREE; the ONLY possible order-27 tie is the other side's
/// Truant at equal speed (a Slaking mirror), which adds exactly ONE tie-shuffle
/// draw (probe_truant_rng.js Q4: tied mirror 9 shuffles vs the no-truant control's
/// 8). Its subOrder is unobservable within the group (both members share it) —
/// the ability effectTypeOrder slot is reused.
const TRUANT_RESIDUAL_ORDER: u64 = 27;
/// **WHITE HERB** end-of-turn restore (`gen3_white_herb_v1`, `whiteherb.onResidualOrder = 29`,
/// `onResidualSubOrder = undefined`) — the LAST modeled residual handler, its own order group
/// AFTER Truant (27). The resolved gen3 item's `onResidual` calls its `onStart` (scan `boosts<0`
/// → restore + consume). It is the site that restores an Intimidate-switch-in drop when the
/// holder does NOT move that turn (a forced-replacement / no-move turn — the omniscient byte-fuzz
/// find rmry3vbgm_ab_7_4: the sim restores at the residual, the port kept the −1). Gathered
/// UNCONDITIONALLY for an active White Herb holder (the item's onResidual registers regardless of
/// the boost state — the `boosts<0` gate lives in the apply), so it participates in the residual
/// tie-shuffle; its ONLY tie is a two-White-Herb-holder board at equal speed (one extra draw).
const WHITE_HERB_RESIDUAL_ORDER: u64 = 29;
/// The **SCREEN** side conditions' residual sort keys (`gen3_screen_residual_tie_shuffle_v1`) —
/// the gen4-mod override gen3 inherits (`data/mods/gen4/moves.ts`: `reflect.condition
/// onSideResidualOrder: 1`, `lightscreen.condition onSideResidualOrder: 2`; both
/// `onSideResidualSubOrder: undefined`). Order 1/2 sort FIRST — well before the weather field
/// residual (order 8) and every mon handler (order 10) — so a single screen's `-sideend` expiry
/// stays first among the residuals (its old direct-decrement stream position). priority 0.
const REFLECT_RESIDUAL_ORDER: u64 = 1;
const LIGHT_SCREEN_RESIDUAL_ORDER: u64 = 2;
/// A (non-slot) SIDE condition's residual subOrder — `resolvePriority`'s effectTypeOrder for a
/// `Condition` whose `state.target instanceof Side` (not a slot condition) = **4**
/// (`sim/battle.ts` `resolvePriority`). Unobservable for the sort ORDER (reflect 1 / lightscreen
/// 2 already differ), but load-bearing for the TIE: the two same-type screens share it, so they
/// tie exactly (order/priority/speed/subOrder all equal) → the size-2 shuffle draws.
const SIDE_CONDITION_SUBORDER: i32 = 4;

/// A resolved per-move action, in the order it will be run. `slot` is the team
/// slot of the actor; `target_slot` the opposing active.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MoveAction {
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
pub(crate) struct MoveResolution {
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
pub(crate) enum SubAbsorb {
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
    /// The **MUSTRECHARGE** (`mustrecharge`, Hyper Beam) `duration: 2` residual duration
    /// handler on the CAST turn (`gen3_perside_residual_faint_upkeep_order_v1` D4 fix). It
    /// registers the SAME NO_ORDER/subOrder-2 tie-group handler as the duration:1 group, but —
    /// UNLIKE that group — it does NOT end this turn: `mustrecharge` is `duration: 2`, so its
    /// only residual tick (the cast turn) decrements 2 → 1 (the volatile is removed by the NEXT
    /// turn's recharge cant, never by a residual). Per the sim's `fieldEvent('Residual')`, a
    /// duration handler whose `duration--` is NON-ZERO FALLS THROUGH to the per-handler
    /// `faintMessages()` (only a `duration-- == 0` handler `continue`s past it). So this arm must
    /// NOT `continue` (the blanket `else { continue; }` the duration:1 group uses would SKIP a
    /// faint an earlier order-≤12 handler [e.g. Perish] enqueued-but-deferred, mis-emitting that
    /// mon's `|faint|` BEFORE `|upkeep|`). No-op apply; participates in the tie-shuffle like its
    /// siblings.
    MustRechargeDuration { side: usize, slot: usize },
    /// The **TAUNT** volatile's residual duration handler (`gen3_taunt_disable_v1`, order 10,
    /// subOrder 15): decrement `MonState::taunt` and, on reaching 0, CLEAR it (→ `None`, freeing
    /// the mon's Status moves). NO HP effect (`taunt.onResidual` only ticks the duration + emits
    /// `-end`), DRAW-FREE. It participates in the residual speed-sort (a taunted mon adds ONE
    /// tied handler at order 10 subOrder 15 — its only tie is the OTHER mon's taunt at equal
    /// speed).
    TauntDuration { side: usize, slot: usize },
    /// The **YAWN** volatile's residual duration handler (`gen3_yawn_v1`, order 10, subOrder 19):
    /// decrement `MonState::yawn`'s duration and, at 1 → 0, fire the `onEnd` — emit
    /// `|-end|<target>|move: Yawn|[silent]` then `trySetStatus('slp', source)` via the existing
    /// [`crate::state::BattleState::try_set_status`] path (so the sleep `random(2,6)` onStart draw,
    /// the gen3ou Sleep Clause block, AND the gen3ou SetStatus 2-clause shuffle all come for free).
    /// The 2 → 1 (cast-turn) tick is unobservable + draw-free. Its ONLY residual tie is the other
    /// mon's yawn at equal speed (unique subOrder 19). On the 1 → 0 (resolve) tick the sleep set
    /// draws the `random(2,6)` in gen3customgame / the SetStatus shuffle (+ the `random(2,6)` if it
    /// lands) in gen3ou.
    Yawn { side: usize, slot: usize },
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
    /// The **CURSE** residual chip (`gen3_move_coverage_batch3_v1`, order 10 subOrder 8):
    /// the CURSED holder loses `floor(maxhp/4)` (`curse.onResidual` → `this.damage(baseMaxhp/
    /// 4)`), emitting `|-damage|<foe>|<hp>|[from] Curse`. DRAW-FREE. Gathered with the
    /// volatiles (after leech, before Taunt). The `source_side` is stored for parity with
    /// leech but the chip has no source-heal (unlike leech).
    Curse { side: usize, slot: usize },
    /// The **WISH** slot condition's delayed heal (`gen3_move_coverage_batch3_v1`, order 7 —
    /// BEFORE the sand chip + all order-10 handlers): decrement the side's `wish_pending`
    /// duration; on reaching 0, if the slot's active mon is not fainted, heal `floor(maxhp/2)`
    /// (a NON-zero heal emits `|-heal|<mon>|<hp>|[from] move: Wish|[wisher] <name>`; a
    /// heal-at-full resolves SILENTLY). DRAW-FREE apply; the handler ties-shuffles with the
    /// other side's Wish at equal speed. `side` is the slot (the healed side).
    Wish { side: usize },
    /// TRUANT's end-of-turn toggle (`gen3_ability_batch4_v1`, order **27** — its own
    /// group, AFTER every order-10 handler): `truant_turn = !truant_turn`, DRAW-FREE.
    /// Gathered UNCONDITIONALLY for an active Truant holder (the parity clock ticks on
    /// move AND loaf turns alike — probe_truant_rng.js Q1/Q2); its only tie is a Truant
    /// mirror at equal speed (ONE extra shuffle draw, probe Q4). A holder that fainted
    /// earlier this residual is skipped (the effectHolder.fainted guard), which is what
    /// leaves a POST-residual replacement's armed `true` un-toggled (probe edge E1).
    TruantToggle { side: usize, slot: usize },
    /// **WHITE HERB**'s end-of-turn restore (`gen3_white_herb_v1`, order **29** — the LAST
    /// residual, its own group after Truant 27): `white_herb_restore` — if the holder has any
    /// negative boost, restore the negatives to 0 + consume the item (`-enditem`/`-clearnegativeboost`);
    /// else a no-op. DRAW-FREE apply. Gathered UNCONDITIONALLY for an active holder (the item's
    /// onResidual registers regardless of the boost state), so it participates in the residual
    /// tie-shuffle (its only tie is another White Herb holder at equal speed). This restores an
    /// Intimidate-switch-in drop on a turn the holder does NOT move (the residual — vs
    /// `onAnyAfterMove` when it does move).
    WhiteHerb { side: usize, slot: usize },
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
    /// The **TWOTURNMOVE** volatile's residual duration handler (`gen3_move_coverage_
    /// batch4c_v1`, Solar Beam — order NO_ORDER, subOrder 2, the protect/stall/flinch tie
    /// group): decrement `MonState::two_turn`'s `duration` (2 → 1 on the charge-turn
    /// residual, 1 → 0 on the fire-turn residual → REMOVE the volatile — `twoturnmove.
    /// onEnd`'s `removeVolatile('solarbeam')` is a no-op by then when the beam fired).
    /// NO HP effect, DRAW-FREE apply; its PRESENCE changes the residual tie-shuffle COUNT
    /// (probe: the volatile registers on BOTH the charge-turn and fire-turn residuals —
    /// a same-cached-speed foe duration handler would tie it). After a fire-turn KO the
    /// lingering volatile is cleaned by the RESUMED tail's residual via this handler.
    TwoTurnDuration { side: usize, slot: usize },
    /// The pending **FUTUREMOVE** slot condition's residual handler (`gen3_move_coverage_
    /// batch4c_v1`, Doom Desire / Future Sight — order **11**, after every order-10 mon
    /// handler): decrement the side's `future_move` duration; on reaching 0 RESOLVE the
    /// strike (`onEnd` → ONE accuracy roll + the STORED damage fixed-damage-style + the
    /// two landed-resolve `eachEvent('Update')` shuffles — see `apply_future_move`).
    /// Gathered EVERY end-of-turn while pending (speed = the slot occupant's cached
    /// speed — an equal-speed FS mirror tie-shuffles once per residual). `side` is the
    /// TARGET slot's side.
    FutureMove { side: usize },
    /// The **ENCORE** volatile's residual handler (`gen3_move_coverage_batch6_v1`,
    /// order 10 subOrder 14 — the taunt-15 precedent): the generic duration decrement
    /// (`|-end|` at 0) + the `onResidual` 0-PP EARLY removal (the encored slot hit 0 PP
    /// → same-turn `-end` even at duration 5 — probe EN5). NO HP effect, DRAW-FREE
    /// apply; its only tie is the OTHER mon's encore at equal speed.
    EncoreDuration { side: usize, slot: usize },
    /// The **PERISH SONG** counter's residual handler (`gen3_move_coverage_batch6_v1`,
    /// order 12 — LAST in the ladder): decrement, then `|-start|<mon>|perish<d>`; the
    /// 1 → 0 tick fires `onEnd` — `perish0` + FAINT the holder (processed per-handler
    /// by the caller's `faintMessages`, so a mutual perish-out is a same-residual
    /// double faint → double replacement, NO Quick Claw). DRAW-FREE apply; two
    /// perished mons at equal cached speed tie (ONE shuffle per residual — P5).
    Perish { side: usize, slot: usize },
    /// A **SCREEN** side condition's residual DURATION handler (`gen3_screen_residual_tie_
    /// shuffle_v1`): Reflect (`onSideResidualOrder` 1) / Light Screen (order 2), priority 0,
    /// speed 0, subOrder 4 (the `resolvePriority` SideCondition effectTypeOrder). Gathered per
    /// side via `fieldEvent('Residual')`'s `findSideEventHandlers(side, 'onSideResidual',
    /// 'duration')` — a callback-less duration-only handler — so it SPEED-SORTS with every other
    /// residual handler: when BOTH sides carry the SAME screen the two tie (order 1↔1 / 2↔2,
    /// speed 0↔0) and the tie-group Fisher-Yates shuffle draws ONE `random(0,2)` (the draw the
    /// old direct-decrement MISSED — the residual sibling of the per-hit ModifyDamagePhase1
    /// shuffle). The apply mirrors the sim's `handler.state.duration--; if (!duration) end()`:
    /// decrement the counter, emit `|-sideend|<side>|<Effect>` at 0. `is_reflect` picks the
    /// counter/effect (`reflect` → `Reflect`, `light_screen` → `move: Light Screen`). NO HP
    /// effect, DRAW-FREE apply; the effectHolder is the SIDE (a fainted active never skips it).
    ScreenDuration { side: usize, is_reflect: bool },
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
    /// The active mon's HELD item's dex `num` (`0` = itemless, `gen3_trick_v1`) — so the
    /// Trick differential asserts WHICH item each side holds after an item SWAP (a two-item
    /// swap keeps `item_held` true on BOTH sides, so the boolean can't distinguish it — the
    /// num pins the identity). Copy-safe (`u16`); no gen-3 item has num 0, and both the
    /// omniscient sim (`dex.items.get('').num`) and the port map itemless → 0, so it is a
    /// collision-free itemless sentinel.
    pub item_num: u16,
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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
    /// The FIELD weather after this decision (`gen3_move_coverage_batch2_v1`) — `None` =
    /// clear. A MOVE-set weather (Rain Dance / Sunny Day) counts down + expires; the
    /// ability weather (Sand Stream) is permanent. The batch-2 differential asserts it.
    pub weather: Option<Weather>,
    /// The FIELD weather's remaining turns after this decision (`gen3_move_coverage_batch2_v1`,
    /// `field.weather_turns`). 0 = clear OR permanent (ability weather); 1..=5 = a MOVE-set
    /// timed weather counting down.
    pub weather_turns: u8,
    /// Per-side **Light Screen** remaining turns after this decision (`side.light_screen`,
    /// `gen3_move_coverage_batch2_v1`, 0..=5). A SIDE condition (reported per side).
    pub light_screen: [u8; 2],
    /// Per-side **Reflect** remaining turns after this decision (`side.reflect`,
    /// `gen3_move_coverage_batch2_v1`, 0..=5). A SIDE condition (reported per side).
    pub reflect: [u8; 2],
    /// Per-side **CURSE** flag after this decision (`gen3_move_coverage_batch3_v1`): whether
    /// each side's ACTIVE mon carries the `curse` volatile (a ghost Curse laid on it). A mon
    /// volatile (reported for the current active), cleared on switch/faint; Baton-Pass-transferable.
    pub curse: [bool; 2],
    /// Per-side **WISH-PENDING** duration after this decision (`gen3_move_coverage_batch3_v1`,
    /// `side.wish_pending`): the slot condition's remaining turns (0 = none, 2 → 1 → resolve).
    /// A SIDE/slot condition (reported per side regardless of which mon is active — it survives
    /// a switch).
    pub wish_pending: [u8; 2],
    /// Per-side **SUB-HP** after this decision (`gen3_move_coverage_batch3_v1`): the active
    /// mon's Substitute decoy HP (0 = no sub). Reported here so the batch-3 differential can
    /// assert a Baton-Passed sub's HP transfer (the substitute is a mon volatile).
    pub sub_hp: [u16; 2],
    /// Per-side pending **FUTUREMOVE** duration after this decision
    /// (`gen3_move_coverage_batch4c_v1`, `side.future_move`): the slot condition's remaining
    /// residual ticks (0 = none; 3 → 2 → 1 → the strike resolves on the 1→0 tick). A SIDE/slot
    /// condition like `wish_pending` (it survives the occupant switching/fainting).
    pub future_pending: [u8; 2],
    /// Per side, the ACTIVE mon's ENCORE remaining duration (`gen3_move_coverage_
    /// batch6_v1` — the sim's `volatiles.encore.duration`), 0 when not encored. The
    /// willMove ±1 duration branch + the residual tick + the 0-PP early end are all
    /// observable here.
    pub encore: [u8; 2],
    /// Per side, the ACTIVE mon's PERISH SONG counter (`volatiles.perishsong.
    /// duration`), 0 when none — 3 at the cast turn's boundary (4-at-apply minus the
    /// cast-turn residual tick) down to the faint.
    pub perish: [u8; 2],
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
pub(crate) enum QAction {
    /// `eachEvent('BeforeTurn')` action (order 4): one trailing-Update tail.
    BeforeTurn,
    /// A per-move `beforeTurnMove` action (order 5, between beforeTurn=4 and switch=103) —
    /// created for a move carrying a `beforeTurnCallback` (Focus Punch / Pursuit,
    /// `gen3_move_coverage_batch4_v1`). Runs the callback (draw-free: Focus Punch adds its
    /// `focuspunch` volatile to the user + emits `-singleturn`; Pursuit lays the `pursuit`
    /// volatile on the foe, skipped if the pursuer is frz/slp) then the standard gen<5
    /// runAction trailing `eachEvent('Update')` tail. Keyed by the actor's STABLE `uid`.
    /// Participates in the action-order `speed_sort` at order 5 (so two beforeTurnMove actions
    /// tie at order 5 → the mirror tie-shuffle).
    BeforeTurnMove { side: usize, uid: usize, move_index: usize },
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
            QAction::BeforeTurnMove { .. } => 5,
            QAction::RunSwitch { .. } => 101,
            QAction::Switch { .. } => 103,
            QAction::Move { .. } => 200,
            QAction::Residual => 300,
        }
    }
}


// ===========================================================================
// The resumable stepping primitive (`FullBattleDriver`) — ONE request boundary
// per fed `ScriptDecision`, on a LIVE `&mut BattleState`. This is the ONE
// turn-loop in the codebase: `run_full_battle` (batch), `BattleStream::write_line`
// (streaming), and `bridge::BridgeSession` (per-side) all drive THIS. Extracting
// the old monolithic `run_full_battle` outer-loop body here — behaviour-preserving
// by construction — lets a drive PAUSE at any request boundary and RESUME on the
// next fed decision, so the incremental (stream/bridge) paths never re-simulate a
// prior turn.
// ===========================================================================

/// The resumable state of a full-battle drive. Every per-turn control datum the
/// old `run_full_battle` kept in locals lives here so a drive can pause between
/// turns (`AwaitMove`) or mid-turn at a forced replacement (`AwaitSwitch`). Plain
/// data (the queue + choices are `Clone`/`Copy`), so a future `Battle::serialize`
/// can snapshot it mechanically (Tier 3 — not built here).
pub(crate) struct FullBattleDriver {
    decisions: Vec<DecisionRecord>,
    /// The per-side pending `move`-request accumulator (the sim's `side.choose`,
    /// FIRST-accepted-wins) — held across feeds while one side is set and the other
    /// isn't yet.
    pending: ScriptDecision,
    turn_already_opened: bool,
    /// COMMITTED-turn counter for the `BATTLE_TURN_CAP` runaway watchdog
    /// (`gen3_bridge_turn_watchdog_v1`, relocated from the old `run_full_battle`
    /// local into the primitive so it still fires on a non-terminating battle).
    committed_turns: u32,
    phase: DrivePhase,
}

/// Where a paused drive is waiting.
enum DrivePhase {
    /// Between turns, waiting for a top-of-turn `move` decision.
    AwaitMove,
    /// Paused mid-turn for a forced replacement (`makeRequest('switch')`).
    AwaitSwitch(SwitchWait),
    /// The battle ended (`winner`, `None` on a tie).
    Ended(Option<usize>),
}

/// The mid-turn pause state for a forced replacement — the saved turn tail plus
/// the per-side replacement accumulator, exactly the locals the old inner loop held
/// across a `NeedSwitch` pause.
struct SwitchWait {
    /// The saved turn-loop tail (`oldQueue`) the instaswitch(es) prepend before.
    queue: Vec<QAction>,
    /// Which side(s) must replace this pause.
    force: [bool; 2],
    /// The per-side replacement target accepted so far.
    have: [Option<usize>; 2],
    /// The request the NEXT boundary record answers (`ForceSwitch{force}`).
    request: RequestKind,
    /// The turn's first mover (carried across every boundary record of the turn).
    first_mover: Option<usize>,
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


#[cfg(test)]
mod tests {
    use super::*;
    use super::helpers::*;
    use crate::battle::{BattleOptions, PackedTeam, PlayerOptions};
    // External names the tests reference directly. Before the turn/ submodule split
    // these arrived via `use super::*` re-exporting turn.rs's own module-level `use`s;
    // post-split those `use`s were trimmed to what the non-test root code needs, so the
    // test module imports them explicitly (allow-guarded — a superset for robustness).
    #[allow(unused_imports)]
    use crate::damage::{
        calc_damage, AtkStatMod, BpMod, Combatant, DamageContext, MoveInput, Weather as DmgWeather,
    };
    #[allow(unused_imports)]
    use crate::dex::{to_id, Dex, DmgFold, MoveCategory, Type, TypeBoostFold};
    #[allow(unused_imports)]
    use crate::event::{single_event_ability_start, speed_sort, EventHandler, NO_ORDER};
    #[allow(unused_imports)]
    use crate::protocol::{Cause, HpStatus, MonRef, Player, ProtocolLine};
    #[allow(unused_imports)]
    use crate::state::{BattleState, FutureMove, Status, TwoTurnMove, Weather, BOOST_LEN};

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
    // TRICK (`gen3_trick_v1`) is now MODELED (the item-swap move) — so this smoke asserts the
    // SWAP instead of a fail-loud panic. (This test was re-keyed from a `#[should_panic]`
    // fail-loud guard as Trick moved from unmodeled → modeled, mirroring the Snatch re-key; the
    // deep draw model + fail/block set are pinned bit-for-bit in `regression_test.rs` +
    // `trick_test.rs`.) A Trick between two item-holders SWAPS the items and never lands.
    #[test]
    fn trick_swaps_the_two_items_and_never_lands() {
        let d = dex();
        // Suicune (Leftovers) Tricks a Choice-Band Snorlax → the items swap; not landed.
        let suicune = "Suicune||leftovers||trick,surf|Bold|252,,252,,,|||||";
        let snorlax = "Snorlax||choiceband||bodyslam,earthquake|Adamant|252,252,,,,|||||";
        let mut state = BattleState::start(&opts_cg(suicune, snorlax, "1,2,3,4"), &d).expect("start");
        let res = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        assert_eq!(to_id(&state.sides[0].pokemon[0].item), "choiceband", "user gains the foe's Choice Band");
        assert_eq!(to_id(&state.sides[1].pokemon[0].item), "leftovers", "foe gains the user's Leftovers");
        assert!(!res.landed && !res.missed, "a status Trick never lands and never misses");
    }

    // SNATCH (`gen3_snatch_v1`) is now MODELED (the LAST gen-3 status move — this closes
    // 722/722). The CAST is DRAW-FREE: it sets the `snatch` singleturn volatile on the
    // user + emits `|-singleturn|<user>|Snatch`, consuming NO PRNG (the seed is unchanged
    // by the cast — only the foe's Drill Peck + endTurn draw). (This test was re-keyed
    // from a fail-loud panic — the deep steal mechanics + draw model are pinned bit-for-bit
    // in `regression_test.rs` MC100-MC104 + `movecoverage_snatch_test.rs`.)
    #[test]
    fn snatch_cast_sets_the_volatile_draw_free() {
        let d = dex();
        // A FAST Gengar Snatch vs a slower Skarmory Drill Peck: the +4 cast sets the
        // volatile before the foe's attack, and its own execution draws NOTHING.
        let gengar = "Gengar||leftovers|levitate|snatch,thunderbolt|Timid|252,,,,,252|||||";
        let skarmory = "Skarmory||leftovers||drillpeck,spikes|Impish|252,,252,,,|||||";
        let mut state = BattleState::start(&opts_cg(gengar, skarmory, "1,2,3,4"), &d).expect("start");
        let before = state.prng_seed();
        let res = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        assert!(state.sides[0].pokemon[0].snatch, "Snatch sets the `snatch` volatile on the user");
        assert!(!res.landed && !res.missed, "a status Snatch never lands and never misses");
        assert_eq!(state.prng_seed(), before, "the Snatch cast consumes no PRNG");
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

    // ENDURE is MODELED as of `gen3_move_coverage_batch6_v1` (it rides the Protect
    // stallingMove machinery — the SHARED `stall` counter + the `endure` volatile's
    // onDamage survive-at-1 clamp). This replaces the old `endure_panics_fail_loud`
    // gate: a FIRST Endure succeeds DRAW-FREE (no stall roll at counter 0), sets the
    // `endure` volatile + the shared counter to 2, and does NOT set `protected`.
    #[test]
    fn endure_first_use_draw_free_sets_the_volatile_and_shared_stall_counter() {
        let d = dex();
        let snorlax = "Snorlax||leftovers||endure,bodyslam|Careful|252,,,,252,|||||";
        let tauros = "Tauros||leftovers||bodyslam,earthquake|Adamant|,252,,,,252|||||";
        let mut state = BattleState::start(&opts_cg(snorlax, tauros, "1,2,3,4"), &d).expect("start");
        let seed_before = state.prng_seed();
        let res = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
        assert!(!res.landed, "a status move never lands (no in-tryMoveHit Update)");
        assert!(state.sides[0].pokemon[0].endure, "the endure volatile is up");
        assert!(!state.sides[0].pokemon[0].protected, "endure does NOT set the protect move-block");
        assert_eq!(state.sides[0].pokemon[0].protect_counter, 2, "the SHARED stall counter escalates to 2");
        assert_eq!(state.prng_seed(), seed_before, "a first Endure draws NOTHING (no stall roll at counter 0)");
    }

    // FAIL-LOUD: a DEFERRED fixed-damage move (Counter — a reactive damageCallback) is in
    // `is_fixed_damage_move` (so it can never silently no-op via the bp==0 fall-through) but
    // has NO `fixed_damage_amount` entry, so it PANICS in `run_fixed_damage_move` rather than
    // desync. Mirrors `endure_panics_fail_loud`; pins the deferred-set guarantee so a future
    // change that drops a deferred id from `is_fixed_damage_move` (or adds a wrong amount)
    // can't silently regress it. (Psywave / the OHKO moves / Bide share this path;
    // Counter / Mirror Coat / Endeavor are MODELED since `gen3_move_coverage_batch5_v1`,
    // so the pin is keyed on Bide now.)
    #[test]
    #[should_panic(expected = "unmodeled FIXED-DAMAGE move")]
    fn deferred_fixed_damage_move_panics_fail_loud() {
        let d = dex();
        let snorlax = "Snorlax||leftovers||bide,bodyslam|Careful|252,,,,252,|||||";
        let tauros = "Tauros||leftovers||bodyslam,earthquake|Adamant|,252,,,,252|||||";
        let mut state = BattleState::start(&opts_cg(snorlax, tauros, "1,2,3,4"), &d).expect("start");
        // Bide is is_fixed_damage_move=true but fixed_damage_amount=None → fail-loud.
        let _ = state.run_move(MoveAction { side: 0, slot: 0, move_index: 0, struggle: false }, true, true, &d);
    }

    // FAIL-LOUD: SNORE (`gen3_move_coverage_batch5_v1` scope edge) — a bp-40 damaging
    // move with `sleepUsable` + an asleep-only `onTry`, NEITHER modeled (Sleep Talk is
    // the only modeled sleepUsable move). Without the guard it runs as a plain damaging
    // move: a silent mismodel both awake (sim: silent onTry fail) and asleep (sim:
    // cant-then-PROCEEDS; port: cant + blocked). The e2e picker blocklists it, but the
    // engine itself must fail loud for any future team source (the review's Lens-1
    // finding: the "Snore is fail-loud" claim was picker-only before this guard).
    #[test]
    #[should_panic(expected = "unmodeled sleepUsable move 'snore'")]
    fn snore_panics_fail_loud() {
        let d = dex();
        let snorlax = "Snorlax||leftovers||snore,bodyslam|Careful|252,,,,252,|||||";
        let tauros = "Tauros||leftovers||bodyslam,earthquake|Adamant|,252,,,,252|||||";
        let mut state = BattleState::start(&opts_cg(snorlax, tauros, "1,2,3,4"), &d).expect("start");
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
            state.execute_switch(1, 1, false, true, &d, &mut queue); // swap Snorlax to active + enqueue runSwitch
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
            state.execute_switch(1, slot, false, true, &d, &mut queue);
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
        state.execute_switch(1, 1, false, true, &d, &mut queue);
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
        state.execute_switch(1, 1, false, true, &d, &mut queue); // Snorlax out, Blissey in
        // The outgoing Snorlax (now bench slot 1) is un-restricted.
        let out = &state.sides[1].pokemon[1];
        assert!(out.taunt.is_none() && out.disable.is_none() && out.last_move.is_none(),
            "taunt/disable/last_move all clear on switch-out (clearVolatile)");
    }
}


