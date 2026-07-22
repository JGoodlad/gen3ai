//! Event — the generic event-dispatch core (`run_event` / `single_event`) and
//! the `>start` switch-in sequence wired on top of it.
//!
//! # What this layer is
//!
//! Showdown funnels every reactive game effect through two primitives in
//! `sim/battle.ts`:
//!
//! - **`singleEvent`** (`battle.ts:571`) — fire ONE effect's `on<Event>` callback
//!   on one target. No handler gathering, no sort, no speed, no PRNG, no
//!   `this.modify`. Gen ≤ 4 uses it to fire an ability's / item's `Start` at
//!   switch-in (the gen ≥ 5 `onSwitchIn`→`onStart` fallback in `getCallback` is
//!   gated `this.gen >= 5`).
//! - **`runEvent`** (`battle.ts:758`) — the core. Gathers handlers from every
//!   effect on/around the target, sorts them (`order → priority → speed →
//!   subOrder → effectOrder`, descending), and runs them in order under the
//!   `relayVar` return protocol. The sort is a *selection* sort whose
//!   **per-tie-group Fisher-Yates shuffle is the ONLY PRNG draw inside a
//!   `runEvent`** — and the determinism crux of the whole engine (consume the RNG
//!   in the exact same order/count as Showdown or every downstream draw desyncs).
//!
//! This module ports both primitives in a reusable shape (moves / residuals will
//! reuse them later) but only wires the handlers THIS step needs: the lead
//! switch-in abilities (Intimidate, Sand Stream / Drizzle / Drought).
//!
//! # The `>start` switch-in sequence (gen 3 path), as wired here
//!
//! Verified against the real sim with an instrumented PRNG (seed `[1,2,3,4]`):
//!
//! 1. `runAction('start')` (`battle.ts:2691`) iterates sides in **side-index
//!    order** (p1 then p2) calling `switchIn(lead, 0)`, which (gen 3, non-drag)
//!    **queues a `runSwitch` action per lead**.
//! 2. The two `runSwitch` actions run, ordered by **raw Speed (faster first)** —
//!    a queue ordering, NOT a re-sort. On a speed TIE the 2nd `runSwitch`'s
//!    `insertChoice` splices at a random index (`battle-queue.ts:283`, one PRNG
//!    draw); on distinct speeds it is deterministic (zero draws).
//! 3. gen 3 inherits **gen 4's `runSwitch`** (`data/mods/gen4/scripts.ts:18`),
//!    run ONCE PER MON: `runEvent('EntryHazard')` (no hazards turn 0),
//!    `runEvent('SwitchIn')` (no gen-3 ability has an `onSwitchIn` that touches the
//!    boosts/weather validated here — Truant's only sets `truantTurn`; Trace acts
//!    via `onStart`), then `singleEvent('Start', ability)` then `singleEvent('Start',
//!    item)`. **The ability `Start` is where Intimidate / Sand Stream actually
//!    act.**
//!
//! The **observable end state** this step validates — the foe's Atk boost (−1,
//! clamped to ≥ −6) from Intimidate and `field.weather` from Sand Stream / Drizzle
//! / Drought — is **order-independent** for the lead pairs we test, so it is
//! deterministic regardless of order. The **wired** switch-in path orders the two
//! leads by raw Speed and draws NOTHING (correct for the distinct-speed pairs we
//! test; on a tie it falls back to a deterministic side-order placement — a
//! documented simplification, see [`BattleState::run_start_switchins`]). The
//! reusable [`speed_sort`] primitive (with its faithful per-tie-group shuffle
//! draw) is built + unit-tested here for the later `runEvent` paths, but is NOT on
//! the switch-in path itself.
//!
//! ## Out-of-scope PRNG draws (documented, NOT reproduced here)
//!
//! Two real draws bracket the switch-in window but belong to other phases:
//! the per-mon **gender `sample(['M','F'])`** in the `Pokemon` constructor
//! (a *construction*/`addPokemon` draw — `pokemon.ts:340`), and the gen-3
//! **Quick Claw `randomChance(1,5)`** in `endTurn` (`battle.ts:1485`, a
//! *turn-loop* draw). Neither affects the boosts/weather this step asserts, and
//! both live in phases this bounded step does not build, so a full PRNG-state
//! parity test is deferred. See `BattleState::run_start_switchins`.

use crate::dex::to_id;
use crate::state::{BattleState, Weather};

// ===========================================================================
// A. The sort key (comparePriority) — order → priority → speed → subOrder →
//    effectOrder, all DESCENDING (Showdown's `b - a`). `battle.ts:404-411`.
// ===========================================================================

/// Default `order` when an effect declares none. Showdown uses `order || 2**32`
/// so "no explicit order sinks to the end" under the descending sort. Public so
/// callers building an [`EventHandler`] with no explicit order use the same
/// sentinel the comparator expects.
pub const NO_ORDER: u64 = 4_294_967_296; // 2**32

/// One gathered event handler, with its resolved sort key. Mirrors the
/// `EventListener` Showdown builds in `findEventHandlers` + `resolvePriority`.
///
/// `speed` is an `f64` because `resolvePriority` subtracts a sub-1 fractional
/// field-position offset from each `SwitchIn` handler's speed
/// (`battle.ts:1008-1013`) so same-speed mons' handlers keep their pre-sorted
/// rank instead of falsely tying.
#[derive(Debug, Clone, Copy)]
pub struct EventHandler<H> {
    /// `order` (default [`NO_ORDER`]). Higher sorts first.
    pub order: u64,
    /// `priority` (default 0). Higher first.
    pub priority: i32,
    /// `speed` (the holder's cached Speed, minus the SwitchIn field offset).
    /// Higher first.
    pub speed: f64,
    /// `subOrder` (default-by-effect-type: Ability 7 < Item 8). Higher first.
    pub sub_order: i32,
    /// `effectOrder` (the effect-state creation counter; only set for
    /// `*SwitchIn`/`*RedirectTarget`). Higher first.
    pub effect_order: i32,
    /// The concrete handler payload (what to actually run).
    pub handler: H,
}

impl<H> EventHandler<H> {
    /// `comparePriority(a, b)` (`battle.ts:404-411`), where `self == a`,
    /// `other == b`. Returns the FIRST nonzero of the descending key chain (or
    /// `0.0` for an EXACT tie — the only thing that triggers the speed-tie
    /// shuffle). The sort consumes it exactly like Showdown's `speedSort`
    /// (`delta < 0` keeps the incumbent, `delta > 0` replaces it, `delta == 0`
    /// ties).
    ///
    /// Each term mirrors the source's negation so a SMALLER `order`/`subOrder`/
    /// `effectOrder` and a LARGER `priority`/`speed` sort first:
    /// - `-((b.order) - (a.order))` = `a.order - b.order`
    /// - `(b.priority) - (a.priority)`
    /// - `(b.speed) - (a.speed)`
    /// - `-((b.subOrder) - (a.subOrder))` = `a.subOrder - b.subOrder`
    /// - `-((b.effectOrder) - (a.effectOrder))` = `a.effectOrder - b.effectOrder`
    fn compare_priority(&self, other: &Self) -> f64 {
        let a = self;
        let b = other;
        if a.order != b.order {
            return a.order as f64 - b.order as f64;
        }
        if a.priority != b.priority {
            return (b.priority - a.priority) as f64;
        }
        if a.speed != b.speed {
            return b.speed - a.speed;
        }
        if a.sub_order != b.sub_order {
            return (a.sub_order - b.sub_order) as f64;
        }
        if a.effect_order != b.effect_order {
            return (a.effect_order - b.effect_order) as f64;
        }
        0.0
    }
}

/// Default `subOrder` by effect type at switch-in (`resolvePriority`
/// `battle.ts:955-993`): Ability 7 sorts BEFORE Item 8 on a priority+speed tie.
/// Kept as named constants so future effect kinds slot in.
pub const SUBORDER_ABILITY: i32 = 7;
pub const SUBORDER_ITEM: i32 = 8;

/// `speedSort(handlers)` (`battle.ts:429-460`) — the selection sort that orders
/// gathered handlers AND draws the speed-tie Fisher-Yates shuffle from the PRNG.
///
/// This is the **RNG-consumption crux**: for each output position it scans the
/// remaining tail collecting every index that ties the current best
/// (`compare == 0`), swaps them into place in order, and — only when a tie group
/// has size > 1 — shuffles that group via [`crate::prng::Prng::shuffle_range`],
/// which draws exactly `k - 1` times for a group of size `k` (size-1 groups draw
/// zero). Mirroring the tie-grouping exactly keeps the shuffle's `(start, end)`
/// range — hence the draw count — identical to Showdown.
pub fn speed_sort<H>(handlers: &mut [EventHandler<H>], prng: &mut crate::prng::Prng) {
    let n = handlers.len();
    if n < 2 {
        return; // `battle.ts:430`
    }
    let mut sorted = 0usize;
    while sorted + 1 < n {
        // Grab the list of "next" (sorts-first) indexes, collecting exact ties.
        // Showdown: `delta = comparator(list[nextIndexes[0]], list[i])`; here
        // `a = nextIndexes[0]`, `b = i`. `delta < 0` ⇒ keep incumbent (continue);
        // `delta > 0` ⇒ `i` is strictly better, reset the group; `delta == 0` ⇒
        // tie, push.
        let mut next_indexes = vec![sorted];
        for i in (sorted + 1)..n {
            let delta = handlers[next_indexes[0]].compare_priority(&handlers[i]);
            if delta < 0.0 {
                continue;
            }
            if delta > 0.0 {
                next_indexes = vec![i];
            } else {
                next_indexes.push(i);
            }
        }
        // Put the collected indexes where they belong. `nextIndexes` is in
        // ascending order and (per the source's invariant) never disturbed by an
        // earlier swap, so a plain swap into `sorted + i` is exact — no rewrite.
        for (i, &index) in next_indexes.iter().enumerate() {
            if index != sorted + i {
                handlers.swap(sorted + i, index);
            }
        }
        let group_len = next_indexes.len();
        if group_len > 1 {
            // THE speed-tie shuffle — the only PRNG draw in the sort.
            prng.shuffle_range(handlers, sorted, sorted + group_len);
        }
        sorted += group_len;
    }
}

// ===========================================================================
// B. singleEvent — fire one effect's callback on one target. No gather/sort/RNG.
// ===========================================================================

/// `singleEvent(eventid, effect, …, target)` (`battle.ts:571`), specialized to
/// the ability `Start` event this step needs.
///
/// Showdown's `singleEvent` is fully generic over the callback; here the only
/// `Start` callbacks we model are the switch-in abilities, dispatched by id via
/// [`ability_on_start`]. It runs NO sort, NO speed, NO PRNG, and NO
/// `this.modify` (unlike `run_event`), exactly mirroring the source.
///
/// `holder` is the side index (0/1) + team slot of the mon whose ability fires;
/// the callback mutates `state` in place (e.g. boosts the foe, sets weather).
/// `draw_trace`: whether a TRACE holder's `randomFoe()` sample is DRAWN here. A
/// MID-BATTLE switch-in (`turn.rs::run_switch`) passes `true` — the sample is a live,
/// probe-verified draw (`probe_trace_shedskin_rng.js` T1: `random(1)` even for a single
/// foe). The BATTLE-START path (`run_start_switchins`) passes `false`: the `>start`
/// window's draws (gender samples, the lead trace sample, the turn-0 Quick Claw) all
/// happen BEFORE the pre-first-decision seed every golden/e2e run starts from, so the
/// port applies the lead-trace COPY (deterministic in singles — the one foe) WITHOUT
/// consuming the already-elapsed draw (the same PRNG-honesty contract as the gender
/// samples — see `run_start_switchins`' doc).
pub fn single_event_ability_start(state: &mut BattleState, holder_side: usize, holder_slot: usize, draw_trace: bool) {
    // `MonState::ability` starts as the dex DISPLAY name (`"Sand Stream"`) — normalize
    // to the id (`"sandstream"`) the handler dispatch keys on, exactly as Showdown keys
    // effects by `to_id`. (A traced value is already an id/display copied verbatim.)
    let ability_id = to_id(&state.sides[holder_side].pokemon[holder_slot].ability);
    ability_on_start(state, holder_side, holder_slot, &ability_id, draw_trace);
}

/// The abilities a gen-3 TRACE copy can land on WITHOUT desyncing the port — the e2e
/// harness's `MODELED_ABILITIES ∪ NOOP_ABILITIES` (gen_e2e_fuzz.js, the ONE source of
/// truth for what the engine prices; keep in lockstep). Trace copies the OPPOSING
/// active's CURRENT ability, and the e2e filter admits only teams whose EVERY mon
/// (both sides) carries a modeled/no-op ability — so a filtered battle can never trip
/// this. The engine still FAIL-LOUDS (panics) on an unmodeled copy rather than
/// silently no-op'ing an ability the sim would run (the batch-3 Trace safety mandate).
const TRACE_COPYABLE: &[&str] = &[
    // MODELED_ABILITIES:
    "intimidate", "sandstream", "drizzle", "drought", "levitate", "flashfire",
    "waterabsorb", "voltabsorb", "thickfat", "clearbody", "whitesmoke",
    "hypercutter", "keeneye", "serenegrace", "shielddust", "owntempo",
    "arenatrap", "magnetpull",
    "torrent", "blaze", "overgrow", "swarm", "hugepower", "purepower", "guts", "marvelscale",
    "compoundeyes", "sandveil", "hustle",
    "naturalcure",
    "limber", "insomnia", "vitalspirit", "immunity", "waterveil", "magmaarmor",
    "shellarmor", "battlearmor", "chlorophyll", "swiftswim",
    "cloudnine", "airlock", "speedboost", "raindish",
    "static", "poisonpoint", "flamebody", "effectspore", "roughskin",
    "soundproof", "damp", "suctioncups", "synchronize",
    "trace", "shedskin",
    "truant", "innerfocus", "shadowtag", "cutecharm", "colorchange",
    // NOOP_ABILITIES:
    "pressure", "oblivious", "runaway", "illuminate", "honeygather", "pickup",
    "stench", "sturdy", "rockhead", "earlybird", "noability",
    "plus", "minus", "lightningrod", "stickyhold",
    "", // an empty ability field (a hacked/constructed set)
];

/// TRACE `onStart` (`gen3_berry_trace_shedskin_v1` — the gen3-RESOLVED handler; the
/// mod-chain law: the base/gen4 seek/notrace/onUpdate machinery is REPLACED):
///
/// ```js
/// onStart(pokemon) {
///   const target = pokemon.side.randomFoe();
///   if (!target || target.fainted) return;
///   pokemon.setAbility(target.getAbility(), target);
/// }
/// ```
///
/// PROBE-settled (`probe_trace_shedskin_rng.js`):
///   - T1: `randomFoe()` = `battle.sample(foes())` — it DRAWS `random(1)` even with a
///     SINGLE foe (the phaze-n=1 gotcha); a fainted/absent foe → no draw, no copy.
///   - T2/T2b: gen3 `setAbility` does NOT fire the copied onStart (`battle.gen > 3`
///     gate) — a Traced Intimidate/Drought does nothing on copy.
///   - T3: NO guard — No Ability / Wonder Guard / a traced-through ability all copy
///     (the foe's CURRENT ability, not its base).
///   - T6: the copied ability's PASSIVE effects are LIVE (every engine read goes through
///     `MonState::ability`).
///   - T4: switch-out reverts to Trace (`execute_switch`); re-entry re-traces (a draw).
fn trace_on_start(state: &mut BattleState, side: usize, slot: usize, draw: bool) {
    let foe = 1 - side;
    let foe_active = state.sides[foe].active;
    if state.sides[foe].pokemon[foe_active].fainted {
        return; // `foes()` excludes a fainted active → no target: NO draw, NO copy
    }
    if draw {
        // `battle.sample([the one foe])` → `random(1)` — consumed even though the
        // result is always index 0 (singles has exactly one live foe).
        let _ = state.prng.random_below(1);
    }
    let copied = state.sides[foe].pokemon[foe_active].ability.clone();
    // FAIL-LOUD on an unmodeled copy: an ability outside the modeled/no-op universe
    // would silently no-op in this engine while the sim runs its handlers — a
    // divergence. The e2e filter keeps both TEAMS all-modeled, so a filtered battle
    // never reaches this; a constructed/hacked battle fails loud instead of quiet.
    let copied_id = to_id(&copied);
    assert!(
        TRACE_COPYABLE.contains(&copied_id.as_str()),
        "gen3_berry_trace_shedskin_v1: Trace copied the UNMODELED ability {copied_id:?} — \
         the port cannot represent its behavior bit-for-bit. Keep the opposing team's \
         abilities inside MODELED_ABILITIES ∪ NOOP_ABILITIES (gen_e2e_fuzz.js), or model \
         the ability before tracing it."
    );
    state.sides[side].pokemon[slot].ability = copied;
    // (Protocol: the sim emits `|-ability|<holder>|<Copied>|Trace|[from] ability: Trace|
    // [of] <foe>` for this copy. That line IS now emitted — this fn only mutates the
    // ability STATE; the emission lives in `turn.rs::emit_ability_start_lines`
    // (`ability_traced`), byte-verified vs the `trace_switchin` capture golden [lead +
    // mid-battle re-trace], `gen3_protocol_phase3_v1`.)
}

// ===========================================================================
// C. The switch-in ability handlers (the gen3ou `onStart`s wired this step).
// ===========================================================================

/// Dispatch an ability's `onStart` (the gen ≤ 4 switch-in `Start` event) by id.
/// Data-light + extensible: a new switch-in ability is one match arm. Unknown /
/// no-`onStart` abilities are a silent no-op (matching the `getCallback`
/// undefined-callback pass-through).
fn ability_on_start(state: &mut BattleState, side: usize, slot: usize, ability_id: &str, draw_trace: bool) {
    match ability_id {
        "intimidate" => intimidate_on_start(state, side, slot),
        // Drizzle/Drought's `if (source.item === 'blueorb'/'redorb') return` guard is
        // gen-6+ Primal Reversion — irrelevant to gen 3, so it is intentionally absent.
        "sandstream" => set_weather_on_start(state, Weather::Sand),
        "drizzle" => set_weather_on_start(state, Weather::Rain),
        "drought" => set_weather_on_start(state, Weather::Sun),
        // TRACE (`gen3_berry_trace_shedskin_v1`): copy the foe active's CURRENT ability
        // (the `randomFoe()` sample draw is gated by `draw_trace` — see the fn docs).
        "trace" => trace_on_start(state, side, slot, draw_trace),
        _ => {} // no onStart in gen3ou scope (Pressure/Levitate/etc. are passive)
    }
}

/// Intimidate `onStart` (gen 3 override, `data/mods/gen3/abilities.ts:61-86`):
/// drop each adjacent foe's Atk stage by 1 (clamped to ≥ −6). In singles the one
/// adjacent foe is the opposing active.
///
/// We mirror `boost({atk:-1}, target, pokemon, null, true)` (`battle.ts:2017`):
/// the `TryBoost` aux event runs the foe's `onTryBoost` immunity handlers FIRST —
/// **Clear Body / White Smoke** block ALL stat drops, **Hyper Cutter** blocks Atk
/// (so an Intimidate INTO any of these is a no-op, real-team-common e.g. Salamence
/// vs Metagross). And the gen-3 quirk — the gen3 mod's Intimidate skips a foe
/// **behind a SUBSTITUTE** ("no effect if every foe has a Substitute"): inert at
/// the battle-start switch-in (no subs exist turn 0) but LIVE on every MID-BATTLE
/// Intimidate switch-in — an Intimidate entrant does NOT drop a subbed foe's Atk
/// (probe-verified vs the sim: sub up → no `-unboost`, boosts unchanged; the
/// `gen3_trapping_v1` e2e regen surfaced the gap on real teams, e2e_171/e2e_204 —
/// a Jynx that Substituted the turn before a Salamence switch-in kept Atk 0 in the
/// sim while the port dropped it to −1). All DRAW-FREE (no PRNG in
/// `onTryBoost`/`boost`/the sub check — probed identical seeds with/without the
/// sub), so this is a STATE-only gate. When not blocked, the net effect is the
/// clamped Atk-stage decrement.
fn intimidate_on_start(state: &mut BattleState, side: usize, slot: usize) {
    let foe = 1 - side; // singles: the one opposing side
    let foe_active = state.sides[foe].active;
    // FORCED-REPLACEMENT mis-target guard (`gen3_intimidate_forced_replacement_v1`, M3): the
    // deferred `RunSwitch` fires this switch-in Intimidate LATER than the switch action. If the
    // intended foe (the one present when this mon switched in — its captured `switchin_foe_uid`)
    // was FORCE-REPLACED because it FAINTED between the switch and this deferred fire (a
    // Destiny-Bond / on-entry-faint forced replacement), Showdown's entrant Intimidate resolved
    // INLINE against the (fainted) original — an EMPTY `adjacentFoes()` → NO drop (the ab_1381_0
    // "not activated" hint). So SUPPRESS the drop only in that case (the original faints
    // regardless → state-equivalent). NARROW: suppress ONLY when the captured original is now
    // FAINTED/absent — NOT for a live foe-active change (a DOUBLE VOLUNTARY switch leaves the
    // original alive on the bench and the sim DOES drop the new foe, e2e-verified). The lead
    // switch-in path leaves `switchin_foe_uid == None` (draw-free `run_start_switchins`), so a
    // lead Intimidate is never suppressed. DRAW-FREE (a boost-suppression is state-only).
    if let Some(target_uid) = state.sides[side].pokemon[slot].switchin_foe_uid {
        if state.sides[foe].pokemon[foe_active].uid != target_uid {
            // The intended target was replaced. Suppress ONLY if it FAINTED (a forced
            // replacement — the M3 case); if it merely switched to the bench alive, drop the
            // new foe (matching the sim + the pre-fix behavior).
            let original_fainted = state.sides[foe]
                .pokemon
                .iter()
                .find(|m| m.uid == target_uid)
                .map(|m| m.fainted || m.hp == 0)
                .unwrap_or(true); // absent entirely → treat as gone
            if original_fainted {
                return;
            }
        }
    }
    // Belt-and-braces: a fainted/0-HP CURRENT foe active is a genuine no-op (`adjacentFoes()`
    // excludes it).
    if state.sides[foe].pokemon[foe_active].fainted
        || state.sides[foe].pokemon[foe_active].hp == 0
    {
        return;
    }
    // gen-3 Intimidate vs SUBSTITUTE: a subbed foe is NOT dropped (the gen3 mod's
    // per-foe substitute skip; probe `harness/probe_intimidate_substitute_rng.js` + the sim's
    // `-unboost` absence). DRAW-FREE.
    if state.sides[foe].pokemon[foe_active].substitute.is_some() {
        return;
    }
    let foe_ability = to_id(&state.sides[foe].pokemon[foe_active].ability);
    // gen-3 `onTryBoost` Atk-drop immunity (Clear Body / White Smoke = all;
    // Hyper Cutter = atk). Mirrors `stat_drop_blocked(_, atk)` on the move path.
    if matches!(foe_ability.as_str(), "clearbody" | "whitesmoke" | "hypercutter") {
        return;
    }
    let boosts = &mut state.sides[foe].pokemon[foe_active].boosts;
    // boosts index 0 == atk (`[atk, def, spa, spd, spe, accuracy, evasion]`).
    boosts[0] = (boosts[0] - 1).max(-6);
}

/// Sand Stream / Drizzle / Drought `onStart` (`data/abilities.ts:3952/1068/1078`)
/// → `this.field.setWeather(...)`. The gen-3 quirk: ability-set weather is
/// **PERMANENT** — `setWeather`'s `durationCallback` yields 5, but the
/// weather condition's `onFieldStart` sets `effectState.duration = 0` for an
/// Ability source when `gen <= 5` (`data/conditions.ts:648/499/573`), verified
/// in the sim as `weatherState.duration === 0`. So `weather_turns = 0` is the
/// permanent sentinel (NOT 5).
///
/// The "already this weather" no-op guard (`field.ts:50-52`) matters only for a
/// mirror double-setter; at switch-in only one lead sets weather in our tests.
/// We still honor it: a re-set of the SAME ability-permanent weather is a no-op.
fn set_weather_on_start(state: &mut BattleState, weather: Weather) {
    if state.field.weather == Some(weather) && state.field.weather_turns == 0 {
        return; // ability-permanent same-weather re-set is a no-op (field.ts:50-52)
    }
    state.field.weather = Some(weather);
    state.field.weather_turns = 0; // ability source ⇒ permanent (gen3 quirk)
}

impl BattleState {
    /// Run the `>start` switch-in sequence on an already-constructed state:
    /// switch in both leads and fire their switch-in ability events.
    ///
    /// This completes the deferred half of `>start` that
    /// [`BattleState::start`] leaves out (it builds the pre-event board only).
    /// After this call, `field.weather` carries any Sand Stream / Drizzle /
    /// Drought weather and the opposing actives carry any Intimidate Atk drop.
    ///
    /// # Order (gen 3, verified)
    ///
    /// The two leads' ability `onStart`s fire in **raw-Speed order (faster
    /// first)** — the order the two `runSwitch` actions dequeue (set by
    /// `BattleQueue.insertChoice`'s raw-Speed comparator), NOT the `SwitchIn`
    /// `runEvent` handler sort (whose per-mon fractional field-position offset
    /// makes equal-speed leads' handlers never tie, so it adds no draw). We order
    /// the two leads by raw Speed, faster first, then fire each ability `Start`
    /// via [`single_event_ability_start`]. When both leads set weather the slower
    /// fires LAST so its weather wins (verified: Kyogre Drizzle faster vs
    /// Tyranitar Sand Stream slower ⇒ sandstorm).
    ///
    /// # Scope / PRNG honesty (read before extending)
    ///
    /// This dispatch is **draw-free**. The real sim's `>start` window is NOT —
    /// even at distinct speeds it draws the per-mon **gender `sample`** (a
    /// *construction*/`addPokemon` draw) and the **Quick Claw `randomChance(1,5)`**
    /// (a *turn-loop* `endTurn` draw) — but both live in phases this bounded step
    /// does not build, so we DELIBERATELY OMIT them (the Rust `prng` stays put while
    /// the sim's advances; this is an omission, NOT prng-state parity). On a raw-Speed TIE the real sim
    /// resolves the two leads' relative order with a `BattleQueue.insertChoice`
    /// random splice (one draw) plus the per-action `eachEvent('Update')`
    /// active-mon shuffles — all **queue / turn-loop machinery deferred to a later
    /// step**. So here a raw-Speed tie falls back to a deterministic side-order
    /// placement and draws nothing; the boosts/weather end state is unaffected
    /// (a same-speed mirror's effects are order-independent). The reusable
    /// [`speed_sort`] core (with its faithful per-tie-group shuffle draw) is what
    /// the later `runEvent` paths use; it is exercised directly by its own test.
    /// Net: `state.prng` after this call is NOT yet a full `>start` RNG-state
    /// oracle — the validated output is the boosts/weather.
    pub fn run_start_switchins(&mut self) {
        // The two active leads, with their raw Speed (stats[5]).
        let mut leads: Vec<(usize, usize, u16)> = (0..self.sides.len())
            .map(|side| {
                let slot = self.sides[side].active;
                (side, slot, self.sides[side].pokemon[slot].stats[5])
            })
            .collect();

        // Order by raw Speed, faster first; a tie keeps side order (the
        // queue-tie-break draw is deferred — see the doc above). `sort_by` is a
        // stable sort, so equal-speed leads retain ascending side order.
        leads.sort_by(|a, b| b.2.cmp(&a.2));

        // Fire each lead's ability Start in resolved order (slower last ⇒ its
        // weather wins). `draw_trace=false`: a LEAD Trace's `randomFoe()` sample is a
        // `>start`-window draw — it elapsed BEFORE the pre-first-decision seed every
        // golden/e2e run starts from — so the copy applies WITHOUT consuming a draw
        // here (the same contract as the skipped gender samples; see the doc above).
        for (side, slot, _spe) in leads {
            single_event_ability_start(self, side, slot, false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prng::Prng;

    /// Build a handler whose only distinguishing key is `speed` (everything else
    /// at the switch-in defaults), carrying a `tag` payload so we can read the
    /// resolved order.
    fn h(speed: f64, tag: usize) -> EventHandler<usize> {
        EventHandler {
            order: NO_ORDER,
            priority: 0,
            speed,
            sub_order: SUBORDER_ABILITY,
            effect_order: 0,
            handler: tag,
        }
    }

    #[test]
    fn speed_sort_orders_distinct_speeds_fast_first_no_draw() {
        // Distinct speeds ⇒ no tie group ⇒ ZERO draws; faster sorts first.
        let mut prng = Prng::new("1,2,3,4");
        let mut hs = vec![h(100.0, 0), h(300.0, 1), h(200.0, 2)];
        speed_sort(&mut hs, &mut prng);
        let order: Vec<usize> = hs.iter().map(|x| x.handler).collect();
        assert_eq!(order, vec![1, 2, 0], "300 > 200 > 100");
        assert_eq!(prng.get_seed(), "1,2,3,4", "no tie ⇒ no PRNG draw");
    }

    #[test]
    fn speed_sort_higher_priority_and_order_win() {
        let mut prng = Prng::new("1,2,3,4");
        // Same speed but different priority (higher first) — distinct keys, no tie.
        let mut hs = vec![
            EventHandler { order: NO_ORDER, priority: 0, speed: 100.0, sub_order: 0, effect_order: 0, handler: 0usize },
            EventHandler { order: NO_ORDER, priority: 3, speed: 100.0, sub_order: 0, effect_order: 0, handler: 1usize },
        ];
        speed_sort(&mut hs, &mut prng);
        assert_eq!(hs[0].handler, 1, "priority 3 before priority 0");
        // Smaller `order` sorts first.
        let mut hs2 = vec![
            EventHandler { order: 101, priority: 0, speed: 0.0, sub_order: 0, effect_order: 0, handler: 0usize },
            EventHandler { order: 1, priority: 0, speed: 0.0, sub_order: 0, effect_order: 0, handler: 1usize },
        ];
        speed_sort(&mut hs2, &mut prng);
        assert_eq!(hs2[0].handler, 1, "order 1 before order 101");
        // Ability (subOrder 7) before Item (subOrder 8) on a priority+speed tie.
        let mut hs3 = vec![
            EventHandler { order: NO_ORDER, priority: 0, speed: 50.0, sub_order: SUBORDER_ITEM, effect_order: 0, handler: 0usize },
            EventHandler { order: NO_ORDER, priority: 0, speed: 50.0, sub_order: SUBORDER_ABILITY, effect_order: 0, handler: 1usize },
        ];
        speed_sort(&mut hs3, &mut prng);
        assert_eq!(hs3[0].handler, 1, "Ability (7) before Item (8)");
    }

    #[test]
    fn speed_sort_tie_group_draws_fisher_yates_exactly() {
        // A size-2 exact tie draws ONE random_range(0,2); a size-3 tie draws TWO
        // (random_range(0,3), random_range(1,3)) — the Fisher-Yates count. We
        // verify both the draw count (via the post seed vs an oracle Prng) AND
        // that the resulting permutation matches an oracle shuffle bit-for-bit.

        // size-2 tie:
        let mut prng = Prng::new("1,2,3,4");
        let mut hs = vec![h(100.0, 10), h(100.0, 20)];
        speed_sort(&mut hs, &mut prng);
        let mut oracle = Prng::new("1,2,3,4");
        let mut idx = [0usize, 1usize]; // the tags' source positions
        oracle.shuffle_range(&mut idx, 0, 2);
        let tags = [10usize, 20usize];
        let exp: Vec<usize> = idx.iter().map(|&i| tags[i]).collect();
        assert_eq!(hs.iter().map(|x| x.handler).collect::<Vec<_>>(), exp, "size-2 tie permutation");
        assert_eq!(prng.get_seed(), oracle.get_seed(), "size-2 tie draws exactly 1");

        // size-3 tie:
        let mut prng = Prng::new("1,2,3,4");
        let mut hs = vec![h(50.0, 1), h(50.0, 2), h(50.0, 3)];
        speed_sort(&mut hs, &mut prng);
        let mut oracle = Prng::new("1,2,3,4");
        let mut idx = [0usize, 1usize, 2usize];
        oracle.shuffle_range(&mut idx, 0, 3);
        let tags = [1usize, 2usize, 3usize];
        let exp: Vec<usize> = idx.iter().map(|&i| tags[i]).collect();
        assert_eq!(hs.iter().map(|x| x.handler).collect::<Vec<_>>(), exp, "size-3 tie permutation");
        assert_eq!(prng.get_seed(), oracle.get_seed(), "size-3 tie draws exactly 2");
    }

    #[test]
    fn speed_sort_partial_tie_only_shuffles_the_tied_group() {
        // [fast, tieA, tieB] — fast placed deterministically (no draw), then the
        // two tied entries shuffle (one draw). The fast handler must stay first.
        let mut prng = Prng::new("1,2,3,4");
        let mut hs = vec![h(300.0, 9), h(100.0, 1), h(100.0, 2)];
        speed_sort(&mut hs, &mut prng);
        assert_eq!(hs[0].handler, 9, "the strictly-faster handler is first, undisturbed");
        // Exactly one draw (the size-2 tie among slots 1..3).
        let mut oracle = Prng::new("1,2,3,4");
        let mut idx = [0usize, 1usize];
        oracle.shuffle_range(&mut idx, 0, 2);
        assert_eq!(prng.get_seed(), oracle.get_seed(), "only the tied group shuffles");
    }
}
