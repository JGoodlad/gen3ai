use crate::dex::{to_id, Dex, Type};
use crate::event::{speed_sort, EventHandler, NO_ORDER};
use crate::state::{Status, Weather};
use super::*;
use super::helpers::*;

impl crate::state::BattleState {

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
    pub(crate) fn each_event_shuffle(&mut self) -> Vec<usize> {
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
    pub(crate) fn set_status_event_shuffle(&mut self) {
        // Two tied handlers (the 2 Standard clauses) at equal everything → speed_sort
        // draws one size-2 shuffle. Shared with the ModifyDamagePhase1 screen pair.
        self.two_tied_handler_shuffle();
    }

    /// A size-2 speed-sort shuffle over two handlers at EQUAL order/priority/speed (0) — an
    /// UNCONDITIONAL one-draw `random(0,2)` (unlike [`each_event_shuffle`], which draws only
    /// on a mon-SPEED tie). Used for any `runEvent` whose 2 gathered handlers are both
    /// speed-less effect handlers that ALWAYS tie: the gen3ou SetStatus clause pair (Sleep +
    /// Freeze Clause Mod, [`set_status_event_shuffle`]) AND the `ModifyDamagePhase1`
    /// Reflect + Light-Screen pair (`gen3_move_coverage_batch2_v1` — two `onAnyModifyDamage
    /// Phase1` side-condition handlers). The payload is irrelevant — we only need the DRAW.
    pub(crate) fn two_tied_handler_shuffle(&mut self) {
        self.n_tied_handler_shuffle(2);
    }

    /// The `runEvent('ModifyDamagePhase1')` HANDLER-SORT SHUFFLE, generalized over the FULL
    /// screen tie group (`gen3_move_coverage_batch7_v1` — the multi-strike / random-mode fix).
    /// gen3 Reflect + Light Screen register `onAnyModifyDamagePhase1` side-condition handlers —
    /// the **`onAny`** prefix means each fires for damage on EITHER side, so EVERY screen up
    /// across BOTH sides gathers into ONE tie group (equal order/priority/speed(0)/subOrder), and
    /// a size-k tie draws `k-1` in the Fisher-Yates speed-sort (0/1 handler → no draw). The prior
    /// `two_tied_handler_shuffle`-gated-on-`foe.reflect && foe.light_screen` MISSED the cross-side
    /// combos (e.g. BOTH sides carrying Light Screen = 2 handlers → 1 draw, or a foe's two screens
    /// + the attacker's own screen = 3 handlers → 2 draws) — a latent single-hit desync the
    /// admitted multi-strike moves amplified (each strike re-runs `getDamage` → this event) and
    /// the random-mode byte fuzz surfaced (both-Light-Screen boards). Flash Fire's
    /// `onModifyDamagePhase1` is a DIFFERENT (attacker-speed) tie group → NOT counted here.
    /// VERIFIED vs the sim's per-hit `speedSort <- runEvent('ModifyDamagePhase1')` draw
    /// (`harness/probe_batch7_multihit.js` + the dec-44 Bullet-Seed trace).
    pub(crate) fn modify_damage_phase1_shuffle(&mut self) {
        let n = (self.sides[0].reflect > 0) as usize
            + (self.sides[0].light_screen > 0) as usize
            + (self.sides[1].reflect > 0) as usize
            + (self.sides[1].light_screen > 0) as usize;
        if n >= 2 {
            self.n_tied_handler_shuffle(n);
        }
    }

    /// Run the Fisher-Yates speed-sort tie-shuffle over `n` handlers at EQUAL order/priority/
    /// speed(0)/subOrder (a group that ALWAYS ties) — draws `n-1` `random(range)` calls, exactly
    /// as the sim's `speedSort` shuffles a size-`n` tie group. The payload is irrelevant; only the
    /// DRAW matters. `n < 2` draws nothing.
    pub(crate) fn n_tied_handler_shuffle(&mut self, n: usize) {
        let mut handlers: Vec<EventHandler<usize>> = (0..n)
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
    /// slots, **encore** [`gen3_encore_disable_move_shuffle_v1`] disables every non-encored
    /// slot). If a mon carries ≥2 of these volatiles at once, its handlers all TIE (same
    /// Condition order=false / priority 0 / SAME mon's speed / subOrder) → `speedSort` shuffles
    /// the tie group, drawing `n-1` `random(range)` calls (a size-2 group = 1 draw). Fires per
    /// mon INDEPENDENTLY, in array order. It runs in `endTurn` AFTER the residual, BEFORE the
    /// Quick Claw roll (`if gen===3 quickClawRoll = randomChance(1,5)`), so the draw order is
    /// `… residual … → [DisableMove shuffle per mon] → Quick Claw`. A mon with 0 or 1 such
    /// volatile draws NOTHING. VERIFIED vs the sim (`harness/probe_taunt_disable_rng.js` +
    /// `probe_disable_full_lifecycle.js`): a taunt+disable mon draws one size-2 shuffle at
    /// endTurn; a taunt-only / disable-only mon draws none. This is the ONLY `DisableMove`-event
    /// draw (`singleEvent('DisableMove', …)` per moveslot is a no-sort/no-RNG `singleEvent`).
    pub(crate) fn disable_move_event_shuffle(&mut self, dex: &Dex) {
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
            // RB1 (`gen3_encore_disable_move_shuffle_v1`): the gen3 `encore` condition ALSO
            // registers an `onDisableMove` handler (it disables every non-encored slot), so an
            // encored mon that co-carries taunt / disable / choicelock reaches n>=2 and draws
            // the size->=2 endTurn DisableMove tie-shuffle. A LONE-encore mon (n stays 1) draws
            // nothing new. Omitting encore mis-counted the handler set → a MISSING draw one call
            // before the Quick Claw on an encore+choicelock/taunt/disable endTurn.
            let n = (mon.taunt.is_some() as usize)
                + (mon.disable.is_some() as usize)
                + (mon.choice_locked_move.is_some() as usize)
                + (mon.encore.is_some() as usize);
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
            // The `choicelock` handler's SELF-REMOVAL (`gen3_choicelock_lazy_release_v1`): the
            // resolved gen3 `choicelock.onDisableMove` opens with
            //     if (!pokemon.getItem().isChoice || !pokemon.hasMove(effectState.move))
            //         { pokemon.removeVolatile('choicelock'); return; }
            // — so a mon whose Choice item was Knocked Off / Thief'd / Trick'd away still
            // GATHERS the handler for THIS event (it counted in `n` above) and only then drops
            // the volatile. Doing it here (after the count) is what keeps the endTurn draw
            // count bit-for-bit; the SIM PROBE `harness/probe_rb_choicelock.js` pins both
            // halves (the Knock-Off turn's endTurn still draws the encore+choicelock shuffle;
            // the next turn's does not). Draw-free.
            let mon = &mut self.sides[side].pokemon[slot];
            if mon.choice_locked_move.is_some()
                && !dex
                    .item(&crate::dex::to_id(&mon.item))
                    .map(|i| i.choice)
                    .unwrap_or(false)
            {
                mon.choice_locked_move = None;
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
    /// With >= 2 handlers the sort ties iff the holders' cached `pokemon.speed` AND their
    /// `subOrder` are equal (`effectOrder` is only resolved for SwitchIn/RedirectTarget
    /// callbacks) → ONE Fisher-Yates draw per tied GROUP per event. `subOrder` is the
    /// `effectTypeOrder` table at `sim/battle.ts:957`: **Condition 2, Ability 7** — so a
    /// Condition handler NEVER ties an Ability handler. The MAGNETON MIRROR (both actives
    /// Magnet Pull, equal speeds) therefore draws **4 per endTurn** (2 events × 2 mons, 1
    /// each — probed 11 vs the Sturdy-control's 7 draws/turn, seed-verified); the DUGTRIO
    /// MIRROR (Arena Trap is `onFoe`-only → 1 handler per event) draws ZERO (probed
    /// byte-identical seeds to a Sand Veil control).
    ///
    /// **THE CONDITION HANDLERS** (`gen3_partial_trap_v1`, found by the ROUND-32 byte fuzz —
    /// repro `rmsde6xp4_ab_10_9`, an Onix that Blocks and then Sand Tombs the same
    /// Wartortle). TWO gen-3 volatiles carry `onTrapPokemon`: the trap-MOVE `trapped`
    /// (Mean Look / Spider Web / Block — `MonState::trapped_by`) and `partiallytrapped`
    /// (the wrap family — `MonState::partial_trap`). Alone, each adds ONE handler at
    /// subOrder 2 that can never tie an Ability's 7, which is why the pre-round-32 model
    /// (abilities only) was correct while only ONE of them could exist. Held TOGETHER on the
    /// same mon they are two handlers with the SAME speed AND the SAME subOrder ⇒ a TIE ⇒
    /// **ONE extra `random(0,2)` per endTurn, for as long as both are live**.
    /// PROBE-MEASURED (`harness/probe_ptrap_trapevent.js`, idle-turn draws vs a clean
    /// control): block-only **+0**, sandtomb-only **+0**, BOTH live **+1**,
    /// arena-trap-only **+0**. Exactly ONE, not two, because **only `TrapPokemon` gathers
    /// them** — neither condition carries an `onMaybeTrapPokemon`, so the MaybeTrapPokemon
    /// event still sees only the ability handlers. Hence the two events are built from
    /// DIFFERENT handler lists below.
    fn trap_event_shuffles(&mut self, side: usize) {
        let foe_side = 1 - side;
        let me = &self.sides[side].pokemon[self.sides[side].active];
        let foe = &self.sides[foe_side].pokemon[self.sides[foe_side].active];
        // The ABILITY handlers (subOrder 7) — present on BOTH trap events.
        let mut abilities: Vec<f64> = Vec::with_capacity(2);
        // alliesAndSelf: own onAny* — gen-3 Magnet Pull only.
        if to_id(&me.ability) == "magnetpull" {
            abilities.push(me.cached_speed as f64);
        }
        // foes(): onFoe* (Arena Trap) + onAny* (Magnet Pull) — `foes()` filters fainted.
        if !foe.fainted {
            let foe_ability = to_id(&foe.ability);
            if foe_ability == "arenatrap" || foe_ability == "magnetpull" {
                abilities.push(foe.cached_speed as f64);
            }
        }
        // The CONDITION handlers (subOrder 2) — the TARGET mon's OWN volatiles, and ONLY on
        // the `TrapPokemon` event (neither carries `onMaybeTrapPokemon`).
        let mut conditions: Vec<f64> = Vec::with_capacity(2);
        if me.trapped_by.is_some() {
            conditions.push(me.cached_speed as f64);
        }
        if me.partial_trap.is_some() {
            conditions.push(me.cached_speed as f64);
        }

        const ABILITY_SUBORDER: i32 = 7;
        const CONDITION_SUBORDER: i32 = 2;
        let sort = |st: &mut Self, list: &[(f64, i32)]| {
            if list.len() < 2 {
                return; // 0/1 handlers → no sort tie possible → draw-free (the common case)
            }
            let mut handlers: Vec<EventHandler<usize>> = list
                .iter()
                .enumerate()
                .map(|(i, &(speed, sub_order))| EventHandler {
                    order: NO_ORDER,
                    priority: 0,
                    speed,
                    sub_order,
                    effect_order: 0,
                    handler: i,
                })
                .collect();
            speed_sort(&mut handlers, &mut st.prng);
        };

        // (1) TrapPokemon — conditions (the target's own volatiles, gathered by
        //     `findPokemonEventHandlers` before the allies/foes sweep) THEN abilities.
        let mut trap_list: Vec<(f64, i32)> =
            conditions.iter().map(|&s| (s, CONDITION_SUBORDER)).collect();
        trap_list.extend(abilities.iter().map(|&s| (s, ABILITY_SUBORDER)));
        sort(self, &trap_list);
        // (2) MaybeTrapPokemon — abilities only.
        let maybe_list: Vec<(f64, i32)> =
            abilities.iter().map(|&s| (s, ABILITY_SUBORDER)).collect();
        sort(self, &maybe_list);
    }

    /// `updateSpeed()` (`battle.js:241` / `pokemon.js:283`): refresh BOTH actives'
    /// cached `pokemon.speed` to the live `getActionSpeed()` (para/boost/ModifySpe-
    /// aware). Showdown refreshes the cached speed at `commitChoices` (turn start,
    /// `battle.js:2494`) and the start of the `residual` action (`battle.js:2342`); the
    /// entrant's speed is also (re)established on SWITCH-IN (see `execute_switch`).
    /// Between those sites the cached value is STALE — a mon paralyzed mid-turn keeps its
    /// turn-start (full) speed for the rest of the move phase, which is what the
    /// `eachEvent` shuffles read. DRAW-FREE.
    pub(crate) fn update_speed(&mut self, dex: &Dex) {
        for side in 0..2 {
            let slot = self.sides[side].active;
            self.sides[side].pokemon[slot].cached_speed = self.effective_speed(side, slot, dex);
        }
    }

    /// Build the two move actions and order them by (priority → effective speed),
    /// wiring [`speed_sort`] so a priority+speed TIE draws the action-order
    /// Fisher-Yates shuffle from the PRNG (the production-path wiring). Returns the
    /// actions in resolved run order.
    pub(crate) fn order_actions(
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
    /// The **Quick Claw** `speed = 65535` override (a QC holder whose end-of-prev-turn
    /// `randomChance(1,5)` hit) IS now applied at the top of this fn
    /// (`gen3_quick_claw_speed_v1`, reading the persisted `self.quick_claw_roll`).
    /// **Remaining deferred `onModifySpe` input (none on our golden teams, so seed-parity
    /// holds):** the gen-3 weather-speed abilities are handled via the `weather_speed` chain
    /// below; no other `getActionSpeed` overrides apply in gen3 singles.
    pub(crate) fn effective_speed(&self, side: usize, slot: usize, dex: &Dex) -> u32 {
        let mon = &self.sides[side].pokemon[slot];
        // gen3 `getActionSpeed` (scripts.js:47-48): a Quick-Claw HOLDER whose end-of-
        // PREV-turn `battle.quickClawRoll` hit TRUE returns `speed = 65535` — moving FIRST
        // within its priority bracket regardless of raw Speed (`gen3_quick_claw_speed_v1`).
        // Uncapped (the gen3 override returns the raw 65535, NOT the 10000 `min`), so two
        // proc'd QC holders both read 65535 → an exact tie → the action-order shuffle draws,
        // matching the sim. Applied BEFORE the boost/para/weather chain (the override wholly
        // replaces the computed speed in `getActionSpeed`).
        if self.quick_claw_roll && to_id(&mon.item) == "quickclaw" {
            return 65535;
        }
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
    pub(crate) fn effective_weather(&self, dex: &Dex) -> Option<crate::state::Weather> {
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
    pub(crate) fn effective_accuracy(
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
    pub(crate) fn roll_accuracy(
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
    /// Sort a slice of queued actions by the full `comparePriority` key
    /// (`order → priority → speed → subOrder → effectOrder`, descending) via
    /// [`speed_sort`] — wiring the action-order Fisher-Yates tie-shuffle DRAW onto
    /// the switch path (two same-kind switches / two equal moves at a full tie).
    ///
    /// The handler `speed` is the OUTGOING (acting) mon's effective speed (the
    /// gen-3 `getActionSpeed`), so a faster mon's switch/move resolves first.
    /// `order` is the action's queue order (switch 103 before move 200, etc.).
    pub(crate) fn sort_actions(&mut self, actions: &mut Vec<QAction>, dex: &Dex) {
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
                        // A MUSTRECHARGE mon's action is the `recharge` pseudo-move (priority
                        // 0 — `gen3_move_coverage_batch4c_v1`), NOT slot 0's real move (whose
                        // priority could differ); the volatile is still set at sort time
                        // (cleared only at the cant). A CHARGING mon's `move_index` was
                        // already translated to its locked Solar Beam slot (priority 0).
                        let pr = if *struggle || self.sides[*side].pokemon[slot].must_recharge {
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
                    // A `beforeTurnMove` (order 5) sorts by the actor's speed; getActionSpeed
                    // sets NO `action.priority` for a non-`move` choice (undefined → 0 in
                    // comparePriority — VERIFIED vs the sim source `getActionSpeed`). Its speed
                    // is the actor's action speed (resolved by uid, in case the array swapped —
                    // it never does before the sort, but keep it robust). So two beforeTurnMove
                    // actions tie at order 5 / priority 0 / equal speed → the mirror shuffle.
                    QAction::BeforeTurnMove { side, uid, .. } => {
                        let slot = self.slot_of_uid(*side, *uid).unwrap_or(self.sides[*side].active);
                        (*side, slot, 0)
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

    /// The `comparePriority` key for one queued action (order, priority, speed).
    pub(crate) fn action_sort_key(&self, a: &QAction, dex: &Dex) -> (u64, i32, f64) {
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
            // A `beforeTurnMove` is never inserted via `insertChoice` (it's built at
            // queue-construction, not spliced mid-turn), but keep a total key for safety.
            QAction::BeforeTurn | QAction::BeforeTurnMove { .. } | QAction::Residual => {
                (a.order(), 0, 0.0)
            }
        }
    }
}
