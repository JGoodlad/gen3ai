use crate::damage::{
    AtkStatMod, BpMod, DamageContext,
};
use crate::dex::{to_id, Dex, DmgFold, MoveCategory, Type, TypeBoostFold};
use crate::protocol::{Cause, HpStatus, MonRef, Player};
use crate::state::{Status, Weather};
use super::*;

impl crate::state::BattleState {
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
        self.disable_move_event_shuffle(dex);

        // `activeTurns++` per active mon (`endTurn`, battle.ts:1762) — DRAW-FREE, AFTER the
        // DisableMove event + BEFORE the Quick Claw. Feeds NEXT turn's Speed Boost residual
        // gate (`gen3_ability_batch1_v1`). A mon that switched in this turn was reset to 0 in
        // `execute_switch`, so it becomes 1 here (its first boost comes NEXT turn's residual).
        self.bump_active_turns();

        // --- [16] END-OF-TURN Quick Claw randomChance(1,5): UNCONDITIONAL in gen3
        //     once endTurn completes (no faint this turn). PERSIST the result (was
        //     drawn-and-discarded) — read NEXT turn by `effective_speed` for the gen3
        //     `getActionSpeed` speed=65535 override (`gen3_quick_claw_speed_v1`). ---
        self.quick_claw_roll = self.prng.random_chance(1, 5);
        result.quick_claw_drawn = true;

        result
    }

    /// `endTurn`'s per-active `pokemon.activeTurns++` (battle.ts:1762) — DRAW-FREE, run after
    /// the residual/DisableMove and before the Quick Claw. Feeds the Speed Boost residual gate
    /// (`gen3_ability_batch1_v1`). A fainted mon is skipped (`if (pokemon.fainted) continue`).
    pub(crate) fn bump_active_turns(&mut self) {
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
                p1: self.side_snapshot(0, dex),
                p2: self.side_snapshot(1, dex),
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
    fn side_snapshot(&self, side: usize, dex: &Dex) -> MonSnapshot {
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
            item_num: dex.item(&to_id(&mon.item)).map(|i| i.num).unwrap_or(0),
        }
    }

    /// Clear the FLINCH volatile on BOTH actives (the `duration:1` end-of-turn
    /// expiry). DRAW-FREE — flinch carries no PRNG.
    pub(crate) fn clear_flinch(&mut self) {
        for side in 0..2 {
            let a = self.sides[side].active;
            self.sides[side].pokemon[a].flinch = false;
            // REVENGE's this-turn damage record (`gen3_bp_modifier_cluster_v1`) — the sim's
            // `attackedBy[].thisTurn`, so it expires exactly like `flinch`.
            self.sides[side].pokemon[a].damaged_by_foe_this_turn = false;
            // --- The `protect` this-turn volatile (`duration: 1`) expires at the next
            //     turn-top, exactly like `flinch` — clear it. The longer-lived `stall`
            //     volatile (`duration: 2`) is NOT touched here: its expiry happens at the
            //     RESIDUAL (`run_residuals` decrements `stall_duration` and zeros
            //     `protect_counter` at 0), faithful to `fieldEvent('Residual')`'s
            //     duration-handler `end`. (A SUCCESSFUL consecutive protect re-set
            //     `protected` true + reset `stall_duration` to 2 in `run_protect`, so it
            //     survives; a non-protect turn lets the residual run it down.) ---
            self.sides[side].pokemon[a].protected = false;
            // --- The FOCUS PUNCH (`focuspunch`) + PURSUIT (`pursuit`) `duration: 1`
            //     volatiles (`gen3_move_coverage_batch4_v1`) expire at the next turn-top,
            //     exactly like `flinch`. Both are RE-ADDED each turn by the `beforeTurnMove`
            //     action (Focus Punch on the user, Pursuit on the foe) IF the move is queued
            //     again — so clearing here + re-adding there is the residual `duration`
            //     countdown. A Pursuit volatile whose foe SWITCHED was already consumed
            //     (set to `None`) by the interrupt in `execute_switch`; this clears the
            //     "foe stayed in" case. DRAW-FREE. ---
            self.sides[side].pokemon[a].focus_punch = None;
            self.sides[side].pokemon[a].pursuit = None;
            // --- BEAT UP's `beatup` `duration: 1` volatile (`gen3_move_coverage_batch4b_v1`)
            //     expires at the next turn-top, exactly like `flinch`/`focus_punch`. Re-added
            //     each turn Beat Up runs (`run_beat_up`). DRAW-FREE. ---
            self.sides[side].pokemon[a].beat_up = false;
            // --- COUNTER / MIRROR COAT's reactive `duration: 1` volatile
            //     (`gen3_move_coverage_batch5_v1`) expires at the next turn-top, exactly
            //     like `focus_punch`. Re-added (with a RESET damage record) each turn the
            //     move is selected, by the order-5 `beforeTurnMove`. DRAW-FREE. ---
            self.sides[side].pokemon[a].reactive = None;
            // --- ENDURE's `duration: 1` volatile (`gen3_move_coverage_batch6_v1`)
            //     expires at the next turn-top, exactly like `protect` (its sibling —
            //     same stall machinery, different effect: the onDamage survive-at-1
            //     clamp instead of the move block). DRAW-FREE. ---
            self.sides[side].pokemon[a].endure = false;
            // --- SNATCH's `duration: 1` singleturn volatile (`gen3_snatch_v1`) expires at
            //     the next turn-top, exactly like `flinch`/`focus_punch`/`endure`. Set on
            //     the priority-+4 cast, it lives through the CAST turn (so it can intercept
            //     the foe's snatchable move + register the residual duration handler) and
            //     is cleared here at the FOLLOWING turn-top (probe SN1: t2 vols=(none)).
            //     DRAW-FREE. ---
            self.sides[side].pokemon[a].snatch = false;
        }
    }

    pub(crate) fn apply_damage(&mut self, side: usize, slot: usize, dmg: u16) -> bool {
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
    pub(crate) fn record_faint_order(&mut self, side: usize, slot: usize) {
        if slot == self.sides[side].active && !self.faint_emit_queue.contains(&side) {
            self.faint_emit_queue.push(side);
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
    /// Whether protocol emission is active (gates all the emit hooks — the
    /// disabled path pushes nothing AND, crucially, skips the state reads that
    /// build the emit args, so the hot seed-suite loop is untouched).
    #[inline]
    pub(crate) fn logging(&self) -> bool {
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
    pub(crate) fn display_name(&self, side: usize, slot: usize, dex: &Dex) -> String {
        let mon = &self.sides[side].pokemon[slot];
        let nick = &mon.set.name;
        if !nick.is_empty() {
            return nick.clone();
        }
        // `pokemon.name` is `set.name || set.species`, fixed at CONSTRUCTION — neither
        // `transformInto` (`gen3_transform_v1`: `|move|p1a: Ditto|Body Slam|…`, never
        // `p1a: Snorlax`) nor a non-permanent Forecast `formeChange` (`gen3_forecast_v1`,
        // probe S1: every ident of a FORMED Castform stays `p1a: Castform`) ever touches it.
        // Read the construction-fixed [`crate::state::MonState::base_species_id`] — which
        // also gets the FORMED-Castform-then-Transforms corner right (the overlay's stored
        // species would be the FORME).
        dex.species(&mon.base_species_id)
            .map_or_else(|| mon.base_species_id.clone(), |s| s.name.clone())
    }

    /// The SPECIES display name of `side`'s `slot` mon (e.g. `Zapdos`), for the
    /// `|switch|`/`|drag|` DETAILS field — distinct from the ident nickname
    /// ([`display_name`]).
    fn species_name(&self, side: usize, slot: usize, dex: &Dex) -> String {
        let mon = &self.sides[side].pokemon[slot];
        dex.species(&mon.species_id).map_or_else(|| mon.species_id.clone(), |s| s.name.clone())
    }

    /// A `MonRef` (`p<N>a: <Name>`) for `side`'s `slot` mon.
    pub(crate) fn mon_ref(&self, side: usize, slot: usize, dex: &Dex) -> MonRef {
        MonRef { side, name: self.display_name(side, slot, dex) }
    }

    /// A faithful `Pokemon.toString()` ident STRING for `side`'s `slot` mon
    /// (`gen3_omniscient_byte_fuzz_v1`, the future-move `-miss` source ref). Showdown's
    /// `Pokemon.toString()` (sim/pokemon.ts:532-534) renders an ACTIVE mon as the slot form
    /// `pNa: <Name>` (== [`mon_ref`]'s Display) but a mon NOT on the field as the SLOT-LESS
    /// `pN: <Name>`. A resolving Future Sight / Doom Desire's CASTER has usually switched out
    /// or fainted since casting (`conditions.ts` futuremove.onEnd strikes via the STORED
    /// `data.source`), so the `-miss` SOURCE must render slot-less when the caster isn't the
    /// current active — unlike `mon_ref`, whose `MonRef::Display` ALWAYS produces `pNa:`. Used
    /// only for the future-move resolve-miss caster ident (via [`ProtocolBuilder::miss_raw_user`]).
    pub(crate) fn mon_toref(&self, side: usize, slot: usize, dex: &Dex) -> String {
        let name = self.display_name(side, slot, dex);
        let tag = Player::from_side(side).tag();
        if slot == self.sides[side].active {
            format!("{tag}a: {name}")
        } else {
            format!("{tag}: {name}")
        }
    }

    /// A `MonRef` for `side`'s ACTIVE mon.
    fn active_ref(&self, side: usize, dex: &Dex) -> MonRef {
        self.mon_ref(side, self.sides[side].active, dex)
    }

    /// The HP-field (`x/y` / `x/y <status>` / `0 fnt`) for `side`'s `slot` mon,
    /// reading its live hp/maxhp/status.
    pub(crate) fn hp_status(&self, side: usize, slot: usize) -> HpStatus {
        let mon = &self.sides[side].pokemon[slot];
        HpStatus { hp: mon.hp, maxhp: mon.maxhp, status: status_token(mon.status) }
    }

    /// The `switch`/`drag` Details string (`Pokemon.details`): the species display
    /// name, then `, L<level>` iff level != 100, then `, <Gender>` when the mon has a
    /// real gender ('M'/'F'), then `, shiny` for a shiny set. Showdown emits `, L<n>`
    /// in details/switch/drag for a non-L100 mon and OMITS it at L100 — probe-confirmed
    /// (`/tmp/probe_level_details.js`, gen3randombattle): order is
    /// `<Species>[, L<level>][, <gender>][, shiny]` (level BEFORE gender). gen3ou is
    /// always L100 so the pool goldens never emit it (byte-identical); randbats surface
    /// it. Genderless capture teams show just the species
    /// (`|switch|p1a: Snorlax|Snorlax|524/524`); a gendered mon shows
    /// `|switch|p2a: Snorlax|Snorlax, M|461/461`; a non-L100 mon shows
    /// `|switch|p1a: Lunatone|Lunatone, L84|255/255`. This is observation-only: the
    /// protocol/writeline goldens use L100 genderless teams (so the output is unchanged
    /// there), and the e2e_fuzz gate compares SEED+STATE, not protocol lines.
    pub(crate) fn switch_details(&self, side: usize, slot: usize, dex: &Dex) -> String {
        let species = self.species_name(side, slot, dex);
        let mon = &self.sides[side].pokemon[slot];
        // The sim's `Pokemon` details = `<species>[, L<level>][, <gender>][, shiny]`
        // (level omitted at 100). The `, L<n>` suffix sits AFTER species, BEFORE gender.
        let mut s = species;
        if mon.level != 100 {
            s.push_str(&format!(", L{}", mon.level));
        }
        // The SHINY flag (`gen3_omniscient_byte_fuzz_v1` FORM 12 — a pool team declaring
        // a shiny mon shows `|switch|…|Quagsire, M, shiny|…`; the port dropped the
        // `set.shiny` flag).
        match mon.gender {
            Some('M') => s.push_str(", M"),
            Some('F') => s.push_str(", F"),
            _ => {}
        }
        if mon.set.shiny {
            s.push_str(", shiny");
        }
        s
    }

    /// Emit the battle-init framing (once, at the top of a logged battle), in the
    /// sim's exact order (the golden's first ~14 lines). The switch-in ability
    /// lines (`|-ability|`/`|-weather|`/`|-unboost|`) are DEFERRED (Phase 2) — the
    /// leads' `|switch|` + the `|turn|1` marker ARE emitted here (Phase 1). The
    /// `tier`/`rule` strings are the gen3customgame Custom-Game values the capture
    /// records (the port targets that format).
    pub(crate) fn emit_framing(&mut self, dex: &Dex) {
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
        // state (observation-only; the construction already applied the effects, before logging
        // was enabled). Emitted in the SAME order the leads' `runSwitch` actions actually fired:
        // the RECORDED `turn0_switchin_order` on the `gen3_turn0_construction_v1` bridge path
        // (whose speed-TIE order comes from a real `insertChoice` PRNG draw), else the draw-free
        // `run_start_switchins` faster-first / tie-keeps-side-order model (verified vs the
        // golden: Salamence Intimidate before Tyranitar Sand Stream when Salamence is faster).
        self.emit_switchin_ability_lines(dex);
        self.log.turn(1);
    }

    /// Reconstruct + emit the leads' switch-in ability lines (Phase 2), in the order the two
    /// leads' `runSwitch` actions ACTUALLY fired, then per lead emit — for **Intimidate** an
    /// `|-ability|<lead>|Intimidate|boost` + (if the foe's Atk drop was NOT blocked by
    /// Clear Body / White Smoke / Hyper Cutter) an `|-unboost|<foe>|atk|1`; for a
    /// **weather** ability (Sand Stream / Drizzle / Drought) whose resulting weather is
    /// STILL the current `field.weather` (the winning setter) an `|-weather|<Weather>|[from]
    /// ability: <AbilityName>|[of] <lead>` SET line. DRAW-FREE (a formatting read of state
    /// the construction already resolved).
    ///
    /// # Fire ORDER (`gen3_turn0_construction_mirror_order_v1`)
    ///
    /// The order comes from [`crate::state::BattleState::turn0_switchin_order`] when the battle
    /// came through the `gen3_turn0_construction_v1` window — there the two `runSwitch` actions
    /// are queued by the REAL `insertChoice`, whose raw-Speed TIE is broken by a PRNG
    /// `random(firstIndex, lastIndex+1)` draw, so a tie resolves p2-FIRST half the time.
    /// Re-deriving "faster first, tie = side order" here (the model of the DRAW-FREE
    /// [`crate::state::BattleState::run_start_switchins`] path) emitted the Intimidate /
    /// weather-setter block in the WRONG order on every p2-first tie — the Masquerain-mirror
    /// divergence `sbd_msdd8698_b293`. The fallback keeps that model for the draw-free path,
    /// whose STATE half orders the same way, so the committed goldens stay byte-identical.
    fn emit_switchin_ability_lines(&mut self, dex: &Dex) {
        if !self.logging() {
            return;
        }
        let leads: Vec<(usize, usize, u16)> = match self.turn0_switchin_order.clone() {
            // The construction window RESOLVED the order (a tie broken by its real draw).
            Some(order) => order
                .iter()
                .map(|&side| {
                    let slot = self.sides[side].active;
                    (side, slot, self.sides[side].pokemon[slot].stats[5])
                })
                .collect(),
            // The draw-free `start_with_switchins` path: faster raw Speed (stats[5]) first,
            // tie = side order (mirrors `run_start_switchins`'s stable
            // `sort_by(|a,b| b.spe.cmp(&a.spe))`).
            None => {
                let mut leads: Vec<(usize, usize, u16)> = (0..self.sides.len())
                    .map(|side| {
                        let slot = self.sides[side].active;
                        (side, slot, self.sides[side].pokemon[slot].stats[5])
                    })
                    .collect();
                leads.sort_by(|a, b| b.2.cmp(&a.2));
                leads
            }
        };

        // Simulate the sim's `Field.setWeather` emit decision in FIRE order (the resolved
        // order above). A weather ability emits its `-weather` SET line UNLESS the field
        // is ALREADY that (permanent) weather — in gen ≤ 5 `setWeather` returns false
        // (no line) when an Ability source re-sets the same weather at duration 0
        // (`gen3_omniscient_byte_fuzz_v1` FORM 7: two Sand-Stream leads emit ONE
        // `-weather` line, the FIRST setter's — the port used to emit one per lead).
        // Two DIFFERENT weathers still emit twice (Drizzle sets Rain, then Sand
        // Stream overwrites → both lines), so we track the running set weather.
        let mut emitted_weather: Option<Weather> = None;
        for (side, slot, _spe) in leads {
            // The lead's SET ability (== its current ability for everything but a
            // Trace lead, whose current ability is already the copy the construction
            // applied).
            let ability = to_id(&self.sides[side].pokemon[slot].set.ability);
            let desired = match ability.as_str() {
                "sandstream" => Some(Weather::Sand),
                "drizzle" => Some(Weather::Rain),
                "drought" => Some(Weather::Sun),
                _ => None,
            };
            let weather_line = match desired {
                // Emit only when this lead's weather is a REAL change from the running
                // set weather (a same-weather permanent re-set is a setWeather no-op).
                Some(w) => {
                    let emit = emitted_weather != Some(w);
                    emitted_weather = Some(w);
                    emit
                }
                None => false,
            };
            // The framing/lead reconstruction: a battle-start foe is never below −1, so an
            // Intimidate clamp never bites → the applied delta is always −1 (`None`).
            self.emit_ability_start_lines(side, slot, weather_line, None, dex);
        }
        // FORECAST (`gen3_forecast_v1`): a lead Castform FORMED by the start window emits
        // its `|-formechange|` INSIDE the framing, after the setter's `-weather` line and
        // before `|turn|1` (probe E1: `|switch|…|switch|…|-weather|SunnyDay|[from] ability:
        // Drought|…` → `|-formechange|p1a: Castform|Castform-Sunny|[msg]|[from] ability:
        // Forecast`). At most ONE lead can be formed here (two Castform leads means no
        // weather ability on the field), so a side walk needs no order model. The STATE was
        // already applied by the start path (`start_with_switchins` post-hoc / the
        // construction window's live wiring); this is the reconstruction's emit half.
        for side in 0..2 {
            let slot = self.sides[side].active;
            let mon = &self.sides[side].pokemon[slot];
            if to_id(&mon.base_species_id) == "castform" && mon.species_id != mon.base_species_id
            {
                let name = dex
                    .species(&mon.species_id)
                    .map(|s| s.name.clone())
                    .unwrap_or_else(|| mon.species_id.clone());
                let r = self.mon_ref(side, slot, dex);
                self.log.forme_change_forecast(&r, &name);
            }
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
    pub(crate) fn emit_ability_start_lines(
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
                // FORCED-REPLACEMENT mis-target guard (`gen3_intimidate_forced_replacement_v1`,
                // M3): if the intended foe was FORCE-REPLACED because it FAINTED between this mon's
                // switch action and the deferred `RunSwitch`, `intimidate_on_start` applied NO drop
                // — so emit NOTHING (matching the unchanged foe boosts), never a phantom `-unboost
                // atk|0`. This mirrors the `event::intimidate_on_start` suppression predicate
                // EXACTLY (the STATE and the EMISSION must agree): suppress ONLY when the captured
                // original is now FAINTED/absent (a forced replacement), NOT for a live foe-active
                // change (a DOUBLE VOLUNTARY switch leaves the original alive → the sim DOES drop
                // the new foe). The SIM instead resolves this Intimidate INLINE within the switch
                // action (empty `adjacentFoes()` → the gen3 "not activated" `-hint`); the port
                // DEFERS the entrant's RunSwitch past the turn boundary, so a byte-level match of
                // that hint's POSITION is out of scope — suppressing the drop is the STATE fix this
                // pins. DRAW-FREE.
                let replaced_by_faint = self.sides[side].pokemon[slot]
                    .switchin_foe_uid
                    .map(|u| {
                        u != self.sides[foe].pokemon[foe_slot].uid
                            && self.sides[foe]
                                .pokemon
                                .iter()
                                .find(|m| m.uid == u)
                                .map(|m| m.fainted || m.hp == 0)
                                .unwrap_or(true)
                    })
                    .unwrap_or(false);
                let mistarget = replaced_by_faint
                    || self.sides[foe].pokemon[foe_slot].fainted
                    || self.sides[foe].pokemon[foe_slot].hp == 0;
                if mistarget {
                    return;
                }
                if self.sides[foe].pokemon[foe_slot].substitute.is_some() {
                    // The gen3 mod's sub skip: NO `-ability`, just the hint.
                    self.log.hint(
                        "In Gen 3, Intimidate does not activate if every target has a Substitute.",
                        false, // gen3/abilities.ts:73 passes `false` → fires on EVERY occurrence
                    );
                    return;
                }
                let mon = self.mon_ref(side, slot, dex);
                self.log.ability(&mon, "Intimidate", Some("boost"));
                let foe_ability = to_id(&self.sides[foe].pokemon[foe_slot].ability);
                let foe_ref = self.mon_ref(foe, foe_slot, dex);
                if matches!(foe_ability.as_str(), "clearbody" | "whitesmoke" | "hypercutter") {
                    let stat = unboost_fail_stat_token(&foe_ability); // Hyper Cutter → "Attack"
                    let display = dex
                        .ability(&foe_ability)
                        .map(|a| a.name.clone())
                        .unwrap_or(foe_ability);
                    self.log.fail_unboost_from_ability(&foe_ref, &display, stat);
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
                    // WHITE HERB restore during the LEAD framing (`gen3_white_herb_v1`,
                    // `whiteherb.onAnySwitchIn`): a SLOWER White Herb holder dropped by this
                    // (faster) Intimidate lead restores at its own switch-in, emitting
                    // `-enditem`/`-clearnegativeboost` RIGHT AFTER the `-unboost` (the sim's byte
                    // order — the omniscient byte-fuzz find ab_6_13). FRAMING-ONLY (`intim_atk_pre
                    // .is_none()` — the lead reconstruction; a MID-BATTLE Intimidate switch-in does
                    // NOT restore here, since the holder's onAnySwitchIn already fired BEFORE this
                    // entrant's Start → no drop yet → it restores at `onAnyAfterMove` / the order-29
                    // residual). The STATE restore already fired in `start_with_switchins`; this is
                    // the reconstructed emission, keyed on the dropped foe being a CONSUMED White
                    // Herb holder (SET item whiteherb, current item now empty). DRAW-FREE.
                    if intim_atk_pre.is_none() {
                        let set_white_herb = dex
                            .item(&to_id(&self.sides[foe].pokemon[foe_slot].set.item))
                            .map_or(false, |i| i.boost_restore);
                        let consumed =
                            to_id(&self.sides[foe].pokemon[foe_slot].item).is_empty();
                        if set_white_herb && consumed {
                            self.log.push_raw(format!("|-enditem|{foe_ref}|White Herb"));
                            self.log
                                .push_raw(format!("|-clearnegativeboost|{foe_ref}|[silent]"));
                        }
                    }
                }
            }
            "sandstream" | "drizzle" | "drought" => {
                if weather_line {
                    // Emit this lead's OWN ability weather (NOT `field.weather`): with two
                    // DIFFERENT weather setters the FIRST (e.g. Drizzle→Rain) emits its
                    // weather even though the field ends on the second setter's (Sand).
                    let w = match ability.as_str() {
                        "sandstream" => Weather::Sand,
                        "drizzle" => Weather::Rain,
                        _ => Weather::Sun,
                    };
                    let mon = self.mon_ref(side, slot, dex);
                    let wname = weather_display(w);
                    let aname = self.sides[side].pokemon[slot].ability.clone();
                    self.log.weather(wname, Some(&Cause::Ability(aname)), Some(&mon), false);
                }
            }
            "pressure" => {
                let mon = self.mon_ref(side, slot, dex);
                self.log.ability_silent(&mon, "Pressure");
            }
            "trace" => {
                // Emit iff the copy applied — which mirrors `trace_on_start`'s only
                // no-copy case: a FAINTED/absent foe active (`foes()` excludes it → no
                // target → no copy, no draw, no line). Do NOT gate on `cur != "trace"`:
                // when the FOE's ability is ALSO Trace (a Porygon2-vs-Porygon Trace mirror)
                // the copy DOES apply yet leaves `cur == "trace"`, and the sim STILL emits
                // `|-ability|<mon>|Trace|Trace|[from] ability: Trace|[of] <foe>` (golden
                // ab_112_13 lines 14-15 — BOTH leads trace each other's Trace).
                let foe = 1 - side;
                let foe_active = self.sides[foe].active;
                if !self.sides[foe].pokemon[foe_active].fainted {
                    let cur = to_id(&self.sides[side].pokemon[slot].ability);
                    let copied_display = dex
                        .ability(&cur)
                        .map(|a| a.name.clone())
                        .unwrap_or_else(|| self.sides[side].pokemon[slot].ability.clone());
                    let foe_ref = self.mon_ref(foe, foe_active, dex);
                    let mon = self.mon_ref(side, slot, dex);
                    self.log.ability_traced(&mon, &copied_display, &foe_ref);
                }
            }
            _ => {}
        }
    }

    pub(crate) fn boundary_record(
        &self,
        request: RequestKind,
        first_mover: Option<usize>,
        dex: &Dex,
    ) -> DecisionRecord {
        DecisionRecord {
            request,
            active: [self.active_snapshot(0, dex), self.active_snapshot(1, dex)],
            active_species: [self.active_species_id(0), self.active_species_id(1)],
            pokemon_left: [self.sides[0].pokemon_left, self.sides[1].pokemon_left],
            spikes: [self.sides[0].spikes, self.sides[1].spikes],
            seed_after: self.prng.get_seed(),
            // A PURSUIT-INTERRUPT turn overrides the sorted-queue first_mover with the pursuer
            // (`gen3_move_coverage_batch4_v1`): the pursuer's `|move|Pursuit` is emitted before
            // the switcher's `|switch|`, so the sim reads the pursuer as the first mover.
            first_mover: if matches!(request, RequestKind::Move) {
                self.pursuit_first_mover.or(first_mover)
            } else {
                None
            },
            explosion_self_ko: self.pending_explosion_self_ko,
            phaze_drag: self.pending_phaze_drag,
            trapped: [self.is_trapped(0, dex), self.is_trapped(1, dex)],
            weather: self.field.weather,
            weather_turns: self.field.weather_turns,
            light_screen: [self.sides[0].light_screen, self.sides[1].light_screen],
            reflect: [self.sides[0].reflect, self.sides[1].reflect],
            curse: [
                self.sides[0].pokemon[self.sides[0].active].curse.is_some(),
                self.sides[1].pokemon[self.sides[1].active].curse.is_some(),
            ],
            wish_pending: [
                self.sides[0].wish_pending.as_ref().map(|(d, _)| *d).unwrap_or(0),
                self.sides[1].wish_pending.as_ref().map(|(d, _)| *d).unwrap_or(0),
            ],
            sub_hp: [
                self.sides[0].pokemon[self.sides[0].active].substitute.unwrap_or(0),
                self.sides[1].pokemon[self.sides[1].active].substitute.unwrap_or(0),
            ],
            future_pending: [
                self.sides[0].future_move.as_ref().map(|f| f.duration).unwrap_or(0),
                self.sides[1].future_move.as_ref().map(|f| f.duration).unwrap_or(0),
            ],
            encore: [
                self.sides[0].pokemon[self.sides[0].active].encore.map(|(_, t)| t).unwrap_or(0),
                self.sides[1].pokemon[self.sides[1].active].encore.map(|(_, t)| t).unwrap_or(0),
            ],
            perish: [
                self.sides[0].pokemon[self.sides[0].active].perish.unwrap_or(0),
                self.sides[1].pokemon[self.sides[1].active].perish.unwrap_or(0),
            ],
        }
    }

    /// Active-mon snapshot (the per-decision STATE the differential asserts).
    fn active_snapshot(&self, side: usize, dex: &Dex) -> MonSnapshot {
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
            item_num: dex.item(&to_id(&mon.item)).map(|i| i.num).unwrap_or(0),
        }
    }

    /// The active mon's resolved species id (e.g. `tyranitar`) — proves a switch
    /// brought the right mon to the active slot (the array-swap correctness).
    fn active_species_id(&self, side: usize) -> String {
        let a = self.sides[side].active;
        self.sides[side].pokemon[a].species_id.clone()
    }
}


/// Compare two `comparePriority` keys (order asc, priority desc, speed desc).
/// Returns <0 if `a` sorts BEFORE `b`, >0 if after, 0 on a full tie — matching
/// `comparePriority`'s sign convention used by `insertChoice` (`compared <= 0`
/// ⇒ a sorts at/before b).
pub(crate) fn compare_keys(a: &(u64, i32, f64), b: &(u64, i32, f64)) -> f64 {
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

/// Recover the exact `(num, den)` integer ratio from a `recoil`/`drain` FRACTION
/// (`gen3_move_coverage_batch1_v1`). The data stores the pre-divided float
/// (`recoilFraction`/`drainFraction`); the engine needs the EXACT integer floor/ceil, so it
/// reconstructs the smallest `den` making `frac·den` integral and `num = round(frac·den)`.
/// The gen-3 (+ modeled) values are a fixed small set — `[1,2]`, `[1,3]`, `[1,4]`, `[3,4]`,
/// `[33,100]`, `[1,2]` — resolved by exact-float compare (they round-trip bit-for-bit; a stray
/// float would GIGO-panic rather than silently mis-round). Used by `apply_recoil`/`apply_drain`.
fn fraction_to_ratio(frac: f64) -> (u16, u16) {
    // Compare against the exact stored floats (division of small ints in f64 is exact for
    // these). Ordered by frequency (gen-3 OU: [1,2] drain, [1,3]/[1,4] recoil).
    const TABLE: &[(u16, u16)] = &[
        (1, 2),
        (1, 3),
        (1, 4),
        (3, 4),
        (33, 100),
        (1, 8),
        (1, 16),
    ];
    for &(n, d) in TABLE {
        if frac == (n as f64) / (d as f64) {
            return (n, d);
        }
    }
    panic!(
        "unexpected recoil/drain fraction {frac} — not in the known gen-3 ratio table; \
         add its (num, den) to `fraction_to_ratio` (GIGO guard so it can never mis-round)."
    );
}

/// See [`fraction_to_ratio`] — the recoil-fraction wrapper (kept as a named alias for the
/// `apply_recoil` call site's readability).
pub(crate) fn recoil_fraction_to_ratio(frac: f64) -> (u16, u16) {
    fraction_to_ratio(frac)
}

/// See [`fraction_to_ratio`] — the drain-fraction wrapper.
pub(crate) fn drain_fraction_to_ratio(frac: f64) -> (u16, u16) {
    fraction_to_ratio(frac)
}

/// The gen-3 SUBSTITUTE cost = `floor(maxhp/4)`, which is ALSO the created sub's HP
/// (`directDamage(maxhp/4)` floors, and the volatile's `onStart` sets `effectState.hp =
/// Math.floor(maxhp/4)`). A Substitute FAILS (draw-free) if `hp <= floor(maxhp/4)` — the
/// user can't afford to make a sub it has exactly enough HP for (the gen-3 `<=` boundary;
/// VERIFIED: hp == floor(maxhp/4) FAILS, hp == that + 1 SUCCEEDS).
pub(crate) fn sub_cost(maxhp: u16) -> u16 {
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
pub(crate) fn move_is_immune(ctx: &DamageContext, dex: &Dex) -> bool {
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
pub(crate) fn status_type_immune(status: &str, types: &[Type]) -> bool {
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
pub(crate) fn nature_minus_stat(nature: &str, dex: &Dex) -> Option<String> {
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

/// Whether a foe holding **Pressure** is in the move's `pressureTargets` — i.e. the
/// move actually TARGETS a foe (so Pressure's `onDeductPP` fires the +1 extra PP drop).
/// `gen3_pressure_allyteam_v1` — mirrors Showdown's `Pokemon.getMoveTargets` +
/// `pressureTargets` resolution (`sim/pokemon.ts:792-861`) for gen-3 SINGLES: the foe is
/// a pressure target for every FOE-directed target (`normal` / `any` / `randomNormal` /
/// `adjacentFoe` / `allAdjacentFoes` / `scripted`, plus the spread `all` / `allAdjacent`
/// whose target list includes foes), and NOT for the ally/self-directed targets (`self` /
/// `allyTeam` [Aromatherapy / Heal Bell] / `allySide` / `allies` / `adjacentAlly` /
/// `adjacentAllyOrSelf`) NOR `foeSide` (Spikes — `pressureTargets` is explicitly emptied).
/// VERIFIED vs the sim (`harness/probe_pressure_allyteam_rng.js` + the real-battle PP
/// count): Aromatherapy / Heal Bell under a Pressure foe deduct 1, ThunderVane / Seismic
/// Toss deduct 2. This replaces the wrong `!targets_self` predicate — the e2e_182 cause.
/// Whether a status move's `|move|<user>|<Name>|<TARGET>` announce renders the USER as
/// the TARGET field (`gen3_omniscient_byte_fuzz_v1`): a NON-foe-directed move (`self` /
/// `allySide` / `all` / `allyTeam` / `allies` / `adjacentAlly` / `adjacentAllyOrSelf`)
/// shows the source mon; a foe-directed (`normal`/`any`/…) or `foeSide` (Spikes) move
/// shows the FOE active. (Distinct from `pressure_targets_foe`, which treats `foeSide` as
/// non-foe for PP; here `foeSide` renders the foe active.)
pub(crate) fn status_move_announce_renders_user(target: &str) -> bool {
    matches!(
        target,
        "self" | "allyTeam" | "allySide" | "allies" | "adjacentAlly" | "adjacentAllyOrSelf" | "all"
    )
}

pub(crate) fn pressure_targets_foe(target: &str) -> bool {
    // `foeSide` (the only gen-3 `foeSide` move is Spikes) DOES put the Pressure foe in the
    // move's `pressureTargets` → the −2 deduction (SIM-PROBE-CONFIRMED, `gen3_pressure_foeside_v1`:
    // Skarmory Spikes vs a Pressure Suicune = pp 30/32 = −2, vs a non-Pressure foe = 31/32 = −1).
    // Only `self` / `allyTeam` / `allySide` / `allies` / ally-adjacent (Aromatherapy / Heal Bell /
    // setup / recovery) escape the extra — the e2e_182 `allyTeam` case stays −1.
    !matches!(
        target,
        "self" | "allyTeam" | "allySide" | "allies" | "adjacentAlly" | "adjacentAllyOrSelf"
    )
}

pub(crate) fn status_token(status: Option<Status>) -> Option<&'static str> {
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
pub(crate) fn weather_display(weather: Weather) -> &'static str {
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
pub(crate) fn modeled_phaze_move(move_id: &str) -> bool {
    matches!(move_id, "roar" | "whirlwind")
}

pub(crate) fn modeled_status_move(move_id: &str) -> Option<&'static str> {
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

/// The MODELED WEATHER-SET moves → the [`Weather`] they set for 5 turns. Rain Dance →
/// Rain, Sunny Day → Sun (`gen3_move_coverage_batch2_v1`); **Hail → Hail, Sandstorm →
/// Sand (`gen3_forecast_v1`, ROUND 35** — Forecast needs a reachable TIMED hail for
/// Castform-Snowy, and the class machinery is weather-generic). All four are never-miss,
/// 5-turn timed (gen 3 has no Damp/Heat/Icy/Smooth Rock), share the set / fail-into-same /
/// upkeep / expiry byte forms, and the hail/sand CHIP + Ice / Rock-Ground-Steel immunities
/// were already modeled for the ability weathers — probe-verified on TIED boards by
/// `harness/probe_r35_weather_moves.js` (set `|-weather|Hail` + WeatherChange draw; upkeep
/// chip `[from] Hail` at max(1,maxhp/16); re-cast `[still]`+`-fail`; expiry `-weather|none`
/// + the UNCONDITIONAL WeatherChange draw).
pub(crate) fn modeled_weather_set_move(move_id: &str) -> Option<Weather> {
    match move_id {
        "raindance" => Some(Weather::Rain),
        "sunnyday" => Some(Weather::Sun),
        "hail" => Some(Weather::Hail),
        "sandstorm" => Some(Weather::Sand),
        _ => None,
    }
}

/// The MODELED SCREEN moves (`gen3_move_coverage_batch2_v1`) → `Some(true)` for Reflect
/// (halves PHYSICAL), `Some(false)` for Light Screen (halves SPECIAL), `None` otherwise.
pub(crate) fn modeled_screen_move(move_id: &str) -> Option<bool> {
    match move_id {
        "reflect" => Some(true),
        "lightscreen" => Some(false),
        _ => None,
    }
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
pub(crate) fn self_boost_spec(move_id: &str, dex: &Dex) -> Option<Vec<(usize, i8)>> {
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
pub(crate) fn recovery_heal_amount(move_id: &str, mon: &crate::state::MonState, weather: Option<Weather>) -> Option<u16> {
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
/// MODELED (`gen3_move_coverage_batch5_v1`, the arms below): **Counter / Mirror Coat**
/// (the reactive volatile's recorded 2× damage — the caller's onTry gate already
/// zero-draw-failed when un-armed) and **Endeavor** (`target.hp − pokemon.hp`, the
/// caller's onTry gate already failed `hp >= target.hp`).
///
/// DEFERRED (fail-loud — these need extra RNG or accumulator/OHKO machinery the caller
/// PANICS on, never silently no-ops): **Psywave** (variable — draws RNG), the OHKO
/// moves **Fissure / Horn Drill / Guillotine** (accuracy-gated instakill + the level
/// gate), **Bide** (a 2-turn accumulator). They are NOT in this set → the caller's
/// fixed-damage fail-loud guard fires.
pub(crate) fn fixed_damage_amount(
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
        // COUNTER / MIRROR COAT (`gen3_move_coverage_batch5_v1`): `damageCallback` =
        // the reactive volatile's recorded `damage` (already 2× the last qualifying
        // hit — see `MonState::reactive`). The caller's onTry gate (`run_fixed_damage_
        // move`) already failed zero-draw when the volatile is missing/un-armed, so an
        // unarmed read here is a programming error, not a silent fallback.
        "counter" | "mirrorcoat" => attacker
            .reactive
            .and_then(|r| r.damage)
            .expect("counter/mirrorcoat amount read past the onTry gate (must be armed)"),
        // ENDEAVOR (`gen3_move_coverage_batch5_v1`): `damageCallback` = target.hp −
        // pokemon.hp — sets the target's MON hp to EXACTLY the user's hp (never a KO;
        // reads the MON's hp behind a sub, the number then lands on the SUB with no
        // carry — probed E4). The caller's onTry gate already failed `hp >= target.hp`
        // (EQUALITY INCLUDED — probed 50v50 fails), so the subtraction never underflows.
        "endeavor" => defender.hp - attacker.hp,
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
/// Shade / Sonic Boom / Dragon Rage / Super Fang, plus the `gen3_move_coverage_batch5_v1`
/// reactive family Counter / Mirror Coat / Endeavor) AND the DEFERRED fixed-damage family
/// (Psywave — variable, draws RNG; the OHKO moves Fissure / Horn Drill / Guillotine —
/// accuracy-gated instakill + a level gate; Bide — a 2-turn accumulator). Listing the
/// deferred ones here (not relying on the `base_power == 0` fall-through) makes their
/// exclusion EXPLICIT + FAIL-LOUD: a real team that carries one PANICS in
/// `run_fixed_damage_move` instead of quietly desyncing.
pub(crate) fn is_fixed_damage_move(move_id: &str) -> bool {
    matches!(
        move_id,
        // Modeled (bit-for-bit):
        "seismictoss" | "nightshade" | "sonicboom" | "dragonrage" | "superfang"
        | "counter" | "mirrorcoat" | "endeavor"
        // Deferred (fail-loud in run_fixed_damage_move):
        | "psywave" | "fissure" | "horndrill" | "guillotine" | "bide"
    )
}

/// Whether `move_id` (already `to_id`-normalized) carries a gen3 `beforeTurnCallback`
/// (`gen3_move_coverage_batch4_v1`), so the queue-builder must unshift a `beforeTurnMove`
/// action (order 5) for its move. The gen-3 (resolved via the gen4 mod) carriers are
/// **Focus Punch** (adds its `focuspunch` volatile to the user) and **Pursuit** (lays the
/// `pursuit` volatile on the target). Id-gated per the `is_fixed_damage_move` precedent
/// (`gen3_moves.json` carries no `beforeTurnCallback` marker). The resolved gen3 dex has FOUR
/// carriers — `counter` / `mirrorcoat` / `pursuit` / `focuspunch` — ALL modeled now:
/// Counter + Mirror Coat (`gen3_move_coverage_batch5_v1`) add their reactive volatile to
/// the USER (its onStart RESETS `{slot:null, damage:0}` every selection turn) at order 5,
/// with a `duration:1` residual duration handler like Focus Punch's.
pub(crate) fn move_has_before_turn_callback(move_id: &str) -> bool {
    matches!(move_id, "focuspunch" | "pursuit" | "counter" | "mirrorcoat")
}

/// The gen-3 VARIABLE-BP `basePowerCallback` family with a bp-0 data row
/// (`gen3_move_coverage_batch5_v1`) — the engine-computed BP, all deterministic STATE
/// reads consuming ZERO PRNG (the Water Spout precedent). DRAW-NEUTRALITY
/// probe-proven for Return h255 vs h3 / Frustration h0 vs h252 / Flail full vs 1 HP
/// (each pair ends at byte-identical seeds, `probe_batch5_varbp.js`); the probe's Low
/// Kick heavy-vs-light leg is KO-CONFOUNDED (the heavy case KO'd the target → 3 vs 4
/// draws, legitimately different flows) — Low Kick's neutrality rests on the SAME
/// zero-PRNG code path plus the 18548-row batch-5 golden, not that leg's seed compare.
/// Returns `Some(bp)` for the five members, `None` otherwise.
/// Formulas from the RESOLVED gen3 sources (dumped in `probe_batch5_varbp.js`):
///
///   * **Return**: `floor(happiness * 10 / 25) || 1` — h255 → 102 (the natural max, no
///     explicit cap); h ∈ {0,1,2} floors to 0 → the `|| 1` clamp → BP **1** (NOT a fail).
///   * **Frustration**: `floor((255 − happiness) * 10 / 25) || 1` — h0 → 102.
///   * **Flail / Reversal** (identical callback): `ratio = max(floor(hp*48/maxhp), 1)`,
///     then the FLOORED-integer bands `<2 → 200, <5 → 150, <10 → 100, <17 → 80,
///     <33 → 40, else 20` (live-verified edges at maxhp 461: hp 1-19 → 200, 20-48 → 150,
///     49-96 → 100, 97-163 → 80, 164-316 → 40, 317+ → 20; gen4 changes the 48 to 64 —
///     gen3 is 48).
///   * **Low Kick**: from the TARGET's `getWeight()` in HECTOGRAMS (`SpeciesData::
///     weighthg` — gen3 has NO ModifyWeight handler): `>=2000 → 120, >=1000 → 100,
///     >=500 → 80, >=250 → 60, >=100 → 40, else 20` (cutoffs swept exact at
///     100/250/500/1000/2000 hg).
///
/// No fail conditions — the `|| 1` clamp means BP can never be 0. The multiplies widen
/// to u32 (`48 * 714` etc. overflow u16).
pub(crate) fn variable_bp(
    move_id: &str,
    user: &crate::state::MonState,
    target: &crate::state::MonState,
    dex: &Dex,
) -> Option<u16> {
    Some(match move_id {
        "return" => (((user.set.happiness as u32) * 10 / 25).max(1)) as u16,
        "frustration" => ((((255 - user.set.happiness as u32) * 10) / 25).max(1)) as u16,
        "flail" | "reversal" => {
            let ratio = ((48u32 * user.hp as u32) / user.maxhp.max(1) as u32).max(1);
            match ratio {
                0..=1 => 200,
                2..=4 => 150,
                5..=9 => 100,
                10..=16 => 80,
                17..=32 => 40,
                _ => 20,
            }
        }
        "lowkick" => {
            // FAIL-LOUD weight read (GIGO guard, review nit): every real gen3 species
            // weighs >= 0.1 kg = 1 hg, and `dex/species.rs` defaults `weighthg` to 0
            // when the JSON field is ABSENT — so a missing species or a 0 weight means
            // a data resync dropped the field, which would silently price EVERY Low
            // Kick at BP 20. PANIC instead (the dex `batch5_tests` weighthg anchors
            // are the value gate; this guards the missing-field class).
            let hg = dex
                .species(&target.species_id)
                .unwrap_or_else(|| {
                    panic!(
                        "lowkick: unknown target species {:?} — the weight-based BP \
                         ladder needs SpeciesData",
                        target.species_id
                    )
                })
                .weighthg;
            assert!(
                hg >= 1,
                "lowkick: species {:?} has weighthg {hg} — gen3_species.json lost its \
                 `weighthg` field (a resync regression); every real species is >= 1 hg",
                target.species_id
            );
            match hg {
                0..=99 => 20,
                100..=249 => 40,
                250..=499 => 60,
                500..=999 => 80,
                1000..=1999 => 100,
                _ => 120,
            }
        }
        _ => return None,
    })
}

/// Whether `move_id` (already `to_id`-normalized) carries the gen3 `flags.defrost`
/// (`gen3_defrost_v1`): a FROZEN user of such a move still draws the 1/5 thaw roll,
/// but on a FAILED roll it PROCEEDS anyway and is thawed draw-free by
/// `frz.onModifyMove` (see `on_before_move`'s FREEZE arm). Sacred Fire and Flame
/// Wheel are the ONLY two gen3 defrost carriers (probe
/// `harness/probe_sacredfire_defrost.js` prints the resolved flags; Flare Blitz is
/// gen4). Id-gated per the `is_fixed_damage_move` precedent — `gen3_moves.json`
/// carries no defrost flag (`contact`/`sound` are the only extracted flags).
pub(crate) fn is_defrost_move(move_id: &str) -> bool {
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
pub(crate) fn status_move_checks_type_immunity(move_id: &str) -> bool {
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
pub(crate) fn stat_drop_blocked(ability_id: &str, stat_idx: usize) -> bool {
    match ability_id {
        "clearbody" | "whitesmoke" => true,
        "hypercutter" => stat_idx == 0,  // atk
        "keeneye" => stat_idx == 5,      // accuracy
        _ => false,
    }
}

/// The `-fail|<mon>|unboost|<Stat>|…` STAT TOKEN a boost-blocking ability's `onTryBoost`
/// emits, SIM-PROBED (abilities.js): the SINGLE-STAT blockers carry the ability's own literal
/// token — Hyper Cutter → `"Attack"`, Keen Eye → `"accuracy"` — while the WHOLE-table
/// blockers (Clear Body / White Smoke) carry NONE. (`gen3_omniscient_byte_fuzz_v1`.)
pub(crate) fn unboost_fail_stat_token(ability_id: &str) -> Option<&'static str> {
    match ability_id {
        "hypercutter" => Some("Attack"),
        "keeneye" => Some("accuracy"),
        _ => None, // clearbody / whitesmoke → whole-table block, no stat token
    }
}

/// Whether a move of `t` is in the gen3 mod's Hustle physical-type list
/// (`gen3_accuracy_pipeline_v1`). The RESOLVED gen3 `hustle.onSourceModifyAccuracy` gates
/// its ×0.8 on `physicalTypes.includes(move.type)` — the gen1-3 TYPE-based physical/special
/// split — NOT `move.category`. Probe-confirmed vs the resolved dist
/// (`harness/probe_accuracy_tohit.js`): a Normal move (Tackle) drops to ×0.8, an Electric
/// move (Thunder) is unaffected.
pub(crate) fn hustle_boosts_accuracy_type(t: Type) -> bool {
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
pub(crate) fn accuracy_chain_modify(value: u64, mods: &[(u64, u64)]) -> u64 {
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
pub(crate) fn apply_boost(stat: u32, boost: i8) -> u32 {
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
impl crate::state::BattleState {
    /// The ACTIVE mon currently locked into UPROAR, on EITHER side (`gen3_uproar_v1`).
    /// The condition's handler is `onAnySetStatus`, so one uproar anywhere on the field
    /// blocks sleep for everyone — hence a field-wide scan rather than a per-side read.
    pub(crate) fn any_uproarer(&self) -> Option<(usize, usize)> {
        for side in 0..2 {
            let slot = self.sides[side].active;
            let mon = &self.sides[side].pokemon[slot];
            if mon.uproar.is_some() && !mon.fainted {
                return Some((side, slot));
            }
        }
        None
    }
}

pub(crate) fn mon_types(mon: &crate::state::MonState, dex: &Dex) -> Vec<Type> {
    match &mon.types_override {
        Some(t) => t.clone(),
        None => species_types(&mon.species_id, dex),
    }
}

/// The holder-species gate for a SPECIES_STAT item (`ItemData.stat_mods.only_species`
/// — Thick Club Cubone/Marowak, Light Ball Pikachu, DeepSea* Clamperl, Metal Powder
/// Ditto, Soul Dew Lati@s). Empty ⇒ unconditional (Choice Band).
///
/// `untransformed_only` (Metal Powder's `!pokemon.transformed`) is now WIRED
/// (`gen3_transform_v1`). It is only *independently* load-bearing on the DITTO MIRROR — any
/// other transform already moves `species_id` off `ditto` and fails the `only_species` gate —
/// but that board is reachable (both sides roll Ditto; one copies the other while holding
/// Metal Powder) and the Def ×2 is a damage-visible difference.
fn species_gate_passes(
    mods: &crate::dex::StatMods,
    species_id: &str,
    holder_transformed: bool,
) -> bool {
    if mods.untransformed_only && holder_transformed {
        return false;
    }
    mods.only_species.is_empty() || {
        let sid = to_id(species_id);
        mods.only_species.iter().any(|s| *s == sid)
    }
}

impl crate::state::BattleState {
    /// The EFFECTIVE crit ratio for a move used by `(side, slot)` — the move's base ratio
    /// (`gen3_moves.json critRatio`, 1 normal / 2 for the high-crit set) folded with the two
    /// gen-3 `onModifyCritRatio` handlers the port models, then `clampIntRange(_, 0, 5)` (the
    /// `CRIT_MULT` table cap): **FOCUS ENERGY** (+2 — in gen3 reachable only via a Lansat Berry
    /// eat) + a **CRIT_ITEM** (`gen3_crit_item_v1`, `ItemData.crit_boost` — Scope Lens +1
    /// unconditional; Lucky Punch +2 Chansey; Stick +2 Farfetch'd; the species gate is
    /// `user.species.id`). BOTH are DRAW-FREE folds: they shift the `randomChance(1,
    /// CRIT_MULT[ratio])` DENOMINATOR (1→3 ⇒ 1/16→1/4), never the draw COUNT. Used at every
    /// crit-roll site (`run_move` / `run_multihit` / `run_beat_up` / the jump-kick crash).
    pub(crate) fn effective_crit_ratio(&self, side: usize, slot: usize, base: u8, dex: &Dex) -> u32 {
        let mon = &self.sides[side].pokemon[slot];
        let mut ratio = base as u32;
        if mon.focus_energy {
            ratio += 2;
        }
        if let Some(cb) = dex.item(&to_id(&mon.item)).and_then(|i| i.crit_boost.as_ref()) {
            let species_ok = cb.only_species.is_empty()
                || cb.only_species.iter().any(|s| *s == to_id(&mon.species_id));
            if species_ok {
                ratio += cb.boost as u32;
            }
        }
        ratio.min(5)
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
pub(crate) fn resolve_atk_stat_mods(
    item: &str,
    ability: &str,
    species_id: &str,
    // `pokemon.transformed` for the ITEM HOLDER (the attacker) — Metal Powder's
    // `untransformedOnly` gate (`gen3_transform_v1`).
    holder_transformed: bool,
    move_type: Option<Type>,
    category: MoveCategory,
    attacker_statused: bool,
    dex: &Dex,
) -> Vec<AtkStatMod> {
    let mut mods = Vec::new();
    if let Some(it) = dex.item(item) {
        if let Some(sm) = &it.stat_mods {
            if species_gate_passes(sm, species_id, holder_transformed) {
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
pub(crate) fn resolve_def_stat_mods(
    item: &str,
    ability: &str,
    species_id: &str,
    // `pokemon.transformed` for the ITEM HOLDER (the defender) — Metal Powder's
    // `untransformedOnly` gate (`gen3_transform_v1`).
    holder_transformed: bool,
    category: MoveCategory,
    defender_statused: bool,
    dex: &Dex,
) -> Vec<AtkStatMod> {
    let mut mods = Vec::new();
    if let Some(it) = dex.item(item) {
        if let Some(sm) = &it.stat_mods {
            if species_gate_passes(sm, species_id, holder_transformed) {
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
pub(crate) fn resolve_bp_mods(
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
pub(crate) fn tryhit_absorb_ability(ability_id: &str, move_type: Option<Type>) -> Option<&'static str> {
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
pub(crate) fn resolve_defender_ability(
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

/// The gen-3 **PARTIAL-TRAP** move family (`gen3_partial_trap_v1`) — the six moves whose
/// resolved gen-3 dex row carries `volatileStatus: 'partiallytrapped'`:
///
/// | move | num | type | cat | BP | acc |
/// |---|---|---|---|---|---|
/// | wrap | 35 | Normal | Physical (contact) | 15 | 85 |
/// | bind | 20 | Normal | Physical (contact) | 15 | 75 |
/// | firespin | 83 | Fire | Special | 15 | 70 |
/// | clamp | 128 | Water | Special (contact) | 35 | 75 |
/// | whirlpool | 250 | Water | Special | 15 | 70 |
/// | sandtomb | 328 | Ground | Physical | 15 | 70 |
///
/// HAND-LISTED rather than data-driven because `gen3_moves.json` (the shared RL data
/// facade) carries no `volatileStatus` column, and widening it is a retrain-class change
/// to the observation pipeline for a fact with exactly six carriers. The set is pinned by
/// `partial_trap_family_is_exactly_the_six_volatilestatus_carriers` (dex-derived: every
/// member must exist with the BP/accuracy above) and by the harness's
/// `MODELED_PARTIALTRAP_MOVES` mirror in `gen_e2e_fuzz.js`.
pub(crate) fn is_partial_trap_move(move_id: &str) -> bool {
    matches!(move_id, "wrap" | "bind" | "firespin" | "clamp" | "whirlpool" | "sandtomb")
}
