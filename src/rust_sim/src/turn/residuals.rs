use crate::dex::{to_id, Dex, Type};
use crate::event::{speed_sort, EventHandler, NO_ORDER};
use crate::protocol::Cause;
use crate::state::{FutureMove, Status, Weather};
use super::*;
use super::helpers::*;

impl crate::state::BattleState {

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
    pub(crate) fn run_residuals(&mut self, dex: &Dex) {
        // The `residual` action refreshes the cached `pokemon.speed` FIRST
        // (`battle.js:2342` `this.updateSpeed()` before `fieldEvent("Residual")`), so
        // the handler-sort below + the nested `eachEvent('Weather')` shuffle tie on the
        // CURRENT (para/boost-aware) speed — even for a mon that switched in mid-turn
        // with a stale raw cached speed.
        self.update_speed(dex);

        // --- SCREENS are gathered as SIDE-condition residual DURATION handlers below
        //     (`ResidualAction::ScreenDuration`, `gen3_screen_residual_tie_shuffle_v1`), NOT
        //     decremented directly here. Reflect `onSideResidualOrder` 1 / Light Screen 2
        //     (the gen4-mod override gen3 inherits, `data/mods/gen4/moves.ts`), priority 0,
        //     speed 0, subOrder 4 (the `resolvePriority` SideCondition effectTypeOrder). They
        //     sort well BEFORE the field weather residual (order 8) and the mon handlers (order
        //     10), so the one-screen `-sideend` expiry still emits at the SAME stream position
        //     (first among the residuals). THE DRAW: `fieldEvent('Residual')` gathers a side's
        //     `onSideResidual` duration handlers via `findSideEventHandlers` and speed-sorts them
        //     with EVERYTHING else — so when BOTH sides carry the SAME screen (two Reflects tie at
        //     order 1, two Light Screens at order 2; speed 0 both) the tie-group Fisher-Yates
        //     shuffle draws ONE `random(0,2)` — the draw the old direct decrement MISSED (the
        //     residual-side sibling of the batch-7 per-hit `modify_damage_phase1_shuffle`). A
        //     SINGLE screen (one side) is alone in its order group → NO tie → NO draw, byte-neutral. ---

        // --- Gather residual handlers (one per active per applicable effect), each
        //     with its resolved comparePriority key + a typed action. ---
        let mut handlers: Vec<EventHandler<ResidualAction>> = Vec::new();

        // --- The WISH (order 7) + FUTUREMOVE (order 11) SLOT CONDITIONS are gathered
        //     PER-ACTIVE at the END of the per-active loop below (after that active's
        //     item), NOT here in a pre-loop (`gen3_leftovers_slotcond_gather_order_v1`,
        //     the R2 fix). THE PRE-SORT ORDER IS LOAD-BEARING: `speed_sort` is a
        //     NON-STABLE selection sort whose swaps DISTURB the relative order of the
        //     tied handlers, so the tie-group Fisher-Yates shuffle reads whatever pre-sort
        //     order the selection-sort swaps LEFT the tied pair in. Showdown's
        //     `fieldEvent('Residual')` gathers a side's slot conditions via
        //     `findSideEventHandlers(side, 'onResidual', …, active)` — which runs AFTER
        //     `findPokemonEventHandlers(active)` — so Wish/FutureMove sit AFTER that
        //     active's Leftovers in the pre-sort array. Gathering them FIRST (the old
        //     pre-loop) made the selection sort's Wish/weather swaps REVERSE the tied
        //     Leftovers pair vs the sim → the two `-heal` lines emitted in the OPPOSITE
        //     order at the SAME shuffle value (the R2 byte-fuzz divergence — a Jolteon
        //     mirror, both Leftovers, sandstorm, a pending p2 Wish: the sim heals p1 first,
        //     the port healed p2 first, seed IDENTICAL). Moving them into the per-active
        //     loop is DRAW-NEUTRAL (same handlers, same sort keys, same tie-group COUNT →
        //     the seed is unchanged) — only the pre-sort POSITION, hence the emit
        //     permutation, changes to match the sim. ---

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
        // RM3 (`gen3_sand_upkeep_under_air_lock_v1`): the WeatherChip field-residual handler
        // is scheduled off the RAW `field.weather` for ALL weather (sun/rain AND sand/hail).
        // The sim's `onFieldResidual` (the whole order-8 handler, INCLUDING its unconditional
        // `|-weather|<W>|[upkeep]` line emission) is NOT gated on `effectiveWeather()` — only
        // the eachEvent('Weather') shuffle + the per-active chip are. So under Air Lock / Cloud
        // Nine the sim STILL emits the upkeep line; the port used to omit it entirely (the
        // handler was unscheduled), which let the order-10.5 leech `-damage` land where the sim
        // emits the order-8 sand `[upkeep]`. DRAW-NEUTRAL: the negater-suppressed shuffle is
        // gated OFF inside `apply_weather_chip` (`effective == false` → no `each_event_shuffle`),
        // so a negated sand/hail residual pushes the handler + emits the upkeep line but draws
        // NOTHING — identical draw count to the prior unscheduled behavior. The handler is alone
        // in its order-8 sort group (field speed 0), so it never ties any mon handler.
        let raw_weather = self.field.weather;
        if let Some(w) = raw_weather {
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
            // --- SCREENS (Reflect / Light Screen) as `onSideResidual` DURATION handlers
            //     (`gen3_screen_residual_tie_shuffle_v1`). Gathered PER SIDE, BEFORE this side's
            //     mon handlers — mirroring `fieldEvent('Residual')`, which concats
            //     `findSideEventHandlers(side, 'onSideResidual', 'duration')` BEFORE the per-active
            //     `findPokemonEventHandlers(active, 'onResidual', 'duration')`. And INDEPENDENT of
            //     the active mon being fainted (a side handler's effectHolder is the SIDE, not the
            //     active — so the gather sits above the `mon.fainted` continue below). speed 0 /
            //     priority 0; Reflect order 1, Light Screen order 2, subOrder 4. Only the SAME
            //     screen on the OTHER side ties (order 1↔1 / 2↔2, speed 0↔0), so the relative
            //     gather order of reflect vs lightscreen within a side is unobservable (distinct
            //     orders never tie). The apply decrements the counter and emits `|-sideend|` at 0. ---
            if self.sides[side].reflect > 0 {
                handlers.push(EventHandler {
                    order: REFLECT_RESIDUAL_ORDER,
                    priority: 0,
                    speed: 0.0,
                    sub_order: SIDE_CONDITION_SUBORDER,
                    effect_order: 0,
                    handler: ResidualAction::ScreenDuration { side, is_reflect: true },
                });
            }
            if self.sides[side].light_screen > 0 {
                handlers.push(EventHandler {
                    order: LIGHT_SCREEN_RESIDUAL_ORDER,
                    priority: 0,
                    speed: 0.0,
                    sub_order: SIDE_CONDITION_SUBORDER,
                    effect_order: 0,
                    handler: ResidualAction::ScreenDuration { side, is_reflect: false },
                });
            }

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

            // --- The CURSE volatile's residual chip (`gen3_move_coverage_batch3_v1`). Order
            //     10, subOrder 8 (the gen-3 `curse` condition's `onResidualOrder: 10,
            //     onResidualSubOrder: 8`) — so at order 10 it sorts AFTER Leftovers (4) / Leech
            //     Seed (5) / status DoT (6) but BEFORE Taunt (15). It is a VOLATILE, gathered
            //     with the volatiles group (after leech, before Taunt), mirroring
            //     `findPokemonEventHandlers`'s status→volatiles→item order. The cursed HOLDER
            //     is the effect holder (the mon carrying `curse`), so its speed is the sort
            //     key; its only tie is the OTHER mon's curse at equal speed. Its apply chips
            //     `floor(maxhp/4)`, DRAW-FREE. ---
            if mon.curse.is_some() {
                handlers.push(EventHandler {
                    order: STATUS_RESIDUAL_ORDER,
                    priority: 0,
                    speed,
                    sub_order: CURSE_RESIDUAL_SUBORDER,
                    effect_order: 0,
                    handler: ResidualAction::Curse { side, slot },
                });
            }

            // --- The PARTIAL-TRAP volatile's residual handler (`gen3_partial_trap_v1`).
            //     Order 10, subOrder 9 (the gen4-mod `partiallytrapped` override gen3
            //     inherits) — so at order 10 it sorts AFTER Leftovers (4) / Leech Seed (5) /
            //     status DoT (6) / Curse (8) and BEFORE Encore (14) / Taunt (15) / Yawn (19).
            //     It is a VOLATILE, gathered in the volatiles group (after curse, before
            //     encore — the port's insertion-order convention). subOrder 9 is UNIQUE among
            //     the order-10 handlers, so its only tie is the OTHER mon's partial trap at
            //     equal cached speed (a MUTUAL wrap → ONE extra tie-shuffle, probe I); the
            //     intra-mon gather position is therefore unobservable except through the
            //     selection sort's swaps. The apply does the duration tick, the trapper-gone
            //     release and the maxhp/16 chip (see `ResidualAction::PartialTrap`). ---
            if mon.partial_trap.is_some() {
                handlers.push(EventHandler {
                    order: STATUS_RESIDUAL_ORDER,
                    priority: 0,
                    speed,
                    sub_order: PARTIAL_TRAP_SUBORDER,
                    effect_order: 0,
                    handler: ResidualAction::PartialTrap { side, slot },
                });
            }

            // --- The ENCORE volatile's residual handler (`gen3_move_coverage_batch6_v1`).
            //     Order 10, subOrder 14 (the resolved gen-3 `encore` condition — one below
            //     Taunt's 15). Duration decrement + the onResidual 0-PP early removal ride
            //     the SAME visit. A VOLATILE, gathered in the volatiles group (after curse,
            //     before taunt — the insertion-order convention; its only tie is the other
            //     mon's encore at equal speed, so the intra-mon position is unobservable). ---
            if mon.encore.is_some() {
                handlers.push(EventHandler {
                    order: STATUS_RESIDUAL_ORDER,
                    priority: 0,
                    speed,
                    sub_order: ENCORE_RESIDUAL_SUBORDER,
                    effect_order: 0,
                    handler: ResidualAction::EncoreDuration { side, slot },
                });
            }

            // NOTE: the PERISH SONG counter's residual handler is gathered LATER — after
            // the NO_ORDER duration volatiles (protect/stall/flinch/…/two_turn) — so its
            // pre-shuffle GATHER position matches Showdown's `pokemon.volatiles` INSERTION
            // order (`gen3_perish_start_volatile_insertion_order_v1`). See the block below.

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

            // --- The YAWN volatile's duration handler (`gen3_yawn_v1`). Order 10, subOrder 19
            //     (the gen-3 `yawn` condition's `onResidualOrder: 10, onResidualSubOrder: 19`,
            //     VERIFIED vs the resolved dex) — so at order 10 it sorts AFTER Leftovers (4) /
            //     Leech (5) / status DoT (6) / Curse (8) / Encore (14) / Taunt (15). It is a
            //     VOLATILE, gathered in the volatiles group (after taunt). Unlike the pure
            //     duration handlers it has a REAL `onEnd` effect: at 1 → 0 it emits `-end [silent]`
            //     + `trySetStatus('slp')` (the sleep `random(2,6)` at RESOLVE — the draw crux). Its
            //     ONLY tie is the OTHER mon's yawn at equal speed (subOrder 19 is unique among
            //     order-10 handlers), so its gather position vs the other order-10 volatiles is
            //     unobservable; a yawn MIRROR at equal speed draws one handler-sort tie-shuffle. ---
            if mon.yawn.is_some() {
                handlers.push(EventHandler {
                    order: STATUS_RESIDUAL_ORDER,
                    priority: 0,
                    speed,
                    sub_order: YAWN_RESIDUAL_SUBORDER,
                    effect_order: 0,
                    handler: ResidualAction::Yawn { side, slot },
                });
            }

            // --- PERISH, gathered EARLY (`gen3_perish_volatile_insertion_turn_v1`).
            //     Showdown gathers a mon's volatile handlers in `pokemon.volatiles` INSERTION
            //     order, and `speed_sort` is a NON-STABLE selection sort whose swaps hand the
            //     tie-group Fisher-Yates shuffle whatever pre-sort order it finds — so the
            //     relative order of this mon's `perish` and its NO_ORDER duration volatiles is
            //     LOAD-BEARING for the emitted `-start|<mon>|perish<n>` order (and the faint
            //     order that follows it).
            //
            //     A single FIXED position cannot serve both real boards:
            //       * SAME-TURN perish-after-protect (`rmrr03rmc_ab_453_2`, corpus fixture 32):
            //         Protect is priority 3, so it moves FIRST and its `stall` volatile is
            //         inserted BEFORE `perish` → perish must gather AFTER the NO_ORDER block.
            //       * CROSS-TURN perish-then-protect (soak3 `sbd_msb1zfxs_b134`): Perish Song
            //         lands on turn 16 and the mon Protects on turn 18, so `perish` is inserted
            //         FIRST and must gather BEFORE the NO_ORDER block. The port's fixed position
            //         got this backwards, reversing the tied perish pair and emitting
            //         `|-start|p1a: Misdreavus|perish0` where the sim emits p2a first.
            //     Comparing the INSERTION TURNS separates them exactly: an earlier perish turn
            //     gathers first; equal turns keep the same-turn ordering above (within a turn the
            //     higher-priority Protect always inserts first). DRAW-FREE — the tie-group SIZE
            //     and draw COUNT are unchanged; only the pre-shuffle PERMUTATION moves. ---
            let perish_gathers_early = mon.perish.is_some()
                && mon.noorder_vol_turn != 0
                && mon.perish_turn < mon.noorder_vol_turn;
            if perish_gathers_early {
            handlers.push(EventHandler {
                    order: PERISH_RESIDUAL_ORDER,
                    priority: 0,
                    speed,
                    sub_order: VOLATILE_RESIDUAL_SUBORDER,
                    effect_order: 0,
                    handler: ResidualAction::Perish { side, slot },
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
            // --- The ENDURE `duration: 1` volatile (`gen3_move_coverage_batch6_v1`)
            //     registers a NO_ORDER/subOrder-2 residual DURATION handler exactly like
            //     `protect` (its sibling above). THE CRUX (probe ED1/ED2/ED9): a
            //     SUCCESSFUL endure turn holds BOTH `endure` (duration 1) + `stall`
            //     (duration 2) — TWO tied handlers on the SAME mon → an INTRA-mon
            //     tie-group shuffle draws ONE `random(0,2)` at ANY speed configuration
            //     (the protect+stall pair class); a FAILED-roll turn has no endure
            //     volatile → no tie → no shuffle (ED2 t4). NO HP effect (the survive-at-1
            //     clamp is the onDamage; the clear is the turn-top `clear_flinch`); the
            //     apply is a no-op `VolatileDuration{is_stall:false}`. The tie
            //     PERMUTATION among the no-op handlers is unobservable, so the gather
            //     position vs `stall` below only needs the pair to EXIST. ---
            if mon.endure {
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

            // --- FOCUS PUNCH + PURSUIT `duration: 1` volatiles (`gen3_move_coverage_batch4_v1`)
            //     each register a NO_ORDER/subOrder-2 residual DURATION handler (gathered by
            //     `findPokemonEventHandlers(..., 'duration')`, same tie-group as
            //     protect/stall/flinch/disable). NO HP effect (the actual clear is the turn-top
            //     `clear_flinch` reset) — a no-op `VolatileDuration{is_stall:false}` apply — but
            //     their PRESENCE changes the residual tie-shuffle COUNT: a FOCUS PUNCH MIRROR at
            //     equal speed adds ONE tie-shuffle draw (VERIFIED: the bulky both-FP mirror's
            //     +1 residual shuffle), and a normal Pursuit whose foe STAYS in leaves the
            //     pursuit volatile up through the residual. An FP user can't ALSO protect/flinch
            //     (FP blocks flinch; it's a damaging move, not Protect), so the same-mon gather
            //     position is unobservable — the only tie is the OTHER mon's same volatile. ---
            if mon.focus_punch.is_some() {
                handlers.push(EventHandler {
                    order: NO_ORDER,
                    priority: 0,
                    speed,
                    sub_order: VOLATILE_RESIDUAL_SUBORDER,
                    effect_order: 0,
                    handler: ResidualAction::VolatileDuration { side, slot, is_stall: false },
                });
            }
            if mon.pursuit.is_some() {
                handlers.push(EventHandler {
                    order: NO_ORDER,
                    priority: 0,
                    speed,
                    sub_order: VOLATILE_RESIDUAL_SUBORDER,
                    effect_order: 0,
                    handler: ResidualAction::VolatileDuration { side, slot, is_stall: false },
                });
            }
            // --- COUNTER / MIRROR COAT's reactive `duration: 1` volatile
            //     (`gen3_move_coverage_batch5_v1`) registers the SAME NO_ORDER/subOrder-2
            //     residual DURATION handler as focus-punch/pursuit. NO HP effect (the
            //     clear is the turn-top `clear_flinch`), but its PRESENCE changes the
            //     residual tie-shuffle COUNT: a COUNTER MIRROR at equal speed adds ONE
            //     tie-shuffle draw (the probed CM +1 residual-duration tie — part of the
            //     +4 delta vs the both-splash control). A counter user can't ALSO
            //     protect/flinch this turn, so the same-mon gather position is
            //     unobservable. ---
            if mon.reactive.is_some() {
                handlers.push(EventHandler {
                    order: NO_ORDER,
                    priority: 0,
                    speed,
                    sub_order: VOLATILE_RESIDUAL_SUBORDER,
                    effect_order: 0,
                    handler: ResidualAction::VolatileDuration { side, slot, is_stall: false },
                });
            }
            // --- BEAT UP's `beatup` `duration: 1` volatile (`gen3_move_coverage_batch4b_v1`)
            //     registers the SAME NO_ORDER/subOrder-2 residual DURATION handler as
            //     focus-punch/pursuit/protect/stall/flinch. NO HP effect (the clear is the
            //     turn-top `clear_flinch`), but its PRESENCE changes the residual tie-shuffle
            //     COUNT: a BEAT UP MIRROR at equal speed adds ONE tie-shuffle draw (the e2e_217
            //     desync — both Charizards' beatup volatiles tie). A Beat Up user can't ALSO
            //     protect/flinch, so the same-mon gather position is unobservable. ---
            if mon.beat_up {
                handlers.push(EventHandler {
                    order: NO_ORDER,
                    priority: 0,
                    speed,
                    sub_order: VOLATILE_RESIDUAL_SUBORDER,
                    effect_order: 0,
                    handler: ResidualAction::VolatileDuration { side, slot, is_stall: false },
                });
            }
            // --- SNATCH's `snatch` `duration: 1` volatile (`gen3_snatch_v1`) registers the
            //     SAME NO_ORDER/subOrder-2 residual DURATION handler as
            //     focus-punch/beat-up/endure. NO HP effect (the clear is the turn-top
            //     `clear_flinch`), but its PRESENCE changes the residual tie-shuffle COUNT:
            //     a SNATCH MIRROR at equal speed adds ONE tie-shuffle draw (PROBE-VERIFIED:
            //     both-Snatch draws 8 vs the both-Splash control's 7 — the extra is the
            //     residual `fieldEvent` handler-sort shuffle). A snatcher can't ALSO
            //     protect/flinch this turn (it spent its +4 action on Snatch), so the
            //     same-mon gather position is unobservable — the only tie is the OTHER
            //     mon's same volatile. ---
            if mon.snatch {
                handlers.push(EventHandler {
                    order: NO_ORDER,
                    priority: 0,
                    speed,
                    sub_order: VOLATILE_RESIDUAL_SUBORDER,
                    effect_order: 0,
                    handler: ResidualAction::VolatileDuration { side, slot, is_stall: false },
                });
            }
            // --- MUSTRECHARGE's `duration: 2` volatile (`gen3_move_coverage_batch4c_v1`,
            //     Hyper Beam) registers a NO_ORDER/subOrder-2 residual DURATION handler on
            //     the CAST turn's residual (the volatile is removed at the NEXT turn's
            //     recharge cant, BEFORE that turn's residual — so only the cast-turn
            //     residual gathers it). NO HP effect (the 2 → 1 decrement is unobservable —
            //     the volatile never expires by duration), DRAW-FREE apply; its PRESENCE
            //     changes the residual tie-shuffle COUNT (an HB MIRROR at equal speed adds
            //     one tie draw). Its `duration: 2` NON-ending decrement means it must FALL
            //     THROUGH to the per-handler faintMessages (NOT the duration:1 group's blanket
            //     `continue`) — hence its OWN `MustRechargeDuration` variant, D4 fix. ---
            if mon.must_recharge {
                handlers.push(EventHandler {
                    order: NO_ORDER,
                    priority: 0,
                    speed,
                    sub_order: VOLATILE_RESIDUAL_SUBORDER,
                    effect_order: 0,
                    handler: ResidualAction::MustRechargeDuration { side, slot },
                });
            }
            // --- TWOTURNMOVE's `duration: 2` volatile (`gen3_move_coverage_batch4c_v1`,
            //     Solar Beam) — a REAL duration countdown (2 → 1 on the charge-turn
            //     residual, 1 → 0 on the fire-turn residual → removed). Registers on BOTH
            //     residuals (probe), the same NO_ORDER/subOrder-2 tie group. ---
            if mon.two_turn.is_some() {
                handlers.push(EventHandler {
                    order: NO_ORDER,
                    priority: 0,
                    speed,
                    sub_order: VOLATILE_RESIDUAL_SUBORDER,
                    effect_order: 0,
                    handler: ResidualAction::TwoTurnDuration { side, slot },
                });
            }

            // --- The PERISH SONG counter's residual handler (`gen3_move_coverage_batch6_v1`).
            //     Order **12** — its OWN group, LAST in the modeled ladder (after the
            //     order-10 mon handlers + the order-11 futuremove, before Truant's 27 and
            //     the NO_ORDER duration volatiles by sort key). subOrder = the Condition
            //     effectTypeOrder default 2 (no explicit onResidualSubOrder). A VOLATILE
            //     on the holder; its only tie is the OTHER mon's perish counter at equal
            //     cached speed (the P5 mirror: ONE `random(0,2)` per residual — the
            //     decrement + the `-start perish<d>` print ride ONE handler visit, so a
            //     perished pair is a size-2 tie group, NOT two).
            //
            //     GATHER POSITION (`gen3_perish_start_volatile_insertion_order_v1`, the
            //     Stage-B byte-fuzz find rmrr03rmc_ab_453_2): the `speed_sort` is a NON-STABLE
            //     selection sort, so the tie-group Fisher-Yates shuffle permutes the two
            //     equal-speed perish handlers in whatever PRE-SORT order the selection swaps
            //     left them in — which depends on their GATHER index. Showdown gathers a mon's
            //     volatile handlers in `pokemon.volatiles` INSERTION order (`findPokemonEventHandlers`
            //     iterates the volatiles object). A perishing mon's `perish` volatile is
            //     inserted (by the FOE's Perish Song, priority 0) AFTER any `protect`/`stall`
            //     it added this turn (Protect is priority 3 → moves first → its volatile is
            //     inserted first), so `perish` sorts AFTER the NO_ORDER duration volatiles in
            //     the mon's gather. Gathering perish HERE — after the whole NO_ORDER
            //     duration-volatile block — reproduces that for the reproduced (same-turn
            //     perish-after-protect) board: the repro flips to 168/168 byte-clean with NO
            //     seed divergence (draw-free — the tie-group SIZE + draw COUNT are unchanged;
            //     only the pre-shuffle PERMUTATION of the tied perish pair moves). HONEST
            //     LIMITATION: a mon perished on an EARLIER turn than it Protects would have
            //     `perish` inserted BEFORE `protect`; that distinct (unreproduced) board is
            //     not modeled by this fixed position — a full per-volatile insertion-sequence
            //     stamp would be the general fix. Emission-only + draw-free + state-free. ---
            if mon.perish.is_some() && !perish_gathers_early {
                handlers.push(EventHandler {
                    order: PERISH_RESIDUAL_ORDER,
                    priority: 0,
                    speed,
                    sub_order: VOLATILE_RESIDUAL_SUBORDER,
                    effect_order: 0,
                    handler: ResidualAction::Perish { side, slot },
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

            // WHITE HERB (`gen3_white_herb_v1`): order **29** (the LAST residual, its own group
            // after Truant 27). Gathered UNCONDITIONALLY for an active White Herb holder (the item
            // registers its onResidual regardless of the boost state — the `boosts<0` check lives in
            // the apply), so it participates in the residual tie-shuffle. This is the site that
            // restores an Intimidate-switch-in drop on a turn the holder does NOT move (a
            // forced-replacement / no-move turn — the omniscient byte-fuzz find ab_7_4). subOrder is
            // unset in the sim (undefined); its only tie is a two-White-Herb-holder equal-speed board.
            if dex.item(&to_id(&mon.item)).map_or(false, |i| i.boost_restore) {
                handlers.push(EventHandler {
                    order: WHITE_HERB_RESIDUAL_ORDER,
                    priority: 0,
                    speed,
                    sub_order: 0,
                    effect_order: 0,
                    handler: ResidualAction::WhiteHerb { side, slot },
                });
            }

            // --- THE SIDE'S SLOT CONDITIONS (`findSideEventHandlers(side, 'onResidual',
            //     …, active)`), gathered PER-ACTIVE AFTER that active's pokemon handlers
            //     (`gen3_leftovers_slotcond_gather_order_v1`, the R2 fix — see the
            //     pre-sort-order note above the weather handler). WISH (order 7, resolves
            //     BEFORE the sand chip + all order-10 handlers) + FUTUREMOVE (order 11,
            //     AFTER every order-10 mon handler). Their `speed` is the slot's CURRENT
            //     active mon's cached speed (the effectHolder is the slot occupant), so two
            //     Wishes (or two FutureMoves) resolving the same turn at EQUAL speed TIE at
            //     their order → ONE tie-shuffle draw; distinct speed draws none. Gathered
            //     whenever PENDING (the duration decrement + fire-at-0 resolve live in the
            //     apply). No subOrder (unset → effectTypeOrder default; each order is unique
            //     among the modeled residuals). Sitting AFTER the item here reproduces the
            //     sim's selection-sort pre-shuffle order so a tied Leftovers pair emits its
            //     two `-heal` lines in the SAME order as Showdown. ---
            if self.sides[side].wish_pending.is_some() {
                handlers.push(EventHandler {
                    order: WISH_RESIDUAL_ORDER,
                    priority: 0,
                    speed,
                    sub_order: 0,
                    effect_order: 0,
                    handler: ResidualAction::Wish { side },
                });
            }
            if self.sides[side].future_move.is_some() {
                handlers.push(EventHandler {
                    order: FUTURE_RESIDUAL_ORDER,
                    priority: 0,
                    speed,
                    sub_order: 0,
                    effect_order: 0,
                    handler: ResidualAction::FutureMove { side },
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
                ResidualAction::WhiteHerb { side, slot } => {
                    // `whiteherb.onResidual` → its `onStart` (restore negatives + consume, emitting
                    // `-enditem`/`-clearnegativeboost`; a no-op when no boost is negative). DRAW-FREE.
                    // The fainted guard mirrors `if (handler.effectHolder.fainted) continue`.
                    if self.sides[side].pokemon[slot].fainted {
                        continue;
                    }
                    self.white_herb_restore(side, slot, dex);
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
                    // THE DURATION-END `continue` (`gen3_perside_residual_faint_upkeep_order_v1`,
                    // D4): the sim's `fieldEvent('Residual')` SKIPS the per-handler
                    // `faintMessages()` for a duration handler that hits 0 this turn (the
                    // `handler.state.duration-- → end() → continue` branch). So a NO_ORDER
                    // duration tick that ENDS must not drain a faint an EARLIER handler
                    // ENQUEUED-but-deferred (the order-12 Perish `continue`) — else the port
                    // emits that mon's `|faint|` BEFORE `|upkeep|`, where the sim defers it to
                    // AFTER `|upkeep|` (the tail `process_faints`). The is_stall volatile is
                    // duration:2 (may not end); the non-stall duration:1 volatiles
                    // (protect/flinch/focuspunch/beatup/endure/snatch) ALWAYS end on their
                    // residual → always `continue`. A non-ending decrement falls through to the
                    // per-handler faintMessages (matching the sim's non-end path).
                    if is_stall {
                        let mon = &mut self.sides[side].pokemon[slot];
                        mon.stall_duration = mon.stall_duration.saturating_sub(1);
                        if mon.stall_duration == 0 {
                            mon.protect_counter = 0; // the volatile ended → reset
                            continue; // duration-END → skip the per-handler faintMessages
                        }
                    } else {
                        continue; // a duration:1 volatile always ends → skip faintMessages
                    }
                }
                // SCREEN duration tick (`gen3_screen_residual_tie_shuffle_v1`): the Reflect /
                // Light Screen SIDE condition. NO HP effect, DRAW-FREE apply. Mirror
                // `fieldEvent('Residual')`'s `handler.state.duration--; if (!duration) end()`
                // for a duration-only handler (callback undefined): decrement the counter; on
                // reaching 0 remove the side condition → emit `|-sideend|<side>|<Effect>` (the
                // sim's `removeSideCondition`). The effectHolder is the SIDE, so a fainted active
                // NEVER skips it (no per-mon fainted guard). On the duration-END emit, `continue`
                // (skip the per-handler faintMessages — the D4 order fix; a screen can't faint
                // anything and, being order 1/2, runs FIRST so nothing is deferred yet — but keep
                // the sim's END-branch `continue` exact). A non-ending decrement falls through to
                // the per-handler process_faints (a no-op for a screen).
                ResidualAction::ScreenDuration { side, is_reflect } => {
                    let ptr = if is_reflect {
                        &mut self.sides[side].reflect
                    } else {
                        &mut self.sides[side].light_screen
                    };
                    // Gathered only when > 0, but guard anyway (a prior handler never touches it).
                    if *ptr == 0 {
                        continue;
                    }
                    *ptr -= 1;
                    if *ptr == 0 {
                        // [EMIT] `|-sideend|<side>|Reflect` / `|…|move: Light Screen`.
                        if self.logging() {
                            let side_ref = crate::protocol::ProtocolBuilder::side_ref(
                                side,
                                &self.sides[side].name,
                            );
                            let effect = if is_reflect { "Reflect" } else { "move: Light Screen" };
                            self.log.sideend(&side_ref, effect);
                        }
                        continue; // duration-END → skip faintMessages (D4 order fix)
                    }
                }
                // MUSTRECHARGE duration handler on the Hyper Beam CAST turn
                // (`gen3_perside_residual_faint_upkeep_order_v1` D4 fix). `mustrecharge` is
                // `duration: 2`, so its only residual tick decrements 2 → 1 (NON-zero → the sim's
                // `fieldEvent` does NOT take the `duration-- == 0` end/`continue` branch). So —
                // UNLIKE the duration:1 group above — this arm FALLS THROUGH to the per-handler
                // `process_faints`, draining any faint an earlier order-≤12 handler (Perish)
                // enqueued-but-deferred (the sim emits that `|faint|` BEFORE `|upkeep|`; the old
                // blanket `continue` mis-deferred it PAST `|upkeep|`). No-op apply; a fainted
                // holder is still skipped (mirroring `fieldEvent`'s pre-duration fainted guard).
                ResidualAction::MustRechargeDuration { side, slot } => {
                    if self.sides[side].pokemon[slot].fainted {
                        continue;
                    }
                    // no `continue` — fall through to the per-handler faintMessages below
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
                            // duration-END `continue` (`gen3_perside_residual_faint_upkeep_order_v1`):
                            // skip the per-handler faintMessages so a deferred earlier faint
                            // (Perish order 12) is not drained before `|upkeep`.
                            continue;
                        } else {
                            mon.taunt = Some(next);
                        }
                    }
                }
                // YAWN duration tick + RESOLVE (`gen3_yawn_v1`, order 10 subOrder 19): decrement
                // the yawn duration; at 1 → 0 fire the `onEnd` — `|-end|<target>|move: Yawn|
                // [silent]` then `trySetStatus('slp', source)` routed through the EXISTING status
                // path (so the sleep `random(2,6)` onStart draw, the gen3ou Sleep Clause block, AND
                // the gen3ou SetStatus 2-clause shuffle all come for free + bit-for-bit). The 2 → 1
                // (cast-turn) tick is a draw-free no-op. If the target got statused between cast +
                // resolve (or is now sleep-immune / Sleep-Clause-blocked), the `-end [silent]` still
                // fires but no sleep sets (`try_set_status` no-ops draw-appropriately).
                ResidualAction::Yawn { side, slot } => {
                    if self.sides[side].pokemon[slot].fainted {
                        continue;
                    }
                    let ended = {
                        let mon = &mut self.sides[side].pokemon[slot];
                        match mon.yawn {
                            Some((dur, src)) => {
                                let next = dur.saturating_sub(1);
                                if next == 0 {
                                    mon.yawn = None; // the volatile expired (onEnd fires below)
                                    Some(src)
                                } else {
                                    mon.yawn = Some((next, src));
                                    None
                                }
                            }
                            None => None,
                        }
                    };
                    if ended.is_some() {
                        // [EMIT] `|-end|<target>|move: Yawn|[silent]` (the yawn `onEnd`).
                        if self.logging() {
                            let m = self.mon_ref(side, slot, dex);
                            self.log.volatile_end_silent(&m, "move: Yawn");
                        }
                        // `target.trySetStatus('slp', this.effectState.source)`. In gen-3 singles
                        // the source side is always the OPPOSITE of the yawn holder; the exact
                        // source slot never affects a `slp` apply (only the SIDE matters, for the
                        // Sleep Clause self-exemption), so pass that side's CURRENT active. The
                        // resolve emits a PLAIN `|-status|<target>|slp` (the yawn condition is not a
                        // Move, so `src_move` is `None` → the plain `-status` branch).
                        let src_side = 1 - side;
                        let src_slot = self.sides[src_side].active;
                        self.try_set_status(side, slot, "slp", Some((src_side, src_slot)), dex);
                        // duration-END → skip the per-handler faintMessages, mirroring the sim's
                        // `fieldEvent('Residual')` `handler.state.duration-- → end() → continue`
                        // (the D4 order fix). Setting sleep cannot faint, so this only preserves the
                        // deferred-faint ORDER exactly (a Perish [order 12] faint stays deferred past
                        // `|upkeep`). A non-ending (2 → 1) tick falls through to faintMessages.
                        continue;
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
                            continue; // duration-END → skip faintMessages (D4 order fix)
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
                // CURSE chip (`gen3_move_coverage_batch3_v1`, order 10 subOrder 8): the cursed
                // holder loses floor(maxhp/4). DRAW-FREE.
                ResidualAction::Curse { side, slot } => {
                    if self.sides[side].pokemon[slot].fainted {
                        continue;
                    }
                    self.apply_curse(side, slot, dex);
                }
                // PARTIAL TRAP (`gen3_partial_trap_v1`, order 10 subOrder 9): the duration tick,
                // the trapper-gone silent release, then the floor(maxhp/16) chip. The
                // duration-END branch `continue`s (the D4 skip-faintMessages rule for a
                // `duration-- == 0` handler); the other two branches FALL THROUGH so a chip KO
                // drains through the per-handler faintMessages exactly like the sim's
                // `singleEvent(onResidual) → faintMessages(true)`.
                ResidualAction::PartialTrap { side, slot } => {
                    if self.sides[side].pokemon[slot].fainted {
                        continue;
                    }
                    if self.apply_partial_trap(side, slot, dex) {
                        continue; // duration-END → skip the per-handler faintMessages
                    }
                }
                // WISH delayed heal (`gen3_move_coverage_batch3_v1`, order 7): decrement the
                // pending duration; fire the heal at 0. DRAW-FREE apply.
                ResidualAction::Wish { side } => {
                    // The Wish's effectHolder is the slot (not a specific mon); a fainted
                    // active does NOT skip the DECREMENT — but the heal is gated on
                    // `!target.fainted` inside apply_wish. We still process it (the slot
                    // condition ticks regardless).
                    self.apply_wish(side, dex);
                }
                // TWOTURNMOVE duration tick (`gen3_move_coverage_batch4c_v1`, Solar Beam):
                // decrement + remove at 0. NO HP effect, DRAW-FREE.
                ResidualAction::TwoTurnDuration { side, slot } => {
                    if self.sides[side].pokemon[slot].fainted {
                        continue;
                    }
                    let mon = &mut self.sides[side].pokemon[slot];
                    if let Some(t) = mon.two_turn.as_mut() {
                        t.duration = t.duration.saturating_sub(1);
                        if t.duration == 0 {
                            mon.two_turn = None; // the volatile expired (onEnd)
                            continue; // duration-END → skip faintMessages (D4 order fix)
                        }
                    }
                }
                // FUTUREMOVE tick / resolve (`gen3_move_coverage_batch4c_v1`): like Wish,
                // the effectHolder is the SLOT — a fainted occupant does NOT skip the
                // decrement; the fainted-skip gate lives inside `apply_future_move` (the
                // sim's onEnd skips the STRIKE, with the condition consumed either way).
                ResidualAction::FutureMove { side } => {
                    self.apply_future_move(side, dex);
                }
                // ENCORE duration tick (`gen3_move_coverage_batch6_v1`): the generic
                // decrement (`-end` at 0) THEN the `onResidual` 0-PP check (the encored
                // slot at 0 PP removes the volatile EARLY — probe EN5: same-turn `-end`
                // even at duration 5). NO HP effect, DRAW-FREE.
                ResidualAction::EncoreDuration { side, slot } => {
                    if self.sides[side].pokemon[slot].fainted {
                        continue;
                    }
                    let mut ended = false;
                    {
                        let mon = &mut self.sides[side].pokemon[slot];
                        if let Some((eslot, turns)) = mon.encore {
                            let next = turns.saturating_sub(1);
                            if next == 0 {
                                mon.encore = None; // the duration expired (onEnd)
                                ended = true;
                            } else if mon.move_pp.get(eslot).copied().unwrap_or(0) == 0 {
                                // encore.onResidual: the encored slot is out of PP →
                                // removeVolatile NOW (the early `-end`).
                                mon.encore = None;
                                ended = true;
                            } else {
                                mon.encore = Some((eslot, next));
                            }
                        }
                    }
                    // [EMIT] `|-end|<mon>|Encore` (`encore.onEnd`) — both removal paths.
                    if ended {
                        if self.logging() {
                            let m = self.mon_ref(side, slot, dex);
                            self.log.volatile_end(&m, "Encore");
                        }
                        continue; // duration-END → skip faintMessages (D4 order fix)
                    }
                }
                // PERISH SONG tick (`gen3_move_coverage_batch6_v1`): decrement, then
                // `|-start|<mon>|perish<d>`; at 1 → 0 the `onEnd` fires `perish0` + the
                // holder FAINTS (hp → 0 draw-free; the caller's per-handler
                // faintMessages sets `fainted` + a game-ending KO aborts the rest — a
                // mutual perish-out is a same-residual double faint → double
                // replacement, NO Quick Claw on the faint turn).
                ResidualAction::Perish { side, slot } => {
                    if self.sides[side].pokemon[slot].fainted {
                        continue;
                    }
                    let next = {
                        let mon = &mut self.sides[side].pokemon[slot];
                        match mon.perish {
                            Some(d) => {
                                let n = d.saturating_sub(1);
                                if n == 0 {
                                    mon.perish = None; // onEnd (the faint below)
                                } else {
                                    mon.perish = Some(n);
                                }
                                n
                            }
                            None => continue,
                        }
                    };
                    if next == 0 {
                        // [EMIT] `|-start|<mon>|perish0` then the faint (onEnd).
                        if self.logging() {
                            let m = self.mon_ref(side, slot, dex);
                            self.log.volatile_start(&m, "perish0");
                        }
                        let hp = self.sides[side].pokemon[slot].hp;
                        self.apply_damage(side, slot, hp); // zero + record the faint order
                        // THE DURATION-END `continue` (probe MC89 — the mutual
                        // perish-out TIE): the sim's `fieldEvent` duration-END branch
                        // (`handler.state.duration-- → end() → continue`) SKIPS the
                        // per-handler `faintMessages()` — the perish faint is only
                        // ENQUEUED here, so a speed-tied MIRROR runs BOTH perish
                        // handlers back-to-back (both `perish0` lines print before
                        // EITHER `|faint|`) and the double faint processes at the
                        // residual's tail → both LAST mons → the gen-3 TIE. Running
                        // the per-handler faintMessages here (the onResidual-callback
                        // convention — burn DoT etc.) would end the battle after the
                        // FIRST perish faint with a WRONG winner. Perish is order 12
                        // (LAST in the ladder), so no later handler is skipped.
                        continue;
                    } else if self.logging() {
                        // [EMIT] `|-start|<mon>|perish<duration>` (the onResidual print,
                        // post-decrement — the cast turn's residual prints perish3).
                        let m = self.mon_ref(side, slot, dex);
                        self.log.volatile_start(&m, &format!("perish{next}"));
                    }
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
        // RM3 (`gen3_sand_upkeep_under_air_lock_v1`): is THIS weather effective RIGHT NOW (no
        // Air Lock / Cloud Nine active)? The upkeep-line / `-weather none` emission is
        // UNCONDITIONAL (mirroring the sim's un-gated `onFieldResidual`), but the
        // eachEvent('Weather') shuffle + the per-active chip fire ONLY when effective for
        // sand/hail (sun/rain always run the shuffle — their `onFieldResidual` isWeather guard
        // reads effectiveWeather() but the eachEvent is unconditional; they chip nothing).
        let effective = self.effective_weather(dex) == Some(weather);
        // --- TIMED-WEATHER (`gen3_move_coverage_batch2_v1`) countdown + expiry. A MOVE-set
        //     weather (Rain Dance / Sunny Day) carries `weather_turns` 1..=5; the ability
        //     weather (Sand Stream) is PERMANENT (`weather_turns == 0` — never expires). At
        //     the field residual: `weather_turns == 1` → EXPIRE this turn (emit `|-weather|
        //     none` INSTEAD of the `[upkeep]` line, clear the weather), and STILL run the
        //     eachEvent('Weather') shuffle below (VERIFIED: the expiry turn draws the SAME
        //     count as an upkeep turn — the field residual's shuffle fires either way).
        //     `weather_turns > 1` → decrement + upkeep. `weather_turns == 0` → permanent, no
        //     decrement. The decrement is DRAW-FREE + state-only. ---
        let expiring = self.field.weather_turns == 1;
        if expiring {
            self.field.weather = None;
            self.field.weather_turns = 0;
            // [EMIT] `|-weather|none` (the weather condition's `onFieldEnd`).
            if self.logging() {
                self.log.weather("none", None, None, false);
            }
            // The field residual STILL fires its eachEvent('Weather') shuffle on the expiry
            // turn (probe: the expiry turn draws the same count as an upkeep turn). No chip
            // (the weather is gone; even a would-be sand/hail chip is skipped — the shuffle
            // is the whole residual). RM3: the shuffle is gated the same as the main path —
            // sun/rain always run it, sand/hail only when effective (a negater suppresses it).
            if matches!(weather, Weather::Sun | Weather::Rain) || effective {
                self.each_event_shuffle();
            }
            return;
        }
        if self.field.weather_turns > 1 {
            self.field.weather_turns -= 1;
        }
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
        // Fired for ALL EFFECTIVE weather (sun/rain INCLUDED — `gen3_ability_batch1_v1`): the
        // `eachEvent('Weather')` shuffle draws on a tie regardless of whether the weather chips.
        // RM3: under Air Lock / Cloud Nine a sand/hail residual STILL emits its upkeep line
        // (above, unconditional) but draws NOTHING — the eachEvent shuffle is gated OFF, so the
        // negated residual is draw-neutral vs the prior unscheduled behavior.
        let run_eachevent = matches!(weather, Weather::Sun | Weather::Rain) || effective;
        let shuffled = if run_eachevent {
            self.each_event_shuffle()
        } else {
            Vec::new()
        };

        // Sun / Rain have NO chip (`onWeather` has no damage handler) — the shuffle above IS the
        // whole field-residual for them. Return before the per-active chip walk (so the
        // Sand/Hail-only chip code below is unreachable for sun/rain by CONSTRUCTION, not by the
        // per-mon `weather_immune` short-circuit).
        if matches!(weather, Weather::Rain | Weather::Sun) {
            return;
        }

        // RM3: a NEGATED sand/hail residual (Air Lock / Cloud Nine active) emitted its upkeep
        // line but must chip NOTHING and draw NOTHING (the shuffle above was suppressed).
        if !effective {
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
            let chip = self.focus_band_damage(side, slot, chip, false, false, dex);
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
        // [EMIT] the ability ANNOUNCE `|-ability|<mon>|Speed Boost|boost` THEN `|-boost|<mon>|
        // spe|1` — only when the stage actually rose (a +6-capped boost draws NO `-boost` in
        // the sim's `boost()`, so the ability announce is skipped too; the clamped-delta-sign
        // convention used elsewhere). The `-ability|…|boost` line is the sim's `boost()`
        // ability-source announce (SIM-PROBED: `|-ability|p1a: Yanma|Speed Boost|boost`
        // precedes the `-boost`); the prior port omitted it (`gen3_omniscient_byte_fuzz_v1`).
        if self.logging() && self.sides[side].pokemon[slot].boosts[4] != before {
            let mon_ref = self.mon_ref(side, slot, dex);
            self.log.ability(&mon_ref, "Speed Boost", Some("boost"));
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
    pub(crate) fn apply_heal(&mut self, side: usize, slot: usize, amount: u16) -> bool {
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

    /// RECOIL (`gen3_move_coverage_batch1_v1`, the DAMAGING-move recoil family — Double-Edge
    /// `recoil:[1,3]`, Take Down / Submission `[1,4]`): the USER takes
    /// `max(floor(damageDealt · num/den), 1)` HP AFTER a landed hit — the exact gen3
    /// `calcRecoilDamage` = `clampIntRange(floor(dmg·num/den), 1)`. DRAW-FREE (`this.damage`
    /// consumes no PRNG), and it fires whether the damage hit the MON or a SUBSTITUTE (gen-3
    /// `substitute.onTryPrimaryHit` runs the SAME `calcRecoilDamage` on the sub damage), so
    /// `dealt` is the actual damage dealt to whatever absorbed it. **Rock Head** negates
    /// recoil (`rockhead.onDamage` returns null for a `recoil` effect) → a no-op here.
    /// `dealt > 0` is the `move.totalDamage` gate. Emitted as `|-damage|<user>|<HP>|[from]
    /// Recoil|[of] <target>` via [`crate::protocol::ProtocolBuilder::damage_of`]. Struggle's
    /// recoil is a SEPARATE gen3 path (`[1,4]` with its own `damage_of` line, already modeled).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_recoil(
        &mut self,
        side: usize,
        slot: usize,
        target_side: usize,
        target_slot: usize,
        recoil_fraction: f64,
        dealt: u16,
        dex: &Dex,
    ) {
        if recoil_fraction <= 0.0 || dealt == 0 {
            return;
        }
        // gen3 `battle.damage`→`spreadDamage` early-returns 0 with NO `-damage` emit when the
        // recoiling USER is ALREADY at 0 HP (`if (!target || !target.hp) { retVals=0; continue }`,
        // battle.js:1727). A CONTACT recoil ability — Rough Skin — fires INSIDE moveHit's
        // `runEvent('DamagingHit')` (BEFORE this batch-1 recoil at gen3 scripts.ts:460) and can
        // faint the attacker to 0 HP (deferred faint, `hp == 0` but not yet `fainted`); the sim
        // then SKIPS the move's own recoil entirely. The old code computed `recoil.min(hp==0)==0`
        // → `apply_damage(0)` no-op but STILL emitted a spurious `|-damage|<user>|0 fnt|[from]
        // Recoil`. (repro rmrz81mki_ab_47_23)
        if self.sides[side].pokemon[slot].hp == 0 {
            return;
        }
        // ROCK HEAD negates recoil (the `onDamage` returns null for a `recoil` effect —
        // draw-free no-op). Read the CURRENT ability (Trace-aware).
        if to_id(&self.sides[side].pokemon[slot].ability) == "rockhead" {
            return;
        }
        // gen3 `calcRecoilDamage`: clampIntRange(floor(dmg·num/den), 1). Compute via INTEGER
        // rational math (recovered from the fraction — [1,3]→(1,3), [1,4]→(1,4)) so 1/3 of 120
        // is EXACTLY floor(40) == 40 (a float `120·0.333…` would be 39.999… — the reason the
        // old code needed an epsilon; the integer path is exact + can't mis-round).
        let (num, den) = recoil_fraction_to_ratio(recoil_fraction);
        let recoil = (((dealt as u32) * num as u32) / den as u32) as u16;
        let recoil = recoil.max(1);
        let user_hp_before = self.sides[side].pokemon[slot].hp;
        let recoil = recoil.min(user_hp_before);
        // FOCUS BAND: the recoil is a Damage event into the user (effect 'recoil', NOT a
        // Move) — the roll draws, no survive. `gen3_ability_batch4_v1`.
        let recoil = self.focus_band_damage(side, slot, recoil, false, false, dex);
        self.apply_damage(side, slot, recoil);
        if self.logging() {
            let user = self.mon_ref(side, slot, dex);
            let target = self.mon_ref(target_side, target_slot, dex);
            let hp = self.hp_status(side, slot);
            self.log.damage_of(&user, &hp, &Cause::Bare("Recoil".into()), &target);
        }
    }

    /// DRAIN (`gen3_move_coverage_batch1_v1`, Giga Drain / Absorb / Mega Drain / Leech Life —
    /// `drain:[1,2]`): the USER HEALS by the drain fraction of the damage dealt AFTER a landed
    /// hit. DRAW-FREE (`this.heal` consumes no PRNG); heal-at-full-HP FAILS draw-free (via
    /// `apply_heal`). It fires whether the damage hit the MON or a SUBSTITUTE. **The gen<5
    /// floor/ceil split**: the non-sub path (`battle.ts::damage`) uses `floor(dmg·num/den)`
    /// clamped to `>=1`; the SUB path (`substitute.onTryPrimaryHit`) uses `ceil(dmg·num/den)` —
    /// so `absorbed` selects the rounding (equal for `[1,2]` and EVEN `dealt`, differ by 1 for
    /// odd). **LIQUID OOZE** (`gen3_liquid_ooze_v1`, the drain TARGET's ability, `onSourceTryHeal`)
    /// REVERSES it: the USER (drainer) takes the would-be heal as DAMAGE instead of healing —
    /// `|-damage|<user>|<HP>|[from] ability: Liquid Ooze|[of] <target>` (NO `-heal`; can KO the
    /// user). Dream Eater is EXCLUDED from the reversal in gen3 (the gen4 override's
    /// `activeMove?.id !== 'dreameater'` gate) — moot here since Dream Eater is not a
    /// `MODELED_DRAIN_MOVES` member, so it never routes to `apply_drain`. DRAW-FREE reversal
    /// (probe `harness/probe_batch89_abilities_items.js`: a Giga Drain into a Liquid Ooze mon
    /// draws the SAME accuracy/crit/damage chain as any drain move). Otherwise emitted as the
    /// normal `|-heal|<user>|<HP>|[from] drain|[of] <target>`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_drain(
        &mut self,
        side: usize,
        slot: usize,
        target_side: usize,
        target_slot: usize,
        drain_fraction: f64,
        dealt: u16,
        absorbed: bool,
        dex: &Dex,
    ) {
        if drain_fraction <= 0.0 || dealt == 0 {
            return;
        }
        // gen<5: non-sub `floor(dmg·num/den)` clamped `>=1` (battle.ts:2168); the SUB path
        // `ceil(dmg·num/den)` (substitute.onTryPrimaryHit). Compute via INTEGER rational math
        // (recovered from the fraction: `[1,2]`=0.5 / `[3,4]`=0.75) — NOT a float floor/ceil
        // with an epsilon (an epsilon on an EXACT even product like 68·0.5==34.0 wrongly
        // ceil'd to 35 — the e2e_33 Giga-Drain-into-a-sub bug). `den` is the smallest integer
        // making `frac·den` integral (1/2 → 2, 3/4 → 4); `num = round(frac·den)`.
        let (num, den) = drain_fraction_to_ratio(drain_fraction);
        let prod = (dealt as u32) * (num as u32);
        let heal = if absorbed {
            // ceil(prod/den)
            (((prod + den as u32 - 1) / den as u32) as u16).max(1)
        } else {
            // floor(prod/den), clamped >=1
            ((prod / den as u32) as u16).max(1)
        };
        // LIQUID OOZE (`gen3_liquid_ooze_v1`): the drain TARGET reverses the heal into DAMAGE on
        // the USER. Focus Band on the USER draws its `onDamage` roll (but NEVER survives — the
        // reversal's effect is the ABILITY, not a Move, so `is_move = false`); the damage can KO
        // the drainer through the normal deferred-faint machinery (like a recoil KO).
        if to_id(&self.sides[target_side].pokemon[target_slot].ability) == "liquidooze" {
            let dmg = self.focus_band_damage(side, slot, heal, false, false, dex);
            self.apply_damage(side, slot, dmg);
            if self.logging() {
                let user = self.mon_ref(side, slot, dex);
                let target = self.mon_ref(target_side, target_slot, dex);
                let hp = self.hp_status(side, slot);
                self.log.damage_of(&user, &hp, &Cause::Ability("Liquid Ooze".into()), &target);
            }
            // INSTAFAINT (`gen3_liquid_ooze_instafaint_v1`) — the drain sibling of the
            // Leech-Seed reversal: gen4's liquidooze calls `this.damage(…, instafaint = true)`,
            // so a reversal that zeroes the DRAINER drains the faint queue right here with
            // `lastFirst = true` (the drainer announced before the KO'd drain target).
            self.liquid_ooze_instafaint(side, dex);
            return;
        }
        let healed = self.apply_heal(side, slot, heal);
        if self.logging() && healed {
            let user = self.mon_ref(side, slot, dex);
            let target = self.mon_ref(target_side, target_slot, dex);
            let hp = self.hp_status(side, slot);
            self.log.heal_of(&user, &hp, &Cause::Bare("drain".into()), &target);
        }
    }

    /// SELF STAT-DROP (`gen3_move_coverage_batch1_v1`, `move.self.boosts` — Overheat −2 SpA,
    /// Superpower −1 Atk/−1 Def): apply the self-drop to the USER after a landed hit.
    ///
    /// **THE `selfDrops` DRAW (probe-settled — the "draw-free" hypothesis was WRONG; the mod
    /// chain is the only oracle).** gen3 `selfDrops` (battle-actions.ts:1338) draws ONE
    /// `random(100)` (the `secondaryRoll`) when `moveData.self.boosts` exists and this isn't a
    /// secondary, THEN applies the drop if `secondaryRoll < self.chance` OR — Overheat /
    /// Superpower have `self.chance === undefined` — UNCONDITIONALLY. So the drop ALWAYS
    /// applies but the roll is ALWAYS DRAWN (verified via a per-call-site PRNG trace: Overheat
    /// draws `random(100)` at the `selfDrops` position, AFTER the move's own damage + BEFORE
    /// the foe's move). This is the ONE draw the batch-1 post-hit effects add — and the reason
    /// the port's Overheat/Superpower were never seed-verified (they were MISMODELED, skipping
    /// this). The port draws-then-DISCARDS one `random_below(100)` here.
    ///
    /// The APPLY is a plain `boost()` on the user (±6 clamp, draw-free), IDENTICAL to the
    /// setup-move self-boost path. It fires on ANY landed hit INCLUDING behind a SUBSTITUTE
    /// (the gen3 `selfDrops` targets the USER, not the sub — probe-verified). Our OWN Clear
    /// Body / Hyper Cutter never blocks our own self-drop (the `onTryBoost` immunity is
    /// FOE-drop-only). Emitted `|-boost|`/`|-unboost|` per stat by the CLAMPED-applied delta's
    /// sign — a drop into the −6 floor is a delta-0 no-op that emits NOTHING (the sim's
    /// `boost()` skips it — probe-verified).
    pub(crate) fn apply_self_drops(&mut self, side: usize, slot: usize, self_drops: &[(usize, i8)], dex: &Dex) {
        // THE `selfDrops` `random(100)` — drawn UNCONDITIONALLY (Overheat / Superpower have
        // `self.chance === undefined`, so the drop always applies but the roll still fires).
        let _ = self.prng.random_below(100);
        for &(idx, stages) in self_drops {
            let cur = self.sides[side].pokemon[slot].boosts[idx] as i32;
            let next = (cur + stages as i32).clamp(-6, 6);
            self.sides[side].pokemon[slot].boosts[idx] = next as i8;
            if self.logging() {
                let delta = (next - cur) as i8; // the applied (post-clamp) magnitude
                let user = self.mon_ref(side, slot, dex);
                self.log.boost(&user, idx, delta);
            }
        }
        // NOTE: the WHITE HERB restore is DELIBERATELY **not** fired here. The resolved gen3
        // White Herb has NO `onAfterBoost`/`onUpdate` trigger — its only in-turn hook is
        // `onAnyAfterMove`, which the sim runs at the END of `runMove`, i.e. AFTER the whole
        // move body (selfDrops → secondaries → onAfterHit → the DamagingHit-phase procs).
        // SIM-PROBED (`harness/probe_rb_tail.js` C2a/C2b): a White-Herb Superpower into a
        // POISON POINT / ROUGH SKIN holder emits
        //     |-unboost|<user>|atk|1 → |-unboost|<user>|def|1
        //     → |-status|<user>|psn|[from] ability: Poison Point|[of] <foe>   (the proc)
        //     → |-enditem|<user>|White Herb → |-clearnegativeboost|<user>|[silent]
        // so restoring inline here emitted the item lines one DamagingHit phase too early.
        // The driver's `QAction::Move` tail (`turn/driver.rs`, the `onAnyAfterMove` site) is
        // the single in-turn restore point; the switch-in (`onAnySwitchIn`) + the order-29
        // residual keep their own sites. Emission-order only (the restore is draw-free and
        // still lands before the next decision boundary).
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
    pub(crate) fn apply_flash_fire_activation(&mut self, side: usize, slot: usize, move_type: Option<Type>) {
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
        // The bare-status `[from]` token for the emitted `|-damage|`. NOTE the Toxic DoT is
        // reported `[from] psn` (NOT `tox`) — Showdown's `tox` residual damages with the
        // `psn` cause (the HP-field status token STAYS `tox` via `hp_status`; only this
        // `[from]` cause collapses to `psn`). The byte fuzzer surfaced this latent gap on
        // real Toxic teams (`gen3_omniscient_byte_fuzz_v1`); the constructed protocol golden
        // never realized a Toxic residual chip.
        let (dmg, from_tok): (u16, &'static str) = match mon.status {
            Some(Status::Burn) => ((maxhp / 8).max(1), "brn"),
            Some(Status::Poison) => ((maxhp / 8).max(1), "psn"),
            Some(Status::Toxic(stage)) => {
                // gen3: ramp the stage (cap 15) BEFORE computing the damage.
                let next = if stage < 15 { stage + 1 } else { 15 };
                mon.status = Some(Status::Toxic(next));
                ((maxhp / 16).max(1) * next as u16, "psn")
            }
            _ => return,
        };
        // FOCUS BAND: the onDamage roll draws on the DoT chip (no survive — not a Move).
        let dmg = self.focus_band_damage(side, slot, dmg, false, false, dex);
        self.apply_damage(side, slot, dmg);
        // [EMIT] `|-damage|<mon>|<HP>|[from] <brn|psn|tox>` (the residual DoT chip;
        // the HP is post-chip, `0 fnt` if the DoT KO'd). Observation-only.
        if dmg > 0 && self.logging() {
            let mon_ref = self.mon_ref(side, slot, dex);
            let hp = self.hp_status(side, slot);
            self.log.damage(&mon_ref, &hp, Some(&Cause::Bare(from_tok.to_string())));
        }
    }

    /// The **CURSE** residual chip (`curse.onResidual`, `gen3_move_coverage_batch3_v1`):
    /// the cursed holder (`side`/`slot`) loses `floor(baseMaxhp/4)` (`this.damage(pokemon.
    /// baseMaxhp/4)`), emitting `|-damage|<foe>|<hp>|[from] Curse`. DRAW-FREE. Order 10,
    /// subOrder 8 (see `CURSE_RESIDUAL_SUBORDER`). Focus Band's onDamage roll applies (no
    /// survive — the curse damage is not a Move effect, so it can't survive-at-1). VERIFIED
    /// vs `harness/probe_batch3_curse.js`: a cursed Snorlax (maxhp 524) loses 131/turn.
    fn apply_curse(&mut self, side: usize, slot: usize, dex: &Dex) {
        let mon = &self.sides[side].pokemon[slot];
        if mon.fainted || mon.curse.is_none() {
            return;
        }
        let dmg = (mon.maxhp / 4).max(1);
        // Focus Band's onDamage roll draws on the curse chip (no survive — not a Move).
        let dmg = self.focus_band_damage(side, slot, dmg, false, false, dex);
        self.apply_damage(side, slot, dmg);
        // [EMIT] `|-damage|<foe>|<HP>|[from] Curse`.
        if dmg > 0 && self.logging() {
            let mon_ref = self.mon_ref(side, slot, dex);
            let hp = self.hp_status(side, slot);
            self.log.damage(&mon_ref, &hp, Some(&Cause::Bare("Curse".to_string())));
        }
    }

    /// The **WISH** slot-condition delayed heal (`wish.onEnd`, `gen3_move_coverage_batch3_v1`):
    /// decrement the side's `wish_pending` duration; when it reaches 0 the Wish RESOLVES —
    /// if the slot's active mon is not fainted, heal `floor(baseMaxhp/2)`. A NON-zero heal
    /// emits `|-heal|<mon>|<hp>|[from] move: Wish|[wisher] <name>` (the wisher name stored at
    /// cast); a heal-at-FULL resolves SILENTLY (`this.heal` returns 0 → the `if(damage)` guard
    /// skips the line). DRAW-FREE. Order 7 (fires BEFORE the sand chip + all order-10 handlers
    /// — VERIFIED). VERIFIED vs `harness/probe_batch3_wish.js`: Blissey 714→+357; Charizard
    /// 298→+149 (floor); a switched-in Chansey (704) gets +352.
    fn apply_wish(&mut self, side: usize, dex: &Dex) {
        // Decrement the pending duration (`fieldEvent('Residual')` decrements the slot
        // condition's `duration`); fire the `onEnd` heal at 0. The wisher name is consumed
        // from the pending state.
        let (dur, wisher) = match self.sides[side].wish_pending.clone() {
            Some(p) => p,
            None => return,
        };
        let next = dur.saturating_sub(1);
        if next > 0 {
            // Not yet resolving — just tick the duration down.
            self.sides[side].wish_pending = Some((next, wisher));
            return;
        }
        // RESOLVE: clear the pending state, then heal the CURRENT slot occupant.
        self.sides[side].wish_pending = None;
        let slot = self.sides[side].active;
        let mon = &self.sides[side].pokemon[slot];
        if mon.fainted {
            // `wish.onEnd` guards `if (!target.fainted)` — a fainted slot heals nothing.
            return;
        }
        let amount = mon.maxhp / 2; // floor(baseMaxhp/2)
        let healed = self.apply_heal(side, slot, amount);
        // [EMIT] `|-heal|<mon>|<HP>|[from] move: Wish|[wisher] <name>` — only when a NON-zero
        // heal landed (a heal-at-full resolves silently, `heal` returns 0).
        if healed && self.logging() {
            let mon_ref = self.mon_ref(side, slot, dex);
            let hp = self.hp_status(side, slot);
            self.log.heal_wish(&mon_ref, &hp, &wisher);
        }
    }

    /// The pending **FUTUREMOVE** slot condition's residual tick / RESOLVE
    /// (`gen3_move_coverage_batch4c_v1`, Doom Desire / Future Sight — the gen-3
    /// `futuremove` condition's order-11 `onResidual`/`onEnd`, probe-settled bit-for-bit
    /// vs `harness/probe_batch4c_doomdesire.js`):
    ///
    ///   * a NON-final tick just decrements the duration (3 → 2 → 1), DRAW-FREE;
    ///   * the FINAL tick (1 → 0) RESOLVES: `removeSlotCondition` (the pending state is
    ///     consumed EITHER WAY) → `onEnd`:
    ///       - SKIP (no strike, zero draws) iff the CURRENT slot occupant is FAINTED
    ///         (`target === source` — the other skip — is impossible in gen-3 singles).
    ///         The sim emits a `-hint` here whose exact text is UNCAPTURED → deliberately
    ///         not emitted (the honesty discipline — like the sub-absorbed Pay Day form).
    ///       - else `|-end|<target>|move: <Name>`, remove the target's PROTECT volatile
    ///         (the strike ignores a resolve-turn Protect — probed full 366 through it),
    ///         then ONE accuracy roll (`randomChance(85|90,100)` — the standard
    ///         `hitStepAccuracy` acc/eva-stage + accMod fold; the STAGE fold at resolve is
    ///         probe-UNREACHED [no modeled gen3 path raises evasion] and modeled as the
    ///         standard fold, per the probe's PLAUSIBLE-standard-fold disposition).
    ///       - HIT: the STORED cast-time number lands fixed-damage-style — NO crit roll,
    ///         NO damage roll; a SUBSTITUTE absorbs it (accuracy still drawn; breaks, no
    ///         carry); Focus Band (a Move-effect Damage event into the occupant) can roll
    ///         its survive; typeless '???' → no type-chart row → NEVER immune (DD is
    ///         neutral into Fire, FS hits Dark). A LANDED resolve then draws TWO extra
    ///         `eachEvent('Update')` shuffles (`hitStepMoveHitLoop` — tie-only; zero at
    ///         distinct speed; probed: the FS-mirror resolve turn draws 14 vs the
    ///         distinct-speed 3).
    ///       - MISS: `|-miss|<caster>|<target>`, NO damage, NO extra Updates — plus the
    ///         sim's `attrLastMove('[miss]')` PROTOCOL QUIRK (the `[miss]` retro-appends
    ///         to the LAST `|move|` line of the turn — the TARGET'S OWN move line).
    ///     The strike resolves even when the CASTER has switched out or FAINTED (probed:
    ///     a fainted-benched Jirachi's DD still dealt the full stored 366); the caster is
    ///     resolved by its stable uid for the accuracy fold + the `-miss` ident.
    ///     A resolve KO rides the caller's per-handler `faintMessages` + `checkWin` (a
    ///     game-ending resolve KO aborts the remaining residual handlers; the Quick Claw
    ///     is deferred past the forced replacement — the deferred-faint protocol).
    fn apply_future_move(&mut self, side: usize, dex: &Dex) {
        let fm = match self.sides[side].future_move.clone() {
            Some(f) => f,
            None => return,
        };
        if fm.duration > 1 {
            self.sides[side].future_move =
                Some(FutureMove { duration: fm.duration - 1, ..fm });
            return;
        }
        // RESOLVE: consume the slot condition (removeSlotCondition), then strike.
        self.sides[side].future_move = None;
        let slot = self.sides[side].active;
        let move_name = dex
            .moves(&fm.move_id)
            .map(|m| m.name.clone())
            .unwrap_or_else(|| fm.move_id.clone());
        if self.sides[side].pokemon[slot].fainted {
            // A fainted occupant skips the strike (`futuremove.onEnd`'s early-return); the sim
            // emits `|-hint|<Move> did not hit because the target is fainted.` (`once` falsy →
            // fires every time, no dedup). The `target === source` "the user" branch is
            // unreachable in gen3 singles (a foe-target future move can't occupy the caster's
            // slot). Zero draws either way. (`gen3_omniscient_byte_fuzz_v1` — was uncaptured.)
            if self.logging() {
                self.log
                    .hint(&format!("{move_name} did not hit because the target is fainted."), false);
            }
            return;
        }
        // [EMIT] `|-end|<target>|move: Doom Desire` (on the TARGET, with the `move:`
        // prefix — probe-observed shape).
        if self.logging() {
            let t = self.mon_ref(side, slot, dex);
            self.log.volatile_end(&t, &format!("move: {move_name}"));
        }
        // `onEnd` removes the target's Protect/Endure volatile BEFORE the strike (the
        // stored number lands through a resolve-turn Protect — probed).
        self.sides[side].pokemon[slot].protected = false;
        // The caster (may be benched / fainted — the strike still resolves).
        let src_side = fm.source_side;
        let src_slot = self
            .slot_of_uid(src_side, fm.source_uid)
            .unwrap_or(self.sides[src_side].active);
        // ONE accuracy roll (typeless → no move_type; never_miss false).
        let hit = self.roll_accuracy(src_side, src_slot, side, slot, fm.accuracy, false, None, dex);
        if !hit {
            // [EMIT] the `attrLastMove('[miss]')` retro-edit (the `[miss]` lands on the
            // LAST `|move|` line of the turn — the target's own move line, the observed
            // protocol quirk) then `|-miss|<caster>|<target>`.
            if self.logging() {
                self.log.attr_last_move_miss();
                // The `-miss` SOURCE is the future-move CASTER, which may have switched out /
                // fainted since casting → render via `Pokemon.toString()` (slot-less `pN:` when
                // not the active), NOT `mon_ref`'s always-`pNa:` form (golden L226:
                // `|-miss|p1: Jirachi|p2a: Suicune`). `gen3_omniscient_byte_fuzz_v1` class B.
                let u = self.mon_toref(src_side, src_slot, dex);
                let t = self.mon_ref(side, slot, dex);
                self.log.miss_raw_user(&u, &t);
            }
            return;
        }
        // The STORED number lands fixed-damage-style (sub-intercept first; Focus Band on
        // a direct hit; deferred faint via the caller's per-handler faintMessages).
        let realized = fm.damage;
        let sub = self.absorb_into_sub(side, slot, realized);
        if sub == SubAbsorb::NoSub {
            // ENDURE does NOT clamp the future-move resolve (`gen3_futuremove_no_endure_v1`,
            // the random-mode byte-fuzz find ab_19_1 @ master-seed 200724 — CORRECTING the prior
            // CLASS-INFERRED, "not probe-verified" guess that endure clamps it). endure's
            // `duration: 1` means it is only active the turn it is used, and it EXPIRES at THAT
            // turn's residual BEFORE the order-11 future-move resolve (the LAST residual handler),
            // so its `onDamage` is no longer registered when the resolve deals its Move damage —
            // the sim faints the target with NO `|-activate|move: Endure` line (golden ab_19_1:
            // an Enduring Vaporeon takes a lethal Future Sight resolve → `0 fnt`). Since a
            // future-move ALWAYS resolves at a residual (order 11) and endure is ALWAYS
            // duration-1-same-turn, endure can NEVER protect a resolve — so no endure_clamp here
            // (unlike the plain / fixed / per-multihit-strike Move-phase sites, where endure IS
            // live). Focus Band still rolls (its `randomChance(1,10)` draws on any Damage event).
            // DRAW-FREE (endure_clamp consumes no PRNG either way).
            let realized = self.focus_band_damage(side, slot, realized, true, true, dex);
            self.apply_damage(side, slot, realized);
            if realized > 0 && self.logging() {
                let t = self.mon_ref(side, slot, dex);
                let hp = self.hp_status(side, slot);
                self.log.damage(&t, &hp, None);
            }
        } else if self.logging() {
            let t = self.mon_ref(side, slot, dex);
            match sub {
                SubAbsorb::Held => self.log.activate(&t, "Substitute", Some("[damage]")),
                SubAbsorb::Broke => self.log.volatile_end(&t, "Substitute"),
                SubAbsorb::NoSub => unreachable!(),
            }
        }
        // A LANDED resolve (direct OR sub-absorbed — both are a hit) fires TWO extra
        // `eachEvent('Update')`s inside `hitStepMoveHitLoop` (tie-only draws), with the
        // in-loop `faintMessages(false, false, !pokemon.hp)` (battle-actions.js:852)
        // BETWEEN them — so on a resolve KO the FIRST Update still gathers the 0-HP
        // not-yet-fainted target (a tie draws) while the SECOND no longer does
        // (`getAllActive` excludes the now-fainted mon → one active → no tie, no draw).
        // PROBE-verified on the fs_mirror board: a non-KO tie resolve draws acc+2 Updates;
        // the KO resolve draws acc+1 (the |faint| emits between them). The inner
        // faintMessages also ENDS a battle the KO decided (the caller's per-handler
        // `check_win` return then aborts the remaining residual handlers).
        self.each_event_shuffle();
        self.process_faints(dex);
        self.each_event_shuffle();
    }

    /// The **PARTIAL-TRAP** residual (`gen3_partial_trap_v1` — Wrap / Bind / Fire Spin /
    /// Clamp / Whirlpool / Sand Tomb). Returns `true` iff this was the DURATION-END visit
    /// (the caller then `continue`s, skipping the per-handler `faintMessages` exactly like
    /// the sim's `fieldEvent('Residual')` `duration-- == 0 → end() → continue` branch).
    ///
    /// The resolved gen-3 condition is the base `partiallytrapped` with the **gen5**
    /// `onStart`/`onResidual` and the **gen4** `durationCallback`/order override:
    ///
    /// ```text
    /// onResidual(pokemon) {                                  // data/mods/gen5/conditions.ts
    ///   const trapper = this.effectState.source;
    ///   if (trapper && (!trapper.isActive || trapper.hp <= 0 || !trapper.activeTurns)) {
    ///     delete pokemon.volatiles['partiallytrapped'];
    ///     this.add('-end', pokemon, sourceEffect, '[partiallytrapped]', '[silent]');
    ///     return;
    ///   }
    ///   this.damage(pokemon.baseMaxhp / this.effectState.boundDivisor);   // 16 in gen3
    /// }
    /// onEnd(pokemon) { this.add('-end', pokemon, sourceEffect, '[partiallytrapped]'); }
    /// ```
    ///
    /// Behaviours (probe-settled — `harness/probe_ptrap_edges{,2}.js`):
    ///   * **DURATION FIRST.** `fieldEvent` decrements before it calls `onResidual`, so a
    ///     turn on which the duration expires AND the trapper leaves emits the NON-silent
    ///     `onEnd` form (probe O) and takes no chip. Chip turns == `duration − 1` (2–5).
    ///   * **THE TRAPPER-GONE TEST** is `!isActive || hp <= 0 || !activeTurns` — the port's
    ///     equivalent is "the stored uid is no longer the foe side's live, non-freshly-
    ///     switched-in active". A trapper that switched out (probe: RELEASE), fainted (probe
    ///     D), was phazed, or Baton-Passed away all release identically + SILENTLY.
    ///   * **THE CHIP** is `floor(maxhp/16)` (gen-3 `baseMaxhp == maxhp`), via the sim's
    ///     `this.damage()` → `clampIntRange(_, 1)` so a sub-16-maxhp mon still takes 1. The
    ///     holder's **Focus Band** `onDamage` DRAWS its roll (the gen-3 handler rolls
    ///     `randomChance(1,10)` FIRST, before the `effectType === 'Move'` test) but can never
    ///     survive — the same `is_move: false` shape as the Leech-Seed drain. A chip CAN KO
    ///     (probe E: `|-damage|…|0 fnt|[from] move: Wrap|[partiallytrapped]` then `|faint|`).
    fn apply_partial_trap(&mut self, side: usize, slot: usize, dex: &Dex) -> bool {
        // (1) The DURATION tick — `handler.state.duration--; if (!duration) { end(); continue; }`.
        let (expired, move_name, source_uid) = {
            let mon = &mut self.sides[side].pokemon[slot];
            match mon.partial_trap.as_mut() {
                Some(pt) => {
                    pt.duration = pt.duration.saturating_sub(1);
                    (pt.duration == 0, pt.move_name.clone(), pt.source_uid)
                }
                None => return false, // gathered only when present; belt-and-braces
            }
        };
        if expired {
            self.sides[side].pokemon[slot].partial_trap = None;
            // [EMIT] `|-end|<mon>|<Move>|[partiallytrapped]` — the `onEnd` (NON-silent). The
            // effect token is the BARE move name (the condition passes the sourceEffect
            // OBJECT, which stringifies to `name`), NOT `move: <Move>`.
            if self.logging() {
                let m = self.mon_ref(side, slot, dex);
                self.log.partial_trap_end(&m, &move_name, false);
            }
            return true;
        }

        // (2) THE TRAPPER-GONE TEST. In gen-3 singles the trapper is always on the opposite
        // side, so `isActive` reduces to "the stored uid is that side's current active"; the
        // `!activeTurns` arm additionally releases when that mon only switched in THIS turn
        // (unreachable on any board probed here, but it is the sim's condition).
        let trapper_side = 1 - side;
        let trapper_slot = self.sides[trapper_side].active;
        let trapper = &self.sides[trapper_side].pokemon[trapper_slot];
        let trapper_gone = trapper.uid != source_uid
            || trapper.fainted
            || trapper.hp == 0
            || trapper.active_turns == 0;
        if trapper_gone {
            self.sides[side].pokemon[slot].partial_trap = None;
            // [EMIT] `|-end|<mon>|<Move>|[partiallytrapped]|[silent]` — the onResidual release.
            if self.logging() {
                let m = self.mon_ref(side, slot, dex);
                self.log.partial_trap_end(&m, &move_name, true);
            }
            return false; // NOT a duration-end → fall through to faintMessages
        }

        // (3) THE CHIP: `this.damage(baseMaxhp / 16)`.
        let hp_before = self.sides[side].pokemon[slot].hp;
        let chip = (self.sides[side].pokemon[slot].maxhp / 16).max(1);
        // FOCUS BAND: the onDamage roll DRAWS on the chip (no survive — not a Move effect).
        let chip = self.focus_band_damage(side, slot, chip, false, false, dex);
        let dealt = chip.min(hp_before);
        self.apply_damage(side, slot, chip);
        // [EMIT] `|-damage|<mon>|<HP>|[from] move: <Move>|[partiallytrapped]`.
        if self.logging() && dealt > 0 {
            let m = self.mon_ref(side, slot, dex);
            let hp = self.hp_status(side, slot);
            self.log.damage_partially_trapped(&m, &hp, &move_name);
        }
        false
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
    ///   * **LIQUID OOZE** (`gen3_liquid_ooze_v1`, the seeded mon's ability, `onSourceTryHeal`)
    ///     REVERSES it — the SEEDER takes the drained amount as DAMAGE instead of healing
    ///     (`|-damage|<seeder-active>|<HP>|[from] ability: Liquid Ooze|[of] <seeded>`, AFTER the
    ///     seeded mon's own `-damage [from] Leech Seed`; can KO the seeder). DRAW-FREE (probe:
    ///     the residual draws nothing either way). Focus Band on the seeder draws its `onDamage`
    ///     roll but never survives (the effect is the ABILITY, not a Move).
    fn apply_leech_seed(&mut self, side: usize, slot: usize, seeder_side: usize, dex: &Dex) {
        let seeded_liquid_ooze = to_id(&self.sides[side].pokemon[slot].ability) == "liquidooze";

        // The seeder's CURRENT active (the heal recipient / `getAtSlot(sourceSlot)`).
        let seeder_slot = self.sides[seeder_side].active;
        let seeder = &self.sides[seeder_side].pokemon[seeder_slot];
        // SEEDER-FAINTED gate: `if (!target || target.fainted || target.hp <= 0) return;`
        // — no drain, no heal, the whole onResidual returns.
        if seeder.fainted || seeder.hp == 0 {
            return;
        }

        // DRAIN the seeded mon `floor(maxhp/8)`, clamped to its HP (apply_damage saturates
        // at 0 + returns whether it KO'd; the dealt amount is min(drain, hp_before)). The gen-3
        // `this.damage(baseMaxhp/8)` passes through `clampIntRange(_, 1)`, so the drain is at LEAST
        // 1 for any live mon: BYTE-IDENTICAL for maxhp >= 8 (floor >= 1), but a sub-8-maxhp mon
        // (Shedinja, maxhp 1 → floor 0) still takes 1 — the `gen3_wonder_guard_v1` Leech-Seed
        // residual bypass (a residual is NOT a move → Wonder Guard doesn't block it → a 1-HP
        // Shedinja dies to the drain, VERIFIED vs the sim).
        let seeded = &self.sides[side].pokemon[slot];
        let hp_before = seeded.hp;
        let drain = (seeded.maxhp / 8).max(1); // floor, clampIntRange(_, 1); gen-3 baseMaxhp == maxhp
        // FOCUS BAND: the onDamage roll draws on the drain (no survive — not a Move).
        let drain = self.focus_band_damage(side, slot, drain, false, false, dex);
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
            if seeded_liquid_ooze {
                // LIQUID OOZE (`gen3_liquid_ooze_v1`): the SEEDER takes `dealt` as DAMAGE
                // instead of healing. Focus Band on the seeder draws its `onDamage` roll but
                // never survives (the effect is the ability, `is_move = false`); the reversal
                // can KO the seeder via the residual per-handler faint machinery.
                let dmg = self.focus_band_damage(seeder_side, seeder_slot, dealt, false, false, dex);
                self.apply_damage(seeder_side, seeder_slot, dmg);
                // [EMIT] `|-damage|<seeder-active>|<HP>|[from] ability: Liquid Ooze|[of] <seeded>`
                // — AFTER the seeded mon's `-damage [from] Leech Seed` above (probe-verified order).
                if self.logging() {
                    let seeder_ref = self.mon_ref(seeder_side, seeder_slot, dex);
                    let hp = self.hp_status(seeder_side, seeder_slot);
                    let of = self.mon_ref(side, slot, dex);
                    self.log.damage_of(&seeder_ref, &hp, &Cause::Ability("Liquid Ooze".into()), &of);
                }
                // INSTAFAINT (`gen3_liquid_ooze_instafaint_v1`): gen4's liquidooze reverses via
                // `this.damage(damage, null, null, null, true)` — instafaint TRUE — so a reversal
                // that zeroes the seeder runs `faintMessages(lastFirst = true)` RIGHT HERE,
                // announcing the seeder's `|faint|` BEFORE the already-queued seeded corpse's.
                self.liquid_ooze_instafaint(seeder_side, dex);
            } else {
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
    }
}
