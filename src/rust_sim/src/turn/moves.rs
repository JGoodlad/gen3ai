use crate::damage::{
    calc_damage, AtkStatMod, BpMod, Combatant, DamageContext, MoveInput, Weather as DmgWeather,
};
use crate::dex::moves::MultiHit;
use crate::dex::{to_id, Dex, DmgFold, MoveCategory, Type};
use crate::protocol::Cause;
use crate::state::{FutureMove, Status, TwoTurnMove, Weather};
use super::status::DamagingHitPhase;
use super::*;
use super::helpers::*;

impl crate::state::BattleState {

    /// Resolve the [`MoveData`]-like move for a side's active move slot. `None` for
    /// an out-of-range slot or unknown move id.
    pub(crate) fn move_at<'a>(
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

    /// Resolve and APPLY one damaging move (accuracy → crit → damage → HP → faint).
    /// Consumes the PRNG in the exact verified order. Returns a [`MoveResolution`]
    /// whose `landed` flag gates the in-tryMoveHit `eachEvent('Update')` shuffle (it
    /// fires only when the move actually hits — a miss/immune returns from
    /// `tryMoveHit` before it).
    pub(crate) fn run_move(&mut self, action: MoveAction, will_act: bool, foe_will_move: bool, dex: &Dex) -> MoveResolution {
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
            pressure_targets_foe,
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
                m.display_name().to_string(), // "Struggle"
                m.contact,    // Struggle IS a contact move (gen-3 `flags.contact`) → it CAN proc a contact ability
                pressure_targets_foe(&m.target), // Struggle target=normal → foe in pressureTargets
            )
        } else {
            match self.move_at(side, slot, move_index, dex) {
                Some(m) => {
                // CURSE PP CRUX (`gen3_pressure_allyteam_v1`, D1): gen-3 `curse.onModifyMove`
                // RE-TARGETS a NON-GHOST user's Curse to `self` at runtime (`nonGhostTarget`),
                // but the static dex `target` is `"normal"`. The Pressure PP path (below) must
                // read the RUNTIME-effective target — else a non-Ghost Curse facing a Pressure
                // foe wrongly deducts 2 PP (static "normal" → foe in `pressureTargets`) instead
                // of 1 (self → foe NOT in `pressureTargets`), draining its PP ~1 cycle/turn early
                // and forcing Struggle turns the sim still Curses (the byte-fuzz 5_6 desync). So
                // feed BOTH `targets_self` and `pressure_targets_foe` the effective "self" target.
                // (The GHOST branch keeps the base `normal` foe target — a ghost Curse is
                // foe-directed, so `pressure_targets_foe` stays true.)
                let is_nonghost_curse = to_id(&m.id) == "curse"
                    && !mon_types(&self.sides[side].pokemon[slot], dex).contains(&Type::Ghost);
                let eff_target: &str = if is_nonghost_curse { "self" } else { &m.target };
                (
                    m.accuracy,
                    m.never_miss,
                    m.base_power,
                    m.move_type,
                    m.category,
                    to_id(&m.id) == "explosion" || to_id(&m.id) == "selfdestruct",
                    m.crit_ratio,
                    // `is_fire` gates the frozen-defender thaw (turn.rs thaw site). gen3
                    // `frz.onDamagingHit` (conditions.ts:45-50) thaws ONLY when the move's
                    // BASE-dex type is Fire — with the explicit "don't count Hidden Power or
                    // Weather Ball as Fire-type" comment. dex.moves.get('hiddenpower').type is
                    // 'Normal' and 'weatherball' is 'Normal', so a typed HP-Fire (runtime type
                    // Fire, move nums 355-370) must NOT thaw. We compute the RESOLVED runtime
                    // type here, so exclude those two by base id to match the base-type
                    // semantics (`gen3_omniscient_byte_fuzz_v1` freeze-persistence fix).
                    m.move_type == Some(Type::Fire)
                        && m.category != MoveCategory::Status
                        && !to_id(&m.id).starts_with("hiddenpower")
                        && to_id(&m.id) != "weatherball",
                    to_id(&m.id),
                    m.status_inflicted.clone(),
                    m.is_protect,
                    eff_target == "self",
                    m.display_name().to_string(),
                    m.contact,
                    pressure_targets_foe(eff_target),
                )
                }
                // Not a known move — resolve as a no-op (draws nothing). This is a
                // programming error in scope (the caller picks damaging slots).
                None => return MoveResolution::no_op(),
            }
        };

        // TYPED HIDDEN POWER canonicalizes to the bare `Hidden Power` in EVERY emitted
        // protocol line (`gen3_omniscient_byte_fuzz_v1`): gen-3 hides the HP TYPE (it is
        // hidden info), so the sim's `|move|` / `|cant|` / fail lines always read
        // `Hidden Power`, never the typed dex display name (`Hidden Power Ice`). The engine
        // resolves the packed team's TYPED variant (num 355-370) whose `MoveData.name` IS
        // the typed name, so rebind here for the announce. The typed row still drives the
        // real BP/type in the DAMAGE calc (unchanged); only the DISPLAY collapses. The byte
        // fuzzer surfaced this the moment HP became pickable (pool mode).
        let move_name = if move_id.starts_with("hiddenpower") {
            "Hidden Power".to_string()
        } else {
            move_name
        };

        // --- SNORE fail-loud guard (`gen3_move_coverage_batch5_v1` scope edge): Snore
        //     is a bp-40 damaging move carrying `sleepUsable` + an asleep-only `onTry`
        //     — the port models NEITHER (Sleep Talk is the ONLY modeled sleepUsable
        //     move; `on_before_move`'s slp arm proceeds only for sleeptalk). Running it
        //     as a plain damaging move silently mismodels BOTH branches: AWAKE the sim
        //     onTry-fails silently (the port would deal damage); ASLEEP the sim cants
        //     then sleepUsable-PROCEEDS (the port cants + blocks). No pool team carries
        //     it (e2e-picker-blocklisted, `gen_e2e_fuzz.js`), but a future team source
        //     (e.g. the training bridge) could reach the engine directly — PANIC per
        //     the fail-loud canon, BEFORE any draw (the guard sits before
        //     `on_before_move`, so an asleep Snore selection panics too). ---
        if move_id == "snore" {
            panic!(
                "unmodeled sleepUsable move 'snore' reached run_move — Snore's asleep-only \
                 onTry + the sleepUsable cant-then-proceed are NOT modeled (Sleep Talk is the \
                 only modeled sleepUsable move). Model it bit-for-bit or keep it off the \
                 pickable set — do NOT let it run as a plain bp-40 damaging move."
            );
        }

        // BATCH-1 post-hit effect specs (`gen3_move_coverage_batch1_v1`): recoil / drain /
        // self-drop fractions read from the move data. Struggle's recoil rides its OWN
        // dedicated gen-3 path below (so leave it 0 here to avoid a double recoil). Rapid
        // Spin / Knock Off / Thief / Covet are id-driven (no data field). All draw-free.
        let (recoil_fraction, drain_fraction, self_drops): (f64, f64, Vec<(usize, i8)>) = if struggle
        {
            (0.0, 0.0, Vec::new())
        } else {
            match self.move_at(side, slot, move_index, dex) {
                Some(m) => (m.recoil_fraction, m.drain_fraction, m.self_drops.clone()),
                None => (0.0, 0.0, Vec::new()),
            }
        };

        // --- PURSUIT INTERRUPT (`gen3_move_coverage_batch4_v1`): read+clear the transient
        //     `pursuit_strike` flag `execute_switch` sets for the ONE run_move that resolves the
        //     interrupt strike (the sim's bare `useMove(pursuit, source, {target: switcher})`
        //     inside `onBeforeSwitchOut`). When set, Pursuit's `basePowerCallback` DOUBLES the BP
        //     (`target.beingCalledBack` — the switcher) and its `onModifyMove` makes it NEVER-MISS
        //     (`move.accuracy = true`), AND the strike SKIPS `on_before_move` / PP / lastMove
        //     (already handled manually by the interrupt), mirroring `useMove` (not `runMove`). ---
        let pursuit_strike = self.pursuit_strike;
        self.pursuit_strike = false;
        // --- SLEEP TALK CALLED MOVE (`gen3_move_coverage_batch5_v1`): read+clear the
        //     transient `sleep_talk_call` flag the Sleep Talk arm sets for the ONE
        //     recursive run_move that executes the SAMPLED move (the sim's bare
        //     `actions.useMove(picked)` inside `sleeptalk.onHit`). Like the pursuit
        //     strike, the called move SKIPS on_before_move / PP / lastMove / the
        //     mustrecharge gate (all owned by the OUTER Sleep Talk run) but otherwise
        //     runs its FULL NORMAL draw chain. ---
        let sleep_talk_call = self.sleep_talk_call;
        self.sleep_talk_call = false;
        // FAKE OUT (`gen3_fakeout_v1`): the sim increments `activeMoveActions` at the TOP of
        // `runMove`, before BeforeMove/PP/announce — so a turn spent CANT-ed still burns the gate.
        // Both transient callers below are `useMove` in the sim, which does NOT bump it.
        if !pursuit_strike && !sleep_talk_call {
            self.sides[side].pokemon[slot].active_move_actions =
                self.sides[side].pokemon[slot].active_move_actions.saturating_add(1);
        }
        let (never_miss, base_power) = if pursuit_strike {
            (true, base_power.saturating_mul(2))
        } else {
            (never_miss, base_power)
        };

        // --- MUSTRECHARGE cant (`gen3_move_coverage_batch4c_v1`, Hyper Beam's locked
        //     turn): the gen3-resolved `mustrecharge.onBeforeMove` at priority **11** —
        //     BEFORE sleep/frz (10) / truant (9) / flinch (8) / disable (7) / confusion
        //     (3) / attract (2) / paralysis (1) — emits `|cant|<user>|recharge`, removes
        //     the volatile (+ `removeVolatile('truant')`, a NO-OP in the port's
        //     `truant_turn` toggle model: the recharge cant fires before the truant gate
        //     and the order-27 residual toggle consumes the loaf — the probed Slaking
        //     HB/recharge/HB/recharge cadence with no truant cant ever), and returns null.
        //     ZERO draws (a par'd user draws NO para roll — probed: the recharge-turn seed
        //     advance is IDENTICAL with and without par; a slp'd user's counter does NOT
        //     decrement — `|cant|recharge`, not `|cant|slp`), NO PP (the recharge is not a
        //     slot; PP deduction sits after on_before_move, never reached). The lock fully
        //     clears — Hyper Beam is selectable again the following turn. This gate sits
        //     BEFORE the unmodeled-sibling fail-loud below: a locked mon never RESOLVES a
        //     move at all in the sim (the priority-11 cant precedes move resolution), so an
        //     unmodeled charge/recharge move sitting in the resolved slot must not panic on
        //     a recharge turn (review finding, batch-4c Lens 2). ---
        if !pursuit_strike && !sleep_talk_call && self.sides[side].pokemon[slot].must_recharge {
            self.sides[side].pokemon[slot].must_recharge = false;
            // DESTINY BOND's `onMoveAborted` (`gen3_move_coverage_batch6_v1`): a cant
            // CLOSES the window — the recharge cant is a BeforeMove abort, so a DB
            // volatile still up from a prior cast is removed here (draw-free).
            self.sides[side].pokemon[slot].destiny_bond = false;
            if self.logging() {
                let user = self.mon_ref(side, slot, dex);
                self.log.cant(&user, "recharge", None);
            }
            return MoveResolution::done(false, false, false);
        }

        // --- FAIL-LOUD: the UNMODELED recharge / charge siblings (`gen3_move_coverage_
        //     batch4c_v1`). Blast Burn / Frenzy Plant / Hydro Cannon ARE gen3-legal with
        //     the identical `self:mustrecharge` machinery but are UNPROBED for their
        //     specifics → fail-loud rather than silently run recharge-less. Razor Wind /
        //     Sky Attack / Skull Bash (+ the semi-invulnerable Fly / Dig / Dive / Bounce,
        //     whose conditions DIFFER from Solar Beam's) are `flags.charge` moves the port
        //     does not model → fail-loud rather than silently collapse to one turn.
        //     (gigaimpact / rockwrecker / roaroftime are `isNonstandard:'Future'` — not
        //     gen3-legal, absent from gen3_moves.json.) Only `hyperbeam` / `solarbeam` are
        //     modeled; `doomdesire` / `futuresight` are BOTH modeled (probe-settled
        //     same-mechanic). ---
        match move_id.as_str() {
            "blastburn" | "frenzyplant" | "hydrocannon" => panic!(
                "gen3_move_coverage_batch4c_v1: the RECHARGE move {move_id:?} is NOT modeled \
                 (only hyperbeam is) — fail-loud, not a silent recharge-less desync"
            ),
            "razorwind" | "skyattack" | "skullbash" | "fly" | "dig" | "dive" | "bounce" => panic!(
                "gen3_move_coverage_batch4c_v1: the CHARGE move {move_id:?} is NOT modeled \
                 (only solarbeam is) — fail-loud, not a silent one-turn collapse"
            ),
            _ => {}
        }

        // --- WATER SPOUT variable BP (`gen3_move_coverage_batch4b_v1`,
        //     `waterspout.basePowerCallback`): `bp = clampIntRange(150 * hp / maxhp, 1)` =
        //     `max(floor(150·hp/maxhp), 1)` — a deterministic STATE read of the USER's
        //     CURRENT hp, computed BEFORE the crit/damage draws, so DRAW-NEUTRAL (probe
        //     `harness/probe_batch4b_waterspout.js`: full-HP bp 150 and half-HP bp 74 end at
        //     the SAME seed with the same draws; only the damage magnitude differs). At 1 HP
        //     the raw 0.44 floors to 0 → clamped to min BP 1 (a min-damage HIT, does NOT
        //     fail — the `if (!basePower)` early-return only triggers on EXACT 0, unreachable
        //     while hp>=1). The multiply MUST widen to u32 (`150·714` overflows u16). Routed
        //     like Pursuit's ×2 id-gate; the integer division `(150·hp)/maxhp` is exact-equal
        //     to the JS `Math.floor(150·hp/maxhp)` since `150·hp` is an exact integer. ---
        //     ERUPTION joins it unchanged (`gen3_eruption_v1`): PROBE-SETTLED side by side in
        //     `harness/probe_varbp_cluster.js` — the SAME resolved callback, the same dataBP 150,
        //     and at identical hp the two derive the identical BP (150/297 -> 75 for both). It is
        //     Fire/SPECIAL where Water Spout is Water/SPECIAL, which the type split already
        //     handles, so the only thing it needed was admission to this gate.
        // --- THE BP-MODIFIER CLUSTER (`gen3_bp_modifier_cluster_v1`) — three
        //     `basePowerCallback`s over a NON-zero data row, all DRAW-NEUTRAL (deterministic
        //     STATE reads, computed before the crit/damage draws). Ground truth
        //     `harness/probe_bp_cluster_clean.js`, whose boards deliberately strip the two
        //     confounds that made the first measurement unreadable: a CRIT is itself a ×2 that
        //     mimics a BP doubling, and Fury Cutter's 95 accuracy means a MISS silently shifts
        //     the whole ladder. ---
        let base_power = if matches!(move_id.as_str(), "rollout" | "iceball") {
            // --- ROLLOUT / ICE BALL (`gen3_rollout_defensecurl_v1`): bp doubles per EXECUTION
            //     (30/60/120/240/480 — probe-measured 30/57/101/213 as HP deltas), and doubles
            //     AGAIN while the user carries the DEFENSE CURL volatile (probe: 56/108/204,
            //     i.e. each rung ×2). The count is EXECUTIONS, not turns: a MISS never reaches
            //     the callback so it does not advance the ladder. No duration draw — the lock
            //     is a fixed 5 executions. DRAW-NEUTRAL. ---
            let hits = self.sides[side].pokemon[slot].rollout.map(|(h, _)| h).unwrap_or(0);
            let curled = self.sides[side].pokemon[slot].defense_curl;
            let mut bp = base_power as u32 * (1u32 << hits.min(4));
            if curled {
                bp *= 2;
            }
            bp.min(u16::MAX as u32) as u16
        } else if move_id == "revenge" {
            // ×2 if the user was DAMAGED BY THIS TARGET this turn. Priority −4 means the foe's
            // move almost always lands first, which is what makes the doubling the common case.
            if self.sides[side].pokemon[slot].damaged_by_foe_this_turn {
                base_power.saturating_mul(2)
            } else {
                base_power
            }
        } else if move_id == "smellingsalts" {
            // ×2 vs a PARALYZED target (the onHit then CURES it — applied in the landed tail).
            if self.sides[foe].pokemon[foe_slot].status == Some(Status::Paralysis) {
                base_power.saturating_mul(2)
            } else {
                base_power
            }
        } else if move_id == "furycutter" {
            // The sim ADDS the volatile from INSIDE `basePowerCallback` and then reads it, so
            // the FIRST use is bp 10 × 1 and each CONSECUTIVE use doubles (multiplier `<< 1`
            // while `< 16`), clamped to 160. `duration: 2` refreshed on every restart, so one
            // non-Fury-Cutter turn lapses it.
            let mult = match self.sides[side].pokemon[slot].fury_cutter {
                None => 1u8,
                Some((m, _)) => {
                    if m < 16 {
                        m * 2
                    } else {
                        m
                    }
                }
            };
            self.sides[side].pokemon[slot].fury_cutter = Some((mult, 2));
            (base_power as u32 * mult as u32).clamp(1, 160) as u16
        } else if matches!(move_id.as_str(), "waterspout" | "eruption") {
            let mon = &self.sides[side].pokemon[slot];
            ((150u32 * mon.hp as u32) / mon.maxhp as u32).max(1) as u16
        } else {
            base_power
        };

        // --- HIDDEN POWER IV-derived BASE POWER (`gen3_iv_derived_hidden_power_bp_v1`):
        //     gen-3 computes HP's BP from the ATTACKER's IVs (`Dex.getHiddenPower`,
        //     range 30..=70), NOT the flat 70 the data ships (all 16 typed HP rows are
        //     BP 70 in `gen3_moves.json`). The port precomputes `MonState.hidden_power_bp`
        //     in `from_set`; override the data BP with it for any `hiddenpower*` id here
        //     (the same id-gate site as Water Spout / Pursuit). The TYPE stays from the
        //     typed id (`move_type` untouched). DRAW-NEUTRAL — a deterministic STATE read
        //     of a precomputed value, no PRNG, so seed goldens are unaffected; it only
        //     changes the damage magnitude for an HP mon whose IVs give BP != 70. ---
        let base_power = if move_id.starts_with("hiddenpower") {
            self.sides[side].pokemon[slot].hidden_power_bp as u16
        } else {
            base_power
        };

        // --- BARE HIDDEN POWER type + category resolution (`gen3_typed_hidden_power_ids_v1`
        //     / `gen3_iv_derived_hidden_power_bp_v1`, the round-12 pool-crash P0 fix): a
        //     packed gen3ou team can store the move SLOT as the BARE `hiddenpower` (num 237,
        //     data type Normal, BP 0), with the real HP type carried ONLY by the IVs (Showdown's
        //     `Pokemon` constructor resolves a bare HiddenPower slot to its typed variant at
        //     construction). The data-237 row derives category **Status** (BP 0) AND type
        //     **Normal** — BOTH WRONG — so without this the bp-0 Status mis-classification routes
        //     a real damaging Hidden Power into `run_status_move`'s fail-loud guard (the crash),
        //     and even if routed correctly it would deal Normal-type (not the mon's actual HP
        //     type) damage. Resolve the RUNTIME type from the attacker's IVs
        //     (`hidden_power_type`, the sibling of the BP override above) and RE-DERIVE the
        //     category from the overridden BP + that type — exactly like the variable-BP block
        //     immediately below, so the bare HP executes as the correct TYPED damaging move
        //     (right type, right BP, right phys/spec split). The TYPED ids (`hiddenpower<type>`,
        //     nums 355-370) already carry the right type+category, so gate on the BARE id only.
        //     `is_fire` (the frozen-defender thaw gate) was computed above and already excludes
        //     `hiddenpower*` by base id (a bare HP Fire must NOT thaw), so it stays correct; the
        //     reactive counter/mirrorcoat qualification keys on `starts_with("hiddenpower")`, so
        //     Counter still catches a physical-type bare HP and Mirror Coat never a bare HP —
        //     matching the sim's typed→bare collapse. DRAW-NEUTRAL (a deterministic IV read, no
        //     PRNG). ---
        // --- WEATHER BALL (`gen3_weather_ball_v1`) — the move's TYPE, BASE POWER and
        //     CATEGORY are all a function of the EFFECTIVE weather, resolved before the crit
        //     and damage draws, so the whole thing is DRAW-NEUTRAL. The dex row is Normal /
        //     bp 50 / Physical; under weather it becomes bp 100 and retypes:
        //         rain -> Water, sun -> Fire, sandstorm -> Rock, hail -> Ice.
        //
        //     ⚠️ THE CATEGORY FLIPS WITH THE TYPE, and that is the part a careless test cannot
        //     see. gen-3 splits phys/spec BY TYPE, so Rock (sandstorm) is PHYSICAL while
        //     Water/Fire/Ice are SPECIAL — and on the natural measurement board (a Mew mirror,
        //     base 100 across the board) Atk == SpA and Def == SpD, so the flip is INVISIBLE in
        //     the damage number. The discriminator that does work regardless of stats is
        //     COUNTER vs MIRROR COAT: under sandstorm Counter answers it and Mirror Coat does
        //     nothing; under rain the reverse (probe `harness/probe_weatherball2.js` section B).
        //     That is what the pin uses.
        //
        //     It reads `effective_weather()`, so a Cloud Nine / Air Lock holder on either side
        //     reverts it fully to Normal / bp 50 / Physical. Re-deriving the category through
        //     the shared `derive_category` (the Hidden Power precedent immediately below) is
        //     what makes the split fall out of the type rather than being a second id-list. ---
        let (move_type, base_power, category) = if move_id == "weatherball" {
            let wt = match self.effective_weather(dex) {
                Some(crate::state::Weather::Rain) => Some(Type::Water),
                Some(crate::state::Weather::Sun) => Some(Type::Fire),
                Some(crate::state::Weather::Sand) => Some(Type::Rock),
                Some(crate::state::Weather::Hail) => Some(Type::Ice),
                None => None,
            };
            match wt {
                Some(t) => (Some(t), 100u16, crate::dex::moves::derive_category(3, 100, Some(t))),
                // No effective weather: the dex row stands (Normal / 50 / Physical).
                None => (move_type, base_power, category),
            }
        } else {
            (move_type, base_power, category)
        };

        let (move_type, category) = if move_id == "hiddenpower" {
            let hp_type = Type::from_name(crate::state::hidden_power_type(
                &self.sides[side].pokemon[slot].set.ivs,
            ));
            (hp_type, crate::dex::moves::derive_category(3, base_power, hp_type))
        } else {
            (move_type, category)
        };

        // --- The VARIABLE-BP family (`gen3_move_coverage_batch5_v1`, `basePowerCallback`
        //     over a bp-0 data row — Return / Frustration / Flail / Reversal / Low Kick):
        //     compute the engine BP (see `variable_bp` — deterministic STATE reads, ZERO
        //     PRNG, probe-proven DRAW-NEUTRAL) and RE-DERIVE the category (the bp-0 data
        //     row mis-derived Status; the true gen3 category is the type split — all five
        //     are Physical [Normal / Fighting]). Everything downstream is the ordinary
        //     single-hit damaging draw model (acc-100 DRAWN, crit 1/16 — gen3 Flail /
        //     Reversal CAN crit, gen2's willCrit=false is NOT inherited — damage
        //     random(16), Quick Claw; NO secondary), with the normal type immunity
        //     (Normal→Ghost, Fighting→Ghost) reporting `-immune` AFTER the accuracy
        //     draw. `|| 1`-clamped, so bp is never 0 → never the `base_power == 0`
        //     no-op below. ---
        let (base_power, category) = match variable_bp(
            &move_id,
            &self.sides[side].pokemon[slot],
            &self.sides[foe].pokemon[foe_slot],
            dex,
        ) {
            Some(bp) => (bp, crate::dex::moves::derive_category(3, bp, move_type)),
            None => (base_power, category),
        };

        // --- THUNDER weather-accuracy mutation (`gen3_move_coverage_batch4b_v1`,
        //     `thunder.onModifyMove`): the id-gated gen3 rule REWRITES the BASE move.accuracy
        //     by the TARGET's `effectiveWeather()` BEFORE the accuracy pipeline runs —
        //     effective RAIN → `accuracy = true` (never-miss: the WHOLE accuracy block AND its
        //     `randomChance` are SKIPPED, so ONE FEWER draw), effective SUN → base 50, else
        //     (none / sand / hail / Cloud-Nine-or-Air-Lock-SUPPRESSED) → base 70 unchanged.
        //     The acc/eva stages + accMod (Bright Powder / Sand Veil / …) then fold ON TOP of
        //     this weather-set base via `effective_accuracy` (weather FIRST, accMod SECOND); in
        //     rain the `never_miss` short-circuits the entire stage+accMod chain (the
        //     `accuracy === true` guard), so a Bright Powder is inert and still 0 acc draws.
        //     `effective_weather()` already zeroes under Cloud Nine / Air Lock (→ base 70),
        //     matching the probe. VERIFIED bit-for-bit vs `harness/probe_batch4b_thunder.js`
        //     (rain 5 draws → seed 22534,42410,55299,35327; sun/base/sand/suppressed 2 draws →
        //     seed 60880,31090,7619,34922 — the accuracy `random(100)` is the SAME consumption
        //     regardless of the numerator, only the miss threshold differs). ---
        let (never_miss, accuracy) = if move_id == "thunder" {
            match self.effective_weather(dex) {
                Some(Weather::Rain) => (true, accuracy),
                Some(Weather::Sun) => (false, 50),
                _ => (never_miss, accuracy),
            }
        } else {
            (never_miss, accuracy)
        };

        // --- onBeforeMove STATUS draws (BEFORE accuracy), mirroring
        //     runEvent('BeforeMove') at runMove (battle-actions.ts:255), which
        //     precedes useMove/PP/accuracy. Handlers run priority-DESC (sleep 10,
        //     freeze 10, flinch 8, confusion 3, par 1), SHORT-CIRCUITING on the first
        //     abort (a lower-priority status then never draws). A move that aborts
        //     here draws NOTHING further — no accuracy/crit/damage/secondary.
        //     SKIPPED for the pursuit strike (a bare useMove; the pursuer already acted). ---
        // --- The LOCKED FIRE turn of a two-turn move (`gen3_move_coverage_batch4c_v1`,
        //     Solar Beam): the mon is CHARGING (`two_turn.charging`) and this action is
        //     the locked fire. The fire deducts NO PP (the sim's `useMoveInner` lockedmove
        //     path skips `deductPP`); everything else (onBeforeMove first — an abort
        //     LOSES the charge — then normal accuracy/crit/damage) runs below. Computed
        //     BEFORE the on_before_move/PP block reads it. ---
        let locked_fire = !pursuit_strike
            && move_id == "solarbeam"
            && self.sides[side].pokemon[slot].two_turn.map_or(false, |t| t.charging);
        // UPROAR spends PP ONCE, on the CAST (`gen3_uproar_v1`, probed 16 -> 15 across a
        // 5-turn lock) — every CONTINUING turn is a lockedmove and skips deductPP, exactly
        // like Solar Beam's fire turn.
        // --- IMPRISON (`gen3_imprison_v1`): a queued move the FOE has imprisoned is CANT-ed
        //     here — DRAW-FREE, and NO PP is spent (the block sits before the deduct, which is
        //     why it must precede the PP region rather than ride the immunity short-circuit).
        //     [EMIT] `|cant|<user>|move: Imprison|<Move>`. ---
        if !pursuit_strike && !sleep_talk_call && self.imprisoned_for(side, &move_id, dex) {
            if self.logging() {
                let u = self.mon_ref(side, slot, dex);
                let name = dex.moves(&move_id).map(|m| m.display_name().to_string()).unwrap_or_default();
                self.log.cant(&u, "move: Imprison", Some(&name));
            }
            return MoveResolution::done(false, false, false);
        }

        let locked_uproar =
            !pursuit_strike && move_id == "uproar" && self.sides[side].pokemon[slot].uproar.is_some();
        // OUTRAGE / PETAL DANCE / THRASH likewise spend PP once, on the CAST.
        // ROLLOUT / ICE BALL also spend PP once, on the first execution.
        let locked_rollout = !pursuit_strike
            && matches!(move_id.as_str(), "rollout" | "iceball")
            && self.sides[side].pokemon[slot].rollout.is_some();
        let locked_lockin = !pursuit_strike
            && matches!(move_id.as_str(), "outrage" | "petaldance" | "thrash")
            && self.sides[side].pokemon[slot].locked_move.is_some();

        // The PRE-move Choice-lock snapshot (`gen3_move_coverage_batch5_v1`, the Sleep
        // Talk `onTryHit` choicelock gate): the lock the PP block below sets for THIS
        // very use must NOT count (a CB mon's FIRST Sleep Talk samples + executes —
        // probed; only a lock from a PREVIOUS turn fails the next Sleep Talk).
        let was_choice_locked = self.sides[side].pokemon[slot].choice_locked_move.is_some();

        if !pursuit_strike && !sleep_talk_call {
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
            //
            // `twoturnmove.onMoveAborted` (`gen3_move_coverage_batch4c_v1`): a cant on the
            // FIRE turn removes BOTH the twoturnmove volatile and its `solarbeam` sub —
            // THE CHARGE IS LOST (probed: Spore on the fire turn → `|cant|slp`, vols=[];
            // on wake the mon starts a FRESH charge and pays PP again). A charge-turn cant
            // has no volatile yet (it is added below, after this gate) — the clear is a
            // no-op then. DRAW-FREE.
            self.sides[side].pokemon[slot].two_turn = None;
            // DESTINY BOND's `onMoveAborted` (`gen3_move_coverage_batch6_v1`): ANY cant
            // (full-para / slp / frz / flinch / confusion self-hit / disable / taunt)
            // removes a still-up DB volatile — the window closes at the move ATTEMPT
            // whether or not the move ran (probe DB2's class; draw-free). The CHARGE
            // volatile's own `onMoveAborted` (any move != charge) is consumed by the
            // CALLER (turn_loop's post-run_move consumption — the abort path included).
            self.sides[side].pokemon[slot].destiny_bond = false;
            // ABORTED (cant) — the sim runs `MoveAborted` and returns BEFORE `AfterMove`, so
            // the caller must SKIP the `onAnyAfterMove` White Herb restore this move.
            return MoveResolution::aborted();
        }

        // --- DESTINY BOND's `onBeforeMove` (priority −1, `gen3_move_coverage_batch6_v1`):
        //     the LOWEST-priority BeforeMove handler — it runs only after every cant gate
        //     PASSED (an abort was handled above via onMoveAborted) and removes a still-up
        //     DB volatile for any move `!= destinybond` (a DB RE-CAST keeps it up through
        //     its own attempt — the onPrepareHit then draw-free re-adds it, DB6). So the
        //     DB window is exactly "until the user's next move ATTEMPT" (probe DB2: the
        //     user splashed then was KO'd the same turn → NO mutual faint). DRAW-FREE. ---
        if move_id != "destinybond" {
            self.sides[side].pokemon[slot].destiny_bond = false;
        }

        // --- PP DEDUCTION (`gen3_pp_tracking_v1`), mirroring `deductPP` at
        //     battle-actions.ts:282 (right AFTER BeforeMove passes, BEFORE accuracy).
        //     DRAW-FREE. The mon's OWN move deducts 1 from its used slot; a foe holding
        //     **Pressure** deducts 2 (the `runEvent('DeductPP')` extra, battle-actions.ts:
        //     472-483 — VERIFIED −2, no RNG). Struggle deducts NOTHING (it is not a slot).
        //     The Pressure extra fires ONLY when the Pressure FOE is in the move's
        //     **`pressureTargets`** (`getMoveTargets`, pokemon.ts:854-861) — i.e. the move
        //     TARGETS a foe. That is NOT the same as `!targets_self`: an **`allyTeam`** move
        //     (Aromatherapy / Heal Bell) or an `allySide` / `allies` / `foeSide` move does
        //     NOT put the foe in `pressureTargets`, so it deducts only 1 even under a
        //     Pressure foe — `pressure_targets_foe` (below) mirrors the exact target rule
        //     (`gen3_pressure_allyteam_v1`, VERIFIED vs the sim: Blissey's Aromatherapy under
        //     a Pressure Zapdos deducts 1, not 2 — the e2e_182 root cause). A self-target
        //     heal/setup/protect (SoftBoiled / Calm Mind / Protect) is `pressureTargets=[self]`
        //     → also 1.
        //     A LOCKED FIRE (Solar Beam's second turn) deducts NO PP — the sim's
        //     `useMoveInner` lockedmove path skips `deductPP` (`gen3_move_coverage_
        //     batch4c_v1`; probed: PP is paid ONCE, at the CHARGE — 16→15, or 16→14 under
        //     a Pressure foe — and the fire turn leaves it untouched).
        if !struggle && !locked_fire && !locked_uproar && !locked_lockin && !locked_rollout {
            let pressure_extra = pressure_targets_foe
                && to_id(&self.sides[foe].pokemon[foe_slot].ability) == "pressure";
            let deduct = if pressure_extra { 2 } else { 1 };
            self.sides[side].pokemon[slot].deduct_pp(move_index, deduct);

            // --- CHOICE LOCK: NOT set here. `gen3_choicelock_after_move_v1` moved it to the
            //     AfterMove site in `turn/driver.rs` (the sim adds the `choicelock` volatile from
            //     `choiceband.onAfterMove`, i.e. AFTER the move body — see the comment there).
            //     Setting it at PP-deduct time read the item ONE PHASE TOO EARLY and so missed
            //     every move that CHANGES the user's own item while resolving (Thief / Covet /
            //     Trick). ---
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
        // `lastMoveUsed` (`gen3_conversion_v1`) — an ID, not a slot, and set for a STRUGGLE too
        // (where `last_move` is deliberately None). Conversion 2 special-cases "struggle" to
        // Normal and gen-3 struggle's dex type is ALREADY Normal, so that arm is a no-op either
        // way. Sitting inside the same guard is what makes a Sleep-Talk-called move leave the
        // OUTER move recorded, matching the sim.
        self.sides[side].pokemon[slot].last_move_used = Some(if struggle {
            "struggle".to_string()
        } else {
            move_id.clone()
        });
        // `gen3_mimic_disable_self_overwrite_v1`: reset the self-overwrite flag for EVERY move;
        // the Mimic success block re-sets it TRUE after it overlays its own slot.
        self.sides[side].pokemon[slot].last_move_was_self_overwrite = false;
        // RAGE's volatile is removed by the HOLDER's OWN next move (`onBeforeMove` at
        // priority 100) — so the boost window is exactly "from casting Rage until I act
        // again". Cleared AFTER the Rage arm below re-sets it for a fresh cast.
        self.sides[side].pokemon[slot].rage = false;
        } // end `if !pursuit_strike && !sleep_talk_call` (on_before_move + PP + lastMove)

        // --- FOCUS PUNCH onTry cancel (`gen3_move_coverage_batch4_v1`,
        //     `focuspunch.move.onTry`): fires at the `singleEvent('Try')` step INSIDE
        //     `useMoveInner` (battle-actions.ts:489) — AFTER PP/lastMove (above) but BEFORE
        //     accuracy. If the user's `focuspunch` volatile has `lost_focus` (it was HIT by a
        //     non-Status move earlier this turn), the punch is CANCELLED draw-free: the sim
        //     retro-edits the `|move|` announce to `[still]` + emits `|cant|<user>|Focus
        //     Punch|Focus Punch`, and returns null (NOT landed → no in-tryMoveHit Update, no
        //     acc/crit/dmg). PP + lastMove were already consumed (the deductPP/moveUsed at
        //     runMove precede onTry — VERIFIED vs the sim). ---
        if move_id == "focuspunch" && self.sides[side].pokemon[slot].focus_punch == Some(true) {
            if self.logging() {
                let user = self.mon_ref(side, slot, dex);
                // `|move|<user>|Focus Punch||[still]` (attrLastMove('[still]')) then
                // `|cant|<user>|Focus Punch|Focus Punch`.
                self.log.move_used(&user, &move_name, None, false, true);
                self.log.cant(&user, "Focus Punch", Some("Focus Punch"));
            }
            return MoveResolution::done(false, false, false);
        }

        // --- FUTURE MOVE cast (`gen3_move_coverage_batch4c_v1`, Doom Desire / Future
        //     Sight — the gen-3 `onTry`, which fires BEFORE tryMoveHit's accuracy AND
        //     before the TryHit protect check, so a cast-turn Protect does NOT block it):
        //     queue the CAST-TIME DAMAGE SNAPSHOT as the target slot's pending strike.
        //     PP + lastMove were already consumed above (deductPP/moveUsed precede onTry
        //     — probed: a FAILED double-cast still deducts PP, 7→6). ---
        if move_id == "doomdesire" || move_id == "futuresight" {
            return self.run_future_move_cast(
                side, slot, foe, foe_slot, base_power, category, accuracy, &move_id,
                &move_name, dex,
            );
        }

        // --- SOLAR BEAM (`gen3_move_coverage_batch4c_v1`) — the two-turn CHARGE move
        //     (`onTryMove`, which runs AFTER PP was deducted above — the CHARGE turn pays
        //     the PP; the fire turn paid none via `locked_fire`). Probe
        //     `harness/probe_batch4c_solarbeam.js`:
        //       * CHARGE (no volatile yet): emit `|move|<user>|Solar Beam||[still]` +
        //         `|-prepare|<user>|Solar Beam`, then check the user's EFFECTIVE weather
        //         (Cloud Nine / Air Lock suppress — a negater forces the charge back even
        //         under Drought sun, probed): SUN → the charge is SKIPPED — emit
        //         `|-anim|<user>|Solar Beam|<target>` and fall through to the NORMAL
        //         damaging execution (acc+crit+dmg — 3 draws exactly like a normal move,
        //         NO volatile); else add the `twoturnmove` volatile (duration 2, charging)
        //         and return null — ZERO move draws, `landed` false (no in-tryMoveHit
        //         Update).
        //       * FIRE (`locked_fire`): `removeVolatile('solarbeam')` (the sub-volatile —
        //         `charging` = false; the twoturnmove itself lingers to its residual
        //         expiry) → NORMAL execution below with the `|[from]lockedmove` announce
        //         attr. Accuracy 100 is DRAWN (always passes); crit + damage follow; NO
        //         secondary.
        //     `suppress_announce` marks the sun-skip (its `|move|` line already emitted in
        //     the `[still]`+`-prepare`+`-anim` form); `announce_lockedmove` marks the fire
        //     (the downstream announce gains the lockedmove attr). ---
        let mut suppress_announce = false;
        let mut announce_lockedmove = false;
        if move_id == "solarbeam" {
            if locked_fire {
                if let Some(t) = self.sides[side].pokemon[slot].two_turn.as_mut() {
                    t.charging = false; // removeVolatile('solarbeam') → the beam FIRES
                }
                announce_lockedmove = true;
            } else {
                // CHARGE turn (or the sun skip). [EMIT] the `[still]` announce + -prepare.
                if self.logging() {
                    let user = self.mon_ref(side, slot, dex);
                    self.log.move_used(&user, &move_name, None, false, true);
                    self.log.prepare(&user, &move_name);
                }
                if self.effective_weather(dex) == Some(Weather::Sun) {
                    // SUN SKIP: -anim then IMMEDIATE normal execution (PP already paid -1).
                    if self.logging() {
                        let user = self.mon_ref(side, slot, dex);
                        let target = self.mon_ref(foe, foe_slot, dex);
                        self.log.anim(&user, &move_name, &target);
                    }
                    suppress_announce = true;
                } else {
                    self.sides[side].pokemon[slot].two_turn = Some(TwoTurnMove {
                        move_index,
                        duration: 2,
                        charging: true,
                    });
                    return MoveResolution::done(false, false, false);
                }
            }
        }

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
                will_act,
                was_choice_locked,
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
                // [EMIT] the sim announces the move FIRST (`useMoveInner` emits `|move|<user>|
                // <Move>|<target>` before `runEvent('TryMove')`), then Damp's `onAnyTryMove`
                // retro-edits it to the `[still]` did-nothing form (`attrLastMove('[still]')`,
                // blank target + append `|[still]`) and adds the `|cant|`. So the golden is TWO
                // lines: `|move|<user>|<Move>||[still]` then the cant — we emit the `[still]`
                // announce directly (`move_used(None, still=true)` renders the identical bytes).
                // The CANT is on the DAMP HOLDER (the sim's `this.effectState.target`), the move
                // name, and `[of]` the move's user. Observation-only; the move draws nothing.
                if self.logging() {
                    let user = self.mon_ref(side, slot, dex);
                    let holder = self.mon_ref(damp_side, damp_slot, dex);
                    self.log.move_used(&user, &move_name, None, false, true);
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

        // --- FAKE OUT's FIRST-TURN GATE (`gen3_fakeout_v1`). The sim runs this as the move's
        //     `onTry` at gen-3 `tryMoveHit`'s `singleEvent("Try", move)` — BEFORE invulnerability,
        //     the type-immunity report, the accuracy roll and TryHit/Protect. So a blocked Fake
        //     Out into a GHOST prints ONLY the hint, never `-immune`.
        //
        //     ZERO draws on the block, but PP IS already paid (and doubled under Pressure — the
        //     deduction precedes the Try gate). The failure form is UNIQUE: the announce plus a
        //     `|-hint|`, with NO `[still]` attr and NO `-fail`. `once = false`, so a repeat-spammed
        //     Fake Out emits one hint EVERY time — the sim never dedupes it.
        //
        //     The predicate is `active_move_actions > 1` AFTER this run's increment: probe-settled
        //     that a CANT turn burns the gate while a CANCELLED action does not. `landed = false`.
        if move_id == "fakeout" && self.sides[side].pokemon[slot].active_move_actions > 1 {
            if self.logging() {
                let user = self.mon_ref(side, slot, dex);
                let target = self.mon_ref(foe, foe_slot, dex);
                self.log.move_used(&user, &move_name, Some(&target), false, false);
                self.log.hint("Fake Out only works on your first turn out.", false);
            }
            return MoveResolution::done(false, false, false);
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
            if self.logging() && !suppress_announce {
                let user = self.mon_ref(side, slot, dex);
                let target = self.mon_ref(foe, foe_slot, dex);
                self.log.move_used(&user, &move_name, Some(&target), false, false);
                if announce_lockedmove {
                    self.log.attr_last_move_from_lockedmove();
                }
                self.log.activate(&target, "Protect", None);
            } else if self.logging() {
                // A sun-skip Solar Beam into a Protect: the announce already emitted in
                // the [still]+prepare+anim form; only the block line follows.
                let target = self.mon_ref(foe, foe_slot, dex);
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

        // --- SOUNDPROOF (`gen3_ability_batch2_v1`, `soundproof.onTryHit`): a DAMAGING
        //     `flags.sound` move (Hyper Voice / Uproar) into a Soundproof holder is IMMUNE —
        //     the mirror of the STATUS-move Soundproof gate in `run_status_move` (Sing / Grass
        //     Whistle). It fires at the `TryHit` event AFTER the accuracy roll (`acc_hit`-gated,
        //     so a MISS never reaches TryHit → the genuine-miss `-miss` path below) and BEFORE
        //     the crit/damage draws — so a BLOCKED move draws ONLY its accuracy roll (EXACTLY a
        //     type-immune / Wonder-Guard-blocked move's draw count), then `-immune`. PROBE-SETTLED
        //     bit-for-bit vs the sim (`harness/probe_soundproof_damaging.js`): Hyper Voice into a
        //     Soundproof Mr. Mime draws `random(100)`(accuracy) then `-immune`, NO crit / damage
        //     roll (4 draws vs the 7-draw non-Soundproof control). It sits after the Protect block
        //     (Protect wins TryHit) and before the type-immunity short-circuit (matching the
        //     status path's Protect → Soundproof → naturalImmunity order); the `move_is_sound`
        //     helper already existed but was UNUSED on this damaging path. Emission-only past the
        //     accuracy roll (draw-free). ---
        if acc_hit
            && self.move_is_sound(&move_id, dex)
            && dex
                .ability(&to_id(&self.sides[foe].pokemon[foe_slot].ability))
                .map(|a| a.blocks_sound)
                .unwrap_or(false)
        {
            // [EMIT] `|move|<user>|<Name>|<foe>` then `|-immune|<foe>|[from] ability: Soundproof`.
            // The `-immune` line ALWAYS emits; only the `|move|` announce is skipped under
            // `suppress_announce` (a SUN-SKIP Solar Beam already emitted its
            // `[still]`+`-prepare`+`-anim` form) — the same split the immune / miss paths use.
            if self.logging() {
                let target = self.mon_ref(foe, foe_slot, dex);
                if !suppress_announce {
                    let user = self.mon_ref(side, slot, dex);
                    self.log.move_used(&user, &move_name, Some(&target), false, false);
                    if announce_lockedmove {
                        self.log.attr_last_move_from_lockedmove();
                    }
                }
                self.log.immune_from_ability(&target, "Soundproof");
            }
            // acc_hit is true here → not missed, not landed (no in-tryMoveHit Update).
            return MoveResolution::done(false, false, false);
        }

        // --- Build the DamageContext (no PRNG; resolves the gen-3 ability/Levitate +
        //     type immunities + stat mods + weather). ---
        let mut ctx = self.build_damage_context(
            side, slot, foe, foe_slot, base_power, move_type, category, halves_def, dex,
        );
        // MINIMIZE (`gen3_minimize_v1`): the MOVE half of the pair. `build_damage_context`
        // has no move id in scope, so its `minimize_doubles` defaults false and is set here.
        // The gen-3-legal carriers are exactly stomp / astonish / extrasensory / needlearm —
        // read from the dex FLAG, not a hand-list, because bodyslam & co. gained the flag in
        // gen 9 and a list written from modern knowledge would be wrong. None of the four is
        // multi-hit or fixed-damage, so this single-hit site is the whole surface.
        ctx.mv.minimize_doubles = dex
            .moves(&to_id(&move_id))
            .map(|m| m.minimize_doubles)
            .unwrap_or(false);

        // --- WONDER GUARD (`gen3_wonder_guard_v1`): the SE-ONLY damage gate. The gen4-override
        //     (gen3-inherited) `onTryHit` blocks a DAMAGING move into a Wonder Guard holder unless
        //     it is STRICTLY super-effective (`runEffectiveness(move) > 0`) AND not type-immune,
        //     emitting `-immune|<t>|[from] ability: Wonder Guard`. It runs at the `TryHit` event —
        //     AFTER the accuracy roll (already drawn; `acc_hit`-gated, since a MISS never reaches
        //     TryHit → the genuine-miss `-miss` return below) and BEFORE the crit/damage/secondary
        //     draws — so a BLOCKED move draws ONLY its accuracy roll (EXACTLY a type-immune move's
        //     draw count), then `-immune`. It fires BEFORE the plain type-immunity short-circuit
        //     (`move_is_immune`), so a 0×-type-immune move ALSO routes through WG's `-immune` (a
        //     DISTINCT byte form from a plain type `-immune` — probe-confirmed vs the sim: Tackle
        //     into Shedinja shows `[from] ability: Wonder Guard`). BYPASSED (WG never gates them):
        //     Status moves (this is the damaging path, so `category == Status` never reaches here),
        //     a SELF-target hit (`targets_self`), a TYPELESS `???`/Struggle move
        //     (`move_type.is_none()`), and ALL residual damage (a MOVE hook only — the Leech Seed /
        //     weather chip / burn / poison residuals bypass it, so a 1-HP Shedinja still dies to a
        //     residual). The gen3 `runEffectiveness` log2-SUM `> 0` (STRICTLY SE) is bit-equivalent
        //     to the type-chart effectiveness PRODUCT `> 1.0` (each per-type factor ∈ {0.5,1,2}; a
        //     0× factor makes the product 0 → not `> 1.0` → the WG `-immune` form). No new state (a
        //     read-only incoming-move gate; a Traced Wonder Guard reads its LIVE copied ability). ---
        if acc_hit
            && category != MoveCategory::Status
            && !targets_self
            && move_type.is_some()
            && dex
                .ability(&self.sides[foe].pokemon[foe_slot].ability)
                .map(|a| a.wonder_guard)
                .unwrap_or(false)
        {
            let connects = match move_type {
                Some(t) => dex.type_chart().effectiveness(t, &ctx.defender.types) > 1.0,
                None => true, // unreachable (guarded by move_type.is_some() above)
            };
            if !connects {
                // [EMIT] `|move|<user>|<Name>|<foe>` then `|-immune|<foe>|[from] ability: Wonder
                // Guard`. Observation-only: draws nothing beyond the accuracy roll already drawn.
                // The `-immune` line ALWAYS emits; only the `|move|` announce is skipped under
                // `suppress_announce` (a SUN-SKIP Solar Beam already emitted its
                // `[still]`+`-prepare`+`-anim` form). SIM-PROBED (`harness/probe_rb_tail.js` S1a):
                // a sun-skipped Solar Beam into Shedinja emits `|move|…||[still]` → `|-prepare|`
                // → `|-anim|` → `|-immune|p1a: Shedinja|[from] ability: Wonder Guard`; the port
                // used to swallow the `-immune` with the announce.
                if self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    if !suppress_announce {
                        let user = self.mon_ref(side, slot, dex);
                        self.log.move_used(&user, &move_name, Some(&target), false, false);
                        if announce_lockedmove {
                            self.log.attr_last_move_from_lockedmove();
                        }
                    }
                    self.log.immune_from_ability(&target, "Wonder Guard");
                }
                // acc_hit is true here → not missed, not landed (no in-tryMoveHit Update).
                return MoveResolution::done(false, false, false);
            }
        }

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

        // --- CHARGE ×2 on the next Electric move (`gen3_move_coverage_batch6_v1`, the
        //     `charge` volatile's `onBasePower chainModify(2)` iff `move.type ===
        //     'Electric'` — probe CH1: control Thunderbolt 117, charged 208, post-charge
        //     back to ~114 at IDENTICAL draw counts). A BASE-POWER chain member (joins
        //     the ONE accumulated 4096 modifier — ×2 is exact under the chain rounding,
        //     so its accumulate position is commutative). The volatile is CONSUMED by
        //     the user's next move attempt of ANY kind (onAfterMove/onMoveAborted —
        //     handled by the CALLER after this move resolves), so the fold reads the
        //     still-up flag. gen3 Charge has NO +1 SpD. DRAW-FREE. ---
        if self.sides[side].pokemon[slot].charge && move_type == Some(Type::Electric) {
            ctx.bp_mods.push(BpMod::Chain(2, 1));
        }

        // --- SOLAR BEAM weather BP-halving (`gen3_move_coverage_batch4c_v1` — gen3 DOES
        //     have the modern halving, PROBED: rain 54 vs no-weather 105 on the same
        //     Kyogre, and in sand): the resolved gen3 `solarbeam.onBasePower`
        //     `chainModify(0.5)` when the USER's `effectiveWeather()` is rain / sandstorm
        //     / hail (+primordialsea/snowscape — N/A gen3). Suppression-aware (a Cloud
        //     Nine / Air Lock mon kills the halving too — `effective_weather`), read at
        //     DAMAGE time (the fire turn / the sun-skip). DRAW-FREE (a BP-chain fold —
        //     state-only, seed-identical to the unhalved control). ---
        if move_id == "solarbeam"
            && matches!(
                self.effective_weather(dex),
                Some(Weather::Rain | Weather::Sand | Weather::Hail)
            )
        {
            ctx.bp_mods.push(BpMod::Chain(1, 2));
        }

        // --- IMMUNITY short-circuit (the draw-COUNT crux): a type-chart 0× or an
        //     ability/Levitate immunity. gen3 `tryMoveHit` knows immunity up front,
        //     draws accuracy UNCONDITIONALLY, then emits `-immune` and returns — so an
        //     immune move draws ONLY accuracy (NO crit, NO damage roll) and reports
        //     IMMUNE (not missed), regardless of the accuracy roll (verified vs the
        //     sim: Earthquake into a Flying/Levitate mon draws `randomChance(100,100)`
        //     then `-immune`). Drawing crit/damage here would desync every later draw. ---
        // --- DREAM EATER's `onTryImmunity` (`gen3_bp_modifier_cluster_v1`): the target must
        //     be ASLEEP and NOT behind a Substitute. It resolves at `runImmunity`, i.e. the
        //     PRE-accuracy class (like Levitate), so it reports `|-immune|<target>` with a
        //     BARE form (no `[from]`) — probe-confirmed against an awake foe. Folded in beside
        //     the type/ability immunity so the whole short-circuit (accuracy drawn, then
        //     `-immune`, NO crit/damage roll) is shared. ---
        let dream_eater_blocked = move_id == "dreameater"
            && !(matches!(self.sides[foe].pokemon[foe_slot].status, Some(Status::Sleep(_)))
                && self.sides[foe].pokemon[foe_slot].substitute.is_none());
        if dream_eater_blocked || move_is_immune(&ctx, dex) {
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
            // Apply the WATER/VOLT ABSORB heal (state) FIRST when the move HIT, so the emit
            // can render `-heal` (the heal landed) vs `-immune` (full HP). `apply_absorb_heal`
            // is a no-op / false for a non-absorb ability. (The separate `acc_hit` block below
            // used to call this — the heal moved here so the emit sees its result.)
            let absorb_healed = if acc_hit && absorb_name.is_some() {
                self.apply_absorb_heal(foe, foe_slot, move_type)
            } else {
                false
            };
            if self.logging() {
                let user = self.mon_ref(side, slot, dex);
                let target = self.mon_ref(foe, foe_slot, dex);
                if !suppress_announce {
                    self.log.move_used(&user, &move_name, Some(&target), false, false);
                    if announce_lockedmove {
                        self.log.attr_last_move_from_lockedmove();
                    }
                }
                let fm = &self.sides[foe].pokemon[foe_slot];
                let ff_case = is_tryhit_ff && acc_hit;
                if ff_case {
                    if fm.flash_fire {
                        self.log.immune_from_ability(&target, "Flash Fire");
                    } else {
                        self.log.volatile_start(&target, "ability: Flash Fire");
                    }
                } else if let Some(name) = absorb_name.filter(|_| acc_hit) {
                    // FORM 9: a LANDED absorb heal → `-heal|<t>|<HP>|[from] ability: <Name>|[of]
                    // <user>`; a full-HP absorb (no heal) → `-immune|<t>|[from] ability: <Name>`.
                    if absorb_healed {
                        let hp = self.hp_status(foe, foe_slot);
                        self.log.heal_of(&target, &hp, &Cause::Ability(name.to_string()), &user);
                    } else {
                        self.log.immune_from_ability(&target, name);
                    }
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
                // (WATER/VOLT ABSORB heal already applied above — it feeds the `-heal`/`-immune`
                // emit choice, FORM 9.)
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
                if !suppress_announce {
                    if announce_lockedmove {
                        // A locked Solar Beam fire-turn MISS: the sim appends `[from] lockedmove`
                        // (in the useMove announce) BEFORE the `[miss]` (in the accuracy check) →
                        // `|move|…|[from] lockedmove|[miss]`. So emit the BARE move line, the
                        // lockedmove attr, THEN the miss attr (NOT `move_used(miss=true)`, which
                        // would put `[miss]` first). SIM-PROBED (`gen3_omniscient_byte_fuzz_v1`).
                        self.log.move_used(&user, &move_name, Some(&target), false, false);
                        self.log.attr_last_move_from_lockedmove();
                        self.log.attr_last_move_miss();
                    } else {
                        self.log.move_used(&user, &move_name, Some(&target), true, false);
                    }
                } else {
                    // SUN-SKIP Solar Beam (`suppress_announce`): its `[still]`+`-prepare`+`-anim`
                    // lines already emitted BEFORE the accuracy roll. A MISS appends `[miss]` to
                    // the last move-family line — the `|-anim|` — via `attrLastMove('[miss]')`
                    // (`|-anim|<u>|Solar Beam|<t>|[miss]`, golden ab_34_10 line 120), THEN `-miss`.
                    self.log.attr_last_move_miss();
                }
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

        // --- BEAT UP (`gen3_move_coverage_batch4b_v1`) — a MULTI-STRIKE move: ONE strike
        //     PER healthy (non-fainted, NON-STATUSED) party member of the USER's side, each a
        //     TYPELESS flat-BP-10 Special hit with the STAT SWAP (the ally's dex `baseStats.atk`
        //     → the attacker's SpA, the target's dex `baseStats.def` → the defender's SpD, both
        //     at `event.modifier=1` → NO boosts / items / CB / burn / abilities touch the stat).
        //     The whole-move accuracy roll already drew here (acc 100 → always passes; Beat Up
        //     is never immune [typeless '???'] and was not protect-blocked), so this is only
        //     reached on a hit — the per-strike crit+damage draws + the KO-mid-sequence break
        //     live in `run_beat_up`. It emits its OWN `|move|` announce (no effectiveness line),
        //     so branch BEFORE the standard `|move|` emit below. ---
        if move_id == "beatup" {
            return self.run_beat_up(side, slot, foe, foe_slot, crit_ratio, &move_name, dex);
        }

        // --- MULTI-STRIKE moves (`gen3_move_coverage_batch7_v1`) — Double Kick / Twineedle /
        //     Bonemerang (fixed 2), Triple Kick (fixed 3, multiaccuracy → FAIL-LOUD in
        //     `run_multihit`), and the variable [2,5] family (Pin Missile / Bullet Seed / Icicle
        //     Spear / Rock Blast / Barrage / Comet Punch / Double Slap / Spike Cannon / Arm
        //     Thrust / Fury Attack / Fury Swipes / Bone Rush). Routed here (like Beat Up) AFTER
        //     the shared accuracy + immunity/protect checks, BEFORE the single-hit block: each
        //     strike runs the NORMAL damage path over the `ctx` `run_move` already built. The
        //     whole-move accuracy roll already drew (`acc_hit`); a variable count draws ONE more
        //     `sample` HERE, then per strike crit + `random(16)` + the move's per-strike
        //     secondary. ---
        if let Some(mh) = self.move_at(side, slot, move_index, dex).and_then(|m| m.multihit) {
            let multiaccuracy = self
                .move_at(side, slot, move_index, dex)
                .map(|m| m.multiaccuracy)
                .unwrap_or(false);
            return self.run_multihit(
                side, slot, foe, foe_slot, ctx, move_type, category, move_index, crit_ratio,
                &move_id, &move_name, is_contact, is_fire, suppress_announce, mh, multiaccuracy,
                dex,
            );
        }

        // --- BRICK BREAK screen-break (RM1, `gen3_brick_break_screens_v1`) — Brick Break is
        //     the ONLY gen3 screen-breaking move. Its `onTryHit` removes BOTH the FOE side's
        //     screens (Reflect + Light Screen) BEFORE the damage step, draw-free. Gated on a
        //     CONFIRMED-LANDED, non-immune (the immunity short-circuit above — probe: Brick
        //     Break into a Ghost does NOT remove the screen, since `runImmunity` precedes
        //     `onTryHit`), non-protect-blocked, non-miss hit (both the protect block and the
        //     genuine-miss `!acc_hit` returns are above). Clearing the STATE (sides[foe]
        //     screens) drops them from the `modify_damage_phase1_shuffle` count below (it counts
        //     every screen up across BOTH sides) — the sim removes both foe screens in `onTryHit`
        //     before ModifyDamagePhase1, so those handlers no longer gather; and
        //     clearing the CTX makes `calc_damage` compute screen-free full damage. Both are
        //     probe-confirmed vs the sim (`harness/probe_brickbreak_screens.js`). ---
        let (bb_removed_reflect, bb_removed_ls) = if move_id == "brickbreak" {
            let rr = self.sides[foe].reflect > 0;
            let rl = self.sides[foe].light_screen > 0;
            if rr {
                self.sides[foe].reflect = 0;
            }
            if rl {
                self.sides[foe].light_screen = 0;
            }
            ctx.reflect = false;
            ctx.light_screen = false;
            (rr, rl)
        } else {
            (false, false)
        };

        // [EMIT] `|move|<user>|<Name>|<foe>` (a landed damaging move — the effectiveness
        // / crit / damage lines follow). Emitted BEFORE the crit roll so the line order
        // matches the sim; the emit reads already-resolved state (draws nothing).
        if self.logging() {
            let user = self.mon_ref(side, slot, dex);
            let target = self.mon_ref(foe, foe_slot, dex);
            if !suppress_announce {
                self.log.move_used(&user, &move_name, Some(&target), false, false);
                if announce_lockedmove {
                    self.log.attr_last_move_from_lockedmove();
                }
                // BRICK BREAK (RM1): the screen `|-sideend|` lines emit BETWEEN the `|move|`
                // line and the `|-supereffective|` line (probe-confirmed order: |move| then
                // |-sideend| then |-supereffective|), Reflect first then Light Screen, on the
                // FOE side ref, with the exact tokens the residual-expiry path uses.
                if bb_removed_reflect || bb_removed_ls {
                    let side_ref =
                        crate::protocol::ProtocolBuilder::side_ref(foe, &self.sides[foe].name);
                    if bb_removed_reflect {
                        self.log.sideend(&side_ref, "Reflect");
                    }
                    if bb_removed_ls {
                        self.log.sideend(&side_ref, "move: Light Screen");
                    }
                }
            }
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
        //     damaging, non-immune move (critRatio >= 1). After accuracy, before damage.
        //     `effective_crit_ratio` folds the `onModifyCritRatio` handlers (FOCUS ENERGY +2
        //     — Lansat-berry-only; a CRIT_ITEM +N — Scope Lens / Lucky Punch / Stick) then
        //     `clampIntRange(_, 0, 5)`, so the DENOMINATOR shifts (1→3 ⇒ 1/4; a high-crit
        //     2→4 ⇒ 1/3) while the draw COUNT is unchanged. ---
        let eff_crit_ratio = self.effective_crit_ratio(side, slot, crit_ratio, dex);
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

        // --- 2b. THE `runEvent('ModifyDamagePhase1')` HANDLER-SORT SHUFFLE
        //     (`gen3_move_coverage_batch2_v1`) — gen3 `modifyDamage` (scripts.js:61) runs
        //     `runEvent('ModifyDamagePhase1')` to fold the screens (Reflect ×0.5 physical,
        //     Light Screen ×0.5 special) + Flash Fire ×1.5. When the DEFENDER's side has
        //     BOTH Reflect AND Light Screen up, their two `onAnyModifyDamagePhase1` handlers
        //     TIE (same order/priority/speed — both side-condition handlers) → a size-2
        //     Fisher-Yates speed-sort shuffle draws EXACTLY one `random(0,2)` per damaging
        //     hit. VERIFIED bit-for-bit vs the sim (`probe_batch2` / the shuffle trace): ONE
        //     screen (or none) → NO tie → NO draw; BOTH screens → 1 draw. Flash Fire is in a
        //     DIFFERENT tie group (attacker vs defender speed) so it never adds to this count
        //     (probed: FF+one-screen = 0, FF+both = 1). Drawn AFTER the crit roll, BEFORE the
        //     `random(16)` damage roll — the exact position in the sim's draw stream. A crit
        //     ignores screens for the DAMAGE, but the handlers still GATHER (the event runs
        //     regardless of crit), so the shuffle draws even under a crit. ---
        self.modify_damage_phase1_shuffle();

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
        // `dealt` = `move.totalDamage` (the sim's `damage()` return) — the ACTUAL HP delta,
        // used as the recoil/drain BASIS. It is the roll clamped to the target's remaining HP,
        // AND (for a NON-sub hit) further reduced by a Focus Band survive-at-1 (the sim's drain/
        // recoil read `move.totalDamage` AFTER the Focus Band `onDamage` reduction — gen3
        // scripts.ts:398/406/408). `dealt` is refreshed inside the `!absorbed` block below to the
        // post-Focus-Band value so a recoil/drain move that a Focus Band saves from a KO
        // recoils/heals off `hp-1`, not the full lethal roll. For a sub the basis is the sub HP
        // (Focus Band never applies to a sub-absorbed hit).
        let mut dealt = realized.min(target_hp_before);
        let sub = self.absorb_into_sub(foe, foe_slot, realized);
        let absorbed = sub != SubAbsorb::NoSub;

        // --- APPLY HP + faint at 0 (only when the sub did NOT absorb it). ---
        if !absorbed {
            // ENDURE survive-at-1 (`gen3_move_coverage_batch6_v1`, `endure.onDamage`
            // priority **−10** — BEFORE Focus Band's −40 in the priority-DESC Damage
            // event, source-derived; the composition itself is unprobed — no gen3 board
            // pairs them — but FB's roll draws UNCONDITIONALLY either way, so the draw
            // stream is order-independent and only the survive branch reads the order):
            // a MOVE hit that would KO an endure-volatile holder is clamped to `hp − 1`,
            // `|-activate|…|move: Endure` emitted BEFORE the `|-damage|` (probe ED1).
            // DRAW-FREE. Fixed damage + each multihit strike have their own sites.
            // FALSE SWIPE (`gen3_bp_modifier_cluster_v1`): an `onDamage` at priority -20 —
            // `if (damage >= target.hp) return target.hp - 1` — so a would-be KO is clamped to
            // leave exactly 1 HP. It is NOT a BP change: the probe shows the move computing
            // 839 damage into a 17-HP target and the target ending at 1. Priority -20 puts it
            // AFTER Endure, which is why it sits here rather than inside the damage calc.
            // DRAW-FREE, and skipped for a sub-absorbed hit (the sub is not `target`).
            let realized = if move_id == "falseswipe" && !absorbed {
                let hp = self.sides[foe].pokemon[foe_slot].hp;
                if realized >= hp && hp > 0 { hp - 1 } else { realized }
            } else {
                realized
            };
            let realized = self.endure_clamp(foe, foe_slot, realized, dex);
            // FOCUS BAND (`gen3_ability_batch4_v1`): the onDamage roll draws AFTER the
            // damage rolls, BEFORE the apply; a lethal MOVE hit that passes survives at
            // 1 HP (probe seed 8: crit path included). A sub-absorbed hit never draws.
            let realized = self.focus_band_damage(foe, foe_slot, realized, true, true, dex);
            // Refresh `dealt` to the POST-Focus-Band applied amount (the recoil/drain basis).
            dealt = realized.min(target_hp_before);
            self.apply_damage(foe, foe_slot, realized);
            // DESTINY BOND trigger record (`gen3_move_coverage_batch6_v1`, the
            // `onFaint` gate: a FOE-side **Move**-effect hit — this path — that zeroed
            // a DB-volatile holder's HP marks the pending mutual faint; the chain
            // itself runs in `process_faints` (corpse `|faint|` → `-activate` → the
            // killer's faint). A sub-absorbed hit / recoil / residual / confusion
            // self-hit never reaches this site; futuremove has its own site and is
            // EXCLUDED (`!effect.flags['futuremove']` — probe DB4's class). DRAW-FREE.
            if self.sides[foe].pokemon[foe_slot].hp == 0
                && self.sides[foe].pokemon[foe_slot].destiny_bond
            {
                self.sides[foe].pokemon[foe_slot].destiny_bond_ko_by = Some(side);
            }
            // FOCUS PUNCH lostFocus (`gen3_move_coverage_batch4_v1`,
            // `focuspunch.condition.onHit`): a NON-Status move that HIT the FP user DIRECTLY
            // (this `!absorbed` block = the mon took the damage, not its sub) sets
            // `lost_focus` → the user's queued Focus Punch is CANCELLED at its onTry. The
            // damaging path here is always non-Status; a chip ABSORBED by the user's own
            // Substitute goes through the `absorbed` branch and does NOT reach here (the sub
            // intercept precedes the focuspunch onHit — VERIFIED vs the sim). DRAW-FREE.
            if let Some(fp) = self.sides[foe].pokemon[foe_slot].focus_punch.as_mut() {
                *fp = true;
            }
            // COUNTER / MIRROR COAT recorder (`gen3_move_coverage_batch5_v1`): a DIRECT
            // (non-sub-absorbed) foe Move hit arms the defender's reactive volatile with
            // 2× the post-Focus-Band applied damage. `category` here is the type-derived
            // gen3 category (Struggle passes Physical explicitly; a variable-BP move was
            // re-categorized Physical at its BP override). DRAW-FREE.
            self.record_reactive_hit(foe, foe_slot, category, &move_id, dealt);
            // RAGE (`gen3_rage_secretpower_v1`): a non-Status FOE move that HITS a mon holding
            // the `rage` volatile raises THAT mon's Atk by one stage. DRAW-FREE, ±6 clamped.
            if dealt > 0
                && category != MoveCategory::Status
                && self.sides[foe].pokemon[foe_slot].rage
                && !self.sides[foe].pokemon[foe_slot].fainted
            {
                let cur = self.sides[foe].pokemon[foe_slot].boosts[1] as i32;
                let next = (cur + 1).clamp(-6, 6);
                self.sides[foe].pokemon[foe_slot].boosts[1] = next as i8;
                if self.logging() && next != cur {
                    let m = self.mon_ref(foe, foe_slot, dex);
                    self.log.boost_applied(&m, 1, 1, (next - cur) as i8, next as i8);
                }
            }
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

        // --- HYPER BEAM's MUSTRECHARGE (`gen3_move_coverage_batch4c_v1`,
        //     `hyperbeam.self = {volatileStatus:'mustrecharge'}`): applied on a SUCCESSFUL
        //     damaging hit — plain hit, sub-absorb AND sub-break, AND a target-KO (the
        //     `|-mustrecharge|` prints BEFORE `|faint|`, which `process_faints` emits
        //     later; the lock persists across the opponent's force-switch). NOT on a miss
        //     / immune / Protect-block (those returned above). DRAW-FREE (probed — the
        //     self.volatileStatus apply consumes no PRNG, unlike the selfDrops boost
        //     random(100)). Emitted `|-mustrecharge|<user>` right after the damage/sub
        //     line (the probed position). ---
        if move_id == "hyperbeam" {
            self.sides[side].pokemon[slot].must_recharge = true;
            if self.logging() {
                let user = self.mon_ref(side, slot, dex);
                self.log.must_recharge(&user);
            }
        }

        // --- PARTIAL TRAP (`gen3_partial_trap_v1` — Wrap / Bind / Fire Spin / Clamp /
        //     Whirlpool / Sand Tomb, `moveData.volatileStatus = 'partiallytrapped'`): applied
        //     in `runMoveEffects`, i.e. AFTER the `-damage` line and BEFORE selfDrops /
        //     secondaries (none of the six carries either, so the position is only observable
        //     against the `-damage`). This is the family's ONE new draw: the gen4-mod
        //     `durationCallback` fires `this.random(3, 7)` INSIDE `addVolatile`.
        //
        //     THE FOUR GATES (each probe-settled — `harness/probe_ptrap_edges.js`), all of
        //     which suppress the DRAW as well as the volatile:
        //       * `!absorbed` — a SUBSTITUTE intercepts at `onTryPrimaryHit` and returns before
        //         `runMoveEffects`, so a Wrap into a sub emits only `|-activate|…|Substitute|
        //         [damage]`: NO volatile, ZERO duration draws (probe A).
        //       * the target is ALIVE — `addVolatile` early-returns on `!this.hp` (the
        //         `affectsFainted` gate), so a cast that KOs draws nothing (probe C).
        //       * the target is NOT ALREADY partially trapped — `addVolatile` returns false on
        //         a present volatile (`partiallytrapped` has NO `onRestart`), and there is no
        //         `-fail` line: the re-cast just deals its damage (probe B).
        //       * a MISS / type-IMMUNE / Protect-blocked cast returned long before this point
        //         (probes G/H).
        //     A mutual wrap and a Baton-Passed trap both fall out of the state below without
        //     special-casing. ---
        if super::helpers::is_partial_trap_move(&move_id)
            && !absorbed
            && self.sides[foe].pokemon[foe_slot].hp > 0
            && self.sides[foe].pokemon[foe_slot].partial_trap.is_none()
        {
            // THE ONE NEW DRAW: gen4's `partiallytrapped.durationCallback` → `this.random(3, 7)`
            // (uniform over {3,4,5,6}; gen3 has no Grip Claw so the `return 6` arm is dead).
            // Probe-measured over 120 landings: {3:25, 4:26, 5:25, 6:27} and chip turns
            // {2:25, 3:26, 4:25, 5:27} == duration − 1.
            let duration = self.prng.random_range(3, 7) as u8;
            let source_uid = self.sides[side].pokemon[slot].uid;
            self.sides[foe].pokemon[foe_slot].partial_trap = Some(crate::state::PartialTrap {
                source_uid,
                move_name: move_name.clone(),
                duration,
            });
            // [EMIT] `|-activate|<target>|move: <Move>|[of] <user>` — the gen5-override
            // `onStart` (`this.add('-activate', pokemon, 'move: ' + sourceEffect, '[of] ' +
            // source)`), immediately after the `-damage`. Observation-only.
            if self.logging() {
                let target = self.mon_ref(foe, foe_slot, dex);
                let user = self.mon_ref(side, slot, dex);
                let effect = format!("move: {move_name}");
                let of = format!("[of] {user}");
                self.log.activate(&target, &effect, Some(&of));
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

        // --- BATCH-1 DRAIN (`gen3_move_coverage_batch1_v1`) — the USER heals a fraction of
        //     the damage dealt, INSIDE the sim's `damage()` (`battle.ts:2167`, gen<=4), so it
        //     fires right after the `-damage`/sub line + BEFORE self.boosts/secondaries. It
        //     fires whether the mon or a SUB took the hit (`dealt` = the damage dealt); the
        //     non-sub path floors, the SUB path ceils (`absorbed` selects). DRAW-FREE. ---
        if drain_fraction > 0.0 {
            self.apply_drain(side, slot, foe, foe_slot, drain_fraction, dealt, absorbed, dex);
        }

        // --- BATCH-1 SELF STAT-DROP (`gen3_move_coverage_batch1_v1`, `move.self.boosts`) —
        //     the sim's `selfDrops` (battle-actions.ts:1338), AFTER runMoveEffects/drain +
        //     BEFORE secondaries. Fires on ANY landed hit INCLUDING behind a sub (it targets
        //     the USER). **NOT draw-free** — gen3 `selfDrops` draws ONE `random(100)` (the
        //     `secondaryRoll`), applied unconditionally (`self.chance === undefined`); see
        //     `apply_self_drops`. This is the ONE draw batch-1 adds. ---
        if !self_drops.is_empty() {
            self.apply_self_drops(side, slot, &self_drops, dex);
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

        // --- UPROAR's LOCK is armed here (`gen3_uproar_v1`): the `random(2,6)` duration is
        //     drawn ONCE, on the CAST turn, AFTER the damage, and NEVER re-drawn — every
        //     continuing turn just re-runs the locked slot. A CAST that did not LAND (miss /
        //     type-IMMUNE Ghost / Protect-blocked) returned before this point, so it applies
        //     NO volatile and draws no duration, though its PP is already spent — probed. ---
        // --- SMELLING SALTS' onHit CURE (`gen3_bp_modifier_cluster_v1`): after a landed hit,
        //     a PARALYZED target is cured. The BP doubling above read the status BEFORE this,
        //     which is the sim's order (basePowerCallback runs inside getDamage, onHit after).
        //     DRAW-FREE. [EMIT] `|-curestatus|<target>|par|[msg]`. ---
        if move_id == "smellingsalts"
            && self.sides[foe].pokemon[foe_slot].status == Some(Status::Paralysis)
            && !self.sides[foe].pokemon[foe_slot].fainted
        {
            self.sides[foe].pokemon[foe_slot].status = None;
            if self.logging() {
                let t = self.mon_ref(foe, foe_slot, dex);
                self.log.curestatus(&t, "par", true);
            }
        }

        // --- UPROAR's WAKE (`gen3_uproar_v1`): a LANDED uproar cures `slp` on BOTH ACTIVES
        //     (the MOVE's own `onTryHit`, DRAW-FREE) — NOT on a miss, NOT on a type-IMMUNE
        //     target, and NEVER for a BENCHED sleeper. Emitted as a BARE
        //     `|-curestatus|<mon>|slp|[msg]` with no `[from]` clause. Both of those returned
        //     before this point, so simply reaching here means the hit landed. ---
        if move_id == "uproar" {
            for s2 in 0..2 {
                let sl2 = self.sides[s2].active;
                if matches!(self.sides[s2].pokemon[sl2].status, Some(Status::Sleep(_)))
                    && !self.sides[s2].pokemon[sl2].fainted
                {
                    self.sides[s2].pokemon[sl2].status = None;
                    self.sides[s2].pokemon[sl2].sleep_from_rest = false;
                    if self.logging() {
                        let m = self.mon_ref(s2, sl2, dex);
                        self.log.curestatus(&m, "slp", true);
                    }
                }
            }
        }

        // --- THE LOCK-IN FAMILY (`gen3_lockin_family_v1`): OUTRAGE / PETAL DANCE / THRASH
        //     share ONE `lockedmove` condition. The duration is a `random(2,4)` (→ 2 or 3)
        //     drawn ONCE on the CAST turn, after the damage. Unlike Uproar the condition emits
        //     NO `-start` line — only the `[from]lockedmove` attr on each continuing turn's
        //     announce — and unlike Uproar the lock ends in CONFUSION at the residual. ---
        // ROLLOUT / ICE BALL advance the execution counter on every LANDED hit; the lock ends
        // after the 5th (`gen3_rollout_defensecurl_v1`).
        if matches!(move_id.as_str(), "rollout" | "iceball") {
            let hits = self.sides[side].pokemon[slot].rollout.map(|(h, _)| h).unwrap_or(0) + 1;
            self.sides[side].pokemon[slot].rollout = if hits >= 5 {
                None
            } else {
                Some((hits, move_index))
            };
        }

        // RAGE (`gen3_rage_secretpower_v1`) arms its `singlemove` volatile on a landed hit.
        // [EMIT] `|-singlemove|<user>|Rage`.
        if move_id == "rage" {
            self.sides[side].pokemon[slot].rage = true;
            if self.logging() {
                let u = self.mon_ref(side, slot, dex);
                self.log.singlemove(&u, "Rage");
            }
        }

        if matches!(move_id.as_str(), "outrage" | "petaldance" | "thrash")
            && self.sides[side].pokemon[slot].locked_move.is_none()
        {
            let n = self.prng.random_range(2, 4) as u8;
            self.sides[side].pokemon[slot].locked_move = Some((n, move_index));
        }

        if move_id == "uproar" && self.sides[side].pokemon[slot].uproar.is_none() {
            let dur = self.prng.random_range(2, 6) as u8;
            self.sides[side].pokemon[slot].uproar = Some((dur, move_index));
            // [EMIT] `|-start|<user>|Uproar`.
            if self.logging() {
                let u = self.mon_ref(side, slot, dex);
                self.log.volatile_start(&u, "Uproar");
            }
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

        // --- BATCH-1 onAfterHit ITEM REMOVAL + RAPID SPIN (`gen3_move_coverage_batch1_v1`) —
        //     the sim's `onAfterHit` (battle-actions.ts:1144), AFTER secondaries + BEFORE the
        //     gen<5 DamagingHit (contact proc). DRAW-FREE. `!struggle` (Struggle carries no
        //     onAfterHit; `move_index` is its stale scripted slot). Knock Off / Thief / Covet
        //     fire ONLY when the MON was damaged (`!absorbed` — the sim's damagedTargets is
        //     empty behind a sub, so the target keeps its item). Rapid Spin ALSO carries an
        //     `onAfterSubDamage`, so it clears even behind a sub (`dealt > 0`, mon OR sub). ---
        if !struggle {
            match move_id.as_str() {
                // `onAfterHit` (item removal) fires via `singleEvent("AfterHit", …)` over
                // `damagedTargets` (battle-actions.js:976-979), and `damagedTargets` includes any
                // target the move dealt a NUMERIC damage to, INCLUDING 0 — an Endure / Focus-Band
                // survive-at-1 clamps the realized damage to 0 but the mon is STILL a damaged target
                // (`typeof damage[i] === "number"`; `Pokemon.damage(0)` returns 0). So the item
                // removal fires on a DIRECT (non-sub) hit even when `dealt == 0`: gate on `!absorbed`
                // (the mon took the hit), NOT `dealt > 0` — the old `dealt > 0` wrongly skipped the
                // Knock Off when Endure clamped it to 0 net damage (repro rmrz81mki_ab_43_12: a 1-HP
                // Hitmonchan Endures a Knock Off → the port kept its Deep Sea Scale while the sim
                // knocked it off). DRAW-FREE. NOTE the sibling `DamagingHit` procs (contact-proc
                // Static / Rough Skin) DIFFER — they fire via `runEvent("DamagingHit", …)`, which
                // gates on the per-target damage relayVar, so they do NOT proc on a 0-dealt Endure
                // hit (verified vs the sim: ab_34_19 dec51, Waterfall into an Enduring Static mon
                // draws NO Static roll) and stay `dealt > 0`.
                "knockoff" | "thief" | "covet" if !absorbed => {
                    self.apply_item_removal(side, slot, foe, foe_slot, &move_id, dex);
                }
                "rapidspin" if dealt > 0 => {
                    self.apply_rapid_spin(side, slot, dex);
                }
                _ => {}
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
        //
        //     ORDER within `runEvent('DamagingHit')` — `DamagingHit` is one of the four events
        //     sorted by `compareLeftToRightOrder` (battle.ts:421/789-790), NOT `speedSort`:
        //     ASCENDING `order` (a MISSING order ⇒ 4294967296), then priority, then the gather
        //     `index`. So the region runs in TWO phases around the defender's `frz` FIRE-THAW
        //     (a STATUS handler carrying NO order):
        //       (a) ORDERED handlers first — in gen3 the ONLY carrier is ROUGH SKIN
        //           (`onDamagingHitOrder: 1`, abilities.ts:3894), so its recoil PRECEDES the thaw;
        //       (b) the thaw (status, un-ordered — gathered before the ability by
        //           `findPokemonEventHandlers`'s status → volatiles → ABILITY → item order,
        //           battle.ts:1100-1123);
        //       (c) the UN-ORDERED abilities (Static / Poison Point / Flame Body / Effect Spore /
        //           Cute Charm), which therefore run AFTER the thaw.
        //     SIM-PROBED both ways: Fire Punch into a FROZEN STATIC holder emits `|-damage|…frz`
        //     → `|-curestatus|<t>|frz|[msg]` → `|-status|<a>|par|…Static` (`probe_rb_tail.js` C3),
        //     while Fire Punch into a FROZEN ROUGH SKIN holder emits `|-damage|…frz` →
        //     `|-damage|<a>|…|[from] ability: Rough Skin|[of] <t>` → `|-curestatus|<t>|frz|[msg]`
        //     (the 24h-fuzz repros ab_70_2 / ab_2293_23). Round 24 modeled only (b)→(c) and so
        //     put Rough Skin on the wrong side of the thaw. Emission-ORDER only: the thaw is
        //     draw-free and touches only the DEFENDER's status while the procs roll + status the
        //     ATTACKER, so the draw stream is unchanged. ---
        if is_contact && !absorbed && dealt > 0 {
            self.apply_contact_proc(side, slot, foe, foe_slot, DamagingHitPhase::Ordered, dex);
        }
        if is_fire && !absorbed && self.sides[foe].pokemon[foe_slot].status == Some(Status::Freeze) {
            // `cureStatus()` early-returns on a 0-HP mon (`if (!this.hp || !this.status)
            // return false`), so a KO-ing fire move does NEITHER half: no
            // `|-curestatus|<target>|frz|[msg]` (FORM 13 — the thaw reveal, `[msg]` because
            // there is no sourceEffect; `gen3_omniscient_byte_fuzz_v1` caught the emit
            // wrongly preceding the sim's `|faint|`) AND no status clear. The corpse keeps
            // `frz` until `checkFainted` overwrites it with `fnt` — and when the KO ENDS the
            // battle `checkFainted` never runs, so `frz` is what the referee readout reports.
            //
            // The CLEAR used to be unconditional, on the reasoning that "a faint overrides it
            // anyway". It does not: `outcome.pN.active_status` on a DECIDING faint read `""`
            // where node read `"frz"` (`gen3_fire_thaw_ko_keeps_status_v1`, found by
            // `replay_impl_parity` on a freshly generated golden). Draw-free either way.
            if self.sides[foe].pokemon[foe_slot].hp > 0 {
                self.sides[foe].pokemon[foe_slot].status = None;
                if self.logging() {
                    let t = self.mon_ref(foe, foe_slot, dex);
                    self.log.curestatus(&t, "frz", true);
                }
            }
        }

        // The DEFENDER's UN-ORDERED ABILITY `onDamagingHit` — phase (c): gathered AFTER its
        // status handler above (the ORDERED Rough Skin pass already ran, phase (a)).
        if is_contact && !absorbed && dealt > 0 {
            self.apply_contact_proc(side, slot, foe, foe_slot, DamagingHitPhase::Unordered, dex);
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
        // `hp > 0` gate: the gen3 `battle.damage`→`spreadDamage` `!target.hp` early-return
        // (battle.js:1727) — a Struggle whose CONTACT hit fainted the user via Rough Skin
        // (fired at the DamagingHit position ABOVE) skips its own recoil, emitting NOTHING (the
        // same class as the batch-1 `apply_recoil` fix, repro rmrz81mki_ab_47_23).
        if struggle && dealt > 0 && self.sides[side].pokemon[slot].hp > 0 {
            let user_hp_before = self.sides[side].pokemon[slot].hp;
            let recoil = (dealt / 4).max(1).min(user_hp_before);
            // FOCUS BAND: the recoil is a Damage event into the user (effect 'recoil',
            // not a Move) — the roll draws, no survive. `gen3_ability_batch4_v1`.
            let recoil = self.focus_band_damage(side, slot, recoil, false, false, dex);
            self.apply_damage(side, slot, recoil);
            if self.logging() {
                let user = self.mon_ref(side, slot, dex);
                let target = self.mon_ref(foe, foe_slot, dex);
                let hp = self.hp_status(side, slot);
                self.log.damage_of(&user, &hp, &Cause::Bare("Recoil".into()), &target);
            }
        }

        // --- BATCH-1 RECOIL (`gen3_move_coverage_batch1_v1`, Double-Edge / Take Down /
        //     Submission) — the gen3 `tryMoveHit` recoil (scripts.ts:460), AFTER moveHit
        //     returns (so LAST in the landed-hit tail, after the contact procs). Fires
        //     whether the mon or a SUB took the hit (`dealt`); Rock Head negates. DRAW-FREE.
        //     `!struggle` (Struggle's recoil is the dedicated path above — its
        //     `recoil_fraction` is 0 here). ---
        if !struggle && recoil_fraction > 0.0 && dealt > 0 {
            self.apply_recoil(side, slot, foe, foe_slot, recoil_fraction, dealt, dex);
        }

        // The move landed (acc-hit + non-immune; a sub hit STILL fires the in-tryMoveHit
        // Update — gen-3 `moveHit` returns a truthy `true` for a HIT_SUBSTITUTE so
        // `tryMoveHit`'s `if (!damage && damage !== 0) return` does NOT short-circuit).
        MoveResolution::done(false, crit, true)
    }

    /// BEAT UP (`gen3_move_coverage_batch4b_v1`) — the multi-strike STAT-SWAP move. ONE
    /// strike PER healthy (non-fainted, NON-STATUSED) party member of the USER's side, in
    /// PARTY ORDER (the sim's `move.allies = side.pokemon.filter(a => !a.fainted && !a.status)`,
    /// `move.multihit = allies.length`; the ACTIVE user itself strikes when healthy — a
    /// single-mon side is exactly 1 strike — and a burned/paralyzed ACTIVE user skips its OWN
    /// strike via the uniform filter). The whole-move accuracy roll (acc 100 →
    /// `randomChance(100,100)`, drawn ONCE BEFORE the loop) already fired in `run_move` and is
    /// NOT re-drawn here; Beat Up is never immune (typeless '???' → 1× / hits Ghost) so this is
    /// only reached on a hit. Per strike, in the getDamage/modifyDamage order: [crit]
    /// `randomChance(1, critMult[critRatio=1]=16)` (1/16, independent per strike), then [damage]
    /// `random(16)`. The strike damage runs the standard gen-3 calc with **level = the ACTIVE
    /// user's level**, **BP 10**, **TYPELESS** (no STAB / 1× / hits Ghost), **category Special**,
    /// but the attacker's SpA REPLACED by the ally's dex `baseStats.atk` and the defender's SpD
    /// by the target's dex `baseStats.def`, both at `event.modifier=1` (NO boosts / items / CB /
    /// abilities / burn touch the stat) — so the strike depends only on ally base-atk, target
    /// base-def, level, crit (×2, screens ignored), and Light Screen (Special → applies) + the
    /// 85-100% roll. Each strike routes through the sub-intercept (a break lets later strikes hit
    /// the mon directly); the multihit STOPS at the target's faint (later strikes + the Quick
    /// Claw skip — the deferred-faint protocol). If ALL party members are fainted/statused the
    /// ally list is empty → the basePowerCallback returns null → the move FIZZLES draw-free (a
    /// rare fail-safe: no strikes / no hitcount / not landed). VERIFIED bit-for-bit vs
    /// `harness/probe_batch4b_beatup.js` (6 healthy = 12 strike draws; statused/fainted skips;
    /// the KO-mid-sequence break; typeless-into-Ghost; the sub break + later direct strikes).
    #[allow(clippy::too_many_arguments)]
    fn run_beat_up(
        &mut self,
        side: usize,
        slot: usize,
        foe: usize,
        foe_slot: usize,
        crit_ratio: u8,
        move_name: &str,
        dex: &Dex,
    ) -> MoveResolution {
        // Add the `beatup` volatile (`gen3_move_coverage_batch4b_v1`, the sim's `onModifyMove`
        // `pokemon.addVolatile("beatup")` BEFORE the strikes). Its ONLY observable effect here is
        // the `duration: 1` residual DURATION handler (the stat swap is computed directly below),
        // so a BEAT UP MIRROR at equal speed adds one residual tie-shuffle draw. Cleared at the
        // next turn-top (`clear_flinch`).
        self.sides[side].pokemon[slot].beat_up = true;

        // Snapshot the strike list up front so the &mut self hits below don't fight the
        // immutable dex borrow: party order, healthy = `!fainted && status == None`. Each
        // entry carries the ally's SLOT INDEX (for the `[of]` display name) + its dex base-atk.
        let user_level = self.sides[side].pokemon[slot].level;
        let target_base_def = dex
            .species(&self.sides[foe].pokemon[foe_slot].species_id)
            .expect("beatup: target species must be in the dex")
            .base_stats
            .def;
        let strikes: Vec<(usize, u16)> = (0..self.sides[side].pokemon.len())
            .filter(|&i| {
                let a = &self.sides[side].pokemon[i];
                !a.fainted && a.status.is_none()
            })
            .map(|i| {
                let base_atk = dex
                    .species(&self.sides[side].pokemon[i].species_id)
                    .expect("beatup: ally species must be in the dex")
                    .base_stats
                    .atk;
                (i, base_atk)
            })
            .collect();

        // [EMIT] the move announce `|move|<user>|Beat Up|<foe>` (NO effectiveness line —
        // each strike is typeless). Observation-only (reads resolved state, draws nothing).
        if self.logging() {
            let user = self.mon_ref(side, slot, dex);
            let target = self.mon_ref(foe, foe_slot, dex);
            self.log.move_used(&user, move_name, Some(&target), false, false);
        }

        // Degenerate: no eligible allies → the basePowerCallback returns null → the move
        // FIZZLES draw-free (no strikes / no hitcount). Rare (every party member
        // fainted/statused). NOT landed → no in-tryMoveHit Update. [EMIT] the did-nothing
        // `[still]` announce form + a bare `|-fail|<user>` (`gen3_omniscient_byte_fuzz_v1`
        // FORMS 1+2: `|move|p1a: Charizard|Beat Up||[still]`, `|-fail|p1a: Charizard`).
        if strikes.is_empty() {
            if self.logging() {
                self.log.attr_last_move_still();
                let user = self.mon_ref(side, slot, dex);
                self.log.fail(&user, None, false);
            }
            return MoveResolution::done(false, false, false);
        }

        let light_screen = self.sides[foe].light_screen > 0;
        let crit_immune = dex
            .ability(&self.sides[foe].pokemon[foe_slot].ability)
            .map(|a| a.crit_immune)
            .unwrap_or(false);

        // The ACTIVE user's `onModifyCritRatio` fold (Focus Energy / a CRIT_ITEM) — read
        // ONCE (constant across the multi-strike; the crit-ratio event reads the SOURCE, the
        // active user, not each ally). Beat Up's dex critRatio is 1 → 1/16 by default.
        let eff_crit_ratio = self.effective_crit_ratio(side, slot, crit_ratio, dex);
        let mut hits = 0u32;
        for (ally_idx, ally_atk) in strikes {
            // [crit] `randomChance(1, critMult[critRatio])` — independent per strike.
            // CRIT_IMMUNE (Battle/Shell Armor) draws the roll then overrides it to false
            // (draw-count unchanged), like the main damaging path.
            let mut crit = self.prng.random_chance(1, CRIT_MULT[eff_crit_ratio as usize]);
            if crit && crit_immune {
                crit = false;
            }

            // The stat-swap DamageContext: typeless / Special / level = the ACTIVE user's,
            // BP 10, attacker SpA = ally base-atk, defender SpD = target base-def, modifier=1
            // (no boosts / items / burn / abilities). Light Screen (special) applies; Reflect
            // does not. The typeless move has no type-chart row → 1× (hits every type incl.
            // Ghost) and no STAB (attacker types empty).
            let ctx = DamageContext {
                attacker: Combatant {
                    level: user_level,
                    atk_stat: 0,
                    spa_stat: ally_atk,
                    def_stat: 0,
                    spd_stat: 0,
                    types: Vec::new(),
                    boosts: [0; 5],
                    burned: false,
                    has_guts: false,
                },
                defender: Combatant {
                    level: user_level, // the defender's level is unused by the formula
                    atk_stat: 0,
                    spa_stat: 0,
                    def_stat: 0,
                    spd_stat: target_base_def,
                    types: Vec::new(),
                    boosts: [0; 5],
                    burned: false,
                    has_guts: false,
                },
                mv: MoveInput {
                    minimize_doubles: false,
                    base_power: 10,
                    move_type: None,
                    category: MoveCategory::Special,
                    halves_defense: false,
                },
                crit,
                weather: None,
                reflect: false,
                light_screen,
                atk_stat_mods: Vec::new(),
                atk_direct_modify: None,
                def_stat_mods: Vec::new(),
                bp_mods: Vec::new(),
                defender_thick_fat: false,
                defender_minimized: false,
                immune: false,
                flash_fire: false,
            };
            let dmg = calc_damage(&ctx, dex);
            // [damage] `random(16)` selects the roll — every other damaging move's order.
            let r = self.prng.random_below(16) as usize;
            let realized = dmg.rolls[r];

            // [EMIT] the per-strike `|-activate|<user>|move: Beat Up|[of] <ally>` (the gen3
            // mod's `beatup.condition.onModifySpA`, PROBE-SETTLED against the omniscient stream
            // + the resolved dist — `gen3_omniscient_byte_fuzz_v1` FORM 8: the byte fuzzer's
            // gen3customgame L-rows DO carry it, contradicting the task's "no activate" claim).
            // GATED on `!beatupnicknamesmod`: the mod's `if (!this.ruleTable.has(
            // "beatupnicknamesmod")) this.add("-activate", …)` — the rule is part of the gen3
            // **Standard** bundle (present in gen3ou, absent in gen3customgame), so it aligns
            // exactly with `sleep_clause` (both derive from `format_has_sleep_clause`). `[of]`
            // is the CURRENT strike's ally (`move.allies[0]` before the shift). Then (if crit)
            // `|-crit|<foe>`, then the `|-damage|` / Substitute result.
            if self.logging() {
                if !self.sleep_clause {
                    let user = self.mon_ref(side, slot, dex);
                    let ally_name = self.display_name(side, ally_idx, dex);
                    self.log.activate(&user, "move: Beat Up", Some(&format!("[of] {ally_name}")));
                }
                if crit {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.crit(&target);
                }
            }

            // Route through the sub-intercept: a break lets later strikes hit the mon.
            let sub = self.absorb_into_sub(foe, foe_slot, realized);
            let absorbed = sub != SubAbsorb::NoSub;
            if !absorbed {
                let pre_hp = self.sides[foe].pokemon[foe_slot].hp;
                // ENDURE clamps EVERY strike of a multihit (`gen3_move_coverage_batch6_v1`
                // — the probe ED8 Arm-Thrust class: each strike `-activate`s + survives at
                // 1). DRAW-FREE.
                let realized = self.endure_clamp(foe, foe_slot, realized, dex);
                // FOCUS BAND per strike (`gen3_beatup_focus_band_v1`): the gen4-inherited
                // `focusband.onDamage` puts `randomChance(1,10)` FIRST in its `&&`, so the roll
                // draws on EVERY move-damage hit into a Focus Band holder (JS short-circuit), NOT
                // only lethal ones — and a lethal strike that passes survives at 1 HP. A Beat Up
                // STRIKE runs the full `spreadMoveHit` → `spreadDamage` → `runEvent('Damage')` →
                // the Focus Band handler, exactly like the single-hit path (`run_move`) and the
                // generic `[2,5]` multihit (`run_multihit`). `run_beat_up` used to apply the strike
                // damage DIRECTLY (no `focus_band_damage`), so a Beat Up strike into a FB holder
                // MISSED the draw → a one-fewer-draw desync (random-mode byte fuzz find ab_7_7 @
                // master-seed 200724: Houndour's 1-strike Beat Up into a Focus-Band Geodude — the
                // sim drew the FB `randomChance(1,10)`, the port didn't → the seed desynced). A
                // sub-absorbed strike never reaches here (the `!absorbed` gate). `emit_survive_zero
                // = true` mirrors the `realized > 0` `-damage` gate below (self-emits the 1/max
                // line when a survive nets 0 at 1 HP).
                let realized = self.focus_band_damage(foe, foe_slot, realized, true, true, dex);
                self.apply_damage(foe, foe_slot, realized);
                // DESTINY BOND trigger record (`gen3_move_coverage_batch6_v1`): a Beat Up
                // strike is a FOE Move hit — a KO strike with the DB volatile up marks the
                // pending mutual faint (the chain runs in process_faints). DRAW-FREE.
                if self.sides[foe].pokemon[foe_slot].hp == 0
                    && self.sides[foe].pokemon[foe_slot].destiny_bond
                {
                    self.sides[foe].pokemon[foe_slot].destiny_bond_ko_by = Some(side);
                }
                // COUNTER / MIRROR COAT recorder (`gen3_move_coverage_batch5_v1`): Beat
                // Up's strikes are **Special** (the resolved gen3 category — Dark) Move
                // hits, so each DIRECT strike arms the target's MIRROR COAT (never
                // Counter), OVERWRITING per strike → the return is 2× the LAST strike
                // (probed `probe_batch5_reactive_edges.js` BU-M: 3 strikes → MC armed
                // with the last, then `-immune` into the Dark attacker).
                self.record_reactive_hit(
                    foe,
                    foe_slot,
                    MoveCategory::Special,
                    "beatup",
                    realized.min(pre_hp),
                );
                // FOCUS PUNCH lostFocus (`focuspunch.condition.onHit`): a Beat Up strike that
                // HIT the target DIRECTLY (not its sub) sets `lost_focus` if the target has a
                // pending Focus Punch → its queued FP is CANCELLED at onTry. Beat Up is a
                // non-Status damaging move, so each direct strike counts (the sim fires the
                // onHit per hit). A sub-absorbed strike does NOT (the sub intercept precedes the
                // focuspunch onHit — the `!absorbed` gate, same as run_move). DRAW-FREE. Missing
                // this let a Beat Up into a Focus-Punch user wrongly leave the FP landing (e2e_196).
                if let Some(fp) = self.sides[foe].pokemon[foe_slot].focus_punch.as_mut() {
                    *fp = true;
                }
                if realized > 0 && self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    let hp = self.hp_status(foe, foe_slot);
                    self.log.damage(&target, &hp, None);
                }
            } else if self.logging() {
                let target = self.mon_ref(foe, foe_slot, dex);
                match sub {
                    SubAbsorb::Held => self.log.activate(&target, "Substitute", Some("[damage]")),
                    SubAbsorb::Broke => self.log.volatile_end(&target, "Substitute"),
                    SubAbsorb::NoSub => unreachable!(),
                }
            }
            hits += 1;

            // --- THE PER-STRIKE `eachEvent('Update')` (gen3 `tryMoveHit` scripts.js:~400, the
            //     `for (i = 0; ...)` multihit loop fires `this.battle.eachEvent("Update")` INSIDE
            //     the loop AFTER each `moveHit`, BEFORE the loop's `target.hp` re-check). It draws
            //     one `random(0,2)` speed-tie shuffle on a TIE (zero at distinct speed — which is
            //     why the distinct-speed dedicated golden hid this), then the item onUpdate (cure
            //     berries, draw-free). It fires even for the KO strike (the target is 0-HP but NOT
            //     yet `fainted` — `getAllActive` still counts it). The TRAILING in-tryMoveHit
            //     Update (scripts.js:75) is fired by the CALLER via `landed=true`. Missing these
            //     per-strike Updates was the e2e_52 real-team desync (Charizard↔Salamence tie). ---
            let upd = self.each_event_shuffle();
            self.run_update_items(&upd, dex);

            // The multihit STOPS at the target's faint (hp == 0) — the remaining strikes +
            // the end-of-turn Quick Claw skip (the deferred-faint protocol). A sub-absorbed
            // strike never faints the mon. (The per-strike Update above already fired for the
            // KO strike, mirroring the loop's post-moveHit `eachEvent` then `target.hp` check.)
            if self.sides[foe].pokemon[foe_slot].hp == 0 {
                break;
            }
        }

        // [EMIT] `|-hitcount|<target>|N` (N = the strikes that actually FIRED — 6 for a full
        // healthy side, fewer on a statused/fainted skip or a KO-mid-sequence break).
        if self.logging() {
            let target = self.mon_ref(foe, foe_slot, dex);
            self.log.hitcount(&target, hits);
        }

        // The move landed (>=1 strike hit → moveHit returns truthy → the in-tryMoveHit Update
        // fires, even on a KO — before process_faints). No move-level crit (the per-strike
        // crits were already applied to the strike damage).
        MoveResolution::done(false, false, true)
    }

    /// Resolve a GENERIC MULTI-STRIKE move (`gen3_move_coverage_batch7_v1`) — Double Kick /
    /// Twineedle / Bonemerang (fixed 2), and the variable [2,5] family (Pin Missile / Bullet
    /// Seed / Icicle Spear / Rock Blast / Barrage / Comet Punch / Double Slap / Spike Cannon /
    /// Arm Thrust / Fury Attack / Fury Swipes / Bone Rush). Unlike Beat Up (a stat-swap
    /// `basePowerCallback`), each strike runs the NORMAL damage path with the move's real
    /// type/BP/category, so this REUSES the `ctx` `run_move` already built (refreshing only the
    /// crit flag + the defender types per strike — the latter for a mid-multihit Color Change).
    ///
    /// The DRAW MODEL, verified bit-for-bit vs the omniscient sim
    /// (`harness/probe_batch7_multihit.js` + the resolved `hitStepMoveHitLoop`,
    /// battle-actions.ts:748):
    ///   - the whole-move ACCURACY roll ALREADY drew in `run_move` (`acc_hit`); this is only
    ///     reached on a landed, non-immune, non-protect-blocked hit.
    ///   - the COUNT: `Fixed(n)` draws NOTHING; `Range(2,5)` (gen<5) draws ONE
    ///     `sample([2,2,2,3,3,3,4,5])` = one `random(8)` (8-elem, power-of-2 → clean), drawn
    ///     HERE, AFTER accuracy and BEFORE the per-strike loop.
    ///   - PER STRIKE (the sim's `spreadMoveHit`): the effectiveness lines, crit
    ///     `randomChance(1,critMult)`, (both screens → the ModifyDamagePhase1 tie-shuffle),
    ///     damage `random(16)`, then the move's SECONDARY `random(100)` (Twineedle 20% psn — PER
    ///     STRIKE, already-statused-gated after the first) + King's Rock + the DEFENDER's
    ///     contact-proc / fire-thaw / Color Change, then the per-strike `eachEvent('Update')`
    ///     (drawn on a speed tie). The loop STOPS when the target FAINTS (mirroring the sim's
    ///     `targets.every(!hp)` guard at the top of each iteration); a KO on the last landing
    ///     strike defers the Quick Claw (the deferred-faint protocol).
    /// Emits `|move|` once + per-strike effectiveness/`-crit`/`-damage` + `|-hitcount|N`.
    /// TRIPLE KICK (multiaccuracy) — the ONLY gen3 carrier — re-rolls accuracy per strike AND
    /// escalates BP per strike; it FAIL-LOUDS here rather than silently mismodel (no gen3ou team
    /// carries it, and the picker never admits it).
    #[allow(clippy::too_many_arguments)]
    fn run_multihit(
        &mut self,
        side: usize,
        slot: usize,
        foe: usize,
        foe_slot: usize,
        ctx: DamageContext,
        move_type: Option<Type>,
        category: MoveCategory,
        move_index: usize,
        crit_ratio: u8,
        move_id: &str,
        move_name: &str,
        is_contact: bool,
        is_fire: bool,
        suppress_announce: bool,
        mh: MultiHit,
        multiaccuracy: bool,
        dex: &Dex,
    ) -> MoveResolution {
        assert!(
            !multiaccuracy,
            "run_multihit: multiaccuracy move ({move_id}) is unmodeled (per-strike accuracy \
             re-roll + escalating BP — Triple Kick) — fail-loud rather than silently desync"
        );

        // [EMIT] the move announce `|move|<user>|<Name>|<foe>` ONCE, before the strikes. A
        // Sleep-Talk-called multihit carries `[from] Sleep Talk` via the ProtocolBuilder's
        // one-shot (set by the caller); no gen3 multihit move is a lockedmove, so no lockedmove
        // attr. Observation-only (reads resolved state, draws nothing).
        if self.logging() && !suppress_announce {
            let user = self.mon_ref(side, slot, dex);
            let target = self.mon_ref(foe, foe_slot, dex);
            self.log.move_used(&user, move_name, Some(&target), false, false);
        }

        // THE COUNT — `Fixed` draws nothing; the `[2,5]` variable draws ONE `sample` (gen<5).
        let count: u32 = match mh {
            MultiHit::Fixed(n) => n as u32,
            MultiHit::Range(2, 5) => {
                const MH_2_5: [u32; 8] = [2, 2, 2, 3, 3, 3, 4, 5];
                MH_2_5[self.prng.random_below(8) as usize]
            }
            MultiHit::Range(lo, hi) => {
                // No non-`[2,5]` gen3 multihit range exists; mirror the sim's `random(lo, hi+1)`
                // defensively so a future data shape can't silently desync.
                lo as u32 + self.prng.random_below((hi + 1 - lo) as u32)
            }
        };

        let crit_immune = dex
            .ability(&self.sides[foe].pokemon[foe_slot].ability)
            .map(|a| a.crit_immune)
            .unwrap_or(false);
        // The `onModifyCritRatio` fold (Focus Energy / a CRIT_ITEM — Scope Lens / Lucky
        // Punch / Stick), read once (constant across the multi-strike).
        let eff_crit_ratio = self.effective_crit_ratio(side, slot, crit_ratio, dex);

        let mut hits = 0u32;
        for _ in 0..count {
            // STOP if the target already fainted (a prior strike KO'd it, OR the target entered
            // at 0 HP) — the sim's `targets.every(!hp)` guard at the top of each iteration.
            if self.sides[foe].pokemon[foe_slot].hp == 0 {
                break;
            }

            // Per-strike EFFECTIVENESS — read from the LIVE defender types (a mid-multihit Color
            // Change re-types later strikes). Emitted BEFORE `-crit`, per the sim's per-hit order.
            let def_types = mon_types(&self.sides[foe].pokemon[foe_slot], dex);
            if let (Some(mt), true) = (move_type, self.logging()) {
                let eff = dex.type_chart().effectiveness(mt, &def_types);
                let target = self.mon_ref(foe, foe_slot, dex);
                if eff > 1.0 {
                    self.log.supereffective(&target);
                } else if eff < 1.0 && eff > 0.0 {
                    self.log.resisted(&target);
                }
            }

            // [crit] per strike — `randomChance(1, critMult[effRatio])`; CRIT_IMMUNE draws the
            // roll then overrides to false (draw-count unchanged), like the single-hit path.
            let mut crit = self.prng.random_chance(1, CRIT_MULT[eff_crit_ratio as usize]);
            if crit && crit_immune {
                crit = false;
            }
            if crit && self.logging() {
                let target = self.mon_ref(foe, foe_slot, dex);
                self.log.crit(&target);
            }

            // REBUILD the ctx from the LIVE battle state per strike — the sim re-runs
            // `getDamage` fresh per hit (`spreadMoveHit`), so a mid-multihit state change
            // is reflected in later strikes' damage: an ATTACKER status gained from a
            // CONTACT-PROC ability (Poison Point poisoning a GUTS user → Atk ×1.5 on
            // strikes 2+ — the ab_34_9 repro), a per-strike SECONDARY statusing the
            // defender (→ Marvel Scale), a pinch-berry HP threshold, a Color Change
            // re-type, etc. `build_damage_context` is DRAW-FREE (seed-neutral) and, for
            // an UNCHANGED state (strike 1, or no proc), reproduces the passed ctx exactly
            // — so this stays byte-identical everywhere except a genuine mid-multihit
            // change. No gen3 multihit move carries the Facade/Charge/Solar-Beam
            // post-build bp_mods (all single-hit, id-gated), so a full rebuild loses none.
            let mut sctx = self.build_damage_context(
                side,
                slot,
                foe,
                foe_slot,
                ctx.mv.base_power,
                move_type,
                category,
                ctx.mv.halves_defense,
                dex,
            );
            sctx.crit = crit;
            let dmg = crate::damage::calc_damage(&sctx, dex);

            // [ModifyDamagePhase1 shuffle] the FULL screen tie group (all screens across BOTH
            // sides — `onAnyModifyDamagePhase1`) → `k-1` draws for k screens, PER STRIKE
            // (getDamage runs per hit; drawn AFTER crit, BEFORE the `random(16)`).
            self.modify_damage_phase1_shuffle();

            // [damage] `random(16)` per strike.
            let r = self.prng.random_below(16) as usize;
            let realized = dmg.rolls[r];

            // SUBSTITUTE absorb (a break lets later strikes hit the mon) — the same sub-intercept
            // as the single-hit path; the acc/crit/damage draws are unchanged.
            let target_hp_before = match self.sides[foe].pokemon[foe_slot].substitute {
                Some(sub_hp) => sub_hp,
                None => self.sides[foe].pokemon[foe_slot].hp,
            };
            let mut dealt = realized.min(target_hp_before);
            let sub = self.absorb_into_sub(foe, foe_slot, realized);
            let absorbed = sub != SubAbsorb::NoSub;

            if !absorbed {
                // ENDURE clamps EVERY strike; FOCUS BAND draws per strike (a Damage event).
                let realized = self.endure_clamp(foe, foe_slot, realized, dex);
                let realized = self.focus_band_damage(foe, foe_slot, realized, true, true, dex);
                dealt = realized.min(target_hp_before);
                self.apply_damage(foe, foe_slot, realized);
                // DESTINY BOND + FOCUS PUNCH lostFocus + COUNTER/MIRROR COAT recorder — each a
                // DIRECT foe Move hit, per strike (the recorder OVERWRITES → 2× the LAST strike).
                if self.sides[foe].pokemon[foe_slot].hp == 0
                    && self.sides[foe].pokemon[foe_slot].destiny_bond
                {
                    self.sides[foe].pokemon[foe_slot].destiny_bond_ko_by = Some(side);
                }
                if let Some(fp) = self.sides[foe].pokemon[foe_slot].focus_punch.as_mut() {
                    *fp = true;
                }
                self.record_reactive_hit(foe, foe_slot, category, move_id, dealt);
                if realized > 0 && self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    let hp = self.hp_status(foe, foe_slot);
                    self.log.damage(&target, &hp, None);
                }
            } else if self.logging() {
                let target = self.mon_ref(foe, foe_slot, dex);
                match sub {
                    SubAbsorb::Held => self.log.activate(&target, "Substitute", Some("[damage]")),
                    SubAbsorb::Broke => self.log.volatile_end(&target, "Substitute"),
                    SubAbsorb::NoSub => unreachable!(),
                }
            }

            // Per-strike SECONDARY (Twineedle psn) → King's Rock → fire-thaw / contact-proc /
            // Color Change — the sim's `spreadMoveHit` per-hit tail (secondaries → onAfterHit →
            // DamagingHit), each on its own mon. Inside DamagingHit the ORDERED handlers run
            // first (Rough Skin, `onDamagingHitOrder: 1`), then the defender's un-ordered `frz`
            // STATUS handler, then its un-ordered ABILITY handler (see the single-hit note above).
            self.apply_secondaries(side, slot, foe, foe_slot, move_index, absorbed, dex);
            self.apply_kings_rock_secondary(side, slot, foe, foe_slot, move_id, absorbed, dex);
            if is_contact && !absorbed && dealt > 0 {
                self.apply_contact_proc(side, slot, foe, foe_slot, DamagingHitPhase::Ordered, dex);
            }
            if is_fire
                && !absorbed
                && self.sides[foe].pokemon[foe_slot].status == Some(Status::Freeze)
            {
                // Same `cureStatus()` 0-HP early-return as the single-hit path above
                // (`gen3_fire_thaw_ko_keeps_status_v1`): a KO-ing strike neither emits nor
                // clears.
                if self.sides[foe].pokemon[foe_slot].hp > 0 {
                    self.sides[foe].pokemon[foe_slot].status = None;
                    if self.logging() {
                        let t = self.mon_ref(foe, foe_slot, dex);
                        self.log.curestatus(&t, "frz", true);
                    }
                }
            }
            if is_contact && !absorbed && dealt > 0 {
                self.apply_contact_proc(side, slot, foe, foe_slot, DamagingHitPhase::Unordered, dex);
            }
            if !absorbed && dealt > 0 {
                self.apply_color_change(foe, foe_slot, move_type, dex);
            }

            hits += 1;

            // The per-strike `eachEvent('Update')` (drawn on a speed tie) + item onUpdate — the
            // sim's `eachEvent("Update")` at the end of each multihit iteration.
            let upd = self.each_event_shuffle();
            self.run_update_items(&upd, dex);

            // STOP at the target's faint (the remaining strikes + the Quick Claw skip).
            if self.sides[foe].pokemon[foe_slot].hp == 0 {
                break;
            }
        }

        // [EMIT] `|-hitcount|<target>|N` (N = the strikes that actually FIRED).
        if self.logging() {
            let target = self.mon_ref(foe, foe_slot, dex);
            self.log.hitcount(&target, hits);
        }

        // The move landed (>=1 strike hit → the in-tryMoveHit Update fires via `landed=true`).
        MoveResolution::done(false, false, true)
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
    /// Guillotine / Bide) is routed here by `is_fixed_damage_move` but has NO
    /// `fixed_damage_amount` entry, so this PANICS rather than silently no-op (they
    /// need extra RNG or reactive/OHKO machinery). Counter / Mirror Coat / Endeavor
    /// are MODELED since `gen3_move_coverage_batch5_v1` (the onTry gates below + the
    /// reactive-volatile `fixed_damage_amount` arms).
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
        // --- COUNTER / MIRROR COAT onTry gate (`gen3_move_coverage_batch5_v1`, BEFORE
        //     the accuracy roll): the move FAILS with **ZERO draws** when the reactive
        //     volatile is missing (never selected via beforeTurnMove — unreachable here)
        //     or UN-ARMED (`slot === null` — no qualifying foe hit landed THIS turn:
        //     no damage / wrong category / a sub-absorbed hit / the foe switched / a
        //     prev-turn-only hit). The fail protocol shape (probed): a bare
        //     `|move|<user>|Counter|<current foe active>` line — NO `-fail`, NO
        //     `[still]`; PP already deducted by the caller. A foe that SWITCHED this
        //     turn is announced as the NEW active (probed C2). ---
        if move_id == "counter" || move_id == "mirrorcoat" {
            let armed = self.sides[side].pokemon[slot]
                .reactive
                .filter(|r| r.mirror == (move_id == "mirrorcoat"))
                .and_then(|r| r.damage);
            if armed.is_none() {
                if self.logging() {
                    let user = self.mon_ref(side, slot, dex);
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.move_used(&user, move_name, Some(&target), false, false);
                }
                return MoveResolution::done(false, false, false);
            }
        }

        // --- ENDEAVOR onTry gate (`gen3_move_coverage_batch5_v1`, BEFORE the accuracy
        //     roll): fails at `pokemon.hp >= target.hp` — EQUALITY INCLUDED (probed
        //     50v50 fails; STRICT less-than required to proceed). The compare reads the
        //     TARGET MON's hp (not its sub's). Emits the normal target-form announce +
        //     `|-fail|<user>`, ZERO draws; PP already deducted. ---
        if move_id == "endeavor"
            && self.sides[side].pokemon[slot].hp >= self.sides[foe].pokemon[foe_slot].hp
        {
            if self.logging() {
                let user = self.mon_ref(side, slot, dex);
                let target = self.mon_ref(foe, foe_slot, dex);
                self.log.move_used(&user, move_name, Some(&target), false, false);
                self.log.fail(&user, None, false);
            }
            return MoveResolution::done(false, false, false);
        }

        // FAIL-LOUD: only the modeled fixed-damage set has a formula. A DEFERRED
        // fixed-damage move (Psywave / OHKO / Bide) reaches here via
        // `is_fixed_damage_move` but has no amount → PANIC (never a silent desync —
        // these draw an unmodeled number/order or need accumulator machinery).
        let amount = match fixed_damage_amount(
            move_id,
            &self.sides[side].pokemon[slot],
            &self.sides[foe].pokemon[foe_slot],
        ) {
            Some(a) => a,
            None => panic!(
                "unmodeled FIXED-DAMAGE move {move_id:?} routed to run_fixed_damage_move — \
                 Psywave (variable RNG) / the OHKO moves (Fissure/Horn Drill/Guillotine) / \
                 Bide (a 2-turn accumulator) are DEFERRED. Model it bit-for-bit or keep it \
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

        // --- WONDER GUARD (`gen3_wonder_guard_v1`) on the FIXED-DAMAGE path: a fixed-damage move
        //     (Seismic Toss / Night Shade / Sonic Boom / Dragon Rage / Super Fang / Counter /
        //     Mirror Coat / Endeavor) into a Wonder Guard holder is BLOCKED unless STRICTLY
        //     super-effective — the SAME gate as `run_move`'s normal-damage path. Fixed-damage
        //     moves route to `run_fixed_damage_move` BEFORE the `category == Status` branch, so
        //     they bypassed the `run_move` WG gate: a NEUTRAL one (Dragon Rage vs Bug/Ghost) would
        //     DEAL its fixed number (KO'ing the 1-HP Shedinja → skipping the endTurn Quick Claw →
        //     a SEED desync, ab_2_13), and a 0×-immune one (Counter Fighting → Ghost) would emit
        //     the PLAIN type `-immune` instead of the WG byte form (ab_5_20). It fires at the
        //     `TryHit` event: AFTER the accuracy roll (`acc_hit`-gated — a MISS never reaches
        //     TryHit → the plain type-immune / genuine-miss returns below win) and BEFORE damage
        //     (a fixed-damage move draws NO crit/damage roll → the block draws ONLY the accuracy
        //     roll already rolled, the SAME count as the plain type-immune path). Placed BEFORE
        //     `move_is_immune` so a 0×-type-immune fixed-damage move ALSO routes through WG's
        //     `-immune` (the byte-form distinction). Byte-fuzz master-seed 80808, ab_5_20/ab_2_13.
        if acc_hit
            && !targets_self
            && move_type.is_some()
            && dex
                .ability(&self.sides[foe].pokemon[foe_slot].ability)
                .map(|a| a.wonder_guard)
                .unwrap_or(false)
        {
            let def_types = mon_types(&self.sides[foe].pokemon[foe_slot], dex);
            let connects = match move_type {
                Some(t) => dex.type_chart().effectiveness(t, &def_types) > 1.0,
                None => true, // unreachable (guarded by move_type.is_some())
            };
            if !connects {
                // [EMIT] `|move|<user>|<Name>|<foe>` then `|-immune|<foe>|[from] ability: Wonder
                // Guard`. Observation-only: draws nothing beyond the accuracy roll already drawn.
                if self.logging() {
                    let user = self.mon_ref(side, slot, dex);
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.move_used(&user, move_name, Some(&target), false, false);
                    self.log.immune_from_ability(&target, "Wonder Guard");
                }
                return MoveResolution::done(false, false, false);
            }
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
            // ENDURE clamps a FIXED-damage Move hit too (`gen3_move_coverage_batch6_v1`
            // — probe ED6: a lethal Seismic Toss into an endurer leaves it at 1 HP;
            // endure −10 precedes Focus Band's −40). DRAW-FREE.
            let amount = self.endure_clamp(foe, foe_slot, amount, dex);
            // FOCUS BAND: a fixed-damage move is a MOVE-effect Damage event — the roll
            // draws and a lethal Seismic Toss can be survived at 1. `gen3_ability_batch4_v1`.
            let pre_hp = self.sides[foe].pokemon[foe_slot].hp;
            let amount = self.focus_band_damage(foe, foe_slot, amount, true, true, dex);
            self.apply_damage(foe, foe_slot, amount);
            // DESTINY BOND trigger record (`gen3_move_coverage_batch6_v1`): a
            // fixed-damage foe Move KO — incl. a Counter/Mirror-Coat RETURN (a foe
            // Move) — with the DB volatile up marks the pending mutual faint. DRAW-FREE.
            if self.sides[foe].pokemon[foe_slot].hp == 0
                && self.sides[foe].pokemon[foe_slot].destiny_bond
            {
                self.sides[foe].pokemon[foe_slot].destiny_bond_ko_by = Some(side);
            }
            // COUNTER / MIRROR COAT recorder (`gen3_move_coverage_batch5_v1`): a
            // fixed-damage foe hit IS a Move-effect Damage event — Seismic Toss
            // (Fighting → Physical) arms Counter for 2×100 (probed CS), Dragon Rage
            // (Dragon → Special) would arm Mirror Coat. The gen3 category is the
            // TYPE-derived one (`derive_category` with a nonzero bp forces the type
            // branch); `dealt` = the post-Focus-Band applied amount, clamped.
            let dealt = amount.min(pre_hp);
            let cat = crate::dex::moves::derive_category(3, 1, move_type);
            self.record_reactive_hit(foe, foe_slot, cat, move_id, dealt);
            // FOCUS PUNCH lostFocus (`gen3_move_coverage_batch4_v1`,
            // `focuspunch.condition.onHit`): a FIXED-DAMAGE hit is a NON-Status move
            // that HIT the FP user DIRECTLY, so it cancels a queued Focus Punch exactly
            // like the normal damaging path (SIM-VERIFIED by the batch-5 e2e admission
            // itself — e2e_202 dec44: a Blissey Seismic Toss into a Focus-Punch
            // Dragonite → `|cant|…|Focus Punch`; the port's missing set let the punch
            // land, KO'ing the Blissey — a LATENT gap unreachable while the
            // fixed-damage family was blocklist-shadowed out of the e2e). DRAW-FREE.
            if let Some(fp) = self.sides[foe].pokemon[foe_slot].focus_punch.as_mut() {
                *fp = true;
            }
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
        // --- CONTACT_PROC (+ Rough Skin recoil) on a CONTACT fixed-damage hit
        //     (`gen3_ability_batch2_v1` × the fixed-damage family — Seismic Toss /
        //     Super Fang / Counter / Endeavor carry `flags.contact`; Night Shade /
        //     Sonic Boom / Dragon Rage / Mirror Coat do NOT): the DEFENDER's
        //     `onDamagingHit` fires for a fixed-damage hit exactly like the normal
        //     damaging path (AFTER the appended King's Rock secondary, NOT behind a
        //     sub). THE e2e_7 FIX (`gen3_move_coverage_batch6_v1` regen): the sim
        //     rolled Effect Spore's `randomChance(1,10)` after a Seismic Toss into
        //     Breloom (mods/gen3/abilities.js onDamagingHit) while the port drew
        //     nothing — a LATENT batch-5-era gap (fixed damage was only e2e-admitted
        //     in batch 5, and no ST-into-a-contact-proc-holder board was sampled until
        //     the batch-6 corpus reshuffle). Pin MC99. ---
        //     PHASE: `All` — no fire-thaw precedes this tail (no modeled fixed-damage move is
        //     Fire-typed), so there is no un-ordered handler for an ORDERED one to sort against
        //     and the `gen3_damaging_hit_order_v1` split is moot here.
        let is_contact = dex.moves(move_id).map(|m| m.contact).unwrap_or(false);
        if is_contact && !absorbed && amount > 0 {
            self.apply_contact_proc(side, slot, foe, foe_slot, DamagingHitPhase::All, dex);
        }
        if !absorbed && amount > 0 {
            self.apply_color_change(foe, foe_slot, move_type, dex);
        }

        // The move landed (a sub hit STILL fires the in-tryMoveHit Update). NO crit.
        MoveResolution::done(false, false, true)
    }

    /// The FUTURE-MOVE **CAST** (`gen3_move_coverage_batch4c_v1`, Doom Desire / Future
    /// Sight — the gen-3 `onTry`, probe-settled bit-for-bit vs
    /// `harness/probe_batch4c_doomdesire.js`):
    ///
    ///   * DOUBLE-CAST (any futuremove already pending on the TARGET slot — DD-after-FS
    ///     included, one condition per slot): FAILS with a bare `|move|` line (NO
    ///     `[still]`, NO `-fail`, NO `-start`), ZERO move draws — `addSlotCondition`
    ///     fails BEFORE `getDamage`. PP was already deducted by the caller (probed 7→6).
    ///   * else: `addSlotCondition(target,'futuremove')` → `getDamage(source, target,
    ///     {bp 120(DD)/80(FS), category Physical(DD)/Special(FS), type '???',
    ///     willCrit:false})` IMMEDIATELY — the CAST-TIME DAMAGE SNAPSHOT: typeless (no
    ///     STAB, no type chart → no immunity ever), NO accuracy roll, NO crit roll,
    ///     exactly ONE `random(16)` (the modifyDamage randomizer), with cast-time
    ///     stats/boosts/items/burn/screens (the full getDamage semantics — attacker Calm
    ///     Mind / defender Amnesia AFTER the cast change NOTHING; a cast at Skarmory hits
    ///     a switched-in Blissey with the Skarmory-computed number). The `modifyDamage`
    ///     run ALSO fires `runEvent('ModifyDamagePhase1')` — a BOTH-screens defender side
    ///     draws the size-2 handler tie-shuffle exactly like a normal damaging hit
    ///     (mirrored control flow; unprobed only because no probe scenario stacked both
    ///     screens at cast). Emits `|-start|<caster>|Doom Desire` (on the CASTER, no
    ///     `move:` prefix) and returns null — `landed` FALSE (no in-tryMoveHit Update; a
    ///     tie-turn cast control probed 8 draws vs idle 7, the getDamage being the only
    ///     delta). A cast-turn Protect on the target does NOT block the cast (onTry
    ///     precedes TryHit — the caller gates this path before the protect block).
    #[allow(clippy::too_many_arguments)]
    fn run_future_move_cast(
        &mut self,
        side: usize,
        slot: usize,
        foe: usize,
        foe_slot: usize,
        base_power: u16,
        category: MoveCategory,
        accuracy: u16,
        move_id: &str,
        move_name: &str,
        dex: &Dex,
    ) -> MoveResolution {
        // [EMIT] the announce — BOTH arms show the plain `|move|<user>|<Name>|<target>`
        // (the double-cast fail is a bare `|move|` line, probed: no [still], no -fail).
        if self.logging() {
            let user = self.mon_ref(side, slot, dex);
            let target = self.mon_ref(foe, foe_slot, dex);
            self.log.move_used(&user, move_name, Some(&target), false, false);
        }
        if self.sides[foe].future_move.is_some() {
            // DOUBLE-CAST fail: draw-free (the pending strike resolves on schedule).
            return MoveResolution::done(false, false, false);
        }
        // The CAST-TIME SNAPSHOT: full getDamage semantics over a typeless move.
        let mut ctx = self.build_damage_context(
            side, slot, foe, foe_slot, base_power, None, category, false, dex,
        );
        ctx.crit = false; // willCrit: false — no crit roll, crit never true
        // The ModifyDamagePhase1 handler-sort shuffle (the FULL screen tie group across BOTH
        // sides — `onAnyModifyDamagePhase1`; `k-1` draws for k screens), mirroring the damaging
        // path's position: inside modifyDamage, BEFORE the randomizer.
        self.modify_damage_phase1_shuffle();
        let dmg = crate::damage::calc_damage(&ctx, dex);
        // ONE random(16) — the getDamage randomizer (the cast's only draw).
        let r = self.prng.random_below(16) as usize;
        let stored = dmg.rolls[r];
        self.sides[foe].future_move = Some(FutureMove {
            duration: FUTURE_MOVE_DURATION,
            damage: stored,
            move_id: move_id.to_string(),
            accuracy,
            source_side: side,
            source_uid: self.sides[side].pokemon[slot].uid,
        });
        // [EMIT] `|-start|<caster>|Doom Desire` — on the CASTER, no `move:` prefix
        // (probe-observed shape).
        if self.logging() {
            let user = self.mon_ref(side, slot, dex);
            self.log.volatile_start(&user, move_name);
        }
        MoveResolution::done(false, false, false)
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
                self.sides[side].pokemon[slot].sleep_skipped = 0;
                // [EMIT] `|-curestatus|<mon>|slp|[msg]` — a natural sleep wake (the `[msg]`
                // shows the client wake message). Emitted BEFORE the mon's own `|move|`
                // (it wakes then moves the same turn). Observation-only.
                if self.logging() {
                    let mon_ref = self.mon_ref(side, slot, dex);
                    self.log.curestatus(&mon_ref, "slp", true);
                }
                // (fall through to the next-lower handlers, like the sim's `return`
                //  from the slp handler — but slp is exclusive, so none else fire.)
                // A WAKE-turn Sleep Talk then executes AWAKE → its onTry fails SILENTLY
                // in the Sleep Talk arm (`gen3_move_coverage_batch5_v1`, probed).
            } else {
                self.sides[side].pokemon[slot].status = Some(Status::Sleep(next));
                // [EMIT] `|cant|<mon>|slp` — still asleep. Emitted for BOTH branches
                // below (the sim prints the cant BEFORE checking `move.sleepUsable`).
                if self.logging() {
                    let mon_ref = self.mon_ref(side, slot, dex);
                    self.log.cant(&mon_ref, "slp", None);
                }
                // --- `move.sleepUsable` (`gen3_move_coverage_batch5_v1`, SLEEP TALK —
                //     the ONLY modeled sleepUsable move; Snore is UNMODELED — it
                //     FAIL-LOUD-panics at run_move's top, `snore_panics_fail_loud`):
                //     the slp handler `return`s instead of
                //     `return false`, so the move PROCEEDS while still asleep — and the
                //     gen3 slp state does `skippedTime++` (a normal blocked cant RESETS
                //     it to 0; the onSwitchIn restore lives in `run_switch`). The bare
                //     `return` means LOWER handlers still run (an asleep+confused
                //     sleep-talker draws its confusion gate), mirroring the sim's
                //     handler chain. DRAW-FREE either way. ---
                let sleep_usable = !struggle
                    && self
                        .move_at(side, slot, move_index, dex)
                        .map(|m| m.id == "sleeptalk")
                        .unwrap_or(false);
                if sleep_usable {
                    self.sides[side].pokemon[slot].sleep_skipped =
                        self.sides[side].pokemon[slot].sleep_skipped.saturating_add(1);
                    // fall through to the lower handlers → the move proceeds.
                } else {
                    self.sides[side].pokemon[slot].sleep_skipped = 0;
                    return false; // still asleep → ABORT (no further draws).
                }
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
                        .map(|m| m.display_name().to_string())
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
                            .map(|m| m.display_name().to_string())
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
                // [EMIT] `|-end|<mon>|confusion` (`gen3_omniscient_byte_fuzz_v1` — the sim's
                // `confusion.onEnd`, fired by `removeVolatile('confusion')` when the counter
                // hits 0; then the mon acts NORMALLY). Draw-free / observation-only.
                if self.logging() {
                    let m = self.mon_ref(side, slot, dex);
                    self.log.volatile_end(&m, "confusion");
                }
            } else {
                self.sides[side].pokemon[slot].confusion = Some(next);
                // [EMIT] `|-activate|<mon>|confusion` (`gen3_omniscient_byte_fuzz_v1` FORM
                // 6b) — the sim's `confusion.onBeforeMove` reveal, emitted (still confused)
                // BEFORE the `randomChance(1,2)` self-hit roll. Draw-free / observation-only.
                if self.logging() {
                    let m = self.mon_ref(side, slot, dex);
                    self.log.activate(&m, "confusion", None);
                }
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
                    .map(|m| m.display_name().to_string())
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
            mon.transformed(),
            None,
            MoveCategory::Physical,
            self_statused,
            dex,
        );
        let def_stat_mods = resolve_def_stat_mods(
            &mon.item,
            &ability,
            &mon.species_id,
            mon.transformed(),
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
                minimize_doubles: false,
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
            defender_minimized: false,
            immune: false,
            // Flash Fire is Fire-type-gated; the typeless '???' self-hit is NOT Fire, so an
            // FF-armed mon that confusion-self-hits gets NO ×1.5 (probe-consistent). false.
            flash_fire: false,
        };
        let dmg = calc_damage(&ctx, dex);
        // --- THE `runEvent('ModifyDamagePhase1')` HANDLER-SORT SHUFFLE (the confusion
        //     self-hit sibling of the normal-move `modify_damage_phase1_shuffle`,
        //     `gen3_confusion_self_hit_screen_shuffle_v1`) — gen-4 confusion (gen-3-inherited)
        //     runs the FULL `getDamage(self,self,40)` → `modifyDamage` → `runEvent(
        //     'ModifyDamagePhase1')`, which GATHERS the screens' `onAnyModifyDamagePhase1`
        //     SIDE handlers exactly like a normal hit — once per side across BOTH sides (the
        //     `onAny` prefix; `findEventHandlers`'s Side loop pushes them for every side). So
        //     BOTH sides carrying a screen (or any ≥2 combo) TIE → a size-k Fisher-Yates
        //     speed-sort shuffle draws `k-1`. The screen handlers' `target !== source` guard
        //     makes them a DAMAGE no-op for a self-hit (no reduction — `ctx.reflect/light_screen`
        //     stay false above), but the handlers still GATHER, so the shuffle DRAWS. The port
        //     used to skip it entirely → a missing draw whenever ≥2 screens were up during a
        //     confusion self-hit (random-mode byte fuzz find ab_3_17 @ master-seed 100125,
        //     Tangela self-hits with Reflect on BOTH sides → `shuffle(len=2)`; sim-probe-confirmed
        //     the draw sits AFTER the confusion `randomChance(1,2)`, BEFORE the `random(16)`).
        //     `modify_damage_phase1_shuffle` counts every screen on both sides (once each) — the
        //     same count the sim's gather produces — and draws nothing for 0/1. ---
        self.modify_damage_phase1_shuffle();
        // ONE random(16) randomizer roll for the self-hit damage.
        let r = self.prng.random_below(16) as usize;
        let realized = dmg.rolls[r];
        // FOCUS BAND: the confusion self-hit is dealt with `effectType: 'Move'` (the
        // gen4-inherited condition) — the roll draws AFTER the randomizer (probe
        // `probe_focusband_confusion_rng.js`) and a lethal self-hit can be survived.
        let realized = self.focus_band_damage(side, slot, realized, true, false, dex);
        self.apply_damage(side, slot, realized);
        // [EMIT] `|-damage|<mon>|<HP>|[from] confusion` (`gen3_omniscient_byte_fuzz_v1` —
        // the confusion self-hit damage line, following the `-activate|<mon>|confusion`
        // FORM 6b line from `on_before_move`). Draw-free / observation-only.
        if self.logging() {
            let m = self.mon_ref(side, slot, dex);
            let hp = self.hp_status(side, slot);
            self.log.damage(&m, &hp, Some(&Cause::Bare("confusion".to_string())));
        }
    }

    /// The resolved gen-3 move type for a move id (a thin dex lookup — used by the
    /// status-fail classifier to reproduce the type-immunity gate outside `run_move`).
    pub(crate) fn move_type_of(&self, move_id: &str, dex: &Dex) -> Option<Type> {
        dex.moves(move_id).and_then(|m| m.move_type)
    }

    /// Whether `move_id` is a SOUND move (`flags.sound`, `gen3_ability_batch2_v1`) — Soundproof
    /// is immune to it. A thin dex lookup (Sing / Grass Whistle / Roar / Perish Song are sound).
    pub(crate) fn move_is_sound(&self, move_id: &str, dex: &Dex) -> bool {
        dex.moves(move_id).map(|m| m.is_sound).unwrap_or(false)
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

        let mv = MoveInput {
            // Defaulted here (no move id in scope); `run_move` sets it — together with
            // `defender_minimized` — right after this returns, where the id IS in scope.
            minimize_doubles: false,
            base_power,
            move_type,
            category,
            halves_defense: halves_def,
        };

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
            atk_mon.transformed(),
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
            def_mon.transformed(),
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
            // MINIMIZE (`gen3_minimize_v1`): the DEFENDER's volatile. Paired with the move's
            // own `minimize_doubles` flag (set by the caller, which has the move id) it
            // doubles the damage at the FINAL ModifyDamage step.
            defender_minimized: self.sides[foe].pokemon[foe_slot].minimize,
            crit: false, // set by the caller after the crit roll
            weather,
            // SCREENS (`gen3_move_coverage_batch2_v1`): the DEFENDER's side conditions halve
            // the incoming damage — Reflect ×0.5 physical, Light Screen ×0.5 special (both
            // crit-bypassed, folded in `damage.rs::modify_damage`). Read from the DEFENDING
            // (`foe`) side's turn counters; a non-zero count = the screen is up.
            reflect: self.sides[foe].reflect > 0,
            light_screen: self.sides[foe].light_screen > 0,
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
        // The crash `getDamage` draws the SAME crit + damage-roll sequence as a landed hit
        // (the `onModifyCritRatio` denominator shift — Focus Energy / a CRIT_ITEM — + the
        // crit-immune draw-free override).
        let eff_crit_ratio = self.effective_crit_ratio(side, slot, crit_ratio, dex);
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
        let crash = self.focus_band_damage(side, slot, crash, true, false, dex);
        self.apply_damage(side, slot, crash);
        // [EMIT] `|-damage|<user>|<HP>` (after the `-miss` / Protect `-activate` line).
        if self.logging() {
            let user = self.mon_ref(side, slot, dex);
            let hp = self.hp_status(side, slot);
            self.log.damage(&user, &hp, None);
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
    pub(crate) fn absorb_into_sub(&mut self, side: usize, slot: usize, dmg: u16) -> SubAbsorb {
        let mon = &mut self.sides[side].pokemon[slot];
        let sub_hp = match mon.substitute {
            Some(hp) => hp,
            None => return SubAbsorb::NoSub, // no sub → the caller hits the mon
        };
        let taken = dmg.min(sub_hp); // excess does NOT carry to the mon (gen-3)
        let remaining = sub_hp - taken;
        if remaining == 0 {
            mon.substitute = None;
            // `addVolatile('substitutebroken')` rides the same branch as the removal in the
            // sim (gen4 `moves.ts:1303-1305`, inherited by gen3). Inert in gen 3 — it exists
            // so the readouts can name it (`gen3_substitute_broken_volatile_v1`).
            mon.substitute_broken = true;
            SubAbsorb::Broke // the sub is gone → `-end`
        } else {
            mon.substitute = Some(remaining);
            SubAbsorb::Held // the sub survived → `-activate …|[damage]`
        }
    }
}
