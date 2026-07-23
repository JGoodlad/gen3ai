use crate::dex::{to_id, Dex};
use super::helpers::*;

impl crate::state::BattleState {

    /// Apply a damaging move's SECONDARY effects (battle-actions.ts secondaries(),
    /// 1357-1373): for each surviving secondary, draw ONE `random_below(100)` and
    /// apply the effect if `roll < chance`. The chance is the RAW move chance
    /// (Serene Grace ×2 baked in at onModifyMove; Shield Dust on the DEFENDER FILTERS
    /// the target secondaries OUT — they then draw NO random(100), a draw-COUNT
    /// effect). For our 4 test moves each is ONE secondary (par/frz/flinch). A KO'd
    /// target still draws (setStatus no-ops on hp==0). The status applies via the
    /// onTrySetStatus gates (already-statused / type-immunity — draw-free).
    pub(crate) fn apply_secondaries(
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
    pub(crate) fn apply_one_secondary(
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
                // FOCUS PUNCH flinch-immunity (`gen3_move_coverage_batch4_v1`,
                // `focuspunch.condition.onTryAddVolatile` → null for `flinch`): a Focus
                // Punch user (its `focuspunch` volatile up) CANNOT be flinched — the flinch
                // secondary's random(100) was ALREADY drawn (draw-then-block, like Inner
                // Focus), so the count is unchanged, but the flinch is NOT set. This is
                // DRAW-RELEVANT via the residual duration-handler tally: a mon with BOTH
                // `focus_punch` AND `flinch` would register TWO tied NO_ORDER duration
                // handlers → an intra-mon tie-shuffle the sim never draws (VERIFIED: the
                // Rock-Slide-into-a-FP-user turn draws no such shuffle).
                let mon = &mut self.sides[foe].pokemon[foe_slot];
                if !mon.fainted
                    && mon.hp > 0
                    && to_id(&mon.ability) != "innerfocus"
                    && mon.focus_punch.is_none()
                {
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
                self.apply_secondary_boost(side, slot, foe, foe_slot, false, boosts, false, dex);
            }
            "self_boost" => {
                self.apply_secondary_boost(side, slot, foe, foe_slot, true, boosts, false, dex);
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
    pub(crate) fn add_confusion(&mut self, foe: usize, foe_slot: usize, _dex: &Dex) {
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
        // [EMIT] `|-start|<mon>|confusion` (`gen3_omniscient_byte_fuzz_v1` FORM 6a) — the
        // sim's `confusion.onStart` reveal on a successful add (Water Pulse / Confuse Ray /
        // Dynamic Punch / …). Draw-free / observation-only.
        if self.logging() {
            let m = self.mon_ref(foe, foe_slot, _dex);
            self.log.volatile_start(&m, "confusion");
        }
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
    pub(crate) fn apply_secondary_boost(
        &mut self,
        side: usize,
        slot: usize,
        foe: usize,
        foe_slot: usize,
        want_self: bool,
        boosts: &[crate::dex::moves::SecondaryBoost],
        // Whether this foe stat-DROP is a MOVE's PRIMARY effect (Screech / Charm / Memento /
        // Tickle) vs a damaging move's SECONDARY (Crunch's −SpD). The sim's Clear Body /
        // White Smoke / Hyper Cutter `onTryBoost` emits the `-fail|unboost|[from] ability|[of]`
        // line ONLY when `!effect.secondaries` — i.e. for a primary drop
        // (`gen3_omniscient_byte_fuzz_v1` FORM 2, Screech-into-Clear-Body). A secondary drop
        // blocks SILENTLY. Ignored for a self-raise.
        is_primary_drop: bool,
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
        // [EMIT] the Clear Body / White Smoke / Hyper Cutter / Keen Eye `-fail` — ONCE, BEFORE
        // any surviving `-boost`/`-unboost` (the sim's `onTryBoost` runs over the whole boost
        // object first). Only for a PRIMARY drop (FORM 2); a SECONDARY drop is silent. The
        // single-stat blockers (Hyper Cutter / Keen Eye) carry their `unboost|<Stat>` token
        // (`unboost_fail_stat_token`), the whole-table ones (Clear Body / White Smoke) none.
        if is_primary_drop && !want_self && self.logging() {
            let any_blocked = spec
                .boosts
                .iter()
                .any(|&(idx, _)| stat_drop_blocked(&foe_ability, idx));
            if any_blocked {
                let name = _dex.ability(&foe_ability).map(|a| a.name.clone()).unwrap_or_else(|| foe_ability.clone());
                let stat = unboost_fail_stat_token(&foe_ability);
                let target = self.mon_ref(t_side, t_slot, _dex);
                self.log.fail_unboost_from_ability(&target, &name, stat);
            }
        }
        for &(idx, stages) in &spec.boosts {
            if !want_self && stat_drop_blocked(&foe_ability, idx) {
                continue; // Clear Body / White Smoke / Hyper Cutter / Keen Eye.
            }
            let cur = self.sides[t_side].pokemon[t_slot].boosts[idx] as i32;
            let next = (cur + stages as i32).clamp(-6, 6);
            self.sides[t_side].pokemon[t_slot].boosts[idx] = next as i8;
            // [EMIT] `|-boost|`/`|-unboost|<target>|<stat>|<mag>`. A PRIMARY foe-drop (a
            // stat-drop MOVE — Charm / Feather Dance / Memento) STILL emits the line at a
            // 0-delta into the −6 floor (`|-unboost|<foe>|atk|0` — the sim's `boost()`
            // `!isSecondary && !isSelf` branch, `gen3_omniscient_byte_fuzz_v1` FORM 10), so it
            // routes through `boost_applied`; a SECONDARY drop (`is_primary_drop == false`)
            // keeps the zero-skipping `boost`. Observation-only.
            if self.logging() {
                let delta = (next - cur) as i8;
                let target = self.mon_ref(t_side, t_slot, _dex);
                if is_primary_drop && !want_self {
                    self.log.boost_applied(&target, idx, stages, delta, next as i8);
                } else {
                    self.log.boost(&target, idx, delta);
                }
            }
        }
    }
}
