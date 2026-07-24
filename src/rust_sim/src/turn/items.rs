use crate::dex::{to_id, Dex, Type};
use crate::protocol::Cause;
use super::helpers::*;

impl crate::state::BattleState {

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

    /// WHITE HERB — the BOOST_RESTORE trigger (`gen3_white_herb_v1`, `ItemData.boost_restore`).
    /// If the mon at `(side, slot)` holds White Herb AND has ANY boost stage `< 0`, restore
    /// ONLY the negative stages to 0 (positives untouched) and CONSUME the item, emitting
    /// `|-enditem|<mon>|White Herb` then `|-clearnegativeboost|<mon>|[silent]`. Otherwise a
    /// no-op (retained). DRAW-FREE (no PRNG). The resolved gen3 item fires from
    /// `onAnyAfterMove` / `onAnySwitchIn` / `onResidual(order 29)`, so this is called RIGHT
    /// AFTER any stat-drop resolves on the holder — its OWN self-drop move (Overheat /
    /// Superpower / Psycho Boost / Curse-non-ghost's −Spe, via `apply_self_drops`), a foe's
    /// stat-drop move / secondary (Growl / Screech / Crunch, via `apply_secondary_boost`), or a
    /// foe's Intimidate-on-switch-in (via `run_switch` / `start_with_switchins`). Restoring only
    /// the negatives is what makes +2-then-−1 (net +1) NOT trigger (no negative stage) while a
    /// genuine −1 on a different stat DOES (that lone stage restored, the positives kept).
    ///
    /// Emission order matches the resolved gen3 `useItem` (which emits `-enditem` BEFORE
    /// running `onUse`'s `setBoost` + `-clearnegativeboost`) — probe-confirmed
    /// (`harness/probe_batch89_abilities_items.js`: `-enditem White Herb` then
    /// `-clearnegativeboost [silent]`).
    pub(crate) fn white_herb_restore(&mut self, side: usize, slot: usize, dex: &Dex) {
        // Data-driven identify (BOOST_RESTORE class == White Herb, the sole gen-3 member).
        let is_wh = {
            let item_id = to_id(&self.sides[side].pokemon[slot].item);
            dex.item(&item_id).map_or(false, |i| i.boost_restore)
        };
        if !is_wh {
            return;
        }
        {
            let mon = &mut self.sides[side].pokemon[slot];
            if mon.fainted || !mon.boosts.iter().any(|&b| b < 0) {
                return; // no negative stage to restore ⇒ item retained (the net-positive case)
            }
            for b in mon.boosts.iter_mut() {
                if *b < 0 {
                    *b = 0;
                }
            }
            mon.item = String::new(); // single-use consumption
        }
        if self.logging() {
            // [EMIT] `|-enditem|<mon>|White Herb` then `|-clearnegativeboost|<mon>|[silent]`.
            let m = self.mon_ref(side, slot, dex);
            self.log.push_raw(format!("|-enditem|{m}|White Herb"));
            self.log.push_raw(format!("|-clearnegativeboost|{m}|[silent]"));
        }
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
    pub(crate) fn apply_berry_residual(&mut self, side: usize, slot: usize, dex: &Dex) {
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
                        // `add_confusion` ALREADY emits the `|-start|<mon>|confusion` reveal on a
                        // successful add — do NOT emit it a second time here (the prior port
                        // double-emitted the Figy/Mago/Iapapa/Aguav/Wiki `-start confusion`,
                        // `gen3_omniscient_byte_fuzz_v1`).
                        self.add_confusion(side, slot, dex);
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
    /// `|-boost|<mon>|<stat>|<n>` reveal. **AT the +6 cap the sim STILL emits a delta-0
    /// `-boost` line** (`gen3_liechi_atcap_boost_v1`, the omniscient-byte-fuzz find ab_1_6 —
    /// a Liechi Berry eaten by a Swords-Danced-to-+6 Scyther: `|-enditem|…|Liechi Berry|[eat]`
    /// then `|-boost|p2a: Scyther|atk|0`, PROBE-verified via `harness/probe_liechi_atcap.js`).
    /// The CAUSE differs by delta (battle.js:1673-1707): delta > 0 → the Item branch carries
    /// `[from] item: <Berry>`; delta == 0 (`boostBy == 0`) falls to the `!isSecondary && !isSelf`
    /// branch → NO cause (the Agility@+6 `spe|0` precedent). A pinch berry is always +N and the
    /// holder is at +6, so `msg` is always `-boost` (never `-unboost`). The eat already happened
    /// (the caller consumed the item first).
    fn berry_boost(&mut self, side: usize, slot: usize, stat: usize, stages: i8, item_name: &str, dex: &Dex) {
        let s = &mut self.sides[side].pokemon[slot].boosts[stat];
        let before = *s;
        *s = (*s + stages).min(6);
        let delta = self.sides[side].pokemon[slot].boosts[stat] - before;
        if self.logging() {
            let m = self.mon_ref(side, slot, dex);
            let stat_tok = crate::protocol::STAT_TOKENS[stat];
            if delta > 0 {
                self.log
                    .push_raw(format!("|-boost|{m}|{stat_tok}|{delta}|[from] item: {item_name}"));
            } else {
                // At the +6 cap: the delta-0 `-boost` line, NO `[from] item:` cause.
                self.log.push_raw(format!("|-boost|{m}|{stat_tok}|0"));
            }
        }
    }

    /// The `eachEvent('Update')` ITEM handlers (`gen3_berry_trace_shedskin_v1`) — the CURE
    /// berries + Leppa, run at EVERY Update site right after its tie-shuffle, per active in
    /// the shuffled speed order (the per-mon effects are independent + DRAW-FREE, so the
    /// order only shapes protocol line order). The HEAL/PINCH berries do NOT fire here
    /// (their gen3 onUpdate is mod-DELETED — residual-only). Probe: a cure lands at the
    /// FIRST Update after the status (before the holder's own move — it never rolls
    /// full-para that turn); an eaten berry never fires again.
    pub(crate) fn run_update_items(&mut self, order: &[usize], dex: &Dex) {
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
            // [EMIT] Showdown's STATUS_IMMUNE `onUpdate` (abilities.ts) does
            // `this.add('-activate', pokemon, 'ability: <A>'); pokemon.cureStatus();` → TWO
            // lines the port used to skip (setting `status = None` SILENTLY), so the next
            // `|move|` slid up into the 2-line gap (golden ab_1370_13 [256-259]:
            // `|-activate|p1a: Porygon2|ability: Immunity` → `|-curestatus|p1a: Porygon2|tox|[msg]`).
            // `gen3_statusimmune_onupdate_cure_v1` (byte). DRAW-FREE (the cure was already
            // applied; the emit is protocol-only). Emit BEFORE clearing so `tok`/the ident hold.
            if self.logging() {
                let ability_name = dex
                    .ability(&to_id(&self.sides[side].pokemon[slot].ability))
                    .map(|a| a.name.clone())
                    .unwrap_or_else(|| self.sides[side].pokemon[slot].ability.clone());
                let m = self.mon_ref(side, slot, dex);
                self.log.activate(&m, &format!("ability: {ability_name}"), None);
                self.log.curestatus(&m, tok, true);
            }
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
    pub(crate) fn berry_after_set_status(&mut self, side: usize, slot: usize, dex: &Dex) {
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

    /// Water Absorb / Volt Absorb `onTryHit` heal (`floor(maxhp/4)`, capped at
    /// maxhp, no-op at full HP — gen3 `this.heal(target.baseMaxhp/4)`). Called on the
    /// immune short-circuit when the defender's ability is the matching absorb for
    /// the move type. DRAW-FREE (heal consumes no PRNG). A faint-guard: a 0-HP mon
    /// never reaches here (the move that KO'd it already resolved), but match
    /// `apply_leftovers`'s guard for safety.
    /// Returns whether the absorb actually HEALED (hp increased). The sim's
    /// `voltabsorb`/`waterabsorb` `onTryHit` heals `maxhp/4`; the emitted line is `-heal`
    /// when the heal lands and `-immune` when it doesn't (full HP) —
    /// `gen3_omniscient_byte_fuzz_v1` FORM 9 (the F3 capture only exercised the full-HP
    /// `-immune` case). The caller reads this to choose the line.
    pub(crate) fn apply_absorb_heal(&mut self, side: usize, slot: usize, move_type: Option<Type>) -> bool {
        let mon = &mut self.sides[side].pokemon[slot];
        if mon.fainted || mon.hp == 0 {
            return false;
        }
        let ability = to_id(&mon.ability);
        let heals = matches!(
            (ability.as_str(), move_type),
            ("waterabsorb", Some(Type::Water)) | ("voltabsorb", Some(Type::Electric))
        );
        if !heals {
            return false;
        }
        let heal = mon.maxhp / 4; // floor
        let new_hp = (mon.hp + heal).min(mon.maxhp);
        let healed = new_hp > mon.hp;
        mon.hp = new_hp;
        healed
    }

    /// ITEM REMOVAL (`gen3_move_coverage_batch1_v1`, the `onAfterHit` item mechanics —
    /// KNOCK OFF removes the target's item; THIEF / COVET STEAL it iff the attacker holds
    /// none). Runs ONLY on a hit that DAMAGED THE MON (`!absorbed` — the sim's `onAfterHit`
    /// iterates the `damagedTargets`, which a sub-absorbed hit leaves empty; the target keeps
    /// its item behind a sub — probe-verified). DRAW-FREE (the `TakeItem` event / `takeItem`
    /// consume no PRNG — probe-verified). gen3 gotchas (probe-settled vs the resolved dex):
    ///   - **Knock Off** removes but does NOT boost damage (gen3; the dmg-boost is gen4+),
    ///     emitting `|-enditem|<target>|<Item>|[from] move: Knock Off|[of] <user>` + the
    ///     gens-3-4 `|-hint|`. It sets `item = ""`.
    ///   - **Thief / Covet** STEAL: only if the ATTACKER holds NO item; the attacker GAINS the
    ///     target's item (`|-item|<user>|<Item>|[from] move: Thief/Covet|[of] <target>`), Thief
    ///     also emitting the silent `|-enditem|<target>|<Item>|[silent]|[from] move: Thief`.
    ///   - **Sticky Hold** BLOCKS all three (`onTakeItem` returns false for a foe source) —
    ///     `|-activate|<target>|ability: Sticky Hold`, item unchanged.
    ///   - **Mail** does NOT block these three (its `onTakeItem` returns false only for OTHER
    ///     take-item moves — Trick etc. — so knock/thief/covet go through; probe-verified).
    /// A target with NO item is a no-op (no line). `move_id` selects the mechanic.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_item_removal(
        &mut self,
        side: usize,
        slot: usize,
        target_side: usize,
        target_slot: usize,
        move_id: &str,
        dex: &Dex,
    ) {
        // No item to remove.
        if self.sides[target_side].pokemon[target_slot].item.is_empty() {
            return;
        }
        let is_steal = move_id == "thief" || move_id == "covet";
        // Thief/Covet only steal if the ATTACKER holds NO item.
        if is_steal && !self.sides[side].pokemon[slot].item.is_empty() {
            return;
        }
        // **THE gen3 `itemKnockedOff` GATE (pokemon.ts:1853)**: `target.takeItem(source)` returns
        // FALSE in gen≤4 when EITHER the target OR the source (attacker) has `itemKnockedOff` —
        // gen3 Knock Off makes the slot "unusable, cannot obtain a new item". Thief / Covet route
        // through `takeItem`, so a Thief/Covet whose ATTACKER was Knocked-Off (or whose TARGET
        // was) does NOTHING to items (no removal, no gain). VERIFIED vs the sim (`probe_batch1_
        // *`): the e2e_83 Skarmory (item Knocked-Off) Thiefs a Leftovers Salamence → the item
        // stays on Salamence, Skarmory gains nothing. Knock Off itself does NOT use `takeItem`
        // (it `runEvent("TakeItem")`s directly), so it is UNGATED — a fresh item is still removed.
        if is_steal
            && (self.sides[target_side].pokemon[target_slot].item_knocked_off
                || self.sides[side].pokemon[slot].item_knocked_off)
        {
            return;
        }
        // STICKY HOLD on the target BLOCKS the take (foe source). `-activate`, item unchanged.
        if to_id(&self.sides[target_side].pokemon[target_slot].ability) == "stickyhold" {
            if self.logging() {
                let target = self.mon_ref(target_side, target_slot, dex);
                self.log.activate(&target, "ability: Sticky Hold", None);
            }
            return;
        }
        // The item id + its display name (before we clear it).
        let item_id = self.sides[target_side].pokemon[target_slot].item.clone();
        let item_name = dex
            .item(&to_id(&item_id))
            .map(|i| i.name.clone())
            .unwrap_or_else(|| item_id.clone());
        // Remove from the target.
        self.sides[target_side].pokemon[target_slot].item = String::new();
        // CHOICE-LOCK RELEASE (`gen3_move_coverage_batch4_v1`, the e2e_126 fix): the Choice
        // lock is enforced by the ITEM's `choiceband.onDisableMove` — with the Choice Band
        // GONE (Thief steals it / Knock Off removes it), that handler no longer fires, so the
        // mon can freely pick any move again. The port's `choice_locked_move` is a cached
        // shadow of that item-gated lock, so clearing the item MUST release it (else the port
        // keeps rejecting the now-legal move → a decision-stream desync). VERIFIED vs the sim:
        // e2e_126 Skarmory Thiefs a Choice-Band Aerodactyl (both itemless → the steal fires),
        // and the sim then offers Aerodactyl ALL its moves (RockSlide after Earthquake). A
        // no-op for a non-Choice-item target (its lock is already None). DRAW-FREE.
        self.sides[target_side].pokemon[target_slot].choice_locked_move = None;
        if is_steal {
            // The attacker GAINS the item (only reached when the attacker was itemless). Store
            // the SAME string form the target held (the raw set value, e.g. "Leftovers"), so
            // the port's `MonState::item` field stays consistent (every read normalizes with
            // `to_id`, so the exact stored form is cosmetic — but keep it uniform).
            let move_name = if move_id == "thief" { "Thief" } else { "Covet" };
            self.sides[side].pokemon[slot].item = item_id.clone();
            if self.logging() {
                let target = self.mon_ref(target_side, target_slot, dex);
                let user = self.mon_ref(side, slot, dex);
                if move_id == "thief" {
                    self.log.enditem_thief_silent(&target, &item_name, &user);
                }
                self.log.item_stolen(&user, &item_name, move_name, &target);
            }
        } else {
            // KNOCK OFF: remove + mark the slot `itemKnockedOff` (gen3 — "unusable, cannot obtain
            // a new item"), so a later Thief/Covet on this mon (or by this mon) does nothing.
            self.sides[target_side].pokemon[target_slot].item_knocked_off = true;
            if self.logging() {
                let target = self.mon_ref(target_side, target_slot, dex);
                let user = self.mon_ref(side, slot, dex);
                self.log.enditem_knockoff(&target, &item_name, &user);
                self.log.hint(
                    "In Gens 3-4, Knock Off only makes the target's item unusable; \
                     it cannot obtain a new item.",
                    true, // gen4/moves.ts:703 passes `once` — fires ONCE per battle
                );
            }
        }
    }

    /// RAPID SPIN (`gen3_move_coverage_batch1_v1`, the `onAfterHit` + `onAfterSubDamage`
    /// hazard-clear): after a LANDED hit (mon OR sub — Rapid Spin carries BOTH hooks, so it
    /// clears behind a substitute too, unlike Knock Off — probe-verified), CLEAR the USER's
    /// OWN side hazards (Spikes) + the USER's Leech Seed + partial-trap. DRAW-FREE. gen3 has
    /// only Spikes among the hazards (Toxic Spikes / Stealth Rock are later gens); partial-trap
    /// is not modeled (no partial-trap move in scope) → a documented no-op. `dealt > 0` gates
    /// (a landed hit). Emitted: `|-end|<user>|Leech Seed|[from] move: Rapid Spin|[of] <user>`
    /// then `|-sideend|<user-side>|Spikes|[from] move: Rapid Spin|[of] <user>`, in that order
    /// (probe-verified).
    pub(crate) fn apply_rapid_spin(&mut self, side: usize, slot: usize, dex: &Dex) {
        // Clear the USER's Leech Seed FIRST (the sim order), if present.
        if self.sides[side].pokemon[slot].leech_seed.is_some() {
            self.sides[side].pokemon[slot].leech_seed = None;
            if self.logging() {
                let user = self.mon_ref(side, slot, dex);
                self.log.volatile_end_from_move(&user, "Leech Seed", "Rapid Spin", &user);
            }
        }
        // Clear the USER's OWN side Spikes (the only gen-3 hazard).
        if self.sides[side].spikes > 0 {
            self.sides[side].spikes = 0;
            if self.logging() {
                let side_ref =
                    crate::protocol::ProtocolBuilder::side_ref(side, &self.sides[side].name);
                let user = self.mon_ref(side, slot, dex);
                self.log.sideend_from_move(&side_ref, "Spikes", "Rapid Spin", &user);
            }
        }
        // Partial-trap: no partial-trap move is modeled → nothing to clear (documented no-op).
    }

    /// `emit_survive_zero`: when a Focus-Band survive nets ZERO damage (the holder is ALREADY
    /// at 1 HP → `hp - 1 == 0`), the sim STILL emits `|-damage|<mon>|1/<max>` after the
    /// `-activate`, but the standard damaging-path caller's `realized > 0` emission gate SKIPS
    /// it. Pass `true` at those `>0`-gated sites (main / fixed / futuremove-resolve — each of
    /// which also runs `endure_clamp` first) so this fn emits the plain `-damage` itself; pass
    /// `false` at the sites that emit `-damage` UNCONDITIONALLY (the confusion self-hit, whose
    /// line carries `[from] confusion`, and the jump-kick crash) and at every `is_move=false`
    /// caller (which never reaches the survive branch). Mirrors `endure_clamp`'s survive-at-1
    /// self-emit (`gen3_omniscient_byte_fuzz_v1`, the Focus-Band-at-1 repro ab_9_9 dec469).
    pub(crate) fn focus_band_damage(
        &mut self,
        side: usize,
        slot: usize,
        dmg: u16,
        is_move: bool,
        emit_survive_zero: bool,
        dex: &Dex,
    ) -> u16 {
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
            // When the holder is ALREADY at 1 HP the survive nets 0 (stays at 1) — self-emit
            // the `-damage|1/<max>` line the `>0`-gated caller would skip (apply_damage(0) is a
            // no-op, so current HP == post-apply HP). For hp > 1 the caller emits it (hp-1 > 0),
            // so this never double-emits.
            if hp == 1 && emit_survive_zero && self.logging() {
                let mon_ref = self.mon_ref(side, slot, dex);
                let hp_status = self.hp_status(side, slot);
                self.log.damage(&mon_ref, &hp_status, None);
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
    pub(crate) fn apply_kings_rock_secondary(
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
}
