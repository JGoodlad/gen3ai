use crate::dex::{to_id, Dex, Type};
use crate::event::single_event_ability_start;
use crate::protocol::Cause;
use crate::state::Status;
use super::*;
use super::helpers::*;

impl crate::state::BattleState {

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
        // The TRAP-MOVE volatile (`gen3_move_coverage_batch6_v1`, Mean Look / Spider Web
        // / Block — the linked `trapped` volatile): a FIRM trap (`trapped.onTrapPokemon`
        // → bare `tryTrap()`, NOT `'hidden'` — the Shadow-Tag request shape, probed).
        // NO type/grounded gate (a grounded GHOST IS trapped — T-scenarios). The link is
        // cleared eagerly the moment the TRAPPER leaves the field (execute_switch /
        // process_faints), so a live `trapped_by` here always means an active trapper.
        if me.trapped_by.is_some() {
            return true;
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
        // The trap-MOVE volatile (`gen3_move_coverage_batch6_v1`) is a FIRM trap: the
        // probed request shape is `trapped:true` on the very FIRST move request (no
        // `maybeTrapped` phase) + an `[Invalid choice]` reject with NO re-request —
        // byte-identical to the Shadow-Tag shape (`gen3_shadowtag_firm_trap_v1`).
        if self.sides[side].pokemon[self.sides[side].active].trapped_by.is_some() {
            return true;
        }
        let foe = &self.sides[1 - side].pokemon[self.sides[1 - side].active];
        to_id(&foe.ability) == "shadowtag"
    }

    /// Whether either side's active mon is fainted (the trailing-Quick-Claw gate).
    pub(crate) fn any_active_fainted(&self) -> bool {
        (0..2).any(|s| {
            let a = self.sides[s].active;
            self.sides[s].pokemon[a].fainted
        })
    }

    /// Process pending faints (`faintMessages` setting `fainted = true`): mark every
    /// 0-HP active mon `fainted` and decrement its side's `pokemon_left`. Runs at the
    /// END of a `runAction` — AFTER the in-`tryMoveHit` Update shuffle — so a 0-HP mon
    /// is excluded from `getAllActive()` (the later eachEvent / residual shuffles)
    /// only from this point on. Returns whether any active was newly fainted.
    pub(crate) fn process_faints(&mut self, dex: &Dex) -> bool {
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
        let mut ordered: Vec<usize> = {
            let mut o: Vec<usize> = std::mem::take(&mut self.faint_emit_queue);
            for side in 0..2 {
                if !o.contains(&side) {
                    o.push(side);
                }
            }
            o
        };

        let mut any = false;
        // A WORKLIST (index-based) rather than a plain iterator: DESTINY BOND's mutual
        // faint (`gen3_move_coverage_batch6_v1`) enqueues the KILLER mid-drain — the
        // sim's `destinybond.onFaint` calls `source.faint()`, which pushes the source
        // onto `faintQueue` and the SAME `faintMessages` while-loop processes it next.
        let mut i = 0usize;
        while i < ordered.len() {
            let side = ordered[i];
            i += 1;
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
                // clearVolatile also drops the CURSE volatile on faint
                // (`gen3_move_coverage_batch3_v1`) — a fainted cursed mon is no longer cursed
                // (like leech_seed; the golden reads the curse flag on a live mon).
                mon.curse = None;
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
                mon.last_move_was_self_overwrite = false;
                // clearVolatile also drops the FLASH FIRE activation on faint — a fainted FF
                // mon carries no boost, and if re-encoded must not show a stale `flash_fire`.
                mon.flash_fire = false;
                // clearVolatile also drops the FOCUS PUNCH + PURSUIT `duration: 1` volatiles on
                // faint (`gen3_move_coverage_batch4_v1`) — a fainted mon carries neither.
                mon.focus_punch = None;
                mon.pursuit = None;
                // ...and the BEAT UP `beatup` `duration: 1` volatile (`gen3_move_coverage_batch4b_v1`).
                mon.beat_up = false;
                // ...and the MUSTRECHARGE / TWOTURNMOVE volatiles (`gen3_move_coverage_
                // batch4c_v1`) — a fainted mon carries neither (a fainted charger's
                // replacement enters fully unlocked; the corpse never gathers a residual
                // duration handler either way).
                mon.must_recharge = false;
                mon.two_turn = None;
                // ...and the COUNTER/MIRROR COAT reactive volatile (`gen3_move_coverage_
                // batch5_v1`) — a fainted counter user's corpse must not gather a residual
                // duration handler; sleep_skipped is meaningless without the slp status.
                mon.reactive = None;
                mon.sleep_skipped = 0;
                // ...and the BATCH-6 volatiles (`gen3_move_coverage_batch6_v1`,
                // clearVolatile): ENCORE / PERISH / ENDURE / CHARGE / the trap-move
                // TRAPPED link / the MIMIC moveslot overlay (the `baseMoveSlots`
                // revert). The DESTINY BOND flag + its pending-KO record are consumed
                // BELOW (the mutual-faint chain reads them first).
                mon.encore = None;
                mon.perish = None;
                mon.endure = false;
                mon.charge = false;
                // SNATCH (`gen3_snatch_v1`): drop the singleturn volatile with the corpse
                // (the `beat_up` stale-flag hazard class — a fainted snatcher must not
                // gather a residual duration handler on re-entry).
                mon.snatch = false;
                mon.trapped_by = None;
                // YAWN (`gen3_yawn_v1`): drop the pending delayed-sleep volatile with the corpse
                // (clearVolatile) — a fainted mon carries no pending Yawn, and its re-entry must
                // not gather a stale residual duration handler (the `beat_up` stale-flag class).
                mon.yawn = None;
                // The pending switch-in Intimidate target (`gen3_intimidate_forced_replacement_v1`,
                // M3) — cleared with the corpse so a fainted mon carries no stale foe uid.
                mon.switchin_foe_uid = None;
                mon.restore_mimic_overlay();
                // DESTINY BOND (`gen3_move_coverage_batch6_v1`, `destinybond.onFaint`):
                // consume the pending-KO record (set at the qualifying FOE-Move damage
                // sites — never by a residual / sub-absorbed hit / futuremove) + drop
                // the volatile with the corpse.
                let db_killer = mon.destiny_bond_ko_by.take();
                mon.destiny_bond = false;
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
                // --- TRAP-LINK end on the TRAPPER's faint (`gen3_move_coverage_
                //     batch6_v1`, probe T9): a fainted trapper frees its target — the
                //     foe active's `trapped_by == this corpse's uid` link is removed
                //     draw-free (the freed mon's next request drops `trapped:true`).
                //     NOTE (revert-verified honesty): this clear is observationally
                //     REDUNDANT today — the corpse's forced replacement runs
                //     `execute_switch`, whose source-left clear removes the same link,
                //     and `is_trapped` already returns false while the foe is fainted
                //     — so no pin can distinguish its removal. Kept as the faithful
                //     mirror of the sim's clearVolatile → removeLinkedVolatiles
                //     (defense in depth for any future path that faints a trapper
                //     without an immediate replacement). ---
                {
                    let corpse_uid = self.sides[side].pokemon[slot].uid;
                    let foe = 1 - side;
                    let foe_active = self.sides[foe].active;
                    if self.sides[foe].pokemon[foe_active].trapped_by == Some(corpse_uid) {
                        self.sides[foe].pokemon[foe_active].trapped_by = None;
                    }
                }
                // --- DESTINY BOND's mutual faint (`gen3_move_coverage_batch6_v1`,
                //     `destinybond.onFaint` — the runEvent('Faint') handler): the gate
                //     (foe source + a Move effect + !futuremove) was encoded at the
                //     damage sites; here the chain runs DRAW-FREE (probed DB1/DB3):
                //     the corpse's `|faint|` FIRST (above), then
                //     `|-activate|<corpse>|move: Destiny Bond`, then `source.faint()` —
                //     the killer's HP zeroes and its side joins the SAME faintMessages
                //     drain (its `|faint|` emits when the worklist reaches it). A
                //     both-last-mons mutual faint ends winner="" — the gen-3 TIE. ---
                if let Some(killer_side) = db_killer {
                    // [EMIT] `|-activate|<corpse>|move: Destiny Bond` BEFORE the
                    // killer's `|faint|` (the probed order; the sim emits it
                    // unconditionally before `source.faint()` — which itself no-ops on
                    // an already-fainted source).
                    if self.logging() {
                        let corpse_ref = self.mon_ref(side, slot, dex);
                        self.log.activate(&corpse_ref, "move: Destiny Bond", None);
                    }
                    let kslot = self.sides[killer_side].active;
                    let killer = &mut self.sides[killer_side].pokemon[kslot];
                    if !killer.fainted && killer.hp > 0 {
                        killer.hp = 0;
                        // `source.faint()` — enqueue the killer for THIS drain (the
                        // remaining worklist may already contain its side from the
                        // both-sides fallback; only push when it does not).
                        if !ordered[i..].contains(&killer_side) {
                            ordered.push(killer_side);
                        }
                    }
                }
            }
        }
        any
    }

    /// The sim's **`faintMessages(lastFirst = true)`** — the INSTAFAINT drain
    /// (`spreadDamage(…, instafaint = true)`, battle.ts:2183-2187) that gen-4's LIQUID OOZE
    /// triggers via `this.damage(damage, null, null, null, true)`
    /// (`data/mods/gen4/abilities.ts::liquidooze.onSourceTryHeal` — gen3-inherited).
    ///
    /// `lastFirst` UNSHIFTS the LAST-queued corpse to the FRONT of `faintQueue`
    /// (battle.ts:2555-2558) before draining it, so on a Liquid-Ooze DOUBLE faint the mon the
    /// REVERSED heal just KO'd is announced BEFORE the one the drain/leech already KO'd —
    /// the opposite of plain enqueue order. SIM-PROBED (`harness/probe_rb_tail.js` S3): a
    /// Leech-Seed drain that KOs the seeded Liquid-Ooze holder AND whose reversed heal KOs the
    /// seeder emits
    /// ```text
    /// |-damage|p2a: Swalot|0 fnt|[from] Leech Seed|[of] p1a: Jumpluff
    /// |-damage|p1a: Jumpluff|0 fnt|[from] ability: Liquid Ooze|[of] p2a: Swalot
    /// |faint|p1a: Jumpluff      <- the LAST-enqueued corpse first
    /// |faint|p2a: Swalot
    /// ```
    /// Called ONLY when the reversal actually zeroed the healer's HP (the sim's
    /// `if (target.hp <= 0)` instafaint gate). Draw-free itself (`process_faints` may still
    /// fire a corpse's Cloud-Nine `onEnd` WeatherChange shuffle — exactly as `faintMessages`
    /// does at this point in the sim).
    pub(crate) fn liquid_ooze_instafaint(&mut self, healer_side: usize, dex: &Dex) {
        let slot = self.sides[healer_side].active;
        let m = &self.sides[healer_side].pokemon[slot];
        if m.hp != 0 || m.fainted {
            return;
        }
        // `lastFirst`: move the last queue entry to the front.
        if let Some(last) = self.faint_emit_queue.pop() {
            self.faint_emit_queue.insert(0, last);
        }
        self.process_faints(dex);
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
    pub(crate) fn cancel_active_actions(&mut self, queue: &mut Vec<QAction>) {
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
    pub(crate) fn check_fainted(&mut self) {
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
    pub(crate) fn switch_request_gate(&mut self) -> [bool; 2] {
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
    pub(crate) fn slot_of_uid(&self, side: usize, uid: usize) -> Option<usize> {
        self.sides[side].pokemon.iter().position(|m| m.uid == uid)
    }

    /// Whether `side` has a non-active, non-fainted bench mon to switch to.
    pub(crate) fn can_switch(&self, side: usize) -> bool {
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
    pub(crate) fn eligible_switch_ins(&self, side: usize) -> Vec<usize> {
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
    pub(crate) fn drag_in(&mut self, side: usize, dex: &Dex, queue: &mut Vec<QAction>) {
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
        //     `is_voluntary = false` — a phaze DRAG never fires the pursuit interrupt
        //     (the sim's `BeforeSwitchOut` is `!isDrag`-gated).
        self.execute_switch(side, target, true, false, dex, queue);
    }

    /// `is_voluntary` = a VOLUNTARY menu switch (`QAction::Switch`), the ONLY switch-out
    /// the sim runs `BeforeSwitchOut` for — so the PURSUIT INTERRUPT fires only for it.
    /// A Baton-Pass selfSwitch / faint replacement (`QAction::InstaSwitch`) and a phaze
    /// drag (`drag_in`) pass `false`. Orthogonal to `is_drag` (which distinguishes a phaze
    /// drag from a switch for the `|drag|`-vs-`|switch|` emit + the Baton-Pass copyVolatileFrom).
    pub(crate) fn execute_switch(&mut self, side: usize, target: usize, is_drag: bool, is_voluntary: bool, dex: &Dex, queue: &mut Vec<QAction>) {
        let active = self.sides[side].active;

        // --- PURSUIT INTERRUPT (`gen3_move_coverage_batch4_v1`,
        //     `pursuit.condition.onBeforeSwitchOut`, fired by `switchIn`'s `runEvent(
        //     'BeforeSwitchOut', oldActive)` at battle-actions.ts:94, `!isDrag`-gated): a
        //     VOLUNTARY MENU switch-out of a mon carrying the `pursuit` volatile lets the
        //     pursuer STRIKE the switching mon BEFORE the switch resolves. It is gated on
        //     `is_voluntary` (NOT merely `!is_drag`): the sim runs `runEvent('BeforeSwitchOut')`
        //     — which dispatches pursuit's `onBeforeSwitchOut` — ONLY for a menu switch. A phaze
        //     DRAG (`!isDrag`-gated in switchIn), a BATON-PASS selfSwitch (`batonpass.self.onHit`
        //     sets `skipBeforeSwitchOutEventFlag`, moves.ts:1109 — probe-settled: the passer is
        //     NOT struck, Pursuit runs normally against the entrant next decision), and a FAINT
        //     replacement (its corpse's `pursuit` volatile is already cleared) all pass
        //     `is_voluntary = false` → no interrupt. (`gen3_move_coverage_batch4_v1`, the
        //     bench-order-desync fix.) Runs at the very TOP of `execute_switch` (before the
        //     clearVolatile block below) since the strike targets the still-active switcher. ---
        if is_voluntary {
            if let Some(pursuer_uid) = self.sides[side].pokemon[active].pursuit {
                let pside = 1 - side;
                // The pursuer must still be its side's ACTIVE + ALIVE (the sim's
                // `if (!this.queue.cancelMove(source) || !source.hp) continue`); resolve its
                // queued Pursuit Move action to get the slot's move_index (the Pursuit slot).
                let strike = self.slot_of_uid(pside, pursuer_uid).and_then(|pslot| {
                    if self.sides[pside].active == pslot && !self.sides[pside].pokemon[pslot].fainted {
                        queue.iter().find_map(|a| match a {
                            QAction::Move { side: s, uid, move_index, .. }
                                if *s == pside && *uid == pursuer_uid =>
                            {
                                Some((pslot, *move_index))
                            }
                            _ => None,
                        })
                    } else {
                        None
                    }
                });
                if let Some((pslot, pmi)) = strike {
                    // FIRST-MOVER attribution: the pursuer strikes NOW (its `|move|Pursuit` is the
                    // FIRST action line emitted this turn — before the switcher's `|switch|`), so
                    // the sim's `firstMoverSince` reads the PURSUER as the first mover. Record the
                    // override so `boundary_record` reports `pside`, not the sorted-queue switch.
                    self.pursuit_first_mover = Some(pside);
                    // (a) `this.queue.cancelMove(source)` — remove the pursuer's queued Pursuit
                    //     from the queue so it does NOT also act this turn. DRAW-FREE.
                    queue.retain(
                        |a| !matches!(a, QAction::Move { side: s, uid, .. } if *s == pside && *uid == pursuer_uid),
                    );
                    // (b) [EMIT] `|-activate|<switcher>|move: Pursuit`.
                    if self.logging() {
                        let sw = self.mon_ref(side, active, dex);
                        self.log.activate(&sw, "move: Pursuit", None);
                    }
                    // (c) `runMove('pursuit', source, {target: switcher})` deducts the
                    //     pursuer's Pursuit PP with the SWITCHER as the target — so if the
                    //     SWITCHING mon (the Pursuit target, `self.sides[side].pokemon[active]`)
                    //     has **Pressure**, its `onDeductPP` returns 1 → the pursuer's Pursuit
                    //     loses an EXTRA 1 PP (−2 total). Pursuit is a foe-directed (`normal`)
                    //     move, so it always puts the switcher in `pressureTargets`. (The prior
                    //     "no Pressure extra" claim was WRONG — the per-side/request byte fuzzer's
                    //     `bab_7_1`: a Umbreon Pursuit intercepting a switching Pressure Moltres →
                    //     the sim's request shows Pursuit `pp:29` where the flat-1 port showed
                    //     `pp:30`; `gen3_pursuit_pressure_pp_v1`, the Snatch-Pressure `bab_4_16`
                    //     sibling.) + `source.moveUsed` (sets lastMove).
                    let pressure_extra =
                        to_id(&self.sides[side].pokemon[active].ability) == "pressure";
                    self.sides[pside].pokemon[pslot]
                        .deduct_pp(pmi, if pressure_extra { 2 } else { 1 });
                    self.sides[pside].pokemon[pslot].last_move = Some(pmi);
                    // (d) `useMove(pursuit, source, {target: switcher})` — the strike at ×2 BP +
                    //     never-miss (the transient `pursuit_strike` flag, read+cleared inside
                    //     run_move). The switcher is still at `self.sides[side].active`, so
                    //     run_move's foe/foe_slot resolve to it.
                    // [EMIT] the strike's `|move|<pursuer>|Pursuit|<switcher>|[from] Pursuit`
                    // — the sim's `useMove(pursuit, source, {sourceEffect: pursuit})` folds the
                    // `[from] Pursuit` tag into the announce (`gen3_omniscient_byte_fuzz_v1`,
                    // byte-fuzzer-surfaced: the port omitted it). One-shot, consumed by the
                    // strike's `move_used`.
                    self.log.set_next_move_from("Pursuit");
                    self.pursuit_strike = true;
                    let res = self.run_move(
                        MoveAction { side: pside, slot: pslot, move_index: pmi, struggle: false },
                        false,
                        false,
                        dex,
                    );
                    self.pursuit_strike = false; // belt-and-braces (run_move already cleared it)
                    if res.landed {
                        // The strike's in-`tryMoveHit` `eachEvent('Update')` (draws on a
                        // pursuer↔switcher speed tie; the switcher is hp-0-but-not-yet-fainted
                        // here, so it is still in getAllActive — process_faints runs below).
                        let upd = self.each_event_shuffle();
                        self.run_update_items(&upd, dex);
                    }
                    // (e) `if (useMove(...) && source.getItem().isChoice) addVolatile('choicelock')`
                    //     — a Choice-item pursuer (gen-3: Choice Band) locks to Pursuit.
                    if to_id(&self.sides[pside].pokemon[pslot].item) == "choiceband" {
                        self.sides[pside].pokemon[pslot].choice_locked_move = Some(pmi);
                    }
                    // (f) PURSUITFAINT: if the strike KO'd the switcher, process the faint NOW
                    //     (before the swap) — pokemon_left decrements + `fainted` is set — but the
                    //     ALREADY-CHOSEN switch STILL brings in the replacement (the gen 2-4
                    //     `-hint`, verified vs the sim probe #3: Snorlax still came in, QC drawn).
                    //     process_faints reads the ACTIVE (still the switcher at this point).
                    if self.sides[side].pokemon[active].hp == 0
                        && !self.sides[side].pokemon[active].fainted
                    {
                        // NATURAL CURE onSwitchOut fires on the 0-HP-but-NOT-YET-fainted
                        // switcher BEFORE faintMessages (`gen3_natural_cure_v1` /
                        // `gen3_omniscient_byte_fuzz_v1`): gen 2-4 Pursuit runs the switcher's
                        // SwitchOut event (hence naturalcure.onSwitchOut) as part of the
                        // interrupted switch, so an NC holder's tox/etc is CURED — its
                        // `onSwitchOut` guard is `if (!status || status==='fnt') return`, and
                        // `fnt` is only set below by process_faints, so status is still live
                        // here. The sim emits `|-curestatus|<mon>|<tok>|[from] ability: Natural
                        // Cure|[silent]` BEFORE the `|-hint|`/`|faint|` (byte-fuzz repro
                        // rmroh04is_ab_10_1 dec461: the port's later clearVolatile NC block is
                        // `!fainted`-gated, so process_faints skipped it — emit it here first).
                        // DRAW-FREE. Curing status now is state-neutral (the mon faints anyway).
                        {
                            let m = &mut self.sides[side].pokemon[active];
                            if to_id(&m.ability) == "naturalcure" {
                                if let Some(tok) = status_token(m.status) {
                                    m.status = None;
                                    if self.logging() {
                                        let mr = self.mon_ref(side, active, dex);
                                        self.log.curestatus_from_ability_silent(
                                            &mr,
                                            tok,
                                            "Natural Cure",
                                        );
                                    }
                                }
                            }
                        }
                        if self.logging() {
                            self.log.hint(
                                "Previously chosen switches continue in Gen 2-4 after a Pursuit target faints.",
                                false, // battle.ts:2791 — no `once` → fires on EVERY occurrence
                            );
                        }
                        self.process_faints(dex);
                    }
                }
                // (g) The `pursuit` volatile is CONSUMED by the interrupt (whether or not a strike
                //     ran — e.g. the pursuer had already switched/fainted). Clear it before the
                //     swap so the residual + turn-top don't double-handle it.
                self.sides[side].pokemon[active].pursuit = None;
            }
        }
        // --- BATON PASS copyVolatileFrom (`gen3_move_coverage_batch3_v1`): if this side has a
        //     pending Baton Pass, SNAPSHOT the OUTGOING mon's PASS-SET (its 7 boosts + the
        //     copyable `noCopy == false` volatiles the port models: substitute / leech_seed /
        //     confusion / curse) BEFORE the clearVolatile block below zeros them. The snapshot
        //     is applied to the entrant AFTER the array swap, and the `|switch|` line is tagged
        //     `[from] Baton Pass`. `is_drag` is false for a Baton Pass (it's a self-switch, not
        //     a phaze drag) — but we compute `bp` from the marker (a phaze can't set it). Major
        //     STATUS is NOT a volatile → not passed (stays with the outgoing mon). ---
        let bp = self.sides[side].baton_pass_pending && !is_drag;
        let bp_snapshot = if bp {
            let m = &self.sides[side].pokemon[active];
            // BATCH-6 additions to the pass-set (`gen3_move_coverage_batch6_v1`, the
            // resolved-dex noCopy facts — probe_batch6_dexfacts.js): **perishsong**
            // (noCopy false — the entrant inherits the counter), the trap-move
            // **trapped** volatile (noCopy FALSE, behaviorally probed T3b — the entrant
            // is STILL firm-trapped, the link re-points so the ORIGINAL trapper's later
            // exit frees it), and **charge** (noCopy undefined → falsy → copied).
            // Encore / Destiny Bond are noCopy TRUE → NOT passed (and DB was already
            // removed by the passer's own BP move attempt at onBeforeMove −1).
            //
            // BATON-PASS residual-handler tie fix (`gen3_batonpass_stall_pursuit_copy_v1`,
            // R21 — the bab_9_4 dec-2 hidden draw): `copyVolatileFrom` copies EVERY
            // non-`noCopy` volatile, so the passer's **`stall`** (from a prior Protect;
            // `protect_counter` + `stall_duration`, both noCopy FALSE) AND **`pursuit`**
            // (the beforeTurnMove-laid volatile, if the passer is being Pursuit-targeted
            // THIS turn — noCopy FALSE) both transfer to the entrant. Each registers a
            // NO_ORDER/subOrder-2 residual DURATION handler (`run_residuals` lines
            // `protect_counter > 0` / `pursuit.is_some()`), so on the entrant they TIE at
            // the entrant's cached speed → the end-of-turn residual handler-sort draws ONE
            // Fisher-Yates `random(0,2)` the port previously MISSED (SIM-verified vs this
            // repro's turn-2 residual: Metagross carried BOTH `stall` + `pursuit` handlers
            // at speed 190/subOrder 2). Appended LAST (never reorder the pass-set tuple).
            Some((
                m.boosts, m.substitute, m.leech_seed, m.confusion, m.curse, m.perish,
                m.trapped_by, m.charge, m.protect_counter, m.stall_duration, m.pursuit,
            ))
        } else {
            None
        };
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
        // The NATURAL CURE status token, captured inside the clearVolatile block (its
        // status is cleared there) and emitted as `-curestatus` AFTER the block but
        // BEFORE the array swap, so the line still refers to the OUTGOING mon at `active`.
        let mut nc_cure_token: Option<&'static str> = None;
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
            // The CURSE volatile clears on switch-out (`clearVolatile` sets `this.volatiles = {}`,
            // pokemon.js — curse is an ordinary volatile, dropped like leech/sub). MISSING this
            // left a cursed mon STILL cursed on the bench, so on re-entry the order-10/subOrder-8
            // residual re-chipped it `floor(maxhp/4)` per turn where the sim had cleared it (the
            // curse-cluster state divergences — repro rmrz8ngky_ab_40_3 / ab_0_14 / ab_26_0 /
            // ab_40_14, ~maxhp/4-low with the seed matching). The Baton-Pass pass-set snapshot
            // (above, pre-clearVolatile) already captured `m.curse` and re-applies it to the
            // entrant post-swap (noCopy false), so this clear does not disturb Baton Pass.
            m.curse = None;
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
            m.last_move_was_self_overwrite = false;
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
                // Capture the status TOKEN before clearing — the `-curestatus` line
                // (FORM 4, emitted after this block, pre-swap) needs it.
                nc_cure_token = status_token(m.status);
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
            // The FOCUS PUNCH + PURSUIT `duration: 1` volatiles clear on switch-out
            // (`clearVolatile`, `gen3_move_coverage_batch4_v1`). A switching FP user drops its
            // focuspunch; a switching pursued mon that reaches here (a non-intercepted path)
            // drops its pursuit. (A voluntary pursued switch-out was already consumed by the
            // interrupt above; this is the belt-and-braces reset.)
            m.focus_punch = None;
            m.pursuit = None;
            // The BEAT UP `beatup` `duration: 1` volatile ALSO clears on switch-out
            // (`clearVolatile`, `gen3_move_coverage_batch4b_v1`). MISSING this stranded a stale
            // `beat_up = true` on a mon phazed out the same turn it Beat Up'd (Roar priority -6
            // resolves after the move); the flag survived the bench (the turn-top `clear_flinch`
            // is ACTIVE-mon-only) and, when the mon re-entered, the active-only residual gather
            // pushed a SPURIOUS NO_ORDER/subOrder-2 VolatileDuration handler → at a residual
            // speed tie one extra `random(0,2)` tie-shuffle vs the sim (a silent draw-order
            // desync — the class this batch exists to remove). Sibling one-liners above.
            m.beat_up = false;
            // The MUSTRECHARGE + TWOTURNMOVE volatiles clear on switch-out (`clearVolatile`,
            // `gen3_move_coverage_batch4c_v1`): a locked mon can only LEAVE the field via a
            // phaze DRAG (its own request is trapped) or a Baton-Pass-free forced path — the
            // dragged-out mon re-enters fully unlocked (a fresh Hyper Beam / charge starts
            // over). Same stale-flag hazard class as `beat_up` above.
            m.must_recharge = false;
            m.two_turn = None;
            // The COUNTER / MIRROR COAT reactive volatile clears on switch-out
            // (`clearVolatile`, `gen3_move_coverage_batch5_v1`) — a phazed-out counter
            // user must not gather a stale residual duration handler on re-entry (the
            // `beat_up` stale-flag hazard class). NOT Baton-Passable (`noCopy`-class
            // per-turn state; the pass-set snapshot above never includes it).
            m.reactive = None;
            // --- BATCH-6 volatiles clear on switch-out (`clearVolatile`,
            //     `gen3_move_coverage_batch6_v1`): ENCORE (a switched-out encored mon
            //     returns un-locked — noCopy true, never passed), PERISH (the counter
            //     clears; an entrant while the song runs gets NOTHING — but the BP
            //     snapshot above PASSES it, noCopy false), ENDURE (duration-1,
            //     belt-and-braces beside the turn-top clear), CHARGE (cleared like any
            //     volatile — the BP snapshot passes it, noCopy falsy), DESTINY BOND (+
            //     its pending-KO record — noCopy true, and the window closed at the BP
            //     move attempt anyway), the trap-move TRAPPED link (a dragged-out /
            //     replaced holder is freed; the BP snapshot passes it, noCopy false),
            //     and the MIMIC moveslot overlay (the `baseMoveSlots` revert — the slot
            //     reverts to Mimic with Mimic's OWN remaining PP; probed MI-switch). ---
            m.encore = None;
            m.perish = None;
            m.endure = false;
            m.charge = false;
            // SNATCH (`gen3_snatch_v1`): the singleturn volatile clears on switch-out
            // (`clearVolatile`) — the `beat_up` stale-flag hazard class (a snatcher phazed
            // out the same turn it cast Snatch must not gather a stale residual duration
            // handler on re-entry). A snatch is +4, so a voluntary switch same-turn is
            // impossible; the drag-out path is the reachable one.
            m.snatch = false;
            m.destiny_bond = false;
            m.destiny_bond_ko_by = None;
            m.trapped_by = None;
            // YAWN (`gen3_yawn_v1`): the pending delayed-sleep volatile clears on switch-out
            // (`clearVolatile`) — a mon that pivots out before the Yawn resolves comes back with no
            // pending sleep (a fresh Yawn is needed); noCopy so it is never Baton-Passed.
            m.yawn = None;
            // The pending switch-in Intimidate target (`gen3_intimidate_forced_replacement_v1`,
            // M3) — a transient consumed at run_switch; clear it on switch-out too so a corpse /
            // outgoing mon never carries a stale foe uid.
            m.switchin_foe_uid = None;
            m.restore_mimic_overlay();
        }
        // [EMIT] NATURAL CURE `-curestatus` (`gen3_omniscient_byte_fuzz_v1` FORM 4):
        // `|-curestatus|<outgoing>|<status>|[from] ability: Natural Cure|[silent]`, emitted
        // BEFORE the switch/drag line (the sim's `naturalcure.onCheckShow`-less onSwitchOut
        // reveal, at `runEvent('SwitchOut')` before `clearVolatile` + the swap). `active`
        // still points to the OUTGOING mon here (the swap is below). Draw-free.
        if let Some(tok) = nc_cure_token {
            if self.logging() {
                let mon = self.mon_ref(side, active, dex);
                self.log.curestatus_from_ability_silent(&mon, tok, "Natural Cure");
            }
        }
        // --- TRAP-LINK source-left clear (`gen3_move_coverage_batch6_v1`, the linked
        //     `trapped`/`trapper` pair — probed T1/T4/T9 + T3b): the trap ENDS the
        //     moment the TRAPPER leaves the field ANY way (voluntary switch / Baton
        //     Pass / phaze drag; the FAINT path mirrors this in `process_faints`). When
        //     the DEPARTING mon's uid is the foe active's `trapped_by` link, remove the
        //     volatile — the freed mon's very next request drops `trapped:true`.
        //     DRAW-FREE (no protocol line — the `trapped` condition has no onEnd). The
        //     attract source-left clear below is the same mechanism class. ---
        {
            let departing_uid = self.sides[side].pokemon[self.sides[side].active].uid;
            let foe = 1 - side;
            let foe_active = self.sides[foe].active;
            if self.sides[foe].pokemon[foe_active].trapped_by == Some(departing_uid) {
                self.sides[foe].pokemon[foe_active].trapped_by = None;
            }
        }
        // ATTRACT source-left clear (`attract.onUpdate`, `gen3_ability_batch4_v1`): when the
        // DEPARTING mon is the SOURCE of the foe active's attraction, the volatile is removed +
        // `|-end|<mon>|Attract|[silent]` emitted (probe: Miltank pivots out → Zangoose's Attract
        // ends). DRAW-FREE. The sim fires this as the onUpdate AFTER the switch-in, so the `-end`
        // must FOLLOW the `|switch|` line (`gen3_attract_end_order_v1` — the random-mode byte-fuzz
        // Cute-Charm ORDER fix: the port emitted it BEFORE the switch). CAPTURED here (pre-swap,
        // while `side`'s active is still the departing source); the CLEAR + `-end` run AFTER the
        // `|switch|` emit below. (`side`'s switch does not change the FOE active, so the captured
        // foe_active stays valid.)
        let attract_clear_foe: Option<usize> = {
            let departing_uid = self.sides[side].pokemon[self.sides[side].active].uid;
            let foe = 1 - side;
            let foe_active = self.sides[foe].active;
            (self.sides[foe].pokemon[foe_active].attract == Some((side, departing_uid)))
                .then_some(foe_active)
        };
        // SWAP the team-array entries (the entrant → active position, outgoing →
        // the entrant's old bench position) + fix each mon's `position` field.
        self.sides[side].pokemon.swap(active, target);
        self.sides[side].pokemon[active].position = active;
        self.sides[side].pokemon[target].position = target;
        // Capture the foe active's uid at SWITCH-IN time (`gen3_intimidate_forced_replacement_v1`,
        // M3) — ONLY for a VOLUNTARY switch. The deferred `RunSwitch` fires this entrant's
        // switch-in Intimidate LATER (a separate runAction), by which point a mid-action foe
        // faint may have force-replaced the intended target. A VOLUNTARY switch's Intimidate
        // resolves INLINE in Showdown (during the switch action, before the foe's forced
        // replacement at the next `makeRequest`), so recording the foe present NOW lets
        // `event::intimidate_on_start` suppress a drop whose target FAINTED (the ab_1381_0
        // Pursuit→Destiny-Bond→replacement mis-target). A FORCED REPLACEMENT (is_voluntary=false,
        // e.g. a mutual-Self-Destruct DOUBLE replacement) instead switches BOTH sides in together
        // and fires abilities AFTER — its Intimidate DOES drop the co-replacement, so leave the
        // uid None (the pre-fix behavior, e2e-verified). A phaze DRAG likewise leaves it None.
        // Draw-free (a boost-suppression is state-only).
        if is_voluntary {
            self.sides[side].pokemon[active].switchin_foe_uid =
                Some(self.sides[1 - side].pokemon[self.sides[1 - side].active].uid);
        }
        // --- BATON PASS copyVolatileFrom APPLY (`gen3_move_coverage_batch3_v1`): the entrant
        //     is now at `active` (the clearVolatile block above already zeroed its fresh state).
        //     Apply the passer's snapshot: the boosts array + the four copyable volatile fields
        //     (substitute HP / leech-seed seeder / confusion counter / curse source). The
        //     leech/curse fields keep the seeder/curse-source SIDE, so the residual keeps
        //     chipping the new mon. Applied BEFORE `cached_speed` is (re)established, so a
        //     passed +Spe/-Spe boost is reflected in the entrant's post-switch cached speed
        //     (Showdown's copyVolatileFrom runs inside switchIn before the speed cache). Clear
        //     the pending marker. ---
        if let Some((
            boosts, sub, leech, conf, curse, perish, trapped_by, charge, protect_counter,
            stall_duration, pursuit,
        )) = bp_snapshot.clone()
        {
            let m = &mut self.sides[side].pokemon[active];
            m.boosts = boosts;
            m.substitute = sub;
            m.leech_seed = leech;
            m.confusion = conf;
            m.curse = curse;
            // The batch-6 noCopy-false volatiles (`gen3_move_coverage_batch6_v1`):
            // the perish counter, the trap-move `trapped` link (the entrant is STILL
            // firm-trapped by the SAME trapper — T3b), and the charge volatile.
            m.perish = perish;
            m.trapped_by = trapped_by;
            m.charge = charge;
            // The `stall` + `pursuit` noCopy-false volatiles (`gen3_batonpass_stall_pursuit_copy_v1`,
            // R21): the entrant inherits the stall counter (Protect success odds) + the pursuit
            // volatile, so BOTH register their NO_ORDER/subOrder-2 residual duration handlers on
            // the entrant → the tie-shuffle the sim draws (the bab_9_4 dec-2 hidden draw).
            m.protect_counter = protect_counter;
            m.stall_duration = stall_duration;
            m.pursuit = pursuit;
        }
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
        // there. Clear the side's switch flag — it's been answered. Also clear the
        // Baton Pass marker (the copyVolatileFrom has been applied above).
        self.sides[side].switch_flag = false;
        self.sides[side].baton_pass_pending = false;

        // [EMIT] `|switch|<entrant>|<Details>|<HP>` (a voluntary/forced-replacement
        // switch) OR `|drag|…` (a Roar/Whirlwind phaze). A BATON PASS switch-in carries
        // `[from] Baton Pass` (`gen3_move_coverage_batch3_v1`). Emitted AFTER the array
        // swap, so `active` is the entrant — its ident/details/HP are the fresh entrant's.
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
            } else if bp {
                self.log.switch_from(&entrant, &details, &hp, "Baton Pass");
            } else {
                self.log.switch(&entrant, &details, &hp);
            }
        }

        // [EMIT — AFTER the |switch| line] the captured ATTRACT source-left clear + `-end`
        // (`gen3_attract_end_order_v1`): the sim's `attract.onUpdate` runs AFTER the switch-in, so
        // the `|-end|<foe>|Attract|[silent]` must FOLLOW the `|switch|` line (probe/byte-fuzz).
        if let Some(foe_active) = attract_clear_foe {
            let foe = 1 - side;
            self.sides[foe].pokemon[foe_active].attract = None;
            if self.logging() {
                let mon_ref = self.mon_ref(foe, foe_active, dex);
                self.log.volatile_end_silent(&mon_ref, "Attract");
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
    pub(crate) fn insert_runswitch(&mut self, side: usize, queue: &mut Vec<QAction>, dex: &Dex) {
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
    pub(crate) fn run_switch(&mut self, side: usize, dex: &Dex) -> bool {
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
        // The OTHER modeled SwitchIn handler is `slp.onSwitchIn` (`gen3_move_coverage_
        // batch5_v1`, the Sleep Talk `skippedTime` restore — live-probed: time=3 →
        // talk,talk → time=1,skipped=2 → switch out+in → time=3 again): `time +=
        // skippedTime; skippedTime = 0`. Same runSwitch site as the tox reset, so the
        // SAME cancellation law applies (a CANCELLED runSwitch keeps both). DRAW-FREE.
        if let Some(Status::Sleep(t)) = self.sides[side].pokemon[slot].status {
            let skipped = self.sides[side].pokemon[slot].sleep_skipped;
            if skipped > 0 {
                self.sides[side].pokemon[slot].status =
                    Some(Status::Sleep(t.saturating_add(skipped)));
                self.sides[side].pokemon[slot].sleep_skipped = 0;
            }
        }
        // (2b) runEvent('SwitchIn') ALSO fires `whiteherb.onAnySwitchIn` (priority −2, items.js:7681)
        // for EVERY active White Herb holder — restoring a holder that ALREADY carries a negative
        // boost + consuming the herb, at the SwitchIn step BEFORE the entrant's ability `Start`
        // (mods/gen4/scripts.js:41-45). Draw-free; `white_herb_restore` is a no-op unless the holder
        // has a pending negative stage, so the entrant (fresh, boosts cleared) never restores. The
        // BEFORE-Start timing is exactly why an ENTRANT'S OWN Intimidate does NOT restore the foe at
        // THIS switch-in (the drop is at Start, AFTER SwitchIn → the foe restores later via its
        // onAnyAfterMove — the ab_4_6 case, unchanged). The PRE-EXISTING-drop case is the fix (repro
        // rmrz81mki_ab_29_5: Nuzleaf's construction-Intimidate −1 Atk, un-restored because Nuzleaf
        // switched in FASTER than the drop, is restored when Quilava switches in mid-turn-1 — BEFORE
        // Nuzleaf's move, not at the fallback order-29 residual). Fires before the entrant's hp check
        // (the sim's SwitchIn precedes `if (!pokemon.hp) return`), so a spikes-KO'd entrant still
        // lets the foe's herb restore.
        self.white_herb_restore(1 - side, self.sides[1 - side].active, dex);
        self.white_herb_restore(side, slot, dex);
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
        // WHITE HERB is NOT restored here: the resolved gen3 `whiteherb.onAnySwitchIn` fires at
        // the `runEvent('SwitchIn')` step, which precedes the entrant's ability `Start`
        // (`singleEvent('Start', …)` — Intimidate), so at the SwitchIn event the opposing active
        // has NO negative boost yet (Intimidate hasn't dropped it). The restore therefore fires
        // LATER via `onAnyAfterMove` — after the holder's next move — NOT at this switch-in
        // (`gen3_white_herb_v1`; the omniscient byte-fuzz find ab_4_6, master-seed 80808: the sim
        // has Weepinbell MOVE then White Herb fires, where the port fired it at the switch-in).
        // The general `onAnyAfterMove` hook in `driver.rs` owns the post-move restore.
        // The switch-in Intimidate (STATE via `intimidate_on_start`, EMISSION via
        // `emit_ability_start_lines`) has now BOTH resolved — clear the captured foe uid so a
        // later switch of a DIFFERENT mon into this slot can't read a stale target
        // (`gen3_intimidate_forced_replacement_v1`, M3). Draw-free.
        self.sides[side].pokemon[slot].switchin_foe_uid = None;
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
        let dmg = self.focus_band_damage(side, slot, dmg, false, false, dex);
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
