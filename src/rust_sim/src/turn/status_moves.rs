use crate::dex::{to_id, Dex, Type};
use crate::protocol::Cause;
use crate::state::Status;
use super::*;
use super::helpers::*;

impl crate::state::BattleState {

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
    pub(crate) fn run_status_move(
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
        // `queue.willAct()` (the Protect gate) — threaded so a Sleep-Talk-CALLED Protect
        // runs the same `run_protect` machinery (`gen3_move_coverage_batch5_v1`).
        will_act: bool,
        // The PRE-move Choice-lock snapshot (the Sleep Talk `onTryHit` choicelock gate —
        // see `run_move`; the lock THIS use just set does not count).
        was_choice_locked: bool,
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
        // --- CURSE (`gen3_move_coverage_batch3_v1`) — the gen-3 `curse.onModifyMove`
        //     RE-TARGETS the move at runtime based on the USER's type:
        //       * NON-GHOST user → `move.self = {boosts:{atk:1,def:1,spe:-1}}`, `move.target =
        //         nonGhostTarget` (SELF). So the announce renders the USER (`|move|<user>|
        //         Curse|<user>`) and the effect is a self-boost — DELETING the volatileStatus
        //         + onHit (no curse laid, no HP cost).
        //       * GHOST user with the foe SUBSTITUTED → `onModifyMove` deletes both → the move
        //         does NOTHING (`[still]` + `-fail`, no HP cost). The base `move.target` is
        //         `normal` (foe), so the un-redirected announce renders the FOE.
        //       * GHOST user, non-subbed → lays the `curse` volatile on the FOE + pays
        //         floor(maxhp/2) HP (`|move|<user>|Curse|<foe>`, foe-target).
        //     `curse_ghost` = the user is a Ghost; `curse_still` = the ghost-into-a-sub
        //     did-nothing case (announce as `[still]`, like Spikes-at-cap). The full effect
        //     branch runs in the dedicated arm below; this block only fixes the ANNOUNCE.
        let curse_ghost = move_id == "curse"
            && mon_types(&self.sides[_side].pokemon[_slot], dex).contains(&Type::Ghost);
        let curse_still = move_id == "curse"
            && curse_ghost
            && self.sides[foe].pokemon[foe_slot].substitute.is_some();
        // --- YAWN disposition (`gen3_yawn_v1`) — the TryHit-order resolution of a Yawn cast,
        //     computed ONCE (draw-free reads) so the announce `[still]` form + the yawn arm below
        //     agree. The sim's gen-3 `yawn.onTryHit(target)` fails if `target.status ||
        //     !target.runStatusImmunity('slp')`; a Protect (`protect: 1`) blocks it at TryHit
        //     BEFORE that; a Substitute (no `bypasssub`) blocks it at `onTryPrimaryHit` AFTER.
        //     TryHit priority order: Protect > already-statused > sleep-immune > substitute > add.
        //     `yawn_protect`/`yawn_statused`/`yawn_immune`/`yawn_subbed` are mutually exclusive by
        //     construction (each gates on the prior being false); `yawn_still` = the two cases whose
        //     announce is the `[still]` did-nothing form (already-statused OR substituted; Protect
        //     + sleep-immune keep the normal target announce). VERIFIED bit-for-bit vs the
        //     omniscient sim (`harness/probe_batch89_haze_trick_yawn.js` + `probe_yawn_edges`). ---
        let yawn_protect = move_id == "yawn" && self.protect_blocks(foe, foe_slot, false);
        let yawn_statused = move_id == "yawn"
            && !yawn_protect
            && self.sides[foe].pokemon[foe_slot].status.is_some();
        let yawn_immune = move_id == "yawn"
            && !yawn_protect
            && !yawn_statused
            && dex
                .ability(&to_id(&self.sides[foe].pokemon[foe_slot].ability))
                .and_then(|a| a.status_immune.as_ref())
                .map(|si| si.blocks("slp"))
                .unwrap_or(false);
        let yawn_subbed = move_id == "yawn"
            && !yawn_protect
            && !yawn_statused
            && !yawn_immune
            && self.sides[foe].pokemon[foe_slot].substitute.is_some();
        let yawn_still = yawn_statused || yawn_subbed;
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
            // A ghost-Curse-into-a-sub is the SAME did-nothing `[still]` case; a Yawn cast into
            // an already-statused OR substituted (non-protected, non-immune) foe is likewise
            // (`yawn_still`, `gen3_yawn_v1`).
            let still_form =
                (move_id == "spikes" && self.sides[foe].spikes >= 3) || curse_still || yawn_still;
            if still_form {
                self.log.move_used(&user, move_name, None, false, true);
            } else {
                // The `|move|` TARGET field renders the USER for any NON-foe-directed move
                // (`self`/`allySide`/`all`/`allyTeam` — Light Screen / Reflect / Sunny Day /
                // Rain Dance / Perish Song / Heal Bell), NOT just `target:self`
                // (`gen3_omniscient_byte_fuzz_v1`, byte-fuzzer-surfaced: the port hard-rendered
                // the FOE for these). A `foeSide` move (Spikes) + a `normal` foe move render the
                // FOE active. Curse's NON-GHOST branch re-targets to SELF (announce the USER);
                // its GHOST branch keeps the base foe target.
                let tgt = dex.moves(move_id).map(|m| m.target.clone()).unwrap_or_default();
                let renders_self = status_move_announce_renders_user(&tgt)
                    || (move_id == "curse" && !curse_ghost);
                let target = if renders_self {
                    user.clone()
                } else {
                    self.mon_ref(foe, foe_slot, dex)
                };
                self.log.move_used(&user, move_name, Some(&target), false, false);
            }
        }
        // --- SNATCH INTERCEPTION (`gen3_snatch_v1`, the `snatch` condition's
        //     `onAnyPrepareHit`, `onAnyPrepareHitPriority = -1`) — probe-settled bit-for-bit
        //     vs the omniscient sim (`harness/probe_snatch.js`). When the FOE (`_side`,
        //     the current move's USER) uses a `flags.snatch` SELF-targeted status move
        //     while the OTHER active (`foe`, the snatcher who cast Snatch this turn) has the
        //     `snatch` volatile up, the snatcher STEALS it: the snatcher executes the move
        //     itself and the FOE's move does nothing. The interception fires INSIDE the
        //     foe's `tryMoveHit`, AFTER the foe's `|move|` line (emitted just above) and
        //     AFTER the foe's PP was deducted (in `run_move` before this dispatch). The
        //     exact sim ordering (settled by the probe):
        //       (1) `removeVolatile('snatch')` on the SNATCHER — FIRST, so the snatcher's
        //           own nested `useMove` below can't re-trigger the interception;
        //       (2) `|-activate|SNATCHER|move: Snatch|[of] FOE`;
        //       (3) `runEvent('DeductPP', source=VICTIM, snatchUser=SNATCHER, Snatch)` —
        //           DRAW-FREE. The EVENT-TARGET is `source` = the FOE whose self-target
        //           move was stolen (the VICTIM, `_side`/`_slot`). If that victim has
        //           **Pressure**, its `onDeductPP(target, source){ if(target===source)
        //           return; return 1; }` returns 1 → the SNATCHER's Snatch loses an
        //           EXTRA 1 PP (`snatchUser.deductPP('snatch', 1)`). So a Snatch steal
        //           costs the snatcher 1 Snatch PP (the cast) + 1 more iff the stolen
        //           move's user has Pressure (`bab_4_16`: a Blissey snatching a Pressure
        //           Zapdos's Rest → the sim deducts 2 Snatch PP total → `pp:14`; the
        //           port used to model (3) as a pure no-op → under-deducted → `pp:15`);
        //       (4) `this.actions.useMove(stolenId, snatcher)` — the snatcher executes the
        //           stolen move in ITS OWN context (a bare useMove: no accuracy/on_before_move/
        //           PP/lastMove, just the effect + the stolen move's NATIVE draws — SwordsDance/
        //           Recover/Substitute draw 0 extra, Rest draws its sleep `random(2,6)`). Its
        //           announce carries the `|[from] Snatch` fold (set below);
        //       (5) `return null` → the FOE's move aborts (does nothing).
        //     SNATCH INTRODUCES ZERO DRAWS OF ITS OWN (cast + steal are draw-free); the ONLY
        //     snatch-attributable draw is the residual duration-handler tie-shuffle a MIRROR
        //     draws (gathered in `run_residuals`, above). Priority +4 guarantees the volatile
        //     is up before ANY foe move even for a SLOW snatcher (SN3 == SN2, seed-identical).
        if dex.moves(move_id).map(|m| m.is_snatchable).unwrap_or(false)
            && self.sides[foe].pokemon[foe_slot].snatch
            && !self.sides[foe].pokemon[foe_slot].fainted
            && self.sides[foe].pokemon[foe_slot].hp > 0
        {
            // (1) removeVolatile('snatch') on the SNATCHER first.
            self.sides[foe].pokemon[foe_slot].snatch = false;
            // (2) `|-activate|SNATCHER|move: Snatch|[of] FOE`.
            if self.logging() {
                let snatcher = self.mon_ref(foe, foe_slot, dex);
                let victim = self.mon_ref(_side, _slot, dex);
                self.log
                    .activate(&snatcher, "move: Snatch", Some(&format!("[of] {victim}")));
            }
            // (3) DeductPP — draw-free. The event-target is the VICTIM (`_side`/`_slot`,
            //     the stolen move's user); if it has Pressure, deduct 1 EXTRA Snatch PP
            //     from the SNATCHER (`foe`/`foe_slot`) — mirroring `runEvent("DeductPP",
            //     victim, snatcher, Snatch)` returning 1 for a Pressure source
            //     (`gen3_snatch_pressure_pp_v1`, the `bab_4_16` per-side request find).
            if crate::dex::to_id(&self.sides[_side].pokemon[_slot].ability) == "pressure" {
                if let Some(snatch_slot) = self.sides[foe].pokemon[foe_slot]
                    .set
                    .moves
                    .iter()
                    .position(|m| crate::dex::to_id(m) == "snatch")
                {
                    self.sides[foe].pokemon[foe_slot].deduct_pp(snatch_slot, 1);
                }
            }
            // (4) the SNATCHER uses the stolen move (the bare `useMove` — a recursive
            //     `run_status_move` in the snatcher's context, with the `[from] Snatch`
            //     announce fold). The stolen self-target moves are all category Status, so
            //     they route through THIS fn; the recursion re-checks the interception but
            //     the snatcher's volatile is now cleared (and the victim has none), so it
            //     never re-fires. Its resolution is DISCARDED — the nested useMove is not the
            //     snatcher's own queued action, so it fires no trailing Update (probe: SN2
            //     SwordsDance draws only endTurn; SN5 Rest draws its sleep roll + endTurn).
            self.log.set_next_move_from("Snatch");
            self.run_status_move(
                foe, foe_slot, _side, _slot, accuracy, never_miss, move_type, move_id,
                move_name, targets_self, status_inflicted, /*foe_will_move*/ false,
                /*will_act*/ will_act, /*was_choice_locked*/ false, dex,
            );
            // (5) return null — the FOE's move does nothing (not landed, not missed).
            return MoveResolution::done(false, false, false);
        }
        // --- SLEEP TALK (`gen3_move_coverage_batch5_v1`, the gen-3 `sleeptalk` — probe
        //     `harness/probe_batch5_sleeptalk.js`, the resolved source dumped there):
        //     usable ONLY while asleep; picks ONE of the user's OTHER eligible moves at
        //     RANDOM (a REAL `sample` draw — even for a 1-move pool) and executes it via
        //     a bare `useMove`. The user reached here THROUGH the slp onBeforeMove
        //     (`sleepUsable` — the `|cant|slp` printed, the counter decremented,
        //     `skippedTime`++), and Sleep Talk's OWN PP was deducted (every path below
        //     keeps that: PP −1 on ALL fail paths too — probed). ---
        if move_id == "sleeptalk" {
            // onTry(source){ return source.status === 'slp' } — an AWAKE use (incl. the
            // WAKE turn, whose cure fired in on_before_move before this ran) fails
            // SILENTLY: the normal self-target `|move|` announce (already emitted), NO
            // `[still]`, NO `-fail`, ZERO draws (probed — identical protocol to a
            // never-asleep use).
            if !matches!(self.sides[_side].pokemon[_slot].status, Some(Status::Sleep(_))) {
                return MoveResolution::done(false, false, false);
            }
            // onTryHit(pokemon){ return !volatiles['choicelock'] && !volatiles['encore'] }
            // — a Choice lock from a PREVIOUS turn OR the `encore` volatile fails Sleep
            // Talk: the `[still]` retro-edit + `|-fail|<user>`, NO sample draw.
            //   * CHOICE LOCK: the lock is always Sleep Talk ITSELF (choicelock records
            //     the CHOSEN move, so CB + Sleep Talk works exactly ONCE then fails every
            //     later turn of the lock — probed); the lock THIS use just set does NOT
            //     count (`was_choice_locked` is the pre-move snapshot).
            //   * ENCORE (`gen3_encore_sleeptalk_trylhit_v1`, the R13 byte-fuzz fix — since
            //     batch 6 MODELS Encore, this gate is now LIVE): a mon carrying the `encore`
            //     volatile fails Sleep Talk DRAW-FREE. The encore is checked at its CURRENT
            //     value (NOT a pre-move snapshot) because the foe's Encore that set it can
            //     land THIS turn BEFORE the sleeper's Sleep Talk (a faster foe Encores an
            //     asleep mon → its `move_usable`/`onOverrideAction` redirects the queued move
            //     to the encored Sleep Talk slot → THIS gate fails it draw-free, matching the
            //     sim's onTryHit which reads the live `volatiles['encore']`). So an
            //     Encored-into-Sleep-Talk mon can NEVER actually resolve Sleep Talk while the
            //     encore holds (probe: `sleeptalk onTryHit -> false` when `volatiles['encore']`
            //     is present). VERIFIED vs the sim on the R13 repro (ab_15_15 dec 47): the
            //     port used to SAMPLE + run the picked move (+4 draws) where the sim fails it.
            let encored = self.sides[_side].pokemon[_slot].encore.is_some();
            if was_choice_locked || encored {
                if self.logging() {
                    self.log.attr_last_move_still();
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.fail(&user, None, false);
                }
                return MoveResolution::done(false, false, false);
            }
            // onHit — build the pool from the user's moveSlots IN SLOT ORDER, keeping
            // `{move, pp}` where `!flags.nosleeptalk && !flags.charge` (data-driven:
            // `MoveData::{no_sleep_talk, is_charge}`; Sleep Talk excludes ITSELF via its
            // own nosleeptalk flag). NO pp filter (a 0-PP member stays IN the pool) and
            // NO disabled/Taunt filter (the resolved source has none — a Disabled pool
            // member is still pickable).
            let mut pool: Vec<(usize, String, u16)> = Vec::new();
            for k in 0..self.sides[_side].pokemon[_slot].set.moves.len() {
                if let Some(m) = self.move_at(_side, _slot, k, dex) {
                    if !m.no_sleep_talk && !m.is_charge {
                        let pp = self.sides[_side].pokemon[_slot].move_pp.get(k).copied().unwrap_or(0);
                        pool.push((k, m.id.clone(), pp));
                    }
                }
            }
            if pool.is_empty() {
                // EMPTY pool → onHit returns false → `|move|…|Sleep Talk||[still]` +
                // `|-fail|<user>`, ZERO Sleep-Talk draws (probed: only the QC drew).
                if self.logging() {
                    self.log.attr_last_move_still();
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.fail(&user, None, false);
                }
                return MoveResolution::done(false, false, false);
            }
            // THE SAMPLE: `this.sample(pool)` = `random(n)` — drawn EVEN at n == 1
            // (`random(1)` returns 0 but still consumes a draw — probed). `random(n)=k`
            // → the k-th eligible move in slot order (probed mapping).
            let pick = self.prng.random_below(pool.len() as u32) as usize;
            let (k, picked_id, picked_pp) = pool[pick].clone();
            if picked_pp == 0 {
                // The picked slot has 0 PP → `|cant|<user>|nopp|<raw move id>` and STOP
                // (the turn is wasted; NO further draws — probed).
                if self.logging() {
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.cant(&user, "nopp", Some(&picked_id));
                }
                return MoveResolution::done(false, false, false);
            }
            // EXECUTE the picked move — the sim's bare `actions.useMove(picked)`: the
            // recursive `run_move` under the `sleep_talk_call` transient (skips
            // on_before_move / PP / lastMove; the picked move's PP is NEVER consumed;
            // the FULL normal draw chain otherwise — acc/crit/damage/secondary for a
            // damaging pick, the status arms for a status pick — probed: Body Slam ran
            // acc+crit+dmg+secondary; Curse rolled its selfDrops random(100); an asleep
            // Rest pick silently no-ops in `run_rest`). The announce carries the
            // byte-exact `|[from] Sleep Talk` fold. The called resolution PROPAGATES
            // (a landed pick fires the caller's in-tryMoveHit Update; a called Roar's
            // drag rides `force_switch_foe`).
            self.log.set_next_move_from("Sleep Talk");
            self.sleep_talk_call = true;
            return self.run_move(
                MoveAction { side: _side, slot: _slot, move_index: k, struggle: false },
                will_act,
                foe_will_move,
                dex,
            );
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
                    // [EMIT] a genuine miss (acc<100 via Bright Powder / Lax Incense / a
                    // defender evasion boost — never at the flat 100, but reachable): the
                    // announce gains the `[miss]` attr + `|-miss|`, matching the sim
                    // (`gen3_status_move_miss_emit_v1` — these arms previously returned
                    // draw-free but emitted NOTHING, so a missed Encore/Roar/setup showed
                    // `|move|…` without `[miss]`).
                    if self.logging() {
                        let user = self.mon_ref(_side, _slot, dex);
                        let target = self.mon_ref(foe, foe_slot, dex);
                        self.log.attr_last_move_miss();
                        self.log.miss(&user, Some(&target));
                    }
                    return MoveResolution::done(true, false, false);
                }
            }
            // APPLY the self-boost on the USER (`_side`/`_slot` is the caster). DRAW-FREE,
            // ±6 clamp, no immunity gate (self-boosts are never blocked). Emit `|-boost|` per
            // stat with the CLAMPED-applied delta — and, unlike a secondary boost, a PRIMARY
            // self-boost MOVE STILL emits the line at a 0-delta at the +6 cap (Agility@+6 →
            // `|-boost|…|spe|0`; the sim's `boost()` `!isSecondary && !isSelf` branch —
            // `gen3_omniscient_byte_fuzz_v1` FORM 10). Route through `boost_applied` (emits 0),
            // NOT the zero-skipping `boost`.
            for &(idx, stages) in &self_boosts {
                let cur = self.sides[_side].pokemon[_slot].boosts[idx] as i32;
                let next = (cur + stages as i32).clamp(-6, 6);
                self.sides[_side].pokemon[_slot].boosts[idx] = next as i8;
                if self.logging() {
                    let delta = (next - cur) as i8; // the applied (post-clamp) magnitude
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.boost_applied(&user, idx, stages, delta, next as i8);
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
            // full-HP / heal-0 FAIL emission differs by MOVE FAMILY (the `bab_7_1`
            // per-side byte-fuzz find, `gen3_omniscient_byte_fuzz_v1`):
            //   - the DECLARATIVE `move.heal:[1,2]` family (Recover / Soft-Boiled /
            //     Slack Off / Milk Drink): at full HP `this.heal` returns false → the
            //     move FAILS → `attrLastMove('[still]')` + `-fail|<user>|heal` (the sim
            //     path — corpus fixture 04, e.g. `|move|…|Recover||[still]`,`|-fail|…|heal`).
            //   - the ONHIT weather-heal family (Morning Sun / Moonlight / Synthesis):
            //     `heal:undefined` + an `onHit` fn that calls `this.heal(...)`; at full HP
            //     that `this.heal` returns 0 but the `onHit` returns UNDEFINED (truthy),
            //     so the move SUCCEEDS — the sim renders the NORMAL self-target announce
            //     (already emitted at the top of this fn) and simply OMITS the `-heal`
            //     line (never `[still]`/`-fail`). VERIFIED vs the sim getPlayerStreams:
            //     sim `|move|p2a: Moltres|Morning Sun|p2a: Moltres` (self-target success)
            //     vs the old port `|move|…|Morning Sun||[still]` (the did-nothing form).
            let onhit_weather_heal =
                matches!(move_id, "morningsun" | "moonlight" | "synthesis");
            if self.logging() {
                let user = self.mon_ref(_side, _slot, dex);
                if healed {
                    let hp = self.hp_status(_side, _slot);
                    self.log.heal(&user, &hp, None);
                } else if !onhit_weather_heal {
                    // The declarative Recover family: the `[still]`+`-fail|heal` gate.
                    self.log.attr_last_move_still();
                    self.log.fail(&user, Some("heal"), false);
                }
                // else: the onHit weather-heal family at heal-0 emits NOTHING extra —
                // the plain self-target announce already stands (the move SUCCEEDED).
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

        // ═══════════════════ MOVE-COVERAGE BATCH 6 (`gen3_move_coverage_batch6_v1`) ═══════════════════
        // The final UNMODELED tail — probe-settled bit-for-bit (`harness/probe_batch6_{locks,
        // field_trap,utility}.js` + `probe_batch6_dexfacts.js`). Every arm below is id-gated;
        // the announce already emitted at the top of this fn.

        // --- ENCORE — lock the foe into its LAST-USED move for random(3,7) turns. The
        //     probed draw model (EN1-EN10): [1] accuracy `randomChance(100,100)` (acc 100,
        //     NOT never-miss — DRAWN, always passes), [2] the `durationCallback`
        //     `random(3,7)` (rolled 3..6) INSIDE addVolatile — nothing else (NO
        //     in-tryMoveHit Update: `landed` FALSE, EN8). The FAIL split is NOT uniform:
        //       * ALREADY-ENCORED (addVolatile false BEFORE the durationCallback): the
        //         accuracy draw ONLY, `[still]` + `-fail|<user>`, the existing volatile
        //         UNCHANGED (EN6);
        //       * no-lastMove / lastMove-has-failencore (Struggle/Mimic/…) / lastMove at
        //         0 PP: accuracy AND durationCallback BOTH drawn (the durationCallback
        //         fires before onStart rejects), then `[still]` + `-fail|<user>` (EN3/
        //         EN4/EN10);
        //       * SUCCESS: `stored = willMove(target) ? rolled : rolled + 1` (the exact
        //         Disable branch — a FASTER encore user stores `rolled`, a SLOWER one
        //         `rolled + 1`; EN1/EN2 share boundary seeds with DIFFERENT stored
        //         durations), `|-start|<foe>|Encore`.
        //     Flags: `protect: 1` (a Protect blocks AFTER the accuracy draw — the generic
        //     TryHit position, before addVolatile → NO durationCallback) + `bypasssub: 1`
        //     (a Substitute does NOT block). noCopy → NOT Baton-Passable. The lock's
        //     selection restriction lives in `move_usable`; the execution OVERRIDE
        //     (`onOverrideAction`) in `turn_loop`; the residual tick (order 10/subOrder
        //     14, incl. the 0-PP early end) in `run_residuals`. ---
        if move_id == "encore" {
            debug_assert!(
                !never_miss && accuracy == 100,
                "encore expected gen-3 accuracy 100 + not never_miss, got \
                 accuracy={accuracy} never_miss={never_miss}"
            );
            // (1) ACCURACY — drawn (always passes at 100).
            if !never_miss {
                let acc_hit = self.roll_accuracy(_side, _slot, foe, foe_slot, accuracy, never_miss, move_type, dex);
                if !acc_hit {
                    // [EMIT] a genuine miss (acc<100 via Bright Powder / Lax Incense / a
                    // defender evasion boost — never at the flat 100, but reachable): the
                    // announce gains the `[miss]` attr + `|-miss|`, matching the sim
                    // (`gen3_status_move_miss_emit_v1` — these arms previously returned
                    // draw-free but emitted NOTHING, so a missed Encore/Roar/setup showed
                    // `|move|…` without `[miss]`).
                    if self.logging() {
                        let user = self.mon_ref(_side, _slot, dex);
                        let target = self.mon_ref(foe, foe_slot, dex);
                        self.log.attr_last_move_miss();
                        self.log.miss(&user, Some(&target));
                    }
                    return MoveResolution::done(true, false, false);
                }
            }
            // (2) PROTECT BLOCK (flags.protect — the generic TryHit position, BEFORE
            //     addVolatile → no durationCallback draw). bypasssub → NO substitute check.
            if self.protect_blocks(foe, foe_slot, false) {
                if self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.activate(&target, "Protect", None);
                }
                return MoveResolution::done(false, false, false);
            }
            // (3) ALREADY-ENCORED — addVolatile returns false BEFORE the durationCallback:
            //     the accuracy draw was the ONLY draw; `[still]` + `-fail|<user>`; the
            //     existing volatile continues UNCHANGED (EN6).
            if self.sides[foe].pokemon[foe_slot].encore.is_some() {
                if self.logging() {
                    self.log.attr_last_move_still();
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.fail(&user, None, false);
                }
                return MoveResolution::done(false, false, false);
            }
            // (4) durationCallback — `random(3,7)` (3..6), drawn BEFORE onStart can reject.
            let rolled = self.prng.random_range(3, 7) as u8;
            // (5) onStart rejects: no lastMove / lastMove carries `failencore` (data-driven
            //     `MoveData::fail_encore` — Struggle stores `last_move = None` so it falls
            //     under no-lastMove here, matching the sim's failencore-flagged Struggle) /
            //     the lastMove slot at 0 PP. All → `[still]` + `-fail|<user>`, draws consumed.
            let reject = match self.sides[foe].pokemon[foe_slot].last_move {
                None => true,
                Some(lslot) => {
                    let fail_flag = self
                        .move_at(foe, foe_slot, lslot, dex)
                        .map(|m| m.fail_encore)
                        .unwrap_or(true);
                    fail_flag
                        || self.sides[foe].pokemon[foe_slot].move_pp.get(lslot).copied().unwrap_or(0) == 0
                }
            };
            if reject {
                if self.logging() {
                    self.log.attr_last_move_still();
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.fail(&user, None, false);
                }
                return MoveResolution::done(false, false, false);
            }
            // (6) SUCCESS — store the branch-adjusted duration + the locked slot.
            let lslot = self.sides[foe].pokemon[foe_slot].last_move.expect("checked above");
            let stored = if foe_will_move { rolled } else { rolled + 1 };
            self.sides[foe].pokemon[foe_slot].encore = Some((lslot, stored));
            // [EMIT] `|-start|<foe>|Encore` (the volatile's onStart).
            if self.logging() {
                let target = self.mon_ref(foe, foe_slot, dex);
                self.log.volatile_start(&target, "Encore");
            }
            return MoveResolution::done(false, false, false);
        }

        // --- DESTINY BOND — a self-target, never-miss Ghost Status move; the cast is
        //     ZERO draws (DB7: cast-turn count == a splash control; the duration-less
        //     volatile registers NO residual handler → no residual tie either). A
        //     re-cast on consecutive turns SUCCEEDS draw-free (the move's onPrepareHit
        //     removes the old volatile, then the volatileStatus re-adds — no fail line,
        //     PP −1 each cast, DB6). The volatile persists until the user's NEXT MOVE
        //     ATTEMPT (removed at onBeforeMove −1 / onMoveAborted — see run_move); the
        //     mutual-faint chain (a FOE-Move KO while up → the killer faints too,
        //     DRAW-FREE) lives in the damage sites + `process_faints`. `bypasssub` +
        //     self-target → no block of any kind. noCopy → NOT Baton-Passable. ---
        if move_id == "destinybond" {
            self.sides[_side].pokemon[_slot].destiny_bond = true;
            // [EMIT] `|-singlemove|<user>|Destiny Bond` (the volatile's onStart — fires
            // on the re-cast too, after the silent onPrepareHit removal).
            if self.logging() {
                let user = self.mon_ref(_side, _slot, dex);
                self.log.singlemove(&user, "Destiny Bond");
            }
            return MoveResolution::done(false, false, false);
        }

        // --- PERISH SONG — a field-wide (`target: all`) never-miss sound move; the
        //     cast is COMPLETELY DRAW-FREE in every branch (P1-P5: cast / all-counted
        //     re-cast / immune re-cast all match the splash-control seed trajectory).
        //     `onHitField` loops getAllActive() in SIDE order (p1 active then p2,
        //     regardless of caster): per mon — Invulnerability (unreachable: fly/dig
        //     unmodeled) → TryHit (SOUNDPROOF blocks: `|-immune|<mon>|[from] ability:
        //     Soundproof` — the holder is immune, everyone else INCLUDING THE CASTER is
        //     still counted) → if no perish counter yet: apply (duration 4) +
        //     `|-start|<mon>|perish3|[silent]`. ONE `|-fieldactivate|move: Perish Song`
        //     iff >= 1 NEWLY applied. NO new application anywhere: every active already
        //     counted → `[still]` + `-fail|<user>`; but >= 1 Soundproof-immune →
        //     SILENT success (bare `|move|` + the `-immune`s, no fail, no
        //     fieldactivate). The counter tick / faint is the order-12 residual; a
        //     switch-out clears it (ordinary volatile); Baton Pass PASSES it (noCopy
        //     false — the resolved gen3 condition). ---
        if move_id == "perishsong" {
            let mut newly = false;
            let mut any_result = false; // `result` in the sim: immune OR newly applied
            for s in 0..2 {
                let a = self.sides[s].active;
                if self.sides[s].pokemon[a].fainted {
                    continue; // getAllActive excludes a fainted mon
                }
                let blocks_sound = dex
                    .ability(&to_id(&self.sides[s].pokemon[a].ability))
                    .map(|ab| ab.blocks_sound)
                    .unwrap_or(false);
                if blocks_sound {
                    // TryHit null — Soundproof. Counted as a `result` (the silent-success
                    // gate) but NOT a new application.
                    any_result = true;
                    if self.logging() {
                        let m = self.mon_ref(s, a, dex);
                        self.log.immune_from_ability(&m, "Soundproof");
                    }
                    continue;
                }
                if self.sides[s].pokemon[a].perish.is_none() {
                    self.sides[s].pokemon[a].perish = Some(4);
                    any_result = true;
                    newly = true;
                    // [EMIT] `|-start|<mon>|perish3|[silent]` (the apply's silent form;
                    // the residual then prints the non-silent perish3/2/1 ticks).
                    if self.logging() {
                        let m = self.mon_ref(s, a, dex);
                        self.log.volatile_start_silent(&m, "perish3");
                    }
                }
            }
            if !any_result {
                // Every active already counted → `[still]` + `-fail|<user>` (P2's
                // all-already-counted re-cast). DRAW-FREE.
                if self.logging() {
                    self.log.attr_last_move_still();
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.fail(&user, None, false);
                }
            } else if newly && self.logging() {
                // [EMIT] ONE `|-fieldactivate|move: Perish Song` iff >= 1 newly applied.
                self.log.fieldactivate_move("Perish Song");
            }
            // (>= 1 result but 0 newly applied = the SILENT success — no fail, no
            // fieldactivate; the `-immune`s already emitted.)
            return MoveResolution::done(false, false, false);
        }

        // --- MEAN LOOK / SPIDER WEB / BLOCK — the trap-VOLATILE moves (identical
        //     mechanic; never-miss → the landed cast is ZERO draws, T1-T9 all match the
        //     splash-control trajectory). `onHit → target.addVolatile('trapped', source,
        //     move, 'trapper')` — the LINKED pair: the target's FIRM trap (`trapped:true`
        //     on its very FIRST request — the Shadow-Tag shape, NO maybeTrapped phase;
        //     a rejected switch is `[Invalid choice]` with NO re-request) that ENDS the
        //     moment the TRAPPER leaves the field ANY way (voluntary switch T1 / faint
        //     T9 / drag T4 — both linked volatiles removed draw-free). gen3
        //     `trapped.noCopy` is FALSE → a Baton-Passing HOLDER passes the volatile
        //     (still trapped, same link — T3b; BP itself is LEGAL for a trapped mon,
        //     selfSwitch bypasses the trap gate). A grounded GHOST IS trapped (the
        //     arenatrap precedent). Blocks: `protect: 1` (blocked draw-free — these are
        //     never-miss, so no roll precedes it) + a SUBSTITUTE blocks (NO bypasssub:
        //     `[still]` + `-fail`, T5); a RE-APPLICATION into an already-trapped foe
        //     FAILS (`[still]` + `-fail`, draw-free). The volatiles add ZERO endTurn
        //     draws (the Condition handler's subOrder 2 never ties an Ability's 7). ---
        if matches!(move_id, "meanlook" | "spiderweb" | "block") {
            // (1) PROTECT BLOCK (never-miss → draw-free block, no accuracy roll before it).
            if self.protect_blocks(foe, foe_slot, false) {
                if self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.activate(&target, "Protect", None);
                }
                return MoveResolution::done(false, false, false);
            }
            // (2) SUBSTITUTE blocks (no bypasssub): `[still]` + `-fail|<user>`, draw-free.
            if self.sides[foe].pokemon[foe_slot].substitute.is_some() {
                if self.logging() {
                    self.log.attr_last_move_still();
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.fail(&user, None, false);
                }
                return MoveResolution::done(false, false, false);
            }
            // (3) ALREADY-TRAPPED re-application fails (`addVolatile` false): `[still]`
            //     + `-fail|<user>`, draw-free; the existing link unchanged.
            if self.sides[foe].pokemon[foe_slot].trapped_by.is_some() {
                if self.logging() {
                    self.log.attr_last_move_still();
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.fail(&user, None, false);
                }
                return MoveResolution::done(false, false, false);
            }
            // (4) APPLY the linked trap (the trapper's uid — the link end condition).
            let trapper_uid = self.sides[_side].pokemon[_slot].uid;
            self.sides[foe].pokemon[foe_slot].trapped_by = Some(trapper_uid);
            // [EMIT] `|-activate|<target>|trapped` (the `trapped` condition's onStart).
            if self.logging() {
                let target = self.mon_ref(foe, foe_slot, dex);
                self.log.activate(&target, "trapped", None);
            }
            return MoveResolution::done(false, false, false);
        }

        // --- BELLY DRUM — pay floor(maxhp/2), SET Atk to +6 exactly. DRAW-FREE both
        //     branches (probed). FAILS (`[still]` + `-fail|<user>`) iff `hp <= maxhp/2`
        //     (the FLOAT compare — integer-exact as `2*hp <= maxhp`: 262/524 fails, 263
        //     succeeds; odd 523 → 261 fails, 262 succeeds) OR Atk already >= +6 OR
        //     maxhp == 1. On success: `directDamage(floor(maxhp/2))` (bypasses the
        //     onDamage handlers — no Endure/Focus Band; can never faint, hp > cost) then
        //     `boost({atk:12})` → a SET to +6 from ANY stage, emitted as
        //     `|-setboost|<user>|atk|6|[from] move: Belly Drum` (the battle.boost
        //     bellydrum special-case — NOT `-boost`). Self-target → no blocks. ---
        if move_id == "bellydrum" {
            let (hp, maxhp, atk) = {
                let m = &self.sides[_side].pokemon[_slot];
                (m.hp, m.maxhp, m.boosts[0])
            };
            if 2 * (hp as u32) <= maxhp as u32 || atk >= 6 || maxhp == 1 {
                if self.logging() {
                    self.log.attr_last_move_still();
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.fail(&user, None, false);
                }
                return MoveResolution::done(false, false, false);
            }
            let cost = maxhp / 2; // floor — odd 523 pays 261, leaving 262
            self.sides[_side].pokemon[_slot].hp = hp - cost;
            // [EMIT] the directDamage `|-damage|<user>|<HP>` then the `-setboost`.
            if self.logging() {
                let user = self.mon_ref(_side, _slot, dex);
                let hps = self.hp_status(_side, _slot);
                self.log.damage(&user, &hps, None);
            }
            self.sides[_side].pokemon[_slot].boosts[0] = 6;
            if self.logging() {
                let user = self.mon_ref(_side, _slot, dex);
                self.log.setboost_from_move(&user, "atk", 6, "Belly Drum");
            }
            return MoveResolution::done(false, false, false);
        }

        // --- CHARGE — add the `charge` volatile (×2 the next Electric move's BP — the
        //     onBasePower chain fold in run_move; gen3 has NO +1 SpD, probed: boosts
        //     stayed {}). DRAW-FREE. The volatile lasts until the user's NEXT move
        //     attempt OF ANY KIND (onAfterMove + onMoveAborted for any move != charge —
        //     consumed by turn_loop after that move resolves; an idle Splash consumes
        //     it, only an Electric next move gets the ×2). A re-charge while up re-adds
        //     (onRestart — `-start` again, no draw). Self-target → no blocks. ---
        if move_id == "charge" {
            self.sides[_side].pokemon[_slot].charge = true;
            // [EMIT] `|-start|<user>|Charge` (onStart AND onRestart both announce).
            if self.logging() {
                let user = self.mon_ref(_side, _slot, dex);
                self.log.volatile_start(&user, "Charge");
            }
            return MoveResolution::done(false, false, false);
        }

        // --- MEMENTO — the user FAINTS (`selfdestruct: 'ifHit'`), the foe takes
        //     −2 Atk / −2 SpA. Never-miss in the RESOLVED gen3 (the base/gen4 acc-100
        //     is overridden to true — the probed surprise) → ZERO draws in all
        //     branches. A PROTECT blocks it (`flags.protect` → `-activate Protect`)
        //     and the user does NOT faint (ifHit); a SUBSTITUTE blocks it (`[still]` +
        //     `-fail|<user>`) and the user does NOT faint. On a HIT: the drops apply
        //     FIRST (via the shared boost machinery — Clear-Body-class `onTryBoost`
        //     gates block the drop but the move still HIT → the user STILL faints; at
        //     a −6 floor the sim emits delta-0 `-unboost` lines — an emission nuance
        //     the shared helper skips, state-identical), THEN the user faints
        //     (deferred-faint protocol: the foe's queued move is CANCELLED by gen3
        //     faint-cancels-all; no Quick Claw — a landed memento turn where the user
        //     moves first consumes 0 draws TOTAL, ME1). Last-mon memento → the foe
        //     WINS (the standard pokemon_left machinery). ---
        if move_id == "memento" {
            debug_assert!(never_miss, "memento expected gen-3 accuracy true (never-miss)");
            if self.protect_blocks(foe, foe_slot, false) {
                if self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.activate(&target, "Protect", None);
                }
                return MoveResolution::done(false, false, false);
            }
            if self.sides[foe].pokemon[foe_slot].substitute.is_some() {
                if self.logging() {
                    self.log.attr_last_move_still();
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.fail(&user, None, false);
                }
                return MoveResolution::done(false, false, false);
            }
            // The −2 Atk / −2 SpA foe drops (the move's declarative `boosts` — id-gated
            // like the screen/weather ids; the extractor's statDropBoosts deliberately
            // excludes Memento for its selfdestruct). Clear Body / White Smoke block
            // both, Hyper Cutter the Atk half — reusing the shared stat-drop machinery.
            let spec = crate::dex::moves::SecondaryBoost {
                chance: 100,
                target_self: false,
                boosts: vec![(0, -2), (2, -2)], // atk −2, spa −2
            };
            self.apply_secondary_boost(_side, _slot, foe, foe_slot, false, std::slice::from_ref(&spec), true, dex);
            // The SELF-FAINT (`selfdestruct: 'ifHit'` — AFTER the drops applied/blocked;
            // the user faints even when the drops are fully blocked/floored). The
            // deferred-faint protocol: zero HP + record the faint order; process_faints
            // at the runAction tail emits `|faint|` + cancels the foe's queued move.
            let user_hp = self.sides[_side].pokemon[_slot].hp;
            self.apply_damage(_side, _slot, user_hp);
            return MoveResolution::done(false, false, false);
        }

        // --- MIMIC — copy the target's LAST-USED move over the user's Mimic slot for
        //     this field stay. Never-miss → ZERO draws in all branches (probed). FAILS
        //     (`[still]` + `-fail|<user>`, each draw-free): the target behind a
        //     SUBSTITUTE (an EXPLICIT onHit check — the `bypasssub` flag does NOT save
        //     it, MI4) / no lastMove / lastMove carries `failmimic` (data-driven
        //     `MoveData::fail_mimic`; a Struggle lastMove is `None` here, matching its
        //     failmimic flag) / the user already KNOWS the move. (The transformed-user
        //     gate is unreachable — Transform is unmodeled + fail-loud.) A PROTECT
        //     blocks it (`flags.protect`). On success the Mimic slot is OVERWRITTEN:
        //     `{move, pp: min(5, copied.pp), maxpp: calculatePP(copied, 3)}` +
        //     `|-activate|<user>|move: Mimic|<MoveName>`; the slot REVERTS on
        //     switch-out (Mimic's own remaining PP persists — `restore_mimic_overlay`). ---
        if move_id == "mimic" {
            debug_assert!(never_miss, "mimic expected gen-3 accuracy true (never-miss)");
            if self.protect_blocks(foe, foe_slot, false) {
                if self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.activate(&target, "Protect", None);
                }
                return MoveResolution::done(false, false, false);
            }
            let fail = {
                if self.sides[foe].pokemon[foe_slot].substitute.is_some() {
                    true
                } else {
                    match self.sides[foe].pokemon[foe_slot].last_move {
                        None => true,
                        Some(lslot) => {
                            match self.move_at(foe, foe_slot, lslot, dex) {
                                None => true,
                                Some(lm) => {
                                    // gen3_mimic_hidden_power_type_v1: compare the BARE-canonicalized
                                    // copied id, so a mimicker that owns its OWN Hidden Power fails
                                    // to Mimic the foe's Hidden Power (Showdown compares bare
                                    // `hiddenpower == hiddenpower`).
                                    let copied_id = if lm.id.starts_with("hiddenpower") {
                                        "hiddenpower".to_string()
                                    } else {
                                        lm.id.clone()
                                    };
                                    let fail_mimic = lm.fail_mimic;
                                    fail_mimic
                                        || self.sides[_side].pokemon[_slot].set.moves.iter().any(|mid| {
                                            dex.moves(mid)
                                                .map(|m| {
                                                    let m_id = if m.id.starts_with("hiddenpower") {
                                                        "hiddenpower"
                                                    } else {
                                                        m.id.as_str()
                                                    };
                                                    m_id == copied_id
                                                })
                                                .unwrap_or(false)
                                        })
                                }
                            }
                        }
                    }
                }
            };
            if fail {
                if self.logging() {
                    self.log.attr_last_move_still();
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.fail(&user, None, false);
                }
                return MoveResolution::done(false, false, false);
            }
            // SUCCESS — overlay the Mimic slot with the copied move.
            let lslot = self.sides[foe].pokemon[foe_slot].last_move.expect("checked above");
            let (copied_id, copied_name, copied_pp, copied_maxpp) = {
                let lm = self.move_at(foe, foe_slot, lslot, dex).expect("checked above");
                // gen3_mimic_hidden_power_type_v1: Hidden Power's TYPE is derived from the USER's
                // IVs at use time (Showdown stores the moveslot id bare `hiddenpower`), so a MIMIC
                // of Hidden Power must store the BARE id (+ bare "Hidden Power" name), else the
                // mimicker keeps the copied mon's TYPED HP type (e.g. Grass) instead of re-deriving
                // from its OWN IVs (the ab_777_3 Claydol-Mimic-Charizard-HP over-resist bug).
                if lm.id.starts_with("hiddenpower") {
                    ("hiddenpower".to_string(), "Hidden Power".to_string(), lm.pp.min(5), lm.max_pp())
                } else {
                    (lm.id.clone(), lm.name.clone(), lm.pp.min(5), lm.max_pp())
                }
            };
            let mslot = (0..self.sides[_side].pokemon[_slot].set.moves.len()).find(|&k| {
                self.move_at(_side, _slot, k, dex).map(|m| m.id == "mimic").unwrap_or(false)
            });
            let mslot = match mslot {
                Some(k) => k,
                None => {
                    // `mimicIndex < 0` (the sim's guard — unreachable: the user just
                    // USED Mimic, so the slot exists). Draw-free silent fail.
                    return MoveResolution::done(false, false, false);
                }
            };
            {
                let m = &mut self.sides[_side].pokemon[_slot];
                let base_pp = m.move_pp.get(mslot).copied().unwrap_or(0);
                let base_maxpp = m.move_maxpp.get(mslot).copied().unwrap_or(0);
                m.mimic_overlay = Some(crate::state::MimicOverlay { slot: mslot, base_pp, base_maxpp });
                m.set.moves[mslot] = copied_id;
                if let Some(pp) = m.move_pp.get_mut(mslot) {
                    *pp = copied_pp;
                }
                if let Some(mp) = m.move_maxpp.get_mut(mslot) {
                    *mp = copied_maxpp;
                }
            }
            // `gen3_mimic_disable_self_overwrite_v1`: Mimic OVERWROTE its own slot, so the
            // move actually USED (`mimic`) is no longer in any slot. A FOE's Disable on this
            // mon (while `lastMove` is still this Mimic) must FAIL (Showdown's `onStart`
            // `!hasMove` return-false) — flag it so the disable arm reproduces that.
            self.sides[_side].pokemon[_slot].last_move_was_self_overwrite = true;
            // [EMIT] `|-activate|<user>|move: Mimic|<MoveName>`.
            if self.logging() {
                let user = self.mon_ref(_side, _slot, dex);
                self.log.activate(&user, "move: Mimic", Some(&copied_name));
            }
            return MoveResolution::done(false, false, false);
        }

        // --- PAIN SPLIT — average the two actives' HP. Never-miss → ZERO draws in all
        //     branches (probed; a Ghost target WORKS — Status ignoreImmunity). A
        //     PROTECT blocks it (`flags.protect`); a SUBSTITUTE blocks it (`[still]` +
        //     `-fail`, no bypasssub). On success: `avg = floor((t.hp + u.hp) / 2)`;
        //     BOTH are set to `min(avg, own maxhp)` (the sim's sethp clamps — Blissey
        //     41+714 → avg 377: Blissey drops to 377, Gengar caps at its 261 maxhp —
        //     NOT conservative). Lines: the TARGET's `-sethp … [silent]` THEN the
        //     USER's `-sethp`. Equal HP still succeeds (lines emitted, no fail). ---
        if move_id == "painsplit" {
            debug_assert!(never_miss, "painsplit expected gen-3 accuracy true (never-miss)");
            if self.protect_blocks(foe, foe_slot, false) {
                if self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.activate(&target, "Protect", None);
                }
                return MoveResolution::done(false, false, false);
            }
            if self.sides[foe].pokemon[foe_slot].substitute.is_some() {
                if self.logging() {
                    self.log.attr_last_move_still();
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.fail(&user, None, false);
                }
                return MoveResolution::done(false, false, false);
            }
            let avg = {
                let u = self.sides[_side].pokemon[_slot].hp as u32;
                let t = self.sides[foe].pokemon[foe_slot].hp as u32;
                (((u + t) / 2) as u16).max(1) // `|| 1` (unreachable — both alive ≥ 1)
            };
            {
                let t = &mut self.sides[foe].pokemon[foe_slot];
                t.hp = avg.min(t.maxhp);
            }
            {
                let u = &mut self.sides[_side].pokemon[_slot];
                u.hp = avg.min(u.maxhp);
            }
            // [EMIT] target `-sethp [silent]` then user `-sethp` (the probed order).
            if self.logging() {
                let t = self.mon_ref(foe, foe_slot, dex);
                let thp = self.hp_status(foe, foe_slot);
                self.log.sethp_from_move(&t, &thp, "Pain Split", true);
                let u = self.mon_ref(_side, _slot, dex);
                let uhp = self.hp_status(_side, _slot);
                self.log.sethp_from_move(&u, &uhp, "Pain Split", false);
            }
            return MoveResolution::done(false, false, false);
        }

        // --- PSYCH UP — copy ALL the target's boost stages VERBATIM (incl. negatives
        //     AND zeros — the user's own prior stages are fully overwritten; all 7
        //     stages incl. acc/eva). Never-miss → ZERO draws (probed). NO `protect`
        //     flag → it copies THROUGH a Protect; `bypasssub` → works through a sub
        //     (probed: Calm Mind boosts made behind a sub are copied). A +Spe copy
        //     leaves `cached_speed` stale until the next re-cache site (the Dragon
        //     Dance convention). An all-zero copy is a SUCCESS, not a fail. ---
        if move_id == "psychup" {
            debug_assert!(never_miss, "psychup expected gen-3 accuracy true (never-miss)");
            let copied = self.sides[foe].pokemon[foe_slot].boosts;
            self.sides[_side].pokemon[_slot].boosts = copied;
            // [EMIT] `|-copyboost|<user>|<target>|[from] move: Psych Up`.
            if self.logging() {
                let user = self.mon_ref(_side, _slot, dex);
                let target = self.mon_ref(foe, foe_slot, dex);
                self.log.copyboost_from_move(&user, &target, "Psych Up");
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
                    // [EMIT] a genuine miss (acc<100 via Bright Powder / Lax Incense / a
                    // defender evasion boost — never at the flat 100, but reachable): the
                    // announce gains the `[miss]` attr + `|-miss|`, matching the sim
                    // (`gen3_status_move_miss_emit_v1` — these arms previously returned
                    // draw-free but emitted NOTHING, so a missed Encore/Roar/setup showed
                    // `|move|…` without `[miss]`).
                    if self.logging() {
                        let user = self.mon_ref(_side, _slot, dex);
                        let target = self.mon_ref(foe, foe_slot, dex);
                        self.log.attr_last_move_miss();
                        self.log.miss(&user, Some(&target));
                    }
                    return MoveResolution::done(true, false, false);
                }
            }
            // (2a) PROTECT BLOCK — gen-3 Roar / Whirlwind BOTH carry the `protect: 1` flag, so
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
            //      This PRECEDES the Soundproof immune check — gen3 runs Protect's `onTryHit`
            //      ahead of Soundproof's within the SAME TryHit event, so a Roar into a
            //      Protecting + Soundproof foe emits `-activate Protect`, NOT `-immune Soundproof`
            //      (SIM-PROBED: Loudred Roar into a Protecting Soundproof Electrode →
            //      `|-activate|Protect`). `gen3_protect_before_soundproof_v1`.
            if self.protect_blocks(foe, foe_slot, false) {
                // [EMIT] `|-activate|<protector>|Protect` — the blocked phaze (the `|move|`
                // announce already showed). No drag, no `-fail`. Observation-only.
                if self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.activate(&target, "Protect", None);
                }
                return MoveResolution::done(false, false, false);
            }
            // (2b) SOUNDPROOF — Roar IS a `flags.sound` move (Whirlwind is NOT), so a Roar into a
            //      Soundproof holder (that is NOT protecting — the Protect check above already
            //      returned) is IMMUNE at `onTryHit` (after the accuracy roll): `-immune|
            //      [from] ability: Soundproof`, NO `forceSwitchFlag`, NO drag → **NO `sample`
            //      draw**. VERIFIED vs the sim: identical to the natural-immune status path
            //      (accuracy drawn, then immune, no further roll).
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
            // [EMIT] a phaze with NO eligible foe bench mon FAILS → the did-nothing `[still]`
            // announce form + a bare `|-fail|<caster>` (`gen3_omniscient_byte_fuzz_v1` FORMS
            // 1+2 Roar-no-bench: `|move|p1a: Zapdos|Roar||[still]`, `|-fail|p1a: Zapdos`). The
            // `|drag|` line for a SUCCESS is emitted later at the runAction tail (Phase 1's
            // `drag_in`).
            if force.is_none() && self.logging() {
                self.log.attr_last_move_still();
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
                    // [EMIT] a genuine miss (acc<100 via Bright Powder / Lax Incense / a
                    // defender evasion boost): the announce gains the `[miss]` attr THEN
                    // `|-miss|` (`gen3_status_move_miss_emit_v1` — the attr was missing).
                    if self.logging() {
                        let user = self.mon_ref(_side, _slot, dex);
                        let target = self.mon_ref(foe, foe_slot, dex);
                        self.log.attr_last_move_miss();
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
            // (4a) THE MIMIC SELF-OVERWRITE GUARD (`gen3_mimic_disable_self_overwrite_v1`) —
            //     the target's LAST move was a Mimic that OVERWROTE its own slot, so the used
            //     move id (`"mimic"`) is no longer in any moveSlot. Showdown's Disable stores
            //     `lastMove.id` and its `onStart` `!hasMove` returns false → Disable FAILS. Like
            //     the 0-PP guard this is an onStart return-false, so it sits AFTER the
            //     `random(2,6)` draw (DRAW-CONSUMED — the `durationCallback` fired first) and
            //     emits the `[still]` did-nothing form + `-fail`, NO volatile / `-start` /
            //     residual handler. The A/B repro rmry3vbgm_ab_6_16: a Mimic-copied Dynamic
            //     Punch the port wrongly disabled by SLOT where the sim leaves it usable.
            if self.sides[foe].pokemon[foe_slot].last_move_was_self_overwrite {
                if self.logging() {
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.attr_last_move_still();
                    self.log.fail(&user, None, false);
                }
                return MoveResolution::done(false, false, false);
            }
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

        // ============================================================================
        // MOVE-COVERAGE BATCH 3 (`gen3_move_coverage_batch3_v1`) — three STATEFUL,
        // DRAW-FREE move classes: CURSE / WISH / BATON PASS. Each was probe-settled
        // bit-for-bit vs the omniscient sim (`harness/probe_batch3_*.js`).
        // ============================================================================

        // --- CURSE (`curse`, `curse.onModifyMove` + `onHit`) — the type-conditional
        //     move. The `onModifyMove` (announce block above computes `curse_ghost` /
        //     `curse_still`) re-targets by the USER's type:
        //       * NON-GHOST user: `move.self = {boosts:{atk:1,def:1,spe:-1}}`, target SELF.
        //         So it is a DRAW-FREE self-boost `{atk:+1, def:+1, spe:-1}` on the USER —
        //         the mixed +/- setup. Line order (VERIFIED vs the sim): `-unboost <user>
        //         spe|1`, `-boost <user> atk|1`, `-boost <user> def|1` (the -Spe emitted
        //         FIRST). The `-1 Spe` updates only `boosts[4]`; `cached_speed` stays STALE
        //         (like Dragon Dance) so it can flip the first mover NEXT turn.
        //       * GHOST user, foe SUBSTITUTED: `onModifyMove` deletes the volatileStatus +
        //         onHit → the move does NOTHING: `[still]` (announced above) + `|-fail|
        //         <user>`, no HP cost, no volatile change. DRAW-FREE.
        //       * GHOST user, already-cursed foe: re-curse FAILS: `[still]` + `-fail`, no HP
        //         cost, no volatile change. DRAW-FREE (`addVolatile` returns false).
        //       * GHOST user, non-subbed, non-cursed foe: lay the `curse` volatile on the FOE
        //         (the `onHit` runs `addVolatile('curse')` — `curse.onStart` emits `|-start|
        //         <foe>|Curse|[of] <user>`) THEN pay `floor(maxhp/2)` HP (`this.damage(source.
        //         baseMaxhp/2, source, source)` — a bare `|-damage|<user>|<hp>`, NO `[from]`).
        //         The volatile is laid BEFORE the HP cost (VERIFIED: at hp<=maxhp/2 the ghost
        //         lays the curse THEN faints from the self-cost, foe still cursed). A GHOST
        //         target is NOT immune (the curse volatile has no type gate). DRAW-FREE.
        //     `landed` FALSE (a status `moveHit` returns undefined). The Curse RESIDUAL (order
        //     10, subOrder 8) lives in `run_residuals` / `apply_curse` (chips the cursed foe
        //     floor(maxhp/4)/turn); the volatile clears on switch-out + faint like leech_seed.
        if move_id == "curse" {
            if !curse_ghost {
                // NON-GHOST: the self-boost {atk:+1, def:+1, spe:-1} on the USER. Its
                // `move.self = {boosts}` goes through the gen3 `selfDrops` path
                // (battle-actions.ts:1338, `gen3_move_coverage_batch1_v1`), which DRAWS ONE
                // `random(100)` (the `secondaryRoll`) — then applies the boost UNCONDITIONALLY
                // (`self.chance === undefined`). So the non-ghost curse is NOT draw-free: it
                // consumes ONE `random(100)` (the value is discarded), exactly like
                // Overheat/Superpower's self-drop. VERIFIED vs the sim (`probe_batch3_curse.js`
                // /the iso probe: Snorlax curse draws +1 vs a plain Harden). Draw it here, then
                // apply the boost.
                let _ = self.prng.random_below(100);
                // Line order: -Spe first, then +Atk, then +Def (the `move.self.boosts`
                // iteration order atk→def→spe but the -Spe is emitted FIRST — VERIFIED vs the
                // golden's line order). Apply in the fixed (atk,def,spe) index order but emit
                // spe before atk/def to match.
                for &(idx, stages) in &[(4usize, -1i8), (0, 1), (1, 1)] {
                    let cur = self.sides[_side].pokemon[_slot].boosts[idx] as i32;
                    let next = (cur + stages as i32).clamp(-6, 6);
                    self.sides[_side].pokemon[_slot].boosts[idx] = next as i8;
                    if self.logging() {
                        let delta = (next - cur) as i8;
                        let user = self.mon_ref(_side, _slot, dex);
                        self.log.boost(&user, idx, delta);
                    }
                }
                return MoveResolution::done(false, false, false);
            }
            // GHOST branch. A foe SUBSTITUTE or an ALREADY-CURSED foe → the move does NOTHING
            // (draw-free, no HP cost). The `[still]` announce already emitted above for the
            // SUB case; the already-cursed case shows the FOE-target announce (emitted above)
            // then `[still]` retro-edit + `-fail`. Both emit `-fail` on the USER.
            let subbed = self.sides[foe].pokemon[foe_slot].substitute.is_some();
            let already_cursed = self.sides[foe].pokemon[foe_slot].curse.is_some();
            if subbed || already_cursed {
                if self.logging() {
                    // The sub case already announced `[still]`; the already-cursed case
                    // needs the `[still]` retro-edit (the foe-target announce showed first).
                    if !subbed {
                        self.log.attr_last_move_still();
                    }
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.fail(&user, None, false);
                }
                return MoveResolution::done(false, false, false);
            }
            // Lay the curse on the FOE (recording the caster side for the `[of]` clause),
            // THEN pay floor(maxhp/2) HP. Both DRAW-FREE.
            self.sides[foe].pokemon[foe_slot].curse = Some(_side);
            if self.logging() {
                let foe_ref = self.mon_ref(foe, foe_slot, dex);
                let user = self.mon_ref(_side, _slot, dex);
                self.log.volatile_start_of(&foe_ref, "Curse", &user);
            }
            // The self-cost: floor(baseMaxhp/2), applied via apply_damage so a self-KO goes
            // through the normal faint machinery (VERIFIED: at hp<=maxhp/2 the ghost lays the
            // curse THEN faints; foe stays cursed). No `[from]` cause (a bare `-damage`).
            let cost = self.sides[_side].pokemon[_slot].maxhp / 2;
            self.apply_damage(_side, _slot, cost);
            if self.logging() {
                let user = self.mon_ref(_side, _slot, dex);
                let hp = self.hp_status(_side, _slot);
                self.log.damage(&user, &hp, None);
            }
            return MoveResolution::done(false, false, false);
        }

        // --- WISH (`wish`, `slotCondition:'Wish'`, duration 2) — a slot-keyed DELAYED heal.
        //     Never-miss (`target: self`), DRAW-FREE at cast. The gen-3 model (VERIFIED vs
        //     `harness/probe_batch3_wish.js`):
        //       * CAST: if a Wish is ALREADY pending on this side → FAILS: `|move|<user>|
        //         Wish||[still]` (the `[still]` announce form) + a bare `|-fail|<user>`
        //         (`gen3_omniscient_byte_fuzz_v1` FORMS 1+2 double-Wish — the `slotCondition`
        //         add returns false → `runMoveEffects` sets `moveResult=false` → `-fail` +
        //         `attrLastMove('[still]')`; captured L-row ab_0_9: `|move|…|Wish||[still]`,
        //         `|-fail|p2a: Salamence`), draw-free, the existing Wish untouched. Else set
        //         `wish_pending = (2, wisher_name)`.
        //       * The heal fires at the end-of-turn RESIDUAL (`wish.onEnd`, order 7 — see
        //         `run_residuals` / `apply_wish`).
        //     `landed` FALSE.
        if move_id == "wish" {
            debug_assert!(never_miss, "wish expected gen-3 never_miss (target self)");
            if self.sides[_side].wish_pending.is_some() {
                // Double-Wish: `[still]` retro-edit + a bare `|-fail|<user>`.
                if self.logging() {
                    self.log.attr_last_move_still();
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.fail(&user, None, false);
                }
                return MoveResolution::done(false, false, false);
            }
            // Set the pending Wish (duration 2). The wisher name is the CASTER's display name
            // (`source.name`) for the `[wisher]` clause at resolution.
            let wisher = self.display_name(_side, _slot, dex);
            self.sides[_side].wish_pending = Some((WISH_DURATION, wisher));
            return MoveResolution::done(false, false, false);
        }

        // --- BATON PASS (`batonpass`, `selfSwitch:'copyvolatile'`, `onHit`) — a self-switch
        //     that PASSES the outgoing mon's boosts + copyable volatiles to the entrant. The
        //     gen-3 model (VERIFIED vs `harness/probe_batch3_batonpass.js`):
        //       * NO eligible bench (last mon / every other fainted) → `onHit` FAILS:
        //         `|move|<user>|Baton Pass||[still]` + `|-fail|<user>`, draw-free (the move
        //         still counts as used — `NOT_FAIL`). The self-target announce showed the USER
        //         above; add the `[still]` retro-edit + `-fail`.
        //       * An eligible bench exists → set the side's `switch_flag` + the
        //         `baton_pass_pending` marker; the FORCED switch-in (via the normal switch
        //         machinery) then runs `copyVolatileFrom` in `execute_switch` (snapshot the
        //         passer's pass-set BEFORE clearVolatile, apply to the entrant AFTER the swap,
        //         tag the `|switch|` with `[from] Baton Pass`). DRAW-FREE at the move; the
        //         forced switch-in draws exactly like a normal switch. `landed` FALSE.
        if move_id == "batonpass" {
            debug_assert!(never_miss, "batonpass expected gen-3 never_miss (target self)");
            if self.can_switch(_side) {
                // Signal the self-switch: force this side to replace + mark the pass so
                // `execute_switch` runs copyVolatileFrom. The turn loop pulls the forced
                // `switch` request for this side after the move resolves.
                self.sides[_side].switch_flag = true;
                self.sides[_side].baton_pass_pending = true;
            } else {
                // No eligible bench: FAIL draw-free. `[still]` retro-edit + `-fail` on the USER.
                if self.logging() {
                    self.log.attr_last_move_still();
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.fail(&user, None, false);
                }
            }
            return MoveResolution::done(false, false, false);
        }

        // ============================================================================
        // MOVE-COVERAGE BATCH 2 (`gen3_move_coverage_batch2_v1`) — four DRAW-friendly
        // status-move classes: STATUS-CURE / WEATHER-SET / STAT-DROP / SCREENS. Each was
        // probe-settled bit-for-bit vs the omniscient sim (`harness/probe_batch2_movecoverage.js`).
        // ============================================================================

        // --- STATUS-CURE (`refresh` self / `healbell` + `aromatherapy` whole-team) — a
        //     never-miss Status move that clears major status. DRAW-FREE (VERIFIED: a cure
        //     turn draws only the existing action-order / Quick Claw draws — the cure's
        //     `onHit` consumes NO PRNG):
        //       * REFRESH (`curesSelfStatus`, `target: self`) — cures the USER's par / psn /
        //         brn ONLY (`onHit`: `if (["", "slp", "frz"].includes(status)) return false`),
        //         so it FAILS (draw-free `-fail`) on none/sleep/freeze; emits
        //         `|-curestatus|<user>|<status>|[msg]` on a cure.
        //       * HEAL BELL (`curesTeamStatus`, `sound`) — emits `|-activate|<user>|move: Heal
        //         Bell` then iterates the WHOLE team (active + bench), SKIPPING a Soundproof
        //         ally (`|-immune|<ally>|[from] ability: Soundproof` if the ally is active),
        //         curing each other ally draw-free (`|-curestatus|<ident>|<status>|[silent]`,
        //         the bench renders as a SIDE ref).
        //       * AROMATHERAPY (`curesTeamStatus`, NOT sound) — emits `|-cureteam|<user>|[from]
        //         move: Aromatherapy` then clears EVERY ally's status (no Soundproof gate, no
        //         per-mon `-curestatus`).
        //     `landed` FALSE (a status `moveHit` returns undefined → no in-tryMoveHit Update).
        if dex.moves(move_id).map(|m| m.cures_self_status).unwrap_or(false) {
            // REFRESH — self-cure ANY major status EXCEPT sleep / freeze / none (the gen3
            // `onHit`: `if (["", "slp", "frz"].includes(status)) return false; cureStatus()`).
            // So it cures par / psn / **tox** / brn (Toxic IS cured — the missing case that
            // desynced the e2e Refresh teams). NEVER-MISS → no accuracy draw, DRAW-FREE.
            let cured = matches!(
                self.sides[_side].pokemon[_slot].status,
                Some(Status::Paralysis) | Some(Status::Poison) | Some(Status::Burn) | Some(Status::Toxic(_))
            );
            if cured {
                let tok = status_token(self.sides[_side].pokemon[_slot].status).unwrap_or("");
                self.sides[_side].pokemon[_slot].status = None;
                if self.logging() {
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.curestatus(&user, tok, true); // `[msg]` form
                }
            } else if self.logging() {
                // no curable status (none / slp / frz) → `onHit` returns false → the
                // did-nothing `[still]` announce form + a bare `|-fail|<user>`
                // (`gen3_omniscient_byte_fuzz_v1` FORMS 1+2 Refresh-no-status:
                // `|move|p2a: Milotic|Refresh||[still]`, `|-fail|p2a: Milotic`).
                self.log.attr_last_move_still();
                let user = self.mon_ref(_side, _slot, dex);
                self.log.fail(&user, None, false);
            }
            return MoveResolution::done(false, false, false);
        }
        if dex.moves(move_id).map(|m| m.cures_team_status).unwrap_or(false) {
            // HEAL BELL / AROMATHERAPY — whole-team major-status cure (never-miss → no
            // accuracy draw). DRAW-FREE. `_side` owns the team.
            let is_heal_bell = self.move_is_sound(move_id, dex); // healbell = sound; aromatherapy not
            if self.logging() {
                let user = self.mon_ref(_side, _slot, dex);
                if is_heal_bell {
                    self.log.activate(&user, "move: Heal Bell", None);
                } else {
                    self.log.cureteam_aromatherapy(&user);
                }
            }
            let active = self.sides[_side].active;
            let team_len = self.sides[_side].pokemon.len();
            for i in 0..team_len {
                // HEAL BELL only: a Soundproof ally is SKIPPED (not cured); emits `-immune`
                // if that ally is ACTIVE. Aromatherapy has no sound flag → cures everyone.
                if is_heal_bell
                    && to_id(&self.sides[_side].pokemon[i].ability) == "soundproof"
                {
                    if self.logging() && i == active {
                        let ally = self.mon_ref(_side, i, dex);
                        self.log.immune_from_ability(&ally, "Soundproof");
                    }
                    continue;
                }
                if self.sides[_side].pokemon[i].status.is_none() {
                    continue; // nothing to cure on this ally
                }
                let tok = status_token(self.sides[_side].pokemon[i].status).unwrap_or("");
                self.sides[_side].pokemon[i].status = None;
                // Heal Bell emits a per-mon `-curestatus [silent]`; Aromatherapy emits none
                // (its single `-cureteam` banner covers the side).
                if is_heal_bell && self.logging() {
                    // The cured ally's ident: an ACTIVE ally is `pNa: <name>`; a BENCH ally is
                    // the position-less `pN: <name>` (the sim's `Pokemon.toString()` for a
                    // non-active mon) — the MON's nickname/species, NOT the player name. The
                    // prior port rendered `side_ref` = `pN: <player_name>` (e.g. `p2: P2`), a
                    // byte divergence on any Heal Bell curing a statused BENCH mon
                    // (`gen3_omniscient_byte_fuzz_v1`).
                    let ident = if i == active {
                        self.mon_ref(_side, i, dex).to_string()
                    } else {
                        format!("p{}: {}", _side + 1, self.display_name(_side, i, dex))
                    };
                    self.log.curestatus_silent(&ident, tok);
                }
            }
            return MoveResolution::done(false, false, false);
        }

        // --- WEATHER-SET (`raindance` → Rain / `sunnyday` → Sun) — a never-miss `target:all`
        //     Status move that SETS field weather for 5 TURNS (a TIMED weather, distinct from
        //     the permanent ability weather). VERIFIED bit-for-bit vs the omniscient sim:
        //       * NEVER-MISS → NO accuracy draw. DRAW-FREE at the move itself (a distinct-
        //         speed set turn draws only Quick Claw). The eachEvent('WeatherChange') tie-
        //         shuffle DOES fire when the two actives tie on cached speed (the SAME model
        //         as the ability switch-in weather, already wired via `run_switch`) — handled
        //         by the shared each_event_shuffle at the WEATHER SET below.
        //       * `field.setWeather` (field.ts): if the SAME weather is already active it
        //         FAILS for a MOVE source (`gen > 2 → return false`), emitting `|-weather|<W>`
        //         then `|-fail|<caster>` and leaving the weather (incl. its duration) UNCHANGED.
        //         A DIFFERENT weather OVERWRITES (5-turn timer). Emits `|-weather|<W>` (the
        //         move-source onFieldStart form — NO `[from] ability`), sets `weather_turns=5`.
        //       * The 5-turn UPKEEP tick (`|-weather|<W>|[upkeep]`) + the expiry (`|-weather|
        //         none`) are handled at the end-of-turn field residual (`apply_weather_chip` /
        //         the field-residual duration countdown). `landed` FALSE.
        if let Some(new_weather) = modeled_weather_set_move(move_id) {
            debug_assert!(never_miss, "weather-set {move_id:?} expected never_miss");
            // setWeather: SAME weather already active → FAIL (gen>2 move source), draw-free.
            // The sim's `field.setWeather` returns false with NO `-weather` line (the
            // `this.weather === status.id` → `if (this.gen > 2) return false` early-out), then
            // the move's `onHit` returns false → `runMoveEffects` emits the did-nothing
            // `[still]` announce form + a bare `|-fail|<caster>` (`gen3_omniscient_byte_fuzz_v1`
            // FORMS 1+2 weather-set-into-same: `|move|p1a: Magneton|Rain Dance||[still]`,
            // `|-fail|…` — the port previously emitted a SPURIOUS `-weather` line + no `[still]`).
            if self.field.weather == Some(new_weather) {
                if self.logging() {
                    self.log.attr_last_move_still();
                    let caster = self.mon_ref(_side, _slot, dex);
                    self.log.fail(&caster, None, false);
                }
                return MoveResolution::done(false, false, false);
            }
            // OVERWRITE (or set from clear): a 5-turn TIMED weather. gen3 has no Damp/Heat
            // Rock → always 5. The eachEvent('WeatherChange') tie-shuffle fires iff the
            // actives tie on cached speed (mirrors the ability switch-in weather change).
            self.field.weather = Some(new_weather);
            self.field.weather_turns = WEATHER_MOVE_DURATION;
            if self.logging() {
                // The move-source onFieldStart line: `|-weather|<W>` (no upkeep, no ability).
                self.log.weather(weather_display(new_weather), None, None, false);
            }
            // The `eachEvent('WeatherChange')` draw (only on a speed tie). Return value
            // unused — this is the DRAW, not an emit-order (matches `run_switch`'s use).
            self.each_event_shuffle();
            return MoveResolution::done(false, false, false);
        }

        // --- SCREENS (`lightscreen` / `reflect`) — a never-miss `target: allySide` Status
        //     move that sets a 5-turn SIDE CONDITION halving incoming special / physical
        //     damage. VERIFIED bit-for-bit vs the sim:
        //       * NEVER-MISS → NO accuracy draw. DRAW-FREE (a set turn draws only Quick Claw).
        //       * The condition begins on the CASTER's own side with duration 5 (gen3 has no
        //         Light Clay → always 5); `|-sidestart|<side>|move: Light Screen` (Light
        //         Screen) / `|-sidestart|<side>|Reflect` (Reflect). A re-use while ALREADY up
        //         FAILS (`addSideCondition` false → `|-fail|<caster>`), the existing timer
        //         UNCHANGED, draw-free.
        //       * The damage calc reads `sides[foe].light_screen/reflect > 0` (already wired in
        //         `build_damage_context`). The 5-turn countdown + expiry (`|-sideend|`) are the
        //         end-of-turn SIDE residual. `landed` FALSE. ---
        if let Some(is_reflect) = modeled_screen_move(move_id) {
            debug_assert!(never_miss, "screen {move_id:?} expected never_miss");
            let up = if is_reflect {
                self.sides[_side].reflect > 0
            } else {
                self.sides[_side].light_screen > 0
            };
            if up {
                // already up → addSideCondition false → the move's `onHit` returns false →
                // the did-nothing `[still]` announce form + a bare `|-fail|<caster>`
                // (`gen3_omniscient_byte_fuzz_v1` FORMS 1+2 Light-Screen-already-up:
                // `|move|p1a: Zapdos|Light Screen||[still]`, `|-fail|…`). Draw-free.
                if self.logging() {
                    self.log.attr_last_move_still();
                    let caster = self.mon_ref(_side, _slot, dex);
                    self.log.fail(&caster, None, false);
                }
                return MoveResolution::done(false, false, false);
            }
            if is_reflect {
                self.sides[_side].reflect = SCREEN_DURATION;
            } else {
                self.sides[_side].light_screen = SCREEN_DURATION;
            }
            if self.logging() {
                let side_ref = crate::protocol::ProtocolBuilder::side_ref(_side, &self.sides[_side].name);
                // Light Screen's onSideStart emits `move: Light Screen`; Reflect emits `Reflect`.
                let effect = if is_reflect { "Reflect" } else { "move: Light Screen" };
                self.log.sidestart(&side_ref, effect);
            }
            return MoveResolution::done(false, false, false);
        }

        // --- STAT-DROP MOVE (Screech −2 Def / Charm −2 Atk / Metal Sound −2 SpD / Feather
        //     Dance −2 Atk / Tickle −1 Atk/−1 Def / Fake Tears −2 SpD / Cotton Spore / Scary
        //     Face −2 Spe) — a standalone foe-targeting Status move whose ENTIRE effect is a
        //     foe stat drop (`statDropBoosts` in the data). VERIFIED bit-for-bit vs the sim:
        //       1. ACCURACY — `randomChance(acc,100)` drawn unless never-miss (Screech / Metal
        //          Sound / Cotton Spore are acc-85/85/85 and CAN miss; Charm / Feather Dance /
        //          Tickle / Fake Tears are acc-100 but NOT never-miss so they STILL draw one
        //          roll). This is the ONLY per-move draw.
        //       2. SOUNDPROOF — Screech / Metal Sound carry `flags.sound`, so vs a Soundproof
        //          holder the move is IMMUNE at TryHit (accuracy drawn, then `-immune|[from]
        //          ability: Soundproof`, no drop). Charm / Feather Dance / Tickle / Fake Tears
        //          are NOT sound.
        //       3. PROTECT BLOCK — all carry `protect: 1` → a Protect/Detect on the target
        //          blocks it (accuracy drawn, `-activate Protect`, no drop). Tickle also
        //          `bypasssub` but the others don't — a SUBSTITUTE blocks a non-bypasssub
        //          stat-drop (the sub's onTryHit). (Handled below.)
        //       4. APPLY `boost()` on the FOE, ±6 clamp, DRAW-FREE, with the Clear Body /
        //          White Smoke / Hyper Cutter / Keen Eye `onTryBoost` immunity gates (reusing
        //          `apply_secondary_boost`, which emits `|-unboost|` per applied stat OR the
        //          `|-fail|…|unboost|[from] ability|…` block on an immunity). `landed` FALSE. ---
        if !dex.moves(move_id).map(|m| m.stat_drop_boosts.is_empty()).unwrap_or(true) {
            // (1) ACCURACY — drawn unless never-miss (the modeled set is never never-miss).
            let acc_hit = if never_miss {
                true
            } else {
                self.roll_accuracy(_side, _slot, foe, foe_slot, accuracy, never_miss, move_type, dex)
            };
            if !acc_hit {
                if self.logging() {
                    self.log.attr_last_move_miss();
                    let user = self.mon_ref(_side, _slot, dex);
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.miss(&user, Some(&target));
                }
                return MoveResolution::done(true, false, false);
            }
            // (2) PROTECT BLOCK (foe-targeting): blocked at TryHit after accuracy — and
            //     BEFORE the Soundproof immune check. gen3 runs Protect's `onTryHit` ahead of
            //     Soundproof's within the SAME TryHit event, so a sound stat-drop into a
            //     Protecting + Soundproof foe emits `-activate Protect`, NOT `-immune Soundproof`
            //     (SIM-PROBED: Loudred Screech into a Protecting Soundproof Electrode →
            //     `|-activate|Protect`). `gen3_protect_before_soundproof_v1`.
            if self.protect_blocks(foe, foe_slot, false) {
                if self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.activate(&target, "Protect", None);
                }
                return MoveResolution::done(false, false, false);
            }
            // (3) SOUNDPROOF (sound stat-drops only): immune after the accuracy roll + AFTER
            //     the Protect check (a Protecting foe already returned above).
            if self.move_is_sound(move_id, dex)
                && dex.ability(&to_id(&self.sides[foe].pokemon[foe_slot].ability)).map(|a| a.blocks_sound).unwrap_or(false)
            {
                if self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.immune_from_ability(&target, "Soundproof");
                }
                return MoveResolution::done(false, false, false);
            }
            // (3b) SUBSTITUTE BLOCK — a non-`bypasssub` stat-drop into a substituted foe is
            //      blocked by the sub (accuracy drawn, no drop). Tickle carries `bypasssub`,
            //      so it is EXEMPT (it drops a subbed foe). Draw-free past accuracy. The sim
            //      renders the did-nothing FORM-1: `|move|<user>|<Move>||[still]` + a BARE
            //      `|-fail|<user>` (captured `harness/probe_statdrop_substitute.js` — a Growl
            //      into a subbed Snorlax emits `|move|…|Growl||[still]` then `|-fail|…`, the
            //      SAME framing every other sub-blocked/did-nothing status arm emits). The
            //      byte fuzzer surfaced this residual (`gen3_omniscient_byte_fuzz_v1`).
            if move_id != "tickle" && self.sides[foe].pokemon[foe_slot].substitute.is_some() {
                if self.logging() {
                    self.log.attr_last_move_still();
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.fail(&user, None, false);
                }
                return MoveResolution::done(false, false, false);
            }
            // (4) APPLY the foe stat drop (DRAW-FREE) via the shared secondary-boost helper.
            //     Build the `(want_self=false)` spec from the move's `statDropBoosts`.
            let spec = crate::dex::moves::SecondaryBoost {
                chance: 100,
                target_self: false,
                boosts: dex.moves(move_id).map(|m| m.stat_drop_boosts.clone()).unwrap_or_default(),
            };
            self.apply_secondary_boost(_side, _slot, foe, foe_slot, false, std::slice::from_ref(&spec), true, dex);
            return MoveResolution::done(false, false, false);
        }

        // --- SNATCH CAST (`gen3_snatch_v1`, `snatch.onStart`) — a category-Status,
        //     priority-+4, never-miss, `target:self` move that sets the `snatch` volatile
        //     (`duration: 1`). DRAW-FREE: the `|move|<user>|Snatch|<user>` announce was
        //     emitted at the top of this fn (self-target); here we set the volatile + emit
        //     `|-singleturn|<user>|Snatch`. The volatile lives through THIS turn — so it can
        //     intercept the foe's snatchable status move (the interception block above) AND
        //     register the residual duration handler (`run_residuals`) — and clears at the
        //     next turn-top (`clear_flinch`; probe SN1: t2 vols=(none)). Casting into
        //     NOTHING just expires (draw-free removal). `landed` FALSE (a status `moveHit`
        //     returns undefined → no in-tryMoveHit Update). Snatch itself is NOT snatchable
        //     (its flags carry no `snatch` — a mirror steals nothing, probe SN12). ---
        if move_id == "snatch" {
            self.sides[_side].pokemon[_slot].snatch = true;
            // [EMIT] `|-singleturn|<user>|Snatch`.
            if self.logging() {
                let user = self.mon_ref(_side, _slot, dex);
                self.log.singleturn(&user, "Snatch");
            }
            return MoveResolution::done(false, false, false);
        }

        // --- HAZE (`gen3_haze_v1`, the gen-3 `haze` — a Status FIELD move,
        //     `type Ice`, `accuracy: true` [never-miss → NO accuracy draw], `target: all`,
        //     `priority 0`, resolved at the user's speed slot [NOT a residual]). Its gen-3
        //     `onHitField`: `this.add('-clearallboost'); for (const p of this.getAllActive())
        //     p.clearBoosts();` — so it emits ONE `|-clearallboost` FIELD line (NO per-mon
        //     `-clearboost`) and zeroes BOTH actives' boost stages INCLUDING the USER's own
        //     (`getAllActive()`). DRAW-FREE (probe `harness/probe_batch89_haze_trick_yawn.js` +
        //     the re-probe here: a Haze turn draws the SAME count as a Splash control — only the
        //     endTurn Quick Claw). `landed` FALSE (a status `moveHit` returns undefined → the
        //     in-tryMoveHit Update is skipped). The `|move|<user>|Haze|<user>` announce is
        //     already emitted at the top (`target: all` → `status_move_announce_renders_user`).
        //     No type-immunity / Substitute interaction (it's a field effect). ---
        if move_id == "haze" {
            // [EMIT] the ONE `|-clearallboost` field line (before the silent per-mon clears).
            if self.logging() {
                self.log.push_raw("|-clearallboost".to_string());
            }
            // clearBoosts() on BOTH actives (getAllActive — incl. the USER's own boosts).
            for (s, sl) in [(_side, _slot), (foe, foe_slot)] {
                self.sides[s].pokemon[sl].boosts = [0; crate::state::BOOST_LEN];
            }
            return MoveResolution::done(false, false, false);
        }

        // --- YAWN (`gen3_yawn_v1`, the gen-3 `yawn` — a category-Status foe-target move, type
        //     Normal, `accuracy: true` [never-miss → NO accuracy draw], `volatileStatus: 'yawn'`,
        //     flags `{protect, reflectable, mirror, metronome}` [NO `bypasssub`]). DELAYED SLEEP —
        //     the CRUX is that the sleep `random(2,6)` fires at RESOLVE (the residual `onEnd`), not
        //     at cast: the CAST is entirely DRAW-FREE. VERIFIED bit-for-bit vs the omniscient sim
        //     (`harness/probe_batch89_haze_trick_yawn.js` + `probe_yawn_edges`).
        //
        //     TryHit-order resolution (dispositions computed draw-free at the top): Protect
        //     (`-activate Protect`) > onTryHit already-statused (`[still]` + `-fail|<user>`, no
        //     volatile) > onTryHit sleep-immune (Insomnia / Vital Spirit via `runStatusImmunity
        //     ('slp')` → `-immune|<target>|[from] ability: <A>`, no volatile) > onTryPrimaryHit
        //     substitute (`[still]` + `-fail|<user>`, no volatile) > ADD the `yawn` volatile
        //     (`duration: 2`, DRAW-FREE) + `|-start|<target>|move: Yawn|[of] <source>`. `landed`
        //     ALWAYS FALSE. The RESOLVE (the residual `Yawn` handler at order 10 subOrder 19)
        //     routes through the existing `try_set_status('slp')` path. ---
        if move_id == "yawn" {
            // (1) PROTECT block (TryHit, highest priority — the announce kept the normal target
            //     form since `yawn_still` excludes the protected case).
            if yawn_protect {
                // [EMIT] `|-activate|<target>|Protect` (the standard block line).
                if self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.activate(&target, "Protect", None);
                }
                return MoveResolution::done(false, false, false);
            }
            // (2) onTryHit: the target ALREADY has a major status → `[still]` + `-fail|<user>`, no
            //     volatile. The announce ALREADY emitted the `[still]` form up-front via
            //     `yawn_still` (the Spikes-at-cap / ghost-Curse-into-sub precedent), so DO NOT
            //     re-append `[still]` here (that double-`[still]` bug — `|Yawn||[still]|[still]` —
            //     was the omniscient byte-fuzz find, master-seed 80808, ab_0_13/ab_4_12). Emit
            //     only the `|-fail|<user>`. DRAW-FREE.
            if yawn_statused {
                // [EMIT] just `|-fail|<user>` (the `[still]` is already on the announce).
                if self.logging() {
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.fail(&user, None, false);
                }
                return MoveResolution::done(false, false, false);
            }
            // (3) onTryHit: the target is SLEEP-IMMUNE (Insomnia / Vital Spirit — their `onImmunity`
            //     blocks `runStatusImmunity('slp')`) → `|-immune|<target>|[from] ability: <A>`, no
            //     volatile (the announce used the NORMAL target form). DRAW-FREE.
            if yawn_immune {
                if self.logging() {
                    let ability = to_id(&self.sides[foe].pokemon[foe_slot].ability);
                    let display = dex
                        .ability(&ability)
                        .map(|a| a.name.clone())
                        .unwrap_or_else(|| ability.clone());
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.immune_from_ability(&target, &display);
                }
                return MoveResolution::done(false, false, false);
            }
            // (4) onTryPrimaryHit: a SUBSTITUTE (no `bypasssub`) blocks the volatile → `[still]` +
            //     `-fail|<user>`, no volatile. Like (2), the announce ALREADY carried `[still]` via
            //     `yawn_still` up-front — emit only the `|-fail|<user>` (no double `[still]`).
            if yawn_subbed {
                if self.logging() {
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.fail(&user, None, false);
                }
                return MoveResolution::done(false, false, false);
            }
            // (4b) ALREADY-YAWNED GUARD (`gen3_yawn_recast_v1`) — the target ALREADY has a
            //      pending `yawn` volatile: Showdown's `addVolatile('yawn')` returns false (yawn
            //      has no `onRestart`), so the re-cast FAILS and the EXISTING yawn is UNCHANGED
            //      (it resolves on its ORIGINAL schedule). The port used to RE-SET the duration
            //      to 2 → the resolve (the sleep `random(2,6)`) slipped ONE turn late → a
            //      draw-ORDER desync (the A/B repro rmry3ytkn_ab_6_22 seed@44 — a Swalot
            //      re-Yawning a still-pending Blastoise). DRAW-FREE; emit `|move|…|Yawn||[still]`
            //      + `|-fail|<user>` (probe-verified, `harness/probe_yawn_recast_rng.js`).
            if self.sides[foe].pokemon[foe_slot].yawn.is_some() {
                if self.logging() {
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.attr_last_move_still();
                    self.log.fail(&user, None, false);
                }
                return MoveResolution::done(false, false, false);
            }
            // (5) ADD the `yawn` volatile (duration 2), recording the SOURCE uid for the `[of]`
            //     clause + the `trySetStatus` source. DRAW-FREE.
            let src_uid = self.sides[_side].pokemon[_slot].uid;
            self.sides[foe].pokemon[foe_slot].yawn = Some((2, src_uid));
            // [EMIT] `|-start|<target>|move: Yawn|[of] <source>` (the `yawn.onStart`).
            if self.logging() {
                let target = self.mon_ref(foe, foe_slot, dex);
                let user = self.mon_ref(_side, _slot, dex);
                self.log.volatile_start_of(&target, "move: Yawn", &user);
            }
            return MoveResolution::done(false, false, false);
        }

        // --- TRICK (`gen3_trick_v1`, the gen-3 `trick` num 271 — a category-Status ITEM-SWAP
        //     move, type Psychic, `accuracy: 100` [NOT never-miss → draws ONE
        //     randomChance(100,100)], `target: normal`, `ignoreImmunity: true`, flags
        //     `{protect, mirror, allyanim, noassist, failcopycat}` [NO `bypasssub`]). `switcheroo`
        //     (num 415, gen4) is NOT gen3-legal — do not add. DRAW MODEL: ONE accuracy draw, then
        //     a DRAW-FREE swap (the swap adds nothing). Probe-settled bit-for-bit vs the
        //     omniscient sim (`harness/probe_batch89_trick_edges.js` +
        //     `probe_batch89_haze_trick_yawn.js` + `probe_trick_open_qs{,2}.js`). The
        //     `|move|<user>|Trick|<foe>` announce is already emitted at the top (target: normal →
        //     foe form; Trick has no `move.status` so `foe_status_move_fail` is None).
        //
        //     Resolution ORDER (probe-verified priority): ACCURACY draw → on HIT: Protect
        //     (`-activate Protect`) > Sticky Hold `onTryImmunity` (PLAIN `-immune|<foe>`, NO
        //     `[from] ability`) > Substitute `onTryPrimaryHit` (no `bypasssub` → `[still]` +
        //     `-fail|<user>`) > the `onHit` item conditions. The gen-3 `trick.onHit` (moves.ts):
        //       `yourItem = target.takeItem(source); myItem = source.takeItem();`
        //       `takeItem` returns FALSE (gen<=4) iff EITHER side is `itemKnockedOff` (gen3 Knock
        //       Off CLEARS the item AND marks the slot, so a knocked-off mon is itemless-BUT-
        //       marked and its `takeItem` returns `false`, NOT the plain-itemless `undefined`);
        //       a truly itemless (non-knocked) mon returns `undefined` (falsy, not `=== false`).
        //       FAIL (`return false` → `[still]` + `-fail|<user>`, no swap) iff
        //       `yourItem === false || myItem === false || (!yourItem && !myItem)` — i.e. EITHER
        //       side `item_knocked_off`, OR both TRULY itemless. In gen3 Mail AND berries SWAP
        //       fine (their `TakeItem` event passes for Trick — probe-confirmed; the task's
        //       "Mail untradeable" hypothesis was WRONG).
        //       Else SWAP: `-activate|<user>|move: Trick|[of] <foe>`, then the TARGET's new-item
        //       line FIRST (user HAD an item → `-item|<foe>|<myItem>`, else the target loses its
        //       own item `-enditem|<foe>|<yourItem>|[silent]`), then the USER's (foe HAD an item →
        //       `-item|<user>|<yourItem>`, else `-enditem|<user>|<myItem>|[silent]`), all
        //       `|[from] move: Trick` (NO `[of]` — DISTINCT from Thief/Knock-Off).
        //     CHOICE-LOCK interaction: a Choice-Band mon that Tricks AWAY its Band loses the lock
        //       (the item's `choiceband.onDisableMove` no longer fires — probe (C): the CB user is
        //       UNLOCKED next turn); the RECEIVER of a Choice Band gets locked on its NEXT move via
        //       the existing `run_move` set-lock logic (probe (D): the receiver locks on the move it
        //       makes AFTER receiving the Band — AUTOMATIC, needs no special handling). So on a
        //       successful swap clear BOTH sides' `choice_locked_move` (mirroring
        //       `apply_item_removal`'s clear; a no-op for a non-Choice holder). `landed` FALSE (a
        //       status `moveHit` returns undefined → no in-tryMoveHit Update — probe: draws =
        //       accuracy + endTurn Quick Claw only). ---
        if move_id == "trick" {
            // ACCURACY (acc 100, not never-miss → ONE draw; always passes barring an evasion boost).
            let acc_hit = if never_miss {
                true
            } else {
                self.roll_accuracy(_side, _slot, foe, foe_slot, accuracy, never_miss, move_type, dex)
            };
            if !acc_hit {
                // A genuine miss (evasion): `[miss]` retro-edit + `-miss` (the standalone path).
                if self.logging() {
                    self.log.attr_last_move_miss();
                    let user = self.mon_ref(_side, _slot, dex);
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.miss(&user, Some(&target));
                }
                return MoveResolution::done(false, false, false);
            }
            // (TryHit) PROTECT blocks Trick (`protect: 1` flag) — `-activate|<foe>|Protect`.
            if self.protect_blocks(foe, foe_slot, false) {
                if self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.activate(&target, "Protect", None);
                }
                return MoveResolution::done(false, false, false);
            }
            // (onTryImmunity) STICKY HOLD on the target → PLAIN `-immune|<foe>` (NO `[from]
            // ability`), no swap. (Probe: `|move|…|Trick|p2a: Muk` then `|-immune|p2a: Muk`.)
            if to_id(&self.sides[foe].pokemon[foe_slot].ability) == "stickyhold" {
                if self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.immune(&target);
                }
                return MoveResolution::done(false, false, false);
            }
            // The takeItem-false gate (either side Knocked-Off), the Substitute block (no
            // `bypasssub`), and the both-itemless case all FAIL the same: `[still]` (retro-edit)
            // + `-fail|<user>`, no swap. DRAW-FREE past the accuracy roll.
            let user_item = self.sides[_side].pokemon[_slot].item.clone();
            let foe_item = self.sides[foe].pokemon[foe_slot].item.clone();
            let knocked = self.sides[_side].pokemon[_slot].item_knocked_off
                || self.sides[foe].pokemon[foe_slot].item_knocked_off;
            let subbed = self.sides[foe].pokemon[foe_slot].substitute.is_some();
            let both_itemless = user_item.is_empty() && foe_item.is_empty();
            if subbed || knocked || both_itemless {
                if self.logging() {
                    self.log.attr_last_move_still();
                    let user = self.mon_ref(_side, _slot, dex);
                    self.log.fail(&user, None, false);
                }
                return MoveResolution::done(false, false, false);
            }
            // SWAP: at least one side has an item, neither knocked-off, no sub. Resolve display
            // names BEFORE mutating.
            let name_of = |id: &str| -> String {
                if id.is_empty() {
                    return String::new();
                }
                dex.item(&to_id(id))
                    .map(|i| i.name.clone())
                    .unwrap_or_else(|| id.to_string())
            };
            let my_name = name_of(&user_item); // the USER's OLD item's display name
            let your_name = name_of(&foe_item); // the TARGET's OLD item's display name
            let user_had = !user_item.is_empty();
            let foe_had = !foe_item.is_empty();
            // The swap: the user GAINS the foe's item, the foe GAINS the user's item.
            self.sides[_side].pokemon[_slot].item = foe_item.clone();
            self.sides[foe].pokemon[foe_slot].item = user_item.clone();
            // Release BOTH Choice locks (the lock-enforcing item changed on each side). A no-op
            // for a non-Choice holder; the receiver re-acquires its lock on its NEXT move.
            self.sides[_side].pokemon[_slot].choice_locked_move = None;
            self.sides[foe].pokemon[foe_slot].choice_locked_move = None;
            if self.logging() {
                let user = self.mon_ref(_side, _slot, dex);
                let target = self.mon_ref(foe, foe_slot, dex);
                self.log
                    .activate(&user, "move: Trick", Some(&format!("[of] {target}")));
                // TARGET's new-item line FIRST (the sim's `if (myItem)` branch).
                if user_had {
                    self.log.item_from_move(&target, &my_name, "Trick");
                } else {
                    self.log.enditem_silent_from_move(&target, &your_name, "Trick");
                }
                // USER's new-item line (the sim's `if (yourItem)` branch).
                if foe_had {
                    self.log.item_from_move(&user, &your_name, "Trick");
                } else {
                    self.log.enditem_silent_from_move(&user, &my_name, "Trick");
                }
            }
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

        // --- The gen-3 `tryMoveHit` ORDER (mods/gen3/scripts.ts, the omniscient oracle):
        //     `naturalImmunity` is COMPUTED before accuracy but the `-immune` line is emitted
        //     LATER, and only AFTER `runEvent('TryHit')`. So on a HIT (accPass) the TryHit
        //     handlers — **Protect** (`-activate Protect`) + **Soundproof** (`-immune|[from]
        //     ability: Soundproof`) — win BEFORE the naturalImmunity `-immune`
        //     (`gen3_omniscient_byte_fuzz_v1` FORM 11: Thunder Wave into a Ground-typed
        //     Protecting foe shows `-activate Protect`, NOT `-immune`). On an accuracy MISS
        //     the TryHit event never runs → naturalImmunity still wins (`-immune`, NO `-miss`),
        //     else `[miss]` + `-miss`. All reads are DRAW-FREE (the accuracy roll already
        //     drew) — this is an emission reorder, observation-only. ---
        if acc_hit {
            // (TryHit) PROTECT — a Protecting foe blocks the status move at `runEvent('TryHit')`.
            if self.protect_blocks(foe, foe_slot, false) {
                // [EMIT] `|-activate|<protector>|Protect` (the `|move|` announce already showed).
                if self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.activate(&target, "Protect", None);
                }
                return MoveResolution::done(false, false, false);
            }
            // (TryHit) SOUNDPROOF — a SOUND status move (Sing / Grass Whistle) into a
            // Soundproof holder is IMMUNE (`gen3_ability_batch2_v1`, `soundproof.onTryHit`).
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
            // Type-immune (Thunder Wave → Ground, Glare → Ghost) — after TryHit passes.
            if natural_immunity {
                // [EMIT] `|-immune|<target>`.
                if self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.immune(&target);
                }
                return MoveResolution::done(false, false, false);
            }
        } else {
            // Accuracy MISS: naturalImmunity STILL wins (immune, NO `-miss`); else `-miss`.
            if natural_immunity {
                // [EMIT] `|-immune|<target>`.
                if self.logging() {
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.immune(&target);
                }
                return MoveResolution::done(false, false, false);
            }
            // [EMIT] the `[miss]` retro-edit on the announce (`attrLastMove('[miss]')`) then
            // `|-miss|<user>|<target>` (Will-O-Wisp 75, Hypnosis 60, …). Byte-verified vs the
            // status_immune_lines capture (Phase 3). (The pre-Phase-3 claim that "a status-move
            // `|move|` never carries `[miss]`" was a corpus artifact the capture DISPROVED.)
            if self.logging() {
                self.log.attr_last_move_miss();
                let user = self.mon_ref(_side, _slot, dex);
                let target = self.mon_ref(foe, foe_slot, dex);
                self.log.miss(&user, Some(&target));
            }
            return MoveResolution::done(true, false, false);
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
            // [EMIT] the did-nothing `[still]` announce form + a bare `|-fail|<user>`
            // (`gen3_omniscient_byte_fuzz_v1` FORMS 1+2 Toxic-vs-Sub): the sim's
            // `substitute.onTryPrimaryHit` sees `getDamage` undefined for a status move →
            // `this.add('-fail', source)` + `this.attrLastMove('[still]')`. NO sub-tag.
            if self.logging() {
                self.log.attr_last_move_still();
                let user = self.mon_ref(_side, _slot, dex);
                self.log.fail(&user, None, false);
            }
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
        // `src_move=Some(move_name)`: a landed SLEEP carries `[from] move: <Name>` (FORM 5).
        self.try_set_status_impl(foe, foe_slot, status, Some((_side, _slot)), true, None, Some(move_name), dex); // ability_reveal=None

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
        // (0) ALREADY-ASLEEP GUARD (`gen3_move_coverage_batch5_v1`): a Rest by an
        //     ALREADY-asleep user SILENTLY no-ops — NO `-fail` line, NO heal, NO counter
        //     reset, ZERO draws (probed via Sleep Talk, the ONLY path that can execute a
        //     move while asleep: a Sleep-Talk-picked Rest emits just the two `|move|`
        //     lines and nothing else). Checked BEFORE the full-HP guard (a full-HP
        //     asleep RestTalker must NOT emit the `-fail|heal` form).
        if matches!(mon.status, Some(Status::Sleep(_))) {
            return MoveResolution::done(false, false, false);
        }
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

        // (2c) SLEEP-IMMUNE ABILITY GUARD (`gen3_rest_sleep_immune_v1`) — gen3
        //      `setStatus('slp')` is BLOCKED by the user's OWN sleep-immunity ability
        //      (INSOMNIA / VITAL SPIRIT, `AbilityData.status_immune.blocks("slp")`) at
        //      `runEvent('SetStatus')`, BEFORE `singleEvent('Start', slp)` — so the sleep
        //      `random(2,6)` is NEVER drawn and Rest FAILS (`onHit` returns false: no sleep,
        //      no heal). The port used to draw the `random(2,6)` + sleep + heal
        //      unconditionally → +1 draw + wrong state (the A/B repro rmry3vbgm_ab_1_1
        //      seed@15 = a damaged Insomnia Hypno Resting). It sits AFTER the gen3ou clause
        //      shuffle above (the ability sorts into its own speed group → the 2 clauses stay
        //      a size-2 tie → the shuffle STILL draws — `gen3_status_immune_v1`) and emits the
        //      sim's `|-fail|<user>|[from] ability: <A>|[of] <user>` (probe-verified,
        //      `harness/probe_rest_sleep_immune_rng.js`).
        {
            let ability_id = to_id(&self.sides[side].pokemon[slot].ability);
            let blocks_slp = dex
                .ability(&ability_id)
                .and_then(|a| a.status_immune.as_ref())
                .map(|si| si.blocks("slp"))
                .unwrap_or(false);
            if blocks_slp {
                if self.logging() {
                    let user = self.mon_ref(side, slot, dex);
                    let ability_name =
                        dex.ability(&ability_id).map(|a| a.name.clone()).unwrap_or(ability_id);
                    self.log
                        .push_raw(format!("|-fail|{user}|[from] ability: {ability_name}|[of] {user}"));
                }
                return MoveResolution::done(false, false, false);
            }
        }

        // (3)+(4) SELF-SLEEP + status CURE (setStatus overrides any prior major status).
        //     `setStatus('slp')` runs `slp.onStart`, which DRAWS `random(2,6)` for the
        //     duration — so the DRAW HAPPENS (we must consume it for seed parity) — and
        //     Rest's `onHit` then OVERWRITES the time to a FIXED 3. So: draw + DISCARD the
        //     value, then store a FIXED Sleep(3). The counter is decremented draw-free by
        //     the existing on_before_move handler (3 attempts → wake).
        let _discarded = self.prng.random_range(2, 6); // slp.onStart's random(2,6) — value discarded
        self.sides[side].pokemon[slot].status = Some(Status::Sleep(3));
        // SELF-inflicted (Rest) → EXEMPT from the opponent's Sleep Clause Mod
        // (`statusState.source.isAlly(target)`, `gen3_sleep_clause_self_rest_exempt_v1`):
        // a self-Rested mon must NOT count as the side's one foe-asleep, so a foe sleep
        // move on a DIFFERENT mon still lands.
        self.sides[side].pokemon[slot].sleep_from_rest = true;
        // A fresh slp statusState → skippedTime 0 (`gen3_move_coverage_batch5_v1`).
        self.sides[side].pokemon[slot].sleep_skipped = 0;

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
    /// FAIL-LOUD: Protect / Detect / **Endure** (`gen3_move_coverage_batch6_v1` — the
    /// same stallingMove machinery, SHARED stall counter, but the `endure` volatile's
    /// survive-at-1-HP `onDamage` clamp instead of the move block) are modeled. Any
    /// OTHER `isProtect` move (the gen4+ Quick Guard / Wide Guard / King's Shield /
    /// Spiky Shield / … which gen3 doesn't have) PANICS so a future protection
    /// variant can never silently desync.
    pub(crate) fn run_protect(
        &mut self,
        side: usize,
        slot: usize,
        move_id: &str,
        move_name: &str,
        will_act: bool,
        dex: &Dex,
    ) -> MoveResolution {
        if move_id != "protect" && move_id != "detect" && move_id != "endure" {
            panic!(
                "protect-class move {move_id:?} is not modeled — only Protect / Detect \
                 (identical full-turn protection) + Endure (survive-at-1-HP, \
                 `gen3_move_coverage_batch6_v1`) are. Quick Guard / Wide Guard / King's \
                 Shield / etc. are DEFERRED (and gen3 has none of them). Model it (or \
                 exclude it from isModeledMove) before a battle can use it."
            );
        }
        // ENDURE (`gen3_move_coverage_batch6_v1`) rides the IDENTICAL stallingMove
        // machinery (probe ED1-ED9): the same `willAct()` gate, the same SHARED `stall`
        // counter/ladder (an endure escalates a prior Protect's denominator and vice
        // versa — ED3/ED4 byte-identical both orders), the same no-delete-on-fail
        // persistence. Only the SUCCESS effect differs: the `endure` volatile (the
        // onDamage survive-at-1 clamp + a NO_ORDER residual duration handler) instead
        // of the `protected` move-block, and a `|-singleturn|<user>|move: Endure` line.
        // gen3 Endure is priority **4** (one ABOVE Protect's 3 — the dex feeds
        // `sort_actions`; not this fn's concern).
        let is_endure = move_id == "endure";

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
            // (`|move|<user>|Protect||[still]`) + a bare `|-fail|<user>`, with NO
            // `-singleturn`. In the sim `onPrepareHit` returns `false` (the `willAct()`
            // short-circuit), so `tryMoveHit`'s PrepareHit check emits `this.add('-fail',
            // pokemon)` + `attrLastMove('[still]')` — the SAME `[still]`+`-fail` form as a
            // failed stall roll (`gen3_omniscient_byte_fuzz_v1` FORM 2, byte-fuzzer-surfaced:
            // a Protect that resolves after the foe SWITCHED emits `-fail`; the port omitted
            // it). Draw-free (no stall roll).
            if self.logging() {
                let user = self.mon_ref(side, slot, dex);
                self.log.move_used(&user, move_name, None, false, true);
                self.log.fail(&user, None, false);
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
                // Endure's onStart announces `move: Endure`; Protect/Detect's bare `Protect`.
                self.log.singleturn(&user, if is_endure { "move: Endure" } else { "Protect" });
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
            //     ENDURE sets its OWN volatile instead of the move block (the onDamage
            //     survive-at-1 clamp reads `endure`; `protect_blocks` never reads it).
            if is_endure {
                mon.endure = true;
            } else {
                mon.protected = true;
            }
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
    pub(crate) fn protect_blocks(&self, foe: usize, foe_slot: usize, targets_self: bool) -> bool {
        !targets_self && self.sides[foe].pokemon[foe_slot].protected
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
}
