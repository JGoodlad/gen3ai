use crate::dex::{to_id, Dex};
use crate::protocol::ProtocolLine;
use crate::state::{BattleState, Status};
use super::*;
use super::helpers::*;

impl crate::state::BattleState {

    /// Play a full scripted battle to game-end WITH protocol emission (level-2, Phase 1):
    /// enable the log, emit the battle-init framing (+ `|turn|1`), run [`run_full_battle`]
    /// (which emits the per-turn/`|move|`/`|-damage|`/`|switch|`/`|faint|`/… lines at its
    /// hook points), and return the `(BattleOutcome, Vec<ProtocolLine>)` — the outcome plus
    /// the accumulated OMNISCIENT stream.
    ///
    /// **Observation-only:** this is the SAME engine path as `run_full_battle`; the ONLY
    /// difference is `self.log` is enabled, and every emit hook is a PRNG-free read of
    /// already-computed state. So the seed sequence (hence every seed assertion in the
    /// existing suite) is UNCHANGED — the byte layer rides on top of the bit-for-bit engine.
    ///
    /// The battle must be constructed via [`crate::state::BattleState::start_with_switchins`]
    /// (the framing emits the leads' `|switch|`, matching the sim's `>start` sequence).
    pub fn run_full_battle_logged(
        &mut self,
        script: &[ScriptDecision],
        dex: &Dex,
    ) -> (BattleOutcome, Vec<ProtocolLine>) {
        self.log.enable();
        self.emit_framing(dex);
        let outcome = self.run_full_battle(script, dex);
        (outcome, self.log.drain())
    }
    /// Play a FULL scripted battle to game-end (or until the `script` is
    /// exhausted), handling voluntary switches, post-faint replacements (single
    /// AND double), and win/loss — consuming the PRNG in Showdown's EXACT order +
    /// count across every turn AND replacement sub-step.
    ///
    /// # SWITCHING — the verified draw model
    ///
    /// The driver builds the turn's action queue and runs it through a turnLoop
    /// that can PAUSE for a forced replacement and RESUME the saved tail (mirroring
    /// `commitChoices`/`turnLoop`). The draw sites, all the existing
    /// [`speed_sort`]/`random_range`:
    /// - **action-order shuffle** ([`speed_sort`] over the turn's actions, in
    ///   `commitChoices`): a two-switch / switch-vs-move ordering. Switches (order
    ///   103) sort before moves (200) by `order`; two same-kind switches tie on
    ///   order+priority → the speed-tie shuffle draws.
    /// - **per-action trailing `eachEvent('Update')`** ([`each_event_shuffle`]) at
    ///   the END of every runAction — UNLESS the runAction paused for a switch
    ///   request (`makeRequest('switch'); return`) or its next queued action is an
    ///   `instaswitch` (`battle.ts:2372`); both SKIP the trailing Update.
    /// - **gen3 runSwitch is DRAW-FREE** (the gen4 override: no `speedSort(allActive)`,
    ///   no `fieldEvent('SwitchIn')`) — only the entrant's ability `Start` fires
    ///   (draw-free for our abilities). Do NOT add a SwitchIn tie-shuffle.
    /// - **double-replacement `insertChoice` splice** (`battle-queue.ts:283`): when
    ///   the 2nd instaswitch's `switchIn` enqueues its `runSwitch` into a queue
    ///   already holding the 1st instaswitch's `runSwitch` (same order 101), an
    ///   order-tie window draws ONE `random(firstIndex, lastIndex+1)`. A SINGLE
    ///   replacement's runSwitch inserts with no tie window → no draw.
    /// - **Quick Claw** `randomChance(1,5)` at `endTurn` — UNLESS the turn ended on
    ///   a faint pause OR a game-ending faint (no trailing residual/Quick Claw on
    ///   the deciding turn).
    ///
    /// # Win / loss
    ///
    /// `checkWin` runs inside the faint protocol: a side with `pokemon_left == 0`
    /// loses, its foe wins; both 0 → a gen-3 TIE (`win(None)`). The deciding faint
    /// draws no trailing Quick Claw.
    ///
    /// # Script contract
    ///
    /// One [`ScriptDecision`] per REQUEST boundary, in order: a `move` request
    /// consumes the next decision's both-side choices; a forced-`switch` request
    /// consumes the next decision's flagged-side choice(s). The driver stops when
    /// the battle ends OR the script is exhausted (a missing required choice ends
    /// the run, recorded but not asserted as game-end).
    pub fn run_full_battle(&mut self, script: &[ScriptDecision], dex: &Dex) -> BattleOutcome {
        // Thin wrapper over the resumable stepping primitive [`FullBattleDriver`] — the
        // SINGLE turn-loop that the streaming surface (`BattleStream::write_line`) and the
        // per-side bridge (`bridge::BridgeSession`) also drive. Feeding the whole script in
        // one loop reproduces the previous monolithic driver BIT-FOR-BIT (the batch seed
        // suites + the e2e capstone are the byte oracle for this primitive), while the SAME
        // primitive advances a LIVE battle ONE request boundary per fed decision for the
        // O(1)-per-input streaming/bridge paths (no genesis replay).
        let mut driver = FullBattleDriver::new();
        let mut it = script.iter();
        while !driver.is_ended() {
            let dec = match it.next() {
                Some(d) => *d,
                None => break,
            };
            driver.feed(self, dec, dex);
        }
        driver.into_outcome()
    }

    /// Whether ONE side's top-of-turn `move`-request choice would be ACCEPTED by the
    /// sim's PER-SIDE `side.choose` (Phase 3 refactor of the old whole-decision
    /// `move_decision_is_legal` — same gates, per-side granularity): a `Move(K)` must
    /// name a slot the CURRENT active mon has ("Your <mon> doesn't have a move K") that
    /// is usable-or-Struggle-substituted; a VOLUNTARY `Switch(t)` must name a live,
    /// non-active bench slot and the active must not be TRAPPED (`gen3_trapping_v1`).
    /// A rejected choice draws nothing and leaves the side's half of the boundary open —
    /// see the per-side pending-choice accumulator at the top of [`run_full_battle`].
    fn choice_is_legal(&self, side: usize, choice: Choice, dex: &Dex) -> bool {
        {
            match Some(choice) {
                Some(Choice::Move(mi)) => {
                    let active = self.sides[side].active;
                    let mon = &self.sides[side].pokemon[active];
                    // (0) MOVE-LOCKED request (`gen3_move_coverage_batch4c_v1` — a
                    // MUSTRECHARGE / CHARGING mon): the request offers a SINGLE pseudo/
                    // locked move entry, so ONLY `move 1` (Move(0)) is accepted; a
                    // `move 2` is rejected ("Your <mon> doesn't have a move 2" — probed)
                    // and the PP/usable gates below (which read the REAL moveset) do not
                    // apply to the single-entry request.
                    if mon.move_locked() {
                        return mi == 0;
                    }
                    // (1) OUT-OF-RANGE slot (the forced-replacement resume phantom): a
                    // `Move(K)` whose slot the current active mon doesn't have.
                    if mi >= mon.set.moves.len() {
                        return false;
                    }
                    // (2) UN-USABLE slot WHILE another move is usable (`gen3_pp_tracking_v1` +
                    // `gen3_taunt_disable_v1`): the sim's `side.choose` REJECTS a `move K` that is
                    // out of PP, DISABLED, or TAUNTED ("(move) is disabled") when the mon still has
                    // a usable move — draw-free, boundary stays open (VERIFIED vs the sim: a stale
                    // `move 1` on a 0-PP Earthquake while Crunch has PP drew 0 and did not advance;
                    // the same reject applies to a taunted Status move or a disabled move). This
                    // mirrors the out-of-range case. EXCEPTION: when the mon has NO usable move at
                    // all (`must_struggle`), the `move K` is NOT rejected — `side.choose`
                    // SUBSTITUTES `moveid:'struggle'` instead (handled at queue-build time), so the
                    // decision commits and runs Struggle. So reject ONLY when this specific slot is
                    // un-usable AND a different usable move exists. `move_usable` now folds in the
                    // Disable/Taunt selection restriction (dex read for Taunt's per-slot category).
                    if !mon.move_usable(mi, dex) && !mon.must_struggle(dex) {
                        return false;
                    }
                }
                Some(Choice::Switch(t)) => {
                    // (2a) MOVE-LOCKED request (`gen3_move_coverage_batch4c_v1`): a
                    // MUSTRECHARGE / CHARGING mon's request is `trapped:true` — a
                    // voluntary switch is REJECTED draw-free ("[Invalid choice] Can't
                    // switch: The active Pokémon is trapped", probed — the FIRM-trap
                    // shape, no maybeTrapped phase).
                    if self.sides[side].pokemon[self.sides[side].active].move_locked() {
                        return false;
                    }
                    // (2b) INVALID TARGET (Phase 3 — the capture plans now submit blind
                    // `switch N` tokens, so a scripted switch to a FAINTED / out-of-range /
                    // already-active slot must be SKIPPED exactly like the sim's
                    // `chooseSwitch` reject ("can't switch to a fainted Pokémon" — draw-free,
                    // boundary stays open; the capture records the rejected decision with an
                    // UNCHANGED seedAfter). The older harness plans always computed a live
                    // bench target, so this gate never fired for the pre-Phase-3 goldens.
                    let s = &self.sides[side];
                    if t >= s.pokemon.len() || s.pokemon[t].fainted || t == s.active {
                        return false;
                    }
                    // (3) TRAPPED (`gen3_trapping_v1` — the SWITCH-legality gate, mirroring
                    // the move gate above): the sim's `chooseSwitch` at a `move` request
                    // REJECTS a voluntary switch by a trapped mon ("Can't switch: The active
                    // Pokémon is trapped") — DRAW-FREE, boundary stays open (VERIFIED vs the
                    // sim `harness/probe_trapping_rng.js`: the rejected `switch 2` left the
                    // seed byte-identical + the request open). This gates ONLY the
                    // top-of-turn `move`-request path — a FORCED replacement (`requestState
                    // === 'switch'`, the `NeedSwitch` branch) never consults `trapped`, and
                    // a PHAZE drag (`drag_in`) moves a trapped mon regardless. (The target's
                    // live-bench validity is enforced elsewhere, as before.)
                    if self.is_trapped(side, dex) {
                        return false;
                    }
                }
                None => {}
            }
        }
        true
    }

    /// Run queued actions until the queue drains, a faint requests a replacement,
    /// or the battle ends — mirroring `turnLoop` + the per-`runAction` tail
    /// (faintMessages → checkWin → checkFainted → the switch-request gate → the
    /// gen<5 trailing `eachEvent('Update')`).
    ///
    /// On `NeedSwitch`/`Ended` it RETURNS with the queue's remaining tail intact
    /// (the saved `oldQueue` the caller resumes after committing replacements).
    fn turn_loop(&mut self, queue: &mut Vec<QAction>, dex: &Dex) -> TurnLoopStop {
        // Runaway guard (`gen3_bridge_turn_watchdog_v1`): a re-enqueuing cycle would spin here
        // forever. Cap far above any legit turn and PANIC with the queue so the cycling action
        // is self-evident. See TURN_LOOP_ACTION_CAP.
        let mut guard = 0usize;
        while !queue.is_empty() {
            guard += 1;
            if guard > TURN_LOOP_ACTION_CAP {
                let head: Vec<&QAction> = queue.iter().take(6).collect();
                panic!(
                    "turn_loop runaway: >{} actions in ONE turn (turn={}, queue_len={}, head={:?}) \
                     — a queued action is re-enqueuing in a cycle",
                    TURN_LOOP_ACTION_CAP, self.turn, queue.len(), head,
                );
            }
            let action = queue.remove(0);

            // --- Run the action body. A move/residual may faint a mon (HP zeroed,
            //     `fainted` set by process_faints which we call as faintMessages). ---
            match action {
                QAction::BeforeTurn => {
                    self.each_event_shuffle(); // eachEvent('BeforeTurn')
                }
                QAction::BeforeTurnMove { side, uid, move_index } => {
                    // `case 'beforeTurnMove'` (battle.ts:2265): `if (!isActive) return false;
                    // if (fainted) return false;` then run the move's beforeTurnCallback. A
                    // `return false` skips the runAction tail (no trailing Update). DRAW-FREE.
                    let slot = match self.slot_of_uid(side, uid) {
                        Some(s) if self.sides[side].active == s && !self.sides[side].pokemon[s].fainted => s,
                        _ => continue, // inactive/fainted → return false, no tail
                    };
                    let move_id = match self.move_at(side, slot, move_index, dex) {
                        Some(m) => to_id(&m.id),
                        None => continue,
                    };
                    if move_id == "focuspunch" {
                        // focuspunch.beforeTurnCallback: addVolatile('focuspunch') on the USER.
                        // The volatile's onStart emits `|-singleturn|<user>|move: Focus Punch`.
                        self.sides[side].pokemon[slot].focus_punch = Some(false);
                        if self.logging() {
                            let user = self.mon_ref(side, slot, dex);
                            self.log.singleturn(&user, "move: Focus Punch");
                        }
                    } else if move_id == "counter" || move_id == "mirrorcoat" {
                        // counter/mirrorcoat.beforeTurnCallback (`gen3_move_coverage_batch5_v1`):
                        // `pokemon.addVolatile('counter'|'mirrorcoat')` on the USER — DRAW-FREE,
                        // no protocol line (the conditions have no announcing onStart). The
                        // onStart RESETS `{slot: null, damage: 0}` EVERY selection turn (so
                        // prev-turn damage never counts — probed C1 t2); the un-armed state is
                        // `damage: None`. The recorder (`record_reactive_hit`) then arms it on a
                        // qualifying foe hit this turn.
                        self.sides[side].pokemon[slot].reactive = Some(crate::state::Reactive {
                            mirror: move_id == "mirrorcoat",
                            damage: None,
                        });
                    } else if move_id == "pursuit" {
                        // pursuit.beforeTurnCallback: `if (frz|slp) return; if (isAlly) return;
                        // target.addVolatile('pursuit'); sources.push(pokemon)`. Lay the volatile
                        // on the FOE (the pursuer's target), recording the pursuer's uid — UNLESS
                        // the pursuer (this actor) is frozen or asleep (then NO volatile → no
                        // interrupt). No `-singleturn`/`-start` line (the pursuit condition has
                        // no onStart). Draw-free.
                        let st = self.sides[side].pokemon[slot].status;
                        let skip = matches!(st, Some(Status::Freeze) | Some(Status::Sleep(_)));
                        if !skip {
                            let foe = 1 - side;
                            let foe_slot = self.sides[foe].active;
                            let pursuer_uid = self.sides[side].pokemon[slot].uid;
                            self.sides[foe].pokemon[foe_slot].pursuit = Some(pursuer_uid);
                        }
                    }
                }
                QAction::Move { side, uid, move_index, struggle } => {
                    // The `case 'move'` guard (battle.ts:2239-2240): a move whose actor
                    // is no longer the active (KO'd / switched out) `return false`s from
                    // runAction IMMEDIATELY — drawing NOTHING and running NO TAIL (no
                    // faintMessages, no trailing Update). So we `continue` the loop,
                    // skipping the entire tail. (This is the just-fainted mon's
                    // surviving-but-no-op queued move; its presence in the queue does
                    // NOT add a trailing-Update draw — verified vs the sim's
                    // double-replacement trace, where `move p2:Electrode` drew nothing
                    // and got no trailing Update.) The actor is resolved by its STABLE
                    // uid (its array slot may have swapped).
                    let live = self
                        .slot_of_uid(side, uid)
                        .map(|slot| !self.sides[side].pokemon[slot].fainted && self.sides[side].active == slot)
                        .unwrap_or(false);
                    if !live {
                        continue; // runAction returned false → no tail, no draws
                    }
                    let slot = self.slot_of_uid(side, uid).unwrap();
                    // --- ENCORE onOverrideAction (`gen3_move_coverage_batch6_v1`, probe
                    //     EN7): a QUEUED move that differs from the encored slot is
                    //     OVERRIDDEN at EXECUTION to the encored move — the ENCORED
                    //     slot's PP deducts, the announce shows the encored move, and
                    //     the draw count is a normal move turn's. This only fires when
                    //     the encore LANDED THIS TURN (a faster encore user; the
                    //     target's queued move was chosen pre-lock) — at a request
                    //     boundary `move_usable` already restricts selection to the
                    //     encored slot. The action keeps its QUEUED sort position (the
                    //     sim sorted by the queued move's priority; the override fires
                    //     inside runAction). DRAW-FREE. Struggle / a recharge-locked
                    //     turn are exempt (no slot / the priority-11 cant precedes). ---
                    let move_index = if !struggle {
                        match self.sides[side].pokemon[slot].encore {
                            Some((eslot, _)) if eslot != move_index => eslot,
                            _ => move_index,
                        }
                    } else {
                        move_index
                    };
                    // `willAct()` (battle-queue.ts:310) — does ANY move/switch/instaswitch/
                    // shift action REMAIN in the queue? Protect/Detect's `onPrepareHit`
                    // requires it (`!!this.queue.willAct() && runEvent('StallMove')`): a
                    // Protect that resolves with NO pending foe action (the foe SWITCHED — a
                    // switch is order 103 < the protect's move order 200, so it already ran)
                    // FAILS (draw-free, no volatile). The residual (the only other queued
                    // action here) is NOT a move/switch, so this is true iff a foe move/
                    // switch is still pending. VERIFIED vs the sim: Protect vs a foe SWITCH
                    // leaves the protector with NO volatiles (`|move|…Protect||[still]`).
                    let will_act = queue.iter().any(|a| {
                        matches!(
                            a,
                            QAction::Move { .. } | QAction::Switch { .. } | QAction::InstaSwitch { .. }
                        )
                    });
                    // `willMove(target)` for Disable's duration `+1` (`gen3_taunt_disable_v1`):
                    // does the FOE (the disable target) still have a PENDING move/switch action in
                    // the queue? False iff the foe has already acted this turn (disabler moved 2nd)
                    // → Disable's onStart does `duration++`.
                    let foe = 1 - side;
                    let foe_will_move = queue.iter().any(|a| match a {
                        QAction::Move { side: s, .. }
                        | QAction::Switch { side: s, .. }
                        | QAction::InstaSwitch { side: s, .. } => *s == foe,
                        _ => false,
                    });
                    let res = self.run_move(MoveAction { side, slot, move_index, struggle }, will_act, foe_will_move, dex);
                    if res.landed {
                        let upd = self.each_event_shuffle(); // in-tryMoveHit Update (landed)
                        self.run_update_items(&upd, dex); // cure-berry/Leppa onUpdate (draw-free)
                    }
                    // --- CHARGE consumption (`gen3_move_coverage_batch6_v1` —
                    //     `charge.onAfterMove` + `onMoveAborted`, both `if (move.id !==
                    //     'charge') removeVolatile('charge')`): the volatile lasts until
                    //     the user's NEXT move attempt OF ANY KIND — an idle Splash
                    //     consumes it, a cant (onMoveAborted — full-para / slp / recharge
                    //     / …) consumes it, only an Electric move first gets the ×2 BP
                    //     fold (already applied inside run_move). The AfterMove event
                    //     fires from `runMove` with the OUTER (queued, post-encore-
                    //     override) move — so a Sleep-Talk turn consumes it keyed on
                    //     `sleeptalk`, not the called move; the pursuit-interrupt strike
                    //     is a bare `useMove` (NO runMove → NO AfterMove → NOT consumed
                    //     there, faithful to the source). DRAW-FREE; the removal emits
                    //     `|-end|<user>|Charge|[silent]` (`charge.onEnd`).
                    //     A 0-HP user (a SELF-DESTRUCT / Explosion self-KO'd charger) does NOT
                    //     consume/emit it: the sim's `Pokemon.removeVolatile` returns false for
                    //     `!this.hp`, so `charge.onAfterMove`'s `removeVolatile('charge')` is a
                    //     no-op → NO `|-end|Charge` line (the faint's later silent clearVolatile
                    //     drops it). The port used to emit a spurious `|-end|Charge|[silent]`
                    //     BEFORE the `|faint|` (`gen3_charge_selfko_no_end_v1`, random-mode byte
                    //     fuzz find ab_12_17 @ master-seed 200724: a Charge-holding Electrode
                    //     Self-Destructs; sim-probe-confirmed via `harness/probe_charge_selfko.js`).
                    //     Gate on hp > 0 (the `removeVolatile` false-on-0-HP rule). ---
                    if self.sides[side].pokemon[slot].charge
                        && self.sides[side].pokemon[slot].hp > 0
                    {
                        let executed_is_charge = !struggle
                            && self
                                .move_at(side, slot, move_index, dex)
                                .map(|m| m.id == "charge")
                                .unwrap_or(false);
                        if !executed_is_charge {
                            self.sides[side].pokemon[slot].charge = false;
                            if self.logging() {
                                let user = self.mon_ref(side, slot, dex);
                                self.log.volatile_end_silent(&user, "Charge");
                            }
                        }
                    }
                    // --- WHITE HERB `onAnyAfterMove` (`gen3_white_herb_v1`): at the AfterMove
                    //     event (runMove's tail, AFTER the move body / in-tryMoveHit Update,
                    //     BEFORE the runAction-tail forceSwitch), a White Herb holder — EITHER
                    //     active, `onAnyAfterMove` — with any negative boost restores it + consumes
                    //     the item, DRAW-FREE. This is the sim's ONLY in-turn White-Herb hook and
                    //     therefore the SINGLE in-turn restore site: the self-drop (Overheat /
                    //     Superpower, `apply_self_drops`) + foe-secondary/stat-drop (Crunch /
                    //     Charm, `apply_secondary_boost`) paths deliberately do NOT restore inline
                    //     — the sim runs `AfterMove` at the END of `runMove`, i.e. AFTER the
                    //     DamagingHit-phase procs (Poison Point / Rough Skin), so an inline restore
                    //     emitted `|-enditem|…|White Herb` one phase too early (SIM-PROBED,
                    //     `harness/probe_rb_tail.js` C2a/C2b). It IS the site that restores an
                    //     Intimidate-switch-in drop AFTER the holder's next move (ab_4_6): the
                    //     switch-in `onAnySwitchIn` fires before the entrant's Intimidate Start, so
                    //     the drop is only visible here. Mover's side first (the common holder).
                    //     SKIP on a CANT (`res.aborted`): the sim's `AfterMove` NEVER fires for an
                    //     `onBeforeMove`-aborted move (it runs `MoveAborted` + returns before it), so
                    //     a cant'd holder's Intimidate drop waits for the end-of-turn `onResidual`
                    //     (order 29), NOT this tail (`gen3_white_herb_v1`, golden ab_154_12: a slp
                    //     Hoppip's White Herb fires at the residual `|` boundary, not after the cant).
                    if !res.aborted {
                        let wh_foe = 1 - side;
                        self.white_herb_restore(side, self.sides[side].active, dex);
                        self.white_herb_restore(wh_foe, self.sides[wh_foe].active, dex);
                    }
                    // --- CHOICE LOCK (`gen3_choicelock_after_move_v1`,
                    //     `choiceband.onAfterMove(pokemon) { pokemon.addVolatile('choicelock'); }`):
                    //     the lock is a VOLATILE added at the AfterMove event — the END of runMove,
                    //     AFTER the move body — gated on the user HOLDING a Choice item AT THAT
                    //     MOMENT. The volatile's `onStart` keys on `pokemon.lastMove.id`, so it
                    //     records the EXECUTED move (our `last_move`), which is why an encore-
                    //     overridden / Sleep-Talk turn locks to the OUTER move the port already
                    //     stores there. A Struggle leaves `last_move` None → `onStart` returns
                    //     false → no volatile (the sim's `hasMove('struggle')` release, same net).
                    //
                    //     WHY THE TIMING IS LOAD-BEARING (it is the whole bug): the port used to
                    //     set this at PP-deduct time, BEFORE the move body, so it read the item one
                    //     phase too early and disagreed with the sim on every move that changes the
                    //     user's OWN item while resolving:
                    //       * THIEF / COVET steal a Band → the sim holds it at AfterMove → LOCKED;
                    //         the port saw the pre-move itemless slot → NOT locked. Round 6
                    //         (`gen3_choice_lock_request_disabled_v1`) patched around this by
                    //         SYNTHESIZING a lock in the request fold from "holds Choice ∧ has a
                    //         lastMove" — which then OVER-locked the opposite case:
                    //       * TRICKED a Band by a SLOWER foe AFTER moving (soak3 `sbd_msb1zfxs_b97`,
                    //         Piloswine spe 157 > Kecleon 137): the mon holds a Band and has a
                    //         lastMove, but never MOVED while holding it, so the sim adds NO
                    //         volatile and leaves all four moves selectable. The port hid three
                    //         legal moves from the policy — externally visible to poke-env and
                    //         DRAW-FREE, so only the external-consistency gate could see it.
                    //     Fixing the TIMING makes both fall out, so the round-6 request-fold
                    //     workaround is deleted (`bridge.rs::move_disabled`). SIM-PROBED all four
                    //     directions by `harness/probe_choicelock_acquired_item.js`.
                    //
                    //     Gated on `!res.aborted` (AfterMove never fires for an onBeforeMove-
                    //     aborted move — the White Herb rule above) and on `hp > 0` (`addVolatile`
                    //     returns false for a 0-HP mon, the `charge.onAfterMove` precedent, so a
                    //     Self-Destruct user never locks). The pursuit-interrupt strike is a bare
                    //     `useMove` with NO runMove → NO AfterMove → it does not lock here; it
                    //     keeps its own explicit set-site in `turn/switch.rs`. ---
                    //     MIMIC SELF-OVERWRITE (`gen3_mimic_choice_lock_self_overwrite_v1`, round
                    //     23): skipped when the move just overwrote its OWN slot. The sim DOES add
                    //     the volatile here (keyed 'mimic') and then RELEASES it at the very next
                    //     `runEvent('DisableMove')`, whose `hasMove(effectState.move)` is now false
                    //     — so both engines end the turn unlocked. Not adding it is observationally
                    //     EQUIVALENT rather than merely convenient: the only way to tell the two
                    //     apart is the endTurn handler-sort tie-shuffle, which needs a SECOND
                    //     DisableMove handler on the mon at that same endTurn, and on a Mimic turn
                    //     none can exist — a Choice-item mon can only use Mimic as its FIRST move
                    //     (anything earlier locks it), so it has no `lastMove` yet and neither
                    //     Disable nor Encore can be on it, while Taunt CANTS Mimic outright so it
                    //     never resolves. The handler is necessarily alone and draws nothing either
                    //     way. SIM-PROBED (cases A + D of
                    //     `harness/probe_choicelock_mimic_release.js`); pinned by corpus fixture
                    //     `49_mimic_overwrites_choice_locked_slot.txt`, which regressed to
                    //     `kind=seed` when this relocation first landed without the guard. ---
                    if !res.aborted && self.sides[side].pokemon[slot].hp > 0 {
                        let m = &self.sides[side].pokemon[slot];
                        let holds_choice = dex
                            .item(&crate::dex::to_id(&m.item))
                            .map(|i| i.choice)
                            .unwrap_or(false);
                        if let (true, false, Some(lm)) =
                            (holds_choice, m.last_move_was_self_overwrite, m.last_move)
                        {
                            self.sides[side].pokemon[slot].choice_locked_move = Some(lm);
                        }
                    }
                    // --- forceSwitchFlag consumption (battle.ts:2348-2353): at the END of
                    //     the move's runAction (AFTER the move body / any in-tryMoveHit
                    //     Update, BEFORE faintMessages), a SUCCESSFUL phaze (Roar /
                    //     Whirlwind) DRAGS in a random foe (the `sample` draw + the forced
                    //     switch + the runSwitch enqueue). `drag_in` does exactly that. A
                    //     status move is never `landed`, so the order is: phaze accuracy
                    //     (drawn in run_move) → drag sample (here) → faintMessages → … →
                    //     the trailing Update, mirroring the source EXACTLY. ---
                    if let Some(foe) = res.force_switch_foe {
                        self.drag_in(foe, dex, &mut *queue);
                    }
                }
                QAction::Switch { side, target } => {
                    // A VOLUNTARY menu switch (order 103) — `is_voluntary = true` so the
                    // PURSUIT INTERRUPT fires (the sim's `runEvent('BeforeSwitchOut')` runs
                    // for a menu switch, dispatching pursuit's `onBeforeSwitchOut` strike).
                    self.execute_switch(side, target, false, true, dex, &mut *queue);
                }
                QAction::InstaSwitch { side, target } => {
                    // A FORCED replacement (order 3) — a Baton-Pass selfSwitch OR a faint
                    // replacement. `is_voluntary = false` so the PURSUIT INTERRUPT does NOT
                    // fire (`gen3_move_coverage_batch4_v1` fix): the sim's `BeforeSwitchOut`
                    // is SUPPRESSED for a Baton Pass (`batonpass.self.onHit` sets
                    // `source.skipBeforeSwitchOutEventFlag = true`, moves.ts:1109) and a
                    // faint replacement's outgoing corpse carries no `pursuit` volatile
                    // (clearVolatile in faintMessages). PROBE-SETTLED bit-for-bit
                    // (`harness/probe_bp_pursuit_settle.js`): a pursued Baton-Pass passer is
                    // NOT struck — it survives, passes its boosts, and the pursuer's Pursuit
                    // runs NORMALLY (bp 40) against the ENTRANT next decision.
                    self.execute_switch(side, target, false, false, dex, &mut *queue);
                }
                QAction::RunSwitch { side } => {
                    // gen3 runSwitch: EntryHazard (Spikes damage, draw-free) → ability Start.
                    // The ability Start is draw-free EXCEPT when a Sand Stream / Drizzle /
                    // Drought entrant CHANGES the weather: `field.setWeather` then fires
                    // `eachEvent('WeatherChange')` (field.ts:87) — a 2-active speedSort that
                    // draws ONE tie-shuffle iff the actives tie on cached speed. It fires
                    // INSIDE this runSwitch runAction (before the trailing Update), so do it
                    // here, ahead of the tail. A Spikes-KO here faints the entrant; the tail
                    // forces another replacement.
                    let weather_changed = self.run_switch(side, dex);
                    if weather_changed {
                        self.each_event_shuffle(); // eachEvent('WeatherChange') tie-shuffle
                    }
                }
                QAction::Residual => {
                    // [EMIT] the `|` separator that CLOSES the action block + OPENS the
                    // residual block (always emitted when a turn reaches its residual —
                    // even with no residual HP lines, matching the golden). Then the
                    // residual `|-heal|`/`|-damage|` lines emit inside `run_residuals`.
                    if self.logging() {
                        self.log.separator();
                    }
                    self.run_residuals(dex);
                    // [EMIT] `|upkeep` — the end-of-turn marker, emitted AFTER the residual
                    // lines. SKIPPED when a residual KO ended the battle (then
                    // `run_full_battle` emits `|win|`/`|tie|` instead); `check_win` detects
                    // that. The `|turn|N+1` marker is NOT emitted here — it lands at the TOP
                    // of the NEXT turn's outer-loop iteration (after any forced replacement
                    // this residual triggered), matching the sim's `makeRequest('move')`
                    // placement. A RESIDUAL faint's replacement `|switch|` must precede
                    // `|turn|N+1`; a MOVE faint's replacement (before the residual) already
                    // did — so deferring the marker to the next-turn top is correct for both.
                    if self.logging() && self.check_win().is_none() {
                        self.log.upkeep();
                    }
                }
            }

            // --- runAction tail (battle.ts:2357-2424). ---
            // faintMessages: set `fainted`, decrement pokemonLeft, then checkWin.
            let any_faint = self.process_faints(dex);
            if any_faint {
                // gen-3 singles cancelAction-all (battle.ts:2140-2148): "in gen 3,
                // fainting skips all moves and switches" — remove the REST of this
                // turn's queued move/switch actions whose actor is a NON-fainted
                // active. The just-fainted mon's own queued move is NOT cancelled
                // (it's excluded from getAllActive) but runs as a no-op (the
                // `actor.fainted` guard above). The residual survives.
                self.cancel_active_actions(queue);
            }
            if let Some(end) = self.check_win() {
                return TurnLoopStop::Ended { winner: end };
            }

            // checkFainted (gen3 replaces after EACH action): every 0-HP active gets
            // switchFlag. (Showdown gates this on `!peek || peek in [move,residual]`;
            // in our linear gen3-singles queue the equivalent is: any fainted active
            // that CAN switch needs a replacement — set the flag unconditionally and
            // let the gate below decide.)
            self.check_fainted();

            // `else if peek === 'instaswitch' return false` (battle.ts:2372): if the
            // NEXT queued action is an instaswitch, this runAction returns IMMEDIATELY
            // — BEFORE the switch-request gate AND before the trailing Update. This is
            // the double-replacement resume: the FIRST instaswitch's tail does NOT
            // re-request a switch (the SECOND instaswitch is already queued to run),
            // and it draws no trailing Update. Must precede the gate (battle.ts:2372
            // is `else if`, ahead of the gate at 2418).
            if matches!(queue.first(), Some(QAction::InstaSwitch { .. })) {
                continue;
            }

            // The switch-request gate: a side with a flagged active that can switch
            // pauses the turn (makeRequest('switch'); return) — SKIPPING the trailing
            // Update.
            let force = self.switch_request_gate();
            if force[0] || force[1] {
                return TurnLoopStop::NeedSwitch { force };
            }

            // The gen<5 trailing eachEvent('Update') (battle.ts:2424) — reached only
            // when the runAction did NOT pause / was not pre-empted by an instaswitch.
            let upd = self.each_event_shuffle();
            self.run_update_items(&upd, dex); // cure-berry/Leppa onUpdate (draw-free)
        }
        TurnLoopStop::Done
    }

    /// The `>start` CONSTRUCTION WINDOW — advance the PRNG (and set the sampled
    /// genders + weather/boosts) from the RAW `>start` seed to the sim's
    /// PRE-FIRST-DECISION state, reproducing every turn-0 draw bit-for-bit
    /// (`gen3_turn0_construction_v1`). This is the bridge's replacement for the pure
    /// [`crate::bridge::advance_seed_for_construction`] seed-advance (which modeled
    /// ONLY the Quick Claw, so it desynced a speed-TIED lead or an unspecified-gender
    /// mon). It runs ONLY on the bridge's raw-seed path — the committed seed goldens
    /// seed at the POST-construction `initSeed` and keep the draw-free
    /// [`BattleState::start_with_switchins`].
    ///
    /// The window, in Showdown's EXACT draw order (probe-verified against the real sim
    /// — `Side.addPokemon` gender draws, then `Battle.start`'s `start` action + the two
    /// `runSwitch` actions, then `endTurn`; NO turn-0 residual — `start` sets
    /// `midTurn=true` so `turnLoop` skips the `beforeTurn`/`residual` inserts, hence no
    /// chip + no `|upkeep|`):
    ///
    /// 1. **Gender samples** ([`draw_turn0_genders`]) — one uniform `sample(['M','F'])`
    ///    per mon (p1's team then p2's, in position order) whose PACKED gender is
    ///    unspecified AND whose species has no FIXED gender.
    /// 2. **The `start` action's `switchIn`s** — enqueue both leads' `runSwitch`
    ///    actions via [`insert_runswitch`], which draws the `insertChoice` tie-break
    ///    `random(2)` when the leads TIE on raw Speed (the 2nd insert's order-101 tie
    ///    window; distinct speed → no draw).
    /// 3. **The `start` action's trailing `eachEvent('Update')`** (the gen<5 runAction
    ///    tail) — [`each_event_shuffle`] draws one tie-shuffle iff the actives tie.
    /// 4. **The two `runSwitch` runActions** via [`turn_loop`] — each fires the lead's
    ///    ability `Start` (Intimidate boosts / Sand Stream weather; a weather CHANGE
    ///    fires the `eachEvent('WeatherChange')` tie-shuffle inside the runAction) then
    ///    the trailing `eachEvent('Update')` tail. Reuses the validated mid-battle
    ///    switch-in machinery, so mixed-weather ties + boost order are handled by
    ///    construction. (There is NO `residual` action, so `turn_loop` returns `Done`
    ///    with an empty queue and draws no weather chip — matching the sim's turn 0.)
    /// 5. **`endTurn`** — the `runEvent('DisableMove')` handler-sort shuffle (a no-op at
    ///    turn 0 — no mon carries a disabling volatile) then the unconditional gen-3
    ///    Quick Claw `randomChance(1,5)`, persisted for the next turn's speed override.
    ///
    /// Runs with logging OFF (the caller enables logging + `emit_framing` AFTER this,
    /// reconstructing the switch-in ability lines from the post-construction state), so
    /// the `run_switch` ability/weather emissions are suppressed here.
    pub fn run_turn0_construction(&mut self, dex: &Dex) {
        // (1) the per-mon gender `sample(['M','F'])` (addPokemon order: p1 then p2).
        self.draw_turn0_genders(dex);
        // (2) the `start` action's two `switchIn`s → the runSwitch queue (+ the
        //     insertChoice tie-break draw on a raw-Speed tie).
        let mut queue: Vec<QAction> = Vec::new();
        for side in 0..self.sides.len() {
            self.insert_runswitch(side, &mut queue, dex);
        }
        // (2b) RECORD the RESOLVED dequeue order (`gen3_turn0_construction_mirror_order_v1`).
        // The bridge re-emits the leads' switch-in ability lines AFTER this window, from the
        // post-construction board, and must use the order the queue ACTUALLY resolved to — at a
        // raw-Speed TIE that is the `insertChoice` PRNG draw made just above, NOT the draw-free
        // "faster first, tie = side order" model. See `BattleState::turn0_switchin_order`.
        self.turn0_switchin_order = Some(
            queue
                .iter()
                .filter_map(|a| match a {
                    QAction::RunSwitch { side } => Some(*side),
                    _ => None,
                })
                .collect(),
        );
        // (3) the `start` action's trailing gen<5 `eachEvent('Update')` tail.
        self.each_event_shuffle();
        // (4) the two runSwitch runActions (ability Start + WeatherChange + Update
        //     tail), through the shared turn machinery. No residual is queued → `Done`.
        let _ = self.turn_loop(&mut queue, dex);
        // (5) endTurn: the DisableMove handler shuffle (no-op at turn 0) + Quick Claw.
        self.disable_move_event_shuffle(dex);
        self.quick_claw_roll = self.prng.random_chance(1, 5);
    }

    /// The turn-0 per-mon GENDER draw (`gen3_turn0_construction_v1`,
    /// `Pokemon` ctor `gender = set.gender || species.gender || sample(['M','F'])`,
    /// pokemon.ts): for every mon on both sides, in `addPokemon` order (p1's team then
    /// p2's, position order), a mon whose PACKED set left the gender unspecified
    /// (`MonState::gender == None`) resolves it — to the species' FIXED gender
    /// (`SpeciesData::gender`, genderless `'N'` / a fixed `'M'`/`'F'`) DRAW-FREE, else
    /// (a normal RATIO'd species) via ONE uniform `sample(['M','F'])`. A mon whose set
    /// already specified the gender is untouched (no draw). This both advances the PRNG
    /// exactly as the sim does AND fills the gender the `|switch|` details render
    /// (`, M`/`, F`; a genderless `'N'` shows no suffix, like the sim's `'N' → ''`).
    fn draw_turn0_genders(&mut self, dex: &Dex) {
        for side in 0..self.sides.len() {
            for slot in 0..self.sides[side].pokemon.len() {
                if self.sides[side].pokemon[slot].gender.is_some() {
                    continue; // the packed set specified it — no draw
                }
                let species_gender = dex
                    .species(&self.sides[side].pokemon[slot].species_id)
                    .and_then(|s| s.gender);
                let g = match species_gender {
                    Some(fixed) => fixed,                  // genderless/fixed — draw-free
                    None => *self.prng.sample(&['M', 'F']), // ratio'd — the uniform draw
                };
                self.sides[side].pokemon[slot].gender = Some(g);
            }
        }
    }

    /// `checkWin` (battle.ts:2155): both sides out → gen-3 TIE (`Ended{winner:None}`);
    /// a side whose FOE is out → that side wins. Returns `Some(winner_or_tie)` when
    /// the battle ends, `None` otherwise. (Wrapped in an `Option<Option<usize>>` via
    /// the caller; here `Some(Some(side))` = win, `Some(None)` = tie.)
    pub(crate) fn check_win(&self) -> Option<Option<usize>> {
        if self.sides[0].pokemon_left == 0 && self.sides[1].pokemon_left == 0 {
            return Some(None); // gen-3: both out → tie (win(null))
        }
        for side in 0..2 {
            let foe = 1 - side;
            if self.sides[foe].pokemon_left == 0 {
                return Some(Some(side)); // foe out → `side` wins
            }
        }
        None
    }
}

impl FullBattleDriver {
    pub(crate) fn new() -> Self {
        FullBattleDriver {
            decisions: Vec::new(),
            pending: ScriptDecision::default(),
            turn_already_opened: false,
            committed_turns: 0,
            phase: DrivePhase::AwaitMove,
        }
    }

    /// Whether the drive has reached game-end.
    pub(crate) fn is_ended(&self) -> bool {
        matches!(self.phase, DrivePhase::Ended(_))
    }

    /// The final [`BattleOutcome`] — `ended:true` ONLY once game-end was reached (a
    /// script-exhaustion at any boundary is a partial `ended:false`, matching the old
    /// driver's returns exactly).
    pub(crate) fn into_outcome(self) -> BattleOutcome {
        match self.phase {
            DrivePhase::Ended(winner) => {
                BattleOutcome { winner, ended: true, decisions: self.decisions }
            }
            _ => BattleOutcome { winner: None, ended: false, decisions: self.decisions },
        }
    }

    /// Feed ONE [`ScriptDecision`], advancing the live battle up to the next request
    /// boundary (or game-end). Byte-identical to the old `run_full_battle` consuming the
    /// same decision at the same point.
    pub(crate) fn feed(&mut self, bs: &mut BattleState, dec: ScriptDecision, dex: &Dex) {
        match std::mem::replace(&mut self.phase, DrivePhase::AwaitMove) {
            DrivePhase::AwaitMove => self.feed_await_move(bs, dec, dex),
            DrivePhase::AwaitSwitch(sw) => self.feed_await_switch(bs, sw, dec, dex),
            DrivePhase::Ended(w) => self.phase = DrivePhase::Ended(w),
        }
    }

    /// A top-of-turn `move` decision — the old outer-loop body from the `move`-request
    /// pull through building + sorting the action queue, then run the turn to its next
    /// boundary. Per-side FIRST-accepted-wins acceptance is preserved across partial
    /// feeds via `self.pending`.
    fn feed_await_move(&mut self, bs: &mut BattleState, dec: ScriptDecision, dex: &Dex) {
        // [side.choose validation — the sim's per-side reject-and-re-request] A `move`
        // request commits ONLY when both choosing sides submit a VALID choice; an illegal
        // side draws nothing and leaves its half of the boundary open. (See `choice_is_legal`.)
        for side in 0..2 {
            if self.pending.for_side(side).is_none() {
                if let Some(c) = dec.for_side(side) {
                    if bs.choice_is_legal(side, c, dex) {
                        self.pending.set_side(side, c);
                    }
                }
            }
        }
        if self.pending.p1.is_none() || self.pending.p2.is_none() {
            // The boundary stays OPEN (any one-side HELD choice included); wait for the
            // next fed decision. Draw-free / zero-state / zero-EMIT.
            self.phase = DrivePhase::AwaitMove;
            return;
        }
        let dec = self.pending;
        self.pending = ScriptDecision::default();

        self.committed_turns += 1;
        if self.committed_turns > BATTLE_TURN_CAP {
            panic!(
                "run_full_battle runaway: >{} committed turns without game-end (self.turn={}) \
                 — a non-terminating battle (stalemate the forfeit failed to end)",
                BATTLE_TURN_CAP, bs.turn,
            );
        }

        // [makeRequest -> commitChoices framing]: increment the turn (unless the previous
        // turn's end already opened it EAGERLY) + emit the batch separator + `|t:|`.
        if !self.turn_already_opened {
            bs.turn += 1;
            if bs.logging() && bs.turn >= 2 {
                bs.log.turn(bs.turn);
            }
        }
        if bs.logging() {
            bs.log.separator();
            bs.log.timestamp();
        }

        // Clear the per-turn diagnostic flags; `run_move`/`drag_in`/`execute_switch` re-set them.
        bs.pending_explosion_self_ko = false;
        bs.pending_phaze_drag = false;
        bs.pursuit_first_mover = None;
        // Expire the `duration:1` FLINCH volatiles from the previous turn. DRAW-FREE.
        bs.clear_flinch();
        // `commitChoices` refreshes the cached `pokemon.speed` at turn start (para/boost-aware)
        // BEFORE the action-order sort + the per-action `eachEvent` shuffles read it.
        bs.update_speed(dex);

        // --- Build + sort the turn's action queue. ---
        let mut actions: Vec<QAction> = Vec::new();
        for side in 0..2 {
            let active = bs.sides[side].active;
            let uid = bs.sides[side].pokemon[active].uid;
            match dec.for_side(side) {
                Some(Choice::Move(mi)) => {
                    // MOVE-LOCKED requests: a MUSTRECHARGE mon's request offered ONLY
                    // `Recharge` and a CHARGING (twoturnmove) mon's ONLY its locked Solar
                    // Beam — the accepted `move 1` maps to the recharge pseudo-move / the
                    // LOCKED move's REAL slot; neither substitutes Struggle nor unshifts a
                    // beforeTurnMove.
                    let locked = bs.sides[side].pokemon[active].move_locked();
                    let mi = bs.sides[side].pokemon[active]
                        .two_turn
                        .filter(|t| t.charging)
                        .map(|t| t.move_index)
                        .unwrap_or(mi);
                    // A mon with NO usable move (all slots at 0 PP) has `moveid:'struggle'`
                    // substituted by `side.choose` — regardless of the scripted slot `mi`.
                    let struggle =
                        !locked && bs.sides[side].pokemon[active].must_struggle(dex);
                    // `beforeTurnMove` unshift (Focus Punch / Pursuit): a `move` whose move
                    // carries a `beforeTurnCallback` ALSO enqueues an order-5 `beforeTurnMove`.
                    if !struggle && !locked {
                        if let Some(m) = bs.move_at(side, active, mi, dex) {
                            if move_has_before_turn_callback(&to_id(&m.id)) {
                                actions.push(QAction::BeforeTurnMove { side, uid, move_index: mi });
                            }
                        }
                    }
                    actions.push(QAction::Move { side, uid, move_index: mi, struggle });
                }
                Some(Choice::Switch(target)) => {
                    actions.push(QAction::Switch { side, target });
                }
                None => {} // no choice this side (not expected on a move request)
            }
        }
        // [commitChoices] ACTION-ORDER sort (the tie shuffle on an order+priority+speed tie).
        bs.sort_actions(&mut actions, dex);

        // First mover = the first move/switch action (post action-order sort).
        let first_mover = actions.iter().find_map(|a| match a {
            QAction::Move { side, .. } => Some(*side),
            QAction::Switch { side, .. } => Some(*side),
            _ => None,
        });

        // The turn's queue: [beforeTurn, <sorted actions>, residual].
        let mut queue: Vec<QAction> = Vec::with_capacity(actions.len() + 2);
        queue.push(QAction::BeforeTurn);
        queue.extend(actions);
        queue.push(QAction::Residual);

        self.run_queue(bs, queue, RequestKind::Move, first_mover, dex);
    }

    /// A forced-replacement decision — the old `NeedSwitch` replacement pull + instaswitch
    /// build + resume. PER-SIDE accumulation across feeds (a double replacement may arrive
    /// as two one-sided decisions).
    fn feed_await_switch(
        &mut self,
        bs: &mut BattleState,
        mut sw: SwitchWait,
        dec: ScriptDecision,
        dex: &Dex,
    ) {
        for side in 0..2 {
            if sw.force[side] && sw.have[side].is_none() {
                if let Some(Choice::Switch(t)) = dec.for_side(side) {
                    let sd = &bs.sides[side];
                    if t < sd.pokemon.len() && !sd.pokemon[t].fainted && t != sd.active {
                        sw.have[side] = Some(t);
                    }
                }
            }
        }
        let satisfied = (0..2).all(|s| !sw.force[s] || sw.have[s].is_some());
        if !satisfied {
            self.phase = DrivePhase::AwaitSwitch(sw);
            return;
        }
        let mut insta: Vec<QAction> = Vec::new();
        for side in 0..2 {
            if sw.force[side] {
                insta.push(QAction::InstaSwitch { side, target: sw.have[side].expect("satisfied") });
                bs.sides[side].switch_flag = false;
            }
        }
        // [commitChoices on a switch request] `updateSpeed()` at the TOP on EVERY choice
        // commit — including a mid-turn FORCED-replacement submit (para/boost-aware).
        bs.update_speed(dex);
        // [EMIT] open the REPLACEMENT batch: `|` (separator) + `|t:|`.
        if bs.logging() {
            bs.log.separator();
            bs.log.timestamp();
        }
        // sort ONLY the new instaswitch(es) (the double-replacement tie shuffle), then
        // PREPEND before the saved remainder (oldQueue).
        bs.sort_actions(&mut insta, dex);
        let mut queue = insta;
        queue.extend(sw.queue);
        self.run_queue(bs, queue, sw.request, sw.first_mover, dex);
    }

    /// Run the (committed) turn queue via `turn_loop` ONCE and handle its result — the
    /// shared tail of both feed paths (the old post-`turn_loop` match). `Done` records the
    /// boundary + eager-opens the next turn; `NeedSwitch` records the boundary + PAUSES for
    /// the replacement; `Ended` emits win/tie + records + ends.
    fn run_queue(
        &mut self,
        bs: &mut BattleState,
        mut queue: Vec<QAction>,
        request: RequestKind,
        first_mover: Option<usize>,
        dex: &Dex,
    ) {
        match bs.turn_loop(&mut queue, dex) {
            TurnLoopStop::Ended { winner } => {
                // [EMIT] the deciding `|` separator then `|win|<PlayerName>` (or `|tie`).
                if bs.logging() {
                    bs.log.separator();
                    match winner {
                        Some(side) => {
                            let name = bs.sides[side].name.clone();
                            bs.log.win(&name);
                        }
                        None => bs.log.tie(),
                    }
                }
                self.decisions.push(bs.boundary_record(request, first_mover, dex));
                self.phase = DrivePhase::Ended(winner);
            }
            TurnLoopStop::Done => {
                // endTurn: the `runEvent('DisableMove')` handler-sort shuffle fires BEFORE
                // the unconditional Quick Claw roll (no faint pause).
                bs.disable_move_event_shuffle(dex);
                bs.bump_active_turns();
                // PERSIST the endTurn Quick Claw roll — read NEXT turn by
                // `effective_speed`/`update_speed` for the gen3 speed=65535 override.
                bs.quick_claw_roll = bs.prng.random_chance(1, 5);
                self.decisions.push(bs.boundary_record(request, first_mover, dex));
                // [EMIT] the NEXT turn's `|turn|N+1` marker EAGERLY (the sim's
                // `makeRequest('move')` flushes it in the COMPLETING write's chunk).
                bs.turn += 1;
                if bs.logging() && bs.turn >= 2 {
                    bs.log.turn(bs.turn);
                }
                self.turn_already_opened = true;
                self.phase = DrivePhase::AwaitMove;
            }
            TurnLoopStop::NeedSwitch { force } => {
                // makeRequest('switch') paused the turn. Record the boundary we JUST finished
                // (the move request, or the prior forced switch) at the PAUSE seed, then PAUSE
                // for the replacement decision.
                // [SEND] `makeRequest('switch')` runs `sendUpdates()`, STREAMING every
                // buffered line so far so a LATER retro-edit can no longer tag an already-
                // flushed `|move|` line.
                if bs.logging() {
                    bs.log.mark_sent();
                }
                self.decisions.push(bs.boundary_record(request, first_mover, dex));
                self.phase = DrivePhase::AwaitSwitch(SwitchWait {
                    queue,
                    force,
                    have: [None, None],
                    // The NEXT boundary we record answers THIS forced switch.
                    request: RequestKind::ForceSwitch { force },
                    first_mover,
                });
            }
        }
    }
}
