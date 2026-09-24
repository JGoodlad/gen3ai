//! `view.rs` — the ONE-SIDED VIEW readout (`gen3_one_sided_view_v1`).
//!
//! The omniscient counterpart is [`crate::search::pre_state`] / [`crate::search::outcome_of`]:
//! the REFEREE's board, which must never reach the observation encoder. **This module is the
//! other half of that wall** — the PROJECTION of the same `BattleState` onto what ONE side has
//! actually observed, in exactly the shape `agents.battle.live_view.LiveView` holds.
//!
//! # Why it exists
//!
//! A search successor's observation is materialized today by REPLAYING that ply's one-sided
//! protocol text through poke-env's state tracker and folding the event log
//! (`agents.training.obs_materializer`). The port already holds the resulting board — it IS the
//! engine — so the parse is re-deriving, in Python, a state Rust never lost. Emitting the
//! projection directly removes the parse from the per-successor path without duplicating one
//! line of encoding logic: the Python sub-encoders keep reading a `LiveView`, only its
//! constructor changes.
//!
//! # The wall
//!
//! The view for side `v` carries:
//!
//! * **`ours`** — side `v`'s own six mons, in full. Nothing is hidden from its owner.
//! * **`opp`** — ONLY the mons side `v` has seen switch in, each carrying only what side `v`
//!   has been told: the moves it has watched being used, an item/ability the protocol
//!   disclosed, and HP as a PERCENT (the gen3ou `reportPercentages` fold) rather than the true
//!   integer pair.
//!
//! An unrevealed opposing mon has NO row at all — not a redacted one — so a consumer cannot
//! learn the team SIZE composition either. `tests/one_sided_view_test.rs` asserts that against
//! the omniscient [`crate::search::pre_state`] on a real mid-battle fixture whose loser still
//! has unrevealed mons: every species string in that side's `pre_state` that is absent from the
//! view must also be absent from the view's BYTES.
//!
//! # What is a SIM FACT here and what is a poke-env PRESENTATION RULE
//!
//! The split is deliberate and is the contract
//! (`designs/rust_sim/one_sided_view.md`): **this module emits sim facts plus REVEAL flags; the
//! Python adapter applies poke-env's presentation rules on top.** So the port never needs
//! poke-env's dex, and a rule that is really poke-env behaviour (the "a species with exactly one
//! possible ability is known without being disclosed" inference; the `unknown_item` sentinel)
//! stays where the library it mirrors lives.
//!
//! The one presentation rule this module DOES own is the **reveal fold** itself, because it is a
//! function of the protocol lines and nothing else can see them: see [`SideObservation`].

use std::collections::BTreeMap;

use crate::bridge::BridgeSession;
use crate::dex::Dex;
use crate::search::json_quote;
use crate::state::{BattleState, MonState, Status, Weather};

// ===========================================================================
// The reveal fold
// ===========================================================================

/// One revealed move, and HOW MANY TIMES the watcher has seen it used.
///
/// The count is here because poke-env's `Move.current_pp` is a COUNTER, not a wire fact:
/// `Pokemon.moved()` calls `move.use()` for an opponent exactly as it does for our own mon, so a
/// watched Earthquake reads `12/16` after four sightings. The port cannot send `current_pp`
/// itself — the true engine PP is privileged information the watcher does not have, and would
/// diverge the moment anything (Pressure, a Sleep-Talk call) decremented it without a `|move|`
/// line the watcher saw.
#[derive(Debug, Clone, Default)]
pub struct MoveObs {
    pub id: String,
    pub uses: u32,
    /// Every sighting that may cost an EXTRA PP, with the count of identical ones. poke-env
    /// decrements a watched move by TWO when `_pressure_on` holds — the target's ABILITY as
    /// poke-env knows it AT USE TIME (single-possible-ability inference, a Trace overlay, a
    /// switch-out clearing that overlay), against a foe-targeting move. So the sighting carries
    /// everything that decision needs and the adapter makes it (`view_adapter._move`).
    pub sightings: BTreeMap<Sighting, u32>,
}

/// One `|move|` line's Pressure inputs (reading rule V3). The two targets are both carried
/// because `Battle._get_target_mon` picks between them by the move's target TYPE, which is
/// poke-env's dex, not the port's: an `all`-target move (Perish Song, Haze, the weather moves)
/// and a line with no target field take the OTHER side's active (`dflt`), everything else the
/// named `target`. Each carries the index into that mon's `ability_events` at USE time, so the
/// adapter replays exactly the ability poke-env held then.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Sighting {
    /// The move whose target type and Pressure decide the extra PP — the move itself, or the
    /// CALLED move of a `[from]move: Sleep Talk` / Metronome line.
    pub mv: String,
    /// A called sighting costs the caller NOTHING unless `_pressure_on` holds for the called
    /// move (`Move.use(pressure, overridden=True)` decrements `1 + pressure − 1`); a plain one
    /// has already been counted in `uses`.
    pub called: bool,
    pub target: String,
    pub target_own: bool,
    pub target_k: usize,
    pub dflt: String,
    pub dflt_k: usize,
}

/// One entry in [`MonObservation::ability_events`]. `id` empty = the mon left the field.
#[derive(Debug, Clone, Default)]
pub struct AbilityEvent {
    pub id: String,
    pub trace: bool,
    /// poke-env assigns this one ONLY while the mon's ability reads `None` (the `-activate`
    /// handler's `if holder_mon.ability is None` guard) — whether it does is the adapter's
    /// question, because it depends on the single-possible-ability inference.
    pub if_unknown: bool,
}

/// One announced volatile — its ANNOUNCEMENT HISTORY, which is what poke-env's rules consume.
///
/// `starts` holds the side's `|turn|` tick at each announcement (`-start` / `-activate` /
/// `-singleturn` / `-singlemove`, re-announcements included). poke-env's lifecycle is a function
/// of exactly that history and the ticks since: `start_effect` creates the effect at 0 when it is
/// absent and increments an `is_action_countable` one when present; `Pokemon.end_turn` (every
/// `|turn|`) increments an `is_turn_countable` one and DELETES an `ends_on_turn` one — so a
/// Protect re-announced a turn after its first `-singleturn` is a FRESH effect, not a restart of
/// the first (a two-counter summary read it as expired). Which effect is which lives in poke-env's
/// `Effect` enum, so the adapter replays the history (`view_adapter._volatiles`).
///
/// `bp_carried` > 0 marks the first `bp_carried` starts as INHERITED through Baton Pass: the
/// entrant received the passer's volatile, and the adapter keeps it only when poke-env's
/// `BATON_PASS_COPIED_EFFECTS` holds it (`Pokemon.apply_baton_pass`, the sim's
/// `copyVolatileFrom`) — the §4b "missing `substitute`" finding.
#[derive(Debug, Clone, Default)]
pub struct VolObs {
    pub name: String,
    pub starts: Vec<u32>,
    pub bp_carried: u32,
}

/// What ONE side has been TOLD — the protocol-derived half of the view.
///
/// Folded line by line as the per-side stream is emitted (`BridgeChunks::push_chunk`), keyed by
/// the mon's protocol IDENT NAME — the nickname, or the base-species display name — because that
/// is the only handle a protocol line carries and it is exactly what poke-env keys its
/// `opponent_team` by.
///
/// It rides [`crate::bridge::BridgeChunks`] rather than the chunk list because it must SURVIVE
/// `clear_chunks()`: a search branch drops its parent's chunk history so its own chunks are its
/// suffix, but what the side has SEEN is cumulative from turn 1 and is not a property of the
/// suffix.
#[derive(Debug, Clone, Default)]
pub struct SideObservation {
    /// Ident names of opposing mons seen on the field, in FIRST-SEEN order — which is the order
    /// poke-env's `opponent_team` dict holds them, i.e. which obs SLOT each occupies.
    pub order: Vec<String>,
    /// Per ident name, what this side knows about the mons it does NOT own.
    pub mons: BTreeMap<String, MonObservation>,
    /// The same fold for this side's OWN mons. Only [`MonObservation::volatiles`] is read from
    /// it: a volatile is protocol-announced for both sides alike, so an own mon's is no more a
    /// projection of engine state than a foe's.
    pub own_mons: BTreeMap<String, MonObservation>,
    /// Own ident names that have been on the field (poke-env's `Pokemon.revealed`, which is set
    /// off the `|switch|` LINE for both sides alike).
    pub own_seen: std::collections::BTreeSet<String>,
    /// The ident name currently on the field, per map — `[own, foe]`.
    ///
    /// Needed because `|switch|p1a: Skarmory|…` names the mon coming IN, while poke-env clears
    /// effects on the mon going OUT (`Battle.switch` calls `outgoing.switch_out()`). Clearing the
    /// incoming mon's volatiles instead left a Pursuit `|-activate|` standing on a mon poke-env
    /// had already wiped.
    pub on_field: [Option<String>; 2],
    /// Own ident names in the order of the FIRST `|request|`'s roster.
    ///
    /// 🚨 **This, not the team sheet, is the obs slot order for our own side.** poke-env builds
    /// `battle.team` in `side.pokemon` order at the first request and a dict never reorders
    /// afterwards, while Showdown floats the ACTIVE mon to `side.pokemon[0]` on every LATER
    /// request. Emitting the port's own `sides[i].pokemon` order permuted all six 122-dim own
    /// slots; measured against poke-env on real boards.
    pub own_order: Vec<String>,
    /// `|turn|` lines seen — the tick every volatile's `starts` are stamped with.
    pub tick: u32,
    /// The protocol's own turn number (`|turn|N`), which is what poke-env stamps a screen with.
    pub proto_turn: u32,
    /// Timed side conditions as poke-env stores them — `[own, foe]`, condition id → the TURN it
    /// started (`abstract_battle._side_start`: `conditions[c] = self.turn`, only when absent;
    /// `side_end` pops it). Reading rule V11: the engine keeps REMAINING turns, a different
    /// quantity (a Light Screen read `4` where poke-env had `25`).
    pub screens: [BTreeMap<String, u32>; 2],
}

/// The per-mon half of [`SideObservation`].
#[derive(Debug, Clone, Default)]
pub struct MonObservation {
    /// poke-env's `status_counter`, folded (reading rule V5,
    /// `designs/rust_sim/one_sided_view.md` §4b).
    ///
    /// The SLEEP half is NOT the sim's counter: `Pokemon.moved` AND `Pokemon.cant_move` increment
    /// it once per `|move|` / `|cant|` LINE while poke-env holds the mon asleep, so a mon the
    /// engine has at `Sleep(3)` reads 0 until it tries to act. The TOXIC half IS the sim's stage
    /// since the fork's PE-R1b fix (`gen3_pe_reading_fixes_v1`): +1 per residual `[from] psn` chip
    /// (capped at 15), reset at the switch-in and at a toxic `switch_out`. A NEW status zeroes it
    /// (the R1 fix), as does `cure_status(<the named status>)`. The obs normalises it
    /// (`min(n,4)/4` asleep, `min(n,8)/8` toxic).
    pub status_counter: u32,
    /// poke-env's OWN `Pokemon._status` for this mon, as the protocol has written it — `"slp"`,
    /// `"tox"`, `"fnt"`, … or `None`. The counter's increments and resets are conditioned on THIS
    /// (not on the engine's status), because poke-env conditions them on its own field, and the
    /// two differ exactly where a line clears a status without naming it (`-cureteam`, an HP
    /// token with no status). Written by every line poke-env writes `_status` from: `-status`,
    /// `-curestatus` (only when it names the held status), `-cureteam` (the named side's
    /// non-fainted mons), `faint`, the HP field of `-damage` / `-heal` / `-sethp` / `switch` /
    /// `drag` / `replace`, and — our own side — every `|request|` roster `condition`.
    pub pstatus: Option<String>,
    /// poke-env's `protect_counter` — the number of consecutive stall moves this mon has USED.
    ///
    /// 🚨 **Not the sim's stall counter, which is a different quantity.** `MonState::protect_counter`
    /// is the gen-3 `stall` volatile's DENOMINATOR (0 -> 2 -> 4 -> 8, advanced only on a SUCCESS
    /// and never reset by a failed roll); poke-env's is a plain count (`Pokemon.moved`:
    /// `+= 1` when the move `is_protect_counter` and the move line did not carry a miss/still
    /// suffix, `= 0` otherwise; `cant_move` and `switch_out` reset it). They disagree in value AND
    /// in when they move, and `pokemon.encode` feeds this one to
    /// `gen3_mechanics.protect_success_probability`. Folded from the protocol for that reason.
    pub protect_counter: u32,
    /// Revealed moves with their sighting counts (the Python side sorts — `LiveView.moves` is
    /// sorted by the poke-env moves-dict key).
    pub moves: Vec<MoveObs>,
    /// The HELD item, once the protocol named it. Cleared when it is consumed.
    pub item: Option<String>,
    /// The item that was consumed/removed, retaining its identity.
    pub consumed_item: Option<String>,
    /// The ability ANNOUNCEMENTS, in order, each flagged with whether the line carried a
    /// `[from] ability: Trace` clause — plus a marker for every switch-out.
    ///
    /// 🚨 **poke-env keeps TWO ability slots and the getter prefers the temporary one.**
    /// `Pokemon.ability`'s setter writes `_ability` only while it is None and `_temporary_ability`
    /// thereafter, `switch_out` clears the temporary, and the `-ability` handler has a Trace
    /// special case that assigns TWICE. Whether a given announcement lands in the base or the
    /// temporary slot therefore depends on the single-possible-ability INFERENCE, which is
    /// poke-env's — so the raw events cross and `view_adapter` replays the slot rules. A Traced
    /// Magnet Pull that read `magnetpull` forever, where poke-env reverts to `trace` the moment
    /// the mon pivots out, is what this replaced.
    pub ability_events: Vec<AbilityEvent>,
    /// Announced volatiles — see [`VolObs`].
    pub volatiles: Vec<VolObs>,
}

impl MonObservation {
    fn start_volatile(&mut self, raw: &str, tick: u32) {
        let v = raw.strip_prefix("move: ").unwrap_or(raw);
        if v.is_empty() {
            return;
        }
        if let Some(existing) = self.volatiles.iter_mut().find(|x| x.name == v) {
            existing.starts.push(tick);
            return;
        }
        self.volatiles.push(VolObs { name: v.to_string(), starts: vec![tick], bp_carried: 0 });
    }

    fn end_volatile(&mut self, raw: &str) {
        let v = raw.strip_prefix("move: ").unwrap_or(raw);
        self.volatiles.retain(|x| x.name != v);
    }

    fn pstatus_is(&self, s: &str) -> bool {
        self.pstatus.as_deref() == Some(s)
    }

    /// The `Pokemon.status` SETTER (`-status`): writes `_status`, and — the fork's R1 fix
    /// (`one_sided_view.md` §4b) — zeroes `_status_counter` when the status CHANGES, so a Rest
    /// taken while badly poisoned starts its sleep count at 0, as the sim's does.
    fn set_pstatus(&mut self, status: &str) {
        if !self.pstatus_is(status) {
            self.status_counter = 0;
        }
        self.pstatus = Some(status.to_string());
    }

    /// `Pokemon.set_hp_status`: `"0 fnt"` is a faint; an `hp status` token writes the status;
    /// a bare `hp` CLEARS it. None of the three touches the counter.
    fn set_hp_status(&mut self, hp: &str) {
        let hp = hp.trim();
        if hp == "0 fnt" {
            self.note_faint();
        } else if let Some((_, st)) = hp.split_once(' ') {
            self.pstatus = Some(st.to_string());
        } else {
            self.pstatus = None;
        }
    }

    /// `Pokemon.faint()` sets `_status = Status.FNT` and leaves `_status_counter` alone, so the
    /// counter FREEZES at its last value and neither the per-turn toxic tick nor `switch_out`'s
    /// toxic reset can touch it again. Clearing the counter here instead read an own Heracross at
    /// 0 where poke-env had 1 — the reset fired when its replacement switched in.
    /// ...and `faint()` does not touch `_protect_counter` either: a mon that Protected and then
    /// fainted before moving again still reads its streak at the replacement decision
    /// (`switch_out` is what zeroes it). Zeroing it here read an own Swampert at 0 where
    /// poke-env had 1 (parity harness slice V, `random_44`).
    fn note_faint(&mut self) {
        self.pstatus = Some("fnt".to_string());
    }

    /// `Pokemon.cure_status(status)`: only when `status` IS the held status does it clear it AND
    /// zero the counter; any other named status is a no-op.
    fn cure_named(&mut self, status: &str) {
        if self.pstatus_is(status) {
            self.pstatus = None;
            self.status_counter = 0;
        }
    }

    /// `Pokemon.cure_status()` with no status (`-cureteam`): clears a non-fainted mon's status
    /// and leaves the counter alone.
    fn cure_unnamed(&mut self) {
        if !self.pstatus_is("fnt") {
            self.pstatus = None;
        }
    }

    fn note_item(&mut self, id: String) {
        self.item = Some(id);
        self.consumed_item = None;
    }
}

impl SideObservation {
    fn entry(&mut self, name: &str) -> &mut MonObservation {
        if !self.mons.contains_key(name) {
            self.order.push(name.to_string());
            self.mons.insert(name.to_string(), MonObservation::default());
        }
        self.mons.get_mut(name).expect("just inserted")
    }

    /// True once `name` has been seen on the field.
    pub fn is_revealed(&self, name: &str) -> bool {
        self.mons.contains_key(name)
    }

    /// The per-mon record for either side, read-only.
    pub fn mon(&self, name: &str, own: bool) -> Option<&MonObservation> {
        if own { self.own_mons.get(name) } else { self.mons.get(name) }
    }

    /// Fold ONE line of this side's stream. `owner_is_self` says whether the line's ident
    /// belongs to the VIEWER.
    pub fn observe(&mut self, line: &str, owner_is_self: bool) {
        let parts: Vec<&str> = line.split('|').collect();
        let Some(tag) = parts.get(1) else { return };

        // `|turn|N` carries no ident and is the tick poke-env's `Pokemon.end_turn` runs on.
        if *tag == "turn" {
            self.tick += 1;
            if let Some(n) = parts.get(2).and_then(|n| n.trim().parse::<u32>().ok()) {
                self.proto_turn = n;
            }
            // `Pokemon.end_turn` advances NO status counter: the fork's PE-R1b fix
            // (`gen3_pe_reading_fixes_v1`) moved the badly-poisoned tick to the residual chip
            // (`fold_status`'s `-damage` arm).
            return;
        }
        // `|request|{...}` — the FIRST one fixes our own team's obs slot order, and EVERY one
        // writes each own mon's `_status` from its roster `condition`
        // (`Pokemon.update_from_request` → `set_hp_status(condition)`).
        if *tag == "request" {
            if self.own_order.is_empty() {
                self.own_order = request_roster_names(line);
            }
            for (name, cond) in request_roster_conditions(line) {
                let m = self.own_mons.entry(name).or_default();
                m.set_hp_status(&cond);
                // `set_hp_status("0 fnt")` calls `faint()` AGAIN, which `_clear_effects()` —
                // so an effect a line re-attached to the corpse after its faint (the
                // `|-activate|…|move: Destiny Bond` that follows the KO) is gone by the decision.
                if cond.trim() == "0 fnt" {
                    m.volatiles.clear();
                }
            }
            return;
        }
        // `|-cureteam|pNa: X|…` — `for mon in team.values(): mon.cure_status()` over the NAMED
        // side's team as poke-env holds it (for the foe: the mons it has seen).
        if *tag == "-cureteam" {
            let own_side = owner_is_self;
            let map = if own_side { &mut self.own_mons } else { &mut self.mons };
            for m in map.values_mut() {
                m.cure_unnamed();
            }
            // (no further fold for this line: it names the user, and nothing below reads it)
            return;
        }

        // An item is disclosed by a `[from] item: X` clause on a `|-damage|` or a `|-heal|` —
        // and ONLY those two. poke-env reads the clause in `_check_damage_message_for_item` /
        // `_check_heal_message_for_item`, which are called from exactly those two handlers;
        // scanning EVERY line instead resurrected a Salac Berry that the preceding `|-enditem|`
        // had already consumed, because the `|-boost|` it triggers carries the same clause.
        // The holder is the `[of] pNa: …` mon when the clause names one, else the line's ident.
        if matches!(*tag, "-damage" | "-heal") {
          if let Some((item, holder)) = item_clause(&parts) {
            let holder_name = holder
                .and_then(|h| ident_name(h))
                .or_else(|| parts.get(2).and_then(|i| ident_name(i)));
            if let Some(hname) = holder_name {
                // poke-env's `_check_*_message_for_item` is side-agnostic; for our OWN mon the
                // held item is already known from the request, but the clause is what CLEARS a
                // stale `consumed_item`, so the fold must see it on both sides.
                if owner_is_self {
                    self.own_mons.entry(hname).or_default().note_item(item);
                } else {
                    self.entry(&hname).note_item(item);
                }
            }
          }
        }

        let Some(ident) = parts.get(2) else { return };
        let Some(name) = ident_name(ident) else { return };

        // ABILITY DISCLOSURES off lines that are NOT `|-ability|` (reading rule V8; the §4b
        // finding "opp ability reads None where poke-env has it"). poke-env writes a mon's ability
        // from four more handlers, each with its own exact shape — see [`ability_disclosure`].
        // An own mon's announcements are folded too: our own `ability` field is the engine's,
        // but a Pressure decision on a watched move replays the ability poke-env held AT USE
        // TIME, and a Traced Pressure on our own Porygon2 is exactly that case.
        if let Some((holder, id, if_unknown)) = ability_disclosure(&parts) {
            let line_side = ident.trim_start().get(..2);
            let holder_side = holder.trim_start().get(..2);
            let holder_is_self = if holder_side == line_side { owner_is_self } else { !owner_is_self };
            if let Some(hname) = ident_name(holder) {
                let ev = AbilityEvent { id, trace: false, if_unknown };
                if holder_is_self {
                    self.own_mons.entry(hname).or_default().ability_events.push(ev);
                } else {
                    self.entry(&hname).ability_events.push(ev);
                }
            }
        }
        // `Pokemon.faint()` clears `temporary_ability` exactly as `switch_out` does (and the sim's
        // faint `clearVolatile` restores the base ability) — so a fainted Traced mon reads its
        // BASE ability again at the replacement decision. The same empty-id marker both sides.
        if *tag == "faint" {
            let map = if owner_is_self { &mut self.own_mons } else { &mut self.mons };
            map.entry(name.clone()).or_default().ability_events.push(AbilityEvent::default());
        }

        // `|-sidestart|pN: Player|Reflect` / `|-sideend|…` — reading rule V11. A timed condition
        // is stored as the turn it STARTED (only when absent); Spikes is a layer COUNT, which the
        // engine holds exactly, so only the timed ones are folded here.
        if matches!(*tag, "-sidestart" | "-sideend") {
            if let Some(cond) = parts.get(3) {
                let id = side_condition_id(cond);
                let map = &mut self.screens[if owner_is_self { 0 } else { 1 }];
                if *tag == "-sideend" {
                    map.remove(&id);
                } else if id != "spikes" {
                    let turn = self.proto_turn;
                    map.entry(id).or_insert(turn);
                }
            }
            return;
        }

        if owner_is_self {
            if *tag == "-ability" {
                if let Some(ab) = parts.get(3) {
                    let trace = parts
                        .iter()
                        .skip(4)
                        .any(|t| t.trim().starts_with("[from] ability: Trace"));
                    self.own_mons.entry(name.clone()).or_default().ability_events.push(
                        AbilityEvent { id: to_id(ab), trace, if_unknown: false },
                    );
                }
            }
            if matches!(*tag, "-item" | "-enditem") {
                if let Some(item) = parts.get(3) {
                    let id = to_id(item);
                    let e = self.own_mons.entry(name.clone()).or_default();
                    if *tag == "-item" {
                        e.note_item(id);
                    } else {
                        e.item = None;
                        e.consumed_item = Some(id);
                    }
                }
            }
            let mut carried: Vec<VolObs> = Vec::new();
            if matches!(*tag, "switch" | "drag" | "replace") {
                self.own_seen.insert(name.clone());
                if let Some(prev) = self.on_field[0].replace(name.clone()) {
                    if prev != name {
                        let p = self.own_mons.entry(prev).or_default();
                        if from_baton_pass(&parts) {
                            carried = baton_pass_carry(&p.volatiles);
                        }
                        p.volatiles.clear();
                        switch_out_status(p);
                        p.ability_events.push(AbilityEvent::default());
                    }
                }
            }
            let tick = self.tick;
            let e = self.own_mons.entry(name).or_default();
            fold_status(e, tag, &parts);
            fold_volatiles(e, tag, &parts, tick);
            e.volatiles.extend(carried);
            return;
        }

        let mut carried: Vec<VolObs> = Vec::new();
        match *tag {
            "switch" | "drag" | "replace" => {
                self.entry(&name);
                if let Some(prev) = self.on_field[1].replace(name.clone()) {
                    if prev != name {
                        let p = self.entry(&prev);
                        if from_baton_pass(&parts) {
                            carried = baton_pass_carry(&p.volatiles);
                        }
                        p.volatiles.clear();
                        switch_out_status(p);
                        // `Pokemon.switch_out` clears `temporary_ability`; which announcement
                        // that was is poke-env's slot question, so the EVENT crosses instead.
                        p.ability_events.push(AbilityEvent::default());
                    }
                }
            }
            "move" => {
                if let Some(raw_move) = parts.get(3) {
                    let id = to_id(raw_move);
                    // `Battle._get_target_mon`: a move line with NO target (absent, or the empty
                    // field a `[still]` / `[notarget]` fail prints) targets the OTHER side's
                    // active — i.e. OUR active, for a foe's move. That is the mon `_pressure_on`
                    // charges a second PP against; dropping the sighting's target instead lost it
                    // (a Will-O-Wisp that failed into our asleep Pressure Zapdos read 1 PP high).
                    let dflt = self.on_field[0].clone().unwrap_or_default();
                    let named = parts.get(4).filter(|t| ident_name(t).is_some());
                    let (target, target_own) = match named {
                        // a foe's line: its ident's side is the FOE's, so a named target on the
                        // other prefix is ours
                        Some(t) => (
                            ident_name(t).unwrap_or_default(),
                            t.trim_start().get(..2) != ident.trim_start().get(..2),
                        ),
                        None => (dflt.clone(), true),
                    };
                    let k_of = |own: bool, n: &str| -> usize {
                        let m = if own { self.own_mons.get(n) } else { self.mons.get(n) };
                        m.map_or(0, |m| m.ability_events.len())
                    };
                    let mut s = Sighting {
                        mv: id.clone(),
                        called: false,
                        target_k: k_of(target_own, &target),
                        dflt_k: k_of(true, &dflt),
                        target,
                        target_own,
                        dflt,
                    };
                    let callable =
                        !id.is_empty() && id != "struggle" && id != "recharge" && id != "fight";
                    if let Some(caller) = called_from(&parts) {
                        // `|move|X|Called|…|[from]move: Sleep Talk` (and Metronome): poke-env runs
                        // `mon.moved(Called, use=False, reveal=…)` — the called move is REVEALED
                        // (Sleep Talk only) with NO use — then
                        // `mon.moves[caller].use(pressure, overridden=True)`, which charges the
                        // CALLER one PP only when `_pressure_on` holds for the CALLED move. So the
                        // called sighting crosses on the caller, keyed by the called move, and the
                        // adapter decides Pressure against the called move's target.
                        let e = self.entry(&name);
                        if callable && caller == "sleeptalk" {
                            move_slot(e, &id);
                        }
                        s.called = true;
                        *move_slot(e, &caller).sightings.entry(s).or_insert(0) += 1;
                    } else if callable && move_reveals(&parts) {
                        let slot = move_slot(self.entry(&name), &id);
                        slot.uses += 1;
                        *slot.sightings.entry(s).or_insert(0) += 1;
                    }
                }
            }
            "-item" => {
                if let Some(item) = parts.get(3) {
                    let id = to_id(item);
                    self.entry(&name).note_item(id);
                }
            }
            "-enditem" => {
                if let Some(item) = parts.get(3) {
                    let id = to_id(item);
                    let e = self.entry(&name);
                    e.item = None;
                    e.consumed_item = Some(id);
                }
            }
            "-ability" => {
                if let Some(ab) = parts.get(3) {
                    let trace = parts
                        .iter()
                        .skip(4)
                        .any(|t| t.trim().starts_with("[from] ability: Trace"));
                    self.entry(&name)
                        .ability_events
                        .push(AbilityEvent { id: to_id(ab), trace, if_unknown: false });
                }
            }
            _ => {}
        }
        let tick = self.tick;
        let e = self.entry(&name);
        fold_status(e, tag, &parts);
        fold_volatiles(e, tag, &parts, tick);
        e.volatiles.extend(carried);
    }
}

/// `|switch|…|[from] Baton Pass` — the tag `abstract_battle` reads (`from_baton_pass=any(tag
/// .startswith("[from]") and "baton pass" in tag.lower() for tag in event[5:])`).
fn from_baton_pass(parts: &[&str]) -> bool {
    parts
        .iter()
        .skip(5)
        .any(|t| t.starts_with("[from]") && t.to_ascii_lowercase().contains("baton pass"))
}

/// The passer's volatiles as the ENTRANT inherits them (`Pokemon.apply_baton_pass`): every one
/// crosses, flagged as carried, and the adapter keeps those in `BATON_PASS_COPIED_EFFECTS`.
fn baton_pass_carry(passer: &[VolObs]) -> Vec<VolObs> {
    passer
        .iter()
        .map(|v| VolObs { bp_carried: v.starts.len() as u32, ..v.clone() })
        .collect()
}

/// poke-env's `SideCondition.from_showdown_message` id, as `LiveView` keys it
/// (`Light Screen` / `move: Light Screen` → `light_screen`).
fn side_condition_id(raw: &str) -> String {
    let v = raw.trim();
    let v = v.strip_prefix("move: ").unwrap_or(v);
    v.to_ascii_lowercase().replace(' ', "_")
}

/// poke-env's `_PROTECT_COUNTER_MOVES`, narrowed to the three that exist in gen 3.
const STALL_MOVES: [&str; 3] = ["protect", "detect", "endure"];

fn is_stall_move(parts: &[&str]) -> bool {
    parts
        .get(3)
        .map(|m| STALL_MOVES.contains(&to_id(m).as_str()))
        .unwrap_or(false)
}

/// The three suffixes `abstract_battle.parse_message` turns into `failed=True` on a `|move|`.
fn move_line_failed(parts: &[&str]) -> bool {
    parts
        .last()
        .map(|t| matches!(t.trim(), "[miss]" | "[still]" | "[notarget]"))
        .unwrap_or(false)
}

/// `Pokemon.switch_out`: `if self._status == Status.TOX: self._status_counter = 0`. The SLEEP
/// counter is deliberately NOT reset — gen 3 sleep persists across a pivot, and poke-env models
/// that by resetting only the toxic one.
fn switch_out_status(e: &mut MonObservation) {
    if e.pstatus_is("tox") {
        e.status_counter = 0;
    }
    e.protect_counter = 0;
}

/// poke-env's `status_counter` bookkeeping, identical for both sides (reading rule V5). Every
/// arm names the poke-env line it mirrors; `-cureteam` and the `|request|` roster are in
/// `observe` (they are not about the line's own ident).
fn fold_status(e: &mut MonObservation, tag: &str, parts: &[&str]) {
    match tag {
        // `abstract_battle` `-status` → the `status` setter: no counter reset.
        "-status" => {
            if let Some(st) = parts.get(3) {
                e.set_pstatus(st.trim());
            }
        }
        // `-curestatus` → `cure_status(status)`.
        "-curestatus" => {
            if let Some(st) = parts.get(3) {
                e.cure_named(st.trim());
            }
        }
        "faint" => e.note_faint(),
        // `-damage` → `damage` → `set_hp_status`; `-heal` → `set_hp_status`; `-sethp` →
        // `set_hp` → `set_hp_status`. The HP token carries (or omits) the status.
        "-damage" | "-heal" | "-sethp" => {
            if let Some(hp) = parts.get(3) {
                e.set_hp_status(hp);
            }
            // `AbstractBattle` `-damage` → `Pokemon.note_residual_chip` (the fork's PE-R1b fix,
            // `gen3_pe_reading_fixes_v1`): a `[from] psn` chip on a mon STILL holding `tox` after
            // its HP token is one stage of the sim's `tox`, capped at 15. A chip that KOs has
            // already made the status `fnt`, so it does not count.
            if tag == "-damage" && e.pstatus_is("tox") && parts.iter().skip(4).any(|t| *t == "[from] psn") {
                e.status_counter = (e.status_counter + 1).min(15);
            }
        }
        // `Battle.switch` → `pokemon.set_hp_status(hp_status)` on the ENTRANT. The counter
        // reset belongs to the one going OUT (`switch_out_status`, via `on_field` in `observe`).
        "switch" | "drag" | "replace" => {
            // `Battle.switch` (the fork's PE-R1b fix): `tox.onSwitchIn` resets the stage — read
            // off the status the ENTRANT held before this line's HP token.
            if tag != "replace" && e.pstatus_is("tox") {
                e.status_counter = 0;
            }
            if let Some(hp) = parts.get(4) {
                e.set_hp_status(hp);
            }
        }
        // `Pokemon.moved` AND `Pokemon.cant_move` both do
        // `if self._status == Status.SLP: self._status_counter += 1` — a sleeping mon that is
        // TOLD it cannot move counts that turn exactly like one that acted, and EVERY `|move|`
        // line counts (a Sleep Talk turn is `cant` + `move Sleep Talk` + the called move).
        "move" | "cant" => {
            if e.pstatus_is("slp") {
                e.status_counter += 1;
            }
            // `Pokemon.moved`: the stall streak advances on a stall move that was not flagged
            // failed by the move LINE, and resets on anything else. `cant_move` resets it.
            if tag == "cant" {
                e.protect_counter = 0;
            } else if is_stall_move(parts) && !move_line_failed(parts) {
                e.protect_counter += 1;
            } else {
                e.protect_counter = 0;
            }
        }
        _ => {}
    }
}

/// The volatile fold, identical for both sides (see [`MonObservation::volatiles`]).
fn fold_volatiles(e: &mut MonObservation, tag: &str, parts: &[&str], tick: u32) {
    match tag {
        // poke-env clears every effect on switch-out and on faint. The INCOMING mon is wiped
        // too (it arrives clean); the OUTGOING one is handled by the `on_field` bookkeeping in
        // `observe`, because a `|switch|` line names only the entrant.
        "switch" | "drag" | "replace" | "faint" => e.volatiles.clear(),
        "-start" | "-activate" | "-singleturn" | "-singlemove" => {
            if let Some(raw) = parts.get(3) {
                e.start_volatile(raw, tick);
            }
        }
        "-end" => {
            if let Some(raw) = parts.get(3) {
                e.end_volatile(raw);
            }
        }
        "move" => {
            // poke-env's one SILENT starter: `|move|IDENT|Minimize` adds MINIMIZE with no
            // `|-start|` of its own (`abstract_battle.parse_message`).
            if parts.get(3).map(|m| m.trim().eq_ignore_ascii_case("minimize")) == Some(true) {
                e.start_volatile("MINIMIZE", tick);
            }
        }
        _ => {}
    }
}

/// `("leftovers", Some("p2a: Snorlax"))` for a line carrying `[from] item: Leftovers` (and,
/// optionally, `[of] p2a: Snorlax`). `None` when there is no item clause.
fn item_clause<'a>(parts: &[&'a str]) -> Option<(String, Option<&'a str>)> {
    let mut item: Option<String> = None;
    let mut of: Option<&str> = None;
    for p in parts.iter().skip(3) {
        let t = p.trim();
        if let Some(rest) = t.strip_prefix("[from] item: ").or_else(|| t.strip_prefix("[from]item: "))
        {
            item = Some(to_id(rest));
        } else if let Some(rest) = t.strip_prefix("[of] ") {
            of = Some(rest);
        }
    }
    item.map(|i| (i, of))
}

/// The roster ident NAMES in a `|request|` line, in order — scanned rather than JSON-parsed
/// because the only thing needed is the `"ident":"pN: <name>"` sequence.
fn request_roster_names(line: &str) -> Vec<String> {
    let needle = "\"ident\":\"";
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(i) = rest.find(needle) {
        rest = &rest[i + needle.len()..];
        let Some(j) = rest.find('"') else { break };
        if let Some(n) = ident_name(&rest[..j]) {
            if !out.contains(&n) {
                out.push(n);
            }
        }
        rest = &rest[j..];
    }
    out
}

/// `(ident name, condition)` for every roster entry of a `|request|` line, in order — the
/// `side.pokemon[i].ident` / `.condition` pairs `Pokemon.update_from_request` reads. Scanned like
/// [`request_roster_names`]: each entry's `"condition"` is the first one after its `"ident"`.
fn request_roster_conditions(line: &str) -> Vec<(String, String)> {
    let (ni, nc) = ("\"ident\":\"", "\"condition\":\"");
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(i) = rest.find(ni) {
        rest = &rest[i + ni.len()..];
        let Some(j) = rest.find('"') else { break };
        let name = ident_name(&rest[..j]);
        rest = &rest[j..];
        let Some(k) = rest.find(nc) else { break };
        // the condition must belong to THIS entry: no later ident may sit between them
        if rest.find(ni).map_or(false, |x| x < k) {
            continue;
        }
        let after = &rest[k + nc.len()..];
        let Some(e) = after.find('"') else { break };
        if let Some(n) = name {
            out.push((n, after[..e].to_string()));
        }
        rest = &after[e..];
    }
    out
}

/// An ability poke-env ASSIGNS off a line other than `|-ability|`: `(holder ident, ability id,
/// only-if-unknown)`. The four `abstract_battle` paths, each at poke-env's EXACT shape (a length
/// check is part of the rule — a `-heal` with no `[of]` discloses nothing):
///
/// * `_check_damage_message_for_ability` — `|-damage|X|hp|[from] ability: A|[of] Y` (6 fields):
///   `Y.ability = A` (Rough Skin, …);
/// * `_check_heal_message_for_ability` — `|-heal|X|hp|[from] ability: A|[of] Y` (6 fields):
///   `X.ability = A` (Water / Volt Absorb);
/// * `-immune` — `|-immune|X|[from] ability: A` (4 fields): `X.ability = A` (Levitate, Volt
///   Absorb, Immunity, …);
/// * `-activate` — `|-activate|X|ability: A|…[of] Y…`: the `[of]` holder (else X), and ONLY
///   `if holder_mon.ability is None`. The Dancer / Mummy / Wandering Spirit / Symbiosis branches
///   precede it and never reach it.
fn ability_disclosure<'a>(parts: &[&'a str]) -> Option<(&'a str, String, bool)> {
    let tag = *parts.get(1)?;
    let from_ability = |t: &str| t.starts_with("[from] ability:").then(|| to_id(&t["[from] ability:".len()..]));
    match tag {
        "-damage" if parts.len() == 6 && parts[5].starts_with("[of]") => {
            let id = from_ability(parts[4])?;
            Some((parts[5]["[of]".len()..].trim(), id, false))
        }
        "-heal" if parts.len() == 6 => {
            let id = from_ability(parts[4])?;
            if id == "hospitality" {
                return Some((parts[5].trim_start_matches("[of] ").trim(), id, false));
            }
            Some((parts[2], id, false))
        }
        "-immune" if parts.len() == 4 => Some((parts[2], from_ability(parts[3])?, false)),
        "-activate" if parts.len() >= 4 && !parts[2].is_empty() => {
            let eff = parts[3].strip_prefix("ability: ")?;
            if matches!(eff, "Dancer" | "Mummy" | "Wandering Spirit" | "Symbiosis") {
                return None;
            }
            let holder = parts[4..]
                .iter()
                .find_map(|t| t.strip_prefix("[of] "))
                .unwrap_or(parts[2]);
            Some((holder, to_id(eff), true))
        }
        _ => None,
    }
}

/// The move slot `id` on `e`, created (revealed, 0 uses) when absent — `Pokemon._add_move`.
fn move_slot<'a>(e: &'a mut MonObservation, id: &str) -> &'a mut MoveObs {
    if let Some(i) = e.moves.iter().position(|m| m.id == id) {
        return &mut e.moves[i];
    }
    e.moves.push(MoveObs { id: id.to_string(), ..MoveObs::default() });
    e.moves.last_mut().expect("just pushed")
}

/// The CALLER of a `|move|` line whose LAST field — after the `[miss]` / `[still]` /
/// `[notarget]` / `[spread]` / `[anim]` suffixes poke-env strips first — is `[from] move: M` /
/// `[from]move: M` (or the legacy `[from] Sleep Talk`), as an id: the one position
/// `abstract_battle`'s move handler reads it from. `None` for every other line (a
/// `lockedmove` / Mirror Move / Snatch clause is not a caller: poke-env neither reveals nor uses
/// those, see [`move_reveals`]).
fn called_from(parts: &[&str]) -> Option<String> {
    let mut end = parts.len();
    while end > 4 {
        let t = parts[end - 1].trim();
        if matches!(t, "[miss]" | "[still]" | "[notarget]")
            || t.starts_with("[spread]")
            || t.starts_with("[anim]")
        {
            end -= 1;
        } else {
            break;
        }
    }
    let last = parts.get(end.checked_sub(1)?)?.trim();
    if end <= 4 {
        return None;
    }
    if last == "[from] Sleep Talk" {
        return Some("sleeptalk".into());
    }
    last.strip_prefix("[from] move: ").or_else(|| last.strip_prefix("[from]move: ")).map(to_id)
}

/// Does this `|move|` line REVEAL the move to the watcher?
///
/// **The `[from]` suppression rules are poke-env's, narrowed to gen 3.**
/// `abstract_battle.parse_message` sets `reveal = False` for a move whose `[from]` clause says
/// the actor did not choose it; in gen 3 the live cases are `lockedmove` (the Thrash / Outrage /
/// Petal Dance / Rollout continuation turns) and `move: Metronome` / `Mirror Move` / `Snatch` /
/// `Magic Coat`. `move: Sleep Talk` DOES reveal (poke-env's explicit `pass`), which is why it is
/// an exception rather than falling through.
fn move_reveals(parts: &[&str]) -> bool {
    for p in parts.iter().skip(4) {
        let t = p.trim();
        let low = t.to_ascii_lowercase();
        if low == "[from] lockedmove" || low == "[from]lockedmove" {
            return false;
        }
        if let Some(src) = t
            .strip_prefix("[from] move: ")
            .or_else(|| t.strip_prefix("[from]move: "))
        {
            if !src.eq_ignore_ascii_case("Sleep Talk") {
                return false;
            }
        }
        if low.starts_with("[from] mirror move")
            || low.starts_with("[from] snatch")
            || low.starts_with("[from]snatch")
            || low.starts_with("[from] magic coat")
        {
            return false;
        }
    }
    true
}

/// `"p2a: Suicune"` → `Some("Suicune")`; `"p2: Suicune"` likewise. `None` when the token is not
/// a mon ident.
pub fn ident_name(ident: &str) -> Option<String> {
    let id = ident.trim_start();
    let rest = id.strip_prefix("p1").or_else(|| id.strip_prefix("p2"))?;
    let rest = rest.strip_prefix('a').unwrap_or(rest);
    let rest = rest.strip_prefix(": ")?;
    if rest.is_empty() {
        None
    } else {
        Some(rest.to_string())
    }
}

/// Showdown's `toID`: lowercase, strip everything that is not `[a-z0-9]`.
pub fn to_id(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

// ===========================================================================
// The renderer
// ===========================================================================

/// The one-sided view of the CURRENT board as `side` (0 = p1, 1 = p2) has observed it, as one
/// JSON object. `"null"` when the session has no state.
///
/// Shape (the CONTRACT — `designs/rust_sim/one_sided_view.md` owns it):
///
/// ```text
/// {"side":"p1","turn":N,"finished":bool,"won":bool|null,"lost":bool|null,
///  "weather":{"weather":id|null,"is_permanent":bool,"turns_active":n},
///  "ours":SIDE,"opp":SIDE,"request":<the |request| payload>|null}
/// SIDE = {"team_size":n,"active":<species id>|null,"side_conditions":{id:n},"mons":[MON,…]}
/// ```
///
/// Needs the session's reveal fold ([`BridgeSession::enable_view_fold`], off by default so the
/// training transport never pays it — `gen3_view_fold_opt_in_v1`); without it this REFUSES rather
/// than render a view that claims nothing was ever revealed.
pub fn one_sided_view(sess: &BridgeSession, side: usize, dex: &Dex) -> Result<String, String> {
    let Some(st) = sess.battle_state() else {
        return Ok("null".to_string());
    };
    let opp = 1 - side;
    let obs = sess
        .observed(side)
        .ok_or("one_sided_view: the session's view fold is off (call BridgeSession::enable_view_fold)")?;
    let ended = sess.is_ended();
    let (won, lost) = match (ended, sess.winner()) {
        (false, _) => ("null".to_string(), "null".to_string()),
        (true, None) => ("false".to_string(), "false".to_string()), // a gen3 TIE
        (true, Some(w)) => {
            let we = w == side;
            (we.to_string(), (!we).to_string())
        }
    };
    let request = match sess.active_request_json(side) {
        Some(line) => line.strip_prefix("|request|").unwrap_or(line).to_string(),
        None => "null".to_string(),
    };
    // THE TURN THE PROTOCOL HAS ANNOUNCED. The driver bumps `bs.turn` EAGERLY at every turn end
    // (with the `|turn|N+1` it emits), but the FIRST `|turn|1` is emitted by the construction
    // framing while `bs.turn` stays 0 until the first commit — so the turn-1 board read 0 where
    // the sim (`this.turn`) and poke-env both hold 1, and every weather `turns_active` on it
    // inherited the lag. Measured by the parity harness's slice V on every battle's first
    // decision; a search root never opens at turn 1, which is why the search gate never saw it.
    let turn = st.turn.max(1);
    Ok(format!(
        "{{\"side\":\"p{}\",\"turn\":{},\"finished\":{},\"won\":{},\"lost\":{},\
         \"weather\":{},\"ours\":{},\"opp\":{},\"request\":{}}}",
        side + 1,
        turn,
        ended,
        won,
        lost,
        weather_json(st, turn),
        side_json(st, side, side, obs, dex, !ended),
        side_json(st, opp, side, obs, dex, !ended),
        request,
    ))
}

fn weather_json(st: &BattleState, turn: u32) -> String {
    let Some(w) = st.field.weather else {
        return "{\"weather\":null,\"is_permanent\":false,\"turns_active\":0}".to_string();
    };
    // `weather_turns == 0` with weather SET is the port's spelling of ability-sourced
    // (permanent) weather — `event.rs` sets it explicitly on the Sand Stream / Drizzle /
    // Drought path, and the residual never decrements it. A MOVE-set weather carries 1..=5
    // REMAINING, so `LiveWeather.turns_active` (which counts UP from the set turn, over a
    // gen3 total of 5) is `5 - remaining`.
    let permanent = st.field.weather_turns == 0;
    // `LiveWeather.turns_active` is `now_turn - start_turn` on BOTH branches — poke-env folds it
    // from the `|-weather|` event's turn and knows nothing about the sim's countdown. Deriving it
    // from `weather_turns` works only for MOVE weather (5 - remaining) and reads 0 forever for
    // ability weather, which is why `Field::weather_start_turn` exists.
    let turns_active = turn.saturating_sub(st.field.weather_start_turn);
    format!(
        "{{\"weather\":{},\"is_permanent\":{},\"turns_active\":{}}}",
        json_quote(weather_id(w)),
        permanent,
        turns_active,
    )
}

fn weather_id(w: Weather) -> &'static str {
    match w {
        Weather::Sand => "sandstorm",
        Weather::Rain => "raindance",
        Weather::Sun => "sunnyday",
        Weather::Hail => "hail",
    }
}

/// One side of the board as `viewer` sees it. `viewer == which` ⇒ the owner's full view.
fn side_json(
    st: &BattleState,
    which: usize,
    viewer: usize,
    obs: &SideObservation,
    dex: &Dex,
    running: bool,
) -> String {
    let is_own = which == viewer;
    let sd = &st.sides[which];
    let active_slot = sd.active;
    let mut rows: Vec<String> = Vec::new();
    let mut active_species = "null".to_string();
    // ROW ORDER IS PART OF THE CONTRACT. poke-env's `battle.team` is built in ROSTER order
    // (the request's `side.pokemon` array) while `battle.opponent_team` is a dict filled in
    // REVEAL order — and the obs encoder walks exactly those two, so slot *i* of the opponent
    // block is the *i*-th mon that was SEEN, not the *i*-th on their team sheet. Emitting the
    // opponent in roster order would silently permute six 122-dim slots.
    let order: Vec<usize> = if is_own {
        if obs.own_order.is_empty() {
            (0..sd.pokemon.len()).collect()
        } else {
            obs.own_order
                .iter()
                .filter_map(|name| sd.pokemon.iter().position(|m| &display_name(m, dex) == name))
                .collect()
        }
    } else {
        obs.order
            .iter()
            .filter_map(|name| sd.pokemon.iter().position(|m| &display_name(m, dex) == name))
            .collect()
    };
    for slot in order {
        let mon = &sd.pokemon[slot];
        let name = display_name(mon, dex);
        let is_active = slot == active_slot;
        if !is_own && !obs.is_revealed(&name) {
            // THE WALL: an unseen opposing mon has no row at all.
            continue;
        }
        let mon_obs = obs.mon(&name, is_own);
        let revealed = if is_own { obs.own_seen.contains(&name) || is_active } else { true };
        rows.push(mon_json(st, mon, is_active, is_own, revealed, mon_obs, dex, running, viewer, obs.tick));
        if is_active {
            active_species = json_quote(identity_species(mon));
        }
    }
    format!(
        "{{\"team_size\":{},\"active\":{},\"side_conditions\":{},\"mons\":[{}]}}",
        sd.pokemon.len(),
        active_species,
        side_conditions_json(sd, &obs.screens[if is_own { 0 } else { 1 }]),
        rows.join(","),
    )
}

fn side_conditions_json(sd: &crate::state::SideState, started: &BTreeMap<String, u32>) -> String {
    let mut parts: Vec<String> = Vec::new();
    // poke-env keys `side_conditions` by the lowercased SideCondition enum NAME and stores
    // Spikes as a LAYER count while the timed screens store the TURN THEY STARTED (reading rule
    // V11). PRESENCE is the engine's (a sim fact); a timed screen's VALUE is the protocol's start
    // turn from the reveal fold — the engine's remaining-turn counter is a different quantity,
    // and is emitted only if the fold somehow missed the `-sidestart`, so the parity gate sees it.
    if sd.spikes > 0 {
        parts.push(format!("\"spikes\":{}", sd.spikes));
    }
    for (id, remaining) in
        [("reflect", sd.reflect), ("light_screen", sd.light_screen), ("safeguard", sd.safeguard)]
    {
        if remaining > 0 {
            let v = started.get(id).copied().unwrap_or(remaining as u32);
            parts.push(format!("\"{id}\":{v}"));
        }
    }
    format!("{{{}}}", parts.join(","))
}

/// One mon. `own` ⇒ every field; otherwise the reveal-gated projection.
fn mon_json(
    st: &BattleState,
    mon: &MonState,
    is_active: bool,
    own: bool,
    revealed: bool,
    obs: Option<&MonObservation>,
    dex: &Dex,
    running: bool,
    viewer: usize,
    tick: u32,
) -> String {
    let sp = dex.species(&mon.species_id);
    let base_stats = match sp {
        Some(s) => format!(
            "{{\"hp\":{},\"atk\":{},\"def\":{},\"spa\":{},\"spd\":{},\"spe\":{}}}",
            s.base_stats.hp, s.base_stats.atk, s.base_stats.def, s.base_stats.spa,
            s.base_stats.spd, s.base_stats.spe
        ),
        None => "{}".to_string(),
    };
    let types: Vec<String> = live_types(mon, dex)
        .iter()
        .map(|t| json_quote(&format!("{t:?}").to_ascii_lowercase()))
        .collect();

    // HP: the OWNER sees the true integer pair; everyone else sees the ceil-% fold that
    // `bridge::derive_side` already applies to the wire (so the two agree by construction).
    let (cur_hp, max_hp) = if own {
        (mon.hp as u32, mon.maxhp as u32)
    } else {
        (percent_hp(mon.hp, mon.maxhp), 100)
    };
    let hp_fraction = if cur_hp == 0 { 0.0 } else { cur_hp as f64 / max_hp as f64 };

    let moves = if own {
        let rows: Vec<String> = mon
            .set
            .moves
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let pp = mon.move_pp.get(i).copied().unwrap_or(0);
                let maxpp = mon.move_maxpp.get(i).copied().unwrap_or(0);
                // `id` is the poke-env MOVES-DICT KEY (a typed Hidden Power is re-keyed bare by
                // the wire) and `move_id` is the `Move.id` that key maps to (typed). Both are
                // needed: `LiveView.moves` is sorted by the KEY, while `MovesEncoder` sorts by
                // `Move.id` and looks the dex row up with it.
                // The typed-HP resolution is the ONE resolver `state::typed_hp_move_id` — the
                // same one the `|request|` roster goes through. A set can store Hidden Power
                // BARE with the type in a separate marker, in which case `to_id` alone yields
                // `hiddenpower` (dex num 237) while poke-env, reading the roster, holds the
                // typed id (355-370). Measured: one own BENCH mon's move-slot id per board.
                let real = to_id(&crate::state::typed_hp_move_id(mon, id));
                // OUR OWN pp is the ENGINE's, which is also exactly what the `|request|` states —
                // and poke-env asserts its own counter equal to that request
                // (`Pokemon.check_move_consistency`). DEFERRAL D7: when poke-env has NOT yet
                // identified a Pressure holder its counter drifts one below the wire per
                // sighting, and it is the drifted value the protocol road encodes. Reproducing
                // the drift would mean folding our own PP from sightings too, which was measured
                // WORSE (the `|move|` line names "Hidden Power" while the set token is
                // "hiddenpowerfire", so the slots do not key against each other).
                format!(
                    "{{\"id\":{},\"move_id\":{},\"current_pp\":{},\"max_pp\":{}}}",
                    json_quote(&bare_move_key(&real)),
                    json_quote(&real),
                    pp,
                    maxpp
                )
            })
            .collect();
        format!("[{}]", rows.join(","))
    } else {
        // Gen 3 Showdown does not tell the watcher an opponent's PP, so poke-env reports a
        // revealed opposing move at FULL pp; the port must say the same thing, from the DEX
        // rather than from the true `move_pp` it happens to hold.
        let rows: Vec<String> = obs
            .map(|o| o.moves.clone())
            .unwrap_or_default()
            .iter()
            .map(|m| {
                let maxpp = dex.moves(&m.id).map(|d| (d.pp as u32) * 8 / 5).unwrap_or(0);
                // `own` below is the TARGET's side from the VIEWER's seat: our mons are
                // `sides[viewer]`. Resolving the species per side is what keeps a MIRROR (both
                // teams running Zapdos) from reading the wrong one's ability.
                let sp_of = |name: &str, target_own: bool| {
                    species_of_ident(st, name, dex, if target_own { viewer } else { 1 - viewer })
                };
                let sightings: Vec<String> = m
                    .sightings
                    .iter()
                    .map(|(s, n)| {
                        format!(
                            "{{\"mv\":{},\"called\":{},\"t\":{},\"t_own\":{},\"t_k\":{},\
                             \"d\":{},\"d_k\":{},\"n\":{}}}",
                            json_quote(&s.mv),
                            s.called,
                            json_quote(&sp_of(&s.target, s.target_own)),
                            s.target_own,
                            s.target_k,
                            json_quote(&sp_of(&s.dflt, true)),
                            s.dflt_k,
                            n
                        )
                    })
                    .collect();
                format!(
                    "{{\"id\":{},\"move_id\":{},\"uses\":{},\"max_pp\":{},\"sightings\":[{}]}}",
                    json_quote(&m.id),
                    json_quote(&m.id),
                    m.uses,
                    maxpp,
                    sightings.join(",")
                )
            })
            .collect();
        format!("[{}]", rows.join(","))
    };

    let item = if own {
        // An itemless OWN mon reads the EMPTY STRING, not null: poke-env takes `item` straight
        // from the request (`""`), and `LivePokemon` only maps the `unknownitem` SENTINEL to
        // None. `""` is falsy, so `ItemsEncoder` zeroes the block either way — but the
        // read-models are compared field by field and `'' != None`.
        json_quote(&to_id(&mon.item))
    } else {
        obs.and_then(|o| o.item.clone()).map_or("null".to_string(), |i| json_quote(&i))
    };
    // `consumed_item` is the PROTOCOL's, on BOTH sides. poke-env sets it in `Pokemon.end_item`,
    // i.e. off the `|-enditem|` line, and clears it only when a truthy item is set. The engine's
    // `last_item` is not the same quantity — it survives a Trick/Knock Off that poke-env would
    // report differently, and it read `None` where poke-env had a consumed Leftovers on an own
    // mon (caught only on the SECOND fresh seed, which is why two are required).
    let consumed = obs
        .and_then(|o| o.consumed_item.clone())
        .map_or("null".to_string(), |i| json_quote(&i));
    // OURS: the engine's, which is what our own `|request|` states. THEIRS: the announcement
    // EVENTS — `view_adapter` replays poke-env's two-slot rules over them.
    // A FAINTED mon's ability is its BASE one: the sim's faint runs `clearVolatile`, which sets
    // `ability = baseAbility` (`sim/pokemon.ts` clearVolatile), and poke-env's `faint()` clears the
    // temporary slot. The port reverts a Trace only at the switch-out (`turn/switch.rs`), so a
    // Porygon2 that died holding a Traced Intimidate read `intimidate` on the replacement board.
    let ability = if own {
        let id = to_id(if mon.fainted { &mon.set.ability } else { &mon.ability });
        if id.is_empty() { "null".to_string() } else { json_quote(&id) }
    } else {
        "null".to_string()
    };
    // OURS: the set's ability — poke-env's own BASE slot (the single-ability inference or the
    // request's `baseAbility`, which agree for any legal set). The adapter replays our own
    // `ability_events` over it when a watched move's Pressure needs the ability at USE time.
    let base_ability = if own { json_quote(&to_id(&mon.set.ability)) } else { "null".to_string() };
    let ability_events = {
        let rows: Vec<String> = obs
            .map(|o| o.ability_events.as_slice())
            .unwrap_or(&[])
            .iter()
            .map(|e| {
                format!(
                    "{{\"id\":{},\"trace\":{},\"if_unknown\":{}}}",
                    json_quote(&e.id),
                    e.trace,
                    e.if_unknown
                )
            })
            .collect();
        format!("[{}]", rows.join(","))
    };

    let spread = if own {
        format!(
            "\"ivs\":[{}],\"evs\":[{}],\"nature\":{},\"spread_known\":true",
            mon.set.ivs.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","),
            mon.set.evs.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","),
            if mon.set.nature.is_empty() {
                json_quote("serious")
            } else {
                json_quote(&mon.set.nature.to_ascii_lowercase())
            },
        )
    } else {
        "\"ivs\":null,\"evs\":null,\"nature\":null,\"spread_known\":false".to_string()
    };

    // poke-env only ever learns the five non-HP stats from OUR OWN `|request|`; an opponent's
    // `stats` dict stays all-`None`, which the adapter spells as `null`s.
    let stats = if own {
        format!(
            "{{\"hp\":{},\"atk\":{},\"def\":{},\"spa\":{},\"spd\":{},\"spe\":{}}}",
            mon.stats[0], mon.stats[1], mon.stats[2], mon.stats[3], mon.stats[4], mon.stats[5]
        )
    } else {
        "null".to_string()
    };

    format!(
        "{{\"species\":{},\"active\":{},\"fainted\":{},\"revealed\":{},\
         \"hp_fraction\":{},\"current_hp\":{},\"max_hp\":{},\
         \"status\":{},\"status_counter\":{},\"protect_counter\":{},\
         \"types\":[{}],\"moves\":{},\"item\":{},\"consumed_item\":{},\"ability\":{},\"base_ability\":{},\"ability_events\":{},\
         \"boosts\":{},\"volatiles\":{},\"base_stats\":{},{},\"stats\":{}}}",
        json_quote(identity_species(mon)),
        is_active,
        mon.fainted,
        revealed,
        render_f64(hp_fraction),
        cur_hp,
        max_hp,
        status_json(mon, is_active, running),
        obs.map_or(0, |o| o.status_counter),
        obs.map_or(0, |o| o.protect_counter),
        types.join(","),
        moves,
        item,
        consumed,
        ability,
        base_ability,
        ability_events,
        boosts_json(&mon.boosts),
        volatiles_json(obs, tick),
        base_stats,
        spread,
        stats,
    )
}

/// The SPECIES id behind a protocol ident NAME, on either side. `""` when nothing matches (a
/// target the viewer never saw) — the Python side then simply has no ability for it.
fn species_of_ident(st: &BattleState, name: &str, dex: &Dex, side: usize) -> String {
    for m in &st.sides[side].pokemon {
        if display_name(m, dex) == name {
            return identity_species(m).to_string();
        }
    }
    String::new()
}

/// The mon's IDENTITY species — its own, even while TRANSFORMED. The engine's `species_id`
/// becomes the target's on Transform (the sim's `setSpecies(…, isTransform)`), but the ident, the
/// request `details` and poke-env's `Pokemon.species` all keep the mon's own (`transform()` only
/// overlays types / ability / moves / boosts / base stats). A Smeargle that Transformed into
/// Starmie read `starmie` as its row, active slot and species key — so the two roads disagreed
/// about WHICH mon was on the field (found by the procedural-generator milestone sweep).
fn identity_species(mon: &MonState) -> &str {
    mon.transform.as_ref().map_or(mon.species_id.as_str(), |t| t.base_species_id.as_str())
}

/// The mon's CURRENT types (a Conversion / Forecast override wins over the dex row).
fn live_types(mon: &MonState, dex: &Dex) -> Vec<crate::dex::Type> {
    if let Some(ts) = &mon.types_override {
        return ts.clone();
    }
    dex.species(&mon.species_id).map(|s| s.types.clone()).unwrap_or_default()
}

/// `ceil(100*hp/maxhp)`, clamped so a full-looking bar with `hp < maxhp` shows 99 — the same
/// arithmetic `bridge::hp_percent` applies to the wire, kept here rather than shared because
/// that one is private to the fold.
fn percent_hp(hp: u16, maxhp: u16) -> u32 {
    if maxhp == 0 || hp == 0 {
        return 0;
    }
    let mut pct = (100 * hp as u32).div_ceil(maxhp as u32);
    if pct == 100 && hp < maxhp {
        pct = 99;
    }
    pct
}

fn status_json(mon: &MonState, _is_active: bool, _running: bool) -> String {
    // A fainted mon reads `fnt` WHEREVER it sits. `search::slot_status_token`'s active-slot /
    // still-running conditions are about the WIRE's `condition` token, and copying them here was
    // wrong for the READ-MODEL: poke-env's `Pokemon.faint()` sets `Status.FNT` and nothing on the
    // bench clears it, so a fainted benched mon reads `fnt` for the rest of the battle. Measured
    // against poke-env on real boards (`one_sided_view_parity_integration_test`).
    if mon.fainted {
        return json_quote("fnt");
    }
    match mon.status {
        None => "null".to_string(),
        Some(Status::Burn) => json_quote("brn"),
        Some(Status::Paralysis) => json_quote("par"),
        Some(Status::Sleep(_)) => json_quote("slp"),
        Some(Status::Freeze) => json_quote("frz"),
        Some(Status::Poison) => json_quote("psn"),
        Some(Status::Toxic(_)) => json_quote("tox"),
    }
}

const BOOST_NAMES: [&str; 7] = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"];

fn boosts_json(stages: &[i8; crate::state::BOOST_LEN]) -> String {
    // `LiveView` keeps only the NONZERO stages (`{k: v for k, v in mon.boosts.items() if v}`).
    let parts: Vec<String> = stages
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != 0)
        .map(|(i, v)| format!("\"{}\":{}", BOOST_NAMES[i], v))
        .collect();
    format!("{{{}}}", parts.join(","))
}

/// The ANNOUNCED volatiles, as a JSON array of the raw protocol strings.
///
/// It is an ARRAY, not the `{id: counter}` map `LiveView.volatiles` holds, because neither the
/// id nor the counter can be computed here: the id is `Effect.from_showdown_message(raw)`
/// normalised, and the counter is poke-env's per-effect int (incremented only for the
/// action-countable effects). Both live in poke-env, so both are applied by
/// `agents.battle.view_adapter`, and this side sends the input to that function rather than a
/// guess at its output.
fn volatiles_json(obs: Option<&MonObservation>, now: u32) -> String {
    let parts: Vec<String> = obs
        .map(|o| o.volatiles.as_slice())
        .unwrap_or(&[])
        .iter()
        .map(|v| {
            let starts: Vec<String> = v.starts.iter().map(|t| t.to_string()).collect();
            format!(
                "{{\"name\":{},\"starts\":[{}],\"now\":{},\"bp_carried\":{}}}",
                json_quote(&v.name),
                starts.join(","),
                now,
                v.bp_carried
            )
        })
        .collect();
    format!("[{}]", parts.join(","))
}

/// The mon's display name — the nickname when the set carries one, else the BASE species
/// display name. Duplicated from `bridge::display_name` (private there) and pinned equal to it
/// by `tests/one_sided_view_test.rs`, because the reveal fold keys on exactly this token.
pub fn display_name(mon: &MonState, dex: &Dex) -> String {
    if !mon.set.name.is_empty() {
        return mon.set.name.clone();
    }
    dex.species(&mon.base_species_id)
        .map(|s| s.name.clone())
        .unwrap_or_else(|| mon.base_species_id.clone())
}

/// A typed Hidden Power is keyed under the BARE id in poke-env's moves dict (the live server
/// request re-keys it), so `LiveView.moves` carries `hiddenpower` — never `hiddenpowerice`.
fn bare_move_key(id: &str) -> String {
    if id.starts_with("hiddenpower") {
        "hiddenpower".to_string()
    } else {
        id.to_string()
    }
}

/// Render an f64 the way `JSON.stringify` does for the values this module emits (a ratio in
/// `[0, 1]`): shortest round-tripping form, and a whole number without a trailing `.0`.
fn render_f64(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}
