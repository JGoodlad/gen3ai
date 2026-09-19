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
    /// Sightings keyed by the TARGET's ident name. poke-env decrements a watched move by TWO
    /// when the target has Pressure (`_pressure_on`), and whether it does depends on the
    /// ABILITY as poke-env knows it — which includes its single-possible-ability inference (a
    /// gen-3 Zapdos reads `pressure` before anything disclosed it). So the target is carried
    /// across and the decision is made in `agents.battle.view_adapter`, against the read-model's
    /// own abilities.
    pub uses_vs: BTreeMap<String, u32>,
}

/// One entry in [`MonObservation::ability_events`]. `id` empty = the mon left the field.
#[derive(Debug, Clone, Default)]
pub struct AbilityEvent {
    pub id: String,
    pub trace: bool,
}

/// One announced volatile, with the two counters poke-env's rules need.
///
/// `turns` is `|turn|` boundaries since it was announced and `restarts` is re-announcements.
/// Neither is a poke-env value — they are the INPUTS its rules consume: `Pokemon.end_turn`
/// removes an `ends_on_turn` effect and increments an `is_turn_countable` one, while
/// `start_effect` increments an `is_action_countable` one on re-announcement. Which effect is
/// which lives in poke-env's `Effect` enum, so the decision is made in
/// `agents.battle.view_adapter` and only the raw material crosses.
#[derive(Debug, Clone, Default)]
pub struct VolObs {
    pub name: String,
    pub turns: u32,
    pub restarts: u32,
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
}

/// The per-mon half of [`SideObservation`].
#[derive(Debug, Clone, Default)]
pub struct MonObservation {
    /// poke-env's `status_counter`, folded.
    ///
    /// It is NOT the sim's counter: `Pokemon.moved` increments it once per MOVE while asleep and
    /// `Pokemon.end_turn` once per turn while badly poisoned AND ACTIVE, so a mon the engine has
    /// at `Sleep(3)` reads 0 in poke-env until it tries to act. The obs normalises it
    /// (`min(n,4)/4` asleep, `min(n,8)/8` toxic), so the difference is visible in the vector.
    pub status_counter: u32,
    /// Which counter is running: `Some(true)` = asleep, `Some(false)` = badly poisoned,
    /// `None` = neither (no status, cured, or FAINTED).
    pub counting: Option<bool>,
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
    fn start_volatile(&mut self, raw: &str) {
        let v = raw.strip_prefix("move: ").unwrap_or(raw);
        if v.is_empty() {
            return;
        }
        if let Some(existing) = self.volatiles.iter_mut().find(|x| x.name == v) {
            existing.restarts += 1;
            return;
        }
        self.volatiles.push(VolObs { name: v.to_string(), turns: 0, restarts: 0 });
    }

    fn end_volatile(&mut self, raw: &str) {
        let v = raw.strip_prefix("move: ").unwrap_or(raw);
        self.volatiles.retain(|x| x.name != v);
    }

    fn note_status(&mut self, status: &str) {
        self.status_counter = 0;
        self.counting = match status {
            "slp" => Some(true),
            "tox" => Some(false),
            _ => None,
        };
    }

    /// `Pokemon.faint()` sets `_status = Status.FNT` and leaves `_status_counter` alone, so the
    /// counter FREEZES at its last value and neither the per-turn toxic tick nor `switch_out`'s
    /// toxic reset can touch it again. Clearing the counter here instead read an own Heracross at
    /// 0 where poke-env had 1 — the reset fired when its replacement switched in.
    fn note_faint(&mut self) {
        self.counting = None;
        self.protect_counter = 0;
    }

    fn clear_status(&mut self) {
        self.status_counter = 0;
        self.counting = None;
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
            for m in self.mons.values_mut().chain(self.own_mons.values_mut()) {
                for v in m.volatiles.iter_mut() {
                    v.turns += 1;
                }
            }
            // `Pokemon.end_turn` runs for the ACTIVE mons only (`all_active_pokemons`), and the
            // only counter it advances in gen 3 is the badly-poisoned one.
            for (i, on) in [self.on_field[0].clone(), self.on_field[1].clone()].into_iter().enumerate()
            {
                let Some(nm) = on else { continue };
                let map = if i == 0 { &mut self.own_mons } else { &mut self.mons };
                if let Some(m) = map.get_mut(&nm) {
                    if m.counting == Some(false) {
                        m.status_counter += 1;
                    }
                }
            }
            return;
        }
        // `|request|{...}` — the FIRST one fixes our own team's obs slot order.
        if *tag == "request" {
            if self.own_order.is_empty() {
                self.own_order = request_roster_names(line);
            }
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

        if owner_is_self {
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
            if matches!(*tag, "switch" | "drag" | "replace") {
                self.own_seen.insert(name.clone());
                if let Some(prev) = self.on_field[0].replace(name.clone()) {
                    if prev != name {
                        let p = self.own_mons.entry(prev).or_default();
                        p.volatiles.clear();
                        switch_out_status(p);
                    }
                }
            }
            let e = self.own_mons.entry(name).or_default();
            fold_status(e, tag, &parts);
            fold_volatiles(e, tag, &parts);
            return;
        }

        match *tag {
            "switch" | "drag" | "replace" => {
                self.entry(&name);
                if let Some(prev) = self.on_field[1].replace(name.clone()) {
                    if prev != name {
                        let p = self.entry(&prev);
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
                    if move_reveals(&parts) {
                        let id = to_id(raw_move);
                        if !id.is_empty() && id != "struggle" && id != "recharge" && id != "fight"
                        {
                            let tgt = parts
                                .get(4)
                                .and_then(|t| ident_name(t))
                                .unwrap_or_default();
                            let e = self.entry(&name);
                            let slot = match e.moves.iter_mut().find(|m| m.id == id) {
                                Some(m) => m,
                                None => {
                                    e.moves.push(MoveObs {
                                        id,
                                        uses: 0,
                                        uses_vs: BTreeMap::new(),
                                    });
                                    e.moves.last_mut().expect("just pushed")
                                }
                            };
                            slot.uses += 1;
                            *slot.uses_vs.entry(tgt).or_insert(0) += 1;
                        }
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
                        .push(AbilityEvent { id: to_id(ab), trace });
                }
            }
            _ => {}
        }
        let e = self.entry(&name);
        fold_status(e, tag, &parts);
        fold_volatiles(e, tag, &parts);
    }
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
    if e.counting == Some(false) {
        e.status_counter = 0;
    }
    e.protect_counter = 0;
}

/// poke-env's `status_counter` bookkeeping, identical for both sides.
fn fold_status(e: &mut MonObservation, tag: &str, parts: &[&str]) {
    match tag {
        "-status" => {
            if let Some(st) = parts.get(3) {
                e.note_status(st.trim());
            }
        }
        "-curestatus" => e.clear_status(),
        "faint" => e.note_faint(),
        // A `|switch|` names the mon coming IN; the counter reset belongs to the one going
        // OUT and is done by the `on_field` bookkeeping in `observe` (same shape as the
        // volatile clear). Nothing to do for the entrant, whose counter is already whatever it
        // carried the last time it was on the field — poke-env keeps it.
        "switch" | "drag" | "replace" => {}
        // `Pokemon.moved` AND `Pokemon.cant_move` both do
        // `if self._status == Status.SLP: self._status_counter += 1` — a sleeping mon that is
        // TOLD it cannot move counts that turn exactly like one that acted.
        "move" | "cant" => {
            if e.counting == Some(true) {
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
fn fold_volatiles(e: &mut MonObservation, tag: &str, parts: &[&str]) {
    match tag {
        // poke-env clears every effect on switch-out and on faint. The INCOMING mon is wiped
        // too (it arrives clean); the OUTGOING one is handled by the `on_field` bookkeeping in
        // `observe`, because a `|switch|` line names only the entrant.
        "switch" | "drag" | "replace" | "faint" => e.volatiles.clear(),
        "-start" | "-activate" | "-singleturn" | "-singlemove" => {
            if let Some(raw) = parts.get(3) {
                e.start_volatile(raw);
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
                e.start_volatile("MINIMIZE");
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
pub fn one_sided_view(sess: &BridgeSession, side: usize, dex: &Dex) -> String {
    let Some(st) = sess.battle_state() else {
        return "null".to_string();
    };
    let opp = 1 - side;
    let obs = sess.observed(side);
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
    format!(
        "{{\"side\":\"p{}\",\"turn\":{},\"finished\":{},\"won\":{},\"lost\":{},\
         \"weather\":{},\"ours\":{},\"opp\":{},\"request\":{}}}",
        side + 1,
        st.turn,
        ended,
        won,
        lost,
        weather_json(st),
        side_json(st, side, side, obs, dex, !ended),
        side_json(st, opp, side, obs, dex, !ended),
        request,
    )
}

fn weather_json(st: &BattleState) -> String {
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
    let turns_active = st.turn.saturating_sub(st.field.weather_start_turn);
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
        rows.push(mon_json(st, mon, is_active, is_own, revealed, mon_obs, dex, running));
        if is_active {
            active_species = json_quote(&mon.species_id);
        }
    }
    format!(
        "{{\"team_size\":{},\"active\":{},\"side_conditions\":{},\"mons\":[{}]}}",
        sd.pokemon.len(),
        active_species,
        side_conditions_json(sd),
        rows.join(","),
    )
}

fn side_conditions_json(sd: &crate::state::SideState) -> String {
    let mut parts: Vec<String> = Vec::new();
    // poke-env keys `side_conditions` by the lowercased SideCondition enum NAME and stores
    // Spikes as a LAYER count while the timed screens store the turn they started; the four
    // below are every side condition gen 3 has that the port models.
    if sd.spikes > 0 {
        parts.push(format!("\"spikes\":{}", sd.spikes));
    }
    if sd.reflect > 0 {
        parts.push(format!("\"reflect\":{}", sd.reflect));
    }
    if sd.light_screen > 0 {
        parts.push(format!("\"light_screen\":{}", sd.light_screen));
    }
    if sd.safeguard > 0 {
        parts.push(format!("\"safeguard\":{}", sd.safeguard));
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
                let vs: Vec<String> = m
                    .uses_vs
                    .iter()
                    .filter(|(k, _)| !k.is_empty())
                    .map(|(k, n)| format!("{}:{}", json_quote(&species_of_ident(st, k, dex)), n))
                    .collect();
                format!(
                    "{{\"id\":{},\"move_id\":{},\"uses\":{},\"max_pp\":{},\"uses_vs\":{{{}}}}}",
                    json_quote(&m.id),
                    json_quote(&m.id),
                    m.uses,
                    maxpp,
                    vs.join(",")
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
    let ability = if own {
        let id = to_id(&mon.ability);
        if id.is_empty() { "null".to_string() } else { json_quote(&id) }
    } else {
        "null".to_string()
    };
    let ability_events = {
        let rows: Vec<String> = obs
            .map(|o| o.ability_events.as_slice())
            .unwrap_or(&[])
            .iter()
            .map(|e| format!("{{\"id\":{},\"trace\":{}}}", json_quote(&e.id), e.trace))
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
         \"types\":[{}],\"moves\":{},\"item\":{},\"consumed_item\":{},\"ability\":{},\"ability_events\":{},\
         \"boosts\":{},\"volatiles\":{},\"base_stats\":{},{},\"stats\":{}}}",
        json_quote(&mon.species_id),
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
        ability_events,
        boosts_json(mon),
        volatiles_json(obs),
        base_stats,
        spread,
        stats,
    )
}

/// The SPECIES id behind a protocol ident NAME, on either side. `""` when nothing matches (a
/// target the viewer never saw) — the Python side then simply has no ability for it.
fn species_of_ident(st: &BattleState, name: &str, dex: &Dex) -> String {
    for side in 0..2 {
        for m in &st.sides[side].pokemon {
            if display_name(m, dex) == name {
                return m.species_id.clone();
            }
        }
    }
    String::new()
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

fn boosts_json(mon: &MonState) -> String {
    // `LiveView` keeps only the NONZERO stages (`{k: v for k, v in mon.boosts.items() if v}`).
    let parts: Vec<String> = mon
        .boosts
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
fn volatiles_json(obs: Option<&MonObservation>) -> String {
    let parts: Vec<String> = obs
        .map(|o| o.volatiles.as_slice())
        .unwrap_or(&[])
        .iter()
        .map(|v| {
            format!(
                "{{\"name\":{},\"turns\":{},\"restarts\":{}}}",
                json_quote(&v.name),
                v.turns,
                v.restarts
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
