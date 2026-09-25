//! The per-decision HISTORY trackers of `agents/training/episode_tracker.py` — recency (E9 step 1),
//! pair history (Tier H-A), the 32-row event window (Tier H-B) — and the whole-log wish / sleep folds
//! of `agents/observation/{wish,sleep}_belief.py`. Each is a line-for-line port over the side's
//! READINGS (`BattleEvent`s), so the obs content they feed is byte-identical by construction; slice T
//! of the parity harness holds them equal at every decision.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::ev;
use super::turnview::classify_faint_cause;
use crate::core_error::{refuse, CoreResult, PyExc};
use crate::core_events::json_out;
use crate::core_events::reading::{implied_target, Implied};
use crate::core_events::{EventKind as K, Reading, Rel};
use crate::dex::Dex;

/// `(side, species)` — every per-mon dict key.
pub type MonKey = (Rel, String);

fn rel_rank(r: Rel) -> u8 {
    match r {
        Rel::Ours => 0,
        Rel::Opp => 1,
    }
}

/// A `MonKey`-keyed map in a stable order (the Python dicts are compared as dicts).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MonMap<V>(BTreeMap<(u8, String), V>);

impl<V: Clone> MonMap<V> {
    pub fn get(&self, k: &(Rel, &str)) -> Option<&V> {
        self.0.get(&(rel_rank(k.0), k.1.to_string()))
    }
    pub fn insert(&mut self, k: (Rel, &str), v: V) {
        self.0.insert((rel_rank(k.0), k.1.to_string()), v);
    }
    pub fn iter(&self) -> impl Iterator<Item = ((Rel, &str), &V)> {
        self.0.iter().map(|((r, s), v)| ((if *r == 0 { Rel::Ours } else { Rel::Opp }, s.as_str()), v))
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// `[[side, species, value], …]`.
fn mon_map_json<V: Clone>(m: &MonMap<V>, out: &mut String, val: impl Fn(&V, &mut String)) {
    out.push('[');
    for (i, ((r, sp), v)) in m.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('[');
        json_out::str_into(out, r.as_str());
        out.push(',');
        json_out::str_into(out, sp);
        out.push(',');
        val(v, out);
        out.push(']');
    }
    out.push(']');
}

fn int_into(out: &mut String, n: i64) {
    out.push_str(&n.to_string());
}

/// Python's `max(d.get(key, dflt), et)` write.
fn bump(m: &mut MonMap<i64>, k: (Rel, &str), dflt: i64, et: i64) {
    let cur = *m.get(&k).unwrap_or(&dflt);
    m.insert(k, cur.max(et));
}

// ---------------------------------------------------------------------------- recency

/// `RecencyTracker`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Recency {
    pub turn: i64,
    pub seen: MonMap<i64>,
    pub acted: MonMap<i64>,
    pub hit: MonMap<i64>,
}

impl Recency {
    pub fn update(&mut self, turn: i64, events: &[&Reading], our_active: Option<&str>, opp_active: Option<&str>) {
        self.turn = self.turn.max(turn);
        for e in events {
            let (Some(a), Some(side)) = (ev::actor(e), e.side) else { continue };
            let et = e.turn as i64;
            match e.kind {
                K::Move => {
                    bump(&mut self.acted, (side, a), et, et);
                    bump(&mut self.seen, (side, a), et, et);
                }
                K::Switch => bump(&mut self.seen, (side, a), et, et),
                K::Damage => bump(&mut self.hit, (side, a), et, et),
                _ => {}
            }
        }
        for (side, sp) in [(Rel::Ours, our_active), (Rel::Opp, opp_active)] {
            if let Some(sp) = ev::nz(sp) {
                bump(&mut self.seen, (side, sp), 0, self.turn);
            }
        }
    }

    pub fn json_into(&self, out: &mut String) {
        out.push_str(&format!("{{\"turn\":{},\"seen\":", self.turn));
        mon_map_json(&self.seen, out, |v, o| int_into(o, *v));
        out.push_str(",\"acted\":");
        mon_map_json(&self.acted, out, |v, o| int_into(o, *v));
        out.push_str(",\"hit\":");
        mon_map_json(&self.hit, out, |v, o| int_into(o, *v));
        out.push('}');
    }
}

// ---------------------------------------------------------------------------- pair history

/// Gen-3 damaging moves the dex records at basePower 0 (`episode_tracker._ZERO_BP_DAMAGING`).
const ZERO_BP_DAMAGING: [&str; 19] = [
    "seismictoss", "nightshade", "sonicboom", "dragonrage", "psywave", "superfang", "counter", "mirrorcoat", "bide",
    "endeavor", "return", "frustration", "flail", "reversal", "magnitude", "present", "lowkick", "spitup", "beatup",
];

/// `episode_tracker._move_is_damaging`.
pub fn move_is_damaging(move_id: Option<&str>, dex: &Dex) -> bool {
    let Some(m) = ev::nz(move_id) else { return false };
    if m.starts_with("hiddenpower") || ZERO_BP_DAMAGING.contains(&m) {
        return true;
    }
    dex.moves(m).is_some_and(|d| d.is_damaging())
}

/// One side's last action (H-A1).
#[derive(Debug, Clone, PartialEq)]
pub struct LastAction {
    pub move_id: Option<String>,
    pub was_switch: bool,
    pub missed: bool,
    pub failed: bool,
    pub crit: bool,
    pub turn: i64,
}

/// `PairHistoryTracker`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PairHistory {
    pub max_seq: i64,
    pub turn: i64,
    pub our_active: Option<String>,
    pub opp_active: Option<String>,
    /// `_last`, by side (ours, opp).
    pub last: [Option<LastAction>; 2],
    pub switch_ins: BTreeMap<(String, String), i64>,
    pub attacks: BTreeMap<(String, String), i64>,
    pub status_clicks: BTreeMap<(String, String), i64>,
    pub shared_count: BTreeMap<(String, String), i64>,
    pub shared_last_turn: BTreeMap<(String, String), i64>,
}

fn ri(r: Rel) -> usize {
    rel_rank(r) as usize
}

impl PairHistory {
    pub fn new() -> PairHistory {
        PairHistory { max_seq: -1, ..Default::default() }
    }

    fn observe_pairing(&mut self, t: i64) {
        let (Some(i), Some(j)) = (ev::nz(self.opp_active.as_deref()), ev::nz(self.our_active.as_deref())) else {
            return;
        };
        let key = (i.to_string(), j.to_string());
        let last = self.shared_last_turn.get(&key).copied();
        if last.is_none() || t > last.unwrap() {
            *self.shared_count.entry(key.clone()).or_insert(0) += 1;
            self.shared_last_turn.insert(key, t);
        }
    }

    pub fn update(&mut self, turn: i64, events: &[&Reading], our_active: Option<&str>, opp_active: Option<&str>, dex: &Dex) {
        self.turn = self.turn.max(turn);
        for e in events {
            let seq = e.seq as i64;
            if seq <= self.max_seq {
                continue;
            }
            self.max_seq = seq;
            let side = e.side;
            let sp = ev::actor(e);
            let et = e.turn as i64;
            match (e.kind, side, sp) {
                (K::Move, Some(side), Some(sp)) => {
                    self.last[ri(side)] = Some(LastAction {
                        move_id: ev::move_id(e).map(str::to_string),
                        was_switch: false,
                        missed: false,
                        failed: false,
                        crit: false,
                        turn: et,
                    });
                    if side == Rel::Opp {
                        if let Some(oa) = ev::nz(self.our_active.as_deref()) {
                            let key = (sp.to_string(), oa.to_string());
                            let d = if move_is_damaging(ev::move_id(e), dex) { &mut self.attacks } else { &mut self.status_clicks };
                            *d.entry(key).or_insert(0) += 1;
                        }
                    }
                    self.observe_pairing(et);
                }
                (K::Switch | K::Drag, Some(side), Some(sp)) => {
                    if e.kind == K::Switch && side == Rel::Opp {
                        if let Some(oa) = ev::nz(self.our_active.as_deref()) {
                            *self.switch_ins.entry((sp.to_string(), oa.to_string())).or_insert(0) += 1;
                        }
                    }
                    self.last[ri(side)] =
                        Some(LastAction { move_id: None, was_switch: true, missed: false, failed: false, crit: false, turn: et });
                    match side {
                        Rel::Ours => self.our_active = Some(sp.to_string()),
                        Rel::Opp => self.opp_active = Some(sp.to_string()),
                    }
                    self.observe_pairing(et);
                }
                (K::Miss | K::Fail | K::Crit, Some(side), _) => {
                    if let Some(la) = self.last[ri(side)].as_mut() {
                        if !la.was_switch && la.turn == et {
                            match e.kind {
                                K::Miss => la.missed = true,
                                K::Fail => la.failed = true,
                                _ => la.crit = true,
                            }
                        }
                    }
                }
                (K::Faint, Some(side), Some(sp)) => {
                    if side == Rel::Ours && self.our_active.as_deref() == Some(sp) {
                        self.our_active = None;
                    } else if side == Rel::Opp && self.opp_active.as_deref() == Some(sp) {
                        self.opp_active = None;
                    }
                }
                _ => {}
            }
        }
        if let Some(a) = ev::nz(our_active) {
            self.our_active = Some(a.to_string());
        }
        if let Some(a) = ev::nz(opp_active) {
            self.opp_active = Some(a.to_string());
        }
        self.observe_pairing(self.turn);
    }

    pub fn json_into(&self, out: &mut String) {
        out.push_str(&format!("{{\"max_seq\":{},\"turn\":{},\"our_active\":", self.max_seq, self.turn));
        json_out::opt_str_into(out, self.our_active.as_deref());
        out.push_str(",\"opp_active\":");
        json_out::opt_str_into(out, self.opp_active.as_deref());
        out.push_str(",\"last\":{");
        let mut first = true;
        for (i, r) in [Rel::Ours, Rel::Opp].iter().enumerate() {
            let Some(la) = &self.last[i] else { continue };
            if !first {
                out.push(',');
            }
            first = false;
            json_out::str_into(out, r.as_str());
            out.push_str(":{\"move_id\":");
            json_out::opt_str_into(out, la.move_id.as_deref());
            out.push_str(&format!(
                ",\"was_switch\":{},\"missed\":{},\"failed\":{},\"crit\":{},\"turn\":{}}}",
                la.was_switch, la.missed, la.failed, la.crit, la.turn
            ));
        }
        out.push('}');
        for (name, m) in [
            ("switch_ins", &self.switch_ins),
            ("attacks", &self.attacks),
            ("status_clicks", &self.status_clicks),
            ("shared_count", &self.shared_count),
            ("shared_last_turn", &self.shared_last_turn),
        ] {
            out.push_str(&format!(",\"{name}\":["));
            for (i, ((a, b), v)) in m.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push('[');
                json_out::str_into(out, a);
                out.push(',');
                json_out::str_into(out, b);
                out.push_str(&format!(",{v}]"));
            }
            out.push(']');
        }
        out.push('}');
    }
}

// ---------------------------------------------------------------------------- event window

/// `EVENT_T_*` (`agents.observation.constants`).
pub mod t {
    pub const MOVE: u8 = 1;
    pub const SWITCH_IN: u8 = 2;
    pub const FAINT: u8 = 3;
    pub const STATUS_APPLIED: u8 = 4;
    pub const STATUS_CURED: u8 = 5;
    pub const BOOST: u8 = 6;
    pub const ITEM_REVEAL: u8 = 7;
    pub const HAZARD: u8 = 8;
    pub const SWITCH_REJECTED: u8 = 9;
    pub const CANT: u8 = 10;
}

/// `ITEM_TR_*`.
pub mod item_tr {
    pub const REVEALED: u8 = 1;
    pub const CONSUMED: u8 = 2;
    pub const REMOVED: u8 = 3;
    pub const SWAPPED: u8 = 4;
}

/// `EventWindowTracker` capacity (`EVENT_WINDOW_N`).
pub const EVENT_WINDOW_N: usize = 32;

/// `_event_status_id` — CRASH on an unknown word (the `ValueError` the Python raises).
fn event_status_id(status: Option<&str>) -> CoreResult<u8> {
    let Some(s) = ev::nz(status) else { return Ok(0) };
    let key = s.trim().to_lowercase();
    if key.starts_with('[') {
        return Ok(0);
    }
    Ok(match key.as_str() {
        "brn" => 1,
        "par" => 2,
        "slp" => 3,
        "frz" => 4,
        "psn" => 5,
        "tox" => 6,
        _ => {
            return Err(refuse(
                PyExc::ValueError,
                format!("unknown status {s:?} for the H-B event window (EVENT_STATUS_IDS)"),
            ))
        }
    })
}

/// `_classify_item_transition`. A transfer move's item line is SWAPPED on BOTH mons — Trick's two
/// `|-item|`s, Thief / Covet's taker `|-item|` (`gen3_event_window_semantics_fixes_v1`, W3).
fn classify_item_transition(kind: K, from_clause: Option<&str>) -> u8 {
    let fc = from_clause.unwrap_or("").trim().to_lowercase();
    if ["trick", "thief", "covet", "switcheroo"].iter().any(|w| fc.contains(w)) {
        item_tr::SWAPPED
    } else if kind == K::Item {
        item_tr::REVEALED
    } else if fc.contains("knock off") || fc.contains("knockoff") {
        item_tr::REMOVED
    } else {
        item_tr::CONSUMED
    }
}

/// One H-B row (the Python record dict; `faint_cause` / `item_tr` / `cant` only on their types).
#[derive(Debug, Clone, PartialEq)]
pub struct EventRecord {
    pub id: u64,
    pub t: u8,
    pub actor: Option<String>,
    pub side: Option<Rel>,
    pub target: Option<String>,
    pub move_id: Option<String>,
    pub hp_delta: f64,
    pub missed: bool,
    pub failed: bool,
    pub crit: bool,
    pub eff: u8,
    pub we_first: bool,
    pub status: u8,
    pub turn: i64,
    pub forced_window: f64,
    pub faint_cause: Option<&'static str>,
    pub item_tr: Option<u8>,
    /// `Some(reason)` on a CANT row (the key is present, its value may be `None`).
    pub cant: Option<Option<String>>,
}

impl EventRecord {
    fn new(t: u8, actor: Option<&str>, side: Option<Rel>, turn: i64) -> EventRecord {
        EventRecord {
            id: 0,
            t,
            actor: actor.map(str::to_string),
            side,
            target: None,
            move_id: None,
            hp_delta: 0.0,
            missed: false,
            failed: false,
            crit: false,
            eff: 0,
            we_first: false,
            status: 0,
            turn,
            forced_window: 0.0,
            faint_cause: None,
            item_tr: None,
            cant: None,
        }
    }

    pub fn json_into(&self, out: &mut String) {
        out.push_str(&format!("{{\"t\":{},\"actor\":", self.t));
        json_out::opt_str_into(out, self.actor.as_deref());
        out.push_str(",\"side\":");
        json_out::opt_str_into(out, self.side.map(Rel::as_str));
        out.push_str(",\"target\":");
        json_out::opt_str_into(out, self.target.as_deref());
        out.push_str(",\"move_id\":");
        json_out::opt_str_into(out, self.move_id.as_deref());
        out.push_str(",\"hp_delta\":");
        json_out::f64_into(out, self.hp_delta);
        out.push_str(&format!(
            ",\"missed\":{},\"failed\":{},\"crit\":{},\"eff\":{},\"we_first\":{},\"status\":{},\"turn\":{},\"forced_window\":",
            self.missed, self.failed, self.crit, self.eff, self.we_first, self.status, self.turn
        ));
        json_out::f64_into(out, self.forced_window);
        if let Some(fc) = self.faint_cause {
            out.push_str(",\"faint_cause\":");
            json_out::str_into(out, fc);
        }
        if let Some(it) = self.item_tr {
            out.push_str(&format!(",\"item_tr\":{it}"));
        }
        if let Some(c) = &self.cant {
            out.push_str(",\"cant\":");
            json_out::opt_str_into(out, c.as_deref());
        }
        out.push('}');
    }
}

/// `EventWindowTracker`.
#[derive(Debug, Clone, PartialEq)]
pub struct EventWindow {
    pub events: VecDeque<EventRecord>,
    next_id: u64,
    pub max_seq: i64,
    pub turn: i64,
    pub our_active: Option<String>,
    pub opp_active: Option<String>,
    /// `_open_move`: side → the open MOVE record's id.
    open_move: [Option<u64>; 2],
    last_dmg_cause: [Option<Option<String>>; 2],
    /// `_last_dmg_lethal`: did that last damage take the mon to 0 HP (W2).
    last_dmg_lethal: [Option<bool>; 2],
    used_selfko: [Option<bool>; 2],
    /// `_last_mover`: the side of the latest `|move|` — a bare `-damage` is the open move's own hit
    /// only while its user is the one moving.
    last_mover: Option<Rel>,
    first_mover_turn: i64,
    first_mover_side: Option<Rel>,
    forced: [bool; 2],
}

impl Default for EventWindow {
    fn default() -> Self {
        EventWindow {
            events: VecDeque::with_capacity(EVENT_WINDOW_N),
            next_id: 0,
            max_seq: -1,
            turn: 0,
            our_active: None,
            opp_active: None,
            open_move: [None, None],
            last_dmg_cause: [None, None],
            last_dmg_lethal: [None, None],
            used_selfko: [None, None],
            last_mover: None,
            first_mover_turn: -1,
            first_mover_side: None,
            forced: [false, false],
        }
    }
}

const SELF_KO: [&str; 2] = ["explosion", "selfdestruct"];

impl EventWindow {
    fn append(&mut self, mut rec: EventRecord) -> u64 {
        rec.forced_window = if self.forced[0] || self.forced[1] { 1.0 } else { 0.0 };
        rec.id = self.next_id;
        self.next_id += 1;
        if self.events.len() == EVENT_WINDOW_N {
            self.events.pop_front();
        }
        self.events.push_back(rec);
        self.next_id - 1
    }

    /// The still-held record with this id (an evicted one is gone, as in the Python deque — a write
    /// to it changes nothing the window holds).
    fn by_id(&mut self, id: u64) -> Option<&mut EventRecord> {
        let first = self.events.front()?.id;
        if id < first {
            return None;
        }
        self.events.get_mut((id - first) as usize)
    }

    fn open(&mut self, side: Rel) -> Option<&mut EventRecord> {
        let id = self.open_move[ri(side)]?;
        self.by_id(id)
    }

    fn active(&self, side: Rel) -> Option<&str> {
        match side {
            Rel::Ours => self.our_active.as_deref(),
            Rel::Opp => self.opp_active.as_deref(),
        }
    }

    pub fn update(&mut self, turn: i64, events: &[&Reading], our_active: Option<&str>, opp_active: Option<&str>)
        -> CoreResult<()> {
        self.turn = self.turn.max(turn);
        for e in events {
            let seq = e.seq as i64;
            if seq <= self.max_seq {
                continue;
            }
            self.max_seq = seq;
            let side = e.side;
            let sp = ev::actor(e);
            let et = e.turn as i64;
            match e.kind {
                K::Move if side.is_some() && sp.is_some() => {
                    let side = side.unwrap();
                    if self.first_mover_turn != et {
                        self.first_mover_turn = et;
                        self.first_mover_side = Some(side);
                    }
                    let mut r = EventRecord::new(t::MOVE, sp, Some(side), et);
                    // gen3_move_target_class_v1: the move's dex target class (the reading's R4) —
                    // a self / side / field move is the USER's row; `adjacentAlly` has none.
                    r.target = match implied_target(ev::move_id(e), sp) {
                        Implied::User => sp.map(str::to_string),
                        Implied::None => None,
                        Implied::Foe => self.active(ev::other(side)).map(str::to_string),
                    };
                    r.move_id = ev::move_id(e).map(str::to_string);
                    r.we_first = Some(side) == self.first_mover_side;
                    let id = self.append(r);
                    self.open_move[ri(side)] = Some(id);
                    self.last_mover = Some(side);
                    self.used_selfko[ri(side)] = Some(ev::move_id(e).is_some_and(|m| SELF_KO.contains(&m)));
                }
                K::Damage if side.is_some() => {
                    let side = side.unwrap();
                    let fc = ev::from_clause(e);
                    self.last_dmg_cause[ri(side)] = Some(fc.map(str::to_string));
                    self.last_dmg_lethal[ri(side)] = Some(ev::damage_is_lethal(e));
                    let mover = ev::other(side);
                    let amt = ev::amount(e);
                    let moving = self.last_mover == Some(mover);
                    if let Some(om) = self.open(mover) {
                        if om.turn == et
                            && ev::nz(fc).is_none()
                            && moving
                            && sp.is_some()
                            && om.target.as_deref() == sp
                        {
                            if let Some(a) = amt {
                                om.hp_delta += a;
                            }
                        }
                    }
                }
                K::Miss | K::Fail | K::Crit if side.is_some() => {
                    let external = e.kind == K::Fail && !matches!(ev::from_clause(e), None | Some("move-suffix"));
                    if let Some(om) = self.open(side.unwrap()) {
                        if om.turn == et && !external {
                            match e.kind {
                                K::Miss => om.missed = true,
                                K::Fail => om.failed = true,
                                _ => om.crit = true,
                            }
                        }
                    }
                }
                K::Activate if side.is_some() && ev::is_protect_block(ev::effect(e)) => {
                    // W4: the PROTECTOR is named; the moving side's open move was stopped.
                    let mover = ev::other(side.unwrap());
                    let moving = self.last_mover == Some(mover);
                    if let Some(om) = self.open(mover) {
                        if om.turn == et && moving {
                            om.failed = true;
                        }
                    }
                }
                K::Immune | K::Resisted | K::Supereffective if side.is_some() => {
                    if let Some(om) = self.open(side.unwrap()) {
                        if om.turn == et {
                            om.eff = match e.kind {
                                K::Supereffective => 1,
                                K::Resisted => 2,
                                _ => 3,
                            };
                        }
                    }
                }
                K::Switch | K::Drag if side.is_some() && sp.is_some() => {
                    let side = side.unwrap();
                    let mut r = EventRecord::new(t::SWITCH_IN, sp, Some(side), et);
                    r.target = self.active(ev::other(side)).map(str::to_string);
                    self.append(r);
                    self.forced[ri(side)] = false;
                    match side {
                        Rel::Ours => self.our_active = sp.map(str::to_string),
                        Rel::Opp => self.opp_active = sp.map(str::to_string),
                    }
                    self.last_dmg_cause[ri(side)] = None;
                    self.last_dmg_lethal[ri(side)] = None;
                    self.used_selfko[ri(side)] = None;
                }
                K::Faint if side.is_some() && sp.is_some() => {
                    let side = side.unwrap();
                    let mut r = EventRecord::new(t::FAINT, sp, Some(side), et);
                    let fc = self.last_dmg_cause[ri(side)].clone().flatten();
                    r.faint_cause = Some(classify_faint_cause(
                        fc.as_deref(),
                        self.used_selfko[ri(side)].unwrap_or(false),
                        self.last_dmg_lethal[ri(side)].unwrap_or(false),
                    ));
                    self.append(r);
                    self.last_dmg_cause[ri(side)] = None;
                    self.last_dmg_lethal[ri(side)] = None;
                    self.used_selfko[ri(side)] = None;
                    if side == Rel::Ours && self.our_active.as_deref() == sp {
                        self.our_active = None;
                        self.forced[0] = true;
                    } else if side == Rel::Opp && self.opp_active.as_deref() == sp {
                        self.opp_active = None;
                        self.forced[1] = true;
                    }
                }
                K::Status | K::Curestatus if sp.is_some() => {
                    let tt = if e.kind == K::Status { t::STATUS_APPLIED } else { t::STATUS_CURED };
                    let mut r = EventRecord::new(tt, sp, side, et);
                    r.status = event_status_id(ev::status(e))?;
                    self.append(r);
                }
                K::Boost | K::Unboost if sp.is_some() => {
                    let amt = match ev::amount(e) {
                        Some(a) if a != 0.0 => a,
                        _ => 0.0,
                    };
                    let mut r = EventRecord::new(t::BOOST, sp, side, et);
                    // W1: the reading's `amount` is ALREADY SIGNED (an `|-unboost|` reads negative)
                    r.hp_delta = amt;
                    self.append(r);
                }
                K::Item | K::Enditem if sp.is_some() => {
                    let mut r = EventRecord::new(t::ITEM_REVEAL, sp, side, et);
                    r.item_tr = Some(classify_item_transition(e.kind, ev::from_clause(e)));
                    self.append(r);
                }
                K::Side if side.is_some() => {
                    // W5: +1 a side condition started, −1 one ended (Rapid Spin's clear, a screen's end)
                    let mut r = EventRecord::new(t::HAZARD, None, side, et);
                    r.hp_delta = if ev::s(e, "op") == Some("sideend") { -1.0 } else { 1.0 };
                    self.append(r);
                }
                K::Cant if sp.is_some() => {
                    let cs = ev::blocked_side(e).or(side);
                    let ca = ev::nz(ev::blocked_actor(e)).or(sp);
                    let mut r = EventRecord::new(t::CANT, ca, cs, et);
                    r.move_id = ev::cant_move(e).map(str::to_string);
                    r.failed = true;
                    r.cant = Some(ev::reason(e).map(str::to_string));
                    self.append(r);
                }
                K::ChoiceRejected => {
                    let mut r = EventRecord::new(t::SWITCH_REJECTED, None, Some(Rel::Ours), et);
                    r.actor = self.our_active.clone();
                    self.append(r);
                }
                _ => {}
            }
        }
        if let Some(a) = ev::nz(our_active) {
            self.our_active = Some(a.to_string());
            self.forced[0] = false;
        }
        if let Some(a) = ev::nz(opp_active) {
            self.opp_active = Some(a.to_string());
            self.forced[1] = false;
        }
        Ok(())
    }

    /// The folded records, oldest first.
    pub fn window(&self) -> impl Iterator<Item = &EventRecord> {
        self.events.iter()
    }

    pub fn json_into(&self, out: &mut String) {
        out.push_str(&format!("{{\"turn\":{},\"max_seq\":{},\"our_active\":", self.turn, self.max_seq));
        json_out::opt_str_into(out, self.our_active.as_deref());
        out.push_str(",\"opp_active\":");
        json_out::opt_str_into(out, self.opp_active.as_deref());
        out.push_str(&format!(",\"forced\":[{},{}],\"rows\":[", self.forced[0], self.forced[1]));
        for (i, r) in self.events.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            r.json_into(out);
        }
        out.push_str("]}");
    }
}

// ---------------------------------------------------------------------------- wish / sleep

/// The whole-log WISH fold (`wish_belief.build_wish_pending`), incrementally.
#[derive(Debug, Clone, PartialEq)]
pub struct WishFold {
    last_ok: [i64; 2],
    ok: [BTreeSet<i64>; 2],
}

impl Default for WishFold {
    fn default() -> Self {
        WishFold { last_ok: [-10, -10], ok: [BTreeSet::new(), BTreeSet::new()] }
    }
}

impl WishFold {
    pub fn fold(&mut self, e: &Reading) {
        if e.kind == K::Move && ev::move_id(e) == Some("wish") {
            if let Some(s) = e.side {
                let t = e.turn as i64;
                if self.last_ok[ri(s)] != t - 1 {
                    self.last_ok[ri(s)] = t;
                    self.ok[ri(s)].insert(t);
                }
            }
        }
    }
    /// `{ours, opp}` — a Wish resolves at the end of `cur_turn`.
    pub fn pending(&self, cur_turn: i64) -> [bool; 2] {
        [self.ok[0].contains(&(cur_turn - 1)), self.ok[1].contains(&(cur_turn - 1))]
    }
}

/// `sleep_belief._reason_is_rest`.
fn reason_is_rest(reason: Option<&str>) -> bool {
    let Some(r) = ev::nz(reason) else { return false };
    let r = r.to_lowercase();
    let r = match r.split_once(':') {
        Some((_, b)) => b.to_string(),
        None => r,
    };
    r.trim().replace(' ', "") == "rest"
}

/// The whole-log SLEEP-source fold (`sleep_belief.build_sleep_sources`), incrementally:
/// `(side, species) → (is_rest_sleep, sleep_usable_move_seen)` for each mon's CURRENT sleep.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SleepFold {
    slp_seq: MonMap<i64>,
    pub sources: MonMap<(bool, bool)>,
}

impl SleepFold {
    pub fn fold(&mut self, e: &Reading) {
        let (Some(side), Some(a)) = (e.side, ev::actor(e)) else { return };
        if e.kind == K::Status && ev::status(e) == Some("slp") {
            self.slp_seq.insert((side, a), e.seq as i64);
            self.sources.insert((side, a), (reason_is_rest(ev::reason(e)), false));
        } else if e.kind == K::Move && matches!(ev::move_id(e), Some("sleeptalk" | "snore")) {
            if let Some(s0) = self.slp_seq.get(&(side, a)).copied() {
                if (e.seq as i64) > s0 {
                    let rest = self.sources.get(&(side, a)).map_or(false, |v| v.0);
                    self.sources.insert((side, a), (rest, true));
                }
            }
        }
    }

    pub fn json_into(&self, out: &mut String) {
        mon_map_json(&self.sources, out, |v, o| o.push_str(&format!("[{},{}]", v.0, v.1)));
    }
}
