//! The Rust CORE's event layer (`gen3_core_events_v1`) — the Rust Core Program's milestone M1
//! (`designs/endstate/program_rust_core.md`).
//!
//! # The model
//!
//! * **Typed at the source.** Every omniscient line the engine emits is built by a typed
//!   [`crate::protocol::ProtocolBuilder`] method as a [`line::Line`] — a keyword plus typed fields
//!   (identifiers, HP integers, `[from]` causes, `[of]` sources, tags) — and the text the bridge
//!   ships is [`line::Line::render`] of it. There is ONE representation; the text cannot disagree
//!   with the typed value. With recording on, the builder also keeps a [`SourceRec`] per line: the
//!   typed line + the turn + the ENGINE's action [`Scope`] (the one attribution fact the protocol
//!   never prints). Conservation is structural: one record per committed line.
//! * **One side's stream.** A player receives ONE side's text (the privacy fold, the owner-only
//!   drops, the request frames). [`side`] derives that stream's typed lines from the source records
//!   (the typed twin of `bridge::derive_side`, and it REFUSES unless it renders the exact bytes the
//!   production bridge shipped); [`parse`] reads the same stream from text.
//! * **The reading.** [`reading::Reader`] folds a side's typed lines into [`Reading`]s — exactly the
//!   `BattleEvent`s that side's `agents.battle.gen3_battle.Gen3Battle` would record
//!   (`gen3_event_value_schema_v1` verbatim, the move-suffix synthetic MISS/FAIL included). Every
//!   place the reading is not the simulator's truth is a NAMED reading rule (`reading.rs`, R1-R8
//!   and on), each pinned to the poke-env line it mirrors.
//! * **A [`CoreEvent`]** is one per-side line with everything the core knows about it: the typed
//!   line, the index of its source record (step path), the TRUE move owner of an outcome line, and
//!   the 0..=2 readings it produced. `parse(text) == step` on every field is the M1 gate.
//! * **The record** ([`record`]) persists a side's text + the typed stream under a versioned header.

pub mod jsonval;
pub mod line;
pub mod parse;
pub mod reading;
pub mod record;
pub mod schema;
pub mod side;

pub use line::{Cause, Field, Hp, Ident, Line};
pub use schema::{EventKind, Kw};

/// How a poke-env `Player` + `Gen3Battle` routes a protocol line (generated per keyword in
/// [`schema`] from `MESSAGE_POLICY` + the player's intercepts).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// `Gen3Battle.parse_message` records an event of this kind.
    Event(EventKind),
    Control,
    Cosmetic,
    StateOnly,
    /// A non-gen-3 line: `classify` RAISES.
    Unsupported,
    /// `Player._handle_battle_message` handles it before `parse_message`.
    Intercept(Intercept),
    /// A line with no `|` (a one-element split — the player skips it).
    Plain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intercept {
    Request,
    ShowTeam,
    Win,
    Tie,
    Error,
    BigError,
    /// The player's own `MESSAGES_TO_IGNORE`.
    Ignored,
}

/// Which engine action is running when a line is emitted — the SOURCE answer to "whose move
/// owns this outcome line", which poke-env infers from the last `|move|` line (reading rule R1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Scope {
    /// Battle construction / the leads' switch-in (before any queued action).
    #[default]
    Start,
    /// A `move` (or `beforeTurnMove`) action of this side, AND a move this side's mon executes
    /// NESTED inside another action — Pursuit's strike inside the switcher's switch, a
    /// Snatch-stolen move inside the victim's move (`gen3_core_nested_move_scope_v1`: the sim's
    /// `useMoveInner` makes the nested user the active mon).
    Move(u8),
    /// A `switch` / `instaswitch` / `runSwitch` action of this side.
    Switch(u8),
    /// The end-of-turn residual action.
    Residual,
    /// `beforeTurn`, the runAction tail (faint messages) and the turn's framing.
    Other,
}

impl Scope {
    pub fn label(&self) -> String {
        match self {
            Scope::Start => "start".into(),
            Scope::Move(s) => format!("move:p{}", s + 1),
            Scope::Switch(s) => format!("switch:p{}", s + 1),
            Scope::Residual => "residual".into(),
            Scope::Other => "other".into(),
        }
    }
    /// The side whose MOVE this is, if it is one.
    pub fn move_side(&self) -> Option<u8> {
        match self {
            Scope::Move(s) => Some(*s),
            _ => None,
        }
    }
}

/// One committed OMNISCIENT line, typed at the source.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceRec {
    pub line: Line,
    /// The engine's turn counter when the line was committed.
    pub turn: u32,
    pub scope: Scope,
}

/// A `value` entry of a [`Reading`] (`BattleEvent.value`).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Int(i64),
    Float(f64),
    Str(String),
}

/// `ours` / `opp`, relative to the viewer (`BattleEvent.side`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rel {
    Ours,
    Opp,
}

impl Rel {
    pub fn as_str(self) -> &'static str {
        match self {
            Rel::Ours => "ours",
            Rel::Opp => "opp",
        }
    }
}

/// One `BattleEvent` as the viewer's `Gen3Battle` records it (`gen3_event_value_schema_v1`).
#[derive(Debug, Clone, PartialEq)]
pub struct Reading {
    pub seq: u32,
    pub turn: u32,
    pub kind: EventKind,
    pub side: Option<Rel>,
    pub actor: Option<String>,
    pub target: Option<String>,
    /// In the Python builder's insertion order.
    pub value: Vec<(&'static str, Value)>,
    /// `tuple(split_message)` of the line.
    pub raw: Vec<String>,
}

impl Reading {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.value.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
    }
    /// The schema's two halves (`gen3_event_value_schema_v1`): every required key present, no key
    /// outside required ∪ optional. `Err` names the offending key.
    pub fn check_schema(&self) -> crate::core_error::CoreResult<()> {
        let req = self.kind.required_keys();
        let opt = self.kind.optional_keys();
        for k in req {
            if self.get(k).is_none() {
                return Err(crate::core_error::fault(format!("{} lacks required key {k:?}", self.kind.name())));
            }
        }
        for (k, _) in &self.value {
            if !req.contains(k) && !opt.contains(k) {
                return Err(crate::core_error::fault(format!("{} carries undeclared key {k:?}", self.kind.name())));
            }
        }
        Ok(())
    }
}

/// One per-side protocol line with everything the core knows about it.
#[derive(Debug, Clone, PartialEq)]
pub struct CoreEvent {
    /// Position in this side's stream (0-based, every line counted).
    pub idx: u32,
    /// The typed per-side line (`line.render()` is the text the side received).
    pub line: Line,
    /// The SOURCE record this line derives from (step path only; `None` for a side-only frame —
    /// a request, an error, a format rule — and for every line read by [`parse`]).
    pub src: Option<u32>,
    /// On an OUTCOME line (`-crit`/`-miss`/`-fail`/`-notarget`/`-nothing`/`-immune`/`-resisted`/
    /// `-supereffective`): the side whose MOVE the engine was running — the sim's truth. From the
    /// engine scope on the step path, from line order on the parse path (a move opens a scope;
    /// `|switch|`, `|turn|`, `|upkeep` and the bare `|` close it; `|drag|` does not). `None` on
    /// every other line, and on an outcome line no move owns.
    pub owner: Option<u8>,
    /// The readings this line produced (0, 1, or 2 for a `|move|…|[miss]` / `[notarget]`).
    pub readings: Vec<Reading>,
}

/// Whether `kw` is an outcome keyword (the lines [`CoreEvent::owner`] is defined on).
pub fn is_outcome(kw: Kw) -> bool {
    matches!(
        kw,
        Kw::Crit | Kw::Miss | Kw::Fail | Kw::Notarget | Kw::Nothing | Kw::Immune | Kw::Resisted | Kw::Supereffective
    )
}

/// `poke_env.data.normalize.to_id_str`: `"".join(c for c in s if c.isalnum()).lower()`.
pub fn to_id(s: &str) -> String {
    // ASCII (every protocol id, name and species this side of a nickname): the same filter and
    // fold byte by byte — on ASCII, `char::is_alphanumeric` is `[0-9A-Za-z]` and `to_lowercase`
    // is the ASCII fold.
    if s.is_ascii() {
        let mut o = String::with_capacity(s.len());
        for b in s.bytes() {
            if b.is_ascii_alphanumeric() {
                o.push(b.to_ascii_lowercase() as char);
            }
        }
        return o;
    }
    s.chars().filter(|c| c.is_alphanumeric()).flat_map(|c| c.to_lowercase()).collect()
}

/// `int("".join(c for c in part if c.isdigit()))` for an ASCII-digit filter: the digits of `part`
/// read as a `u32` (what `str::parse` of the filtered run returns), `None` when there are none or
/// they overflow — read in place, no filtered copy.
pub(crate) fn int_of_digits(part: &str) -> Option<u32> {
    let mut v: Option<u32> = None;
    for c in part.bytes().filter(u8::is_ascii_digit) {
        v = Some(v.unwrap_or(0).checked_mul(10)?.checked_add((c - b'0') as u32)?);
    }
    v
}

/// Minimal JSON rendering for the core's records (std-only, like the rest of the crate).
pub mod json_out {
    use std::fmt::Write as _;

    pub fn str_into(out: &mut String, s: &str) {
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => {
                    let _ = write!(out, "\\u{:04x}", c as u32);
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }
    pub fn opt_str_into(out: &mut String, s: Option<&str>) {
        match s {
            Some(s) => str_into(out, s),
            None => out.push_str("null"),
        }
    }
    /// A float that round-trips exactly (`{:?}` is Rust's shortest round-trip form and always
    /// carries a `.` or an exponent, so a JSON reader keeps it a float).
    pub fn f64_into(out: &mut String, x: f64) {
        let _ = write!(out, "{x:?}");
    }
}

impl Reading {
    pub fn json_into(&self, out: &mut String) {
        use json_out::*;
        use std::fmt::Write as _;
        let _ = write!(out, "{{\"seq\":{},\"turn\":{},\"kind\":\"{}\",\"side\":", self.seq, self.turn, self.kind.name());
        opt_str_into(out, self.side.map(Rel::as_str));
        out.push_str(",\"actor\":");
        opt_str_into(out, self.actor.as_deref());
        out.push_str(",\"target\":");
        opt_str_into(out, self.target.as_deref());
        out.push_str(",\"value\":{");
        for (i, (k, v)) in self.value.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            str_into(out, k);
            out.push(':');
            match v {
                Value::Null => out.push_str("null"),
                Value::Int(n) => {
                    let _ = write!(out, "{n}");
                }
                Value::Float(x) => f64_into(out, *x),
                Value::Str(s) => str_into(out, s),
            }
        }
        out.push_str("},\"raw\":[");
        for (i, r) in self.raw.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            str_into(out, r);
        }
        out.push_str("]}");
    }
}

impl CoreEvent {
    pub fn json_into(&self, out: &mut String) {
        use json_out::*;
        use std::fmt::Write as _;
        let _ = write!(out, "{{\"i\":{},\"text\":", self.idx);
        str_into(out, &self.line.render());
        out.push_str(",\"src\":");
        match self.src {
            Some(s) => {
                let _ = write!(out, "{s}");
            }
            None => out.push_str("null"),
        }
        out.push_str(",\"owner\":");
        match self.owner {
            Some(s) => {
                let _ = write!(out, "{s}");
            }
            None => out.push_str("null"),
        }
        out.push_str(",\"readings\":[");
        for (i, r) in self.readings.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            r.json_into(out);
        }
        out.push_str("]}");
    }
}
