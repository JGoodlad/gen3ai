//! **SPIKE — `rust_core_phase0_2026-09-23`, not a production module.** Compiled ONLY under
//! `--features event_spike`; the default build (every production binary, `cargo test` without the
//! feature) contains none of this and none of the hooks that feed it.
//!
//! Question it answers: can a `BattleEvent`'s ATTRIBUTION be emitted by the simulator AT THE
//! SOURCE — where it is a fact, not an inference — and how does that differ from what
//! `agents.battle.gen3_battle.Gen3Battle` reads back out of the protocol text?
//!
//! Mechanism. [`crate::protocol::ProtocolBuilder`] is the single funnel every omniscient line
//! passes through. Under the feature it carries a [`Sink`]: each TYPED emit method stages a
//! [`Typed`] record built from its own typed arguments (the `MonRef`, the `Cause`, the `HpStatus`
//! integers), and `push_raw` — the one place a line is actually appended — commits exactly one
//! [`Rec`] per appended line, typed or not. So the sink is COMPLETE by construction (one record
//! per line, the conservation invariant at the source) and a line that bypasses the typed
//! methods shows up as `typed: None` rather than vanishing.
//!
//! The one fact the builder cannot see is WHICH ACTION the engine is running. The turn loop sets
//! [`Scope`] from the `QAction` it dispatches (`turn/driver.rs`), and Pursuit's in-switch strike
//! re-scopes around its nested `run_move` (`turn/switch.rs`). That is the SOURCE answer to
//! "whose move is resolving" that poke-env infers from the last `|move|` line.

use std::fmt::Write as _;

/// Which engine action is running when a line is emitted — the source fact behind every
/// "which side's move owns this outcome line" rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Scope {
    /// Battle construction / the leads' switch-in (before any queued action).
    #[default]
    Start,
    /// A `move` (or `beforeTurnMove`) action of this side, including a Pursuit strike.
    Move(usize),
    /// A `switch` / `instaswitch` / `runSwitch` action of this side.
    Switch(usize),
    /// The end-of-turn residual action.
    Residual,
    /// `beforeTurn`, and the runAction TAIL (faint messages) between actions.
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
}

/// A mon as the emit site named it: side + on-field name (the nickname the ident carries).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mon {
    pub side: usize,
    pub name: String,
}

impl Mon {
    pub fn of(r: &crate::protocol::MonRef) -> Mon {
        Mon { side: r.side, name: r.name.clone() }
    }
}

/// The `[from]` clause's text WITHOUT the `[from] ` prefix — the exact string
/// `Gen3Battle._parse_from` stores as `reason` / `from`.
pub fn cause_str(c: &crate::protocol::Cause) -> String {
    let s = c.to_string();
    s.strip_prefix("[from] ").map(str::to_string).unwrap_or(s)
}

/// The typed facts an emit site holds. Only the Phase-0 SUBSET is typed; every other line is
/// committed with `typed: None` and counted by keyword.
#[derive(Debug, Clone)]
pub enum Typed {
    /// `miss` = the announce itself carried `[miss]` (the retro-edit form is `Rec::miss_suffix`).
    Move { user: Mon, move_name: String, target: Option<Mon>, from: Option<String>, miss: bool },
    Switch { mon: Mon, details: String, hp: (u16, u16), from: Option<String> },
    Drag { mon: Mon, details: String, hp: (u16, u16) },
    Faint { mon: Mon },
    Damage { mon: Mon, hp: (u16, u16), cause: Option<String>, of: Option<Mon> },
    Heal { mon: Mon, hp: (u16, u16), cause: Option<String>, of: Option<Mon> },
    /// `crit` / `supereffective` / `resisted` / `immune` — `mon` is the DEFENDER.
    Outcome { op: &'static str, mon: Mon, cause: Option<String> },
    /// `-miss|<user>[|<target>]`; `user_raw` is set when the source rendered a slot-less ident.
    Miss { user: Option<Mon>, user_raw: Option<String>, target: Option<Mon> },
    Fail { mon: Mon, cause: Option<String> },
    Status { mon: Mon, status: String, cause: Option<String>, of: Option<Mon> },
    CureStatus { mon: Option<Mon>, ident_raw: Option<String>, status: String, cause: Option<String> },
    Cant { mon: Mon, reason: String, move_name: Option<String>, of: Option<Mon> },
}

/// One committed line.
#[derive(Debug, Clone)]
pub struct Rec {
    pub line: String,
    pub turn: u32,
    pub scope: Scope,
    pub typed: Option<Typed>,
    /// Set on a `Move` by the `attrLastMove` retro-edits.
    pub miss_suffix: bool,
    pub still_suffix: bool,
}

#[derive(Debug, Clone, Default)]
pub struct Sink {
    pub recs: Vec<Rec>,
    pub scope: Scope,
    pub turn: u32,
    pending: Option<Typed>,
    /// Records committed before the builder's last `drain()` — the builder's line index is
    /// relative to its (drained) buffer, the record index is absolute.
    base: usize,
}

impl Sink {
    /// A sink holding exactly `recs` (the reader's render path).
    pub fn from_recs(recs: Vec<Rec>) -> Sink {
        Sink { recs, ..Default::default() }
    }
    /// Stage the typed record for the NEXT committed line.
    pub fn stage(&mut self, t: Typed) {
        self.pending = Some(t);
    }
    /// A typed method whose push was suppressed (`set_enabled(false)`) must not leak its record
    /// into the next line.
    pub fn drop_pending(&mut self) {
        self.pending = None;
    }
    pub fn commit(&mut self, line: &str) {
        let typed = self.pending.take();
        self.recs.push(Rec {
            line: line.to_string(),
            turn: self.turn,
            scope: self.scope,
            typed,
            miss_suffix: false,
            still_suffix: false,
        });
    }
    /// Mirror of the builder's retro-edits: flag the record of the line at `line_idx`. The sink
    /// has exactly one record per line, so the builder's line index IS the record index.
    pub fn retro(&mut self, line_idx: usize, miss: bool, still: bool) {
        if let Some(r) = self.recs.get_mut(self.base + line_idx) {
            if miss {
                r.miss_suffix = true;
            }
            if still {
                r.still_suffix = true;
            }
        }
    }
    /// The builder drained `n` lines: later line indices restart at 0.
    pub fn drained(&mut self, n: usize) {
        self.base += n;
    }
    /// JSON lines, one per record (std-only renderer).
    pub fn to_jsonl(&self) -> String {
        let mut out = String::new();
        for r in &self.recs {
            out.push_str(&rec_json(r));
            out.push('\n');
        }
        out
    }
}

fn q(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn opt(s: &Option<String>) -> String {
    s.as_deref().map(q).unwrap_or_else(|| "null".into())
}

fn mon(m: &Mon) -> String {
    format!("{{\"side\":{},\"name\":{}}}", m.side, q(&m.name))
}

fn omon(m: &Option<Mon>) -> String {
    m.as_ref().map(mon).unwrap_or_else(|| "null".into())
}

fn hp(h: (u16, u16)) -> String {
    format!("[{},{}]", h.0, h.1)
}

fn typed_json(t: &Typed) -> String {
    match t {
        Typed::Move { user, move_name, target, from, miss } => format!(
            "{{\"k\":\"move\",\"user\":{},\"move\":{},\"target\":{},\"from\":{},\"miss\":{}}}",
            mon(user), q(move_name), omon(target), opt(from), miss
        ),
        Typed::Switch { mon: m, details, hp: h, from } => format!(
            "{{\"k\":\"switch\",\"mon\":{},\"details\":{},\"hp\":{},\"from\":{}}}",
            mon(m), q(details), hp(*h), opt(from)
        ),
        Typed::Drag { mon: m, details, hp: h } => format!(
            "{{\"k\":\"drag\",\"mon\":{},\"details\":{},\"hp\":{}}}",
            mon(m), q(details), hp(*h)
        ),
        Typed::Faint { mon: m } => format!("{{\"k\":\"faint\",\"mon\":{}}}", mon(m)),
        Typed::Damage { mon: m, hp: h, cause, of } => format!(
            "{{\"k\":\"damage\",\"mon\":{},\"hp\":{},\"cause\":{},\"of\":{}}}",
            mon(m), hp(*h), opt(cause), omon(of)
        ),
        Typed::Heal { mon: m, hp: h, cause, of } => format!(
            "{{\"k\":\"heal\",\"mon\":{},\"hp\":{},\"cause\":{},\"of\":{}}}",
            mon(m), hp(*h), opt(cause), omon(of)
        ),
        Typed::Outcome { op, mon: m, cause } => format!(
            "{{\"k\":{},\"mon\":{},\"cause\":{}}}",
            q(op), mon(m), opt(cause)
        ),
        Typed::Miss { user, user_raw, target } => format!(
            "{{\"k\":\"miss\",\"user\":{},\"user_raw\":{},\"target\":{}}}",
            omon(user), opt(user_raw), omon(target)
        ),
        Typed::Fail { mon: m, cause } => {
            format!("{{\"k\":\"fail\",\"mon\":{},\"cause\":{}}}", mon(m), opt(cause))
        }
        Typed::Status { mon: m, status, cause, of } => format!(
            "{{\"k\":\"status\",\"mon\":{},\"status\":{},\"cause\":{},\"of\":{}}}",
            mon(m), q(status), opt(cause), omon(of)
        ),
        Typed::CureStatus { mon: m, ident_raw, status, cause } => format!(
            "{{\"k\":\"curestatus\",\"mon\":{},\"ident_raw\":{},\"status\":{},\"cause\":{}}}",
            omon(m), opt(ident_raw), q(status), opt(cause)
        ),
        Typed::Cant { mon: m, reason, move_name, of } => format!(
            "{{\"k\":\"cant\",\"mon\":{},\"reason\":{},\"move\":{},\"of\":{}}}",
            mon(m), q(reason), opt(move_name), omon(of)
        ),
    }
}

fn rec_json(r: &Rec) -> String {
    format!(
        "{{\"line\":{},\"turn\":{},\"scope\":{},\"miss_suffix\":{},\"still_suffix\":{},\"typed\":{}}}",
        q(&r.line),
        r.turn,
        q(&r.scope.label()),
        r.miss_suffix,
        r.still_suffix,
        r.typed.as_ref().map(typed_json).unwrap_or_else(|| "null".into())
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_record_per_commit_and_pending_is_consumed() {
        let mut s = Sink::default();
        s.stage(Typed::Faint { mon: Mon { side: 0, name: "Blissey".into() } });
        s.commit("|faint|p1a: Blissey");
        s.commit("|upkeep");
        assert_eq!(s.recs.len(), 2);
        assert!(s.recs[0].typed.is_some());
        assert!(s.recs[1].typed.is_none(), "a staged record must not leak into the next line");
        s.stage(Typed::Fail { mon: Mon { side: 1, name: "X".into() }, cause: None });
        s.drop_pending();
        s.commit("|-fail|p2a: X");
        assert!(s.recs[2].typed.is_none(), "a suppressed push drops its staged record");
    }

    #[test]
    fn jsonl_escapes_and_renders_every_record() {
        let mut s = Sink::default();
        s.scope = Scope::Move(1);
        s.turn = 3;
        s.stage(Typed::Damage {
            mon: Mon { side: 0, name: "Mr. \"Q\"".into() },
            hp: (10, 300),
            cause: Some("psn".into()),
            of: None,
        });
        s.commit("|-damage|p1a: Mr. \"Q\"|10/300 psn|[from] psn");
        let j = s.to_jsonl();
        assert!(j.contains("\\\"Q\\\""));
        assert!(j.contains("\"scope\":\"move:p2\""));
        assert!(j.contains("\"hp\":[10,300]"));
        assert_eq!(j.lines().count(), 1);
    }
}
