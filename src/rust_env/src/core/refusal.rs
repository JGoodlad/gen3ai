//! THE REFUSAL POLICY — what the core does with a battle it cannot run, and how every failure
//! reaches the caller TYPED (M5 Lane 0, gate ④).
//!
//! Every error inside an env is classified ONCE, here ([`Class`]):
//!
//! | origin | class | what happens |
//! |---|---|---|
//! | ANY error the PORT returns for one battle — a `CoreError::Refusal` (a reading poke-env refuses the same way), `Malformed` or `Fault` (the port's own invariant, e.g. the HP tracker's "all candidates eliminated", F-M5-1) — or the ENGINE stopping on a mechanic it does not model (`fatal()`) | **quarantine** | the battle's INPUT LOG is BANKED (memory + `bank_dir`) with its `kind` (`refusal` / `malformed` / `fault` / `engine`) and Python class, the env reads `done = 1`, `refused = 1`, `reward = 0` and starts its next episode; the batch SUCCEEDS. Over the declared `refusal_budget` it is a `BUDGET` failure instead. Why a port FAULT quarantines too: it is confined to one battle's discarded state, and it is COUNTED, BANKED replayable and BUDGETED, never absorbed silently |
//! | the ENV CORE's own invariant (an alignment break — one write opening two decisions, a request that is not the write's last line —, a stuck env, a lost chain) | `FAULT` | the batch fails, the input log banked, the pool POISONED: the pool machinery itself is wrong |
//! | the CALLER's input (an action the mask forbids, a team index outside the table, a bad seed) | `CALLER` | the batch fails, the pool poisoned |
//! | a panic | `PANIC` | caught at the env; the batch fails, the input log banked, the pool poisoned |
//!
//! A poisoned pool refuses every later op (`LIFECYCLE`): after a batch failure an env's state is
//! not known to be whole, and a silent continuation is exactly what the declared lifecycle forbids.
//!
//! The INPUT LOG is the battle as a `sim_bridge` / `core_events` script — `START <json>` then every
//! command the env fed, in order (`CHOOSE pN <tok>`, `FORCELOSE p1`) — so a banked battle re-runs
//! alone through either binary, with no pool around it.

use std::path::Path;

/// A JSON string literal (the mandatory escapes; std-only).
pub fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// One battle's inputs, enough to re-run it alone.
#[derive(Clone, Debug, Default)]
pub struct InputLog {
    pub format_id: String,
    pub seed: String,
    pub names: [String; 2],
    pub teams: [String; 2],
    /// The command lines after START, verbatim.
    pub cmds: Vec<String>,
}

impl InputLog {
    /// The START JSON; `extra` is spliced in first (e.g. a `core_obs` key, `"persistent":true,`).
    pub fn start_json(&self, extra: &str) -> String {
        format!(
            "{{{extra}\"formatid\":{},\"seed\":{},\"p1\":{{\"name\":{},\"team\":{}}},\"p2\":{{\"name\":{},\"team\":{}}}}}",
            json_str(&self.format_id),
            json_str(&self.seed),
            json_str(&self.names[0]),
            json_str(&self.teams[0]),
            json_str(&self.names[1]),
            json_str(&self.teams[1]),
        )
    }

    /// The script: `START <json>` + one command per line (no trailing `END`; the reader appends it).
    pub fn script(&self, extra: &str) -> String {
        let mut s = format!("START {}\n", self.start_json(extra));
        for c in &self.cmds {
            s.push_str(c);
            s.push('\n');
        }
        s
    }
}

/// How an env's error is handled (see the module table).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// Quarantine the battle; the batch goes on.
    Quarantine,
    /// `status::FAULT`.
    Fault,
    /// `status::CALLER`.
    Caller,
    /// `status::PANIC`.
    Panic,
}

/// An env's error, classified at its origin.
#[derive(Clone, Debug)]
pub struct EnvError {
    pub class: Class,
    /// `refusal` / `engine` / `malformed` / `fault` / `caller` / `panic`.
    pub kind: &'static str,
    /// The Python exception class a port refusal mirrors (`PyExc::name`), when there is one.
    pub py_class: Option<&'static str>,
    pub message: String,
}

impl EnvError {
    pub fn caller(msg: impl Into<String>) -> EnvError {
        EnvError { class: Class::Caller, kind: "caller", py_class: None, message: msg.into() }
    }
    pub fn fault(msg: impl Into<String>) -> EnvError {
        EnvError { class: Class::Fault, kind: "fault", py_class: None, message: msg.into() }
    }
    pub fn engine(msg: impl Into<String>) -> EnvError {
        EnvError { class: Class::Quarantine, kind: "engine", py_class: None, message: msg.into() }
    }
    pub fn panic(msg: impl Into<String>) -> EnvError {
        EnvError { class: Class::Panic, kind: "panic", py_class: None, message: msg.into() }
    }
    /// A `CoreError` from the port, with a location prefix. EVERY port error QUARANTINES: it is
    /// confined to one battle's state (its parse chain / engine), which the quarantine discards.
    /// The variant rides along as `kind` (`refusal` / `malformed` / `fault`) and a `Refusal`'s
    /// Python class as `py_class`, so a banked port FAULT (a port bug) stays distinguishable.
    pub fn from_core(e: pokesim::core_error::CoreError, at: &str) -> EnvError {
        use pokesim::core_error::CoreError as E;
        let message = format!("{at}: {}", e.message());
        let (kind, py_class) = match &e {
            E::Refusal { exc, .. } => ("refusal", Some(exc.name())),
            E::Malformed(_) => ("malformed", None),
            E::Fault(_) => ("fault", None),
        };
        EnvError { class: Class::Quarantine, kind, py_class, message }
    }
}

/// A banked battle.
#[derive(Clone, Debug)]
pub struct Banked {
    pub env: usize,
    pub episode: u32,
    pub error: EnvError,
    pub log: InputLog,
}

impl Banked {
    /// `{"env":…,"episode":…,"kind":…,"class":…|null,"message":…,"script":…}` — what a front end
    /// hands the Python side (`protocol.py`'s classes carry `script`).
    pub fn json(&self) -> String {
        format!(
            "{{\"env\":{},\"episode\":{},\"kind\":{},\"class\":{},\"message\":{},\"script\":{}}}",
            self.env,
            self.episode,
            json_str(self.error.kind),
            self.error.py_class.map_or("null".to_string(), json_str),
            json_str(&self.error.message),
            json_str(&self.log.script("")),
        )
    }

    /// Write the record to `<dir>/env<E>_ep<K>_<kind>.json` + the replayable `.script`.
    pub fn write_to(&self, dir: &Path) -> std::io::Result<()> {
        let stem = format!("env{:04}_ep{:06}_{}", self.env, self.episode, self.error.kind);
        std::fs::write(dir.join(format!("{stem}.json")), self.json())?;
        std::fs::write(dir.join(format!("{stem}.script")), self.log.script("") + "END\n")
    }
}

/// The QUARANTINE BANK: capacity reserved at startup = the refusal budget (the declared lifecycle:
/// it never grows).
pub struct Bank {
    items: Vec<Banked>,
    budget: usize,
    /// Reallocations observed after startup (must stay 0: `BANK_GROWTH_AFTER_FREEZE`).
    pub growth: u64,
}

impl Bank {
    pub fn new(budget: usize) -> Bank {
        Bank { items: Vec::with_capacity(budget), budget, growth: 0 }
    }
    /// Bank one quarantine; `Err` when it would exceed the budget (the caller turns that into
    /// `status::BUDGET`).
    pub fn push(&mut self, b: Banked) -> Result<(), Banked> {
        if self.items.len() >= self.budget {
            return Err(b);
        }
        let cap = self.items.capacity();
        self.items.push(b);
        if self.items.capacity() != cap {
            self.growth += 1;
        }
        Ok(())
    }
    pub fn items(&self) -> &[Banked] {
        &self.items
    }
    pub fn len(&self) -> usize {
        self.items.len()
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_str_escapes_what_json_needs() {
        assert_eq!(json_str("a\"b\\c\nd\u{1}"), "\"a\\\"b\\\\c\\nd\\u0001\"");
    }

    #[test]
    fn the_bank_never_grows_past_its_reservation() {
        let mut bank = Bank::new(2);
        let b = || Banked { env: 0, episode: 0, error: EnvError::engine("x"), log: InputLog::default() };
        assert!(bank.push(b()).is_ok());
        assert!(bank.push(b()).is_ok());
        assert!(bank.push(b()).is_err(), "the budget is a hard cap");
        assert_eq!(bank.growth, 0);
        assert_eq!(bank.len(), 2);
        let mut none = Bank::new(0);
        assert!(none.push(b()).is_err(), "a zero budget quarantines nothing");
    }
}
