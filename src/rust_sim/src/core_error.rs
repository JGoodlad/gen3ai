//! [`CoreError`] — the Rust core's error type (`gen3_core_error_v1`, the Rust Core Program's pre-M3
//! hand-off; `designs/endstate/program_rust_core.md` §2 "Before M3 starts").
//!
//! A `Result<T, String>` made a REFUSAL and a BUG indistinguishable. The core's errors are three
//! different things and are now three variants:
//!
//! * [`CoreError::Refusal`] — the input is one poke-env itself REFUSES, and the core refuses it the
//!   same way: the variant carries the Python exception CLASS poke-env / `Gen3Battle` raises
//!   ([`PyExc`]). The parity gate compares that class, not the message
//!   (`agents/battle/rust_core_present_test.py::test_refusals_raise_the_same_class`).
//! * [`CoreError::Malformed`] — input no protocol can carry (a request whose JSON does not decode,
//!   a choice that names nothing): the caller sent something broken.
//! * [`CoreError::Fault`] — the core broke one of its own invariants (a gate disagreement, a
//!   version asked for something its origin cannot do, a conservation miss): a BUG in the core.
//!
//! The rendered message ([`CoreError::message`]) is what the old `String` was, byte for byte, so
//! every existing error text (the `__ERR__` frames `sim_bridge` writes included) is unchanged.

use std::fmt;

/// The Python exception class a REFUSAL mirrors — the class poke-env / `Gen3Battle` raises on the
/// same input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyExc {
    KeyError,
    ValueError,
    IndexError,
    AssertionError,
    NotImplementedError,
    RuntimeError,
    /// `agents.battle.battle_event.UnknownMessageType` — a keyword not in `MESSAGE_POLICY`.
    UnknownMessageType,
    /// `agents.battle.battle_event.UnsupportedMessageType` — a non-gen-3 keyword.
    UnsupportedMessageType,
}

impl PyExc {
    /// The Python class name (`type(e).__name__`).
    pub fn name(self) -> &'static str {
        match self {
            PyExc::KeyError => "KeyError",
            PyExc::ValueError => "ValueError",
            PyExc::IndexError => "IndexError",
            PyExc::AssertionError => "AssertionError",
            PyExc::NotImplementedError => "NotImplementedError",
            PyExc::RuntimeError => "RuntimeError",
            PyExc::UnknownMessageType => "UnknownMessageType",
            PyExc::UnsupportedMessageType => "UnsupportedMessageType",
        }
    }
}

/// See the module docs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    /// poke-env refuses this input with `exc`; the core refuses it the same way.
    Refusal { exc: PyExc, msg: String },
    /// Input no protocol can carry.
    Malformed(String),
    /// A broken invariant inside the core — a bug.
    Fault(String),
}

/// The core's result type.
pub type CoreResult<T> = Result<T, CoreError>;

/// A refusal poke-env makes with `exc`.
pub fn refuse(exc: PyExc, msg: impl Into<String>) -> CoreError {
    CoreError::Refusal { exc, msg: msg.into() }
}

/// Malformed input.
pub fn malformed(msg: impl Into<String>) -> CoreError {
    CoreError::Malformed(msg.into())
}

/// A core fault (a bug).
pub fn fault(msg: impl Into<String>) -> CoreError {
    CoreError::Fault(msg.into())
}

impl CoreError {
    /// `refusal` / `malformed` / `fault`.
    pub fn kind(&self) -> &'static str {
        match self {
            CoreError::Refusal { .. } => "refusal",
            CoreError::Malformed(_) => "malformed",
            CoreError::Fault(_) => "fault",
        }
    }
    /// The Python class a refusal mirrors; `None` otherwise.
    pub fn class(&self) -> Option<PyExc> {
        match self {
            CoreError::Refusal { exc, .. } => Some(*exc),
            _ => None,
        }
    }
    /// The message — what the pre-`CoreError` `String` was.
    pub fn message(&self) -> &str {
        match self {
            CoreError::Refusal { msg, .. } | CoreError::Malformed(msg) | CoreError::Fault(msg) => msg,
        }
    }
    /// The same error with `prefix` prepended to its message (a location), class kept.
    pub fn context(self, prefix: impl fmt::Display) -> CoreError {
        let re = |m: String| format!("{prefix}{m}");
        match self {
            CoreError::Refusal { exc, msg } => CoreError::Refusal { exc, msg: re(msg) },
            CoreError::Malformed(m) => CoreError::Malformed(re(m)),
            CoreError::Fault(m) => CoreError::Fault(re(m)),
        }
    }
    /// `{"kind":…,"class":…|null,"message":…}` — how a core binary reports it.
    pub fn json(&self) -> String {
        let mut o = format!("{{\"kind\":\"{}\",\"class\":", self.kind());
        crate::core_events::json_out::opt_str_into(&mut o, self.class().map(PyExc::name));
        o.push_str(",\"message\":");
        crate::core_events::json_out::str_into(&mut o, self.message());
        o.push('}');
        o
    }
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for CoreError {}

/// A protocol line the schema does not know: `Gen3Battle`'s `classify` raises
/// `UnknownMessageType` on exactly these (the schema is GENERATED from `MESSAGE_POLICY`).
impl From<crate::core_events::line::LineError> for CoreError {
    fn from(e: crate::core_events::line::LineError) -> Self {
        refuse(PyExc::UnknownMessageType, e.to_string())
    }
}

/// Where the core's error meets a `String`-typed caller (the transport, the binaries' refusals): the
/// message, unchanged — so every rendered error text is what it was before `CoreError`.
impl From<CoreError> for String {
    fn from(e: CoreError) -> String {
        match e {
            CoreError::Refusal { msg, .. } | CoreError::Malformed(msg) | CoreError::Fault(msg) => msg,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::version::SideStream;

    /// Fold `lines` into a fresh p1 stream; the error its first failing line raises.
    fn err_of(lines: &[&str]) -> CoreError {
        let mut s = SideStream::new(0, "me", None).expect("stream");
        for l in lines {
            if let Err(e) = s.fold_text(l) {
                return e;
            }
        }
        panic!("no line refused: {lines:?}");
    }

    const HEAD: [&str; 3] = ["|player|p1|me||", "|player|p2|foe||", "|teamsize|p1|2"];

    fn with(tail: &str) -> Vec<&str> {
        let mut v = HEAD.to_vec();
        v.push(tail);
        v
    }

    #[test]
    fn an_unknown_keyword_is_a_refusal_poke_env_raises_as_unknown_message_type() {
        let e = err_of(&with("|-dynamaxplus|p1a: X"));
        assert_eq!((e.kind(), e.class()), ("refusal", Some(PyExc::UnknownMessageType)), "{e}");
    }

    #[test]
    fn a_non_gen3_keyword_is_unsupported_message_type() {
        let e = err_of(&with("|-mega|p1a: X|Venusaur|Venusaurite"));
        assert_eq!(e.class(), Some(PyExc::UnsupportedMessageType), "{e}");
    }

    #[test]
    fn a_gen_mismatch_is_the_runtime_error_abstract_battle_raises() {
        let e = err_of(&with("|gen|4"));
        assert_eq!(e.class(), Some(PyExc::RuntimeError), "{e}");
    }

    #[test]
    fn a_non_integer_turn_is_the_value_error_int_raises() {
        let e = err_of(&with("|turn|x"));
        assert_eq!(e.class(), Some(PyExc::ValueError), "{e}");
    }

    #[test]
    fn an_undecodable_request_is_malformed_input_not_a_refusal() {
        let e = err_of(&with("|request|{\"side\":"));
        assert_eq!((e.kind(), e.class()), ("malformed", None), "{e}");
    }

    #[test]
    fn a_broken_core_invariant_is_a_fault() {
        // A version asked to step without an engine is the CALLER misusing the core; a gate
        // disagreement is the core contradicting itself — both are faults, never refusals.
        let p = std::sync::Arc::new(crate::version::BattleVersion::parse_root(0, "me", None).unwrap());
        let Err(e) = p.fork_session() else { panic!("a parse-built version forked") };
        assert_eq!(e.kind(), "fault", "{e}");
    }

    #[test]
    fn the_message_is_the_pre_core_error_string_byte_for_byte() {
        let e = refuse(PyExc::KeyError, "team[\"p1: X\"]: KeyError").context("line 3: ");
        assert_eq!(String::from(e.clone()), "line 3: team[\"p1: X\"]: KeyError");
        assert_eq!(e.to_string(), e.message());
        assert_eq!(e.json(), "{\"kind\":\"refusal\",\"class\":\"KeyError\",\"message\":\"line 3: team[\\\"p1: X\\\"]: KeyError\"}");
    }
}
