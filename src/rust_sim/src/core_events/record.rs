//! The persisted event RECORD (`gen3_core_event_v1`) — `program_rust_core.md` M1, "The persisted
//! RECORD".
//!
//! A record is ONE side's stream as JSON lines: a header, then one line per protocol line of that
//! side, each carrying the line's TEXT (the authority, §6b) and the typed stream the core derived
//! from it (the outcome owner and the readings). The typed LINE itself is not stored twice: it is
//! [`Line::parse`] of the stored text, exactly (a gated property), so a reader re-derives it — the
//! text is the one representation.
//!
//! ```text
//! {"record":"gen3ai_core_record","event_schema":"gen3_core_event_v1","core_commit":"<sha>","path":"step","viewer":"p1","format":"gen3ou","showdown_version":null,"battle":"<label>","lines":N}
//! {"i":0,"text":"|t:|<NORMALIZED>","owner":null,"readings":[]}
//! …
//! ```
//!
//! * **Read only through the core.** [`read`] REFUSES an unknown `record` kind or `event_schema`
//!   (it never guesses), a line count that disagrees with the header, and a text that no longer
//!   parses. Floats are written in Rust's shortest round-trip form (always with a `.` or an
//!   exponent), so a JSON reader keeps each value's int/float type and its exact value.
//! * **Byte-stable.** `write(read(bytes)) == bytes` for every record [`write`] produced.
//! * **Migrate by re-parse.** A schema change bumps [`EVENT_SCHEMA`] and ships a migration that
//!   re-derives the typed stream from the stored text ([`reparse`]) — old records are re-parsed,
//!   never hand-edited. [`check_reparse`] is the golden gate's "stored text → stored stream".
//! * **Scope.** Only persistence points pay for it (an eval game's end, an online game, a prober
//!   export). A search successor is never serialised.

use super::line::Line;
use super::parse::parse;
use super::schema::EventKind;
use super::jsonval::Val;
use super::{json_out, CoreEvent, Reading, Rel, Value};
use crate::core_error::{fault, malformed, CoreError, CoreResult};

/// The record kind tag.
pub const RECORD_KIND: &str = "gen3ai_core_record";
/// The typed-stream schema this build writes and reads. A change to the reading projection's
/// shape (a key, a kind, a rule) bumps this and ships a [`reparse`] migration.
pub const EVENT_SCHEMA: &str = "gen3_core_event_v1";

/// Which path produced the typed stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Path {
    /// The simulator's own transition (typed at the source).
    Step,
    /// `parse` of text a server sent.
    Parse,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Header {
    pub event_schema: String,
    pub core_commit: String,
    pub path: Path,
    /// 0 = p1, 1 = p2.
    pub viewer: u8,
    pub format: String,
    /// The Showdown version, when a real server produced the text (`None` for our simulator).
    pub showdown_version: Option<String>,
    pub battle: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    pub header: Header,
    /// One per line of the side's stream (`src` is always `None` in a record: the source link
    /// is a step-internal index, not part of the persisted stream).
    pub events: Vec<CoreEvent>,
}

impl Record {
    /// A record of `events` (a step or parse path's output) under `header`.
    pub fn new(header: Header, events: &[CoreEvent]) -> Record {
        let events = events.iter().map(|e| CoreEvent { src: None, ..e.clone() }).collect();
        Record { header, events }
    }
}

/// Serialize a record (deterministic: the same record always writes the same bytes).
pub fn write(r: &Record) -> String {
    let h = &r.header;
    let mut out = String::new();
    out.push_str("{\"record\":");
    json_out::str_into(&mut out, RECORD_KIND);
    out.push_str(",\"event_schema\":");
    json_out::str_into(&mut out, &h.event_schema);
    out.push_str(",\"core_commit\":");
    json_out::str_into(&mut out, &h.core_commit);
    out.push_str(",\"path\":");
    json_out::str_into(&mut out, if h.path == Path::Step { "step" } else { "parse" });
    out.push_str(",\"viewer\":");
    json_out::str_into(&mut out, if h.viewer == 0 { "p1" } else { "p2" });
    out.push_str(",\"format\":");
    json_out::str_into(&mut out, &h.format);
    out.push_str(",\"showdown_version\":");
    json_out::opt_str_into(&mut out, h.showdown_version.as_deref());
    out.push_str(",\"battle\":");
    json_out::str_into(&mut out, &h.battle);
    out.push_str(&format!(",\"lines\":{}}}\n", r.events.len()));
    for e in &r.events {
        out.push_str(&format!("{{\"i\":{},\"text\":", e.idx));
        json_out::str_into(&mut out, &e.line.render());
        out.push_str(",\"owner\":");
        match e.owner {
            Some(o) => out.push_str(&o.to_string()),
            None => out.push_str("null"),
        }
        out.push_str(",\"readings\":[");
        for (i, rd) in e.readings.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            rd.json_into(&mut out);
        }
        out.push_str("]}\n");
    }
    out
}

/// Read a record. REFUSES an unknown record kind / `event_schema`, a malformed line, a line
/// count that disagrees with the header, and a stored text that no longer parses.
pub fn read(text: &str) -> CoreResult<Record> {
    let mut lines = text.lines();
    let head = lines.next().ok_or_else(|| malformed("empty record"))?;
    let h = Val::parse(head).map_err(|e| malformed(format!("header: {e}")))?;
    let kind = h.str_at("record").ok_or_else(|| malformed("header: no `record` kind"))?;
    if kind != RECORD_KIND {
        return Err(malformed(format!("not a core record (`record` = {kind:?})")));
    }
    let schema = h.str_at("event_schema").ok_or_else(|| malformed("header: no `event_schema`"))?;
    if schema != EVENT_SCHEMA {
        return Err(malformed(format!(
            "REFUSED: event_schema {schema:?} is not this reader's {EVENT_SCHEMA:?} — migrate the record by re-parsing its text, never guess"
        )));
    }
    let path = match h.str_at("path") {
        Some("step") => Path::Step,
        Some("parse") => Path::Parse,
        other => return Err(malformed(format!("header: bad path {other:?}"))),
    };
    let viewer = match h.str_at("viewer") {
        Some("p1") => 0,
        Some("p2") => 1,
        other => return Err(malformed(format!("header: bad viewer {other:?}"))),
    };
    let header = Header {
        event_schema: schema.to_string(),
        core_commit: h.str_at("core_commit").ok_or_else(|| malformed("header: no core_commit"))?.to_string(),
        path,
        viewer,
        format: h.str_at("format").ok_or_else(|| malformed("header: no format"))?.to_string(),
        showdown_version: match h.get("showdown_version") {
            Some(Val::Str(s)) => Some(s.clone()),
            Some(Val::Null) | None => None,
            Some(v) => return Err(malformed(format!("header: bad showdown_version {v:?}"))),
        },
        battle: h.str_at("battle").ok_or_else(|| malformed("header: no battle"))?.to_string(),
    };
    let n = match h.get("lines") {
        Some(Val::Int(n)) => *n as usize,
        _ => return Err(malformed("header: no line count")),
    };
    let mut events = Vec::with_capacity(n);
    for (k, l) in lines.enumerate() {
        let v = Val::parse(l).map_err(|e| malformed(format!("line {k}: {e}")))?;
        let idx = match v.get("i") {
            Some(Val::Int(i)) if *i as usize == k => k as u32,
            other => return Err(malformed(format!("line {k}: bad index {other:?}"))),
        };
        let text = v.str_at("text").ok_or_else(|| malformed(format!("line {k}: no text")))?;
        let line = Line::parse(text).map_err(|e| CoreError::from(e).context(format!("line {k}: ")))?;
        let owner = match v.get("owner") {
            Some(Val::Int(o)) if *o == 0 || *o == 1 => Some(*o as u8),
            Some(Val::Null) => None,
            other => return Err(malformed(format!("line {k}: bad owner {other:?}"))),
        };
        let readings = match v.get("readings") {
            Some(Val::Arr(rs)) => rs.iter().map(reading_of).collect::<Result<Vec<_>, _>>().map_err(|e| CoreError::from(e).context(format!("line {k}: ")))?,
            _ => return Err(malformed(format!("line {k}: no readings"))),
        };
        events.push(CoreEvent { idx, line, src: None, owner, readings });
    }
    if events.len() != n {
        return Err(malformed(format!("header says {n} lines, the record has {}", events.len())));
    }
    Ok(Record { header, events })
}

/// Re-derive the typed stream from the stored text — the migration (and the parse path).
pub fn reparse(r: &Record) -> CoreResult<Record> {
    let texts: Vec<String> = r.events.iter().map(|e| e.line.render()).collect();
    let events = parse(&texts, r.header.viewer as usize)?;
    Ok(Record { header: Header { event_schema: EVENT_SCHEMA.into(), ..r.header.clone() }, events })
}

/// The golden gate: the stored text re-parses to the STORED typed stream, exactly.
pub fn check_reparse(r: &Record) -> CoreResult<()> {
    let again = reparse(r)?;
    for (a, b) in r.events.iter().zip(&again.events) {
        if a != b {
            return Err(fault(format!("line {}: stored {:?} vs re-parsed {:?}", a.idx, a, b)));
        }
    }
    Ok(())
}

fn kind_of(name: &str) -> CoreResult<EventKind> {
    use EventKind as K;
    const ALL: [EventKind; 36] = [
        K::Move, K::Switch, K::Drag, K::Faint, K::Damage, K::Heal, K::Boost, K::Unboost, K::Setboost,
        K::Clearboost, K::Status, K::Curestatus, K::Cant, K::Crit, K::Miss, K::Fail, K::Immune, K::Resisted,
        K::Supereffective, K::Item, K::Enditem, K::Ability, K::Weather, K::Field, K::Side, K::VolatileStart,
        K::VolatileEnd, K::Activate, K::Prepare, K::Mustrecharge, K::Transform, K::Formechange, K::Swap,
        K::Sethp, K::ChoiceRejected, K::Unknown,
    ];
    ALL.iter().copied().find(|k| k.name() == name).ok_or_else(|| malformed(format!("unknown event kind {name:?}")))
}

fn reading_of(v: &Val) -> CoreResult<Reading> {
    let kind = kind_of(v.str_at("kind").ok_or_else(|| malformed("reading: no kind"))?)?;
    let opt = |k: &str| -> CoreResult<Option<String>> {
        match v.get(k) {
            Some(Val::Str(s)) => Ok(Some(s.clone())),
            Some(Val::Null) | None => Ok(None),
            other => Err(malformed(format!("reading {k}: {other:?}"))),
        }
    };
    let side = match opt("side")?.as_deref() {
        Some("ours") => Some(Rel::Ours),
        Some("opp") => Some(Rel::Opp),
        None => None,
        Some(o) => return Err(malformed(format!("reading side {o:?}"))),
    };
    let int = |k: &str| -> CoreResult<u32> {
        match v.get(k) {
            Some(Val::Int(n)) => Ok(*n as u32),
            other => Err(malformed(format!("reading {k}: {other:?}"))),
        }
    };
    let mut value = Vec::new();
    match v.get("value") {
        Some(Val::Obj(kv)) => {
            for (k, x) in kv {
                // Map the key onto the schema's own `'static` vocabulary (and refuse any other).
                let key = kind
                    .required_keys()
                    .iter()
                    .chain(kind.optional_keys().iter())
                    .find(|d| **d == k.as_str())
                    .copied()
                    .ok_or_else(|| malformed(format!("{} carries undeclared key {k:?}", kind.name())))?;
                let val = match x {
                    Val::Null => Value::Null,
                    Val::Int(n) => Value::Int(*n),
                    Val::Float(f) => Value::Float(*f),
                    Val::Str(s) => Value::Str(s.clone()),
                    other => return Err(malformed(format!("value {k}: {other:?}"))),
                };
                value.push((key, val));
            }
        }
        _ => return Err(malformed("reading: no value")),
    }
    let raw = match v.get("raw") {
        Some(Val::Arr(a)) => a
            .iter()
            .map(|x| match x {
                Val::Str(s) => Ok(s.clone()),
                o => Err(malformed(format!("raw: {o:?}"))),
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err(malformed("reading: no raw")),
    };
    Ok(Reading { seq: int("seq")?, turn: int("turn")?, kind, side, actor: opt("actor")?, target: opt("target")?, value, raw })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Record {
        let lines = [
            "|player|p1|me||",
            "|player|p2|foe||",
            "|switch|p1a: Metagross|Metagross|301/301",
            "|switch|p2a: Gyarados|Gyarados, M|100/100",
            "|turn|1",
            "|move|p1a: Metagross|Meteor Mash|p2a: Gyarados",
            "|-damage|p2a: Gyarados|54/100",
            "|-damage|p1a: Metagross|280/301|[from] Recoil|[of] p2a: Gyarados",
        ];
        let events = parse(&lines, 0).unwrap();
        Record::new(
            Header {
                event_schema: EVENT_SCHEMA.into(),
                core_commit: "abc".into(),
                path: Path::Parse,
                viewer: 0,
                format: "gen3ou".into(),
                showdown_version: None,
                battle: "unit \"quoted\"".into(),
            },
            &events,
        )
    }

    #[test]
    fn a_record_round_trips_byte_identically_and_reparses_from_its_text() {
        let r = sample();
        let bytes = write(&r);
        let back = read(&bytes).unwrap();
        assert_eq!(back, r);
        assert_eq!(write(&back), bytes, "write(read(bytes)) == bytes");
        check_reparse(&back).unwrap();
        // A float value keeps its type and exact value (-0.46 on a percent-HP damage line).
        assert!(bytes.contains("\"amount\":-0.45999999999999996"));
    }

    #[test]
    fn an_unknown_schema_is_refused_not_guessed() {
        let bytes = write(&sample()).replacen(EVENT_SCHEMA, "gen3_core_event_v0", 1);
        let err = read(&bytes).unwrap_err();
        assert!(err.message().contains("REFUSED"), "{err}");
        // An unknown schema is input this reader cannot carry — MALFORMED, not a core fault.
        assert_eq!(err.kind(), "malformed");
    }

    #[test]
    fn a_tampered_stream_fails_the_reparse_gate() {
        let bytes = write(&sample()).replacen("\"hp_after\":0.54", "\"hp_after\":0.55", 1);
        let r = read(&bytes).unwrap();
        assert!(check_reparse(&r).is_err());
    }
}
