//! `search_side_elision_test.rs` — the gate for `gen3_expand_many_side_elision_v1`.
//!
//! `expand_many` renders BOTH sides' one-sided payload (`pN_chunks`; `core_pN` on a core arm) because the
//! driver holds both boards; a search reads exactly ONE of them. Measured on 864 banked arms the
//! discarded copy was **43.0% of the reply bytes**
//! (`designs/research_state/measurements/expand_many_2026-09-22/README.md`), paid twice — once
//! rendering it here, once in the caller's `json.loads`.
//!
//! The optimisation is therefore a request field, and the ONLY property that makes it safe is
//! that **what the requested side gets back does not change by one byte**. That is what this file
//! pins, and the three negatives around it:
//!
//!   1. `side:"p1"` OMITS `p2_chunks` — and the surviving `p1_chunks` are
//!      byte-identical to the same arm expanded with no `side` at all.
//!   2. the mirror, `side:"p2"`.
//!   3. NO `side` is the historical body — every field present, in its historical order.
//!   4. an unrecognised `side` is an ERROR, never a silent fall-back to both. A typo that
//!      returned everything would read as a working elision that saved nothing, and the only
//!      symptom would be a benchmark that refused to move.
//!
//! The arms are expanded from the SAME root node in the SAME process, so the byte comparison in
//! (1)/(2) is against a genuinely identical board — a second process would reseed and the
//! comparison would be vacuous.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};

const BIN: &str = env!("CARGO_BIN_EXE_search_driver");

const P1_TEAM: &str = "Blissey|||NoAbility|tackle,headbutt|Serious|252,,252,,,|F||||]Regice|||NoAbility|tackle,headbutt|Serious|252,,252,,,|N||||";
const P2_TEAM: &str = "Blissey|||NoAbility|tackle,headbutt|Serious|252,,252,,,|F||||]Zapdos|||NoAbility|tackle,headbutt|Serious|252,,252,,,|N||||";
const SEED: &str = "1,2,3,4";

/// A record with an EMPTY command stream — enough to open the turn-1 root, which is all the
/// elision question needs (it is about rendering, not about how the board was reached).
fn record_json() -> String {
    format!(
        "{{\"v\":1,\"format_id\":\"gen3customgame\",\"prng_seed\":\"{SEED}\",\
          \"input_log\":[\"\\u003estart {{\\\"formatid\\\":\\\"gen3customgame\\\",\\\"seed\\\":\\\"{SEED}\\\"}}\",\
          \"\\u003eplayer p1 {{\\\"name\\\":\\\"P1\\\",\\\"team\\\":\\\"{p1}\\\"}}\",\
          \"\\u003eplayer p2 {{\\\"name\\\":\\\"P2\\\",\\\"team\\\":\\\"{p2}\\\"}}\"],\
          \"commands\":[]}}",
        p1 = P1_TEAM.replace('\\', "\\\\"),
        p2 = P2_TEAM.replace('\\', "\\\\"),
    )
}

struct Driver {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    seq: u32,
}

impl Driver {
    fn start() -> Driver {
        let mut child = Command::new(BIN)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn search_driver");
        let stdin = child.stdin.take().expect("stdin");
        let stdout = BufReader::new(child.stdout.take().expect("stdout"));
        Driver { child, stdin, stdout, seq: 0 }
    }

    fn call(&mut self, body: &str) -> String {
        self.seq += 1;
        writeln!(self.stdin, "{{\"id\":{},{}}}", self.seq, body).expect("write");
        let mut line = String::new();
        self.stdout.read_line(&mut line).expect("a reply");
        line
    }

    fn open_root(&mut self) {
        let rec = record_json();
        let line = self.call(&format!("\"cmd\":\"open_root\",\"record\":{rec},\"turn\":1"));
        assert!(line.contains("\"ok\":true"), "open_root must succeed: {line}");
        assert!(line.contains("\"node_id\":\"n0\""), "expected the root node n0: {line}");
    }

    /// Expand one arm from the root, optionally asking for a single side.
    fn expand(&mut self, side: Option<&str>) -> String {
        let sel = side.map_or(String::new(), |s| format!("\"side\":\"{s}\","));
        self.call(&format!(
            "\"cmd\":\"expand_many\",{sel}\"arms\":[{{\"node_id\":\"n0\",\
             \"p1_action\":\"move 1\",\"p2_action\":\"move 1\",\
             \"seed\":\"9,9,9,9\",\"label\":0}}]"
        ))
    }

    fn close(mut self) {
        let _ = self.call("\"cmd\":\"close\"");
        assert!(self.child.wait().expect("wait").success(), "close must exit 0");
    }
}

/// The substring `"<name>":<value>` up to the matching field boundary.
///
/// A deliberately dumb scan rather than a JSON parse: the claim being made is about the BYTES on
/// the wire, and a parse would normalise exactly the thing under test.
fn field(line: &str, name: &str) -> Option<String> {
    let key = format!("\"{name}\":");
    let start = line.find(&key)? + key.len();
    let bytes = line.as_bytes();
    let (open, close) = match bytes[start] {
        b'[' => (b'[', b']'),
        b'{' => (b'{', b'}'),
        _ => return None,
    };
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (i, &b) in bytes[start..].iter().enumerate() {
        if esc {
            esc = false;
            continue;
        }
        match b {
            b'\\' if in_str => esc = true,
            b'"' => in_str = !in_str,
            c if !in_str && c == open => depth += 1,
            c if !in_str && c == close => {
                depth -= 1;
                if depth == 0 {
                    return Some(line[start..start + i + 1].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

// ===========================================================================
// 1 + 2. THE SURVIVING SIDE IS BYTE-IDENTICAL, AND THE OTHER IS GONE
// ===========================================================================

#[test]
fn asking_for_p1_only_drops_p2_and_leaves_p1_byte_identical() {
    let mut d = Driver::start();
    d.open_root();
    let both = d.expand(None);
    let only = d.expand(Some("p1"));
    d.close();

    // NON-VACUITY FIRST: the un-elided reply must actually carry both sides, or the assertions
    // below would pass on an empty payload.
    let both_c1 = field(&both, "p1_chunks").expect("both: p1_chunks");
    let both_c2 = field(&both, "p2_chunks").expect("both: p2_chunks");
    assert!(both_c1.len() > 20, "the fixture must produce real chunks: {both_c1}");

    assert!(!only.contains("\"p2_chunks\":"), "side=p1 must OMIT p2_chunks: {only}");
    assert_eq!(field(&only, "p1_chunks").expect("only: p1_chunks"), both_c1,
               "the REQUESTED side's chunks must be byte-identical to the un-elided one");
    assert_ne!(both_c1, both_c2, "the two sides' chunks must differ, or this gate proves nothing");
}

#[test]
fn asking_for_p2_only_drops_p1_and_leaves_p2_byte_identical() {
    let mut d = Driver::start();
    d.open_root();
    let both = d.expand(None);
    let only = d.expand(Some("p2"));
    d.close();

    let both_c2 = field(&both, "p2_chunks").expect("both: p2_chunks");
    assert!(!only.contains("\"p1_chunks\":"), "side=p2 must OMIT p1_chunks: {only}");
    assert_eq!(field(&only, "p2_chunks").expect("only: p2_chunks"), both_c2);
}

// ===========================================================================
// 3. NO `side` IS THE HISTORICAL BODY
// ===========================================================================

#[test]
fn a_request_without_side_renders_every_field_in_its_historical_order() {
    let mut d = Driver::start();
    d.open_root();
    let both = d.expand(None);
    d.close();

    // The field ORDER is what a byte-level golden would see. The one-sided fields keep
    // their historical positions, which is what lets `search_impl_parity` (whose golden requests
    // carry no `side`) go on comparing this driver against node's field-for-field.
    let order = ["\"label\":", "\"node_id\":", "\"ended\":", "\"stuck\":", "\"outcome\":",
                 "\"requests\":", "\"choices_used\":", "\"p1_chunks\":", "\"p2_chunks\":"];
    let mut at = 0usize;
    for key in order {
        let found = both[at..].find(key)
            .unwrap_or_else(|| panic!("{key} missing or out of order in: {both}"));
        at += found + key.len();
    }
}

// ===========================================================================
// 4. THE NEGATIVE — an unrecognised side is an ERROR
// ===========================================================================

#[test]
fn an_unrecognised_side_is_refused_rather_than_silently_meaning_both() {
    let mut d = Driver::start();
    d.open_root();
    let line = d.expand(Some("p3"));
    assert!(line.contains("\"ok\":false"), "an unknown side must FAIL: {line}");
    assert!(line.contains("side must be"), "the error must name the contract: {line}");
    // Still alive: a bad request fails ONE call, never the session.
    let good = d.expand(Some("p1"));
    assert!(good.contains("\"ok\":true"), "the session must survive a refused side: {good}");
    d.close();
}
