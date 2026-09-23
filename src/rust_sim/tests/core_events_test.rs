//! core_events_test.rs — the Rust half of the M1 gate (`gen3_core_events_v1`).
//!
//! On every battle of three committed corpora — the 22-scenario PROTOCOL capture golden (every
//! battle; built for the ambiguity-prone shapes random battles never reach: Damp's `[of]` cant,
//! the slot-less future-move `-miss`, Heal Bell's bench `-curestatus`, `[from] lockedmove`), the
//! TRAPPING bridge golden (rejected choices → `|error|` → the out-of-band CHOICE_REJECTED reading)
//! and the 1000-turn TURN-LIMIT golden — replayed through a core-recording `BridgeSession`:
//!
//! 1. **typed at the source, canonically**: every source record re-parses from its own rendering
//!    to itself (`Line::parse(render(l)) == l`), and renders the log's bytes;
//! 2. **conservation**: one source record per omniscient line;
//! 3. **the step path**: each side's shipped lines re-derive from their typed source records,
//!    byte-for-byte, with per-side conservation (`BridgeSession::core_events` refuses otherwise);
//! 4. **`parse(emit(step)) == step`**: `parse` of each side's TEXT reproduces the step path's
//!    typed lines, outcome owners and readings exactly.
//!
//! The reading's equality with `Gen3Battle` itself is the Python harness's job
//! (`src/agents/battle/rust_core_parity_test.py`).

use std::collections::BTreeMap;

use pokesim::battle::{BattleOptions, PackedTeam, PlayerOptions};
use pokesim::bridge::{bridge_opts, parse_bridge_golden, BridgeSession, Cmd, WireChoice};
use pokesim::core_events::parse::{parse, parse_matches_step};
use pokesim::core_events::{is_outcome, EventKind, Kw, Line};
use pokesim::dex::Dex;

struct Tally {
    battles: usize,
    lines: usize,
    readings: usize,
    kinds: BTreeMap<&'static str, usize>,
    owners: usize,
    truncated: usize,
}

fn check_session(label: &str, sess: &BridgeSession, t: &mut Tally) {
    let bs = sess.battle_state().expect("state");
    let recs = sess.source_recs().expect("a core session records");
    let log = bs.log.lines();
    assert_eq!(recs.len(), log.len(), "[{label}] CONSERVATION: one source record per omniscient line");
    for (i, (r, l)) in recs.iter().zip(log).enumerate() {
        assert_eq!(r.line.render(), l.0, "[{label}] source record {i} renders the log's bytes");
        assert_eq!(Line::parse(&l.0).as_ref(), Ok(&r.line), "[{label}] source record {i} is CANONICAL: {:?}", l.0);
    }
    for side in 0..2 {
        let step = sess.core_events(side).unwrap_or_else(|e| panic!("[{label}] step path p{}: {e}", side + 1));
        let text = sess.side_lines(side);
        let parsed = parse(&text, side).unwrap_or_else(|e| panic!("[{label}] parse p{}: {e}", side + 1));
        parse_matches_step(&step, &parsed).unwrap_or_else(|e| panic!("[{label}] parse != step (p{}): {e}", side + 1));
        t.lines += step.len();
        for ev in &step {
            for r in &ev.readings {
                t.readings += 1;
                *t.kinds.entry(r.kind.name()).or_default() += 1;
            }
            if is_outcome(ev.line.kw) && ev.owner.is_some() {
                t.owners += 1;
            }
        }
    }
    t.battles += 1;
}

/// A capture/fuzz golden's battles (`SCEN`/`TEAM`/`FMT`/`INIT`/`DEC` grammar: the protocol capture
/// golden and every byte-fuzz fixture), as bridge commands from the post-construction seed.
fn golden_corpus(data: &str, tag: &str) -> Vec<(String, BattleOptions, bool, Vec<Cmd>)> {
    let mut teams: BTreeMap<String, [String; 2]> = BTreeMap::new();
    let mut formats: BTreeMap<String, String> = BTreeMap::new();
    let mut out: Vec<(String, BattleOptions, bool, Vec<Cmd>)> = Vec::new();
    let mut index: BTreeMap<(String, String), usize> = BTreeMap::new();
    let mut last_init: BTreeMap<String, usize> = BTreeMap::new();
    for line in data.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "TEAM" => {
                let e = teams.entry(f[1].to_string()).or_default();
                e[if f[2] == "p1" { 0 } else { 1 }] = line.splitn(4, '\t').nth(3).unwrap().to_string();
            }
            "FMT" => {
                formats.insert(f[1].to_string(), f[2].to_string());
            }
            "INIT" => {
                let t = &teams[f[1]];
                // protocol golden: `INIT id battleNo seed`; fuzz chunk: `INIT id seed chooseSeed [qc]`.
                let (key, seed, qc) = if f[3].contains(',') {
                    (f[2].to_string(), f[3].to_string(), false)
                } else {
                    ("0".to_string(), f[2].to_string(), f.len() == 5 && f[4] == "1")
                };
                let opts = BattleOptions {
                    format_id: formats.get(f[1]).cloned().unwrap_or_else(|| "gen3customgame".to_string()),
                    seed: Some(seed),
                    p1: PlayerOptions { name: "P1".into(), team: PackedTeam(t[0].clone()) },
                    p2: PlayerOptions { name: "P2".into(), team: PackedTeam(t[1].clone()) },
                };
                index.insert((f[1].to_string(), key.clone()), out.len());
                last_init.insert(f[1].to_string(), out.len());
                out.push((format!("{tag}{}/{key}", f[1]), opts, qc, Vec::new()));
            }
            "DEC" => {
                // protocol golden: `DEC scen battleNo decNo req fp1 fp2 cp1 cp2 …`;
                // fuzz chunk: `DEC id decNo req fp1 fp2 cp1 cp2 …` (no battleNo column).
                let (i, c) = match index.get(&(f[1].to_string(), f[2].to_string())) {
                    Some(&i) if !out[i].0.contains(':') => (i, 7),
                    _ => (last_init[f[1]], 6),
                };
                for (side, tok) in [(0usize, f[c]), (1usize, f[c + 1])] {
                    if tok == "-" {
                        continue;
                    }
                    let n: usize = tok[1..].parse().unwrap();
                    let choice = if tok.starts_with('m') { WireChoice::Move(n) } else { WireChoice::Switch(n) };
                    out[i].3.push(Cmd { side, choice });
                }
            }
            _ => {}
        }
    }
    out
}

/// Feed a golden's per-decision choices: a side whose choice this boundary already ACCEPTED is
/// skipped (the golden repeats it after the other side's reject; `side.choose` keeps the first).
/// Any other bridge FATAL fails the test.
fn replay_golden(opts: &BattleOptions, qc: bool, cmds: &[Cmd], dex: &Dex, label: &str) -> BridgeSession {
    let mut sess = BridgeSession::new_core(opts, qc, dex).expect("session");
    for c in cmds {
        if sess.is_ended() {
            break;
        }
        if !sess.is_choice_done(c.side) {
            sess.feed_cmd(c.clone(), dex);
        }
        if let Some(f) = sess.fatal() {
            // The capture's blind script re-sent a rejected choice past the bridge's no-progress
            // cap (a stall loop in the capture): TRUNCATE here; everything emitted is still checked.
            assert!(f.contains("no-progress reject loop"), "[{label}] bridge fatal: {f}");
            break;
        }
    }
    sess
}

fn protocol_corpus() -> Vec<(String, BattleOptions, bool, Vec<Cmd>)> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/protocol_capture_golden.txt");
    golden_corpus(&std::fs::read_to_string(path).expect("protocol capture golden"), "")
}

/// Every byte-fuzz fixture (`tests/vectors/byte_fuzz_corpus/*.txt`) — where the four
/// ambiguity-prone shapes actually live (the protocol capture golden carries none of them).
fn byte_fuzz_corpus() -> Vec<(String, BattleOptions, bool, Vec<Cmd>)> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/byte_fuzz_corpus");
    let mut files: Vec<_> = std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("txt")).collect();
    files.sort();
    files.iter().flat_map(|p| {
        let name = p.file_stem().unwrap().to_string_lossy().to_string();
        golden_corpus(&std::fs::read_to_string(p).unwrap(), &format!("{name}:"))
    }).collect()
}

#[test]
fn the_protocol_and_byte_fuzz_corpora_round_trip_typed_and_parse_equals_step() {
    let dex = Dex::for_gen(3);
    let mut t = Tally { battles: 0, lines: 0, readings: 0, kinds: BTreeMap::new(), owners: 0, truncated: 0 };
    let protocol = protocol_corpus();
    let fuzz = byte_fuzz_corpus();
    assert!(protocol.len() >= 130, "the protocol corpus (22 scenarios)");
    assert!(fuzz.len() >= 70, "the byte-fuzz corpus");
    let mut shapes = BTreeMap::<&str, usize>::new();
    for (label, opts, qc, cmds) in protocol.iter().chain(fuzz.iter()) {
        let sess = replay_golden(opts, *qc, cmds, &dex, label);
        t.truncated += sess.fatal().is_some() as usize;
        check_session(label, &sess, &mut t);
        for l in sess.side_lines(0) {
            let line = Line::parse(&l).unwrap();
            let slotless = |i: usize| line.ident(i).map_or(false, |id| id.pos.is_none());
            if line.kw == Kw::Cant && line.of_source().is_some() {
                *shapes.entry("damp [of] cant").or_default() += 1;
            }
            if line.kw == Kw::Miss && slotless(0) {
                *shapes.entry("slot-less -miss").or_default() += 1;
            }
            if line.kw == Kw::Curestatus && slotless(0) {
                *shapes.entry("bench -curestatus").or_default() += 1;
            }
            if l.contains("[from] lockedmove") {
                *shapes.entry("lockedmove").or_default() += 1;
            }
        }
    }
    eprintln!(
        "[core events / protocol + byte-fuzz corpora] {} battles ({} truncated at a capture's reject loop), {} per-side lines, {} readings, {} owned outcome lines; shapes {:?}; kinds {:?}",
        t.battles, t.truncated, t.lines, t.readings, t.owners, shapes, t.kinds
    );
    // The four shapes Phase 0's random corpus never reached must be IN these corpora, or the gate
    // is vacuous for them.
    for s in ["damp [of] cant", "slot-less -miss", "bench -curestatus", "lockedmove"] {
        assert!(shapes.get(s).copied().unwrap_or(0) > 0, "the corpora no longer exercise {s}");
    }
    assert!(t.kinds.len() >= 25, "reading kinds exercised: {:?}", t.kinds);
}

#[test]
fn the_trapping_and_turn_limit_goldens_parse_equals_step() {
    let dex = Dex::for_gen(3);
    let mut t = Tally { battles: 0, lines: 0, readings: 0, kinds: BTreeMap::new(), owners: 0, truncated: 0 };
    let trapping = parse_bridge_golden(include_str!("vectors/bridge_trapping_golden.txt")).unwrap();
    for b in &trapping {
        let opts = bridge_opts(&b.format_id, b.seed.clone(), &b.p1_team, &b.p2_team);
        let mut sess = BridgeSession::new_core(&opts, b.quick_claw_roll, &dex).unwrap();
        sess.feed_cmds(&b.cmds, &dex);
        // The core session's per-side bytes are the golden's (it IS the production path).
        let flat = sess.chunks().flatten();
        assert_eq!(flat.p1, b.p1_expected, "[{}] core session p1 bytes", b.id);
        check_session(&b.id, &sess, &mut t);
    }
    assert!(t.kinds.get(EventKind::ChoiceRejected.name()).copied().unwrap_or(0) >= 2,
            "the trapping golden must exercise CHOICE_REJECTED: {:?}", t.kinds);
    // The turn limit: 1000 turns, the `|bigerror|` countdown, the `|message|` + tie.
    let golden = include_str!("vectors/turn_limit_golden.txt");
    let mut teams = [String::new(), String::new()];
    let mut seed = String::new();
    let mut cmds = Vec::new();
    for line in golden.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "TEAM" => teams[if f[2] == "p1" { 0 } else { 1 }] = line.splitn(4, '\t').nth(3).unwrap().to_string(),
            "INIT" => seed = f[3].to_string(),
            "CMD" => cmds.push(Cmd { side: if f[4] == "p1" { 0 } else { 1 }, choice: pokesim::bridge::parse_choice(f[5]).unwrap() }),
            _ => {}
        }
    }
    let opts = bridge_opts("gen3ou", seed, &teams[0], &teams[1]);
    let mut sess = BridgeSession::new_core(&opts, false, &dex).unwrap();
    sess.feed_cmds(&cmds, &dex);
    check_session("turn_limit", &sess, &mut t);
    assert!(sess.side_lines(0).iter().any(|l| Line::parse(l).unwrap().kw == Kw::Bigerror));
    eprintln!("[core events / trapping + turn limit] {} battles, {} lines, {} readings", t.battles, t.lines, t.readings);
}

#[test]
fn recording_changes_no_byte_of_the_production_stream() {
    // A core session and a plain one fed the same commands ship identical chunks.
    let dex = Dex::for_gen(3);
    for (label, opts, qc, cmds) in protocol_corpus().into_iter().step_by(11).chain(byte_fuzz_corpus().into_iter().step_by(7)) {
        let mut plain = BridgeSession::new_with_quick_claw(&opts, qc, &dex).unwrap();
        for c in &cmds {
            if !plain.is_choice_done(c.side) {
                plain.feed_cmd(c.clone(), &dex);
            }
        }
        let core = replay_golden(&opts, qc, &cmds, &dex, &label);
        let flat = |s: &BridgeSession| -> Vec<(usize, Vec<String>)> {
            s.chunks().chunks.iter().map(|c| (c.side, c.lines.clone())).collect()
        };
        assert_eq!(flat(&plain), flat(&core), "[{label}]");
        assert_eq!(plain.battle_state().unwrap().prng_seed(), core.battle_state().unwrap().prng_seed(), "[{label}]");
    }
}
