//! `BattleStream::write_line` — the STREAMING drop-in surface's PER-WRITE byte
//! differential gate (`gen3_writeline_stream_v1`).
//!
//! `harness/gen_writeline_capture.js` drives the REAL Node `BattleStream` with the
//! full protocol-scenario corpus (19 scenarios × 2 seeds = 38 battles), writing every
//! command SEPARATELY (`>start`, `>player p1`, `>player p2`, then each side's choice
//! as its own `>pN …` write — the bridge's streaming pattern) and recording the
//! omniscient lines flushed after EACH write. This test replays the IDENTICAL command
//! stream through the Rust [`pokesim::battle::BattleStream`] and asserts, PER WRITE,
//! that the flushed chunks are byte-equal after the protocol gate's filtering
//! discipline (only the gated line types compared — `debug`/`error`/`-message` etc.
//! dropped from both sides; `|t:|` position-asserted, value-normalized).
//!
//! # Honesty (scope)
//!
//! - The `>start` seed convention: the Rust surface replays from the recorded
//!   pre-first-decision `initSeed` (the sim's turn-0 construction window — gender
//!   samples, the turn-0 Quick Claw — is deliberately unmodeled; same convention as
//!   `protocol_test`). The harness records `initSeed` in the INIT row; this test
//!   BUILDS the Rust `>start` from it while every other command replays VERBATIM.
//! - `|request|` / `sideupdate` frames + the per-side privacy fold are OUT of scope
//!   (the OMNISCIENT log stream is the drop-in deliverable).
//! - A battle the capture TRUNCATED (safety cap, no `|win|`) is compared for every
//!   write the capture recorded — the same prefix discipline as protocol_test.

use pokesim::battle::BattleStream;
use std::collections::{BTreeMap, BTreeSet};

/// The gated line types (the protocol_test Phase-3 filter — keep in lockstep).
fn gated_types() -> BTreeSet<&'static str> {
    [
        "t:", "gametype", "player", "gen", "tier", "rule", "teamsize", "start", "",
        "turn", "upkeep",
        "move", "switch", "drag", "faint", "cant",
        "-damage", "-heal",
        "-crit", "-supereffective", "-resisted", "-immune", "-miss", "-fail",
        "-status", "-curestatus",
        "-boost", "-unboost", "-weather", "-ability",
        "-start", "-end", "-activate", "-singleturn", "-sidestart",
        "-enditem", "-hint", "-nothing", "-fieldactivate",
        "win", "tie",
    ]
    .into_iter()
    .collect()
}

fn line_type(raw: &str) -> &str {
    raw.strip_prefix('|').map_or(raw, |rest| rest.split('|').next().unwrap_or(""))
}

fn filter(lines: &[String], types: &BTreeSet<&'static str>) -> Vec<String> {
    lines
        .iter()
        .filter(|l| types.contains(line_type(l)))
        .map(|l| if l.starts_with("|t:|") { "|t:|<NORMALIZED>".to_string() } else { l.clone() })
        .collect()
}

#[derive(Debug, Clone)]
struct WriteRec {
    command: String,
    lines: Vec<String>,
}

#[derive(Debug, Clone)]
struct BattleCase {
    scen: String,
    battle_no: u32,
    init_seed: String,
    writes: Vec<WriteRec>,
}

fn parse_golden() -> Vec<BattleCase> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/writeline_capture_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing writeline capture golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_writeline_capture.js")
    });
    let mut cases: Vec<BattleCase> = Vec::new();
    let mut index: BTreeMap<(String, u32), usize> = BTreeMap::new();
    for line in data.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.splitn(5, '\t').collect();
        match f[0] {
            "SCEN" | "TEAM" | "END" => {}
            "INIT" => {
                let key = (f[1].to_string(), f[2].parse().unwrap());
                index.insert(key.clone(), cases.len());
                cases.push(BattleCase {
                    scen: key.0,
                    battle_no: key.1,
                    init_seed: f[3].to_string(),
                    writes: Vec::new(),
                });
            }
            "W" => {
                // W <scen> <n> <writeNo> <command> — the command is the LAST field
                // (splitn(5) keeps embedded tabs intact; commands have none, but the
                // JSON payloads may contain pipes).
                let idx = index[&(f[1].to_string(), f[2].parse().unwrap())];
                cases[idx].writes.push(WriteRec { command: f[4].to_string(), lines: Vec::new() });
            }
            "L" => {
                // L <scen> <n> <writeNo> <lineNo> <raw> — splitn(6) for the payload.
                let g: Vec<&str> = line.splitn(6, '\t').collect();
                let idx = index[&(g[1].to_string(), g[2].parse().unwrap())];
                let wi: usize = g[3].parse().unwrap();
                cases[idx].writes[wi].lines.push(g[5].to_string());
            }
            other => panic!("unknown record {other:?}"),
        }
    }
    cases
}

#[test]
fn writeline_stream_matches_showdown_per_write() {
    let cases = parse_golden();
    assert!(cases.len() >= 20, "expected >=20 captured battles, got {}", cases.len());
    let types = gated_types();

    let mut battles = 0usize;
    let mut writes_asserted = 0usize;
    let mut lines_asserted = 0usize;
    let mut scen_asserted: BTreeSet<String> = BTreeSet::new();

    for case in &cases {
        let mut stream = BattleStream::new();
        for (wi, w) in case.writes.iter().enumerate() {
            // The Rust `>start` uses the recorded pre-first-decision seed (see the
            // module docs); every other command replays VERBATIM.
            let cmd = if w.command.starts_with(">start ") {
                let seed_json: Vec<String> =
                    case.init_seed.split(',').map(|s| s.to_string()).collect();
                format!(
                    ">start {{\"formatid\":\"gen3customgame\",\"seed\":[{}]}}",
                    seed_json.join(",")
                )
            } else {
                w.command.clone()
            };
            let got: Vec<String> = stream.write_line(&cmd).into_iter().map(|l| l.0).collect();
            let want_f = filter(&w.lines, &types);
            let got_f = filter(&got, &types);
            if want_f != got_f {
                panic!(
                    "[{}/{}] write {wi} ({:?}) chunk divergence (init_seed {}):\n  \
                     golden ({} lines): {:#?}\n  engine ({} lines): {:#?}\n\
                     FIX THE EMITTED BYTES / CHUNK ATTRIBUTION — do not loosen the assert.",
                    case.scen, case.battle_no, w.command.chars().take(60).collect::<String>(),
                    case.init_seed, want_f.len(), want_f, got_f.len(), got_f,
                );
            }
            writes_asserted += 1;
            lines_asserted += want_f.len();
        }
        battles += 1;
        scen_asserted.insert(case.scen.clone());
    }

    assert!(battles >= 20, "expected >=20 asserted battles, got {battles}");
    assert!(writes_asserted >= 1000, "expected >=1000 per-write chunks, got {writes_asserted}");
    assert!(scen_asserted.len() >= 19, "expected all 19 scenarios, got {}", scen_asserted.len());
    eprintln!(
        "writeline stream: {} battles, {} writes byte-equal per-chunk ({} filtered lines) across {} scenarios",
        battles, writes_asserted, lines_asserted, scen_asserted.len()
    );
}
