//! The FROZEN byte-fuzz regression CORPUS gate (`gen3_omniscient_byte_fuzz_v1`).
//!
//! The A/B `--protocol` byte fuzzer (`harness/ab_fuzz.js` + `src/bin/ab_replay.rs`)
//! FINDS emission-form divergences, but it is a long-running unattended hunter — NOT
//! a `cargo test` gate. This test freezes a CORPUS of full-battle repros under
//! `tests/vectors/byte_fuzz_corpus/*.txt` (each a self-contained fuzzer chunk — the
//! `SCEN`/`TEAM`/`FMT`/`INIT`/`DEC`/`END`/`L` rows a repro dir's `battle.txt` carries)
//! so every FIXED emission form has a PERMANENT end-to-end byte gate that can't
//! silently regress.
//!
//! Each fixture is a real bridge-pool battle that EXERCISES a now-fixed `|...|`
//! emission form (its name says which — e.g. `01_recover_at_full_still_fail.txt`
//! guards the Recover-at-full `[still]`+`-fail|heal` form) and REPLAYS CLEAN through
//! the emitting engine today. The gate invokes the built `ab_replay` binary
//! (`CARGO_BIN_EXE_ab_replay`, provided by Cargo to integration tests — no manual
//! build ordering) in byte mode on each fixture and asserts NO `kind=protocol`
//! divergence (nor any `seed`/`state`/`panic`/`parse_error`). A per-file panic names
//! the diverging file + line on failure.
//!
//! To ADD a fixture: drop a clean repro `battle.txt` into the corpus dir (see the
//! folder README) — this test auto-discovers every `*.txt`.

use std::path::PathBuf;
use std::process::Command;

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/byte_fuzz_corpus")
}

/// Discover every `*.txt` fixture (sorted for a stable failure order).
fn corpus_files() -> Vec<PathBuf> {
    let dir = corpus_dir();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read corpus dir {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("txt"))
        .collect();
    files.sort();
    files
}

/// Minimal field pluck from one JSON verdict line: `"key":"value"` (string) — good
/// enough for the flat, escape-free verdict lines `ab_replay` emits.
fn pluck<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\":\"");
    let start = line.find(&needle)? + needle.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(&rest[..end])
}

#[test]
fn byte_fuzz_corpus_replays_clean() {
    let files = corpus_files();

    // FLOOR: the corpus can't silently shrink below the mandated coverage size.
    assert!(
        files.len() >= 15,
        "the byte-fuzz corpus must hold >= 15 fixtures (found {}); dropping repros \
         erodes the emission-form gate — see tests/vectors/byte_fuzz_corpus/README.md",
        files.len()
    );

    let bin = env!("CARGO_BIN_EXE_ab_replay");
    let mut checked = 0usize;

    for file in &files {
        let out = Command::new(bin)
            .arg("--protocol")
            .arg(file)
            .output()
            .unwrap_or_else(|e| panic!("failed to run ab_replay on {}: {e}", file.display()));
        let stdout = String::from_utf8_lossy(&out.stdout);

        let mut saw_battle = false;
        for line in stdout.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("{\"chunk_summary\"") {
                continue;
            }
            saw_battle = true;
            let verdict = pluck(line, "verdict").unwrap_or("<none>");
            if verdict != "ok" {
                let kind = pluck(line, "kind").unwrap_or("-");
                let dec = line
                    .find("\"decision\":")
                    .map(|i| &line[i + 11..])
                    .and_then(|r| r.split([',', '}']).next())
                    .unwrap_or("-");
                let detail = pluck(line, "detail").unwrap_or("-");
                panic!(
                    "byte-fuzz corpus REGRESSION in {}: verdict={verdict} kind={kind} \
                     decision={dec}\n  detail: {detail}\n  full verdict: {line}",
                    file.display()
                );
            }
        }
        assert!(
            saw_battle,
            "corpus fixture {} produced no battle verdict (empty/malformed chunk?)",
            file.display()
        );
        checked += 1;
    }

    assert_eq!(checked, files.len(), "every discovered fixture must be checked");
    eprintln!("byte-fuzz corpus: {checked} fixtures replayed CLEAN (no kind=protocol)");
}
