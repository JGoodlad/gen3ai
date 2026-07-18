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
//!
//! ── The KNOWN-RESIDUAL allowlist gate ("no NEW kinds", `gen3_omniscient_byte_fuzz_v1`) ──
//! Two fixture classes, so the allowlist is AUDITABLE + BOUNDED (it can only grow by a
//! reviewed reason+fixture pair, and any un-cataloged divergence kind is a hard failure):
//!   1. an EMISSION-FORM fixture (the default, no tag) MUST replay `ok` — byte-clean.
//!   2. an ALLOWLIST fixture carrying a `# ALLOWLIST <reason>` header line MUST replay to a
//!      `diverged` verdict whose `allowlisted` reason EXACTLY equals `<reason>` — the tagged
//!      documented residual (e.g. the R1 turn-0 construction-mirror `[of]` attribution). So
//!      (i) a residual fixture that stops matching its reason (a new divergence appears BEFORE
//!      it, or the residual is fixed) FAILS, and (ii) nobody can add a silently-ignored
//!      divergence: every escape must be a named allowlist entry backed by a tagged repro.

use std::path::{Path, PathBuf};
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

/// The allowlist REASON a fixture is tagged with, from a `# ALLOWLIST <reason>` header
/// comment line (whitespace-trimmed). `None` = an untagged emission-form fixture (must
/// replay `ok`).
fn allowlist_tag(file: &Path) -> Option<String> {
    let data = std::fs::read_to_string(file).ok()?;
    for line in data.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("# ALLOWLIST ") {
            return Some(rest.trim().to_string());
        }
    }
    None
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
        // A `# ALLOWLIST <reason>` header tags a KNOWN-RESIDUAL fixture (must diverge with
        // that exact `allowlisted` reason); an untagged fixture must replay `ok`.
        let tag = allowlist_tag(file);

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
            match &tag {
                // Untagged emission-form fixture: MUST be byte-clean.
                None => {
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
                // Tagged allowlist fixture: MUST diverge with EXACTLY the tagged reason.
                Some(reason) => {
                    let got_reason = pluck(line, "allowlisted");
                    if verdict != "diverged" || got_reason != Some(reason.as_str()) {
                        let kind = pluck(line, "kind").unwrap_or("-");
                        panic!(
                            "byte-fuzz ALLOWLIST fixture {} must diverge with allowlisted=\
                             {reason:?}, but got verdict={verdict} kind={kind} \
                             allowlisted={got_reason:?}\n  (a residual fixture stopped matching \
                             its reason — a new divergence appeared before it, or the residual \
                             was fixed: re-tag or promote it)\n  full verdict: {line}",
                            file.display()
                        );
                    }
                }
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
