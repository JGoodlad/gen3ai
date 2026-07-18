//! The FROZEN PER-SIDE / `|request|` regression CORPUS gate (`gen3_perside_request_byte_fuzz_v1`).
//!
//! The per-side/request A/B byte fuzzer (`harness/bridge_ab_fuzz.js` +
//! `src/bin/bridge_replay.rs --ab`) is a long-running unattended hunter over real pool
//! teams — NOT a `cargo test` gate. This test freezes a CORPUS of single-battle repros
//! under `tests/vectors/bridge_corpus/*.txt` (each a self-contained bridge-fuzzer
//! `battle.txt`: the `SCEN`/`TEAM`/`INIT`/`CMD`/`SEED`/`CHUNK`/`END` rows) so the per-side
//! HP-privacy fold + the `|request|` JSON serializer have a PERMANENT byte gate that can't
//! silently regress, AND so the two NARROW documented request-DISPLAY deferrals stay
//! auditable (each backed by a tagged repro — the allowlist can only grow by a reviewed
//! reason+fixture pair).
//!
//! Two fixture classes:
//!   1. an UNTAGGED fixture (a clean per-side/request battle) MUST replay `ok` — every
//!      per-side chunk + `|request|` frame byte-identical to the real `getPlayerStreams`,
//!      AND the SEED ANCHOR (each decision's post-decision engine seed == the recorded
//!      omniscient `seedAfter`) holds.
//!   2. an ALLOWLIST fixture carrying a `# ALLOWLIST <reason>` header MUST replay to a
//!      `diverge` verdict whose `allowlisted` reason EXACTLY equals `<reason>` — the tagged
//!      request-DISPLAY deferral (Curse `target:self` vs `normal`; the `return102` numeric
//!      alias). So (i) a residual fixture that stops matching its reason FAILS, and (ii)
//!      nobody can add a silently-ignored per-side/request divergence.
//!
//! The gate invokes the built `bridge_replay` binary (`CARGO_BIN_EXE_bridge_replay`,
//! provided by Cargo — no manual build ordering; this is a SEPARATE binary from the live
//! fuzzer's isolated build, so it never touches `/tmp/pokesim_target_bridge`) in `--ab`
//! mode on each fixture. A per-file panic names the diverging file + kind on failure.
//!
//! To ADD a fixture: drop a clean `battle.txt` (a bridge-fuzzer repro dir's file) into the
//! corpus dir — this test auto-discovers every `*.txt`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/bridge_corpus")
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

/// Minimal field pluck from one JSON verdict line: `"key":"value"` (string).
fn pluck<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\":\"");
    let start = line.find(&needle)? + needle.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(&rest[..end])
}

/// The allowlist REASON a fixture is tagged with, from a `# ALLOWLIST <reason>` header
/// comment line. `None` = an untagged fixture (must replay `ok`).
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
fn bridge_corpus_replays_clean() {
    let files = corpus_files();

    // FLOOR: the corpus can't silently shrink below the mandated coverage size (4 clean
    // per-side/request fixtures + the 2 tagged deferrals).
    assert!(
        files.len() >= 6,
        "the bridge per-side/request corpus must hold >= 6 fixtures (found {}); dropping \
         repros erodes the per-side/request gate — see tests/vectors/bridge_corpus/README.md",
        files.len()
    );

    let bin = env!("CARGO_BIN_EXE_bridge_replay");
    let mut checked = 0usize;
    let mut clean = 0usize;
    let mut allow = 0usize;

    for file in &files {
        let tag = allowlist_tag(file);

        let out = Command::new(bin)
            .arg(file)
            .arg("--ab")
            .output()
            .unwrap_or_else(|e| panic!("failed to run bridge_replay on {}: {e}", file.display()));
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
                // Untagged fixture: MUST be byte-clean (ok — includes the SEED ANCHOR pass).
                None => {
                    if verdict != "ok" {
                        let kind = pluck(line, "kind").unwrap_or("-");
                        panic!(
                            "bridge per-side/request corpus REGRESSION in {}: verdict={verdict} \
                             kind={kind}\n  full verdict: {line}",
                            file.display()
                        );
                    }
                    clean += 1;
                }
                // Tagged allowlist fixture: MUST diverge with EXACTLY the tagged reason.
                Some(reason) => {
                    let got_reason = pluck(line, "allowlisted");
                    if verdict != "diverge" || got_reason != Some(reason.as_str()) {
                        let kind = pluck(line, "kind").unwrap_or("-");
                        panic!(
                            "bridge ALLOWLIST fixture {} must diverge with allowlisted={reason:?}, \
                             but got verdict={verdict} kind={kind} allowlisted={got_reason:?}\n  \
                             (a residual fixture stopped matching its reason — the request form \
                             changed, or a new divergence appeared before it)\n  full verdict: {line}",
                            file.display()
                        );
                    }
                    allow += 1;
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
    assert!(clean >= 3, "the corpus must hold >= 3 CLEAN per-side/request fixtures (have {clean})");
    assert!(allow >= 2, "the corpus must hold the 2 tagged request-DISPLAY deferrals (have {allow})");
    eprintln!("bridge per-side/request corpus: {clean} clean + {allow} allowlisted fixtures verified");
}
