//! The EMISSION SELF-CHECK (`gen3_core_emission_selfcheck_v1`, `src/emission_check.rs`) — the
//! properties that hold of the WIRING rather than of one check:
//!
//! * it is ON in a `cargo test` build and every emission path of a real bridge battle reaches it
//!   (the omniscient builder, the per-side fold, the bridge's own frames, the `|split|` shape);
//! * a failure inside a per-request `catch_unwind` binary KILLS the process (exit
//!   `EXIT_STATUS`, the message on stderr) instead of becoming a recoverable `__ERR__`;
//! * every call site is behind the `cfg` that compiles it out of `cargo build --release`, so a
//!   production binary cannot pay for it (the byte-identity + benchmark proof is the measurement
//!   record `designs/research_state/measurements/rust_core_emission_selfcheck_2026-09-23/`).

use std::io::Write;
use std::process::{Command, Stdio};

use pokesim::bridge::{bridge_opts, parse_choice, BridgeSession, Cmd};
use pokesim::dex::Dex;
use pokesim::emission_check as ec;

fn team() -> String {
    let mon = "Snorlax|||thickfat|watergun|Brave|252,,252,,4,|||||";
    std::iter::repeat(mon).take(6).collect::<Vec<_>>().join("]")
}

#[test]
fn the_check_is_on_in_a_test_build() {
    assert!(ec::ENABLED, "cargo test must run the emission self-check (debug_assertions)");
}

#[test]
fn a_bridge_battle_reaches_every_check() {
    let dex = Dex::for_gen(3);
    let t = team();
    let opts = bridge_opts("gen3ou", "1,2,3,4".to_string(), &t, &t);
    let before = ec::counts();
    let mut sess = BridgeSession::new(&opts, &dex).unwrap();
    let cmds: Vec<Cmd> = (0..6)
        .flat_map(|_| [0, 1].map(|side| Cmd { side, choice: parse_choice("move 1").unwrap() }))
        .collect();
    sess.feed_cmds(&cmds, &dex);
    sess.forfeit(0);
    // The replay family's `battle.log` shape (`|split|pN` triples) over the whole battle.
    let log = pokesim::search::turn_log(&sess, 0);
    assert!(log.iter().any(|l| l.starts_with("|split|p")), "the battle must produce split lines");
    let after = ec::counts();
    // Tests run in parallel and share the counters: a strict lower bound per kind.
    let lines = sess.battle_state().unwrap().log.lines().len() as u64;
    assert!(after[0] - before[0] >= lines, "omniscient checks {:?} -> {:?} for {lines} lines", before, after);
    assert!(after[1] - before[1] >= 2 * lines, "per-viewer checks {:?} -> {:?}", before, after);
    assert!(after[2] - before[2] >= lines, "split-shape checks {:?} -> {:?}", before, after);
    assert!(after[3] > before[3], "frame checks {:?} -> {:?}", before, after);
}

/// A self-check failure inside `sim_bridge` (which turns an ordinary handler panic into a
/// recoverable `__ERR__` and keeps serving) must END the process: a test build never continues.
#[test]
fn a_selfcheck_failure_kills_sim_bridge_instead_of_becoming_an_err_frame() {
    let t = team();
    let script = format!(
        "START {{\"seed\":[1,2,3,4],\"formatid\":\"gen3ou\",\"p1\":{{\"name\":\"A\",\"team\":\"{t}\"}},\
         \"p2\":{{\"name\":\"B\",\"team\":\"{t}\"}}}}\nCHOOSE p1 move 1\nCHOOSE p2 move 1\nEND\n"
    );
    let mut child = Command::new(env!("CARGO_BIN_EXE_sim_bridge"))
        .env("POKESIM_EMISSION_SELFCHECK_INJECT", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn sim_bridge");
    child.stdin.as_mut().unwrap().write_all(script.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(ec::EXIT_STATUS), "stdout {stdout}\nstderr {stderr}");
    assert!(stderr.contains(ec::MARKER) && stderr.contains("[injected]"), "{stderr}");
    assert!(!stdout.contains("__ERR__"), "a self-check failure must not become an __ERR__ frame: {stdout}");
    // And without the injection the same script is clean.
    let mut child = Command::new(env!("CARGO_BIN_EXE_sim_bridge"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.as_mut().unwrap().write_all(script.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
}

/// Every call into `emission_check` from the crate's code is behind the `cfg` that compiles it
/// out of `--release` — so nobody adds a check production would pay for. (The attribute must sit
/// within the three lines above the call: on the statement, or on the block/loop holding it.)
#[test]
fn every_call_site_is_compiled_out_of_release() {
    const CFG: &str = "#[cfg(any(debug_assertions, feature = \"emission-selfcheck\"))]";
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut stack = vec![root];
    let mut sites = 0;
    while let Some(dir) = stack.pop() {
        for e in std::fs::read_dir(&dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if p.extension().and_then(|x| x.to_str()) != Some("rs") || p.ends_with("emission_check.rs") {
                continue;
            }
            let text = std::fs::read_to_string(&p).unwrap();
            let lines: Vec<&str> = text.lines().collect();
            for (i, l) in lines.iter().enumerate() {
                let code = l.split("//").next().unwrap();
                if !code.contains("emission_check::") {
                    continue;
                }
                sites += 1;
                let guarded = lines[i.saturating_sub(3)..i].iter().any(|a| a.trim() == CFG);
                assert!(guarded, "{}:{}: an emission_check call without `{CFG}` above it", p.display(), i + 1);
            }
        }
    }
    assert!(sites >= 10, "found only {sites} call sites — the scan is not seeing the wiring");
}
