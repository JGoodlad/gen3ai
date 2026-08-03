//! `sim_bridge` SEED handling — the `gen3_bridge_seedless_fixed_seed_v1` /
//! `gen3_bridge_seed_forms_v1` regression gate.
//!
//! Four defects shipped together because EVERY pre-existing gate on this code path
//! passes an explicit `[m,n,o,p]` array seed, so the PRODUCTION spelling (no seed at all
//! — what `bridge_session` and `run_local_battles` actually send) and the STRING spelling
//! (what `prober.falsifier.fresh_seeds` emits) were never executed:
//!
//! * **B1** a seedless `START` fell through to the engine's `DEFAULT_CONSTRUCT_SEED`
//!   (`"0,0,0,0"`), so every training episode and eval game replayed ONE dice stream —
//!   identically across every parallel env worker;
//! * **B2** and, because the record is skipped without a resolved seed, emitted no
//!   `__RECON__`, so a rust eval wrote no `*_reconstruction.json`;
//! * **B3** a STRING seed (`"1,2,3,4"` / `"sodium,…"`) was SILENTLY IGNORED by the
//!   array-only parser — a node-recorded battle replayed on rust ran a DIFFERENT battle;
//! * **B4** `resumeReseed.seed` was array-only too, but its only producer emits the
//!   string form `new PRNG()` requires — so the counterfactual re-roll hard-errored.
//!
//! These drive the REAL binary over its REAL stdin/stdout protocol
//! (`CARGO_BIN_EXE_sim_bridge`, the `bridge_corpus_test` pattern), because the bug lived
//! in the binary's START parser, not in the engine.

use std::io::Write;
use std::process::{Command, Stdio};

/// Six bulky Snorlax whose only move is the FEEBLE Water Gun. Deliberate: the battle
/// consumes dice every turn (gender sampling at construction, then a damage roll +
/// accuracy per hit) so two seeds diverge on turn 1, yet ~24 damage into ~524 HP means
/// NOTHING faints inside [`N_TURNS`] — so a blind mutual `move 1` script never hits a
/// force-switch boundary (which the bridge, correctly, fail-louds on as a driver desync).
fn team() -> String {
    let mon = "Snorlax|||thickfat|watergun|Brave|252,,252,,4,|||||";
    std::iter::repeat(mon).take(6).collect::<Vec<_>>().join("]")
}

/// Turns of mutual `move 1` per scripted battle — comfortably inside both the first
/// faint (~turn 22) and Water Gun's 25 PP.
const N_TURNS: usize = 12;

/// Run one battle through the binary: `START <json>`, `n_turns` mutual `move 1`, then a
/// `FORCELOSE` so the battle actually ENDS (which is what emits `__RECON__` + `__END__` —
/// a bare `END` just exits the process). Returns the child's raw stdout.
fn run_start(start_json: &str, n_turns: usize) -> String {
    let mut script = format!("START {start_json}\n");
    for _ in 0..n_turns {
        script.push_str("CHOOSE p1 move 1\nCHOOSE p2 move 1\n");
    }
    script.push_str("FORCELOSE p1\nEND\n");

    let mut child = Command::new(env!("CARGO_BIN_EXE_sim_bridge"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn sim_bridge");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(script.as_bytes())
        .expect("write script");
    let out = child.wait_with_output().expect("sim_bridge output");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// A `START` payload with the given extra fields spliced in (e.g. `"\"seed\":[1,2,3,4],"`).
fn start_json(extra: &str) -> String {
    let t = team();
    format!(
        "{{{extra}\"formatid\":\"gen3customgame\",\
         \"p1\":{{\"name\":\"A\",\"team\":\"{t}\"}},\
         \"p2\":{{\"name\":\"B\",\"team\":\"{t}\"}}}}"
    )
}

fn err_lines(stdout: &str) -> Vec<&str> {
    stdout.lines().filter(|l| l.starts_with("__ERR__")).collect()
}

/// Just the per-side protocol frames — i.e. the BATTLE, with the `__RECON__` record
/// (which echoes the caller's seed SPELLING) dropped.
fn protocol_only(stdout: &str) -> Vec<&str> {
    stdout
        .lines()
        .filter(|l| l.starts_with("p1 ") || l.starts_with("p2 "))
        .collect()
}

fn has_recon(stdout: &str) -> bool {
    stdout.lines().any(|l| l.starts_with("__RECON__"))
}

/// The base64 `__RECON__` payload, decoded — for the `>start` seed assertion.
fn recon_json(stdout: &str) -> String {
    let line = stdout
        .lines()
        .find(|l| l.starts_with("__RECON__ "))
        .expect("a __RECON__ frame");
    let b64 = &line["__RECON__ ".len()..];
    String::from_utf8(b64_decode(b64)).expect("utf8 record")
}

fn b64_decode(s: &str) -> Vec<u8> {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let val = |c: u8| A.iter().position(|&x| x == c).map(|i| i as u32);
    let mut out = Vec::new();
    let bytes: Vec<u8> = s.bytes().filter(|&c| c != b'=' && !c.is_ascii_whitespace()).collect();
    for chunk in bytes.chunks(4) {
        let mut n = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            n |= val(c).expect("base64 alphabet") << (18 - 6 * i);
        }
        let take = chunk.len() * 6 / 8;
        for i in 0..take {
            out.push(((n >> (16 - 8 * i)) & 0xff) as u8);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// B1 — a seedless START must MINT a seed, not reuse a constant.
// ---------------------------------------------------------------------------

#[test]
fn b1_seedless_starts_run_distinct_battles() {
    let json = start_json("");
    let a = run_start(&json, N_TURNS);
    let b = run_start(&json, N_TURNS);
    assert!(err_lines(&a).is_empty(), "seedless START errored: {:?}", err_lines(&a));
    assert!(!a.is_empty() && !b.is_empty(), "seedless START produced no protocol");
    assert_ne!(
        a, b,
        "two SEEDLESS battles produced a byte-identical protocol stream — the bridge is \
         replaying ONE dice stream (gen3_bridge_seedless_fixed_seed_v1). Every training \
         episode and eval game sends exactly this seedless START."
    );
}

/// The minted seed must be a REAL seed, not the engine's `0,0,0,0` construct default.
#[test]
fn b1_seedless_minted_seed_is_not_the_construct_default() {
    let out = run_start(&start_json(""), N_TURNS);
    let rec = recon_json(&out);
    assert!(
        !rec.contains("\\\"seed\\\":\\\"0,0,0,0\\\"") && !rec.contains("\"prng_seed\":\"0,0,0,0\""),
        "the seedless battle resolved to the DEFAULT_CONSTRUCT_SEED: {rec}"
    );
    assert!(
        rec.contains("\"prng_seed\":\"sodium,"),
        "a minted seed should be Showdown's default `sodium,<hex>` form: {rec}"
    );
}

// ---------------------------------------------------------------------------
// B2 — a seedless battle must still emit __RECON__ (it now has a resolved seed).
// ---------------------------------------------------------------------------

#[test]
fn b2_seedless_battle_emits_recon() {
    let out = run_start(&start_json(""), N_TURNS);
    assert!(
        has_recon(&out),
        "a seedless battle emitted no __RECON__ — a rust eval would write no \
         *_reconstruction.json and the prober's forensic commands go dark"
    );
    let rec = recon_json(&out);
    // The `>start` seed must be a JSON STRING (mirrors the sim's own inputLog[0]); a bare
    // `[...]` array spelling would be INVALID JSON for a minted `sodium,<hex>` seed.
    assert!(
        rec.contains("\\\">start {\\\"formatid\\\"") || rec.contains(">start {"),
        "record has no >start line: {rec}"
    );
    assert!(!rec.contains("\\\"seed\\\":[sodium"), "seed rendered as a bare array: {rec}");
}

// ---------------------------------------------------------------------------
// B3 — every seed form `new PRNG()` accepts must be accepted, and mean the same thing.
// ---------------------------------------------------------------------------

#[test]
fn b3_array_and_string_seed_forms_are_identical() {
    let arr = run_start(&start_json("\"seed\":[1,2,3,4],"), N_TURNS);
    let s = run_start(&start_json("\"seed\":\"1,2,3,4\","), N_TURNS);
    let hex = run_start(&start_json("\"seed\":\"gen5,0001000200030004\","), N_TURNS);
    assert!(err_lines(&s).is_empty(), "string seed errored: {:?}", err_lines(&s));
    assert_eq!(
        arr, s,
        "the STRING seed \"1,2,3,4\" ran a different battle than the ARRAY [1,2,3,4] — a \
         node-recorded battle replayed on rust would silently be a different battle"
    );
    // Protocol only: the record legitimately echoes the seed SPELLING the caller used
    // (node does the same — `prngSeed` is `PRNG.startingSeed`, the string as given), so the
    // `__RECON__` frame differs while the battle must not.
    assert_eq!(
        protocol_only(&arr),
        protocol_only(&hex),
        "`gen5,<hex16>` is the same seed as its decimal quadruple"
    );
}

#[test]
fn b3_sodium_seed_is_honoured_and_reproducible() {
    let a = run_start(&start_json("\"seed\":\"sodium,deadbeef\","), N_TURNS);
    let b = run_start(&start_json("\"seed\":\"sodium,deadbeef\","), N_TURNS);
    let other = run_start(&start_json("\"seed\":\"sodium,feedface\","), N_TURNS);
    assert!(err_lines(&a).is_empty(), "sodium seed errored: {:?}", err_lines(&a));
    assert_eq!(a, b, "the same sodium seed must reproduce the same battle");
    assert_ne!(a, other, "a different sodium seed was IGNORED (ran the same battle)");
    // And it is not just falling through to the seedless/mint path:
    let seedless = run_start(&start_json(""), N_TURNS);
    assert_ne!(a, seedless, "the sodium seed was ignored (matched an unseeded run)");
}

// ---------------------------------------------------------------------------
// B3/B4 — a seed that is PRESENT but unusable must FAIL LOUD, never fall back.
// ---------------------------------------------------------------------------

#[test]
fn b3_unparseable_seed_fails_loud() {
    for bad in [
        "\"seed\":\"not-a-seed\",",
        "\"seed\":12345,",
        "\"seed\":true,",
        "\"seed\":\"1,2,3\",",
        "\"seed\":[1,2,\"x\",4],",
        "\"seed\":\"gen5,zz\",",
    ] {
        let out = run_start(&start_json(bad), N_TURNS);
        assert!(
            !err_lines(&out).is_empty(),
            "{bad} was accepted silently — the battle ran on some OTHER dice stream. \
             stdout: {out}"
        );
        assert!(
            !out.lines().any(|l| l.starts_with("p1 ")),
            "{bad} produced protocol output despite an unusable seed"
        );
    }
}

#[test]
fn b4_resume_reseed_accepts_the_string_seed_its_producer_emits() {
    // `main.prober.falsifier.fresh_seeds` emits "a,b,c,d" — the ONLY form node's
    // `new PRNG(seed)` accepts for the mid-battle swap. This used to hard-error with
    // "START: resumeReseed needs both `turn` and `seed`".
    let s = start_json("\"seed\":[1,2,3,4],\"resumeReseed\":{\"turn\":3,\"seed\":\"9,8,7,6\"},");
    let out = run_start(&s, N_TURNS);
    assert!(
        err_lines(&out).is_empty(),
        "a STRING resumeReseed.seed was rejected: {:?}",
        err_lines(&out)
    );
    // Same spec in the array form ⇒ the same battle.
    let arr = start_json("\"seed\":[1,2,3,4],\"resumeReseed\":{\"turn\":3,\"seed\":[9,8,7,6]},");
    assert_eq!(out, run_start(&arr, N_TURNS), "array/string resumeReseed disagree");
    // And it actually re-rolled: without the reseed the battle differs from turn 3 on.
    let plain = run_start(&start_json("\"seed\":[1,2,3,4],"), N_TURNS);
    assert_ne!(out, plain, "resumeReseed did not change the post-divergence dice");
}

#[test]
fn b4_unparseable_resume_reseed_fails_loud() {
    for bad in ["\"zzz\"", "17", "[1,2,3]"] {
        let s = start_json(&format!(
            "\"seed\":[1,2,3,4],\"resumeReseed\":{{\"turn\":3,\"seed\":{bad}}},"
        ));
        let out = run_start(&s, N_TURNS);
        assert!(
            !err_lines(&out).is_empty(),
            "resumeReseed.seed={bad} was accepted silently — the counterfactual would \
             quietly answer the WRONG question. stdout: {out}"
        );
    }
}
