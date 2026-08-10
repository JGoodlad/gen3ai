//! replay_driver_test.rs — the node-free gate for the REPLAY family
//! (`gen3_rust_replay_driver_v1`): the one-shot `replay` / `reroll` / `reroll_many` verbs
//! the `search_driver` binary now serves alongside the persistent search protocol.
//!
//! The differential lives in `tmp/replay_impl_parity.py` (it drives BOTH
//! `node replay_driver.js` and this binary on the identical request and diffs every field).
//! That harness is scratch and needs node + a captured record, so the properties below —
//! the ones a `cargo test` run must never lose — are pinned here instead:
//!
//!   1. **THE ONE-SHOT DISPATCH** — a request carrying `mode` is answered with a BARE JSON
//!      object (no `id`, no `ok`, NO trailing newline) and the process EXITS: 0 on success,
//!      1 with `{"error": …}` on failure. `reconstruction.py::_run_driver` checks the exit
//!      code BEFORE it parses stdout, so a wrong code turns a clean error into an opaque
//!      "driver failed" — and a missing exit turns a one-shot caller into a hang.
//!   2. **THE SEARCH PROTOCOL IS UNTOUCHED** — a `{id, cmd}` request still gets the
//!      newline-terminated `{id, ok, …}` reply and the process stays ALIVE for the next one.
//!      This is the regression the dispatch could plausibly cause.
//!   3. **`recordedQueues` REFUSAL-PULL** — the replay family's `"recorded"` source is a
//!      QUEUE, not the single-shot latch the search path uses: a refused choice pulls the
//!      NEXT recorded command. This is the ONE kernel the search half never needed, and
//!      getting it wrong would silently answer a refusal with an INVENTED follow-up pick.
//!   4. **`recorded_queues` ITSELF** — per-side split, the `forcelose` stop, and the cap.
//!
//! These drive the REAL BINARY over its REAL stdin/stdout protocol (the
//! `bridge_corpus_test` `CARGO_BIN_EXE_` pattern) wherever the property IS the protocol,
//! because the bug class here lives in the driver's dispatch, not in the kernels.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

use pokesim::battle::BattleOptions;
use pokesim::bridge::{bridge_opts, parse_choice, BridgeSession, Cmd, RequestState};
use pokesim::dex::Dex;
use pokesim::search::{
    aux_rng_from_seed, recorded_queues, resolve_turn_sourced, Record, TurnSource,
    RECORDED_QUEUE_CAP,
};

const BIN: &str = env!("CARGO_BIN_EXE_search_driver");

// ===========================================================================
// A minimal, self-contained RECORD — the same gen3customgame mirror board the search
// gate uses, driven far enough to have a real command stream. Built here rather than
// loaded from a captured file so this test needs neither node nor a fixture.
// ===========================================================================

const P1_TEAM: &str = "Blissey|||NoAbility|tackle,headbutt|Serious|252,,252,,,|F||||]Regice|||NoAbility|tackle,headbutt|Serious|252,,252,,,|N||||";
const P2_TEAM: &str = "Blissey|||NoAbility|tackle,headbutt|Serious|252,,252,,,|F||||]Zapdos|||NoAbility|tackle,headbutt|Serious|252,,252,,,|N||||";
const SEED: &str = "1,2,3,4";

fn opts() -> BattleOptions {
    bridge_opts("gen3customgame", SEED.to_string(), P1_TEAM, P2_TEAM)
}

fn cmd(side: usize, tok: &str) -> Cmd {
    Cmd { side, choice: parse_choice(tok).expect("parse choice") }
}

/// Play the fixture board to game-end with a trivial policy (`move 1`, and the first live
/// bench slot on a forced replacement) and return the COMMAND STREAM it produced.
///
/// Built through [`BridgeSession::new_construct_turn0`] on purpose: that is the constructor
/// `search::session_from_record` uses, so the recorded stream replays bit-for-bit. Recording
/// through `BridgeSession::new` (the post-construction-seed convention) would roll different
/// dice on replay and a recorded `switch N` could land on a mon that is no longer there.
fn finished_battle_commands() -> Vec<(usize, String)> {
    let dex = Dex::for_gen(3);
    let mut sess = BridgeSession::new_construct_turn0(&opts(), &dex).expect("session");
    let mut cmds: Vec<(usize, String)> = Vec::new();
    for _ in 0..4000 {
        if sess.is_ended() || sess.fatal().is_some() {
            break;
        }
        let mut wrote = false;
        for s in 0..2 {
            let Some(kind) = sess.request_kind(s) else { continue };
            if kind == RequestState::Wait || sess.is_choice_done(s) {
                continue;
            }
            let tok = if kind == RequestState::Switch {
                let st = sess.battle_state().expect("state");
                let side = &st.sides[s];
                let Some(i) = side
                    .pokemon
                    .iter()
                    .enumerate()
                    .position(|(i, m)| i != side.active && !m.fainted)
                else {
                    continue;
                };
                format!("switch {}", i + 1)
            } else {
                "move 1".to_string()
            };
            cmds.push((s, tok.clone()));
            sess.feed_cmd(cmd(s, &tok), &dex);
            wrote = true;
        }
        if !wrote {
            break;
        }
    }
    assert!(sess.is_ended(), "the fixture policy must drive the board to game-end");
    cmds
}

/// A record over the fixture board. `n_cmds == None` uses the whole finished stream (so
/// `replay` succeeds); a `Some(k)` TRUNCATES it (so `replay` must report "not ended").
fn record_json(n_cmds: Option<usize>) -> String {
    let all = finished_battle_commands();
    let take = n_cmds.unwrap_or(all.len()).min(all.len());
    let cmds = all[..take]
        .iter()
        .map(|(s, tok)| format!("[\"p{}\",\"{}\"]", s + 1, tok))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"v\":1,\"format_id\":\"gen3customgame\",\"prng_seed\":\"{SEED}\",\
          \"input_log\":[\"\\u003estart {{\\\"formatid\\\":\\\"gen3customgame\\\",\\\"seed\\\":\\\"{SEED}\\\"}}\",\
          \"\\u003eplayer p1 {{\\\"name\\\":\\\"P1\\\",\\\"team\\\":\\\"{p1}\\\"}}\",\
          \"\\u003eplayer p2 {{\\\"name\\\":\\\"P2\\\",\\\"team\\\":\\\"{p2}\\\"}}\"],\
          \"commands\":[{cmds}]}}",
        p1 = P1_TEAM.replace('\\', "\\\\"),
        p2 = P2_TEAM.replace('\\', "\\\\"),
    )
}

/// Run the binary as a ONE-SHOT caller does (`subprocess.run(input=…)`): write the request,
/// close stdin, read everything. Returns `(exit_code, stdout)`.
fn one_shot(request: &str) -> (i32, String) {
    let mut child = Command::new(BIN)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn search_driver");
    child.stdin.take().expect("stdin").write_all(request.as_bytes()).expect("write");
    let out = child.wait_with_output().expect("wait");
    (out.status.code().unwrap_or(-1), String::from_utf8_lossy(&out.stdout).into_owned())
}

// ===========================================================================
// 1. THE ONE-SHOT DISPATCH
// ===========================================================================

#[test]
fn a_mode_request_answers_once_and_exits_zero_with_no_trailing_newline() {
    let req = format!("{{\"mode\":\"replay\",\"record\":{}}}", record_json(None));
    let (code, out) = one_shot(&req);
    assert_eq!(code, 0, "a successful one-shot must exit 0; stdout: {}", &out[..out.len().min(400)]);
    // NO trailing newline — Node's `process.stdout.write(JSON.stringify(obj))`. A caller that
    // reads the whole stream does not care, but a line-oriented one would hang, and the
    // difference is free to preserve.
    assert!(!out.ends_with('\n'), "one-shot output must not be newline-terminated");
    // The BARE object shape: the replay family has no `id` / `ok` envelope.
    assert!(out.starts_with("{\"p1_chunks\":"), "unexpected body: {}", &out[..out.len().min(200)]);
    assert!(!out.contains("\"ok\":"), "the replay family must not carry the search envelope");
    assert!(out.contains("\"outcome\":"), "replay must report the final outcome");
}

#[test]
fn a_failing_one_shot_exits_one_with_an_error_object() {
    // A record whose command stream stops mid-battle: `replay` must detect that the battle
    // never ended rather than report a half-finished outcome as success.
    let req = format!("{{\"mode\":\"replay\",\"record\":{}}}", record_json(Some(3)));
    let (code, out) = one_shot(&req);
    assert_eq!(code, 1, "a failing one-shot must exit 1 (reconstruction.py branches on it)");
    assert!(out.starts_with("{\"error\":"), "unexpected body: {out}");
    assert!(out.contains("but battle has not ended"), "wrong error class: {out}");
}

#[test]
fn an_unknown_mode_is_an_error_not_a_silent_success() {
    let (code, out) = one_shot("{\"mode\":\"teleport\",\"record\":{}}");
    assert_eq!(code, 1);
    assert!(out.contains("unknown mode teleport"), "unexpected body: {out}");
}

#[test]
fn an_invalid_turn_is_rejected_before_any_battle_is_built() {
    for (turn, shown) in [("0", "invalid turn 0"), ("2.5", "invalid turn 2.5")] {
        let req = format!(
            "{{\"mode\":\"reroll\",\"record\":{},\"turn\":{turn},\"seeds\":[]}}",
            record_json(None)
        );
        let (code, out) = one_shot(&req);
        assert_eq!(code, 1, "turn {turn} must fail");
        assert!(out.contains(shown), "turn {turn}: unexpected body: {out}");
    }
}

// ===========================================================================
// 2. THE SEARCH PROTOCOL IS UNTOUCHED BY THE DISPATCH
// ===========================================================================

#[test]
fn a_cmd_request_still_gets_the_persistent_envelope_and_the_process_stays_alive() {
    let mut child = Command::new(BIN)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    let mut stdin = child.stdin.take().expect("stdin");
    let mut stdout = BufReader::new(child.stdout.take().expect("stdout"));
    let mut line = String::new();

    let rec = record_json(None);
    writeln!(stdin, "{{\"id\":7,\"cmd\":\"open_root\",\"record\":{rec},\"turn\":3}}").unwrap();
    stdout.read_line(&mut line).expect("a reply");
    assert!(line.ends_with('\n'), "the persistent protocol IS newline-delimited");
    assert!(line.contains("\"id\":7"), "the id must be echoed: {line}");
    assert!(line.contains("\"ok\":true"), "open_root should succeed: {line}");
    assert!(line.contains("\"node_id\":\"n0\""), "expected a root node: {line}");

    // STILL ALIVE — this is the regression the `mode` dispatch could have introduced.
    line.clear();
    writeln!(stdin, "{{\"id\":8,\"cmd\":\"teleport\"}}").unwrap();
    stdout.read_line(&mut line).expect("a second reply");
    assert!(line.contains("\"ok\":false") && line.contains("unknown cmd teleport"), "{line}");

    line.clear();
    writeln!(stdin, "{{\"id\":9,\"cmd\":\"close\"}}").unwrap();
    stdout.read_line(&mut line).expect("a bye");
    assert!(line.contains("\"bye\":true"), "{line}");
    assert!(child.wait().expect("wait").success(), "close must exit 0");
}

// ===========================================================================
// 3. THE `recordedQueues` REFUSAL-PULL (the kernel the search half never needed)
// ===========================================================================

/// A session paused at a mid-battle `move` boundary, with the non-vacuity guards the search
/// gate uses (a live board, both sides on a `move` request, a real bench).
fn paused_session(dex: &Dex, turns: usize) -> BridgeSession {
    let mut sess = BridgeSession::new(&opts(), dex).expect("session");
    for _ in 0..turns {
        sess.feed_cmd(cmd(0, "move 1"), dex);
        sess.feed_cmd(cmd(1, "move 1"), dex);
    }
    assert!(!sess.is_ended(), "fixture ended early");
    assert_eq!(sess.request_kind(0), Some(RequestState::Move));
    assert_eq!(sess.request_kind(1), Some(RequestState::Move));
    sess
}

#[test]
fn a_recorded_queue_pulls_the_next_entry_when_the_first_is_refused() {
    let dex = Dex::for_gen(3);
    let mut sess = paused_session(&dex, 2);

    // p1's queue leads with a choice the sim REFUSES (`switch 1` is the ACTIVE slot), then
    // carries the correction — the shape a live refused-then-corrected round records.
    let p1_recorded = vec!["switch 1".to_string(), "move 2".to_string()];
    let p2_recorded = vec!["move 1".to_string()];
    let mut sources = [
        TurnSource::from_replay_spec("recorded", &p1_recorded),
        TurnSource::from_replay_spec("recorded", &p2_recorded),
    ];
    let mut rng = aux_rng_from_seed("original");
    let r = resolve_turn_sourced(&mut sess, &mut sources, "random", &mut rng, &dex);

    assert!(!r.stuck, "the corrected choice should settle the turn");
    // THE PROPERTY: the refusal pulled the NEXT recorded entry, so BOTH appear in order.
    // A single-shot source would have fallen to the follow-up POLICY and recorded an
    // invented pick instead of `move 2` — silently answering a question the record already
    // answered.
    assert_eq!(r.used[0], vec!["switch 1".to_string(), "move 2".to_string()], "p1 queue pull");
    // p2 is DELIBERATELY not asserted beyond its first pick. The port's driver-level reject
    // rebuilds the whole boundary (bridge.rs models only the trapped-SWITCH reject), so the
    // UNREJECTED side is asked a second time and its queue-then-follow-up answer is recorded
    // too — where Node re-opens only the rejected side. That is a PRE-EXISTING bridge gap,
    // explicitly allowlisted (with its hit count) by `tmp/replay_impl_parity.py`; pinning it
    // here would freeze the gap in place instead of the property.
    assert_eq!(r.used[1][0], "move 1", "p2's FIRST answer still comes from its queue");
}

#[test]
fn a_single_shot_source_falls_to_the_followup_instead_of_re_sending() {
    let dex = Dex::for_gen(3);
    let mut sess = paused_session(&dex, 2);

    // The CONTRAST that makes the test above mean something: the SAME refused choice from an
    // EXPLICIT (single-shot) source must NOT be re-sent — the follow-up policy answers.
    let mut sources = [
        TurnSource::from_replay_spec("switch 1", &[]),
        TurnSource::from_replay_spec("move 1", &[]),
    ];
    let mut rng = aux_rng_from_seed("original");
    let r = resolve_turn_sourced(&mut sess, &mut sources, "default", &mut rng, &dex);

    assert!(!r.stuck, "the follow-up should settle the turn");
    assert_eq!(r.used[0][0], "switch 1", "the explicit choice is tried once");
    assert_eq!(r.used[0].len(), 2, "then exactly one follow-up: {:?}", r.used[0]);
    assert_ne!(r.used[0][1], "switch 1", "a single-shot source must not re-send its choice");
}

// ===========================================================================
// 4. `recorded_queues` ITSELF
// ===========================================================================

#[test]
fn recorded_queues_splits_by_side_stops_at_forcelose_and_caps() {
    let rec = Record::parse(&pokesim::json::Json::parse(&record_json(Some(4))).expect("json"))
        .expect("record");
    let q = recorded_queues(&rec, 0, RECORDED_QUEUE_CAP);
    assert_eq!(q[0], vec!["move 1".to_string(); 2], "p1's two commands");
    assert_eq!(q[1], vec!["move 1".to_string(); 2], "p2's two commands");

    // `from` skips the consumed prefix, exactly like `recorded_turn_choices`.
    let q = recorded_queues(&rec, 2, RECORDED_QUEUE_CAP);
    assert_eq!(q[0].len(), 1);
    assert_eq!(q[1].len(), 1);

    // The cap bounds each side independently.
    let q = recorded_queues(&rec, 0, 1);
    assert_eq!(q[0].len(), 1);
    assert_eq!(q[1].len(), 1);

    // A `forcelose` ENDS the scan — everything past a forfeit belongs to no turn.
    let mut forfeited = rec.clone();
    forfeited.commands.insert(1, ("forcelose".to_string(), "p1".to_string()));
    let q = recorded_queues(&forfeited, 0, RECORDED_QUEUE_CAP);
    assert_eq!(q[0], vec!["move 1".to_string()], "p1 keeps only the pre-forfeit command");
    assert!(q[1].is_empty(), "p2 has nothing before the forfeit");
}
