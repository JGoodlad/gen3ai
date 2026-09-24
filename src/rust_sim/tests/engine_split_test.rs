//! engine_split_test.rs — the gate for the ENGINE / TRANSPORT split of a bridge session
//! (`gen3_core_engine_split_v1`, the Rust Core Program's pre-M3 hand-off).
//!
//! The engine holds the requests as TYPED values and the transport ships their JSON; a version owns
//! the engine alone and forks it into a FRESH transport (`BridgeSession::resume`). Three properties
//! make that sound, each checked at every boundary of the trapping golden (the constructed
//! trapped-switch state machine: `trapped:true` + `update:true` re-requests) and of a disabled-move
//! reject (the `disabledSource` re-request):
//!
//! 1. the engine's typed request, rendered LATER (when a fork or a search node asks), is EXACTLY
//!    the bytes the wire shipped at issue time (`Engine::request_json ==
//!    BridgeSession::active_request_json`) — i.e. nothing the rendering reads moves while a
//!    boundary is open. (The wire itself is rendered from the same typed value, so whether that
//!    value is the RIGHT request is not this file's question: `bridge_test`'s goldens against real
//!    Showdown own it — a flipped `update` flag on the trapped re-request fails
//!    `bridge_incremental_matches_genesis_replay`, verified by mutation.)
//! 2. a resumed transport around an engine clone emits, from that boundary on, the SAME chunks as
//!    the full session it was cloned from, and commits the same decisions;
//! 3. a resumed transport carries no wire history (no chunks, no script, no seed anchors).
//!
//! The byte identity of `sim_bridge` itself is the transcript gate recorded in
//! `designs/research_state/measurements/rust_core_m3_2026-09-24/`, and `bridge_test`'s
//! incremental-vs-genesis parity is unchanged.

use pokesim::battle::{BattleOptions, PackedTeam, PlayerOptions};
use pokesim::bridge::{bridge_opts, parse_bridge_golden, BridgeSession, Cmd, WireChoice};
use pokesim::dex::Dex;

const TRAPPING_GOLDEN: &str = include_str!("vectors/bridge_trapping_golden.txt");

fn assert_requests_typed(sess: &BridgeSession, dex: &Dex, where_: &str) -> usize {
    let mut n = 0;
    for side in 0..2 {
        let wire = sess.active_request_json(side).map(str::to_string);
        let typed = sess.engine().request_json(side, dex);
        assert_eq!(typed, wire, "{where_} p{}: the typed request does not render the shipped bytes", side + 1);
        let issued = sess.engine().issued_json(side).map(|s| s.to_string());
        assert_eq!(issued, wire, "{where_} p{}: the engine's issued bytes are not the wire's", side + 1);
        n += wire.is_some() as usize;
    }
    n
}

/// Every chunk shipped after index `from`, as `(side, lines)`.
fn suffix(sess: &BridgeSession, from: usize) -> Vec<(usize, Vec<String>)> {
    sess.chunks().chunks.iter().skip(from).map(|c| (c.side, c.lines.clone())).collect()
}

/// Drive `cmds` one at a time; at every boundary check (1), and from every boundary fork a resumed
/// transport and check (2) + (3) on the rest of the script. Returns (requests checked, forks).
fn drive(opts: &BattleOptions, cmds: &[Cmd], dex: &Dex, id: &str) -> (usize, usize) {
    let mut sess = BridgeSession::new(opts, dex).expect("session");
    let mut checked = assert_requests_typed(&sess, dex, id);
    let mut forks = 0;
    for (k, c) in cmds.iter().enumerate() {
        if sess.is_ended() || sess.fatal().is_some() {
            break;
        }
        // (2) + (3): from THIS boundary, a resumed engine clone vs the full session.
        let mut full = sess.snapshot();
        let mark = full.chunks().chunks.len();
        let script_mark = full.script().len();
        let mut fork = BridgeSession::resume(sess.engine().clone());
        assert!(fork.chunks().chunks.is_empty() && fork.script().is_empty() && fork.request_seeds().is_empty());
        for rest in &cmds[k..] {
            full.feed_cmd(rest.clone(), dex);
            fork.feed_cmd(rest.clone(), dex);
        }
        let got = suffix(&fork, 0);
        let want = suffix(&full, mark);
        assert_eq!(got, want, "{id} fork@{k}: a resumed engine emitted different chunks");
        assert_eq!(fork.script(), &full.script()[script_mark..], "{id} fork@{k}: different commits");
        forks += 1;

        sess.feed_cmd(c.clone(), dex);
        checked += assert_requests_typed(&sess, dex, &format!("{id} after cmd {k}"));
    }
    (checked, forks)
}

#[test]
fn the_typed_requests_and_a_resumed_fork_match_the_wire_on_the_trapping_golden() {
    let dex = Dex::for_gen(3);
    let battles = parse_bridge_golden(TRAPPING_GOLDEN).expect("golden");
    let (mut checked, mut forks, mut trapped) = (0, 0, 0);
    for b in &battles {
        let opts = bridge_opts(&b.format_id, b.seed.clone(), &b.p1_team, &b.p2_team);
        let (c, f) = drive(&opts, &b.cmds, &dex, &b.id);
        checked += c;
        forks += f;
        trapped += b.p1_expected.iter().chain(&b.p2_expected).filter(|l| l.contains("\"trapped\":true")).count();
    }
    assert!(battles.len() >= 2, "only {} battles", battles.len());
    assert!(trapped > 0, "non-vacuity: the golden must carry a trapped re-request");
    assert!(checked >= 40 && forks >= 20, "only {checked} requests / {forks} forks checked");
}

#[test]
fn a_disabled_source_re_request_is_a_typed_value_too() {
    // The Choice-locked Aerodactyl of `bridge_choice_reject_test`: its refused Earthquake is
    // re-requested with `"update":true` and `"disabledSource":""` on the slot.
    let p1 = "Aerodactyl||choiceband|rockhead|doubleedge,earthquake,rockslide,substitute|||||100|]\
              Snorlax||leftovers|immunity|bodyslam,earthquake,rest,curse|||||100|";
    let p2 = "Snorlax||leftovers|immunity|bodyslam,earthquake,rest,curse|||||100|]\
              Blissey||leftovers|naturalcure|seismictoss,softboiled,toxic,icebeam|||||100|";
    let opts = BattleOptions {
        format_id: "gen3customgame".to_string(),
        seed: Some("7,11,13,17".to_string()),
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(p1.to_string()) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(p2.to_string()) },
    };
    let dex = Dex::for_gen(3);
    let mv = |side, slot| Cmd { side, choice: WireChoice::Move(slot) };
    let cmds = vec![mv(0, 0), mv(1, 0), mv(0, 1), mv(0, 0), mv(1, 0)];
    let (checked, forks) = drive(&opts, &cmds, &dex, "disabled");
    let mut s = BridgeSession::new(&opts, &dex).unwrap();
    for c in &cmds[..3] {
        s.feed_cmd(c.clone(), &dex);
    }
    let req = s.active_request_json(0).expect("the re-request");
    assert!(req.contains("\"disabledSource\":\"\"") && req.contains("\"update\":true"), "non-vacuity: {req}");
    assert_eq!(s.engine().request(0).and_then(|r| r.disabled_source), Some(1));
    assert!(checked >= 6 && forks >= 4, "{checked} / {forks}");
}
