//! version_test.rs — the gate for the Rust core's persistent battle state (`gen3_core_version_v1`,
//! the Rust Core Program's M2): `BattleVersion` built from the simulator's step and from one side's
//! text, forks as shared handles, compaction, the typed-vs-text integrity check, and the teeth of
//! the board audit.
//!
//! The corpus-scale gate is `core_events` (every parity-corpus battle is replayed as a version chain
//! and each side's parse-built twin must agree at every step — `version::parse_matches_step`); this
//! file pins the properties one battle can show, each with a non-vacuity guard.

use std::sync::Arc;

use pokesim::bridge::{bridge_opts, parse_choice, BridgeSession, Cmd};
use pokesim::dex::Dex;
use pokesim::present::check_view;
use pokesim::version::{parse_matches_step, BattleVersion};

const P1: &str = "Metagross|||clearbody|meteormash,earthquake,explosion,agility|Adamant|252,252,,,4,|||||]\
Suicune|||pressure|calmmind,surf,rest,icebeam|Bold|252,,252,,4,|||||";
const P2: &str = "Zapdos|||pressure|thunderbolt,hiddenpowerice,rest,toxic|Modest|252,,,252,4,||,0,,30,,|||]\
Snorlax|||immunity|bodyslam,earthquake,rest,curse|Careful|252,,4,,252,|||||";

fn cmd(side: usize, tok: &str) -> Cmd {
    Cmd { side, choice: parse_choice(tok).expect("choice") }
}

fn session(dex: &Dex) -> BridgeSession {
    let opts = bridge_opts("gen3ou", "7,11,13,17".to_string(), P1, P2);
    BridgeSession::new_construct_turn0_core(&opts, dex).expect("session")
}

/// A FORK root: the version owns the engine, the transport is dropped.
fn root(dex: &Dex) -> BattleVersion {
    BattleVersion::root(session(dex), ["P1", "P2"], [Some(P1), Some(P2)], [true, true]).expect("root")
}

/// A LINEAR chain's root: the version OBSERVES a session the test drives.
fn observed(sess: &BridgeSession) -> BattleVersion {
    BattleVersion::observe_root(sess, ["P1", "P2"], [Some(P1), Some(P2)], [true, true]).expect("root")
}

/// A fixed script that exercises a switch, damage both ways, status and a stat boost.
fn script() -> Vec<Vec<Cmd>> {
    vec![
        vec![cmd(0, "move 4"), cmd(1, "move 4")],
        vec![cmd(0, "move 1"), cmd(1, "move 1")],
        vec![cmd(0, "switch 2"), cmd(1, "move 2")],
        vec![cmd(0, "move 1"), cmd(1, "switch 2")],
        vec![cmd(0, "move 4"), cmd(1, "move 1")],
        vec![cmd(0, "move 2"), cmd(1, "move 4")],
    ]
}

#[test]
fn a_step_built_chain_equals_its_parse_built_twin_version_by_version() {
    let dex = Dex::for_gen(3);
    let mut sess = session(&dex);
    let mut v = observed(&sess);
    let mut parsed = [
        Some(BattleVersion::parse_root(0, "P1", Some(P1)).unwrap()),
        Some(BattleVersion::parse_root(1, "P2", Some(P2)).unwrap()),
    ];
    let mut compared = 0;
    for (i, cmds) in std::iter::once(Vec::new()).chain(script()).enumerate() {
        if i > 0 {
            if sess.is_ended() {
                break;
            }
            sess.feed_cmds(&cmds, &dex);
            v = v.observe(&sess).unwrap();
        }
        for side in 0..2 {
            let p = parsed[side].take().unwrap();
            let from = p.stream(side).unwrap().lines;
            let text = sess.side_lines(side);
            let next = p.parse_advance(&text[from..]).unwrap();
            parse_matches_step(&v, &next, side).unwrap_or_else(|e| panic!("step {i} p{}: {e}", side + 1));
            parsed[side] = Some(next);
            compared += 1;
        }
    }
    assert!(compared >= 10, "only {compared} versions compared");
    let view = v.view(0).unwrap();
    assert!(view.turn >= 4, "the script must actually play turns (turn {})", view.turn);
    assert!(view.opp.mons.len() == 2, "both foes revealed");
}

#[test]
fn a_fork_leaves_its_parent_untouched_and_shares_its_past() {
    let dex = Dex::for_gen(3);
    let r = Arc::new(root(&dex));
    let before = r.view(0).unwrap().json();
    let a = r.step(&[cmd(0, "move 1"), cmd(1, "move 1")], &dex).unwrap();
    let b = r.step(&[cmd(0, "move 2"), cmd(1, "move 1")], &dex).unwrap();
    assert!(Arc::ptr_eq(a.parent().unwrap(), &r) && Arc::ptr_eq(b.parent().unwrap(), &r));
    assert_eq!(r.view(0).unwrap().json(), before, "forking must not touch the parent");
    assert_ne!(a.view(0).unwrap().json(), b.view(0).unwrap().json(), "the two arms must diverge");
    assert!(!a.events(0).is_empty() && !b.events(0).is_empty());
}

#[test]
fn a_fork_chain_equals_the_linear_replay() {
    let dex = Dex::for_gen(3);
    let mut sess = session(&dex);
    let mut lin = observed(&sess);
    let mut fork = Arc::new(root(&dex));
    assert!(lin.engine().is_none(), "an observed version holds no engine — the caller's session is the referee");
    let mut compared = 0;
    for cmds in script() {
        if sess.is_ended() {
            break;
        }
        sess.feed_cmds(&cmds, &dex);
        lin = lin.observe(&sess).unwrap();
        fork = fork.step(&cmds, &dex).unwrap();
        for side in 0..2 {
            assert_eq!(lin.view(side).unwrap(), fork.view(side).unwrap(), "p{} view", side + 1);
            assert!(lin.stream(side).unwrap().board_reading == fork.stream(side).unwrap().board_reading);
            compared += 1;
        }
        let board = sess.battle_state().unwrap();
        assert!(lin.audit_on(0, board, &dex).unwrap().divergences.is_empty());
    }
    assert!(compared >= 10, "only {compared} versions compared");
}

/// `gen3_core_engine_split_v1`: a fork's transport starts EMPTY at the parent's boundary — no
/// chunk history, no script, no seed anchors — and its outstanding requests are the engine's
/// typed values rendered to the exact bytes the parent's wire shipped.
#[test]
fn a_fork_session_carries_the_engine_and_no_wire_history() {
    let dex = Dex::for_gen(3);
    let mut sess = session(&dex);
    for cmds in script().into_iter().take(3) {
        sess.feed_cmds(&cmds, &dex);
    }
    assert!(sess.script().len() >= 3 && !sess.request_seeds().is_empty(), "non-vacuity: the wire has history");
    let want = [sess.active_request_json(0).map(str::to_string), sess.active_request_json(1).map(str::to_string)];
    assert!(want[0].is_some() && want[1].is_some(), "non-vacuity: a boundary is open");
    let v = Arc::new(BattleVersion::root(sess, ["P1", "P2"], [Some(P1), Some(P2)], [true, true]).unwrap());
    let f = v.fork_session().unwrap();
    assert_eq!(f.side_line_count(0) + f.side_line_count(1), 0);
    assert!(f.script().is_empty() && f.request_seeds().is_empty());
    for side in 0..2 {
        assert_eq!(f.active_request_json(side).map(str::to_string), want[side], "p{}", side + 1);
    }
}

/// The ONE observation path (`parse(render)`, program §6c; the typed shortcut is deleted): a
/// search tree's versions are forked from a session that records NO source (its lines carry no
/// engine scope, so every outcome's owner comes from line order), and must fold the SAME stream —
/// reading board, events with their owners, view — as a recording session's, whose lines carry
/// the engine's scope. TEETH: a different successor does not compare equal.
#[test]
fn a_non_recording_fork_folds_the_same_stream_as_a_recording_one() {
    let dex = Dex::for_gen(3);
    let opts = bridge_opts("gen3ou", "7,11,13,17".to_string(), P1, P2);
    let plain = BridgeSession::new_construct_turn0(&opts, &dex).expect("session");
    assert!(!plain.is_core());
    let rp = Arc::new(BattleVersion::root(plain, ["P1", "P2"], [Some(P1), Some(P2)], [true, true]).unwrap());
    let rc = Arc::new(root(&dex));
    let (mut vp, mut vc) = (Arc::clone(&rp), Arc::clone(&rc));
    let mut n = 0;
    for cmds in script() {
        if vc.engine().map_or(true, |e| e.is_ended()) {
            break;
        }
        vp = vp.step(&cmds, &dex).unwrap();
        vc = vc.step(&cmds, &dex).unwrap();
        for side in 0..2 {
            parse_matches_step(&vc, &vp, side).unwrap_or_else(|e| panic!("p{}: {e:?}", side + 1));
            n += 1;
        }
    }
    assert!(n >= 8, "only {n} boundaries compared");
    let a = rc.step(&[cmd(0, "move 1"), cmd(1, "move 1")], &dex).unwrap();
    let b = rp.step(&[cmd(0, "move 2"), cmd(1, "move 1")], &dex).unwrap();
    assert!(parse_matches_step(&a, &b, 0).is_err(), "a different successor must not compare equal");
}

#[test]
fn the_board_audit_passes_the_truth_and_catches_a_tampered_view() {
    let dex = Dex::for_gen(3);
    let mut v = Arc::new(root(&dex));
    for cmds in script().into_iter().take(3) {
        v = v.step(&cmds, &dex).unwrap();
    }
    let board = v.engine().unwrap().battle_state().unwrap();
    let view = v.view(0).unwrap().clone();
    let clean = check_view(&view, board, 0, &dex, true);
    assert!(clean.divergences.is_empty(), "{:?}", clean.divergences);
    assert!(clean.checks > 60, "only {} checks", clean.checks);

    let mut bad = view.clone();
    bad.ours.mons[0].current_hp += 1;
    let a = check_view(&bad, board, 0, &dex, true);
    assert!(a.divergences.iter().any(|(c, _)| c == "ours.hp"), "{:?}", a.divergences);

    let mut bad = view.clone();
    let i = bad.opp.active.unwrap();
    bad.opp.mons[i].item = Some("choiceband".into());
    let a = check_view(&bad, board, 0, &dex, true);
    assert!(a.divergences.iter().any(|(c, _)| c == "opp.item-is-held"), "{:?}", a.divergences);

    let mut bad = view.clone();
    bad.opp.mons[i].boosts.push(("spe", 1));
    let a = check_view(&bad, board, 0, &dex, true);
    assert!(a.divergences.iter().any(|(c, _)| c.ends_with("boosts")), "{:?}", a.divergences);
}

/// V15 — only the active mon the CURRENT request re-synced has the sim's PP; every other own mon's
/// is poke-env's sighting count, which can LAG a deduction it cannot see (found by slice V on
/// procedural teams: a Forretress's Explosion into an unannounced Pressure, the user fainting
/// before any request re-synced it — view 7, engine 6). The audit names it instead of failing,
/// and still fails a count that runs AHEAD of the sim.
#[test]
fn the_audit_names_v15_on_an_unsynced_own_mon_and_keeps_its_teeth() {
    let dex = Dex::for_gen(3);
    let mut v = Arc::new(root(&dex));
    for cmds in script().into_iter().take(3) {
        v = v.step(&cmds, &dex).unwrap();
    }
    let board = v.engine().unwrap().battle_state().unwrap();
    let view = v.view(0).unwrap().clone();
    let act = view.ours.active.expect("an active mon");
    let bench = (0..view.ours.mons.len()).find(|j| *j != act).expect("a benched mon");
    assert!(!view.ours.mons[bench].moves.is_empty(), "non-vacuity: the benched mon has moves");

    // A benched mon's count one AHEAD of the sim's deduction (the sighting count missed one).
    let mut lag = view.clone();
    lag.ours.mons[bench].moves[0].current_pp += 1;
    let a = check_view(&lag, board, 0, &dex, true);
    assert!(a.divergences.is_empty(), "V15 is a named rule, not a divergence: {:?}", a.divergences);
    assert_eq!(a.rules_fired.get("V15"), Some(&1));
    // …but a count BELOW the sim's is impossible for a sighting count: still a divergence.
    let mut ahead = view.clone();
    ahead.ours.mons[bench].moves[0].current_pp -= 1;
    let a = check_view(&ahead, board, 0, &dex, true);
    assert!(a.divergences.iter().any(|(c, _)| c == "ours.[V15] moves[pp]"), "{:?}", a.divergences);
    // The request-synced active mon is held to the sim exactly.
    let mut act_bad = view.clone();
    act_bad.ours.mons[act].moves[0].current_pp += 1;
    let a = check_view(&act_bad, board, 0, &dex, true);
    assert!(a.divergences.iter().any(|(c, _)| c == "ours.moves[pp]"), "{:?}", a.divergences);
    // With no active block in the current request, the active mon falls under V15 too.
    let a = check_view(&act_bad, board, 0, &dex, false);
    assert!(a.divergences.is_empty() && a.rules_fired.get("V15") == Some(&1), "{:?}", a.divergences);
}

#[test]
fn a_parse_built_version_has_no_engine_and_refuses_to_step() {
    let dex = Dex::for_gen(3);
    let p = Arc::new(BattleVersion::parse_root(0, "P1", Some(P1)).unwrap());
    assert!(p.engine().is_none());
    assert!(p.step(&[cmd(0, "move 1")], &dex).is_err());
    assert!(p.audit(0, &dex).is_err(), "no board, no audit");
}
