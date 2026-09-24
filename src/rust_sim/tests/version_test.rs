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
use pokesim::version::{parse_matches_step, streams_equal, BattleVersion};

const P1: &str = "Metagross|||clearbody|meteormash,earthquake,explosion,agility|Adamant|252,252,,,4,|||||]\
Suicune|||pressure|calmmind,surf,rest,icebeam|Bold|252,,252,,4,|||||";
const P2: &str = "Zapdos|||pressure|thunderbolt,hiddenpowerice,rest,toxic|Modest|252,,,252,4,||,0,,30,,|||]\
Snorlax|||immunity|bodyslam,earthquake,rest,curse|Careful|252,,4,,252,|||||";

fn cmd(side: usize, tok: &str) -> Cmd {
    Cmd { side, choice: parse_choice(tok).expect("choice") }
}

fn root(dex: &Dex, compact: bool) -> BattleVersion {
    let opts = bridge_opts("gen3ou", "7,11,13,17".to_string(), P1, P2);
    let sess = BridgeSession::new_construct_turn0_core(&opts, dex).expect("session");
    BattleVersion::root(sess, ["P1", "P2"], [Some(P1), Some(P2)], compact, [true, true]).expect("root")
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
    let mut v = root(&dex, false);
    let mut parsed = [
        Some(BattleVersion::parse_root(0, "P1", Some(P1)).unwrap()),
        Some(BattleVersion::parse_root(1, "P2", Some(P2)).unwrap()),
    ];
    let mut compared = 0;
    for (i, cmds) in std::iter::once(Vec::new()).chain(script()).enumerate() {
        if i > 0 {
            if v.engine().unwrap().is_ended() {
                break;
            }
            v = v.advance_with(|e| {
                e.feed_cmds(&cmds, &dex);
                Ok(())
            })
            .unwrap();
        }
        for side in 0..2 {
            let p = parsed[side].take().unwrap();
            let from = p.stream(side).unwrap().lines;
            let text = v.engine().unwrap().side_lines(side);
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
    let r = Arc::new(root(&dex, true));
    let before = r.view(0).unwrap().json();
    let a = r.step(&[cmd(0, "move 1"), cmd(1, "move 1")], &dex).unwrap();
    let b = r.step(&[cmd(0, "move 2"), cmd(1, "move 1")], &dex).unwrap();
    assert!(Arc::ptr_eq(a.parent().unwrap(), &r) && Arc::ptr_eq(b.parent().unwrap(), &r));
    assert_eq!(r.view(0).unwrap().json(), before, "forking must not touch the parent");
    assert_ne!(a.view(0).unwrap().json(), b.view(0).unwrap().json(), "the two arms must diverge");
    assert!(!a.events(0).is_empty() && !b.events(0).is_empty());
}

#[test]
fn a_compacted_fork_chain_equals_the_linear_replay() {
    let dex = Dex::for_gen(3);
    let mut lin = root(&dex, false);
    let mut fork = Arc::new(root(&dex, true));
    assert_eq!(fork.engine().unwrap().side_line_count(0), 0, "a compacted version keeps no chunk history");
    for cmds in script() {
        if lin.engine().unwrap().is_ended() {
            break;
        }
        lin = lin
            .advance_with(|e| {
                e.feed_cmds(&cmds, &dex);
                Ok(())
            })
            .unwrap();
        fork = fork.step(&cmds, &dex).unwrap();
        assert_eq!(fork.engine().unwrap().side_line_count(1), 0);
        for side in 0..2 {
            assert_eq!(lin.view(side).unwrap(), fork.view(side).unwrap(), "p{} view", side + 1);
            assert!(lin.stream(side).unwrap().tracker == fork.stream(side).unwrap().tracker);
        }
    }
}

#[test]
fn the_typed_shortcut_and_the_text_path_fold_the_same_version() {
    let dex = Dex::for_gen(3);
    let r = Arc::new(root(&dex, true));
    let mut n = 0;
    for cmds in script().into_iter().take(3) {
        let mut e = r.engine().unwrap().snapshot();
        e.feed_cmds(&cmds, &dex);
        let typed = r.child(e.snapshot()).unwrap();
        let text = r.child_text(e).unwrap();
        for side in 0..2 {
            streams_equal(&typed, &text, side).unwrap();
            n += 1;
        }
    }
    assert!(n >= 6);
    // TEETH: two different successors must NOT compare equal (the integrity check can fail).
    let a = r.step(&[cmd(0, "move 1"), cmd(1, "move 1")], &dex).unwrap();
    let b = r.step(&[cmd(0, "move 2"), cmd(1, "move 1")], &dex).unwrap();
    assert!(streams_equal(&a, &b, 0).is_err(), "streams_equal must see a different successor");
}

#[test]
fn the_board_audit_passes_the_truth_and_catches_a_tampered_view() {
    let dex = Dex::for_gen(3);
    let mut v = Arc::new(root(&dex, true));
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
    let mut v = Arc::new(root(&dex, true));
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
