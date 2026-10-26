//! silent_volatile_end_test.rs — `gen3_volatile_silent_ends_v1` (2026-10-26 GIGO audit A-C1 / A-C2): the
//! volatiles the sim ENDS WITHOUT A PROTOCOL LINE, read off the stream exactly when the engine drops them.
//!
//! The reading ended an effect only on a line, at `|turn|` (an `ends_on_turn` row) or when its holder left the
//! field, so four classes stayed set on the observation's active-context volatile cells for the rest of a stint:
//!
//! * Endure / Snatch / Helping Hand (`duration: 1`) — now `ends_on_turn`, like Protect;
//! * Destiny Bond, Grudge, Rage — removed at the holder's next move ATTEMPT (`onBeforeMove`; Destiny Bond also in
//!   `onMoveAborted`, i.e. a `|cant|`), unless the move re-uses Destiny Bond;
//! * Mean Look's `trapped` — removed when its TRAPPER leaves (`removeLinkedVolatiles`);
//! * and `trapped` is Baton-Pass COPIED in gen 3 (`data/mods/gen4/conditions.ts`), which the reading did not copy.
//!
//! Each battle is checked TWO ways at every boundary: the engine-truth board audit for both viewers (which now names
//! `destinybond` and `trapped`) and the explicit expectation on the view, so a revert fails here and the audit keeps
//! the class from coming back on any corpus battle (`agents/battle/core_corpus_test.py` runs it at every decision).

use std::sync::Arc;

use pokesim::bridge::{bridge_opts, parse_choice, BridgeSession, Cmd};
use pokesim::dex::Dex;
use pokesim::present::check_view;
use pokesim::version::BattleVersion;

fn cmd(side: usize, tok: &str) -> Cmd {
    Cmd { side, choice: parse_choice(tok).expect("choice") }
}

fn root(dex: &Dex, p1: &str, p2: &str) -> Arc<BattleVersion> {
    let opts = bridge_opts("gen3ou", "7,11,13,17".to_string(), p1, p2);
    let sess = BridgeSession::new_construct_turn0_core(&opts, dex).expect("session");
    Arc::new(BattleVersion::root(sess, ["P1", "P2"], [Some(p1), Some(p2)], [true, true]).expect("root"))
}

/// The volatile keys `viewer` reads on `who`'s ACTIVE mon (`who` 0 = the viewer's own side, 1 = the foe).
fn vols(v: &BattleVersion, viewer: usize, who: usize) -> Vec<String> {
    let view = v.view(viewer).expect("view");
    let side = if who == 0 { &view.ours } else { &view.opp };
    let a = side.active.expect("an active mon");
    side.mons[a].volatiles.iter().map(|(k, _)| k.clone()).collect()
}

/// Step, then the board audit for both viewers must be clean (the engine is the referee).
fn step(v: &Arc<BattleVersion>, cmds: &[Cmd], dex: &Dex) -> Arc<BattleVersion> {
    let n = v.step(cmds, dex).expect("step");
    let board = n.engine().expect("engine").battle_state().expect("board");
    for viewer in 0..2 {
        let a = check_view(n.view(viewer).expect("view"), board, viewer, dex, true);
        assert!(a.divergences.is_empty(), "viewer p{}: {:?}", viewer + 1, a.divergences);
    }
    n
}

#[test]
fn destiny_bond_ends_at_the_next_move_and_endure_at_the_turn() {
    let dex = Dex::for_gen(3);
    let p1 = "Gengar|||levitate|destinybond,endure,thunderbolt,hypnosis|Timid|4,,,252,,252|||||]\
              Snorlax|||immunity|bodyslam,rest,curse,earthquake|Careful|252,,4,,252,|||||";
    let p2 = "Snorlax|||immunity|curse,rest,sleeptalk,selfdestruct|Careful|252,,4,,252,|||||]\
              Blissey|||naturalcure|softboiled,seismictoss,toxic,aromatherapy|Bold|252,,252,,4,|||||";
    let mut v = root(&dex, p1, p2);
    v = step(&v, &[cmd(0, "move 1"), cmd(1, "move 1")], &dex);         // turn 1: Destiny Bond / Curse
    assert!(vols(&v, 1, 1).contains(&"destinybond".to_string()), "non-vacuity: the bond is set {:?}", vols(&v, 1, 1));
    v = step(&v, &[cmd(0, "move 2"), cmd(1, "move 1")], &dex);         // turn 2: Endure (ends the bond) / Curse
    for viewer in 0..2 {
        let who = if viewer == 0 { 0 } else { 1 };
        let have = vols(&v, viewer, who);
        assert!(!have.contains(&"destinybond".to_string()), "p{}: Destiny Bond outlived its next move: {have:?}", viewer + 1);
        assert!(!have.contains(&"endure".to_string()), "p{}: Endure outlived its turn: {have:?}", viewer + 1);
    }
    v = step(&v, &[cmd(0, "move 3"), cmd(1, "move 1")], &dex);         // turn 3: Thunderbolt / Curse
    assert!(!vols(&v, 1, 1).iter().any(|k| k == "destinybond" || k == "endure"), "{:?}", vols(&v, 1, 1));
}

#[test]
fn trapped_ends_when_its_trapper_leaves_and_passes_with_baton_pass() {
    let dex = Dex::for_gen(3);
    let p1 = "Umbreon|||synchronize|meanlook,curse,rest,toxic|Careful|252,,4,,252,|||||]\
              Snorlax|||immunity|bodyslam,rest,curse,earthquake|Careful|252,,4,,252,|||||";
    let p2 = "Celebi|||naturalcure|batonpass,recover,calmmind,psychic|Bold|252,,252,,4,|||||]\
              Starmie|||naturalcure|recover,surf,thunderbolt,icebeam|Timid|4,,,252,,252|||||";
    let trapped = |v: &BattleVersion| vols(v, 0, 1).contains(&"trapped".to_string());
    let mut v = root(&dex, p1, p2);
    v = step(&v, &[cmd(0, "move 1"), cmd(1, "move 2")], &dex);         // Mean Look / Recover
    assert!(trapped(&v), "non-vacuity: Celebi is trapped {:?}", vols(&v, 0, 1));
    v = step(&v, &[cmd(0, "switch 2"), cmd(1, "move 2")], &dex);       // the trapper leaves
    assert!(!trapped(&v), "trapped outlived its trapper: {:?}", vols(&v, 0, 1));
    v = step(&v, &[cmd(0, "switch 2"), cmd(1, "move 2")], &dex);       // Umbreon back
    v = step(&v, &[cmd(0, "move 1"), cmd(1, "move 2")], &dex);         // Mean Look again
    assert!(trapped(&v), "non-vacuity: trapped again {:?}", vols(&v, 0, 1));
    v = step(&v, &[cmd(0, "move 2"), cmd(1, "move 1")], &dex);         // Curse / Baton Pass
    v = step(&v, &[cmd(1, "switch 2")], &dex);                          // the pass lands on Starmie
    assert!(trapped(&v), "a Baton Pass did not carry `trapped` to the receiver: {:?}", vols(&v, 0, 1));
}
