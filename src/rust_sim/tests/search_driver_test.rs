//! search_driver_test.rs — the gate for the SEARCH kernels (`gen3_rust_search_driver_v1`,
//! [`pokesim::search`]), covering everything that does NOT need the Node golden.
//!
//! The golden differential (`harness/search_impl_parity.py`, which replayed a captured
//! `search_driver.js` golden through this binary) and the Node driver were DELETED on
//! 2026-10-10 (owner: "delete node"); the properties below — the ones a `cargo test` run must
//! never lose — are the node-free gate:
//!
//!   1. **AUX-RNG BIT-EXACTNESS** — the mulberry32-over-FNV-1a stream matches tables
//!      produced by the (deleted) `replay_kernels.js`'s OWN `auxRngFromSeed` under node. A single wrong
//!      operator (a signed shift, a non-wrapping multiply, a missing `>>> 0`) diverges by
//!      draw 1, and every "random" arm in a search is downstream of this.
//!   2. **`random_choice` DETERMINISM + LEGALITY** — the same seed reproduces the same
//!      pick sequence on the same board, a different seed does not, and every pick is in
//!      the board's legal option set (so an arm can never feed a choice the sim refuses
//!      for a reason the kernel should have known).
//!   3. **CLONE INDEPENDENCE ACROSS ARMS** — two arms expanded from one node diverge from
//!      each other and leave the PARENT bit-for-bit untouched; re-expanding the parent
//!      with the same action reproduces the first arm byte-for-byte. This is the property
//!      the whole beam rests on.
//!   4. **THE `stuck` GUARD** — a turn that cannot settle terminates and is REPORTED, and
//!      it does so after EXACTLY Node's iteration count (`if (guard++ > 40)` allows 41
//!      writes, not 40 — an off-by-one here silently changes a `stuck` verdict).
//!
//! Non-vacuity matters as much as in `bridge_clone_branch_test.rs`: each test first
//! asserts the mechanic is genuinely exercised (the fixture really is a mid-battle `move`
//! boundary, the two arms really do diverge, the wedge really did wedge).

use pokesim::battle::BattleOptions;
use pokesim::bridge::{bridge_opts, parse_choice, BridgeSession, Cmd, RequestState};
use pokesim::dex::Dex;
use pokesim::search::{
    aux_rng_from_seed, pick_uniform, random_choice, resolve_turn, resolve_turn_sourced, side_chunk_strings,
    ActionSpec, TurnSource,
};

// ===========================================================================
// Fixture — the `bridge_clone_branch_test` board: a bulky Blissey MIRROR with two
// damaging moves and a live bench, gen3customgame, a fixed seed.
//
// Bulky on purpose: a lead that faints inside the prefix would turn the pause into a
// forced replacement and the branch assertions would stop testing branching. The mirror
// also ties on Speed, which adds the action-order tie-shuffle to the dice stream.
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

/// A session paused at a mid-battle `move` boundary, with a NON-VACUITY guard: the battle
/// must still be running, both sides must be on a `move` request, and both must have a
/// live bench (so `random_choice`'s switch options are real).
fn paused_session(dex: &Dex, turns: usize) -> BridgeSession {
    let mut sess = BridgeSession::new(&opts(), dex).expect("session");
    for _ in 0..turns {
        sess.feed_cmd(cmd(0, "move 1"), dex);
        sess.feed_cmd(cmd(1, "move 1"), dex);
    }
    assert!(!sess.is_ended(), "fixture ended early — it must pause mid-battle");
    assert_eq!(sess.request_kind(0), Some(RequestState::Move), "p1 must be at a move request");
    assert_eq!(sess.request_kind(1), Some(RequestState::Move), "p2 must be at a move request");
    let st = sess.battle_state().expect("state");
    for side in 0..2 {
        let s = &st.sides[side];
        assert!(
            s.pokemon.iter().enumerate().any(|(i, m)| i != s.active && !m.fainted),
            "p{} must have a live bench for the switch options to be real",
            side + 1
        );
    }
    sess
}

// ===========================================================================
// 1. AUX-RNG BIT-EXACTNESS
// ===========================================================================

/// Draw tables produced by running `replay_kernels.js`'s OWN `auxRngFromSeed` under node (the file is
/// deleted since 2026-10-10; recover it with `git show 84796da2:src/utils/bridge/replay_kernels.js`):
///
/// ```text
/// node -e "const K=require('./src/utils/bridge/replay_kernels.js');
///          const r=K.auxRngFromSeed('original');
///          for(let i=0;i<8;i++) console.log(r().toString());"
/// ```
///
/// Four seeds because the FNV-1a hash and the mulberry32 step fail differently: a broken
/// hash still yields a self-consistent stream for any single seed.
const JS_TABLES: [(&str, [f64; 8]); 4] = [
    (
        "original",
        [
            0.4164961345959455,
            0.9473469110671431,
            0.8452919183764607,
            0.7579158749431372,
            0.9486947702243924,
            0.8511117193847895,
            0.8158230790868402,
            0.40831451094709337,
        ],
    ),
    (
        "sodium,deadbeef",
        [
            0.7899761279113591,
            0.5415243275929242,
            0.1143056454602629,
            0.8615184212103486,
            0.7086296426132321,
            0.28879759926348925,
            0.0987005636561662,
            0.20932659949176013,
        ],
    ),
    (
        "sodium,0011223344556677",
        [
            0.11991429282352328,
            0.23673467221669853,
            0.7197471773251891,
            0.6517343944869936,
            0.03725410206243396,
            0.9997551601845771,
            0.801125324331224,
            0.24378477479331195,
        ],
    ),
    (
        "gen5,1234567890abcdef",
        [
            0.33972905902191997,
            0.13040070584975183,
            0.09747063997201622,
            0.5377306716982275,
            0.5261576927732676,
            0.0696458900347352,
            0.7887676761019975,
            0.6241381070576608,
        ],
    ),
];

#[test]
fn aux_rng_matches_the_js_draw_tables() {
    for (seed, want) in JS_TABLES.iter() {
        let mut r = aux_rng_from_seed(seed);
        for (i, w) in want.iter().enumerate() {
            // EXACT: both sides are the same IEEE-754 double (`u32 / 2^32` is exactly
            // representable), so a tolerance would only hide a real divergence.
            assert_eq!(r.next_f64(), *w, "seed {seed} draw {i}");
        }
    }
}

#[test]
fn pick_uniform_matches_js_and_uses_exactly_one_draw() {
    let items: Vec<String> = (0..5).map(|i| format!("move {i}")).collect();
    let mut r = aux_rng_from_seed("original");
    let mut control = aux_rng_from_seed("original");
    // Ground truth: `K.pickUniform(K.auxRngFromSeed('original'), [...5 items])`.
    assert_eq!(pick_uniform(&mut r, &items), "move 2");
    // Exactly one draw consumed: a control advanced once must now track it.
    control.next_f64();
    assert_eq!(r.next_f64(), control.next_f64());
}

// ===========================================================================
// 2. `random_choice` — determinism per seed, and legality
// ===========================================================================

#[test]
fn random_choice_is_deterministic_per_seed_and_always_legal() {
    let dex = Dex::for_gen(3);
    let sess = paused_session(&dex, 2);

    // The board's LEGAL option set, derived independently of the kernel: every move slot
    // with PP that is not disabled, plus every live bench slot (nothing here is trapped).
    let legal = |side: usize| -> Vec<String> {
        let st = sess.battle_state().unwrap();
        let s = &st.sides[side];
        let mon = &s.pokemon[s.active];
        let mut v: Vec<String> = (0..mon.move_pp.len())
            .filter(|&k| mon.move_usable(k, &dex))
            .map(|k| format!("move {}", k + 1))
            .collect();
        v.extend(
            s.pokemon
                .iter()
                .enumerate()
                .filter(|(i, m)| *i != s.active && !m.fainted)
                .map(|(i, _)| format!("switch {}", i + 1)),
        );
        v
    };
    let opts_p1 = legal(0);
    assert!(opts_p1.len() >= 3, "fixture must offer several options, got {opts_p1:?}");

    let run = |seed: &str| -> Vec<String> {
        let mut rng = aux_rng_from_seed(seed);
        (0..12)
            .map(|i| random_choice(&sess, i % 2, &mut rng, &dex).expect("a choice"))
            .collect()
    };

    let a = run("sodium,deadbeef");
    assert_eq!(a, run("sodium,deadbeef"), "same seed must reproduce the pick sequence");
    assert_ne!(
        a,
        run("sodium,cafe0001"),
        "a different seed must not reproduce it (else the seed is not wired through)"
    );
    // Non-vacuity: the sequence must actually vary, else "deterministic" is trivial.
    assert!(a.iter().collect::<std::collections::HashSet<_>>().len() > 1, "picks: {a:?}");
    for (i, c) in a.iter().enumerate() {
        assert!(legal(i % 2).contains(c), "illegal pick {c:?} for p{}", i % 2 + 1);
    }
}

// ===========================================================================
// 3. CLONE INDEPENDENCE ACROSS ARMS
// ===========================================================================

/// Expand `parent` as a search arm would: snapshot, drop the parent's chunk history, and
/// resolve one joint turn with an explicit p1 action.
fn expand(parent: &BridgeSession, p1: &str, dex: &Dex) -> (BridgeSession, Vec<String>, Vec<String>) {
    let mut sess = parent.snapshot();
    sess.clear_chunks();
    let spec = [ActionSpec::Explicit(p1.to_string()), ActionSpec::Explicit("move 1".to_string())];
    let mut rng = aux_rng_from_seed("original");
    let r = resolve_turn(&mut sess, &spec, "random", &mut rng, dex);
    assert!(!r.stuck, "arm {p1} should settle");
    assert_eq!(r.used[0], vec![p1.to_string()], "the explicit p1 action must be the one used");
    let p1c = side_chunk_strings(&sess, 0);
    let p2c = side_chunk_strings(&sess, 1);
    (sess, p1c, p2c)
}

#[test]
fn two_arms_from_one_node_diverge_and_leave_the_parent_untouched() {
    let dex = Dex::for_gen(3);
    let parent = paused_session(&dex, 3);

    let before_turn = parent.turn();
    let before_chunks = side_chunk_strings(&parent, 0);
    let before_req = parent.active_request_json(0).map(str::to_string);
    let before_seed = format!("{:?}", parent.battle_state().unwrap().prng_seed());

    let (_a, a1, a2) = expand(&parent, "move 1", &dex);
    let (_b, b1, b2) = expand(&parent, "move 2", &dex);

    // DIVERGENCE (non-vacuity): the two arms must actually be different battles, else
    // "independent" would be satisfied by two identical no-ops.
    assert_ne!(a1, b1, "the two arms produced identical p1 chunks — nothing was tested");
    assert!(!a1.is_empty() && !a2.is_empty() && !b2.is_empty(), "arms must emit chunks");

    // INDEPENDENCE: the parent is bit-for-bit what it was.
    assert_eq!(parent.turn(), before_turn, "parent turn moved");
    assert_eq!(side_chunk_strings(&parent, 0), before_chunks, "parent chunk stream moved");
    assert_eq!(parent.active_request_json(0).map(str::to_string), before_req, "parent request moved");
    assert_eq!(
        format!("{:?}", parent.battle_state().unwrap().prng_seed()),
        before_seed,
        "parent PRNG advanced — the snapshot shared its dice"
    );

    // REPRODUCIBILITY: a third arm with the first arm's action is byte-identical to it.
    let (_c, c1, c2) = expand(&parent, "move 1", &dex);
    assert_eq!(c1, a1, "re-expanding the same arm must be byte-identical");
    assert_eq!(c2, a2, "re-expanding the same arm must be byte-identical");
}

// ===========================================================================
// 4. THE `stuck` GUARD
// ===========================================================================

#[test]
fn a_wedged_turn_is_reported_stuck_after_nodes_exact_iteration_count() {
    let dex = Dex::for_gen(3);
    let mut sess = paused_session(&dex, 1);

    // WEDGE the session deterministically: answer for p1 TWICE. The second CMD names a
    // side this boundary no longer needs, which the bridge treats as an upstream desync
    // and stops on (`gen3_bridge_stopped_failloud_v1`) — from then on the battle cannot
    // advance, which is exactly the shape `resolve_turn`'s guard exists to bound.
    sess.feed_cmd(cmd(0, "move 1"), &dex);
    sess.feed_cmd(cmd(0, "move 1"), &dex);
    assert!(sess.fatal().is_some(), "the wedge did not take — the guard would not be exercised");
    let turn_before = sess.turn();

    let spec = [ActionSpec::Random, ActionSpec::Random];
    let mut rng = aux_rng_from_seed("original");
    let r = resolve_turn(&mut sess, &spec, "random", &mut rng, &dex);

    assert!(r.stuck, "a turn that cannot settle must be reported stuck");
    assert_eq!(sess.turn(), turn_before, "the wedged turn must not have advanced");
    // Node's `if (guard++ > 40)` compares the value BEFORE the increment, so 41 iterations
    // write before the 42nd trips. p2 is the only side still needed, so it writes once per
    // iteration: 41 entries. A `guard += 1; if guard > 40` port would give 40 and could
    // flip a `stuck` verdict on a turn that settles on its 41st round.
    assert_eq!(r.used[1].len(), 41, "guard arithmetic must match Node's post-increment compare");
    assert!(r.used[0].is_empty(), "p1 already answered; it must not be asked again");
}

// ===========================================================================
// 5. A REPLACEMENT SWITCH THAT ENDS THE TURN STOPS THE ARM (F-LI-1)
// ===========================================================================

/// A Shedinja lead (1 HP, sand-vulnerable) against a Sand Stream Tyranitar: the sand chip at the
/// end of turn 1 KOs the Shedinja, so its side owes a REPLACEMENT switch while the other side
/// WAITS — and that switch is what completes turn 1 and opens turn 2's move requests. Tackle is
/// Normal, so neither lead can hurt the other (Ghost immunity / Wonder Guard): the only faint is
/// the sand chip. `shedinja_side` picks which side holds it.
fn replacement_session(shedinja_side: usize, dex: &Dex) -> BridgeSession {
    let shed = "Shedinja|||wonderguard|tackle|Serious||||||]Blissey|||naturalcure|tackle,headbutt|Serious||||||";
    let ttar = "Tyranitar|||sandstream|tackle|Serious||||||]Regice|||clearbody|tackle,headbutt|Serious||||||";
    let (p1, p2) = if shedinja_side == 0 { (shed, ttar) } else { (ttar, shed) };
    let sess = BridgeSession::new(&bridge_opts("gen3customgame", SEED.to_string(), p1, p2), dex).expect("session");
    let st = sess.battle_state().expect("state");
    assert!(st.sides.iter().all(|s| s.pokemon.len() == 2), "each side must field a lead AND a bench mon");
    assert_eq!(sess.request_kind(0), Some(RequestState::Move), "p1 must open on a move request");
    assert_eq!(sess.request_kind(1), Some(RequestState::Move), "p2 must open on a move request");
    sess
}

/// `resolve_turn_sourced_with` must STOP at the turn boundary even INSIDE one loop iteration.
///
/// The loop feeds p1 then p2. When p1 owes the replacement switch (p2 waits), p1's switch ends
/// turn 1 and the sim opens turn 2's move request for p2 AT ONCE; before the fix the same
/// iteration then answered it from the follow-up policy — a turn-2 choice committed inside the
/// turn-1 arm (M5 Lane I F-LI-1). The child node's engine then had p2's turn-2 choice already
/// made while its leaf (p2's stream: `|turn|2` + a fresh request) said it was p2's decision, and
/// expanding it with the leaf's own token failed the whole batch (`unresolvable choice`).
///
/// The p2-owes-the-switch arm is the CONTROL: its switch is the iteration's last feed, so it was
/// right before the fix too — the property is the same for both sides.
#[test]
fn a_replacement_switch_that_ends_the_turn_does_not_answer_the_next_turn() {
    let dex = Dex::for_gen(3);
    for shed in 0..2 {
        let other = 1 - shed;
        let mut sess = replacement_session(shed, &dex);
        let spec = [ActionSpec::Explicit("move 1".into()), ActionSpec::Explicit("move 1".into())];
        let mut rng = aux_rng_from_seed("original");
        let r = resolve_turn(&mut sess, &spec, "random", &mut rng, &dex);

        // Non-vacuity: the turn really did contain the replacement round, on the side we meant.
        assert!(!r.stuck && !sess.is_ended(), "shed={shed}: the turn must settle and the battle go on");
        assert_eq!(r.used[shed].len(), 2, "shed={shed}: the Shedinja side must move AND replace: {:?}", r.used);
        assert!(r.used[shed][1].starts_with("switch "), "shed={shed}: the follow-up must be the replacement: {:?}", r.used);
        assert_eq!(sess.turn(), 2, "shed={shed}: the replacement must have ended turn 1");

        // THE PROPERTY: the other side answered turn 1 ONCE, and turn 2 is still open for BOTH.
        assert_eq!(r.used[other], vec!["move 1".to_string()], "shed={shed}: a turn-2 choice leaked into the turn-1 arm");
        for side in 0..2 {
            assert_eq!(sess.request_kind(side), Some(RequestState::Move), "shed={shed}: p{} must be at turn 2's move request", side + 1);
            assert!(!sess.is_choice_done(side), "shed={shed}: p{} committed a turn-2 choice inside the turn-1 arm", side + 1);
        }

        // And the child serves the next ply with ITS OWN choices: an explicit turn-2 switch for the
        // non-Shedinja side is the one that runs (before the fix the leaked follow-up ran instead).
        let mut spec2 = [ActionSpec::Explicit("move 1".into()), ActionSpec::Explicit("move 1".into())];
        spec2[other] = ActionSpec::Explicit("switch 2".into());
        let r2 = resolve_turn(&mut sess, &spec2, "random", &mut rng, &dex);
        assert!(!r2.stuck && sess.fatal().is_none(), "shed={shed}: the next ply must resolve: {:?}", sess.fatal());
        assert_eq!(r2.used[other], vec!["switch 2".to_string()], "shed={shed}: the next ply ran another choice");
        assert_eq!(sess.turn(), 3, "shed={shed}: the next ply must be ONE turn");
    }
}

/// The replay family's record-aligned `"recorded"` source shares the loop: a queue that holds
/// the NEXT turn's recorded command must not have it popped into this turn's arm.
#[test]
fn a_recorded_queue_is_not_drained_across_a_replacement_boundary() {
    let dex = Dex::for_gen(3);
    let mut sess = replacement_session(0, &dex);
    let mut sources = [
        TurnSource::Explicit("move 1".into()),
        TurnSource::Queue(["move 1", "switch 2"].iter().map(|s| s.to_string()).collect()),
    ];
    let mut rng = aux_rng_from_seed("original");
    let r = resolve_turn_sourced(&mut sess, &mut sources, "random", &mut rng, &dex);
    assert_eq!(sess.turn(), 2, "the replacement must have ended turn 1");
    assert_eq!(r.used[0].len(), 2, "p1 must move and replace: {:?}", r.used);
    assert_eq!(r.used[1], vec!["move 1".to_string()], "p2's turn-2 recorded command was consumed in turn 1");
    assert!(!sess.is_choice_done(1), "p2 committed a turn-2 choice inside the turn-1 arm");
    match &sources[1] {
        TurnSource::Queue(q) => assert_eq!(q.iter().collect::<Vec<_>>(), vec!["switch 2"], "the queue lost turn 2's entry"),
        other => panic!("source changed kind: {other:?}"),
    }
}
