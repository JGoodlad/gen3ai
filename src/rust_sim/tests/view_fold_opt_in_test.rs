//! view_fold_opt_in_test.rs — `gen3_view_fold_opt_in_v1`: the one-sided view's reveal fold
//! (`SideObservation::observe`, run per shipped line in `BridgeChunks::push_chunk_lines`) is OFF
//! unless a session that reads the view turns it on.
//!
//! Why this is pinned: the fold is the view road's input and nothing else, yet the TRAINING
//! transport (`sim_bridge`) paid it on every line — ≈ 60 % of M1's +13 % per-decision Rust CPU
//! (`designs/research_state/measurements/m1_transport_throughput_2026-09-23/`). A training session
//! must build NO view; these tests fail the day one does.

use pokesim::bridge::{bridge_opts, BridgeSession};
use pokesim::dex::Dex;
use pokesim::search::{aux_rng_from_seed, resolve_turn, ActionSpec};
use pokesim::view::one_sided_view;

const P1: &str = "Metagross|||clearbody|meteormash,earthquake,explosion,agility|Adamant|252,252,,,4,|||||]\
Suicune|||pressure|calmmind,surf,rest,icebeam|Bold|252,,252,,4,|||||]\
Gengar|||levitate|thunderbolt,icepunch,hypnosis,explosion|Timid|,,,252,4,252|||||";
const P2: &str = "Zapdos|||pressure|thunderbolt,hiddenpowerice,rest,toxic|Modest|252,,,252,4,||,0,,30,,|||]\
Snorlax|||immunity|bodyslam,earthquake,rest,curse|Careful|252,,4,,252,|||||]\
Skarmory|||keeneye|spikes,roar,drillpeck,rest|Impish|252,,252,,4,|||||";

fn opts() -> pokesim::battle::BattleOptions {
    bridge_opts("gen3ou", "7,11,13,17".to_string(), P1, P2)
}

/// Play `sess` for up to `turns` whole turns with the search kernels' seeded random policy.
fn play(sess: &mut BridgeSession, turns: usize, dex: &Dex) -> usize {
    let mut rng = aux_rng_from_seed("view-fold-opt-in");
    let spec = [ActionSpec::parse("random"), ActionSpec::parse("random")];
    let mut n = 0;
    while n < turns && !sess.is_ended() && sess.fatal().is_none() {
        let r = resolve_turn(sess, &spec, "random", &mut rng, dex);
        assert!(!r.stuck, "the random policy stalled");
        n += 1;
    }
    n
}

fn all_chunks(sess: &BridgeSession) -> Vec<(usize, Vec<String>)> {
    sess.chunks().chunks.iter().map(|c| (c.side, c.lines.clone())).collect()
}

/// THE PIN: `sim_bridge`'s constructor (`new_construct_turn0`) — and every other plain or core
/// constructor — builds no view, across a whole battle.
#[test]
fn a_training_session_never_builds_the_view() {
    let dex = Dex::for_gen(3);
    let mut sess = BridgeSession::new_construct_turn0(&opts(), &dex).expect("session");
    let turns = play(&mut sess, 400, &dex);
    assert!(turns >= 5, "the battle must actually be played ({turns} turns)");
    assert!(!sess.chunks().chunks.is_empty(), "lines must have shipped");
    assert!(!sess.view_fold_enabled(), "a training session folded the one-sided view");
    assert!(sess.observed(0).is_none() && sess.observed(1).is_none());
    assert!(one_sided_view(&sess, 0, &dex).is_err(), "a view without its fold must refuse, not read empty");

    for s in [
        BridgeSession::new(&opts(), &dex).unwrap(),
        BridgeSession::new_with_quick_claw(&opts(), true, &dex).unwrap(),
        BridgeSession::new_core(&opts(), false, &dex).unwrap(),
        BridgeSession::new_construct_turn0_core(&opts(), &dex).unwrap(),
    ] {
        assert!(!s.view_fold_enabled(), "a constructor turned the view fold on by default");
    }
}

/// The training binary's source builds its session with the fold-free constructor and never asks
/// for the view. A source scan, because a unit test of the library cannot see what `main` calls.
#[test]
fn the_sim_bridge_binary_never_turns_the_fold_on() {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/bin/sim_bridge.rs"))
        .expect("read sim_bridge.rs");
    assert!(src.contains("BridgeSession::new_construct_turn0("), "sim_bridge's constructor moved; re-point this pin");
    for banned in ["enable_view_fold", "one_sided_view", ".observed("] {
        assert!(!src.contains(banned), "sim_bridge.rs names `{banned}` — the training transport must build no view");
    }
}

/// The fold changes no shipped byte: a folding and a non-folding session emit identical chunks.
#[test]
fn the_fold_changes_no_shipped_byte() {
    let dex = Dex::for_gen(3);
    let mut off = BridgeSession::new_construct_turn0(&opts(), &dex).unwrap();
    let mut on = BridgeSession::new_construct_turn0(&opts(), &dex).unwrap();
    on.enable_view_fold().unwrap();
    let a = play(&mut off, 400, &dex);
    let b = play(&mut on, 400, &dex);
    assert_eq!(a, b);
    assert_eq!(all_chunks(&off), all_chunks(&on));
}

/// LAZY: enabling the fold late folds the history shipped so far, so the view equals the one an
/// eager fold gives; after a `clear_chunks` with the fold off, a late enable REFUSES.
#[test]
fn a_late_enable_folds_the_history_and_a_cleared_history_refuses() {
    let dex = Dex::for_gen(3);
    let mut eager = BridgeSession::new_construct_turn0(&opts(), &dex).unwrap();
    eager.enable_view_fold().unwrap();
    let mut late = BridgeSession::new_construct_turn0(&opts(), &dex).unwrap();
    let n = play(&mut eager, 6, &dex);
    assert_eq!(play(&mut late, 6, &dex), n);
    assert!(n >= 3);
    late.enable_view_fold().unwrap();
    for side in 0..2 {
        let v = one_sided_view(&eager, side, &dex).unwrap();
        assert!(v.contains("\"opp\""), "non-vacuity: a real view");
        assert_eq!(v, one_sided_view(&late, side, &dex).unwrap(), "p{} late != eager", side + 1);
    }
    // Idempotent.
    late.enable_view_fold().unwrap();

    let mut cleared = BridgeSession::new_construct_turn0(&opts(), &dex).unwrap();
    play(&mut cleared, 2, &dex);
    cleared.clear_chunks();
    assert!(cleared.enable_view_fold().is_err(), "a fold started mid-battle must refuse");
    // …while a session that folded before the clear keeps its fold across it.
    eager.clear_chunks();
    assert!(eager.view_fold_enabled());
}
