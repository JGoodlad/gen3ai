//! encoder_test.rs — the Rust encoder's own invariants on a real battle (`gen3_core_encoder_v1`, the
//! Rust Core Program's M4). The byte parity against the Python encoder is slice O
//! (`agents/battle/rust_core_parity_obs.py`); this file pins what one battle can show without it:
//!
//! * in a TEST build the row is NaN-prefilled, and at every decision of both viewers the encode
//!   leaves NO NaN — every cell of every block is written, on every branch the battle takes;
//! * the step-built and the parse-built versions encode the SAME bytes (the §6c one-path property:
//!   the row is a function of one side's stream);
//! * a stream without trackers REFUSES to encode, and a wrong-length row is REFUSED, not converted.

use pokesim::bridge::{bridge_opts, parse_choice, BridgeSession, Cmd};
use pokesim::dex::Dex;
use pokesim::encoder::{self, wire, OBS_DIM};
use pokesim::trackers::clock::ClockConfig;
use pokesim::version::BattleVersion;

const P1: &str = "Metagross|||clearbody|meteormash,earthquake,explosion,agility|Adamant|252,252,,,4,|||||]\
Suicune|||pressure|calmmind,surf,rest,icebeam|Bold|252,,252,,4,|||||";
const P2: &str = "Zapdos|||pressure|thunderbolt,hiddenpowerice,rest,toxic|Modest|252,,,252,4,||,0,,30,,|||]\
Snorlax|||immunity|bodyslam,earthquake,rest,curse|Careful|252,,4,,252,|||||";

fn cmd(side: usize, tok: &str) -> Cmd {
    Cmd { side, choice: parse_choice(tok).expect("choice") }
}

fn script() -> Vec<Vec<Cmd>> {
    vec![
        vec![cmd(0, "move 4"), cmd(1, "move 4")],
        vec![cmd(0, "move 1"), cmd(1, "move 1")],
        vec![cmd(0, "switch 2"), cmd(1, "move 2")],
        vec![cmd(0, "move 1"), cmd(1, "switch 2")],
        vec![cmd(0, "move 4"), cmd(1, "move 1")],
        vec![cmd(0, "move 2"), cmd(1, "move 4")],
        vec![cmd(0, "move 3"), cmd(1, "move 3")],
    ]
}

/// Every decision's row, per viewer, from the step-built chain and from each side's parse chain.
fn rows() -> (Vec<[Vec<u8>; 2]>, Vec<[Vec<u8>; 2]>, usize) {
    let dex = Dex::for_gen(3);
    let opts = bridge_opts("gen3ou", "7,11,13,17".to_string(), P1, P2);
    let mut sess = BridgeSession::new_construct_turn0_core(&opts, &dex).expect("session");
    let cfg = Some(ClockConfig::default());
    let mut v = BattleVersion::observe_root_with(&sess, ["P1", "P2"], [Some(P1), Some(P2)], [true, true], cfg)
        .expect("root");
    let mut parsed = [
        Some(BattleVersion::parse_root_with(0, "P1", Some(P1), cfg).unwrap()),
        Some(BattleVersion::parse_root_with(1, "P2", Some(P2), cfg).unwrap()),
    ];
    let (mut step, mut parse) = (Vec::new(), Vec::new());
    let mut nan_free = 0;
    for (i, cmds) in std::iter::once(Vec::new()).chain(script()).enumerate() {
        if i > 0 {
            if sess.is_ended() {
                break;
            }
            sess.feed_cmds(&cmds, &dex);
            v = v.observe(&sess).unwrap();
        }
        let mut s_rows: [Vec<u8>; 2] = [Vec::new(), Vec::new()];
        let mut p_rows: [Vec<u8>; 2] = [Vec::new(), Vec::new()];
        for side in 0..2 {
            let p = parsed[side].take().unwrap();
            let from = p.stream(side).unwrap().lines;
            let next = p.parse_advance(&sess.side_lines(side)[from..]).unwrap();
            if v.decision(side).is_some() {
                let mut row = [0.0f32; OBS_DIM];
                v.encode(side, &mut row).unwrap_or_else(|e| panic!("step {i} p{}: {e:?}", side + 1));
                let nan: Vec<usize> = (0..OBS_DIM).filter(|&k| row[k].is_nan()).collect();
                assert!(nan.is_empty(), "step {i} p{}: {} cells never written, first {:?}", side + 1, nan.len(), &nan[..nan.len().min(8)]);
                nan_free += 1;
                s_rows[side] = wire::row_bytes(&row);
                assert!(next.decision(side).is_some(), "the parse chain decided too");
                let mut prow = [0.0f32; OBS_DIM];
                next.encode(side, &mut prow).unwrap();
                p_rows[side] = wire::row_bytes(&prow);
            }
            parsed[side] = Some(next);
        }
        step.push(s_rows);
        parse.push(p_rows);
    }
    (step, parse, nan_free)
}

#[test]
fn every_cell_is_written_at_every_decision_under_the_nan_poison() {
    assert!(encoder::NAN_POISON, "cargo test is a NaN-prefill build");
    let (_, _, n) = rows();
    assert!(n >= 10, "only {n} decisions encoded — vacuous");
}

#[test]
fn the_step_built_and_the_parse_built_versions_encode_the_same_bytes() {
    let (step, parse, n) = rows();
    assert!(n >= 10);
    assert_eq!(step, parse);
}

#[test]
fn a_stream_without_trackers_refuses_and_a_wrong_length_row_is_refused() {
    let dex = Dex::for_gen(3);
    let opts = bridge_opts("gen3ou", "7,11,13,17".to_string(), P1, P2);
    let sess = BridgeSession::new_construct_turn0_core(&opts, &dex).expect("session");
    let v = BattleVersion::observe_root(&sess, ["P1", "P2"], [Some(P1), Some(P2)], [true, true]).expect("root");
    let mut row = [0.0f32; OBS_DIM];
    let err = v.encode(0, &mut row).unwrap_err();
    assert!(err.message().contains("no trackers"), "{err:?}");

    let cfg = Some(ClockConfig::default());
    let v = BattleVersion::observe_root_with(&sess, ["P1", "P2"], [Some(P1), Some(P2)], [true, true], cfg).expect("root");
    let s = v.stream(0).unwrap();
    let view = v.view(0).unwrap();
    let legal = v.legal(0);
    let trk = s.trk.as_ref().unwrap();
    let inp = encoder::Inputs { reading: &s.board_reading, view, legal: legal.as_ref(), trackers: &trk.trackers };
    let mut short = vec![0.0f32; OBS_DIM - 1];
    let err = encoder::encode_slice(&inp, &mut short).unwrap_err();
    assert!(err.message().contains("2500 cells"), "{err:?}");
    assert!(short.iter().all(|x| *x == 0.0), "a refused row is left untouched");
    let mut exact = vec![0.0f32; OBS_DIM];
    encoder::encode_slice(&inp, &mut exact).unwrap();
}
