//! obs_stage_truth_test.rs — two observation cells against the ENGINE, the referee, on one constructed battle
//! each (`gen3_toxic_stage_scale_v1`, `gen3_wish_flag_truth_v1`).
//!
//! * TOXIC — a Blissey (Soft-Boiled + Leftovers) is badly poisoned on turn 1 by a Zapdos that then only Protects. Its
//!   heal outpaces the ramp until the 12th tick (hand-checked: HP 64 after tick 12, dead at tick 13), so the battle
//!   shows every stage 1..=12 at a decision — past the old 8-tick saturation of the cell — and at EVERY decision, for
//!   BOTH viewers, the encoded cell is the engine's `Toxic(n)` stage over its true cap: `min(n, 15) / 15`.
//! * WISH — the board's "Wish pending" flag (`SideTrackers::wish_pending`, 0.5 on the row) is the engine's own slot
//!   condition (`SideState::wish_pending`, duration 2 -> 1 -> lands) at EVERY decision of seeded random battles with
//!   Wish on every team: it is set exactly when the Wish LANDS AT THE NEXT END-OF-TURN RESIDUAL. That covers the failed
//!   double-Wish (`|move|…|Wish||[still]` + `|-fail|`), the wisher's faint (gen 3 does not cancel it: the slot's next
//!   occupant is healed), and the phase of the decision: a replacement decision AFTER the end-of-turn faint is on the
//!   same turn number as the residual that already ran, so it must drop a Wish that just landed and keep one cast
//!   this turn (`gen3_wish_flag_truth_v1`).
//!
//! The cell is found by the status one-hot (the slot of the mon whose condition is `tox`), not by an index the test
//! assumed, and the test asserts it SAW the toxic mon at each decision (a gate that never saw the mechanic proves
//! nothing).

use pokesim::bridge::{bridge_opts, parse_choice, BridgeSession, Cmd};
use pokesim::dex::Dex;
use pokesim::encoder::{layout::*, OBS_DIM};
use pokesim::present::choice_tokens;
use pokesim::state::{BattleState, Status};
use pokesim::trackers::clock::ClockConfig;
use pokesim::version::BattleVersion;

const BLISSEY: &str = "Blissey||leftovers|naturalcure|softboiled,toxic,protect,seismictoss|Calm|252,,,,252,|||||]\
Snorlax|||immunity|bodyslam,earthquake,rest,curse|Careful|252,,4,,252|||||";
const ZAPDOS: &str = "Zapdos||leftovers|pressure|toxic,protect,rest,substitute|Modest|252,,,252,4,||,0,,30,,|||]\
Snorlax|||immunity|bodyslam,earthquake,rest,curse|Careful|252,,4,,252|||||";

fn cmd(side: usize, tok: &str) -> Cmd {
    Cmd { side, choice: parse_choice(tok).expect("choice") }
}

/// `min(n, 15) / 15` in the row's own f32 arithmetic (the encoder writes an f64 quotient narrowed to f32).
fn cell(n: u8) -> f32 {
    (n.min(TOXIC_STAGE_MAX as u8) as f64 / TOXIC_STAGE_MAX as f64) as f32
}

/// The toxic cells of one team block that sit under a `tox` status one-hot: `(slot, cell)`.
fn tox_cells(row: &[f32; OBS_DIM], team_offset: usize) -> Vec<(usize, f32)> {
    (0..TEAM_SIZE)
        .filter_map(|s| {
            let base = team_offset + s * POKEMON_FULL_DIM;
            // CONDITION one-hot: [none, brn, par, slp, frz, psn, tox] — tox is the last of the 7.
            (row[base + POKEMON_CONDITION_OFFSET + 6] == 1.0).then(|| (s, row[base + POKEMON_COUNTER_OFFSET + 1]))
        })
        .collect()
}

fn engine_stage(st: &BattleState, side: usize) -> Option<u8> {
    st.sides[side].pokemon.iter().find_map(|m| match m.status {
        Some(Status::Toxic(n)) if !m.fainted => Some(n),
        _ => None,
    })
}

#[test]
fn the_toxic_cell_is_the_engines_stage_over_its_true_cap_of_15() {
    let dex = Dex::for_gen(3);
    // A seed on which the turn-1 Toxic lands (85 % accuracy): found once, then FIXED — asserted, not branched on.
    let opts = bridge_opts("gen3customgame", "1,2,3,4".to_string(), BLISSEY, ZAPDOS);
    let mut sess = BridgeSession::new_construct_turn0_core(&opts, &dex).expect("session");
    let cfg = Some(ClockConfig::default());
    let mut v = BattleVersion::observe_root_with(&sess, ["P1", "P2"], [Some(BLISSEY), Some(ZAPDOS)], [true, true], cfg)
        .expect("root");
    let mut seen: Vec<u8> = Vec::new();
    for turn in 1..=14 {
        if sess.is_ended() {
            break;
        }
        // p1 Soft-Boiled every turn; p2 Toxic on turn 1, then Protect (no damage ever reaches Blissey)
        let cmds = vec![cmd(0, "move 1"), cmd(1, if turn == 1 { "move 1" } else { "move 2" })];
        sess.feed_cmds(&cmds, &dex);
        v = v.observe(&sess).expect("observe");
        if sess.is_ended() {
            break;
        }
        let st = sess.battle_state().expect("board");
        let Some(n) = engine_stage(st, 0) else { continue };
        for side in 0..2 {
            if v.decision(side).is_none() {
                continue;
            }
            let mut row = [f32::NAN; OBS_DIM];
            v.encode(side, &mut row).expect("encode");
            // viewer `side` sees Blissey in its OWN team block (side 0) or the OPP block (side 1)
            let block = if side == 0 { OFFSET_OUR_TEAM } else { OFFSET_OPP_TEAM };
            let got = tox_cells(&row, block);
            assert_eq!(got.len(), 1, "turn {turn} p{}: exactly one toxic slot expected, got {got:?}", side + 1);
            assert_eq!(got[0].1, cell(n), "turn {turn} p{}: engine stage {n}, encoded {}", side + 1, got[0].1);
            // …and the OTHER block holds no toxic mon (Zapdos is healthy)
            let other = if side == 0 { OFFSET_OPP_TEAM } else { OFFSET_OUR_TEAM };
            assert!(tox_cells(&row, other).is_empty());
        }
        seen.push(n);
    }
    assert!(seen.contains(&1), "Toxic never landed on turn 1 — pick another seed ({seen:?})");
    assert!(seen.iter().any(|&n| n >= 12), "the battle never reached stage 12 at a decision ({seen:?}) — vacuous");
    // the old cell would have read 1.0 (= 8/8) from stage 8 on; the new one keeps climbing
    assert!(cell(9) > cell(8) && cell(12) != cell(8) && cell(15) == 1.0 && cell(40) == 1.0);
}

// ------------------------------------------------------------------------------------------------------- WISH

const WISH_P1: &str = "Blissey||leftovers|naturalcure|wish,protect,softboiled,seismictoss|Calm|252,,,,252,|||||]\
Tyranitar||leftovers|sandstream|rockslide,earthquake,crunch,wish|Adamant|252,252,,,4,|||||]\
Starmie||leftovers|naturalcure|surf,icebeam,wish,recover|Timid|252,,4,,,252|||||";
const WISH_P2: &str = "Umbreon||leftovers|synchronize|wish,protect,toxic,faintattack|Calm|252,,,,252,|||||]\
Snorlax||leftovers|immunity|bodyslam,earthquake,wish,rest|Careful|252,,4,,252|||||]\
Zapdos||leftovers|pressure|thunderbolt,wish,drillpeck,protect|Modest|252,,,252,4,||,0,,30,,|||";

/// A tiny deterministic LCG (no RNG crate in the test).
struct Lcg(u64);
impl Lcg {
    fn pick(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % n
    }
}

#[derive(Default, Debug)]
struct WishTally {
    decisions: usize,
    /// side-readings at which the engine's slot condition was one residual from landing (the flag must be set)
    landing_next: usize,
    /// …of which AFTER this turn's residual (`|upkeep|` read): the Wish was cast this turn, lands at the end of the next
    landing_next_post_residual: usize,
    /// side-readings AFTER this turn's residual where the side's Wish had just landed (the flag must be CLEAR)
    just_landed_post_residual: usize,
    /// …where the side had a fainted mon and a Wish still pending (the wisher's faint does not cancel it)
    pending_with_a_faint: usize,
    /// failed double-Wish casts played (`|move|…|Wish||[still]`)
    failed_casts: usize,
    mismatches: Vec<String>,
}

fn play_wish(seed: u64, t: &mut WishTally) {
    let dex = Dex::for_gen(3);
    let s = seed as u32;
    let opts = bridge_opts("gen3customgame", format!("{},{},{},{}", s + 1, s + 2, s + 3, s + 4), WISH_P1, WISH_P2);
    let mut sess = BridgeSession::new_construct_turn0_core(&opts, &dex).expect("session");
    let cfg = Some(ClockConfig::default());
    let mut v = BattleVersion::observe_root_with(&sess, ["P1", "P2"], [Some(WISH_P1), Some(WISH_P2)], [true, true], cfg)
        .expect("root");
    let mut rng = Lcg(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
    // each (viewer, relative side)'s engine duration at the PREVIOUS decision, to see a Wish land between two
    let mut prev: [[Option<u8>; 2]; 2] = [[None; 2]; 2];
    for step in 0..80 {
        if sess.is_ended() {
            break;
        }
        let mut cmds = Vec::new();
        for side in 0..2 {
            if v.decision(side).is_none() {
                continue;
            }
            let st = sess.battle_state().expect("board");
            let trk = v.trackers(side).expect("trackers");
            let post = v.view(side).expect("view").residual_done;
            t.decisions += 1;
            for rel in 0..2 {
                let abs = if rel == 0 { side } else { 1 - side };
                let dur = st.sides[abs].wish_pending.as_ref().map(|(d, _)| *d);
                let want = dur == Some(1);
                let got = trk.wish_pending[rel];
                if got != want {
                    t.mismatches.push(format!(
                        "seed {seed} step {step} viewer p{} rel {rel}: tracker {got}, engine duration {dur:?}, residual_done {post}",
                        side + 1
                    ));
                }
                t.landing_next += want as usize;
                t.landing_next_post_residual += (want && post) as usize;
                t.just_landed_post_residual += (post && dur.is_none() && prev[side][rel] == Some(1)) as usize;
                t.pending_with_a_faint += (want && st.sides[abs].pokemon.iter().any(|m| m.fainted)) as usize;
                prev[side][rel] = dur;
            }
            let legal = v.legal(side).expect("legal");
            let reading = &v.stream(side).expect("stream").board_reading;
            let toks = choice_tokens(reading, &legal).expect("tokens");
            if toks.is_empty() {
                continue;
            }
            // lean on moves 2:1 (Wish is among every set's moves), uniformly among them
            let moves: Vec<&(usize, String)> = toks.iter().filter(|(i, _)| *i >= 6).collect();
            let tok = if !moves.is_empty() && rng.pick(3) != 0 { &moves[rng.pick(moves.len())].1 } else { &toks[rng.pick(toks.len())].1 };
            cmds.push(Cmd { side, choice: parse_choice(tok).expect("choice") });
        }
        if cmds.is_empty() {
            break;
        }
        sess.feed_cmds(&cmds, &dex);
        v = v.observe(&sess).expect("observe");
    }
    t.failed_casts += sess.side_lines(0).iter().filter(|l| l.contains("|Wish||[still]")).count();
}

#[test]
fn the_wish_flag_is_the_engines_slot_condition_one_residual_from_landing() {
    let mut t = WishTally::default();
    for seed in 0..120u64 {
        play_wish(seed, &mut t);
    }
    eprintln!("wish truth: {t:?}");
    assert!(t.mismatches.is_empty(), "{} mismatches, first: {:#?}", t.mismatches.len(), &t.mismatches[..t.mismatches.len().min(8)]);
    // coverage: a gate that never saw the mechanic proves nothing
    assert!(t.decisions > 3_000, "{t:?}");
    assert!(t.landing_next > 500, "{t:?}");
    assert!(t.failed_casts > 100, "no failed double-Wish was played: {t:?}");
    assert!(t.landing_next_post_residual >= 10, "no post-residual replacement decision with a Wish cast this turn: {t:?}");
    assert!(t.just_landed_post_residual >= 10, "no post-residual decision after a Wish had just landed: {t:?}");
    assert!(t.pending_with_a_faint >= 10, "no Wish stayed pending past a faint: {t:?}");
}
