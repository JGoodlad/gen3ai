//! obs_facts_truth_test.rs — the OBS-FACTS block (`gen3_obs_facts_v1`) against the ENGINE, the
//! referee, at every decision of seeded random-legal battles built around its mechanics.
//!
//! Slice O holds the Rust encoder byte-equal to the Python one; this holds the ENCODED FACT equal to
//! the sim's truth, which no parity gate can (both encoders read the same reading):
//!
//! * SCREENS — the encoded turns left on Reflect / Light Screen / Safeguard equals the engine's
//!   remaining `duration` exactly, both sides (Mist is not modelled by the engine);
//! * VOL — present exactly when the engine holds Encore / Taunt / Disable / Uproar / the partial trap,
//!   and the engine's remaining duration lies inside the encoded [min_left, max_left];
//! * CHOICE — SOUND evidence: while the engine's opponent active is choice-locked, neither NOT-locked
//!   proof fires and the stint's first move IS the locked move.

use pokesim::bridge::{bridge_opts, parse_choice, BridgeSession, Cmd};
use pokesim::dex::Dex;
use pokesim::encoder::{self, layout::*, OBS_DIM};
use pokesim::present::choice_tokens;
use pokesim::state::{BattleState, MonState};
use pokesim::trackers::clock::ClockConfig;
use pokesim::version::BattleVersion;

const P1: &str = "Tyranitar||choiceband|sandstream|rockslide,earthquake,crunch,firepunch|Adamant|252,252,,,4,|||||]\
Smeargle||leftovers|owntempo|encore,taunt,disable,uproar|Jolly|252,,4,,,252|||||]\
Starmie||leftovers|naturalcure|reflect,lightscreen,safeguard,surf|Timid|252,,4,,,252|||||";
const P2: &str = "Snorlax||choiceband|immunity|bodyslam,earthquake,shadowball,return|Adamant|252,252,,,4,|||||]\
Gyarados||leftovers|intimidate|whirlpool,taunt,hydropump,dragondance|Adamant|252,252,,,4,|||||]\
Mew||leftovers|synchronize|encore,disable,reflect,firespin|Timid|252,,4,,,252|||||";

/// A tiny deterministic LCG (no RNG crate in the test).
struct Lcg(u64);
impl Lcg {
    fn pick(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % n
    }
}

#[derive(Default, Debug)]
struct Tally {
    decisions: usize,
    screens: usize,
    vol: [usize; 5],
    locked: usize,
    failures: Vec<String>,
}

fn screen_rem(st: &BattleState, side: usize, j: usize) -> u8 {
    let sd = &st.sides[side];
    match j {
        0 => sd.reflect,
        1 => sd.light_screen,
        2 => sd.safeguard,
        _ => 0, // Mist: not modelled by the engine
    }
}

fn vol_rem(m: &MonState, j: usize) -> Option<u8> {
    match j {
        0 => m.encore.map(|(_, t)| t),
        1 => m.taunt,
        2 => m.disable.map(|(_, t)| t),
        3 => m.uproar.map(|(t, _)| t),
        _ => m.partial_trap.as_ref().map(|p| p.duration),
    }
}

fn check(row: &[f32; OBS_DIM], st: &BattleState, viewer: usize, where_: &str, t: &mut Tally) {
    let f = &row[OFFSET_OBS_FACTS..OFFSET_OBS_FACTS + OBS_FACTS_DIM];
    t.decisions += 1;
    for (s, side) in [viewer, 1 - viewer].into_iter().enumerate() {
        // SCREENS
        for j in 0..3 {
            let enc = f[FACTS_SCREENS_OFFSET + s * FACTS_SCREENS.len() + j] as f64 * FACTS_SCREEN_TURNS as f64;
            let want = screen_rem(st, side, j) as f64;
            if want > 0.0 {
                t.screens += 1;
            }
            if (enc - want).abs() > 1e-6 {
                t.failures.push(format!("{where_} side {s} {}: encoded {enc} engine {want}", FACTS_SCREENS[j]));
            }
        }
        // VOL
        let sd = &st.sides[side];
        let m = &sd.pokemon[sd.active];
        if m.hp == 0 {
            continue;
        }
        for j in 0..FACTS_VOL_EFFECTS.len() {
            let o = FACTS_VOL_OFFSET + s * FACTS_VOL_SIDE_DIM + j * FACTS_VOL_CELL_DIM;
            let (lo, hi) = (f[o + 1] as f64 * FACTS_TURN_NORM, f[o + 2] as f64 * FACTS_TURN_NORM);
            match vol_rem(m, j) {
                None => {
                    if f[o] != 0.0 || lo != 0.0 || hi != 0.0 {
                        t.failures.push(format!("{where_} side {s} {}: encoded [{lo},{hi}] engine none", FACTS_VOL_EFFECTS[j]));
                    }
                }
                Some(rem) => {
                    t.vol[j] += 1;
                    let r = rem as f64;
                    if !(lo - 1e-6 <= r && r <= hi + 1e-6) {
                        t.failures.push(format!("{where_} side {s} {}: engine {r} outside encoded [{lo},{hi}]", FACTS_VOL_EFFECTS[j]));
                    }
                }
            }
        }
    }
    // CHOICE: the opponent active, while the engine says it is locked
    let sd = &st.sides[1 - viewer];
    let m = &sd.pokemon[sd.active];
    if m.hp > 0 {
        if let Some(k) = m.choice_locked_move {
            t.locked += 1;
            let c = &f[FACTS_CHOICE_OFFSET..FACTS_CHOICE_OFFSET + FACTS_CHOICE_DIM];
            let mv = pokesim::dex::to_id(&m.set.moves[k]);
            let num = encoder::data::tables().moves.get(&mv).map_or(-1.0, |r| r.num as f64);
            if c[0] != 0.0 || c[1] != 0.0 {
                t.failures.push(format!("{where_}: a NOT-locked proof fired on a locked {mv} ({c:?})"));
            }
            if (c[2] as f64 - num).abs() > 1e-6 {
                t.failures.push(format!("{where_}: stint first move {} is not the locked {mv} ({num})", c[2]));
            }
        }
    }
}

fn play(seed: u64, t: &mut Tally) {
    let dex = Dex::for_gen(3);
    let s = seed as u32;
    let opts = bridge_opts("gen3customgame", format!("{},{},{},{}", s + 1, s + 2, s + 3, s + 4), P1, P2);
    let mut sess = BridgeSession::new_construct_turn0_core(&opts, &dex).expect("session");
    let cfg = Some(ClockConfig::default());
    let mut v = BattleVersion::observe_root_with(&sess, ["P1", "P2"], [Some(P1), Some(P2)], [true, true], cfg).expect("root");
    let mut rng = Lcg(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
    for step in 0..120 {
        if sess.is_ended() {
            break;
        }
        let mut cmds = Vec::new();
        for side in 0..2 {
            if v.decision(side).is_none() {
                continue;
            }
            let mut row = [0.0f32; OBS_DIM];
            v.encode(side, &mut row).expect("encode");
            let st = sess.battle_state().expect("board");
            check(&row, st, side, &format!("seed {seed} step {step} p{}", side + 1), t);
            let legal = v.legal(side).expect("legal");
            let reading = &v.stream(side).expect("stream").board_reading;
            let toks = choice_tokens(reading, &legal).expect("tokens");
            if toks.is_empty() {
                continue;
            }
            // lean on moves (index ≥ 6) 3:1 so the volatiles get used
            let moves: Vec<&(usize, String)> = toks.iter().filter(|(i, _)| *i >= 6).collect();
            let tok = if !moves.is_empty() && rng.pick(4) != 0 { &moves[rng.pick(moves.len())].1 } else { &toks[rng.pick(toks.len())].1 };
            cmds.push(Cmd { side, choice: parse_choice(tok).expect("choice") });
        }
        if cmds.is_empty() {
            break;
        }
        sess.feed_cmds(&cmds, &dex);
        v = v.observe(&sess).expect("observe");
    }
}

#[test]
fn every_encoded_fact_matches_the_engine_at_every_decision() {
    let mut t = Tally::default();
    for seed in 0..60u64 {
        play(seed, &mut t);
    }
    eprintln!("obs-facts truth: {t:?}");
    assert!(t.failures.is_empty(), "{} failures, first: {:#?}", t.failures.len(), &t.failures[..t.failures.len().min(10)]);
    // coverage: a gate that never saw the mechanic proves nothing
    assert!(t.decisions > 2_000, "{t:?}");
    assert!(t.screens > 50, "{t:?}");
    assert!(t.vol.iter().all(|&n| n > 5), "every duration volatile exercised: {t:?}");
    assert!(t.locked > 50, "{t:?}");
}
