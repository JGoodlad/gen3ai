//! The gender DRAW PAD (`gen3_gender_draw_pad_v1`; packed gender `M*` / `F*` / `N*`,
//! `team::GENDER_DRAW_PAD`) — our extension for the offline search world builder.
//!
//! The turn-0 construction window draws ONE `sample(['M','F'])` for every mon whose packed set
//! names no gender and whose species has no fixed gender (p1's team then p2's, team order). A
//! search world re-draws the opponent's never-seen mons and must leave the pre-root dice exactly
//! where the TRUE team left them, so a world mon on a true draw slot carries the pad: its gender
//! is NAMED (kept), yet it consumes (and discards) exactly one gender draw at its place in the
//! order. These pins assert the OBSERVABLE: the post-construction PRNG state (the full state, the
//! string Showdown's `getSeed` resumes from) and each mon's gender. Every equality is paired with
//! a control the same team WITHOUT the marker fails — the pad is load-bearing, not vacuous.
//!
//! With NO marker nothing changes: `turn0_construction_test.rs` (the sim-captured seeds) and the
//! rest of the differential ladder are the default path's gates.

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::team::{pack, unpack, GENDER_DRAW_PAD};
use pokesim::version::BattleVersion;

fn dex() -> Dex {
    Dex::for_gen(3)
}

fn opts(seed: &str, p1: &str, p2: &str) -> BattleOptions {
    BattleOptions {
        format_id: "gen3customgame".to_string(),
        seed: Some(seed.to_string()),
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(p1.to_string()) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(p2.to_string()) },
    }
}

/// The bridge's RAW-seed entry (`sim_bridge`, `core_events`, `search_driver`, the env core).
fn construct(seed: &str, p1: &str, p2: &str) -> Battle {
    Battle::start_with_turn0_construction(&opts(seed, p1, p2), &dex()).expect("turn-0 construction")
}

fn seed(b: &Battle) -> String {
    b.state().unwrap().prng_seed()
}

fn gender(b: &Battle, side: usize, slot: usize) -> Option<char> {
    b.state().unwrap().side(side).pokemon[slot].gender
}

const SEEDS: [&str; 3] = ["[1,2,3,4]", "[3,2,3,4]", "[5,6,7,8]"];

const SNORLAX_M: &str = "Snorlax||||Splash|Serious||M||||";
const STEELIX_M: &str = "Steelix||||Splash|Serious||M||||";
/// A gender-RATIO'd species with no gender named — a TRUE draw slot.
const GENGAR_U: &str = "Gengar||||Splash|Serious||||||";
const STEELIX_U: &str = "Steelix||||Splash|Serious||||||";

/// Every species gender class a world may put on a true draw slot, with the pad, and the gender
/// it must keep: genderless (Magneton, `N`), fixed-female (Blissey), fixed-male (Hitmonlee), and a
/// ratio'd species with a named gender (Gengar).
const PADDED: [(&str, &str, char); 4] = [
    ("Magneton||||Splash|Serious||N*||||", "Magneton||||Splash|Serious||N||||", 'N'),
    ("Blissey||||Splash|Serious||F*||||", "Blissey||||Splash|Serious||F||||", 'F'),
    ("Hitmonlee||||Splash|Serious||M*||||", "Hitmonlee||||Splash|Serious||M||||", 'M'),
    ("Gengar||||Splash|Serious||F*||||", "Gengar||||Splash|Serious||F||||", 'F'),
];

fn team(sets: &[&str]) -> String {
    sets.join("]")
}

/// A pad mon at slot k leaves the PRNG in exactly the state the TRUE team (an unnamed ratio'd mon
/// at k) leaves it, for every species gender class, and keeps its named gender. The unpadded
/// control (the same named gender, no marker) draws nothing there, so its state DIFFERS.
#[test]
fn a_padded_slot_draws_exactly_like_an_unnamed_ratiod_mon() {
    for s in SEEDS {
        let truth = construct(s, &team(&[SNORLAX_M, GENGAR_U]), STEELIX_M);
        for (padded, plain, g) in PADDED {
            let world = construct(s, &team(&[SNORLAX_M, padded]), STEELIX_M);
            assert_eq!(seed(&world), seed(&truth), "seed {s}, {padded}: the pad must draw like the truth");
            assert_eq!(gender(&world, 0, 1), Some(g), "seed {s}, {padded}: the NAMED gender is kept");
            let control = construct(s, &team(&[SNORLAX_M, plain]), STEELIX_M);
            assert_ne!(seed(&control), seed(&truth), "seed {s}, {plain}: without the pad the dice move");
        }
    }
}

/// A pad on a mon that would draw ANYWAY (a ratio'd species) is still exactly ONE draw, not two:
/// `Gengar|…|F*` equals an unnamed Gengar, and differs from TWO unnamed ratio'd mons.
#[test]
fn a_pad_on_a_ratiod_species_is_one_draw_not_two() {
    for s in SEEDS {
        let one = construct(s, &team(&[SNORLAX_M, GENGAR_U]), STEELIX_M);
        let padded = construct(s, &team(&[SNORLAX_M, "Gengar||||Splash|Serious||M*||||"]), STEELIX_M);
        let two = construct(s, &team(&[SNORLAX_M, GENGAR_U, GENGAR_U]), STEELIX_M);
        assert_eq!(seed(&padded), seed(&one), "seed {s}");
        assert_ne!(seed(&padded), seed(&two), "seed {s}: a second draw would show here");
        assert_eq!(gender(&padded, 0, 1), Some('M'), "seed {s}");
    }
}

/// The pad holds its PLACE in the draw order: a later unnamed ratio'd mon gets the SAME sampled
/// gender as in the truth (it draws the truth's second value). Non-vacuity: without the pad that
/// later mon takes the FIRST value, which differs from the truth on at least one seed.
#[test]
fn a_pad_keeps_its_place_in_the_draw_order() {
    let mut position_mattered = false;
    for s in SEEDS {
        let truth = construct(s, &team(&[SNORLAX_M, GENGAR_U, STEELIX_U]), STEELIX_M);
        let world = construct(s, &team(&[SNORLAX_M, PADDED[0].0, STEELIX_U]), STEELIX_M);
        let control = construct(s, &team(&[SNORLAX_M, PADDED[0].1, STEELIX_U]), STEELIX_M);
        assert_eq!(seed(&world), seed(&truth), "seed {s}");
        assert_eq!(gender(&world, 0, 2), gender(&truth, 0, 2), "seed {s}: the later mon's sampled gender");
        position_mattered |= gender(&control, 0, 2) != gender(&truth, 0, 2);
    }
    assert!(position_mattered, "no seed separates the draw positions — pick seeds whose two draws differ");
}

/// The order spans BOTH sides (p1's team then p2's): a pad on p2's bench, behind an unnamed p1
/// mon, matches the truth's p2 draw slot, and p1's sampled gender is untouched.
#[test]
fn a_pad_on_the_second_side_matches_the_truth() {
    for s in SEEDS {
        let truth = construct(s, &team(&[SNORLAX_M, GENGAR_U]), &team(&[STEELIX_M, GENGAR_U, STEELIX_U]));
        let world = construct(s, &team(&[SNORLAX_M, GENGAR_U]), &team(&[STEELIX_M, PADDED[1].0, STEELIX_U]));
        assert_eq!(seed(&world), seed(&truth), "seed {s}");
        assert_eq!(gender(&world, 0, 1), gender(&truth, 0, 1), "seed {s}: p1's draw");
        assert_eq!(gender(&world, 1, 2), gender(&truth, 1, 2), "seed {s}: p2's later draw");
        assert_eq!(gender(&world, 1, 1), Some('F'), "seed {s}: the padded Blissey stays F");
    }
}

/// The POST-construction-seed path (`start_with_switchins`: the committed seed goldens seed PAST
/// the window) draws no gender at all, so a pad there is inert — the same state as the named mon.
#[test]
fn the_post_construction_seed_path_ignores_the_pad() {
    let d = dex();
    for s in ["1,2,3,4", "3,2,3,4"] {
        let padded = Battle::start_with_switchins(&opts(s, &team(&[SNORLAX_M, PADDED[0].0]), STEELIX_M), &d).unwrap();
        let plain = Battle::start_with_switchins(&opts(s, &team(&[SNORLAX_M, PADDED[0].1]), STEELIX_M), &d).unwrap();
        assert_eq!(seed(&padded), seed(&plain), "seed {s}");
        assert_eq!(gender(&padded, 0, 1), Some('N'));
    }
}

/// The codec: `<g>*` decodes to the named gender + the pad and re-packs byte-identically; no
/// marker means no pad; a marker after anything but exactly one of M / F / N is refused.
#[test]
fn the_marker_round_trips_and_a_malformed_one_is_refused() {
    let d = dex();
    assert_eq!(GENDER_DRAW_PAD, '*');
    for (padded, plain, g) in PADDED {
        let sets = unpack(padded, &d).unwrap();
        assert_eq!((sets[0].gender, sets[0].gender_draw_pad), (Some(g), true), "{padded}");
        assert_eq!(pack(&sets, &d), padded, "{padded} re-packs byte-identically");
        let sets = unpack(plain, &d).unwrap();
        assert_eq!((sets[0].gender, sets[0].gender_draw_pad), (Some(g), false), "{plain}");
        assert_eq!(pack(&sets, &d), plain);
    }
    let unnamed = unpack(GENGAR_U, &d).unwrap();
    assert_eq!((unnamed[0].gender, unnamed[0].gender_draw_pad), (None, false));
    for bad in ["*", "X*", "MF*", "M**", "m*"] {
        let packed = format!("Gengar||||Splash|Serious||{bad}||||");
        assert!(unpack(&packed, &d).is_err(), "{bad:?} must be refused");
    }
}

/// The reading's own-team parse (`BattleVersion::parse_root*`, the env core's and the search
/// game's per-side chain) tolerates a padded team.
#[test]
fn the_reading_accepts_a_padded_team() {
    let t = team(&[SNORLAX_M, PADDED[0].0, PADDED[1].0, PADDED[2].0]);
    BattleVersion::parse_root(1, "P2", Some(&t)).expect("parse_root");
    BattleVersion::parse_root_unrecorded(1, "P2", Some(&t), Default::default()).expect("parse_root_unrecorded");
}
