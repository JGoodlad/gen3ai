//! The `>start` CONSTRUCTION-WINDOW gate (`gen3_turn0_construction_v1`).
//!
//! `Battle::start_with_turn0_construction` (the bridge's raw-seed entry) must reach
//! the real sim's PRE-FIRST-DECISION PRNG state — bit-for-bit — from the RAW `>start`
//! seed, for EVERY gen3ou lead configuration: distinct-speed, speed-TIED, a
//! weather-setter tie, and an unspecified-gender mon (the two draw families the old
//! pure `advance_seed_for_construction` seed hack missed — the speed-tie eachEvent
//! shuffles + the per-mon gender `sample(['M','F'])`).
//!
//! Ground truth captured from the omniscient `BattleStream` (`battle.prng.getSeed()` +
//! each mon's `.gender` + `field.weather`) at the first `|request|`, seed `[1,2,3,4]`,
//! `gen3customgame` — the same convention `state_test`/`fullbattle_test` use, but here
//! we feed the RAW seed and reproduce the construction rather than seeding past it.
//! Regenerate via `node /tmp/probe_gt.js` (the probe in the Phase-B session) after any
//! PRNG/construction-draw change.

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;

fn dex() -> Dex {
    Dex::for_gen(3)
}

/// Construct from a RAW seed via the turn-0 window; return the live `Battle`.
fn construct(p1: &str, p2: &str) -> Battle {
    let opts = BattleOptions {
        format_id: "gen3customgame".to_string(),
        seed: Some("[1,2,3,4]".to_string()), // the RAW `>start` bracketed-array form
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(p1.to_string()) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(p2.to_string()) },
    };
    Battle::start_with_turn0_construction(&opts, &dex()).expect("turn-0 construction")
}

fn gender(b: &Battle, side: usize, slot: usize) -> Option<char> {
    b.state().unwrap().side(side).pokemon[slot].gender
}
fn seed(b: &Battle) -> String {
    b.state().unwrap().prng_seed()
}
fn weather(b: &Battle) -> Option<pokesim::state::Weather> {
    b.state().unwrap().field.weather
}

// --- Packed teams (from `node /tmp/probe_pk.js`, the Phase-B probe). ---
const ELECTRODE_M: &str = "Electrode||||splash|Serious|,,,,,252|M||||";
const SNORLAX_M: &str = "Snorlax||||splash|Serious||M||||";
const GENGAR_U_FAST: &str = "Gengar||||splash|Serious|,,,,,252|||||";
const STEELIX_U: &str = "Steelix||||splash|Serious||||||";
const TTAR_SAND_M: &str = "Tyranitar|||SandStream|splash|Serious||M||||";
const BENCH_P1: &str = "Snorlax||||splash|Serious||M||||]Gengar||||splash|Serious||||||";
const BENCH_P2: &str = "Steelix||||splash|Serious||M||||]Magneton||||splash|Serious||||||";

#[test]
fn distinct_speed_explicit_gender_draws_only_the_quick_claw() {
    // Electrode (spe 379) vs Snorlax (spe 96): distinct speed, explicit genders → the
    // construction draws ONLY the turn-0 Quick Claw (no tie shuffles, no gender sample).
    let b = construct(ELECTRODE_M, SNORLAX_M);
    assert_eq!(seed(&b), "30982,33910,19571,50263", "distinct-speed post-construction seed");
    assert_eq!(gender(&b, 0, 0), Some('M'));
    assert_eq!(gender(&b, 1, 0), Some('M'));
    assert!(weather(&b).is_none());
}

#[test]
fn speed_tie_explicit_gender_draws_the_endturn_shuffles() {
    // Snorlax mirror (spe 96 tie): the insertChoice tie-break + 3 eachEvent('Update')
    // shuffles + Quick Claw — the draws the old seed hack missed.
    let b = construct(SNORLAX_M, SNORLAX_M);
    assert_eq!(seed(&b), "55250,62519,52978,42619", "speed-tie post-construction seed");
    assert!(weather(&b).is_none());
}

#[test]
fn distinct_speed_unspecified_gender_draws_the_gender_samples() {
    // Gengar (unspec, spe 319) vs Steelix (unspec, spe 96): distinct speed, so no tie
    // shuffles — but each ratio'd mon draws ONE uniform `sample(['M','F'])` (p1 then
    // p2), then the Quick Claw. The sampled genders MUST match the sim (M / F).
    let b = construct(GENGAR_U_FAST, STEELIX_U);
    assert_eq!(seed(&b), "43514,9542,40559,8561", "distinct+gender post-construction seed");
    assert_eq!(gender(&b, 0, 0), Some('M'), "p1 Gengar sampled gender");
    assert_eq!(gender(&b, 1, 0), Some('F'), "p2 Steelix sampled gender");
    assert!(weather(&b).is_none());
}

#[test]
fn speed_tie_unspecified_gender_composes_samples_and_shuffles() {
    // Gengar mirror (spe 319 tie), both unspecified gender: 2 gender samples THEN the
    // tie window (insertChoice + 3 Updates) THEN Quick Claw — the full composition.
    let b = construct(GENGAR_U_FAST, GENGAR_U_FAST);
    assert_eq!(seed(&b), "62891,40560,22227,62965", "tie+gender post-construction seed");
    assert_eq!(gender(&b, 0, 0), Some('M'));
    assert_eq!(gender(&b, 1, 0), Some('F'));
}

#[test]
fn weather_setter_speed_tie_draws_the_weatherchange_shuffle() {
    // Tyranitar mirror (Sand Stream, spe 158 tie): the tie window gains an EXTRA
    // eachEvent('WeatherChange') shuffle (Update, WeatherChange, Update, Update) when a
    // lead's ability sets the weather — the interleaving the machinery reuse handles.
    let b = construct(TTAR_SAND_M, TTAR_SAND_M);
    assert_eq!(seed(&b), "37673,46633,62039,8266", "weather-tie post-construction seed");
    assert_eq!(weather(&b), Some(pokesim::state::Weather::Sand), "Sand Stream set the weather");
    assert_eq!(gender(&b, 0, 0), Some('M'));
    assert_eq!(gender(&b, 1, 0), Some('M'));
}

#[test]
fn gender_samples_only_ratiod_bench_mons_in_team_order() {
    // p1 = [Snorlax (M, explicit), Gengar (unspec)]; p2 = [Steelix (M, explicit),
    // Magneton (GENDERLESS)]. Only the ratio'd BENCH Gengar draws a `sample` — the
    // explicit mons + the genderless Magneton draw-free. Snorlax leads (spe 96 tie).
    let b = construct(BENCH_P1, BENCH_P2);
    assert_eq!(seed(&b), "37673,46633,62039,8266", "bench-mix post-construction seed");
    assert_eq!(gender(&b, 0, 0), Some('M'), "p1 lead Snorlax explicit");
    assert_eq!(gender(&b, 0, 1), Some('M'), "p1 bench Gengar SAMPLED");
    assert_eq!(gender(&b, 1, 0), Some('M'), "p2 lead Steelix explicit");
    // Magneton is genderless: species.gender 'N' → set draw-free (the sim maps 'N' → ""
    // so its `|switch|` details show no suffix; `switch_details` renders 'N' the same).
    assert_eq!(gender(&b, 1, 1), Some('N'), "p2 bench Magneton genderless — no draw");
    assert!(weather(&b).is_none());
}
