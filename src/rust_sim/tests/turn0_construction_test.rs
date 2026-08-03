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
    construct_seeded("[1,2,3,4]", p1, p2)
}

/// Same, at an explicit RAW `>start` seed (the bracketed-array form poke-env sends).
fn construct_seeded(seed: &str, p1: &str, p2: &str) -> Battle {
    let opts = BattleOptions {
        format_id: "gen3customgame".to_string(),
        seed: Some(seed.to_string()),
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

// ===========================================================================
// The turn-0 MIRROR ORDER gate (`gen3_turn0_construction_mirror_order_v1`, ROUND 34).
//
// The construction window's PRNG model was already bit-for-bit (the four seed pins
// above), but the bridge RE-EMITS the leads' switch-in ability lines afterwards from the
// post-construction board, and that reconstruction re-derived the fire order as "faster
// raw Speed first, a TIE keeps side order". At a raw-Speed tie the TRUE order is the
// `insertChoice` `random(firstIndex, lastIndex+1)` draw the construction ALREADY made —
// p2-FIRST half the time — so every p2-first tie emitted the Intimidate / weather-setter
// block in the WRONG order while the seed + board stayed correct. That is the
// `sbd_msdd8698_b293` Masquerain-mirror divergence the external-consistency gate
// (`harness/gen_sim_bridge_diff.js`) caught: node `-ability|p2a` first, port `p1a` first.
//
// GROUND TRUTH from the real sim (`harness/probe_r34_mirror_order_groundtruth.js`):
//   Masquerain mirror (Intimidate, spe 156 TIE) — seed [1,2,3,4] -> p2a,p1a
//                                                 seed [3,2,3,4] -> p1a,p2a
//   Masquerain (156) vs Tauros (256), both Intimidate, DISTINCT speed -> p2a,p1a
//
// BOTH tie cases are asserted on purpose: a test that only ever saw the p2-first board
// would also pass on a hard-coded flip (and the PRE-FIX code passes the p1-first case),
// so only the PAIR pins the draw as the discriminator.
// ===========================================================================

use pokesim::bridge::{bridge_opts, BridgeSession};

const MASQ_INTIM_M: &str = "Masquerain|||Intimidate|splash|Serious||M||||";
/// Same ability, EV-invested Speed (256) — strictly faster than Masquerain's 156, so the
/// order is decided by Speed and NOT by the tie draw (the fallback model's control).
const TAUROS_INTIM_FAST_M: &str = "Tauros|||Intimidate|splash|Serious|252,,,,,|M||||";

/// The `p<N>a` idents of the emitted `|-ability|…|Intimidate|boost` lines, in emission
/// order, from a battle built the way `sim_bridge` builds one (RAW `>start` seed →
/// `new_construct_turn0` → framing chunks).
fn intimidate_order(seed: &str, p1: &str, p2: &str) -> Vec<String> {
    let dex = dex();
    let opts = bridge_opts("gen3customgame", seed.to_string(), p1, p2);
    let sess = BridgeSession::new_construct_turn0(&opts, &dex).expect("turn-0 bridge session");
    sess.chunks()
        .side_chunks(0)
        .flat_map(|c| c.lines.iter())
        .filter(|l| l.contains("|Intimidate|boost"))
        .map(|l| l.split('|').nth(2).unwrap()[..3].to_string())
        .collect()
}

/// The tie draw decides which lead's Intimidate is emitted FIRST — the port must emit the
/// order its OWN construction queue resolved, not a side-order default.
///
/// WRONG (pre-fix): `emit_switchin_ability_lines` re-derived "faster first, tie = side
/// order", so BOTH seeds emitted `p1a` first — the `[1,2,3,4]` case below FAILS against the
/// pre-fix code with `["p1a", "p2a"]`.
#[test]
fn turn0_speed_tie_mirror_emits_the_intimidate_block_in_the_drawn_order() {
    let p2_first = intimidate_order("[1,2,3,4]", MASQ_INTIM_M, MASQ_INTIM_M);
    let p1_first = intimidate_order("[3,2,3,4]", MASQ_INTIM_M, MASQ_INTIM_M);
    // Non-vacuity: the mechanic must actually fire (2 Intimidate lines) on BOTH seeds.
    assert_eq!(p2_first.len(), 2, "seed [1,2,3,4] must emit BOTH mirror Intimidates");
    assert_eq!(p1_first.len(), 2, "seed [3,2,3,4] must emit BOTH mirror Intimidates");

    assert_eq!(p2_first, vec!["p2a", "p1a"], "seed [1,2,3,4]: the sim fires p2's runSwitch first");
    assert_eq!(p1_first, vec!["p1a", "p2a"], "seed [3,2,3,4]: the sim fires p1's runSwitch first");
    assert_ne!(p2_first, p1_first, "the two seeds MUST differ — otherwise the draw is being ignored");
}

/// The control: at DISTINCT speeds no tie draw happens and the order is pure Speed — the
/// FASTER lead's Intimidate first, even when that is p2. Guards against "fixing" the tie
/// case by simply inverting the side order.
#[test]
fn turn0_distinct_speed_emits_the_faster_lead_intimidate_first() {
    let order = intimidate_order("[1,2,3,4]", MASQ_INTIM_M, TAUROS_INTIM_FAST_M);
    assert_eq!(order.len(), 2, "both leads must emit an Intimidate line");
    assert_eq!(order, vec!["p2a", "p1a"], "Tauros (256) outspeeds Masquerain (156) → p2 first");
}

const KYOGRE_DRIZZLE: &str = "Kyogre|||Drizzle|splash|Serious||N||||";
const GROUDON_DROUGHT: &str = "Groudon|||Drought|splash|Serious||N||||";

/// The EMISSION↔STATE consistency pin at a MIXED weather-setter tie (Kyogre Drizzle vs
/// Groudon Drought, both spe 216). The LAST setter to fire wins the field, so the drawn
/// order is visible in BOTH the `-weather` line order AND `field.weather` — they must
/// AGREE. Sim ground truth (`harness/probe_r34_mirror_order_groundtruth.js`):
///   seed [1,2,3,4] → Drought line, then Drizzle line, weather = Rain
///   seed [3,2,3,4] → Drizzle line, then Drought line, weather = Sun
///
/// WRONG (pre-fix): the reconstruction ordered a tie by SIDE, so seed [1,2,3,4] emitted
/// Drizzle-then-Drought while the board said Rain — a stream whose protocol and state
/// contradicted each other.
#[test]
fn turn0_weather_setter_tie_emits_the_lines_in_the_order_the_board_agrees_with() {
    let dex = dex();
    for (seed, want_lines, want_weather) in [
        ("[1,2,3,4]", ["Drought", "Drizzle"], pokesim::state::Weather::Rain),
        ("[3,2,3,4]", ["Drizzle", "Drought"], pokesim::state::Weather::Sun),
    ] {
        let opts = bridge_opts("gen3customgame", seed.to_string(), KYOGRE_DRIZZLE, GROUDON_DROUGHT);
        let sess = BridgeSession::new_construct_turn0(&opts, &dex).expect("turn-0 bridge session");
        let got: Vec<String> = sess
            .chunks()
            .side_chunks(0)
            .flat_map(|c| c.lines.iter())
            .filter(|l| l.starts_with("|-weather|"))
            .cloned()
            .collect();
        // Non-vacuity: BOTH setters must have emitted (a single line means the mechanic
        // collapsed and the ordering is untested).
        assert_eq!(got.len(), 2, "seed {seed}: expected 2 `-weather` lines, got {got:?}");
        for (line, want) in got.iter().zip(want_lines) {
            assert!(line.contains(want), "seed {seed}: expected `{want}` in `{line}`");
        }
        // …and the BOARD the same construction produced must agree with the last line.
        let b = construct_seeded(seed, KYOGRE_DRIZZLE, GROUDON_DROUGHT);
        assert_eq!(weather(&b), Some(want_weather), "seed {seed}: last setter wins the field");
    }
}
