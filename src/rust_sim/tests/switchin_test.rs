//! Switch-in-event tests — the deferred `>start` half (`runEvent`/`singleEvent`
//! dispatch + the lead switch-in abilities):
//!
//!   - `switchin_golden_matches_showdown` — differential vs the REAL Showdown
//!     sim's POST-switch-in EVENT outputs (golden from
//!     `harness/gen_switchin_golden.js`): `Battle::start_with_switchins` from the
//!     SAME teams + seed reproduces every scenario's `field.weather` (Sand Stream
//!     / Drizzle / Drought) AND each active lead's Atk boost stage (the
//!     Intimidate −1 drop). Covers the ASYMMETRIC direction (only one foe is an
//!     Intimidater), the single-setter weathers, and the ORDER-dependent
//!     double-weather case (slower lead fires last ⇒ its weather wins).
//!   - `switchin_dispatch_is_draw_free` — the PRNG-consumption gate: OUR switch-in
//!     dispatch draws ZERO (the seed is unchanged after both distinct-speed AND
//!     mirror-tie leads). This is NOT prng-state parity with the sim — the real
//!     `>start` window DOES draw (gender sample, Quick Claw, queue-tie splice), but
//!     those live in the construction / queue / turn-loop phases this bounded step
//!     doesn't build, so they are deliberately omitted. The reusable speed-tie
//!     shuffle is validated directly in `event.rs`'s own unit tests.
//!   - `switchin_handlers_smoke` — authoritative hand-checks independent of the
//!     golden: Intimidate drops Atk by −1 (the ≥−6 clamp is present but
//!     unreachable from one switch-in), Sand Stream/Drizzle/Drought set the right
//!     `Weather` with the permanent (`weather_turns == 0`) gen-3 duration, a double
//!     Sand Stream hits the same-weather no-op guard, a non-weather/non-Intimidate
//!     lead leaves the board clean, and construction (`Battle::start`) still
//!     produces the pristine pre-event board.
//!
//! Mirrors the structure of `tests/state_test.rs`.

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::state::Weather;

fn dex() -> Dex {
    Dex::for_gen(3)
}

// ---------------------------------------------------------------------------
// Golden parsing
// ---------------------------------------------------------------------------

/// One scenario's parsed golden row.
struct GoldenScen {
    name: String,
    teams: [String; 2],
    seed: String,
    weather: Option<Weather>,
    /// Per-side active-lead Atk boost stage.
    boost_atk: [i8; 2],
}

fn side_index(tag: &str) -> usize {
    match tag {
        "p1" => 0,
        "p2" => 1,
        other => panic!("bad side tag {other:?}"),
    }
}

fn parse_weather(s: &str) -> Option<Weather> {
    match s {
        "none" => None,
        "sandstorm" => Some(Weather::Sand),
        "raindance" => Some(Weather::Rain),
        "sunnyday" => Some(Weather::Sun),
        other => panic!("unknown weather {other:?}"),
    }
}

fn parse_golden() -> Vec<GoldenScen> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/switchin_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing switchin golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_switchin_golden.js")
    });

    use std::collections::BTreeMap;
    // Accumulate by scenario name (preserving first-seen order).
    let mut order: Vec<String> = Vec::new();
    let mut teams: BTreeMap<String, [Option<String>; 2]> = BTreeMap::new();
    let mut seeds: BTreeMap<String, String> = BTreeMap::new();
    let mut weather: BTreeMap<String, Option<Weather>> = BTreeMap::new();
    let mut boosts: BTreeMap<String, [Option<i8>; 2]> = BTreeMap::new();

    for (i, line) in data.lines().enumerate() {
        let ln = i + 1;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "SCEN" => {
                assert_eq!(f.len(), 2, "SCEN needs <name> (line {ln})");
                let name = f[1].to_string();
                order.push(name.clone());
                teams.entry(name.clone()).or_insert([None, None]);
                boosts.entry(name).or_insert([None, None]);
            }
            "TEAM" => {
                assert_eq!(f.len(), 4, "TEAM needs <scen> <side> <packed> (line {ln})");
                teams.entry(f[1].to_string()).or_insert([None, None])[side_index(f[2])] =
                    Some(f[3].to_string());
            }
            "SEED" => {
                assert_eq!(f.len(), 3, "SEED needs <scen> <m,n,o,p> (line {ln})");
                seeds.insert(f[1].to_string(), f[2].to_string());
            }
            "WEATHER" => {
                assert_eq!(f.len(), 4, "WEATHER needs <scen> <id> <dur> (line {ln})");
                // duration column is informational (we assert weather_turns==0 in Rust).
                weather.insert(f[1].to_string(), parse_weather(f[2]));
            }
            "BOOST" => {
                assert_eq!(f.len(), 4, "BOOST needs <scen> <side> <stage> (line {ln})");
                let stage: i8 = f[3].parse().unwrap_or_else(|e| panic!("bad boost (line {ln}): {e}"));
                boosts.entry(f[1].to_string()).or_insert([None, None])[side_index(f[2])] = Some(stage);
            }
            other => panic!("unknown record {other:?} (line {ln})"),
        }
    }

    let mut out = Vec::new();
    for name in &order {
        let t = teams.get(name).expect("scenario teams");
        let b = boosts.get(name).expect("scenario boosts");
        out.push(GoldenScen {
            name: name.clone(),
            teams: [
                t[0].clone().unwrap_or_else(|| panic!("{name}: missing TEAM p1")),
                t[1].clone().unwrap_or_else(|| panic!("{name}: missing TEAM p2")),
            ],
            seed: seeds.get(name).unwrap_or_else(|| panic!("{name}: missing SEED")).clone(),
            weather: *weather.get(name).unwrap_or_else(|| panic!("{name}: missing WEATHER")),
            boost_atk: [
                b[0].unwrap_or_else(|| panic!("{name}: missing BOOST p1")),
                b[1].unwrap_or_else(|| panic!("{name}: missing BOOST p2")),
            ],
        });
    }
    out
}

fn opts_from(teams: &[String; 2], seed: &str) -> BattleOptions {
    BattleOptions {
        format_id: "gen3ou".to_string(),
        seed: Some(format!("[{seed}]")), // the `>start` bracketed-array form
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(teams[0].clone()) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(teams[1].clone()) },
    }
}

#[test]
fn switchin_golden_matches_showdown() {
    let d = dex();
    let scenarios = parse_golden();
    assert!(scenarios.len() >= 5, "expected >=5 golden scenarios, got {}", scenarios.len());

    let mut checked = 0usize;
    for sc in &scenarios {
        let opts = opts_from(&sc.teams, &sc.seed);
        let battle = Battle::start_with_switchins(&opts, &d)
            .unwrap_or_else(|e| panic!("scenario {}: start_with_switchins failed: {e}", sc.name));
        let state = battle.state().expect("battle constructed");

        // Weather: the switch-in ability set it (or left it clear).
        assert_eq!(
            state.field.weather, sc.weather,
            "scenario {}: weather mismatch (got {:?} exp {:?})",
            sc.name, state.field.weather, sc.weather
        );
        if state.field.weather.is_some() {
            // gen-3 ability-set weather is PERMANENT (duration 0 sentinel).
            assert_eq!(
                state.field.weather_turns, 0,
                "scenario {}: ability weather must be permanent (weather_turns==0)",
                sc.name
            );
        }

        // Per-side active-lead Atk boost (the Intimidate −1 drop or 0).
        for side in 0..2 {
            let active = state.side(side).active();
            assert_eq!(
                active.boosts[0], sc.boost_atk[side],
                "scenario {}: side {} active {} Atk-boost mismatch (got {} exp {})",
                sc.name, side, active.species_id, active.boosts[0], sc.boost_atk[side]
            );
        }
        checked += 1;
    }
    eprintln!("switchin golden: {checked} scenarios matched (post-switch-in events)");
}

// ---------------------------------------------------------------------------
// PRNG-consumption gate: the switch-in dispatch draws ONLY on a speed tie.
// ---------------------------------------------------------------------------

/// OUR switch-in dispatch consumes NO PRNG — the construction-time
/// (`Battle::start`) seed is pristine `[1,2,3,4]` and stays so after the
/// switch-in events fire, for BOTH distinct-speed leads (no tie) AND a mirror
/// (raw-speed tie, whose queue-tie-break draw is deferred to the turn-loop step).
/// This is NOT prng-state parity with the sim: the real `>start` window DOES draw
/// (per-mon gender `sample`, the gen-3 Quick Claw `randomChance(1,5)`, the
/// queue-tie splice) — but all in phases this bounded step does not build, so they
/// are deliberately omitted (the Rust prng stays put while the sim's advances).
/// The reusable speed-tie shuffle is validated directly in `event.rs`'s unit tests.
#[test]
fn switchin_dispatch_is_draw_free() {
    let d = dex();

    let tauros = "Tauros||leftovers|intimidate|bodyslam,earthquake,doubleedge,hiddenpowerghost|Jolly|,252,,,4,252|||||";
    let ttar = "Tyranitar||leftovers|sandstream|rockslide,earthquake,crunch,dragondance|Adamant|252,252,,,4,|||||";

    let opts = |p1: &str, p2: &str| BattleOptions {
        format_id: "gen3ou".to_string(),
        seed: Some("[1,2,3,4]".to_string()),
        p1: PlayerOptions { name: "A".to_string(), team: PackedTeam(p1.to_string()) },
        p2: PlayerOptions { name: "B".to_string(), team: PackedTeam(p2.to_string()) },
    };

    // Construction alone must not touch the PRNG (the carry-forward invariant).
    let constructed = Battle::start(&opts(tauros, tauros), &d).expect("start");
    assert_eq!(
        constructed.state().unwrap().prng_seed(),
        "1,2,3,4",
        "construction must not consume the PRNG"
    );

    // A) Mirror (raw-speed TIE) ⇒ still draw-free here; both Intimidaters drop
    //    the opposing lead to −1 (order-independent end state).
    let tie = Battle::start_with_switchins(&opts(tauros, tauros), &d).expect("start");
    let ts = tie.state().unwrap();
    assert_eq!(ts.prng_seed(), "1,2,3,4", "switch-in dispatch draws nothing (tie-break deferred)");
    assert_eq!(ts.side(0).active().boosts[0], -1, "tie: p1 atk -1");
    assert_eq!(ts.side(1).active().boosts[0], -1, "tie: p2 atk -1");

    // B) Distinct speeds ⇒ also draw-free; Tauros (Intimidate, spe 350) drops
    //    Tyranitar (Sand Stream, spe 158) which sets sand.
    let distinct = Battle::start_with_switchins(&opts(tauros, ttar), &d).expect("start");
    let ds = distinct.state().unwrap();
    assert_eq!(ds.prng_seed(), "1,2,3,4", "distinct-speed switch-in draws nothing");
    assert_eq!(ds.side(0).active().boosts[0], 0, "distinct: Tauros not Intimidated");
    assert_eq!(ds.side(1).active().boosts[0], -1, "distinct: Tyranitar dropped to -1");
    assert_eq!(ds.field.weather, Some(Weather::Sand), "distinct: sandstorm set");
}

// ---------------------------------------------------------------------------
// Hand-checks, golden-independent.
// ---------------------------------------------------------------------------

#[test]
fn switchin_handlers_smoke() {
    let d = dex();

    // Helper to build a 1v1 and run the switch-ins.
    let build = |p1: &str, p2: &str| -> Battle {
        let opts = BattleOptions {
            format_id: "gen3ou".to_string(),
            seed: Some("[1,2,3,4]".to_string()),
            p1: PlayerOptions { name: "A".to_string(), team: PackedTeam(p1.to_string()) },
            p2: PlayerOptions { name: "B".to_string(), team: PackedTeam(p2.to_string()) },
        };
        Battle::start_with_switchins(&opts, &d).expect("start_with_switchins")
    };

    let ttar = "Tyranitar||leftovers|sandstream|rockslide,earthquake,crunch,dragondance|Adamant|252,252,,,4,|||||";
    let salamence = "Salamence||choiceband|intimidate|earthquake,rockslide,hiddenpowerflying,brickbreak|Adamant|,252,,,4,252|||||";
    let blissey = "Blissey||leftovers|naturalcure|softboiled,seismictoss,toxic,aromatherapy|Calm|252,,,,252,4|||||";
    let kyogre = "Kyogre||leftovers|drizzle|surf,icebeam,thunder,calmmind|Modest|4,,,252,,252|||||";
    let groudon = "Groudon||leftovers|drought|earthquake,rockslide,swordsdance,hiddenpowerbug|Adamant|4,252,,,,252|||||";

    // Sand Stream sets permanent sand; Intimidate (faster Salamence) drops Tyranitar.
    let b = build(ttar, salamence);
    let s = b.state().unwrap();
    assert_eq!(s.field.weather, Some(Weather::Sand));
    assert_eq!(s.field.weather_turns, 0, "ability weather is permanent");
    assert_eq!(s.side(0).active().boosts[0], -1, "Tyranitar dropped by Salamence Intimidate");
    assert_eq!(s.side(1).active().boosts[0], 0, "Salamence not dropped (Tyranitar has no Intimidate)");

    // Drizzle ⇒ rain; Drought ⇒ sun (single setter vs neutral Blissey).
    assert_eq!(build(kyogre, blissey).state().unwrap().field.weather, Some(Weather::Rain));
    assert_eq!(build(groudon, blissey).state().unwrap().field.weather, Some(Weather::Sun));

    // Double Sand Stream: the 2nd setter hits the same-weather no-op guard
    // (field.ts:50-52 — ability source + duration 0 ⇒ setWeather returns false),
    // so the board stays permanent Sand (this exercises the guard branch).
    let mirror = build(ttar, ttar);
    let mf = mirror.state().unwrap();
    assert_eq!(mf.field.weather, Some(Weather::Sand), "double Sand Stream stays Sand");
    assert_eq!(mf.field.weather_turns, 0, "still permanent after the no-op re-set");

    // A clean board: neither lead has a switch-in ability ⇒ no weather, no boosts.
    let clean = build(blissey, blissey);
    let cs = clean.state().unwrap();
    assert!(cs.field.weather.is_none(), "no weather without a setter");
    assert_eq!(cs.side(0).active().boosts[0], 0);
    assert_eq!(cs.side(1).active().boosts[0], 0);

    // Construction-only (`Battle::start`) still yields the PRISTINE pre-event
    // board even for Sand Stream / Intimidate leads — the construction golden's
    // invariant is preserved.
    let opts = BattleOptions {
        format_id: "gen3ou".to_string(),
        seed: Some("[1,2,3,4]".to_string()),
        p1: PlayerOptions { name: "A".to_string(), team: PackedTeam(ttar.to_string()) },
        p2: PlayerOptions { name: "B".to_string(), team: PackedTeam(salamence.to_string()) },
    };
    let pre = Battle::start(&opts, &d).expect("start (construction only)");
    let ps = pre.state().unwrap();
    assert!(ps.field.weather.is_none(), "construction: Sand Stream is an EVENT, no weather built");
    assert_eq!(ps.side(0).active().boosts, [0i8; 7], "construction: no Intimidate boost");
    assert_eq!(ps.side(1).active().boosts, [0i8; 7], "construction: boosts pristine");
}
