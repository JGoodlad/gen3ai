//! FORECAST (Castform forme swap) test (`gen3_forecast_v1`, ROUND 35) — the per-seed
//! PER-DECISION STATE+HP+STATUS+BOOSTS+**FORME(species)**+**WEATHER(id+turns)**+SEED+winner
//! differential proving Forecast matches Showdown EXACTLY, to GAME-END.
//!
//! Forecast swaps Castform's forme + TYPE to follow the EFFECTIVE weather, DRAW-FREE, at every
//! `eachEvent('WeatherChange')` site + the entrant's own `onStart` + the start window, reverting
//! silently at `clearVolatile`. Two things make this gate bite:
//!   * **the SPECIES column IS the forme** — the port's `DecisionRecord.active_species` reports the
//!     LIVE `species_id`, so `castform` / `castformrainy` / `castformsunny` / `castformsnowy` are
//!     asserted per decision. A forme that fires late, early, on the wrong weather, or never
//!     reverts diverges here.
//!   * **the forme carries the TYPE**, and the type is load-bearing for the HP columns: a
//!     `castformsnowy` is ICE → hail-chip IMMUNE, so a missed/extra Snowy forme shows up as an HP
//!     divergence in `fc_hail_snowy` even if the species column were somehow satisfied.
//! Forecast is DRAW-FREE, so the per-decision SEED must match the sim bit-for-bit — a spurious
//! draw at any of the wiring sites desyncs the LCG here.
//!
//! Scenarios (`harness/gen_forecast_golden.js`, 7 × 40 seeds, all decisive P1 wins):
//!   fc_rain_cycle   — the weather-set MOVE site: forme Rainy, hold, REVERT at the 5-turn expiry.
//!   fc_hail_snowy   — the same on HAIL → Snowy (ICE, hail-chip immune while the foe chips).
//!   fc_sand_base    — SAND is the DEFAULT arm: stays BASE `castform` (and takes the chip). NO
//!                     formechange fires — the case a "any weather → a forme" model gets wrong.
//!   fc_ability_in   — a DRIZZLE Politoed switches in (the `run_switch` WeatherChange site).
//!   fc_pivot        — a FORMED Castform pivots out (silent revert) + back in (re-forme onStart).
//!   fc_suppressed   — a CLOUD NINE holder suppresses the rain, then FAINTS → the negater's onEnd
//!                     WeatherChange with the dying holder EXCLUDED → the Castform formes.
//!   fc_start_window — a DROUGHT lead formes the Castform BEFORE decision 0.
//! The revert-verified FC1-FC8 pins in `regression_test.rs` are the deterministic per-site
//! discriminators; this sweep is the breadth complement.

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::state::{Status, Weather, BOOST_LEN};
use pokesim::turn::{Choice, RequestKind, ScriptDecision};
use std::collections::BTreeMap;

fn dex() -> Dex {
    Dex::for_gen(3)
}

#[derive(Debug, Clone, Default)]
struct ScenMeta {
    teams: [Option<String>; 2],
}

#[derive(Debug, Clone)]
struct RunCase {
    scen: String,
    init_seed: String,
    decisions: Vec<DecExpect>,
    ended: bool,
    winner: WinTok,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WinTok {
    P1,
    P2,
    Tie,
    None,
}

#[derive(Debug, Clone)]
struct DecExpect {
    request: ReqTok,
    force: [bool; 2],
    choice: [Option<Choice>; 2],
    seed_after: String,
    p1: SideExpect,
    p2: SideExpect,
    /// The FIELD weather id ('-' = clear) + its remaining duration (0 = permanent/none).
    weather: Option<Weather>,
    weather_turns: u8,
    first_mover: String,
    /// The sim-side marker: a `|-formechange|` fired this decision (coverage).
    forme: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReqTok {
    Move,
    Switch,
}

#[derive(Debug, Clone)]
struct SideExpect {
    /// The LIVE species id — for a Forecast Castform this IS the forme.
    species: String,
    hp: u16,
    maxhp: u16,
    fainted: bool,
    status: Option<Status>,
    left: usize,
    boosts: [i8; BOOST_LEN],
}

fn parse_choice(tok: &str) -> Option<Choice> {
    if tok == "-" {
        return None;
    }
    let (kind, num) = tok.split_at(1);
    let n: usize = num.parse().unwrap_or_else(|e| panic!("bad choice token {tok:?}: {e}"));
    match kind {
        "m" => Some(Choice::Move(n)),
        "s" => Some(Choice::Switch(n)),
        other => panic!("bad choice kind {other:?} in {tok:?}"),
    }
}

fn parse_status(tok: &str) -> Option<Status> {
    match tok {
        "-" | "fnt" => None,
        "brn" => Some(Status::Burn),
        "par" => Some(Status::Paralysis),
        "slp" => Some(Status::Sleep(0)),
        "frz" => Some(Status::Freeze),
        "psn" => Some(Status::Poison),
        "tox" => Some(Status::Toxic(0)),
        other => panic!("unknown status token {other:?}"),
    }
}

fn parse_boosts(tok: &str) -> [i8; BOOST_LEN] {
    let mut out = [0i8; BOOST_LEN];
    let parts: Vec<&str> = tok.split(',').collect();
    assert_eq!(parts.len(), BOOST_LEN, "boost column needs {BOOST_LEN} stages, got {tok:?}");
    for (i, p) in parts.iter().enumerate() {
        out[i] = p.parse().unwrap_or_else(|e| panic!("bad boost {p:?}: {e}"));
    }
    out
}

/// The sim's field-weather id → the port's [`Weather`]. `-` / `` = clear.
fn parse_weather(tok: &str) -> Option<Weather> {
    match tok {
        "-" | "" | "none" => None,
        "raindance" => Some(Weather::Rain),
        "sunnyday" => Some(Weather::Sun),
        "sandstorm" => Some(Weather::Sand),
        "hail" => Some(Weather::Hail),
        other => panic!("unknown weather token {other:?}"),
    }
}

fn status_variant_eq(a: Option<Status>, b: Option<Status>) -> bool {
    use Status::*;
    matches!(
        (a, b),
        (None, None)
            | (Some(Burn), Some(Burn))
            | (Some(Paralysis), Some(Paralysis))
            | (Some(Sleep(_)), Some(Sleep(_)))
            | (Some(Freeze), Some(Freeze))
            | (Some(Poison), Some(Poison))
            | (Some(Toxic(_)), Some(Toxic(_)))
    )
}

fn parse_golden() -> (BTreeMap<String, ScenMeta>, Vec<RunCase>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/forecast_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing forecast golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_forecast_golden.js")
    });

    let mut meta: BTreeMap<String, ScenMeta> = BTreeMap::new();
    let mut cases: Vec<RunCase> = Vec::new();
    let mut cur: Option<RunCase> = None;

    let flush = |cur: &mut Option<RunCase>, cases: &mut Vec<RunCase>| {
        if let Some(c) = cur.take() {
            cases.push(c);
        }
    };

    for (i, line) in data.lines().enumerate() {
        let ln = i + 1;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "SCEN" => {
                meta.entry(f[1].to_string()).or_default();
            }
            "TEAM" => {
                assert_eq!(f.len(), 4, "TEAM needs 4 fields (line {ln})");
                let side = if f[2] == "p1" { 0 } else { 1 };
                meta.entry(f[1].to_string()).or_default().teams[side] = Some(f[3].to_string());
            }
            "INIT" => {
                assert_eq!(f.len(), 4, "INIT needs 4 fields (line {ln})");
                flush(&mut cur, &mut cases);
                cur = Some(RunCase {
                    scen: f[1].to_string(),
                    init_seed: f[2].to_string(),
                    decisions: Vec::new(),
                    ended: false,
                    winner: WinTok::None,
                });
            }
            "DEC" => {
                // DEC <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter>
                //   p1(species hp maxhp fnt status left)[9..15) p1boosts[15]
                //   p2(...)[16..22) p2boosts[22]
                //   weather[23] weatherTurns[24] first[25] forme[26]
                assert_eq!(f.len(), 27, "DEC needs 27 fields (line {ln}), got {}", f.len());
                let req = match f[3] {
                    "move" => ReqTok::Move,
                    "switch" => ReqTok::Switch,
                    other => panic!("bad request {other:?} (line {ln})"),
                };
                let force = [f[4] == "1", f[5] == "1"];
                let choice = [parse_choice(f[6]), parse_choice(f[7])];
                let seed_after = f[8].to_string();
                let g = |i: usize| {
                    f[i].parse::<u16>().unwrap_or_else(|e| panic!("bad num f[{i}] (line {ln}): {e}"))
                };
                let p1 = SideExpect {
                    species: f[9].to_string(),
                    hp: g(10),
                    maxhp: g(11),
                    fainted: f[12] == "1",
                    status: parse_status(f[13]),
                    left: g(14) as usize,
                    boosts: parse_boosts(f[15]),
                };
                let p2 = SideExpect {
                    species: f[16].to_string(),
                    hp: g(17),
                    maxhp: g(18),
                    fainted: f[19] == "1",
                    status: parse_status(f[20]),
                    left: g(21) as usize,
                    boosts: parse_boosts(f[22]),
                };
                let weather = parse_weather(f[23]);
                let weather_turns = f[24].parse::<u8>().unwrap_or_else(|e| panic!("bad weatherTurns (line {ln}): {e}"));
                let first_mover = f[25].to_string();
                let forme = f[26] == "1";
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect {
                    request: req, force, choice, seed_after, p1, p2, weather, weather_turns, first_mover, forme,
                });
            }
            "END" => {
                assert_eq!(f.len(), 5, "END needs 5 fields (line {ln})");
                let c = cur.as_mut().unwrap_or_else(|| panic!("END before INIT (line {ln})"));
                c.ended = f[3] == "1";
                c.winner = match f[4] {
                    "p1" => WinTok::P1,
                    "p2" => WinTok::P2,
                    "tie" => WinTok::Tie,
                    "none" => WinTok::None,
                    other => panic!("bad winner {other:?} (line {ln})"),
                };
            }
            other => panic!("unknown record {other:?} (line {ln})"),
        }
    }
    flush(&mut cur, &mut cases);
    (meta, cases)
}

fn opts_for(meta: &ScenMeta, init_seed: &str) -> BattleOptions {
    let t = &meta.teams;
    BattleOptions {
        format_id: "gen3customgame".to_string(),
        seed: Some(init_seed.to_string()),
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(t[0].clone().expect("p1 team")) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(t[1].clone().expect("p2 team")) },
    }
}

fn species_id(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_lowercase()).collect()
}

fn req_eq(rust: &RequestKind, golden: ReqTok, force: [bool; 2]) -> bool {
    match (rust, golden) {
        (RequestKind::Move, ReqTok::Move) => true,
        (RequestKind::ForceSwitch { force: rf }, ReqTok::Switch) => *rf == force,
        _ => false,
    }
}

#[test]
fn forecast_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(meta.len() >= 7, "expected >=7 scenarios, got {}", meta.len());
    assert!(cases.len() >= 200, "expected the per-seed corpus (>=200 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut forme_assertions = 0usize;
    let mut weather_assertions = 0usize;
    let mut forme_rows = 0usize;
    let mut forme_per_scen: BTreeMap<String, usize> = BTreeMap::new();
    // Per-scenario: did the port's OWN species column ever reach the expected forme / revert?
    let mut reached: BTreeMap<String, usize> = BTreeMap::new();
    let mut reverted: BTreeMap<String, usize> = BTreeMap::new();
    let mut start_window_formed = 0usize;
    let mut win_runs = 0usize;
    let mut tie_runs = 0usize;

    for case in &cases {
        let m = meta.get(&case.scen).unwrap_or_else(|| panic!("no meta for {}", case.scen));
        assert!(!case.decisions.is_empty(), "[{}] empty run", case.scen);

        let opts = opts_for(m, &case.init_seed);
        let mut battle = Battle::start_with_switchins(&opts, &d)
            .unwrap_or_else(|e| panic!("[{}] start failed: {e}", case.scen));
        assert_eq!(
            battle.state().unwrap().prng_seed(),
            case.init_seed,
            "[{}] init prng seed must equal the sim's (the switch-ins + the START-window forme are draw-free)",
            case.scen
        );

        let script: Vec<ScriptDecision> = case
            .decisions
            .iter()
            .map(|dec| ScriptDecision { p1: dec.choice[0], p2: dec.choice[1] })
            .collect();
        let outcome = battle.state_mut().unwrap().run_full_battle(&script, &d);

        assert_eq!(
            outcome.decisions.len(),
            case.decisions.len(),
            "[{}] decision count mismatch (init_seed {}): rust {} vs golden {}",
            case.scen, case.init_seed, outcome.decisions.len(), case.decisions.len()
        );

        // The port's OWN forme timeline (side 0 is always the Castform side in this sweep).
        let port_species: Vec<String> =
            outcome.decisions.iter().map(|r| species_id(&r.active_species[0])).collect();

        for (di, (rec, exp)) in outcome.decisions.iter().zip(case.decisions.iter()).enumerate() {
            assert!(
                req_eq(&rec.request, exp.request, exp.force),
                "[{}] decision {} request mismatch (init_seed {}): got {:?} exp {:?} force {:?}",
                case.scen, di, case.init_seed, rec.request, exp.request, exp.force
            );

            for (idx, (snap, e, sp)) in [
                (0usize, (&rec.active[0], &exp.p1, &rec.active_species[0])),
                (1usize, (&rec.active[1], &exp.p2, &rec.active_species[1])),
            ] {
                // --- THE FORECAST GATE (1): the LIVE species IS the forme. A forme that fires on
                //     the wrong weather (sand!), fires late/early, or never reverts diverges here. ---
                assert_eq!(
                    species_id(sp),
                    species_id(&e.species),
                    "[{}] dec {} side {} FORME/species mismatch (init_seed {}): got {:?} exp {:?}\n  \
                     the live species IS the Castform forme — check forecast_weather_change's \
                     weather→forme map (SAND and NONE both map to the BASE `castform`), the \
                     WeatherChange wiring sites, and the clearVolatile revert.",
                    case.scen, di, idx, case.init_seed, species_id(sp), species_id(&e.species)
                );
                forme_assertions += 1;
                // --- THE FORECAST GATE (2): the forme carries the TYPE, and the type drives the
                //     weather chip — a missed Snowy forme shows up as an HP divergence under hail. ---
                assert_eq!(
                    snap.hp, e.hp,
                    "[{}] dec {} side {} HP mismatch (init_seed {}): got {} exp {}\n  \
                     under hail this is the FORME-TYPE proof: a `castformsnowy` is ICE → chip-IMMUNE.",
                    case.scen, di, idx, case.init_seed, snap.hp, e.hp
                );
                assert_eq!(snap.maxhp, e.maxhp, "[{}] dec {} side {} maxhp", case.scen, di, idx);
                assert_eq!(
                    snap.fainted, e.fainted,
                    "[{}] dec {} side {} fainted mismatch (init_seed {})",
                    case.scen, di, idx, case.init_seed
                );
                if !e.fainted {
                    assert!(
                        status_variant_eq(snap.status, e.status),
                        "[{}] dec {} side {} STATUS mismatch (init_seed {}): got {:?} exp {:?}",
                        case.scen, di, idx, case.init_seed, snap.status, e.status
                    );
                    assert_eq!(
                        &snap.boosts[..], &e.boosts[..],
                        "[{}] dec {} side {} BOOST mismatch (init_seed {})",
                        case.scen, di, idx, case.init_seed
                    );
                }
            }
            assert_eq!(rec.pokemon_left[0], exp.p1.left, "[{}] dec {} p1 left", case.scen, di);
            assert_eq!(rec.pokemon_left[1], exp.p2.left, "[{}] dec {} p2 left", case.scen, di);

            // --- THE FORECAST GATE (3): the WEATHER the formes key off — id AND the timed
            //     countdown, so a mis-timed expiry (which is ALSO the forme-revert trigger, and
            //     the site of the ROUND-35 unconditional-WeatherChange draw fix) diverges. ---
            assert_eq!(
                rec.weather, exp.weather,
                "[{}] dec {} WEATHER mismatch (init_seed {}): got {:?} exp {:?}",
                case.scen, di, case.init_seed, rec.weather, exp.weather
            );
            assert_eq!(
                rec.weather_turns, exp.weather_turns,
                "[{}] dec {} WEATHER-TURNS mismatch (init_seed {}): got {} exp {}\n  \
                 the 5-turn timed countdown drives the expiry — the forme-revert trigger.",
                case.scen, di, case.init_seed, rec.weather_turns, exp.weather_turns
            );
            weather_assertions += 1;

            if exp.request == ReqTok::Move {
                let sim_first: Option<usize> = match exp.first_mover.as_str() {
                    "p1" => Some(0),
                    "p2" => Some(1),
                    _ => None,
                };
                if sim_first.is_some() {
                    assert_eq!(
                        rec.first_mover, sim_first,
                        "[{}] dec {} first-mover mismatch (init_seed {})",
                        case.scen, di, case.init_seed
                    );
                }
            }

            // --- PER-DECISION SEED PARITY: Forecast is DRAW-FREE — a spurious draw at any wiring
            //     site (or a MISSING one, like the pre-round-35 gated expiry WeatherChange) desyncs
            //     the LCG here. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 a Forecast wiring site consumed or skipped a PRNG draw it must not (the forme is \
                 draw-free; the EXPIRY WeatherChange shuffle is UNCONDITIONAL).",
                case.scen, di, case.init_seed, rec.seed_after, exp.seed_after
            );
            seed_assertions += 1;
            dec_assertions += 1;
            if exp.forme {
                forme_rows += 1;
                *forme_per_scen.entry(case.scen.clone()).or_default() += 1;
            }
        }

        // Per-run forme-timeline coverage, read off the PORT's own species column (not the
        // golden's) — so these floors prove the PORT formed/reverted, not just that the sim did.
        let want = match case.scen.as_str() {
            "fc_rain_cycle" | "fc_ability_in" | "fc_pivot" | "fc_suppressed" => Some("castformrainy"),
            "fc_hail_snowy" => Some("castformsnowy"),
            _ => None,
        };
        if let Some(w) = want {
            if let Some(first) = port_species.iter().position(|s| s == w) {
                *reached.entry(case.scen.clone()).or_default() += 1;
                // NOT fc_ability_in: Drizzle sets PERMANENT weather and that Castform never
                // pivots, so it is correctly Rainy for the rest of the battle.
                if case.scen != "fc_ability_in"
                    && port_species[first + 1..].iter().any(|s| s == "castform")
                {
                    *reverted.entry(case.scen.clone()).or_default() += 1;
                }
            }
        }
        if case.scen == "fc_start_window" && port_species[0] == "castformsunny" {
            start_window_formed += 1;
        }

        assert_eq!(outcome.ended, case.ended, "[{}] ended mismatch (init_seed {})", case.scen, case.init_seed);
        let rust_win = match outcome.winner {
            Some(0) => WinTok::P1,
            Some(1) => WinTok::P2,
            Some(other) => panic!("[{}] bad winner side {other}", case.scen),
            None if outcome.ended => WinTok::Tie,
            None => WinTok::None,
        };
        assert_eq!(
            rust_win, case.winner,
            "[{}] WINNER mismatch (init_seed {}): got {:?} exp {:?}",
            case.scen, case.init_seed, rust_win, case.winner
        );
        match case.winner {
            WinTok::P1 | WinTok::P2 => win_runs += 1,
            WinTok::Tie => tie_runs += 1,
            WinTok::None => {}
        }
    }

    // ── Coverage floors (NON-VACUITY): the sweep must actually exercise each wiring site, and
    //    the SAND scenario must exercise the DEFAULT arm (no forme at all). ──
    for scen in ["fc_rain_cycle", "fc_hail_snowy", "fc_ability_in", "fc_pivot", "fc_suppressed"] {
        let n = forme_per_scen.get(scen).copied().unwrap_or(0);
        assert!(n >= 10, "[{scen}] only {n} formechange rows (<10) — the forme never fired");
        let r = reached.get(scen).copied().unwrap_or(0);
        assert!(r >= 10, "[{scen}] only {r} runs where the PORT reached the expected forme (<10)");
    }
    for scen in ["fc_rain_cycle", "fc_hail_snowy", "fc_pivot", "fc_suppressed"] {
        let r = reverted.get(scen).copied().unwrap_or(0);
        assert!(
            r >= 10,
            "[{scen}] only {r} runs where the PORT REVERTED to base after forming (<10) — \
             a forme-but-never-revert model would pass without this"
        );
    }
    let sand = forme_per_scen.get("fc_sand_base").copied().unwrap_or(0);
    assert_eq!(sand, 0, "[fc_sand_base] SAND is the DEFAULT arm — expected 0 formechange rows, got {sand}");
    assert!(
        start_window_formed >= 10,
        "expected the START-window forme at dec0 (>=10 runs), got {start_window_formed}"
    );
    assert!(seed_assertions >= 1000, "expected the per-decision seed corpus (>=1000), got {seed_assertions}");
    assert!(win_runs >= 200, "expected real game-end WIN runs (>=200), got {win_runs}");

    eprintln!(
        "forecast golden: {} runs over {} scenarios, {dec_assertions} STATE rows, \
         {forme_assertions} forme/species assertions, {weather_assertions} weather assertions, \
         {seed_assertions} seed assertions, {forme_rows} formechange rows, \
         {start_window_formed} start-window formes, {win_runs} wins, {tie_runs} ties",
        cases.len(),
        meta.len()
    );
}
