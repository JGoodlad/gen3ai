//! MOVE-COVERAGE BATCH 2 full-battle test (`gen3_move_coverage_batch2_v1`) — the per-seed
//! PER-DECISION STATE(+HP+STATUS+BOOSTS+WEATHER+SCREENS)+SEED+winner differential that proves
//! the four DRAW-friendly status-move classes match Showdown EXACTLY, to GAME-END:
//!
//!   * **STATUS-CURE** — Refresh (self par/psn/brn), Heal Bell + Aromatherapy (whole-team
//!     major-status cure incl. bench). NEVER-MISS + DRAW-FREE. Proved by the STATUS columns
//!     (active + a bench cure shows in a later switch-in's status). Heal Bell skips a
//!     Soundproof ally.
//!   * **WEATHER-SET** — Rain Dance (Rain) / Sunny Day (Sun): a 5-turn TIMED weather. The
//!     eachEvent('WeatherChange') tie-shuffle draws only on a speed tie; setWeather FAILS
//!     (draw-free) into the SAME weather, OVERWRITES a DIFFERENT one. Proved by the WEATHER
//!     (id+turns) column + the SEED (a wrong shuffle/upkeep draw desyncs).
//!   * **STAT-DROP** — Screech −2 Def / Charm −2 Atk / Metal Sound −2 SpD / … : accuracy draw +
//!     boost() (draw-free, ±6 clamp, Clear Body / Hyper Cutter / Soundproof gated). Proved by
//!     the BOOST columns + the SEED (the accuracy roll must be drawn once).
//!   * **SCREENS** — Light Screen / Reflect (5 turns, halve special / physical). NEVER-MISS +
//!     DRAW-FREE; re-use while up FAILS. Proved by the per-side SCREEN columns + HP (Reflect
//!     halves an incoming physical hit).
//!
//! The cures / weather-set (at distinct speed) / screens are DRAW-FREE; the weather-set at a
//! SPEED TIE draws the eachEvent('WeatherChange') shuffle; the stat-drops draw ONE accuracy
//! roll. So the SEED parity is the draw-model proof (a stray/missing draw desyncs the LCG) and
//! the STATE columns are the effect proof. The golden TAB format extends the batch-1 one:
//! it drops the spikes/leech/item tail and appends WEATHER(id+turns) + per-side SCREEN columns
//! → DEC has 42 fields, plus an INJECT line per scenario (weather/status/benchStatus/hp).

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::json::Json;
use pokesim::state::{Status, Weather};
use pokesim::turn::{Choice, RequestKind, ScriptDecision};
use std::collections::BTreeMap;

fn dex() -> Dex {
    Dex::for_gen(3)
}

#[derive(Debug, Clone, Default)]
struct ScenMeta {
    teams: [Option<String>; 2],
    inject: Vec<Inject>,
}

/// One injected board mutation (a STATE-only set, no PRNG — so seed parity is unaffected).
#[derive(Debug, Clone, Default)]
struct Inject {
    weather: Option<Weather>,
    side: Option<usize>,
    status: Option<Status>,
    /// The status to inject on the side's BENCH mon (team slot 1) — for the Heal Bell /
    /// Aromatherapy whole-team cure scenarios.
    bench_status: Option<Status>,
    hp: Option<u16>,
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
    first_mover: String,
    weather: Option<Weather>,
    weather_turns: u8,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReqTok {
    Move,
    Switch,
}

#[derive(Debug, Clone)]
struct SideExpect {
    species: String,
    hp: u16,
    maxhp: u16,
    fainted: bool,
    status: Option<Status>,
    stage: u8,
    left: usize,
    boosts: [i8; 5],
    confusion: u8,
    light_screen: u8,
    reflect: u8,
}

fn parse_status(tok: &str, stage: u16) -> Option<Status> {
    match tok {
        "-" => None,
        "fnt" => None,
        "brn" => Some(Status::Burn),
        "par" => Some(Status::Paralysis),
        "slp" => Some(Status::Sleep(stage as u8)),
        "frz" => Some(Status::Freeze),
        "psn" => Some(Status::Poison),
        "tox" => Some(Status::Toxic(stage as u8)),
        other => panic!("unknown status token {other:?}"),
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

fn status_stage(s: Option<Status>) -> u8 {
    match s {
        Some(Status::Sleep(n)) => n,
        Some(Status::Toxic(n)) => n,
        _ => 0,
    }
}

fn parse_weather(tok: &str) -> Option<Weather> {
    match tok {
        "-" => None,
        "sandstorm" => Some(Weather::Sand),
        "raindance" => Some(Weather::Rain),
        "sunnyday" => Some(Weather::Sun),
        "hail" => Some(Weather::Hail),
        other => panic!("unknown weather token {other:?}"),
    }
}

fn parse_inject(json: &str, ln: usize) -> Vec<Inject> {
    let parsed = Json::parse(json).unwrap_or_else(|e| panic!("bad INJECT json (line {ln}): {e}"));
    let arr = parsed.as_array().unwrap_or_else(|| panic!("INJECT not an array (line {ln})"));
    let status_of = |s: &str| match s {
        "brn" => Status::Burn,
        "par" => Status::Paralysis,
        "psn" => Status::Poison,
        "tox" => Status::Toxic(0),
        "frz" => Status::Freeze,
        "slp" => Status::Sleep(0),
        other => panic!("unknown INJECT status {other:?} (line {ln})"),
    };
    arr.iter()
        .map(|e| {
            let weather = e.str_at("weather").map(|w| match w {
                "sandstorm" | "sand" => Weather::Sand,
                "raindance" | "rain" => Weather::Rain,
                "sunnyday" | "sun" => Weather::Sun,
                "hail" => Weather::Hail,
                other => panic!("unknown INJECT weather {other:?} (line {ln})"),
            });
            let side = e.get("side").and_then(|s| s.as_f64()).map(|f| f as usize);
            let status = e.str_at("status").map(status_of);
            let bench_status = e.str_at("benchStatus").map(status_of);
            let hp = e.get("hp").and_then(|h| h.as_f64()).map(|f| f as u16);
            Inject { weather, side, status, bench_status, hp }
        })
        .collect()
}

fn parse_golden() -> (BTreeMap<String, ScenMeta>, Vec<RunCase>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/movecoverage_batch2_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "missing batch2 golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_movecoverage_batch2_golden.js"
        )
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
            "INJECT" => {
                assert_eq!(f.len(), 3, "INJECT needs 3 fields (line {ln})");
                meta.entry(f[1].to_string()).or_default().inject = parse_inject(f[2], ln);
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
                //   p1(species hp max fnt status stage left atk def spa spd spe confusion)[9..22)
                //   p2(...)[22..35)  first[35]
                //   weatherId[36] weatherTurns[37]  p1LS[38] p1Refl[39]  p2LS[40] p2Refl[41]
                assert_eq!(f.len(), 42, "DEC needs 42 fields (line {ln}), got {}", f.len());
                let req = match f[3] {
                    "move" => ReqTok::Move,
                    "switch" => ReqTok::Switch,
                    other => panic!("bad request {other:?} (line {ln})"),
                };
                let force = [f[4] == "1", f[5] == "1"];
                let choice = [parse_choice(f[6]), parse_choice(f[7])];
                let seed_after = f[8].to_string();
                let g = |i: usize| f[i].parse::<u16>().unwrap_or_else(|e| panic!("bad num f[{i}] (line {ln}): {e}"));
                let gi = |i: usize| f[i].parse::<i8>().unwrap_or_else(|e| panic!("bad int f[{i}] (line {ln}): {e}"));
                let p1 = SideExpect {
                    species: f[9].to_string(),
                    hp: g(10),
                    maxhp: g(11),
                    fainted: f[12] == "1",
                    status: parse_status(f[13], g(14)),
                    stage: g(14) as u8,
                    left: g(15) as usize,
                    boosts: [gi(16), gi(17), gi(18), gi(19), gi(20)],
                    confusion: g(21) as u8,
                    light_screen: g(38) as u8,
                    reflect: g(39) as u8,
                };
                let p2 = SideExpect {
                    species: f[22].to_string(),
                    hp: g(23),
                    maxhp: g(24),
                    fainted: f[25] == "1",
                    status: parse_status(f[26], g(27)),
                    stage: g(27) as u8,
                    left: g(28) as usize,
                    boosts: [gi(29), gi(30), gi(31), gi(32), gi(33)],
                    confusion: g(34) as u8,
                    light_screen: g(40) as u8,
                    reflect: g(41) as u8,
                };
                let first_mover = f[35].to_string();
                let weather = parse_weather(f[36]);
                let weather_turns = g(37) as u8;
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect {
                    request: req, force, choice, seed_after, p1, p2, first_mover, weather, weather_turns,
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

fn script_from_decisions(case: &RunCase) -> Vec<ScriptDecision> {
    case.decisions
        .iter()
        .map(|dec| ScriptDecision { p1: dec.choice[0], p2: dec.choice[1] })
        .collect()
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

/// Apply the scenario's one-time STATE-only INJECT (weather/status/benchStatus/HP) to the
/// freshly-constructed battle — mirroring the sim harness's post-start inject. NO PRNG. An
/// injected weather is PERMANENT (weather_turns 0), like the sim's Drizzle-style inject.
fn apply_inject(battle: &mut Battle, inject: &[Inject]) {
    let state = battle.state_mut().expect("state");
    for inj in inject {
        if let Some(w) = inj.weather {
            state.field.weather = Some(w);
            state.field.weather_turns = 0;
        }
        if let Some(side) = inj.side {
            let active = state.sides[side].active;
            if let Some(bs) = inj.bench_status {
                // team slot 1 = the bench mon the sim harness statuses.
                if let Some(bench) = state.sides[side].pokemon.get_mut(1) {
                    bench.status = Some(bs);
                }
            }
            let mon = &mut state.sides[side].pokemon[active];
            if let Some(st) = inj.status {
                mon.status = Some(st);
            }
            if let Some(hp) = inj.hp {
                mon.hp = hp;
            }
        }
    }
}

#[test]
fn movecoverage_batch2_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(meta.len() >= 15, "expected >=15 scenarios, got {}", meta.len());
    assert!(cases.len() >= 1000, "expected the per-seed corpus (>=1000 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut hp_assertions = 0usize;
    let mut weather_assertions = 0usize;
    let mut screen_assertions = 0usize;
    let mut weather_set_rows = 0usize; // a decision where weather is active (a set / upkeep turn)
    let mut screen_up_rows = 0usize; // a decision where a screen is up on either side
    let mut boost_dropped_rows = 0usize; // a foe stat-drop landed (a negative boost stage)
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
            "[{}] init prng seed must equal the sim's (switch-ins draw-free)",
            case.scen
        );

        apply_inject(&mut battle, &m.inject);

        let script = script_from_decisions(case);
        let outcome = battle.state_mut().unwrap().run_full_battle(&script, &d);

        assert_eq!(
            outcome.decisions.len(),
            case.decisions.len(),
            "[{}] decision count mismatch (init_seed {}): rust {} vs golden {}",
            case.scen, case.init_seed, outcome.decisions.len(), case.decisions.len()
        );

        for (di, (rec, exp)) in outcome.decisions.iter().zip(case.decisions.iter()).enumerate() {
            assert!(
                req_eq(&rec.request, exp.request, exp.force),
                "[{}] decision {} request mismatch (init_seed {}): got {:?} exp {:?} force {:?}",
                case.scen, di, case.init_seed, rec.request, exp.request, exp.force
            );

            // --- WEATHER (id + turns) — a MOVE-set weather counts 5→0 + expires; a wrong
            //     upkeep/expiry model diverges the turns; a wrong set/overwrite the id. ---
            assert_eq!(
                rec.weather, exp.weather,
                "[{}] dec {} WEATHER mismatch (init_seed {}): got {:?} exp {:?}\n  \
                 the weather-set / overwrite / expiry model is wrong. FIX THE MODEL.",
                case.scen, di, case.init_seed, rec.weather, exp.weather
            );
            assert_eq!(
                rec.weather_turns, exp.weather_turns,
                "[{}] dec {} WEATHER-TURNS mismatch (init_seed {}): got {} exp {}\n  \
                 the 5-turn upkeep countdown is wrong. FIX THE MODEL.",
                case.scen, di, case.init_seed, rec.weather_turns, exp.weather_turns
            );
            weather_assertions += 1;
            if exp.weather.is_some() {
                weather_set_rows += 1;
            }

            for (idx, (snap, e, ls, refl)) in [
                (0usize, (&rec.active[0], &exp.p1, rec.light_screen[0], rec.reflect[0])),
                (1usize, (&rec.active[1], &exp.p2, rec.light_screen[1], rec.reflect[1])),
            ] {
                assert_eq!(
                    species_id(&rec.active_species[idx]), species_id(&e.species),
                    "[{}] dec {} side {} active species mismatch (init_seed {}): got {:?} exp {:?}",
                    case.scen, di, idx, case.init_seed, rec.active_species[idx], e.species
                );
                // HP: the SCREEN halving (Reflect ×0.5 physical, Light Screen ×0.5 special)
                // lands HERE — a screen not consumed by the calc over-damages the defender.
                assert_eq!(
                    snap.hp, e.hp,
                    "[{}] dec {} side {} HP mismatch (init_seed {}): got {} exp {}\n  \
                     a screen halving / a cure / a weather effect is wrong. FIX THE MODEL, do not loosen.",
                    case.scen, di, idx, case.init_seed, snap.hp, e.hp
                );
                hp_assertions += 1;
                assert_eq!(
                    snap.maxhp, e.maxhp,
                    "[{}] dec {} side {} maxhp mismatch (init_seed {})",
                    case.scen, di, idx, case.init_seed
                );
                assert_eq!(
                    snap.fainted, e.fainted,
                    "[{}] dec {} side {} fainted mismatch (init_seed {}): got {} exp {}",
                    case.scen, di, idx, case.init_seed, snap.fainted, e.fainted
                );
                // SCREENS: Light Screen / Reflect turn counters (5→0). A wrong duration /
                // a failed re-use that reset the timer diverges HERE.
                assert_eq!(
                    ls, e.light_screen,
                    "[{}] dec {} side {} LIGHT-SCREEN mismatch (init_seed {}): got {} exp {}",
                    case.scen, di, idx, case.init_seed, ls, e.light_screen
                );
                assert_eq!(
                    refl, e.reflect,
                    "[{}] dec {} side {} REFLECT mismatch (init_seed {}): got {} exp {}",
                    case.scen, di, idx, case.init_seed, refl, e.reflect
                );
                screen_assertions += 2;
                if ls > 0 || refl > 0 {
                    screen_up_rows += 1;
                }

                if !e.fainted {
                    // STATUS: the CURE lands HERE (Refresh / Heal Bell / Aromatherapy clear
                    // the major status → None). A mis-cured / uncured mon diverges.
                    assert!(
                        status_variant_eq(snap.status, e.status),
                        "[{}] dec {} side {} STATUS mismatch (init_seed {}): got {:?} exp {:?}\n  \
                         the status-cure model is wrong. FIX THE MODEL, do not loosen.",
                        case.scen, di, idx, case.init_seed, snap.status, e.status
                    );
                    assert_eq!(
                        status_stage(snap.status), e.stage,
                        "[{}] dec {} side {} STATUS-COUNTER mismatch (init_seed {}): got {} exp {}",
                        case.scen, di, idx, case.init_seed, status_stage(snap.status), e.stage
                    );
                    // BOOST: the STAT-DROP (Screech −2 Def / Charm −2 Atk / …) lands HERE —
                    // clamped to the −6 floor, blocked by Clear Body / Hyper Cutter / Soundproof.
                    assert_eq!(
                        &snap.boosts[0..5], &e.boosts[..],
                        "[{}] dec {} side {} BOOST mismatch (init_seed {}): got {:?} exp {:?}\n  \
                         the foe stat-drop is wrong. FIX THE MODEL, do not loosen.",
                        case.scen, di, idx, case.init_seed, &snap.boosts[0..5], e.boosts
                    );
                    if e.boosts.iter().any(|&b| b < 0) {
                        boost_dropped_rows += 1;
                    }
                    let rust_conf = snap.confusion.unwrap_or(0);
                    assert_eq!(
                        rust_conf, e.confusion,
                        "[{}] dec {} side {} CONFUSION mismatch (init_seed {}): got {} exp {}",
                        case.scen, di, idx, case.init_seed, rust_conf, e.confusion
                    );
                }
            }
            assert_eq!(rec.pokemon_left[0], exp.p1.left, "[{}] dec {} p1 left", case.scen, di);
            assert_eq!(rec.pokemon_left[1], exp.p2.left, "[{}] dec {} p2 left", case.scen, di);

            if exp.request == ReqTok::Move {
                let sim_first: Option<usize> = match exp.first_mover.as_str() {
                    "p1" => Some(0),
                    "p2" => Some(1),
                    _ => None,
                };
                if sim_first.is_some() {
                    assert_eq!(
                        rec.first_mover, sim_first,
                        "[{}] dec {} FIRST-MOVER mismatch (init_seed {}): got {:?} exp {:?}",
                        case.scen, di, case.init_seed, rec.first_mover, sim_first
                    );
                }
            }

            // --- PER-DECISION SEED PARITY (the draw-order+count proof). The cures / screens
            //     / distinct-speed weather-set are DRAW-FREE; a SPEED-TIE weather-set draws
            //     the eachEvent('WeatherChange') shuffle; the stat-drops draw ONE accuracy
            //     roll. A stray/missing draw desyncs HERE. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 a batch-2 status-move drew (or skipped) a PRNG call — cures/screens/distinct-\
                 speed weather-set are draw-free; a speed-tie weather-set draws the WeatherChange \
                 shuffle; the stat-drops draw ONE accuracy roll. FIX THE DRAW MODEL.",
                case.scen, di, case.init_seed, rec.seed_after, exp.seed_after
            );
            seed_assertions += 1;
            dec_assertions += 1;
        }

        assert_eq!(
            outcome.ended, case.ended,
            "[{}] ended mismatch (init_seed {}): got {} exp {}",
            case.scen, case.init_seed, outcome.ended, case.ended
        );
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

    // Coverage floors: each mechanic must actually realize across the corpus.
    assert!(seed_assertions >= 1000, "expected the per-decision seed corpus (>=1000), got {seed_assertions}");
    assert!(hp_assertions >= 1000, "expected per-decision HP assertions (>=1000), got {hp_assertions}");
    assert!(weather_assertions >= 1000, "expected per-decision weather assertions (>=1000), got {weather_assertions}");
    assert!(screen_assertions >= 1000, "expected per-decision screen assertions (>=1000), got {screen_assertions}");
    assert!(weather_set_rows >= 100, "expected weather-active rows (>=100), got {weather_set_rows}");
    assert!(screen_up_rows >= 100, "expected screen-up rows (>=100), got {screen_up_rows}");
    assert!(boost_dropped_rows >= 100, "expected foe-stat-drop rows (>=100), got {boost_dropped_rows}");
    assert!(win_runs >= 40, "expected real game-end WIN runs (>=40), got {win_runs}");

    eprintln!(
        "movecoverage batch2 golden: {} runs, {dec_assertions} STATE rows, {seed_assertions} seed assertions, \
         {hp_assertions} HP assertions, {weather_assertions} weather assertions, {screen_assertions} screen \
         assertions ({weather_set_rows} weather-active rows, {screen_up_rows} screen-up rows, {boost_dropped_rows} \
         stat-drop rows), {win_runs} wins, {tie_runs} ties",
        cases.len()
    );
}
