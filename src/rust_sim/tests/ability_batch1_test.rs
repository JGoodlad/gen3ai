//! ABILITY BATCH-1 class-sweep tests (`gen3_ability_batch1_v1`) — the per-seed PER-DECISION
//! STATE+HP+SPE-BOOST+SEED+winner differential proving the four DRAW-FREE / STRUCTURAL ability
//! classes wired in this batch match Showdown EXACTLY, to GAME-END:
//!
//!   CRIT_IMMUNE (Shell Armor / Battle Armor) — a hit into the armor holder NEVER crits (the
//!     crit `randomChance` is DRAWN then OVERRIDDEN false). A high-crit foe move (Slash /
//!     Crabhammer, 1/8) maximizes the chance the crit roll comes up → a model that didn't
//!     override deals 2× on those seeds and diverges the HP. A no-op-ability control DOES crit.
//!   WEATHER_SPEED (Chlorophyll / Swift Swim) — ×2 effective speed in sun / rain folded into the
//!     cached speed the tie-shuffles read. A slow weather-speed mon FLIPS the first-mover once
//!     the foe's own Drought / Drizzle sets the weather. A no-op control never flips.
//!   WEATHER_NEGATE (Cloud Nine / Air Lock) — suppresses the weather's effects: a non-immune mon
//!     takes NO sand chip (HP). The raw weather persists; only the effects die.
//!   RESIDUAL (Speed Boost / Rain Dish) — Speed Boost +1 spe stage per active turn (activeTurns-
//!     gated), Rain Dish +maxhp/16 heal each end-of-turn in rain — asserted via the spe-boost
//!     stage + HP.
//!
//! The golden (`harness/gen_ability_batch1_golden.js`) drives the OMNISCIENT BattleStream to
//! game-end; this test replays each (scenario, seed) from the sim's init seed WITHOUT re-seeding
//! and asserts, per decision boundary: both actives' species/hp/maxhp/fainted/status + the SPE
//! boost stage + pokemon_left + the request kind + the first mover + the post-decision PRNG seed
//! + the final winner. All four classes are DRAW-FREE (the crit roll STILL fires — the override is
//! a return-value, not a skipped draw), so ANY extra/missing draw desyncs the seed here.

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::state::Status;
use pokesim::turn::{Choice, RequestKind, ScriptDecision};
use std::collections::BTreeMap;

fn dex() -> Dex {
    Dex::for_gen(3)
}

/// The spe boost-stage index in `MonSnapshot::boosts` (`boosts[4]` == spe).
const SPE_BOOST_IDX: usize = 4;

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
    first_mover: String,
    /// The sim-side coverage marker: this class's observable effect fired this decision.
    covered: bool,
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
    left: usize,
    spe_boost: i8,
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
        "-" => None,
        "fnt" => None,
        "brn" => Some(Status::Burn),
        "par" => Some(Status::Paralysis),
        "slp" => Some(Status::Sleep(0)),
        "frz" => Some(Status::Freeze),
        "psn" => Some(Status::Poison),
        "tox" => Some(Status::Toxic(0)),
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

fn parse_golden() -> (BTreeMap<String, ScenMeta>, Vec<RunCase>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/ability_batch1_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing ability batch1 golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_ability_batch1_golden.js")
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
                //   p1(species hp maxhp fnt status left speBoost)[9..16)
                //   p2(...)[16..23)  first[23]  covered[24]
                assert_eq!(f.len(), 25, "DEC needs 25 fields (line {ln}), got {}", f.len());
                let req = match f[3] {
                    "move" => ReqTok::Move,
                    "switch" => ReqTok::Switch,
                    other => panic!("bad request {other:?} (line {ln})"),
                };
                let force = [f[4] == "1", f[5] == "1"];
                let choice = [parse_choice(f[6]), parse_choice(f[7])];
                let seed_after = f[8].to_string();
                let gu16 = |i: usize| {
                    f[i].parse::<u16>().unwrap_or_else(|e| panic!("bad num f[{i}] (line {ln}): {e}"))
                };
                let gi8 = |i: usize| {
                    f[i].parse::<i8>().unwrap_or_else(|e| panic!("bad boost f[{i}] (line {ln}): {e}"))
                };
                let p1 = SideExpect {
                    species: f[9].to_string(),
                    hp: gu16(10),
                    maxhp: gu16(11),
                    fainted: f[12] == "1",
                    status: parse_status(f[13]),
                    left: gu16(14) as usize,
                    spe_boost: gi8(15),
                };
                let p2 = SideExpect {
                    species: f[16].to_string(),
                    hp: gu16(17),
                    maxhp: gu16(18),
                    fainted: f[19] == "1",
                    status: parse_status(f[20]),
                    left: gu16(21) as usize,
                    spe_boost: gi8(22),
                };
                let first_mover = f[23].to_string();
                let covered = f[24] == "1";
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect {
                    request: req,
                    force,
                    choice,
                    seed_after,
                    p1,
                    p2,
                    first_mover,
                    covered,
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
fn ability_batch1_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(
        meta.len() >= 10,
        "expected >=10 scenarios (all four classes + controls), got {}",
        meta.len()
    );
    assert!(cases.len() >= 250, "expected the per-seed corpus (>=250 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut spe_boost_assertions = 0usize;
    let mut covered_rows = 0usize;
    let mut covered_per_scen: BTreeMap<String, usize> = BTreeMap::new();
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
            case.scen,
            case.init_seed,
            outcome.decisions.len(),
            case.decisions.len()
        );

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
                assert_eq!(
                    species_id(sp),
                    species_id(&e.species),
                    "[{}] dec {} side {} active species mismatch (init_seed {})",
                    case.scen, di, idx, case.init_seed
                );
                // --- THE STATE GATE: a wrong crit-override (CRIT_IMMUNE), a wrong sand-chip
                //     suppression (WEATHER_NEGATE), or a wrong Rain-Dish heal lands a different
                //     HP here. ---
                assert_eq!(
                    snap.hp, e.hp,
                    "[{}] dec {} side {} HP mismatch (init_seed {}): got {} exp {}\n  \
                     a batch-1 ability effect is wrong (a crit that should have been prevented \
                     by Shell/Battle Armor, a sand chip a Cloud Nine / Air Lock mon should have \
                     suppressed, or a Rain Dish heal). Check the crit override + effective_weather \
                     + the RESIDUAL apply.",
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
                    // --- THE SPE-BOOST GATE: Speed Boost's +1/turn residual (activeTurns-gated)
                    //     must land the exact stage; a wrong gate / timing diverges here. ---
                    assert_eq!(
                        snap.boosts[SPE_BOOST_IDX], e.spe_boost,
                        "[{}] dec {} side {} SPE-BOOST mismatch (init_seed {}): got {} exp {}\n  \
                         Speed Boost's +1 spe residual is wrong (the activeTurns gate / timing / \
                         residual order).",
                        case.scen, di, idx, case.init_seed, snap.boosts[SPE_BOOST_IDX], e.spe_boost
                    );
                    spe_boost_assertions += 1;
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
                    // --- THE FIRST-MOVER GATE: WEATHER_SPEED's ×2 (and WEATHER_NEGATE's
                    //     suppression of it) flips who moves first. A wrong effective speed
                    //     picks the wrong first mover here. ---
                    assert_eq!(
                        rec.first_mover, sim_first,
                        "[{}] dec {} first-mover mismatch (init_seed {}): the weather-speed ×2 \
                         (or its negater suppression) resolved the wrong action order",
                        case.scen, di, case.init_seed
                    );
                }
            }

            // --- PER-DECISION SEED PARITY: all four classes are DRAW-FREE (CRIT_IMMUNE still
            //     DRAWS the crit roll — the override is a return value, not a skipped draw). An
            //     accidental extra/missing/mis-ordered draw desyncs the LCG here. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 a batch-1 ability consumed/skipped a PRNG draw it must not (the classes are \
                 draw-free; the crit roll STILL fires under Shell/Battle Armor). FIX THE DRAW \
                 ORDER, do not loosen the assert.",
                case.scen, di, case.init_seed, rec.seed_after, exp.seed_after
            );
            seed_assertions += 1;
            dec_assertions += 1;
            if exp.covered {
                covered_rows += 1;
                *covered_per_scen.entry(case.scen.clone()).or_default() += 1;
            }
        }

        assert_eq!(
            outcome.ended, case.ended,
            "[{}] ended mismatch (init_seed {})",
            case.scen, case.init_seed
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

    // Coverage floors: EVERY scenario (each class member + control) must land its observable
    // effect repeatedly across the corpus.
    for (scen, _) in meta.iter() {
        let n = covered_per_scen.get(scen).copied().unwrap_or(0);
        assert!(n >= 10, "[{scen}] only {n} covered rows (<10) — the class effect never fired");
    }
    assert!(seed_assertions >= 800, "expected the per-decision seed corpus (>=800), got {seed_assertions}");
    assert!(covered_rows >= 400, "expected covered rows (>=400), got {covered_rows}");
    assert!(win_runs >= 200, "expected real game-end WIN runs (>=200), got {win_runs}");

    eprintln!(
        "ability batch1 golden: {} runs over {} scenarios, {dec_assertions} STATE rows, \
         {seed_assertions} seed assertions, {spe_boost_assertions} spe-boost assertions, \
         {covered_rows} covered rows, {win_runs} wins, {tie_runs} ties",
        cases.len(),
        meta.len()
    );
}
