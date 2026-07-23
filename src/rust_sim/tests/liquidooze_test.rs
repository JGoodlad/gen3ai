//! LIQUID OOZE heal-reversal test (`gen3_liquid_ooze_v1`) — the per-seed PER-DECISION
//! STATE+HP+STATUS+SEED+winner differential proving the reversal matches Showdown EXACTLY, to
//! GAME-END. Liquid Ooze turns a `drain` / Leech Seed heal into DAMAGE on the HEALER
//! (`onSourceTryHeal`), DRAW-FREE.
//!
//! The reversal is OBSERVABLE on the HP timeline: the healer's HP goes DOWN (not up) after a drain
//! / on the leech residual (`|-damage|<healer>|<HP>|[from] ability: Liquid Ooze|[of] <ooze-mon>`).
//! `lo_control_heal` runs the IDENTICAL plan with a Clear-Body Tentacruel (Venusaur HEALS from Giga
//! Drain) — the discriminator: a modeled reversal is the only thing that makes lo_drain_reversal's
//! HP timeline differ from the control's.
//!
//!   lo_drain_reversal — Giga Drain into a Liquid Ooze mon: the DRAINER takes the would-be heal as
//!                       damage (its HP drops).
//!   lo_leech_reversal — Leech Seed into a Liquid Ooze mon: each residual the SEEDER takes the
//!                       drained amount as damage.
//!   lo_control_heal   — the wrong-target control (Clear Body → Venusaur HEALS, 0 reversal rows).
//!
//! The golden (`harness/gen_liquidooze_golden.js`) drives the OMNISCIENT BattleStream to game-end
//! (p2 never attacks → guaranteed P1 win); this test replays each (scenario, seed) from the sim's
//! init seed WITHOUT re-seeding and asserts, per decision boundary: both actives' species / hp /
//! maxhp / fainted / status + pokemon_left + request kind + first mover + the post-decision PRNG
//! seed + the final winner. The reversal is DRAW-FREE, so the per-decision seed must match the
//! drain move / leech residual bit-for-bit; the HP column is the reversal proof.

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::state::Status;
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
    first_mover: String,
    /// The sim-side marker: a Liquid-Ooze reversal `-damage` fired this decision (coverage only).
    oozed: bool,
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
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/liquidooze_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing liquidooze golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_liquidooze_golden.js")
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
                //   p1(species hp maxhp fnt status left)[9..15) p2(...)[15..21) first[21] oozed[22]
                assert_eq!(f.len(), 23, "DEC needs 23 fields (line {ln}), got {}", f.len());
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
                };
                let p2 = SideExpect {
                    species: f[15].to_string(),
                    hp: g(16),
                    maxhp: g(17),
                    fainted: f[18] == "1",
                    status: parse_status(f[19]),
                    left: g(20) as usize,
                };
                let first_mover = f[21].to_string();
                let oozed = f[22] == "1";
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect { request: req, force, choice, seed_after, p1, p2, first_mover, oozed });
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
fn liquidooze_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(meta.len() >= 3, "expected >=3 scenarios (drain + leech reversal + control), got {}", meta.len());
    assert!(cases.len() >= 90, "expected the per-seed corpus (>=90 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut hp_assertions = 0usize;
    let mut ooze_rows = 0usize;
    let mut ooze_per_scen: BTreeMap<String, usize> = BTreeMap::new();
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
            case.scen, case.init_seed, outcome.decisions.len(), case.decisions.len()
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
                // --- THE LIQUID OOZE GATE: the HP timeline. A drain / leech into a Liquid Ooze mon
                //     DAMAGES the healer (its HP drops) instead of healing; a wrong/absent reversal
                //     leaves the healer at the healed HP → diverges here (and the winner). ---
                assert_eq!(
                    snap.hp, e.hp,
                    "[{}] dec {} side {} HP mismatch (init_seed {}): got {} exp {}\n  \
                     Liquid Ooze must DAMAGE the drainer/seeder by the would-be heal (apply_drain / \
                     apply_leech_seed reversal), not heal it.",
                    case.scen, di, idx, case.init_seed, snap.hp, e.hp
                );
                hp_assertions += 1;
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
                        "[{}] dec {} first-mover mismatch (init_seed {})",
                        case.scen, di, case.init_seed
                    );
                }
            }

            // --- PER-DECISION SEED PARITY: the reversal is DRAW-FREE — a spurious draw would
            //     desync the LCG here. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 the Liquid Ooze reversal consumed/skipped a PRNG draw it must not (draw-free).",
                case.scen, di, case.init_seed, rec.seed_after, exp.seed_after
            );
            seed_assertions += 1;
            dec_assertions += 1;
            if exp.oozed {
                ooze_rows += 1;
                *ooze_per_scen.entry(case.scen.clone()).or_default() += 1;
            }
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

    // Coverage floor: the drain + leech reversal scenarios fire the reversal repeatedly; the
    // Clear-Body control fires ZERO (the wrong-target discriminator).
    for scen in ["lo_drain_reversal", "lo_leech_reversal"] {
        let n = ooze_per_scen.get(scen).copied().unwrap_or(0);
        assert!(n >= 10, "[{scen}] only {n} Liquid-Ooze reversal rows (<10) — the reversal never fired");
    }
    let ctl = ooze_per_scen.get("lo_control_heal").copied().unwrap_or(0);
    assert_eq!(ctl, 0, "[lo_control_heal] expected 0 reversal rows (Clear Body heals), got {ctl}");

    assert!(seed_assertions >= 500, "expected the per-decision seed corpus (>=500), got {seed_assertions}");
    assert!(win_runs >= 90, "expected real game-end WIN runs (>=90), got {win_runs}");

    eprintln!(
        "liquidooze golden: {} runs over {} scenarios, {dec_assertions} STATE rows, \
         {hp_assertions} HP assertions, {seed_assertions} seed assertions, \
         {ooze_rows} reversal rows, {win_runs} wins, {tie_runs} ties",
        cases.len(),
        meta.len()
    );
}
