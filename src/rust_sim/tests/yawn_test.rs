//! YAWN delayed-sleep test (`gen3_yawn_v1`) — the per-seed PER-DECISION
//! STATE+HP+STATUS+SLEEP-COUNTER+SEED+winner differential proving Yawn matches Showdown EXACTLY,
//! to GAME-END.
//!
//! gen-3 Yawn (`volatileStatus: 'yawn'`, `accuracy: true`) is a category-Status foe-target move
//! whose CRUX is that the sleep `random(2,6)` fires at RESOLVE (the residual `onEnd`), not at cast —
//! the CAST is entirely DRAW-FREE. The `yawn` condition (`duration: 2`, order 10 subOrder 19)
//! decrements 2 → 1 (end of the cast turn) then 1 → 0 (end of the NEXT turn); on the 1 → 0 tick the
//! `onEnd` emits `|-end|<t>|move: Yawn|[silent]` then `target.trySetStatus('slp', source)` — routed
//! through the EXISTING `try_set_status` path, so the sleep `random(2,6)` onStart draw + the sleep
//! counter come for free. So the sleep lands at the END of the turn AFTER cast, at the exact counter.
//!
//! The golden (`harness/gen_yawn_golden.js`, p1 = mono Snorlax that never takes damage → guaranteed
//! decisive P1) drives the OMNISCIENT BattleStream to game-end; this test replays each (scenario,
//! seed) from the sim's init seed WITHOUT re-seeding and asserts, per decision boundary: both
//! actives' species / hp / maxhp / fainted / STATUS + the SLEEP INNER COUNTER + pokemon_left +
//! request kind + first mover + the post-decision PRNG seed + the final winner. `yawn_resolve_
//! sleep_wake` observes the full lifecycle (cast → resolve → the counter → the wake → KO);
//! `yawn_into_statused` / `yawn_into_vitalspirit` prove the draw-free fail / immune (NEVER sleeps);
//! `yawn_statused_between` proves the `-end [silent]` with NO sleep; `yawn_real_battle` proves Yawn
//! composes in a multi-mon game with a forced replacement. The KEY draw-model crux is the SEED: the
//! resolve's `random(2,6)` must be at the RIGHT turn (a wrong place/absent desyncs the seed).

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
    /// The sim-side marker: a Yawn `|-end|...|move: Yawn` RESOLVE fired this decision (coverage).
    yawn_resolved: bool,
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
    /// The SLEEP inner counter (`statusState.time` = remaining turns; 0 else) — THE draw-model
    /// signal that the resolve's `random(2,6)` landed the exact duration + the wake decrements.
    sleep: u8,
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

/// The sleep INNER COUNTER (`Sleep(n)` → `n`; else 0) — asserted SEPARATELY from the variant so a
/// wrong resolve duration (the `random(2,6)`) or a wrong decrement diverges here.
fn sleep_counter(s: Option<Status>) -> u8 {
    match s {
        Some(Status::Sleep(n)) => n,
        _ => 0,
    }
}

fn parse_golden() -> (BTreeMap<String, ScenMeta>, Vec<RunCase>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/yawn_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing yawn golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_yawn_golden.js")
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
                //   p1(species hp maxhp fnt status left)[9..15) sleep1[15]
                //   p2(...)[16..22) sleep2[22]  first[23]  yawnResolved[24]
                assert_eq!(f.len(), 25, "DEC needs 25 fields (line {ln}), got {}", f.len());
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
                let sc = |i: usize| {
                    f[i].parse::<u8>().unwrap_or_else(|e| panic!("bad sleep f[{i}] (line {ln}): {e}"))
                };
                let p1 = SideExpect {
                    species: f[9].to_string(),
                    hp: g(10),
                    maxhp: g(11),
                    fainted: f[12] == "1",
                    status: parse_status(f[13]),
                    left: g(14) as usize,
                    sleep: sc(15),
                };
                let p2 = SideExpect {
                    species: f[16].to_string(),
                    hp: g(17),
                    maxhp: g(18),
                    fainted: f[19] == "1",
                    status: parse_status(f[20]),
                    left: g(21) as usize,
                    sleep: sc(22),
                };
                let first_mover = f[23].to_string();
                let yawn_resolved = f[24] == "1";
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect { request: req, force, choice, seed_after, p1, p2, first_mover, yawn_resolved });
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
fn yawn_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(meta.len() >= 5, "expected the 5 yawn scenarios, got {}", meta.len());
    assert!(cases.len() >= 100, "expected the per-seed corpus (>=100 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut sleep_counter_assertions = 0usize;
    let mut sleep_rows = 0usize;
    let mut resolve_rows_per_scen: BTreeMap<String, usize> = BTreeMap::new();
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
                assert_eq!(
                    snap.hp, e.hp,
                    "[{}] dec {} side {} HP mismatch (init_seed {}): got {} exp {}",
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
                        "[{}] dec {} side {} STATUS mismatch (init_seed {}): got {:?} exp {:?}\n  \
                         a Yawn wrongly slept / failed / immune'd (or the wrong turn).",
                        case.scen, di, idx, case.init_seed, snap.status, e.status
                    );
                    // --- THE YAWN GATE: the SLEEP inner counter must match. The resolve's
                    //     random(2,6) sets the exact duration; a wrong duration (or a sleep on the
                    //     wrong turn / an absent sleep) diverges here. ---
                    assert_eq!(
                        sleep_counter(snap.status), e.sleep,
                        "[{}] dec {} side {} SLEEP-COUNTER mismatch (init_seed {}): got {} exp {}\n  \
                         the Yawn resolve's random(2,6) duration / the wake decrement diverged.",
                        case.scen, di, idx, case.init_seed, sleep_counter(snap.status), e.sleep
                    );
                    sleep_counter_assertions += 1;
                    if matches!(e.status, Some(Status::Sleep(_))) {
                        sleep_rows += 1;
                    }
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

            // --- PER-DECISION SEED PARITY (THE DRAW-MODEL CRUX): the Yawn CAST is DRAW-FREE and the
            //     RESOLVE draws exactly ONE random(2,6) — at the RIGHT turn. A cast that drew, a
            //     resolve at the wrong place, or an absent random(2,6) desyncs the LCG here. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 the Yawn cast/resolve consumed/skipped a PRNG draw it must not (cast draw-free; \
                 resolve draws random(2,6) at the RIGHT turn).",
                case.scen, di, case.init_seed, rec.seed_after, exp.seed_after
            );
            seed_assertions += 1;
            dec_assertions += 1;
            if exp.yawn_resolved {
                *resolve_rows_per_scen.entry(case.scen.clone()).or_default() += 1;
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

    // Coverage floor: the two sleeping scenarios must fire the resolve repeatedly, and the fail /
    // immune / statused-between scenarios must NEVER sleep (proven by the STATUS + counter columns).
    for scen in ["yawn_resolve_sleep_wake", "yawn_real_battle"] {
        let n = resolve_rows_per_scen.get(scen).copied().unwrap_or(0);
        assert!(n >= 10, "[{scen}] only {n} Yawn-resolve rows (<10) — the resolve never fired");
    }
    assert!(sleep_rows >= 30, "expected the asleep corpus (>=30 asleep rows), got {sleep_rows}");
    assert!(seed_assertions >= 500, "expected the per-decision seed corpus (>=500), got {seed_assertions}");
    assert!(win_runs >= 100, "expected real game-end WIN runs (>=100), got {win_runs}");

    eprintln!(
        "yawn golden: {} runs over {} scenarios, {dec_assertions} STATE rows, \
         {sleep_counter_assertions} sleep-counter assertions, {seed_assertions} seed assertions, \
         {sleep_rows} asleep rows, {win_runs} wins, {tie_runs} ties",
        cases.len(),
        meta.len()
    );
}
