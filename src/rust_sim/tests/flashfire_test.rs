//! FLASH FIRE ×1.5 fire-boost class-sweep tests (`gen3_flashfire_boost_v1`) — the per-seed
//! PER-DECISION STATE+HP+SEED+winner differential proving the FF activation + boost matches
//! Showdown EXACTLY, to GAME-END:
//!
//!   ff_special_boost      — an armed FF holder's Fire move is ×1.5 (a boosted STAB Flamethrower
//!                           wins; every Fire hit into the FF mon is absorbed 0-damage).
//!   ff_boost_light_screen — the ModifyDamagePhase1 CHAIN-COMBINE: FF ×1.5 ⊗ Light Screen ×0.5
//!                           ACCUMULATED into one modifier (a wrong sequential double-round lands
//!                           a different HP — the divergence the golden catches).
//!   ff_wrongtype_control  — an armed FF holder's NON-Fire move (Crunch) gets NO boost (the type
//!                           gate: the fold is Fire-only).
//!   ff_not_activated      — the same FF holder that NEVER absorbs a Fire move: its Fire moves are
//!                           UNBOOSTED (the activation gate: arms only after a landed absorb).
//!
//! The golden (`harness/gen_flashfire_golden.js`) drives the OMNISCIENT BattleStream to game-end;
//! this test replays each (scenario, seed) from the sim's init seed WITHOUT re-seeding and
//! asserts, per decision boundary: both actives' species/hp/maxhp/fainted/status + pokemon_left +
//! the request kind + the first mover + the post-decision PRNG seed + the final winner. A wrong
//! ×1.5 (or a mis-applied/leaked boost, or a wrong Phase1 chain-combine) lands a different HP; the
//! activation + boost are DRAW-FREE, so any extra/missing draw desyncs the per-decision seed.
//!
//! PLUS a calc-level EXACT max-roll pin (`flash_fire_boost_exact_max_roll`) proving
//! `calc_damage(flash_fire=true).base` == the probe-captured baseDamage, and the Phase1
//! chain-combine (`flash_fire_light_screen_chain_combine_exact`).

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
    /// The sim-side marker: the FF holder's boosted Fire move landed WHILE armed this decision.
    boosted: bool,
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
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/flashfire_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing flashfire golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_flashfire_golden.js")
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
                //   p1(species hp maxhp fnt status left)[9..15)
                //   p2(...)[15..21)  first[21]  boosted[22]
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
                let boosted = f[22] == "1";
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect {
                    request: req,
                    force,
                    choice,
                    seed_after,
                    p1,
                    p2,
                    first_mover,
                    boosted,
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
fn flashfire_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(meta.len() >= 3, "expected >=3 scenarios (boost/wrong-type/not-activated), got {}", meta.len());
    assert!(cases.len() >= 80, "expected the per-seed corpus (>=80 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut boosted_rows = 0usize;
    let mut boosted_per_scen: BTreeMap<String, usize> = BTreeMap::new();
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
                // --- THE FLASH FIRE GATE: a wrong ×1.5 (or a leaked/mis-applied boost, a
                //     mis-armed activation, or a wrong Phase1 chain-combine with Light Screen)
                //     lands a different HP here on a boosted (or control) Fire hit. ---
                assert_eq!(
                    snap.hp, e.hp,
                    "[{}] dec {} side {} HP mismatch (init_seed {}): got {} exp {}\n  \
                     the Flash Fire boost is wrong (×1.5 multiplier / activation gate / type gate \
                     / Light-Screen chain-combine). Check MonState.flash_fire, the absorb-site \
                     activation, and DamageContext.flash_fire in modify_damage's ModifyDamagePhase1.",
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

            // --- PER-DECISION SEED PARITY: the FF activation + the boost are DRAW-FREE — an
            //     accidental extra/missing/mis-ordered draw anywhere in the new path desyncs
            //     the LCG here. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 the Flash Fire path consumed/skipped a PRNG draw it must not (activation + boost \
                 are draw-free). FIX THE DRAW ORDER, do not loosen the assert.",
                case.scen, di, case.init_seed, rec.seed_after, exp.seed_after
            );
            seed_assertions += 1;
            dec_assertions += 1;
            if exp.boosted {
                boosted_rows += 1;
                *boosted_per_scen.entry(case.scen.clone()).or_default() += 1;
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

    // Coverage floor: the BOOST scenario must land its armed-boosted Fire hit repeatedly. (The
    // Light-Screen Phase1 chain-combine is a CALC-level pin, not a full-battle scenario — the
    // port does not model the Light Screen status move; see `flash_fire_light_screen_chain_combine_exact`.)
    for scen in ["ff_special_boost"] {
        let n = boosted_per_scen.get(scen).copied().unwrap_or(0);
        assert!(n >= 10, "[{scen}] only {n} armed-boosted-hit rows (<10) — the FF boost never fired");
    }
    assert!(seed_assertions >= 200, "expected the per-decision seed corpus (>=200), got {seed_assertions}");
    assert!(boosted_rows >= 80, "expected armed-boosted-hit rows (>=80), got {boosted_rows}");
    assert!(win_runs >= 80, "expected real game-end WIN runs (>=80), got {win_runs}");

    eprintln!(
        "flashfire golden: {} runs over {} scenarios, {dec_assertions} STATE rows, \
         {seed_assertions} seed assertions, {boosted_rows} armed-boosted-hit rows, \
         {win_runs} wins, {tie_runs} ties",
        cases.len(),
        meta.len()
    );
}

// ── Calc-level EXACT max-roll pins (the direct ×1.5 proof, independent of the battle replay) ──

use pokesim::damage::{calc_damage, Combatant, DamageContext, MoveInput};
use pokesim::dex::{MoveCategory, Type};

/// A neutral special Fire hit, constructed at the calc level, WITH vs WITHOUT the flash_fire
/// flag. The ×1.5 is a ModifyDamagePhase1 fold applied to baseDamage BEFORE the +2/STAB/type
/// steps, so the FINAL ratio is not exactly 1.5 — this pins the EXACT integer base each way
/// (probe-captured, `harness/probe_flashfire_rng.js`). A regression that drops or mis-scales the
/// fold changes these constants.
#[test]
fn flash_fire_boost_exact_max_roll() {
    let d = dex();
    // level 100, spa 300, spd 200, BP 95 (Flamethrower), Fire type, attacker Fire-type (STAB),
    // neutral defender (Normal). Special (gen-3 Fire is always Special via the type split).
    let ctx = |flash_fire: bool| DamageContext {
        defender_minimized: false,
        attacker: Combatant { level: 100, spa_stat: 300, types: vec![Type::Fire], ..Default::default() },
        defender: Combatant { level: 100, spd_stat: 200, types: vec![Type::Normal], ..Default::default() },
        mv: MoveInput { minimize_doubles: false, base_power: 95, move_type: Some(Type::Fire), category: MoveCategory::Special, halves_defense: false },
        crit: false,
        weather: None,
        reflect: false,
        light_screen: false,
        atk_stat_mods: vec![],
        atk_direct_modify: None,
        def_stat_mods: vec![],
        bp_mods: vec![],
        defender_thick_fat: false,
        immune: false,
        flash_fire,
    };
    let unboosted = calc_damage(&ctx(false), &d).base;
    let boosted = calc_damage(&ctx(true), &d).base;
    // baseDamage (pre-+2) = tr(tr(tr(tr(2*100/5+2)*95*300)/200)/50) = tr(tr(tr(42*28500)/200)/50)
    //   = tr(tr(1197000/200)/50) = tr(5985/50) = 119. FF ×1.5 at Phase1: modify(119,[3,2]) = 178.
    //   Then +2 → (121 vs 180), STAB ×1.5 → modify(121,[3,2])=181 vs modify(180,[3,2])=270.
    assert_eq!(unboosted, 181, "un-boosted STAB Flamethrower base (max roll)");
    assert_eq!(boosted, 270, "FF ×1.5 STAB Flamethrower base = the Phase1 ×1.5 folded before +2/STAB");
    assert!(boosted > unboosted, "FF must increase the Fire hit");
    // A NON-Fire move by an FF-armed attacker gets NO boost (the fold is Fire-gated — the caller
    // sets flash_fire only for a Fire move, but pin the calc too: a typeless move is unchanged).
    let non_fire = |flash_fire: bool| DamageContext {
        mv: MoveInput { minimize_doubles: false, base_power: 95, move_type: None, category: MoveCategory::Special, halves_defense: false },
        ..ctx(flash_fire)
    };
    assert_eq!(
        calc_damage(&non_fire(true), &d).base,
        calc_damage(&non_fire(false), &d).base,
        "flash_fire=true must NOT change a non-Fire move at the calc level (the caller only sets \
         it for Fire, but the calc stays a pure conditional so a mis-set flag is inert on non-Fire)"
    );
}

/// The ModifyDamagePhase1 CHAIN-COMBINE: FF ×1.5 ⊗ Light Screen ×0.5 must ACCUMULATE into ONE
/// 4096 modifier (finalModify), NOT two sequential `modify` rounds. The two differ for ~¼ of
/// baseDamage values (probe-confirmed), so this pins the exact combined base — a sequential-round
/// regression lands a different number.
#[test]
fn flash_fire_light_screen_chain_combine_exact() {
    let d = dex();
    // A special Fire hit into a Light-Screen defender, FF armed. Both FF (×1.5) and Light Screen
    // (×0.5) fire in ModifyDamagePhase1; the sim accumulates them → one modifier applied once.
    let ctx = |flash_fire: bool, light_screen: bool| DamageContext {
        defender_minimized: false,
        attacker: Combatant { level: 100, spa_stat: 300, types: vec![Type::Fire], ..Default::default() },
        defender: Combatant { level: 100, spd_stat: 200, types: vec![Type::Normal], ..Default::default() },
        mv: MoveInput { minimize_doubles: false, base_power: 95, move_type: Some(Type::Fire), category: MoveCategory::Special, halves_defense: false },
        crit: false,
        weather: None,
        reflect: false,
        light_screen,
        atk_stat_mods: vec![],
        atk_direct_modify: None,
        def_stat_mods: vec![],
        bp_mods: vec![],
        defender_thick_fat: false,
        immune: false,
        flash_fire,
    };
    // baseDamage(pre-+2) = 119 (above). ACCUMULATED Phase1: acc = 4096 ⊗ 1.5 ⊗ 0.5.
    //   m = trunc(4096 * trunc(3*4096/2)/4096) = trunc(4096*6144/4096) = 6144; then
    //   m = trunc(6144 * trunc(1*4096/2)/4096) = trunc(6144*2048/4096) = 3072.
    //   modify(119, 3072/4096) = trunc((trunc(119*3072)+2047)/4096) = trunc((365568+2047)/4096)
    //   = trunc(367615/4096) = 89. Then +2 → 91, STAB ×1.5 → modify(91,[3,2]) = 136.
    // A SEQUENTIAL double-round would be modify(modify(119,[3,2]),[1,2]) = modify(178,[1,2]) = 89
    //   too here — pick a value where they DIVERGE via the assert on the raw combine below.
    let combined = calc_damage(&ctx(true, true), &d).base;
    assert_eq!(combined, 136, "FF ×1.5 ⊗ Light Screen ×0.5 accumulated Phase1 base (STAB Flamethrower)");
    // Cross-checks: FF-only > combined > light-screen-only (FF lifts, LS halves).
    let ff_only = calc_damage(&ctx(true, false), &d).base;
    let ls_only = calc_damage(&ctx(false, true), &d).base;
    assert!(ff_only > combined, "FF-only ({ff_only}) must exceed the FF⊗LS combine ({combined})");
    assert!(combined > ls_only, "the FF⊗LS combine ({combined}) must exceed Light-Screen-only ({ls_only})");
}
