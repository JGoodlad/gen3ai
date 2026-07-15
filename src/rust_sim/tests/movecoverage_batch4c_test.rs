//! MOVE-COVERAGE BATCH 4c full-battle test (`gen3_move_coverage_batch4c_v1`) — the per-seed
//! PER-DECISION STATE(+HP+STATUS+BOOSTS+SUB-HP+WISH+FUTURE-PENDING)+SEED+winner differential
//! that proves the TURN-SPANNING move classes match Showdown EXACTLY, to GAME-END:
//!
//!   * **HYPER BEAM** (`mustrecharge`) — a SUCCESSFUL damaging hit (plain / sub-absorb /
//!     sub-break / target-KO) locks the user: the next turn is `|cant|…|recharge` at the
//!     user's normal speed-order position, drawing ZERO and costing NO PP (the priority-11
//!     onBeforeMove precedes every status draw — a par'd/slp'd user rolls NOTHING on the
//!     locked turn), then the lock clears. A MISS / IMMUNE / PROTECT-block does NOT lock
//!     (PP still consumed). The cast-turn residual gathers a NO_ORDER duration handler
//!     (the HB-mirror tie draw).
//!   * **SOLAR BEAM** (`twoturnmove`) — CHARGE: PP paid (−1 / −2 vs Pressure), zero move
//!     draws, the volatile registers residual duration handlers on BOTH turns; FIRE: a
//!     locked single-move request, NO PP, accuracy-100 DRAWN → crit → damage; SUN skips
//!     the charge (3 draws like a normal move); an ABORT on the fire turn LOSES the
//!     charge; rain/sand/hail HALVE the BP (state-only — the damage differs at the SAME
//!     seed).
//!   * **DOOM DESIRE / FUTURE SIGHT** (`futuremove`) — the cast draws exactly ONE
//!     random(16) (the cast-time typeless damage SNAPSHOT); the pending slot condition
//!     registers an order-11 residual handler every end-of-turn (the FUTURE-PENDING column
//!     asserts the 3→2→1→resolve countdown); the resolve draws ONE accuracy roll then the
//!     STORED number lands on WHOEVER occupies the slot (sub absorbs; a resolve KO defers
//!     the Quick Claw past the forced replacement); a double-cast fails draw-free with PP
//!     still deducted.
//!
//! Each move's draws must be in Showdown's EXACT place/count — a stray/missing draw
//! desyncs the LCG at the SEED assertion; the STATE columns are the effect proof. The
//! golden EXTENDS the batch-3/4 42-field TAB format with 2 trailing per-side
//! FUTURE-PENDING columns → DEC has 44 fields. CURSE stays 0 (asserted); WISH is live in
//! ONE scenario (the residual-order composition).

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::json::Json;
use pokesim::state::Status;
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
    side: Option<usize>,
    slot: Option<usize>,
    status: Option<Status>,
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
    curse: bool,
    wish: u8,
    sub_hp: u16,
    future: u8,
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
            let side = e.get("side").and_then(|s| s.as_f64()).map(|f| f as usize);
            let slot = e.get("slot").and_then(|s| s.as_f64()).map(|f| f as usize);
            let status = e.str_at("status").map(status_of);
            let hp = e.get("hp").and_then(|h| h.as_f64()).map(|f| f as u16);
            Inject { side, slot, status, hp }
        })
        .collect()
}

fn parse_golden() -> (BTreeMap<String, ScenMeta>, Vec<RunCase>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/movecoverage_batch4c_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "missing batch4c golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_movecoverage_batch4c_golden.js"
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
                //   p1Curse[36] p1Wish[37] p1SubHp[38]  p2Curse[39] p2Wish[40] p2SubHp[41]
                //   p1Future[42] p2Future[43]
                assert_eq!(f.len(), 44, "DEC needs 44 fields (line {ln}), got {}", f.len());
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
                    curse: f[36] == "1",
                    wish: g(37) as u8,
                    sub_hp: g(38),
                    future: g(42) as u8,
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
                    curse: f[39] == "1",
                    wish: g(40) as u8,
                    sub_hp: g(41),
                    future: g(43) as u8,
                };
                let first_mover = f[35].to_string();
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect {
                    request: req, force, choice, seed_after, p1, p2, first_mover,
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

/// Apply the scenario's one-time STATE-only INJECT (status/HP) to the freshly-constructed
/// battle — mirroring the sim harness's post-start inject. NO PRNG.
fn apply_inject(battle: &mut Battle, inject: &[Inject]) {
    let state = battle.state_mut().expect("state");
    for inj in inject {
        if let Some(side) = inj.side {
            let idx = inj.slot.unwrap_or(state.sides[side].active);
            let mon = &mut state.sides[side].pokemon[idx];
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
fn movecoverage_batch4c_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(meta.len() >= 20, "expected >=20 scenarios, got {}", meta.len());
    assert!(cases.len() >= 1500, "expected the per-seed corpus (>=1500 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut hp_assertions = 0usize;
    let mut curse_rows = 0usize; // a decision where a mon is cursed (must stay 0)
    let mut wish_rows = 0usize; // a decision where a Wish is pending (the residual-order scenario)
    let mut future_rows = 0usize; // a decision where a FUTUREMOVE strike is pending
    let mut sub_rows = 0usize; // a decision where a sub is up
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

            for (idx, (snap, e, curse, wish, sub_hp, future)) in [
                (0usize, (&rec.active[0], &exp.p1, rec.curse[0], rec.wish_pending[0], rec.sub_hp[0], rec.future_pending[0])),
                (1usize, (&rec.active[1], &exp.p2, rec.curse[1], rec.wish_pending[1], rec.sub_hp[1], rec.future_pending[1])),
            ] {
                assert_eq!(
                    species_id(&rec.active_species[idx]), species_id(&e.species),
                    "[{}] dec {} side {} active species mismatch (init_seed {}): got {:?} exp {:?}",
                    case.scen, di, idx, case.init_seed, rec.active_species[idx], e.species
                );
                // HP: the HB damage / SB fire (incl. the weather-halved BP) / the future
                // strike's STORED cast-time snapshot land HERE.
                assert_eq!(
                    snap.hp, e.hp,
                    "[{}] dec {} side {} HP mismatch (init_seed {}): got {} exp {}\n  \
                     an HB hit / SB fire (weather-halved?) / future-strike snapshot amount is wrong. FIX THE MODEL.",
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
                assert_eq!(
                    curse, e.curse,
                    "[{}] dec {} side {} CURSE mismatch (init_seed {}): got {} exp {}",
                    case.scen, di, idx, case.init_seed, curse, e.curse
                );
                if curse {
                    curse_rows += 1;
                }
                assert_eq!(
                    wish, e.wish,
                    "[{}] dec {} side {} WISH-PENDING mismatch (init_seed {}): got {} exp {}",
                    case.scen, di, idx, case.init_seed, wish, e.wish
                );
                if wish > 0 {
                    wish_rows += 1;
                }
                // FUTURE-PENDING: the futuremove slot condition's 3 → 2 → 1 → resolve
                // countdown (a wrong duration model / a double-cast that wrongly landed /
                // a resolve that wrongly skipped desyncs HERE).
                assert_eq!(
                    future, e.future,
                    "[{}] dec {} side {} FUTURE-PENDING mismatch (init_seed {}): got {} exp {}\n  \
                     the futuremove duration countdown / double-cast fail / resolve timing is wrong. FIX THE MODEL.",
                    case.scen, di, idx, case.init_seed, future, e.future
                );
                if future > 0 {
                    future_rows += 1;
                }
                assert_eq!(
                    sub_hp, e.sub_hp,
                    "[{}] dec {} side {} SUB-HP mismatch (init_seed {}): got {} exp {}",
                    case.scen, di, idx, case.init_seed, sub_hp, e.sub_hp
                );
                if sub_hp > 0 {
                    sub_rows += 1;
                }

                if !e.fainted {
                    assert!(
                        status_variant_eq(snap.status, e.status),
                        "[{}] dec {} side {} STATUS mismatch (init_seed {}): got {:?} exp {:?}",
                        case.scen, di, idx, case.init_seed, snap.status, e.status
                    );
                    assert_eq!(
                        status_stage(snap.status), e.stage,
                        "[{}] dec {} side {} STATUS-COUNTER mismatch (init_seed {}): got {} exp {}\n  \
                         (a slp counter that wrongly decremented on a RECHARGE-locked turn lands here)",
                        case.scen, di, idx, case.init_seed, status_stage(snap.status), e.stage
                    );
                    assert_eq!(
                        &snap.boosts[0..5], &e.boosts[..],
                        "[{}] dec {} side {} BOOST mismatch (init_seed {}): got {:?} exp {:?}",
                        case.scen, di, idx, case.init_seed, &snap.boosts[0..5], e.boosts
                    );
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
                        "[{}] dec {} FIRST-MOVER mismatch (init_seed {}): got {:?} exp {:?}\n  \
                         (a recharge cant sorts at the user's NORMAL speed-order position — priority 0)",
                        case.scen, di, case.init_seed, rec.first_mover, sim_first
                    );
                }
            }

            // --- PER-DECISION SEED PARITY (the draw-order+count proof). The RECHARGE turn
            //     draws ZERO for the locked user (incl. NO status roll); the SB CHARGE turn
            //     draws ZERO move draws while the FIRE draws acc+crit+dmg (and the sun skip
            //     3 like a normal move); the future CAST draws exactly ONE random(16), the
            //     idle turns zero, the RESOLVE one accuracy roll (+ the two landed-resolve
            //     tie-only Updates); the residual duration/order-11 handlers shift the
            //     tie-shuffle COUNT on the mirror boards. A stray/missing/mis-ordered draw
            //     desyncs HERE. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 a batch-4c draw is wrong — the recharge-locked turn's ZERO draws, Solar Beam's \
                 charge/fire draw split (or sun skip), or the future move's cast/idle/resolve \
                 draw model. FIX THE DRAW MODEL.",
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

    // Coverage floors: each mechanic must actually realize across the corpus. The
    // recharge/charge/fire/cast/resolve branch realization is enforced by the GENERATOR's
    // require gates; here we pin the STATE columns' exercise.
    assert!(seed_assertions >= 10000, "expected the per-decision seed corpus (>=10000), got {seed_assertions}");
    assert!(hp_assertions >= 10000, "expected per-decision HP assertions (>=10000), got {hp_assertions}");
    assert_eq!(curse_rows, 0, "BATCH 4c must never set the curse volatile, got {curse_rows}");
    assert!(wish_rows >= 50, "expected wish-pending rows (>=50 — the residual-order scenario), got {wish_rows}");
    assert!(future_rows >= 500, "expected FUTURE-PENDING rows (>=500), got {future_rows}");
    assert!(sub_rows >= 50, "expected substitute-up rows (>=50 — HB/SB/DD into a sub), got {sub_rows}");
    assert!(win_runs >= 40, "expected real game-end WIN runs (>=40), got {win_runs}");

    eprintln!(
        "movecoverage batch4c golden: {} runs, {dec_assertions} STATE rows, {seed_assertions} seed assertions, \
         {hp_assertions} HP assertions ({curse_rows} cursed rows, {wish_rows} wish-pending rows, {future_rows} \
         future-pending rows, {sub_rows} sub-up rows), {win_runs} wins, {tie_runs} ties",
        cases.len()
    );
}
