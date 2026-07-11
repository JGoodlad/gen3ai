//! PP-tracking + STRUGGLE full-battle tests — the per-seed PER-DECISION
//! STATE(+HP+STATUS+PP)+SEED+winner differential that proves the NEW PP/Struggle
//! EXECUTION PATH matches Showdown EXACTLY, sustained to GAME-END (`gen3_pp_tracking_v1`).
//!
//! The draw model (verified bit-for-bit vs the omniscient sim,
//! `harness/probe_pp_struggle_rng.js`):
//!   1. PP INIT: a moveslot's in-battle PP is `calculatePP(move, 3) = pp * 8 / 5` (the ctor's
//!      default 3 PP-ups), or the raw `pp` for a `noPPBoosts` move.
//!   2. PP DECREMENT: −1 per USE, DRAW-FREE, ONLY when the mon MOVES (a full-para / sleep /
//!      flinch / frozen / confusion-self-hit turn deducts NOTHING). A MISS / an IMMUNE hit
//!      STILL decrement.
//!   3. PRESSURE −2: a move TARGETING a Pressure holder deducts 2 PP, DRAW-FREE.
//!   4. FORCED STRUGGLE: the mon has NO usable move (all slots 0 PP, OR Choice-Band locks it
//!      to a slot that hit 0 PP) → `moveid:'struggle'` is substituted for the scripted `move K`.
//!   5. STRUGGLE: typeless '???' (no STAB, hits Ghosts), BP 50, PHYSICAL, accuracy 100 →
//!      DRAWS accuracy, then crit + damage like a normal move; recoil = `max(floor(dmg/4),1)`
//!      (the gen-3 `recoil:[1,4]` path), DRAW-FREE. Struggle consumes no PP.
//!   6. PP PERSISTS across switch-out (gen-3, no reset).
//!
//! `pp_struggle_golden_matches_showdown` — the DIFFERENTIAL gate. For each (scenario, seed) in
//! `harness/gen_pp_struggle_golden.js`'s golden (FORMAT gen3customgame), seed a `BattleState`
//! at the sim's PRNG state at the first decision (`init_seed`), run `run_full_battle(script)`
//! WITHOUT re-seeding, and assert per DECISION BOUNDARY: (a) each side's post-decision active
//! (species/HP/maxhp/fainted/STATUS + the sleep/Toxic inner counter + the 4 move slots' PP)
//! + pokemon_left + request kind + first mover; AND (b) the post-decision PRNG seed == the
//! sim's `seed_after`. PLUS the final WINNER. A PP mismatch catches a wrong decrement (−1 vs
//! −2 Pressure, a spurious decrement on a can't-move turn, a wrong Struggle trigger — PP is
//! draw-free, so a wrong count is a STATE bug); a SEED mismatch catches a wrong Struggle draw
//! model (a Struggle that mis-drew accuracy / crit / damage, or a mis-applied recoil that
//! perturbed a later draw); an HP mismatch catches a wrong Struggle recoil AMOUNT.

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WinTok {
    P1,
    P2,
    Tie,
    None,
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
    /// The 4 move slots' current PP (`-1` = no such slot) — the PRIMARY new signal.
    pp: [i16; 4],
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
    struggle: bool,
    recoil: bool,
    pressure2: bool,
    immune: bool,
}

#[derive(Debug, Clone)]
struct RunCase {
    scen: String,
    init_seed: String,
    decisions: Vec<DecExpect>,
    ended: bool,
    winner: WinTok,
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

fn parse_status(tok: &str, stage: u16) -> Option<Status> {
    match tok {
        "-" | "fnt" => None,
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

fn parse_golden() -> (BTreeMap<String, ScenMeta>, Vec<RunCase>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/pp_struggle_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing pp-struggle golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_pp_struggle_golden.js")
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
                // DEC id seed req fP1 fP2 cP1 cP2 seedAfter
                //   p1(species hp max fnt status stage left pp0 pp1 pp2 pp3)[9..20)
                //   p2(...)[20..31)  first[31]  struggle[32] recoil[33] pressure2[34] immune[35]
                assert_eq!(f.len(), 36, "DEC needs 36 fields (line {ln}), got {}", f.len());
                let req = match f[3] {
                    "move" => ReqTok::Move,
                    "switch" => ReqTok::Switch,
                    other => panic!("bad request {other:?} (line {ln})"),
                };
                let force = [f[4] == "1", f[5] == "1"];
                let choice = [parse_choice(f[6]), parse_choice(f[7])];
                let seed_after = f[8].to_string();
                let g = |i: usize| f[i].parse::<u16>().unwrap_or_else(|e| panic!("bad num f[{i}] (line {ln}): {e}"));
                let pp_at = |base: usize| -> [i16; 4] {
                    let mut out = [-1i16; 4];
                    for (k, slot) in out.iter_mut().enumerate() {
                        *slot = f[base + k].parse::<i16>().unwrap_or_else(|e| panic!("bad pp f[{}] (line {ln}): {e}", base + k));
                    }
                    out
                };
                let p1 = SideExpect {
                    species: f[9].to_string(),
                    hp: g(10),
                    maxhp: g(11),
                    fainted: f[12] == "1",
                    status: parse_status(f[13], g(14)),
                    stage: g(14) as u8,
                    left: g(15) as usize,
                    pp: pp_at(16),
                };
                let p2 = SideExpect {
                    species: f[20].to_string(),
                    hp: g(21),
                    maxhp: g(22),
                    fainted: f[23] == "1",
                    status: parse_status(f[24], g(25)),
                    stage: g(25) as u8,
                    left: g(26) as usize,
                    pp: pp_at(27),
                };
                let first_mover = f[31].to_string();
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect {
                    request: req,
                    force,
                    choice,
                    seed_after,
                    p1,
                    p2,
                    first_mover,
                    struggle: f[32] == "1",
                    recoil: f[33] == "1",
                    pressure2: f[34] == "1",
                    immune: f[35] == "1",
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

#[test]
fn pp_struggle_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(meta.len() >= 5, "expected >=5 scenarios, got {}", meta.len());
    assert!(cases.len() >= 300, "expected the per-seed corpus (>=300 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut pp_assertions = 0usize; // per-live-mon PP-array assertions (the decrement signal)
    let mut struggle_rows = 0usize; // decisions that forced a Struggle
    let mut recoil_rows = 0usize; // decisions with a Struggle recoil
    let mut pressure2_rows = 0usize; // decisions with a Pressure −2 decrement
    let mut immune_rows = 0usize; // decisions where an immune hit still decremented
    let mut zero_pp_rows = 0usize; // a live mon with a slot at 0 PP (the pre-Struggle state)
    let mut win_runs = 0usize;

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

        let script = script_from_decisions(case);
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
                "[{}] dec {} request mismatch (init_seed {}): got {:?} exp {:?} force {:?}",
                case.scen, di, case.init_seed, rec.request, exp.request, exp.force
            );

            for (idx, (snap, e, sp)) in [
                (0usize, (&rec.active[0], &exp.p1, &rec.active_species[0])),
                (1usize, (&rec.active[1], &exp.p2, &rec.active_species[1])),
            ] {
                assert_eq!(
                    species_id(sp), species_id(&e.species),
                    "[{}] dec {} side {} active species mismatch (init_seed {}): got {:?} exp {:?}",
                    case.scen, di, idx, case.init_seed, sp, e.species
                );
                // HP — a wrong Struggle recoil AMOUNT (a floor(dmg/4) error) diverges here.
                assert_eq!(
                    snap.hp, e.hp,
                    "[{}] dec {} side {} HP mismatch (init_seed {}): got {} exp {}\n  \
                     a Struggle recoil (max(floor(dmg/4),1)) or a hit dealt the wrong amount.",
                    case.scen, di, idx, case.init_seed, snap.hp, e.hp
                );
                assert_eq!(snap.maxhp, e.maxhp, "[{}] dec {} side {} maxhp", case.scen, di, idx);
                assert_eq!(
                    snap.fainted, e.fainted,
                    "[{}] dec {} side {} fainted mismatch (init_seed {}): got {} exp {}",
                    case.scen, di, idx, case.init_seed, snap.fainted, e.fainted
                );

                if !e.fainted {
                    // THE PRIMARY NEW SIGNAL: per-slot PP. A wrong decrement (−1 vs −2 Pressure,
                    // a spurious decrement on a can't-move turn, a wrong Struggle trigger, a PP
                    // reset on switch) diverges here. PP is DRAW-FREE, so a wrong count is a
                    // STATE bug (not a seed bug).
                    assert_eq!(
                        snap.move_pp, e.pp,
                        "[{}] dec {} side {} PP mismatch (init_seed {}): got {:?} exp {:?}\n  \
                         a wrong PP decrement (−1/−2 Pressure), a decrement on a can't-move turn, \
                         a wrong Struggle trigger, or a PP reset on switch. FIX THE PP, not the assert.",
                        case.scen, di, idx, case.init_seed, snap.move_pp, e.pp
                    );
                    pp_assertions += 1;
                    if e.pp.iter().any(|&p| p == 0) {
                        zero_pp_rows += 1;
                    }

                    // STATUS (+ inner counter) — pinned so a Struggle turn / sleep / Toxic is caught.
                    assert!(
                        status_variant_eq(snap.status, e.status),
                        "[{}] dec {} side {} STATUS mismatch (init_seed {}): got {:?} exp {:?}",
                        case.scen, di, idx, case.init_seed, snap.status, e.status
                    );
                    assert_eq!(
                        status_stage(snap.status), e.stage,
                        "[{}] dec {} side {} STATUS-COUNTER mismatch (init_seed {}): got {} exp {}",
                        case.scen, di, idx, case.init_seed, status_stage(snap.status), e.stage
                    );
                }
            }
            assert_eq!(rec.pokemon_left[0], exp.p1.left, "[{}] dec {} p1 left", case.scen, di);
            assert_eq!(rec.pokemon_left[1], exp.p2.left, "[{}] dec {} p2 left", case.scen, di);

            // FIRST-MOVER (pinned for the shared format).
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

            // --- PER-DECISION SEED PARITY (the draw-order+count proof). PP decrement + the
            //     Choice lock + the forced-Struggle substitution are ALL DRAW-FREE. A forced
            //     Struggle draws acc + crit + damage like a normal move (+ Quick Claw when not
            //     the deciding faint); its recoil is draw-free. So the seed change is a function
            //     of the moves' own draws + residuals + the eachEvent tie-shuffles. A Struggle
            //     that mis-drew accuracy/crit/damage, or a mis-applied recoil that perturbed a
            //     later draw, desyncs the LCG here. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 a Struggle mis-drew its acc/crit/damage, or PP tracking wrongly perturbed a draw. \
                 FIX THE DRAW MODEL, do not loosen.",
                case.scen, di, case.init_seed, rec.seed_after, exp.seed_after
            );
            seed_assertions += 1;
            dec_assertions += 1;

            if exp.struggle {
                struggle_rows += 1;
            }
            if exp.recoil {
                recoil_rows += 1;
            }
            if exp.pressure2 {
                pressure2_rows += 1;
            }
            if exp.immune {
                immune_rows += 1;
            }
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
        if matches!(case.winner, WinTok::P1 | WinTok::P2) {
            win_runs += 1;
        }
    }

    // Coverage floors — every PP/Struggle branch must actually realize across the corpus.
    assert!(seed_assertions >= 3000, "expected the per-decision seed corpus (>=3000), got {seed_assertions}");
    assert!(pp_assertions >= 3000, "expected per-decision PP assertions (>=3000), got {pp_assertions}");
    assert!(struggle_rows >= 200, "expected forced-Struggle decisions (>=200), got {struggle_rows}");
    assert!(recoil_rows >= 200, "expected Struggle-recoil decisions (>=200), got {recoil_rows}");
    assert!(pressure2_rows >= 40, "expected Pressure −2 decisions (>=40), got {pressure2_rows}");
    assert!(immune_rows >= 200, "expected immune-decrement decisions (>=200), got {immune_rows}");
    assert!(zero_pp_rows >= 100, "expected 0-PP-slot rows (the pre-Struggle state, >=100), got {zero_pp_rows}");
    assert!(win_runs >= 40, "expected real game-end WIN runs (>=40), got {win_runs}");

    eprintln!(
        "pp-struggle golden: {} runs, {dec_assertions} STATE rows, {seed_assertions} seed assertions, \
         {pp_assertions} PP assertions ({struggle_rows} forced-Struggle, {recoil_rows} recoil, \
         {pressure2_rows} Pressure−2, {immune_rows} immune-decrement, {zero_pp_rows} 0-PP rows), {win_runs} wins",
        cases.len()
    );
}
