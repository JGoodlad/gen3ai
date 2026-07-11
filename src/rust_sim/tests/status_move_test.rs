//! Standalone STATUS-MOVE full-battle tests — the per-seed PER-DECISION
//! STATE(+STATUS+sleep/Toxic counter)+SEED differential that proves the NEW
//! status-move EXECUTION PATH matches Showdown EXACTLY, sustained to GAME-END:
//!
//!   The standalone status-inflicting moves (category Status, bp 0): Thunder Wave /
//!   Stun Spore / Glare [par], Toxic [tox], Poison Powder / Poison Gas [psn],
//!   Will-O-Wisp [brn], Spore / Sleep Powder / Hypnosis / Sing / Lovely Kiss / Grass
//!   Whistle [slp]. Their draw model (verified bit-for-bit vs `data/mods/gen3/
//!   scripts.ts::tryMoveHit`):
//!     1. MOVE-TYPE IMMUNITY (DRAW-FREE) — only Thunder Wave (Electric→Ground) +
//!        Glare (Normal→Ghost) check it (`ignoreImmunity:false`); all others ignore it.
//!     2. ACCURACY `randomChance(acc,100)` — ALWAYS drawn (unless never_miss), even on
//!        a type-immune target (gen3 draws accuracy THEN reports `-immune`).
//!     3. APPLY via try_set_status (already-statused / status-type immunity / ability
//!        immunity / SLEEP CLAUSE gates) — SLEEP draws ONE `random(2,6)` duration,
//!        TOXIC starts at stage 0 (NO draw; the residual ramps it to 1). NO crit/damage/secondary, never fires the
//!        in-`tryMoveHit` Update shuffle.
//!
//!   `status_move_golden_matches_showdown` — the DIFFERENTIAL gate. For each
//!   (scenario, seed) in `harness/gen_status_move_golden.js`'s golden (FORMAT gen3ou,
//!   so the SLEEP CLAUSE MOD is active), seed a `BattleState` at the sim's PRNG state
//!   at the first decision (`init_seed`), run `run_full_battle(script)` WITHOUT
//!   re-seeding, and assert per DECISION BOUNDARY: (a) each side's post-decision active
//!   (species/hp/maxhp/fainted/STATUS + the sleep/Toxic inner counter) + boosts +
//!   confusion + pokemon_left + request kind + first mover; AND (b) the post-decision
//!   PRNG seed == the sim's `seed_after` — the EXACT cross-decision draw-order+count
//!   proof to game-end (the new accuracy draw + the sleep random(2,6) must be in the
//!   exact place/count). PLUS the final WINNER.

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
    /// RNG outcomes the sim reported (floor counters only): per side
    /// `[fullpara, wake, thaw, selfhit, flinch]`, then status-landed.
    out_p1: [bool; 5],
    out_p2: [bool; 5],
    status_landed: bool,
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
    /// The sleep/Toxic inner counter (`slp` = remaining turns, `tox` = stage); 0 else.
    stage: u8,
    left: usize,
    boosts: [i8; 5],
    confusion: u8,
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

/// The inner counter for `slp` / `tox` (else 0). Asserted SEPARATELY from the variant
/// so a wrong sleep duration (the `random(2,6)`) or a wrong Toxic stage diverges here.
fn status_stage(s: Option<Status>) -> u8 {
    match s {
        Some(Status::Sleep(n)) => n,
        Some(Status::Toxic(n)) => n,
        _ => 0,
    }
}

fn parse_golden() -> (BTreeMap<String, ScenMeta>, Vec<RunCase>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/status_move_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing status-move golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_status_move_golden.js")
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
                //   p1(species hp max fnt status stage left atk def spa spd spe confusion)[9..22)
                //   p2(...)[22..35)  first[35]
                //   p1 out(fullpara wake thaw selfhit flinch)[36..41)
                //   p2 out(...)[41..46)  statusLanded[46]
                assert_eq!(f.len(), 47, "DEC needs 47 fields (line {ln}), got {}", f.len());
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
                };
                let first_mover = f[35].to_string();
                let b = |i: usize| f[i] == "1";
                let out_p1 = [b(36), b(37), b(38), b(39), b(40)];
                let out_p2 = [b(41), b(42), b(43), b(44), b(45)];
                let status_landed = b(46);
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect {
                    request: req, force, choice, seed_after, p1, p2, first_mover,
                    out_p1, out_p2, status_landed,
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
        // gen3ou → Sleep Clause Mod ACTIVE (the golden's battle format). The engine
        // derives `sleep_clause` from this id.
        format_id: "gen3ou".to_string(),
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
fn status_move_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(meta.len() >= 8, "expected >=8 scenarios, got {}", meta.len());
    assert!(cases.len() >= 400, "expected the per-seed corpus (>=400 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut status_assertions = 0usize;
    let mut stage_assertions = 0usize; // per-live-mon sleep/Toxic counter assertions
    let mut statused_dec_rows = 0usize;
    let mut status_landed_rows = 0usize;
    let mut sleep_rows = 0usize; // a live mon is asleep (the random(2,6) counter matched)
    let mut tox_ramp_rows = 0usize; // a live mon is badly-poisoned with stage >= 2
    let mut par_rows = 0usize;
    let mut brn_rows = 0usize;
    let mut fullpara_rows = 0usize;
    let mut wake_rows = 0usize;
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

            for (idx, (snap, e, sp)) in [
                (0usize, (&rec.active[0], &exp.p1, &rec.active_species[0])),
                (1usize, (&rec.active[1], &exp.p2, &rec.active_species[1])),
            ] {
                assert_eq!(
                    species_id(sp), species_id(&e.species),
                    "[{}] dec {} side {} active species mismatch (init_seed {}): got {:?} exp {:?}",
                    case.scen, di, idx, case.init_seed, sp, e.species
                );
                assert_eq!(
                    snap.hp, e.hp,
                    "[{}] dec {} side {} HP mismatch (init_seed {}): got {} exp {}",
                    case.scen, di, idx, case.init_seed, snap.hp, e.hp
                );
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
                if !e.fainted {
                    // STATUS variant: a status move that wrongly applied/skipped (immune
                    // gate, Sleep Clause, ability immunity) or an onBeforeMove cure/wake
                    // misfired diverges HERE.
                    assert!(
                        status_variant_eq(snap.status, e.status),
                        "[{}] dec {} side {} STATUS mismatch (init_seed {}): got {:?} exp {:?}\n  \
                         a status move applied/skipped wrongly (immune/Sleep-Clause/ability) \
                         or an onBeforeMove wake/cure misfired.",
                        case.scen, di, idx, case.init_seed, snap.status, e.status
                    );
                    status_assertions += 1;
                    if e.status.is_some() {
                        statused_dec_rows += 1;
                    }

                    // The sleep/Toxic INNER COUNTER (the random(2,6) sleep duration / the
                    // Toxic stage ramp). A wrong/missing sleep duration ALSO desyncs the
                    // seed; a wrong Toxic stage is a draw-free STATE bug. Pinned here.
                    let rust_stage = status_stage(snap.status);
                    assert_eq!(
                        rust_stage, e.stage,
                        "[{}] dec {} side {} STATUS-COUNTER mismatch (init_seed {}): got {} exp {}\n  \
                         the sleep random(2,6) duration or the Toxic stage ramp diverged.",
                        case.scen, di, idx, case.init_seed, rust_stage, e.stage
                    );
                    stage_assertions += 1;
                    if matches!(e.status, Some(Status::Sleep(_))) {
                        sleep_rows += 1;
                    }
                    if matches!(e.status, Some(Status::Toxic(_))) && e.stage >= 2 {
                        tox_ramp_rows += 1;
                    }
                    if matches!(e.status, Some(Status::Paralysis)) {
                        par_rows += 1;
                    }
                    if matches!(e.status, Some(Status::Burn)) {
                        brn_rows += 1;
                    }

                    // BOOSTS + CONFUSION are unaffected by a pure status move but pinned
                    // for completeness (a stray boost/confusion would be a bug).
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
                        "[{}] dec {} first-mover mismatch (init_seed {}): got {:?} exp {:?}",
                        case.scen, di, case.init_seed, rec.first_mover, sim_first
                    );
                }
            }

            // --- PER-DECISION SEED PARITY (the draw-order+count proof). The status
            //     move draws accuracy (always) + a landed sleep's random(2,6); a
            //     type-immune one draws accuracy only; a Sleep-Clause/ability-immune
            //     one draws accuracy then NO random(2,6). One extra/missing/mis-ordered
            //     draw desyncs the LCG here. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 a status-move accuracy draw or the landed-sleep random(2,6) is \
                 mis-ordered/missing/extra (e.g. a sleep random(2,6) drawn on a \
                 Sleep-Clause/immune block, or a status move that wrongly drew crit/\
                 damage). FIX THE DRAW ORDER, do not loosen the assert.",
                case.scen, di, case.init_seed, rec.seed_after, exp.seed_after
            );
            seed_assertions += 1;
            dec_assertions += 1;

            if exp.status_landed {
                status_landed_rows += 1;
            }
            if exp.out_p1[0] || exp.out_p2[0] {
                fullpara_rows += 1;
            }
            if exp.out_p1[1] || exp.out_p2[1] {
                wake_rows += 1;
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
        match case.winner {
            WinTok::P1 | WinTok::P2 => win_runs += 1,
            WinTok::Tie => tie_runs += 1,
            WinTok::None => {}
        }
    }

    // Coverage floors: every status branch must actually realize across the corpus.
    assert!(seed_assertions >= 2000, "expected the per-decision seed corpus (>=2000), got {seed_assertions}");
    assert!(status_assertions >= 2000, "expected per-decision status assertions (>=2000), got {status_assertions}");
    assert!(stage_assertions >= 2000, "expected per-decision sleep/Toxic-counter assertions (>=2000), got {stage_assertions}");
    assert!(statused_dec_rows >= 100, "expected statused-active decision rows (>=100), got {statused_dec_rows}");
    assert!(status_landed_rows >= 50, "expected status-landed rows (>=50), got {status_landed_rows}");
    assert!(sleep_rows >= 50, "expected asleep-active rows (the random(2,6) counter, >=50), got {sleep_rows}");
    assert!(tox_ramp_rows >= 20, "expected Toxic-ramp rows (stage>=2, >=20), got {tox_ramp_rows}");
    assert!(par_rows >= 50, "expected paralyzed-active rows (>=50), got {par_rows}");
    assert!(brn_rows >= 10, "expected burned-active rows (>=10), got {brn_rows}");
    assert!(fullpara_rows >= 20, "expected full-para onBeforeMove rows (>=20), got {fullpara_rows}");
    assert!(wake_rows >= 5, "expected sleep-wake onBeforeMove rows (>=5), got {wake_rows}");
    assert!(win_runs >= 100, "expected real game-end WIN runs (>=100), got {win_runs}");

    eprintln!(
        "status-move golden: {} runs, {dec_assertions} STATE rows, {seed_assertions} seed assertions, \
         {status_assertions} status assertions ({statused_dec_rows} statused), \
         {stage_assertions} counter assertions, {status_landed_rows} status-landed, \
         {sleep_rows} asleep, {tox_ramp_rows} tox-ramp, {par_rows} para, {brn_rows} burn, \
         {fullpara_rows} full-para, {wake_rows} wake, {win_runs} wins, {tie_runs} ties",
        cases.len()
    );
}
