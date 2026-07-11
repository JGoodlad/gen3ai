//! PROTECT / DETECT full-battle tests — the per-seed PER-DECISION
//! STATE(+HP+STATUS+STALL-COUNTER)+SEED+winner differential that proves the NEW
//! Protect/Detect EXECUTION PATH matches Showdown EXACTLY, sustained to GAME-END:
//!
//!   Protect and Detect (identical full-turn protection). DEFERRED (fail-loud in the
//!   engine): Endure (survive-at-1-HP, a different onDamage mechanic), Quick Guard /
//!   Wide Guard / King's Shield / Spiky Shield (gen4+, none in gen3). The draw model
//!   (verified bit-for-bit vs the omniscient sim's PRNG probe + the resolved gen3 `stall`
//!   condition — `harness/probe_protect_rng.js`):
//!
//!   THE STALL / CONSECUTIVE-SUCCESS DRAW (the user's own Protect/Detect):
//!     * Protect/Detect are NEVER-MISS (no accuracy draw) + PRIORITY 3 (resolve BEFORE
//!       the foe's attack so the volatile is up when the foe's move runs).
//!     * `onPrepareHit` → `runEvent('StallMove')` → the `stall` volatile's `onStallMove`
//!       draws `randomChance(1, counter)` ONLY when the volatile is ALREADY present. The
//!       FIRST protect (no volatile) SHORT-CIRCUITS with NO DRAW → success.
//!     * ON SUCCESS the volatile is (re)added: from 0 `onStart`s to counter 2, else
//!       `onRestart`s `counter *= 2`, capped at the gen3 `counterMax` 8 — so consecutive
//!       successes give `0→2→4→8→8→…` (success 100%/50%/25%/12.5%/12.5%, the floor 1/8).
//!     * ON FAILURE the volatile is DELETED → counter 0, no protection that turn.
//!     * The volatile RESETS (counter → 0) the first turn the user does NOT successfully
//!       protect (a different move / a failed protect / a switch-out clearVolatile).
//!
//!   THE MOVE-BLOCK DRAW (a foe move targeting the protected mon):
//!     * In gen3 `tryMoveHit` accuracy is drawn FIRST; only if it PASSES does protect's
//!       `onTryHit` fire (at TryHit, AFTER accuracy, BEFORE the immunity report). So a
//!       BLOCKED foe move DRAWS its accuracy roll (unless never-miss), then is blocked —
//!       NO crit / damage / secondary / status. A miss never reaches the block. Protect
//!       only blocks moves TARGETING the protected mon (a self-target move is never
//!       blocked). DRAW-FREE block.
//!
//!   `protect_move_golden_matches_showdown` — the DIFFERENTIAL gate. For each (scenario,
//!   seed) in `harness/gen_protect_move_golden.js`'s golden (FORMAT gen3customgame), seed a
//!   `BattleState` at the sim's PRNG state at the first decision (`init_seed`), run
//!   `run_full_battle(script)` WITHOUT re-seeding, and assert per DECISION BOUNDARY: (a) each
//!   side's post-decision active (species/HP/maxhp/fainted/STATUS + the sleep/Toxic inner
//!   counter) + boosts + confusion + pokemon_left + THE STALL COUNTER + request kind + first
//!   mover; AND (b) the post-decision PRNG seed == the sim's `seed_after`. PLUS the final
//!   WINNER. A HP mismatch on a protect turn catches a missed/wrongly-applied block (a
//!   blocked move's damage must NOT land); a STALL-COUNTER mismatch catches a wrong
//!   escalation/reset; a SEED mismatch catches a wrong stall draw model (a first-protect
//!   that wrongly drew, a consecutive protect with the wrong denominator, or a blocked move
//!   that wrongly skipped/added its accuracy roll).
//!
//! The golden EXTENDS the recovery/setup TAB format with a 2-col stall-counter tail
//! (p1Stall, p2Stall) → DEC has 49 fields.

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
    /// The sim-reported `block` flag (a foe move was blocked by protect this decision). A
    /// floor counter only (the actual block is asserted via HP + the stall counter).
    block: bool,
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
    /// The gen-3 PROTECT/DETECT stall counter (`volatiles.stall.counter`, 0 = no volatile)
    /// — the PRIMARY protect signal: it proves the consecutive-use denominator escalation
    /// (`0→2→4→8`) AND the reset (a non-protect/switch turn → 0).
    stall: u8,
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

fn parse_golden() -> (BTreeMap<String, ScenMeta>, Vec<RunCase>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/protect_move_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing protect-move golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_protect_move_golden.js")
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
                //   p2 out(...)[41..46)  block[46]  p1Stall[47] p2Stall[48]
                assert_eq!(f.len(), 49, "DEC needs 49 fields (line {ln}), got {}", f.len());
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
                    stall: g(47) as u8,
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
                    stall: g(48) as u8,
                };
                let first_mover = f[35].to_string();
                let block = f[46] == "1";
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect {
                    request: req, force, choice, seed_after, p1, p2, first_mover, block,
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
        // gen3customgame → NO Sleep Clause / SetStatus shuffle (the golden's battle format).
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
fn protect_move_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(meta.len() >= 6, "expected >=6 scenarios, got {}", meta.len());
    assert!(cases.len() >= 400, "expected the per-seed corpus (>=400 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut hp_assertions = 0usize;
    let mut stall_assertions = 0usize; // per-live-mon stall-counter assertions
    let mut block_dec_rows = 0usize; // a foe move was blocked by protect this decision (the sim flag)
    let mut stall_nonzero_rows = 0usize; // a live mon has the stall volatile up (counter > 0)
    let mut stall_escalated_rows = 0usize; // a live mon has counter >= 4 (a CONSECUTIVE success escalated it)
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
                // HP: a BLOCKED foe move's damage must NOT land. A protect turn leaves the
                // protected mon's HP unchanged by the blocked attack (only residuals tick) —
                // a wrongly-unblocked move (or a wrongly-blocked one) diverges HERE.
                assert_eq!(
                    snap.hp, e.hp,
                    "[{}] dec {} side {} HP mismatch (init_seed {}): got {} exp {}\n  \
                     a protect BLOCK failed to stop the foe's damage (or wrongly stopped it) — \
                     a blocked move must apply NO damage. FIX THE BLOCK MODEL, do not loosen.",
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
                if !e.fainted {
                    // STATUS — a protect-blocked status move (Thunder Wave) must NOT set
                    // status; a wrongly-unblocked one diverges here.
                    assert!(
                        status_variant_eq(snap.status, e.status),
                        "[{}] dec {} side {} STATUS mismatch (init_seed {}): got {:?} exp {:?}\n  \
                         a protect-blocked status move wrongly set (or wrongly skipped) status.",
                        case.scen, di, idx, case.init_seed, snap.status, e.status
                    );
                    assert_eq!(
                        status_stage(snap.status), e.stage,
                        "[{}] dec {} side {} STATUS-COUNTER mismatch (init_seed {}): got {} exp {}",
                        case.scen, di, idx, case.init_seed, status_stage(snap.status), e.stage
                    );

                    // THE PRIMARY PROTECT SIGNAL: the stall counter. Proves the consecutive-
                    // use denominator escalation (0→2→4→8→8) AND the reset (non-protect /
                    // switch → 0). A wrong escalation/reset diverges HERE (and the wrong
                    // denominator also desyncs the SEED on the next protect's roll).
                    assert_eq!(
                        snap.protect_counter, e.stall,
                        "[{}] dec {} side {} STALL-COUNTER mismatch (init_seed {}): got {} exp {}\n  \
                         the gen-3 protect stall counter must escalate 0→2→4→8 (cap 8) on \
                         CONSECUTIVE successes and reset to 0 on a non-protect/switch turn (or a \
                         failed roll). FIX THE STALL MODEL, do not loosen.",
                        case.scen, di, idx, case.init_seed, snap.protect_counter, e.stall
                    );
                    stall_assertions += 1;
                    if snap.protect_counter > 0 {
                        stall_nonzero_rows += 1;
                    }
                    if snap.protect_counter >= 4 {
                        stall_escalated_rows += 1;
                    }

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
                        "[{}] dec {} FIRST-MOVER mismatch (init_seed {}): got {:?} exp {:?}",
                        case.scen, di, case.init_seed, rec.first_mover, sim_first
                    );
                }
            }

            // --- PER-DECISION SEED PARITY (the draw-order+count proof). Protect/Detect are
            //     never-miss (no accuracy draw); the FIRST protect draws NOTHING (counter 0
            //     short-circuit); a CONSECUTIVE protect draws exactly one randomChance(1,
            //     counter) (denominator 2/4/8). A BLOCKED foe move draws its accuracy roll
            //     (unless never-miss) but NO crit/damage/secondary. A wrong stall denominator,
            //     a first-protect that wrongly drew, or a blocked move that wrongly skipped/
            //     added its accuracy roll desyncs the LCG HERE. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 the protect stall draw model is wrong (first-protect drew when it should \
                 short-circuit, a consecutive protect used the wrong denominator, or a blocked \
                 foe move wrongly skipped/added its accuracy roll). FIX THE DRAW MODEL, do not loosen.",
                case.scen, di, case.init_seed, rec.seed_after, exp.seed_after
            );
            seed_assertions += 1;
            dec_assertions += 1;

            if exp.block {
                block_dec_rows += 1;
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

    // Coverage floors: every protect branch must actually realize across the corpus.
    assert!(seed_assertions >= 2000, "expected the per-decision seed corpus (>=2000), got {seed_assertions}");
    assert!(hp_assertions >= 2000, "expected per-decision HP assertions (>=2000), got {hp_assertions}");
    assert!(stall_assertions >= 2000, "expected per-decision stall-counter assertions (>=2000), got {stall_assertions}");
    assert!(block_dec_rows >= 200, "expected protect-block decision rows (>=200), got {block_dec_rows}");
    assert!(stall_nonzero_rows >= 200, "expected stall-volatile-up rows (>=200), got {stall_nonzero_rows}");
    assert!(stall_escalated_rows >= 50, "expected escalated stall rows (counter>=4, >=50), got {stall_escalated_rows}");
    assert!(win_runs >= 100, "expected real game-end WIN runs (>=100), got {win_runs}");

    eprintln!(
        "protect-move golden: {} runs, {dec_assertions} STATE rows, {seed_assertions} seed assertions, \
         {hp_assertions} HP assertions, {stall_assertions} stall-counter assertions \
         ({block_dec_rows} block decisions, {stall_nonzero_rows} stall-up rows, {stall_escalated_rows} escalated), \
         {win_runs} wins, {tie_runs} ties",
        cases.len()
    );
}
