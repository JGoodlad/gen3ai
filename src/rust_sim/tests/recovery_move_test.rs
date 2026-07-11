//! Self-heal / RECOVERY MOVE full-battle tests — the per-seed PER-DECISION
//! STATE(+HP+STATUS)+SEED+winner differential that proves the NEW recovery EXECUTION PATH
//! matches Showdown EXACTLY, sustained to GAME-END:
//!
//!   The self-targeting HP-recovery moves (category Status, bp 0, target self, isHeal):
//!   Recover / Soft-Boiled / Slack Off / Milk Drink (heal floor(maxhp/2)), Rest (full heal
//!   + a FIXED Sleep(3) self-sleep + a prior-status cure), and Moonlight / Synthesis /
//!   Morning Sun (WEATHER-conditional heal). Their draw model (verified bit-for-bit vs
//!   `data/moves.ts` + the gen4-inherited `onHit`):
//!     1. ACCURACY — every recovery move is NEVER-MISS → NO accuracy draw.
//!     2. HEAL `this.heal(amount)` on the USER. **DRAW-FREE** (heal() consumes no PRNG).
//!        Amounts (gen3 maxhp == baseMaxhp, integer truncation):
//!          * Recover/Soft-Boiled/Slack Off/Milk Drink: floor(maxhp/2).
//!          * Moonlight/Synthesis/Morning Sun: NONE → floor(maxhp/2); SUN → floor(maxhp*2/3);
//!            SAND/RAIN/HAIL → floor(maxhp/4). (gen4-inherited PLAIN integer, NOT 4096-modify.)
//!        A heal at full HP / heal-0 FAILS (`-fail`), draw-free either way.
//!     3. REST — a FULL heal + a prior-status CURE + a self-sleep that DRAWS one
//!        random(2,6) which is then DISCARDED and overwritten to a FIXED Sleep(3) (gen-3
//!        `slp.onStart` ALWAYS rolls the duration; Rest's `onHit` overwrites `time`). So
//!        Rest is NOT draw-free: in gen3customgame (the golden's format) it draws exactly
//!        the one random(2,6); in gen3ou the SetStatus clause shuffle draws FIRST, then the
//!        random(2,6). The user wakes via the existing sleep counter (3 attempts).
//!     4. `landed` is ALWAYS FALSE — a status moveHit returns `undefined`, so the in-
//!        tryMoveHit `eachEvent('Update')` shuffle is SKIPPED. So a non-Rest recovery move
//!        is DRAW-FREE (the seed is a function of only the OTHER draws; only the user's HP
//!        changes); Rest additionally consumes the discarded random(2,6) above.
//!
//!   `recovery_move_golden_matches_showdown` — the DIFFERENTIAL gate. For each (scenario,
//!   seed) in `harness/gen_recovery_move_golden.js`'s golden (FORMAT gen3customgame), seed a
//!   `BattleState` at the sim's PRNG state at the first decision (`init_seed`), run
//!   `run_full_battle(script)` WITHOUT re-seeding, and assert per DECISION BOUNDARY: (a) each
//!   side's post-decision active (species/HP/maxhp/fainted/STATUS + the sleep/Toxic inner
//!   counter) + boosts + confusion + pokemon_left + request kind + first mover; AND (b) the
//!   post-decision PRNG seed == the sim's `seed_after`. PLUS the final WINNER. A HP mismatch
//!   catches a wrong heal AMOUNT (a 1-HP error desyncs the STATE — heal() is draw-free, so a
//!   wrong amount is a STATE bug, not a seed bug); a STATUS mismatch catches a wrong Rest
//!   sleep/cure; a SEED mismatch catches a wrong draw model (e.g. a Rest that MISSED its
//!   random(2,6), or a recovery move that wrongly drew accuracy).
//!
//! The golden SHARES the setup_move_golden TAB format, so the parser is identical; the
//! recovery primary signal is HP + STATUS (vs setup's boost stages).

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
    /// The sim-reported `healed` flag (a recovery `-heal` line fired this decision). A floor
    /// counter only (the actual HP is asserted in `SideExpect::hp`).
    healed: bool,
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
    /// The sleep/Toxic inner counter — the Rest sleep timer is the PRIMARY status signal
    /// here (3 → 2 → 1 → wake). Asserted on every live mon.
    stage: u8,
    left: usize,
    /// The 5 stat-stage boosts `[atk, def, spa, spd, spe]` (0 throughout the recovery
    /// scenarios — kept for the shared golden format; asserted for completeness).
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

fn status_stage(s: Option<Status>) -> u8 {
    match s {
        Some(Status::Sleep(n)) => n,
        Some(Status::Toxic(n)) => n,
        _ => 0,
    }
}

fn parse_golden() -> (BTreeMap<String, ScenMeta>, Vec<RunCase>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/recovery_move_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing recovery-move golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_recovery_move_golden.js")
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
                //   p2 out(...)[41..46)  healed[46]
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
                // f[36..46) are the per-side onBeforeMove outcome flags (unused — the
                // STATE+SEED assertions pin those; we only read `healed` at f[46]).
                let healed = f[46] == "1";
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect {
                    request: req, force, choice, seed_after, p1, p2, first_mover, healed,
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
        // gen3customgame → NO Sleep Clause / SetStatus shuffle (the golden's battle format;
        // a Rest here draws nothing — the gen3ou SetStatus shuffle is OFF).
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
fn recovery_move_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(meta.len() >= 8, "expected >=8 scenarios, got {}", meta.len());
    assert!(cases.len() >= 500, "expected the per-seed corpus (>=500 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut hp_assertions = 0usize; // per-live-mon HP assertions (the heal-amount signal)
    let mut heal_dec_rows = 0usize; // a recovery `-heal` fired this decision (the sim flag)
    let mut rest_sleep_rows = 0usize; // a live mon is asleep via Rest's Sleep(n)
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
                // THE PRIMARY SIGNAL: HP. A recovery move that healed the WRONG AMOUNT (a
                // 1-HP floor error on Recover/Rest/the weather fractions) diverges HERE.
                // heal() is draw-free, so a wrong amount is a STATE bug (not a seed bug).
                assert_eq!(
                    snap.hp, e.hp,
                    "[{}] dec {} side {} HP mismatch (init_seed {}): got {} exp {}\n  \
                     a recovery move healed the WRONG amount (floor(maxhp/2) / floor(maxhp*2/3) \
                     sun / floor(maxhp/4) sand / Rest full-heal) — a 1-HP error desyncs the STATE. \
                     FIX THE HEAL AMOUNT, do not loosen the assert.",
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
                    // STATUS — the Rest self-sleep is the secondary signal (a wrong Rest
                    // sleep/cure diverges here). The inner sleep counter (3→2→1→wake) too.
                    assert!(
                        status_variant_eq(snap.status, e.status),
                        "[{}] dec {} side {} STATUS mismatch (init_seed {}): got {:?} exp {:?}\n  \
                         a Rest that mis-slept or mis-cured (or a recovery move that wrongly \
                         set status) diverges here.",
                        case.scen, di, idx, case.init_seed, snap.status, e.status
                    );
                    assert_eq!(
                        status_stage(snap.status), e.stage,
                        "[{}] dec {} side {} STATUS-COUNTER mismatch (init_seed {}): got {} exp {}\n  \
                         the Rest sleep timer (3→2→1→wake) or the Toxic stage diverged.",
                        case.scen, di, idx, case.init_seed, status_stage(snap.status), e.stage
                    );
                    if matches!(snap.status, Some(Status::Sleep(_))) {
                        rest_sleep_rows += 1;
                    }

                    // Boosts (0 throughout — pinned so a stray boost is caught).
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

            // FIRST-MOVER (pinned for the shared format; recovery doesn't touch speed but a
            // wrong draw/order would diverge it).
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

            // --- PER-DECISION SEED PARITY (the draw-order+count proof). A non-Rest recovery
            //     move is DRAW-FREE (never-miss → no accuracy draw, heal() no PRNG, no
            //     in-tryMoveHit Update); REST draws exactly one random(2,6) (discarded, then
            //     a FIXED Sleep(3)). So the seed change is a function of those plus the OTHER
            //     draws (the foe's move, residuals, the per-action eachEvent tie-shuffles). A
            //     recovery move that wrongly drew accuracy, a Rest that MISSED its random(2,6)
            //     sleep duration, or a wrong clause-shuffle (gen3customgame draws none beyond
            //     Rest's random(2,6)) desyncs the LCG here. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 a recovery move wrongly drew accuracy, OR a Rest wrongly drew the random(2,6) \
                 sleep duration (it must be a FIXED Sleep(3)), OR a stray SetStatus clause \
                 shuffle fired (gen3customgame draws none). FIX THE DRAW MODEL, do not loosen.",
                case.scen, di, case.init_seed, rec.seed_after, exp.seed_after
            );
            seed_assertions += 1;
            dec_assertions += 1;

            if exp.healed {
                heal_dec_rows += 1;
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

    // Coverage floors: every recovery branch must actually realize across the corpus.
    assert!(seed_assertions >= 2000, "expected the per-decision seed corpus (>=2000), got {seed_assertions}");
    assert!(hp_assertions >= 2000, "expected per-decision HP assertions (>=2000), got {hp_assertions}");
    assert!(heal_dec_rows >= 200, "expected recovery -heal decision rows (>=200), got {heal_dec_rows}");
    assert!(rest_sleep_rows >= 100, "expected Rest-sleep rows (an asleep mon, >=100), got {rest_sleep_rows}");
    assert!(win_runs >= 100, "expected real game-end WIN runs (>=100), got {win_runs}");

    eprintln!(
        "recovery-move golden: {} runs, {dec_assertions} STATE rows, {seed_assertions} seed assertions, \
         {hp_assertions} HP assertions ({heal_dec_rows} -heal decisions, {rest_sleep_rows} Rest-sleep rows), \
         {win_runs} wins, {tie_runs} ties",
        cases.len()
    );
}
