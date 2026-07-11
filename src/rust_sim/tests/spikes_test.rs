//! SPIKES (entry hazard) full-battle tests — the per-seed PER-DECISION
//! STATE(+HP+STATUS+SPIKES-LAYERS)+SEED+winner differential that proves the NEW Spikes
//! mechanic matches Showdown EXACTLY, sustained to GAME-END:
//!
//!   **Spikes** — the gen-3 ENTRY HAZARD (the first SIDE CONDITION: a per-side persistent
//!   layer count). DEFERRED (excluded / fail-loud in the engine): Toxic Spikes + Stealth
//!   Rock (NOT gen3), Rapid Spin (the hazard-clear move). Spikes is the only gen-3 entry
//!   hazard. The model (verified bit-for-bit vs the omniscient sim's PRNG probe —
//!   `harness/probe_spikes_rng.js`):
//!
//!   THE SPIKES MOVE (`sideCondition:'spikes'`, `target:'foeSide'`):
//!     * NEVER-MISS → NO accuracy draw. Increments the CASTER's FOE side's `spikes` layer
//!       count by 1, CAPPED at 3 (a Spikes at 3 FAILS, `-fail`). DRAW-FREE both ways;
//!       `landed` FALSE (no in-tryMoveHit Update). So a Spikes-vs-move turn draws ONLY the
//!       existing action-order/eachEvent shuffles.
//!
//!   THE SWITCH-IN DAMAGE (the gen-3 `runSwitch`'s `runEvent('EntryHazard')`, gen4-
//!   inherited; ORDER: EntryHazard → SwitchIn → `if (!hp) return` → ability Start):
//!     * GROUNDED-ONLY: a Flying-type / Levitate entrant takes ZERO.
//!     * Amount: 1 layer `max(floor(maxhp/8),1)`, 2 layers `max(floor(maxhp/6),1)`, 3
//!       layers `max(floor(maxhp/4),1)`. DRAW-FREE. A Spikes hit that KOs the entrant
//!       faints it → forces ANOTHER replacement (which ALSO takes Spikes); no Quick Claw.
//!
//!   `spikes_golden_matches_showdown` — the DIFFERENTIAL gate. For each (scenario, seed)
//!   in `harness/gen_spikes_golden.js`'s golden (FORMAT gen3customgame), seed a
//!   `BattleState` at the sim's PRNG state at the first decision (`init_seed`), run
//!   `run_full_battle(script)` WITHOUT re-seeding, and assert per DECISION BOUNDARY: (a)
//!   each side's post-decision active (species/HP/maxhp/fainted/STATUS + the sleep/Toxic
//!   inner counter) + boosts + confusion + pokemon_left + THE SPIKES LAYERS (per side) +
//!   request kind + first mover; AND (b) the post-decision PRNG seed == the sim's
//!   `seed_after`. PLUS the final WINNER. A HP mismatch on a switch-in catches a wrong
//!   spikes amount / a missed grounded gate (a Flying/Levitate entry must take ZERO; a
//!   grounded entry must take the exact floor); a SPIKES-LAYERS mismatch catches a wrong
//!   lay/cap; a SEED mismatch catches a wrong draw model (the Spikes move / the switch-in
//!   damage must each be draw-free).
//!
//! The golden EXTENDS the protect/recovery TAB format with a 2-col spikes-layers tail
//! (p1Spikes, p2Spikes) → DEC has 49 fields.

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
    /// The sim-reported `spikesDamage` flag (a grounded switch-in took spikes this
    /// decision). A floor counter only (the actual chip is asserted via HP).
    spikes_damage: bool,
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
    /// The gen-3 SPIKES layer count on this side (`side.sideConditions.spikes.layers`,
    /// 0 = absent) — the PRIMARY hazard signal: it proves the lay/stack (0→1→2→3), the cap
    /// (stays 3 on a 4th Spikes), and the PERSISTENCE across switches.
    spikes: u8,
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
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/spikes_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing spikes golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_spikes_golden.js")
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
                //   p2 out(...)[41..46)  spikesDamage[46]  p1Spikes[47] p2Spikes[48]
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
                    spikes: g(47) as u8,
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
                    spikes: g(48) as u8,
                };
                let first_mover = f[35].to_string();
                let spikes_damage = f[46] == "1";
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect {
                    request: req, force, choice, seed_after, p1, p2, first_mover, spikes_damage,
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
fn spikes_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(meta.len() >= 5, "expected >=5 scenarios, got {}", meta.len());
    assert!(cases.len() >= 350, "expected the per-seed corpus (>=350 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut hp_assertions = 0usize;
    let mut spikes_assertions = 0usize; // per-decision per-side spikes-layer assertions
    let mut spikes_damage_rows = 0usize; // a grounded switch-in took spikes (the sim flag)
    let mut spikes_up_rows = 0usize; // a side has >=1 spikes layer
    let mut spikes_max_rows = 0usize; // a side has 3 spikes layers (the cap)
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

            for (idx, (snap, e, sp, spikes)) in [
                (0usize, (&rec.active[0], &exp.p1, &rec.active_species[0], rec.spikes[0])),
                (1usize, (&rec.active[1], &exp.p2, &rec.active_species[1], rec.spikes[1])),
            ] {
                assert_eq!(
                    species_id(sp), species_id(&e.species),
                    "[{}] dec {} side {} active species mismatch (init_seed {}): got {:?} exp {:?}",
                    case.scen, di, idx, case.init_seed, sp, e.species
                );
                // HP: a grounded switch-in must take the EXACT spikes chip; a Flying/Levitate
                // entrant must take ZERO. A wrong amount or a missed grounded gate diverges HERE.
                assert_eq!(
                    snap.hp, e.hp,
                    "[{}] dec {} side {} HP mismatch (init_seed {}): got {} exp {}\n  \
                     a spikes switch-in chip was wrong (a grounded entry must take the exact \
                     floor [maxhp/8, /6, /4]; a Flying/Levitate entry must take ZERO). FIX THE \
                     SPIKES DAMAGE MODEL, do not loosen.",
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

                // THE PRIMARY SPIKES SIGNAL: the per-side layer count. Proves the lay/stack
                // (0→1→2→3), the cap (stays 3), and the PERSISTENCE across switches. The
                // spikes layers are a SIDE condition, so they're asserted regardless of which
                // mon is active (and even if the active fainted this decision).
                assert_eq!(
                    spikes, e.spikes,
                    "[{}] dec {} side {} SPIKES-LAYERS mismatch (init_seed {}): got {} exp {}\n  \
                     the gen-3 spikes layer count must climb 0→1→2→3 (cap 3; a 4th Spikes \
                     FAILS) and PERSIST across switches. FIX THE SPIKES STATE MODEL, do not loosen.",
                    case.scen, di, idx, case.init_seed, spikes, e.spikes
                );
                spikes_assertions += 1;
                if spikes >= 1 {
                    spikes_up_rows += 1;
                }
                if spikes == 3 {
                    spikes_max_rows += 1;
                }

                if !e.fainted {
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

            // --- PER-DECISION SEED PARITY (the draw-order+count proof). The Spikes MOVE is
            //     never-miss + draw-free (it only bumps the side-condition layers; a Spikes-
            //     at-max FAILS draw-free). The switch-in spikes damage is ALSO draw-free (the
            //     deterministic this.damage; the nested runEvent('Damage') has no drawing
            //     handler for the modeled abilities). So a Spikes turn / a spikes switch-in
            //     adds NO draw beyond the existing action-order/eachEvent shuffles. A wrong
            //     draw model (the move or the hazard wrongly drawing) desyncs the LCG HERE. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 the spikes draw model is wrong (the Spikes move or the switch-in hazard \
                 wrongly drew/skipped a PRNG call). Both must be DRAW-FREE. FIX THE DRAW \
                 MODEL, do not loosen.",
                case.scen, di, case.init_seed, rec.seed_after, exp.seed_after
            );
            seed_assertions += 1;
            dec_assertions += 1;

            if exp.spikes_damage {
                spikes_damage_rows += 1;
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

    // Coverage floors: every spikes branch must actually realize across the corpus.
    assert!(seed_assertions >= 1500, "expected the per-decision seed corpus (>=1500), got {seed_assertions}");
    assert!(hp_assertions >= 1500, "expected per-decision HP assertions (>=1500), got {hp_assertions}");
    assert!(spikes_assertions >= 1500, "expected per-decision spikes-layer assertions (>=1500), got {spikes_assertions}");
    assert!(spikes_damage_rows >= 200, "expected spikes switch-in DAMAGE rows (>=200), got {spikes_damage_rows}");
    assert!(spikes_up_rows >= 200, "expected spikes-up rows (>=200), got {spikes_up_rows}");
    assert!(spikes_max_rows >= 50, "expected spikes-at-cap (3 layers) rows (>=50), got {spikes_max_rows}");
    assert!(win_runs >= 100, "expected real game-end WIN runs (>=100), got {win_runs}");

    eprintln!(
        "spikes golden: {} runs, {dec_assertions} STATE rows, {seed_assertions} seed assertions, \
         {hp_assertions} HP assertions, {spikes_assertions} spikes-layer assertions \
         ({spikes_damage_rows} switch-in-damage decisions, {spikes_up_rows} spikes-up rows, {spikes_max_rows} at-cap), \
         {win_runs} wins, {tie_runs} ties",
        cases.len()
    );
}
