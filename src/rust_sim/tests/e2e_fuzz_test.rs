//! THE CAPSTONE — the end-to-end full-battle fuzz gate.
//!
//! `harness/gen_e2e_fuzz.js` drives the OMNISCIENT in-process Showdown BattleStream
//! over REAL teams (data/teams/, imported + gen3ou-validated + packed) to GAME-END,
//! picking RANDOM legal choices RESTRICTED to mechanics this port models (damaging
//! fixed-BP moves with a modeled secondary shape; else a switch). Every battle in
//! the FILTERED golden additionally has, on BOTH teams, only modeled abilities +
//! items (the harness pre-filters). This test seeds a `BattleState` ONCE at the
//! sim's pre-first-decision PRNG state (`init_seed`) and replays the recorded choice
//! sequence via `run_full_battle` WITHOUT re-seeding, asserting per DECISION
//! BOUNDARY, to game-end:
//!   (a) each side's active species / hp / maxhp / fainted / STATUS + the 5 stat
//!       stages + the confusion counter + pokemon_left + the request kind (move vs
//!       forceSwitch) + (on a move turn) the first mover; AND
//!   (b) the post-decision PRNG seed equals the sim's `seed_after` — the EXACT
//!       cross-decision draw-order+count proof (a single mis-ordered/missing/extra
//!       draw desyncs the LCG); PLUS the final WINNER (or tie).
//!
//! This is the capstone proof: REAL teams, FULL battles, bit-identical, over a
//! corpus the harness sized to the modeled-mechanics-only filter. `filtered_diverged
//! == 0` is the gate — a divergence means either the filter let an unmodeled
//! mechanic in (tighten the harness allow/blocklist) or a real engine bug.
//!
//! Reproducibility: each battle carries its `init_seed` + recorded choice tokens, so
//! a failing battle re-runs deterministically (and the whole golden is reproducible
//! from the harness's `MASTER_SEED`).

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
    boosts: [i8; 5],
    confusion: u8,
    left: usize,
    /// The gen-3 SPIKES layer count on this side (`side.sideConditions.spikes.layers`,
    /// 0 = absent) — the entry-hazard SIDE CONDITION. Asserted per side so a switch-in
    /// onto a spiked side (real Skarmory/Forretress/Cloyster spikers) takes the right chip.
    spikes: u8,
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

fn parse_golden() -> (BTreeMap<String, ScenMeta>, Vec<RunCase>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/e2e_fuzz_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing e2e fuzz golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_e2e_fuzz.js")
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
                // INIT <id> <init_seed> <choose_seed>
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
                // DEC <id> <di> <req> <fp1> <fp2> <cp1> <cp2> <seedAfter>
                //   p1(species hp max fnt status atk def spa spd spe conf left)[9..21)
                //   p2(...)[21..33)  first[33]  p1Spikes[34] p2Spikes[35]
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
                let gi = |i: usize| f[i].parse::<i8>().unwrap_or_else(|e| panic!("bad int f[{i}] (line {ln}): {e}"));
                // p1 block f[9..21): species hp max fnt status atk def spa spd spe conf left
                let p1 = SideExpect {
                    species: f[9].to_string(),
                    hp: g(10),
                    maxhp: g(11),
                    fainted: f[12] == "1",
                    status: parse_status(f[13], 0),
                    boosts: [gi(14), gi(15), gi(16), gi(17), gi(18)],
                    confusion: g(19) as u8,
                    left: g(20) as usize,
                    spikes: g(34) as u8,
                };
                let p2 = SideExpect {
                    species: f[21].to_string(),
                    hp: g(22),
                    maxhp: g(23),
                    fainted: f[24] == "1",
                    status: parse_status(f[25], 0),
                    boosts: [gi(26), gi(27), gi(28), gi(29), gi(30)],
                    confusion: g(31) as u8,
                    left: g(32) as usize,
                    spikes: g(35) as u8,
                };
                let first_mover = f[33].to_string();
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect { request: req, force, choice, seed_after, p1, p2, first_mover });
            }
            "END" => {
                // END <id> <ended> <winner>
                assert_eq!(f.len(), 4, "END needs 4 fields (line {ln})");
                let c = cur.as_mut().unwrap_or_else(|| panic!("END before INIT (line {ln})"));
                c.ended = f[2] == "1";
                c.winner = match f[3] {
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
        // The e2e harness runs battles in gen3customgame (no clauses → NO SetStatus
        // handler-sort shuffle); the Rust format MUST match so `sleep_clause` is OFF
        // (a `gen3ou` label would wrongly enable the gen3ou-only status-apply shuffle).
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

/// Diagnostic (ignored): replay every golden battle and, for the first ~40 that
/// diverge, print the FIRST diverging decision split into SEED-diverged (a draw
/// bug) vs STATE-diverged-while-seed-matches (a pure mechanic bug). Run with:
///   cargo test --test e2e_fuzz_test e2e_diag -- --ignored --nocapture
#[test]
#[ignore]
fn e2e_diag() {
    let d = dex();
    let (meta, cases) = parse_golden();
    let mut shown = 0;
    for case in &cases {
        let m = meta.get(&case.scen).unwrap();
        let opts = opts_for(m, &case.init_seed);
        let mut battle = match Battle::start_with_switchins(&opts, &d) {
            Ok(b) => b,
            Err(e) => { eprintln!("[{}] START-ERR {e}", case.scen); continue; }
        };
        let script = script_from_decisions(case);
        let outcome = battle.state_mut().unwrap().run_full_battle(&script, &d);
        let n = outcome.decisions.len().min(case.decisions.len());
        let mut first: Option<String> = None;
        for di in 0..n {
            let rec = &outcome.decisions[di];
            let exp = &case.decisions[di];
            let seed_ok = rec.seed_after == exp.seed_after;
            let hp_ok = rec.active[0].hp == exp.p1.hp && rec.active[1].hp == exp.p2.hp;
            let sp_ok = species_id(&rec.active_species[0]) == species_id(&exp.p1.species)
                && species_id(&rec.active_species[1]) == species_id(&exp.p2.species);
            let fm_ok = exp.request != ReqTok::Move || {
                let sf = match exp.first_mover.as_str() { "p1" => Some(0), "p2" => Some(1), _ => None };
                sf.is_none() || rec.first_mover == sf
            };
            if !(seed_ok && hp_ok && sp_ok && fm_ok) {
                let kind = if !seed_ok { "SEED" } else if !fm_ok { "FIRSTMOVER" } else if !sp_ok { "SPECIES" } else { "STATE" };
                first = Some(format!(
                    "[{}] dec {di} {kind}: seed_ok={seed_ok} fm_ok={fm_ok} sp_ok={sp_ok} hp_ok={hp_ok} \
                     | rust p1.hp={} p2.hp={} sp=({},{}) fm={:?} | exp p1.hp={} p2.hp={} sp=({},{}) fm={} req={:?}",
                    case.scen,
                    rec.active[0].hp, rec.active[1].hp, rec.active_species[0], rec.active_species[1], rec.first_mover,
                    exp.p1.hp, exp.p2.hp, exp.p1.species, exp.p2.species, exp.first_mover, exp.request,
                ));
                break;
            }
        }
        if outcome.decisions.len() != case.decisions.len() && first.is_none() {
            first = Some(format!("[{}] DECISION-COUNT rust {} golden {} (all common decisions matched)",
                case.scen, outcome.decisions.len(), case.decisions.len()));
        }
        if let Some(msg) = first {
            eprintln!("{msg}");
            shown += 1;
            if shown >= 60 { break; }
        }
    }
    eprintln!("--- diag shown {shown} diverging battles ---");
}

#[test]
fn e2e_fuzz_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(meta.len() >= 50, "expected the filtered corpus (>=50 battles), got {}", meta.len());
    assert!(cases.len() >= 50, "expected the per-battle corpus (>=50), got {}", cases.len());

    let mut filtered_matched = 0usize; // battles bit-for-bit to game-end
    let mut filtered_diverged = 0usize; // MUST be 0 (the engine is bit-for-bit)
    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut status_assertions = 0usize;
    let mut status_present_rows = 0usize; // decisions where a live active carries a major status
    let mut spikes_assertions = 0usize; // per-side spikes-layer assertions
    let mut spikes_present_rows = 0usize; // per-SIDE rows where that side has >=1 spikes layer
    let mut spikes_decisions = 0usize; // DECISIONS where EITHER side has >=1 spikes layer ("uses Spikes")
    let mut substitute_decisions = 0usize; // DECISIONS where EITHER active carries a substitute ("uses Substitute")
    let mut explosion_decisions = 0usize; // DECISIONS where an Explosion/Self-Destruct self-KO fired ("uses Explosion")
    let mut phaze_decisions = 0usize; // DECISIONS where a Roar/Whirlwind phaze DRAG actually fired ("uses Phaze")
    let mut taunt_decisions = 0usize; // DECISIONS where EITHER active is TAUNTED at the boundary ("uses Taunt")
    let mut disable_decisions = 0usize; // DECISIONS where EITHER active has a DISABLED slot at the boundary ("uses Disable")
    let mut trapped_decisions = 0usize; // DECISIONS where EITHER active is TRAPPED at the boundary (`gen3_trapping_v1`)
    let mut boost_assertions = 0usize;
    let mut switch_req_rows = 0usize;
    let mut win_runs = 0usize;
    let mut tie_runs = 0usize;
    let mut faint_carry_runs = 0usize;
    let mut first_diverge_msgs: Vec<String> = Vec::new();

    for case in &cases {
        let m = meta.get(&case.scen).unwrap_or_else(|| panic!("no meta for {}", case.scen));
        assert!(!case.decisions.is_empty(), "[{}] empty run", case.scen);

        let opts = opts_for(m, &case.init_seed);
        let mut battle = match Battle::start_with_switchins(&opts, &d) {
            Ok(b) => b,
            Err(e) => {
                filtered_diverged += 1;
                first_diverge_msgs.push(format!("[{}] start failed: {e}", case.scen));
                continue;
            }
        };

        // The constructed prng must equal the sim's init seed (switch-ins draw-free).
        if battle.state().unwrap().prng_seed() != case.init_seed {
            filtered_diverged += 1;
            first_diverge_msgs.push(format!(
                "[{}] init prng seed mismatch: got {} exp {}",
                case.scen,
                battle.state().unwrap().prng_seed(),
                case.init_seed
            ));
            continue;
        }

        let script = script_from_decisions(case);
        let outcome = battle.state_mut().unwrap().run_full_battle(&script, &d);

        // A divergence in this battle: count it and record the first message, but do
        // NOT abort — we want the full gate tally. (The asserts below force a hard
        // fail if filtered_diverged > 0 at the end.)
        let mut diverged = false;
        macro_rules! diverge {
            ($($arg:tt)*) => {{
                if !diverged {
                    first_diverge_msgs.push(format!($($arg)*));
                }
                diverged = true;
            }};
        }

        if outcome.decisions.len() != case.decisions.len() {
            diverge!(
                "[{}] decision count mismatch (init_seed {}): rust {} vs golden {}",
                case.scen, case.init_seed, outcome.decisions.len(), case.decisions.len()
            );
        }

        let mut saw_faint = false;
        for (di, (rec, exp)) in outcome.decisions.iter().zip(case.decisions.iter()).enumerate() {
            if !req_eq(&rec.request, exp.request, exp.force) {
                diverge!(
                    "[{}] dec {} request mismatch (init_seed {}): got {:?} exp {:?} force {:?}",
                    case.scen, di, case.init_seed, rec.request, exp.request, exp.force
                );
            }
            if exp.request == ReqTok::Switch {
                switch_req_rows += 1;
                saw_faint = true;
            }

            for (idx, (snap, e, sp)) in [
                (0usize, (&rec.active[0], &exp.p1, &rec.active_species[0])),
                (1usize, (&rec.active[1], &exp.p2, &rec.active_species[1])),
            ] {
                if species_id(sp) != species_id(&e.species) {
                    diverge!(
                        "[{}] dec {} side {} active species mismatch (init_seed {}): got {:?} exp {:?}",
                        case.scen, di, idx, case.init_seed, sp, e.species
                    );
                }
                if snap.hp != e.hp {
                    diverge!(
                        "[{}] dec {} side {} HP mismatch (init_seed {}): got {} exp {}",
                        case.scen, di, idx, case.init_seed, snap.hp, e.hp
                    );
                }
                if snap.maxhp != e.maxhp {
                    diverge!("[{}] dec {} side {} maxhp mismatch (init_seed {})", case.scen, di, idx, case.init_seed);
                }
                if snap.fainted != e.fainted {
                    diverge!(
                        "[{}] dec {} side {} fainted mismatch (init_seed {}): got {} exp {}",
                        case.scen, di, idx, case.init_seed, snap.fainted, e.fainted
                    );
                }
                // SPIKES layers (the entry-hazard SIDE CONDITION) — asserted per side,
                // regardless of faint (it's a side state, not the active mon's). A wrong
                // lay/cap diverges here; a wrong switch-in chip diverges via the HP above.
                if rec.spikes[idx] != e.spikes {
                    diverge!(
                        "[{}] dec {} side {} SPIKES-LAYERS mismatch (init_seed {}): got {} exp {}",
                        case.scen, di, idx, case.init_seed, rec.spikes[idx], e.spikes
                    );
                }
                spikes_assertions += 1;
                if e.spikes >= 1 {
                    spikes_present_rows += 1;
                }
                if !e.fainted {
                    if !status_variant_eq(snap.status, e.status) {
                        diverge!(
                            "[{}] dec {} side {} STATUS mismatch (init_seed {}): got {:?} exp {:?}",
                            case.scen, di, idx, case.init_seed, snap.status, e.status
                        );
                    }
                    status_assertions += 1;
                    if e.status.is_some() {
                        status_present_rows += 1;
                    }
                    if snap.boosts[0..5] != e.boosts[..] {
                        diverge!(
                            "[{}] dec {} side {} BOOST mismatch (init_seed {}): got {:?} exp {:?}",
                            case.scen, di, idx, case.init_seed, &snap.boosts[0..5], e.boosts
                        );
                    }
                    boost_assertions += 1;
                    let rust_conf = snap.confusion.unwrap_or(0);
                    if rust_conf != e.confusion {
                        diverge!(
                            "[{}] dec {} side {} CONFUSION mismatch (init_seed {}): got {} exp {}",
                            case.scen, di, idx, case.init_seed, rust_conf, e.confusion
                        );
                    }
                }
            }
            if rec.pokemon_left[0] != exp.p1.left {
                diverge!("[{}] dec {} p1 pokemon_left mismatch: got {} exp {}", case.scen, di, rec.pokemon_left[0], exp.p1.left);
            }
            if rec.pokemon_left[1] != exp.p2.left {
                diverge!("[{}] dec {} p2 pokemon_left mismatch: got {} exp {}", case.scen, di, rec.pokemon_left[1], exp.p2.left);
            }
            // A DECISION that "uses Spikes" = the entry hazard is in play on EITHER side at
            // this boundary (a Spikes was laid + a grounded entrant takes the chip on a
            // switch into it). Counted from the golden's per-side spikes columns.
            if exp.p1.spikes >= 1 || exp.p2.spikes >= 1 {
                spikes_decisions += 1;
            }
            // A DECISION that "uses Substitute" = either active carries a SUBSTITUTE volatile
            // at this boundary (a Substitute was created + is ABSORBING). Counted from the
            // Rust snapshot's `substitute` field — the gate replayed it bit-for-bit (the sub
            // absorb already manifests in the asserted active HP + the running seed, so a
            // mis-absorb diverges above; this counter confirms the mechanic is genuinely
            // exercised on real teams now that SUBSTITUTE_E2E_EXCLUDED = false). A sub up on a
            // FAINTED active is impossible (faint clears volatiles), so reading both is safe.
            if rec.active[0].substitute.is_some() || rec.active[1].substitute.is_some() {
                substitute_decisions += 1;
            }
            // A DECISION that "uses Explosion" = an Explosion / Self-Destruct SELF-KO fired during
            // this turn (the acting mon fainted as part of the move). Read from the engine's
            // `DecisionRecord.explosion_self_ko` flag — the mechanic is MOMENTARY (no persistent
            // board state like a substitute), so it can't be recovered from the post-turn snapshot;
            // the flag is a coverage/diagnostic signal ONLY (the self-KO is applied via the normal
            // faint machinery, already asserted via the fainted flag + pokemon_left + the seed).
            if rec.explosion_self_ko {
                explosion_decisions += 1;
            }
            // A DECISION that "uses Phaze" = a Roar / Whirlwind DRAG fired during this turn (the
            // `sample` ran + a foe mon was dragged in). Read from the engine's
            // `DecisionRecord.phaze_drag` flag — the drag is MOMENTARY (the dragged mon is just the
            // new active, indistinguishable from a voluntary switch on the post-turn snapshot), so
            // it can't be recovered from the board; the flag is a coverage/diagnostic signal ONLY
            // (the drag is applied via the normal switch machinery, already asserted via the active
            // species + spikes + seed). A Protect-blocked / no-bench phaze does NOT set it, so this
            // counts only drags that genuinely exercised the `sample` path (the multi-phaze desync
            // this floor now guards).
            if rec.phaze_drag {
                phaze_decisions += 1;
            }
            // A DECISION that "uses Taunt" / "uses Disable" = either active carries the
            // selection-restriction volatile at this boundary (`gen3_taunt_disable_v1`).
            // Counted from the Rust snapshot's `taunted` / `disabled_slot` fields — the gate
            // replayed the whole battle bit-for-bit (a wrong taunt/disable duration or a
            // missed cant already desyncs the seed/state asserts above; these counters
            // confirm the mechanic is genuinely exercised on real teams). NOTE the honest
            // coverage state: several sample teams carry TAUNT (real coverage expected); NO
            // gen3ou sample team carries DISABLE, so disable_decisions is expected 0 — the
            // disable mechanic is proven by its DEDICATED golden (taunt_disable_test.rs, 720
            // runs) + the TD1-TD4 regression pins, not the e2e (the leech-seed situation).
            if rec.active[0].taunted || rec.active[1].taunted {
                taunt_decisions += 1;
            }
            if rec.active[0].disabled_slot >= 0 || rec.active[1].disabled_slot >= 0 {
                disable_decisions += 1;
            }
            // A DECISION that "involves a TRAPPED mon" = either active is switch-trapped by the
            // foe's Arena Trap / Magnet Pull at this move boundary (`gen3_trapping_v1`,
            // `DecisionRecord.trapped` — the port's live `is_trapped`, == the sim's endTurn-cached
            // `pokemon.trapped` at every move boundary). The generator's voluntary-switch picker
            // respected the sim's flag while capturing, so every recorded choice by a trapped mon
            // is a MOVE — the port's own `is_trapped` gate then accepts exactly those (a
            // wrongly-trapping port would SKIP a recorded free switch → decision-count mismatch;
            // the mirror tie-shuffle draws are seed-asserted above).
            if matches!(rec.request, RequestKind::Move) && (rec.trapped[0] || rec.trapped[1]) {
                trapped_decisions += 1;
            }

            if exp.request == ReqTok::Move {
                let sim_first: Option<usize> = match exp.first_mover.as_str() {
                    "p1" => Some(0),
                    "p2" => Some(1),
                    _ => None,
                };
                if sim_first.is_some() && rec.first_mover != sim_first {
                    diverge!(
                        "[{}] dec {} first-mover mismatch (init_seed {}): got {:?} exp {:?}",
                        case.scen, di, case.init_seed, rec.first_mover, sim_first
                    );
                }
            }

            if rec.seed_after != exp.seed_after {
                diverge!(
                    "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                     a draw is mis-ordered/missing/extra — the filter let an unmodeled mechanic in \
                     (tighten the harness allow/blocklist) OR a real engine bug.",
                    case.scen, di, case.init_seed, rec.seed_after, exp.seed_after
                );
            }
            seed_assertions += 1;
            dec_assertions += 1;

            // HEADLINE HONESTY: the assertion tallies are CLEAN-ONLY — once a battle
            // diverges, its post-desync rows are meaningless (every later state cascades
            // off the wrong seed), so stop counting this battle's decisions at the first
            // divergence. The engine is bit-for-bit (filtered_diverged == 0), so this
            // never trips today; it keeps the headline numbers honest if a regression
            // ever diverges a battle.
            if diverged {
                break;
            }
        }

        if saw_faint && !diverged {
            faint_carry_runs += 1;
        }

        if outcome.ended != case.ended {
            diverge!("[{}] ended mismatch (init_seed {}): got {} exp {}", case.scen, case.init_seed, outcome.ended, case.ended);
        }
        let rust_win = match outcome.winner {
            Some(0) => WinTok::P1,
            Some(1) => WinTok::P2,
            Some(other) => {
                diverge!("[{}] bad winner side {other}", case.scen);
                WinTok::None
            }
            None if outcome.ended => WinTok::Tie,
            None => WinTok::None,
        };
        if rust_win != case.winner {
            diverge!("[{}] WINNER mismatch (init_seed {}): got {:?} exp {:?}", case.scen, case.init_seed, rust_win, case.winner);
        }

        if diverged {
            filtered_diverged += 1;
        } else {
            filtered_matched += 1;
            match case.winner {
                WinTok::P1 | WinTok::P2 => win_runs += 1,
                WinTok::Tie => tie_runs += 1,
                WinTok::None => {}
            }
        }
    }

    // HEADLINE: the assertion tallies are CLEAN-ONLY (the per-decision loop breaks at
    // the first divergence, so no post-desync rows are counted), and the engine is
    // bit-for-bit so `filtered_diverged == 0` over ALL battles.
    eprintln!(
        "E2E CAPSTONE: {} battles, filtered_matched {filtered_matched}, filtered_diverged {filtered_diverged} \
         (STRICT — bit-for-bit, no escape hatch); \
         {dec_assertions} STATE rows, {seed_assertions} seed assertions, {status_assertions} status, \
         {boost_assertions} boost, {spikes_assertions} spikes-layer ({spikes_present_rows} spikes-up side-rows, \
         {spikes_decisions} decisions USE SPIKES, {substitute_decisions} decisions USE SUBSTITUTE, \
         {explosion_decisions} decisions USE EXPLOSION, {phaze_decisions} decisions USE PHAZE, \
         {taunt_decisions} decisions USE TAUNT, {disable_decisions} decisions USE DISABLE, \
         {trapped_decisions} decisions involve a TRAPPED mon), \
         {switch_req_rows} forced-switch reqs, \
         {win_runs} wins, {tie_runs} ties, {faint_carry_runs} past-faint runs",
        cases.len()
    );

    // THE GATE (STRICT): EVERY filtered battle must be bit-for-bit to game-end — state,
    // status, boosts, confusion, running PRNG seed, and winner all match the omniscient
    // sim, with NO per-battle reclassification. The former end-of-turn residual-vs-faint
    // ordering gap under active weather is FIXED in `turn.rs` (per-handler faintMessages
    // + the cached `pokemon.speed` model), so there is no longer a documented engine gap
    // to tolerate. ANY divergence is a hard fail (a real engine bug or the filter
    // letting an unmodeled mechanic in).
    if filtered_diverged != 0 {
        let show: Vec<String> = first_diverge_msgs.into_iter().take(20).collect();
        panic!(
            "E2E CAPSTONE: {filtered_diverged} FILTERED battle(s) DIVERGED (must be 0 — the engine is \
             bit-for-bit).\n  {}\n  \
             FIX: diagnose the engine (run `e2e_diag` / `e2e_trace_one`) — OR, if the filter let an \
             unmodeled mechanic in, tighten the harness allow/blocklist + regenerate the golden.",
            show.join("\n  ")
        );
    }

    // Coverage floors: the gate must genuinely exercise full battles + switches, and the
    // whole filtered corpus must be bit-for-bit clean.
    assert_eq!(
        filtered_diverged, 0,
        "the engine is bit-for-bit: filtered_diverged MUST be 0 over all {} battles",
        cases.len()
    );
    assert!(filtered_matched >= 50, "expected >=50 bit-for-bit battles, got {filtered_matched}");
    assert_eq!(
        filtered_matched, cases.len(),
        "EVERY filtered battle must be bit-for-bit clean (got {filtered_matched} / {})",
        cases.len()
    );
    assert!(seed_assertions >= 1000, "expected the per-decision seed corpus (>=1000), got {seed_assertions}");
    assert!(switch_req_rows >= 20, "expected forced-switch decision rows, got {switch_req_rows}");
    assert!(win_runs >= 30, "expected real game-end WIN runs, got {win_runs}");
    assert!(faint_carry_runs >= 30, "expected runs that continued PAST a faint, got {faint_carry_runs}");
    // Status-move coverage floor: the expanded allow-list must keep EXERCISING status
    // (Thunder Wave/Toxic/Will-O-Wisp/Spore/… land in real battles). A regen that
    // silently dropped status-move usage (e.g. an allow-list regression) would crater
    // this far below the ~5700 status-bearing rows the committed golden carries.
    assert!(
        status_present_rows >= 500,
        "expected the expanded golden to exercise status moves (>=500 status-bearing decision rows), got {status_present_rows}"
    );
    // SPIKES coverage floor: the expanded allow-list must EXERCISE the entry hazard on real
    // teams (Skarmory / Forretress / Cloyster spikers laying Spikes + grounded switch-ins
    // taking the chip). A regen that silently dropped Spikes (an allow-list regression) would
    // crater this. Asserts the gen-3 hazard is genuinely in play across the filtered corpus.
    assert!(
        spikes_decisions >= 50,
        "expected the expanded golden to exercise Spikes (>=50 decisions where the entry hazard is up), got {spikes_decisions}"
    );
    // SUBSTITUTE coverage floor (re-enabled — SUBSTITUTE_E2E_EXCLUDED = false after the
    // switch-tie-weather `eachEvent('WeatherChange')` fix). The committed golden carries ~284
    // substitute-MOVE decisions; the sub is then HELD (absorbing) across many more boundaries.
    // A regen that silently dropped substitute usage (a re-exclusion / allow-list regression)
    // would crater this. Asserts the substitute mechanic is genuinely in play across the
    // filtered corpus, bit-for-bit, on the same battle (e2e_84) that the desync used to crash.
    assert!(
        substitute_decisions >= 50,
        "expected the expanded golden to exercise Substitute (>=50 decisions where a sub is up), got {substitute_decisions}"
    );
    // EXPLOSION coverage floor (RE-ENABLED — `EXPLOSION_E2E_EXCLUDED = false` after fixing the
    // double-faint → double-replacement → SPIKES-CASCADE state bug). The Explosion / Self-Destruct
    // SELF-KO is FULLY modeled bit-for-bit (the DEDICATED `explosion_test.rs` golden + the E1-E4
    // `regression_test.rs` pins) AND is now genuinely EXERCISED across the filtered corpus (the
    // committed golden carries 424 explosion-move decisions, producing the mutual/into-a-KO DOUBLE
    // FAINTS that stress the double-replacement + entry-hazard cascade). The cascade bug is fixed:
    // on a faint, gen-3 singles `cancelAction(getAllActive())` now drops the OTHER side's pending
    // `runSwitch` (turn.rs `cancel_active_actions`), so a cascade entrant's hazard is not
    // re-applied to the foe's already-settled entrant (e2e_9); and the confusion self-hit now folds
    // Choice Band via the full `getDamage` chain (e2e_194). A regen that silently dropped Explosion
    // (a re-exclusion / allow-list regression) would crater this — asserts the self-KO mechanic +
    // its double-faint cascades are in play across the corpus, bit-for-bit. Pinned by
    // `regression_test.rs::double_replacement_cascade_does_not_rechip_the_other_sides_entrant` +
    // `confusion_self_hit_applies_choice_band`. See CLAUDE.md + EDGE_CASES.md.
    assert!(
        explosion_decisions >= 50,
        "expected the expanded golden to exercise Explosion / Self-Destruct (>=50 self-KO decisions), \
         got {explosion_decisions} (did EXPLOSION_E2E_EXCLUDED get re-flipped to true?)"
    );
    // PHAZE coverage floor (RE-ENABLED — `PHAZE_E2E_EXCLUDED = false` after fixing the multi-phaze
    // `sample` draw-POSITION desync). Roar / Whirlwind are FULLY modeled bit-for-bit (the DEDICATED
    // `phaze_test.rs` golden + the P1-P3 `regression_test.rs` pins) AND are now genuinely EXERCISED
    // across the filtered corpus (real Skarmory / Suicune / Zapdos phaze teams Roar / Whirlwind
    // repeatedly across long switch histories). The desync ROOT CAUSE: gen-3 Roar / Whirlwind carry
    // the `protect: 1` flag, so a Protect / Detect BLOCKS the phaze at `TryHit` (after the accuracy
    // roll) → NO `forceSwitchFlag` → NO drag → NO `sample` draw. The port's phaze arm was missing
    // that block, so it dragged a mon (an EXTRA `sample`) into a protected foe the sim left in place,
    // shifting every LATER phaze's `sample` PRNG position (the "same total draw COUNT, wrong `sample`
    // INDEX" bug — pd_1 dec5: Flygon Protects, foe Suicune Roars, the port dragged Aerodactyl while
    // the sim kept Flygon). FIXED in `turn.rs` (the phaze arm now checks `protect_blocks` after the
    // accuracy roll, mirroring the leechseed / standalone-status arms). A regen that silently dropped
    // Phaze (a re-exclusion / allow-list regression) would crater this. Pinned by
    // `regression_test.rs::phaze_blocked_by_protect_draws_no_sample_and_leaves_the_target`. See
    // CLAUDE.md + EDGE_CASES.md.
    assert!(
        phaze_decisions >= 50,
        "expected the expanded golden to exercise Roar / Whirlwind phaze DRAGS (>=50 drag decisions), \
         got {phaze_decisions} (did PHAZE_E2E_EXCLUDED get re-flipped to true?)"
    );
    // TAUNT coverage floor (`gen3_taunt_disable_v1`): several gen3ou sample teams carry Taunt, so
    // admitting it to `MODELED_RESTRICTION_MOVES` yields real taunted boundaries (230 at the current
    // regen). A regen that silently dropped Taunt (an allow-list regression) would crater this.
    // NO floor for DISABLE — no sample team carries it (disable_decisions is expected 0; the honest
    // disclosure — disable is proven by the dedicated taunt_disable golden + the TD1-TD4 pins).
    assert!(
        taunt_decisions >= 50,
        "expected the expanded golden to exercise Taunt (>=50 taunted-boundary decisions), \
         got {taunt_decisions} (did taunt fall out of MODELED_RESTRICTION_MOVES?)"
    );
    // TRAPPING coverage floor (`gen3_trapping_v1`): arenatrap (Dugtrio) + magnetpull (Magneton)
    // are the #3/#4 team-carry gaps in the taxonomy — real gen3ou teams are saturated with them,
    // so admitting them yields real trapped boundaries across the filtered corpus. A regen that
    // silently dropped them (an allow-list regression) would crater this.
    assert!(
        trapped_decisions >= 50,
        "expected the expanded golden to exercise TRAPPING (>=50 trapped-boundary decisions), \
         got {trapped_decisions} (did arenatrap/magnetpull fall out of MODELED_ABILITIES?)"
    );
}

#[test]
#[ignore]
fn e2e_trace_one() {
    let d = dex();
    let (meta, cases) = parse_golden();
    let want = std::env::var("E2E_TRACE").unwrap_or_else(|_| "e2e_90".into());
    for case in &cases {
        if case.scen != want { continue; }
        let m = meta.get(&case.scen).unwrap();
        let opts = opts_for(m, &case.init_seed);
        let mut battle = Battle::start_with_switchins(&opts, &d).unwrap();
        let script = script_from_decisions(case);
        let outcome = battle.state_mut().unwrap().run_full_battle(&script, &d);
        for (di, (rec, exp)) in outcome.decisions.iter().zip(case.decisions.iter()).enumerate() {
            let lo: usize = std::env::var("E2E_LO").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
            let hi: usize = std::env::var("E2E_HI").ok().and_then(|s| s.parse().ok()).unwrap_or(7);
            if di < lo || di > hi { continue; }
            eprintln!("dec{di} req={:?} | RUST p1.hp={} p2.hp={} spk[{},{}] | SIM p1.hp={} p2.hp={} spk[{},{}] | seed_ok={}",
                rec.request, rec.active[0].hp, rec.active[1].hp, rec.spikes[0], rec.spikes[1],
                exp.p1.hp, exp.p2.hp, exp.p1.spikes, exp.p2.spikes, rec.seed_after==exp.seed_after);
        }
        break;
    }
}
