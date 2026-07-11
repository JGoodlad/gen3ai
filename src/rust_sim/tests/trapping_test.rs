//! TRAPPING (Arena Trap / Magnet Pull) full-battle tests — the per-seed PER-DECISION
//! STATE(+per-side TRAPPED)+SEED+winner differential that proves the gen-3
//! SWITCH-legality layer (`gen3_trapping_v1`) matches Showdown EXACTLY, to GAME-END.
//!
//! The mechanics (verified bit-for-bit vs the omniscient sim — the probe is
//! `harness/probe_trapping_rng.js`; do NOT trust intuition, gen-3 trapping has
//! probe-settled surprises):
//!
//!   ARENA TRAP (`arenatrap`): the foe active traps every GROUNDED opposing mon
//!     (Flying-type / Levitate escape; gen-3 grounded == not-Flying && not-Levitate).
//!     A grounded GHOST **IS** trapped — Showdown-gen3 resolves NO `trapped` type
//!     immunity (Ghost `damageTaken.trapped` = undefined in the gen3 dex; the
//!     cartridge gen6+ Ghost escape does not exist here). `onFoeTrapPokemon` (base
//!     data) → ONE handler per TrapPokemon event → the trapped computation is
//!     DRAW-FREE even in the Dugtrio MIRROR (mutual trap, probe: seeds byte-identical
//!     to a Sand Veil control).
//!
//!   MAGNET PULL (`magnetpull`): traps STEEL-type foes — groundedness IRRELEVANT
//!     (Skarmory, Steel/Flying, is trapped). gen3 overrides the handlers to
//!     `onAnyTrapPokemon`/`onAnyMaybeTrapPokemon` (data/mods/gen3/abilities.ts), so a
//!     Magnet Pull holder's OWN handler registers on every trap event too (its body
//!     no-ops on `isAdjacent(self,self)===false` — no self-trap — but it still SORTS):
//!     the speed-TIED MAGNETON MIRROR draws ONE Fisher-Yates tie-shuffle per event per
//!     mon = **4 draws per endTurn** (probe: 11 draws/turn vs the Sturdy control's 7),
//!     and an Arena-Trap-vs-Magnet-Pull cross at equal speed draws 2 (both events on
//!     the Magnet Pull holder). The draws sit INSIDE the endTurn per-mon loop
//!     (DisableMove → TrapPokemon → MaybeTrapPokemon per mon, battle.ts:1689-1755),
//!     BEFORE the gen3 quickClawRoll (battle.ts:1795).
//!
//!   TRAPPED gates ONLY the voluntary switch: the sim's `chooseSwitch` at a `move`
//!     request rejects it DRAW-FREE ("Can't switch: The active Pokémon is trapped");
//!     a PHAZE still drags a trapped mon, a fainted mon's forced replacement is
//!     accepted, and the trapping mon itself switches freely.
//!
//! `trapping_golden_matches_showdown` — the DIFFERENTIAL gate. For each
//! (scenario, seed) in `harness/gen_trapping_golden.js`'s golden (gen3customgame),
//! seed a `BattleState` at the sim's PRNG state at the first decision (`init_seed`),
//! run `run_full_battle(script)` WITHOUT re-seeding, and assert per DECISION
//! BOUNDARY: each side's post-decision active (species/hp/maxhp/fainted/status) +
//! pokemon_left + request kind + first mover + **the per-side TRAPPED flag** (at
//! `move`-request boundaries — `is_trapped` vs the sim's `pokemon.trapped`), AND the
//! post-decision PRNG seed == the sim's `seed_after` — the draw-order+count proof to
//! game-end (the Magneton mirror's 4-per-endTurn trap-event shuffles, their
//! disappearance when a para breaks the speed tie, the cross-pair's 2, and the
//! Dugtrio mirror's ZERO must all be in the exact place/count). PLUS the winner.
//!
//! The switch-REJECTION path (a trapped mon's scripted `Switch` skipped draw-free) is
//! pinned by `tests/regression_test.rs` T1-T4 — the golden scripts never submit an
//! illegal switch (the sim would reject it and the harness fails loud), mirroring how
//! the PP layer split its golden vs its reject-gate pins.

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
    /// Per-side TRAPPED at this boundary: `Some(bool)` at a `move`-request boundary
    /// (the sim's `pokemon.trapped` truthiness, freshly recomputed by the endTurn that
    /// closed the decision), `None` ('-') at a mid-turn forced-switch pause / game end
    /// (where the sim's flag is stale — not asserted).
    trapped: [Option<bool>; 2],
    /// Whether a phaze DRAG fired during this decision (the `|drag|` line).
    drag: bool,
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
    left: usize,
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

fn parse_trap_tok(tok: &str) -> Option<bool> {
    match tok {
        "-" => None,
        "1" => Some(true),
        "0" => Some(false),
        other => panic!("bad trapped token {other:?}"),
    }
}

fn parse_golden() -> (BTreeMap<String, ScenMeta>, Vec<RunCase>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/trapping_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing trapping golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_trapping_golden.js")
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
                //   p2(...)[15..21)  first[21]  trapP1[22]  trapP2[23]  drag[24]
                assert_eq!(f.len(), 25, "DEC needs 25 fields (line {ln}), got {}", f.len());
                let req = match f[3] {
                    "move" => ReqTok::Move,
                    "switch" => ReqTok::Switch,
                    other => panic!("bad request {other:?} (line {ln})"),
                };
                let force = [f[4] == "1", f[5] == "1"];
                let choice = [parse_choice(f[6]), parse_choice(f[7])];
                let seed_after = f[8].to_string();
                let g = |i: usize| f[i].parse::<u16>().unwrap_or_else(|e| panic!("bad num f[{i}] (line {ln}): {e}"));
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
                let trapped = [parse_trap_tok(f[22]), parse_trap_tok(f[23])];
                let drag = f[24] == "1";
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect {
                    request: req, force, choice, seed_after, p1, p2, first_mover, trapped, drag,
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
fn trapping_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(meta.len() >= 8, "expected >=8 scenarios, got {}", meta.len());
    assert!(cases.len() >= 400, "expected the per-seed corpus (>=400 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut trapped_assertions = 0usize;
    let mut trapped_rows = 0usize; // a side is trapped at an asserted boundary
    let mut mutual_trap_rows = 0usize; // BOTH sides trapped (the mirrors)
    let mut drag_rows = 0usize; // a phaze dragged a trapped mon
    let mut voluntary_switch_rows = 0usize; // an accepted free voluntary switch
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
            "[{}] init prng seed must equal the sim's (switch-ins draw-free; the sim's \
             pre-turn-1 endTurn trap draws are absorbed into the recorded initSeed)",
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
                    assert!(
                        status_variant_eq(snap.status, e.status),
                        "[{}] dec {} side {} STATUS mismatch (init_seed {}): got {:?} exp {:?}",
                        case.scen, di, idx, case.init_seed, snap.status, e.status
                    );
                }
            }
            assert_eq!(rec.pokemon_left[0], exp.p1.left, "[{}] dec {} p1 left", case.scen, di);
            assert_eq!(rec.pokemon_left[1], exp.p2.left, "[{}] dec {} p2 left", case.scen, di);

            // --- PER-SIDE TRAPPED (the switch-legality fact). Asserted at every
            //     `move`-request boundary the golden marked (the sim's endTurn just
            //     recomputed `pokemon.trapped` there; the port computes it live at the
            //     identical instant). A wrong grounded/Steel/Levitate rule, a missed
            //     mutual-mirror trap, or a phantom self-trap diverges HERE. ---
            for s in 0..2usize {
                if let Some(exp_trap) = exp.trapped[s] {
                    assert_eq!(
                        rec.trapped[s], exp_trap,
                        "[{}] dec {} side {} TRAPPED mismatch (init_seed {}): got {} exp {}\n  \
                         is_trapped() disagrees with the sim's pokemon.trapped: check the \
                         Arena-Trap grounded rule (Flying/Levitate escape; a grounded Ghost \
                         IS trapped), the Magnet-Pull Steel rule (flying-irrelevant), and \
                         that a mon never traps ITSELF.",
                        case.scen, di, s, case.init_seed, rec.trapped[s], exp_trap
                    );
                    trapped_assertions += 1;
                    if exp_trap {
                        trapped_rows += 1;
                    }
                }
            }
            if exp.trapped[0] == Some(true) && exp.trapped[1] == Some(true) {
                mutual_trap_rows += 1;
            }
            if exp.drag {
                drag_rows += 1;
            }
            if exp.request == ReqTok::Move
                && exp.choice.iter().any(|c| matches!(c, Some(Choice::Switch(_))))
            {
                voluntary_switch_rows += 1;
            }

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

            // --- PER-DECISION SEED PARITY (the draw-order+count proof). The trapped
            //     COMPUTATION is draw-free, but the endTurn TrapPokemon/MaybeTrapPokemon
            //     handler-sort tie-shuffles are NOT when two trap handlers speed-tie:
            //     the Magneton mirror draws 4 per endTurn (2 events x 2 mons), the
            //     AT-vs-MP cross draws 2, the Dugtrio mirror ZERO, and a para that
            //     breaks the speed tie silences them. One extra/missing draw desyncs
            //     the LCG here. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 an endTurn trap-event tie-shuffle draw is mis-placed/missing/extra \
                 (the gen3 magnetpull onAny 2-handler tie model). FIX THE DRAW ORDER, \
                 do not loosen the assert.",
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

    // Coverage floors: the trap semantics + draw model must actually realize.
    assert!(seed_assertions >= 4000, "expected the per-decision seed corpus (>=4000), got {seed_assertions}");
    assert!(trapped_assertions >= 6000, "expected per-decision trapped assertions (>=6000), got {trapped_assertions}");
    assert!(trapped_rows >= 1000, "expected trapped-active rows (>=1000), got {trapped_rows}");
    assert!(mutual_trap_rows >= 300, "expected mutual-trap (mirror) rows (>=300), got {mutual_trap_rows}");
    assert!(drag_rows >= 50, "expected phaze-drags-a-trapped-mon rows (>=50), got {drag_rows}");
    assert!(voluntary_switch_rows >= 200, "expected accepted free voluntary-switch rows (>=200), got {voluntary_switch_rows}");
    assert!(win_runs >= 400, "expected real game-end WIN runs (>=400), got {win_runs}");

    eprintln!(
        "trapping golden: {} runs, {dec_assertions} STATE rows, {seed_assertions} seed assertions, \
         {trapped_assertions} trapped assertions ({trapped_rows} trapped, {mutual_trap_rows} mutual), \
         {drag_rows} drag rows, {voluntary_switch_rows} free voluntary switches, \
         {win_runs} wins, {tie_runs} ties",
        cases.len()
    );
}
