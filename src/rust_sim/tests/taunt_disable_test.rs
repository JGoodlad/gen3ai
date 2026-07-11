//! TAUNT + DISABLE full-battle tests — the per-seed PER-DECISION
//! STATE(+STATUS+TAUNT+DISABLED-SLOT)+SEED+winner differential that proves the
//! gen-3 move-SELECTION-restriction layer matches Showdown EXACTLY, to GAME-END.
//!
//! The mechanics (verified bit-for-bit vs the omniscient sim — the probes are
//! `harness/probe_taunt_disable_rng.js` + `probe_disable_full_lifecycle.js` +
//! `probe_taunt_duration_branch.js`; do NOT trust intuition on the durations):
//!
//!   TAUNT (`taunt`): Dark, Status, accuracy 100 (DRAWS randomChance(100,100), NOT
//!     never-miss). Disables EVERY Status-category move (selection restriction ->
//!     forced Struggle if all usable moves are Status). Duration a CONSTANT 2 in
//!     ALL branches — the gen3 override sets `duration:2`, and the base onStart's
//!     `if (target.activeTurns && !willMove(target)) duration++` DOES NOT manifest
//!     (a taunter-SECOND-on-turn>=2 lasts the SAME 2 turns as a taunter-first —
//!     PROVEN vs the sim). NO duration draw. Residual tick at order 10, subOrder 15.
//!
//!   DISABLE (`disable`): Normal, Status, accuracy 55 (DRAWS randomChance(55,100),
//!     CAN miss). Disables the target's lastMove for a stored duration that DEPENDS
//!     on the move order at the disabler's onStart (`willMove(target)`):
//!       - disabler moved FIRST (willMove(target) TRUE)  -> stored = random(2,6)
//!       - disabler moved SECOND (willMove(target) FALSE) -> stored = random(2,6)+1
//!     (VERIFIED post-onStart vs the sim; the residual DisableDuration handler then
//!     ticks it -1 each residual — including the disable turn's own — and frees the
//!     move at 0, so the FASTER-disabler case frees one turn EARLIER than the SLOWER
//!     one). onTryHit FAILS draw-free with no lastMove. Residual at NO_ORDER subOrder 2.
//!
//!   `taunt_disable_golden_matches_showdown` — the DIFFERENTIAL gate. For each
//!   (scenario, seed) in `harness/gen_taunt_disable_golden.js`'s golden (FORMAT
//!   gen3customgame, `sleep_clause` OFF), seed a `BattleState` at the sim's PRNG state
//!   at the first decision (`init_seed`), run `run_full_battle(script)` WITHOUT
//!   re-seeding, and assert per DECISION BOUNDARY: each side's post-decision active
//!   (species/hp/maxhp/fainted/status + **taunted** + the **disabled slot**) +
//!   pokemon_left + request kind + first mover; AND the post-decision PRNG seed ==
//!   the sim's `seed_after` — the EXACT cross-decision draw-order+count proof to
//!   game-end (the taunt accuracy(100) + the disable accuracy(55) + its random(2,6)
//!   must be in the exact place/count). PLUS the final WINNER.
//!
//!   THE OFF-BY-ONE GATE (BLOCKER 1): the `disabled` slot is asserted at EVERY
//!   boundary. If the port stored the WRONG disable duration (e.g. `rolled-1`/`rolled`
//!   OR `rolled+1`/`rolled+2` instead of the sim-correct `rolled`/`rolled+1`), the
//!   disabled slot would clear one boundary too early/late vs the sim -> a STATE
//!   divergence at the free-up boundary in `disable_faster_disabler_free_up` /
//!   `disable_slower_disabler_free_up`. This test FAILS on any such off-by-one and
//!   PASSES only on the sim-exact formula.

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
    /// The sim-reported branch outcomes: `[tauntStart, disableStart, miss, fail, struggle]`.
    outcomes: [bool; 5],
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
    /// Whether this active mon is TAUNTED at the boundary (the golden's `taunted` column).
    taunted: bool,
    /// The DISABLED move slot (`-1` = none, `0..=3` = the disabled slot).
    disabled: i8,
}

fn parse_status(tok: &str) -> Option<Status> {
    match tok {
        "-" => None,
        "fnt" => None,
        "brn" => Some(Status::Burn),
        "par" => Some(Status::Paralysis),
        // The taunt/disable golden records only the status TOKEN (no stage column); the
        // variant is asserted, the sleep/Toxic inner counter (a separate draw) is proven by
        // the SEED parity, not a stage assert here.
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
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/taunt_disable_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing taunt/disable golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_taunt_disable_golden.js")
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
                //   p1(species hp maxhp fnt status left taunted disabled)[9..17)
                //   p2(...)[17..25)  first[25]
                //   oc(tauntStart disableStart miss fail struggle)[26..31)
                assert_eq!(f.len(), 31, "DEC needs 31 fields (line {ln}), got {}", f.len());
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
                    status: parse_status(f[13]),
                    left: g(14) as usize,
                    taunted: f[15] == "1",
                    disabled: gi(16),
                };
                let p2 = SideExpect {
                    species: f[17].to_string(),
                    hp: g(18),
                    maxhp: g(19),
                    fainted: f[20] == "1",
                    status: parse_status(f[21]),
                    left: g(22) as usize,
                    taunted: f[23] == "1",
                    disabled: gi(24),
                };
                let first_mover = f[25].to_string();
                let b = |i: usize| f[i] == "1";
                let outcomes = [b(26), b(27), b(28), b(29), b(30)];
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect {
                    request: req, force, choice, seed_after, p1, p2, first_mover, outcomes,
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
        // gen3customgame -> NO clauses (no SetStatus handler-sort shuffle). The taunt/disable
        // moves never reach the SetStatus event, so this only matters for the target's Rest.
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
fn taunt_disable_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(meta.len() >= 9, "expected >=9 scenarios, got {}", meta.len());
    assert!(cases.len() >= 400, "expected the per-seed corpus (>=400 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut taunt_assertions = 0usize;
    let mut disable_assertions = 0usize;
    let mut taunted_rows = 0usize; // a live mon is taunted at the boundary
    let mut disabled_rows = 0usize; // a live mon has a disabled slot at the boundary
    let mut taunt_start_rows = 0usize;
    let mut taunt_end_rows = 0usize;
    let mut disable_start_rows = 0usize;
    let mut disable_end_rows = 0usize;
    let mut miss_rows = 0usize;
    let mut fail_rows = 0usize;
    let mut struggle_rows = 0usize;
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
                    assert!(
                        status_variant_eq(snap.status, e.status),
                        "[{}] dec {} side {} STATUS mismatch (init_seed {}): got {:?} exp {:?}",
                        case.scen, di, idx, case.init_seed, snap.status, e.status
                    );

                    // --- TAUNT presence (the selection-restriction volatile). A wrong taunt
                    //     duration (LANDS/EXPIRES a boundary early/late) diverges HERE. ---
                    assert_eq!(
                        snap.taunted, e.taunted,
                        "[{}] dec {} side {} TAUNTED mismatch (init_seed {}): got {} exp {}\n  \
                         the taunt volatile landed/expired at the wrong boundary (a wrong \
                         FIXED-2 duration or a mis-timed residual tick).",
                        case.scen, di, idx, case.init_seed, snap.taunted, e.taunted
                    );
                    taunt_assertions += 1;
                    if e.taunted {
                        taunted_rows += 1;
                    }

                    // --- DISABLED SLOT (the off-by-one gate). If the stored disable duration
                    //     is wrong on EITHER the faster (rolled) or slower (rolled+1) branch,
                    //     the disabled slot clears one boundary early/late vs the sim -> HERE. ---
                    assert_eq!(
                        snap.disabled_slot, e.disabled,
                        "[{}] dec {} side {} DISABLED-SLOT mismatch (init_seed {}): got {} exp {}\n  \
                         the disable landed on the wrong slot, or its stored duration is wrong \
                         (the faster-disabler branch must store random(2,6); the slower one \
                         random(2,6)+1 — a +/-1 off-by-one frees the move a boundary early/late).",
                        case.scen, di, idx, case.init_seed, snap.disabled_slot, e.disabled
                    );
                    disable_assertions += 1;
                    if e.disabled >= 0 {
                        disabled_rows += 1;
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
                        "[{}] dec {} first-mover mismatch (init_seed {}): got {:?} exp {:?}",
                        case.scen, di, case.init_seed, rec.first_mover, sim_first
                    );
                }
            }

            // --- PER-DECISION SEED PARITY (the draw-order+count proof). Taunt draws
            //     accuracy(100); Disable draws accuracy(55) then (on a landed hit into a
            //     mon with a lastMove) ONE random(2,6). A no-lastMove / already-disabled /
            //     Protect-blocked Disable draws accuracy only. One extra/missing/mis-ordered
            //     draw desyncs the LCG here. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 a taunt/disable accuracy draw or the landed-disable random(2,6) is \
                 mis-ordered/missing/extra. FIX THE DRAW ORDER, do not loosen the assert.",
                case.scen, di, case.init_seed, rec.seed_after, exp.seed_after
            );
            seed_assertions += 1;
            dec_assertions += 1;

            if exp.outcomes[0] {
                taunt_start_rows += 1;
            }
            if exp.outcomes[1] {
                disable_start_rows += 1;
            }
            if exp.outcomes[2] {
                miss_rows += 1;
            }
            if exp.outcomes[3] {
                fail_rows += 1;
            }
            if exp.outcomes[4] {
                struggle_rows += 1;
            }
        }

        // Count taunt/disable FREE-UP boundaries: a boundary where a live active mon was
        // taunted/disabled on the PREVIOUS boundary but is not now (the residual expiry). These
        // are the off-by-one-sensitive boundaries — asserted per-decision above; counted here.
        for w in outcome.decisions.windows(2) {
            for s in 0..2usize {
                if !w[0].active[s].fainted && !w[1].active[s].fainted {
                    if w[0].active[s].taunted && !w[1].active[s].taunted {
                        taunt_end_rows += 1;
                    }
                    if w[0].active[s].disabled_slot >= 0 && w[1].active[s].disabled_slot < 0 {
                        disable_end_rows += 1;
                    }
                }
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

    // Coverage floors: every taunt/disable branch must actually realize across the corpus,
    // INCLUDING both disabler-duration branches' free-ups (the off-by-one gate) and the
    // taunter-second (MINOR A) path.
    assert!(seed_assertions >= 2000, "expected the per-decision seed corpus (>=2000), got {seed_assertions}");
    assert!(taunt_assertions >= 2000, "expected per-decision taunt assertions (>=2000), got {taunt_assertions}");
    assert!(disable_assertions >= 2000, "expected per-decision disabled-slot assertions (>=2000), got {disable_assertions}");
    assert!(taunted_rows >= 100, "expected taunted-active rows (>=100), got {taunted_rows}");
    assert!(disabled_rows >= 100, "expected disabled-active rows (>=100), got {disabled_rows}");
    assert!(taunt_start_rows >= 50, "expected taunt-applied rows (>=50), got {taunt_start_rows}");
    assert!(taunt_end_rows >= 20, "expected taunt-free-up boundaries (>=20), got {taunt_end_rows}");
    assert!(disable_start_rows >= 30, "expected disable-applied rows (>=30), got {disable_start_rows}");
    assert!(disable_end_rows >= 20, "expected disable-free-up boundaries (the off-by-one gate, >=20), got {disable_end_rows}");
    assert!(miss_rows >= 20, "expected disable-miss rows (>=20), got {miss_rows}");
    assert!(fail_rows >= 20, "expected no-lastMove -fail rows (>=20), got {fail_rows}");
    assert!(struggle_rows >= 30, "expected forced-Struggle rows (>=30), got {struggle_rows}");
    assert!(win_runs >= 100, "expected real game-end WIN runs (>=100), got {win_runs}");

    eprintln!(
        "taunt/disable golden: {} runs, {dec_assertions} STATE rows, {seed_assertions} seed assertions, \
         {taunt_assertions} taunt + {disable_assertions} disabled-slot assertions \
         ({taunted_rows} taunted, {disabled_rows} disabled), \
         start(taunt {taunt_start_rows}, disable {disable_start_rows}), \
         free-up(taunt {taunt_end_rows}, disable {disable_end_rows}), \
         {miss_rows} miss, {fail_rows} fail, {struggle_rows} struggle, {win_runs} wins, {tie_runs} ties",
        cases.len()
    );
}
