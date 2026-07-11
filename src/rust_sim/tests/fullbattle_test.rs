//! Full-battle (move + switch + post-faint replacement → win/loss) tests — the
//! per-seed PER-DECISION STATE+SEED differential that proves the SWITCH-PHASE PRNG
//! draw order (the two-switch action-order shuffle, the around-switch
//! `eachEvent('Update')` shuffles, the double-replacement `insertChoice` splice,
//! the gen3-runSwitch draw-FREE switch-in, and the win-decides-no-QuickClaw rule)
//! matches Showdown EXACTLY, sustained across faints + replacements to GAME-END.
//!
//!   - `fullbattle_golden_matches_showdown` — the DIFFERENTIAL gate. For each
//!     (scenario, seed) in `harness/gen_fullbattle_golden.js`'s golden, seed a
//!     `BattleState` with the sim's PRNG state at the FIRST decision's pre-choice
//!     boundary (`init_seed`), then run `run_full_battle(script)` WITHOUT
//!     re-seeding, and assert per DECISION BOUNDARY (each `move` turn AND each
//!     forced-`switch` replacement sub-step):
//!       (a) each side's post-decision active (species/hp/maxhp/fainted/status) +
//!           pokemon_left match, the request kind (move vs forceSwitch) matches,
//!           and (on a move turn) the first mover matches; AND
//!       (b) the post-decision PRNG seed equals the sim's `seed_after` — the EXACT
//!           cross-decision draw-order+count proof to game-end (a single
//!           extra/missing/mis-ordered switch-phase draw desyncs the LCG).
//!     PLUS the final WINNER (or tie) matches the sim's `|win|`/`winner`.
//!
//!   It runs BOTH a single-seed cross-decision carry (the final-seed + winner) AND
//!   a per-decision pass that asserts EVERY boundary (pinpointing the first
//!   diverging decision). Mirrors `tests/battle_test.rs` extended past faints.

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

/// One recorded full battle (one scenario at one seed).
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
    /// The EXACT choices the sim was given (the faithful script the Rust replays).
    choice: [Option<Choice>; 2],
    seed_after: String,
    p1: SideExpect,
    p2: SideExpect,
    first_mover: String,
}

/// Decode a golden choice token: `m<K>` ⇒ `Move(K)`, `s<N>` ⇒ `Switch(N)`,
/// `-` ⇒ `None`.
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

fn parse_status(tok: &str, stage: u16) -> Option<Status> {
    match tok {
        "-" => None,
        // A fainted mon reports status="fnt" in the sim; we track fainted as a
        // separate flag and carry no major status on a fainted mon (the Rust
        // `MonState::status` stays whatever it was — but the sim shows 'fnt', and a
        // fainted mon's status is asserted only via the `fainted` flag, not here).
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

fn parse_golden() -> (BTreeMap<String, ScenMeta>, Vec<RunCase>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/fullbattle_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing fullbattle golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_fullbattle_golden.js")
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
                // INIT <id> <init_seed> <group_seed>
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
                // DEC <id> <grp> <di> <req> <fp1> <fp2> <cp1> <cp2> <seedAfter>
                //     p1(species hp max fnt status stage left) p2(...) first
                assert_eq!(f.len(), 25, "DEC needs 25 fields (line {ln}), got {}", f.len());
                let req = match f[4] {
                    "move" => ReqTok::Move,
                    "switch" => ReqTok::Switch,
                    other => panic!("bad request {other:?} (line {ln})"),
                };
                let force = [f[5] == "1", f[6] == "1"];
                let choice = [parse_choice(f[7]), parse_choice(f[8])];
                let seed_after = f[9].to_string();
                // p1 block: f[10..17) = species hp max fnt status stage left
                let g = |i: usize| f[i].parse::<u16>().unwrap_or_else(|e| panic!("bad num f[{i}] (line {ln}): {e}"));
                let p1 = SideExpect {
                    species: f[10].to_string(),
                    hp: g(11),
                    maxhp: g(12),
                    fainted: f[13] == "1",
                    status: parse_status(f[14], g(15)),
                    left: g(16) as usize,
                };
                // p2 block: f[17..24) = species hp max fnt status stage left
                let p2 = SideExpect {
                    species: f[17].to_string(),
                    hp: g(18),
                    maxhp: g(19),
                    fainted: f[20] == "1",
                    status: parse_status(f[21], g(22)),
                    left: g(23) as usize,
                };
                let first_mover = f[24].to_string();
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect { request: req, force, choice, seed_after, p1, p2, first_mover });
            }
            "END" => {
                // END <id> <grp> <ended> <winner>
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

/// Build the [`ScriptDecision`] list that REPRODUCES the sim's EXACT recorded
/// choices, read directly from the golden's `choice` tokens — no species-based
/// reconstruction (so duplicate-species teams are unambiguous). This makes the
/// Rust driver play byte-for-byte the SAME sequence the sim recorded.
fn script_from_decisions(case: &RunCase) -> Vec<ScriptDecision> {
    case.decisions
        .iter()
        .map(|dec| ScriptDecision { p1: dec.choice[0], p2: dec.choice[1] })
        .collect()
}

/// Normalize a species name (display form `Tyranitar` or id `tyranitar`) to an id
/// for comparison: lowercase, strip non-alphanumerics.
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
fn fullbattle_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(meta.len() >= 6, "expected >=6 scenarios, got {}", meta.len());
    assert!(cases.len() >= 200, "expected the per-seed corpus (>=200 runs), got {}", cases.len());

    let mut dec_assertions = 0usize; // (scenario,seed,decision) STATE rows
    let mut seed_assertions = 0usize; // per-decision seed-parity assertions
    let mut switch_req_rows = 0usize; // forced-switch decision boundaries
    let mut double_replace_rows = 0usize;
    let mut win_runs = 0usize;
    let mut tie_runs = 0usize;
    let mut faint_carry_runs = 0usize; // runs that continued PAST a faint

    for case in &cases {
        let m = meta.get(&case.scen).unwrap_or_else(|| panic!("no meta for {}", case.scen));
        assert!(!case.decisions.is_empty(), "[{}] empty run", case.scen);

        let opts = opts_for(m, &case.init_seed);
        let mut battle = Battle::start_with_switchins(&opts, &d)
            .unwrap_or_else(|e| panic!("[{}] start failed: {e}", case.scen));

        // The constructed prng must equal the sim's init seed (switch-ins draw-free).
        assert_eq!(
            battle.state().unwrap().prng_seed(),
            case.init_seed,
            "[{}] init prng seed must equal the sim's (switch-ins are draw-free)",
            case.scen
        );

        let script = script_from_decisions(case);
        assert_eq!(
            script.len(),
            case.decisions.len(),
            "[{}] script length must match recorded decisions",
            case.scen
        );

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

        let mut saw_faint_before_end = false;
        for (di, (rec, exp)) in outcome.decisions.iter().zip(case.decisions.iter()).enumerate() {
            // --- request kind + force table ---
            assert!(
                req_eq(&rec.request, exp.request, exp.force),
                "[{}] decision {} request mismatch (init_seed {}): got {:?} exp {:?} force {:?}",
                case.scen, di, case.init_seed, rec.request, exp.request, exp.force
            );
            if exp.request == ReqTok::Switch {
                switch_req_rows += 1;
                saw_faint_before_end = true;
                if exp.force[0] && exp.force[1] {
                    double_replace_rows += 1;
                }
            }

            // --- (a) STATE: per-side active species / hp / maxhp / fainted / status
            //     + pokemon_left. ---
            for (idx, (snap, e, sp)) in [
                (0usize, (&rec.active[0], &exp.p1, &rec.active_species[0])),
                (1usize, (&rec.active[1], &exp.p2, &rec.active_species[1])),
            ] {
                // ACTIVE-MON IDENTITY: the Rust active species must match the sim's
                // (proves the switchIn array-swap brought the right mon to active).
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
                // Status only matters on a LIVE mon — a fainted mon reports 'fnt' in
                // the sim while the Rust keeps its pre-faint status; both are dead, so
                // we assert status only when not fainted.
                if !e.fainted {
                    assert!(
                        status_variant_eq(snap.status, e.status),
                        "[{}] dec {} side {} status mismatch (init_seed {}): got {:?} exp {:?}",
                        case.scen, di, idx, case.init_seed, snap.status, e.status
                    );
                }
            }
            assert_eq!(
                rec.pokemon_left[0], exp.p1.left,
                "[{}] dec {} p1 pokemon_left mismatch (init_seed {}): got {} exp {}",
                case.scen, di, case.init_seed, rec.pokemon_left[0], exp.p1.left
            );
            assert_eq!(
                rec.pokemon_left[1], exp.p2.left,
                "[{}] dec {} p2 pokemon_left mismatch (init_seed {}): got {} exp {}",
                case.scen, di, case.init_seed, rec.pokemon_left[1], exp.p2.left
            );

            // first mover on a move turn.
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

            // --- (b) PER-DECISION SEED PARITY: the post-decision prng seed must equal
            //     the sim's seed_after at EVERY boundary. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 a switch-phase draw is mis-ordered/missing/extra (action-order shuffle, \
                 around-switch eachEvent, insertChoice splice, or a wrongly-drawn Quick \
                 Claw on a deciding faint). FIX THE DRAW ORDER, do not loosen the assert.",
                case.scen, di, case.init_seed, rec.seed_after, exp.seed_after
            );
            seed_assertions += 1;
            dec_assertions += 1;
        }

        if saw_faint_before_end {
            faint_carry_runs += 1;
        }

        // --- WINNER + ended ---
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

    assert!(seed_assertions >= 1500, "expected the per-decision seed corpus (>=1500), got {seed_assertions}");
    assert!(switch_req_rows >= 50, "expected forced-switch decision rows, got {switch_req_rows}");
    assert!(double_replace_rows >= 10, "expected DOUBLE-replacement rows, got {double_replace_rows}");
    assert!(win_runs >= 50, "expected real game-end WIN runs, got {win_runs}");
    assert!(faint_carry_runs >= 50, "expected runs that continued PAST a faint, got {faint_carry_runs}");

    eprintln!(
        "fullbattle golden: {} runs, {dec_assertions} (scenario,seed,decision) STATE rows, \
         {seed_assertions} per-decision seed assertions, {switch_req_rows} forced-switch reqs \
         ({double_replace_rows} double), {win_runs} wins, {tie_runs} ties, {faint_carry_runs} \
         past-faint runs",
        cases.len()
    );
}

