//! ABILITY BATCH-2 class-sweep tests (`gen3_ability_batch2_v1`) — the per-seed PER-DECISION
//! STATE+HP+STATUS+SEED+winner differential proving the DRAW-BEARING "reactive" ability classes
//! + the block tail match Showdown EXACTLY, to GAME-END:
//!
//!   CONTACT_PROC (Static par / Poison Point psn / Flame Body brn / Effect Spore slp|par|psn) —
//!     an onDamagingHit that draws `randomChance(chance)` (Static/PP/FB = 1/3; Effect Spore = 1/10
//!     then a sample(3)) on a CONTACT hit into the holder and statuses the ATTACKER. The proc's
//!     randomChance draws INSIDE runEvent('DamagingHit') (gen<5) — AFTER the move's own secondary
//!     random(100). The ATTACKER's STATUS timeline (asserted per decision) proves the proc; any
//!     extra/missing/mis-ordered draw desyncs the SEED here. A no-op / non-contact control does not proc.
//!   CONTACT recoil (Rough Skin) — a DRAW-FREE baseMaxhp/16 recoil to the ATTACKER (asserted via HP).
//!   BLOCK — Damp (Explosion cancelled at TryMove: the user does NOT self-KO — asserted via the
//!     user's `fainted`/HP), Soundproof (a sound move is immune — no status), Suction Cups (a phaze
//!     into the holder draws no sample — the holder stays ACTIVE, asserted via species).
//!   SYNCHRONIZE — reflect a foe status back to the SOURCE (slp/frz exempt; tox→psn) — asserted via
//!     the SOURCE's status; draw-free in gen3customgame (this golden's format).
//!
//! The golden (`harness/gen_ability_batch2_golden.js`) drives the OMNISCIENT BattleStream to
//! game-end; this test replays each (scenario, seed) from the sim's init seed WITHOUT re-seeding
//! and asserts, per decision boundary: both actives' species/hp/maxhp/fainted/status + pokemon_left
//! + the request kind + the first mover + the post-decision PRNG seed + the final winner. The
//! contact-proc `randomChance` / Effect Spore `sample` are the ONLY new draws, so ANY
//! extra/missing/mis-ordered one desyncs the seed here.

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::state::Status;
use pokesim::turn::{Choice, RequestKind, ScriptDecision};
use std::collections::BTreeMap;

fn dex() -> Dex {
    Dex::for_gen(3)
}

const SPE_BOOST_IDX: usize = 4;

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
    covered: bool,
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
    spe_boost: i8,
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

/// Single-pass parse threading the current scenario id. The batch2 golden's INIT line carries NO
/// scenario id (unlike batch1's), so each INIT/DEC/END run belongs to the most recently declared
/// SCEN (`last_scen`) — the golden emits SCEN, TEAM×2, then a run of INIT/DEC/END per scenario.
fn parse_golden() -> (BTreeMap<String, ScenMeta>, Vec<RunCase>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/ability_batch2_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing ability batch2 golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_ability_batch2_golden.js")
    });
    let mut meta: BTreeMap<String, ScenMeta> = BTreeMap::new();
    let mut cases: Vec<RunCase> = Vec::new();
    let mut cur: Option<RunCase> = None;
    let mut last_scen = String::new();
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
                last_scen = f[1].to_string();
                meta.entry(last_scen.clone()).or_default();
            }
            "TEAM" => {
                // The batch2 golden emits `TEAM <side> <packed>` (3 fields, belonging to the
                // most-recently-declared SCEN) — unlike batch1's 4-field `TEAM <scen> <side> …`.
                assert_eq!(f.len(), 3, "TEAM needs 3 fields (line {ln})");
                let side = if f[1] == "p1" { 0 } else { 1 };
                meta.entry(last_scen.clone()).or_default().teams[side] = Some(f[2].to_string());
            }
            "INIT" => {
                assert_eq!(f.len(), 2, "INIT needs 2 fields (line {ln})");
                flush(&mut cur, &mut cases);
                cur = Some(RunCase {
                    scen: last_scen.clone(),
                    init_seed: f[1].to_string(),
                    decisions: Vec::new(),
                    ended: false,
                    winner: WinTok::None,
                });
            }
            "DEC" => {
                // DEC <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> <c1> <c2> <first> <covered>
                //   (11 TAB fields; c1/c2 are COMMA-joined snapshots
                //    `species,hp,maxhp,fainted,status,left,speBoost`).
                assert_eq!(f.len(), 11, "DEC needs 11 fields (line {ln}), got {}", f.len());
                let req = match f[1] {
                    "move" => ReqTok::Move,
                    "switch" => ReqTok::Switch,
                    other => panic!("bad request {other:?} (line {ln})"),
                };
                let force = [f[2] == "1", f[3] == "1"];
                let choice = [parse_choice(f[4]), parse_choice(f[5])];
                let seed_after = f[6].to_string();
                let parse_side = |field: &str| -> SideExpect {
                    let g: Vec<&str> = field.split(',').collect();
                    assert_eq!(g.len(), 7, "snapshot needs 7 comma fields (line {ln}): {field:?}");
                    SideExpect {
                        species: g[0].to_string(),
                        hp: g[1].parse().unwrap_or_else(|e| panic!("bad hp (line {ln}): {e}")),
                        maxhp: g[2].parse().unwrap_or_else(|e| panic!("bad maxhp (line {ln}): {e}")),
                        fainted: g[3] == "1",
                        status: parse_status(g[4]),
                        left: g[5].parse::<u16>().unwrap_or_else(|e| panic!("bad left (line {ln}): {e}")) as usize,
                        spe_boost: g[6].parse().unwrap_or_else(|e| panic!("bad speBoost (line {ln}): {e}")),
                    }
                };
                let p1 = parse_side(f[7]);
                let p2 = parse_side(f[8]);
                let first_mover = f[9].to_string();
                let covered = f[10] == "1";
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect { request: req, force, choice, seed_after, p1, p2, first_mover, covered });
            }
            "END" => {
                assert_eq!(f.len(), 3, "END needs 3 fields (line {ln})");
                let c = cur.as_mut().unwrap_or_else(|| panic!("END before INIT (line {ln})"));
                // `battle.winner` is the PLAYER NAME ("P1"/"P2"), not "p1"/"p2" — accept both.
                c.winner = match f[1] {
                    "p1" | "P1" => WinTok::P1,
                    "p2" | "P2" => WinTok::P2,
                    "tie" => WinTok::Tie,
                    "none" => WinTok::None,
                    other => panic!("bad winner {other:?} (line {ln})"),
                };
                c.ended = f[2] == "1";
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
fn ability_batch2_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(
        meta.len() >= 10,
        "expected >=10 scenarios (contact procs + recoil + BLOCK + SYNCHRONIZE + controls), got {}",
        meta.len()
    );
    assert!(cases.len() >= 800, "expected the per-seed corpus (>=800 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut status_assertions = 0usize;
    let mut covered_rows = 0usize;
    let mut covered_per_scen: BTreeMap<String, usize> = BTreeMap::new();
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

        let script: Vec<ScriptDecision> = case
            .decisions
            .iter()
            .map(|dec| ScriptDecision { p1: dec.choice[0], p2: dec.choice[1] })
            .collect();
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
                "[{}] decision {} request mismatch (init_seed {}): got {:?} exp {:?} force {:?}",
                case.scen, di, case.init_seed, rec.request, exp.request, exp.force
            );

            for (idx, (snap, e, sp)) in [
                (0usize, (&rec.active[0], &exp.p1, &rec.active_species[0])),
                (1usize, (&rec.active[1], &exp.p2, &rec.active_species[1])),
            ] {
                // --- THE SPECIES GATE: a Suction Cups holder that WRONGLY got dragged (or a
                //     phaze that wrongly stayed) makes the WRONG mon active here. ---
                assert_eq!(
                    species_id(sp),
                    species_id(&e.species),
                    "[{}] dec {} side {} active species mismatch (init_seed {})",
                    case.scen, di, idx, case.init_seed
                );
                // --- THE HP GATE: a Rough-Skin recoil that mis-applied, or a Damp-blocked
                //     Explosion whose user WRONGLY self-KO'd (fainted → HP 0), lands a wrong HP. ---
                assert_eq!(
                    snap.hp, e.hp,
                    "[{}] dec {} side {} HP mismatch (init_seed {}): got {} exp {}\n  \
                     a batch-2 effect is wrong (a Rough-Skin recoil, or a Damp-blocked Explosion \
                     whose user should NOT have self-KO'd).",
                    case.scen, di, idx, case.init_seed, snap.hp, e.hp
                );
                assert_eq!(snap.maxhp, e.maxhp, "[{}] dec {} side {} maxhp", case.scen, di, idx);
                // --- THE FAINTED GATE: Damp — the Explosion user must NOT be fainted. ---
                assert_eq!(
                    snap.fainted, e.fainted,
                    "[{}] dec {} side {} fainted mismatch (init_seed {}): a Damp-blocked \
                     Explosion user wrongly self-KO'd (or a proc'd status wrongly KO'd a mon)",
                    case.scen, di, idx, case.init_seed
                );
                if !e.fainted {
                    // --- THE STATUS GATE (the crux): a CONTACT_PROC status on the ATTACKER, or a
                    //     SYNCHRONIZE reflect on the SOURCE, or a Soundproof-immune sound move (NO
                    //     status), must match. A wrong proc / reflect / immunity diverges here. ---
                    assert!(
                        status_variant_eq(snap.status, e.status),
                        "[{}] dec {} side {} STATUS mismatch (init_seed {}): got {:?} exp {:?}\n  \
                         a CONTACT_PROC status (on the ATTACKER), a SYNCHRONIZE reflect (on the \
                         SOURCE), or a Soundproof block is wrong.",
                        case.scen, di, idx, case.init_seed, snap.status, e.status
                    );
                    status_assertions += 1;
                    assert_eq!(
                        snap.boosts[SPE_BOOST_IDX], e.spe_boost,
                        "[{}] dec {} side {} SPE-BOOST mismatch (init_seed {})",
                        case.scen, di, idx, case.init_seed
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
                        "[{}] dec {} first-mover mismatch (init_seed {})",
                        case.scen, di, case.init_seed
                    );
                }
            }

            // --- PER-DECISION SEED PARITY (the draw-model gate): the contact-proc `randomChance`
            //     + the Effect Spore `sample` are the ONLY new draws (Rough Skin / Damp / Soundproof
            //     / Suction Cups / Synchronize are draw-free-or-fewer). An accidental extra/missing/
            //     mis-ordered draw — the contact proc in the WRONG position vs the move secondary,
            //     a sample that fired when it shouldn't, a Damp block that still drew acc/crit/dmg —
            //     desyncs the LCG here. FIX THE DRAW ORDER, do not loosen the assert. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 a batch-2 ability consumed/skipped/mis-ordered a PRNG draw (the contact-proc \
                 randomChance / Effect Spore sample position, or a Damp-blocked Explosion that \
                 wrongly drew acc/crit/dmg).",
                case.scen, di, case.init_seed, rec.seed_after, exp.seed_after
            );
            seed_assertions += 1;
            dec_assertions += 1;
            if exp.covered {
                covered_rows += 1;
                *covered_per_scen.entry(case.scen.clone()).or_default() += 1;
            }
        }

        assert_eq!(
            outcome.ended, case.ended,
            "[{}] ended mismatch (init_seed {})",
            case.scen, case.init_seed
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

    // Coverage floors: EVERY effect-bearing scenario must land its observable effect repeatedly.
    // The two CONTROLS (noncontact_no_proc, contact_proc_control_noop) MUST have 0 (they don't proc).
    let controls = ["noncontact_no_proc", "contact_proc_control_noop"];
    for (scen, _) in meta.iter() {
        let n = covered_per_scen.get(scen).copied().unwrap_or(0);
        if controls.contains(&scen.as_str()) {
            assert_eq!(n, 0, "[{scen}] a CONTROL must have 0 contact-proc cover, got {n}");
        } else {
            assert!(n >= 1, "[{scen}] {n} covered rows (<1) — the class effect never fired");
        }
    }
    assert!(seed_assertions >= 2000, "expected the per-decision seed corpus (>=2000), got {seed_assertions}");
    assert!(covered_rows >= 300, "expected covered rows (>=300), got {covered_rows}");
    assert!(win_runs >= 200, "expected real game-end WIN runs (>=200), got {win_runs}");

    eprintln!(
        "ability batch2 golden: {} runs over {} scenarios, {dec_assertions} STATE rows, \
         {seed_assertions} seed assertions, {status_assertions} status assertions, \
         {covered_rows} covered rows, {win_runs} wins, {tie_runs} ties",
        cases.len(),
        meta.len()
    );
}
