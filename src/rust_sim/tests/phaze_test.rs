//! PHAZING (Roar / Whirlwind) full-battle tests — the per-seed PER-DECISION
//! STATE(+HP+STATUS+SPIKES-LAYERS+DRAG-SPECIES)+SEED+winner differential that proves the
//! NEW phaze mechanic matches Showdown EXACTLY, sustained to GAME-END:
//!
//!   **Roar** + **Whirlwind** — the gen-3 `forceSwitch: true` moves: force the FOE to
//!   switch to a RANDOM eligible team member. DEFERRED (fail-loud in the engine): Haze
//!   (resets boosts — a DIFFERENT mechanic), Perish Song, Roar of Time (not gen3). The
//!   model (verified bit-for-bit vs the omniscient sim's PRNG probe —
//!   `harness/probe_phaze_rng.js`):
//!
//!   THE PHAZE MOVE (`forceSwitch: true`, `target:'normal'`, priority −6):
//!     * PRIORITY −6 → the phazer almost always moves LAST.
//!     * ACCURACY — gen-3 Roar/Whirlwind resolve to `accuracy: 100` (NOT `true`!), so they
//!       DRAW `randomChance(100, 100)` (it ALWAYS passes but CONSUMES a draw). A phaze is
//!       NOT never-miss.
//!     * THE RANDOM TARGET DRAW — on a successful phaze (the foe has >= 1 eligible bench
//!       mon), the runAction tail `dragIn`s: `getRandomSwitchable` → `sample` → `random(n)`
//!       — ONE draw, EVEN when n == 1 (`random(1)` returns 0 but STILL draws). A phaze with
//!       NO eligible target (the foe's last mon) FAILS draw-free (only the accuracy roll).
//!
//!   THE DRAG (forced switch-in, `dragIn` → `switchIn(isDrag=true)`):
//!     * The dragged-in mon takes Spikes via the existing runSwitch EntryHazard (drag →
//!       EntryHazard → SwitchIn → ability Start), fires its switch-in ability Start, and a
//!       Spikes-KO on the dragged-in mon faints it → forces a NORMAL replacement. Boosts/
//!       volatiles of the phazed-OUT mon are cleared. The dragged mon does NOT act this turn.
//!
//!   `phaze_golden_matches_showdown` — the DIFFERENTIAL gate. For each (scenario, seed) in
//!   `harness/gen_phaze_golden.js`'s golden (FORMAT gen3customgame), seed a `BattleState` at
//!   the sim's PRNG state at the first decision (`init_seed`), run `run_full_battle(script)`
//!   WITHOUT re-seeding, and assert per DECISION BOUNDARY: (a) each side's post-decision
//!   active (species/HP/maxhp/fainted/STATUS + the sleep/Toxic inner counter) + boosts +
//!   confusion + pokemon_left + THE SPIKES LAYERS (per side) + request kind + first mover;
//!   AND (b) the post-decision PRNG seed == the sim's `seed_after`. PLUS the final WINNER.
//!   The DRAGGED-IN mon is proved by the post-decision ACTIVE SPECIES (a wrong sampled mon
//!   diverges the active species — a STATE desync); the SEED proves the draw model (the
//!   accuracy roll + the n=1 sample draw + the no-draw on a FAIL must each be exact).
//!
//! The golden EXTENDS the spikes TAB format with a 1-col dragged-species tail → DEC has 50
//! fields.

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::state::Status;
use pokesim::turn::{Choice, RequestKind, ScriptDecision};
use std::collections::{BTreeMap, BTreeSet};

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
    /// The sim-reported `spikesDamage` flag (a phaze-dragged grounded mon took spikes).
    spikes_damage: bool,
    /// The species the sim dragged in THIS decision (`|drag|`, else `-`). The DRAG proof:
    /// the active species after a phaze MUST equal this (a wrong sampled mon diverges).
    drag_species: String,
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
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/phaze_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing phaze golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_phaze_golden.js")
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
                //   p2 out(...)[41..46)  spikesDamage[46]  p1Spikes[47] p2Spikes[48] dragSpecies[49]
                assert_eq!(f.len(), 50, "DEC needs 50 fields (line {ln}), got {}", f.len());
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
                let drag_species = f[49].to_string();
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect {
                    request: req, force, choice, seed_after, p1, p2, first_mover, spikes_damage, drag_species,
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
fn phaze_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(meta.len() >= 5, "expected >=5 scenarios, got {}", meta.len());
    assert!(cases.len() >= 350, "expected the per-seed corpus (>=350 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut hp_assertions = 0usize;
    let mut spikes_assertions = 0usize;
    let mut drag_rows = 0usize; // a phaze dragged a mon in this decision
    let mut phaze_into_spikes_rows = 0usize; // a dragged grounded mon took spikes
    let mut win_runs = 0usize;
    let mut tie_runs = 0usize;
    // Per scenario: the set of DISTINCT dragged species (proves the random-target draw).
    let mut distinct_drags: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

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
                // THE DRAG PROOF: a wrong sampled mon makes the WRONG mon active → this
                // species check diverges (a STATE desync the seed match alone wouldn't catch
                // if the alt mon happened to share HP). The post-phaze active species MUST
                // equal the sim's dragged mon.
                assert_eq!(
                    species_id(sp), species_id(&e.species),
                    "[{}] dec {} side {} active species mismatch (init_seed {}): got {:?} exp {:?}\n  \
                     a phaze dragged the WRONG random mon (the `sample` over the eligible bench \
                     must match `possibleSwitches`'s array order). FIX THE TARGET DRAW, do not loosen.",
                    case.scen, di, idx, case.init_seed, sp, e.species
                );
                // HP: a phaze-dragged grounded mon must take the EXACT spikes chip; a phazed-in
                // Flying/Levitate mon takes ZERO; a phaze that drags into a Spikes-KO faints it.
                assert_eq!(
                    snap.hp, e.hp,
                    "[{}] dec {} side {} HP mismatch (init_seed {}): got {} exp {}\n  \
                     a phaze-into-spikes chip was wrong (the dragged grounded mon must take the \
                     exact floor). FIX THE MODEL, do not loosen.",
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
                assert_eq!(
                    spikes, e.spikes,
                    "[{}] dec {} side {} SPIKES-LAYERS mismatch (init_seed {}): got {} exp {}",
                    case.scen, di, idx, case.init_seed, spikes, e.spikes
                );
                spikes_assertions += 1;

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
                    // BOOSTS: a phazed-OUT mon's boosts are cleared (it left); the phazed-IN
                    // mon enters at 0 boosts (modulo Intimidate). A wrong clear diverges here.
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

            // --- PER-DECISION SEED PARITY (the draw-order+count proof). A phaze move draws
            //     its accuracy roll (`randomChance(100,100)`, gen-3 Roar/Whirlwind are acc
            //     100, NOT never-miss); a SUCCESSFUL phaze ALSO draws ONE `sample` (the random
            //     target, even for n==1); a FAILED phaze (foe's last mon) draws ONLY the
            //     accuracy roll. The drag's runSwitch (EntryHazard/Spikes → ability Start) is
            //     draw-free. A wrong draw model (skipping the accuracy roll, treating n==1 as
            //     draw-free, or drawing the sample on a fail) desyncs the LCG HERE. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 the phaze draw model is wrong (the accuracy roll, the n>=1 `sample` target \
                 draw, or the no-draw-on-FAIL must each be exact). FIX THE DRAW MODEL, do not loosen.",
                case.scen, di, case.init_seed, rec.seed_after, exp.seed_after
            );
            seed_assertions += 1;
            dec_assertions += 1;

            if exp.drag_species != "-" {
                drag_rows += 1;
                distinct_drags
                    .entry(case.scen.clone())
                    .or_default()
                    .insert(species_id(&exp.drag_species));
            }
            if exp.spikes_damage {
                phaze_into_spikes_rows += 1;
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

    // Coverage floors: every phaze branch must actually realize across the corpus.
    assert!(seed_assertions >= 1500, "expected the per-decision seed corpus (>=1500), got {seed_assertions}");
    assert!(hp_assertions >= 1500, "expected per-decision HP assertions (>=1500), got {hp_assertions}");
    assert!(spikes_assertions >= 1500, "expected per-decision spikes-layer assertions (>=1500), got {spikes_assertions}");
    assert!(drag_rows >= 200, "expected phaze DRAG rows (>=200), got {drag_rows}");
    assert!(phaze_into_spikes_rows >= 100, "expected phaze-into-Spikes DAMAGE rows (>=100), got {phaze_into_spikes_rows}");
    assert!(win_runs >= 100, "expected real game-end WIN runs (>=100), got {win_runs}");

    // THE RANDOM-TARGET PROOF: at least one multi-bench scenario must drag >= 2 DISTINCT
    // species across the seed sweep (else the "random" target isn't proven — a fixed pick
    // would also pass the per-decision checks). The roar/whirlwind multi-bench scenarios do.
    let any_random = distinct_drags.values().any(|set| set.len() >= 2);
    assert!(
        any_random,
        "expected >= 1 scenario where the phaze dragged >= 2 DISTINCT mons across the seed \
         sweep (the random-target proof); got per-scenario distinct drags: {:?}",
        distinct_drags.iter().map(|(k, v)| (k, v.len())).collect::<Vec<_>>()
    );

    eprintln!(
        "phaze golden: {} runs, {dec_assertions} STATE rows, {seed_assertions} seed assertions, \
         {hp_assertions} HP assertions, {spikes_assertions} spikes-layer assertions \
         ({drag_rows} drag decisions, {phaze_into_spikes_rows} phaze-into-spikes-damage rows), \
         {win_runs} wins, {tie_runs} ties; distinct-drag scenarios: {:?}",
        cases.len(),
        distinct_drags.iter().map(|(k, v)| format!("{}={}", k, v.len())).collect::<Vec<_>>()
    );
}
