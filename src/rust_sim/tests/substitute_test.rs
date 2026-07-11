//! SUBSTITUTE full-battle tests — the per-seed PER-DECISION
//! STATE(+HP+STATUS+SPIKES-LAYERS+SUB-HP)+SEED+winner differential that proves the NEW
//! Substitute mechanic matches Showdown EXACTLY, sustained to GAME-END:
//!
//!   **Substitute** (`substitute`) — a self-targeting Status move (never-miss) that spends
//!   `floor(maxhp/4)` HP to create a decoy (`substitute` volatile) with that much HP, which
//!   ABSORBS incoming foe hits until it breaks. The model (verified bit-for-bit vs the
//!   omniscient sim's PRNG probes — `harness/probe_substitute_*.js`):
//!
//!   THE SUBSTITUTE MOVE (`volatileStatus:'substitute'`, `target:'self'`, never-miss):
//!     * NEVER-MISS → NO accuracy draw. FAIL (DRAW-FREE) if a `substitute` is ALREADY present
//!       OR `hp <= floor(maxhp/4)` (can't afford — VERIFIED: hp == floor(maxhp/4) FAILS, +1
//!       SUCCEEDS). SUCCESS: pay `floor(maxhp/4)` HP + create the volatile with that HP.
//!       DRAW-FREE. `landed` FALSE.
//!
//!   A FOE MOVE INTO A SUBSTITUTED MON:
//!     * A DAMAGING move draws acc+crit+damage (UNCHANGED count) and the damage hits the SUB's
//!       HP; the sub BREAKS at 0 (the excess does NOT carry to the mon). THE SECONDARY
//!       draw-COUNT SURPRISE (vs the task's stated assumption): in gen-3 the per-move SECONDARY
//!       `random(100)` is STILL DRAWN against a sub (the same count as a bare hit — the sim
//!       iterates the now-`null` target), but its EFFECT does NOT apply (no status / no
//!       stat-drop / no flinch, AND no confusion `random(2,6)` / Tri-Attack `random(3)`
//!       follow-on). So a damaging move into a sub is SEED-identical to a bare hit but the
//!       STATE (status/boosts) is unchanged — both pinned below.
//!     * A STATUS / stat-DROP move is BLOCKED by the sub (accuracy still drawn, then no effect).
//!     * A CONFUSION self-hit hits the MON, NOT the sub (the self-hit `this.damage` bypasses the
//!       sub-intercept) — the sub HP is unchanged, the mon's HP drops. Draw model unchanged.
//!     * PHAZE (Roar / Whirlwind) BYPASSES the sub — the user is dragged anyway.
//!
//!   `substitute_golden_matches_showdown` — the DIFFERENTIAL gate. For each (scenario, seed)
//!   in `harness/gen_substitute_golden.js`'s golden (FORMAT gen3customgame), seed a
//!   `BattleState` at the sim's PRNG state at the first decision (`init_seed`), run
//!   `run_full_battle(script)` WITHOUT re-seeding, and assert per DECISION BOUNDARY: (a) each
//!   side's post-decision active (species/HP/maxhp/fainted/STATUS + counters) + boosts +
//!   confusion + pokemon_left + the SPIKES LAYERS + THE SUBSTITUTE HP (per side) + request kind
//!   + first mover; AND (b) the post-decision PRNG seed == the sim's `seed_after`. PLUS the
//!   final WINNER. The sub HP column proves create/absorb/break/clear; the mon HP proves the
//!   cost / the absorbed-or-not damage / the confusion-self-hit; the STATUS/BOOSTS columns prove
//!   the secondary is NOT applied behind a sub; the SEED proves the secondary `random(100)` IS
//!   STILL DRAWN (a wrong draw model — skipping it — desyncs HERE).
//!
//! The golden TAB format mirrors the leechseed one: the per-side leechSeeded flags are
//! replaced by the per-side SUBSTITUTE HP (an integer, 0 = no sub) → DEC has 50 fields.

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
    /// The SUBSTITUTE HP at the decision boundary (the sim's `substitute` volatile `hp`), or 0
    /// when there is no sub — the volatile-state proof (create / absorb-drop / break / clear).
    sub_hp: u16,
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
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/substitute_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing substitute golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_substitute_golden.js")
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
                //   p2 out(...)[41..46)  p1Spikes[46] p2Spikes[47]  p1SubHp[48] p2SubHp[49]
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
                    spikes: g(46) as u8,
                    sub_hp: g(48),
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
                    spikes: g(47) as u8,
                    sub_hp: g(49),
                };
                let first_mover = f[35].to_string();
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect { request: req, force, choice, seed_after, p1, p2, first_mover });
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

/// The mid-battle STATE-only injections the sim harness applies at a SPECIFIC decision
/// boundary (the low-HP create-fail boundary + the confusion-on-a-subbed-mon scenario). We
/// reproduce them deterministically by SCENARIO ID + decision index, since they are
/// `injectAt`-driven and NOT recorded as a golden field (they are PURE STATE sets, no PRNG, so
/// the seed parity is unaffected — the golden's seed_after already bakes them in). A function
/// of (scenario, decision, battle) so the Rust run reproduces the IDENTICAL board.
fn apply_inject_at(scen: &str, di: usize, battle: &mut Battle) {
    let state = battle.state_mut().expect("state");
    match scen {
        // create_fail_at_low_hp_boundary: dec 0 → hp = floor(maxhp/4) (FAIL); dec 1 →
        // hp = floor(maxhp/4)+1 (SUCCEED). p1 (Snorlax) is the subber.
        "create_fail_at_low_hp_boundary" => {
            let active = state.sides[0].active;
            let mon = &mut state.sides[0].pokemon[active];
            let maxhp = mon.maxhp;
            if di == 0 {
                mon.hp = maxhp / 4;
            } else if di == 1 {
                mon.hp = maxhp / 4 + 1;
            }
        }
        // confusion_self_hit_hits_the_mon: dec 1 → confuse p1 (Snorlax) once its sub is up.
        "confusion_self_hit_hits_the_mon" => {
            if di == 1 {
                let active = state.sides[0].active;
                let mon = &mut state.sides[0].pokemon[active];
                // addVolatile('confusion') → a random(2,6) onStart duration. The golden's
                // seed_after at THIS boundary bakes that draw in (the sim addVolatile'd it
                // before the move), so we must draw it identically: set the counter from a
                // PRNG draw matching the sim's `addVolatile` onStart (random_range(2,6)).
                if mon.confusion.is_none() {
                    let dur = state.prng.random_range(2, 6) as u8;
                    let active = state.sides[0].active; // re-borrow after the prng draw
                    state.sides[0].pokemon[active].confusion = Some(dur);
                }
            }
        }
        _ => {}
    }
}

/// The decisions in `create_fail_at_low_hp_boundary` / `confusion_self_hit_hits_the_mon` carry
/// a per-decision inject; every other scenario has none. We need to know, when running
/// `run_full_battle`, to interleave the inject at the right boundary. Since `run_full_battle`
/// runs the whole script at once, we instead run those two scenarios DECISION-BY-DECISION
/// (one `ScriptDecision` per `run_full_battle` call would lose the cross-turn state) — so for
/// the injected scenarios we drive a manual per-decision loop; for the rest we run the whole
/// script in one shot (identical result, fewer calls).
fn scenario_has_inject(scen: &str) -> bool {
    matches!(scen, "create_fail_at_low_hp_boundary" | "confusion_self_hit_hits_the_mon")
}

#[test]
fn substitute_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(meta.len() >= 7, "expected >=7 scenarios, got {}", meta.len());
    assert!(cases.len() >= 350, "expected the per-seed corpus (>=350 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut hp_assertions = 0usize;
    let mut sub_state_assertions = 0usize;
    let mut sub_rows = 0usize; // a side had a sub at the boundary
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

        // For inject scenarios we drive decision-by-decision (applying the STATE-only inject at
        // the right boundary, matching the sim's `injectAt`); for the rest one shot.
        let outcome = if scenario_has_inject(&case.scen) {
            run_with_injects(&case.scen, &mut battle, &script, &d)
        } else {
            battle.state_mut().unwrap().run_full_battle(&script, &d)
        };

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
                // HP: the create cost (floor(maxhp/4)), an ABSORBED hit (mon HP unchanged), a
                // CONFUSION self-hit (mon HP drops while the sub is intact), a BROKEN sub
                // (no carry-over) all land in HP. A wrong absorb / break / confusion-target
                // diverges HERE.
                assert_eq!(
                    snap.hp, e.hp,
                    "[{}] dec {} side {} HP mismatch (init_seed {}): got {} exp {}\n  \
                     the Substitute cost / absorb / break / confusion-self-hit target is wrong. \
                     FIX THE MODEL, do not loosen.",
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

                // THE SUBSTITUTE HP: create (floor(maxhp/4)), an absorbed hit (the sub HP
                // DROPS by the dealt damage), a break (→ 0), a switch-out/faint clear (→ 0),
                // and a BLOCKED status / a confusion self-hit (the sub HP is UNCHANGED) all
                // diverge this column. The engine maps None → 0 (a present sub is always >=1).
                let rust_sub = snap.substitute.unwrap_or(0);
                assert_eq!(
                    rust_sub, e.sub_hp,
                    "[{}] dec {} side {} SUB-HP mismatch (init_seed {}): got {} exp {}\n  \
                     the substitute state is wrong (create / absorb-drop / break / clear / \
                     status-block-leaves-it / confusion-leaves-it). FIX THE MODEL, do not loosen.",
                    case.scen, di, idx, case.init_seed, rust_sub, e.sub_hp
                );
                sub_state_assertions += 1;
                if e.sub_hp > 0 {
                    sub_rows += 1;
                }

                if !e.fainted {
                    // STATUS: a secondary BLOCKED by the sub leaves the status `-` (Body Slam
                    // par / Tri Attack into a sub never paralyzes), and a blocked Thunder Wave
                    // / Toxic never statuses. A wrong block (the secondary applied behind the
                    // sub) diverges HERE.
                    assert!(
                        status_variant_eq(snap.status, e.status),
                        "[{}] dec {} side {} STATUS mismatch (init_seed {}): got {:?} exp {:?}\n  \
                         a secondary / status move was wrongly applied behind a Substitute.",
                        case.scen, di, idx, case.init_seed, snap.status, e.status
                    );
                    assert_eq!(
                        status_stage(snap.status), e.stage,
                        "[{}] dec {} side {} STATUS-COUNTER mismatch (init_seed {}): got {} exp {}",
                        case.scen, di, idx, case.init_seed, status_stage(snap.status), e.stage
                    );
                    // BOOSTS: a stat-DROP secondary (Crunch -1 SpD) BLOCKED by the sub leaves
                    // the boosts at 0. A wrong block diverges HERE.
                    assert_eq!(
                        &snap.boosts[0..5], &e.boosts[..],
                        "[{}] dec {} side {} BOOST mismatch (init_seed {}): got {:?} exp {:?}\n  \
                         a stat-drop secondary was wrongly applied behind a Substitute.",
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

            // --- PER-DECISION SEED PARITY (the draw-order+count proof). The SUBSTITUTE move is
            //     DRAW-FREE (never-miss + a draw-free create/fail). A DAMAGING move into a sub
            //     draws the SAME acc+crit+dmg+SECONDARY(100) as a bare hit (the gen-3 secondary
            //     IS still drawn against a sub — the surprise this layer pinned), then NOTHING
            //     further (no confusion random(2,6) / Tri-Attack random(3) behind the sub). A
            //     status move into a sub draws ONLY its accuracy. A confusion self-hit behind a
            //     sub draws randomChance(1,2)+random(16) (unchanged). A wrong draw model
            //     (skipping the secondary random(100), or drawing the suppressed follow-on)
            //     desyncs the LCG HERE. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 the Substitute draw model is wrong. The secondary random(100) MUST still be \
                 drawn against a sub (gen-3), but its effect (and any confusion random(2,6) / \
                 Tri-Attack random(3)) must NOT. FIX THE DRAW MODEL.",
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

    // Coverage floors: the substitute mechanic must actually realize across the corpus.
    assert!(seed_assertions >= 1500, "expected the per-decision seed corpus (>=1500), got {seed_assertions}");
    assert!(hp_assertions >= 1500, "expected per-decision HP assertions (>=1500), got {hp_assertions}");
    assert!(sub_state_assertions >= 1500, "expected per-decision sub-state assertions (>=1500), got {sub_state_assertions}");
    assert!(sub_rows >= 100, "expected substitute STATE rows (>=100), got {sub_rows}");
    assert!(win_runs >= 50, "expected real game-end WIN runs (>=50), got {win_runs}");

    eprintln!(
        "substitute golden: {} runs, {dec_assertions} STATE rows, {seed_assertions} seed assertions, \
         {hp_assertions} HP assertions, {sub_state_assertions} sub-state assertions \
         ({sub_rows} sub rows), {win_runs} wins, {tie_runs} ties",
        cases.len()
    );
}

/// Drive a scenario DECISION-BY-DECISION, applying the STATE-only inject at the right boundary
/// (mirroring the sim harness's `injectAt`), accumulating one `BattleOutcome` whose decisions
/// concatenate every per-decision boundary. Because `run_full_battle` runs a whole script in
/// one call, we feed it ONE `ScriptDecision` at a time across the SAME `BattleState` (the state
/// carries forward), injecting before each call — so the cross-turn state + the seed continuity
/// are preserved EXACTLY as a single run would be, with the board mutation interposed.
fn run_with_injects(
    scen: &str,
    battle: &mut Battle,
    script: &[ScriptDecision],
    d: &Dex,
) -> pokesim::turn::BattleOutcome {
    use pokesim::turn::BattleOutcome;
    let mut all_decisions = Vec::new();
    let mut ended = false;
    let mut winner = None;

    // `run_full_battle` consumes a `move`-request decision per call; a single-decision script
    // runs exactly one move turn (plus any forced-replacement sub-steps it triggers — but the
    // inject scenarios never faint, so each call yields exactly one decision). We apply the
    // inject for boundary `di` BEFORE the di-th call.
    for (di, dec) in script.iter().enumerate() {
        apply_inject_at(scen, di, battle);
        let state = battle.state_mut().expect("state");
        let out = state.run_full_battle(std::slice::from_ref(dec), d);
        all_decisions.extend(out.decisions);
        ended = out.ended;
        winner = out.winner;
        if ended {
            break;
        }
    }
    BattleOutcome { winner, ended, decisions: all_decisions }
}
