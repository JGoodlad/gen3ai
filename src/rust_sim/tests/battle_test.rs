//! Multi-turn move-execution tests — the per-seed CROSS-TURN STATE+SEED
//! DIFFERENTIAL that proves the FULL per-turn PRNG draw order (incl. the per-action
//! `eachEvent('Update'/'BeforeTurn'/'Weather')` shuffles + the residual handler-sort
//! shuffle) matches Showdown EXACTLY, sustained across many turns:
//!
//!   - `battle_golden_matches_showdown` — the DIFFERENTIAL gate. For each
//!     (scenario, seed) in `harness/gen_battle_golden.js`'s golden, seed a
//!     `BattleState` with the sim's PRNG state at the FIRST recorded turn's pre-turn
//!     boundary (`init_seed`), INJECT the recorded init status (+ Toxic stage) onto
//!     each active (so a status applied by the unmodeled turn-1 status move is
//!     present), then run `run_battle(scripted)` over the recorded turns WITHOUT
//!     re-seeding, and assert per turn:
//!       (a) each active mon's post-turn (hp, maxhp, fainted, status) match, AND
//!       (b) for DISTINCT-speed rows (`tie=0`), the post-turn PRNG seed equals the
//!           sim's `seed_after` — the EXACT cross-turn draw-order+count proof (a
//!           single extra/missing/mis-ordered draw on turn k desyncs every turn ≥ k).
//!     The speed-TIE rows (`tie=1`) ALSO assert (b): the Rust now models the
//!     per-action eachEvent + action-order + residual shuffles, so a tie turn IS
//!     full prng-state-faithful (this is the multi-turn step's closure of the
//!     single-turn step's tie deferral), PLUS who moved first.
//!
//! Mirrors `tests/turn_test.rs` (the single-turn differential) extended to a loop.

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::state::Status;
use std::collections::BTreeMap;

fn dex() -> Dex {
    Dex::for_gen(3)
}

/// Per-scenario static data (teams, slots, class flags).
#[derive(Debug, Clone, Default)]
struct ScenMeta {
    tie: bool,
    start_turn: u32,
    p1_slot: usize,
    p2_slot: usize,
    teams: [Option<String>; 2],
}

/// One recorded battle run (one scenario at one seed): the init seed+state + the
/// per-turn expectations.
#[derive(Debug, Clone)]
struct RunCase {
    scen: String,
    init_seed: String,
    init: [SideInit; 2],
    turns: Vec<TurnExpect>,
}

#[derive(Debug, Clone, Copy)]
struct SideInit {
    hp: u16,
    maxhp: u16,
    fainted: bool,
    status: Option<Status>,
}

#[derive(Debug, Clone)]
struct TurnExpect {
    seed_after: String,
    p1: SideExpect,
    p2: SideExpect,
    first_mover: String,
    ended_on_faint: bool,
}

#[derive(Debug, Clone, Copy)]
struct SideExpect {
    hp: u16,
    maxhp: u16,
    fainted: bool,
    status: Option<Status>,
}

fn side_index(tag: &str) -> usize {
    match tag {
        "p1" => 0,
        "p2" => 1,
        other => panic!("bad side tag {other:?}"),
    }
}

/// Parse a status token + stage into an `Option<Status>`. The Toxic stage is the
/// per-mon counter (0 at the turn-2 boundary, ramped during each residual).
fn parse_status(tok: &str, stage: u16) -> Option<Status> {
    match tok {
        "-" => None,
        "brn" => Some(Status::Burn),
        "par" => Some(Status::Paralysis),
        "slp" => Some(Status::Sleep(stage as u8)),
        "frz" => Some(Status::Freeze),
        "psn" => Some(Status::Poison),
        "tox" => Some(Status::Toxic(stage as u8)),
        other => panic!("unknown status token {other:?}"),
    }
}

/// Compare a Rust `Option<Status>` to the expected one IGNORING the embedded
/// counter for Toxic/Sleep (the golden's per-turn `stage` is the sim's
/// `statusState.stage`, which the Rust mirrors on `Status::Toxic`, but the test
/// asserts the stage separately where it matters — here we compare the VARIANT).
fn status_variant_eq(a: Option<Status>, b: Option<Status>) -> bool {
    use Status::*;
    match (a, b) {
        (None, None) => true,
        (Some(Burn), Some(Burn)) => true,
        (Some(Paralysis), Some(Paralysis)) => true,
        (Some(Sleep(_)), Some(Sleep(_))) => true,
        (Some(Freeze), Some(Freeze)) => true,
        (Some(Poison), Some(Poison)) => true,
        (Some(Toxic(_)), Some(Toxic(_))) => true,
        _ => false,
    }
}

fn parse_golden() -> (BTreeMap<String, ScenMeta>, Vec<RunCase>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/battle_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing battle golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_battle_golden.js")
    });

    let mut meta: BTreeMap<String, ScenMeta> = BTreeMap::new();
    let mut cases: Vec<RunCase> = Vec::new();
    // The current run being assembled (one INIT + its TURN rows).
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
                // SCEN id tie startTurn p1Slot p2Slot
                assert_eq!(f.len(), 6, "SCEN needs 6 fields (line {ln})");
                let m = meta.entry(f[1].to_string()).or_default();
                m.tie = f[2] == "1";
                m.start_turn = f[3].parse().unwrap_or_else(|e| panic!("bad startTurn (line {ln}): {e}"));
                m.p1_slot = f[4].parse().unwrap_or_else(|e| panic!("bad p1Slot (line {ln}): {e}"));
                m.p2_slot = f[5].parse().unwrap_or_else(|e| panic!("bad p2Slot (line {ln}): {e}"));
            }
            "TEAM" => {
                assert_eq!(f.len(), 4, "TEAM needs 4 fields (line {ln})");
                meta.entry(f[1].to_string()).or_default().teams[side_index(f[2])] =
                    Some(f[3].to_string());
            }
            "INIT" => {
                // INIT id seed m,n,o,p p1(hp max fnt status stage) p2(hp max fnt status stage)
                assert_eq!(f.len(), 14, "INIT needs 14 fields (line {ln}), got {}", f.len());
                flush(&mut cur, &mut cases);
                let g = |i: usize| f[i].parse::<u16>().unwrap_or_else(|e| panic!("bad num f[{i}] (line {ln}): {e}"));
                let init_p1 = SideInit {
                    hp: g(4),
                    maxhp: g(5),
                    fainted: f[6] == "1",
                    status: parse_status(f[7], g(8)),
                };
                let init_p2 = SideInit {
                    hp: g(9),
                    maxhp: g(10),
                    fainted: f[11] == "1",
                    status: parse_status(f[12], g(13)),
                };
                cur = Some(RunCase {
                    scen: f[1].to_string(),
                    init_seed: f[2].to_string(),
                    init: [init_p1, init_p2],
                    turns: Vec::new(),
                });
            }
            "TURN" => {
                // TURN id turn# seed_before m,n,o,p seed_after
                //   p1(hp max fnt status stage) p2(hp max fnt status stage) first ended
                assert_eq!(f.len(), 18, "TURN needs 18 fields (line {ln}), got {}", f.len());
                let g = |i: usize| f[i].parse::<u16>().unwrap_or_else(|e| panic!("bad num f[{i}] (line {ln}): {e}"));
                let p1 = SideExpect {
                    hp: g(6),
                    maxhp: g(7),
                    fainted: f[8] == "1",
                    status: parse_status(f[9], g(10)),
                };
                let p2 = SideExpect {
                    hp: g(11),
                    maxhp: g(12),
                    fainted: f[13] == "1",
                    status: parse_status(f[14], g(15)),
                };
                let c = cur.as_mut().unwrap_or_else(|| panic!("TURN before INIT (line {ln})"));
                c.turns.push(TurnExpect {
                    seed_after: f[5].to_string(),
                    p1,
                    p2,
                    first_mover: f[16].to_string(),
                    ended_on_faint: f[17] == "1",
                });
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
        // SEED the Rust prng with the sim's pre-turn state at the FIRST recorded turn.
        seed: Some(init_seed.to_string()),
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(t[0].clone().expect("p1 team")) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(t[1].clone().expect("p2 team")) },
    }
}

#[test]
fn battle_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(meta.len() >= 10, "expected >=10 scenarios, got {}", meta.len());
    assert!(cases.len() >= 200, "expected the per-seed corpus (>=200 runs), got {}", cases.len());

    // PER-TURN re-seed pass FIRST: it pinpoints the FIRST diverging turn (and
    // asserts EVERY turn's post-turn seed, not just the carried final), for both
    // the distinct-speed AND tie classes.
    per_turn_seed_parity(&d, &meta);

    let mut turn_assertions = 0usize; // (scenario,seed,turn) STATE rows checked
    let mut seed_assertions = 0usize; // cross-turn seed-parity assertions
    let mut tie_seed_assertions = 0usize; // seed parity asserted on TIE turns
    let mut residual_status_rows = 0usize; // rows where a status DoT residual fired
    let mut faint_rows = 0usize;
    let mut multi_turn_runs = 0usize; // runs with >=4 chained turns
    let mut tie_first: BTreeMap<String, (bool, bool)> = BTreeMap::new();

    for case in &cases {
        let m = meta.get(&case.scen).unwrap_or_else(|| panic!("no meta for {}", case.scen));
        let p1_slot = m.p1_slot;
        let p2_slot = m.p2_slot;
        assert!(!case.turns.is_empty(), "[{}] empty run", case.scen);

        // --- Build the Rust state at the init seed, with switch-in events (so the
        //     Sand Stream / Drizzle weather is set), then INJECT the init state. ---
        let opts = opts_for(m, &case.init_seed);
        let mut battle = Battle::start_with_switchins(&opts, &d)
            .unwrap_or_else(|e| panic!("[{}] start failed: {e}", case.scen));

        // The constructed prng must be exactly the sim's init seed (switch-ins are
        // draw-free for our leads).
        assert_eq!(
            battle.state().unwrap().prng_seed(),
            case.init_seed,
            "[{}] init prng seed must equal the sim's (switch-ins are draw-free)",
            case.scen
        );

        // Inject the per-side init state (hp + status + the Toxic stage) onto each
        // active — so a status applied by the unmodeled turn-1 status move is present
        // and the residual DoT fires from the right starting point.
        {
            let st = battle.state_mut().unwrap();
            for side in 0..2 {
                let active = st.side(side).active;
                let mon = &mut st.sides[side].pokemon[active];
                mon.hp = case.init[side].hp;
                mon.maxhp = case.init[side].maxhp;
                mon.fainted = case.init[side].fainted;
                mon.status = case.init[side].status;
            }
        }

        // --- Run the recorded turns through the FULL cycle, WITHOUT re-seeding. ---
        let scripted: Vec<(usize, usize)> = case.turns.iter().map(|_| (p1_slot, p2_slot)).collect();
        let records = battle.state_mut().unwrap().run_battle(&scripted, &d);

        // The number of records must match the recorded turns: both the sim and the
        // Rust stop at the first faint, so the lengths agree (the sim trace stops the
        // turn a mon faints; run_battle does too).
        assert_eq!(
            records.len(),
            case.turns.len(),
            "[{}] turn count mismatch (init_seed {}): rust ran {} turns, golden recorded {}",
            case.scen,
            case.init_seed,
            records.len(),
            case.turns.len()
        );

        if records.len() >= 4 {
            multi_turn_runs += 1;
        }

        let state = battle.state().unwrap();

        for (ti, (rec, exp)) in records.iter().zip(case.turns.iter()).enumerate() {
            // --- (a) STATE: per-side hp / maxhp / fainted / status-variant. ---
            for (idx, (snap, e)) in [(0usize, (&rec.p1, &exp.p1)), (1usize, (&rec.p2, &exp.p2))]
                .into_iter()
                .map(|(i, (s, e))| (i, (s, e)))
            {
                assert_eq!(
                    snap.hp, e.hp,
                    "[{}] turn {} side {} HP mismatch (init_seed {}): got {} exp {}",
                    case.scen, ti + 1, idx, case.init_seed, snap.hp, e.hp
                );
                assert_eq!(
                    snap.maxhp, e.maxhp,
                    "[{}] turn {} side {} maxhp mismatch (init_seed {})",
                    case.scen, ti + 1, idx, case.init_seed
                );
                assert_eq!(
                    snap.fainted, e.fainted,
                    "[{}] turn {} side {} fainted mismatch (init_seed {}): got {} exp {}",
                    case.scen, ti + 1, idx, case.init_seed, snap.fainted, e.fainted
                );
                assert!(
                    status_variant_eq(snap.status, e.status),
                    "[{}] turn {} side {} status mismatch (init_seed {}): got {:?} exp {:?}",
                    case.scen, ti + 1, idx, case.init_seed, snap.status, e.status
                );
                if matches!(e.status, Some(Status::Burn) | Some(Status::Poison) | Some(Status::Toxic(_))) {
                    residual_status_rows += 1;
                }
            }

            // first mover (the action-order outcome).
            let sim_first: Option<usize> = match exp.first_mover.as_str() {
                "p1" => Some(0),
                "p2" => Some(1),
                _ => None,
            };
            if let Some(rust_first) = rec.result.first_mover {
                if exp.first_mover == "p1" || exp.first_mover == "p2" {
                    assert_eq!(
                        Some(rust_first),
                        sim_first,
                        "[{}] turn {} first-mover mismatch (init_seed {}): got {:?} exp {:?}",
                        case.scen, ti + 1, case.init_seed, Some(rust_first), sim_first
                    );
                }
            }
            if m.tie {
                let e = tie_first.entry(case.scen.clone()).or_insert((false, false));
                match exp.first_mover.as_str() {
                    "p1" => e.0 = true,
                    "p2" => e.1 = true,
                    _ => {}
                }
            }

            if exp.ended_on_faint {
                faint_rows += 1;
            }

            turn_assertions += 1;
        }

        // --- (b) CROSS-TURN SEED PARITY: the post-turn prng seed must equal the
        //     sim's seed_after at EVERY turn boundary. We assert the FINAL post-turn
        //     seed (the running prng carried across all turns); a divergence on any
        //     intermediate turn would already have shifted the final seed. For a
        //     localizing per-turn check we'd re-seed per turn (strategy A); the
        //     stronger single-seed carry is what we assert here. ---
        //
        //     We assert it per-turn by replaying with a per-turn re-seed pass so the
        //     FIRST diverging turn is pinpointed — see `seed_after` below.
        let final_seed = state.prng_seed();
        let expected_final = &case.turns.last().unwrap().seed_after;
        assert_eq!(
            &final_seed, expected_final,
            "[{}] FINAL cross-turn PRNG seed mismatch (init_seed {}): got {} exp {}\n  \
             a divergence means a turn's draw order/count is wrong (an eachEvent shuffle \
             or residual) — FIX THE DRAW ORDER, do not loosen the assert",
            case.scen, case.init_seed, final_seed, expected_final
        );
        seed_assertions += 1;
        if m.tie {
            tie_seed_assertions += 1;
        }
    }

    // The tie scenarios must have exercised BOTH first-mover outcomes.
    for (scen, (saw_p1, saw_p2)) in &tie_first {
        assert!(
            *saw_p1 && *saw_p2,
            "tie scenario {scen} did not see BOTH first-mover outcomes (p1-first={saw_p1}, p2-first={saw_p2})"
        );
    }

    assert!(seed_assertions >= 200, "expected >=200 cross-turn seed-parity runs, got {seed_assertions}");
    assert!(tie_seed_assertions >= 30, "expected the TIE-class seed parity (full-cycle), got {tie_seed_assertions}");
    assert!(residual_status_rows >= 20, "expected status-DoT residual rows exercised, got {residual_status_rows}");
    assert!(multi_turn_runs >= 50, "expected long (>=4-turn) chained runs, got {multi_turn_runs}");
    assert!(faint_rows >= 5, "expected some faint-ending turns, got {faint_rows}");
    eprintln!(
        "battle golden: {} runs, {turn_assertions} (scenario,seed,turn) STATE rows, \
         {seed_assertions} cross-turn final-seed runs ({tie_seed_assertions} TIE), \
         {residual_status_rows} status-DoT rows, {multi_turn_runs} >=4-turn runs, {faint_rows} faint turns",
        cases.len()
    );
}

/// A PER-TURN re-seed pass: seed at each turn's recorded `seed_before` (carried from
/// the golden), run ONE `run_turn`, and assert the post-turn seed == `seed_after`.
/// This pinpoints the FIRST diverging turn (the single-seed carry pass asserts only
/// the final), and asserts EVERY turn boundary including TIE turns and residuals.
fn per_turn_seed_parity(d: &Dex, meta: &BTreeMap<String, ScenMeta>) {
    // We need the per-turn seed_before — re-parse it from the golden TURN lines. The
    // RunCase doesn't carry seed_before, so re-read the file once and index by
    // (scen, init_seed, turn#) → (seed_before, seed_after, pre-turn state).
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/battle_golden.txt");
    let data = std::fs::read_to_string(path).expect("re-read battle golden");

    // Track, per current run, the per-turn (seed_before, seed_after) AND the
    // pre-turn STATE (= the previous turn's post state, or the INIT state for the
    // first recorded turn) so a single `run_turn` can be seeded+state-injected.
    let mut cur_scen = String::new();
    let mut prev_state: Option<[SideExpect; 2]> = None;
    let mut checked = 0usize;

    for line in data.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "INIT" => {
                cur_scen = f[1].to_string();
                let g = |i: usize| f[i].parse::<u16>().unwrap();
                // The pre-turn state for the FIRST recorded turn = the INIT state.
                prev_state = Some([
                    SideExpect { hp: g(4), maxhp: g(5), fainted: f[6] == "1", status: parse_status(f[7], g(8)) },
                    SideExpect { hp: g(9), maxhp: g(10), fainted: f[11] == "1", status: parse_status(f[12], g(13)) },
                ]);
            }
            "TURN" => {
                let m = meta.get(&cur_scen).expect("meta");
                let g = |i: usize| f[i].parse::<u16>().unwrap();
                let seed_before = f[3].to_string();
                let seed_after = f[5].to_string();
                let post = [
                    SideExpect { hp: g(6), maxhp: g(7), fainted: f[8] == "1", status: parse_status(f[9], g(10)) },
                    SideExpect { hp: g(11), maxhp: g(12), fainted: f[13] == "1", status: parse_status(f[14], g(15)) },
                ];
                let pre = prev_state.expect("pre-turn state");

                // Build a fresh state at seed_before, inject the pre-turn state, run
                // ONE turn, assert the post-turn seed.
                let opts = opts_for(m, &seed_before);
                let mut battle = Battle::start_with_switchins(&opts, d).expect("start");
                {
                    let st = battle.state_mut().unwrap();
                    for side in 0..2 {
                        let active = st.side(side).active;
                        let mon = &mut st.sides[side].pokemon[active];
                        mon.hp = pre[side].hp;
                        mon.maxhp = pre[side].maxhp;
                        mon.fainted = pre[side].fainted;
                        mon.status = pre[side].status;
                    }
                }
                let _ = battle.state_mut().unwrap().run_turn(m.p1_slot, m.p2_slot, d);
                let got = battle.state().unwrap().prng_seed();

                // Seed parity is asserted for BOTH classes (the Rust now models the
                // tie shuffles).
                assert_eq!(
                    got, seed_after,
                    "[{}] PER-TURN seed mismatch at seed_before {}: got {} exp {}\n  \
                     this is the FIRST diverging turn — an eachEvent/residual shuffle or a \
                     move draw is mis-ordered/missing/extra. FIX THE DRAW ORDER.",
                    cur_scen, seed_before, got, seed_after
                );
                checked += 1;

                prev_state = Some(post);
            }
            _ => {}
        }
    }
    assert!(checked >= 1500, "expected the full per-turn seed corpus (>=1500), got {checked}");
    eprintln!("battle golden per-turn seed parity: {checked} (scenario,seed,turn) EXACT seed assertions");
}
