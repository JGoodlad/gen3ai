//! GATE ④ (the refusal policy) and the DECLARED LIFECYCLE (M5 Lane 0).
//!
//! * lifecycle: no op before freeze, no STEP before RESET, one freeze, no rebind (counted in
//!   `COLUMN_REBINDS_AFTER_FREEZE`), no unknown opcode — each a typed `LIFECYCLE` status;
//! * caller errors (a team index outside the table, a seed word >= 65536, an action the mask
//!   forbids) are `CALLER` and POISON the pool (every later op is `LIFECYCLE`);
//! * a battle the core REFUSES is QUARANTINED — banked with its typed class and a replayable input
//!   log (also written to `bank_dir`), `done = refused = 1`, `reward = 0`, the env already in its
//!   next episode — and the batch succeeds; over the declared budget it is `BUDGET`.
//!
//! The refusal is a real one: the Phase-A benchmark's banked battle
//! (`rust_core_m5_transport_2026-09-26/repro/hp_tracker_env46.script`, finding F-M5-1 — a
//! transformed mon KO'd by Hidden Power; both the Python and the Rust HP tracker refuse it).

mod common;

use pokesim::json::Json;
use pokesim_env::core::columns::{col, counter, status, ColAddrs};
use pokesim_env::core::{Core, OwnedCols};

const REPRO: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../designs/research_state/measurements/rust_core_m5_transport_2026-09-26/repro/hp_tracker_env46.script"
);

fn small_core(n: usize) -> (Core, OwnedCols) {
    let teams = common::corpus_teams();
    let core = Core::new(common::spec(n, 1, teams)).unwrap();
    let mut cols = OwnedCols::new(n);
    let mut rng = common::Rng(3);
    for i in 0..n {
        common::stage(&mut cols, i, &mut rng, common::corpus_teams().len());
    }
    (core, cols)
}

#[test]
fn the_lifecycle_is_enforced_and_typed() {
    let (mut core, mut cols) = small_core(2);
    let a = cols.addrs();
    assert_eq!(core.dispatch(b'R', a), status::LIFECYCLE, "an op before freeze");
    let mut bad = a;
    bad.0[col::COUNTERS] += 4; // u64 column, 4-aligned only
    assert!(core.freeze(bad).is_err(), "a misaligned column is refused at freeze");
    let mut null = a;
    null.0[col::OBS] = 0;
    assert!(core.freeze(null).is_err(), "a null column is refused at freeze");
    core.freeze(a).unwrap();
    assert!(core.freeze(a).is_err(), "a column set is bound once");
    assert_eq!(core.dispatch(b'S', a), status::LIFECYCLE, "STEP before the first RESET");
    assert_eq!(core.dispatch(b'X', a), status::LIFECYCLE, "an unknown opcode");
    let mut other = OwnedCols::new(2);
    assert_eq!(core.dispatch(b'R', other.addrs()), status::LIFECYCLE, "a rebind");
    assert_eq!(cols.slice::<u64>(col::COUNTERS)[counter::COLUMN_REBINDS_AFTER_FREEZE], 1, "the rebind is COUNTED");
    // A lifecycle refusal does not poison: the bound set still works.
    assert_eq!(core.dispatch(b'R', a), status::OK, "{:?}", core.last_error());
    let need = cols.slice::<u8>(col::NEED);
    assert!(need.chunks(2).all(|e| e.iter().any(|&x| x == 1)), "every env has a decision open after RESET");
}

#[test]
fn caller_errors_are_typed_and_poison_the_pool() {
    // A team index outside the table.
    let (mut core, mut cols) = small_core(2);
    let a = cols.addrs();
    core.freeze(a).unwrap();
    cols.slice_mut::<u32>(col::EP_TEAM)[3] = 10_000;
    assert_eq!(core.dispatch(b'R', a), status::CALLER);
    let e = core.last_error().unwrap();
    assert_eq!(e.env, Some(1));
    assert!(e.message.contains("outside the declared team table"), "{}", e.message);
    assert_eq!(core.dispatch(b'R', a), status::LIFECYCLE, "a failed batch POISONS the pool");
    assert!(core.last_error().unwrap().message.contains("POISONED"));

    // A seed word out of range.
    let (mut core, mut cols) = small_core(1);
    let a = cols.addrs();
    core.freeze(a).unwrap();
    cols.slice_mut::<u32>(col::EP_SEED)[2] = 65536;
    assert_eq!(core.dispatch(b'R', a), status::CALLER);

    // An action the mask forbids: the error carries the env's input log.
    let (mut core, mut cols) = small_core(1);
    let a = cols.addrs();
    core.freeze(a).unwrap();
    assert_eq!(core.dispatch(b'R', a), status::OK);
    let side = if cols.slice::<u8>(col::NEED)[0] == 1 { 0 } else { 1 };
    let illegal = (0..11).find(|&k| cols.slice::<u8>(col::MASK)[side * 11 + k] == 0).expect("some action is masked") as i32;
    cols.slice_mut::<i32>(col::ACTION)[side] = illegal;
    assert_eq!(core.dispatch(b'S', a), status::CALLER);
    let e = core.last_error().unwrap();
    assert!(e.message.contains("not legal here"), "{}", e.message);
    assert!(e.script.as_deref().is_some_and(|s| s.starts_with("START {")), "the input log rides with the error");
    let j = Json::parse(&e.json()).expect("the error's json parses");
    assert_eq!(j.get("status").and_then(Json::as_f64), Some(status::CALLER as f64));
}

#[test]
fn a_bad_team_fails_at_startup_not_mid_run() {
    let mut teams = common::corpus_teams();
    teams.push("Notamonatall|||||||||||".into());
    let e = Core::new(common::spec(1, 1, teams)).err().expect("a team that does not unpack is refused at startup");
    assert!(e.contains("the engine refuses it"), "{e}");
    eprintln!("startup refusal: {e}");
}

struct Repro {
    teams: [String; 2],
    seed: [u32; 4],
    cmds: Vec<String>,
}

fn repro() -> Repro {
    let text = std::fs::read_to_string(REPRO).expect("the banked repro script");
    let mut lines = text.lines();
    let start = Json::parse(lines.next().unwrap().strip_prefix("START ").unwrap()).unwrap();
    let team = |k: &str| start.get(k).and_then(|p| p.str_at("team")).unwrap().to_string();
    let seed: Vec<u32> = start.str_at("seed").unwrap().split(',').map(|w| w.parse().unwrap()).collect();
    Repro {
        teams: [team("p1"), team("p2")],
        seed: seed.try_into().unwrap(),
        cmds: lines.filter(|l| !l.is_empty() && *l != "END").map(str::to_string).collect(),
    }
}

/// Drive a one-env inline core through the repro's commands; returns the status of the step that
/// consumed the last command, the core and its columns.
fn replay_repro(budget: usize, bank_dir: Option<std::path::PathBuf>) -> (i32, Core, OwnedCols, Vec<String>, Vec<[i32; 2]>) {
    let r = repro();
    let mut spec = common::spec(1, 1, vec![r.teams[0].clone(), r.teams[1].clone()]);
    spec.refusal_budget = budget;
    spec.bank_dir = bank_dir;
    let mut core = Core::new(spec).unwrap();
    let mut cols = OwnedCols::new(1);
    cols.slice_mut::<u32>(col::EP_TEAM).copy_from_slice(&[0, 1]);
    cols.slice_mut::<u32>(col::EP_SEED).copy_from_slice(&r.seed);
    let a: ColAddrs = cols.addrs();
    core.freeze(a).unwrap();
    assert_eq!(core.dispatch(b'R', a), status::OK);
    let mut q = r.cmds.clone().into_iter().peekable();
    let mut fed = Vec::new();
    let mut steps: Vec<[i32; 2]> = Vec::new();
    loop {
        let env = core.inline_env(0).unwrap();
        let mut acts = [-1i32; 2];
        for side in 0..2 {
            if cols.slice::<u8>(col::NEED)[side] == 1 {
                let c = q.next().expect("the script ran out before the refusal");
                let tok = c.strip_prefix(&format!("CHOOSE p{} ", side + 1)).unwrap_or_else(|| panic!("expected p{} next, got {c}", side + 1));
                let idx = env.open_tokens(side).unwrap().iter().find(|(_, t)| t == tok).unwrap_or_else(|| panic!("{tok} is not legal")).0;
                acts[side] = idx as i32;
                fed.push(c);
            }
        }
        cols.slice_mut::<i32>(col::ACTION).copy_from_slice(&acts);
        steps.push(acts);
        let st = core.dispatch(b'S', a);
        if st != status::OK || q.peek().is_none() || cols.slice::<u8>(col::DONE)[0] == 1 {
            return (st, core, cols, fed, steps);
        }
    }
}

#[test]
fn a_refused_battle_is_quarantined_banked_and_replayable() {
    let dir = std::env::temp_dir().join(format!("rust_env_bank_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let (st, core, cols, fed, _) = replay_repro(4, Some(dir.clone()));
    assert_eq!(st, status::OK, "a quarantine is not a batch failure: {:?}", core.last_error());
    assert_eq!(cols.slice::<u8>(col::DONE)[0], 1);
    assert_eq!(cols.slice::<u8>(col::REFUSED)[0], 1, "the quarantine is told apart from a tie by `refused`");
    assert_eq!(cols.slice::<f32>(col::REWARD)[0], 0.0);
    assert_eq!(cols.slice::<u32>(col::EPISODE)[0], 1, "the env is already in its next episode");
    assert_eq!(cols.slice::<u64>(col::COUNTERS)[counter::REFUSALS], 1);
    assert_eq!(core.bank().len(), 1);
    let b = &core.bank().items()[0];
    // F-M5-1: the port types this elimination a `CoreError::Fault` (Python raises ValueError) —
    // banked with kind `fault`, so a port bug stays distinguishable from a poke-env-class refusal.
    assert_eq!(b.error.kind, "fault", "{:?}", b.error);
    assert!(b.error.message.contains("all candidates eliminated"), "{:?}", b.error);
    // The banked input log IS the battle up to the refusal: the same START and the same commands.
    let script = b.log.script("");
    let banked_cmds: Vec<&str> = script.lines().skip(1).collect();
    assert_eq!(banked_cmds, fed.iter().map(String::as_str).collect::<Vec<_>>());
    let written: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.file_name().into_string().unwrap()).collect();
    assert!(written.iter().any(|f| f.ends_with("_fault.script")) && written.iter().any(|f| f.ends_with("_fault.json")), "{written:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_quarantine_over_the_declared_budget_is_a_budget_failure() {
    let (st, core, _cols, _, _) = replay_repro(0, None);
    assert_eq!(st, status::BUDGET, "{:?}", core.last_error());
    let e = core.last_error().unwrap();
    assert!(e.message.contains("refusal budget") && e.script.is_some(), "{}", e.message);
    assert_eq!(core.counters()[counter::BANK_GROWTH_AFTER_FREEZE], 0);
}

/// Gate ② WITH quarantines (the random-policy determinism corpus has none): N envs all replay the
/// refused battle's action indices, so every env quarantines at the same step — the bank (its order,
/// its records) and every column are identical at 1, 3 and N worker threads.
#[test]
fn quarantines_are_thread_count_invariant() {
    let r = repro();
    let (_, _, _, _, steps) = replay_repro(4, None);
    let run = |threads: usize| {
        let n = 6;
        let mut spec = common::spec(n, threads, vec![r.teams[0].clone(), r.teams[1].clone()]);
        spec.refusal_budget = 2 * n;
        let mut core = Core::new(spec).unwrap();
        let mut cols = OwnedCols::new(n);
        for i in 0..n {
            cols.slice_mut::<u32>(col::EP_TEAM)[2 * i..2 * i + 2].copy_from_slice(&[0, 1]);
            cols.slice_mut::<u32>(col::EP_SEED)[4 * i..4 * i + 4].copy_from_slice(&r.seed);
        }
        let a = cols.addrs();
        core.freeze(a).unwrap();
        assert_eq!(core.dispatch(b'R', a), status::OK);
        let mut digests = vec![common::digest(&cols)];
        for acts in &steps {
            for i in 0..n {
                cols.slice_mut::<i32>(col::ACTION)[2 * i..2 * i + 2].copy_from_slice(acts);
            }
            assert_eq!(core.dispatch(b'S', a), status::OK, "{:?}", core.last_error());
            digests.push(common::digest(&cols));
        }
        let bank: Vec<String> = core.bank().items().iter().map(|b| b.json()).collect();
        (digests, bank, cols.slice::<u8>(col::REFUSED).to_vec())
    };
    let base = run(1);
    assert_eq!(base.1.len(), 6, "every env quarantined once");
    assert!(base.2.iter().all(|&x| x == 1));
    for t in [3, 6] {
        let got = run(t);
        assert_eq!(got.0, base.0, "threads = {t}: the columns differ");
        assert_eq!(got.1, base.1, "threads = {t}: the bank differs (order or content)");
    }
}
