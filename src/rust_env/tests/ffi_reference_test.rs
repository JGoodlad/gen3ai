//! Lane A gate ①'s IN-RUST half — the core's run on Lane 0's gate-① corpus, RECORDED so the FFI
//! front end can replay the same inputs and be compared byte for byte (M5 Lane A).
//!
//! IGNORED: it is a recorder driven by `src/utils/rust_env/ffi_integration_test.py`, which sets
//! `RUST_ENV_FFI_REF_OUT` (a directory) and, for the ladder tier, the same `RUST_ENV_PARITY_TEAMS` /
//! `_N` / `_STEPS` / `_SEED` knobs as gate ① (`sim_bridge_parity_test.rs`). The run is gate ①'s own
//! (the same staging RNG, the same seeded random policy, one inline pool). Written:
//!
//! * `spec.json` — `Spec::to_json` of the run (threads = 1);
//! * `trace.bin` — per op: the opcode byte, then every INPUT column's bytes as the core saw them,
//!   then every OUTPUT column's bytes after it (table order, `counters` included);
//! * `bank.json` — the refusal bank (`Banked::json`, a JSON array).
//!
//! A second recorder replays Lane 0's banked F-M5-1 refusal (`hp_tracker_env46.script`, the battle
//! `lifecycle_refusal_test.rs` quarantines) and plays on after it, so a real QUARANTINE — `done`,
//! `refused`, the auto-reset and the bank — crosses the FFI too.

mod common;

use std::io::Write;

use pokesim::json::Json;
use pokesim_env::core::columns::{col, col_bytes, Dir, COLUMNS};
use pokesim_env::core::{Core, OwnedCols};

fn env_or(key: &str, default: u64) -> u64 {
    std::env::var(key).map(|v| v.parse().unwrap_or_else(|_| panic!("{key}={v:?} is not an integer"))).unwrap_or(default)
}

fn write_cols(w: &mut impl Write, cols: &OwnedCols, dir: Dir) {
    let b = col_bytes(cols.n);
    for (i, c) in COLUMNS.iter().enumerate() {
        if c.dir == dir {
            let bytes = cols.bytes(i);
            assert_eq!(bytes.len(), b[i]);
            w.write_all(bytes).unwrap();
        }
    }
}

#[test]
#[ignore]
fn record_the_gate_1_run_for_the_ffi_front_end() {
    let out = std::path::PathBuf::from(std::env::var("RUST_ENV_FFI_REF_OUT").expect("RUST_ENV_FFI_REF_OUT (a directory)"));
    std::fs::create_dir_all(&out).unwrap();
    let teams: Vec<String> = match std::env::var("RUST_ENV_PARITY_TEAMS") {
        Ok(f) => std::fs::read_to_string(&f).unwrap_or_else(|e| panic!("{f}: {e}")).lines().filter(|l| !l.is_empty()).map(str::to_string).collect(),
        Err(_) => common::corpus_teams(),
    };
    let nt = teams.len();
    let n = env_or("RUST_ENV_PARITY_N", 6) as usize;
    let steps = env_or("RUST_ENV_PARITY_STEPS", 400) as usize;
    let seed = env_or("RUST_ENV_PARITY_SEED", 11);
    let mut spec = common::spec(n, 1, teams);
    spec.refusal_budget = n * steps;
    std::fs::write(out.join("spec.json"), spec.to_json()).unwrap();
    let mut core = Core::new(spec).unwrap();
    let mut cols = OwnedCols::new(n);
    let mut stage_rng = common::Rng(0x0B5E ^ seed);
    let mut act_rng = common::Rng(seed);
    for i in 0..n {
        common::stage(&mut cols, i, &mut stage_rng, nt);
    }
    let a = cols.addrs();
    core.freeze(a).unwrap();
    let mut w = std::io::BufWriter::new(std::fs::File::create(out.join("trace.bin")).unwrap());
    let mut op = |core: &mut Core, cols: &mut OwnedCols, code: u8, w: &mut std::io::BufWriter<std::fs::File>| {
        w.write_all(&[code]).unwrap();
        write_cols(w, cols, Dir::In);
        assert_eq!(core.dispatch(code, a), 0, "{:?}", core.last_error().map(|e| e.json()));
        write_cols(w, cols, Dir::Out);
    };
    op(&mut core, &mut cols, b'R', &mut w);
    for _ in 0..steps {
        for i in 0..n {
            common::stage(&mut cols, i, &mut stage_rng, nt);
        }
        common::random_actions(&mut cols, &mut act_rng);
        op(&mut core, &mut cols, b'S', &mut w);
    }
    w.flush().unwrap();
    let bank: Vec<String> = core.bank().items().iter().map(|b| b.json()).collect();
    std::fs::write(out.join("bank.json"), format!("[{}]", bank.join(","))).unwrap();
    eprintln!("ffi reference: {} ops, n {n}, {nt} teams, {} banked", steps + 1, bank.len());
}

const REPRO: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../designs/research_state/measurements/rust_core_m5_transport_2026-09-26/repro/hp_tracker_env46.script"
);

#[test]
#[ignore]
fn record_the_quarantine_run_for_the_ffi_front_end() {
    let out = std::path::PathBuf::from(std::env::var("RUST_ENV_FFI_REF_OUT").expect("RUST_ENV_FFI_REF_OUT (a directory)"));
    std::fs::create_dir_all(&out).unwrap();
    let text = std::fs::read_to_string(REPRO).expect("the banked repro script");
    let mut lines = text.lines();
    let start = Json::parse(lines.next().unwrap().strip_prefix("START ").unwrap()).unwrap();
    let team = |k: &str| start.get(k).and_then(|p| p.str_at("team")).unwrap().to_string();
    let seed: Vec<u32> = start.str_at("seed").unwrap().split(',').map(|w| w.parse().unwrap()).collect();
    let mut q = lines.filter(|l| !l.is_empty() && *l != "END").map(str::to_string).peekable();
    let mut spec = common::spec(1, 1, vec![team("p1"), team("p2")]);
    spec.refusal_budget = 4;
    std::fs::write(out.join("spec.json"), spec.to_json()).unwrap();
    let mut core = Core::new(spec).unwrap();
    let mut cols = OwnedCols::new(1);
    cols.slice_mut::<u32>(col::EP_TEAM).copy_from_slice(&[0, 1]);
    cols.slice_mut::<u32>(col::EP_SEED).copy_from_slice(&seed);
    let a = cols.addrs();
    core.freeze(a).unwrap();
    let mut w = std::io::BufWriter::new(std::fs::File::create(out.join("trace.bin")).unwrap());
    let mut op = |core: &mut Core, cols: &mut OwnedCols, code: u8, w: &mut std::io::BufWriter<std::fs::File>| {
        w.write_all(&[code]).unwrap();
        write_cols(w, cols, Dir::In);
        assert_eq!(core.dispatch(code, a), 0, "{:?}", core.last_error().map(|e| e.json()));
        write_cols(w, cols, Dir::Out);
    };
    op(&mut core, &mut cols, b'R', &mut w);
    // The refused battle, its commands mapped to action indices as the lifecycle test does.
    while cols.slice::<u8>(col::REFUSED)[0] == 0 {
        let env = core.inline_env(0).unwrap();
        let mut acts = [-1i32; 2];
        for side in 0..2 {
            if cols.slice::<u8>(col::NEED)[side] == 1 {
                let c = q.next().expect("the script ran out before the refusal");
                let tok = c.strip_prefix(&format!("CHOOSE p{} ", side + 1)).unwrap_or_else(|| panic!("expected p{} next, got {c}", side + 1));
                acts[side] = env.open_tokens(side).unwrap().iter().find(|(_, t)| t == tok).unwrap_or_else(|| panic!("{tok} is not legal")).0 as i32;
            }
        }
        cols.slice_mut::<i32>(col::ACTION).copy_from_slice(&acts);
        op(&mut core, &mut cols, b'S', &mut w);
    }
    // Play on in the auto-reset episode (the same teams, another seed).
    cols.slice_mut::<u32>(col::EP_SEED).copy_from_slice(&[3, 1, 4, 1]);
    let mut rng = common::Rng(46);
    for _ in 0..60 {
        common::random_actions(&mut cols, &mut rng);
        op(&mut core, &mut cols, b'S', &mut w);
    }
    w.flush().unwrap();
    let bank: Vec<String> = core.bank().items().iter().map(|b| b.json()).collect();
    assert_eq!(bank.len(), 1, "the repro must quarantine exactly once");
    std::fs::write(out.join("bank.json"), format!("[{}]", bank.join(","))).unwrap();
    eprintln!("ffi quarantine reference: 1 quarantine banked");
}
