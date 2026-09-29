//! THE ENV CORE — the one implementation both front ends call (M5 Lane 0,
//! `designs/endstate/program_rust_core.md` §2 M5).
//!
//! * [`columns`] — the GENERATED column contract, opcodes, statuses and counters
//!   (`python -m utils.rust_env.columns --write`; never edit it by hand).
//! * [`spec`] — the STARTUP declaration (one JSON object, every key required).
//! * [`pool`] — N envs on T persistent workers, acquired at startup and frozen.
//! * [`dispatch`] — [`dispatch::Core`], THE entry both front ends call: `dispatch(op, cols)`.
//! * [`refusal`] — the refusal policy: quarantine + banked input log + typed class.
pub mod columns;
pub mod dispatch;
pub mod pool;
pub mod refusal;
pub mod spec;

pub use dispatch::{Core, DispatchError, OwnedCols};

/// THE BUILD STAMP (`build.rs`): `stamp=v1;commit=…;src=…;nfiles=…;nan_poison=…;schema=…;data=…`.
/// Both front ends report it and `src/utils/rust_env/stamp.py` REFUSES a build that is not this
/// tree's (gate ⑤).
pub const STAMP: &str = env!("POKESIM_ENV_STAMP");
pub use spec::Spec;

#[cfg(test)]
mod tests {
    use super::columns::*;

    #[test]
    fn the_generated_contract_is_self_consistent() {
        let b = col_bytes(3);
        assert_eq!(b[col::OBS], 3 * SIDES * OBS_DIM * 4);
        assert_eq!(b[col::MASK], 3 * SIDES * ACT);
        assert_eq!(b[col::COUNTERS], NCOUNTERS * 8, "a pool column does not scale with N");
        assert_eq!(counter::NAMES.len(), NCOUNTERS);
        for (i, c) in COLUMNS.iter().enumerate() {
            assert!(c.row_elems > 0, "{}", c.name);
            assert!(COLUMNS[..i].iter().all(|d| d.name != c.name), "duplicate {}", c.name);
        }
        for &k in &counter::AFTER_FREEZE {
            assert!(counter::NAMES[k].ends_with("_AFTER_FREEZE"));
        }
        assert_eq!(OBS_DIM, pokesim::encoder::OBS_DIM);
    }

    #[test]
    fn the_stamp_is_well_formed_and_names_this_schema() {
        let f: Vec<(&str, &str)> = super::STAMP.split(';').map(|kv| kv.split_once('=').unwrap()).collect();
        let keys: Vec<&str> = f.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, ["stamp", "commit", "src", "nfiles", "nan_poison", "schema", "data"]);
        assert!(f.contains(&("schema", SCHEMA_ID)));
        assert!(f.contains(&("nan_poison", if cfg!(any(debug_assertions, feature = "emission-selfcheck")) { "1" } else { "0" })));
        // Printed so `core_cargo_test.py` can check the Python recompute agrees with this build.
        println!("POKESIM_ENV_STAMP={}", super::STAMP);
    }
}
