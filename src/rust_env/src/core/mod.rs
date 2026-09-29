//! THE ENV CORE — the one implementation both front ends call (M5 Lane 0,
//! `designs/endstate/program_rust_core.md` §2 M5).
//!
//! * [`columns`] — the GENERATED column contract, opcodes, statuses and counters
//!   (`python -m utils.rust_env.columns --write`; never edit it by hand).
pub mod columns;

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
}
