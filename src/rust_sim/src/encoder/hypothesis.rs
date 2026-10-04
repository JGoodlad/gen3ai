//! X5's HYPOTHESIS ROW (`gen3_x5_dex_rows_v1`; `designs/endstate/design_x5_belief_tokens.md` §3.4,
//! build unit U1): the per-mon SLOT this encoder writes for an opponent mon of species `s` that is
//! PRESENT but UNREVEALED beyond its species — "species s present, unrevealed set, full HP, no
//! status". An unseen gen-3 mon is pristine (it has never been on the field), so its HP, status and
//! boosts are exactly the defaults.
//!
//! **There is no second encoder.** The encoder never emits this state on its own: an opponent mon
//! enters the reading only by APPEARING, and by then it is on the field. So this module builds the
//! smallest SYNTHETIC input that state needs and hands it to the SAME slot writer the real team loop
//! uses ([`super::slot::populated_slot`]):
//!
//! * the raw reading: `PMon::from_species(s)` (poke-env's `Pokemon(species=…)`: the pokedex's base
//!   stats, types and — for a one-ability species — the inferred ability, exactly as a real switch-in
//!   builds it), then `set_hp_status("100/100")` (the opponent's HP as the protocol reports it);
//! * its view: `present::view::mon_view(mon, active = false, own = false)` (what `present()` builds
//!   for every opponent mon);
//! * the trackers: `SideTrackers::new` — a side that has seen NOTHING (no recency, no Hidden-Power
//!   evidence, no sleep source, no last action), since the mon has never appeared;
//! * the tail: not trapped (an opponent slot never is), not active.
//!
//! The table U2's hypothesis builder gathers from is generated from [`hypothesis_slot`] by
//! `python -m agents.model.hypothesis_dex_rows --write` (through `core_events --dex-rows`), committed
//! beside the model code and byte-gated. The REAL-STATE cross-check
//! (`tests/hypothesis_dex_rows_test.rs`) holds this synthetic row to the row the encoder writes at a
//! real first appearance of the species, on every cell except the DECLARED on-field cells of
//! [`CELLS`] — byte equality, no tolerance.

use super::data;
use super::layout::*;
use super::slot;
use crate::core_error::{fault, CoreResult};
use crate::present::mon::PMon;
use crate::trackers::clock::ClockConfig;
use crate::trackers::SideTrackers;

/// The opponent's HP as the protocol reports a fresh mon (gen-3 OU shows the opponent in percent).
const FULL_HP: &str = "100/100";

/// The slot the encoder writes for a HYPOTHESISED opponent mon of `species` (a poke-env species id,
/// e.g. `skarmory`). Every cell is written (a test / self-check build NaN-prefills the slot, and a
/// NaN left over is a FAULT here). An id the pokedex or the encoder's species table does not hold
/// is REFUSED with the encoder's own error.
pub fn hypothesis_slot(species: &str) -> CoreResult<[f32; POKEMON_FULL_DIM]> {
    let t = data::tables();
    let mut mon = PMon::from_species(species, None)?;
    mon.set_hp_status(FULL_HP, true)?;
    let live = crate::present::view::mon_view(&mon, false, false)?;
    let trk = SideTrackers::new(ClockConfig::default());
    let mut s = [if super::NAN_POISON { f32::NAN } else { 0.0 }; POKEMON_FULL_DIM];
    slot::populated_slot(&trk, t, &mon, &live, false, None, [false, false, false], &mut s)?;
    if let Some(k) = s.iter().position(|x| x.is_nan()) {
        return Err(fault(format!("hypothesis_slot({species}): cell {k} was never written")));
    }
    Ok(s)
}

/// How the real-state cross-check treats a block of the slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellClass {
    /// Depends on the species (the dex row's content): compared, byte for byte.
    Species,
    /// A pristine default that a fresh opponent mon also carries at its first appearance (no Hidden
    /// Power evidence, no spread, not trapped, protect odds 1.0, the slot populated): compared.
    Default,
    /// Revealed only by what happens ON THE FIELD (an item shown by Leftovers, an ability announced on
    /// entry, a move used): compared UNLESS the real row's own flag cell says the field revealed it
    /// (and the synthetic row does not already carry it, as a one-ability species does).
    RevealedOnField { flag: usize },
    /// The mon's on-field state at that decision (HP, status, recency, last action, the active flag):
    /// legitimately different at a real first appearance, so EXCLUDED — never compared.
    OnField,
}

/// One declared block of the slot: `(name, lo, hi, class)`, cells `lo..hi`.
pub type Cell = (&'static str, usize, usize, CellClass);

const MOVES_END: usize = POKEMON_MOVES_OFFSET + 4 * MOVE_SLOT_DIM;

/// The DECLARED classification of all `POKEMON_FULL_DIM` cells (it tiles the slot exactly —
/// `the_declared_cells_tile_the_slot`). The `OnField` blocks are the cross-check's whole exclusion
/// list; every other cell is compared.
pub const CELLS: [Cell; 17] = [
    ("species", POKEMON_SPECIES_OFFSET, POKEMON_ITEMS_OFFSET, CellClass::Species),
    ("item", POKEMON_ITEMS_OFFSET, POKEMON_TYPES_OFFSET, CellClass::RevealedOnField { flag: POKEMON_ITEMS_OFFSET + ITEM_ID_DIM }),
    ("types", POKEMON_TYPES_OFFSET, POKEMON_ABILITIES_OFFSET, CellClass::Species),
    ("ability", POKEMON_ABILITIES_OFFSET, POKEMON_CONDITION_OFFSET, CellClass::RevealedOnField { flag: POKEMON_CONDITION_OFFSET - 1 }),
    ("status", POKEMON_CONDITION_OFFSET, POKEMON_MOVES_OFFSET, CellClass::OnField),
    ("moves", POKEMON_MOVES_OFFSET, MOVES_END, CellClass::RevealedOnField { flag: POKEMON_MOVES_OFFSET + 6 }),
    ("hp_fraction", POKEMON_HP_OFFSET, POKEMON_SPECIES_KNOWN_OFFSET, CellClass::OnField),
    ("species_known", POKEMON_SPECIES_KNOWN_OFFSET, POKEMON_COUNTER_OFFSET, CellClass::Default),
    ("status_counters", POKEMON_COUNTER_OFFSET, POKEMON_SPREAD_OFFSET, CellClass::OnField),
    ("spread", POKEMON_SPREAD_OFFSET, POKEMON_HP_REVEALED_OFFSET, CellClass::Default),
    ("hidden_power", POKEMON_HP_REVEALED_OFFSET, POKEMON_SLEEP_BELIEF_OFFSET, CellClass::Default),
    ("sleep_belief", POKEMON_SLEEP_BELIEF_OFFSET, POKEMON_RECENCY_OFFSET, CellClass::OnField),
    ("recency", POKEMON_RECENCY_OFFSET, POKEMON_PROTECT_OFFSET, CellClass::OnField),
    ("protect", POKEMON_PROTECT_OFFSET, POKEMON_LAST_ACTION_OFFSET, CellClass::Default),
    ("last_action", POKEMON_LAST_ACTION_OFFSET, POKEMON_VECTOR_DIM, CellClass::OnField),
    ("trapped", POKEMON_TRAPPED_OFFSET, POKEMON_ACTIVE_OFFSET, CellClass::Default),
    ("active", POKEMON_ACTIVE_OFFSET, POKEMON_FULL_DIM, CellClass::OnField),
];

/// The class's name as the artifact header spells it.
pub fn class_name(c: CellClass) -> &'static str {
    match c {
        CellClass::Species => "species",
        CellClass::Default => "default",
        CellClass::RevealedOnField { .. } => "revealed_on_field",
        CellClass::OnField => "on_field",
    }
}

/// What one first-appearance comparison did.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Compared {
    /// Cells compared byte for byte (and equal).
    pub cells: usize,
    /// The `RevealedOnField` blocks skipped because the field revealed them, by block name.
    pub revealed: Vec<&'static str>,
}

/// The REAL-STATE cross-check of one first appearance: `real` (the slot the encoder wrote at the
/// first decision the mon was in the reading) against `synthetic` ([`hypothesis_slot`] of the same
/// species). Every cell outside an `OnField` block (and outside a `RevealedOnField` block the field
/// revealed) must be BYTE-equal — `f32::to_bits`, so `-0.0` is not `0.0` and NaN equals nothing.
/// `Err` names every differing block and its first differing cell.
pub fn cross_check(real: &[f32], synthetic: &[f32]) -> Result<Compared, Vec<String>> {
    assert_eq!(real.len(), POKEMON_FULL_DIM);
    assert_eq!(synthetic.len(), POKEMON_FULL_DIM);
    let mut out = Compared::default();
    let mut errs = Vec::new();
    for (name, lo, hi, class) in CELLS {
        match class {
            CellClass::OnField => continue,
            CellClass::RevealedOnField { flag } if real[flag] == 1.0 && synthetic[flag] != 1.0 => {
                out.revealed.push(name);
                continue;
            }
            _ => {}
        }
        match (lo..hi).find(|&k| real[k].to_bits() != synthetic[k].to_bits()) {
            Some(k) => errs.push(format!("{name}: cell {k} real {:?} != synthetic {:?}", real[k], synthetic[k])),
            None => out.cells += hi - lo,
        }
    }
    if errs.is_empty() {
        Ok(out)
    } else {
        Err(errs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_declared_cells_tile_the_slot() {
        let mut at = 0;
        for (name, lo, hi, class) in CELLS {
            assert_eq!(lo, at, "{name} starts at {lo}, the previous block ended at {at}");
            assert!(hi > lo, "{name} is empty");
            if let CellClass::RevealedOnField { flag } = class {
                assert!((lo..hi).contains(&flag), "{name}'s flag cell {flag} lies outside it");
            }
            at = hi;
        }
        assert_eq!(at, POKEMON_FULL_DIM);
    }

    #[test]
    fn a_hypothesis_row_is_the_pristine_unrevealed_mon() {
        let s = hypothesis_slot("skarmory").unwrap();
        assert_eq!(s[POKEMON_SPECIES_OFFSET], 227.0, "the national-dex num");
        assert_eq!(s[POKEMON_SPECIES_OFFSET + 2], (80.0f64 / 255.0) as f32, "atk / 255");
        assert_eq!(s[POKEMON_HP_OFFSET], 1.0, "full HP");
        assert_eq!(s[POKEMON_SPECIES_KNOWN_OFFSET], 1.0);
        assert!(s[POKEMON_ITEMS_OFFSET..POKEMON_TYPES_OFFSET].iter().all(|x| *x == 0.0), "no item revealed");
        assert!(s[POKEMON_CONDITION_OFFSET..MOVES_END].iter().all(|x| *x == 0.0), "no status, no move revealed");
        assert_eq!(s[POKEMON_CONDITION_OFFSET - 1], 0.0, "two possible abilities: the prior, not a reveal");
        assert!(s[POKEMON_RECENCY_OFFSET..POKEMON_PROTECT_OFFSET].iter().all(|x| *x == 1.0), "never seen: saturated recency");
        assert_eq!(s[POKEMON_PROTECT_OFFSET], 1.0);
        assert!(s[POKEMON_LAST_ACTION_OFFSET..].iter().all(|x| *x == 0.0), "no last action, not trapped, not active");
        // a ONE-ability species carries its ability as known — poke-env's single-ability inference
        let m = hypothesis_slot("metagross").unwrap();
        assert_eq!(m[POKEMON_CONDITION_OFFSET - 1], 1.0, "clear body is metagross's only ability");
    }

    #[test]
    fn an_unknown_species_is_refused() {
        assert!(hypothesis_slot("notamon").is_err());
    }

    #[test]
    fn the_cross_check_names_a_differing_compared_cell_and_ignores_the_on_field_ones() {
        let syn = hypothesis_slot("skarmory").unwrap();
        // the on-field cells may differ freely
        let mut real = syn;
        real[POKEMON_HP_OFFSET] = 0.5;
        real[POKEMON_ACTIVE_OFFSET] = 1.0;
        real[POKEMON_RECENCY_OFFSET] = 0.0;
        assert_eq!(cross_check(&real, &syn).unwrap().cells, 122 - 23);
        // a revealed item is skipped, and said so
        let mut real2 = real;
        real2[POKEMON_ITEMS_OFFSET] = 44.0;
        real2[POKEMON_ITEMS_OFFSET + ITEM_ID_DIM] = 1.0;
        assert_eq!(cross_check(&real2, &syn).unwrap().revealed, vec!["item"]);
        // ... but an item cell that differs WITHOUT the reveal flag fails
        let mut real3 = real;
        real3[POKEMON_ITEMS_OFFSET] = 44.0;
        assert!(cross_check(&real3, &syn).unwrap_err()[0].starts_with("item: cell 7"));
        // each compared class has teeth, and a signed zero is a difference
        for k in [POKEMON_TYPES_OFFSET, POKEMON_HP_REVEALED_OFFSET, POKEMON_PROTECT_OFFSET, POKEMON_SPREAD_OFFSET + 3, POKEMON_TRAPPED_OFFSET] {
            let mut bad = real;
            bad[k] = if bad[k] == 0.0 { -0.0 } else { 0.0 };
            let e = cross_check(&bad, &syn).unwrap_err();
            assert!(e[0].contains(&format!("cell {k} ")), "{e:?}");
        }
    }
}
