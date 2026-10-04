//! Families `hp_type` (`hp_type_label` / `hp_type_mask`, `Gen3Env._hp_type_labels`) and `item`
//! (`item_label` / `item_mask`, `Gen3Env._item_labels`) — the per-REVEALED-slot truth lookups of
//! `agents/observation/belief_labels.py`, rule for rule.
//!
//! Both map each truth mon's species num (the OTHER side's own team, `battle2.team`; a later mon of the
//! same species overrides an earlier one, as a Python dict assignment does) to a value, then give
//! each revealed slot (`species_known >= 0.5`, zipped with the revealed species in encoder order)
//! its species' value with mask 1; every other slot PAD (-1) / mask 0.
//!
//! * `hp_type`: the FIRST of the mon's moves (`Pokemon.moves` order) whose id is `hiddenpower<type>`
//!   with a type in `HP_TYPE_NAMES` (`hp_type_idx_from_move_id`; a bare `hiddenpower` is none).
//! * `item`: the mon's CURRENT item as the truth reading holds it — none / empty ⇒ num 0 (the
//!   "nothing" class); an id with no row in `gen3_items.json` is an `Err` (a FAULT; F-X5-3 — it was
//!   a silent mask 0).
//!
//! Species are matched by DEX NUM (`belief::revealed_nums`: a forme shares its base species' num),
//! and a revealed species absent from the truth team is an `Err`, never a silent mask 0.

use pokesim::encoder::data::tables;
use pokesim::encoder::layout::TEAM_SIZE;
use pokesim::encoder::oracle::Oracle;
use pokesim::present::board_reading::BoardReading;
use pokesim::present::dex::to_id;

use super::belief::{revealed_nums, species_known, species_num};

/// `belief_labels.HP_TYPE_NAMES` — the 16 types in the op's / tracker's order.
pub const HP_TYPE_NAMES: [&str; 16] = [
    "bug", "dark", "dragon", "electric", "fighting", "fire", "flying", "ghost", "grass", "ground", "ice", "poison",
    "psychic", "rock", "steel", "water",
];

/// `hp_type_idx_from_move_id`.
pub fn hp_type_idx(move_id: &str) -> Option<i64> {
    let t = move_id.strip_prefix("hiddenpower")?;
    HP_TYPE_NAMES.iter().position(|n| *n == t).map(|i| i as i64)
}

/// Fill `label` / `mask` for the revealed slots from `value(species num)`: `Ok(None)` is a
/// legitimate absence (the mon runs no Hidden Power); a lookup that cannot be made is an `Err`.
fn fill(
    own: &BoardReading,
    truth: &BoardReading,
    oracle: Option<&Oracle>,
    row: &[f32],
    label: &mut [i64],
    mask: &mut [f32],
    value: impl Fn(i64) -> Result<Option<i64>, String>,
) -> Result<(), String> {
    label.fill(-1);
    mask.fill(0.0);
    let known = species_known(row);
    let slots = (0..TEAM_SIZE).filter(|&i| known[i] >= 0.5);
    for (slot, num) in slots.zip(revealed_nums(own, truth, oracle)?) {
        if let Some(v) = value(num)? {
            label[slot] = v;
            mask[slot] = 1.0;
        }
    }
    Ok(())
}

/// `species num -> value` over the truth team, the LAST mon of a species winning (`Ok(None)`: the
/// mon has no value, e.g. no Hidden Power).
fn by_species(
    truth: &BoardReading,
    f: impl Fn(&pokesim::present::mon::PMon) -> Result<Option<i64>, String>,
) -> Result<Vec<(i64, i64)>, String> {
    let mut out: Vec<(i64, i64)> = Vec::with_capacity(TEAM_SIZE);
    for (_, m) in &truth.team {
        let Some(v) = f(m)? else { continue };
        let num = species_num(&to_id(&m.species))?;
        match out.iter_mut().find(|(k, _)| *k == num) {
            Some(e) => e.1 = v,
            None => out.push((num, v)),
        }
    }
    Ok(out)
}

fn lookup(map: &[(i64, i64)], num: i64) -> Option<i64> {
    map.iter().find(|(k, _)| *k == num).map(|(_, v)| *v)
}

pub fn write_hp_type(own: &BoardReading, truth: &BoardReading, oracle: Option<&Oracle>, row: &[f32], label: &mut [i64], mask: &mut [f32]) -> Result<(), String> {
    let map = by_species(truth, |m| Ok(m.moves.moves_ref().iter().find_map(|(_, mv)| hp_type_idx(&mv.id))))?;
    // absent = the revealed mon runs no Hidden Power (a legitimate mask 0)
    fill(own, truth, oracle, row, label, mask, |num| Ok(lookup(&map, num)))
}

pub fn write_item(own: &BoardReading, truth: &BoardReading, oracle: Option<&Oracle>, row: &[f32], label: &mut [i64], mask: &mut [f32]) -> Result<(), String> {
    let map = by_species(truth, |m| match m.item.as_deref() {
        Some(id) if !id.is_empty() => match tables().items.get(&to_id(id)) {
            Some(n) if *n >= 1.0 => Ok(Some(*n as i64)),
            _ => Err(format!(
                "item labels: item {id:?} of truth mon {:?} has no valid num in gen3_items.json — never a silent mask 0 (F-X5-3)",
                m.species
            )),
        },
        _ => Ok(Some(0)), // no item = the "nothing" class
    })?;
    // every truth mon has an entry and every revealed species is on the truth team (`fill`)
    fill(own, truth, oracle, row, label, mask, |num| {
        lookup(&map, num).map(Some).ok_or_else(|| format!("item labels: no truth item for species num {num}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hp_type_order_is_the_python_table() {
        assert_eq!(hp_type_idx("hiddenpowerbug"), Some(0));
        assert_eq!(hp_type_idx("hiddenpowerwater"), Some(15));
        assert_eq!(hp_type_idx("hiddenpower"), None);
        assert_eq!(hp_type_idx("hiddenpowerfire70"), None);
        assert_eq!(hp_type_idx("thunderbolt"), None);
    }
}
