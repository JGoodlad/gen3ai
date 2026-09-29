//! Families `hp_type` (`hp_type_label` / `hp_type_mask`, `Gen3Env._hp_type_labels`) and `item`
//! (`item_label` / `item_mask`, `Gen3Env._item_labels`) — the per-REVEALED-slot truth lookups of
//! `agents/observation/belief_labels.py`, rule for rule.
//!
//! Both map each truth mon's species (the OTHER side's own team, `battle2.team`; a later mon of the
//! same species overrides an earlier one, as a Python dict assignment does) to a value, then give
//! each revealed slot (`species_known >= 0.5`, zipped with the revealed species in encoder order)
//! its species' value with mask 1; every other slot PAD (-1) / mask 0.
//!
//! * `hp_type`: the FIRST of the mon's moves (`Pokemon.moves` order) whose id is `hiddenpower<type>`
//!   with a type in `HP_TYPE_NAMES` (`hp_type_idx_from_move_id`; a bare `hiddenpower` is none).
//! * `item`: the mon's CURRENT item as the truth reading holds it — none / empty ⇒ num 0 (the
//!   "nothing" class); an id with no row in `gen3_items.json` ⇒ absent (mask 0).

use pokesim::encoder::data::tables;
use pokesim::encoder::layout::TEAM_SIZE;
use pokesim::present::board_reading::BoardReading;
use pokesim::present::dex::to_id;

use super::belief::{revealed_species, species_known};

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

/// Fill `label` / `mask` for the revealed slots from `value(species id)`.
fn fill(own: &BoardReading, row: &[f32], label: &mut [i64], mask: &mut [f32], value: impl Fn(&str) -> Option<i64>) {
    label.fill(-1);
    mask.fill(0.0);
    let known = species_known(row);
    let slots = (0..TEAM_SIZE).filter(|&i| known[i] >= 0.5);
    for (slot, sp) in slots.zip(revealed_species(own)) {
        if let Some(v) = value(&to_id(sp)) {
            label[slot] = v;
            mask[slot] = 1.0;
        }
    }
}

/// `species -> value` over the truth team, the LAST mon of a species winning.
fn by_species(truth: &BoardReading, f: impl Fn(&pokesim::present::mon::PMon) -> Option<i64>) -> Vec<(String, i64)> {
    let mut out: Vec<(String, i64)> = Vec::with_capacity(TEAM_SIZE);
    for (_, m) in &truth.team {
        let Some(v) = f(m) else { continue };
        let sp = to_id(&m.species);
        match out.iter_mut().find(|(k, _)| *k == sp) {
            Some(e) => e.1 = v,
            None => out.push((sp, v)),
        }
    }
    out
}

fn lookup(map: &[(String, i64)], sp: &str) -> Option<i64> {
    map.iter().find(|(k, _)| k == sp).map(|(_, v)| *v)
}

pub fn write_hp_type(own: &BoardReading, truth: &BoardReading, row: &[f32], label: &mut [i64], mask: &mut [f32]) {
    let map = by_species(truth, |m| m.moves.moves_ref().iter().find_map(|(_, mv)| hp_type_idx(&mv.id)));
    fill(own, row, label, mask, |sp| lookup(&map, sp).filter(|t| (0..16).contains(t)));
}

pub fn write_item(own: &BoardReading, truth: &BoardReading, row: &[f32], label: &mut [i64], mask: &mut [f32]) {
    let map = by_species(truth, |m| match m.item.as_deref() {
        Some(id) if !id.is_empty() => tables().items.get(&to_id(id)).map(|n| *n as i64),
        _ => Some(0),
    });
    fill(own, row, label, mask, |sp| lookup(&map, sp).filter(|n| *n >= 0));
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
