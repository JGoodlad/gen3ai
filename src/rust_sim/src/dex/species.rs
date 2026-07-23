//! Species reference data, parsed from `data/pokemon/gen3_species.json`.
//! Mirrors `agents.gen3_data.species.SpeciesData`.

use super::types::{BaseStats, Type};
use crate::json::Json;
use std::collections::HashMap;

/// Static facts about one species (national-dex `num`, base stats, types).
#[derive(Debug, Clone)]
pub struct SpeciesData {
    pub id: String,
    pub num: u16,
    pub name: String,
    pub base_stats: BaseStats,
    pub types: Vec<Type>,
    /// A FIXED max-HP override (Showdown's `pokedex.ts` `maxHP`), present only for
    /// Shedinja (`maxHP: 1`). `None` for every normal species. `Pokemon.setSpecies`
    /// overwrites the computed HP stat with this when set (`pokemon.js:990`); the
    /// stat calc mirrors that hook. Parsed from the `maxHP` JSON key when present.
    pub max_hp: Option<u16>,
    /// The species WEIGHT in HECTOGRAMS (`gen3_move_coverage_batch5_v1` — the unit
    /// Showdown's `pokemon.getWeight()` returns: `weighthg = round(weightkg * 10)`,
    /// clamped `>= 1`). Low Kick's BP ladder compares on it (`>=2000` → 120, `>=1000`
    /// → 100, `>=500` → 80, `>=250` → 60, `>=100` → 40, else 20 — probe-swept exact
    /// cutoffs). gen3 has NO `ModifyWeight` handler, so the species value IS the live
    /// weight (probe: Skarmory `getWeight()` == 505 == weighthg). 0 when the data
    /// omits it (no gen-3 species does; a 0 would floor into the lightest bucket).
    pub weighthg: u32,
    /// The species' FIXED gender (`gen3_turn0_construction_v1`, Showdown pokedex
    /// `gender`): `Some('N')` genderless (Magnemite/Ditto/Metagross), `Some('M')`/
    /// `Some('F')` a fixed-gender species (Nidoran-M/F, Tauros, the Lati twins).
    /// `None` for a normal RATIO'd species (Snorlax etc. — the pokedex carries a
    /// `genderRatio` instead, which Showdown IGNORES for the actual assignment). This
    /// is the `species.gender` in the ctor's `gender = set.gender || species.gender ||
    /// sample(['M','F'])` (pokemon.ts): a mon whose PACKED set omits the gender AND
    /// whose species has `gender == None` here draws ONE construction-time uniform
    /// `sample(['M','F'])`; a `Some(_)` species gender is used draw-free. Parsed from
    /// the `gender` JSON key (present only for the 58 fixed/genderless gen-3 species).
    pub gender: Option<char>,
}

pub(super) fn parse(root: &Json) -> Result<HashMap<String, SpeciesData>, String> {
    let obj = root.as_object().ok_or("species: expected a JSON object")?;
    let mut out = HashMap::with_capacity(obj.len());
    for (id, v) in obj {
        let bs = v.get("baseStats");
        let stat = |k: &str| bs.and_then(|b| b.get(k)).and_then(Json::as_f64).map_or(0, |n| n as u16);
        let types = v
            .get("types")
            .and_then(Json::as_array)
            .map(|a| a.iter().filter_map(|t| t.as_str().and_then(Type::from_name)).collect())
            .unwrap_or_default();
        out.insert(
            id.clone(),
            SpeciesData {
                id: id.clone(),
                num: v.int_or("num", 0) as u16,
                name: v.str_at("name").unwrap_or(id).to_string(),
                base_stats: BaseStats {
                    hp: stat("hp"),
                    atk: stat("atk"),
                    def: stat("def"),
                    spa: stat("spa"),
                    spd: stat("spd"),
                    spe: stat("spe"),
                },
                types,
                max_hp: v.get("maxHP").and_then(Json::as_f64).map(|n| n as u16),
                weighthg: v.int_or("weighthg", 0) as u32,
                gender: v.str_at("gender").and_then(|s| s.chars().next()),
            },
        );
    }
    Ok(out)
}
