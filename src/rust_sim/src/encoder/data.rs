//! The encoder's REFERENCE TABLES, read from `data/pokemon/*.json` exactly as the Python facade
//! reads them (`agents.gen3_data` + `state_encoder.load_mappings`): the raw species / item /
//! ability dicts the sub-encoders index, the move dex (`gen3_data.moves._build`), the Smogon
//! ability priors (`AbilitiesEncoder._species_priors`, `sleep_belief.early_bird_probability`) and
//! the nature multipliers (`gen3_data.natures.multipliers`). Read at RUNTIME from the same files,
//! so a `tools/` regeneration reaches both encoders at once (program §3, "data changes propagate
//! through `data/`"). Loaded once per process.

use std::collections::HashMap as StdMap;
use std::sync::OnceLock;

/// The tables' maps: keyed by an id, hashed with [`crate::present::dex::IdHasher`] (every encode
/// looks up each mon's species, item, ability and moves; only `get` is ever called on them, so the
/// hasher is not observable).
pub type HashMap<K, V> = StdMap<K, V, std::hash::BuildHasherDefault<crate::present::dex::IdHasher>>;

use super::layout::TYPE_TO_IDX;
use crate::json::Json;

/// One `gen3_species.json` row as `SpeciesEncoder` reads it.
#[derive(Debug, Clone)]
pub struct SpeciesRec {
    /// `entry["num"]` — REQUIRED (a row without one is a load error, `num_of`).
    pub num: f64,
    /// `entry["baseStats"]` as `stats.get(k, 100)` per key (hp, atk, def, spa, spd, spe); `None`
    /// when the row has no `baseStats` (the encoder then reads the mon's own `base_stats`).
    pub base_stats: Option<[f64; 6]>,
}

/// One `gen3_data.moves.MoveData`, the fields the obs reads.
#[derive(Debug, Clone)]
pub struct MoveRec {
    pub num: i64,
    pub base_power: i64,
    /// `TypeEncoder.TYPE_TO_IDX` of the move's type (`"???"` for Curse), 0 if unmapped.
    pub type_idx: usize,
    pub accuracy: i64,
    pub never_miss: bool,
    pub has_secondary: bool,
    pub has_recoil: bool,
}

pub struct Tables {
    pub species: HashMap<String, SpeciesRec>,
    pub items: HashMap<String, f64>,
    pub abilities: HashMap<String, f64>,
    pub moves: HashMap<String, MoveRec>,
    /// `AbilitiesEncoder._species_priors`: species → (ability1 num, ability2 num, P(ability1)).
    pub ability_rank: HashMap<String, (f64, f64, f64)>,
    /// `gen3_data.priors.ability_raw()` — species → {ability: probability}.
    pub ability_priors: HashMap<String, HashMap<String, f64>>,
    /// `gen3_data.natures.multipliers()` — nature → [atk, def, spa, spd, spe] (a key the row
    /// lacks is `None`, which the spread encoder reads as 1.0).
    pub natures: HashMap<String, [Option<f64>; 5]>,
}

fn load(name: &str) -> Result<Json, String> {
    let path = crate::dex::default_data_dir().join(name);
    let text = std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Json::parse(&text).map_err(|e| format!("parse {}: {e}", path.display()))
}

fn obj(j: &Json) -> Result<&StdMap<String, Json>, String> {
    j.as_object().ok_or_else(|| "expected a JSON object".to_string())
}

/// `int(v.get(key, default))` — the facade's integer reads (a JSON bool reads 0/1, as `int()` does).
fn int_of(v: &Json, key: &str, default: i64) -> i64 {
    match v.get(key) {
        None => default,
        Some(Json::Bool(b)) => *b as i64,
        Some(j) => j.as_f64().map_or(default, |n| n as i64),
    }
}

/// A dex row's `num`, REQUIRED (F-X5-5): a row with no numeric `num` is a load-time error, never a silent
/// 0 — Python's `get("num", 0)` default is gone too (`gen3_data._base.load_dex_json` raises on the same rows).
/// A JSON bool is not a number here (Python's check rejects it as well).
fn num_of(v: &Json, file: &str, id: &str) -> Result<i64, String> {
    match v.get("num") {
        Some(Json::Num(n)) => Ok(*n as i64),
        other => Err(format!("{file}: row {id:?} has no numeric `num` (got {other:?}) — it would silently encode as 0")),
    }
}

fn bool_of(v: &Json, key: &str) -> bool {
    match v.get(key) {
        None | Some(Json::Null) => false,
        Some(Json::Bool(b)) => *b,
        Some(Json::Num(n)) => *n != 0.0,
        Some(Json::Str(s)) => !s.is_empty(),
        Some(_) => true,
    }
}

/// `TypeEncoder.TYPE_TO_IDX.get(name, 0)`.
pub fn type_idx(name: &str) -> usize {
    TYPE_TO_IDX.iter().find(|(k, _)| *k == name).map_or(0, |(_, v)| *v)
}

fn build() -> Result<Tables, String> {
    const STATS: [&str; 6] = ["hp", "atk", "def", "spa", "spd", "spe"];
    let mut species = HashMap::default();
    for (id, v) in obj(&load("gen3_species.json")?)? {
        let base_stats = v.get("baseStats").map(|bs| {
            let mut s = [0.0; 6];
            for (i, k) in STATS.iter().enumerate() {
                s[i] = bs.get(k).and_then(Json::as_f64).unwrap_or(100.0);
            }
            s
        });
        species.insert(id.clone(), SpeciesRec { num: num_of(v, "gen3_species.json", id)? as f64, base_stats });
    }
    let nums = |name: &str| -> Result<HashMap<String, f64>, String> {
        let mut m = HashMap::default();
        for (id, v) in obj(&load(name)?)? {
            if v.as_object().is_some() {
                m.insert(id.clone(), num_of(v, name, id)? as f64);
            }
        }
        Ok(m)
    };
    let items = nums("gen3_items.json")?;
    let abilities = nums("gen3_abilities.json")?;
    let mut moves = HashMap::default();
    for (id, v) in obj(&load("gen3_moves.json")?)? {
        // `_resolve_type`: "???" is THREE_QUESTION_MARKS, else the upper-cased name; the move
        // encoder then spells THREE_QUESTION_MARKS back as "???" for `TYPE_TO_IDX`.
        let ty = v.str_at("type").unwrap_or("Normal");
        let name = if ty == "???" { "???".to_string() } else { ty.to_ascii_uppercase() };
        moves.insert(
            id.clone(),
            MoveRec {
                num: num_of(v, "gen3_moves.json", id)?,
                base_power: int_of(v, "basePower", 0),
                type_idx: type_idx(&name),
                accuracy: int_of(v, "accuracy", 100),
                never_miss: bool_of(v, "never_miss"),
                has_secondary: bool_of(v, "hasSecondary"),
                has_recoil: bool_of(v, "hasRecoil"),
            },
        );
    }
    let mut ability_priors: HashMap<String, HashMap<String, f64>> = HashMap::default();
    let mut ability_rank = HashMap::default();
    for (sp, probs) in obj(&load("gen3_ability_priors.json")?)? {
        let mut m = HashMap::default();
        for (ab, p) in obj(probs)? {
            m.insert(ab.clone(), p.as_f64().ok_or("ability prior: not a number")?);
        }
        if !m.is_empty() {
            // sorted by (-p, name): the favourite first, ties by name.
            let mut ranked: Vec<(&String, f64)> = m.iter().map(|(k, v)| (k, *v)).collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("finite prior").then_with(|| a.0.cmp(b.0)));
            let num = |a: &str| abilities.get(a).copied().unwrap_or(0.0);
            let n1 = num(ranked[0].0);
            let n2 = ranked.get(1).map_or(0.0, |r| num(r.0));
            ability_rank.insert(sp.clone(), (n1, n2, ranked[0].1));
        }
        ability_priors.insert(sp.clone(), m);
    }
    let mut natures = HashMap::default();
    for (name, v) in obj(&load("gen3_natures.json")?)? {
        let mut mults = [None; 5];
        for (i, k) in super::layout::NATURE_STAT_ORDER.iter().enumerate() {
            mults[i] = v.get(k).and_then(Json::as_f64);
        }
        natures.insert(name.clone(), mults);
    }
    Ok(Tables { species, items, abilities, moves, ability_rank, ability_priors, natures })
}

/// The process-wide tables (crash-don't-drop: a missing / unparsable data file panics with its
/// path — a silent gap would corrupt every row).
pub fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| build().unwrap_or_else(|e| panic!("encoder tables: {e}")))
}

#[cfg(test)]
mod num_tests {
    use super::*;

    fn row(s: &str) -> Json {
        Json::parse(s).unwrap()
    }

    #[test]
    fn a_dex_row_needs_a_numeric_num_and_a_real_zero_is_fine() {
        assert_eq!(num_of(&row(r#"{"num": 7}"#), "f.json", "a"), Ok(7));
        assert_eq!(num_of(&row(r#"{"num": 0}"#), "f.json", "z"), Ok(0));
        for bad in [r#"{"name": "x"}"#, r#"{"num": null}"#, r#"{"num": true}"#, r#"{"num": "7"}"#, r#""not a row""#] {
            let e = num_of(&row(bad), "f.json", "bad").unwrap_err();
            assert!(e.contains("row \"bad\" has no numeric `num`"), "{bad}: {e}");
        }
    }

    #[test]
    fn every_shipped_dex_row_has_a_numeric_num() {
        // The tables load (`build` runs `num_of` on every species / item / ability / move row).
        assert!(super::tables().species.len() > 400);
    }
}
