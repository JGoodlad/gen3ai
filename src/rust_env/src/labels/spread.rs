//! Family `spread` — `belief_spread` (+ mask), `belief_nature` (+ mask), `belief_ev` (+ mask)
//! (`Gen3Env._spread_labels` / `_nature_ev_map`, `belief_labels.build_known_spread_labels` /
//! `build_known_nature_ev_labels`, `belief_tables.invert_nature_evs`), rule for rule.
//!
//! * `belief_spread`: each revealed slot gets its species' TRUE derived stats (atk, def, spa, spd,
//!   spe) from the OTHER side's own reading (`mon.stats`, the request's `baseStoredStats`); a mon
//!   with any of the five unknown is omitted (mask 0). A later same-species mon overrides.
//! * `belief_nature` / `belief_ev`: those stats INVERTED — for every nature, the smallest EV in
//!   `0, 4, …, 252` with `gen3_stat(base, ev, mult) == stat` per stat, all five found and `Σ ≤ 510`;
//!   among the valid natures the one with the highest Smogon spread-usage weight for the species,
//!   then the smallest num. No valid nature ⇒ omitted (mask 0). A species with no dex row ⇒ omitted.
//!   Computed ONCE per episode per side, at the side's first labelled decision (the Python env
//!   caches it per battle, at its first label call).
//!
//! The nature table and the spread priors are read from the SAME `data/pokemon` files the Python
//! facade reads (`gen3_natures.json`, `gen3_spread_priors.json`), from the directory the build's
//! stamp names (the port's compile-time data path), at STARTUP when the family is declared.

use std::collections::HashMap;
use std::sync::OnceLock;

use pokesim::encoder::data::tables;
use pokesim::encoder::layout::TEAM_SIZE;
use pokesim::json::Json;
use pokesim::present::board_reading::BoardReading;
use pokesim::present::dex::to_id;

use super::belief::{revealed_species, species_known};

pub const N_SPREAD: usize = 5;
/// `SPREAD_STAT_ORDER` as indices into poke-env's `STAT_KEYS` (hp, atk, def, spa, spd, spe).
const STAT_IDX: [usize; N_SPREAD] = [1, 2, 3, 4, 5];
const SPREAD_KEYS: [&str; N_SPREAD] = ["atk", "def", "spa", "spd", "spe"];

struct Tables {
    /// (num, multipliers in SPREAD order) per nature, in file order.
    natures: Vec<(i64, [f64; N_SPREAD])>,
    /// lower-cased nature name -> num.
    nature_num: HashMap<String, i64>,
    /// species id -> [(nature name, weight)] in file order.
    spreads: HashMap<String, Vec<(String, f64)>>,
}

static TABLES: OnceLock<Result<Tables, String>> = OnceLock::new();

fn data_dir() -> String {
    crate::core::STAMP.split(';').find_map(|kv| kv.strip_prefix("data=")).expect("the stamp names the data dir").to_string()
}

fn load_json(name: &str) -> Result<Json, String> {
    let path = std::path::Path::new(&data_dir()).join(name);
    let text = std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Json::parse(&text).map_err(|e| format!("parse {}: {e}", path.display()))
}

fn build() -> Result<Tables, String> {
    let nat = load_json("gen3_natures.json")?;
    let mut natures = Vec::new();
    let mut nature_num = HashMap::new();
    for (name, v) in nat.as_object().ok_or("gen3_natures.json: not an object")? {
        let num = v.get("num").and_then(Json::as_f64).ok_or_else(|| format!("nature {name}: no num"))? as i64;
        let mult: [f64; N_SPREAD] = std::array::from_fn(|j| v.get(SPREAD_KEYS[j]).and_then(Json::as_f64).unwrap_or(1.0));
        natures.push((num, mult));
        nature_num.insert(name.to_lowercase(), num);
    }
    let sp = load_json("gen3_spread_priors.json")?;
    let mut spreads = HashMap::new();
    for (sid, rows) in sp.as_object().ok_or("gen3_spread_priors.json: not an object")? {
        let mut out = Vec::new();
        for r in rows.as_array().ok_or_else(|| format!("spreads {sid}: not a list"))? {
            let a = r.as_array().ok_or_else(|| format!("spreads {sid}: a row is not a list"))?;
            let nature = a.first().and_then(Json::as_str).ok_or_else(|| format!("spreads {sid}: no nature"))?;
            let w = a.get(2).and_then(Json::as_f64).ok_or_else(|| format!("spreads {sid}: no weight"))?;
            out.push((nature.to_string(), w));
        }
        spreads.insert(sid.clone(), out);
    }
    Ok(Tables { natures, nature_num, spreads })
}

/// Load the tables (STARTUP, from `labels::declare`); the error names the file.
pub fn prepare() -> Result<(), String> {
    TABLES.get_or_init(build).as_ref().map(|_| ()).map_err(Clone::clone)
}

fn t() -> &'static Tables {
    TABLES.get().and_then(|r| r.as_ref().ok()).expect("spread tables are loaded at startup (labels::declare)")
}

/// `priors.gen3_stat`: L100, IV 31, exact integer nature math.
pub fn gen3_stat(base: i64, ev: i64, mult: f64) -> i64 {
    let pre = 2 * base + 31 + ev / 4 + 5;
    if mult > 1.0 {
        pre * 11 / 10
    } else if mult < 1.0 {
        pre * 9 / 10
    } else {
        pre
    }
}

/// `belief_tables.invert_nature_evs(derived, base, species_id)`.
pub fn invert(derived: [i64; N_SPREAD], base: [i64; N_SPREAD], species_id: &str) -> Option<(i64, [i64; N_SPREAD])> {
    let tb = t();
    let mut weight: HashMap<i64, f64> = HashMap::new();
    if let Some(rows) = tb.spreads.get(species_id) {
        for (nature, w) in rows {
            if let Some(&num) = tb.nature_num.get(&nature.to_lowercase()) {
                *weight.entry(num).or_insert(0.0) += *w;
            }
        }
    }
    let mut best: Option<(f64, i64, [i64; N_SPREAD])> = None;
    for &(num, mult) in &tb.natures {
        let mut evs = [0i64; N_SPREAD];
        let mut ok = true;
        for j in 0..N_SPREAD {
            match (0..=252).step_by(4).find(|&ev| gen3_stat(base[j], ev, mult[j]) == derived[j]) {
                Some(ev) => evs[j] = ev,
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok || evs.iter().sum::<i64>() > 510 {
            continue;
        }
        let w = weight.get(&num).copied().unwrap_or(0.0);
        // highest prior weight, then the smallest num (Python: sort (w, -num, …) reverse)
        let better = match &best {
            None => true,
            Some((bw, bn, _)) => w > *bw || (w == *bw && num < *bn),
        };
        if better {
            best = Some((w, num, evs));
        }
    }
    best.map(|(_, num, evs)| (num, evs))
}

/// The per-episode, per-side inversion cache (`Gen3Env._nature_ev_map`).
#[derive(Default)]
pub struct Cache {
    map: Option<Vec<(String, (i64, [i64; N_SPREAD]))>>,
}

impl Cache {
    pub fn clear(&mut self) {
        self.map = None;
    }
}

fn truth_stats(m: &pokesim::present::mon::PMon) -> Option<[i64; N_SPREAD]> {
    let mut out = [0i64; N_SPREAD];
    for (j, &i) in STAT_IDX.iter().enumerate() {
        out[j] = m.stats[i]?;
    }
    Some(out)
}

fn nature_ev_map(truth: &BoardReading) -> Vec<(String, (i64, [i64; N_SPREAD]))> {
    let mut out: Vec<(String, (i64, [i64; N_SPREAD]))> = Vec::new();
    for (_, m) in &truth.team {
        let Some(derived) = truth_stats(m) else { continue };
        let sid = to_id(&m.species);
        let Some(sd) = tables().species.get(&sid) else { continue };
        let base: [i64; N_SPREAD] = match sd.base_stats {
            Some(b) => std::array::from_fn(|j| b[STAT_IDX[j]].round() as i64),
            None => [0; N_SPREAD],
        };
        if let Some(r) = invert(derived, base, &sid) {
            match out.iter_mut().find(|(k, _)| *k == sid) {
                Some(e) => e.1 = r,
                None => out.push((sid, r)),
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub fn write(
    own: &BoardReading,
    truth: &BoardReading,
    row: &[f32],
    cache: &mut Cache,
    spread: &mut [f32],
    spread_mask: &mut [f32],
    nature: &mut [i64],
    nature_mask: &mut [f32],
    ev: &mut [f32],
    ev_mask: &mut [f32],
) {
    spread.fill(0.0);
    spread_mask.fill(0.0);
    nature.fill(0);
    nature_mask.fill(0.0);
    ev.fill(0.0);
    ev_mask.fill(0.0);
    let known = species_known(row);
    let revealed: Vec<String> = revealed_species(own).into_iter().map(to_id).collect();
    // species -> TRUE derived stats (a later same-species mon overrides)
    let mut stats: Vec<(String, [i64; N_SPREAD])> = Vec::new();
    for (_, m) in &truth.team {
        if let Some(s) = truth_stats(m) {
            let sid = to_id(&m.species);
            match stats.iter_mut().find(|(k, _)| *k == sid) {
                Some(e) => e.1 = s,
                None => stats.push((sid, s)),
            }
        }
    }
    let ne = cache.map.get_or_insert_with(|| nature_ev_map(truth));
    let slots = (0..TEAM_SIZE).filter(|&i| known[i] >= 0.5);
    for (slot, sp) in slots.zip(revealed.iter()) {
        if let Some((_, s)) = stats.iter().find(|(k, _)| k == sp) {
            for j in 0..N_SPREAD {
                spread[slot * N_SPREAD + j] = s[j] as f32;
            }
            spread_mask[slot] = 1.0;
        }
        if let Some((_, (num, evs))) = ne.iter().find(|(k, _)| k == sp) {
            nature[slot] = *num;
            nature_mask[slot] = 1.0;
            for j in 0..N_SPREAD {
                ev[slot * N_SPREAD + j] = evs[j] as f32;
            }
            ev_mask[slot] = 1.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gen3_stat_is_the_python_formula() {
        // 2*100 + 31 + 252/4 + 5 = 299; ×1.1 floor = 328; ×0.9 floor = 269
        assert_eq!(gen3_stat(100, 252, 1.0), 299);
        assert_eq!(gen3_stat(100, 252, 1.1), 328);
        assert_eq!(gen3_stat(100, 252, 0.9), 269);
    }

    #[test]
    fn an_inversion_reproduces_the_stats_it_was_given() {
        prepare().unwrap();
        let base = [134, 110, 95, 100, 61]; // Tyranitar's atk..spe
        let mults = [1.1, 1.0, 0.9, 1.0, 1.0]; // Adamant
        let evs = [252, 4, 0, 0, 252];
        let derived: [i64; N_SPREAD] = std::array::from_fn(|j| gen3_stat(base[j], evs[j], mults[j]));
        let (num, got) = invert(derived, base, "tyranitar").expect("a valid spread inverts");
        let (_, mult) = t().natures.iter().find(|(n, _)| *n == num).unwrap();
        for j in 0..N_SPREAD {
            assert_eq!(gen3_stat(base[j], got[j], mult[j]), derived[j]);
        }
        assert!(invert([1, 1, 1, 1, 1], base, "tyranitar").is_none());
    }
}
