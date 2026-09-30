//! Family `spread` — `belief_spread` (+ mask), `belief_nature` (+ mask), `belief_ev` (+ mask)
//! (`Gen3Env._spread_labels` / `_nature_ev_map`, `belief_labels.build_known_spread_labels` /
//! `build_known_nature_ev_labels`, `belief_tables.true_nature_ev_label`), rule for rule.
//!
//! * `belief_spread`: each revealed slot gets its species' TRUE derived stats (atk, def, spa, spd,
//!   spe) from the OTHER side's own reading (`mon.stats`, the request's `baseStoredStats`); a mon
//!   with any of the five unknown is omitted (mask 0). A later same-species mon overrides.
//! * `belief_nature` / `belief_ev` (`gen3_true_spread_labels_v1`): the truth mon's DECLARED spread,
//!   which the reading backfills from the side's packed team exactly as poke-env does — the
//!   nature's num and the EVs at their stat-effective `4·⌊ev/4⌋`. A THROWING guard: the declared
//!   spread, at L100 with its TRUE IVs, must reproduce the five derived stats (else `Err`, which the
//!   caller makes a FAULT — the Python env raises `SpreadLabelError`). A missing spread or an
//!   unknown nature is an `Err` too. A mon with an unknown stat or no dex row is omitted (mask 0).
//!   Read per decision, never cached (F-LC-6: a species-set-keyed cache served stale labels).
//!
//! The nature table is read from the SAME `data/pokemon/gen3_natures.json` the Python facade reads,
//! from the directory the build's stamp names (the port's compile-time data path), at STARTUP when
//! the family is declared.

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
    /// lower-cased nature name -> (num, multipliers in SPREAD order).
    natures: HashMap<String, (i64, [f64; N_SPREAD])>,
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
    let mut natures = HashMap::new();
    for (name, v) in nat.as_object().ok_or("gen3_natures.json: not an object")? {
        let num = v.get("num").and_then(Json::as_f64).ok_or_else(|| format!("nature {name}: no num"))? as i64;
        let mult: [f64; N_SPREAD] = std::array::from_fn(|j| v.get(SPREAD_KEYS[j]).and_then(Json::as_f64).unwrap_or(1.0));
        natures.insert(name.to_lowercase(), (num, mult));
    }
    Ok(Tables { natures })
}

/// Load the tables (STARTUP, from `labels::declare`); the error names the file.
pub fn prepare() -> Result<(), String> {
    TABLES.get_or_init(build).as_ref().map(|_| ()).map_err(Clone::clone)
}

fn t() -> &'static Tables {
    TABLES.get().and_then(|r| r.as_ref().ok()).expect("spread tables are loaded at startup (labels::declare)")
}

/// `priors.gen3_stat(base, ev, mult, iv)`: L100, exact integer nature math.
pub fn gen3_stat(base: i64, ev: i64, mult: f64, iv: i64) -> i64 {
    let pre = 2 * base + iv + ev / 4 + 5;
    if mult > 1.0 {
        pre * 11 / 10
    } else if mult < 1.0 {
        pre * 9 / 10
    } else {
        pre
    }
}

/// poke-env's default nature for a set that declares none (`backfill_spread_from_teambuilder`).
const DEFAULT_NATURE: &str = "serious";

/// `belief_tables.true_nature_ev_label(species_id, derived, base, nature, evs, ivs)`: the declared
/// `(nature num, [ev×5])`, or `Err` when the declared spread does not reproduce `derived`.
pub fn true_label(
    species_id: &str,
    derived: [i64; N_SPREAD],
    base: [i64; N_SPREAD],
    nature: Option<&str>,
    evs: Option<&[i64]>,
    ivs: Option<&[i64]>,
) -> Result<(i64, [i64; N_SPREAD]), String> {
    let (Some(evs), Some(ivs)) = (evs, ivs) else {
        return Err(format!("{species_id}: the truth mon carries no declared spread (evs={evs:?}, ivs={ivs:?})"));
    };
    if evs.len() != 6 || ivs.len() != 6 {
        return Err(format!("{species_id}: a spread list is not six long (evs={evs:?}, ivs={ivs:?})"));
    }
    let nname = nature.unwrap_or(DEFAULT_NATURE).to_lowercase();
    let &(num, mult) = t().natures.get(&nname).ok_or_else(|| format!("{species_id}: unknown nature {nname:?}"))?;
    let mut out = [0i64; N_SPREAD];
    let mut got = [0i64; N_SPREAD];
    for j in 0..N_SPREAD {
        let k = STAT_IDX[j];
        out[j] = evs[k] / 4 * 4;
        got[j] = gen3_stat(base[j], evs[k], mult[j], ivs[k]);
    }
    if got != derived {
        return Err(format!(
            "{species_id}: the declared spread (nature {nname}, evs {evs:?}, ivs {ivs:?}) gives {got:?} at L100 but the server's stats are {derived:?}"
        ));
    }
    Ok((num, out))
}

fn truth_stats(m: &pokesim::present::mon::PMon) -> Option<[i64; N_SPREAD]> {
    let mut out = [0i64; N_SPREAD];
    for (j, &i) in STAT_IDX.iter().enumerate() {
        out[j] = m.stats[i]?;
    }
    Some(out)
}

fn nature_ev_map(truth: &BoardReading) -> Result<Vec<(String, (i64, [i64; N_SPREAD]))>, String> {
    let mut out: Vec<(String, (i64, [i64; N_SPREAD]))> = Vec::new();
    for (_, m) in &truth.team {
        let Some(derived) = truth_stats(m) else { continue };
        let sid = to_id(&m.species);
        let Some(sd) = tables().species.get(&sid) else { continue };
        let base: [i64; N_SPREAD] = match sd.base_stats {
            Some(b) => std::array::from_fn(|j| b[STAT_IDX[j]].round() as i64),
            None => [0; N_SPREAD],
        };
        let r = true_label(&sid, derived, base, m.nature.as_deref(), m.evs.as_deref(), m.ivs.as_deref())?;
        match out.iter_mut().find(|(k, _)| *k == sid) {
            Some(e) => e.1 = r,
            None => out.push((sid, r)),
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn write(
    own: &BoardReading,
    truth: &BoardReading,
    row: &[f32],
    spread: &mut [f32],
    spread_mask: &mut [f32],
    nature: &mut [i64],
    nature_mask: &mut [f32],
    ev: &mut [f32],
    ev_mask: &mut [f32],
) -> Result<(), String> {
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
    let ne = nature_ev_map(truth)?;
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gen3_stat_is_the_python_formula() {
        // 2*100 + 31 + 252/4 + 5 = 299; ×1.1 floor = 328; ×0.9 floor = 269; IV 30 is one point less
        assert_eq!(gen3_stat(100, 252, 1.0, 31), 299);
        assert_eq!(gen3_stat(100, 252, 1.1, 31), 328);
        assert_eq!(gen3_stat(100, 252, 0.9, 31), 269);
        assert_eq!(gen3_stat(100, 0, 1.0, 30), 235);
    }

    #[test]
    fn the_label_is_the_declared_set_and_a_mismatch_is_refused() {
        prepare().unwrap();
        let base = [134, 110, 95, 100, 61]; // Tyranitar's atk..spe
        let (num, mult) = t().natures["adamant"];
        let evs = [4, 252, 0, 0, 0, 255]; // hp..spe; 255 labels as 252
        let ivs = [31, 30, 30, 31, 30, 31]; // Hidden Power Bug: def / spd have no IV-31 decomposition at 0 EVs
        let derived: [i64; N_SPREAD] = std::array::from_fn(|j| gen3_stat(base[j], evs[j + 1], mult[j], ivs[j + 1]));
        assert_eq!(true_label("tyranitar", derived, base, Some("adamant"), Some(&evs), Some(&ivs)), Ok((num, [252, 0, 0, 0, 252])));
        assert!(true_label("tyranitar", derived, base, Some("adamant"), Some(&evs), Some(&[31; 6])).unwrap_err().contains("server's stats"));
        assert!(true_label("tyranitar", derived, base, Some("jolly"), Some(&evs), Some(&ivs)).is_err());
        assert!(true_label("tyranitar", derived, base, Some("grumpy"), Some(&evs), Some(&ivs)).unwrap_err().contains("unknown nature"));
        assert!(true_label("tyranitar", derived, base, Some("adamant"), None, None).unwrap_err().contains("no declared spread"));
    }
}
