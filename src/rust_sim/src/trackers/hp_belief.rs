//! `agents.training.hidden_power_tracker.HiddenPowerTracker` + `EpisodeTracker`'s two feeders of it
//! (`_maybe_observe_hidden_power`, `_scan_opp_movesets_for_no_hp`): the per-opponent-species
//! Hidden-Power type belief, a (16,) float32 vector narrowed by each observed effectiveness.
//! The priors are `data/pokemon/gen3_hidden_power_priors.json` — the file the Python facade reads.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::OnceLock;

use super::turnview::DamagingMove;
use crate::core_error::{fault, CoreResult};
use crate::core_events::json_out;
use crate::dex::{Dex, Type};
use crate::json::Json;
use crate::present::OneSidedView;

/// `HIDDEN_POWER_TYPE_ORDER` — alphabetical, the obs block's order.
pub const HP_TYPES: [&str; 16] = [
    "bug", "dark", "dragon", "electric", "fighting", "fire", "flying", "ghost", "grass", "ground", "ice", "poison",
    "psychic", "rock", "steel", "water",
];

type Priors = HashMap<String, Vec<(String, f64)>>;

fn priors() -> &'static Priors {
    static P: OnceLock<Priors> = OnceLock::new();
    P.get_or_init(|| {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/pokemon/gen3_hidden_power_priors.json");
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let root = Json::parse(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
        let mut out = HashMap::new();
        for (sp, row) in root.as_object().expect("HP priors: an object") {
            let row = row.as_object().expect("HP priors row: an object");
            out.insert(sp.clone(), row.iter().map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(0.0))).collect());
        }
        out
    })
}

/// `ABILITY_TYPE_MULTIPLIER` + Wonder Guard + frozen Flash Fire (`gen3_mechanics._eff_cached`).
fn effective_multiplier(dex: &Dex, move_type: Type, t1: Option<Type>, t2: Option<Type>, ability: &str, frozen: bool) -> f64 {
    let chart = dex.type_chart();
    let base = match t1 {
        None => 1.0,
        Some(a) => chart.multiplier(a, move_type) * t2.map_or(1.0, |b| chart.multiplier(b, move_type)),
    };
    if ability == "wonderguard" {
        return if base > 1.0 { base } else { 0.0 };
    }
    if ability == "flashfire" && frozen {
        return base;
    }
    let m = match (ability, move_type) {
        ("levitate", Type::Ground) | ("voltabsorb", Type::Electric) | ("waterabsorb", Type::Water) | ("flashfire", Type::Fire) => 0.0,
        ("thickfat", Type::Ice | Type::Fire) => 0.5,
        _ => 1.0,
    };
    base * m
}

/// `bucket_effectiveness`.
fn bucket(m: f64) -> f64 {
    if m == 0.0 {
        0.0
    } else if m < 1.0 {
        0.5
    } else if m == 1.0 {
        1.0
    } else {
        2.0
    }
}

/// The target of an opponent's Hidden Power, as `_wrap_hp_target` builds it.
struct Target {
    t1: Option<Type>,
    t2: Option<Type>,
    ability: String,
    frozen: bool,
}

/// `HiddenPowerTracker`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HpBelief {
    pub state: BTreeMap<String, [f32; 16]>,
    pub ruled_out: BTreeSet<String>,
    pub infeasible: u32,
    pub revision: u32,
}

impl HpBelief {
    fn feasible(dex: &Dex, eff: f64, t: &Target) -> bool {
        HP_TYPES.iter().any(|n| {
            let ty = Type::from_name(n).expect("HP type");
            bucket(effective_multiplier(dex, ty, t.t1, t.t2, &t.ability, t.frozen)) == eff
        })
    }

    fn observe(&mut self, dex: &Dex, species: &str, eff: f64, t: &Target) -> CoreResult<()> {
        let v = self.state.entry(species.to_string()).or_insert_with(|| {
            let mut v = [1.0f32 / 16.0; 16];
            if let Some(row) = priors().get(species).filter(|r| !r.is_empty()) {
                for (i, n) in HP_TYPES.iter().enumerate() {
                    v[i] = row.iter().find(|(k, _)| k == n).map_or(0.0, |(_, p)| *p) as f32;
                }
            }
            v
        });
        for (i, n) in HP_TYPES.iter().enumerate() {
            if v[i] != 0.0 {
                let ty = Type::from_name(n).expect("HP type");
                if bucket(effective_multiplier(dex, ty, t.t1, t.t2, &t.ability, t.frozen)) != eff {
                    v[i] = 0.0;
                }
            }
        }
        self.revision += 1;
        if v.iter().all(|x| *x == 0.0) {
            return Err(fault(format!(
                "HiddenPowerTracker: all candidates eliminated for {species:?} after observing {eff}x"
            )));
        }
        Ok(())
    }

    /// `_maybe_observe_hidden_power(live, ctx)`.
    pub fn maybe_observe(&mut self, dex: &Dex, live: &OneSidedView, forced_switch: bool,
                         dm: Option<&DamagingMove>) -> CoreResult<()> {
        if forced_switch {
            return Ok(());
        }
        let Some(dm) = dm else { return Ok(()) };
        if dm.move_id.as_deref() != Some("hiddenpower") {
            return Ok(());
        }
        let target_species = dm.target_species.as_deref().unwrap_or("");
        let Some(m) = live.ours.mons.iter().find(|m| m.species == target_species) else { return Ok(()) };
        if m.types.is_empty() {
            return Ok(());
        }
        let t = Target {
            t1: Type::from_name(&m.types[0]),
            t2: m.types.get(1).and_then(|x| Type::from_name(x)),
            ability: m.ability.clone().unwrap_or_default().to_lowercase(),
            frozen: dm.target_status.as_deref() == Some("FRZ"),
        };
        let eff = dm.effectiveness.unwrap_or(1.0);
        if !Self::feasible(dex, eff, &t) {
            self.infeasible += 1;
            return Ok(());
        }
        let user = dm.user_species.as_deref().unwrap_or("");
        self.observe(dex, user, eff, &t)
    }

    /// `_scan_opp_movesets_for_no_hp(live)`.
    pub fn scan_no_hp(&mut self, live: &OneSidedView) {
        for m in &live.opp.mons {
            if m.species.is_empty() {
                continue;
            }
            if m.moves.len() >= 4 && !m.moves.iter().any(|mv| mv.id.starts_with("hiddenpower")) && self.ruled_out.insert(m.species.clone()) {
                self.revision += 1;
            }
        }
    }

    pub fn json_into(&self, out: &mut String) {
        out.push_str("{\"state\":[");
        for (i, (sp, v)) in self.state.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push('[');
            json_out::str_into(out, sp);
            out.push_str(",[");
            for (j, x) in v.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                json_out::f64_into(out, *x as f64);
            }
            out.push_str("]]");
        }
        out.push_str("],\"ruled_out\":[");
        for (i, sp) in self.ruled_out.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            json_out::str_into(out, sp);
        }
        out.push_str(&format!("],\"infeasible\":{},\"revision\":{}}}", self.infeasible, self.revision));
    }
}
