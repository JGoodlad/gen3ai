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
#[derive(Debug, Clone, PartialEq)]
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
    /// `_prior_discarded` (`gen3_hp_prior_support_v1`): species whose usage-prior row their own
    /// observations refuted, restarted from the flat prior.
    pub prior_discarded: BTreeSet<String>,
    /// `_obs_targets`: every observation of a species, replayable against the flat prior.
    log: BTreeMap<String, Vec<(f64, Target)>>,
}

impl HpBelief {
    fn feasible(dex: &Dex, eff: f64, t: &Target) -> bool {
        HP_TYPES.iter().any(|n| {
            let ty = Type::from_name(n).expect("HP type");
            bucket(effective_multiplier(dex, ty, t.t1, t.t2, &t.ability, t.frozen)) == eff
        })
    }

    /// `HiddenPowerTracker._narrow`: zero every candidate that could not produce `eff` on `t`.
    fn narrow(dex: &Dex, v: &mut [f32; 16], eff: f64, t: &Target) {
        for (i, n) in HP_TYPES.iter().enumerate() {
            if v[i] != 0.0 {
                let ty = Type::from_name(n).expect("HP type");
                if bucket(effective_multiplier(dex, ty, t.t1, t.t2, &t.ability, t.frozen)) != eff {
                    v[i] = 0.0;
                }
            }
        }
    }

    /// `HiddenPowerTracker.observe`. A usage prior's zero is not an impossibility
    /// (`gen3_hp_prior_support_v1`): every one of the 16 types is a legal Hidden Power (the IVs',
    /// `sim/dex.ts` `getHiddenPower`; an unset IV is 31, `sim/pokemon.ts:387-394`, so an IV-less set
    /// is HP Dark). When the observations eliminate every type the species' prior row gives mass but
    /// some type explains them all, the species restarts from the flat prior and its log is
    /// replayed; only a log NO type explains is refused.
    fn observe(&mut self, dex: &Dex, species: &str, eff: f64, t: &Target) -> CoreResult<()> {
        let prior = priors().get(species).filter(|r| !r.is_empty());
        let v = self.state.entry(species.to_string()).or_insert_with(|| {
            let mut v = [1.0f32 / 16.0; 16];
            if let Some(row) = prior {
                for (i, n) in HP_TYPES.iter().enumerate() {
                    v[i] = row.iter().find(|(k, _)| k == n).map_or(0.0, |(_, p)| *p) as f32;
                }
            }
            v
        });
        Self::narrow(dex, v, eff, t);
        self.revision += 1;
        let log = self.log.entry(species.to_string()).or_default();
        log.push((eff, t.clone()));
        if v.iter().any(|x| *x != 0.0) {
            return Ok(());
        }
        if prior.is_some() && !self.prior_discarded.contains(species) {
            let mut flat = [1.0f32 / 16.0; 16];
            for (e, tg) in log.iter() {
                Self::narrow(dex, &mut flat, *e, tg);
            }
            if flat.iter().any(|x| *x != 0.0) {
                *v = flat;
                self.prior_discarded.insert(species.to_string());
                return Ok(());
            }
        }
        Err(fault(format!(
            "HiddenPowerTracker: all candidates eliminated for {species:?} after observing {eff}x \
             (no Hidden Power type explains the observations, even under the flat prior)"
        )))
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
        out.push_str(&format!("],\"infeasible\":{},\"prior_discarded\":[", self.infeasible));
        for (i, sp) in self.prior_discarded.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            json_out::str_into(out, sp);
        }
        out.push_str(&format!("],\"revision\":{}}}", self.revision));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tgt(t1: &str, t2: Option<&str>, ability: &str) -> Target {
        Target { t1: Type::from_name(t1), t2: t2.and_then(Type::from_name), ability: ability.into(), frozen: false }
    }

    fn survivors(b: &HpBelief, sp: &str) -> Vec<&'static str> {
        let v = b.state.get(sp).expect("observed");
        HP_TYPES.iter().zip(v).filter(|(_, p)| **p != 0.0).map(|(n, _)| *n).collect()
    }

    /// `gen3_hp_prior_support_v1`: a prior row the evidence refutes is replaced by the flat prior and
    /// the WHOLE log replayed — the first observation (which the row survived) bites in the replay.
    #[test]
    fn a_refuted_usage_prior_restarts_from_flat_and_replays_the_log() {
        let dex = Dex::for_gen(3);
        let mut b = HpBelief::default();
        // Lunatone's real row: electric / fighting / fire / grass / ice / rock / water — no dark.
        // 0.5x on Tyranitar (Rock / Dark): only Fire of the row.
        b.observe(&dex, "lunatone", 0.5, &tgt("rock", Some("dark"), "sandstream")).expect("resisted by Tyranitar");
        assert_eq!(survivors(&b, "lunatone"), vec!["fire"]);
        assert!(b.prior_discarded.is_empty());
        // 2x on Gengar (Ghost / Poison): Fire is 1x — the row is refuted. Replayed from flat, both
        // observations: Dark / Ghost (Psychic is 2x on Gengar but 0x on Dark).
        b.observe(&dex, "lunatone", 2.0, &tgt("ghost", Some("poison"), "levitate")).expect("2x on Gengar");
        assert_eq!(survivors(&b, "lunatone"), vec!["dark", "ghost"]);
        assert!(b.state["lunatone"].iter().all(|p| *p == 0.0 || *p == 1.0 / 16.0), "flat mass");
        assert_eq!(b.prior_discarded.iter().collect::<Vec<_>>(), vec!["lunatone"]);
        assert_eq!(b.revision, 2);
    }

    /// The loud half stays: a log NO Hidden Power type explains is refused, even under the flat prior.
    #[test]
    fn a_log_no_hidden_power_type_explains_is_still_refused() {
        let dex = Dex::for_gen(3);
        let mut b = HpBelief::default();
        // 2x on Normal: Fighting only; 2x on Ghost / Poison: Dark / Ghost / Psychic — disjoint.
        b.observe(&dex, "lunatone", 2.0, &tgt("normal", None, "naturalcure")).expect("fighting survives");
        let e = b.observe(&dex, "lunatone", 2.0, &tgt("ghost", Some("poison"), "levitate")).expect_err("no type explains both");
        assert!(e.message().contains("all candidates eliminated"), "{e}");
    }
}
