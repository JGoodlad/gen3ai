//! Dex — static game data (species / moves / items / abilities / type chart /
//! natures / learnsets), generation-parameterized.
//!
//! **Source of truth.** Loads this repo's `data/pokemon/*.json` — the SAME files
//! `agents.gen3_data` reads (see root CLAUDE.md "Data Dependencies"), not a
//! re-derivation from poke-env/Showdown. So the Rust dex and the Python facade
//! agree on every base stat and base power by construction; the one piece of
//! *logic* (move category derivation) is pinned by a parity test
//! (`tests/dex_parity.rs`, golden from `harness/gen_dex_golden.py`).
//!
//! **Cross-gen shape.** Showdown layers data gen9→gen3 as deltas. Here, the
//! generation is an explicit parameter and the only gen-specific *logic* is
//! [`moves::derive_category`]; everything else is data. Other gens become a new
//! data file + a category branch, never an engine edit.

pub mod abilities;
pub mod accmod;
pub mod items;
pub mod moves;
pub mod species;
pub mod type_chart;
pub mod types;

pub use abilities::{AbilityData, ContactProc, DmgFold, DmgMod, StatusImmune, StatusImmunePhase};
pub use accmod::{AccMod, AccOp, AccSide};
pub use items::{BerryEffect, CritBoost, ItemData, StatMods, TypeBoost, TypeBoostFold};
pub use moves::MoveData;
pub use species::SpeciesData;
pub use type_chart::TypeChart;
pub use types::{BaseStats, MoveCategory, Type};

use crate::json::Json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// A nature's stat multipliers (atk/def/spa/spd/spe; HP is never affected).
#[derive(Debug, Clone)]
pub struct NatureData {
    pub name: String,
    pub atk: f64,
    pub def: f64,
    pub spa: f64,
    pub spd: f64,
    pub spe: f64,
}

/// Normalize an identifier the way poke-env's `to_id_str` does: keep ASCII
/// alphanumerics, lowercased. Dex keys are already in this form, so lookups
/// accept a raw display name or a pre-normalized id interchangeably.
pub fn to_id(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// Default data dir: `<repo>/data/pokemon`, relative to this crate.
fn default_data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/pokemon")
}

/// Read-only handle over the static data for one generation.
pub struct Dex {
    gen: u8,
    species: HashMap<String, SpeciesData>,
    moves: HashMap<String, MoveData>,
    items: HashMap<String, ItemData>,
    abilities: HashMap<String, AbilityData>,
    natures: HashMap<String, NatureData>,
    type_chart: TypeChart,
    learnset: HashMap<String, Vec<String>>,
    /// Move-ID aliases (`gen3_move_alias_resolution_v1`): `alias_id -> canonical_move_id`
    /// (e.g. `wisp -> willowisp`, `sd -> swordsdance`, `hpice -> hiddenpowerice`), mirroring
    /// how Showdown's `dex.moves.get()` resolves a packed-team token through `aliases.ts`.
    /// A packed team CAN carry an alias (the sample pool's Gengar writes `wisp`), and the sim
    /// runs it as the canonical move — so the port MUST resolve it too, else `moves()` returns
    /// `None`, the move NO-OPs, and the draw count desyncs bit-for-bit (the e2e_86 cascade).
    move_aliases: HashMap<String, String>,
}

impl Dex {
    /// Open the dex for a generation from the default in-repo data dir.
    /// Crash-don't-drop: panics with a clear message on any load/parse failure
    /// (a silent data gap would corrupt everything downstream).
    pub fn for_gen(gen: u8) -> Self {
        Self::open_at(&default_data_dir(), gen)
            .unwrap_or_else(|e| panic!("dex load failed (gen {gen}): {e}"))
    }

    /// Open the dex from an explicit data directory (testing / non-default layouts).
    pub fn open_at(dir: &Path, gen: u8) -> Result<Self, String> {
        let load = |name: &str| -> Result<Json, String> {
            let path = dir.join(name);
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("read {}: {e}", path.display()))?;
            Json::parse(&text).map_err(|e| format!("parse {}: {e}", path.display()))
        };
        Ok(Dex {
            gen,
            species: species::parse(&load("gen3_species.json")?)?,
            moves: moves::parse(&load("gen3_moves.json")?, gen)?,
            items: items::parse(&load("gen3_items.json")?)?,
            abilities: abilities::parse(&load("gen3_abilities.json")?)?,
            natures: parse_natures(&load("gen3_natures.json")?)?,
            type_chart: TypeChart::parse(&load("gen3_type_chart.json")?)?,
            learnset: parse_learnset(&load("gen3_learnset.json")?)?,
            move_aliases: parse_move_aliases(&load("gen3_move_aliases.json")?)?,
        })
    }

    pub fn generation(&self) -> u8 {
        self.gen
    }

    /// Species record by id (`None` if not a known gen3 species).
    pub fn species(&self, id: &str) -> Option<&SpeciesData> {
        self.species.get(&to_id(id))
    }
    /// Move record by id (`None` if unknown — e.g. an unrevealed opponent move).
    ///
    /// Resolves move-ID ALIASES first (`gen3_move_alias_resolution_v1`): a direct hit on the
    /// canonical movedex wins; otherwise the id is looked up in the alias map (`wisp ->
    /// willowisp`) and re-resolved once. This mirrors Showdown's `dex.moves.get()`, so a
    /// packed team carrying a shorthand token (a real thing in the sample pool) resolves to
    /// the SAME move the sim runs — the fix for the e2e_86 draw-count cascade (the port used
    /// to NO-OP an aliased move, drawing nothing while the sim ran it).
    pub fn moves(&self, id: &str) -> Option<&MoveData> {
        let key = to_id(id);
        if let Some(m) = self.moves.get(&key) {
            return Some(m);
        }
        // One-hop alias resolution (the alias map's values are canonical ids, never aliases,
        // so a single re-lookup suffices — no cycle risk).
        self.move_aliases
            .get(&key)
            .and_then(|canonical| self.moves.get(canonical))
    }
    /// Test-only mutable move-map access — lets a unit test FORGE a move's secondary
    /// shape to exercise a guard (e.g. the fail-loud multi-secondary panic) without a
    /// real move of that shape. Not used by the engine.
    #[cfg(test)]
    pub fn moves_mut(&mut self) -> &mut std::collections::HashMap<String, MoveData> {
        &mut self.moves
    }
    pub fn item(&self, id: &str) -> Option<&ItemData> {
        self.items.get(&to_id(id))
    }
    pub fn ability(&self, id: &str) -> Option<&AbilityData> {
        self.abilities.get(&to_id(id))
    }
    /// Every gen-3 ability id (the 76 rows of `gen3_abilities.json`), unordered.
    ///
    /// Exists so a hand-maintained ability ALLOWLIST can be pinned against the dex
    /// rather than against another hand-maintained list — see
    /// `event::TRACE_COPYABLE`, whose drift from the modeled set shipped three times
    /// before `trace_copyable_covers_every_gen3_ability` closed it.
    pub fn ability_ids(&self) -> impl Iterator<Item = &str> {
        self.abilities.keys().map(String::as_str)
    }
    pub fn nature(&self, name: &str) -> Option<&NatureData> {
        self.natures.get(&to_id(name))
    }
    pub fn type_chart(&self) -> &TypeChart {
        &self.type_chart
    }

    /// The gen3 legal movepool for a species (empty slice if the species is unknown).
    pub fn learnset(&self, species: &str) -> &[String] {
        self.learnset.get(&to_id(species)).map_or(&[], Vec::as_slice)
    }
    /// Whether `species` can legally carry `move_id` in gen3 (the hard legality gate).
    pub fn can_learn(&self, species: &str, move_id: &str) -> bool {
        let m = to_id(move_id);
        self.learnset
            .get(&to_id(species))
            .is_some_and(|l| l.iter().any(|x| *x == m))
    }

    pub fn species_count(&self) -> usize {
        self.species.len()
    }
    pub fn move_count(&self) -> usize {
        self.moves.len()
    }
}

fn parse_natures(root: &Json) -> Result<HashMap<String, NatureData>, String> {
    let obj = root.as_object().ok_or("natures: expected a JSON object")?;
    Ok(obj
        .iter()
        .map(|(name, v)| {
            (
                name.clone(),
                NatureData {
                    name: name.clone(),
                    atk: v.f64_or("atk", 1.0),
                    def: v.f64_or("def", 1.0),
                    spa: v.f64_or("spa", 1.0),
                    spd: v.f64_or("spd", 1.0),
                    spe: v.f64_or("spe", 1.0),
                },
            )
        })
        .collect())
}

fn parse_learnset(root: &Json) -> Result<HashMap<String, Vec<String>>, String> {
    let obj = root.as_object().ok_or("learnset: expected a JSON object")?;
    let mut out = HashMap::with_capacity(obj.len());
    for (sid, arr) in obj {
        let moves = arr
            .as_array()
            .ok_or("learnset entry: expected an array")?
            .iter()
            .filter_map(|m| m.as_str().map(str::to_string))
            .collect();
        out.insert(sid.clone(), moves);
    }
    Ok(out)
}

/// Parse `gen3_move_aliases.json` — a flat `{alias_id: canonical_move_id}` map
/// (`gen3_move_alias_resolution_v1`). Keys + values are already normalized move ids.
fn parse_move_aliases(root: &Json) -> Result<HashMap<String, String>, String> {
    let obj = root.as_object().ok_or("move_aliases: expected a JSON object")?;
    let mut out = HashMap::with_capacity(obj.len());
    for (alias, canonical) in obj {
        let c = canonical
            .as_str()
            .ok_or("move_aliases entry: expected a string canonical id")?;
        out.insert(alias.clone(), c.to_string());
    }
    Ok(out)
}

#[cfg(test)]
mod pp_tests {
    use super::*;

    // `gen3_pp_tracking_v1`: a moveslot's in-battle MAX PP is the ctor's
    // `calculatePP(move, 3) = pp * 8 / 5` (default 3 PP-ups) for a normal move, or the raw
    // `pp` for a `noPPBoosts` move. Pins the values the port initializes `MonState::move_pp`
    // to — VERIFIED vs the sim's `moveSlots[k].maxpp` (harness/probe_pp_struggle_rng.js).
    #[test]
    fn max_pp_is_base_pp_times_eight_fifths_gen3() {
        let d = Dex::for_gen(3);
        let cases = [
            ("surf", 15u16, 24u16),         // 15 * 8/5 = 24
            ("earthquake", 10, 16),          // 10 * 8/5 = 16
            ("thunderbolt", 15, 24),
            ("thunder", 10, 16),
            ("bodyslam", 15, 24),
            ("splash", 40, 64),              // 40 * 8/5 = 64
            ("extremespeed", 5, 8),          // 5 * 8/5 = 8 (the lowest)
            ("crunch", 15, 24),
            ("rockslide", 10, 16),
            ("recover", 20, 32),
        ];
        for (id, base, maxpp) in cases {
            let m = d.moves(id).unwrap_or_else(|| panic!("move {id} missing"));
            assert_eq!(m.pp, base, "{id} base pp");
            assert!(!m.no_pp_boosts, "{id} is not noPPBoosts");
            assert_eq!(m.max_pp(), maxpp, "{id} max_pp = base * 8/5");
        }
        // STRUGGLE: noPPBoosts → max_pp == raw pp == 1 (NOT 8/5-scaled).
        let struggle = d.moves("struggle").expect("struggle present");
        assert!(struggle.no_pp_boosts, "struggle is noPPBoosts");
        assert_eq!(struggle.pp, 1, "struggle base pp 1");
        assert_eq!(struggle.max_pp(), 1, "struggle max_pp stays 1 (no PP-ups)");
        // Struggle's gen-3 facts the port relies on: typeless resolution happens in the engine
        // (the dex entry keeps type Normal), BP 50, accuracy 100 (NOT never-miss → draws acc),
        // recoil fraction 0.25 (the gen-3 `recoil:[1,4]` path).
        assert_eq!(struggle.base_power, 50, "struggle BP 50");
        assert_eq!(struggle.accuracy, 100, "struggle accuracy 100");
        assert!(!struggle.never_miss, "gen-3 struggle is NOT never-miss (it draws accuracy)");
        assert!((struggle.recoil_fraction - 0.25).abs() < 1e-9, "struggle recoil fraction 0.25");
    }
}

#[cfg(test)]
mod alias_tests {
    use super::*;

    /// `gen3_move_alias_resolution_v1`: `dex.moves()` resolves a packed-team move ALIAS to
    /// the canonical move (mirroring Showdown's `dex.moves.get()` through `aliases.ts`), so a
    /// team carrying a shorthand token (the sample pool's Gengar writes `wisp`) runs the SAME
    /// move the sim runs — the fix for the e2e_86 draw-count cascade (pre-fix the port
    /// returned `None` for `wisp` and NO-OP'd the move, drawing nothing while the sim ran it).
    #[test]
    fn move_aliases_resolve_to_the_canonical_move() {
        let d = Dex::for_gen(3);
        // The e2e_86 repro alias + a spread of the shorthand family.
        let cases = [
            ("wisp", "willowisp"),
            ("wow", "willowisp"),
            ("sd", "swordsdance"),
            ("cm", "calmmind"),
            ("dd", "dragondance"),
            ("sub", "substitute"),
            ("twave", "thunderwave"),
            ("tbolt", "thunderbolt"),
            ("eq", "earthquake"),
            ("stoss", "seismictoss"),
            // The TYPED Hidden Power shorthand resolves to the DISTINCT typed name (NOT bare
            // `hiddenpower`) — the port's `gen3_typed_hidden_power_ids_v1` representation.
            ("hpice", "hiddenpowerice"),
            ("hpgrass", "hiddenpowergrass"),
        ];
        for (alias, canonical) in cases {
            let via_alias = d.moves(alias).unwrap_or_else(|| panic!("alias {alias} did not resolve"));
            let direct = d.moves(canonical).unwrap_or_else(|| panic!("canonical {canonical} missing"));
            assert_eq!(
                to_id(&via_alias.id),
                to_id(&direct.id),
                "alias {alias} must resolve to the canonical move {canonical} (same move id)"
            );
        }
        // The bare `hp` alias is DELIBERATELY excluded (it maps to bare `hiddenpower`, which
        // the port never uses — it represents HP by its typed name). A canonical id resolves
        // unchanged; a genuinely-unknown token is still `None`.
        assert!(d.moves("hp").is_none(), "bare `hp` alias is excluded (typed-HP model)");
        assert!(d.moves("earthquake").is_some(), "a canonical id still resolves directly");
        assert!(d.moves("notarealmove").is_none(), "an unknown token stays None");
    }
}

#[cfg(test)]
mod batch5_tests {
    use super::*;

    // `gen3_move_coverage_batch5_v1`: TAUNT does NOT block the bp-0 VARIABLE-BP family
    // (their resolved gen3 category is Physical, like the fixed-damage family) nor the
    // reactive fixed-damage moves — but Sleep Talk (a genuine Status move) IS blocked.
    // PROBE-verified vs the live sim (a taunted Snorlax's request: return/flail/counter
    // stay `disabled:false`, sleeptalk flips `disabled:true`).
    #[test]
    fn taunt_blocks_sleep_talk_but_not_the_variable_bp_or_reactive_families() {
        let d = Dex::for_gen(3);
        for id in ["return", "frustration", "flail", "reversal", "lowkick"] {
            let m = d.moves(id).unwrap();
            assert!(m.is_variable_bp(), "{id} is in the variable-BP family");
            assert!(!m.blocked_by_taunt(), "{id} must stay selectable under Taunt");
        }
        for id in ["counter", "mirrorcoat", "endeavor", "seismictoss"] {
            assert!(!d.moves(id).unwrap().blocked_by_taunt(), "{id} must stay selectable under Taunt");
        }
        // The BARE `hiddenpower` (num 237, data BP 0 → derived Status) is a real damaging move
        // (`gen3_iv_derived_hidden_power_bp_v1`): a TAUNTED mon's Hidden Power stays selectable
        // (VERIFIED vs the sim — the ab_233_8 over-KO). The typed ids carry BP 70 → non-Status.
        for id in ["hiddenpower", "hiddenpowerdark", "hiddenpowergrass", "hiddenpowerice"] {
            assert!(!d.moves(id).unwrap().blocked_by_taunt(), "{id} must stay selectable under Taunt");
        }
        let st = d.moves("sleeptalk").unwrap();
        assert!(st.blocked_by_taunt(), "Sleep Talk (Status) IS taunt-blocked");
        // The pool-exclusion flags ride the data (never hand-listed): sleeptalk excludes
        // ITSELF via nosleeptalk; solarbeam is BOTH nosleeptalk and charge; fly is charge.
        assert!(st.no_sleep_talk, "sleeptalk carries flags.nosleeptalk");
        let sb = d.moves("solarbeam").unwrap();
        assert!(sb.no_sleep_talk && sb.is_charge, "solarbeam carries both exclusion flags");
        assert!(d.moves("fly").unwrap().is_charge, "fly carries flags.charge");
        assert!(!d.moves("rest").unwrap().no_sleep_talk, "Rest IS Sleep-Talk-eligible");
        // Low Kick's data need: the species weighthg (round(weightkg*10)) — the probed
        // ladder anchors.
        for (sp, hg) in [("pichu", 20u32), ("wobbuffet", 285), ("gengar", 405), ("slaking", 1305), ("snorlax", 4600), ("skarmory", 505)] {
            assert_eq!(d.species(sp).unwrap().weighthg, hg, "{sp} weighthg");
        }
    }
}
