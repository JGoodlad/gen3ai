//! [`PMon`] — poke-env's `Pokemon` as the READING holds it, one method per poke-env method.
//!
//! Every method names the `pokemon.py` / `move.py` method it mirrors. The state is exactly the
//! slots `LiveView.from_battle` (and the transitions that write them) read; the ones poke-env
//! keeps for other consumers (`_last_cant_reason`, `_must_recharge`, `_preparing_*`,
//! `_active_turns`, `_dancing`, level / gender / shiny, height / weight) have no reader on this
//! path and are not carried.

use super::dex::{self, EffectId, Status};
use super::tables::UNKNOWN_ITEM;

/// `Pokemon._boosts`' key order.
pub const BOOST_KEYS: [&str; 7] = ["accuracy", "atk", "def", "evasion", "spa", "spd", "spe"];
/// `Pokemon._stats`' key order (and `baseStats`').
pub const STAT_KEYS: [&str; 6] = ["hp", "atk", "def", "spa", "spd", "spe"];

pub type R<T> = Result<T, String>;

/// poke-env's `Move` — the fields the reading reads.
#[derive(Debug, Clone, PartialEq)]
pub struct PMove {
    /// `Move._id` — the dex id, TYPED for a Hidden Power built from a typed raw id.
    pub id: String,
    pub current_pp: u32,
    pub from_transform: bool,
    /// `Move._base_power_override` — the digits of a `hiddenpowerfire70` raw id.
    pub base_power_override: Option<u32>,
}

/// The fields of `Move.entry` the reading reads.
pub struct Entry {
    pub pp: u32,
    pub base_power: u32,
    pub typ: &'static str,
    pub protect_counter: bool,
}

impl PMove {
    /// `Move.__init__(move_id, gen, raw_id, from_mimic, from_transform)`: a Hidden Power built
    /// from a raw id keeps the raw id's TYPE (`hiddenpowerfire`) and its digits as the base
    /// power; `_current_pp` starts at `max_pp`, or `min(5, max_pp)` for a Transform copy.
    pub fn new(move_id: &str, raw_id: Option<&str>, from_transform: bool) -> R<PMove> {
        let mut id = move_id.to_string();
        let mut bpo = None;
        if move_id.starts_with("hiddenpower") {
            if let Some(raw) = raw_id {
                let digits: String = raw.chars().filter(|c| c.is_ascii_digit()).collect();
                if !digits.is_empty() {
                    bpo = digits.parse().ok();
                }
                id = dex::to_id(raw).chars().filter(|c| !c.is_ascii_digit()).collect();
            }
        }
        let mut m = PMove { id, current_pp: 0, from_transform, base_power_override: bpo };
        let max = m.max_pp()?;
        m.current_pp = if from_transform { max.min(5) } else { max };
        Ok(m)
    }

    /// `Move.entry`: the dex row; a `z`-prefixed id falls back to its base move; `recharge` /
    /// `fight` get poke-env's synthetic row; anything else is `ValueError("Unknown move")`.
    pub fn entry(&self) -> R<Entry> {
        let row = dex::move_row(&self.id).or_else(|| {
            self.id.strip_prefix('z').and_then(dex::move_row)
        });
        if let Some(r) = row {
            return Ok(Entry { pp: r.pp as u32, base_power: r.base_power as u32, typ: r.typ, protect_counter: r.protect_counter });
        }
        if self.id == "recharge" || self.id == "fight" {
            return Ok(Entry { pp: 1, base_power: 0, typ: "NORMAL", protect_counter: false });
        }
        Err(format!("Unknown move: {} (ValueError)", self.id))
    }

    /// `Move.max_pp` (gen 3: `entry["pp"] * 8 // 5`, no Transform cap before gen 5).
    pub fn max_pp(&self) -> R<u32> {
        Ok(self.entry()?.pp * 8 / 5)
    }

    /// `Move.use(pressure, overridden)`: `1 + pressure − overridden`, floored at 0.
    pub fn use_move(&mut self, pressure: bool, overridden: bool) {
        let dec = 1 + pressure as i64 - overridden as i64;
        self.current_pp = (self.current_pp as i64 - dec).max(0) as u32;
    }

    /// `Move.base_power`.
    pub fn base_power(&self) -> R<u32> {
        Ok(match self.base_power_override {
            Some(b) => b,
            None => self.entry()?.base_power,
        })
    }
}

/// poke-env's `MoveSet`: the learned moves, a Mimic overlay and a Transform replacement.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MoveSet {
    /// `_base_moves`, insertion-ordered by dict KEY (`Move.retrieve_id`).
    pub base: Vec<(String, PMove)>,
    /// `_mimic_move`.
    pub mimic: Option<PMove>,
    /// `_transform_moves`.
    pub transform: Option<Box<MoveSet>>,
}

impl MoveSet {
    /// `MoveSet._resolved()`.
    pub fn resolved(&self) -> &MoveSet {
        match &self.transform {
            Some(t) => t.resolved(),
            None => self,
        }
    }
    fn resolved_mut(&mut self) -> &mut MoveSet {
        if self.transform.is_some() {
            self.transform.as_mut().expect("checked").resolved_mut()
        } else {
            self
        }
    }
    /// `MoveSet.base_moves` (resolved, no Mimic substitution).
    pub fn base_moves(&self) -> &Vec<(String, PMove)> {
        &self.resolved().base
    }
    pub fn base_moves_mut(&mut self) -> &mut Vec<(String, PMove)> {
        &mut self.resolved_mut().base
    }
    /// `MoveSet.mimic_move` setter (on the resolved set).
    pub fn set_mimic(&mut self, m: Option<PMove>) {
        self.resolved_mut().mimic = m;
    }
    /// `MoveSet.moves`: the resolved base moves with the `mimic` key replaced by the Mimic
    /// overlay (keyed by its `Move.id`). Returned as `(key, move)` pairs in dict order.
    pub fn moves(&self) -> Vec<(String, PMove)> {
        let r = self.resolved();
        match &r.mimic {
            None => r.base.clone(),
            Some(mm) => r
                .base
                .iter()
                .map(|(k, v)| if k == "mimic" { (mm.id.clone(), mm.clone()) } else { (k.clone(), v.clone()) })
                .collect(),
        }
    }
    /// `key in self.moves`.
    pub fn contains(&self, key: &str) -> bool {
        let r = self.resolved();
        match &r.mimic {
            None => r.base.iter().any(|(k, _)| k == key),
            Some(mm) => r.base.iter().any(|(k, _)| (k == "mimic" && mm.id == key) || (k != "mimic" && k == key)),
        }
    }
    /// The move object `self.moves[key]` names: the Mimic overlay when `key` is its id and the
    /// base set still holds the `mimic` slot, else the base move stored under `key`.
    pub fn lookup(&self, key: &str) -> Option<MoveRef> {
        let r = self.resolved();
        if let Some(mm) = &r.mimic {
            if mm.id == key && r.base.iter().any(|(k, _)| k == "mimic") {
                return Some(MoveRef::Mimic);
            }
            if key == "mimic" {
                return None;
            }
        }
        r.base.iter().any(|(k, _)| k == key).then(|| MoveRef::Base(key.to_string()))
    }
    /// A mutable handle on the object a [`MoveRef`] names.
    pub fn get_mut(&mut self, at: &MoveRef) -> Option<&mut PMove> {
        let r = self.resolved_mut();
        match at {
            MoveRef::Mimic => r.mimic.as_mut(),
            MoveRef::Base(k) => r.base.iter_mut().find(|(kk, _)| kk == k).map(|(_, v)| v),
        }
    }
    pub fn get(&self, at: &MoveRef) -> Option<&PMove> {
        let r = self.resolved();
        match at {
            MoveRef::Mimic => r.mimic.as_ref(),
            MoveRef::Base(k) => r.base.iter().find(|(kk, _)| kk == k).map(|(_, v)| v),
        }
    }
}

/// WHICH `Move` object a poke-env lookup returned — the Mimic overlay or a base-set entry.
#[derive(Debug, Clone, PartialEq)]
pub enum MoveRef {
    Mimic,
    Base(String),
}

/// The spread a teambuilder entry declares (`TeambuilderPokemon.from_packed`).
#[derive(Debug, Clone, PartialEq)]
pub struct TbMon {
    pub nickname: Option<String>,
    pub species: Option<String>,
    pub evs: Vec<i64>,
    pub ivs: Vec<i64>,
    pub nature: Option<String>,
}

impl TbMon {
    /// `Teambuilder.parse_packed_team` → `TeambuilderPokemon.from_packed`, the spread half.
    pub fn parse_team(packed: &str) -> R<Vec<TbMon>> {
        let mut out = Vec::new();
        for pm in packed.split(']') {
            if pm.is_empty() {
                continue;
            }
            let f: Vec<&str> = pm.split('|').collect();
            if f.len() != 12 {
                return Err(format!("from_packed({pm:?}): expected 12 fields, got {} (ValueError)", f.len()));
            }
            let opt = |s: &str| if s.is_empty() { None } else { Some(s.to_string()) };
            let parse_list = |s: &str, dflt: i64| -> R<Option<Vec<i64>>> {
                if s.is_empty() {
                    return Ok(None);
                }
                s.split(',')
                    .map(|x| if x.is_empty() { Ok(dflt) } else { x.parse::<i64>().map_err(|_| format!("int({x:?})")) })
                    .collect::<R<Vec<i64>>>()
                    .map(Some)
            };
            out.push(TbMon {
                nickname: opt(f[0]),
                species: opt(f[1]),
                evs: parse_list(f[6], 0)?.unwrap_or_else(|| vec![0; 6]),
                ivs: parse_list(f[8], 31)?.unwrap_or_else(|| vec![31; 6]),
                nature: opt(f[5]),
            });
        }
        Ok(out)
    }
}

/// poke-env's `Pokemon`, as the reading holds it.
#[derive(Debug, Clone, PartialEq)]
pub struct PMon {
    pub species: String,
    base_stats: [u16; 6],
    type_1: &'static str,
    type_2: Option<&'static str>,
    possible_abilities: Vec<&'static str>,
    /// `_ability` — the BASE slot.
    ability_base: Option<String>,
    temporary_ability: Option<String>,
    forme_change_ability: Option<String>,
    pub name: Option<String>,
    /// `_item`, the `unknown_item` sentinel included.
    pub item: Option<String>,
    /// `_consumed_item`, raw (id-normalised only at the read-model).
    pub consumed_item: Option<String>,
    last_details: String,
    pub active: bool,
    pub revealed: bool,
    pub boosts: [i32; 7],
    pub current_hp: Option<u32>,
    pub max_hp: Option<u32>,
    /// `_effects`, insertion-ordered.
    pub effects: Vec<(EffectId, u32)>,
    pub status: Option<Status>,
    pub status_counter: u32,
    pub protect_counter: u32,
    pub stats: [Option<i64>; 6],
    pub moves: MoveSet,
    temporary_base_stats: Option<[u16; 6]>,
    temporary_types: Vec<&'static str>,
    pub ivs: Option<Vec<i64>>,
    pub evs: Option<Vec<i64>>,
    pub nature: Option<String>,
    /// `_last_request` — the roster record `was_illusioned` re-applies.
    last_request: Option<super::board_reading::ReqMon>,
}

impl PMon {
    fn blank() -> PMon {
        PMon {
            species: String::new(),
            base_stats: [0; 6],
            type_1: "NORMAL",
            type_2: None,
            possible_abilities: Vec::new(),
            ability_base: None,
            temporary_ability: None,
            forme_change_ability: None,
            name: None,
            item: Some(UNKNOWN_ITEM.to_string()),
            consumed_item: None,
            last_details: String::new(),
            active: false,
            revealed: false,
            boosts: [0; 7],
            current_hp: Some(0),
            max_hp: Some(0),
            effects: Vec::new(),
            status: None,
            status_counter: 0,
            protect_counter: 0,
            stats: [None; 6],
            moves: MoveSet::default(),
            temporary_base_stats: None,
            temporary_types: Vec::new(),
            ivs: None,
            evs: None,
            nature: None,
            last_request: None,
        }
    }

    /// `Pokemon(species=…)` — `_update_from_pokedex(species)`.
    pub fn from_species(species: &str, name: Option<String>) -> R<PMon> {
        let mut m = PMon::blank();
        m.update_from_pokedex(species, true)?;
        if name.is_some() {
            m.name = name;
        }
        Ok(m)
    }

    /// `Pokemon(details=…)` — `_update_from_details(details)`.
    pub fn from_details(details: &str, name: Option<String>) -> R<PMon> {
        let mut m = PMon::blank();
        m.update_from_details(details)?;
        if name.is_some() {
            m.name = name;
        }
        Ok(m)
    }

    /// `Pokemon(request_pokemon=…)` — `update_from_request(request)`.
    pub fn from_request(req: &super::board_reading::ReqMon, name: Option<String>) -> R<PMon> {
        let mut m = PMon::blank();
        m.update_from_request(req)?;
        if name.is_some() {
            m.name = name;
        }
        Ok(m)
    }

    // ---------------------------------------------------------------- the read accessors

    /// `Pokemon.ability`: the temporary slot, else the forme-change one, else the base slot.
    pub fn ability(&self) -> Option<&str> {
        self.temporary_ability.as_deref().or(self.forme_change_ability.as_deref()).or(self.ability_base.as_deref())
    }
    /// `Pokemon.ability` setter: the BASE slot while it is None, the temporary slot after.
    pub fn set_ability(&mut self, a: &str) {
        if self.ability_base.is_none() {
            self.ability_base = Some(dex::to_id(a));
        } else {
            self.temporary_ability = Some(dex::to_id(a));
        }
    }
    /// `Pokemon.temporary_ability` setter.
    pub fn set_temporary_ability(&mut self, a: Option<&str>) {
        self.temporary_ability = a.map(dex::to_id);
    }
    pub fn temporary_ability(&self) -> Option<&str> {
        self.temporary_ability.as_deref()
    }
    /// `mon._ability = None` (the `-ability` handler's Trace case).
    pub fn clear_base_ability(&mut self) {
        self.ability_base = None;
    }
    /// `Pokemon._last_details` — the details string a later one is compared against.
    pub fn last_details(&self) -> &str {
        &self.last_details
    }
    /// `Pokemon.base_ability`.
    pub fn base_ability(&self) -> Option<&str> {
        self.forme_change_ability.as_deref().or(self.ability_base.as_deref())
    }
    /// `Pokemon.fainted`.
    pub fn fainted(&self) -> bool {
        self.status == Some(Status::Fnt)
    }
    /// `Pokemon.current_hp` (`_current_hp or 0`).
    pub fn current_hp(&self) -> u32 {
        self.current_hp.unwrap_or(0)
    }
    /// `Pokemon.max_hp` (`_max_hp or 0`).
    pub fn max_hp(&self) -> u32 {
        self.max_hp.unwrap_or(0)
    }
    /// `Pokemon.current_hp_fraction`: `current_hp / max_hp` if `current_hp` else `0`.
    pub fn hp_fraction(&self) -> f64 {
        let c = self.current_hp();
        if c != 0 {
            c as f64 / self.max_hp() as f64
        } else {
            0.0
        }
    }
    /// `Pokemon.base_stats`: the Transform overlay, else the dex's.
    pub fn base_stats(&self) -> [u16; 6] {
        self.temporary_base_stats.unwrap_or(self.base_stats)
    }
    /// `Pokemon.types` (gen 3: no Terastallization): the temporary types, else `[type_1,
    /// type_2]` — as `PokemonType` member NAMES.
    pub fn types(&self) -> Vec<&'static str> {
        if !self.temporary_types.is_empty() {
            return self.temporary_types.clone();
        }
        let mut t = vec![self.type_1];
        if let Some(t2) = self.type_2 {
            t.push(t2);
        }
        t
    }
    /// `Pokemon.base_species`: the dex row's `baseSpecies`, id'd (a `KeyError` for a species
    /// the dex does not hold — R10's nickname-species mon).
    pub fn base_species(&self) -> R<&'static str> {
        Ok(dex::species(&self.species)?.base_species)
    }
    /// `Pokemon.identifies_as(ident)`.
    pub fn identifies_as(&self, ident: &str) -> R<bool> {
        let b = self.base_species()?;
        Ok(b == dex::to_id(ident) || ident.split('-').any(|s| dex::to_id(s) == b))
    }
    /// `Pokemon.transformed`.
    pub fn transformed(&self) -> bool {
        self.moves.transform.is_some()
    }
    pub fn has_effect(&self, e: EffectId) -> bool {
        self.effects.iter().any(|(x, _)| *x == e)
    }

    // ---------------------------------------------------------------- the pokedex / details

    /// `Pokemon._update_from_pokedex(species, store_species)`: base stats, types, and either the
    /// forme-change ability (a Mega / Primal forme) or the possible abilities — setting the BASE
    /// slot outright when there is exactly ONE (the single-possible-ability inference, rule V8).
    pub fn update_from_pokedex(&mut self, species: &str, store_species: bool) -> R<()> {
        let sid = dex::to_id(species);
        let row = dex::species(&sid)?;
        if store_species {
            self.species = sid;
        }
        self.base_stats = row.base_stats;
        self.type_1 = dex::type_from_name(row.types.first().copied().unwrap_or("???"))?;
        self.type_2 = if row.types.len() == 1 { None } else { Some(dex::type_from_name(row.types[1])?) };
        if let Some(fca) = row.forme_change_ability {
            self.forme_change_ability = Some(fca.to_string());
        } else if self.forme_change_ability.is_none() {
            self.possible_abilities = row.abilities.to_vec();
            if self.possible_abilities.len() == 1 {
                self.ability_base = Some(self.possible_abilities[0].to_string());
            }
        } else {
            self.forme_change_ability = None;
        }
        Ok(())
    }

    /// `Pokemon._update_from_details(details)`: a changed details string re-reads the pokedex
    /// whenever its species token differs from `_species` (a raw `Blissey` never equals the
    /// id `blissey`, so a CHANGED details string always re-reads it).
    pub fn update_from_details(&mut self, details: &str) -> R<()> {
        if details == self.last_details {
            return Ok(());
        }
        self.last_details = details.to_string();
        let d = details.replace(", shiny", "");
        let mut parts: Vec<&str> = d.split(", ").collect();
        if let Some(i) = parts.iter().position(|p| p.starts_with("tera:")) {
            parts.remove(i);
        }
        let species: String = match parts.len() {
            3 | 2 => parts[0].to_string(),
            _ => dex::to_id(parts[0]),
        };
        if species != self.species {
            self.update_from_pokedex(&species, true)?;
        }
        Ok(())
    }

    /// `Pokemon.forme_change(species)`: re-read the pokedex WITHOUT renaming the species (R11).
    pub fn forme_change(&mut self, species: &str) -> R<()> {
        let s = species.split(',').next().unwrap_or("");
        self.update_from_pokedex(s, false)
    }

    // ---------------------------------------------------------------- the request

    /// `Pokemon.update_from_request(request_pokemon)`.
    pub fn update_from_request(&mut self, req: &super::board_reading::ReqMon) -> R<()> {
        self.active = req.active;
        if self.ability().is_none() {
            let base = req.base_ability.as_deref().ok_or("request mon without baseAbility (KeyError)")?;
            self.set_ability(base);
        }
        if let Some(a) = &req.ability {
            if Some(a.as_str()) != req.base_ability.as_deref() {
                self.set_temporary_ability(Some(a));
            }
        }
        self.last_request = Some(req.clone());
        self.set_hp_status(&req.condition, true)?;
        self.name = Some(req.ident.get(4..).unwrap_or("").to_string());
        self.item = Some(req.item.clone());
        if !req.item.is_empty() {
            self.consumed_item = None;
        }
        self.update_from_details(&req.details)?;
        for m in &req.moves {
            self.add_move(m)?;
        }
        if let Some(st) = &req.stats {
            for (k, v) in st {
                if let Some(i) = STAT_KEYS.iter().position(|s| s == k) {
                    self.stats[i] = Some(*v);
                } else {
                    return Err(format!("request stat {k:?} is not a poke-env stat key"));
                }
            }
        }
        Ok(())
    }

    /// `Pokemon.backfill_spread_from_teambuilder(tb)` — IVs / EVs / nature, once.
    pub fn backfill_spread(&mut self, tb: &TbMon) {
        if self.ivs.is_some() {
            return;
        }
        self.evs = Some(tb.evs.clone());
        self.ivs = Some(tb.ivs.clone());
        self.nature = Some(tb.nature.as_deref().map(str::to_lowercase).unwrap_or_else(|| "serious".into()));
    }

    /// `Pokemon._add_move(move_id)`: stored under `Move.retrieve_id`, only when
    /// `Move.should_be_stored`; created with `raw_id` (so a typed Hidden Power keeps its type)
    /// and `from_transform=self.transformed`. Returns WHICH move object poke-env returned.
    pub fn add_move(&mut self, move_id: &str) -> R<Option<MoveRef>> {
        let id = dex::retrieve_id(move_id);
        if self.moves.contains(&id) {
            return Ok(self.moves.lookup(&id));
        }
        if !dex::should_be_stored(&id) {
            return Ok(None);
        }
        let transformed = self.transformed();
        if !self.moves.base_moves().iter().any(|(k, _)| *k == id) {
            let m = PMove::new(&id, Some(move_id), transformed)?;
            self.moves.base_moves_mut().push((id.clone(), m));
        }
        Ok(Some(MoveRef::Base(id)))
    }

    // ---------------------------------------------------------------- HP / status

    /// `Pokemon.set_hp_status(hp_status, store)`: `0 fnt` is a faint; an `hp status` token
    /// writes `_status` DIRECTLY (not through the setter — no counter reset) and ends a Yawn on
    /// a new sleep; a bare `hp` CLEARS the status.
    pub fn set_hp_status(&mut self, hp_status: &str, store: bool) -> R<()> {
        if hp_status == "0 fnt" {
            self.faint();
            return Ok(());
        }
        let hp = if hp_status.contains(' ') {
            let parts: Vec<&str> = hp_status.split(' ').collect();
            if parts.len() != 2 {
                return Err(format!("set_hp_status({hp_status:?}): too many values to unpack (ValueError)"));
            }
            self.status = Some(dex::status_from(parts[1])?);
            if self.status == Some(Status::Slp) {
                let yawn = dex::effect_from_message("yawn");
                if self.has_effect(yawn) {
                    self.end_effect("yawn");
                }
            }
            parts[0]
        } else {
            self.status = None;
            hp_status
        };
        let digits: String = hp.chars().filter(|c| c.is_ascii_digit() || *c == '/').collect();
        let parts: Vec<&str> = digits.split('/').collect();
        if parts.len() != 2 {
            return Err(format!("set_hp_status({hp_status:?}): expected cur/max (ValueError)"));
        }
        let cur: u32 = parts[0].parse().map_err(|_| format!("set_hp_status({hp_status:?}): int() (ValueError)"))?;
        let max: u32 = parts[1].parse().map_err(|_| format!("set_hp_status({hp_status:?}): int() (ValueError)"))?;
        self.current_hp = Some(cur);
        self.max_hp = Some(max);
        if store {
            self.stats[0] = Some(max as i64);
        }
        Ok(())
    }

    /// `Pokemon.status` SETTER (the `-status` handler). With the fork's R1 fix
    /// (`one_sided_view.md` §4b): a status that CHANGES starts its own count.
    pub fn set_status(&mut self, status: Option<Status>) {
        if status != self.status {
            self.status_counter = 0;
        }
        self.status = status;
    }

    /// The badly-poisoned STAGE — `Pokemon.note_residual_chip` (the fork's PE-R1b fix,
    /// `gen3_pe_reading_fixes_v1`): one residual toxic chip (a `-damage … [from] psn` on a mon
    /// holding `tox`) is one stage, capped at the sim's 15 (`tox.onResidual` ramps the stage
    /// before it chips). Upstream poke-env ticked at every `|turn|` instead.
    pub fn note_residual_chip(&mut self) {
        if self.status == Some(Status::Tox) {
            self.status_counter = (self.status_counter + 1).min(15);
        }
    }

    /// `Pokemon.cure_status(status)`: a NAMED status clears only itself (and the counter); no
    /// name clears a living mon's status and leaves the counter.
    pub fn cure_status(&mut self, status: Option<&str>) -> R<()> {
        match status {
            Some(s) if !s.is_empty() => {
                if Some(dex::status_from(s)?) == self.status {
                    self.status = None;
                    self.status_counter = 0;
                }
            }
            Some(_) => {}
            None => {
                if !self.fainted() {
                    self.status = None;
                }
            }
        }
        Ok(())
    }

    /// `Pokemon.faint()`: HP 0, FNT, the temporary ability / base stats / Transform / Mimic
    /// dropped, every effect cleared — and the stat stages CLEARED, which is the sim's truth (its
    /// faint `clearVolatile` zeroes `boosts`; the fork's PE-V10 fix, `gen3_pe_reading_fixes_v1`).
    pub fn faint(&mut self) {
        self.clear_boosts();
        self.current_hp = Some(0);
        self.status = Some(Status::Fnt);
        self.temporary_ability = None;
        self.temporary_base_stats = None;
        self.moves.transform = None;
        self.moves.set_mimic(None);
        self.clear_effects();
    }

    // ---------------------------------------------------------------- effects

    /// `Pokemon.start_effect(effect_str, details)`.
    pub fn start_effect(&mut self, effect_str: &str, details: Option<StartDetails>) -> R<()> {
        let e = dex::effect_from_message(effect_str);
        match self.effects.iter_mut().find(|(x, _)| *x == e) {
            None => self.effects.push((e, 0)),
            Some((_, c)) => {
                if dex::effect(e).action_countable {
                    *c += 1;
                }
            }
        }
        if dex::effect(e).breaks_protect {
            self.protect_counter = 0;
        }
        let name = dex::effect(e).name;
        match (name, details) {
            ("TYPECHANGE", Some(StartDetails::Types(types))) => {
                self.temporary_types = types
                    .split('/')
                    .map(dex::type_from_name)
                    .collect::<R<Vec<_>>>()?;
            }
            ("SKILL_SWAP", Some(StartDetails::Abilities(ab))) if !ab.is_empty() => {
                if self.ability().is_none() {
                    let a1 = ab.get(1).ok_or("skill swap details[1] (IndexError)")?.clone();
                    self.set_ability(&a1);
                }
                self.set_temporary_ability(Some(&ab[0]));
            }
            _ => {}
        }
        Ok(())
    }

    /// `Pokemon.end_effect(effect_str)`: pop it; TYPECHANGE also drops the temporary types and
    /// SKILL_SWAP the temporary ability.
    pub fn end_effect(&mut self, effect_str: &str) {
        let e = dex::effect_from_message(effect_str);
        self.effects.retain(|(x, _)| *x != e);
        match dex::effect(e).name {
            "TYPECHANGE" => self.temporary_types.clear(),
            "SKILL_SWAP" => self.temporary_ability = None,
            "QUARK_DRIVE" | "PROTOSYNTHESIS" => {
                let pre = dex::effect(e).name.replace('_', "");
                for suffix in ["ATK", "DEF", "SPA", "SPD", "SPE"] {
                    if let Some(x) = dex::effect_by_name(&format!("{pre}{suffix}")) {
                        self.effects.retain(|(y, _)| *y != x);
                    }
                }
            }
            _ => {}
        }
    }

    /// `Pokemon._clear_effects()`: `end_effect(effect.name)` for every effect.
    pub fn clear_effects(&mut self) {
        let names: Vec<&'static str> = self.effects.iter().map(|(e, _)| dex::effect(*e).name).collect();
        for n in names {
            self.end_effect(n);
        }
    }

    /// `Pokemon.end_turn()` — every `|turn|` for the ACTIVE mons: a turn-countable effect counts,
    /// an `ends_on_turn` one ends (rule V4). The badly-poisoned stage is NOT ticked here (upstream
    /// poke-env did — the fork's PE-R1b fix); it follows the residual chips ([`Self::note_residual_chip`]).
    pub fn end_turn(&mut self) {
        let snapshot: Vec<EffectId> = self.effects.iter().map(|(e, _)| *e).collect();
        for e in snapshot {
            let row = dex::effect(e);
            if row.turn_countable {
                if let Some((_, c)) = self.effects.iter_mut().find(|(x, _)| *x == e) {
                    *c += 1;
                }
            }
            if row.ends_on_turn {
                self.end_effect(row.name);
            }
        }
    }

    // ---------------------------------------------------------------- boosts

    fn boost_idx(stat: &str) -> R<usize> {
        BOOST_KEYS.iter().position(|k| *k == stat).ok_or_else(|| format!("boosts[{stat:?}]: KeyError"))
    }
    /// `Pokemon.boost(stat, amount)`: add, clamped to ±6.
    pub fn boost(&mut self, stat: &str, amount: i32) -> R<()> {
        let i = Self::boost_idx(stat)?;
        self.boosts[i] = (self.boosts[i] + amount).clamp(-6, 6);
        Ok(())
    }
    /// `Pokemon.set_boost(stat, amount)` (asserts `|amount| <= 6`).
    pub fn set_boost(&mut self, stat: &str, amount: i32) -> R<()> {
        if amount.abs() > 6 {
            return Err(format!("set_boost({stat}, {amount}): AssertionError"));
        }
        let i = Self::boost_idx(stat)?;
        self.boosts[i] = amount;
        Ok(())
    }
    pub fn boost_mut(&mut self, stat: &str) -> R<&mut i32> {
        let i = Self::boost_idx(stat)?;
        Ok(&mut self.boosts[i])
    }
    pub fn clear_boosts(&mut self) {
        self.boosts = [0; 7];
    }
    pub fn clear_negative_boosts(&mut self) {
        for b in &mut self.boosts {
            if *b < 0 {
                *b = 0;
            }
        }
    }
    pub fn clear_positive_boosts(&mut self) {
        for b in &mut self.boosts {
            if *b > 0 {
                *b = 0;
            }
        }
    }
    pub fn invert_boosts(&mut self) {
        for b in &mut self.boosts {
            *b = -*b;
        }
    }

    // ---------------------------------------------------------------- moves

    /// `Pokemon.moved(move_id, failed, use, reveal, pressure)`: reveal (add) the move, use it,
    /// the protect streak (rule V6), a sleeping mon's count (rule V5), and the silent ends
    /// (Glaive Rush; a damaging Electric move ends Charge, a damaging Fire move ends Flash
    /// Fire — poke-env's rule, whatever the gen).
    pub fn moved(&mut self, move_id: &str, failed: bool, use_: bool, reveal: bool, pressure: bool) -> R<()> {
        let mut at: Option<MoveRef> = None;
        if reveal {
            at = self.add_move(move_id)?;
        }
        if use_ {
            if let Some(a) = &at {
                if let Some(m) = self.moves.get_mut(a) {
                    m.use_move(pressure, false);
                }
            }
        }
        let mv: Option<PMove> = at.as_ref().and_then(|a| self.moves.get(a).cloned());
        let protect = match &mv {
            Some(m) => m.entry()?.protect_counter,
            None => false,
        };
        if mv.is_some() && protect && !failed {
            self.protect_counter += 1;
        } else {
            self.protect_counter = 0;
        }
        if self.status == Some(Status::Slp) {
            self.status_counter += 1;
        }
        let glaive = dex::effect_by_name("GLAIVE_RUSH").expect("GLAIVE_RUSH");
        let charge = dex::effect_by_name("CHARGE").expect("CHARGE");
        if self.has_effect(glaive) {
            self.end_effect("Glaive Rush");
        } else if self.has_effect(charge) {
            if let Some(m) = &mv {
                if m.base_power()? > 0 && m.entry()?.typ == "ELECTRIC" && use_ {
                    self.end_effect("Charge");
                }
            }
        }
        // Flash Fire is NOT ended by its holder's Fire move: the sim's `flashfire` volatile lasts
        // until the mon leaves the field (`clearVolatile`), boosting every Fire move. Upstream
        // poke-env ended it here whatever the gen — the fork's PE-V16 fix removed that.
        Ok(())
    }

    /// `Pokemon.cant_move(reason)`: the protect streak resets; a sleeping mon counts the turn.
    pub fn cant_move(&mut self) {
        self.protect_counter = 0;
        if self.status == Some(Status::Slp) {
            self.status_counter += 1;
        }
    }

    // ---------------------------------------------------------------- switching

    /// `Pokemon.switch_in(details)`.
    pub fn switch_in(&mut self, details: Option<&str>) -> R<()> {
        self.active = true;
        if let Some(d) = details {
            if !d.is_empty() {
                self.update_from_details(d)?;
            }
        }
        self.revealed = true;
        Ok(())
    }

    /// `Pokemon.switch_out(fields)`: stages, effects, protect streak, the temporary ability /
    /// base stats / types, Transform and Mimic all dropped; a badly-poisoned count resets (the
    /// sleep count does not — gen-3 sleep persists across a pivot).
    pub fn switch_out(&mut self) {
        // The Regenerator heal (`ability == "regenerator"`) — no gen-3 mon can hold it, and
        // neutralizing gas is a field this reading never tracks; kept for exactness.
        if self.ability() == Some("regenerator") && !self.fainted() {
            let cur = self.current_hp() as f64 + self.max_hp() as f64 / 3.0;
            self.current_hp = Some((cur as u32).min(self.max_hp()));
        }
        self.active = false;
        self.clear_boosts();
        self.clear_effects();
        self.protect_counter = 0;
        self.temporary_ability = None;
        self.temporary_base_stats = None;
        self.moves.transform = None;
        self.moves.set_mimic(None);
        self.temporary_types.clear();
        if self.status == Some(Status::Tox) {
            self.status_counter = 0;
        }
    }

    /// `Pokemon.was_illusioned(fields)` — the request said this mon is NOT active while the
    /// reading had it active: HP / status forgotten, the last roster record re-applied, then a
    /// `switch_out`.
    pub fn was_illusioned(&mut self) -> R<()> {
        self.current_hp = None;
        self.max_hp = None;
        self.status = None;
        let last = self.last_request.take();
        if let Some(l) = last {
            self.update_from_request(&l)?;
        }
        self.switch_out();
        Ok(())
    }

    /// `Pokemon.baton_pass_snapshot()`: the stages and the `BATON_PASS_COPIED_EFFECTS` (rule V4).
    pub fn baton_pass_snapshot(&self) -> ([i32; 7], Vec<(EffectId, u32)>) {
        (self.boosts, self.effects.iter().filter(|(e, _)| dex::effect(*e).bp_copied).copied().collect())
    }

    /// `Pokemon.apply_baton_pass(snapshot)`.
    pub fn apply_baton_pass(&mut self, snap: ([i32; 7], Vec<(EffectId, u32)>)) {
        self.boosts = snap.0;
        for (e, c) in snap.1 {
            match self.effects.iter_mut().find(|(x, _)| *x == e) {
                Some((_, cc)) => *cc = c,
                None => self.effects.push((e, c)),
            }
        }
    }

    /// `Pokemon.transform(into)` — the overlay a Transformed mon reads (rule V14): the target's
    /// dex base stats and types, its KNOWN ability, its KNOWN moves at 5 PP each, its stages.
    pub fn transform(&mut self, into: &PMon) -> R<()> {
        let row = dex::species(&into.species)?;
        self.temporary_base_stats = Some(row.base_stats);
        if let Some(a) = into.ability() {
            self.temporary_ability = Some(a.to_string());
        }
        self.temporary_types = row.types.iter().map(|t| dex::type_from_name(t)).collect::<R<Vec<_>>>()?;
        let mut base = Vec::new();
        for (_, m) in into.moves.moves() {
            let pm = PMove::new(&m.id, None, true)?;
            match base.iter_mut().find(|(k, _): &&mut (String, PMove)| *k == m.id) {
                Some(slot) => slot.1 = pm,
                None => base.push((m.id.clone(), pm)),
            }
        }
        self.moves.transform = Some(Box::new(MoveSet { base, mimic: None, transform: None }));
        self.boosts = into.boosts;
        Ok(())
    }
}

/// The `details` argument `start_effect` takes: a typechange's type string, or Skill Swap's
/// revealed ability pair.
pub enum StartDetails {
    Types(String),
    Abilities(Vec<String>),
}
