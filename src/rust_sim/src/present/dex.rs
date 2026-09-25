//! poke-env's name normalisations and table lookups, over the GENERATED [`super::tables`].
//!
//! Every function here mirrors ONE poke-env function, named at its site. They are the reading's
//! vocabulary: a volatile's id, a side condition's key, a move's dict key, a type's name — each
//! is poke-env's own spelling, because `LiveView` is keyed by poke-env's spelling.

use super::tables::{
    EffectRow, MoveRow, SpeciesRow, EFFECTS, FIELD_NAMES, MOVES, SIDE_CONDITIONS, SPECIAL_MOVES, SPECIES, TYPE_NAMES,
};
use crate::core_error::{refuse, CoreResult, PyExc};

/// `poke_env.data.normalize.to_id_str`: `"".join(c for c in s if c.isalnum()).lower()`.
pub fn to_id(s: &str) -> String {
    crate::core_events::to_id(s)
}

/// FNV-1a — the hasher of the id INDEXES below: a lookup is on every move and mon of every
/// reading, view and encode, and the std SipHash (or the binary search over the sorted table the
/// index replaces) costs several times more on a ~10-byte id. Deterministic, std-only.
#[derive(Clone, Copy)]
pub struct IdHasher(u64);
impl Default for IdHasher {
    fn default() -> IdHasher {
        IdHasher(0xcbf2_9ce4_8422_2325)
    }
}
impl std::hash::Hasher for IdHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn finish(&self) -> u64 {
        self.0
    }
}
/// A map keyed by a table's `id` column.
pub type IdIndex = std::collections::HashMap<&'static str, usize, std::hash::BuildHasherDefault<IdHasher>>;

/// The row index of a table whose `id`s are UNIQUE (the generated tables are sorted strictly by
/// `id` — `the_tables_are_sorted_so_binary_search_is_sound` — so the index answers exactly what the
/// binary search over the sorted table answered).
fn index_of<'a>(ids: impl Iterator<Item = &'static str>) -> IdIndex {
    let mut m = IdIndex::default();
    for (i, id) in ids.enumerate() {
        m.entry(id).or_insert(i);
    }
    m
}

/// `GenData.pokedex[id]` (a `KeyError` when absent).
pub fn species(id: &str) -> CoreResult<&'static SpeciesRow> {
    static IX: std::sync::OnceLock<IdIndex> = std::sync::OnceLock::new();
    IX.get_or_init(|| index_of(SPECIES.iter().map(|r| r.id)))
        .get(id)
        .map(|&i| &SPECIES[i])
        .ok_or_else(|| refuse(PyExc::KeyError, format!("pokedex[{id:?}]: KeyError (not in poke-env's gen-3 pokedex)")))
}

/// `GenData.moves.get(id)`.
pub fn move_row(id: &str) -> Option<&'static MoveRow> {
    static IX: std::sync::OnceLock<IdIndex> = std::sync::OnceLock::new();
    IX.get_or_init(|| index_of(MOVES.iter().map(|r| r.id))).get(id).map(|&i| &MOVES[i])
}

/// `move.SPECIAL_MOVES`.
pub fn is_special_move(id: &str) -> bool {
    SPECIAL_MOVES.contains(&id)
}

/// `Move.retrieve_id`: the moves-dict KEY a move name is stored under — `to_id_str`, then
/// `return*` / `frustration*` / `hiddenpower*` collapsed to their bare id.
pub fn retrieve_id(name: &str) -> String {
    let id = to_id(name);
    for bare in ["return", "frustration", "hiddenpower"] {
        if id.starts_with(bare) {
            return bare.to_string();
        }
    }
    id
}

/// [`retrieve_id`] without a copy when `name` is already an id (lower-case ASCII letters and
/// digits — every request moveset entry): `to_id` of an id is the id, so only the collapse applies.
pub fn retrieve_id_of(name: &str) -> std::borrow::Cow<'_, str> {
    if !name.is_empty() && name.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit()) {
        for bare in ["return", "frustration", "hiddenpower"] {
            if name.starts_with(bare) {
                return std::borrow::Cow::Borrowed(bare);
            }
        }
        return std::borrow::Cow::Borrowed(name);
    }
    std::borrow::Cow::Owned(retrieve_id(name))
}

/// `Move.should_be_stored(id, 3)`: not a special move, present in the move table, not a Z or Max
/// move (the table's `storable` bit carries the last two).
pub fn should_be_stored(id: &str) -> bool {
    !is_special_move(id) && move_row(id).is_some_and(|r| r.storable)
}

/// An `Effect` member, by index into [`EFFECTS`].
pub type EffectId = u16;

/// `Effect.from_showdown_message`: strip every `item: ` / `move: ` / `ability: `, spaces and
/// dashes to underscores, upper-case; `FALLENUNDEFINED` → `FALLEN`; an unknown name is
/// `Effect.UNKNOWN` (poke-env logs a warning and carries on).
pub fn effect_from_message(message: &str) -> EffectId {
    let m = message.replace("item: ", "").replace("move: ", "").replace("ability: ", "");
    let mut m = m.replace(' ', "_").replace('-', "_").to_uppercase();
    if m == "FALLENUNDEFINED" {
        m = "FALLEN".to_string();
    }
    effect_by_name(&m).unwrap_or_else(|| effect_by_name("UNKNOWN").expect("Effect.UNKNOWN exists"))
}

/// `Effect[name]`.
pub fn effect_by_name(name: &str) -> Option<EffectId> {
    EFFECTS.binary_search_by(|r| r.name.cmp(name)).ok().map(|i| i as EffectId)
}

pub fn effect(id: EffectId) -> &'static EffectRow {
    &EFFECTS[id as usize]
}

/// `live_view._id(effect)`: the member NAME lower-cased with the underscores removed
/// (`LEECH_SEED` → `leechseed`) — the key `LivePokemon.volatiles` holds.
pub fn effect_live_id(id: EffectId) -> String {
    effect(id).name.to_ascii_lowercase().replace('_', "")
}

/// `SideCondition.from_showdown_message`: strip `move: `, spaces and dashes to underscores,
/// upper-case; unknown → `UNKNOWN`. Returns the member NAME and its `STACKABLE_CONDITIONS` cap.
pub fn side_condition(message: &str) -> (&'static str, u8) {
    let m = message.replace("move: ", "").replace(' ', "_").replace('-', "_").to_uppercase();
    match SIDE_CONDITIONS.binary_search_by(|(n, _)| n.cmp(&m.as_str())) {
        Ok(i) => SIDE_CONDITIONS[i],
        Err(_) => {
            let i = SIDE_CONDITIONS.binary_search_by(|(n, _)| n.cmp(&"UNKNOWN")).expect("UNKNOWN");
            SIDE_CONDITIONS[i]
        }
    }
}

/// `Field.from_showdown_message`: strip `ability: ` / `move: `, spaces to underscores, a bare
/// `…terrain` suffix split to `…_terrain`, upper-case; unknown → `UNKNOWN`. The member NAME.
pub fn field(message: &str) -> &'static str {
    let mut m = message.replace("ability: ", "").replace("move: ", "").replace(' ', "_");
    if m.ends_with("terrain") && !m.ends_with("_terrain") {
        m = m.replace("terrain", "_terrain");
    }
    let up = m.to_uppercase();
    FIELD_NAMES.iter().copied().find(|f| *f == up).unwrap_or("UNKNOWN")
}

/// `PokemonType.from_name`: `???` → `THREE_QUESTION_MARKS`, else `PokemonType[name.upper()]`
/// (a `KeyError` for a name poke-env has no member for). Returns the member NAME.
pub fn type_from_name(name: &str) -> CoreResult<&'static str> {
    let up = if name == "???" { "THREE_QUESTION_MARKS".to_string() } else { name.to_uppercase() };
    TYPE_NAMES
        .iter()
        .copied()
        .find(|t| *t == up)
        .ok_or_else(|| refuse(PyExc::KeyError, format!("PokemonType[{up:?}]: KeyError")))
}

/// `Status[tok.upper()]` — the member NAME, or a `KeyError`.
pub fn status_from(tok: &str) -> CoreResult<Status> {
    Ok(match tok.to_uppercase().as_str() {
        "BRN" => Status::Brn,
        "FNT" => Status::Fnt,
        "FRZ" => Status::Frz,
        "PAR" => Status::Par,
        "PSN" => Status::Psn,
        "SLP" => Status::Slp,
        "TOX" => Status::Tox,
        other => return Err(refuse(PyExc::KeyError, format!("Status[{other:?}]: KeyError"))),
    })
}

/// poke-env's `Status` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Brn,
    Fnt,
    Frz,
    Par,
    Psn,
    Slp,
    Tox,
}

impl Status {
    /// `live_view._enum_name(status)`.
    pub fn live(self) -> &'static str {
        match self {
            Status::Brn => "brn",
            Status::Fnt => "fnt",
            Status::Frz => "frz",
            Status::Par => "par",
            Status::Psn => "psn",
            Status::Slp => "slp",
            Status::Tox => "tox",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tables_are_sorted_so_binary_search_is_sound() {
        assert!(SPECIES.windows(2).all(|w| w[0].id < w[1].id));
        assert!(MOVES.windows(2).all(|w| w[0].id < w[1].id));
        assert!(EFFECTS.windows(2).all(|w| w[0].name < w[1].name));
        assert!(SIDE_CONDITIONS.windows(2).all(|w| w[0].0 < w[1].0));
    }

    #[test]
    fn the_id_indexes_answer_what_the_binary_search_answered() {
        for (i, r) in MOVES.iter().enumerate() {
            assert!(std::ptr::eq(move_row(r.id).unwrap(), &MOVES[i]));
            assert_eq!(MOVES.binary_search_by(|x| x.id.cmp(r.id)), Ok(i));
        }
        for (i, r) in SPECIES.iter().enumerate() {
            assert!(std::ptr::eq(species(r.id).unwrap(), &SPECIES[i]));
        }
        for miss in ["", "zzz", "Earthquake", "earthquak", "earthquakee", "hiddenpowerfire70"] {
            assert_eq!(move_row(miss).is_some(), MOVES.binary_search_by(|x| x.id.cmp(miss)).is_ok(), "{miss:?}");
            assert_eq!(species(miss).is_ok(), SPECIES.binary_search_by(|x| x.id.cmp(miss)).is_ok(), "{miss:?}");
        }
    }

    #[test]
    fn to_id_ascii_path_is_the_unicode_fold() {
        let reference = |s: &str| -> String { s.chars().filter(|c| c.is_alphanumeric()).flat_map(|c| c.to_lowercase()).collect() };
        for s in ["", "Mr. Mime", "p2a: Ho-Oh", "Farfetch’d", "Flabébé", "NIDORAN♀", "Hidden Power [Fire]", "\t\n 1-2_3 ~Zz", "İstanbul", "ǅ"] {
            assert_eq!(to_id(s), reference(s), "{s:?}");
        }
        for b in 0u8..128 {
            let s = (b as char).to_string();
            assert_eq!(to_id(&s), reference(&s), "{b}");
        }
    }

    #[test]
    fn effect_names_normalise_like_poke_env() {
        assert_eq!(effect_live_id(effect_from_message("move: Leech Seed")), "leechseed");
        assert_eq!(effect_live_id(effect_from_message("ability: Flash Fire")), "flashfire");
        assert_eq!(effect_live_id(effect_from_message("perish3")), "perish3");
        assert_eq!(effect_live_id(effect_from_message("no such effect")), "unknown");
    }

    #[test]
    fn move_keys_collapse_like_retrieve_id() {
        for n in ["", "protect", "hiddenpowergrass", "hiddenpower", "return102", "return", "frustration", "Hidden Power", "Protect", "self-destruct", "x1", "ho oh", "ƒoo"] {
            assert_eq!(retrieve_id_of(n), retrieve_id(n), "{n:?}");
        }
        assert_eq!(retrieve_id("Hidden Power"), "hiddenpower");
        assert_eq!(retrieve_id("hiddenpowerfire70"), "hiddenpower");
        assert_eq!(retrieve_id("return102"), "return");
        assert!(!should_be_stored("struggle"));
        assert!(should_be_stored("earthquake"));
    }

    #[test]
    fn side_conditions_key_like_poke_env() {
        assert_eq!(side_condition("move: Light Screen"), ("LIGHT_SCREEN", 0));
        assert_eq!(side_condition("Spikes"), ("SPIKES", 3));
        assert_eq!(side_condition("Mud Sport").0, "UNKNOWN");
    }
}
