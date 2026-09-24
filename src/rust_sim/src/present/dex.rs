//! poke-env's name normalisations and table lookups, over the GENERATED [`super::tables`].
//!
//! Every function here mirrors ONE poke-env function, named at its site. They are the reading's
//! vocabulary: a volatile's id, a side condition's key, a move's dict key, a type's name — each
//! is poke-env's own spelling, because `LiveView` is keyed by poke-env's spelling.

use super::tables::{
    EffectRow, MoveRow, SpeciesRow, EFFECTS, FIELD_NAMES, MOVES, SIDE_CONDITIONS, SPECIAL_MOVES, SPECIES, TYPE_NAMES,
};

/// `poke_env.data.normalize.to_id_str`: `"".join(c for c in s if c.isalnum()).lower()`.
pub fn to_id(s: &str) -> String {
    crate::core_events::to_id(s)
}

/// `GenData.pokedex[id]` (a `KeyError` when absent).
pub fn species(id: &str) -> Result<&'static SpeciesRow, String> {
    SPECIES
        .binary_search_by(|r| r.id.cmp(id))
        .map(|i| &SPECIES[i])
        .map_err(|_| format!("pokedex[{id:?}]: KeyError (not in poke-env's gen-3 pokedex)"))
}

/// `GenData.moves.get(id)`.
pub fn move_row(id: &str) -> Option<&'static MoveRow> {
    MOVES.binary_search_by(|r| r.id.cmp(id)).ok().map(|i| &MOVES[i])
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
pub fn type_from_name(name: &str) -> Result<&'static str, String> {
    let up = if name == "???" { "THREE_QUESTION_MARKS".to_string() } else { name.to_uppercase() };
    TYPE_NAMES
        .iter()
        .copied()
        .find(|t| *t == up)
        .ok_or_else(|| format!("PokemonType[{up:?}]: KeyError"))
}

/// `Status[tok.upper()]` — the member NAME, or a `KeyError`.
pub fn status_from(tok: &str) -> Result<Status, String> {
    Ok(match tok.to_uppercase().as_str() {
        "BRN" => Status::Brn,
        "FNT" => Status::Fnt,
        "FRZ" => Status::Frz,
        "PAR" => Status::Par,
        "PSN" => Status::Psn,
        "SLP" => Status::Slp,
        "TOX" => Status::Tox,
        other => return Err(format!("Status[{other:?}]: KeyError")),
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
    fn effect_names_normalise_like_poke_env() {
        assert_eq!(effect_live_id(effect_from_message("move: Leech Seed")), "leechseed");
        assert_eq!(effect_live_id(effect_from_message("ability: Flash Fire")), "flashfire");
        assert_eq!(effect_live_id(effect_from_message("perish3")), "perish3");
        assert_eq!(effect_live_id(effect_from_message("no such effect")), "unknown");
    }

    #[test]
    fn move_keys_collapse_like_retrieve_id() {
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
