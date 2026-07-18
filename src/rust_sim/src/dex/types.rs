//! Core game-data value types shared across the dex (and, later, the engine):
//! Pokémon [`Type`], [`MoveCategory`], and [`BaseStats`].

/// A Pokémon type. The 18 standard types; Gen 3 doesn't use Fairy, but keeping
/// it here costs nothing and keeps the enum gen-generic (the chart carries the
/// all-`1` Fairy rows). Typeless moves (`"???"`, e.g. Curse) are represented as
/// `Option::None` at the use site, not a variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Type {
    Normal,
    Fire,
    Water,
    Electric,
    Grass,
    Ice,
    Fighting,
    Poison,
    Ground,
    Flying,
    Psychic,
    Bug,
    Rock,
    Ghost,
    Dragon,
    Dark,
    Steel,
    Fairy,
}

impl Type {
    /// Parse a type name (case-insensitive). Returns `None` for `"???"`/typeless
    /// or any unknown string, so callers can model typelessness as `Option<Type>`.
    pub fn from_name(name: &str) -> Option<Type> {
        Some(match name.to_ascii_uppercase().as_str() {
            "NORMAL" => Type::Normal,
            "FIRE" => Type::Fire,
            "WATER" => Type::Water,
            "ELECTRIC" => Type::Electric,
            "GRASS" => Type::Grass,
            "ICE" => Type::Ice,
            "FIGHTING" => Type::Fighting,
            "POISON" => Type::Poison,
            "GROUND" => Type::Ground,
            "FLYING" => Type::Flying,
            "PSYCHIC" => Type::Psychic,
            "BUG" => Type::Bug,
            "ROCK" => Type::Rock,
            "GHOST" => Type::Ghost,
            "DRAGON" => Type::Dragon,
            "DARK" => Type::Dark,
            "STEEL" => Type::Steel,
            "FAIRY" => Type::Fairy,
            _ => return None,
        })
    }

    /// The canonical UPPERCASE name (the key form used by the type chart and by
    /// `agents.gen3_data`'s `PokemonType.name`).
    pub fn name(self) -> &'static str {
        match self {
            Type::Normal => "NORMAL",
            Type::Fire => "FIRE",
            Type::Water => "WATER",
            Type::Electric => "ELECTRIC",
            Type::Grass => "GRASS",
            Type::Ice => "ICE",
            Type::Fighting => "FIGHTING",
            Type::Poison => "POISON",
            Type::Ground => "GROUND",
            Type::Flying => "FLYING",
            Type::Psychic => "PSYCHIC",
            Type::Bug => "BUG",
            Type::Rock => "ROCK",
            Type::Ghost => "GHOST",
            Type::Dragon => "DRAGON",
            Type::Dark => "DARK",
            Type::Steel => "STEEL",
            Type::Fairy => "FAIRY",
        }
    }

    /// The DISPLAY-CASED (Title-Case) type name, as Showdown renders it in the protocol
    /// stream (e.g. a Color Change `|-start|<mon>|typechange|Psychic|…`). Distinct from
    /// [`Self::name`], which is the UPPERCASE key form the type chart / `agents.gen3_data`
    /// use. Used ONLY for protocol emission.
    pub fn display_name(self) -> &'static str {
        match self {
            Type::Normal => "Normal",
            Type::Fire => "Fire",
            Type::Water => "Water",
            Type::Electric => "Electric",
            Type::Grass => "Grass",
            Type::Ice => "Ice",
            Type::Fighting => "Fighting",
            Type::Poison => "Poison",
            Type::Ground => "Ground",
            Type::Flying => "Flying",
            Type::Psychic => "Psychic",
            Type::Bug => "Bug",
            Type::Rock => "Rock",
            Type::Ghost => "Ghost",
            Type::Dragon => "Dragon",
            Type::Dark => "Dark",
            Type::Steel => "Steel",
            Type::Fairy => "Fairy",
        }
    }

    /// Gen 1-3 physical/special split is by TYPE (per-move categories arrived in
    /// Gen 4). These eight types are special; everything else is physical.
    /// Mirrors poke-env's `Move.SPECIAL_TYPES` for gen ≤ 3.
    pub fn is_special_gen3(self) -> bool {
        matches!(
            self,
            Type::Fire
                | Type::Water
                | Type::Grass
                | Type::Electric
                | Type::Ice
                | Type::Psychic
                | Type::Dark
                | Type::Dragon
        )
    }
}

/// Damage category. In Gen ≤ 3 it is derived from base power + type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveCategory {
    Status,
    Physical,
    Special,
}

impl MoveCategory {
    /// UPPERCASE name, matching `agents.gen3_data`'s `MoveCategory.name`.
    pub fn name(self) -> &'static str {
        match self {
            MoveCategory::Status => "STATUS",
            MoveCategory::Physical => "PHYSICAL",
            MoveCategory::Special => "SPECIAL",
        }
    }
}

/// The six base stats of a species.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BaseStats {
    pub hp: u16,
    pub atk: u16,
    pub def: u16,
    pub spa: u16,
    pub spd: u16,
    pub spe: u16,
}
