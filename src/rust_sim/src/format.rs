//! The FORMAT table (`gen3_format_rules_v1`): what a format id changes in a battle.
//!
//! Showdown's formats differ in ways a battle can see, and the engine used to read one bit off
//! the id (`!id.ends_with("customgame")`), running every other format as gen3ou. The rule
//! tables (`config/formats.ts`, resolved by `Dex.formats.getRuleTable`) say otherwise:
//!
//! * **Freeze Clause Mod** is in `[Gen 3] OU` (and so `[Gen 3] UUBL`, whose ruleset is OU's) and
//!   `[Gen 3] Draft`, NOT in the gen-3 `Standard` ruleset (`data/mods/gen3/rulesets.ts`: Sleep
//!   Clause Mod, Switch Priority, Species, Nickname, OHKO, Evasion Items, Evasion Moves). So a
//!   status application in Ubers / UU / RU / NU / PU / ZU / LC / Random Battle gathers ONE clause
//!   `onSetStatus` handler, which sorts alone, and draws NO speed-sort shuffle; OU's two handlers
//!   tie and draw one `random` (measured: a Thunder Wave turn is 4 PRNG calls in OU, 3 in UU).
//! * the `|tier|` and `|rule|` framing lines;
//! * Species Clause (Draft has none), Beat Up Nicknames Mod, the unheld-trapping-ability leak
//!   and the abilities a banlist removes from it (UU / RU ban Arena Trap), exact HP (`debug`).
//!
//! The table is checked against Showdown by `tests/format_rules_test.rs`, which reads the vector
//! `harness/dump_gen3_formats.js` writes (`--check` is its drift gate). An id that is not in the
//! table is REFUSED at construction: a format the engine does not model never runs as another.
//! Ids are compared after Showdown's `toID` (`[Gen 3] OU` and `gen3ou` are one format).

/// One format's battle-visible rules.
#[derive(Debug, PartialEq, Eq)]
pub struct FormatRules {
    /// The `toID` form (`gen3ou`).
    pub id: &'static str,
    /// The `|tier|` text.
    pub tier: &'static str,
    /// The `|rule|` texts, in Showdown's order.
    pub rules: &'static [&'static str],
    /// Sleep Clause Mod: a second foe-inflicted sleep on a side fails.
    pub sleep_clause: bool,
    /// Freeze Clause Mod: a second freeze on a side fails.
    pub freeze_clause: bool,
    /// Species Clause: a team holding two mons of one dex number is refused.
    pub species_clause: bool,
    /// Beat Up Nicknames Mod: Beat Up emits no `-activate|…|[of] <ally>` per strike.
    pub beatup_nicknames: bool,
    /// The endTurn "canceling switches would leak information" loop runs: a foe whose SPECIES
    /// could hold a trapping ability raises `maybeTrapped` (`battle.ts`, skipped for
    /// `(+hackmons || !obtainableabilities) && !format.team`).
    pub ability_leak: bool,
    /// Trapping abilities the banlist removes from that loop (`ruleTable.has('-ability:<id>')`).
    pub leak_banned_abilities: &'static [&'static str],
    /// `debug: true` (Custom Game): every stream shows exact HP.
    pub exact_hp: bool,
}

impl FormatRules {
    /// The `onSetStatus` clause handlers tie, so `runEvent('SetStatus')` draws the size-2
    /// speed-sort shuffle. True exactly when BOTH clause handlers are present.
    pub fn set_status_shuffle(&self) -> bool {
        self.sleep_clause && self.freeze_clause
    }

    /// The pseudo-weathers the field carries: the rules with a non-`onBegin` event handler, which
    /// in gen 3 are exactly the two clause mods (sorted, the form `search::pre_state` reports).
    pub fn pseudo_weather(&self) -> &'static [&'static str] {
        match (self.freeze_clause, self.sleep_clause) {
            (true, true) => &["freezeclausemod", "sleepclausemod"],
            (true, false) => &["freezeclausemod"],
            (false, true) => &["sleepclausemod"],
            (false, false) => &[],
        }
    }
}

const HP_PERCENT: &str = "HP Percentage Mod: HP is shown in percentages";
const BEAT_UP_NICKNAMES: &str = "Beat Up Nicknames Mod: Beat Up will not reveal any party members";
const ENDLESS: &str = "Endless Battle Clause: Forcing endless battles is banned";
const SLEEP: &str = "Sleep Clause Mod: Limit one foe put to sleep";
const SWITCH_PRIORITY: &str = "Switch Priority Clause Mod: Faster Pokémon switch first";
const SPECIES: &str = "Species Clause: Limit one of each Pokémon";
const OHKO: &str = "OHKO Clause: OHKO moves are banned";
const EVASION_ITEMS: &str = "Evasion Items Clause: Evasion items are banned";
const EVASION_MOVES: &str = "Evasion Moves Clause: Evasion moves are banned";
const ONE_BOOST_PASSER: &str = "One Boost Passer Clause: Limit one Baton Passer that has a way to boost its stats";
const FREEZE: &str = "Freeze Clause Mod: Limit one foe frozen";
const SLEEP_MOVES: &str = "Sleep Moves Clause: Sleep-inducing moves are banned";
const ACCURACY_MOVES: &str = "Accuracy Moves Clause: Accuracy-lowering moves are banned";

/// A `Standard` tier: these nine rule lines, then the tier's own.
macro_rules! standard {
    ($($extra:expr),* $(,)?) => {
        &[HP_PERCENT, BEAT_UP_NICKNAMES, ENDLESS, SLEEP, SWITCH_PRIORITY, SPECIES, OHKO,
          EVASION_ITEMS, EVASION_MOVES, $($extra),*]
    };
}

/// A clause tier with no Freeze Clause (the gen-3 `Standard` ruleset alone).
const fn standard_tier(id: &'static str, tier: &'static str, rules: &'static [&'static str]) -> FormatRules {
    FormatRules {
        id,
        tier,
        rules,
        sleep_clause: true,
        freeze_clause: false,
        species_clause: true,
        beatup_nicknames: true,
        ability_leak: true,
        leak_banned_abilities: &[],
        exact_hp: false,
    }
}

/// Every gen-3 singles format the engine runs. `[Gen 3] 1v1` is refused (Team Preview), as are
/// the doubles formats and the `gen3rs` / `gen3frlg` / `gen3colosseum` mods.
pub static FORMATS: &[FormatRules] = &[
    FormatRules {
        id: "gen3customgame",
        tier: "[Gen 3] Custom Game",
        rules: &[HP_PERCENT],
        sleep_clause: false,
        freeze_clause: false,
        species_clause: false,
        beatup_nicknames: false,
        ability_leak: false,
        leak_banned_abilities: &[],
        exact_hp: true,
    },
    FormatRules {
        id: "gen3draft",
        tier: "[Gen 3] Draft",
        rules: &[
            BEAT_UP_NICKNAMES,
            SLEEP,
            OHKO,
            "Evasion Clause: Evasion abilities, items, and moves are banned",
            "Evasion Abilities Clause: Evasion abilities are banned",
            EVASION_ITEMS,
            EVASION_MOVES,
            ENDLESS,
            HP_PERCENT,
            ONE_BOOST_PASSER,
            FREEZE,
            ACCURACY_MOVES,
        ],
        sleep_clause: true,
        freeze_clause: true,
        species_clause: false,
        beatup_nicknames: true,
        ability_leak: true,
        leak_banned_abilities: &[],
        exact_hp: false,
    },
    standard_tier(
        "gen3lc",
        "[Gen 3] LC",
        standard!(
            "Pomeg Glitch Clause: Gen 3 Pokémon at level 5+ can have any of their level-up moves.",
            SLEEP_MOVES,
            ACCURACY_MOVES,
        ),
    ),
    standard_tier("gen3nu", "[Gen 3] NU", standard!(ONE_BOOST_PASSER)),
    FormatRules {
        freeze_clause: true,
        ..standard_tier(
            "gen3ou",
            "[Gen 3] OU",
            standard!(ONE_BOOST_PASSER, FREEZE, "Baton Pass Stat Clause: No Baton Passer may have a way to boost its Speed"),
        )
    },
    standard_tier(
        "gen3pu",
        "[Gen 3] PU",
        standard!("Baton Pass Stat Clause: No Baton Passer may have a way to boost its stats"),
    ),
    standard_tier("gen3randombattle", "[Gen 3] Random Battle", standard!()),
    FormatRules { leak_banned_abilities: &["arenatrap"], ..standard_tier("gen3ru", "[Gen 3] RU", standard!()) },
    standard_tier(
        "gen3ubers",
        "[Gen 3] Ubers",
        standard!(
            "Deoxys Camouflage Clause Mod: Reveals the Deoxys forme when it is sent in battle.",
            "One Baton Pass Clause: Limit one Baton Passer",
        ),
    ),
    FormatRules { leak_banned_abilities: &["arenatrap"], ..standard_tier("gen3uu", "[Gen 3] UU", standard!()) },
    FormatRules {
        freeze_clause: true,
        ..standard_tier(
            "gen3uubl",
            "[Gen 3] UUBL",
            standard!(ONE_BOOST_PASSER, FREEZE, "Baton Pass Stat Clause: No Baton Passer may have a way to boost its Speed"),
        )
    },
    standard_tier(
        "gen3zu",
        "[Gen 3] ZU",
        standard!(
            SLEEP_MOVES,
            "Baton Pass Stat Trap Clause: No Baton Passer may have a way to boost stats or trap Pokémon",
        ),
    ),
];

/// Formats Showdown defines for the gen-3 singles engine that the engine refuses, and why.
pub static REFUSED: &[(&str, &str)] =
    &[("gen31v1", "Team Preview + Picked Team Size = 1: the engine has no team-preview phase")];

/// The rules of `format_id` (any spelling Showdown's `toID` folds to a table id), or the reason
/// the engine refuses it.
pub fn rules(format_id: &str) -> Result<&'static FormatRules, String> {
    let id = crate::dex::to_id(format_id);
    if let Some(r) = FORMATS.iter().find(|r| r.id == id) {
        return Ok(r);
    }
    if let Some((_, why)) = REFUSED.iter().find(|(r, _)| *r == id) {
        return Err(format!("format {format_id:?} is not supported: {why}"));
    }
    Err(format!(
        "format {format_id:?} is not supported: the engine models the gen-3 singles formats {:?} \
         (gen3_format_rules_v1); a doubles format, another mod or another generation is refused \
         rather than run as one of them",
        FORMATS.iter().map(|r| r.id).collect::<Vec<_>>()
    ))
}

