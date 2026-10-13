//! THE ORACLE REVEAL (`--oracle-reveal <spec>`; `designs/endstate/design_information_levers.md` §6, the training
//! observation's consumer of the ONE information-lever spec `agents.info_levers.spec`; its X32 origin is
//! `designs/endstate/design_x5_belief_tokens.md` §7.6) — a DIAGNOSTIC observation mode, never production. The
//! observation's opponent team block is TOLD graded facts of the opponent's TRUE team: per (class, part) a LEVER of
//! mass p ([`OracleSpec`]), for the classes `species` (a never-seen mon's identity: its tail slot exists), `moves`,
//! `item`, `ability` and `spread` (nature, EVs, IVs; `spread_known` 1), each on the SEEN mons and / or the UNSEEN ones.
//! The facts enter the SHARED TRUNK through the observation, not through a side input to any head. The X32 levels are
//! two of its specs: `species` = `species:unseen` (team-preview semantics), `full` =
//! `species:unseen,moves:both,item:both,ability:both,spread:both` (the whole set as already-known facts).
//!
//! **One mechanism, no second encoder.** The reveal is decided in the ENCODER's opponent block and
//! nowhere else (the reading, the view, the trackers, the legality and every other block are the
//! chain's own, untouched):
//!
//! * the opponent slots the viewer has SEEN keep their reveal-order positions. A seen mon none of whose `seen` levers
//!   fires is written exactly as the mode `off` writes it; otherwise it gains only the facts of its FIRED classes play
//!   has NOT revealed yet ([`Oracle::overlay`]) — an observed move keeps its tracked PP, a consumed or removed item
//!   stays consumed or removed, a revealed ability is the reading's: no stale preview value fights a live one;
//! * after them, one slot per opponent mon NOT yet seen whose `species:unseen` lever FIRES, in dex-num order (the
//!   order the belief labels already give an unseen mon, `belief_labels.assign_hidden_to_slots`). With no set class
//!   fired for it, it is written by [`super::hypothesis::hypothesis_slot`] — the SAME slot writer a real first
//!   appearance goes through ("species s present, unrevealed set, full HP, no status"), fed a mon that has never been
//!   on the field. With set classes fired it is the row the encoder writes for an OWN mon at full HP that holds
//!   exactly the told facts of its set ([`told_slot`]: the same `slot::populated_slot`, the spread block written only
//!   when the spread is told, because the mon's view then says `spread_known`);
//! * a mon that play then reveals LEAVES the tail and joins the seen prefix (matched by dex num — a
//!   forme shares its base species' num — one oracle entry per revealed mon, so a duplicate species
//!   is not double-counted), where its `seen` levers apply.
//!
//! **Mass.** The observation can only state a fact or not: a lever of mass p TELLS its fact for a whole episode with
//! probability p, decided per (episode seed, side, the mon's packed index, class) by a deterministic hash
//! ([`uniform_bp`]): ONE draw u per (mon, class), compared with the mass of the part that applies (u < p_seen while
//! the mon is seen, u < p_unseen while it is not), so a class told at equal masses stays told when the mon is
//! revealed. A mass-1 lever needs no hash; `off` builds nothing (the chain holds no oracle).
//!
//! A tail slot is the encoder's row for a mon that has never appeared, so it carries nothing the
//! viewer could not be TOLD: an untold set class is ZERO (the item, the moves, the spread and `hp_revealed`) or the
//! species' Smogon prior (the ability block); the status, the Hidden-Power probabilities, the sleep belief, the last
//! action and the active / trapped flags are ZERO; the recency cells read "never seen" (saturated); the HP fraction is
//! 1.0 (an unseen gen-3 mon is pristine). [`slot_cells`] declares every cell's rule for each told / untold class
//! combination and [`check_slot`] THROWS when a slot breaks one — it runs on every tail slot at construction, so a
//! regression in the shared slot writer that let a hidden fact into a tail slot is a FAULT at the first episode,
//! never a quiet leak.
//!
//! Scripted bots read the view, not the row: they are unaffected. The reveal is per-CHAIN (each
//! side's chain gets the OTHER side's team), so the training pool's two sides — the trainee and a
//! policy opponent served by T2 — are symmetric by construction.

use super::layout::*;
use super::{data, hypothesis, slot};
use crate::core_error::{fault, CoreResult};
use crate::dex::Dex;
use crate::present::dex::to_id;
use crate::present::mon::{PMon, TbMon};
use crate::present::tables::UNKNOWN_ITEM;
use crate::present::view::{mon_view, MonView};
use crate::team::PokemonSet;
use crate::trackers::clock::ClockConfig;
use crate::trackers::SideTrackers;

// ------------------------------------------------------------------------------------------------ the spec

/// A fact class the observation can be told (`agents.info_levers.spec.CLASSES`, the ones the OBS consumer honours;
/// `hp` and `counters` have no cell and are refused by name).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Species = 0,
    Moves = 1,
    Item = 2,
    Ability = 3,
    Spread = 4,
}

impl Class {
    pub const ALL: [Class; 5] = [Class::Species, Class::Moves, Class::Item, Class::Ability, Class::Spread];
    /// The SET classes (a fact of the packed set).
    pub const SET: [Class; 4] = [Class::Moves, Class::Item, Class::Ability, Class::Spread];

    pub fn as_str(self) -> &'static str {
        match self {
            Class::Species => "species",
            Class::Moves => "moves",
            Class::Item => "item",
            Class::Ability => "ability",
            Class::Spread => "spread",
        }
    }

    fn parse(s: &str) -> Option<Class> {
        Class::ALL.into_iter().find(|c| c.as_str() == s)
    }
}

/// The part of the opponent team a lever applies to: the mons the viewer has SEEN, or the ones it has not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part {
    Seen = 0,
    Unseen = 1,
}

/// A mass is a whole number of 1/10,000ths (`spec.MASS_DENOM`): the exact canonical text, the same integer as Python.
pub const MASS_DENOM: u16 = 10_000;

/// What the observation is told: per (class, part) a mass in 1/10,000ths (0 = not told), source `truth` only. Parsed
/// from EXACTLY the X32 level names (`off` / `species` / `full`) or the CANONICAL clause text
/// `agents.info_levers.spec.InfoReveal.canonical` produces ([`OracleSpec::parse`]); rendered back as its LABEL (the
/// level name when it is that level, else the canonical text: [`OracleSpec::label`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OracleSpec {
    mass: [[u16; 2]; 5],
}

/// The OBS consumer's reason for refusing `hp` / `counters` (`agents.info_levers.consumers.OBS.why`).
fn refused_class(cls: &str) -> Option<&'static str> {
    match cls {
        "hp" => Some("the opponent's HP cell is the DISPLAYED fraction; an exact-HP cell would be a layout change"),
        "counters" => Some("the observation has no cell for a hidden duration (the elapsed sleep / toxic counters it has are public)"),
        _ => None,
    }
}

fn mass_text(bp: u16) -> String {
    if bp == MASS_DENOM {
        return String::new();
    }
    format!("@0.{}", format!("{bp:04}").trim_end_matches('0'))
}

impl OracleSpec {
    /// Nothing: the observation is byte-identical to the build that had no reveal.
    pub const OFF: OracleSpec = OracleSpec { mass: [[0; 2]; 5] };
    /// X32 `species` = `species:unseen`: the six species (team-preview semantics), nothing else.
    pub const SPECIES: OracleSpec = OracleSpec { mass: [[0, MASS_DENOM], [0; 2], [0; 2], [0; 2], [0; 2]] };
    /// X32 `full` = `species:unseen,moves:both,item:both,ability:both,spread:both`: the whole set as already-known facts.
    pub const FULL: OracleSpec = OracleSpec { mass: [[0, MASS_DENOM], [MASS_DENOM; 2], [MASS_DENOM; 2], [MASS_DENOM; 2], [MASS_DENOM; 2]] };
    /// The X32 level names, in the order the flag listed them.
    pub const LEVELS: [(&'static str, OracleSpec); 3] = [("off", OracleSpec::OFF), ("species", OracleSpec::SPECIES), ("full", OracleSpec::FULL)];

    /// The mass (1/10,000ths) of the lever on (`class`, `part`); 0 = not told.
    pub fn mass(&self, class: Class, part: Part) -> u16 {
        self.mass[class as usize][part as usize]
    }

    /// A spec with exactly these levers (each mass in 0..=10,000; `species:seen` must be 0). Not validated against the
    /// OBS consumer's `needs` — [`OracleSpec::parse`] is the validating constructor.
    pub fn with(levers: &[(Class, Part, u16)]) -> OracleSpec {
        let mut s = OracleSpec::OFF;
        for &(c, p, m) in levers {
            s.mass[c as usize][p as usize] = m.min(MASS_DENOM);
        }
        s
    }

    pub fn is_off(&self) -> bool {
        *self == OracleSpec::OFF
    }

    /// Does any `seen` lever exist (a seen mon can be overlaid)?
    pub fn tells_seen(&self) -> bool {
        Class::SET.iter().any(|&c| self.mass(c, Part::Seen) > 0)
    }

    /// Every positive mass raised to 1: the spec under which EVERY lever fires for every mon (the startup validation
    /// builds each team's slots at it, so a slot no episode draw could build is refused before the run starts).
    pub fn saturated(&self) -> OracleSpec {
        let mut s = *self;
        for row in s.mass.iter_mut() {
            for m in row.iter_mut() {
                if *m > 0 {
                    *m = MASS_DENOM;
                }
            }
        }
        s
    }

    /// The canonical text (`InfoReveal.canonical`): clauses in class order, the scope explicit (`both` when a class's
    /// seen and unseen masses are equal), a mass below 1 as `@0.` + up to four decimals; `off` when nothing is told.
    pub fn canonical(&self) -> String {
        let mut out: Vec<String> = Vec::new();
        for c in Class::ALL {
            let (s, u) = (self.mass(c, Part::Seen), self.mass(c, Part::Unseen));
            if s > 0 && s == u {
                out.push(format!("{}:both{}", c.as_str(), mass_text(s)));
                continue;
            }
            if s > 0 {
                out.push(format!("{}:seen{}", c.as_str(), mass_text(s)));
            }
            if u > 0 {
                out.push(format!("{}:unseen{}", c.as_str(), mass_text(u)));
            }
        }
        if out.is_empty() {
            "off".into()
        } else {
            out.join(",")
        }
    }

    /// The LABEL (`InfoReveal.label`): `off` / `species` / `full` when the spec IS that X32 level, else the canonical
    /// text — what `model_config.json`'s `oracle_reveal` and the env spec's JSON carry.
    pub fn label(&self) -> String {
        match OracleSpec::LEVELS.iter().find(|(_, s)| s == self) {
            Some((name, _)) => (*name).to_string(),
            None => self.canonical(),
        }
    }

    /// Parse EXACTLY a level name or a CANONICAL clause text (strict: the text must equal this parser's own canonical
    /// rendering of what it parsed). Refused BY NAME: `hp` / `counters` (no cell), a `~placebo` / `~map` source (no
    /// prior sampler on this side), an unseen set-class part without `species:unseen` (no slot holds a never-seen
    /// mon otherwise), and anything else.
    pub fn parse(text: &str) -> Result<OracleSpec, String> {
        if let Some((_, s)) = OracleSpec::LEVELS.iter().find(|(n, _)| *n == text) {
            return Ok(*s);
        }
        let mut spec = OracleSpec::OFF;
        let mut given = [[false; 2]; 5];
        for clause in text.split(',') {
            if clause.is_empty() {
                return Err(format!("oracle_reveal {text:?}: an empty clause"));
            }
            let (head, source) = match clause.split_once('~') {
                Some((h, s)) => (h, Some(s)),
                None => (clause, None),
            };
            if let Some(src) = source {
                return Err(match src {
                    "placebo" | "map" => format!(
                        "oracle_reveal {text:?}: the training observation cannot honour the source ~{src} in {clause:?} \
                         (it states the TRUE fact or nothing; it has no prior sampler for a wrong or a mode fact)"
                    ),
                    _ => format!("oracle_reveal {text:?}: {clause:?} is not canonical (the source is `truth`, written by omission)"),
                });
            }
            let (head, mass) = match head.split_once('@') {
                Some((h, m)) => (h, Some(m)),
                None => (head, None),
            };
            let (cls, scope) = head.split_once(':').unwrap_or((head, ""));
            if let Some(why) = refused_class(cls) {
                return Err(format!("oracle_reveal {text:?}: the training observation cannot honour the lever {clause:?} ({why})"));
            }
            let Some(class) = Class::parse(cls) else {
                return Err(format!(
                    "oracle_reveal {text:?}: {clause:?} names no class the observation can be told (classes: species, moves, \
                     item, ability, spread; or a level: off, species, full)"
                ));
            };
            let parts: &[Part] = match (class, scope) {
                (Class::Species, "unseen") => &[Part::Unseen],
                (Class::Species, _) => {
                    return Err(format!(
                        "oracle_reveal {text:?}: {clause:?} — the species lever's canonical scope is `unseen` (a seen mon's \
                         species is public)"
                    ))
                }
                (_, "seen") => &[Part::Seen],
                (_, "unseen") => &[Part::Unseen],
                (_, "both") => &[Part::Seen, Part::Unseen],
                _ => return Err(format!("oracle_reveal {text:?}: {clause:?} is not canonical (the scope seen / unseen / both is explicit)")),
            };
            let bp = match mass {
                None => MASS_DENOM,
                Some(m) => {
                    let digits = m.strip_prefix("0.").filter(|d| (1..=4).contains(&d.len()) && d.bytes().all(|b| b.is_ascii_digit()));
                    let Some(d) = digits else {
                        return Err(format!("oracle_reveal {text:?}: {clause:?}: the mass {m:?} is not canonical (`0.` + one to four decimals; mass 1 is written by omission)"));
                    };
                    let bp: u16 = format!("{d:0<4}").parse().map_err(|_| format!("oracle_reveal {text:?}: {clause:?}: bad mass"))?;
                    if bp == 0 {
                        return Err(format!("oracle_reveal {text:?}: {clause:?}: a mass-0 lever is no lever (leave it out)"));
                    }
                    bp
                }
            };
            for &p in parts {
                if given[class as usize][p as usize] {
                    return Err(format!("oracle_reveal {text:?}: {}:{} is given twice", class.as_str(), if p == Part::Seen { "seen" } else { "unseen" }));
                }
                given[class as usize][p as usize] = true;
                spec.mass[class as usize][p as usize] = bp;
            }
        }
        if spec.mass(Class::Species, Part::Unseen) == 0 {
            if let Some(c) = Class::SET.into_iter().find(|&c| spec.mass(c, Part::Unseen) > 0) {
                return Err(format!(
                    "oracle_reveal {text:?}: the lever {}:unseen can never apply without species:unseen (no slot of the \
                     observation holds a never-seen mon otherwise)",
                    c.as_str()
                ));
            }
        }
        let canon = spec.canonical();
        if canon != text {
            return Err(format!("oracle_reveal {text:?} is not canonical: the canonical text of what it says is {canon:?}"));
        }
        Ok(spec)
    }
}

// ------------------------------------------------------------------------------------------------ the per-episode draw

/// Which episode and side an oracle is built for: the battle's seed words (each < 65,536, the `ep_seed` row) and the
/// side whose chain is told (0 = p1). The firing of every lever of mass below 1 is a pure function of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EpisodeKey {
    pub seed: [u32; 4],
    pub side: usize,
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The ONE uniform draw in 0..10,000 for (episode, side, the mon's packed index, class): splitmix64 over the seed
/// words (16 bits each), then over (domain tag, side, index, class). Deterministic, platform-independent.
pub fn uniform_bp(key: &EpisodeKey, mon: usize, class: Class) -> u16 {
    let seed = key.seed.iter().fold(0u64, |acc, &w| (acc << 16) | (w as u64 & 0xffff));
    let tag = 0x6f72_6163_6c65_0000u64 ^ ((key.side as u64) << 16) ^ ((mon as u64) << 8) ^ (class as u64);
    (splitmix64(splitmix64(seed) ^ tag) % MASS_DENOM as u64) as u16
}

/// Does a lever of mass `bp` fire on the draw for (key, mon, class)? Mass 1 always (no hash), mass 0 never.
pub fn fires(bp: u16, key: &EpisodeKey, mon: usize, class: Class) -> bool {
    match bp {
        0 => false,
        MASS_DENOM => true,
        _ => uniform_bp(key, mon, class) < bp,
    }
}

/// The set classes told for one mon in one part: [moves, item, ability, spread].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Told(pub [bool; 4]);

impl Told {
    pub const NONE: Told = Told([false; 4]);
    pub const ALL: Told = Told([true; 4]);

    pub fn has(self, c: Class) -> bool {
        match c {
            Class::Species => false,
            _ => self.0[c as usize - 1],
        }
    }

    pub fn any(self) -> bool {
        self.0.iter().any(|&b| b)
    }
}

// ------------------------------------------------------------------------------------------------ the declared cells

/// How one block of a TAIL slot may differ from the pristine "never seen" mon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    /// A function of the species alone (the dex row, the Smogon ability prior): free.
    SpeciesDerived,
    /// Exactly 0.0, bit for bit: the fact is hidden until play reveals it.
    Zero,
    /// Exactly 1.0, bit for bit: the state of a mon that has never been on the field (full HP, the
    /// slot populated, recency "never seen", full Protect odds) — or `hp_revealed` under a told spread.
    One,
    /// A fact of the TRUE SET whose class is told: free here — gated by the differential against the opposing chain's
    /// own-team slot of the same mon.
    SetFact,
}

/// One declared block of the 122-cell slot: `(name, lo, hi, rule)`, cells `lo..hi`.
pub type SlotCells = (&'static str, usize, usize, Rule);

const MOVES_END: usize = POKEMON_MOVES_OFFSET + 4 * MOVE_SLOT_DIM;

/// The DECLARED rule of every cell of a tail slot whose set classes `told` are told (it tiles the slot exactly —
/// `the_declared_cells_tile_the_slot`, for all sixteen combinations). A told class's block is the set's fact; an
/// untold item / moves / spread is ZERO and an untold ability the species' prior; `hp_revealed` is 1 exactly when the
/// spread is told (the view says `spread_known`, as for an own mon); every other cell is the pristine state of a
/// never-seen mon. `Told::NONE` is the X32 `species` level's table, `Told::ALL` its `full` level's.
pub fn slot_cells(told: Told) -> [SlotCells; 18] {
    let fact = |c: Class, untold: Rule| if told.has(c) { Rule::SetFact } else { untold };
    [
        ("species", POKEMON_SPECIES_OFFSET, POKEMON_ITEMS_OFFSET, Rule::SpeciesDerived),
        ("item", POKEMON_ITEMS_OFFSET, POKEMON_TYPES_OFFSET, fact(Class::Item, Rule::Zero)),
        ("types", POKEMON_TYPES_OFFSET, POKEMON_ABILITIES_OFFSET, Rule::SpeciesDerived),
        ("ability", POKEMON_ABILITIES_OFFSET, POKEMON_CONDITION_OFFSET, fact(Class::Ability, Rule::SpeciesDerived)),
        ("status", POKEMON_CONDITION_OFFSET, POKEMON_MOVES_OFFSET, Rule::Zero),
        ("moves", POKEMON_MOVES_OFFSET, MOVES_END, fact(Class::Moves, Rule::Zero)),
        ("hp_fraction", POKEMON_HP_OFFSET, POKEMON_SPECIES_KNOWN_OFFSET, Rule::One),
        ("species_known", POKEMON_SPECIES_KNOWN_OFFSET, POKEMON_COUNTER_OFFSET, Rule::One),
        ("status_counters", POKEMON_COUNTER_OFFSET, POKEMON_SPREAD_OFFSET, Rule::Zero),
        ("spread", POKEMON_SPREAD_OFFSET, POKEMON_HP_REVEALED_OFFSET, fact(Class::Spread, Rule::Zero)),
        ("hp_revealed", POKEMON_HP_REVEALED_OFFSET, POKEMON_HP_REVEALED_OFFSET + 1, if told.has(Class::Spread) { Rule::One } else { Rule::Zero }),
        ("hp_type_probs", POKEMON_HP_REVEALED_OFFSET + 1, POKEMON_SLEEP_BELIEF_OFFSET, Rule::Zero),
        ("sleep_belief", POKEMON_SLEEP_BELIEF_OFFSET, POKEMON_RECENCY_OFFSET, Rule::Zero),
        ("recency", POKEMON_RECENCY_OFFSET, POKEMON_PROTECT_OFFSET, Rule::One),
        ("protect", POKEMON_PROTECT_OFFSET, POKEMON_LAST_ACTION_OFFSET, Rule::One),
        ("last_action", POKEMON_LAST_ACTION_OFFSET, POKEMON_VECTOR_DIM, Rule::Zero),
        ("trapped_maybe_trapped", POKEMON_TRAPPED_OFFSET, POKEMON_ACTIVE_OFFSET, Rule::Zero),
        ("active", POKEMON_ACTIVE_OFFSET, POKEMON_FULL_DIM, Rule::Zero),
    ]
}

fn told_names(told: Told) -> String {
    let t: Vec<&str> = Class::SET.into_iter().filter(|&c| told.has(c)).map(Class::as_str).collect();
    if t.is_empty() {
        "species only".into()
    } else {
        format!("species + {}", t.join(" + "))
    }
}

/// THROWS unless `slot` obeys the declared cells of a tail slot told `told` ([`slot_cells`]) — the producer's guard: a
/// tail slot that carries a fact its told classes do not reveal (a move, an item, a spread, a status, an on-field
/// flag, …) is a FAULT. Bit comparison: `-0.0` is not `0.0`, a NaN equals nothing.
pub fn check_slot(told: Told, species: &str, slot: &[f32]) -> CoreResult<()> {
    if slot.len() != POKEMON_FULL_DIM {
        return Err(fault(format!("oracle: the {species} slot has {} cells, not {POKEMON_FULL_DIM}", slot.len())));
    }
    for (name, lo, hi, rule) in slot_cells(told) {
        let want = match rule {
            Rule::SpeciesDerived | Rule::SetFact => continue,
            Rule::Zero => 0.0f32,
            Rule::One => 1.0f32,
        };
        if let Some(k) = (lo..hi).find(|&k| slot[k].to_bits() != want.to_bits()) {
            return Err(fault(format!(
                "oracle: the {species} tail slot breaks its declared rule: block `{name}` cell {k} is {:?}, must be exactly {want:?} \
                 — a hidden fact would reach the observation (told: {})",
                slot[k],
                told_names(told)
            )));
        }
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------ the oracle

/// One opponent mon the oracle knows: its species, its dex num, its true set, which of its levers fired this episode
/// and the slot the encoder writes for it while it is unseen (if its species is told).
#[derive(Debug, Clone)]
pub struct OracleMon {
    /// The species id (`skarmory`).
    pub species: String,
    /// The dex num — a forme shares its base species' num, which is what a revealed mon is matched
    /// on (`gen3_data.species.base_form_ids`; Species Clause is by num).
    pub num: i64,
    /// The true set as `team::unpack` read it (the overlay reads it).
    set: PokemonSet,
    /// Is its species told while it is unseen (its `species:unseen` lever fired: it has a tail slot)?
    pub species_told: bool,
    /// The set classes told while it is UNSEEN (its tail slot) and while it is SEEN (the overlay).
    pub unseen_told: Told,
    pub seen_told: Told,
    slot: Option<[f32; POKEMON_FULL_DIM]>,
}

impl OracleMon {
    /// The 122 cells of this mon's tail slot (`None`: its species is not told, so it has none).
    pub fn slot(&self) -> Option<&[f32; POKEMON_FULL_DIM]> {
        self.slot.as_ref()
    }
}

/// The opponent's team as the observation is told it — built once per episode and side from the
/// packed team, then read at every decision of that side's chain.
#[derive(Debug, Clone)]
pub struct Oracle {
    spec: OracleSpec,
    key: EpisodeKey,
    mons: Vec<OracleMon>,
}

/// The dex num of a species id the encoder's table knows (a missing row, or one with no positive
/// num, is a FAULT — never a quiet slot).
pub fn species_num(id: &str) -> CoreResult<i64> {
    match data::tables().species.get(id) {
        Some(r) if r.num >= 1.0 => Ok(r.num as i64),
        Some(r) => Err(fault(format!("oracle: species {id:?} has dex num {} (not a valid num)", r.num))),
        None => Err(fault(format!("oracle: species {id:?} has no row in the encoder's species table"))),
    }
}

/// A packed move's id as the OWNER's request spells it: a bare `hiddenpower` becomes `hiddenpower<type>`, the type
/// the set declares (`hp_type`) or the IVs derive (gen 3) — `state::typed_hp_move_id`, which needs a `MonState`.
fn typed_move_id(set: &PokemonSet, mv: &str) -> String {
    let id = to_id(mv);
    if id == "hiddenpower" {
        let ty = if set.hp_type.is_empty() { crate::state::hidden_power_type(&set.ivs).to_string() } else { to_id(&set.hp_type) };
        if !ty.is_empty() && ty != "normal" {
            return format!("hiddenpower{ty}");
        }
    }
    id
}

/// The id a TOLD move is written under. The Hidden Power TYPE is a fact of the SPREAD class (the IVs determine it;
/// `design_information_levers.md` §3), not of `moves`: with the spread told the move is the owner's typed id (an own
/// mon's row, so `full` is X32's bytes exactly); with only `moves` told it is the BARE `hiddenpower` — written exactly
/// as an opponent's Hidden Power play revealed (no type, the Hidden-Power belief cells as play left them).
fn told_move_id(set: &PokemonSet, mv: &str, told: Told) -> String {
    let id = to_id(mv);
    if told.has(Class::Spread) {
        typed_move_id(set, mv)
    } else if id.starts_with("hiddenpower") {
        // a packed set may spell the TYPED move (`Hidden Power Grass`): untold, it is the bare move all the same
        "hiddenpower".to_string()
    } else {
        id
    }
}

/// The told facts of `set` an opponent's reading cannot hold until play reveals them, written onto `m` WITHOUT
/// touching what the reading already knows: the item (only while the reading's is the unknown sentinel — a revealed,
/// consumed or removed one stays), the ability (only while none is revealed), and the spread (never revealed in
/// play). An untold class is left exactly as the reading has it.
fn apply_set(m: &mut PMon, set: &PokemonSet, told: Told) -> CoreResult<()> {
    // `None` is NOT unknown: it is an item play consumed or removed (the reading's `item` is the sentinel until the
    // item is learned), so only the sentinel is overwritten
    if told.has(Class::Item) && m.item.as_deref() == Some(UNKNOWN_ITEM) {
        m.item = if set.item.is_empty() { None } else { Some(to_id(&set.item)) };
    }
    if told.has(Class::Ability) && m.ability().is_none() && !set.ability.is_empty() {
        // not a PROTOCOL reveal: the told set leaves the public flags alone (`gen3_obs_facts_v1`)
        m.set_ability_from_request(&set.ability);
    }
    if told.has(Class::Spread) {
        // the SAME backfill an own mon's reading takes from its packed team (`Pokemon.backfill_spread_from_teambuilder`):
        // the nature lower-cased (the encoder's table is keyed so), `serious` when none is declared
        m.backfill_spread(&TbMon {
            nickname: None,
            species: Some(set.species.clone()),
            evs: set.evs.iter().map(|&x| x as i64).collect(),
            ivs: set.ivs.iter().map(|&x| x as i64).collect(),
            nature: if set.nature.is_empty() { None } else { Some(set.nature.clone()) },
        });
    }
    Ok(())
}

/// The view of a mon whose spread is told: `spread_known`, with the spread the slot writer reads.
fn told_view(base: MonView, m: &PMon) -> MonView {
    MonView { ivs: m.ivs.clone(), evs: m.evs.clone(), nature: m.nature.clone(), spread_known: true, ..base }
}

/// The slot of an UNSEEN mon whose species is told and whose set classes `told` (at least one) are: the row the
/// encoder writes for an OWN mon at full HP holding exactly those facts of its set, never on the field — the same
/// `slot::populated_slot` as every other slot, fed the mon (`PMon::from_species`, the reported full HP, the told item
/// / ability / spread and the told moves at full PP), its view (`spread_known` only when the spread is told), a side
/// that has seen nothing, and the opponent's perspective. At `Told::ALL` it is the X32 `full` level's slot, byte for
/// byte; at `Told::NONE` it equals [`hypothesis::hypothesis_slot`] (`a_told_slot_with_nothing_told_is_the_hypothesis_row`).
pub fn told_slot(id: &str, set: &PokemonSet, told: Told) -> CoreResult<[f32; POKEMON_FULL_DIM]> {
    let t = data::tables();
    let mut mon = PMon::from_species(id, None)?;
    mon.set_hp_status("100/100", true)?;
    apply_set(&mut mon, set, told)?;
    if told.has(Class::Moves) {
        for mv in &set.moves {
            mon.learn_move(&told_move_id(set, mv, told))?;
        }
    }
    let base = mon_view(&mon, false, false)?;
    let live = if told.has(Class::Spread) { told_view(base, &mon) } else { base };
    let trk = SideTrackers::new(ClockConfig::default());
    let mut s = [if super::NAN_POISON { f32::NAN } else { 0.0 }; POKEMON_FULL_DIM];
    slot::populated_slot(&trk, t, &mon, &live, false, None, [false, false, false], &mut s)?;
    if let Some(k) = s.iter().position(|x| x.is_nan()) {
        return Err(fault(format!("oracle: told_slot({id}): cell {k} was never written")));
    }
    Ok(s)
}

impl Oracle {
    /// The oracle for `packed` (the OTHER side's packed team) at `spec`, for the episode and side `key`. An `off` spec
    /// has no oracle (the chain holds `None`), so it is refused here. Every lever's firing is decided now
    /// ([`fires`]) and every tail slot is built now and checked against its declared cells ([`check_slot`]).
    pub fn new(spec: OracleSpec, packed: &str, dex: &Dex, key: EpisodeKey) -> CoreResult<Oracle> {
        if spec.is_off() {
            return Err(fault("oracle: the `off` spec builds no oracle (a chain without one is the `off` mode)"));
        }
        let sets = crate::team::unpack(packed, dex).map_err(|e| fault(format!("oracle: the opponent team does not unpack: {e}")))?;
        if sets.is_empty() || sets.len() > TEAM_SIZE {
            return Err(fault(format!("oracle: the opponent team holds {} mons (1..={TEAM_SIZE} expected)", sets.len())));
        }
        let mut mons = Vec::with_capacity(sets.len());
        for (k, set) in sets.iter().enumerate() {
            // `unpack` keeps the packed team's spelling (`Tyranitar`, `Deoxys-Attack`): the encoder's
            // tables and the reading are keyed by poke-env's id (`tyranitar`, `deoxysattack`)
            let id = to_id(&set.species);
            let fire = |c: Class, p: Part| fires(spec.mass(c, p), &key, k, c);
            let species_told = fire(Class::Species, Part::Unseen);
            let told = |p: Part| Told([fire(Class::Moves, p), fire(Class::Item, p), fire(Class::Ability, p), fire(Class::Spread, p)]);
            let (unseen_told, seen_told) = (told(Part::Unseen), told(Part::Seen));
            let slot = if species_told {
                let s = if unseen_told.any() { told_slot(&id, set, unseen_told)? } else { hypothesis::hypothesis_slot(&id)? };
                check_slot(unseen_told, &id, &s)?;
                Some(s)
            } else {
                None
            };
            mons.push(OracleMon { num: species_num(&id)?, species: id, set: set.clone(), species_told, unseen_told, seen_told, slot });
        }
        Ok(Oracle { spec, key, mons })
    }

    pub fn spec(&self) -> OracleSpec {
        self.spec
    }

    pub fn key(&self) -> EpisodeKey {
        self.key
    }

    /// Every oracle mon, in the team's packed order.
    pub fn mons(&self) -> &[OracleMon] {
        &self.mons
    }

    /// The opponent mons the viewer has NOT yet seen AND whose species is told, in the order their slots follow the
    /// seen prefix: dex num ascending (stable: a duplicate species keeps its packed order). One oracle entry is
    /// consumed per revealed mon, matched by dex num, over the WHOLE team (told or not); a revealed mon the oracle
    /// team does not hold is a FAULT (the producer's and the labels' shared invariant).
    pub fn tail(&self, revealed: &[(String, PMon)]) -> CoreResult<Vec<&OracleMon>> {
        let mut left: Vec<&OracleMon> = self.mons.iter().collect();
        for (_, m) in revealed {
            let num = species_num(&m.species)?;
            match left.iter().position(|o| o.num == num) {
                Some(i) => {
                    left.remove(i);
                }
                None => {
                    let team: Vec<&str> = self.mons.iter().map(|o| o.species.as_str()).collect();
                    return Err(fault(format!(
                        "oracle: the revealed opponent {:?} (dex num {num}) is not on the oracle team {team:?}",
                        m.species
                    )));
                }
            }
        }
        left.retain(|o| o.species_told);
        left.sort_by_key(|o| o.num);
        Ok(left)
    }

    /// The view of a SEEN opponent mon under its fired `seen` levers: the reading's `mon` and its `live` view with the
    /// facts of the told classes of the true set play has NOT revealed written onto them, nothing the play HAS
    /// revealed overwritten — an observed move keeps its tracked PP, a consumed / removed / swapped item stays so, a
    /// revealed (or changed) ability is the reading's. `None` when the spec tells no seen part (before any lookup:
    /// the X32 `species` level) or none of this mon's `seen` levers fired. A transformed mon shows its TARGET's moves
    /// (the true set adds none of them), and a bare `hiddenpower` the reading learned stands for the set's typed one
    /// (no fifth move).
    pub fn overlay(&self, mon: &PMon, live: &MonView) -> CoreResult<Option<(PMon, MonView)>> {
        if !self.spec.tells_seen() {
            return Ok(None);
        }
        let num = species_num(&mon.species)?;
        let Some(om) = self.mons.iter().find(|o| o.num == num) else {
            let team: Vec<&str> = self.mons.iter().map(|o| o.species.as_str()).collect();
            return Err(fault(format!("oracle: the seen opponent {:?} (dex num {num}) is not on the oracle team {team:?}", mon.species)));
        };
        let told = om.seen_told;
        if !told.any() {
            return Ok(None);
        }
        let mut m = mon.clone();
        apply_set(&mut m, &om.set, told)?;
        if told.has(Class::Moves) && !m.transformed() {
            for mv in &om.set.moves {
                let id = told_move_id(&om.set, mv, told);
                let stands_in = id.starts_with("hiddenpower") && m.moves.contains("hiddenpower");
                if !stands_in && !m.moves.contains(&id) && m.moves.moves_ref().len() < 4 {
                    m.learn_move(&id)?;
                }
            }
        }
        let view = if told.has(Class::Spread) { told_view(live.clone(), &m) } else { live.clone() };
        Ok(Some((m, view)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mass-0.5 firing count over the fixed draw set of
    /// `a_half_mass_lever_fires_on_a_pinned_count_of_a_fixed_draw_set_and_mass_one_always` (6,000 draws), recorded when
    /// the draw was built. A change to [`uniform_bp`] moves which episodes a recorded run was told: never silently.
    const PINNED_HALF_MASS_FIRES: u32 = 3_020;

    fn every_told() -> impl Iterator<Item = Told> {
        (0u8..16).map(|b| Told([b & 1 != 0, b & 2 != 0, b & 4 != 0, b & 8 != 0]))
    }

    #[test]
    fn the_declared_cells_tile_the_slot() {
        for told in every_told() {
            let mut at = 0;
            for (name, lo, hi, _) in slot_cells(told) {
                assert_eq!(lo, at, "{told:?}: {name} starts at {lo}, the previous block ended at {at}");
                assert!(hi > lo, "{name} is empty");
                at = hi;
            }
            assert_eq!(at, POKEMON_FULL_DIM);
        }
    }

    fn guard_teeth(told: Told, good: &[f32; POKEMON_FULL_DIM]) {
        for (name, lo, _, rule) in slot_cells(told) {
            if matches!(rule, Rule::SpeciesDerived | Rule::SetFact) {
                continue;
            }
            let mut bad = *good;
            bad[lo] = if rule == Rule::Zero { 1.0 } else { 0.5 };
            let e = check_slot(told, "x", &bad).unwrap_err().message().to_string();
            assert!(e.contains(&format!("`{name}`")) && e.contains("declared rule"), "{told:?} {name}: {e}");
            let mut neg = *good;
            neg[lo] = if rule == Rule::Zero { -0.0 } else { f32::NAN };
            assert!(check_slot(told, "x", &neg).is_err(), "{told:?} {name}: -0.0 / NaN must fail");
        }
    }

    #[test]
    fn a_species_slot_passes_its_guard_and_every_block_has_teeth() {
        let good = hypothesis::hypothesis_slot("skarmory").unwrap();
        check_slot(Told::NONE, "skarmory", &good).unwrap();
        guard_teeth(Told::NONE, &good);
    }

    // ---- the set levels

    fn set(species: &str, moves: &[&str], item: &str, ability: &str) -> PokemonSet {
        let mut evs = [0u16; 6];
        evs[1] = 252;
        PokemonSet {
            name: species.into(),
            species: species.into(),
            item: item.into(),
            ability: ability.into(),
            moves: moves.iter().map(|m| m.to_string()).collect(),
            nature: "Impish".into(),
            evs,
            ..Default::default()
        }
    }

    fn oracle_at(spec: OracleSpec, sets: &[PokemonSet]) -> Oracle {
        let dex = Dex::for_gen(3);
        Oracle::new(spec, &crate::team::pack(sets, &dex), &dex, EpisodeKey::default()).expect("oracle")
    }

    fn skarmory() -> PokemonSet {
        set("skarmory", &["spikes", "roar", "drillpeck", "toxic"], "leftovers", "keeneye")
    }

    fn seen(species: &str) -> (PMon, MonView) {
        let m = PMon::from_species(species, None).unwrap();
        let v = mon_view(&m, true, false).unwrap();
        (m, v)
    }

    fn tail_slot(o: &Oracle) -> [f32; POKEMON_FULL_DIM] {
        *o.mons()[0].slot().expect("a told species has a tail slot")
    }

    #[test]
    fn a_full_tail_slot_tells_the_set_as_an_own_mons_row_would() {
        let sl = tail_slot(&oracle_at(OracleSpec::FULL, &[skarmory()]));
        let items = &data::tables().items;
        assert_eq!(sl[POKEMON_ITEMS_OFFSET] as f64, items["leftovers"], "the item's num");
        assert_eq!((sl[POKEMON_ITEMS_OFFSET + 1], sl[POKEMON_ITEMS_OFFSET + 2]), (1.0, 0.0), "known, not consumed");
        let known: Vec<f32> = (0..4).map(|m| sl[POKEMON_MOVES_OFFSET + m * MOVE_SLOT_DIM + 6]).collect();
        assert_eq!(known, [1.0; 4], "four moves, each marked known");
        for m in 0..4 {
            let b = POKEMON_MOVES_OFFSET + m * MOVE_SLOT_DIM;
            assert_eq!(sl[b + 7], sl[b + 8], "an unused move is at full PP");
        }
        assert_eq!(sl[POKEMON_SPREAD_OFFSET + 12], 1.0, "spread_known");
        assert_eq!(sl[POKEMON_SPREAD_OFFSET + 6 + 1], 1.0, "252 Atk EVs / 252");
        assert_eq!((sl[POKEMON_SPREAD_OFFSET + 13 + 1], sl[POKEMON_SPREAD_OFFSET + 13 + 2]), (1.1f32, 0.9f32), "Impish: +Def -SpA");
        assert_eq!(sl[POKEMON_HP_REVEALED_OFFSET], 1.0, "the told spread determines the Hidden-Power type");
        assert_eq!(sl[POKEMON_ACTIVE_OFFSET], 0.0, "listed, never on the field");
        assert_eq!(sl[POKEMON_RECENCY_OFFSET], 1.0, "never seen");
    }

    #[test]
    fn a_full_slot_passes_its_guard_every_pristine_block_has_teeth_and_the_levels_refuse_each_others_slots() {
        let full = tail_slot(&oracle_at(OracleSpec::FULL, &[skarmory()]));
        check_slot(Told::ALL, "skarmory", &full).unwrap();
        guard_teeth(Told::ALL, &full);
        let species = tail_slot(&oracle_at(OracleSpec::SPECIES, &[skarmory()]));
        let e = check_slot(Told::ALL, "skarmory", &species).unwrap_err().message().to_string();
        assert!(e.contains("`hp_revealed`"), "a species slot is not a full one: {e}");
        let e = check_slot(Told::NONE, "skarmory", &full).unwrap_err().message().to_string();
        assert!(e.contains("`item`"), "a full slot leaks the item through the species guard: {e}");
    }

    #[test]
    fn every_partial_tail_slot_passes_its_own_guard_has_teeth_and_leaks_through_no_narrower_one() {
        let sk = skarmory();
        let hyp = hypothesis::hypothesis_slot("skarmory").unwrap();
        for told in every_told() {
            let s = told_slot("skarmory", &sk, told).unwrap();
            check_slot(told, "skarmory", &s).unwrap_or_else(|e| panic!("{told:?}: {}", e.message()));
            guard_teeth(told, &s);
            // each told class's block differs from the never-seen row (it carries the fact) and each untold one is it
            for (c, lo, hi) in [
                (Class::Item, POKEMON_ITEMS_OFFSET, POKEMON_TYPES_OFFSET),
                (Class::Ability, POKEMON_ABILITIES_OFFSET, POKEMON_CONDITION_OFFSET),
                (Class::Moves, POKEMON_MOVES_OFFSET, MOVES_END),
                (Class::Spread, POKEMON_SPREAD_OFFSET, POKEMON_SLEEP_BELIEF_OFFSET),
            ] {
                let same = (lo..hi).all(|k| s[k].to_bits() == hyp[k].to_bits());
                assert_eq!(same, !told.has(c), "{told:?}: the {} block", c.as_str());
            }
            // a slot told a fact the ZERO rule hides (item, moves, spread) fails the guard of the combination without it
            for c in [Class::Item, Class::Moves, Class::Spread].into_iter().filter(|&c| told.has(c)) {
                let mut narrower = told;
                narrower.0[c as usize - 1] = false;
                assert!(check_slot(narrower, "skarmory", &s).is_err(), "{told:?}: a told {} passes the guard without it", c.as_str());
            }
        }
        // nothing told is exactly the never-seen row
        let none = told_slot("skarmory", &sk, Told::NONE).unwrap();
        assert_eq!(none.map(f32::to_bits), hyp.map(f32::to_bits), "a_told_slot_with_nothing_told_is_the_hypothesis_row");
    }

    /// The (move num, type cell) of each written move of a slot.
    fn move_cells(sl: &[f32; POKEMON_FULL_DIM]) -> Vec<(f32, f32)> {
        (0..4).map(|m| POKEMON_MOVES_OFFSET + m * MOVE_SLOT_DIM).filter(|&b| sl[b] != 0.0).map(|b| (sl[b], sl[b + 4])).collect()
    }

    #[test]
    fn the_hidden_power_type_is_a_spread_fact_never_a_moves_one() {
        let mut hp = set("skarmory", &["hiddenpower", "spikes", "roar", "toxic"], "leftovers", "keeneye");
        hp.ivs = [31, 30, 30, 31, 31, 31]; // a non-Dark type: the typed id differs from the bare one
        let bare_num = data::tables().moves["hiddenpower"].num as f32;
        let typed = typed_move_id(&hp, "hiddenpower");
        assert!(typed.starts_with("hiddenpower") && typed != "hiddenpower", "{typed}");
        let moves_only = told_slot("skarmory", &hp, Told([true, false, false, false])).unwrap();
        let with_spread = told_slot("skarmory", &hp, Told([true, false, false, true])).unwrap();
        let hp_of = |sl: &[f32; POKEMON_FULL_DIM]| move_cells(sl).into_iter().find(|(n, _)| *n == bare_num);
        let typed_num = data::tables().moves[typed.as_str()].num as f32;
        assert_ne!(typed_num, bare_num, "a typed Hidden Power has its own move num");
        let nums = |sl: &[f32; POKEMON_FULL_DIM]| move_cells(sl).into_iter().map(|(n, _)| n).collect::<Vec<f32>>();
        let bare = move_cells(&moves_only).into_iter().find(|(n, _)| *n == bare_num).expect("the bare Hidden Power is written");
        assert_eq!(bare.1, 0.0, "moves without spread: the BARE Hidden Power (no type)");
        assert!(!nums(&moves_only).contains(&typed_num), "moves without spread: no TYPED Hidden Power");
        assert!(nums(&with_spread).contains(&typed_num) && !nums(&with_spread).contains(&bare_num),
                "moves with spread: the owner's typed Hidden Power");
        assert_eq!(moves_only[POKEMON_HP_REVEALED_OFFSET], 0.0, "no Hidden-Power type is revealed without the spread");
        assert_eq!(with_spread[POKEMON_HP_REVEALED_OFFSET], 1.0);
        // a set that SPELLS the typed move is bare all the same without the spread
        let mut spelled = hp.clone();
        spelled.moves[0] = "Hidden Power Grass".into();
        let sl = told_slot("skarmory", &spelled, Told([true, false, false, false])).unwrap();
        assert!(nums(&sl).contains(&bare_num) && !nums(&sl).iter().any(|n| (355.0..=370.0).contains(n)), "{:?}", nums(&sl));
        // the seen overlay likewise
        let o = oracle_at(OracleSpec::parse("moves:seen").unwrap(), &[hp.clone()]);
        let (m, v) = seen("skarmory");
        let ids = |o: &Oracle| -> Vec<String> {
            o.overlay(&m, &v).unwrap().unwrap().0.moves.moves_ref().into_iter().map(|(_, mv)| mv.id.to_string()).collect()
        };
        let got = ids(&o);
        assert!(got.contains(&"hiddenpower".to_string()) && !got.contains(&typed), "moves:seen: the bare move {got:?}");
        let got = ids(&oracle_at(OracleSpec::FULL, &[hp]));
        assert!(got.contains(&typed), "full (spread told) keeps X32's typed move: {got:?}");
    }

    #[test]
    fn the_overlay_tells_a_seen_mon_only_what_play_has_not_revealed() {
        let o = oracle_at(OracleSpec::FULL, &[skarmory()]);
        // the item: the sentinel is told; a revealed one and a consumed / removed one (None) are the reading's
        let (m, v) = seen("skarmory");
        assert_eq!(m.item.as_deref(), Some(UNKNOWN_ITEM));
        let (m2, v2) = o.overlay(&m, &v).unwrap().expect("full overlays");
        assert_eq!(m2.item.as_deref(), Some("leftovers"));
        assert!(v2.spread_known && v2.nature.as_deref() == Some("impish"), "the spread is told, nature lower-cased like an own mon's");
        assert!(!v.spread_known, "the reading's own view is untouched");
        let (mut m, v) = seen("skarmory");
        m.item = Some("sitrusberry".into());
        assert_eq!(o.overlay(&m, &v).unwrap().unwrap().0.item.as_deref(), Some("sitrusberry"), "a revealed item stays");
        let (mut m, v) = seen("skarmory");
        m.item = None;
        m.consumed_item = Some("leftovers".into());
        let (m2, _) = o.overlay(&m, &v).unwrap().unwrap();
        assert_eq!((m2.item.as_deref(), m2.consumed_item.as_deref()), (None, Some("leftovers")), "a consumed / removed item stays so");
        // the ability: told while none is revealed, the reading's once it is
        let (m, v) = seen("skarmory");
        assert!(m.ability().is_none(), "two possible abilities: none revealed");
        assert_eq!(o.overlay(&m, &v).unwrap().unwrap().0.ability(), Some("keeneye"));
        let (mut m, v) = seen("skarmory");
        m.set_ability("sturdy");
        assert_eq!(o.overlay(&m, &v).unwrap().unwrap().0.ability(), Some("sturdy"), "a revealed (or changed) ability stays");
        // the moves: an observed move keeps its slot and tracked PP, the rest of the set is added, none duplicated
        let (mut m, v) = seen("skarmory");
        m.learn_move("spikes").unwrap();
        m.moves.base_moves_mut()[0].1.current_pp = 5;
        let (m2, _) = o.overlay(&m, &v).unwrap().unwrap();
        let got: Vec<(&str, u32)> = m2.moves.moves_ref().into_iter().map(|(k, mv)| (k, mv.current_pp)).collect();
        assert_eq!(got.len(), 4, "{got:?}");
        assert_eq!(got[0], ("spikes", 5), "the observed slot keeps its tracked PP");
        assert!(["roar", "drillpeck", "toxic"].iter().all(|id| got.iter().any(|(k, pp)| k == id && *pp > 5)), "{got:?}");
        // a bare Hidden Power the reading learned stands for the set's typed one (no fifth move)
        let hp = oracle_at(OracleSpec::FULL, &[set("skarmory", &["hiddenpower", "spikes"], "leftovers", "keeneye")]);
        let (mut m, v) = seen("skarmory");
        m.learn_move("hiddenpower").unwrap();
        let (m2, _) = hp.overlay(&m, &v).unwrap().unwrap();
        let ids: Vec<&str> = m2.moves.moves_ref().into_iter().map(|(k, _)| k).collect();
        assert_eq!(ids.len(), 2, "{ids:?}");
        assert!(ids.contains(&"hiddenpower") && ids.contains(&"spikes"), "{ids:?}");
        // a transformed mon shows its TARGET's moves: the set adds none
        let (mut m, v) = seen("skarmory");
        m.moves.transform = Some(Box::new(Default::default()));
        assert!(o.overlay(&m, &v).unwrap().unwrap().0.moves.moves_ref().is_empty(), "no true move is added to a transformed mon");
    }

    #[test]
    fn a_partial_overlay_tells_only_its_classes() {
        let only = |c: Class| oracle_at(OracleSpec::with(&[(c, Part::Seen, MASS_DENOM)]), &[skarmory()]);
        let (m, v) = seen("skarmory");
        let (mi, vi) = only(Class::Item).overlay(&m, &v).unwrap().expect("item:seen overlays");
        assert_eq!(mi.item.as_deref(), Some("leftovers"));
        assert!(mi.ability().is_none() && mi.moves.moves_ref().is_empty() && !vi.spread_known, "item only");
        let (mm, vm) = only(Class::Moves).overlay(&m, &v).unwrap().expect("moves:seen overlays");
        assert_eq!(mm.moves.moves_ref().len(), 4);
        assert_eq!(mm.item.as_deref(), Some(UNKNOWN_ITEM), "the item stays unknown");
        assert!(mm.ability().is_none() && !vm.spread_known, "moves only");
        let (ma, va) = only(Class::Ability).overlay(&m, &v).unwrap().expect("ability:seen overlays");
        assert_eq!(ma.ability(), Some("keeneye"));
        assert!(ma.moves.moves_ref().is_empty() && !va.spread_known);
        let (ms, vs) = only(Class::Spread).overlay(&m, &v).unwrap().expect("spread:seen overlays");
        assert!(vs.spread_known && vs.nature.as_deref() == Some("impish"));
        assert!(ms.moves.moves_ref().is_empty() && ms.ability().is_none() && ms.item.as_deref() == Some(UNKNOWN_ITEM));
        // an unseen-only spec overlays no seen mon
        let u = oracle_at(OracleSpec::parse("species:unseen,moves:unseen").unwrap(), &[skarmory()]);
        assert!(u.overlay(&m, &v).unwrap().is_none());
    }

    #[test]
    fn the_overlay_is_none_without_a_seen_lever_and_a_seen_mon_off_the_team_is_a_fault() {
        let (m, v) = seen("skarmory");
        assert!(oracle_at(OracleSpec::SPECIES, &[skarmory()]).overlay(&m, &v).unwrap().is_none());
        let (z, zv) = seen("zapdos");
        // the species level never looks the mon up (byte-identical to X32's)
        assert!(oracle_at(OracleSpec::SPECIES, &[skarmory()]).overlay(&z, &zv).unwrap().is_none());
        let e = oracle_at(OracleSpec::FULL, &[skarmory()]).overlay(&z, &zv).unwrap_err().message().to_string();
        assert!(e.contains("zapdos") && e.contains("not on the oracle team"), "{e}");
    }

    // ---- the spec text

    #[test]
    fn the_levels_and_their_clause_texts_parse_to_the_same_spec_and_render_as_the_level() {
        assert_eq!(OracleSpec::parse("off").unwrap(), OracleSpec::OFF);
        assert_eq!(OracleSpec::parse("species").unwrap(), OracleSpec::SPECIES);
        assert_eq!(OracleSpec::parse("species:unseen").unwrap(), OracleSpec::SPECIES);
        assert_eq!(OracleSpec::parse("full").unwrap(), OracleSpec::FULL);
        assert_eq!(OracleSpec::parse("species:unseen,moves:both,item:both,ability:both,spread:both").unwrap(), OracleSpec::FULL);
        for (name, s) in OracleSpec::LEVELS {
            assert_eq!(s.label(), name);
        }
        assert_eq!(OracleSpec::OFF.canonical(), "off");
        assert_eq!(OracleSpec::SPECIES.canonical(), "species:unseen");
        assert_eq!(OracleSpec::FULL.canonical(), "species:unseen,moves:both,item:both,ability:both,spread:both");
    }

    /// Canonical texts `agents.info_levers.spec.InfoReveal.canonical` writes; the Python side pins the SAME list against
    /// the spec module and feeds each to the core (`src/utils/rust_env/oracle_reveal_integration_test.py`), so the two
    /// languages cannot disagree on one.
    pub const CANONICAL_SAMPLES: [&str; 12] = [
        "moves:seen@0.95", "item:seen@0.95", "spread:seen@0.95", "moves:seen@0.5", "moves:seen@0.25", "moves:seen@0.0001",
        "species:unseen@0.5", "species:unseen,moves:unseen", "species:unseen,moves:both@0.9",
        "species:unseen,moves:seen,moves:unseen@0.5", "moves:seen,item:seen,ability:seen,spread:seen",
        "species:unseen@0.25,item:seen@0.75,item:unseen@0.125,spread:both",
    ];

    #[test]
    fn the_canonical_texts_round_trip() {
        for t in CANONICAL_SAMPLES {
            let s = OracleSpec::parse(t).unwrap_or_else(|e| panic!("{t}: {e}"));
            assert_eq!(s.canonical(), t);
            assert_eq!(s.label(), t);
        }
        assert_eq!(OracleSpec::parse("moves:seen@0.95").unwrap().mass(Class::Moves, Part::Seen), 9_500);
        assert_eq!(OracleSpec::parse("species:unseen@0.125,moves:unseen").unwrap().mass(Class::Species, Part::Unseen), 1_250);
    }

    #[test]
    fn everything_else_is_refused_by_name() {
        for (t, want) in [
            ("hp:seen", "hp:seen"), ("counters:seen", "hidden duration"), ("moves:seen@0.95~placebo", "~placebo"),
            ("moves:seen~map", "~map"), ("moves:unseen", "without species:unseen"), ("item:both", "without species:unseen"),
            ("everything", "names no class"), ("moves", "not canonical"), ("moves:seen@1", "not canonical"),
            ("moves:seen@0.50", "not canonical"), ("moves:seen@.5", "not canonical"), ("moves:seen@0.00001", "not canonical"),
            ("moves:seen@0", "not canonical"), ("moves:seen@0.0", "mass-0"), ("item:seen,moves:seen", "not canonical"),
            ("species:unseen,moves:seen,moves:unseen", "not canonical"), ("moves:seen,moves:seen", "twice"), ("species:seen", "public"),
            ("species:both", "public"), ("moves:seen~truth", "not canonical"), ("", "empty clause"), ("moves:seen,", "empty clause"),
            ("Moves:seen", "names no class"), (" off", "names no class"), ("species:unseen@1.0", "not canonical"),
        ] {
            let e = OracleSpec::parse(t).expect_err(t);
            assert!(e.contains(want), "{t:?}: {e}");
        }
    }

    // ---- the per-episode draw

    #[test]
    fn a_half_mass_lever_fires_on_a_pinned_count_of_a_fixed_draw_set_and_mass_one_always() {
        // 1,000 episodes x 6 mons: the draw is a pure function of (seed, side, mon, class)
        let mut fired = 0u32;
        let mut n = 0u32;
        for e in 0..1000u32 {
            let key = EpisodeKey { seed: [e % 65_536, (e * 7) % 65_536, 3, 4], side: (e % 2) as usize };
            for mon in 0..6 {
                n += 1;
                fired += fires(5_000, &key, mon, Class::Moves) as u32;
                assert!(fires(MASS_DENOM, &key, mon, Class::Moves) && !fires(0, &key, mon, Class::Moves));
            }
        }
        assert_eq!(n, 6_000);
        eprintln!("mass-0.5 fires: {fired} / {n}");
        // PINNED: a change to the hash moves which episodes are told (a recorded run's realised reveal)
        assert_eq!(fired, PINNED_HALF_MASS_FIRES, "the mass-0.5 firing count over the fixed draw set moved");
        // the draws of one (episode, side, mon) differ by class and by side (independent levers)
        let k = EpisodeKey { seed: [1, 2, 3, 4], side: 0 };
        let u: Vec<u16> = Class::ALL.iter().map(|&c| uniform_bp(&k, 0, c)).collect();
        assert!(u.windows(2).any(|w| w[0] != w[1]), "{u:?}");
        assert_ne!(uniform_bp(&k, 0, Class::Moves), uniform_bp(&EpisodeKey { side: 1, ..k }, 0, Class::Moves));
    }

    #[test]
    fn a_partial_oracle_decides_per_mon_and_its_tail_holds_only_the_told_species() {
        let team = vec![
            skarmory(),
            set("tyranitar", &["crunch"], "leftovers", "sandstream"),
            set("blissey", &["softboiled"], "leftovers", "naturalcure"),
            set("swampert", &["surf"], "leftovers", "torrent"),
            set("metagross", &["meteormash"], "leftovers", "clearbody"),
            set("zapdos", &["thunderbolt"], "leftovers", "pressure"),
        ];
        let dex = Dex::for_gen(3);
        let packed = crate::team::pack(&team, &dex);
        let spec = OracleSpec::parse("species:unseen@0.5,moves:both@0.5").unwrap();
        let (mut some, mut none, mut moves_split) = (false, false, false);
        for e in 0..64u32 {
            let key = EpisodeKey { seed: [e, 1, 2, 3], side: 1 };
            let o = Oracle::new(spec, &packed, &dex, key).unwrap();
            let told: Vec<bool> = o.mons().iter().map(|m| m.species_told).collect();
            for (k, m) in o.mons().iter().enumerate() {
                assert_eq!(m.species_told, fires(5_000, &key, k, Class::Species));
                assert_eq!(m.slot().is_some(), m.species_told, "a slot exactly for a told species");
                // ONE draw per (mon, class): at equal masses the seen and unseen parts agree
                assert_eq!(m.unseen_told, m.seen_told);
                moves_split |= m.species_told && !m.unseen_told.has(Class::Moves);
            }
            let tail = o.tail(&[]).unwrap();
            assert_eq!(tail.len(), told.iter().filter(|&&t| t).count());
            assert!(tail.windows(2).all(|w| w[0].num <= w[1].num), "dex-num order");
            some |= told.iter().any(|&t| t);
            none |= told.iter().any(|&t| !t);
            // the same key builds the same oracle (deterministic)
            let again = Oracle::new(spec, &packed, &dex, key).unwrap();
            assert_eq!(again.mons().iter().map(|m| (m.species_told, m.unseen_told)).collect::<Vec<_>>(),
                       o.mons().iter().map(|m| (m.species_told, m.unseen_told)).collect::<Vec<_>>());
        }
        assert!(some && none && moves_split, "non-vacuity: told, untold and species-without-moves all occurred");
    }

    #[test]
    fn the_off_spec_builds_no_oracle() {
        let dex = Dex::for_gen(3);
        let e = Oracle::new(OracleSpec::OFF, &crate::team::pack(&[skarmory()], &dex), &dex, EpisodeKey::default()).unwrap_err();
        assert!(e.message().contains("builds no oracle"));
    }
}
