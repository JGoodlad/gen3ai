//! Team — bit-faithful port of Showdown's `Teams.pack` / `Teams.unpack`
//! (`deps/pokemon-showdown/sim/teams.ts`), scoped to one team of gen-3 sets.
//!
//! **Why this is load-bearing.** The bridge feeds a PACKED team string to
//! `>player p1 {…}`; that string must round-trip through Showdown's
//! pack/unpack byte-identically, and the unpacked sets feed exact stat
//! computation. Our actual producer is poke-env's `TeambuilderPokemon.packed`
//! (a reimplementation of `Teams.pack` that emits LOWERCASE ids), but the
//! consumer at `>player` is Node's Showdown, so we mirror **Showdown's**
//! pack/unpack semantics here:
//!
//! - [`pack`] emits the case-preserving `packName` form (Showdown writes
//!   `Leftovers`/`IceBeam`, not `leftovers`/`icebeam`), the raw ability id (NOT
//!   a slot letter — sim `pack()` never computes one), and collapses all-default
//!   EV/IV/level/happiness fields exactly like `teams.ts`.
//! - [`unpack`] is case-INSENSITIVE (it canonicalizes ids through the dex like
//!   `unpackName`), so it accepts both Showdown's and poke-env's packed strings
//!   and reproduces the same [`PokemonSet`] either way.
//!
//! ## Pack format (per set, 11 `|`-fields + a folded comma tail)
//!
//! `name | species | item | ability | moves | nature | evs | gender | ivs |
//! shiny | level | happiness,hpType,pokeball,gmax,dmaxLevel,teraType`
//!
//! Sets are joined by `]` (no trailing). See `teams.ts:120-209` (pack) and
//! `teams.ts:215-346` (unpack). The defaults a faithful port must reproduce:
//! EV missing → 0, IV missing → 31, level omitted at 100, happiness omitted at
//! 255, species field empty when the nickname equals the species, shiny `'S'`.
//!
//! ## Gen-3 note on ability abbreviations
//!
//! Showdown's unpack accepts the slot letters `''`/`'0'`/`'1'`/`'H'`/`'S'`,
//! resolving them against `species.abilities[slot]`. Our gen-3 data
//! (`gen3_species.json`) carries NO per-species ability list, so we cannot
//! resolve them. In practice this never bites: both Showdown's `pack()` and
//! poke-env's packer emit the **raw ability id** (`oblivious`, `levitate`), so
//! abbreviations do not occur in any team string our bridge produces. We decode
//! an unresolvable abbreviation to the empty string (No Ability) rather than
//! crash — see [`unpack`].
//!
//! ## Bounded assumptions (validator-clean input)
//!
//! The codec is byte-exact for the strings our bridge actually carries (teams
//! that passed Showdown's validator). Two deliberate, bounded deviations on
//! *malformed* input — neither reachable from a real gen-3 OU team:
//!
//! - Numeric fields (EV/IV/level/happiness) are stored in their natural integer
//!   widths (`u8`/`u16`). Showdown stores an out-of-range token verbatim and
//!   defers the range check to its validator; a value past our width falls back
//!   to its default instead of round-tripping. In-range values are byte-exact.
//! - `gender` keeps only the first character (`'M'`/`'F'`/`'N'`); a multi-char
//!   gender field would not round-trip. Real packed strings carry one char.

use crate::dex::{to_id, Dex};

/// Per-stat order used throughout the packed format: HP, Atk, Def, SpA, SpD, Spe.
pub const STAT_ORDER: [&str; 6] = ["hp", "atk", "def", "spa", "spd", "spe"];

/// A fully-decoded Pokémon set, the unit `pack`/`unpack` round-trips.
///
/// Fields mirror Showdown's `PokemonSet` for the gen-3-relevant subset.
/// Defaults are materialized at decode time (level 100, EVs 0, IVs 31,
/// happiness 255) so a `PokemonSet` is always a concrete, ready-to-use set —
/// unlike Showdown's `unpack`, which leaves those keys `undefined` and defers
/// the defaults downstream. For exact re-packing those concrete defaults
/// collapse back to empty fields (see [`pack`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PokemonSet {
    /// Nickname; equals the species display intent when there is no real nickname.
    pub name: String,
    /// Species id (normalized, e.g. `suicune`).
    pub species: String,
    /// Item id, or `""` for none.
    pub item: String,
    /// Ability id, or `""` for No Ability.
    pub ability: String,
    /// Move ids in slot order (1–4 for gen-3).
    pub moves: Vec<String>,
    /// Nature name (raw, e.g. `Timid`); `""` if none.
    pub nature: String,
    /// EVs, HP/Atk/Def/SpA/SpD/Spe; defaults to 0.
    pub evs: [u16; 6],
    /// IVs, HP/Atk/Def/SpA/SpD/Spe; defaults to 31.
    pub ivs: [u8; 6],
    /// Gender override (`'M'`/`'F'`/`'N'`), or `None` for species default.
    pub gender: Option<char>,
    /// Shiny flag.
    pub shiny: bool,
    /// Level; defaults to 100.
    pub level: u8,
    /// Happiness; defaults to 255 (matters for Return/Frustration BP).
    pub happiness: u8,
    /// Hidden Power type override from the misc tail (`misc[1]`); usually `""`
    /// in our data (HP type is carried in the move id instead). Round-tripped.
    pub hp_type: String,
}

impl Default for PokemonSet {
    fn default() -> Self {
        PokemonSet {
            name: String::new(),
            species: String::new(),
            item: String::new(),
            ability: String::new(),
            moves: Vec::new(),
            nature: String::new(),
            evs: [0; 6],
            ivs: [31; 6],
            gender: None,
            shiny: false,
            level: 100,
            happiness: 255,
            hp_type: String::new(),
        }
    }
}

/// `packName` (teams.ts:349-352): strip every non-alphanumeric char,
/// CASE-PRESERVING (unlike `to_id`, which lowercases). This is what Showdown's
/// `pack()` emits for species/item/ability/moves.
fn pack_name(name: &str) -> String {
    name.chars().filter(char::is_ascii_alphanumeric).collect()
}

/// `unpackName` (teams.ts:355-362): canonicalize an id token through the dex.
/// When the dex resolves the token, return its canonical DISPLAY name (e.g.
/// `Leftovers`, `Ice Beam`, `Mr. Mime`) — this is what Showdown's `unpackName`
/// returns and what lets [`pack`] reproduce the exact bytes via `pack_name`.
/// When it doesn't resolve, fall back to a heuristic re-spacing of the stripped
/// id (mirroring `unpackName`'s `dex.get(name).exists ? .name : re-space`).
/// Empty token → `""`. The lookup closure makes this id-table-agnostic (species
/// / item / ability / move).
fn unpack_name(token: &str, lookup: impl Fn(&str) -> Option<String>) -> String {
    if token.is_empty() {
        return String::new();
    }
    // Both Showdown's case-preserving ids and poke-env's lowercase ids normalize
    // to the same dex key under `to_id`, so a case-insensitive lookup resolves
    // either producer's string.
    lookup(&to_id(token)).unwrap_or_else(|| respace_id(token))
}

/// `unpackName`'s fallback re-spacing for an id the dex can't resolve
/// (teams.ts:357-361): `/([0-9]+)/g → ' $1 '`, `/([A-Z])/g → ' $1'`, collapse
/// double-spaces, trim. Reproduced for bit-faithfulness on unknown tokens
/// (validated gen-3 OU teams never hit it).
fn respace_id(token: &str) -> String {
    let mut s = String::new();
    let chars: Vec<char> = token.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if c.is_ascii_digit() {
            let prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
            if !prev_digit {
                s.push(' ');
            }
            s.push(c);
            let next_digit = chars.get(i + 1).is_some_and(char::is_ascii_digit);
            if !next_digit {
                s.push(' ');
            }
        } else if c.is_ascii_uppercase() {
            s.push(' ');
            s.push(c);
        } else {
            s.push(c);
        }
    }
    // Collapse double-spaces then trim.
    while s.contains("  ") {
        s = s.replace("  ", " ");
    }
    s.trim().to_string()
}

/// Decode one packed team string into its sets (Showdown `Teams.unpack`,
/// teams.ts:215-346), scoped to a single team.
///
/// Returns `Err` on a malformed string (a missing `|` delimiter through the
/// level field aborts the whole unpack with no partial team, mirroring
/// `teams.ts`'s `if (j < 0) return null`). An empty input is an empty team.
///
/// `dex` resolves each id token to its canonical DISPLAY name (Showdown's
/// dex-backed `unpackName`) so the decoded set re-packs byte-identically.
/// Ability slot abbreviations that cannot be resolved (our gen-3 data has no
/// per-species ability list) decode to `""` (see the module note).
pub fn unpack(packed: &str, dex: &Dex) -> Result<Vec<PokemonSet>, String> {
    if packed.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let mut i = 0usize;
    // Mirror Showdown's Teams.unpack cursor walk (teams.ts:215-346): each set is
    // 11 `|`-terminated fields (name..level) followed by a misc field that runs to
    // the next `]` (set separator) or end-of-string. A `]` inside ANY of the first
    // 11 fields is ordinary data, NOT a separator — a nickname may legally contain
    // `]`. (A top-level `split(']')` would corrupt such a set into two chunks.)
    // Delimiters are ASCII, so the byte offsets are always char boundaries even
    // when a field holds multibyte text (e.g. Nidoran♀).
    while i < packed.len() {
        let mut fields: Vec<&str> = Vec::with_capacity(12);
        for _ in 0..11 {
            match packed[i..].find('|') {
                Some(rel) => {
                    let j = i + rel;
                    fields.push(&packed[i..j]);
                    i = j + 1;
                }
                // A missing `|` through the level field aborts the whole unpack
                // with no partial team (teams.ts's `if (j < 0) return null`).
                None => {
                    return Err(format!(
                        "packed set truncated: missing `|` after field {} near {:?}",
                        fields.len(),
                        &packed[i..]
                    ))
                }
            }
        }
        // Misc field: runs to the next `]` (set separator) or end-of-string.
        let end = packed[i..].find(']').map_or(packed.len(), |rel| i + rel);
        fields.push(&packed[i..end]);
        out.push(build_set(&fields, dex)?);
        if end >= packed.len() {
            break;
        }
        i = end + 1; // skip the `]`
    }
    Ok(out)
}

/// Build one `PokemonSet` from its 12 already-split fields (name..level + misc).
fn build_set(fields: &[&str], dex: &Dex) -> Result<PokemonSet, String> {
    debug_assert_eq!(fields.len(), 12, "the cursor walk always yields 12 fields");
    let mut set = PokemonSet::default();

    // 1. name — raw substring, verbatim.
    set.name = fields[0].to_string();

    // 2. species — unpackName(.., Dex.species) || name. Empty/unknown token
    //    falls back to the nickname (teams.ts:244 `|| set.name`).
    let sp = unpack_name(fields[1], |id| dex.species(id).map(|s| s.name.clone()));
    set.species = if sp.is_empty() { set.name.clone() } else { sp };

    // 3. item — unpackName(.., Dex.items); empty → "".
    set.item = unpack_name(fields[2], |id| dex.item(id).map(|n| n.name.clone()));

    // 4. ability — '' / '0' / '1' / 'H' / 'S' resolve against species.abilities
    //    (we cannot — no per-species ability data), else unpackName(Dex.abilities).
    //    Our producers emit raw ids, so the abbreviation branch is decoded to "".
    set.ability = match fields[3] {
        "" => String::new(),
        "0" | "1" | "H" | "S" => String::new(), // unresolvable without species abilities
        other => unpack_name(other, |id| dex.ability(id).map(|n| n.name.clone())),
    };

    // 5. moves — comma-split ids, FAITHFUL to Showdown: empty tokens are kept
    //    (`'BodySlam,'` → `["Body Slam", ""]`) and an empty field yields a single
    //    empty token (`''.split(',')` → `[""]`), so pack(unpack(x)) is byte-exact.
    //    Valid gen-3 sets carry 1–4 real moves; the empty-token case is latent.
    set.moves = fields[4]
        .split(',')
        .map(|tok| unpack_name(tok, |id| dex.moves(id).map(|m| m.name.clone())))
        .collect();

    // 6. nature — raw name kept (case-preserving).
    set.nature = fields[5].to_string();

    // 7. evs — present field → per-stat Number||0; empty field → all-0 default.
    //    Reads only the first 6 comma-slots (matching JS `split(',', 6)` truncation).
    if !fields[6].is_empty() {
        let parts: Vec<&str> = fields[6].split(',').collect();
        for (k, slot) in set.evs.iter_mut().enumerate() {
            *slot = parts.get(k).and_then(|s| s.parse().ok()).unwrap_or(0);
        }
    }

    // 8. gender — verbatim first char if non-empty, else species default.
    set.gender = fields[7].chars().next();

    // 9. ivs — present field → per-slot. Showdown: `ivs[k] === '' ? 31 : Number(ivs[k])||0`.
    //    A LITERAL empty slot ('') → 31; a slot BEYOND a short field (undefined) → 0.
    //    (Conflating the two would mis-decode IVs, hence stats.)
    if !fields[8].is_empty() {
        let parts: Vec<&str> = fields[8].split(',').collect();
        for (k, slot) in set.ivs.iter_mut().enumerate() {
            *slot = match parts.get(k) {
                Some(&"") => 31,           // present-but-empty slot
                Some(s) => s.parse().unwrap_or(0),
                None => 0,                 // index beyond a short field
            };
        }
    }

    // 10. shiny — any non-empty span → true.
    set.shiny = !fields[9].is_empty();

    // 11. level — parseInt if present, else default 100.
    if !fields[10].is_empty() {
        set.level = parse_int_prefix(fields[10]).unwrap_or(100);
    }

    // 12. happiness + misc tail — comma list [happiness, hpType, pokeball, gmax,
    //     dmaxLevel, teraType]. Present-but-blank misc[0] → 255; an entirely-empty
    //     misc field → no override (keep the 255 default).
    let misc_field = fields[11];
    if !misc_field.is_empty() {
        let misc: Vec<&str> = misc_field.split(',').collect();
        if let Some(&h) = misc.first() {
            set.happiness = if h.is_empty() {
                255
            } else {
                parse_int_prefix(h).unwrap_or(255)
            };
        }
        set.hp_type = misc.get(1).map_or(String::new(), |s| s.to_string());
        // pokeball / gmax / dmaxLevel / teraType are gen-3-irrelevant and always
        // empty in our data; pack re-emits only non-default fields, none set here.
    }

    Ok(set)
}

/// `parseInt`-style leading-decimal-integer parse (tolerates trailing junk),
/// matching `teams.ts`'s `parseInt(substr)` for level/happiness.
fn parse_int_prefix(s: &str) -> Option<u8> {
    let digits: String = s
        .trim_start()
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

/// Pack a team into Showdown's packed string (`Teams.pack`, teams.ts:120-209),
/// scoped to one team. Mirrors every default-collapse rule so the output is
/// byte-identical to Showdown's `pack()`:
///
/// - species field empty when `packName(name) == packName(species)`;
/// - item/ability/moves via case-preserving `packName`;
/// - ability is the RAW packName id (sim `pack()` emits no slot letter);
/// - EV field collapses to empty when all six are 0, else `hp,atk,def,spa,spd,spe`
///   with each 0 written as an empty slot;
/// - IV field collapses to empty when all six are 31, else each 31 written empty;
/// - level omitted at 100; happiness omitted at 255; shiny `'S'` else empty.
///
/// `dex` is accepted for signature symmetry with [`unpack`] (Showdown's pack is
/// dex-free); it is currently unused.
pub fn pack(team: &[PokemonSet], _dex: &Dex) -> String {
    let mut buf = String::new();
    for set in team {
        if !buf.is_empty() {
            buf.push(']'); // set separator, prepended before every set after the first
        }
        pack_set(&mut buf, set);
    }
    buf
}

fn pack_set(buf: &mut String, set: &PokemonSet) {
    // 1. name — raw (= species when no real nickname).
    let name = if set.name.is_empty() { &set.species } else { &set.name };
    buf.push_str(name);
    buf.push('|');

    // 2. species — empty when the nickname IS the species. Showdown compares
    //    `packName(name) == packName(species)` (teams.ts:135-136), which is
    //    CASE-PRESERVING (not lowercased). A name/species that are id-equal but
    //    case-different (e.g. name "ho-oh" vs species "Ho-Oh") still emit the
    //    species field — matching `to_id` here would wrongly drop it.
    let id_species = pack_name(&set.species);
    if pack_name(name) != id_species {
        buf.push_str(&id_species);
    }
    buf.push('|');

    // 3. item.
    buf.push_str(&pack_name(&set.item));
    buf.push('|');

    // 4. ability — raw packName id (no slot letter; sim pack() emits the id).
    buf.push_str(&pack_name(&set.ability));
    buf.push('|');

    // 5. moves — packName ids joined by ','.
    let moves: Vec<String> = set.moves.iter().map(|m| pack_name(m)).collect();
    buf.push_str(&moves.join(","));
    buf.push('|');

    // 6. nature — raw.
    buf.push_str(&set.nature);
    buf.push('|');

    // 7. evs — collapse to empty when all 0, else 6-tuple with 0 → "".
    if set.evs.iter().all(|&e| e == 0) {
        // empty field
    } else {
        let evs: Vec<String> = set
            .evs
            .iter()
            .map(|&e| if e == 0 { String::new() } else { e.to_string() })
            .collect();
        buf.push_str(&evs.join(","));
    }
    buf.push('|');

    // 8. gender — raw char else empty.
    if let Some(g) = set.gender {
        buf.push(g);
    }
    buf.push('|');

    // 9. ivs — collapse to empty when all 31, else 6-tuple with 31 → "".
    if set.ivs.iter().all(|&v| v == 31) {
        // empty field
    } else {
        let ivs: Vec<String> = set
            .ivs
            .iter()
            .map(|&v| if v == 31 { String::new() } else { v.to_string() })
            .collect();
        buf.push_str(&ivs.join(","));
    }
    buf.push('|');

    // 10. shiny — 'S' else empty.
    if set.shiny {
        buf.push('S');
    }
    buf.push('|');

    // 11. level — omitted at 100 (and at the falsy 0, matching Showdown's
    //     `if (set.level && set.level !== 100)`, teams.ts:189).
    if set.level != 0 && set.level != 100 {
        buf.push_str(&set.level.to_string());
    }
    buf.push('|');

    // 12. happiness + misc tail.
    if set.happiness != 255 {
        buf.push_str(&set.happiness.to_string());
    }
    // Trailing misc sub-list: emitted only if any of hpType/pokeball/gmax/
    // non-default dmaxLevel/teraType is set. For gen-3 only hpType can appear.
    // When the tail fires, teams.ts:202-208 ALWAYS appends all FIVE comma
    // fields `,hpType,pokeball,gmax,dmaxLevel,teraType` — so with only hpType
    // set the bytes are `,<hpType>,,,,` (the four trailing empties are present,
    // not omitted). We emit exactly that to stay byte-identical.
    if !set.hp_type.is_empty() {
        buf.push(',');
        buf.push_str(&set.hp_type);
        buf.push_str(",,,,"); // pokeball, gmax, dmaxLevel, teraType (all empty in gen-3)
    }
}
