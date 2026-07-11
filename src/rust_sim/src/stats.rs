//! Stats — compute a Pokémon's in-battle stats from a [`PokemonSet`] + the
//! [`Dex`], EXACTLY as Showdown's Gen 3 does.
//!
//! **The formula** is Showdown's `Dex.statModify` (`sim/battle.ts`, compiled
//! `dist/sim/battle.js:1956-1972`), driven from `Pokemon.setSpecies`
//! (`pokemon.js:989` → `spreadModify` → `statModify`). Stat order everywhere is
//! `[hp, atk, def, spa, spd, spe]` (matching [`crate::team::STAT_ORDER`]).
//!
//! ```text
//! tr = trunc = floor for the non-negative values here (dex.js:343, `num >>> 0`).
//!
//! HP   : hp = tr( tr(2*base_hp + iv_hp + tr(ev_hp/4) + 100) * level/100 + 10 )
//!        (HP never gets a nature multiplier.)
//!
//! OTHER: stat = tr( tr(2*base + iv + tr(ev/4)) * level/100 + 5 )
//!        then, AFTER the +5:
//!          if nature BOOSTS this stat  (×1.1): stat = tr( tr(stat*110, 16) / 100 )
//!          if nature HINDERS this stat (×0.9): stat = tr( tr(stat*90,  16) / 100 )
//!          neutral nature: unchanged.
//! ```
//!
//! **Integer nature math (not f64).** Showdown applies the nature as an integer
//! modifier — `tr(tr(stat*110, 16) / 100)`, i.e. `floor(stat * 110 / 100)`. We
//! reproduce that exact integer form for source-fidelity: it's the operation the
//! sim actually runs. (For legal gen-3 magnitudes the `bits=16` mask is a no-op
//! and `floor(stat*110/100)` happens to equal `floor(stat*1.1)` for every value,
//! so this isn't fixing a visible f64 mis-round — it's matching the source op,
//! and it stays correct in the general `tr(stat*N, 16)` form if a stat ever exceeds 595.)
//!
//! **No `overflowstatmod`.** The `Min(stat, 595/728)` clamps in `statModify`
//! (`battle.js:1965/1969`) are gated on the `overflowstatmod` ruleset, which is
//! hackmons-only and absent from `gen3ou`/`gen3customgame`. We omit them.
//!
//! **Shedinja hook.** `setSpecies` overrides `stats.hp` with `species.maxHP`
//! when set (`pokemon.js:990`). Showdown's `pokedex.ts` gives Shedinja `maxHP: 1`,
//! so its HP is a FIXED 1 — NOT the base-stat formula's output (which would yield
//! ~206 at level 100). The base-HP-1 formula does NOT reproduce this. We mirror
//! the hook exactly: [`SpeciesData::max_hp`] carries the optional `maxHP` (parsed
//! from `gen3_species.json`; `Some(1)` for Shedinja, `None` otherwise) and this
//! function returns it verbatim for HP when present, generation-generically.
//!
//! Validated against the sim's OWN computed stats (`a.storedStats` + `a.maxhp`,
//! the truth `damage_probe.js` reads) via `harness/gen_stats_golden.js` /
//! `tests/stats_test.rs`.

use crate::dex::Dex;
use crate::team::PokemonSet;

/// In-battle stats in `[hp, atk, def, spa, spd, spe]` order (= [`crate::team::STAT_ORDER`]).
pub type StatTable = [u16; 6];

/// `trunc(num, 0)` from `dex.js:343` — unsigned-32-bit floor. For the
/// non-negative `u32` intermediates in this calc it is a plain identity/floor;
/// kept as a named function so the formula reads like the source.
#[inline]
fn tr(num: u32) -> u32 {
    num
}

/// Apply a nature multiplier to one already-computed stat, the EXACT integer way
/// Showdown does (`battle.js:1963-1970`): a boosting nature is `floor(stat*110/100)`,
/// a hindering nature is `floor(stat*90/100)`. `mult` is the per-stat float from
/// the dex (1.1 boost / 0.9 hinder / 1.0 neutral); we branch on it but compute in
/// integers so the rounding matches the sim bit-for-bit (never `stat as f64 * mult`).
///
/// The inner `tr(stat*N, 16)` masks to `(value >>> 0) % 2^16`; for legal gen-3
/// stat magnitudes (`stat*110` is far under 2^16·… — actually well under 2^32 and
/// the post-`/100` result under 2^16) the mask is a no-op, so `floor(stat*N/100)`
/// is exact.
#[inline]
fn apply_nature(stat: u32, mult: f64) -> u32 {
    // Use a tolerance so 1.1/0.9 floats compare cleanly; the dex stores exactly
    // these three values (1.0 / 1.1 / 0.9).
    if mult > 1.0 + 1e-9 {
        // boosting nature: tr( tr(stat*110, 16) / 100 )
        let masked = (stat * 110) % (1 << 16);
        tr(masked / 100)
    } else if mult < 1.0 - 1e-9 {
        // hindering nature: tr( tr(stat*90, 16) / 100 )
        let masked = (stat * 90) % (1 << 16);
        tr(masked / 100)
    } else {
        stat
    }
}

/// Compute the six in-battle stats for `set` using `dex`, the EXACT Gen-3 way.
///
/// Looks up the species base stats and the nature's per-stat multipliers, applies
/// the floor-placement-exact formula (EV/4 floored before the inner sum; the
/// outer `*level/100` floored; nature applied AFTER the `+5`, integer-only, and
/// NEVER to HP), and returns `[hp, atk, def, spa, spd, spe]`.
///
/// **Crash-don't-drop:** returns `Err` (never a silent wrong value) when the
/// species or nature is unknown to the dex — a missing input would corrupt every
/// downstream damage number.
pub fn compute_stats(set: &PokemonSet, dex: &Dex) -> Result<StatTable, String> {
    let species = dex
        .species(&set.species)
        .ok_or_else(|| format!("compute_stats: unknown species {:?}", set.species))?;
    // An EMPTY nature field is treated as NEUTRAL (all multipliers 1.0) — the sim's
    // behavior: a packed-team set can OMIT the nature (`Suicune||Item|Ability|moves||EVs`),
    // and Showdown's `natures.get("")` returns a NONEXISTENT nature with NO plus/minus, so
    // `spreadModify` boosts/hinders no stat — bit-identical to Serious (VERIFIED vs the sim:
    // an empty-nature Suicune's `storedStats` == its Serious-nature stats). A real gen3ou
    // sample team (Suicune in e2e_8/e2e_73, admitted by the STATUS_IMMUNE regen) carries this.
    // A NON-empty UNKNOWN nature is still a hard error (a genuine typo would corrupt stats).
    let neutral;
    let nature = if crate::dex::to_id(&set.nature).is_empty() {
        neutral = crate::dex::NatureData {
            name: String::new(),
            atk: 1.0,
            def: 1.0,
            spa: 1.0,
            spd: 1.0,
            spe: 1.0,
        };
        &neutral
    } else {
        dex.nature(&set.nature)
            .ok_or_else(|| format!("compute_stats: unknown nature {:?}", set.nature))?
    };

    let base = &species.base_stats;
    let level = set.level as u32;

    // Per-stat arrays in [hp, atk, def, spa, spd, spe] order so we can index uniformly.
    let base_arr: [u32; 6] = [
        base.hp as u32,
        base.atk as u32,
        base.def as u32,
        base.spa as u32,
        base.spd as u32,
        base.spe as u32,
    ];
    let iv_arr: [u32; 6] = [
        set.ivs[0] as u32,
        set.ivs[1] as u32,
        set.ivs[2] as u32,
        set.ivs[3] as u32,
        set.ivs[4] as u32,
        set.ivs[5] as u32,
    ];
    let ev_arr: [u32; 6] = [
        set.evs[0] as u32,
        set.evs[1] as u32,
        set.evs[2] as u32,
        set.evs[3] as u32,
        set.evs[4] as u32,
        set.evs[5] as u32,
    ];
    // Nature multipliers for atk/def/spa/spd/spe (indices 1..6); HP (index 0) never modified.
    let nature_mult: [f64; 6] = [1.0, nature.atk, nature.def, nature.spa, nature.spd, nature.spe];

    let mut out: StatTable = [0; 6];

    // HP (index 0): tr( tr(2*base + iv + tr(ev/4) + 100) * level/100 + 10 ),
    // UNLESS the species carries a fixed `maxHP` override (Showdown's
    // `setSpecies`, `pokemon.js:990` — Shedinja's `maxHP: 1`), which replaces the
    // computed HP entirely. We mirror that hook for bit-for-bit parity.
    out[0] = match species.max_hp {
        Some(fixed) => fixed,
        None => {
            let inner_hp = tr(2 * base_arr[0] + iv_arr[0] + tr(ev_arr[0] / 4) + 100);
            tr(inner_hp * level / 100 + 10) as u16
        }
    };

    // OTHER stats (atk/def/spa/spd/spe).
    for i in 1..6 {
        let inner = tr(2 * base_arr[i] + iv_arr[i] + tr(ev_arr[i] / 4));
        let stat = tr(inner * level / 100 + 5);
        out[i] = apply_nature(stat, nature_mult[i]) as u16;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::compute_stats;
    use crate::dex::Dex;
    use crate::team::unpack;

    /// An EMPTY nature field computes the NEUTRAL (Serious) stats — the sim's behavior for a
    /// packed set that OMITS the nature (`Suicune||Item|Ability|moves||EVs`). VERIFIED vs the
    /// omniscient sim (`node`-probed: an empty-nature Suicune's storedStats == its Serious
    /// stats), the real-team gap the STATUS_IMMUNE e2e regen surfaced (Suicune in e2e_8/e2e_73).
    /// WRONG (pre-fix): `compute_stats` PANICKED `unknown nature ""`. A NON-empty unknown nature
    /// is still a hard error.
    #[test]
    fn empty_nature_computes_the_neutral_stats() {
        let d = Dex::for_gen(3);
        // The e2e_8 Suicune (empty nature between the moves and the EVs), 252 HP EVs, IV31.
        let empty = "Suicune||Leftovers|Pressure|Rest,CalmMind,Surf,Roar||252,,,,,|N|,0,,,,|||";
        let serious = "Suicune||Leftovers|Pressure|Rest,CalmMind,Surf,Roar|Serious|252,,,,,|N|,0,,,,|||";
        let s_empty = unpack(empty, &d).expect("unpack empty-nature");
        let s_serious = unpack(serious, &d).expect("unpack serious");
        let st_empty = compute_stats(&s_empty[0], &d).expect("empty nature must NOT panic (= neutral)");
        let st_serious = compute_stats(&s_serious[0], &d).expect("serious");
        assert_eq!(
            st_empty, st_serious,
            "an EMPTY nature must compute the NEUTRAL (Serious) stats bit-for-bit (the sim's behavior)"
        );
        // The exact sim-probed values (a guard against a wrong neutral fallback).
        assert_eq!(st_empty, [404, 155, 266, 216, 266, 206], "the sim's storedStats for the empty-nature Suicune");
    }

    /// A NON-empty UNKNOWN nature is still a hard error (crash-don't-drop — a genuine typo would
    /// corrupt every downstream stat).
    #[test]
    fn a_nonempty_unknown_nature_still_errors() {
        let d = Dex::for_gen(3);
        let typo = "Suicune||Leftovers|Pressure|Rest,CalmMind,Surf,Roar|Notanature|252,,,,,|||||";
        let s = unpack(typo, &d).expect("unpack");
        assert!(
            compute_stats(&s[0], &d).is_err(),
            "a non-empty UNKNOWN nature must still error (only an EMPTY nature is the neutral fallback)"
        );
    }
}
