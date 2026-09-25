//! The per-mon SLOT (122 dims) — `PokemonEncoder.encode` + `state_encoder`'s appended tail
//! (`agents/observation/pokemon.py`, `state_encoder.py` §1–2).
//!
//! The Python encoder reads two surfaces per mon, and so does this: the `LivePokemon` view
//! (species / base stats, status, HP, counters, spread, protect streak) and the RAW poke-env
//! `Pokemon` — here the [`PMon`] the reading holds — for the item, type, ability and move
//! sub-encoders and the sleep belief's Early-Bird read.

use super::data::{self, Tables};
use super::layout::*;
use super::{live_get, put, raise, zero, Inputs};
use crate::core_error::{fault, CoreResult, PyExc};
use crate::core_events::Rel;
use crate::present::mon::PMon;
use crate::present::view::MonView;
use crate::trackers::history::Recency;

/// `RecencyTracker.values(side, species)` — (seen, acted, was_hit), each log-saturated.
fn recency(r: &Recency, side: Rel, species: &str) -> [f64; 3] {
    let sat = RECENCY_SAT as i64;
    let mut out = [0.0; 3];
    for (k, d) in [&r.seen, &r.acted, &r.hit].into_iter().enumerate() {
        let n = match d.get(&(side, species)) {
            None => sat,
            Some(last) => (r.turn - last).max(0),
        };
        out[k] = SAT_LUT[if n < sat { n } else { sat } as usize];
    }
    out
}

/// `state_encoder._last_action_tuple(side)` — [move_num, was_switch, hit, miss, fail, crit].
fn last_action(inp: &Inputs, t: &Tables, side: Rel) -> [f64; 6] {
    let i = if side == Rel::Ours { 0 } else { 1 };
    let Some(la) = inp.trackers.pair.last[i].as_ref() else { return [0.0; 6] };
    if la.was_switch {
        return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    }
    let num = match la.move_id.as_deref().filter(|m| !m.is_empty()) {
        Some(m) => t.moves.get(m).map_or(0.0, |md| md.num as f64),
        None => 0.0,
    };
    let (hit, miss, fail) = if la.missed { (0.0, 1.0, 0.0) } else if la.failed { (0.0, 0.0, 1.0) } else { (1.0, 0.0, 0.0) };
    [num, 0.0, hit, miss, fail, if la.crit { 1.0 } else { 0.0 }]
}

/// Both teams' six slots, in the team-list order (ours: `battle.team`; theirs: reveal order).
pub(super) fn team(inp: &Inputs, t: &Tables, own: bool, out: &mut [f32; OBS_DIM]) -> CoreResult<()> {
    let (roster, side_view, rel, base) = if own {
        (&inp.reading.team, &inp.view.ours, Rel::Ours, OFFSET_OUR_TEAM)
    } else {
        (&inp.reading.opp, &inp.view.opp, Rel::Opp, OFFSET_OPP_TEAM)
    };
    let la = last_action(inp, t, rel);
    for i in 0..TEAM_SIZE {
        let start = base + i * POKEMON_FULL_DIM;
        let s = &mut out[start..start + POKEMON_FULL_DIM];
        let Some((_, mon)) = roster.get(i) else {
            zero(s, 0, POKEMON_FULL_DIM);
            continue;
        };
        let live = live_get(side_view, &mon.species)
            .ok_or_else(|| fault(format!("encode: no view mon for {} slot {i} ({})", rel.as_str(), mon.species)))?;
        let rec = recency(&inp.trackers.recency, rel, &mon.species);
        let is_active = if own { mon.active } else { live.active };
        mon_vector(inp, t, mon, live, own, rec, if is_active { Some(&la) } else { None }, &mut s[..POKEMON_VECTOR_DIM])?;
        // the appended tail: our ACTIVE's trapping bits, then the active flag (LAST)
        let (trapped, maybe) = match (own && is_active, inp.legal) {
            (true, Some(l)) => (l.trapped, l.maybe_trapped),
            _ => (false, false),
        };
        put(s, POKEMON_TRAPPED_OFFSET, if trapped { 1.0 } else { 0.0 });
        put(s, POKEMON_MAYBE_TRAPPED_OFFSET, if maybe { 1.0 } else { 0.0 });
        put(s, POKEMON_ACTIVE_OFFSET, if is_active { 1.0 } else { 0.0 });
    }
    Ok(())
}

/// `pokemon._STATUS_STR_IDX.get(status, 0)`.
fn status_idx(status: Option<&str>) -> usize {
    status.and_then(|st| CONDITION_STATUS_IDX.iter().find(|(k, _)| *k == st)).map_or(0, |(_, i)| *i)
}

/// `PokemonEncoder.encode(mon, is_own, hp_probs, hp_known, live_mon, sleep_sources,
/// recency_vals, last_action_vals)` — the 119-dim vector of one populated slot.
#[allow(clippy::too_many_arguments)]
fn mon_vector(inp: &Inputs, t: &Tables, mon: &PMon, live: &MonView, own: bool, rec: [f64; 3],
              last: Option<&[f64; 6]>, v: &mut [f32]) -> CoreResult<()> {
    zero(v, 0, POKEMON_VECTOR_DIM);
    species(t, live, &mut v[POKEMON_SPECIES_OFFSET..POKEMON_SPECIES_OFFSET + 1 + STATS_DIM])?;
    item(t, mon, &mut v[POKEMON_ITEMS_OFFSET..POKEMON_ITEMS_OFFSET + 3])?;
    types(mon, &mut v[POKEMON_TYPES_OFFSET..POKEMON_TYPES_OFFSET + COMBINED_TYPES_DIM]);
    ability(t, mon, &mut v[POKEMON_ABILITIES_OFFSET..POKEMON_ABILITIES_OFFSET + 4])?;
    let sidx = status_idx(live.status);
    if sidx > 0 {
        v[POKEMON_CONDITION_OFFSET + sidx] = 1.0;
    }
    moves(t, mon, &mut v[POKEMON_MOVES_OFFSET..POKEMON_MOVES_OFFSET + 4 * MOVE_SLOT_DIM])?;
    put(v, POKEMON_HP_OFFSET, live.hp_fraction);
    v[POKEMON_SPECIES_KNOWN_OFFSET] = 1.0;
    let is_slp = live.status == Some("slp");
    let is_tox = live.status == Some("tox");
    let ctr = live.status_counter as i64;
    put(v, POKEMON_COUNTER_OFFSET, if is_slp { ctr.min(4) as f64 / 4.0 } else { 0.0 });
    put(v, POKEMON_COUNTER_OFFSET + 1, if is_tox { ctr.min(8) as f64 / 8.0 } else { 0.0 });
    if is_slp {
        let (det, p_wake, reliable) = sleep_belief(inp, mon, own, ctr);
        put(v, POKEMON_SLEEP_BELIEF_OFFSET, det);
        put(v, POKEMON_SLEEP_BELIEF_OFFSET + 1, p_wake);
        put(v, POKEMON_SLEEP_BELIEF_OFFSET + 2, reliable);
    }
    for k in 0..POKEMON_RECENCY_DIM {
        put(v, POKEMON_RECENCY_OFFSET + k, rec[k]);
    }
    if let Some(la) = last {
        for k in 0..POKEMON_LAST_ACTION_DIM {
            put(v, POKEMON_LAST_ACTION_OFFSET + k, la[k]);
        }
    }
    put(v, POKEMON_PROTECT_OFFSET, protect_success_probability(live.protect_counter as i64));
    if own {
        spread(t, live, &mut v[POKEMON_SPREAD_OFFSET..POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]);
        v[POKEMON_HP_REVEALED_OFFSET] = 1.0;
    } else {
        let hp = &inp.trackers.hp;
        let known = hp.ruled_out.contains(&mon.species) || hp.state.contains_key(&mon.species);
        if known {
            v[POKEMON_HP_REVEALED_OFFSET] = 1.0;
            if !hp.ruled_out.contains(&mon.species) {
                if let Some(p) = hp.state.get(&mon.species) {
                    v[POKEMON_HP_PROBS_OFFSET..POKEMON_HP_PROBS_OFFSET + 16].copy_from_slice(p);
                }
            }
        }
    }
    Ok(())
}

/// `SpeciesEncoder.encode(live_mon)` — [num, base stats / 255 ×6].
fn species(t: &Tables, live: &MonView, v: &mut [f32]) -> CoreResult<()> {
    let e = t.species.get(&live.species).ok_or_else(|| {
        raise(PyExc::ValueError, format!("Unrecognized species: {}. Update data/mappings/gen3_mapping.json", live.species))
    })?;
    put(v, 0, e.num);
    let stats = e.base_stats.unwrap_or_else(|| live.base_stats.map(|s| s as f64));
    for k in 0..6 {
        put(v, 1 + k, stats[k] / 255.0);
    }
    Ok(())
}

/// `ItemsEncoder.encode(mon)` — [num, known, consumed] off the RAW `mon.item` / `consumed_item`.
fn item(t: &Tables, mon: &PMon, v: &mut [f32]) -> CoreResult<()> {
    match mon.item.as_deref().filter(|s| !s.is_empty()) {
        Some(it) => {
            let key: String = it.to_lowercase().replace(' ', "").replace('_', "");
            if key == "unknownitem" {
                return Ok(());
            }
            let num = t.items.get(&key).ok_or_else(|| {
                raise(PyExc::ValueError, format!("Unrecognized item: {key}. Update data/pokemon/gen3_items.json"))
            })?;
            put(v, 0, *num);
            v[ITEM_ID_DIM] = 1.0;
        }
        None => {
            if let Some(c) = mon.consumed_item.as_deref().filter(|s| !s.is_empty()) {
                if let Some(num) = t.items.get(&crate::present::dex::to_id(c)) {
                    put(v, 0, *num);
                    v[ITEM_ID_DIM] = 1.0;
                    v[ITEM_ID_DIM + ITEM_KNOWN_DIM] = 1.0;
                }
            }
        }
    }
    Ok(())
}

/// `TypeEncoder.encode(mon)` — `type_1` / `type_2` NAMES, sorted, as `TYPE_TO_IDX` ids.
fn types(mon: &PMon, v: &mut [f32]) {
    let all = mon.types();
    let mut names: Vec<&str> = all.iter().take(2).copied().collect();
    names.sort_unstable();
    for (i, n) in names.iter().enumerate() {
        put(v, i, data::type_idx(n) as f64);
    }
}

/// `AbilitiesEncoder.encode(mon)` — revealed `[num, 0, 1, 1]`, else the Smogon top-2 prior.
fn ability(t: &Tables, mon: &PMon, v: &mut [f32]) -> CoreResult<()> {
    if let Some(a) = mon.ability().filter(|s| !s.is_empty()) {
        let key: String = a.to_lowercase().replace(' ', "").replace('_', "");
        if key != UNKNOWN_ABILITY {
            let num = t.abilities.get(&key).ok_or_else(|| {
                raise(PyExc::ValueError, format!("Unrecognized ability: {key}. Update data/pokemon/gen3_abilities.json"))
            })?;
            put(v, 0, *num);
            v[2] = 1.0;
            v[3] = 1.0;
            return Ok(());
        }
    }
    if !mon.species.is_empty() {
        if let Some((n1, n2, dom)) = t.ability_rank.get(&mon.species) {
            put(v, 0, *n1);
            put(v, 1, *n2);
            put(v, 2, *dom);
        }
    }
    Ok(())
}

/// `moves._category_val` — poke-env's gen-3 `Move.category` (0 status, 1 physical, 2 special).
fn category(mv: &crate::present::mon::PMove) -> CoreResult<f64> {
    let e = mv.entry()?;
    Ok(if !e.damaging {
        0.0
    } else if SPECIAL_TYPES_PRE_SPLIT.contains(&e.typ) {
        2.0
    } else {
        1.0
    })
}

/// `MovesEncoder.encode(mon)` — the moveset sorted by `Move.id`, 4 × 11.
fn moves(t: &Tables, mon: &PMon, v: &mut [f32]) -> CoreResult<()> {
    let mut ms: Vec<crate::present::mon::PMove> = mon.moves.moves().into_iter().map(|(_, m)| m).collect();
    ms.sort_by(|a, b| a.id.cmp(&b.id));
    for (i, mv) in ms.iter().take(4).enumerate() {
        let md = t.moves.get(&mv.id).ok_or_else(|| {
            raise(PyExc::ValueError, format!("Unrecognized move: {}. Update data/pokemon/gen3_moves.json", mv.id))
        })?;
        let (power, type_id) = if mv.id == "hiddenpower" { (70, 0) } else { (md.base_power, md.type_idx) };
        let b = i * MOVE_SLOT_DIM;
        put(v, b, md.num as f64);
        put(v, b + 1, power as f64 / 200.0);
        put(v, b + 2, if md.has_secondary { 1.0 } else { 0.0 });
        put(v, b + 3, if md.has_recoil { 1.0 } else { 0.0 });
        put(v, b + 4, type_id as f64);
        put(v, b + 5, category(mv)?);
        v[b + 6] = 1.0;
        put(v, b + 7, mv.current_pp as f64 / MAX_PP as f64);
        put(v, b + 8, mv.max_pp()? as f64 / MAX_PP as f64);
        put(v, b + 9, md.accuracy as f64 / 100.0);
        put(v, b + 10, if md.never_miss { 1.0 } else { 0.0 });
    }
    Ok(())
}

/// The OWN spread block (18): IVs / 31 (default all-31), EVs / 252, known = 1, nature ×5.
fn spread(t: &Tables, live: &MonView, v: &mut [f32]) {
    let ivs: Vec<i64> = live.ivs.clone().unwrap_or_else(|| vec![31; 6]);
    for (j, iv) in ivs.iter().enumerate() {
        put(v, j, *iv as f64 / 31.0);
    }
    if let Some(evs) = &live.evs {
        for (j, ev) in evs.iter().enumerate() {
            put(v, 6 + j, *ev as f64 / 252.0);
        }
    }
    v[12] = 1.0;
    let name = live.nature.as_deref().unwrap_or("serious");
    let mods = t.natures.get(name);
    for j in 0..5 {
        put(v, 13 + j, mods.and_then(|m| m[j]).unwrap_or(1.0));
    }
}

/// `gen3_mechanics.protect_success_probability(k)` — 1.0, else `1 / min(2**k, 8)`.
fn protect_success_probability(k: i64) -> f64 {
    if k <= 0 {
        return 1.0;
    }
    let p = if k >= 63 { u64::MAX } else { 1u64 << k };
    1.0 / p.min(PROTECT_COUNTER_MAX) as f64
}

/// `sleep_belief.sleep_belief_features(counter, mon, is_own, sleep_sources)` →
/// (sleep_is_deterministic, p_wake, sleep_counter_reliable).
fn sleep_belief(inp: &Inputs, mon: &PMon, own: bool, counter: i64) -> (f64, f64, f64) {
    let t = data::tables();
    let side = if own { Rel::Ours } else { Rel::Opp };
    let (is_rest, usable) = inp.trackers.sleep.sources.get(&(side, mon.species.as_str())).copied().unwrap_or((false, false));
    // `early_bird_probability(mon)`: the revealed ability exactly, else the species' Smogon prior.
    let p_eb = match mon.ability().filter(|a| !a.is_empty() && *a != UNKNOWN_ABILITY) {
        Some(a) => {
            if a == EARLY_BIRD {
                1.0
            } else {
                0.0
            }
        }
        None if mon.species.is_empty() => 0.0,
        None => t.ability_priors.get(&mon.species).and_then(|m| m.get(EARLY_BIRD)).copied().unwrap_or(0.0),
    };
    let k = if counter < 0 { 0 } else if counter as usize <= SLEEP_MAX_K { counter as usize } else { SLEEP_MAX_K };
    let (eb, noeb) = if is_rest { (&SLEEP_REST_EB, &SLEEP_REST_NOEB) } else { (&SLEEP_OPP_EB, &SLEEP_OPP_NOEB) };
    let p_wake = if p_eb <= 0.0 {
        noeb[k]
    } else if p_eb >= 1.0 {
        eb[k]
    } else {
        p_eb * eb[k] + (1.0 - p_eb) * noeb[k]
    };
    (if is_rest { 1.0 } else { 0.0 }, p_wake, if usable { 0.0 } else { 1.0 })
}

#[cfg(test)]
pub(super) fn protect_for_test(k: i64) -> f64 {
    protect_success_probability(k)
}
