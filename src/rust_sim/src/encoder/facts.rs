//! The OBS-FACTS block (`gen3_obs_facts_v1`) — `agents/observation/obs_facts.py`'s
//! `encode_obs_facts`, cell for cell, in the same f64 arithmetic rounded once at the write.
//!
//! The row's LAST block (`OFFSET_OBS_FACTS`, written by `encode_into` as step 8; appended at the X5
//! version break, config v144, part 3): slice O holds the whole row byte-equal to Python, the block
//! with it. [`crate::version::BattleVersion::encode_facts`] computes the block alone (the engine-truth
//! test reads it).
//!
//! SEEN (what the opponent has seen of our team), CHOICE (the opponent active's Choice-lock
//! evidence), VOL (each active's Encore / Taunt / Disable / Uproar / partial-trap turns) and SCREENS
//! (each side's Reflect / Light Screen / Safeguard / Mist turns left). The mechanics and their
//! `deps/pokemon-showdown` sources are stated once, in `obs_facts.py` and `constants.FACTS_*`.

use super::data::Tables;
use super::layout::*;
use super::{live_get, put, zero, Inputs};
use crate::core_events::Rel;
use crate::present::view::MonView;
use crate::trackers::facts::FactsFold;

/// `obs_facts._vol_counter`: the reading's counter for a duration volatile (the partial trap: the
/// max over every trap id present).
fn vol_counter(m: &MonView, key: &str) -> Option<i64> {
    let get = |k: &str| m.volatiles.iter().find(|(id, _)| id == k).map(|(_, c)| *c as i64);
    if key != "partiallytrapped" {
        return get(key);
    }
    let mut best: Option<i64> = None;
    for k in FACTS_PARTIAL_TRAP_IDS {
        if let Some(c) = get(k) {
            if best.is_none_or(|b| c > b) {
                best = Some(c);
            }
        }
    }
    best
}

/// `obs_facts._write_vol`.
fn write_vol(out: &mut [f32], base: usize, mon: Option<&MonView>, side: Rel, fold: &FactsFold, residual: i64) {
    let Some(m) = mon else { return };
    if m.fainted {
        return;
    }
    for (j, key) in FACTS_VOL_EFFECTS.iter().enumerate() {
        let Some(c) = vol_counter(m, key) else { continue };
        let (lo, hi, adjustable) = FACTS_VOL_DURATION[j];
        let (mut adj_lo, mut adj_hi) = (0i64, 0i64);
        if adjustable {
            match fold.adjusted(side, key) {
                None => adj_hi = 1,
                Some(true) => {
                    adj_lo = 1;
                    adj_hi = 1;
                }
                Some(false) => {}
            }
        }
        let e = c + residual;
        let min_left = (lo + adj_lo - e).max(1);
        let max_left = (hi + adj_hi - e).max(min_left);
        let o = base + j * FACTS_VOL_CELL_DIM;
        // `min(e, FACTS_TURN_NORM) / FACTS_TURN_NORM` — an int vs 8.0, then a float divide
        put(out, o, (e as f64).min(FACTS_TURN_NORM) / FACTS_TURN_NORM);
        put(out, o + 1, min_left as f64 / FACTS_TURN_NORM);
        put(out, o + 2, max_left as f64 / FACTS_TURN_NORM);
    }
}

/// `encode_obs_facts(vec, off, live, our_species, event_window)` into `out` (the block's
/// `OBS_FACTS_DIM` cells; every cell written).
pub fn obs_facts(inp: &Inputs, t: &Tables, out: &mut [f32]) {
    zero(out, 0, OBS_FACTS_DIM);
    let v = inp.view;
    let fold = &inp.trackers.window.facts;
    let residual: i64 = if v.residual_done { 1 } else { 0 };

    // 1. SEEN — row i is our team slot i
    for i in 0..TEAM_SIZE {
        let Some((_, pm)) = inp.reading.team.get(i) else { continue };
        let Some(m) = live_get(&v.ours, &pm.species) else { continue };
        let o = FACTS_SEEN_OFFSET + i * FACTS_SEEN_ROW_DIM;
        put(out, o, if m.revealed { 1.0 } else { 0.0 });
        for (k, mv) in m.moves.iter().take(4).enumerate() {
            put(out, o + 1 + k, if mv.seen { 1.0 } else { 0.0 });
        }
        put(out, o + 5, if m.item_public { 1.0 } else { 0.0 });
        put(out, o + 6, if m.ability_public { 1.0 } else { 0.0 });
    }

    // 2. CHOICE — the opponent active
    if let Some(oa) = v.opp.active.map(|i| &v.opp.mons[i]).filter(|m| !m.fainted) {
        let o = FACTS_CHOICE_OFFSET;
        put(out, o, if oa.item_public && oa.item.as_deref() != Some("choiceband") { 1.0 } else { 0.0 });
        let (first, distinct2, run) = fold.stint(Rel::Opp);
        put(out, o + 1, if distinct2 { 1.0 } else { 0.0 });
        let num = first.filter(|m| !m.is_empty()).and_then(|m| t.moves.get(m)).map_or(0.0, |m| m.num as f64);
        put(out, o + 2, num);
        let cap = SAT_LUT.len() - 1;
        put(out, o + 3, SAT_LUT[(run as usize).min(cap)]);
    }

    // 3. VOL — each side's active
    let ours = v.ours.active.map(|i| &v.ours.mons[i]);
    let opp = v.opp.active.map(|i| &v.opp.mons[i]);
    write_vol(out, FACTS_VOL_OFFSET, ours, Rel::Ours, fold, residual);
    write_vol(out, FACTS_VOL_OFFSET + FACTS_VOL_SIDE_DIM, opp, Rel::Opp, fold, residual);

    // 4. SCREENS — side-major
    let turn = v.turn as i64;
    for (s, side) in [&v.ours, &v.opp].into_iter().enumerate() {
        for (j, name) in FACTS_SCREENS.iter().enumerate() {
            let Some(start) = side.side_conditions.iter().find(|(k, _)| k == name).map(|(_, x)| *x as i64) else {
                continue;
            };
            let left = (FACTS_SCREEN_TURNS as i64 - ((turn - start) + residual)).max(0);
            put(out, FACTS_SCREENS_OFFSET + s * FACTS_SCREENS.len() + j, left as f64 / FACTS_SCREEN_TURNS as f64);
        }
    }
}
