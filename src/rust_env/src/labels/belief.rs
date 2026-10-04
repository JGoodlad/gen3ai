//! Family `belief` — `belief_species`, `belief_moves`, `known_moves` (`Gen3Env._belief_labels`,
//! `agents/observation/belief_labels.py`), rule for rule.
//!
//! * TRUTH: the OTHER side's own reading of its team (`battle2.team`): each mon's species and its
//!   moves as poke-env's `Pokemon.moves` returns them (`MoveSet::moves_ref` — transform- and
//!   Mimic-resolved, `Move._id`, a Hidden Power TYPED).
//! * READING: `species_known` read from the side's OWN row (the cells `BeliefSlots` keys on), and
//!   the revealed mons in encoder slot order (`reading.opp`, the list the encoder packs).
//! * `assign_hidden_to_slots`: hidden = truth team minus the revealed species, STABLE-sorted by
//!   species num; the j-th fills the j-th believed slot (`species_known < 0.5`); the rest PAD (-1).
//!   Moves: up to 4 nums in set order.
//! * `known_moves`: each revealed slot (the leading-contiguous block, in order) gets its species'
//!   FULL truth moveset (the first truth mon of that species).
//! * A revealed species is matched to its truth mon by DEX NUM, never by id: a forme SHARES its
//!   base species' num (`gen3_data.species.base_form_ids`; `gen3_species.json` carries every gen-3
//!   forme under its base num, and Species Clause is by num), so a forme the two readings spell
//!   differently still matches.
//!
//! 🚨 **Every lookup THROWS** (F-X5-3, `gen3_label_lookup_guard_v1`). A species or move with no dex
//! num, a revealed species absent from the truth team, more hidden truth mons than believed slots,
//! and a revealed block that is NOT leading-contiguous are each an `Err` — a FAULT, which fails the
//! batch and poisons the pool (`core::refusal`). The writer used to SKIP the first three in
//! silence, which would undercount a label (and X5's OTHER label) with nothing to show for it.
//! Measured dormant before the guard (2026-10-03): 0 skips over 4,001 pool + corpus episodes
//! (572,451 decisions) and 30,000 ladder-corpus episodes (5,253,432 decisions).

use pokesim::encoder::data::tables;
use pokesim::encoder::oracle::Oracle;
use pokesim::encoder::layout::{OFFSET_OPP_TEAM, POKEMON_FULL_DIM, POKEMON_SPECIES_KNOWN_OFFSET, TEAM_SIZE};
use pokesim::present::board_reading::BoardReading;
use pokesim::present::dex::to_id;

pub const MOVE_SLOTS: usize = 4;
const PAD: i64 = -1;

/// `species_known` of the six opp slots, from the side's own row.
pub fn species_known(row: &[f32]) -> [f32; TEAM_SIZE] {
    std::array::from_fn(|i| row[OFFSET_OPP_TEAM + i * POKEMON_FULL_DIM + POKEMON_SPECIES_KNOWN_OFFSET])
}

/// The revealed opp species in encoder slot order (`get_team_list(b1, is_opponent=True)`).
pub fn revealed_species(own: &BoardReading) -> Vec<&str> {
    own.opp.iter().map(|(_, m)| m.species.as_str()).collect()
}

/// The dex num of a species id — an `Err` when `gen3_species.json` has no row for it, or a row
/// whose num is not positive (the table reads a missing `num` as 0). Never a skip (F-X5-3).
pub fn species_num(id: &str) -> Result<i64, String> {
    match tables().species.get(id) {
        Some(r) if r.num >= 1.0 => Ok(r.num as i64),
        Some(r) => Err(format!("label: species {id:?} has dex num {} in gen3_species.json (not a valid num)", r.num)),
        None => Err(format!(
            "label: species {id:?} has no row in gen3_species.json — a label lookup never skips a species (F-X5-3)"
        )),
    }
}

/// The dex num of a move id — an `Err` when `gen3_moves.json` has no row for it, or a non-positive
/// num. Never a skip (F-X5-3).
pub fn move_num(id: &str) -> Result<i64, String> {
    match tables().moves.get(id) {
        Some(r) if r.num >= 1 => Ok(r.num),
        Some(r) => Err(format!("label: move {id:?} has dex num {} in gen3_moves.json (not a valid num)", r.num)),
        None => Err(format!("label: move {id:?} has no row in gen3_moves.json — a label lookup never skips a move (F-X5-3)")),
    }
}

/// The truth-team index of the FIRST mon whose species has dex num `num` (a forme shares its base
/// species' num), or an `Err` naming the truth team: a species the OTHER side does not field is a
/// broken reading, never a slot left PAD in silence.
pub fn truth_index(truth: &BoardReading, num: i64, what: &str) -> Result<usize, String> {
    for (i, (_, m)) in truth.team.iter().enumerate() {
        if species_num(&to_id(&m.species))? == num {
            return Ok(i);
        }
    }
    let team: Vec<String> = truth.team.iter().map(|(_, m)| to_id(&m.species)).collect();
    Err(format!("label: {what} (dex num {num}) is not on the truth team {team:?}"))
}

/// The dex nums of every opp slot whose species the OBSERVATION states (`species_known`), in encoder
/// slot order, each checked to be on the truth team: the mons the side has SEEN (reveal order), then —
/// under the ORACLE REVEAL (`pokesim::encoder::oracle`) — the unseen ones the encoder appends after them
/// (`Oracle::tail`, the SAME function the encoder calls, so the label's slot order cannot drift from the
/// row's). `oracle = None` (the mode `off`) is the seen mons alone.
pub fn revealed_nums(own: &BoardReading, truth: &BoardReading, oracle: Option<&Oracle>) -> Result<Vec<i64>, String> {
    let mut out = Vec::with_capacity(TEAM_SIZE);
    for sp in revealed_species(own) {
        let num = species_num(&to_id(sp))?;
        truth_index(truth, num, &format!("revealed species {sp:?}"))?;
        out.push(num);
    }
    if let Some(o) = oracle {
        for unseen in o.tail(&own.opp).map_err(|e| e.message().to_string())? {
            truth_index(truth, unseen.num, &format!("oracle species {:?}", unseen.species))?;
            out.push(unseen.num);
        }
    }
    Ok(out)
}

/// Up to `MOVE_SLOTS` move nums of one truth mon, in set order; a move with no num is an `Err`.
fn move_nums(mon: &pokesim::present::mon::PMon, out: &mut [i64]) -> Result<(), String> {
    for (m, (_, mv)) in mon.moves.moves_ref().into_iter().take(MOVE_SLOTS).enumerate() {
        out[m] = move_num(&to_id(&mv.id)).map_err(|e| format!("{e} (truth mon {:?})", mon.species))?;
    }
    Ok(())
}

/// Write the side's three columns (`bs` 6, `bm` 6×4, `km` 6×4). `Err` = a broken label invariant
/// (module docs; the caller turns it into a FAULT).
pub fn write(own: &BoardReading, truth: &BoardReading, oracle: Option<&Oracle>, row: &[f32], bs: &mut [i64], bm: &mut [i64], km: &mut [i64]) -> Result<(), String> {
    bs.fill(PAD);
    bm.fill(PAD);
    km.fill(PAD);
    let known = species_known(row);
    let n_known = known.iter().filter(|&&s| s >= 0.5).count();
    if known[n_known..].iter().any(|&s| s >= 0.5) {
        return Err(format!(
            "belief labels: opp species_known {known:?} is not leading-contiguous — the encoder's revealed-first opp-slot \
             packing changed, breaking the believed-slot alignment with the model's BeliefSlots"
        ));
    }
    let revealed = revealed_nums(own, truth, oracle)?;
    // ---- belief_species / belief_moves (the unknown slots)
    let mut hidden: Vec<(i64, usize)> = Vec::with_capacity(TEAM_SIZE);
    for (idx, (_, m)) in truth.team.iter().enumerate() {
        let num = species_num(&to_id(&m.species))?;
        if !revealed.contains(&num) {
            hidden.push((num, idx));
        }
    }
    hidden.sort_by_key(|t| t.0); // stable, as Python's sort
    let believed: Vec<usize> = (0..TEAM_SIZE).filter(|&i| known[i] < 0.5).collect();
    if hidden.len() > believed.len() {
        return Err(format!(
            "belief labels: {} hidden truth mons but only {} believed slots — a hidden mon would go unlabelled",
            hidden.len(),
            believed.len()
        ));
    }
    // the consumer's twin of the encoder's producer guard: the species the row states and the species
    // the label walks are the same list (a drift between the two would shift every label by a slot)
    if revealed.len() != n_known {
        return Err(format!(
            "belief labels: the row states {n_known} opp species but the reading + oracle give {} — the encoder's opp-slot \
             packing and the label's slot order drifted",
            revealed.len()
        ));
    }
    for (&slot, &(num, idx)) in believed.iter().zip(&hidden) {
        bs[slot] = num;
        move_nums(&truth.team[idx].1, &mut bm[slot * MOVE_SLOTS..(slot + 1) * MOVE_SLOTS])?;
    }
    // ---- known_moves (the revealed slots): the species' first truth mon
    let revealed_slots = (0..TEAM_SIZE).filter(|&i| known[i] >= 0.5);
    for (slot, &num) in revealed_slots.zip(revealed.iter()) {
        let idx = truth_index(truth, num, "a revealed species")?;
        move_nums(&truth.team[idx].1, &mut km[slot * MOVE_SLOTS..(slot + 1) * MOVE_SLOTS])?;
    }
    Ok(())
}
