//! Family `belief` — `belief_species`, `belief_moves`, `known_moves` (`Gen3Env._belief_labels`,
//! `agents/observation/belief_labels.py`), rule for rule.
//!
//! * TRUTH: the OTHER side's own reading of its team (`battle2.team`): each mon's species and its
//!   moves as poke-env's `Pokemon.moves` returns them (`MoveSet::moves_ref` — transform- and
//!   Mimic-resolved, `Move._id`, a Hidden Power TYPED).
//! * READING: `species_known` read from the side's OWN row (the cells `BeliefSlots` keys on), and
//!   the revealed mons in encoder slot order (`reading.opp`, the list the encoder packs).
//! * `assign_hidden_to_slots`: hidden = truth team minus the revealed species (a species with no
//!   num skipped), STABLE-sorted by species num; the j-th fills the j-th believed slot
//!   (`species_known < 0.5`); the rest PAD (-1). Moves: up to 4 nums in set order, an id with no
//!   num skipped.
//! * `known_moves`: each revealed slot (the leading-contiguous block, in order) gets its species'
//!   FULL truth moveset (the first truth mon of that species).
//! * A revealed block that is NOT leading-contiguous is a FAULT (the Python env raises
//!   `RuntimeError` there: crash over mis-slotted supervision).

use pokesim::encoder::data::tables;
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

fn species_num(id: &str) -> Option<i64> {
    tables().species.get(id).map(|r| r.num as i64)
}

fn move_num(id: &str) -> Option<i64> {
    tables().moves.get(id).map(|r| r.num)
}

/// Up to `MOVE_SLOTS` move nums of one truth mon, in set order, unknown ids skipped.
fn move_nums(mon: &pokesim::present::mon::PMon, out: &mut [i64]) {
    let mut m = 0;
    for (_, mv) in mon.moves.moves_ref() {
        if m >= MOVE_SLOTS {
            break;
        }
        if let Some(n) = move_num(&to_id(&mv.id)) {
            out[m] = n;
            m += 1;
        }
    }
}

/// Write the side's three columns (`bs` 6, `bm` 6×4, `km` 6×4). `Err` = the revealed block is not
/// leading-contiguous (the caller turns it into a FAULT).
pub fn write(own: &BoardReading, truth: &BoardReading, row: &[f32], bs: &mut [i64], bm: &mut [i64], km: &mut [i64]) -> Result<(), String> {
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
    let revealed: Vec<String> = revealed_species(own).into_iter().map(to_id).collect();
    // ---- belief_species / belief_moves (the unknown slots)
    let mut hidden: Vec<(i64, usize)> = Vec::with_capacity(TEAM_SIZE);
    for (idx, (_, m)) in truth.team.iter().enumerate() {
        let sp = to_id(&m.species);
        if revealed.contains(&sp) {
            continue;
        }
        if let Some(num) = species_num(&sp) {
            hidden.push((num, idx));
        }
    }
    hidden.sort_by_key(|t| t.0); // stable, as Python's sort
    let believed = (0..TEAM_SIZE).filter(|&i| known[i] < 0.5);
    for (j, slot) in believed.enumerate() {
        let Some(&(num, idx)) = hidden.get(j) else { break };
        bs[slot] = num;
        move_nums(&truth.team[idx].1, &mut bm[slot * MOVE_SLOTS..(slot + 1) * MOVE_SLOTS]);
    }
    // ---- known_moves (the revealed slots): the species' first truth mon
    let revealed_slots = (0..TEAM_SIZE).filter(|&i| known[i] >= 0.5);
    for (slot, sp) in revealed_slots.zip(revealed.iter()) {
        if let Some((_, m)) = truth.team.iter().find(|(_, m)| to_id(&m.species) == *sp) {
            move_nums(m, &mut km[slot * MOVE_SLOTS..(slot + 1) * MOVE_SLOTS]);
        }
    }
    Ok(())
}
