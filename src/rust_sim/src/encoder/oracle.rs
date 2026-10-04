//! THE ORACLE REVEAL (`--oracle-reveal`; `designs/endstate/design_x5_belief_tokens.md` §7.6, backlog
//! X32) — a DIAGNOSTIC observation mode, never production. At its `species` level the observation's
//! opponent team block carries every opponent mon's TRUE species from turn 1, as if the game had a
//! team preview (gen 5+): the facts enter the SHARED TRUNK through the observation, not through a
//! side input to any head.
//!
//! **One mechanism, no second encoder.** The reveal is decided in the ENCODER's opponent block and
//! nowhere else (the reading, the view, the trackers, the legality and every other block are the
//! chain's own, untouched):
//!
//! * the opponent slots the viewer has SEEN keep their reveal-order positions and their bytes —
//!   exactly what the mode `off` writes;
//! * after them, one slot per opponent mon NOT yet seen, in dex-num order (the order the belief
//!   labels already give an unseen mon, `belief_labels.assign_hidden_to_slots`), each written by
//!   [`super::hypothesis::hypothesis_slot`] — the SAME slot writer a real first appearance goes
//!   through ("species s present, unrevealed set, full HP, no status"), fed a mon that has never been
//!   on the field;
//! * a mon the play then reveals LEAVES the tail and joins the seen prefix (matched by dex num — a
//!   forme shares its base species' num — one oracle entry per revealed mon, so a duplicate species
//!   is not double-counted); the seen prefix's bytes are `off`'s at every decision.
//!
//! A tail slot is the encoder's row for a mon that has never appeared, so it carries nothing the
//! viewer could not know: the item, the moves, the spread, the status, the Hidden-Power belief, the
//! sleep belief, the last action and the active / trapped flags are ZERO; the recency cells read
//! "never seen" (saturated); the HP fraction is 1.0 (an unseen gen-3 mon is pristine). What it
//! carries is DERIVED FROM THE SPECIES ALONE: the dex num and base stats, the types, and the ability
//! block (the species' Smogon ability prior, or its one ability for a one-ability species).
//! [`SPECIES_SLOT_CELLS`] declares every cell's rule and [`check_species_slot`] THROWS when a slot
//! breaks one — it runs on every slot at construction, so a regression in the shared slot writer
//! that let a hidden fact into a tail slot is a FAULT at the first episode, never a quiet leak.
//!
//! Scripted bots read the view, not the row: they are unaffected. The reveal is per-CHAIN (each
//! side's chain gets the OTHER side's team), so the training pool's two sides — the trainee and a
//! policy opponent served by T2 — are symmetric by construction.

use super::layout::*;
use super::{data, hypothesis};
use crate::core_error::{fault, CoreResult};
use crate::dex::Dex;
use crate::present::dex::to_id;
use crate::present::mon::PMon;

/// How much of the opponent's team the observation is told.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// Nothing: the observation is byte-identical to the build that had no reveal.
    Off,
    /// The six species (team-preview semantics): nothing else.
    Species,
}

impl Level {
    /// Every level, in the order the flag lists them.
    pub const ALL: [Level; 2] = [Level::Off, Level::Species];

    pub fn as_str(self) -> &'static str {
        match self {
            Level::Off => "off",
            Level::Species => "species",
        }
    }

    pub fn parse(s: &str) -> Result<Level, String> {
        Level::ALL.into_iter().find(|l| l.as_str() == s).ok_or_else(|| {
            format!("oracle_reveal {s:?} is not one of {:?}", Level::ALL.map(Level::as_str))
        })
    }
}

/// How one block of a TAIL slot may differ from the pristine "never seen" mon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    /// A function of the species alone (the dex row, the Smogon ability prior): free.
    SpeciesDerived,
    /// Exactly 0.0, bit for bit: the fact is hidden until play reveals it.
    Zero,
    /// Exactly 1.0, bit for bit: the state of a mon that has never been on the field (full HP, the
    /// slot populated, recency "never seen", full Protect odds).
    One,
}

/// One declared block of the 122-cell slot: `(name, lo, hi, rule)`, cells `lo..hi`.
pub type SlotCells = (&'static str, usize, usize, Rule);

const MOVES_END: usize = POKEMON_MOVES_OFFSET + 4 * MOVE_SLOT_DIM;

/// The DECLARED rule of every cell of a `species`-level tail slot (it tiles the slot exactly —
/// `the_declared_cells_tile_the_slot`). Everything the mode reveals is in the three
/// `SpeciesDerived` blocks; every other cell is pristine.
pub const SPECIES_SLOT_CELLS: [SlotCells; 17] = [
    ("species", POKEMON_SPECIES_OFFSET, POKEMON_ITEMS_OFFSET, Rule::SpeciesDerived),
    ("item", POKEMON_ITEMS_OFFSET, POKEMON_TYPES_OFFSET, Rule::Zero),
    ("types", POKEMON_TYPES_OFFSET, POKEMON_ABILITIES_OFFSET, Rule::SpeciesDerived),
    ("ability", POKEMON_ABILITIES_OFFSET, POKEMON_CONDITION_OFFSET, Rule::SpeciesDerived),
    ("status", POKEMON_CONDITION_OFFSET, POKEMON_MOVES_OFFSET, Rule::Zero),
    ("moves", POKEMON_MOVES_OFFSET, MOVES_END, Rule::Zero),
    ("hp_fraction", POKEMON_HP_OFFSET, POKEMON_SPECIES_KNOWN_OFFSET, Rule::One),
    ("species_known", POKEMON_SPECIES_KNOWN_OFFSET, POKEMON_COUNTER_OFFSET, Rule::One),
    ("status_counters", POKEMON_COUNTER_OFFSET, POKEMON_SPREAD_OFFSET, Rule::Zero),
    ("spread", POKEMON_SPREAD_OFFSET, POKEMON_HP_REVEALED_OFFSET, Rule::Zero),
    ("hidden_power", POKEMON_HP_REVEALED_OFFSET, POKEMON_SLEEP_BELIEF_OFFSET, Rule::Zero),
    ("sleep_belief", POKEMON_SLEEP_BELIEF_OFFSET, POKEMON_RECENCY_OFFSET, Rule::Zero),
    ("recency", POKEMON_RECENCY_OFFSET, POKEMON_PROTECT_OFFSET, Rule::One),
    ("protect", POKEMON_PROTECT_OFFSET, POKEMON_LAST_ACTION_OFFSET, Rule::One),
    ("last_action", POKEMON_LAST_ACTION_OFFSET, POKEMON_VECTOR_DIM, Rule::Zero),
    ("trapped_maybe_trapped", POKEMON_TRAPPED_OFFSET, POKEMON_ACTIVE_OFFSET, Rule::Zero),
    ("active", POKEMON_ACTIVE_OFFSET, POKEMON_FULL_DIM, Rule::Zero),
];

/// THROWS unless `slot` obeys [`SPECIES_SLOT_CELLS`] — the producer's guard: a tail slot that
/// carries a hidden fact (a move, an item, a spread, a status, an on-field flag, …) is a FAULT.
/// Bit comparison: `-0.0` is not `0.0`, a NaN equals nothing.
pub fn check_species_slot(species: &str, slot: &[f32]) -> CoreResult<()> {
    if slot.len() != POKEMON_FULL_DIM {
        return Err(fault(format!("oracle: the {species} slot has {} cells, not {POKEMON_FULL_DIM}", slot.len())));
    }
    for (name, lo, hi, rule) in SPECIES_SLOT_CELLS {
        let want = match rule {
            Rule::SpeciesDerived => continue,
            Rule::Zero => 0.0f32,
            Rule::One => 1.0f32,
        };
        if let Some(k) = (lo..hi).find(|&k| slot[k].to_bits() != want.to_bits()) {
            return Err(fault(format!(
                "oracle: the {species} tail slot breaks its declared rule: block `{name}` cell {k} is {:?}, must be exactly {want:?} \
                 — a hidden fact would reach the observation (the `species` level reveals the species only)",
                slot[k]
            )));
        }
    }
    Ok(())
}

/// One opponent mon the oracle knows: its species, its dex num and the slot the encoder writes for
/// it while it is unseen.
#[derive(Debug, Clone)]
pub struct OracleMon {
    /// The species id (`skarmory`).
    pub species: String,
    /// The dex num — a forme shares its base species' num, which is what a revealed mon is matched
    /// on (`gen3_data.species.base_form_ids`; Species Clause is by num).
    pub num: i64,
    slot: [f32; POKEMON_FULL_DIM],
}

impl OracleMon {
    /// The 122 cells of this mon's tail slot.
    pub fn slot(&self) -> &[f32; POKEMON_FULL_DIM] {
        &self.slot
    }
}

/// The opponent's team as the observation is told it — built once per episode and side from the
/// packed team, then read at every decision of that side's chain.
#[derive(Debug, Clone)]
pub struct Oracle {
    level: Level,
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

impl Oracle {
    /// The oracle for `packed` (the OTHER side's packed team) at `level`. `Level::Off` has no
    /// oracle (the chain holds `None`), so it is refused here. Every tail slot is built now and
    /// checked against [`SPECIES_SLOT_CELLS`].
    pub fn new(level: Level, packed: &str, dex: &Dex) -> CoreResult<Oracle> {
        if level == Level::Off {
            return Err(fault("oracle: Level::Off builds no oracle (a chain without one is the `off` mode)"));
        }
        let sets = crate::team::unpack(packed, dex).map_err(|e| fault(format!("oracle: the opponent team does not unpack: {e}")))?;
        if sets.is_empty() || sets.len() > TEAM_SIZE {
            return Err(fault(format!("oracle: the opponent team holds {} mons (1..={TEAM_SIZE} expected)", sets.len())));
        }
        let mut mons = Vec::with_capacity(sets.len());
        for set in &sets {
            // `unpack` keeps the packed team's spelling (`Tyranitar`, `Deoxys-Attack`): the encoder's
            // tables and the reading are keyed by poke-env's id (`tyranitar`, `deoxysattack`)
            let id = to_id(&set.species);
            let slot = hypothesis::hypothesis_slot(&id)?;
            check_species_slot(&id, &slot)?;
            mons.push(OracleMon { num: species_num(&id)?, species: id, slot });
        }
        Ok(Oracle { level, mons })
    }

    pub fn level(&self) -> Level {
        self.level
    }

    /// Every oracle mon, in the team's packed order.
    pub fn mons(&self) -> &[OracleMon] {
        &self.mons
    }

    /// The opponent mons the viewer has NOT yet seen, in the order their slots follow the seen
    /// prefix: dex num ascending (stable: a duplicate species keeps its packed order). One oracle
    /// entry is consumed per revealed mon, matched by dex num; a revealed mon the oracle team does
    /// not hold is a FAULT (the producer's and the labels' shared invariant).
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
        left.sort_by_key(|o| o.num);
        Ok(left)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_declared_cells_tile_the_slot() {
        let mut at = 0;
        for (name, lo, hi, _) in SPECIES_SLOT_CELLS {
            assert_eq!(lo, at, "{name} starts at {lo}, the previous block ended at {at}");
            assert!(hi > lo, "{name} is empty");
            at = hi;
        }
        assert_eq!(at, POKEMON_FULL_DIM);
    }

    #[test]
    fn a_hypothesis_slot_passes_its_guard_and_every_block_has_teeth() {
        let good = hypothesis::hypothesis_slot("skarmory").unwrap();
        check_species_slot("skarmory", &good).unwrap();
        for (name, lo, _, rule) in SPECIES_SLOT_CELLS {
            if rule == Rule::SpeciesDerived {
                continue;
            }
            let mut bad = good;
            bad[lo] = if rule == Rule::Zero { 1.0 } else { 0.5 };
            let e = check_species_slot("skarmory", &bad).unwrap_err().message().to_string();
            assert!(e.contains(&format!("`{name}`")) && e.contains("declared rule"), "{name}: {e}");
            // a signed zero is a difference
            let mut neg = good;
            neg[lo] = if rule == Rule::Zero { -0.0 } else { f32::NAN };
            assert!(check_species_slot("skarmory", &neg).is_err(), "{name}: -0.0 / NaN must fail");
        }
    }

    #[test]
    fn the_level_round_trips_and_an_unknown_one_is_refused() {
        for l in Level::ALL {
            assert_eq!(Level::parse(l.as_str()).unwrap(), l);
        }
        assert!(Level::parse("full").unwrap_err().contains("not one of"));
    }
}
