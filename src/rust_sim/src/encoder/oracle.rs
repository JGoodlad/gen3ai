//! THE ORACLE REVEAL (`--oracle-reveal`; `designs/endstate/design_x5_belief_tokens.md` §7.6, backlog
//! X32) — a DIAGNOSTIC observation mode, never production. At its `species` level the observation's
//! opponent team block carries every opponent mon's TRUE species from turn 1, as if the game had a
//! team preview (gen 5+): the facts enter the SHARED TRUNK through the observation, not through a
//! side input to any head. At its `full` level it carries the opponent's whole SET as already-known
//! facts: the four moves, the item, the ability and the spread (nature, EVs, IVs; `spread_known` 1).
//!
//! **One mechanism, no second encoder.** The reveal is decided in the ENCODER's opponent block and
//! nowhere else (the reading, the view, the trackers, the legality and every other block are the
//! chain's own, untouched):
//!
//! * the opponent slots the viewer has SEEN keep their reveal-order positions. At `species` their
//!   bytes are exactly what the mode `off` writes; at `full` they gain only the facts play has NOT
//!   revealed yet ([`Oracle::overlay`]) — an observed move keeps its tracked PP, a consumed or removed
//!   item stays consumed or removed, a revealed ability is the reading's: no stale preview value
//!   fights a live one;
//! * after them, one slot per opponent mon NOT yet seen, in dex-num order (the order the belief
//!   labels already give an unseen mon, `belief_labels.assign_hidden_to_slots`). At `species` each is
//!   written by [`super::hypothesis::hypothesis_slot`] — the SAME slot writer a real first appearance
//!   goes through ("species s present, unrevealed set, full HP, no status"), fed a mon that has never
//!   been on the field. At `full` it is the row the encoder writes for an OWN mon of that set at full
//!   HP ([`full_slot`]: the same `slot::populated_slot`, the spread block written because the mon's view
//!   says `spread_known`);
//! * a mon that play then reveals LEAVES the tail and joins the seen prefix (matched by dex num — a
//!   forme shares its base species' num — one oracle entry per revealed mon, so a duplicate species
//!   is not double-counted).
//!
//! A tail slot is the encoder's row for a mon that has never appeared, so it carries nothing the
//! viewer could not be TOLD at its level: at `species` the item, the moves, the spread, the status, the
//! Hidden-Power belief, the sleep belief, the last action and the active / trapped flags are ZERO; the
//! recency cells read "never seen" (saturated); the HP fraction is 1.0 (an unseen gen-3 mon is
//! pristine); what it carries is DERIVED FROM THE SPECIES ALONE (the dex num and base stats, the types,
//! the ability block's Smogon prior). At `full` the set's facts are written too and everything else is
//! unchanged. [`SPECIES_SLOT_CELLS`] / [`FULL_SLOT_CELLS`] declare every cell's rule and [`check_slot`]
//! THROWS when a slot breaks one — it runs on every slot at construction, so a regression in the shared
//! slot writer that let a hidden fact into a tail slot is a FAULT at the first episode, never a quiet
//! leak.
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

/// How much of the opponent's team the observation is told.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// Nothing: the observation is byte-identical to the build that had no reveal.
    Off,
    /// The six species (team-preview semantics): nothing else.
    Species,
    /// The whole set as already-known facts: species, four moves, item, ability, nature, EVs and IVs (the spread
    /// block with `spread_known` 1), and `hp_revealed` 1 (the Hidden-Power type is determined by the moves).
    Full,
}

impl Level {
    /// Every level, in the order the flag lists them.
    pub const ALL: [Level; 3] = [Level::Off, Level::Species, Level::Full];

    pub fn as_str(self) -> &'static str {
        match self {
            Level::Off => "off",
            Level::Species => "species",
            Level::Full => "full",
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
    /// A fact of the TRUE SET (the `full` level): free here — gated by the differential against the
    /// opposing chain's own-team slot of the same mon.
    SetFact,
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

/// The DECLARED rule of every cell of a `full`-level tail slot: the set's facts (item, ability, moves, the spread
/// block) are free, `hp_revealed` is 1 (the moves determine the Hidden-Power type, as for an own mon), and every other
/// cell is the pristine state of a never-seen mon, exactly as at the `species` level.
pub const FULL_SLOT_CELLS: [SlotCells; 18] = [
    ("species", POKEMON_SPECIES_OFFSET, POKEMON_ITEMS_OFFSET, Rule::SpeciesDerived),
    ("item", POKEMON_ITEMS_OFFSET, POKEMON_TYPES_OFFSET, Rule::SetFact),
    ("types", POKEMON_TYPES_OFFSET, POKEMON_ABILITIES_OFFSET, Rule::SpeciesDerived),
    ("ability", POKEMON_ABILITIES_OFFSET, POKEMON_CONDITION_OFFSET, Rule::SetFact),
    ("status", POKEMON_CONDITION_OFFSET, POKEMON_MOVES_OFFSET, Rule::Zero),
    ("moves", POKEMON_MOVES_OFFSET, MOVES_END, Rule::SetFact),
    ("hp_fraction", POKEMON_HP_OFFSET, POKEMON_SPECIES_KNOWN_OFFSET, Rule::One),
    ("species_known", POKEMON_SPECIES_KNOWN_OFFSET, POKEMON_COUNTER_OFFSET, Rule::One),
    ("status_counters", POKEMON_COUNTER_OFFSET, POKEMON_SPREAD_OFFSET, Rule::Zero),
    ("spread", POKEMON_SPREAD_OFFSET, POKEMON_HP_REVEALED_OFFSET, Rule::SetFact),
    ("hp_revealed", POKEMON_HP_REVEALED_OFFSET, POKEMON_HP_REVEALED_OFFSET + 1, Rule::One),
    ("hp_type_probs", POKEMON_HP_REVEALED_OFFSET + 1, POKEMON_SLEEP_BELIEF_OFFSET, Rule::Zero),
    ("sleep_belief", POKEMON_SLEEP_BELIEF_OFFSET, POKEMON_RECENCY_OFFSET, Rule::Zero),
    ("recency", POKEMON_RECENCY_OFFSET, POKEMON_PROTECT_OFFSET, Rule::One),
    ("protect", POKEMON_PROTECT_OFFSET, POKEMON_LAST_ACTION_OFFSET, Rule::One),
    ("last_action", POKEMON_LAST_ACTION_OFFSET, POKEMON_VECTOR_DIM, Rule::Zero),
    ("trapped_maybe_trapped", POKEMON_TRAPPED_OFFSET, POKEMON_ACTIVE_OFFSET, Rule::Zero),
    ("active", POKEMON_ACTIVE_OFFSET, POKEMON_FULL_DIM, Rule::Zero),
];

/// THROWS unless `slot` obeys `level`'s declared cells ([`SPECIES_SLOT_CELLS`] / [`FULL_SLOT_CELLS`]) — the
/// producer's guard: a tail slot that carries a fact its level does not reveal (a move, an item, a spread, a
/// status, an on-field flag, …) is a FAULT. Bit comparison: `-0.0` is not `0.0`, a NaN equals nothing.
pub fn check_slot(level: Level, species: &str, slot: &[f32]) -> CoreResult<()> {
    let cells: &[SlotCells] = match level {
        Level::Off => return Err(fault("oracle: Level::Off has no tail slot to check")),
        Level::Species => &SPECIES_SLOT_CELLS,
        Level::Full => &FULL_SLOT_CELLS,
    };
    if slot.len() != POKEMON_FULL_DIM {
        return Err(fault(format!("oracle: the {species} slot has {} cells, not {POKEMON_FULL_DIM}", slot.len())));
    }
    for &(name, lo, hi, rule) in cells {
        let want = match rule {
            Rule::SpeciesDerived | Rule::SetFact => continue,
            Rule::Zero => 0.0f32,
            Rule::One => 1.0f32,
        };
        if let Some(k) = (lo..hi).find(|&k| slot[k].to_bits() != want.to_bits()) {
            return Err(fault(format!(
                "oracle: the {species} tail slot breaks its declared rule: block `{name}` cell {k} is {:?}, must be exactly {want:?} \
                 — a hidden fact would reach the observation (the `{}` level does not tell it)",
                slot[k],
                level.as_str()
            )));
        }
    }
    Ok(())
}

/// One opponent mon the oracle knows: its species, its dex num, its true set and the slot the encoder
/// writes for it while it is unseen.
#[derive(Debug, Clone)]
pub struct OracleMon {
    /// The species id (`skarmory`).
    pub species: String,
    /// The dex num — a forme shares its base species' num, which is what a revealed mon is matched
    /// on (`gen3_data.species.base_form_ids`; Species Clause is by num).
    pub num: i64,
    /// The true set as `team::unpack` read it (the `full` level's overlay reads it).
    set: PokemonSet,
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

/// The reading's mon for a true set at full HP: poke-env's `Pokemon` for the species, then the set's item, ability,
/// moves (full PP) and spread — what the reading of an OWN mon of that set holds before it is ever on the field.
fn true_mon(id: &str, set: &PokemonSet) -> CoreResult<PMon> {
    let mut m = PMon::from_species(id, None)?;
    m.set_hp_status("100/100", true)?;
    apply_set(&mut m, set)?;
    for mv in &set.moves {
        m.learn_move(&typed_move_id(set, mv))?;
    }
    Ok(m)
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

/// The facts of `set` an opponent's reading cannot hold until play reveals them, written onto `m` WITHOUT touching
/// what the reading already knows: the item (only while the reading's is the unknown sentinel — a revealed, consumed
/// or removed one stays), the ability (only while none is revealed), and the spread (never revealed in play).
fn apply_set(m: &mut PMon, set: &PokemonSet) -> CoreResult<()> {
    // `None` is NOT unknown: it is an item play consumed or removed (the reading's `item` is the sentinel until the
    // item is learned), so only the sentinel is overwritten
    if m.item.as_deref() == Some(UNKNOWN_ITEM) {
        m.item = if set.item.is_empty() { None } else { Some(to_id(&set.item)) };
    }
    if m.ability().is_none() && !set.ability.is_empty() {
        m.set_ability(&set.ability);
    }
    // the SAME backfill an own mon's reading takes from its packed team (`Pokemon.backfill_spread_from_teambuilder`):
    // the nature lower-cased (the encoder's table is keyed so), `serious` when none is declared
    m.backfill_spread(&TbMon {
        nickname: None,
        species: Some(set.species.clone()),
        evs: set.evs.iter().map(|&x| x as i64).collect(),
        ivs: set.ivs.iter().map(|&x| x as i64).collect(),
        nature: if set.nature.is_empty() { None } else { Some(set.nature.clone()) },
    });
    Ok(())
}

/// The view of a mon whose set is told: `spread_known`, with the spread the slot writer reads.
fn told_view(base: MonView, m: &PMon) -> MonView {
    MonView { ivs: m.ivs.clone(), evs: m.evs.clone(), nature: m.nature.clone(), spread_known: true, ..base }
}

/// The `full` level's slot for an UNSEEN mon: the row the encoder writes for an OWN mon of this set at full HP, never
/// on the field — the same `slot::populated_slot` as every other slot, fed the true set (the spread block is written
/// because the view says `spread_known`), a side that has seen nothing, and the opponent's perspective.
fn full_slot(id: &str, set: &PokemonSet) -> CoreResult<[f32; POKEMON_FULL_DIM]> {
    let t = data::tables();
    let mon = true_mon(id, set)?;
    let live = told_view(mon_view(&mon, false, false)?, &mon);
    let trk = SideTrackers::new(ClockConfig::default());
    let mut s = [if super::NAN_POISON { f32::NAN } else { 0.0 }; POKEMON_FULL_DIM];
    slot::populated_slot(&trk, t, &mon, &live, false, None, [false, false, false], &mut s)?;
    if let Some(k) = s.iter().position(|x| x.is_nan()) {
        return Err(fault(format!("oracle: full_slot({id}): cell {k} was never written")));
    }
    Ok(s)
}

impl Oracle {
    /// The oracle for `packed` (the OTHER side's packed team) at `level`. `Level::Off` has no
    /// oracle (the chain holds `None`), so it is refused here. Every tail slot is built now and
    /// checked against its level's declared cells.
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
            let slot = match level {
                Level::Species => hypothesis::hypothesis_slot(&id)?,
                Level::Full => full_slot(&id, set)?,
                Level::Off => unreachable!("refused above"),
            };
            check_slot(level, &id, &slot)?;
            mons.push(OracleMon { num: species_num(&id)?, species: id, set: set.clone(), slot });
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

    /// The `full` level's view of a SEEN opponent mon: the reading's `mon` and its `live` view with the facts of the
    /// true set play has NOT revealed written onto them, nothing the play HAS revealed overwritten — an observed move
    /// keeps its tracked PP, a consumed / removed / swapped item stays so, a revealed (or changed) ability is the
    /// reading's. `None` below `full`. A transformed mon shows its TARGET's moves (the true set adds none of them), and
    /// a bare `hiddenpower` the reading learned stands for the set's typed one (no fifth move).
    pub fn overlay(&self, mon: &PMon, live: &MonView) -> CoreResult<Option<(PMon, MonView)>> {
        if self.level != Level::Full {
            return Ok(None);
        }
        let num = species_num(&mon.species)?;
        let Some(om) = self.mons.iter().find(|o| o.num == num) else {
            let team: Vec<&str> = self.mons.iter().map(|o| o.species.as_str()).collect();
            return Err(fault(format!("oracle: the seen opponent {:?} (dex num {num}) is not on the oracle team {team:?}", mon.species)));
        };
        let mut m = mon.clone();
        apply_set(&mut m, &om.set)?;
        if !m.transformed() {
            for mv in &om.set.moves {
                let id = typed_move_id(&om.set, mv);
                let stands_in = id.starts_with("hiddenpower") && m.moves.contains("hiddenpower");
                if !stands_in && !m.moves.contains(&id) && m.moves.moves_ref().len() < 4 {
                    m.learn_move(&id)?;
                }
            }
        }
        let view = told_view(live.clone(), &m);
        Ok(Some((m, view)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_declared_cells_tile_the_slot() {
        for table in [&SPECIES_SLOT_CELLS[..], &FULL_SLOT_CELLS[..]] {
            let mut at = 0;
            for &(name, lo, hi, _) in table {
                assert_eq!(lo, at, "{name} starts at {lo}, the previous block ended at {at}");
                assert!(hi > lo, "{name} is empty");
                at = hi;
            }
            assert_eq!(at, POKEMON_FULL_DIM);
        }
    }

    fn guard_teeth(level: Level, cells: &[SlotCells], good: &[f32; POKEMON_FULL_DIM]) {
        for &(name, lo, _, rule) in cells {
            if matches!(rule, Rule::SpeciesDerived | Rule::SetFact) {
                continue;
            }
            let mut bad = *good;
            bad[lo] = if rule == Rule::Zero { 1.0 } else { 0.5 };
            let e = check_slot(level, "x", &bad).unwrap_err().message().to_string();
            assert!(e.contains(&format!("`{name}`")) && e.contains("declared rule"), "{name}: {e}");
            let mut neg = *good;
            neg[lo] = if rule == Rule::Zero { -0.0 } else { f32::NAN };
            assert!(check_slot(level, "x", &neg).is_err(), "{name}: -0.0 / NaN must fail");
        }
    }

    #[test]
    fn a_species_slot_passes_its_guard_and_every_block_has_teeth() {
        let good = hypothesis::hypothesis_slot("skarmory").unwrap();
        check_slot(Level::Species, "skarmory", &good).unwrap();
        guard_teeth(Level::Species, &SPECIES_SLOT_CELLS, &good);
    }

    // ---- the `full` level

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

    fn oracle_at(level: Level, sets: &[PokemonSet]) -> Oracle {
        let dex = Dex::for_gen(3);
        Oracle::new(level, &crate::team::pack(sets, &dex), &dex).expect("oracle")
    }

    fn skarmory() -> PokemonSet {
        set("skarmory", &["spikes", "roar", "drillpeck", "toxic"], "leftovers", "keeneye")
    }

    fn seen(species: &str) -> (PMon, MonView) {
        let m = PMon::from_species(species, None).unwrap();
        let v = mon_view(&m, true, false).unwrap();
        (m, v)
    }

    #[test]
    fn a_full_tail_slot_tells_the_set_as_an_own_mons_row_would() {
        let o = oracle_at(Level::Full, &[skarmory()]);
        let sl = o.mons()[0].slot();
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
        assert_eq!(sl[POKEMON_HP_REVEALED_OFFSET], 1.0, "the moves determine the Hidden-Power type");
        assert_eq!(sl[POKEMON_ACTIVE_OFFSET], 0.0, "listed, never on the field");
        assert_eq!(sl[POKEMON_RECENCY_OFFSET], 1.0, "never seen");
    }

    #[test]
    fn a_full_slot_passes_its_guard_every_pristine_block_has_teeth_and_the_levels_refuse_each_others_slots() {
        let full = *oracle_at(Level::Full, &[skarmory()]).mons()[0].slot();
        check_slot(Level::Full, "skarmory", &full).unwrap();
        guard_teeth(Level::Full, &FULL_SLOT_CELLS, &full);
        let species = *oracle_at(Level::Species, &[skarmory()]).mons()[0].slot();
        let e = check_slot(Level::Full, "skarmory", &species).unwrap_err().message().to_string();
        assert!(e.contains("`hp_revealed`"), "a species slot is not a full one: {e}");
        let e = check_slot(Level::Species, "skarmory", &full).unwrap_err().message().to_string();
        assert!(e.contains("`item`"), "a full slot leaks the item through the species guard: {e}");
    }

    #[test]
    fn the_overlay_tells_a_seen_mon_only_what_play_has_not_revealed() {
        let o = oracle_at(Level::Full, &[skarmory()]);
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
        let hp = oracle_at(Level::Full, &[set("skarmory", &["hiddenpower", "spikes"], "leftovers", "keeneye")]);
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
    fn the_overlay_is_none_below_full_and_a_seen_mon_off_the_team_is_a_fault() {
        let (m, v) = seen("skarmory");
        assert!(oracle_at(Level::Species, &[skarmory()]).overlay(&m, &v).unwrap().is_none());
        let (z, zv) = seen("zapdos");
        let e = oracle_at(Level::Full, &[skarmory()]).overlay(&z, &zv).unwrap_err().message().to_string();
        assert!(e.contains("zapdos") && e.contains("not on the oracle team"), "{e}");
    }

    #[test]
    fn the_level_round_trips_and_an_unknown_one_is_refused() {
        for l in Level::ALL {
            assert_eq!(Level::parse(l.as_str()).unwrap(), l);
        }
        assert!(Level::parse("everything").unwrap_err().contains("not one of"));
    }
}
