//! The ENDLESS BATTLE CLAUSE (`gen3_b10_endless_battle_v1`, gen3 full parity B10, row M7):
//! `sim/battle.ts` `maybeTriggerEndlessBattleClause` (1800-1899), called from `endTurn` (1770) after
//! `this.turn++` and the per-side trap / staleness loop (1661-1767), before the `|turn|` line and the
//! gen-3 Quick Claw roll. Every gen-3 singles format but Custom Game carries the rule
//! (`FormatRules::endless_battle_clause`, from `ruleTable.has('endlessbattleclause')`).
//!
//! The settled gen-3 reading (probe `harness/probe_b10_moves.js`, the EBC boards):
//!   * Nothing happens on or before turn 100 (`if (this.turn <= 100) return;`). The turn-1000 tie
//!     and the `|bigerror|` countdown come first (`gen3_turn_limit_tie_v1`), so a countdown line
//!     precedes the clause's lines on a turn that has both.
//!   * A mon's staleness is `volatileStaleness || staleness`. In gen 3 `volatileStaleness` is set
//!     ONLY by Skill Swap on its target (`'external'`, 1336-1337) and cleared at switch-out; the
//!     permanent `staleness` is set ONLY by eating a RESTORATIVE berry while a pending staleness is
//!     up, and the pending value is ALWAYS `'internal'` in gen 3 (every `setItem` — Trick, Thief,
//!     Covet, Recycle — sets it; gen-3 Trick's `setItem` passes no `source`, so the `'external'`
//!     branch of `sim/pokemon.ts:1873-1876` is unreachable). So EXTERNAL staleness is Skill Swap's
//!     alone, and no gen-3 battle reaches it without a Skill Swap AND a berry cycle.
//!   * The clause fires when every side's active is stale, at least one EXTERNALLY, and NOT every
//!     side can switch to a non-stale mon (a TRAPPED side cannot; else a side can when a
//!     non-fainted party mon is neither volatile- nor permanently stale).
//!   * The loser test reads each side's SETS in the CURRENT party order (`side.pokemon`, which a
//!     switch reorders — the port's array swaps the same way): `berry` is RE-ASSIGNED per mon
//!     (`berry = RESTORATIVE_BERRIES.has(toID(pokemon.set.item))`), `cycle` is sticky (a Harvest /
//!     Pickup set ability or a Recycle set move), and the loop breaks when both hold. Exactly one
//!     such side LOSES: `|-message|<Name>'s team started with the rudimentary means to perform
//!     restorative berry-cycling and thus loses.` then `|` + `|win|<Foe>`. Both: `|-message|Each
//!     side's team started …` then a tie. Neither: a bare tie. The battle ends draw-free (the Quick
//!     Claw roll and the `|turn|` line never come).
use crate::dex::Dex;

/// `RESTORATIVE_BERRIES` (`sim/pokemon.ts:41-45`, exported): the berries that restore PP or HP.
/// Checked against the export by `tests/b10_moves_test.rs` B10-0.
pub const RESTORATIVE_BERRIES: [&str; 9] = [
    "leppaberry", "aguavberry", "enigmaberry", "figyberry", "iapapaberry", "magoberry", "sitrusberry", "wikiberry",
    "oranberry",
];

/// The set abilities that count as a berry-cycling means (`sim/battle.ts:1881`). Harvest is gen 5;
/// Pickup is in gen 3 (it has no battle effect there, but the clause still reads the set).
pub const BERRY_CYCLE_ABILITIES: [&str; 2] = ["harvest", "pickup"];

/// How the clause ends the battle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EndlessOutcome {
    /// Exactly one side's team could berry-cycle: it loses.
    Loses { loser: usize },
    /// Both sides' teams could: a tie announced with the "Each side's team" message.
    BothCycled,
    /// Neither: a bare tie.
    Tie,
}

impl crate::state::BattleState {
    /// Whether the clause ends the battle at the `endTurn` whose incremented turn counter is
    /// `turn` (the sim's `this.turn`, i.e. the port's `bs.turn + 1`). Pure: reads state, draws
    /// nothing.
    pub(crate) fn endless_battle_clause_outcome(&self, turn: u32, dex: &Dex) -> Option<EndlessOutcome> {
        if !self.rules.endless_battle_clause || turn <= 100 {
            return None;
        }
        // `stalenessBySide`: every side's (live) active stale, at least one externally.
        let mut external = false;
        for side in &self.sides {
            let m = &side.pokemon[side.active];
            if m.fainted || !(m.volatile_staleness || m.staleness) {
                return None;
            }
            external |= m.volatile_staleness;
        }
        if !external {
            return None;
        }
        // "Can both sides switch to a non-stale Pokemon?" — a trapped side cannot.
        let can_switch = (0..self.sides.len()).all(|i| {
            !self.is_trapped(i, dex)
                && self.sides[i].pokemon.iter().any(|p| !p.fainted && !(p.volatile_staleness || p.staleness))
        });
        if can_switch {
            return None;
        }
        let losers: Vec<usize> = (0..self.sides.len())
            .filter(|&i| {
                let (mut berry, mut cycle) = (false, false);
                for p in &self.sides[i].pokemon {
                    berry = p.set_restorative_berry;
                    cycle |= p.set_berry_cycle;
                    if berry && cycle {
                        break;
                    }
                }
                berry && cycle
            })
            .collect();
        Some(match losers.len() {
            1 => EndlessOutcome::Loses { loser: losers[0] },
            0 => EndlessOutcome::Tie,
            _ => EndlessOutcome::BothCycled,
        })
    }
}
