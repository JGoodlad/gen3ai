//! THE MOVE-CALLERS (`gen3_b11_moves_v1`, gen3 full parity B11): Metronome, Mirror Move, Assist and
//! Nature Power, which run another move through Showdown's bare `actions.useMove(id, pokemon)`, and
//! Sketch, which copies one permanently. The settled spec, every draw and line, is
//! `harness/probe_b11_moves.js`'s header; the live differential is `tests/b11_moves_golden_test.rs`.
//!
//! ONE calling path serves every caller, Sleep Talk included ([`BattleState::call_move`]): the
//! called move runs as the nested `run_move` under the `use_move_call` transient, which is the sim's
//! `useMove` (`sim/battle-actions.ts` useMove; the gen-3 `useMoveInner`, `data/mods/gen3/scripts.ts:
//! 126-262`): no BeforeMove event (no cant, no Disable / Imprison / Taunt gate, no Rage or Grudge
//! removal), no PP (and so no Pressure extra, `scripts.ts:177`), no `moveUsed` (the caller stays the
//! user's `lastMove`), and the announce carries `|[from] <Caller>` (`scripts.ts:165`). It does set
//! `lastMoveUsed` to the called move (`scripts.ts:132`), which Conversion 2 reads.
//!
//! A callee the user holds in a moveslot runs from that slot (its id is what `lastMove` will name
//! on a locked repeat, so Disable / Encore / Spite / Mimic / Sketch read it right); any other callee
//! runs from the CALLED-MOVE virtual slot ([`crate::state::CALLED_MOVE_SLOT`]), which the locks a
//! called move starts (two-turn charge, Outrage family, Rollout, Uproar, Bide) then hold. A callee the
//! engine does not model reaches its own fail-loud panic on that `run_move` (the status-move
//! fall-through names it), never an approximation.
use crate::dex::{to_id, Dex, Type};
use super::*;

/// The gen-3 moves WITHOUT `flags.metronome` (the RESOLVED `Dex.mod('gen3')`: the base data's flags,
/// `data/mods/gen4/moves.ts` metronome, `data/mods/gen3/moves.ts` assist / mirrormove): METRONOME
/// never calls them. The bare Hidden Power IS in the pool (typed by the user's IVs); the port's
/// typed `hiddenpower<type>` rows are not ([`metronome_pool`]). Pinned against the resolved dex, ORDER included, by
/// `tests/b11_moves_test.rs` B11-1.
pub const METRONOME_EXCLUDED: [&str; 18] = [
    "counter", "covet", "destinybond", "detect", "endure", "focuspunch", "followme", "helpinghand",
    "metronome", "mimic", "mirrorcoat", "protect", "sketch", "sleeptalk", "snatch", "struggle", "thief",
    "trick",
];

/// The gen-3 moves carrying `flags.noassist` (resolved `Dex.mod('gen3')`): ASSIST never calls them.
/// Pinned by `tests/b11_moves_test.rs` B11-1.
pub const NOASSIST_MOVES: [&str; 20] = [
    "assist", "counter", "covet", "destinybond", "detect", "endure", "focuspunch", "followme",
    "helpinghand", "metronome", "mimic", "mirrorcoat", "mirrormove", "protect", "sketch", "sleeptalk",
    "snatch", "struggle", "thief", "trick",
];

/// MIRROR MOVE's gen-3 `noMirror` list (`data/mods/gen3/moves.ts` mirrormove.onHit; the gen-3 move
/// has NO `mirror`-flag check: its `onTryHit` is `undefined`). Pinned against the resolved handler
/// source by `tests/b11_moves_test.rs` B11-1.
pub const MIRROR_MOVE_EXCLUDED: [&str; 19] = [
    "assist", "curse", "doomdesire", "focuspunch", "futuresight", "magiccoat", "metronome", "mimic",
    "mirrormove", "naturepower", "psychup", "roleplay", "sketch", "sleeptalk", "spikes", "spitup",
    "taunt", "teeterdance", "transform",
];

/// The gen-3 moves carrying `flags.nosketch` (resolved `Dex.mod('gen3')`): SKETCH fails on them.
/// Pinned by `tests/b11_moves_test.rs` B11-1.
pub const NOSKETCH_MOVES: [&str; 2] = ["sketch", "struggle"];

/// What NATURE POWER calls in a gen-3 link battle (`data/mods/gen3/moves.ts` naturepower.onHit:
/// `this.actions.useMove('swift', target)`).
pub const NATURE_POWER_CALLS: &str = "swift";

/// A moveslot id with a typed Hidden Power collapsed to the bare `hiddenpower` (Showdown's moveslot
/// and `move.id` form).
fn bare_id(id: &str) -> String {
    let id = to_id(id);
    if id.starts_with("hiddenpower") {
        "hiddenpower".to_string()
    } else {
        id
    }
}

/// METRONOME's pool (`data/mods/gen4/moves.ts` metronome.onHit, inherited by gen 3): every move with
/// `flags.metronome` (`!isNonstandard`, gen ≤ 3 in the gen-3 dex), SORTED BY `num` before the
/// `sample`. Gen 3 does not drop the user's own moves (that clause is gen 2 / 4 only).
pub fn metronome_pool(dex: &Dex) -> Vec<String> {
    let mut pool: Vec<(u16, &str)> = dex
        .all_moves()
        // The bare `hiddenpower` row (num 237) carries `flags.metronome`; the 16 typed rows are
        // the port's own (Showdown keeps the type on the mon), so they never enter the pool.
        .filter(|m| (m.id == "hiddenpower" || !m.id.starts_with("hiddenpower")) && !METRONOME_EXCLUDED.contains(&m.id.as_str()))
        .map(|m| (m.num, m.id.as_str()))
        .collect();
    pool.sort_unstable();
    pool.into_iter().map(|(_, id)| id.to_string()).collect()
}

impl crate::state::BattleState {
    /// The slot a called move runs from (`gen3_b11_moves_v1`): the user's own moveslot holding it
    /// (Hidden Power matched bare), else the CALLED-MOVE virtual slot, loaded with the callee (a bare
    /// Hidden Power resolved to the USER's type: the sim types the called `hiddenpower` by
    /// `pokemon.hpType`, `data/mods/gen3/moves.ts` hiddenpower via `sim/pokemon.ts:410`).
    pub(crate) fn callee_slot(&mut self, side: usize, slot: usize, id: &str) -> usize {
        let want = bare_id(id);
        let mon = &mut self.sides[side].pokemon[slot];
        if let Some(k) = mon.set.moves.iter().position(|m| bare_id(m) == want) {
            return k;
        }
        let resolved = if want == "hiddenpower" {
            crate::state::typed_hp_move_id(mon, "hiddenpower")
        } else {
            want
        };
        mon.called_move = Some(resolved);
        crate::state::CALLED_MOVE_SLOT
    }

    /// THE ONE CALLING PATH (`gen3_b11_moves_v1`): the sim's bare `actions.useMove(<callee>,
    /// pokemon)` from a caller's `onHit` (Sleep Talk, Metronome, Mirror Move, Assist, Nature Power).
    /// `k` is the callee's slot ([`Self::callee_slot`], or Sleep Talk's own sampled slot). Sets
    /// `lastMoveUsed` (`useMoveInner`'s first act, `data/mods/gen3/scripts.ts:132`), arms the
    /// `|[from] <from>` announce fold (`scripts.ts:165`) and runs the callee under `use_move_call`.
    /// The callee's resolution propagates (its landed Update, a Roar's drag).
    pub(crate) fn call_move(
        &mut self,
        side: usize,
        slot: usize,
        k: usize,
        from: &str,
        will_act: bool,
        foe_will_move: bool,
        dex: &Dex,
    ) -> MoveResolution {
        let used = self.sides[side].pokemon[slot].slot_move_id(k).map(to_id);
        if let Some(id) = used {
            self.sides[side].pokemon[slot].last_move_used = Some(id);
        }
        self.log.set_next_move_from(from);
        self.use_move_call = true;
        self.run_move(MoveAction { side, slot, move_index: k, struggle: false }, will_act, foe_will_move, dex)
    }

    /// `pokemon.gotAttacked` (`sim/pokemon.ts:918`, `gen3_b11_moves_v1`): record that `src`'s move
    /// reached `moveHit` against `target` (gen-3 `tryMoveHit` pushes it after the hit,
    /// `data/mods/gen3/scripts.ts:464`; `target !== pokemon`). Called at each move path's
    /// "reached moveHit" point. Draw-free; read only by Mirror Move.
    pub(crate) fn note_attacked(&mut self, target: (usize, usize), src: (usize, usize), move_id: &str) {
        if target == src {
            return;
        }
        let uid = self.sides[src.0].pokemon[src.1].uid;
        self.sides[target.0].pokemon[target.1].attacked_by.push(crate::state::AttackedBy {
            side: src.0,
            uid,
            move_id: bare_id(move_id),
        });
    }

    /// endTurn's `attackedBy` sweep (`sim/battle.ts` endTurn, the `for (const attack of
    /// pokemon.attackedBy)` loop): an entry whose source is no longer active is dropped (the
    /// `thisTurn` flag it also clears is read by nothing in gen 3 singles).
    pub(crate) fn prune_attacked_by(&mut self) {
        for side in 0..2 {
            let slot = self.sides[side].active;
            let mut entries = std::mem::take(&mut self.sides[side].pokemon[slot].attacked_by);
            entries.retain(|a| {
                let s = &self.sides[a.side];
                let m = &s.pokemon[s.active];
                m.uid == a.uid && !m.fainted
            });
            self.sides[side].pokemon[slot].attacked_by = entries;
        }
    }

    /// The five B11 callers' `onHit` (`gen3_b11_moves_v1`), run from `run_status_move` after the
    /// self-targeted announce (`|move|<u>|<Caller>|<u>`; Sketch's names its target). `None` = not a
    /// B11 move.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_b11_move(
        &mut self,
        side: usize,
        slot: usize,
        foe: usize,
        foe_slot: usize,
        move_id: &str,
        will_act: bool,
        foe_will_move: bool,
        dex: &Dex,
    ) -> Option<MoveResolution> {
        match move_id {
            // METRONOME (`data/mods/gen4/moves.ts` metronome.onHit): ONE `sample` over the
            // num-sorted pool, drawn right after the announce, then `useMove`. Never fails in gen 3.
            "metronome" => {
                let pool = metronome_pool(dex);
                let pick = self.prng.random_below(pool.len() as u32) as usize;
                let k = self.callee_slot(side, slot, &pool[pick]);
                Some(self.call_move(side, slot, k, "Metronome", will_act, foe_will_move, dex))
            }
            // ASSIST (`data/mods/gen4/moves.ts` assist.onHit): the moveslots of every OTHER mon of
            // the side, in party (position) order, fainted ones included, each `noassist` id
            // dropped and duplicates kept; ONE `sample`, or a draw-free fail on an empty pool.
            "assist" => {
                let active = self.sides[side].active;
                let mut pool: Vec<String> = Vec::new();
                for (i, mon) in self.sides[side].pokemon.iter().enumerate() {
                    if i == active {
                        continue;
                    }
                    for mv in &mon.set.moves {
                        let id = bare_id(mv);
                        if !NOASSIST_MOVES.contains(&id.as_str()) {
                            pool.push(id);
                        }
                    }
                }
                if pool.is_empty() {
                    self.fail_still(side, slot, dex);
                    return Some(MoveResolution::done(false, false, false));
                }
                let pick = self.prng.random_below(pool.len() as u32) as usize;
                let k = self.callee_slot(side, slot, &pool[pick]);
                Some(self.call_move(side, slot, k, "Assist", will_act, foe_will_move, dex))
            }
            // MIRROR MOVE (`data/mods/gen3/moves.ts` mirrormove.onHit): the LAST `attackedBy` entry;
            // it fails (draw-free) when there is none, when its source has no `lastMove` (switched
            // out or fainted since: `clearVolatile` nulls it), when the move is on `noMirror`, or when
            // the source no longer holds it (`hasMove`: a Metronome / Sleep Talk / Assist callee
            // the source does not carry).
            "mirrormove" => {
                let Some(last) = self.sides[side].pokemon[slot].attacked_by.last().cloned() else {
                    self.fail_still(side, slot, dex);
                    return Some(MoveResolution::done(false, false, false));
                };
                let src = self.sides[last.side].pokemon.iter().find(|m| m.uid == last.uid);
                let ok = src.is_some_and(|m| {
                    m.last_move_used.is_some() && m.set.moves.iter().any(|mv| bare_id(mv) == last.move_id)
                }) && !MIRROR_MOVE_EXCLUDED.contains(&last.move_id.as_str());
                if !ok {
                    self.fail_still(side, slot, dex);
                    return Some(MoveResolution::done(false, false, false));
                }
                let k = self.callee_slot(side, slot, &last.move_id);
                Some(self.call_move(side, slot, k, "Mirror Move", will_act, foe_will_move, dex))
            }
            // NATURE POWER (`data/mods/gen3/moves.ts` naturepower: accuracy 95, target self): the
            // move's OWN accuracy roll against its user (the user's accuracy / evasion stages and
            // ModifyAccuracy holders fold in, as Bide's), `[miss]` + `|-miss|<u>|<u>` on a miss;
            // else `useMove('swift')`.
            "naturepower" => {
                if !self.roll_accuracy(side, slot, side, slot, 95, false, Some(Type::Normal), dex) {
                    if self.logging() {
                        let user = self.mon_ref(side, slot, dex);
                        self.log.attr_last_move_miss();
                        self.log.miss(&user, Some(&user));
                    }
                    return Some(MoveResolution::done(true, false, false));
                }
                let k = self.callee_slot(side, slot, NATURE_POWER_CALLS);
                Some(self.call_move(side, slot, k, "Nature Power", will_act, foe_will_move, dex))
            }
            "sketch" => Some(self.run_sketch(side, slot, foe, foe_slot, dex)),
            _ => None,
        }
    }

    /// The draw-free `onHit` fail: `|move|…||[still]` + `|-fail|<user>`.
    fn fail_still(&mut self, side: usize, slot: usize, dex: &Dex) {
        if self.logging() {
            self.log.attr_last_move_still();
            let user = self.mon_ref(side, slot, dex);
            self.log.fail(&user, None, false);
        }
    }

    /// SKETCH (`data/mods/gen4/moves.ts` sketch.onHit, inherited by gen 3; flags from
    /// `data/mods/gen3/moves.ts`: no `protect`, `bypasssub`). Draw-free. Fails (`[still]` +
    /// `-fail|<user>`) when the user is transformed, the target has no `lastMove`, the target is
    /// behind a Substitute, the `lastMove` is `nosketch` (Sketch, Struggle) or already one of the
    /// user's moves (Hidden Power matched bare). Else the Sketch slot becomes the copied move
    /// PERMANENTLY (`moveSlots` AND `baseMoveSlots`): PP and max PP the move's BASE PP (the Sketch
    /// slot has 0 PP Ups), no `target` key in the request (the literal omits it), and
    /// `|-activate|<user>|move: Mimic|<Move>`.
    fn run_sketch(&mut self, side: usize, slot: usize, foe: usize, foe_slot: usize, dex: &Dex) -> MoveResolution {
        // No `protect` flag and every fail is `onHit`'s, inside `moveHit`: `gotAttacked`.
        self.note_attacked((foe, foe_slot), (side, slot), "sketch");
        let copied = self.last_move_id(foe, foe_slot);
        let fail = self.sides[side].pokemon[slot].transformed()
            || self.sides[foe].pokemon[foe_slot].substitute.is_some()
            || match copied.as_deref() {
                None => true,
                Some(id) => {
                    NOSKETCH_MOVES.contains(&id)
                        || self.sides[side].pokemon[slot].set.moves.iter().any(|m| bare_id(m) == id)
                }
            };
        let sketch_slot = self.sides[side].pokemon[slot].set.moves.iter().position(|m| to_id(m) == "sketch");
        let (Some(id), Some(k), false) = (copied, sketch_slot, fail) else {
            self.fail_still(side, slot, dex);
            return MoveResolution::done(false, false, false);
        };
        let (name, pp) = match dex.moves(&id) {
            Some(m) => (m.display_name().to_string(), m.pp),
            None => panic!("gen3_b11_moves_v1: Sketch copied {id:?}, which the dex lacks"),
        };
        {
            let m = &mut self.sides[side].pokemon[slot];
            m.set.moves[k] = id;
            m.move_pp[k] = pp;
            m.move_maxpp[k] = pp;
            m.sketched |= 1u32 << k;
        }
        if self.logging() {
            let user = self.mon_ref(side, slot, dex);
            self.log.activate(&user, "move: Mimic", Some(&name));
        }
        MoveResolution::done(false, false, false)
    }
}
