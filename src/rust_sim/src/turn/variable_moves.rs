//! The gen-3 full-parity B8 moves (`gen3_b8_moves_v1`): BIDE, MAGNITUDE, PSYWAVE, PRESENT, TRIPLE
//! KICK and the four OHKO moves (Fissure, Horn Drill, Guillotine, Sheer Cold), each against the
//! RESOLVED `Dex.mod('gen3')`. The settled spec, with every board that settled it, is the header of
//! `harness/probe_b8_moves.js`; the citations below are the files that supply each resolved handler.
//!
//! * The gen-3 hit pipeline is `data/mods/gen3/scripts.ts` `tryMoveHit` (line 265): PrepareHit /
//!   Try, the Invulnerability event (300), the natural-immunity check, the accuracy calculation
//!   with the OHKO branch (346-361), the roll (364), TryHit (369), then `moveHit`; the multihit
//!   loop with the `multiaccuracy` re-roll (414-439); `-ohko` (466); the in-tryMoveHit Update (470).
//! * `useMoveInner` (the same file, line 120) runs the move's `onModifyMove` (145) BEFORE the
//!   `|move|` announce and `onUseMoveMessage` (195) right after it — where Magnitude and Present
//!   draw, and where Magnitude prints `|-activate|<user>|move: Magnitude|<n>`.
//! * `getDamage` (`sim/battle-actions.ts`): `if (move.ohko) return gen === 3 ? target.hp : …`
//!   (1610), `damageCallback` (1611), `if (!basePower) return undefined` (1624) — so Present's
//!   heal branch draws no crit and no damage roll.
//!
//! Every new damage draw that is a gen-3 `random(16)` randomizer roll goes through the existing
//! `BattleState::damage_roll` sites (Magnitude and Present's damage branches are the single-hit
//! site, each Triple Kick strike the multi-hit site). Psywave's `random(50, 151)`, Magnitude's
//! `random(100)` and Present's `random(10)` are not randomizer rolls: like the crit roll they stay
//! the battle seed's own dice under the damage-roll control (`search_and_replay_drivers.md`).

use crate::dex::{Dex, MoveCategory, Type};
use super::status::DamagingHitPhase;
use super::*;
use super::helpers::*;

/// Every gen-3 move with `ohko: true` in the resolved `Dex.mod('gen3')` (gen 3 has no type-gated
/// OHKO; `move.ohko === true` for all four). Pinned against the dex by `tests/b8_moves_test.rs`.
pub const OHKO_MOVES: [&str; 4] = ["fissure", "guillotine", "horndrill", "sheercold"];

/// The gen-3 moves whose base power is drawn in `onModifyMove` over a bp-0 data row — Magnitude
/// and Present. Their resolved category is the gen-3 type split (Ground / Normal → Physical), so
/// Taunt does not block them; the port's bp-0 data row derives Status, hence this list.
pub const RANDOM_BP_MOVES: [&str; 2] = ["magnitude", "present"];

/// Magnitude's `onModifyMove` table (`data/moves.ts` magnitude, unchanged by the gen-3..8 mods):
/// `const i = this.random(100)` and the first row whose bound exceeds `i` gives (magnitude, BP).
pub const MAGNITUDE_TABLE: [(u32, u8, u16); 7] =
    [(5, 4, 10), (15, 5, 30), (35, 6, 50), (65, 7, 70), (85, 8, 90), (95, 9, 110), (100, 10, 150)];

/// Present's `onModifyMove` (`data/moves.ts` present): `const rand = this.random(10)`; below 2 the
/// move HEALS the target 1/4 (`move.heal = [1, 4]`), else base power 40 (< 6), 80 (< 9) or 120.
pub const PRESENT_TABLE: [(u32, u16); 4] = [(2, 0), (6, 40), (9, 80), (10, 120)];

/// Whether `move_id` is an OHKO move (see [`OHKO_MOVES`]).
pub(crate) fn is_ohko_move(move_id: &str) -> bool {
    OHKO_MOVES.contains(&move_id)
}

/// Whether `move_id` draws its base power in `onModifyMove` (see [`RANDOM_BP_MOVES`]).
pub(crate) fn is_random_bp_move(move_id: &str) -> bool {
    RANDOM_BP_MOVES.contains(&move_id)
}

impl crate::state::BattleState {
    /// MAGNITUDE's `onModifyMove` draw: ONE `random(100)`, mapped through [`MAGNITUDE_TABLE`].
    /// Returns `(magnitude, base_power)`.
    pub(crate) fn magnitude_roll(&mut self) -> (u8, u16) {
        let i = self.prng.random_below(100);
        let &(_, mag, bp) = MAGNITUDE_TABLE.iter().find(|(bound, _, _)| i < *bound).expect("i < 100");
        (mag, bp)
    }

    /// PRESENT's `onModifyMove` draw: ONE `random(10)`. `None` is the heal branch, else the BP.
    pub(crate) fn present_roll(&mut self) -> Option<u16> {
        let r = self.prng.random_below(10);
        let &(_, bp) = PRESENT_TABLE.iter().find(|(bound, _)| r < *bound).expect("r < 10");
        (bp > 0).then_some(bp)
    }

    /// The OHKO accuracy (`data/mods/gen3/scripts.ts:346-361`): it REPLACES the stage-folded
    /// accuracy and skips `ModifyAccuracy` (Bright Powder, Sand Veil, Hustle, Compound Eyes play
    /// no part). `accuracy = 30 + (user.level - target.level)` when the user's level is at least
    /// the target's, else `None`: `|-immune|<target>|[ohko]` and the move ends with NO draw (before
    /// Protect and before the natural-immunity report). The `!target.isSemiInvulnerable()` guard
    /// is false only for a target the user has LOCKED ON to (every other semi-invulnerable target
    /// already missed at the Invulnerability event; no OHKO move bypasses Fly / Bounce / Dig /
    /// Dive): the caller skips this check there (`gen3_bf_lockon_ohko_v1`, pins `bf17`).
    pub(crate) fn ohko_accuracy(&self, side: usize, slot: usize, foe: usize, foe_slot: usize) -> Option<u32> {
        let ul = self.sides[side].pokemon[slot].level as u32;
        let tl = self.sides[foe].pokemon[foe_slot].level as u32;
        (ul >= tl).then(|| 30 + (ul - tl))
    }

    /// PSYWAVE's `damageCallback` (`data/moves.ts` psywave, accuracy 80 from `data/mods/gen5/
    /// moves.ts`): `this.random(50, 151) * pokemon.level / 100` — ONE draw, after the accuracy roll
    /// and the TryHit gates (it runs inside `getDamage`). Returns `(to the mon, to a Substitute)`:
    /// `spreadDamage` floors the float and lifts it to 1 (`clampIntRange(damage, 1)`), but the
    /// Substitute's `onTryPrimaryHit` subtracts the RAW float from the sub's HP. A fractional
    /// amount into a Substitute (a user below level 100 whose roll times level is not a multiple
    /// of 100) would leave the sub at a non-integer HP the port cannot hold: FAIL LOUD there.
    pub(crate) fn psywave_damage(&mut self, side: usize, slot: usize, into_sub: bool) -> u16 {
        let r = self.prng.random_range(50, 151);
        let n = r * self.sides[side].pokemon[slot].level as u32;
        if into_sub && n % 100 != 0 {
            panic!(
                "gen3_b8_moves_v1: Psywave dealt the fractional {n}/100 into a Substitute (user level {}); \
                 Showdown keeps the sub's HP as a float, which the port does not model — fail-loud",
                self.sides[side].pokemon[slot].level
            );
        }
        ((n / 100) as u16).max(1)
    }

    /// The BIDE damage store (`bide.condition.onDamage`, `onDamagePriority: -101`): every
    /// Move-effect damage to a biding mon adds the amount after Endure / Focus Band, i.e. the HP the
    /// hit took (a hit that takes more KOs the holder, which ends the volatile). Called from
    /// `record_move_damage` (each move strike) and from the two Move-effect damage sites that do not
    /// feed `move.totalDamage`: a Future Sight / Doom Desire strike and a Bide release.
    pub(crate) fn bide_store(&mut self, side: usize, slot: usize, lost: u32) {
        if let Some(b) = self.sides[side].pokemon[slot].bide.as_mut() {
            b.total_damage = b.total_damage.saturating_add(lost);
        }
    }

    /// `MoveAborted` on a biding mon (`bide.condition.onMoveAborted` → `removeVolatile('bide')` →
    /// `onEnd`: `|-end|<mon>|move: Bide|[silent]`). Every BeforeMove cant of a locked turn (sleep,
    /// freeze, flinch, Disable, Imprison, a confusion self-hit, attraction, full paralysis) ends it.
    /// `removeVolatile` does nothing on a 0-HP mon, so a holder that is already down prints nothing.
    pub(crate) fn bide_end_on_abort(&mut self, side: usize, slot: usize, dex: &Dex) {
        if self.sides[side].pokemon[slot].bide.take().is_some()
            && self.sides[side].pokemon[slot].hp > 0
            && self.logging()
        {
            let u = self.mon_ref(side, slot, dex);
            self.log.volatile_end_silent(&u, "move: Bide");
        }
    }

    /// The BIDE CAST (`data/moves.ts` bide with the gen-3 `accuracy: 100, priority: 0` override in
    /// `data/mods/gen3/moves.ts:57`; target `self`, `volatileStatus: 'bide'`): the announce names the
    /// user as the target, then the accuracy roll — a real one (accuracy 100, not `true`) folding
    /// the user's own accuracy and evasion stages and its own `ModifyAccuracy` holders (Bright
    /// Powder, Sand Veil, Hustle: Bide is physical), so a +evasion user can miss itself
    /// (`|move|<u>|Bide|<u>|[miss]` + `|-miss|<u>|<u>`). On a hit `|-start|<u>|move: Bide`
    /// (`onStart`, `totalDamage = 0`, `duration: 3`), then the King's Rock flinch secondary its
    /// `onModifyMove` appended (`data/mods/gen3/items.ts` lists `bide`) rolls against the USER (the
    /// move's only target): a flinch there is silent and only joins the residual tie group.
    /// `landed` is false (`moveHit` returns undefined → no in-tryMoveHit Update).
    pub(crate) fn run_bide_cast(&mut self, side: usize, slot: usize, move_index: usize, move_name: &str, dex: &Dex) -> MoveResolution {
        let hit = self.roll_accuracy(side, slot, side, slot, 100, false, Some(Type::Normal), dex);
        if self.logging() {
            let u = self.mon_ref(side, slot, dex);
            self.log.move_used(&u, move_name, Some(&u), !hit, false);
            if hit {
                self.log.volatile_start(&u, "move: Bide");
            } else {
                self.log.miss(&u, Some(&u));
            }
        }
        if !hit {
            return MoveResolution::done(true, false, false);
        }
        self.sides[side].pokemon[slot].bide = Some(crate::state::Bide { duration: 3, total_damage: 0, move_index });
        self.apply_kings_rock_secondary(side, slot, side, slot, "bide", false, dex);
        MoveResolution::done(false, false, false)
    }

    /// BIDE's `onBeforeMove` on a locked turn (the resolved gen-3 condition, `data/mods/gen3/
    /// moves.ts:63-98`), reached after every higher-priority BeforeMove handler passed (Bide's has
    /// priority 0; only Destiny Bond's −1 runs after it). `beforeMoveCallback` (`if
    /// (pokemon.volatiles['bide']) return true`) then ends `runMove` — no PP, no lastMove, no
    /// `AfterMove`, no `MoveAborted` — on the WAIT turn (`duration` 2): `|-activate|<u>|move: Bide`.
    /// On the RELEASE turn (`duration` 1): `|-end|<u>|move: Bide`; no damage stored →
    /// `|-fail|<u>`, the handler returns false and `MoveAborted` removes the volatile
    /// (`|-end|…|[silent]`). Otherwise the release runs `tryMoveHit` with a fixed-damage move —
    /// see [`Self::run_bide_release`] — then `removeVolatile('bide')`.
    pub(crate) fn bide_before_move(&mut self, side: usize, slot: usize, dex: &Dex) -> MoveResolution {
        let b = self.sides[side].pokemon[slot].bide.expect("bide_before_move on a biding mon");
        if b.duration != 1 {
            if self.logging() {
                let u = self.mon_ref(side, slot, dex);
                self.log.activate(&u, "move: Bide", None);
            }
            return MoveResolution::bide_wait();
        }
        if self.logging() {
            let u = self.mon_ref(side, slot, dex);
            self.log.volatile_end(&u, "move: Bide");
            if b.total_damage == 0 {
                self.log.fail(&u, None, false);
            }
        }
        if b.total_damage > 0 {
            // The target: `effectState.lastDamageSource`, or `getRandomTarget` when it left the
            // field — in singles both are the current foe active (`sim/battle.ts` getRandomTarget:
            // `if (this.gameType === 'singles') return pokemon.side.foe.active[0]`). A fainted foe
            // active cannot be the target here: gen 3 cancels every remaining action after a faint.
            let foe = 1 - side;
            let foe_slot = self.sides[foe].active;
            let amount = (b.total_damage.saturating_mul(2)).min(u16::MAX as u32) as u16;
            self.run_bide_release(side, slot, foe, foe_slot, amount, dex);
        }
        self.bide_end_on_abort(side, slot, dex);
        MoveResolution::aborted()
    }

    /// The BIDE RELEASE: `this.actions.tryMoveHit(target, pokemon, {accuracy: 100, damage:
    /// totalDamage * 2, category: 'Physical', flags: {contact, protect}, type: 'Normal'})` — called
    /// from BeforeMove, so there is NO `|move|` line, no PP, no `ModifyMove` (no King's Rock) and no
    /// `AfterMoveSecondarySelf` (no Shell Bell). The hit pipeline is the ordinary gen-3 one: a
    /// semi-invulnerable target misses with no draw; the accuracy roll (the user's accuracy and the
    /// target's evasion stages, the target's Bright Powder / Sand Veil, the user's Hustle); Protect
    /// and Wonder Guard at TryHit; a Ghost's `-immune` (`ignoreImmunity` is unset, so the Normal
    /// type IS checked); then the fixed damage through the Substitute / Endure / Focus Band / faint
    /// machinery, the contact procs, Color Change and the in-tryMoveHit Update. A miss appends
    /// `[miss]` to the LAST `|move|` line (`attrLastMove` — this turn's, e.g. the foe's own Fly
    /// charge; a line already streamed in an earlier turn stays as it was sent).
    fn run_bide_release(&mut self, side: usize, slot: usize, foe: usize, foe_slot: usize, amount: u16, dex: &Dex) {
        if self.semi_invulnerable_miss(side, slot, foe, foe_slot, "bide") {
            self.emit_release_miss(side, slot, foe, foe_slot, dex);
            return;
        }
        let acc_hit = self.roll_accuracy(side, slot, foe, foe_slot, 100, false, Some(Type::Normal), dex);
        if acc_hit && self.protect_blocks(foe, foe_slot, false) {
            if self.logging() {
                let t = self.mon_ref(foe, foe_slot, dex);
                self.log.activate(&t, "Protect", None);
            }
            return;
        }
        if acc_hit && self.wonder_guard_blocks(foe, foe_slot, Some(Type::Normal), dex) {
            if self.logging() {
                let t = self.mon_ref(foe, foe_slot, dex);
                self.log.immune_from_ability(&t, "Wonder Guard");
            }
            return;
        }
        let ctx = self.build_damage_context(side, slot, foe, foe_slot, 1, Some(Type::Normal), MoveCategory::Physical, false, dex);
        if move_is_immune(&ctx, dex) {
            if self.logging() {
                let t = self.mon_ref(foe, foe_slot, dex);
                self.log.immune(&t);
            }
            return;
        }
        if !acc_hit {
            self.emit_release_miss(side, slot, foe, foe_slot, dex);
            return;
        }
        // The release's `tryMoveHit` reached `moveHit`: `gotAttacked('bide')` (`gen3_b11_moves_v1`).
        self.note_attacked((foe, foe_slot), (side, slot), "bide");
        self.apply_fixed_hit(side, slot, foe, foe_slot, amount, Some(Type::Normal), "bide", FixedHit::BideRelease, dex);
        // The in-tryMoveHit `eachEvent('Update')` (`data/mods/gen3/scripts.ts:470`) runs BEFORE the
        // caller's `removeVolatile('bide')`, so it is drawn here rather than by the driver.
        let upd = self.each_event_shuffle();
        self.run_update_items(&upd, dex);
    }

    fn emit_release_miss(&mut self, side: usize, slot: usize, foe: usize, foe_slot: usize, dex: &Dex) {
        if self.logging() {
            let u = self.mon_ref(side, slot, dex);
            let t = self.mon_ref(foe, foe_slot, dex);
            self.log.attr_last_move_miss();
            self.log.miss(&u, Some(&t));
        }
    }

    /// Whether Wonder Guard (`data/mods/gen4/abilities.ts` wonderguard `onTryHit`) blocks a
    /// damaging, typed, non-self move into `foe`: everything but a strictly super-effective hit.
    pub(crate) fn wonder_guard_blocks(&self, foe: usize, foe_slot: usize, move_type: Option<Type>, dex: &Dex) -> bool {
        let Some(t) = move_type else { return false };
        let wg = dex
            .ability(&self.sides[foe].pokemon[foe_slot].ability)
            .map(|a| a.wonder_guard)
            .unwrap_or(false);
        wg && dex.type_chart().effectiveness(t, &mon_types(&self.sides[foe].pokemon[foe_slot], dex)) <= 1.0
    }

    /// PRESENT's HEAL branch (`move.heal = [1, 4]`, base power left at 0): the ordinary gen-3
    /// `tryMoveHit` gates — a semi-invulnerable target misses with no draw; the accuracy roll
    /// (90, the full pipeline); Protect and Wonder Guard (Present is a typed physical move) at
    /// TryHit; a Ghost's `-immune` — then `moveHit`. `getDamage` returns undefined at `if
    /// (!basePower)` BEFORE the crit roll, so the heal draws nothing more. A Substitute's
    /// `onTryPrimaryHit` (gen 3 checks only `bypasssub`, not `infiltrates`) calls `getDamage`, gets
    /// undefined and fails: `|-fail|<user>` + the `[still]` retro-edit. Otherwise `runMoveEffects`:
    /// a full-HP target `|-fail|<target>|heal` + `[still]`, else `this.heal(floor(baseMaxhp / 4))`
    /// → `|-heal|<target>|<hp>` (a Move effect: no `[from]`). Never `landed` (`moveHit` returns
    /// undefined, so `tryMoveHit` skips the Update).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_present_heal(
        &mut self,
        side: usize,
        slot: usize,
        foe: usize,
        foe_slot: usize,
        accuracy: u16,
        move_name: &str,
        dex: &Dex,
    ) -> MoveResolution {
        if self.semi_invulnerable_miss(side, slot, foe, foe_slot, "present") {
            self.emit_damaging_miss(side, slot, foe, foe_slot, move_name, false, false, dex);
            return MoveResolution::done(true, false, false);
        }
        let acc_hit = self.roll_accuracy(side, slot, foe, foe_slot, accuracy, false, Some(Type::Normal), dex);
        let announce = |s: &mut Self| {
            if s.logging() {
                let u = s.mon_ref(side, slot, dex);
                let t = s.mon_ref(foe, foe_slot, dex);
                s.log.move_used(&u, move_name, Some(&t), false, false);
            }
        };
        if acc_hit && self.protect_blocks(foe, foe_slot, false) {
            announce(self);
            if self.logging() {
                let t = self.mon_ref(foe, foe_slot, dex);
                self.log.activate(&t, "Protect", None);
            }
            return MoveResolution::done(false, false, false);
        }
        if acc_hit && self.wonder_guard_blocks(foe, foe_slot, Some(Type::Normal), dex) {
            announce(self);
            if self.logging() {
                let t = self.mon_ref(foe, foe_slot, dex);
                self.log.immune_from_ability(&t, "Wonder Guard");
            }
            return MoveResolution::done(false, false, false);
        }
        let ctx = self.build_damage_context(side, slot, foe, foe_slot, 1, Some(Type::Normal), MoveCategory::Physical, false, dex);
        if move_is_immune(&ctx, dex) {
            announce(self);
            if self.logging() {
                let t = self.mon_ref(foe, foe_slot, dex);
                self.log.immune(&t);
            }
            return MoveResolution::done(!acc_hit, false, false);
        }
        if !acc_hit {
            self.emit_damaging_miss(side, slot, foe, foe_slot, move_name, false, false, dex);
            return MoveResolution::done(true, false, false);
        }
        // Reached `moveHit` (the sub's fail and the full-HP fail are inside it): `gotAttacked`.
        self.note_attacked((foe, foe_slot), (side, slot), "present");
        announce(self);
        if self.sides[foe].pokemon[foe_slot].substitute.is_some() {
            if self.logging() {
                let u = self.mon_ref(side, slot, dex);
                self.log.fail(&u, None, false);
                self.log.attr_last_move_still();
            }
            return MoveResolution::done(false, false, false);
        }
        let t = &self.sides[foe].pokemon[foe_slot];
        if t.hp >= t.maxhp {
            if self.logging() {
                let tr = self.mon_ref(foe, foe_slot, dex);
                self.log.fail(&tr, Some("heal"), false);
                self.log.attr_last_move_still();
            }
            return MoveResolution::done(false, false, false);
        }
        let amount = t.maxhp / 4;
        if self.apply_heal(foe, foe_slot, amount) && self.logging() {
            let tr = self.mon_ref(foe, foe_slot, dex);
            let hp = self.hp_status(foe, foe_slot);
            self.log.heal(&tr, &hp, None);
        }
        // The heal still reaches `runMoveEffects`' `runEvent('Hit', target, source, move)` (review F1,
        // `gen3_bf_present_heal_hit_v1`, pins `bf22a-b`): Present is Physical, so the target's RAGE
        // boosts (`rage.condition.onHit`) and a pending FOCUS PUNCH loses focus
        // (`focuspunch.condition.onHit`). The full-HP and Substitute fails above never reach it.
        self.rage_on_hit(foe, foe_slot, dex);
        if let Some(fp) = self.sides[foe].pokemon[foe_slot].focus_punch.as_mut() {
            *fp = true;
        }
        MoveResolution::done(false, false, false)
    }

    /// Apply a fixed amount of Move damage to `foe` after the gates passed and the `|move|` line
    /// (if any) printed: the Substitute intercept, Endure, Focus Band, the faint record, the
    /// Destiny Bond / Grudge / Counter / Focus Punch records, the `|-damage|` line, the King's Rock
    /// secondary (`kind` decides), the contact procs and Color Change — the fixed-damage tail
    /// `run_fixed_damage_move` always ran, shared with the Bide release and the OHKO moves
    /// (`-ohko` right after a Substitute line, else after the tail when the target is at 0 HP —
    /// `data/mods/gen4/moves.ts:1309` and `data/mods/gen3/scripts.ts:466`). The same `amount` meets
    /// a Substitute (an OHKO's is the MON's current HP, `getDamage` reading `target.hp`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_fixed_hit(
        &mut self,
        side: usize,
        slot: usize,
        foe: usize,
        foe_slot: usize,
        amount: u16,
        move_type: Option<Type>,
        move_id: &str,
        kind: FixedHit,
        dex: &Dex,
    ) {
        let sub = self.absorb_into_sub(foe, foe_slot, amount);
        let absorbed = sub != SubAbsorb::NoSub;
        if !absorbed {
            // ENDURE clamps a FIXED-damage Move hit too (`gen3_move_coverage_batch6_v1`
            // — probe ED6: a lethal Seismic Toss into an endurer leaves it at 1 HP;
            // endure −10 precedes Focus Band's −40). DRAW-FREE.
            let amount = self.endure_clamp(foe, foe_slot, amount, dex);
            // FOCUS BAND: a fixed-damage move is a MOVE-effect Damage event — the roll
            // draws and a lethal Seismic Toss can be survived at 1. `gen3_ability_batch4_v1`.
            let pre_hp = self.sides[foe].pokemon[foe_slot].hp;
            let amount = self.focus_band_damage(foe, foe_slot, amount, true, true, dex);
            self.apply_damage(foe, foe_slot, amount);
            if kind == FixedHit::BideRelease {
                // Not part of a `move.totalDamage` (no Shell Bell): only Bide's own store.
                let lost = pre_hp.saturating_sub(self.sides[foe].pokemon[foe_slot].hp) as u32;
                self.bide_store(foe, foe_slot, lost);
            } else {
                self.record_move_damage(foe, foe_slot, pre_hp);
            }
            // DESTINY BOND trigger record (`gen3_move_coverage_batch6_v1`): a
            // fixed-damage foe Move KO — incl. a Counter/Mirror-Coat RETURN (a foe
            // Move) — with the DB volatile up marks the pending mutual faint. DRAW-FREE.
            if self.sides[foe].pokemon[foe_slot].hp == 0 && self.sides[foe].pokemon[foe_slot].destiny_bond {
                self.sides[foe].pokemon[foe_slot].destiny_bond_ko_by = Some(side);
            }
            // GRUDGE trigger record (`gen3_b5_moves_v1`): the same foe-Move KO gate (the chain runs
            // in `process_faints`, after the corpse's `|faint|`). DRAW-FREE.
            if self.sides[foe].pokemon[foe_slot].hp == 0 && self.sides[foe].pokemon[foe_slot].grudge {
                self.sides[foe].pokemon[foe_slot].grudge_ko_by = Some(side);
            }
            // COUNTER / MIRROR COAT recorder (`gen3_move_coverage_batch5_v1`): a
            // fixed-damage foe hit IS a Move-effect Damage event — Seismic Toss
            // (Fighting → Physical) arms Counter for 2×100 (probed CS), Dragon Rage
            // (Dragon → Special) would arm Mirror Coat. The gen3 category is the
            // TYPE-derived one (`derive_category` with a nonzero bp forces the type
            // branch); `dealt` = the post-Focus-Band applied amount, clamped.
            let dealt = amount.min(pre_hp);
            let cat = crate::dex::moves::derive_category(3, 1, move_type);
            self.record_reactive_hit(foe, foe_slot, cat, move_id, dealt);
            // FOCUS PUNCH lostFocus (`gen3_move_coverage_batch4_v1`,
            // `focuspunch.condition.onHit`): a FIXED-DAMAGE hit is a NON-Status move
            // that HIT the FP user DIRECTLY, so it cancels a queued Focus Punch exactly
            // like the normal damaging path (SIM-VERIFIED by the batch-5 e2e admission
            // itself — e2e_202 dec44). DRAW-FREE.
            if let Some(fp) = self.sides[foe].pokemon[foe_slot].focus_punch.as_mut() {
                *fp = true;
            }
            // [EMIT] `|-damage|<foe>|<HP>` with the POST-damage HP. The modeled fixed-
            // damage moves always deal >= 1 to a non-immune target, so the emit fires.
            if amount > 0 && self.logging() {
                let target = self.mon_ref(foe, foe_slot, dex);
                let hp = self.hp_status(foe, foe_slot);
                self.log.damage(&target, &hp, None);
            }
            // RAGE reacts to a fixed-damage hit too: Seismic Toss / Night Shade / Counter / Psywave
            // / an OHKO / a Bide release are Physical or Special, so `runEvent('Hit')` boosts the
            // holder (`gen3_bf_rage_v1`; a KO'd holder gets nothing).
            self.rage_on_hit(foe, foe_slot, dex);
        } else if self.logging() {
            // [EMIT] the SUBSTITUTE result (INSTEAD of `-damage`): a survived absorb shows
            // `|-activate|<foe>|Substitute|[damage]`, a broken sub `|-end|<foe>|Substitute`.
            let target = self.mon_ref(foe, foe_slot, dex);
            match sub {
                SubAbsorb::Held => self.log.activate(&target, "Substitute", Some("[damage]")),
                SubAbsorb::Broke => self.log.volatile_end(&target, "Substitute"),
                SubAbsorb::NoSub => unreachable!(),
            }
            // An OHKO into a Substitute: the sub's `onTryPrimaryHit` prints `-ohko` right after
            // its own line, broken or not (`data/mods/gen4/moves.ts:1309`).
            if kind == FixedHit::Ohko {
                self.log.ohko();
            }
        }

        // --- KING'S ROCK on a fixed-damage LISTED move (Seismic Toss / Psywave — probed: the
        //     appended secondary rolls right after the damage apply, `random(100,)` with
        //     no crit/randomizer preceding it). The Bide release runs no ModifyMove, so it has
        //     none. `gen3_ability_batch4_v1`. ---
        if kind != FixedHit::BideRelease {
            self.apply_kings_rock_secondary(side, slot, foe, foe_slot, move_id, absorbed, dex);
        }
        // --- CONTACT_PROC (+ Rough Skin recoil) on a CONTACT fixed-damage hit
        //     (`gen3_ability_batch2_v1` × the fixed-damage family — Seismic Toss /
        //     Super Fang / Counter / Endeavor / Horn Drill / Guillotine / the Bide release carry
        //     `flags.contact`): the DEFENDER's `onDamagingHit` fires for a fixed-damage hit exactly
        //     like the normal damaging path (AFTER the appended King's Rock secondary, NOT behind a
        //     sub). Pin MC99. PHASE: `All` — no fire-thaw precedes this tail (no fixed-damage move
        //     is Fire-typed). ---
        let is_contact = dex.moves(move_id).map(|m| m.contact).unwrap_or(false);
        if is_contact && !absorbed && amount > 0 {
            self.apply_contact_proc(side, slot, foe, foe_slot, DamagingHitPhase::All, dex);
        }
        if !absorbed && amount > 0 {
            self.apply_color_change(foe, foe_slot, move_type, dex);
        }
        // `if (move.ohko && !target.hp) this.battle.add('-ohko')` — after `moveHit` (the contact
        // procs and Color Change ran inside it), before the in-tryMoveHit Update.
        if kind == FixedHit::Ohko && !absorbed && self.sides[foe].pokemon[foe_slot].hp == 0 && self.logging() {
            self.log.ohko();
        }
    }
}

/// Which fixed-damage tail [`crate::state::BattleState::apply_fixed_hit`] runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FixedHit {
    /// A `damage:` / `damageCallback` move used through `useMove` (Seismic Toss … Psywave).
    Normal,
    /// An OHKO move: the `-ohko` line.
    Ohko,
    /// Bide's release: no King's Rock, no Shell Bell basis.
    BideRelease,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_magnitude_table_partitions_random_100() {
        let mut counts = [0u32; 7];
        for i in 0..100u32 {
            let k = MAGNITUDE_TABLE.iter().position(|(b, _, _)| i < *b).unwrap();
            counts[k] += 1;
        }
        assert_eq!(counts, [5, 10, 20, 30, 20, 10, 5]);
    }

    #[test]
    fn the_present_table_partitions_random_10() {
        let mut counts = [0u32; 4];
        for r in 0..10u32 {
            counts[PRESENT_TABLE.iter().position(|(b, _)| r < *b).unwrap()] += 1;
        }
        assert_eq!(counts, [2, 4, 3, 1]);
    }

    #[test]
    fn the_ohko_and_random_bp_families_are_disjoint() {
        for m in OHKO_MOVES {
            assert!(!is_random_bp_move(m) && is_ohko_move(m));
        }
        for m in RANDOM_BP_MOVES {
            assert!(!is_ohko_move(m) && is_random_bp_move(m));
        }
    }
}
