use crate::dex::{to_id, Dex, MoveCategory, Type};
use crate::protocol::Cause;
use crate::state::{Status, Weather};
use super::helpers::*;

impl crate::state::BattleState {

    /// The COUNTER / MIRROR COAT damage RECORDER (`gen3_move_coverage_batch5_v1` — the
    /// reactive volatile's `onDamage`, priority **−101** = LAST, so it reads the FINAL
    /// post-Focus-Band damage). Called at each site where a FOE **Move**-effect hit
    /// lands on the MON directly (a SUB-absorbed hit never fires the mon's Damage event
    /// → the sites never call this for an absorbed hit; confusion self-hits / recoil /
    /// residuals are NOT Move-effect foe hits → no site calls it). `dealt` is the
    /// ACTUAL applied damage (clamped + post-Focus-Band). The qualifying rule
    /// (probe-settled, `probe_batch5_reactive.js` + `probe_batch5_reactive_edges.js`):
    ///
    ///   * `counter`   ← `category == Physical || id == hiddenpower` (gen3 type-derived
    ///     category: Seismic Toss / Endeavor / Struggle ARE Physical; a bare Hidden
    ///     Power is countered UNCONDITIONALLY — Showdown normalizes every typed
    ///     `hiddenpowerX` id to bare `hiddenpower` at Pokemon construction, so the
    ///     `starts_with` fold mirrors that);
    ///   * `mirrorcoat` ← `category == Special && id != hiddenpower` (Beat Up's strikes
    ///     are Special → recorded, 2× the LAST strike; HP NEVER arms Mirror Coat even
    ///     special-typed — probed).
    ///
    /// Each qualifying hit OVERWRITES `damage = 2 × dealt` (the multihit last-strike
    /// rule). DRAW-FREE. (A FUTURE-move resolve is unobservable here: the volatile
    /// RESETS at the next selection's beforeTurnMove before the move could ever read a
    /// residual-time record, so no future-resolve site calls this.)
    ///
    /// **A ZERO-damage hit NEVER records** (`dealt > 0` gate): a Focus-Band proc on a
    /// holder at exactly 1 HP reduces the applied damage to 0, and the sim's
    /// `runEvent('Damage')` BREAKS its handler chain on a falsy relayVar
    /// (battle.js:695 `if (!relayVar || fastExit) break`) BEFORE the counter's
    /// priority-−101 recorder ever runs — so the sim leaves Counter/Mirror Coat
    /// UN-ARMED (a bare zero-draw `|move|` fail next), while an armed-with-0 port
    /// would draw an extra accuracy roll (a latent seed desync). Behavioral probe
    /// `harness/probe_lens1_batch5_review.js` R3 (seed [6,6,6,6]: Drill Peck lethal
    /// into a 1-HP Focus-Band Snorlax → `|-activate|item: Focus Band`, 0 dealt, then
    /// a bare `|move|…|Counter|…` with NO accuracy draw, Skarmory untouched); pinned
    /// by `regression_test.rs::focus_band_zero_damage_hit_does_not_arm_counter`.
    pub(crate) fn record_reactive_hit(
        &mut self,
        def_side: usize,
        def_slot: usize,
        category: MoveCategory,
        move_id: &str,
        dealt: u16,
    ) {
        let Some(r) = self.sides[def_side].pokemon[def_slot].reactive.as_mut() else {
            return;
        };
        let is_hp = move_id.starts_with("hiddenpower");
        let qualifies = if r.mirror {
            category == MoveCategory::Special && !is_hp
        } else {
            category == MoveCategory::Physical || is_hp
        };
        if qualifies && dealt > 0 {
            r.damage = Some(dealt.saturating_mul(2));
        }
    }

    /// The `(side, slot)` of an ACTIVE mon with the Damp ability (`gen3_ability_batch2_v1`), if
    /// any — the holder whose `onAnyTryMove` cancels Explosion/Self-Destruct. gen-3 singles: at
    /// most 2 actives; `onAny*` fires for BOTH sides, so an Explosion is blocked if EITHER active
    /// has Damp (incl. the user's OWN side). Returns the FIRST found (either side blocks
    /// identically — the `|cant|` names the holder). `None` ⇒ no active Damp mon (Explosion runs).
    pub(crate) fn damp_holder(&self, dex: &Dex) -> Option<(usize, usize)> {
        for s in 0..2 {
            let slot = self.sides[s].active;
            let mon = &self.sides[s].pokemon[slot];
            if !mon.fainted
                && dex.ability(&to_id(&mon.ability)).map(|a| a.blocks_explosion).unwrap_or(false)
            {
                return Some((s, slot));
            }
        }
        None
    }

    /// CONTACT_PROC + CONTACT recoil (`gen3_ability_batch2_v1`) — the DEFENDER's reactive
    /// `onDamagingHit` ability firing against the ATTACKER after a CONTACT hit that dealt damage.
    /// Called from `run_move`'s landed-hit tail (the `runEvent('DamagingHit')` position, AFTER
    /// `apply_secondaries`). `atk_side`/`atk_slot` = the ATTACKER (who gets statused / takes
    /// recoil); `def_side`/`def_slot` = the ability HOLDER (the DEFENDER).
    ///
    /// The draw model (PROBE-settled `harness/probe_contact_proc_{rng,lands}.js` +
    /// `probe_effectspore_sample.js`):
    ///   - **Status procs** (Static par / Poison Point psn / Flame Body brn — `sample=false`):
    ///     ONE `randomChance(num, den)` (one `random(den)`); on a PASS, `try_set_status(the-one-
    ///     status, attacker)` with the ATTACKER as target and the DEFENDER as the reflect source
    ///     (so gen-3 type/ability/already-statused gates apply, and gen3ou draws the SetStatus
    ///     shuffle — draw-free in the e2e customgame). The `randomChance` draws UNCONDITIONALLY
    ///     (even if the attacker is already-statused / type-immune — the gate is inside
    ///     trySetStatus, AFTER the roll).
    ///   - **Effect Spore** (`sample=true`): `randomChance(1,10)` gate; on a PASS one
    ///     `sample(["slp","par","psn"])` (a `random(3)`) picks the status → `try_set_status`.
    ///   - **Rough Skin** (`contact_recoil`): DRAW-FREE — `baseMaxhp/16` recoil to the attacker
    ///     (`this.damage`, no PRNG). A recoil KO faints the attacker via the normal machinery.
    /// Nothing fires for a non-CONTACT_PROC / non-Rough-Skin holder (draw-free no-op).
    pub(crate) fn apply_contact_proc(&mut self, atk_side: usize, atk_slot: usize, def_side: usize, def_slot: usize, dex: &Dex) {
        let ability = to_id(&self.sides[def_side].pokemon[def_slot].ability);
        let (contact_proc, contact_recoil, contact_attract) = match dex.ability(&ability) {
            Some(a) => (a.contact_proc.clone(), a.contact_recoil, a.contact_attract),
            None => (None, false, None),
        };
        if let Some(cp) = contact_proc {
            // The `randomChance(num, den)` gate — DRAWN unconditionally on a contact-damaging hit.
            let passed = self.prng.random_chance(cp.chance.0, cp.chance.1);
            if passed {
                // Which status: a single (Static/PoisonPoint/FlameBody) applies directly; a
                // `sample` (Effect Spore) draws ONE `random(len)` to pick.
                let status: &str = if cp.sample {
                    let idx = self.prng.random_below(cp.statuses.len() as u32) as usize;
                    &cp.statuses[idx]
                } else {
                    &cp.statuses[0]
                };
                // Apply to the ATTACKER; the DEFENDER (the ability holder) is BOTH the status
                // SOURCE (so a Synchronize attacker reflects it back — the choke point handles it)
                // AND the `[of]` in the emitted line. [EMIT] `|-status|<attacker>|<status>|[from]
                // ability: <Ability>|[of] <holder>` — the sim's `source.trySetStatus(status,
                // target)` runs INSIDE the ability's `onDamagingHit`, so `setStatus`'s null
                // `sourceEffect` falls back to `this.battle.effect` = the ABILITY → the status
                // `onStart`'s `sourceEffect.effectType === 'Ability'` gate is TRUE → the
                // `[from] ability` form. (`gen3_omniscient_byte_fuzz_v1`, byte-fuzzer-surfaced
                // golden `|-status|p1a: Metagross|par|[from] ability: Effect Spore|[of] p2a:
                // Breloom`; the port used to emit the BARE form — a stale conclusion the live sim
                // capture refuted.) Draws (the gen3ou clause shuffle) unchanged — emission-only.
                let status_owned = status.to_string();
                let ability_name =
                    dex.ability(&ability).map(|a| a.name.clone()).unwrap_or_else(|| ability.clone());
                self.try_set_status_impl(
                    atk_side, atk_slot, &status_owned, Some((def_side, def_slot)), false,
                    Some((def_side, def_slot, ability_name)), None, dex,
                );
            }
        } else if let Some((num, den)) = contact_attract {
            // CUTE CHARM (`gen3_ability_batch4_v1`): the SAME DamagingHit-position roll as the
            // status procs — `randomChance(1,3)` drawn UNCONDITIONALLY on a damaging contact
            // hit (the gender gate lives INSIDE attract.onStart: a same-gender / genderless
            // attacker still DRAWS the roll and the volatile fails draw-free — probe
            // `probe_cutecharm_attract_rng.js`). On a pass, add ATTRACT to the ATTACKER.
            let passed = self.prng.random_chance(num, den);
            if passed {
                self.try_add_attract(atk_side, atk_slot, def_side, def_slot, dex);
            }
        } else if contact_recoil {
            // ROUGH SKIN: DRAW-FREE recoil `baseMaxhp/16` to the ATTACKER (the sim's
            // `this.damage(source.baseMaxhp / 16, source, target)`). gen-3 `maxhp == baseMaxhp`.
            let recoil = (self.sides[atk_side].pokemon[atk_slot].maxhp / 16).max(1);
            let recoil = recoil.min(self.sides[atk_side].pokemon[atk_slot].hp);
            let recoil = self.focus_band_damage(atk_side, atk_slot, recoil, false, false, dex);
            if recoil > 0 {
                self.apply_damage(atk_side, atk_slot, recoil);
                // [EMIT] `|-damage|<attacker>|<HP>|[from] ability: Rough Skin|[of] <holder>`.
                if self.logging() {
                    let name = dex.ability(&ability).map(|a| a.name.clone()).unwrap_or_else(|| ability.clone());
                    let atk_ref = self.mon_ref(atk_side, atk_slot, dex);
                    let holder_ref = self.mon_ref(def_side, def_slot, dex);
                    let hp = self.hp_status(atk_side, atk_slot);
                    self.log.damage_of(&atk_ref, &hp, &Cause::Ability(name), &holder_ref);
                }
            }
        }
    }

    /// FOCUS BAND (`gen3_ability_batch4_v1`, `ItemData::survive_lethal`) — the `onDamage`
    /// (priority -40) handler: `randomChance(1,10) && damage >= target.hp && effect.effectType
    /// === 'Move'`. The JS `&&` order means the roll DRAWS FIRST — on EVERY Damage event into
    /// the holder (move hits, burn/psn/tox chips, sand/hail chips, the Leech Seed drain,
    /// Spikes, Struggle/Rough-Skin recoil, confusion self-hits) — while the survive-at-1-HP
    /// fires only when the roll passed AND the damage is lethal AND the effect is a MOVE
    /// (`is_move`: the damaging/fixed-damage hit + the confusion self-hit, whose gen4-inherited
    /// effect is `effectType: 'Move'`; a lethal chip/recoil still faints). NOT called for a
    /// sub-absorbed hit (the mon's Damage event never runs — no draw). Returns the (possibly
    /// hp-1-capped) damage; emits `|-activate|<mon>|item: Focus Band` on a survive.
    /// PROBE-settled `probe_focusband_rng.js` + `probe_focusband_confusion_rng.js`.
    /// ENDURE's survive-at-1 clamp (`gen3_move_coverage_batch6_v1`, the resolved gen-3
    /// `endure.onDamage` at priority **−10**): a **Move**-effect damage that would KO the
    /// endure-volatile holder (`damage >= hp`) is clamped to `hp − 1`, with
    /// `|-activate|<holder>|move: Endure` emitted BEFORE the `-damage` line. Applies to
    /// every qualifying Move hit — plain hits, FIXED damage (Seismic Toss → 1 HP, probe
    /// ED6), EVERY strike of a multihit (each strike `-activate`s + clamps, probe ED8),
    /// and the future-move resolve (class-inferred: the resolve's Damage event runs the
    /// onDamage handlers — the Focus-Band-rolls-there precedent). It does NOT guard
    /// non-Move damage (residual chips KO a 1-HP endurer — ED5; confusion self-hits are
    /// a Move-effect but SELF-inflicted and gen3 boards never pair them with Endure —
    /// the sim's gate is effectType-only, so we clamp there too via the caller sites,
    /// none of which route a self-hit here). DRAW-FREE. Every caller runs this BEFORE
    /// `focus_band_damage` (endure −10 precedes FB's −40 in the priority-DESC event).
    pub(crate) fn endure_clamp(&mut self, side: usize, slot: usize, dmg: u16, dex: &Dex) -> u16 {
        let mon = &self.sides[side].pokemon[slot];
        if !mon.endure || mon.fainted || dmg == 0 || dmg < mon.hp {
            return dmg;
        }
        // [EMIT] `|-activate|<holder>|move: Endure` — BEFORE the `-damage`.
        if self.logging() {
            let m = self.mon_ref(side, slot, dex);
            self.log.activate(&m, "move: Endure", None);
        }
        let clamped = self.sides[side].pokemon[slot].hp - 1;
        // When the endurer is ALREADY at 1 HP, the clamped damage is 0 (the mon stays at
        // 1) — the caller's `realized > 0` emission gate then SKIPS the `-damage` line the
        // sim STILL emits (`|-damage|<mon>|1/<max>` after `-activate move: Endure`; the
        // gen3 `spreadDamage` reports the survived HP even for a 0-HP-delta clamp — the
        // `gen3_omniscient_byte_fuzz_v1` Endure-at-1 repro rmroh04is_ab_8_4 dec225). Emit
        // it here (apply_damage(0) is a no-op, so the current HP == the post-apply HP).
        // For hp > 1 the caller emits it post-apply (clamped > 0), so this never double-emits.
        if clamped == 0 && self.logging() {
            let target = self.mon_ref(side, slot, dex);
            let hp = self.hp_status(side, slot);
            self.log.damage(&target, &hp, None);
        }
        clamped
    }

    /// The ATTRACT volatile's add (`attract.onStart`, `gen3_ability_batch4_v1`) — from a Cute
    /// Charm contact proc. The gates, ALL draw-free (the CC `randomChance(1,3)` was already
    /// drawn by the caller): already-attracted (addVolatile returns false, no re-add), a
    /// fainted/0-HP target, the GENDER gate (M<->F opposite pairs only; a same-gender or
    /// genderless pair fails — probe: F-into-F / genderless draw the roll but never attract),
    /// and OBLIVIOUS (the runEvent('Attract') block — probed draw-free). A target with an
    /// UNSPECIFIED gender (the sim would have SAMPLED one at construction — an init draw the
    /// port does not model) PANICS fail-loud. On success the volatile records the SOURCE
    /// (side, uid) — cleared when the source leaves the field or the holder switches out.
    fn try_add_attract(&mut self, side: usize, slot: usize, src_side: usize, src_slot: usize, dex: &Dex) {
        {
            let mon = &self.sides[side].pokemon[slot];
            if mon.attract.is_some() || mon.fainted || mon.hp == 0 {
                return;
            }
        }
        let need = |g: Option<char>, who: &str| -> char {
            g.unwrap_or_else(|| {
                panic!(
                    "attract gender-compare reached a mon with an UNSPECIFIED gender ({who}) — \
                     the sim samples one at construction (an unmodeled init draw); pin genders \
                     explicitly in any team that can reach Cute Charm."
                )
            })
        };
        let gt = need(self.sides[side].pokemon[slot].gender, "target");
        let gs = need(self.sides[src_side].pokemon[src_slot].gender, "source");
        let opposite = (gt == 'M' && gs == 'F') || (gt == 'F' && gs == 'M');
        if !opposite {
            return; // incompatible gender -> draw-free fail
        }
        if to_id(&self.sides[side].pokemon[slot].ability) == "oblivious" {
            return; // the runEvent('Attract') block -> draw-free fail
        }
        let src_uid = self.sides[src_side].pokemon[src_slot].uid;
        self.sides[side].pokemon[slot].attract = Some((src_side, src_uid));
        // [EMIT] `|-start|<mon>|Attract|[from] ability: Cute Charm|[of] <source>`.
        if self.logging() {
            let mon_ref = self.mon_ref(side, slot, dex);
            let src_ref = self.mon_ref(src_side, src_slot, dex);
            self.log
                .push_raw(format!("|-start|{mon_ref}|Attract|[from] ability: Cute Charm|[of] {src_ref}"));
        }
    }

    /// COLOR CHANGE (`gen3_ability_batch4_v1`) — the DEFENDER's `onDamagingHit` type override:
    /// on a damaging hit that reached the MON (NOT a sub-absorbed hit — the mon's DamagingHit
    /// never fires behind a sub, same as the contact procs; probe t2: a TBolt into the sub
    /// leaves Kecleon Normal), while the holder is ALIVE (no change on the KO hit), for a
    /// non-Status move with a real type (typeless `???` — Struggle, the confusion self-hit —
    /// never changes) that the holder doesn't already have: `types_override = [move.type]`.
    /// DRAW-FREE. The override feeds every later `mon_types` read (STAB / chart / status
    /// type-immunity / sand-chip immunity). Cleared on switch-out. Probe `probe_colorchange_rng.js`.
    pub(crate) fn apply_color_change(&mut self, side: usize, slot: usize, move_type: Option<Type>, dex: &Dex) {
        {
            let mon = &self.sides[side].pokemon[slot];
            if to_id(&mon.ability) != "colorchange" || mon.fainted || mon.hp == 0 {
                return;
            }
            let t = match move_type {
                Some(t) => t,
                None => return, // typeless ??? never changes
            };
            if mon_types(mon, dex).contains(&t) {
                return; // already that type -> no-op (no line)
            }
        }
        let t = move_type.expect("checked above");
        self.sides[side].pokemon[slot].types_override = Some(vec![t]);
        // [EMIT] `|-start|<mon>|typechange|<Type>|[from] ability: Color Change`.
        if self.logging() {
            let mon_ref = self.mon_ref(side, slot, dex);
            self.log
                .push_raw(format!("|-start|{mon_ref}|typechange|{}|[from] ability: Color Change", t.display_name()));
        }
    }

    /// Try to apply a major status `effect` to `foe`/`foe_slot`, mirroring gen-3
    /// `pokemon.trySetStatus`/`setStatus` (all the onTrySetStatus gates + the gen3ou SetStatus
    /// clause shuffle + the STATUS_IMMUNE abilities). `source` is `Some((side, slot))` when a
    /// FOE effect inflicted the status (a status move / a damaging move's secondary / a contact
    /// proc), `None` for a source-less / self-inflicted apply (or when the caller does not want
    /// the Synchronize reflect). When the status successfully applies to a **Synchronize** holder
    /// and `source` is a DIFFERENT mon, gen-3 `synchronize.onAfterSetStatus` REFLECTS the status
    /// back to the source (slp/frz EXEMPT; tox→psn) — draw-free in gen3customgame (the e2e
    /// format), and drawing the reflected status's own SetStatus shuffle in gen3ou. `gen3_ability_batch2_v1`.
    pub(crate) fn try_set_status(&mut self, foe: usize, foe_slot: usize, effect: &str, source: Option<(usize, usize)>, dex: &Dex) {
        self.try_set_status_impl(foe, foe_slot, effect, source, false, None, None, dex);
    }

    /// [`Self::try_set_status`] with the two Phase-3 emission nuances:
    /// - `announce_immune_block`: a `setStatus`-phase STATUS_IMMUNE ability's `onSetStatus`
    ///   handler emits its `|-immune|<target>|[from] ability: <A>` line ONLY when the source
    ///   effect is a status MOVE (`(effect as Move)?.status` — Thunder Wave / Will-O-Wisp /
    ///   Toxic / Hypnosis; byte-verified vs the status_immune_lines capture). A SECONDARY /
    ///   contact-proc / Tri-Attack block is SILENT (same gate, no line). Only the
    ///   standalone-status-move arm passes `true`.
    /// - `ability_reveal`: `Some((holder_side, holder_slot, ability_name))` when this apply's
    ///   `-status` line carries a `[from] ability: <ability_name>|[of] <holder>` reveal INSTEAD
    ///   of the plain form (at the same position). TWO cases share it: a **Synchronize** reflect
    ///   (`ability_name = "Synchronize"`, so the holder's Lum `-enditem`/`-curestatus` tail lands
    ///   AFTER it — byte-verified vs the synchronize_lum_rest capture) AND a **contact-proc**
    ///   status (Static/Poison Point/Flame Body/Effect Spore — the sim's `source.trySetStatus`
    ///   runs inside the ability's `onDamagingHit`, so `setStatus`'s `sourceEffect` falls back to
    ///   `this.battle.effect` = the ABILITY → `<status>.onStart` renders `[from] ability: <A>|[of]
    ///   <holder>`; `gen3_omniscient_byte_fuzz_v1`, byte-fuzzer-surfaced golden
    ///   `|-status|p1a: Metagross|par|[from] ability: Effect Spore|[of] p2a: Breloom` — the port
    ///   used to emit the BARE form, a stale-conclusion bug the live sim capture refuted).
    pub(crate) fn try_set_status_impl(
        &mut self,
        foe: usize,
        foe_slot: usize,
        effect: &str,
        source: Option<(usize, usize)>,
        announce_immune_block: bool,
        ability_reveal: Option<(usize, usize, String)>,
        // The SOURCE-MOVE display name, when this apply comes from a status MOVE (the
        // standalone status-move arm passes it; the secondary / ability / Rest paths pass
        // `None`). Consumed ONLY by the SLEEP emit: gen-3 `slp.onStart` carries `[from]
        // move: <Name>` when the source effect is a Move (`gen3_omniscient_byte_fuzz_v1`
        // FORM 5 — Spore/Hypnosis/Sing/…; par/psn/brn/frz from a move emit BARE, per the
        // dist `conditions.js` per-status onStart). Draw-free / observation-only.
        src_move: Option<&str>,
        dex: &Dex,
    ) {
        let mon = &self.sides[foe].pokemon[foe_slot];
        // (3) A KO'd / 0-HP target cannot be statused (setStatus: `if (!this.hp)`).
        if mon.fainted || mon.hp == 0 {
            return;
        }
        // (1) Already-statused → fail (a mon with ANY major status can't take another).
        if mon.status.is_some() {
            return;
        }
        // (2) Status-type immunity — the gen-3 rules the repo type chart lacks
        //     (multipliers-only): brn→Fire, frz→Ice, psn/tox→Poison&Steel; par/slp none.
        //     `setStatus` returns at `runStatusImmunity` BEFORE `runEvent('SetStatus')`,
        //     so a type-immune status does NOT draw the SetStatus handler-sort shuffle.
        //     [EMIT] `|-immune|<target>` (`gen3_omniscient_byte_fuzz_v1` FORM 3): the sim's
        //     `setStatus` emits `-immune` at `runStatusImmunity` ONLY when the source effect
        //     is a status MOVE (`sourceEffect?.status` — Toxic→Steel/Poison, Will-O-Wisp→Fire,
        //     Poison Powder→Poison; a SECONDARY-inflicted status has no top-level `move.status`
        //     → silent). `announce_immune_block` == "the source is a status move" (only the
        //     standalone status-move arm passes it), so it's the exact `sourceEffect?.status`
        //     gate. Draw-free / observation-only.
        let types = mon_types(mon, dex);
        if status_type_immune(effect, &types) {
            if announce_immune_block && self.logging() {
                let target = self.mon_ref(foe, foe_slot, dex);
                self.log.immune(&target);
            }
            return;
        }
        // (2a) SUN → freeze immunity (`gen3_sun_freeze_immunity_v1`). The base `sunnyday`
        //      weather registers `onImmunity(type) { if effectiveWeather()!=='sunnyday'
        //      return; if type==='frz' return false; }` (conditions.ts) — so while the
        //      field weather is Sun (Drought / Sunny Day), a mon CANNOT be frozen.
        //      `setStatus` → `runStatusImmunity('frz')` → `runEvent('Immunity', 'frz')`
        //      returns FALSE, so the freeze is NOT applied. This is the SAME `runStatusImmunity`
        //      position as the type immunity above — checked BEFORE `runEvent('SetStatus')`
        //      — so like the type gate it SKIPS the gen3ou clause shuffle, and the sun
        //      `onImmunity` handler is itself DRAW-FREE. So the freeze SECONDARY's
        //      `random(100)` still fires (the seed matches) but the freeze must simply not
        //      land — the A/B-fuzz "ice-freeze cluster" (196 repros, expected=None
        //      got=Some(Freeze), seed matching) was this missing gate. Probe-verified
        //      (`harness/probe_sun_freeze_immunity.js`): under Drought the same seed that
        //      freezes in no-sun leaves the mon UN-frozen with an IDENTICAL draw count
        //      (customgame), and ONE FEWER draw in gen3ou (the skipped clause shuffle).
        //      An ALREADY-frozen mon PERSISTS under sun (frz has no onWeather cure) — this
        //      gates only the APPLICATION, never a thaw. (No modeled gen-3 mon carries a
        //      weather-suppressing ability — Air Lock / Cloud Nine — is now MODELED
        //      (`gen3_ability_batch1_v1`), so this gates on the EFFECTIVE weather: an Air Lock
        //      / Cloud Nine mon on the field suppresses the sun → a freeze CAN land under
        //      raw-Sun-but-negated weather (matching `sunnyday.onImmunity` reading
        //      `effectiveWeather()`).
        if effect == "frz" && self.effective_weather(dex) == Some(Weather::Sun) {
            return;
        }
        // (2d) STATUS_IMMUNE ability, `immunity` PHASE (`gen3_status_immune_v1`, data-driven
        //      via `AbilityData.status_immune`) — an ability whose block is an
        //      `onImmunity(status)` that returns false at `runStatusImmunity`, BEFORE
        //      `runEvent('SetStatus')`. The ONLY gen-3 member is **Magma Armor** (frz) —
        //      structurally the SAME position as the sun-freeze gate above (probe-settled:
        //      `harness/probe_statusimmune_magmaarmor.js` — MA blocks at the Immunity event,
        //      the SetStatus event is NEVER reached, so NO clause shuffle fires in gen3ou).
        //      So it MUST be gated HERE, before the shuffle — DRAW-FREE. (A secondary-freeze
        //      move's `random(100)` already drew upstream; MA just blocks the application.)
        let ability = to_id(&mon.ability);
        if let Some(si) = self.status_immune_of(&ability, dex) {
            if si.phase == crate::dex::StatusImmunePhase::Immunity && si.blocks(effect) {
                return;
            }
        }

        // --- The `runEvent('SetStatus')` HANDLER-SORT SHUFFLE (the gen3ou-only draw):
        //     in formats with the `Standard` clauses (Sleep Clause Mod + Freeze Clause
        //     Mod), `setStatus` calls `runEvent('SetStatus')`, which GATHERS the 2
        //     format-clause `onSetStatus` handlers. At equal order/priority/speed they
        //     TIE → a size-2 Fisher-Yates speed-sort shuffle draws ONE `random(0,2)`
        //     EVERY time the event is reached (a status that PASSED hp/already-statused/
        //     type-immunity — incl. one the clause OR a STATUS_IMMUNE ability then BLOCKS).
        //     In gen3customgame (no clauses → the ONLY handler is the ability's own, size-1
        //     → NO tie) NO shuffle is drawn.
        //
        //     THE CRUX (probe-settled — `harness/probe_statusimmune_shuffle_size.js`): a
        //     `setStatus`-phase STATUS_IMMUNE ability (Limber/Insomnia/…) DOES register an
        //     `onSetStatus` handler → in gen3ou the SetStatus event gathers **3** handlers.
        //     But `speedSort` sorts `order→priority→speed→subOrder`, and the ability handler
        //     carries a DEFINED `speed` (the mon's speed, 96 in the probe) while the two
        //     clause handlers have `speed=undefined` → the ability sorts into its OWN group
        //     (index 0, no tie), leaving the **2 clauses a size-2 tie** → the Fisher-Yates
        //     `shuffle(list, 1, 3)` draws EXACTLY ONE `random` — IDENTICAL to the control's
        //     size-2 `shuffle(list, 0, 2)`. So a STATUS_IMMUNE ability does NOT change the
        //     draw count; `set_status_event_shuffle()` (one draw) is correct for it too. Only
        //     a genuinely UNMODELED `onSetStatus` ability (one not in the `status_immune`
        //     table AND without a known size-2-preserving sort) could desync — FAIL-LOUD on
        //     that (a future scenario can never silently break). ---
        if self.sleep_clause {
            if self.ability_unmodeled_on_set_status(&ability, dex) {
                panic!(
                    "status target ability {ability:?} has an UNMODELED onSetStatus handler \
                     — its participation in the gen3ou SetStatus handler-sort shuffle is not \
                     modeled (the STATUS_IMMUNE members sort into their own speed group so \
                     the 2-clause tie stays size-2, but this ability is NOT one of them). \
                     Probe its sort position + model it before using it under a clause format."
                );
            }
            // The size-2 clause-handler tie shuffle: exactly one `random(0,2)`
            // (bit-identical to a 2-active eachEvent shuffle on a tie). A `setStatus`-phase
            // STATUS_IMMUNE ability adds a distinctly-sorted 3rd handler that does NOT change
            // this count (the crux above).
            self.set_status_event_shuffle();
        }

        // (2b) STATUS_IMMUNE ability, `setStatus` PHASE (`gen3_status_immune_v1`, data-driven)
        //      — an `onSetStatus` handler that RETURNS false INSIDE the SetStatus event the
        //      shuffle already drew: Insomnia/Vital Spirit block slp; Limber par; Immunity
        //      psn/tox; Water Veil brn. DRAW-FREE (the block is a handler return; the shuffle
        //      already fired above). (Magma Armor's frz block is the `immunity`-phase gate
        //      above, BEFORE the shuffle — handled at (2d), which stays SILENT: the sim's
        //      `runStatusImmunity(status.id)` passes no message, and no direct gen-3 freeze
        //      move exists to capture a line against.)
        //      [EMIT] `|-immune|<target>|[from] ability: <Ability>` — the handler's own
        //      announce, ONLY for a status-MOVE source (`announce_immune_block`; a blocked
        //      SECONDARY is silent — the `(effect as Move)?.status` gate). Byte-verified vs
        //      the status_immune_lines capture (Phase 3).
        if let Some(si) = self.status_immune_of(&ability, dex) {
            if si.phase == crate::dex::StatusImmunePhase::SetStatus && si.blocks(effect) {
                if announce_immune_block && self.logging() {
                    let display = dex
                        .ability(&ability)
                        .map(|a| a.name.clone())
                        .unwrap_or_else(|| ability.clone());
                    let target = self.mon_ref(foe, foe_slot, dex);
                    self.log.immune_from_ability(&target, &display);
                }
                return;
            }
        }
        // (2c) SLEEP CLAUSE MOD (a clause `onSetStatus` RETURN false — DRAW-FREE; the
        //      handler-sort shuffle above already drew). ONLY in formats that carry it
        //      (gen3ou via `Standard`; NOT gen3customgame). A sleep move FAILS if any
        //      LIVING mon on the TARGET's side already has a (non-self-inflicted) sleep.
        //      All sleep in this engine is foe-inflicted (Rest is out of scope), so
        //      simply: fail if any living slot is asleep. The block is BEFORE the
        //      onStart → NO random(2,6) on a clause block (the shuffle still drew).
        if effect == "slp" && self.sleep_clause && self.side_has_sleeper(foe) {
            // [EMIT] the Sleep Clause Mod block lines (`gen3_omniscient_byte_fuzz_v1`,
            // gen3ou): `|-message|Sleep Clause Mod activated.` + a per-battle-deduped
            // `|-hint|…` (rulesets.js:6028-6029). Draw-free / observation-only.
            if self.logging() {
                self.log.push_raw("|-message|Sleep Clause Mod activated.");
                self.log.hint("Sleep Clause Mod prevents players from putting more than one of their opponent's Pok\u{e9}mon to sleep at a time", false); // rulesets.ts:1395 — no `once` → fires on EVERY block
            }
            return;
        }
        // (2c-frz) FREEZE CLAUSE MOD (`gen3_freeze_clause_v1` — the handler-completeness
        //      audit's second real miss; the resolved rule's `onSetStatus` returns false
        //      INSIDE the SetStatus event, so the handler-sort shuffle above already
        //      drew — the block itself is DRAW-FREE, probe
        //      `harness/probe_freeze_clause_rng.js`: a blocked second freeze's turn draw
        //      count == a landed freeze's). ONLY in clause formats (gen3ou etc. — the
        //      same `sleep_clause` flag; gen3's Standard ruleset carries BOTH clauses).
        //      A freeze FAILS if any LIVING mon on the TARGET's side is already frozen.
        //      A FAINTED mon never counts — the sim sets `status = 'fnt'` on faint, so
        //      the rule's `pokemon.status === "frz"` scan can only match living mons
        //      (probe B). The rule's `source?.isAlly(target)` self-exemption is moot:
        //      gen3 has no self-inflicted freeze.
        if effect == "frz" && self.sleep_clause && self.side_has_frozen(foe) {
            // [EMIT] the Freeze Clause Mod block line (gen3ou): `|-message|Freeze Clause
            // activated.` (rulesets.js:6099 — NO hint, unlike Sleep Clause). Draw-free.
            if self.logging() {
                self.log.push_raw("|-message|Freeze Clause activated.");
            }
            return;
        }
        // Apply the status. SLEEP draws the onStart random(2,6) duration (1-4 turns,
        // gen-3 `slp.onStart`); Toxic starts at stage 0 (draw-free; the residual ramps
        // it to 1 before the first chip — mirrors `statusState.stage`); all others
        // draw-free.
        let new = match effect {
            "par" => Status::Paralysis,
            "brn" => Status::Burn,
            "frz" => Status::Freeze,
            "psn" => Status::Poison,
            "slp" => {
                // gen-3 `this.random(2,6)` (= `random_range(2,6)` ∈ {2,3,4,5}).
                let dur = self.prng.random_range(2, 6) as u8;
                Status::Sleep(dur)
            }
            // gen-3 `tox.onStart` sets `effectState.stage = 0`; the first RESIDUAL
            // ramps it to 1 (and chips 1×). `Status::Toxic(stage)` mirrors the sim's
            // `statusState.stage` EXACTLY (so the differential can assert it), and
            // `apply_status_dot` does the `if stage<15 stage++` ramp BEFORE the damage.
            "tox" => Status::Toxic(0),
            _ => return,
        };
        // A FRESH status gets a FRESH `statusState` — the slp `skippedTime` starts at 0
        // (`gen3_move_coverage_batch5_v1`; meaningless for the non-sleep statuses).
        self.sides[foe].pokemon[foe_slot].sleep_skipped = 0;
        // FOE-inflicted sleep (a sleep move / Effect Spore — never Rest, which sets its
        // own flag): this DOES count toward the target side's Sleep Clause Mod cap
        // (`gen3_sleep_clause_self_rest_exempt_v1`). Always reset here so a mon re-slept by
        // a foe after a prior self-Rest sleep is correctly counted.
        self.sides[foe].pokemon[foe_slot].sleep_from_rest = false;
        self.sides[foe].pokemon[foe_slot].status = Some(new);
        // [EMIT] `|-status|<target>|<status>` — a foe-inflicted status (secondary /
        // standalone status move / Tri Attack). NO `[from]` (only Rest's self-inflict
        // carries `[from] move: Rest`, emitted by `run_rest` directly). A SYNCHRONIZE
        // REFLECT apply (`sync_reveal`) instead carries the `[from] ability: Synchronize|
        // [of] <holder>` reveal — emitted HERE so the holder's Lum tail (below) follows
        // it in the byte-verified order. Observation-only.
        if self.logging() {
            let target = self.mon_ref(foe, foe_slot, dex);
            match &ability_reveal {
                // SLEEP is the gen3 EXCEPTION: `data/mods/gen3/conditions.js::slp.onStart`
                // ONLY renders `[from] move: <Name>` for a MOVE source and falls to a PLAIN
                // `else` for everything else — it DROPS the base-`conditions.js` `[from]
                // ability` branch every OTHER status keeps. So Effect Spore inflicting sleep
                // emits a BARE `|-status|<mon>|slp` (SIM-PROBED); the port used to over-attribute
                // it `[from] ability: Effect Spore|[of]` (`gen3_omniscient_byte_fuzz_v1`).
                Some(_) if effect == "slp" => self.log.status(&target, effect, None),
                Some((h_side, h_slot, ability_name)) => {
                    let holder_ref = self.mon_ref(*h_side, *h_slot, dex);
                    self.log.status_from_ability(&target, effect, ability_name, &holder_ref);
                }
                // SLEEP from a status MOVE carries `[from] move: <Name>` (FORM 5); every
                // OTHER status (par/psn/brn/frz) — and sleep from a non-move source — is
                // BARE (the per-status `onStart` in dist `conditions.js`).
                None if effect == "slp" && src_move.is_some() => {
                    self.log.status(&target, effect, Some(&Cause::Move(src_move.unwrap().to_string())));
                }
                None => self.log.status(&target, effect, None),
            }
        }

        // --- SYNCHRONIZE (`gen3_ability_batch2_v1`, `synchronize.onAfterSetStatus`) — when the
        //     holder (`foe`) is inflicted a MAJOR status by a FOE `source`, REFLECT it back to
        //     that source. The resolved gen3 handler:
        //       if (!source || source === target) return;  // no self-inflict / source-less
        //       let id = status.id; if (id==='slp'||id==='frz') return;  // slp/frz EXEMPT
        //       if (id==='tox') id='psn';                   // tox reflects as psn
        //       source.trySetStatus(id, target);
        //     The reflected `trySetStatus` runs with `source=None` (Synchronize does NOT chain a
        //     re-reflect — the reflected apply's own onAfterSetStatus has `source===target`
        //     false but the ORIGINAL holder isn't a Synchronize-reflect target of itself; a
        //     source-less reflect also avoids any infinite ping-pong). DRAW-FREE in gen3customgame
        //     (the reflected status draws no clause shuffle); in gen3ou it draws the reflected
        //     status's OWN SetStatus 2-clause shuffle (via the recursive `try_set_status`).
        //     PROBE-settled: `harness/probe_synchronize_rng.js`.
        if let Some((src_side, src_slot)) = source {
            if (src_side, src_slot) != (foe, foe_slot)
                && dex.ability(&to_id(&self.sides[foe].pokemon[foe_slot].ability)).map(|a| a.synchronize).unwrap_or(false)
            {
                // slp/frz are exempt; tox reflects as psn; the rest reflect as-is.
                let reflect = match effect {
                    "slp" | "frz" => None,
                    "tox" => Some("psn"),
                    other => Some(other),
                };
                if let Some(refl) = reflect {
                    // The `[from] ability: Synchronize|[of] <holder>` reveal is emitted by the
                    // reflected `try_set_status` as a plain `|-status|` — Showdown adds the
                    // `[from] ability: Synchronize` via the effect context. We mirror the reveal
                    // form directly for the source (observation-only; the seed suites are
                    // unaffected since the reflected apply's DRAW behaviour is what matters).
                    self.apply_synchronize_reflect(foe, foe_slot, src_side, src_slot, refl, dex);
                }
            }
        }

        // --- LUM's IMMEDIATE eat (`gen3_berry_trace_shedskin_v1`, `lum.onAfterSetStatus`
        //     priority -1 — AFTER Synchronize's priority-0 reflect, probe-verified line
        //     order `|-status|holder| → |-status|source|…Synchronize| → |-enditem| →
        //     |-curestatus|`). DRAW-FREE. The single-status cure berries do NOT fire here
        //     (no onAfterSetStatus) — they wait for the next Update site. ---
        self.berry_after_set_status(foe, foe_slot, dex);
    }

    /// Reflect a Synchronize holder's status back to its SOURCE (`gen3_ability_batch2_v1`).
    /// Applies `refl` to `(src_side, src_slot)` via the status choke point with `source=None`
    /// (no re-reflect chain) and emits the `|-status|…|[from] ability: Synchronize|[of] <holder>`
    /// reveal. Separated so the recursive `try_set_status` call is explicit + the emit form is
    /// the Synchronize one (a foe status inflicted by the ability, not a bare move status).
    fn apply_synchronize_reflect(
        &mut self,
        holder_side: usize,
        holder_slot: usize,
        src_side: usize,
        src_slot: usize,
        refl: &str,
        dex: &Dex,
    ) {
        // The recursive apply. `source=None` → no re-reflect (Synchronize doesn't ping-pong; the
        // reflected mon isn't a Synchronize target of the holder). This draws the reflected
        // status's own SetStatus shuffle in gen3ou / draw-free in gen3customgame, and applies the
        // gen-3 type/ability/already-statused gates on the SOURCE. `sync_reveal` makes the
        // apply's own `-status` emit carry the `[from] ability: Synchronize|[of] <holder>`
        // reveal (a blocked/no-op'd reflect emits nothing), and lets the SOURCE's own Lum
        // `-enditem`/`-curestatus` tail fire INSIDE the apply, AFTER the reveal — the
        // byte-verified Phase-3 interleave (`-status holder → -status source [from]
        // Synchronize → -enditem → -curestatus`).
        self.try_set_status_impl(
            src_side, src_slot, refl, None, false,
            Some((holder_side, holder_slot, "Synchronize".to_string())), None, dex,
        );
    }

    /// The STATUS_IMMUNE params of `ability_id` (`gen3_status_immune_v1`, data-driven via
    /// `AbilityData.status_immune`) — the class of abilities that grant immunity to a
    /// specific MAJOR status (Limber par / Insomnia+Vital Spirit slp / Immunity psn,tox /
    /// Water Veil brn via `onSetStatus`; Magma Armor frz via `onImmunity`). `None` for an
    /// ability with no status immunity (or an unknown id). A cheap dex lookup — the caller
    /// checks `.phase` + `.blocks(status)`.
    fn status_immune_of<'a>(&self, ability_id: &str, dex: &'a Dex) -> Option<&'a crate::dex::StatusImmune> {
        dex.ability(ability_id).and_then(|a| a.status_immune.as_ref())
    }

    /// Whether `ability_id` has an `onSetStatus` handler the port does NOT model for the
    /// gen3ou SetStatus handler-sort shuffle. The MODELED `onSetStatus` abilities are exactly
    /// the `setStatus`-phase STATUS_IMMUNE members (Limber/Insomnia/Vital Spirit/Immunity/
    /// Water Veil) — probe-proven to sort into their OWN speed group so the 2-clause tie stays
    /// size-2 (`set_status_event_shuffle`'s one draw is unchanged). Any OTHER gen-3 ability
    /// with an `onSetStatus` (Leaf Guard — num 102, not even gen-3-legal; Synchronize's
    /// `onAfterSetStatus` is a DIFFERENT event) would need its own sort-position probe, so we
    /// FAIL-LOUD on it under a clause format. Magma Armor is NOT here (it blocks via
    /// `onImmunity`, phase=Immunity, so it registers NO SetStatus handler).
    fn ability_unmodeled_on_set_status(&self, ability_id: &str, dex: &Dex) -> bool {
        // The set of abilities whose onSetStatus participation IS modeled = the
        // setStatus-phase STATUS_IMMUNE members. Everything else with an onSetStatus is
        // unmodeled. We can't enumerate "has an onSetStatus" from the data (the JS callback
        // is invisible), so we hardcode the gen-3 abilities KNOWN to carry one and check
        // whether each is modeled. In gen-3 the onSetStatus abilities are exactly the 5
        // setStatus-phase STATUS_IMMUNE members (probe-enumerated: Limber/Insomnia/Immunity/
        // Water Veil/Vital Spirit) + Synchronize (which uses onAfterSetStatus, a DIFFERENT
        // event that does NOT add a SetStatus handler) — so the modeled check is exactly the
        // status_immune-with-setStatus-phase membership.
        let known_on_set_status = matches!(
            ability_id,
            "limber" | "insomnia" | "immunity" | "waterveil" | "vitalspirit"
        );
        if !known_on_set_status {
            return false;
        }
        // Modeled iff it is a setStatus-phase STATUS_IMMUNE member (all 5 above are).
        !matches!(
            self.status_immune_of(ability_id, dex).map(|si| si.phase),
            Some(crate::dex::StatusImmunePhase::SetStatus)
        )
    }

    /// Whether `side` has a living mon (active OR bench) asleep from a FOE-inflicted
    /// sleep — the gen3ou Sleep Clause check. A mon asleep from its OWN **Rest**
    /// (`sleep_from_rest`) is EXEMPT and does NOT count (the sim's
    /// `!statusState.source?.isAlly` gate; `gen3_sleep_clause_self_rest_exempt_v1`,
    /// R15-P1). NOTE Rest IS modeled (batch 5) — the old "Rest is out of scope, so an
    /// asleep mon always counts" premise was false and bred R15-P1.
    fn side_has_sleeper(&self, side: usize) -> bool {
        // Sleep Clause Mod counts a living sleeper ONLY if its sleep was FOE-inflicted
        // (`!pokemon.statusState.source?.isAlly(pokemon)`, `rulesets.ts`
        // `sleepclausemod.onSetStatus`): a mon that put ITSELF to sleep via **Rest**
        // (`sleep_from_rest`) is EXEMPT and does NOT block a foe sleep move on a different
        // mon (probe-verified — a self-Rested Zapdos does not block a foe's Spore on
        // Suicune). `gen3_sleep_clause_self_rest_exempt_v1` (R15-P1).
        self.sides[side].pokemon.iter().any(|m| {
            m.hp > 0
                && !m.fainted
                && matches!(m.status, Some(Status::Sleep(_)))
                && !m.sleep_from_rest
        })
    }

    /// Any LIVING mon on `side` frozen — the Freeze Clause Mod scan
    /// (`gen3_freeze_clause_v1`). A fainted mon's sim status is `'fnt'`, never `'frz'`
    /// (probe `harness/probe_freeze_clause_rng.js` B), so only living mons count.
    fn side_has_frozen(&self, side: usize) -> bool {
        self.sides[side]
            .pokemon
            .iter()
            .any(|m| m.hp > 0 && !m.fainted && matches!(m.status, Some(Status::Freeze)))
    }
}
