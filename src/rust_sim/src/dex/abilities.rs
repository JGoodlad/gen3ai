//! Ability data — `{name, num}` plus the DATA-DRIVEN MECHANICS FRAMEWORK's
//! structured `dmgMod` damage-modifier params (`gen3_item_mechanics_v1`, ability side).
//!
//! The `dmgMod` field is emitted by `tools/pokemon_data_extractor/sync.py` from a
//! curated table derived from the RESOLVED `Dex.mod('gen3')` (the mod-chain law: gen3
//! resolves through gen4, which REPLACES/DELETES handlers — so probe the resolved
//! dist, never a raw `.ts`). Drift gate:
//! `node src/rust_sim/harness/dump_gen3_mechanics.js --check`.
//!
//! The engine consumes these via `turn.rs::resolve_atk_stat_mods` /
//! `resolve_def_stat_mods` / `resolve_bp_mods` — ONE generic path per class, exactly
//! like the item folds. The wired class members (Phase 2):
//!   - **pinch BP** (Torrent/Blaze/Overgrow/Swarm): `fold=basePower`, `type`, `pinch`
//!     — a `chainModify(1.5)` on the BASE-POWER chain gated on the move's type AND the
//!     user at `hp <= maxhp/3` (bit-exactly `3*hp <= maxhp` for integer hp/maxhp).
//!   - **Atk** (Huge/Pure Power): `fold=atk`, unconditional — `onModifyAtk chainModify(2)`
//!     (PHYSICAL only, since ModifyAtk fires only on the physical stat).
//!   - **Atk when statused** (Guts): `fold=atk`, `whenStatused` — `chainModify(1.5)` on
//!     the ModifyAtk chain when the user has a major status (the burn-halve is
//!     SUPPRESSED separately in `modify_damage` via `Combatant::has_guts`).
//!   - **Def when statused** (Marvel Scale): `fold=def`, `whenStatused` — `chainModify(1.5)`
//!     on the ModifyDef chain when the DEFENDER has a major status (physical Def only).
//!
//! Two members stay DATA-ONLY (parsed, not wired here):
//!   - **Hustle** (`fold=atk`, `direct`) — a `this.modify(atk, 1.5)` (NOT chainModify) that
//!     ALSO carries an ×0.8 physical-accuracy side; it MUST ship WITH the accuracy
//!     pipeline (its Atk half alone would desync the to-hit roll). Left `direct`, unwired.
//!   - **Thick Fat** (`fold=sourceBasePower`) — already modeled by the port's dedicated
//!     `DamageContext::defender_thick_fat` (an `onSourceBasePower ×0.5` for Ice/Fire); the
//!     `dmgMod` is recorded for completeness but the engine keeps its existing hook.

use crate::dex::types::Type;
use crate::json::Json;
use std::collections::HashMap;

/// Which stat/BP chain a [`DmgMod`] folds into (the resolved gen3 handler shape).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmgFold {
    /// `onBasePower chainModify(x)` — the BASE-POWER chain (the pinch family, gated by
    /// the move type + the pinch-HP condition).
    BasePower,
    /// `onModifyAtk chainModify(x)` — the offensive ATTACK stat chain (Huge/Pure Power
    /// unconditional; Guts when statused). PHYSICAL only.
    Atk,
    /// `onModifyDef chainModify(x)` — the DEFENDER's physical Def stat chain (Marvel
    /// Scale when the defender is statused).
    Def,
    /// `onSourceBasePower chainModify(x)` — the ATTACKER'S base power halved by the
    /// DEFENDER's ability for gated types (Thick Fat, Ice/Fire ×0.5). The port models
    /// this via `DamageContext::defender_thick_fat`; recorded here for completeness.
    SourceBasePower,
}

/// The damage-modifier params of a DMG_MOD ability (`dmgMod` in `gen3_abilities.json`).
#[derive(Debug, Clone)]
pub struct DmgMod {
    /// The exact multiplier as a rational (`trunc(num*4096/den)` == Showdown's fixed
    /// point). `[3,2]`=×1.5, `[2,1]`=×2, `[1,2]`=×0.5.
    pub num: u64,
    pub den: u64,
    pub fold: DmgFold,
    /// The gated move type(s). One `type` (the pinch family: Water/Fire/Grass/Bug) or a
    /// list (`types` — Thick Fat's Ice/Fire). Empty ⇒ type-agnostic (Huge/Pure/Guts/Marvel).
    pub types: Vec<Type>,
    /// Fires only when the user is in a pinch (`hp <= maxhp/3`).
    pub pinch: bool,
    /// Fires only when the relevant mon has a major status (Guts: attacker; Marvel
    /// Scale: defender).
    pub when_statused: bool,
    /// The `this.modify` DIRECT-replace shape (Hustle). Left unwired this phase — a
    /// `direct` member is NOT resolved into the chain folds (it needs the accuracy
    /// pipeline). Recorded so the drift gate stays exact + the wiring phase can find it.
    pub direct: bool,
}

/// WHICH gen-3 event does a STATUS_IMMUNE ability's block fire in — the draw-model
/// determinant (`gen3_status_immune_v1`, PROBE-settled by
/// `harness/probe_statusimmune_*.js`). The two phases block at DIFFERENT points in
/// `pokemon.ts::setStatus`, which matters ONLY in a clause-carrying format (gen3ou), where
/// `runEvent('SetStatus')` draws a size-2 clause tie-shuffle:
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusImmunePhase {
    /// `onSetStatus` — the handler returns `false` INSIDE `runEvent('SetStatus')` (Limber /
    /// Insomnia / Vital Spirit / Immunity / Water Veil). The SetStatus event IS reached, so
    /// in gen3ou its 2-clause tie-shuffle STILL DRAWS (the ability handler sorts into its own
    /// speed group — its defined `speed` beats the clauses' `undefined` — so the 2 clauses
    /// remain a size-2 tie → 1 draw, IDENTICAL to a normal status apply). The block itself is
    /// DRAW-FREE. Model: fire the shuffle THEN block draw-free.
    SetStatus,
    /// `onImmunity(status)` — the handler returns `false` at `runStatusImmunity`, BEFORE
    /// `runEvent('SetStatus')` (Magma Armor → frz; the same event/position as the base Sun
    /// weather's frz `onImmunity`). So the SetStatus event — hence its clause shuffle — is
    /// NEVER reached. Model: block BEFORE the shuffle, DRAW-FREE (exactly like the sun-freeze
    /// gate).
    Immunity,
}

/// STATUS_IMMUNE class params (`statusImmune` in `gen3_abilities.json`) — an ability that
/// grants immunity to one or more specific MAJOR statuses. The `statuses` are the gen-3
/// status ids (`par`/`slp`/`psn`/`tox`/`brn`/`frz`); `phase` is the resolved-dist handler
/// that does the block (see [`StatusImmunePhase`]).
#[derive(Debug, Clone)]
pub struct StatusImmune {
    /// The blocked major-status ids (e.g. `["psn","tox"]` for Immunity).
    pub statuses: Vec<String>,
    pub phase: StatusImmunePhase,
}

impl StatusImmune {
    /// Whether this ability grants immunity to `status` (a `to_id`-normalized gen-3 status
    /// id — `par`/`slp`/`psn`/`tox`/`brn`/`frz`).
    pub fn blocks(&self, status: &str) -> bool {
        self.statuses.iter().any(|s| s == status)
    }
}

/// CONTACT_PROC class params (`contactProc` in `gen3_abilities.json`, `gen3_ability_batch2_v1`)
/// — a reactive `onDamagingHit` ability that, when the HOLDER is hit by a CONTACT move,
/// draws `randomChance(chance)` and (on a pass) inflicts a status on the ATTACKER. PROBE-settled
/// draw model (`harness/probe_contact_proc_{rng,lands}.js` + `probe_effectspore_sample.js`):
/// the randomChance draws INSIDE `runEvent('DamagingHit')` (gen<5, AFTER the move's own
/// `secondaries()`), on a contact hit that dealt damage (incl. behind a SUBSTITUTE, and on a
/// KO). The status lands on the ATTACKER with the gen-3 type/ability/already-statused gates.
#[derive(Debug, Clone)]
pub struct ContactProc {
    /// The inflictable status id(s). ONE for Static (par) / Poison Point (psn) / Flame Body
    /// (brn) — applied directly. THREE for Effect Spore (`["slp","par","psn"]`) — the `sample`
    /// list (one is chosen by `random(len)`).
    pub statuses: Vec<String>,
    /// The proc chance as `[numerator, denominator]` — `[1,3]` for Static/PoisonPoint/FlameBody,
    /// `[1,10]` for Effect Spore. Drawn as `randomChance(num, den)` (one `random(den)`).
    pub chance: (u32, u32),
    /// `true` ⇒ Effect Spore's NESTED draw: on a `randomChance` PASS, one `sample(statuses)` (a
    /// `random(statuses.len())`) picks WHICH of `statuses` to inflict. `false` ⇒ the single
    /// `statuses[0]` is inflicted directly (no sample draw).
    pub sample: bool,
}

/// An ability record: `{id, name, num}` + the structured `dmgMod` mechanics field.
/// (`name`/`num` mirror the pre-framework `NamedNum` so the handful of `dex.ability()`
/// callers that only read those keep compiling.)
#[derive(Debug, Clone)]
pub struct AbilityData {
    pub id: String,
    pub num: u16,
    pub name: String,
    /// DMG_MOD class params (the pinch family / Huge-Pure / Guts / Marvel Scale / Thick
    /// Fat / Hustle). `None` for an ability with no damage modifier.
    pub dmg_mod: Option<DmgMod>,
    /// ACCURACY class params (Compound Eyes ×1.3 / Sand Veil ×0.8-in-sand / Hustle
    /// ×3277/4096 physical). Folded into `turn.rs`'s to-hit roll. `None` otherwise.
    /// (Hustle carries BOTH `dmg_mod` [Atk ×1.5] AND `acc_mod` [acc ×0.8] — they ship
    /// together this phase.)
    pub acc_mod: Option<crate::dex::accmod::AccMod>,
    /// STATUS_IMMUNE class params (`gen3_status_immune_v1`) — Limber (par) / Insomnia +
    /// Vital Spirit (slp) / Immunity (psn,tox) / Water Veil (brn) / Magma Armor (frz).
    /// `None` for an ability that grants no major-status immunity. Read by
    /// `turn.rs::try_set_status`.
    pub status_immune: Option<StatusImmune>,
    /// CRIT_IMMUNE class (`gen3_ability_batch1_v1`) — Battle Armor / Shell Armor. `true`
    /// when the holder cannot be critically hit. The crit `randomChance` roll is DRAWN
    /// normally then OVERRIDDEN to false (draw-free — the resolved `onCriticalHit=false`
    /// fires AFTER the roll; `harness/probe_critimmune_rng.js`). Read in `turn.rs`'s crit path.
    pub crit_immune: bool,
    /// WEATHER_SPEED class (`gen3_ability_batch1_v1`) — Chlorophyll / Swift Swim. `Some(w)`
    /// ⇒ the holder's Speed is ×2 while `field.weather == w` (an `onModifySpe chainModify(2)`).
    /// A SPEED modifier feeding `getActionSpeed` (the tie-shuffles); draw-free itself. Read
    /// by `state.rs::effective_speed` / `update_speed`.
    pub weather_speed: Option<crate::state::Weather>,
    /// WEATHER_NEGATE class (`gen3_ability_batch1_v1`) — Cloud Nine / Air Lock. `true` ⇒
    /// while the holder is active, weather's damage/chip/speed effects are all suppressed
    /// (`field.effectiveWeather()` returns ''). Draw-free (changes deterministic values).
    /// Read by the weather-chip / weather-speed / weather-damage sites.
    pub weather_negate: bool,
    /// CONTACT_PROC class (`gen3_ability_batch2_v1`) — Static (par) / Poison Point (psn) /
    /// Flame Body (brn) / Effect Spore (slp/par/psn sample). `Some(..)` ⇒ a contact hit into
    /// the holder draws `randomChance(chance)` and (on a pass) statuses the ATTACKER. Read by
    /// `turn.rs::apply_contact_proc`. `None` for a non-CONTACT_PROC ability. (Cute Charm draws
    /// the same roll but adds the `attract` volatile — see `contact_attract`.)
    pub contact_proc: Option<ContactProc>,
    /// CONTACT_ATTRACT (`gen3_ability_batch4_v1`) — Cute Charm. `Some((num,den))` ⇒ when the
    /// holder is hit by a damaging CONTACT move, draw `randomChance(num,den)` (UNCONDITIONAL —
    /// the gender gate lives inside the attract volatile's onStart, which fails DRAW-FREE for
    /// same-gender / genderless pairs) and on a pass add the ATTRACT volatile to the ATTACKER.
    /// Same DamagingHit position as `contact_proc`. Read by `turn.rs::apply_contact_proc`.
    /// Probe `probe_cutecharm_attract_rng.js`.
    pub contact_attract: Option<(u32, u32)>,
    /// CONTACT recoil class (`gen3_ability_batch2_v1`) — Rough Skin. `true` ⇒ a contact hit
    /// into the holder deals `baseMaxhp/16` recoil to the ATTACKER, DRAW-FREE. Read by
    /// `turn.rs::apply_contact_proc`.
    pub contact_recoil: bool,
    /// BLOCK class — Soundproof (`gen3_ability_batch2_v1`). `true` ⇒ the holder is IMMUNE to a
    /// SOUND move (`move.flags.sound`), reporting `-immune` after the accuracy roll (the same
    /// draw model as a type-immune move). Read by `turn.rs`'s move-immunity path.
    pub blocks_sound: bool,
    /// BLOCK class — Damp (`gen3_ability_batch2_v1`). `true` ⇒ while the holder is on the
    /// field, Explosion / Self-Destruct (either side's) is CANCELLED at `runEvent('TryMove')`
    /// — BEFORE the self-KO faint and the accuracy roll (the user does NOT self-KO, the move
    /// draws nothing). Read by `turn.rs`'s explosion path.
    pub blocks_explosion: bool,
    /// BLOCK class — Suction Cups (`gen3_ability_batch2_v1`). `true` ⇒ a Roar/Whirlwind into
    /// the holder does NOT drag it (the `sample` is NOT drawn); the phaze's accuracy roll still
    /// draws, then `-activate Suction Cups`. Read by `turn.rs`'s phaze-drag path.
    pub blocks_phaze_drag: bool,
    /// SYNCHRONIZE (`gen3_ability_batch2_v1`) — `true` ⇒ when the holder is inflicted a MAJOR
    /// status by a FOE source, it REFLECTS the status back to the source (slp/frz exempt;
    /// tox→psn). Draw-free in gen3customgame (the e2e format). Read by `turn.rs::try_set_status`
    /// (the single status choke point).
    pub synchronize: bool,
    /// SHED SKIN (`gen3_berry_trace_shedskin_v1`) — `true` ⇒ a residual handler at order 10
    /// subOrder 3 (the Speed Boost / Rain Dish slot) that, while the holder has a MAJOR
    /// status, draws ONE `randomChance(33,100)` and on a pass CURES it (BEFORE the same-mon
    /// status DoT at subOrder 6 — a cure turn takes no chip). Confusion is NOT cured; an
    /// unstatused holder draws NOTHING (the handler is still GATHERED — it participates in
    /// the residual tie-shuffle regardless of status). Probe `probe_trace_shedskin_rng.js`.
    pub shed_skin: bool,
    /// TRACE (`gen3_berry_trace_shedskin_v1`) — `true` ⇒ the gen3-resolved onStart: at
    /// switch-in, ONE `sample` over the live foe actives (it DRAWS even for a single foe —
    /// the phaze-n=1 gotcha) and the foe's CURRENT ability is COPIED (no guard; the copied
    /// onStart does NOT fire in gen3 — `setAbility`'s `gen > 3` gate). The copy is LIVE for
    /// every ability read; switch-out reverts to Trace. Read by the switch-in ability-start
    /// dispatch. Probe `probe_trace_shedskin_rng.js`.
    pub trace: bool,
    /// WONDER GUARD (`gen3_wonder_guard_v1`) — Shedinja's SE-only damage gate. `true` ⇒ a
    /// DAMAGING move into the holder CONNECTS only if it is STRICTLY super-effective
    /// (`runEffectiveness > 0`, i.e. the type-chart effectiveness product `> 1.0`) AND not
    /// type-immune; every neutral / resisted / 0×-immune move is BLOCKED with
    /// `-immune|<t>|[from] ability: Wonder Guard`. The gen4-override `onTryHit` runs AFTER the
    /// accuracy roll and BEFORE crit/damage, so a blocked move draws ONLY its accuracy roll
    /// (the type-immune draw count). BYPASSED by Status moves, self-target hits, typeless
    /// (`???`/Struggle) moves, and ALL residual damage (a MOVE hook only). Read by
    /// `turn.rs::run_move`. `false` for a non-Wonder-Guard ability.
    pub wonder_guard: bool,
}

fn parse_ratio(v: &Json) -> Result<(u64, u64), String> {
    let arr = v.as_array().ok_or("dmgMod.mod: expected [num, den]")?;
    if arr.len() != 2 {
        return Err(format!("dmgMod.mod: expected 2 elements, got {}", arr.len()));
    }
    let n = arr[0].as_f64().ok_or("dmgMod.mod: non-numeric num")? as u64;
    let d = arr[1].as_f64().ok_or("dmgMod.mod: non-numeric den")? as u64;
    if n == 0 || d == 0 {
        return Err("dmgMod.mod: zero num/den".to_string());
    }
    Ok((n, d))
}

fn parse_dmg_mod(id: &str, dm: &Json) -> Result<DmgMod, String> {
    let (num, den) = parse_ratio(dm.get("mod").ok_or_else(|| format!("ability {id}: dmgMod.mod missing"))?)?;
    let fold = match dm.str_at("fold") {
        Some("basePower") => DmgFold::BasePower,
        Some("atk") => DmgFold::Atk,
        Some("def") => DmgFold::Def,
        Some("sourceBasePower") => DmgFold::SourceBasePower,
        other => return Err(format!("ability {id}: unknown dmgMod.fold {other:?}")),
    };
    // `type` (a single) or `types` (a list) — the gated move type(s). Absent ⇒ agnostic.
    let mut types = Vec::new();
    if let Some(t) = dm.str_at("type") {
        types.push(Type::from_name(t).ok_or_else(|| format!("ability {id}: unknown dmgMod.type {t:?}"))?);
    }
    if let Some(arr) = dm.get("types").and_then(|x| x.as_array()) {
        for t in arr {
            let s = t.as_str().ok_or_else(|| format!("ability {id}: dmgMod.types entry not a string"))?;
            types.push(Type::from_name(s).ok_or_else(|| format!("ability {id}: unknown dmgMod.types {s:?}"))?);
        }
    }
    Ok(DmgMod {
        num,
        den,
        fold,
        types,
        pinch: dm.bool_or("pinch", false),
        when_statused: dm.bool_or("whenStatused", false),
        direct: dm.bool_or("direct", false),
    })
}

fn parse_status_immune(id: &str, si: &Json) -> Result<StatusImmune, String> {
    let statuses = si
        .get("statuses")
        .and_then(|x| x.as_array())
        .ok_or_else(|| format!("ability {id}: statusImmune.statuses missing / not an array"))?
        .iter()
        .map(|s| s.as_str().map(|x| x.to_string()).ok_or_else(|| format!("ability {id}: statusImmune.statuses entry not a string")))
        .collect::<Result<Vec<_>, _>>()?;
    if statuses.is_empty() {
        return Err(format!("ability {id}: statusImmune.statuses is empty"));
    }
    let phase = match si.str_at("phase") {
        Some("setStatus") => StatusImmunePhase::SetStatus,
        Some("immunity") => StatusImmunePhase::Immunity,
        other => return Err(format!("ability {id}: unknown statusImmune.phase {other:?}")),
    };
    Ok(StatusImmune { statuses, phase })
}

fn parse_contact_proc(id: &str, cp: &Json) -> Result<ContactProc, String> {
    let statuses = cp
        .get("statuses")
        .and_then(|x| x.as_array())
        .ok_or_else(|| format!("ability {id}: contactProc.statuses missing / not an array"))?
        .iter()
        .map(|s| s.as_str().map(|x| x.to_string()).ok_or_else(|| format!("ability {id}: contactProc.statuses entry not a string")))
        .collect::<Result<Vec<_>, _>>()?;
    if statuses.is_empty() {
        return Err(format!("ability {id}: contactProc.statuses is empty"));
    }
    let chance_arr = cp
        .get("chance")
        .and_then(|x| x.as_array())
        .ok_or_else(|| format!("ability {id}: contactProc.chance missing / not [num,den]"))?;
    if chance_arr.len() != 2 {
        return Err(format!("ability {id}: contactProc.chance expected 2 elements"));
    }
    let num = chance_arr[0].as_f64().ok_or_else(|| format!("ability {id}: contactProc.chance num non-numeric"))? as u32;
    let den = chance_arr[1].as_f64().ok_or_else(|| format!("ability {id}: contactProc.chance den non-numeric"))? as u32;
    if num == 0 || den == 0 {
        return Err(format!("ability {id}: contactProc.chance zero num/den"));
    }
    let sample = cp.bool_or("sample", false);
    // GIGO guard: `sample` requires >1 status to choose from; a single-status proc must not
    // carry `sample` (the draw model differs — a `random(len)` vs a direct apply).
    if sample && statuses.len() < 2 {
        return Err(format!("ability {id}: contactProc.sample=true but only {} status(es)", statuses.len()));
    }
    if !sample && statuses.len() != 1 {
        return Err(format!("ability {id}: contactProc.sample=false requires exactly 1 status, got {}", statuses.len()));
    }
    Ok(ContactProc { statuses, chance: (num, den), sample })
}

pub(super) fn parse(root: &Json) -> Result<HashMap<String, AbilityData>, String> {
    let obj = root.as_object().ok_or("abilities: expected a JSON object")?;
    let mut out = HashMap::with_capacity(obj.len());
    for (id, v) in obj {
        let dmg_mod = match v.get("dmgMod") {
            Some(dm) => Some(parse_dmg_mod(id, dm)?),
            None => None,
        };
        let status_immune = match v.get("statusImmune") {
            Some(si) => Some(parse_status_immune(id, si)?),
            None => None,
        };
        // WEATHER_SPEED: `weatherSpeed: {weather: "<id>"}` → the Weather the holder's speed
        // doubles under (`gen3_ability_batch1_v1`).
        let weather_speed = match v.get("weatherSpeed") {
            Some(ws) => {
                let wid = ws.str_at("weather").ok_or_else(|| format!("ability {id}: weatherSpeed.weather missing"))?;
                Some(crate::state::Weather::from_id(wid).ok_or_else(|| format!("ability {id}: unknown weatherSpeed.weather {wid:?}"))?)
            }
            None => None,
        };
        let contact_proc = match v.get("contactProc") {
            Some(cp) => Some(parse_contact_proc(id, cp)?),
            None => None,
        };
        let contact_attract = match v.get("contactAttract") {
            Some(ca) => {
                let arr = ca
                    .get("chance")
                    .and_then(|x| x.as_array())
                    .ok_or_else(|| format!("ability {id}: contactAttract.chance missing / not [num,den]"))?;
                if arr.len() != 2 {
                    return Err(format!("ability {id}: contactAttract.chance expected 2 elements"));
                }
                let num = arr[0].as_f64().ok_or_else(|| format!("ability {id}: contactAttract.chance num non-numeric"))? as u32;
                let den = arr[1].as_f64().ok_or_else(|| format!("ability {id}: contactAttract.chance den non-numeric"))? as u32;
                if num == 0 || den == 0 {
                    return Err(format!("ability {id}: contactAttract.chance zero num/den"));
                }
                Some((num, den))
            }
            None => None,
        };
        out.insert(
            id.clone(),
            AbilityData {
                id: id.clone(),
                num: v.int_or("num", 0) as u16,
                name: v.str_at("name").unwrap_or(id).to_string(),
                dmg_mod,
                acc_mod: crate::dex::accmod::parse(id, v)?,
                status_immune,
                crit_immune: v.bool_or("critImmune", false),
                weather_speed,
                weather_negate: v.bool_or("weatherNegate", false),
                contact_proc,
                contact_attract,
                contact_recoil: v.bool_or("contactRecoil", false),
                blocks_sound: v.bool_or("blocksSound", false),
                blocks_explosion: v.bool_or("blocksExplosion", false),
                blocks_phaze_drag: v.bool_or("blocksPhazeDrag", false),
                synchronize: v.bool_or("synchronize", false),
                shed_skin: v.bool_or("shedSkin", false),
                trace: v.bool_or("trace", false),
                wonder_guard: v.bool_or("wonderGuard", false),
            },
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dex::{Dex, Type};

    /// `gen3_item_mechanics_v1` (ability side) dex-parse pins — the committed
    /// `gen3_abilities.json` `dmgMod` fields parse to the probe-settled values (the
    /// resolved-dist facts; see `harness/dump_gen3_mechanics.js --check`).
    #[test]
    fn ability_dmg_mod_fields_parse() {
        let d = Dex::for_gen(3);
        let dm = |id: &str| {
            d.ability(id)
                .unwrap_or_else(|| panic!("{id} missing"))
                .dmg_mod
                .clone()
                .unwrap_or_else(|| panic!("{id} has no dmgMod"))
        };

        // Pinch family: BasePower ×3/2, gated on the single element type + pinch.
        for (id, t) in [
            ("torrent", Type::Water),
            ("blaze", Type::Fire),
            ("overgrow", Type::Grass),
            ("swarm", Type::Bug),
        ] {
            let m = dm(id);
            assert_eq!((m.num, m.den, m.fold), (3, 2, DmgFold::BasePower), "{id}");
            assert_eq!(m.types, vec![t], "{id} type");
            assert!(m.pinch, "{id} pinch");
            assert!(!m.when_statused && !m.direct, "{id}");
        }

        // Huge / Pure Power: Atk ×2, unconditional (no type, no pinch, no status).
        for id in ["hugepower", "purepower"] {
            let m = dm(id);
            assert_eq!((m.num, m.den, m.fold), (2, 1, DmgFold::Atk), "{id}");
            assert!(m.types.is_empty() && !m.pinch && !m.when_statused && !m.direct, "{id}");
        }

        // Guts: Atk ×3/2 whenStatused (type-agnostic).
        let g = dm("guts");
        assert_eq!((g.num, g.den, g.fold), (3, 2, DmgFold::Atk));
        assert!(g.when_statused && g.types.is_empty() && !g.pinch && !g.direct);

        // Marvel Scale: Def ×3/2 whenStatused (defender-side).
        let ms = dm("marvelscale");
        assert_eq!((ms.num, ms.den, ms.fold), (3, 2, DmgFold::Def));
        assert!(ms.when_statused && ms.types.is_empty() && !ms.pinch && !ms.direct);

        // Hustle: Atk ×3/2 DIRECT (data-only — the acc side ships it, not this phase).
        let h = dm("hustle");
        assert_eq!((h.num, h.den, h.fold), (3, 2, DmgFold::Atk));
        assert!(h.direct && !h.when_statused && !h.pinch);

        // Thick Fat: sourceBasePower ×1/2 for Ice+Fire (modeled via defender_thick_fat).
        let tf = dm("thickfat");
        assert_eq!((tf.num, tf.den, tf.fold), (1, 2, DmgFold::SourceBasePower));
        assert_eq!(tf.types, vec![Type::Ice, Type::Fire]);
        assert!(!tf.pinch && !tf.when_statused && !tf.direct);

        // A plain ability carries no dmgMod.
        assert!(d.ability("intimidate").expect("intimidate").dmg_mod.is_none());
        assert!(d.ability("levitate").expect("levitate").dmg_mod.is_none());
        assert!(d.ability("intimidate").expect("intimidate").acc_mod.is_none());
    }

    /// `gen3_accuracy_pipeline_v1` (ability side) — the ACCURACY-class accMod fields parse
    /// to the RESOLVED gen3 handlers: Compound Eyes ×1.3 (attacker chain), Sand Veil ×0.8
    /// in sand (defender chain), Hustle ×3277/4096 for a physical-type move (attacker chain,
    /// shipping alongside its dmgMod Atk ×1.5).
    #[test]
    fn ability_acc_mod_fields_parse() {
        use crate::dex::accmod::{AccOp, AccSide};
        let d = Dex::for_gen(3);
        let ce = d.ability("compoundeyes").unwrap().acc_mod.expect("compoundeyes accMod");
        assert_eq!(ce.op, AccOp::Chain(13, 10));
        assert_eq!(ce.side, AccSide::Attacker);
        assert!(!ce.weather_sand && !ce.physical_types_only);
        let sv = d.ability("sandveil").unwrap().acc_mod.expect("sandveil accMod");
        assert_eq!(sv.op, AccOp::Chain(8, 10));
        assert_eq!((sv.side, sv.weather_sand), (AccSide::Defender, true));
        let hu = d.ability("hustle").unwrap();
        let ha = hu.acc_mod.expect("hustle accMod");
        assert_eq!(ha.op, AccOp::Chain(3277, 4096));
        assert_eq!((ha.side, ha.physical_types_only), (AccSide::Attacker, true));
        // Hustle carries BOTH the accMod AND the Atk ×1.5 dmgMod (they ship together).
        assert!(hu.dmg_mod.is_some(), "Hustle keeps its Atk ×1.5 dmgMod");
    }

    /// `gen3_status_immune_v1` dex-parse pins — the committed `gen3_abilities.json`
    /// `statusImmune` fields parse to the PROBE-settled statuses + phase (the resolved-dist
    /// facts; `harness/probe_statusimmune_*.js`, drift-gated by `dump_gen3_mechanics.js --check`).
    #[test]
    fn ability_status_immune_fields_parse() {
        let d = Dex::for_gen(3);
        let si = |id: &str| {
            d.ability(id)
                .unwrap_or_else(|| panic!("{id} missing"))
                .status_immune
                .clone()
                .unwrap_or_else(|| panic!("{id} has no statusImmune"))
        };
        // The 5 onSetStatus members (block INSIDE the SetStatus event, phase=SetStatus).
        for (id, statuses) in [
            ("limber", vec!["par"]),
            ("insomnia", vec!["slp"]),
            ("vitalspirit", vec!["slp"]),
            ("immunity", vec!["psn", "tox"]),
            ("waterveil", vec!["brn"]),
        ] {
            let s = si(id);
            assert_eq!(s.phase, StatusImmunePhase::SetStatus, "{id} phase");
            assert_eq!(s.statuses, statuses, "{id} statuses");
            for st in &statuses {
                assert!(s.blocks(st), "{id} should block {st}");
            }
        }
        // Magma Armor blocks freeze at runStatusImmunity (phase=Immunity, BEFORE the event).
        let ma = si("magmaarmor");
        assert_eq!(ma.phase, StatusImmunePhase::Immunity);
        assert_eq!(ma.statuses, vec!["frz"]);
        assert!(ma.blocks("frz") && !ma.blocks("brn"));
        // Own Tempo / Oblivious block a VOLATILE (confusion/attract), NOT a major status —
        // so they carry NO statusImmune (a different mechanism).
        assert!(d.ability("owntempo").expect("owntempo").status_immune.is_none());
        assert!(d.ability("oblivious").expect("oblivious").status_immune.is_none());
        // A plain ability carries no statusImmune.
        assert!(d.ability("pressure").expect("pressure").status_immune.is_none());
    }

    /// `gen3_ability_batch1_v1` dex-parse pins — the CRIT_IMMUNE / WEATHER_SPEED /
    /// WEATHER_NEGATE fields parse to the PROBE-settled values (drift-gated by
    /// `dump_gen3_mechanics.js --check`).
    #[test]
    fn ability_batch1_fields_parse() {
        use crate::state::Weather;
        let d = Dex::for_gen(3);
        // CRIT_IMMUNE: Battle Armor / Shell Armor.
        assert!(d.ability("battlearmor").unwrap().crit_immune);
        assert!(d.ability("shellarmor").unwrap().crit_immune);
        assert!(!d.ability("intimidate").unwrap().crit_immune);
        // WEATHER_SPEED: Chlorophyll (Sun) / Swift Swim (Rain).
        assert_eq!(d.ability("chlorophyll").unwrap().weather_speed, Some(Weather::Sun));
        assert_eq!(d.ability("swiftswim").unwrap().weather_speed, Some(Weather::Rain));
        assert_eq!(d.ability("intimidate").unwrap().weather_speed, None);
        // WEATHER_NEGATE: Cloud Nine / Air Lock.
        assert!(d.ability("cloudnine").unwrap().weather_negate);
        assert!(d.ability("airlock").unwrap().weather_negate);
        assert!(!d.ability("intimidate").unwrap().weather_negate);
    }

    /// `gen3_ability_batch2_v1` dex-parse pins — the CONTACT_PROC / recoil / BLOCK / SYNCHRONIZE
    /// fields parse to the PROBE-settled values (drift-gated by `dump_gen3_mechanics.js --check`).
    #[test]
    fn ability_batch2_fields_parse() {
        let d = Dex::for_gen(3);
        // CONTACT_PROC single-status members: randomChance(1,3) → one status on the attacker.
        for (id, st) in [("static", "par"), ("poisonpoint", "psn"), ("flamebody", "brn")] {
            let cp = d.ability(id).unwrap().contact_proc.clone().unwrap_or_else(|| panic!("{id} contactProc"));
            assert_eq!(cp.chance, (1, 3), "{id} chance");
            assert_eq!(cp.statuses, vec![st], "{id} statuses");
            assert!(!cp.sample, "{id} sample");
        }
        // Effect Spore: randomChance(1,10) → sample(["slp","par","psn"]).
        let es = d.ability("effectspore").unwrap().contact_proc.clone().expect("effectspore contactProc");
        assert_eq!(es.chance, (1, 10));
        assert_eq!(es.statuses, vec!["slp", "par", "psn"]);
        assert!(es.sample);
        // Rough Skin: contact recoil, no contactProc.
        assert!(d.ability("roughskin").unwrap().contact_recoil);
        assert!(d.ability("roughskin").unwrap().contact_proc.is_none());
        // BLOCK abilities.
        assert!(d.ability("soundproof").unwrap().blocks_sound);
        assert!(d.ability("damp").unwrap().blocks_explosion);
        assert!(d.ability("suctioncups").unwrap().blocks_phaze_drag);
        // SYNCHRONIZE.
        assert!(d.ability("synchronize").unwrap().synchronize);
        // Cute Charm carries contactAttract (batch 4), NOT a contactProc; Color Change is an
        // ENGINE flag (the types_override thread), no data fields.
        let cc = d.ability("cutecharm").unwrap();
        assert!(cc.contact_proc.is_none() && !cc.contact_recoil, "cutecharm is not a contactProc");
        let col = d.ability("colorchange").unwrap();
        assert!(col.contact_proc.is_none() && !col.contact_recoil, "colorchange has no batch-2 fields");
        // A plain ability carries none of the batch-2 fields.
        let intim = d.ability("intimidate").unwrap();
        assert!(intim.contact_proc.is_none() && !intim.contact_recoil && !intim.blocks_sound
            && !intim.blocks_explosion && !intim.blocks_phaze_drag && !intim.synchronize);
    }

    /// `gen3_ability_batch4_v1` dex-parse pins — Cute Charm's contactAttract parses to the
    /// probe-settled roll (randomChance(1,3), unconditional — the gender gate is inside the
    /// attract volatile's onStart). Drift-gated by `dump_gen3_mechanics.js --check`.
    #[test]
    fn ability_batch4_fields_parse() {
        let d = Dex::for_gen(3);
        assert_eq!(d.ability("cutecharm").unwrap().contact_attract, Some((1, 3)));
        // The contactProc family + plain abilities carry no contactAttract.
        assert!(d.ability("static").unwrap().contact_attract.is_none());
        assert!(d.ability("intimidate").unwrap().contact_attract.is_none());
    }

    /// `gen3_berry_trace_shedskin_v1` dex-parse pins — Shed Skin + Trace parse to the
    /// probe-settled flags (drift-gated by `dump_gen3_mechanics.js --check`; probes
    /// `probe_trace_shedskin_rng.js`).
    #[test]
    fn ability_batch3_fields_parse() {
        let d = Dex::for_gen(3);
        assert!(d.ability("shedskin").unwrap().shed_skin);
        assert!(d.ability("trace").unwrap().trace);
        // Frisk-style randomFoe reporters / plain abilities carry neither flag.
        let intim = d.ability("intimidate").unwrap();
        assert!(!intim.shed_skin && !intim.trace);
        assert!(!d.ability("naturalcure").unwrap().shed_skin);
        assert!(!d.ability("synchronize").unwrap().trace);
    }

    /// `gen3_wonder_guard_v1` dex-parse pin — Wonder Guard carries the `wonderGuard` flag; a
    /// plain ability does not (drift-gated by `dump_gen3_mechanics.js --check`).
    #[test]
    fn ability_wonder_guard_field_parses() {
        let d = Dex::for_gen(3);
        assert!(d.ability("wonderguard").unwrap().wonder_guard);
        // Non-members carry no wonderGuard flag.
        assert!(!d.ability("intimidate").unwrap().wonder_guard);
        assert!(!d.ability("levitate").unwrap().wonder_guard);
        assert!(!d.ability("liquidooze").unwrap().wonder_guard);
    }
}
