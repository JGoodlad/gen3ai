//! Accuracy-modifier params — the ACCURACY pipeline's item/ability to-hit mods
//! (`gen3_accuracy_pipeline_v1`).
//!
//! Shared by items (Bright Powder / Lax Incense) and abilities (Compound Eyes /
//! Sand Veil / Hustle). Emitted by `tools/pokemon_data_extractor/sync.py` as an
//! `accMod` field (curated from the RESOLVED `Dex.mod('gen3')` via
//! `src/rust_sim/harness/dump_gen3_mechanics.js`; drift-gated by its `--check`) —
//! the mod-chain law: NEVER read a raw `.ts`. The resolved gen3 handlers differ
//! sharply from the base `.ts` (e.g. base Bright Powder is `chainModify([3686,
//! 4096])` but the gen3 mod REWRITES it to a DIRECT `accuracy * 0.9`).
//!
//! The engine consumes this in `turn.rs`'s accuracy roll: `effAcc = move.accuracy
//! × acc/eva stage table × the accMod handlers`, then `random(100) < effAcc` (the
//! ONE to-hit draw). See `turn.rs::effective_accuracy`.
//!
//! The RESOLVED gen3 accuracy modifiers (probe `harness/probe_accuracy_tohit.js` +
//! `probe_accuracy_intguard.js`), priority-DESC, all applied in ONE
//! `runEvent('ModifyAccuracy')`:
//!   - Compound Eyes (attacker, prio 9): `chainModify(1.3)`      → Chain [13,10]
//!   - Sand Veil     (defender, prio 8): `chainModify(0.8)` in sand → Chain [8,10] weather=sand
//!   - Hustle        (attacker, prio 7): `chainModify([3277,4096])` iff move.type ∈
//!                    the gen3 physical-type list                 → Chain [3277,4096] physTypes
//!   - Bright Powder (defender, prio 5): `accuracy * 0.9`  (DIRECT float) → Multiply 0.9
//!   - Lax Incense   (defender, prio 5): `accuracy * 0.95` (DIRECT float) → Multiply 0.95
//!
//! CHAIN vs DIRECT is the mod-chain subtlety: a chainModify accumulates into
//! `event.modifier` and is applied at the END of `runEvent` via `modify(relayVar,
//! modifier)` — BUT ONLY if `relayVar` is a NON-NEGATIVE INTEGER
//! (`relayVar === Math.abs(Math.floor(relayVar))`, battle.js). A DIRECT multiply
//! returns a new relayVar unconditionally. So a stage (float) or a direct member
//! having made the accuracy non-integer SKIPS every chain member — the exact
//! integer-guard the engine mirrors.

use crate::json::Json;

/// The operation shape of an accuracy modifier (the two resolved gen3 handler forms).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccOp {
    /// A DIRECT `accuracy * factor` (Bright Powder ×0.9 / Lax Incense ×0.95) — returns
    /// a new relayVar unconditionally (so it applies even when accuracy is a non-integer
    /// float). The factor is the EXACT f64 literal from the resolved handler; stored as a
    /// JSON float so Rust and JS parse identical bits (a rational `9/10` would differ in
    /// the last bit for many values — proven).
    Multiply(f64),
    /// A `chainModify([num, den])` accumulated into the runEvent modifier (Compound Eyes
    /// ×1.3, Sand Veil ×0.8, Hustle ×3277/4096) — applied at the END of `runEvent` via
    /// `modify(relayVar, modifier)`, and ONLY when relayVar is a non-negative integer.
    Chain(u64, u64),
}

/// Which side's item/ability owns the modifier: the ATTACKER (`onSourceModifyAccuracy`
/// — Compound Eyes / Hustle) or the DEFENDER (`onModifyAccuracy` — Bright Powder / Lax
/// Incense / Sand Veil).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccSide {
    Attacker,
    Defender,
}

/// An accuracy modifier: the operation + which side owns it + its gates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AccMod {
    pub op: AccOp,
    pub side: AccSide,
    /// Sand Veil: fires only while the field weather is Sandstorm.
    pub weather_sand: bool,
    /// Hustle: fires only when the MOVE's type is in the gen3 physical-type list
    /// (Normal/Fighting/Flying/Poison/Ground/Rock/Bug/Ghost/Steel — the resolved
    /// gen3-mod handler checks `physicalTypes.includes(move.type)`, NOT move.category).
    pub physical_types_only: bool,
}

/// Parse the `accMod` field of an item/ability entry. `None` when absent.
pub fn parse(id: &str, v: &Json) -> Result<Option<AccMod>, String> {
    let am = match v.get("accMod") {
        Some(am) => am,
        None => return Ok(None),
    };
    let op = match am.str_at("op") {
        Some("multiply") => {
            let f = am
                .get("mod")
                .and_then(Json::as_f64)
                .ok_or_else(|| format!("{id}: accMod.mod (multiply) missing/non-numeric"))?;
            if !(f > 0.0) {
                return Err(format!("{id}: accMod.mod (multiply) must be > 0"));
            }
            AccOp::Multiply(f)
        }
        Some("chain") => {
            let arr = am
                .get("mod")
                .and_then(Json::as_array)
                .ok_or_else(|| format!("{id}: accMod.mod (chain) must be [num, den]"))?;
            if arr.len() != 2 {
                return Err(format!("{id}: accMod.mod (chain) expected 2 elements"));
            }
            let n = arr[0].as_f64().ok_or_else(|| format!("{id}: accMod.mod num non-numeric"))? as u64;
            let d = arr[1].as_f64().ok_or_else(|| format!("{id}: accMod.mod den non-numeric"))? as u64;
            if n == 0 || d == 0 {
                return Err(format!("{id}: accMod.mod (chain) zero num/den"));
            }
            AccOp::Chain(n, d)
        }
        other => return Err(format!("{id}: unknown accMod.op {other:?}")),
    };
    let side = match am.str_at("side") {
        Some("attacker") => AccSide::Attacker,
        Some("defender") => AccSide::Defender,
        other => return Err(format!("{id}: unknown accMod.side {other:?}")),
    };
    Ok(Some(AccMod {
        op,
        side,
        weather_sand: am.str_at("weather") == Some("sandstorm"),
        physical_types_only: am.bool_or("physicalTypesOnly", false),
    }))
}
