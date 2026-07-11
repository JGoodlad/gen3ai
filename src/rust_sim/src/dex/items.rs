//! Item data — `{name, num}` plus the DATA-DRIVEN MECHANICS FRAMEWORK's structured
//! per-item mechanics fields (`gen3_item_mechanics_v1`).
//!
//! The mechanics fields (`typeBoost` / `statMods` / `onlySpecies` / `untransformedOnly` /
//! `choice` / `isBerry`) are emitted by `tools/pokemon_data_extractor/sync.py` from a
//! curated table derived from the RESOLVED `Dex.mod('gen3')` (the mod-chain law: gen3
//! resolves through gen4, which REPLACES/DELETES handlers — e.g. gen3 Light Ball is
//! SpA-ONLY x2; never trust a single raw data file). Drift gates:
//! `node src/rust_sim/harness/dump_gen3_mechanics.js --check` (committed JSON vs the
//! resolved dist) + the class-sweep golden `tests/item_mods_test.rs` (real sim battles).
//!
//! The engine consumes these via `turn.rs::resolve_atk_stat_mods` /
//! `resolve_def_stat_mods` / the base-power fold — ONE generic path per mechanic class
//! instead of a hardcoded per-id match arm (the drift that let Pink Bow / Polkadot Bow /
//! the 4 incenses sit "modeled" in the e2e while the engine priced none of them).

use crate::dex::types::Type;
use crate::json::Json;
use std::collections::HashMap;

/// Where a [`TypeBoost`] folds into the damage formula (probe-settled per item —
/// three DIFFERENT resolved handler shapes, three DIFFERENT roundings).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeBoostFold {
    /// gen3-mod `onModifyAtk`/`onModifySpA` `chainModify` — folds into the OFFENSIVE
    /// STAT chain (the x1.1 family + Sea Incense x1.05).
    Stat,
    /// base-data `onBasePower` `chainModify` — folds into the BASE-POWER chain, one
    /// accumulated 4096 modifier rounded once (the 4 gen4-named incenses, x4915/4096).
    BasePower,
    /// base-data `onBasePower` `return basePower * x` (Pink Bow / Polkadot Bow) — a
    /// DIRECT float return that REPLACES the event's relayVar; the non-integer result
    /// SKIPS runEvent's final chain-modifier application and `clampIntRange` floors it.
    BasePowerDirect,
}

/// A type-gated offensive boost (`typeBoost` in `gen3_items.json`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeBoost {
    /// The move type the boost gates on.
    pub type_: Type,
    /// The exact multiplier as a rational (`trunc(num*4096/den)` == Showdown's
    /// `trunc(float*4096)` for every emitted entry — bit-identical fixed-point).
    pub num: u64,
    pub den: u64,
    pub fold: TypeBoostFold,
}

/// Species-gated (or unconditional, for Choice Band) stat-event multipliers
/// (`statMods` + `onlySpecies` + `untransformedOnly`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StatMods {
    pub atk: Option<(u64, u64)>,
    pub spa: Option<(u64, u64)>,
    pub def: Option<(u64, u64)>,
    pub spd: Option<(u64, u64)>,
    /// Holder-species gate (lowercase ids). Empty == unconditional (Choice Band).
    pub only_species: Vec<String>,
    /// Metal Powder's `!pokemon.transformed` guard. The port has no Transform, so a
    /// holder is always untransformed — recorded for fidelity, read by nothing yet.
    pub untransformed_only: bool,
}

/// The BERRY consumption effect (`gen3_berry_trace_shedskin_v1`, `berryEffect` in
/// `gen3_items.json`) — ONE consumption mechanism (eat → the item becomes NONE for the
/// battle) + these per-berry parameters. All probe-settled against the resolved gen3 dist
/// (`harness/probe_berry_rng.js` / `probe_berry_sub_tie_rng.js`):
///   - `Cure` fires at every `eachEvent('Update')` (draw-free); Lum ALSO fires IMMEDIATELY
///     inside `try_set_status` (`immediate`, after a Synchronize reflect).
///   - `Heal` / `Pinch` fire at the RESIDUAL (order 10 subOrder 4 — the Leftovers slot,
///     gathered in the item position) on `2*hp <= maxhp` / `4*hp <= maxhp` exactly.
///   - `Pp` (Leppa) fires at every `eachEvent('Update')` when a move slot hits 0 PP.
#[derive(Debug, Clone, PartialEq)]
pub enum BerryEffect {
    /// CURE_BERRY: `statuses` = the major-status ids cured (`par`/`slp`/`psn`/`tox`/`brn`/
    /// `frz`; empty for Persim), `cures_confusion` (Persim / Lum), `immediate` (Lum's
    /// in-setStatus eat).
    Cure { statuses: Vec<String>, cures_confusion: bool, immediate: bool },
    /// HEAL_BERRY at `hp <= maxhp/2`: a FIXED heal (Oran 10 / Sitrus 30).
    HealFixed { amount: u16 },
    /// HEAL_BERRY at `hp <= maxhp/2`: `floor(maxhp/frac)` (the Figy family, frac=8 — the
    /// RESOLVED gen3 amount, NOT the base .ts /3) + a confusion volatile (drawing its
    /// `random(2,6)`) if the holder's nature LOWERS `confuse_if_minus`.
    HealFrac { frac: u16, confuse_if_minus: String },
    /// PINCH_BERRY at `hp <= maxhp/4`: +1 stage on the stat (boosts index 0..4).
    PinchBoost { stat: usize },
    /// PINCH_BERRY: Starf — ONE `sample` over the non-capped [atk,def,spa,spd,spe] (in that
    /// order; the sample DRAWS even for a single candidate; no candidates ⇒ no draw, no
    /// boost — the berry is still eaten), then +2 on the drawn stat.
    StarfRandom2,
    /// PINCH_BERRY: Lansat — the `focusenergy` volatile (crit stage +2), draw-free.
    LansatFocusEnergy,
    /// PP_BERRY: Leppa — restore `min(pp+10, maxpp)` on the first 0-PP move slot.
    PpRestore { amount: u16 },
}

/// An item record: `{name, num}` + the structured mechanics fields.
#[derive(Debug, Clone)]
pub struct ItemData {
    pub id: String,
    pub num: u16,
    pub name: String,
    /// TYPE_BOOST class (Magnet / Charcoal / ... / the bows / the incenses).
    pub type_boost: Option<TypeBoost>,
    /// SPECIES_STAT class (Thick Club / Light Ball / DeepSea* / Metal Powder /
    /// Soul Dew) + Choice Band's unconditional Atk x1.5.
    pub stat_mods: Option<StatMods>,
    /// CHOICE class — Choice Band (the engine's move-lock keys on this).
    pub choice: bool,
    /// Berry classes (PINCH/HEAL/CURE/PP) — the declarative flag; the CONSUMPTION
    /// parameters live in `berry_effect`.
    pub is_berry: bool,
    /// The BERRY consumption effect (`gen3_berry_trace_shedskin_v1`) — `Some` for the 22
    /// gen-3 berries with an in-battle trigger (7 cure + 7 heal + 7 pinch + Leppa); `None`
    /// for a flavor berry / gen-2 twin / non-berry. Read by `turn.rs`'s berry sites
    /// (residual order-10-subOrder-4 handler, the Update sites, the setStatus tail).
    pub berry_effect: Option<BerryEffect>,
    /// ACCURACY_ITEM class (Bright Powder ×0.9 / Lax Incense ×0.95 — DEFENDER-side
    /// `onModifyAccuracy`). Folded into `turn.rs`'s to-hit roll. `None` for a plain item.
    pub acc_mod: Option<crate::dex::accmod::AccMod>,
    /// PROC_ITEM: King's Rock (`gen3_ability_batch4_v1`, `flinchSecondary`) — the holder's
    /// LISTED moves gain a trailing `{chance, flinch}` secondary (an ORDINARY secondary:
    /// rolled AFTER the move's own secondary, BEFORE the foe's contact proc; Serene Grace
    /// doubles the chance; Shield Dust filters the draw; behind a Substitute it draws but
    /// does not apply — probe-settled `probe_kingsrock_rng.js` / `probe_kingsrock_order_rng.js`).
    /// `moves` uses the SIM ids (the 17 Hidden Powers collapse to the one id `hiddenpower`
    /// — the engine canonicalizes its typed HP ids for this lookup).
    pub flinch_secondary: Option<FlinchSecondary>,
    /// PROC_ITEM: Focus Band (`gen3_ability_batch4_v1`, `surviveLethal`) — an `onDamage`
    /// `randomChance(num,den)` that DRAWS FIRST on EVERY Damage event into the holder
    /// (move hits, residual chips, Spikes, recoil, confusion self-hits — NOT sub-absorbed
    /// hits) and, when it passes AND the damage is lethal AND the effect is a MOVE, caps
    /// the damage at `hp - 1` (survive at 1 HP). Probe-settled `probe_focusband_rng.js` /
    /// `probe_focusband_confusion_rng.js`.
    pub survive_lethal: Option<(u32, u32)>,
}

/// King's Rock's appended flinch secondary (`flinchSecondary` in `gen3_items.json`).
#[derive(Debug, Clone, PartialEq)]
pub struct FlinchSecondary {
    /// The secondary chance in percent (10; Serene Grace doubles it at roll time).
    pub chance: u16,
    /// The affected move ids (sim ids — sorted, execution-derived from the resolved
    /// `onModifyMove` by `dump_gen3_mechanics.js`).
    pub moves: Vec<String>,
}

fn parse_ratio(v: &Json) -> Result<(u64, u64), String> {
    let arr = v.as_array().ok_or("item mod: expected [num, den]")?;
    if arr.len() != 2 {
        return Err(format!("item mod: expected 2 elements, got {}", arr.len()));
    }
    let n = arr[0].as_f64().ok_or("item mod: non-numeric num")? as u64;
    let d = arr[1].as_f64().ok_or("item mod: non-numeric den")? as u64;
    if n == 0 || d == 0 {
        return Err("item mod: zero num/den".to_string());
    }
    Ok((n, d))
}

/// The pinch-boost / confuse-if-minus stat name → the engine's boosts index
/// ([atk, def, spa, spd, spe] = 0..4 — `crate::state::BOOST_*` order).
fn stat_index(id: &str, name: &str) -> Result<usize, String> {
    match name {
        "atk" => Ok(0),
        "def" => Ok(1),
        "spa" => Ok(2),
        "spd" => Ok(3),
        "spe" => Ok(4),
        other => Err(format!("item {id}: unknown berryEffect stat {other:?}")),
    }
}

fn parse_berry_effect(id: &str, be: &Json) -> Result<BerryEffect, String> {
    match be.str_at("class") {
        Some("cure") => {
            let statuses = be
                .get("statuses")
                .and_then(|s| s.as_array())
                .ok_or_else(|| format!("item {id}: berryEffect.statuses missing"))?
                .iter()
                .map(|s| s.as_str().map(str::to_string).ok_or_else(|| format!("item {id}: non-string status")))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(BerryEffect::Cure {
                statuses,
                cures_confusion: be.bool_or("curesConfusion", false),
                immediate: be.bool_or("immediate", false),
            })
        }
        Some("heal") => {
            if let Some(h) = be.get("heal") {
                let amount = h.as_f64().ok_or_else(|| format!("item {id}: berryEffect.heal non-numeric"))? as u16;
                Ok(BerryEffect::HealFixed { amount })
            } else {
                let frac = be
                    .get("healFrac")
                    .and_then(|f| f.as_f64())
                    .ok_or_else(|| format!("item {id}: berryEffect.healFrac missing"))? as u16;
                let minus = be
                    .str_at("confuseIfMinus")
                    .ok_or_else(|| format!("item {id}: berryEffect.confuseIfMinus missing"))?;
                // Validate the stat name eagerly (fail-loud at load, not at eat time).
                stat_index(id, minus)?;
                Ok(BerryEffect::HealFrac { frac, confuse_if_minus: minus.to_string() })
            }
        }
        Some("pinch") => match be.str_at("boost") {
            Some("random2") => Ok(BerryEffect::StarfRandom2),
            Some("focusenergy") => Ok(BerryEffect::LansatFocusEnergy),
            Some(stat) => Ok(BerryEffect::PinchBoost { stat: stat_index(id, stat)? }),
            None => Err(format!("item {id}: berryEffect.boost missing")),
        },
        Some("pp") => {
            let amount = be
                .get("restore")
                .and_then(|r| r.as_f64())
                .ok_or_else(|| format!("item {id}: berryEffect.restore missing"))? as u16;
            Ok(BerryEffect::PpRestore { amount })
        }
        other => Err(format!("item {id}: unknown berryEffect.class {other:?}")),
    }
}

pub(super) fn parse(root: &Json) -> Result<HashMap<String, ItemData>, String> {
    let obj = root.as_object().ok_or("items: expected a JSON object")?;
    let mut out = HashMap::with_capacity(obj.len());
    for (id, v) in obj {
        let type_boost = match v.get("typeBoost") {
            Some(tb) => {
                let tname = tb.str_at("type").ok_or_else(|| format!("item {id}: typeBoost.type missing"))?;
                let type_ = Type::from_name(tname)
                    .ok_or_else(|| format!("item {id}: unknown typeBoost.type {tname:?}"))?;
                let (num, den) = parse_ratio(
                    tb.get("mod").ok_or_else(|| format!("item {id}: typeBoost.mod missing"))?,
                )?;
                let fold = match tb.str_at("fold") {
                    Some("stat") => TypeBoostFold::Stat,
                    Some("basePower") => TypeBoostFold::BasePower,
                    Some("basePowerDirect") => TypeBoostFold::BasePowerDirect,
                    other => return Err(format!("item {id}: unknown typeBoost.fold {other:?}")),
                };
                Some(TypeBoost { type_, num, den, fold })
            }
            None => None,
        };
        let stat_mods = match v.get("statMods") {
            Some(sm) => {
                let mut mods = StatMods::default();
                let smo = sm.as_object().ok_or_else(|| format!("item {id}: statMods not an object"))?;
                for (stat, ratio) in smo {
                    let r = Some(parse_ratio(ratio)?);
                    match stat.as_str() {
                        "atk" => mods.atk = r,
                        "spa" => mods.spa = r,
                        "def" => mods.def = r,
                        "spd" => mods.spd = r,
                        other => return Err(format!("item {id}: unknown statMods key {other:?}")),
                    }
                }
                if let Some(arr) = v.get("onlySpecies").and_then(|s| s.as_array()) {
                    mods.only_species =
                        arr.iter().filter_map(|s| s.as_str().map(str::to_string)).collect();
                }
                mods.untransformed_only = v.bool_or("untransformedOnly", false);
                Some(mods)
            }
            None => None,
        };
        let berry_effect = match v.get("berryEffect") {
            Some(be) => Some(parse_berry_effect(id, be)?),
            None => None,
        };
        let flinch_secondary = match v.get("flinchSecondary") {
            Some(fs) => {
                let chance = fs
                    .get("chance")
                    .and_then(|c| c.as_f64())
                    .ok_or_else(|| format!("item {id}: flinchSecondary.chance missing"))? as u16;
                let moves = fs
                    .get("moves")
                    .and_then(|m| m.as_array())
                    .ok_or_else(|| format!("item {id}: flinchSecondary.moves missing"))?
                    .iter()
                    .map(|m| m.as_str().map(str::to_string).ok_or_else(|| format!("item {id}: non-string move id")))
                    .collect::<Result<Vec<_>, _>>()?;
                if moves.is_empty() {
                    return Err(format!("item {id}: flinchSecondary.moves empty"));
                }
                Some(FlinchSecondary { chance, moves })
            }
            None => None,
        };
        let survive_lethal = match v.get("surviveLethal") {
            Some(sl) => {
                let (n, d) = parse_ratio(
                    sl.get("chance").ok_or_else(|| format!("item {id}: surviveLethal.chance missing"))?,
                )?;
                Some((n as u32, d as u32))
            }
            None => None,
        };
        out.insert(
            id.clone(),
            ItemData {
                id: id.clone(),
                num: v.int_or("num", 0) as u16,
                name: v.str_at("name").unwrap_or(id).to_string(),
                type_boost,
                stat_mods,
                choice: v.bool_or("choice", false),
                is_berry: v.bool_or("isBerry", false),
                berry_effect,
                acc_mod: crate::dex::accmod::parse(id, v)?,
                flinch_secondary,
                survive_lethal,
            },
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use crate::dex::{Dex, Type};

    /// `gen3_item_mechanics_v1` dex-parse pins — the committed `gen3_items.json`
    /// mechanics fields parse to the probe-settled values (the resolved-dist facts;
    /// see `harness/dump_gen3_mechanics.js --check` for the upstream drift gate).
    #[test]
    fn item_mechanics_fields_parse() {
        use super::TypeBoostFold::*;
        let d = Dex::for_gen(3);

        // TYPE_BOOST: 24 members — 18 stat-fold (17 x1.1 + Sea Incense x1.05),
        // 2 bows (basePowerDirect x1.1), 4 incenses (basePower x4915/4096 ~= 1.2).
        let tb = |id: &str| d.item(id).unwrap_or_else(|| panic!("{id} missing")).type_boost
            .unwrap_or_else(|| panic!("{id} has no typeBoost"));
        let magnet = tb("magnet");
        assert_eq!((magnet.type_, magnet.num, magnet.den, magnet.fold), (Type::Electric, 11, 10, Stat));
        let sea = tb("seaincense");
        assert_eq!((sea.type_, sea.num, sea.den, sea.fold), (Type::Water, 21, 20, Stat));
        // The bows: DIRECT float base-power x1.1 (NOT the stat fold, NOT chainModify).
        for id in ["pinkbow", "polkadotbow"] {
            let b = tb(id);
            assert_eq!((b.type_, b.num, b.den, b.fold), (Type::Normal, 11, 10, BasePowerDirect), "{id}");
        }
        // The gen4-named incenses: x4915/4096 (~1.2 — NOT x1.1) at the BP chain fold.
        for (id, t) in [
            ("oddincense", Type::Psychic),
            ("rockincense", Type::Rock),
            ("roseincense", Type::Grass),
            ("waveincense", Type::Water),
        ] {
            let b = tb(id);
            assert_eq!((b.type_, b.num, b.den, b.fold), (t, 4915, 4096, BasePower), "{id}");
        }
        let stat_fold_count = ["blackbelt", "blackglasses", "charcoal", "dragonfang", "hardstone",
            "magnet", "metalcoat", "miracleseed", "mysticwater", "nevermeltice", "poisonbarb",
            "seaincense", "sharpbeak", "silkscarf", "silverpowder", "softsand", "spelltag",
            "twistedspoon"]
        .iter()
        .filter(|id| tb(id).fold == Stat)
        .count();
        assert_eq!(stat_fold_count, 18, "all 18 stat-fold type boosters parse");

        // SPECIES_STAT: the gen3-RESOLVED semantics (the mod-chain law pins).
        let sm = |id: &str| d.item(id).unwrap_or_else(|| panic!("{id} missing")).stat_mods.clone()
            .unwrap_or_else(|| panic!("{id} has no statMods"));
        let lb = sm("lightball");
        assert_eq!(lb.spa, Some((2, 1)), "gen3 Light Ball is SpA x2");
        assert_eq!(lb.atk, None, "gen3 Light Ball is SpA-ONLY (the gen4 Atk half does NOT exist)");
        assert_eq!(lb.only_species, vec!["pikachu"]);
        let tc = sm("thickclub");
        assert_eq!(tc.atk, Some((2, 1)));
        assert_eq!(tc.only_species, vec!["cubone", "marowak"]);
        let mp = sm("metalpowder");
        assert_eq!(mp.def, Some((2, 1)));
        assert!(mp.untransformed_only, "Metal Powder is untransformed-Ditto-only");
        let dst = sm("deepseatooth");
        assert_eq!((dst.spa, dst.only_species.as_slice()), (Some((2, 1)), &["clamperl".to_string()][..]));
        let dss = sm("deepseascale");
        assert_eq!(dss.spd, Some((2, 1)), "DeepSeaScale is the DEFENSE-side (SpD) member");
        let sd = sm("souldew");
        assert_eq!((sd.spa, sd.spd), (Some((3, 2)), Some((3, 2))), "Soul Dew is SpA+SpD x1.5");

        // CHOICE: Choice Band = the lock flag + an unconditional Atk x1.5 statMod.
        let cb = d.item("choiceband").expect("choiceband");
        assert!(cb.choice);
        assert_eq!(cb.stat_mods.as_ref().unwrap().atk, Some((3, 2)));
        assert!(cb.stat_mods.as_ref().unwrap().only_species.is_empty());

        // isBerry rides along (DATA-ONLY this phase).
        assert!(d.item("lumberry").expect("lumberry").is_berry);
        assert!(d.item("salacberry").expect("salacberry").is_berry);
        assert!(!d.item("leftovers").expect("leftovers").is_berry);

        // Plain items carry NO mechanics fields.
        let lefties = d.item("leftovers").unwrap();
        assert!(lefties.type_boost.is_none() && lefties.stat_mods.is_none() && !lefties.choice);
        assert!(lefties.acc_mod.is_none());
    }

    /// `gen3_berry_trace_shedskin_v1` — the 22 gen-3 berry `berryEffect` rows parse to the
    /// probe-settled parameters (resolved-dist facts; drift gate `dump_gen3_mechanics.js
    /// --check`; probes `probe_berry_rng.js` / `probe_berry_sub_tie_rng.js`).
    #[test]
    fn item_berry_effect_fields_parse() {
        use super::BerryEffect::*;
        let d = Dex::for_gen(3);
        let be = |id: &str| d.item(id).unwrap_or_else(|| panic!("{id} missing")).berry_effect.clone()
            .unwrap_or_else(|| panic!("{id} has no berryEffect"));

        // CURE (7): the single-status six + Lum (all + confusion + IMMEDIATE).
        for (id, statuses) in [
            ("cheriberry", vec!["par"]), ("chestoberry", vec!["slp"]), ("pechaberry", vec!["psn", "tox"]),
            ("rawstberry", vec!["brn"]), ("aspearberry", vec!["frz"]),
        ] {
            assert_eq!(
                be(id),
                Cure { statuses: statuses.iter().map(|s| s.to_string()).collect(), cures_confusion: false, immediate: false },
                "{id}"
            );
        }
        assert_eq!(be("persimberry"), Cure { statuses: vec![], cures_confusion: true, immediate: false });
        assert_eq!(
            be("lumberry"),
            Cure {
                statuses: ["par", "slp", "psn", "tox", "brn", "frz"].iter().map(|s| s.to_string()).collect(),
                cures_confusion: true,
                immediate: true,
            }
        );

        // HEAL (7): Oran 10 / Sitrus 30 (the RESOLVED gen3 amount) / the Figy family
        // maxhp/8 (NOT the base .ts /3 — the mod-chain law) + nature-gated confusion.
        assert_eq!(be("oranberry"), HealFixed { amount: 10 });
        assert_eq!(be("sitrusberry"), HealFixed { amount: 30 });
        for (id, minus) in [("figyberry", "atk"), ("wikiberry", "spa"), ("magoberry", "spe"),
                            ("aguavberry", "spd"), ("iapapaberry", "def")] {
            assert_eq!(be(id), HealFrac { frac: 8, confuse_if_minus: minus.to_string() }, "{id}");
        }

        // PINCH (7): +1 boost members + Starf (sample +2) + Lansat (focus energy).
        for (id, stat) in [("liechiberry", 0usize), ("ganlonberry", 1), ("petayaberry", 2),
                            ("apicotberry", 3), ("salacberry", 4)] {
            assert_eq!(be(id), PinchBoost { stat }, "{id}");
        }
        assert_eq!(be("starfberry"), StarfRandom2);
        assert_eq!(be("lansatberry"), LansatFocusEnergy);

        // PP (1): Leppa.
        assert_eq!(be("leppaberry"), PpRestore { amount: 10 });

        // A flavor berry / a gen-2 twin / a non-berry carries NO berryEffect.
        assert!(d.item("razzberry").expect("razzberry").berry_effect.is_none());
        assert!(d.item("przcureberry").expect("przcureberry").berry_effect.is_none(), "gen-2 twin stays unmodeled");
        assert!(d.item("leftovers").expect("leftovers").berry_effect.is_none());
    }

    /// `gen3_ability_batch4_v1` (item side) — the PROC_ITEM fields parse to the
    /// probe-settled values (King's Rock's appended 10% flinch secondary over the
    /// execution-derived move list; Focus Band's onDamage randomChance(1,10)).
    #[test]
    fn item_proc_fields_parse() {
        let d = Dex::for_gen(3);
        let kr = d.item("kingsrock").expect("kingsrock").flinch_secondary.clone()
            .expect("kingsrock has flinchSecondary");
        assert_eq!(kr.chance, 10);
        assert_eq!(kr.moves.len(), 130, "the deduped execution-derived gen3 list");
        // Spot pins from the probes: listed (Slash/Tackle/Muddy Water/Seismic Toss/
        // Struggle/Earthquake + the bare hiddenpower), NOT listed (Body Slam, Crunch —
        // moves with their own non-flinch secondaries stay off unless listed).
        for id in ["slash", "tackle", "muddywater", "seismictoss", "struggle", "earthquake", "hiddenpower"] {
            assert!(kr.moves.iter().any(|m| m == id), "{id} is KR-listed");
        }
        for id in ["bodyslam", "crunch", "bite", "rockslide", "thunderbolt", "splash"] {
            assert!(!kr.moves.iter().any(|m| m == id), "{id} is NOT KR-listed");
        }
        assert!(kr.moves.windows(2).all(|w| w[0] < w[1]), "sorted + deduped");

        let fb = d.item("focusband").expect("focusband").survive_lethal
            .expect("focusband has surviveLethal");
        assert_eq!(fb, (1, 10));

        // Plain items carry neither.
        let lefties = d.item("leftovers").unwrap();
        assert!(lefties.flinch_secondary.is_none() && lefties.survive_lethal.is_none());
    }

    /// `gen3_accuracy_pipeline_v1` (item side) — the ACCURACY_ITEM accMod fields parse to
    /// the RESOLVED gen3 handlers (Bright Powder ×0.9 / Lax Incense ×0.95, DIRECT multiply,
    /// DEFENDER-side; NOT the base `.ts` chainModify — the mod-chain law).
    #[test]
    fn item_acc_mod_fields_parse() {
        use crate::dex::accmod::{AccOp, AccSide};
        let d = Dex::for_gen(3);
        let bp = d.item("brightpowder").unwrap().acc_mod.expect("brightpowder accMod");
        assert_eq!(bp.op, AccOp::Multiply(0.9));
        assert_eq!(bp.side, AccSide::Defender);
        assert!(!bp.weather_sand && !bp.physical_types_only);
        let lax = d.item("laxincense").unwrap().acc_mod.expect("laxincense accMod");
        assert_eq!(lax.op, AccOp::Multiply(0.95));
        assert_eq!(lax.side, AccSide::Defender);
    }
}
