//! Move reference data, parsed from `data/pokemon/gen3_moves.json`.
//! Mirrors `agents.gen3_data.moves.MoveData`, including the **derived** category.

use super::types::{MoveCategory, Type};
use crate::json::Json;
use std::collections::HashMap;

/// Static, gen3-only facts about a move. The `category` is derived (see
/// [`derive_category`]); everything else is read straight from the data file.
#[derive(Debug, Clone)]
pub struct MoveData {
    pub id: String,
    pub num: u16,
    pub name: String,
    /// `None` == typeless (`"???"`, e.g. Curse).
    pub move_type: Option<Type>,
    pub base_power: u16,
    pub category: MoveCategory,
    pub accuracy: u16,
    pub never_miss: bool,
    pub priority: i8,
    /// Gen-3 crit stage: 1 (normal → 1/16) or 2 (high-crit set → 1/8). Data-driven
    /// from `critRatio` in `gen3_moves.json` (absent ⇒ 1).
    pub crit_ratio: u8,
    pub target: String,
    /// The major status this move's PURPOSE is to inflict (else `None`).
    pub status_inflicted: Option<String>,
    pub has_secondary: bool,
    pub has_recoil: bool,
    pub is_boost: bool,
    pub is_heal: bool,
    pub is_protect: bool,
    pub is_phaze: bool,
    pub is_hazard: bool,
    pub cures_self_status: bool,
    pub cures_team_status: bool,
    pub drain_fraction: f64,
    pub recoil_fraction: f64,
    /// The move's BASE PP (before PP-ups), from `pp` in `gen3_moves.json`
    /// (`gen3_pp_tracking_v1`). Showdown computes a moveslot's in-battle PP as
    /// `calculatePP(move, ppUps) = pp * (5 + ppUps) / 5`; the `Pokemon` constructor
    /// defaults `ppUps` to 3 (max) for every non-`no_pp_boosts` move — so a mon's
    /// in-battle PP is `pp * 8 / 5` (gen3, integer). See [`MoveData::max_pp`].
    pub pp: u16,
    /// Whether the move gets NO PP-ups (`noPPBoosts` in the data — Struggle etc.).
    /// A `no_pp_boosts` move keeps its raw `pp` (0 PP-ups); every other move gets 3.
    pub no_pp_boosts: bool,
    /// Whether the move makes physical CONTACT (`flags.contact`, `gen3_ability_batch2_v1`).
    /// The CONTACT_PROC abilities (Static/Poison Point/Flame Body/Effect Spore/Rough Skin) react
    /// ONLY to a contact move. Body Slam/Tackle/Crunch = true; Earthquake/Surf/Flamethrower = false.
    pub contact: bool,
    /// Whether the move is SOUND-based (`flags.sound`, `gen3_ability_batch2_v1`). Soundproof is
    /// IMMUNE to it. Of the modeled moves: Sing / Grass Whistle (sleep) + Roar (phaze) are sound;
    /// Whirlwind is NOT.
    pub is_sound: bool,
    /// Whether the move CANNOT be called by Sleep Talk (`flags.nosleeptalk` →
    /// `noSleepTalk` in the data, `gen3_move_coverage_batch5_v1`). Sleep Talk's onHit
    /// builds its pool from the user's moveSlots keeping only
    /// `!nosleeptalk && !charge` — data-enumerated, never hand-listed (the gen3
    /// carriers: sleeptalk itself, focuspunch, uproar, bide, metronome, mirrormove,
    /// assist + the charge family).
    pub no_sleep_talk: bool,
    /// Whether the move is a TWO-TURN CHARGE move (`flags.charge` → `isCharge` in the
    /// data, `gen3_move_coverage_batch5_v1`): solarbeam / fly / dig / dive / bounce /
    /// razorwind / skullbash / skyattack. Excluded from the Sleep Talk pool (with
    /// `no_sleep_talk`); the engine models only Solar Beam (the rest stay fail-loud).
    pub is_charge: bool,
    /// Whether ENCORE FAILS when this move is the target's lastMove (`flags.failencore`
    /// → `failEncore` in the data, `gen3_move_coverage_batch6_v1`; the gen3 carriers:
    /// encore / mimic / mirrormove / sketch / struggle / transform). Data-enumerated,
    /// never hand-listed. Read by `run_status_move`'s encore arm (the onStart reject —
    /// AFTER the accuracy + durationCallback draws, probe-settled).
    pub fail_encore: bool,
    /// Whether MIMIC FAILS when this move is the target's lastMove (`flags.failmimic`
    /// → `failMimic` in the data, `gen3_move_coverage_batch6_v1`; the gen3 carriers:
    /// metronome / mimic / sketch / struggle). Draw-free fail (`[still]` + `-fail`).
    pub fail_mimic: bool,
    /// Whether this move CAN BE SNATCHED (`flags.snatch` → `isSnatchable` in the data,
    /// `gen3_snatch_v1`). A `target:self` status move (self-boost / recover / Rest /
    /// status-cures / Substitute / Belly Drum / …) that a foe's Snatch STEALS. The 44
    /// gen3-legal carriers are data-enumerated, never hand-listed (Wish / Spikes /
    /// Thunder Wave / Snatch itself are NOT snatchable — the sim overturned the
    /// hypotheses). Read by `run_status_move`'s snatch interception.
    pub is_snatchable: bool,
    /// Sorted `(effect, percent)` pairs from `secondaryEffects`.
    pub secondary_effects: Vec<(String, u16)>,
    /// The STRUCTURED secondary stat-boost spec (`secondaryBoosts` in the data) the
    /// flat `secondary_effects` `{col:percent}` loses. One entry per `secondary`/
    /// `secondaries[i]` block that drops a foe stat or raises a self stat. Empty for
    /// the vast majority of moves (only the ~24 gen-3 boost-secondary moves). The
    /// engine applies these draw-free after the landed secondary `random(100)`.
    pub secondary_boosts: Vec<SecondaryBoost>,
    /// The PRIMARY self-boost spec (`selfBoosts` in the data) for a PURE setup move —
    /// a `target: self` Status move whose ENTIRE effect is raising the USER'S stat
    /// stages (Swords Dance `[(0, 2)]` = +2 Atk, Dragon Dance `[(0,1),(4,1)]` = +1
    /// Atk/+1 Spe, Calm Mind `[(2,1),(3,1)]` = +1 SpA/+1 SpD, …). Each `(stat-index,
    /// stages)` is a [`crate::state::MonState::boosts`] index (`[atk, def, spa, spd,
    /// spe, accuracy, evasion]`); all stages are POSITIVE. Empty for every non-setup
    /// move (only the ~17 pure gen-3 setup moves carry it — moves with an extra effect,
    /// a `volatileStatus` [Defense Curl/Minimize], an evasion boost [Double Team], or an
    /// HP cost [Belly Drum], are EXCLUDED, so the engine fail-loud guards them). The
    /// engine applies these DRAW-FREE on the user (the boost itself consumes no PRNG),
    /// clamped to `[-6, 6]`.
    pub self_boosts: Vec<(usize, i8)>,
    /// The top-level `move.self.boosts` SELF STAT-DROP spec on a DAMAGING move
    /// (`gen3_move_coverage_batch1_v1`, `selfDrops` in the data): Overheat `[(2, -2)]`
    /// = −2 SpA, Superpower `[(0,-1),(1,-1)]` = −1 Atk/−1 Def. Each `(stat-index,
    /// stages)` is a [`crate::state::MonState::boosts`] index; all stages are NEGATIVE.
    /// Empty for every non-self-drop move. gen3 `selfDrops` (battle-actions.ts:1338) DRAWS
    /// ONE `random(100)` (the `secondaryRoll`) then applies the drop UNCONDITIONALLY (Overheat/
    /// Superpower have `self.chance === undefined`) — so it is **NOT draw-free** (probe-verified
    /// via a per-call-site PRNG trace). The engine draws-then-discards the `random(100)` then
    /// applies the boosts on the USER after the hit, clamped to `[-6, 6]` — see
    /// `turn.rs::apply_self_drops`.
    pub self_drops: Vec<(usize, i8)>,
    /// The declarative FOE STAT-DROP spec for a standalone STAT-DROP STATUS move
    /// (`gen3_move_coverage_batch2_v1`, `statDropBoosts` in the data): Screech `[(1,-2)]`
    /// = −2 Def, Charm `[(0,-2)]` = −2 Atk, Metal Sound `[(3,-2)]` = −2 SpD, Feather Dance
    /// `[(0,-2)]`, Tickle `[(0,-1),(1,-1)]`, Fake Tears `[(3,-2)]`, Cotton Spore / Scary
    /// Face `[(4,-2)]`. Each `(stat-index, stages)` is a [`crate::state::MonState::boosts`]
    /// index; all stages NEGATIVE. Empty for every non-stat-drop move. The MOVE draws its
    /// accuracy roll (Screech/Metal Sound acc 85 CAN miss; Charm/Feather Dance/Tickle/Fake
    /// Tears acc 100 always pass) then applies the drop DRAW-FREE on the FOE via `boost()`
    /// (±6 clamp, Clear Body / White Smoke / Hyper Cutter / Keen Eye `onTryBoost` gated) —
    /// see `turn.rs`'s stat-drop-move arm in `run_status_move`.
    pub stat_drop_boosts: Vec<(usize, i8)>,
}

/// A foe stat-drop / self stat-raise SECONDARY (`secondaryBoosts[i]` in the data).
/// The `chance` mirrors the secondary block's chance (the SAME one `random(100)` the
/// secondary already draws — this carries only the apply spec, DRAW-FREE itself); the
/// boost is applied to the `target` (foe / self), each `(stat_index, stages)` clamped
/// to `[-6, 6]`. `stat_index` is the [`crate::state::MonState::boosts`] index
/// (`[atk, def, spa, spd, spe, accuracy, evasion]`).
#[derive(Debug, Clone)]
pub struct SecondaryBoost {
    pub chance: u16,
    /// `true` = the boost applies to the USER (self stat-raise, e.g. Meteor Mash
    /// +1 Atk); `false` = the FOE (stat-drop, e.g. Crunch −1 SpD).
    pub target_self: bool,
    /// `(boost-array index, stages)` pairs, e.g. `[(3, -1)]` for −1 SpD, or all five
    /// for Ancient Power. Stages are the signed stage delta.
    pub boosts: Vec<(usize, i8)>,
}

/// Map a Showdown boost stat id to the [`crate::state::MonState::boosts`] array index
/// (`[atk, def, spa, spd, spe, accuracy, evasion]`). Returns `None` for an unknown id
/// (a GIGO guard — the caller should reject rather than silently mis-apply).
pub fn boost_stat_index(stat: &str) -> Option<usize> {
    match stat {
        "atk" => Some(0),
        "def" => Some(1),
        "spa" => Some(2),
        "spd" => Some(3),
        "spe" => Some(4),
        "accuracy" => Some(5),
        "evasion" => Some(6),
        _ => None,
    }
}

impl MoveData {
    pub fn is_damaging(&self) -> bool {
        self.base_power > 0
    }

    /// The move's in-battle MAX PP, mirroring Showdown's `Pokemon` constructor:
    /// `calculatePP(move, ppUps)` with the constructor's default `ppUps == 3` (max)
    /// for a normal move, `0` for a `no_pp_boosts` move. gen3 (`gen <= 4`):
    /// `noPPBoosts ? pp : pp * (5 + 3) / 5 == pp * 8 / 5` (integer). Every move in a
    /// gen-3 set gets 3 PP-ups by default (the constructor hardcodes it, NOT read from
    /// the set) — VERIFIED vs the sim's `side.active[0].moveSlots[k].pp/.maxpp`.
    pub fn max_pp(&self) -> u16 {
        if self.no_pp_boosts {
            self.pp
        } else {
            // pp * (5 + 3) / 5, integer (gen<=2's `pp==40` special-case is gen1/2 only).
            self.pp * 8 / 5
        }
    }

    /// Whether TAUNT blocks this move (`gen3_taunt_disable_v1`). Showdown's taunt gates on
    /// `move.category === 'Status'` using the move's TRUE per-move category. Our `category` is
    /// DERIVED from base power (bp 0 → Status), which MIS-classifies the gen-3 FIXED-DAMAGE
    /// moves — Seismic Toss / Night Shade / Sonic Boom / Super Fang (Physical) and Dragon Rage
    /// / Mirror Coat (Special) all have bp 0 but a NON-Status Showdown category, so Taunt does
    /// NOT block them (VERIFIED vs the sim: under Taunt, Seismic Toss stays `disabled:false`).
    /// So a move is taunt-blocked iff it is derived-Status AND not one of those fixed-damage /
    /// counter-family moves. (Every REAL gen-3 Status move — boosts/heal/hazard/phaze/status-
    /// inflict/protect/etc. — has bp 0 and a genuine Status category, so it is still blocked.)
    ///
    /// The BARE `hiddenpower` (num 237) is the SAME mis-classification class
    /// (`gen3_iv_derived_hidden_power_bp_v1`, the round-12 pool-crash fix's LEGALITY sibling): the
    /// data ships it BP 0 → derived Status, but a packed gen3ou team stores a real damaging Hidden
    /// Power there (the type/BP come from the attacker's IVs, resolved at `run_move`). Showdown
    /// resolves the slot to a TYPED variant whose category is the type-split (Special) → NOT
    /// taunt-blocked, so a TAUNTED mon's Hidden Power stays selectable (VERIFIED vs the sim). The
    /// port must NOT reject it (else `choice_is_legal` rejects the legal HP → the choice stream
    /// misaligns and the wrong move runs — the ab_233_8 over-KO). The TYPED ids
    /// (`hiddenpower<type>`, nums 355-370) already carry BP 70 → non-Status → not blocked, so
    /// gate on the bare id — every `hiddenpower*` id must stay selectable under Taunt.
    pub fn blocked_by_taunt(&self) -> bool {
        self.category == MoveCategory::Status
            && !self.is_fixed_damage()
            && !self.is_variable_bp()
            && !self.id.starts_with("hiddenpower")
    }

    /// Whether this is a gen-3 FIXED-DAMAGE / `damageCallback` / Counter-family move (bp 0 but
    /// a non-Status Showdown category, so it deals damage rather than acting as a status move).
    /// Kept in lockstep with `turn.rs::is_fixed_damage_move`.
    pub fn is_fixed_damage(&self) -> bool {
        matches!(
            self.id.as_str(),
            "seismictoss" | "nightshade" | "sonicboom" | "dragonrage" | "superfang"
                | "psywave" | "fissure" | "horndrill" | "guillotine"
                | "counter" | "mirrorcoat" | "bide" | "endeavor"
        )
    }

    /// Whether this is a gen-3 VARIABLE-BP `basePowerCallback` move with a bp-0 data row
    /// (`gen3_move_coverage_batch5_v1`): Return / Frustration (happiness), Flail /
    /// Reversal (the HP-ratio ladder), Low Kick (the target-weight ladder). These carry
    /// `basePower: 0` in the data (the BP is engine-computed, the Water Spout precedent
    /// — Water Spout itself carries its data bp 150 so it is NOT in this set), which
    /// mis-derives their category as Status — the SAME mis-classification the
    /// fixed-damage family gets. Their TRUE Showdown category is type-derived Physical
    /// (Normal / Fighting), so Taunt does NOT block them. Kept in lockstep with
    /// `turn.rs::variable_bp`.
    pub fn is_variable_bp(&self) -> bool {
        matches!(
            self.id.as_str(),
            "return" | "frustration" | "flail" | "reversal" | "lowkick"
        )
    }
}

/// Gen ≤ 3 category: a 0-power move is STATUS; a damaging move is SPECIAL or
/// PHYSICAL by its type (the gen 1-3 type-based split). Behind the `gen`
/// parameter so a future Gen 4+ (per-move categories) slots in here, not in the
/// engine.
pub fn derive_category(gen: u8, base_power: u16, move_type: Option<Type>) -> MoveCategory {
    if base_power == 0 {
        return MoveCategory::Status;
    }
    if gen <= 3 {
        match move_type {
            Some(t) if t.is_special_gen3() => MoveCategory::Special,
            _ => MoveCategory::Physical,
        }
    } else {
        // Gen 4+ stores a per-move category; not needed for the Gen 3 target yet.
        MoveCategory::Physical
    }
}

pub(super) fn parse(root: &Json, gen: u8) -> Result<HashMap<String, MoveData>, String> {
    let obj = root.as_object().ok_or("moves: expected a JSON object")?;
    let mut out = HashMap::with_capacity(obj.len());
    for (id, v) in obj {
        let move_type = v.str_at("type").and_then(Type::from_name); // "???" -> None
        let base_power = v.int_or("basePower", 0) as u16;
        let mut secondary_effects: Vec<(String, u16)> = v
            .get("secondaryEffects")
            .and_then(Json::as_object)
            .map(|m| {
                m.iter()
                    .map(|(k, c)| (k.clone(), c.as_f64().map_or(0, |n| n as u16)))
                    .collect()
            })
            .unwrap_or_default();
        secondary_effects.sort();

        // The structured `secondaryBoosts` spec (only-when-present): the (stat, stages,
        // target) the flat `secondaryEffects` loses, so the engine can apply the real
        // foe stat-drop / self stat-raise. Throw on an unknown stat id (GIGO — never a
        // silent mis-apply).
        let mut secondary_boosts: Vec<SecondaryBoost> = Vec::new();
        if let Some(arr) = v.get("secondaryBoosts").and_then(Json::as_array) {
            for block in arr {
                let chance = block.int_or("chance", 0) as u16;
                let target = block.str_at("target").unwrap_or("");
                let target_self = match target {
                    "self" => true,
                    "foe" => false,
                    other => {
                        return Err(format!(
                            "move {id}: secondaryBoosts target must be foe|self, got {other:?}"
                        ))
                    }
                };
                let boosts_obj = block
                    .get("boosts")
                    .and_then(Json::as_object)
                    .ok_or_else(|| format!("move {id}: secondaryBoosts entry missing boosts object"))?;
                let mut boosts: Vec<(usize, i8)> = Vec::with_capacity(boosts_obj.len());
                for (stat, st) in boosts_obj {
                    let idx = boost_stat_index(stat)
                        .ok_or_else(|| format!("move {id}: unknown boost stat {stat:?}"))?;
                    let stages = st.as_f64().map_or(0, |n| n as i64);
                    boosts.push((idx, stages as i8));
                }
                // Deterministic order (HashMap iteration is unordered) so the apply is
                // reproducible; the apply itself is order-independent (additive clamp).
                boosts.sort_by_key(|&(idx, _)| idx);
                secondary_boosts.push(SecondaryBoost { chance, target_self, boosts });
            }
        }

        // The PRIMARY self-boost spec (`selfBoosts`, only-when-present) — the `{stat:
        // stages}` map for a pure setup move (Swords Dance / Dragon Dance / …). Same
        // GIGO discipline as `secondaryBoosts`: throw on an unknown stat id. Each entry
        // is a (boost-array index, signed stages) pair; sorted by index for a
        // deterministic, order-independent apply.
        let mut self_boosts: Vec<(usize, i8)> = Vec::new();
        if let Some(obj) = v.get("selfBoosts").and_then(Json::as_object) {
            for (stat, st) in obj {
                let idx = boost_stat_index(stat)
                    .ok_or_else(|| format!("move {id}: unknown selfBoosts stat {stat:?}"))?;
                let stages = st.as_f64().map_or(0, |n| n as i64);
                self_boosts.push((idx, stages as i8));
            }
            self_boosts.sort_by_key(|&(idx, _)| idx);
        }

        // The top-level SELF STAT-DROP spec (`selfDrops`, only-when-present) — the
        // `{stat: stages}` (negative) map for a damaging self-drop move (Overheat /
        // Superpower). Same GIGO discipline as `selfBoosts`: throw on an unknown stat.
        let mut self_drops: Vec<(usize, i8)> = Vec::new();
        if let Some(obj) = v.get("selfDrops").and_then(Json::as_object) {
            for (stat, st) in obj {
                let idx = boost_stat_index(stat)
                    .ok_or_else(|| format!("move {id}: unknown selfDrops stat {stat:?}"))?;
                let stages = st.as_f64().map_or(0, |n| n as i64);
                self_drops.push((idx, stages as i8));
            }
            self_drops.sort_by_key(|&(idx, _)| idx);
        }

        // The declarative FOE STAT-DROP spec (`statDropBoosts`, only-when-present) — the
        // `{stat: stages}` (negative) map for a standalone stat-drop STATUS move (Screech /
        // Charm / Metal Sound / …). Same GIGO discipline as `selfDrops`: throw on an
        // unknown stat id. `gen3_move_coverage_batch2_v1`.
        let mut stat_drop_boosts: Vec<(usize, i8)> = Vec::new();
        if let Some(obj) = v.get("statDropBoosts").and_then(Json::as_object) {
            for (stat, st) in obj {
                let idx = boost_stat_index(stat)
                    .ok_or_else(|| format!("move {id}: unknown statDropBoosts stat {stat:?}"))?;
                let stages = st.as_f64().map_or(0, |n| n as i64);
                stat_drop_boosts.push((idx, stages as i8));
            }
            stat_drop_boosts.sort_by_key(|&(idx, _)| idx);
        }

        out.insert(
            id.clone(),
            MoveData {
                id: id.clone(),
                num: v.int_or("num", 0) as u16,
                name: v.str_at("name").unwrap_or(id).to_string(),
                move_type,
                base_power,
                category: derive_category(gen, base_power, move_type),
                accuracy: v.int_or("accuracy", 0) as u16,
                never_miss: v.bool_or("never_miss", false),
                priority: v.int_or("priority", 0) as i8,
                crit_ratio: v.int_or("critRatio", 1).clamp(1, 4) as u8,
                target: v.str_at("target").unwrap_or("").to_string(),
                status_inflicted: v.str_at("status").map(str::to_string),
                has_secondary: v.bool_or("hasSecondary", false),
                has_recoil: v.bool_or("hasRecoil", false),
                is_boost: v.bool_or("isBoost", false),
                is_heal: v.bool_or("isHeal", false),
                is_protect: v.bool_or("isProtect", false),
                is_phaze: v.bool_or("isPhaze", false),
                is_hazard: v.bool_or("isHazard", false),
                cures_self_status: v.bool_or("curesSelfStatus", false),
                cures_team_status: v.bool_or("curesTeamStatus", false),
                drain_fraction: v.f64_or("drainFraction", 0.0),
                recoil_fraction: v.f64_or("recoilFraction", 0.0),
                pp: v.int_or("pp", 0) as u16,
                no_pp_boosts: v.bool_or("noPPBoosts", false),
                contact: v.bool_or("contact", false),
                is_sound: v.bool_or("sound", false),
                no_sleep_talk: v.bool_or("noSleepTalk", false),
                is_charge: v.bool_or("isCharge", false),
                fail_encore: v.bool_or("failEncore", false),
                fail_mimic: v.bool_or("failMimic", false),
                is_snatchable: v.bool_or("isSnatchable", false),
                secondary_effects,
                secondary_boosts,
                self_boosts,
                self_drops,
                stat_drop_boosts,
            },
        );
    }
    Ok(out)
}
