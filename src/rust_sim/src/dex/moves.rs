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
    /// The dex display name. **PRIVATE ON PURPOSE** (`gen3_hidden_power_name_accessor_v1`):
    /// for a TYPED Hidden Power (nums 355-370) this is the typed name — `Hidden Power Grass` —
    /// and gen-3 HIDES the HP type, so emitting it is an INFORMATION LEAK, not a cosmetic byte
    /// difference. Reach it through [`MoveData::display_name`] (collapses typed HP, the ONLY
    /// thing an emitter may use) or [`MoveData::raw_name`] (explicit, never for emission).
    ///
    /// This was a leak TWICE — the round-8 BF1 fix collapsed three announce sites, and a
    /// FOURTH (the Disable `-start`) was later written the same way from this field and leaked
    /// again for months. Privacy makes the mistake unrepresentable rather than a convention.
    name: String,
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
    /// `flags.minimize` — the MINIMIZE volatile's `onSourceModifyDamage` DOUBLES this move's
    /// damage against a minimized target (`gen3_minimize_v1`). Data-driven precisely because
    /// the gen-3-legal carrier set (stomp / astonish / extrasensory / needlearm) is NOT the
    /// modern one — bodyslam & co. gained the flag in gen 9, so a hand-list written from
    /// current knowledge would be wrong.
    pub minimize_doubles: bool,
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
    /// The MULTI-STRIKE spec (`gen3_move_coverage_batch7_v1`, `multihit` in the data). `None` =
    /// a single hit. `Fixed(n)` = exactly n strikes (Double Kick / Twineedle / Bonemerang 2,
    /// Triple Kick 3). `Range(lo, hi)` = a variable count — the only gen3 shape is `[2, 5]`,
    /// whose count is `sample([2,2,2,3,3,3,4,5])` (gen<5) = ONE `random(8)` draw, drawn AFTER
    /// the whole-move accuracy roll and BEFORE the per-strike loop. Each strike runs the NORMAL
    /// damage path (crit + `random(16)` + the move's own secondary, per strike) then the
    /// per-strike `eachEvent('Update')`; the loop STOPS when the target faints. See
    /// `turn.rs::run_multihit`.
    pub multihit: Option<MultiHit>,
    /// `multiaccuracy` (`gen3_move_coverage_batch7_v1`): each strike AFTER the first re-rolls
    /// the move's accuracy, breaking on a miss (Triple Kick — the ONLY gen3 carrier, which ALSO
    /// escalates BP per strike). The engine FAIL-LOUDS on a multiaccuracy move rather than
    /// silently mismodel the per-strike accuracy re-roll; no gen3ou team carries Triple Kick.
    pub multiaccuracy: bool,
}

/// A move's multi-strike count spec (`gen3_move_coverage_batch7_v1`). See [`MoveData::multihit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiHit {
    /// Exactly `n` strikes, no count draw (Double Kick / Twineedle / Bonemerang 2, Triple Kick 3).
    Fixed(u8),
    /// A variable count over `[lo, hi]`; the only gen3 shape is `[2, 5]`, sampled from
    /// `[2,2,2,3,3,3,4,5]` (one draw). Stored as the raw pair for the sampler.
    Range(u8, u8),
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
    /// The name an EMITTER may use. Collapses a TYPED Hidden Power to the bare
    /// `Hidden Power`, because gen-3 hides the HP type (hidden information). Every
    /// `|move|` / `|cant|` / `|-start|` / fail line must go through this.
    pub fn display_name(&self) -> &str {
        if self.id.starts_with("hiddenpower") {
            "Hidden Power"
        } else {
            &self.name
        }
    }

    /// The RAW dex name, typed Hidden Power included. **Never emit this.** It exists for
    /// non-protocol consumers — packed-team round-tripping, diagnostics — where the typed
    /// name is the correct value. If you are formatting a `|...|` line, you want
    /// [`MoveData::display_name`].
    pub fn raw_name(&self) -> &str {
        &self.name
    }

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

        // The MULTI-STRIKE spec (`multihit`, only-when-present, `gen3_move_coverage_batch7_v1`):
        // a plain integer (Fixed — Double Kick 2, Triple Kick 3) or a 2-element `[lo, hi]` array
        // (Range — the only gen3 shape is `[2, 5]`). GIGO-guarded: a malformed array throws.
        let multihit = match v.get("multihit") {
            None => None,
            Some(j) => {
                if let Some(arr) = j.as_array() {
                    if arr.len() != 2 {
                        return Err(format!("move {id}: multihit array must be [lo, hi], got {arr:?}"));
                    }
                    let lo = arr[0].as_f64().map_or(0, |n| n as u8);
                    let hi = arr[1].as_f64().map_or(0, |n| n as u8);
                    Some(MultiHit::Range(lo, hi))
                } else if let Some(n) = j.as_f64() {
                    Some(MultiHit::Fixed(n as u8))
                } else {
                    return Err(format!("move {id}: multihit must be an int or [lo, hi]"));
                }
            }
        };

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
                minimize_doubles: v.bool_or("minimize", false),
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
                multihit,
                multiaccuracy: v.bool_or("multiaccuracy", false),
            },
        );
    }
    Ok(out)
}

#[cfg(test)]
mod hidden_power_name_tests {
    use super::*;

    /// `display_name` collapses EVERY typed Hidden Power (nums 355-370) plus the bare one,
    /// and leaves everything else alone. gen-3 HIDES the HP type, so a typed name reaching a
    /// `|...|` line is an information LEAK — the opponent is not supposed to know the type.
    #[test]
    fn display_name_collapses_every_typed_hidden_power() {
        let d = crate::dex::Dex::for_gen(3);
        let mut seen = 0usize;
        for t in [
            "bug", "dark", "dragon", "electric", "fighting", "fire", "flying", "ghost",
            "grass", "ground", "ice", "poison", "psychic", "rock", "steel", "water",
        ] {
            let id = format!("hiddenpower{t}");
            let m = d.moves(&id).unwrap_or_else(|| panic!("{id} must exist"));
            assert_eq!(
                m.display_name(),
                "Hidden Power",
                "{id}: display_name must hide the TYPE"
            );
            // NON-VACUITY: the raw name really is the typed one, so the collapse is doing work.
            assert_ne!(
                m.raw_name(),
                "Hidden Power",
                "{id}: raw_name is expected to carry the typed name — if it does not, this test \
                 proves nothing about the collapse"
            );
            seen += 1;
        }
        assert_eq!(seen, 16, "all 16 typed Hidden Powers must be covered");

        let bare = d.moves("hiddenpower").expect("bare HP");
        assert_eq!(bare.display_name(), "Hidden Power");

        // And an ORDINARY move is untouched by the collapse.
        let tackle = d.moves("tackle").expect("tackle");
        assert_eq!(tackle.display_name(), "Tackle");
        assert_eq!(tackle.raw_name(), "Tackle");
    }

    /// THE STRUCTURAL GATE. `MoveData::name` is PRIVATE, so an emitter physically cannot reach
    /// the typed name — it must go through `display_name` (collapsed) or `raw_name` (explicit).
    /// This test pins the small, deliberate set of `raw_name` callers so the exception cannot
    /// quietly grow: a new one is a decision someone has to make on purpose.
    ///
    /// The leak shipped TWICE before this — round-8 BF1 collapsed three announce sites by hand,
    /// and a FOURTH written the same way leaked for months. Enumerating the escapes is what
    /// turns "remember to collapse it" into something the compiler and this test enforce.
    #[test]
    fn raw_name_callers_are_an_enumerated_allowlist() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut found: Vec<String> = Vec::new();
        let mut stack = vec![root];
        while let Some(dir) = stack.pop() {
            for e in std::fs::read_dir(&dir).expect("readdir") {
                let p = e.expect("entry").path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    let txt = std::fs::read_to_string(&p).expect("read");
                    for (i, line) in txt.lines().enumerate() {
                        // ⚠️ Skip ONLY this file (where the accessor is defined). An earlier
                        // draft used `!p.ends_with("moves.rs")`, which ALSO excluded
                        // `src/turn/moves.rs` — the single biggest emitter in the crate, i.e.
                        // exactly the file most likely to leak. Mutation testing caught it:
                        // a raw_name() call planted there passed the gate. Compare the full
                        // relative path, never a filename suffix.
                        let is_this_file = p.ends_with("dex/moves.rs");
                        if line.contains("raw_name()") && !is_this_file {
                            found.push(format!(
                                "{}:{}",
                                p.file_name().unwrap().to_string_lossy(),
                                i + 1
                            ));
                        }
                    }
                }
            }
        }
        found.sort();
        let files: Vec<&str> = found.iter().map(|f| f.split(':').next().unwrap()).collect();
        assert_eq!(
            files,
            vec!["bridge.rs", "team.rs"],
            "raw_name() has a NEW caller. It is correct in exactly two places — the OWNER-ONLY \
             request display (bridge.rs) and packed-team round-tripping (team.rs). Anything \
             formatting a `|...|` protocol line must use display_name(), or it LEAKS the Hidden \
             Power type. If the new caller is genuinely non-protocol, add it here deliberately. \
             found: {found:?}"
        );
    }
}
