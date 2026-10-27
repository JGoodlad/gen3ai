//! Damage — the **Gen-3 single-hit damage calculation**, a self-contained,
//! bit-faithful port of Showdown's `getDamage` (`sim/battle-actions.ts:1589-1726`,
//! the SHARED base) + the gen-3-SPECIFIC `modifyDamage`
//! (`data/mods/gen3/scripts.ts:33-119` — gen3 OVERRIDES the base modifier chain;
//! the base-class `modifyDamage` is NOT used and must not be ported).
//!
//! **Why this is a separate function (not a `BattleState` method).** The
//! authoritative oracle (`utils/bridge/damage_probe.js`) constructs single-turn
//! scenarios with EXACT both-side stats; the cleanest validation is a pure
//! function over EXPLICIT inputs (the attacker/defender stats, types, boosts,
//! status, the move, the field, and a `crit` bool), so this module deliberately
//! does NOT require a full battle. The event engine (a later step) will call
//! `calc_damage` after resolving the per-mon facts.
//!
//! ## The two-stage split (load-bearing — the classic gen-3 trap)
//!
//! The source computes damage in TWO stages with DIFFERENT floor points; this
//! module mirrors that split exactly:
//!
//! 1. **`get_damage` (base).** Read the offensive/defensive stat via
//!    `calculateStat` (boost-table floor), apply the **STAT-event** modifiers
//!    (Choice Band ×1.5, the type-boost items ×1.1 / Sea Incense ×1.05, Flash Fire
//!    ×1.5) as 4096-fixed-point `modify` on the STAT, apply the **BASE-POWER**
//!    modifiers (gen3 **Thick Fat** is `onSourceBasePower ×0.5`, NOT a stat
//!    modifier — gen3 inherits gen4's override; it halves the move's BP for
//!    Ice/Fire), then the four nested `trunc`s:
//!    `tr(tr(tr(tr(2*level/5 + 2) * basePower * attack) / defense) / 50)`. All of
//!    these fold in BEFORE the formula, NOT into the modifier chain.
//! 2. **`modify_damage` (the gen-3 chain).** In EXACT order:
//!    burn ×0.5 → screens (Phase1, crit-bypassed) → weather → physical-min-1 →
//!    `+2` → crit ×2 (on `baseDamage+2`) → **`Math.floor` (Phase2)** → STAB ×1.5 →
//!    type effectiveness (raw `*2` / `floor(/2)`, NOT `modify`) → randomizer
//!    (85-100%) → final floor (min 1).
//!
//! The classic bug is porting the MAINLINE order (which burns LAST and randomizes
//! BEFORE STAB/type). Gen-3 burns FIRST and randomizes SECOND-TO-LAST.
//!
//! ## Integer math (no f64 where Showdown floors)
//!
//! `tr` = `(n >>> 0)` (u32 truncation, `dex.ts:365-368`); for the non-negative
//! values here it is a plain floor. `modify(v, num, den)` =
//! `tr((tr(v*modifier) + 2048 - 1) / 4096)` with `modifier = tr(num*4096/den)`
//! (`battle.ts:2345-2356`) — a fixed-point 4096 multiply with `+2047` rounding.
//! The ONLY non-4096 ops are the base formula's four `trunc`s, the
//! type-effectiveness `*2` / `floor(/2)` loop, and `randomizer`.
//!
//! ## Crit (gen-3 = ×2)
//!
//! A crit (`crit: true`): (a) multiplies `baseDamage+2` by 2 via `modify`
//! (`move.critModifier || 2`); (b) IGNORES the defender's POSITIVE def/spd boost
//! and the attacker's NEGATIVE atk/spa boost (zeroed at the stat layer, exactly
//! `battle-actions.ts:1690-1704`); (c) BYPASSES Reflect/Light Screen (the screen
//! handler's own crit guard). It does NOT ignore burn, weather, the attacker's
//! favourable boosts, STAB, or type. The crit RNG roll itself is the engine's job
//! (this function takes the resolved `crit` bool).
//!
//! ## Immunity
//!
//! A type-chart `0×` (e.g. Electric→Ground) or an ability immunity (Levitate vs
//! Ground, Flash Fire vs Fire, Water/Volt Absorb) returns **0 damage** — Showdown
//! handles these via `runImmunity` BEFORE the calc; we surface them through
//! `DamageContext::immune` (the caller resolves the ability/levitate rule) and the
//! type chart's `0×`. Either way `calc_damage` returns all-zero.
//!
//! Validated against the sim's EXACT realized damage on constructed single-modifier
//! scenarios (`harness/gen_damage_golden.js` forces the MAX damage roll so the
//! realized HP-delta == the deterministic pre-roll baseDamage — an EXACT check, not
//! a band) via `tests/damage_test.rs`.

use crate::dex::{Dex, MoveCategory, Type};

/// `trunc(num)` = `(num >>> 0)` (`dex.ts:365-368`). For the non-negative `u64`
/// intermediates here it is a plain floor; named so the formula reads like the
/// source. We use `u64` for the base formula (`2L/5+2` ≈ 42, `× BP × atk` can
/// reach ~42·250·~1500 ≈ 1.6e7, well within u64; matching JS's f64-exact integer
/// arithmetic for these magnitudes).
#[inline]
fn tr(num: u64) -> u64 {
    num
}

/// `battle.modify(value, num, den)` (`battle.ts:2345-2356`): the 4096 fixed-point
/// multiply with `+2047` rounding —
/// `modifier = tr(num*4096/den)`, then `tr((tr(value*modifier) + 2048 - 1) / 4096)`.
///
/// `num`/`den` express the rational multiplier exactly (so `1.5` is `(3,2)`, `0.5`
/// is `(1,2)`, `1.1` is `(11,10)`, `1.05` is `(21,20)`) — we never form a float
/// modifier, matching the integer chain bit-for-bit.
#[inline]
fn modify(value: u64, num: u64, den: u64) -> u64 {
    let modifier = tr(num * 4096 / den);
    tr((tr(value * modifier) + 2048 - 1) / 4096)
}

/// `chainModify` + `finalModify` (`battle.ts:2334`/`2394`): `runEvent('ModifyAtk'…)`
/// accumulates EVERY handler into ONE combined 4096 fixed-point modifier
/// (`acc = (acc*next + 2048) >> 12` per mod), then applies a SINGLE `modify`
/// (`+2047` round) at the end — NOT a per-mod `modify` call. Sequential per-mod
/// `modify` double-rounds and DIVERGES for 2+ stacked modifiers (e.g. a Flash-Fire
/// mon with Choice Band: ×1.5 + ×1.5 on a 359 Atk → chained 808 vs sequential 807).
/// An empty list returns `value` unchanged (byte-identical to the no-mod path, so
/// the single-modifier golden scenarios are unaffected).
#[inline]
pub(crate) fn chain_modify(value: u64, mods: &[(u64, u64)]) -> u64 {
    if mods.is_empty() {
        return value;
    }
    let mut acc: u64 = 4096;
    for &(num, den) in mods {
        let next = tr(num * 4096 / den);
        acc = (acc * next + 2048) >> 12;
    }
    tr((tr(value * acc) + 2048 - 1) / 4096)
}

/// The five non-HP stats, in the boost/calc order. (HP is never an attack/defense
/// stat.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stat {
    Atk,
    Def,
    Spa,
    Spd,
    Spe,
}

/// One side's relevant offensive *or* defensive facts for a single hit. The
/// `*_stat` values are the sim's OWN computed stats (the `storedStats` the oracle
/// reads), so Rust and Showdown share the exact base numbers.
#[derive(Debug, Clone)]
pub struct Combatant {
    pub level: u8,
    /// Computed Atk stat (pre-boost, pre-modifier) — the `storedStats.atk`.
    pub atk_stat: u16,
    /// Computed SpA stat.
    pub spa_stat: u16,
    /// Computed Def stat.
    pub def_stat: u16,
    /// Computed SpD stat.
    pub spd_stat: u16,
    /// The mon's types (1 or 2). STAB keys on these; effectiveness keys on the
    /// defender's.
    pub types: Vec<Type>,
    /// Stat-stage boosts in `[atk, def, spa, spd, spe]` order (no accuracy/evasion
    /// here — damage uses only these five). All 0 for an un-boosted mon.
    pub boosts: [i8; 5],
    /// Major status — only `brn` matters to damage (the physical burn halving).
    /// `true` == burned.
    pub burned: bool,
    /// Whether the mon has Guts (burn does NOT halve a Guts mon's physical damage).
    pub has_guts: bool,
}

impl Default for Combatant {
    fn default() -> Self {
        Combatant {
            level: 100,
            atk_stat: 0,
            spa_stat: 0,
            def_stat: 0,
            spd_stat: 0,
            types: Vec::new(),
            boosts: [0; 5],
            burned: false,
            has_guts: false,
        }
    }
}

impl Combatant {
    fn boost(&self, s: Stat) -> i8 {
        match s {
            Stat::Atk => self.boosts[0],
            Stat::Def => self.boosts[1],
            Stat::Spa => self.boosts[2],
            Stat::Spd => self.boosts[3],
            Stat::Spe => self.boosts[4],
        }
    }
    fn stat(&self, s: Stat) -> u16 {
        match s {
            Stat::Atk => self.atk_stat,
            Stat::Def => self.def_stat,
            Stat::Spa => self.spa_stat,
            Stat::Spd => self.spd_stat,
            // Speed is never an attack/defense stat in the damage formula — reaching
            // here is a programming error; fail loud rather than return wrong (0) damage.
            Stat::Spe => unreachable!("Speed is not an attack/defense stat in the damage formula"),
        }
    }
}

/// The active weather (only the two that touch damage in gen-3 OU). Sand's
/// Rock-SpD boost is gen-4+ ONLY (gen-3 nulls it — `gen3/conditions.ts:52-55`), so
/// Sand has no damage multiplier here and is intentionally omitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weather {
    Rain,
    Sun,
}

/// A STAT-event multiplier the attacker carries that folds into the offensive stat
/// in `get_damage` (BEFORE the base formula), expressed as an exact `(num, den)`
/// rational so `modify` stays integer. These are the gen-3 modifiers the spec
/// flags as stat-phase (NOT modifier-chain) modifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtkStatMod {
    /// Choice Band ×1.5 (PHYSICAL only).
    ChoiceBand,
    /// A type-boost item ×1.1 (when the move's type matches the item) — Magnet,
    /// Charcoal, Mystic Water, Miracle Seed, etc.
    TypeBoostItem,
    /// Sea Incense ×1.05.
    SeaIncense,
    /// Flash Fire ×1.5 (Fire moves, while the Flash Fire boost is active).
    FlashFire,
    /// A DATA-DRIVEN item stat modifier (`gen3_item_mechanics_v1`) carrying its exact
    /// rational from the dex (`ItemData.type_boost` at `fold=stat` / `stat_mods`) —
    /// the generic member the engine emits; the named variants above are kept for the
    /// pre-framework test mirrors (same ratios ⇒ byte-identical chains).
    Item { num: u64, den: u64 },
}

impl AtkStatMod {
    fn ratio(self) -> (u64, u64) {
        match self {
            AtkStatMod::ChoiceBand => (3, 2),
            AtkStatMod::TypeBoostItem => (11, 10),
            AtkStatMod::SeaIncense => (21, 20),
            AtkStatMod::FlashFire => (3, 2),
            AtkStatMod::Item { num, den } => (num, den),
        }
    }
}

/// A BASE-POWER-phase modifier (`runEvent('BasePower')`, gen3: AFTER the crit roll,
/// BEFORE the base formula) — the fold the gen4-named incenses, the gen2 bows, and
/// (already) defender Thick Fat live at. Two Showdown handler shapes with DIFFERENT
/// rounding:
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BpMod {
    /// `chainModify(num/den)` — accumulates into the event's ONE 4096 fixed-point
    /// modifier, applied ONCE (`modify`, +2047 round) at the end of the event
    /// (the incenses ×4915/4096; Thick Fat ×1/2 joins this same chain).
    Chain(u64, u64),
    /// `return basePower * (num/den)` (Pink Bow / Polkadot Bow ×1.1) — a DIRECT float
    /// return that REPLACES the event's relayVar. Mirrored EXACTLY: the product is a
    /// non-integer f64, so runEvent's final-modifier guard (`relayVar ===
    /// Math.abs(Math.floor(relayVar))`) SKIPS the accumulated chain entirely, and
    /// `clampIntRange(basePower, 1)` floors the float afterwards. `num/den` in f64
    /// (11.0/10.0) is bit-identical to the JS literal `1.1`.
    Direct(u64, u64),
}

/// The move being used, reduced to the damage-relevant facts. Built from the dex
/// via [`MoveInput::from_dex`], or hand-constructed for a synthetic move.
#[derive(Debug, Clone)]
pub struct MoveInput {
    /// `flags.minimize` — doubles this move's damage against a MINIMIZED target at the final
    /// ModifyDamage step (`gen3_minimize_v1`). Data-driven from the dex row.
    pub minimize_doubles: bool,
    pub base_power: u16,
    /// `None` == typeless (`"???"` — no STAB, neutral effectiveness).
    pub move_type: Option<Type>,
    pub category: MoveCategory,
    /// Gen-3 Explosion/Self-Destruct halve the defender's Def (`gen<=4` rule,
    /// `battle-actions.ts:1716`). Set for those two moves.
    pub halves_defense: bool,
}

impl MoveInput {
    /// Look a move up in the dex and reduce it to the damage-relevant fields.
    /// Returns `None` for an unknown move id.
    pub fn from_dex(move_id: &str, dex: &Dex) -> Option<MoveInput> {
        let m = dex.moves(move_id)?;
        let id = crate::dex::to_id(move_id);
        Some(MoveInput {
            minimize_doubles: m.minimize_doubles,
            base_power: m.base_power,
            move_type: m.move_type,
            category: m.category,
            halves_defense: id == "explosion" || id == "selfdestruct",
        })
    }
}

/// All explicit inputs for one single-hit damage computation. Take EXPLICIT inputs
/// (not a `BattleState`) so any scenario can be constructed directly.
#[derive(Debug, Clone)]
pub struct DamageContext {
    pub attacker: Combatant,
    pub defender: Combatant,
    pub mv: MoveInput,
    /// A resolved crit (the engine rolls the crit RNG; this is the outcome). ×2
    /// damage + ignore-boosts/screens, exactly the gen-3 crit rule.
    pub crit: bool,
    /// Active weather (gen-3-damage-relevant only), or `None`.
    pub weather: Option<Weather>,
    /// The DEFENDER's side has Reflect up (halves incoming PHYSICAL, crit-bypassed).
    pub reflect: bool,
    /// The DEFENDER's side has Light Screen up (halves incoming SPECIAL,
    /// crit-bypassed).
    pub light_screen: bool,
    /// Attacker STAT-event multipliers (Choice Band / type item / Sea Incense / the
    /// ModifyAtk ability folds Huge/Pure Power / Guts). Applied to the offensive stat, in
    /// this order, BEFORE the base formula. The caller resolves which apply (item/ability +
    /// type match); the calc just folds them. (Flash Fire is NOT here — the probe proved it
    /// is a ModifyDamagePhase1 damage fold, `flash_fire` below, not a stat mod.)
    pub atk_stat_mods: Vec<AtkStatMod>,
    /// Hustle's Atk ×3/2 (`gen3_accuracy_pipeline_v1`): a `this.modify(atk, 1.5)` that
    /// REPLACES the relayVar (its OWN rounding), NOT a `chainModify` accumulation — the
    /// gen3 Hustle onModifyAtk comment says "applied directly to the stat as opposed to
    /// chaining with the others". Applied as a SEPARATE `modify` BEFORE the `atk_stat_mods`
    /// chain (Hustle's onModifyAtk priority 5 runs first, then the chain members
    /// accumulate on the already-Hustle-modified stat). `None` ⇒ no Hustle. For a LONE
    /// Hustle this equals a single chain member; the separate round only differs when
    /// Hustle stacks with another Atk mod (rare — Hustle mons don't run CB competitively).
    pub atk_direct_modify: Option<(u64, u64)>,
    /// DEFENDER STAT-event multipliers (`runEvent('ModifyDef'/'ModifySpD')` —
    /// DeepSeaScale / Metal Powder / Soul Dew SpD). Folded into the defensive stat
    /// with the same accumulate-once chain, AFTER the boost table and BEFORE the
    /// gen<=4 Explosion Def-halve (the `getDamage` order). Empty ⇒ byte-identical
    /// no-op.
    pub def_stat_mods: Vec<AtkStatMod>,
    /// ATTACKER-item BASE-POWER-phase modifiers (the incense chain / the bow direct
    /// float — see [`BpMod`]). Defender Thick Fat keeps its dedicated flag below and
    /// joins the same accumulated chain. Empty ⇒ byte-identical no-op.
    pub bp_mods: Vec<BpMod>,
    /// Thick Fat on the DEFENDER (halves the attacker's Atk/SpA for Ice/Fire moves
    /// — `onSourceModifyAtk/SpA ×0.5`). The caller sets this when the defender has
    /// Thick Fat AND the move is Ice/Fire.
    pub defender_thick_fat: bool,
    /// The DEFENDER holds the MINIMIZE volatile (`gen3_minimize_v1`). Combined with the
    /// move's own `minimize_doubles` flag it doubles the damage at the FINAL ModifyDamage
    /// step. Note gen-3 has no `onAccuracy` bypass, so this is the volatile's ONLY effect
    /// beyond the evasion stage the accuracy pipeline already folds.
    pub defender_minimized: bool,
    /// Hard immunity (an ability/Levitate immunity the caller resolved — e.g.
    /// Levitate vs Ground, Water Absorb vs Water). A type-chart `0×` is detected
    /// internally from the chart; this covers the ABILITY immunities the chart
    /// can't express. Either way → 0 damage.
    pub immune: bool,
    /// FLASH FIRE ×1.5 on the ATTACKER for THIS Fire move (`flashfire` volatile's
    /// `onModifyDamagePhase1 chainModify(1.5)`). The caller sets this when the attacker has
    /// an ARMED `flash_fire` volatile AND the move is Fire-type. It is a ModifyDamagePhase1
    /// DAMAGE fold (the SAME phase as Reflect/Light Screen — probe-settled: the flashfire
    /// condition's `onModifyAtk`/`onModifySpA` are `undefined`, only `onModifyDamagePhase1`
    /// carries the boost), so it applies to BOTH physical AND special Fire moves and is NOT
    /// crit-bypassed (no crit guard on the handler). Folded WITH any screen into ONE
    /// accumulated chain modifier so the combined result is bit-exact. `false` ⇒ no-op.
    pub flash_fire: bool,
}

/// The result of a single-hit damage computation: the deterministic pre-roll
/// `base` (the MAX-roll damage, `random(16)==0`) and the full 16-roll spread
/// `rolls[r] = randomizer with r` for `r` in `0..16`. `rolls[0] == base`,
/// `rolls[15]` is the 85% min roll.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DamageResult {
    /// The deterministic damage at `random(16)==0` (the max roll, `100/100`).
    pub base: u16,
    /// The 16 possible realized values (`r` in `0..16`); `rolls[0] == base`.
    pub rolls: [u16; 16],
}

impl DamageResult {
    /// An all-zero result (immunity / 0-BP).
    fn zero() -> DamageResult {
        DamageResult { base: 0, rolls: [0; 16] }
    }
}

/// Compute the Gen-3 single-hit damage for `ctx`. Returns the deterministic base
/// (max-roll) damage plus the 16-roll spread. Bit-faithful to Showdown's
/// `getDamage` + gen-3 `modifyDamage`.
///
/// 0 damage is returned for an immunity (ability/Levitate via `ctx.immune`, or a
/// type-chart `0×`) and for a 0-base-power move.
pub fn calc_damage(ctx: &DamageContext, dex: &Dex) -> DamageResult {
    // Status moves / 0-BP never reach the damage formula.
    if ctx.mv.base_power == 0 || ctx.mv.category == MoveCategory::Status {
        return DamageResult::zero();
    }
    // Ability / Levitate immunity (caller-resolved) → 0.
    if ctx.immune {
        return DamageResult::zero();
    }

    // Type effectiveness (the product over the defender's types). A 0× here is a
    // type immunity (Electric→Ground etc.) → 0 damage, handled like runImmunity.
    let move_type = ctx.mv.move_type;
    let type_mult: f64 = match move_type {
        Some(t) => dex.type_chart().effectiveness(t, &ctx.defender.types),
        None => 1.0, // typeless: neutral
    };
    if type_mult == 0.0 {
        return DamageResult::zero();
    }

    let base = get_base_damage(ctx);
    let rolled = modify_damage(base, ctx, type_mult);
    rolled
}

/// **Stage 1 — `get_damage`** (the shared base, `battle-actions.ts`): resolve the
/// offensive/defensive stats (boost-table floor + the stat-event modifiers) and
/// apply the four-nested-trunc base formula. Returns the integer `baseDamage` fed
/// to `modify_damage`.
fn get_base_damage(ctx: &DamageContext) -> u64 {
    let is_physical = ctx.mv.category == MoveCategory::Physical;
    let (atk_stat, def_stat) = if is_physical {
        (Stat::Atk, Stat::Def)
    } else {
        (Stat::Spa, Stat::Spd)
    };

    // --- crit boost-ignore (battle-actions.ts:1690-1704) ---
    // A crit zeros the attacker's NEGATIVE offensive boost and the defender's
    // POSITIVE defensive boost (keeps the attacker's favourable ones).
    let mut atk_boost = ctx.attacker.boost(atk_stat);
    let mut def_boost = ctx.defender.boost(def_stat);
    if ctx.crit {
        if atk_boost < 0 {
            atk_boost = 0;
        }
        if def_boost > 0 {
            def_boost = 0;
        }
    }

    // --- calculateStat: boost table + Math.floor (pokemon.ts:561-595) ---
    let attack = apply_boost(ctx.attacker.stat(atk_stat) as u64, atk_boost);
    let mut defense = apply_boost(ctx.defender.stat(def_stat) as u64, def_boost);

    // --- stat-event modifiers on the ATTACK stat (runEvent ModifyAtk/SpA) ---
    // Showdown's runEvent accumulates EVERY handler into ONE combined 4096 modifier
    // (chainModify) and rounds ONCE (finalModify) — NOT a per-mod `modify`. Folding
    // them here keeps a multi-modifier stat (e.g. Flash Fire + Choice Band) bit-exact.
    // (Choice Band ×1.5, type-boost items ×1.1, Sea Incense ×1.05, Flash Fire ×1.5,
    // the species stat items — Thick Club/Light Ball/DeepSeaTooth/Soul Dew SpA.)
    // Hustle's Atk ×3/2 is a SEPARATE `this.modify(atk, 1.5)` (its own rounding, priority
    // 5) applied BEFORE the chain — NOT a chainModify accumulation (`gen3_accuracy_pipeline_v1`).
    let attack = match ctx.atk_direct_modify {
        Some((n, d)) => modify(attack, n, d),
        None => attack,
    };
    let atk_mods: Vec<(u64, u64)> = ctx.atk_stat_mods.iter().map(|m| m.ratio()).collect();
    let attack = chain_modify(attack, &atk_mods);

    // --- stat-event modifiers on the DEFENSE stat (runEvent ModifyDef/SpD) ---
    // `getDamage` runs the defender's stat event right after the attacker's and
    // BEFORE the gen<=4 Explosion Def-halve (battle-actions.ts:1448-1452). The gen3
    // members: DeepSeaScale (SpD ×2 Clamperl), Metal Powder (Def ×2 untransformed
    // Ditto), Soul Dew (SpD ×1.5 Lati@s). Same accumulate-once chain; a crit ignores
    // the defender's POSITIVE BOOST (above) but NOT these stat-event items.
    let def_mods: Vec<(u64, u64)> = ctx.def_stat_mods.iter().map(|m| m.ratio()).collect();
    defense = chain_modify(defense, &def_mods);

    // --- BASE-POWER modifiers (runEvent BasePower, gen3, AFTER crit, pre-formula) ---
    // Thick Fat in gen3 is `onSourceBasePower chainModify(0.5)` (gen3 inherits the
    // gen4 override `data/mods/gen4/abilities.ts:491`, NOT the modern
    // `onSourceModifyAtk/SpA` stat modifier) — so it halves the move's BASE POWER for
    // Ice/Fire moves, not the attack stat. (This is a gen3-specific fact the modern
    // data hides; pinned by the `thick_fat_ice` golden = base 38, not 39.)
    //
    // `gen3_item_mechanics_v1` adds the item members of the SAME event: the gen4-named
    // incenses (`chainModify([4915,4096])` — joins Thick Fat in the ONE accumulated
    // event modifier, rounded once) and the gen2 bows (`return basePower * 1.1` — a
    // DIRECT float return replacing relayVar). Mirroring `runEvent`'s tail exactly
    // (battle.ts:709): the accumulated chain is applied ONLY when relayVar is still a
    // non-negative INTEGER-VALUED number — a direct float product USUALLY makes it
    // non-integer and SKIPS it, BUT NOT ALWAYS: `70 * 1.1 == 77` exactly in f64, so a
    // BP-70 bow-boosted move keeps an integer relayVar and the chain RE-APPLIES —
    // then `clampIntRange(basePower, 1)` truncs. The direct/chain conditions are NOT
    // type-disjoint since `gen3_facade_v1` (Facade is a Normal-type CHAIN member, the
    // bows are Normal-type DIRECTs): a poisoned Pink-Bow Facade is BP
    // modify(77, ×2) = 154 (probe `harness/probe_facade_gen3.js` + the FA-d pin).
    let mut bp = ctx.mv.base_power.max(1) as u64; // clampIntRange(>=1)
    let mut bp_float: Option<f64> = None;
    let mut bp_chain: Vec<(u64, u64)> = Vec::new();
    for m in &ctx.bp_mods {
        match *m {
            BpMod::Direct(n, d) => {
                let cur = bp_float.unwrap_or(bp as f64);
                bp_float = Some(cur * (n as f64 / d as f64));
            }
            BpMod::Chain(n, d) => bp_chain.push((n, d)),
        }
    }
    if ctx.defender_thick_fat {
        bp_chain.push((1, 2));
    }
    bp = match bp_float {
        // Direct handler fired → relayVar is a float. The sim's runEvent tail
        // (battle.js:709) applies the accumulated chain modifier iff relayVar is
        // STILL a non-negative INTEGER-VALUED number
        // (`relayVar === Math.abs(Math.floor(relayVar))`); otherwise the chain is
        // SKIPPED and clampIntRange truncs the float. This is the EXACT guard —
        // the old "Direct discards the chain" shortcut broke the day a Direct and a
        // Chain member could co-fire on ONE move: `gen3_facade_v1` made Facade a
        // Normal-type BP-CHAIN member (×2 when statused) that composes with the
        // Normal-type ×1.1 bows, and `70 * 1.1 == 77` EXACTLY in f64 → the guard
        // PASSES and the chain re-applies (probe `harness/probe_facade_gen3.js`:
        // Pink Bow + poisoned Facade = BP 154 = modify(77, ×2), NOT 77). A
        // non-integer float (e.g. 75 × 1.1 = 82.5) still skips the chain. The f64
        // product is bit-identical to JS's (same IEEE-754 double math).
        Some(f) => {
            if f >= 0.0 && f == f.floor() {
                chain_modify(f as u64, &bp_chain)
            } else {
                f as u64 // clampIntRange truncs the float; chain skipped
            }
        }
        None => chain_modify(bp, &bp_chain),
    };
    bp = bp.max(1);

    // --- gen<=4 Explosion/Self-Destruct: defender Def halved (clampIntRange >=1) ---
    if ctx.mv.halves_defense && def_stat == Stat::Def {
        defense = tr(defense / 2).max(1);
    }

    // --- base formula: tr(tr(tr(tr(2*L/5 + 2) * BP * atk) / def) / 50) ---
    let level = ctx.attacker.level as u64;
    let step1 = tr(2 * level / 5 + 2);
    let step2 = tr(step1 * bp * attack);
    let step3 = tr(step2 / defense.max(1));
    tr(step3 / 50)
}

/// `calculateStat`'s boost-table application (`pokemon.ts:584-591`):
/// `boost>=0 -> floor(stat * boostTable[boost])`, `boost<0 -> floor(stat /
/// boostTable[-boost])`, with `boostTable = [1,1.5,2,2.5,3,3.5,4]`. The trailing
/// `modify(stat, 1)` is identity (modifier 1) so it is omitted. We keep the
/// fraction exact (×3/2, ×2, ×5/2, …) to floor identically to JS's `* 1.5` for
/// these magnitudes.
fn apply_boost(stat: u64, boost: i8) -> u64 {
    let b = boost.clamp(-6, 6);
    // boostTable entries as exact (num, den): [1, 3/2, 2, 5/2, 3, 7/2, 4].
    const TABLE: [(u64, u64); 7] = [(1, 1), (3, 2), (2, 1), (5, 2), (3, 1), (7, 2), (4, 1)];
    if b >= 0 {
        let (n, d) = TABLE[b as usize];
        stat * n / d
    } else {
        let (n, d) = TABLE[(-b) as usize];
        // floor(stat / (n/d)) = floor(stat * d / n)
        stat * d / n
    }
}

/// **Stage 2 — gen-3 `modifyDamage`** (`data/mods/gen3/scripts.ts:33-119`): apply
/// the modifier chain in the EXACT gen-3 order, then the 16-roll randomizer.
/// `type_mult` is the precomputed product over the defender's types (>0 here;
/// immunity short-circuited in `calc_damage`).
fn modify_damage(base: u64, ctx: &DamageContext, type_mult: f64) -> DamageResult {
    let is_physical = ctx.mv.category == MoveCategory::Physical;
    let mut bd = base;

    // 1. BURN ×0.5 — physical, non-Guts, burned, and baseDamage != 0.
    if ctx.attacker.burned && bd != 0 && is_physical && !ctx.attacker.has_guts {
        bd = modify(bd, 1, 2);
    }

    // 2. ModifyDamagePhase1 (`runEvent('ModifyDamagePhase1')`): the SCREENS (Reflect/Light
    //    Screen ×0.5, crit-BYPASSED) AND FLASH FIRE (×1.5 for a Fire move, NOT crit-bypassed).
    //    The sim accumulates ALL Phase1 `chainModify` handlers into ONE 4096 modifier applied
    //    once (finalModify) — so when a screen AND Flash Fire both fire (a physical Fire move
    //    by an FF-armed attacker into a Reflect defender) the port MUST chain them together,
    //    not apply two sequential `modify` rounds (which double-round → diverge for ~¼ of bd
    //    values; probe-confirmed). Reflect halves physical, Light Screen halves special. Flash
    //    Fire's handler has no crit guard, so it applies under crit too; the screens are
    //    crit-bypassed (the gen-3 crit ignores screens).
    {
        let mut phase1: Vec<(u64, u64)> = Vec::new();
        // Flash Fire ×1.5 (the flashfire volatile `onModifyDamagePhase1`): the resolved handler
        // gates on `move.type === 'Fire'`, so mirror that here (the caller ALSO only sets
        // `flash_fire` for a Fire move — a defensive double-gate so a mis-set flag on a non-Fire
        // move is inert). Applies regardless of crit (no crit guard on the handler).
        let is_fire = ctx.mv.move_type == Some(Type::Fire);
        if ctx.flash_fire && is_fire {
            phase1.push((3, 2));
        }
        // Screens ×0.5 — crit-BYPASSED (the gen-3 crit ignores Reflect/Light Screen).
        if !ctx.crit {
            if is_physical && ctx.reflect {
                phase1.push((1, 2));
            }
            if !is_physical && ctx.light_screen {
                phase1.push((1, 2));
            }
        }
        bd = chain_modify(bd, &phase1);
    }

    // 3. spread modifier — singles only here (always 1×), skipped.

    // 4. WeatherModifyDamage — rain/sun ×1.5 / ×0.5 on Water/Fire.
    bd = apply_weather(bd, ctx);

    // 5. physical min-1 BEFORE +2.
    if is_physical && bd == 0 {
        bd = 1;
    }

    // 6. +2.
    bd += 2;

    // 7. CRIT ×2 (on baseDamage+2), via modify(critModifier || 2).
    if ctx.crit {
        bd = modify(bd, 2, 1);
    }

    // 8. Math.floor(ModifyDamagePhase2) — Phase2 has no handler in our scope; the
    //    floor is on the (already-integer) value, so it's identity here. (The
    //    explicit floor matters only if a Phase2 modifier produced a fraction.)
    //    bd stays integer.

    // 9. STAB ×1.5 (only if the move has a type AND the attacker shares it).
    if let Some(t) = ctx.mv.move_type {
        if ctx.attacker.types.contains(&t) {
            bd = modify(bd, 3, 2);
        }
    }

    // 10. TYPE effectiveness — raw *2 per +stage, floor(/2) per -stage (NOT modify).
    //     type_mult ∈ {0.25, 0.5, 1, 2, 4}; recover the integer stage count.
    let stage = type_stage(type_mult);
    if stage > 0 {
        for _ in 0..stage {
            bd *= 2;
        }
    } else if stage < 0 {
        for _ in 0..(-stage) {
            bd /= 2; // integer division == Math.floor for non-negative bd
        }
    }

    // 11. ModifyDamage — the sim's FINAL modifier, after the type multiplier and before the
    //     randomizer (gen3 `scripts.ts:109`). MINIMIZE (`gen3_minimize_v1`) is the one member
    //     in scope: the volatile's `onSourceModifyDamage` is
    //     `if (move.flags['minimize']) return this.chainModify(2)`, so a move carrying the
    //     FLAG deals double damage to a minimized target. Probe-measured EXACTLY ×2 on a
    //     controlled pair (Double Team vs Minimize, same seed, byte-identical draw streams:
    //     Stomp 86 → 172, Extrasensory 31 → 62, and Tackle — no flag — 48 in both).
    if ctx.defender_minimized && ctx.mv.minimize_doubles {
        bd = modify(bd, 2, 1);
    }

    // 12 + 13. randomizer (85-100%) then final floor (min 1), over all 16 rolls.
    randomize_all(bd)
}

/// rain/sun `WeatherModifyDamage` (`conditions.ts:486-496` / `556-569`): a Water
/// move is ×1.5 in rain / ×0.5 in sun; a Fire move is ×0.5 in rain / ×1.5 in sun.
/// Applied via `modify` (the chainModify the event accumulates).
fn apply_weather(bd: u64, ctx: &DamageContext) -> u64 {
    let Some(w) = ctx.weather else { return bd };
    let Some(t) = ctx.mv.move_type else { return bd };
    match (w, t) {
        (Weather::Rain, Type::Water) => modify(bd, 3, 2),
        (Weather::Rain, Type::Fire) => modify(bd, 1, 2),
        (Weather::Sun, Type::Fire) => modify(bd, 3, 2),
        (Weather::Sun, Type::Water) => modify(bd, 1, 2),
        _ => bd,
    }
}

/// Recover the integer effectiveness STAGE from the float multiplier: +2 (4×), +1
/// (2×), 0 (1×), -1 (0.5×), -2 (0.25×). Clamped to [-6,6] like the source (a
/// dual-type stack can't exceed ±2 in gen-3 but the clamp is kept for fidelity).
fn type_stage(mult: f64) -> i32 {
    // mult is a product of chart entries (each 0.5/1/2), so it is one of the exact
    // powers of two; log2 recovers the stage. Use tolerant comparisons.
    let stage = if mult >= 3.5 {
        2
    } else if mult >= 1.5 {
        1
    } else if mult <= 0.3 {
        -2
    } else if mult <= 0.75 {
        -1
    } else {
        0
    };
    stage.clamp(-6, 6)
}

/// `randomizer` over all 16 rolls (`battle.ts:2404-2407`):
/// `rolls[r] = tr(tr(bd * (100 - r)) / 100)` for `r` in `0..16`, then the final
/// `if !floor return 1` min-1. `rolls[0]` (r==0) is `bd` unchanged (the max roll).
fn randomize_all(bd: u64) -> DamageResult {
    let mut rolls = [0u16; 16];
    for (r, slot) in rolls.iter_mut().enumerate() {
        let rolled = tr(tr(bd * (100 - r as u64)) / 100);
        let final_dmg = if rolled == 0 { 1 } else { rolled };
        *slot = final_dmg as u16;
    }
    DamageResult { base: rolls[0], rolls }
}

#[cfg(test)]
mod tests {
    use super::*;

    // modify() exact fixed-point checks against the documented values.
    #[test]
    fn modify_fixed_point() {
        // 0.5 -> modifier 2048: modify(100, 1, 2) = tr((tr(100*2048)+2047)/4096)
        //   = (204800+2047)/4096 = 206847/4096 = 50.
        assert_eq!(modify(100, 1, 2), 50);
        // 1.5 -> modifier 6144: modify(100,3,2) = (614400+2047)/4096 = 616447/4096 = 150.
        assert_eq!(modify(100, 3, 2), 150);
        // 2.0 -> modifier 8192: modify(100,2,1) = (819200+2047)/4096 = 821247/4096 = 200.
        assert_eq!(modify(100, 2, 1), 200);
        // 1.1 -> modifier tr(11*4096/10)=4505: modify(100,11,10)=(450500+2047)/4096
        //   = 452547/4096 = 110.
        assert_eq!(modify(100, 11, 10), 110);
    }

    // apply_boost mirrors the [1,1.5,2,2.5,3,3.5,4] table with Math.floor.
    #[test]
    fn boost_table() {
        assert_eq!(apply_boost(100, 0), 100);
        assert_eq!(apply_boost(100, 1), 150); // ×1.5
        assert_eq!(apply_boost(100, 2), 200); // ×2
        assert_eq!(apply_boost(101, 1), 151); // floor(101*1.5)=151
        assert_eq!(apply_boost(100, -1), 66); // floor(100/1.5)=66
        assert_eq!(apply_boost(100, -2), 50); // /2
        assert_eq!(apply_boost(100, 6), 400); // ×4 (clamped)
        assert_eq!(apply_boost(100, 7), 400); // clamp >6 -> 6
    }

    // type_stage recovers the integer stage from the float product.
    #[test]
    fn type_stage_recovery() {
        assert_eq!(type_stage(4.0), 2);
        assert_eq!(type_stage(2.0), 1);
        assert_eq!(type_stage(1.0), 0);
        assert_eq!(type_stage(0.5), -1);
        assert_eq!(type_stage(0.25), -2);
    }

    // chain_modify == chainModify+finalModify: single mod is byte-identical to
    // modify(), but a 2-mod stack rounds ONCE (not per-mod) and can diverge.
    #[test]
    fn chain_modify_matches_sim() {
        assert_eq!(chain_modify(359, &[]), 359); // no mods → unchanged
        assert_eq!(chain_modify(359, &[(3, 2)]), modify(359, 3, 2)); // single == modify
        assert_eq!(chain_modify(100, &[(11, 10)]), modify(100, 11, 10));
        // Flash Fire ×1.5 + Choice Band ×1.5 on a 359 Atk: chainModify acc = 9216
        // (2.25 in 4096 fp) → floor((359*9216 + 2047)/4096) = 808. The buggy
        // sequential modify(modify(359,3,2),3,2) double-rounds low to 807.
        assert_eq!(chain_modify(359, &[(3, 2), (3, 2)]), 808);
        assert_eq!(modify(modify(359, 3, 2), 3, 2), 807, "sequential under-rounds (the bug)");
    }

    // crit zeros the attacker's NEGATIVE offensive boost but keeps a positive one
    // (battle-actions.ts:1690). The defender-positive-Def half is golden-tested.
    #[test]
    fn crit_ignores_attacker_negative_boost() {
        let dex = Dex::for_gen(3);
        let dmg = |atk_boost: i8, crit: bool| {
            let mut a = Combatant { level: 100, atk_stat: 300, types: vec![Type::Normal], ..Default::default() };
            a.boosts[0] = atk_boost; // Atk stage
            let d = Combatant { level: 100, def_stat: 200, types: vec![Type::Normal], ..Default::default() };
            let ctx = DamageContext {
                attacker: a,
                defender: d,
                mv: MoveInput { minimize_doubles: false, base_power: 100, move_type: Some(Type::Fighting), category: MoveCategory::Physical, halves_defense: false },
                crit,
                weather: None,
                reflect: false,
                light_screen: false,
                atk_stat_mods: vec![],
                atk_direct_modify: None,
                def_stat_mods: vec![],
                bp_mods: vec![],
                defender_thick_fat: false,
                defender_minimized: false,
                immune: false,
                flash_fire: false,
            };
            calc_damage(&ctx, &dex).base
        };
        assert_eq!(dmg(-1, true), dmg(0, true), "crit zeros a -1 attacker boost");
        assert!(dmg(2, true) > dmg(0, true), "crit keeps a +2 attacker boost");
        assert!(dmg(-1, false) < dmg(0, false), "no crit: -1 boost DOES reduce damage");
    }

    #[test]
    fn move_input_from_dex_maps_fields() {
        let dex = Dex::for_gen(3);
        assert!(MoveInput::from_dex("explosion", &dex).unwrap().halves_defense);
        assert!(MoveInput::from_dex("selfdestruct", &dex).unwrap().halves_defense);
        assert!(!MoveInput::from_dex("earthquake", &dex).unwrap().halves_defense);
        let tb = MoveInput::from_dex("thunderbolt", &dex).unwrap();
        assert_eq!(tb.base_power, 95);
        assert_eq!(tb.category, MoveCategory::Special);
        assert_eq!(tb.move_type, Some(Type::Electric));
        assert!(MoveInput::from_dex("notarealmove", &dex).is_none());
    }

    // randomizer max roll == base, min roll == 85% (floored).
    #[test]
    fn randomizer_spread() {
        let r = randomize_all(100);
        assert_eq!(r.rolls[0], 100); // r=0 -> 100/100
        assert_eq!(r.base, 100);
        assert_eq!(r.rolls[15], 85); // r=15 -> tr(tr(100*85)/100)=85
        // monotone non-increasing
        for w in r.rolls.windows(2) {
            assert!(w[0] >= w[1]);
        }
    }
}
