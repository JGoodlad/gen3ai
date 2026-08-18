//! Damage-calc tests:
//!   - `damage_golden_matches_showdown` — the DIFFERENTIAL gate: reconstruct each
//!     scenario's `DamageContext` from `harness/gen_damage_golden.js`'s golden and
//!     assert `calc_damage(&ctx, dex).base == the sim's EXACT max-roll damage`
//!     (and `rolls[15] == the sim's min roll`). The golden FORCED the max damage
//!     roll (`random(16)==0`) so the realized HP-delta equals the deterministic
//!     pre-roll baseDamage — an EXACT equality, not a range-membership band. Every
//!     scenario isolates one mechanic (neutral / STAB / SE / resist / 4x /
//!     type-immune / ability-immune / Thick Fat / Choice Band / type-item / Sea
//!     Incense / burn / Reflect / Light Screen / rain / sun / +Atk / -Def /
//!     defender +Def under crit / crit-through-screen / Explosion def-halve /
//!     min-vs-max stats / low level — PLUS the `gen3_item_mechanics_v1` item class:
//!     the gen2 bows [DIRECT ×1.1 base-power float], the 4 gen4-named incenses
//!     [×4915/4096 base-power chain — NOT ×1.1], the species stat items
//!     [Thick Club / gen3 SpA-only Light Ball / DeepSea* / Metal Powder / Soul Dew,
//!     atk- AND def-side] with wrong-type/wrong-species controls).
//!   - `damage_smoke` — a few authoritative hand-checks of the modifier ORDER +
//!     crit/STAB/type integer math, independent of the golden.
//!
//! Mirrors the structure of `tests/stats_test.rs`.

use pokesim::damage::{calc_damage, AtkStatMod, BpMod, Combatant, DamageContext, MoveInput, Weather};
use pokesim::dex::{Dex, DmgFold, MoveCategory, Type, TypeBoostFold};

fn dex() -> Dex {
    Dex::for_gen(3)
}

fn parse_types(s: &str) -> Vec<Type> {
    s.split(',')
        .filter(|t| *t != "???" && !t.is_empty())
        .map(|t| Type::from_name(t).unwrap_or_else(|| panic!("unknown type {t:?}")))
        .collect()
}

fn parse_boosts(s: &str) -> [i8; 5] {
    let v: Vec<i8> = s.split(',').map(|x| x.parse().unwrap()).collect();
    assert_eq!(v.len(), 5, "boosts need 5 values, got {s:?}");
    [v[0], v[1], v[2], v[3], v[4]]
}

fn parse_category(s: &str) -> MoveCategory {
    match s {
        "Physical" => MoveCategory::Physical,
        "Special" => MoveCategory::Special,
        "Status" => MoveCategory::Status,
        _ => panic!("unknown category {s:?}"),
    }
}

/// Resolve the attacker's STAT-event modifiers from the golden's (item, species,
/// move-type, category) — DATA-DRIVEN off the dex's `gen3_item_mechanics_v1` fields
/// (an independent mirror of `turn.rs::resolve_atk_stat_mods` exercising the SAME
/// committed data): Choice Band ×1.5 physical + the species stat items on the
/// category's offensive stat (`statMods`, species-gated), and the stat-fold type
/// boosters (`typeBoost.fold == Stat`) when the move type matches. (No scenario
/// stages an attacking Flash Fire, which would be an `AtkStatMod` too.)
#[allow(clippy::too_many_arguments)]
fn resolve_atk_mods(
    item: &str,
    ability: &str,
    species: &str,
    move_type: Option<Type>,
    category: MoveCategory,
    attacker_statused: bool,
    dex: &Dex,
) -> Vec<AtkStatMod> {
    let mut mods = Vec::new();
    if let Some(it) = dex.item(item) {
        if let Some(sm) = &it.stat_mods {
            if sm.only_species.is_empty() || sm.only_species.iter().any(|s| s == species) {
                let ratio = match category {
                    MoveCategory::Physical => sm.atk,
                    MoveCategory::Special => sm.spa,
                    MoveCategory::Status => None,
                };
                if let Some((num, den)) = ratio {
                    mods.push(AtkStatMod::Item { num, den });
                }
            }
        }
        if let Some(tb) = &it.type_boost {
            if tb.fold == TypeBoostFold::Stat && move_type == Some(tb.type_) {
                mods.push(AtkStatMod::Item { num: tb.num, den: tb.den });
            }
        }
    }
    // Ability DMG_MOD fold=atk (Huge/Pure Power ×2 uncond, Guts ×1.5 whenStatused) —
    // PHYSICAL only; `direct` (Hustle) stays unwired.
    if category == MoveCategory::Physical {
        if let Some(m) = dex.ability(ability).and_then(|a| a.dmg_mod.as_ref()) {
            if m.fold == DmgFold::Atk && !m.direct && (!m.when_statused || attacker_statused) {
                mods.push(AtkStatMod::Item { num: m.num, den: m.den });
            }
        }
    }
    mods
}

/// Resolve the DEFENDER's STAT-event modifiers (ModifyDef/ModifySpD) from the
/// golden's (def_item, def_species, category) — DeepSeaScale / Metal Powder /
/// Soul Dew SpD. Mirrors `turn.rs::resolve_def_stat_mods` off the same data.
fn resolve_def_mods(
    item: &str,
    ability: &str,
    species: &str,
    category: MoveCategory,
    defender_statused: bool,
    dex: &Dex,
) -> Vec<AtkStatMod> {
    let mut mods = Vec::new();
    if let Some(it) = dex.item(item) {
        if let Some(sm) = &it.stat_mods {
            if sm.only_species.is_empty() || sm.only_species.iter().any(|s| s == species) {
                let ratio = match category {
                    MoveCategory::Physical => sm.def,
                    MoveCategory::Special => sm.spd,
                    MoveCategory::Status => None,
                };
                if let Some((num, den)) = ratio {
                    mods.push(AtkStatMod::Item { num, den });
                }
            }
        }
    }
    // Ability DMG_MOD fold=def (Marvel Scale ×1.5 whenStatused) — physical Def only.
    if category == MoveCategory::Physical {
        if let Some(m) = dex.ability(ability).and_then(|a| a.dmg_mod.as_ref()) {
            if m.fold == DmgFold::Def && !m.direct && (!m.when_statused || defender_statused) {
                mods.push(AtkStatMod::Item { num: m.num, den: m.den });
            }
        }
    }
    mods
}

/// Resolve the attacker-item BASE-POWER-phase modifiers — the incense chain
/// (`fold == BasePower`, ×4915/4096) and the bows' DIRECT ×1.1 float replace
/// (`fold == BasePowerDirect`). Mirrors `turn.rs::resolve_bp_mods`.
fn resolve_bp_mods(
    item: &str,
    ability: &str,
    move_type: Option<Type>,
    attacker_in_pinch: bool,
    dex: &Dex,
) -> Vec<BpMod> {
    let mut mods = Vec::new();
    if let Some(it) = dex.item(item) {
        if let Some(tb) = &it.type_boost {
            if move_type == Some(tb.type_) {
                match tb.fold {
                    TypeBoostFold::Stat => {}
                    TypeBoostFold::BasePower => mods.push(BpMod::Chain(tb.num, tb.den)),
                    TypeBoostFold::BasePowerDirect => mods.push(BpMod::Direct(tb.num, tb.den)),
                }
            }
        }
    }
    // Ability DMG_MOD PINCH (Torrent/Blaze/Overgrow/Swarm) — BP ×1.5 for the ability's
    // type at hp<=maxhp/3 (Thick Fat's sourceBasePower is modeled via defender_thick_fat).
    if let (Some(mt), Some(m)) = (move_type, dex.ability(ability).and_then(|a| a.dmg_mod.as_ref())) {
        if m.fold == DmgFold::BasePower && m.pinch && attacker_in_pinch && m.types.contains(&mt) {
            mods.push(BpMod::Chain(m.num, m.den));
        }
    }
    mods
}

/// Resolve a DEFENDER ability immunity / Thick Fat from (ability, move-type). The
/// type-chart 0× immunities (Electric→Ground) are handled inside `calc_damage`;
/// THIS covers the ability ones (Levitate vs Ground, Flash Fire vs Fire, Water /
/// Volt Absorb) + Thick Fat (Ice/Fire ×0.5 on the attacker's stat).
fn resolve_defender(ability: &str, move_type: Option<Type>) -> (bool, bool) {
    let immune = match ability {
        "levitate" => move_type == Some(Type::Ground),
        "flashfire" => move_type == Some(Type::Fire),
        "waterabsorb" => move_type == Some(Type::Water),
        "voltabsorb" => move_type == Some(Type::Electric),
        _ => false,
    };
    let thick_fat = ability == "thickfat"
        && (move_type == Some(Type::Ice) || move_type == Some(Type::Fire));
    (immune, thick_fat)
}

#[test]
fn damage_golden_matches_showdown() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/damage_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing damage golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_damage_golden.js")
    });
    let d = dex();

    let mut checked = 0usize;
    for (i, line) in data.lines().enumerate() {
        let ln = i + 1;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f[0], "DMG", "unknown record {:?} (line {ln})", f[0]);
        // 25 pre-existing fields + the 3 `gen3_item_mechanics_v1` item columns
        // (atk_species/def_species/def_item) + the 4 ability-side columns
        // (atk_hp/atk_maxhp/def_status/def_ability) — pre-existing indices unchanged.
        assert_eq!(f.len(), 32, "DMG needs 32 fields (line {ln}), got {}", f.len());

        let id = f[1];
        let bp: u16 = f[2].parse().unwrap();
        let move_type = Type::from_name(f[3]); // None for "???"
        let category = parse_category(f[4]);
        let crit = f[5] == "1";
        let weather = match f[6] {
            "rain" => Some(Weather::Rain),
            "sun" => Some(Weather::Sun),
            "none" => None,
            other => panic!("unknown weather {other:?} (line {ln})"),
        };
        let reflect = f[7] == "1";
        let light_screen = f[8] == "1";
        let halves_def = f[9] == "1";
        let atk_level: u8 = f[10].parse().unwrap();
        let atk_stat: u16 = f[11].parse().unwrap();
        let spa_stat: u16 = f[12].parse().unwrap();
        let atk_types = parse_types(f[13]);
        let atk_boosts = parse_boosts(f[14]);
        let atk_status = f[15];
        let atk_ability = f[16];
        let atk_item = f[17];
        let def_stat: u16 = f[18].parse().unwrap();
        let spd_stat: u16 = f[19].parse().unwrap();
        let def_types = parse_types(f[20]);
        let def_boosts = parse_boosts(f[21]);
        let def_ability = f[22];
        let exp_base: u16 = f[23].parse().unwrap(); // base (max-roll) damage
        let exp_min: u16 = f[24].parse().unwrap(); // min-roll (r==15) damage
        let atk_species = f[25];
        let def_species = f[26];
        let def_item = f[27];
        // gen3_item_mechanics_v1 ability side:
        let atk_hp: u32 = f[28].parse().unwrap();
        let atk_maxhp: u32 = f[29].parse().unwrap();
        let def_status = f[30];
        let def_ability_id = f[31];

        // The ability DMG_MOD runtime conditions the folds gate on.
        let attacker_statused = atk_status != "none";
        let defender_statused = def_status != "none";
        // PINCH: bit-exactly the sim's integer-hp `hp <= maxhp/3` float compare.
        let attacker_in_pinch = 3 * atk_hp <= atk_maxhp;
        // Guts suppresses the physical burn-halve (its `dmgMod` fold=atk whenStatused, but
        // the burn skip is a separate `has_guts` flag in modify_damage).
        let atk_has_guts = pokesim::dex::to_id(atk_ability) == "guts";

        let (immune, thick_fat) = resolve_defender(def_ability, move_type);
        let atk_stat_mods =
            resolve_atk_mods(atk_item, atk_ability, atk_species, move_type, category, attacker_statused, &d);
        let def_stat_mods =
            resolve_def_mods(def_item, def_ability_id, def_species, category, defender_statused, &d);
        let bp_mods = resolve_bp_mods(atk_item, atk_ability, move_type, attacker_in_pinch, &d);

        let attacker = Combatant {
            level: atk_level,
            atk_stat,
            spa_stat,
            def_stat: 0,
            spd_stat: 0,
            types: atk_types,
            boosts: atk_boosts,
            burned: atk_status == "brn",
            has_guts: atk_has_guts,
        };
        let defender = Combatant {
            level: 100,
            atk_stat: 0,
            spa_stat: 0,
            def_stat,
            spd_stat,
            types: def_types,
            boosts: def_boosts,
            burned: def_status == "brn",
            has_guts: pokesim::dex::to_id(def_ability_id) == "guts",
        };
        let mv = MoveInput { minimize_doubles: false, base_power: bp, move_type, category, halves_defense: halves_def };

        let ctx = DamageContext {
            defender_minimized: false,
            attacker,
            defender,
            mv,
            crit,
            weather,
            reflect,
            light_screen,
            atk_stat_mods,
            // Hustle Atk-direct is not exercised by these probes (Hustle is the accuracy
            // phase's own golden); the DMG_MOD max-roll probes use only chain members.
            atk_direct_modify: None,
            def_stat_mods,
            bp_mods,
            defender_thick_fat: thick_fat,
            immune,
            // These damage-golden probes never stage a Flash-Fire-ARMED attacker (the golden's
            // FF scenario is an IMMUNITY, base 0); the FF ×1.5 BOOST is proven by the dedicated
            // flashfire golden + the calc-level max-roll pins in flashfire_test.rs.
            flash_fire: false,
        };

        let got = calc_damage(&ctx, &d);
        assert_eq!(
            got.base, exp_base,
            "[{id}] base (max-roll) mismatch (line {ln}): got {} exp {}\n  bp={bp} type={:?} cat={:?} crit={crit} weather={:?} reflect={reflect} ls={light_screen} halves_def={halves_def}\n  atk={atk_stat}/{spa_stat} boosts={:?} item={atk_item} status={atk_status}\n  def={def_stat}/{spd_stat} boosts={:?} ability={def_ability}",
            got.base, exp_base, move_type, category, weather, ctx.attacker.boosts, ctx.defender.boosts,
        );
        // The min roll (r==15, 85%) must match too — exercises the randomizer's
        // floor. (Immunities have base==0, so rolls[15]==0 == exp_min trivially.)
        assert_eq!(
            got.rolls[15], exp_min,
            "[{id}] min-roll (rolls[15]) mismatch (line {ln}): got {} exp {}",
            got.rolls[15], exp_min,
        );
        checked += 1;
    }
    // 31 pre-existing + the 17 gen3_item_mechanics_v1 item probes + the 15 ability
    // DMG_MOD probes (pinch family / Huge-Pure / Guts+burn / Marvel Scale + controls).
    assert!(checked >= 63, "expected the full damage golden corpus, got {checked}");
    eprintln!("damage golden: {checked} scenarios passed (EXACT max-roll match)");
}

#[test]
fn damage_smoke() {
    let d = dex();

    // A hand-built neutral physical hit: level 100, atk 300, def 200, BP 100, no
    // mods. base = tr(tr(tr(tr(2*100/5+2)*100*300)/200)/50) = tr(tr(tr(42*30000)/200)/50)
    //   = tr(tr(1260000/200)/50) = tr(6300/50) = 126; +2 = 128 (no crit/STAB/type).
    let base_ctx = |category, move_type, crit, atk, def, boosts_def: [i8; 5]| DamageContext {
        defender_minimized: false,
        attacker: Combatant { level: 100, atk_stat: atk, spa_stat: atk, types: vec![Type::Normal], ..Default::default() },
        defender: Combatant { level: 100, def_stat: def, spd_stat: def, types: vec![Type::Normal], boosts: boosts_def, ..Default::default() },
        mv: MoveInput { minimize_doubles: false, base_power: 100, move_type, category, halves_defense: false },
        crit,
        weather: None,
        reflect: false,
        light_screen: false,
        atk_stat_mods: vec![],
        atk_direct_modify: None,
        def_stat_mods: vec![],
        bp_mods: vec![],
        defender_thick_fat: false,
        immune: false,
        flash_fire: false,
    };
    // Neutral, no STAB (Normal move, Normal attacker has STAB! use a typeless move).
    let neutral = calc_damage(&base_ctx(MoveCategory::Physical, None, false, 300, 200, [0; 5]), &d);
    assert_eq!(neutral.base, 128, "neutral no-STAB base = floor + 2");

    // STAB ×1.5: Normal move off a Normal attacker → modify(128, 3/2) = 192.
    let stab = calc_damage(&base_ctx(MoveCategory::Physical, Some(Type::Normal), false, 300, 200, [0; 5]), &d);
    assert_eq!(stab.base, 192, "STAB applies modify(bd,1.5) after +2");

    // CRIT ×2 on a no-STAB typeless hit: modify(128, 2) — but the +2 happens BEFORE
    // crit, so it is modify(baseDamage+2, 2). baseDamage(pre-+2)=126, +2=128,
    // ×2 = 256. (No STAB/type.)
    let crit = calc_damage(&base_ctx(MoveCategory::Physical, None, true, 300, 200, [0; 5]), &d);
    assert_eq!(crit.base, 256, "crit ×2 on (baseDamage+2)");

    // CRIT IGNORES the defender's +2 Def: with def +2 boost the un-crit damage drops,
    // but the crit zeroes the positive Def boost, so crit == the +0-boost crit (256).
    let crit_ignores = calc_damage(&base_ctx(MoveCategory::Physical, None, true, 300, 200, [0, 2, 0, 0, 0]), &d);
    assert_eq!(crit_ignores.base, 256, "crit ignores defender's positive Def boost");
    // ...and the NON-crit version with +2 Def IS reduced (def ×2 → ~half damage).
    let noncrit_boosted = calc_damage(&base_ctx(MoveCategory::Physical, None, false, 300, 200, [0, 2, 0, 0, 0]), &d);
    assert!(noncrit_boosted.base < neutral.base, "+2 Def reduces a non-crit hit");

    // TYPE 0× (immunity) → 0 damage. A Ground move vs a Flying defender (chart 0×).
    let immune_ctx = DamageContext {
        defender_minimized: false,
        attacker: Combatant { level: 100, atk_stat: 300, types: vec![Type::Ground], ..Default::default() },
        defender: Combatant { level: 100, def_stat: 200, types: vec![Type::Flying], ..Default::default() },
        mv: MoveInput { minimize_doubles: false, base_power: 100, move_type: Some(Type::Ground), category: MoveCategory::Physical, halves_defense: false },
        crit: false, weather: None, reflect: false, light_screen: false,
        atk_stat_mods: vec![], atk_direct_modify: None, def_stat_mods: vec![], bp_mods: vec![],
        defender_thick_fat: false, immune: false, flash_fire: false,
    };
    assert_eq!(calc_damage(&immune_ctx, &d).base, 0, "Ground vs Flying = 0× immunity");

    // Ability immunity flag → 0 even if the chart is neutral.
    let mut ability_immune = immune_ctx.clone();
    ability_immune.defender.types = vec![Type::Normal]; // chart-neutral now
    ability_immune.immune = true;
    assert_eq!(calc_damage(&ability_immune, &d).base, 0, "ability immunity → 0");

    // The 16-roll spread: max == base, min (rolls[15]) == 85% floored, monotone.
    let spread = calc_damage(&base_ctx(MoveCategory::Physical, Some(Type::Normal), false, 300, 200, [0; 5]), &d);
    assert_eq!(spread.rolls[0], spread.base);
    assert_eq!(spread.rolls[15], (spread.base as u32 * 85 / 100) as u16);
    for w in spread.rolls.windows(2) {
        assert!(w[0] >= w[1], "rolls are non-increasing");
    }
}
