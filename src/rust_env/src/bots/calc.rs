//! The helpers the bots share — `agents.gen3_mechanics`, `SimpleHeuristicsPlayer`'s static
//! methods and `agents/opponents.py`'s module functions — one function per Python function, named
//! at its site (M5 Lane F).
//!
//! BIT-EXACTNESS: every float expression keeps Python's operand order (`a * b * c` is `(a * b) * c`),
//! ints are widened where Python widens them, and `max(xs, key=…)` keeps the FIRST maximal element
//! (Python's `max` replaces only on a strictly greater key). Rust never contracts to FMA on its own,
//! so the same order is the same bits.

use pokesim::present::dex::Status;

use super::tables::{self as t};
use super::view::{MonV, MoveV, View};
use super::BotError;

type R<T> = Result<T, BotError>;

/// Python's `max(xs, key=f)` → the index of the FIRST maximal element (`None` for an empty list).
pub fn argmax<T>(xs: &[T], mut key: impl FnMut(&T) -> R<f64>) -> R<Option<usize>> {
    let mut best: Option<(usize, f64)> = None;
    for (i, x) in xs.iter().enumerate() {
        let k = key(x)?;
        match best {
            Some((_, b)) if !(k > b) => {}
            _ => best = Some((i, k)),
        }
    }
    Ok(best.map(|(i, _)| i))
}

fn in_set(set: &[&str], x: &str) -> bool {
    set.contains(&x)
}

fn chart(table: &[(&str, &str, f64)], defending: &str, attacking: &str) -> Option<f64> {
    table
        .binary_search_by(|(d, a, _)| (*d, *a).cmp(&(defending, attacking)))
        .ok()
        .map(|i| table[i].2)
}

/// `PokemonType.damage_multiplier(type_1, type_2, type_chart=GenData(3).type_chart)` — i.e.
/// `Pokemon.damage_multiplier(type)` of the DEFENDING `mon`. A missing chart key is Python's KeyError.
pub fn damage_multiplier(attacking: &str, mon: &MonV) -> R<f64> {
    if in_set(t::NULL_TYPES, attacking) || in_set(t::NULL_TYPES, mon.t1) {
        return Ok(1.0);
    }
    let get = |d: &str| chart(t::POKEENV_CHART, d, attacking).ok_or_else(|| BotError(format!("type_chart[{d}][{attacking}] (KeyError)")));
    let m = get(mon.t1)?;
    match mon.t2 {
        Some(t2) => Ok(m * get(t2)?),
        None => Ok(m),
    }
}

/// `gen3_mechanics.effective_multiplier(move_type, mon)` (→ `_eff_cached`).
pub fn effective_multiplier(move_type: &str, mon: &MonV) -> f64 {
    let ability = mon.ability.as_deref().unwrap_or("").to_lowercase();
    let frozen = mon.status == Some(Status::Frz);
    let base = if !in_set(t::MECH_REAL_TYPES, move_type) || in_set(t::NULL_TYPES, mon.t1) {
        1.0
    } else {
        let mut b = chart(t::MECH_CHART, mon.t1, move_type).unwrap_or(1.0);
        if let Some(t2) = mon.t2 {
            b *= chart(t::MECH_CHART, t2, move_type).unwrap_or(1.0);
        }
        b
    };
    if ability == "wonderguard" {
        return if base > 1.0 { base } else { 0.0 };
    }
    if ability == "flashfire" && frozen {
        return base;
    }
    let m = t::ABILITY_TYPE_MULTIPLIER
        .iter()
        .find(|(a, ty, _)| *a == ability && *ty == move_type)
        .map_or(1.0, |(_, _, v)| *v);
    base * m
}

/// `gen3_mechanics.is_status_move_immune(move_id, mon)`.
pub fn is_status_move_immune(move_id: &str, mon: &MonV) -> bool {
    let immune = t::STATUS_MOVE_IMMUNITY
        .iter()
        .any(|(m, ty)| *m == move_id && (*ty == mon.t1 || Some(*ty) == mon.t2));
    immune || mon.status.is_some()
}

/// `SimpleHeuristicsPlayer._estimate_matchup(mon, opponent)`.
pub fn estimate_matchup(mon: &MonV, opponent: &MonV) -> R<f64> {
    let mut score = f64::NEG_INFINITY;
    for ty in &mon.types {
        let v = damage_multiplier(ty, opponent)?;
        if v > score {
            score = v;
        }
    }
    let mut worst = f64::NEG_INFINITY;
    for ty in &opponent.types {
        let v = damage_multiplier(ty, mon)?;
        if v > worst {
            worst = v;
        }
    }
    if mon.types.is_empty() || opponent.types.is_empty() {
        return Err(BotError("max() arg is an empty sequence (ValueError)".into()));
    }
    score -= worst;
    let (ms, os) = (mon.base_stat("spe"), opponent.base_stat("spe"));
    if ms > os {
        score += t::SPEED_TIER_COEFICIENT;
    } else if os > ms {
        score -= t::SPEED_TIER_COEFICIENT;
    }
    score += mon.hp * t::HP_FRACTION_COEFICIENT;
    score -= opponent.hp * t::HP_FRACTION_COEFICIENT;
    Ok(score)
}

/// `SimpleHeuristicsPlayer._stat_estimation(mon, stat)`.
pub fn stat_estimation(mon: &MonV, stat: &str) -> f64 {
    let b = mon.boost(stat);
    let boost = if b > 1 { (2 + b) as f64 / 2.0 } else { 2.0 / (2 - b) as f64 };
    ((2 * mon.base_stat(stat) as i64 + 31 + 5) as f64) * boost
}

/// `SimpleHeuristicsPlayer._should_switch_out(battle)`.
pub fn should_switch_out(v: &View) -> R<bool> {
    let (Some(active), Some(opponent)) = (v.active_mon(), v.opp_active.as_ref()) else {
        return Err(BotError("_should_switch_out on a battle without both actives (AttributeError)".into()));
    };
    let mut decent = false;
    for &s in &v.switches {
        if estimate_matchup(&v.team[s], opponent)? > 0.0 {
            decent = true;
        }
    }
    if decent {
        if active.boost("def") <= -3 || active.boost("spd") <= -3 {
            return Ok(true);
        }
        if active.boost("atk") <= -3 && active.stat("atk")? >= active.stat("spa")? {
            return Ok(true);
        }
        if active.boost("spa") <= -3 && active.stat("atk")? <= active.stat("spa")? {
            return Ok(true);
        }
        if estimate_matchup(active, opponent)? < t::SWITCH_OUT_MATCHUP_THRESHOLD {
            return Ok(true);
        }
    }
    Ok(false)
}

fn stab(m: &MoveV, attacker: &MonV) -> f64 {
    if attacker.types.contains(&m.typ) {
        1.5
    } else {
        1.0
    }
}

/// `opponents._effective_damage_score(move, active, opponent)`.
pub fn effective_damage_score(m: &MoveV, active: &MonV, opponent: &MonV) -> f64 {
    if m.bp == 0 {
        return 0.0;
    }
    m.bp as f64 * stab(m, active) * effective_multiplier(m.typ, opponent) * m.acc
}

/// `opponents._best_damage_move(battle)` → an index into `v.moves`.
pub fn best_damage_move(v: &View) -> R<Option<usize>> {
    let (Some(active), Some(opp)) = (v.active_mon(), v.opp_active.as_ref()) else { return Ok(None) };
    argmax(&v.moves, |m| Ok(effective_damage_score(m, active, opp)))
}

/// `opponents._best_switch(battle)` → an index into `v.team`.
pub fn best_switch(v: &View) -> R<Option<usize>> {
    let Some(opp) = v.opp_active.as_ref() else { return Ok(None) };
    Ok(argmax(&v.switches, |&s| estimate_matchup(&v.team[s], opp))?.map(|i| v.switches[i]))
}

/// `opponents._damage_ratios(active, opponent)`.
pub fn damage_ratios(active: &MonV, opponent: &MonV) -> (f64, f64) {
    (
        stat_estimation(active, "atk") / stat_estimation(opponent, "def"),
        stat_estimation(active, "spa") / stat_estimation(opponent, "spd"),
    )
}

/// `opponents._damage_score_v2(move, active, opponent, pr, sr)`.
pub fn damage_score_v2(m: &MoveV, active: &MonV, opponent: &MonV, pr: f64, sr: f64) -> f64 {
    if m.bp == 0 {
        return 0.0;
    }
    let ratio = if m.cat == "PHYSICAL" { pr } else { sr };
    m.bp as f64 * stab(m, active) * ratio * m.acc * m.hits * effective_multiplier(m.typ, opponent)
}

/// `opponents._best_damage_move_v2(battle)`.
pub fn best_damage_move_v2(v: &View) -> R<Option<usize>> {
    let (Some(active), Some(opp)) = (v.active_mon(), v.opp_active.as_ref()) else { return Ok(None) };
    if v.moves.is_empty() {
        return Ok(None);
    }
    let (pr, sr) = damage_ratios(active, opp);
    argmax(&v.moves, |m| Ok(damage_score_v2(m, active, opp, pr, sr)))
}

/// `opponents._used_protect_last_turn(active)`.
pub fn used_protect_last_turn(active: &MonV) -> bool {
    active.last_move().is_some_and(|m| in_set(t::PROTECT_MOVES, &m.id))
}

/// `opponents._estimate_max_hp(mon)`.
fn estimate_max_hp(mon: &MonV) -> f64 {
    2.0 * mon.base_stat("hp") + 160.0
}

/// `opponents._estimate_damage_fraction(move, attacker, defender)`.
pub fn estimate_damage_fraction(m: &MoveV, attacker: &MonV, defender: &MonV) -> f64 {
    if m.bp == 0 {
        return 0.0;
    }
    let eff = effective_multiplier(m.typ, defender);
    if eff == 0.0 {
        return 0.0;
    }
    let (atk, dfn) = if m.cat == "PHYSICAL" {
        (stat_estimation(attacker, "atk"), stat_estimation(defender, "def"))
    } else {
        (stat_estimation(attacker, "spa"), stat_estimation(defender, "spd"))
    };
    let base = (42.0 * m.bp as f64 * atk / dfn) / 50.0 + 2.0;
    let dmg = base * stab(m, attacker) * eff * m.hits * t::AVG_DAMAGE_ROLL;
    // `max(defender.current_hp_fraction, 1e-6)`: the first argument wins a tie
    let frac = if 1e-6 > defender.hp { 1e-6 } else { defender.hp };
    let cur_hp = estimate_max_hp(defender) * frac;
    if cur_hp > 0.0 {
        dmg / cur_hp
    } else {
        0.0
    }
}

/// `opponents._best_move_and_ko_fraction(battle)`.
pub fn best_move_and_ko_fraction(v: &View) -> R<(Option<usize>, f64)> {
    let best = best_damage_move_v2(v)?;
    match best {
        Some(i) if v.moves[i].bp != 0 => {
            let (a, o) = (v.active_mon().expect("v2 had one"), v.opp_active.as_ref().expect("v2 had one"));
            Ok((Some(i), estimate_damage_fraction(&v.moves[i], a, o)))
        }
        b => Ok((b, 0.0)),
    }
}

fn max_or(xs: impl Iterator<Item = f64>, default: f64) -> f64 {
    let mut best: Option<f64> = None;
    for x in xs {
        if best.is_none_or(|b| x > b) {
            best = Some(x);
        }
    }
    best.unwrap_or(default)
}

/// `opponents._opp_known_max_damage_fraction(battle)`.
pub fn opp_known_max_damage_fraction(v: &View) -> f64 {
    let (Some(opp), Some(active)) = (v.opp_active.as_ref(), v.active_mon()) else { return 0.0 };
    max_or(opp.moves.iter().filter(|m| m.bp > 0).map(|m| estimate_damage_fraction(m, opp, active)), 0.0)
}

/// `opponents._best_switch_v2(battle)` → an index into `v.team`.
pub fn best_switch_v2(v: &View) -> R<Option<usize>> {
    let Some(opp) = v.opp_active.as_ref() else { return Ok(None) };
    if v.switches.is_empty() {
        return Ok(None);
    }
    let score = |&s: &usize| -> R<f64> {
        let mon = &v.team[s];
        let incoming = max_or(opp.moves.iter().filter(|m| m.bp > 0).map(|m| estimate_damage_fraction(m, opp, mon)), 0.0);
        let outgoing = max_or(mon.moves.iter().filter(|m| m.bp > 0).map(|m| estimate_damage_fraction(m, mon, opp)), 0.0);
        Ok(estimate_matchup(mon, opp)? + outgoing - 1.5 * incoming)
    };
    Ok(argmax(&v.switches, score)?.map(|i| v.switches[i]))
}

/// `move.target == "self"` — ALWAYS False: poke-env's `Move.target` is a `Target` ENUM and the bots
/// compare it to a STRING. So the setup branch of `SimpleHeuristicsPlayer`, `Gen3HeuristicV2Player`,
/// `Gen3SetupSweepPlayer` and `Gen3SetupSweepV2Player` never fires (finding F-LF-1). Ported as the
/// Python bot behaves; `bot_tables_test.py` pins the Python fact, so a Python fix fails there first.
pub fn target_is_self_str(_m: &MoveV) -> bool {
    false
}
