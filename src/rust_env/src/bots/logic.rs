//! The bots' `choose_move`, one function per Python class, each branch in the Python order (M5 Lane F).
//!
//! Sources: `poke_env/player/baselines.py` (`RandomPlayer`, `SimpleHeuristicsPlayer`),
//! `agents/opponents.py` (the eight `Gen3*Player`s), `agents/baitbot.py` (`Gen3BaitBotPlayer`).
//! A draw is taken exactly where Python takes it — `and` short-circuits included — because the
//! gate checks the stream offset after every decision.

use pokesim::present::dex::Status;

use super::calc::*;
use super::rng::PyRandom;
use super::tables as t;
use super::view::{Order, View};
use super::BotError;

type R<T> = Result<T, BotError>;

/// Record WHICH return site decided (its source line — the gate's branch coverage) and pass the
/// order through.
fn hit(br: &mut u32, line: u32, o: Order) -> Order {
    *br = line;
    o
}

fn has(set: &[&str], x: &str) -> bool {
    set.contains(&x)
}

/// `Player.choose_random_move(battle)` with the player's `_choice_rng` (`_random_singles_order`).
pub fn random_move(v: &View, choice: &mut PyRandom) -> Order {
    let orders = v.valid_orders();
    if orders.is_empty() {
        return Order::Default;
    }
    let i = (choice.random() * orders.len() as f64) as usize;
    orders[i].clone()
}

fn first_move(v: &View, pred: impl Fn(&super::view::MoveV) -> bool) -> Option<Order> {
    v.moves.iter().position(pred).map(Order::Move)
}

/// `SimpleHeuristicsPlayer.choose_singles_move(battle)[0]`.
pub fn heuristic(v: &View, choice: &mut PyRandom, br: &mut u32) -> R<Order> {
    let (Some(active), Some(opp)) = (v.active_mon(), v.opp_active.as_ref()) else {
        return Ok(hit(br, line!(), random_move(v, choice)));
    };
    let (pr, sr) = damage_ratios(active, opp);
    if !v.moves.is_empty() && (!should_switch_out(v)? || v.switches.is_empty()) {
        let n_remaining = v.team.iter().filter(|m| !m.fainted).count();
        let n_opp_remaining = 6 - v.opp_fainted as i64;
        for (i, m) in v.moves.iter().enumerate() {
            let hazard = t::ENTRY_HAZARDS.iter().find(|(id, _)| *id == m.id);
            if n_opp_remaining >= 3 && hazard.is_some_and(|(_, sc)| !v.opp_side.contains(sc)) {
                return Ok(hit(br, line!(), Order::Move(i)));
            } else if !v.side.is_empty() && has(t::ANTI_HAZARDS_MOVES, &m.id) && n_remaining >= 2 {
                return Ok(hit(br, line!(), Order::Move(i)));
            }
        }
        // Setup: `active.current_hp_fraction == 1 and _estimate_matchup(...) > 0`, then the first move
        // whose `self_setup_boosts` (Target.SELF boosts, or a non-Ghost's Curse) raise >= 2 stages
        // with an uncapped raised stat.
        if active.hp == 1.0 && estimate_matchup(active, opp)? > 0.0 {
            for (i, m) in v.moves.iter().enumerate() {
                if let Some(b) = self_setup_boosts(m, active).filter(|b| !b.is_empty()) {
                    if raised_stages(b) >= 2
                        && b.iter().filter(|(_, x)| *x > 0).map(|(s, _)| active.boost(s)).min().expect("raised >= 2") < 6
                    {
                        // F-LF-1 SETUP SITE — bots_gate_test.py asserts the COMMIT bank reaches it
                        return Ok(hit(br, line!(), Order::Move(i)));
                    }
                }
            }
        }
        let best = argmax(&v.moves, |m| {
            let stab = if active.types.contains(&m.typ) { 1.5 } else { 1.0 };
            let ratio = if m.cat == "PHYSICAL" { pr } else { sr };
            Ok(m.bp as f64 * stab * ratio * m.acc * m.hits * damage_multiplier(m.typ, opp)?)
        })?;
        return Ok(hit(br, line!(), Order::Move(best.expect("non-empty"))));
    }
    if !v.switches.is_empty() {
        let i = argmax(&v.switches, |&s| estimate_matchup(&v.team[s], opp))?.expect("non-empty");
        return Ok(hit(br, line!(), Order::Switch(v.switches[i])));
    }
    Ok(hit(br, line!(), random_move(v, choice)))
}

/// `Gen3StallerPlayer.choose_move`.
pub fn staller(v: &View, choice: &mut PyRandom, protect: &mut PyRandom, br: &mut u32) -> R<Order> {
    let (Some(active), Some(opp)) = (v.active_mon(), v.opp_active.as_ref()) else {
        return Ok(hit(br, line!(), random_move(v, choice)));
    };
    if !v.moves.is_empty() {
        if opp.status.is_none() {
            if let Some(o) = first_move(v, |m| has(t::STATUS_MOVES, &m.id) && m.bp == 0) {
                return Ok(hit(br, line!(), o));
            }
        }
        if opp.status == Some(Status::Tox) && protect.random() < t::PROTECT_PROBABILITY {
            if let Some(o) = first_move(v, |m| has(t::PROTECT_MOVES, &m.id)) {
                return Ok(hit(br, line!(), o));
            }
        }
        if active.hp < t::RECOVERY_HP_THRESHOLD {
            if let Some(o) = first_move(v, |m| has(t::RECOVERY_MOVES, &m.id)) {
                return Ok(hit(br, line!(), o));
            }
        }
        if !v.side.is_empty() {
            if let Some(o) = first_move(v, |m| has(t::HAZARD_CLEAR_MOVES, &m.id)) {
                return Ok(hit(br, line!(), o));
            }
        }
        if let Some(i) = best_damage_move(v)? {
            return Ok(hit(br, line!(), Order::Move(i)));
        }
    }
    if let Some(s) = best_switch(v)? {
        return Ok(hit(br, line!(), Order::Switch(s)));
    }
    Ok(hit(br, line!(), random_move(v, choice)))
}

/// `Gen3AggressivePlayer.choose_move`.
pub fn aggressive(v: &View, choice: &mut PyRandom, br: &mut u32) -> R<Order> {
    let (Some(active), Some(opp)) = (v.active_mon(), v.opp_active.as_ref()) else {
        return Ok(hit(br, line!(), random_move(v, choice)));
    };
    if !v.moves.is_empty() {
        let damaging: Vec<usize> = (0..v.moves.len()).filter(|&i| v.moves[i].bp > 0).collect();
        if !damaging.is_empty() {
            let i = argmax(&damaging, |&i| Ok(effective_damage_score(&v.moves[i], active, opp)))?.expect("non-empty");
            return Ok(hit(br, line!(), Order::Move(damaging[i])));
        }
        let i = argmax(&v.moves, |m| Ok(m.bp as f64))?.expect("non-empty");
        return Ok(hit(br, line!(), Order::Move(i)));
    }
    if !v.switches.is_empty() {
        let i = argmax(&v.switches, |&s| {
            let b = &v.team[s];
            let (a, sp) = (b.base_stat("atk"), b.base_stat("spa"));
            Ok(if sp > a { sp } else { a })
        })?
        .expect("non-empty");
        return Ok(hit(br, line!(), Order::Switch(v.switches[i])));
    }
    Ok(hit(br, line!(), random_move(v, choice)))
}

/// The setup step of both setup sweepers: the first move with `move.id in _SETUP_MOVES and
/// move.target is Target.SELF` that raises a `_SETUP_STATS` stat not yet at +6.
fn setup_move(v: &View, active: &super::view::MonV) -> Option<usize> {
    v.moves.iter().position(|m| {
        has(t::SETUP_MOVES, &m.id)
            && self_setup_boosts(m, active)
                .is_some_and(|b| b.iter().any(|(s, x)| *x > 0 && has(t::SETUP_STATS, s) && active.boost(s) < 6))
    })
}

/// `sum(v for v in boosts.values() if v > 0)` — the RAISED stages (Curse's −1 Spe does not count).
fn raised_stages(b: &[(&str, i32)]) -> i32 {
    b.iter().map(|(_, x)| *x).filter(|x| *x > 0).sum()
}

fn offensive_boosts(v: &View) -> i32 {
    let a = v.active_mon().expect("checked");
    t::SETUP_STATS.iter().map(|s| a.boost(s)).sum()
}

/// `Gen3SetupSweepPlayer.choose_move`.
pub fn setup_sweep(v: &View, choice: &mut PyRandom, br: &mut u32) -> R<Order> {
    let (Some(active), Some(opp)) = (v.active_mon(), v.opp_active.as_ref()) else {
        return Ok(hit(br, line!(), random_move(v, choice)));
    };
    let matchup = estimate_matchup(active, opp)?;
    let total = offensive_boosts(v);
    let should_switch = matchup < t::SWITCH_OUT_MATCHUP_THRESHOLD && !v.switches.is_empty();
    if !v.moves.is_empty() && !should_switch {
        if active.hp >= t::SETUP_HP_THRESHOLD && matchup > 0.0 && total < t::SETUP_BOOST_CAP {
            if let Some(i) = setup_move(v, active) {
                // F-LF-1 SETUP SITE — bots_gate_test.py asserts the COMMIT bank reaches it
                return Ok(hit(br, line!(), Order::Move(i)));
            }
        }
        if let Some(i) = best_damage_move(v)? {
            return Ok(hit(br, line!(), Order::Move(i)));
        }
    }
    if !v.switches.is_empty() {
        if let Some(s) = best_switch(v)? {
            return Ok(hit(br, line!(), Order::Switch(s)));
        }
    }
    if !v.moves.is_empty() {
        return Ok(hit(br, line!(), Order::Move(0)));
    }
    Ok(hit(br, line!(), random_move(v, choice)))
}

/// `Gen3StallerV2Player.choose_move`.
pub fn staller_v2(v: &View, choice: &mut PyRandom, protect: &mut PyRandom, br: &mut u32) -> R<Order> {
    let (Some(active), Some(opp)) = (v.active_mon(), v.opp_active.as_ref()) else {
        return Ok(hit(br, line!(), random_move(v, choice)));
    };
    if !v.switches.is_empty() && opp_known_max_damage_fraction(v) >= 0.85 {
        if let Some(s) = best_switch_v2(v)? {
            return Ok(hit(br, line!(), Order::Switch(s)));
        }
    }
    if !v.moves.is_empty() {
        if active.hp < t::RECOVERY_HP_THRESHOLD {
            if let Some(o) = first_move(v, |m| has(t::RECOVERY_MOVES, &m.id)) {
                return Ok(hit(br, line!(), o));
            }
        }
        if opp.status.is_none() {
            if let Some(o) = first_move(v, |m| has(t::STATUS_MOVES, &m.id) && m.bp == 0 && !is_status_move_immune(&m.id, opp)) {
                return Ok(hit(br, line!(), o));
            }
        }
        if opp.status == Some(Status::Tox) && protect.random() < t::PROTECT_PROBABILITY && !used_protect_last_turn(active) {
            if let Some(o) = first_move(v, |m| has(t::PROTECT_MOVES, &m.id)) {
                return Ok(hit(br, line!(), o));
            }
        }
        if !v.side.is_empty() {
            if let Some(o) = first_move(v, |m| has(t::HAZARD_CLEAR_MOVES, &m.id)) {
                return Ok(hit(br, line!(), o));
            }
        }
        if let Some(i) = best_damage_move_v2(v)? {
            return Ok(hit(br, line!(), Order::Move(i)));
        }
    }
    if let Some(s) = best_switch_v2(v)? {
        return Ok(hit(br, line!(), Order::Switch(s)));
    }
    Ok(hit(br, line!(), random_move(v, choice)))
}

/// `Gen3AggressiveV2Player.choose_move`.
pub fn aggressive_v2(v: &View, choice: &mut PyRandom, br: &mut u32) -> R<Order> {
    let (Some(active), Some(opp)) = (v.active_mon(), v.opp_active.as_ref()) else {
        return Ok(hit(br, line!(), random_move(v, choice)));
    };
    if !v.moves.is_empty() {
        let damaging: Vec<usize> = (0..v.moves.len()).filter(|&i| v.moves[i].bp > 0).collect();
        if !damaging.is_empty() {
            let ko: Vec<usize> =
                damaging.iter().copied().filter(|&i| estimate_damage_fraction(&v.moves[i], active, opp) >= t::KO_FRACTION).collect();
            let best = if !ko.is_empty() {
                ko[argmax(&ko, |&i| Ok(v.moves[i].acc * v.moves[i].hits))?.expect("non-empty")]
            } else {
                best_damage_move_v2(v)?.expect("moves, active and opponent are present")
            };
            if !v.switches.is_empty() {
                let immune = effective_multiplier(v.moves[best].typ, opp) == 0.0;
                let matchup = estimate_matchup(active, opp)?;
                let mut badly = false;
                if matchup < t::SWITCH_OUT_MATCHUP_THRESHOLD {
                    for &s in &v.switches {
                        if estimate_matchup(&v.team[s], opp)? > matchup {
                            badly = true;
                            break;
                        }
                    }
                }
                if immune || badly {
                    if let Some(s) = best_switch_v2(v)? {
                        return Ok(hit(br, line!(), Order::Switch(s)));
                    }
                }
            }
            return Ok(hit(br, line!(), Order::Move(best)));
        }
        let i = argmax(&v.moves, |m| Ok(m.bp as f64))?.expect("non-empty");
        return Ok(hit(br, line!(), Order::Move(i)));
    }
    if !v.switches.is_empty() {
        if let Some(s) = best_switch_v2(v)? {
            return Ok(hit(br, line!(), Order::Switch(s)));
        }
    }
    Ok(hit(br, line!(), random_move(v, choice)))
}

/// `Gen3SetupSweepV2Player.choose_move`.
pub fn setup_sweep_v2(v: &View, choice: &mut PyRandom, br: &mut u32) -> R<Order> {
    let (Some(active), Some(opp)) = (v.active_mon(), v.opp_active.as_ref()) else {
        return Ok(hit(br, line!(), random_move(v, choice)));
    };
    let matchup = estimate_matchup(active, opp)?;
    let total = offensive_boosts(v);
    let (best, best_frac) = best_move_and_ko_fraction(v)?;
    if !v.moves.is_empty() {
        if let Some(b) = best.filter(|_| best_frac >= t::KO_FRACTION) {
            return Ok(hit(br, line!(), Order::Move(b)));
        }
    }
    let safe_to_setup = opp_known_max_damage_fraction(v) < 0.55;
    let should_switch = matchup < t::SWITCH_OUT_MATCHUP_THRESHOLD && !v.switches.is_empty() && total == 0;
    if !v.moves.is_empty() && !should_switch {
        if active.hp >= t::SETUP_HP_THRESHOLD && matchup > 0.0 && safe_to_setup && total < t::SETUP_BOOST_CAP {
            if let Some(i) = setup_move(v, active) {
                // F-LF-1 SETUP SITE — bots_gate_test.py asserts the COMMIT bank reaches it
                return Ok(hit(br, line!(), Order::Move(i)));
            }
        }
        if let Some(b) = best {
            return Ok(hit(br, line!(), Order::Move(b)));
        }
    }
    if !v.switches.is_empty() {
        if let Some(s) = best_switch_v2(v)? {
            return Ok(hit(br, line!(), Order::Switch(s)));
        }
    }
    if !v.moves.is_empty() {
        return Ok(hit(br, line!(), Order::Move(0)));
    }
    Ok(hit(br, line!(), random_move(v, choice)))
}

/// `Gen3HeuristicV2Player.choose_move`.
pub fn heuristic_v2(v: &View, choice: &mut PyRandom, br: &mut u32) -> R<Order> {
    let (Some(active), Some(opp)) = (v.active_mon(), v.opp_active.as_ref()) else {
        return Ok(hit(br, line!(), random_move(v, choice)));
    };
    let (best, best_frac) = if !v.moves.is_empty() { best_move_and_ko_fraction(v)? } else { (None, 0.0) };
    let we_faster = active.base_stat("spe") > opp.base_stat("spe");
    let matchup = estimate_matchup(active, opp)?;
    if let Some(b) = best {
        if best_frac >= t::KO_FRACTION && (we_faster || opp.hp <= t::KO_FINISH_THRESHOLD) {
            return Ok(hit(br, line!(), Order::Move(b)));
        }
    }
    if !v.switches.is_empty() && !we_faster && best_frac < t::KO_FRACTION && opp_known_max_damage_fraction(v) >= 0.85 {
        if let Some(s) = best_switch_v2(v)? {
            return Ok(hit(br, line!(), Order::Switch(s)));
        }
    }
    if !v.switches.is_empty() && should_switch_out(v)? {
        if let Some(s) = best_switch_v2(v)? {
            return Ok(hit(br, line!(), Order::Switch(s)));
        }
    }
    if !v.moves.is_empty() {
        let n_remaining = v.team.iter().filter(|m| !m.fainted).count();
        let n_opp_remaining = 6 - v.opp_fainted as i64;
        if active.hp < t::RECOVERY_HP_THRESHOLD && (we_faster || matchup > 0.0) {
            if let Some(o) = first_move(v, |m| has(t::RECOVERY_MOVES, &m.id)) {
                return Ok(hit(br, line!(), o));
            }
        }
        for (i, m) in v.moves.iter().enumerate() {
            let hazard = t::ENTRY_HAZARDS.iter().find(|(id, _)| *id == m.id);
            if n_opp_remaining >= 3 && hazard.is_some_and(|(_, sc)| !v.opp_side.contains(sc)) {
                return Ok(hit(br, line!(), Order::Move(i)));
            }
            if !v.side.is_empty() && has(t::ANTI_HAZARDS_MOVES, &m.id) && n_remaining >= 2 {
                return Ok(hit(br, line!(), Order::Move(i)));
            }
        }
        // 5. Setup at full HP into a winning matchup.
        if active.hp == 1.0 && matchup > 0.0 {
            for (i, m) in v.moves.iter().enumerate() {
                if let Some(b) = self_setup_boosts(m, active).filter(|b| !b.is_empty()) {
                    if raised_stages(b) >= 2
                        && b.iter().filter(|(_, x)| *x > 0).map(|(s, _)| active.boost(s)).min().expect("raised >= 2") < 6
                    {
                        // F-LF-1 SETUP SITE — bots_gate_test.py asserts the COMMIT bank reaches it
                        return Ok(hit(br, line!(), Order::Move(i)));
                    }
                }
            }
        }
        if opp.status.is_none() && matchup <= 0.0 && best_frac < t::WALL_FRACTION && opp.hp > t::STATUS_WALL_HP_THRESHOLD {
            if let Some(o) = first_move(v, |m| has(t::STATUS_MOVES, &m.id) && m.bp == 0 && !is_status_move_immune(&m.id, opp)) {
                return Ok(hit(br, line!(), o));
            }
        }
        if let Some(b) = best {
            return Ok(hit(br, line!(), Order::Move(b)));
        }
    }
    if let Some(s) = best_switch_v2(v)? {
        return Ok(hit(br, line!(), Order::Switch(s)));
    }
    Ok(hit(br, line!(), random_move(v, choice)))
}

/// `baitbot.blocks(move, defender)`.
fn blocks(m: &super::view::MoveV, defender: &super::view::MonV) -> bool {
    if m.bp == 0 {
        return false;
    }
    effective_multiplier(m.typ, defender) == 0.0
}

/// `Gen3BaitBotPlayer.choose_move` (`p_bait` its dial; `bait` its `_rng`).
pub fn baitbot(v: &View, p_bait: f64, choice: &mut PyRandom, bait: &mut PyRandom, br: &mut u32) -> R<Order> {
    let (Some(_), Some(opp)) = (v.active_mon(), v.opp_active.as_ref()) else {
        return Ok(hit(br, line!(), random_move(v, choice)));
    };
    if v.moves.is_empty() && !v.switches.is_empty() {
        return Ok(hit(br, line!(), match best_switch_v2(v)? {
            Some(s) => Order::Switch(s),
            None => Order::Switch(v.switches[bait.choice(v.switches.len()).expect("non-empty")]),
        }));
    }
    let attacks: Vec<&super::view::MoveV> = opp.moves.iter().filter(|m| m.bp > 0).collect();
    let targets: Vec<usize> = if attacks.is_empty() {
        Vec::new()
    } else {
        v.switches.iter().copied().filter(|&s| !v.team[s].fainted && attacks.iter().all(|m| blocks(m, &v.team[s]))).collect()
    };
    if !targets.is_empty() && !v.switches.is_empty() && bait.random() < p_bait {
        let i = argmax(&targets, |&s| Ok(v.team[s].hp))?.expect("non-empty");
        return Ok(hit(br, line!(), Order::Switch(targets[i])));
    }
    if let Some(i) = best_damage_move_v2(v)? {
        return Ok(hit(br, line!(), Order::Move(i)));
    }
    if !v.moves.is_empty() {
        let i = argmax(&v.moves, |m| Ok(m.bp as f64))?.expect("non-empty");
        return Ok(hit(br, line!(), Order::Move(i)));
    }
    Ok(hit(br, line!(), random_move(v, choice)))
}
