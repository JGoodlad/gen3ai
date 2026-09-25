//! [`choice_tokens`] — the sim CHOICE STRING of every legal action, as the real action mapper
//! produces it (`view_successor._choice_map` → `Gen3ActionMapper.action_to_order(idx, battle,
//! legal=legal).message[len("/choose "):]`): the tokens a search branches on at the next ply.
//!
//! The mapper's three steps, each mirrored at one site:
//!
//! * `action_to_choice(idx, legal)` — a switch slot, a request move slot (never a disabled one), or
//!   Struggle;
//! * `choice_to_order(choice, battle)` — a switch is `SingleBattleOrder(team_list[slot])` iff the mon
//!   is in `available_switches` (message `switch <Pokemon.name>`); a move is the FIRST of
//!   `battle.available_moves` whose `Move.id` equals the request id, or — Hidden Power — the first
//!   typed `hiddenpower*` when the request id is `hiddenpower*` too (`move <Move.id>`; `recharge`
//!   is `move 1`); Struggle is `move struggle`;
//! * the ROUND TRIP `order_to_action(order, battle, legal)` must give `idx` back — the mapper raises
//!   otherwise, and `_choice_map` drops the index.
//!
//! Slice O compares these tokens against the Python mapper at every decision of every corpus battle
//! (`agents/battle/rust_core_parity_obs.py`).

use super::board_reading::BoardReading;
use super::legal::{mask, LegalActions};
use crate::core_error::{fault, CoreResult};

/// `battle.available_moves` as `Move.id`s — each request id resolved to the moveset object poke-env
/// holds for it (a bare `hiddenpower` to the single typed one; a special / caller-granted id is its
/// own `Move`).
fn available_move_ids(r: &BoardReading) -> Vec<String> {
    let active = r.active_index(true).map(|i| &r.team[i].1);
    r.available_moves
        .iter()
        .map(|id| {
            let Some(mon) = active else { return id.clone() };
            let moves = mon.moves.moves();
            if let Some((_, m)) = moves.iter().find(|(k, _)| k == id) {
                return m.id.clone();
            }
            if id == "hiddenpower" {
                let hps: Vec<&String> = moves.iter().filter(|(k, _)| k.starts_with("hiddenpower")).map(|(_, m)| &m.id).collect();
                if hps.len() == 1 {
                    return hps[0].clone();
                }
            }
            id.clone()
        })
        .collect()
}

/// `serialize.order_to_action` for a MOVE order — the first request slot whose id matches (Hidden
/// Power by prefix), or 0.
fn move_round_trip(move_id: &str, legal: &LegalActions) -> usize {
    if move_id == "struggle" {
        return 10;
    }
    for (i, m) in legal.move_slots.iter().enumerate() {
        if m.id == move_id || (m.id.starts_with("hiddenpower") && move_id.starts_with("hiddenpower")) {
            return 6 + i;
        }
    }
    0
}

/// `{action index: choice string}` for every legal action (ascending index).
pub fn choice_tokens(r: &BoardReading, legal: &LegalActions) -> CoreResult<Vec<(usize, String)>> {
    let m = mask(legal);
    let avail = available_move_ids(r);
    let mut out = Vec::new();
    for idx in (0..m.len()).filter(|&i| m[i] == 1) {
        let token = if idx < 6 {
            let Some(sw) = legal.switches.iter().find(|s| s.slot == idx) else { continue };
            let Some((_, mon)) = r.team.get(sw.slot) else { continue };
            if !r.available_switches.contains(&sw.slot) {
                continue;
            }
            let name = mon.name.as_deref().ok_or_else(|| fault(format!("choice token: team slot {} has no name", sw.slot)))?;
            // the round trip: the order's Pokemon is found at its own team index
            format!("switch {name}")
        } else if idx < 10 {
            let slot = idx - 6;
            let Some(lm) = legal.move_slots.get(slot).filter(|lm| !lm.disabled) else { continue };
            let found = avail.iter().find(|a| {
                *a == &lm.id || (a.starts_with("hiddenpower") && lm.id.starts_with("hiddenpower"))
            });
            let Some(mid) = found else { continue };
            if move_round_trip(mid, legal) != idx {
                continue;
            }
            if mid == "recharge" {
                "move 1".to_string()
            } else {
                format!("move {mid}")
            }
        } else {
            if !legal.struggle {
                continue;
            }
            "move struggle".to_string()
        };
        out.push((idx, token));
    }
    Ok(out)
}

/// The tokens as a JSON object `{"<idx>":"<token>",…}`.
pub fn tokens_json(tokens: &[(usize, String)]) -> String {
    let mut o = String::from("{");
    for (i, (idx, t)) in tokens.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        o.push_str(&format!("\"{idx}\":"));
        crate::core_events::json_out::str_into(&mut o, t);
    }
    o.push('}');
    o
}
