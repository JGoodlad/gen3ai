//! [`legal_actions`] — `LegalActions.from_battle` over a [`Tracker`] (the raw `|request|` plus
//! poke-env's parse of it), and [`mask`] — `Gen3ActionMasker.mask_from_legal`.
//!
//! **Hybrid sourcing, kept exactly as the Python surface keeps it** (rule V13): `move_slots` is
//! WIRE truth (`request["active"][0]["moves"]`, Struggle filtered out), while the six flags are
//! poke-env's parse (`Battle.parse_request`): `switches` from `_available_switches` indexed by the
//! STABLE team order, `trapped` / `maybe_trapped` / `force_switch` / `wait`, and `struggle` =
//! `any(m.id == "struggle" for m in available_moves)`.

use super::tracker::{truthy, Tracker};
use crate::core_events::json_out;
use crate::core_events::jsonval::Val;

/// `LegalMove`.
#[derive(Debug, Clone, PartialEq)]
pub struct LegalMove {
    pub id: String,
    pub current_pp: i64,
    pub max_pp: i64,
    pub disabled: bool,
    pub target: Option<String>,
}

/// `LegalSwitch`.
#[derive(Debug, Clone, PartialEq)]
pub struct LegalSwitch {
    pub species: String,
    pub slot: usize,
}

/// `LegalActions` (minus the `last_request` mirror, which is [`Tracker::last_request_text`]).
#[derive(Debug, Clone, PartialEq)]
pub struct LegalActions {
    pub move_slots: Vec<LegalMove>,
    pub switches: Vec<LegalSwitch>,
    pub force_switch: bool,
    pub trapped: bool,
    pub maybe_trapped: bool,
    pub wait: bool,
    pub struggle: bool,
    pub own_hp_typed_id: Option<String>,
}

/// `LegalActions.from_battle(battle)` — `None` when the side has no request at all (the
/// `last_request or None` guard makes every field empty; the caller treats it as no decision).
pub fn legal_actions(t: &Tracker) -> Option<LegalActions> {
    let req = t.last_request.as_ref()?;
    let mut move_slots = Vec::new();
    if let Some(Val::Arr(active)) = req.get("active") {
        if let Some(Val::Arr(moves)) = active.first().and_then(|a| a.get("moves")) {
            for m in moves {
                let id = m.str_at("id").unwrap_or("").to_string();
                if id == "struggle" {
                    continue;
                }
                let int = |k: &str| match m.get(k) {
                    Some(Val::Int(n)) => *n,
                    _ => 0,
                };
                move_slots.push(LegalMove {
                    current_pp: int("pp"),
                    max_pp: int("maxpp"),
                    disabled: truthy(m.get("disabled")),
                    target: m.str_at("target").map(str::to_string),
                    id,
                });
            }
        }
    }
    // `for i, mon in enumerate(battle.team.values()) if mon in available` — team order.
    let mut idx = t.available_switches.clone();
    idx.sort_unstable();
    idx.dedup();
    let switches = idx.iter().map(|&i| LegalSwitch { species: t.team[i].1.species.clone(), slot: i }).collect();
    // `_own_hp_typed_id`: the active mon's single `hiddenpower*` Move.id.
    let own_hp_typed_id = t.active_index(true).and_then(|i| {
        let typed: Vec<String> = t.team[i]
            .1
            .moves
            .moves()
            .into_iter()
            .map(|(_, m)| m.id)
            .filter(|id| id.starts_with("hiddenpower"))
            .collect();
        if typed.len() == 1 {
            typed.into_iter().next()
        } else {
            None
        }
    });
    Some(LegalActions {
        move_slots,
        switches,
        force_switch: t.force_switch,
        trapped: t.trapped,
        maybe_trapped: t.maybe_trapped,
        wait: t.wait,
        struggle: t.available_moves.iter().any(|m| m == "struggle"),
        own_hp_typed_id,
    })
}

/// The action space's layout (`agents/action/constants.py`): six switch slots, four move slots,
/// Struggle.
pub const ACTION_SPACE_SIZE: usize = 11;
const SWITCH_END: usize = 6;
const MOVE_START: usize = 6;
const N_MOVE_SLOTS: usize = 4;
const STRUGGLE: usize = 10;

/// `Gen3ActionMasker.mask_from_legal(legal)`.
pub fn mask(legal: &LegalActions) -> [u8; ACTION_SPACE_SIZE] {
    let mut m = [0u8; ACTION_SPACE_SIZE];
    for sw in &legal.switches {
        if sw.slot < SWITCH_END {
            m[sw.slot] = 1;
        }
    }
    for (i, mv) in legal.move_slots.iter().enumerate() {
        if i < N_MOVE_SLOTS && !mv.disabled {
            m[i + MOVE_START] = 1;
        }
    }
    if legal.struggle {
        m[STRUGGLE] = 1;
    }
    m
}

impl LegalActions {
    /// JSON in `LegalActions`' own shape, plus the 11-bit `mask`.
    pub fn json(&self) -> String {
        let mut o = String::from("{\"move_slots\":[");
        for (i, m) in self.move_slots.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            o.push_str("{\"id\":");
            json_out::str_into(&mut o, &m.id);
            o.push_str(&format!(",\"current_pp\":{},\"max_pp\":{},\"disabled\":{},\"target\":", m.current_pp, m.max_pp, m.disabled));
            json_out::opt_str_into(&mut o, m.target.as_deref());
            o.push('}');
        }
        o.push_str("],\"switches\":[");
        for (i, s) in self.switches.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            o.push_str("{\"species\":");
            json_out::str_into(&mut o, &s.species);
            o.push_str(&format!(",\"slot\":{}}}", s.slot));
        }
        o.push_str(&format!(
            "],\"force_switch\":{},\"trapped\":{},\"maybe_trapped\":{},\"wait\":{},\"struggle\":{},\"own_hp_typed_id\":",
            self.force_switch, self.trapped, self.maybe_trapped, self.wait, self.struggle
        ));
        json_out::opt_str_into(&mut o, self.own_hp_typed_id.as_deref());
        o.push_str(",\"mask\":[");
        o.push_str(&mask(self).iter().map(|b| b.to_string()).collect::<Vec<_>>().join(","));
        o.push_str("]}");
        o
    }
}
