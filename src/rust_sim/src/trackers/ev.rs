//! Typed accessors over a [`Reading`] — `agents.battle.battle_event.BattleEvent`'s properties,
//! with Python's truthiness where the trackers rely on it (an empty string reads as absent).

use crate::core_events::{EventKind, Reading, Rel, Value};

/// `value[key]` as a string (a `Null` / absent / non-string entry is `None`).
pub fn s<'a>(r: &'a Reading, key: &str) -> Option<&'a str> {
    match r.get(key) {
        Some(Value::Str(x)) => Some(x.as_str()),
        _ => None,
    }
}

/// `value[key]` as a number (an int or a float).
pub fn num(r: &Reading, key: &str) -> Option<f64> {
    match r.get(key) {
        Some(Value::Int(n)) => Some(*n as f64),
        Some(Value::Float(x)) => Some(*x),
        _ => None,
    }
}

/// Python truthiness of an optional string.
pub fn nz(x: Option<&str>) -> Option<&str> {
    x.filter(|s| !s.is_empty())
}

pub fn actor(r: &Reading) -> Option<&str> {
    nz(r.actor.as_deref())
}
pub fn move_id(r: &Reading) -> Option<&str> {
    s(r, "move_id")
}
pub fn status(r: &Reading) -> Option<&str> {
    s(r, "status")
}
pub fn reason(r: &Reading) -> Option<&str> {
    s(r, "reason")
}
/// `BattleEvent.from_clause`: `value.get("reason") or value.get("from")`.
pub fn from_clause(r: &Reading) -> Option<&str> {
    match nz(reason(r)) {
        Some(x) => Some(x),
        None => s(r, "from"),
    }
}
pub fn amount(r: &Reading) -> Option<f64> {
    num(r, "amount")
}
pub fn item(r: &Reading) -> Option<&str> {
    s(r, "item")
}
pub fn stat(r: &Reading) -> Option<&str> {
    s(r, "stat")
}
pub fn multiplier(r: &Reading) -> Option<f64> {
    num(r, "multiplier")
}
pub fn cant_move(r: &Reading) -> Option<&str> {
    s(r, "move")
}
pub fn blocked_side(r: &Reading) -> Option<Rel> {
    match s(r, "of_side") {
        Some("ours") => Some(Rel::Ours),
        Some("opp") => Some(Rel::Opp),
        _ => None,
    }
}
pub fn blocked_actor(r: &Reading) -> Option<&str> {
    s(r, "of_actor")
}

pub fn is(r: &Reading, k: EventKind) -> bool {
    r.kind == k
}

/// The other side.
pub fn other(r: Rel) -> Rel {
    match r {
        Rel::Ours => Rel::Opp,
        Rel::Opp => Rel::Ours,
    }
}

/// `BattleEvent.effect`: `value.get("effect") or value.get("condition")`.
pub fn effect(r: &Reading) -> Option<&str> {
    match nz(s(r, "effect")) {
        Some(x) => Some(x),
        None => s(r, "condition"),
    }
}

/// `turn_view.is_protect_block` — an `|-activate|` effect that is a Protect / Detect BLOCK of the move
/// being used (`gen3_event_window_semantics_fixes_v1`, W4). Endure's `-activate` is not a block.
pub fn is_protect_block(effect: Option<&str>) -> bool {
    matches!(
        effect.unwrap_or("").trim().to_lowercase().as_str(),
        "protect" | "detect" | "move: protect" | "move: detect"
    )
}

/// `turn_view.damage_is_lethal` — FALSE iff the damage is KNOWN to have left its mon above 0 HP
/// (`hp_after` is an optional key: an absent one keeps the classic reading).
pub fn damage_is_lethal(r: &Reading) -> bool {
    num(r, "hp_after").is_none_or(|h| h <= 0.0)
}
