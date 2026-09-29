//! Family `intent` — `opp_action_kind`, `opp_action_num`, `opp_switch_slot`, `opp_switch_species`
//! (`Gen3Env._opp_intent_labels`, `opp_intent_labels.build_opp_intent_label`), rule for rule.
//!
//! The DECISION-vs-CONSEQUENCE rules are the port's `trackers::IntentLabel` (slice T already holds
//! it equal to the Python label in ids, both viewers, every decision); this module is the num step
//! the Python env adds:
//! * no label (no delta yet) or `KIND_UNKNOWN` ⇒ `(2, 0, SWITCH_SLOT_NONE, 0)`;
//! * MOVE ⇒ the move's num; a bare Hidden Power (num 237) is resolved to the ATTACKER's TRUE typed
//!   num from the OTHER side's own team (`_intent_move_num_resolver`: the first truth mon of that
//!   species, its first typed `hiddenpower<type>` move; none ⇒ 237 stays). An id with no num ⇒ the
//!   UNKNOWN label (masked, never guessed);
//! * SWITCH ⇒ `(1, 0, slot in the previous decision's revealed frame or SWITCH_SLOT_NONE, species
//!   num or 0)`.

use pokesim::encoder::data::tables;
use pokesim::present::board_reading::BoardReading;
use pokesim::present::dex::to_id;
use pokesim::trackers::{SideTrackers, KIND_MOVE, KIND_SWITCH, KIND_UNKNOWN};

use super::per_slot::{hp_type_idx, HP_TYPE_NAMES};

/// `opp_intent_labels.SWITCH_SLOT_NONE`.
pub const SWITCH_SLOT_NONE: i64 = -100;
/// `dex_ids.HIDDEN_POWER_NUM` — the Python constant (the bare, typeless Hidden Power's num).
pub const HIDDEN_POWER_NUM: i64 = 237;

fn move_num(id: &str) -> Option<i64> {
    tables().moves.get(&to_id(id)).map(|r| r.num)
}

/// `_intent_move_num_resolver(delta)(move_id)`.
fn resolve(move_id: &str, attacker: Option<&str>, truth: &BoardReading) -> Option<i64> {
    let num = move_num(move_id)?;
    if num != HIDDEN_POWER_NUM {
        return Some(num);
    }
    let Some(attacker) = attacker.filter(|a| !a.is_empty()) else { return Some(num) };
    let key = to_id(attacker);
    if let Some((_, m)) = truth.team.iter().find(|(_, m)| to_id(&m.species) == key) {
        for (_, mv) in m.moves.moves_ref() {
            if let Some(t) = hp_type_idx(&mv.id) {
                let typed = format!("hiddenpower{}", HP_TYPE_NAMES[t as usize]);
                return Some(tables().moves.get(&typed).map(|r| r.num).expect("every typed Hidden Power has a dex num"));
            }
        }
    }
    Some(num)
}

/// `(kind, move_num, switch_slot, switch_species)`.
pub fn label(trk: Option<&SideTrackers>, truth: &BoardReading) -> [i64; 4] {
    const UNKNOWN: [i64; 4] = [KIND_UNKNOWN as i64, 0, SWITCH_SLOT_NONE, 0];
    let Some(l) = trk.and_then(|t| t.label.as_ref()) else { return UNKNOWN };
    match l.kind {
        KIND_MOVE => match l.move_id.as_deref().and_then(|m| resolve(m, l.attacker.as_deref(), truth)) {
            Some(num) => [KIND_MOVE as i64, num, SWITCH_SLOT_NONE, 0],
            None => UNKNOWN,
        },
        KIND_SWITCH => {
            let sp = l.switch_species.as_deref().unwrap_or("");
            let num = if sp.is_empty() { 0 } else { tables().species.get(&to_id(sp)).map_or(0, |r| r.num as i64) };
            [KIND_SWITCH as i64, 0, l.switch_slot.map_or(SWITCH_SLOT_NONE, |s| s as i64), num]
        }
        _ => UNKNOWN,
    }
}
