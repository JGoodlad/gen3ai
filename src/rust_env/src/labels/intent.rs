//! Family `intent` — `opp_action_kind`, `opp_action_num`, `opp_switch_slot`, `opp_switch_species`
//! (`Gen3Env._opp_intent_labels`, `opp_intent_labels.build_opp_intent_label`), rule for rule.
//!
//! The DECISION-vs-CONSEQUENCE rules are the port's `trackers::IntentLabel` (slice T already holds
//! it equal to the Python label in ids, both viewers, every decision); this module is the num step
//! the Python env adds:
//! * no label (no delta yet) or `KIND_UNKNOWN` ⇒ `(2, 0, SWITCH_SLOT_NONE, 0)`;
//! * MOVE ⇒ the move's num; a bare Hidden Power (num 237) is resolved to the ATTACKER's TRUE typed
//!   num from the OTHER side's own team (`_intent_move_num_resolver`: the first truth mon of that
//!   species, matched by dex num, its first typed `hiddenpower<type>` move; none ⇒ 237 stays).
//!   An id with no num, or an attacker not on the truth team, is an `Err` (a FAULT; F-X5-3 — it
//!   was a silent UNKNOWN label / a silent 237);
//! * SWITCH ⇒ `(1, 0, slot in the previous decision's revealed frame or SWITCH_SLOT_NONE, species
//!   num)`; a switch-in with no species or no num is an `Err` (it was a silent 0).

use pokesim::encoder::data::tables;
use pokesim::present::board_reading::BoardReading;
use pokesim::present::dex::to_id;
use pokesim::trackers::{SideTrackers, KIND_MOVE, KIND_SWITCH, KIND_UNKNOWN};

use super::belief::{move_num, species_num, truth_index};
use super::per_slot::{hp_type_idx, HP_TYPE_NAMES};

/// `opp_intent_labels.SWITCH_SLOT_NONE`.
pub const SWITCH_SLOT_NONE: i64 = -100;
/// `dex_ids.HIDDEN_POWER_NUM` — the Python constant (the bare, typeless Hidden Power's num).
pub const HIDDEN_POWER_NUM: i64 = 237;

/// `_intent_move_num_resolver(delta)(move_id)`. A move with no num, or a bare Hidden Power whose
/// attacker is not on the truth team, is an `Err` (F-X5-3) — it was a silent UNKNOWN / 237.
fn resolve(move_id: &str, attacker: Option<&str>, truth: &BoardReading) -> Result<i64, String> {
    let num = move_num(&to_id(move_id)).map_err(|e| format!("intent labels: {e}"))?;
    if num != HIDDEN_POWER_NUM {
        return Ok(num);
    }
    let Some(attacker) = attacker.filter(|a| !a.is_empty()) else { return Ok(num) };
    let a_num = species_num(&to_id(attacker)).map_err(|e| format!("intent labels: {e}"))?;
    let (_, m) = &truth.team[truth_index(truth, a_num, &format!("intent labels: the Hidden Power attacker {attacker:?}"))?];
    for (_, mv) in m.moves.moves_ref() {
        if let Some(t) = hp_type_idx(&mv.id) {
            let typed = format!("hiddenpower{}", HP_TYPE_NAMES[t as usize]);
            return Ok(tables().moves.get(&typed).map(|r| r.num).expect("every typed Hidden Power has a dex num"));
        }
    }
    Ok(num)
}

/// `(kind, move_num, switch_slot, switch_species)`. `Err` = a lookup that cannot be made (F-X5-3).
pub fn label(trk: Option<&SideTrackers>, truth: &BoardReading) -> Result<[i64; 4], String> {
    const UNKNOWN: [i64; 4] = [KIND_UNKNOWN as i64, 0, SWITCH_SLOT_NONE, 0];
    let Some(l) = trk.and_then(|t| t.label.as_ref()) else { return Ok(UNKNOWN) };
    Ok(match l.kind {
        KIND_MOVE => match l.move_id.as_deref() {
            Some(m) => [KIND_MOVE as i64, resolve(m, l.attacker.as_deref(), truth)?, SWITCH_SLOT_NONE, 0],
            None => UNKNOWN,
        },
        KIND_SWITCH => {
            let sp = l.switch_species.as_deref().filter(|s| !s.is_empty()).ok_or_else(|| {
                "intent labels: a SWITCH label with no switch-in species — never a silent species 0 (F-X5-3)".to_string()
            })?;
            let num = species_num(&to_id(sp)).map_err(|e| format!("intent labels: {e}"))?;
            [KIND_SWITCH as i64, 0, l.switch_slot.map_or(SWITCH_SLOT_NONE, |s| s as i64), num]
        }
        _ => UNKNOWN,
    })
}
