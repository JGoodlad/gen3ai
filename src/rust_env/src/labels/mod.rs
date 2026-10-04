//! THE TRAINING LABELS — the label columns the env core writes per decision (M5 Lane C,
//! `designs/endstate/program_rust_core.md` §2 M5; the inventory and each key's derivation:
//! `designs/rust_sim/env_labels.md`).
//!
//! The columns and the family → column map are GENERATED (`core::columns::labels`, from
//! `src/utils/rust_env/label_inventory.py`). This module owns which families are BUILT and, per
//! family, the producer. A spec that declares a family the core does not write is refused at
//! STARTUP (the DECLARED LIFECYCLE): a `host_const` / `host_episode` family belongs to the host, a
//! `core` family not yet built is refused as such — never a silently stale column.

use pokesim::present::board_reading::BoardReading;

use crate::core::columns::labels::{FAMILIES, NOT_CORE};
use crate::core::columns::EnvCols;

pub mod belief;
pub mod intent;
pub mod margin;
pub mod per_slot;
pub mod spread;

/// Per-episode label state (one env; cleared at every episode start). No family keeps any today:
/// `spread` reads the truth team's declared spread per decision (its old per-episode inversion cache
/// is gone with the inversion, `gen3_true_spread_labels_v1`). The hook stays so a family that needs
/// episode state has one place to put it, cleared by the pool at every start.
#[derive(Default)]
pub struct EpisodeState {}

impl EpisodeState {
    pub fn clear(&mut self) {}
}

/// The `core` families whose producer exists (grows one Lane-C unit at a time).
pub const BUILT: &[&str] = &["belief", "hp_type", "item", "spread", "intent", "margin"];

/// Write every DECLARED family's columns for `side`'s open decision. `own` is the side's reading,
/// `truth` the OTHER side's (its own team = `battle2.team`), `oracle` the ORACLE REVEAL's opponent team
/// when the row states unseen species (`None` = the mode `off`), `row` the row just encoded for `side`.
/// `Err` is a label invariant the Python env raises on (the caller makes it a FAULT).
pub fn write(
    families: &[&'static str],
    side: usize,
    own: &BoardReading,
    own_trk: Option<&pokesim::trackers::SideTrackers>,
    own_view: Option<&pokesim::present::view::OneSidedView>,
    dec_n: u32,
    truth: &BoardReading,
    oracle: Option<&pokesim::encoder::oracle::Oracle>,
    row: &[f32],
    _st: &mut EpisodeState,
    c: &mut EnvCols,
) -> Result<(), String> {
    use pokesim::encoder::layout::TEAM_SIZE as T;
    for &f in families {
        match f {
            "belief" => {
                let m = belief::MOVE_SLOTS;
                belief::write(
                    own,
                    truth,
                    oracle,
                    row,
                    &mut c.belief_species[side * T..(side + 1) * T],
                    &mut c.belief_moves[side * T * m..(side + 1) * T * m],
                    &mut c.known_moves[side * T * m..(side + 1) * T * m],
                )?
            }
            "hp_type" => per_slot::write_hp_type(
                own,
                truth,
                oracle,
                row,
                &mut c.hp_type_label[side * T..(side + 1) * T],
                &mut c.hp_type_mask[side * T..(side + 1) * T],
            )?,
            "item" => per_slot::write_item(
                own,
                truth,
                oracle,
                row,
                &mut c.item_label[side * T..(side + 1) * T],
                &mut c.item_mask[side * T..(side + 1) * T],
            )?,
            "spread" => {
                let n = spread::N_SPREAD;
                spread::write(
                    own,
                    truth,
                    oracle,
                    row,
                    &mut c.belief_spread[side * T * n..(side + 1) * T * n],
                    &mut c.belief_spread_mask[side * T..(side + 1) * T],
                    &mut c.belief_nature[side * T..(side + 1) * T],
                    &mut c.belief_nature_mask[side * T..(side + 1) * T],
                    &mut c.belief_ev[side * T * n..(side + 1) * T * n],
                    &mut c.belief_ev_mask[side * T..(side + 1) * T],
                )?
            }
            "intent" => {
                let [kind, num, slot, species] = intent::label(own_trk, truth)?;
                c.opp_action_kind[side] = kind;
                c.opp_action_num[side] = num;
                c.opp_switch_slot[side] = slot;
                c.opp_switch_species[side] = species;
            }
            "margin" => {
                let v = own_view.ok_or("family \"margin\" needs the side's view")?;
                margin::write(v, dec_n, &mut c.win_margin[side..side + 1]);
            }
            other => return Err(format!("label family {other:?} was declared but has no producer (labels::BUILT drifted)")),
        }
    }
    Ok(())
}

/// Validate the spec's `labels` declaration; the families in TABLE order (deterministic).
pub fn declare(names: &[String]) -> Result<Vec<&'static str>, String> {
    for (i, n) in names.iter().enumerate() {
        if names[..i].contains(n) {
            return Err(format!("spec: `labels` names {n:?} twice"));
        }
        if let Some((_, why)) = NOT_CORE.iter().find(|(f, _)| f == n) {
            return Err(format!("spec: label family {n:?} is filled by the HOST ({why}), not by the core"));
        }
        if !FAMILIES.iter().any(|(f, _)| f == n) {
            return Err(format!("spec: unknown label family {n:?}"));
        }
        if !BUILT.contains(&n.as_str()) {
            return Err(format!("spec: label family {n:?} is not built yet in the Rust env (M5 Lane C)"));
        }
    }
    let out: Vec<&'static str> = FAMILIES.iter().map(|(f, _)| *f).filter(|f| names.iter().any(|n| n == f)).collect();
    if out.contains(&"spread") {
        spread::prepare().map_err(|e| format!("spec: label family \"spread\": {e}"))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_declaration_is_refused_by_kind() {
        assert_eq!(declare(&[]).unwrap(), Vec::<&str>::new());
        let e = |n: &str| declare(&[n.to_string()]).unwrap_err();
        assert!(e("winprob").contains("HOST"), "{}", e("winprob"));
        assert!(e("opp_class").contains("HOST"), "{}", e("opp_class"));
        assert!(e("nope").contains("unknown"));
        for (f, _) in FAMILIES {
            if !BUILT.contains(&f) {
                assert!(e(f).contains("not built"), "{f}");
            }
        }
    }
}
