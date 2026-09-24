//! The per-decision CONTEXT (`BattleContext`, the fields a consumer reads) and the frozen
//! `TurnDelta`'s LOSSY PROJECTION — only the fields the two live consumers read: the progress
//! clock (an obs scalar) and the α/β opponent-intent label. It is not the core's record of what
//! happened (that is [`super::record`]); it exists so those two consumers are byte-identical to the
//! Python path (`TurnDelta.build_from_events`, `agents/training/turn_delta.py`).

use super::ev;
use super::turnview::{DamagingMove, TurnView};
use crate::core_error::{fault, CoreResult};
use crate::core_events::json_out;
use crate::core_events::{EventKind as K, Reading, Rel};
use crate::present::view::MonView;
use crate::present::{LegalActions, OneSidedView};

/// `BOOST_STATS` order (`agents.gen3_mechanics`).
pub const BOOST_STATS: [&str; 7] = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"];

/// `SlotRegistry` — species → a stable slot 0..6 in first-seen order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Slots(pub Vec<String>);

impl Slots {
    pub fn assign(&mut self, species: &str) -> CoreResult<usize> {
        if let Some(i) = self.0.iter().position(|s| s == species) {
            return Ok(i);
        }
        if self.0.len() >= 6 {
            return Err(fault(format!("SlotRegistry overflow: {species:?} exceeds 6 slots ({:?})", self.0)));
        }
        self.0.push(species.to_string());
        Ok(self.0.len() - 1)
    }
    pub fn get(&self, species: &str) -> Option<usize> {
        self.0.iter().position(|s| s == species)
    }
    pub fn json_into(&self, out: &mut String) {
        out.push('[');
        for (i, s) in self.0.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            json_out::str_into(out, s);
        }
        out.push(']');
    }
}

/// The CONTEXT fields a consumer reads (`BattleContext.from_battle`, via `view_context`).
#[derive(Debug, Clone, PartialEq)]
pub struct Ctx {
    pub turn: u32,
    pub forced_switch: bool,
    pub our_slots: Slots,
    pub opp_slots: Slots,
    pub our_hp: [f32; 6],
    pub opp_hp: [f32; 6],
    /// `"NONE"` when no (living) active.
    pub our_active: String,
    pub opp_active: String,
    pub our_fainted: usize,
    pub opp_fainted: usize,
    pub active_move_ids: [Option<String>; 4],
    pub our_boosts: [i8; 7],
    pub opp_boosts: [i8; 7],
    pub our_team_order: Vec<String>,
    pub own_hp_typed_id: Option<String>,
    /// `battle.opp_last_damaging_move` — poke-env's PROMOTED damaging move (an effectiveness
    /// emission for the defender landed in the move's turn), TURN-GATED to the turn that just
    /// resolved: the Hidden-Power belief's input. Read off the board reading
    /// (`BoardReading::last_damaging_move`), NOT the decision window — a window that opened at a
    /// mid-turn forced switch holds only the turn's tail.
    pub opp_last_damaging: Option<DamagingMove>,
}

fn boosts(m: Option<&MonView>) -> [i8; 7] {
    let mut out = [0i8; 7];
    if let Some(m) = m {
        for (k, v) in &m.boosts {
            if let Some(i) = BOOST_STATS.iter().position(|s| s == k) {
                out[i] = *v as i8;
            }
        }
    }
    out
}

fn hp_by_slot(mons: &[MonView], slots: &Slots) -> [f32; 6] {
    let mut out = [0f32; 6];
    for m in mons {
        if let Some(i) = slots.get(&m.species) {
            out[i] = m.hp_fraction as f32;
        }
    }
    out
}

impl Ctx {
    /// `view_context(live, legal, mask, our_slots, opp_slots, events)` — MUTATES the registries,
    /// exactly as `from_battle` does.
    pub fn build(live: &OneSidedView, legal: Option<&LegalActions>, our_slots: &mut Slots, opp_slots: &mut Slots,
                 opp_last_damaging: Option<DamagingMove>) -> CoreResult<Ctx> {
        for m in &live.ours.mons {
            our_slots.assign(&m.species)?;
        }
        for m in &live.opp.mons {
            opp_slots.assign(&m.species)?;
        }
        let our_a = live.ours.active.map(|i| &live.ours.mons[i]);
        let opp_a = live.opp.active.map(|i| &live.opp.mons[i]);
        let mut active_move_ids: [Option<String>; 4] = Default::default();
        if let Some(l) = legal {
            for (i, m) in l.move_slots.iter().take(4).enumerate() {
                active_move_ids[i] = Some(m.id.clone());
            }
        }
        Ok(Ctx {
            turn: live.turn,
            forced_switch: legal.is_some_and(|l| l.force_switch),
            our_hp: hp_by_slot(&live.ours.mons, our_slots),
            opp_hp: hp_by_slot(&live.opp.mons, opp_slots),
            our_slots: our_slots.clone(),
            opp_slots: opp_slots.clone(),
            our_active: our_a.filter(|m| !m.fainted).map_or("NONE".into(), |m| m.species.clone()),
            opp_active: opp_a.map_or("NONE".into(), |m| m.species.clone()),
            our_fainted: live.ours.mons.iter().filter(|m| m.fainted).count(),
            opp_fainted: live.opp.mons.iter().filter(|m| m.fainted).count(),
            active_move_ids,
            our_boosts: boosts(our_a),
            opp_boosts: boosts(opp_a),
            our_team_order: live.ours.mons.iter().map(|m| m.species.clone()).collect(),
            own_hp_typed_id: legal.and_then(|l| l.own_hp_typed_id.clone()),
            opp_last_damaging,
        })
    }
}

/// The frozen `TurnDelta`, restricted to what the clock and the intent label read.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DeltaProjection {
    pub our_move_id: Option<String>,
    pub our_switch_to: Option<String>,
    pub our_prev_active: Option<String>,
    pub opp_move_id: Option<String>,
    pub opp_switch_to: Option<String>,
    pub opp_prev_active: Option<String>,
    pub our_hp_delta: [f32; 6],
    pub opp_hp_delta: [f32; 6],
    pub we_fainted: bool,
    pub opp_fainted: bool,
    pub our_failed_to_move: bool,
    pub our_move_outcome: Option<&'static str>,
    pub our_status_applied: Option<String>,
    pub our_status_cured: Option<String>,
    pub opp_status_applied: Option<String>,
    pub our_damaging_event: Option<DamagingMove>,
    pub opp_damaging_event: Option<DamagingMove>,
    pub opp_target_hp_delta: Option<f32>,
    pub phase_is_forced_switch: bool,
    pub decision_was_forced_switch: bool,
}

/// A float32 sum in index order — `np.float32 array .sum()` for a 6-vector.
fn sum32(a: &[f32; 6]) -> f32 {
    a.iter().fold(0f32, |s, x| s + x)
}

impl DeltaProjection {
    pub fn our_hp_sum(&self) -> f32 {
        sum32(&self.our_hp_delta)
    }
    pub fn opp_hp_sum(&self) -> f32 {
        sum32(&self.opp_hp_delta)
    }
    /// `TurnDelta.opp_resolved_move_id`.
    pub fn opp_resolved_move_id(&self) -> Option<&str> {
        match &self.opp_damaging_event {
            Some(d) => Some(d.move_id.as_deref().unwrap_or("")),
            None => self.opp_move_id.as_deref(),
        }
    }

    /// `TurnDelta.empty()`.
    pub fn empty() -> DeltaProjection {
        DeltaProjection { our_prev_active: Some("NULL".into()), opp_prev_active: Some("NULL".into()), ..Default::default() }
    }

    /// `TurnDelta.build_from_events(prev, curr, action, events)`, the projected fields.
    pub fn build(prev: &Ctx, curr: &Ctx, events: &[&Reading]) -> DeltaProjection {
        let typed = |mid: Option<&str>| -> Option<String> {
            match (mid, prev.own_hp_typed_id.as_deref()) {
                (Some("hiddenpower"), Some(t)) if !t.is_empty() => Some(t.to_string()),
                (m, _) => m.map(str::to_string),
            }
        };
        let (mut our_d, mut opp_d) = fold_hp_deltas(events, curr, prev);
        let forced = curr.forced_switch;
        let decided_forced = prev.forced_switch;
        if events.is_empty() {
            for i in 0..6 {
                our_d[i] = curr.our_hp[i] - prev.our_hp[i];
                opp_d[i] = curr.opp_hp[i] - prev.opp_hp[i];
            }
            for (i, sp) in curr.opp_slots.0.iter().enumerate() {
                if prev.opp_slots.get(sp).is_none() && opp_d[i] > 0.0 {
                    opp_d[i] = 0.0;
                }
            }
            return DeltaProjection {
                our_prev_active: Some(prev.our_active.clone()),
                opp_prev_active: Some(prev.opp_active.clone()),
                our_hp_delta: our_d,
                opp_hp_delta: opp_d,
                we_fainted: curr.our_fainted > prev.our_fainted,
                opp_fainted: curr.opp_fainted > prev.opp_fainted,
                phase_is_forced_switch: forced,
                decision_was_forced_switch: decided_forced,
                ..Default::default()
            };
        }
        let tv = TurnView::from_events(events);
        let (our, opp) = (&tv.ours, &tv.opp);
        let faints = tv.faint_details();
        let our_damaging_event = our.damaging_move.clone();
        let opp_damaging_event = opp.damaging_move.clone();
        let target_delta = |ev: &Option<DamagingMove>, deltas: &[f32; 6], slots: &Slots| -> Option<f32> {
            let t = ev.as_ref()?.target_species.as_deref().unwrap_or("");
            slots.get(t).map(|i| deltas[i])
        };
        DeltaProjection {
            our_move_id: typed(our.move_id.as_deref()),
            our_switch_to: if our.switched { our.switched_to.clone() } else { None },
            our_prev_active: Some(prev.our_active.clone()),
            opp_move_id: opp.move_id.clone(),
            opp_switch_to: if opp.switched { opp.switched_to.clone() } else { None },
            opp_prev_active: Some(prev.opp_active.clone()),
            opp_target_hp_delta: target_delta(&our_damaging_event, &opp_d, &curr.opp_slots),
            our_hp_delta: our_d,
            opp_hp_delta: opp_d,
            we_fainted: faints.iter().any(|f| f.side == Rel::Ours),
            opp_fainted: faints.iter().any(|f| f.side == Rel::Opp),
            our_failed_to_move: our.cant_reason.is_some(),
            our_move_outcome: our.outcome(),
            our_status_applied: our.status_applied.clone().filter(|s| status_known(s)),
            our_status_cured: our.status_cured.clone().filter(|s| status_known(s)),
            opp_status_applied: opp.status_applied.clone().filter(|s| status_known(s)),
            our_damaging_event,
            opp_damaging_event,
            phase_is_forced_switch: forced,
            decision_was_forced_switch: decided_forced,
        }
    }

    pub fn json_into(&self, out: &mut String) {
        let o = |out: &mut String, k: &str, v: Option<&str>| {
            out.push_str(&format!(",\"{k}\":"));
            json_out::opt_str_into(out, v);
        };
        out.push_str("{\"v\":1");
        o(out, "our_move_id", self.our_move_id.as_deref());
        o(out, "our_switch_to", self.our_switch_to.as_deref());
        o(out, "our_prev_active", self.our_prev_active.as_deref());
        o(out, "opp_move_id", self.opp_move_id.as_deref());
        o(out, "opp_switch_to", self.opp_switch_to.as_deref());
        o(out, "opp_prev_active", self.opp_prev_active.as_deref());
        o(out, "opp_resolved_move_id", self.opp_resolved_move_id());
        o(out, "our_move_outcome", self.our_move_outcome);
        o(out, "our_status_applied", self.our_status_applied.as_deref());
        o(out, "our_status_cured", self.our_status_cured.as_deref());
        o(out, "opp_status_applied", self.opp_status_applied.as_deref());
        out.push_str(&format!(
            ",\"we_fainted\":{},\"opp_fainted\":{},\"our_failed_to_move\":{},\"phase_is_forced_switch\":{},\
             \"decision_was_forced_switch\":{},\"our_damaging\":{},\"opp_damaging\":{},\"opp_target_hp_delta\":",
            self.we_fainted,
            self.opp_fainted,
            self.our_failed_to_move,
            self.phase_is_forced_switch,
            self.decision_was_forced_switch,
            self.our_damaging_event.is_some(),
            self.opp_damaging_event.is_some()
        ));
        match self.opp_target_hp_delta {
            Some(x) => json_out::f64_into(out, x as f64),
            None => out.push_str("null"),
        }
        for (k, a) in [("our_hp_delta", &self.our_hp_delta), ("opp_hp_delta", &self.opp_hp_delta)] {
            out.push_str(&format!(",\"{k}\":["));
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                json_out::f64_into(out, *x as f64);
            }
            out.push(']');
        }
        out.push('}');
    }
}

/// `Status.__members__.get(s.upper())` — a word outside the enum reads as `None`.
fn status_known(s: &str) -> bool {
    matches!(s.to_ascii_uppercase().as_str(), "BRN" | "FNT" | "FRZ" | "PAR" | "PSN" | "SLP" | "TOX")
}

/// `turn_delta._fold_hp_deltas` — per-slot END HP from the last `hp_after` (FAINT pins 0), minus the
/// previous HP ONCE (float32, bit-identical to the snapshot diff); a mon first revealed this window
/// reads 0.
fn fold_hp_deltas(events: &[&Reading], curr: &Ctx, prev: &Ctx) -> ([f32; 6], [f32; 6]) {
    let mut our_end = prev.our_hp;
    let mut opp_end = prev.opp_hp;
    for e in events {
        let after = match e.kind {
            K::Damage | K::Heal => ev::num(e, "hp_after"),
            K::Sethp => ev::num(e, "hp"),
            _ => continue,
        };
        let Some(after) = after else { continue };
        let Some(a) = e.actor.as_deref() else { continue };
        match e.side {
            Some(Rel::Ours) => {
                if let Some(i) = curr.our_slots.get(a) {
                    our_end[i] = after as f32;
                }
            }
            Some(Rel::Opp) => {
                if let Some(i) = curr.opp_slots.get(a) {
                    opp_end[i] = after as f32;
                }
            }
            None => {}
        }
    }
    for e in events {
        if e.kind != K::Faint {
            continue;
        }
        let (Some(a), Some(side)) = (ev::actor(e), e.side) else { continue };
        match side {
            Rel::Ours => {
                if let Some(i) = curr.our_slots.get(a) {
                    our_end[i] = 0.0;
                }
            }
            Rel::Opp => {
                if let Some(i) = curr.opp_slots.get(a) {
                    opp_end[i] = 0.0;
                }
            }
        }
    }
    let mut our = [0f32; 6];
    let mut opp = [0f32; 6];
    for i in 0..6 {
        our[i] = our_end[i] - prev.our_hp[i];
        opp[i] = opp_end[i] - prev.opp_hp[i];
    }
    for (i, sp) in curr.opp_slots.0.iter().enumerate() {
        if prev.opp_slots.get(sp).is_none() {
            opp[i] = 0.0;
        }
    }
    (our, opp)
}
