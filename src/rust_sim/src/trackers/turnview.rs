//! `agents.battle.turn_view.TurnView` over a window of [`Reading`]s — the per-side "what happened"
//! fold the Python consumers (the progress clock, the α/β intent label, the Hidden-Power belief's
//! damaging-move capture) read through `TurnDelta.build_from_events`.
//!
//! This is the FROZEN, LOSSY reading (one move per side per window, the LAST one; one
//! effectiveness; one status transition): it exists so the core reproduces those consumers byte
//! for byte. What actually happened, in order and with attribution, is the native record
//! ([`super::record`]); the catalogue of what this fold loses is
//! `designs/research_state/measurements/rust_core_m3_2026-09-24/`.

use super::ev;
use crate::core_events::{EventKind as K, Reading, Rel};

/// `DamagingMove`.
#[derive(Debug, Clone, PartialEq)]
pub struct DamagingMove {
    pub user_species: Option<String>,
    pub target_species: Option<String>,
    /// The `Status` NAME (`"PAR"`) the target had when the move fired.
    pub target_status: Option<String>,
    pub move_id: Option<String>,
    pub effectiveness: Option<f64>,
}

/// `SideTurn`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SideTurn {
    pub moved: bool,
    pub move_id: Option<String>,
    pub called_via: Option<String>,
    pub switched: bool,
    pub switched_to: Option<String>,
    pub drag: bool,
    pub fainted: bool,
    pub cant_reason: Option<String>,
    pub cant_move: Option<String>,
    pub crit: bool,
    pub missed: bool,
    pub failed: bool,
    pub effectiveness: Option<f64>,
    pub target_species: Option<String>,
    pub damaging_move: Option<DamagingMove>,
    pub status_applied: Option<String>,
    pub status_cured: Option<String>,
    pub item_lost: Option<String>,
    pub item_gained: Option<String>,
    pub attempted_rejected: bool,
}

impl SideTurn {
    /// `"hit"` / `"miss"` / `"fail"`, or `None` when no move resolved.
    pub fn outcome(&self) -> Option<&'static str> {
        if !self.moved {
            None
        } else if self.missed {
            Some("miss")
        } else if self.failed {
            Some("fail")
        } else {
            Some("hit")
        }
    }
}

/// `FaintDetail`.
#[derive(Debug, Clone, PartialEq)]
pub struct FaintDetail {
    pub species: String,
    pub side: Rel,
    pub cause: &'static str,
}

/// `FAINT_CAUSE_VOCAB`.
pub const FAINT_CAUSE_VOCAB: [&str; 8] = ["attack", "hazard", "weather", "status", "recoil", "selfko", "leechseed", "other"];

const SELF_KO_MOVES: [&str; 2] = ["explosion", "selfdestruct"];

/// `turn_view._classify_faint_cause`.
pub fn classify_faint_cause(from_clause: Option<&str>, used_selfko: bool) -> &'static str {
    if used_selfko {
        return "selfko";
    }
    let Some(fc) = from_clause else { return "attack" };
    let fc = fc.trim().to_lowercase();
    if fc == "spikes" {
        "hazard"
    } else if fc == "sandstorm" || fc == "hail" {
        "weather"
    } else if matches!(fc.as_str(), "psn" | "tox" | "brn" | "burn") {
        "status"
    } else if fc.contains("recoil") {
        "recoil"
    } else if fc.contains("leech seed") || fc.contains("leechseed") {
        "leechseed"
    } else {
        "other"
    }
}

/// `TurnView.from_events(events)`.
pub struct TurnView<'a> {
    pub turn: u32,
    pub events: &'a [&'a Reading],
    pub ours: SideTurn,
    pub opp: SideTurn,
    move_order: Vec<Rel>,
}

impl<'a> TurnView<'a> {
    pub fn from_events(events: &'a [&'a Reading]) -> TurnView<'a> {
        let turn = events.first().map_or(0, |e| e.turn);
        let mut tv = TurnView { turn, events, ours: SideTurn::default(), opp: SideTurn::default(), move_order: Vec::new() };
        tv.ours = tv.fold_side(Rel::Ours);
        tv.opp = tv.fold_side(Rel::Opp);
        for e in events {
            if e.kind == K::Move {
                if let Some(s) = e.side {
                    if !tv.move_order.contains(&s) {
                        tv.move_order.push(s);
                    }
                }
            }
        }
        tv
    }

    pub fn side(&self, s: Rel) -> &SideTurn {
        match s {
            Rel::Ours => &self.ours,
            Rel::Opp => &self.opp,
        }
    }

    fn fold_side(&self, side: Rel) -> SideTurn {
        let mut st = SideTurn::default();
        let side_events: Vec<&Reading> = self.events.iter().copied().filter(|e| e.side == Some(side)).collect();
        let primary = side_events.iter().copied().filter(|e| e.kind == K::Move).last();
        if let Some(p) = primary {
            st.moved = true;
            st.move_id = ev::move_id(p).map(str::to_string);
            st.called_via = ev::s(p, "from_move").map(str::to_string);
            st.target_species = p.target.clone();
        }
        for e in &side_events {
            match e.kind {
                K::Switch => {
                    st.switched = true;
                    st.switched_to = e.actor.clone();
                }
                K::Drag => {
                    st.switched = true;
                    st.drag = true;
                    st.switched_to = e.actor.clone();
                }
                K::Faint => st.fainted = true,
                K::Cant => {
                    st.cant_reason = ev::reason(e).map(str::to_string);
                    st.cant_move = ev::cant_move(e).map(str::to_string);
                }
                K::Crit => st.crit = true,
                K::Miss => st.missed = true,
                K::Fail => {
                    if matches!(ev::from_clause(e), None | Some("move-suffix")) {
                        st.failed = true;
                    }
                }
                K::Immune | K::Resisted | K::Supereffective => st.effectiveness = ev::multiplier(e),
                K::Status => st.status_applied = ev::status(e).map(str::to_string),
                K::Curestatus => st.status_cured = ev::status(e).map(str::to_string),
                K::Enditem => st.item_lost = ev::item(e).map(str::to_string),
                K::Item => st.item_gained = ev::item(e).map(str::to_string),
                K::ChoiceRejected => st.attempted_rejected = true,
                _ => {}
            }
        }
        if let Some(p) = primary {
            let target = p.target.as_deref();
            let dealt = match target {
                Some(t) if !t.is_empty() => self.damage_on(t, Some(ev::other(side))),
                _ => 0.0,
            };
            if st.effectiveness.is_some() || dealt < 0.0 {
                st.damaging_move = Some(DamagingMove {
                    user_species: p.actor.clone(),
                    target_species: p.target.clone(),
                    target_status: ev::s(p, "target_status").map(str::to_string),
                    move_id: ev::move_id(p).map(str::to_string),
                    effectiveness: st.effectiveness.or(if dealt < 0.0 { Some(1.0) } else { None }),
                });
            }
        }
        st
    }

    /// `TurnView.damage_on(species, side)`.
    pub fn damage_on(&self, species: &str, side: Option<Rel>) -> f64 {
        let mut total = 0.0;
        for e in self.events {
            if e.actor.as_deref() == Some(species)
                && (side.is_none() || e.side == side)
                && matches!(e.kind, K::Damage | K::Heal | K::Sethp)
            {
                total += ev::amount(e).unwrap_or(0.0);
            }
        }
        total
    }

    /// `TurnView.we_moved_first`.
    pub fn we_moved_first(&self) -> Option<bool> {
        if self.move_order.len() < 2 {
            None
        } else {
            Some(self.move_order[0] == Rel::Ours)
        }
    }

    /// `TurnView.faint_details()`.
    pub fn faint_details(&self) -> Vec<FaintDetail> {
        let mut last: Vec<((String, Rel), Option<String>)> = Vec::new();
        for e in self.events {
            if e.kind == K::Damage {
                if let (Some(a), Some(s)) = (ev::actor(e), e.side) {
                    let key = (a.to_string(), s);
                    let val = ev::reason(e).map(str::to_string);
                    match last.iter_mut().find(|(k, _)| *k == key) {
                        Some(slot) => slot.1 = val,
                        None => last.push((key, val)),
                    }
                }
            }
        }
        let mut out = Vec::new();
        for e in self.events {
            if e.kind != K::Faint {
                continue;
            }
            let (Some(a), Some(s)) = (ev::actor(e), e.side) else { continue };
            let st = self.side(s);
            let used_selfko = st.moved && st.move_id.as_deref().is_some_and(|m| SELF_KO_MOVES.contains(&m));
            let fc = last.iter().find(|(k, _)| k.0 == a && k.1 == s).and_then(|(_, v)| v.as_deref());
            out.push(FaintDetail { species: a.to_string(), side: s, cause: classify_faint_cause(fc, used_selfko) });
        }
        out
    }
}
