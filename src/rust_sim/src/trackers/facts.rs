//! The OBS-FACTS fold (`gen3_obs_facts_v1`) — `agents/training/obs_facts_fold.py`, line for line.
//!
//! Owned by [`super::history::EventWindow`] and fed the same seq-deduplicated readings: per side, the
//! active's STINT (the moves it freely selected since it entered — the opponent's is the Choice-lock
//! evidence) and, at each Encore / Disable `-start`, whether the target's side had already ACTED that
//! turn (Showdown's `!willMove` +1). The encoded block is byte-gated by slice O.

use super::ev;
use crate::core_events::{EventKind as K, Reading, Rel};

/// The volatiles whose gen-3 duration Showdown adjusts by the target's `willMove` at `-start`.
pub const ADJUSTED_EFFECTS: [&str; 2] = ["encore", "disable"];

/// `obs_facts_fold.effect_key`: lower-cased, trimmed, a `move:` prefix dropped, alphanumerics only.
pub fn effect_key(effect: Option<&str>) -> String {
    let s = effect.unwrap_or("").trim().to_lowercase();
    let s = s.strip_prefix("move:").unwrap_or(&s);
    s.chars().filter(|c| c.is_alphanumeric()).collect()
}

fn ri(r: Rel) -> usize {
    match r {
        Rel::Ours => 0,
        Rel::Opp => 1,
    }
}

/// `_Stint`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Stint {
    pub first: Option<String>,
    pub distinct2: bool,
    pub run_move: Option<String>,
    pub run_len: u32,
}

impl Stint {
    fn use_move(&mut self, mv: &str) {
        match &self.first {
            None => self.first = Some(mv.to_string()),
            Some(f) if f != mv => self.distinct2 = true,
            _ => {}
        }
        if self.run_move.as_deref() == Some(mv) {
            self.run_len += 1;
        } else {
            self.run_move = Some(mv.to_string());
            self.run_len = 1;
        }
    }
}

/// `ObsFactsFold`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FactsFold {
    act_turn: Option<i64>,
    acted: [bool; 2],
    pub stint: [Stint; 2],
    /// `(side, effect index in ADJUSTED_EFFECTS)` → the start's adjustment.
    adj: [[Option<bool>; 2]; 2],
}

impl FactsFold {
    /// `observe(e, et)` — ONE reading, already seq-deduplicated by the caller.
    pub fn observe(&mut self, e: &Reading, et: i64) {
        if self.act_turn != Some(et) {
            self.act_turn = Some(et);
            self.acted = [false, false];
        }
        let side = e.side;
        match e.kind {
            K::Move => {
                let Some(s) = side else { return };
                self.acted[ri(s)] = true;
                if let Some(mid) = ev::nz(ev::move_id(e)) {
                    if ev::nz(ev::s(e, "from_move")).is_none() && mid != "struggle" {
                        self.stint[ri(s)].use_move(mid);
                    }
                }
            }
            K::Cant => {
                if let Some(s) = ev::blocked_side(e).or(side) {
                    self.acted[ri(s)] = true;
                }
            }
            K::Damage => {
                if let Some(s) = side {
                    if ev::from_clause(e).unwrap_or("").trim().to_lowercase() == "confusion" {
                        self.acted[ri(s)] = true;
                    }
                }
            }
            K::Switch | K::Drag | K::Faint => {
                if let Some(s) = side {
                    self.stint[ri(s)] = Stint::default();
                }
            }
            K::VolatileStart => {
                if let Some(s) = side {
                    if ev::s(e, "op") == Some("start") {
                        let key = effect_key(ev::effect(e));
                        if let Some(j) = ADJUSTED_EFFECTS.iter().position(|x| *x == key) {
                            self.adj[ri(s)][j] = Some(self.acted[ri(s)]);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// `stint(side)` → `(first move id, a second distinct move was used, trailing run length)`.
    pub fn stint(&self, side: Rel) -> (Option<&str>, bool, u32) {
        let st = &self.stint[ri(side)];
        (st.first.as_deref(), st.distinct2, st.run_len)
    }

    /// `adjusted(side, effect)` — `None` when no start of it was folded (or the effect is not one
    /// Showdown adjusts).
    pub fn adjusted(&self, side: Rel, effect: &str) -> Option<bool> {
        ADJUSTED_EFFECTS.iter().position(|x| *x == effect).and_then(|j| self.adj[ri(side)][j])
    }
}
