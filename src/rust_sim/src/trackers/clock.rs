//! `agents.training.progress_clock.ProgressClock` — the episode-scoped `turns_since_progress`
//! counter whose `value()` is an OBS scalar. Ported for `n` and every piece of state `n` depends
//! on; the reward half (`last_penalty`, the no-progress TAX, a shaped term) is NOT ported (program
//! M3: the core's reward is the win indicator alone), so neither is `switch_legal`, whose only
//! reader is that charge.

use std::collections::BTreeSet;

use super::delta::DeltaProjection;
use crate::core_events::json_out;
use crate::dex::Dex;
use crate::present::{LegalActions, OneSidedView};

/// `PROGRESS_DMG_EPS`.
pub const PROGRESS_DMG_EPS: f64 = 0.03;
/// `PROGRESS_CLOCK_CAP`.
pub const PROGRESS_CLOCK_CAP: i64 = 10;
/// `HEAL_FREEZE_GRACE`.
const HEAL_FREEZE_GRACE: i64 = 2;
/// `gen3_mechanics.INVULNERABLE_MOVES`.
const INVULNERABLE: [&str; 3] = ["protect", "detect", "endure"];

/// The clock's two opt-in intent-restoring fixes (`--progress-decision-tense`,
/// `--progress-switch-freeze`), both OFF in `designs/production_config.json`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ClockConfig {
    pub decision_tense: bool,
    pub switch_freeze: bool,
}

/// `ProgressClock` (the obs half).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProgressClock {
    pub cfg: ClockConfig,
    pub n: i64,
    prev_spikes: i64,
    prev_our_spikes: i64,
    prev_our_boost_sum: i64,
    prev_our_has_sub: bool,
    heal_streak: i64,
    rested_species: BTreeSet<String>,
    is_rest_loop: bool,
}

fn spikes(side: &crate::present::view::SideView) -> i64 {
    side.side_conditions.iter().find(|(k, _)| k == "spikes").map_or(0, |(_, v)| *v as i64)
}

fn our_active(live: &OneSidedView) -> Option<&crate::present::view::MonView> {
    live.ours.active.map(|i| &live.ours.mons[i])
}

fn our_boost_sum(live: &OneSidedView) -> i64 {
    our_active(live).map_or(0, |m| m.boosts.iter().filter(|(_, v)| *v > 0).map(|(_, v)| *v as i64).sum())
}

fn our_has_sub(live: &OneSidedView) -> bool {
    our_active(live).is_some_and(|m| m.volatiles.iter().any(|(k, _)| k == "substitute"))
}

/// `_winning_residual`.
fn winning_residual(d: &DeltaProjection, live: &OneSidedView) -> bool {
    let Some(i) = live.opp.active else { return false };
    let m = &live.opp.mons[i];
    let vol = |k: &str| m.volatiles.iter().any(|(v, _)| v == k);
    if !(matches!(m.status, Some("tox" | "psn" | "brn")) || vol("leechseed") || vol("curse") || vol("nightmare")) {
        return false;
    }
    (d.opp_hp_sum() as f64) <= -PROGRESS_DMG_EPS
}

impl ProgressClock {
    pub fn new(cfg: ClockConfig) -> ProgressClock {
        ProgressClock { cfg, ..Default::default() }
    }

    /// `value()` — the obs scalar.
    pub fn value(&self) -> f64 {
        (1.0 + self.n.min(PROGRESS_CLOCK_CAP) as f64).ln() / (1.0 + PROGRESS_CLOCK_CAP as f64).ln()
    }

    /// `_gates` — only `forced_window` survives (`switch_legal` feeds the unported tax).
    fn forced_window(&self, d: &DeltaProjection) -> bool {
        if self.cfg.decision_tense {
            d.decision_was_forced_switch
        } else {
            d.phase_is_forced_switch
        }
    }

    fn update_rest_loop(&mut self, d: &DeltaProjection, live: &OneSidedView) {
        self.is_rest_loop = false;
        if d.our_move_id.as_deref() != Some("rest") || d.our_failed_to_move {
            return;
        }
        if !d.our_status_applied.as_deref().is_some_and(|s| s.eq_ignore_ascii_case("slp")) {
            return;
        }
        let Some(sp) = d.our_prev_active.as_deref().filter(|s| !s.is_empty()) else { return };
        let has_sleeptalk = our_active(live).is_some_and(|m| m.moves.iter().any(|mv| mv.id == "sleeptalk"));
        if self.rested_species.contains(sp) && !has_sleeptalk {
            self.is_rest_loop = true;
        }
        self.rested_species.insert(sp.to_string());
    }

    fn is_wasted_self_cure(d: &DeltaProjection, dex: &Dex) -> bool {
        let Some(mid) = d.our_move_id.as_deref() else { return false };
        if d.our_failed_to_move || d.our_status_cured.is_some() {
            return false;
        }
        dex.moves(mid).is_some_and(|m| m.cures_self_status)
    }

    #[allow(clippy::too_many_arguments)]
    fn is_progress(d: &DeltaProjection, live: &OneSidedView, prev_spikes: i64, opp_spikes_now: i64,
                   prev_boost: i64, boost_now: i64, prev_sub: bool, sub_now: bool) -> bool {
        if d.our_damaging_event.is_some() {
            if let Some(t) = d.opp_target_hp_delta {
                if (t as f64) <= -PROGRESS_DMG_EPS {
                    return true;
                }
            }
        }
        if d.opp_status_applied.is_some() || opp_spikes_now - prev_spikes > 0 || d.opp_switch_to.is_some() {
            return true;
        }
        if winning_residual(d, live) || boost_now > prev_boost || (sub_now && !prev_sub) {
            return true;
        }
        d.our_move_id.as_deref() == Some("wish") && !d.our_failed_to_move && d.our_move_outcome != Some("fail")
    }

    fn denial_kind(d: &DeltaProjection, dex: &Dex) -> Option<&'static str> {
        if d.our_failed_to_move {
            return Some("exogenous");
        }
        let outcome = d.our_move_outcome;
        if outcome == Some("miss") {
            return Some("exogenous");
        }
        if outcome == Some("fail") {
            let opp = d.opp_resolved_move_id();
            let ours = d.our_move_id.as_deref();
            if opp.is_some_and(|m| INVULNERABLE.contains(&m)) && !ours.is_some_and(|m| INVULNERABLE.contains(&m)) {
                return Some("exogenous");
            }
        }
        if let Some(mid) = d.our_move_id.as_deref() {
            if (d.our_hp_sum() as f64) > PROGRESS_DMG_EPS && dex.moves(mid).is_some_and(|m| m.is_heal) {
                return Some("heal");
            }
        }
        None
    }

    /// `update(delta, live, legal, legal_prev)` — `n` only (see the module docs).
    pub fn update(&mut self, d: &DeltaProjection, live: &OneSidedView, _legal: Option<&LegalActions>, dex: &Dex) {
        let forced = self.forced_window(d);
        let opp_now = spikes(&live.opp);
        let prev_spikes = std::mem::replace(&mut self.prev_spikes, opp_now);
        let our_now = spikes(&live.ours);
        let prev_our_spikes = std::mem::replace(&mut self.prev_our_spikes, our_now);
        let boost_now = our_boost_sum(live);
        let prev_boost = std::mem::replace(&mut self.prev_our_boost_sum, boost_now);
        let sub_now = our_has_sub(live);
        let prev_sub = std::mem::replace(&mut self.prev_our_has_sub, sub_now);
        self.update_rest_loop(d, live);
        if forced {
            return;
        }
        let bump = |n: i64| (n + 1).min(PROGRESS_CLOCK_CAP);
        if d.our_move_id.as_deref() == Some("spikes")
            && opp_now >= 3
            && opp_now - prev_spikes <= 0
            && !winning_residual(d, live)
        {
            self.n = bump(self.n);
            self.heal_streak = 0;
            return;
        }
        if Self::is_wasted_self_cure(d, dex) && !winning_residual(d, live) {
            self.n = bump(self.n);
            self.heal_streak = 0;
            return;
        }
        let filler_spin = d.our_move_id.as_deref() == Some("rapidspin") && prev_our_spikes == 0 && !d.opp_fainted;
        if !filler_spin && Self::is_progress(d, live, prev_spikes, opp_now, prev_boost, boost_now, prev_sub, sub_now) {
            self.n = 0;
            self.heal_streak = 0;
            return;
        }
        match Self::denial_kind(d, dex) {
            Some("exogenous") => return,
            Some("heal") => {
                self.heal_streak += 1;
                if !self.is_rest_loop && self.heal_streak <= HEAL_FREEZE_GRACE {
                    return;
                }
            }
            _ => self.heal_streak = 0,
        }
        if self.cfg.switch_freeze && d.our_switch_to.is_some() {
            return;
        }
        self.n = bump(self.n);
    }

    pub fn json_into(&self, out: &mut String) {
        out.push_str(&format!(
            "{{\"n\":{},\"prev_spikes\":{},\"prev_our_spikes\":{},\"prev_our_boost_sum\":{},\"prev_our_has_sub\":{},\
             \"heal_streak\":{},\"is_rest_loop\":{},\"rested_species\":[",
            self.n, self.prev_spikes, self.prev_our_spikes, self.prev_our_boost_sum, self.prev_our_has_sub,
            self.heal_streak, self.is_rest_loop
        ));
        for (i, s) in self.rested_species.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            json_out::str_into(out, s);
        }
        out.push_str("]}");
    }
}
