//! Family `margin` — `win_margin` (`Gen3Env._merge_training_keys` ← `Gen3RewardManager.
//! _last_material_margin` ← `agents/training/material_margin.material_margin(live)`), rule for rule.
//!
//! * The Python env computes it in `calc_reward` on the trainee's `LiveView` every step, and the
//!   reward manager's reset sets it to 0.0 — so the RESET observation (the episode's first
//!   decision, `dec_n` 0) reads 0.0, and every later decision reads the margin of the board the
//!   observation describes.
//! * `margin = clamp(Φ, ±B) / B`, `Φ = 2·(Σ our_hp − Σ opp_hp) + 1.25·(n_alive_ours − n_alive_opp)`,
//!   `B = (2 + 1.25)·6`, over the first 6 mons of each side of the side's own view (`present()`);
//!   the opponent's unrevealed declared slots (`team_size`, 0 ⇒ 6, capped at 6) count full-HP-alive;
//!   an empty own team ⇒ 0.0. f64 throughout, in the Python summation order, stored as f32.

use pokesim::encoder::layout::TEAM_SIZE;
use pokesim::present::view::OneSidedView;

/// `MAT_HP_WEIGHT`, `MAT_ALIVE_WEIGHT` (`material_margin.py`).
const HP_W: f64 = 2.0;
const ALIVE_W: f64 = 1.25;

pub fn material_margin(live: &OneSidedView) -> f64 {
    let our = &live.ours.mons;
    if our.is_empty() {
        return 0.0;
    }
    let ours = &our[..our.len().min(TEAM_SIZE)];
    let our_hp: f64 = ours.iter().filter(|m| !m.fainted).fold(0.0, |a, m| a + m.hp_fraction);
    let our_alive = ours.iter().filter(|m| !m.fainted).count() as f64;
    let size = if live.opp.team_size == 0 { TEAM_SIZE } else { live.opp.team_size }.min(TEAM_SIZE);
    let opp = &live.opp.mons[..live.opp.mons.len().min(TEAM_SIZE)];
    let mut opp_hp: f64 = opp.iter().filter(|m| !m.fainted).fold(0.0, |a, m| a + m.hp_fraction);
    let mut opp_alive = opp.iter().filter(|m| !m.fainted).count() as f64;
    let unrevealed = size.saturating_sub(opp.len()) as f64;
    opp_hp += unrevealed * 1.0;
    opp_alive += unrevealed;
    let phi = HP_W * (our_hp - opp_hp) + ALIVE_W * (our_alive - opp_alive);
    let bound = HP_W * TEAM_SIZE as f64 + ALIVE_W * TEAM_SIZE as f64;
    phi.min(bound).max(-bound) / bound
}

pub fn write(view: &OneSidedView, dec_n: u32, out: &mut [f32]) {
    out[0] = if dec_n == 0 { 0.0 } else { material_margin(view) as f32 };
}
