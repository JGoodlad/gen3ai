"""The PROGRESS and HAZARD PBRS potentials (design_markovian_reward_and_features.md §5):
Φ_progress (the anti-stall clock potential) and Φ_hazard (Spikes), both gated behind
--all-shaping-pbrs. Pure (no battle sim) — the shared fakes live in `reward_test_fakes.py`.
"""
import unittest

from agents.training.reward_manager import (
    Gen3RewardManager, RewardConfig, PBRS_GAMMA, HAZARD_WEIGHT,
)
from agents.training.progress_clock import ProgressClock
from agents.training.reward_test_fakes import (
    _mgr_additive_bias, _mgr_pbrs, _Live, _Battle, _delta, _full_team_live,
)


class TestProgressPBRS(unittest.TestCase):
    """Φ_progress (design §5): the anti-stall PBRS potential. Φ = −W·clock.value(); telescoping,
    terminal-zeroing; gated on --stall-pbrs (OFF = byte-identical default; --all-shaping-pbrs alone
    keeps the no_progress_tax tilt instead). The _mgr_pbrs helper turns BOTH switches on."""

    def test_phi_progress_zero_at_start(self):
        """n=0 at episode start → Φ_progress ≈ 0."""
        m = _mgr_pbrs()
        self.assertAlmostEqual(m._compute_phi_progress(None), 0.0, places=6)

    def test_phi_progress_negative_when_stalling(self):
        """n>0 → Φ_progress < 0 (FIRST guard against a sign flip rewarding stalling)."""
        m = _mgr_pbrs()
        m.progress_clock.n = 5
        self.assertLess(m._compute_phi_progress(None), 0.0)

    def test_phi_progress_none_clock_returns_zero(self):
        """Inference / standalone (no clock) → Φ_progress = 0 (no NoneType deref)."""
        m = Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=True))  # no clock
        self.assertEqual(m._compute_phi_progress(None), 0.0)

    def test_phi_progress_scales_with_no_progress_penalty(self):
        """Doubling no_progress_penalty doubles |Φ_progress| (it is the weight)."""
        m1 = Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=True, no_progress_penalty=0.15),
                               progress_clock=ProgressClock())
        m2 = Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=True, no_progress_penalty=0.30),
                               progress_clock=ProgressClock())
        m1.progress_clock.n = m2.progress_clock.n = 4
        self.assertAlmostEqual(m2._compute_phi_progress(None), 2.0 * m1._compute_phi_progress(None),
                               places=6)

    def test_fold_gated_off_by_default(self):
        """Default run (all_shaping_pbrs OFF): pbrs_progress ≡ 0, _prev_phi_progress stays None even
        with the clock charged."""
        m = Gen3RewardManager(progress_clock=ProgressClock())   # default: all_shaping_pbrs False
        m.progress_clock.n = 5
        m.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())
        m.progress_clock.n = 6
        m.process_turn_reward(_Battle(_full_team_live(), turn=2), _delta())
        self.assertEqual(m._last_breakdown.pbrs_progress, 0.0)
        self.assertIsNone(m._prev_phi_progress)

    def test_fold_active_on_stall(self):
        """Under the flag, an increasing clock yields a negative pbrs_progress (Φ falls further)."""
        m = _mgr_pbrs()
        m.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())   # prev = Φ(n=0) = 0
        m.progress_clock.n = 4                                                # stalled
        m.process_turn_reward(_Battle(_full_team_live(), turn=2), _delta())
        self.assertLess(m._last_breakdown.pbrs_progress, 0.0)

    def test_terminal_zeroes_phi(self):
        """A terminal window zeroes Φ → shaped = −prev (small), NOT a stall bonus."""
        m = _mgr_pbrs()
        m.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())
        m.progress_clock.n = 5
        m.process_turn_reward(_Battle(_full_team_live(), turn=2), _delta())   # set prev<0
        prev = m._prev_phi_progress
        self.assertLess(prev, 0.0)
        win_live = _Live([1.0] * 6, [0.0] * 6, won=True, finished=True)
        m.process_turn_reward(_Battle(win_live, turn=3), _delta(opp_fainted=True))
        self.assertAlmostEqual(m._last_breakdown.pbrs_progress, -prev, places=6)  # γ·0 − prev

    def test_telescopes_to_zero_net(self):
        """Clock up then reset to 0 then terminal → Σ pbrs_progress ≈ 0 (Φ(s_0)=0, policy-invariant)."""
        m = _mgr_pbrs()
        ns = [0, 3, 6, 0, 0]   # the clock trajectory; last window terminal
        total = 0.0
        for i, n in enumerate(ns):
            m.progress_clock.n = n
            term = (i == len(ns) - 1)
            live = _Live([1.0] * 6, [0.0] * 6, won=True, finished=True) if term else _full_team_live()
            m.process_turn_reward(_Battle(live, turn=i + 1), _delta())
            total += m._last_breakdown.pbrs_progress
        resid = abs(PBRS_GAMMA - 1.0) * len(ns) * abs(RewardConfig().no_progress_penalty) + 1e-6
        self.assertAlmostEqual(total, 0.0, delta=resid)


class TestHazardPBRS(unittest.TestCase):
    """Φ_hazard (design §2.6): the spikes/hazard potential. Telescoping, terminal zeroing,
    futile-waste dissolution."""

    def test_phi_hazard_zero_at_start(self):
        m = Gen3RewardManager()
        self.assertAlmostEqual(m._compute_phi_hazard(_full_team_live()), 0.0, places=6)

    def test_phi_opp_layer_raises(self):
        """Opp gains a spikes layer (0→1) → Φ_hazard increases by HAZARD_WEIGHT."""
        m = Gen3RewardManager()
        live1 = _full_team_live(); live1.opp.side_conditions = {"spikes": 1}
        self.assertAlmostEqual(m._compute_phi_hazard(live1) - m._compute_phi_hazard(_full_team_live()),
                               HAZARD_WEIGHT, places=6)

    def test_phi_our_layer_lowers(self):
        """We gain a spikes layer (our side) → Φ_hazard decreases by HAZARD_WEIGHT (symmetric)."""
        m = Gen3RewardManager()
        live1 = _full_team_live(); live1.ours.side_conditions = {"spikes": 1}
        self.assertAlmostEqual(m._compute_phi_hazard(_full_team_live()) - m._compute_phi_hazard(live1),
                               HAZARD_WEIGHT, places=6)

    def test_phi_symmetry(self):
        """opp 3, us 1 → Φ_hazard = HAZARD_WEIGHT·(3−1) = +1.0."""
        m = Gen3RewardManager()
        live = _full_team_live()
        live.opp.side_conditions = {"spikes": 3}
        live.ours.side_conditions = {"spikes": 1}
        self.assertAlmostEqual(m._compute_phi_hazard(live), HAZARD_WEIGHT * (3 - 1), places=6)

    def test_fold_gated_off_under_no_all_shaping_pbrs(self):
        """`--no-all-shaping-pbrs`: pbrs_hazard ≡ 0, _prev_phi_hazard stays None (the spikes /
        futile_spikes BIAS terms carry hazard value there). DEFAULT-ON since 2026-08-18."""
        m = _mgr_additive_bias()
        m.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())
        live2 = _full_team_live(); live2.opp.side_conditions = {"spikes": 1}
        m.process_turn_reward(_Battle(live2, turn=2), _delta(our_move_id="spikes"))
        self.assertEqual(m._last_breakdown.pbrs_hazard, 0.0)
        self.assertIsNone(m._prev_phi_hazard)

    def test_fold_layer_added_then_cleared(self):
        """Under the flag: opp gaining a layer RAISES Φ (+pbrs_hazard); a held layer ≈0; clearing
        (Rapid Spin) DROPS Φ (−pbrs_hazard)."""
        m = _mgr_pbrs()
        m.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())   # prev = 0
        layer = _full_team_live(); layer.opp.side_conditions = {"spikes": 1}
        m.process_turn_reward(_Battle(layer, turn=2), _delta(our_move_id="spikes"))
        self.assertAlmostEqual(m._last_breakdown.pbrs_hazard, PBRS_GAMMA * HAZARD_WEIGHT, places=6)
        held = _full_team_live(); held.opp.side_conditions = {"spikes": 1}
        m.process_turn_reward(_Battle(held, turn=3), _delta())
        self.assertLess(abs(m._last_breakdown.pbrs_hazard), 1e-3)              # ≈0 (γ−1)·w
        m.process_turn_reward(_Battle(_full_team_live(), turn=4), _delta())    # cleared
        self.assertLess(m._last_breakdown.pbrs_hazard, 0.0)

    def test_futile_spikes_dissolves(self):
        """Spikes at the 3-layer cap → ΔΦ_hazard ≈ 0 AND bd.futile_spikes == 0 under suppression."""
        m = _mgr_pbrs()
        capped = _full_team_live(); capped.opp.side_conditions = {"spikes": 3}
        m._prev_opp_spikes = 3   # already at cap before this window
        m.process_turn_reward(_Battle(capped, turn=1), _delta())   # prev φ set
        m.process_turn_reward(_Battle(capped, turn=2), _delta(our_move_id="spikes"))
        self.assertLess(abs(m._last_breakdown.pbrs_hazard), 1e-3)
        self.assertEqual(m._last_breakdown.futile_spikes, 0.0)     # dissolved under suppression

    def test_telescopes_to_zero_net(self):
        """Hazard up then cleared then terminal → Σ pbrs_hazard ≈ 0 (Φ(s_0)=0, policy-invariant)."""
        m = _mgr_pbrs()
        s1 = _full_team_live()
        s2 = _full_team_live(); s2.opp.side_conditions = {"spikes": 1}
        s3 = _full_team_live(); s3.opp.side_conditions = {"spikes": 2}
        s4 = _full_team_live(); s4.opp.side_conditions = {"spikes": 3}
        s5 = _full_team_live()   # cleared
        s6 = _Live([1.0] * 6, [0.0] * 6, won=True, finished=True)
        total = 0.0
        for i, live in enumerate((s1, s2, s3, s4, s5, s6)):
            m.process_turn_reward(_Battle(live, turn=i + 1), _delta())
            total += m._last_breakdown.pbrs_hazard
        resid = abs(PBRS_GAMMA - 1.0) * 6 * HAZARD_WEIGHT + 1e-6
        self.assertAlmostEqual(total, 0.0, delta=resid)


if __name__ == "__main__":
    unittest.main()
