"""The BOOST-family PBRS potentials and the roar term: Φ_boost (our active's stages),
Φ_opp_boosts (theirs) and the phazing reward that Φ_opp_boosts subsumes.
Pure (no battle sim) — the shared fakes live in `reward_test_fakes.py`.
"""
import unittest

import numpy as np

from agents.training.reward_manager import (
    Gen3RewardManager, RewardConfig, PBRS_GAMMA, BOOST_WEIGHT, OPP_BOOST_WEIGHT,
    ROAR_BOOST_WEIGHT,
)
from agents.training.reward_test_fakes import (
    _mgr_additive_bias, _mgr_pbrs, _Live, _Battle, _delta, _full_team_live,
)


class TestBoostPBRS(unittest.TestCase):
    """Φ_boost (design): stored-boost offensive potential; positive-only, HP-scaled, telescoping."""

    def test_phi_zero_no_boosts(self):
        m = _mgr_pbrs()
        self.assertAlmostEqual(m._compute_phi_boost(_full_team_live()), 0.0, places=6)

    def test_phi_counts_positive_only(self):
        """atk +2, spa +1, spe −1 → Σ max(0,·) = 3; full HP → Φ = BOOST_WEIGHT·3."""
        m = _mgr_pbrs()
        live = _full_team_live()
        live.ours.active.boosts = {"atk": 2, "spa": 1, "spe": -1}
        self.assertAlmostEqual(m._compute_phi_boost(live), BOOST_WEIGHT * 3 * 1.0, places=6)

    def test_phi_scales_with_hp(self):
        """Half-HP active → half the Φ_boost of full HP."""
        m = _mgr_pbrs()
        full = _full_team_live(); full.ours.active.boosts = {"atk": 2}
        half = _full_team_live(our_hp=0.5); half.ours.active.boosts = {"atk": 2}
        self.assertAlmostEqual(m._compute_phi_boost(half), m._compute_phi_boost(full) * 0.5, places=6)

    def test_phi_zero_at_cap_no_bonus(self):
        """Setting a boost already at +6 (capped) → Φ unchanged → ΔΦ ≈ 0."""
        m = _mgr_pbrs()
        capped = _full_team_live(); capped.ours.active.boosts = {"atk": 6}
        m.process_turn_reward(_Battle(capped, turn=1), _delta())   # prev φ at cap
        capped2 = _full_team_live(); capped2.ours.active.boosts = {"atk": 6}
        m.process_turn_reward(_Battle(capped2, turn=2), _delta(our_move_id="swordsdance"))
        self.assertAlmostEqual(m._last_breakdown.pbrs_boost, 0.0, places=3)

    def test_phi_faint_drops_potential(self):
        """A boosted active fainting (new active has no boosts) → Φ drops → −pbrs_boost."""
        m = _mgr_pbrs()
        boosted = _full_team_live(); boosted.ours.active.boosts = {"atk": 3, "spa": 2}
        m.process_turn_reward(_Battle(boosted, turn=1), _delta())
        self.assertGreater(m._prev_phi_boost, 0.0)
        fainted = _full_team_live(our_alive=5)   # mon fainted, new active boostless
        m.process_turn_reward(_Battle(fainted, turn=2), _delta(we_fainted=True))
        self.assertLess(m._last_breakdown.pbrs_boost, 0.0)

    def test_fold_gated_off_under_no_all_shaping_pbrs(self):
        """DEFAULT-ON since 2026-08-18; this pins the fallback's OFF branch."""
        m = _mgr_additive_bias()
        live = _full_team_live(); live.ours.active.boosts = {"atk": 3}
        m.process_turn_reward(_Battle(live, turn=1), _delta())
        self.assertEqual(m._last_breakdown.pbrs_boost, 0.0)
        self.assertIsNone(m._prev_phi_boost)

    def test_telescopes_to_zero_net(self):
        """Setup → held → faint → terminal: Σ pbrs_boost ≈ 0 (policy-invariant)."""
        m = _mgr_pbrs()
        s1 = _full_team_live()
        s2 = _full_team_live(); s2.ours.active.boosts = {"atk": 2, "spa": 1}
        s3 = _full_team_live(); s3.ours.active.boosts = {"atk": 2, "spa": 1}
        s4 = _full_team_live(our_alive=5)   # boosted mon fainted, new active boostless
        s5 = _Live([1.0] * 6, [0.0] * 6, won=True, finished=True)
        total = 0.0
        for i, live in enumerate((s1, s2, s3, s4, s5)):
            m.process_turn_reward(_Battle(live, turn=i + 1), _delta())
            total += m._last_breakdown.pbrs_boost
        resid = abs(PBRS_GAMMA - 1.0) * 5 * BOOST_WEIGHT * 6 * 1.0 + 1e-6
        self.assertAlmostEqual(total, 0.0, delta=resid)

    def test_boost_bias_terms_suppressed_under_flag(self):
        """Under the flag, boost_utilized / futile_setup / setup_low_hp all leave the breakdown == 0."""
        m = _mgr_pbrs()
        # A capped setup at low HP would normally charge futile_setup + setup_low_hp.
        m._our_active_hp_before = 0.2
        live = _full_team_live(our_hp=0.2); live.ours.active.boosts = {"atk": 6}
        m.process_turn_reward(_Battle(live, turn=1),
                              _delta(our_move_id="swordsdance",
                                     our_boost_delta=np.zeros(7, dtype=np.int8)))
        bd = m._last_breakdown
        self.assertEqual(bd.boost_utilized, 0.0)
        self.assertEqual(bd.futile_setup, 0.0)
        self.assertEqual(bd.setup_low_hp, 0.0)


class TestOppBoostsPBRS(unittest.TestCase):
    """Φ_opp_boosts (design): phaze-boost-disruption potential. Negative with opp boosts; a forced
    switch clearing boosts raises Φ; a failed Roar leaves them → ΔΦ=0. Telescoping, terminal zero."""

    def test_phi_zero_at_start(self):
        m = _mgr_pbrs()
        self.assertAlmostEqual(m._compute_phi_opp_boosts(_full_team_live()), 0.0, places=6)

    def test_phi_negative_with_boosts(self):
        """opp atk +3 → Φ = −OPP_BOOST_WEIGHT·3 = −0.45."""
        m = _mgr_pbrs()
        live = _full_team_live(); live.opp.active.boosts = {"atk": 3}
        self.assertAlmostEqual(m._compute_phi_opp_boosts(live), -OPP_BOOST_WEIGHT * 3, places=6)

    def test_phi_sums_positive(self):
        """All positive stages sum: atk+2, spe+1, spa+3 → −OPP_BOOST_WEIGHT·6."""
        m = _mgr_pbrs()
        live = _full_team_live(); live.opp.active.boosts = {"atk": 2, "spe": 1, "spa": 3}
        self.assertAlmostEqual(m._compute_phi_opp_boosts(live), -OPP_BOOST_WEIGHT * 6, places=6)

    def test_phi_ignores_negative(self):
        """A negative opp boost (def −2) does not contribute."""
        m = _mgr_pbrs()
        live = _full_team_live(); live.opp.active.boosts = {"atk": 2, "def": -2}
        self.assertAlmostEqual(m._compute_phi_opp_boosts(live), -OPP_BOOST_WEIGHT * 2, places=6)

    def test_fold_disabled_under_no_all_shaping_pbrs(self):
        """DEFAULT-ON since 2026-08-18; this pins the fallback's OFF branch."""
        m = _mgr_additive_bias()
        live = _full_team_live(); live.opp.active.boosts = {"atk": 3}
        m.process_turn_reward(_Battle(live, turn=1), _delta())
        self.assertEqual(m._last_breakdown.pbrs_opp_boosts, 0.0)
        self.assertIsNone(m._prev_phi_opp_boosts)

    def test_roar_forces_switch_clears_boosts(self):
        """opp +3, then Roar forces a switch (new active boostless) → Φ rises → +pbrs_opp_boosts."""
        m = _mgr_pbrs()
        boosted = _full_team_live(); boosted.opp.active.boosts = {"atk": 3}
        m.process_turn_reward(_Battle(boosted, turn=1), _delta())
        prev = m._prev_phi_opp_boosts
        self.assertAlmostEqual(prev, -OPP_BOOST_WEIGHT * 3, places=6)
        cleared = _full_team_live()   # forced-in mon, no boosts
        m.process_turn_reward(_Battle(cleared, turn=2),
                              _delta(our_move_id="roar", opp_switch_to="new"))
        self.assertAlmostEqual(m._last_breakdown.pbrs_opp_boosts, PBRS_GAMMA * 0.0 - prev, places=6)
        self.assertGreater(m._last_breakdown.pbrs_opp_boosts, 0.0)

    def test_failed_roar_no_shaping(self):
        """Roar fails (opp stays, boosts unchanged) → ΔΦ ≈ 0 (failed_roar dissolves); only the
        bounded (γ−1)·Φ telescoping residual remains, exactly like a held status."""
        m = _mgr_pbrs()
        b1 = _full_team_live(); b1.opp.active.boosts = {"spa": 2}
        m.process_turn_reward(_Battle(b1, turn=1), _delta())
        b2 = _full_team_live(); b2.opp.active.boosts = {"spa": 2}
        m.process_turn_reward(_Battle(b2, turn=2), _delta(our_move_id="roar", opp_switch_to=None))
        self.assertLess(abs(m._last_breakdown.pbrs_opp_boosts), 1e-3)   # ≈0 (γ−1)·w

    def test_terminal_zeroes_phi(self):
        """A terminal zeroes Φ → shaped = −prev (small)."""
        m = _mgr_pbrs()
        b1 = _full_team_live(); b1.opp.active.boosts = {"atk": 2}
        m.process_turn_reward(_Battle(b1, turn=1), _delta())
        prev = m._prev_phi_opp_boosts
        win = _Live([1.0] * 6, [0.0] * 6, won=True, finished=True)
        m.process_turn_reward(_Battle(win, turn=2), _delta(opp_fainted=True))
        self.assertAlmostEqual(m._last_breakdown.pbrs_opp_boosts, -prev, places=6)


class TestRoarPBRS(unittest.TestCase):
    """Φ_roar (folded into --all-shaping-pbrs): the DEDICATED phaze-out-boosts PBRS. Same state-potential
    shape as Φ_opp_boosts but its own weight (ROAR_BOOST_WEIGHT); proportional to boosts roared out."""

    def _mgr_roar(self, **cfg):
        return Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=True, **cfg))

    def test_phi_zero_at_start(self):
        self.assertAlmostEqual(self._mgr_roar()._compute_phi_roar(_full_team_live()), 0.0, places=6)

    def test_phi_proportional_to_positive_boosts(self):
        """opp atk+2, spe+1 → Φ_roar = −ROAR_BOOST_WEIGHT·3 (negatives ignored)."""
        m = self._mgr_roar()
        live = _full_team_live(); live.opp.active.boosts = {"atk": 2, "spe": 1, "def": -1}
        self.assertAlmostEqual(m._compute_phi_roar(live), -ROAR_BOOST_WEIGHT * 3, places=6)

    def test_fold_disabled_under_no_all_shaping_pbrs(self):
        """The fallback regime: pbrs_roar stays 0, prev stays None. DEFAULT-ON since 2026-08-18."""
        m = _mgr_additive_bias()
        live = _full_team_live(); live.opp.active.boosts = {"atk": 3}
        m.process_turn_reward(_Battle(live, turn=1), _delta())
        self.assertEqual(m._last_breakdown.pbrs_roar, 0.0)
        self.assertIsNone(m._prev_phi_roar)

    def test_roar_clears_boosts_pays_out_proportionally(self):
        """opp +4 stages, then Roar forces a boostless switch-in → Φ rises → +pbrs_roar = ROAR_BOOST_WEIGHT·4."""
        m = self._mgr_roar()
        boosted = _full_team_live(); boosted.opp.active.boosts = {"atk": 2, "spe": 2}
        m.process_turn_reward(_Battle(boosted, turn=1), _delta())
        prev = m._prev_phi_roar
        self.assertAlmostEqual(prev, -ROAR_BOOST_WEIGHT * 4, places=6)
        cleared = _full_team_live()   # forced-in mon, no boosts
        m.process_turn_reward(_Battle(cleared, turn=2), _delta(our_move_id="roar", opp_switch_to="new"))
        self.assertAlmostEqual(m._last_breakdown.pbrs_roar, PBRS_GAMMA * 0.0 - prev, places=6)
        self.assertGreater(m._last_breakdown.pbrs_roar, 0.0)

    def test_failed_roar_no_payout(self):
        """Roar fails (opp stays, boosts unchanged) → ΔΦ ≈ 0 (only the bounded (γ−1)·Φ residual)."""
        m = self._mgr_roar()
        b1 = _full_team_live(); b1.opp.active.boosts = {"spa": 2}
        m.process_turn_reward(_Battle(b1, turn=1), _delta())
        b2 = _full_team_live(); b2.opp.active.boosts = {"spa": 2}
        m.process_turn_reward(_Battle(b2, turn=2), _delta(our_move_id="roar", opp_switch_to=None))
        self.assertLess(abs(m._last_breakdown.pbrs_roar), 1e-3)

    def test_stacks_with_opp_boosts_under_all_shaping(self):
        """Under --all-shaping-pbrs BOTH potentials fire on the same roar — they stack (safe, both PBRS)."""
        m = Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=True))
        boosted = _full_team_live(); boosted.opp.active.boosts = {"atk": 3}
        m.process_turn_reward(_Battle(boosted, turn=1), _delta())
        cleared = _full_team_live()
        m.process_turn_reward(_Battle(cleared, turn=2), _delta(our_move_id="roar", opp_switch_to="new"))
        bd = m._last_breakdown
        self.assertAlmostEqual(bd.pbrs_opp_boosts, OPP_BOOST_WEIGHT * 3, places=6)
        self.assertAlmostEqual(bd.pbrs_roar, ROAR_BOOST_WEIGHT * 3, places=6)

    def test_terminal_zeroes_phi(self):
        m = self._mgr_roar()
        b1 = _full_team_live(); b1.opp.active.boosts = {"atk": 2}
        m.process_turn_reward(_Battle(b1, turn=1), _delta())
        prev = m._prev_phi_roar
        win = _Live([1.0] * 6, [0.0] * 6, won=True, finished=True)
        m.process_turn_reward(_Battle(win, turn=2), _delta(opp_fainted=True))
        self.assertAlmostEqual(m._last_breakdown.pbrs_roar, -prev, places=6)


if __name__ == "__main__":
    unittest.main()
