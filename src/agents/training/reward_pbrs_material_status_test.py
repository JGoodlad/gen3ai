"""The MATERIAL and STATUS PBRS potentials (design_markovian_reward_and_features.md §2):
Φ_mat (declared-team material) and Φ_status (tempo) — telescoping, terminal zeroing, side
symmetry. Pure (no battle sim) — the shared fakes live in `reward_test_fakes.py`.
"""
import unittest

from agents.training.reward_manager import (
    Gen3RewardManager, RewardConfig, MAT_HP_WEIGHT, MAT_ALIVE_WEIGHT, STATUS_TEMPO_WEIGHT,
    PBRS_GAMMA, VICTORY_VALUE,
)
from agents.training.reward_test_fakes import (
    _mgr_additive_bias, _Live, _Battle, _delta, _full_team_live,
)


class TestMaterialPBRS(unittest.TestCase):
    """Φ_mat (design §2): declared-team material potential, telescoping, terminal zeroing."""

    def _mgr(self):
        return Gen3RewardManager()

    def test_phi_mat_start_is_zero(self):
        """6v6 full HP over DECLARED teams → Φ_mat(s_0) ≈ 0 (no reveal jumps / start variance)."""
        m = self._mgr()
        phi = m._compute_phi_mat(_full_team_live())
        self.assertAlmostEqual(phi, 0.0, places=6)

    def test_phi_mat_unrevealed_opp_counts_full(self):
        """Only the opp lead revealed (team_size 6) → opp still counts 6 full-HP-alive → Φ_mat≈0."""
        m = self._mgr()
        live = _Live([1.0] * 6, [1.0], opp_team_size=6)   # opp has 1 revealed mon, 5 unrevealed
        self.assertAlmostEqual(m._compute_phi_mat(live), 0.0, places=6)

    def test_phi_mat_faint_lowers_never_raises(self):
        """An our-faint drops Φ_mat (HP term + alive term); it must never raise it."""
        m = self._mgr()
        before = m._compute_phi_mat(_full_team_live(our_alive=6))
        after = m._compute_phi_mat(_full_team_live(our_alive=5))   # one of ours fainted
        self.assertLess(after, before)
        # Drop ≈ 2·1.0 (HP) + MAT_ALIVE_WEIGHT (alive bit).
        self.assertAlmostEqual(before - after, MAT_HP_WEIGHT * 1.0 + MAT_ALIVE_WEIGHT, places=5)

    def test_phi_mat_stashes_normalized_margin(self):
        """_compute_phi_mat stashes the normalized material margin ∈ [−1,1] (clamped Φ_mat / bound) as a
        pure by-product for the win-prob obs key: even board → ~0, our full wipe-lead → +1, reset → 0."""
        m = self._mgr()
        bound = MAT_HP_WEIGHT * 6 + MAT_ALIVE_WEIGHT * 6
        phi0 = m._compute_phi_mat(_full_team_live())                    # even
        self.assertAlmostEqual(m._last_material_margin, 0.0, places=6)
        self.assertAlmostEqual(m._last_material_margin, phi0 / bound, places=6)
        m._compute_phi_mat(_full_team_live(opp_alive=0))               # we have everything, they nothing
        self.assertAlmostEqual(m._last_material_margin, 1.0, places=6)
        m.reset()
        self.assertEqual(m._last_material_margin, 0.0)

    def test_telescopes_to_minus_phi0(self):
        """Drive the manager over a controlled-HP trajectory ending terminal; the UNDISCOUNTED sum
        of pbrs_material ≈ −Φ_mat(s_0) + the bounded (γ−1) residual (coarse telescoping check)."""
        m = self._mgr()
        # Trajectory: we chip the opp down, lose one mon, then win. Φ_mat(s_0)=0 (6v6 full).
        livesteps = [
            _full_team_live(opp_hp=1.0),                 # s_1 (first window → no shaping)
            _full_team_live(opp_hp=0.5),                 # dealt damage
            _full_team_live(our_alive=5, opp_hp=0.5),    # we lost a mon
            _full_team_live(our_alive=5, opp_alive=2),   # KO'd some opp
            _full_team_live(our_alive=5, opp_alive=0, won=True, finished=True),   # win (terminal)
        ]
        phi0 = m._compute_phi_mat(livesteps[0])
        s = 0.0
        for i, live in enumerate(livesteps):
            m.process_turn_reward(_Battle(live, turn=i + 1), _delta())
            s += m._last_breakdown.pbrs_material
        # Σ(γΦ′−Φ) telescopes to γ^T·Φ_T − Φ_0 + (γ−1)Σ_interior; Φ_T zeroed at terminal.
        resid = abs((PBRS_GAMMA - 1.0)) * 6 * (MAT_HP_WEIGHT * 6 + MAT_ALIVE_WEIGHT * 6)
        self.assertAlmostEqual(s, -phi0, delta=resid + 1e-6)

    def test_terminal_zeroes_phi_no_dominant_bonus(self):
        """The highest-risk bug (design §2.3): a 6-0 win has Φ_mat(post-win)≈+19.5, which MUST be
        zeroed → pbrs_material = −prev (small), NOT a +19.5 dominant-win bonus."""
        m = self._mgr()
        # window 1: even board → set _prev_phi_mat.
        m.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())
        prev = m._prev_phi_mat
        # window 2: we win 6-0 at full HP (post-win Φ_mat would be +19.5 un-zeroed).
        win_live = _Live([1.0] * 6, [0.0] * 6, won=True, finished=True)
        m.process_turn_reward(_Battle(win_live, turn=2), _delta(opp_fainted=True))
        bd = m._last_breakdown
        self.assertAlmostEqual(bd.pbrs_material, -prev, places=6)   # = γ·0 − prev
        self.assertLess(abs(bd.pbrs_material), 1.0)                  # small, not +19.5
        self.assertAlmostEqual(bd.win_loss, VICTORY_VALUE, places=6)


class TestStatusPBRS(unittest.TestCase):
    """Φ_status (design §2.7 / §7.4): the non-damaging-tempo standing potential. Restores the
    standing value the event-form status BIAS drops — gated on bias_redesign, telescoping, net-zero."""

    @staticmethod
    def _set_status(live, side, idx, status):
        """Set `status` on the idx-th mon of `live.<side>` ('ours'/'opp'). The _Side tuple is
        immutable but each _Mon is mutable."""
        getattr(live, side).mons[idx].status = status

    def test_phi_status_zero_when_nobody_statused(self):
        m = Gen3RewardManager()
        self.assertAlmostEqual(m._compute_phi_status(_full_team_live()), 0.0, places=6)

    def test_phi_status_counts_non_damaging_only(self):
        """par/slp/frz move Φ_status; tox/brn/psn do NOT (their value is the HP chip → Φ_mat)."""
        m = Gen3RewardManager()
        for dmg in ("tox", "brn", "psn"):
            live = _full_team_live()
            self._set_status(live, "opp", 0, dmg)
            self.assertAlmostEqual(m._compute_phi_status(live), 0.0, places=6,
                                   msg=f"{dmg} must not move Φ_status (it is in Φ_mat)")
        for tempo in ("par", "slp", "frz"):
            live = _full_team_live()
            self._set_status(live, "opp", 0, tempo)
            self.assertAlmostEqual(m._compute_phi_status(live), STATUS_TEMPO_WEIGHT, places=6,
                                   msg=f"{tempo} must raise Φ_status by one weight")

    def test_phi_status_side_symmetry(self):
        """opp statused → +weight (good for us); our statused → −weight."""
        m = Gen3RewardManager()
        opp_live = _full_team_live(); self._set_status(opp_live, "opp", 0, "slp")
        our_live = _full_team_live(); self._set_status(our_live, "ours", 0, "par")
        self.assertAlmostEqual(m._compute_phi_status(opp_live), +STATUS_TEMPO_WEIGHT, places=6)
        self.assertAlmostEqual(m._compute_phi_status(our_live), -STATUS_TEMPO_WEIGHT, places=6)

    def test_fold_gated_off_without_either_switch(self):
        """`--no-all-shaping-pbrs` and no `--bias-redesign`: pbrs_status ≡ 0 and _prev_phi_status
        stays None even with a sleeping opp mon (the count-diff `status` BIAS pays the standing
        value there instead, so folding Φ_status would double-count). This was the DEFAULT until
        2026-08-18; it is now the fallback, and `test_fold_active_by_default` covers the default."""
        m = _mgr_additive_bias()
        live1 = _full_team_live()
        m.process_turn_reward(_Battle(live1, turn=1), _delta())
        live2 = _full_team_live(); self._set_status(live2, "opp", 0, "slp")
        m.process_turn_reward(_Battle(live2, turn=2), _delta())
        self.assertEqual(m._last_breakdown.pbrs_status, 0.0)
        self.assertIsNone(m._prev_phi_status)

    def test_fold_active_under_redesign_application_and_cure(self):
        """Under redesign: an opp falling asleep RAISES Φ → +pbrs_status; the subsequent cure (wake)
        DROPS Φ → −pbrs_status. A held status pays ≈0 (γ≈1)."""
        m = Gen3RewardManager(config=RewardConfig(bias_redesign=True))
        m.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())  # prev=0
        sleep_live = _full_team_live(); self._set_status(sleep_live, "opp", 0, "slp")
        m.process_turn_reward(_Battle(sleep_live, turn=2), _delta())          # apply
        applied = m._last_breakdown.pbrs_status
        self.assertGreater(applied, 0.0)
        self.assertAlmostEqual(applied, PBRS_GAMMA * STATUS_TEMPO_WEIGHT, places=6)
        held_live = _full_team_live(); self._set_status(held_live, "opp", 0, "slp")
        m.process_turn_reward(_Battle(held_live, turn=3), _delta())           # held
        self.assertLess(abs(m._last_breakdown.pbrs_status), 1e-3)             # ≈0 (γ−1)·w
        m.process_turn_reward(_Battle(_full_team_live(), turn=4), _delta())   # cured
        self.assertLess(m._last_breakdown.pbrs_status, 0.0)

    def test_telescopes_to_zero_net(self):
        """Status applied then cured before a terminal that starts & ends with nobody statused →
        the UNDISCOUNTED sum of pbrs_status ≈ −Φ_status(s_0) = 0 (modulo the bounded γ residual)."""
        m = Gen3RewardManager(config=RewardConfig(bias_redesign=True))
        s1 = _full_team_live()
        s2 = _full_team_live(); self._set_status(s2, "opp", 0, "frz")
        s3 = _full_team_live(); self._set_status(s3, "opp", 0, "frz")
        s4 = _full_team_live()                                                # thawed (cured)
        s5 = _Live([1.0] * 6, [0.0] * 6, won=True, finished=True)            # win (terminal)
        total = 0.0
        for i, live in enumerate((s1, s2, s3, s4, s5)):
            m.process_turn_reward(_Battle(live, turn=i + 1), _delta())
            total += m._last_breakdown.pbrs_status
        resid = abs(PBRS_GAMMA - 1.0) * 5 * STATUS_TEMPO_WEIGHT + 1e-6
        self.assertAlmostEqual(total, 0.0, delta=resid)


if __name__ == "__main__":
    unittest.main()
