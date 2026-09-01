"""The END-STATE reward terms: the four end-state potentials' DROPS under --all-shaping-pbrs,
the global no-op-equivalence of the OFF branch, the DRAW / 250-turn-timeout terminal and the
HP-scaled self-KO penalty. Pure (no battle sim) — shared fakes in `reward_test_fakes.py`.
"""
import unittest

import numpy as np

from agents.training.reward_manager import (
    Gen3RewardManager, RewardConfig, RewardClass, RewardBreakdown, VICTORY_VALUE,
    FINISHING_BLOW_BONUS, EXPLOSION_BLOCK_BONUS, _TIMEOUT_TURN_CAP,
)
from agents.training.progress_clock import ProgressClock
from agents.training.reward_test_fakes import (
    _mgr_additive_bias, _Live, _Battle, _delta, _full_team_live,
)


class TestEndStateDrops(unittest.TestCase):
    """The two end-state switches (mirrors TestBiasDrops):
      --all-shaping-pbrs ("everything but stall") zeroes EVERY BIAS term EXCEPT the anti-stall tilt
        `no_progress_tax`.
      --stall-pbrs ("stall") zeroes `no_progress_tax` + `stall_tax`.
    Both OFF = byte-identical no-op; both ON ⇒ the WHOLE BIAS class is zero (fully-PBRS reward)."""

    _KEPT_BY_ALL = "no_progress_tax"            # the one BIAS term --all-shaping-pbrs keeps (the tilt)
    _STALL = ("no_progress_tax", "stall_tax")   # what --stall-pbrs zeroes

    def _bd_all_bias_nonzero(self):
        bd = RewardBreakdown()
        for f in bd.registry_fields(RewardClass.BIAS):
            setattr(bd, f, 1.0)
        return bd

    def test_both_flags_off_is_a_noop(self):
        """The fallback regime leaves every BIAS term standing. Was the DEFAULT until 2026-08-18 —
        `test_default_keeps_only_the_stall_tilt` below now covers the default."""
        m = _mgr_additive_bias()
        bd = self._bd_all_bias_nonzero()
        m._apply_pbrs_suppression(bd)
        for f in bd.registry_fields(RewardClass.BIAS):
            self.assertEqual(getattr(bd, f), 1.0, f"{f} must be untouched with both flags off")

    def test_default_keeps_only_the_stall_tilt(self):
        """THE DEFAULT COMPOSITION, asserted through the suppression path a real turn takes: a
        default-constructed manager zeroes every BIAS term except `no_progress_tax`."""
        m = Gen3RewardManager()   # --all-shaping-pbrs ON, --stall-pbrs off
        bd = self._bd_all_bias_nonzero()
        m._apply_pbrs_suppression(bd)
        survivors = [f for f in bd.registry_fields(RewardClass.BIAS) if getattr(bd, f) != 0.0]
        self.assertEqual(survivors, [self._KEPT_BY_ALL])

    def test_all_shaping_zeros_everything_but_stall_tilt(self):
        """--all-shaping-pbrs zeros every BIAS term EXCEPT no_progress_tax (the kept anti-stall tilt) —
        so status, stall_tax, matchup_penalty, the switch family, etc. all go."""
        m = Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=True))
        bd = self._bd_all_bias_nonzero()
        m._apply_pbrs_suppression(bd)
        for f in bd.registry_fields(RewardClass.BIAS):
            if f == self._KEPT_BY_ALL:
                self.assertEqual(getattr(bd, f), 1.0, "no_progress_tax (stall tilt) must be KEPT")
            else:
                self.assertEqual(getattr(bd, f), 0.0, f"{f} should be zeroed by --all-shaping-pbrs")

    def test_stall_pbrs_zeros_only_the_stall_terms(self):
        """--stall-pbrs alone zeros no_progress_tax + stall_tax; non-stall BIAS is untouched.
        ALONE means without --all-shaping-pbrs, which since 2026-08-18 must be said explicitly."""
        m = Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=False, stall_pbrs=True))
        bd = self._bd_all_bias_nonzero()
        m._apply_pbrs_suppression(bd)
        for f in self._STALL:
            self.assertEqual(getattr(bd, f), 0.0, f"{f} should be zeroed by --stall-pbrs")
        self.assertEqual(bd.spikes, 1.0)   # a non-stall term is untouched by --stall-pbrs alone

    def test_both_flags_zero_entire_bias_class(self):
        """--all-shaping-pbrs + --stall-pbrs ⇒ EVERY BIAS term is 0 → TERMINAL + PBRS only."""
        m = Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=True, stall_pbrs=True))
        bd = self._bd_all_bias_nonzero()
        m._apply_pbrs_suppression(bd)
        for f in bd.registry_fields(RewardClass.BIAS):
            self.assertEqual(getattr(bd, f), 0.0, f"{f} must be zeroed when both flags on")

    def test_finishing_blow_default_emits_then_dropped(self):
        """A clean opp-KO with a damaging move: OFF → +0.5; ON → 0."""
        live = _full_team_live(our_alive=6, opp_alive=1, opp_hp=0.5)
        live.ours.active.move_ids = ("tackle",)
        delta = _delta(opp_fainted=True, our_move_id="tackle",
                       opp_hp_delta=np.array([-0.5, 0, 0, 0, 0, 0], dtype=np.float32))
        m_off = _mgr_additive_bias()
        m_off.process_turn_reward(_Battle(live, turn=1), delta)
        self.assertAlmostEqual(m_off._last_breakdown.finishing_blow, FINISHING_BLOW_BONUS, places=6)
        m_on = Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=True))
        m_on.process_turn_reward(_Battle(live, turn=1), delta)
        self.assertEqual(m_on._last_breakdown.finishing_blow, 0.0)

    def test_explosion_block_default_then_dropped(self):
        """Surviving an opp Explosion (0 damage): OFF → +1.0; ON → 0."""
        from types import SimpleNamespace
        opp_event = SimpleNamespace(move_id="explosion", effectiveness=0.0, target_species=None)
        delta = _delta(opp_damaging_event=opp_event, we_fainted=False,
                       our_hp_delta=np.zeros(6, dtype=np.float32))
        m_off = _mgr_additive_bias()
        m_off.process_turn_reward(_Battle(_full_team_live(), turn=1), delta)
        self.assertAlmostEqual(m_off._last_breakdown.explosion_block, EXPLOSION_BLOCK_BONUS, places=6)
        m_on = Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=True))
        m_on.process_turn_reward(_Battle(_full_team_live(), turn=1), delta)
        self.assertEqual(m_on._last_breakdown.explosion_block, 0.0)

    def test_all_shaping_keeps_no_progress_tilt(self):
        """--all-shaping-pbrs WITHOUT --stall-pbrs + a charged no-op: no_progress_tax SURVIVES as the
        anti-stall tilt (and is activated even without --bias-redesign), and pbrs_progress stays 0
        (Φ_progress gates on --stall-pbrs)."""
        m = Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=True),
                              progress_clock=ProgressClock())
        m.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())
        m.progress_clock.last_penalty = -m.config.no_progress_penalty
        m.progress_clock.n = 3
        m.process_turn_reward(_Battle(_full_team_live(), turn=2), _delta())
        bd = m._last_breakdown
        self.assertLess(bd.no_progress_tax, 0.0)         # KEPT as the tilt
        self.assertEqual(bd.pbrs_progress, 0.0)          # Φ_progress off (needs --stall-pbrs)

    def test_stall_pbrs_converts_tilt_to_phi_progress(self):
        """Both switches on + a charged no-op: no_progress_tax == 0 (suppressed, after
        _apply_progress_clock) and pbrs_progress < 0 (Φ_progress carries the anti-stall telescoping)."""
        m = Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=True, stall_pbrs=True),
                              progress_clock=ProgressClock())
        m.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())   # prev φ = 0
        m.progress_clock.last_penalty = -m.config.no_progress_penalty
        m.progress_clock.n = 3
        m.process_turn_reward(_Battle(_full_team_live(), turn=2), _delta())
        bd = m._last_breakdown
        self.assertEqual(bd.no_progress_tax, 0.0)        # suppressed by --stall-pbrs
        self.assertLess(bd.pbrs_progress, 0.0)           # Φ_progress carries it telescoping

    def test_config_round_trips(self):
        from types import SimpleNamespace
        from dataclasses import asdict
        rc = RewardConfig.from_args(SimpleNamespace(all_shaping_pbrs=True, stall_pbrs=True,
                                                    no_progress_penalty=0.25))
        self.assertTrue(rc.all_shaping_pbrs and rc.stall_pbrs)
        self.assertEqual(rc.no_progress_penalty, 0.25)
        self.assertEqual(RewardConfig.from_dict(asdict(rc)), rc)   # round-trips for eval/resume


class TestAllShapingPbrsNoOpDefault(unittest.TestCase):
    """Global no-op-equivalence: with all_shaping_pbrs=False the four pbrs_* fields stay 0 and the
    four _prev_phi_* slots stay None across a multi-window episode (byte-identical default)."""

    def test_no_all_shaping_pbrs_leaves_new_pbrs_inert(self):
        m = _mgr_additive_bias(progress_clock=ProgressClock())
        livesteps = [
            _full_team_live(),
            (lambda l: (setattr(l.opp, "side_conditions", {"spikes": 1}), l)[1])(_full_team_live()),
            (lambda l: (setattr(l.ours.active, "boosts", {"atk": 2}), l)[1])(_full_team_live()),
            (lambda l: (setattr(l.opp.active, "boosts", {"spa": 3}), l)[1])(_full_team_live()),
            _Live([1.0] * 6, [0.0] * 6, won=True, finished=True),
        ]
        for i, live in enumerate(livesteps):
            m.progress_clock.n = i   # would charge Φ_progress if the flag were on
            m.process_turn_reward(_Battle(live, turn=i + 1), _delta())
            bd = m._last_breakdown
            self.assertEqual((bd.pbrs_progress, bd.pbrs_hazard, bd.pbrs_boost, bd.pbrs_opp_boosts),
                             (0.0, 0.0, 0.0, 0.0))
        self.assertIsNone(m._prev_phi_progress)
        self.assertIsNone(m._prev_phi_hazard)
        self.assertIsNone(m._prev_phi_boost)
        self.assertIsNone(m._prev_phi_opp_boosts)


class TestDrawPenalty(unittest.TestCase):
    """The DRAW / 250-turn-timeout terminal (draw_penalty). The trainee FORFEITS at the turn cap, so
    the timeout is detected by turn>=cap (not won/lost) and can be scored worse than a clean loss."""

    def _mgr(self, draw_penalty=-30.0):
        return Gen3RewardManager(config=RewardConfig(draw_penalty=draw_penalty))

    def _seed(self, m):
        m.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())   # set _prev_phi_mat

    def test_timeout_uses_draw_penalty(self):
        m = self._mgr(draw_penalty=-35.0); self._seed(m)
        live = _full_team_live(lost=True, finished=True)          # forfeit at the cap
        m.process_turn_reward(_Battle(live, turn=_TIMEOUT_TURN_CAP), _delta())
        self.assertAlmostEqual(m._last_breakdown.win_loss, -35.0, places=6)

    def test_decisive_loss_before_cap_unaffected(self):
        m = self._mgr(draw_penalty=-35.0); self._seed(m)
        live = _full_team_live(our_alive=0, opp_alive=3, lost=True, finished=True)
        m.process_turn_reward(_Battle(live, turn=40), _delta(we_fainted=True))   # lost well before cap
        self.assertAlmostEqual(m._last_breakdown.win_loss, -VICTORY_VALUE, places=6)

    def test_win_unaffected(self):
        m = self._mgr(draw_penalty=-35.0); self._seed(m)
        live = _full_team_live(our_alive=3, opp_alive=0, won=True, finished=True)
        m.process_turn_reward(_Battle(live, turn=_TIMEOUT_TURN_CAP), _delta(opp_fainted=True))
        self.assertAlmostEqual(m._last_breakdown.win_loss, VICTORY_VALUE, places=6)

    def test_default_draw_penalty_equals_loss_value(self):
        """Default -30 == prior behavior: a timeout scores identically to a decisive loss (byte-unchanged)."""
        m = self._mgr(); self._seed(m)                            # default -30
        live = _full_team_live(lost=True, finished=True)
        m.process_turn_reward(_Battle(live, turn=_TIMEOUT_TURN_CAP), _delta())
        self.assertAlmostEqual(m._last_breakdown.win_loss, -VICTORY_VALUE, places=6)


class TestSelfKoPenalty(unittest.TestCase):
    """The HP-scaled self-KO penalty (--self-ko-hp-penalty): −w·hp when our mon self-KOs, OFF by
    default, gated on actually-fainted + actually-executed, and untouched for non-self-KO moves."""

    def _mgr(self, weight, hp_before=1.0):
        m = Gen3RewardManager(config=RewardConfig(self_ko_hp_penalty=weight))
        m._our_active_hp_before = hp_before
        return m

    def test_off_by_default_is_zero(self):
        """w=0.0 (default) → no penalty even on a healthy self-KO (byte-unchanged)."""
        m = self._mgr(0.0, hp_before=1.0)
        self.assertEqual(
            m._compute_self_ko_penalty(_delta(our_move_id="explosion", we_fainted=True)), 0.0)

    def test_healthy_self_ko_charges_full_weight(self):
        """A full-HP Explosion that faints us → −w·1.0 (the blunder this lever targets)."""
        m = self._mgr(2.5, hp_before=1.0)
        self.assertAlmostEqual(
            m._compute_self_ko_penalty(_delta(our_move_id="explosion", we_fainted=True)),
            -2.5, places=6)

    def test_low_hp_self_ko_is_scaled_down(self):
        """A dying-mon Self-Destruct (10% HP) is the legitimate sac-for-KO → only −w·0.1."""
        m = self._mgr(2.5, hp_before=0.1)
        self.assertAlmostEqual(
            m._compute_self_ko_penalty(_delta(our_move_id="selfdestruct", we_fainted=True)),
            -0.25, places=6)

    def test_survived_explosion_no_penalty(self):
        """Blocked / immune Explosion (we did NOT faint) threw nothing away → no penalty."""
        m = self._mgr(2.5, hp_before=1.0)
        self.assertEqual(
            m._compute_self_ko_penalty(_delta(our_move_id="explosion", we_fainted=False)), 0.0)

    def test_failed_to_move_no_penalty(self):
        """KO'd before our Explosion fired (our_failed_to_move) → the faint wasn't our self-KO."""
        m = self._mgr(2.5, hp_before=1.0)
        self.assertEqual(
            m._compute_self_ko_penalty(
                _delta(our_move_id="explosion", we_fainted=True, our_failed_to_move=True)), 0.0)

    def test_non_self_ko_move_no_penalty(self):
        """Fainting to a normal attack while holding a non-self-KO move → no penalty."""
        m = self._mgr(2.5, hp_before=1.0)
        self.assertEqual(
            m._compute_self_ko_penalty(_delta(our_move_id="earthquake", we_fainted=True)), 0.0)

    def test_post_self_ko_forced_switch_window_no_double_charge(self):
        """The forced-switch follow-up window after a self-KO has our_move_id=None — the penalty must
        NOT fire there, so one self-KO is charged exactly once (no double-charge across the window)."""
        m = self._mgr(2.5, hp_before=1.0)
        self.assertEqual(
            m._compute_self_ko_penalty(_delta(our_move_id=None, we_fainted=True)), 0.0)

    def test_hp_clamped_to_unit_interval(self):
        """A stray >1 or <0 HP snapshot can't blow the penalty past [−w, 0]."""
        self.assertAlmostEqual(
            self._mgr(2.0, hp_before=1.5)._compute_self_ko_penalty(
                _delta(our_move_id="explosion", we_fainted=True)), -2.0, places=6)
        self.assertEqual(
            self._mgr(2.0, hp_before=-0.3)._compute_self_ko_penalty(
                _delta(our_move_id="explosion", we_fainted=True)), 0.0)


if __name__ == "__main__":
    unittest.main()
