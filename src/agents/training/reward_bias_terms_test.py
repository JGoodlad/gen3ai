"""The BIAS-class reward terms (design_markovian_reward_and_features.md §3): the
accumulate-refund additivity blend, the redundant/switch bias DROPS, the belief-risk-scaled
switch-bias lever (design_reward_switching.md §6) and the obs-keyed bias_redesign reframes.
Pure (no battle sim) — the shared fakes live in `reward_test_fakes.py`.
"""
import unittest

from agents.training.reward_manager import (
    Gen3RewardManager, RewardConfig, RewardClass, RewardBreakdown, PBRS_GAMMA,
    SWITCH_RISK_THRESHOLD, STAY_RISK_TAX_FLOOR, ESCAPE_RISK_FRACTION,
)
from agents.training.reward_test_fakes import (
    _mgr_additive_bias, _Live, _Battle, _delta, _full_team_live,
)


class TestBiasAdditivity(unittest.TestCase):
    """The bias-additivity accumulate-refund (design §1.2)."""

    def _run_episode(self, lam, bias_per_turn=(0.2, -0.1, 0.3), terminal_win=True):
        """Drive the manager over windows that each inject a known BIAS value via `roar`, then a
        terminal. Returns (per-window totals, episode bias contribution)."""
        m = Gen3RewardManager(config=RewardConfig(bias_additivity=lam))
        totals = []
        acc = 0.0
        n = len(bias_per_turn)
        for i, b in enumerate(bias_per_turn):
            live = _full_team_live()
            is_term = terminal_win and (i == n - 1)
            if is_term:
                live = _Live([1.0] * 6, [0.0] * 6, won=True, finished=True)
            m.process_turn_reward(_Battle(live, turn=i + 1), _delta())
            # Inject the bias post-hoc by reusing the fold: simplest is to set roar in the breakdown
            # and recompute — instead, assert via a dedicated manager hook. We emulate by adding b to
            # the recorded bias accumulator through the public refund formula.
            acc += b
            totals.append(b)
        return totals, acc

    def test_lambda_one_refund_is_zero(self):
        """λ=1 → bias_refund ≡ 0 on every window (byte no-op for the bias class)."""
        m = Gen3RewardManager(config=RewardConfig(bias_additivity=1.0))
        # A window with a nonzero bias (roar) and a material change.
        m.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())
        m.process_turn_reward(_Battle(_full_team_live(opp_hp=0.5), turn=2),
                              _delta(our_move_id="roar"))
        self.assertEqual(m._last_breakdown.bias_refund, 0.0)

    def test_parameterized_blend_episode_sum(self):
        """Episode-summed BIAS contribution == λ·acc for λ ∈ {0, 0.5, 1} (accumulate-refund). We
        drive a known per-window bias accumulator and check the refund column telescopes to
        −(1−λ)·acc (modulo the bounded γ residual)."""
        # Use the manager's own accumulator by feeding windows whose ONLY bias is stall_tax (a
        # deterministic function of turn). Compare the summed (bias + refund) across λ.
        def episode_bias_sum(lam):
            m = Gen3RewardManager(config=RewardConfig(bias_additivity=lam))
            bias_total = 0.0
            refund_total = 0.0
            turns = [120, 140, 160]   # past STALL_TAX_START_TURN → nonzero stall_tax each window
            for i, t in enumerate(turns):
                live = _full_team_live()
                if i == len(turns) - 1:
                    live = _Live([1.0] * 6, [0.0] * 6, won=True, finished=True)
                m.process_turn_reward(_Battle(live, turn=t), _delta())
                bd = m._last_breakdown
                bias_total += sum(getattr(bd, f) for f in bd.registry_fields(RewardClass.BIAS))
                refund_total += bd.bias_refund
            return bias_total, refund_total

        b1, r1 = episode_bias_sum(1.0)
        b0, r0 = episode_bias_sum(0.0)
        bh, rh = episode_bias_sum(0.5)
        # Per-window bias values are IDENTICAL across λ (only the refund differs).
        self.assertAlmostEqual(b1, b0, places=9)
        self.assertAlmostEqual(b1, bh, places=9)
        # λ=1 → refund 0; λ=0 → refund ≈ −acc; λ=0.5 → ≈ −0.5·acc (bounded γ residual).
        self.assertAlmostEqual(r1, 0.0, places=9)
        resid = abs(PBRS_GAMMA - 1.0) * abs(b1) * 3 + 1e-9
        self.assertAlmostEqual(b0 + r0, 0.0, delta=resid)          # contribution λ·acc = 0
        self.assertAlmostEqual(bh + rh, 0.5 * b1, delta=resid)     # contribution 0.5·acc


class TestBiasDrops(unittest.TestCase):
    """De-bias cleanup (audit TIER-1 distorter removal): --drop-redundant-bias / --drop-switch-bias
    ZERO their BIAS terms before the refund fold (so they also leave the accumulator). Both default
    OFF = byte-identical no-op."""

    _REDUNDANT = ("stall_tax", "matchup_penalty")
    _SWITCH = ("switch_base", "switch_bouncing_tax", "escape_threat_switch", "se_switch",
               "pivot_protect", "pivot_status", "pivot_damage", "sleep_out", "sleep_in")

    def _bd_all_bias_nonzero(self):
        bd = RewardBreakdown()
        for f in bd.registry_fields(RewardClass.BIAS):
            setattr(bd, f, 1.0)
        return bd

    def test_dropped_terms_are_registered_bias(self):
        """Every dropped field must be a BIAS-class registry member, else the drop is a typo no-op."""
        bias = set(RewardBreakdown().registry_fields(RewardClass.BIAS))
        for f in self._REDUNDANT + self._SWITCH:
            self.assertIn(f, bias, f"{f} is not a BIAS field")

    def test_default_is_noop(self):
        m = Gen3RewardManager()   # both flags default False
        bd = self._bd_all_bias_nonzero()
        m._apply_bias_drops(bd)
        for f in bd.registry_fields(RewardClass.BIAS):
            self.assertEqual(getattr(bd, f), 1.0, f"{f} must be untouched by default")

    def test_drop_redundant_bias_zeros_only_those(self):
        m = Gen3RewardManager(config=RewardConfig(drop_redundant_bias=True))
        bd = self._bd_all_bias_nonzero()
        m._apply_bias_drops(bd)
        for f in self._REDUNDANT:
            self.assertEqual(getattr(bd, f), 0.0, f"{f} should be dropped")
        for f in self._SWITCH:
            self.assertEqual(getattr(bd, f), 1.0, f"{f} should be untouched")

    def test_drop_switch_bias_zeros_only_those(self):
        m = Gen3RewardManager(config=RewardConfig(drop_switch_bias=True))
        bd = self._bd_all_bias_nonzero()
        m._apply_bias_drops(bd)
        for f in self._SWITCH:
            self.assertEqual(getattr(bd, f), 0.0, f"{f} should be dropped")
        for f in self._REDUNDANT:
            self.assertEqual(getattr(bd, f), 1.0, f"{f} should be untouched")

    def test_both_flags_drop_union(self):
        m = Gen3RewardManager(config=RewardConfig(drop_redundant_bias=True, drop_switch_bias=True))
        bd = self._bd_all_bias_nonzero()
        m._apply_bias_drops(bd)
        for f in self._REDUNDANT + self._SWITCH:
            self.assertEqual(getattr(bd, f), 0.0)

    def test_drop_redundant_zeros_stall_tax_end_to_end(self):
        """A real turn past STALL_TAX_START_TURN charges stall_tax; the flag zeros it in the breakdown."""
        # Both arms are in the additive-BIAS regime: --all-shaping-pbrs (the DEFAULT since
        # 2026-08-18) already zeroes stall_tax, which would make --drop-redundant-bias untestable.
        m_off = _mgr_additive_bias()
        m_off.process_turn_reward(_Battle(_full_team_live(), turn=120), _delta())
        self.assertLess(m_off._last_breakdown.stall_tax, 0.0)   # charged when off
        m_on = _mgr_additive_bias(config=RewardConfig(all_shaping_pbrs=False,
                                                      drop_redundant_bias=True))
        m_on.process_turn_reward(_Battle(_full_team_live(), turn=120), _delta())
        self.assertEqual(m_on._last_breakdown.stall_tax, 0.0)   # dropped when on

    def test_config_flows_through_from_args_and_from_dict(self):
        from types import SimpleNamespace
        from dataclasses import asdict
        rc = RewardConfig.from_args(SimpleNamespace(drop_redundant_bias=True, drop_switch_bias=True))
        self.assertTrue(rc.drop_redundant_bias and rc.drop_switch_bias)
        self.assertEqual(RewardConfig.from_dict(asdict(rc)), rc)   # round-trips for eval/resume


class TestSwitchBias(unittest.TestCase):
    """The belief-risk-scaled switch BIAS lever (design_reward_switching.md §6) — the under-switch
    fix that `pbrs_belief` (policy-invariant) can't be. Tests the gates + scaling directly off the
    decision-time snapshots (`_prev_active_ko_risk` / `_prev_safe_pivot`), which is what the fold sets."""

    def _mgr(self, weight):
        # `--no-all-shaping-pbrs`: these are BIAS-class terms, and the DEFAULT composition (since
        # 2026-08-18) zeroes every BIAS term but `no_progress_tax`. The lever is only reachable in
        # the additive-BIAS regime, so the regime is stated rather than inherited.
        return Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=False,
                                                     switch_bias_weight=weight))

    def _armed(self, weight, risk=0.7, safe=True):
        """A manager with the decision-time threat snapshot pre-set (as the fold would)."""
        m = self._mgr(weight)
        m._prev_active_ko_risk = risk
        m._prev_safe_pivot = safe
        return m

    # --- stay-into-KO tax -------------------------------------------------- #
    def test_off_by_default(self):
        """switch_bias_weight=0.0 (default) → no tax even in a maximal-danger state (single-variable)."""
        m = self._armed(0.0, risk=1.0, safe=True)
        self.assertEqual(m._compute_stay_risk_tax(_delta(our_switch_to=None)), 0.0)

    def test_stay_tax_fires_scaled_by_risk(self):
        m = self._armed(1.5, risk=0.7, safe=True)
        self.assertAlmostEqual(m._compute_stay_risk_tax(_delta(our_switch_to=None)), -1.5 * 0.7, places=6)

    def test_stay_tax_clamped_at_floor(self):
        m = self._armed(3.0, risk=1.0, safe=True)   # -3.0 would exceed the floor
        self.assertAlmostEqual(m._compute_stay_risk_tax(_delta(our_switch_to=None)),
                               STAY_RISK_TAX_FLOOR, places=6)

    def test_no_tax_without_safe_pivot(self):
        """A forced stay (no bench-in would survive) is NEVER taxed — the key false-positive guard."""
        m = self._armed(1.5, risk=0.9, safe=False)
        self.assertEqual(m._compute_stay_risk_tax(_delta(our_switch_to=None)), 0.0)

    def test_no_tax_below_risk_threshold(self):
        m = self._armed(1.5, risk=SWITCH_RISK_THRESHOLD - 0.01, safe=True)
        self.assertEqual(m._compute_stay_risk_tax(_delta(our_switch_to=None)), 0.0)

    def test_no_tax_when_we_switched(self):
        m = self._armed(1.5, risk=0.9, safe=True)
        self.assertEqual(m._compute_stay_risk_tax(_delta(our_switch_to=2)), 0.0)

    def test_no_tax_when_we_kod_them(self):
        """If our move KO'd the opp, staying WON the exchange — not a misplay, no tax."""
        m = self._armed(1.5, risk=0.9, safe=True)
        self.assertEqual(m._compute_stay_risk_tax(_delta(our_switch_to=None, opp_fainted=True)), 0.0)

    def test_no_tax_when_trapped(self):
        """The key false-positive guard: a TRAPPED stay (no legal switch this decision) is NOT taxed,
        even with a 'safe' bench it can't reach."""
        m = self._armed(1.5, risk=0.95, safe=True)
        m._cur_can_switch = False                       # mask had no legal switch (Arena Trap etc.)
        self.assertEqual(m._compute_stay_risk_tax(_delta(our_switch_to=None)), 0.0)

    def test_no_tax_on_rng_fizzle(self):
        """A flinch / full-para / sleep / freeze (our_failed_to_move) is not a deliberate stay."""
        m = self._armed(1.5, risk=0.95, safe=True)
        self.assertEqual(m._compute_stay_risk_tax(_delta(our_switch_to=None, our_failed_to_move=True)), 0.0)

    def test_tax_fires_on_stay_and_die(self):
        """Staying-and-fainting is the TARGET pathology — it must be taxed (not exempted)."""
        m = self._armed(1.5, risk=0.95, safe=True)
        self.assertAlmostEqual(
            m._compute_stay_risk_tax(_delta(our_switch_to=None, we_fainted=True)), -1.5 * 0.95, places=6)

    # --- escape reward ----------------------------------------------------- #
    def test_escape_bonus_fires_scaled_and_asymmetric(self):
        m = self._armed(1.5, risk=0.8, safe=True)
        m._last_reward_metadata = {"type": "VOLUNTARY", "target_species": "x",
                                   "decision_turn": 5, "switch_from": "y"}
        m.last_switch_turn = 0
        m._last_switched_from = "NULL"
        bd = RewardBreakdown()
        m._apply_switch_outcome(_delta(our_switch_to=2), bd)
        self.assertAlmostEqual(bd.escape_risk_bonus, 1.5 * ESCAPE_RISK_FRACTION * 0.8, places=6)
        # asymmetric: the escape reward is strictly smaller than the equivalent stay-tax magnitude
        self.assertLess(bd.escape_risk_bonus, 1.5 * 0.8)

    def test_escape_bonus_needs_safe_pivot(self):
        """Reward escaping TO safety, not sacrificing a fresh mon into the same threat (kills the
        no-safe-pivot rotation farm)."""
        m = self._armed(1.5, risk=0.8, safe=False)      # high risk but NO safe pivot
        m._last_reward_metadata = {"type": "VOLUNTARY", "target_species": "x",
                                   "decision_turn": 5, "switch_from": "y"}
        m.last_switch_turn = 0
        m._last_switched_from = "NULL"
        bd = RewardBreakdown()
        m._apply_switch_outcome(_delta(our_switch_to=2), bd)
        self.assertEqual(bd.escape_risk_bonus, 0.0)

    def test_escape_bonus_off_by_default(self):
        m = self._armed(0.0, risk=0.8, safe=True)
        m._last_reward_metadata = {"type": "VOLUNTARY", "target_species": "x",
                                   "decision_turn": 5, "switch_from": "y"}
        m.last_switch_turn = 0
        m._last_switched_from = "NULL"
        bd = RewardBreakdown()
        m._apply_switch_outcome(_delta(our_switch_to=2), bd)
        self.assertEqual(bd.escape_risk_bonus, 0.0)

    def test_escape_bonus_needs_high_risk(self):
        m = self._armed(1.5, risk=SWITCH_RISK_THRESHOLD - 0.01, safe=True)
        m._last_reward_metadata = {"type": "VOLUNTARY", "target_species": "x",
                                   "decision_turn": 5, "switch_from": "y"}
        m.last_switch_turn = 0
        m._last_switched_from = "NULL"
        bd = RewardBreakdown()
        m._apply_switch_outcome(_delta(our_switch_to=2), bd)
        self.assertEqual(bd.escape_risk_bonus, 0.0)

    # --- class membership + plumbing -------------------------------------- #
    def test_new_fields_are_bias_class(self):
        bd = RewardBreakdown()
        self.assertIn("stay_risk_tax", bd.registry_fields(RewardClass.BIAS))
        self.assertIn("escape_risk_bonus", bd.registry_fields(RewardClass.BIAS))

    def test_stay_tax_flows_through_process_turn_reward(self):
        """End-to-end through the full fold (not just the helper): a high-risk stay with a safe pivot
        lands a nonzero stay_risk_tax in the breakdown and in the summed reward."""
        m = self._mgr(1.5)
        m._prev_active_ko_risk = 0.9          # decision-time snapshot, as the prior turn's fold sets
        m._prev_safe_pivot = True
        m._last_reward_metadata = {"type": "ATTACK"}   # a stay (move), not a switch
        m.process_turn_reward(_Battle(_full_team_live(), turn=3),
                              _delta(our_switch_to=None, our_move_id="thunderbolt"))
        bd = m._last_breakdown
        self.assertAlmostEqual(bd.stay_risk_tax, -1.5 * 0.9, places=6)
        self.assertLess(bd.total, 0.0)        # the tax pulls the (otherwise ~0) turn negative

    def test_belief_potential_returns_safe_pivot_signal(self):
        """The fold's source: _belief_potential_and_risk now returns the 3rd value (min bench P(KO))."""
        m = self._mgr(1.5)
        out = m._belief_potential_and_risk(_full_team_live())
        self.assertEqual(len(out), 3)   # (phi, active_risk, min_bench_pko)

    def test_fold_belief_pbrs_sets_safe_pivot_snapshot(self):
        """_fold_belief_pbrs must populate _prev_safe_pivot (so the next turn's stay-tax can gate)."""
        m = self._mgr(1.5)
        bd = RewardBreakdown()
        m._fold_belief_pbrs(bd, _full_team_live(), is_terminal=False)
        self.assertIsInstance(m._prev_safe_pivot, bool)
        # terminal zeroes the snapshot (no shaping carried past the game end)
        m._fold_belief_pbrs(bd, _full_team_live(), is_terminal=True)
        self.assertFalse(m._prev_safe_pivot)


class TestBiasRedesignReframes(unittest.TestCase):
    """The obs-keyed reframes that activate ONLY under bias_redesign (design §3 #18/#25/#29). The
    default run (redesign OFF) keeps today's hidden-state behavior (byte-identical) — pinned by the
    legacy TestSwitchSubsidy / TestSeSwitchBonus / TestStatus suites running at the default config."""

    def _mgr(self, redesign):
        return Gen3RewardManager(config=RewardConfig(bias_redesign=redesign))

    # --- status (#29): transition events under redesign, count diff by default ---
    def test_status_redesign_keys_on_transition_events(self):
        from agents.enums import Status
        live = _full_team_live()   # mock snapshot shows nobody statused
        # opp GAINED a status this window (the event) → + even though the count snapshot is 0.
        reward, d_opp = self._mgr(True)._compute_status_reward(_delta(opp_status_applied=Status.TOX), live)
        self.assertGreater(reward, 0.0)
        self.assertGreater(d_opp, 0)
        # opp CURED a status (e.g. Rest) → − (we lost the pressure).
        reward_c, _ = self._mgr(True)._compute_status_reward(_delta(opp_status_cured=Status.TOX), live)
        self.assertLess(reward_c, 0.0)

    def test_status_default_keys_on_count_diff(self):
        statused = _full_team_live()
        statused.opp.mons[0].status = "tox"   # a statused opp mon, no transition event in the delta
        reward, d_opp = self._mgr(False)._compute_status_reward(_delta(), statused)
        self.assertGreater(reward, 0.0)   # count diff (0→1) drives it; events would have given 0
        self.assertEqual(d_opp, 1)

    # --- switch anti-bounce (#18 + the ai_v5_6 bounce-farm regression fix) ---
    def _settle_switch(self, m, decision_turn, last_switch_turn, family=None):
        m._last_reward_metadata = {"type": "VOLUNTARY", "decision_turn": decision_turn,
                                   "switch_from": "zapdos", "target_species": "tyranitar"}
        m._last_switched_from = "snorlax"   # != target → not a 2-cycle bounce
        m.last_switch_turn = last_switch_turn
        bd = RewardBreakdown()
        if family:   # pre-set the rewards process_turn_reward folds BEFORE _apply_switch_outcome
            bd.se_switch, bd.pivot_protect, bd.pivot_status, bd.pivot_damage = family
        m._apply_switch_outcome(_delta(our_switch_to="tyranitar"), bd)
        return bd

    def test_switch_base_spam_gated_under_redesign(self):
        """ai_v5_6 fix: switch_base is now spam-gated under the redesign too (was flat = the bug). A
        back-to-back switch (decision 5, last 4) → 0; a switch after committing (>1-turn gap) → full."""
        from agents.training.reward_manager import SWITCH_BASE_BONUS
        self.assertAlmostEqual(self._settle_switch(self._mgr(True), 5, 4).switch_base, 0.0, places=6)
        self.assertAlmostEqual(self._settle_switch(self._mgr(True), 5, 2).switch_base,
                               SWITCH_BASE_BONUS, places=6)

    def test_switch_base_spam_gated_by_default(self):
        bd = self._settle_switch(self._mgr(False), decision_turn=5, last_switch_turn=4)
        self.assertAlmostEqual(bd.switch_base, 0.0, places=6)   # back-to-back → spam-gated to 0

    def test_bounce_farm_zeros_whole_switch_family_under_redesign(self):
        """THE ai_v5_6 regression guard: a BACK-TO-BACK switch must collect NOTHING from any switch
        reward (switch_base / se_switch / escape / pivot_*) under the redesign, so an every-turn
        bounce-farm (any cycle length) is unprofitable. (Real run lost to random at 94% switches.)"""
        m = self._mgr(True)
        m._prev_opp_se_threat = True   # would otherwise fire escape_threat_switch
        bd = self._settle_switch(m, decision_turn=5, last_switch_turn=4,
                                 family=(0.2, 0.1, 0.1, 0.1))   # se_switch + pivots pre-folded
        self.assertEqual(
            (bd.switch_base, bd.se_switch, bd.escape_threat_switch,
             bd.pivot_protect, bd.pivot_status, bd.pivot_damage),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def test_legit_switch_keeps_family_under_redesign(self):
        """The gate only nukes back-to-back farming — a switch after committing (>1-turn gap) keeps
        the full switch-reward family, so legitimate pivoting is unaffected."""
        m = self._mgr(True)
        m._prev_opp_se_threat = True
        bd = self._settle_switch(m, decision_turn=5, last_switch_turn=2,   # 3-turn gap
                                 family=(0.2, 0.1, 0.1, 0.1))
        self.assertGreater(bd.switch_base, 0.0)
        self.assertEqual(bd.se_switch, 0.2)               # untouched
        self.assertEqual(bd.pivot_damage, 0.1)            # untouched
        self.assertGreater(bd.escape_threat_switch, 0.0)  # paid (threat present, not back-to-back)

    def test_switch_bouncing_tax_survives_clock_under_redesign(self):
        """The escalating 2-cycle bounce tax is NOT suppressed by the no-progress clock (it stays a
        real negative brake); repetition / struggle / dead-matchup ARE still subsumed."""
        from agents.training.progress_clock import ProgressClock
        m = self._mgr(True); m.progress_clock = ProgressClock()
        bd = RewardBreakdown()
        bd.switch_bouncing_tax = -1.5; bd.repetition_tax = -0.2; bd.dead_matchup_tax = -0.3
        m._apply_progress_clock(bd)
        self.assertEqual(bd.switch_bouncing_tax, -1.5)   # KEPT (the fix)
        self.assertEqual(bd.repetition_tax, 0.0)         # subsumed
        self.assertEqual(bd.dead_matchup_tax, 0.0)       # subsumed


if __name__ == "__main__":
    unittest.main()
