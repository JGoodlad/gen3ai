"""Unit pins for the suppressed-term fast path (`gen3_reward_skip_suppressed_v1`).

`process_turn_reward` used to compute ~20 BIAS helpers and then hand the results to
`_apply_pbrs_suppression`, which zeroes them under the production composition. The skip stops
computing what the composition forces to zero. Because this is THE OBJECTIVE — a wrong reward
trains a wrong policy with no error anywhere — the skip is gated three ways:

  * here, on the DERIVATION (the active set is `_bias_term_active`'s, not a hand-copied list,
    and it is exactly the complement of what the suppression zeroes);
  * here, on the shadow mode (`GEN3AI_REWARD_VERIFY=1` must actually catch a divergence);
  * and in `reward_skip_parity_fuzz_test.py`, on real bridge battles across three compositions,
    per-field bit-identity.

The `total` micro also lives here: the property now sums a cached field-NAME tuple instead of
re-deriving `dataclasses.fields()` per turn, which is only equivalent because every declared
field is a float. `test_every_breakdown_field_is_declared_float` is that pin.
"""

import dataclasses
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agents.training.reward_manager import (
    Gen3RewardManager,
    RewardBreakdown,
    RewardClass,
    RewardConfig,
    _bias_term_active,
    reward_class_composition,
)
from agents.training.reward_verify import RewardVerifyMismatch, verify_turn


def _verify(manager, battle, bd):
    """The manager's own shadow comparison, with the arguments it passes at the call site."""
    return verify_turn(manager._verify_twin, battle, None, bd,
                       RewardBreakdown.field_names(), manager._active_bias)

# The three documented compositions the fuzz also sweeps, plus the corners that switch on the
# weight-gated and drop-gated terms — so the derivation is pinned where it is non-trivial.
_COMPOSITIONS = {
    "production (default)": RewardConfig(),
    "--no-all-shaping-pbrs": RewardConfig(all_shaping_pbrs=False),
    "--stall-pbrs": RewardConfig(all_shaping_pbrs=True, stall_pbrs=True),
    "bias-redesign": RewardConfig(all_shaping_pbrs=False, bias_redesign=True),
    "drops": RewardConfig(all_shaping_pbrs=False, drop_redundant_bias=True,
                          drop_switch_bias=True),
    "weight-gated on": RewardConfig(all_shaping_pbrs=False, switch_bias_weight=0.5,
                                    self_ko_hp_penalty=1.0),
}


class TestActiveBiasDerivation(unittest.TestCase):
    """The skip's active set must BE the census's, not a copy that can drift from it."""

    def test_active_set_matches_the_census_for_every_composition(self):
        for label, cfg in _COMPOSITIONS.items():
            with self.subTest(composition=label):
                m = Gen3RewardManager(config=cfg)
                self.assertEqual(
                    m._active_bias,
                    frozenset(reward_class_composition(cfg)["bias_terms"]),
                    "the fast path's active BIAS set diverged from the startup census — "
                    "they must both read _bias_term_active",
                )

    def test_production_composition_activates_only_the_anti_stall_tilt(self):
        m = Gen3RewardManager(config=RewardConfig())
        self.assertEqual(m._active_bias, frozenset({"no_progress_tax"}))

    def test_no_all_shaping_pbrs_activates_the_full_bias_class(self):
        m = Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=False))
        bias = set(RewardBreakdown.registry_fields(RewardClass.BIAS))
        # Everything except the two weight-gated terms (both default to weight 0) and
        # no_progress_tax (charged only under bias_redesign / all_shaping_pbrs).
        self.assertEqual(
            bias - m._active_bias,
            {"stay_risk_tax", "escape_risk_bonus", "self_ko_penalty", "no_progress_tax"},
        )

    def test_stall_pbrs_zeroes_the_whole_bias_class(self):
        m = Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=True, stall_pbrs=True))
        self.assertEqual(m._active_bias, frozenset())

    def test_the_active_set_covers_every_bias_field_and_nothing_else(self):
        """A term that is neither active nor a declared BIAS field would be skipped forever."""
        bias = set(RewardBreakdown.registry_fields(RewardClass.BIAS))
        for label, cfg in _COMPOSITIONS.items():
            with self.subTest(composition=label):
                m = Gen3RewardManager(config=cfg)
                self.assertTrue(m._active_bias <= bias)

    def test_bias_active_gate_reads_the_active_set(self):
        m = Gen3RewardManager(config=RewardConfig())
        self.assertTrue(m._bias_active("no_progress_tax"))
        self.assertFalse(m._bias_active("finishing_blow"))
        # Multi-field form: ANY member active is enough (the pivot family shares one helper).
        self.assertTrue(m._bias_active("finishing_blow", "no_progress_tax"))
        self.assertFalse(m._bias_active("finishing_blow", "roar", "se_switch"))

    def test_shadow_manager_never_skips(self):
        """The twin is the full-computation oracle — its gate must be unconditionally open."""
        shadow = Gen3RewardManager(config=RewardConfig(), _shadow=True)
        self.assertFalse(shadow._skip_inactive_bias)
        self.assertTrue(shadow._bias_active("finishing_blow"))
        self.assertIsNone(shadow._verify_twin)

    def test_every_registry_bias_field_has_an_activeness_answer(self):
        """`_bias_term_active` must be total over the BIAS class — a name it doesn't know would
        fall through to its `return True` default and quietly never be skipped."""
        for name in RewardBreakdown.registry_fields(RewardClass.BIAS):
            with self.subTest(field=name):
                self.assertIsInstance(_bias_term_active(RewardConfig(), name), bool)


class TestBreakdownMicros(unittest.TestCase):
    """`registry_fields` and `total` are now memoized; the memos must equal the derivations."""

    def test_every_breakdown_field_is_declared_float(self):
        """`total` sums the cached NAME tuple with no per-value isinstance guard, which is only
        equivalent to the old `isinstance(v, float)` filter because every field is a float."""
        for f in dataclasses.fields(RewardBreakdown):
            with self.subTest(field=f.name):
                self.assertIn(f.type, ("float", float),
                              f"{f.name} is not declared float — `total` would now sum it "
                              f"where it used to skip it")

    def test_field_names_matches_dataclasses_fields(self):
        self.assertEqual(RewardBreakdown.field_names(),
                         tuple(f.name for f in dataclasses.fields(RewardBreakdown)))

    def test_registry_fields_memo_matches_the_registry(self):
        for cls in RewardClass:
            # subTest kwargs are serialized by xdist — pass the NAME, not the enum member.
            with self.subTest(reward_class=cls.name):
                self.assertEqual(
                    RewardBreakdown.registry_fields(cls),
                    tuple(n for n, c in RewardBreakdown._REGISTRY.items() if c is cls))
        # Cached call returns the same object, not a rebuilt tuple.
        self.assertIs(RewardBreakdown.registry_fields(RewardClass.BIAS),
                      RewardBreakdown.registry_fields(RewardClass.BIAS))

    def test_total_sums_every_field(self):
        bd = RewardBreakdown()
        for i, name in enumerate(RewardBreakdown.field_names()):
            setattr(bd, name, float(i))
        n = len(RewardBreakdown.field_names())
        self.assertEqual(bd.total, float(n * (n - 1) // 2))

    def test_total_includes_bias_refund(self):
        bd = RewardBreakdown()
        bd.win_loss = 1.0
        bd.bias_refund = -0.25
        self.assertEqual(bd.total, 0.75)

    def test_registry_covers_every_field_except_the_refund_mechanism(self):
        """Unchanged contract, re-pinned here because the memos are derived from it."""
        declared = set(RewardBreakdown.field_names())
        registered = set(RewardBreakdown._REGISTRY)
        self.assertEqual(declared - registered, {"bias_refund"})
        self.assertEqual(registered - declared, set())


class TestSharedOppBoostSum(unittest.TestCase):
    """Φ_opp_boosts and Φ_roar are the same Σ at two weights; passing it in must not change either."""

    @staticmethod
    def _live(boosts):
        active = SimpleNamespace(boosts=boosts)
        return SimpleNamespace(opp=SimpleNamespace(active=active))

    def test_precomputed_sum_equals_the_recomputed_one(self):
        m = Gen3RewardManager(config=RewardConfig())
        for boosts in ({}, {"atk": 2}, {"atk": 2, "spe": 1, "def": -3}, {"atk": -6},
                       {"atk": 6, "spa": 6, "spe": 6}):
            with self.subTest(boosts=boosts):
                live = self._live(boosts)
                shared = m._opp_positive_boost_stages(live)
                self.assertEqual(m._compute_phi_opp_boosts(live, shared),
                                 m._compute_phi_opp_boosts(live))
                self.assertEqual(m._compute_phi_roar(live, shared),
                                 m._compute_phi_roar(live))

    def test_no_opp_active_is_zero_either_way(self):
        m = Gen3RewardManager(config=RewardConfig())
        live = SimpleNamespace(opp=SimpleNamespace(active=None))
        self.assertEqual(m._opp_positive_boost_stages(live), 0)
        self.assertEqual(m._compute_phi_opp_boosts(live, 0), 0.0)
        self.assertEqual(m._compute_phi_roar(live, 0), 0.0)


class TestShadowVerifyMode(unittest.TestCase):
    """`GEN3AI_REWARD_VERIFY=1` must (a) build a twin and (b) actually catch a divergence.

    A shadow that cannot fail is worth nothing, so the second test INJECTS a divergence rather
    than trusting that a passing comparison means the comparison works.
    """

    def test_no_twin_without_the_env_var(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEN3AI_REWARD_VERIFY", None)
            self.assertIsNone(Gen3RewardManager(config=RewardConfig())._verify_twin)

    def test_env_var_builds_a_full_computation_twin_on_the_same_config(self):
        cfg = RewardConfig(all_shaping_pbrs=False)
        with patch.dict(os.environ, {"GEN3AI_REWARD_VERIFY": "1"}):
            m = Gen3RewardManager(config=cfg)
        self.assertIsNotNone(m._verify_twin)
        self.assertIs(m._verify_twin.config, cfg)
        self.assertFalse(m._verify_twin._skip_inactive_bias)
        self.assertIsNone(m._verify_twin._verify_twin)   # no infinite recursion

    def test_the_comparison_raises_on_an_injected_divergence(self):
        with patch.dict(os.environ, {"GEN3AI_REWARD_VERIFY": "1"}):
            m = Gen3RewardManager(config=RewardConfig())
        twin_bd = RewardBreakdown()
        twin_bd.finishing_blow = 0.5              # the term the fast path would have skipped
        m._verify_twin._last_breakdown = twin_bd
        m._verify_twin.process_turn_reward = lambda *_a, **_k: twin_bd.total
        with self.assertRaises(RewardVerifyMismatch) as ctx:
            _verify(m, SimpleNamespace(turn=7), RewardBreakdown())
        self.assertIn("finishing_blow", str(ctx.exception))

    def test_the_comparison_passes_when_the_two_agree(self):
        with patch.dict(os.environ, {"GEN3AI_REWARD_VERIFY": "1"}):
            m = Gen3RewardManager(config=RewardConfig())
        bd = RewardBreakdown()
        bd.pbrs_material = -0.125
        twin_bd = RewardBreakdown()
        twin_bd.pbrs_material = -0.125
        m._verify_twin._last_breakdown = twin_bd
        m._verify_twin.process_turn_reward = lambda *_a, **_k: twin_bd.total
        _verify(m, SimpleNamespace(turn=7), bd)   # must not raise

    def test_reset_and_record_action_drive_the_twin(self):
        """A twin whose cross-turn counters drift would flag state divergence as a skip bug."""
        with patch.dict(os.environ, {"GEN3AI_REWARD_VERIFY": "1"}):
            m = Gen3RewardManager(config=RewardConfig())
        m.switch_count = m._verify_twin.switch_count = 4
        m.reset()
        self.assertEqual(m._verify_twin.switch_count, 0)
        calls = []
        m._verify_twin.record_action = lambda ctx, a: calls.append(a)
        import numpy as np
        ctx = SimpleNamespace(mask=np.ones(11, dtype=np.int8), our_slot_map={"pikachu": 0},
                              opp_slot_map={"raichu": 0}, our_active="pikachu",
                              opp_active="raichu", our_hp=np.ones(6), opp_hp=np.ones(6),
                              our_boosts=np.zeros(7, dtype=np.int8), turn=3, phase="move")
        m.record_action(ctx, 6)
        self.assertEqual(calls, [6])


if __name__ == "__main__":
    unittest.main()
