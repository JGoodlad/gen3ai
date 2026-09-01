"""The reward REGISTRY and RewardConfig as the single source of truth
(design_markovian_reward_and_features.md §1.1): the class registry is exhaustive and
non-overlapping, `from_args`/`from_dict` round-trip, and PBRS_GAMMA matches the PPO discount.
Pure (no battle sim) — the shared fakes live in `reward_test_fakes.py`.
"""
import unittest
from dataclasses import fields

from agents.training.reward_manager import RewardConfig, RewardClass, RewardBreakdown, PBRS_GAMMA


class TestRewardRegistry(unittest.TestCase):
    """The registry (design §1.1) is exhaustive, non-overlapping, and drives the fold."""

    def test_every_float_field_has_exactly_one_class(self):
        bd = RewardBreakdown()
        float_fields = {f.name for f in fields(bd) if isinstance(getattr(bd, f.name), float)}
        reg = set(bd._REGISTRY)
        # Only bias_refund (the fold mechanism, not a term) is outside the registry.
        self.assertEqual(float_fields - reg, {"bias_refund"})
        self.assertEqual(reg - float_fields, set())

    def test_groups_cover_every_registry_field_exactly_once(self):
        """Pin the THIRD parallel list (_GROUPS, used by to_dict's log line) against the registry —
        a field added to the registry but missed in _GROUPS would silently vanish from the breakdown
        log with no other test failure (design-review DRY finding #5)."""
        bd = RewardBreakdown()
        grouped = [f for _name, fields_ in bd._GROUPS for f in fields_]
        self.assertEqual(len(grouped), len(set(grouped)), "a field appears in two _GROUPS buckets")
        # Every registry term + the bias_refund mechanism must appear in exactly one bucket.
        expected = set(bd._REGISTRY) | {"bias_refund"}
        self.assertEqual(set(grouped), expected)

    def test_class_membership_matches_design(self):
        bd = RewardBreakdown()
        self.assertEqual(set(bd.registry_fields(RewardClass.TERMINAL)), {"win_loss"})
        self.assertEqual(set(bd.registry_fields(RewardClass.PBRS)),
                         {"pbrs_material", "pbrs_belief", "pbrs_status",
                          "pbrs_progress", "pbrs_hazard", "pbrs_boost", "pbrs_opp_boosts", "pbrs_roar"})
        # Everything else is BIAS.
        self.assertIn("no_progress_tax", bd.registry_fields(RewardClass.BIAS))
        self.assertIn("roar", bd.registry_fields(RewardClass.BIAS))

    def test_total_sums_all_terms(self):
        bd = RewardBreakdown(win_loss=30.0, pbrs_material=-1.5, roar=0.2,
                             no_progress_tax=-0.15, bias_refund=0.05)
        self.assertAlmostEqual(bd.total, 30.0 - 1.5 + 0.2 - 0.15 + 0.05, places=6)


class TestPbrsGammaInvariant(unittest.TestCase):
    def test_pbrs_gamma_default_matches_ppo(self):
        # The train_rl_agent assert pins PBRS_GAMMA == model.gamma (0.9999) at build time.
        self.assertAlmostEqual(PBRS_GAMMA, 0.9999, places=9)
        self.assertAlmostEqual(RewardConfig().gamma, PBRS_GAMMA, places=9)


class TestRewardConfigSingleSource(unittest.TestCase):
    """RewardConfig is the single source of truth: `from_args` builds it ONCE from the CLI, `from_dict`
    reconstructs it from model_config.json for eval/resume. So adding a reward flag (field + matching
    CLI arg) flows everywhere with no hand-threading — the gap that once let eval silently measure with
    a default (bias_redesign=False) reward (the ai_v5_6 mismeasurement)."""

    def test_from_args_pulls_every_field_by_name(self):
        from types import SimpleNamespace
        args = SimpleNamespace(bias_additivity=0.5, mat_alive_weight=1.4, no_progress_penalty=0.3,
                               switch_bias_weight=2.0, bias_redesign=True, draw_penalty=-35.0,
                               unrelated="ignored")
        rc = RewardConfig.from_args(args)
        self.assertEqual((rc.bias_additivity, rc.mat_alive_weight, rc.no_progress_penalty,
                          rc.switch_bias_weight, rc.bias_redesign, rc.draw_penalty),
                         (0.5, 1.4, 0.3, 2.0, True, -35.0))
        self.assertAlmostEqual(rc.gamma, 0.9999)   # fixed PPO discount, never from args

    def test_from_args_defaults_match_explicit(self):
        """The explicit values here ARE the dataclass defaults — including the two flipped on
        2026-08-18 (`all_shaping_pbrs=True`, `draw_penalty=-35.0`). `reward_defaults_test.py`
        separately pins that the argparse defaults agree with these."""
        from types import SimpleNamespace
        args = SimpleNamespace(bias_additivity=1.0, mat_alive_weight=1.25, no_progress_penalty=0.15,
                               switch_bias_weight=0.0, bias_redesign=False, draw_penalty=-35.0,
                               all_shaping_pbrs=True, stall_pbrs=False)
        self.assertEqual(RewardConfig.from_args(args), RewardConfig())

    def test_from_dict_round_trips(self):
        from dataclasses import asdict
        rc = RewardConfig(bias_redesign=True, draw_penalty=-35.0, no_progress_penalty=0.25,
                          switch_bias_weight=1.5, bias_additivity=0.5)
        self.assertEqual(RewardConfig.from_dict(asdict(rc)), rc)

    def test_from_dict_ignores_unknown_and_defaults_missing(self):
        # a real model_config.json carries arch fields + use_popart (ignored), and may PREDATE a
        # reward field (defaulted) — reconstruction must never KeyError on a stray/missing key.
        rc = RewardConfig.from_dict({"bias_redesign": True, "arch_signature": "x", "use_popart": True})
        self.assertTrue(rc.bias_redesign)
        self.assertEqual(rc.draw_penalty, RewardConfig().draw_penalty)   # absent → default
        self.assertEqual(RewardConfig.from_dict(None), RewardConfig())   # None → all defaults

    def test_eval_player_and_builder_require_reward_factory(self):
        """No silent default: a missing reward factory must be a loud TypeError, not a default reward
        config — that default is exactly what mismeasured eval. (Signature guard; no heavy construction.)"""
        import inspect
        from agents.training.eval_callback import EvalRLPlayer, build_eval_players
        for fn, pname in ((EvalRLPlayer.__init__, "reward_fn_factory"),
                          (build_eval_players, "reward_fn_factory")):
            p = inspect.signature(fn).parameters[pname]
            self.assertIs(p.default, inspect.Parameter.empty,
                          f"{fn.__qualname__}.{pname} must be REQUIRED (no silent default)")


if __name__ == "__main__":
    unittest.main()
