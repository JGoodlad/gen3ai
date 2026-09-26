"""The reward REGISTRY and RewardConfig as the single source of truth: the class registry is
exhaustive and non-overlapping (and TERMINAL-only since the shaped-reward deletion), and
`from_args`/`from_dict` round-trip. Pure (no battle sim).
"""
import unittest
from dataclasses import fields

from agents.training.reward_manager import RewardConfig, RewardClass, RewardBreakdown, PBRS_GAMMA


class TestRewardRegistry(unittest.TestCase):
    """The registry is exhaustive and non-overlapping — and since the shaped-reward deletion
    (2026-09-26) it holds ONE term, the terminal."""

    def test_every_float_field_has_exactly_one_class(self):
        bd = RewardBreakdown()
        float_fields = {f.name for f in fields(bd) if isinstance(getattr(bd, f.name), float)}
        self.assertEqual(float_fields, set(bd._REGISTRY))

    def test_groups_cover_every_registry_field_exactly_once(self):
        bd = RewardBreakdown()
        grouped = [f for _name, fields_ in bd._GROUPS for f in fields_]
        self.assertEqual(len(grouped), len(set(grouped)), "a field appears in two _GROUPS buckets")
        self.assertEqual(set(grouped), set(bd._REGISTRY))

    def test_the_reward_is_the_TERMINAL_alone(self):
        """The shaped classes are GONE, not merely empty: a PBRS / BIAS class reappearing means
        shaping came back, which is a research event, not a refactor."""
        self.assertEqual([c.name for c in RewardClass], ["TERMINAL"])
        self.assertEqual(RewardBreakdown().registry_fields(RewardClass.TERMINAL), ("win_loss",))

    def test_total_is_the_terminal(self):
        self.assertEqual(RewardBreakdown(win_loss=1.0).total, 1.0)
        self.assertEqual(RewardBreakdown().total, 0.0)


class TestPbrsGammaInvariant(unittest.TestCase):
    def test_default_gamma_is_the_historical_ppo_discount(self):
        self.assertAlmostEqual(PBRS_GAMMA, 0.9999, places=9)
        self.assertAlmostEqual(RewardConfig().gamma, PBRS_GAMMA, places=9)


class TestRewardConfigSingleSource(unittest.TestCase):
    """RewardConfig is the single source of truth: `from_args` builds it ONCE from the CLI, `from_dict`
    reconstructs it from model_config.json for eval/resume. So adding a reward flag (field + matching
    CLI arg) flows everywhere with no hand-threading — the gap that once let eval silently measure with
    a default reward (the ai_v5_6 mismeasurement)."""

    def test_from_args_pulls_every_field_by_name(self):
        from types import SimpleNamespace
        args = SimpleNamespace(victory_value=1.0, terminal_indicator=True, draw_penalty=0.0,
                               progress_decision_tense=True, unrelated="ignored")
        rc = RewardConfig.from_args(args)
        self.assertEqual((rc.victory_value, rc.terminal_indicator, rc.draw_penalty,
                          rc.progress_decision_tense), (1.0, True, 0.0, True))
        self.assertAlmostEqual(rc.gamma, 0.9999)   # unset --gamma → the historical discount

    def test_from_args_defaults_match_explicit(self):
        from types import SimpleNamespace
        args = SimpleNamespace(victory_value=30.0, terminal_indicator=False, draw_penalty=-35.0,
                               progress_decision_tense=False, progress_switch_freeze=False)
        self.assertEqual(RewardConfig.from_args(args), RewardConfig())

    def test_from_dict_round_trips(self):
        from dataclasses import asdict
        rc = RewardConfig(terminal_indicator=True, victory_value=1.0, draw_penalty=0.0)
        self.assertEqual(RewardConfig.from_dict(asdict(rc)), rc)

    def test_from_dict_ignores_unknown_and_defaults_missing(self):
        # a real model_config.json carries arch fields + use_popart (ignored) — and a PRE-deletion
        # one carries the deleted shaped fields, which a frozen/eval reconstruction must ignore
        # (the resume/fork path refuses them first: model_version.shaped_reward).
        rc = RewardConfig.from_dict({"terminal_indicator": True, "arch_signature": "x",
                                     "use_popart": True, "hand_shaping": False,
                                     "bias_additivity": 1.0})
        self.assertTrue(rc.terminal_indicator)
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
