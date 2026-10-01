"""THE CRITIC -> DISCOUNT PAIRING (`agents.model.critic_mode.critic_gamma`) — cutover loose end 2.

`--arch production --critic shaped` used to resolve `--gamma` to recipe.fresh's 1.0: the recipe
applied every untyped row as-is, and gamma is the winprob critic's IDENTITY value, not a recipe knob.
Every shaped and pre-critic run in `models/` trained at 0.9999 (208 of them, the shaped ladder
controls `ai_v12_26/27` included, which otherwise took the win-prob reward values); every winprob run
at 1.0. Never a shaped critic at 1.0. So the discount is now PAIRED with the critic, declared once,
applied by the recipe, REFUSED under winprob when typed otherwise, and guarded at the launch when
untyped otherwise. Every test here fails on a revert of the piece it names.
"""
from __future__ import annotations

import contextlib
import copy
import io

import pytest

from main.train import recipe_surface as rs
from main.train.parser import build_parser


def _resolve(argv, capsys=None):
    from main.train.config import resolve_config

    p = build_parser()
    ns = p.parse_args(["--steps", "1", *argv])
    resolve_config(ns, p)
    return ns


def test_the_pairing_is_declared_once():
    from agents.model.critic_mode import CRITIC_SHAPED, CRITIC_WINPROB, critic_gamma
    from agents.training.reward_weights import PBRS_GAMMA

    assert critic_gamma(CRITIC_WINPROB) == 1.0
    assert critic_gamma(CRITIC_SHAPED) == PBRS_GAMMA == 0.9999


def test_a_typed_shaped_critic_under_arch_production_gets_its_own_gamma(capsys):
    # `--env-core python`: the production core (rust, the M5 switch) refuses a shaped critic (F-LD-2)
    ns = _resolve(["--arch", "production", "--critic", "shaped", "--env-core", "python"])
    assert ns.critic == "shaped" and ns.gamma == 0.9999          # was 1.0: recipe.fresh's winprob value
    assert "[Critic] gamma=0.9999 — the --critic shaped pairing" in capsys.readouterr().out


def test_the_untyped_production_critic_keeps_recipe_fresh_gamma(capsys):
    ns = _resolve(["--arch", "production"])
    assert ns.critic == "winprob" and ns.gamma == 1.0 == rs.production_recipe()["gamma"]
    assert "[Critic] gamma=1 — the --critic winprob pairing" in capsys.readouterr().out


def test_a_typed_gamma_is_announced_as_typed(capsys):
    ns = _resolve(["--arch", "production", "--critic", "shaped", "--gamma", "0.999", "--env-core", "python"])
    assert ns.gamma == 0.999
    assert "[Critic] gamma=0.999 — TYPED (the --critic shaped pairing is 0.9999)" in capsys.readouterr().out


def test_the_recipe_surface_reports_the_paired_gamma_without_refusing():
    from main.checkargs import check

    rep = check(["--steps", "1", "--arch", "production", "--critic", "shaped"])["recipe"]
    by = {d.dest: d for d in rep.diffs}
    assert by["critic"].source == "argv"
    assert by["gamma"].source == "paired" and by["gamma"].resolved == 0.9999
    assert "paired with the TYPED --critic" in by["gamma"].line()
    assert not rep.silent and not rep.refuses


def test_winprob_refuses_a_typed_gamma_other_than_one():
    from main.checkargs import check

    names = [c.name for c, _ in check(["--steps", "1", "--arch", "production", "--gamma", "0.99"])["combinations"]]
    assert "winprob_critic_needs_unit_gamma" in names
    ok = [c.name for c, _ in check(["--steps", "1", "--arch", "production", "--gamma", "1.0"])["combinations"]]
    assert "winprob_critic_needs_unit_gamma" not in ok
    shaped = [c.name for c, _ in check(["--steps", "1", "--arch", "production", "--critic", "shaped",
                                         "--gamma", "0.99"])["combinations"]]
    assert "winprob_critic_needs_unit_gamma" not in shaped     # the shaped critic's gamma is its tunable


def test_a_recipe_block_whose_gamma_is_not_its_critics_is_refused():
    raw = rs._raw_mirror()
    bad = copy.deepcopy(raw)
    bad["recipe"]["fresh"]["gamma"] = 0.9999                     # winprob critic, shaped discount
    with pytest.raises(rs.RecipeError, match="gamma"):
        rs.production_recipe(bad)


def test_the_launch_guard_refuses_an_untyped_gamma_from_the_other_critic(monkeypatch):
    """The pre-fix applier (every recipe row as-is) put winprob's 1.0 under a TYPED shaped critic;
    the launch now refuses that rather than training it."""
    from main.exit_codes import TrainExitCode

    real = rs.apply_production_recipe

    def pre_fix_applier(ns, mirror=None):
        out = real(ns, mirror)
        ns.gamma = rs.production_recipe(mirror)["gamma"]
        return out

    monkeypatch.setattr(rs, "apply_production_recipe", pre_fix_applier)
    with contextlib.redirect_stderr(io.StringIO()) as err, pytest.raises(SystemExit) as e:
        _resolve(["--arch", "production", "--critic", "shaped"])
    assert e.value.code == int(TrainExitCode.FATAL_CONFIG)
    assert "nobody typed it" in err.getvalue()
