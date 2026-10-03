"""THE CRITIC -> DISCOUNT PAIRING (`agents.model.critic_mode.critic_gamma`) — cutover loose end 2.

(Deletion pass U3: `--critic shaped` is refused at the PARSER now — the Python env core served it — so
the shaped half of this file's cases, which TYPED it, went with it; `critic_gamma(shaped)` stays declared
and pinned below because an OLD shaped checkpoint still loads. The launch guard for an untyped gamma from
the other critic has no reachable case any more: a census candidate.)

`--arch production --critic shaped` used to resolve `--gamma` to recipe.fresh's 1.0: the recipe
applied every untyped row as-is, and gamma is the winprob critic's IDENTITY value, not a recipe knob.
Every shaped and pre-critic run in `models/` trained at 0.9999 (208 of them, the shaped ladder
controls `ai_v12_26/27` included, which otherwise took the win-prob reward values); every winprob run
at 1.0. Never a shaped critic at 1.0. So the discount is now PAIRED with the critic, declared once,
applied by the recipe, REFUSED under winprob when typed otherwise, and guarded at the launch when
untyped otherwise. Every test here fails on a revert of the piece it names.
"""
from __future__ import annotations

import copy

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


def test_the_untyped_production_critic_keeps_recipe_fresh_gamma(capsys):
    ns = _resolve(["--arch", "production"])
    assert ns.critic == "winprob" and ns.gamma == 1.0 == rs.production_recipe()["gamma"]
    assert "[Critic] gamma=1 — the --critic winprob pairing" in capsys.readouterr().out


def test_a_typed_gamma_is_announced_as_typed(capsys):
    ns = _resolve(["--arch", "production", "--gamma", "1.0"])
    assert ns.gamma == 1.0
    assert "[Critic] gamma=1 — TYPED (the --critic winprob pairing is 1)" in capsys.readouterr().out


def test_winprob_refuses_a_typed_gamma_other_than_one():
    from main.checkargs import check

    names = [c.name for c, _ in check(["--steps", "1", "--arch", "production", "--gamma", "0.99"])["combinations"]]
    assert "winprob_critic_needs_unit_gamma" in names
    ok = [c.name for c, _ in check(["--steps", "1", "--arch", "production", "--gamma", "1.0"])["combinations"]]
    assert "winprob_critic_needs_unit_gamma" not in ok


def test_a_recipe_block_whose_gamma_is_not_its_critics_is_refused():
    raw = rs._raw_mirror()
    bad = copy.deepcopy(raw)
    bad["recipe"]["fresh"]["gamma"] = 0.9999                     # winprob critic, shaped discount
    with pytest.raises(rs.RecipeError, match="gamma"):
        rs.production_recipe(bad)
