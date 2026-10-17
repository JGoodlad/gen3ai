"""Our POOL team source must not depend on the working directory (2026-10-17 FINDING).

`TeamLoader()`'s default base dir is the CWD-relative ``"data/teams"``. The X26 anchor read ran
`python -m main.anchors --teamset home` from ``src/``: the loader found ZERO teams, and
`Gen3Teambuilder([])` died with an `IndexError` out of its own error message, so every unit failed
at once as ``our_side_error`` (5 / 5 units, 10 attempts). These tests run from a foreign CWD.
"""
from __future__ import annotations

import pytest

from main.anchors.session import SeriesFailure, build_team_source, pool_team_texts


def test_the_pool_resolves_from_a_FOREIGN_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                        # no data/teams under here
    teams = pool_team_texts()
    assert len(teams) > 100, len(teams)                # the 719-team pool, not an empty list


@pytest.mark.integration                               # the builder validates through Node
def test_the_pool_team_source_DRAWS_a_team_from_a_FOREIGN_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    draws: list = []
    tb = build_team_source({"kind": "pool"}, 0, draws)
    assert tb.yield_team()                             # reverted: IndexError at construction
    assert draws and draws[0] is not None


def test_an_EMPTY_pool_is_a_NAMED_refusal(tmp_path, monkeypatch):
    import utils.paths

    monkeypatch.setattr(utils.paths, "repo_path", lambda *parts: tmp_path.joinpath(*parts))
    with pytest.raises(SeriesFailure) as exc:
        pool_team_texts()
    assert exc.value.cause == "no_teams"


def test_an_EMPTY_team_list_raises_a_NAMED_ValueError_not_an_IndexError():
    from utils.teambuilder import Gen3Teambuilder

    with pytest.raises(ValueError, match="EMPTY team list"):
        Gen3Teambuilder([], rng_seed=0)
