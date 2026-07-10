"""Eval-worker unit tests — the trainee-teambuilder selection (the specialist eval-alignment fix).

The worker used to HARDCODE the default full-pool trainee teambuilder, so a `--trainee-team` run's
eval (win rates / ELO / vs-ext verdicts) measured the model piloting RANDOM teams it never trained
on — the ai_v7_05–08 "plateau" was this out-of-distribution measurement, not the training. These
pin the fix: `cfg['trainee_team_str']` → the pinned single-team builder; absent → the default,
byte-identical to the old behavior."""

import pytest

from main.eval_worker import _build_trainee_tb, _fixed_opponent_tb
from utils.team_loader import TeamLoader


@pytest.fixture(scope="module")
def teams():
    loader = TeamLoader()
    return loader.get_all_teams(), loader.get_sample_teams()


@pytest.fixture(scope="module")
def tss_str():
    with open("data/teams/specialist/tss_starmie.txt", encoding="utf-8") as f:
        return f.read()


def test_pinned_team_builds_single_team_builder(teams, tss_str):
    all_teams, sample_teams = teams
    tb = _build_trainee_tb({"trainee_team_str": tss_str}, all_teams, sample_teams)
    assert len(tb.packed_teams) == 1              # eval pilots EXACTLY the trained team
    assert not tb.bias_packed_teams
    # every draw is that team
    assert len({tb.yield_team() for _ in range(5)}) == 1


def test_absent_key_is_the_default_pool(teams):
    all_teams, sample_teams = teams
    tb = _build_trainee_tb({}, all_teams, sample_teams)
    assert len(tb.packed_teams) == len(all_teams)  # the full pool (old behavior, byte-identical)
    assert tb.bias_prob == pytest.approx(0.1)


def test_none_value_is_the_default_pool(teams):
    """The callbacks always send the key (None when no pin) — None must mean default, not crash."""
    all_teams, sample_teams = teams
    tb = _build_trainee_tb({"trainee_team_str": None}, all_teams, sample_teams)
    assert len(tb.packed_teams) == len(all_teams)


# ── fold-back: a FIXED opponent's own pinned team ─────────────────────────────

class _Item:
    def __init__(self, team_str=None):
        self.team_str = team_str


def test_pinned_fixed_opponent_pilots_its_own_team(tss_str):
    tb = _fixed_opponent_tb(_Item(team_str=tss_str), opp_tb="POOL")
    assert len(tb.packed_teams) == 1
    assert len({tb.yield_team() for _ in range(5)}) == 1


def test_unpinned_fixed_opponent_keeps_the_pool_builder():
    pool = object()
    assert _fixed_opponent_tb(_Item(team_str=None), opp_tb=pool) is pool
    assert _fixed_opponent_tb(object(), opp_tb=pool) is pool   # pre-team_str item (no attr)
