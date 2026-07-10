"""Gen3Env construction unit tests — the opponent-team seam (the training-mirror bug fix).

PokeEnv passes its single ``team=`` kwarg to BOTH internal ``_EnvPlayer``s, and the per-episode
opponent Players are pure decision functions over ``battle2`` (agent2 does the networking), so
agent2's ``_team`` is what actually decides the opponent's team. Before the ``opponent_team`` seam,
a ``--trainee-team`` pin therefore silently pinned the OPPONENTS to the trainee's team too — every
specialist run trained in a single-team MIRROR vs bot pilots (the root cause of the inflated ~100%
training win rates). These pin the seam."""

import pytest

from poke_env import AccountConfiguration

from agents.observation.state_encoder import load_mappings
from agents.training.gen3_env import Gen3Env
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder


@pytest.fixture(scope="module")
def builders():
    all_teams = TeamLoader().get_all_teams()
    with open("data/teams/specialist/tss_starmie.txt", encoding="utf-8") as f:
        tss = Gen3Teambuilder([f.read()])
    return tss, Gen3Teambuilder(all_teams)


def _env(**kw):
    return Gen3Env(load_mappings(), battle_format="gen3ou",
                   account_configuration1=AccountConfiguration("EnvTeamT", None),
                   start_listening=False, **kw)


def test_opponent_team_seam_splits_the_sides(builders):
    tss, pool = builders
    env = _env(team=tss, opponent_team=pool)
    assert env.agent1._team is tss          # the trainee pilots the pinned team…
    assert env.agent2._team is pool         # …the OPPONENT side draws the diverse pool
    assert env.agent1._team is not env.agent2._team


def test_default_keeps_both_sides_on_team(builders):
    """opponent_team=None → the pre-fix behavior (both sides `team=`), byte-identical — correct for
    non-pinned runs where both sides draw the same pool anyway."""
    tss, _ = builders
    env = _env(team=tss)
    assert env.agent1._team is tss and env.agent2._team is tss
