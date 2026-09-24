"""The shared TEAM-SOURCE hook (`gen3_ladder_usage_corpus_v1`): the pool path is the one every
gate used before, and the other two sources are deterministic."""
import pytest

from utils import ladder_corpus, team_sources
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder


def test_the_pool_source_is_the_pre_existing_pool():
    assert team_sources.team_list("pool") == TeamLoader().get_all_teams()
    pool = TeamLoader().get_all_teams()
    assert team_sources.pair("pool", 5) == (pool[5], pool[6])
    assert isinstance(team_sources.teambuilder("pool"), Gen3Teambuilder)


def test_the_ladder_source_is_the_corpus():
    assert team_sources.team_list("ladder") == ladder_corpus.teams("milestone")
    assert team_sources.team_list("ladder", ladder_tier="commit") == ladder_corpus.teams("commit")
    assert team_sources.pair("ladder", 3) == ladder_corpus.pair(3)


def test_a_packed_pool_draws_from_its_own_seeded_stream():
    teams = ladder_corpus.teams("commit")
    a = team_sources.PackedTeamPool(teams, rng_seed=7)
    b = team_sources.PackedTeamPool(teams, rng_seed=7)
    draws = [a.yield_team() for _ in range(20)]
    assert draws == [b.yield_team() for _ in range(20)]
    assert set(draws) <= set(teams) and len(set(draws)) > 1


def test_an_unknown_source_is_refused():
    with pytest.raises(ValueError):
        team_sources.team_list("randbats")
    with pytest.raises(ValueError):
        team_sources.PackedTeamPool([])


@pytest.mark.integration
def test_the_procedural_source_is_seeded():
    a = team_sources.procedural_teams(3, seed=11)
    assert a == team_sources.procedural_teams(3, seed=11)
    assert len(a) == 3 and all(len(t.split("]")) == 6 for t in a)
