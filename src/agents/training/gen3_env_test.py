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


# ── distill_mask: a teacher may own MANY teams (multi-team z-cluster exploiter) ────────────────

def _mask_env(distill_team_species):
    """A Gen3Env shell exercising ONLY the _distill_mask matching (no PokeEnv construction)."""
    env = Gen3Env.__new__(Gen3Env)
    env._distill_team_species = [
        [sp] if isinstance(sp, (set, frozenset)) else list(sp) for sp in (distill_team_species or [])
    ]
    env._distill_team_id = None
    return env


def _battle_with(species):
    class _M:
        def __init__(self, s): self.species = s
    class _B:
        team = None
    b = _B()
    b.team = {s: _M(s) for s in species}
    return b


A = ["skarmory", "blissey", "tyranitar", "swampert", "gengar", "starmie"]
B = ["salamence", "hariyama", "claydol", "suicune", "jirachi", "blissey"]
C = ["celebi", "charizard", "metagross", "swampert", "tyranitar", "zapdos"]


def _fs(names):
    return frozenset(names)


def test_distill_mask_multi_team_teacher_fires_on_any_of_its_teams():
    # ONE teacher owning teams A and B → both map to teacher-id 1 (a multi-team exploiter teacher).
    env = _mask_env([[_fs(A), _fs(B)]])
    env.battle1 = _battle_with(A)
    assert env._distill_mask() == 1.0
    env._distill_team_id = None
    env.battle1 = _battle_with(B)
    assert env._distill_mask() == 1.0
    env._distill_team_id = None
    env.battle1 = _battle_with(C)          # not this teacher's → 0
    assert env._distill_mask() == 0.0


def test_distill_mask_teacher_ids_are_1_indexed_positions():
    env = _mask_env([[_fs(A)], [_fs(B), _fs(C)]])
    env.battle1 = _battle_with(A)
    assert env._distill_mask() == 1.0      # teacher 1
    env._distill_team_id = None
    env.battle1 = _battle_with(C)
    assert env._distill_mask() == 2.0      # teacher 2 (its SECOND team)


def test_distill_mask_accepts_the_legacy_bare_set_shape():
    # back-compat: a bare frozenset per teacher (the pre-multi-team form) is wrapped and still matches.
    env = _mask_env([_fs(A), _fs(B)])
    env.battle1 = _battle_with(B)
    assert env._distill_mask() == 2.0
