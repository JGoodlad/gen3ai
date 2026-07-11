"""MatchupSpec unit tests — CLI mapping, builder parity with the legacy construction, provenance."""
from types import SimpleNamespace

import pytest

from agents.training.matchup_spec import (
    DEFAULT_TRAINEE_BIAS_PROB, MatchupSpec, PlayMode, TeamSource,
)
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder


@pytest.fixture(scope="module")
def teams():
    loader = TeamLoader()
    return loader.get_all_teams(), loader.get_sample_teams()


def _args(**kw):
    base = dict(trainee_team=None, exploiter=None, self_play=False, bot_weights=None,
                exploiter_keep_bots=False, exploiter_bot_fraction=0.5,
                exploiter_temp_start=None, exploiter_temp_mode="fixed",
                stable_opponent_temp=1.0)
    base.update(kw)
    return SimpleNamespace(**base)


# ── builder parity: the spec must reproduce the legacy constructions byte-for-byte ─

def test_default_spec_builders_match_legacy(teams):
    all_teams, sample_teams = teams
    spec = MatchupSpec.from_args(_args())
    tb = spec.trainee_teams.build(all_teams, sample_teams)
    legacy = Gen3Teambuilder(all_teams, bias_teams=sample_teams, bias_prob=DEFAULT_TRAINEE_BIAS_PROB)
    assert tb.packed_teams == legacy.packed_teams
    assert tb.bias_packed_teams == legacy.bias_packed_teams and tb.bias_prob == legacy.bias_prob
    ob = spec.opponent_teams.build(all_teams, sample_teams)
    assert ob.packed_teams == Gen3Teambuilder(all_teams).packed_teams and not ob.bias_packed_teams


def test_pinned_spec_builder_matches_legacy(teams):
    all_teams, sample_teams = teams
    spec = MatchupSpec.from_args(_args(trainee_team="data/teams/specialist/tss_starmie.txt"))
    tb = spec.trainee_teams.build(all_teams, sample_teams)
    assert len(tb.packed_teams) == 1
    with open("data/teams/specialist/tss_starmie.txt", encoding="utf-8") as f:
        assert tb.packed_teams == Gen3Teambuilder([f.read()]).packed_teams
    # the two sides are independent BY CONSTRUCTION (the mirror-bug class)
    ob = spec.opponent_teams.build(all_teams, sample_teams)
    assert len(ob.packed_teams) == len(all_teams) > 1


def test_pin_biased_kind_builds_team_prob(teams):
    """The future --trainee-team-prob shape: pinned team bias_prob of the time, else the pool."""
    all_teams, sample_teams = teams
    with open("data/teams/specialist/tss_starmie.txt", encoding="utf-8") as f:
        pin = f.read()
    tb = TeamSource(kind="pin_biased", pin_str=pin, bias_prob=0.5).build(all_teams, sample_teams)
    assert len(tb.packed_teams) == len(all_teams)
    assert len(tb.bias_packed_teams) == 1 and tb.bias_prob == 0.5


# ── CLI mapping ────────────────────────────────────────────────────────────────

def test_from_args_mix_kinds():
    assert MatchupSpec.from_args(_args()).mix_kind == "bots"
    assert MatchupSpec.from_args(_args(self_play=True)).mix_kind == "self_play"
    spec = MatchupSpec.from_args(_args(exploiter="models/x", exploiter_keep_bots=True))
    assert spec.mix_kind == "exploiter" and spec.exploiter_target == "models/x"
    assert spec.exploiter_keep_bots


def test_from_args_play_mode_ratchet():
    spec = MatchupSpec.from_args(_args(exploiter="models/x", exploiter_temp_start=5.0,
                                       exploiter_temp_mode="ratchet"))
    assert spec.opponent_play == PlayMode(kind="stochastic", temperature=5.0, schedule="ratchet")


def test_eval_trainee_defaults_to_trainee(teams):
    spec = MatchupSpec.from_args(_args(trainee_team="data/teams/specialist/tss_starmie.txt"))
    assert spec.eval_trainee_teams is spec.trainee_teams   # eval pilots what training pilots


# ── provenance ─────────────────────────────────────────────────────────────────

def test_hash_stable_and_regime_sensitive():
    a = MatchupSpec.from_args(_args())
    b = MatchupSpec.from_args(_args())
    assert a.spec_hash() == b.spec_hash()
    c = MatchupSpec.from_args(_args(trainee_team="data/teams/specialist/tss_starmie.txt"))
    assert c.spec_hash() != a.spec_hash()                  # different regime, different tag
    d = a.to_dict()
    assert d["trainee_teams"]["kind"] == "default_biased" and d["opponent_teams"]["kind"] == "pool"


def test_summary_lines_echo_the_essentials():
    spec = MatchupSpec.from_args(_args(exploiter="models/x", exploiter_keep_bots=True,
                                       exploiter_temp_start=5.0, exploiter_temp_mode="ratchet",
                                       trainee_team="data/teams/specialist/tss_starmie.txt"))
    text = "\n".join(spec.summary_lines())
    for token in ("MATCHUP", "PINNED", "full pool", "exploiter", "models/x", "ratchet", "eval:"):
        assert token in text, f"echo missing {token!r}:\n{text}"


def test_bad_kind_and_missing_pin_raise():
    with pytest.raises(ValueError):
        TeamSource(kind="bogus")
    with pytest.raises(ValueError):
        TeamSource(kind="pinned")


def test_describe_drift_names_changed_fields():
    from agents.training.matchup_spec import describe_drift
    a = MatchupSpec.from_args(_args()).to_dict()
    b = MatchupSpec.from_args(_args(exploiter="models/x")).to_dict()
    lines = describe_drift(a, b)
    assert any(l.startswith("mix_kind:") for l in lines)
    assert any(l.startswith("exploiter_target:") for l in lines)
    assert describe_drift(a, dict(a)) == []            # no drift → no lines
    assert describe_drift(None, None) == []
