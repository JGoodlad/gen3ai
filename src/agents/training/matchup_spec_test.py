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
    base = dict(trainee_team=None, trainee_teams=None, exploiter=None, self_play=False,
                bot_weights=None, exploiter_keep_bots=False, exploiter_bot_fraction=0.5,
                exploiter_temp_start=None, exploiter_temp_mode="fixed",
                stable_opponent_temp=1.0)
    base.update(kw)
    return SimpleNamespace(**base)


# ── multi-team pin (--trainee-teams): the 1-vs-3-team exploiter A/B ─

_K6_TRIO = ("data/teams/sample/9d5f845869e899ee.txt",   # 564b9be3ae
            "data/teams/sample/f7ba5702fe856292.txt",   # 4771662cf7
            "data/teams/sample/0972146213a667c9.txt")   # 45995e432f


def test_pin_multi_builds_the_fixed_set(teams):
    all_teams, sample_teams = teams
    spec = MatchupSpec.from_args(_args(trainee_teams=",".join(_K6_TRIO)))
    ts = spec.trainee_teams
    assert ts.kind == "pin_multi" and len(ts.pin_strs) == 3
    # pin_str mirrors the first member so single-team consumers (eval pin, provenance) still work
    assert ts.pin_str == ts.pin_strs[0]
    tb = ts.build(all_teams, sample_teams)
    expected = [open(f, encoding="utf-8").read() for f in _K6_TRIO]
    assert tb.packed_teams == Gen3Teambuilder(expected).packed_teams and len(tb.packed_teams) == 3
    # provenance records every member's fingerprint
    d = spec.to_dict()["trainee_teams"]
    assert len(d["pin_shas"]) == 3


def test_pin_multi_exploiter_sample_gate(teams):
    from agents.training.matchup_spec import validate_exploiter_trainee_is_sample
    all_teams, sample_teams = teams
    # all-sample trio passes
    spec = MatchupSpec.from_args(_args(trainee_teams=",".join(_K6_TRIO), exploiter="models/x"))
    validate_exploiter_trainee_is_sample(spec, sample_teams)   # no raise
    # a non-sample member is rejected — build a pin_multi with one non-sample team directly
    non_sample = "FakeMon @ Leftovers\nAbility: Levitate\n- Tackle\n"
    bad = MatchupSpec(
        trainee_teams=TeamSource(kind="pin_multi", pin_strs=(sample_teams[0], non_sample),
                                 pin_files=("s0.txt", "bad.txt")),
        opponent_teams=TeamSource(kind="pool"), mix_kind="exploiter")
    with pytest.raises(ValueError, match="NOT one of"):
        validate_exploiter_trainee_is_sample(bad, sample_teams)


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


# ── exploiter team-source guard (only-ever-sample-teams) ─────────────────────────

def _sample_and_other():
    from utils.team_loader import TeamLoader
    loader = TeamLoader()
    sample = loader.get_sample_teams()
    other = [t for t in loader.get_all_teams() if t not in sample]
    return sample, other


def test_exploiter_pinned_sample_team_passes(tmp_path):
    from agents.training.matchup_spec import validate_exploiter_trainee_is_sample
    sample, _ = _sample_and_other()
    pin = tmp_path / "s.txt"
    pin.write_text(sample[0] + "\n\n")                       # raw file w/ trailing whitespace
    spec = MatchupSpec.from_args(_args(exploiter="models/x", trainee_team=str(pin)))
    validate_exploiter_trainee_is_sample(spec, sample)       # no raise — strip-normalized member


def test_exploiter_pinned_nonsample_team_fatals(tmp_path):
    from agents.training.matchup_spec import validate_exploiter_trainee_is_sample
    sample, other = _sample_and_other()
    pin = tmp_path / "o.txt"
    pin.write_text(other[0])                                 # a bulk-downloaded 'other' team
    spec = MatchupSpec.from_args(_args(exploiter="models/x", trainee_team=str(pin)))
    with pytest.raises(ValueError, match="curated SAMPLE teams"):
        validate_exploiter_trainee_is_sample(spec, sample)


def test_non_exploiter_pin_is_unconstrained(tmp_path):
    # a bots/self-play specialist may pin any team — the guard is exploiter-only
    from agents.training.matchup_spec import validate_exploiter_trainee_is_sample
    sample, other = _sample_and_other()
    pin = tmp_path / "o.txt"
    pin.write_text(other[0])
    spec = MatchupSpec.from_args(_args(trainee_team=str(pin)))   # no --exploiter → mix_kind=bots
    validate_exploiter_trainee_is_sample(spec, sample)          # no raise


def test_exploiter_unpinned_trainee_is_out_of_scope():
    from agents.training.matchup_spec import validate_exploiter_trainee_is_sample
    sample, _ = _sample_and_other()
    spec = MatchupSpec.from_args(_args(exploiter="models/x"))   # default_biased trainee, no pin
    validate_exploiter_trainee_is_sample(spec, sample)          # no raise (full-pool exploiter)


def test_tss_specialist_pin_is_a_sample_team():
    # the shipped TSS specialist recipe must keep passing the guard
    from agents.training.matchup_spec import validate_exploiter_trainee_is_sample
    sample, _ = _sample_and_other()
    spec = MatchupSpec.from_args(_args(exploiter="models/x",
                                       trainee_team="data/teams/specialist/tss_starmie.txt"))
    validate_exploiter_trainee_is_sample(spec, sample)


# ── DISTILLATION must eval on the TAUGHT teams (eval-pilots-what-training-pilots) ────────────────

def _distill_args(pairs):
    return _args(self_play=True, _distill_pairs=pairs)


def test_distill_run_evals_the_trainee_on_the_TAUGHT_teams():
    """The distill path biases --distill-team-bias of TRAINING onto the teacher teams, so eval must
    measure the trainee ON THOSE TEAMS. Before this, eval used the full pool, so
    win_rate_vs_ext_<teacher> compared a random-pool trainee against a teacher piloting its own pin —
    mostly the teacher's TEAM ADVANTAGE, not whether the distillation transferred (the eval read 0.36
    while an offline per-team probe of the same model read 0.710)."""
    pairs = [("models/T1", ["data/teams/sample/9d5f845869e899ee.txt",
                            "data/teams/sample/f7ba5702fe856292.txt"]),
             ("models/T2", ["data/teams/sample/0972146213a667c9.txt"])]
    spec = MatchupSpec.from_args(_distill_args(pairs))
    ev = spec.eval_trainee_teams
    assert ev.kind == "pin_multi"
    assert len(ev.pin_strs) == 3                     # ALL teacher teams across ALL teachers
    assert ev is not spec.trainee_teams              # eval source is distinct from the training source
    # provenance records every taught team's fingerprint
    assert len(spec.to_dict()["eval_trainee_teams"]["pin_shas"]) == 3


def test_non_distill_eval_source_is_unchanged():
    # no distillation -> eval_trainee_teams still defaults to trainee_teams (byte-identical behaviour)
    spec = MatchupSpec.from_args(_args())
    assert spec.eval_trainee_teams is spec.trainee_teams


def test_distill_eval_pin_is_a_LIST_while_training_stays_pool_shaped():
    """A distillation matchup's EVAL source is `pin_multi` (a LIST of team exports) while its TRAINING
    source is still pool-shaped — and every consumer must keep those two straight.

    This asymmetry is the whole point of `eval_trainee_teams` (train on the pool with a bias, eval on the
    TAUGHT teams), but it is also a trap: `train_rl_agent` derived its `[SPECIALIST]` startup line from
    the EVAL source and called `.splitlines()` on it, which crashed **every** `--distill-coef` launch at
    startup with `AttributeError: 'list' object has no attribute 'splitlines'` — the single-teacher case
    included, since one teacher still yields `pin_multi`. It shipped in the same commit that repointed the
    variable at eval and went unseen because no distillation run has launched since. Pinned here so the
    shape divergence is a stated contract rather than an accident."""
    spec = MatchupSpec.from_args(_distill_args([("models/T1", ["data/teams/sample/9d5f845869e899ee.txt"])]))
    assert spec.eval_trainee_teams.kind == "pin_multi"        # LIST-valued, even for ONE teacher/team
    assert spec.trainee_teams.kind != "pin_multi"             # training is the biased pool
    # The training source is what a "trainee pinned to ONE team" message may read: None here, so no
    # single-team line is emitted at all (the crash was reading the eval source instead).
    assert spec.trainee_teams.pin_str is None
