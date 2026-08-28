"""`gen3_distill_bias_at_coef0_v1` — `--distill-team-bias` must be EFFECTIVE at `--distill-coef 0`.

THE INCIDENT. `--distill-team-bias` points the trainee's team draw at the TEACHER TEAMS, and it
reads `args._distill_pairs` to know which teams those are. `main.train.config` populated
`_distill_pairs` only under `if args.distill_coef > 0`, so a run launched with teachers named and
the coefficient ZERO — the CONTROL-arm shape, whose entire purpose is to hold the team distribution
constant while folding no loss — trained at an EFFECTIVE bias of **0.0** while its argv, its
`metadata.json` and its startup banner all said 0.4. `ai_v9_58_R2CTRL_0827` is the run; the rev-2
capstone's "team-bias constancy" design was violated by the config layer, invisibly, and the
difference between the arms was a team distribution nobody had chosen.

The rule the tests below pin: **the coefficient gates the LOSS, not the BOOKKEEPING.**

  * pairs + team bias: keyed on `--distill-teacher`, at any coefficient (tests 1-3);
  * teacher LOADING and the loss fold: keyed on the coefficient, so a coef-0 control pays for no
    teacher network (test 4);
  * `_distill_species` (⇒ the training-only `distill_mask` obs key, ⇒ the observation SPACE): also
    keyed on the coefficient — the arms must not differ in their obs space (test 3);
  * a bias with no teacher to bias toward is refused rather than silently inert (test 5).

Tests 6-8 are the byte-identity half: a run with no `--distill-teacher`, and a run with teachers at
coef > 0, must be exactly what they were before the fix.
"""
import random
from types import SimpleNamespace

import pytest

# Two real sample-pool teams (the `matchup_spec_test` convention) — the config layer parses paths,
# the teambuilder reads and validates the exports, so they have to exist and be legal gen3ou.
TEACHER_TEAMS = ("data/teams/sample/9d5f845869e899ee.txt",
                 "data/teams/sample/f7ba5702fe856292.txt")
TEACHER_SPEC = f"models/T1:{TEACHER_TEAMS[0]},{TEACHER_TEAMS[1]}"


def _argv(*extra):
    # `--use-bridge node` keeps resolve_config off the rust sim_bridge binary (the
    # checkpoint_cadence_test convention); irrelevant to everything under test here.
    return ["--steps", "1", "--use-bridge", "node", *extra]


def _resolved(*extra):
    """Parse + resolve a real argv through the real parser. Raises SystemExit on a refused config."""
    from main.train.config import resolve_config
    from main.train.parser import build_parser

    parser = build_parser()
    args = parser.parse_args(_argv(*extra))
    resolve_config(args, parser)
    return args


# R2-CTRL's exact shape: teachers named, coefficient ZERO, bias typed.
_R2CTRL = ("--distill-teacher", TEACHER_SPEC, "--distill-coef", "0.0",
           "--distill-team-bias", "0.4")


# ────────────────────────────────────────────── 1. the pairs exist at coef 0

def test_r2ctrl_shape_populates_the_distill_pairs():
    """THE REGRESSION. Before the fix this list was empty and every downstream reader — the team
    bias, the eval pin — silently did nothing."""
    args = _resolved(*_R2CTRL)
    assert args._distill_pairs == [("models/T1", [TEACHER_TEAMS[0], TEACHER_TEAMS[1]])]
    assert args.distill_team_bias == 0.4
    assert args.distill_coef == 0.0


# ────────────────────────────────────────────── 2. the team DRAW is actually biased at coef 0

@pytest.fixture(scope="module")
def pool_and_teacher_teams():
    """A small POOL disjoint from the teacher teams, so a drawn team identifies its source."""
    from utils.team_loader import TeamLoader

    teacher_strs = [open(f, encoding="utf-8").read() for f in TEACHER_TEAMS]
    pool = [t for t in TeamLoader().get_sample_teams()
            if t.strip() not in {s.strip() for s in teacher_strs}][:3]
    assert len(pool) == 3
    return pool, teacher_strs


def _bias_args(coef, bias):
    return SimpleNamespace(
        _distill_pairs=[("models/T1", list(TEACHER_TEAMS))],
        distill_coef=coef, distill_team_bias=bias,
        team_pfsp="off", team_pfsp_cap=3.0, team_pfsp_floor=0.05)


def test_the_trainee_team_draw_is_biased_at_coef_zero(pool_and_teacher_teams):
    """The MEASUREMENT, not the flag value: draw 4000 teams and count how many came from the
    teacher pool. Pre-fix this was 0 of 4000 (the pairs were empty, so the builder was never
    replaced); the target is 0.4 (n=4000 ⇒ sd 0.0077, so ±0.04 is ~5 sd)."""
    from main.train.matchup_setup import apply_distill_team_bias

    pool, _ = pool_and_teacher_teams
    args = _bias_args(coef=0.0, bias=0.4)
    baseline = object()                      # the builder that must be REPLACED
    tb = apply_distill_team_bias(args, pool, baseline)

    assert tb is not baseline, "the trainee teambuilder was not replaced ⇒ effective bias 0.0"
    assert tb.bias_prob == 0.4 and len(tb.bias_packed_teams) == 2
    assert not set(tb.bias_packed_teams) & set(tb.packed_teams)   # sources are distinguishable

    random.seed(20260827)
    draws = [tb.yield_team() for _ in range(4000)]
    frac = sum(d in set(tb.bias_packed_teams) for d in draws) / len(draws)
    assert 0.36 <= frac <= 0.44, f"teacher-team draw fraction {frac:.3f} is not the requested 0.4"


def test_no_teacher_means_the_teambuilder_is_untouched(pool_and_teacher_teams):
    """Byte-identity: with no `--distill-teacher` the helper returns its argument unchanged."""
    from main.train.matchup_setup import apply_distill_team_bias

    pool, _ = pool_and_teacher_teams
    args = _bias_args(coef=0.0, bias=0.4)
    args._distill_pairs = []
    baseline = object()
    assert apply_distill_team_bias(args, pool, baseline) is baseline
    assert args._distill_species is None


# ────────────────────────────────────────────── 3. the OBS SPACE does not move at coef 0

def test_the_distill_mask_obs_key_is_not_emitted_at_coef_zero(pool_and_teacher_teams):
    """`_distill_species` is what makes `Gen3Env` add the training-only `distill_mask` obs key.
    Emitting it for a run that folds no distill term would change the observation SPACE — an
    unrequested difference between the arms, and a resume-breaker for a live control run."""
    from main.train.matchup_setup import apply_distill_team_bias

    pool, _ = pool_and_teacher_teams
    off = _bias_args(coef=0.0, bias=0.4)
    apply_distill_team_bias(off, pool, object())
    assert off._distill_species is None

    on = _bias_args(coef=1.0, bias=0.4)
    apply_distill_team_bias(on, pool, object())
    assert on._distill_species is not None and len(on._distill_species) == 1
    assert len(on._distill_species[0]) == 2               # one species-set per team of teacher 1
    assert all(len(s) == 6 for s in on._distill_species[0])


# ────────────────────────────────────────────── 4. NO teacher network is loaded at coef 0

def _hparam_args(**over):
    """An `args` carrying every row of `_TRAINING_HPARAMS` (the table reads them unconditionally)."""
    from main.train.model_build import _F0, _F0_OPT, _TRAINING_HPARAMS

    base = {name: (0.0 if how in (_F0, _F0_OPT) else None) for name, how in _TRAINING_HPARAMS}
    base.update(capacity_telemetry=False, search_teacher=None, opd_coef=0.0,
                distill_coef=0.0, _distill_pairs=[("models/T1", list(TEACHER_TEAMS))])
    base.update(over)
    return SimpleNamespace(**base)


def test_no_teacher_model_is_loaded_at_coef_zero(monkeypatch):
    """A coef-0 control names N teachers so the TEAM BIAS can point at their teams — but it must
    not pay N frozen networks of RAM and a forward per minibatch to multiply them by zero. The
    tripwire is the first call the load block makes."""
    import agents.model.snapshot as snapshot
    from main.train import model_build

    def _boom(*a, **k):                       # noqa: ANN001 — a tripwire, never called
        raise AssertionError("a distill teacher was loaded at --distill-coef 0")

    monkeypatch.setattr(snapshot, "current_model_version", _boom)
    monkeypatch.setattr(snapshot, "load_foreign_opponent", _boom)
    # The load block turns ANY exception into `os._exit(FATAL_CONFIG)`, which would kill the whole
    # pytest process rather than fail this test — make it a normal failure instead.
    monkeypatch.setattr(model_build.os, "_exit", _boom)

    model = SimpleNamespace()
    model_build.apply_training_hparams(model, _hparam_args(), mappings=None,
                                       attach_cf_labels=lambda _m: None)
    assert model._distill_teachers == []
    assert model.distill_coef == 0.0


def test_the_teacher_load_is_still_attempted_above_coef_zero(monkeypatch):
    """The other side of the same gate — the fix must not have disabled distillation itself. Same
    tripwire, coefficient > 0 ⇒ it FIRES."""
    import agents.model.snapshot as snapshot
    from main.train import model_build

    def _boom(*a, **k):                       # noqa: ANN001
        raise AssertionError("reached the loader")

    monkeypatch.setattr(snapshot, "current_model_version", _boom)
    monkeypatch.setattr(model_build, "_run_arch_toggles", lambda _a: {})   # skip the arch fan-out
    monkeypatch.setattr(model_build.os, "_exit", _boom)
    with pytest.raises(AssertionError, match="reached the loader"):
        model_build.apply_training_hparams(SimpleNamespace(), _hparam_args(distill_coef=1.0),
                                           mappings=None, attach_cf_labels=lambda _m: None)


# ────────────────────────────────────────────── 5. the LOUD guard

def test_a_team_bias_with_no_teacher_is_refused():
    """`--distill-team-bias 0.4` with nothing to bias toward is the flag doing nothing at all —
    exactly the class of silence this whole fix is about."""
    with pytest.raises(SystemExit):
        _resolved("--distill-team-bias", "0.4")


def test_an_explicit_zero_team_bias_with_no_teacher_is_allowed():
    """The guard is about a bias that cannot be honoured, not about the flag's presence."""
    assert _resolved("--distill-team-bias", "0.0").distill_team_bias == 0.0


def test_a_pin_alongside_a_teacher_is_refused():
    """The bias REPLACES the trainee teambuilder, so at coef 0 a `--trainee-teams` pin would be
    silently discarded rather than merely redundant. Refuse instead of choosing for the user."""
    with pytest.raises(SystemExit):
        _resolved("--distill-teacher", TEACHER_SPEC, "--distill-coef", "0.0",
                  "--trainee-teams", TEACHER_TEAMS[0])


# ────────────────────────────────────────────── 6-8. everything else is unchanged

def test_a_run_with_no_distill_flags_is_unchanged():
    args = _resolved()
    assert args._distill_pairs == []
    assert args.distill_team_bias == 0.4, "the unset flag must still resolve to the 0.4 default"


def test_the_flag_defaults_to_none_so_a_typed_value_is_detectable():
    """The None sentinel is what makes the guard above possible: with a hard 0.4 argparse default,
    'the user asked for a bias' and 'the user typed nothing' are the same value."""
    from main.train.parser import build_parser

    assert build_parser().parse_args([]).distill_team_bias is None
    assert build_parser().parse_args(["--distill-team-bias", "0.6"]).distill_team_bias == 0.6


def test_a_coefficient_with_no_teacher_still_errors():
    with pytest.raises(SystemExit):
        _resolved("--distill-coef", "1.0")


@pytest.mark.parametrize("spec", ["models/T1", "team.txt", ":team.txt", "models/T1:"])
def test_a_malformed_spec_is_refused_at_coef_zero_TOO(spec):
    """Every parse error still fires with the coefficient at 0 — the pairs are now parsed there,
    so a typo in a control arm's spec must fail at the CLI rather than at the first team draw."""
    with pytest.raises(SystemExit):
        _resolved("--distill-teacher", spec, "--distill-coef", "0.0")


def test_teachers_at_a_live_coefficient_are_unchanged():
    args = _resolved("--distill-teacher", TEACHER_SPEC, "--distill-coef", "1.0")
    assert args._distill_pairs == [("models/T1", [TEACHER_TEAMS[0], TEACHER_TEAMS[1]])]
    assert args.distill_team_bias == 0.4
