"""The reward DEFAULTS, and the reward flag surface — which since 2026-09-26 is the TERMINAL's alone.

History (kept because the defaults are a contract): the 2026-08-18 owner decision pinned
`--all-shaping-pbrs` ON and `--draw-penalty -35` after the silent v8→v9 composition drift. The shaped
reward path was then DELETED (`gen3_shaped_reward_deletion_v1`, program_rust_core §4 M3 row); its 14
flags are in `designs/deleted_flags.md`. What this file pins now:

1. **The terminal defaults, by value** — ±30, draw −35, the signed terminal; the production values
   are typed, never defaulted.
2. **The deleted flags are GONE from the parser** — a revert that brought one back would be a
   shaped reward reachable by flag again.
3. **The three declarations agree** — `RewardConfig`, `ModelVersion`'s fields and
   `_REWARD_IMMUTABLE_FIELDS` decide what an ABSENT field means; a divergence is a silent drift.
4. **The actionable resume error** — it names the flags to re-pass, and they parse back.
"""

import dataclasses

import pytest

from agents.model.model_version import (
    DELETED_SHAPED_REWARD_FIELDS,
    _REWARD_FIELD_FLAGS,
    _REWARD_IMMUTABLE_FIELDS,
    ModelVersion,
    ModelVersionError,
)
from agents.training.reward_manager import RewardConfig, reward_class_composition
from main.train_rl_agent import build_parser


def _args(argv):
    return build_parser().parse_args(list(argv))


def _version(**reward_fields):
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    v = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
    return dataclasses.replace(v, **reward_fields)


# --------------------------------------------------------------------------- the parser defaults

def test_the_terminal_defaults_are_pinned_by_value():
    a = _args([])
    assert (a.victory_value, a.draw_penalty, a.terminal_indicator) == (30.0, -35.0, False)
    assert (a.progress_decision_tense, a.progress_switch_freeze) == (False, False)


def test_draw_penalty_opt_out_is_the_old_number():
    assert _args(["--draw-penalty", "-30"]).draw_penalty == -30.0


@pytest.mark.parametrize("flag", sorted(set(DELETED_SHAPED_REWARD_FIELDS.values())))
def test_every_deleted_shaped_flag_is_REFUSED_by_the_parser(flag, capsys):
    """Bringing one back is bringing a shaped reward back — a research event, not a refactor."""
    with pytest.raises(SystemExit):
        _args([flag, "1"] if flag in ("--bias-additivity", "--mat-alive-weight",
                                      "--no-progress-penalty", "--switch-bias-weight",
                                      "--self-ko-hp-penalty") else [flag])
    assert "unrecognized arguments" in capsys.readouterr().err
    dest = next(k for k, v in DELETED_SHAPED_REWARD_FIELDS.items() if v == flag)
    assert not hasattr(_args([]), dest)


# --------------------------------------------------- the three declarations agree

def test_reward_config_dataclass_defaults_match_the_parser():
    parsed = RewardConfig.from_args(_args([]))
    assert parsed == dataclasses.replace(RewardConfig(), gamma=parsed.gamma)


def test_model_version_default_reward_fields_match_reward_config():
    cfg = RewardConfig()
    for name, fallback in _REWARD_IMMUTABLE_FIELDS.items():
        assert getattr(cfg, name) == fallback, f"{name}: RewardConfig disagrees with the version fallback"
        field = ModelVersion.__dataclass_fields__[name]
        assert field.default == fallback, f"{name}: ModelVersion's dataclass default disagrees"
    for name in DELETED_SHAPED_REWARD_FIELDS:
        assert not hasattr(cfg, name) and name not in ModelVersion.__dataclass_fields__, name


def test_every_immutable_reward_field_has_a_flag():
    """The resume error is only actionable if every flag it can print is a flag that exists."""
    assert set(_REWARD_IMMUTABLE_FIELDS) == set(_REWARD_FIELD_FLAGS)
    known = build_parser()._option_string_actions
    for name, flag in _REWARD_FIELD_FLAGS.items():
        assert flag in known, f"{flag} ({name}) is not a real flag"
        assert known[flag].dest == name, f"{flag} does not set {name}"


def test_the_default_composition_is_one_terminal():
    comp = reward_class_composition(RewardConfig.from_args(_args([])))
    assert (comp["terminal"], comp["pbrs"], comp["bias"]) == (1, 0, 0)


# ------------------------------------------------- the resume FATAL, and that it names the fix

def _saved_production():
    return _version(terminal_indicator=True, victory_value=1.0, draw_penalty=0.0)


def test_a_production_run_resumed_under_the_defaults_is_a_hard_error():
    """A flagless resume of a win-indicator run would request the signed ±30 terminal; that must
    FATAL, never flip silently under a live run."""
    with pytest.raises(ModelVersionError) as exc:
        _saved_production().check_reward_config(RewardConfig())
    msg = str(exc.value)
    assert "terminal_indicator" in msg and "victory_value" in msg and "draw_penalty" in msg
    assert "start a fresh run" in msg


def test_the_re_passed_flags_actually_parse_back_to_the_saved_values():
    saved = _saved_production()
    with pytest.raises(ModelVersionError) as exc:
        saved.check_reward_config(RewardConfig())
    fix = str(exc.value).split("re-pass `")[1].split("`")[0]
    assert "--terminal-indicator" in fix and "--victory-value 1.0" in fix
    saved.check_reward_config(RewardConfig.from_args(_args(fix.split())))   # must not raise


def test_a_fresh_default_run_resumes_flaglessly():
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    v = ModelVersion.from_layout_and_policy_kwargs(
        layout, {"net_arch": [512, 512]}, reward_config=RewardConfig())
    v.check_reward_config(RewardConfig.from_args(_args([])))   # must not raise


def test_frozen_opponents_are_exempt_from_the_reward_check():
    """`check_compatible` gates EVERY load — eval workers, sentinels, teachers — whose forward never
    reads the reward. A reward field inside it would make a snapshot unloadable as an opponent."""
    current = _version()
    for name in _REWARD_IMMUTABLE_FIELDS:
        saved = getattr(current, name)
        other = (not saved) if isinstance(saved, bool) else saved + 1.0
        current.check_compatible(dataclasses.replace(current, **{name: other}))  # must not raise
