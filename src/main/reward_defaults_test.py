"""The reward DEFAULTS, and the reward surface — which since 2026-09-26 is the TERMINAL alone, and since deletion
pass P11b has NO flags (the terminal is a constant of the trainer's namespace, `parser/objective.py`).

History (kept because the defaults are a contract): the 2026-08-18 owner decision pinned
`--all-shaping-pbrs` ON and `--draw-penalty -35` after the silent v8→v9 composition drift. The shaped
reward path was then DELETED (`gen3_shaped_reward_deletion_v1`, program_rust_core §4 M3 row); its 14
flags are in `designs/deleted_flags.md`. What this file pins now:

1. **The terminal constants, by value** — the namespace's are the win-prob critic's (+1 indicator, draw 0),
   while an ABSENT recorded field still means the historical signed terminal (±30, draw −35) — two
   questions, two answers.
2. **The deleted flags are GONE from the parser** — a revert that brought one back would be a
   shaped reward reachable by flag again.
3. **The three declarations agree** — `RewardConfig`, `ModelVersion`'s fields and
   `_REWARD_IMMUTABLE_FIELDS` decide what an ABSENT field means; a divergence is a silent drift. The
   parser differs from them on exactly the three winprob-terminal fields, and nowhere else.
4. **The resume refusal** — a recorded NON-production reward is refused with the values it recorded and the way
   out (run it PINNED, or start fresh); there is no flag to re-pass.
"""

import dataclasses

import pytest

from agents.model.model_version import (
    DELETED_SHAPED_REWARD_FIELDS,
    _REWARD_IMMUTABLE_FIELDS,
    ModelVersion,
    ModelVersionError,
)
from agents.training.reward_config import RewardConfig
from agents.training.reward_composition import reward_class_composition
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
    assert (a.victory_value, a.draw_penalty, a.terminal_indicator) == (1.0, 0.0, True)
    assert (a.progress_decision_tense, a.progress_switch_freeze) == (False, False)


@pytest.mark.parametrize("argv", [["--no-terminal-indicator"], ["--terminal-indicator"], ["--victory-value", "30"],
                                  ["--draw-penalty", "-35"]])
def test_the_signed_terminal_is_NOT_reachable_by_typing_it(argv, capsys):
    """It was, until P11b: the winprob critic refused it, so the flags offered a value nothing could train."""
    with pytest.raises(SystemExit):
        _args(argv)
    assert "was DELETED" in capsys.readouterr().err


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

#: The fields where the bare PARSER (the winprob terminal) and the ABSENT-record meaning (the historical
#: signed terminal) deliberately differ — and the parser's values.
_WINPROB_TERMINAL = {"terminal_indicator": True, "victory_value": 1.0, "draw_penalty": 0.0}


def test_reward_config_dataclass_defaults_match_the_parser_off_the_winprob_terminal():
    parsed = RewardConfig.from_args(_args([]))
    assert parsed == dataclasses.replace(RewardConfig(), gamma=parsed.gamma, **_WINPROB_TERMINAL)
    for name, value in _WINPROB_TERMINAL.items():
        assert _REWARD_IMMUTABLE_FIELDS[name] != value, (
            f"{name}: an ABSENT record must keep the historical value, not follow the parser")


def test_model_version_default_reward_fields_match_reward_config():
    cfg = RewardConfig()
    for name, fallback in _REWARD_IMMUTABLE_FIELDS.items():
        assert getattr(cfg, name) == fallback, f"{name}: RewardConfig disagrees with the version fallback"
        field = ModelVersion.__dataclass_fields__[name]
        assert field.default == fallback, f"{name}: ModelVersion's dataclass default disagrees"
    for name in DELETED_SHAPED_REWARD_FIELDS:
        assert not hasattr(cfg, name) and name not in ModelVersion.__dataclass_fields__, name


def test_no_immutable_reward_field_has_a_flag_any_more():
    """The resume refusal is only honest if it does not point at a flag: every immutable reward field is a
    namespace CONSTANT (P11b; the no-progress clock's two switches, P11d), none an option."""
    known = build_parser()._option_string_actions
    for name in _REWARD_IMMUTABLE_FIELDS:
        assert "--" + name.replace("_", "-") not in known, f"{name} has a flag again"
        assert hasattr(_args([]), name), f"{name} is not a constant of the namespace"


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


@pytest.mark.parametrize("field", ["progress_decision_tense", "progress_switch_freeze"])
def test_a_run_that_recorded_a_deleted_clock_variant_is_REFUSED_by_name(field):
    """P11d deleted the no-progress clock's two opt-in variants (flags, Python clock, Rust clock). The
    RECORDED run fields stay readable and value-checked: a checkpoint that trained under one cannot resume
    on this code (its observation stream would silently change), and the refusal names the field and the way
    out — the same pattern as P11b's reward cluster, with no RETIRED row and no version bump."""
    saved = _version(terminal_indicator=True, victory_value=1.0, draw_penalty=0.0, **{field: True})
    with pytest.raises(ModelVersionError) as exc:
        saved.check_reward_config(RewardConfig.from_args(_args([])))
    msg = str(exc.value)
    assert field in msg and "saved=True, requested=False" in msg and "PINNED" in msg and "fresh run" in msg
    # …and the production values of both fields resume
    _version(terminal_indicator=True, victory_value=1.0, draw_penalty=0.0).check_reward_config(
        RewardConfig.from_args(_args([])))


def test_the_refusal_names_the_way_out_and_no_flag():
    saved = _saved_production()
    with pytest.raises(ModelVersionError) as exc:
        saved.check_reward_config(RewardConfig())
    msg = str(exc.value)
    assert "PINNED" in msg and "re-pass" not in msg and "--terminal-indicator" not in msg
    saved.check_reward_config(RewardConfig.from_args(_args([])))   # the production reward resumes: no raise


def test_a_fresh_default_run_resumes_flaglessly():
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    fresh = RewardConfig.from_args(_args([]))
    v = ModelVersion.from_layout_and_policy_kwargs(
        layout, {"net_arch": [512, 512]}, reward_config=fresh)
    v.check_reward_config(RewardConfig.from_args(_args([])))   # must not raise


def test_a_SIGNED_terminal_run_resumed_flaglessly_is_REFUSED_by_name():
    """The other side of the flip: a run recorded on the signed terminal, resumed with no reward
    flags, now meets the winprob constants — and must FATAL naming the fields, never train on."""
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    v = ModelVersion.from_layout_and_policy_kwargs(
        layout, {"net_arch": [512, 512]}, reward_config=RewardConfig())
    with pytest.raises(ModelVersionError, match="terminal_indicator.*PINNED|PINNED"):
        v.check_reward_config(RewardConfig.from_args(_args([])))


def test_frozen_opponents_are_exempt_from_the_reward_check():
    """`check_compatible` gates EVERY load — eval workers, sentinels, teachers — whose forward never
    reads the reward. A reward field inside it would make a snapshot unloadable as an opponent."""
    current = _version()
    for name in _REWARD_IMMUTABLE_FIELDS:
        saved = getattr(current, name)
        other = (not saved) if isinstance(saved, bool) else saved + 1.0
        current.check_compatible(dataclasses.replace(current, **{name: other}))  # must not raise
