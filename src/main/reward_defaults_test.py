"""The reward DEFAULTS are the validated ai_v8 composition — and the defaults themselves are the
contract.

Owner decision (2026-08-18), after `designs/research_state/ledger.md` recorded the drift: all 20
`ai_v8_*` runs trained with `--all-shaping-pbrs` ON and `--draw-penalty -35`; every `ai_v9_*` run
through gen-14 trained with neither, with no recorded rationale anywhere. The flag was simply not
carried across the fresh-generation reset. It was invisible for a year because the reward config is
**training-only** — it bumps no `ARCH_SIGNATURE`, is absent from `check_compatible`, and no launch
line ever stated what the reward was composed of. Nothing failed; the objective just quietly became
a fully-additive 26-term BIAS where the validated one was near-policy-invariant.

So the two defaults are pinned here BY VALUE, beside their opt-outs, in the shape
`compile_defaults_test.py` established for the same class of change (an inverted default is
invisible once landed — nothing fails, the run just quietly takes a different path).

Three things this file holds that a `assert default is True` would not:

1. **The composition pins.** The counts and the BIAS term list under each of the two regimes, so
   the drift class is legible forever: the default composition has ONE acknowledged bias term, and
   the fallback has 26. A future edit that silently re-adds an additive term fails here.
2. **The actionable resume error.** Flipping a resume-immutable default makes every pre-flip run
   mismatch on a flagless resume — correct and required (a live run's reward must never flip under
   it), but only useful if the error names the flags to re-pass.
3. **The default agreement between `RewardConfig` and `ModelVersion`.** They are separate
   declarations of the same field set; a divergence would mean an absent field meant one thing to
   the reward and another to the version record — the drift class one level down.
"""

import dataclasses

import pytest

from agents.model.model_version import (
    _REWARD_FIELD_FLAGS,
    _REWARD_IMMUTABLE_FIELDS,
    ModelVersion,
    ModelVersionError,
)
from agents.training.reward_manager import (
    RewardBreakdown,
    RewardClass,
    RewardConfig,
    format_reward_composition,
    reward_class_composition,
)
from main.train_rl_agent import build_parser


def _args(argv):
    return build_parser().parse_args(list(argv))


def _v9_config():
    """The composition every ai_v9 run through gen-14 actually trained with."""
    return RewardConfig(all_shaping_pbrs=False, draw_penalty=-30.0)


# --------------------------------------------------------------------------- the parser defaults

def test_all_shaping_pbrs_defaults_on():
    assert _args([]).all_shaping_pbrs is True


def test_all_shaping_pbrs_opt_out_is_no_all_shaping_pbrs():
    assert _args(["--no-all-shaping-pbrs"]).all_shaping_pbrs is False


def test_all_shaping_pbrs_also_takes_an_explicit_value():
    """BoolFlag's value form — the spelling every other tri-state toggle here accepts."""
    assert _args(["--all-shaping-pbrs", "false"]).all_shaping_pbrs is False
    assert _args(["--all-shaping-pbrs=false"]).all_shaping_pbrs is False
    assert _args(["--all-shaping-pbrs"]).all_shaping_pbrs is True


def test_draw_penalty_defaults_to_minus_35():
    assert _args([]).draw_penalty == -35.0


def test_draw_penalty_opt_out_is_the_old_number():
    """A float flag has no negation — the way back is the old value, which is why the resume
    error renders it as `--draw-penalty -30.0` rather than a `--no-` form."""
    assert _args(["--draw-penalty", "-30"]).draw_penalty == -30.0


def test_stall_pbrs_stays_default_off():
    """DELIBERATELY unchanged. Zero-bias is a later, single-variable step: --stall-pbrs additionally
    zeroes `no_progress_tax`, and that tilt carries a documented stall-regression risk. Bundling it
    with this flip would make the generation a two-variable change."""
    assert _args([]).stall_pbrs is False
    assert _args(["--stall-pbrs"]).stall_pbrs is True


@pytest.mark.parametrize("dest,expected", [
    ("bias_additivity", 1.0), ("mat_alive_weight", 1.25), ("bias_redesign", False),
    ("switch_bias_weight", 0.0), ("self_ko_hp_penalty", 0.0),
    ("drop_redundant_bias", False), ("drop_switch_bias", False),
    ("no_progress_penalty", 0.15),
])
def test_every_other_reward_default_is_unchanged(dest, expected):
    """The flip is exactly two fields wide. This is the guard against it growing by accident."""
    assert getattr(_args([]), dest) == expected


# --------------------------------------------------- RewardConfig agrees with the parser defaults

def test_reward_config_dataclass_defaults_match_the_parser():
    """`RewardConfig()` is what `from_dict` falls back to for a field an older config omits, so its
    defaults are a second declaration of the same contract and must not drift from argparse."""
    parsed = RewardConfig.from_args(_args([]))
    assert parsed == dataclasses.replace(RewardConfig(), gamma=parsed.gamma)


def test_model_version_default_reward_fields_match_reward_config():
    """The third declaration. `ModelVersion`'s field defaults and `_REWARD_IMMUTABLE_FIELDS` decide
    what an ABSENT field means to the version record; `RewardConfig` decides what it means to the
    reward. A divergence is the drift class one level down from the one this file exists for."""
    cfg = RewardConfig()
    for name, fallback in _REWARD_IMMUTABLE_FIELDS.items():
        assert getattr(cfg, name) == fallback, f"{name}: RewardConfig disagrees with the version fallback"
        field = ModelVersion.__dataclass_fields__[name]
        assert field.default == fallback, f"{name}: ModelVersion's dataclass default disagrees"


def test_every_immutable_reward_field_has_a_flag():
    """The resume error is only actionable if every flag it can print is a flag that exists — a
    renamed CLI option would otherwise produce a confidently-wrong instruction."""
    assert set(_REWARD_IMMUTABLE_FIELDS) == set(_REWARD_FIELD_FLAGS)
    known = build_parser()._option_string_actions
    for name, flag in _REWARD_FIELD_FLAGS.items():
        assert flag in known, f"{flag} ({name}) is not a real flag"
        assert known[flag].dest == name, f"{flag} does not set {name}"


# ---------------------------------------------------------------- the composition, both regimes

def test_default_composition_is_the_v8_shape():
    """1 TERMINAL + 7 PBRS + exactly ONE acknowledged BIAS term.

    (The ledger's prose says "8 PBRS"; it counted the PBRS registry class size. `pbrs_progress` is
    `--stall-pbrs`-gated and that flag stays off, so 7 potentials are actually reachable — the
    census counts what the config can emit, not what the class contains.)
    """
    comp = reward_class_composition(RewardConfig())
    assert comp["terminal"] == 1
    assert comp["pbrs"] == 7
    assert comp["bias"] == 1
    assert comp["bias_terms"] == ["no_progress_tax"]
    assert set(comp["pbrs_terms"]) == {
        "pbrs_material", "pbrs_belief", "pbrs_status",
        "pbrs_hazard", "pbrs_boost", "pbrs_opp_boosts", "pbrs_roar"}


def test_no_all_shaping_pbrs_composition_is_the_v9_shape():
    """The fallback restores the fully-additive objective: 2 potentials and 26 BIAS terms, none of
    them telescoping. This is what every ai_v9 run through gen-14 trained."""
    comp = reward_class_composition(_v9_config())
    assert comp["terminal"] == 1
    assert comp["pbrs"] == 2
    assert set(comp["pbrs_terms"]) == {"pbrs_material", "pbrs_belief"}
    assert comp["bias"] == 26
    # `no_progress_tax` is the one BIAS term the v9 regime does NOT have — its clock charge is gated
    # on `bias_redesign OR all_shaping_pbrs`, so turning the flag off also disarms the stall tilt.
    assert "no_progress_tax" not in comp["bias_terms"]
    assert {"stall_tax", "matchup_penalty", "switch_base", "status"} <= set(comp["bias_terms"])


def test_the_two_regimes_are_the_whole_point_of_the_census():
    """Stated as a single comparison so the drift is one assertion, not two files."""
    v8, v9 = reward_class_composition(RewardConfig()), reward_class_composition(_v9_config())
    assert v8["bias"] == 1 and v9["bias"] == 26


def test_composition_covers_the_registry_exactly():
    """Every census member is a real registry field of its class — a typo'd name would otherwise
    read as a silently missing term."""
    comp = reward_class_composition(RewardConfig())
    reg = RewardBreakdown._REGISTRY
    assert all(reg[n] is RewardClass.PBRS for n in comp["pbrs_terms"])
    assert all(reg[n] is RewardClass.BIAS for n in comp["bias_terms"])
    assert comp["terminal"] == len(RewardBreakdown.registry_fields(RewardClass.TERMINAL))


def test_stall_pbrs_zeroes_the_last_bias_term():
    """The zero-bias destination, for reference — running BOTH switches empties the BIAS class."""
    comp = reward_class_composition(RewardConfig(stall_pbrs=True))
    assert comp["bias"] == 0 and comp["bias_terms"] == []
    assert "pbrs_progress" in comp["pbrs_terms"]   # Φ_progress carries the anti-stall signal instead


def test_the_announcer_line_names_the_sole_bias_term():
    line = format_reward_composition(RewardConfig())
    assert line == "[Reward] composition: 1 TERMINAL + 7 PBRS + 1 BIAS (no_progress_tax)"


def test_the_announcer_truncates_the_additive_pathology_but_keeps_the_count():
    """26 term names is not a line anyone reads; the COUNT is the signal."""
    line = format_reward_composition(_v9_config())
    assert "26 BIAS" in line and "+20 more" in line and len(line) < 200


def test_the_announcer_says_so_when_there_is_no_bias_left():
    assert "none — fully policy-invariant" in format_reward_composition(RewardConfig(stall_pbrs=True))


def test_the_census_reads_a_model_version_too():
    """Duck-typed on field names, so the recorded `ModelVersion` of an ARCHIVED run can be censused
    without reconstructing its RewardConfig — what an offline launch-diff needs."""
    class _V9Version:
        all_shaping_pbrs, stall_pbrs, bias_redesign = False, False, False
        drop_redundant_bias = drop_switch_bias = False
        switch_bias_weight = self_ko_hp_penalty = 0.0
    assert reward_class_composition(_V9Version()) == reward_class_composition(_v9_config())


# ------------------------------------------------- the resume FATAL, and that it names the fix

def _saved_v9_version():
    """A ModelVersion as a pre-flip (v9-era) run recorded it."""
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    v = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
    return dataclasses.replace(v, all_shaping_pbrs=False, draw_penalty=-30.0)


def test_a_v9_run_resumed_under_the_new_defaults_is_a_hard_error():
    """THE hazard of flipping a resume-immutable default. A flagless resume of a pre-flip run now
    requests a different reward than the one it trained under; that must FATAL, never flip
    silently under a live run."""
    with pytest.raises(ModelVersionError) as exc:
        _saved_v9_version().check_reward_config(RewardConfig())
    msg = str(exc.value)
    assert "all_shaping_pbrs" in msg and "draw_penalty" in msg


def test_the_resume_error_names_both_flags_to_re_pass():
    """A diff alone leaves the reader to reconstruct the flag spelling — including that the opt-out
    is `--no-all-shaping-pbrs` and that a float flag's way back is just the old number."""
    with pytest.raises(ModelVersionError) as exc:
        _saved_v9_version().check_reward_config(RewardConfig())
    msg = str(exc.value)
    # Fields are reported in `_REWARD_IMMUTABLE_FIELDS` order (draw_penalty precedes
    # all_shaping_pbrs), so the assertion is per-field rather than on one baked sentence.
    assert "This run recorded draw_penalty=-30.0, all_shaping_pbrs=False." in msg
    assert "re-pass `--draw-penalty -30.0 --no-all-shaping-pbrs`" in msg
    assert "start a fresh run" in msg


def test_the_re_passed_flags_actually_parse_back_to_the_saved_values():
    """The strongest form of "actionable": take the flags out of the message, feed them to the real
    parser, and the resulting config must pass the very check that produced the message."""
    saved = _saved_v9_version()
    with pytest.raises(ModelVersionError) as exc:
        saved.check_reward_config(RewardConfig())
    fix = str(exc.value).split("re-pass `")[1].split("`")[0]
    saved.check_reward_config(RewardConfig.from_args(_args(fix.split())))   # must not raise


def test_a_v9_run_resumed_with_its_own_flags_is_accepted():
    _saved_v9_version().check_reward_config(_v9_config())   # must not raise


def test_a_fresh_default_run_resumes_flaglessly():
    """The other direction: a run STARTED under the new defaults must resume with no reward flags
    at all, or the flip would have made every launcher restart a FATAL."""
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    v = ModelVersion.from_layout_and_policy_kwargs(
        layout, {"net_arch": [512, 512]}, reward_config=RewardConfig())
    v.check_reward_config(RewardConfig.from_args(_args([])))   # must not raise


def test_frozen_opponents_are_exempt_from_the_reward_check():
    """`check_compatible` gates EVERY load — eval workers, self-play sentinels, distill teachers —
    whose forward never reads the reward. A reward field inside it would make a v9-era snapshot
    unloadable as an opponent, which is a different and much worse failure than a resume FATAL."""
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    current = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
    for name in _REWARD_IMMUTABLE_FIELDS:
        saved = getattr(current, name)
        other = (not saved) if isinstance(saved, bool) else saved + 1.0
        current.check_compatible(dataclasses.replace(current, **{name: other}))  # must not raise
