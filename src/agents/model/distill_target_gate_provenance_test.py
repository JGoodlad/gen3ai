"""Gates for `gen3_distill_target_gate_v1` (config v103) — the advantage-gated / action-form
distillation family + the rank tripwire (design_advantage_gated_distillation.md §3.1/§3.3/§4.1/§7).

Seven TRAINING-only knobs in the `td_aux_coef` provenance genre exactly: argparse default `None`
(the precondition for `_resolve` to fire — the v100 lesson: a `_resolve` line beside a hard
argparse default is DEAD CODE that passes presence tests), recorded on `ModelVersion` for
provenance, `_resolve`-inherited on a flagless resume, NEVER gated by `check_compatible` (a
frozen eval/pool/distill opponent runs no loss, so gating a loss knob there is a false
rejection). The v103 migration defaults a pre-v103 config to the argparse defaults — not a
guess: "kl" IS the loss every pre-v103 run trained with, no run ever had a gate, and the
tripwire did not exist.

The loss/fold behavior itself is pinned in `agents/training/instrumented_ppo_test.py`; the
tripwire state machine in `agents/training/rank_tripwire_test.py`.
"""
import dataclasses
import json

import pytest

from agents.model.model_version import (
    MODEL_CONFIG_VERSION,
    ModelVersion,
    _migrate_config,
)

# (field, cli flag, argparse/genre default, a distinct legal value)
_FAMILY = [
    ("distill_target",     "--distill-target",     "kl",    "action"),
    ("distill_topk",       "--distill-topk",       1,       3),
    ("distill_gate",       "--distill-gate",       "none",  "advantage"),
    ("distill_gate_tau",   "--distill-gate-tau",   0.0,     0.5),
    ("distill_beta",       "--distill-beta",       1.0,     2.0),
    ("rank_tripwire",      "--rank-tripwire",      "warn",  "abort"),
    ("rank_tripwire_drop", "--rank-tripwire-drop", 0.20,    0.30),
]
_IDS = [f[0] for f in _FAMILY]


@pytest.fixture(scope="module")
def layout():
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    return Gen3ObservationEncoder(load_mappings()).get_layout()


# ------------------------------------------------------------------------ 1. the argparse surface

@pytest.mark.parametrize("name,cli,default,other", _FAMILY, ids=_IDS)
def test_the_cli_flag_defaults_to_none_so_a_flagless_resume_can_inherit(name, cli, default, other):
    from main.train.parser import build_parser

    assert getattr(build_parser().parse_args([]), name) is None


@pytest.mark.parametrize("name,cli,default,other", _FAMILY, ids=_IDS)
def test_the_flag_has_a_resolve_line(name, cli, default, other):
    import inspect
    import re

    from main.train.config import resolve_config

    names = set(re.findall(r"_resolve\(\s*\"([a-z0-9_]+)\"", inspect.getsource(resolve_config)))
    assert name in names, (
        f"no `_resolve('{name}', ...)` in resolve_config — a flagless resume will not inherit it")


@pytest.mark.parametrize("name,cli,default,other", _FAMILY, ids=_IDS)
def test_the_flag_still_parses_an_explicit_value(name, cli, default, other):
    from main.train.parser import build_parser

    assert getattr(build_parser().parse_args([cli, str(other)]), name) == other


# -------------------------------------------------------------------------- 2. recorded, 3. never gated

@pytest.mark.parametrize("name,cli,default,other", _FAMILY, ids=_IDS)
def test_the_field_is_recorded_and_round_trips(layout, name, cli, default, other):
    v = ModelVersion.from_layout_and_policy_kwargs(
        layout, {"net_arch": [512, 512]}, **{name: other})
    assert getattr(v, name) == other
    assert getattr(ModelVersion(**json.loads(v.to_json())), name) == other


@pytest.mark.parametrize("name,cli,default,other", _FAMILY, ids=_IDS)
def test_it_is_a_model_version_field_with_the_genre_default(name, cli, default, other):
    fields = {f.name: f for f in dataclasses.fields(ModelVersion)}
    assert name in fields, f"{name} is not a ModelVersion field"
    assert fields[name].default == default


@pytest.mark.parametrize("name,cli,default,other", _FAMILY, ids=_IDS)
def test_it_is_not_gated_by_check_compatible(layout, name, cli, default, other):
    """A frozen eval / pool / distill opponent runs NO loss (and no tripwire callback), so gating
    any of these would be a false rejection that breaks league play."""
    pk = {"net_arch": [512, 512]}
    a = ModelVersion.from_layout_and_policy_kwargs(layout, pk)
    b = ModelVersion.from_layout_and_policy_kwargs(layout, pk, **{name: other})
    a.check_compatible(b)
    b.check_compatible(a)


# ------------------------------------------------------------------------------- 4. the migration

@pytest.mark.parametrize("name,cli,default,other", _FAMILY, ids=_IDS)
def test_a_pre_v103_config_migrates_to_the_genre_default(layout, name, cli, default, other):
    """Not a guess: "kl" is the loss every pre-v103 run trained with, no run had a gate, and the
    tripwire did not exist."""
    v = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
    old = json.loads(v.to_json())
    old.pop(name)
    old["config_version"] = 102

    migrated = _migrate_config(old)
    # `>= 103`, not `== 103` (the cf-family lesson): the property is that a pre-v103 config lands
    # on the LIVE version with this field defaulted, whatever the live version is.
    assert migrated["config_version"] == MODEL_CONFIG_VERSION >= 103
    assert migrated[name] == default
    ModelVersion(**migrated)   # `cls(**data)` must not TypeError


@pytest.mark.parametrize("name,cli,default,other", _FAMILY, ids=_IDS)
def test_a_recorded_value_survives_the_migration_untouched(layout, name, cli, default, other):
    """`setdefault`, not assignment — a G2 config carrying gate="advantage" must not be reset by
    the branch that exists to help pre-v103 ones."""
    v = ModelVersion.from_layout_and_policy_kwargs(
        layout, {"net_arch": [512, 512]}, **{name: other})
    migrated = _migrate_config(json.loads(v.to_json()))
    assert migrated[name] == other


# ----------------------------------------------------------- 5. the §7.5 dependency validations

def _resolve(argv):
    from main.train.config import resolve_config
    from main.train.parser import build_parser

    parser = build_parser()
    # `--use-bridge node`: keep resolve_config from touching the rust sim_bridge binary
    # (the checkpoint_cadence_test convention); irrelevant to the flags under test.
    args = parser.parse_args(argv + ["--steps", "1", "--use-bridge", "node"])
    return resolve_config(args, parser)


@pytest.mark.parametrize("argv", [
    ["--distill-target", "action"],                                   # action without a distill term
    ["--distill-topk", "3"],                                          # top-K without the action form
    ["--distill-topk", "0", "--distill-target", "action"],            # K < 1 is meaningless
    ["--distill-gate", "advantage"],                                  # the gate without the action form
    ["--distill-gate-tau", "0.5"],                                    # tau without the advantage gate
    ["--distill-beta", "0.0", "--distill-target", "action"],          # a non-positive AWR temperature
    ["--rank-tripwire-drop", "1.5"],                                  # not a fractional drop
    ["--rank-tripwire-drop", "0.0"],
], ids=["action_needs_coef", "topk_needs_action", "topk_floor", "gate_needs_action",
        "tau_needs_gate", "beta_positive", "drop_above_one", "drop_zero"])
def test_incoherent_combinations_are_refused(argv):
    with pytest.raises(SystemExit):
        _resolve(argv)


def test_the_validations_read_resolved_values_not_just_typed_ones():
    """The checks run AFTER `_resolve`, so an incoherent combination is refused whether typed or
    inherited — the defaults alone must of course pass."""
    _resolve([])   # a flag-less config resolves cleanly (kl / none / warn are self-coherent)
