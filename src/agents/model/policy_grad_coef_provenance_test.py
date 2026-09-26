"""Gates for `gen3_policy_grad_coef_v1` (config v102) — the policy-gradient term's own weight.

`--policy-grad-coef` multiplies ONLY the clipped PPO surrogate (`policy_grad_coef * policy_loss`) in the loss
fold; 1.0 (the default) is the upstream expression byte-for-byte and 0.0 is the arm-F
pure-distill/aux phase the flag exists for. It is the `td_aux_coef` provenance genre exactly:
TRAINING-only, recorded on `ModelVersion` for provenance, `_resolve`-inherited on a flagless
resume, and NEVER gated by `check_compatible` — a frozen eval/pool/distill opponent runs no
loss at all, so gating a loss coefficient there would be a false rejection.

The four properties, mirroring `cf_coef_provenance_test.py` (the v100 family's gate):

1. the argparse default is `None` — the precondition for `_resolve` to fire at all (the v100
   lesson: a `_resolve` line beside a hard argparse default is DEAD CODE that passes presence
   tests);
2. the field is recorded on `ModelVersion` and round-trips through JSON;
3. it is never gated — two versions differing on it are mutually compatible;
4. the v102 migration defaults a pre-v102 config to 1.0 — not a guess: the term entered the
   loss at an implicit 1.0 in every run ever made.

The loss-fold behavior itself (byte-identity at 1.0, exact removal at 0.0) is pinned in
`agents/training/instrumented_ppo_test.py`.
"""
import dataclasses
import json

import pytest

from agents.model.model_version import (
    MODEL_CONFIG_VERSION,
    ModelVersion,
    ModelVersionError,
    _migrate_config,
)

_DEFAULT, _OTHER = 1.0, 0.0   # 0.0 is the distinct value that matters: the arm-F phase


@pytest.fixture(scope="module")
def layout():
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    return Gen3ObservationEncoder(load_mappings()).get_layout()


# ------------------------------------------------------------------------ 1. the argparse surface

def test_the_cli_flag_defaults_to_none_so_a_flagless_resume_can_inherit():
    """`_resolve` fires on `getattr(args, name) is None`. A hard argparse default would OVERWRITE
    the value the run is already training with, on every 3-hour launcher restart."""
    from main.train.parser import build_parser

    assert build_parser().parse_args([]).policy_grad_coef is None


def test_the_flag_has_a_resolve_line():
    import inspect
    import re

    from main.train.config import resolve_config

    names = set(re.findall(r"_resolve\(\s*\"([a-z0-9_]+)\"", inspect.getsource(resolve_config)))
    assert "policy_grad_coef" in names, (
        "no `_resolve('policy_grad_coef', ...)` in resolve_config — a flagless resume will not inherit it")


def test_the_flag_still_parses_an_explicit_value():
    from main.train.parser import build_parser

    assert build_parser().parse_args(["--policy-grad-coef", "0.0"]).policy_grad_coef == 0.0
    assert build_parser().parse_args(["--policy-grad-coef", "0.5"]).policy_grad_coef == 0.5


def test_a_negative_value_is_refused():
    """A negative coef would ASCEND the surrogate. 0.0 is the intended floor, and this parser
    check is the only gate (training-only, never version-checked)."""
    from main.train.config import resolve_config
    from main.train.parser import build_parser

    parser = build_parser()
    # `--use-bridge node`: keep resolve_config from touching the rust sim_bridge binary
    # (the checkpoint_cadence_test convention); irrelevant to the coefficient under test.
    args = parser.parse_args(["--policy-grad-coef", "-0.5", "--steps", "1", "--use-bridge", "node"])
    with pytest.raises(SystemExit):
        resolve_config(args, parser)


# -------------------------------------------------------------------------- 2. recorded, 3. never gated

def test_the_field_is_recorded_and_round_trips(layout):
    v = ModelVersion.from_layout_and_policy_kwargs(
        layout, {"net_arch": [512, 512]}, policy_grad_coef=_OTHER)
    assert getattr(v, "policy_grad_coef") == _OTHER
    assert getattr(ModelVersion(**json.loads(v.to_json())), "policy_grad_coef") == _OTHER


def test_it_is_a_model_version_field_with_the_upstream_default():
    fields = {f.name: f for f in dataclasses.fields(ModelVersion)}
    assert "policy_grad_coef" in fields, "policy_grad_coef is not a ModelVersion field"
    assert fields["policy_grad_coef"].default == _DEFAULT


def test_it_is_not_gated_by_check_compatible(layout):
    """A frozen eval / pool / distill opponent runs NO loss, so gating a loss coefficient there
    would be a false rejection that breaks league play."""
    pk = {"net_arch": [512, 512]}
    a = ModelVersion.from_layout_and_policy_kwargs(layout, pk)
    b = ModelVersion.from_layout_and_policy_kwargs(layout, pk, policy_grad_coef=_OTHER)
    a.check_compatible(b)
    b.check_compatible(a)


# ------------------------------------------------------------------------------- 4. the migration

def test_a_pre_v102_config_is_refused_and_a_current_one_records_upstream_one(layout):
    """The v102 `setdefault("policy_grad_coef", 1.0)` branch is FLOORED AWAY: gen3_event_record_v2
    raised MIGRATION_FLOOR to 121 (archived verbatim in `_migrate_config`'s v97–v120 history), so a
    config lacking the field is pre-generation and is REFUSED with the diagnosis. The surviving
    property: a fresh current config RECORDS the coefficient explicitly at upstream's implicit 1.0,
    and the current migration passes it through to a constructible ModelVersion."""
    v = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
    old = json.loads(v.to_json())
    old.pop("policy_grad_coef")
    old["config_version"] = 101
    with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
        _migrate_config(old)

    fresh = json.loads(v.to_json())
    assert fresh["config_version"] == MODEL_CONFIG_VERSION >= 121
    assert fresh["policy_grad_coef"] == _DEFAULT
    migrated = _migrate_config(fresh)
    assert migrated["policy_grad_coef"] == _DEFAULT
    ModelVersion(**migrated)   # `cls(**data)` must not TypeError


def test_a_recorded_value_survives_the_migration_untouched(layout):
    """`setdefault`, not assignment — an arm-F config carrying 0.0 must not be reset to 1.0 by
    the branch that exists to help pre-v102 ones."""
    v = ModelVersion.from_layout_and_policy_kwargs(
        layout, {"net_arch": [512, 512]}, policy_grad_coef=_OTHER)
    migrated = _migrate_config(json.loads(v.to_json()))
    assert migrated["policy_grad_coef"] == _OTHER
