"""Gates for `gen3_pg_coef_v1` (config v102) — the policy-gradient term's own weight.

`--pg-coef` multiplies ONLY the clipped PPO surrogate (`pg_coef * policy_loss`) in the loss
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

    assert build_parser().parse_args([]).pg_coef is None


def test_the_flag_has_a_resolve_line():
    import inspect
    import re

    from main.train.config import resolve_config

    names = set(re.findall(r"_resolve\(\s*\"([a-z0-9_]+)\"", inspect.getsource(resolve_config)))
    assert "pg_coef" in names, (
        "no `_resolve('pg_coef', ...)` in resolve_config — a flagless resume will not inherit it")


def test_the_flag_still_parses_an_explicit_value():
    from main.train.parser import build_parser

    assert build_parser().parse_args(["--pg-coef", "0.0"]).pg_coef == 0.0
    assert build_parser().parse_args(["--pg-coef", "0.5"]).pg_coef == 0.5


def test_a_negative_value_is_refused():
    """A negative coef would ASCEND the surrogate. 0.0 is the intended floor, and this parser
    check is the only gate (training-only, never version-checked)."""
    from main.train.config import resolve_config
    from main.train.parser import build_parser

    parser = build_parser()
    # `--use-bridge node`: keep resolve_config from touching the rust sim_bridge binary
    # (the checkpoint_cadence_test convention); irrelevant to the coefficient under test.
    args = parser.parse_args(["--pg-coef", "-0.5", "--steps", "1", "--use-bridge", "node"])
    with pytest.raises(SystemExit):
        resolve_config(args, parser)


# -------------------------------------------------------------------------- 2. recorded, 3. never gated

def test_the_field_is_recorded_and_round_trips(layout):
    v = ModelVersion.from_layout_and_policy_kwargs(
        layout, {"net_arch": [512, 512]}, pg_coef=_OTHER)
    assert getattr(v, "pg_coef") == _OTHER
    assert getattr(ModelVersion(**json.loads(v.to_json())), "pg_coef") == _OTHER


def test_it_is_a_model_version_field_with_the_upstream_default():
    fields = {f.name: f for f in dataclasses.fields(ModelVersion)}
    assert "pg_coef" in fields, "pg_coef is not a ModelVersion field"
    assert fields["pg_coef"].default == _DEFAULT


def test_it_is_not_gated_by_check_compatible(layout):
    """A frozen eval / pool / distill opponent runs NO loss, so gating a loss coefficient there
    would be a false rejection that breaks league play."""
    pk = {"net_arch": [512, 512]}
    a = ModelVersion.from_layout_and_policy_kwargs(layout, pk)
    b = ModelVersion.from_layout_and_policy_kwargs(layout, pk, pg_coef=_OTHER)
    a.check_compatible(b)
    b.check_compatible(a)


# ------------------------------------------------------------------------------- 4. the migration

def test_a_pre_v102_config_migrates_to_upstream_one(layout):
    """Not a guess: the policy-gradient term entered the loss at an implicit 1.0 in every
    pre-v102 run, so 1.0 is what every such run trained with."""
    v = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
    old = json.loads(v.to_json())
    old.pop("pg_coef")
    old["config_version"] = 101

    migrated = _migrate_config(old)
    # `>= 102`, not `== 102` (the cf-family lesson): the property is that a pre-v102 config
    # lands on the LIVE version with this field defaulted, whatever the live version is.
    assert migrated["config_version"] == MODEL_CONFIG_VERSION >= 102
    assert migrated["pg_coef"] == _DEFAULT
    ModelVersion(**migrated)   # `cls(**data)` must not TypeError


def test_a_recorded_value_survives_the_migration_untouched(layout):
    """`setdefault`, not assignment — an arm-F config carrying 0.0 must not be reset to 1.0 by
    the branch that exists to help pre-v102 ones."""
    v = ModelVersion.from_layout_and_policy_kwargs(
        layout, {"net_arch": [512, 512]}, pg_coef=_OTHER)
    migrated = _migrate_config(json.loads(v.to_json()))
    assert migrated["pg_coef"] == _OTHER
