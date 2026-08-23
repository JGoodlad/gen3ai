"""Gates for `gen3_cf_coef_provenance_v1` (config v100) — the counterfactual COEFFICIENT family.

Ten training-only knobs on the counterfactual value-grounding stack leave the `--opd-coef` genre
(argparse-only) for the `td_aux_coef` one (recorded on `ModelVersion` + `_resolve`-inherited).

**What this closes is silent by construction, which is why it needs a test rather than a comment.**
An R1 arm launched with `--cf-winprob-coef 1.0` and resumed flaglessly kept training, kept logging,
and simply stopped applying the term it existed to measure — no error, no FATAL, just a metric that
goes quiet. Worse, the three STRUCTURAL cf flags (`cf_evidential` v98, `cf_twin_heads` /
`cf_shadow_critic` v99) were already recorded and GATED, so before this bump a resume could keep the
HEAD and lose the COEFFICIENT that drives it — a head that exists, costs parameters, and does
nothing.

Four properties carry it:

1. **Every flag's argparse default is `None`** — the precondition for `_resolve` to fire at all.
   (The general form of that is `flag_registry_test.test_cli_flags_argparse_default_is_none`; here
   it is asserted directly on the family, because the registry does not carry these rows.)
2. **Each is a recorded `ModelVersion` field** that round-trips through JSON.
3. **None is GATED** — `check_compatible` must accept two versions differing on all ten, or a
   frozen eval/pool/distill opponent (which runs no loss at all) would be refused.
4. **The v100 migration defaults a pre-v100 config** to the argparse defaults — not a guess: before
   v100 the fields did not exist, so a flagless resume necessarily got exactly those values.
"""
import dataclasses
import json

import pytest

from agents.model.model_version import (
    MODEL_CONFIG_VERSION,
    ModelVersion,
    _migrate_config,
)

# field -> (the value the CLI defaults to, a DISTINCT value to prove the field carries content)
CF_COEF_FIELDS = {
    "cf_records": (False, True),
    "cf_records_keep": (512, 64),
    "cf_winprob_coef": (0.0, 1.0),
    "cf_head_only": (True, False),
    "cf_label_lag_steps": (150_000, 4_242),
    "cf_label_likelihood": ("binomial", "bce"),
    "cf_evidential_coef": (0.0, 0.5),
    "cf_evidential_reg": (1e-3, 0.25),
    "cf_twin_coef": (0.0, 2.0),
    "cf_shadow_coef": (0.0, 0.75),
}

# The CLI spelling each field is set by. Kept explicit rather than derived from the field name so a
# renamed flag fails here instead of silently testing a flag that no longer exists.
CF_CLI_FLAG = {
    "cf_records": "--cf-records",
    "cf_records_keep": "--cf-records-keep",
    "cf_winprob_coef": "--cf-winprob-coef",
    "cf_head_only": "--cf-head-only",
    "cf_label_lag_steps": "--cf-label-lag-steps",
    "cf_label_likelihood": "--cf-label-likelihood",
    "cf_evidential_coef": "--cf-evidential-coef",
    "cf_evidential_reg": "--cf-evidential-reg",
    "cf_twin_coef": "--cf-twin-coef",
    "cf_shadow_coef": "--cf-shadow-coef",
}


@pytest.fixture(scope="module")
def layout():
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    return Gen3ObservationEncoder(load_mappings()).get_layout()


# ------------------------------------------------------------------------ 1. the argparse surface

@pytest.mark.parametrize("field", sorted(CF_COEF_FIELDS))
def test_the_cli_flag_defaults_to_none_so_a_flagless_resume_can_inherit(field):
    """`_resolve` fires on `getattr(args, name) is None`. A hard argparse default would OVERWRITE
    the value the run is already training with, on every 3-hour launcher restart."""
    from main.train.parser import build_parser

    assert getattr(build_parser().parse_args([]), field) is None


@pytest.mark.parametrize("field", sorted(CF_COEF_FIELDS))
def test_the_flag_has_a_resolve_line(field):
    """Present AND reachable — the test above supplies the reachability half."""
    import inspect
    import re

    from main.train.config import resolve_config

    names = set(re.findall(r"_resolve\(\s*\"([a-z0-9_]+)\"", inspect.getsource(resolve_config)))
    assert field in names, (
        f"no `_resolve({field!r}, ...)` in resolve_config — a flagless resume will not inherit it")


@pytest.mark.parametrize("field", sorted(CF_COEF_FIELDS))
def test_the_flag_still_parses_an_explicit_value(field):
    """The None default must not have cost the flag its ability to be SET."""
    from main.train.parser import build_parser

    _default, other = CF_COEF_FIELDS[field]
    argv = [CF_CLI_FLAG[field], str(other).lower() if isinstance(other, bool) else str(other)]
    parsed = getattr(build_parser().parse_args(argv), field)
    assert type(parsed)(parsed) == type(parsed)(other) and bool(parsed) == bool(other)


# -------------------------------------------------------------------------- 2. recorded, 3. never gated

@pytest.mark.parametrize("field", sorted(CF_COEF_FIELDS))
def test_the_field_is_recorded_and_round_trips(layout, field):
    _default, other = CF_COEF_FIELDS[field]
    v = ModelVersion.from_layout_and_policy_kwargs(
        layout, {"net_arch": [512, 512]}, **{field: other})
    assert getattr(v, field) == other
    assert getattr(ModelVersion(**json.loads(v.to_json())), field) == other


def test_every_family_member_is_a_model_version_field():
    """One assertion over the whole family, so ADDING a cf coefficient without recording it fails
    here rather than the next time somebody resumes an arm."""
    fields = {f.name for f in dataclasses.fields(ModelVersion)}
    missing = sorted(set(CF_COEF_FIELDS) - fields)
    assert not missing, f"not ModelVersion fields: {missing}"


def test_none_of_them_is_gated_by_check_compatible(layout):
    """A frozen eval / pool / distill opponent runs NO loss, so gating a loss coefficient there
    would be a false rejection that breaks league play. All ten differ at once, deliberately: a
    gate on any single one would fail this."""
    pk = {"net_arch": [512, 512]}
    a = ModelVersion.from_layout_and_policy_kwargs(layout, pk)
    b = ModelVersion.from_layout_and_policy_kwargs(
        layout, pk, **{f: other for f, (_d, other) in CF_COEF_FIELDS.items()})
    a.check_compatible(b)
    b.check_compatible(a)


# ------------------------------------------------------------------------------- 4. the migration

def test_a_pre_v100_config_migrates_to_the_argparse_defaults(layout):
    """Not a guess: the fields did not exist before v100, so a flagless resume of any such run got
    exactly the argparse default. Reachable because v99 is above MIGRATION_FLOOR (96)."""
    v = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
    old = json.loads(v.to_json())
    for field in CF_COEF_FIELDS:
        old.pop(field)
    old["config_version"] = 99

    migrated = _migrate_config(old)
    # `>= 100`, not `== 100`: the property under test is that a pre-v100 config lands on the LIVE
    # version with this family defaulted. Pinning the live number here made every later, unrelated
    # schema bump fail in the counterfactual suite (it did, at v101).
    assert migrated["config_version"] == MODEL_CONFIG_VERSION >= 100
    for field, (default, _other) in CF_COEF_FIELDS.items():
        assert migrated[field] == default, field
    ModelVersion(**migrated)   # the whole point of the setdefault: `cls(**data)` must not TypeError


def test_a_recorded_value_survives_the_migration_untouched(layout):
    """`setdefault`, not assignment — a v100 config that already carries a live coefficient must
    not be reset to OFF by the branch that exists to help pre-v100 ones."""
    v = ModelVersion.from_layout_and_policy_kwargs(
        layout, {"net_arch": [512, 512]},
        **{f: other for f, (_d, other) in CF_COEF_FIELDS.items()})
    migrated = _migrate_config(json.loads(v.to_json()))
    for field, (_default, other) in CF_COEF_FIELDS.items():
        assert migrated[field] == other, field
