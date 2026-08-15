"""The `config_only` tier's contract, end to end.

A demoted toggle keeps every role a flag ever had EXCEPT the one it lost. The whole claim is that
removing the argparse entry costs nothing in explicitness, so the three surviving properties are
asserted directly rather than argued:

  1. **RECORDED** — a fresh run's `model_config.json` carries the frozen value. If it did not, the
     demotion would have quietly turned a version-checked architecture toggle into an untracked
     constant, which is the exact failure the registry exists to prevent.
  2. **GATED** — a resume whose saved config disagrees is still a hard `ModelVersionError`, from
     the SAME `check_compatible` compare as before. The demotion touched the tier, not the class.
  3. **UNSETTABLE** — there is no argparse entry, and no `_resolve` line reading an attribute that
     no longer exists. A "frozen" default that a flag could still override would be a lie.

Plus the two that make the tier worth having at all:

  4. the extractor's **constructor kwarg survives**, so the non-frozen value stays reachable for an
     experiment or a probe — the tier removes the launch surface, not the capability;
  5. the frozen value **wins over a stale `args` attribute**, so a leftover namespace entry (a
     resumed argparse namespace, a test stub) cannot silently un-freeze it.

`flag_registry_test.py` checks these per-flag against the registry table; this file proves the
PATTERN once, through the real save/load path, so the contract is pinned by behaviour and not only
by a source scan.
"""
from __future__ import annotations

import dataclasses
import inspect
import json
import os
import re

import pytest

from agents.model import extractor_arch as EA
from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.model.flag_registry import Tier, config_only_flags
from agents.model.model_version import ModelVersion, ModelVersionError
from agents.model.snapshot import save_model_snapshot

_TRAIN_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "main", "train_rl_agent.py")


def _a_config_only_flag():
    flags = config_only_flags()
    assert flags, (
        "no config_only rows in the registry — this file pins a PATTERN, so it needs at least one "
        "demoted toggle. If the last demotion was reverted, revert this file with it.")
    return flags[0]


def _version_with(**overrides) -> ModelVersion:
    """A minimal ModelVersion; only the fields under test matter to the compares exercised here."""
    base = dict(
        config_version=1, arch_signature="test_sig",
        species_embedding_dim=1, max_species=1, move_embedding_dim=1, max_moves=1,
        item_embedding_dim=1, max_items=1, ability_embedding_dim=1, max_abilities=1,
        type_embedding_dim=1, max_types=1, total_dim=1, active_context_dim=1,
        role_token_size=1, projection_dim=1, move_net_hidden=[1], role_encoder_hidden=[1],
        n_history_turns=1, net_arch=[1],
    )
    base.update(overrides)
    return ModelVersion(**base)


# ------------------------------------------------------------------ 1. RECORDED in a fresh config
def test_a_config_only_toggle_is_recorded_in_model_config_json(tmp_path):
    """The frozen value must reach `model_config.json` on a fresh save — that file IS the record."""
    frozen = {f.name: f.default for f in config_only_flags()}
    version = _version_with(**frozen)
    save_model_snapshot(str(tmp_path), version, git_hash="config-only-test")

    with open(tmp_path / "model_config.json") as fh:
        written = json.load(fh)
    for name, value in frozen.items():
        assert name in written, (
            f"{name!r} is tier=config_only but absent from a freshly written model_config.json — "
            f"the demotion dropped the RECORD role, not just the SELECT role.")
        assert written[name] == value, f"{name}: recorded {written[name]!r}, frozen at {value!r}"


def test_the_recorded_value_round_trips_through_from_json_file(tmp_path):
    """And it must survive the read back — a migration that popped it would break the gate below.

    Stamped at the LIVE `MODEL_CONFIG_VERSION` so `_migrate_config` takes the no-op path; the point
    here is the field surviving `save -> from_json_file`, not migration behaviour.
    """
    from agents.model.model_version import MODEL_CONFIG_VERSION
    frozen = {f.name: f.default for f in config_only_flags()}
    version = _version_with(config_version=MODEL_CONFIG_VERSION, **frozen)
    save_model_snapshot(str(tmp_path), version, git_hash="config-only-test")
    reloaded = ModelVersion.from_json_file(str(tmp_path / "model_config.json"))
    for name, value in frozen.items():
        assert getattr(reloaded, name) == value


# --------------------------------------------------------------------- 2. GATED on a bad resume
@pytest.mark.parametrize("flag", config_only_flags(), ids=lambda f: f.name)
def test_a_mismatched_resume_is_still_rejected(flag):
    """The demotion must not have loosened the gate: a saved config that disagrees still FATALs.

    The value flipped here is the OTHER one — whatever the frozen default is not — because that is
    precisely the checkpoint a demotion could silently start accepting.
    """
    other = (not flag.default) if isinstance(flag.default, bool) else f"{flag.default}-other"
    current = _version_with(**{flag.name: flag.default})
    saved = _version_with(**{flag.name: other})
    with pytest.raises(ModelVersionError) as exc:
        current.check_compatible(saved)
    assert flag.name in str(exc.value), (
        f"the resume was rejected, but the message does not name {flag.name!r} — a demoted toggle "
        f"must still explain itself, since the user can no longer see the flag in --help.")


@pytest.mark.parametrize("flag", config_only_flags(), ids=lambda f: f.name)
def test_a_matching_resume_is_accepted(flag):
    """The other half: the gate must not reject the value every CLI run now produces."""
    current = _version_with(**{flag.name: flag.default})
    saved = _version_with(**{flag.name: flag.default})
    current.check_compatible(saved)          # must not raise


# ------------------------------------------------------------------------ 3. UNSETTABLE from a CLI
@pytest.mark.parametrize("flag", config_only_flags(), ids=lambda f: f.name)
def test_no_argparse_entry_and_no_resolve_line(flag):
    with open(_TRAIN_PY) as fh:
        src = fh.read()
    opts = set()
    for call in re.finditer(r"parser\.add_argument\(\s*((?:\"[^\"]*\"\s*,?\s*)+)", src):
        opts.update(m.group(1) for m in re.finditer(r"\"(--[a-zA-Z0-9_-]+)\"", call.group(1)))
    assert flag.cli_flag not in opts, f"{flag.cli_flag} is still settable — the demotion is a lie"
    assert flag.arg not in set(re.findall(r"_resolve\(\s*\"([a-z0-9_]+)\"", src)), (
        f"_resolve({flag.arg!r}) survives but the argparse entry does not, so it reads an "
        f"attribute that no longer exists — an AttributeError on every launch.")


# ----------------------------------------------------- 4. the CONSTRUCTOR kwarg is still the surface
@pytest.mark.parametrize("flag", config_only_flags(), ids=lambda f: f.name)
def test_the_extractor_constructor_still_accepts_it(flag):
    """The tier removes the LAUNCH surface, not the capability — an experiment must still reach it."""
    params = inspect.signature(Gen3FeaturesExtractor.__init__).parameters
    assert flag.name in params, (
        f"{flag.name!r} is tier=config_only but is no longer an extractor constructor kwarg, so "
        f"the non-frozen value is unreachable by ANY route. config_only keeps the experiment "
        f"surface; if the toggle is genuinely gone, delete the registry row too.")


@pytest.mark.parametrize("flag", config_only_flags(), ids=lambda f: f.name)
def test_it_is_still_a_recorded_model_version_field(flag):
    assert flag.name in {f.name for f in dataclasses.fields(ModelVersion)}


# ------------------------------------------------------- 5. the frozen value beats a stale attribute
def test_frozen_value_is_not_overridable_by_a_leftover_args_attribute():
    """A resumed namespace or a test stub can still carry the attribute; it must not win."""
    import types
    flag = _a_config_only_flag()
    stub = types.SimpleNamespace(**{attr: None for attr in EA.ARCH_ARG_KEYS.values()})
    stub.opp_belief_aux_coef = 0.0
    stub.opp_intent_coef = 0.0
    setattr(stub, flag.arg, "A-STALE-VALUE-FROM-AN-OLD-NAMESPACE")
    built = EA.build_extractor_arch_kwargs(stub)
    assert built[flag.name] == flag.default
    # and the version gate must be handed the same value the extractor was built with
    assert EA.arch_toggles_from_args(stub)[flag.name] == flag.default


def test_config_only_flags_are_absent_from_arch_arg_keys():
    """They must be sourced from FROZEN_ARCH_KWARGS, never from an `args` read."""
    for f in config_only_flags():
        assert f.name not in EA.ARCH_ARG_KEYS
        assert f.name in EA.FROZEN_ARCH_KWARGS
        assert f.tier is Tier.CONFIG_ONLY
