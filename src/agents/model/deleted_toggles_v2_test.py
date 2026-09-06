"""The `gen3_dead_flag_purge_v2` deletion contract for `threat_prob_outspeed` (config v108).

WHAT WENT, AND ON WHAT EVIDENCE. The uncertainty-aware P(outspeed) — divide the speed gap by the
believed speed STD instead of a fixed scale (gen3_bidir_threat_trunk_v1 #3, v36) — is deleted with
its flag. The census in `designs/research_state/flag_census_2026-09-06.md` measured all four of the
deletion conditions: never ON in any of the 124 gen-9+ runs, OFF in the production config, no
`state_dict` key depends on it, and no live design doc names it as a lever (zero mentions anywhere
under `designs/research_state/`).

WHY IT IS REFUSED RATHER THAN POPPED, WHICH IS THE POINT OF THIS FILE. Every other member of the
JUDGED list is there because its ON value named PARAMETERS, so popping it would hand SB3 an
unplaceable `state_dict`. This one is the opposite and the more dangerous shape: it built NO
parameters, so a True checkpoint and a False checkpoint are BYTE-IDENTICAL in every key. It only
chose a divisor inside `DamageOperator._p_outspeed`. A silent pop would therefore load cleanly, pass
every shape gate, and run a checkpoint under physics it was never trained on — permanently, with
nothing to notice. "Loads cleanly" is the reason to refuse it, not a reason to allow it.

The two mechanisms are pinned separately because they are read at different moments by different
code: `_migrate_config` handles the config JSON that drives the version GATE, and
`sanitize_dead_extractor_kwargs` handles the zip's pickled `features_extractor_kwargs`, which is
what SB3 splats into the extractor constructor. A key dropped from one and not the other either
fails the gate for the wrong reason or `TypeError`s inside the constructor.
"""
from __future__ import annotations

import dataclasses

import pytest

from agents.model.model_version import (
    ARCH_SIGNATURE, MIGRATION_FLOOR, MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError,
    _migrate_config,
)
from agents.model.snapshot import sanitize_dead_extractor_kwargs

_DELETED = "threat_prob_outspeed"


def _current_config(**extra) -> dict:
    """A minimal AT-FLOOR config, i.e. one this purge can actually be reached through."""
    data = {
        "config_version": MIGRATION_FLOOR, "arch_signature": ARCH_SIGNATURE,
        "species_embedding_dim": 32, "max_species": 400, "move_embedding_dim": 16,
        "max_moves": 400, "item_embedding_dim": 16, "max_items": 600,
        "ability_embedding_dim": 16, "max_abilities": 100, "type_embedding_dim": 16,
        "max_types": 20, "total_dim": 2501, "active_context_dim": 58,
        "role_token_size": 128, "projection_dim": 512, "move_net_hidden": [96, 32],
        "role_encoder_hidden": [256, 128], "net_arch": [512, 512],
    }
    data.update(extra)
    return data


# ------------------------------------------------------------------------------ the dataclass side
def test_the_deleted_field_is_gone_from_the_dataclass():
    fields = {f.name for f in dataclasses.fields(ModelVersion)}
    assert _DELETED not in fields, "still declared on ModelVersion"


def test_the_flag_is_gone_from_the_registry():
    from agents.model.flag_registry import REGISTRY
    assert _DELETED not in {f.name for f in REGISTRY}


def test_the_parser_rejects_the_deleted_flag():
    """argparse must REFUSE it, not silently ignore it — a launch that types a deleted flag has to
    find out at parse time rather than train for hours under a config it did not ask for."""
    import sys
    argv, sys.argv = sys.argv, ["train_rl_agent.py"]
    try:
        from main.train_rl_agent import build_parser
        parser = build_parser()
        assert _DELETED not in {a.dest for a in parser._actions}
        with pytest.raises(SystemExit):
            parser.parse_args(["--threat-prob-outspeed"])
    finally:
        sys.argv = argv


def test_the_combination_check_went_with_it():
    """`--threat-prob-outspeed requires --damage-op` cannot outlive the flag it constrains — a
    dangling check is the shape that makes `main.checkargs` report a rule nothing can satisfy."""
    from main.train.combination_checks import COMBINATION_CHECKS
    assert not [c for c in COMBINATION_CHECKS if _DELETED in c.name or _DELETED in c.dests]


# ----------------------------------------------------------------------------- the config-JSON side
def test_an_off_config_pops_the_key_and_still_loads():
    data = _migrate_config(_current_config(**{_DELETED: False}))
    assert _DELETED not in data, "the stale key must POP or `cls(**data)` TypeErrors on it"
    ModelVersion(**data)                       # must not raise


def test_a_config_with_no_such_key_is_untouched():
    """The overwhelmingly common case — every config written after the deletion."""
    data = _migrate_config(_current_config())
    assert _DELETED not in data
    ModelVersion(**data)


def test_a_config_that_recorded_it_ON_is_refused_with_a_re_read_route():
    with pytest.raises(ModelVersionError) as exc:
        _migrate_config(_current_config(**{_DELETED: True}))
    msg = str(exc.value)
    assert _DELETED in msg
    assert "gen3_dead_flag_purge_v2" in msg
    # A refusal must say how to read the checkpoint ANYWAY. Without this it reads as data loss
    # rather than a code-version boundary — the v75 rule, and the reason every purge carries it.
    assert "git_hash" in msg, (
        "a refusal must name its own metadata.json git_hash as the way back to this checkpoint")


# -------------------------------------------------------------------------- the pickled-kwargs side
def test_off_extractor_kwargs_are_stripped():
    fek = {"damage_op": True, _DELETED: False}
    assert sanitize_dead_extractor_kwargs(fek) is True     # it changed something
    assert _DELETED not in fek
    assert fek["damage_op"] is True                        # and touched nothing else


def test_on_extractor_kwargs_are_refused():
    with pytest.raises(ModelVersionError):
        sanitize_dead_extractor_kwargs({"damage_op": True, _DELETED: True})


# ------------------------------------------------------------------------------------ the stamp
def test_the_signature_did_not_move_but_the_config_version_did():
    """THE SAFETY RULE OF THIS PURGE, asserted rather than promised.

    The flag named no parameters, so no `state_dict` key moves and every existing checkpoint stays
    loadable — which is exactly why `ARCH_SIGNATURE` must NOT be bumped. A bump would refuse every
    gen-17 checkpoint for a deletion that cannot affect them, and (per the floor contract) would
    drag `MIGRATION_FLOOR` up with it in the same commit. The version stamp still advances, because
    the recorded config genuinely changed shape."""
    assert ARCH_SIGNATURE == "gen3_critic_route_wave_v1", (
        "a dead-flag purge that moves no state_dict key must not bump the signature")
    assert MODEL_CONFIG_VERSION >= 108
    assert MIGRATION_FLOOR == 96, "no signature bump ⇒ no floor raise"
