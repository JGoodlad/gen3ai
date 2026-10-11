"""`--trunk-precision` (`gen3_trunk_precision_flag_v1`, config v155; T25 lever A): the flag's surface and its two
invariants.

* parse / resolve / inherit / migrate: argparse default None, a fresh run resolves to fp32, a flagless resume
  INHERITS the recorded value, a typed one wins, a pre-v155 config migrates to fp32, the field is recorded;
* `apply_training_hparams` sets the live policy's rounds (both build paths call it) and a namespace without the
  dest is fp32 — the learner golden is untouched;
* `fp32_scope` restores EXACTLY the classes it found (also on an exception) and is a no-op at fp32;
* THE K9(b) PROBE READS THE fp32 FUNCTION on a bf16 learner: `probe_forward` returns bit-identical log-probs whether
  the policy's rounds are bf16 or fp32 (T2 serves fp32), and the rounds are bf16 again afterwards.
"""
from __future__ import annotations

import argparse

import numpy as np
import pytest
import torch

from agents.model import trunk_precision as TP
from agents.model.team_transformer import BiasedEncoderLayer
from main.train.parser import build_parser


def test_the_argparse_default_is_None_and_the_choices_are_closed():
    assert build_parser().parse_args([]).trunk_precision is None
    assert build_parser().parse_args(["--trunk-precision", "bf16"]).trunk_precision == "bf16"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--trunk-precision", "fp16"])


def test_a_flagless_resume_INHERITS_the_recorded_precision_and_a_typed_one_overrides_it():
    import inspect

    import main.train.config as cfg
    from agents.model.model_version import ModelVersion
    from main.train.config import inherit_saved_flag
    saved = ModelVersion.__new__(ModelVersion)
    saved.trunk_precision = "bf16"
    args = build_parser().parse_args([])
    assert inherit_saved_flag(args, saved, "trunk_precision", "fp32") is True
    assert args.trunk_precision == "bf16"
    args = build_parser().parse_args(["--trunk-precision", "fp32"])
    assert inherit_saved_flag(args, saved, "trunk_precision", "fp32") is False
    assert args.trunk_precision == "fp32"
    fresh = build_parser().parse_args([])
    assert inherit_saved_flag(fresh, None, "trunk_precision", "fp32") is False and fresh.trunk_precision == "fp32"
    assert '_resolve("trunk_precision", "fp32")' in inspect.getsource(cfg)


def test_it_is_a_RECORDED_field_and_a_pre_v155_config_migrates_to_fp32():
    from agents.model.model_version.constants import MODEL_CONFIG_VERSION
    from agents.model.model_version.fields import ModelVersionFields
    from agents.model.model_version.migrations import _migrate_config
    assert MODEL_CONFIG_VERSION >= 155
    assert ModelVersionFields.__dataclass_fields__["trunk_precision"].default == "fp32"
    old = _migrate_config({"config_version": 154})
    assert old["trunk_precision"] == "fp32" and old["config_version"] >= 155
    kept = _migrate_config({"config_version": 154, "trunk_precision": "bf16"})
    assert kept["trunk_precision"] == "bf16"


class _Model:
    def __init__(self) -> None:
        self.policy = torch.nn.ModuleList([BiasedEncoderLayer(), BiasedEncoderLayer()])


def test_apply_training_hparams_sets_the_live_rounds_and_a_namespace_without_the_dest_is_fp32():
    m = _Model()
    assert TP.apply_run_precision(m, "bf16") == 2
    assert m.trunk_precision == "bf16" and TP.trunk_precision_of(m.policy) == "bf16"
    assert TP.apply_run_precision(m, None) == 2
    assert m.trunk_precision == "fp32" and TP.trunk_precision_of(m.policy) == "fp32"
    with pytest.raises(ValueError):
        TP.apply_run_precision(m, "fp16")
    import inspect

    import main.train.model_build as MB
    src = inspect.getsource(MB.apply_training_hparams)
    assert 'apply_run_precision(model, getattr(args, "trunk_precision", None))' in src


def test_the_learner_golden_builds_fp32():
    from agents.training import learner_golden as LG
    model = LG.build_learner()
    assert getattr(model, "trunk_precision", "fp32") == "fp32"
    assert TP.trunk_precision_of(model.policy) == "fp32"


def test_fp32_scope_restores_EXACTLY_the_classes_it_found_also_on_an_exception():
    m = torch.nn.ModuleList([BiasedEncoderLayer(), BiasedEncoderLayer()])
    with TP.fp32_scope(m):                                    # a no-op at fp32
        assert TP.trunk_precision_of(m) == "fp32"
    TP.set_trunk_precision(m, "bf16")
    with TP.fp32_scope(m):
        assert TP.trunk_precision_of(m) == "fp32"
        assert all(type(r) is BiasedEncoderLayer for r in m)
    assert TP.trunk_precision_of(m) == "bf16"
    with pytest.raises(RuntimeError, match="boom"):
        with TP.fp32_scope(m):
            raise RuntimeError("boom")
    assert all(type(r) is TP.Bf16BiasedEncoderLayer for r in m)


def test_the_K9b_probe_reads_the_fp32_function_on_a_bf16_learner():
    from agents.training import learner_golden as LG
    from agents.training.rust_rollout import consistency as C
    model = LG.build_learner()
    LG.load_buffer_into(model)
    buf = model.rollout_buffer
    n = 16
    obs = {k: np.asarray(v).reshape(-1, *np.asarray(v).shape[2:])[:n] for k, v in buf.observations.items()}
    acts = torch.as_tensor(np.asarray(buf.actions).reshape(-1)[:n]).long()
    masks = torch.as_tensor(np.asarray(buf.action_masks).reshape(-1, np.asarray(buf.action_masks).shape[-1])[:n])
    ref, _ = C.probe_forward(model, obs, acts, masks)
    TP.set_trunk_precision(model.policy, "bf16")
    got, _ = C.probe_forward(model, obs, acts, masks)
    assert TP.trunk_precision_of(model.policy) == "bf16"
    np.testing.assert_array_equal(got, ref)
    # the teeth: without the scope the bf16 forward moves the log-probs (so the equality above is not vacuous)
    with torch.no_grad():
        _v, lp, _e = model.policy.evaluate_actions(
            {k: torch.as_tensor(v) for k, v in obs.items()}, acts, action_masks=masks)
    assert not np.array_equal(lp.double().numpy(), ref)


def test_the_parser_help_names_the_flag_in_the_hyperparameter_group():
    p = build_parser()
    acts = {a.dest: a for a in p._actions if isinstance(a, argparse.Action)}
    assert acts["trunk_precision"].choices == ("fp32", "bf16")


def test_a_T2_slot_replica_SERVES_fp32_whatever_the_learner_trains_with():
    """Design (b): T2 deep-copies the learner policy into its slots (`slots.served_replica`); a bf16 learner's
    round classes must NOT ride into T2 (`k9_bf16_2026-10-10` FINDING 7), and the template keeps its own."""
    from agents.inference.service.slots import served_replica
    m = torch.nn.ModuleList([BiasedEncoderLayer(), BiasedEncoderLayer()])
    TP.set_trunk_precision(m, "bf16")
    r = served_replica(m)
    assert TP.trunk_precision_of(r) == "fp32" and all(type(x) is BiasedEncoderLayer for x in r)
    assert TP.trunk_precision_of(m) == "bf16"
