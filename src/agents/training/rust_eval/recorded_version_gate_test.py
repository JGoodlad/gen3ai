"""THE RECORDED-VERSION GATE (gen3_recorded_version_gate_v1, 2026-10-11): a POOL EVAL's sentinel load gates each
snapshot against the run's RECORDED ``ModelVersion`` (``snapshot.recorded_model_version``: the model's own policy
kwargs), never one rebuilt from toggles read off the built modules.

The incident: the closing test's ``--arch endstate`` seed ``rb_es_end_s2001`` (pinned ``b132b099``) FATALed at its
first self-play eval after the pool filled (~6M steps) with ``intent_move_cell mismatch: saved=True, current=False``.
``--move-resolution on`` BUILT the seven per-action blocks and then RETIRED them, so a built-module read said False
where the run's ``model_config.json`` and every snapshot it wrote recorded True. (The blocks, their fields and the
retirement were deleted in legacy deletion U3; the rule — read the RECORDED kwargs, never the built modules — is
general and stays pinned here on every surviving gated field.)

What a revert breaks:

* ``arch_toggles_from_model`` reading the built modules again → a toggle the built extractor does not carry reads
  its default, not the recorded value, and ``load_sentinels`` can raise ``ModelVersionError``;
* ``load_sentinels`` rebuilding its version any other way than from the recorded kwargs → the same refusal.

Each arm's model is built from the RESOLVED ``--arch <arm>`` namespace (the arm overlay applied as a launch applies
it), its snapshot written beside a ``model_config.json`` built the way the TRAINER builds the one it saves
(``ModelVersion.from_layout_and_policy_kwargs`` over the fresh run's policy kwargs, ``main.train.model_build``), and
the pool eval's own sentinel loader (``rust_eval.launch.load_sentinels``) loads it from a non-empty pool.

Run: python -m pytest src/agents/training/rust_eval/recorded_version_gate_test.py -q
"""
from __future__ import annotations

import contextlib
import functools
import io
import types
from typing import Any, Tuple

import pytest

#: The seven per-action blocks `--move-resolution on` built and then retired — deleted with their fields in legacy
#: deletion U3, so no recorded version carries them any more.
DELETED_BY_U3 = ("intent_move_cell", "intent_threshold", "intent_conditional", "pair_outcome_cell",
                 "pair_outcome_switch", "switch_branch_cell", "conditional_threat_cell")


@functools.lru_cache(maxsize=None)
def _arm_args_cached(arm: str) -> Any:
    from main.train.config import resolve_config
    from main.train.parser import build_parser

    parser = build_parser()
    a = parser.parse_args(["--steps", "1", "--arch", arm])
    with contextlib.redirect_stdout(io.StringIO()):
        resolve_config(a, parser)
    return a


def arm_args(arm: str) -> Any:
    """The resolved training namespace of a real FRESH ``--arch <arm>`` launch (a fresh deep copy)."""
    import copy

    return copy.deepcopy(_arm_args_cached(arm))


def write_arm_snapshot(dst: Any, arm: str, seed: int) -> Tuple[Any, str]:
    """``(model, zip)``: a seeded PERTURBED-fresh ``--arch <arm>`` model saved as ``dst/snapshot_<seed>.zip`` beside
    a ``model_config.json`` built as the TRAINER builds the one it saves — never through the helper under test."""
    return write_snapshot(dst, arm_args(arm), seed)


def write_snapshot(dst: Any, args: Any, seed: int) -> Tuple[Any, str]:
    """``write_arm_snapshot`` for an explicit resolved training namespace ``args`` (an architecture that is no
    ``--arch`` arm, e.g. production + ``--trunk-layers 2`` in ``main/h2h/recorded_version_cross_test.py``)."""
    from agents.model.model_version import ModelVersion
    from agents.model.parity_probe import PERTURB_SCALE, perturb_
    from agents.training.rust_eval import offline as PAR
    from main.fresh_checkpoint import build_fresh_model

    dst.mkdir(parents=True, exist_ok=True)
    with PAR.declared_torch_state(1):
        model, _, pk = build_fresh_model(seed, args=args)
        perturb_(model.policy, seed=3000 + seed, scale=PERTURB_SCALE)
        path = dst / f"snapshot_{seed:012d}.zip"
        model.save(str(path))
    cfg = ModelVersion.from_layout_and_policy_kwargs(pk["features_extractor_kwargs"]["layout"], pk,
                                                     vf_coef=float(args.vf_coef))
    (dst / "model_config.json").write_text(cfg.to_json())
    return model, str(path)


@pytest.mark.slow  # ship gate (2026-10-19): 9 s beside a live run
@pytest.mark.parametrize("arm", ["endstate", "production"])     # `static_recovery` was deleted (legacy deletion U0)
def test_a_pool_eval_loads_the_runs_own_sentinel_under_every_named_arm(tmp_path, arm):
    from agents.model.model_version import ModelVersion
    from agents.model.snapshot import arch_toggles_from_model, recorded_model_version
    from agents.training.eval_sharding import SENTINEL
    from agents.training.rust_eval.launch import load_sentinels

    model, zp = write_arm_snapshot(tmp_path / f"run_{arm}", arm, seed=1)
    saved = ModelVersion.from_json_file(str(tmp_path / f"run_{arm}" / "model_config.json"))
    pool = types.SimpleNamespace(items=[types.SimpleNamespace(kind=SENTINEL, key="s0", path=zp)])
    loaded = load_sentinels(pool, model)                       # the crash site: must load, not FATAL
    assert set(loaded) == {"s0"}
    # the toggles a worker / h2h reads are the RECORDED ones, gated field for gated field
    toggles = arch_toggles_from_model(model)
    rv = recorded_model_version(model)
    rv.check_opponent_snapshot_compatible(saved)
    gated = [k for k in toggles if hasattr(saved, k)]
    assert len(gated) > 30, gated
    for k in gated:
        assert toggles[k] == getattr(saved, k) == getattr(rv, k), k
    fe = model.policy.features_extractor
    assert fe.move_resolution_cell is not None                # production and its alias build the family (derived)
    for k in DELETED_BY_U3:
        assert k not in toggles and not hasattr(saved, k), k


def test_a_model_without_recorded_policy_kwargs_is_refused():
    """No silent fall-back to a built-module read: a model that records no extractor kwargs is a typed refusal."""
    from agents.model.model_version import ModelVersionError
    from agents.model.snapshot import arch_toggles_from_model, recorded_model_version

    bare = types.SimpleNamespace(policy=types.SimpleNamespace(features_extractor=types.SimpleNamespace()))
    for fn in (arch_toggles_from_model, recorded_model_version):
        with pytest.raises(ModelVersionError, match="RECORDED architecture"):
            fn(bare)
