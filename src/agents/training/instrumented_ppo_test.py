"""Tests for InstrumentedMaskablePPO drift detection and instrumentation."""

import hashlib
import inspect

import pytest
from sb3_contrib import MaskablePPO

from agents.training import instrumented_ppo
from agents.training.instrumented_ppo import (
    InstrumentedMaskablePPO,
    _EXPECTED_UPSTREAM_TRAIN_HASH,
    _VALUE_TAIL_FRAC,
    _verify_upstream_unchanged,
)
import torch as th
import torch.nn.functional as F


class _TailStub:
    """Minimal stand-in to exercise the pure _value_loss_from_se without building a full PPO."""
    def __init__(self, w):
        self.value_tail_weight = w


def test_value_loss_w0_is_byte_identical_to_mse():
    """β=0 → plain MSE, byte-identical to upstream F.mse_loss (the default-off no-op)."""
    th.manual_seed(0)
    target, pred = th.randn(256), th.randn(256)
    se = (target - pred) ** 2
    out = InstrumentedMaskablePPO._value_loss_from_se(_TailStub(0.0), se)
    assert th.allclose(out, F.mse_loss(target, pred))


def test_value_loss_w_positive_blends_in_cvar():
    """β>0 → (1-β)·MSE + β·CVaR(worst _VALUE_TAIL_FRAC). Strictly ≥ MSE (CVaR ≥ mean), and exact."""
    th.manual_seed(1)
    se = th.rand(500)
    w = 0.5
    out = InstrumentedMaskablePPO._value_loss_from_se(_TailStub(w), se)
    k = max(1, int(_VALUE_TAIL_FRAC * se.numel()))
    expected = (1 - w) * se.mean() + w * th.topk(se, k).values.mean()
    assert th.allclose(out, expected)
    assert out.item() >= se.mean().item()   # the tail is the worst errors → blend lifts the loss


def test_value_loss_default_attr_is_zero():
    """An unconfigured InstrumentedMaskablePPO defaults to β=0 (the class attribute) → MSE."""
    assert InstrumentedMaskablePPO.value_tail_weight == 0.0


def test_subclass_inherits_from_maskable_ppo():
    assert issubclass(InstrumentedMaskablePPO, MaskablePPO)


def test_train_method_is_overridden():
    # The subclass must define its own train, not inherit from MaskablePPO.
    assert InstrumentedMaskablePPO.train is not MaskablePPO.train


def test_recorded_upstream_hash_matches_current_upstream_source():
    """The pinned _EXPECTED_UPSTREAM_TRAIN_HASH must match the live upstream
    source. If this fails, sb3_contrib was updated and the vendored override
    in instrumented_ppo.py needs to be re-synced."""
    src = inspect.getsource(MaskablePPO.train)
    actual = hashlib.sha256(src.encode("utf-8")).hexdigest()
    assert actual == _EXPECTED_UPSTREAM_TRAIN_HASH, (
        "Upstream sb3_contrib.MaskablePPO.train() has changed; re-port the "
        "override in src/agents/training/instrumented_ppo.py and update "
        "_EXPECTED_UPSTREAM_TRAIN_HASH."
    )


def test_verify_upstream_unchanged_raises_on_mismatch(monkeypatch):
    """If the pinned hash doesn't match upstream, _verify_... must raise
    with a message that names the file and both hashes."""
    monkeypatch.setattr(
        instrumented_ppo,
        "_EXPECTED_UPSTREAM_TRAIN_HASH",
        "0" * 64,  # deliberately wrong
    )
    with pytest.raises(RuntimeError) as exc:
        _verify_upstream_unchanged()
    msg = str(exc.value)
    assert "DRIFT DETECTED" in msg
    assert "0" * 64 in msg  # the expected (wrong) hash is shown
    assert "instrumented_ppo.py" in msg  # the file to fix is named
    assert "ACTION REQUIRED" in msg


def test_instrumentation_marker_present_in_override():
    """Sanity check: the +INSTRUMENTATION marker survives in the override.
    If someone removes the instrumentation by accident, this fails."""
    src = inspect.getsource(InstrumentedMaskablePPO.train)
    assert "vf_clip_fractions" in src
    assert "train/clip_fraction_vf" in src
    assert "+INSTRUMENTATION" in src
