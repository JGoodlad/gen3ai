"""Tests for InstrumentedMaskablePPO drift detection and instrumentation."""

import hashlib
import inspect

import pytest
from sb3_contrib import MaskablePPO

from agents.training import instrumented_ppo
from agents.training.instrumented_ppo import (
    InstrumentedMaskablePPO,
    _EXPECTED_UPSTREAM_TRAIN_HASH,
    _verify_upstream_unchanged,
)


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
