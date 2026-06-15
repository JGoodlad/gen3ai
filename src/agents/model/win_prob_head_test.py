"""Win-probability head — module build, byte-identical-off, gradient gating, and v22 version gate."""

import numpy as np
import pytest
import torch
from gymnasium import spaces

from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.model.model_version import ModelVersion, ModelVersionError, _migrate_config


@pytest.fixture(scope="module")
def ek_and_space():
    mappings = load_mappings()
    ek = Gen3ObservationEncoder(mappings).get_features_extractor_kwargs()
    total = ek["layout"]["total_dim"]
    space = spaces.Dict({
        "observation": spaces.Box(-np.inf, np.inf, (total,), np.float32),
        "action_mask": spaces.Box(0, 1, (11,), np.int8),
    })
    return ek, space, total


def _build(ek, space, mode):
    return Gen3FeaturesExtractor(space, **{**ek, "win_prob_mode": mode})


def test_invalid_mode_raises(ek_and_space):
    ek, space, _ = ek_and_space
    with pytest.raises(ValueError):
        _build(ek, space, "bogus")


def test_off_is_byte_identical_dims(ek_and_space):
    """'none' builds no head; read_only/shaping add the head but it is a SIDE readout, so the
    policy/value projection dims are identical across all three modes."""
    ek, space, total = ek_and_space
    obs = {"observation": torch.zeros(3, total)}
    f_none, f_ro, f_sh = (_build(ek, space, m) for m in ("none", "read_only", "shaping"))
    pn = [tuple(t.shape) for t in f_none(obs)]
    assert pn == [tuple(t.shape) for t in f_ro(obs)] == [tuple(t.shape) for t in f_sh(obs)]
    assert f_none.win_head is None and f_none.last_win_prob_logits is None
    assert f_ro.win_head is not None and f_sh.win_head is not None
    # read_only and shaping have IDENTICAL params (mode only changes the grad path), both > none.
    n_none = sum(p.numel() for p in f_none.parameters())
    assert sum(p.numel() for p in f_ro.parameters()) == sum(p.numel() for p in f_sh.parameters()) > n_none


def test_stash_shape(ek_and_space):
    ek, space, total = ek_and_space
    f = _build(ek, space, "read_only")
    f({"observation": torch.zeros(4, total)})
    assert f.last_win_prob_logits.shape == (4, 1)


def _trunk_grad_present(fe, total):
    fe.zero_grad()
    fe.forward_internal({"observation": torch.zeros(1, total)})
    fe.last_win_prob_logits.sum().backward()
    return any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in fe.team_transformer.parameters())


def test_read_only_blocks_trunk_gradient(ek_and_space):
    ek, space, total = ek_and_space
    assert _trunk_grad_present(_build(ek, space, "read_only"), total) is False


def test_shaping_flows_trunk_gradient(ek_and_space):
    ek, space, total = ek_and_space
    assert _trunk_grad_present(_build(ek, space, "shaping"), total) is True


# ── v22 version gate ──────────────────────────────────────────────────────────

def _ver(mode="none"):
    mappings = load_mappings()
    ek = Gen3ObservationEncoder(mappings).get_features_extractor_kwargs()
    ek["win_prob_mode"] = mode
    pk = {"features_extractor_class": Gen3FeaturesExtractor, "features_extractor_kwargs": ek,
          "net_arch": [512, 512]}
    return ModelVersion.from_layout_and_policy_kwargs(ek["layout"], pk)


def test_version_records_mode():
    assert _ver("shaping").win_prob_mode == "shaping"
    assert _ver().win_prob_mode == "none" and _ver().win_prob_coef == 1.0


@pytest.mark.parametrize("saved,current", [
    ("none", "read_only"),   # adding the head (state_dict change)
    ("read_only", "none"),   # removing it
    ("none", "shaping"),     # adding the head (the other direction)
    ("shaping", "none"),
    ("read_only", "shaping"),  # IMMUTABLE mode flip (same params, but grad-flow change)
    ("shaping", "read_only"),
])
def test_mode_mismatch_fatals(saved, current):
    with pytest.raises(ModelVersionError, match="win_prob_mode"):
        _ver(current).check_compatible(_ver(saved))


@pytest.mark.parametrize("mode", ["none", "read_only", "shaping"])
def test_matching_mode_loads(mode):
    _ver(mode).check_compatible(_ver(mode))  # no raise


def test_coef_not_version_locked():
    """win_prob_coef is training-only — a mismatch must NOT block a load (it touches no forward)."""
    a, b = _ver("shaping"), _ver("shaping")
    a.win_prob_coef, b.win_prob_coef = 0.3, 2.0
    a.check_compatible(b)  # no raise


def test_migration_v21_to_v22():
    d = _migrate_config({"config_version": 21})
    assert d["win_prob_mode"] == "none" and d["win_prob_coef"] == 1.0 and d["config_version"] == 22
