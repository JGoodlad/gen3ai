"""Distributional value head (v29) — module build, byte-identical-off, gradient gating, version gate.

Mirrors win_prob_head_test.py (the side-readout precedent). The value-dist head is a SIDE readout off
value_pooled (never in pi/vf), so all three modes share the policy/value projection dims; the structural
toggles are the mode (none↔head + read_only↔shaping) AND the atom count `bins` (the head's output width),
both version-checked; the support (vmin/vmax) is resume-only-checked.
"""

import numpy as np
import pytest
import torch
from gymnasium import spaces

from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.model.model_version import ModelVersion, ModelVersionError, _migrate_config

BINS = 32
VMIN, VMAX = -5.0, 5.0


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


def _build(ek, space, mode, bins=BINS):
    # 'none' forces bins 0 (the constructor rejects a nonzero count with no head); on-modes use BINS.
    b = bins if mode != "none" else 0
    return Gen3FeaturesExtractor(space, **{
        **ek, "value_dist_mode": mode, "value_dist_bins": b,
        "value_dist_vmin": VMIN, "value_dist_vmax": VMAX,
    })


def test_invalid_mode_raises(ek_and_space):
    ek, space, _ = ek_and_space
    with pytest.raises(ValueError):
        _build(ek, space, "bogus")


def test_bins_required_when_on(ek_and_space):
    """mode on ⇒ bins must be > 0; mode none ⇒ bins must be 0 (no half-states)."""
    ek, space, _ = ek_and_space
    with pytest.raises(ValueError):
        _build(ek, space, "read_only", bins=0)
    with pytest.raises(ValueError):
        Gen3FeaturesExtractor(space, **{**ek, "value_dist_mode": "none", "value_dist_bins": BINS})


def test_off_is_byte_identical_dims(ek_and_space):
    """'none' builds no head; read_only/shaping add the head but it is a SIDE readout, so the
    policy/value projection dims are identical across all three modes."""
    ek, space, total = ek_and_space
    obs = {"observation": torch.zeros(3, total)}
    f_none, f_ro, f_sh = (_build(ek, space, m) for m in ("none", "read_only", "shaping"))
    pn = [tuple(t.shape) for t in f_none(obs)]
    assert pn == [tuple(t.shape) for t in f_ro(obs)] == [tuple(t.shape) for t in f_sh(obs)]
    assert f_none.value_dist_head is None and f_none.last_value_dist_logits is None
    assert f_ro.value_dist_head is not None and f_sh.value_dist_head is not None
    # read_only and shaping have IDENTICAL params (mode only changes the grad path), both > none.
    n_none = sum(p.numel() for p in f_none.parameters())
    assert sum(p.numel() for p in f_ro.parameters()) == sum(p.numel() for p in f_sh.parameters()) > n_none


def test_stash_shape(ek_and_space):
    ek, space, total = ek_and_space
    f = _build(ek, space, "read_only")
    f({"observation": torch.zeros(4, total)})
    assert f.last_value_dist_logits.shape == (4, BINS)


def test_atoms_buffer_non_persistent(ek_and_space):
    """The atom support is deterministic from bins+range → out of the state_dict (only the head's
    params, whose final Linear is `bins`-wide, define the loadable shape)."""
    ek, space, _ = ek_and_space
    f = _build(ek, space, "read_only")
    assert f.value_dist_head.atoms.shape == (BINS,)
    assert not any("atoms" in k for k in f.state_dict())


def test_mean_recovers_atom_expectation(ek_and_space):
    """mean(logits) = Σ atomsᵢ·softmaxᵢ — uniform logits ⇒ the support midpoint (≈0 for [-5,5])."""
    ek, space, _ = ek_and_space
    head = _build(ek, space, "read_only").value_dist_head
    m = head.mean(torch.zeros(2, BINS))
    assert m.shape == (2, 1)
    assert torch.allclose(m, torch.full((2, 1), (VMIN + VMAX) / 2), atol=1e-5)


def _trunk_grad_present(fe, total):
    fe.zero_grad()
    fe.forward_internal({"observation": torch.zeros(1, total)})
    fe.last_value_dist_logits.sum().backward()
    return any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in fe.team_transformer.parameters())


def test_read_only_blocks_trunk_gradient(ek_and_space):
    ek, space, total = ek_and_space
    assert _trunk_grad_present(_build(ek, space, "read_only"), total) is False


def test_shaping_flows_trunk_gradient(ek_and_space):
    ek, space, total = ek_and_space
    assert _trunk_grad_present(_build(ek, space, "shaping"), total) is True


# ── v29 version gate ──────────────────────────────────────────────────────────

def _ver(mode="none", bins=0, vmin=0.0, vmax=0.0):
    mappings = load_mappings()
    ek = Gen3ObservationEncoder(mappings).get_features_extractor_kwargs()
    ek["value_dist_mode"] = mode
    ek["value_dist_bins"] = bins
    ek["value_dist_vmin"] = vmin
    ek["value_dist_vmax"] = vmax
    pk = {"features_extractor_class": Gen3FeaturesExtractor, "features_extractor_kwargs": ek,
          "net_arch": [512, 512]}
    return ModelVersion.from_layout_and_policy_kwargs(ek["layout"], pk)


def test_version_records_fields():
    v = _ver("shaping", BINS, VMIN, VMAX)
    assert v.value_dist_mode == "shaping" and v.value_dist_bins == BINS
    assert v.value_dist_vmin == VMIN and v.value_dist_vmax == VMAX
    assert _ver().value_dist_mode == "none" and _ver().value_dist_bins == 0


@pytest.mark.parametrize("saved,current", [
    (("none", 0), ("read_only", BINS)),   # adding the head (state_dict change)
    (("read_only", BINS), ("none", 0)),   # removing it
    (("read_only", BINS), ("shaping", BINS)),  # IMMUTABLE mode flip (grad-flow change)
    (("read_only", 32), ("read_only", 51)),    # atom count = weight-shape change
])
def test_structural_mismatch_fatals(saved, current):
    s = _ver(saved[0], saved[1], VMIN, VMAX)
    c = _ver(current[0], current[1], VMIN, VMAX)
    with pytest.raises(ModelVersionError, match="value_dist"):
        c.check_compatible(s)


@pytest.mark.parametrize("mode,bins", [("none", 0), ("read_only", BINS), ("shaping", BINS)])
def test_matching_loads(mode, bins):
    _ver(mode, bins, VMIN, VMAX).check_compatible(_ver(mode, bins, VMIN, VMAX))  # no raise


def test_support_is_resume_only_not_in_check_compatible():
    """vmin/vmax are value-meaning — a drift FATALs on the resume-only check_value_dist, but
    check_compatible (which gates frozen opponents) ignores them."""
    a = _ver("read_only", BINS, VMIN, VMAX)
    b = _ver("read_only", BINS, VMIN, VMAX + 3.0)
    a.check_compatible(b)               # support drift does NOT block a load
    a.check_value_dist(VMIN, VMAX)      # matching support: no raise
    with pytest.raises(ModelVersionError, match="value_dist support"):
        a.check_value_dist(VMIN, VMAX + 3.0)


def test_migration_v28_to_v29():
    from agents.model.model_version import MODEL_CONFIG_VERSION
    d = _migrate_config({"config_version": 28})
    assert d["value_dist_mode"] == "none" and d["value_dist_bins"] == 0
    assert d["value_dist_vmin"] == 0.0 and d["value_dist_vmax"] == 0.0
    assert d["config_version"] == MODEL_CONFIG_VERSION
