"""Unit tests for the PRE-transformer move-belief reinjection (`move_belief_prefuse`, v32).

Pins: the forward-behavior nature (no new modules → state_dict + projection widths identical to the
post-fuse model), the dependency guard (requires move_belief_mode != off), gradient still reaching the
move-belief head, the logits stash surviving the timing move (so the damage op + aux loss still read
it), the damage-op interop, and the KEY property that reinjecting BEFORE vs AFTER the transformer is a
REAL computational change (the believed moves co-refine through attention) — not a no-op reorder.
"""
import numpy as np
import gymnasium as gym
import torch
import pytest

from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_mappings = load_mappings()
_layout = Gen3ObservationEncoder(_mappings).get_layout()


def _make_model(**kw):
    obs_space = gym.spaces.Box(0.0, 1.0, shape=(_layout["total_dim"],), dtype=np.float32)
    return Gen3FeaturesExtractor(obs_space, layout=_layout, mappings=_mappings, **kw)


# --------------------------------------------------------------------------- structure / guards
def test_prefuse_off_state_dict_and_dims_identical():
    """prefuse adds NO modules (same MoveBelief, different call site) → on/off state_dict keys identical
    and projection widths unchanged (a pure forward-behavior toggle)."""
    on = _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed", move_belief_prefuse=True)
    off = _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed", move_belief_prefuse=False)
    assert on.move_belief_prefuse is True and off.move_belief_prefuse is False
    assert set(on.state_dict()) == set(off.state_dict())
    assert on.projection_input_dim == off.projection_input_dim
    assert on.value_projection_input_dim == off.value_projection_input_dim


def test_prefuse_requires_move_belief_on():
    with pytest.raises(ValueError, match="move_belief_prefuse"):
        _make_model(attend_unrevealed_opponents=True, move_belief_mode="off", move_belief_prefuse=True)
    # valid with a head present.
    _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed", move_belief_prefuse=True)


# --------------------------------------------------------------------------- forward behavior
def test_prefuse_stashes_move_belief_logits():
    """The pre-transformer path still stashes last_move_belief_logits (consumed by the damage op +
    the BCE aux loss) — the timing move must not drop the stash."""
    model = _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed",
                        move_belief_prefuse=True)
    model.eval()
    with torch.no_grad():
        model.forward({"observation": torch.rand(2, _layout["total_dim"])})
    assert model.last_move_belief_logits is not None
    assert model.last_move_belief_logits.shape[0] == 2


def test_prefuse_grad_reaches_move_head():
    """ON: the forward is finite and gradients reach the move-belief head — it is in the live path."""
    model = _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed",
                        move_belief_prefuse=True)
    model.train()
    pi, vf = model.forward({"observation": torch.rand(6, _layout["total_dim"])})
    assert torch.isfinite(pi).all() and torch.isfinite(vf).all()
    (pi.sum() + vf.sum()).backward()
    g = model.move_belief.move_head.weight.grad
    assert g is not None and g.abs().sum() > 0


def test_prefuse_changes_forward_vs_postfuse():
    """Reinjecting the move belief BEFORE vs AFTER the transformer is a REAL computational change: with
    identical weights the two timings produce materially different pi/vf (the believed moves co-refine
    through the 2 attention layers in the prefuse path). This is the whole point of the toggle —
    asserting it is NOT a near-noop reorder."""
    model = _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed",
                        move_belief_prefuse=True)
    model.eval()
    obs = {"observation": torch.rand(4, _layout["total_dim"])}
    with torch.no_grad():
        pi_pre, vf_pre = model.forward(obs)
        model.move_belief_prefuse = False        # POST-transformer reinjection, SAME weights
        pi_post, vf_post = model.forward(obs)
        model.move_belief_prefuse = True
    assert (pi_pre - pi_post).abs().max() > 1e-4, "prefuse should change the policy output"
    assert (vf_pre - vf_post).abs().max() > 1e-4, "prefuse should change the value output"


def test_prefuse_with_damage_op_forward_finite():
    """prefuse + damage_op: the op reads last_move_belief_logits, which the PRE-transformer block sets
    before the op runs — the full stack is finite."""
    model = _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed",
                        damage_op=True, move_belief_prefuse=True)
    model.eval()
    with torch.no_grad():
        pi, vf = model.forward({"observation": torch.rand(3, _layout["total_dim"])})
    assert torch.isfinite(pi).all() and torch.isfinite(vf).all()
    assert model.last_move_belief_logits is not None


def test_prefuse_modes_forward_finite():
    """Every move_belief_mode reinjects correctly in the prefuse path (the mask differs per mode)."""
    for mode in ("revealed", "unrevealed", "both"):
        model = _make_model(attend_unrevealed_opponents=True, move_belief_mode=mode,
                            move_belief_prefuse=True)
        model.eval()
        with torch.no_grad():
            pi, vf = model.forward({"observation": torch.rand(2, _layout["total_dim"])})
        assert torch.isfinite(pi).all() and torch.isfinite(vf).all(), mode
