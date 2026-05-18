"""
Unit tests for the active-context pre-attention injection in Gen3FeaturesExtractor.

Tests verify that active_ctx_to_role correctly injects boost/volatile state into
the active Pokémon's role token before the 5 attention paths run.
"""
import torch
import numpy as np
import gymnasium as gym
import pytest

from agents.model.features_extractor import Gen3FeaturesExtractor, ROLE_TOKEN_SIZE
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings


def _make_model():
    """Build a freshly initialised extractor with the live layout."""
    mappings = load_mappings()
    encoder = Gen3ObservationEncoder(mappings)
    layout = encoder.get_layout()
    total_dim = layout["total_dim"]
    obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(total_dim,), dtype=np.float32)
    model = Gen3FeaturesExtractor(obs_space, layout=layout, mappings=mappings)
    model.eval()
    return model, layout


def _zero_obs(layout):
    """Return a zero observation tensor [1, total_dim]."""
    return torch.zeros(1, layout["total_dim"])


# ---------------------------------------------------------------------------
# Shape sanity check
# ---------------------------------------------------------------------------

def test_active_ctx_to_role_output_shape():
    model, layout = _make_model()
    active_ctx_dim = layout.get("active_context_dim", 22)
    x = torch.zeros(3, active_ctx_dim)
    out = model.active_ctx_to_role(x)
    assert out.shape == (3, ROLE_TOKEN_SIZE), (
        f"active_ctx_to_role should output [B, {ROLE_TOKEN_SIZE}], got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Functional tests: non-zero active context must change the output
# ---------------------------------------------------------------------------

def test_active_ctx_injection_our_side():
    """Non-zero our-side active context (e.g. +6 Atk boost) must change forward_internal output."""
    model, layout = _make_model()
    ctx_start = layout["parts"]["context"]["start"]

    obs_zero = _zero_obs(layout)
    obs_ctx = obs_zero.clone()
    obs_ctx[0, ctx_start] = 1.0  # first dim of our active context: +6 Atk boost (positive half)

    with torch.no_grad():
        out_zero = model.forward_internal({"observation": obs_zero})
        out_ctx = model.forward_internal({"observation": obs_ctx})

    assert not torch.allclose(out_zero, out_ctx), (
        "forward_internal output should differ when our active context is non-zero"
    )


def test_active_ctx_injection_opp_side():
    """Non-zero opp-side active context must change forward_internal output."""
    model, layout = _make_model()
    ctx_start = layout["parts"]["context"]["start"]
    active_ctx_dim = layout.get("active_context_dim", 22)

    obs_zero = _zero_obs(layout)
    obs_ctx = obs_zero.clone()
    obs_ctx[0, ctx_start + active_ctx_dim] = 1.0  # first dim of opp active context

    with torch.no_grad():
        out_zero = model.forward_internal({"observation": obs_zero})
        out_ctx = model.forward_internal({"observation": obs_ctx})

    assert not torch.allclose(out_zero, out_ctx), (
        "forward_internal output should differ when opp active context is non-zero"
    )


# ---------------------------------------------------------------------------
# Isolation test: active_ctx_to_role is the pre-attention injection path
# ---------------------------------------------------------------------------

def test_active_ctx_injection_only_via_active_ctx_to_role():
    """
    Zeroing active_ctx_to_role (weights + bias) changes the output even for the same
    obs, proving the layer is wired into the forward pass and contributing pre-attention
    signal through the role tokens.
    """
    model, layout = _make_model()
    ctx_start = layout["parts"]["context"]["start"]

    obs_ctx = _zero_obs(layout)
    obs_ctx[0, ctx_start] = 1.0  # our active boost (non-zero context)

    with torch.no_grad():
        out_ctx_full = model.forward_internal({"observation": obs_ctx})

        # Zero the injection MLP (weight + bias → outputs zero regardless of input)
        for p in model.active_ctx_to_role.parameters():
            p.zero_()

        out_ctx_no_inject = model.forward_internal({"observation": obs_ctx})

    # Same obs but different injection MLP state → output must differ.
    # This proves active_ctx_to_role is connected and changing the role tokens
    # at the active slot before the attention paths run.
    assert not torch.allclose(out_ctx_full, out_ctx_no_inject), (
        "Zeroing active_ctx_to_role should change forward_internal output for the same "
        "non-zero active context obs, proving the injection path is wired in and active."
    )


if __name__ == "__main__":
    test_active_ctx_to_role_output_shape()
    test_active_ctx_injection_our_side()
    test_active_ctx_injection_opp_side()
    test_active_ctx_injection_only_via_active_ctx_to_role()
    print("✅ All active-context injection tests passed!")
