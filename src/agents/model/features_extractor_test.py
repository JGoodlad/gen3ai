"""
Unit tests for pre-attention injection in Gen3FeaturesExtractor:
  - active_ctx_to_role: boost/volatile state into active role tokens
  - td_conditioner: TurnDelta effectiveness + move-order into active role tokens
  - turn_history_attn: N-turn MHA attention over TurnDelta history
"""
import torch
import numpy as np
import gymnasium as gym
import pytest

from agents.model.features_extractor import (
    Gen3FeaturesExtractor,
    ROLE_TOKEN_SIZE,
    N_HISTORY_TURNS,
    TD_STRATEGIC_DIM,
    TD_STRATEGIC_OFFSET,
)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.observation.turn_delta_encoder import TURN_DELTA_DIM


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


# ===========================================================================
# TurnDelta Conditioner tests
# ===========================================================================

def _td_strategic_start(layout: dict) -> int:
    """Absolute index in the observation vector of the first strategic TurnDelta dim."""
    return layout["base_dim"] + 11 + TD_STRATEGIC_OFFSET


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_td_strategic_constants_consistent():
    """TD_STRATEGIC_DIM and TD_STRATEGIC_OFFSET must be consistent with TurnDelta layout."""
    from agents.observation.turn_delta_encoder import TURN_DELTA_DIM, EFF_DIM, ORDER_DIM
    assert TD_STRATEGIC_DIM == EFF_DIM * 2 + ORDER_DIM, (
        "TD_STRATEGIC_DIM should be our_eff(4) + opp_eff(4) + order(2) = 10"
    )
    assert TD_STRATEGIC_OFFSET + TD_STRATEGIC_DIM == TURN_DELTA_DIM, (
        "Strategic slice must sit flush at the tail of the TurnDelta block"
    )


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def test_td_conditioner_output_shape():
    """td_conditioner should map [B, TD_STRATEGIC_DIM] → [B, ROLE_TOKEN_SIZE]."""
    model, layout = _make_model()
    x = torch.zeros(4, TD_STRATEGIC_DIM)
    out = model.td_conditioner(x)
    assert out.shape == (4, ROLE_TOKEN_SIZE), (
        f"td_conditioner should output [B, {ROLE_TOKEN_SIZE}], got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Functional: non-zero strategic dims change the output
# ---------------------------------------------------------------------------

def test_td_conditioning_our_effectiveness_changes_output():
    """Super-effective hit on our side must change forward_internal output."""
    model, layout = _make_model()
    start = _td_strategic_start(layout)

    obs_zero = _zero_obs(layout)
    obs_se = obs_zero.clone()
    obs_se[0, start + 3] = 1.0  # our_eff[3] = super-effective

    with torch.no_grad():
        out_zero = model.forward_internal({"observation": obs_zero})
        out_se   = model.forward_internal({"observation": obs_se})

    assert not torch.allclose(out_zero, out_se), (
        "forward_internal output should differ when our_effectiveness = super-effective"
    )


def test_td_conditioning_opp_effectiveness_changes_output():
    """Immune hit on opp side must change forward_internal output."""
    model, layout = _make_model()
    start = _td_strategic_start(layout)

    obs_zero   = _zero_obs(layout)
    obs_immune = obs_zero.clone()
    obs_immune[0, start + 4] = 1.0  # opp_eff[0] = immune

    with torch.no_grad():
        out_zero   = model.forward_internal({"observation": obs_zero})
        out_immune = model.forward_internal({"observation": obs_immune})

    assert not torch.allclose(out_zero, out_immune), (
        "forward_internal output should differ when opp_effectiveness = immune"
    )


def test_td_conditioning_move_order_changes_output():
    """we_moved_first bit must change forward_internal output."""
    model, layout = _make_model()
    start = _td_strategic_start(layout)

    obs_zero  = _zero_obs(layout)
    obs_order = obs_zero.clone()
    obs_order[0, start + 8] = 1.0  # move_order[0] = we_first

    with torch.no_grad():
        out_zero  = model.forward_internal({"observation": obs_zero})
        out_order = model.forward_internal({"observation": obs_order})

    assert not torch.allclose(out_zero, out_order), (
        "forward_internal output should differ when we_moved_first = True"
    )


# ---------------------------------------------------------------------------
# Isolation: td_conditioner is wired into the forward pass pre-attention
# ---------------------------------------------------------------------------

def test_td_conditioner_wired_into_forward_pass():
    """
    Zeroing td_conditioner weights changes forward_internal output when strategic
    dims are non-zero, proving the MLP is connected and contributes pre-attention
    signal through the active role tokens.
    """
    model, layout = _make_model()
    start = _td_strategic_start(layout)

    obs = _zero_obs(layout)
    obs[0, start + 3] = 1.0  # our_eff = super-effective

    with torch.no_grad():
        out_full = model.forward_internal({"observation": obs})

        for p in model.td_conditioner.parameters():
            p.zero_()

        out_zeroed = model.forward_internal({"observation": obs})

    assert not torch.allclose(out_full, out_zeroed), (
        "Zeroing td_conditioner should change forward_internal output for a non-zero "
        "strategic obs, proving the module is wired into the forward pass."
    )


# ---------------------------------------------------------------------------
# Asymmetry: our_eff ≠ opp_eff (the two sides are distinguished)
# ---------------------------------------------------------------------------

def test_td_conditioning_distinguishes_our_vs_opp_effectiveness():
    """
    'We hit SE, they hit resisted' and 'they hit SE, we hit resisted' must produce
    different outputs — the model treats the two effectiveness sides asymmetrically.
    This verifies that our_eff conditions our_active and opp_eff conditions opp_active
    (via the perspective-flipped shared MLP), not the same token.
    """
    model, layout = _make_model()
    start = _td_strategic_start(layout)

    # Scenario A: we hit SE (our_eff[3]=1), they hit resisted (opp_eff[1]=1)
    obs_a = _zero_obs(layout)
    obs_a[0, start + 3] = 1.0  # our_eff = SE
    obs_a[0, start + 5] = 1.0  # opp_eff = resisted

    # Scenario B: they hit SE (opp_eff[3]=1), we hit resisted (our_eff[1]=1)
    obs_b = _zero_obs(layout)
    obs_b[0, start + 1] = 1.0  # our_eff = resisted
    obs_b[0, start + 7] = 1.0  # opp_eff = SE

    with torch.no_grad():
        out_a = model.forward_internal({"observation": obs_a})
        out_b = model.forward_internal({"observation": obs_b})

    assert not torch.allclose(out_a, out_b), (
        "Swapping our_eff↔opp_eff should produce different outputs — "
        "the model must distinguish 'we are winning the matchup' from 'they are winning'."
    )


# ===========================================================================
# Turn-History Attention tests
# ===========================================================================

def _history_start(layout: dict) -> int:
    """Absolute index of the first dim of the turn-history block in obs."""
    return layout["base_dim"] + 11


def test_turn_history_modules_exist():
    """turn_history_pos_emb, turn_history_attn, turn_history_norm must be present."""
    model, _ = _make_model()
    assert hasattr(model, "turn_history_pos_emb")
    assert hasattr(model, "turn_history_attn")
    assert hasattr(model, "turn_history_norm")


def test_turn_history_pos_emb_shape():
    """Positional embedding must have N_HISTORY_TURNS entries, each of size _td_embed_dim."""
    model, _ = _make_model()
    assert model.turn_history_pos_emb.num_embeddings == N_HISTORY_TURNS
    assert model.turn_history_pos_emb.embedding_dim == model._td_embed_dim


def test_turn_history_attn_embed_dim():
    """MHA embed_dim must equal _td_embed_dim."""
    model, _ = _make_model()
    assert model.turn_history_attn.embed_dim == model._td_embed_dim


def test_turn_history_td_embed_dim_value():
    """_td_embed_dim = 2*move_emb + 2*type_emb + (TURN_DELTA_DIM - 4 raw IDs)."""
    model, layout = _make_model()
    expected = (
        2 * layout["move_embedding_dim"]
        + 2 * layout["type_embedding_dim"]
        + TURN_DELTA_DIM - 4
    )
    assert model._td_embed_dim == expected, (
        f"_td_embed_dim={model._td_embed_dim} != expected {expected}"
    )


def test_turn_history_block_changes_output():
    """Non-zero values in the turn-history block must change forward_internal output."""
    model, layout = _make_model()
    start = _history_start(layout)

    obs_zero = _zero_obs(layout)
    obs_hist = obs_zero.clone()
    obs_hist[0, start] = 1.0  # first dim of oldest TurnDelta slot

    with torch.no_grad():
        out_zero = model.forward_internal({"observation": obs_zero})
        out_hist = model.forward_internal({"observation": obs_hist})

    assert not torch.allclose(out_zero, out_hist), (
        "forward_internal output should differ when turn-history block is non-zero"
    )


def test_turn_history_most_recent_slot_changes_output():
    """The most-recent (last) history slot must influence the output."""
    model, layout = _make_model()
    start = _history_start(layout)
    # Last TurnDelta slot: offset = (N_HISTORY_TURNS - 1) * TURN_DELTA_DIM within the block
    last_slot_start = start + (N_HISTORY_TURNS - 1) * TURN_DELTA_DIM

    obs_zero = _zero_obs(layout)
    obs_hist = obs_zero.clone()
    obs_hist[0, last_slot_start] = 1.0

    with torch.no_grad():
        out_zero = model.forward_internal({"observation": obs_zero})
        out_hist = model.forward_internal({"observation": obs_hist})

    assert not torch.allclose(out_zero, out_hist), (
        "Changing the most-recent (last) history slot must affect forward_internal output"
    )


def test_turn_history_oldest_slot_changes_output():
    """The oldest (first) history slot must also influence the output via attention."""
    model, layout = _make_model()
    start = _history_start(layout)

    obs_zero = _zero_obs(layout)
    obs_hist = obs_zero.clone()
    obs_hist[0, start] = 1.0  # first dim of oldest slot

    with torch.no_grad():
        out_zero = model.forward_internal({"observation": obs_zero})
        out_hist = model.forward_internal({"observation": obs_hist})

    assert not torch.allclose(out_zero, out_hist), (
        "The oldest history slot must influence forward_internal output via cross-turn attention"
    )


def test_turn_history_positional_embedding_wired_in():
    """Zeroing turn_history_pos_emb changes output when history is non-zero.

    This proves pos_emb is wired into the forward pass, not a dead module.
    """
    model, layout = _make_model()
    start = _history_start(layout)

    obs = _zero_obs(layout)
    obs[0, start] = 1.0

    with torch.no_grad():
        out_before = model.forward_internal({"observation": obs})
        for p in model.turn_history_pos_emb.parameters():
            p.zero_()
        out_after = model.forward_internal({"observation": obs})

    assert not torch.allclose(out_before, out_after), (
        "Zeroing turn_history_pos_emb must change output for a non-zero history obs"
    )


def test_turn_history_two_distinct_histories_differ():
    """Two different non-zero history blocks must produce different outputs."""
    model, layout = _make_model()
    start = _history_start(layout)

    obs_a = _zero_obs(layout)
    obs_b = _zero_obs(layout)
    obs_a[0, start] = 1.0                      # non-zero in oldest slot
    obs_b[0, start + TURN_DELTA_DIM] = 1.0     # non-zero in second-oldest slot

    with torch.no_grad():
        out_a = model.forward_internal({"observation": obs_a})
        out_b = model.forward_internal({"observation": obs_b})

    assert not torch.allclose(out_a, out_b), (
        "Different history patterns should produce different outputs"
    )
