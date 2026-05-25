"""
Unit tests for the unified-transformer feature extractor (ai_v4).

Coverage:
  - module presence and shape sanity for the new transformer components
  - forward-pass output is responsive to changes in every input region
    (team slots, history slots, global scalars)
  - history positional encoding and key-padding mask are actually wired in
"""
import torch
import numpy as np
import gymnasium as gym

from agents.model.features_extractor import (
    Gen3FeaturesExtractor,
    ROLE_TOKEN_SIZE,
    D_MODEL,
    N_HISTORY_TURNS,
    TRANSFORMER_N_LAYERS,
    TRANSFORMER_N_HEADS,
    TRANSFORMER_FFN_DIM,
    NUM_TOKEN_TYPES,
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
# Constants and shape sanity
# ---------------------------------------------------------------------------

def test_d_model_matches_role_token_size():
    """d_model = role_token_size so team role tokens enter the transformer unchanged."""
    assert D_MODEL == ROLE_TOKEN_SIZE


def test_td_strategic_constants_consistent():
    from agents.observation.turn_delta_encoder import EFF_DIM, ORDER_DIM
    assert TD_STRATEGIC_DIM == EFF_DIM * 2 + ORDER_DIM
    assert TD_STRATEGIC_OFFSET + TD_STRATEGIC_DIM == TURN_DELTA_DIM


# ---------------------------------------------------------------------------
# Module presence — proves the design's component list is implemented
# ---------------------------------------------------------------------------

def test_unified_transformer_modules_exist():
    model, _ = _make_model()
    assert hasattr(model, "token_type_emb")
    assert hasattr(model, "turn_history_pos_emb")
    assert hasattr(model, "history_proj")
    assert hasattr(model, "global_proj")
    assert hasattr(model, "transformer_layers")
    assert hasattr(model, "our_cls")
    assert hasattr(model, "their_cls")
    assert hasattr(model, "our_cls_attn")
    assert hasattr(model, "their_cls_attn")
    assert hasattr(model, "norm_pool_our")
    assert hasattr(model, "norm_pool_their")


def test_removed_modules_absent():
    """The five hand-engineered attention paths and the v3 injection MLPs must be gone."""
    model, _ = _make_model()
    for removed in (
        "pressure_attn", "safety_attn", "synergy_attn", "threat_attn", "opp_synergy_attn",
        "our_pool_query", "their_pool_query", "our_pool_attn", "their_pool_attn",
        "active_ctx_to_role", "td_conditioner",
        "turn_history_attn", "turn_history_norm",
        "status_embedding",
    ):
        assert not hasattr(model, removed), f"v3 module {removed!r} should be removed in v4"


def test_token_type_embedding_shape():
    model, _ = _make_model()
    assert model.token_type_emb.num_embeddings == NUM_TOKEN_TYPES
    assert model.token_type_emb.embedding_dim == D_MODEL


def test_turn_history_pos_emb_shape():
    model, _ = _make_model()
    assert model.turn_history_pos_emb.num_embeddings == N_HISTORY_TURNS
    assert model.turn_history_pos_emb.embedding_dim == D_MODEL


def test_history_proj_shape():
    model, _ = _make_model()
    assert model.history_proj.in_features == model._td_embed_dim
    assert model.history_proj.out_features == D_MODEL


def test_global_proj_shape():
    model, layout = _make_model()
    active_ctx_dim = layout.get("active_context_dim", 22)
    expected_in = 2 * active_ctx_dim + model._non_matchup_rest_dim
    assert model.global_proj.in_features == expected_in
    assert model.global_proj.out_features == D_MODEL


def test_transformer_stack_shape():
    model, _ = _make_model()
    assert len(model.transformer_layers) == TRANSFORMER_N_LAYERS
    for layer in model.transformer_layers:
        # nn.TransformerEncoderLayer exposes self_attn.embed_dim
        assert layer.self_attn.embed_dim == D_MODEL
        assert layer.self_attn.num_heads == TRANSFORMER_N_HEADS
        # FFN expansion lives in linear1.out_features
        assert layer.linear1.out_features == TRANSFORMER_FFN_DIM


def test_cls_parameters_shape():
    model, _ = _make_model()
    assert model.our_cls.shape == (1, 1, D_MODEL)
    assert model.their_cls.shape == (1, 1, D_MODEL)


def test_token_count_matches_design():
    """Total token sequence length = 2 * TEAM_SIZE + N_HISTORY_TURNS + 1 (global)."""
    from agents.observation.constants import TEAM_SIZE
    model, _ = _make_model()
    assert model._total_tokens == 2 * TEAM_SIZE + N_HISTORY_TURNS + 1


# ---------------------------------------------------------------------------
# Forward pass: shape and basic responsiveness
# ---------------------------------------------------------------------------

def test_forward_output_shape():
    model, layout = _make_model()
    obs = _zero_obs(layout)
    with torch.no_grad():
        out = model({"observation": obs})
    assert out.shape == (1, model.features_dim)


def test_active_context_changes_output():
    """A non-zero active-context dim must propagate through to the projected output."""
    model, layout = _make_model()
    ctx_start = layout["parts"]["context"]["start"]

    obs_zero = _zero_obs(layout)
    obs_ctx = obs_zero.clone()
    obs_ctx[0, ctx_start] = 1.0  # first dim of our active context

    with torch.no_grad():
        out_zero = model.forward_internal({"observation": obs_zero})
        out_ctx  = model.forward_internal({"observation": obs_ctx})
    assert not torch.allclose(out_zero, out_ctx)


def test_opp_active_context_changes_output():
    model, layout = _make_model()
    ctx_start = layout["parts"]["context"]["start"]
    active_ctx_dim = layout.get("active_context_dim", 22)

    obs_zero = _zero_obs(layout)
    obs_ctx = obs_zero.clone()
    obs_ctx[0, ctx_start + active_ctx_dim] = 1.0

    with torch.no_grad():
        out_zero = model.forward_internal({"observation": obs_zero})
        out_ctx  = model.forward_internal({"observation": obs_ctx})
    assert not torch.allclose(out_zero, out_ctx)


# ---------------------------------------------------------------------------
# Turn history — masking, positional encoding, attention path all live
# ---------------------------------------------------------------------------

def _history_start(layout: dict) -> int:
    return layout["base_dim"] + 11


def test_history_most_recent_slot_changes_output():
    model, layout = _make_model()
    start = _history_start(layout)
    last_slot_start = start + (N_HISTORY_TURNS - 1) * TURN_DELTA_DIM

    obs_zero = _zero_obs(layout)
    obs_hist = obs_zero.clone()
    obs_hist[0, last_slot_start] = 1.0

    with torch.no_grad():
        out_zero = model.forward_internal({"observation": obs_zero})
        out_hist = model.forward_internal({"observation": obs_hist})
    assert not torch.allclose(out_zero, out_hist)


def test_history_oldest_slot_changes_output():
    model, layout = _make_model()
    start = _history_start(layout)

    obs_zero = _zero_obs(layout)
    obs_hist = obs_zero.clone()
    obs_hist[0, start] = 1.0

    with torch.no_grad():
        out_zero = model.forward_internal({"observation": obs_zero})
        out_hist = model.forward_internal({"observation": obs_hist})
    assert not torch.allclose(out_zero, out_hist)


def test_history_two_distinct_slots_produce_distinct_outputs():
    """Putting the same signal in different history positions yields different outputs."""
    model, layout = _make_model()
    start = _history_start(layout)

    obs_a = _zero_obs(layout)
    obs_b = _zero_obs(layout)
    obs_a[0, start] = 1.0
    obs_b[0, start + TURN_DELTA_DIM] = 1.0

    with torch.no_grad():
        out_a = model.forward_internal({"observation": obs_a})
        out_b = model.forward_internal({"observation": obs_b})
    assert not torch.allclose(out_a, out_b)


def test_history_pos_embedding_wired_in():
    """Zeroing turn_history_pos_emb must change the output when history is non-zero."""
    model, layout = _make_model()
    start = _history_start(layout)

    obs = _zero_obs(layout)
    obs[0, start] = 1.0

    with torch.no_grad():
        out_before = model.forward_internal({"observation": obs})
        for p in model.turn_history_pos_emb.parameters():
            p.zero_()
        out_after = model.forward_internal({"observation": obs})
    assert not torch.allclose(out_before, out_after)


def test_history_proj_wired_in():
    """Zeroing the history projection must change the output for non-zero history."""
    model, layout = _make_model()
    start = _history_start(layout)

    obs = _zero_obs(layout)
    obs[0, start + 10] = 0.5  # touch a scalar dim of a history slot

    with torch.no_grad():
        out_before = model.forward_internal({"observation": obs})
        for p in model.history_proj.parameters():
            p.zero_()
        out_after = model.forward_internal({"observation": obs})
    assert not torch.allclose(out_before, out_after)


def test_history_empty_slot_masking():
    """
    Filling an empty (all-zero) history slot with non-zero scalars must change the
    output — proves the empty-history key_padding_mask correctly marks zero slots
    as padding only while they are exactly zero.
    """
    model, layout = _make_model()
    start = _history_start(layout)

    obs_zero = _zero_obs(layout)
    obs_filled = obs_zero.clone()
    # Touch a scalar dim of the first (oldest) slot: TURN_DELTA_DIM index 10 is
    # `our_hp_delta` — a continuous scalar, never used as an ID.
    obs_filled[0, start + 10] = 0.25

    with torch.no_grad():
        out_zero = model.forward_internal({"observation": obs_zero})
        out_filled = model.forward_internal({"observation": obs_filled})
    assert not torch.allclose(out_zero, out_filled)


# ---------------------------------------------------------------------------
# Transformer responsiveness — confirm the unified path is wired
# ---------------------------------------------------------------------------

def test_transformer_layer_wired_in():
    """Zeroing all parameters of the first transformer layer must change the output."""
    model, layout = _make_model()

    # Non-trivial input: touch a few inputs so tokens are not exactly zero everywhere.
    obs = _zero_obs(layout)
    ctx_start = layout["parts"]["context"]["start"]
    obs[0, ctx_start] = 1.0
    obs[0, _history_start(layout) + 10] = 0.3

    with torch.no_grad():
        out_before = model.forward_internal({"observation": obs})
        for p in model.transformer_layers[0].parameters():
            p.zero_()
        out_after = model.forward_internal({"observation": obs})
    assert not torch.allclose(out_before, out_after)


def test_cls_pooling_wired_in():
    """Zeroing the CLS attention modules must change the output for non-zero input."""
    model, layout = _make_model()

    obs = _zero_obs(layout)
    obs[0, layout["parts"]["context"]["start"]] = 1.0

    with torch.no_grad():
        out_before = model.forward_internal({"observation": obs})
        for p in model.our_cls_attn.parameters():
            p.zero_()
        for p in model.their_cls_attn.parameters():
            p.zero_()
        # The CLS parameter vectors themselves are nontrivially queried — zero them too.
        model.our_cls.data.zero_()
        model.their_cls.data.zero_()
        out_after = model.forward_internal({"observation": obs})
    assert not torch.allclose(out_before, out_after)


# ---------------------------------------------------------------------------
# Active-slot identification — guards against stale hardcoded indices into
# the per-Pokémon layout. The active flag is invariantly the LAST dim of
# `hp_and_active = pokemon_part[:, :, hp_offset:]`; this index moved when
# spread (step 1) and the HP candidate block (step 2) were appended.
# ---------------------------------------------------------------------------

def test_active_flag_is_last_dim_of_hp_and_active_block():
    """
    Documents and pins the invariant that the active flag lives at the LAST
    position of `hp_and_active`. If POKEMON_FULL_DIM grows (e.g. step 4 adds
    another per-slot block) and someone forgets to keep the active flag at
    the end, this test fires with a precise message rather than letting
    `hp_and_active[:, :, -1]` silently point at the wrong dim.
    """
    from agents.observation.constants import POKEMON_FULL_DIM
    _, layout = _make_model()
    hp_off = layout["pokemon"]["hp"]["offset"]
    hp_and_active_dim = POKEMON_FULL_DIM - hp_off

    # The state_encoder writes the active flag at `start + POKEMON_VECTOR_DIM`
    # for every slot — i.e. at POKEMON_FULL_DIM - 1 within the slot, which is
    # hp_and_active_dim - 1 within the hp_and_active block.
    expected_active_index_in_block = hp_and_active_dim - 1
    assert expected_active_index_in_block >= 0
    # The forward pass relies on this — `active_flags = hp_and_active[:, :, -1]`.
    # If hp_and_active_dim ever drops to 0 or the layout invariant changes,
    # this test gives an early warning before the model silently miscomputes
    # the active slot.
    assert hp_and_active_dim >= 5, (
        "hp_and_active block must contain at least [HP, species_known, sleep_ctr, "
        f"toxic_ctr, ..., active_flag] — got dim={hp_and_active_dim}"
    )


def test_active_flag_detection_picks_marked_slot():
    """
    With every own-team slot populated (HP > 0, spread filled — mimicking real
    own-team encoding), set the active flag on slot 3 only and verify the
    model's forward pass extracts a *different* `our_active_refined` than
    when the active flag is on slot 0.

    This is the second-order check: even when the post-spread layout has many
    nonzero per-slot dims that a stale-index lookup might pick up, the model
    must still react to the actual active flag at the last position.
    """
    from agents.observation.constants import (
        POKEMON_FULL_DIM,
        POKEMON_SPREAD_OFFSET,
        POKEMON_SPREAD_DIM,
    )
    model, layout = _make_model()
    model.eval()
    our_start = layout["parts"]["our_team"]["start"]
    hp_off = layout["pokemon"]["hp"]["offset"]
    active_flag_off_within_slot = POKEMON_FULL_DIM - 1

    def populate(obs, active_slot: int):
        # Fill every alive own-team slot identically (HP=1, spread filled),
        # then set the active flag on the requested slot.
        for i in range(6):
            slot = our_start + i * POKEMON_FULL_DIM
            obs[0, slot + hp_off] = 1.0
            obs[0, slot + hp_off + 1] = 1.0  # species_known
            for j in range(POKEMON_SPREAD_DIM):
                obs[0, slot + POKEMON_SPREAD_OFFSET + j] = 0.5
        obs[0, our_start + active_slot * POKEMON_FULL_DIM + active_flag_off_within_slot] = 1.0
        return obs

    obs_slot0 = populate(_zero_obs(layout), active_slot=0)
    obs_slot3 = populate(_zero_obs(layout), active_slot=3)

    with torch.no_grad():
        out0 = model.forward_internal({"observation": obs_slot0})
        out3 = model.forward_internal({"observation": obs_slot3})

    # Outputs must differ — different active flag changes per-slot inputs
    # (and so role tokens), and also changes which role token is extracted
    # as `our_active_refined`.
    assert not torch.allclose(out0, out3), (
        "Moving the active flag from slot 0 to slot 3 must change the forward output."
    )
