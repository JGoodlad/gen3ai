"""Per-phase unit tests for the decomposed Gen3FeaturesExtractor.

These exercise each phase module in isolation — something the monolithic
forward_internal made impossible. ObsUnpack slices are asserted exactly;
CLSPool and ProjectionAssembler run on a hand-built ExtractorContext with
only the fields they consume, with no need to drive a full forward pass.
"""
import dataclasses

import numpy as np
import gymnasium as gym
import torch

from agents.model.features_extractor import (
    Gen3FeaturesExtractor,
    ExtractorContext,
    Embeddings,
    locate_active_slot,
    turn_delta_embed_dim,
    ROLE_TOKEN_SIZE,
    D_MODEL,
    ACTIVE_CTX_HIDDEN,
    N_HISTORY_TURNS,
)
from agents.observation.constants import POKEMON_FULL_DIM, TEAM_SIZE
from agents.observation.moves import HIDDEN_POWER_MOVE_NUM
from agents.observation.turn_delta_encoder import TURN_DELTA_DIM
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.action.constants import ACTION_SPACE_SIZE


def _make_model():
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(layout["total_dim"],), dtype=np.float32)
    model = Gen3FeaturesExtractor(obs_space, layout=layout, mappings=mappings)
    model.eval()
    return model, layout


def _zeros(layout):
    return torch.zeros(1, layout["total_dim"])


def _dummy_ctx(**overrides) -> ExtractorContext:
    """Build an ExtractorContext with every field None, overriding only what a phase reads.
    Lets CLSPool/ProjectionAssembler be tested without a real observation."""
    vals = {f.name: None for f in dataclasses.fields(ExtractorContext)}
    vals.update(overrides)
    return ExtractorContext(**vals)


# ---------------------------------------------------------------------------
# locate_active_slot (pure helper)
# ---------------------------------------------------------------------------

def test_locate_active_slot_picks_set_flag():
    flags = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert locate_active_slot(flags).tolist() == [2, 0]


def test_locate_active_slot_falls_back_to_zero():
    flags = torch.zeros(1, 6)
    assert locate_active_slot(flags).tolist() == [0]


# ---------------------------------------------------------------------------
# ObsUnpack — exact slice extraction
# ---------------------------------------------------------------------------

def test_obsunpack_species_id_slice_is_exact():
    model, layout = _make_model()
    our_start = layout["parts"]["our_team"]["start"]
    species_info = layout["pokemon"]["species"]
    species_idx = species_info["offset"] + species_info["layout"]["species_id"]["offset"]

    obs = _zeros(layout)
    obs[0, our_start + 1 * POKEMON_FULL_DIM + species_idx] = 135.0  # our slot 1 → Jolteon
    ctx = model.unpack({"observation": obs})
    assert ctx.species_ids[0, 1].item() == 135
    assert ctx.species_ids[0, 0].item() == 0


def test_obsunpack_history_span_is_exact():
    model, layout = _make_model()
    hist_start = layout["base_dim"] + ACTION_SPACE_SIZE
    obs = _zeros(layout)
    obs[0, hist_start : hist_start + TURN_DELTA_DIM] = 1.0  # fill the oldest slot
    ctx = model.unpack({"observation": obs})
    assert ctx.turn_history_raw.shape == (1, N_HISTORY_TURNS * TURN_DELTA_DIM)
    assert torch.allclose(ctx.turn_history_raw[0, :TURN_DELTA_DIM], torch.ones(TURN_DELTA_DIM))
    assert ctx.turn_history_raw[0, TURN_DELTA_DIM:].abs().sum() == 0


def test_obsunpack_active_idx_and_fainted_mask():
    model, layout = _make_model()
    our_start = layout["parts"]["our_team"]["start"]
    hp_off = layout["pokemon"]["hp"]["offset"]

    obs = _zeros(layout)
    # Active flag (last dim of slot) on our slot 2, with HP > 0 so it isn't fainted.
    obs[0, our_start + 2 * POKEMON_FULL_DIM + (POKEMON_FULL_DIM - 1)] = 1.0
    obs[0, our_start + 2 * POKEMON_FULL_DIM + hp_off] = 0.9
    ctx = model.unpack({"observation": obs})

    assert ctx.our_active_idx[0].item() == 2
    # Active slot is never masked; an HP=0 bench slot is fainted.
    assert bool(ctx.fainted_mask_ours[0, 2]) is False
    assert bool(ctx.fainted_mask_ours[0, 0]) is True


# ---------------------------------------------------------------------------
# Embeddings — shared table owner
# ---------------------------------------------------------------------------

def test_embeddings_hp_soft_type_is_weighted_lookup():
    model, _ = _make_model()
    emb = model.embeddings
    probs = torch.zeros(1, 1, 16)
    probs[0, 0, 3] = 1.0  # one-hot on candidate type 3
    soft = emb.hp_soft_type(probs)                                   # [1, 1, type_dim]
    expected = emb.type_embedding(emb.hp_type_idx_map[3]).view(1, 1, -1)
    assert torch.allclose(soft, expected, atol=1e-6)


def test_embeddings_delta_slot_width_matches_history_proj():
    model, layout = _make_model()
    out = model.embeddings.embed_delta_slot(torch.zeros(2, TURN_DELTA_DIM))
    assert out.shape == (2, turn_delta_embed_dim(layout))
    assert out.shape[1] == model.team_transformer.history_proj.in_features


def test_embeddings_hp_type_map_matches_type_encoder_order():
    from agents.observation.types import TypeEncoder
    from agents.training.hidden_power_tracker import HIDDEN_POWER_TYPE_ORDER
    model, _ = _make_model()
    expected = [TypeEncoder.TYPE_TO_IDX[t.name] for t in HIDDEN_POWER_TYPE_ORDER]
    assert model.embeddings.hp_type_idx_map.tolist() == expected


# ---------------------------------------------------------------------------
# PokemonEncoder
# ---------------------------------------------------------------------------

def test_pokemon_encoder_role_token_shape():
    model, layout = _make_model()
    ctx = model.unpack({"observation": _zeros(layout)})
    role = model.pokemon_encoder(ctx, model.embeddings)
    assert role.shape == (1, 2 * TEAM_SIZE, ROLE_TOKEN_SIZE)


def test_pokemon_encoder_move_self_attn_wired():
    model, layout = _make_model()
    ctx = model.unpack({"observation": torch.rand(1, layout["total_dim"]) * 4.0})
    before = model.pokemon_encoder(ctx, model.embeddings)
    for p in model.pokemon_encoder.move_self_attn.parameters():
        p.data.zero_()
    after = model.pokemon_encoder(ctx, model.embeddings)
    assert not torch.allclose(before, after)


# ---------------------------------------------------------------------------
# TeamTransformer
# ---------------------------------------------------------------------------

def test_team_transformer_output_shapes():
    model, layout = _make_model()
    ctx = model.unpack({"observation": torch.rand(1, layout["total_dim"]) * 4.0})
    role = model.pokemon_encoder(ctx, model.embeddings)
    our_out, their_out = model.team_transformer(role, ctx, model.embeddings)
    assert our_out.shape == (1, TEAM_SIZE, D_MODEL)
    assert their_out.shape == (1, TEAM_SIZE, D_MODEL)


# ---------------------------------------------------------------------------
# CLSPool — true isolation on a synthetic context
# ---------------------------------------------------------------------------

def test_clspool_active_refined_is_exact_gather():
    model, _ = _make_model()
    B = 3
    our_out = torch.randn(B, TEAM_SIZE, D_MODEL)
    their_out = torch.randn(B, TEAM_SIZE, D_MODEL)
    our_active_idx = torch.tensor([0, 4, 2])
    ctx = _dummy_ctx(
        batch_size=B, device=torch.device("cpu"),
        fainted_mask_ours=torch.zeros(B, TEAM_SIZE, dtype=torch.bool),
        fainted_mask_opp=torch.zeros(B, TEAM_SIZE, dtype=torch.bool),
        our_active_idx=our_active_idx,
    )
    _, _, our_active_refined = model.cls_pool(our_out, their_out, ctx)
    expected = our_out[torch.arange(B), our_active_idx]
    assert torch.allclose(our_active_refined, expected)


def test_clspool_ignores_fainted_keyed_slot():
    """Changing a fainted (key-masked) team slot must not move the pooled output."""
    model, _ = _make_model()
    B = 1
    base = torch.randn(B, TEAM_SIZE, D_MODEL)
    fainted = torch.zeros(B, TEAM_SIZE, dtype=torch.bool)
    fainted[0, 5] = True                       # slot 5 fainted/masked
    ctx = _dummy_ctx(
        batch_size=B, device=torch.device("cpu"),
        fainted_mask_ours=fainted,
        fainted_mask_opp=torch.zeros(B, TEAM_SIZE, dtype=torch.bool),
        our_active_idx=torch.tensor([0]),      # active slot 0, unmasked
    )
    their = torch.randn(B, TEAM_SIZE, D_MODEL)
    pooled_a, _, _ = model.cls_pool(base, their, ctx)
    mutated = base.clone()
    mutated[0, 5] = torch.randn(D_MODEL)       # perturb only the masked slot
    pooled_b, _, _ = model.cls_pool(mutated, their, ctx)
    assert torch.allclose(pooled_a, pooled_b, atol=1e-6)


# ---------------------------------------------------------------------------
# ProjectionAssembler — true isolation
# ---------------------------------------------------------------------------

def test_projection_assembler_width_and_passthrough():
    model, layout = _make_model()
    active_ctx_dim = layout["active_context_dim"]
    B, K = 2, 7
    our_pool = torch.randn(B, D_MODEL)
    their_pool = torch.randn(B, D_MODEL)
    active = torch.randn(B, D_MODEL)
    non_matchup = torch.randn(B, K)
    ctx = _dummy_ctx(
        our_ctx_raw=torch.randn(B, active_ctx_dim),
        opp_ctx_raw=torch.randn(B, active_ctx_dim),
        non_matchup_rest=non_matchup,
    )
    combined = model.assembler(our_pool, their_pool, active, ctx)
    expected_width = 3 * D_MODEL + 2 * ACTIVE_CTX_HIDDEN[1] + K
    assert combined.shape == (B, expected_width)
    # The non-matchup tail is concatenated verbatim.
    assert torch.allclose(combined[:, -K:], non_matchup)
