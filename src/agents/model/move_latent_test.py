"""Unit tests for the gen3_unified_move_system_v1 MoveLatentEncoder (Stage 2).

The "Rock Slide ≈ Hidden Power Rock" closeness is a TRAINED property (random at init), so these tests
pin the WIRING that makes it learnable: off byte-identical, the move-network widening, the context-free
latent table (the Stage-3 grading target), type-sensitivity (HP type resolves per slot), and that the
structured MOVE_ATTR actually feeds the latent.
"""
import gymnasium as gym
import numpy as np
import torch

from agents.model.features_extractor import (
    Gen3FeaturesExtractor, MoveLatentEncoder, MOVE_LATENT_DIM, MOVE_NET_HIDDEN, TEAM_SIZE,
)
from agents.model import damage_tables as dt
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.observation.types import TypeEncoder
from agents import gen3_data

_T2I = TypeEncoder.TYPE_TO_IDX
_mp = load_mappings()
_layout = Gen3ObservationEncoder(_mp).get_layout()


def _model(**kw):
    obs_space = gym.spaces.Box(0.0, 1.0, shape=(_layout["total_dim"],), dtype=np.float32)
    return Gen3FeaturesExtractor(obs_space, layout=_layout, mappings=_mp, **kw)


def test_off_builds_no_encoder_and_move_network_unchanged():
    """move_latent=False = baseline: no encoder module, the move-network input is unwidened."""
    off, on = _model(), _model(move_latent=True)
    assert not hasattr(off.pokemon_encoder, "move_latent_encoder")
    assert hasattr(on.pokemon_encoder, "move_latent_encoder")
    off_in = off.pokemon_encoder.move_network[0].in_features
    on_in = on.pokemon_encoder.move_network[0].in_features
    assert on_in - off_in == MOVE_LATENT_DIM
    # No move_latent_encoder weights leak into the OFF state_dict.
    assert not any("move_latent_encoder" in k for k in off.state_dict())
    assert any("move_latent_encoder" in k for k in on.state_dict())


def test_off_path_projection_dims_unchanged():
    """The latent feeds the role token via the move network — the projection inputs are unchanged
    (unlike the damage op, which appends to the projections)."""
    off, on = _model(), _model(move_latent=True)
    assert on.projection_input_dim == off.projection_input_dim
    assert on.value_projection_input_dim == off.value_projection_input_dim


def test_latent_table_shape_and_context_free():
    """The grading table is [n_moves, MOVE_LATENT_DIM] and depends ONLY on the embeddings + buffers —
    not on any battle state (so it is a stable grading target)."""
    on = _model(move_latent=True)
    enc = on.pokemon_encoder.move_latent_encoder
    t1 = enc.latent_table(on.embeddings)
    t2 = enc.latent_table(on.embeddings)
    assert t1.shape == (_layout["max_moves"], MOVE_LATENT_DIM)
    assert torch.equal(t1, t2)                          # deterministic / context-free


def test_forward_stashes_table_only_in_training():
    """The table is stashed for the loss only under grad (training) — rollout/eval pay nothing."""
    on = _model(move_latent=True)
    x = {"observation": torch.rand(2, _layout["total_dim"])}
    on.eval()
    with torch.no_grad():
        on.forward(x)
    assert on.last_move_latent_table is None            # no-grad path skips it
    on.train()
    on.forward(x)
    assert on.last_move_latent_table is not None and on.last_move_latent_table.shape[0] == _layout["max_moves"]


def test_per_slot_latent_resolves_hp_type():
    """The per-slot latent uses the RESOLVED type embedding, so Hidden Power with a Rock type-embedding
    gets a DIFFERENT latent than HP with a Water type-embedding (the type is wired in — the mechanism
    behind 'HP Rock ≈ Rock Slide'). Same move id (237), same MOVE_ATTR — only the type differs."""
    enc = MoveLatentEncoder(_layout)
    emb = torch.nn.Embedding(_layout["max_moves"], _layout["move_embedding_dim"])
    tyemb = torch.nn.Embedding(19, _layout["type_embedding_dim"])
    hp_id = torch.tensor([237])
    move_e = emb(hp_id)
    rock = tyemb(torch.tensor([_T2I["ROCK"]]))
    water = tyemb(torch.tensor([_T2I["WATER"]]))
    lat_rock = enc(move_e, rock, hp_id)
    lat_water = enc(move_e, water, hp_id)
    assert not torch.allclose(lat_rock, lat_water)      # the type embedding changes the latent


def test_move_attr_feeds_the_latent():
    """The structured MOVE_ATTR is part of the encoder input — two moves with the SAME move/type
    embedding but DIFFERENT attributes produce different latents (proves attr-grounding)."""
    enc = MoveLatentEncoder(_layout)
    # Body Slam (par secondary) vs a fabricated attr-zero move: share fixed move/type emb, vary the id
    # so MOVE_ATTR differs. Use Body Slam's num (has secondary) vs Recover's num (status, no secondary).
    bs = torch.tensor([gen3_data.moves.get("bodyslam").num])
    rc = torch.tensor([gen3_data.moves.get("recover").num])
    fixed_move = torch.zeros(1, _layout["move_embedding_dim"])
    fixed_type = torch.zeros(1, _layout["type_embedding_dim"])
    assert not torch.allclose(enc(fixed_move, fixed_type, bs), enc(fixed_move, fixed_type, rc))
    # And the buffers carry the right structured values (MOVE_ATTR is the single source).
    ai = {c: i for i, c in enumerate(dt.MOVE_ATTR_COLS)}
    assert abs(enc.MOVE_ATTR[bs.item(), ai["par"]].item() - 0.30) < 1e-4
    assert enc.MOVE_ATTR[rc.item(), ai["is_heal"]].item() == 1.0


def test_move_latent_forward_finite():
    """A full forward with move_latent on is finite for both heads."""
    on = _model(move_latent=True)
    on.train()
    pi, vf = on.forward({"observation": torch.rand(3, _layout["total_dim"])})
    assert torch.isfinite(pi).all() and torch.isfinite(vf).all()
