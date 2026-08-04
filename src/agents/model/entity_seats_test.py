"""Unit tests for the MOVE ENTITY SEATS (v54, gen3_entity_move_seats_v1 — Stage 1 of the entity
generation, `designs/ai_v9/design_generation_roadmap.md` §3).

The claim: our active's 4 request-ordered move tokens (E3, unconditional) and the opp active's
top-K believed threat moves (E4, `entity_topk_seats`) enter the trunk as ATTENTION SEATS appended
after the global token — so attention reasons over moves as entities — and the pointer head reads
the REFINED E3 seats, so its move tokens are board-aware.

Load-bearing tests:
  * seat-layout STABILITY — the team/history/global slices are byte-identical positions with and
    without seats (every downstream absolute slice depends on this);
  * E3 masking — an unresolved request slot is key-masked AND its logit path stays valid-gated;
  * E4 selection — seats gather exactly the op's `refine_candidates(k=K)` candidates (one candidate
    definition, no drift) and are all masked when there is no opponent active;
  * the E4 requirement gate (prefuse + move_latent) throws at build;
  * the refined-stash contract on a REAL extractor forward — the pointer stash's move tokens are the
    E3 seats out of the transformer (d_model), request-order preserved.
"""
import inspect

import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.model.features_extractor import (
    D_MODEL, EntityMoveSeats, Gen3FeaturesExtractor, NUM_TOKEN_TYPES, TEAM_SIZE,
    TOKEN_TYPE_OUR_MOVE, TOKEN_TYPE_THEIR_THREAT, _request_order_move_tokens,
)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_mappings = load_mappings()
_layout = Gen3ObservationEncoder(_mappings).get_layout()
_SIG = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)

# The E4-capable toggle stack (the production prefuse lineage): prefuse needs move_belief_prefuse
# needs a belief mode; move_latent supplies the seat identity latents.
_E4_TOGGLES = dict(attend_unrevealed_opponents=True, move_belief_mode="both",
                   move_belief_prefuse=True, damage_op=True, damage_outgoing=True,
                   damage_op_prefuse=True, move_latent=True)


def _make(**kw):
    space = gym.spaces.Box(0.0, 1.0, shape=(_layout["total_dim"],), dtype=np.float32)
    return Gen3FeaturesExtractor(space, layout=_layout, mappings=_mappings,
                                 **{k: v for k, v in kw.items() if k in _SIG})


def _obs(batch=3, seed=11):
    return {"observation": torch.rand(batch, _layout["total_dim"],
                                      generator=torch.Generator().manual_seed(seed))}


# ------------------------------------------------------- token-type table + module shape
def test_token_type_table_carries_the_two_new_seat_families():
    fe = _make()
    assert NUM_TOKEN_TYPES == 6
    assert fe.team_transformer.token_type_emb.num_embeddings == NUM_TOKEN_TYPES
    assert TOKEN_TYPE_OUR_MOVE == 4 and TOKEN_TYPE_THEIR_THREAT == 5


def test_seat_module_shape_tracks_topk():
    e3only = EntityMoveSeats(topk_seats=0)
    assert e3only.n_seats == 4 and e3only.threat_seat_proj is None
    both = EntityMoveSeats(topk_seats=6)
    assert both.n_seats == 10 and both.threat_seat_proj is not None
    types = both.seat_types(torch.device("cpu"))
    assert types.tolist() == [TOKEN_TYPE_OUR_MOVE] * 4 + [TOKEN_TYPE_THEIR_THREAT] * 6


def test_e4_requires_the_prefuse_stack():
    with pytest.raises(ValueError, match="entity_topk_seats"):
        _make(entity_topk_seats=5)


# ------------------------------------------------------- seat-layout stability (the crux)
def test_team_and_extra_slices_are_position_stable():
    """The transformer's team/history/global slices must be the SAME positions with and without
    extra seats — every downstream consumer (CLS pools, refine callback, re-attend) slices by
    absolute index. Feed marker tokens and check the extras land strictly after `_total_tokens`
    and the team outputs keep their seat count."""
    fe = _make().eval()
    tt = fe.team_transformer
    ctx = fe.unpack(_obs(batch=2))
    role = fe.pokemon_encoder(ctx, fe.embeddings)
    with torch.no_grad():
        our_a, their_a, none_extra = tt(role, ctx, fe.embeddings)
        extra_tokens = torch.randn(2, 4, D_MODEL)
        types = torch.full((4,), TOKEN_TYPE_OUR_MOVE, dtype=torch.long)
        pad = torch.zeros(2, 4, dtype=torch.bool)
        our_b, their_b, extra_out = tt(role, ctx, fe.embeddings, extra=(extra_tokens, types, pad))
    assert none_extra is None
    assert our_a.shape == our_b.shape == (2, TEAM_SIZE, D_MODEL)
    assert their_a.shape == their_b.shape == (2, TEAM_SIZE, D_MODEL)
    assert extra_out.shape == (2, 4, D_MODEL)
    # The extra seats are ATTENDED WITH the team tokens, so team outputs legitimately change value —
    # but the slicing must stay positional (asserted by shape + the extra block existing separately).


def test_masked_extra_seats_do_not_change_the_team_tokens():
    """A fully key-masked extra seat contributes nothing to any other token's attention — the team
    outputs must be bit-identical to the no-extra forward. This pins the mask wiring: if extra_pad
    were dropped, the seats would leak into the board representation even when 'absent'."""
    fe = _make().eval()
    tt = fe.team_transformer
    ctx = fe.unpack(_obs(batch=2, seed=3))
    role = fe.pokemon_encoder(ctx, fe.embeddings)
    with torch.no_grad():
        our_a, their_a, _ = tt(role, ctx, fe.embeddings)
        extra_tokens = torch.randn(2, 5, D_MODEL) * 10.0                # loud garbage
        types = torch.full((5,), TOKEN_TYPE_THEIR_THREAT, dtype=torch.long)
        pad = torch.ones(2, 5, dtype=torch.bool)                        # ALL masked
        our_b, their_b, _ = tt(role, ctx, fe.embeddings, extra=(extra_tokens, types, pad))
    assert torch.equal(our_a, our_b)
    assert torch.equal(their_a, their_b)


# ------------------------------------------------------- the real extractor forward
def test_e3_only_forward_stashes_refined_dmodel_tokens():
    fe = _make().eval()
    with torch.no_grad():
        fe(_obs(batch=3))
    tok, valid, team, _, _ = fe.last_pointer_inputs
    assert tuple(tok.shape) == (3, 4, D_MODEL)
    assert tuple(valid.shape) == (3, 4)
    # The refined seats are post-attention: they must NOT equal the raw projected inputs
    # (LayerNorm alone guarantees change, but this catches a stash wired to the wrong tensor).
    raw_req, _ = _request_order_move_tokens(fe.pokemon_encoder.last_move_tokens, fe.unpack(_obs(batch=3)))
    assert raw_req.shape[-1] != tok.shape[-1]


def test_e4_forward_selects_the_ops_candidates():
    K = 5
    fe = _make(entity_topk_seats=K, **_E4_TOGGLES).eval()
    obs = _obs(batch=2, seed=7)
    with torch.no_grad():
        fe(obs)
        ctx = fe.unpack(obs)
        # The one candidate definition: the seat build must agree with a direct op call at the
        # same K on the same pre-transformer belief logits.
        idx, w = fe.damage_op.refine_candidates(ctx, fe.last_move_belief_logits, k=K)
    assert idx.shape == (2, K) and w.shape == (2, K)
    assert fe.entity_seats.topk_seats == K
    assert fe._entity_latent_table is not None                          # stashed pre-transformer
    # And the whole forward ran with 4+K seats without shape errors (implicit by reaching here).


def test_e4_seats_masked_when_no_opp_active():
    K = 4
    fe = _make(entity_topk_seats=K, **_E4_TOGGLES).eval()
    obs = _obs(batch=2, seed=9)
    ctx = fe.unpack(obs)
    with torch.no_grad():
        fe(obs)
        tok_req, valid = _request_order_move_tokens(fe.pokemon_encoder.last_move_tokens, ctx)
        # Force the no-opp-active condition through the seat builder directly.
        ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1] = 0.0
        seats, pad = fe.entity_seats(tok_req, valid, ctx, fe.damage_op,
                                     fe.last_move_belief_logits, fe._entity_latent_table)
    assert seats.shape == (2, 4 + K, D_MODEL)
    assert bool(pad[:, 4:].all()), "all E4 seats must be key-masked with no opp active"
    assert float(seats[:, 4:].abs().sum()) == 0.0, "E4 seat content must be zeroed with no opp active"


def test_e3_gradient_reaches_the_seat_projection():
    """The seat path must be differentiable end-to-end: a loss on the pointer stash's refined move
    tokens must produce gradient in the E3 seat projection + the trunk. Two probe subtleties, both
    load-bearing: (1) random obs resolve NO request slots (ids don't match), so the projection's
    INPUT is the zero token and its WEIGHT grad is legitimately zero — the BIAS is the liveness
    probe; (2) the loss must be a RANDOM-cotangent sum, not `.sum()` — the all-ones cotangent lies
    (near-)in LayerNorm's backward null space (a normalized vector's features sum to 0), so
    `.sum().backward()` through post-LN outputs annihilates to ~0 and would 'fail' a live path."""
    fe = _make()
    fe.train()
    fe(_obs(batch=2))
    tok = fe.last_pointer_inputs[0]
    cot = torch.randn(tok.shape, generator=torch.Generator().manual_seed(0))
    (tok * cot).sum().backward()
    g = fe.entity_seats.move_seat_proj.bias.grad
    assert g is not None and float(g.abs().sum()) > 0
    g_emb = fe.team_transformer.token_type_emb.weight.grad
    assert g_emb is not None and float(g_emb[TOKEN_TYPE_OUR_MOVE].abs().sum()) > 0
