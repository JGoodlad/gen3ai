"""Unit tests for the EDGE-BIAS trunk (v56, gen3_edge_bias_trunk_v1 — Stage 2 of the entity
generation, `designs/ai_v9/design_generation_roadmap.md` §3 Stage 2).

The claim: the encoder stack is a biased-attention clone of the stock layer (same math; the
key-pad mask rides the float bias), and computed physics reaches attention as per-pair per-head
additive logit biases — D1 at (E3 move seat, opp-mon seat) pairs, D3 at (E4 threat seat, our-mon
seat) pairs — through zero-init maps (identity at init).

Load-bearing tests:
  * LAYER PARITY — `BiasedEncoderLayer` with weights copied from a stock
    `nn.TransformerEncoderLayer` reproduces its masked output (the mask-as-float-bias equivalence);
  * IDENTITY AT INIT — families ON vs OFF, same weights, bitwise-equal pi/vf (the zero-init maps);
  * PLACEMENT — a probe map writes cells at exactly the documented seat blocks and nowhere else;
  * the family requirement gates + the v55 version gate/migration;
  * fullgraph compile (the 6.5x compiled-opponent lever must survive);
  * gradient liveness through the bias maps (random cotangent — the LN `.sum()` trap).
"""
import inspect

import dataclasses
import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.model.features_extractor import (
    BiasedEncoderLayer, D_MODEL, EdgeBias, Gen3FeaturesExtractor, TEAM_SIZE,
    TRANSFORMER_FFN_DIM, TRANSFORMER_N_HEADS, _EDGE_D1_CELL, _EDGE_D2_CELL, _EDGE_D3_CELL,
    _EDGE_S1_CELL, _EDGE_D4_CELL, _EDGE_S3_CELL, _EDGE_V_CELL, _KEY_PAD_NEG,
)
from agents.model.model_version import (
    ARCH_SIGNATURE, MODEL_CONFIG_VERSION, ModelVersion, _migrate_config,
)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_mappings = load_mappings()
_layout = Gen3ObservationEncoder(_mappings).get_layout()
_SIG = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)

# The full Stage-2 stack: prefuse (E4/D3 prerequisites) + outgoing (D1) + both families.
_EDGE_TOGGLES = dict(attend_unrevealed_opponents=True, move_belief_mode="both",
                     move_belief_prefuse=True, move_belief_single_compute=True,
                     damage_op=True, damage_outgoing=True, move_latent=True,
                     damage_op_prefuse=True, move_prior_fusion=True,
                     entity_topk_seats=5, edge_bias_families="d1,d2,d3,d4,s1,s3,v")


def _make(**kw):
    space = gym.spaces.Box(0.0, 1.0, shape=(_layout["total_dim"],), dtype=np.float32)
    return Gen3FeaturesExtractor(space, layout=_layout, mappings=_mappings,
                                 **{k: v for k, v in kw.items() if k in _SIG})


def _obs(batch=2, seed=1):
    return {"observation": torch.rand(batch, _layout["total_dim"],
                                      generator=torch.Generator().manual_seed(seed))}


# ------------------------------------------------------- layer parity (the crux)
def test_layer_matches_stock_transformer_layer():
    """Weights copied stock → biased; a key-pad mask expressed as a -1e9 float bias must reproduce
    the stock layer's masked output. This pins BOTH the layer math AND the mask-as-bias delivery."""
    torch.manual_seed(0)
    stock = torch.nn.TransformerEncoderLayer(
        d_model=D_MODEL, nhead=TRANSFORMER_N_HEADS, dim_feedforward=TRANSFORMER_FFN_DIM,
        dropout=0.0, activation="relu", batch_first=True, norm_first=False).eval()
    ours = BiasedEncoderLayer().eval()
    with torch.no_grad():
        ours.in_proj.weight.copy_(stock.self_attn.in_proj_weight)
        ours.in_proj.bias.copy_(stock.self_attn.in_proj_bias)
        ours.out_proj.weight.copy_(stock.self_attn.out_proj.weight)
        ours.out_proj.bias.copy_(stock.self_attn.out_proj.bias)
        for name in ("linear1", "linear2", "norm1", "norm2"):
            getattr(ours, name).load_state_dict(getattr(stock, name).state_dict())
    B, n = 3, 24
    x = torch.randn(B, n, D_MODEL)
    pad = torch.zeros(B, n, dtype=torch.bool)
    pad[:, -5:] = True                                                   # mask the tail seats
    bias = (pad[:, None, None, :].float() * _KEY_PAD_NEG).expand(
        B, TRANSFORMER_N_HEADS, n, n)
    with torch.no_grad():
        ref = stock(x, src_key_padding_mask=pad)
        got = ours(x, bias=bias)
    # Unmasked-seat outputs must agree tightly. (Masked-QUERY rows differ by convention — stock
    # nn.MultiheadAttention NaN-guards fully-masked rows differently — and nothing downstream
    # reads a key-masked seat's own output.)
    live = ~pad[0]
    assert float((ref[:, live] - got[:, live]).abs().max()) < 1e-5


# ------------------------------------------------------- identity at init
def test_families_on_is_bitwise_identical_at_init():
    obs = _obs()
    fe_on = _make(**_EDGE_TOGGLES).eval()
    fe_off = _make(**dict(_EDGE_TOGGLES, edge_bias_families="off")).eval()
    fe_off.load_state_dict(fe_on.state_dict(), strict=False)             # share every common weight
    with torch.no_grad():
        pi_on, vf_on = fe_on(obs)
        pi_off, vf_off = fe_off(obs)
    assert torch.equal(pi_on, pi_off) and torch.equal(vf_on, vf_off)


# ------------------------------------------------------- placement
def test_probe_map_writes_exactly_the_documented_blocks():
    """All five families at once — every documented pair receives a bias, nothing leaks outside.
    D2 targets the batch-varying opp-ACTIVE column via the one-hot, so its block is per-batch."""
    eb = EdgeBias("d1,d2,d3,d4,s1,s3,v")
    with torch.no_grad():                                                # make the maps visible
        for fam in ("d1", "d2", "d3", "d4", "s1", "s3", "v"):
            getattr(eb, f"{fam}_map").bias.fill_(1.0)
    B, K, H = 2, 5, TRANSFORMER_N_HEADS
    base = 20                                                            # the v54 base seat count
    n = base + 4 + K
    bias = torch.zeros(B, H, n, n)
    cells = {
        "d1": torch.randn(B, 4, TEAM_SIZE, _EDGE_D1_CELL),
        "s1": torch.randn(B, 4, TEAM_SIZE, _EDGE_S1_CELL),
        "d2": torch.randn(B, TEAM_SIZE, _EDGE_D2_CELL),
        "d3": torch.randn(B, K, TEAM_SIZE, _EDGE_D3_CELL),
        "s3": torch.randn(B, K, TEAM_SIZE, _EDGE_S3_CELL),
        "v": torch.randn(B, TEAM_SIZE, TEAM_SIZE, _EDGE_V_CELL),
        "d4": torch.randn(B, TEAM_SIZE, TEAM_SIZE, _EDGE_D4_CELL),
    }
    opp_oh = torch.zeros(B, TEAM_SIZE)
    opp_oh[0, 2] = 1.0                                                   # batch 0: opp active slot 2
    opp_oh[1, 4] = 1.0                                                   # batch 1: opp active slot 4
    out = eb(bias, base, cells, opp_oh)
    e3, e4 = base, base + 4
    for b, opp_slot in ((0, 2), (1, 4)):
        mask = torch.zeros(n, n, dtype=torch.bool)
        mask[e3:e3 + 4, TEAM_SIZE:2 * TEAM_SIZE] = True                  # D1+S1 q=move, k=opp mon
        mask[TEAM_SIZE:2 * TEAM_SIZE, e3:e3 + 4] = True                  # transpose
        mask[e4:e4 + K, 0:TEAM_SIZE] = True                              # D3+S3 q=threat, k=our mon
        mask[0:TEAM_SIZE, e4:e4 + K] = True                              # transpose
        mask[0:TEAM_SIZE, TEAM_SIZE + opp_slot] = True                   # D2 q=our mon, k=opp ACTIVE
        mask[TEAM_SIZE + opp_slot, 0:TEAM_SIZE] = True                   # D2 transpose
        mask[0:TEAM_SIZE, TEAM_SIZE:2 * TEAM_SIZE] = True                # V: the full mon↔mon block
        mask[TEAM_SIZE:2 * TEAM_SIZE, 0:TEAM_SIZE] = True                # V transpose
        assert bool((out[b, :, mask] != 0).all()), "every documented pair must receive a bias"
        assert float(out[b, :, ~mask].abs().sum()) == 0.0, "no bias may leak outside the pairs"


# ------------------------------------------------------- requirement gates
def test_family_requirement_gates():
    with pytest.raises(ValueError, match="d1"):
        _make(edge_bias_families="d1")                                   # no damage_op/outgoing
    with pytest.raises(ValueError, match="d3"):
        _make(**dict(_EDGE_TOGGLES, entity_topk_seats=0, edge_bias_families="d3"))
    with pytest.raises(ValueError, match="unknown"):
        EdgeBias("d1,bogus")
    with pytest.raises(ValueError, match="d2"):
        _make(edge_bias_families="d2")                                   # no damage_op
    with pytest.raises(ValueError, match="d3/s3"):
        _make(**dict(_EDGE_TOGGLES, entity_topk_seats=0, edge_bias_families="s3"))
    # The frozen alias: "d" stays {d1,d3} even as new families exist (a saved "d" config must never
    # silently grow maps under newer code).
    eb = EdgeBias("d")
    assert eb.families == {"d1", "d3"} and eb.d2_map is None and eb.s1_map is None


# ------------------------------------------------------- compile
def test_biased_layer_compiles_fullgraph():
    layer = BiasedEncoderLayer().eval()
    compiled = torch.compile(layer, fullgraph=True, dynamic=False)
    x = torch.randn(2, 29, D_MODEL)
    bias = torch.randn(2, TRANSFORMER_N_HEADS, 29, 29) * 0.1
    with torch.no_grad():
        got, ref = compiled(x, bias), layer(x, bias)
    assert float((got - ref).abs().max()) < 1e-4


# ------------------------------------------------------- gradient liveness
def test_map_gradient_direct():
    """The maps are differentiable end-to-end at the module level: nonzero cells + a loss on the
    written bias yield weight gradient in EVERY enabled map (d bias / d W = cell ⊗ upstream)."""
    eb = EdgeBias("d1,d2,d3,d4,s1,s3,v")
    B, K, H = 2, 5, TRANSFORMER_N_HEADS
    base, n = 20, 20 + 4 + 5
    bias = torch.zeros(B, H, n, n)
    cells = {
        "d1": torch.randn(B, 4, TEAM_SIZE, _EDGE_D1_CELL),
        "s1": torch.randn(B, 4, TEAM_SIZE, _EDGE_S1_CELL),
        "d2": torch.randn(B, TEAM_SIZE, _EDGE_D2_CELL),
        "d3": torch.randn(B, K, TEAM_SIZE, _EDGE_D3_CELL),
        "s3": torch.randn(B, K, TEAM_SIZE, _EDGE_S3_CELL),
        "v": torch.randn(B, TEAM_SIZE, TEAM_SIZE, _EDGE_V_CELL),
        "d4": torch.randn(B, TEAM_SIZE, TEAM_SIZE, _EDGE_D4_CELL),
    }
    opp_oh = torch.zeros(B, TEAM_SIZE); opp_oh[:, 0] = 1.0
    out = eb(bias, base, cells, opp_oh)
    (out * torch.randn_like(out)).sum().backward()
    for fam in ("d1", "d2", "d3", "d4", "s1", "s3", "v"):
        assert float(getattr(eb, f"{fam}_map").weight.grad.abs().sum()) > 0, fam


def test_gradient_reaches_the_d3_map_through_the_extractor():
    """End-to-end liveness on the REAL forward via the D3 path (its cells are live whenever an opp
    is active, which random obs satisfy; D1's cells are legitimately zero on random obs — its gates
    [our-active-alive x revealed-opp] see no revealed mons, so its weight grad being 0 there is
    correct behavior, pinned implicitly by identity-at-init). Random cotangent — `.sum()` through
    LayerNorm outputs annihilates (the v54 lesson)."""
    fe = _make(**_EDGE_TOGGLES)
    fe.train()
    pi, vf = fe(_obs())
    cot = torch.randn(pi.shape, generator=torch.Generator().manual_seed(0))
    (pi * cot).sum().backward()
    g3 = fe.edge_bias.d3_map.weight.grad
    assert g3 is not None and float(g3.abs().sum()) > 0


# ------------------------------------------------------- D4 kernel invariants
def test_d4_zeroes_the_active_column_and_unrevealed_attackers():
    """The D4 kernel's cells must be 0 for the opp ACTIVE's column (that quadrant is D3's job) and
    for unrevealed attackers — checked on a real extractor forward's ctx."""
    fe = _make(**_EDGE_TOGGLES).eval()
    obs = _obs(batch=3, seed=13)
    with torch.no_grad():
        fe(obs)
        ctx = fe.unpack(obs)
        cells = fe.damage_op.pairwise_bench_incoming(ctx, fe.last_move_belief_logits)
    assert cells.shape == (3, TEAM_SIZE, TEAM_SIZE, _EDGE_D4_CELL)
    for b in range(3):
        j = int(ctx.opp_active_local[b])
        assert float(cells[b, :, j].abs().sum()) == 0.0, "opp ACTIVE column must be zeroed"
        for jj in range(TEAM_SIZE):
            if bool(ctx.opp_believed_mask[b, jj]):
                assert float(cells[b, :, jj].abs().sum()) == 0.0, "unrevealed attacker must be zeroed"


# ------------------------------------------------------- versioning
def test_version_constants_gate_and_migration():
    assert MODEL_CONFIG_VERSION >= 55
    assert ARCH_SIGNATURE == "gen3_entity_rehome_v1"  # v60 re-home; the biased trunk rides inside it
    fields = {f.name for f in dataclasses.fields(ModelVersion)}
    assert "edge_bias_families" in fields
    migrated = _migrate_config({"config_version": 50})
    assert migrated["edge_bias_families"] == "off" and migrated["config_version"] >= 55
