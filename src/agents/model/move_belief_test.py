"""Unit tests for the MoveBelief reinjection head (predict the opp moveset, FLOW it into the token).

Pins: the forward shapes, the mask GATING the reinjection (only selected slots get the soft-moveset
residual; the rest pass through unchanged bar the shared LayerNorm), the extractor wiring for each mode
(off → no module / None stash; revealed|unrevealed|both → [B,6,M] logits), off being projection-dim-identical
to baseline, the attend-unrevealed dependency guard, and grad flowing through the reinjection into the
trunk (the "make it meaningful" property — the belief reaches the heads, not a dead-end readout).
"""
import numpy as np
import gymnasium as gym
import torch
import pytest

from agents.model.features_extractor import Gen3FeaturesExtractor, MoveBelief, D_MODEL
from agents.observation.constants import TEAM_SIZE
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings


def _make_model(**kwargs):
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    obs_space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    model = Gen3FeaturesExtractor(obs_space, layout=layout, mappings=mappings, **kwargs)
    model.eval()
    return model, layout


def _obs(layout, b=2):
    return {"observation": torch.zeros(b, layout["total_dim"])}


# --------------------------------------------------------------------------- MoveBelief module


def test_move_belief_shapes():
    mb = MoveBelief(n_moves=40, move_emb_dim=16)
    move_emb = torch.nn.Embedding(40, 16)
    opp = torch.randn(2, 6, D_MODEL)
    mask = torch.ones(2, 6, dtype=torch.bool)
    enriched, logits = mb(opp, mask, move_emb)
    assert logits.shape == (2, 6, 40)
    assert enriched.shape == (2, 6, D_MODEL)


def test_move_belief_mask_gates_reinjection():
    """Only the masked-in slots get the soft-moveset residual; the rest pass through unchanged save the
    shared LayerNorm. With a non-trivial reinject weight the selected slot must differ from norm(opp)."""
    torch.manual_seed(0)
    mb = MoveBelief(n_moves=40, move_emb_dim=16)
    torch.nn.init.normal_(mb.reinject.weight, std=1.0)   # make the residual observable (init is ~0)
    move_emb = torch.nn.Embedding(40, 16)
    opp = torch.randn(2, 6, D_MODEL)
    mask = torch.zeros(2, 6, dtype=torch.bool); mask[:, 0] = True    # only slot 0 enriched
    enriched, _ = mb(opp, mask, move_emb)
    norm_only = mb.norm(opp)                              # what an unselected slot collapses to
    for s in range(1, 6):
        assert torch.allclose(enriched[:, s], norm_only[:, s], atol=1e-6)
    assert not torch.allclose(enriched[:, 0], norm_only[:, 0], atol=1e-4)


def test_move_belief_reinject_receives_grad():
    """The reinjection residual must actually affect the enriched output (the "meaningful" path) — a
    gradient on the enriched tokens must reach reinject.weight."""
    torch.manual_seed(0)
    mb = MoveBelief(n_moves=40, move_emb_dim=16)
    move_emb = torch.nn.Embedding(40, 16)
    opp = torch.randn(2, 6, D_MODEL)
    enriched, _ = mb(opp, torch.ones(2, 6, dtype=torch.bool), move_emb)
    enriched.sum().backward()
    assert mb.reinject.weight.grad is not None and mb.reinject.weight.grad.abs().sum() > 0


# --------------------------------------------------------------------------- extractor wiring


@pytest.mark.parametrize("mode", ["revealed", "unrevealed", "both"])
def test_extractor_modes_emit_move_logits(mode):
    model, layout = _make_model(attend_unrevealed_opponents=True, move_belief_mode=mode)
    model.forward_internal(_obs(layout))
    ml = model.last_move_belief_logits
    assert ml is not None
    assert ml.shape == (2, TEAM_SIZE, layout["max_moves"])


def test_extractor_off_stashes_none_and_no_module():
    model, layout = _make_model()
    model.forward_internal(_obs(layout))
    assert model.last_move_belief_logits is None
    assert model.move_belief is None


def test_off_path_projection_dims_unchanged_by_move_belief():
    off, layout = _make_model()
    on, _ = _make_model(attend_unrevealed_opponents=True, move_belief_mode="both")
    pi_off, vf_off = off.forward_internal(_obs(layout))
    pi_on, vf_on = on.forward_internal(_obs(layout))
    assert pi_off.shape[1] == pi_on.shape[1]
    assert vf_off.shape[1] == vf_on.shape[1]


def test_move_belief_requires_attend_unrevealed():
    with pytest.raises(ValueError, match="attend_unrevealed"):
        _make_model(move_belief_mode="revealed")   # attend flag defaults False


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="off|revealed|unrevealed|both"):
        _make_model(attend_unrevealed_opponents=True, move_belief_mode="bogus")


def test_move_belief_loss_backprops_to_head_and_trunk():
    """A loss on the stashed move logits (the supervised path the PPO loop reads) must deposit gradient
    on BOTH the MoveBelief head AND the shared trunk — the move belief is trained in-lineup, not as a
    detached side-head. (Backprop on the stash, mirroring the species aux test; the heads→pooling path
    is exercised separately by the off-vs-on projection-dim test + the module-level reinject grad test —
    an all-zero unit obs key-masks every team slot, so pi/vf carry no trunk grad here.)"""
    model, layout = _make_model(attend_unrevealed_opponents=True, move_belief_mode="both")
    model.train()
    model.forward_internal(_obs(layout))
    ml = model.last_move_belief_logits
    assert ml.requires_grad
    ml.sum().backward()
    assert model.move_belief.move_head.weight.grad.abs().sum() > 0
    trunk_grad = sum(p.grad.abs().sum() for p in model.team_transformer.parameters() if p.grad is not None)
    assert trunk_grad > 0
