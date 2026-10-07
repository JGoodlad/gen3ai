"""Unit tests for the hidden-opponent belief READOUT (BeliefHead) on X5's hypothesis tokens.

Since the X5 version break (config v144) the opponent-belief family (`opp_belief_slots`) builds X5's hypothesis
tokens — a hidden opponent slot holds a hypothesised mon at presence π < 1 — and `BeliefSlots`' constant
learned unknown-mon tokens (the blob path) are DELETED (the class too, part 2 of the break). `BeliefHead` still aux-supervises the hidden slots on species + moves. These tests
pin: the aux-logit shapes, the off-path being baseline byte-for-byte (no projection-width change), the
attend-unrevealed dependency guard, that the aux logits carry grad into the hypothesis tokens, and that the
blob's unknown-slot module is not kept. The label plumbing + loss live in the training-side tests.
"""
import numpy as np
import gymnasium as gym
import torch
import pytest

from agents.model.features_extractor import (
    Gen3FeaturesExtractor,
    BeliefHead,
    D_MODEL,
)
from agents.model.x5_surface_fixture import x5_kwargs
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


# The MODULE (still constructed for its init draw — the K9 golden pins the RNG stream — then discarded
# by the extractor): its own contract, kept while the class exists.


# --------------------------------------------------------------------------- BeliefHead


def test_belief_head_shapes():
    head = BeliefHead(n_species=400, n_moves=400)
    out = head(torch.randn(3, TEAM_SIZE, D_MODEL))
    assert out["species"].shape == (3, TEAM_SIZE, 400)
    assert out["moves"].shape == (3, TEAM_SIZE, 400)


# --------------------------------------------------------------------------- extractor wiring


def test_extractor_on_emits_belief_logits():
    model, layout = _make_model(**x5_kwargs())
    pi, vf = model.forward_internal(_obs(layout))
    bl = model.last_belief_logits
    assert bl is not None
    assert bl["species"].shape == (2, TEAM_SIZE, layout["max_species"])
    assert bl["moves"].shape == (2, TEAM_SIZE, layout["max_moves"])


def test_extractor_stashes_believed_mask_for_belief_decode():
    """The forward stashes which opp slots are believed (hidden), so eval/forensic tooling can decode
    the belief head's per-slot species prediction for exactly those slots (inference/belief_decode)."""
    from agents.inference.belief_decode import decode_species_belief

    model, layout = _make_model(**x5_kwargs())
    model.forward_internal(_obs(layout))                       # all-zero obs ⇒ every opp slot hidden
    mask = model.last_opp_believed_mask
    assert mask is not None
    assert mask.shape == (2, TEAM_SIZE) and mask.dtype == torch.bool
    assert bool(mask.all())                                    # species_known=0 everywhere ⇒ all believed

    species_logits = model.last_belief_logits["species"][0].detach().cpu().numpy()
    decoded = decode_species_belief(species_logits, mask[0].cpu().numpy(), top_k=3)
    assert [e["slot"] for e in decoded] == list(range(TEAM_SIZE))   # all hidden slots decoded
    assert all(len(e["top"]) == 3 for e in decoded)
    assert all(t["prob"].endswith("%") for e in decoded for t in e["top"])


def test_extractor_off_stashes_none():
    model, layout = _make_model()
    model.forward_internal(_obs(layout))
    assert model.last_belief_logits is None
    # The believed mask is single-sourced from ctx (computed every forward), so it's present even
    # with belief OFF — but belief decode is gated on last_belief_logits (None here), so the player
    # emits no belief field on an off run.
    assert model.last_opp_believed_mask is not None
    assert model.last_opp_believed_mask.shape == (2, TEAM_SIZE)


def test_off_path_projection_dims_unchanged_by_belief():
    """In-place injection must NOT widen either projection — the belief reaches the heads purely
    through the transformer refining the slot-tokens (CLS pools attend over them), not via concat."""
    off, layout = _make_model()
    on, _ = _make_model(**x5_kwargs())
    pi_off, vf_off = off.forward_internal(_obs(layout))
    pi_on, vf_on = on.forward_internal(_obs(layout))
    assert pi_off.shape[1] == pi_on.shape[1]
    assert vf_off.shape[1] == vf_on.shape[1]


def test_belief_requires_attend_unrevealed():
    with pytest.raises(ValueError, match="attend_unrevealed"):
        _make_model(opp_belief_slots=True)  # attend flag defaults False


def test_belief_logits_carry_grad():
    model, layout = _make_model(**x5_kwargs())
    model.train()
    model.forward_internal(_obs(layout))
    bl = model.last_belief_logits
    assert bl["species"].requires_grad and bl["moves"].requires_grad
    # gradient flows back to the unknown-slot embeddings (believed tokens are supervised)
    # gradient flows back to the hypothesis tokens the hidden slots hold (their builder's δ_θ) — the
    # believed tokens are supervised; the blob's unknown-slot embeddings are not kept at all
    bl["species"].sum().backward()
    assert not hasattr(model, "belief_slots")
    hb = [p.grad for p in model.hypothesis_builder.parameters() if p.grad is not None]
    assert hb and any(float(g.abs().sum()) > 0 for g in hb)


def test_belief_on_forward_works_without_labels_opponent_and_eval_path():
    """The belief (X5's hypothesis tokens + the BeliefHead readout) is part of the FORWARD and needs NO labels for a forward — so a belief-ON
    model plays fine as a self-play opponent / in eval / at inference, none of which provide the
    training-only belief label keys. Forward an obs with ONLY 'observation' (the opponent/eval path)."""
    model, layout = _make_model(**x5_kwargs())
    model.eval()
    obs = {"observation": torch.zeros(3, layout["total_dim"])}   # NO belief_species/belief_moves keys
    pi, vf = model.forward_internal(obs)
    assert pi.shape[0] == 3 and vf.shape[0] == 3                  # usable action/value readout
    assert model.last_belief_logits is not None                  # belief runs in-trunk (stashed, unused)
