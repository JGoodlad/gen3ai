"""Unit tests for the in-place hidden-opponent belief (BeliefSlots + BeliefHead).

The belief fills the opponent's un-revealed team slots with distinct learned unknown-mon tokens
(so the transformer refines them in-lineup) and a BeliefHead aux-supervises them on species+moves.
These tests pin: the in-place replacement semantics, the aux-logit shapes, the off-path being
baseline byte-for-byte (no projection-width change), the attend-unrevealed dependency guard, and
that the aux logits carry grad. The label plumbing + loss live in the training-side tests.
"""
import numpy as np
import gymnasium as gym
import torch
import pytest

from agents.model.features_extractor import (
    Gen3FeaturesExtractor,
    BeliefSlots,
    BeliefHead,
    D_MODEL,
)
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


# --------------------------------------------------------------------------- BeliefSlots


def test_belief_slots_replaces_only_believed_opp_slots():
    torch.manual_seed(0)
    bs = BeliefSlots()
    role = torch.randn(2, 2 * TEAM_SIZE, D_MODEL)
    # batch 0: opp slots 0,1 revealed; 2..5 believed. batch 1: all opp revealed.
    mask = torch.zeros(2, TEAM_SIZE, dtype=torch.bool)
    mask[0, 2:] = True
    out = bs(role, mask)
    # our team untouched
    assert torch.equal(out[:, :TEAM_SIZE, :], role[:, :TEAM_SIZE, :])
    # revealed opp slots untouched
    assert torch.equal(out[0, TEAM_SIZE:TEAM_SIZE + 2, :], role[0, TEAM_SIZE:TEAM_SIZE + 2, :])
    assert torch.equal(out[1, TEAM_SIZE:, :], role[1, TEAM_SIZE:, :])
    # believed slots become the learned per-position unknown token
    for slot in range(2, TEAM_SIZE):
        assert torch.allclose(out[0, TEAM_SIZE + slot, :], bs.unknown_slot_emb[slot])


def test_unknown_slot_embeddings_are_distinct():
    """Distinct-per-position is the whole point — identical tokens would collapse under the
    permutation-equivariant transformer, defeating per-slot specialisation."""
    torch.manual_seed(0)
    bs = BeliefSlots()
    e = bs.unknown_slot_emb
    assert e.shape == (TEAM_SIZE, D_MODEL)
    # pairwise distinct (no two slot embeddings identical)
    for i in range(TEAM_SIZE):
        for j in range(i + 1, TEAM_SIZE):
            assert not torch.allclose(e[i], e[j])


# --------------------------------------------------------------------------- BeliefHead


def test_belief_head_shapes():
    head = BeliefHead(n_species=400, n_moves=400)
    out = head(torch.randn(3, TEAM_SIZE, D_MODEL))
    assert out["species"].shape == (3, TEAM_SIZE, 400)
    assert out["moves"].shape == (3, TEAM_SIZE, 400)


# --------------------------------------------------------------------------- extractor wiring


def test_extractor_on_emits_belief_logits():
    model, layout = _make_model(attend_unrevealed_opponents=True, opp_belief_slots=True)
    pi, vf = model.forward_internal(_obs(layout))
    bl = model.last_belief_logits
    assert bl is not None
    assert bl["species"].shape == (2, TEAM_SIZE, layout["max_species"])
    assert bl["moves"].shape == (2, TEAM_SIZE, layout["max_moves"])


def test_extractor_stashes_believed_mask_for_belief_decode():
    """The forward stashes which opp slots are believed (hidden), so eval/forensic tooling can decode
    the belief head's per-slot species prediction for exactly those slots (inference/belief_decode)."""
    from agents.inference.belief_decode import decode_species_belief

    model, layout = _make_model(attend_unrevealed_opponents=True, opp_belief_slots=True)
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
    on, _ = _make_model(attend_unrevealed_opponents=True, opp_belief_slots=True)
    pi_off, vf_off = off.forward_internal(_obs(layout))
    pi_on, vf_on = on.forward_internal(_obs(layout))
    assert pi_off.shape[1] == pi_on.shape[1]
    assert vf_off.shape[1] == vf_on.shape[1]


def test_belief_requires_attend_unrevealed():
    with pytest.raises(ValueError, match="attend_unrevealed"):
        _make_model(opp_belief_slots=True)  # attend flag defaults False


def test_belief_logits_carry_grad():
    model, layout = _make_model(attend_unrevealed_opponents=True, opp_belief_slots=True)
    model.train()
    model.forward_internal(_obs(layout))
    bl = model.last_belief_logits
    assert bl["species"].requires_grad and bl["moves"].requires_grad
    # gradient flows back to the unknown-slot embeddings (believed tokens are supervised)
    bl["species"].sum().backward()
    assert model.belief_slots.unknown_slot_emb.grad is not None


def test_aux_loss_backprops_through_stash_to_belief_and_trunk():
    """End-to-end gradient flow: the stashed belief logits + the aux loss must deposit gradient on
    BOTH the belief params (unknown-slot embeddings, BeliefHead) AND the shared trunk (team
    transformer) — i.e. the aux objective actually trains the in-lineup belief representation, not a
    detached side-head. Exercises the same stash the PPO train() loop reads."""
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    model, layout = _make_model(attend_unrevealed_opponents=True, opp_belief_slots=True)
    model.train()
    obs = _obs(layout, b=2)                       # all-zero obs ⇒ every opp slot species_known=0 (believed)
    model.forward_internal(obs)                   # stashes last_belief_logits for this forward
    bl = model.last_belief_logits
    sp = torch.full((2, TEAM_SIZE), -1)
    sp[:, 3] = 10; sp[:, 4] = 22; sp[:, 5] = 5    # 3 believed slots with valid species labels
    mv = torch.full((2, TEAM_SIZE, 4), -1); mv[:, 3, 0] = 2
    aux, _ = InstrumentedMaskablePPO._belief_aux_loss(bl, sp, mv)
    aux.backward()
    # belief-specific params
    assert model.belief_slots.unknown_slot_emb.grad is not None
    assert model.belief_slots.unknown_slot_emb.grad.abs().sum() > 0
    assert model.belief_head.species_head.weight.grad.abs().sum() > 0
    # shared trunk — the belief tokens flow through the team transformer, so the aux trains it too
    trunk_grad = sum(p.grad.abs().sum() for p in model.team_transformer.parameters() if p.grad is not None)
    assert trunk_grad > 0


def test_belief_on_forward_works_without_labels_opponent_and_eval_path():
    """The belief module is part of the FORWARD and needs NO labels for a forward — so a belief-ON
    model plays fine as a self-play opponent / in eval / at inference, none of which provide the
    training-only belief label keys. Forward an obs with ONLY 'observation' (the opponent/eval path)."""
    model, layout = _make_model(attend_unrevealed_opponents=True, opp_belief_slots=True)
    model.eval()
    obs = {"observation": torch.zeros(3, layout["total_dim"])}   # NO belief_species/belief_moves keys
    pi, vf = model.forward_internal(obs)
    assert pi.shape[0] == 3 and vf.shape[0] == 3                  # usable action/value readout
    assert model.last_belief_logits is not None                  # belief runs in-trunk (stashed, unused)
