"""gen3_hidden_slot_move_mixture_v1 (E10): the HIDDEN-slot move prior is the parameter-free Smogon
mixture `Σ_s P_T0(s | revealed) · P(m | s)`, so a hidden slot's move posterior now VARIES with the
revealed opponent species (it was a state-independent constant: the unknown sentinel's flat row).

Each test fails on revert: without the mixture every hidden slot reads `move_prior_logits[0]`
whatever the revealed team is.
"""
import pytest
import torch

from agents import gen3_data
from agents.model.belief_heads import MoveBelief
from agents.model.t0_species import T0SpeciesPrior
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings


@pytest.fixture(scope="module")
def parts():
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    mb = MoveBelief(layout["max_moves"], layout["move_embedding_dim"], prior_fusion=True,
                    n_species=layout["max_species"], move_candidate_floor=0.02)
    t0 = T0SpeciesPrior(layout["max_species"])
    return mb, t0


def _num(sid: str) -> int:
    return gen3_data.species.get(sid).num


def _hidden_row(mb, t0, revealed):
    ids = torch.zeros(1, 6, dtype=torch.long)
    for i, sid in enumerate(revealed):
        ids[0, i] = _num(sid)
    believed = ids == 0
    tokens = torch.zeros(1, 6, mb.move_head.in_features)
    probs = t0(ids, believed)
    logits = mb.move_logits(tokens, ids, torch.zeros(1, 6, 4, dtype=torch.long),
                            hidden_species_probs=probs, opp_believed_mask=believed)
    return logits, probs, believed


def test_hidden_slot_posterior_varies_with_the_revealed_species(parts):
    mb, t0 = parts
    a, _, _ = _hidden_row(mb, t0, ["tyranitar", "skarmory"])
    b, _, _ = _hidden_row(mb, t0, ["zapdos", "blissey", "gengar"])
    # the last slot is hidden in both batches
    assert (a[0, 5] - b[0, 5]).abs().max().item() > 0.1, "the hidden slot is still a constant"
    # and it is no longer the unknown sentinel's flat row
    assert (a[0, 5] - mb.move_prior_logits[0]).abs().max().item() > 0.1


def test_the_mixture_is_the_T0_weighted_sum_of_the_per_species_prior(parts):
    mb, t0 = parts
    logits, probs, _ = _hidden_row(mb, t0, ["tyranitar"])
    want = (probs.double() @ torch.sigmoid(mb.move_prior_logits.double())).clamp(1e-6, 1 - 1e-6)
    got = torch.sigmoid(logits[0, 5].double())       # the head is zero-init ⇒ posterior == prior
    assert torch.allclose(got, want[0], atol=1e-5)


def test_revealed_slots_keep_their_own_species_row(parts):
    mb, t0 = parts
    logits, _, _ = _hidden_row(mb, t0, ["tyranitar", "skarmory"])
    assert torch.equal(logits[0, 0], mb.move_prior_logits[_num("tyranitar")])
    assert torch.equal(logits[0, 1], mb.move_prior_logits[_num("skarmory")])


def test_no_learned_parameter_touches_the_mixture(parts):
    mb, _ = parts
    probs = torch.full((1, mb.move_prior_logits.shape[0]), 1.0 / mb.move_prior_logits.shape[0],
                       requires_grad=True)
    out = mb.hidden_slot_prior_logits(probs)
    out.sum().backward()
    assert all(p.grad is None for p in mb.parameters()), "the E10 prior must be parameter-free"
