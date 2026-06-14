"""Unit tests for InstrumentedMaskablePPO._belief_aux_loss (Hungarian / order-invariant belief loss).

The k believed-slot predictions are matched to the k hidden-mon targets by per-sample min-cost
assignment, so the loss is invariant to which believed slot holds which target (the whole point — it
fixes the reveal-shifting per-slot target). The call returns (aux_tensor, metrics_dict, latent_loss)
— latent_loss is None unless the LATENT escalation is exercised (a "latent" logits key + a
latent_target). Tests cover: the None / empty-minibatch guards, grad flow, FAIL-LOUD on out-of-vocab
ids, the diagnostic metrics, order-invariance + min-cost matching, AND the latent cosine + VICReg path.
"""
import pytest
import torch

from agents.training.instrumented_ppo import InstrumentedMaskablePPO

S, M = 50, 40  # small species / move vocabs
D = 8          # small latent dim (the real model uses D_MODEL=128; the loss is dim-agnostic)
_loss = InstrumentedMaskablePPO._belief_aux_loss


def _logits(B, requires_grad=False, seed=0, latent=False):
    g = torch.Generator().manual_seed(seed)
    bl = {
        "species": torch.randn(B, 6, S, generator=g, requires_grad=requires_grad),
        "moves": torch.randn(B, 6, M, generator=g, requires_grad=requires_grad),
    }
    if latent:
        bl["latent"] = torch.randn(B, 6, D, generator=g, requires_grad=requires_grad)
    return bl


def test_returns_none_when_off_or_labels_absent():
    assert _loss(None, torch.zeros(2, 6), torch.zeros(2, 6, 4)) is None
    bl = _logits(2)
    assert _loss(bl, None, torch.zeros(2, 6, 4)) is None
    assert _loss(bl, torch.zeros(2, 6), None) is None


def test_all_pad_minibatch_returns_none_no_nan():
    bl = _logits(3)
    assert _loss(bl, torch.full((3, 6), -1), torch.full((3, 6, 4), -1)) is None


def test_finite_metrics_in_range():
    bl = _logits(2)
    sp = torch.full((2, 6), -1); mv = torch.full((2, 6, 4), -1)
    sp[0, 4] = 10; sp[0, 5] = 22; sp[1, 3] = 7
    mv[0, 4, :2] = torch.tensor([3, 9]); mv[1, 3, 0] = 5
    aux, m, lat = _loss(bl, sp, mv)
    assert torch.isfinite(aux)
    assert lat is None    # no latent prediction / target → no latent term
    assert m["species_ce"] >= 0 and m["moves_bce"] >= 0
    assert 0.0 <= m["species_acc"] <= 1.0
    assert 0.0 <= m["moves_precision"] <= 1.0 and 0.0 <= m["moves_recall"] <= 1.0
    assert m["k_mean"] >= 1.0 and 0.0 < m["coverage"] <= 1.0
    assert m["species_acc_above_chance"] == pytest.approx(m["species_acc"] - 1.0 / S)
    assert "latent_cosine" not in m


def test_order_invariant_under_label_permutation():
    """Swapping which believed slot holds which target must NOT change the loss — the matching
    re-assigns. This is the property the Hungarian loss exists to provide (fixes the reveal-shift)."""
    bl = _logits(1, seed=3)
    sp1 = torch.full((1, 6), -1); sp1[0, 4] = 10; sp1[0, 5] = 22
    mv1 = torch.full((1, 6, 4), -1); mv1[0, 4, 0] = 3; mv1[0, 5, 0] = 9
    sp2 = torch.full((1, 6), -1); sp2[0, 4] = 22; sp2[0, 5] = 10   # same SET, swapped
    mv2 = torch.full((1, 6, 4), -1); mv2[0, 4, 0] = 9; mv2[0, 5, 0] = 3
    a1, m1, _ = _loss(bl, sp1, mv1)
    a2, m2, _ = _loss(bl, sp2, mv2)
    assert torch.allclose(a1, a2, atol=1e-6)
    assert abs(m1["species_ce"] - m2["species_ce"]) < 1e-5
    assert abs(m1["moves_bce"] - m2["moves_bce"]) < 1e-5


def test_matching_picks_min_cost_assignment():
    """If slot i's logits favour target B and slot j's favour target A, the matched CE must be the
    LOW cross-assignment cost, strictly below the naive same-order assignment."""
    bl = {"species": torch.full((1, 6, S), -5.0), "moves": torch.zeros(1, 6, M)}
    A, B = 10, 22
    bl["species"][0, 4, B] = 5.0   # slot 4 confidently predicts B
    bl["species"][0, 5, A] = 5.0   # slot 5 confidently predicts A
    sp = torch.full((1, 6), -1); sp[0, 4] = A; sp[0, 5] = B   # labels in the "wrong" slot order
    mv = torch.full((1, 6, 4), -1)
    _, m, _ = _loss(bl, sp, mv)
    logp = torch.log_softmax(bl["species"][0], -1)
    ce_naive = (-logp[4, A] - logp[5, B]).item() / 2          # the assignment Hungarian should AVOID
    assert m["species_ce"] < ce_naive - 0.5
    assert m["species_acc"] == 1.0


def test_grad_flows_to_logits():
    bl = _logits(2, requires_grad=True)
    sp = torch.full((2, 6), -1); sp[0, 5] = 3; sp[0, 4] = 8
    mv = torch.full((2, 6, 4), -1); mv[0, 5, 0] = 2
    aux = _loss(bl, sp, mv)[0]
    aux.backward()
    assert bl["species"].grad.abs().sum() > 0
    assert bl["moves"].grad.abs().sum() > 0


def test_fails_loud_on_out_of_vocab_species():
    """An out-of-vocab label id is impossible on real data → corrupt pipeline → CRASH, not silent drop."""
    bl = _logits(1)
    sp = torch.full((1, 6), -1); sp[0, 0] = 10; sp[0, 1] = S + 5   # one valid, one out-of-vocab
    mv = torch.full((1, 6, 4), -1)
    with pytest.raises(ValueError, match="n_species"):
        _loss(bl, sp, mv)


def test_fails_loud_on_out_of_vocab_move():
    bl = _logits(1)
    sp = torch.full((1, 6), -1); sp[0, 0] = 10
    mv = torch.full((1, 6, 4), -1); mv[0, 0, 0] = M + 9
    with pytest.raises(ValueError, match="n_moves"):
        _loss(bl, sp, mv)


def test_moves_weight_zero_skips_bce_fastpath():
    """moves_weight=0 must skip the BCE branch entirely (perf) and contribute 0 to the aux."""
    bl = _logits(2, seed=5)
    sp = torch.full((2, 6), -1); sp[0, 4] = 10; sp[1, 5] = 3
    mv = torch.full((2, 6, 4), -1); mv[0, 4, 0] = 2; mv[1, 5, 0] = 7
    aux0, m0, _ = _loss(bl, sp, mv, moves_weight=0.0)
    aux2, m2, _ = _loss(bl, sp, mv, moves_weight=2.0)
    assert m0["moves_bce"] == 0.0                       # bce branch skipped
    # aux = ce + w*bce; w=0 is pure CE, w=2 adds 2*bce
    assert abs((aux0.item() + 2 * m2["moves_bce"]) - aux2.item()) < 1e-5


def test_varied_k_across_batch():
    """Samples with different believed-slot counts (k) are all matched correctly in one call."""
    bl = _logits(3, seed=1)
    sp = torch.full((3, 6), -1)
    sp[0, 5] = 4                                  # k=1
    sp[1, 3] = 7; sp[1, 4] = 11; sp[1, 5] = 2     # k=3
    sp[2, 2] = 1; sp[2, 3] = 9                    # k=2
    mv = torch.full((3, 6, 4), -1)
    aux, m, _ = _loss(bl, sp, mv)
    assert torch.isfinite(aux) and 0.0 <= m["species_acc"] <= 1.0
    assert m["k_mean"] == pytest.approx((1 + 3 + 2) / 3)
    assert m["coverage"] == 1.0


# --- LATENT escalation -------------------------------------------------------------------------

def test_latent_none_without_target():
    """A latent prediction key with NO target → still no latent term (target is the gate)."""
    bl = _logits(2, latent=True)
    sp = torch.full((2, 6), -1); sp[0, 4] = 10
    mv = torch.full((2, 6, 4), -1)
    aux, m, lat = _loss(bl, sp, mv, latent_target=None)
    assert lat is None and "latent_cosine" not in m


def test_latent_cosine_and_metrics():
    """With a latent pred + target, the 3rd return is a finite latent loss and the latent metrics
    appear. A pred parallel to its matched target gives cosine≈1 (cos_dist≈0)."""
    bl = _logits(2, latent=True)
    sp = torch.full((2, 6), -1); sp[0, 4] = 10; sp[1, 5] = 7   # one believed slot each (k=1 → identity match)
    mv = torch.full((2, 6, 4), -1)
    tgt = torch.zeros(2, 6, D)
    # Make each believed slot's prediction parallel to its target → cosine 1.
    tgt[0, 4] = bl["latent"][0, 4].detach()
    tgt[1, 5] = bl["latent"][1, 5].detach()
    aux, m, lat = _loss(bl, sp, mv, latent_target=tgt)
    assert torch.isfinite(lat) and lat.item() >= 0.0
    assert m["latent_cosine"] == pytest.approx(1.0, abs=1e-4)    # parallel ⇒ similarity ~1
    assert "latent_std" in m and "latent_vicreg" in m and "latent_loss" in m


def test_latent_grad_flows_to_predictions():
    bl = _logits(2, requires_grad=True, latent=True)
    sp = torch.full((2, 6), -1); sp[0, 4] = 10; sp[1, 5] = 7
    mv = torch.full((2, 6, 4), -1)
    tgt = torch.randn(2, 6, D)
    lat = _loss(bl, sp, mv, latent_target=tgt)[2]
    lat.backward()
    assert bl["latent"].grad.abs().sum() > 0
    # The target is stop-grad — it must never require grad in the loss path (we pass a plain tensor).


def test_latent_rides_species_matching_order_invariant():
    """The latent term reuses the SPECIES Hungarian assignment. Swapping (species label + latent
    target) together across slots leaves the latent loss invariant (the matching re-assigns both)."""
    bl = _logits(1, seed=4, latent=True)
    A, B = 10, 22
    sp1 = torch.full((1, 6), -1); sp1[0, 4] = A; sp1[0, 5] = B
    sp2 = torch.full((1, 6), -1); sp2[0, 4] = B; sp2[0, 5] = A    # swapped species
    mv = torch.full((1, 6, 4), -1)
    t1 = torch.zeros(1, 6, D); t1[0, 4] = torch.arange(D).float(); t1[0, 5] = -torch.arange(D).float()
    t2 = torch.zeros(1, 6, D); t2[0, 4] = t1[0, 5].clone(); t2[0, 5] = t1[0, 4].clone()  # swapped targets
    _, m1, lat1 = _loss(bl, sp1, mv, latent_target=t1)
    _, m2, lat2 = _loss(bl, sp2, mv, latent_target=t2)
    assert torch.allclose(lat1, lat2, atol=1e-6)
    assert abs(m1["latent_cosine"] - m2["latent_cosine"]) < 1e-5
