"""Pure unit tests for the gradient-balance + value-scale diagnostics (no battle, no SB3)."""
import numpy as np
import torch as th
from torch import nn

from agents.training.grad_balance import (
    SHARED_TRUNK_PHASES,
    grad_balance_metrics,
    shared_trunk_parameters,
    value_scale_metrics,
)


class _ExtractorStub(nn.Module):
    """Mimics Gen3FeaturesExtractor's phase attributes: 4 shared + 3 head-private."""

    def __init__(self):
        super().__init__()
        # shared trunk
        self.embeddings = nn.Linear(2, 2)
        self.pokemon_encoder = nn.Linear(2, 2)
        self.team_transformer = nn.Linear(2, 2)
        self.assembler = nn.Linear(2, 2)
        # head-private — must be EXCLUDED
        self.cls_pool = nn.Linear(2, 2)
        self.projection = nn.Linear(2, 2)
        self.value_projection = nn.Linear(2, 2)


def test_shared_trunk_selects_only_shared_phases():
    sp = shared_trunk_parameters(_ExtractorStub())
    # 4 shared Linear modules × (weight + bias) = 8 params; the 3 head modules are excluded.
    assert len(sp) == 8
    assert all(p.requires_grad for p in sp)


def test_shared_trunk_empty_for_unrelated_module():
    class _Other(nn.Module):
        def __init__(self):
            super().__init__()
            self.foo = nn.Linear(2, 2)

    assert shared_trunk_parameters(_Other()) == []


def test_shared_trunk_phases_are_a_named_constant():
    # Guards the single-source-of-truth contract: the allow-list excludes the heads + cls_pool.
    assert "team_transformer" in SHARED_TRUNK_PHASES
    assert "cls_pool" not in SHARED_TRUNK_PHASES
    assert "projection" not in SHARED_TRUNK_PHASES


def _trunk_and_heads():
    th.manual_seed(0)
    trunk = nn.Linear(4, 4)
    pi_head = nn.Linear(4, 2)
    vf_head = nn.Linear(4, 1)
    x = th.randn(8, 4)
    h = trunk(x)
    return trunk, pi_head, vf_head, h


def test_grad_balance_ranges_and_keys():
    trunk, pi_head, vf_head, h = _trunk_and_heads()
    policy_term = pi_head(h).pow(2).mean()
    value_term = vf_head(h).pow(2).mean()
    m = grad_balance_metrics(policy_term, value_term, list(trunk.parameters()))
    # No aux → RL-heads-only: policy_share + value_share == 1 (2-way), no aux_share key.
    assert set(m) == {
        "grad/policy_share", "grad/value_share", "grad/value_policy_logratio",
        "grad/policy_value_cosine", "grad/policy_norm_shared", "grad/value_norm_shared",
    }
    assert 0.0 <= m["grad/value_share"] <= 1.0
    assert 0.0 <= m["grad/policy_share"] <= 1.0
    assert abs(m["grad/policy_share"] + m["grad/value_share"] - 1.0) < 1e-6
    assert -1.0 <= m["grad/policy_value_cosine"] <= 1.0
    assert m["grad/policy_norm_shared"] > 0.0
    assert m["grad/value_norm_shared"] > 0.0
    # log-ratio is consistent with value_share: both say the same side dominates.
    import math
    assert math.isclose(
        m["grad/value_policy_logratio"],
        math.log10(m["grad/value_norm_shared"] / m["grad/policy_norm_shared"]),
        rel_tol=1e-6,
    )
    assert (m["grad/value_policy_logratio"] > 0.0) == (m["grad/value_share"] > 0.5)


def test_grad_balance_identical_terms_are_aligned_and_balanced():
    trunk, pi_head, _vf, h = _trunk_and_heads()
    term = pi_head(h).pow(2).mean()
    m = grad_balance_metrics(term, term, list(trunk.parameters()))
    # Same gradient on both sides → perfectly aligned, exactly balanced.
    assert abs(m["grad/policy_value_cosine"] - 1.0) < 1e-5
    assert abs(m["grad/value_share"] - 0.5) < 1e-5
    assert abs(m["grad/policy_share"] - 0.5) < 1e-5
    assert abs(m["grad/value_policy_logratio"] - 0.0) < 1e-5  # ratio 1 → log10 = 0


def test_grad_balance_aux_terms_break_out_each_individually():
    """Each ``aux_terms`` entry adds its OWN ``grad/<name>_{share,norm_shared,policy_cosine}`` block;
    only the names passed appear. All shares ride the shared-trunk pull and stay in [0,1]."""
    trunk, pi_head, vf_head, h = _trunk_and_heads()
    species_head = nn.Linear(4, 3)
    move_head = nn.Linear(4, 7)
    latent_head = nn.Linear(4, 5)
    policy_term = pi_head(h).pow(2).mean()
    value_term = vf_head(h).pow(2).mean()
    aux_terms = {
        "species_belief": species_head(h).pow(2).mean(),
        "move_belief": move_head(h).pow(2).mean(),
        "latent": latent_head(h).pow(2).mean(),
    }
    m = grad_balance_metrics(policy_term, value_term, list(trunk.parameters()), aux_terms=aux_terms)
    for name in ("species_belief", "move_belief", "latent"):
        assert 0.0 <= m[f"grad/{name}_share"] <= 1.0
        assert m[f"grad/{name}_norm_shared"] > 0.0
        assert -1.0 <= m[f"grad/{name}_policy_cosine"] <= 1.0
    assert "grad/aux_share" in m
    # A name NOT passed gets no keys (e.g. win_prob / value_dist off this minibatch).
    assert not any(k.startswith("grad/win_prob") for k in m)
    assert not any(k.startswith("grad/value_dist") for k in m)


def test_grad_balance_shares_sum_to_one_on_common_denominator():
    """policy + value + every aux share are on ONE denominator → they sum to ~1, and aux_share is
    exactly the sum of the per-aux shares (the comparability the common denominator buys)."""
    trunk, pi_head, vf_head, h = _trunk_and_heads()
    policy_term = pi_head(h).pow(2).mean()
    value_term = vf_head(h).pow(2).mean()
    aux_terms = {
        "species_belief": nn.Linear(4, 3)(h).pow(2).mean(),
        "move_belief": nn.Linear(4, 7)(h).pow(2).mean(),
        "latent": nn.Linear(4, 5)(h).pow(2).mean(),
        "win_prob": nn.Linear(4, 1)(h).pow(2).mean(),
    }
    m = grad_balance_metrics(policy_term, value_term, list(trunk.parameters()), aux_terms=aux_terms)
    per_aux = sum(m[f"grad/{name}_share"] for name in aux_terms)
    total = m["grad/policy_share"] + m["grad/value_share"] + per_aux
    assert abs(total - 1.0) < 1e-6
    assert abs(m["grad/aux_share"] - per_aux) < 1e-6


def test_grad_balance_detached_aux_has_zero_share():
    trunk, pi_head, vf_head, h = _trunk_and_heads()
    policy_term = pi_head(h).pow(2).mean()
    value_term = vf_head(h).pow(2).mean()
    latent_term = nn.Linear(4, 5)(h.detach()).pow(2).mean()  # cut off from the trunk
    m = grad_balance_metrics(
        policy_term, value_term, list(trunk.parameters()), aux_terms={"latent": latent_term},
    )
    assert m["grad/latent_norm_shared"] == 0.0
    assert m["grad/latent_share"] == 0.0
    assert m["grad/latent_policy_cosine"] == 0.0  # guarded zero-norm cosine
    assert m["grad/aux_share"] == 0.0  # the only aux is detached


def test_grad_balance_value_detached_has_zero_share():
    trunk, pi_head, vf_head, h = _trunk_and_heads()
    policy_term = pi_head(h).pow(2).mean()
    # Value term cut off from the trunk → no gradient reaches the shared params.
    value_term = vf_head(h.detach()).pow(2).mean()
    m = grad_balance_metrics(policy_term, value_term, list(trunk.parameters()))
    assert m["grad/value_norm_shared"] == 0.0
    assert m["grad/value_share"] == 0.0
    assert m["grad/policy_value_cosine"] == 0.0  # guarded zero-norm cosine
    assert m["grad/value_policy_logratio"] == 0.0  # guarded zero-norm log-ratio


def test_grad_balance_probe_is_read_only():
    # The probe must NOT consume the graph: a real backward still works afterwards.
    trunk, pi_head, vf_head, h = _trunk_and_heads()
    policy_term = pi_head(h).pow(2).mean()
    value_term = vf_head(h).pow(2).mean()
    loss = policy_term + 0.5 * value_term
    grad_balance_metrics(policy_term, value_term, list(trunk.parameters()))
    loss.backward()  # would raise "backward through the graph a second time" if probe freed it
    assert trunk.weight.grad is not None


def test_value_scale_metrics_known_values():
    r = np.array([1.0, -3.0, 2.0])
    v = np.full(3, 0.5)
    m = value_scale_metrics(r, v)
    assert abs(m["train/return_mean"] - 0.0) < 1e-9          # (1 - 3 + 2) / 3
    assert abs(m["train/return_std"] - float(np.std(r))) < 1e-9
    assert abs(m["train/return_abs_max"] - 3.0) < 1e-9
    assert abs(m["train/value_pred_std"] - 0.0) < 1e-9


def test_value_scale_metrics_flattens_and_handles_empty():
    m = value_scale_metrics(np.zeros((4, 3)), np.ones((4, 3)))  # 2-D buffer shapes
    assert m["train/return_abs_max"] == 0.0
    assert m["train/value_pred_std"] == 0.0
    assert value_scale_metrics(np.array([]), np.array([])) == {}
