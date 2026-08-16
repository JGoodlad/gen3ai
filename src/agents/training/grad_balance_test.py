"""Pure unit tests for the gradient-balance + value-scale diagnostics (no battle, no SB3)."""
import numpy as np
import pytest
import torch as th
from torch import nn

from agents.training.grad_balance import (
    SHARED_TRUNK_PHASES,
    edge_family_metrics,
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


# ---------------------------------------------------------------- per-family edge liveness
# The gap: every edge family enters ZERO-INIT, so a family that never learns is bit-identical in
# the logs to one that works. The v79 `h` family shipped into a production run with no in-flight
# way to tell. These pin the two numbers that make it readable.

class _FakeEdgeBias(th.nn.Module):
    def __init__(self, fams):
        super().__init__()
        self.families = set(fams)
        for f in fams:
            lin = th.nn.Linear(4, 8)
            th.nn.init.zeros_(lin.weight); th.nn.init.zeros_(lin.bias)
            setattr(self, f"{f}_map", lin)


class _FakeExtractor(th.nn.Module):
    def __init__(self, fams=("d1", "h")):
        super().__init__()
        self.edge_bias = _FakeEdgeBias(fams)


def test_a_zero_init_family_reads_zero_weight_norm():
    """The init state must be legible as 'has not moved', not as an absent metric."""
    m = edge_family_metrics(_FakeExtractor())
    assert m["edge/h_weight_norm"] == pytest.approx(0.0)
    assert m["edge/d1_weight_norm"] == pytest.approx(0.0)


def test_a_family_that_has_learned_reads_nonzero():
    fe = _FakeExtractor()
    with th.no_grad():
        fe.edge_bias.h_map.weight.add_(0.5)
    m = edge_family_metrics(fe)
    assert m["edge/h_weight_norm"] > 0.0
    assert m["edge/d1_weight_norm"] == pytest.approx(0.0), "only the touched family moves"


def test_grad_norm_appears_only_after_a_backward():
    """The PAIR is the point: weight~0 AND grad~0 = dead; weight~0 with grad>0 = still climbing.
    A weight norm alone cannot tell those apart, so the grad half must actually be emitted."""
    fe = _FakeExtractor()
    assert "edge/h_grad_norm" not in edge_family_metrics(fe), "no backward yet => no grad key"
    fe.edge_bias.h_map(th.ones(2, 4)).sum().backward()
    m = edge_family_metrics(fe)
    assert m["edge/h_grad_norm"] > 0.0
    assert "edge/d1_grad_norm" not in m, "an untouched family still has no grad"


def test_only_ENABLED_families_are_reported():
    """`families` is the enabled set; a map attribute that exists but is disabled must not appear."""
    m = edge_family_metrics(_FakeExtractor(fams=("d1",)))
    assert set(m) == {"edge/d1_weight_norm"}


def test_a_non_gen3_extractor_returns_empty_rather_than_raising():
    assert edge_family_metrics(th.nn.Linear(2, 2)) == {}


def test_it_reads_the_REAL_extractors_families():
    """Against the production module, not a fake — the attribute contract (`families`, `<fam>_map`)
    is what this helper depends on, and a rename there must fail here rather than silently
    returning {} forever."""
    pytest.importorskip("sb3_contrib")
    from agents.model.features_extractor import EdgeBias

    class _Real(th.nn.Module):
        def __init__(self):
            super().__init__()
            self.edge_bias = EdgeBias("d1,d2,h")

    m = edge_family_metrics(_Real())
    assert set(m) == {"edge/d1_weight_norm", "edge/d2_weight_norm", "edge/h_weight_norm"}
    assert all(v == pytest.approx(0.0) for v in m.values()), "production families are zero-init"
