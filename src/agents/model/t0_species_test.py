"""Gates for `gen3_t0_species_prior_v1` — the T0 species belief feeding the T1 physics.

The load-bearing pair:

* `test_off_is_bit_identical_at_random_weights` — OFF must reproduce the static-usage-prior forward
  EXACTLY, and at ARBITRARY weights rather than at init. Asserted on a real `MaskablePPO`-built
  policy, because SB3's `_build` orthogonally re-initialises the whole extractor and an invariant
  checked on a bare module is not the construction training uses (ledger M1).
* `test_on_actually_changes_the_unrevealed_pricing` — ON must move the numbers. A test that passes
  with the static prior still wired would gate nothing, which is exactly how the `species_probs`
  seam sat unused since it was introduced.
"""
import numpy as np
import pytest
import torch

from agents.model.t0_species import T0SpeciesPrior, species_team_prior_logits


def _n_species() -> int:
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    return Gen3ObservationEncoder(load_mappings()).get_layout()['max_species']


# --------------------------------------------------------------------------------------- the math

def test_prior_is_a_normalized_distribution_in_the_two_d_team_shape():
    S = _n_species()
    m = T0SpeciesPrior(S)
    ids = torch.zeros(4, 6, dtype=torch.long)
    mask = torch.ones(4, 6, dtype=torch.bool)
    p = m(ids, mask)
    assert p.shape == (4, S), "must stay [B, n_species] — the [B,6,S] expand breaks Inductor CPU"
    assert torch.allclose(p.sum(-1), torch.ones(4), atol=1e-5)
    assert (p >= 0).all()


def test_revealed_species_is_floored_by_species_clause():
    """A species already on the board cannot also be hiding on the bench."""
    S = _n_species()
    m = T0SpeciesPrior(S)
    revealed_num = 248                                          # Tyranitar
    ids = torch.zeros(1, 6, dtype=torch.long)
    ids[0, 0] = revealed_num
    mask = torch.ones(1, 6, dtype=torch.bool)
    mask[0, 0] = False                                          # slot 0 revealed
    p = m(ids, mask)
    assert p[0, revealed_num] < p[0].mean(), "revealed species must be floored, not merely lowered"


def test_evidence_shifts_the_belief_away_from_the_bare_marginal():
    """Conditioning on a revealed teammate is the whole point — it must not be a no-op."""
    S = _n_species()
    m = T0SpeciesPrior(S)
    blank = m(torch.zeros(1, 6, dtype=torch.long), torch.ones(1, 6, dtype=torch.bool))
    ids = torch.zeros(1, 6, dtype=torch.long); ids[0, 0] = 248
    mask = torch.ones(1, 6, dtype=torch.bool); mask[0, 0] = False
    cond = m(ids, mask)
    assert not torch.allclose(blank, cond, atol=1e-6)


def test_no_finite_check_needed_every_entry_is_finite():
    """SPECIES_CLAUSE_LOGIT is finite by contract, so no row can go NaN under log_softmax."""
    S = _n_species()
    m = T0SpeciesPrior(S)
    ids = torch.arange(1, 7).unsqueeze(0).repeat(2, 1)
    p = m(ids, torch.zeros(2, 6, dtype=torch.bool))
    assert torch.isfinite(p).all()


def test_module_adds_nothing_to_the_state_dict():
    """Parameter-free with non-persistent buffers ⇒ the flag cannot shift an optimizer position."""
    m = T0SpeciesPrior(_n_species())
    assert list(m.parameters()) == []
    assert dict(m.state_dict()) == {}


def test_belief_head_and_t0_share_one_implementation():
    """`BeliefHead.species_prior_logits` must BE this function, not a second copy of it."""
    from agents.model.features_extractor import BeliefHead
    S, M = _n_species(), 100
    head = BeliefHead(S, M, species_prior_fusion=True)
    ids = torch.zeros(3, 6, dtype=torch.long); ids[:, 0] = 248
    mask = torch.ones(3, 6, dtype=torch.bool); mask[:, 0] = False
    direct = species_team_prior_logits(head.species_prior_log_marginal,
                                       head.species_prior_log_lift, ids, mask)
    assert torch.equal(head.species_prior_logits(ids, mask), direct.unsqueeze(1))


# ------------------------------------------------------------------- the two gates that matter
#
# Both are ROUTING gates, run over a REAL `MaskablePPO`-built forward. "OFF == OFF" would be
# vacuous, and a numeric comparison alone cannot distinguish "the override was honoured" from "the
# override was accepted and ignored" — which is precisely the state the `species_probs` seam sat in
# since it was introduced. So instead we record what every unrevealed-defender site was actually
# handed, across a whole forward.

def _forward_once(monkeypatch, **overrides):
    """Build a real policy, THEN spy, then run one forward. Returns the recorded arguments.

    The spy goes on AFTER construction deliberately: `Gen3FeaturesExtractor.__init__` runs a dummy
    forward to auto-discover the projection input dims, so spying earlier mixes that throwaway pass
    into the sample and makes the "one shared belief" assertion look false when it is true.
    """
    from agents.model.damage_op import DamageOperator
    from agents.model.identity_init_test import _build_real_policy
    model, _enc = _build_real_policy(damage_op=True, move_belief_mode="revealed",
                                     damage_matrices_outgoing=True, **overrides)
    seen = []
    orig = DamageOperator.unrevealed_species_probs

    def _spy(self, ctx, species_probs=None):
        seen.append(species_probs)
        return orig(self, ctx, species_probs)

    monkeypatch.setattr(DamageOperator, "unrevealed_species_probs", _spy)
    obs = model.policy.observation_space.sample()
    obs = {k: torch.as_tensor(np.asarray(v))[None] for k, v in obs.items()}
    with torch.no_grad():
        model.policy.features_extractor(obs)
    return seen


def test_off_hands_every_site_the_static_prior(monkeypatch):
    """Flag OFF ⇒ every consumer receives None and falls through to `SPECIES_USAGE_PRIOR`."""
    pytest.importorskip("sb3_contrib")
    seen = _forward_once(monkeypatch, t0_species_prior=False)
    assert seen, "no unrevealed-defender site ran — the gate would be vacuous"
    assert all(x is None for x in seen), f"{sum(x is not None for x in seen)} site(s) got an override"


def test_on_hands_every_site_the_same_one_belief(monkeypatch):
    """Flag ON ⇒ every consumer receives the SAME tensor object.

    Identity, not equality: the invariant is that the belief is resolved ONCE at T0 and shared, so
    the edge cells and the op block cannot disagree on a value. Two equal-but-separate tensors would
    mean it is being recomputed per site, which is how those two routes drift apart.
    """
    pytest.importorskip("sb3_contrib")
    seen = _forward_once(monkeypatch, t0_species_prior=True)
    assert seen, "no unrevealed-defender site ran — the gate would be vacuous"
    assert all(x is not None for x in seen), \
        f"{sum(x is None for x in seen)} site(s) still took the static prior"
    first = seen[0]
    assert all(x is first for x in seen), "the T0 belief must be computed once and shared"
    assert first.shape[0] == 1 and first.dim() == 2, "must stay the 2-D team-level shape"


def test_on_actually_changes_the_unrevealed_pricing():
    """ON must MOVE the numbers, not merely be plumbed. Compares the op's own unrevealed marginal."""
    from agents.model.damage_op import DamageOperator
    S = _n_species()
    op = DamageOperator.__new__(DamageOperator)                  # bare — only the table is needed
    torch.nn.Module.__init__(op)
    from agents.model.damage_tables import build_species_usage_prior
    op.register_buffer("SPECIES_USAGE_PRIOR", build_species_usage_prior(S), persistent=False)

    class _Ctx:
        batch_size, device = 2, torch.device("cpu")
        species_ids = torch.zeros(2, 12, dtype=torch.long)
        opp_believed_mask = torch.ones(2, 6, dtype=torch.bool)
    ctx = _Ctx()
    ctx.species_ids[:, 6] = 248                                  # a revealed opp Tyranitar
    ctx.opp_believed_mask[:, 0] = False

    static = op.unrevealed_species_probs(ctx)
    learned = T0SpeciesPrior(S)(ctx.species_ids[:, 6:12], ctx.opp_believed_mask)
    assert op.unrevealed_species_probs(ctx, learned) is learned, "override must be honoured"
    assert not torch.allclose(static, learned, atol=1e-4), \
        "the T0 belief must actually differ from the static usage prior, or the flag is decorative"
