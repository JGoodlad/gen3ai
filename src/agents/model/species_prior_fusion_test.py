"""Gates for the team-composition SPECIES prior fused into `BeliefHead` (gen3_species_prior_fusion_v1, v68).

WHY THIS EXISTS. Every belief leg in the extractor fuses a PRIOR with a learned DELTA
(``posterior = prior + head_delta``) — except the species head, which was a bare
``Linear(D_MODEL, n_species)`` cold-starting ~uniform over ~400 nums. This file pins the five
properties that make the fix a fusion rather than a rewrite:

  S1  the prior MATCHES the pure numpy reference (`agents.training.species_priors`) — the model path
      and the offline estimator cannot silently disagree;
  S2  it is CONDITIONAL — revealing a teammate MOVES the distribution, in the direction the measured
      co-occurrence lift says, and revealing nothing gives exactly the marginal;
  S3  SPECIES CLAUSE is enforced and every entry stays FINITE (never -inf, which would poison the
      delta's gradient);
  S4  cold-start posterior == prior EXACTLY, on a REAL `MaskablePPO`-built policy — ledger M1: SB3's
      ortho pass destroys extractor zero-inits, and a zero-init asserted only on a bare module is not
      an invariant on the path training uses;
  S5  OFF is byte-identical to the pre-fusion head, and the flip is version-GATED — the state_dict is
      the same either way, so `check_compatible` is the ONLY thing standing between a resume and a
      silently re-meant species head.
"""
import dataclasses
import inspect

import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.model.damage_tables import SPECIES_CLAUSE_LOGIT, build_species_cooccur_prior
from agents.model.features_extractor import BeliefHead, Gen3FeaturesExtractor
from agents.model.identity_init_test import _build_real_policy
from agents.model.model_version import ModelVersionError, _migrate_config
from agents.training.species_priors import species_prior_table

_N_SPECIES = 400
_N_MOVES = 300


def _head(**kw):
    torch.manual_seed(0)
    return BeliefHead(_N_SPECIES, _N_MOVES, **kw).eval()


def _ids_and_mask(rows, tbl):
    """rows: list of revealed species-id lists → ([B,6] nums, [B,6] believed(=hidden) mask)."""
    ids = torch.zeros(len(rows), 6, dtype=torch.long)
    hidden = torch.ones(len(rows), 6, dtype=torch.bool)
    for b, revealed in enumerate(rows):
        for k, sid in enumerate(revealed):
            ids[b, k] = tbl.num(sid)
            hidden[b, k] = False
    return ids, hidden


# ── S1: the model's on-GPU prior == the offline numpy estimator ───────────────────────────────────

def test_gpu_prior_matches_a_plain_numpy_reference():
    """The `[B,S] @ [S,S]` matmul + scatter + clause-floor must compute the SAME naive Bayes as
    a slow, obviously-correct numpy walk over the SAME two tensors: per row, marginal + the sum
    of each revealed teammate's lift column, revealed nums floored to the clause value, then
    log_softmax. (Until 2026-08-15 the reference was `species_priors.species_prior_logits` —
    the POOL estimator; the tables are Smogon-sourced now, so the reference is built from the
    tables themselves and the test pins the vectorized MATH, not the data source.)"""
    from agents.model.damage_tables import build_species_cooccur_prior

    tbl = species_prior_table()
    log_marginal, log_lift = build_species_cooccur_prior(_N_SPECIES)
    rows = [[], ["tyranitar", "skarmory"], ["cloyster"], ["swampert", "celebi", "gengar"]]
    ids, hidden = _ids_and_mask(rows, tbl)
    prior = _head(species_prior_fusion=True).species_prior_logits(ids, hidden)   # [B,1,S]
    assert prior.shape == (len(rows), 1, _N_SPECIES)

    for b, revealed in enumerate(rows):
        got = prior[b, 0].numpy()
        scores = log_marginal.numpy().copy()
        for r in revealed:
            scores = scores + log_lift[:, tbl.num(r)].numpy()
        for r in revealed:
            scores[tbl.num(r)] = SPECIES_CLAUSE_LOGIT
        want = scores - np.log(np.exp(scores - scores.max()).sum()) - scores.max()
        assert float(np.abs(got - want).max()) < 1e-4, revealed


# ── S2: it is CONDITIONAL, not just a marginal ────────────────────────────────────────────────────

def test_no_revealed_species_gives_exactly_the_marginal():
    """With nothing revealed the evidence sum is empty, so the prior must degrade to the marginal —
    the property that makes this safe on turn 1."""
    tbl = species_prior_table()
    head = _head(species_prior_fusion=True)
    ids, hidden = _ids_and_mask([[]], tbl)
    got = head.species_prior_logits(ids, hidden)[0, 0]
    want = torch.log_softmax(head.species_prior_log_marginal, dim=-1)
    assert torch.allclose(got, want, atol=1e-5), float((got - want).abs().max())


def test_revealing_a_teammate_moves_the_prior_along_the_measured_lift():
    """Anchored to the SMOGON teammate source (2026-08-15, the owner's Smogon-only prior rule):
    Jirachi/Flygon is the strongest positive common pair on the ladder (log-lift +1.12/+0.96 —
    the JiraGon core), so revealing Flygon must raise Jirachi. Skarmory/Forretress is a strong
    ANTI-pair (-2.5 — redundant spikers), so revealing Skarmory must lower Forretress. A prior
    that merely returned the marginal passes every shape assertion and fails both of these.
    (The old pool anchors are gone WITH the pool source: Cloyster→Aerodactyl was +1.32 on the
    719 sample teams and is +0.23 on 2.5M ladder battles — a sample-team archetype artifact,
    which is precisely why priors must not be pool-fit.)"""
    tbl = species_prior_table()
    head = _head(species_prior_fusion=True)
    ids, hidden = _ids_and_mask([[], ["flygon"], ["skarmory"]], tbl)
    p = head.species_prior_logits(ids, hidden)[:, 0]                    # [3, S]
    base, with_flygon, with_skarmory = p[0], p[1], p[2]

    jira, forre = tbl.num("jirachi"), tbl.num("forretress")
    assert float(with_flygon[jira]) > float(base[jira]) + 0.7
    assert float(with_skarmory[forre]) < float(base[forre]) - 1.0
    # And it is not a global shift: the conditioned row is still a normalized log-prob.
    for row in (base, with_flygon, with_skarmory):
        assert float(row.exp().sum()) == pytest.approx(1.0, abs=1e-4)


def test_the_prior_is_shared_across_hidden_slots_and_ignores_the_sentinel():
    """"What is in a hidden slot" is a property of the OPPONENT'S TEAM, so the prior is [B,1,S] and
    broadcasts. A hidden slot carries num 0 (the UNKNOWN sentinel) and must contribute NO evidence —
    which is also what makes the `scatter_` at the clamped index 0 safe."""
    tbl = species_prior_table()
    head = _head(species_prior_fusion=True)
    # Same one revealed mon; the other five slots hidden (num 0) in one row, and in the other the
    # SAME mon with the mask claiming every slot is hidden (so it is not evidence either).
    ids, hidden = _ids_and_mask([["tyranitar"]], tbl)
    only_evidence = head.species_prior_logits(ids, hidden)
    all_hidden = head.species_prior_logits(ids, torch.ones_like(hidden))
    empty, _ = _ids_and_mask([[]], tbl)
    marginal = head.species_prior_logits(empty, torch.ones(1, 6, dtype=torch.bool))
    assert torch.allclose(all_hidden, marginal, atol=1e-6)      # masked-hidden ⇒ no evidence
    assert not torch.allclose(only_evidence, marginal, atol=1e-3)


# ── S3: Species Clause, and no -inf anywhere ──────────────────────────────────────────────────────

def test_species_clause_floors_revealed_species_and_everything_stays_finite():
    tbl = species_prior_table()
    head = _head(species_prior_fusion=True)
    revealed = ["tyranitar", "swampert", "skarmory"]
    ids, hidden = _ids_and_mask([revealed], tbl)
    p = head.species_prior_logits(ids, hidden)[0, 0]
    assert bool(torch.isfinite(p).all()), "a -inf would poison the learned delta's gradient"
    for sid in revealed:
        # Floored to the clause value (shifted by the row's normalizer), far under any real candidate.
        assert float(p[tbl.num(sid)]) < SPECIES_CLAUSE_LOGIT + 1.0
    assert float(p[tbl.num("blissey")]) > SPECIES_CLAUSE_LOGIT + 5.0


def test_unobserved_species_are_improbable_but_not_impossible():
    """A species outside the Smogon usage data is UNOBSERVED, not illegal — the same distinction
    the move prior draws between its `floor` and `_ILLEGAL_PROB`. Every floored entry must sit
    above the Species-Clause value so the learned delta has somewhere to lift it from."""
    log_marginal, _ = build_species_cooccur_prior(_N_SPECIES)
    tbl = species_prior_table()
    floor = float(log_marginal[1:].min())                    # the rarest real candidate
    assert SPECIES_CLAUSE_LOGIT < floor < float(log_marginal[tbl.num("tyranitar")])


# ── S4: cold-start posterior == prior, on a REAL policy (ledger M1) ───────────────────────────────

_FUSION_TOGGLES = dict(opp_belief_slots=True, species_prior_fusion=True)


def test_cold_start_species_posterior_equals_the_prior_on_a_real_policy():
    """The delta head is zero-init so the fused posterior starts AT the prior. SB3's `_build()` ortho
    pass destroys extractor zero-inits (M1) — `restore_identity_init` captures the set by observation,
    so this should be protected automatically. Asserting it on a bare module would prove nothing about
    the path training actually uses, which is the whole lesson of M1."""
    model, _ = _build_real_policy(**_FUSION_TOGGLES)
    fe = model.policy.features_extractor
    head = fe.belief_head
    assert head is not None and head.species_prior_fusion
    assert not bool(head.species_head.weight.any()), \
        "SB3's ortho pass survived — species_head is not in the restore_identity_init capture set"
    assert not bool(head.species_head.bias.any())

    tbl = species_prior_table()
    n_species = head.species_head.out_features
    ids, hidden = _ids_and_mask([["tyranitar", "skarmory"]], tbl)
    tokens = torch.randn(1, 6, head.norm.normalized_shape[0])
    with torch.no_grad():
        got = head.species_logits(tokens, ids, hidden)
        want = head.species_prior_logits(ids, hidden).expand_as(got)
    assert torch.allclose(got, want, atol=1e-6), float((got - want).abs().max())
    # The zero delta must not have collapsed the head into a constant — the prior is doing the work.
    assert float(got[0, 0].std()) > 0.1
    assert got.shape == (1, 6, n_species)


# ── S5: OFF is byte-identical, and the flip is version-gated ──────────────────────────────────────

def test_off_is_byte_identical_to_the_pre_fusion_head():
    """OFF must build no buffer, keep the default init, and produce the SAME numbers — the baseline
    claim the flag's help text makes."""
    off = _head(species_prior_fusion=False)
    assert not off.species_prior_fusion
    assert not hasattr(off, "species_prior_log_marginal")
    assert bool(off.species_head.weight.any()), "OFF must keep the default (non-zero) init"

    tokens = torch.randn(2, 6, off.norm.normalized_shape[0])
    tbl = species_prior_table()
    ids, hidden = _ids_and_mask([["tyranitar"], []], tbl)
    with torch.no_grad():
        # Passing the context must be a NO-OP when fusion is off (nothing silently half-fuses).
        assert torch.equal(off.species_logits(tokens), off.species_logits(tokens, ids, hidden))
        assert torch.equal(off(tokens)["species"], off(tokens, ids, hidden)["species"])


def test_state_dict_keys_are_unchanged_by_the_toggle():
    """The co-occurrence tables are NON-PERSISTENT buffers, so ON and OFF must serialize identically.
    That is precisely why the version gate below is load-bearing: nothing in the weights would ever
    catch this flip."""
    assert "species_prior_fusion" in set(
        inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    on = _head(species_prior_fusion=True)
    off = _head(species_prior_fusion=False)
    assert set(on.state_dict()) == set(off.state_dict())


def test_species_prior_fusion_requires_the_belief_slots():
    """Fail LOUD rather than build a fusion with nothing to fuse into (mirrors move_prior_fusion)."""
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    enc = Gen3ObservationEncoder(load_mappings())
    space = gym.spaces.Dict(
        {"observation": gym.spaces.Box(0.0, 1.0, (enc.dimension,), np.float32)})
    with pytest.raises(ValueError, match="species_prior_fusion"):
        Gen3FeaturesExtractor(space, **enc.get_features_extractor_kwargs(),
                              opp_belief_slots=False, species_prior_fusion=True)


def test_a_pre_v69_config_is_below_the_floor():
    # The v69 default branch is pre-floor since gen3_ctx_dedup_v1 raised MIGRATION_FLOOR.
    from agents.model.model_version import ModelVersionError
    import pytest
    with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
        _migrate_config({"config_version": 67})


def test_flipping_species_prior_fusion_is_a_fatal_resume_mismatch():
    from agents.model.snapshot import current_model_version
    from agents.observation.state_encoder import load_mappings
    base = current_model_version(load_mappings())
    on = dataclasses.replace(base, species_prior_fusion=True)
    with pytest.raises(ModelVersionError, match="species_prior_fusion"):
        on.check_compatible(base)
    with pytest.raises(ModelVersionError, match="species_prior_fusion"):
        base.check_compatible(on)
    on.check_compatible(on)          # a matching pair is fine
