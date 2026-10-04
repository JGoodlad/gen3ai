"""X5 U2's hypothesis builder (`gen3_x5_hypothesis_set_v1`; design_x5_belief_tokens.md §3.1–§3.3, §3.8).

What each test pins, and what reverting it would break:

* the fixed-size construction: Σπ = k (fp64 1e-12 / fp32 1e-5, measured 2.7e-15 / 1.4e-6), π strictly
  inside (0, 1), the provable bracket, the k = 0 / k = n structural branches, an independent fp64
  root-find, the BCE finite on extreme scores, ∂BCE/∂τ = 0 (the autograd gradient with τ under
  no_grad IS the exact implicit gradient when the labels count k);
* V: the dex table's valid nums minus the revealed ones (no sentinel 0, none of the 13 phantom nums
  387–399 the T0 prior floors — F-X5-21);
* the ONE stable ordering, ties to the lower number, and §3.1's rule-8 near-tie exclusion;
* the blob arm builds nothing and its forward / params are unchanged; the fixed_mass arm's
  non-X5 INITIAL bytes equal the blob arm's (private seed, `IsolatedLinear`);
* M10: no policy / value / consumer-facing output puts gradient into δ_θ; the presence BCE does;
* the flag: registry, checkargs' requires graph, the migration, the version gate, production stays blob.
"""
from __future__ import annotations

import copy
import math
from collections import deque

import numpy as np
import pytest
import torch

from agents.model.hypothesis_set import (BELIEF_TOKEN_MODES, BISECTION_ITERS, MASKED_LOG_PRESENCE,
                                         MOVE_GROUP_MASS, SELECTION_TIE_EPS, TYPED_HP_NUMS,
                                         HypothesisBuilder, belief_head_team_scores,
                                         boundary_gap, fixed_mass_presence, fixed_size_tau,
                                         label_multi_hot, move_candidates, near_tie_rows,
                                         set_bce, species_candidates, stable_order)
from agents.model.t0_species import T0SpeciesPrior, species_team_prior_logits
from agents.observation.moves import HIDDEN_POWER_MOVE_NUM


# ----------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def layout():
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    return Gen3ObservationEncoder(load_mappings()).get_layout()


@pytest.fixture(scope="module")
def builder(layout):
    return HypothesisBuilder(layout, global_input_dim=50, n_move_seats=6)


@pytest.fixture(scope="module")
def t0(layout):
    return T0SpeciesPrior(layout["max_species"])


def _reveal_batch(valid: torch.Tensor, marg: torch.Tensor, B: int, seed: int):
    """[B,6] opponent species ids + believed mask: r = b % 7 revealed, drawn by the prior marginal."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.zeros(B, 6, dtype=torch.long)
    bel = torch.ones(B, 6, dtype=torch.bool)
    w = marg * valid
    for b in range(B):
        r = b % 7
        pick = torch.multinomial(w, 6, replacement=False, generator=g)
        ids[b, :r] = pick[:r]
        bel[b, :r] = False
    return ids, bel


@pytest.fixture(scope="module")
def prior_batch(builder, t0):
    marg = t0.species_prior_log_marginal.exp()
    ids, bel = _reveal_batch(builder.species_valid, marg, 700, seed=7)
    logp = species_team_prior_logits(t0.species_prior_log_marginal, t0.species_prior_log_lift, ids, bel)
    return ids, bel, logp


# ----------------------------------------------------------------------------------- V (F-X5-21)
def test_V_is_the_dex_tables_valid_nums_minus_the_revealed_ones(builder):
    valid = builder.species_valid
    assert int(valid.sum()) == 386 and not bool(valid[0]) and not bool(valid[387:].any())
    ids = torch.tensor([[25, 6, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    bel = torch.tensor([[False, False, True, True, True, True], [True] * 6])
    cand, k = species_candidates(valid, ids, bel)
    assert k.tolist() == [4, 6]
    assert not bool(cand[0, 25]) and not bool(cand[0, 6]) and bool(cand[1, 25])
    assert int(cand[0].sum()) == 384 and int(cand[1].sum()) == 386
    # the sentinel and the 13 phantom nums the T0 prior floors at 1e-4 are NEVER candidates
    assert not bool(cand[:, 0].any()) and not bool(cand[:, 387:].any())


def test_a_hidden_slot_carrying_a_stale_nonzero_id_is_not_treated_as_revealed(builder):
    """Revealed = not believed AND num > 0 (the T0 prior's own evidence rule)."""
    ids = torch.tensor([[25, 6, 0, 0, 0, 0]])
    bel = torch.tensor([[False, True, True, True, True, True]])
    cand, k = species_candidates(builder.species_valid, ids, bel)
    assert int(k) == 5 and not bool(cand[0, 25]) and bool(cand[0, 6])


# ----------------------------------------------------------------------------------- the construction
def test_sum_of_presence_is_k_in_fp64_and_fp32_and_pi_is_strictly_inside_0_1(builder, prior_batch):
    ids, bel, logp = prior_batch
    cand, k = species_candidates(builder.species_valid, ids, bel)
    for dt, tol in ((torch.float64, 1e-12), (torch.float32, 1e-5)):
        p = fixed_mass_presence(logp.to(dt), cand, k)
        live = p.live
        assert int(live.sum()) == int((k > 0).sum())            # 0 < k < n on every k > 0 row
        resid = (p.pi.sum(-1) - k.to(dt))[live].abs().max().item()
        assert resid <= tol, (dt, resid)
        on = cand & live.unsqueeze(-1)
        assert float(p.pi[on].min()) > 0.0 and float(p.pi[on].max()) < 1.0
        assert bool(torch.isfinite(p.log_pi[on]).all())
    # fp32 and fp64 agree to the fp32 resolution (§3.2 measured 1.5e-7)
    p64 = fixed_mass_presence(logp.double(), cand, k)
    p32 = fixed_mass_presence(logp.float(), cand, k)
    assert float((p32.pi.double() - p64.pi).abs().max()) < 1e-6


def test_the_construction_matches_an_independent_fp64_root_find(builder, prior_batch):
    """Newton on Σσ(a + τ) − k from τ = 0, in numpy fp64 — no shared code with the bisection."""
    ids, bel, logp = prior_batch
    cand, k = species_candidates(builder.species_valid, ids, bel)
    p = fixed_mass_presence(logp.double(), cand, k)
    for b in range(0, 700, 37):
        kb = int(k[b])
        if kb == 0:
            continue
        a = logp[b].double().numpy()[cand[b].numpy()]
        tau = 0.0
        for _ in range(200):
            s = 1.0 / (1.0 + np.exp(-(a + tau)))
            f, df = s.sum() - kb, (s * (1 - s)).sum()
            tau -= f / df
        ref = 1.0 / (1.0 + np.exp(-(a + tau)))
        got = p.pi[b].numpy()[cand[b].numpy()]
        assert np.abs(got - ref).max() < 1e-12


def test_the_bracket_is_provable_sum_below_k_at_lo_and_above_at_hi(builder, prior_batch):
    ids, bel, logp = prior_batch
    cand, k = species_candidates(builder.species_valid, ids, bel)
    a = logp.double()
    n = cand.sum(-1)
    live = (k > 0) & (k < n)
    base = torch.log(k.double()) - torch.log((n - k).double())
    a_max = torch.where(cand, a, torch.full_like(a, -math.inf)).amax(-1)
    a_min = torch.where(cand, a, torch.full_like(a, math.inf)).amin(-1)
    for tau, ok in ((base - a_max, lambda s: s <= k.double() + 1e-12),
                    (base - a_min, lambda s: s >= k.double() - 1e-12)):
        s = torch.where(cand, torch.sigmoid(a + tau.unsqueeze(-1)), torch.zeros_like(a)).sum(-1)
        assert bool(ok(s)[live].all())


def test_the_bisection_needs_its_iterations(builder, prior_batch):
    """Teeth for the residual bound: 10 halvings of the bracket miss Σ = k by far more than 1e-5."""
    ids, bel, logp = prior_batch
    cand, k = species_candidates(builder.species_valid, ids, bel)
    tau = fixed_size_tau(logp.double(), cand, k, n_iter=10)
    s = torch.where(cand, torch.sigmoid(logp.double() + tau.unsqueeze(-1)), torch.zeros(())).sum(-1)
    assert float((s - k.double())[k > 0].abs().max()) > 1e-3
    assert BISECTION_ITERS == 64


def test_k_zero_is_structural_pi_zero_everything_masked(builder, layout):
    ids = torch.tensor([[1, 4, 7, 25, 6, 9]])
    bel = torch.zeros(1, 6, dtype=torch.bool)
    t0 = T0SpeciesPrior(layout["max_species"])
    logp = species_team_prior_logits(t0.species_prior_log_marginal, t0.species_prior_log_lift, ids, bel)
    emb = torch.nn.Embedding(layout["max_species"], layout["species_embedding_dim"])
    hs = builder(logp, ids, bel, torch.randn(1, 6, 128), torch.randn(1, 50), emb)
    p = hs.species
    assert int(p.k) == 0 and not bool(p.live) and not bool(p.full)
    assert float(p.pi.abs().max()) == 0.0
    assert bool((p.log_pi == MASKED_LOG_PRESENCE).all())
    assert not bool(hs.hyp_live.any()) and not bool(hs.other_live) and float(hs.other_mass) == 0.0
    assert float(hs.other_log_mass) == MASKED_LOG_PRESENCE
    assert float(hs.other_token.detach().abs().max()) == 0.0
    assert not bool(hs.slot_is_hypothesis.any()) and float(hs.slot_rows.abs().max()) == 0.0
    y = torch.zeros_like(logp)
    _, n_scored, _ = set_bce(p.logits, p, y)
    assert float(n_scored) == 0.0


def test_k_equal_n_is_structural_pi_one_and_excluded_from_the_bce():
    scores = torch.tensor([[0.3, -2.0, 5.0, 0.0], [0.3, -2.0, 5.0, 0.0]], dtype=torch.float64,
                          requires_grad=True)
    cand = torch.tensor([[True, True, False, False], [True, True, True, False]])
    k = torch.tensor([2, 2])
    p = fixed_mass_presence(scores, cand, k)
    assert p.full.tolist() == [True, False] and p.live.tolist() == [False, True]
    assert p.pi[0].tolist() == [1.0, 1.0, 0.0, 0.0] and p.log_pi[0, :2].tolist() == [0.0, 0.0]
    assert abs(float(p.pi[1].sum()) - 2.0) < 1e-12
    y = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]], dtype=torch.float64)
    loss, n, bad = set_bce(p.logits, p, y)
    assert float(n) == 1.0 and float(bad) == 0.0                 # only the live row is scored
    loss.backward()
    assert float(scores.grad[0].abs().max()) == 0.0               # the k = n row carries nothing
    assert bool(torch.isfinite(scores.grad).all())


def test_the_bce_is_finite_on_extreme_scores_in_fp32_and_its_gradient_too():
    g = torch.Generator().manual_seed(3)
    for spread in (1.0, 45.0, 200.0, 1e4):
        scores = (torch.randn(8, 386, generator=g) * spread).requires_grad_(True)
        cand = torch.ones(8, 386, dtype=torch.bool)
        k = torch.arange(1, 9).clamp(max=6)
        p = fixed_mass_presence(scores, cand, k)
        y = torch.zeros(8, 386)
        for b in range(8):
            y[b, torch.randperm(386, generator=g)[: int(k[b])]] = 1.0
        loss, n, _ = set_bce(p.logits, p, y)
        assert bool(torch.isfinite(loss)) and float(n) == 8.0
        loss.backward()
        assert bool(torch.isfinite(scores.grad).all())


def test_tau_needs_no_gradient_autograd_equals_the_exact_implicit_gradient():
    """With τ under no_grad, d(per-row BCE)/da = π − y. The EXACT gradient through τ(a) adds
    Σ(π − y) · dτ/da, which is zero iff the labels count k — so the two agree exactly then."""
    g = torch.Generator().manual_seed(11)
    a = (torch.randn(4, 50, generator=g, dtype=torch.float64) * 3).requires_grad_(True)
    cand = torch.ones(4, 50, dtype=torch.bool)
    k = torch.tensor([1, 2, 3, 6])
    p = fixed_mass_presence(a, cand, k)
    y = torch.zeros(4, 50, dtype=torch.float64)
    for b in range(4):
        y[b, torch.randperm(50, generator=g)[: int(k[b])]] = 1.0
    torch.nn.functional.binary_cross_entropy_with_logits(p.logits, y, reduction="sum").backward()
    pi = p.pi
    assert float((a.grad - (pi - y)).abs().max()) < 1e-12
    # the full implicit term vanishes: Σ_s (π_s − y_s) = k − k = 0 on every row
    assert float((pi - y).sum(-1).abs().max()) < 1e-12


def test_a_row_whose_labels_do_not_count_k_is_dropped_and_reported():
    scores = torch.zeros(2, 10, dtype=torch.float64)
    cand = torch.ones(2, 10, dtype=torch.bool)
    cand[1, 9] = False
    k = torch.tensor([3, 2])
    p = fixed_mass_presence(scores, cand, k)
    y = torch.zeros(2, 10, dtype=torch.float64)
    y[0, :2] = 1.0                   # 2 labels for k = 3 (an under-6 team): inconsistent
    y[1, 0] = y[1, 9] = 1.0          # a label outside V: inconsistent
    _, n, bad = set_bce(p.logits, p, y)
    assert float(n) == 0.0 and float(bad) == 2.0


# ----------------------------------------------------------------------------------- ordering
def test_the_ordering_is_stable_ties_go_to_the_lower_number():
    pi = torch.tensor([[0.2, 0.5, 0.2, 0.5, 0.1, 0.5, 0.2]])
    cand = torch.tensor([[True, True, True, True, True, False, True]])
    order = stable_order(pi, cand)
    assert order[0].tolist() == [1, 3, 0, 2, 6, 4, 5]     # 0.5s by num, then 0.2s by num; non-cand last


def test_ties_resolve_identically_under_a_batch_of_permuted_layouts():
    """Determinism: the order is a function of (π, num) alone — every all-tied row gives 1, 2, …"""
    pi = torch.full((5, 40), 0.25)
    cand = torch.ones(5, 40, dtype=torch.bool)
    cand[:, 0] = False
    order = stable_order(pi, cand)
    assert all(order[b, :39].tolist() == list(range(1, 40)) for b in range(5))


def test_near_tie_rows_flag_a_gap_under_eps_at_any_boundary(builder, layout):
    sorted_pi = torch.tensor([[0.9, 0.5, 0.5 - 5e-7, 0.1], [0.9, 0.5, 0.4, 0.1]])
    gap = boundary_gap(sorted_pi, torch.tensor([2, 2]), torch.tensor([4, 4]))
    assert bool(gap[0] < SELECTION_TIE_EPS) and not bool(gap[1] < SELECTION_TIE_EPS)
    assert math.isinf(float(boundary_gap(sorted_pi, torch.tensor([4, 0]), torch.tensor([4, 4]))[0]))


def test_the_hypotheses_are_the_top_k_and_fill_the_hidden_slots_in_slot_order(builder, layout):
    ids = torch.tensor([[0, 25, 0, 6, 0, 0]])
    bel = torch.tensor([[True, False, True, False, True, True]])
    t0 = T0SpeciesPrior(layout["max_species"])
    logp = species_team_prior_logits(t0.species_prior_log_marginal, t0.species_prior_log_lift, ids, bel)
    emb = torch.nn.Embedding(layout["max_species"], layout["species_embedding_dim"])
    hs = builder(logp, ids, bel, torch.zeros(1, 6, 128), torch.zeros(1, 50), emb)
    pi = hs.species.pi[0]
    top4 = hs.hyp_species[0, :4]
    assert hs.hyp_live[0].tolist() == [True] * 4 + [False] * 2
    assert float(pi[top4].min()) >= float(pi[hs.species.cand[0]].sort(descending=True).values[3])
    assert hs.slot_species[0].tolist() == [int(top4[0]), 0, int(top4[1]), 0, int(top4[2]), int(top4[3])]
    # the slot's row IS the dex table's row for that species; a revealed slot's row is zero
    assert torch.equal(hs.slot_rows[0, 0], builder.dex_rows[int(top4[0])])
    assert float(hs.slot_rows[0, 1].abs().max()) == 0.0
    # OTHER = the tail's mass, so hypotheses + OTHER = k exactly
    assert abs(float(hs.hyp_pi.sum() + hs.other_mass) - 4.0) < 1e-5
    lse = torch.logsumexp(hs.species.log_pi[0][hs.species.cand[0] & (hs.rank[0] >= 4)], 0)
    assert abs(float(lse.exp()) - float(hs.other_mass)) < 1e-5


# ----------------------------------------------------------------------------------- moves
def test_the_move_group_pins_revealed_first_and_sums_to_four_minus_r(builder):
    M = builder.n_moves
    sp = torch.tensor([248, 248, 248])                            # Tyranitar
    rev = torch.tensor([[0, 0, 0, 0], [89, 0, 0, 0], [89, 157, 242, 349]])
    logits = torch.zeros(3, M)
    mp = builder.move_group(logits, sp, rev)
    assert mp.r.tolist() == [0, 1, 4]
    pres = mp.presence
    assert abs(float(pres.pi[0].sum()) - 4.0) < 1e-5 and abs(float(pres.pi[1].sum()) - 3.0) < 1e-5
    assert int(pres.k[2]) == 0 and not bool(mp.other_live[2])        # all four revealed: OTHER masked
    assert mp.seat_nums[1, 0].item() == 89 and bool(mp.seat_revealed[1, 0])
    assert float(mp.seat_pi[1, 0]) == 1.0
    assert sorted(mp.seat_nums[2, :4].tolist()) == [89, 157, 242, 349]
    # never a candidate: the typeless HP channel, an illegal move, the revealed ones
    assert not bool(pres.cand[:, HIDDEN_POWER_MOVE_NUM].any())
    assert not bool(pres.cand[1, 89])
    assert bool((pres.cand <= builder.move_legal[248].unsqueeze(0)).all())
    # hypotheses-in-seats + OTHER_move = k_m on a live row
    unrev_seat = mp.seat_live[1] & ~mp.seat_revealed[1]
    assert abs(float(mp.seat_pi[1][unrev_seat].sum() + mp.other_mass[1]) - 3.0) < 1e-5


def test_a_revealed_hidden_power_takes_its_typed_channels_out_of_the_candidates(builder):
    legal = torch.ones(2, builder.n_moves, dtype=torch.bool)
    rev = torch.tensor([[HIDDEN_POWER_MOVE_NUM, 0, 0, 0], [89, 0, 0, 0]])
    cand, revealed, r = move_candidates(legal, builder.move_valid, rev)
    typed = list(TYPED_HP_NUMS)
    assert not bool(cand[0, typed].any())
    assert bool(cand[1, typed].all()) and not bool(cand[:, HIDDEN_POWER_MOVE_NUM].any())
    assert r.tolist() == [1, 1] and MOVE_GROUP_MASS == 4


# ----------------------------------------------------------------------------------- the learner
def _obs_from_golden(n: int = 64):
    from agents.training import learner_golden as LG
    with np.load(LG.BUFFER_PATH) as z:
        return {k[4:]: torch.as_tensor(z[k].reshape(-1, *z[k].shape[2:])[:n])
                for k in z.files if k.startswith("obs:")}


def _unperturbed_learner(args):
    """`testkit.fresh_model` MINUS its test perturbation (which draws noise per parameter in
    `named_parameters` order and so shifts every draw after an inserted group — a harness artefact,
    not the production build)."""
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    from agents.training.rust_rollout.build import trainee_spaces
    from agents.training.rust_vec_env import RustVecEnv
    from main.fresh_checkpoint import _production_policy_kwargs
    from utils.torch_state_guard import single_thread_build

    obs, act = trainee_spaces(args)
    env = RustVecEnv(n_envs=4, observation_space=obs, action_space=act, build=lambda m: None)
    _a, _l, pk = _production_policy_kwargs(args)
    torch.manual_seed(0)
    with single_thread_build():
        model = InstrumentedMaskablePPO(Gen3DualHeadMaskablePolicy, env, n_steps=16, batch_size=16,
                                        n_epochs=1, device="cpu", seed=0, policy_kwargs=pk, verbose=0)
    model.ep_info_buffer = deque(maxlen=100)
    return model


@pytest.fixture(scope="module")
def arms():
    from main.train.production_args import production_args
    blob = production_args()
    fm = production_args()
    fm.belief_tokens = "fixed_mass"
    return _unperturbed_learner(blob), _unperturbed_learner(fm)


def test_production_is_blob_and_builds_no_hypothesis_builder(arms):
    from main.train.production_args import production_args
    assert production_args().belief_tokens == "blob"
    m_blob, _ = arms
    fe = m_blob.policy.features_extractor
    assert fe.belief_tokens == "blob" and fe.hypothesis_builder is None
    assert not any("hypothesis_builder" in k for k in m_blob.policy.state_dict())
    assert BELIEF_TOKEN_MODES == ("blob", "fixed_mass")


def test_fixed_mass_leaves_every_non_x5_initial_byte_equal_to_blob(arms):
    m_blob, m_fm = arms
    sd0, sd1 = m_blob.policy.state_dict(), m_fm.policy.state_dict()
    extra = sorted(k for k in sd1 if k not in sd0)
    assert extra and all(".hypothesis_builder." in k for k in extra)
    # F-X5-27 (U3 part 3): fixed_mass does NOT build BeliefSlots (never called there) — the ONLY blob
    # key it lacks; its init draw still ran, so every other byte is unmoved (asserted next).
    missing = sorted(k for k in sd0 if k not in sd1)
    assert missing and all(".belief_slots." in k for k in missing), missing
    moved = [k for k in sd0 if k in sd1 and not torch.equal(sd0[k], sd1[k])]
    assert moved == [], moved
    # every non-X5 parameter keeps its optimizer position RELATIVE to the others
    n0 = [n for n, _ in m_blob.policy.named_parameters() if ".belief_slots." not in n]
    n1 = [n for n, _ in m_fm.policy.named_parameters() if ".hypothesis_builder." not in n]
    assert n0 == n1


def test_cold_start_presence_is_exactly_the_priors_fixed_size_marginal_on_a_real_policy(arms):
    """δ_θ's last layer is zero on a REAL MaskablePPO-built policy (IsolatedLinear: SB3's orthogonal
    re-init skips it), so π is the Smogon prior's fixed-size marginal bit-for-bit at step 0."""
    _, m_fm = arms
    fe = m_fm.policy.features_extractor
    hb = fe.hypothesis_builder
    assert float(hb.delta_out.weight.detach().abs().max()) == 0.0
    assert float(hb.delta_out.bias.detach().abs().max()) == 0.0
    obs = _obs_from_golden()
    with torch.no_grad():
        fe(obs)
    hs = fe.last_hypothesis
    ctx = fe.unpack(obs)
    opp = ctx.species_ids[:, 6:12]
    t0 = fe.t0_species_prior
    logp = species_team_prior_logits(t0.species_prior_log_marginal, t0.species_prior_log_lift, opp,
                                     ctx.opp_believed_mask)
    ref = fixed_mass_presence(logp, hs.species.cand, hs.species.k)
    assert torch.equal(ref.pi, hs.species.pi)


# (U2's "fixed_mass outputs == blob's with the shared weights" test is RETIRED by U3: the hypothesis
# set is now READ by the trunk, the pools and the T0 belief heads, so the arms differ by design. The
# blob arm's byte-identity is pinned by the K9 learner golden and `hypothesis_tokens_test`; the
# fixed_mass reads by `hypothesis_tokens_test` — the class-E invariances, the M10 gradient path.)


def _delta_grads(hb):
    return {n: p.grad for n, p in hb.named_parameters() if n.startswith("delta_")}


def test_no_policy_value_or_consumer_output_reaches_delta_theta_and_the_bce_does(arms):
    """M10: π is DETACHED into every policy / critic use, so δ_θ learns from the presence BCE alone.
    Teeth: compute π from `logits` instead of its detached copy and the consumer half fails."""
    _, m_fm = arms
    pol = copy.deepcopy(m_fm.policy)
    fe = pol.features_extractor
    hb = fe.hypothesis_builder
    with torch.no_grad():                       # a non-zero δ_θ, so a leak would show
        hb.delta_out.weight.normal_(0.0, 0.1)
    obs = _obs_from_golden()
    # (1) the PPO-side outputs: action logits + the value
    pol.zero_grad(set_to_none=True)
    logits = pol.get_distribution(obs).distribution.logits
    v = pol.predict_values(obs)
    (logits.float().nan_to_num(0.0).sum() + v.sum()).backward()
    assert all(g is None or float(g.abs().max()) == 0.0 for g in _delta_grads(hb).values())
    # (2) every consumer-facing tensor of the hypothesis set (what U3 / U4 will read)
    pol.zero_grad(set_to_none=True)
    fe(obs)
    hs = fe.last_hypothesis
    mp = hs.moves
    consumer = [hs.species.pi, hs.species.log_pi.clamp(min=-50), hs.hyp_pi, hs.slot_log_pi.clamp(min=-50),
                hs.other_mass, hs.other_log_mass.clamp(min=-50), hs.other_token, hs.other_tail_mean,
                hs.slot_rows, mp.seat_pi, mp.other_mass, mp.presence.pi]
    loss = sum(t.float().sum() for t in consumer if t.requires_grad)
    if torch.is_tensor(loss):
        loss.backward()
    assert all(g is None or float(g.abs().max()) == 0.0 for g in _delta_grads(hb).values())
    assert not hs.species.pi.requires_grad and not hs.other_mass.requires_grad
    # (3) the presence BCE DOES train δ_θ
    pol.zero_grad(set_to_none=True)
    fe(obs)
    hs = fe.last_hypothesis
    y = label_multi_hot(obs["belief_species"], hs.species.logits.shape[-1], hs.species.logits)
    loss, n, bad = set_bce(hs.species.logits, hs.species, y)
    assert float(n) > 0 and float(bad) == 0.0
    loss.backward()
    assert float(hb.delta_out.weight.grad.abs().max()) > 0.0


def test_the_set_supervision_row_trains_on_the_golden_buffer_and_matches_its_eager_wrapper(arms):
    from agents.training.belief_bank import hypothesis_set_loss
    from agents.training.belief_bank_static import hypothesis_set_terms
    _, m_fm = arms
    fe = copy.deepcopy(m_fm.policy).features_extractor
    obs = _obs_from_golden()
    fe(obs)
    hs, bl = fe.last_hypothesis, fe.last_belief_logits
    t = hypothesis_set_terms(hs, bl, obs["belief_species"], obs["belief_moves"], 1.0)
    assert bool(t.present) and bool(torch.isfinite(t.loss))
    assert float(t.metrics["presence_label_mismatch"][0]) == 0.0
    loss, mets = hypothesis_set_loss(hs, bl, obs["belief_species"], obs["belief_moves"], 1.0)
    assert float(loss) == float(t.loss) and set(mets) == set(t.metrics)
    # BeliefHead re-target: the team score is the hidden-slot mean of its per-slot logits
    team = belief_head_team_scores(bl["species"], hs.slot_is_hypothesis)
    b = 0 if bool(hs.slot_is_hypothesis[0].any()) else int(hs.slot_is_hypothesis.any(-1).nonzero()[0])
    m = hs.slot_is_hypothesis[b]
    assert torch.allclose(team[b], bl["species"][b][m].mean(0))


def test_near_tie_rows_on_real_states_are_reported_not_guessed(arms):
    _, m_fm = arms
    fe = copy.deepcopy(m_fm.policy).features_extractor
    obs = _obs_from_golden()
    with torch.no_grad():
        fe(obs)
    hs = fe.last_hypothesis
    nt = near_tie_rows(hs)
    assert nt.dtype == torch.bool and nt.shape == hs.species.k.shape
    # determinism: a second forward of the same rows gives the same selection on every non-tied row
    with torch.no_grad():
        fe(obs)
    hs2 = fe.last_hypothesis
    keep = ~nt
    assert torch.equal(hs.hyp_species[keep], hs2.hyp_species[keep])
    assert torch.equal(hs.moves.seat_nums[keep], hs2.moves.seat_nums[keep])


# ----------------------------------------------------------------------------------- the flag
def test_checkargs_reads_the_requires_graph_and_blob_needs_nothing():
    from main.checkargs import unsatisfiable_pairs
    bad = unsatisfiable_pairs(["--belief-tokens", "fixed_mass", "--opp-intent-coef", "0"])
    assert ("belief_tokens", "opp_intent", "--opp-intent-coef 0") in bad
    assert unsatisfiable_pairs(["--belief-tokens", "blob", "--opp-intent-coef", "0"]) == []


def test_the_extractor_refuses_fixed_mass_without_its_dependencies(layout):
    import gymnasium as gym
    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.observation.state_encoder import load_mappings
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    with pytest.raises(ValueError, match="belief_tokens must be one of"):
        Gen3FeaturesExtractor(space, layout=layout, mappings=load_mappings(), belief_tokens="bogus")
    with pytest.raises(ValueError, match="requires t0_species_prior"):
        Gen3FeaturesExtractor(space, layout=layout, mappings=load_mappings(), belief_tokens="fixed_mass")


def test_a_pre_v136_config_migrates_to_blob_and_a_mismatch_is_refused():
    import dataclasses
    from agents.model.model_version import ModelVersionError
    from agents.model.model_version.migrations import _migrate_config
    from agents.model.snapshot import current_model_version
    from agents.observation.state_encoder import load_mappings
    out = _migrate_config({"config_version": 135})
    assert out["belief_tokens"] == "blob" and out["config_version"] >= 136
    a = current_model_version(load_mappings())
    assert a.belief_tokens == "blob"
    b = dataclasses.replace(a, belief_tokens="fixed_mass")
    a.check_compatible(dataclasses.replace(a))
    with pytest.raises(ModelVersionError, match="belief_tokens mismatch"):
        b.check_compatible(a)


def test_the_fixed_mass_forward_satisfies_the_tier_contract_with_the_builder_at_T0(arms):
    from agents.model.tier_contract import assert_tier_contract, declared_tier
    _, m_fm = arms
    fe = copy.deepcopy(m_fm.policy).features_extractor
    obs = _obs_from_golden(16)
    assert declared_tier("hypothesis_builder") == 0
    tr = assert_tier_contract(fe, obs)
    assert any("hypothesis_builder" in str(e) for e in tr.order), tr.order   # instrumented, not skipped
