"""The meter's per-unit numbers on hand-built rows: every value below is computed by hand in the comment beside it,
so a change to which units count, to what a unit averages, or to a blend's formula fails here."""
from __future__ import annotations

import math

import numpy as np
import pytest

from main.offpool_belief import metrics as M

LN = math.log


def test_species_unit_is_the_mean_log_presence_of_the_true_hidden_species() -> None:
    # 8 species; V = {1, 2, 3, 5}. Row 0 hides species 2 and 5 (twice the same slot value must count once),
    # row 1 hides nothing, row 2 hides species 3 and species 7 (7 is outside V).
    cand = np.zeros((3, 8), dtype=bool)
    cand[:, [1, 2, 3, 5]] = True
    post = np.full((3, 8), -50.0)
    post[0, [1, 2, 3, 5]] = [LN(0.9), LN(0.6), LN(0.4), LN(0.1)]
    post[2, [1, 2, 3, 5]] = [LN(0.5), LN(0.45), LN(0.25), LN(0.75)]
    prior = np.full((3, 8), -50.0)
    prior[0, [1, 2, 3, 5]] = [LN(0.5), LN(0.5), LN(0.5), LN(0.5)]
    prior[2, [1, 2, 3, 5]] = [LN(0.2), LN(0.3), LN(0.8), LN(0.7)]
    truth = np.array([[2, 5, 5, -1, -1, -1], [-1] * 6, [3, 7, -1, -1, -1, -1]])
    u = M.species_units({M.POST: post, M.PRIOR: prior}, cand, truth)
    assert u.row.tolist() == [0, 2]                                  # row 1 has no hidden mon
    np.testing.assert_allclose(u.values["species_logp"][M.POST], [(LN(0.6) + LN(0.1)) / 2, LN(0.25)])
    np.testing.assert_allclose(u.values["species_logp"][M.PRIOR], [LN(0.5), LN(0.8)])
    # hit: row 0, k = 2: the posterior's top 2 are {1, 2} -> 1 of {2, 5}; the prior ties at 0.5 and the stable
    # order takes {1, 2} -> 1 of 2 as well. row 2, k = 2 (species 7 is a miss for everyone): the posterior's top 2
    # are {5, 1} -> 0 of {3, 7}; the prior's are {3, 5} -> 1 of 2.
    assert u.values["species_hit"][M.POST].tolist() == [0.5, 0.0]
    assert u.values["species_hit"][M.PRIOR].tolist() == [0.5, 0.5]
    assert u.excluded["species_off_support"].tolist() == [2]          # species 7, once
    assert u.excluded["species_hit_near_tie"].tolist() == [0]         # the prior's 0.5 / 0.5 boundary on row 0


def test_a_row_whose_every_hidden_species_is_off_support_is_not_a_unit() -> None:
    cand = np.zeros((1, 4), dtype=bool)
    cand[0, [1, 2]] = True
    lp = np.log(np.full((1, 4), 0.5))
    u = M.species_units({M.POST: lp}, cand, np.array([[3, 3, -1, -1, -1, -1]]))
    assert u.row.size == 0 and u.excluded["species_off_support"].tolist() == [0]


def test_a_non_finite_presence_on_a_true_candidate_raises() -> None:
    cand = np.ones((1, 3), dtype=bool)
    lp = np.array([[0.0, -np.inf, -1.0]])
    with pytest.raises(ValueError, match="not finite"):
        M.species_units({M.POST: lp}, cand, np.array([[1, -1, -1, -1, -1, -1]]))


def _moves_fixture():
    # 10 moves; slot 0 is a revealed mon of species 4 whose true moves are {2, 3, 7, 9}; 9 is Hidden Power's
    # family, 2 is already shown by the observation -> U = {3, 7}. Legal for the species: {1, 2, 3, 5, 7, 9}.
    # Slot 1 is hidden (known_moves pad). Slot 2 is revealed with every true move shown (no unit).
    # Slot 3 is revealed and its unshown true move 6 is NOT in the legal set (excluded).
    known = np.full((1, 6, 4), -1)
    known[0, 0] = [2, 3, 7, 9]
    known[0, 2] = [1, 5, -1, -1]
    known[0, 3] = [6, -1, -1, -1]
    ids = np.zeros((1, 6, 4), dtype=np.int64)
    flag = np.zeros((1, 6, 4))
    ids[0, 0, 0], flag[0, 0, 0] = 2, 1.0
    ids[0, 0, 1], flag[0, 0, 1] = 8, 0.0            # an id with the known flag down is not "shown"
    ids[0, 2, :2], flag[0, 2, :2] = [1, 5], 1.0
    legal = np.zeros((1, 6, 10), dtype=bool)
    legal[0, :, [1, 2, 3, 5, 7, 9]] = True
    species = np.array([[4, 0, 4, 4, 0, 0]])
    return known, ids, flag, legal, species


def test_move_unit_scores_only_the_unshown_non_hidden_power_truth() -> None:
    known, ids, flag, legal, species = _moves_fixture()
    post = np.zeros((1, 6, 10))
    post[0, 0, [1, 3, 5, 7]] = [2.0, 1.0, 0.0, -1.0]     # candidates {1, 3, 5, 7}: top 2 = {1, 3} -> 1 of {3, 7}
    prior = np.zeros((1, 6, 10))
    prior[0, 0, [1, 3, 5, 7]] = [-2.0, 0.5, -1.0, 1.5]   # top 2 = {7, 3} -> 2 of 2
    u = M.move_units({M.POST: post, M.PRIOR: prior}, legal, known, ids, flag, species, frozenset({9}))
    assert u.row.tolist() == [0]
    ls = lambda x: -math.log1p(math.exp(-x))             # noqa: E731 - log sigmoid
    np.testing.assert_allclose(u.values["moves_logp"][M.POST], [(ls(1.0) + ls(-1.0)) / 2])
    np.testing.assert_allclose(u.values["moves_logp"][M.PRIOR], [(ls(0.5) + ls(1.5)) / 2])
    assert u.values["moves_prec"][M.POST].tolist() == [0.5]
    assert u.values["moves_prec"][M.PRIOR].tolist() == [1.0]
    assert u.excluded["moves_truth_illegal"].tolist() == [0]          # slot 3's move 6
    assert u.excluded["moves_prec_near_tie"].size == 0


def test_item_unit_is_a_labelled_real_item_the_observation_does_not_state() -> None:
    logp = np.log(np.array([[[0.1, 0.6, 0.3]] * 6]))                   # [1, 6, 3]
    other = np.log(np.array([[[0.2, 0.3, 0.5]] * 6]))
    label = np.array([[2, 1, 0, 2, -1, 2]])        # slot 2 holds nothing (label 0), slot 4 is pad
    mask = np.array([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
    known = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])                 # slot 3's item is already stated
    u = M.item_units({M.POST: logp, M.PRIOR: other}, label, mask, known)
    assert u.row.tolist() == [0, 0]                                   # slots 0 and 1
    np.testing.assert_allclose(u.values["items_logp"][M.POST], [LN(0.3), LN(0.6)])
    np.testing.assert_allclose(u.values["items_logp"][M.PRIOR], [LN(0.5), LN(0.3)])
    assert u.values["items_top1"][M.POST].tolist() == [0.0, 1.0]
    assert u.values["items_top1"][M.PRIOR].tolist() == [1.0, 0.0]


def test_the_blends_are_the_stated_mixtures_and_carry_the_floor() -> None:
    post, prior = np.log(np.array([1e-9, 0.2, 0.9])), np.log(np.array([0.05, 0.4, 0.3]))
    got = M.mix_log_prob(post, prior, 0.25)
    np.testing.assert_allclose(np.exp(got), 0.75 * np.exp(post) + 0.25 * np.exp(prior))
    # the floor that makes a blend safe: never more than log(1/λ) below the prior, however wrong the posterior
    assert (got >= prior + LN(0.25) - 1e-12).all()
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))             # noqa: E731
    a, b = np.array([-30.0, 0.3, 4.0]), np.array([-1.0, 1.0, -2.0])
    np.testing.assert_allclose(sig(M.mix_bernoulli_logits(a, b, 0.25)), 0.75 * sig(a) + 0.25 * sig(b))
    for lam in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            M.mix_log_prob(post, prior, lam)
    assert M.mix_name(0.25) == "mix0.25"


def test_the_numpy_blends_are_the_models_own_helpers() -> None:
    """The scorer's closed-form blend must be the blend ``belief_prior_mix`` plays with."""
    import torch as th

    from agents.model import belief_prior_mix as BPM

    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(5, 7)) * 3, rng.normal(size=(5, 7)) * 3
    want = BPM.mix_bernoulli_logits(th.as_tensor(a), th.as_tensor(b), 0.25).numpy()
    np.testing.assert_allclose(M.mix_bernoulli_logits(a, b, 0.25), want, rtol=0, atol=1e-5)   # the helper is fp32
    lp, lq = a - np.logaddexp.reduce(a, axis=1, keepdims=True), b - np.logaddexp.reduce(b, axis=1, keepdims=True)
    want_c = th.log_softmax(BPM.mix_categorical_logits(th.as_tensor(a), th.as_tensor(b), 0.25), dim=-1).numpy()
    np.testing.assert_allclose(M.mix_log_prob(lp, lq, 0.25), want_c, rtol=0, atol=1e-5)
