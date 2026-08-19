"""Unit tests for the species-clause exclusivity operator.

Every case here is EXACT arithmetic on a hand-built array — no model, no torch, no battle — because
the operator's whole value is that its output is a checkable function of its input. The two cases
that matter most are the ones that would silently rot: identity-when-already-coherent (an adjusted
view that differs from the raw view on a legal belief is worse than no view) and the measured
gen-15 three-Salamence case that motivated the module.
"""
from __future__ import annotations

import numpy as np
import pytest

from agents.inference.species_exclusivity import (
    ExclusivityInfo,
    coherent_team_hypothesis,
    exclusive_team_posterior,
    exclusive_team_posterior_info,
    expected_counts,
    illegal_mass,
    revealed_leak,
)


# --------------------------------------------------------------------------------------------
# The three invariants the operator exists to establish
# --------------------------------------------------------------------------------------------

def _assert_rows_are_distributions(p: np.ndarray) -> None:
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-12), p.sum(axis=1)
    assert np.all(p >= -1e-15)


def test_a_coherent_belief_is_returned_UNCHANGED():
    """Identity-when-already-coherent. Two hidden slots, disjoint confident guesses, nothing
    revealed — no constraint binds, so the operator must be a no-op to the last bit."""
    raw = np.array([[0.7, 0.2, 0.1, 0.0],
                    [0.1, 0.1, 0.8, 0.0]])
    out, info = exclusive_team_posterior_info(raw, revealed_species=())
    assert np.array_equal(out, raw)              # EXACT, not approximate
    assert info.iterations == 0
    assert info.converged
    assert info.illegal_mass_before == 0.0
    assert np.all(info.total_variation == 0.0)


def test_the_single_hidden_slot_case_can_never_be_incoherent():
    """One row summing to 1 has every column sum ≤ 1 by construction, so H=1 is always an identity
    (modulo revealed masking). Worth pinning because it is the LATE-GAME case — most decisions in a
    finished battle have one hidden slot, and an operator that perturbed them would add noise to the
    majority of the data for no gain."""
    raw = np.array([[0.99, 0.005, 0.005, 0.0]])
    out = exclusive_team_posterior(raw, revealed_species=())
    assert np.array_equal(out, raw)


def test_revealed_species_are_zeroed_and_the_row_renormalizes():
    """(a) zero mass on a revealed species, with the freed mass redistributed PROPORTIONALLY over
    what remains — the minimum-KL completion, not a flat spread."""
    raw = np.array([[0.5, 0.3, 0.2, 0.0]])
    out = exclusive_team_posterior(raw, revealed_species=[0])
    assert out[0, 0] == 0.0
    # 0.3 and 0.2 renormalize over 0.5 → 0.6 / 0.4, exactly.
    assert out[0, 1] == pytest.approx(0.6, abs=1e-12)
    assert out[0, 2] == pytest.approx(0.4, abs=1e-12)
    _assert_rows_are_distributions(out)


def test_the_symmetric_two_slot_case_has_an_exact_fixed_point():
    """Two hidden slots, identical beliefs, two species. E[count(A)] = 1.6 — illegal. By symmetry
    the unique feasible point with both column sums ≤ 1 and both rows summing to 1 is the uniform
    ½/½, and the iteration must land on it exactly."""
    raw = np.array([[0.8, 0.2],
                    [0.8, 0.2]])
    assert illegal_mass(raw) == pytest.approx(0.6)
    out, info = exclusive_team_posterior_info(raw)
    assert np.allclose(out, np.array([[0.5, 0.5], [0.5, 0.5]]), atol=1e-9)
    assert info.converged
    assert info.max_expected_count_before == pytest.approx(1.6)
    assert info.max_expected_count_after == pytest.approx(1.0, abs=1e-9)
    assert info.illegal_mass_after == pytest.approx(0.0, abs=1e-9)


def test_the_measured_gen15_three_salamence_case():
    """The case that motivated the module: three hidden slots reading P(Salamence) = 0.39 / 0.60 /
    0.39, an expected count of 1.38 on a team the clause caps at 1. After the operator the peak must
    sit at 1, the rows must still be distributions, and — the property a naive 'just cap it' rule
    would break — the slot that was MOST confident must stay the most confident."""
    # Species axis: 0 = salamence, 1..3 = three alternatives soaking up the rest of each row.
    raw = np.array([[0.39, 0.31, 0.20, 0.10],
                    [0.60, 0.20, 0.15, 0.05],
                    [0.39, 0.11, 0.30, 0.20]])
    assert expected_counts(raw)[0] == pytest.approx(1.38)
    assert illegal_mass(raw) == pytest.approx(0.38)

    out, info = exclusive_team_posterior_info(raw)
    _assert_rows_are_distributions(out)
    assert info.converged
    assert out.sum(axis=0).max() <= 1.0 + 1e-9
    assert info.illegal_mass_after == pytest.approx(0.0, abs=1e-9)
    # Every slot's Salamence mass falls (the constraint only ever removes from the over-full column)…
    assert np.all(out[:, 0] < raw[:, 0])
    # …and the ORDERING across slots is preserved: the 0.60 slot stays the leading claimant.
    assert out[1, 0] > out[0, 0] and out[1, 0] > out[2, 0]
    # The two 0.39 slots were symmetric and must stay symmetric.
    assert out[0, 0] == pytest.approx(out[2, 0], abs=1e-12)


def test_leak_and_incoherence_are_fixed_TOGETHER():
    """Both defects at once: mass on a revealed species AND an over-full hidden column. The result
    must satisfy all three constraints simultaneously — fixing one at the cost of the other is the
    failure a two-pass implementation invites."""
    raw = np.array([[0.40, 0.45, 0.10, 0.05],
                    [0.35, 0.50, 0.10, 0.05]])
    out, info = exclusive_team_posterior_info(raw, revealed_species=[0])
    assert np.all(out[:, 0] == 0.0)
    _assert_rows_are_distributions(out)
    assert out.sum(axis=0).max() <= 1.0 + 1e-9
    assert info.revealed_leak_before == pytest.approx([0.40, 0.35])


def test_the_column_cap_is_never_exceeded_on_random_beliefs():
    """A property check over 200 random softmax-shaped beliefs — the invariant must hold for inputs
    nobody hand-picked, at every hidden-slot count the game produces."""
    rng = np.random.default_rng(20260818)
    for _ in range(200):
        h = int(rng.integers(1, 6))
        n_revealed = int(rng.integers(0, 3))
        # ⚠️ `s` must leave at least one allowed species PER hidden slot or the feasible set is
        # empty by Hall's condition and the operator correctly fails to converge. The real axis is
        # ~400 species against ≤5 hidden slots, so that regime is unreachable in production — the
        # generator has to respect it or it tests infeasibility, not convergence.
        s = int(rng.integers(h + n_revealed + 2, 40))
        logits = rng.normal(0.0, 3.0, size=(h, s))
        raw = np.exp(logits - logits.max(axis=1, keepdims=True))
        raw /= raw.sum(axis=1, keepdims=True)
        revealed = rng.choice(s, size=n_revealed, replace=False)
        out, info = exclusive_team_posterior_info(raw, revealed_species=revealed)
        assert info.converged, info
        _assert_rows_are_distributions(out)
        assert out.sum(axis=0).max() <= 1.0 + 1e-8
        if len(revealed):
            assert np.all(out[:, revealed] == 0.0)


def test_convergence_is_bounded_and_reported_on_an_INFEASIBLE_input():
    """Two slots each pinned at probability 1.0 on the SAME species, exact zeros elsewhere: the
    feasible set is EMPTY (two rows must sum to 1 with all their mass in one column capped at 1).
    The operator must terminate and SAY it did not converge — looping, or returning a point that
    quietly satisfies neither constraint, are both worse than an honest flag."""
    raw = np.array([[1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]])
    out, info = exclusive_team_posterior_info(raw, max_iter=25)
    assert not info.converged
    assert info.iterations == 25
    _assert_rows_are_distributions(out)   # (c) still holds; only the column cap is unreachable


def test_an_empty_hidden_set_is_handled():
    out, info = exclusive_team_posterior_info(np.zeros((0, 5)))
    assert out.shape == (0, 5)
    assert info.converged and info.iterations == 0


def test_malformed_input_raises_rather_than_reshaping():
    with pytest.raises(ValueError, match=r"\[H, S\]"):
        exclusive_team_posterior(np.zeros(5))
    with pytest.raises(ValueError, match="non-finite"):
        exclusive_team_posterior(np.array([[np.nan, 1.0]]))
    with pytest.raises(ValueError, match="negative"):
        exclusive_team_posterior(np.array([[-0.1, 1.1]]))


def test_an_out_of_range_revealed_id_is_dropped_not_clamped():
    """Clamping a stray id to 0 would zero the UNKNOWN-sentinel column and silently corrupt an
    innocent species; dropping it merely omits a constraint we could not apply."""
    raw = np.array([[0.5, 0.5]])
    out = exclusive_team_posterior(raw, revealed_species=[999, -3])
    assert np.array_equal(out, raw)


# --------------------------------------------------------------------------------------------
# The diagnostics helpers
# --------------------------------------------------------------------------------------------

def test_expected_counts_sum_to_the_hidden_slot_COUNT():
    """Σ_s E[count(s)] == H exactly — the vector redistributes a fixed budget, so only its PEAK can
    be a violation. Pinned because a reader who forgets this will mistake a large total for a large
    violation."""
    raw = np.array([[0.5, 0.3, 0.2],
                    [0.1, 0.6, 0.3],
                    [0.2, 0.2, 0.6]])
    assert expected_counts(raw).sum() == pytest.approx(3.0)


def test_illegal_mass_is_bounded_by_h_minus_one():
    raw = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert illegal_mass(raw) == pytest.approx(2.0)          # H − 1


def test_revealed_leak_is_zero_with_nothing_revealed():
    raw = np.array([[0.5, 0.5], [0.25, 0.75]])
    assert np.all(revealed_leak(raw, None) == 0.0)
    assert np.all(revealed_leak(raw, ()) == 0.0)


# --------------------------------------------------------------------------------------------
# The point hypothesis
# --------------------------------------------------------------------------------------------

def test_the_point_hypothesis_never_names_a_species_twice():
    """The whole job: three slots whose raw top-1 is the SAME mon must resolve to three DIFFERENT
    mons, with the most confident slot keeping the contested one."""
    names = {0: "salamence", 1: "tyranitar", 2: "metagross", 3: "blissey"}
    raw = np.array([[0.39, 0.31, 0.20, 0.10],
                    [0.60, 0.20, 0.15, 0.05],
                    [0.39, 0.11, 0.30, 0.20]])
    hyp = coherent_team_hypothesis(raw, revealed_species=(), num_to_name=names)
    assert [h["slot"] for h in hyp] == [0, 1, 2]
    assert len({h["species"] for h in hyp}) == 3
    by_slot = {h["slot"]: h for h in hyp}
    assert by_slot[1]["species"] == "salamence"          # the 0.60 claimant wins the contest
    assert by_slot[1]["differs"] is False
    # The two 0.39 slots both wanted salamence and both had to move.
    assert by_slot[0]["raw_top1"] == "salamence" and by_slot[0]["differs"] is True
    assert by_slot[2]["raw_top1"] == "salamence" and by_slot[2]["differs"] is True


def test_the_point_hypothesis_is_the_plain_argmax_when_there_is_no_conflict():
    """No duplicate top-1 ⇒ every slot keeps its own argmax and `differs` is False everywhere, so a
    surface that draws only the disagreements draws nothing."""
    names = {0: "a", 1: "b", 2: "c"}
    raw = np.array([[0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1],
                    [0.1, 0.1, 0.8]])
    hyp = coherent_team_hypothesis(raw, num_to_name=names)
    assert [h["species"] for h in hyp] == ["a", "b", "c"]
    assert not any(h["differs"] for h in hyp)


def test_the_point_hypothesis_excludes_revealed_species():
    names = {0: "starmie", 1: "tyranitar", 2: "metagross"}
    raw = np.array([[0.9, 0.06, 0.04]])
    hyp = coherent_team_hypothesis(raw, revealed_species=[0], num_to_name=names)
    assert hyp[0]["species"] == "tyranitar"
    assert hyp[0]["raw_top1"] == "tyranitar"   # raw_top1 is over the ALLOWED set, so it agrees
    assert hyp[0]["differs"] is False


def test_the_point_hypothesis_reports_real_slot_ids():
    """Rows are hidden slots only, so row 0 is rarely team slot 0 — a surface labelling the row
    index would name the wrong mon."""
    names = {0: "a", 1: "b"}
    raw = np.array([[0.9, 0.1], [0.2, 0.8]])
    hyp = coherent_team_hypothesis(raw, num_to_name=names, slot_ids=[3, 5])
    assert [h["slot"] for h in hyp] == [3, 5]
    with pytest.raises(ValueError, match="slot_ids"):
        coherent_team_hypothesis(raw, slot_ids=[3])


def test_the_point_hypothesis_falls_back_to_num_labels_without_a_name_map():
    hyp = coherent_team_hypothesis(np.array([[0.6, 0.4]]))
    assert hyp[0]["species"] == "num_0"


def test_info_total_variation_measures_how_far_each_row_MOVED():
    """The per-row TV distance is what a surface uses to decide whether the adjustment is worth
    drawing at all — a 0.001 move is noise, a 0.3 move is the story."""
    raw = np.array([[0.8, 0.2], [0.8, 0.2]])
    _, info = exclusive_team_posterior_info(raw)
    assert isinstance(info, ExclusivityInfo)
    assert np.allclose(info.total_variation, [0.3, 0.3], atol=1e-9)
