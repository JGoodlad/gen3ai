"""Teeth for the by-reveal-count / belief-dose reads (synthetic data: no archive, no model).

    python -I -m pytest --noconftest -q belief_dose_test.py     (from this folder)
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import belief_dose as BD  # noqa: E402
import stage_facts as SF  # noqa: E402


def _bundle(seed=0, A=6):
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(A))
    T = rng.random(A)
    return {"order": order, "T": T, "WT": rng.random((A, 16)), "X4T": rng.random((A, 16)), "critT": rng.random(A),
            "critTs": rng.random((A, 8)),
            "arms": {"A": {"W": rng.random((A, 8)), "X4": rng.random((A, 8)), "crit": rng.random((A, 8))}}}


def test_imp_of_ties_go_to_the_policy_first_and_the_max_wins():
    Tp = np.array([0.5, 0.9, 0.1])
    assert BD.imp_of(np.array([1.0, 1.0, 0.0]), Tp) == 0.0                    # a tie: the policy's own (index 0)
    assert abs(BD.imp_of(np.array([0.2, 0.9, 0.1]), Tp) - 0.4) < 1e-12        # a clear winner: T[1] - T[0]
    assert BD.imp_of(np.array([1.0 + 5e-7, 1.0, 0.0]), Tp) == 0.0             # inside the 1e-6 tie band


def test_m8_endpoints_are_the_plain_honest_and_true_world_scores():
    b = _bundle(1)
    ordr = np.array(b["order"])
    Tp = b["T"][ordr]
    c = BD.state_curves(b)
    honest = b["arms"]["A"]["W"][ordr].mean(1)
    truth = b["WT"][ordr][:, :8].mean(1)
    assert abs(c["M8|LEAF-K8"][0] - BD.imp_of(honest, Tp)) < 1e-12            # PF[0] = 0
    assert abs(c["M8|LEAF-K8"][-1] - BD.imp_of(truth, Tp)) < 1e-12            # PF[-1] = 1


def test_m5_and_its_placebo_agree_at_p_zero_and_differ_at_p_one():
    seen_diff = 0.0
    for s in range(20):
        c = BD.state_curves(_bundle(s))
        assert abs(c["M5|LEAF-K8"][0] - c["M5W|LEAF-K8"][0]) < 1e-12           # no weight on the special world: identical
        seen_diff += abs(c["M5|LEAF-K8"][-1] - c["M5W|LEAF-K8"][-1])
    assert seen_diff > 1e-6                                                    # p = 1: the true world vs a random honest one


def test_persistent_first_and_equivalent_p():
    assert BD.persistent_first(np.array([False, True, False, True, True])) == 3
    assert BD.persistent_first(np.array([True, True, True])) == 0
    assert BD.persistent_first(np.array([True, False])) is None
    grid = np.array([0.0, 0.5, 1.0])
    curve = np.array([0.0, 0.1, 0.3])
    vec = np.tile(curve, (5, 1))
    assert abs(BD.equivalent_p(curve, vec, 0.05, np.full(5, 0.05), grid)["p_eq"] - 0.25) < 1e-12
    assert BD.equivalent_p(curve, vec, -0.1, np.full(5, -0.1), grid)["p_eq"] == 0.0     # at or below p = 0: the floor
    assert BD.equivalent_p(curve, vec, 0.9, np.full(5, 0.9), grid)["p_eq"] is None      # above the curve's p = 1 value


def test_hp_field_parser():
    assert SF.frac("331/331 par") == 1.0
    assert abs(SF.frac("45/100") - 0.45) < 1e-12
    assert SF.frac("0 fnt") == 0.0
    assert SF.frac("") is None and SF.frac("garbage") is None
