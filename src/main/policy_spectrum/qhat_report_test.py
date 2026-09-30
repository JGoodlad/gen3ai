"""The X4 pre-read's readout: a Q̂ that IS the truth sees every starved near-best action and no
dominated one; a reversed Q̂ sees none; the rule regrets and the bootstrap behave."""

from __future__ import annotations

import numpy as np
import pytest

from main.policy_spectrum import bank as B
from main.policy_spectrum import qhat_report as R
from main.policy_spectrum.bank_test import BANK_V1


@pytest.fixture(scope="module")
def v1():
    return B.load_bank(BANK_V1)


def _world(v1, sign: float, n: int = 30):
    """n free turns (one per battle) with ≥ 3 legal actions: two near-best (+1 on every seed), the
    rest dominated (−1). The policy puts all but 0.5 % on the first near-best one, so the second is
    STARVED. Q̂ is ``sign`` × the truth, seed by seed."""
    truth, q, seen = [], {}, set()
    probs = np.zeros((len(v1.decisions), 11))
    for i, d in enumerate(v1.decisions):
        acts = sorted(int(a) for a in d["tokens"])
        if d["kind"] != "free" or d["battle"] in seen or len(acts) < 3:
            continue
        seen.add(d["battle"])
        out = {str(a): [1.0 if k < 2 else -1.0] * 64 for k, a in enumerate(acts)}
        truth.append({"id": d["id"], "ok": True, "seeds": ["s"] * 64, "outcomes": out})
        q[(d["id"], "R")] = {"id": d["id"], "variant": "R", "ok": True, "other_open": False, "actions": {
            a: {"q": [sign * x for x in v], "src": "V" * 64, "opp": "0" * 64, "sw": "0" * 64} for a, v in out.items()}}
        probs[i, acts] = 0.005 / (len(acts) - 1)
        probs[i, acts[0]] = 0.995
        if len(truth) == n:
            break
    return truth, q, probs


def _read(v1, sign):
    truth, q, probs = _world(v1, sign)
    turns = R.assemble(v1, truth, q, probs, "R", list(range(8)), list(range(8, 64)))
    return R.read(turns, R.Boot([T["battle"] for T in turns]))


def test_a_qhat_equal_to_the_truth_sees_every_starved_near_best_action(v1):
    r = _read(v1, +1.0)
    assert r["b_starved_near"]["qnear"]["mean"] == 1.0
    assert r["b_starved_near"]["top2"]["mean"] == 1.0
    assert r["b_dominated"]["qnear"]["mean"] == 0.0
    assert r["recall_top1_truth_best"]["mean"] == 1.0
    assert r["fp_gap_0.4_inf"]["top2"]["mean"] == 0.0
    assert r["regret"]["all"]["q_argmax"]["mean"] == 0.0
    assert r["regret"]["starved"]["q_top2_uniform"]["mean"] == 0.0


def test_a_reversed_qhat_is_blind(v1):
    r = _read(v1, -1.0)
    assert r["b_starved_near"]["qnear"]["mean"] == 0.0
    assert r["b_dominated"]["qnear"]["mean"] == 1.0
    assert r["regret"]["all"]["q_argmax"]["mean"] == 2.0          # it always picks a −1 action
    assert r["b_starved_minus_dominated_qnear"]["delta"] == -1.0


def test_rule_regrets_and_softmax_limits():
    T = {"id": "t", "acts": np.array([0, 1, 2]), "v": np.array([1.0, 0.0, -1.0]), "vstar": 1.0,
         "q": np.array([0.2, 0.1, 0.0]), "qj": np.array([0.2, 0.1, 0.0]), "pi": np.array([0.0, 1.0, 0.0])}
    rr = R.rule_regrets(T)
    assert rr["q_argmax"] == 0.0 and rr["pi_argmax"] == 1.0 and rr["pi_dist"] == 1.0
    assert rr["q_top2_uniform"] == pytest.approx(0.5) and rr["q_top3_uniform"] == pytest.approx(1.0)
    assert rr["q_softmax_0.01"] < 0.01 < rr["q_softmax_0.3"] < 1.0      # sharp → argmax, broad → uniform


def test_bootstrap_diff_is_joint():
    b = R.Boot([f"b{i}" for i in range(50)])
    cl = [f"b{i}" for i in range(50)]
    x = np.linspace(0, 1, 50)
    d = b.diff((x + 0.3, cl), (x, cl))
    assert d["delta"] == pytest.approx(0.3) and d["ci"][0] == pytest.approx(0.3) and d["ci"][1] == pytest.approx(0.3)
