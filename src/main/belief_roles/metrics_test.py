"""The purpose metrics on synthetic inputs: the intent MISS column is never floored, the common event
space (Hidden Power collapse, slot content), the presence Brier / OTHER reads by hand, rule 8."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from main.belief_roles import metrics as MX
from main.belief_roles.bank_rows import NO_EVENT, SWITCH_BASE, BankRows
from main.belief_roles.forward import Columns, _event_logp

S = 8          # a toy species axis


def test_event_logp_common_event_space():
    """Seats [10, 360 (a typed HP), 20, pad]; SWITCH. Slots: 0 revealed species 3, 1 hidden whose content
    puts 0.25 on species 5, 2 not a legal switch-in."""
    alpha = torch.tensor([[1.0, 2.0, 0.5, -math.inf, 0.0]]).repeat(5, 1)
    seats = torch.tensor([[10, 360, 20, 0]]).repeat(5, 1)
    beta = torch.tensor([[0.0, 1.0, 3.0, 0.0, 0.0, 0.0]]).repeat(5, 1)
    ok = torch.tensor([[True, True, False, False, False, False]]).repeat(5, 1)
    content = torch.full((5, 6, S), -math.inf, dtype=torch.float64)
    content[:, 0, 3] = 0.0
    content[:, 1, 5] = math.log(0.25)
    content[:, 1, 6] = math.log(0.75)
    events = torch.tensor([237, 20, 30, SWITCH_BASE + 5, SWITCH_BASE + 7])
    lp, cov = _event_logp(events, alpha, seats, beta, ok, content)
    la = torch.log_softmax(alpha[0].double(), -1)
    lb = torch.log_softmax(torch.tensor([0.0, 1.0]).double(), -1)
    assert cov.tolist() == [True, True, False, True, False]
    assert abs(lp[0] - la[1]) < 1e-12                         # HP event ← the typed-HP seat (360)
    assert abs(lp[1] - la[2]) < 1e-12
    assert abs(lp[3] - (la[4] + lb[1] + math.log(0.25))) < 1e-12   # α_SW · β_1 · c_1(5)
    assert lp[2] == -math.inf and lp[4] == -math.inf           # a MISS is −inf, never a floor


def _toy(intent_logp, covered, event, k=None, read_tie=None):
    n = len(event)
    k = np.ones(n, dtype=np.int64) if k is None else np.asarray(k)
    zeros = np.zeros((n, S))
    cand = np.ones((n, S), dtype=bool)
    cand[:, 0] = False
    br = BankRows(bank=None, rows=np.zeros((n, 1), np.float32), masks=np.zeros((n, 11), bool), gate={},
                  battle_index=np.arange(n), opp_class=["bot"] * n, on_pool=np.ones(n, bool),
                  true_species=np.zeros((n, 6), np.int64), true_moves=[[set()] * 6] * n,
                  event=np.asarray(event, np.int64))
    cols = Columns(arm="blob", k=k, cand=cand, revealed_nums=np.zeros((n, 6), np.int64), pi_arm=zeros,
                   pi_prior=zeros, sel_tie_arm=np.zeros(n, bool), sel_tie_prior=np.zeros(n, bool),
                   read_tie_arm=np.zeros(n, bool) if read_tie is None else np.asarray(read_tie),
                   other_mass_model=None, intent_logp=np.asarray(intent_logp, np.float64),
                   opp_active_species=np.zeros(n, np.int64),
                   intent_covered=np.asarray(covered), role_M_arm=zeros[:, :0], role_V_arm=zeros[:, :0],
                   role_M_prior=zeros[:, :0], role_V_prior=zeros[:, :0])
    return br, cols


def test_the_miss_column_is_never_floored_into_the_log_loss():
    """Two covered rows (P = 0.5, 0.25), one MISS, one unlabelled, one rule-8 row: the log loss is the
    mean over the two covered rows ONLY and the miss is its own column — no 1e-8 floor anywhere."""
    lp = [math.log(0.5), math.log(0.25), math.nan, math.nan, math.log(0.9)]
    br, cols = _toy(lp, [True, True, False, False, True], [10, 11, 12, NO_EVENT, 13],
                    read_tie=[False, False, False, False, True])
    r = MX.intent_read(br, cols, np.ones(5, bool))["all"]
    assert r["n_labeled"] == 3 and r["n_covered"] == 2 and r["n_miss"] == 1
    assert abs(r["miss_rate"] - 1 / 3) < 1e-12
    assert abs(r["logloss"] - (math.log(2) + math.log(4)) / 2) < 1e-12
    assert MX.intent_read(br, cols, np.ones(5, bool))["excluded_rule8"] == 1


def test_a_struggle_label_is_excluded_not_missed():
    s = MX.struggle_num()
    br, cols = _toy([math.log(0.5), math.nan], [True, False], [10, s])
    r = MX.intent_read(br, cols, np.ones(2, bool))
    assert r["all"]["n_labeled"] == 1 and r["all"]["n_miss"] == 0 and r["excluded_struggle"] == 1


def test_presence_brier_and_other_by_hand():
    # one row, k = 2: π over species 1..7 (0 is not a candidate), truth {1, 4}
    pi = np.array([[0.0, 0.7, 0.6, 0.3, 0.2, 0.1, 0.06, 0.04]])
    y = np.zeros((1, S), bool)
    y[0, [1, 4]] = True
    cand = np.ones((1, S), bool)
    cand[0, 0] = False
    k = np.array([2])
    pr = MX.presence_read(pi, y, cand, k, np.zeros(1), np.ones(1, bool))
    want = 0.3 ** 2 + 0.6 ** 2 + 0.3 ** 2 + 0.8 ** 2 + 0.1 ** 2 + 0.06 ** 2 + 0.04 ** 2
    assert abs(pr["brier"] - want) < 1e-12 and abs(pr["brier_per_mon"] - want / 2) < 1e-12
    # OTHER: the top-2 are species 1, 2; the tail {3..7} holds mass 0.7 and one true species (4)
    o = MX.other_read(pi, y, cand, k, np.zeros(1, bool), np.zeros(1), np.ones(1, bool))
    assert abs(o["mean_mass"] - 0.7) < 1e-12 and o["mean_realised"] == 1.0
    assert abs(o["mean_err"] + 0.3) < 1e-12
    # the model's own OTHER mass must agree with the read's, or the read refuses
    with pytest.raises(AssertionError, match="disagrees"):
        MX.other_read(pi, y, cand, k, np.zeros(1, bool), np.zeros(1), np.ones(1, bool),
                      model_mass=np.array([0.5]))
    # rule 8: a near-tie row is excluded and counted
    o2 = MX.other_read(pi, y, cand, k, np.ones(1, bool), np.zeros(1), np.ones(1, bool))
    assert o2["n"] == 0 and o2["excluded_rule8"] == 1


def test_stable_rank_ties_to_the_lower_num():
    pi = np.array([[0.0, 0.5, 0.5, 0.2]])
    cand = np.array([[False, True, True, True]])
    assert MX.stable_rank(pi, cand).tolist() == [[3, 0, 1, 2]]


def test_cluster_ci_is_deterministic():
    v = np.arange(40, dtype=np.float64)
    c = np.repeat(np.arange(10), 4)
    assert MX.cluster_ci(v, c) == MX.cluster_ci(v, c)


def test_flat_pointer_event_logp():
    """U4's flat list for K = 2: [seat 10, seat 360 (typed HP) · OTHER_move · six slots · OTHER_species].
    OTHER_move's members {20, 30, 357 (typed HP)} carry π 0.2 / 0.6 / 0.2; slot 0 holds revealed species 3,
    slot 1 the hypothesis 4; OTHER_species' tail is {5: 0.25, 6: 0.75}."""
    from types import SimpleNamespace

    from main.belief_roles.forward import _flat_event_logp

    B, K, M, S2 = 7, 2, 400, 10
    logits = torch.tensor([[0.5, 1.0, 0.0, 2.0, 1.0, -math.inf, -math.inf, -math.inf, -math.inf, 0.3]])
    logits = logits.repeat(B, 1)
    live = torch.isfinite(logits)
    cand_ids = torch.tensor([[10, 360, 0, 3, 4, 0, 0, 0, 0, 0]]).repeat(B, 1)
    beyond = torch.zeros(B, M, dtype=torch.bool)
    pi_m = torch.zeros(B, M, dtype=torch.float64)
    for n, p in ((20, 0.2), (30, 0.6), (357, 0.2)):
        beyond[:, n] = True
        pi_m[:, n] = p
    in_tail = torch.zeros(B, S2, dtype=torch.bool)
    in_tail[:, [5, 6]] = True
    p_tail = torch.zeros(B, S2, dtype=torch.float64)
    p_tail[:, 5], p_tail[:, 6] = 0.25, 0.75
    fi = SimpleNamespace(k=K, live=live, cand_ids=cand_ids, beyond=beyond, in_tail=in_tail)
    ev = torch.tensor([237, 30, 99, SWITCH_BASE + 3, SWITCH_BASE + 5, SWITCH_BASE + 7, 10])
    lp, cov = _flat_event_logp(ev, logits, fi, pi_m, p_tail)
    lf = torch.log_softmax(logits[0].double(), -1)
    assert cov.tolist() == [True, True, False, True, True, False, True]
    # HP: the typed seat AND OTHER_move's typed member (0.2 of its 1.0 mass) — both name the one event
    assert abs(lp[0] - torch.logaddexp(lf[1], lf[2] + math.log(0.2))) < 1e-12
    assert abs(lp[1] - (lf[2] + math.log(0.6))) < 1e-12
    assert abs(lp[3] - lf[3]) < 1e-12
    assert abs(lp[4] - (lf[9] + math.log(0.25))) < 1e-12
    assert abs(lp[6] - lf[0]) < 1e-12
    assert lp[2] == -math.inf and lp[5] == -math.inf          # misses: never floored
