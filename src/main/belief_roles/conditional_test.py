"""Amendment 3(b)'s purpose metric (1) — the CONDITIONAL log loss over blob's named set E_row — and its
coverage read, on synthetic inputs: a planted OVER-CERTAIN arm cannot win it, the renormalisation and the
coverage calibration by hand, rule 8 at the denominator, and the dense event distributions behind them."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from main.belief_roles import metrics as MX
from main.belief_roles.bank_rows import SWITCH_BASE, BankRows
from main.belief_roles.eset import ERow, blob_event_dist, event_index, flat_event_dist, set_mask
from main.belief_roles.forward import Columns, _event_logp, _flat_event_logp

S = 8


def _cols(arm, events, p_event, eset_mass, eset_in, intent_cov=None, intent_lp=None, tie=None):
    """A bank of ``len(events)`` rows with the per-row quantities the intent reads take."""
    n = len(events)
    ev = np.asarray(events, np.int64)
    with np.errstate(divide="ignore"):
        lp_ev = np.log(np.asarray(p_event, np.float64))
    cov = np.asarray(intent_cov if intent_cov is not None else np.isfinite(lp_ev))
    ilp = np.asarray(intent_lp if intent_lp is not None else np.where(cov, lp_ev, np.nan), np.float64)
    zeros = np.zeros((n, S))
    br = BankRows(bank=None, rows=np.zeros((n, 1), np.float32), masks=np.zeros((n, 11), bool), gate={},
                  battle_index=np.arange(n) // 4, opp_class=["bot"] * n, on_pool=np.ones(n, bool),
                  true_species=np.zeros((n, 6), np.int64), true_moves=[[set()] * 6] * n, event=ev)
    cols = Columns(arm=arm, k=np.ones(n, np.int64), cand=np.ones((n, S), bool),
                   revealed_nums=np.zeros((n, 6), np.int64), pi_arm=zeros, pi_prior=zeros,
                   sel_tie_arm=np.zeros(n, bool), sel_tie_prior=np.zeros(n, bool),
                   read_tie_arm=np.zeros(n, bool), other_mass_model=None, intent_logp=ilp,
                   intent_covered=cov, opp_active_species=np.zeros(n, np.int64),
                   role_M_arm=zeros[:, :0], role_V_arm=zeros[:, :0], role_M_prior=zeros[:, :0],
                   role_V_prior=zeros[:, :0],
                   eset_mass=np.asarray(eset_mass, np.float64), eset_in=np.asarray(eset_in, bool),
                   eset_tie=np.zeros(n, bool) if tie is None else np.asarray(tie, bool),
                   eset_logp_event=lp_ev)
    return br, cols


def _planted(q_in, q_out, counts, over_certain):
    """Rows realising events {10, 11} (in E_row) and 12 (outside) with the given integer counts. The
    HONEST arm predicts the true q = (q_in…, q_out); the OVER-CERTAIN arm puts ALL its mass on E_row in
    the same ratio (equal discrimination), so it misses every outside row."""
    events = [10] * counts[0] + [11] * counts[1] + [12] * counts[2]
    in_set = [e != 12 for e in events]
    z = sum(q_in)
    if over_certain:
        p = {10: q_in[0] / z, 11: q_in[1] / z, 12: 0.0}
        mass = 1.0
    else:
        p = {10: q_in[0], 11: q_in[1], 12: q_out}
        mass = z
    return _cols("blob" if over_certain else "fixed_mass", events, [p[e] for e in events],
                 [mass] * len(events), in_set)


def test_a_planted_over_certain_arm_cannot_win_metric_1():
    """THE brief's test. The truth is q = (0.6, 0.2 | outside 0.2), realised exactly (60 / 20 / 20 rows).
    An over-certain arm (all mass on its named set, the same 3:1 ratio inside it) and an honestly
    calibrated arm of equal discrimination read EQUAL on the conditional metric — the over-certainty buys
    nothing. The as-built metric (each arm on its own covered rows, unrenormalised) hands the over-certain
    arm a strict win: that is the bias §7.7(b) removes. Fails if the renormalisation or the shared row
    set is reverted."""
    honest = _planted((0.6, 0.2), 0.2, (60, 20, 20), over_certain=False)
    over = _planted((0.6, 0.2), 0.2, (60, 20, 20), over_certain=True)
    c_h = MX.intent_conditional_read(*honest, np.ones(100, bool))["all"]
    c_o = MX.intent_conditional_read(*over, np.ones(100, bool))["all"]
    want = -(60 * math.log(0.75) + 20 * math.log(0.25)) / 80
    assert c_h["n_scored"] == c_o["n_scored"] == 80
    assert abs(c_h["logloss"] - want) < 1e-12 and abs(c_o["logloss"] - want) < 1e-12
    assert not c_o["logloss"] < c_h["logloss"]
    old_h = MX.intent_read(*honest, np.ones(100, bool))["all"]["logloss"]
    old_o = MX.intent_read(*over, np.ones(100, bool))["all"]["logloss"]
    assert old_o < old_h - 0.1                       # the as-built bias, kept REPORTED (descriptive)
    # an over-certain arm with WORSE discrimination inside E_row (1:1) loses the conditional metric
    flat = _cols("blob", [10] * 60 + [11] * 20 + [12] * 20, [0.5] * 80 + [0.0] * 20, [1.0] * 100,
                 [True] * 80 + [False] * 20)
    c_f = MX.intent_conditional_read(*flat, np.ones(100, bool))["all"]
    assert abs(c_f["logloss"] - math.log(2)) < 1e-12 and c_f["logloss"] > c_h["logloss"]


def test_renormalisation_by_hand_and_the_shared_rows():
    """Row 0: P(e) = 0.3 of an E_row mass 0.6 → −log 0.5. Row 1: 0.1 of 0.4 → −log 0.25. Row 2: the
    event is outside E_row — not scored, counted in the set miss rate."""
    br, cols = _cols("fixed_mass", [10, SWITCH_BASE + 3, 12], [0.3, 0.1, 0.2], [0.6, 0.4, 0.5],
                     [True, True, False])
    r = MX.intent_conditional_read(br, cols, np.ones(3, bool))
    assert r["all"]["n_scored"] == 2 and abs(r["all"]["set_miss_rate"] - 1 / 3) < 1e-12
    assert abs(r["all"]["logloss"] - (math.log(2) + math.log(4)) / 2) < 1e-12
    assert abs(r["move"]["logloss"] - math.log(2)) < 1e-12
    assert abs(r["switch"]["logloss"] - math.log(4)) < 1e-12


def test_rule8_a_denominator_within_1e12_of_zero_is_excluded_and_counted():
    br, cols = _cols("fixed_mass", [10, 10, 10], [5e-13, 1e-12, 0.5], [9e-13, 1e-12, 1.0],
                     [True, True, True])
    r = MX.intent_conditional_read(br, cols, np.ones(3, bool))
    assert r["excluded_rule8_denominator"] == 2 and r["all"]["n_scored"] == 1
    assert abs(r["all"]["logloss"] - math.log(2)) < 1e-12
    # a near-tie at the reference blob's seat cut (E_row ill-determined) is excluded too
    br, cols = _cols("fixed_mass", [10, 10], [0.5, 0.25], [1.0, 1.0], [True, True], tie=[True, False])
    r = MX.intent_conditional_read(br, cols, np.ones(2, bool))
    assert r["excluded_rule8_tie"] == 1 and abs(r["all"]["logloss"] - math.log(4)) < 1e-12


def test_a_zero_mass_event_inside_the_set_is_never_floored():
    br, cols = _cols("fixed_mass", [10, 11], [0.5, 0.0], [1.0, 0.5], [True, True])
    r = MX.intent_conditional_read(br, cols, np.ones(2, bool))["all"]
    assert r["n_zero_event_mass"] == 1 and r["logloss"] is None


def test_coverage_calibration_by_hand():
    """10 rows with 0.25 outside mass (2 realised outside) and 20 rows with 0.07 (3 outside): CITL =
    mean mass − freq = (10·0.25 + 20·0.07)/30 − 5/30; the reliability bins hold each group. (Neither
    mass sits within a rounding error of a bin edge — 1 − 0.8 would, rule 8.)"""
    mass = [0.75] * 10 + [0.93] * 20
    ins = [False] * 2 + [True] * 8 + [False] * 3 + [True] * 17
    ev = [12 if not i else 10 for i in ins]
    br, cols = _cols("fixed_mass", ev, [0.1 if not i else 0.5 for i in ins], mass, ins)
    c = MX.intent_conditional_read(br, cols, np.ones(30, bool))["coverage"]
    assert c["n"] == 30
    assert abs(c["mean_outside_mass"] - 0.13) < 1e-12 and abs(c["outside_freq"] - 1 / 6) < 1e-12
    assert abs(c["citl"] - (0.13 - 1 / 6)) < 1e-12
    bins = {b["lo"]: b for b in c["bins"]}
    assert set(bins) == {0.05, 0.2}
    assert bins[0.05]["n"] == 20 and abs(bins[0.05]["mass"] - 0.07) < 1e-12
    assert abs(bins[0.05]["freq"] - 0.15) < 1e-12
    assert bins[0.2]["n"] == 10 and abs(bins[0.2]["freq"] - 0.2) < 1e-12


def test_no_reference_means_no_value_never_a_guess():
    br, cols = _cols("fixed_mass", [10], [0.5], [1.0], [True])
    cols.eset_mass = None
    assert MX.intent_conditional_read(br, cols, np.ones(1, bool))["available"] is False


# --------------------------------------------------------------------------- the dense distributions
def test_blob_dense_distribution_and_its_named_set():
    """The toy of `metrics_test.test_event_logp_common_event_space`: the dense distribution equals the
    scored probability on every covered event, sums to 1, and its named set is exactly the covered
    predicate."""
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
    P, sev, swok = blob_event_dist(alpha, seats, beta, ok, content)
    D = SWITCH_BASE + S
    idx, okv = event_index(events, D)
    pe = P.gather(1, idx.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(pe[cov].log(), lp[cov], atol=1e-12, rtol=0)
    assert torch.allclose(P.sum(-1), torch.ones(5, dtype=torch.float64), atol=1e-12, rtol=0)
    m = set_mask(sev, swok, D)
    assert (m.gather(1, idx.unsqueeze(-1)).squeeze(-1) & okv).tolist() == cov.tolist()
    assert set(torch.nonzero(m[0]).flatten().tolist()) == {10, 20, 237, SWITCH_BASE + 3, SWITCH_BASE + 5,
                                                           SWITCH_BASE + 6}
    assert abs(float((P[0] * m[0]).sum()) - 1.0) < 1e-12           # blob's mass is all on its own set


def test_flat_dense_distribution_matches_the_scored_events():
    """The toy of `metrics_test.test_flat_pointer_event_logp`, with OTHER_move's π and the species tail
    each summing to 1: the dense distribution equals the scored probability on every covered event and
    sums to 1; a reference E_row = {10, 237, 30, switch 3} renormalises it."""
    B, K, M, S2 = 7, 2, 400, 10
    logits = torch.tensor([[0.5, 1.0, 0.0, 2.0, 1.0, -math.inf, -math.inf, -math.inf, -math.inf, 0.3]])
    logits = logits.repeat(B, 1)
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
    fi = SimpleNamespace(k=K, live=torch.isfinite(logits), cand_ids=cand_ids, beyond=beyond,
                         in_tail=in_tail)
    ev = torch.tensor([237, 30, 99, SWITCH_BASE + 3, SWITCH_BASE + 5, SWITCH_BASE + 7, 10])
    lp, cov = _flat_event_logp(ev, logits, fi, pi_m, p_tail)
    P = flat_event_dist(logits, fi, pi_m, p_tail)
    D = SWITCH_BASE + S2
    idx, _ = event_index(ev, D)
    pe = P.gather(1, idx.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(pe[cov].log(), lp[cov], atol=1e-12, rtol=0)
    assert torch.allclose(pe[~cov], torch.zeros(int((~cov).sum()), dtype=torch.float64))
    assert torch.allclose(P.sum(-1), torch.ones(B, dtype=torch.float64), atol=1e-12, rtol=0)
    ref = ERow(seat_events=np.array([[10, 237, 30, 0]] * B), switch_ok=np.zeros((B, S2), bool),
               tie=np.zeros(B, bool))
    ref.switch_ok[:, 3] = True
    m = ref.mask(0, B, D)
    lf = torch.log_softmax(logits[0].double(), -1).exp()
    want = lf[0] + lf[1] + lf[2] * 0.2 + lf[2] * 0.6 + lf[3]   # 10 · HP (seat + member) · 30 · slot 3
    assert abs(float((P[0] * m[0]).sum()) - float(want)) < 1e-12


def test_erow_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    e = ERow(seat_events=rng.integers(0, 400, (50, 6)), switch_ok=rng.random((50, 13)) < 0.3,
             tie=rng.random(50) < 0.1, meta={"label": "b", "checkpoint_sha256": "c" * 64})
    p = tmp_path / "b.erow.npz"
    e.save(p)
    f = ERow.load(p)
    assert f.content_sha256() == e.content_sha256() and f.meta == e.meta
    assert (f.switch_ok == e.switch_ok).all() and (f.seat_events == e.seat_events).all()
    with pytest.raises(AssertionError):
        g = ERow.load(p)
        g.tie[0] = not g.tie[0]
        assert g.content_sha256() == e.content_sha256()
