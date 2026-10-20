"""The opponent-choice likelihood's numerics (``opp_likelihood.log_probs``): a masked softmax in float64, a forced
decision worth exactly 0, the familiarity control never above the chosen action's log-probability's ceiling."""
from __future__ import annotations

import numpy as np
import pytest

from main.value_search import opp_likelihood as OL


def test_a_forced_decision_contributes_exactly_zero():
    logits = np.array([[5.0, -3.0] + [0.0] * 10], dtype=np.float32)
    mask = np.zeros((1, 11), dtype=np.uint8)
    mask[0, 1] = 1
    lp, cf = OL.log_probs(logits, mask, [1])
    assert lp[0] == 0.0 and cf[0] == 0.0


def test_the_masked_softmax_ignores_illegal_logits():
    logits = np.zeros((1, 12), dtype=np.float32)
    logits[0, 0] = 100.0                                 # illegal: must not take any mass
    mask = np.zeros((1, 11), dtype=np.uint8)
    mask[0, [3, 4]] = 1
    lp, cf = OL.log_probs(logits, mask, [3])
    assert lp[0] == pytest.approx(np.log(0.5), abs=1e-12) and cf[0] == pytest.approx(np.log(0.5), abs=1e-12)


def test_the_chosen_and_top_log_probabilities():
    logits = np.array([[0.0, 1.0, 2.0] + [0.0] * 9], dtype=np.float32)
    mask = np.zeros((1, 11), dtype=np.uint8)
    mask[0, :3] = 1
    lp, cf = OL.log_probs(logits, mask, [1])
    z = np.log(np.exp(0.0) + np.exp(1.0) + np.exp(2.0))
    assert lp[0] == pytest.approx(1.0 - z, abs=1e-12) and cf[0] == pytest.approx(2.0 - z, abs=1e-12)
    assert lp[0] <= cf[0] <= 0.0


def test_a_masked_out_choice_is_a_defect():
    mask = np.zeros((1, 11), dtype=np.uint8)
    mask[0, 2] = 1
    with pytest.raises(OL.OppLikelihoodError):
        OL.log_probs(np.zeros((1, 12), dtype=np.float32), mask, [5])


# ------------------------------------------------------------------------------------------------
# STAGE 1c: the importance resample (pure)
# ------------------------------------------------------------------------------------------------

def _worlds(n: int, log_w=None, rolls=True):
    """``n`` stand-in proposals: world j's log is tagged ``wid`` j and, under rolls, carries its own force table."""
    from main.value_search.worlds import Worlds

    logs, rl = [], []
    for j in range(n):
        r = [[j, j % 16]] if rolls else []
        lg = {"wid": j, "cmds": [], **({"rolls": r} if r else {})}
        logs.append(lg)
        rl.append(r)
    return Worlds(logs=logs, teams=[f"team{j}" for j in range(n)], tries=[1] * n, leak={}, revealed={},
                  stats=[{"attempt": 0, "j": j} for j in range(n)], facts=[[{"slot": 0, "j": j}] for j in range(n)],
                  dice="rolls" if rolls else "seed", log_w=list(log_w if log_w is not None else ([0.0] * n if rolls
                                                                                                else [])),
                  rolls=rl if rolls else [])


def test_systematic_keeps_each_proposal_floor_or_ceil_of_k_times_its_weight_and_never_a_zero():
    w = np.array([0.5, 0.25, 0.25, 0.0])
    for u in (0.0, 0.1, 0.49, 0.5, 0.99):
        idx = OL.systematic(w, 4, u)
        assert sorted(idx) == [0, 0, 1, 2], (u, idx)
    rng = np.random.default_rng(3)
    for _ in range(200):
        v = rng.dirichlet(np.ones(9) * 0.3)
        v[rng.integers(9)] = 0.0
        k = int(rng.integers(1, 20))
        idx = OL.systematic(v, k, float(rng.random()))
        cnt = np.bincount(idx, minlength=9)
        p = v / v.sum()
        assert len(idx) == k and np.all(cnt[p == 0] == 0)
        assert np.all(cnt >= np.floor(k * p) - 1e-9) and np.all(cnt <= np.ceil(k * p) + 1e-9)


def test_the_resample_uniform_is_deterministic_per_key():
    assert OL.resample_uniform("a") == OL.resample_uniform("a")
    assert OL.resample_uniform("a") != OL.resample_uniform("b")
    assert all(0.0 <= OL.resample_uniform(str(i)) < 1.0 for i in range(100))


def test_the_posterior_weight_is_the_dice_weight_times_the_opponent_likelihood():
    dice = [0.0, -5.0, 0.0, 0.0]
    ll = [-9.0, 0.0, -1.0, 0.0]
    assert OL.posterior_log_weights(4, dice, ll).tolist() == [-9.0, -5.0, -1.0, 0.0]
    assert OL.posterior_log_weights(4, dice, None).tolist() == dice
    assert OL.posterior_log_weights(4, [], ll).tolist() == ll              # seed dice: no damage-roll term
    with pytest.raises(OL.OppLikelihoodError):
        OL.posterior_log_weights(4, dice, [0.0, 0.0, float("-inf"), 0.0])
    with pytest.raises(OL.OppLikelihoodError):
        OL.posterior_log_weights(4, dice, [0.0] * 3)


def test_the_resample_follows_both_terms_and_spends_the_weights():
    # the dice favour world 0, the opponent's choices world 3: the product decides (world 3 here)
    W = _worlds(4, log_w=[0.0, -30.0, -30.0, -10.0])
    ll = [-30.0, -30.0, -30.0, 0.0]
    out = OL.resample(W, 4, "k", ll=ll)
    assert out.resample["src"] == [3, 3, 3, 3]
    assert [lg["wid"] for lg in out.logs] == [3, 3, 3, 3] and out.teams == ["team3"] * 4
    assert out.log_w == [0.0] * 4                            # spent: a search over them never re-weights by the dice
    # dice only: world 0
    assert OL.resample(W, 4, "k", ll=None).resample["src"] == [0, 0, 0, 0]
    # the opponent term alone (dice off): world 3
    assert OL.resample(W, 4, "k", ll=ll, dice=False).resample["src"] == [3, 3, 3, 3]
    rec = out.resample
    assert rec["n"] == 4 and rec["k"] == 4 and rec["opp"] and rec["dice"]
    assert rec["log_post"] == [-30.0, -60.0, -60.0, -10.0] and rec["ess"] == pytest.approx(1.0, abs=1e-6)


def test_the_kept_worlds_carry_their_own_rolls():
    W = _worlds(6, log_w=[0.0] * 6)
    ll = [-50.0, 0.0, -50.0, -50.0, 0.0, -50.0]                # half the mass on world 1, half on world 4
    out = OL.resample(W, 4, "rolls", ll=ll)
    assert sorted(out.resample["src"]) == [1, 1, 4, 4]
    for lg, r, j in zip(out.logs, out.rolls, out.resample["src"]):
        assert lg["wid"] == j and lg["rolls"] == [[j, j % 16]] and r == [[j, j % 16]]
    assert [f[0]["j"] for f in out.facts] == out.resample["src"]


def test_effective_sample_size():
    assert OL.ess([1.0] * 8) == pytest.approx(8.0)
    assert OL.ess([1.0] + [0.0] * 7) == pytest.approx(1.0)
    assert OL.ess(OL.normalise([0.0, 0.0, -1e9])) == pytest.approx(2.0)
