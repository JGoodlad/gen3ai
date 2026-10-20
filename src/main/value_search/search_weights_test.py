"""``search_decision``'s world weighting (``SearchConfig.weights``): ``likelihood`` averages each action's leaves over
its worlds by the sampler's self-normalised ``Worlds.log_w`` (F1 tier A), ``uniform`` is today's mean. A fake engine and
a fake sampler: pure, no core."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from main.value_search.playout import ACT
from main.value_search.search import SearchConfig, search_decision


class _Engine:
    """Leaf value = VALUES[action][world]; the policy prefers action 0."""

    off = 0
    VALUES = {0: [1.0, 0.0, 0.0, 0.0], 1: [0.0, 0.6, 0.6, 0.6]}

    def forward(self, rows, masks):
        out = np.zeros((len(rows), ACT + 1), dtype=np.float32)
        out[:, 0], out[:, 1], out[:, ACT] = 2.0, 0.0, 0.5
        return out

    def root(self, log, at, side):
        return {"other_open": True}

    def leaves(self, logs, at, side, root_turn, reqs, depth=None):
        return [self.VALUES[r.action][r.world] for r in reqs]


class _Sampler:
    def __init__(self, log_w):
        self.log_w = log_w

    def sample(self, log, at, side, J, key, root_choice_in_log=False):
        return SimpleNamespace(logs=[{}] * J, tries=[1] * J, leak={}, log_w=list(self.log_w))


def _search(weights, log_w, allocation="uniform"):
    mask = np.zeros(ACT, dtype=np.uint8)
    mask[[0, 1]] = 1
    row = np.ones(8, dtype=np.float32)                     # decode_turn reads column 0: turn 0
    cfg = SearchConfig(per_action=4, n_worlds=4, allocation=allocation, rule="argmax", depth=None, weights=weights)
    return search_decision(_Engine(), {}, 0, "p1", row, mask, cfg, key="t", sampler=_Sampler(log_w))


def test_uniform_is_the_plain_mean():
    res = _search("uniform", [0.0, -9.0, -9.0, -9.0])
    assert res.q == pytest.approx({0: 0.25, 1: 0.45})
    assert res.action == 1


def test_likelihood_weights_move_the_pick():
    # world 0 holds nearly all the likelihood: action 0's value there (1.0) wins
    res = _search("likelihood", [0.0, -9.0, -9.0, -9.0])
    w = np.exp([0.0, -9.0, -9.0, -9.0])
    w /= w.sum()
    assert res.q[0] == pytest.approx(float(w[0]))
    assert res.q[1] == pytest.approx(0.6 * float(w[1:].sum()))
    assert res.action == 0


def test_likelihood_on_seed_dice_worlds_is_uniform():
    assert _search("likelihood", []).q == pytest.approx(_search("uniform", []).q)


def test_likelihood_refuses_halving_and_unknown_modes():
    with pytest.raises(ValueError):
        _search("likelihood", [0.0] * 4, allocation="halving")
    with pytest.raises(ValueError):
        _search("posterior", [0.0] * 4)
    assert math.isfinite(_search("uniform", [0.0] * 4).value_target)


# ------------------------------------------------------------------------------------------------
# STAGE 1c: SearchConfig.proposals (draw more worlds, keep n_worlds by the opponent-choice resample)
# ------------------------------------------------------------------------------------------------

class _WorldEngine(_Engine):
    """Leaf value of action 0 is 1 only in the world whose log is tagged ``wid`` 5; action 1 is 0.6 everywhere."""

    def leaves(self, logs, at, side, root_turn, reqs, depth=None):
        return [(1.0 if logs[r.world]["wid"] == 5 else 0.0) if r.action == 0 else 0.6 for r in reqs]


class _ProposalSampler:
    def __init__(self):
        self.asked = []

    def sample(self, log, at, side, J, key, root_choice_in_log=False):
        from main.value_search.worlds import Worlds

        self.asked.append(J)
        return Worlds(logs=[{"wid": j} for j in range(J)], teams=[f"t{j}" for j in range(J)], tries=[1] * J, leak={},
                      revealed={})


def _search_1c(monkeypatch, proposals, n_worlds=4):
    from main.value_search import opp_likelihood as OL
    from main.value_search import worlds as W

    seen = {}

    def fake_logliks(root, logs, forward, steps=None, label=""):
        seen["logs"] = [lg["wid"] for lg in logs]
        return [OL.OppLikelihood(ll=0.0 if lg["wid"] == 5 else -40.0, conf=0.0) for lg in logs]

    monkeypatch.setattr(OL, "world_logliks", fake_logliks)
    monkeypatch.setattr(W, "prepare_root", lambda *a, **k: "root")
    mask = np.zeros(ACT, dtype=np.uint8)
    mask[[0, 1]] = 1
    cfg = SearchConfig(per_action=n_worlds, n_worlds=n_worlds, rule="argmax", depth=None, proposals=proposals)
    smp = _ProposalSampler()
    res = search_decision(_WorldEngine(), {}, 0, "p1", np.ones(8, dtype=np.float32), mask, cfg, key="t", sampler=smp)
    return res, smp, seen


def test_proposals_off_draws_n_worlds_as_today(monkeypatch):
    res, smp, seen = _search_1c(monkeypatch, None)
    assert smp.asked == [4] and "logs" not in seen
    assert res.q == pytest.approx({0: 0.0, 1: 0.6}) and res.action == 1      # worlds 0..3: world 5 never drawn


def test_proposals_resample_the_kept_worlds_by_the_opponent_likelihood(monkeypatch):
    res, smp, seen = _search_1c(monkeypatch, 8)
    assert smp.asked == [8] and seen["logs"] == list(range(8))     # every proposal scored
    assert res.q == pytest.approx({0: 1.0, 1: 0.6}) and res.action == 0      # all 4 kept worlds are world 5


def test_proposals_fewer_than_the_worlds_or_on_the_oracle_are_refused(monkeypatch):
    with pytest.raises(ValueError):
        _search_1c(monkeypatch, 3)
    mask = np.zeros(ACT, dtype=np.uint8)
    mask[[0, 1]] = 1
    with pytest.raises(ValueError):
        search_decision(_WorldEngine(), {}, 0, "p1", np.ones(8, dtype=np.float32), mask,
                        SearchConfig(n_worlds=4, oracle=True, proposals=8), key="t")
