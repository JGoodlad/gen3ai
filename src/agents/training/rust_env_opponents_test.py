"""M5 Lane E unit tier (routine, CPU, no core): the host's opponent routing and sampling rules.

* ``sample_actions`` equals ``RLPlayer``'s ``torch.multinomial(Categorical(logits=x/T).probs, 1, True,
  generator=g)`` BIT FOR BIT on the same probabilities and generator state;
* a slot family never loads over a model a current or staged episode plays, and running out of
  slots is typed;
* the plan declares its route table; the bot seed rule is the core's.

(The draw-for-draw equality of the per-episode opponent draw against the Python wrapper's
``_select_episode_opponent`` — an oracle replayed through the Python env core — was retired with that
core, deletion pass U3 / owner D3.)
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from agents.training import rust_env_opponents as E


def _reference_sample(logp_row, t, g):
    """RLPlayer._predict_best_action's stochastic branch, on one row."""
    x = torch.as_tensor(logp_row).reshape(1, -1)
    x = x / t if t != 1.0 else x
    cat = torch.distributions.Categorical(logits=x)
    return torch.multinomial(cat.probs, 1, True, generator=g).item()


@pytest.mark.parametrize("temps", [(1.0,), (0.7, 1.0, 1.6)])
def test_sample_actions_is_rlplayers_multinomial_bit_for_bit(temps):
    rng = np.random.default_rng(3)
    B, A, n = 24, 11, 60
    ga = [torch.Generator().manual_seed(100 + i) for i in range(B)]
    gb = [torch.Generator().manual_seed(100 + i) for i in range(B)]
    for step in range(n):
        raw = rng.normal(0, 2.5, (B, A)).astype(np.float32)
        mask = rng.random((B, A)) < 0.6
        mask[np.arange(B), rng.integers(0, A, B)] = True
        logp = torch.log_softmax(torch.as_tensor(raw).masked_fill(~torch.as_tensor(mask), -1e8), -1)
        logp = logp.masked_fill(~torch.as_tensor(mask), float("-inf")).numpy()
        t = [temps[(i + step) % len(temps)] for i in range(B)]
        got = E.sample_actions(logp, t, ga)
        want = [_reference_sample(logp[i], t[i], gb[i]) for i in range(B)]
        assert got.tolist() == want, f"step {step}"
        assert mask[np.arange(B), got].all(), "an illegal action was sampled"
    # the streams advanced identically (the next draws still agree)
    assert [g.initial_seed() for g in ga] == [g.initial_seed() for g in gb]
    assert all(torch.equal(x.get_state(), y.get_state()) for x, y in zip(ga, gb))


def test_sampling_follows_the_policy_distribution():
    """Distribution check (chi-square, df = 3) at T = 1 and T = 0.5 against softmax(logp / T)."""
    logp = np.log(np.array([[0.5, 0.3, 0.15, 0.05] + [0.0] * 7], dtype=np.float32))
    for T in (1.0, 0.5):
        g = [torch.Generator().manual_seed(7)]
        counts = np.zeros(11)
        for _ in range(20000):
            counts[E.sample_actions(logp, [T], g)[0]] += 1
        p = np.exp(logp[0, :4] / T)
        p /= p.sum()
        exp = p * 20000
        chi2 = float(((counts[:4] - exp) ** 2 / exp).sum())
        assert counts[4:].sum() == 0
        assert chi2 < 16.3, (T, counts[:4], exp)   # p = 0.001 at df = 3


def test_a_slot_in_use_is_never_loaded_over_and_capacity_is_typed():
    fam = E.SlotFamily("pool", [10, 11, 12])
    loads = []
    fam.admit(["a", "b"], set(), lambda r, m: loads.append((r, m)))
    assert loads == [(10, "a"), (11, "b")]
    # "a" leaves the roster but route 10 is still played → the new model goes to the empty slot 12
    fam.admit(["b", "c"], {10}, lambda r, m: loads.append((r, m)))
    assert loads[-1] == (12, "c") and fam.resident == ["a", "b", "c"]
    # now 10 is free (not in use, not on the roster)
    fam.admit(["b", "c", "d"], set(), lambda r, m: loads.append((r, m)))
    assert loads[-1] == (10, "d")
    with pytest.raises(E.SlotCapacityExceeded):
        fam.admit(["b", "c", "e"], {10}, lambda r, m: loads.append((r, m)))


def test_the_plan_declares_the_route_table():
    plan = E.OpponentPlan(pool_slots=3, stable=(E.StableSpec("x", 0.8, ("T",)),),
                          exploiter=E.ExploiterSpec(temperature=1.3), bots=("heuristic",))
    rows = plan.spec_rows()
    assert rows == [{"kind": "policy", "slot": 0}, {"kind": "policy", "slot": 1}, {"kind": "policy", "slot": 2},
                    {"kind": "policy", "slot": 3}, {"kind": "policy", "slot": 4},
                    {"kind": "bot", "bot": "heuristic", "seed": 5}]
    assert plan.spec_rows(bots="external")[-1] == {"kind": "external"}
    r = plan.routes()
    assert [x.klass for x in r] == [1, 1, 1, 2, 3, 0]
    assert r[3].temperature == 0.8 and r[3].team_strs == ("T",) and r[4].player == "exploiter"
    assert plan.n_policy_slots == 5
    with pytest.raises(ValueError):
        E.OpponentPlan(pool_slots=2)            # no floor bucket


def test_from_args_builds_the_plan_the_run_declares():
    from types import SimpleNamespace as NS

    args = NS(self_play=True, self_play_temp=1.0, stable_opponent_selfplay_share=0.2, stable_opponent_pfsp=True,
              exploiter_keep_bots=False, exploiter_bot_fraction=0.5)
    plan = E.OpponentPlan.from_args(args, bot_names=("heuristic", "staller"),
                                    stable_entries=[NS(label="ext", temperature=0.9, team_strs=())])
    assert plan.pool_slots == 20 + E.DEFAULT_POOL_SPARE and plan.stable[0].temperature == 0.9 and plan.stable_pfsp
    off = E.OpponentPlan.from_args(NS(**{**vars(args), "self_play": False}), bot_names=("heuristic",),
                                   stable_entries=[NS(label="ext", temperature=0.9, team_strs=())])
    assert off.pool_slots == 0 and off.stable == (), "stable opponents join only under self-play"
    random.seed(0)


PINNED = [4517933670823692284, 5390792918547426617, 17462041922349011332]


def test_the_bot_seed_rule_is_the_cores():
    """``bot_stream_seed`` against ``opponents::stream_seed``'s pinned values (the Rust unit test pins
    the same three)."""
    assert E.bot_stream_seed(5, 3, 1) == E.bot_stream_seed(5, 3, 1)
    assert len({E.bot_stream_seed(5, e, k) for e in range(4) for k in range(3)}) == 12
    assert [E.bot_stream_seed(5, 0, 0), E.bot_stream_seed(11, 2, 1), E.bot_stream_seed(2**53 - 1, 47, 2)] == PINNED
