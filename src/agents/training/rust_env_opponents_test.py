"""M5 Lane E unit tier (routine, CPU, no core): the host's opponent routing equals TODAY's rules.

* the per-episode draw equals ``MaskableAgentWrapper._select_episode_opponent`` DRAW FOR DRAW, from
  the same seeds, under every production branch (exploiter sole / keep-bots, the self-play coin,
  the capped stable share, stable PFSP, mastered stables in the floor, ``--bot-weights``, one pool
  snapshot per env per generation through ``SnapshotPool.sample`` with recency x PFSP, a generation
  push that adds a snapshot);
* ``sample_actions`` equals ``RLPlayer``'s ``torch.multinomial(Categorical(logits=x/T).probs, 1, True,
  generator=g)`` BIT FOR BIT on the same probabilities and generator state;
* a slot family never loads over a model a current or staged episode plays, and running out of
  slots is typed.
"""
from __future__ import annotations

import random
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from agents.training import rust_env_opponents as E
from agents.training.snapshot_pool import SnapshotPool
from agents.training.wrappers import MaskableAgentWrapper


def _stub_env():
    env = MagicMock()
    env.agent1.username = "a1"
    env.observation_spaces = {"a1": MagicMock()}
    env.action_spaces = {"a1": MagicMock()}
    return env


def _pool(tmp_path, steps, seed=None, pfsp=0.0):
    for s in steps:
        (tmp_path / f"snapshot_{s:012d}.zip").write_bytes(b"x")
    p = SnapshotPool(tmp_path, current_version=None, pfsp_scale=pfsp, rng_seed=seed)
    p.load_model = lambda entry: entry.step     # the wrapper assigns this to pool_player.model
    return p


CONFIGS = [
    # (label, plan kwargs, fraction, mastered, stable win rates, pool pfsp win rates)
    ("bots-only", dict(pool_slots=0), 0.0, [], {}, {}),
    ("pool+bots", dict(), 0.85, [], {}, {}),
    ("pool+stable", dict(stable=2), 0.9, [], {}, {}),
    ("stable-pfsp", dict(stable=3, stable_pfsp=True), 0.9, [], {"s0": 0.2, "s1": 0.9, "s2": 0.99}, {}),
    ("mastered-floor+weights", dict(stable=2, bot_weights=(1.0, 3.0, 0.5)), 0.7, ["s1"], {}, {}),
    ("pool-pfsp", dict(pfsp=2.0), 0.95, [], {}, {100: 0.2, 300: 0.9}),
    ("exploiter-sole", dict(exploiter=True), 0.9, [], {}, {}),
    ("exploiter-keep-bots", dict(exploiter=True, keep_bots=True), 0.9, [], {}, {}),
]


@pytest.mark.parametrize("label,kw,fraction,mastered,swr,pwr", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_the_draw_is_the_wrappers_draw_for_draw(tmp_path, label, kw, fraction, mastered, swr, pwr):
    bots = ("heuristic", "heuristic2", "staller")
    n_stable = int(kw.get("stable", 0))
    labels = [f"s{j}" for j in range(n_stable)]
    steps = [100, 200, 300, 400]
    ex = kw.get("exploiter", False)
    plan = E.OpponentPlan(
        pool_slots=0 if kw.get("pool_slots") == 0 else len(steps) + 1 + 2,
        stable=tuple(E.StableSpec(lab, 1.0) for lab in labels),
        exploiter=E.ExploiterSpec(temperature=1.0, keep_bots=kw.get("keep_bots", False), bot_fraction=0.4) if ex else None,
        bots=bots, bot_weights=kw.get("bot_weights"), stable_pfsp=kw.get("stable_pfsp", False))
    routes = plan.routes()
    for env_idx in (0, 5):
        wdir, mdir = tmp_path / f"w{env_idx}", tmp_path / f"m{env_idx}"
        wdir.mkdir()
        mdir.mkdir()
        pool_seed = 1000 + env_idx
        use_pool = plan.pool_slots > 0
        wpool = _pool(wdir, steps, seed=pool_seed, pfsp=kw.get("pfsp", 0.0)) if use_pool else None
        mpool = _pool(mdir, steps, seed=None, pfsp=kw.get("pfsp", 0.0)) if use_pool else None
        heur = [MagicMock(name=b) for b in bots]
        stable = [MagicMock(name=lab) for lab in labels]
        exploiter = MagicMock(name="exploiter") if ex else None
        pool_player = MagicMock(name="pool") if use_pool else None
        w = MaskableAgentWrapper(
            _stub_env(), heuristic_opponents=heur, pool=wpool, pool_player=pool_player,
            self_play_fraction=fraction, rng_seed=env_idx, heuristic_weights=plan.bot_weights,
            stable_players=stable or None, stable_labels=labels or None, stable_pfsp=plan.stable_pfsp,
            exploiter_player=exploiter, exploiter_keep_bots=ex and kw.get("keep_bots", False),
            exploiter_bot_fraction=0.4)
        loads = []
        host = E.RustEnvOpponents(
            plan, env_idx + 1, load=lambda r, m: loads.append((r, m)), pool=mpool,
            stable_ids=[f"stable:{lab}" for lab in labels], exploiter_id="exploiter:t0" if ex else None,
            self_play_fraction=fraction, pool_rng_seeds=[pool_seed] * (env_idx + 1),
            rng_seeds=list(range(env_idx + 1)))
        s = host.samplers[env_idx]
        for obj in (w, s):
            obj.set_stable_mastered(mastered)
            obj.set_stable_win_rates(swr)
        if use_pool:
            wpool.set_win_rates(pwr)
            mpool.set_win_rates(pwr)

        def ident_w():
            o = w.opponent
            if o is pool_player:
                return ("pool", int(pool_player.model))
            if o in stable:
                return ("stable", labels[stable.index(o)])
            if o is exploiter:
                return ("exploiter",)
            return ("bot", bots[heur.index(o)])

        def ident_m(r):
            rt = routes[r]
            if rt.family == "pool":
                mid = host.families["pool"].resident[host.families["pool"].routes.index(r)]
                return ("pool", int(mid.split(":")[1]))
            if rt.family.startswith("stable:"):
                return ("stable", rt.family.split(":", 1)[1])
            if rt.family == "exploiter":
                return ("exploiter",)
            return ("bot", rt.bot)

        seen = set()
        for k in range(1500):
            if k in (400, 900):            # an eval push: a promotion adds a snapshot, a new generation
                gen = 1 if k == 400 else 2
                if use_pool:
                    for d in (wdir, mdir):
                        (d / f"snapshot_{500 * gen:012d}.zip").write_bytes(b"x")
                w.set_self_play_target(fraction, gen)
                host.set_self_play_target(fraction, gen)
            w._select_episode_opponent()
            a, b = ident_w(), ident_m(s.draw())
            assert a == b, f"{label} env {env_idx} draw {k}: wrapper {a}, Lane E {b}"
            seen.add(a[0])
        want = {"bots-only": {"bot"}, "exploiter-sole": {"exploiter"}, "exploiter-keep-bots": {"exploiter", "bot"}}
        assert seen >= want.get(label, {"pool", "bot"} | ({"stable"} if n_stable else set())), (label, seen)


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
                          exploiter=E.ExploiterSpec(temperature=1.3, ladder=True), bots=("heuristic",))
    rows = plan.spec_rows()
    assert rows == [{"kind": "policy", "slot": 0}, {"kind": "policy", "slot": 1}, {"kind": "policy", "slot": 2},
                    {"kind": "policy", "slot": 3}, {"kind": "policy", "slot": 4}, {"kind": "policy", "slot": 5},
                    {"kind": "bot", "bot": "heuristic", "seed": 6}]
    assert plan.spec_rows(bots="external")[-1] == {"kind": "external"}
    r = plan.routes()
    assert [x.klass for x in r] == [1, 1, 1, 2, 3, 3, 0]
    assert r[3].temperature == 0.8 and r[3].team_strs == ("T",) and r[4].player == r[5].player == "exploiter"
    assert plan.n_policy_slots == 6
    with pytest.raises(ValueError):
        E.OpponentPlan(pool_slots=2)            # no floor bucket


def test_from_args_mirrors_env_factory():
    from types import SimpleNamespace as NS

    args = NS(self_play=True, self_play_temp=1.0, stable_opponent_selfplay_share=0.2, stable_opponent_pfsp=True,
              exploiter_keep_bots=False, exploiter_bot_fraction=0.5, exploiter_ladder=None)
    plan = E.OpponentPlan.from_args(args, bot_names=("heuristic", "staller"),
                                    stable_entries=[NS(label="ext", temperature=0.9, team_strs=())])
    assert plan.pool_slots == 20 + E.DEFAULT_POOL_SPARE and plan.stable[0].temperature == 0.9 and plan.stable_pfsp
    off = E.OpponentPlan.from_args(NS(**{**vars(args), "self_play": False}), bot_names=("heuristic",),
                                   stable_entries=[NS(label="ext", temperature=0.9, team_strs=())])
    assert off.pool_slots == 0 and off.stable == (), "stable opponents join only under self-play (env_factory)"
    random.seed(0)


PINNED = [4517933670823692284, 5390792918547426617, 17462041922349011332]


def test_the_bot_seed_rule_is_the_cores():
    """``bot_stream_seed`` against ``opponents::stream_seed``'s pinned values (the Rust unit test pins
    the same three)."""
    assert E.bot_stream_seed(5, 3, 1) == E.bot_stream_seed(5, 3, 1)
    assert len({E.bot_stream_seed(5, e, k) for e in range(4) for k in range(3)}) == 12
    assert [E.bot_stream_seed(5, 0, 0), E.bot_stream_seed(11, 2, 1), E.bot_stream_seed(2**53 - 1, 47, 2)] == PINNED
