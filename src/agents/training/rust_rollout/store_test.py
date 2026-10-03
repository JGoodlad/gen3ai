"""Pins for the row arena and the two buffer fills (M5 Lane G).

* ``game_gae`` over a complete game is BIT-IDENTICAL to sb3's ``compute_returns_and_advantage`` on a
  buffer holding the same game (the complete-game GAE is sb3's arithmetic, not a re-derivation);
* ``take_complete`` is FIFO by completion, exact in count, and splits only the straddling game;
* the complete-game fill labels every row with its own game's outcome and carries the precomputed
  advantages; the window fill reproduces sb3's GAE and the callback's back-fill (unfinished = mask 0);
* nothing is dropped: every row allocated is either consumed, carried or still in play;
* a CUT game releases its rows (complete-game) and is refused in window mode.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from agents.training.rollout_buffer import RolloutBuffer

from agents.training.rust_rollout import store as S

GAMMA, LAM = 1.0, 0.8


def _space():
    return spaces.Dict({
        "observation": spaces.Box(-np.inf, np.inf, (3,), np.float32),
        "action_mask": spaces.Box(0, 1, (11,), np.int8),
        "opp_class": spaces.Box(0, 3, (1,), np.int64),
        "win_target": spaces.Box(0.0, 1.0, (1,), np.float32),
        "win_mask": spaces.Box(0.0, 1.0, (1,), np.float32),
    })


def _buf(n_steps, n_envs, gamma=GAMMA, lam=LAM):
    return RolloutBuffer(n_steps, _space(), spaces.Discrete(11), device="cpu", gamma=gamma,
                                     gae_lambda=lam, n_envs=n_envs)


@pytest.mark.parametrize("gamma,lam", [(1.0, 0.8), (0.99, 0.95), (0.9999, 1.0)])
def test_game_gae_is_sb3s_arithmetic_bit_for_bit(gamma, lam):
    rng = np.random.default_rng(0)
    lengths = [7, 13, 1, 22]
    n_steps = sum(lengths)
    buf = _buf(n_steps, 1, gamma, lam)
    buf.reset()
    rewards = np.zeros(n_steps, np.float32)
    starts = np.zeros(n_steps, np.float32)
    values = rng.random(n_steps).astype(np.float32)
    at = 0
    for L in lengths:
        starts[at] = 1.0
        rewards[at + L - 1] = rng.choice([0.0, 1.0])
        at += L
    buf.rewards[:, 0] = rewards
    buf.episode_starts[:, 0] = starts
    buf.values[:, 0] = values
    # the last game ends at the buffer edge: dones = 1 (last_values irrelevant, x 0)
    buf.compute_returns_and_advantage(last_values=th.tensor([0.37]), dones=np.array([True]))
    at = 0
    for L in lengths:
        adv, ret = S.game_gae(rewards[at:at + L], values[at:at + L], gamma, lam)
        assert adv.tobytes() == buf.advantages[at:at + L, 0].tobytes()
        assert ret.tobytes() == buf.returns[at:at + L, 0].tobytes()
        at += L


def _store(capacity=500):
    return S.RowStore(_space(), capacity)


def _play(log, env, n_rows, rng, version=0):
    st = log.store
    slots = st.alloc(n_rows)
    st.value[slots] = rng.random(n_rows).astype(np.float32)
    st.logp[slots] = -rng.random(n_rows).astype(np.float32)
    st.action[slots] = rng.integers(0, 11, n_rows)
    st.version[slots] = version
    st.start[slots] = 0.0
    if not log.cur[env]:
        st.start[slots[0]] = 1.0
    st.obs["observation"][slots] = rng.random((n_rows, 3)).astype(np.float32)
    st.obs["action_mask"][slots] = 1
    for s in slots:
        log.add(np.array([env]), np.array([s]))
    return slots


def test_take_complete_is_fifo_exact_and_splits_only_the_straddler():
    rng = np.random.default_rng(1)
    st = _store()
    log = S.GameLog(st, 2, mode="complete_game", gamma=GAMMA, gae_lambda=LAM, max_game_rows=100)
    a = _play(log, 0, 5, rng)
    log.end(0, reward=1.0, outcome=1.0, draw=False, forfeit=False, terminated=True, truncated=False,
            episode=0, opp_class=0)
    b = _play(log, 1, 7, rng)
    log.end(1, reward=0.0, outcome=0.0, draw=False, forfeit=False, terminated=True, truncated=False,
            episode=0, opp_class=0)
    assert log.completed_rows == 12
    got, whole, split = log.take_complete(8)
    assert got.tolist() == a.tolist() + b[:3].tolist() and (whole, split) == (1, 1)
    assert log.completed_rows == 4
    got2, whole2, split2 = log.take_complete(4)
    assert got2.tolist() == b[3:].tolist() and (whole2, split2) == (1, 0)


def test_complete_fill_labels_every_row_and_carries_game_gae():
    rng = np.random.default_rng(2)
    st = _store()
    n = 2
    log = S.GameLog(st, n, mode="complete_game", gamma=GAMMA, gae_lambda=LAM, max_game_rows=100)
    games = []
    for e, L, won in ((0, 6, 1.0), (1, 5, 0.0), (0, 9, 0.0)):
        s = _play(log, e, L, rng, version=1)
        g = log.end(e, reward=won, outcome=won, draw=False, forfeit=False, terminated=True, truncated=False,
                    episode=len(games), opp_class=2)
        games.append((s, won, g))
    in_play = _play(log, 1, 3, rng, version=2)
    buf = _buf(8, n)
    live_before = st.live_count
    rep, versions = S.fill_complete(buf, log, 16, current_version=2)
    assert rep.rows == 16 and rep.labelled_rows == 16 and rep.carry_rows == 4
    assert rep.in_progress_rows == 3 and rep.age_hist == {1: 16}
    assert (buf.observations["win_mask"] == 1.0).all()
    flat = [s for g in games for s in g[0].tolist()][:16]
    idx = np.asarray(flat).reshape(n, 8).T
    exp_outcome = {int(s): won for s_arr, won, _ in games for s in s_arr}
    assert np.array_equal(buf.observations["win_target"][..., 0],
                          np.vectorize(lambda s: exp_outcome[int(s)])(idx).astype(np.float32))
    for s_arr, won, _ in games:
        adv, ret = S.game_gae(st.reward[s_arr], st.value[s_arr], GAMMA, LAM)
        pos = [flat.index(int(s)) for s in s_arr if int(s) in flat]
        for p, a_, r_ in zip(pos, adv, ret):
            assert buf.advantages[p % 8, p // 8] == a_ and buf.returns[p % 8, p // 8] == r_
    assert np.array_equal(buf.values, st.value[idx]) and np.array_equal(buf.log_probs, st.logp[idx])
    assert versions.shape == (8, 2)
    # nothing dropped: 16 consumed + 4 carried + 3 in play, and exactly 16 released
    assert live_before - st.live_count == 16 and st.live_count == 4 + 3
    assert set(in_play.tolist()) <= set(log.cur[1])


def test_window_fill_is_sb3_gae_and_the_callbacks_backfill():
    rng = np.random.default_rng(3)
    st = _store()
    n, T = 2, 6
    log = S.GameLog(st, n, mode="window", gamma=GAMMA, gae_lambda=LAM, max_game_rows=100)
    # env 0: a game of 4 (won), then a game still running; env 1: one long running game
    _play(log, 0, 4, rng)
    log.end(0, reward=1.0, outcome=1.0, draw=False, forfeit=False, terminated=True, truncated=False,
            episode=0, opp_class=0)
    _play(log, 0, 5, rng)
    _play(log, 1, 8, rng)
    buf = _buf(T, n)
    rep, _v = S.fill_window(buf, log, T, current_version=0)
    # the reference: the same rows through sb3 + the callback's back-fill
    ref = _buf(T, n)
    ref.reset()
    ref.rewards[...] = buf.rewards
    ref.episode_starts[...] = buf.episode_starts
    ref.values[...] = buf.values
    assert buf.rewards[3, 0] == 1.0 and buf.episode_starts[4, 0] == 1.0
    # the bootstrap: env 0's 7th row and env 1's 7th row (both mid-game)
    nxt = np.array([st.value[log.order[0][0]], st.value[log.order[1][0]]], np.float32)
    ref.compute_returns_and_advantage(last_values=th.as_tensor(nxt), dones=np.array([False, False]))
    assert ref.advantages.tobytes() == buf.advantages.tobytes()
    wm = buf.observations["win_mask"][..., 0]
    assert wm[:4, 0].tolist() == [1.0] * 4 and wm[4:, 0].tolist() == [0.0] * 2 and not wm[:, 1].any()
    assert buf.observations["win_target"][:4, 0, 0].tolist() == [1.0] * 4
    assert rep.labelled_rows == 4


def test_a_cut_game_releases_its_rows_and_window_mode_refuses_one():
    rng = np.random.default_rng(4)
    st = _store()
    log = S.GameLog(st, 1, mode="complete_game", gamma=GAMMA, gae_lambda=LAM, max_game_rows=100)
    _play(log, 0, 6, rng)
    before = st.live_count
    assert log.cut(0) == 6 and st.live_count == before - 6 and log.rows_cut == 6
    wlog = S.GameLog(_store(), 1, mode="window", gamma=GAMMA, gae_lambda=LAM, max_game_rows=100)
    _play(wlog, 0, 3, rng)
    with pytest.raises(S.CollectorError, match="CUT"):
        wlog.cut(0)


def test_the_arena_refuses_past_its_declared_capacity_and_a_game_past_max_rows():
    st = _store(capacity=4)
    st.alloc(3)
    with pytest.raises(S.ArenaExhausted):
        st.alloc(2)
    log = S.GameLog(_store(), 1, mode="complete_game", gamma=GAMMA, gae_lambda=LAM, max_game_rows=3)
    with pytest.raises(S.CollectorError, match="max_game_rows"):
        _play(log, 0, 4, np.random.default_rng(0))


def test_unfillable_keys_are_refused_by_name():
    sp = spaces.Dict({"observation": spaces.Box(0, 1, (3,), np.float32),
                      "some_unported_key": spaces.Box(0, 1, (25,), np.float32)})
    with pytest.raises(S.UnfillableKey, match="some_unported_key"):
        S.obs_key_sources(sp, ())


def test_the_fork_mask_key_is_a_host_constant():
    """`fork_pg_m` (declared only with --fork-fraction > 0) is the host's: 1.0 on every collected row."""
    sp = spaces.Dict({"observation": spaces.Box(0, 1, (3,), np.float32),
                      "fork_pg_m": spaces.Box(0, 1, (1,), np.float32)})
    assert S.obs_key_sources(sp, ())["fork_pg_m"] == "host_const"


def test_a_row_ahead_of_the_learner_is_refused():
    with pytest.raises(S.CollectorError, match="AHEAD"):
        S._age_hist(np.array([3]), 2)


def test_both_fills_hand_the_learner_each_row_s_collection_provenance():
    """K9(b) provenance (gen3_behaviour_provenance_v1): `write_rows`' full masked log-prob row and serving
    slot, and the arena's env / episode / dec_n / version / draw, reach the learner [n_steps, n_envs]-aligned
    with the buffer — the stored behaviour log-prob is the provenance row's entry at the stored action."""
    rng = np.random.default_rng(5)
    st = _store()
    n = 2
    log = S.GameLog(st, n, mode="complete_game", gamma=GAMMA, gae_lambda=LAM, max_game_rows=100)
    for e, L in ((0, 6), (1, 5), (0, 9)):
        s = _play(log, e, L, rng, version=3)
        full = np.log(rng.dirichlet(np.ones(11), L)).astype(np.float32)
        st.logp_all[s] = full
        st.logp[s] = full[np.arange(L), st.action[s]]
        st.slot[s] = 7 + e
        st.env[s] = e
        st.episode[s] = 40 + e
        st.dec_n[s] = np.arange(L)
        log.end(e, reward=1.0, outcome=1.0, draw=False, forfeit=False, terminated=True, truncated=False,
                episode=0, opp_class=0)
    buf = _buf(8, n)
    rep, _v = S.fill_complete(buf, log, 16, current_version=3)
    p = rep.provenance
    assert set(S.PROVENANCE_COLUMNS) | {"logp_all"} <= set(p) and p["logp_all"].shape == (8, n, 11)
    for k in S.PROVENANCE_COLUMNS:
        assert p[k].shape == (8, n), k
    acts = buf.actions.astype(np.int64).reshape(8, n, 1)
    stored = np.take_along_axis(p["logp_all"], acts, axis=2)[..., 0]
    assert np.array_equal(stored, buf.log_probs)
    assert set(np.unique(p["slot"])) <= {7, 8} and (p["version"] == 3).all()
    assert np.array_equal(p["slot"] - 7, p["env"]) and np.array_equal(p["episode"] - 40, p["env"])
    # a released-then-reallocated row carries no stale provenance
    s2 = st.alloc(3)
    assert np.isnan(st.logp_all[s2]).all() and (st.slot[s2] == S.ROW_SOURCE_UNKNOWN).all()


def test_write_rows_records_the_full_distribution_and_the_serving_slot():
    st = _store()
    slots = st.alloc(2)
    cols = {"obs": np.zeros((3, 2, 3), np.float32), "mask": np.ones((3, 2, 11), np.int8),
            "episode": np.array([5, 6, 7]), "dec_n": np.zeros((3, 2), np.int32)}
    full = np.log(np.full((2, 11), 1 / 11, np.float32))
    S.write_rows(st, slots, cols, np.array([0, 2]), label_keys=(), opp_class=np.zeros(2),
                 actions=np.array([1, 4]), logp=full[:, 1], values=np.zeros(2, np.float32),
                 version=np.array([9, 9]), u=np.zeros(2), margin=np.ones(2), starts=np.zeros(2, np.float32),
                 logp_all=full, served_by=np.array([3, 4]))
    assert np.array_equal(st.logp_all[slots], full) and st.slot[slots].tolist() == [3, 4]
    assert st.episode[slots].tolist() == [5, 7]
