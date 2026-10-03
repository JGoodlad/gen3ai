"""Gate (b) of the fork arm's Rust core port (``designs/training/forks.md`` §14.8): the BUFFER RULES,
pure unit (no core, no T2).

* the PREFIX is counted once: a branch's rows begin AT the fork step, and the FIFO holds the parent's
  rows exactly once with every branch game right after it;
* every branch's fork-step row carries ``fork_pg_m`` 0 (every later row 1, every collected row 1),
  and the fill hands that key to the learner, whose PG term renormalises by ``m.sum()``;
* the advantages / returns equal a HAND-BUILT GAE over the branch's outcome (indicator reward, no
  bootstrap), ``win_target`` is the branch's outcome bit;
* capped branches are dropped, a fork with < 2 scorable branches is dropped WHOLE, and the fork row
  budget counts (and releases) branch rows;
* the refusals: a non-indicator terminal (non-winprob critic), generator sampling, the window trigger.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from agents.training.rollout_buffer import RolloutBuffer

from agents.training.lever_supply import DryStreakGuard, LeverConfigError, LeverStarvedError
from agents.training.rust_rollout import fork as FK
from agents.training.rust_rollout import store as S

GAMMA, LAM = 1.0, 0.8


def _space():
    return spaces.Dict({
        "observation": spaces.Box(-np.inf, np.inf, (3,), np.float32),
        "action_mask": spaces.Box(0, 1, (11,), np.int8),
        "opp_class": spaces.Box(0, 3, (1,), np.int64),
        "fork_pg_m": spaces.Box(0, 1, (1,), np.float32),
        "win_target": spaces.Box(0.0, 1.0, (1,), np.float32),
        "win_mask": spaces.Box(0.0, 1.0, (1,), np.float32),
    })


def _col(capacity=64):
    st = S.RowStore(_space(), capacity, fork=True)
    log = S.GameLog(st, 1, mode="complete_game", gamma=GAMMA, gae_lambda=LAM, max_game_rows=40)
    return SimpleNamespace(store=st, log=log, version=3)


def _pass(budget=1000):
    p = FK.RustForkPass.__new__(FK.RustForkPass)
    p.decl = FK.ForkDecl(fraction=0.05)
    p.turn_limit, p.victory_value, p.row_budget = 250, 1.0, budget
    p.keep_cmds, p.seeds = False, [None]
    return p


def _parent(col, n=6, outcome=1.0):
    st = col.store
    s = st.alloc(n)
    st.value[s] = np.linspace(0.2, 0.7, n).astype(np.float32)
    st.obs["fork_pg_m"][s] = 1.0
    col.log.add(np.zeros(n, dtype=np.int64), s)
    col.log.end(0, reward=outcome, outcome=outcome, draw=False, forfeit=False, terminated=True,
                truncated=False, episode=0, opp_class=1)
    return s


def _branch_rows(col, p, f, b, values, pg0=True):
    st = col.store
    for k, v in enumerate(values):
        s = p._alloc(col, 1)
        assert s is not None
        st.obs["fork_pg_m"][s] = 0.0 if (k == 0 and pg0) else 1.0
        st.value[s] = np.float32(v)
        b.slots.append(int(s[0]))
        b.values.append(float(v))


def _fork(rec, names_actions, ends):
    f = FK.Fork(rec=rec, t=2, top1=0, top2=1, legal=[0, 1, 2], opp_slot=0, opp_real=False,
                opp_temperature=1.0, opp_greedy=False,
                branches=[FK.Branch(name=n, action=a) for n, a in names_actions])
    for b, e in zip(f.branches, ends):
        b.end = e
    return f


def _rec(slots):
    return FK.GameRecord(env=0, episode=0, slots=slots, run_seed=1, route=0, model_id=None, opp_class=0,
                         script="START {}")


def _hand_gae(values, outcome):
    """sb3's recursion written out for one complete game (indicator reward on the last row)."""
    n = len(values)
    r = np.zeros(n)
    r[-1] = outcome
    adv = np.zeros(n)
    last = 0.0
    for t in range(n - 1, -1, -1):
        nnt = 0.0 if t == n - 1 else 1.0
        nv = 0.0 if t == n - 1 else values[t + 1]
        delta = r[t] + GAMMA * nv * nnt - values[t]
        last = delta + GAMMA * LAM * nnt * last
        adv[t] = last
    return adv, adv + np.asarray(values)


WIN = {"winner": 0, "forfeit": False, "truncated": False, "turn": 30}
LOSS = {"winner": 1, "forfeit": False, "truncated": False, "turn": 31}
CAPPED = {"winner": 1, "forfeit": True, "truncated": False, "turn": 250}


def test_branch_games_follow_their_parent_once_and_carry_a_hand_built_gae():
    col = _col()
    p = _pass()
    other = _parent(col, 4, 0.0)
    parent = _parent(col, 6, 1.0)
    later = _parent(col, 2, 0.0)              # a game that ENDED after the parent: branches go before it
    rec = _rec(parent)
    f = _fork(rec, [("top1", 0), ("top2", 1)], [WIN, LOSS])
    vals = [[0.5, 0.6, 0.7], [0.5, 0.4, 0.3, 0.2]]
    for b, v in zip(f.branches, vals):
        _branch_rows(col, p, f, b, v)
    rep = FK.PassReport()
    p._inject(col, [f], rep)
    fifo = list(col.log.completed)
    # the FIFO: the earlier game, the parent ONCE, then its branches right after it
    assert [g.tolist() for g in fifo[:2]] == [other.tolist(), parent.tolist()]
    assert [g.tolist() for g in fifo[2:4]] == [f.branches[0].slots, f.branches[1].slots]
    assert fifo[4].tolist() == later.tolist() and len(fifo) == 5
    assert col.log.completed_rows == 4 + 6 + 2 + 3 + 4 == rep.injected_rows + 12
    # the prefix is not in any branch: no parent slot appears twice
    every = np.concatenate(fifo)
    assert np.unique(every).size == every.size
    st = col.store
    for b, v, won in zip(f.branches, vals, (1.0, 0.0)):
        s = np.asarray(b.slots)
        adv, ret = _hand_gae(np.asarray(v, dtype=np.float64), won)
        np.testing.assert_allclose(st.adv[s], adv, rtol=0, atol=1e-6)
        np.testing.assert_allclose(st.ret[s], ret, rtol=0, atol=1e-6)
        assert np.all(st.outcome[s] == won)
        assert st.reward[s[-1]] == won and np.all(st.reward[s[:-1]] == 0.0)
        assert st.terminal[s[-1]] and not st.terminal[s[:-1]].any()
    assert rep.masked_rows == 2


def test_the_fill_hands_the_pg_mask_to_the_learner_and_the_term_renormalises():
    col = _col()
    p = _pass()
    parent = _parent(col, 4, 1.0)
    f = _fork(_rec(parent), [("top1", 0), ("top2", 1)], [WIN, LOSS])
    for b in f.branches:
        _branch_rows(col, p, f, b, [0.5, 0.5])
    p._inject(col, [f], FK.PassReport())
    buf = RolloutBuffer(8, _space(), spaces.Discrete(11), device="cpu", gamma=GAMMA, gae_lambda=LAM,
                                    n_envs=1)
    S.fill_complete(buf, col.log, 8, current_version=3)
    m = buf.observations["fork_pg_m"].reshape(-1)
    # parent rows 1.0; each branch's fork step 0.0, its successor 1.0
    assert m.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    assert buf.observations["win_target"].reshape(-1).tolist() == [1, 1, 1, 1, 1, 1, 0, 0]
    # the learner's term (micro_step): masked SUM / KEPT count — the mean over the kept rows
    per_row = th.arange(8, dtype=th.float32)
    mt = th.as_tensor(m)
    masked = -((per_row * mt).sum() / mt.sum().clamp(min=1.0))
    assert float(masked) == pytest.approx(-float(per_row[mt > 0].mean()))


def test_a_capped_branch_is_dropped_and_a_fork_left_with_one_is_dropped_WHOLE():
    col = _col()
    p = _pass()
    parent = _parent(col, 4, 1.0)
    live0 = col.store.live_count
    f = _fork(_rec(parent), [("top1", 0), ("top2", 1)], [WIN, CAPPED])
    for b in f.branches:
        _branch_rows(col, p, f, b, [0.5, 0.5, 0.5])
    assert col.store.fork_live == 6
    rep = FK.PassReport()
    p._inject(col, [f], rep)
    assert f.dropped and rep.dropped_forks == 1 and rep.injected_rows == 0
    assert col.store.fork_live == 0 and col.store.live_count == live0
    assert len(col.log.completed) == 1


def test_the_fork_row_budget_refuses_an_alloc_past_it_and_release_returns_it():
    col = _col()
    p = _pass(budget=3)
    parent = _parent(col, 4, 1.0)
    f = _fork(_rec(parent), [("top1", 0)], [None])
    assert p._alloc(col, 3) is not None
    assert p._alloc(col, 1) is None
    f.branches[0].slots = list(np.flatnonzero(col.store.is_fork))
    p._drop(col, f, "row budget")
    assert col.store.fork_live == 0 and f.dropped == "row budget"


def test_a_parent_missing_from_the_fifo_is_refused_never_appended():
    col = _col()
    with pytest.raises(S.CollectorError, match="not in the completed FIFO"):
        col.log.insert_after(np.asarray([42]), [np.asarray([1, 2])])


def test_collected_rows_carry_the_placeholder_and_their_turn():
    st = S.RowStore(_space(), 8, fork=True)
    cols = {"obs": np.zeros((2, 2, 3), np.float32), "mask": np.ones((2, 2, 11), np.int8),
            "episode": np.zeros(2, np.int64), "dec_n": np.zeros((2, 2), np.int32),
            "turn": np.asarray([5, 9], np.int64)}
    s = st.alloc(2)
    S.write_rows(st, s, cols, np.asarray([0, 1]), label_keys=(), opp_class=np.zeros(2), actions=np.zeros(2),
                 logp=np.zeros(2), values=np.zeros(2), version=np.zeros(2), u=np.zeros(2), margin=np.zeros(2),
                 starts=np.zeros(2))
    assert np.all(st.obs["fork_pg_m"][s] == 1.0) and st.turn[s].tolist() == [5, 9]


def test_off_builds_no_fork_state():
    st = S.RowStore(_space(), 4)
    assert st.turn is None and st.is_fork is None
    with pytest.raises(S.CollectorError):
        st.mark_fork(np.asarray([0]))


def test_script_parsing_and_the_fork_commands_index():
    script = ('START {"formatid":"gen3ou","seed":"1,2,3,4","p1":{"name":"a","team":"T1"},'
              '"p2":{"name":"b","team":"T2"}}\nCHOOSE p1 move 1\nCHOOSE p2 move 2\nCHOOSE p1 switch 3\n')
    log = FK.parse_script(script)
    assert log["seed"] == "1,2,3,4" and log["teams"] == ["T1", "T2"] and len(log["cmds"]) == 3
    assert FK.p1_command_index(log["cmds"], 0) == 0 and FK.p1_command_index(log["cmds"], 1) == 2
    with pytest.raises(S.CollectorError):
        FK.p1_command_index(log["cmds"], 2)


# ---- the refusals (§14.7)
def test_a_non_indicator_terminal_is_refused():
    FK.check_terminal({"terminal_indicator": True})
    with pytest.raises(LeverConfigError, match="indicator"):
        FK.check_terminal({"terminal_indicator": False})


def test_the_decl_refuses_generator_sampling_and_the_window_trigger():
    from agents.training.rust_rollout.build import RustEnvDecl

    fd = FK.ForkDecl(fraction=0.05)
    RustEnvDecl(n_envs=2, fork=fd)
    with pytest.raises(ValueError, match="opponent-sampling keyed"):
        RustEnvDecl(n_envs=2, fork=fd, opponent_sampling="generator")
    with pytest.raises(ValueError, match="complete_game"):
        RustEnvDecl(n_envs=2, fork=fd, trigger="window")
    with pytest.raises(ValueError):
        FK.ForkDecl(fraction=0.0)


def test_off_args_build_no_decl_and_on_args_resolve_every_flag():
    a = SimpleNamespace(fork_fraction=0.0)
    assert FK.fork_decl_from_args(a) is None
    b = SimpleNamespace(fork_fraction=0.02, fork_branches=2, fork_contested_gap=0.3, fork_contested_absv=0.0,
                        fork_max_per_battle=1, fork_crn="dice", seed=7, supply_starve_cycles=None)
    d = FK.fork_decl_from_args(b)
    assert (d.fraction, d.branches, d.contested_gap, d.crn, d.seed) == (0.02, 2, 0.3, "dice", 7)


def test_the_arena_declares_the_fork_budget_at_startup():
    from agents.training.rust_rollout.build import RustEnvDecl
    from agents.training.rust_rollout.trigger import trigger_for

    trig = trigger_for("complete_game", n_envs=2, n_steps=64, micro_batch=64, target=0, band_lo=0, band_hi=0)
    off = RustEnvDecl(n_envs=2, n_steps=64, micro_batch=64)
    on = RustEnvDecl(n_envs=2, n_steps=64, micro_batch=64, fork=FK.ForkDecl(fraction=0.05))
    assert on.capacity(trig) - off.capacity(trig) == int(trig.hi)


def test_a_branch_plays_the_parents_REAL_policy_opponent_only_while_its_slot_serves_that_model():
    """§14.4: a policy route whose slot still serves the model the parent's episode began with ⇒ that
    slot at the route's sampling (`opp_real`); a reloaded slot or a bot route ⇒ the CURRENT trainee
    slot at T = 1, labelled POOL (the substitution `fork/opp_substituted` publishes)."""
    pool = SimpleNamespace(kind="policy", slot=0, stochastic=True, player="pool_player")
    stable = SimpleNamespace(kind="policy", slot=1, stochastic=False, player="stable_player")
    bot = SimpleNamespace(kind="bot", slot=None, stochastic=True, player="bot")
    served = {0: "pool:100", 1: "stable:x"}
    col = SimpleNamespace(current_slot=5, opponents=SimpleNamespace(routes=[pool, stable, bot]),
                          server=SimpleNamespace(force_greedy=False, temperature={"pool_player": 0.7}),
                          svc=SimpleNamespace(model_id=lambda s: served[s]))
    p = _pass()

    def fork(route, model_id):
        rec = FK.GameRecord(env=0, episode=0, slots=np.arange(3), run_seed=1, route=route, model_id=model_id,
                            opp_class=2, script="")
        return p._fork(col, rec, 1, 0, 1, [0, 1, 2], {"top1": 0, "top2": 1})

    f = fork(0, "pool:100")
    assert f.opp_real and f.opp_slot == 0 and f.opp_temperature == 0.7 and not f.opp_greedy
    f = fork(1, "stable:x")
    assert f.opp_real and f.opp_slot == 1 and f.opp_greedy            # a non-stochastic route stays greedy
    f = fork(0, "pool:50")                                             # the slot was reloaded since
    assert not f.opp_real and f.opp_slot == 5 and f.opp_temperature == 1.0 and not f.opp_greedy
    f = fork(2, None)                                                  # a bot route
    assert not f.opp_real and f.opp_slot == 5


def _guarded_pass(floor):
    """A pass with no handles to play and a declared fork supply floor; `pending` is empty, so a pass
    selects nothing (a DRY pass) unless a test stubs the injection."""
    p = _pass()
    p.pending, p.passes, p.last, p.rows_per_fork = [], 0, None, 0.0
    p.handles = [object()]
    p.guard = DryStreakGuard("fork", floor, emit=lambda _line: None)
    return p


def test_a_fork_pass_that_injects_NOTHING_for_its_floor_of_passes_is_FATAL_SUPPLY():
    """The fork lever's supply guard (`gen3_supply_guard_v2`) on the Rust core: the Python arm's twin of
    this test went with that arm (deletion pass L5). Without the guard a run with `--fork-fraction > 0`
    that forks nothing trains on the plain buffer for its whole life, in silence."""
    p = _guarded_pass(2)
    col = _col()
    p.run(col)
    assert p.guard.streak == 1
    with pytest.raises(LeverStarvedError, match="0 rows injected|no game ended"):
        p.run(col)


def test_a_fork_pass_that_injects_rows_resets_the_streak():
    p = _guarded_pass(3)
    col = _col()
    p.run(col)
    assert p.guard.streak == 1

    def _inject(_col, _forks, rep):
        rep.injected_rows = 7

    p._inject = _inject
    p.run(col)
    assert p.guard.streak == 0 and p.guard.total == 7
