"""Gate (a) of the fork arm's Rust core port (``designs/training/forks.md`` §14.8): COMMON RANDOM NUMBERS
on the real env core, CPU (T2 eager, the self-check build, a perturbed fresh production policy).

The parent games are played by the trainee (stream 0) against a SELF-LIKE p2 (the same trainee slot,
keyed on stream 1 at T = 1 — exactly the opponent a branch substitutes for a non-policy route), so a
branch that takes the parent's own action must BE the parent:

* ``dice_and_draws``, the parent's action, the battle's own dice ⇒ the same input-log commands, every
  trainee row byte-equal (observation, mask), the same actions and frame indices, the same outcome;
* the same dice with OTHER draw streams (``dice``) ⇒ the line DIVERGES;
* a RESEEDED dice stream with the parent's draws ⇒ the line DIVERGES (the control that proves the
  comparison can see a change).

Each FAILS on revert of the key it pins (the stream, the frame index, the run seed, the dice).
"""
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pytest

from agents.training import keyed_draw as KD
from agents.training.rust_rollout import fork as FK
from agents.training.rust_rollout import testkit as TK
from agents.training.rust_rollout.build import RustEnvDecl
from agents.training.rust_vec_env import RustVecEnv

pytestmark = [pytest.mark.sim, pytest.mark.integration]

N = 4


def fork_spaces() -> Tuple[Any, Any, Any]:
    from agents.training.rust_rollout.build import trainee_spaces
    from main.train.production_args import production_args

    args = production_args()
    args.fork_fraction = 0.05
    obs, act = trainee_spaces(args)
    assert FK.S.KEY_FORK_PG in obs.spaces, "--fork-fraction > 0 must declare the fork_pg_m key"
    return args, obs, act


class SelfLikeP2:
    """p2 = the trainee's CURRENT slot, keyed on stream 1 at T = 1 (``fork.py``'s substitute)."""

    def __init__(self) -> None:
        self.col: Any = None

    def __call__(self, cols: Any, envs: np.ndarray) -> np.ndarray:
        col = self.col
        d = col.svc.score(int(col.current_slot), cols["obs"][envs, 1], cols["mask"][envs, 1].astype(bool))
        lp = np.asarray(d.numpy()[0], dtype=np.float32).copy()
        u = KD.keyed_uniforms(col.cfg.run_seed, KD.STREAM_OPPONENT, envs, cols["episode"][envs], cols["dec_n"][envs, 1])
        a, _m = KD.keyed_actions(lp, u, 1.0)
        return a


@pytest.fixture(scope="module")
def played():
    TK.build_selfcheck()
    _args, obs, act = fork_spaces()
    decl = RustEnvDecl(n_envs=N, threads=2, front="ffi", profile="selfcheck",
                       n_steps=64, micro_batch=64, device="cpu", backend="eager", run_seed=23, gamma=1.0,
                       gae_lambda=0.8, fork=FK.ForkDecl(fraction=0.05, concurrency=4))
    p2 = SelfLikeP2()
    env = RustVecEnv(n_envs=N, observation_space=obs, action_space=act,
                     build=lambda m: TK.collector_for(m, obs, decl=decl, p2=p2))
    model = TK.fresh_model(env, n_steps=64, batch_size=64)
    col = env.startup(model)
    p2.col = col
    while len([r for r in col.fork.pending if r.script and r.slots.size >= 12]) < 3:
        col.host_step()
    yield col
    col.close()


def _parent_and_fork(col: Any, rec: Any, t: int, other: bool) -> Tuple[Any, int]:
    st = col.store
    slot = int(rec.slots[t])
    parent_action = int(st.action[slot])
    legal = [int(a) for a in np.flatnonzero(st.obs[FK.S.KEY_MASK][slot])]
    acts = {"top1": parent_action}
    if other:
        alt = next(a for a in legal if a != parent_action)
        acts["top2"] = alt
    f = col.fork._fork(col, rec, t, parent_action, acts.get("top2", parent_action), legal, acts)
    return f, parent_action


def _play(col: Any, f: Any, *, crn: str, seeds: List[Any] = (None,)) -> Any:
    fp = col.fork
    decl0, seeds0 = fp.decl, fp.seeds
    import dataclasses

    fp.decl = dataclasses.replace(decl0, crn=crn)
    fp.seeds = list(seeds)
    fp.keep_cmds = True
    try:
        rep = FK.PassReport()
        fp._play_wave(col, [f], rep)
        return f
    finally:
        fp.decl, fp.seeds, fp.keep_cmds = decl0, seeds0, False


def _cases(col: Any) -> List[Tuple[Any, int]]:
    out = []
    for rec in [r for r in col.fork.pending if r.script and r.slots.size >= 12][:3]:
        st = col.store
        for t in (1, rec.slots.size // 2):
            if int(st.obs[FK.S.KEY_MASK][int(rec.slots[t])].sum()) >= 2:
                out.append((rec, t))
    assert len(out) >= 3, "the harness must hold at least three forkable decisions"
    return out


def _release(col: Any, f: Any) -> None:
    for b in f.branches:
        if b.slots:
            col.store.release(np.asarray(b.slots, dtype=np.int64))
        b.slots = []


def test_the_parent_action_with_the_parents_keys_IS_the_parent(played):
    col = played
    st = col.store
    for rec, t in _cases(col):
        f, a = _parent_and_fork(col, rec, t, other=False)
        _play(col, f, crn="dice_and_draws")
        assert not f.dropped, f.dropped
        b = f.branches[0]
        parent = [int(s) for s in rec.slots[t:]]
        log = FK.parse_script(rec.script)
        assert b.cmds == log["cmds"], f"env {rec.env} ep {rec.episode} t {t}: the input log diverged"
        assert len(b.slots) == len(parent)
        for k in (FK.S.KEY_OBSERVATION, FK.S.KEY_MASK):
            assert np.array_equal(st.obs[k][b.slots], st.obs[k][parent]), f"{k} rows differ"
        assert np.array_equal(st.action[b.slots], st.action[parent])
        assert np.array_equal(st.dec_n[b.slots], st.dec_n[parent])
        won = 1.0 if (b.end or {}).get("winner") == 0 else 0.0
        assert won == float(st.outcome[parent[-1]])
        # the fork step is masked from the policy term, every later row is in it
        pg = st.obs[FK.S.KEY_FORK_PG][b.slots, 0]
        assert pg[0] == 0.0 and np.all(pg[1:] == 1.0)
        _release(col, f)


def test_the_same_dice_with_other_draw_streams_DIVERGES(played):
    col = played
    diverged = 0
    for rec, t in _cases(col):
        f, _a = _parent_and_fork(col, rec, t, other=False)
        _play(col, f, crn="dice")
        diverged += int(f.branches[0].cmds != FK.parse_script(rec.script)["cmds"])
        _release(col, f)
    assert diverged >= 1, "unpaired draws never moved a line: the comparison cannot see the draws"


def test_a_reseeded_dice_stream_DIVERGES(played):
    col = played
    diverged = 0
    for rec, t in _cases(col):
        f, _a = _parent_and_fork(col, rec, t, other=False)
        _play(col, f, crn="dice_and_draws", seeds=["1234,5678,9012,3456"])
        diverged += int(f.branches[0].cmds != FK.parse_script(rec.script)["cmds"])
        _release(col, f)
    assert diverged >= 1, "a reseeded dice stream never moved a line: the comparison cannot see the dice"


def test_a_sibling_action_shares_the_prefix_and_the_frame_keys(played):
    """Two branches of one fork: row 0 is the SAME state (the prefix counted once, the fork step per
    branch with its own action), and the k-th p1 decision of every branch is keyed on the same frame
    index ladder (the sibling streams are aligned AT the fork)."""
    col = played
    st = col.store
    rec, t = _cases(col)[0]
    f, a = _parent_and_fork(col, rec, t, other=True)
    _play(col, f, crn="dice_and_draws")
    b0, b1 = f.branches
    assert np.array_equal(st.obs[FK.S.KEY_OBSERVATION][b0.slots[0]], st.obs[FK.S.KEY_OBSERVATION][b1.slots[0]])
    assert int(st.action[b0.slots[0]]) == a and int(st.action[b1.slots[0]]) != a
    assert int(st.dec_n[b0.slots[0]]) == int(st.dec_n[b1.slots[0]]) == int(st.dec_n[int(rec.slots[t])])
    assert int(st.start[b0.slots[0]]) == 1 and int(st.start[b1.slots[0]]) == 1
    _release(col, f)


@pytest.mark.slow  # 43 s on a quiet box (2026-10-08): the tier budget enforces 30 s
def test_the_fork_phase_runs_inside_collect_and_the_lifecycle_stays_clean():
    """Gate (d) at the collector level: ``collect`` = play -> FORK -> fill with the arm ON, across
    updates (a weight move + ``after_update``): branch rows reach the learner's buffer (``fork_pg_m`` 0
    on fork steps, every update exactly the target rows, every row labelled), the ``fork/*`` family is
    published, and every declared-lifecycle counter (core + T2) stays 0 — nothing acquired after the
    freeze, the playout handles included."""
    TK.build_selfcheck()
    _args, obs, act = fork_spaces()
    n_steps = 64
    decl = RustEnvDecl(n_envs=N, threads=2, front="ffi", profile="selfcheck",
                       n_steps=n_steps, micro_batch=n_steps, device="cpu", backend="eager", run_seed=29, gamma=1.0,
                       gae_lambda=0.8, fork=FK.ForkDecl(fraction=0.05, concurrency=4))
    p2 = SelfLikeP2()
    env = RustVecEnv(n_envs=N, observation_space=obs, action_space=act,
                     build=lambda m: TK.collector_for(m, obs, decl=decl, p2=p2))
    model = TK.fresh_model(env, n_steps=n_steps, batch_size=n_steps)
    col = env.startup(model)
    p2.col = col
    cb = TK.NullCallback()
    injected = masked = 0
    seen = set()
    try:
        for k in range(4):
            if k:
                TK.perturb_weights(model, k)
                col.after_update(model)
            assert col.collect(model, cb, model.rollout_buffer)
            buf = model.rollout_buffer
            assert TK.spaces_rows(buf) == N * n_steps
            assert (buf.observations["win_mask"] == 1.0).all()
            m = buf.observations[FK.S.KEY_FORK_PG].reshape(-1)
            assert set(np.unique(m).tolist()) <= {0.0, 1.0}
            masked += int((m == 0.0).sum())
            rep = col.fork.last
            injected += rep.injected_rows
            seen |= set((model._fork_metrics or {}).keys())
        assert injected > 0 and masked > 0, (injected, masked)
        assert {"rate", "branch_share", "tie_rate", "pairwise_acc", "sim_steps_share", "opp_substituted",
                "own_rows", "injected_rows", "dropped_forks"} <= seen, seen
        assert col.store.fork_live <= col.fork.row_budget
        assert all(v == 0 for v in col.check_lifecycle().values())
    finally:
        env.close()
