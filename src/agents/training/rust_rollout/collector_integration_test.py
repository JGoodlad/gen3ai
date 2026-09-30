"""The collector on the real env core (M5 Lane G): the COMPLETE-GAME buffer's invariants, through both
front ends, on CPU (T2 eager, the self-check build, a perturbed fresh production policy, p2 a seeded
random policy on an external route).

* every update consumes EXACTLY the target rows, all from COMPLETED games, every row labelled with its
  own game's outcome (``win_mask`` 1) and carrying its game's GAE;
* NO ROW IS DROPPED: every row the collector played is consumed, carried at the FIFO head, or still in
  a game in play (a cut game — none here — would be counted);
* after an "update" (a weight move + ``after_update``: a T2 LOAD, the version bumped) the next buffer
  holds rows of BOTH versions and the fill's age histogram says so;
* the TRAINEE'S KEYED DRAW replays: every played action is ``keyed_actions(served log-probs, u)`` with
  ``u`` recomputed from the row's key, and the stored behaviour log-prob is the served log-prob of
  that action;
* every declared-lifecycle counter (core + T2) stays 0.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

from agents.training import keyed_draw as KD
from agents.training.rust_rollout import testkit as TK
from agents.training.rust_rollout.build import RustEnvDecl
from agents.training.rust_vec_env import RustVecEnv

pytestmark = [pytest.mark.sim, pytest.mark.integration]

N, N_STEPS = 3, 32


@pytest.fixture(scope="module")
def spaces():
    TK.build_selfcheck()
    return TK.production_spaces()


class Recorder:
    def __init__(self) -> None:
        self.rows: Dict[int, Dict[str, Any]] = {}
        self.played = 0

    def episode_started(self, col: Any, moved: np.ndarray) -> None:
        pass

    def p2(self, col: Any) -> None:
        pass

    def trainee(self, col, envs, logp, act, blogp, value, u, slots) -> None:
        c = col.cols
        for j, e in enumerate(envs.tolist()):
            self.rows[int(slots[j])] = dict(env=e, episode=int(c["episode"][e]), dec_n=int(c["dec_n"][e, 0]),
                                            logp=logp[j].copy(), act=int(act[j]), blogp=float(blogp[j]),
                                            u=float(u[j]))
            self.played += 1


def _run(front: str, spaces) -> Dict[str, Any]:
    _args, obs, act = spaces
    decl = RustEnvDecl(n_envs=N, threads=2, front=front, profile="selfcheck", trigger="complete_game",
                       n_steps=N_STEPS, micro_batch=N_STEPS, device="cpu", backend="eager", run_seed=11,
                       gamma=1.0, gae_lambda=0.8)
    env = RustVecEnv(n_envs=N, observation_space=obs, action_space=act,
                     build=lambda m: TK.collector_for(m, obs, decl=decl))
    model = TK.fresh_model(env, n_steps=N_STEPS, batch_size=N_STEPS)
    col = env.startup(model)
    rec = Recorder()
    col.hooks.append(rec)
    cb = TK.NullCallback()
    consumed = 0
    ages: List[int] = []
    fills = []
    try:
        for k in range(4):
            if k:
                TK.perturb_weights(model, k)
                col.after_update(model)
            assert col.collect(model, cb, model.rollout_buffer)
            buf = model.rollout_buffer
            assert TK.spaces_rows(buf) == N * N_STEPS
            assert (buf.observations["win_mask"] == 1.0).all()
            assert set(np.unique(buf.observations["win_target"]).tolist()) <= {0.0, 1.0}
            assert np.allclose(buf.returns, buf.advantages + buf.values, atol=0)
            consumed += TK.spaces_rows(buf)
            ages += TK.ages_of(model)
            fills.append(col.last_fill)
        life = col.check_lifecycle()
        out = dict(col=col, rec=rec, consumed=consumed, ages=ages, fills=fills, cb=cb, life=life,
                   stats=col.stats, log=col.log)
    finally:
        env.close()
    return out


@pytest.mark.parametrize("front", ["ffi", "proc"])
def test_the_complete_game_collector_keeps_every_row_and_labels_it(spaces, front):
    r = _run(front, spaces)
    log, rec = r["log"], r["rec"]
    # accounting: played = consumed + carried (completed, not yet consumed) + in play + cut
    assert rec.played == r["consumed"] + log.completed_rows + log.in_progress_rows() + log.rows_cut, (
        rec.played, r["consumed"], log.completed_rows, log.in_progress_rows(), log.rows_cut)
    assert log.rows_cut == 0 and r["stats"].quarantines == 0
    # the updates: version 0 only in the first buffer; after an update a buffer's head is the rows
    # completed BEFORE it (carried, now one version old) and games straddling it — the later buffers
    # hold both current and older rows, and the fill's age histogram counts them
    assert r["fills"][0].age_hist == {0: N * N_STEPS}
    later: Dict[int, int] = {}
    for f in r["fills"][1:]:
        assert sum(f.age_hist.values()) == N * N_STEPS
        for a_, v in f.age_hist.items():
            later[a_] = later.get(a_, 0) + v
    assert later.get(0, 0) > 0 and sum(v for a_, v in later.items() if a_ > 0) > 0, later
    # every ended game reported its outcome to the callbacks
    assert len(r["cb"].infos) == r["stats"].games_ended > 0
    assert all({"win_outcome", "win_draw", "opponent_class", "episode"} <= set(i) for i in r["cb"].infos)
    assert all(v == 0 for v in r["life"].values()), r["life"]


def test_the_trainees_keyed_draw_replays_from_its_key(spaces):
    r = _run("ffi", spaces)
    seed = r["col"].cfg.run_seed
    rows = list(r["rec"].rows.values())
    assert len(rows) > 200
    env = np.array([x["env"] for x in rows])
    ep = np.array([x["episode"] for x in rows])
    dec = np.array([x["dec_n"] for x in rows])
    u = KD.keyed_uniforms(seed, KD.STREAM_TRAINEE, env, ep, dec)
    assert np.array_equal(u, np.array([x["u"] for x in rows]))
    logp = np.stack([x["logp"] for x in rows])
    a, _m = KD.keyed_actions(logp, u)
    assert np.array_equal(a, np.array([x["act"] for x in rows]))
    assert np.array_equal(logp[np.arange(len(rows)), a].astype(np.float32),
                          np.array([x["blogp"] for x in rows], dtype=np.float32))
    # the draws are not degenerate: more than one action is played from the same kind of state
    assert len(set(a.tolist())) > 3


def test_per_game_version_pinning_plays_each_game_on_one_version(spaces):
    """``--version-pinning per_game`` (declared, OFF by default): after an update the new weights LOAD into
    a FREE trainee slot; every game in progress keeps the slot — the version — it started with, so every
    completed game's rows carry ONE version; a slot a game still plays is never reloaded."""
    _args, obs, act = spaces
    decl = RustEnvDecl(n_envs=N, threads=2, front="ffi", profile="selfcheck", trigger="complete_game",
                       n_steps=N_STEPS, micro_batch=N_STEPS, device="cpu", backend="eager", run_seed=21,
                       version_pinning=True, trainee_slots=3)
    env = RustVecEnv(n_envs=N, observation_space=obs, action_space=act,
                     build=lambda m: TK.collector_for(m, obs, decl=decl))
    model = TK.fresh_model(env, n_steps=N_STEPS, batch_size=N_STEPS)
    col = env.startup(model)
    rec = Recorder()
    col.hooks.append(rec)
    cb = TK.NullCallback()
    try:
        for k in range(4):
            if k:
                TK.perturb_weights(model, k)
                col.after_update(model)
                # the loaded slot is one no game with rows in play is pinned to
                assert all(int(col.env_slot[e]) != col.current_slot for e in range(N) if col.log.cur[e])
            assert col.collect(model, cb, model.rollout_buffer)
        # every game still resident in the arena (in play, or completed and carried) is ONE version
        versions: Dict[tuple, set] = {}
        for e in range(N):
            if col.log.cur[e]:
                vs = {int(col.store.version[s]) for s in col.log.cur[e]}
                assert len(vs) == 1, (e, vs)
        for g in list(col.log.completed):
            vs = {int(v) for v in col.store.version[g]}
            envs = {int(x) for x in col.store.env[g]}
            eps = {int(x) for x in col.store.episode[g]}
            assert len(vs) == 1 and len(envs) == 1 and len(eps) == 1, (vs, envs, eps)
            versions[(envs.pop(), eps.pop())] = vs
        assert versions
        assert col.stats.updates == 3 and all(v == 0 for v in col.check_lifecycle().values())
    finally:
        env.close()


def test_a_core_child_death_is_a_counted_budgeted_respawn(spaces):
    """The process front end's child is SIGKILLed mid-rollout: the collector recovers (F-LB-1 / F-LB-2) —
    the games in progress are CUT and counted, the run seed moves to a new segment, the core is RESET,
    and the rollout completes; the respawn is counted in PROC_SPAWNS_AFTER_FREEZE and one more than the
    declared budget is a typed LifecycleViolation."""
    import os
    import signal

    from agents.training.rust_rollout.collector import LifecycleViolation

    _args, obs, act = spaces
    decl = RustEnvDecl(n_envs=N, threads=2, front="proc", profile="selfcheck", trigger="complete_game",
                       n_steps=N_STEPS, micro_batch=N_STEPS, device="cpu", backend="eager", run_seed=31,
                       respawn_budget=1)
    env = RustVecEnv(n_envs=N, observation_space=obs, action_space=act,
                     build=lambda m: TK.collector_for(m, obs, decl=decl))
    model = TK.fresh_model(env, n_steps=N_STEPS, batch_size=N_STEPS)
    col = env.startup(model)
    cb = TK.NullCallback()
    try:
        for _ in range(20):
            col.host_step()
        seed0 = col.cfg.run_seed
        os.kill(col.core.pid, signal.SIGKILL)
        assert col.collect(model, cb, model.rollout_buffer)
        assert col.stats.respawns == 1 and col.log.rows_cut > 0 and col.cfg.run_seed != seed0
        assert col.check_lifecycle()["PROC_SPAWNS_AFTER_FREEZE"] == 1
        os.kill(col.core.pid, signal.SIGKILL)
        with pytest.raises(LifecycleViolation, match="budget"):
            col.collect(model, cb, model.rollout_buffer)
    finally:
        env.close()
