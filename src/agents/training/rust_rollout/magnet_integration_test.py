"""The trailing magnet on the real env core (CE-3d / T56, `gen3_magnet_kl_v1`; CPU: T2 eager, the self-check build,
a perturbed fresh production policy, p2 a seeded random policy) — `agents.training.magnet` through the collector.

* **the column is the MAGNET's policy, legal actions only**: while the magnet is the weights that played the rollout
  the column's entry at the played action is the behaviour log-probability T2 served (and 0.0 on every illegal
  action); after the learner moves and the cadence has NOT passed, it is still the old weights' (the frozen copy,
  checked against a learner forward under those weights), not the weights that played;
* **the refresh at the boundary**: the ``after_update`` at which the magnet's age REACHES `--magnet-refresh-steps`
  (equal counts: "at or after") makes the magnet the weights just stepped to and reloads T2's magnet slot, and the
  next column is the new weights';
* **a resume**: the checkpoint's sidecar, read back by `attach` into a NEW learner and a NEW collector, serves the
  same column bit for bit — and a collector whose slot was not handed the magnet does not;
* **a whole ``learn()``** with the lever ON trains through refreshes under ``fatal`` K9(b), records
  ``train/magnet_kl`` and ``magnet/*``, and is a different learner from the flag-OFF golden.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import pytest
import torch as th

from agents.training import magnet as M
from agents.training.rust_rollout import testkit as TK
from agents.training.rust_rollout.build import RustEnvDecl
from agents.training.rust_rollout.consistency import probe_forward
from agents.training.rust_vec_env import RustVecEnv

pytestmark = [pytest.mark.sim, pytest.mark.integration, pytest.mark.slow]  # real-core rollouts: module fixtures

N, N_STEPS = 3, 32
ROLLOUT = N * N_STEPS
BETA = 0.05
FAR = 10 ** 9                 # a cadence no test rollout reaches


@pytest.fixture(scope="module")
def spaces():
    TK.build_selfcheck()
    return TK.production_spaces()


def _setup(spaces, *, refresh: int, seed: int = 11, n_epochs: int = 1, attach: Any = None):
    """A magnet run's learner and collector, in the order a launch builds them: the model, `attach` (``attach`` runs
    in its place when given: the resume), then the env's startup (which declares T2's magnet slot)."""
    _args, obs, act = spaces
    decl = RustEnvDecl(n_envs=N, threads=2, front="ffi", profile="selfcheck", n_steps=N_STEPS, micro_batch=N_STEPS,
                       device="cpu", backend="eager", run_seed=seed, gamma=1.0, gae_lambda=0.8, magnet=True)
    env = RustVecEnv(n_envs=N, observation_space=obs, action_space=act,
                     build=lambda m: TK.collector_for(m, obs, decl=decl))
    model = TK.fresh_model(env, n_steps=N_STEPS, batch_size=N_STEPS, n_epochs=n_epochs)
    model.magnet_kl_coef, model.magnet_refresh_steps = BETA, int(refresh)
    if attach is None:
        M.attach(model, model_path=None, same_run=False, saved_coef=0.0, say=lambda _m: None)
    else:
        attach(model)
    col = env.startup(model)
    return env, model, col


def _rollout(model, col) -> Dict[str, Any]:
    """What one filled rollout holds: the column, the behaviour log-probs, the rows' versions, the legal mask."""
    buf = model.rollout_buffer
    column = np.array(buf.magnet_logp, copy=True)
    acts = buf.actions.reshape(N_STEPS, N).astype(np.int64)
    return dict(column=column, at_action=np.take_along_axis(column, acts[..., None], axis=-1)[..., 0].astype(np.float64),
                mu=np.array(buf.log_probs, dtype=np.float64).reshape(N_STEPS, N),
                current=(np.asarray(model._rust_row_versions).reshape(N_STEPS, N) == col.version),
                legal=np.array(buf.action_masks).reshape(N_STEPS, N, -1) > 0, steps=int(model.num_timesteps))


def _learner_forward(model) -> np.ndarray:
    """The learner's own forward on the buffer's rows under the weights it holds NOW: log π of the played actions."""
    buf = model.rollout_buffer
    obs = {k: v.reshape((ROLLOUT,) + v.shape[2:]) for k, v in buf.observations.items()}
    acts = th.as_tensor(buf.actions.reshape(-1)).long()
    masks = th.as_tensor(buf.action_masks.reshape(ROLLOUT, -1))
    new, _full = probe_forward(model, obs, acts, masks)
    return new.reshape(N_STEPS, N)


def _magnet_forward(model) -> np.ndarray:
    """The same forward UNDER THE MAGNET'S WEIGHTS: log π_mag of the played actions."""
    before = [p.detach().clone() for p in model.policy.parameters()]
    with model._magnet.swapped(model.policy):
        new = _learner_forward(model)
    assert all(th.equal(a, b) for a, b in zip(before, model.policy.parameters()))     # the swap restored the learner
    return new


@pytest.fixture(scope="module")
def run(spaces):
    """R0 (magnet = the weights that play it); a weight move and an ``after_update`` BEFORE the cadence; R1; a second
    move and the ``after_update`` exactly AT the cadence; R2; a third move (so the weights a checkpoint holds are not
    the magnet's). A rollout's env-step count is the core's (the decisions it played, not ``n_steps x n_envs``), so
    the cadence is set from the counter itself: the age the magnet has at the second boundary."""
    env, model, col = _setup(spaces, refresh=FAR)
    cb = TK.NullCallback()
    out: Dict[str, Any] = dict(env=env, model=model, col=col, loads=[])
    real_load = col.svc.load

    def spy(slot, policy, model_id):
        out["loads"].append((int(slot), str(model_id)))
        return real_load(slot, policy, model_id)
    col.svc.load = spy
    try:
        mag = model._magnet
        assert col.magnet_slot is not None and mag.step == 0
        assert col.collect(model, cb, model.rollout_buffer)                       # R0
        out["r0"] = _rollout(model, col)
        TK.perturb_weights(model, 1)
        col.after_update(model)                                                   # one rollout old: NOT due
        out["after_u0"] = (mag.step, mag.refreshes, mag.equals(model.policy))
        assert col.collect(model, cb, model.rollout_buffer)                       # R1: played by v1, magnet still v0
        out["r1"] = _rollout(model, col)
        out["r1_magnet_forward"] = _magnet_forward(model)
        TK.perturb_weights(model, 2)
        mag.refresh_steps = int(model.num_timesteps)                              # age == the cadence, exactly: DUE
        col.after_update(model)
        out["after_u1"] = (mag.step, mag.refreshes, mag.equals(model.policy))
        assert col.collect(model, cb, model.rollout_buffer)                       # R2: played by v2, magnet v2
        out["r2"] = _rollout(model, col)
        out["r2_learner_forward"] = _learner_forward(model)                       # v2, the weights the magnet became
        TK.perturb_weights(model, 3)
        col.after_update(model)                                                   # one rollout old: NOT due
        out["after_u2"] = (mag.step, mag.refreshes, mag.equals(model.policy))
    except BaseException:
        env.close()
        raise
    yield out
    env.close()


def test_the_magnet_slot_is_the_last_T2_slot_and_never_the_trainees(run):
    col = run["col"]
    assert col.magnet_slot not in col.trainee_slots and col.magnet_slot not in col.extra_slots
    assert col.magnet_slot == len(col.svc._slots) - 1


def test_the_column_is_the_magnets_log_probs_over_the_LEGAL_actions(run):
    r0 = run["r0"]
    col, legal = r0["column"], r0["legal"]
    assert col.shape == legal.shape and col.dtype == np.float32
    assert (~legal).any() and (col[~legal] == 0.0).all()                          # exactly 0.0 on every illegal action
    assert np.isfinite(col).all() and (col[legal] <= 0.0).all()
    np.testing.assert_allclose(np.where(legal, np.exp(col.astype(np.float64)), 0.0).sum(-1), 1.0, atol=1e-5)
    # the magnet IS the weights that played R0: its log-prob of the played action is the behaviour log-prob T2
    # served (the same fp32 function; only the bucket padding differs)
    cur = r0["current"]
    assert cur.sum() > ROLLOUT // 2
    np.testing.assert_allclose(r0["at_action"][cur], r0["mu"][cur], atol=1e-5)


def test_the_magnet_is_FROZEN_between_refreshes(run):
    """After the learner moved and before the cadence, the column is the OLD weights' policy: it is what a learner
    forward under the magnet's weights gives, and it is not the policy that played the rollout."""
    assert run["after_u0"] == (0, 0, False)                                       # not refreshed, no longer the weights
    r1 = run["r1"]
    cur = r1["current"]                                                           # the rows v1 played (a fill also
    assert cur.sum() >= 8                                                         # carries rows of games v0 began)
    assert np.abs(r1["at_action"][cur] - r1["mu"][cur]).max() > 1e-3              # NOT the behaviour policy (v1)
    np.testing.assert_allclose(r1["at_action"], run["r1_magnet_forward"], atol=1e-4)      # the v0 weights' policy


def test_the_magnet_refreshes_at_the_first_boundary_at_or_after_the_cadence(run):
    r0, r1, r2 = run["r0"], run["r1"], run["r2"]
    t1 = r1["steps"]
    assert 0 < r0["steps"] < t1 < r2["steps"] and r2["steps"] - t1 < t1          # the third boundary is NOT due
    assert run["after_u1"] == (t1, 1, True)                                       # the weights just stepped to
    assert run["after_u2"] == (t1, 1, False)                                      # and frozen again
    slot = run["col"].magnet_slot
    assert [m for s, m in run["loads"] if s == slot] == [f"magnet@{t1}"]          # ONE reload of T2's magnet slot
    np.testing.assert_allclose(r2["at_action"], run["r2_learner_forward"], atol=1e-4)     # the NEW weights' policy,
    cur = r2["current"]                                                                   # on every row of the fill
    assert cur.any() and not cur.all()
    np.testing.assert_allclose(r2["at_action"][cur], r2["mu"][cur], atol=1e-5)    # = what T2 served the rows v2 played
    assert np.abs(r2["at_action"][~cur] - r2["mu"][~cur]).max() > 1e-3            # and not what older weights served


def test_the_magnet_survives_a_resume_into_a_new_learner_and_collector(run, spaces, tmp_path):
    from main.train.run_io import commit_checkpoint
    model, col = run["model"], run["col"]
    mag = model._magnet
    assert not mag.equals(model.policy)                                           # the checkpoint's weights are v3
    zip_path = str(tmp_path / "checkpoints" / f"checkpoint_{int(model.num_timesteps)}_steps.zip")
    final = commit_checkpoint(model, str(tmp_path), zip_path, pointer=lambda d, rel: None)
    assert os.path.isfile(M.sidecar_path(final))
    want = M.fill_column(col.svc, col.magnet_slot, model.rollout_buffer)          # what the saved run's slot serves

    said: List[str] = []

    def resume(m):                                                                # the checkpoint's weights and step
        m.policy.load_state_dict(model.policy.state_dict())
        m.num_timesteps = int(model.num_timesteps)
        M.attach(m, model_path=final, same_run=True, saved_coef=BETA, say=said.append)
    env2, model2, col2 = _setup(spaces, refresh=mag.refresh_steps, attach=resume)
    try:
        mag2 = model2._magnet
        assert (mag2.step, mag2.refresh_steps) == (mag.step, mag.refresh_steps) and "sidecar" in said[0]
        assert set(mag2.state) == set(mag.state) and all(th.equal(mag2.state[k], mag.state[k]) for k in mag.state)
        # the new collector's slot was built from the learner's CURRENT weights: until startup hands it the magnet
        # it serves the wrong policy
        cold = M.fill_column(col2.svc, col2.magnet_slot, model.rollout_buffer)
        assert np.abs(cold - want).max() > 1e-3
        assert M.load_slot(col2.svc, col2.magnet_slot, model2) is True
        got = M.fill_column(col2.svc, col2.magnet_slot, model.rollout_buffer)
        assert np.array_equal(got, want)                                          # bit for bit
        # and its age survived: it is as old as the saved run's, and due at the same env step
        now = int(model2.num_timesteps)
        assert mag2.age(now) == mag.age(now) == now - mag.step > 0 and not mag2.due(now)
        assert mag2.due(mag.step + mag.refresh_steps) and not mag2.due(mag.step + mag.refresh_steps - 1)
    finally:
        env2.close()


def test_a_whole_learn_with_the_magnet_trains_through_refreshes_and_records_its_meters(spaces):
    cadence = 2 * ROLLOUT
    env, model, col = _setup(spaces, refresh=cadence, n_epochs=1)
    model.behaviour_check = "fatal"
    boundaries: List[int] = []                             # the env-step counter at each update boundary
    fills: List[int] = []                                  # and at each fill
    real_after, real_fill = col.after_update, col._fill

    def after_update(m):
        boundaries.append(int(m.num_timesteps))
        return real_after(m)

    def fill(m, callback, snap):
        fills.append(int(m.num_timesteps))
        return real_fill(m, callback, snap)
    col.after_update, col._fill = after_update, fill
    try:
        dumps = TK.record_dumps(model)
        model.learn(total_timesteps=8 * ROLLOUT)
        updates = int(model._n_updates)                    # n_epochs = 1
        kls = [d[M.TAG_KL] for d in dumps if M.TAG_KL in d]
        assert len(kls) == updates == len(fills) >= 4, (kls, updates, fills)
        assert all(np.isfinite(k) and k >= 0.0 for k in kls) and max(kls) > 1e-6, kls   # a real, non-zero KL
        # an update boundary is the top of the NEXT collection (the loop's `after_update`), at the env step the
        # last fill ended on
        assert boundaries == fills[:-1]
        # the cadence, replayed: the magnet is set at the first boundary whose age reaches it, and a fill's age is
        # the env steps since
        step, refreshes, want_age, want_ref, due = 0, 0, [], [], []
        for k, t in enumerate(fills):
            if k:
                due.append(fills[k - 1] - step >= cadence)
                if due[-1]:
                    step, refreshes = fills[k - 1], refreshes + 1
            want_age.append(float(t - step))
            want_ref.append(float(refreshes))
        assert any(due) and not all(due), (fills, due)     # it crossed a boundary that refreshed and one that did not
        assert [d[M.TAG_AGE] for d in dumps if M.TAG_AGE in d] == want_age
        assert [d[M.TAG_REFRESHES] for d in dumps if M.TAG_REFRESHES in d] == want_ref
        assert all(d[M.TAG_FORWARD_MS] >= 0.0 for d in dumps if M.TAG_AGE in d)
        assert (model._magnet.step, model._magnet.refreshes) == (step, refreshes)
    finally:
        env.close()
