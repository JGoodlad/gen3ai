"""X14 OVERLAP on the real env core (CPU: T2 eager, the self-check build, a perturbed fresh production policy, p2 a
seeded random policy on an external route) — `rust_rollout/overlap.py`.

* **the STALENESS BOOKKEEPING**: a rollout the worker plays while the learner moves from version k to k + 1 holds
  ONLY rows of version <= k (T2 is not reloaded mid-rollout), and once the next ``after_update`` has run its rows
  are age >= 1; the same schedule played serially holds rows of version k + 1. Revert the worker (or reload T2
  under it) and the version histogram moves;
* **K9(b) under the snapshot**: the learner's own probe judges NONE of the overlapped rows (none is at the current
  version), the behaviour-snapshot judge judges them and passes under ``fatal``, and a snapshot that is NOT the
  weights that played them (one slot perturbed) is a typed `BehaviourMismatch` — the judge is what checks them;
* **the replay**: the callbacks see the same ``step`` events (count and infos) and ``num_timesteps`` moves by
  exactly the decisions played;
* **a whole ``learn()``** with the flag ON (``fatal`` K9(b)) trains through overlapped updates, records the
  ``overlap/*`` and ``behaviour_snapshot/*`` tags, and is DETERMINISTIC (two runs, same bytes).
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List

import numpy as np
import pytest

from agents.training.rust_rollout import testkit as TK
from agents.training.rust_rollout.build import RustEnvDecl
from agents.training.rust_vec_env import RustVecEnv

pytestmark = [pytest.mark.sim, pytest.mark.integration, pytest.mark.slow]  # ship gate (2026-10-19, the 8 s bar): 56.5 s + 23.4 s beside a live run; module fixtures, so the whole file

N, N_STEPS = 3, 32


@pytest.fixture(scope="module")
def spaces():
    TK.build_selfcheck()
    return TK.production_spaces()


def _setup(spaces, *, overlap: bool, seed: int = 11, n_epochs: int = 1):
    _args, obs, act = spaces
    decl = RustEnvDecl(n_envs=N, threads=2, front="ffi", profile="selfcheck", n_steps=N_STEPS,
                       micro_batch=N_STEPS, device="cpu", backend="eager", run_seed=seed, gamma=1.0, gae_lambda=0.8,
                       **({"overlap": True} if overlap else {}))
    env = RustVecEnv(n_envs=N, observation_space=obs, action_space=act,
                     build=lambda m: TK.collector_for(m, obs, decl=decl))
    model = TK.fresh_model(env, n_steps=N_STEPS, batch_size=N_STEPS, n_epochs=n_epochs)
    col = env.startup(model)
    return env, model, col


def _versions(model) -> List[int]:
    return np.asarray(model._rust_row_versions).reshape(-1).tolist()


def _schedule(spaces, *, overlap: bool) -> Dict[str, Any]:
    """R0 serial; "update" 0 (a seeded weight move) alone; R1; "update" 1 — with the worker playing R2 during it when
    ``overlap`` (the learner moves to version 2 while T2 still serves version 1); R2 delivered after the next
    ``after_update``."""
    env, model, col = _setup(spaces, overlap=overlap)
    cb = TK.NullCallback()
    try:
        assert col.collect(model, cb, model.rollout_buffer)                      # R0, version 0
        TK.perturb_weights(model, 1)                                              # update 0 (alone)
        col.after_update(model)                                                   # T2 <- v1
        assert col.collect(model, cb, model.rollout_buffer)                       # R1
        ov = col.overlap
        prox_r1 = None
        if overlap:
            v1 = np.asarray(model._rust_row_versions) == 1
            prox_r1 = (np.asarray(model._overlap_prox)[v1], model.rollout_buffer.log_probs[v1].astype(np.float64))
            assert ov.warm and not ov.active
            steps0, t0 = cb.steps, int(model.num_timesteps)
            ov.begin()
            TK.perturb_weights(model, 2)                                          # update 1 — DURING the play
            played = ov.join()
            assert played.ready and played.steps
            assert cb.steps == steps0 and int(model.num_timesteps) == t0          # nothing delivered by the worker
            col.after_update(model)                                               # T2 <- v2, snapshot v2
            assert col.collect_after_overlap(model, cb, model.rollout_buffer, played)
            assert int(model.num_timesteps) - t0 == sum(k for k, _ in played.steps)
            prox_r2 = np.asarray(model._overlap_prox)
        else:
            TK.perturb_weights(model, 2)
            col.after_update(model)
            assert col.collect(model, cb, model.rollout_buffer)                   # R2, serial
        out = dict(model=model, col=col, env=env, cb=cb, versions=_versions(model), version=col.version, prox_r1=prox_r1,
                   prox_r2=prox_r2 if overlap else None)
    except BaseException:
        env.close()
        raise
    return out


@pytest.fixture(scope="module")
def on_run(spaces):
    r = _schedule(spaces, overlap=True)
    yield r
    r["env"].close()


@pytest.fixture(scope="module")
def off_run(spaces):
    r = _schedule(spaces, overlap=False)
    yield r
    r["env"].close()


def test_overlapped_rows_are_one_version_stale_and_the_serial_control_is_not(on_run, off_run):
    on, off = on_run, off_run
    assert on["version"] == off["version"] == 2
    # the overlapped rollout: T2 served version 1 throughout (and a game started at version 0 may carry over)
    assert max(on["versions"]) == 1 and min(on["versions"]) >= 0, sorted(set(on["versions"]))
    ages = [2 - v for v in on["versions"]]
    assert min(ages) >= 1 and ages.count(1) > len(ages) // 2, sorted(set(ages))
    # the serial control: version 2 rows are played after the load, so most rows are current
    assert max(off["versions"]) == 2 and off["versions"].count(2) > 0
    # the callbacks saw every game that ended, in both
    assert on["cb"].infos and len(on["cb"].infos) == on["col"].stats.games_ended


def test_the_fill_serves_the_proximal_log_probs_from_t2_at_the_learners_weights(on_run):
    """pi_prox at a fill is T2's trainee slot after ``after_update``: on the rows played at the version T2 holds it
    is the function mu was served by (equal to fp32 rounding — only the bucket padding differs); on the overlapped
    rows (played at version 1, the slot now at 2) it is a different policy's."""
    prox, mu = on_run["prox_r1"]
    assert prox.size > 0
    np.testing.assert_allclose(prox, mu, atol=1e-5)
    stale = np.abs(on_run["prox_r2"] - on_run["model"].rollout_buffer.log_probs.astype(np.float64))
    assert stale.max() > 1e-3


def test_k9b_judges_the_overlapped_rows_under_their_behaviour_snapshot(on_run):
    import torch as th

    from agents.training.rust_rollout.consistency import BehaviourMismatch, behaviour_probe
    from agents.training.rust_rollout.overlap import judge_snapshot
    from agents.training.train_logger import configure

    r = on_run
    model, col = r["model"], r["col"]
    sl = col.overlap.snapshots
    i = sl.versions.index(1)
    kept = [t.clone() for t in sl._slots[i]]
    try:
        model.set_logger(configure(None, []))
        model.behaviour_check = "fatal"
        model._current_progress_remaining = 1.0
        own = behaviour_probe(model)                                  # the learner (version 2): nothing is current
        assert own is not None and own["behaviour/rows_current"] == 0
        snap = judge_snapshot(model, col.overlap)                     # version 1's snapshot: its rows are judged
        assert snap is not None and snap["behaviour_snapshot/rows_current"] > 0
        assert snap["behaviour_snapshot/max_abs_dlogp_current"] < 1e-4
        assert not any(k.startswith("staleness/") for k in snap)
        before = [p.detach().clone() for p in model.policy.parameters()]
        # a snapshot that is NOT what played the rows: the judge FAILS (so it is what checks them)
        with th.no_grad():
            for t in sl._slots[i]:
                if t.is_floating_point():
                    t.add_(1e-2)
        with pytest.raises(BehaviourMismatch, match="BEHAVIOUR SNAPSHOT"):
            judge_snapshot(model, col.overlap)
        # the swap restored the learner bit-exactly, also on the exception
        assert all(th.equal(a, b) for a, b in zip(before, model.policy.parameters()))
    finally:
        with th.no_grad():
            for t, k in zip(sl._slots[i], kept):
                t.copy_(k)


def _weights_hash(model) -> str:
    h = hashlib.sha256()
    for k, v in model.policy.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _learn(spaces, *, overlap: bool, rollouts: int = 5, n_epochs: int = 2):
    from agents.training.rust_rollout.testkit import record_dumps

    env, model, col = _setup(spaces, overlap=overlap, n_epochs=n_epochs)
    model.behaviour_check = "fatal"
    try:
        dumps = record_dumps(model)
        model.learn(total_timesteps=rollouts * N * N_STEPS)
        return dict(model=model, col=col, dumps=dumps, hash=_weights_hash(model))
    finally:
        env.close()


#: THE OFF GOLDEN: `_learn(overlap=False, rollouts=3, n_epochs=1)`'s final-weights hash at the DEFAULT tau solve. It
#: pins what the flag-OFF loop (collect, update, the K9(b) probe, the fill) trains. A DELIBERATE change to what a serial
#: update trains moves it: re-record by printing `_learn(...)["hash"]` at the new commit (both constants: the failing
#: assert prints the hash), say why here and in the commit, and re-stamp NUMERICS_ANCHOR below.
#:
#: RE-RECORDED 2026-11-10, `0f8c5209...` -> `2653f6fd...` (fp64: `541381a3...` -> `52840c51...`), because of FOUR model
#: commits, each of which changed what the policy's forward computes and none of which ran this `slow` file. Bisected on
#: the CPU, one run per commit (default solve / fp64), every other stretch from the last banked pass (`d8d674b99`,
#: 2026-10-26) to `ec1c3cbd6` having equal endpoints:
#:   `83272a8f9` (config v165, our HP in exact points)             -> `33ff1404...` / `4f057928...`
#:   `7a0153b46` (v166, the EV floor + Leech Seed's integer drain) -> `0ef6876e...` / `a508be5d...`
#:   `566d878b2` (v167, the nature floor + the integer EOT chain)  -> `7dc88425...` / `e123b1d2...`
#:   `3b1733f50` (v168, E13: their max HP as a Smogon distribution) -> `2653f6fd...` / `52840c51...`
#: Why each is that commit's change and not the collector's: (a) each commit's runtime diff is inside
#: `src/agents/model/` (no collector, loop or PPO file; v168 also touched the K9 golden's recorder, which this file never
#: loads); (b) across each, the first flag-OFF rollout's init weights, observations, masks, actions, rewards and
#: episode starts are byte-identical and ONLY the forward's outputs (log-probs, values) move: on 9
#: of 96 rows (v165), on exactly the ONE row with a Leech-Seeded active (v166: why the K9 learner golden did not move),
#: on all 96 (v167, v168); (c) at `83272a8f9`, with `our_hp_points` / `our_healed_hp` put back to the fractions, both
#: old goldens came back bit for bit; (d) at `ec1c3cbd6` the flag-OFF learn equals, bit for bit, the same loop written
#: with every X14 branch removed and runs no `overlap.py` code beyond three predicates (`runner_of`, `decoupled`,
#: `validate_decl`); and a collector built ON and one built OFF hold byte-identical rollouts until the first overlapped
#: play. Earlier: 2026-10-21, `541381a3...` -> `0f8c5209...`, `d4af9f1e2` (FP32_ALT: the default tau solve moved from
#: T62's float64 sum to the fp32 split residual).
OFF_GOLDEN = "2653f6fd68cfd191a2b7c53bd06e1520171aff73e579130ee4c300841509ac81"

#: The same learn with every tau solve forced to T62's float64 naive sum: a SECOND learner, pinned separately. Until
#: 2026-11-10 this constant was literally the golden recorded at the commit before X14 (origin/main `2939cc96`,
#: `541381a3...`). The four model commits above changed the forward itself, so those bytes are no longer reproducible
#: at HEAD; what remains is a CHAIN OF CUSTODY from them (each link above was bisected under this solve too, and (c)
#: reproduced `541381a3...` at `83272a8f9`), plus (d), the direct check made at the re-record.
PRE_X14_GOLDEN_FP64 = "52840c5147f1dad7990209f09f3701e24f25a0ac5832925027587f8ca3b3c077"

#: The registry's stamp (`agents.training.numerics_goldens`): the anchor of the obs-semantics marker and the ROUTINE
#: goldens (the K9 learner golden, the played-bits sentinel) this file was last RUN under. The routine
#: `numerics_goldens_test.py` is red while it is stale: run this file, re-record the two goldens above if they moved,
#: then set it to the anchor `python -m agents.training.numerics_goldens` prints. Never re-stamp without the run.
NUMERICS_ANCHOR = "1ad595516a5fd453"


def test_flag_off_trains_the_same_bytes_as_before_the_overlap_existed(spaces):
    got = _learn(spaces, overlap=False, rollouts=3, n_epochs=1)
    assert getattr(got["col"], "overlap", None) is None
    assert got["hash"] == OFF_GOLDEN, f"the flag-OFF learn() moved: {got['hash']} (golden {OFF_GOLDEN})"


def test_flag_off_under_the_T62_float64_tau_solve_trains_the_pre_X14_bytes(spaces, monkeypatch):
    """The float64 witness: with every tau solve forced to T62's float64 naive sum (the arithmetic the pre-X14 golden
    was recorded under at 2939cc96) the flag-OFF learn() gives PRE_X14_GOLDEN_FP64, byte for byte: those bytes carried
    through the deliberate model changes its comment lists. The seam is the ONE call site every model path shares:
    `hypothesis_set.fixed_mass_presence` calls `fixed_size_tau` through the module global, so the replacement drops
    the caller's ``solve`` and passes fp64 (counted: a vacuous force FAILS)."""
    import inspect

    from agents.model import hypothesis_set as HS

    assert "fixed_size_tau(" in inspect.getsource(HS.fixed_mass_presence)      # the seam's precondition
    orig, calls = HS.fixed_size_tau, []

    def forced(scores, cand, k, n_iter=HS.BISECTION_ITERS, solve=None):
        calls.append(1)
        return orig(scores, cand, k, n_iter, solve="fp64")

    monkeypatch.setattr(HS, "fixed_size_tau", forced)
    got = _learn(spaces, overlap=False, rollouts=3, n_epochs=1)
    assert len(calls) > 0, "the forcing was never reached: a vacuous witness"
    assert got["hash"] == PRE_X14_GOLDEN_FP64, f"the flag-OFF learn() under fp64 moved: {got['hash']}"
    assert got["hash"] != OFF_GOLDEN                                             # and the two solves are two different learners


def test_a_whole_learn_with_overlap_trains_through_overlapped_updates_and_is_deterministic(spaces):
    a = _learn(spaces, overlap=True)
    b = _learn(spaces, overlap=True)
    assert a["hash"] == b["hash"]                                     # deterministic: no timing reaches a row
    ov = a["col"].overlap
    updates = int(a["model"]._n_updates) // 2                          # n_epochs = 2
    assert ov.rollouts == updates - 2 >= 1, (ov.rollouts, updates)     # the first update runs alone; the last plays nothing
    keys = set().union(*[set(d) for d in a["dumps"]])
    for k in ("overlap/play_ms", "overlap/join_wait_ms", "overlap/is_weight_mean", "overlap/is_ess",
              "overlap/prox_forward_ms", "behaviour_snapshot/rows_current", "staleness/age_mean"):
        assert k in keys, k
    # the overlapped rollouts' rows are judged by the snapshot (fatal: a mismatch would have raised)
    judged = [d["behaviour_snapshot/rows_current"] for d in a["dumps"] if "behaviour_snapshot/rows_current" in d]
    assert judged and max(judged) > 0, judged
    assert a["hash"] != _learn(spaces, overlap=False)["hash"]         # it is a different learner (stale rows)
