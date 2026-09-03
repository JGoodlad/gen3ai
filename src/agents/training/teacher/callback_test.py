"""Unit tests for SearchTeacherCallback's collect logic (shard → CorrectionBuffer + metrics) — no
subprocess, no model; hand-written shards mimic what a worker publishes."""

import json
import os
import subprocess
from types import SimpleNamespace

import numpy as np

from agents.training.teacher.buffer import CorrectionBuffer
from agents.training.teacher import callback as cb_mod
from agents.training.teacher.callback import SearchTeacherCallback

_OBS_DIM, _N_ACT = 8, 11


class _Logger:
    def __init__(self):
        self.records = {}

    def record(self, k, v):
        self.records[k] = v


def _write_shard(d, wid, corrs, status, n_cand):
    rbase = os.path.join(d, f"shard_{wid}")
    if corrs:
        np.savez(rbase + ".npz",
                 obs=np.stack([c[0] for c in corrs]).astype(np.float32),
                 mask=np.stack([c[1] for c in corrs]).astype(np.int8))
    scalars = [{"better_action": c[2], "advantage": c[3], "confirmed_value": 0.7,
                "step_produced": 100, "opponent": "sentinel_1"} for c in corrs]
    with open(rbase + ".json", "w") as f:
        json.dump({"scalars": scalars, "status": status, "n_candidates": n_cand}, f)
    return rbase


def _done_worker(rbase):
    return {"proc": SimpleNamespace(poll=lambda: 0), "log": SimpleNamespace(close=lambda: None),
            "rbase": rbase}


def _callback(tmp_path):
    cb = SearchTeacherCallback(str(tmp_path), freq_steps=1000, verbose=0)
    # BaseCallback.logger is a read-only property proxying self.model.logger.
    cb.model = SimpleNamespace(_correction_buffer=CorrectionBuffer(100), logger=_Logger())
    return cb


def test_collect_fills_buffer_and_metrics(tmp_path):
    d = tmp_path / "cyc"; d.mkdir()
    r0 = _write_shard(str(d), 0, [(np.arange(_OBS_DIM), np.ones(_N_ACT), 6, 0.4),
                                  (np.arange(_OBS_DIM), np.ones(_N_ACT), 2, 0.8)],
                      {"ok": 2, "gate_failed": 1}, n_cand=3)
    r1 = _write_shard(str(d), 1, [(np.arange(_OBS_DIM), np.ones(_N_ACT), 8, 0.6)],
                      {"ok": 1, "unresolved": 2}, n_cand=3)
    cb = _callback(tmp_path)
    cb._pending = {"workers": [_done_worker(r0), _done_worker(r1)], "n_candidates": 6,
                   "launched_at": 0.0, "step": 5000}
    cb._collect()

    assert len(cb.model._correction_buffer) == 3            # 2 + 1 corrections collected
    rec = cb.logger.records
    assert rec["teacher/corrections_per_cycle"] == 3.0
    assert abs(rec["teacher/yield"] - 3 / 6) < 1e-9
    assert abs(rec["teacher/mean_confirmed_dwin"] - np.mean([0.4, 0.8, 0.6])) < 1e-6
    assert rec["teacher/buffer_size"] == 3.0
    assert cb._pending is None                              # cycle cleared


def test_collect_handles_a_crashed_worker_with_no_shard(tmp_path):
    d = tmp_path / "cyc"; d.mkdir()
    r0 = _write_shard(str(d), 0, [(np.arange(_OBS_DIM), np.ones(_N_ACT), 6, 0.5)], {"ok": 1}, 2)
    crashed = _done_worker(os.path.join(str(d), "shard_missing"))   # its .json was never written
    cb = _callback(tmp_path)
    cb._pending = {"workers": [_done_worker(r0), crashed], "n_candidates": 4,
                   "launched_at": 0.0, "step": 5000}
    cb._collect()                                          # must not raise on the missing shard
    assert len(cb.model._correction_buffer) == 1
    assert cb._pending is None


def test_all_done_predicate(tmp_path):
    cb = _callback(tmp_path)
    running = {"proc": SimpleNamespace(poll=lambda: None)}
    done = {"proc": SimpleNamespace(poll=lambda: 0)}
    cb._pending = {"workers": [done, done]}
    assert cb._all_done() is True
    cb._pending = {"workers": [done, running]}
    assert cb._all_done() is False


# --- persistent (supply + pool) mode ---

def _persist_cb(tmp_path, n_battles=12):
    cb = SearchTeacherCallback(str(tmp_path), freq_steps=1000, persistent=True,
                               refresh_steps=500_000, n_battles=n_battles, verbose=0)
    cb.model = SimpleNamespace(_correction_buffer=CorrectionBuffer(100), logger=_Logger())
    return cb


def _write_corr_shard(out_dir, wid, seq, corrs):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"corr_{wid}_{seq}")
    np.savez(base + ".npz",
             obs=np.stack([c[0] for c in corrs]).astype(np.float32),
             mask=np.stack([c[1] for c in corrs]).astype(np.int8))
    scalars = [{"better_action": c[2], "advantage": c[3], "confirmed_value": 0.6,
                "step_produced": 7, "opponent": "snapshot_x"} for c in corrs]
    with open(base + ".json", "w") as f:
        json.dump({"scalars": scalars}, f)


def test_persistent_ingest_drains_shards_into_buffer_and_deletes(tmp_path):
    cb = _persist_cb(tmp_path)
    out = os.path.join(cb._persist_dir, "out")
    _write_corr_shard(out, 0, 0, [(np.arange(_OBS_DIM), np.ones(_N_ACT), 6, 0.4)])
    _write_corr_shard(out, 1, 0, [(np.arange(_OBS_DIM), np.ones(_N_ACT), 2, 0.7),
                                  (np.arange(_OBS_DIM), np.ones(_N_ACT), 8, 0.5)])
    cb._ingest()
    assert len(cb.model._correction_buffer) == 3
    assert cb.model.logger.records["teacher/corrections_ingested_total"] == 3.0
    # shards consumed (deleted) so they're never ingested twice.
    assert not os.path.exists(os.path.join(out, "corr_0_0.json"))
    cb._ingest()                                  # second pass: nothing new
    assert len(cb.model._correction_buffer) == 3


def test_persistent_write_control_atomic_roundtrip(tmp_path):
    from main.search_teacher_persistent_worker import _read_control
    cb = _persist_cb(tmp_path)
    cb._freeze_snapshot = lambda: "/snap_v1.zip"   # don't actually save a model in the unit test
    os.makedirs(cb._persist_dir, exist_ok=True)
    cb._control_version = 3
    cb._write_control("/snap_v3.zip")
    ctrl = _read_control(os.path.join(cb._persist_dir, "control.json"))
    assert ctrl == {"snapshot_path": "/snap_v3.zip", "version": 3, "shutdown": False}
    cb._write_control("/snap_v3.zip", shutdown=True)
    assert _read_control(os.path.join(cb._persist_dir, "control.json"))["shutdown"] is True


def test_persistent_opponents_lists_snapshots_and_bots(tmp_path):
    snaps = tmp_path / "snapshots"; snaps.mkdir()
    for s in ("snapshot_000000010.zip", "snapshot_000000020.zip"):
        (snaps / s).write_text("x")
    cb = _persist_cb(tmp_path)
    opps = cb._opponents()
    kinds = {o["kind"] for o in opps}
    assert "sentinel" in kinds and "bot" in kinds
    sent = [o for o in opps if o["kind"] == "sentinel"]
    assert all(o["path"].endswith(".zip") and os.path.exists(o["path"]) for o in sent)
    assert "random" not in {o["label"] for o in opps}   # random is the eval floor, not a teacher opp


def test_version_key_orders_numerically(tmp_path):
    from agents.training.teacher.callback import _version_key
    paths = ["a/trainee_v2.zip", "a/trainee_v10.zip", "a/trainee_v9.zip"]
    assert sorted(paths, key=_version_key) == ["a/trainee_v2.zip", "a/trainee_v9.zip", "a/trainee_v10.zip"]


def test_clean_persist_out_wipes_stale_shards_and_dirs(tmp_path):
    cb = _persist_cb(tmp_path)
    out = os.path.join(cb._persist_dir, "out")
    os.makedirs(os.path.join(out, "gen_0_3"))          # a stranded temp generation dir
    open(os.path.join(out, "corr_0_0.json"), "w").close()
    np.savez(os.path.join(out, "corr_0_0.npz"), x=np.zeros(1))
    cb._clean_persist_out(out)
    assert os.listdir(out) == []                        # both the dir and the orphan shards are gone


def test_reap_respawns_dead_worker_and_records_alive(tmp_path):
    cb = _persist_cb(tmp_path)
    cb.num_timesteps = 100_000
    cb._opponents_cached = [{"label": "heuristic", "kind": "bot"}]
    dead = {"proc": SimpleNamespace(poll=lambda: 0), "log": SimpleNamespace(close=lambda: None), "wid": 0}
    live = {"proc": SimpleNamespace(poll=lambda: None), "log": SimpleNamespace(close=lambda: None), "wid": 1}
    cb._workers = [dead, live]
    respawned = []
    cb._spawn_worker = lambda wid, o, opp: (respawned.append(wid),
                                            cb._workers.append({"proc": SimpleNamespace(poll=lambda: None),
                                                                "log": SimpleNamespace(close=lambda: None),
                                                                "wid": wid}))
    cb._reap_and_respawn()
    assert respawned == [0]                             # only the dead worker (wid 0) is respawned
    assert {w["wid"] for w in cb._workers} == {0, 1}    # pool self-healed back to full
    assert cb.logger.records["teacher/workers_alive"] == 2.0
    assert cb.logger.records["teacher/worker_respawns_total"] == 1.0


def test_reap_respects_respawn_backoff(tmp_path):
    cb = _persist_cb(tmp_path)
    cb.num_timesteps = 100_000
    cb._opponents_cached = [{"label": "heuristic", "kind": "bot"}]
    cb._respawn_step = {0: 99_000}                      # respawned 1000 steps ago (< 5000 backoff)
    cb._workers = [{"proc": SimpleNamespace(poll=lambda: 0),
                    "log": SimpleNamespace(close=lambda: None), "wid": 0}]
    respawned = []
    cb._spawn_worker = lambda wid, o, opp: respawned.append(wid)
    cb._reap_and_respawn()
    assert respawned == []                              # backoff not elapsed → left down this step
    assert cb._workers == []                            # the dead worker is still removed


def test_persistent_ingest_drops_shard_when_delete_fails_no_dup(tmp_path, monkeypatch):
    cb = _persist_cb(tmp_path)
    out = os.path.join(cb._persist_dir, "out")
    _write_corr_shard(out, 0, 0, [(np.arange(_OBS_DIM), np.ones(_N_ACT), 6, 0.4)])
    def _boom(_p):
        raise OSError("disk")
    monkeypatch.setattr("os.remove", _boom)            # consume-before-buffer can't delete → drop, don't add
    cb._ingest()
    cb._ingest()                                        # a 2nd pass must NOT re-ingest (no duplicate)
    assert len(cb.model._correction_buffer) == 0


# --- the two worker-REAP bounds: scaled, not hardcoded -----------------------------------------
#
# gen3_contention_robust_timeouts_v1. ``_on_training_end`` reaps persistent workers after the
# cooperative ``shutdown`` control file; ``_abort`` collects a worker it has already SIGKILLed.
# Both were hardcoded wall-clock waits (10 s / 5 s) until 2026-09-01 — bounds on a subprocess, so
# on a loaded box they measure the box rather than the worker, and both fire INSIDE training where
# the box is busy by construction (its ``local_battle_runner`` twin fired at load ~50 on 16 cores
# and killed a measurement arm outright, 2026-08-31). These pin that both are read at CALL time
# through ``scale_timeout``, and that an idle box is unchanged.


def test_worker_reap_timeouts_are_the_base_values_on_an_idle_box(monkeypatch):
    """Factor 1.0 => still exactly 10.0 s / 5.0 s. The fix must be a no-op when the box is quiet."""
    monkeypatch.setenv("GEN3AI_TIMEOUT_SCALE", "1")
    assert cb_mod._SHUTDOWN_REAP_TIMEOUT == 10.0
    assert cb_mod._ABORT_REAP_TIMEOUT == 5.0
    assert cb_mod._shutdown_reap_timeout() == 10.0
    assert cb_mod._abort_reap_timeout() == 5.0


def test_worker_reap_timeouts_stretch_with_contention(monkeypatch):
    """The whole point: a loaded box gets proportionally longer to reap each worker."""
    monkeypatch.setenv("GEN3AI_TIMEOUT_SCALE", "6")
    assert cb_mod._shutdown_reap_timeout() == 60.0
    assert cb_mod._abort_reap_timeout() == 30.0


# --- `_abort` must not leak the worker LOG HANDLE ------------------------------------------------
#
# The abort path SIGKILLs each cycle worker and then reaps it. The reap used to close the worker's
# log file only on the SUCCESS path: the `TimeoutExpired` branch re-raised past the `close()`
# (deliberately, to preserve the surrounding handler's behaviour) and any other `wait` error was
# swallowed by that same outer handler before reaching it. So every aborted cycle leaked one file
# descriptor per worker that failed to reap — and a hung-cycle abort under contention is exactly
# the path that repeats over a long run. The close is now in a `finally`.


class _NeverReaps:
    """A worker whose `wait` always times out — the starved-reap shape, which is what makes the
    leak RECUR rather than happen once."""

    def __init__(self):
        self.killed = 0

    def poll(self):
        return None                      # still 'running' => _abort kills it

    def kill(self):
        self.killed += 1

    def wait(self, timeout=None):
        raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout)


class _WaitRaises(_NeverReaps):
    """The other lost path: `wait` raising something that is not a timeout."""

    def wait(self, timeout=None):
        raise OSError("no such process")


def _fd_count():
    return len(os.listdir("/proc/self/fd"))


def _abort_worker(tmp_path, proc, name):
    return {"proc": proc, "log": open(os.path.join(str(tmp_path), name), "w"),
            "rbase": os.path.join(str(tmp_path), name)}


def test_abort_closes_every_worker_log_even_when_the_reap_times_out(tmp_path, capsys):
    """Two aborts, two workers each, every reap timing out: every handle closed, no fd growth."""
    cb = _callback(tmp_path)
    cb.verbose = 0
    before = _fd_count()
    for cycle in range(2):
        names = ["w%d_%d.log" % (cycle, i) for i in range(2)]
        workers = [_abort_worker(tmp_path, _NeverReaps(), n) for n in names]
        cb._pending = {"workers": workers, "step": 1000 * (cycle + 1)}
        cb._abort()
        assert cb._pending is None
        for w in workers:
            assert w["proc"].killed == 1, "the abort must still SIGKILL the worker"
            assert w["log"].closed, "the worker log handle was left open on the timeout path"
    assert _fd_count() == before, "aborting leaked file descriptors"
    # and the timeout still self-diagnoses rather than dying silently
    assert "did not reap within" in capsys.readouterr().err


def test_abort_closes_the_log_when_the_reap_raises_something_other_than_a_timeout(tmp_path):
    """The second lost path — the outer `except Exception: pass` used to swallow the error before
    the `close()` on that line was ever reached."""
    cb = _callback(tmp_path)
    cb.verbose = 0
    before = _fd_count()
    w = _abort_worker(tmp_path, _WaitRaises(), "boom.log")
    cb._pending = {"workers": [w], "step": 7}
    cb._abort()
    assert w["log"].closed
    assert _fd_count() == before


def test_abort_still_closes_the_log_on_the_clean_reap_path(tmp_path):
    """The no-op half: a worker that reaps normally is closed exactly as before."""
    cb = _callback(tmp_path)
    cb.verbose = 0
    proc = SimpleNamespace(poll=lambda: 0, kill=lambda: None, wait=lambda timeout=None: 0)
    w = _abort_worker(tmp_path, proc, "clean.log")
    cb._pending = {"workers": [w], "step": 3}
    cb._abort()
    assert w["log"].closed
