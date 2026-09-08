"""Unit tests for SearchTeacherCallback's collect logic (shard → CorrectionBuffer + metrics) — no
subprocess, no model; hand-written shards mimic what a worker publishes."""

import json
import os
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest

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


# --- THE TWO-PHASE CYCLE: selection runs in a CHILD, never on the training step ---------------
#
# The defect these pin (2026-09-07, found by the composition gate): `_launch` called
# `select_for_mode` INLINE from `_on_step`, falsifying every loss trace of the newest eval cycle on
# the trainer — measured ~30 s over 9 traces and ~100 s at the default 60-trace `scan_limit` —
# while the callback's docstring, `src/agents/training/CLAUDE.md` and the `--use-bridge=rust`
# startup banner all described it as non-blocking. A cycle is now `select` (a child) then `search`
# (the workers), and `_on_step` only ever spawns, polls and reads small JSON.


class _FakePopen:
    """A Popen stand-in that records its argv and is already finished."""

    def __init__(self, argv, **kw):
        self.argv, self.kwargs, self.killed = list(argv), kw, 0

    def poll(self):
        return 0

    def kill(self):
        self.killed += 1

    def wait(self, timeout=None):
        return 0


def _trap_selection(monkeypatch):
    """Booby-trap the ONLY definition of the candidate scan.

    Patched on `agents.training.teacher.modes` — the defining module — which is also where
    `main/search_teacher_select_worker.py` READS it (`modes.select_for_mode`, qualified, resolved at
    call time), so this is a live target rather than a definition-site no-op. It is a trap, not a
    stub: nothing in THIS process may reach it, and the whole point of the test is that it doesn't.
    """
    from agents.training.teacher import modes

    def _boom(*a, **kw):                       # pragma: no cover - tripping it IS the failure
        raise AssertionError(
            "select_for_mode ran IN THE TRAINING PROCESS. Selection is 30-100 s of falsify "
            "re-rolls and belongs in the selection worker subprocess; `_on_step` must return in "
            "milliseconds at a cycle boundary.")

    monkeypatch.setattr(modes, "select_for_mode", _boom)


def test_a_cycle_boundary_spawns_the_SELECT_CHILD_and_never_scans_in_process(tmp_path, monkeypatch):
    """The step at a cycle boundary: a spawn, and nothing else."""
    _trap_selection(monkeypatch)
    spawned = []
    monkeypatch.setattr(cb_mod.subprocess, "Popen",
                        lambda argv, **kw: spawned.append(_FakePopen(argv, **kw)) or spawned[-1])
    cb = _callback(tmp_path)
    cb.scan_limit = 37
    cb.num_timesteps = 5000

    t0 = time.perf_counter()
    assert cb._on_step() is True
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(spawned) == 1, "a cycle boundary spawns exactly one selection child"
    assert spawned[0].argv[1:3] == ["-m", "main.search_teacher_select_worker"], spawned[0].argv
    assert cb._pending["phase"] == "select"
    assert elapsed_ms < 1000.0, f"_on_step blocked {elapsed_ms:.0f} ms at a cycle boundary"
    # the child got the run's scan width, and the cycle dir carries a config it can read
    cfg = json.load(open(os.path.join(cb._cycle_dir, "select_config.json")))
    assert cfg["scan_limit"] == 37 and cfg["run_dir"] == str(tmp_path)
    assert cfg["out_path"] == os.path.join(cb._cycle_dir, "candidates.json")
    # and the cost of that step is a RECORDED number, not a claim
    assert cb.logger.records["teacher/step_block_ms"] < 1000.0


def _select_pending(cb, payload):
    """A finished select phase whose child wrote `payload` (or nothing, when payload is None)."""
    os.makedirs(cb._cycle_dir, exist_ok=True)
    out = os.path.join(cb._cycle_dir, "candidates.json")
    if payload is not None:
        with open(out, "w") as f:
            json.dump(payload, f)
    cb._pending = {"phase": "select", "step": 4000, "launched_at": 0.0, "out_path": out,
                   "n_candidates": 0,
                   "workers": [{"proc": SimpleNamespace(poll=lambda: 3),
                                "log": SimpleNamespace(close=lambda: None)}]}


@pytest.mark.parametrize("payload,why", [
    ({"error": "RuntimeError: the re-roll driver died"}, "the child reported an error"),
    (None, "the child died hard and wrote nothing"),
])
def test_a_failed_SELECTION_lands_in_the_status_histogram(tmp_path, capsys, payload, why):
    """It used to be a bare `except Exception` + an optional print.

    A run whose teacher selected nothing for a week then looked exactly like a run whose policy had
    no craters. The failure now reaches BOTH artifacts the composition gate reads — the
    `selection failed` marker AND an `error:selection` key in the per-cycle status histogram — and
    `teacher/corrections_per_cycle` gets its point so the TB series and the collect markers stay
    one-to-one.
    """
    cb = _callback(tmp_path)
    cb.verbose = 1
    _select_pending(cb, payload)

    cb._finish_selection()

    out = capsys.readouterr().out
    assert "[SearchTeacher] selection failed" in out, why
    assert "error:selection" in out, f"the histogram swallowed the failure ({why}): {out!r}"
    assert cb.logger.records["teacher/corrections_per_cycle"] == 0.0
    assert cb.logger.records["teacher/selection_failures_total"] == 1.0
    assert cb._pending is None, "a failed selection must not wedge the cycle slot forever"


def test_an_EMPTY_selection_ends_the_cycle_without_a_cycle_marker(tmp_path, capsys):
    """No candidates is a fact about the policy, not a defect — and the composition gate asserts
    every `cycle @ N: M candidates` marker carries M > 0, so this path must not emit one."""
    cb = _callback(tmp_path)
    cb.verbose = 1
    _select_pending(cb, {"candidates": [], "n_candidates": 0})

    cb._finish_selection()

    out = capsys.readouterr().out
    assert "cycle @" not in out and "collected" not in out
    assert "no candidates" in out
    assert cb._pending is None


def test_the_SELECTED_candidates_reach_the_search_workers_verbatim(tmp_path, monkeypatch):
    """Phase 2: the child's candidate list is what the workers are configured with, round-robin."""
    _trap_selection(monkeypatch)
    spawned = []
    monkeypatch.setattr(cb_mod.subprocess, "Popen",
                        lambda argv, **kw: spawned.append(_FakePopen(argv, **kw)) or spawned[-1])
    cb = _callback(tmp_path)
    cb.verbose = 0
    cb.n_workers = 2
    cb.num_timesteps = 9000
    cb.model.save = lambda p: open(p, "wb").close()
    cands = [{"summary_path": f"/t/{i}_summary.json", "recon_path": f"/t/{i}_reconstruction.json",
              "inv_index": i, "turn": i, "opponent": "sentinel_1", "step": 500,
              "anchor_delta": 1.0 - i / 10.0, "verdict": "MISTAKE"} for i in range(3)]
    _select_pending(cb, {"candidates": cands, "n_candidates": len(cands)})

    cb._finish_selection()

    assert cb._pending["phase"] == "search"
    assert cb._pending["n_candidates"] == 3
    assert len(spawned) == 2 and all(p.argv[2] == "main.search_teacher_worker" for p in spawned)
    got = []
    for wid in range(2):
        got += json.load(open(os.path.join(cb._cycle_dir, f"config_{wid}.json")))["candidates"]
    assert sorted(c["inv_index"] for c in got) == [0, 1, 2]


def test_the_select_config_is_NOT_globbed_by_the_composition_gate(tmp_path, monkeypatch):
    """`config_*.json` in the cycle dir means "a SEARCH worker's config", and the composition gate
    asserts every one of them records `"impl": "rust"`. Selection's own config must stay outside
    that glob: `select_candidates` calls `falsify_battle` with no `impl`, so its re-rolls run on
    node even on a rust run — a real gap, recorded in the backlog rather than papered over by a
    file name that would make the gate assert something false."""
    _trap_selection(monkeypatch)
    monkeypatch.setattr(cb_mod.subprocess, "Popen", lambda argv, **kw: _FakePopen(argv, **kw))
    cb = _callback(tmp_path)
    cb.num_timesteps = 5000
    cb._on_step()
    import glob as _glob
    assert _glob.glob(os.path.join(cb._cycle_dir, "config_*.json")) == []
    assert os.path.exists(os.path.join(cb._cycle_dir, "select_config.json"))


# --- the selection worker itself ---------------------------------------------------------------

def test_the_select_worker_publishes_candidates_atomically(tmp_path, monkeypatch):
    from agents.training.teacher import modes
    from agents.training.teacher.selection import Candidate
    from main import search_teacher_select_worker as sw

    seen = {}

    def _fake(mode, run_dir, **kw):
        seen.update(kw, mode=mode, run_dir=run_dir)
        return [Candidate("/t/a_summary.json", "/t/a_reconstruction.json", 4, 7, "heuristic",
                          500, 0.9, "MISTAKE")]

    monkeypatch.setattr(modes, "select_for_mode", _fake)
    out = tmp_path / "candidates.json"
    cfg = tmp_path / "select_config.json"
    cfg.write_text(json.dumps({"run_dir": "/runs/x", "out_path": str(out), "mode": "crater",
                               "budget": 8, "scan_limit": 12, "falsify_gate": True,
                               "window": 2, "wp_band": 0.15}))

    assert sw.run(str(cfg)) == 0
    assert seen["scan_limit"] == 12 and seen["budget"] == 8 and seen["mode"] == "crater"
    payload = json.loads(out.read_text())
    assert payload["n_candidates"] == 1 and payload["candidates"][0]["inv_index"] == 4
    assert not (tmp_path / "candidates.json.tmp").exists(), "the temp file must be os.replace'd away"


def test_the_select_worker_reports_a_raise_as_an_ERROR_PAYLOAD_and_a_nonzero_exit(tmp_path,
                                                                                 monkeypatch):
    """A selection crash must arrive as data. The parent polls the PROCESS, so an exception that
    only reached stderr would be indistinguishable from an empty candidate list."""
    from agents.training.teacher import modes
    from main import search_teacher_select_worker as sw

    def _boom(*a, **kw):
        raise RuntimeError("the re-roll driver wedged")

    monkeypatch.setattr(modes, "select_for_mode", _boom)
    out = tmp_path / "candidates.json"
    cfg = tmp_path / "select_config.json"
    cfg.write_text(json.dumps({"run_dir": "/runs/x", "out_path": str(out), "mode": "crater",
                               "budget": 8, "scan_limit": 12, "falsify_gate": True}))

    assert sw.run(str(cfg)) == 3
    payload = json.loads(out.read_text())
    assert payload["error"] == "RuntimeError: the re-roll driver wedged"
    assert "Traceback" in payload["traceback"]


def test_a_cycles_worker_CONFIGS_survive_the_next_cycles_select_launch(tmp_path, monkeypatch):
    """The regression the composition gate caught on this change's first run.

    The one-phase `_launch` wiped the cycle dir and wrote the worker configs microseconds later, so
    the wipe was invisible. Moving selection out put a whole SELECTION between the two, and a run
    that ended mid-selection left a cycle dir with no `config_*.json` at all — the artifact the
    composition gate reads to prove the workers ran on rust. Each phase now clears only what it is
    about to rewrite.
    """
    _trap_selection(monkeypatch)
    monkeypatch.setattr(cb_mod.subprocess, "Popen", lambda argv, **kw: _FakePopen(argv, **kw))
    cb = _callback(tmp_path)
    cb.verbose = 0
    cb.n_workers = 1
    cb.num_timesteps = 5000
    cb.model.save = lambda p: open(p, "wb").close()
    cands = [{"summary_path": "/t/a_summary.json", "recon_path": "/t/a_reconstruction.json",
              "inv_index": 0, "turn": 3, "opponent": "heuristic", "step": 500,
              "anchor_delta": 0.5, "verdict": "MISTAKE"}]

    cb._on_step()                                   # cycle 1: select
    _select_pending(cb, {"candidates": cands})
    cb._finish_selection()                          # cycle 1: workers spawned
    worker_cfg = os.path.join(cb._cycle_dir, "config_0.json")
    assert os.path.exists(worker_cfg)

    cb._pending = None
    cb.num_timesteps = 20000
    cb._on_step()                                   # cycle 2: select launched

    assert os.path.exists(worker_cfg), (
        "the next cycle's SELECT launch erased the last cycle's worker config — the run's only "
        "on-disk evidence of which engine its search workers used")
    assert not os.path.exists(os.path.join(cb._cycle_dir, "candidates.json")), (
        "a stale candidates.json would be read as THIS cycle's selection result")
