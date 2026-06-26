"""``SearchTeacherCallback`` — the non-blocking driver: every cycle, select crater candidates, freeze
the trainee, spawn search workers, and collect their verified-better corrections into the model's
``CorrectionBuffer`` (which the AWR aux loss samples each ``train()``). Mirrors the eval callback's
freeze → spawn → poll → collect lifecycle (``spawn_eval_workers``), so it NEVER blocks training: a
trigger while a cycle is still running is skipped, a hung worker is timed out, a crash is logged.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from typing import List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from agents.training.teacher.buffer import Correction

_CYCLE_TIMEOUT_SEC = 3600.0     # a search cycle can be long (Node searches + confirm games); watchdog
_RESPAWN_BACKOFF_STEPS = 5000   # persistent mode: don't tight-loop respawning a worker that keeps dying


def _version_key(path: str) -> int:
    """Numeric trainee-snapshot version (trainee_v10.zip > trainee_v9.zip — a lexical sort wouldn't)."""
    base = os.path.basename(path)
    try:
        return int(base[len("trainee_v"):-len(".zip")])
    except ValueError:
        return -1


class SearchTeacherCallback(BaseCallback):
    """Drives the offline search-teacher (design_search_teacher.md). Constructed only under
    ``--search-teacher``; off ⇒ never added ⇒ byte-identical training."""

    def __init__(self, run_dir: str, *, freq_steps: int, budget: int = 200, n_workers: int = 3,
                 depth: int = 2, beam: int = 3, top_k: int = 4, confirm_rollouts: int = 8,
                 margin_min: float = 0.0, falsify_gate: bool = True, scan_limit: int = 60,
                 persistent: bool = False, refresh_steps: int = 500_000, n_battles: int = 12,
                 verbose: int = 0):
        super().__init__(verbose)
        self.run_dir = run_dir
        self.freq_steps = int(freq_steps)
        self.budget = budget
        self.n_workers = max(1, int(n_workers))
        self.depth, self.beam, self.top_k = depth, beam, top_k
        self.confirm_rollouts, self.margin_min = confirm_rollouts, margin_min
        self.falsify_gate, self.scan_limit = falsify_gate, scan_limit
        self._cycle_dir = os.path.join(run_dir, "teacher_cycle")
        self._pending: Optional[dict] = None
        self._last_launch_step = 0
        # persistent (supply + pool) mode: long-lived workers GENERATE their own fresh losses,
        # re-froze periodically, dripping corrections continuously into the buffer.
        self.persistent = persistent
        self.refresh_steps = int(refresh_steps)
        self.n_battles = int(n_battles)
        self._persist_dir = os.path.join(run_dir, "teacher_persist")
        self._workers: list = []
        self._control_version = 0
        self._last_refreeze_step = 0
        self._ingested_total = 0
        self._opponents_cached: Optional[list] = None
        self._respawn_step: dict = {}        # wid → num_timesteps of its last (re)spawn (backoff)
        self._respawns_total = 0

    # -- lifecycle ----------------------------------------------------------

    def _on_step(self) -> bool:
        if self.persistent:
            if not self._workers:
                self._spawn_persistent()
            else:
                self._reap_and_respawn()            # a crashed worker would otherwise silently stall supply
                if self.num_timesteps - self._last_refreeze_step >= self.refresh_steps:
                    self._refreeze()
            self._ingest()                          # drain any new correction shards into the buffer
            return True
        if self._pending is not None:
            if self._all_done():
                self._collect()
            elif time.time() - self._pending["launched_at"] > _CYCLE_TIMEOUT_SEC:
                self._abort()
            return True
        if self.num_timesteps - self._last_launch_step >= self.freq_steps:
            self._launch()
        return True

    # -- persistent (supply + pool) -----------------------------------------

    def _opponents(self) -> list:
        """The generation opponents: the most-recent pool snapshots (on-distribution self-play
        opponents) + the bot roster. The worker samples from this list."""
        import glob
        from agents.training.eval_callback import eval_opponent_names
        snaps = sorted(glob.glob(os.path.join(self.run_dir, "snapshots", "snapshot_*.zip")))[-6:]
        opps = [{"label": os.path.basename(p)[:-4], "kind": "sentinel", "path": p} for p in snaps]
        opps += [{"label": n, "kind": "bot"} for n in eval_opponent_names() if n != "random"]
        return opps

    def _freeze_snapshot(self) -> str:
        os.makedirs(self._persist_dir, exist_ok=True)
        self._control_version += 1
        snap = os.path.join(self._persist_dir, f"trainee_v{self._control_version}.zip")
        self.model.save(snap)
        self._latest_snap = snap
        return snap

    def _write_control(self, snapshot_path: str, shutdown: bool = False) -> None:
        ctrl = {"snapshot_path": snapshot_path, "version": self._control_version, "shutdown": shutdown}
        tmp = os.path.join(self._persist_dir, "control.json.tmp")
        with open(tmp, "w") as f:
            json.dump(ctrl, f)
        os.replace(tmp, os.path.join(self._persist_dir, "control.json"))   # atomic for the pollers

    def _spawn_persistent(self) -> None:
        out_dir = os.path.join(self._persist_dir, "out")
        self._clean_persist_out(out_dir)            # wipe orphaned shards/gen dirs from a prior crash/restart
        os.makedirs(out_dir, exist_ok=True)
        opponents = self._opponents()
        if not opponents:
            return
        self._opponents_cached = opponents          # reuse for respawns (don't re-glob each time)
        snap = self._freeze_snapshot()
        self._write_control(snap)
        self._last_refreeze_step = self.num_timesteps
        for wid in range(self.n_workers):
            self._spawn_worker(wid, out_dir, opponents)
        if self.verbose:
            print(f"[SearchTeacher] persistent pool: {self.n_workers} workers, "
                  f"{len(opponents)} opponents, refresh every {self.refresh_steps:,} steps")

    def _clean_persist_out(self, out_dir: str) -> None:
        """Remove any leftover correction shards + temp generation dirs (a worker crash or a launcher
        restart can strand them) so a fresh pool never double-ingests stale shards or leaks disk."""
        import shutil
        if not os.path.isdir(out_dir):
            return
        for fn in os.listdir(out_dir):
            p = os.path.join(out_dir, fn)
            try:
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
            except OSError:
                pass

    def _spawn_worker(self, wid: int, out_dir: str, opponents: list) -> None:
        worker_env = {k: v for k, v in os.environ.items() if k != "LAUNCHER_METRICS_FD"}
        cfg = {
            "run_dir": self.run_dir, "worker_id": wid,
            "control_path": os.path.join(self._persist_dir, "control.json"), "output_dir": out_dir,
            "opponents": opponents, "n_battles": self.n_battles,
            "per_iter_budget": max(2, self.budget // 8), "depth": self.depth, "beam": self.beam,
            "top_k": self.top_k, "confirm_rollouts": self.confirm_rollouts,
            "margin_min": self.margin_min,
        }
        cfg_path = os.path.join(self._persist_dir, f"config_{wid}.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)
        logf = open(os.path.join(self._persist_dir, f"worker_{wid}.log"), "w")
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "main.search_teacher_persistent_worker", cfg_path],
                stdout=logf, stderr=subprocess.STDOUT, env=worker_env)
        except Exception:                            # noqa: BLE001 — don't leak the logfile fd if spawn fails
            logf.close()
            raise
        self._workers.append({"proc": proc, "log": logf, "wid": wid})
        self._respawn_step[wid] = self.num_timesteps

    def _reap_and_respawn(self) -> None:
        """Detect dead workers (crash) and respawn them on a step-backoff so a single bad opponent /
        OOM can't tight-loop, and so the supply pool self-heals instead of silently draining to zero."""
        out_dir = os.path.join(self._persist_dir, "out")
        alive, dead = [], []
        for w in self._workers:
            (alive if w["proc"].poll() is None else dead).append(w)
        for w in dead:
            try:
                w["log"].close()
            except Exception:  # noqa: BLE001
                pass
        self._workers = alive
        opponents = self._opponents_cached or self._opponents()
        for w in dead:
            wid = w["wid"]
            if self.num_timesteps - self._respawn_step.get(wid, -_RESPAWN_BACKOFF_STEPS) < _RESPAWN_BACKOFF_STEPS:
                continue                             # backoff not elapsed → leave it down this step
            try:
                self._spawn_worker(wid, out_dir, opponents)
                self._respawns_total += 1
            except Exception as e:  # noqa: BLE001 — a respawn failure must never crash training
                if self.verbose:
                    print(f"[SearchTeacher] respawn of worker {wid} failed: {e}")
        self.logger.record("teacher/workers_alive", float(len(self._workers)))
        self.logger.record("teacher/worker_respawns_total", float(self._respawns_total))

    def _refreeze(self) -> None:
        # Re-freeze so the long-lived workers track the moving policy; bump the version → workers reload.
        snap = self._freeze_snapshot()
        self._write_control(snap)
        self._last_refreeze_step = self.num_timesteps
        # prune old snapshots, keeping the latest THREE — a worker may still be loading the prior one
        # while a new freeze lands; the worker also re-checks existence, so a pruned path just skips.
        import glob
        olds = sorted(glob.glob(os.path.join(self._persist_dir, "trainee_v*.zip")),
                      key=_version_key)[:-3]
        for p in olds:
            try:
                os.remove(p)
            except OSError:
                pass

    def _ingest(self) -> None:
        import glob
        buf = getattr(self.model, "_correction_buffer", None)
        if buf is None:
            return
        ingested = 0
        for jpath in sorted(glob.glob(os.path.join(self._persist_dir, "out", "corr_*.json"))):
            base = jpath[: -len(".json")]
            npz_path = base + ".npz"
            try:
                with open(jpath) as f:
                    sh = json.load(f)
                with np.load(npz_path) as arr:
                    obs, mask = arr["obs"], arr["mask"]   # forced into memory inside the `with`
            except (OSError, ValueError, KeyError):
                continue
            # CONSUME the shard BEFORE buffering it: if a delete fails we drop the shard rather than
            # re-glob + re-ingest it next pass (duplicates would corrupt the bounded ring buffer).
            consumed = True
            for p in (jpath, npz_path):
                try:
                    os.remove(p)
                except OSError:
                    consumed = False
            if not consumed:
                continue
            for i, sc in enumerate(sh.get("scalars") or []):
                buf.add(Correction(
                    obs=obs[i], action_mask=mask[i], better_action=int(sc["better_action"]),
                    advantage=float(sc["advantage"]), confirmed_value=float(sc["confirmed_value"]),
                    step_produced=int(sc["step_produced"]), opponent=sc["opponent"]))
                ingested += 1
        if ingested:
            self._ingested_total += ingested
            self.logger.record("teacher/corrections_ingested_total", float(self._ingested_total))
            self.logger.record("teacher/buffer_size", float(len(buf)))

    def _on_training_end(self) -> None:
        if self.persistent and self._workers:
            try:
                self._write_control(getattr(self, "_latest_snap", ""), shutdown=True)
            except Exception:  # noqa: BLE001
                pass
            for w in self._workers:
                try:
                    w["proc"].wait(timeout=10)
                except Exception:  # noqa: BLE001
                    w["proc"].kill()
                try:
                    w["log"].close()
                except Exception:  # noqa: BLE001
                    pass
            self._workers = []

    def _all_done(self) -> bool:
        return all(w["proc"].poll() is not None for w in self._pending["workers"])

    # -- launch -------------------------------------------------------------

    def _launch(self) -> None:
        self._last_launch_step = self.num_timesteps
        try:
            from agents.training.teacher.selection import select_candidates
            cands = select_candidates(
                self.run_dir, budget=self.budget, scan_limit=self.scan_limit,
                falsify_gate=self.falsify_gate, window=2)
        except Exception as e:  # noqa: BLE001 — selection must never crash training
            if self.verbose:
                print(f"[SearchTeacher] selection failed: {e}")
            return
        if not cands:
            return

        # Fresh cycle dir (no stale shards) + freeze the live trainee to a snapshot the workers load.
        if os.path.isdir(self._cycle_dir):
            for fn in os.listdir(self._cycle_dir):
                try:
                    os.remove(os.path.join(self._cycle_dir, fn))
                except OSError:
                    pass
        os.makedirs(self._cycle_dir, exist_ok=True)
        snap = os.path.join(self._cycle_dir, "trainee.zip")
        self.model.save(snap)

        worker_env = {k: v for k, v in os.environ.items() if k != "LAUNCHER_METRICS_FD"}
        workers, result_bases = [], []
        n = min(self.n_workers, len(cands))
        for wid in range(n):
            slice_cands = cands[wid::n]      # round-robin partition (priority-balanced across workers)
            if not slice_cands:
                continue
            rbase = os.path.join(self._cycle_dir, f"shard_{wid}")
            cfg = {
                "run_dir": self.run_dir, "snapshot_path": snap, "result_path": rbase,
                "candidates": [asdict(c) for c in slice_cands],
                "depth": self.depth, "beam": self.beam, "top_k": self.top_k,
                "confirm_rollouts": self.confirm_rollouts, "margin_min": self.margin_min,
            }
            cfg_path = os.path.join(self._cycle_dir, f"config_{wid}.json")
            with open(cfg_path, "w") as f:
                json.dump(cfg, f)
            logf = open(os.path.join(self._cycle_dir, f"worker_{wid}.log"), "w")
            try:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "main.search_teacher_worker", cfg_path],
                    stdout=logf, stderr=subprocess.STDOUT, env=worker_env)
            except Exception:                        # noqa: BLE001 — don't leak the logfile fd if spawn fails
                logf.close()
                raise
            workers.append({"proc": proc, "log": logf, "rbase": rbase})
            result_bases.append(rbase)
        if not workers:
            return
        self._pending = {"workers": workers, "n_candidates": len(cands),
                         "launched_at": time.time(), "step": self.num_timesteps}
        if self.verbose:
            print(f"[SearchTeacher] cycle @ {self.num_timesteps:,}: {len(cands)} candidates → "
                  f"{len(workers)} workers")

    # -- collect ------------------------------------------------------------

    def _collect(self) -> None:
        corrections: List[Correction] = []
        status: dict = {}
        for w in self._pending["workers"]:
            try:
                w["log"].close()
            except Exception:  # noqa: BLE001
                pass
            jpath = w["rbase"] + ".json"
            if not os.path.exists(jpath):
                status["worker_no_shard"] = status.get("worker_no_shard", 0) + 1   # crash → no shard
                continue
            try:
                with open(jpath) as f:
                    sh = json.load(f)
            except (OSError, ValueError):
                continue
            for k, v in (sh.get("status") or {}).items():
                status[k] = status.get(k, 0) + int(v)
            scalars = sh.get("scalars") or []
            if not scalars:
                continue
            npz = np.load(w["rbase"] + ".npz")
            obs_arr, mask_arr = npz["obs"], npz["mask"]
            for i, sc in enumerate(scalars):
                corrections.append(Correction(
                    obs=obs_arr[i], action_mask=mask_arr[i], better_action=int(sc["better_action"]),
                    advantage=float(sc["advantage"]), confirmed_value=float(sc["confirmed_value"]),
                    step_produced=int(sc["step_produced"]), opponent=sc["opponent"]))

        buf = getattr(self.model, "_correction_buffer", None)
        if buf is not None and corrections:
            buf.extend(corrections)

        n_cand = max(1, self._pending["n_candidates"])
        self.logger.record("teacher/corrections_per_cycle", float(len(corrections)))
        self.logger.record("teacher/yield", float(len(corrections)) / n_cand)
        if corrections:
            self.logger.record("teacher/mean_confirmed_dwin",
                               float(np.mean([c.advantage for c in corrections])))
        if buf is not None:
            self.logger.record("teacher/buffer_size", float(len(buf)))
        if self.verbose:
            print(f"[SearchTeacher] collected {len(corrections)}/{n_cand} corrections; status={status}")
        self._pending = None

    def _abort(self) -> None:
        for w in self._pending["workers"]:
            if w["proc"].poll() is None:
                w["proc"].kill()
            try:
                w["proc"].wait(timeout=5)
                w["log"].close()
            except Exception:  # noqa: BLE001
                pass
        if self.verbose:
            print(f"[SearchTeacher] cycle @ {self._pending['step']:,} timed out — aborted")
        self._pending = None
