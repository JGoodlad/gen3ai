"""THE GPU WINDOW end to end (T46): a REAL launcher supervisor, a REAL tiny CPU child as the run, a REAL guest.

What is pinned, each against a revert:
  * the happy cycle — armed → the child is SIGTERMed and checkpoints → GO → the guest claims and runs → DONE →
    the run re-launches in the RESUME role from the pause's checkpoint, same ``--run-dir``; the guest ran
    strictly AFTER the run left and BEFORE it came back (drop the window wiring in ``_supervise`` and the child
    is never paused: it completes on its own and the guest never runs);
  * a mid-window failure HOLDS — a guest that exits non-zero leaves the run paused (no re-launch) until an
    operator's ``resume`` (drop the HOLD and the run re-launches over the failure);
  * a guest that never starts is MISSED and the run resumes, and a late guest is then refused (it never runs).

The child is ``_FAKE_TRAINER`` below: it records its argv, honours SIGTERM like the trainer's deferred abort
(save ``final_model_interrupted.zip`` + ``latest.txt`` at a "safe point", exit ``INTERRUPTED``), and completes
at once when launched with ``--model`` (the resumed run). No GPU, no torch, no Showdown.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from main.launcher.child import _build_child_env
from main.launcher.gpu_window import cli
from main.launcher.gpu_window import protocol as P
from main.launcher.gpu_window.controller import WindowController
from main.launcher.run import _SessionCtx, _supervise
from main.launcher.state import LauncherState
from utils.paths import src_root

pytestmark = pytest.mark.integration

_FAKE_TRAINER = r'''
import json, os, signal, sys, time
args = sys.argv[1:]
run_dir = args[args.index("--run-dir") + 1]
rec = os.path.join(run_dir, "children.jsonl")
def note(**kw):
    with open(rec, "a") as f:
        f.write(json.dumps({"pid": os.getpid(), "t": time.time(), **kw}) + "\n")
note(event="start", argv=args)
if "--model" in args:
    sys.exit(0)                                   # the resumed run: COMPLETE at once
stop = []
signal.signal(signal.SIGTERM, lambda *a: stop.append(1))
open(os.path.join(run_dir, "ready"), "w").write(str(os.getpid()))
t0 = time.time()
reported = False
while not stop and time.time() - t0 < 45:
    # the first rollout report (the launcher's metrics pipe), held back while `defer_report` exists
    if not reported and not os.path.exists(os.path.join(run_dir, "defer_report")):
        os.write(int(os.environ["LAUNCHER_METRICS_FD"]), b'{"_step": 1, "time/fps": 1.0}\n')
        reported = True
    time.sleep(0.05)
if not stop:
    note(event="complete_unpaused")
    sys.exit(0)                                   # never paused: the run just completes
time.sleep(0.3)                                   # the next "safe point"
open(os.path.join(run_dir, "final_model_interrupted.zip"), "w").write("x")
open(os.path.join(run_dir, "latest.txt"), "w").write("final_model_interrupted.zip")
note(event="exit_interrupted")
sys.exit(15)
'''


def _children(run_dir: Path) -> list:
    p = run_dir / "children.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def _wait_for(pred, timeout: float, what: str):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        v = pred()
        if v:
            return v
        time.sleep(0.05)
    raise AssertionError(f"timed out after {timeout:.0f} s waiting for {what}")


def _guest(run_dir: Path, *cmd: str) -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": str(src_root()), "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.Popen([sys.executable, "-m", "main.launcher.gpu_window", "guest", str(run_dir),
                             "--wait-max-min", "2", "--no-lease-check", "test", "--", *cmd],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


class _Harness:
    """`_supervise` over the fake trainer, with a real `WindowController` (GPU probe: always free)."""

    def __init__(self, tmp_path: Path, monkeypatch, defer_report: bool = False):
        monkeypatch.setenv("GEN3AI_GPU_LOCK", str(tmp_path / "gpu.lock"))   # never the box's real lease
        # `main.launcher.run` the ATTRIBUTE is the `run()` function (the package re-exports it): patch the module.
        monkeypatch.setattr(sys.modules["main.launcher.run"], "_git_hash", lambda: "abc1234")
        self.run_dir = tmp_path / "models" / "rb_window_toy"
        self.run_dir.mkdir(parents=True)
        if defer_report:
            (self.run_dir / "defer_report").write_text("")
        script = tmp_path / "fake_trainer.py"
        script.write_text(_FAKE_TRAINER)
        self.state = LauncherState(interval_hours=0)
        self.state.run_dir = str(self.run_dir)
        self.ctx = _SessionCtx(child_args=["--steps", "1000", "--run-dir", str(self.run_dir)],
                               child_env=_build_child_env(), train_script=str(script), src_dir=str(src_root()),
                               run_dir=str(self.run_dir), run_dir_box=[str(self.run_dir)],
                               session_start=time.time() - 5, interval_seconds=0, grace_seconds=1200.0, pin=False)
        self.window = WindowController(lambda: self.ctx.run_dir_box[0], add_event=self.state.add_event,
                                       set_status=self.state.set_window_status, gpu_pids=lambda: [],
                                       request_poll_s=0.1, wait_poll_s=0.1)
        self.window.announce()
        self.shutdown = threading.Event()

    def supervise(self) -> int:
        return _supervise(self.state, queue.Queue(), self.ctx, interval_hours=0, grace_minutes=20.0,
                          max_crash_restarts=3, shutdown=self.shutdown, window=self.window)

    def arm(self, *extra: str) -> int:
        _wait_for(lambda: (self.run_dir / "ready").exists(), 30, "the fake trainer to be ready")
        # A 20 s claim bound by default: a guest that breaks fails the test in seconds (MISSED), not minutes.
        extra = extra or ("--guest-start-timeout-s", "20")
        return cli.main(["arm", str(self.run_dir), "--precheck", "true", "--no-lease-check", "test", *extra])

    def phase(self):
        return (P.read_current(self.run_dir) or {}).get("phase")

    def events(self) -> str:
        return "\n".join(self.state.snapshot().events)


def _in_thread(fn):
    out: dict = {}

    def run():
        try:
            out["v"] = fn()
        except BaseException as e:  # noqa: BLE001 — re-raised in the test thread
            out["e"] = e
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, out


def test_window_pauses_runs_the_guest_and_resumes_the_same_run(tmp_path, monkeypatch):
    h = _Harness(tmp_path, monkeypatch)
    marker = tmp_path / "guest_ran.json"
    guest_code = f"import json,time; json.dump({{'t': time.time()}}, open({str(marker)!r}, 'w'))"

    def operator():
        assert h.arm() == 0
        g = _guest(h.run_dir, sys.executable, "-c", guest_code)
        out, _ = g.communicate(timeout=60)
        return g.returncode, out

    t, res = _in_thread(operator)
    try:
        code = h.supervise()
    finally:
        h.shutdown.set()
    t.join(60)
    assert "e" not in res, res.get("e")
    guest_rc, guest_out = res["v"]
    assert code == 0
    assert guest_rc == 0, guest_out

    kids = _children(h.run_dir)
    starts = [k for k in kids if k["event"] == "start"]
    assert len(starts) == 2, f"expected the paused child + ONE re-launch, got {kids}"
    assert not any(k["event"] == "complete_unpaused" for k in kids), "the run was never paused"
    paused_exit = next(k for k in kids if k["event"] == "exit_interrupted")
    first, second = starts
    assert "--model" not in first["argv"]
    resumed = second["argv"]
    assert resumed[resumed.index("--model") + 1] == str(h.run_dir / "final_model_interrupted.zip"), \
        "the run must re-launch from the PAUSE's checkpoint"
    assert resumed[resumed.index("--run-dir") + 1] == str(h.run_dir), "the SAME run, not a fork"

    ran_at = json.loads(marker.read_text())["t"]
    assert paused_exit["t"] < ran_at < second["t"], \
        "the guest must run strictly while the run is off the GPU (after its exit, before its re-launch)"

    cur = P.read_current(h.run_dir)
    assert cur["phase"] == P.RESUMED and cur["outcome"] == P.DONE, cur
    wdir = P.window_dir(h.run_dir, cur["window_id"])
    assert P.read_json(wdir / P.DONE_FILE)["rc"] == 0
    assert P.read_json(wdir / P.CLAIM_FILE)["by"] == "guest"
    assert not (P.window_root(h.run_dir) / P.REQUEST_FILE).exists(), "the request is CONSUMED at the trigger"
    ev = h.events()
    assert "GO —" in ev and "guest DONE (rc 0)" in ev
    assert h.state.snapshot().window_status is None, "the status line clears when the window closes"


def test_a_failing_guest_holds_the_run_paused_until_resume(tmp_path, monkeypatch):
    h = _Harness(tmp_path, monkeypatch)
    resumed_at: dict = {}

    def operator():
        assert h.arm() == 0
        g = _guest(h.run_dir, sys.executable, "-c", "import sys; sys.exit(3)")
        g.communicate(timeout=60)
        _wait_for(lambda: h.phase() == P.HOLD, 30, "the HOLD")
        time.sleep(1.0)                    # a HOLD must not re-launch on its own
        n_starts = sum(1 for k in _children(h.run_dir) if k["event"] == "start")
        status = h.state.snapshot().window_status
        resumed_at["t"] = time.time()
        rc = cli.main(["resume", str(h.run_dir), "--reason", "test: the guest's failure is understood"])
        return g.returncode, n_starts, status, rc

    t, res = _in_thread(operator)
    try:
        code = h.supervise()
    finally:
        h.shutdown.set()
    t.join(60)
    assert "e" not in res, res.get("e")
    guest_rc, starts_during_hold, status, resume_rc = res["v"]
    assert guest_rc == 3 and resume_rc == 0 and code == 0
    assert starts_during_hold == 1, "the run re-launched while the window was in HOLD"
    assert status and "HOLD" in status and "rc 3" in status, status
    second = [k for k in _children(h.run_dir) if k["event"] == "start"][1]
    assert second["t"] >= resumed_at["t"], "the run came back before the operator's resume"
    cur = P.read_current(h.run_dir)
    assert cur["phase"] == P.RESUMED and cur["outcome"] == P.RESUMED, cur
    assert "GPU WINDOW HOLD" in h.events()


def test_a_guest_that_never_starts_is_missed_and_the_run_resumes(tmp_path, monkeypatch):
    h = _Harness(tmp_path, monkeypatch)

    def operator():
        return h.arm("--guest-start-timeout-s", "1")

    t, res = _in_thread(operator)
    try:
        code = h.supervise()
    finally:
        h.shutdown.set()
    t.join(60)
    assert "e" not in res, res.get("e")
    assert res["v"] == 0
    assert code == 0
    assert sum(1 for k in _children(h.run_dir) if k["event"] == "start") == 2
    cur = P.read_current(h.run_dir)
    assert cur["outcome"] == P.MISSED, cur
    wdir = P.window_dir(h.run_dir, cur["window_id"])
    assert P.read_json(wdir / P.CLAIM_FILE)["by"] == "launcher"
    # A late guest finds no window to serve and never runs its command.
    late = _guest(h.run_dir, sys.executable, "-c", "raise SystemExit(99)")
    out, _ = late.communicate(timeout=60)
    assert late.returncode == cli.EXIT_REFUSED, out
    assert P.read_json(wdir / P.DONE_FILE) is None



def test_no_pause_before_the_childs_first_report(tmp_path, monkeypatch):
    """A trigger that comes while the child is still STARTING waits for its first rollout report: before
    it, the trainer's stop handlers may not be installed and a SIGTERM kills it unsaved (rc -15 → HOLD)."""
    h = _Harness(tmp_path, monkeypatch, defer_report=True)

    def operator():
        assert h.arm("--guest-start-timeout-s", "1") == 0           # due NOW; no guest: MISSED → resume
        time.sleep(2.0)                                              # 20 request polls
        held_back = (P.window_root(h.run_dir) / P.REQUEST_FILE).exists() and not any(
            k["event"] == "exit_interrupted" for k in _children(h.run_dir))
        (h.run_dir / "defer_report").unlink()
        return held_back

    t, res = _in_thread(operator)
    try:
        code = h.supervise()
    finally:
        h.shutdown.set()
    t.join(60)
    assert "e" not in res, res.get("e")
    assert res["v"] is True, "the window paused a child that had not reported yet"
    assert code == 0
    assert P.read_current(h.run_dir)["outcome"] == P.MISSED, "the window never ran once the child reported"
