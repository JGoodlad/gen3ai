"""``monitor`` never blocks the arm end (the 10-27 03:14 incident) and keeps ``tail -F``'s semantics without a handle.

d3 3504 ended cleanly and passed K9(b); the thin dry run then REFUSED, "pid(s) 2621969 have the run dir open": that
pid was ``arm_chain monitor``, whose follower held ``launcher_child.full.log`` open for its whole life. The supervisor
HELD and the GPU idled. The fix: the follower holds no file open between reads, and an arm at ENDED or later loses its
run-dir files (one last drain, then never opened again). ``test_the_10_27_incident…`` reproduces the incident with a
real monitor process, a fake run dir and the REAL thin planner's ``/proc`` scan; it fails when the fix is reverted.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Iterator, List

import pytest

from main.ops.arm_chain import chain as C
from main.ops.arm_chain import monitor
from main.ops.arm_chain.armend import THIN_PLANNER
from main.ops.arm_chain.chain_test import FakeShipper, deps, runner_for, world_then_wait
from main.ops.arm_chain.live import LiveRun
from main.ops.arm_chain.testing_fakes import World
from utils.paths import repo_path, src_path

#: a run name no other test's fake launch carries: the planner refuses a run any live launcher on the box names
ARM, SEED = "cb47mon", 4701
RUN = f"rb_{ARM}_s{SEED}"
KEY = f"arm:{RUN}"


@pytest.fixture
def w(tmp_path: Any) -> Iterator[World]:
    yield from world_then_wait(tmp_path)


def _fds_into(pid: Any, root: str) -> List[str]:
    """The files under ``root`` that process ``pid`` holds open right now (what the thin planner's scan reads)."""
    d, out = f"/proc/{pid}/fd", []
    for fd in os.listdir(d):
        try:
            t = os.readlink(os.path.join(d, fd))
        except OSError:
            continue
        if t.startswith(os.path.realpath(root) + os.sep):
            out.append(t)
    return out


def _phase(ch: C.Chain, phase: str) -> None:
    ch.state["history"].append({"at": "x", "item": KEY, "from": ch.state["items"][KEY]["phase"], "to": phase,
                                "note": ""})
    ch.state["items"][KEY]["phase"] = phase
    ch._save()


def _finished_run_dir(models: str) -> str:
    """A finished fork as the thin planner reads it: checkpoints, its metadata, ``final_model.zip``, the full log."""
    rd = os.path.join(models, RUN)
    os.makedirs(os.path.join(rd, "checkpoints"))
    for s in (1_000_000, 2_000_000, 2_500_000):
        open(os.path.join(rd, "checkpoints", f"checkpoint_{s}_steps.zip"), "wb").write(b"z")
    json.dump({"lineage": {"fork_step": 0}}, open(os.path.join(rd, "metadata.json"), "w"))
    open(os.path.join(rd, "final_model.zip"), "wb").write(b"final")
    open(os.path.join(rd, "launcher_child.full.log"), "w").write("trainer line\n")
    return rd


class _Lines:
    """The monitor's stdout, read by a thread, so a wait is bounded even when no line comes."""

    def __init__(self, proc: "subprocess.Popen[str]"):
        self.proc, self.got, self.q = proc, [], queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for line in self.proc.stdout or []:
            self.q.put(line.rstrip("\n"))
        self.q.put(None)

    def wait(self, needle: str, timeout_s: float = 60.0) -> None:
        deadline = time.monotonic() + timeout_s
        while (left := deadline - time.monotonic()) > 0:
            try:
                line = self.q.get(timeout=left)
            except queue.Empty:
                break
            if line is None:
                raise AssertionError(f"the monitor exited (rc {self.proc.poll()}) before {needle!r}: {self.got[-20:]}")
            self.got.append(line)
            if needle in line:
                return
        raise AssertionError(f"no {needle!r} from the monitor in {timeout_s:g} s: {self.got[-20:]}")


@pytest.mark.integration
def test_the_10_27_incident_a_running_monitor_never_blocks_the_arm_ends_thin(w):
    """A real ``monitor`` process follows a RUNNING arm's run dir; the arm ENDS; the real thin planner's dry run then
    PASSES while the monitor is still alive (it refused with ``pid(s) <monitor> have the run dir open`` on 10-27)."""
    items = [w.arm(ARM, SEED, w.pin_a)]
    chain_file = os.path.join(w.root, "chain.json")
    json.dump(w.chain_dict(items), open(chain_file, "w"))
    ch = C.Chain(w.spec(items), deps(w, runner_for(w), FakeShipper(w.repo)))
    _phase(ch, "RUNNING")
    rd = _finished_run_dir(w.models)
    log, status = os.path.join(rd, "launcher_child.full.log"), os.path.join(w.job, "status.txt")
    env = {**os.environ, "PYTHONPATH": src_path(), "CUDA_VISIBLE_DEVICES": ""}
    code = ("import sys; from main.ops.arm_chain.monitor import run_monitor; "
            "run_monitor(sys.argv[1], interval_s=0.05, max_hours=1)")
    mon = subprocess.Popen([sys.executable, "-u", "-c", code, chain_file], cwd=w.root, env=env, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = _Lines(mon)
    got = out.got
    try:
        out.wait("SUPERVISOR DEAD")        # its first pass ran (no supervisor holds the lock): past -n0
        with open(log, "a") as f:
            f.write("Traceback (most recent call last):\n")
        out.wait("[launcher_child.full.log] Traceback")      # it follows the RUNNING arm's run dir
        _phase(ch, "ENDED")
        with open(log, "a") as f:
            f.write("Child crashed (exit 1)\n")                             # the last line, written as the arm ends
        out.wait("[launcher_child.full.log] Child crashed")    # never lost to the drop
        with open(status, "a") as f:                                        # a pass that began after ENDED was saved
            f.write("[t9] ARMCHAIN[t]: HOLD: sync marker\n")
        out.wait("HOLD: sync marker")
        dry = subprocess.run([sys.executable, repo_path(THIN_PLANNER), "--thin-run", RUN, "--models-dir", w.models],
                             cwd=w.root, env=env, capture_output=True, text=True, timeout=300)
        assert dry.returncode == 0 and "NOTHING WAS DELETED" in dry.stdout, dry.stdout + dry.stderr
        assert _fds_into(mon.pid, rd) == [], got
        assert mon.poll() is None, "the monitor must keep running beside the thin"
        with open(log, "a") as f:
            f.write("WEDGE: after the end\n")                               # an ended arm's run dir is not read again
        with open(status, "a") as f:
            f.write("[t10] ARMCHAIN[t]: HOLD: second marker\n")
        out.wait("HOLD: second marker")
        assert not any("after the end" in g for g in got), got
    finally:
        mon.terminate()
        mon.wait(timeout=30)


def test_the_follower_holds_no_file_open_between_polls(tmp_path):
    f = tmp_path / "launcher_child.full.log"
    f.write_text("old\n")
    fol = monitor.Follower()
    assert fol.poll([str(f)]) == []                                          # present at the first poll: from its end
    with open(f, "a") as h:
        h.write("new\n")
    assert fol.poll([str(f)]) == [(str(f), "new")]
    assert _fds_into(os.getpid(), str(tmp_path)) == []


def test_a_rotation_by_rename_is_drained_without_a_held_handle(tmp_path):
    """``tail -F``: the old generation's unread tail (found by inode as ``<name>.1``), then the new file from 0."""
    f = tmp_path / "launcher_child.full.log"
    f.write_text("a\n")
    fol = monitor.Follower()
    fol.poll([str(f)])
    with open(f, "a") as h:
        h.write("b\n")
    os.rename(f, str(f) + ".1")
    with open(str(f) + ".1", "a") as h:
        h.write("c (no newline: the generation is final)")
    f.write_text("dddddd\n")
    assert [ln for _, ln in fol.poll([str(f)])] == ["b", "c (no newline: the generation is final)", "dddddd"]
    f.write_text("")                                                         # truncated in place: restart at 0
    with open(f, "a") as h:
        h.write("e\n")
    assert [ln for _, ln in fol.poll([str(f)])] == ["e"]
    assert _fds_into(os.getpid(), str(tmp_path)) == []


def test_a_dropped_path_is_drained_once_then_never_read_again(tmp_path):
    f = tmp_path / "x.log"
    f.write_text("")
    fol = monitor.Follower()
    fol.poll([str(f)])
    with open(f, "a") as h:
        h.write("Traceback: last words\n")
    assert fol.poll([]) == [(str(f), "Traceback: last words")]
    with open(f, "a") as h:
        h.write("later\n")
    assert fol.poll([]) == [] and fol.pos == {}


@pytest.mark.parametrize("phase", monitor.POST_RUN)
def test_chain_files_never_lists_an_ended_arms_run_dir(w, phase):
    items = [w.arm(ARM, SEED, w.pin_a), w.arm("a2", 2, w.pin_b)]
    spec = w.spec(items)
    rd1 = os.path.join(w.models, RUN)
    state = {"items": {KEY: {"phase": phase}, "arm:rb_a2_s2": {"phase": "PENDING"}}}
    files = monitor.chain_files(spec, state, [], lambda k, m: None)
    assert not any(p.startswith(rd1 + os.sep) for p in files), files
    assert os.path.join(w.models, "rb_a2_s2", "launcher_child.full.log") in files
    running = monitor.chain_files(spec, {"items": {KEY: {"phase": "RUNNING"}}}, [], lambda k, m: None)
    assert os.path.join(rd1, "launcher_child.full.log") in running
    # a LIVE launcher in that dir (a by-hand resume) is followed again: the thin refuses a run a launcher names anyway
    resumed = LiveRun(pid=1, start_ticks=1, elapsed_s=9, run_dir=rd1, source="child", stdout=os.path.join(w.job, "x.out"))
    open(resumed.stdout, "w").close()
    assert os.path.join(rd1, "launcher_child.full.log") in monitor.chain_files(spec, state, [resumed], lambda k, m: None)


def test_post_run_is_every_phase_from_ended_on():
    assert monitor.POST_RUN == ("ENDED", "COMPLETE", "K9B_PASS", "THINNED", "RECORDED", "DONE", "DONE_FAILED")
    assert set(C.ARM_PHASES) - set(monitor.POST_RUN) == {"PENDING", "PRECHECKED", "GOING", "LAUNCHED", "SPAWNED",
                                                        "BANNER_OK", "RUNNING"}
