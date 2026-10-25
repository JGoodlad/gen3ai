"""The T46 GUEST's job is a process GROUP, not the guest's pid (backlog row "the GPU window judges its guest by the
WRAPPER pid", T46 adversarial review 2026-10-20). Real processes, a real guest subprocess, no GPU, no launcher.

Each test names what it pins and FAILS on revert of it:
  * the guest runs its command as the leader of a session of its own and records that group in the claim;
  * ``preempt`` stops exactly that job (a nested group inside it too, a SIGTERM-ignorer with ``--kill``) and nothing
    else: not the guest's wrapper, not a bystander in the caller's group;
  * a stop request sent to the GUEST is forwarded to its job (the guest still writes DONE);
  * the launcher judges a dead guest wrapper by its job's group: a job still running is named, with its pids, in
    the HOLD; a job that has ended says so.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import pytest

from main.exit_codes import TrainExitCode
from main.launcher.gpu_window import cli
from main.launcher.gpu_window import protocol as P
from main.launcher.gpu_window.controller import WindowController
from utils import procfs
from utils.paths import src_root

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _private_gpu_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("GEN3AI_GPU_LOCK", str(tmp_path / "gpu.lock"))


@pytest.fixture
def run_dir(tmp_path) -> Path:
    d = tmp_path / "models" / "rb_toy"
    d.mkdir(parents=True)
    return d


def _alive(pid: int) -> bool:
    p = procfs.read_stat(pid)
    return p is not None and p.state not in ("Z", "X")


def _wait_for(pred, what: str, timeout: float = 30.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.05)
    raise AssertionError(f"timed out after {timeout:.0f} s waiting for {what}")


def _killpg(pgid: int) -> None:
    if pgid in (0, 1, os.getpgrp(), os.getsid(0)):          # a mutated implementation must never be able to kill the runner
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _controller(run_dir: Path, events: List[str], **kw) -> WindowController:
    return WindowController(lambda: str(run_dir), add_event=events.append, set_status=lambda s: None,
                            gpu_pids=lambda: [], find_checkpoint=lambda rd: None, request_poll_s=0.0,
                            wait_poll_s=0.01, **kw)


def _window_at_go(run_dir: Path, **arm: str) -> "tuple[WindowController, P.Request, Path]":
    """A window armed, triggered and at GO (no launcher thread, no child): what a guest needs to claim."""
    ctl = _controller(run_dir, [])
    ctl.announce()
    extra = [x for kv in arm.items() for x in (f"--{kv[0].replace('_', '-')}", kv[1])]
    assert cli.main(["arm", str(run_dir), "--no-precheck", "test", "--no-lease-check", "test", *extra]) == 0
    req = ctl.due()
    assert req is not None
    wdir = P.window_dir(run_dir, req.window_id)
    P.write_json_atomic(wdir / P.GO_FILE, {"window_id": req.window_id, "go_epoch": time.time()})
    return ctl, req, wdir


def _guest(run_dir: Path, *cmd: str) -> subprocess.Popen:
    """The guest as a subprocess. Its output goes to a FILE (``.log_path``): a pipe would stay open as long as ANY
    process of the job holds it, which would turn a leftover into a 60 s wait instead of a failed assertion."""
    env = {**os.environ, "PYTHONPATH": str(src_root()), "CUDA_VISIBLE_DEVICES": ""}
    log_path = run_dir.parent / f"guest_{time.monotonic_ns()}.log"
    with open(log_path, "w") as log:
        g = subprocess.Popen([sys.executable, "-m", "main.launcher.gpu_window", "guest", str(run_dir),
                              "--wait-max-min", "2", "--no-lease-check", "test", "--", *cmd],
                             env=env, stdout=log, stderr=subprocess.STDOUT)
    g.log_path = log_path                                     # type: ignore[attr-defined]
    return g


def _mark_claimed(run_dir: Path, wid: str) -> None:
    """What the launcher's CLAIMED phase looks like to `preempt` (no launcher runs in these tests)."""
    P.write_json_atomic(P.window_root(run_dir) / P.CURRENT_FILE, {"window_id": wid, "phase": P.CLAIMED, "reason": "test"})


_TREE = r"""
import json, os, subprocess, sys, signal
if sys.argv[2] == "ignore-term":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
a = subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN) if %r else None; time.sleep(60)" % (sys.argv[2] == "ignore-term")])
b = subprocess.Popen(["sleep", "60"], process_group=0)       # a NESTED group, as a gpu_lock.sh inside the job makes
json.dump({"me": os.getpid(), "a": a.pid, "b": b.pid}, open(sys.argv[1], "w"))
a.wait()
"""


def test_the_guest_runs_its_command_as_the_leader_of_its_own_session_and_records_the_group_in_the_claim(run_dir, tmp_path):
    _, req, wdir = _window_at_go(run_dir)
    out = tmp_path / "me.json"
    g = _guest(run_dir, sys.executable, "-c",
               "import json,os,sys; json.dump({'pid': os.getpid(), 'pgrp': os.getpgrp(), 'sid': os.getsid(0)}, open(sys.argv[1], 'w'))",
               str(out))
    g.wait(timeout=60)
    text = g.log_path.read_text()                             # type: ignore[attr-defined]
    assert g.returncode == 0, text
    me = json.loads(out.read_text())
    assert me["pgrp"] == me["sid"] == me["pid"], "the command leads a session (and so a group) of its own"
    assert me["pgrp"] != os.getpgrp() and me["sid"] != os.getsid(0), "...not the guest's / the launching tool's"
    claim = P.read_json(wdir / P.CLAIM_FILE)
    assert claim["by"] == "guest" and claim["guest"]["pid"] == g.pid, "the claim still names the guest wrapper"
    assert claim["job"]["pgid"] == claim["job"]["sid"] == me["pid"], claim
    assert claim["job"]["leader"]["pid"] == me["pid"] and claim["job"]["leader"]["start_ticks"] > 0
    assert P.read_json(wdir / P.DONE_FILE)["pgid"] == me["pid"]
    assert "preempt" in text and f"-{me['pid']}" in text, "the guest prints how to preempt it"


@pytest.mark.parametrize("mode", ["term", "kill"])
def test_preempt_stops_the_whole_job_a_nested_group_included_and_nothing_else(run_dir, tmp_path, mode):
    """SIGTERM reaches the command, its same-group child AND the nested group (they share the job's SESSION); a job
    that ignores SIGTERM needs ``--kill``. The guest wrapper (another group) and a bystander in the caller's group
    survive; the guest writes DONE with the job's death as its rc."""
    _, req, wdir = _window_at_go(run_dir)
    pids = tmp_path / "tree.json"
    bystander = subprocess.Popen(["sleep", "60"])                       # the TEST's group
    g = _guest(run_dir, sys.executable, "-c", _TREE, str(pids), "ignore-term" if mode == "kill" else "plain")
    tree = None
    try:
        _wait_for(lambda: pids.exists() and pids.read_text().strip() and (P.read_json(wdir / P.CLAIM_FILE) or {}).get("job"),
                  "the job tree and its claim record")
        tree = json.loads(pids.read_text())
        _mark_claimed(run_dir, req.window_id)
        if mode == "kill":
            assert cli.main(["preempt", str(run_dir)]) == 0                 # SIGTERM: the job ignores it
            time.sleep(0.5)
            assert _alive(tree["me"]) and _alive(tree["a"]), "the SIGTERM-ignoring job must survive a plain preempt"
            assert cli.main(["preempt", str(run_dir), "--kill"]) == 0
        else:
            assert cli.main(["preempt", str(run_dir)]) == 0
        g.wait(timeout=60)
        text = g.log_path.read_text()                         # type: ignore[attr-defined]
        for name in ("me", "a", "b"):
            _wait_for(lambda n=name: not _alive(tree[n]), f"{name} to die with the job", timeout=15)
        assert _alive(bystander.pid), "a preempt must not touch the caller's own group"
        assert g.returncode == 128 + (signal.SIGKILL if mode == "kill" else signal.SIGTERM), text
        done = P.read_json(wdir / P.DONE_FILE)
        assert done["rc"] == g.returncode and done["pgid"] == tree["me"], done
    finally:
        for pid in tree.values() if tree else ():
            _killpg(pid)
        bystander.kill()
        bystander.wait(timeout=10)
        g.kill()
        g.wait(timeout=10)


def test_preempt_refuses_without_a_recorded_job_and_outside_a_running_guest(run_dir):
    _, req, wdir = _window_at_go(run_dir)
    assert cli.main(["preempt", str(run_dir)]) == cli.EXIT_REFUSED, "phase PAUSING: no guest yet"
    _mark_claimed(run_dir, req.window_id)
    assert P.create_json_exclusive(wdir / P.CLAIM_FILE, {"by": "launcher", "window_id": req.window_id, "missed_epoch": 0})
    assert cli.main(["preempt", str(run_dir)]) == cli.EXIT_REFUSED, "a claim with no job group on record"
    P.write_json_atomic(wdir / P.CLAIM_FILE, {"by": "guest", "window_id": req.window_id,
                                              "guest": P.process_identity(os.getpid()), "job": None})
    assert cli.main(["preempt", str(run_dir)]) == cli.EXIT_REFUSED, "the guest has not recorded its job yet"


@pytest.mark.parametrize("which", ["the_guests_own_group", "the_callers_own_group", "the_callers_own_session"])
def test_preempt_never_signals_the_guests_or_the_callers_own_group_or_session(run_dir, capsys, monkeypatch, which):
    """A claim whose job record names a group that is the GUEST's (its pid) or a group / session of the CALLER's must be
    refused, not obeyed: a corrupt record cannot make a preempt kill its operator. The stand-in group is a throwaway `sleep` in a
    session of its own, so a regression here costs that sleep and nothing else."""
    _, req, wdir = _window_at_go(run_dir)
    _mark_claimed(run_dir, req.window_id)
    victim = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        guest_pid = victim.pid if which == "the_guests_own_group" else 1
        if which == "the_callers_own_group":
            monkeypatch.setattr(cli.os, "getpgrp", lambda: victim.pid)
        if which == "the_callers_own_session":
            monkeypatch.setattr(cli.os, "getsid", lambda pid: victim.pid)
        P.write_json_atomic(wdir / P.CLAIM_FILE, {"by": "guest", "window_id": req.window_id,
                                                  "guest": {"pid": guest_pid, "start_ticks": 1}, "job": _job_of(victim)})
        assert cli.main(["preempt", str(run_dir)]) == cli.EXIT_REFUSED
        assert "refusing to signal" in capsys.readouterr().err
        assert _alive(victim.pid), "a refused preempt signals nothing"
    finally:
        monkeypatch.undo()
        _killpg(victim.pid)
        victim.wait(timeout=10)


def test_a_stop_request_sent_to_the_guest_is_forwarded_to_its_job_and_the_guest_still_writes_done(run_dir, tmp_path):
    """Before: SIGTERM to the guest killed the wrapper and left the command running (no DONE, a HOLD with the job
    still on the card). Now the guest forwards it to the job's session (its nested group too) and reports the end."""
    _, req, wdir = _window_at_go(run_dir)
    pids = tmp_path / "tree.json"
    g = _guest(run_dir, sys.executable, "-c", _TREE, str(pids), "plain")
    tree = None
    try:
        _wait_for(lambda: pids.exists() and pids.read_text().strip() and (P.read_json(wdir / P.CLAIM_FILE) or {}).get("job"),
                  "the job tree and its claim record")
        tree = json.loads(pids.read_text())
        g.send_signal(signal.SIGTERM)
        g.wait(timeout=60)
        assert g.returncode == 128 + signal.SIGTERM
        for name in ("me", "a", "b"):
            _wait_for(lambda n=name: not _alive(tree[n]), f"{name} to receive the forwarded SIGTERM", timeout=15)
        assert P.read_json(wdir / P.DONE_FILE)["rc"] == 128 + signal.SIGTERM
    finally:
        for pid in tree.values() if tree else ():
            _killpg(pid)
        g.kill()
        g.wait(timeout=10)


# ── the launcher judges a dead guest wrapper by its JOB's group ────────────────────────────────────

def _job_of(proc: subprocess.Popen) -> dict:
    return {"pgid": proc.pid, "sid": proc.pid, "leader": P.process_identity(proc.pid), "started_epoch": time.time()}


def _hold_reason(run_dir: Path, ctl: WindowController, req: P.Request) -> str:
    seen: List[str] = []

    def should_quit() -> bool:
        cur = P.read_current(run_dir) or {}
        if cur.get("phase") == P.HOLD:
            seen.append(cur["reason"])
            return True
        return False
    assert ctl.run_window(req, int(TrainExitCode.INTERRUPTED), time.time() - 1, should_quit) == "quit"
    assert seen, "the window ended without a HOLD"
    return seen[0]


def _claimed_window(run_dir: Path, tmp_path: Path, guest: dict, job: Optional[dict], **arm: str):
    ck = tmp_path / "c.zip"
    ck.write_text("x")
    ctl = WindowController(lambda: str(run_dir), add_event=lambda e: None, set_status=lambda s: None,
                           gpu_pids=lambda: [], find_checkpoint=lambda rd: str(ck), request_poll_s=0.0, wait_poll_s=0.01)
    ctl.announce()
    extra = [x for kv in arm.items() for x in (f"--{kv[0].replace('_', '-')}", kv[1])]
    assert cli.main(["arm", str(run_dir), "--no-precheck", "test", "--no-lease-check", "test", *extra]) == 0
    req = ctl.due()
    wdir = P.window_dir(run_dir, req.window_id)
    assert P.create_json_exclusive(wdir / P.CLAIM_FILE, {"by": "guest", "window_id": req.window_id, "guest": guest, "job": job})
    return ctl, req


def test_a_guest_wrapper_that_died_while_its_job_still_runs_holds_naming_the_group_and_its_pids(run_dir, tmp_path):
    """The wrapper's pid says nothing about the card: its job (here a `sleep` in a session of its own) is still
    there. The HOLD must say so, with the pids, so an operator does not `resume` onto a busy card."""
    job_proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        ctl, req = _claimed_window(run_dir, tmp_path, {"pid": os.getpid(), "start_ticks": -1}, _job_of(job_proc))
        reason = _hold_reason(run_dir, ctl, req)
        assert "died without writing DONE" in reason
        assert f"pgid {job_proc.pid}" in reason and "STILL RUNS" in reason and f"{job_proc.pid} (sleep)" in reason, reason
        assert "preempt" in reason
    finally:
        _killpg(job_proc.pid)
        job_proc.wait(timeout=10)


def test_a_dead_guest_whose_job_has_ended_says_so_and_one_with_no_record_says_that(run_dir, tmp_path):
    job_proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
    job = _job_of(job_proc)
    _killpg(job_proc.pid)
    job_proc.wait(timeout=10)
    ctl, req = _claimed_window(run_dir, tmp_path, {"pid": os.getpid(), "start_ticks": -1}, job)
    assert "has no live process" in _hold_reason(run_dir, ctl, req)
    P.write_json_atomic(P.window_root(run_dir) / P.CURRENT_FILE, {"phase": P.ABANDONED})
    ctl2, req2 = _claimed_window(run_dir, tmp_path, {"pid": os.getpid(), "start_ticks": -1}, None)
    assert "no job group was recorded" in _hold_reason(run_dir, ctl2, req2)


def test_status_reports_the_guests_job_by_its_group(run_dir, tmp_path, capsys):
    job_proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        ctl, req = _claimed_window(run_dir, tmp_path, P.process_identity(os.getpid()), _job_of(job_proc))
        P.write_json_atomic(P.window_root(run_dir) / P.CURRENT_FILE, {"window_id": req.window_id, "phase": P.CLAIMED,
                                                                      "reason": "test", "since_epoch": time.time()})
        assert cli.main(["status", str(run_dir)]) == 0
        out = capsys.readouterr().out
        assert f"guest job: its job (pgid {job_proc.pid}) STILL RUNS 1 process(es): {job_proc.pid} (sleep)" in out, out
    finally:
        _killpg(job_proc.pid)
        job_proc.wait(timeout=10)


def test_an_overrun_holds_naming_the_job_that_may_still_hold_the_gpu(run_dir, tmp_path):
    job_proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        ctl, req = _claimed_window(run_dir, tmp_path, P.process_identity(os.getpid()), _job_of(job_proc),
                                   max_window_min="0.001")
        reason = _hold_reason(run_dir, ctl, req)
        assert "overran" in reason and f"pgid {job_proc.pid}" in reason and "STILL RUNS" in reason, reason
    finally:
        _killpg(job_proc.pid)
        job_proc.wait(timeout=10)
