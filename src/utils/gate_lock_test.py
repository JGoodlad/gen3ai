"""``utils.gate_lock`` / ``scripts/ops/gate_lock.sh`` against REAL processes — never the real slots
(every test points ``$GEN3AI_GATE_LOCK_DIR`` at a temp dir and strips an inherited holder marker, since
the routine gate running THIS file may itself hold a real slot).

Load-bearing pins, each FAILS on revert of what it guards:
* a THIRD taker waits while two hold (N = 2), and gets in when one holder is SIGKILLed — the kernel
  frees a crashed holder's slot;
* a child of a holder does not take a second slot and does not block (N = 1);
* every slot held by an ANCESTOR (a bare flock) is a self-deadlock, refused at once (exit 3);
* the timeout is inside (exit 4).

``integration``: spawns and kills its own processes, by explicit PID. ~10 s.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from utils import gate_lock as GL
from utils import heavy_slots as HS
from utils import procfs as P
from utils.contention import scale_timeout
from utils.paths import repo_path, src_root

pytestmark = pytest.mark.integration

_SH = str(repo_path("scripts", "ops", "gate_lock.sh"))


def _env(d, n=2) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in (GL.HELD_ENV, HS.HELD_ENV)}
    # never the REAL heavy slots (a gate takes one while a run is live), and never a wait on them here
    env.update({GL.DIR_ENV: str(d), GL.SLOTS_ENV: str(n), HS.DIR_ENV: str(d / "heavy"), HS.SLOTS_ENV: "16",
                "PYTHONPATH": os.pathsep.join([str(src_root()), os.environ.get("PYTHONPATH", "")])})
    return env


def _wait_for(pred, what: str, timeout: float = 15.0) -> None:
    end = time.monotonic() + scale_timeout(timeout)
    while time.monotonic() < end:
        if pred():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def _holder(d, n, marker, tmp_path) -> subprocess.Popen:
    """A gate_lock.sh holder whose command touches ``marker`` then sleeps."""
    return subprocess.Popen([_SH, sys.executable, "-c",
                             f"import pathlib, time; pathlib.Path({str(marker)!r}).touch(); time.sleep(120)"],
                            env=_env(d, n), stderr=open(tmp_path / f"{marker.name}.err", "w"),
                            start_new_session=True)


def _kill_tree(p: subprocess.Popen) -> None:
    procs = P.snapshot()
    kids = P.descendants(p.pid, P.children_map(procs))
    for q in [p.pid, *kids]:
        try:
            os.kill(q, signal.SIGKILL)
        except ProcessLookupError:
            pass
    p.wait(timeout=10)
    for q in kids:                       # the orphaned command: reap it too, by explicit pid
        try:
            os.kill(q, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _slot_holders(d, n):
    return [P.holders(P.file_key(str(GL.slot_path(i, d))) or "") for i in range(n)]


def test_a_third_taker_waits_and_a_crashed_holders_slot_frees(tmp_path):
    d = tmp_path / "slots"
    a, b = tmp_path / "a", tmp_path / "b"
    ha, hb = _holder(d, 2, a, tmp_path), _holder(d, 2, b, tmp_path)
    third = None
    try:
        _wait_for(lambda: a.exists() and b.exists(), "two holders inside")
        c = tmp_path / "c"
        third = _holder(d, 2, c, tmp_path)

        def third_waits():
            return any(third.pid in P.waiters(P.file_key(str(GL.slot_path(i, d))) or "") for i in range(2))
        # wait until it IS a kernel waiter, THEN hold it there >= 1 s: a fixed sleep raced a slow interpreter
        # start on a loaded box (the taker reached the wait late and "waited" < 1 s — flaky, 2026-10-12)
        _wait_for(third_waits, "the third taker to block in the kernel")
        time.sleep(1.2)
        assert not c.exists(), "a THIRD gate ran while two slots were held"
        assert third_waits(), "the third taker is not a kernel waiter"
        # CRASH one holder: SIGKILL, no cleanup. The kernel must free its slot.
        _kill_tree(ha)
        _wait_for(c.exists, "the third taker to get the crashed holder's slot")
        err = (tmp_path / "c.err").read_text()
        assert "acquired slot" in err and "after waiting" in err, err
        waited = float(err.split("after waiting ")[1].split(" s")[0])
        assert waited >= 1.0, err
    finally:
        for p in (ha, hb, third):
            if p is not None and p.poll() is None:
                _kill_tree(p)


def test_a_child_of_a_holder_takes_no_second_slot_and_does_not_block(tmp_path):
    d = tmp_path / "slots"
    inner = f"{_SH} --timeout-s 5 {sys.executable} -c 'print(\"inner-ran\")'"
    r = subprocess.run([_SH, "--timeout-s", "20", "bash", "-c", inner], env=_env(d, 1),
                       capture_output=True, text=True, timeout=scale_timeout(60))
    assert r.returncode == 0, r
    assert "inner-ran" in r.stdout
    assert r.stderr.count("acquired slot") == 1, r.stderr       # only the OUTER took one


def test_a_stale_marker_is_ignored(tmp_path):
    d = tmp_path / "slots"
    env = {**_env(d, 1), GL.HELD_ENV: "1:0"}                      # pid 1 holds nothing here
    r = subprocess.run([_SH, "true"], env=env, capture_output=True, text=True, timeout=scale_timeout(60))
    assert r.returncode == 0 and "STALE" in r.stderr and "acquired slot 0" in r.stderr, r


def test_every_slot_held_by_an_ANCESTOR_is_refused_at_once(tmp_path):
    d = tmp_path / "slots"
    d.mkdir()
    slot = GL.slot_path(0, d)
    slot.touch()
    t0 = time.monotonic()
    r = subprocess.run(["flock", str(slot), _SH, "--timeout-s", "60", "true"], env=_env(d, 1),
                       capture_output=True, text=True, timeout=scale_timeout(60))
    assert r.returncode == 3 and "SELF-DEADLOCK" in r.stderr, r
    assert time.monotonic() - t0 < scale_timeout(10)


def test_the_timeout_is_inside(tmp_path):
    d = tmp_path / "slots"
    a = tmp_path / "a"
    h = _holder(d, 1, a, tmp_path)
    try:
        _wait_for(a.exists, "the holder inside")
        r = subprocess.run([_SH, "--timeout-s", "1", "true"], env=_env(d, 1), capture_output=True,
                           text=True, timeout=scale_timeout(60))
        assert r.returncode == 4 and "gave up" in r.stderr, r
    finally:
        _kill_tree(h)


def test_the_timeout_bounds_the_heavy_slot_wait_too_and_frees_the_gate(tmp_path, monkeypatch):
    """``--timeout-s`` bounds the WHOLE wait: with every heavy slot held (a live run wants one) the gate gives
    up with ``TimeoutError`` (exit 4) and releases its own slot, instead of waiting unbounded while holding
    the gate. Found by the 2026-10-20 core review (D-1): the heavy-slot wait ignored the timeout."""
    import fcntl
    import threading

    d = tmp_path / "slots"
    for k, v in _env(d, 1).items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv(HS.SLOTS_ENV, "1")
    for k in (GL.HELD_ENV, HS.HELD_ENV, HS.BLOCKING_ENV):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(HS, "wanted", lambda **_: "a training run is live (test)")
    heavy = d / "heavy"
    heavy.mkdir(parents=True)
    fd = os.open(str(HS.slot_path(0, heavy)), os.O_RDWR | os.O_CREAT, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)        # the one heavy slot: held by "another job"
    out: dict = {}

    def take() -> None:
        try:
            with GL.gate_slot(timeout_s=0.3, poll_s=0.02, log=lambda _m: None):
                out["ran"] = True
        except TimeoutError as exc:
            out["timeout"] = exc

    try:
        t = threading.Thread(target=take, daemon=True)
        t.start()
        t.join(scale_timeout(30))
        assert not t.is_alive(), "the gate waited past --timeout-s for a heavy slot"
        assert "timeout" in out and "ran" not in out, out
        g = os.open(str(GL.slot_path(0, d)), os.O_RDWR)
        try:
            fcntl.flock(g, fcntl.LOCK_EX | fcntl.LOCK_NB)  # the gate slot is free again
        finally:
            os.close(g)
    finally:
        os.close(fd)


def test_slot_count_is_declared_and_refuses_nonsense(monkeypatch):
    monkeypatch.delenv(GL.SLOTS_ENV, raising=False)
    assert GL.slot_count() == GL.GATE_SLOTS == 1
    monkeypatch.setenv(GL.SLOTS_ENV, "0")
    with pytest.raises(ValueError):
        GL.slot_count()


def test_a_live_training_run_is_WARNED_about_never_acted_on():
    """The live-run caveat is mechanical (rule 5b): a gate taken while a `train_rl_agent.py --run-dir`
    lives prints one line naming the run, the agent cores (every cpu the trainer is NOT pinned to), nice
    19 and the measured -n — and names what is wrong with THIS gate's shape; with no live run, nothing."""
    assert GL.live_run_warning({}) is None
    n = os.cpu_count() or 16
    trainer = [c for c in range(n) if c not in (n - 3, n - 2, n - 1)]
    ok = GL.live_run_warning({"/x/models/ai_v15_arm": 4242}, my_cpus=[n - 3, n - 2, n - 1], my_nice=19,
                             trainer_cpus=trainer)
    assert ok is not None and "ai_v15_arm (pid 4242)" in ok and "nice -n 19" in ok
    assert f"taskset -c {n - 3}-{n - 1}" in ok and f"-n {GL.recommended_workers(3)}" in ok
    assert "THIS gate" not in ok
    bad = GL.live_run_warning({"/x/models/ai_v15_arm": 4242}, my_cpus=list(range(n)), my_nice=0,
                              trainer_cpus=trainer)
    assert bad is not None and "THIS gate" in bad and "nice 0" in bad and "OVERLAP" in bad


def test_status_names_the_holder_its_worktree_and_the_queue_and_a_release_writes_history(tmp_path):
    """A REAL holder writes its slot record (pid, worktree, -n, cpus, nice); `--status --json` reads it
    while `/proc/locks` lists that pid holding the slot, and lists a REAL blocked taker as waiting; the
    release removes the record and appends the gate's wall + exit code to the ETA history."""
    import json

    from utils import gate_lock_status as GS

    d = tmp_path / "slots"
    a = tmp_path / "a"
    gate_cmd = [sys.executable, "-c",
                f"import pathlib, time; pathlib.Path({str(a)!r}).touch(); time.sleep(120)",
                "-m", "not slow and not e2e", "-n", "3", "pytest"]
    h = subprocess.Popen([_SH, *gate_cmd], env=_env(d, 1), stderr=open(tmp_path / "h.err", "w"),
                         cwd=str(tmp_path), start_new_session=True)
    w = None
    try:
        _wait_for(a.exists, "the holder inside")
        w = subprocess.Popen([_SH, "true"], env=_env(d, 1), stderr=open(tmp_path / "w.err", "w"),
                             start_new_session=True)
        _wait_for(lambda: P.waiters(P.file_key(str(GL.slot_path(0, d))) or ""), "the taker to block")
        r = subprocess.run([_SH, "--status", "--json", "--sample-s", "0"], env=_env(d, 1),
                           capture_output=True, text=True, timeout=scale_timeout(60))
        assert r.returncode == 0, r
        st = json.loads(r.stdout)
        held = st["slots"][0]
        assert held["pid"] == h.pid and held["since_source"] == "record", held
        assert os.path.realpath(held["cwd"]) == os.path.realpath(tmp_path) and held["workers"] == 3 and held["cpus"] >= 1, held
        assert [x["pid"] for x in st["waiters"]] == [w.pid], st["waiters"]
        human = subprocess.run([_SH, "--status", "--sample-s", "0"], env=_env(d, 1), capture_output=True,
                               text=True, timeout=scale_timeout(60)).stdout
        assert f"HELD by pid {h.pid}" in human and "1 waiting" in human, human
    finally:
        _kill_tree(h)
        if w is not None:
            w.wait(timeout=scale_timeout(30))      # the waiter gets the freed slot, runs `true`, exits
    # the SIGKILLed holder ran no release, so it wrote NO history line; the only one is the waiter's `true`
    first = GS.read_history(d)
    assert len(first) == 1 and first[0]["workers"] is None and not first[0]["routine"], first
    ok = subprocess.run([_SH, *gate_cmd[:2], "pass", "-m", "not slow and not e2e", "-n", "3", "pytest"],
                        env=_env(d, 1), capture_output=True, text=True, timeout=scale_timeout(60))
    assert ok.returncode == 0, ok
    rows = GS.read_history(d)
    assert len(rows) == 2 and rows[-1]["rc"] == 0 and rows[-1]["workers"] == 3 and rows[-1]["routine"], rows
    assert rows[0]["rc"] == 0 and rows[0]["workers"] is None            # the waiter's `true`
    assert GS.read_record(d, 0) is None, "a clean release removes its record"


def test_live_training_runs_reads_a_planted_proc(tmp_path):
    """`utils.procfs.live_training_runs` keeps only train_rl_agent.py processes that name a --run-dir,
    and resolves a relative run dir against that process's cwd."""
    from utils import procfs

    def plant(pid, argv, cwd):
        d = tmp_path / str(pid)
        d.mkdir()
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
        (d / "cwd").symlink_to(cwd)
    plant(11, ["python3", "src/main/train_rl_agent.py", "--run-dir", "models/arm"], tmp_path)
    plant(12, ["python3", "src/main/train_rl_agent.py", "--debug"], tmp_path)
    plant(13, ["python3", "-m", "pytest", "--run-dir", "x"], tmp_path)
    assert procfs.live_training_runs(str(tmp_path)) == {str(tmp_path / "models/arm"): 11}
