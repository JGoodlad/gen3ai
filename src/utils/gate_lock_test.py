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
from utils import procfs as P
from utils.contention import scale_timeout
from utils.paths import repo_path, src_root

pytestmark = pytest.mark.integration

_SH = str(repo_path("scripts", "ops", "gate_lock.sh"))


def _env(d, n=2) -> dict:
    env = {k: v for k, v in os.environ.items() if k != GL.HELD_ENV}
    env.update({GL.DIR_ENV: str(d), GL.SLOTS_ENV: str(n),
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
        time.sleep(1.5)
        assert not c.exists(), "a THIRD gate ran while two slots were held"
        waiting = [P.waiters(P.file_key(str(GL.slot_path(i, d))) or "") for i in range(2)]
        assert any(third.pid in w for w in waiting), f"the third taker is not a kernel waiter: {waiting}"
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


def test_slot_count_is_declared_and_refuses_nonsense(monkeypatch):
    monkeypatch.delenv(GL.SLOTS_ENV, raising=False)
    assert GL.slot_count() == GL.GATE_SLOTS == 2
    monkeypatch.setenv(GL.SLOTS_ENV, "0")
    with pytest.raises(ValueError):
        GL.slot_count()
