"""The GPU LEASE (``utils.gpu_lease``) and ``utils.gpu_lock``'s fail-fast behaviour against it — against REAL
holder processes on a temp lock file, never the real GPU lock and never the GPU. CPU only.

The owner's rule (2026-10-03): nobody blocks on the GPU. Each test here FAILS if the behaviour it pins is
reverted: a refusal that waited, a token that was not honoured, a stale record that counted as a lease.
Waits are on CONDITIONS (``_wait_for``), never a sleep used as timing; "immediately" is pinned both by the
elapsed time of an in-process call AND structurally (no waiter ever appeared in ``/proc/locks``).

``integration``: spawns and signals its own processes. ~15 s.
"""
from __future__ import annotations

import dataclasses
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, List

import pytest

from utils import gpu_lease as L
from utils import gpu_lock as G
from utils import procfs as P
from utils.paths import src_root

pytestmark = pytest.mark.integration


def _env(lock) -> dict:
    e = {k: v for k, v in os.environ.items() if k not in (L.TOKEN_ENV, L.TOKEN_FILE_ENV, G.HELD_ENV)}
    return {**e, G.GPU_LOCK_ENV: str(lock),
            "PYTHONPATH": os.pathsep.join([str(src_root()), os.environ.get("PYTHONPATH", "")])}


def _wait_for(pred, what: str, timeout: float = 30.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


def _free(lock) -> bool:
    fd = os.open(str(lock), os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(fd)


def _holders(lock) -> List[int]:
    key = P.file_key(str(lock))
    return P.holders(key) if key else []


@pytest.fixture
def lock(tmp_path, monkeypatch) -> Iterator[Path]:
    """A temp lock path; every env marker cleared; any lease holder it leaves behind is killed by pid."""
    for k in (L.TOKEN_ENV, L.TOKEN_FILE_ENV, G.HELD_ENV):
        monkeypatch.delenv(k, raising=False)
    p = tmp_path / "gpu.lock"
    monkeypatch.setenv(G.GPU_LOCK_ENV, str(p))
    yield p
    rec = L._read_record(p)
    for pid in _holders(p) + ([rec.pid] if rec else []):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _acquire(lock, owner="agent-A", **kw) -> L.Acquired:
    return L.acquire(lock, owner, env={}, **kw)


def _one_off_holder(lock) -> subprocess.Popen:
    """A SIBLING one-off holder: the lock CLI holding the lock around a long sleep (no lease)."""
    h = subprocess.Popen([sys.executable, "-m", "utils.gpu_lock", "--", "sleep", "120"], env=_env(lock),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _wait_for(lambda: h.pid in _holders(lock), "the one-off holder to take the lock")
    return h


def _reap(h: subprocess.Popen) -> None:
    for q in P.descendants(h.pid, P.children_map(P.snapshot())):
        try:
            os.kill(q, signal.SIGKILL)
        except ProcessLookupError:
            pass
    h.kill()
    h.wait(timeout=30)


def _cli(module: str, lock, *args: str, extra_env=None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", module, *args], env={**_env(lock), **(extra_env or {})},
                          capture_output=True, text=True, timeout=120)


# ------------------------------------------------------------------------------ acquire / release / status

def test_acquire_status_release_roundtrip(lock):
    assert L.status(lock).startswith("FREE")
    got = _acquire(lock, "agent-A", note="x5 u1")
    assert got.token and not got.already_held
    lease = L.read_lease(lock)
    assert lease is not None and lease.owner == "agent-A" and lease.note == "x5 u1"
    assert lease.pid != os.getpid() and _holders(lock) == [lease.pid], "a detached holder owns the flock"
    assert got.token not in L.lease_file(lock).read_text(), "the record keeps only the token's HASH"
    assert json.loads(L.lease_file(lock).read_text())["token_sha256"] == L.token_hash(got.token)
    assert L.status(lock).startswith("LEASED") and "agent-A" in L.status(lock)

    out = L.release(lock, env={L.TOKEN_ENV: got.token})
    assert out.startswith("released")
    assert _holders(lock) == [] and _free(lock), "release returns only after the GPU is free"
    assert not L.lease_file(lock).exists() and L.status(lock).startswith("FREE")
    assert L.release(lock, env={}) == "no lease", "release is idempotent"


def test_the_cli_acquire_prints_the_token_and_release_needs_it(lock):
    r = _cli("utils.gpu_lease", lock, "acquire", "--owner", "agent-A", "--token-file", str(lock.parent / "tok"))
    assert r.returncode == 0, r.stderr
    token = r.stdout.strip().removeprefix(f"export {L.TOKEN_ENV}=")
    assert token and r.stdout.startswith(f"export {L.TOKEN_ENV}="), r.stdout
    assert (lock.parent / "tok").read_text().strip() == token
    assert oct((lock.parent / "tok").stat().st_mode & 0o777) == "0o600"
    # a release without the token is refused (exit 7) and changes nothing
    r = _cli("utils.gpu_lease", lock, "release")
    assert r.returncode == L.EXIT_NOT_OWNER and "not your lease" in r.stderr, r
    assert L.read_lease(lock) is not None
    # the token FILE form also authorises it
    r = _cli("utils.gpu_lease", lock, "release", extra_env={L.TOKEN_FILE_ENV: str(lock.parent / "tok")})
    assert r.returncode == 0 and "released" in r.stderr, r
    assert _free(lock)


def test_a_second_acquire_fails_fast_naming_the_holder(lock):
    first = _acquire(lock, "agent-A", note="holding")
    t0 = time.monotonic()
    with pytest.raises(L.GpuLeased) as ei:
        _acquire(lock, "agent-B")
    assert time.monotonic() - t0 < 1.0
    msg = str(ei.value)
    assert "agent-A" in msg and f"pid {first.lease.pid}" in msg and "since" in msg and "holding" in msg
    r = _cli("utils.gpu_lease", lock, "acquire", "--owner", "agent-B")
    assert r.returncode == L.EXIT_LEASED and "agent-A" in r.stderr and not r.stdout, r
    assert L.read_lease(lock).owner == "agent-A", "the loser must not disturb the lease"


def test_acquire_against_a_one_off_holder_fails_fast_busy(lock):
    h = _one_off_holder(lock)
    try:
        t0 = time.monotonic()
        with pytest.raises(L.GpuBusy) as ei:
            _acquire(lock, "agent-A")
        assert time.monotonic() - t0 < 5.0
        assert f"pid {h.pid}" in str(ei.value) or f"pid {h.pid}" in " ".join(map(str, ei.value.holders)) \
            or h.pid in ei.value.holders
        r = _cli("utils.gpu_lease", lock, "acquire", "--owner", "agent-A")
        assert r.returncode == L.EXIT_BUSY and "GPU BUSY" in r.stderr, r
        assert L.read_lease(lock) is None and L.status(lock).startswith("BUSY")
        assert _holders(lock) == [h.pid], "the failed acquire must not leave a holder behind"
    finally:
        _reap(h)


def test_a_renewing_acquire_with_the_token_is_a_noop_that_extends_expiry(lock):
    got = _acquire(lock, "agent-A", max_hours=1)
    again = L.acquire(lock, "agent-A", max_hours=5, watch_pid=os.getpid(), env={L.TOKEN_ENV: got.token})
    assert again.already_held and again.token is None and again.lease.watch_pid == os.getpid()
    assert again.lease.pid == got.lease.pid and again.lease.expires_epoch > got.lease.expires_epoch + 3 * 3600
    assert _holders(lock) == [got.lease.pid], "still ONE holder"


def test_force_release_by_a_non_owner_and_refusal_without_it(lock):
    _acquire(lock, "agent-A")
    with pytest.raises(PermissionError):
        L.release(lock, env={})
    with pytest.raises(PermissionError):
        L.release(lock, env={L.TOKEN_ENV: "not-the-token"})
    assert "released" in L.release(lock, force=True, env={})
    assert _free(lock) and L.read_lease(lock) is None


# ------------------------------------------------------------------------------ gpu_lock under a lease

@pytest.mark.parametrize("wait", [False, True], ids=["default", "wait_true"])
def test_gpu_lock_under_a_foreign_lease_raises_at_once_and_never_waits(lock, wait):
    got = _acquire(lock, "agent-A")
    key = P.file_key(str(lock))
    t0 = time.monotonic()
    with pytest.raises(G.GpuLeased) as ei:
        with G.gpu_lock(lock, wait=wait, poll_s=5.0, report_every_s=600):
            raise AssertionError("entered the body of a leased GPU")
    assert time.monotonic() - t0 < 1.0, "a refusal must be immediate"
    assert P.waiters(key) == [], "no kernel waiter may ever have been queued"
    assert "agent-A" in str(ei.value) and ei.value.lease.pid == got.lease.pid
    assert _holders(lock) == [got.lease.pid]


def test_the_lock_cli_exits_6_on_a_foreign_lease_even_with_wait(lock):
    _acquire(lock, "agent-A")
    for args in (["--"], ["--wait", "--"]):
        r = _cli("utils.gpu_lock", lock, *args, sys.executable, "-c", "print('RAN')")
        assert r.returncode == L.EXIT_LEASED and "RAN" not in r.stdout and "agent-A" in r.stderr, (args, r)
    assert "LEASED" in _cli("utils.gpu_lock", lock, "--status").stdout


def test_the_owners_token_passes_straight_through_without_the_flock(lock, monkeypatch):
    got = _acquire(lock, "agent-A")
    monkeypatch.setenv(L.TOKEN_ENV, got.token)
    key = P.file_key(str(lock))
    t0 = time.monotonic()
    with G.gpu_lock(lock) as a, G.gpu_lock(lock) as b:          # re-entrant too
        assert a == b == got.lease.pid, "the yielded holder is the lease's"
    assert time.monotonic() - t0 < 1.0
    assert P.waiters(key) == [] and _holders(lock) == [got.lease.pid], "the lease keeps the flock throughout"
    assert G.HELD_ENV not in os.environ
    # a CHILD of the owner (token inherited through the env) is passed through as well, and a wrong token is not
    r = _cli("utils.gpu_lock", lock, "--", sys.executable, "-c",
             "from utils.gpu_lock import gpu_lock\nwith gpu_lock() as h: print('IN', h)",
             extra_env={L.TOKEN_ENV: got.token})
    assert r.returncode == 0 and r.stdout.split() == ["IN", str(got.lease.pid)], r
    r = _cli("utils.gpu_lock", lock, "--", "true", extra_env={L.TOKEN_ENV: "forged"})
    assert r.returncode == L.EXIT_LEASED, r
    # the token FILE (an agent shell keeps no exports) is honoured the same way
    tf = lock.parent / "tok"
    tf.write_text(got.token + "\n")
    monkeypatch.delenv(L.TOKEN_ENV)
    monkeypatch.setenv(L.TOKEN_FILE_ENV, str(tf))
    with G.gpu_lock(lock) as c:
        assert c == got.lease.pid


# ------------------------------------------------------------------------------ stale-lease recovery

def test_a_dead_holder_leaves_a_stale_record_that_is_cleared_deterministically(lock):
    got = _acquire(lock, "agent-A")
    os.kill(got.lease.pid, signal.SIGKILL)                       # the holder, by explicit pid
    _wait_for(lambda: _free(lock), "the killed holder's flock to be released")
    assert L.lease_file(lock).exists(), "the record outlives its holder"
    assert L.read_lease(lock) is None, "a record whose holder is dead is NOT a lease"
    with G.gpu_lock(lock, timeout_s=5):                          # a one-off take succeeds despite the record
        pass
    s = L.status(lock)
    assert "stale lease record was cleared" in s and "FREE" in s
    assert not L.lease_file(lock).exists()
    # and acquire clears it too, then leases normally
    got2 = _acquire(lock, "agent-A")
    os.kill(got2.lease.pid, signal.SIGKILL)
    _wait_for(lambda: _free(lock), "the second holder to die")
    got3 = _acquire(lock, "agent-B")
    assert got3.lease.owner == "agent-B" and L.read_lease(lock).pid == got3.lease.pid


def test_a_record_naming_a_live_pid_with_the_wrong_start_time_is_stale(lock):
    """pid REUSE: the pid is alive and HOLDS the lock (a one-off holder), but its start time is not the
    record's — so it is not the lease holder. With the right start time the same record is valid."""
    h = _one_off_holder(lock)
    try:
        st = P.read_stat(h.pid)
        base = L.Lease(owner="ghost", note="", pid=h.pid, start_ticks=st.starttime + 1, token_sha256="0" * 64,
                       started_epoch=time.time(), expires_epoch=0.0, watch_pid=0, acquirer_cmd="forged")
        L._write_record(lock, base)
        assert L.read_lease(lock) is None
        with pytest.raises(G.GpuBusy):                           # judged as a one-off hold, not a lease
            with G.gpu_lock(lock):
                pass
        L._write_record(lock, dataclasses.replace(base, start_ticks=st.starttime))
        assert L.read_lease(lock) is not None
        with pytest.raises(G.GpuLeased):
            with G.gpu_lock(lock):
                pass
    finally:
        _reap(h)


def test_the_holder_ends_when_its_watched_pid_dies(lock):
    owner = subprocess.Popen(["sleep", "120"])
    try:
        got = _acquire(lock, "agent-A", watch_pid=owner.pid)
        assert _holders(lock) == [got.lease.pid]
        owner.kill()
        owner.wait(timeout=30)
        _wait_for(lambda: _free(lock), "the holder to release when its watched pid died")
        assert L.read_lease(lock) is None
        _wait_for(lambda: not L.lease_file(lock).exists(), "the holder to remove its own record")
    finally:
        owner.kill()


def test_the_lease_expires_at_max_hours(lock):
    got = _acquire(lock, "agent-A", max_hours=0.0005)           # 1.8 s
    assert got.lease.expires_epoch > got.lease.started_epoch
    _wait_for(lambda: _free(lock), "the lease to expire")
    assert L.read_lease(lock) is None


# ------------------------------------------------------------------------------ one-off holders

def test_the_default_against_a_one_off_holder_fails_fast_busy(lock):
    h = _one_off_holder(lock)
    try:
        key = P.file_key(str(lock))
        t0 = time.monotonic()
        with pytest.raises(G.GpuBusy) as ei:
            with G.gpu_lock(lock, poll_s=5.0, report_every_s=600):
                raise AssertionError("entered the body of a busy GPU")
        assert time.monotonic() - t0 < 1.0
        assert P.waiters(key) == [], "fail-fast must not queue a kernel waiter"
        assert ei.value.holders == [h.pid] and f"pid {h.pid}" in str(ei.value)
        r = _cli("utils.gpu_lock", lock, "--", "true")
        assert r.returncode == L.EXIT_BUSY and "GPU BUSY" in r.stderr, r
        assert _holders(lock) == [h.pid]
    finally:
        _reap(h)


def test_wait_true_still_waits_for_a_one_off_holder(lock):
    h = _one_off_holder(lock)
    key = P.file_key(str(lock))
    out: dict = {}

    def waiter() -> None:
        try:
            with G.gpu_lock(lock, wait=True, poll_s=0.05, report_every_s=600, log=lambda m: None) as pid:
                out["pid"] = pid
        except BaseException as e:                               # noqa: BLE001 - reported to the main thread
            out["err"] = e

    t = threading.Thread(target=waiter, daemon=True)
    try:
        t.start()
        _wait_for(lambda: os.getpid() in P.waiters(key), "the wait=True caller to block as a /proc/locks waiter")
        assert t.is_alive() and not out, "it must still be WAITING while the holder lives"
    finally:
        _reap(h)
    t.join(timeout=60)
    assert not t.is_alive() and out == {"pid": os.getpid()}, out


def test_a_lease_appearing_mid_wait_ends_the_wait_with_gpu_leased(lock):
    h = _one_off_holder(lock)
    key = P.file_key(str(lock))
    out: dict = {}

    def waiter() -> None:
        try:
            with G.gpu_lock(lock, wait=True, poll_s=0.05, report_every_s=600, log=lambda m: None):
                out["entered"] = True
        except BaseException as e:                               # noqa: BLE001
            out["err"] = e

    t = threading.Thread(target=waiter, daemon=True)
    try:
        t.start()
        _wait_for(lambda: os.getpid() in P.waiters(key), "the waiter to block")
        st = P.read_stat(h.pid)                                  # a VALID lease record (its holder holds the lock)
        L._write_record(lock, L.Lease(owner="agent-A", note="", pid=h.pid, start_ticks=st.starttime,
                                      token_sha256="0" * 64, started_epoch=time.time(), expires_epoch=0.0,
                                      watch_pid=0, acquirer_cmd="forged"))
        t.join(timeout=60)
        assert not t.is_alive() and isinstance(out.get("err"), G.GpuLeased) and "entered" not in out, out
        assert P.holders(key) == [h.pid], "the abandoned wait must not have taken the lock"
    finally:
        _reap(h)


def test_the_lease_holder_survives_its_acquirer(lock):
    """The holder is DETACHED: the acquiring process exiting must not end the lease (the CLI form)."""
    r = _cli("utils.gpu_lease", lock, "acquire", "--owner", "agent-A")
    assert r.returncode == 0, r.stderr
    assert L.read_lease(lock) is not None and L.read_lease(lock).owner == "agent-A"
    assert not _free(lock)
