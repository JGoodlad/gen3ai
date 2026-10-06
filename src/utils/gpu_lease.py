"""The GPU LEASE — one agent or session owns the GPU for its whole lifetime; nobody ever waits.

USAGE
    scripts/ops/gpu_lease.sh acquire --owner "X5 U1" [--note "..."] [--max-hours 12] [--watch-pid N] [--token-file F]
        # --watch-pid N must be ALIVE after a ~1 s settle (the long-lived launcher, NOT a `nohup setsid ... &` wrapper): else exit 2
        # stdout:  export GEN3AI_GPU_LEASE_TOKEN=<token>      (stderr carries the human line)
        # held by someone else or by a one-off job: exits 6 / 5 AT ONCE naming the holder
    scripts/ops/gpu_lease.sh status               # who holds the GPU now (lease or one-off), stale leases cleared
    scripts/ops/gpu_lease.sh release              # idempotent; needs the token (env) — or --force (orchestrator)
    # then every GPU command, in a shell that has the token (an agent's Bash tool does NOT keep exports
    # across calls: put GEN3AI_GPU_LEASE_TOKEN=... in front of the call, or --token-file F once and
    # GEN3AI_GPU_LEASE_TOKEN_FILE=F in front):
    scripts/ops/gpu_lock.sh <cmd>                 # token matches the lease -> passes straight through

WHY (owner, 2026-10-03). "I would prefer subagents not block at all on the gpu ... Subagents waiting is
painful for throughput and cache misses." So the GPU is not a queue: it is LEASED to ONE owner for the
owner's lifetime, and everyone else gets an immediate, typed refusal (:class:`GpuLeased` /
:class:`GpuBusy`) instead of a wait. The orchestrator grants the lease to one agent at a time and names
that agent in the agent's brief.

MECHANISM. ``acquire`` starts a small DETACHED holder process that takes the SAME flock as
``utils.gpu_lock`` (``$GEN3AI_GPU_LOCK``, default ``~/.claude/jobs/gpu.lock``) NON-BLOCKING and keeps it
for the lease's lifetime, then writes ``<lock>.lease.json`` (owner, holder pid + start time, SHA-256 of
the token — never the token — start, expiry). The holder ends when the record is removed or no longer
names it (``release``), on SIGTERM, at ``--max-hours`` (default 12; 0 = none), or when ``--watch-pid``
dies. A lease is VALID only if its holder pid (with its recorded start time) is listed HOLDING the lock
in ``/proc/locks`` — a record whose holder is dead, or whose pid was reused, is STALE: ``status`` and
``acquire`` clear it (under a microsecond-scale mutex, never racing a new holder), deterministically.
Re-running ``acquire`` with the matching token is a no-op that renews the expiry (and sets ``--watch-pid``
when given — a training agent attaches the launcher's pid after the launch).

The record is the lease; the flock is what makes it exclusive. A caller whose token hash matches a valid
lease is the owner (``utils.gpu_lock`` passes it straight through, re-entrant, no flock). Linux only.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import secrets
import select
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List, Mapping, Optional, Sequence

from utils import procfs

TOKEN_ENV = "GEN3AI_GPU_LEASE_TOKEN"
TOKEN_FILE_ENV = "GEN3AI_GPU_LEASE_TOKEN_FILE"
DEFAULT_MAX_HOURS = 12.0
HOLDER_POLL_S = 0.5
START_TIMEOUT_S = 30.0        # the holder's PROCESS STARTUP (stdlib import) — never a wait on the GPU
RELEASE_GRACE_S = 10.0
WATCH_SETTLE_S = 1.0          # `--watch-pid` must still be alive this long after acquire is asked (see WatchPidGone)

EXIT_BUSY, EXIT_LEASED, EXIT_NOT_OWNER = 5, 6, 7


# ------------------------------------------------------------------------------------------ errors

class GpuUnavailable(RuntimeError):
    """The GPU is not yours and this call will not wait for it."""


class GpuLeased(GpuUnavailable):
    """The GPU is LEASED to someone else. ``lease`` names the owner; nothing waited."""

    def __init__(self, lease: "Lease", path: object, what: str = "") -> None:
        self.lease, self.path = lease, str(path)
        super().__init__(
            f"GPU LEASED (not waiting): {lease.describe()}{what}. The GPU belongs to that owner for its "
            "lifetime; ask the orchestrator, do not poll. (`scripts/ops/gpu_lease.sh status`; only the "
            "orchestrator uses `release --force`.)")


class WatchPidGone(ValueError):
    """``--watch-pid`` names a process that is already gone (or exits within the settle window).

    The holder ends the moment its watched pid dies, so such a lease evaporates on its first poll (2026-10-06:
    a pid of a ``nohup setsid ... &`` WRAPPER, which forks the real job and exits at once, left the lease
    unheld for ~24 s while the agent believed it held the GPU). Refused before anything is taken."""


class GpuBusy(GpuUnavailable):
    """A ONE-OFF hold (no lease) is in progress and the caller did not ask to wait."""

    def __init__(self, path: object, holders: Sequence[int], what: str = "") -> None:
        self.path, self.holders = str(path), list(holders)
        who = ", ".join(f"pid {h} ({procfs.cmdline(h)[:160] or 'gone'!r})" for h in holders) or "an unlisted holder"
        super().__init__(
            f"GPU BUSY (not waiting): {path}{what} is held by {who}. Fail-fast is the default; only the "
            "orchestrator or a training launch passes `--wait` / `wait=True` deliberately.")


# ----------------------------------------------------------------------------------------- record

@dataclass(frozen=True)
class Lease:
    owner: str
    note: str
    pid: int                  # the HOLDER's pid (it holds the flock)
    start_ticks: int          # the holder's /proc starttime — pid-reuse proof
    token_sha256: str
    started_epoch: float
    expires_epoch: float      # 0 = none
    watch_pid: int            # 0 = none
    acquirer_cmd: str

    def describe(self) -> str:
        since = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self.started_epoch))
        exp = (" expires " + time.strftime("%H:%M:%S%z", time.localtime(self.expires_epoch))) if self.expires_epoch else ""
        note = f" note {self.note!r}" if self.note else ""
        return (f"lease owner {self.owner!r}, holder pid {self.pid}, since {since}{exp}{note}; "
                f"taken by {self.acquirer_cmd[:120]!r}")


def lease_file(lock: object) -> Path:
    return Path(str(lock) + ".lease.json")


def _mu_file(lock: object) -> Path:
    return Path(str(lock) + ".lease.mu")


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def caller_token(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The token the caller holds: ``$GEN3AI_GPU_LEASE_TOKEN``, else the contents of the file named by
    ``$GEN3AI_GPU_LEASE_TOKEN_FILE`` (an agent shell does not keep exports between calls)."""
    e = os.environ if env is None else env
    t = (e.get(TOKEN_ENV) or "").strip()
    if t:
        return t
    f = (e.get(TOKEN_FILE_ENV) or "").strip()
    if f:
        try:
            return Path(f).read_text().strip() or None
        except OSError:
            return None
    return None


@contextlib.contextmanager
def _mu(lock: object) -> Iterator[None]:
    """A mutex over the RECORD only (held for microseconds; blocking on it is not waiting on the GPU)."""
    mp = _mu_file(lock)
    mp.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(mp), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _read_record(lock: object) -> Optional[Lease]:
    try:
        d = json.loads(lease_file(lock).read_text())
        return Lease(**d)
    except (OSError, ValueError, TypeError):
        return None


def _write_record(lock: object, rec: Lease) -> None:
    p = lease_file(lock)
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(asdict(rec), f, sort_keys=True)
    os.replace(tmp, p)


def _valid(lock: object, rec: Lease) -> bool:
    """The record's holder is alive (same pid AND start time) and HOLDS the lock file."""
    key = procfs.file_key(str(lock))
    st = procfs.read_stat(rec.pid)
    return bool(key and st is not None and st.starttime == rec.start_ticks and rec.pid in procfs.holders(key))


def read_lease(lock: object) -> Optional[Lease]:
    """The VALID lease on ``lock``, or ``None`` (none, or a stale record — which this never deletes)."""
    rec = _read_record(lock)
    return rec if rec is not None and _valid(lock, rec) else None


def clear_stale(lock: object) -> bool:
    """Delete a record whose holder is dead / does not hold the lock. Under the record mutex, and the
    holder writes its record under the same mutex AFTER it holds the flock, so a fresh lease is never
    deleted. True iff a stale record was removed."""
    with _mu(lock):
        rec = _read_record(lock)
        if rec is None:
            if lease_file(lock).exists():          # an unparseable record can never be a lease
                lease_file(lock).unlink(missing_ok=True)
                return True
            return False
        if _valid(lock, rec):
            return False
        lease_file(lock).unlink(missing_ok=True)
        return True


def token_matches(lease: Lease, env: Optional[Mapping[str, str]] = None) -> bool:
    t = caller_token(env)
    return t is not None and secrets.compare_digest(token_hash(t), lease.token_sha256)


def holds_lease(lock: object, env: Optional[Mapping[str, str]] = None) -> Optional[Lease]:
    """The valid lease on ``lock`` iff the caller's token is its token."""
    lease = read_lease(lock)
    return lease if lease is not None and token_matches(lease, env) else None


# ----------------------------------------------------------------------------------------- holder

_HOLDER_CODE = ("import sys; sys.path.insert(0, sys.argv[1]); from utils.gpu_lease import _holder_main; "
                "sys.exit(_holder_main(sys.argv[2]))")


def _holder_main(blob: str) -> int:
    """The detached lease holder: take the flock NON-BLOCKING, write the record, hold until told to stop."""
    a = json.loads(blob)
    lock = Path(a["lock"])
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("BUSY", flush=True)
        return 1
    stop = threading.Event()
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, lambda *_: stop.set())
    me = os.getpid()
    st = procfs.read_stat(me)
    now = time.time()
    rec = Lease(owner=a["owner"], note=a["note"], pid=me, start_ticks=st.starttime if st else 0,
                token_sha256=a["token_sha256"], started_epoch=now,
                expires_epoch=now + a["max_hours"] * 3600 if a["max_hours"] else 0.0,
                watch_pid=a["watch_pid"], acquirer_cmd=a["acquirer_cmd"])
    with _mu(lock):
        _write_record(lock, rec)
    print("READY", flush=True)
    for h in (0, 1, 2):                              # detach: the acquirer's pipes must be free to close
        with contextlib.suppress(OSError):
            os.close(h)
    os.open(os.devnull, os.O_RDWR)
    os.dup2(0, 1)
    os.dup2(0, 2)
    try:
        while not stop.wait(HOLDER_POLL_S):
            cur = _read_record(lock)
            if cur is None or cur.pid != me or cur.token_sha256 != rec.token_sha256:
                break                                # released (record removed) or taken over
            if cur.expires_epoch and time.time() >= cur.expires_epoch:
                break
            if cur.watch_pid and procfs.read_stat(cur.watch_pid) is None:
                break
    finally:
        with _mu(lock):
            cur = _read_record(lock)
            if cur is not None and cur.pid == me:
                lease_file(lock).unlink(missing_ok=True)
    return 0                                         # the process exit releases the flock


# ------------------------------------------------------------------------------------- operations

@dataclass(frozen=True)
class Acquired:
    lease: Lease
    token: Optional[str]      # None when the caller already held the lease (its token is the one it has)
    already_held: bool


def _pid_alive(pid: int) -> bool:
    st = procfs.read_stat(pid)
    return st is not None and st.state not in ("Z", "X")       # an unreaped exited child is not alive


def _check_watch_pid(watch_pid: int) -> None:
    """Refuse a ``--watch-pid`` that is not alive now, or that dies within :data:`WATCH_SETTLE_S`."""
    if not watch_pid:
        return
    deadline = time.monotonic() + WATCH_SETTLE_S
    while True:
        if not _pid_alive(watch_pid):
            raise WatchPidGone(
                f"--watch-pid {watch_pid} is not alive (it was gone within {WATCH_SETTLE_S:g} s of acquire). "
                f"A lease watching it would end at once. Likely mistake: the pid of a launch wrapper "
                f"(`nohup setsid ... &`, `$!` of a shell that forks the job and exits) — pass the pid of the "
                f"long-lived process itself (the launcher / trainer), or omit --watch-pid.")
        if time.monotonic() >= deadline:
            return
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def acquire(lock: object, owner: str, note: str = "", max_hours: float = DEFAULT_MAX_HOURS,
            watch_pid: int = 0, token_file: Optional[object] = None,
            env: Optional[Mapping[str, str]] = None) -> Acquired:
    """Take the lease NOW or raise :class:`GpuLeased` / :class:`GpuBusy` AT ONCE — never wait.
    Idempotent for the owner: the caller's token matching the valid lease renews its expiry."""
    if not owner.strip():
        raise ValueError("a lease needs an --owner name")
    _check_watch_pid(int(watch_pid))
    clear_stale(lock)
    mine = holds_lease(lock, env)
    if mine is not None:
        with _mu(lock):
            cur = _read_record(lock)
            if cur is not None and cur.pid == mine.pid:
                mine = Lease(**{**asdict(cur), "expires_epoch": time.time() + max_hours * 3600 if max_hours else 0.0,
                                "watch_pid": int(watch_pid) or cur.watch_pid})
                _write_record(lock, mine)       # the holder re-reads the record each poll
        return Acquired(mine, None, True)
    other = read_lease(lock)
    if other is not None:
        raise GpuLeased(other, lock)
    token = secrets.token_urlsafe(24)
    blob = json.dumps({"lock": str(lock), "owner": owner, "note": note, "token_sha256": token_hash(token),
                       "max_hours": float(max_hours), "watch_pid": int(watch_pid),
                       "acquirer_cmd": (procfs.cmdline(os.getppid()) or " ".join(sys.argv))[:300]})
    src = str(Path(__file__).resolve().parent.parent)
    proc = subprocess.Popen([sys.executable, "-c", _HOLDER_CODE, src, blob], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
    threading.Thread(target=proc.wait, daemon=True, name="gpu_lease-reaper").start()
    assert proc.stdout is not None
    ready, _, _ = select.select([proc.stdout], [], [], START_TIMEOUT_S)
    line = proc.stdout.readline().decode().strip() if ready else ""
    proc.stdout.close()
    if line == "READY":
        lease = read_lease(lock)
        if lease is None:
            raise RuntimeError("lease holder reported READY but no valid lease record exists")
        if token_file is not None:
            tf = Path(token_file)
            tf.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(tf), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(token + "\n")
        return Acquired(lease, token, False)
    if line == "BUSY":
        won = read_lease(lock)                       # a concurrent acquirer may have won the race
        if won is not None:
            raise GpuLeased(won, lock)
        key = procfs.file_key(str(lock))
        raise GpuBusy(lock, procfs.holders(key) if key else [])
    with contextlib.suppress(ProcessLookupError):
        os.kill(proc.pid, signal.SIGKILL)            # our own child, by explicit pid
    raise RuntimeError(f"lease holder did not report READY/BUSY within {START_TIMEOUT_S:.0f} s (got {line!r})")


def release(lock: object, force: bool = False, env: Optional[Mapping[str, str]] = None) -> str:
    """End the lease (idempotent). Needs the token unless ``force``. Returns a one-line outcome; raises
    ``PermissionError`` when a valid lease is someone else's and ``force`` is not set. Returns only
    after the holder is gone and the GPU is free."""
    cleared = clear_stale(lock)
    lease = read_lease(lock)
    if lease is None:
        return "no lease (a stale record was cleared)" if cleared else "no lease"
    if not force and not token_matches(lease, env):
        raise PermissionError(f"not your lease: {lease.describe()} — release needs its token "
                              f"(${TOKEN_ENV}) or the orchestrator's --force")
    key = procfs.file_key(str(lock))

    def gone() -> bool:
        return key is None or lease.pid not in procfs.holders(key)

    with contextlib.suppress(ProcessLookupError):
        os.kill(lease.pid, signal.SIGTERM)           # the verified holder's explicit pid
    deadline = time.monotonic() + RELEASE_GRACE_S
    while not gone() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not gone():
        with contextlib.suppress(ProcessLookupError):
            os.kill(lease.pid, signal.SIGKILL)
        deadline = time.monotonic() + RELEASE_GRACE_S
        while not gone() and time.monotonic() < deadline:
            time.sleep(0.05)
    clear_stale(lock)
    return f"released: {lease.describe()}"


def status(lock: object) -> str:
    """One human block: the lease, else the one-off holder, else free. Clears a stale record."""
    cleared = clear_stale(lock)
    lease = read_lease(lock)
    head = "[a stale lease record was cleared]\n" if cleared else ""
    if lease is not None:
        return head + f"LEASED: {lease.describe()}"
    key = procfs.file_key(str(lock))
    hs: List[int] = procfs.holders(key) if key else []
    if hs:
        return head + f"BUSY (one-off hold, no lease): {lock} held by " + ", ".join(
            f"pid {h} ({procfs.cmdline(h)[:160] or 'gone'!r})" for h in hs)
    return head + f"FREE: {lock} (no lease, no holder)"


# ------------------------------------------------------------------------------------------- CLI

def main(argv: Optional[Sequence[str]] = None) -> int:
    from utils.gpu_lock import lock_path

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if args else 2
    cmd, args = args[0], args[1:]
    lock = lock_path()
    try:
        if cmd == "status":
            print(status(lock))
            return 0
        if cmd == "release":
            force = "--force" in args
            extra = [a for a in args if a != "--force"]
            if extra:
                print(f"[gpu_lease] unknown option(s) {extra!r}", file=sys.stderr)
                return 2
            print(release(lock, force=force), file=sys.stderr)
            return 0
        if cmd == "acquire":
            kw: dict = {"owner": "", "note": "", "max_hours": DEFAULT_MAX_HOURS, "watch_pid": 0, "token_file": None}
            while args:
                flag = args.pop(0)
                if flag in ("--owner", "--note", "--max-hours", "--watch-pid", "--token-file") and args:
                    k, v = flag[2:].replace("-", "_"), args.pop(0)
                    kw[k] = float(v) if k == "max_hours" else int(v) if k == "watch_pid" else v
                else:
                    print(f"[gpu_lease] unknown option {flag!r}", file=sys.stderr)
                    return 2
            if not str(kw["owner"]).strip():
                print("[gpu_lease] REFUSING: acquire needs --owner <name>", file=sys.stderr)
                return 2
            got = acquire(lock, **kw)
            if got.already_held:
                print(f"[gpu_lease] already yours (expiry renewed): {got.lease.describe()}", file=sys.stderr)
            else:
                print(f"[gpu_lease] ACQUIRED: {got.lease.describe()}", file=sys.stderr)
                print(f"export {TOKEN_ENV}={got.token}")
            return 0
    except GpuLeased as e:
        print(str(e), file=sys.stderr)
        return EXIT_LEASED
    except GpuBusy as e:
        print(str(e), file=sys.stderr)
        return EXIT_BUSY
    except PermissionError as e:
        print(f"[gpu_lease] REFUSING: {e}", file=sys.stderr)
        return EXIT_NOT_OWNER
    except WatchPidGone as e:
        print(f"[gpu_lease] REFUSING: {e}", file=sys.stderr)
        return 2
    print(f"[gpu_lease] unknown command {cmd!r} (acquire | release | status)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
