"""The Rust env's PROCESS FRONT END — the core in a child process over a shared-memory mapping (M5
Lane B, ``designs/endstate/program_rust_core.md`` §2 M5).

WHY a second front end: the FFI (Lane A) catches a Rust PANIC, but an abort, a stack overflow, an OOM
kill or a fault in unsafe code takes the whole process — for a trainer, the optimizer, the rollout
buffer and the GPU context with it. Here the core runs in ``rust_env_proc`` (``src/rust_env/src/bin/``);
a dead child is a TYPED error (:class:`CoreProcessDied`) plus a RESPAWN (a fresh core, re-stamped),
and the host survives. A THIN front end: the child forwards every core opcode to the same
``Core::dispatch`` the FFI calls; neither side names an op's semantics.

THE TRANSPORT:
  * the columns live in ONE anonymous shared mapping — a ``memfd`` created here and handed to the
    child by fd (``pass_fds``), laid out by :func:`layout` (Rust twin: ``shm::layout``) from the
    column table ``columns.py``. The child writes the header (magic, wire id, n, obs_dim, every
    column's offset) and the parent compares it with its own layout before the first op;
  * one request byte per op on the child's stdin, one reply frame per op on its stdout:
    ``[status u8][len u32 LE][payload]`` (a status is ``protocol.STATUSES``; the payload of a failure
    is ``DispatchError::json``, of ``BANK`` the bank JSON). EOF on the reply pipe IS the death signal —
    a SIGKILLed child is seen at once, never waited on.

/dev/shm HYGIENE BY CONSTRUCTION: the mapping is a ``memfd`` — it never has a name in ``/dev/shm``
(or anywhere), so NO exit path can leak one: normal close, an error, a SIGKILL of the child, even a
SIGKILL of the PARENT (the child then reads EOF on stdin and exits; the kernel frees the memory when
the last map and fd go). The gates check the ``/dev/shm`` listing and this process's memfd fds anyway.

THE STAMP, BEFORE ANY OP: the child's FIRST line is a handshake (wire id, obs_dim, column count,
schema id, the build stamp) printed before it has read a byte. The parent checks every field
(``stamp.check_stamp``) and REFUSES a foreign child — closing its stdin, so it exits having run
nothing — before the spec is even sent. Unlike ``dlopen`` (F-LA-3), every spawn re-reads the binary,
so a respawn is re-stamped for real.

THE DECLARED LIFECYCLE through the process: STARTUP = spawn + handshake + INIT (``Core::new`` + the
mapping + ``freeze``); the child binds the mapped columns ONCE, so a rebind cannot be expressed. The
core's ``*_AFTER_FREEZE`` counters come back in the ``counters`` column; this front end adds one of its
own, ``PROC_SPAWNS_AFTER_FREEZE`` (a respawn is a new process in steady state: counted, never silent).

The WIRE (control bytes, header words, reply framing) is ONE table here, rendered into the marked
region of ``src/rust_env/src/shm.rs`` (``python -m utils.rust_env.proc --write``); its id is compiled
into the child and compared at the handshake; ``proc_test.py`` (routine) pins the region.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import select
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.paths import src_path
from utils.rust_env import columns as C
from utils.rust_env import protocol as P
from utils.rust_env import stamp as S

# ------------------------------------------------------------------ THE WIRE TABLE

WIRE_VERSION = 1
#: Bytes before the first column (one page); the header words live at its start.
HEADER_BYTES = 4096
#: Every column's offset is a multiple of this (a cache line; covers every dtype's alignment).
ALIGN = 64
#: Header word 0 (little-endian u64): "RENVPRC1".
MAGIC = int.from_bytes(b"RENVPRC1", "little")
#: The header's u64 words, in order; ``OFFSETS`` is followed by one word per column (table order).
HEADER_WORDS: Tuple[str, ...] = ("MAGIC", "WIRE_ID", "N", "OBS_DIM", "N_COLUMNS", "TOTAL", "OFFSETS")
BIN_NAME = "rust_env_proc"
HANDSHAKE_TAG = "rust_env_proc"


@dataclass(frozen=True)
class Ctl:
    name: str
    code: int
    doc: str


#: The FRONT END's own request bytes. Every other byte is a core opcode, forwarded to
#: ``Core::dispatch`` unread (so an unknown one is the core's ``LIFECYCLE`` refusal).
CONTROL: Tuple[Ctl, ...] = (
    Ctl("INIT", ord("I"), "STARTUP, the child's FIRST request (once): `[u32 LE length][spec JSON]`; "
                          "`Core::new`, map the columns, write the header, `freeze`"),
    Ctl("BANK", ord("B"), "reply payload: the refusal bank, a JSON array of `Banked::json`"),
    Ctl("FINISHED", ord("F"), "reply payload: the episodes that ENDED in the last op, a JSON array of "
                              "`Finished::json` (M5 Lane H)"),
    Ctl("QUIT", ord("Q"), "reply OK, drop the core (joins its workers), unmap, exit 0"),
    Ctl("PANIC_PROBE", ord("P"), "TEST HOOK: a panic inside the child's guarded op section — status PANIC, "
                                 "and the child's core is POISONED (every later op LIFECYCLE)"),
    Ctl("ABORT_PROBE", ord("A"), "TEST HOOK: `std::process::abort()` — what no in-process front end survives"),
)

MARK_BEGIN = "// ---- @generated-begin by `python -m utils.rust_env.proc --write` — DO NOT EDIT this region."
MARK_END = "// ---- @generated-end"
SHM_RS = src_path("rust_env", "src", "shm.rs")


def wire_text() -> str:
    """The canonical text of the wire (+ the column schema the layout follows)."""
    lines = [f"wire {WIRE_VERSION}", f"header_bytes {HEADER_BYTES}", f"align {ALIGN}", f"magic {MAGIC:#018x}",
             "reply status:u8 len:u32le payload", f"schema {C.schema_id()}", f"columns {len(C.COLUMNS)}"]
    lines += [f"hdr {i} {w}" for i, w in enumerate(HEADER_WORDS)]
    lines += [f"ctl {c.name} {c.code}" for c in CONTROL]
    return "\n".join(lines) + "\n"


def wire_id() -> str:
    h = 0xCBF29CE484222325
    for b in wire_text().encode():
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def _check_table() -> None:
    codes = [c.code for c in CONTROL]
    ops = {o.code for o in P.OPS}
    if len(set(codes)) != len(codes) or not all(0 < c < 128 for c in codes):
        raise AssertionError(f"control bytes must be distinct ASCII: {codes}")
    if set(codes) & ops:
        raise AssertionError(f"a control byte collides with a core opcode: {sorted(set(codes) & ops)}")
    if (len(HEADER_WORDS) - 1 + len(C.COLUMNS)) * 8 > HEADER_BYTES:
        raise AssertionError("the header does not fit in HEADER_BYTES")


_check_table()


def layout(n: int, obs_dim: int) -> Tuple[Dict[str, int], int]:
    """``({column: byte offset}, total bytes)`` of the mapping for a pool of ``n`` (``shm::layout``'s twin)."""
    sizes = C.nbytes(n, obs_dim)
    off: Dict[str, int] = {}
    at = HEADER_BYTES
    for c in C.COLUMNS:
        off[c.name] = at
        at = (at + sizes[c.name] + ALIGN - 1) // ALIGN * ALIGN
    return off, at


def render_region() -> str:
    """The generated region of ``shm.rs`` (markers included)."""
    out: List[str] = [
        MARK_BEGIN,
        "// Source of truth: the wire table in `src/utils/rust_env/proc.py`; pinned by `proc_test.py` (routine).",
        "",
        "/// `proc.wire_id()` — FNV-1a-64 of the wire's canonical text; compared at the handshake.",
        f"pub const WIRE_ID: &str = \"{wire_id()}\";",
        f"pub const WIRE_ID_U64: u64 = 0x{wire_id()};",
        "/// Bytes before the first column (the header words live at its start).",
        f"pub const HEADER_BYTES: usize = {HEADER_BYTES};",
        "/// Every column's offset is a multiple of this.",
        f"pub const ALIGN: usize = {ALIGN};",
        "/// Header word 0.",
        f"pub const MAGIC: u64 = {MAGIC:#018x};",
        "/// The handshake line's first field.",
        f"pub const HANDSHAKE_TAG: &str = \"{HANDSHAKE_TAG}\";",
        "",
        "/// The header's u64 word indices (`OFFSETS` is followed by one word per column).",
        "pub mod hdr {",
    ]
    for i, w in enumerate(HEADER_WORDS):
        out.append(f"    pub const {w}: usize = {i};")
    out += ["}", "", "/// The front end's own request bytes (every other byte is a core opcode).", "pub mod ctl {"]
    for c in CONTROL:
        out.append(f"    /// {c.doc}")
        out.append(f"    pub const {c.name}: u8 = b'{chr(c.code)}';")
    out += ["}", "", MARK_END]
    return "\n".join(out)


def render(current: str) -> str:
    a, b = current.find(MARK_BEGIN), current.find(MARK_END)
    if a < 0 or b < a:
        raise AssertionError(f"{SHM_RS}: the generated-region markers are missing")
    return current[:a] + render_region() + current[b + len(MARK_END):]


# ------------------------------------------------------------------ errors


class ProcLoadError(ImportError):
    """The child is not a matching process front end (another wire, column schema or row length)."""


class CoreProcessDied(P.RustEnvError):
    """The child process ended while an op was in flight (a SIGKILL, an abort, an OOM kill). The host
    is intact. ``respawned`` says whether a FRESH core is already up (then RESET before STEP)."""

    status = -1

    def __init__(self, message: str, *, returncode: Optional[int] = None, respawned: bool = False):
        super().__init__(message)
        self.returncode = returncode
        self.respawned = respawned


class CoreProcessTimeout(CoreProcessDied):
    """The child did not answer within the caller's declared bound; it was KILLED (by its PID) and
    handled as a death. A timeout is never a semantic outcome."""


def default_path(profile: str = "selfcheck") -> Path:
    """THIS checkout's build of the child (``cargo build --bin rust_env_proc`` into ``src/rust_env/target``)."""
    return src_path("rust_env", "target", profile, BIN_NAME).resolve()


def parse_handshake(line: str) -> Dict[str, str]:
    parts = line.rstrip("\n").split("\t")
    if not parts or parts[0] != HANDSHAKE_TAG:
        raise ProcLoadError(f"not a rust env process front end (handshake {line[:200]!r})")
    try:
        return dict(p.split("=", 1) for p in parts[1:])
    except ValueError:
        raise ProcLoadError(f"an unreadable handshake {line[:200]!r}") from None


def check_handshake(hs: Dict[str, str], what: str, *, nan_poison: Optional[bool] = None,
                    stamp_src: Optional[Path] = None) -> Dict[str, str]:
    """REFUSE a child that is not this tree's build — before any op. Returns the parsed stamp."""
    if hs.get("wire") != wire_id():
        raise ProcLoadError(f"{what}: wire {hs.get('wire')}, this tree's {wire_id()} — rebuild it from THIS checkout")
    if hs.get("schema") != C.schema_id() or hs.get("n_columns") != str(len(C.COLUMNS)):
        raise ProcLoadError(f"{what}: column schema {hs.get('schema')} / {hs.get('n_columns')} columns, "
                            f"this tree's {C.schema_id()} / {len(C.COLUMNS)}")
    return S.check_stamp(hs.get("stamp", ""), what, src=stamp_src, nan_poison=nan_poison)


# ------------------------------------------------------------------ the pool


class ProcCore:
    """One env core in a child process: STARTUP (spawn + handshake + INIT: ``Core::new`` + the mapped
    columns + FREEZE) in the constructor, then :meth:`reset` / :meth:`step` — the same surface as
    ``ffi.FfiCore``. ``cols`` are NumPy views of the shared mapping named as in ``columns.COLUMNS``;
    inputs are written in place, outputs are read-only. They stay VALID across a respawn (the
    mapping outlives the child).

    A core failure raises the ``protocol`` class and — as through the FFI (F-LA-1) — every status but
    ``LIFECYCLE`` POISONS the core: call :meth:`respawn` for a fresh one. A dead child raises
    :class:`CoreProcessDied`; with ``auto_respawn`` a fresh core is already up when it is raised.

    ``op_timeout`` (seconds, None = wait) bounds one op; ``startup_timeout`` the handshake + INIT.
    """

    def __init__(self, spec_json: str, *, binary: Optional[Path] = None, nan_poison: Optional[bool] = None,
                 auto_respawn: bool = True, startup_timeout: float = 600.0, op_timeout: Optional[float] = None,
                 stamp_src: Optional[Path] = None):
        self.binary = Path(binary) if binary is not None else default_path()
        if not self.binary.is_absolute():
            raise ProcLoadError(f"the rust env child must be spawned by ABSOLUTE path, got {self.binary}")
        if not self.binary.exists():
            from utils.rust_env.build import build_command

            raise ProcLoadError(f"{self.binary} does not exist — build it: {build_command(self.binary.parent.name)}")
        self.spec_json = spec_json
        self.nan_poison = nan_poison
        self.auto_respawn = auto_respawn
        self.startup_timeout = startup_timeout
        self.op_timeout = op_timeout
        self._stamp_src = stamp_src
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._mm: Optional[mmap.mmap] = None
        self.cols: Dict[str, object] = {}
        self.respawns = 0
        self.stamp = ""
        self._closed = False
        try:
            n = json.loads(spec_json).get("n")
        except (ValueError, AttributeError):
            n = None
        # A spec without a usable `n` still goes to the child, whose parser is the authority (a
        # typed CallerError); the mapping is then never sized.
        self.n = int(n) if isinstance(n, int) and n > 0 else 0
        self.obs_dim = 0
        self._fd = os.memfd_create("gen3ai_rust_env", os.MFD_CLOEXEC)
        try:
            self._spawn()
        except BaseException:
            self.close()
            raise
        self._cidx = P.counter_index()
        self._op = {o.name: o.code for o in P.OPS}

    # ---- the child
    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc is not None else None

    def _read_exact(self, k: int, timeout: Optional[float], where: str) -> bytes:
        assert self._proc is not None and self._proc.stdout is not None
        fd = self._proc.stdout.fileno()
        buf = b""
        deadline = None if timeout is None else time.monotonic() + timeout
        while len(buf) < k:
            if deadline is not None:
                left = deadline - time.monotonic()
                if left <= 0 or not select.select([fd], [], [], left)[0]:
                    raise _Timeout(where)
            chunk = os.read(fd, k - len(buf))
            if not chunk:
                raise _Eof(where)
            buf += chunk
        return buf

    def _write(self, data: bytes, where: str) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        try:
            fd = self._proc.stdin.fileno()
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
        except (BrokenPipeError, ConnectionResetError, OSError):
            raise _Eof(where) from None

    def _reply(self, timeout: Optional[float], where: str) -> Tuple[int, bytes]:
        head = self._read_exact(5, timeout, where)
        k = int.from_bytes(head[1:], "little")
        return head[0], (self._read_exact(k, timeout, where) if k else b"")

    def _spawn(self) -> None:
        """Spawn a child on the mapping, check its handshake BEFORE sending anything, INIT it."""
        self._proc = subprocess.Popen([str(self.binary), "--fd", str(self._fd)], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, pass_fds=(self._fd,), bufsize=0, close_fds=True)
        try:
            line = b""
            while not line.endswith(b"\n"):
                line += self._read_exact(1, self.startup_timeout, "the handshake")
                if len(line) > 1 << 16:
                    raise ProcLoadError(f"{self.binary}: no handshake line in 64 KiB of output")
            hs = parse_handshake(line.decode(errors="replace"))
            st = check_handshake(hs, str(self.binary), nan_poison=self.nan_poison, stamp_src=self._stamp_src)
            obs_dim = int(hs["obs_dim"])
            if self._mm is None:
                self.obs_dim = obs_dim
                if self.n:
                    self._off, self._total = layout(self.n, obs_dim)
                    os.ftruncate(self._fd, self._total)
            elif obs_dim != self.obs_dim:
                raise ProcLoadError(f"{self.binary}: obs_dim {obs_dim}, the mapping was laid out for {self.obs_dim}")
            spec = self.spec_json.encode()
            self._write(bytes([_CTL["INIT"]]) + len(spec).to_bytes(4, "little") + spec, "INIT")
            status, payload = self._reply(self.startup_timeout, "INIT")
            if status != 0:
                raise _error(status, payload)
            if self._mm is None:
                self._map()
            self._check_header()
            self.stamp = hs["stamp"]
            self._stamp = st
        except _Eof:
            rc = self._reap()
            raise CoreProcessDied(f"{self.binary} DIED at startup (returncode {_rc(rc)})", returncode=rc) from None
        except _Timeout as t:
            self._kill()
            raise CoreProcessTimeout(f"{self.binary}: no answer to {t} within {self.startup_timeout} s; killed") from None
        except BaseException:
            self._stop()  # never leave a refused or failed child behind (it exits on stdin EOF)
            raise

    def _map(self) -> None:
        import numpy as np

        self._mm = mmap.mmap(self._fd, self._total)
        shapes = C.shapes(self.n, self.obs_dim)
        cols = {}
        for c in C.COLUMNS:
            cnt = 1
            for d in shapes[c.name]:
                cnt *= d
            a = np.frombuffer(self._mm, dtype=C.numpy_dtype(c), count=cnt, offset=self._off[c.name])
            a = a.reshape(shapes[c.name])
            if c.direction == "out":
                a.flags.writeable = False  # the child writes through its own mapping
            cols[c.name] = a
        self._hdr = np.frombuffer(self._mm, dtype="<u8", count=len(HEADER_WORDS) - 1 + len(C.COLUMNS), offset=0)
        cols["action"].fill(-1)
        self.cols = cols

    def _check_header(self) -> None:
        h = [int(x) for x in self._hdr]
        want = [MAGIC, int(wire_id(), 16), self.n, self.obs_dim, len(C.COLUMNS), self._total]
        want += [self._off[c.name] for c in C.COLUMNS]
        if h != want:
            raise ProcLoadError(f"{self.binary}: the child's mapping header {h} is not this layout {want}")

    def _stop(self) -> None:
        """End the current child: close its stdin (it exits on EOF), wait, KILL by its PID if it
        does not go. Always reaped — no zombie."""
        p = self._proc
        if p is None:
            return
        try:
            if p.stdin is not None:
                p.stdin.close()
        except OSError:
            pass
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._kill()
        self._reap()

    def _kill(self) -> None:
        p = self._proc
        if p is not None and p.poll() is None:
            os.kill(p.pid, signal.SIGKILL)  # our own child, by explicit PID
        self._reap()

    def _reap(self) -> Optional[int]:
        p = self._proc
        if p is None:
            return None
        rc = p.wait()
        for f in (p.stdin, p.stdout):
            try:
                if f is not None:
                    f.close()
            except OSError:
                pass
        self._proc = None
        self._last_rc = rc
        return rc

    def _zero_outputs(self) -> None:
        for c in C.COLUMNS:
            if c.direction == "out":
                a = self.cols[c.name]
                a.flags.writeable = True
                a.fill(0)
                a.flags.writeable = False

    def respawn(self) -> None:
        """A FRESH core in a fresh child (handshake + stamp + INIT again) on the SAME mapping; the
        outputs are zeroed and the caller must RESET. COUNTED in ``PROC_SPAWNS_AFTER_FREEZE``."""
        with self._lock:
            self._respawn_locked()

    def _respawn_locked(self) -> None:
        if self._closed:
            raise P.LifecycleViolation("respawn on a closed ProcCore")
        self._stop()
        self._zero_outputs()
        self.respawns += 1
        self._spawn()

    def _died(self, rc: Optional[int], where: str, timeout: bool) -> CoreProcessDied:
        cls = CoreProcessTimeout if timeout else CoreProcessDied
        what = (f"no answer within {self.op_timeout} s during {where}; the child was KILLED" if timeout
                else f"the rust env child DIED during {where} (returncode {_rc(rc)})")
        if not self.auto_respawn:
            return cls(what + "; this ProcCore has no child — respawn() for a fresh core", returncode=rc)
        try:
            self._respawn_locked()
        except BaseException as e:
            err = cls(what + f"; the RESPAWN FAILED: {type(e).__name__}: {e}", returncode=rc)
            err.__cause__ = e
            return err
        return cls(what + f"; a FRESH core was spawned and re-stamped (pid {self.pid}) — RESET before the next STEP",
                   returncode=rc, respawned=True)

    def _call(self, code: int, where: str) -> bytes:
        """Send one request byte; return the payload of an OK reply or raise the typed error."""
        if self._closed:
            raise P.LifecycleViolation("an op on a closed ProcCore")
        if self._proc is None:
            raise CoreProcessDied("this ProcCore has no child (it died without auto_respawn) — respawn() first")
        try:
            self._write(bytes([code]), where)
            status, payload = self._reply(self.op_timeout, where)
        except _Eof:
            died = self._died(self._reap(), where, timeout=False)
            raise died from died.__cause__  # a failed respawn's error, else nothing
        except _Timeout:
            pid = self.pid
            self._kill()
            died = self._died(self._last_rc, where + f" (pid {pid})", timeout=True)
            raise died from died.__cause__
        if status != 0:
            raise _error(status, payload)
        return payload

    # ---- ops
    def dispatch(self, op: str) -> None:
        with self._lock:
            self._call(self._op[op], op)

    def reset(self) -> None:
        self.dispatch("RESET")

    def step(self) -> None:
        self.dispatch("STEP")

    def _control(self, name: str) -> bytes:
        with self._lock:
            return self._call(_CTL[name], name)

    # ---- reads
    def counters(self) -> Dict[str, int]:
        """The pool counters as the core last wrote them into the ``counters`` column (every dispatch
        writes it, whatever its status; all 0 before the first op and after a respawn)."""
        with self._lock:
            k = [int(x) for x in self.cols["counters"]]
        return {c.name: k[i] for i, c in enumerate(P.COUNTERS)}

    def after_freeze(self) -> Dict[str, int]:
        """The DECLARED-LIFECYCLE counters — the core's plus this front end's ``PROC_SPAWNS_AFTER_FREEZE``
        (children spawned after the first freeze); each must stay 0 for the pool's life."""
        k = self.counters()
        out = {c.name: k[c.name] for c in P.COUNTERS if c.after_freeze}
        out["PROC_SPAWNS_AFTER_FREEZE"] = self.respawns
        return out

    def bank(self) -> List[dict]:
        return json.loads(self._control("BANK").decode())

    def finished(self) -> List[dict]:
        """The episodes that ENDED in the last op (M5 Lane H) — ``FfiCore.finished``'s shape."""
        return json.loads(self._control("FINISHED").decode())

    # ---- teardown
    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            p = self._proc
            if p is not None and p.poll() is None:
                try:
                    self._write(bytes([_CTL["QUIT"]]), "QUIT")
                    self._reply(10.0, "QUIT")
                except (_Eof, _Timeout):
                    pass
            self._stop()
            self.cols = {}
            self._hdr = None
            if self._mm is not None:
                try:
                    self._mm.close()
                except BufferError:
                    pass  # a caller still holds a view; the memory goes with it (it has no name to leak)
                self._mm = None
            if getattr(self, "_fd", -1) >= 0:
                os.close(self._fd)
                self._fd = -1

    def __enter__(self) -> "ProcCore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class _Eof(Exception):
    pass


class _Timeout(Exception):
    pass


_CTL = {c.name: c.code for c in CONTROL}


def _rc(rc: Optional[int]) -> str:
    if rc is not None and rc < 0:
        try:
            return f"{rc}, {signal.Signals(-rc).name}"
        except ValueError:
            pass
    return str(rc)


def _error(status: int, payload: bytes) -> P.RustEnvError:
    if not payload:
        return P.error_for(status, "the child returned a failure with no error recorded")
    return P.error_from_json(payload.decode())


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="rewrite the generated region of src/rust_env/src/shm.rs")
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed region is stale")
    a = ap.parse_args(argv)
    cur = SHM_RS.read_text()
    new = render(cur)
    if a.write:
        SHM_RS.write_text(new)
        print(f"wrote {SHM_RS} (wire {wire_id()})")
        return 0
    if a.check:
        ok = new == cur
        print("shm.rs is current" if ok else "shm.rs is STALE — run with --write")
        return 0 if ok else 1
    sys.stdout.write(render_region() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
