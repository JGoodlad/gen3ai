"""m5_loader — the two THIN Python front ends over the one Rust env core (`proto/`).

`FfiPool`  : ctypes.CDLL over `libm5proto.so` (ctypes releases the GIL for every foreign call).
`ProcPool` : a `m5_envproc` child over a /dev/shm file; one opcode byte per batch each way.

Both expose the SAME attributes and methods — `obs`, `mask`, `need`, `reward`, `done`,
`actions` (NumPy views of the column contract in `proto/src/core.rs`), `reset()`, `step()`,
`successors(k)`, `core_ns`, `close()` — so a caller cannot tell them apart except by cost.

THE BUILD-STAMP GUARD (the 09-09 rust-target incident class): `load_ffi()` recomputes the
source hash of the tree it is importing from (the port's `src/` + `proto/src/`, git blob ids,
FNV-1a-64 over the listing — `proto/build.rs`'s twin) and REFUSES a `.so` whose stamp differs.
Measurement scaffolding, not production.
"""
from __future__ import annotations

import ctypes
import hashlib
import mmap
import os
import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROTO = HERE / "proto"
PORT = (HERE / "../../../../src/rust_sim").resolve()
TARGET = Path(os.environ.get("M5_TARGET", PROTO / "target" / "release"))
ACT = 11


class StampMismatch(ImportError):
    pass


def source_hash() -> tuple[str, int]:
    files = []
    for root, tag in ((PORT / "src", "port"), (PROTO / "src", "proto")):
        for p in root.rglob("*.rs"):
            files.append((f"{tag}/{p.relative_to(root).as_posix()}", p))
    files.sort()
    listing = []
    for rel, p in files:
        data = p.read_bytes()
        blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        listing.append(f"{rel}\t{blob}\n")
    h = 0xCBF29CE484222325
    for b in "".join(listing).encode():
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}", len(files)


def parse_stamp(s: str) -> dict:
    return dict(kv.split("=", 1) for kv in s.strip().strip("\0").split(";"))


def check_stamp(stamp: str, what: str) -> dict:
    st = parse_stamp(stamp)
    want, n = source_hash()
    if st.get("src") != want or int(st.get("nfiles", -1)) != n:
        raise StampMismatch(
            f"{what}: build stamp src={st.get('src')} nfiles={st.get('nfiles')} but this tree hashes to "
            f"src={want} nfiles={n} — a stale or foreign build; rebuild it from THIS tree"
        )
    return st


_LIB = None


def load_ffi(path: Path | None = None, check: bool = True):
    global _LIB
    if _LIB is not None and path is None:
        return _LIB
    lib = ctypes.CDLL(str(path or TARGET / "libm5proto.so"))
    lib.m5_stamp.restype = ctypes.c_char_p
    lib.m5_error.restype = ctypes.c_char_p
    lib.m5_obs_dim.restype = ctypes.c_size_t
    lib.m5_new.restype = ctypes.c_void_p
    lib.m5_new.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint64, ctypes.c_int]
    P = ctypes.c_void_p
    lib.m5_reset.argtypes = [P, P, P, P, P, P]
    lib.m5_step.argtypes = [P, P, P, P, P, P, P]
    lib.m5_successors.argtypes = [P, ctypes.c_size_t, P, P]
    lib.m5_core_ns.restype = ctypes.c_uint64
    lib.m5_core_ns.argtypes = [P]
    lib.m5_free.argtypes = [P]
    lib.m5_refusals.restype = ctypes.c_size_t
    lib.m5_refusals.argtypes = [P]
    if check:
        check_stamp(lib.m5_stamp().decode(), "libm5proto.so")
    if path is None:
        _LIB = lib
    return lib


def _cols(n: int, obs_dim: int, bufs=None):
    if bufs is None:
        return dict(
            obs=np.zeros((n, 2, obs_dim), np.float32),
            mask=np.zeros((n, 2, ACT), np.uint8),
            need=np.zeros((n, 2), np.uint8),
            reward=np.zeros((n,), np.float32),
            done=np.zeros((n,), np.uint8),
            actions=np.full((n, 2), -1, np.int32),
            rows=np.zeros((1024, obs_dim), np.float32),
            ok=np.zeros((1024,), np.uint8),
        )
    return bufs


class FfiPool:
    kind = "ffi"

    def __init__(self, n: int, threads: int, seed: int, opp_external: bool):
        self.lib = load_ffi()
        self.n = n
        self.obs_dim = self.lib.m5_obs_dim()
        self.p = self.lib.m5_new(n, threads, seed, int(opp_external))
        if not self.p:
            raise RuntimeError(self.lib.m5_error().decode())
        for k, v in _cols(n, self.obs_dim).items():
            setattr(self, k, v)
        # The addresses, taken ONCE (`ndarray.ctypes.data` builds an object per access).
        self._a = [self.actions.ctypes.data, self.obs.ctypes.data, self.mask.ctypes.data,
                   self.need.ctypes.data, self.reward.ctypes.data, self.done.ctypes.data]

    def _ok(self, rc: int):
        if rc != 0:
            raise RuntimeError(f"core rc={rc}: {self.lib.m5_error().decode()}")

    def reset(self):
        self._ok(self.lib.m5_reset(self.p, *self._a[1:]))

    def step(self):
        self._ok(self.lib.m5_step(self.p, *self._a))

    def successors(self, k: int):
        self._ok(self.lib.m5_successors(self.p, k, self.rows.ctypes.data, self.ok.ctypes.data))

    @property
    def core_ns(self) -> int:
        return self.lib.m5_core_ns(self.p)

    @property
    def refusals(self) -> int:
        return self.lib.m5_refusals(self.p)

    def close(self):
        if self.p:
            self.lib.m5_free(self.p)
            self.p = None


class ShmLayout:
    HEADER, ERR_LEN, K_MAX = 64, 4096, 1024

    def __init__(self, n: int, obs_dim: int):
        sizes = [self.ERR_LEN, n * 2 * obs_dim * 4, n * 2 * ACT, n * 2, n * 4, n, n * 2 * 4,
                 self.K_MAX * obs_dim * 4, self.K_MAX]
        self.off, at = [], self.HEADER
        for s in sizes:
            self.off.append(at)
            at = (at + s + 63) & ~63
        self.total = at


class ProcPool:
    kind = "proc"

    def __init__(self, n: int, threads: int, seed: int, opp_external: bool, obs_dim: int | None = None,
                 binary: Path | None = None, check: bool = True):
        self.n = n
        self.obs_dim = obs_dim or load_ffi().m5_obs_dim()
        lay = ShmLayout(n, self.obs_dim)
        self.path = f"/dev/shm/m5proto_{os.getpid()}_{id(self)}"
        with open(self.path, "wb") as f:
            f.truncate(lay.total)
        self.fd = os.open(self.path, os.O_RDWR)
        self.mm = mmap.mmap(self.fd, lay.total)
        buf = self.mm
        o = lay.off
        d = self.obs_dim

        def v(i, dtype, shape):
            cnt = int(np.prod(shape))
            return np.frombuffer(buf, dtype=dtype, count=cnt, offset=o[i]).reshape(shape)

        self.hdr = np.frombuffer(buf, dtype=np.uint64, count=8, offset=0)
        self.err = np.frombuffer(buf, dtype=np.uint8, count=lay.ERR_LEN, offset=o[0])
        self.obs = v(1, np.float32, (n, 2, d))
        self.mask = v(2, np.uint8, (n, 2, ACT))
        self.need = v(3, np.uint8, (n, 2))
        self.reward = v(4, np.float32, (n,))
        self.done = v(5, np.uint8, (n,))
        self.actions = v(6, np.int32, (n, 2))
        self.rows = v(7, np.float32, (lay.K_MAX, d))
        self.ok = v(8, np.uint8, (lay.K_MAX,))
        exe = binary or TARGET / "m5_envproc"
        self.proc = subprocess.Popen(
            [str(exe), self.path, str(n), str(threads), str(seed), "1" if opp_external else "0"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
        )
        self.win, self.rout = self.proc.stdin, self.proc.stdout
        stamp = b""
        while not stamp.endswith(b"\n"):
            c = self.rout.read(1)
            if not c:
                raise RuntimeError(f"m5_envproc died at start-up (rc={self.proc.wait()})")
            stamp += c
        self.stamp = stamp.decode().strip()
        if check:
            try:
                check_stamp(self.stamp, "m5_envproc")
            except StampMismatch:
                self.close()  # never leave a refused child or its /dev/shm file behind
                raise

    def _call(self, op: bytes):
        self.win.write(op)
        st = self.rout.read(1)
        if st != b"\x00":
            if not st:
                raise RuntimeError(f"m5_envproc DIED (rc={self.proc.wait()}) — the parent survives")
            msg = bytes(self.err[: int(np.argmax(self.err == 0))]).decode()
            raise RuntimeError(f"core error: {msg}")

    def reset(self):
        self._call(b"R")

    def step(self):
        self._call(b"S")

    def successors(self, k: int):
        self.hdr[2] = k
        self._call(b"F")

    @property
    def core_ns(self) -> int:
        return int(self.hdr[4])

    @property
    def refusals(self) -> int:
        return int(self.hdr[5])

    def close(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.win.write(b"Q")
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
                self.proc.wait()
        self.proc = None
        for a in ("obs", "mask", "need", "reward", "done", "actions", "rows", "ok", "hdr", "err"):
            if hasattr(self, a):
                delattr(self, a)
        try:
            self.mm.close()
        except BufferError:
            pass
        os.close(self.fd)
        os.unlink(self.path)


def random_actions(pool, rng: np.random.Generator):
    """A seeded random policy over the mask, for every side with need = 1 (vectorised)."""
    u = rng.random(pool.mask.shape, dtype=np.float32) + 1.0
    a = np.argmax(u * pool.mask, axis=-1).astype(np.int32)
    pool.actions[...] = np.where(pool.need == 1, a, -1)


def make(kind: str, n: int, threads: int, seed: int, opp_external: bool):
    return (FfiPool if kind == "ffi" else ProcPool)(n, threads, seed, opp_external)


def now() -> int:
    return time.perf_counter_ns()
