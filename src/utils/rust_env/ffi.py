"""The Rust env's FFI FRONT END — the core as a C-ABI ``cdylib``, loaded with ``ctypes`` (M5 Lane A,
``designs/endstate/program_rust_core.md`` §2 M5).

A THIN front end: every call forwards to ``pokesim_env::core::Core`` (``Core::new(Spec)`` →
``freeze(cols)`` → ``dispatch(op, cols)``) and turns a non-OK status into the typed class of
``protocol.py``. It holds no battle logic and names no op's semantics.

THE SIGNATURES ARE GENERATED from ONE table (``FUNCTIONS`` below): ``render()`` writes the
``extern "C"`` wrappers into the marked region of ``src/rust_env/src/ffi.rs`` (``python -m
utils.rust_env.ffi --write``), and :func:`load` sets every ``argtypes`` / ``restype`` from the same
rows — so the two sides cannot disagree about an argument. Each generated wrapper calls a
hand-written ``imp::<name>`` with the SAME argument list, so a Rust implementation whose types drift
from the table does not compile. ``ffi_test.py`` (routine) fails the day the committed region is
stale; the table's hash (``sig_id()``) is compiled into the library and compared at load.

LOADING (the 09-09 class — a stale or foreign binary running silently):
  * by ABSOLUTE path only (default: THIS checkout's ``src/rust_env/target/<profile>/libpokesim_env.so``);
  * BEFORE any op: the build STAMP (``stamp.check_stamp`` — sources, column schema, data dir, build
    kind), the column schema id, ``N_COLUMNS``, and the FFI table's id are compared; any mismatch is
    ``StampMismatch`` / ``FfiLoadError`` and nothing is dispatched.

PANICS: every exported function runs inside ``catch_unwind``; a Rust panic comes back as status
``PANIC`` and :class:`protocol.CorePanic`, never an abort (unwinding out of ``extern "C"`` aborts the
process). What cannot be caught in-process — an abort, a stack overflow, OOM, a fault in unsafe
code — still takes the process with it; that is the process front end's (Lane B) reason to exist.

THE DECLARED LIFECYCLE through FFI: :class:`FfiCore` allocates every column from ``columns.py`` at
construction, FREEZES the binding, and never re-binds; the ``*_AFTER_FREEZE`` counters are readable
(:meth:`FfiCore.after_freeze`) and must stay 0. ``ctypes`` releases the GIL for every foreign call.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.paths import src_path
from utils.rust_env import columns as C
from utils.rust_env import protocol as P
from utils.rust_env import stamp as S

# ------------------------------------------------------------------ THE TABLE

#: ABI type -> (Rust type, ctypes type).
ABI: Dict[str, Tuple[str, object]] = {
    "handle": ("*mut Handle", ctypes.c_void_p),
    "cstr": ("*const c_char", ctypes.c_char_p),
    "addrs": ("*const usize", ctypes.c_void_p),  # N_COLUMNS column addresses, in columns.COLUMNS order
    "u64_out": ("*mut u64", ctypes.c_void_p),
    "u8": ("u8", ctypes.c_uint8),
    "i32": ("i32", ctypes.c_int32),
    "usize": ("usize", ctypes.c_size_t),
    "void": ("()", None),
}


@dataclass(frozen=True)
class Fn:
    name: str
    args: Tuple[Tuple[str, str], ...]  # (argument name, ABI type)
    ret: str  # ABI type
    doc: str


#: Every exported function. A status-returning function returns a ``protocol.STATUSES`` code; on a
#: non-OK code the typed error is ``rust_env_last_error()`` (a JSON object, this thread's).
FUNCTIONS: Tuple[Fn, ...] = (
    Fn("rust_env_stamp", (), "cstr", "the build stamp (`stamp.py`'s format); static"),
    Fn("rust_env_ffi_sig", (), "cstr", "this table's id (`ffi.sig_id()`), compiled in; static"),
    Fn("rust_env_schema_id", (), "cstr", "the column schema id (`columns.schema_id()`); static"),
    Fn("rust_env_obs_dim", (), "usize", "the encoder's row length (`pokesim::encoder::OBS_DIM`)"),
    Fn("rust_env_n_columns", (), "usize", "`columns::N_COLUMNS` (the length of every `addrs` array)"),
    Fn("rust_env_last_error", (), "cstr",
       "the last failure ON THIS THREAD as `DispatchError::json` (valid until this thread's next call)"),
    Fn("rust_env_new", (("spec_json", "cstr"),), "handle",
       "STARTUP: `Core::new(Spec::from_json)`; null on failure (then `rust_env_last_error`)"),
    Fn("rust_env_n", (("h", "handle"),), "usize", "the pool's N (usize::MAX on a null handle or a panic)"),
    Fn("rust_env_freeze", (("h", "handle"), ("addrs", "addrs")), "i32", "FREEZE: `Core::freeze` (once)"),
    Fn("rust_env_dispatch", (("h", "handle"), ("op", "u8"), ("addrs", "addrs")), "i32",
       "THE ENTRY: `Core::dispatch(op, cols)`"),
    Fn("rust_env_counters", (("h", "handle"), ("out", "u64_out"), ("len", "usize")), "i32",
       "copy the pool counters (`len` must be NCOUNTERS) — readable before freeze too"),
    Fn("rust_env_bank_json", (("h", "handle"),), "cstr",
       "the refusal bank as a JSON array of `Banked::json` (valid until this thread's next call)"),
    Fn("rust_env_free", (("h", "handle"),), "void", "drop the core (joins its workers); null is a no-op"),
    Fn("rust_env_panic_probe", (("h", "handle"), ("kind", "i32")), "i32",
       "TEST HOOK: panic on purpose (0: a string payload; 1: a non-string payload; 2: inside the handle's "
       "locked section, which poisons it) — proves a panic becomes status PANIC, never an abort"),
)

MARK_BEGIN = "// ---- @generated-begin by `python -m utils.rust_env.ffi --write` — DO NOT EDIT this region."
MARK_END = "// ---- @generated-end"
FFI_RS = src_path("rust_env", "src", "ffi.rs")
LIB_NAME = "libpokesim_env.so"


def sig_text() -> str:
    """The canonical text of the table (+ the column schema id the addresses follow)."""
    lines = [f"schema {C.schema_id()}", f"columns {len(C.COLUMNS)}"]
    for f in FUNCTIONS:
        lines.append(f"fn {f.name}({', '.join(f'{a}: {t}' for a, t in f.args)}) -> {f.ret}")
    return "\n".join(lines) + "\n"


def sig_id() -> str:
    h = 0xCBF29CE484222325
    for b in sig_text().encode():
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def _check_table() -> None:
    names = [f.name for f in FUNCTIONS]
    if len(set(names)) != len(names):
        raise AssertionError(f"duplicate FFI functions: {names}")
    for f in FUNCTIONS:
        if f.ret not in ABI or any(t not in ABI or t == "void" for _, t in f.args):
            raise AssertionError(f"{f.name}: an unknown ABI type")
        if not f.name.startswith("rust_env_"):
            raise AssertionError(f"{f.name}: every export is prefixed rust_env_")


_check_table()


def render_region() -> str:
    """The generated region of ``ffi.rs`` (markers included)."""
    out: List[str] = [MARK_BEGIN,
                      "// Source of truth: `FUNCTIONS` in `src/utils/rust_env/ffi.py`; pinned by `ffi_test.py` (routine).",
                      "// Each wrapper runs `imp::<name>` (hand-written, the SAME arguments) inside `guard`.",
                      "",
                      "/// `ffi.sig_id()` — FNV-1a-64 of the table's canonical text; compared by the loader.",
                      f"pub const FFI_SIG_ID: &str = \"{sig_id()}\";",
                      "/// The same, NUL-terminated, for `rust_env_ffi_sig`.",
                      f"const FFI_SIG_ID_C: &str = \"{sig_id()}\\0\";",
                      ""]
    for f in FUNCTIONS:
        params = ", ".join(f"{a}: {ABI[t][0]}" for a, t in f.args)
        call = ", ".join(a for a, _ in f.args)
        ret = "" if f.ret == "void" else f" -> {ABI[f.ret][0]}"
        out.append(f"/// {f.doc}")
        out.append("///")
        out.append("/// # Safety")
        out.append("/// The caller passes what the table's row says (a handle from `rust_env_new`, a")
        out.append("/// NUL-terminated string, an array of `N_COLUMNS` live, aligned column addresses).")
        out.append("#[no_mangle]")
        out.append(f"pub unsafe extern \"C\" fn {f.name}({params}){ret} {{")
        out.append(f"    guard(AssertUnwindSafe(|| imp::{f.name.removeprefix('rust_env_')}({call})))")
        out.append("}")
        out.append("")
    out.append(MARK_END)
    return "\n".join(out)


def render(current: str) -> str:
    """``current`` (the file's text) with its generated region replaced by :func:`render_region`."""
    a, b = current.find(MARK_BEGIN), current.find(MARK_END)
    if a < 0 or b < a:
        raise AssertionError(f"{FFI_RS}: the generated-region markers are missing")
    return current[:a] + render_region() + current[b + len(MARK_END):]


# ------------------------------------------------------------------ loading


class FfiLoadError(ImportError):
    """The library at the path is not a loadable, matching FFI build (a missing symbol, another
    table, another column layout)."""


def default_path(profile: str = "selfcheck") -> Path:
    """THIS checkout's build of the cdylib (``cargo build --lib`` into ``src/rust_env/target``)."""
    return src_path("rust_env", "target", profile, LIB_NAME).resolve()


def load(path: Path, *, nan_poison: Optional[bool] = None, check: bool = True) -> ctypes.CDLL:
    """Load the cdylib at the ABSOLUTE ``path``; set every signature from ``FUNCTIONS``; REFUSE a
    build that is not this tree's (stamp, schema, column count, FFI table) before returning.

    ``nan_poison`` demands a self-check (True) or release (False) build. ``check=False`` skips only
    the SOURCE stamp (a test of the refusal itself loads a foreign copy); the ABI checks always run.

    ⚠️ ``dlopen`` returns the ALREADY-LOADED image for a path it has seen in this process — a
    rebuild in place is not re-read until a new process. Load a copy at a new path to compare builds.
    """
    path = Path(path)
    if not path.is_absolute():
        raise FfiLoadError(f"the rust env library must be loaded by ABSOLUTE path, got {path}")
    if not path.exists():
        raise FfiLoadError(f"{path} does not exist — build it: (cd src/rust_env && CARGO_TARGET_DIR=$PWD/target "
                           "cargo build --lib --profile selfcheck --features emission-selfcheck)")
    lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    for f in FUNCTIONS:
        try:
            fn = getattr(lib, f.name)
        except AttributeError:
            raise FfiLoadError(f"{path}: no symbol {f.name} — a build of another FFI table") from None
        fn.argtypes = [ABI[t][1] for _, t in f.args]
        fn.restype = ABI[f.ret][1]
    got_sig = lib.rust_env_ffi_sig().decode()
    if got_sig != sig_id():
        raise FfiLoadError(f"{path}: FFI table {got_sig}, this tree's {sig_id()} — rebuild it from THIS checkout")
    if lib.rust_env_schema_id().decode() != C.schema_id() or lib.rust_env_n_columns() != len(C.COLUMNS):
        raise FfiLoadError(f"{path}: column schema {lib.rust_env_schema_id().decode()} / "
                           f"{lib.rust_env_n_columns()} columns, this tree's {C.schema_id()} / {len(C.COLUMNS)}")
    if check:
        S.check_stamp(lib.rust_env_stamp().decode(), str(path), nan_poison=nan_poison)
    return lib


def _error(lib: ctypes.CDLL, status: int) -> P.RustEnvError:
    text = lib.rust_env_last_error()
    if not text:
        return P.error_for(status, "the core returned a failure with no error recorded")
    return P.error_from_json(text.decode())


# ------------------------------------------------------------------ the pool


class FfiCore:
    """One env core behind the FFI: STARTUP (``Core::new`` + every column allocated + FREEZE) in the
    constructor, then :meth:`reset` / :meth:`step`. The columns are NumPy arrays named as in
    ``columns.COLUMNS`` (``core.cols["obs"]``); inputs are written in place by the caller, outputs
    are read-only views. A failure raises the ``protocol`` class; after one the pool is POISONED
    (every later op raises ``LifecycleViolation``) — build a new one."""

    def __init__(self, spec_json: str, *, lib: Optional[ctypes.CDLL] = None, path: Optional[Path] = None,
                 nan_poison: Optional[bool] = None):
        import numpy as np

        self.lib = lib if lib is not None else load(path or default_path(), nan_poison=nan_poison)
        self._lock = threading.Lock()
        self._h = self.lib.rust_env_new(spec_json.encode())
        if not self._h:
            raise _error(self.lib, P.STATUSES[1].code)
        try:
            self.n = int(self.lib.rust_env_n(self._h))
            self.obs_dim = int(self.lib.rust_env_obs_dim())
            self.cols: Dict[str, "np.ndarray"] = dict(C.allocate(self.n, self.obs_dim))
            want = C.nbytes(self.n, self.obs_dim)
            for c in C.COLUMNS:
                a = self.cols[c.name]
                if not a.flags.c_contiguous or a.nbytes != want[c.name]:
                    raise AssertionError(f"column {c.name}: {a.nbytes} bytes, contiguous={a.flags.c_contiguous}")
            # The addresses, taken ONCE: the frozen binding is exactly this array.
            self._addrs = (ctypes.c_size_t * len(C.COLUMNS))(*[self.cols[c.name].ctypes.data for c in C.COLUMNS])
            self._addrs_p = ctypes.cast(self._addrs, ctypes.c_void_p)
            self._check(self.lib.rust_env_freeze(self._h, self._addrs_p))
            for c in C.COLUMNS:
                if c.direction == "out":
                    self.cols[c.name].flags.writeable = False  # the core writes through its own pointer
        except BaseException:
            self.close()
            raise
        self._cidx = P.counter_index()
        self._op = {o.name: o.code for o in P.OPS}

    # ---- ops
    def _check(self, status: int) -> None:
        if status != 0:
            raise _error(self.lib, status)

    def dispatch(self, op: str) -> None:
        with self._lock:
            if not self._h:
                raise P.LifecycleViolation("dispatch on a closed FfiCore")
            self._check(self.lib.rust_env_dispatch(self._h, self._op[op], self._addrs_p))

    def reset(self) -> None:
        self.dispatch("RESET")

    def step(self) -> None:
        self.dispatch("STEP")

    # ---- reads
    def counters(self) -> Dict[str, int]:
        """The pool counters, read through the FFI (not the column), so they are readable before the
        first op."""
        import numpy as np

        out = np.zeros(len(P.COUNTERS), dtype="<u8")
        with self._lock:
            self._check(self.lib.rust_env_counters(self._h, out.ctypes.data, len(P.COUNTERS)))
        return {c.name: int(out[i]) for i, c in enumerate(P.COUNTERS)}

    def after_freeze(self) -> Dict[str, int]:
        """The DECLARED-LIFECYCLE counters — each must stay 0 for the pool's life."""
        k = self.counters()
        return {c.name: k[c.name] for c in P.COUNTERS if c.after_freeze}

    def bank(self) -> List[dict]:
        with self._lock:
            text = self.lib.rust_env_bank_json(self._h)
        if text is None:
            raise _error(self.lib, P.STATUSES[1].code)
        return json.loads(text.decode())

    # ---- teardown
    def close(self) -> None:
        with self._lock:
            if getattr(self, "_h", None):
                self.lib.rust_env_free(self._h)
            self._h = None

    def __enter__(self) -> "FfiCore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="rewrite the generated region of src/rust_env/src/ffi.rs")
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed region is stale")
    a = ap.parse_args(argv)
    cur = FFI_RS.read_text()
    new = render(cur)
    if a.write:
        FFI_RS.write_text(new)
        print(f"wrote {FFI_RS} (ffi table {sig_id()})")
        return 0
    if a.check:
        ok = new == cur
        print("ffi.rs is current" if ok else "ffi.rs is STALE — run with --write")
        return 0 if ok else 1
    sys.stdout.write(render_region() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
