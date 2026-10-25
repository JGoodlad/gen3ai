"""The Rust BUILD STAMP — recomputed from THIS tree and compared at load / spawn (M5 Lane 0 gate ⑤, widened).

The 09-09 rust-target incident was a stale BINARY running silently; a stale or foreign ``.so`` first on a search path, or
a ``$POKESIM_*_BIN`` naming a binary built before the last Rust commit, is the same class (2026-10-20: main's
``libpokesim_env.so`` and drivers predated two Rust commits, so anything that took them by path ran behaviour no commit
pins). So EVERY Rust artifact Python runs carries a stamp and is judged against the checkout Python runs in, BEFORE its
first op:

* the env core's cdylib / ``rust_env_proc`` / ``bot_reader`` (kind ``env``; ``src/rust_env/build.rs``):
  ``stamp=v2;kind=env;commit=<sha>;src=<fnv64>;nfiles=<k>;nan_poison=<0|1>;schema=<SCHEMA_ID>;data=<abs path>`` —
  ``ffi.load``, the ``proc`` handshake and ``bot_reader_bin`` call :func:`check_stamp` / :func:`check_binary_stamp`;
* the port's drivers ``sim_bridge`` / ``search_driver`` / ``core_events`` / ``live_reader`` (kind ``port``;
  ``src/rust_sim/build.rs``): ``stamp=v2;kind=port;commit=<sha>;src=<fnv64>;nfiles=<k>;selfcheck=<0|1>`` —
  ``<bin> --build-stamp`` prints it and ``utils.bridge.sim_bridge_bin`` runs :func:`check_binary_stamp` on whatever
  path it resolves (a ``$POKESIM_*_BIN`` override included), at every resolve.

``src`` is the refusal key: FNV-1a-64 over the sorted ``"<tag>/<rel>\\t<git blob id>\\n"`` listing of every file that
reaches a NON-TEST build of the crates in the artifact's SCOPE (:func:`source_listing`; ``rust_sim/build_support.rs`` is
the Rust twin: keep the two identical, ``stamp_test.py`` proves it on a real build). Test-only modules, ``tests/``,
goldens and docs are not in it, so editing them never stales a build. ``schema`` must equal this checkout's column
table; ``nan_poison`` can be demanded; ``commit`` is informational (a build script does not re-run when HEAD moves).

WHICH CHECKOUT: the one this module is imported from (``utils.paths``), i.e. the PIN's tree in a pinned run (the launcher
puts the pin's ``src`` on ``PYTHONPATH``), so a pinned trainer compares its own library against its own sources and
can never be refused by main's. An artifact that is not this tree's is refused with a one-line :class:`StampMismatch`
naming the built-at ``src`` / commit, this tree's ``src`` and the rebuild command.

THE OPT-OUT: ``POKESIM_ALLOW_STALE_BUILD=1`` lets a build whose SOURCES differ (or that reports no stamp) run anyway,
logging a loud multi-line warning on stderr (once per artifact and process). It never waives the column schema, the data
directory or the build kind, which are not about staleness. Use it for a deliberate stale run, never to get past a refusal.

``data`` is the COMPILE-TIME default only; the Rust side resolves the dir it READS at RUN time (``pokesim::dex::data_dir``:
``$POKESIM_DATA_DIR`` -> the repo holding the running module -> that default). :func:`runtime_data_dir` re-implements that
rule for the module being loaded and :func:`check_stamp` demands the RESOLVED dir be THIS checkout's ``data/pokemon``
("reads here"), which with ``src`` pins what the module runs and reads.

Dependency-free except ``utils.paths`` and the column table; the recompute is ~5 ms.
"""
from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from utils.paths import repo_path, src_path
from utils.rust_env import columns

log = logging.getLogger(__name__)

#: The stamp grammar this loader reads (``build.rs`` writes it).
STAMP_VERSION = "v2"
#: ``=1`` runs a build whose sources are not this tree's (a deliberate stale run), logging loudly.
ALLOW_STALE_ENV = "POKESIM_ALLOW_STALE_BUILD"
#: The first-argument flag every Rust binary answers with its stamp (``dex::data_dir::exit_on_flags``).
BUILD_STAMP_FLAG = "--build-stamp"

#: tag -> (crate dir under ``src/``, files beside ``Cargo.toml`` that the build runs or includes) — ``build_support.rs``'s
#: ``port_spec`` / ``env_spec``.
_CRATES: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "port": ("rust_sim", ("build.rs", "build_support.rs")),
    "env": ("rust_env", ("build.rs",)),
}
#: kind -> the crates its build compiles, ROOT LAST (the root's ``Cargo.lock`` is hashed; a dependency's is not, because
#: cargo writes only the root's and a later port build would otherwise flip the env hash).
SCOPES: Dict[str, Tuple[str, ...]] = {"port": ("port",), "env": ("port", "env")}


class StampMismatch(ImportError):
    """The build is not THIS tree's build — rebuild it from this checkout."""


def is_test_only(rel: str) -> bool:
    """A crate-relative ``/``-path that only a TEST build compiles (``#[cfg(test)] mod x;``), by name: ``tests.rs``,
    ``*_tests.rs``, ``*_test.rs``, or anything under a ``tests/`` directory (``build_support.rs::is_test_only``).
    ``stamp_test.py`` cross-checks this against the tree's real ``#[cfg(test)] mod`` declarations."""
    parts = rel.split("/")
    if "tests" in parts[:-1]:
        return True
    stem = parts[-1][:-3] if parts[-1].endswith(".rs") else parts[-1]
    return stem == "tests" or stem.endswith("_tests") or stem.endswith("_test")


def source_listing(src: Optional[Path] = None, scope: str = "env") -> List[Tuple[str, Path]]:
    """``(tag/rel, path)`` of every file that reaches a non-test build of ``scope`` (``"port"`` | ``"env"``), sorted."""
    src = Path(src) if src is not None else src_path()
    tags = SCOPES[scope]
    out: List[Tuple[str, Path]] = []
    for tag in tags:
        crate_dir, extras = _CRATES[tag]
        crate = src / crate_dir
        files = sorted((crate / "src").rglob("*.rs")) + [crate / "Cargo.toml"] + [crate / e for e in extras]
        if tag == tags[-1]:
            files.append(crate / "Cargo.lock")
        for p in files:
            if not p.is_file():
                continue
            rel = p.relative_to(crate).as_posix()
            if not is_test_only(rel):
                out.append((f"{tag}/{rel}", p))
    out.sort()
    return out


def _fnv64(data: bytes) -> str:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def source_hash(src: Optional[Path] = None, scope: str = "env") -> Tuple[str, int]:
    """``(src hash, file count)`` of the tree under ``src`` (default: this checkout's ``src/``) for ``scope``."""
    lines = []
    listing = source_listing(src, scope)
    for rel, p in listing:
        data = p.read_bytes()
        blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        lines.append(f"{rel}\t{blob}\n")
    return _fnv64("".join(lines).encode()), len(listing)


def parse_stamp(stamp: str) -> Dict[str, str]:
    s = stamp.strip().strip("\0")
    try:
        d = dict(kv.split("=", 1) for kv in s.split(";"))
    except ValueError:
        raise StampMismatch(f"an unreadable build stamp: {stamp!r} — rebuild the artifact") from None
    if d.get("stamp") != STAMP_VERSION:
        raise StampMismatch(f"a build stamp of version {d.get('stamp')!r}, this loader reads {STAMP_VERSION!r} (an older "
                            f"build script hashed other files): {stamp[:160]!r} — rebuild the artifact")
    return d


def data_dir() -> str:
    return str(repo_path("data", "pokemon").resolve())


#: The run-time override and the marker file — the Rust side's own constants (``dex/data_dir.rs``).
DATA_DIR_ENV = "POKESIM_DATA_DIR"
_MARKER = "gen3_species.json"


def runtime_data_dir(compile_default: str, module: Optional[Path] = None,
                     environ: Optional[Mapping[str, str]] = None) -> Tuple[str, str]:
    """``(dir, source)`` — the data directory the Rust build will READ, by ``dex/data_dir.rs``'s rule: a non-empty
    ``$POKESIM_DATA_DIR`` (``"env"``), else the first ancestor of the loaded ``module`` holding
    ``data/pokemon/gen3_species.json`` (``"module"``), else ``compile_default`` (the stamp's ``data``;
    ``"compile-time"``). ``module`` is resolved through symlinks (the Rust side reads ``/proc/self/maps``)."""
    env = (os.environ if environ is None else environ).get(DATA_DIR_ENV)
    if env:
        return str(Path(env).resolve()), "env"
    if module is not None:
        for d in Path(module).resolve().parents:
            cand = d / "data" / "pokemon"
            if (cand / _MARKER).is_file():
                return str(cand.resolve()), "module"
    return str(Path(compile_default).resolve()), "compile-time"


def stale_allowed(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether ``POKESIM_ALLOW_STALE_BUILD=1`` is set (exactly ``1``; anything else is off)."""
    return (os.environ if environ is None else environ).get(ALLOW_STALE_ENV) == "1"


_warned: set = set()
_warned_lock = threading.Lock()


def _warn_stale_allowed(what: str, detail: str, fix: str) -> None:
    """The LOUD line for a deliberate stale run: stderr and the log, once per (artifact, detail) per process."""
    with _warned_lock:
        if (what, detail) in _warned:
            return
        _warned.add((what, detail))
    msg = (f"🚨🚨🚨 [STALE RUST BUILD ALLOWED] {ALLOW_STALE_ENV}=1 — {what}: {detail}. This process runs Rust behaviour "
           f"that NO commit pins, so nothing it produces is reproducible from a commit or comparable to a checked run. "
           f"Unset {ALLOW_STALE_ENV} and rebuild: {fix}")
    log.error(msg)
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _default_fix(kind: str, module: Optional[Path]) -> str:
    """The rebuild command for an artifact of ``kind`` at ``module`` (its profile is the directory name)."""
    try:
        if kind == "env":
            from utils.rust_env.build import PROFILE_FLAGS, build_command

            prof = Path(module).parent.name if module is not None else "selfcheck"
            return build_command(prof if prof in PROFILE_FLAGS else "selfcheck")
        from utils.bridge.sim_bridge_bin import build_command as port_build_command

        name = Path(module).name if module is not None else "sim_bridge"
        return port_build_command(name)
    except Exception:                                                                    # noqa: BLE001 - a hint, never a blocker
        return "rebuild it from THIS checkout (see src/rust_sim/CLAUDE.md § Builds)"


def check_stamp(stamp: str, what: str, *, src: Optional[Path] = None, nan_poison: Optional[bool] = None,
                module: Optional[Path] = None, kind: str = "env", fix: Optional[str] = None,
                allow_stale: Optional[bool] = None) -> Dict[str, str]:
    """REFUSE (``StampMismatch``, one line naming both sides and the fix) a build that is not this tree's; return the
    parsed stamp otherwise. ``kind`` is the artifact's scope (``"env"``: the env core; ``"port"``: a port driver) and
    must be the stamp's own. ``src`` overrides the tree hashed (tests); ``nan_poison`` demands a self-check (True) or
    release (False) build; ``module`` is the loaded ``.so`` / child binary, whose location (with ``$POKESIM_DATA_DIR``)
    decides the data dir it reads (:func:`runtime_data_dir`). ``fix`` is the rebuild command quoted in the refusal.
    ``allow_stale`` (default: ``$POKESIM_ALLOW_STALE_BUILD == "1"``) waives ONLY the sources problem, with a loud warning."""
    st = parse_stamp(stamp)
    fix = fix or _default_fix(kind, module)
    if allow_stale is None:
        allow_stale = stale_allowed()
    stale: List[str] = []
    other: List[str] = []
    if st.get("kind", "env") != kind:
        other.append(f"build kind: a {st.get('kind')!r} stamp where a {kind!r} one is expected")
    want, n = source_hash(src, kind)
    if st.get("src") != want or st.get("nfiles") != str(n):
        stale.append(f"built from src={st.get('src')} over {st.get('nfiles')} files (commit {st.get('commit')}), this checkout is "
                     f"src={want} over {n} (its Rust sources changed since)")
    if kind == "env":
        if st.get("schema") != columns.schema_id():
            other.append(f"column schema: the build has {st.get('schema')}, this tree's table {columns.schema_id()}")
        if src is None:
            reads, how = runtime_data_dir(st.get("data", ""), module)
            if reads != data_dir():
                other.append(f"data: the build reads {reads} ({how}), this checkout is {data_dir()}")
        if nan_poison is not None and st.get("nan_poison") != str(int(nan_poison)):
            other.append(f"build kind: nan_poison={st.get('nan_poison')}, the caller demands {int(nan_poison)}")
    if stale and allow_stale and not other:
        _warn_stale_allowed(what, "; ".join(stale), fix)
        return st
    problems = stale + other
    if problems:
        escape = f"; or {ALLOW_STALE_ENV}=1 to run it stale on purpose" if stale and not other else ""
        raise StampMismatch(f"{what}: STALE RUST BUILD — " + "; ".join(problems) + f" — rebuild it: {fix}{escape}")
    return st


# ------------------------------------------------------------------ binaries invoked from Python

#: ``(realpath, mtime_ns, size)`` -> the stamp text the binary printed (a binary is exec'd once per identity).
_binary_stamps: Dict[Tuple[str, int, int], str] = {}
_binary_lock = threading.Lock()


class StampUnavailable(StampMismatch):
    """A binary could not give its stamp. ``timed_out`` marks the case that is NOT a finding about the build (no answer in
    time: a timeout is never a semantic outcome); the rest — no ``--build-stamp``, not executable — mean it predates the check."""

    def __init__(self, message: str, *, timed_out: bool = False):
        super().__init__(message)
        self.timed_out = timed_out


def binary_stamp(path: Path, *, run: Callable[..., "subprocess.CompletedProcess"] = subprocess.run,
                 timeout: Optional[float] = None) -> str:
    """What ``<path> --build-stamp`` prints; :class:`StampUnavailable` when the binary cannot say (it predates the flag, is
    not executable, or does not answer within ``timeout``, default 30 s scaled by the box's contention). Cached per
    binary identity (path, mtime, size): a rebuild is re-asked, an unchanged binary is exec'd once."""
    path = Path(path)
    stt = path.stat()
    key = (str(path.resolve()), stt.st_mtime_ns, stt.st_size)
    with _binary_lock:
        hit = _binary_stamps.get(key)
    if hit:
        return hit
    if timeout is None:
        from utils.contention import scale_timeout

        timeout = scale_timeout(30.0)
    try:
        r = run([str(path), BUILD_STAMP_FLAG], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise StampUnavailable(f"{path} did not answer `{BUILD_STAMP_FLAG}` within {timeout:.0f} s (contention-scaled): a timeout is "
                               f"not a verdict on the build — retry", timed_out=True) from None
    except (OSError, subprocess.SubprocessError) as e:
        raise StampUnavailable(f"{path} reports no build stamp (it cannot be executed: {e})") from None
    lines = [ln for ln in (r.stdout or "").splitlines() if ln.startswith("stamp=")]
    if r.returncode != 0 or not lines:
        raise StampUnavailable(f"{path} reports no build stamp (it predates `{BUILD_STAMP_FLAG}`, or is not a Rust binary of ours)")
    with _binary_lock:                                    # only a real answer is cached; a failure is asked again
        _binary_stamps[key] = lines[0]
    return lines[0]


def check_binary_stamp(path: Path, what: str, *, kind: str = "port", fix: Optional[str] = None,
                       src: Optional[Path] = None, allow_stale: Optional[bool] = None,
                       run: Callable[..., "subprocess.CompletedProcess"] = subprocess.run) -> Optional[Dict[str, str]]:
    """Judge the binary at ``path`` before it is spawned: run ``<path> --build-stamp`` and :func:`check_stamp` the answer
    against THIS checkout (``kind`` = the crate it was built from: ``"port"`` for ``sim_bridge`` / ``search_driver`` /
    ``core_events`` / ``live_reader``, ``"env"`` for ``bot_reader`` / ``rust_env_proc``). A binary that reports no stamp
    predates the check and is refused as stale; one that does not answer in time is refused as UNVERIFIED (never as stale).
    ``$POKESIM_ALLOW_STALE_BUILD=1`` lets either run, loudly. Returns the parsed stamp (``None`` when none was available)."""
    path = Path(path)
    fix = fix or _default_fix(kind, path)
    if allow_stale is None:
        allow_stale = stale_allowed()
    try:
        text = binary_stamp(path, run=run)
    except StampUnavailable as e:
        if allow_stale:
            _warn_stale_allowed(what, str(e), fix)
            return None
        if e.timed_out:
            raise StampUnavailable(f"{what}: {e}; or {ALLOW_STALE_ENV}=1 to skip the check", timed_out=True) from None
        raise StampMismatch(f"{what}: STALE RUST BUILD — {e} — rebuild it: {fix}; or {ALLOW_STALE_ENV}=1 to run it stale on "
                            f"purpose") from None
    return check_stamp(text, f"{what} ({path})", src=src, kind=kind, module=path, fix=fix, allow_stale=allow_stale)


# ------------------------------------------------------------------ the command line (for the Node harnesses)


def main(argv: Optional[List[str]] = None) -> int:
    """``python -m utils.rust_env.stamp --check-binary <path> [--kind port|env] [--what NAME]``: judge a binary by its build
    stamp against THIS checkout (``PYTHONPATH`` names which), exactly as the Python resolvers do. Exit 0 = current (or
    ``POKESIM_ALLOW_STALE_BUILD=1``, warned loudly on stderr), 1 = refused (the one-line reason on stderr). The Node
    fuzz harnesses (``src/rust_sim/harness/stamp_guard.js``) call it before they exec a replayer."""
    import argparse

    ap = argparse.ArgumentParser(prog="python -m utils.rust_env.stamp", description=main.__doc__)
    ap.add_argument("--check-binary", required=True, metavar="PATH")
    ap.add_argument("--kind", choices=sorted(SCOPES), default="port")
    ap.add_argument("--what", default=None, help="the name quoted in a refusal (default: the file name)")
    a = ap.parse_args(argv)
    path = Path(a.check_binary)
    try:
        check_binary_stamp(path, a.what or path.name, kind=a.kind)
    except (StampMismatch, OSError) as e:
        sys.stderr.write(f"{e}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
