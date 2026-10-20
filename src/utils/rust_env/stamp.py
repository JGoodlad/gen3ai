"""The Rust env's BUILD STAMP — recomputed from THIS tree and compared at load (M5 Lane 0, gate ⑤).

The 09-09 rust-target incident was a stale BINARY running silently; a stale or foreign ``.so`` first
on a search path is the same class. So every build of the env core carries a stamp
(``src/rust_env/build.rs`` → ``pokesim_env::core::STAMP``) and BOTH front ends call
:func:`check_stamp` on the build they are about to use, BEFORE its first op:

    stamp=v1;commit=<sha>;src=<fnv64>;nfiles=<k>;nan_poison=<0|1>;schema=<SCHEMA_ID>;data=<abs path>

``src`` is the refusal key — FNV-1a-64 over the sorted ``"<tag>/<rel>\\t<git blob id>\\n"`` listing of
every file that reaches the build (:func:`source_listing`; ``build.rs`` is the Rust twin — keep the two
identical). ``schema`` must equal this checkout's column table (``columns.schema_id()``); ``nan_poison`` can be
demanded (a self-check build NaN-prefills a row, a release build zero-fills it). ``commit`` is informational.

``data`` is the COMPILE-TIME default only (the build checkout's ``data/pokemon``). Since ``gen3_dex_runtime_data_dir_v1``
the Rust side resolves the dir it READS at RUN time (``pokesim::dex::data_dir``: ``$POKESIM_DATA_DIR`` -> the repo holding the
running module -> that compile-time default), so comparing the BUILD's checkout with this one would refuse a legitimate
binary (built in a worktree, sitting in main's ``target/``, reading main's data) and say nothing about what it reads.
:func:`runtime_data_dir` re-implements that rule for the module being loaded, and :func:`check_stamp` demands the
RESOLVED dir be THIS checkout's ``data/pokemon`` — "reads here", the property that matters; with ``src`` it pins the sources.
A caller that cannot name the module gets the compile-time dir (the old check).

Dependency-free except ``utils.paths`` and the column table; the recompute is a few ms.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from utils.paths import repo_path, src_path
from utils.rust_env import columns


class StampMismatch(ImportError):
    """The build is not THIS tree's build — rebuild it from this checkout."""


def source_listing(src: Optional[Path] = None) -> List[Tuple[str, Path]]:
    """``(tag/rel, path)`` of every file that reaches the build, sorted (``build.rs``'s set)."""
    src = Path(src) if src is not None else src_path()
    out: List[Tuple[str, Path]] = []
    for crate, tag, extra in ((src / "rust_sim", "port", ()), (src / "rust_env", "env", ("build.rs",))):
        files = sorted((crate / "src").rglob("*.rs")) + [crate / "Cargo.toml"] + [crate / e for e in extra]
        for p in files:
            out.append((f"{tag}/{p.relative_to(crate).as_posix()}", p))
    out.sort()
    return out


def _fnv64(data: bytes) -> str:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def source_hash(src: Optional[Path] = None) -> Tuple[str, int]:
    """``(src hash, file count)`` of the tree under ``src`` (default: this checkout's ``src/``)."""
    lines = []
    listing = source_listing(src)
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
        raise StampMismatch(f"an unreadable build stamp: {stamp!r}") from None
    if d.get("stamp") != "v1":
        raise StampMismatch(f"an unknown build-stamp version: {stamp!r}")
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


def check_stamp(stamp: str, what: str, *, src: Optional[Path] = None,
                nan_poison: Optional[bool] = None, module: Optional[Path] = None) -> Dict[str, str]:
    """REFUSE (``StampMismatch``, naming both sides) a build that is not this tree's; return the
    parsed stamp otherwise. ``src`` overrides the tree hashed (tests); ``nan_poison`` demands a
    self-check (True) or release (False) build; ``module`` is the loaded ``.so`` / child binary, whose
    location (with ``$POKESIM_DATA_DIR``) decides the data dir it reads (:func:`runtime_data_dir`)."""
    st = parse_stamp(stamp)
    want, n = source_hash(src)
    problems = []
    if st.get("src") != want or st.get("nfiles") != str(n):
        problems.append(f"sources: the build hashes src={st.get('src')} over {st.get('nfiles')} files, "
                        f"this tree src={want} over {n}")
    if st.get("schema") != columns.schema_id():
        problems.append(f"column schema: the build has {st.get('schema')}, this tree's table {columns.schema_id()}")
    if src is None:
        reads, how = runtime_data_dir(st.get("data", ""), module)
        if reads != data_dir():
            problems.append(f"data: the build reads {reads} ({how}), this checkout is {data_dir()}")
    if nan_poison is not None and st.get("nan_poison") != str(int(nan_poison)):
        problems.append(f"build kind: nan_poison={st.get('nan_poison')}, the caller demands {int(nan_poison)}")
    if problems:
        raise StampMismatch(f"{what}: a stale or foreign rust env build (commit {st.get('commit')}) — "
                            + "; ".join(problems) + " — rebuild it from THIS checkout")
    return st
