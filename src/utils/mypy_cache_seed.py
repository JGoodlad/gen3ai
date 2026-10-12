"""SEED a checkout's EMPTY mypy cache from a sibling checkout's, by a consistent COPY — never by sharing.

WHY (2026-10-12). The two mypy-backed static gates (``agents/model/mypy_gate_test.py``,
``enum_str_compare_gate_test.py``) pay a COLD cache once per checkout, and every agent works in a FRESH
worktree: measured on the 3 agent cores beside a live training run (nice 19, contention x5.7), the
static tier took **237 s cold against 87 s warm** at ``-n 2`` — the enum-str gate 185 s cold, the mypy
gate 78 s cold, each ~0.3-2 s warm. Most of a cold run is third-party stubs (torch, numpy) whose paths are
identical in every checkout. Since 2026-10-12 the static tier IS the frequent pre-ship gate while the GPU
is leased (root ``CLAUDE.md``), so that cold cost lands on nearly every agent's first ship.

WHY A COPY IS SAFE — mypy validates every cached module itself. A cache entry records the module's
source path, size, mtime and CONTENT HASH, the options that affect it and the mypy version; on a path or
mtime mismatch mypy re-hashes the source and reuses the entry only if the hash matches, and a module
whose dependency's INTERFACE hash changed is re-checked. So a seed can make a run fast, never let an
error through. Pinned by ``mypy_cache_seed_test.py`` on a real mypy: a seed taken from a checkout where
``a.f`` returns ``int``, applied where it returns ``str``, still reports the error in the UNCHANGED
dependent ``b.py``; verified 2026-10-12 on mypy 2.3.1 with torch / numpy imports (seeded run 0.5 s,
error caught; the cold control 34.7 s, same error).

WHY NOT SHARE ONE DIR. mypy rewrites entries with this checkout's paths, so two checkouts on one dir
thrash each other (the enum-str gate's docstring), and a gate must not read a dir another process is
writing mid-run. A copy is taken ONCE, when this checkout has no cache, and is this checkout's own from
then on.

HOW THE COPY STAYS CONSISTENT. mypy 2.x keeps its cache in SQLite shards (``<pyver>/cache.<n>.db``):
each is copied with SQLite's online BACKUP API (a consistent snapshot even while the sibling's mypy
writes) and must pass ``PRAGMA quick_check``; any other file is copied as is and a ``.json`` one must
parse. The copy is built in a staging dir and RENAMED into place, so a concurrent seeder or a crash
leaves either no cache (a cold run, as before) or a whole one. ANY failure discards the seed: the
worst case is today's cold run, never a red gate from a torn file.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Callable, Iterable, Optional

SEED_ENV = "GEN3AI_MYPY_CACHE_SEED"          # "0" turns seeding off (every gate then runs cold)


def has_cache(d: Path) -> bool:
    """Whether ``d`` already holds a mypy cache (SQLite shards or JSON metas) — then it is never seeded."""
    if not d.is_dir():
        return False
    return any(d.glob("*/cache.*.db")) or any(d.glob("*/**/*.meta.json"))


def _newest_mtime(d: Path) -> float:
    try:
        return max((p.stat().st_mtime for p in d.glob("*/cache.*.db")), default=d.stat().st_mtime)
    except OSError:
        return 0.0


def pick_source(own: Path, candidates: Iterable[Path]) -> Optional[Path]:
    """The most recently written candidate that holds a cache (never ``own``, never a symlink, never
    another uid's)."""
    best: Optional[Path] = None
    best_t = -1.0
    own_r = own.resolve() if own.exists() else own
    for c in candidates:
        try:
            if c.is_symlink() or not c.is_dir() or c.stat().st_uid != os.getuid():
                continue
            if c.resolve() == own_r or not has_cache(c):
                continue
        except OSError:
            continue
        t = _newest_mtime(c)
        if t > best_t:
            best, best_t = c, t
    return best


def _snapshot(src: Path, dst: Path) -> None:
    """Copy ``src`` into the NEW dir ``dst``: SQLite shards by the backup API + quick_check, the rest as
    is (a ``.json`` must parse). Raises on anything inconsistent."""
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if p.is_symlink():
            continue
        if p.is_dir():
            (dst / rel).mkdir(parents=True, exist_ok=True)
            continue
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".db":
            s = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=30)
            try:
                d = sqlite3.connect(str(dst / rel))
                try:
                    s.backup(d)
                    ok = d.execute("PRAGMA quick_check").fetchone()
                finally:
                    d.close()
            finally:
                s.close()
            if not ok or ok[0] != "ok":
                raise sqlite3.DatabaseError(f"{rel}: quick_check {ok!r}")
        elif p.name.endswith((".db-wal", ".db-shm", ".db-journal")):
            continue                        # the backup above already holds a consistent image
        else:
            shutil.copy2(p, dst / rel)
            if p.suffix == ".json":
                json.loads((dst / rel).read_text())


def seed(own: Path, candidates: Iterable[Path], *, log: Callable[[str], None] = lambda _m: None) -> Optional[Path]:
    """Seed ``own`` from the best candidate when ``own`` holds no cache; returns the source used, or None
    (seeding off, ``own`` already warm, no candidate, or a failed copy — each a cold run, never an error)."""
    if os.environ.get(SEED_ENV) == "0" or has_cache(own):
        return None
    src = pick_source(own, candidates)
    if src is None:
        return None
    own.parent.mkdir(parents=True, exist_ok=True)
    staging = own.parent / f".{own.name}.seed.{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        staging.mkdir()
        _snapshot(src, staging)
        if own.exists() and not any(own.iterdir()):
            own.rmdir()                     # an empty dir (a first mkdir) is not a cache
        os.rename(staging, own)             # fails if a concurrent seeder or mypy got there first
    except (OSError, sqlite3.Error, ValueError) as exc:
        shutil.rmtree(staging, ignore_errors=True)
        log(f"[mypy cache seed] NOT seeded from {src} ({exc!r}): a cold run")
        return None
    log(f"[mypy cache seed] {own} seeded from {src}")
    return src


def sibling_checkout_caches(name: str = ".mypy_cache") -> list:
    """``<checkout>/<name>`` for the MAIN checkout and every linked worktree of this repo (``git
    worktree list``), this one included (``pick_source`` skips it)."""
    import subprocess

    from utils.paths import repo_root

    try:
        out = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=str(repo_root()),
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [Path(line.split(" ", 1)[1]) / name for line in out.splitlines() if line.startswith("worktree ")]
