#!/usr/bin/env python3
"""tmp_sweep — reclaim bytes AND INODES from the RAM-backed ``/tmp`` without touching anything live.

USAGE
    python3 scripts/ops/tmp_sweep.py                 # DRY RUN: print the plan, delete nothing
    python3 scripts/ops/tmp_sweep.py --apply         # the cron form: delete what the plan names
    python3 scripts/ops/tmp_sweep.py --help

WHY. ``/tmp`` on this box is a 45 GB tmpfs with a FIXED INODE TABLE (~1.05M). On 2026-09-30 it hit
ENOSPC with 20 GB of space still free: 1,047,791 of 1,048,576 inodes were in use, ~395k of them
under ``torchinductor_<user>``. Swap was at 21 GB from what the tmpfs was holding in RAM. The
orchestrator cleared it by hand; this is that sweep as a tool, run daily from cron. It always exits
0 — a sweep that could not free something reports it and moves on.

WHAT IS PLANNED, per TOP-LEVEL entry of ``--root`` (default: the temp dir), first match wins:
  * not owned by ``--uid`` (default: me)                        -> untouched
  * on the declared KEEP list (``claude-<uid>``, ``tmux-<uid>``, ``--keep NAME``) -> untouched
  * on the declared CACHE list — compile caches a live process may be using right now:
      - FILE-pruned (``torchinductor_<user>``, ``gen3ai_inductor_cache``, ``--cache NAME``):
        every file inside older than ``--cache-max-age-hours`` (48) and not held open or mapped
        by a live process is deleted, then the directories that leaves empty (and are themselves
        that old) are removed. The cache root itself stays. A miss costs one recompile.
      - CHILD-pruned (``pytest-of-<user>``): each ``pytest-N`` child is judged like a top-level
        entry at the cache age — pytest copies fixtures with their original mtimes, so a
        file-level cut could hollow out a live session's basetemp.
  * anything else is DELETED WHOLE when ALL of these hold:
      - its NEWEST mtime anywhere inside (lstat, the entry itself included) is older than
        ``--min-age-hours`` (6);
      - no live process references it: cwd, root, an open fd, a memory map, a path on its
        command line or in its environment, or a bound unix socket in ``/proc/net/unix``;
      - it contains no socket, fifo or device node (a listening socket's fd reads
        ``socket:[inode]``, never its path — tmux and ssh-agent live exactly there).

SYMLINKS ARE NEVER FOLLOWED. The walk uses ``lstat``; a top-level symlink is unlinked, never its
target; ``shutil.rmtree`` removes a link inside a tree without descending into it. Nothing outside
``--root`` is ever deleted.

Every candidate is RE-CHECKED at apply time (age and references), so a directory that came alive
between the scan and the delete survives.

OUTPUT: the plan (name, age, bytes, INODES, reason), the skip counts, ``df`` and ``df -i`` for the
filesystem, and a loud ``WARNING`` line when bytes or inodes are above ``--warn-pct`` (70%).
Bytes are ALLOCATED bytes (``st_blocks``), which is what a tmpfs holds in RAM.
"""
from __future__ import annotations

import argparse
import getpass
import os
import re
import shutil
import stat
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

MIN_AGE_HOURS = 6.0
CACHE_MAX_AGE_HOURS = 48.0
WARN_PCT = 70.0
_PROC = "/proc"


def _user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 — no passwd entry (a container): fall back to the uid
        return str(os.getuid())


def default_keep(uid: int) -> Set[str]:
    """Never touched at all: the Claude session scratch and the tmux server socket dir."""
    return {f"claude-{uid}", f"tmux-{uid}"}


def default_file_caches() -> Set[str]:
    return {f"torchinductor_{_user()}", "gen3ai_inductor_cache"}


def default_child_caches() -> Set[str]:
    return {f"pytest-of-{_user()}"}


# ------------------------------------------------------------------ live references ----

@dataclass
class LiveRefs:
    """What live processes hold under the root: top-level NAMES, and exact PATHS (for cache files)."""
    names: Dict[str, List[int]] = field(default_factory=dict)   # top-level name -> pids (0 = socket table)
    paths: Set[str] = field(default_factory=set)
    unreadable_pids: int = 0

    def add(self, root: str, path: str, pid: int) -> None:
        if path.endswith(" (deleted)"):
            path = path[: -len(" (deleted)")]
        if not path.startswith(root + "/"):
            return
        rest = path[len(root) + 1:]
        name = rest.split("/", 1)[0]
        if not name:
            return
        self.names.setdefault(name, []).append(pid)
        self.paths.add(path)


def scan_live_refs(root: str, proc: str = _PROC) -> LiveRefs:
    """Every reference a live process holds under ``root`` (a realpath)."""
    refs = LiveRefs()
    token = re.compile(re.escape(root) + r"/[^\s\0:;'\"=,]+")
    for n in os.listdir(proc):
        if not n.isdigit():
            continue
        pid = int(n)
        base = f"{proc}/{pid}"
        readable = False
        for link in ("cwd", "root"):
            try:
                refs.add(root, os.readlink(f"{base}/{link}"), pid)
                readable = True
            except OSError:
                pass
        try:
            for fd in os.listdir(f"{base}/fd"):
                try:
                    refs.add(root, os.readlink(f"{base}/fd/{fd}"), pid)
                except OSError:
                    pass
            readable = True
        except OSError:
            pass
        try:
            with open(f"{base}/maps", "r", errors="replace") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split(None, 5)
                    if len(parts) == 6:
                        refs.add(root, parts[5].strip(), pid)
        except OSError:
            pass
        for name in ("cmdline", "environ"):
            try:
                with open(f"{base}/{name}", "rb") as fh:
                    text = fh.read().decode("utf-8", "replace")
            except OSError:
                continue
            for m in token.finditer(text):
                refs.add(root, m.group(0), pid)
        if not readable:
            refs.unreadable_pids += 1
    try:   # bound unix sockets: a listener's fd shows socket:[inode], only this table has the path
        with open(f"{proc}/net/unix", "r", errors="replace") as fh:
            next(fh, None)
            for line in fh:
                parts = line.split()
                if len(parts) >= 8:
                    refs.add(root, parts[7].lstrip("@"), 0)
    except OSError:
        pass
    return refs


# ------------------------------------------------------------------------- the walk ----

@dataclass
class Tree:
    newest_mtime: float
    bytes: int
    inodes: int
    special: Optional[str]   # the first socket / fifo / device found, or None


def walk(path: str) -> Tree:
    """lstat-only walk: the newest mtime anywhere, allocated bytes, inode count, any special file."""
    st = os.lstat(path)
    newest, nbytes, inodes = st.st_mtime, st.st_blocks * 512, 1
    special = path if _is_special(st.st_mode) else None
    if not stat.S_ISDIR(st.st_mode):
        return Tree(newest, nbytes, inodes, special)
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False, onerror=lambda e: None):
        for nm in dirnames + filenames:
            p = os.path.join(dirpath, nm)
            try:
                s = os.lstat(p)
            except OSError:
                continue
            newest = max(newest, s.st_mtime)
            nbytes += s.st_blocks * 512
            inodes += 1
            if special is None and _is_special(s.st_mode):
                special = p
    return Tree(newest, nbytes, inodes, special)


def _is_special(mode: int) -> bool:
    return stat.S_ISSOCK(mode) or stat.S_ISFIFO(mode) or stat.S_ISCHR(mode) or stat.S_ISBLK(mode)


# ------------------------------------------------------------------------- planning ----

@dataclass
class Item:
    path: str
    name: str
    kind: str            # "entry" (delete whole) | "cache-file" | "cache-child"
    age_h: float
    bytes: int
    inodes: int
    reason: str


@dataclass
class Plan:
    delete: List[Item] = field(default_factory=list)
    skipped: Dict[str, int] = field(default_factory=dict)
    skipped_examples: Dict[str, List[str]] = field(default_factory=dict)
    cache_files: Dict[str, List[Item]] = field(default_factory=dict)   # cache name -> old files
    cache_dirs: Dict[str, List[str]] = field(default_factory=dict)     # cache name -> empty dirs to remove

    def skip(self, why: str, name: str) -> None:
        self.skipped[why] = self.skipped.get(why, 0) + 1
        ex = self.skipped_examples.setdefault(why, [])
        if len(ex) < 5:
            ex.append(name)


def _held(path: str, refs: LiveRefs) -> bool:
    return any(p == path or p.startswith(path + "/") for p in refs.paths)


def _judge_whole(path: str, name: str, now: float, min_age_s: float, refs: LiveRefs,
                 ref_name: Optional[str], plan: Plan, kind: str) -> Optional[Item]:
    """``ref_name`` = the top-level name to check against live refs; None = check by exact PATH."""
    try:
        t = walk(path)
    except OSError as e:
        plan.skip(f"unreadable ({e.__class__.__name__})", name)
        return None
    age = now - t.newest_mtime
    if age < min_age_s:
        plan.skip("too new (newest mtime inside)", name)
        return None
    if ref_name is not None and ref_name in refs.names:
        pids = sorted(set(refs.names[ref_name]))
        plan.skip("referenced by a live process", f"{name} (pid {','.join(map(str, pids[:3]))})")
        return None
    if ref_name is None and _held(path, refs):
        plan.skip("referenced by a live process", name)
        return None
    if t.special is not None:
        plan.skip("contains a socket/fifo/device", name)
        return None
    return Item(path, name, kind, age / 3600.0, t.bytes, t.inodes,
                "stale: newest mtime inside is older than the bar, no live reference")


def _plan_file_cache(path: str, name: str, now: float, max_age_s: float, refs: LiveRefs,
                     plan: Plan) -> None:
    files: List[Item] = []
    dirs: List[str] = []
    for dirpath, dirnames, filenames in os.walk(path, topdown=False, followlinks=False,
                                                onerror=lambda e: None):
        for nm in filenames + [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]:
            p = os.path.join(dirpath, nm)
            try:
                s = os.lstat(p)
            except OSError:
                continue
            if _is_special(s.st_mode) or now - s.st_mtime < max_age_s or p in refs.paths:
                continue
            files.append(Item(p, name, "cache-file", (now - s.st_mtime) / 3600.0,
                              s.st_blocks * 512, 1, "cache file older than the cache bar"))
        if dirpath != path:
            dirs.append(dirpath)   # bottom-up; removed at apply time only if EMPTY and old
    plan.cache_files[name] = files
    plan.cache_dirs[name] = dirs


def build_plan(root: str, uid: int, keep: Set[str], file_caches: Set[str], child_caches: Set[str],
               min_age_h: float, cache_age_h: float, refs: LiveRefs,
               now: Optional[float] = None) -> Plan:
    now = time.time() if now is None else now
    plan = Plan()
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        try:
            st = os.lstat(path)
        except OSError:
            continue
        if st.st_uid != uid:
            plan.skip("not owned by the uid", name)
            continue
        if name in keep:
            plan.skip("declared KEEP", name)
            continue
        if name in file_caches and stat.S_ISDIR(st.st_mode):
            _plan_file_cache(path, name, now, cache_age_h * 3600.0, refs, plan)
            continue
        if name in child_caches and stat.S_ISDIR(st.st_mode):
            for child in sorted(os.listdir(path)):
                cp = os.path.join(path, child)
                if os.path.islink(cp) and child.endswith("current"):
                    continue   # pytest's own `pytest-current` pointer
                # a live pytest names the cache ROOT; its basetemp is held by exact PATH
                it = _judge_whole(cp, f"{name}/{child}", now, cache_age_h * 3600.0, refs, None,
                                  plan, "cache-child")
                if it is not None:
                    plan.delete.append(it)
            continue
        if stat.S_ISLNK(st.st_mode):
            age = now - st.st_mtime
            if age < min_age_h * 3600.0:
                plan.skip("too new (newest mtime inside)", name)
            elif name in refs.names:
                plan.skip("referenced by a live process", name)
            else:
                plan.delete.append(Item(path, name, "entry", age / 3600.0, 0, 1,
                                        "stale symlink (the LINK is removed, never its target)"))
            continue
        it = _judge_whole(path, name, now, min_age_h * 3600.0, refs, name, plan, "entry")
        if it is not None:
            plan.delete.append(it)
    return plan


# --------------------------------------------------------------------------- apply ----

def _rm(path: str) -> None:
    st = os.lstat(path)
    if stat.S_ISDIR(st.st_mode):
        shutil.rmtree(path)          # never descends into a symlink
    else:
        os.unlink(path)              # a file, or a symlink itself — never its target


def apply_plan(plan: Plan, root: str, min_age_h: float, cache_age_h: float,
               proc: str = _PROC) -> Tuple[int, int, List[str]]:
    """Delete what the plan names, re-checking each item. Returns (bytes, inodes, errors)."""
    refs = scan_live_refs(root, proc)      # fresh: something may have started since the plan
    now = time.time()
    freed_b = freed_i = 0
    errors: List[str] = []
    for it in plan.delete:
        parent = os.path.realpath(os.path.dirname(it.path))
        if parent != root and not parent.startswith(root + "/"):
            errors.append(f"{it.path}: outside the root — refused")
            continue
        bar = (cache_age_h if it.kind == "cache-child" else min_age_h) * 3600.0
        top = it.name.split("/", 1)[0]
        try:
            if it.kind == "entry" and os.path.islink(it.path):
                if now - os.lstat(it.path).st_mtime < bar or top in refs.names:
                    continue
            else:
                t = walk(it.path)
                if now - t.newest_mtime < bar or t.special is not None:
                    continue
                if it.kind == "entry" and top in refs.names:
                    continue
                if it.kind == "cache-child" and _held(it.path, refs):
                    continue
            _rm(it.path)
            freed_b += it.bytes
            freed_i += it.inodes
        except OSError as e:
            errors.append(f"{it.path}: {e}")
    for name, files in plan.cache_files.items():
        # dir mtimes BEFORE any unlink: removing a file bumps its parent's mtime to now, so an
        # age check after the fact would never see a dir this sweep emptied as old
        dir_mtime: Dict[str, float] = {}
        for d in plan.cache_dirs.get(name, []):
            try:
                dir_mtime[d] = os.lstat(d).st_mtime
            except OSError:
                pass
        for f in files:
            try:
                s = os.lstat(f.path)
                if now - s.st_mtime < cache_age_h * 3600.0 or f.path in refs.paths:
                    continue
                os.unlink(f.path)
                freed_b += s.st_blocks * 512
                freed_i += 1
            except OSError as e:
                if not isinstance(e, FileNotFoundError):
                    errors.append(f"{f.path}: {e}")
        for d in plan.cache_dirs.get(name, []):
            try:
                if d not in dir_mtime or now - dir_mtime[d] < cache_age_h * 3600.0 or os.listdir(d):
                    continue
                os.rmdir(d)
                freed_i += 1
            except OSError:
                pass   # not empty after all, or already gone: a cache dir staying is harmless
    return freed_b, freed_i, errors


# -------------------------------------------------------------------------- report ----

def fs_usage(root: str) -> Tuple[float, float, str]:
    """(bytes %, inodes %, a df / df -i style line)."""
    v = os.statvfs(root)
    total_b = v.f_blocks * v.f_frsize
    used_b = (v.f_blocks - v.f_bfree) * v.f_frsize
    used_i = v.f_files - v.f_ffree
    pb = 100.0 * used_b / total_b if total_b else 0.0
    pi = 100.0 * used_i / v.f_files if v.f_files else 0.0
    line = (f"{root}: bytes {_gb(used_b)} / {_gb(total_b)} used ({pb:.0f}%)  ·  "
            f"inodes {used_i:,} / {v.f_files:,} used ({pi:.0f}%)")
    return pb, pi, line


def _gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


def warn_lines(root: str, warn_pct: float) -> List[str]:
    pb, pi, _ = fs_usage(root)
    out = []
    if pb > warn_pct:
        out.append(f"WARNING: {root} BYTES at {pb:.0f}% (> {warn_pct:.0f}%)")
    if pi > warn_pct:
        out.append(f"WARNING: {root} INODES at {pi:.0f}% (> {warn_pct:.0f}%) — "
                   "a tmpfs inode table is fixed; ENOSPC arrives with bytes still free")
    return out


def render(plan: Plan, root: str, apply: bool, min_age_h: float, cache_age_h: float) -> List[str]:
    L: List[str] = []
    L.append(f"tmp_sweep {'APPLY' if apply else 'DRY RUN'} over {root} · entry bar {min_age_h:g} h "
             f"· cache bar {cache_age_h:g} h")
    L.append(fs_usage(root)[2])
    items = sorted(plan.delete, key=lambda i: (-i.inodes, -i.bytes))
    tb = sum(i.bytes for i in items)
    ti = sum(i.inodes for i in items)
    L.append(f"\nwhole entries to delete: {len(items)} · {_gb(tb)} · {ti:,} inodes")
    for it in items[:40]:
        L.append(f"  {it.age_h:7.1f} h  {_gb(it.bytes):>10}  {it.inodes:>8,} inodes  {it.name}")
    if len(items) > 40:
        L.append(f"  … {len(items) - 40} more (sorted by inodes, then bytes)")
    cb = ci = 0
    for name, files in sorted(plan.cache_files.items()):
        b = sum(f.bytes for f in files)
        cb += b
        ci += len(files)
        L.append(f"cache {name}: {len(files):,} files older than {cache_age_h:g} h · {_gb(b)} "
                 f"(+ the dirs that leaves empty)")
    L.append(f"\nTOTAL {'freed' if apply else 'would free'}: {_gb(tb + cb)} · "
             f"{ti + ci:,} inodes (before empty-dir removal)")
    if plan.skipped:
        L.append("skipped:")
        for why, n in sorted(plan.skipped.items(), key=lambda kv: -kv[1]):
            L.append(f"  {n:6d}  {why}  e.g. {', '.join(plan.skipped_examples[why])}")
    return L


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=tempfile.gettempdir(), help="the directory to sweep (default: the temp dir)")
    ap.add_argument("--apply", action="store_true", help="delete; without it nothing is touched")
    ap.add_argument("--min-age-hours", type=float, default=MIN_AGE_HOURS)
    ap.add_argument("--cache-max-age-hours", type=float, default=CACHE_MAX_AGE_HOURS)
    ap.add_argument("--uid", type=int, default=os.getuid())
    ap.add_argument("--keep", action="append", default=[], help="a top-level name never touched (repeatable)")
    ap.add_argument("--cache", action="append", default=[],
                    help="a top-level cache dir to FILE-prune at the cache bar (repeatable)")
    ap.add_argument("--warn-pct", type=float, default=WARN_PCT)
    ap.add_argument("--proc", default=_PROC, help=argparse.SUPPRESS)
    a = ap.parse_args(list(argv) if argv is not None else None)

    root = os.path.realpath(a.root)
    if not os.path.isdir(root):
        print(f"tmp_sweep: no such directory {root} — nothing swept")
        return 0
    try:
        refs = scan_live_refs(root, a.proc)
        plan = build_plan(root, a.uid, default_keep(a.uid) | set(a.keep),
                          default_file_caches() | set(a.cache), default_child_caches(),
                          a.min_age_hours, a.cache_max_age_hours, refs)
        for line in render(plan, root, a.apply, a.min_age_hours, a.cache_max_age_hours):
            print(line)
        if refs.unreadable_pids:
            print(f"note: {refs.unreadable_pids} processes unreadable (another uid); their references are unknown")
        if a.apply:
            fb, fi, errs = apply_plan(plan, root, a.min_age_hours, a.cache_max_age_hours, a.proc)
            print(f"APPLIED: freed {_gb(fb)} · {fi:,} inodes")
            for e in errs[:20]:
                print(f"  error: {e}")
            print(fs_usage(root)[2])
        else:
            print("NOTHING WAS DELETED — dry run; --apply deletes.")
    except Exception as e:  # noqa: BLE001 — a cron sweep reports and exits 0
        print(f"tmp_sweep: ERROR {e.__class__.__name__}: {e}")
    for w in warn_lines(root, a.warn_pct):
        print(w)
        print(w, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
