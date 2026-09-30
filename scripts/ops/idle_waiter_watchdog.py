#!/usr/bin/env python3
"""idle_waiter_watchdog — flag agent WAITERS that are making no progress, and lock SELF-DEADLOCKS.

USAGE
    python3 scripts/ops/idle_waiter_watchdog.py            # the cron form: silent when all is well
    python3 scripts/ops/idle_waiter_watchdog.py --help

Run every 15 minutes by the orchestrator's cron. It prints NOTHING and exits 0 when all is well; it
prints one line per flagged process and exits 1 otherwise (2 on an internal error). It never kills,
signals or writes anything but its own state file. It is a DETECTOR — the layer below the hourly
LLM health check (``designs/ops/ORCHESTRATOR_SOP.md`` §7), built after background waits failed
silently for hours on 2026-09-29/30:

* seven ``while pgrep -f PAT; do sleep N; done`` waiters matched their own ``bash -c`` wrapper and
  looped forever (now also refused up front by ``~/.claude/hooks/self_match_guard.py``);
* ``flock gpu.lock python -m main.rust_core_m5 gates --gpu`` waited 15 min at 0% CPU on a lock its
  own ancestor held, because ``gates.py`` takes that lock itself (class fix: ``utils.gpu_lock``).

WHAT IS FLAGGED
1. **SELF-DEADLOCK** — any process BLOCKED on a lock (``/proc/locks`` ``->`` line) whose holder is
   one of its OWN ancestors. It can never resolve, so it is reported on the FIRST run, at any age.
2. **IDLE-WAITER** — a process descended from a Claude Code session (nearest ancestor whose comm or
   exe is ``claude``, incl. ``~/.local/share/claude/versions/<v>``) that
     * is a WAITER: a POLL LOOP (a shell with a live ``sleep`` child; a shell whose own command
       line is a ``while``/``until`` loop that sleeps; or a shell seen with a ``sleep`` child at the
       previous run), a ``flock`` process, or a process blocked in a lock wait (``/proc/locks`` or
       its ``wchan``). The live ``sleep`` child alone is NOT enough: between two sleeps the shell's
       child is its loop CONDITION (``pgrep``, ``test``, a probe script), and a scan that lands
       there would miss the loop — with a probe as slow as its sleep, half the time;
     * has lived longer than ``--min-age-s`` (10 min);
     * whose SUBTREE CPU (utime+stime+cutime+cstime of it and every live descendant) advanced by
       less than ``--idle-frac`` (1%) of one core since the previous run — a fork-per-poll loop
       burns a little CPU forever, and that is not progress; and
     * whose WAIT TARGETS are not progressing either. Targets are read from the subtree's command
       lines and locks: pids (``kill -0 N``, ``--pid N``, ``/proc/N``, ``ps -p N``, ``wait N``),
       ``pgrep``/``ps | grep`` patterns (matched against every OTHER process — a pattern that now
       matches only the waiter's own tree is the self-match class), the holder of a lock it is
       blocked on, and files it names (written since the last run, or named by a live process
       that is progressing). Progress is TRANSITIVE: a waiter on a pid that is itself blocked on a
       lock whose holder is busy is healthy.
   A shell whose live ``sleep`` child is the SAME process as at the last run is one long deliberate
   sleep, not a poll loop, and is skipped.
3. **DUPLICATE-WAITER** — two or more WAITERS (as in 2) under the SAME Claude session with the same
   NORMALISED command line (whitespace collapsed; the Bash tool's per-call ``pwd -P >|
   /tmp/claude-<hex>-cwd`` suffix erased), each older than ``--dup-min-age-s`` (2 min). A forked
   subshell shares its parent's command line, so only the topmost of an ancestor chain counts.
   Reported on the FIRST run with the count and every pid, whether or not the target is progressing:
   on 2026-09-30 Lane K piled up FIVE identical ``until grep -q … <log>`` loops — each blocking wait
   the Bash tool backgrounded at its 600 s timeout kept looping and every retry added one — and none
   was idle, because the target (a job queued on the GPU lock behind a live holder) was legitimately
   waiting. Never kills: which one to keep is the session's call.

STATE: ``~/.claude/jobs/idle_waiter_watchdog.json`` (``--state``), keyed by ``pid:starttime`` so a
reused PID is a new process. The first run only records a baseline (and reports self-deadlocks);
a run less than ``--min-interval-s`` after the baseline judges nothing and keeps the old baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# This checkout's src/ ships beside the script (scripts/ops/ -> ../../src); utils.procfs is stdlib-only.
_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from utils import procfs as P  # noqa: E402

MIN_AGE_S = 600.0
DUP_MIN_AGE_S = 120.0
MIN_INTERVAL_S = 300.0
IDLE_FRAC = 0.01
STATE_PATH = Path.home() / ".claude" / "jobs" / "idle_waiter_watchdog.json"
CMD_WIDTH = 160

SHELLS = frozenset({"bash", "sh", "dash", "zsh", "ksh"})
# A shell whose OWN command line is a poll loop: ``while``/``until`` ... ``do`` ... ``sleep``.
_POLL_LOOP = re.compile(r"\b(?:while|until)\b.*?\bdo\b.*?\bsleep\b", re.S)
_LOCK_WCHAN = re.compile(r"flock|locks_|posix_lock|fcntl|lease")
_PID_TARGETS = [re.compile(r"\bkill\s+-0\s+(\d+)"), re.compile(r"--pid[= ](\d+)"),
                re.compile(r"/proc/(\d+)(?:/|\b)"), re.compile(r"\bps\s+(?:-\w+\s+)*-p\s*(\d+)"),
                re.compile(r"(?<![-\w])wait\s+(\d+)")]
_QUOTED = r"(?:'([^']*)'|\"([^\"]*)\"|([^\s;|&)]+))"
_PGREP = re.compile(r"\bpgrep\s+((?:-\S+\s+)*)" + _QUOTED)
_PSGREP = re.compile(r"\bps\b[^|;]*\|\s*grep\s+(?:-\S+\s+)*" + _QUOTED)
_PATH = re.compile(r"(?<![\w.$/-])((?:~|/)[\w.@+/-]*\w)")
_GENERIC_DIRS = frozenset({"/", "/tmp", "/var/tmp", "/home", "/usr", "/proc", "/dev", str(Path.home())})
_SKIP_PREFIXES = ("/proc/", "/dev/", "/sys/", "/usr/", "/bin/", "/lib", "/etc/")
# The Bash tool's wrapper ends every command with ``pwd -P >| /tmp/claude-<hex>-cwd``: the one token
# that differs between two otherwise identical calls.
_CWD_FILE = re.compile(r"/tmp/claude-[0-9a-f]+-cwd\b")


def normalise_cmd(cmd: str) -> str:
    """The command line with per-call noise removed, for duplicate detection."""
    return " ".join(_CWD_FILE.sub("<cwd-file>", cmd).split())


@dataclass
class Finding:
    kind: str           # SELF-DEADLOCK / IDLE-WAITER / DUPLICATE-WAITER
    pid: int
    age_s: float
    session: Optional[int]
    session_cwd: str
    cmd: str
    reason: str

    def line(self) -> str:
        h, m = divmod(int(self.age_s) // 60, 60)
        sess = f"{self.session} (cwd {self.session_cwd or '?'})" if self.session else "none"
        cmd = self.cmd if len(self.cmd) <= CMD_WIDTH else self.cmd[:CMD_WIDTH - 3] + "..."
        return f"[idle-waiter-watchdog] {self.kind} pid={self.pid} age={h}h{m:02d}m session={sess} cmd={cmd!r} reason: {self.reason}"


def _is_claude(p: P.Proc, exe: str) -> bool:
    return p.comm == "claude" or os.path.basename(exe) == "claude" or "/claude/versions/" in exe


class Scan:
    """One read of the box, and the judgments over it. Separate from ``main`` so a test can run two
    scans against real processes with its own thresholds and state."""

    def __init__(self, state: Optional[dict], *, now: Optional[float] = None, min_age_s: float = MIN_AGE_S,
                 min_interval_s: float = MIN_INTERVAL_S, idle_frac: float = IDLE_FRAC,
                 scope_pid: Optional[int] = None, self_pid: Optional[int] = None,
                 dup_min_age_s: float = DUP_MIN_AGE_S) -> None:
        self.now = time.time() if now is None else now
        self.uptime = P.uptime_s()
        self.procs = P.snapshot()
        self.kids = P.children_map(self.procs)
        self.cpu = P.subtree_cpu(self.procs, self.kids)
        self.locks = P.read_locks()
        self.min_age_s, self.min_interval_s, self.idle_frac = min_age_s, min_interval_s, idle_frac
        self.dup_min_age_s = dup_min_age_s
        me = os.getpid() if self_pid is None else self_pid
        self.self_chain: Set[int] = {me, *P.ancestors(me, self.procs)}   # the watchdog and its cron chain
        self.claude = {pid for pid, p in self.procs.items() if _is_claude(p, P.readlink(pid, "exe"))}
        self.scope: Optional[Set[int]] = None
        if scope_pid is not None:
            self.scope = {scope_pid, *P.descendants(scope_pid, self.kids)}
        self.prev = state if _valid_state(state) else None
        self.judge = self.prev is not None and self.now - self.prev["t"] >= self.min_interval_s
        self._cmd: Dict[int, str] = {}
        self._holders: Dict[str, Set[int]] = {}
        self._blocked_on: Dict[int, List[str]] = {}
        for e in self.locks:
            if e.blocked:
                self._blocked_on.setdefault(e.pid, []).append(e.file)
            elif e.pid > 0:
                self._holders.setdefault(e.file, set()).add(e.pid)
        self.sleep_child: Dict[str, str] = {}

    # ------------------------------------------------------------------------------- primitives
    def cmd(self, pid: int) -> str:
        if pid not in self._cmd:
            self._cmd[pid] = (P.cmdline(pid) or f"[{self.procs[pid].comm}]") if pid in self.procs else ""
        return self._cmd[pid]

    def age(self, pid: int) -> float:
        return P.age_s(self.procs[pid], self.uptime)

    def session_of(self, pid: int) -> Optional[int]:
        for a in P.ancestors(pid, self.procs):
            if a in self.claude:
                return a
        return None

    def in_scope(self, pid: int) -> bool:
        return self.scope is None or pid in self.scope

    def advanced(self, pid: int) -> Optional[bool]:
        """Did ``pid``'s subtree CPU advance past the idle bar since the last run? ``None`` when
        there is no baseline to judge by (no previous run, or the process is newer than it)."""
        if self.prev is None or pid not in self.procs:
            return None
        before = self.prev["cpu"].get(self.procs[pid].key)
        if before is None:
            return None
        dt = self.now - self.prev["t"]
        return (self.cpu[pid] - before) / P.HZ >= self.idle_frac * dt

    def lock_name(self, pid: int, key: str) -> str:
        return P.fd_path_for(pid, key) or f"inode {key}"

    # ------------------------------------------------------------------------------- 1. deadlocks
    def self_deadlocks(self) -> List[Finding]:
        out = []
        for pid, keys in self._blocked_on.items():
            if pid not in self.procs or not self.in_scope(pid) or pid in self.self_chain:
                continue
            anc = P.ancestors(pid, self.procs)
            for key in keys:
                held = [h for h in sorted(self._holders.get(key, ())) if h in anc]
                if held:
                    h = held[0]
                    out.append(self._finding("SELF-DEADLOCK", pid, (
                        f"blocked on {self.lock_name(pid, key)}, held by pid {h} — its OWN ancestor "
                        f"({self.cmd(h)[:80]!r}); this can never resolve")))
                    break
        return out

    # ------------------------------------------------------------------------------- 2. idle waiters
    def waiter_kind(self, pid: int) -> str:
        p = self.procs[pid]
        if pid in self._blocked_on:
            return "lock-wait"
        if p.comm == "flock":
            return "flock"
        if p.comm in SHELLS:
            for c in self.kids.get(pid, ()):
                if c in self.procs and self.procs[c].comm == "sleep":
                    self.sleep_child[p.key] = self.procs[c].key
                    return "sleep-loop"
            # No live sleep child: the loop may be between sleeps, running its condition. Judge it
            # by what does not change from one instant to the next — its own loop text, or a sleep
            # child OBSERVED at the previous run (a real key, not this "" marker, so the memory
            # lasts one run unless re-observed) — never by which child the snapshot caught.
            if _POLL_LOOP.search(self.cmd(pid)) or (self.prev is not None and self.prev.get("sleep", {}).get(p.key)):
                self.sleep_child[p.key] = ""        # a known poll loop, with no sleep in flight now
                return "sleep-loop"
        if _LOCK_WCHAN.search(P.wchan(pid)):
            return "lock-wait"
        return ""

    def _tree(self, pid: int) -> List[int]:
        return [pid, *P.descendants(pid, self.kids)]

    def progress(self, pid: int, depth: int = 0, seen: Optional[Set[int]] = None) -> Tuple[bool, str]:
        """Is ``pid`` — or anything it is waiting on — making progress? Returns the verdict and,
        when it is not, the evidence in words."""
        seen = set() if seen is None else seen
        seen.add(pid)
        adv = self.advanced(pid)
        if adv is None or adv:
            return True, ""
        if depth > 4:
            return False, "idle (target chain too deep to follow)"
        tree = self._tree(pid)
        mine = set(tree) | set(P.ancestors(pid, self.procs)) | self.self_chain
        notes: List[str] = []
        named_any = False
        cmds = [self.cmd(q) for q in tree]
        # -- pids named on a command line
        for n in sorted({int(m) for c in cmds for rx in _PID_TARGETS for m in rx.findall(c)} - mine):
            named_any = True
            if n not in self.procs:
                notes.append(f"waits on pid {n}, which is GONE")
                continue
            if n in seen:
                continue
            ok, why = self.progress(n, depth + 1, seen)
            if ok:
                return True, ""
            notes.append(f"waits on pid {n} ({self.cmd(n)[:60]!r}), also idle")
        # -- pgrep / ps|grep patterns, matched against every process OUTSIDE the waiter's own tree
        for full, pat in self._patterns(cmds):
            named_any = True
            try:
                rx = re.compile(pat)
            except re.error:
                rx = re.compile(re.escape(pat))
            hits = [q for q in self.procs if q not in mine
                    and rx.search(self.cmd(q) if full else self.procs[q].comm)]
            if not hits:
                notes.append(f"its pattern {pat!r} now matches only its OWN process tree (self-match: target gone)")
                continue
            for q in hits:
                if q not in seen and self.progress(q, depth + 1, seen)[0]:
                    return True, ""
            notes.append(f"its pattern {pat!r} matches pid(s) {hits[:4]}, all idle")
        # -- the holder of any lock the tree is blocked on
        for q in tree:
            for key in self._blocked_on.get(q, ()):
                named_any = True
                hs = sorted(self._holders.get(key, ()))
                if not hs:
                    notes.append(f"blocked on {self.lock_name(q, key)} with no live holder listed")
                for h in hs:
                    if h in seen:
                        continue
                    if self.progress(h, depth + 1, seen)[0]:
                        return True, ""
                    notes.append(f"blocked on {self.lock_name(q, key)}, held by pid {h} ({self.cmd(h)[:60]!r}), also idle")
        # -- files it names: written since the last run, or named by a live process that progresses
        for path in self._paths(cmds):
            named_any = True
            try:
                mtime = os.stat(path).st_mtime
            except OSError:
                mtime = None
            if mtime is not None and self.prev is not None and mtime >= self.prev["t"]:
                return True, ""
            for q in self._naming(path, mine):
                if q not in seen and self.progress(q, depth + 1, seen)[0]:
                    return True, ""
            notes.append(f"names {path} ({'absent' if mtime is None else 'unchanged since last run'}, no progressing process names it)")
        if not named_any:
            notes.append("names no pid, pattern, lock or file it could be waiting on")
        return False, "; ".join(notes)

    @staticmethod
    def _patterns(cmds: Iterable[str]) -> List[Tuple[bool, str]]:
        out = []
        for c in cmds:
            for m in _PGREP.finditer(c):
                flags = m.group(1) or ""
                full = bool(re.search(r"(^|\s)-\w*f|--full", flags))
                out.append((full, next(g for g in m.groups()[1:] if g is not None)))
            for m in _PSGREP.finditer(c):
                out.append((True, next(g for g in m.groups() if g is not None)))
        return list(dict.fromkeys(out))

    @staticmethod
    def _paths(cmds: Iterable[str]) -> List[str]:
        out = []
        for c in cmds:
            for raw in _PATH.findall(c):
                path = os.path.expanduser(raw)
                if path in _GENERIC_DIRS or path.startswith(_SKIP_PREFIXES) or path.startswith(str(_SRC)):
                    continue
                if os.path.isfile(path) and os.access(path, os.X_OK):
                    continue            # a program being run, not a thing being waited on
                out.append(path)
        return list(dict.fromkeys(out))

    def _naming(self, path: str, mine: Set[int]) -> List[int]:
        """Live processes outside the waiter whose command line names the file, or (for a
        non-generic directory) the directory it lives in — the job that will write it."""
        parent = os.path.dirname(path)
        needles = [path] + ([parent] if parent not in _GENERIC_DIRS else [])
        return [q for q in self.procs if q not in mine and q not in self.claude
                and any(n in self.cmd(q) for n in needles)]

    def idle_waiters(self, skip: Set[int]) -> List[Finding]:
        found: Dict[int, Finding] = {}
        for pid in self.procs:
            if pid in self.claude or pid in self.self_chain or pid in skip or not self.in_scope(pid):
                continue
            kind = self.waiter_kind(pid)
            if not kind or self.session_of(pid) is None or self.age(pid) < self.min_age_s or not self.judge:
                continue
            assert self.prev is not None
            key = self.procs[pid].key
            now_sleep = self.sleep_child.get(key)
            if kind == "sleep-loop" and now_sleep and self.prev.get("sleep", {}).get(key) == now_sleep:
                continue                # the SAME sleep as last run: one long deliberate delay
            ok, why = self.progress(pid)
            if ok:
                continue
            idle_min = (self.now - self.prev["t"]) / 60
            found[pid] = self._finding("IDLE-WAITER", pid, f"{kind}, subtree CPU flat for {idle_min:.0f} min; {why}")
        # report the deepest waiter only: a flock whose child loop is flagged — or whose child is the
        # self-deadlocked waiter — is the same stall, already on the page
        return [f for pid, f in found.items()
                if not any(d in found or d in skip for d in P.descendants(pid, self.kids))]

    # ------------------------------------------------------------------------------- 3. duplicates
    def duplicate_waiters(self) -> List[Finding]:
        groups: Dict[Tuple[int, str], List[int]] = {}
        for pid in self.procs:
            if pid in self.claude or pid in self.self_chain or not self.in_scope(pid):
                continue
            if self.age(pid) < self.dup_min_age_s or not self.waiter_kind(pid):
                continue
            s = self.session_of(pid)
            if s is None:
                continue
            groups.setdefault((s, normalise_cmd(self.cmd(pid))), []).append(pid)
        out = []
        for pids in groups.values():
            members = set(pids)
            top = sorted((q for q in pids if not members & set(P.ancestors(q, self.procs))),
                         key=lambda q: -self.age(q))           # oldest first
            if len(top) < 2:
                continue
            ages = ", ".join(f"{q} ({int(self.age(q)) // 60}m)" for q in top)
            out.append(self._finding("DUPLICATE-WAITER", top[0], (
                f"{len(top)} identical waiters in this session: pids {ages} — a retried wait adds "
                f"one each time (a Bash-tool timeout backgrounds the old loop, it does not end it); "
                f"keep ONE and kill the rest by explicit PID")))
        return out

    def _finding(self, kind: str, pid: int, reason: str) -> Finding:
        s = self.session_of(pid)
        return Finding(kind=kind, pid=pid, age_s=self.age(pid), session=s,
                       session_cwd=P.readlink(s, "cwd") if s else "", cmd=self.cmd(pid), reason=reason)

    def run(self) -> List[Finding]:
        dead = self.self_deadlocks()
        return dead + self.idle_waiters({f.pid for f in dead}) + self.duplicate_waiters()

    def next_state(self) -> dict:
        cpu = {self.procs[pid].key: self.cpu[pid] for pid in self.procs}
        if self.prev is not None and not self.judge:
            # Too soon to judge: keep the old baseline (and its time) for every process it knew.
            cpu.update({k: v for k, v in self.prev["cpu"].items() if k in cpu})
            sleep = {**self.sleep_child, **{k: v for k, v in self.prev.get("sleep", {}).items() if k in cpu}}
            return {"version": 1, "t": self.prev["t"], "cpu": cpu, "sleep": sleep}
        return {"version": 1, "t": self.now, "cpu": cpu, "sleep": dict(self.sleep_child)}


def _valid_state(s: object) -> bool:
    return (isinstance(s, dict) and s.get("version") == 1 and isinstance(s.get("t"), (int, float))
            and isinstance(s.get("cpu"), dict))


def load_state(path: Path) -> Optional[dict]:
    try:
        s = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return s if _valid_state(s) else None


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(state, separators=(",", ":")))
    os.replace(tmp, path)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="idle_waiter_watchdog.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", type=Path, default=STATE_PATH, help=f"state file (default {STATE_PATH})")
    ap.add_argument("--min-age-s", type=float, default=MIN_AGE_S, help="flag only waiters older than this")
    ap.add_argument("--dup-min-age-s", type=float, default=DUP_MIN_AGE_S,
                    help="flag duplicate waiters only once at least two are older than this")
    ap.add_argument("--min-interval-s", type=float, default=MIN_INTERVAL_S,
                    help="judge idleness only against a baseline at least this old")
    ap.add_argument("--idle-frac", type=float, default=IDLE_FRAC,
                    help="subtree CPU below this fraction of one core over the interval is idle")
    ap.add_argument("--scope-pid", type=int, default=None, help="consider only this pid's subtree (tests)")
    ap.add_argument("--timing", action="store_true", help="print the run's wall time to stderr")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    t0 = time.perf_counter()
    try:
        scan = Scan(load_state(a.state), min_age_s=a.min_age_s, min_interval_s=a.min_interval_s,
                    idle_frac=a.idle_frac, scope_pid=a.scope_pid, dup_min_age_s=a.dup_min_age_s)
        findings = scan.run()
        save_state(a.state, scan.next_state())
    except Exception as e:  # a watchdog that dies quietly is the failure it exists to catch
        print(f"[idle-waiter-watchdog] ERROR: {type(e).__name__}: {e}", file=sys.stdout)
        return 2
    for f in findings:
        print(f.line())
    if a.timing:
        print(f"[idle-waiter-watchdog] {len(scan.procs)} processes in {time.perf_counter() - t0:.3f} s",
              file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
