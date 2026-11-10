"""THE CORE SPLIT: the ONE owner of which CPUs the trainer gets and which the agents get.

Design, measurement and the A/B that decides a change: ``designs/ops/core_split.md``. The box is one 8-core, 16-thread
CPU where CPU n and n + 8 are SMT siblings, and a split is always made of WHOLE physical cores: the trainer takes
physical cores ``0 .. T-1`` (both threads of each), agent work takes the rest (root ``CLAUDE.md`` rule 5b). A split is
named ``T+A`` (:data:`SPLITS`); :data:`DEFAULT_SPLIT` is ``5+3``.

No other file may spell either list (``core_split_literal_gate_test.py`` fails on one). Code reads :func:`agent_cpus` /
:func:`trainer_cpus` AT THE MOMENT IT LAUNCHES a job, so a long-lived supervisor follows a change without a restart;
a shell script reads the CLI::

    python -m utils.core_split show                      # the active split, a pending one, the live trainers
    python -m utils.core_split agent-cpus                # the agent list, taskset's form (the default split's is 5-7,13-15)
    python -m utils.core_split trainer-cpus              # the trainer list (the default split's is 0-4,8-12)
    python -m utils.core_split trainer-cpus --launch     # A LAUNCH KIT's read: applies a pending split first
    python -m utils.core_split set 4+4 --next-launch     # queue a split for the next launch
    python -m utils.core_split set 5+3                   # set it now; REFUSED while a trainer is live

THE STATE is one JSON file, :func:`state_path` (``~/.local/state/gen3ai/core_split.json``, or the file
``$GEN3AI_CORE_SPLIT_STATE`` names). No file means the default split. A file that cannot be parsed is an ERROR, never
a silent default: two readers that disagree about the split put agent work on the trainer's cores.

THE ORDER (``core_split.md`` §7) is held by two refusals, not by care:

* ``set`` without ``--next-launch`` is refused while any launcher or trainer process is live: a live tree keeps the
  CPUs it was started on, so widening the agents under it lands them on its cores. (A CPU-only run an agent started
  on the agent cores is agent work, not a trainer to wait for: :func:`blocking_trainers`.)
* ``set --next-launch`` only records the split. It becomes active when a launch kit reads its pin with
  ``trainer-cpus --launch`` while no trainer is live: the new tree starts narrow and the agents widen with the same
  write. With a trainer still live that read changes nothing and prints the active list.

:func:`agent_ids_beside` is the rule for a job that reads the live trainer's own affinity (the ship gate): the agents'
CPUs are the online CPUs in neither the trainer's affinity nor the active trainer list, so a trainer narrowed for a
few seconds by the A/B driver (``main.ops.core_split_ab``) frees nothing.

Stdlib and ``utils.procfs`` only. Tests: ``core_split_test.py``, ``core_split_literal_gate_test.py``.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import pwd
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from utils import procfs

#: This box: 8 physical cores, CPU n and n + PHYSICAL_CORES are SMT siblings.
PHYSICAL_CORES = 8
#: The splits ``set`` accepts, trainer physical cores + agent physical cores. Fewer than 3 trainer cores stretches a
#: rollout's worker burst 2x (core_split.md §9); more than 5 was never run.
SPLITS = ("5+3", "4+4", "3+5")
DEFAULT_SPLIT = "5+3"
#: The file that names the state file instead of :func:`state_path`'s default (the test suite seals it: ``conftest.py``).
STATE_ENV = "GEN3AI_CORE_SPLIT_STATE"
SCHEMA = "core_split_v1"
#: The A/B driver's in-progress marker, beside the state file (``main.ops.core_split_ab`` writes and removes it).
AB_MARKER = "core_split_ab.json"
_LOG_KEEP = 20
_TRAINER_SCRIPT = "train_rl_agent" + ".py"

EXIT_OK, EXIT_USAGE, EXIT_REFUSED = 0, 2, 3


class CoreSplitError(RuntimeError):
    """The state is unreadable or a split is not one of :data:`SPLITS`."""


def format_cpus(cpus: Iterable[int]) -> str:
    """CPU ids as ``taskset -c``'s list: ``[0, 2, 3]`` -> ``"0,2-3"``; ``[]`` -> ``""``."""
    out: List[str] = []
    xs = sorted(set(cpus))
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(str(xs[i]) if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ",".join(out)


def parse_cpus(text: str) -> List[int]:
    """``taskset -c``'s list as sorted CPU ids: ``"0-2,5"`` -> ``[0, 1, 2, 5]``. A malformed list raises ValueError."""
    out = set()
    for part in text.strip().split(","):
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        a, b = int(lo), int(hi) if sep else int(lo)
        if b < a:
            raise ValueError(f"cpu list {text!r}: {part!r} runs backwards")
        out.update(range(a, b + 1))
    return sorted(out)


def physical_core(cpu: int) -> int:
    """The physical core of a logical CPU on this box."""
    return cpu % PHYSICAL_CORES


@dataclass(frozen=True)
class Split:
    """A split by its number of TRAINER physical cores."""

    trainer_cores: int

    def __post_init__(self) -> None:
        if not 1 <= self.trainer_cores < PHYSICAL_CORES:
            raise CoreSplitError(f"a split leaves both sides at least one physical core of {PHYSICAL_CORES}; "
                                 f"got {self.trainer_cores} for the trainer")

    @property
    def name(self) -> str:
        return f"{self.trainer_cores}+{PHYSICAL_CORES - self.trainer_cores}"

    @property
    def trainer_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(c + s for c in range(self.trainer_cores) for s in (0, PHYSICAL_CORES)))

    @property
    def agent_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(c + s for c in range(self.trainer_cores, PHYSICAL_CORES) for s in (0, PHYSICAL_CORES)))

    @property
    def trainer_list(self) -> str:
        return format_cpus(self.trainer_ids)

    @property
    def agent_list(self) -> str:
        return format_cpus(self.agent_ids)


def parse_split(name: str) -> Split:
    """A :data:`SPLITS` name as a :class:`Split`; anything else raises :class:`CoreSplitError`."""
    if name not in SPLITS:
        raise CoreSplitError(f"split {name!r} is not one of {', '.join(SPLITS)} (trainer + agent physical cores)")
    return Split(int(name.split("+")[0]))


DEFAULT = parse_split(DEFAULT_SPLIT)


def narrowed(control: Iterable[int], drop: int) -> List[int]:
    """``control`` (CPU ids made of whole physical cores) less its ``drop`` HIGHEST physical cores: the A/B's arms.
    Raises ValueError when ``control`` holds half a core or the result would be empty."""
    ids = sorted(set(control))
    cores = sorted({physical_core(c) for c in ids})
    if sorted(c + s for c in cores for s in (0, PHYSICAL_CORES)) != ids:
        raise ValueError(f"cpus {format_cpus(ids)} are not whole physical cores (CPU n and n + {PHYSICAL_CORES})")
    if not 0 <= drop < len(cores):
        raise ValueError(f"cannot drop {drop} of {len(cores)} physical cores")
    keep = set(cores[:len(cores) - drop])
    return [c for c in ids if physical_core(c) in keep]


# ---------------------------------------------------------------------------------------------------------------------
# the state file


def state_path(env: Optional[Dict[str, str]] = None) -> Path:
    """The state file: ``$GEN3AI_CORE_SPLIT_STATE``, else ``.local/state/gen3ai/core_split.json`` under the home
    directory of the USER (the password database's, never ``$HOME`` or ``$XDG_STATE_HOME``): every reader on the box
    must name the same file whatever its environment, or two of them disagree about the split in silence."""
    e = os.environ if env is None else env
    named = e.get(STATE_ENV)
    if named:
        return Path(named)
    return Path(pwd.getpwuid(os.getuid()).pw_dir) / ".local" / "state" / "gen3ai" / "core_split.json"


def ab_marker_path(path: Optional[Path] = None) -> Path:
    return (state_path() if path is None else path).parent / AB_MARKER


def read_state(path: Optional[Path] = None) -> Dict[str, Any]:
    """The state: ``{"split": name, "next_launch": name or None, "log": [...]}``. No file is the default split."""
    p = state_path() if path is None else path
    try:
        raw = p.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError):
        return {"schema": SCHEMA, "split": DEFAULT_SPLIT, "next_launch": None, "log": []}
    except OSError as e:
        raise CoreSplitError(f"cannot read the core-split state {p}: {e}") from e
    try:
        d = json.loads(raw)
        if not isinstance(d, dict) or d.get("schema") != SCHEMA:
            raise ValueError(f"not a {SCHEMA} object")
        parse_split(d["split"])
        if d.get("next_launch") is not None:
            parse_split(d["next_launch"])
    except (ValueError, KeyError, TypeError, CoreSplitError) as e:
        raise CoreSplitError(f"the core-split state {p} is unreadable ({e}). Nothing falls back to a default: fix or "
                             f"remove the file (no file = {DEFAULT_SPLIT}), then `python -m utils.core_split show`") from e
    d.setdefault("next_launch", None)
    d.setdefault("log", [])
    return d


def _write_state(p: Path, d: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(d, indent=1) + "\n")
        f.flush()
        os.fsync(f.fileno())                      # a power cut must leave the old file or the new one, never an empty one
    os.replace(tmp, p)


class _Locked:
    """An exclusive flock beside the state file for a read-modify-write."""

    def __init__(self, p: Path) -> None:
        self.p = p
        self.fd = -1

    def __enter__(self) -> "_Locked":
        self.p.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(str(self.p) + ".lock", os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc: Any) -> None:
        os.close(self.fd)


def _logged(d: Dict[str, Any], event: str, split: str, note: str, now: float) -> None:
    d["log"] = (list(d.get("log") or []) + [{"at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
                                              "event": event, "split": split, "note": note}])[-_LOG_KEEP:]


def active(path: Optional[Path] = None) -> Split:
    """The split in force now."""
    return parse_split(read_state(path)["split"])


def agent_cpus() -> str:
    """The agent CPUs of the active split, ``taskset -c``'s form. Read it when a job is launched, never at import."""
    return active().agent_list


def trainer_cpus() -> str:
    """The trainer CPUs of the active split, ``taskset -c``'s form."""
    return active().trainer_list


def agent_cpu_ids() -> Tuple[int, ...]:
    return active().agent_ids


def trainer_cpu_ids() -> Tuple[int, ...]:
    return active().trainer_ids


def agent_ids_beside(trainer_affinity: Iterable[int], online: Iterable[int]) -> List[int]:
    """The CPUs agent work may use beside a live trainer whose affinity was READ: every online CPU that is neither in
    that affinity nor in the active split's trainer list. A trainer pinned WIDER than the split is respected; one
    pinned NARROWER (the A/B driver's 8-second arms) frees nothing."""
    held = set(trainer_affinity) | set(trainer_cpu_ids())
    return [c for c in online if c not in held]


# ---------------------------------------------------------------------------------------------------------------------
# live trainers


def _python_target(argv: Sequence[str]) -> Optional[Tuple[str, str, int]]:
    """What a ``python`` command line RUNS: ``("module" | "script", its name, the index its own arguments start at)``,
    or None (not python, ``-c``, stdin, nothing). Interpreter options are skipped the way CPython reads them: ``-X``
    and ``-W`` take an argument, options cluster (``-um mod``), and ``-m`` may carry its module (``-mmod``)."""
    if not argv or not os.path.basename(argv[0]).startswith("python"):
        return None
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--":
            return ("script", argv[i + 1], i + 2) if i + 1 < len(argv) else None
        if a == "-" or not a.startswith("-"):
            return None if a == "-" else ("script", a, i + 1)
        if a.startswith("--"):
            i += 2 if a == "--check-hash-based-pycs" else 1
            continue
        for j, ch in enumerate(a[1:], start=1):
            rest = a[j + 1:]
            if ch == "c":
                return None
            if ch == "m":
                if rest:
                    return "module", rest, i + 1
                return ("module", argv[i + 1], i + 2) if i + 1 < len(argv) else None
            if ch in "XW":
                i += 0 if rest else 1                 # its argument is the rest of this word, or the next word
                break
        i += 1
    return None


def is_trainer_argv(argv: Sequence[str]) -> Optional[str]:
    """``"launcher"`` / ``"trainer"`` when the argv is a training launcher (one that launches: not ``--dry-run`` or
    ``--help``) or its trainer child. It is decided by what python RUNS, never by a word somewhere in the argv:
    ``grep -m main.launcher``, an editor on the trainer script and ``tail -f`` of it are not trainers."""
    target = _python_target(argv)
    if target is None:
        return None
    kind, name, rest = target
    if kind == "script":                              # `python src/main/launcher[/__main__.py]` runs the package too
        parts = Path(name).parts
        parts = parts[:-1] if parts[-1:] == ("__main__.py",) else parts
        if os.path.basename(name) == _TRAINER_SCRIPT:
            return "trainer"
        name = ".".join(parts[-2:]) if parts[-2:] == ("main", "launcher") else ""
    name = name[:-len(".__main__")] if name.endswith(".__main__") else name
    if name == "main.launcher":
        return None if any(a in ("--dry-run", "--help", "-h") for a in argv[rest:]) else "launcher"
    return "trainer" if name == "main." + _TRAINER_SCRIPT[:-3] else None


def live_trainers(proc: str = "/proc") -> Dict[int, str]:
    """``{pid: "launcher" | "trainer"}`` of every live training launcher and trainer child. A launcher counts with no
    child (the seconds of a restart): the child it starts next inherits the launcher's CPUs."""
    out: Dict[int, str] = {}
    for pid in procfs.all_pids(proc):
        kind = is_trainer_argv([a for a in procfs.read_text(pid, "cmdline", proc).split("\0") if a])
        if kind:
            out[pid] = kind
    return out


def _affinity(pid: int) -> Optional[List[int]]:
    try:
        return sorted(os.sched_getaffinity(pid))
    except OSError:
        return None


def is_agent_work(affinity: Optional[Iterable[int]], *splits: "Split") -> bool:
    """THE rule for "this run is agent work, not the pinned trainer": its CPUs (``affinity``) lie inside the agent
    list of every split given (none given: the active split). A CPU-only run an agent started on the agent cores is
    that. An unreadable (``None``) or empty affinity is never agent work. Read by :func:`blocking_trainers`,
    :func:`status` and the ship gate's shape (``utils.ship_gate.split_runs``)."""
    cpus = set(affinity or ())
    return bool(cpus) and all(cpus <= set(s.agent_ids) for s in (splits or (active(),)))


def blocking_trainers(cur: "Split", new: "Split", proc: str = "/proc",
                      affinity: Callable[[int], Optional[List[int]]] = _affinity) -> Dict[int, str]:
    """The live launchers and trainers a change from ``cur`` to ``new`` must wait for. One whose CPUs lie inside the
    agent list of BOTH splits does not count: it is agent work (:func:`is_agent_work`), and it stays on agent cores.
    An unreadable affinity counts."""
    return {pid: kind for pid, kind in live_trainers(proc).items() if not is_agent_work(affinity(pid), cur, new)}


# ---------------------------------------------------------------------------------------------------------------------
# the writes


class Refused(Exception):
    """A ``set`` the order of core_split.md §7 forbids."""


def set_split(name: str, *, next_launch: bool = False, note: str = "", path: Optional[Path] = None,
              live: Optional[Callable[[], Dict[int, str]]] = None, now: Optional[float] = None) -> str:
    """Set the active split (refused while a trainer is live), or with ``next_launch`` queue it for the next launch.
    Returns the line to print."""
    new = parse_split(name)
    p = state_path() if path is None else path
    now = time.time() if now is None else now
    with _Locked(p):
        d = read_state(p)
        cur = d["split"]
        if next_launch:
            if new.name == cur:
                d["next_launch"] = None
                _logged(d, "next-launch-cleared", new.name, note, now)
                _write_state(p, d)
                return f"core split: {cur} is already active; nothing is queued for the next launch"
            d["next_launch"] = new.name
            _logged(d, "set-next-launch", new.name, note, now)
            _write_state(p, d)
            return (f"core split: {new.name} QUEUED for the next launch (active stays {cur}: trainer "
                    f"{parse_split(cur).trainer_list}, agents {parse_split(cur).agent_list}). It applies when a launch "
                    "kit reads `python -m utils.core_split trainer-cpus --launch` with no trainer live")
        procs = blocking_trainers(parse_split(cur), new) if live is None else live()
        if procs:
            raise Refused(f"REFUSING: a training launcher or trainer is live (pid {', '.join(map(str, sorted(procs)))}) "
                          f"and keeps the CPUs it started on, so the split cannot change under it (active: {cur}). "
                          f"Queue it instead: python -m utils.core_split set {new.name} --next-launch")
        d["split"], d["next_launch"] = new.name, None
        _logged(d, "set", new.name, note, now)
        _write_state(p, d)
        return f"core split: {new.name} ACTIVE (was {cur}): trainer {new.trainer_list}, agents {new.agent_list}"


#: The words a launch kit matches (``launch_x35.sh``): a queued split that a launch did NOT get.
STAYS_QUEUED = "stays QUEUED"


def _stays_queued(new: "Split", cur: "Split", procs: Dict[int, str]) -> str:
    return (f"core split: {new.name} {STAYS_QUEUED}: a trainer is live (pid {', '.join(map(str, sorted(procs)))}); "
            f"this launch gets {cur.name}'s list")


def launch_trainer_cpus(*, dry_run: bool = False, path: Optional[Path] = None,
                        live: Optional[Callable[[], Dict[int, str]]] = None,
                        now: Optional[float] = None) -> Tuple[str, str]:
    """A LAUNCH KIT's read of the trainer list: ``(list, note)``. A queued split becomes active first when no trainer
    is live (``dry_run``: reported, not written); with one live the active list is returned and the queue stands."""
    p = state_path() if path is None else path
    now = time.time() if now is None else now
    d = read_state(p)
    if not d.get("next_launch"):                  # the common read: no lock, nothing created
        return parse_split(d["split"]).trainer_list, ""
    if dry_run:                                   # a dry run writes nothing, so it takes no lock (and creates no file)
        cur, new = parse_split(d["split"]), parse_split(d["next_launch"])
        procs = blocking_trainers(cur, new) if live is None else live()
        if procs:
            return cur.trainer_list, _stays_queued(new, cur, procs)
        return new.trainer_list, f"core split: a launch now would make the queued {new.name} active (was {cur.name})"
    with _Locked(p):
        d = read_state(p)
        cur, queued = parse_split(d["split"]), d.get("next_launch")
        if not queued:
            return cur.trainer_list, ""
        new = parse_split(queued)
        procs = blocking_trainers(cur, new) if live is None else live()
        if procs:
            return cur.trainer_list, _stays_queued(new, cur, procs)
        d["split"], d["next_launch"] = new.name, None
        _logged(d, "applied-at-launch", new.name, "", now)
        _write_state(p, d)
        return new.trainer_list, (f"core split: the queued {new.name} is now ACTIVE (was {cur.name}): trainer "
                                  f"{new.trainer_list}, agents {new.agent_list}")


# ---------------------------------------------------------------------------------------------------------------------
# show


def read_ab_marker(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The A/B driver's in-progress marker, or ``None``. An unreadable marker is returned as ``{"unreadable": ...}``."""
    m = ab_marker_path(path)
    try:
        d = json.loads(m.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {"unreadable": "not an object"}
    except (FileNotFoundError, NotADirectoryError):
        return None
    except (OSError, ValueError) as e:
        return {"unreadable": str(e)}


def status(path: Optional[Path] = None, *, live: Optional[Callable[[], Dict[int, str]]] = None,
           affinity: Callable[[int], Optional[List[int]]] = _affinity) -> Dict[str, Any]:
    p = state_path() if path is None else path
    d = read_state(p)
    cur = parse_split(d["split"])
    trainers = []
    for pid, kind in sorted((live_trainers if live is None else live)().items()):
        aff = affinity(pid)
        on_agents = sorted(set(aff or []) & set(cur.agent_ids))
        trainers.append({"pid": pid, "kind": kind, "cpus": format_cpus(aff) if aff is not None else None,
                         "is_trainer_list": aff == list(cur.trainer_ids), "on_agent_cpus": format_cpus(on_agents),
                         "agent_work": is_agent_work(aff, cur)})
    return {"state_file": str(p), "state_file_exists": p.exists(), "split": cur.name, "trainer_cpus": cur.trainer_list,
            "agent_cpus": cur.agent_list, "next_launch": d.get("next_launch"), "trainers": trainers,
            "ab_marker": read_ab_marker(p), "log": d.get("log") or []}


def render(s: Dict[str, Any]) -> str:
    lines = [f"core split {s['split']}: trainer {s['trainer_cpus']}, agents {s['agent_cpus']}"
             + ("" if s["state_file_exists"] else " (the default: no state file)"),
             f"  state file: {s['state_file']}"]
    if s["next_launch"]:
        q = parse_split(s["next_launch"])
        lines.append(f"  QUEUED for the next launch: {q.name} (trainer {q.trainer_list}, agents {q.agent_list})")
    if not s["trainers"]:
        lines.append("  no training launcher or trainer is live")
    for t in s["trainers"]:
        if t["cpus"] is None:
            verdict = "affinity unreadable"
        elif t.get("agent_work"):
            verdict = "inside the agent list: agent work (a CPU-only run), not the pinned trainer"
        elif t["on_agent_cpus"]:
            verdict = f"MISMATCH: it may run on the agent CPUs {t['on_agent_cpus']}"
        elif t["is_trainer_list"]:
            verdict = "the trainer list"
        else:
            verdict = "narrower than the trainer list (an A/B arm, or a tree left narrowed)"
        lines.append(f"  live {t['kind']} pid {t['pid']}: cpus {t['cpus']}: {verdict}")
    m = s["ab_marker"]
    if m is not None:
        lines.append(f"  A/B MARKER present ({ab_marker_path(Path(s['state_file']))}): {json.dumps(m, sort_keys=True)}. "
                     "A driver is running, or one died without restoring: `python -m main.ops.core_split_ab restore`")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m utils.core_split", description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sh = sub.add_parser("show", help="the active split, a queued one, the live trainers and their CPUs")
    sh.add_argument("--json", action="store_true")
    st = sub.add_parser("set", help="set the split (refused while a trainer is live), or queue it with --next-launch")
    st.add_argument("split", choices=SPLITS)
    st.add_argument("--next-launch", action="store_true", help="queue it: active when the next launch reads its pin")
    st.add_argument("--note", default="", help="why (kept in the state file's log)")
    sub.add_parser("agent-cpus", help="the agent CPU list, taskset's form")
    tr = sub.add_parser("trainer-cpus", help="the trainer CPU list, taskset's form")
    tr.add_argument("--launch", action="store_true", help="a launch kit's read: a queued split becomes active first")
    tr.add_argument("--dry-run", action="store_true", help="with --launch: what a launch would get, writing nothing")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        a = _parser().parse_args(argv)
    except SystemExit as e:
        return EXIT_USAGE if e.code not in (0, None) else EXIT_OK
    try:
        if a.cmd == "show":
            s = status()
            print(json.dumps(s, indent=1) if a.json else render(s))
        elif a.cmd == "set":
            print(set_split(a.split, next_launch=a.next_launch, note=a.note))
        elif a.cmd == "agent-cpus":
            print(agent_cpus())
        elif a.launch:
            cpus, note = launch_trainer_cpus(dry_run=a.dry_run)
            if note:
                print(note, file=sys.stderr)
            print(cpus)
        elif a.dry_run:
            print("REFUSING: --dry-run goes with --launch", file=sys.stderr)
            return EXIT_USAGE
        else:
            print(trainer_cpus())
    except Refused as e:
        print(str(e), file=sys.stderr)
        return EXIT_REFUSED
    except CoreSplitError as e:
        print(f"core_split: {e}", file=sys.stderr)
        return EXIT_REFUSED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
