"""T28 — the LIVE-PLAY PARSE-PANIC HALT (owner 2026-10-07).

    "When we design ladder and play against humans, we need a special exit code if we panicked and
    couldn't parse the input. If that happens we should root cause the issue asap and not play any
    more games until we have fixed it, so we can be respectful."

The spec is ``designs/endstate/design_ladder_campaign.md``'s Decision record (2026-10-07). Three parts:

1. **A distinct exit code.** Any unparseable or unclassified protocol input, an encoder raise, or a
   failure to send a choice raises :class:`LiveParseHalt`; the live entry point records the marker and
   exits ``FATAL_LIVE_PARSE`` (``main.exit_codes.TrainExitCode``, 7) — never a generic crash, never a
   retry, never "the next game".
2. **A durable HALT marker** (:func:`record_halt`): the battle id, the raw offending lines (the battle's
   whole received stream up to the failure), the stack trace, the commit and the entry point, written
   atomically to :func:`halt_path`. Every live entry point calls :func:`refuse_if_halted` FIRST and will
   not start while the marker exists — self-vs-self custom games included.
3. **A clear that names the fix** (:func:`clear_halt`, ``python -m main.live.halt clear --fixed-by
   <commit>``): the commit must exist, be an ancestor of this checkout's HEAD, and touch at least one
   TEST file (the regression test built from the captured lines, which must fail on revert — the
   reviewer's half of the rule; the tool checks what a tool can). The cleared marker moves into the
   append-only history beside it with the fixing commit recorded.

WHERE THE MARKER LIVES. ``$GEN3AI_LIVE_HALT_FILE`` when set (authoritative), else
``~/.local/state/gen3ai/live_play_halt.json`` — per USER, not per checkout, so a worktree, the main
checkout and a pinned run all see the same halt. 🚨 Under pytest the default path is REFUSED
(:class:`HaltPathUnsealed`): a test that plants a bad line must never halt the owner's real play, so
every test names its own file through the env var.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Sequence

#: The env var naming the marker file (authoritative when set).
HALT_ENV = "GEN3AI_LIVE_HALT_FILE"
#: The marker's schema tag.
SCHEMA = "gen3_live_parse_halt_v1"
#: How many raw lines of the battle's received stream the marker keeps (the TAIL; the offending
#: lines are always kept in full in ``offending_lines``).
MAX_STREAM_LINES = 4000


class HaltPathUnsealed(RuntimeError):
    """Under pytest with no ``$GEN3AI_LIVE_HALT_FILE``: the real marker must never be touched by a test."""


class HaltActive(RuntimeError):
    """A HALT marker exists: this live entry point refuses to start (T28)."""


class LiveParseHalt(RuntimeError):
    """A live session met input it cannot read, an encoder raise, or a choice it could not send (T28).

    Maps to ``TrainExitCode.FATAL_LIVE_PARSE`` by NAME (``main.exit_codes._FATAL_BY_NAME``). Carries what
    the marker needs: the battle id, the offending raw lines, and the battle's received stream so far."""

    def __init__(self, reason: str, *, battle_id: Optional[str] = None,
                 offending_lines: Sequence[str] = (), stream: Sequence[str] = ()) -> None:
        super().__init__(reason)
        self.reason = reason
        self.battle_id = battle_id
        self.offending_lines = list(offending_lines)
        self.stream = list(stream)


#: Under pytest with no ``$GEN3AI_LIVE_HALT_FILE``, READS see this (never-existing) path: a test of a live
#: entry point must not depend on the owner's real halt state. WRITES raise :class:`HaltPathUnsealed`.
_PYTEST_UNSEALED = Path("/nonexistent/gen3ai-live-halt-unsealed-under-pytest.json")


def halt_path(*, write: bool = False) -> Path:
    """The marker file: ``$GEN3AI_LIVE_HALT_FILE``, else the per-user default.

    Under pytest with no override the real file is never touched: a READ sees no halt, a WRITE raises
    :class:`HaltPathUnsealed` (a planted bad line must never halt the owner's real play)."""
    env = os.environ.get(HALT_ENV)
    if env:
        return Path(env)
    if os.environ.get("PYTEST_CURRENT_TEST"):
        if write:
            raise HaltPathUnsealed(
                f"a test tried to WRITE the REAL live-play halt marker: set ${HALT_ENV} to a tmp path")
        return _PYTEST_UNSEALED
    return Path.home() / ".local" / "state" / "gen3ai" / "live_play_halt.json"


def history_path(path: Optional[Path] = None) -> Path:
    p = path or halt_path()
    return p.with_name(p.stem + "_history.jsonl")


def _head() -> str:
    from utils.git import get_git_hash
    return get_git_hash()


def read_halt(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The active marker, or ``None``. An UNREADABLE marker is still a halt (returned as a stub)."""
    p = path or halt_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        return {"schema": SCHEMA, "unreadable": f"{type(exc).__name__}: {exc}", "path": str(p)}


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def record_halt(*, reason: str, entry_point: str, battle_id: Optional[str] = None,
                offending_lines: Sequence[str] = (), stream: Sequence[str] = (),
                exc: Optional[BaseException] = None, extra: Optional[Dict[str, Any]] = None,
                path: Optional[Path] = None) -> Path:
    """Write the HALT marker. A marker that already exists is KEPT (the first halt is the one to
    root-cause); a later halt is appended to its ``later`` list instead of replacing it."""
    p = path or halt_path(write=True)
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)) if exc else None
    stream = list(stream)
    rec: Dict[str, Any] = {
        "schema": SCHEMA,
        "time": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "entry_point": entry_point,
        "argv": list(sys.argv),
        "commit": _head(),
        "reason": reason,
        "battle_id": battle_id,
        "offending_lines": list(offending_lines),
        "stream_total_lines": len(stream),
        "stream_tail": stream[-MAX_STREAM_LINES:],
        "trace": trace,
    }
    if extra:
        rec["extra"] = dict(extra)
    prior = read_halt(p)
    if prior is not None and "unreadable" not in prior:
        prior.setdefault("later", []).append(rec)
        rec = prior
    _atomic_write(p, json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return p


def banner(rec: Dict[str, Any], path: Path) -> str:
    return (
        "\n" + "=" * 78 + "\n"
        "🛑 FATAL_LIVE_PARSE — a live session could not read its input (T28).\n"
        f"   reason   : {rec.get('reason')}\n"
        f"   battle   : {rec.get('battle_id')}\n"
        f"   offending: {(rec.get('offending_lines') or ['?'])[:3]}\n"
        f"   marker   : {path}\n"
        "   ALL live play is HALTED until the input is root-caused: build a regression test from the\n"
        "   captured lines that FAILS on revert, land the fix, then\n"
        "     python -m main.live.halt clear --fixed-by <commit>\n"
        "   PUSH the orchestrator / owner now.\n" + "=" * 78)


def refuse_if_halted(entry_point: str, path: Optional[Path] = None) -> None:
    """Every live entry point's FIRST call: raise :class:`HaltActive` while a marker exists."""
    p = path or halt_path()
    rec = read_halt(p)
    if rec is None:
        return
    raise HaltActive(
        f"{entry_point}: REFUSED — live play is HALTED by a parse panic (T28). Marker: {p}\n"
        f"  reason: {rec.get('reason', rec.get('unreadable'))}\n"
        f"  battle: {rec.get('battle_id')}  commit: {rec.get('commit')}  at: {rec.get('time')}\n"
        "  Root-cause it (a regression test from the captured lines that fails on revert), land the fix,\n"
        "  then: python -m main.live.halt clear --fixed-by <commit>")


def exit_on_halt(exc: LiveParseHalt, *, entry_point: str, path: Optional[Path] = None) -> NoReturn:
    """Record the marker for ``exc``, print the banner, and exit ``FATAL_LIVE_PARSE``."""
    from main.exit_codes import TrainExitCode
    p = record_halt(reason=exc.reason, entry_point=entry_point, battle_id=exc.battle_id,
                    offending_lines=exc.offending_lines, stream=exc.stream, exc=exc, path=path)
    print(banner(read_halt(p) or {}, p), file=sys.stderr, flush=True)
    sys.exit(int(TrainExitCode.FATAL_LIVE_PARSE))


# -- the clear ----------------------------------------------------------------------------------

class ClearRefused(RuntimeError):
    """The named commit does not qualify as the fix (T28's clear rule)."""


def _git(*args: str, cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    from utils.paths import repo_root
    return subprocess.run(["git", *args], cwd=cwd or str(repo_root()), capture_output=True, text=True)


def is_test_path(path: str) -> bool:
    """A TEST file in this repo's conventions: a ``*_test.py`` (any tier) or a Rust ``tests/*.rs``."""
    name = path.rsplit("/", 1)[-1]
    return name.endswith("_test.py") or (path.endswith(".rs") and "/tests/" in "/" + path)


def check_fix_commit(commit: str, cwd: Optional[str] = None) -> Dict[str, Any]:
    """Resolve ``commit`` and check T28's clear rule: it exists, it is an ancestor of HEAD (the fix is
    in the code that will play), and it touches at least one test file. Returns its facts."""
    r = _git("rev-parse", "--verify", f"{commit}^{{commit}}", cwd=cwd)
    if r.returncode != 0:
        raise ClearRefused(f"--fixed-by {commit!r}: not a commit in this repository")
    sha = r.stdout.strip()
    if _git("merge-base", "--is-ancestor", sha, "HEAD", cwd=cwd).returncode != 0:
        raise ClearRefused(f"--fixed-by {sha[:12]}: not an ancestor of HEAD — the fix is not in the code "
                           "that would play")
    files = [f for f in _git("show", "--name-only", "--format=", sha, cwd=cwd).stdout.splitlines() if f]
    tests = [f for f in files if is_test_path(f)]
    if not tests:
        raise ClearRefused(f"--fixed-by {sha[:12]} touches no test file ({len(files)} files): the clear "
                           "needs the regression test built from the captured lines")
    subject = _git("show", "-s", "--format=%s", sha, cwd=cwd).stdout.strip()
    return {"commit": sha, "subject": subject, "tests": tests}


def clear_halt(fixed_by: str, *, note: str = "", path: Optional[Path] = None,
               cwd: Optional[str] = None) -> Dict[str, Any]:
    """Clear the active marker, recording the fixing commit in the append-only history."""
    p = path or halt_path(write=True)
    rec = read_halt(p)
    if rec is None:
        raise ClearRefused(f"no active halt at {p}")
    fix = check_fix_commit(fixed_by, cwd=cwd)
    entry = {"cleared_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
             "fixed_by": fix, "note": note, "halt": rec}
    h = history_path(p)
    h.parent.mkdir(parents=True, exist_ok=True)
    with open(h, "a") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    p.unlink()
    return entry


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.live.halt",
                                 description="The T28 live-play halt marker: status, show, clear.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="exit 0 when no halt is active, 1 when one is")
    sub.add_parser("show", help="print the active marker as JSON")
    c = sub.add_parser("clear", help="clear the active halt, naming the commit that fixed it")
    c.add_argument("--fixed-by", required=True, metavar="COMMIT",
                   help="the root-cause commit: an ancestor of HEAD that touches a test file")
    c.add_argument("--note", default="", help="one line on the root cause")
    a = ap.parse_args(argv)
    p = halt_path()
    rec = read_halt(p)
    if a.cmd == "status":
        if rec is None:
            print(f"[live-halt] clear ({p} absent)")
            return 0
        print(f"[live-halt] HALTED: {rec.get('reason', rec.get('unreadable'))} "
              f"(battle {rec.get('battle_id')}, {rec.get('time')}) — {p}")
        return 1
    if a.cmd == "show":
        print(json.dumps(rec, indent=1, sort_keys=True) if rec else "null")
        return 0 if rec is None else 1
    try:
        entry = clear_halt(a.fixed_by, note=a.note, path=p)
    except ClearRefused as exc:
        print(f"[live-halt] REFUSED: {exc}", file=sys.stderr)
        return 2
    print(f"[live-halt] cleared by {entry['fixed_by']['commit'][:12]} "
          f"({entry['fixed_by']['subject']}); history: {history_path(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
