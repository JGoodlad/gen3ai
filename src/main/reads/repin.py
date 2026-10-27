"""REPIN: let a running read adopt a newer checkout, but only when it cannot change a game (``designs/tools/reads.md`` §3).

A state's ``code.json`` is the checkout ``plan`` ran at, and every supervisor refuses to work from any other commit (the
pin guard, ``runner._code_holds``). That is right for play code, and wrong for a scheduling fix to the supervisor itself
(``--concurrency``): a long read could never take one. ``python -m main.reads repin --state S --to <commit> --by WHO --why
TEXT`` accepts ``<commit>`` as a second playing commit of the read IF AND ONLY IF the tracked play path
(:mod:`main.reads.playpath`) is byte-identical between ``code.json``'s commit and ``<commit>``. Otherwise it REFUSES and
names the files that differ; nothing is written.

What it writes is ``<state>/repins.jsonl``, APPEND-ONLY (one JSON line per repin; never rewritten). ``code.json`` is left
exactly as ``plan`` wrote it, on purpose: a supervisor of older code, still running from the old checkout, reads only
``code.json['commit']``, so it keeps working through a repin (the S lane stays up while the anchor lane moves). A
supervisor of this code accepts the plan commit or any repinned one (:func:`accepted_commits`), with the same clean-tree
and diff-digest checks. The record lists every repin (``record['repins']``).

A repin never moves a spec's registered ``pin``: a spec that names one refuses.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from main.reads import layout as LY
from main.reads import playpath as PP

SCHEMA = "gen3_read_repin_v1"
REQUIRED = ("schema", "from_commit", "to_commit", "playpath_digest", "playpath_files", "playpath_schema", "by", "why",
            "time")
SHOWN_DIFFS = 12


class RepinError(RuntimeError):
    """A repin that is refused (named; the state is untouched)."""


def repins_file(state: Path) -> Path:
    return LY.repins_file(state)


def load_repins(state: Path) -> List[Dict[str, Any]]:
    """Every repin recorded in ``state``, in order. A line that does not parse or lacks a field is a refusal: the pin
    guard reads this, and a guard that skipped a damaged line would be a guard that guessed."""
    p = repins_file(state)
    if not p.exists():
        return []
    out = []
    for i, line in enumerate(p.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError as exc:
            raise RepinError(f"{p}:{i}: not JSON ({exc})") from None
        missing = [k for k in REQUIRED if k not in rec]
        if missing:
            raise RepinError(f"{p}:{i}: a repin lacks {missing}")
        out.append(rec)
    return out


def accepted_commits(state: Path) -> List[str]:
    """The commits a supervisor may play from: ``code.json``'s, then each repin's, in order."""
    cf = LY.code_file(state)
    if not cf.exists():
        return []
    return [json.loads(cf.read_text())["commit"], *[r["to_commit"] for r in load_repins(state)]]


def repin(state: Path, to: str, *, by: str, why: str, repo: Optional[Path] = None,
          now: Optional[float] = None) -> Dict[str, Any]:
    """Accept ``to`` as a playing commit of ``state``'s read (module docstring). Returns the repin's record (with
    ``already: True`` and nothing written when ``to`` is already accepted)."""
    from main.reads.spec import load as load_spec

    if not (isinstance(by, str) and by.strip()):
        raise RepinError("--by: who is repinning (a name) is required")
    if not (isinstance(why, str) and why.strip()):
        raise RepinError("--why: the reason is required")
    state = LY.check_state_dir(state)
    if not LY.code_file(state).exists() or not LY.spec_file(state).exists():
        raise RepinError(f"{state}: no plan (code.json / spec.json); repin works on a planned state")
    spec = load_spec(LY.spec_file(state))
    if spec.pin is not None:
        raise RepinError(f"the spec registers pin {spec.pin}: the playing commit is part of the registration, and a "
                         "repin never moves it (amend the registration instead)")
    try:
        new = PP.resolve(to, repo)
        old = PP.resolve(json.loads(LY.code_file(state).read_text())["commit"], repo)
    except PP.PlayPathError as exc:
        raise RepinError(str(exc)) from None
    with open(LY.repin_lock(state), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)              # one repin at a time; the check and the append are one step
        if new in accepted_commits(state):
            prior = [r for r in load_repins(state) if r["to_commit"] == new]
            return dict(prior[-1] if prior else {"to_commit": new, "from_commit": old}, already=True)
        try:
            diffs, dig, n = PP.compare(old, new, repo)
        except PP.PlayPathError as exc:
            raise RepinError(str(exc)) from None
        if diffs:
            shown = "; ".join(diffs[:SHOWN_DIFFS]) + (f"; ... and {len(diffs) - SHOWN_DIFFS} more" if len(diffs) > SHOWN_DIFFS
                                                      else "")
            raise RepinError(f"the play path differs between {old[:12]} (code.json) and {new[:12]}: {len(diffs)} "
                             f"file(s): {shown}. A repin is only for a change that cannot alter a game")
        t = time.time() if now is None else now
        rec: Dict[str, Any] = {
            "schema": SCHEMA, "from_commit": old, "to_commit": new, "playpath_digest": dig, "playpath_files": n,
            "playpath_schema": PP.SCHEMA, "by": by.strip(), "why": why.strip(),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t)), "t": t,
            "root": str(repo or PP.default_repo())}
        fd = os.open(repins_file(state), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, (json.dumps(rec, sort_keys=True) + "\n").encode())    # one write: a reader sees a whole line
            os.fsync(fd)
        finally:
            os.close(fd)
        return rec
