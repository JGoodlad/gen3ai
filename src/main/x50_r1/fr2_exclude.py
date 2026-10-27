"""FR-2's battle exclusion (X59-S PREREG Amendment 4c): the battles whose rows the X26 pin and today's reader encode
differently, found from the protocol alone.

    python -m main.x50_r1.fr2_exclude --run <main>/models/rb_x26_s2601 --min-step 360000000 --out <archive>/fr2_exclude.json

``gen3_volatile_silent_ends_v1`` (``d8d674b99``) made the reader clear seven volatiles that Showdown ends WITHOUT a
protocol line (Endure / Snatch / Helping Hand at ``|turn|``, Destiny Bond / Grudge / Rage at the holder's next move,
Mean Look's ``trapped`` when its trapper leaves). X26's traces were recorded before it, so a row after such an end
reads 0 today where the trace recorded 1: G-ENC's 11 DIFF battles, every differing cell one of these volatiles. A
battle is flagged iff either seat's stream holds a line that STARTS one (:data:`PATTERNS`); the flag is per battle
(never per row: a stale cell can sit at a decision only one seat has), and both seats' rows of a flagged battle drop.

The list is JSON (absolute summary paths, as :func:`main.x50_r1.run.trace_summaries` names them, under ``battles``);
:func:`load` returns it with the file's sha256, which ``genc --exclude`` and ``read_fr2 --exclude`` record.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

SCHEMA = "x59s_fr2_exclude_v1"

#: (protocol keyword, effect) pairs whose line STARTS a silently ending volatile. The effect is the line's third field,
#: optionally ``move: ``-prefixed; trailing fields (``[of] …``) are allowed.
STARTS: Tuple[Tuple[str, str], ...] = (
    ("-singleturn", "Endure"), ("-singleturn", "Snatch"), ("-singleturn", "Helping Hand"),
    ("-singlemove", "Destiny Bond"), ("-singlemove", "Grudge"), ("-singlemove", "Rage"),
    ("-activate", "trapped"),
)
PATTERNS: Tuple["re.Pattern[str]", ...] = tuple(
    re.compile(rf"^\|{re.escape(kw)}\|[^|]*\|(?:move: )?{re.escape(eff)}(?:\||$)") for kw, eff in STARTS)


def match(line: str) -> Optional[str]:
    """``"<keyword>:<effect>"`` when ``line`` starts a silently ending volatile, else ``None``."""
    for (kw, eff), pat in zip(STARTS, PATTERNS):
        if pat.match(line):
            return f"{kw}:{eff}"
    return None


def scan_lines(lines: Iterable[str]) -> FrozenSet[str]:
    """The :func:`match` kinds a stream holds."""
    return frozenset(k for k in map(match, lines) if k is not None)


def scan_trace(summary: str) -> FrozenSet[str]:
    """The kinds in either seat's stream of one trace (the public lines are the same; both are read anyway)."""
    from main.x50_r1 import spectate as S

    p = Path(summary)
    ts = json.loads(p.read_text())["meta"]["trace_source"]
    out: set = set()
    for rec in ts["records"].values():
        out |= scan_lines(S.read_stream(p.parent / rec)[1])
    return frozenset(out)


def norm(path: str) -> str:
    """The key a battle is matched by: its summary's real path."""
    return os.path.realpath(path)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load(path: Path) -> Tuple[FrozenSet[str], str]:
    """(the flagged battles' keys, the file's sha256). Refuses a file of another schema."""
    raw = Path(path).read_bytes()
    body = json.loads(raw)
    if body.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {body.get('schema')!r}, not {SCHEMA!r}")
    keys = frozenset(norm(b) for b in body["battles"])
    if len(keys) != len(body["battles"]) or len(keys) != body["count"]:
        raise ValueError(f"{path}: {len(body['battles'])} battles, {len(keys)} distinct, count {body['count']}")
    return keys, sha256_bytes(raw)


def build(summaries: Sequence[str]) -> Dict[str, object]:
    """The list's body over ``summaries`` (flagged battles in the input order)."""
    flagged: List[str] = []
    kinds: collections.Counter = collections.Counter()
    for sp in summaries:
        ks = scan_trace(sp)
        if ks:
            flagged.append(sp)
            kinds.update(ks)
    return {"schema": SCHEMA, "rule": "X59-S PREREG Amendment 4c: a battle whose stream starts a silently ending "
                                      "volatile (gen3_volatile_silent_ends_v1, d8d674b99)",
            "patterns": [p.pattern for p in PATTERNS], "scanned": len(summaries), "count": len(flagged),
            "kinds": dict(sorted(kinds.items())), "battles": flagged}


def main(argv: Optional[List[str]] = None) -> int:
    from main.x50_r1.run import trace_summaries
    from utils.git import get_git_hash

    ap = argparse.ArgumentParser(prog="python -m main.x50_r1.fr2_exclude", description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--min-step", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    body = build(trace_summaries(Path(a.run), a.min_step))
    body.update({"run": a.run, "min_step": a.min_step, "commit": get_git_hash()})
    raw = json.dumps(body, indent=1).encode()
    Path(a.out).write_bytes(raw)
    print(f"[fr2_exclude] {body['count']} of {body['scanned']} battles flagged {body['kinds']}; "
          f"{a.out} sha256 {sha256_bytes(raw)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
