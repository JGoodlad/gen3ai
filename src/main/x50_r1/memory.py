"""THE TEAM-MEMORY EXCLUSION (F-1b-2; ``design_x50_rung1b_human_opponent_model.md`` §3).

The off-pool bank's opponents pilot ``heldout72_hpfix``: 72 teams from Metamon's ``hl_05_26``, which was built from
human ladder replays, so a human in the training window may have piloted one of them. A training battle in which SOME
side has >= :data:`NEAR` mons whose revealed facts all fit the same heldout team's sets is EXCLUDED, both seats.

A mon FITS a heldout set when the set is the same species (``team_overlap``'s forme rule), holds every move the log
revealed for it (a bare, untyped ``hiddenpower`` fits any typed Hidden Power; Struggle is no reveal), and holds its
revealed ORIGINAL item and ability, where revealed. A mon that revealed only its species fits a set of that species:
the rule counts what the log cannot tell apart. Species are unique within a team (the species clause), so the fits of
a side against one team are one-to-one. The reveals are ``agents.info_levers.revealed``'s (the one parser of "what
play revealed"), each side read from the other seat.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

#: the exclusion threshold: mons of one side fitting one heldout team (the off-pool proof's ``teams_sharing_ge4``)
NEAR = 4


class MemoryError_(RuntimeError):
    """The heldout team set does not load as registered (a wrong manifest, a sha256 mismatch, not 72 teams)."""


def load_heldout(manifest: Path) -> Dict[str, Tuple]:
    """The heldout teams by id from ``heldout72_hpfix_manifest.json`` (its ``teams_heldout72_hpfix/`` sibling), each
    file sha256-checked: id -> a tuple of ``team_overlap.MonSet``."""
    from main.anchors.team_overlap import parse_team

    man = json.loads(Path(manifest).read_text())
    d = Path(manifest).parent / "teams_heldout72_hpfix"
    out: Dict[str, Tuple] = {}
    for tid, rec in sorted(man["teams"].items()):
        b = (d / f"{tid}.txt").read_bytes()
        if hashlib.sha256(b).hexdigest() != rec["sha256"]:
            raise MemoryError_(f"{tid}: sha256 differs from {manifest}")
        out[tid] = parse_team(b.decode(), tid).mons
    if len(out) != int(man.get("n", 72)) or len(out) != 72:
        raise MemoryError_(f"{manifest}: {len(out)} teams, not 72")
    return out


def _fits(sid: str, moves: Sequence[str], item: Optional[str], ability: Optional[str], ms) -> bool:
    from main.anchors.team_overlap import _species_id

    if _species_id(sid) != ms.species:
        return False
    for mv in moves:
        if mv == "struggle":
            continue
        if mv == "hiddenpower":
            if not any(x.startswith("hiddenpower") for x in ms.moves):
                return False
        elif mv not in ms.moves:
            return False
    if item is not None and item != ms.item:
        return False
    if ability is not None and ability != ms.ability:
        return False
    return True


class HeldoutMatcher:
    """:meth:`matches` a public log against the heldout teams (module docstring)."""

    def __init__(self, manifest: Path):
        self.teams = load_heldout(manifest)
        self.sha256 = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()

    def side_fits(self, lines: Sequence[str], seat: str) -> Dict[str, int]:
        """Per heldout team, how many of ``seat``'s revealed mons fit one of its sets (teams with none omitted)."""
        from agents.info_levers import revealed
        from agents.info_levers.packed import toid

        rev = revealed.from_lines(list(lines), 0 if seat == "p2" else 1)      # the OTHER seat's view of this one
        mons = [(toid(sid), sorted(toid(x) for x in m.moves), m.item, m.ability) for sid, m in rev.mons.items()]
        out: Dict[str, int] = {}
        for tid, sets in self.teams.items():
            n = sum(1 for (sid, mv, it, ab) in mons if any(_fits(sid, mv, it, ab, ms) for ms in sets))
            if n:
                out[tid] = n
        return out

    def matches(self, lines: Sequence[str]) -> List[Tuple[str, str, int]]:
        """(seat, heldout team id, fitting mons) for every side and team at or above :data:`NEAR`; empty = keep."""
        hits: List[Tuple[str, str, int]] = []
        for seat in ("p1", "p2"):
            for tid, n in sorted(self.side_fits(lines, seat).items()):
                if n >= NEAR:
                    hits.append((seat, tid, n))
        return hits
