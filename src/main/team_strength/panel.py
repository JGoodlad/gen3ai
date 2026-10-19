"""TEAM SETS, team IDENTITY and the frozen REFERENCE PANEL of the team-strength meter (``gen3_tsr_panel_v1``).

* A TEAM SET is a directory of Showdown paste files (``*.txt`` / ``*.gen3ou_team``), read in sorted file order.
  Every file is recorded by its sha256 (of the raw bytes), and the set by ``gen3_team_set_digest_v1`` — sha256 over
  the sorted sha256 of each file's STRIPPED text, the anchors' B-D6 content digest, so a set reads the same whichever
  copy (ours or Metamon's byte copy) is hashed.
* A team's IDENTITY is its six STRICT mon sets (``main.anchors.team_overlap``: species, sorted moves with Hidden Power
  typed, item, ability, nature, EVs): :func:`team_id`. It does not depend on file name, nickname or blank-line layout,
  so one team found in two sets (an ``away`` team copied into home72) is ONE rated team, and a team Metamon sends back
  in a capture's START command is recognised by the same key.
* The REFERENCE PANEL is a fixed draw of teams from named sets (``draw_reference``): each named set's teams, by
  identity, sorted, shuffled once by ``random.Random(seed)`` and taken in that order until the set's quota is met,
  skipping a team already drawn. It is frozen by its member list and :func:`panel_digest` BEFORE any game.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from main.anchors import team_overlap as TO

SCHEMA = "gen3_tsr_panel_v1"
SET_DIGEST = "gen3_team_set_digest_v1"
TEAM_SUFFIXES = (".txt", ".gen3ou_team")


class PanelError(RuntimeError):
    """A team set or a panel that cannot be read honestly (named)."""


def team_id(text: str) -> str:
    """``tsr:`` + 16 hex of sha256 over the sorted STRICT mon-set keys of the paste ``text`` (module docs).
    A paste with no parseable mon is refused (it would collide with every other empty team)."""
    t = TO.parse_team(text)
    if not t.mons:
        raise PanelError("a team with no parseable mon has no identity")
    keys = sorted(repr(k) for k in t.strict_key)
    return "tsr:" + hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]


def mon_key(text: str) -> Any:
    """The species + moves key of a paste (``team_overlap``'s ``mon`` tier) — the capture matcher's fallback."""
    return TO.parse_team(text).key


@dataclass(frozen=True)
class Member:
    set: str
    stem: str
    file: str
    path: str
    sha256: str
    team_id: str
    text: str = field(repr=False, compare=False)

    def block(self) -> Dict[str, Any]:
        return {"set": self.set, "stem": self.stem, "file": self.file, "sha256": self.sha256, "team_id": self.team_id}


@dataclass(frozen=True)
class TeamSet:
    name: str
    dir: str
    members: Tuple[Member, ...]

    @property
    def digest(self) -> str:
        hs = sorted(hashlib.sha256(m.text.strip().encode()).hexdigest() for m in self.members)
        return hashlib.sha256("\n".join(hs).encode()).hexdigest()

    @property
    def team_ids(self) -> List[str]:
        """The set's DISTINCT team ids, in file order (a duplicate paste inside one set counts once)."""
        out: List[str] = []
        for m in self.members:
            if m.team_id not in out:
                out.append(m.team_id)
        return out

    def block(self) -> Dict[str, Any]:
        return {"dir": self.dir, "digest_rule": SET_DIGEST, "digest": self.digest, "n_files": len(self.members),
                "n_teams": len(self.team_ids), "files": {m.stem: m.sha256 for m in self.members},
                "team_ids": {m.stem: m.team_id for m in self.members},
                "file_names": {m.stem: m.file for m in self.members}}


def read_set(name: str, directory: Path) -> TeamSet:
    d = Path(directory)
    if not d.is_dir():
        raise PanelError(f"team set {name}: {d} is not a directory")
    files = sorted(f for f in d.iterdir() if f.suffix in TEAM_SUFFIXES)
    if not files:
        raise PanelError(f"team set {name}: {d} holds no team file")
    members = []
    for f in files:
        raw = f.read_bytes()
        text = raw.decode("utf-8", errors="strict")
        members.append(Member(set=name, stem=f.stem, file=f.name, path=str(f), sha256=hashlib.sha256(raw).hexdigest(),
                              team_id=team_id(text), text=text))
    return TeamSet(name=name, dir=str(d), members=tuple(members))


def check_set(ts: TeamSet, frozen: Mapping[str, Any]) -> None:
    """REFUSE unless ``ts`` is byte for byte the registered set ``frozen`` (a :meth:`TeamSet.block`)."""
    got = {m.stem: m.sha256 for m in ts.members}
    want = dict(frozen["files"])
    if got != want:
        moved = sorted(set(got.items()) ^ set(want.items()))[:5]
        raise PanelError(f"team set {ts.name} at {ts.dir} is not the registered one (first differences: {moved})")
    if ts.digest != frozen["digest"]:
        raise PanelError(f"team set {ts.name}: content digest {ts.digest[:12]} != registered {frozen['digest'][:12]}")


def draw_reference(sets: Mapping[str, TeamSet], quota: Mapping[str, int], seed: int) -> List[Member]:
    """The reference panel (module docs): ``quota[name]`` distinct teams of each named set, in ``quota``'s order."""
    rng = random.Random(int(seed))
    drawn: Dict[str, Member] = {}
    for name, k in quota.items():
        ts = sets[name]
        first = {}
        for m in ts.members:
            first.setdefault(m.team_id, m)
        ids = sorted(first)
        rng.shuffle(ids)
        got = 0
        for tid in ids:
            if got >= int(k):
                break
            if tid in drawn:
                continue
            drawn[tid] = first[tid]
            got += 1
        if got < int(k):
            raise PanelError(f"reference draw: set {name} has only {got} teams not already drawn, quota {k}")
    return list(drawn.values())


def panel_digest(members: Sequence[Member]) -> str:
    """sha256 over the canonical JSON of the panel's ``(set, stem, sha256, team_id)`` rows in draw order."""
    rows = [m.block() for m in members]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def subset(ts: TeamSet, k: int, seed: int, salt: str) -> List[str]:
    """``k`` distinct team ids of ``ts`` (all of them when it has ≤ k): sorted, shuffled by
    ``random.Random(f"{seed}:{salt}")``, the first ``k``, returned sorted."""
    ids = sorted(ts.team_ids)
    if len(ids) <= int(k):
        return ids
    rng = random.Random(f"{int(seed)}:{salt}")
    rng.shuffle(ids)
    return sorted(ids[:int(k)])


def membership(sets: Mapping[str, TeamSet]) -> Dict[str, List[str]]:
    """``{team_id: [set names holding it]}``."""
    out: Dict[str, List[str]] = {}
    for name, ts in sets.items():
        for tid in ts.team_ids:
            out.setdefault(tid, [])
            if name not in out[tid]:
                out[tid].append(name)
    return out


def text_of(sets: Mapping[str, TeamSet]) -> Dict[str, str]:
    """``{team_id: paste text}`` (the first file holding the id, in set order)."""
    out: Dict[str, str] = {}
    for ts in sets.values():
        for m in ts.members:
            out.setdefault(m.team_id, m.text)
    return out


def load_registered(reg: Mapping[str, Any], names: Optional[Sequence[str]] = None) -> Dict[str, TeamSet]:
    """Read every registered set (or ``names``) from its registered dir and REFUSE any that moved."""
    out: Dict[str, TeamSet] = {}
    for name, blk in reg["sets"].items():
        if names is not None and name not in names:
            continue
        ts = read_set(name, Path(blk["dir"]))
        check_set(ts, blk)
        out[name] = ts
    return out
