"""Familiarity of every mon in a parsed game: is its set (mon tier: species + moves) in our 719-team training pool, and
how often is its species there. Teams are matched to the panel's team files by their six (species, moves) pairs, normalised as
``main.anchors.team_overlap`` does, with Hidden Power's type collapsed (the packed START team may not carry it);
the set (typed Hidden Power, item, spread) then comes from the file."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from main.anchors.session import load_team_texts, pool_team_texts
from main.anchors.team_overlap import Reference, _species_id, parse_team

TEAM_DIRS = {
    "home72": "teams_home72",
    "H": "teams_heldout72_hpfix",
    "F": "teams_forum72",
    "P72": "teams_p72",
}
X51 = Path("/home/goodlad/dev/gen3ai-reads/x51_2026-10-20")


def _loose_key(mons) -> tuple:
    return tuple(sorted((sp, tuple(sorted("hiddenpower" if m.startswith("hiddenpower") else m for m in mv)))
                        for sp, mv in mons))


class Familiarity:  # noqa: D101 — see the module docstring
    def __init__(self) -> None:
        pool = [parse_team(t) for t in pool_team_texts()]
        if len(pool) != 719:
            raise SystemExit(f"pool has {len(pool)} teams, expected X26's 719")
        self.n_pool = len(pool)
        self.pool_mon_keys = Counter(k for t in pool for k in {m.key for m in t.mons})
        self.pool_species = Counter(s for t in pool for s in {m.species for m in t.mons})
        self.ref = Reference(pool)
        self._shared = {}
        self.index = {}
        for panel, sub in TEAM_DIRS.items():
            idx = {}
            for label, text in load_team_texts(X51 / sub):
                team = parse_team(text, label)
                idx.setdefault(_loose_key((m.species, m.moves) for m in team.mons), []).append(team)
            self.index[panel] = idx

    def team(self, panel: str, mons: list):
        hits = self.index[panel].get(_loose_key((_species_id(sp), mv) for sp, mv in mons), [])
        if len({t.key for t in hits}) != 1:
            return None  # unmatched or ambiguous: reported, never guessed
        return hits[0]

    def mons(self, panel: str, mons: list):
        """{species_id: (set_seen, species_pool_share)} for one side, or None if the team is not matched."""
        t = self.team(panel, mons)
        if t is None:
            return None
        return {m.species: (self.pool_mon_keys[m.key] > 0, self.pool_species[m.species] / self.n_pool)
                for m in t.mons}

    def max_shared(self, panel: str, mons: list) -> int:
        """The most mon sets (mon tier) this side's team shares with any ONE pool team (team context)."""
        t = self.team(panel, mons)
        if t.key not in self._shared:
            self._shared[t.key] = self.ref.max_shared(t)[0]
        return self._shared[t.key]
