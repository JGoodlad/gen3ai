"""THE TRAINING POOL as the eval opponents played it: the packed teams POOL and POOLSETS read (HEAD side; a MEASUREMENT
construct only — the pool never ships as a prior, owner's rule 2026-08-15).

The pool = ``utils.teambuilder.Gen3Teambuilder(utils.team_loader.TeamLoader().get_all_teams()).packed_teams``: the
trainer's own path (the paste parsed, ``fix_gen3_hp_ivs``, packed, invalid teams skipped by the Node validator). The
plain ``Teambuilder.parse_showdown_team(...).packed`` is NOT it: it skips the Hidden Power IV fix, and only 254 / 246
of the 600 registered opponent teams were raw-equal to it (measured 2026-10-13), against 600 / 600 for this path.

``data/teams`` and the packing code are identical at both pins (62085750, 95d014fa) and at HEAD except the gender
draw pad's constant in ``utils/team_packing.py`` (``git diff --stat <pin> HEAD -- data/teams src/utils/teambuilder.py
src/utils/gen3_utils.py src/utils/team_loader``: empty for 62085750; for 95d014fa only a test file,
``team_loader/committed_team_bytes_test.py``), so HEAD's pool IS the pins' pool. Measured 2026-10-13: 719 teams (717
distinct), sha256 of the joined pool 4083c86bdffa…; 600 / 600 registered opponent teams per checkpoint are raw-equal
to a pool team.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List

_CACHE: Dict[str, "Pool"] = {}


@dataclass
class Pool:
    teams: List[str]                                   # packed, in the trainer's pool order (duplicates kept)
    sha256: str                                        # of "\n".join(teams)
    index_of: Dict[str, List[int]] = field(default_factory=dict)   # packed text → its pool indices


def load_pool() -> Pool:
    """The trainer's packed pool (memoised per process)."""
    if "pool" not in _CACHE:
        from utils.team_loader import TeamLoader
        from utils.teambuilder import Gen3Teambuilder

        raw = TeamLoader().get_all_teams()
        teams = list(Gen3Teambuilder(raw).packed_teams)
        if len(teams) != len(raw):
            raise RuntimeError(f"the validator dropped {len(raw) - len(teams)} of {len(raw)} pool teams")
        idx: Dict[str, List[int]] = {}
        for i, t in enumerate(teams):
            idx.setdefault(t, []).append(i)
        _CACHE["pool"] = Pool(teams=teams, sha256=hashlib.sha256("\n".join(teams).encode()).hexdigest(), index_of=idx)
    return _CACHE["pool"]


def true_indices(pool: Pool, team: str) -> List[int]:
    """The pool indices whose packed text IS ``team`` (the truth's own team)."""
    return list(pool.index_of.get(team, []))


def set_key(entry: str) -> tuple:
    """A packed set's facts modulo name, gender and slot order: species, item, ability, the move SET (typed), nature,
    EVs, IVs — what 'the drawn team is the true team' compares."""
    from main.value_search.worlds import Spread, parse_set, toid, typed_moves

    f = parse_set(entry)
    sp = Spread.of(f)
    return (toid(f[1] or f[0]), toid(f[2]), toid(f[3]), tuple(sorted(typed_moves(f))), toid(sp.nature),
            tuple(sp.evs), tuple(sp.ivs))


def team_key(team: str) -> tuple:
    return tuple(sorted(set_key(e) for e in team.split("]")))
