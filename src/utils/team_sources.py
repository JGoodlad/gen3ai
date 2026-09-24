"""The TEAM-SOURCE hook every fuzz and parity gate shares (`gen3_ladder_usage_corpus_v1`).

Three sources, and a gate that plays teams should be able to play all three:

==========  ============================================  =========================================
source      what                                          surface
==========  ============================================  =========================================
pool        the 719 training teams (``TeamLoader``)       what training plays; one narrow meta
procedural  ``ou_random_teams.js`` (Smogon-derived,       wide, on-format, but no human built it
            ``TeamValidator``-legal, seeded)
ladder      the LADDER-USAGE corpus                       the teams people actually bring —
            (:mod:`utils.ladder_corpus`)                  where the Heal Bell crash hid
==========  ============================================  =========================================

The ``pool`` path is the pre-existing one, unchanged (``Gen3Teambuilder`` over the paste texts), so
a gate that gains ``--team-source`` plays byte-identical pool battles by default. ``procedural`` and
``ladder`` teams are PACKED strings, fed to the players through :class:`PackedTeamPool`.

The Rust Core parity harness uses :func:`pair` for its key recipe (``rust_core_parity.play(key,
source=…)``); slices T and O inherit it by calling the same ``play``.
"""
from __future__ import annotations

import random
import subprocess
from typing import List, Optional, Sequence, Tuple

from poke_env.teambuilder.teambuilder import Teambuilder

from utils.paths import repo_path

TEAM_SOURCES = ("pool", "procedural", "ladder")
OU_RANDOM_TEAMS_JS = repo_path("src", "rust_sim", "harness", "ou_random_teams.js")
#: The procedural generator's seed when a gate names none (the belief-calibration read's seed).
PROCEDURAL_SEED = 20260924


def procedural_teams(n: int, seed: int = PROCEDURAL_SEED) -> List[str]:
    """``n`` packed gen3ou teams from the PROCEDURAL generator (``ou_random_teams.js --emit``):
    Smogon-derived, validated by Showdown's own ``TeamValidator('gen3ou')``, reproducible from
    ``seed``."""
    p = subprocess.run(["node", str(OU_RANDOM_TEAMS_JS), "--emit", str(n), "--seed", str(seed)],
                       capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"ou_random_teams --emit failed: {p.stderr.strip()[-2000:]}")
    teams = [t for t in p.stdout.splitlines() if t.strip()]
    if len(teams) != n:
        raise RuntimeError(f"ou_random_teams --emit returned {len(teams)} teams for {n}")
    return teams


def team_list(source: str, *, ladder_tier: str = "milestone", n: Optional[int] = None,
              seed: int = PROCEDURAL_SEED) -> List[str]:
    """Every team of ``source``: pool paste texts, or packed strings (``n`` sizes the procedural
    draw, default 200; for ``ladder`` the tier sizes it)."""
    if source == "pool":
        from utils.team_loader import TeamLoader

        teams = TeamLoader().get_all_teams()
        if not teams:
            raise RuntimeError("no gen3ou teams found under data/teams")
        return teams
    if source == "procedural":
        return procedural_teams(n or 200, seed)
    if source == "ladder":
        from utils import ladder_corpus

        return ladder_corpus.teams(ladder_tier)
    raise ValueError(f"team source must be one of {TEAM_SOURCES}, got {source!r}")


def pair(source: str, key: int, *, ladder_tier: str = "milestone") -> Tuple[str, str]:
    """The two teams battle ``key`` plays under the parity key recipe: pool ``key`` / ``key+1``;
    ladder ``2·key`` / ``2·key+1`` of the tier (every team once over ``n/2`` keys); procedural a
    fresh seeded pair per key."""
    if source == "pool":
        pool = team_list("pool")
        return pool[key % len(pool)], pool[(key + 1) % len(pool)]
    if source == "ladder":
        from utils import ladder_corpus

        return ladder_corpus.pair(key, ladder_tier)
    if source == "procedural":
        a, b = procedural_teams(2, PROCEDURAL_SEED + key)
        return a, b
    raise ValueError(f"team source must be one of {TEAM_SOURCES}, got {source!r}")


class PackedTeamPool(Teambuilder):
    """Yields one of ``teams`` (packed strings) per battle, from its OWN seeded RNG — a pool of
    teams that did not come through ``Gen3Teambuilder``'s paste parser."""

    def __init__(self, teams: Sequence[str], rng_seed: Optional[int] = None):
        if not teams:
            raise ValueError("PackedTeamPool: no teams")
        self._teams = list(teams)
        self._rng = random.Random(rng_seed)

    def yield_team(self) -> str:
        return self._teams[self._rng.randrange(len(self._teams))]


def teambuilder(source: str, *, rng_seed: Optional[int] = None, ladder_tier: str = "milestone",
                n: Optional[int] = None) -> Teambuilder:
    """A per-battle team draw over ``source``. ``pool`` is the pre-existing ``Gen3Teambuilder``
    (its draw is the global ``random`` stream, exactly as before)."""
    if source == "pool":
        from utils.teambuilder import Gen3Teambuilder

        return Gen3Teambuilder(team_list("pool"))
    return PackedTeamPool(team_list(source, ladder_tier=ladder_tier, n=n), rng_seed=rng_seed)


def add_arguments(ap, default: str = "pool") -> None:
    """``--team-source`` and ``--ladder-tier`` on a fuzz script's parser."""
    from utils.ladder_corpus import TIERS

    ap.add_argument("--team-source", choices=TEAM_SOURCES, default=default,
                    help="which teams to play: the training pool, the procedural generator, or "
                         "the LADDER-USAGE corpus (default %(default)s)")
    ap.add_argument("--ladder-tier", choices=TIERS, default="milestone",
                    help="with --team-source ladder: the corpus tier (default %(default)s)")
