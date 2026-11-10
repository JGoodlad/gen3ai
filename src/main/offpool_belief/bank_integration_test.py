"""The bank's games on the REAL env core (the self-check build): the schedule is played exactly, only the facing
seat's rows are kept, and the rows do not depend on how many envs played them.

The pilot here is a stub that picks a legal action from a hash of its own observation row: a pure function of the
row, like a greedy policy, with no model to load."""
from __future__ import annotations

import zlib
from typing import Dict, List

import numpy as np
import pytest

from main.offpool_belief import bank as B

pytestmark = [pytest.mark.sim, pytest.mark.integration]

N_PANEL, N_HOME, PER_TEAM, TURN_LIMIT = 3, 4, 2, 30


@pytest.fixture(scope="module")
def lib():
    from utils.rust_env import ffi
    from utils.rust_env.build import ensure_built

    ensure_built("selfcheck")
    return ffi.load(ffi.default_path("selfcheck"), nan_poison=True)


def _teams() -> List[str]:
    from utils.ladder_corpus import teams

    return list(teams("commit")[:N_PANEL + N_HOME])


def _hash_actor(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = []
    for o, m in zip(obs, mask):
        legal = np.flatnonzero(m)
        out.append(int(legal[zlib.crc32(o.tobytes()) % len(legal)]))
    return np.asarray(out)


def _play(lib, n_envs: int) -> Dict[str, np.ndarray]:
    teams = _teams()
    sch = B.game_schedule(N_PANEL, N_HOME, PER_TEAM, seed=5, same_panel=False)
    return B.play(teams[:N_PANEL], teams[N_PANEL:], sch, _hash_actor, lib=lib, n_envs=n_envs, turn_limit=TURN_LIMIT)


@pytest.fixture(scope="module")
def rows(lib) -> Dict[str, np.ndarray]:
    return _play(lib, 2)


def test_the_schedule_is_played_exactly(rows) -> None:
    sch = B.game_schedule(N_PANEL, N_HOME, PER_TEAM, seed=5, same_panel=False)
    assert set(rows) == set(B.ROW_KEYS)
    assert sorted(set(rows["game"].tolist())) == list(range(N_PANEL * PER_TEAM))
    assert (np.diff(rows["game"]) >= 0).all()                         # ordered by game
    for g in sch:
        m = rows["game"] == g.g
        assert m.any() and (rows["team"][m] == g.team).all() and (rows["home"][m] == g.home).all()
        assert len(set(rows["winner"][m].tolist())) == 1               # one outcome a game
        assert (np.diff(rows["turn"][m]) >= 0).all()                   # one game's decisions, in order
    n = len(rows["obs"])
    assert all(len(v) == n for v in rows.values()) and np.isfinite(rows["obs"]).all()


def test_the_rows_are_the_seat_that_faces_the_panel_team(rows, lib) -> None:
    """Seat 2's opponent is the PANEL team: the species its labels and its observation name are the panel team's,
    never the home team's."""
    from agents.observation.constants import OFFSET_OPP_TEAM, POKEMON_FULL_DIM, POKEMON_SPECIES_OFFSET

    seen: Dict[int, set] = {}
    for r in range(len(rows["obs"])):
        opp = rows["obs"][r, OFFSET_OPP_TEAM:OFFSET_OPP_TEAM + 6 * POKEMON_FULL_DIM].reshape(6, POKEMON_FULL_DIM)
        ids = {int(round(x)) for x in opp[:, POKEMON_SPECIES_OFFSET]} | {int(s) for s in rows["belief_species"][r]}
        seen.setdefault(int(rows["team"][r]), set()).update(i for i in ids if i > 0)
    # every game against one panel team sees a subset of ONE six-mon roster, whatever home team seat 2 pilots
    assert all(1 <= len(s) <= 6 for s in seen.values()), {t: sorted(s) for t, s in seen.items()}
    assert len({frozenset(s) for s in seen.values()}) == N_PANEL       # and the three panel teams differ
    assert (rows["belief_species"] >= 0).any() and (rows["known_moves"] >= 0).any()


def test_the_rows_do_not_depend_on_the_env_count(rows, lib) -> None:
    again, wider = _play(lib, 2), _play(lib, 5)
    for k in B.ROW_KEYS:
        assert np.array_equal(rows[k], again[k]), f"{k}: a rebuild differs"
        assert np.array_equal(rows[k], wider[k]), f"{k}: 5 envs (with filler games) differ from 2"
