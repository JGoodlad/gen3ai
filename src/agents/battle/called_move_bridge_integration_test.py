"""A REAL bridge battle full of CALLED moves, fully encoded at every decision
(`gen3_called_move_reading_v1`).

``poke_env/battle/called_move_reading_test.py`` pins each protocol shape of a called move on
constructed lines. This plays them on the REFERENCE Showdown (the node bridge: the Rust port
fail-louds on every move-caller, so it cannot play this battle) with two scripted players (fixed
teams, fixed sim seed, concurrency 1, no randomness in either policy), and BOTH run the full
``embed_battle`` observation encode at every decision. Before the fix, the first Metronome call
raised ``ValueError: Unhandled move message format`` in the parse task and the battle hung on
the timer — the live ladder crash (2 of 710 Metamon battles, 4 of 20,000 human logs).

The script (p1 calls, p2 stalls behind a Pressure Zapdos so the calls meet Pressure):

    T1-T4   p1 Clefable Metronome x4          p2 Zapdos Substitute / Protect / Thunder Wave / Protect
    T5      p1 -> Delcatty                     p2 Zapdos Substitute
    T6-T8   p1 Delcatty Assist x3              p2 Zapdos Protect / Thunder Wave / Protect
    T9      p1 -> Ludicolo                     p2 Zapdos Substitute
    T10-T11 p1 Ludicolo Nature Power x2        p2 Zapdos Protect / Substitute

after which both play their first legal damaging move until the battle ends. Each scripted action
is ASSERTED to have happened (its protocol line is present), never branched on.

The reading is judged against the SIM, not poke-env: p2's estimate of p1's caller PP (the
opponent-side sighting count) must equal the PP p1's own ``|request|`` reports at the same turn
— the engine's word — which is exactly the Pressure fact the constructed pins assert (gen3 charges
no Pressure PP for a sourced move).
"""
from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

from poke_env import AccountConfiguration
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.battle.gen3_battle import Gen3Battle
from agents.inference.player import Gen3Player
from utils.bridge.local_battle_runner import run_local_battles

pytestmark = pytest.mark.sim

_P1_TEAM = """
Clefable @ Leftovers
Ability: Cute Charm
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Metronome
- Seismic Toss
- Soft-Boiled
- Calm Mind

Delcatty @ Leftovers
Ability: Cute Charm
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Assist
- Double-Edge
- Heal Bell
- Wish

Ludicolo @ Leftovers
Ability: Swift Swim
EVs: 252 HP / 252 SpA / 4 SpD
Modest Nature
- Nature Power
- Surf
- Giga Drain
- Leech Seed
"""

_P2_TEAM = """
Zapdos @ Leftovers
Ability: Pressure
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Substitute
- Protect
- Thunder Wave
- Thunderbolt

Snorlax @ Leftovers
Ability: Immunity
EVs: 252 HP / 252 Def / 4 SpD
Careful Nature
- Body Slam
- Curse
- Rest
- Earthquake
"""

_P1_SCRIPT = [("move", "metronome")] * 4 + [("switch", "delcatty")] + [("move", "assist")] * 3 \
    + [("switch", "ludicolo")] + [("move", "naturepower")] * 2
_P2_SCRIPT = [("move", m) for m in (
    "substitute", "protect", "thunderwave", "protect", "substitute", "protect", "thunderwave",
    "protect", "substitute", "protect", "substitute")]
_SEED = [7, 70, 700, 7000]
_CALLERS = ("metronome", "assist", "naturepower")


class _ScriptedEncodingPlayer(Gen3Player):
    """Plays a fixed script, then its first legal damaging move; ENCODES the full observation at
    every decision and records, per turn, its own callers' request PP and its reading of the
    opponent's callers' PP."""

    def __init__(self, *, script, **kwargs):
        super().__init__(battle_class=Gen3Battle, **kwargs)
        self._script: List[Tuple[str, str]] = list(script)
        self.decisions = 0
        self.battle: Optional[Gen3Battle] = None
        self.own_pp: Dict[Tuple[int, str], int] = {}
        self.opp_pp: Dict[Tuple[int, str], int] = {}
        self.opp_moves: Dict[str, set] = {}

    def _pick(self, battle, mask) -> int:
        legal = [int(i) for i in np.flatnonzero(mask)]
        orders = [(i, self.action_to_order(i, battle).order) for i in legal]
        if self._script:
            kind, want = self._script[0]
            for i, o in orders:
                if ((kind == "move" and isinstance(o, Move) and o.id == want)
                        or (kind == "switch" and isinstance(o, Pokemon) and o.species == want)):
                    self._script.pop(0)
                    return i
        for i, o in orders:  # after the script / a forced switch: deterministic fallback
            if isinstance(o, Move) and o.base_power > 0:
                return i
        return legal[0]

    def choose_move(self, battle):
        self.battle = battle
        obs = self.embed_battle(battle)  # the full obs, at every decision
        assert np.all(np.isfinite(obs["observation"]))
        self.decisions += 1
        view = battle.live_view()
        for mon in view.ours.mons:
            if mon.active:
                for mv in mon.moves:
                    if mv.id in _CALLERS:
                        self.own_pp[(battle.turn, mv.id)] = mv.current_pp
        for mon in view.opp.mons:
            self.opp_moves.setdefault(mon.species, set()).update(m.id for m in mon.moves)
            if mon.active:
                for mv in mon.moves:
                    if mv.id in _CALLERS:
                        self.opp_pp[(battle.turn, mv.id)] = mv.current_pp
        mask = obs["action_mask"]
        if int(mask.sum()) == 0:
            return self.choose_default_move()
        idx = self._pick(battle, mask)
        self._get_tracker(battle).advance(idx)
        return self.action_to_order(idx, battle)


@pytest.fixture(scope="module")
def played():
    common = dict(battle_format="gen3ou", server_configuration=LocalhostServerConfiguration,
                  start_listening=False, max_concurrent_battles=1)
    p1 = _ScriptedEncodingPlayer(script=_P1_SCRIPT, team=_P1_TEAM,
                                 account_configuration=AccountConfiguration("CmCaller", "pw"), **common)
    p2 = _ScriptedEncodingPlayer(script=_P2_SCRIPT, team=_P2_TEAM,
                                 account_configuration=AccountConfiguration("CmStaller", "pw"), **common)
    asyncio.run(run_local_battles(p1, p2, 1, seed=_SEED, impl="node", concurrency=1))
    return p1, p2


def _lines(player) -> List[str]:
    return ["|".join(e.raw) for e in player.battle.events_since(0) if e.raw]


def test_battle_went_as_scripted_with_every_decision_encoded(played):
    """The precondition, ASSERTED: the battle finished, both scripts were consumed, and every
    caller produced a called-move line in gen3's bare `[from]` form."""
    p1, p2 = played
    assert p1.battle is not None and p1.battle.finished and p2.battle.finished
    assert not p1._script and not p2._script, (
        f"a scripted action never became legal — p1 left {p1._script}, p2 left {p2._script}")
    log = "\n".join(_lines(p1))
    for needle in ("|[from] Metronome", "|[from] Assist", "|[from] Nature Power"):
        assert needle in log, f"the scripted battle never produced {needle!r}:\n{log}"
    assert p1.decisions >= 11 and p2.decisions >= 11


def test_no_called_move_was_revealed_as_the_callers_own(played):
    """The opponent's reading of p1's team: each mon's revealed moves are a subset of its REAL
    moveset (a called Thunderbolt / Explosion / Swift is not Clefable's / Delcatty's / Ludicolo's)."""
    _, p2 = played
    real = {"clefable": {"metronome", "seismictoss", "softboiled", "calmmind"},
            "delcatty": {"assist", "doubleedge", "healbell", "wish"},
            "ludicolo": {"naturepower", "surf", "gigadrain", "leechseed"}}
    for species, seen in p2.opp_moves.items():
        assert seen <= real[species], f"{species}: {sorted(seen - real[species])} are not its moves"
    assert {"metronome", "assist", "naturepower"} <= set().union(*p2.opp_moves.values())


def test_the_opponents_caller_pp_equals_the_sims(played):
    """p2's sighting count of p1's caller PP == the PP in p1's own `|request|` at the same turn —
    the sim's word, and the Pressure fact: Zapdos's Pressure charged nothing for a called move."""
    p1, p2 = played
    common = sorted(set(p1.own_pp) & set(p2.opp_pp))
    assert len(common) >= 6, f"too few matched (turn, caller) rows: {common}"
    diff = {k: (p1.own_pp[k], p2.opp_pp[k]) for k in common if p1.own_pp[k] != p2.opp_pp[k]}
    assert not diff, f"(turn, caller) -> (sim, poke-env reading): {diff}"
