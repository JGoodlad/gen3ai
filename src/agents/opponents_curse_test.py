"""Curse as a SETUP move in the four scripted setup bots (owner 2026-09-29: "allow Curse").

Gen 3 Curse (Showdown `data/mods/gen4/moves.ts`, inherited by gen 3): a NON-Ghost user gets
+1 Atk / +1 Def / -1 Spe on itself; a GHOST user loses half its max HP to curse the target. poke-env's
`Move("curse")` says neither (target NORMAL, no boosts), so before this change no setup step could
pick it. These tests use REAL `Move` and `Pokemon` objects — a mock `Move` is what once hid F-LF-1 —
and each fails on a revert of `baselines.self_setup_boosts`: the non-Ghost cases stop choosing Curse,
and a version that ignored the Ghost check would pick Curse for Gengar.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType
from poke_env.player.baselines import (
    CURSE_NON_GHOST_BOOSTS,
    SimpleHeuristicsPlayer,
    self_setup_boosts,
)

from agents.opponents import (
    Gen3HeuristicV2Player,
    Gen3SetupSweepPlayer,
    Gen3SetupSweepV2Player,
)


def _mon(species: str) -> Pokemon:
    p = Pokemon(gen=3, species=species)
    p._max_hp = p._current_hp = 300   # full HP: every setup step's HP gate is open
    return p


def _battle(active: Pokemon, opponent: Pokemon, moves):
    b = MagicMock()
    b.active_pokemon = active
    b.opponent_active_pokemon = opponent
    b.available_moves = list(moves)
    b.available_switches = []
    b.force_switch = False
    b.team = {"p2: a": active}
    b.opponent_team = {"p1: o": opponent}
    b.side_conditions = {}
    b.opponent_side_conditions = {}
    return b


def _player(cls):
    with patch.object(cls, "__init__", lambda self, **kw: None):
        p = cls.__new__(cls)
    p.choose_random_move = MagicMock(side_effect=AssertionError("fell through to a random move"))
    return p


def _choose(bot: str, battle) -> str:
    if bot == "heuristic":
        order, _ = SimpleHeuristicsPlayer.choose_singles_move(battle)
    else:
        cls = {"heuristic2": Gen3HeuristicV2Player, "setup_sweep": Gen3SetupSweepPlayer,
               "setup_sweep_v2": Gen3SetupSweepV2Player}[bot]
        order = _player(cls).choose_move(battle)
    return order.order.id


BOTS = ("heuristic", "heuristic2", "setup_sweep", "setup_sweep_v2")

# Heracross (Bug/Fighting) into Tyranitar (Rock/Dark): Fighting is 4x, Rock is 1x back, Heracross is
# faster — a winning matchup. Gengar (Ghost/Poison) into Slowbro (Water/Psychic): 2x each way,
# Gengar far faster — also winning. Tackle is the only attack: no KO step fires first.
NON_GHOST = ("heracross", "tyranitar")
GHOST = ("gengar", "slowbro")


def test_the_setup_opportunity_is_real():
    for me, foe in (NON_GHOST, GHOST):
        assert SimpleHeuristicsPlayer._estimate_matchup(_mon(me), _mon(foe)) > 0, (me, foe)
    assert PokemonType.GHOST in _mon("gengar").types
    assert PokemonType.GHOST not in _mon("heracross").types


def test_self_setup_boosts_curse_by_user_type():
    curse = Move("curse", 3)
    assert self_setup_boosts(curse, _mon("heracross")) == {"atk": 1, "def": 1, "spe": -1}
    assert CURSE_NON_GHOST_BOOSTS == {"atk": 1, "def": 1, "spe": -1}
    assert self_setup_boosts(curse, _mon("gengar")) is None
    assert self_setup_boosts(Move("swordsdance", 3), _mon("gengar")) == {"atk": 2}
    assert self_setup_boosts(Move("tackle", 3), _mon("heracross")) is None


@pytest.mark.parametrize("bot", BOTS)
def test_a_non_ghost_sets_up_with_curse(bot):
    me, foe = NON_GHOST
    battle = _battle(_mon(me), _mon(foe), [Move("tackle", 3), Move("curse", 3)])
    assert _choose(bot, battle) == "curse"


@pytest.mark.parametrize("bot", BOTS)
def test_a_ghost_never_curses_as_setup(bot):
    me, foe = GHOST
    battle = _battle(_mon(me), _mon(foe), [Move("tackle", 3), Move("curse", 3)])
    assert _choose(bot, battle) != "curse"
    # ...and not because the opportunity was shut: the same board with Swords Dance sets up.
    battle = _battle(_mon(me), _mon(foe), [Move("tackle", 3), Move("curse", 3), Move("swordsdance", 3)])
    assert _choose(bot, battle) == "swordsdance"


@pytest.mark.parametrize("bot", BOTS)
def test_curse_stops_when_its_raised_stats_are_capped(bot):
    me, foe = NON_GHOST
    active = _mon(me)
    # heuristic / heuristic2 need a RAISED stat (atk, def) below +6; the sweepers an ATK/SPA/SPE one
    # the move raises (only atk) below +6, with the offensive total (atk+spa+spe) under the cap 4.
    active._boosts.update({"atk": 6, "def": 6} if bot.startswith("heuristic") else {"atk": 6, "spe": -3})
    battle = _battle(active, _mon(foe), [Move("tackle", 3), Move("curse", 3)])
    assert _choose(bot, battle) != "curse"
