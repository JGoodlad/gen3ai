"""LiveView tests: faithful current-board snapshot, reveal-gated, NO history.

The contract under test:
  1. LiveView mirrors poke-env's CURRENT state (we don't reimplement the tracker).
  2. Opponent fields are reveal-gated (unknown item -> None, unrevealed moves absent).
  3. LivePokemon carries NO past-turn fields — last_move et al. are physically absent, so
     a consumer is forced to the event log for history.
"""

import dataclasses
import logging

import pytest

from agents.battle.gen3_battle import Gen3Battle
from agents.battle.live_view import LivePokemon, LiveView, LiveWeather

LOG = logging.getLogger("liveview-test")


def _battle(extra):
    b = Gen3Battle("battle-gen3ou-lv", "p1user", LOG, gen=3)
    base = [
        ["", "player", "p1", "p1user", "", ""],
        ["", "player", "p2", "p2user", "", ""],
        ["", "teamsize", "p1", "6"],
        ["", "teamsize", "p2", "6"],
        ["", "gen", "3"],
        ["", "start"],
        ["", "switch", "p1a: Zappy", "Zapdos, L100", "100/100"],
        ["", "switch", "p2a: Tyra", "Tyranitar, L100, M", "100/100"],
        ["", "turn", "1"],
    ]
    for line in base + extra:
        b.parse_message(line)
    return b


# ───────────────────────────── faithfulness ────────────────────────────────
def test_mirrors_current_state():
    b = _battle([
        ["", "-weather", "Sandstorm"],
        ["", "-sidestart", "p2: p2user", "Spikes"],
        ["", "move", "p1a: Zappy", "Thunderbolt", "p2a: Tyra"],
        ["", "-damage", "p2a: Tyra", "52/100"],
        ["", "-status", "p2a: Tyra", "par"],
        ["", "-start", "p2a: Tyra", "Leech Seed"],
        ["", "turn", "2"],
    ])
    lv = b.live_view()

    assert lv.turn == 2
    # bare |-weather|Sandstorm is move-sourced ⇒ finite, not permanent
    assert lv.weather.weather == "sandstorm"
    assert lv.weather.is_permanent is False

    opp = lv.opp.active
    assert opp.species == "tyranitar"
    assert opp.hp_fraction == pytest.approx(0.52)
    assert opp.status == "par"
    assert "leechseed" in opp.volatiles  # Showdown id form (no separators)
    assert opp.has_volatile("Leech Seed")  # accepts display / enum / id forms
    assert opp.has_volatile("LEECH_SEED")
    assert lv.opp.side_conditions == {"spikes": 1}

    ours = lv.ours.active
    assert ours.species == "zapdos"
    assert ours.hp_fraction == pytest.approx(1.0)
    assert ours.status is None

    # current state matches poke-env exactly (we didn't reimplement the tracker)
    assert opp.hp_fraction == b.opponent_active_pokemon.current_hp_fraction
    assert ours.species == b.active_pokemon.species


def test_team_sizes_and_reveal_counts():
    b = _battle([["", "turn", "2"]])
    lv = b.live_view()
    # both declared 6; opponent has revealed only its lead so far
    assert lv.ours.team_size == 6
    assert lv.opp.team_size == 6
    assert lv.opp.revealed_count == 1
    assert lv.opp.remaining == 1  # 1 revealed, none fainted


def test_boosts_are_current_not_historical():
    b = _battle([
        ["", "move", "p1a: Zappy", "Agility", "p1a: Zappy"],
        ["", "-boost", "p1a: Zappy", "spe", "2"],
        ["", "turn", "2"],
    ])
    lv = b.live_view()
    assert lv.ours.active.boosts == {"spe": 2}   # current stage, not a per-turn delta


# ───────────────────────────── reveal gating ───────────────────────────────
def test_opponent_item_hidden_until_revealed():
    b = _battle([["", "turn", "2"]])
    lv = b.live_view()
    # Tyranitar's item not disclosed -> None (the 'unknown_item' sentinel is hidden)
    assert lv.opp.active.item is None


def test_opponent_revealed_item_shows():
    b = _battle([
        ["", "-enditem", "p2a: Tyra", "Salac Berry", "[eat]"],
        ["", "turn", "2"],
    ])
    # after a reveal poke-env knows the item identity (now consumed); the live view
    # surfaces the consumed-item knowledge is on the event log, but the *held* item is
    # gone, so live item is None — and the ENDITEM event carries the identity.
    lv = b.live_view()
    assert lv.opp.active.item is None
    from agents.battle.battle_event import EventKind
    end = next(e for e in b.events if e.kind is EventKind.ENDITEM)
    assert end.item == "salacberry"


def test_opponent_moves_are_only_revealed_ones():
    b = _battle([
        ["", "move", "p2a: Tyra", "Rock Slide", "p1a: Zappy"],
        ["", "-damage", "p1a: Zappy", "70/100"],
        ["", "turn", "2"],
    ])
    lv = b.live_view()
    # only the move it has actually used is known
    assert lv.opp.active.moves == ("rockslide",)


# ───────────────────────────── NO history (the boundary) ────────────────────
_FORBIDDEN_HISTORY_FIELDS = [
    "last_move", "last_cant_reason", "first_turn", "protect_counter",
    "must_recharge", "preparing", "preparing_move", "active_turns",
]


@pytest.mark.parametrize("attr", _FORBIDDEN_HISTORY_FIELDS)
def test_livepokemon_has_no_history_fields(attr):
    b = _battle([
        ["", "move", "p1a: Zappy", "Thunderbolt", "p2a: Tyra"],
        ["", "-damage", "p2a: Tyra", "52/100"],
        ["", "turn", "2"],
    ])
    mon = b.live_view().ours.active
    assert not hasattr(mon, attr), (
        f"LivePokemon must not expose past-turn field {attr!r} — history belongs to the "
        f"event log / TurnView, not the current-board view"
    )


def test_livepokemon_fields_are_exactly_the_minimal_set():
    """Pin the field set so nobody silently grows LivePokemon into a state grab-bag."""
    names = {f.name for f in dataclasses.fields(LivePokemon)}
    assert names == {
        "species", "active", "fainted", "revealed", "hp_fraction", "status",
        "types", "moves", "item", "ability", "boosts", "volatiles",
    }


def test_live_view_is_immutable():
    b = _battle([["", "turn", "2"]])
    mon = b.live_view().ours.active
    with pytest.raises(dataclasses.FrozenInstanceError):
        mon.hp_fraction = 0.5


def test_active_slot_can_hold_a_just_fainted_mon():
    """After a faint and before replacement, poke-env's active accessor still points
    at the fainted mon. The live view must mirror that faithfully (active set, fainted
    True) rather than dropping the active to None."""
    b = _battle([
        ["", "move", "p1a: Zappy", "Thunderbolt", "p2a: Tyra"],
        ["", "-damage", "p2a: Tyra", "0 fnt"],
        ["", "faint", "p2a: Tyra"],
    ])
    lv = b.live_view()
    assert lv.opp.active is not None
    assert lv.opp.active.species == "tyranitar"
    assert lv.opp.active.fainted is True
    assert lv.opp.active.hp_fraction == 0.0
    # and it agrees with poke-env's own accessor (no reimplementation)
    assert lv.opp.active.species == b.opponent_active_pokemon.species


def test_no_pokemon_backreference_leaks_history():
    """A LivePokemon must hold only primitives — no stashed Pokemon object through which a
    consumer could reach last_move. Every field value is a primitive / container."""
    b = _battle([["", "turn", "2"]])
    mon = b.live_view().opp.active
    for f in dataclasses.fields(mon):
        val = getattr(mon, f.name)
        assert isinstance(val, (str, bool, float, int, tuple, dict, type(None))), (
            f"LivePokemon.{f.name} is a {type(val)} — only primitives allowed so no "
            f"history can leak through a back-reference"
        )


# --------------------------------------------------------------------------- #
# Weather — folded from the WEATHER event log (cause-aware), never guessed.    #
# --------------------------------------------------------------------------- #
def test_weather_move_sourced_is_finite():
    b = _battle([["", "-weather", "RainDance"], ["", "turn", "2"]])
    w = b.live_view().weather
    assert w.weather == "raindance"
    assert w.is_permanent is False
    assert w.turns_active == 1          # set on turn 1, now turn 2
    assert w.turns_remaining == 4       # gen3 move weather = 5 turns total


def test_weather_ability_sourced_is_permanent():
    b = _battle([
        ["", "-weather", "Sandstorm", "[from] ability: Sand Stream", "[of] p1a: Tyra"],
        ["", "turn", "2"], ["", "turn", "3"],
    ])
    w = b.live_view().weather
    assert w.weather == "sandstorm"
    assert w.is_permanent is True
    assert w.turns_remaining is None    # permanent ⇒ no finite timer (honest None)


def test_weather_upkeep_does_not_reset_timer():
    b = _battle([
        ["", "-weather", "RainDance"],            # set turn 1
        ["", "turn", "2"],
        ["", "-weather", "RainDance", "[upkeep]"],  # tick — must NOT reset start
        ["", "turn", "3"],
    ])
    w = b.live_view().weather
    assert w.turns_active == 2           # still measured from the turn-1 set
    assert w.turns_remaining == 3


def test_weather_cleared():
    b = _battle([
        ["", "-weather", "RainDance"], ["", "turn", "2"],
        ["", "-weather", "none"], ["", "turn", "3"],
    ])
    w = b.live_view().weather
    assert w.weather is None
    assert w.turns_remaining is None


def test_no_weather_is_empty_not_crash():
    w = _battle([["", "turn", "2"]]).live_view().weather
    assert isinstance(w, LiveWeather)
    assert w.weather is None and w.is_permanent is False
