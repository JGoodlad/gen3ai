"""Unit tests for the spectator-log → decision reader's bug-prone pure logic.

Full end-to-end reconstruction is exercised against the real replay corpus by the
human-agreement probe; here we lock the action-mapping / species-parsing / legal-set
slot logic that a mask/label off-by-one would silently corrupt.
"""


from agents.action.constants import MOVE_START
from agents.battle.live_view import LegalActions, LegalMove, LegalSwitch
from agents.bc.log_reader import (
    SpectatorLogReader,
    _resolve_player_roles,
    _species_of,
    _split_lines,
)


def _legal(move_ids, switch_pairs):
    return LegalActions(
        move_slots=tuple(LegalMove(id=m, current_pp=1, max_pp=1, disabled=False, target=None)
                         for m in move_ids),
        switches=tuple(LegalSwitch(species=s, slot=i) for s, i in switch_pairs),
        force_switch=False, trapped=False, maybe_trapped=False, wait=False,
        struggle=False, last_request=None,
    )


def test_species_of_strips_gender_and_level():
    assert _species_of("Skarmory, F") == "skarmory"
    assert _species_of("Tyranitar, L100, M") == "tyranitar"
    assert _species_of("Zapdos") == "zapdos"


def test_resolve_player_roles_takes_first_rated_occurrence():
    lines = _split_lines(
        "|player|p1|Alice|266|1718\n"
        "|player|p2|Bob|lucas|1894\n"
        "|player|p2|\n"            # later rejoin line must not overwrite
    )
    roles = _resolve_player_roles(lines)
    assert roles == {"p1": "Alice", "p2": "Bob"}


def test_map_move_action_to_slot_index():
    legal = _legal(["spikes", "rockslide", "icebeam"], [])
    # protocol move name 'Rock Slide' -> slot 1 -> action MOVE_START+1
    parts = ["", "move", "p2a: Skarmory", "Rock Slide", "p1a: Blissey"]
    idx, kind = SpectatorLogReader._map_action(parts, legal)
    assert kind == "move" and idx == MOVE_START + 1


def test_map_hidden_power_matches_typed_slot():
    legal = _legal(["thunderbolt", "hiddenpowerice"], [])
    parts = ["", "move", "p2a: Zapdos", "Hidden Power", "p1a: Flygon"]
    idx, kind = SpectatorLogReader._map_action(parts, legal)
    assert kind == "move" and idx == MOVE_START + 1


def test_map_switch_to_revealed_mon_uses_team_slot():
    legal = _legal([], [("skarmory", 0), ("snorlax", 3)])
    parts = ["", "switch", "p2a: Snorlax", "Snorlax, M", "100/100"]
    idx, kind = SpectatorLogReader._map_action(parts, legal)
    assert kind == "switch" and idx == 3


def test_map_switch_to_unrevealed_mon_is_unmappable():
    legal = _legal([], [("skarmory", 0)])
    parts = ["", "switch", "p2a: Gengar", "Gengar, M", "100/100"]
    idx, kind = SpectatorLogReader._map_action(parts, legal)
    assert idx is None and kind == "switch_unrevealed"
