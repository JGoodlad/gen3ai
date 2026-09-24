"""Pins for the fork's READING fixes found by the TRUTH AUDIT (`designs/rust_sim/one_sided_view.md`
§4b, R1–R3). Each is a sim fact poke-env read wrong, established against the engine and the pinned
Showdown source; each test FAILS on the upstream behaviour.

The end-to-end gate for all three is the Rust Core parity harness's slice V
(`agents/battle/rust_core_parity_views.py`), which compares the reading against the SIM at every
decision; these are the constructed, one-rule-per-test pins.
"""
from poke_env.battle.battle import Battle


def _battle() -> Battle:
    b = Battle("tag", "A", None, gen=3)  # type: ignore[arg-type]
    b.player_role = "p1"
    return b


def _feed(battle: Battle, *lines: str) -> None:
    for line in lines:
        battle.parse_message(line.split("|"))


def test_r1_a_new_status_starts_its_own_count():
    """A Rest taken while badly poisoned: the sim's sleep is brand new, so its count is 0 — the
    upstream setter carried the toxic count (2) into the sleep."""
    b = _battle()
    _feed(b, "|switch|p2a: Suicune|Suicune|100/100", "|-status|p2a: Suicune|tox",
          "|turn|2", "|turn|3")
    mon = b.opponent_active_pokemon
    assert mon.status_counter == 2, "fixture: two toxic turns counted"
    _feed(b, "|-status|p2a: Suicune|slp|[from] move: Rest")
    assert mon.status_counter == 0
    _feed(b, "|cant|p2a: Suicune|slp")
    assert mon.status_counter == 1


def test_r1_the_same_status_restated_keeps_its_count():
    b = _battle()
    _feed(b, "|switch|p2a: Snorlax|Snorlax, M|100/100", "|-status|p2a: Snorlax|slp",
          "|cant|p2a: Snorlax|slp")
    mon = b.opponent_active_pokemon
    mon.status = "slp"
    assert mon.status_counter == 1


def test_r2_psych_up_copies_the_TARGETS_stages_onto_the_USER():
    """`|-copyboost|USER|TARGET|[from] move: Psych Up` — the sim: `source.boosts = target.boosts`.
    Upstream wrote the user's (empty) stages over the target's."""
    b = _battle()
    _feed(b, "|switch|p1a: Suicune|Suicune|100/100", "|switch|p2a: Regice|Regice|100/100",
          "|-boost|p1a: Suicune|spa|2", "|-boost|p1a: Suicune|spd|2",
          "|-copyboost|p2a: Regice|p1a: Suicune|[from] move: Psych Up")
    assert b.active_pokemon.boosts["spa"] == 2, "the target keeps its own stages"
    assert b.opponent_active_pokemon.boosts["spa"] == 2, "the user takes the target's stages"
    assert b.opponent_active_pokemon.boosts["spd"] == 2


def test_r3_our_active_pp_is_the_requests():
    """The request's `pp` is the sim's word. A Pressure the client cannot infer costs 2 in the sim
    and 1 in the sighting count; upstream never re-read it."""
    b = _battle()
    side = {"name": "A", "id": "p1", "pokemon": [{
        "ident": "p1: Snorlax", "details": "Snorlax, M", "condition": "400/400", "active": True,
        "stats": {"atk": 250, "def": 200, "spa": 150, "spd": 250, "spe": 100},
        "moves": ["selfdestruct", "bodyslam", "rest", "curse"], "baseAbility": "immunity",
        "item": "leftovers", "pokeball": "pokeball"}]}
    moves = [{"move": "Self-Destruct", "id": "selfdestruct", "pp": 8, "maxpp": 8,
              "target": "allAdjacent", "disabled": False},
             {"move": "Body Slam", "id": "bodyslam", "pp": 24, "maxpp": 24, "target": "normal",
              "disabled": False}]
    b.parse_request({"active": [{"moves": moves}], "side": side})
    b.parse_message("|move|p1a: Snorlax|Self-Destruct|p2a: Aerodactyl".split("|"))
    assert b.active_pokemon.moves["selfdestruct"].current_pp == 7, "fixture: counted 1"
    moves[0]["pp"] = 6                     # the sim charged Pressure: 2
    b.parse_request({"active": [{"moves": moves}], "side": side})
    assert b.active_pokemon.moves["selfdestruct"].current_pp == 6
