"""gen3_obs_facts_v1 — the OBS-FACTS block on CONSTRUCTED protocol sequences.

Each test feeds one side's protocol text to a ``Gen3Battle`` (the live player's dispatch,
``offline_feed``), folds the decision window through the ``EventWindowTracker`` the encoder reads,
and asserts the encoded fact. Every assertion FAILS on revert of the code it names (the reading's
public flags, the residual phase, the stint fold, the Encore / Disable adjustment, the encoder).
The Rust twin is held byte-equal by slice O and to the ENGINE by
``src/rust_sim/tests/obs_facts_truth_test.rs``.
"""
from __future__ import annotations

import json
from typing import List, Optional

import numpy as np
import pytest

from agents.battle.offline_feed import feed_line, new_battle
from agents.observation import constants as C
from agents.observation.obs_facts import encode_obs_facts
from agents.observation.state_encoder import Gen3ObservationEncoder
from agents.training.event_window_tracker import EventWindowTracker

TEAM = ("Metagross|||clearbody|meteormash,earthquake,explosion,agility|Adamant|252,252,,,4,|||||]"
        "Suicune|||pressure|calmmind,surf,rest,icebeam|Bold|252,,252,,4,|||||")


def _req(active: str = "Metagross") -> str:
    am, as_ = ("true", "false") if active == "Metagross" else ("false", "true")
    mv = [{"move": "Meteor Mash", "id": "meteormash", "pp": 16, "maxpp": 16, "target": "normal",
           "disabled": False}]
    return ("|request|{\"active\":[" + json.dumps({"moves": mv}, separators=(",", ":")) + "],"
            '"side":{"name":"me","id":"p1","pokemon":['
            '{"ident":"p1: Metagross","details":"Metagross","condition":"301/301","active":' + am
            + ',"stats":{"atk":405,"def":296,"spa":203,"spd":216,"spe":214},"moves":["meteormash",'
            '"earthquake","explosion","agility"],"baseAbility":"clearbody","item":"leftovers",'
            '"pokeball":"pokeball"},'
            '{"ident":"p1: Suicune","details":"Suicune","condition":"341/341","active":' + as_
            + ',"stats":{"atk":139,"def":266,"spa":260,"spd":308,"spe":213},"moves":["calmmind","surf",'
            '"rest","icebeam"],"baseAbility":"pressure","item":"leftovers","pokeball":"pokeball"}]}}')


PREFIX = ["|player|p1|me||", "|player|p2|foe||", "|teamsize|p1|2", "|teamsize|p2|3", "|gen|3",
          "|tier|[Gen 3] OU", "|start"]
BASE = PREFIX + [_req(), "|switch|p1a: Metagross|Metagross|301/301",
                 "|switch|p2a: Snorlax|Snorlax, M|100/100", "|turn|1"]


def _battle(lines: List[str]):
    b = new_battle("p1", {"p1": "me", "p2": "foe"}, packed_team=TEAM)
    for ln in lines:
        feed_line(b, ln)
    return b


def _facts(lines: List[str], window: Optional[EventWindowTracker] = None) -> np.ndarray:
    """The OBS-FACTS block at the decision the lines end at (the window folds every event)."""
    b = _battle(lines)
    live = b.live_view()
    if window is None:
        window = EventWindowTracker()
        window.update(live.turn, b.events, None, None)
    vec = np.zeros(C.OBS_FACTS_DIM, dtype=np.float32)
    our = [m.species for m in Gen3ObservationEncoder.get_team_list(b, is_opponent=False) if m is not None]
    encode_obs_facts(vec, 0, live, our, window)
    return vec


def _seen_row(f: np.ndarray, i: int) -> List[int]:
    o = C.FACTS_SEEN_OFFSET + i * C.FACTS_SEEN_ROW_DIM
    return [int(x) for x in f[o:o + C.FACTS_SEEN_ROW_DIM]]


def _choice(f: np.ndarray) -> List[float]:
    return [float(x) for x in f[C.FACTS_CHOICE_OFFSET:C.FACTS_CHOICE_OFFSET + C.FACTS_CHOICE_DIM]]


def _vol(f: np.ndarray, side: int, effect: str):
    j = C.FACTS_VOL_EFFECTS.index(effect)
    o = C.FACTS_VOL_OFFSET + side * C.FACTS_VOL_SIDE_DIM + j * C.FACTS_VOL_CELL_DIM
    return tuple(int(round(float(f[o + q]) * C.FACTS_TURN_NORM)) for q in range(3))


def _screen(f: np.ndarray, side: int, name: str) -> int:
    j = C.FACTS_SCREENS.index(name)
    return int(round(float(f[C.FACTS_SCREENS_OFFSET + side * len(C.FACTS_SCREENS) + j]) * C.FACTS_SCREEN_TURNS))


def _num(move: str) -> float:
    from agents.gen3_data import moves
    return float(moves.get(move).num)


# ----------------------------------------------------------------------------- the layout
def test_the_sub_blocks_tile_the_block_and_it_is_NOT_yet_in_the_observation():
    from agents.observation.state_encoder import load_mappings
    enc = Gen3ObservationEncoder(load_mappings())
    # the append lands at the X5 adoption version break (`obs-facts-append`): the row is unchanged
    assert enc.dimension == C.OFFSET_EVENT_WINDOW + C.EVENT_WINDOW_DIM
    assert "obs_facts" not in enc.get_layout()
    assert C.FACTS_SEEN_OFFSET == 0 and C.FACTS_CHOICE_OFFSET == C.FACTS_SEEN_DIM
    assert C.FACTS_VOL_OFFSET == C.FACTS_CHOICE_OFFSET + C.FACTS_CHOICE_DIM
    assert C.FACTS_SCREENS_OFFSET == C.FACTS_VOL_OFFSET + C.FACTS_VOL_DIM
    assert C.OBS_FACTS_DIM == C.FACTS_SCREENS_OFFSET + C.FACTS_SCREENS_DIM


# ----------------------------------------------------------------------------- 1. SEEN (E1)
def test_seen_a_request_reveals_nothing_to_the_opponent():
    f = _facts(BASE)
    # Metagross is on the field (revealed); its four moves / item / ability are only in OUR request
    assert _seen_row(f, 0) == [1, 0, 0, 0, 0, 0, 0]
    assert _seen_row(f, 1) == [0, 0, 0, 0, 0, 0, 0], "a benched mon has never been shown"


def test_seen_a_used_move_is_seen_in_the_slot_move_order():
    f = _facts(BASE + ["|move|p1a: Metagross|Meteor Mash|p2a: Snorlax"])
    # the per-mon slot's order is sorted by id: agility, earthquake, explosion, meteormash
    assert _seen_row(f, 0) == [1, 0, 0, 0, 1, 0, 0]
    b = _battle(BASE + ["|move|p1a: Metagross|Meteor Mash|p2a: Snorlax"])
    assert [m.id for m in b.live_view().ours.get("metagross").moves] == \
        ["agility", "earthquake", "explosion", "meteormash"]


def test_seen_a_called_move_that_is_not_ours_reveals_nothing():
    # Metronome's call is not a reveal (poke-env: reveal=False); the caller is not in our set either
    f = _facts(BASE + ["|move|p1a: Metagross|Metronome|p1a: Metagross",
                       "|move|p1a: Metagross|Earthquake|p2a: Snorlax|[from]move: Metronome"])
    assert _seen_row(f, 0)[1:5] == [0, 0, 0, 0]


def test_seen_item_and_ability_are_public_only_from_a_protocol_line():
    f = _facts(BASE + ["|-heal|p1a: Metagross|301/301|[from] item: Leftovers",
                       "|-immune|p1a: Metagross|[from] ability: Clear Body"])
    assert _seen_row(f, 0)[5:] == [1, 1]
    f = _facts(BASE + ["|switch|p1a: Suicune|Suicune|341/341",
                       "|-enditem|p1a: Suicune|Leftovers|[from] move: Knock Off|[of] p2a: Snorlax"])
    assert _seen_row(f, 1) == [1, 0, 0, 0, 0, 1, 0], "a Knock Off reveals the (now empty) hand"


def test_seen_a_trick_makes_both_items_public():
    b = _battle(BASE + ["|-activate|p1a: Metagross|move: Trick|[of] p2a: Snorlax"])
    live = b.live_view()
    assert live.ours.get("metagross").item_public and live.opp.get("snorlax").item_public


# ------------------------------------------------------------------- the reading's residual phase
def test_residual_done_is_the_upkeep_of_the_current_turn():
    assert not _battle(BASE).live_view().residual_done
    assert _battle(BASE + ["|", "|upkeep"]).live_view().residual_done
    assert not _battle(BASE + ["|", "|upkeep", "|turn|2"]).live_view().residual_done


# ----------------------------------------------------------------------------- 2. CHOICE
def test_choice_first_move_and_run_are_the_stint_of_the_opponent_active():
    lines = BASE + ["|move|p2a: Snorlax|Body Slam|p1a: Metagross", "|turn|2",
                    "|move|p2a: Snorlax|Body Slam|p1a: Metagross", "|turn|3"]
    c = _choice(_facts(lines))
    assert c[0] == 0.0 and c[1] == 0.0, "no proof: one move, item unknown"
    assert c[2] == _num("bodyslam")
    from agents.observation.assembler import SAT_LUT
    assert c[3] == np.float32(SAT_LUT[2])


def test_choice_not_locked_proof_by_two_distinct_moves():
    lines = BASE + ["|move|p2a: Snorlax|Body Slam|p1a: Metagross", "|turn|2",
                    "|move|p2a: Snorlax|Earthquake|p1a: Metagross", "|turn|3"]
    c = _choice(_facts(lines))
    assert c[1] == 1.0
    assert c[2] == _num("bodyslam"), "the FIRST move of the stint is kept"


def test_choice_not_locked_proof_by_a_revealed_non_choice_item():
    c = _choice(_facts(BASE + ["|-heal|p2a: Snorlax|100/100|[from] item: Leftovers"]))
    assert c[0] == 1.0
    c = _choice(_facts(BASE + ["|-item|p2a: Snorlax|Choice Band"]))
    assert c[0] == 0.0, "a revealed Choice Band is no proof"
    c = _choice(_facts(BASE + ["|-enditem|p2a: Snorlax|Choice Band|[from] move: Knock Off|[of] p1a: Metagross"]))
    assert c[0] == 1.0, "a knocked-off Choice Band frees the lock (choicelock.onBeforeMove)"


def test_choice_the_stint_resets_on_a_switch():
    lines = BASE + ["|move|p2a: Snorlax|Body Slam|p1a: Metagross", "|turn|2",
                    "|move|p2a: Snorlax|Earthquake|p1a: Metagross", "|turn|3",
                    "|switch|p2a: Zapdos|Zapdos|100/100", "|turn|4",
                    "|move|p2a: Zapdos|Thunderbolt|p1a: Metagross", "|turn|5"]
    c = _choice(_facts(lines))
    assert c[1] == 0.0 and c[2] == _num("thunderbolt")


def test_choice_struggle_and_a_called_move_are_not_a_selection():
    lines = BASE + ["|move|p2a: Snorlax|Sleep Talk|p2a: Snorlax",
                    "|move|p2a: Snorlax|Body Slam|p1a: Metagross|[from]move: Sleep Talk", "|turn|2",
                    "|move|p2a: Snorlax|Struggle|p1a: Metagross", "|turn|3"]
    c = _choice(_facts(lines))
    assert c[1] == 0.0 and c[2] == _num("sleeptalk")


# ----------------------------------------------------------------------------- 3. VOL
def test_taunt_is_two_residuals_fixed():
    f = _facts(BASE + ["|move|p1a: Metagross|Taunt|p2a: Snorlax", "|-start|p2a: Snorlax|move: Taunt",
                       "|", "|upkeep", "|turn|2"])
    assert _vol(f, 1, "taunt") == (1, 1, 1)


def test_encore_bounds_take_the_adjustment_when_the_target_had_already_acted():
    # the target moved FIRST, then was encored: Showdown adds one (encore.onStart !willMove)
    acted = BASE + ["|move|p2a: Snorlax|Body Slam|p1a: Metagross",
                    "|move|p1a: Metagross|Encore|p2a: Snorlax", "|-start|p2a: Snorlax|Encore",
                    "|", "|upkeep", "|turn|2"]
    assert _vol(_facts(acted), 1, "encore") == (1, 3, 6)
    # the encorer moved first: no adjustment, 3..6 total, one residual elapsed
    before = BASE + ["|move|p1a: Metagross|Encore|p2a: Snorlax", "|-start|p2a: Snorlax|Encore",
                     "|move|p2a: Snorlax|Body Slam|p1a: Metagross", "|", "|upkeep", "|turn|2"]
    assert _vol(_facts(before), 1, "encore") == (1, 2, 5)


def test_a_replacement_decision_counts_the_residual_that_just_ran():
    lines = BASE + ["|move|p1a: Metagross|Taunt|p2a: Snorlax", "|-start|p2a: Snorlax|move: Taunt",
                    "|", "|upkeep", "|turn|2",
                    "|-start|p1a: Metagross|Uproar", "|", "|upkeep"]
    # post-residual (|upkeep| read, no |turn| yet): our Uproar has 1 residual behind it
    assert _vol(_facts(lines), 0, "uproar") == (1, 1, 4)


def test_no_window_takes_the_union_of_both_encore_cases():
    b = _battle(BASE + ["|-start|p2a: Snorlax|Encore", "|", "|upkeep", "|turn|2"])
    vec = np.zeros(C.OBS_FACTS_DIM, dtype=np.float32)
    encode_obs_facts(vec, 0, b.live_view(), ["metagross", "suicune"], None)
    assert _vol(vec, 1, "encore") == (1, 2, 6)


# ----------------------------------------------------------------------------- 4. SCREENS
def test_screens_count_residuals_since_the_start():
    lines = BASE + ["|move|p2a: Snorlax|Reflect|p2a: Snorlax", "|-sidestart|p2: foe|Reflect",
                    "|", "|upkeep", "|turn|2"]
    assert _screen(_facts(lines), 1, "reflect") == 4
    assert _screen(_facts(lines + ["|", "|upkeep", "|turn|3"]), 1, "reflect") == 3
    # at a post-residual decision the residual of the current turn is already spent
    assert _screen(_facts(lines + ["|", "|upkeep"]), 1, "reflect") == 3
    assert _screen(_facts(lines), 0, "reflect") == 0


@pytest.mark.parametrize("cond,name", [("Light Screen", "light_screen"), ("Safeguard", "safeguard"),
                                       ("Mist", "mist")])
def test_every_screen_is_five_residuals(cond, name):
    lines = BASE + [f"|-sidestart|p1: me|move: {cond}", "|", "|upkeep", "|turn|2"]
    assert _screen(_facts(lines), 0, name) == 4
