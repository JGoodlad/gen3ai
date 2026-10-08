"""gen3_obs_facts_v1 — the OBS-FACTS block on CONSTRUCTED protocol sequences.

Each test feeds one side's protocol text to a ``Gen3Battle`` (the live player's dispatch,
``offline_feed``), folds the decision window through the ``EventWindowTracker`` the encoder reads,
and asserts the encoded fact. Every assertion FAILS on revert of the code it names (the reading's
public flags, the residual phase, the stint fold, the Encore / Disable adjustment, the encoder).
The block is the observation's LAST block (the X5 version break's part 3; the encoder writes it every
decision). The Rust twin is held byte-equal by slice O (the whole row) and to the ENGINE by
``src/rust_sim/tests/obs_facts_truth_test.rs``.
"""
from __future__ import annotations

import json
from typing import List

import numpy as np

from agents.battle.offline_feed import feed_line, new_battle
from agents.observation import constants as C
from agents.observation.obs_facts import encode_obs_facts
from agents.observation.state_encoder import Gen3ObservationEncoder

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
def test_the_block_closes_the_observation_and_the_sub_blocks_tile_it():
    """gen3_obs_facts_v1 (the X5 version break's part 3): the block is APPENDED after the event window as
    the observation's last block, 2761 -> 2845. Fails on revert of the append (the dim, the offset, the
    layout keys the model's slicer reads)."""
    from agents.observation.state_encoder import load_mappings
    enc = Gen3ObservationEncoder(load_mappings())
    assert C.OFFSET_OBS_FACTS == C.OFFSET_EVENT_WINDOW + C.EVENT_WINDOW_DIM == 2761
    assert enc.dimension == C.OFFSET_OBS_FACTS + C.OBS_FACTS_DIM == 2845
    lay = enc.get_layout()
    assert (lay["obs_facts_offset"], lay["obs_facts_dim"]) == (C.OFFSET_OBS_FACTS, C.OBS_FACTS_DIM)
    assert {k: (v["offset"], v["dim"]) for k, v in lay["obs_facts"].items()} == {
        "seen": (C.FACTS_SEEN_OFFSET, C.FACTS_SEEN_DIM), "choice": (C.FACTS_CHOICE_OFFSET, C.FACTS_CHOICE_DIM),
        "vol": (C.FACTS_VOL_OFFSET, C.FACTS_VOL_DIM), "screens": (C.FACTS_SCREENS_OFFSET, C.FACTS_SCREENS_DIM)}
    assert C.FACTS_SEEN_OFFSET == 0 and C.FACTS_CHOICE_OFFSET == C.FACTS_SEEN_DIM
    assert C.FACTS_VOL_OFFSET == C.FACTS_CHOICE_OFFSET + C.FACTS_CHOICE_DIM
    assert C.FACTS_SCREENS_OFFSET == C.FACTS_VOL_OFFSET + C.FACTS_VOL_DIM
    assert C.OBS_FACTS_DIM == C.FACTS_SCREENS_OFFSET + C.FACTS_SCREENS_DIM


_ROW_LINES = BASE + ["|move|p1a: Metagross|Meteor Mash|p2a: Snorlax", "|move|p2a: Snorlax|Reflect|p2a: Snorlax",
                     "|-sidestart|p2: foe|Reflect", "|-start|p2a: Snorlax|Encore", "|", "|upkeep", "|turn|2"]


# ----------------------------------------------------------------------------- 1. SEEN (E1)
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
# ----------------------------------------------------------------------------- 3. VOL
def test_no_window_takes_the_union_of_both_encore_cases():
    b = _battle(BASE + ["|-start|p2a: Snorlax|Encore", "|", "|upkeep", "|turn|2"])
    vec = np.zeros(C.OBS_FACTS_DIM, dtype=np.float32)
    encode_obs_facts(vec, 0, b.live_view(), ["metagross", "suicune"], None)
    assert _vol(vec, 1, "encore") == (1, 2, 6)


# ----------------------------------------------------------------------------- 4. SCREENS
