"""gen3_obs_facts_v1 — the OBS-FACTS block's LAYOUT (what the model slices) and its read-back (`describe`).

The per-fact tests that fed constructed protocol to a ``Gen3Battle`` and the Python fold are deleted with
the Python encoder's encode path (T27 P6 slice 6d-2). The block is the Rust encoder's
(``src/rust_sim/src/encoder/facts.rs``), held to the ENGINE by ``src/rust_sim/tests/obs_facts_truth_test.rs``
and frozen by the obs golden (``agents.training.golden_obs_core``).
"""
from __future__ import annotations

import numpy as np

from agents.observation import constants as C
from agents.observation.obs_facts import describe
from agents.observation.state_encoder import Gen3ObservationEncoder


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


def test_describe_reads_back_each_sub_block():
    """`describe` decodes the cells at the declared offsets: a seen row, the choice evidence, one volatile
    window (elapsed / min / max, in turns) and one screen's turns left."""
    block = np.zeros(C.OBS_FACTS_DIM, dtype=np.float32)
    block[C.FACTS_SEEN_OFFSET + 1 * C.FACTS_SEEN_ROW_DIM + 0] = 1.0           # our mon 1 was on the field
    block[C.FACTS_CHOICE_OFFSET:C.FACTS_CHOICE_OFFSET + 4] = (1.0, 0.0, 309.0, 0.5)
    taunt = C.FACTS_VOL_EFFECTS.index("taunt")
    o = C.FACTS_VOL_OFFSET + 1 * C.FACTS_VOL_SIDE_DIM + taunt * C.FACTS_VOL_CELL_DIM
    block[o:o + 3] = np.array([1, 1, 1], dtype=np.float32) / C.FACTS_TURN_NORM
    reflect = C.FACTS_SCREENS.index("reflect")
    block[C.FACTS_SCREENS_OFFSET + reflect] = 3.0 / C.FACTS_SCREEN_TURNS

    d = describe(block)
    assert d["seen"][1][0] == 1 and d["seen"][0] == [0] * C.FACTS_SEEN_ROW_DIM
    assert d["choice"] == {"not_locked_item": True, "not_locked_moves": False, "first_move_num": 309, "run": 0.5}
    assert d["vol"] == {"opp_taunt": (1, 1, 1)}
    assert d["screens"] == {"ours_reflect": 3}
