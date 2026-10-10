"""END TO END: the toxic counter the MODEL reads is the engine's stage over its true cap of 15
(``gen3_toxic_stage_scale_v1``, config v151).

A Blissey (Soft-Boiled + Leftovers) is badly poisoned on turn 1 by a Zapdos that then only Protects; its heal outpaces
the ramp until the 12th tick (hand-checked: HP 64 after tick 12, dead at tick 13). The battle is played by the REAL
Rust core (``core_events --obs --trackers``), and the row the trainee reads at the turn-13 decision carries stage 12:

* the row's toxic cell (``POKEMON_COUNTER_OFFSET + 1``) under the Blissey slot is ``12 / 15`` — it read 1.0 (saturated
  at 8 ticks) before, for every stage from 8 on;
* the model's own decoder reads the NEXT tick off that real row as 13/16: ``min(n + 1, 15) / 16`` is Showdown's
  ``tox.onResidual`` (``if (stage < 15) stage++`` BEFORE the damage ``floor(maxhp/16) * stage``), so the 13th tick costs
  13/16 of max HP, where the saturated cell priced it as 9/16;
* the opponent's view of the same mon carries the same cell (the counter is public).

The Rust twin against the ENGINE's own ``Toxic(n)`` at every decision of both viewers is
``src/rust_sim/tests/obs_stage_truth_test.rs``.
"""
import numpy as np
import pytest
import torch

from agents.battle.core_obs import wrap_row
from agents.battle.core_replay import RecordedBattle, run_core
from agents.model.damage_op_pairwise import toxic_next_tick
from agents.observation import constants as C

pytestmark = [pytest.mark.sim, pytest.mark.integration]

BLISSEY = ("Blissey||leftovers|naturalcure|softboiled,toxic,protect,seismictoss|Calm|252,,,,252,|||||]"
           "Snorlax|||immunity|bodyslam,earthquake,rest,curse|Careful|252,,4,,252|||||")
ZAPDOS = ("Zapdos||leftovers|pressure|toxic,protect,rest,substitute|Modest|252,,,252,4,||,0,,30,,|||]"
          "Snorlax|||immunity|bodyslam,earthquake,rest,curse|Careful|252,,4,,252|||||")
TURNS = 13


def _battle() -> RecordedBattle:
    cmds = []
    for turn in range(1, TURNS + 1):
        cmds += [["p1", "move softboiled"], ["p2", "move toxic" if turn == 1 else "move protect"]]
    return RecordedBattle(label="toxic_stage_12", format_id="gen3customgame", seed="1,2,3,4",
                          p1={"name": "P1", "team": BLISSEY}, p2={"name": "P2", "team": ZAPDOS}, commands=cmds)


def _tox_cells(row: np.ndarray, team_offset: int):
    """``(slot, cell)`` of every mon under a ``tox`` condition one-hot (the last of the 7 condition dims)."""
    out = []
    for s in range(C.TEAM_SIZE):
        base = team_offset + s * C.POKEMON_FULL_DIM
        if row[base + C.POKEMON_CONDITION_OFFSET + 6] == 1.0:
            out.append((s, float(row[base + C.POKEMON_COUNTER_OFFSET + 1])))
    return out


def test_the_row_the_trainee_reads_carries_toxic_stage_12_on_its_true_scale():
    [res] = run_core([_battle()], trackers=True, obs=True)
    assert res["ok"], res.get("error")
    own = [wrap_row(c["obs"]) for c in res["trackers"][0] if "obs" in c]
    opp = [wrap_row(c["obs"]) for c in res["trackers"][1] if "obs" in c]
    # one decision per turn; the turn-13 decision follows 12 residual ticks
    assert len(own) >= TURNS, f"only {len(own)} trainee decisions — Blissey died early or the battle stopped"
    stages = []
    for row in own:
        cells = _tox_cells(row, C.OFFSET_OUR_TEAM)
        stages.append(round(cells[0][1] * C.TOXIC_STAGE_MAX) if cells else 0)
    # decision k (turn k + 1) follows k residual ticks: the ramp is 0, 1, ..., 12 — then Blissey dies to the 13th tick
    assert stages[:13] == list(range(13)), f"the stage ramp 0..12 expected (Toxic landing on turn 1), got {stages}"
    last = own[stages.index(12)]
    [(slot, cell)] = _tox_cells(last, C.OFFSET_OUR_TEAM)
    assert cell == np.float32(12 / C.TOXIC_STAGE_MAX), cell
    assert cell != 1.0, "the cell is saturated: the old min(stage, 8) / 8 scale"
    # the model's decoder, on the REAL row: the 13th tick is 13/16 of max HP (Showdown: stage 13, floor(maxhp/16) * 13)
    nxt = toxic_next_tick(torch.tensor(last[C.OFFSET_OUR_TEAM + slot * C.POKEMON_FULL_DIM + C.POKEMON_COUNTER_OFFSET + 1]))
    assert float(nxt) == pytest.approx(13 / 16, abs=1e-6), float(nxt)
    # the opponent's view of the same mon carries the same public counter
    o_stages = [(_tox_cells(r, C.OFFSET_OPP_TEAM) or [(0, 0.0)])[0][1] for r in opp]
    assert round(max(o_stages) * C.TOXIC_STAGE_MAX) == 12, o_stages


def test_the_next_tick_decoder_reads_every_stage_to_the_engines_cap():
    """``min(n + 1, 15) / 16`` for n = 0..20 ticks taken, from the cell the encoder writes (``min(n, 15) / 15``)."""
    n = torch.arange(0, 21)
    cell = torch.clamp(n, max=C.TOXIC_STAGE_MAX).float() / C.TOXIC_STAGE_MAX
    want = torch.clamp(n + 1, max=15).float() / 16.0
    assert torch.allclose(toxic_next_tick(cell), want, atol=1e-6)
    assert float(toxic_next_tick(torch.tensor(8.0 / 15))) == pytest.approx(9 / 16, abs=1e-6), "the old cap read 9th as 8th"
    assert float(toxic_next_tick(torch.tensor(1.0))) == pytest.approx(15 / 16, abs=1e-6), "never 16/16"
