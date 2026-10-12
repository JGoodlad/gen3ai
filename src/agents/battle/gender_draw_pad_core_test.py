"""The gender DRAW PAD end to end through ``core_events`` (``gen3_gender_draw_pad_v1``).

A search world re-draws the opponent's NEVER-SEEN mons. The engine draws one turn-0 gender ``sample(['M','F'])``
for every mon whose packed set names no gender and whose species has no fixed gender, so a world that swaps a TRUE
draw slot's ratio'd, unnamed mon for any other species must mark the new mon with the pad (``<g>`` +
``utils.team_packing.GENDER_DRAW_PAD``) to leave the dice where the truth left them. Then the OTHER side sees exactly
the truth's battle: its per-side protocol bytes and its whole capture (trackers + obs rows) are identical, and the
swapped side's own stream differs only in its ``|request|`` JSON (the one place its own team shows).

The fixture is one committed Lane S bank battle (``main.policy_spectrum.bank_test.BANK_V1``) where p2's slot 2 (a
Skarmory, gender-ratio species, no gender named) never comes in; the precondition is ASSERTED, not branched on. The
unpadded control (the same named gender, no marker) must move the dice — the pad is load-bearing. The engine-level
pins (every gender class, the draw position, the codec) are ``src/rust_sim/tests/gender_draw_pad_test.rs``.
"""

from __future__ import annotations

import gzip
import json
import re
from typing import Any, Dict, List

import pytest

from agents.battle.core_replay import RecordedBattle, run_core
from agents.gen3_data import species
from utils.paths import repo_path
from utils.showdown_id import to_id_str
from utils.team_packing import GENDER_DRAW_PAD

pytestmark = [pytest.mark.sim, pytest.mark.integration]

BANK_BATTLES = repo_path("designs", "research_state", "measurements", "m5_laneS", "bank_v1", "battles.jsonl.gz")
BATTLE_ID = "ai_v14_01_base/step_36000000/aggressive/loss_s0_005"
SIDE, SLOT = "p2", 1          # the never-seen TRUE draw slot (0-based team position)

#: World mons for that slot, one per gender class: fixed-male, genderless, and a ratio'd species with a named gender.
#: None of them shares a species or name with p2's other five mons (the recorded commands switch by NAME).
WORLD_SETS = (
    "Tauros||leftovers|intimidate|bodyslam,earthquake,rockslide,return|Jolly|,252,,,4,252|M{pad}||||",
    "Starmie||leftovers|naturalcure|surf,thunderbolt,icebeam,recover|Timid|,,,252,4,252|N{pad}||||",
    "Gengar||leftovers|levitate|thunderbolt,icepunch,firepunch,explosion|Timid|,,,252,4,252|F{pad}||||",
)


def _bank_battle() -> Dict[str, Any]:
    with gzip.open(BANK_BATTLES, "rt") as fh:
        for line in fh:
            row = json.loads(line)
            if row["battle_id"] == BATTLE_ID:
                return row
    raise AssertionError(f"{BATTLE_ID} is not in {BANK_BATTLES}")


def _with_slot(row: Dict[str, Any], packed_set: str) -> RecordedBattle:
    side = dict(row[SIDE])
    sets = side["team"].split("]")
    sets[SLOT] = packed_set
    side["team"] = "]".join(sets)
    teams = {"p1": row["p1"], "p2": row["p2"], SIDE: side}
    return RecordedBattle(label=row["battle_id"], format_id=row["format_id"], seed=row["seed"],
                          p1=teams["p1"], p2=teams["p2"], commands=[list(c) for c in row["commands"]])


def _side_lines(res: Dict[str, Any], side: int) -> List[str]:
    return [ln for s, lines in res["chunks"] if s == side for ln in lines]


def test_a_padded_world_slot_leaves_the_other_side_the_truths_battle():
    row = _bank_battle()
    true_set = row[SIDE]["team"].split("]")[SLOT]
    fields = true_set.split("|")
    assert fields[7] == "", "precondition: the true slot names no gender"
    sd = species.get(to_id_str(fields[1] or fields[0]))
    assert sd is not None and sd.gender is None, "precondition: a gender-RATIO species (a draw slot)"

    padded = [s.format(pad=GENDER_DRAW_PAD) for s in WORLD_SETS]
    controls = [s.format(pad="") for s in WORLD_SETS]
    battles = [_with_slot(row, true_set)] + [_with_slot(row, s) for s in padded + controls]
    results = run_core(battles, trackers=True, obs=True)
    truth, worlds, ctrls = results[0], results[1:1 + len(padded)], results[1 + len(padded):]

    assert truth["ok"] and truth["ended"], truth.get("error")
    own = int(SIDE[1]) - 1        # the swapped side's index in the chunks / trackers
    other = 1 - own
    true_name = fields[0]
    seen = "\n".join(_side_lines(truth, other))
    # Precondition: the swapped slot never comes in (no switch / drag / replace of it on the other side's stream).
    assert not re.search(rf"\|(switch|drag|replace)\|{SIDE}a: {re.escape(true_name)}\|", seen), \
        f"precondition: {true_name} must never be revealed in {BATTLE_ID}"

    def strip_requests(lines: List[str]) -> List[str]:
        return [ln for ln in lines if not ln.startswith("|request|")]

    for packed_set, res in zip(padded, worlds):
        assert res["ok"] and res["ended"], (packed_set, res.get("error"))
        assert _side_lines(res, other) == _side_lines(truth, other), f"{packed_set}: the other side's bytes moved"
        assert res["trackers"][other] == truth["trackers"][other], f"{packed_set}: the other side's rows moved"
        assert strip_requests(_side_lines(res, own)) == strip_requests(_side_lines(truth, own)), packed_set
        assert _side_lines(res, own) != _side_lines(truth, own), "non-vacuity: the own |request| shows the swap"

    # The control: the same named gender WITHOUT the pad draws one fewer die, so the battle the other side sees moves
    # (or, with the dice shifted, a recorded command no longer applies and the replay refuses).
    for packed_set, res in zip(controls, ctrls):
        assert not res["ok"] or _side_lines(res, other) != _side_lines(truth, other), \
            f"{packed_set}: an unpadded world must move the dice — else the pad is vacuous here"
