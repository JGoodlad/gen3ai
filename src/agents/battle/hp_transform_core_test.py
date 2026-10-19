"""F-M5-1 REGRESSION PIN, Python side (``gen3_hp_fire_time_target_v1``): a TRANSFORMED mon KO'd by Hidden Power.

The banked battle (``designs/research_state/measurements/rust_core_m5_transport_2026-09-26/repro/hp_tracker_env46.script``)
replayed through ``core_events --trackers``, the core the trainer's rows come from. p2's Smeargle Transforms into p1's
Tyranitar (Rock / Dark) and is KO'd by its Hidden Power Grass (2x). The Hidden-Power belief used to re-read the target's
types from the LIVE board at the next decision, where the faint had cleared the Transform overlay, so it saw Smeargle's
Normal and the log refuted every type: the core REFUSED the battle ("all candidates eliminated"). It now reads the
target as it was WHEN THE MOVE FIRED. Both halves of the fix are held here (each FAILS on its revert):

* the replay is ``ok`` and p2's belief about p1's Tyranitar keeps its TRUE type, Grass;
* p1 files NOTHING under Smeargle: a transformed user's Hidden Power is the COPIED mon's type (gen 3 copies
  ``hpType``, ``sim/pokemon.ts`` ``transformInto``), so it says nothing about Smeargle's own set.

The Python ``HiddenPowerTracker`` this finding first named is DELETED (T27 P6 slice 6d-2); the Rust core is the only
tracker, so this pin runs it from Python. The Rust pins: ``src/rust_env/tests/lifecycle_refusal_test.rs``
``f_m5_1_*`` and ``src/rust_sim/src/trackers/hp_belief.rs``'s unit tests.
"""
from __future__ import annotations

import json
from typing import List

import pytest

from agents.battle.core_replay import RecordedBattle, run_core
from agents.training.hidden_power_tracker import HIDDEN_POWER_TYPE_ORDER
from utils.paths import repo_path

pytestmark = [pytest.mark.sim, pytest.mark.integration]

REPRO = repo_path("designs", "research_state", "measurements", "rust_core_m5_transport_2026-09-26", "repro",
                  "hp_tracker_env46.script")


def _battle() -> RecordedBattle:
    lines = [ln for ln in REPRO.read_text().splitlines() if ln.strip()]
    start = json.loads(lines[0].removeprefix("START "))
    cmds: List[List[str]] = []
    for ln in lines[1:]:
        if ln == "END":
            break
        verb, side, choice = ln.split(" ", 2)
        assert verb == "CHOOSE", ln
        cmds.append([side, choice])
    return RecordedBattle(label="f_m5_1", format_id=start["formatid"], seed=start["seed"], p1=start["p1"],
                          p2=start["p2"], commands=cmds)


def test_a_transformed_mon_kod_by_hidden_power_is_read_at_fire_time():
    (res,) = run_core([_battle()], trackers=True)
    assert res["ok"], f"F-M5-1 is refused again: {res.get('error')}"
    decisions = res["trackers"]
    assert len(decisions) == 2 and decisions[0] and decisions[1], "both viewers decided"

    # p2's belief about p1's Tyranitar (Hidden Power Grass): 2x on Tyranitar, 2x twice on the Smeargle Transformed
    # into a Tyranitar — the true type survives every observation.
    p2_hp = dict((sp, v) for sp, v in decisions[1][-1]["trackers"]["hp"]["state"])
    grass = [t.name.lower() for t in HIDDEN_POWER_TYPE_ORDER].index("grass")
    assert "tyranitar" in p2_hp, p2_hp
    assert p2_hp["tyranitar"][grass] > 0.0, p2_hp["tyranitar"]

    # p1 saw the TRANSFORMED Smeargle fire Hidden Power (the copied Grass) — never filed under Smeargle.
    for d in decisions[0]:
        filed = [sp for sp, _ in d["trackers"]["hp"]["state"]]
        assert "smeargle" not in filed, (d["after"], filed)
