"""``main.reads.rules``: the named estimators (rule 8, the per-cell and per-final SE, the unit-paired difference),
the A1 guard ruling, and ``anchor_trajectory``'s VOID / CLIMBING / RISING branches on constructed cases."""
from __future__ import annotations

import math

import pytest

from main.reads import rules as RU
from main.x35 import readout as R


def test_rule8_never_crosses_within_eps_of_a_boundary():
    assert RU.clears(0.5 + 2e-9, 0.5) == (True, False)
    assert RU.clears(0.5 + 5e-10, 0.5) == (False, True)
    assert RU.clears(0.5 - 5e-10, 0.5, above=False) == (False, True)
    assert RU.clears(-1.0, 0.0, above=False) == (True, False)


def test_cell_se_and_s_se_are_the_pre_reads_declared_interval():
    pc = (1, 2, 4, 2, 3)            # n = 12, mean (0 + 2 + 8 + 6 + 12) / 48 = 28 / 48
    mu = 28 / 48
    var = sum(c * (i / 4 - mu) ** 2 for i, c in enumerate(pc)) / 12
    assert RU.cell_se(pc) == math.sqrt(var / 12)
    assert RU.s_se([3.0, 4.0]) == 2.5
    with pytest.raises(RU.RuleInputError):
        RU.cell_se((0, 0, 0, 0, 0))


def test_the_unit_paired_difference_needs_two_pairs_per_panel():
    per = {("on", "p", s): 0.6 for s in range(3)} | {("off", "p", s): 0.5 + 0.01 * s for s in range(3)}
    d = RU.paired_unit_diff(per, "on", "off", ["p"], range(3))
    assert d is not None and d["pairs"] == {"p": 3} and abs(d["d"] - 0.09) < 1e-12 and d["lo"] < d["d"] < d["hi"]
    assert RU.paired_unit_diff({("on", "p", 0): 0.6, ("off", "p", 0): 0.5}, "on", "off", ["p"], range(3)) is None


def _verdict(cls, verdict):
    return {"r1": {"class": cls}, "verdict": verdict,
            "guards": {g: {"status": "PASS", "reasons": []} for g in R.GUARDS}}


def test_the_a1_ruling_touches_only_a_non_qualifying_primary():
    ruling = {"guards": ["G-F"], "why": "not read", "absent": {"G-E": "by design"}}
    v = RU.guards_not_read(_verdict("NULL", "NULL"), ruling)
    assert v["guards"]["G-F"] == {"status": "INCONCLUSIVE", "reasons": ["not read"]}
    assert v["guards"]["G-E"] == {"status": "PASS", "reasons": ["by design"]}
    q = RU.guards_not_read(_verdict("GAIN", "ADOPT-CANDIDATE"), ruling)
    assert q["guards"]["G-F"]["status"] == "PASS", "a qualifying primary's guards decide; the ruling never applies"


# ------------------------------------------------------------------------------------------------ anchor_trajectory
PANELS = ["a", "b"]
PARAMS = {"on": "new", "off": "old", "anchor_group": "u", "goal_bar": 0.5, "min_unit_frac": 0.9}


def _doc(win_on, win_off, *, units=10, counted=True, s1=None):
    rows = []
    for lab, w in (("new", win_on), ("old", win_off)):
        for k in PANELS:
            for s in range(units):
                rows.append({"unit": f"{lab}__{k}__{s:02d}", "label": lab, "panel": k, "unit_seed": s, "wins": w,
                             "n": 20, "ties": 0, "counted": counted})
    return {"on": "new", "off": "old", "panels": PANELS, "unit_seeds": list(range(units)),
            "units_planned": len(PANELS) * units, "units": rows, "s1": s1, "k9b_verdict": "PASS"}


def test_trajectory_flat_and_climbing():
    flat = RU.decide("anchor_trajectory", _doc(10, 10), PARAMS)["decision"]
    assert flat == {"goal_met": False, "trajectory": "FLAT", "verdict": "goal not met; FLAT"}
    up = RU.decide("anchor_trajectory", _doc(16, 9), PARAMS)["decision"]
    assert up["trajectory"] == "CLIMBING" and up["goal_met"] is True and up["verdict"] == "GOAL MET; CLIMBING"


def test_trajectory_is_void_on_too_few_counted_units_or_a_failed_k9b():
    doc = _doc(10, 10)
    for r in doc["units"][:3]:
        r["counted"] = False
    out = RU.decide("anchor_trajectory", doc, PARAMS)["decision"]
    assert out["verdict"] == RU.VOID and "17 / 20 units counted" in out["reasons"][0]
    doc = _doc(10, 10)
    doc["k9b_verdict"] = "FAIL"
    assert RU.decide("anchor_trajectory", doc, dict(PARAMS, k9b="k.json"))["decision"]["verdict"] == RU.VOID


def test_s1_rises_only_when_its_lower_bound_clears_the_baseline():
    p = dict(PARAMS, s1={"group": "g", "label": "new"}, s_baseline={"s_pp": 50.0, "se_pp": 0.5})
    flat = RU.decide("anchor_trajectory", _doc(10, 10, s1={"s_pp": 50.5, "se_pp": 0.5}), p)["decision"]
    assert flat["s1"]["rising"] is False and flat["d3_revisit"] is False
    up = RU.decide("anchor_trajectory", _doc(10, 10, s1={"s_pp": 53.0, "se_pp": 0.5}), p)["decision"]
    assert up["s1"]["rising"] is True and up["d3_revisit"] is True
    gone = RU.decide("anchor_trajectory", _doc(10, 10, s1=None), p)["decision"]
    assert gone["s1"] is None and gone["d3_revisit"] == "UNREAD (S1 missing)"


def test_an_imported_control_on_another_r_is_refused():
    """sprt_read/assemble.py's STOP ("the two reads used different R"), kept."""
    g1 = {"path": "designs/research_state/measurements/x35_breakers_stage_2026-10-17/g1/results.json",
          "sha256": "393b7d572bb03c25afc089e3c7f9ca121c79aab9e1d436380fbbac96d9cf6786", "arm": "c0"}
    assert set(RU.imported_s(g1)) == {"3501", "3502", "3503"}
    with pytest.raises(RU.RuleInputError, match="not this read's"):
        RU.imported_s(g1, want_refpop="0" * 64)


def test_every_rule_is_registered_and_names_its_record_shape():
    assert set(RU.RULES) == {"x35_r1", "d1", "anchor_trajectory"}
    assert [n for n, r in sorted(RU.RULES.items()) if r.x35_record] == ["d1", "x35_r1"]
    with pytest.raises(RU.RuleInputError):
        RU.RULES["anchor_trajectory"].verdict_of({"decision": {}})
