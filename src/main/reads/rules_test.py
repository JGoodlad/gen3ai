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
    assert set(RU.RULES) == {"x35_r1", "d1", "anchor_trajectory", "prior_mix_ab"}
    assert [n for n, r in sorted(RU.RULES.items()) if r.x35_record] == ["d1", "x35_r1"]
    with pytest.raises(RU.RuleInputError):
        RU.RULES["anchor_trajectory"].verdict_of({"decision": {}})


# ------------------------------------------------------- AM-N1P (2026-10-26): X61P's registered read specs, the d1 rule
X61P = "designs/research_state/measurements/x61p_n1_production_2026-10-26"


@pytest.mark.parametrize("look", [1, 2])
def test_x61ps_committed_specs_parse_with_forum72_alone_at_60_units_and_a_reported_group(look):
    from main.reads.spec import load
    from utils.paths import repo_path

    sp = load(repo_path(f"{X61P}/read_spec_look{look}.json"))
    assert sp.rule == "d1" and sp.params["arm"] == "X61P" and sp.params["look"] == look
    assert sp.anchor["primary"].teamsets == ("forum72",) and len(sp.anchor["primary"].unit_seeds) == 60
    assert set(sp.anchor["reported"].teamsets) == {"heldout72_hpfix", "p72", "home72"}
    assert sorted({m.seed for m in sp.models.values()}) == list(range(3501, 3501 + R.LOOK_SEEDS[look]))
    assert len(sp.models) == 2 * R.LOOK_SEEDS[look]
    assert all(m.ref.kind == ("pinned" if m.arm == "c0" else "final") for m in sp.models.values())


def test_the_d1_rule_forms_x61ps_reported_panels_from_its_second_group_and_refuses_a_repeated_panel():
    from main.reads.spec import parse

    labels = ["c0_s3501", "n1p_s3501"]

    def spec(reported_teamsets):
        return parse({
            "schema": "gen3_read_spec_v1", "read": "x61p_test",
            "models": {lab: {"arm": lab[:-6], "seed": 3501, "ref": {"kind": "pinned", "path": f"/x/{lab}.zip",
                                                                     "sha256": "0" * 64}} for lab in labels},
            "anchor": {"primary": {"opponent": "metamon:Kakuna", "teamsets": ["forum72"], "unit_seeds": "0-59",
                                   "models": labels},
                       "reported": {"opponent": "metamon:Kakuna", "teamsets": reported_teamsets, "unit_seeds": "0-6",
                                    "models": labels}},
            "s": {"p": {"refpop": "/x/r.json", "refpop_content_sha256": "0" * 64, "panel": "home72", "pairs": 500,
                        "batch_pairs": 100, "family": "f", "models": labels}},
            "decision": {"rule": "d1", "params": {"arm": "X61P", "look": 1, "on": "n1p", "off": "c0",
                                                  "anchor_group": "primary", "s_group": "p",
                                                  "reported_group": "reported"}}})

    sp = spec(["heldout72_hpfix"])

    def cell(w, n):
        return {"wins": w, "n": n, "ties": 0, "units": n // 20, "skipped": []}

    cells = {"anchor": {"primary": {"cells": {"c0_s3501": {"forum72": cell(540, 1200)},
                                              "n1p_s3501": {"forum72": cell(600, 1200)}}},
                        "reported": {"cells": {"c0_s3501": {"heldout72_hpfix": cell(70, 140)},
                                               "n1p_s3501": {"heldout72_hpfix": cell(80, 140)}}}},
             "s": {"p": {"nodes": {lab: {"s_pp": 50.0, "population_score": {}} for lab in labels},
                         "inconclusive": [], "panel": "home72", "team_set": "ts"}}}
    doc = RU.inputs("d1", cells, sp.params, sp)
    assert doc["h"]["on"] == {"3501": {"forum72": [600, 1200]}}
    assert doc["reported_panels"] == {"on": {"3501": {"heldout72_hpfix": [80, 140]}},
                                      "off": {"3501": {"heldout72_hpfix": [70, 140]}}}
    with pytest.raises(Exception, match="repeats a primary panel"):
        spec(["forum72", "p72"])


# ---------------------------------------- D3 (X41, 2026-10-26): the d1 rule's banked OFF side + determinism guard
STAGE_D1 = "designs/research_state/measurements/x35_breakers_stage_2026-10-17/d1_read/read_look2_input.json"
D1_INPUT_SHA = "8c4884aafcb5fa77a15b2e8be517460cf64f411a43a6ffa67bf02d1f4f2170b7"
D3_PANELS = ["heldout72_hpfix", "forum72", "p72"]


def _d3_spec(tmp_path, banked_rows, **over):
    import hashlib
    import json

    from main.reads.spec import parse

    bp = tmp_path / "banked.jsonl"
    bp.write_text("".join(json.dumps(r) + "\n" for r in banked_rows))
    on = [f"d3_s{s}" for s in (3501, 3502, 3503)]
    models = {lab: {"arm": "D3", "seed": int(lab[-4:]), "ref": {"kind": "pinned", "path": f"/x/{lab}.zip",
                                                                "sha256": "0" * 64}} for lab in on}
    models["c0_s3501"] = {"arm": "C0", "seed": 3501, "ref": {"kind": "pinned", "path": "/x/c0.zip", "sha256": "1" * 64}}
    params = {"arm": "X41", "look": 1, "on": "D3", "off": "C0", "anchor_group": "h", "s_group": "p",
              "import_off": {"path": STAGE_D1, "sha256": D1_INPUT_SHA,
                             "labels": {"3501": "c0_s3501", "3502": "c0_s3502", "3503": "c0_s3503"},
                             "regimes": {"anchor": "aaaa", "s": "bbbb"}},
              "determinism_guard": {"group": "guard", "label": "c0_s3501",
                                    "banked": {"path": str(bp), "sha256": hashlib.sha256(bp.read_bytes()).hexdigest(),
                                               "cells": {"heldout72_hpfix": "heldout72_hpfix"}}},
              "reported_without": ["heldout72_hpfix"], "dose": {"on": [4.4, 4.3, 4.2], "off": [4.4, 4.0, 4.0]}}
    params.update(over)
    return parse({
        "schema": "gen3_read_spec_v1", "read": "x41_test", "models": models,
        "anchor": {"h": {"opponent": "metamon:Kakuna", "teamsets": D3_PANELS, "unit_seeds": "0-6", "models": on},
                   "guard": {"opponent": "metamon:Kakuna", "teamsets": ["heldout72_hpfix"], "unit_seeds": "0-1",
                             "models": ["c0_s3501"]}},
        "s": {"p": {"refpop": "/x/r.json", "refpop_content_sha256": "0" * 64, "panel": "home72", "pairs": 500,
                    "batch_pairs": 100, "family": "f", "models": on}},
        "decision": {"rule": "d1", "params": params}})


def _d3_cells(guard_games, *, anchor_regime="aaaa", s_regime="bbbb"):
    def cell(w):
        return {"wins": w, "n": 140, "ties": 0, "units": 7, "skipped": []}

    on = {f"d3_s{s}": {ts: cell(75) for ts in D3_PANELS} for s in (3501, 3502, 3503)}
    units = [{"unit": f"c0_s3501__heldout72_hpfix__{s:02d}", "label": "c0_s3501", "panel": "heldout72_hpfix",
              "unit_seed": s, "counted": True, "n": len(g), "wins": sum(x[3] == "win" for x in g), "games": g}
             for s, g in guard_games.items()]
    return {"anchor": {"h": {"cells": on, "regime_id": anchor_regime},
                       "guard": {"cells": {}, "units": units, "regime_id": "cccc"}},
            "s": {"p": {"nodes": {lab: {"s_pp": 56.0, "population_score": {}} for lab in on}, "inconclusive": [],
                        "panel": "home72", "team_set": "ts:08142ee8b0c18e8a", "regime_id": s_regime}}}


GAMES = {0: [["ours_challenge", 1, "t1.txt", "win", 30], ["peer_challenge", 1, "t2.txt", "loss", 40]],
         1: [["ours_challenge", 1, "t3.txt", "loss", 22]]}


def _banked(games):
    return [{"read_cell": "heldout72_hpfix", "unit_seed": s, "half": g[0], "index": g[1], "our_team": g[2],
             "result": g[3], "turns": g[4]} for s, gs in games.items() for g in gs]


def test_d3_pools_the_banked_c0_side_only_when_the_guard_replays_it_game_for_game(tmp_path):
    sp = _d3_spec(tmp_path, _banked(GAMES))
    doc = RU.inputs("d1", _d3_cells(GAMES), sp.params, sp)
    assert doc["import_off"]["h_pooled"] is True and doc["import_off"]["guard"]["passed"] is True
    assert doc["h"]["off"]["3501"] == {"forum72": [72, 140], "heldout72_hpfix": [69, 140], "p72": [59, 140]}
    assert sorted(doc["h"]["off"]) == sorted(doc["p"]["off"]) == ["3501", "3502", "3503"]
    assert doc["p"]["off"]["3501"] == 56.778571428571425 and "c0_s3501" in doc["p"]["per_final"]
    v = RU.decide("d1", doc, sp.params)
    assert v["h"]["class"] is not None and not any("determinism" in r for r in v["reasons"])
    w = v["reported"]["without"]
    assert w["panels"] == ["forum72", "p72"] and set(w["per_panel_reported"]) == {"forum72", "p72"}
    assert v["reported"]["dose"]["read"] is True
    # one game differs: H's banked rows are NOT pooled, H has no OFF side, and the reason names the guard
    bad = {0: [GAMES[0][0], ["peer_challenge", 1, "t2.txt", "win", 40]], 1: GAMES[1]}
    doc = RU.inputs("d1", _d3_cells(bad), sp.params, sp)
    assert doc["import_off"]["h_pooled"] is False and doc["h"]["off"] == {}
    v = RU.decide("d1", doc, sp.params)
    assert v["h"]["class"] is None and any("determinism guard failed" in r for r in v["reasons"])
    # a guard unit short of the banked ones also fails it
    doc = RU.inputs("d1", _d3_cells({0: GAMES[0]}), sp.params, sp)
    assert doc["import_off"]["h_pooled"] is False


def test_d3_refuses_the_banked_side_on_another_regime_or_without_a_guard(tmp_path):
    sp = _d3_spec(tmp_path, _banked(GAMES))
    for kw in ({"anchor_regime": "zzzz"}, {"s_regime": "zzzz"}):
        with pytest.raises(RU.RuleInputError, match="re-play the control"):
            RU.inputs("d1", _d3_cells(GAMES, **kw), sp.params, sp)
    with pytest.raises(Exception, match="determinism_guard"):
        _d3_spec(tmp_path, _banked(GAMES), determinism_guard=None)
    with pytest.raises(Exception, match="reported_without"):
        _d3_spec(tmp_path, _banked(GAMES), reported_without=list(D3_PANELS))
