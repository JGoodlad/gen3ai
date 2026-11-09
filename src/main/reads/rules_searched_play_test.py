"""``main.reads`` rule ``searched_play`` (the S′ live-play read): GO-LEAKY / NULL / STOP and the ``boundary`` /
``go_below_margin`` flags by exact fractions; the robust imputation of post-treatment unpaired deals (worst case for GO
on the LB, for NULL on the UB); a G-PREFIX ``peer_first`` mismatch drops the deal while ``ours_first`` VOIDs; every
VOID reason (a fixture that triggers it, and passes without it); the bootstrap against a hand computation; the live
identity guard's JSON (missing, FAIL, inconsistent, short); the spec check; and ``assemble`` leaving a G-PREFIX
mismatch to this rule. Synthetic cells and a synthetic state, no game."""
from __future__ import annotations

import copy
import json
import time
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pytest

from main.anchors.search_presets import PRESETS
from main.reads import rules as RU
from main.reads import rules_searched_play as SP
from main.reads.peer_divergence_test import synthetic
from main.reads.spec import SpecError, parse
from main.value_search import live_identity as LI

MAXR = 600
SHA = "a" * 64
ENGINE = "port=aaaaaaaaaaaaaaaa/10;env=bbbbbbbbbbbbbbbb/20"
U = ["heldout72_hpfix", "forum72", "p72"]
SEEDS, GAMES = range(4), range(4)
SEARCH = "r0s4_rolls@10"
PRESET = PRESETS["r0s4_rolls"].as_dict()
SPEC = SimpleNamespace(models={"pstar": SimpleNamespace(ref=SimpleNamespace(sha256=SHA))})


def _name(label: str, ts: str, seed: int, game: int) -> str:
    return f"{label}__{ts}__{seed:02d}__g{game:02d}"


def _state(tmp: Path) -> Path:
    """A read state: both labels' plans (every deal of U × SEEDS × GAMES) and one sampled searched record."""
    st = tmp / "state"
    for lab in ("pstar", "pstar_s"):
        d = st / "anchor" / "sp" / lab
        d.mkdir(parents=True)
        rows = [{"name": _name(lab, ts, s, g), "label": lab, "arm": lab, "seed": 2601, "teamset": ts, "unit_seed": s,
                 "model": "/m", "model_sha256": SHA, "games": 1, "game": g,
                 "search": SEARCH if lab == "pstar_s" else None} for ts in U for s in SEEDS for g in GAMES]
        (d / "plan.json").write_text(json.dumps({"units": rows}))
    from main.anchors.search_side import rescore_sampled

    key = next(k for k in (f"su_live_v1|x{j}|t12|n13" for j in range(999)) if rescore_sampled(k))
    cell = st / "anchor" / "sp" / "pstar_s" / "cells" / _name("pstar_s", "forum72", 1, 2)
    cell.mkdir(parents=True)
    (cell / "search_decisions.jsonl").write_text(json.dumps({"key": key, "rescore_sample": True,
                                                             "search": SEARCH}) + "\n")
    _rescore_json(st)
    return st


def _cells_dir(st: Path) -> Path:
    return st / "anchor" / "sp" / "pstar_s" / "cells"


def _rescore_json(st: Path, **over: Any) -> Path:
    cells = _cells_dir(st)
    doc = {"schema": LI.RESCORE_SCHEMA, "inputs": {"cells": str(cells), "panels": U, "engine": ENGINE,
                                                   "model_sha256": SHA, "max": 600, "min": 1},
           "n_sampled": 1, "n_rescored": 1, "rescored": LI.registered_order(cells, U, 600), "mismatches": [],
           "by_panel": {"forum72": 1}, "verdict": "PASS", "reason": None, "cpu_s": 12.0}
    doc.update(over)
    p = st / "rescore.json"
    p.write_text(json.dumps(doc))
    return p


def _params(st: Path, **over: Any) -> Dict[str, Any]:
    p = {"group": "sp", "searched": "pstar_s", "raw": "pstar", "u_panels": list(U), "expected_search": SEARCH,
         "engine": ENGINE, "null_margin_pp": 3.0, "bootstrap": {"B": 2000, "seed": 20261101}, "rule8_eps": 0.0005,
         "max_post_unpaired_frac": 0.1, "max_pre_unpaired_frac": 0.1,
         "identity": {"cells": str(_cells_dir(st)), "rescore_json": str(st / "rescore.json"), "min_rescored": 1, "max_rescored": MAXR},
         "state": str(st)}
    p.update(over)
    return p


def _unit(label: str, ts: str, seed: int, game: int, result: str) -> Dict[str, Any]:
    s = label == "pstar_s"
    u = {"unit": _name(label, ts, seed, game), "label": label, "panel": ts, "unit_seed": seed, "game": game,
         "counted": True, "wins": int(result == "win"), "ties": int(result == "tie"), "n": 1,
         "our_regime": f"greedy+search={SEARCH}" if s else "greedy", "search": SEARCH if s else None,
         "model_sha256": SHA, "engine": ENGINE, "games": [["ours_challenge", game + 1, "t.txt", result, 30]]}
    if s:
        u["search_played"] = {"search": SEARCH, "our_search": SEARCH, "preset": PRESET, "searched": 30,
                              "deviations": 3, "excluded": 1, "search_cpu_s": 600.0}
    return u


def _by_k(k: Dict[str, List[int]]) -> Callable[[str, int, int], "tuple[str, str]"]:
    """Unit (ts, seed) of statistic k[ts][seed] / 4: its first |k| games discordant in k's direction."""
    def out(ts: str, seed: int, game: int) -> "tuple[str, str]":
        kk = k[ts][seed]
        if game < abs(kk):
            return ("loss", "win") if kk > 0 else ("win", "loss")
        return ("win", "win") if game % 2 else ("loss", "loss")
    return out


def _cells(outcome: Callable[[str, int, int], "tuple[str, str]"],
           mutate: Optional[Callable[[List[Dict[str, Any]]], None]] = None, *, exhausted: List[str] = (),
           mismatches: List[Dict[str, Any]] = ()) -> Dict[str, Any]:
    units = []
    for ts in U:
        for s in SEEDS:
            for g in GAMES:
                r, x = outcome(ts, s, g)
                units += [_unit("pstar", ts, s, g, r), _unit("pstar_s", ts, s, g, x)]
    if mutate:
        mutate(units)
    done = {(u["panel"], u["unit_seed"], u["game"], u["label"]) for u in units if u["counted"]}
    checked = sum(1 for (ts, s, g, lab) in done if lab == "pstar" and (ts, s, g, "pstar_s") in done)
    return {"models": {"pstar": {"sha256": SHA}, "pstar_s": {"sha256": SHA}},
            "anchor": {"sp": {"units": units, "engine": ENGINE, "regime_id": "r" * 16, "teamsets": U,
                              "unit_seeds": list(SEEDS), "unplayed": 0, "inconclusive": [],
                              "exhausted_units": list(exhausted),
                              "g_prefix": {"schema": "gen3_read_prefix_v1", "pairs": len(U) * 16,
                                           "checked": checked, "pending": 0, "mismatches": list(mismatches)}}}}


def _read(st: Path, cells: Dict[str, Any], **over: Any) -> Dict[str, Any]:
    p = _params(st, **over)
    doc = json.loads(json.dumps(RU.inputs("searched_play", cells, p, SPEC)))
    return RU.decide("searched_play", doc, p)


@pytest.fixture
def st(tmp_path) -> Path:
    return _state(tmp_path)


def _k(u: List[int]) -> Dict[str, List[int]]:
    return {ts: u for ts in U}


UP, FLAT, WIDE = _k([1, 3, 1, 3]), _k([0, 0, 0, 0]), _k([-2, 2, -2, 2])


# ------------------------------------------------------------------------------------------------ the classes
def test_go_leaky_on_a_clear_gain(st):
    out = _read(st, _cells(_by_k(UP)))
    d = out["decision"]
    assert d["class"] == SP.GO_LEAKY and d["verdict"].startswith("GO-LEAKY: ") and not d["boundary"]
    assert out["estimates"]["paired"]["exact"]["G_U"] == "1/2"
    assert out["score_u"]["searched_lb_above_0_50"]["crosses"] is True


def test_null_on_a_flat_read_is_flagged_boundary_since_lb_sits_on_zero(st):
    out = _read(st, _cells(_by_k(FLAT)))
    d = out["decision"]
    assert out["estimates"]["paired"]["G_U"]["ci95"] == [0.0, 0.0]
    assert d["class"] == SP.NULL and d["boundary"] is True and d["go_below_margin"] is False
    assert d["tests"]["LB(G_U go_imputed) > 0"] == {"crosses": False, "near": True}


def test_stop_when_the_interval_is_wide(st):
    d = _read(st, _cells(_by_k(WIDE)))["decision"]
    assert d["class"] == SP.STOP and not d["boundary"]


def test_go_below_margin_flags_a_gain_whose_ub_is_below_the_margin(st):
    out = _read(st, _cells(_by_k(_k([1, 1, 1, 1]))), null_margin_pp=30.0)       # every unit exactly +1/4
    d = out["decision"]
    assert out["estimates"]["paired"]["G_U"]["ci95"] == [0.25, 0.25]
    assert d["class"] == SP.GO_LEAKY and d["go_below_margin"] is True and "go_below_margin" in d["verdict"]


@pytest.mark.parametrize("margin, cls, near", [(25.0, SP.GO_LEAKY, True), (25.04, SP.GO_LEAKY, True),
                                               (25.06, SP.GO_LEAKY, False), (24.0, SP.GO_LEAKY, False)])
def test_a_ub_within_eps_of_the_margin_is_near_never_below(st, margin, cls, near):
    d = _read(st, _cells(_by_k(_k([1, 1, 1, 1]))), null_margin_pp=margin)["decision"]
    assert d["class"] == cls and d["boundary"] is near and d["go_below_margin"] is (margin == 25.06)


# ------------------------------------------------------------------------------------------------ imputation
def _miss(label: str, ts: str, seed: int, game: int) -> Callable[[List[Dict[str, Any]]], None]:
    def m(units: List[Dict[str, Any]]) -> None:
        for u in units:
            if (u["label"], u["panel"], u["unit_seed"], u["game"]) == (label, ts, seed, game):
                u.update(counted=False, engine=None)
    return m


def test_post_treatment_deals_are_imputed_worst_case_for_go_and_for_null(st):
    # forum72 seed 0 (k = 1): game 0 (raw loss, searched win) loses its SEARCHED unit; game 2 (loss, loss) its RAW one
    def both(units):
        _miss("pstar_s", "forum72", 0, 0)(units)
        _miss("pstar", "forum72", 0, 2)(units)
    cells = _cells(_by_k(UP), mutate=both, exhausted=[_name("pstar_s", "forum72", 0, 0),
                                                      _name("pstar", "forum72", 0, 2)])
    out = _read(st, cells, max_post_unpaired_frac=0.2)
    e = out["estimates"]
    c = out["counts"]["by_panel"]["forum72"]
    assert c["post_treatment"] == {"total": 2, "searched_missing": 1, "raw_missing": 1, "neither": 0}
    assert c["pairs"] == 14 and c["pre_treatment"]["total"] == 0
    # paired: seed 0 keeps games 1, 3 (both concordant) -> 0; seeds 1-3: 3/4, 1/4, 3/4 -> G = 7/16
    assert e["paired"]["exact"]["G"]["forum72"] == "7/16"
    # GO: searched missing (y_raw 0) -> D = 0; raw missing (y_s 0) -> D = -1: seed 0 = (0 + 0 + 0 - 1) / 4 = -1/4
    assert e["go_imputed"]["exact"]["G"]["forum72"] == str((Fraction(-1, 4) + Fraction(7, 4)) / 4)
    # NULL: searched missing -> D = 1 - 0 = 1; raw missing -> D = y_s = 0: seed 0 = 1/4
    assert e["null_imputed"]["exact"]["G"]["forum72"] == str((Fraction(1, 4) + Fraction(7, 4)) / 4)
    assert out["decision"]["class"] == SP.GO_LEAKY


def test_imputed_d_by_cause():
    h = Fraction(1, 2)
    assert [SP.imputed_d(c, y, "go") for c, y in (("searched_missing", h), ("raw_missing", h), ("neither", None))] \
        == [-h, -h, -1]
    assert [SP.imputed_d(c, y, "null") for c, y in (("searched_missing", Fraction(1)), ("raw_missing", Fraction(1)),
                                                    ("neither", None))] == [0, 1, 1]


def test_a_pending_deal_is_unfinished_never_unpaired_and_voids_the_read(st):
    """M3: a unit neither counted nor out of attempts (a FAILED attempt whose retry never ran) is never silently
    dropped: the deal is UNFINISHED and the read is VOID until it is played or its attempts are spent."""
    out = _read(st, _cells(_by_k(UP), mutate=_miss("pstar_s", "p72", 3, 3)))
    c = out["counts"]["by_panel"]["p72"]
    assert c["unfinished"] == 1 and c["post_treatment"]["total"] == 0
    d = out["decision"]
    assert d["class"] == SP.VOID and any(r.startswith("1 deal(s) unfinished: the read is incomplete")
                                         for r in d["reasons"]), d["reasons"]
    done = _read(st, _cells(_by_k(UP), mutate=_miss("pstar_s", "p72", 3, 3),
                            exhausted=[_name("pstar_s", "p72", 3, 3)]), max_post_unpaired_frac=0.1)
    assert done["decision"]["class"] == SP.GO_LEAKY          # its attempts spent: post-treatment, imputed


# ------------------------------------------------------------------------------------------------ the bootstrap
def test_the_bootstrap_reproduces_a_hand_computation(st):
    out = _read(st, _cells(_by_k(WIDE)))
    x = np.array([-0.5, 0.5, -0.5, 0.5])
    for name in SP.ESTIMATES:                       # no unpaired deal: all three are the same draw (a fresh rng each)
        rng = np.random.default_rng(20261101)
        reps = [x[rng.integers(0, 4, size=(2000, 4))].mean(axis=1) for _ts in U]
        g_u = np.stack(reps).mean(axis=0)
        assert out["estimates"][name]["G_U"]["ci95"] == [float(v) for v in np.percentile(g_u, [2.5, 97.5])]
        assert out["estimates"][name]["G"]["heldout72_hpfix"]["ci95"] == \
            [float(v) for v in np.percentile(reps[0], [2.5, 97.5])]


def test_the_discordance_is_b_plus_c_only(st):
    out = _read(st, _cells(_by_k(UP)))
    assert out["discordance"] == {"pairs": 48, "discordant": 24, "share": 0.5}


# ------------------------------------------------------------------------------------------------ peer drops
def _mismatch_captures(st: Path, ts: str, seed: int, game: int, **kw: Any) -> Dict[str, Any]:
    for lab, cap in (("pstar", synthetic(20)), ("pstar_s", synthetic(20, **kw))):
        d = st / "anchor" / "sp" / lab / "cells" / _name(lab, ts, seed, game) / "capture"
        d.mkdir(parents=True, exist_ok=True)
        (d / "battle-gen3ou-1.json").write_text(json.dumps(cap))
    return {"deal": f"{ts}:{seed}:{game}", "searched": _name("pstar_s", ts, seed, game),
            "raw": _name("pstar", ts, seed, game), "why": "p2's protocol differs at line 9 before turn 10"}


def test_a_peer_first_mismatch_drops_the_deal(st):
    m = _mismatch_captures(st, "p72", 2, 1, peer_at=3)
    out = _read(st, _cells(_by_k(UP), mismatches=[m]))
    assert out["decision"]["class"] == SP.GO_LEAKY
    pc = out["peer_drops"]["classes"]
    assert [(c["deal"], c["class"], c["peer_drop"]) for c in pc] == [("p72:2:1", "peer_first", True)]
    c = out["counts"]["by_panel"]["p72"]
    assert c["pre_treatment"] == {"total": 1, "peer_drop": 1} and c["pairs"] == 15
    assert out["peer_drops"]["by_panel"]["p72"] == 1


@pytest.mark.parametrize("kw, cls", [(dict(ours_at=3), "ours_first"), (dict(peer_at=12), "peer_first_late"),
                                     (dict(engine_at=4), "unattributable")])
def test_any_other_mismatch_class_voids(st, kw, cls):
    m = _mismatch_captures(st, "p72", 2, 1, **kw)
    d = _read(st, _cells(_by_k(UP), mismatches=[m]))["decision"]
    assert d["class"] == SP.VOID and any(f"class {cls}" in r for r in d["reasons"]), d["reasons"]


def test_an_unreadable_capture_voids(st):
    m = {"deal": "p72:2:1", "searched": _name("pstar_s", "p72", 2, 1), "raw": _name("pstar", "p72", 2, 1),
         "why": "x"}
    d = _read(st, _cells(_by_k(UP), mismatches=[m]))["decision"]
    assert d["class"] == SP.VOID and any("class unreadable" in r for r in d["reasons"])


# ------------------------------------------------------------------------------------------------ VOID
def _set(pred: Callable[[Dict[str, Any]], bool], **kv: Any) -> Callable[[List[Dict[str, Any]]], None]:
    def m(units: List[Dict[str, Any]]) -> None:
        for u in units:
            if pred(u):
                u.update(copy.deepcopy(kv))
    return m


def _first(label: str) -> Callable[[Dict[str, Any]], bool]:
    return lambda u: u["label"] == label and u["panel"] == "p72" and u["unit_seed"] == 1 and u["game"] == 2


@pytest.mark.parametrize("mutate, over, needle", [
    (_set(_first("pstar"), engine="port=cccccccccccccccc/10;env=bbbbbbbbbbbbbbbb/20"), {}, "span 2 engines"),
    (_set(_first("pstar_s"), engine=None), {}, "unknown engine"),
    (None, {"engine": "port=dddddddddddddddd/1;env=eeeeeeeeeeeeeeee/2"}, "not the registered"),
    (_set(_first("pstar"), model_sha256="b" * 64), {}, "played other bytes"),
    (_set(_first("pstar_s"), our_regime="greedy+search=r0s4@10"), {}, "search config"),
    (_set(_first("pstar_s"), search_played={"search": SEARCH, "our_search": SEARCH,
                                            "preset": PRESETS["r0s4"].as_dict()}), {}, "search config"),
    (_set(_first("pstar_s"), search_played=None), {}, "search config"),
    (_set(_first("pstar"), our_regime=f"greedy+search={SEARCH}"), {}, "raw arm"),
    (_set(_first("pstar"), search_played={"search": SEARCH}), {}, "raw arm"),
    (_set(_first("pstar"), games=[["ours_challenge", 3, "t.txt", "tie", 30]]), {}, "games.jsonl disagrees"),
    (_set(lambda u: u["panel"] == "forum72" and u["unit_seed"] > 0, counted=False), {}, "forum72: 1 unit(s) hold a pair"),
])
def test_each_void_reason(st, mutate, over, needle):
    assert _read(st, _cells(_by_k(UP)))["decision"]["class"] == SP.GO_LEAKY      # passes without it
    d = _read(st, _cells(_by_k(UP), mutate=mutate), **over)["decision"]
    assert d["class"] == SP.VOID and d["verdict"] == SP.VOID and any(needle in r for r in d["reasons"]), d["reasons"]


def test_void_on_post_and_pre_treatment_unpaired_above_their_caps(st):
    gone = [_name("pstar_s", "forum72", 0, g) for g in (0, 1)]
    cells = _cells(_by_k(UP), mutate=lambda us: [_miss("pstar_s", "forum72", 0, g)(us) for g in (0, 1)],
                   exhausted=gone)
    assert _read(st, cells, max_post_unpaired_frac=0.125)["decision"]["class"] == SP.GO_LEAKY     # 2 / 16 = the cap
    d = _read(st, cells, max_post_unpaired_frac=0.1)["decision"]
    assert d["class"] == SP.VOID and any("post-treatment unpaired" in r for r in d["reasons"])
    m = _mismatch_captures(st, "p72", 2, 1, peer_at=3)
    cells = _cells(_by_k(UP), mismatches=[m])
    assert _read(st, cells, max_pre_unpaired_frac=0.0625)["decision"]["class"] == SP.GO_LEAKY      # 1 / 16
    d = _read(st, cells, max_pre_unpaired_frac=0.05)["decision"]
    assert d["class"] == SP.VOID and any("pre-treatment unpaired" in r for r in d["reasons"])


def test_void_on_g_prefix_not_run_or_an_unchecked_pair(st):
    cells = _cells(_by_k(UP))
    cells["anchor"]["sp"]["g_prefix"] = None
    assert "G-PREFIX: not run on the group" in _read(st, cells)["decision"]["reasons"]
    cells = _cells(_by_k(UP))
    cells["anchor"]["sp"]["g_prefix"]["checked"] -= 1
    assert any("every pair must be checked" in r for r in _read(st, cells)["decision"]["reasons"])


def test_void_on_a_resolved_model_on_other_bytes(st):
    cells = _cells(_by_k(UP))
    cells["models"]["pstar_s"]["sha256"] = "c" * 64
    assert any("'pstar_s' resolved to" in r for r in _read(st, cells)["decision"]["reasons"])


@pytest.mark.parametrize("over, needle", [
    ({"verdict": "FAIL", "reason": "1 mismatch(es)"}, "reads 'FAIL'"),
    ({"inputs": {"cells": "/elsewhere", "panels": U, "engine": ENGINE, "model_sha256": SHA, "max": 600}}, "cells"),
    ({"inputs": {"cells": "X", "panels": ["forum72"], "engine": ENGINE, "model_sha256": SHA, "max": 600}}, "panels"),
    ({"inputs": {"cells": "X", "panels": U, "engine": "E2", "model_sha256": SHA, "max": 600}}, "engine"),
    ({"inputs": {"cells": "X", "panels": U, "engine": ENGINE, "model_sha256": "b" * 64, "max": 600}}, "model_sha256"),
    ({"rescored": []}, "registered-order first 1"),
    ({"n_rescored": 0}, "fewer than min_rescored"),
    ({"inputs": {"cells": "X", "panels": U, "engine": ENGINE, "model_sha256": SHA, "max": 50}},
     "not the registered max_rescored"),
])
def test_the_identity_guard_voids(st, over, needle):
    if "inputs" in over and over["inputs"]["cells"] == "X":
        over["inputs"]["cells"] = str(_cells_dir(st))
    _rescore_json(st, **over)
    d = _read(st, _cells(_by_k(UP)))["decision"]
    assert d["class"] == SP.VOID and any(r.startswith("identity:") and needle in r for r in d["reasons"]), d["reasons"]


def test_a_missing_rescore_json_voids_and_its_sha_is_recorded(st):
    out = _read(st, _cells(_by_k(UP)))
    assert len(out["identity"]["rescore_json"]["sha256"]) == 64 and out["identity"]["n_rescored"] == 1
    (st / "rescore.json").unlink()
    d = _read(st, _cells(_by_k(UP)))["decision"]
    assert d["class"] == SP.VOID and any(r.startswith("identity:") and "missing" in r for r in d["reasons"]), d["reasons"]


def test_a_record_the_rescore_did_not_cover_voids(st):
    from main.anchors.search_side import rescore_sampled

    key = next(k for k in (f"su_live_v1|y{j}|t12|n13" for j in range(999)) if rescore_sampled(k))
    cell = _cells_dir(st) / _name("pstar_s", "heldout72_hpfix", 0, 0)
    cell.mkdir(parents=True)
    (cell / "search_decisions.jsonl").write_text(json.dumps({"key": key, "rescore_sample": True}) + "\n")
    d = _read(st, _cells(_by_k(UP)))["decision"]
    assert d["class"] == SP.VOID and any("registered-order first 2" in r for r in d["reasons"])


def test_the_search_report_over_paired_games(st):
    s = _read(st, _cells(_by_k(UP)))["search"]
    assert s["U"]["games"] == 48 and s["U"]["deviation_rate"] == 0.1 and s["forum72"]["excluded_per_game"] == 1.0
    assert s["U"]["search_cpu_s_per_game"] == 600.0


# ------------------------------------------------------------------------------------------------ S′-HONEST
HSEARCH = "r0s4_honest@10"
HPRESET = PRESETS["r0s4_honest"].as_dict()


def _honest(units: List[Dict[str, Any]]) -> None:
    """Stamp every searched unit as the leak-honest preset played it."""
    for u in units:
        if u["label"] == "pstar_s":
            u.update(our_regime=f"greedy+search={HSEARCH}", search=HSEARCH)
            u["search_played"] = dict(u["search_played"], search=HSEARCH, our_search=HSEARCH, preset=HPRESET)


def _honest_rescore(st: Path, fields: Any = None) -> None:
    f = list(LI.RESCORE_FIELDS + LI.HONEST_FIELDS) if fields is None else fields
    _rescore_json(st, inputs={"cells": str(_cells_dir(st)), "panels": U, "engine": ENGINE, "model_sha256": SHA,
                              "max": 600, "min": 1, "fields": f})


def test_a_leak_honest_preset_reads_go_honest_and_goal_b_reached(st):
    _honest_rescore(st)
    out = _read(st, _cells(_by_k(UP), mutate=_honest), expected_search=HSEARCH)
    d = out["decision"]
    assert d["class"] == SP.GO_HONEST and d["verdict"].startswith("GO-HONEST: ") and "(GO-HONEST iff > 0)" in d["verdict"]
    assert d["goal_b"]["reached"] is True and d["goal_b"]["label"] == "GOAL-B-REACHED"
    assert "GOAL-B-REACHED" in d["verdict"] and d["excess_exclusions"] is None
    leaky = _read(st, _cells(_by_k(UP)))["decision"]                  # the same read on r0s4_rolls: GO-LEAKY, leaky goal b
    assert leaky["class"] == SP.GO_LEAKY and leaky["goal_b"]["label"] == "GOAL-B-REACHED (leaky)"


def test_a_null_never_reaches_goal_b(st):
    _honest_rescore(st)
    d = _read(st, _cells(_by_k(FLAT), mutate=_honest), expected_search=HSEARCH)["decision"]
    assert d["class"] == SP.NULL and d["goal_b"]["reached"] is False and d["goal_b"]["label"] is None


def test_a_leak_honest_rescore_that_skipped_the_build_record_voids(st):
    _honest_rescore(st, fields=list(LI.RESCORE_FIELDS))
    d = _read(st, _cells(_by_k(UP), mutate=_honest), expected_search=HSEARCH)["decision"]
    assert d["class"] == SP.VOID and any("must compare" in r for r in d["reasons"]), d["reasons"]


ISEARCH = "r0s4_honest_inc@10"
IPRESET = PRESETS["r0s4_honest_inc"].as_dict()


def _incremental(units: List[Dict[str, Any]]) -> None:
    """Stamp every searched unit as the INCREMENTAL leak-honest preset played it."""
    for u in units:
        if u["label"] == "pstar_s":
            u.update(our_regime=f"greedy+search={ISEARCH}", search=ISEARCH)
            u["search_played"] = dict(u["search_played"], search=ISEARCH, our_search=ISEARCH, preset=IPRESET)


def _inc_identity(st: Path, carry: Any = "rebuild") -> Dict[str, Any]:
    return {"cells": str(_cells_dir(st)), "rescore_json": str(st / "rescore.json"), "min_rescored": 1,
            "max_rescored": MAXR, "carry": carry}


def _inc_rescore(st: Path, carry: Any = "rebuild") -> None:
    _rescore_json(st, inputs={"cells": str(_cells_dir(st)), "panels": U, "engine": ENGINE, "model_sha256": SHA,
                              "max": 600, "min": 1, "fields": list(LI.RESCORE_FIELDS + LI.HONEST_FIELDS),
                              "carry": carry})


def test_the_rule_and_the_live_searcher_name_the_same_leak_honest_dice():
    from main.anchors.search_side import HONEST_DICE

    assert tuple(SP.HONEST_DICE) == tuple(HONEST_DICE) and SP.INC_DICE in HONEST_DICE
    assert SP.go_class(IPRESET) == SP.go_class(HPRESET) == SP.GO_HONEST and SP.go_class(PRESET) == SP.GO_LEAKY


def test_an_incremental_preset_reads_go_honest_with_its_registered_carry(st):
    """S′-HONEST on ``r0s4_honest_inc`` (S′H-14): GO-HONEST, and the identity guard holds the re-score to the
    registered carry mode; a re-score in the other mode, or one that did not compare the build record, VOIDs."""
    _inc_rescore(st)
    d = _read(st, _cells(_by_k(UP), mutate=_incremental), expected_search=ISEARCH, identity=_inc_identity(st))
    assert d["decision"]["class"] == SP.GO_HONEST and d["decision"]["goal_b"]["label"] == "GOAL-B-REACHED"
    assert d["distill"]["fields_searched"][-1] == "search_key"
    _inc_rescore(st, carry="seeded")
    v = _read(st, _cells(_by_k(UP), mutate=_incremental), expected_search=ISEARCH, identity=_inc_identity(st))
    assert v["decision"]["class"] == SP.VOID and any("carry 'seeded', not the registered 'rebuild'" in r
                                                      for r in v["decision"]["reasons"]), v["decision"]["reasons"]
    _honest_rescore(st)                                     # a re-score written before the carry existed: no mode
    v = _read(st, _cells(_by_k(UP), mutate=_incremental), expected_search=ISEARCH, identity=_inc_identity(st))
    assert v["decision"]["class"] == SP.VOID and any("carry None" in r for r in v["decision"]["reasons"])
    # the cold builder's read never registers a carry, and its re-score never ran one
    _honest_rescore(st)
    assert _read(st, _cells(_by_k(UP), mutate=_honest), expected_search=HSEARCH)["decision"]["class"] == SP.GO_HONEST


def test_goal_b_imputes_a_missing_searched_game_as_a_loss(st):
    # forum72 seed 0 (k = 1): game 0 (raw loss, searched WIN) loses its searched unit -> a loss here; game 1
    # (win, win) loses its raw unit -> its counted searched win stays a win
    def both(units):
        _miss("pstar_s", "forum72", 0, 0)(units)
        _miss("pstar", "forum72", 0, 1)(units)
    cells = _cells(_by_k(UP), mutate=both, exhausted=[_name("pstar_s", "forum72", 0, 0), _name("pstar", "forum72", 0, 1)])
    out = _read(st, cells, max_post_unpaired_frac=0.2)
    # paired searched wins on forum72: seed 0 keeps games 2 (loss) and 3 (win) -> 1; seeds 1-3 (k 3, 1, 3) -> 4 + 3 + 4
    assert out["score_u"]["searched"]["counts"]["forum72"] == [12, 14]
    assert out["decision"]["goal_b"]["counts"]["forum72"] == [13, 16]
    assert out["decision"]["goal_b"]["counts"]["p72"] == out["score_u"]["searched"]["counts"]["p72"] == [14, 16]


def test_goal_b_within_eps_of_half_is_never_reached(monkeypatch):
    from main.reads import rules as R

    fake = SimpleNamespace(mean=0.6, ci=(0.5004, 0.7))
    monkeypatch.setattr(R, "strat_score", lambda counts, panels: fake)
    pairs = {ts: {0: [(Fraction(0), Fraction(1))]} for ts in U}
    tally = {ts: {"post_deals": []} for ts in U}
    g = SP.goal_b(pairs, tally, U, SP.GO_HONEST, 0.0005)
    assert g["reached"] is False and g["lb_above_0_50"] == {"crosses": False, "near": True}
    fake.ci = (0.5006, 0.7)
    assert SP.goal_b(pairs, tally, U, SP.GO_HONEST, 0.0005)["reached"] is True
    assert SP.goal_b(pairs, tally, U, SP.STOP, 0.0005)["reached"] is False


@pytest.mark.parametrize("bound, excess, near", [(0.05, False, False), (0.03, True, False), (0.0325, True, True)])
def test_the_exclusion_flag_narrows_and_never_voids(st, bound, excess, near):
    # every searched unit: 30 searched, 1 excluded -> a share of 1/31 = 0.03226 on every panel
    out = _read(st, _cells(_by_k(UP)), max_excluded_frac={ts: bound for ts in U})
    ex = out["exclusions"]["by_panel"]
    assert ex["forum72"]["exact"] == "1/31" and ex["U"]["exact"] == "1/31" and ex["U"]["units"] == 48
    assert ex["p72"]["test"] == {"bound": bound, "exceeds": excess, "near": near}
    d = out["decision"]
    assert d["class"] == SP.GO_LEAKY and d["excess_exclusions"] is excess
    assert ("excess_exclusions" in d["verdict"]) is excess


def test_the_exclusion_flag_reads_one_panel_alone(st):
    def more(units):
        for u in units:
            if u["label"] == "pstar_s" and u["panel"] == "forum72":
                u["search_played"] = dict(u["search_played"], searched=20, excluded=10)
    out = _read(st, _cells(_by_k(UP), mutate=more), max_excluded_frac={"heldout72_hpfix": 0.05, "forum72": 0.30,
                                                                       "p72": 0.05})
    ex = out["exclusions"]["by_panel"]
    assert ex["forum72"]["exact"] == "1/3" and ex["forum72"]["test"]["exceeds"] is True
    assert ex["p72"]["test"]["exceeds"] is False and out["decision"]["excess_exclusions"] is True


def test_the_distillation_fields_are_reported(st):
    cell = _cells_dir(st) / _name("pstar_s", "p72", 2, 3)
    cell.mkdir(parents=True)
    full = {"searched": True, "q": {"1": 0.5}, "log_pi": {"1": -0.1}, "root_v": 0.5, "frame_sha256": "f",
            "log_sha256": "l", "key": "k1"}
    rows = [full, dict(full, q=None, key="k2"), {"searched": False, "why": "excluded", "frame_sha256": "f",
                                                 "log_sha256": "l", "key": "k3"},
            {"searched": False, "why": "before_turn"}]
    (cell / "search_decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (cell / "capture").mkdir()
    (cell / "capture" / "battle-gen3ou-1.json").write_text("{}")
    out = _read(st, _cells(_by_k(UP)))
    p72 = out["distill"]["by_panel"]["p72"]
    assert (p72["units"], p72["searched"], p72["excluded"], p72["lacking"], p72["by_field"]) == (16, 2, 1, 1, {"q": 1})
    assert p72["units_without_capture"] == 15 and out["distill"]["fields_searched"] == list(SP.DISTILL_SEARCHED)
    _honest_rescore(st)
    hon = _read(st, _cells(_by_k(UP), mutate=_honest), expected_search=HSEARCH)["distill"]
    assert hon["fields_searched"][-1] == "search_key" and hon["by_panel"]["p72"]["by_field"] == {"q": 1, "search_key": 3}
    # the checkpoint's form: every complete attempt dir on disk, counted or not, never a .partial one
    part = _cells_dir(st) / (_name("pstar_s", "p72", 2, 3) + ".partial.1")
    part.mkdir()
    (part / "search_decisions.jsonl").write_text(json.dumps(dict(full, q=None)) + "\n")
    disk = SP.distill_cells(_cells_dir(st), honest=False)
    assert set(disk["by_panel"]) == {"forum72", "p72"} and disk["by_panel"]["p72"]["units"] == 1
    assert disk["by_panel"]["p72"]["by_field"] == {"q": 1} and disk["by_panel"]["p72"]["units_without_capture"] == 0


# ------------------------------------------------------------------------------------------------ the spec check
def _spec(st: Path, **over: Any) -> Dict[str, Any]:
    ref = {"kind": "pinned", "path": "/x/checkpoint_280000170_steps.zip", "sha256": SHA}
    g = {"opponent": "metamon:Kakuna", "teamsets": U, "unit_seeds": "300-344", "per_game": True,
         "models": ["pstar", "pstar_s"]}
    g.update(over.pop("group_over", {}))
    return {"schema": "gen3_read_spec_v1", "read": "sprime_demo_2026-11-01",
            "models": {"pstar": {"arm": "pstar", "seed": 2601, "ref": ref},
                       "pstar_s": {"arm": "pstar_s", "seed": 2601, "ref": ref, "load": {"search": "r0s4_rolls"}}},
            "anchor": {"sp": g}, "decision": {"rule": "searched_play", "params": _params(st, **over)}}


def test_the_spec_check_accepts_the_registered_params(tmp_path):
    spec = parse(_spec(tmp_path))
    assert spec.rule == "searched_play" and spec.params["expected_search"] == SEARCH


@pytest.mark.parametrize("over, needle", [
    ({"bogus": 1}, "unknown 'bogus'"),
    ({"group": "nope"}, "is not an anchor group"),
    ({"searched": "pstar"}, "exactly the searched label"),
    ({"expected_search": "r0s4_rolls"}, "spell it 'r0s4_rolls@10'"),
    ({"expected_search": "r0s4@10"}, "not expected_search"),
    ({"u_panels": ["heldout72_hpfix", "forum72"]}, "must be exactly group"),
    ({"u_panels": ["home72", *U]}, "must be exactly group"),
    ({"null_margin_pp": 0}, "null_margin_pp"),
    ({"bootstrap": {"B": 100, "seed": 1}}, "bootstrap"),
    ({"rule8_eps": 0.5}, "rule8_eps"),
    ({"max_post_unpaired_frac": 0.3}, "max_post_unpaired_frac"),
    ({"max_pre_unpaired_frac": -0.1}, "max_pre_unpaired_frac"),
    ({"state": "relative/dir"}, "state"),
    ({"identity": {"cells": "/x", "rescore_json": "/y"}}, "identity"),
    ({"identity": {"cells": "/x/cells", "rescore_json": "/y", "min_rescored": 1, "max_rescored": MAXR}}, "not the searched label's cells"),
    ({"identity": {"cells": "/x/cells", "rescore_json": "rel", "min_rescored": 0, "max_rescored": 1}}, "min_rescored"),
    ({"max_excluded_frac": {"forum72": 0.1}}, "max_excluded_frac"),
    ({"max_excluded_frac": {"heldout72_hpfix": 0.1, "forum72": 1.5, "p72": 0.1}}, "max_excluded_frac"),
    ({"identity": {"cells": "/x/cells", "rescore_json": "/y", "min_rescored": 1, "max_rescored": 1,
                   "carry": "rebuild"}}, "for an incremental preset only"),
])
def test_the_spec_check_refuses(tmp_path, over, needle):
    with pytest.raises(SpecError, match="searched_play") as e:
        parse(_spec(tmp_path, **over))
    assert needle in str(e.value)


def test_a_pinned_leak_honest_read_holds_its_rescore_to_the_pins_play_path(st):
    """M5 (review): the re-score must have run a clean checkout whose play path is the pin's; a missing record, a dirty
    tree or another play path is a problem (VOID). ``f7da935ab`` and ``335355e6f`` differ on the play path (S′H-15's
    docstring edit to ``honest_inc.py``); a commit against itself does not."""
    pin = "335355e6f"
    assert SP._rescore_at_pin({"commit": pin, "dirty": False}, pin) == []
    assert "DIRTY" in SP._rescore_at_pin({"commit": pin, "dirty": True}, pin)[0]
    assert "must record its checkout" in SP._rescore_at_pin(None, pin)[0]
    assert "play path differs" in SP._rescore_at_pin({"commit": "f7da935ab", "dirty": False}, pin)[0]
    # the same play path but another comparator or root guard (live_identity / sign_gate are OFF the play path): refused
    from main.reads import playpath as PP

    real_compare, real_git = PP.compare, PP._git
    try:
        PP.compare = lambda a, b, repo=None: ([], "d", 1)
        PP._git = lambda repo, *a, stdin=None: (b"x" if a[-1].startswith(pin) else b"y") \
            if a[-1].endswith("live_identity.py") else b"same"
        got = SP._rescore_at_pin({"commit": "abc123", "dirty": False}, pin)
        assert got and "re-score code differs" in got[0] and "live_identity.py" in got[0] and "sign_gate" not in got[0]
        PP._git = lambda repo, *a, stdin=None: b"same"
        assert SP._rescore_at_pin({"commit": "abc123", "dirty": False}, pin) == []
    finally:
        PP.compare, PP._git = real_compare, real_git
    _inc_rescore(st)                                   # no inputs.code: a pinned read VOIDs, an unpinned one reads
    pinned = SimpleNamespace(models=SPEC.models, pin=pin)
    cells = _cells(_by_k(UP), mutate=_incremental)
    p = _params(st, expected_search=ISEARCH, identity=_inc_identity(st))
    doc = json.loads(json.dumps(RU.inputs("searched_play", cells, p, pinned)))
    assert any("must record its checkout" in x for x in doc["identity"]["problems"])
    assert RU.decide("searched_play", doc, p)["decision"]["class"] == SP.VOID
    assert _read(st, cells, expected_search=ISEARCH, identity=_inc_identity(st))["decision"]["class"] == SP.GO_HONEST


def _inc_spec(st: Path, identity: Dict[str, Any]) -> Dict[str, Any]:
    sp = _spec(st, expected_search=ISEARCH, identity=identity)
    sp["models"]["pstar_s"]["load"] = {"search": ISEARCH}
    return sp


def test_the_spec_check_holds_an_incremental_preset_to_a_carry_mode(tmp_path):
    """``r0s4_honest_inc`` registers how its re-score gets each state's carry (``identity.carry``), one of
    ``live_identity.CARRY_MODES``; missing or unknown is refused."""
    base = _inc_identity(tmp_path)
    for mode in LI.CARRY_MODES:
        assert parse(_inc_spec(tmp_path, dict(base, carry=mode))).params["identity"]["carry"] == mode
    for bad, needle in (({k: v for k, v in base.items() if k != "carry"}, "how the re-score gets an incremental"),
                        (dict(base, carry="cold"), "identity.carry: one of")):
        with pytest.raises(SpecError, match="searched_play") as e:
            parse(_inc_spec(tmp_path, bad))
        assert needle in str(e.value)


def test_the_rule_check_refuses_a_group_that_is_not_per_game(tmp_path):
    spec = parse(_spec(tmp_path))
    g = SimpleNamespace(**{**spec.anchor["sp"].__dict__, "per_game": False})
    assert any("is not per_game" in x for x in SP.check(spec.params, models=spec.models, anchor={"sp": g}, s={}))


# ------------------------------------------------------------------------------------------------ assemble
def test_assemble_leaves_a_g_prefix_mismatch_to_a_rule_that_owns_it(tmp_path, monkeypatch):
    """With ``searched_play`` the diverged pair stays in ``g_prefix.mismatches`` and the group is NOT inconclusive
    for it (the rule classifies it); with any other rule it is (``search_read_test``)."""
    from main.reads import api
    from main.reads import runner as RN
    from main.reads import search_read_test as SR
    from utils.hashing import sha256_file

    z = tmp_path / "ckpt" / "checkpoint_280000170_steps.zip"
    z.parent.mkdir()
    (z.parent / "model_config.json").write_text("{}")
    z.write_bytes(b"pstar")
    state = tmp_path / "state"
    doc = SR._doc(str(z), sha256_file(z))
    doc["decision"] = {"rule": "searched_play", "params": dict(
        _params(state, u_panels=["home72", "forum72"], expected_search="r0s4@10", group="su"),
        identity={"cells": str(state / "anchor" / "su" / "pstar_s" / "cells"), "rescore_json": "/nope.json",
                  "min_rescored": 1, "max_rescored": MAXR})}
    spec_p = tmp_path / "read_spec.json"
    spec_p.write_text(json.dumps(doc))
    RN.plan(spec_p, state, code=SR.CODE, step_of=lambda p: 280000170)
    monkeypatch.setattr(RN, "checkout_code", lambda root=None: SR.CODE)
    assert RN.run(state, "anchor", time.time() + 3600, run_unit=SR._fake_unit([], diverge=(("forum72", 0), 9)),
                  sleep=lambda s: None) == 0
    grp = api.assemble(state)["anchor"]["su"]
    assert grp["g_prefix"]["mismatches"][0]["deal"] == "forum72:0:0"
    assert not any(w.startswith("G-PREFIX") for w in grp["inconclusive"])
    assert RU.RULES["searched_play"].owns_g_prefix and not RU.RULES["search_headroom"].owns_g_prefix


# ------------------------------------------------------------------------------------------------ the timer flag
TIMER = {"reserve_s": 10.0, "overhead_s": 1.0, "max_cut_frac": 0.5}


def _decision_logs(st: Path, slow: Optional[Dict[str, float]] = None) -> None:
    """Every counted searched unit's decision log: turns 1-9 off the path, then 30 searched decisions of 3 s; a unit
    named in ``slow`` spends that many seconds on its turn-12 decision. The state's one sampled record stays (last,
    off the path) and the re-score JSON is re-derived."""
    slow = slow or {}
    for ts in U:
        for s in SEEDS:
            for g in GAMES:
                name = _name("pstar_s", ts, s, g)
                d = _cells_dir(st) / name
                d.mkdir(parents=True, exist_ok=True)
                p = d / "search_decisions.jsonl"
                kept = [dict(json.loads(x), turn=40) for x in p.read_text().splitlines()] if p.is_file() else []
                rows = [{"turn": t, "searched": False, "why": "before_turn"} for t in range(1, 10)]
                rows += [{"turn": t, "searched": True, "wall_s": slow.get(name, 3.0) if t == 12 else 3.0}
                         for t in range(10, 40)]
                p.write_text("".join(json.dumps(r) + "\n" for r in rows + kept))
    _rescore_json(st)


def test_timer_safe_is_set_on_zero_forfeits_and_a_cut_share_under_its_bound(st):
    # forum72 0/0's 145-s decision passes the bank-aware deadline (150 - 10 - 1 = 139 s): cut; its 6-s overrun fits the
    # next decision's deadline, so one cut in 48 x 30 searched decisions; a 300-s one forfeits as played
    _decision_logs(st, slow={_name("pstar_s", "forum72", 0, 0): 145.0, _name("pstar_s", "p72", 2, 2): 300.0})
    out = _read(st, _cells(_by_k(UP)), timer=dict(TIMER, max_cut_frac=0.05))
    f = out["timer"]["flag"]
    p72 = out["timer"]["by_panel"]["p72"]["cut_searched"]
    assert p72 > 2                                       # the 300-s step running on cuts its successors too
    assert (f["games"], f["searched"], f["forfeits"], f["cut_searched"]) == (48, 1440, 0, 1 + p72)
    assert f["safe"] is True and out["timer"]["by_panel"]["forum72"]["cut_searched"] == 1
    assert out["timer"]["by_panel"]["heldout72_hpfix"]["cut_searched"] == 0
    d = out["decision"]
    assert d["class"] == SP.GO_LEAKY and d["timer_safe"] is True and d["verdict"].endswith("; TIMER-SAFE")
    # the reported table: the prototype's policies; as played the 300-s decision forfeits its game, the 145-s one not
    assert out["timer"]["table"]["policies"]["as_played"]["U"]["counts"]["forfeit"] == 1


@pytest.mark.parametrize("bound, safe, near", [(0.002, True, False), (0.0001, False, False), (0.001, False, True),
                                               (Fraction(1, 1440), False, True)])
def test_timer_safe_holds_the_cut_share_to_its_bound_exactly_and_near_is_never_safe(st, bound, safe, near):
    # one cut: the share is 1/1440 = 0.000694; within rule8_eps 0.0005 of the bound is near, never safe (rule 8)
    _decision_logs(st, slow={_name("pstar_s", "forum72", 0, 0): 145.0})
    out = _read(st, _cells(_by_k(UP)), timer=dict(TIMER, max_cut_frac=float(bound)))
    f = out["timer"]["flag"]
    assert f["exact"] == "1/1440" and f["safe"] is safe and f["cut_test"]["near"] is near
    d = out["decision"]
    assert d["class"] == SP.GO_LEAKY and d["timer_safe"] is safe                    # never moves the class
    assert d["verdict"].endswith("; TIMER-SAFE" if safe else "; NOT TIMER-SAFE")


def test_a_forfeit_in_the_replay_is_never_timer_safe(st):
    # a reserve of 0 answers at the request's whole limit: the cut decision forfeits its game
    _decision_logs(st, slow={_name("pstar_s", "p72", 1, 1): 300.0})
    out = _read(st, _cells(_by_k(UP)), timer=dict(TIMER, reserve_s=0.0, max_cut_frac=0.9))
    f = out["timer"]["flag"]
    assert f["forfeits"] == 1 and f["safe"] is False and out["decision"]["timer_safe"] is False


def test_no_timer_block_leaves_the_record_as_before(st):
    out = _read(st, _cells(_by_k(UP)))
    assert "timer" not in out and "timer_safe" not in out["decision"]
    assert "TIMER-SAFE" not in out["decision"]["verdict"]


def test_a_counted_searched_unit_without_a_decision_log_is_refused_when_a_timer_is_registered(st):
    with pytest.raises(Exception, match="no decision log"):
        _read(st, _cells(_by_k(UP)), timer=TIMER)


def test_the_safety_nets_recorded_cuts_are_counted_beside_the_exclusions(st):
    def net(units):
        for u in units:
            if u["label"] == "pstar_s" and u["panel"] == "p72":
                u["search_played"] = dict(u["search_played"], net_cut=3)
    ex = _read(st, _cells(_by_k(UP), mutate=net))["exclusions"]["by_panel"]
    assert ex["p72"]["net_cut"] == 48 and ex["forum72"]["net_cut"] == 0 and ex["U"]["net_cut"] == 48
    assert ex["p72"]["net_cut_share_of_searched"] == 48 / 480 and ex["p72"]["exact"] == "1/31"   # not an exclusion


@pytest.mark.parametrize("timer, needle", [
    ({"reserve_s": 10.0, "overhead_s": 1.0}, "timer: {reserve_s, overhead_s, max_cut_frac}"),
    (dict(TIMER, reserve_s=-1), "timer.reserve_s"),
    (dict(TIMER, max_cut_frac=1.0), "timer.max_cut_frac"),
])
def test_the_spec_check_refuses_a_malformed_timer_block(tmp_path, timer, needle):
    with pytest.raises(SpecError, match="searched_play") as e:
        parse(_spec(tmp_path, timer=timer))
    assert needle in str(e.value)


def test_a_preset_with_a_safety_net_registers_the_net_it_played():
    assert SP._check_timer(dict(TIMER, max_cut_frac=0.02), "r0s4_honest_inc_k4@10") == []
    for bad in (dict(TIMER, reserve_s=5.0), dict(TIMER, overhead_s=0.0)):
        got = SP._check_timer(bad, "r0s4_honest_inc_cap@10")
        assert got and "the replay must be that net" in got[0]
    assert SP._check_timer(dict(TIMER, reserve_s=5.0), "r0s4_honest_inc@10") == []        # no net: any replay
