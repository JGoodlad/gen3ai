"""``main.reads`` rule ``n4_screen`` (N4 PREREG §5 + its review's M2 / M4 / M5; X48 PREREG §5, §8, §10) on constructed
cells: every class boundary at rule-8-safe inputs (exactly on a bar, within eps, clearly past), the class order, MIXED
never GO, the LR-confounded label, a meter on two engines (or an S cell short of its pairs, or K short beyond one unit)
INCONCLUSIVE, the near scope, the reported-only slope, the X35-stage record shape and the spec-load check, plus X48's
draft read spec. Synthetic cells, no game."""
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest

from main.reads import rules as RU
from main.reads import rules_n4_screen as N4
from main.x35.readout import welch_interval

ON = ["x48kffn_s3501", "x48kffn_s3502", "x48kffn_s3503"]
OFF = ["c0_s3501", "c0_s3502", "c0_s3503"]
SEED = {lab: 3501 + i for labs in (ON, OFF) for i, lab in enumerate(labs)}
ENGINE_K = "port=aaaa/1;env=bbbb/2"
ENGINE_S = "env=bbbb/2"
UNITS = 50
SCOPE = {"lever": "per-kind FFN experts", "dose": "20M", "start": "P*"}


def _params(**over: Any) -> Dict[str, Any]:
    p = {"on": list(ON), "off": list(OFF), "s_group": "pool", "anchor_group": "unfamiliar", "s_bar": 2.0,
         "k_bar": 0.015, "confidence": 0.90, "z": 1.6449, "rule8_eps": 1e-9, "mixed_veto": True,
         "void": {"k9b_dir": "k9b", "runs": {lab: f"models/rb_x35_x48kffn_s{SEED[lab]}" for lab in ON}},
         "nogo_scope": dict(SCOPE)}
    p.update(over)
    return p


def _spec(tmp: Path, lr: Optional[Dict[str, float]] = None) -> SimpleNamespace:
    models = {}
    for lab in ON + OFF:
        run = tmp / lab
        run.mkdir(exist_ok=True)
        if lr is not None and lab in lr:
            (run / "metadata.json").write_text(json.dumps(
                {"snapshot_history": {f"checkpoint_{i}_steps.zip": {"lr": lr[lab]} for i in range(20)}}))
        models[lab] = SimpleNamespace(arm="x48" if lab in ON else "c0", seed=SEED[lab], load={},
                                      ref=SimpleNamespace(kind="pinned", path=str(run / "final_model.zip"), run=None))
    return SimpleNamespace(models=models, s={"pool": SimpleNamespace(pairs=2000), "slope": SimpleNamespace(pairs=500)})


def _node(s: float, pairs: int = 2000) -> Dict[str, Any]:
    return {"s_pp": s, "se_pp": 0.3, "per_member_pp": {"m1": s - 1.0, "x26_pstar": s + 1.0},
            "n_pairs": {"m1": pairs, "x26_pstar": pairs}, "inconclusive": []}


def _cells(s: Dict[str, float], k: Dict[str, int], *, k_engines: Optional[Dict[str, str]] = None,
           s_engine: Optional[str] = ENGINE_S, pairs: Optional[Dict[str, int]] = None,
           lost_units: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """``s``: label -> S pp; ``k``: label -> forum72 wins of 1,000 games (50 units of 20); ``lost_units``: label ->
    units uncounted (20 games each, wins scaled)."""
    cells, units = {}, []
    for lab in ON + OFF:
        lost = (lost_units or {}).get(lab, 0)
        n = 20 * (UNITS - lost)
        cells[lab] = {"forum72": {"wins": k[lab] * n // 1000, "ties": 0, "n": n, "units": UNITS - lost,
                                  "planned": UNITS, "skipped": [f"{lab}__u{i}" for i in range(lost)]}}
        units += [{"unit": f"{lab}__forum72__{i:02d}", "label": lab, "panel": "forum72", "unit_seed": i,
                   "counted": i >= lost, "engine": (k_engines or {}).get(lab, ENGINE_K)} for i in range(UNITS)]
    return {"s": {"pool": {"engine": s_engine, "regime_id": "a" * 16, "inconclusive": [],
                           "nodes": {lab: _node(v, (pairs or {}).get(lab, 2000)) for lab, v in s.items()}}},
            "anchor": {"unfamiliar": {"teamsets": ["forum72"], "unit_seeds": list(range(UNITS)), "regime_id": "b" * 16,
                                      "inconclusive": [], "unplayed": 0, "units": units, "cells": cells}}}


S_GO = dict(zip(ON + OFF, [58.0, 58.5, 59.0, 56.5, 57.0, 57.5]))            # Δ +1.5, LB > 0
S_FLAT = dict(zip(ON + OFF, [57.2, 57.7, 58.2, 56.5, 57.0, 57.5]))          # Δ +0.7, LB < 0 < UB < 2
K_FLAT = dict(zip(ON + OFF, [490, 500, 510, 490, 500, 510]))


def _run(tmp: Path, cells: Dict[str, Any], p: Optional[Dict[str, Any]] = None,
         lr: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    p = p or _params()
    spec = _spec(tmp, lr if lr is not None else {lab: 1e-4 for lab in ON + OFF})
    doc = RU.inputs("n4_screen", cells, p, spec)
    return RU.decide("n4_screen", json.loads(json.dumps(doc)), p)


# ------------------------------------------------------------------------------------------- the meters, end to end
def test_s_is_welch_at_90_and_k_is_the_d1_estimators_with_phi_printed(tmp_path):
    from main.x35.d1 import newcombe_strat, overdispersion, strat_score, widen

    d = _run(tmp_path, _cells(S_GO, K_FLAT))
    w = welch_interval([58.0, 58.5, 59.0], [56.5, 57.0, 57.5], confidence=0.90)
    s = d["screen"]["s"]
    assert (s["delta"], s["ci"]) == (w.delta, list(w.ci))
    on = {str(SEED[lab]): {"forum72": (K_FLAT[lab], 1000)} for lab in ON}
    off = {str(SEED[lab]): {"forum72": (K_FLAT[lab], 1000)} for lab in OFF}
    so, sf = (strat_score({"forum72": (1500, 3000)}, ("forum72",), 1.6449) for _ in range(2))
    dk, lo, hi = newcombe_strat(so, sf)
    od = overdispersion(on, off, ("forum72",))
    assert d["screen"]["k"]["ci"] == list(widen(dk, lo, hi, od.factor))
    assert d["screen"]["k"]["phi_hat"] == od.phi and "φ̂" in d["screen"]["summary"]
    assert d["verdict"] == "GO (in-house)" and d["class"] == N4.GO
    assert N4.GO_TEXT in d["screen"]["text"][0] and "Δ̂_S +1.50 pp" in d["screen"]["text"][0]
    assert d["family_false_go"].startswith("≈ 0.09")


def test_the_record_is_an_x35_stage_record_whose_top_level_holds_no_r1(tmp_path):
    d = _run(tmp_path, _cells(S_FLAT, K_FLAT))
    assert (d["arm"], d["look"], d["final"]) == ("X48", 1, True)      # batch.READ_LABEL["x48"]; one look
    assert "r1" not in d and "h" not in d                          # batch.final_class must never read R1 as the class
    assert d["reported"]["r1_look1"]["look"] == 1
    assert RU.RULES["n4_screen"].x35_record and RU.RULES["n4_screen"].verdict_of(d) == d["verdict"]


# ------------------------------------------------------------------------------------------- INCONCLUSIVE
def test_k_units_on_two_engines_are_inconclusive(tmp_path):
    ok = _run(tmp_path, _cells(S_GO, K_FLAT))
    assert ok["class"] == N4.GO
    d = _run(tmp_path, _cells(S_GO, K_FLAT, k_engines={ON[0]: "port=cccc/1;env=dddd/2"}))
    assert d["class"] == d["verdict"] == N4.INCONCLUSIVE and d["screen"] is None
    assert any("2 engine(s)" in r for r in d["reasons"])


def test_s_cells_without_one_known_engine_are_inconclusive(tmp_path):
    d = _run(tmp_path, _cells(S_GO, K_FLAT, s_engine=None))
    assert d["class"] == N4.INCONCLUSIVE
    assert any("ONE known engine" in r for r in d["reasons"])


def test_an_s_cell_one_pair_below_the_registered_pairs_is_inconclusive(tmp_path):
    assert _run(tmp_path, _cells(S_GO, K_FLAT, pairs={ON[1]: 2000}))["class"] == N4.GO
    d = _run(tmp_path, _cells(S_GO, K_FLAT, pairs={ON[1]: 1999}))
    assert d["class"] == N4.INCONCLUSIVE and any("below the registered 2000 pairs" in r for r in d["reasons"])


def test_k_tolerates_one_uncounted_unit_per_arm_and_no_more(tmp_path):
    assert _run(tmp_path, _cells(S_GO, K_FLAT, lost_units={ON[0]: 1}))["class"] == N4.GO
    d = _run(tmp_path, _cells(S_GO, K_FLAT, lost_units={ON[0]: 1, ON[2]: 1}))
    assert d["class"] == N4.INCONCLUSIVE and any("2960 counted games < 2980" in r for r in d["reasons"])


def test_a_missing_final_is_inconclusive(tmp_path):
    cells = _cells(S_GO, K_FLAT)
    del cells["s"]["pool"]["nodes"][OFF[2]]
    d = _run(tmp_path, cells)
    assert d["class"] == N4.INCONCLUSIVE and any("no node" in r for r in d["reasons"])


# ------------------------------------------------------------------------------------------- the class boundaries
def _m(s_ci, k_ci, s_d: float = 0.5, k_d: float = 0.0) -> Dict[str, Any]:
    return {"s": {"delta": s_d, "ci": list(s_ci), "confidence": 0.9},
            "k": {"delta": k_d, "ci": list(k_ci), "phi_hat": 0.5}}


LR_OK = {"status": "OK", "ratio": 1.0, "confounded": False, "elevated": False}
# K bounds that decide nothing: LB < 0, UB > k_bar
K_U = (-0.02, 0.03)
# S bounds that decide nothing: LB < 0, UB > s_bar
S_U = (-1.0, 3.0)


@pytest.mark.parametrize("s_ci,k_ci,want", [
    # GO-S iff LB_S > 0: exactly 0 and half an eps from it are NOT past it (never AT eps: rule 8)
    ((0.0, 3.0), K_U, N4.UNDECIDED), ((5e-10, 3.0), K_U, N4.UNDECIDED), ((0.25, 3.0), K_U, N4.GO),
    # NO-GO-S iff UB_S < +2.0 pp
    ((-1.0, 2.0), K_U, N4.UNDECIDED), ((-1.0, 2.0 - 5e-10), K_U, N4.UNDECIDED), ((-1.0, 1.75), K_U, N4.NO_GO),
    # GO-K iff LB_K > 0
    (S_U, (0.0, 0.03), N4.UNDECIDED), (S_U, (5e-10, 0.03), N4.UNDECIDED), (S_U, (0.0025, 0.03), N4.GO),
    # NO-GO-K alone (UB_K < 0.015) is not NO-GO: NO-GO needs NO-GO-S
    (S_U, (-0.02, 0.0125), N4.UNDECIDED),
    # NO-GO needs not GO-K: NO-GO-S with GO-K reads GO (unfamiliar)
    ((-1.0, 1.75), (0.0025, 0.03), N4.GO),
    # MIXED: GO-S with UB_K < 0; UB_K exactly 0 is not below 0
    ((0.25, 3.0), (-0.03, -0.0025), N4.MIXED), ((0.25, 3.0), (-0.03, 0.0), N4.GO),
    # MIXED: GO-K with UB_S < 0; UB_S exactly 0 is not below 0
    ((-2.0, -0.25), (0.0025, 0.03), N4.MIXED), ((-2.0, 0.0), (0.0025, 0.03), N4.GO),
])
def test_every_class_boundary_in_the_registered_order(s_ci, k_ci, want):
    c = N4.classify(_m(s_ci, k_ci), _params(), LR_OK)
    assert c["class"] == want


def test_k_bar_is_compared_exactly_on_its_decimal():
    # UB_K = the float 0.015 is within eps of the decimal bar: not past it
    c = N4.classify(_m((-1.0, 3.0), (-0.02, 0.015)), _params(), LR_OK)
    assert not c["meters"]["k"]["no_go"] and c["tests"]["UB_K < k_bar"]["near"]
    c = N4.classify(_m((-1.0, 3.0), (-0.02, 0.0149)), _params(), LR_OK)
    assert c["meters"]["k"]["class"] == N4.NO_GO_K


def test_the_go_labels():
    assert N4.classify(_m((0.25, 3.0), (0.0025, 0.03)), _params(), LR_OK)["verdict"] == "GO (both)"
    assert N4.classify(_m((0.25, 3.0), K_U), _params(), LR_OK)["verdict"] == "GO (in-house)"
    assert N4.classify(_m(S_U, (0.0025, 0.03)), _params(), LR_OK)["verdict"] == "GO (unfamiliar)"


def test_mixed_is_never_go_and_without_the_veto_the_harm_case_reads_go():
    grid_s = [(-2.0, -0.25), (-1.0, 0.0), (-1.0, 1.75), (0.0, 3.0), (0.25, 1.75), (0.25, 3.0), (-1.0, 3.0)]
    grid_k = [(-0.03, -0.0025), (-0.03, 0.0), (-0.02, 0.0125), (0.0, 0.03), (0.0025, 0.0125), (0.0025, 0.03)]
    seen = set()
    for s_ci, k_ci in itertools.product(grid_s, grid_k):
        c = N4.classify(_m(s_ci, k_ci), _params(), LR_OK)
        seen.add(c["class"])
        harm = (s_ci[0] > 0 and k_ci[1] < 0) or (k_ci[0] > 0 and s_ci[1] < 0)
        assert (c["class"] == N4.MIXED) == harm
        if c["class"] == N4.MIXED:
            assert c["classed_as"] == N4.UNDECIDED and not c["verdict"].startswith("GO")
    assert {N4.MIXED, N4.GO, N4.NO_GO, N4.UNDECIDED} <= seen
    c = N4.classify(_m((0.25, 3.0), (-0.03, -0.0025)), _params(mixed_veto=False), LR_OK)
    assert c["verdict"] == "GO (in-house)"


def test_the_nogo_text_is_m5s_with_the_k_line():
    c = N4.classify(_m((-1.0, 1.75), (-0.02, 0.0125)), _params(), LR_OK)
    assert c["verdict"] == N4.NO_GO
    assert c["text"][0] == "no ≥ 2 pp in-house gain from per-kind FFN experts within 20M of P*"
    assert c["text"][1] == f"NO-GO-K: {N4.K_ONLY_LINE}"
    assert "capacity does not bind" not in " ".join(c["text"])
    c = N4.classify(_m((-1.0, 1.75), K_U), _params(), LR_OK)
    assert c["text"][1] == f"UNDECIDED-K: {N4.K_ONLY_LINE}"
    q = "with the experts' AdamW state copied from the shared FFN (Q-K3)"
    c = N4.classify(_m((-1.0, 1.75), K_U), _params(nogo_scope=dict(SCOPE, qualifier=q)), LR_OK)
    assert c["text"][0] == f"no ≥ 2 pp in-house gain from per-kind FFN experts within 20M of P*, {q}"


def test_near_scope_meter_reads_the_whole_meter_undecided():
    # LB_S within eps of 0 while UB_S is clearly below the bar: "bound" (X48) reads NO-GO, "meter" (N4) UNDECIDED
    m = _m((1e-10, 1.75), K_U)
    assert N4.classify(m, _params(), LR_OK)["class"] == N4.NO_GO
    assert N4.classify(m, _params(near_scope="meter"), LR_OK)["class"] == N4.UNDECIDED


# ------------------------------------------------------------------------------------------- M4: the LR label
def _lr(on: float, off: float) -> Dict[str, float]:
    return {**{lab: on for lab in ON}, **{lab: off for lab in OFF}}


def test_a_nogo_is_lr_confounded_below_0_9x_and_only_a_nogo(tmp_path):
    flat = _cells(S_FLAT, K_FLAT)
    assert _run(tmp_path, flat, lr=_lr(0.85e-4, 1e-4))["verdict"] == "NO-GO (LR-confounded)"
    assert _run(tmp_path, flat, lr=_lr(0.95e-4, 1e-4))["verdict"] == "NO-GO"
    d = _run(tmp_path, flat, lr=_lr(0.9e-4 - 1e-15, 1e-4))      # ratio 0.9 - 1e-11: within eps of 0.9, not below
    assert d["verdict"] == "NO-GO" and d["reported"]["lr"]["near"] and d["reported"]["lr"]["ratio"] < 0.9
    assert _run(tmp_path, _cells(S_GO, K_FLAT), lr=_lr(0.5e-4, 1e-4))["verdict"] == "GO (in-house)"


def test_a_go_is_lr_elevated_above_1_1x_and_only_a_go(tmp_path):
    go, flat = _cells(S_GO, K_FLAT), _cells(S_FLAT, K_FLAT)
    assert _run(tmp_path, go, lr=_lr(1.2e-4, 1e-4))["verdict"] == "GO (in-house; LR-elevated)"
    assert _run(tmp_path, go, lr=_lr(1.05e-4, 1e-4))["verdict"] == "GO (in-house)"
    d = _run(tmp_path, go, lr=_lr(1.1e-4 + 1e-15, 1e-4))         # ratio 1.1 + 1e-11: within eps of 1.1, not above
    assert d["verdict"] == "GO (in-house)" and d["reported"]["lr"]["near"] and d["reported"]["lr"]["ratio"] > 1.1
    assert _run(tmp_path, flat, lr=_lr(1.2e-4, 1e-4))["verdict"] == "NO-GO"


def test_an_unreadable_lr_prints_the_label_unavailable(tmp_path):
    lr = _lr(1e-4, 1e-4)
    del lr[OFF[0]]
    d = _run(tmp_path, _cells(S_FLAT, K_FLAT), lr=lr)
    assert d["verdict"] == "NO-GO (LR label UNAVAILABLE)" and d["reported"]["lr"]["status"] == "UNAVAILABLE"
    assert _run(tmp_path, _cells(S_GO, K_FLAT), lr=lr)["verdict"] == "GO (in-house; LR label UNAVAILABLE)"


def test_the_lr_is_the_mean_over_the_runs_snapshot_history(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    (run / "metadata.json").write_text(json.dumps({"snapshot_history": {"a": {"lr": 0.5}, "b": {"lr": 0.25}}}))
    got = N4.mean_lr(str(run))
    assert (got["mean_lr"], got["n_nodes"]) == (0.375, 2)
    assert N4.run_dir(SimpleNamespace(ref=SimpleNamespace(kind="final", run="models/rb_x"))) == "models/rb_x"
    ck = SimpleNamespace(ref=SimpleNamespace(kind="pinned", path="models/rb_x/checkpoints/c.zip"))
    assert N4.run_dir(ck) == "models/rb_x"


# ------------------------------------------------------------------------------------------- reported only
def test_the_slope_is_reported_and_an_incomplete_one_never_decides(tmp_path):
    sl_on, sl_off = [f"{lab}_10m" for lab in ON], [f"{lab}_10m" for lab in OFF]
    p = _params(slope={"group": "slope", "on": sl_on, "off": sl_off})
    assert RU.RULES["n4_screen"].reported(p) == [("s", "slope")]
    spec = _spec(tmp_path, _lr(1e-4, 1e-4))
    for lab, src in zip(sl_on + sl_off, ON + OFF):
        spec.models[lab] = copy.copy(spec.models[src])
    cells = _cells(S_GO, K_FLAT)
    vals = dict(zip(sl_on + sl_off, [57.0, 57.5, 58.0, 56.5, 56.75, 57.75]))     # rises 1, 1, 1 / 0, .25, -.25
    cells["s"]["slope"] = {"engine": ENGINE_S, "regime_id": "c" * 16, "inconclusive": [],
                           "nodes": {lab: _node(v, 500) for lab, v in vals.items()}}
    d = RU.decide("n4_screen", RU.inputs("n4_screen", cells, p, spec), p)
    sl = d["reported"]["slope"]
    assert sl["status"] == "OK" and sl["per_final"][sl_on[0]]["rise_pp"] == 1.0
    assert sl["delta_slope"]["delta"] == pytest.approx(1.0) and sl["delta_slope"]["confidence"] == 0.90
    cells["s"]["slope"]["nodes"][sl_on[0]]["n_pairs"]["m1"] = 499
    d2 = RU.decide("n4_screen", RU.inputs("n4_screen", cells, p, spec), p)
    assert d2["reported"]["slope"]["status"] == "INCOMPLETE" and d2["verdict"] == d["verdict"]


def test_the_reported_lines_carry_no_decision(tmp_path):
    p = _params(pin_drift={"s_pp": 56.98, "source": "x35_g1/results.json (E1, 500 pairs)"})
    d = _run(tmp_path, _cells(S_FLAT, K_FLAT), p)
    rep = d["reported"]
    assert rep["pin_drift"]["difference_pp"] == pytest.approx(57.0 - 56.98)
    assert set(rep["per_member_delta_s"]) == {"m1", "x26_pstar"}
    pw = rep["s_power_at_observed_sigma"]
    assert 0 <= pw["at_+0pp"]["go_s"] <= 1 and pw == _run(tmp_path, _cells(S_FLAT, K_FLAT), p)["reported"][
        "s_power_at_observed_sigma"]                                  # seeded: a re-decision prints the same numbers
    assert rep["engines"] == {"s": ENGINE_S, "k": ENGINE_K} and set(d["not_reported"]) >= {"rho", "approx_kl"}


# ------------------------------------------------------------------------------------------- the spec
def _spec_doc() -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    for lab in ON:
        models[lab] = {"arm": "x48", "seed": SEED[lab], "ref": {"kind": "final", "run": f"models/rb_x35_x48kffn_s"
                                                                f"{SEED[lab]}", "min_step": 300000170, "quiet_s": 300}}
    for lab in OFF:
        models[lab] = {"arm": "c0", "seed": SEED[lab], "ref": {"kind": "pinned", "sha256": "0" * 64,
                                                               "path": f"models/rb_x35_c0_s{SEED[lab]}/final_model.zip"}}
    return {"schema": "gen3_read_spec_v1", "read": "n4_screen_test_2026-10-27", "models": models,
            "anchor": {"unfamiliar": {"opponent": "metamon:Kakuna", "teamsets": ["forum72"], "unit_seeds": "0-49",
                                      "models": ON + OFF}},
            "s": {"pool": {"refpop": "r.json", "refpop_content_sha256": "1" * 64, "panel": None, "pairs": 2000,
                           "batch_pairs": 100, "family": "x35_x48", "models": ON + OFF}},
            "decision": {"rule": "n4_screen", "params": _params()}}


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d["decision"]["params"].pop("void"), "void"),
    (lambda d: d["decision"]["params"].pop("mixed_veto"), "mixed_veto"),
    (lambda d: d["decision"]["params"].update(mixed_veto=1), "mixed_veto"),
    (lambda d: d["decision"]["params"].update(off=ON), "both sides"),
    (lambda d: d["s"]["pool"].update(pairs=400), "< 500"),
    (lambda d: d["s"]["pool"].update(models=ON + OFF[:2]), "does not play"),
    (lambda d: d["decision"]["params"].update(near_scope="any"), "near_scope"),
    (lambda d: d["decision"]["params"]["nogo_scope"].pop("dose"), "nogo_scope"),
    (lambda d: d["decision"]["params"]["nogo_scope"].update(why="x"), "nogo_scope"),
    (lambda d: d["decision"]["params"].update(slope={"group": "pool", "on": ON, "off": OFF}), "slope.group"),
])
def test_the_spec_load_check_refuses_a_malformed_registration(mutate, match):
    from main.reads.spec import SpecError, parse

    parse(_spec_doc())
    doc = _spec_doc()
    mutate(doc)
    with pytest.raises(SpecError, match=match):
        parse(doc)


X48 = Path("designs/research_state/measurements/x48_kffn_production_2026-10-27")


def test_x48s_draft_read_spec_parses_with_the_rule_registered_as_built():
    from main.reads.spec import load
    from utils.paths import repo_path

    spec = load(repo_path(str(X48 / "read_spec.draft.json")))
    p = spec.params
    assert spec.rule == "n4_screen" and spec.pin is None and spec.s["pool"].pairs == 2000
    assert (p["s_bar"], p["k_bar"], p["confidence"], p["z"], p["rule8_eps"], p["mixed_veto"]) == \
        (2.0, 0.015, 0.90, 1.6449, 1e-9, True)
    assert spec.anchor["unfamiliar"].teamsets == ("forum72",) and len(spec.anchor["unfamiliar"].unit_seeds) == 50
    assert p["nogo_scope"] == dict(SCOPE, qualifier="with the experts' AdamW state copied from the shared FFN (Q-K3)")
    assert p.get("near_scope", "bound") == "bound"
    assert (p["void"]["pin"], p["void"]["pool_seed"]) == ("f4b25238def57900c0305bb2a3a5a8f9edaef285",
                                                          {"fork_step": 280000170})
    assert set(p["void"]["runs"]) == set(p["on"])
    shas = {spec.models[lab].ref.sha256 for lab in p["off"]}
    assert shas == {"09a6a2df48509a59fd9aefe6d20d9722aad6717f1730c6dfd75274446b283303",
                    "7e9a9890e2b397de6d8b87588c597d7f53e39b84efd3d1080305f77dbf22feec",
                    "99096be8eccfc1cef4268ab4d6de15063a386cda475e72340212fdbf3a1b7e26"}

