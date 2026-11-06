"""``main.reads`` rule ``x60`` (X60-T PREREG §5-§11, §15): the ported interval against ``power.py`` itself (fixed and
simulated inputs), §11's verdict over every class combination, the pooled groups' one-cell checks, the determinism
guard, the spec-load check, the ``dose`` VOID guard and the look-1 DRAFT spec. Synthetic cells, no game."""
from __future__ import annotations

import importlib.util
import itertools
import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from main.reads import rules as RU
from main.reads import rules_x60 as X
from main.x35 import d1 as D
from utils.paths import repo_path

POWER = repo_path("designs/research_state/measurements/x60_redesign_2026-10-26/power.py")
ENGINE = "port=0d3d7629f16a0eaf/91;env=393631b886965b0c/126"
SIDES = {"peer_code": "b817d3216c418f75", "policy_mode": "eval"}


def _power() -> Any:
    spec = importlib.util.spec_from_file_location("x60_power_frozen", POWER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------ power.py, verbatim
def test_the_constants_are_power_pys():
    P = _power()
    assert (X.Z, X.SIGMA_F, X.BAND, X.TIE) == (P.Z, P.SIGMA_F, P.BAND, P.TIE)


def _as_counts(arr: np.ndarray, n: Sequence[int]) -> List[List[Tuple[int, int]]]:
    return [[(int(round(arr[s, k] * n[k])), int(n[k])) for k in range(arr.shape[1])] for s in range(arr.shape[0])]


def _near_boundary(lo: float, hi: float, d1: float, guard_lb: float) -> bool:
    """Rule 8: a draw whose deciding statistic sits within 1e-12 of a boundary is excluded (the two codes round
    differently there), deterministically."""
    eps = 1e-12
    return any(abs(x - b) < eps for x, b in ((lo, X.TIE), (hi, -X.TIE), (hi, X.BAND - X.TIE), (lo, -X.BAND + X.TIE),
                                             (d1, X.TIE), (guard_lb, X.SIGMA_F ** 2)))


FIXED_ON = [0.5075, 0.5208, 0.4983, 0.5083, 0.4858, 0.5158, 0.5183, 0.5133]
FIXED_OFF = [0.5058, 0.5208, 0.4983, 0.5083, 0.4858, 0.5158, 0.5183, 0.4808]


@pytest.mark.parametrize("shift", [-0.05, -0.01, 0.0, 0.012, 0.03, 0.06])
def test_the_rule_reproduces_power_py_on_fixed_inputs(shift):
    P = _power()
    n = 1200
    on = np.round((np.array(FIXED_ON) + shift) * n) / n
    off = np.round(np.array(FIXED_OFF) * n) / n
    a, c = on[None, :, None], off[None, :, None]
    fin, stop, h2, g2, eq = P.rule_x60(a, c, [n])
    d, hw, g = P.x60_interval(a, c, [n])
    got = X.rule_x60(_as_counts(a[0], [n]), _as_counts(c[0], [n]), [0, 1, 2])
    assert got["class"] == fin[0] and got["stop"] == bool(stop[0]) and got["equivalent"] == bool(eq[0])
    s2 = X.x60_interval(_as_counts(a[0], [n]), _as_counts(c[0], [n]))
    assert s2["delta"] == pytest.approx(float(d[0]), abs=1e-14)
    assert s2["halfwidth"] == pytest.approx(float(hw[0]), abs=1e-14) and s2["welch_used"] == bool(g[0])
    assert s2["halfwidth"] == pytest.approx(float(h2[0]), abs=1e-14)


def test_the_welch_guard_branch_reproduces_power_py():
    """A spread wide enough that σ̂²'s one-sided 95 % lower bound exceeds the floor: the Welch t interval decides."""
    P = _power()
    n = 1200
    on = np.round(np.array([0.44, 0.58, 0.47, 0.55, 0.42, 0.60, 0.50, 0.53]) * n) / n
    off = np.round(np.array([0.41, 0.57, 0.45, 0.52, 0.43, 0.56, 0.48, 0.49]) * n) / n
    d, hw, g = P.x60_interval(on[None, :, None], off[None, :, None], [n])
    s = X.x60_interval(_as_counts(on[:, None], [n]), _as_counts(off[:, None], [n]))
    assert bool(g[0]) and s["welch_used"]
    assert s["halfwidth"] == pytest.approx(float(hw[0]), abs=1e-13) and s["delta"] == pytest.approx(float(d[0]), abs=1e-14)


@pytest.mark.parametrize("sigma,delta", [(0.0, 0.0), (0.01, 0.0), (0.01, 0.03), (0.02, 0.02), (0.03, 0.0),
                                         (0.03, 0.04)])
def test_the_rule_reproduces_power_py_on_simulated_draws(sigma, delta):
    """power.py's own ``finals`` (8 seeds, F alone, 1,200 a final) through BOTH rules: every class, stop, half-width
    and guard equal, draw for draw."""
    P = _power()
    P.REPS = 300
    rng = np.random.default_rng(20261106)
    a = P.finals(rng, ("F",), [1200], [delta], sigma, seeds=8)
    c = P.finals(rng, ("F",), [1200], [0.0], sigma, seeds=8)
    fin, stop, h2, g2, eq = P.rule_x60(a, c, [1200])
    d1, h1, _ = P.x60_interval(a[:, :3], c[:, :3], [1200])
    d2, _, _ = P.x60_interval(a, c, [1200])
    compared = 0
    for r in range(P.REPS):
        lo, hi = (d1[r] - h1[r], d1[r] + h1[r]) if stop[r] else (d2[r] - h2[r], d2[r] + h2[r])
        w_on, w_off = a[r].mean(1), c[r].mean(1)
        w = (w_on.var(ddof=1) + w_off.var(ddof=1)) / 2
        b = ((a[r] * (1 - a[r]) / 1200).sum(1).mean() + (c[r] * (1 - c[r]) / 1200).sum(1).mean()) / 2
        from scipy.stats import chi2
        if _near_boundary(lo, hi, d1[r], w * 14 / chi2.ppf(0.95, 14) - b):
            continue
        got = X.rule_x60(_as_counts(a[r], [1200]), _as_counts(c[r], [1200]), [0, 1, 2])
        assert (got["class"], got["stop"], got["equivalent"]) == (fin[r], bool(stop[r]), bool(eq[r])), r
        if not stop[r]:
            assert got["look2"]["halfwidth"] == pytest.approx(float(h2[r]), abs=1e-12)
            assert got["look2"]["welch_used"] == bool(g2[r])
        compared += 1
    assert compared >= 295


def test_the_contrast_reproduces_power_pys_two_panel_two_sided_read():
    P = _power()
    P.REPS = 200
    rng = np.random.default_rng(7)
    a = P.finals(rng, ("H", "F"), [600, 600], [0.03, 0.03], 0.01)
    c = P.finals(rng, ("H", "F"), [600, 600], [0.0, 0.0], 0.01)
    d, hw, g = P.x60_interval(a, c, [600, 600])
    cls = P.classes(d, hw, one_sided=False)
    for r in range(P.REPS):
        s = X.x60_interval(_as_counts(a[r], [600, 600]), _as_counts(c[r], [600, 600]))
        assert s["halfwidth"] == pytest.approx(float(hw[r]), abs=1e-12) and s["welch_used"] == bool(g[r])
        assert X.classes(s["lo"], s["hi"], one_sided=False) == cls[r]


def test_a_short_final_reads_each_arms_pooled_rate():
    """The coded reading of "ON's pooled F rate" inside the floor: Σwins / Σgames, each cell's own games."""
    on = [[(600, 1200)], [(590, 1180)], [(610, 1200)]]
    off = [[(580, 1200)], [(600, 1200)], [(590, 1200)]]
    s = X.x60_interval(on, off)
    assert s["delta"] == pytest.approx(1800 / 3580 - 1770 / 3600, abs=1e-15)
    with pytest.raises(ValueError):
        X.x60_interval(on, off[:2])


# ------------------------------------------------------------------------------------------------ §11's verdict
NI, NE = D.NON_INFERIOR, D.NOT_EXCLUDED
I_F_STATES = [None, {"gain": False, "lb90_pos": False}, {"gain": False, "lb90_pos": True},
              {"gain": True, "lb90_pos": True}]


def _row(look: int, cls: str, stop: bool, i_f: Dict[str, bool], r72: bool, p: str, ga: bool,
         guards: Dict[str, str]) -> str:
    """§11's table written as its rows' predicates, every input known; exactly one row may match."""
    rows = []
    l2 = look == 2 and not stop
    if cls == X.LOSS:
        rows.append(X.LOSS)
    if look == 1 and not stop:
        rows.append(X.CONTINUE)
    if stop and cls == X.NUG:
        rows.append(X.NUG_LOOK1)
    if stop and cls == X.NOT_RESOLVED:
        rows.append(X.NOT_RESOLVED)
    if l2 and cls == X.GAIN and i_f["lb90_pos"] and p == NI and ga and all(v == X.PASS for v in guards.values()):
        rows.append(X.ADOPT)
    if l2 and cls == X.GAIN and i_f["lb90_pos"] and (p != NI or not ga or any(v == X.FAIL for v in guards.values())):
        rows.append(X.TRADE)
    if l2 and cls == X.GAIN and not i_f["lb90_pos"]:
        rows.append(X.KSPEC)
    if l2 and cls == X.NUG and r72 and not i_f["gain"]:
        rows.append(X.NO_TRANSFER)
    if l2 and cls == X.NUG and r72 and i_f["gain"]:
        rows.append(X.IN_HOUSE_ONLY)
    if l2 and cls == X.NUG and not r72:
        rows.append(X.NOT_DELIVERED)
    if l2 and cls == X.NOT_RESOLVED:
        rows.append(X.NOT_RESOLVED)
    assert len(rows) == 1, (look, cls, stop, rows)
    return rows[0]


def _completions(v: Any, domain: Sequence[Any]) -> Sequence[Any]:
    return domain if v is None else [v]


def test_the_verdict_over_every_class_combination():
    """For every combination (each meter known or not): the verdict is the one §11 row every completion of the
    unknowns agrees on, and INCONCLUSIVE exactly when the completions disagree."""
    gdom = [X.PASS, X.FAIL, "INCONCLUSIVE"]
    n = 0
    for look, stop, cls in itertools.chain(
            ((1, True, c) for c in (X.LOSS, X.NUG, X.NOT_RESOLVED)), ((1, False, X.NOT_RESOLVED),),
            ((2, True, c) for c in (X.LOSS, X.NUG, X.NOT_RESOLVED)),
            ((2, False, c) for c in (X.GAIN, X.LOSS, X.NUG, X.NOT_RESOLVED))):
        for i_f, r72, p, ga, gf, ge, gc in itertools.product(I_F_STATES, [None, True, False], [None, NI, NE],
                                                             [None, True, False], gdom, gdom, gdom):
            guards = {"G-F": gf, "G-E": ge, "G-C": gc}
            want = {_row(look, cls, stop, i, r, pp, g, dict(zip(X.GUARDS, gs)))
                    for i in _completions(i_f, I_F_STATES[1:]) for r in _completions(r72, [True, False])
                    for pp in _completions(p, [NI, NE]) for g in _completions(ga, [True, False])
                    for gs in itertools.product(*[[x] if x != "INCONCLUSIVE" else [X.PASS, X.FAIL]
                                                  for x in (gf, ge, gc)])}
            got, why = X.verdict(look, {"class": cls, "stop": stop}, i_f=i_f, r72=r72, p=p, goal_a=ga, guards=guards)
            assert got == (want.pop() if len(want) == 1 else X.INCONCLUSIVE), (look, stop, cls, i_f, r72, p, ga, guards)
            assert why
            n += 1
    assert n == 4 * 3 * 3 * 3 * 27 * 11
    assert X.verdict(2, None)[0] == X.INCONCLUSIVE


# ------------------------------------------------------------------------------------------------ synthetic cells
def _agroup(teamsets: Sequence[str], counts: Dict[str, Dict[str, Tuple[int, int]]], *, engine: Optional[str] = ENGINE,
            regime: str = "r" * 16, sides: Optional[Dict[str, str]] = None, pack: str = "legacy",
            imported_from: Optional[str] = None, inconclusive: Sequence[str] = (), unplayed: int = 0) -> Dict[str, Any]:
    cells: Dict[str, Any] = {}
    units: List[Dict[str, Any]] = []
    for lab, by in counts.items():
        cells[lab] = {}
        for ts, (w, n) in by.items():
            k = n // 20
            cells[lab][ts] = {"wins": w, "ties": 0, "n": n, "units": k, "planned": k, "skipped": []}
            for u in range(k):
                wu = w // k + (u < w % k)
                games = [["a" if j < 10 else "b", j, f"t{j}", "win" if j < wu else "loss", 30] for j in range(20)]
                units.append({"unit": f"{lab}__{ts}__{u}", "label": lab, "panel": ts, "unit_seed": u, "counted": True,
                              "n": 20, "wins": wu, "games": games, "peer_pack": pack, "cell_dir": None,
                              **({"imported_from": imported_from} if imported_from else {})})
    return {"teamsets": list(teamsets), "engine": engine, "regime_id": "g" * 16,
            "regime_ids": {ts: regime for ts in teamsets}, **(sides or SIDES), "cells": cells, "units": units,
            "unplayed": unplayed, "inconclusive": list(inconclusive)}


def _sgroup(panel: str, nodes: Dict[str, float], *, engine: str = "env=393631b886965b0c/126",
            regime: str = "d4de0fd7c16a2658") -> Dict[str, Any]:
    return {"engine": engine, "regime_id": regime, "panel": panel, "refpop_content_sha256": "c" * 64,
            "nodes": {lab: {"s_pp": v, "se_pp": 0.3} for lab, v in nodes.items()}, "inconclusive": []}


ON8 = [f"on_s{s}" for s in range(3501, 3509)]
OFF8 = [f"c0_s{s}" for s in range(3501, 3509)]
D1 = [f"d1_s{s}" for s in range(3501, 3506)]
SEED = {**{lab: 3501 + i for i, lab in enumerate(ON8)}, **{lab: 3501 + i for i, lab in enumerate(OFF8)},
        **{lab: 3501 + i for i, lab in enumerate(D1)}}


def _spec(groups: Dict[str, Tuple[Sequence[str], Sequence[str], int, bool]], s: Dict[str, Tuple[str, Sequence[str]]]
          ) -> SimpleNamespace:
    arm = {**{lab: "x60t" for lab in ON8}, **{lab: "c0" for lab in OFF8}, **{lab: "d1" for lab in D1}}
    models = {lab: SimpleNamespace(arm=arm[lab], seed=SEED[lab], load={}) for lab in arm}
    anchor = {g: SimpleNamespace(name=g, teamsets=tuple(ts), models=tuple(ms), unit_seeds=tuple(range(units)),
                                 per_game=False, imports=SimpleNamespace(models=()) if imp else None)
              for g, (ts, ms, units, imp) in groups.items()}
    sg = {g: SimpleNamespace(name=g, panel=panel, models=tuple(ms)) for g, (panel, ms) in s.items()}
    return SimpleNamespace(models=models, anchor=anchor, s=sg)


F_ON = [612, 640, 598, 650, 630, 622, 645, 618]          # ≈ +3 to +5 pp over C0
F_OFF = [607, 625, 598, 610, 583, 619, 622, 577]


def _look2(*, p_on: float = 57.0, i_on: float = 52.0, r_on: float = 51.0, f_on: Sequence[int] = F_ON,
           home_wins: int = 70, off_engine: str = ENGINE, d1_regime: str = "r" * 16) -> Tuple[Dict, Dict, Any]:
    g1 = {**{lab: {"forum72": (w, 1200)} for lab, w in zip(ON8, f_on)},
          **{lab: {"forum72": (w, 1200)} for lab, w in zip(OFF8[5:], F_OFF[5:])}}
    cells = {"anchor": {
        "primary": _agroup(["forum72"], g1),
        "c0_look1": _agroup(["forum72"], {lab: {"forum72": (w, 1200)} for lab, w in zip(OFF8[:3], F_OFF[:3])},
                            imported_from="/x61p/look1", engine=off_engine),
        "c0_look2": _agroup(["forum72"], {lab: {"forum72": (w, 1200)} for lab, w in zip(OFF8[3:5], F_OFF[3:5])},
                            imported_from="/x61p/look2"),
        "home": _agroup(["home72"], {lab: {"home72": (home_wins, 140)} for lab in ON8}),
        "contrast_h": _agroup(["heldout72_hpfix"], {lab: {"heldout72_hpfix": (320, 600)} for lab in ON8[:5]}),
        "d1": _agroup(["heldout72_hpfix", "forum72"], {lab: {"heldout72_hpfix": (290, 600), "forum72": (305, 600)}
                                                     for lab in D1}, regime=d1_regime)},
        "s": {"p": _sgroup("home72", {**{lab: p_on + 0.1 * i for i, lab in enumerate(ON8)},
                                      **{lab: 57.0 + 0.1 * (i % 3) for i, lab in enumerate(OFF8[5:])}}),
              "p_c0_look1": _sgroup("home72", {lab: 56.8 + 0.2 * i for i, lab in enumerate(OFF8[:3])}),
              "p_c0_look2": _sgroup("home72", {lab: 57.2 - 0.3 * i for i, lab in enumerate(OFF8[3:5])}),
              "i_f": _sgroup("forum72", {**{lab: i_on + 0.2 * (i % 3) for i, lab in enumerate(ON8)},
                                         **{lab: 50.0 + 0.2 * (i % 3) for i, lab in enumerate(OFF8)}}),
              "r72": _sgroup("r72", {**{lab: r_on + 0.2 * (i % 3) for i, lab in enumerate(ON8)},
                                     **{lab: 50.0 + 0.2 * (i % 3) for i, lab in enumerate(OFF8)}})},
        "guards": {"rederived": "count", "choices": "require"}}
    p = {"look": 2, "on": ON8, "off": OFF8, "look1_seeds": [3501, 3502, 3503],
         "primary": {"groups": ["primary", "c0_look1", "c0_look2"], "teamset": "forum72", "units": 60},
         "p": {"groups": ["p", "p_c0_look1", "p_c0_look2"], "panel": "home72"},
         "i_f": {"groups": ["i_f"], "panel": "forum72"}, "r72": {"groups": ["r72"], "panel": "r72"},
         "goal_a": {"groups": ["home"], "teamset": "home72", "units": 7},
         "contrast": {"on": ON8[:5], "off": D1, "groups": ["primary", "contrast_h", "d1"],
                      "panels": ["heldout72_hpfix", "forum72"], "units": 30}}
    spec = _spec({"primary": (["forum72"], ON8 + OFF8[5:], 60, False),
                  "c0_look1": (["forum72"], OFF8[:3], 60, True), "c0_look2": (["forum72"], OFF8[3:5], 60, True),
                  "home": (["home72"], ON8, 7, False), "contrast_h": (["heldout72_hpfix"], ON8[:5], 30, False),
                  "d1": (["heldout72_hpfix", "forum72"], D1, 30, False)},
                 {"p": ("home72", ON8 + OFF8[5:]), "p_c0_look1": ("home72", OFF8[:3]),
                  "p_c0_look2": ("home72", OFF8[3:5]), "i_f": ("forum72", ON8 + OFF8), "r72": ("r72", ON8 + OFF8)})
    return cells, p, spec


def _decide(cells: Dict, p: Dict, spec: Any) -> Dict[str, Any]:
    assert RU.RULES["x60"].check_params(p, models=spec.models, anchor=spec.anchor, s=spec.s) == []
    doc = json.loads(json.dumps(RU.inputs("x60", cells, p, spec)))         # the decision reads the FILE's form
    dec = RU.decide("x60", doc, p)
    assert RU.RULES["x60"].verdict_of(dec) == dec["decision"]["verdict"]
    return dec


def test_look2_pools_c0_from_two_states_and_equals_rule_x60_on_the_same_counts(monkeypatch):
    cells, p, spec = _look2()
    from main.x35 import readout as R

    monkeypatch.setattr(R, "guard_c_record", lambda rec, on, off: R.GuardResult("G-C", R.PASS))
    dec = _decide(cells, p, spec)
    want = X.rule_x60([[(w, 1200)] for w in F_ON], [[(w, 1200)] for w in F_OFF], [0, 1, 2])
    assert dec["primary"]["class"] == want["class"] == X.GAIN
    assert dec["primary"]["look2"]["delta"] == pytest.approx(want["look2"]["delta"], abs=1e-15)
    assert dec["p"]["class"] == NI and dec["i_f"]["gain"] and dec["r72"]["gain"] and dec["goal_a"]["pass"] is False
    assert dec["decision"]["verdict"] == X.TRADE and "goal (a) fails" in dec["decision"]["reasons"][0]
    assert dec["reported"]["contrast_t_minus_d1"]["status"] == "READ"
    assert dec["decision"]["contrast_class"] in (X.LADDER, X.FORUM, X.NO_DIFF, X.NOT_RESOLVED)


def test_a_gain_with_every_meter_passing_is_adopt_only_with_the_guards_read(tmp_path, monkeypatch):
    cells, p, spec = _look2(home_wins=95)
    dec = _decide(cells, p, spec)
    assert dec["decision"]["verdict"] == X.INCONCLUSIVE                      # no guards artefact: not read
    assert all(g["status"] == "INCONCLUSIVE" for g in dec["guards"].values())
    rng = np.random.default_rng(3)
    art = tmp_path / "guards.json"
    art.write_text(json.dumps({"g_f": {"h_pp": (50 + rng.normal(0, 1, (8, 8))).tolist()},
                               "r2": {"on": [30, 200], "off": [31, 200]}, "g_c": {"schema": "not a record"}}))
    from utils.hashing import sha256_file

    p["guards"] = {"path": str(art), "sha256": sha256_file(art)}
    dec = _decide(cells, p, spec)
    assert dec["guards"]["G-F"]["status"] == "PASS" and dec["guards"]["G-E"]["status"] == "PASS"
    assert dec["guards"]["G-C"]["status"] == "INCONCLUSIVE" and dec["decision"]["verdict"] == X.INCONCLUSIVE
    from main.x35 import readout as R

    monkeypatch.setattr(R, "guard_c_record", lambda rec, on, off: R.GuardResult("G-C", R.PASS))
    assert _decide(cells, p, spec)["decision"]["verdict"] == X.ADOPT
    p["guards"]["sha256"] = "0" * 64
    assert "sha256" in _decide(cells, p, spec)["guards"]["G-F"]["reasons"][0]


def test_kakuna_specific_trade_on_p_and_the_nug_branches():
    cells, p, spec = _look2(i_on=50.0)
    assert _decide(cells, p, spec)["decision"]["verdict"] == X.KSPEC
    cells, p, spec = _look2(p_on=52.0)
    dec = _decide(cells, p, spec)
    assert dec["decision"]["verdict"] == X.TRADE and dec["decision"]["exchange_rate"]["rate"] > 0
    flat = [612, 630, 600, 605, 580, 615, 618, 575]          # look 1 Δ̂ > 0 (continues); 8 v 8 Δ̂ ≈ 0
    cells, p, spec = _look2(f_on=flat)
    dec = _decide(cells, p, spec)
    assert dec["primary"]["class"] == X.NUG and dec["decision"]["verdict"] == X.IN_HOUSE_ONLY
    assert dec["decision"]["guards_not_read"].startswith("Δ_F is NO-USEFUL-GAIN")
    cells, p, spec = _look2(f_on=flat, i_on=50.0)
    assert _decide(cells, p, spec)["decision"]["verdict"] == X.NO_TRANSFER
    cells, p, spec = _look2(f_on=flat, r_on=50.0)
    assert _decide(cells, p, spec)["decision"]["verdict"] == X.NOT_DELIVERED


@pytest.mark.parametrize("break_", ["engine", "regime", "sides", "pack", "twice", "unplayed", "floor"])
def test_pooled_groups_must_be_one_cell(break_):
    cells, p, spec = _look2()
    g = cells["anchor"]["c0_look2"]
    if break_ == "engine":
        g["engine"] = "port=aaaa/1;env=bbbb/2"
    elif break_ == "regime":
        g["regime_ids"]["forum72"] = "x" * 16
    elif break_ == "sides":
        g["policy_mode"] = "train"
    elif break_ == "pack":
        for u in g["units"]:
            u["peer_pack"] = "changed"
    elif break_ == "twice":
        g["cells"][OFF8[0]] = cells["anchor"]["c0_look1"]["cells"][OFF8[0]]
    elif break_ == "unplayed":
        g["unplayed"] = 3
    elif break_ == "floor":
        g["cells"][OFF8[3]]["forum72"].update(n=1100, wins=560)
        cells["anchor"]["c0_look1"]["cells"][OFF8[0]]["forum72"].update(n=1180, wins=600)
    dec = _decide(cells, p, spec)
    assert dec["decision"]["verdict"] == X.INCONCLUSIVE and dec["primary"]["class"] is None


def test_one_short_unit_per_arm_stays_inside_the_floor():
    cells, p, spec = _look2()
    cells["anchor"]["c0_look1"]["cells"][OFF8[0]]["forum72"].update(n=1180, wins=597)
    assert _decide(cells, p, spec)["primary"]["class"] == X.GAIN


def test_the_contrast_is_not_read_when_short_and_never_blocks():
    cells, p, spec = _look2(d1_regime="z" * 16)
    dec = _decide(cells, p, spec)
    assert dec["reported"]["contrast_t_minus_d1"]["status"] == "NOT READ"
    assert dec["decision"]["contrast_class"] == "NOT READ"
    assert ("anchor", "d1") in X.reported(p) and ("anchor", "contrast_h") in X.reported(p)
    assert ("anchor", "primary") not in X.reported(p)


def _look1(f_on: Sequence[int]) -> Tuple[Dict, Dict, Any]:
    cells = {"anchor": {"primary": _agroup(["forum72"], {
        **{lab: {"forum72": (w, 1200)} for lab, w in zip(ON8[:3], f_on)},
        **{lab: {"forum72": (w, 1200)} for lab, w in zip(OFF8[:3], F_OFF[:3])}})}, "s": {},
        "guards": {"choices": "require"}}
    p = {"look": 1, "on": ON8[:3], "off": OFF8[:3], "look1_seeds": [3501, 3502, 3503],
         "primary": {"groups": ["primary"], "teamset": "forum72", "units": 60}}
    spec = _spec({"primary": (["forum72"], ON8[:3] + OFF8[:3], 60, True)}, {})
    return cells, p, spec


def test_look1_is_futility_only():
    dec = _decide(*_look1([640, 650, 630]))
    assert dec["decision"]["verdict"] == X.CONTINUE and not dec["primary"]["stop"]
    dec = _decide(*_look1([600, 620, 590]))
    assert dec["primary"]["stop"] and dec["decision"]["verdict"] == X.NUG_LOOK1
    dec = _decide(*_look1([560, 570, 550]))
    assert dec["decision"]["verdict"] == X.LOSS
    tie = _decide(*_look1(F_OFF[:3]))                       # Δ̂ = 0 exactly: within TIE counts as <= 0 (stop)
    assert tie["primary"]["stop"] and tie["primary"]["look1"]["delta"] == 0.0


def test_look2_stops_where_look1_would_have():
    f_on = [600, 615, 590, 700, 700, 700, 700, 700]
    cells, p, spec = _look2(f_on=f_on)
    dec = _decide(cells, p, spec)
    assert dec["primary"]["stop"] and dec["decision"]["verdict"] in (X.NUG_LOOK1, X.NOT_RESOLVED, X.LOSS)
    assert "should have stopped at look 1" in dec["decision"]["reasons"][-1]


def test_the_determinism_guard_compares_the_replayed_unit_with_the_imported_one():
    cells, p, spec = _look1([640, 650, 630])
    imp = _agroup(["forum72"], {OFF8[0]: {"forum72": (607, 1200)}}, imported_from="/x61p/look1")
    cells["anchor"]["primary"]["units"] += imp["units"]
    guard = _agroup(["forum72"], {OFF8[0]: {"forum72": (11, 20)}})
    guard["units"][0]["games"] = [list(g) for g in imp["units"][0]["games"]]
    cells["anchor"]["determinism"] = guard
    cells["guards"]["choices"] = "five_tuple"               # synthetic rows carry no capture
    spec.anchor["determinism"] = SimpleNamespace(name="determinism", teamsets=("forum72",), models=(OFF8[0],),
                                                 unit_seeds=(0,), per_game=False, imports=None)
    p["determinism_guard"] = {"group": "determinism", "against": "primary"}
    dec = _decide(cells, p, spec)
    assert dec["determinism_guard"]["passed"] and dec["decision"]["verdict"] == X.CONTINUE
    guard["units"][0]["games"][3][3] = "win" if guard["units"][0]["games"][3][3] == "loss" else "loss"
    dec = _decide(cells, p, spec)
    assert not dec["determinism_guard"]["passed"] and dec["decision"]["verdict"] == X.INCONCLUSIVE
    assert "X7" in dec["decision"]["reasons"][0]
    cells["guards"]["choices"] = "require"
    guard["units"][0]["games"] = [list(g) for g in imp["units"][0]["games"]]
    assert not _decide(cells, p, spec)["determinism_guard"]["passed"]      # a missing capture fails closed


# ------------------------------------------------------------------------------------------------ the spec-load check
@pytest.mark.parametrize("mutate,needle", [
    (lambda p: p.update(on=p["on"][:5], off=p["off"][:5]), "registers 8 v 8"),
    (lambda p: p["primary"].update(units=7), "registered forum72 / 60"),
    (lambda p: p["primary"].update(groups=["primary", "c0_look1"]), "in 0 of its groups"),
    (lambda p: p["p"].update(panel="forum72"), "registered home72"),
    (lambda p: p.pop("r72"), "r72: required at look 2"),
    (lambda p: p["contrast"].update(on=ON8[:4]), "always 5 v 5"),
    (lambda p: p["contrast"].update(units=7), "registered"),
    (lambda p: p.update(look1_seeds=[3501, 3502, 3509]), "look1_seeds"),
    (lambda p: p.update(guards={"path": "x"}), "guards:"),
    (lambda p: p.update(extra=1), "unknown 'extra'"),
])
def test_the_spec_load_check_refuses(mutate, needle):
    _, p, spec = _look2()
    mutate(p)
    probs = RU.RULES["x60"].check_params(p, models=spec.models, anchor=spec.anchor, s=spec.s)
    assert any(needle in x for x in probs), probs


def test_look1_refuses_the_look2_meters_and_the_contrast():
    _, p, spec = _look1([640, 650, 630])
    p["goal_a"] = {"groups": ["primary"], "teamset": "home72", "units": 7}
    p["contrast"] = {"on": ON8[:3], "off": D1, "groups": ["primary"], "panels": ["forum72"], "units": 30}
    probs = RU.RULES["x60"].check_params(p, models=spec.models, anchor=spec.anchor, s=spec.s)
    assert any("futility only" in x for x in probs) and any("only after look 2" in x for x in probs)


# ------------------------------------------------------------------------------------------------ the dose VOID guard
def test_the_dose_guard_holds_each_field_to_the_reference(monkeypatch, tmp_path):
    ref = {"lr_median": 1e-5, "updates_per_env_step": 2.044055033449610e-03, "dose_rate": 2.0440550334496103e-08}
    runs = {"ref": ref, "same": dict(ref), "moved": dict(ref, dose_rate=2.05e-08), "none": dict(ref, lr_median=None)}
    import main.dose as DO

    monkeypatch.setattr(DO, "read_run", lambda d: dict(runs[d.rsplit("/", 1)[-1]]))
    assert RU.void_dose_guard("/r/same", "/r/ref", 1e-6)["pass"]
    moved = RU.void_dose_guard("/r/moved", "/r/ref", 1e-6)
    assert not moved["pass"] and "dose_rate" in moved["why"]
    assert not RU.void_dose_guard("/r/none", "/r/ref", 1e-6)["pass"]
    models = {"on_s3501": SimpleNamespace(arm="x60t")}
    ok = {"void": {"k9b_dir": "k", "runs": {"on_s3501": "models/x"}, "dose": {"reference": "models/c0",
                                                                             "rel_tol": 1e-6}}, "on": "x60t"}
    assert RU._void_check(ok, models, required=False) == []
    bad = json.loads(json.dumps(ok))
    bad["void"]["dose"]["rel_tol"] = 1
    assert any("void.dose" in x for x in RU._void_check(bad, models, required=False))
    assert "dose: dose_rate" in RU.void_failures({"on_s3501": {"pass": False, "k9b": "PASS", "dose": moved}})["on_s3501"]


# ------------------------------------------------------------------------------------------------ the look-1 DRAFT spec
C0_SHA = {3501: "09a6a2df48509a59fd9aefe6d20d9722aad6717f1730c6dfd75274446b283303",
          3502: "7e9a9890e2b397de6d8b87588c597d7f53e39b84efd3d1080305f77dbf22feec",
          3503: "99096be8eccfc1cef4268ab4d6de15063a386cda475e72340212fdbf3a1b7e26"}
X61P_LOOK1_SPEC_SHA = "1266daadeefe4142a2f2f339bbdedaf4e052c3f2ef3fb8a8b0d5550fb8729f03"


def look1_draft(side_pin: str = "PENDING") -> Dict[str, Any]:
    """X60-T look 1's spec as DRAFTED (the coordinator's file is the registration): ON s3501-s3503 by the settled
    ``final`` rule (§13.1), C0 s3501-s3503 pinned and IMPORTED from X61P look 1, the determinism guard's one re-played
    C0 unit, the VOID gate with the side pin (``PENDING`` until it exists)."""
    on = {f"x60t_s{s}": {"arm": "x60t", "seed": s, "ref": {"kind": "final", "run": f"models/rb_x60t_s{s}",
                                                          "min_step": 300000170, "quiet_s": 300}}
          for s in (3501, 3502, 3503)}
    c0 = {f"c0_s{s}": {"arm": "c0", "seed": s, "ref": {"kind": "pinned", "path": f"models/rb_x35_c0_s{s}/final_model.zip",
                                                      "sha256": C0_SHA[s]}} for s in (3501, 3502, 3503)}
    return {
        "schema": "gen3_read_spec_v1", "read": "x60t_look1_2026-11-06",
        "prereg": "designs/research_state/measurements/x60t_h_teams_2026-11-06/PREREG.md",
        "note": "DRAFT: X60-T look 1, futility only on Δ_F (forum72, 60 units a final); C0 imported from X61P look 1",
        "models": {**on, **c0},
        "anchor": {
            "primary": {"opponent": "metamon:Kakuna", "teamsets": ["forum72"], "unit_seeds": "0-59",
                        "models": sorted(on) + sorted(c0),
                        "import": {"state": "~/gen3ai_archive/x61p_n1p_read/look1",
                                   "spec_sha256": X61P_LOOK1_SPEC_SHA, "group": "primary", "models": sorted(c0)}},
            "determinism": {"opponent": "metamon:Kakuna", "teamsets": ["forum72"], "unit_seeds": "0",
                            "models": ["c0_s3501"]}},
        "decision": {"rule": "x60", "params": {
            "look": 1, "on": sorted(on), "off": sorted(c0), "look1_seeds": [3501, 3502, 3503],
            "primary": {"groups": ["primary"], "teamset": "forum72", "units": 60},
            "determinism_guard": {"group": "determinism", "against": "primary"},
            "void": {"k9b_dir": "designs/research_state/measurements/x60t_h_teams_2026-11-06/k9b",
                     "runs": {lab: f"models/rb_{lab}" for lab in sorted(on)},
                     "pin": side_pin, "dose": {"reference": "models/rb_x35_c0_s3501", "rel_tol": 1e-6}}}}}


def test_the_look1_draft_spec_parses_pending_and_with_a_dummy_side_pin():
    from main.reads import spec as SP

    pend = SP.parse(look1_draft(), allow_pending=True)
    assert pend.pending == ("spec.decision.params.void.pin",)
    with pytest.raises(SP.SpecError, match="PENDING"):
        SP.parse(look1_draft())
    got = SP.parse(look1_draft(side_pin="f4b25238"))           # DUMMY: f4b25238 stands in for the side pin
    assert got.rule == "x60" and got.anchor["primary"].imports.models == ("c0_s3501", "c0_s3502", "c0_s3503")
    assert X.reported(got.params) == []
