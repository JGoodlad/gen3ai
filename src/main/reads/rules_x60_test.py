"""``main.reads`` rule ``x60`` (X60-T PREREG §5-§11, §15): the ported interval against ``power.py`` itself (fixed and
simulated inputs), §11's verdict over every class combination, the pooled groups' one-cell checks, the determinism
guard, the spec-load check, the ``dose`` VOID guard and the look-1 DRAFT spec. Synthetic cells, no game."""
from __future__ import annotations

import hashlib
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
#: Each synthetic final's sha256, as the state's ``models.json`` resolves it (the guards' provenance).
SHA = {lab: hashlib.sha256(lab.encode()).hexdigest() for lab in SEED}
#: The synthetic runs' dirs: absent, so no recorded seed is readable (``recorded_seeds`` "where readable").
NO_RUNS = "/nonexistent/x60_rule_test"
PINS = {"x60t": "7fe15726c77cfa823aee78110444bf25c3c7370e", "c0": "f4b25238d", "d1": "440b61c32"}


def _spec(groups: Dict[str, Tuple[Sequence[str], Sequence[str], int, bool]], s: Dict[str, Tuple[str, Sequence[str]]]
          ) -> SimpleNamespace:
    arm = {**{lab: "x60t" for lab in ON8}, **{lab: "c0" for lab in OFF8}, **{lab: "d1" for lab in D1}}
    models = {lab: SimpleNamespace(arm=arm[lab], seed=SEED[lab], load={},
                                   ref=SimpleNamespace(kind="final", run=f"{NO_RUNS}/rb_{lab}")) for lab in arm}
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
        "guards": {"rederived": "count", "choices": "require"},
        "models": {lab: {"label": lab, "sha256": SHA[lab]} for lab in ON8 + OFF8 + D1}}
    p = {"look": 2, "on": ON8, "off": OFF8,
         "primary": {"groups": ["primary", "c0_look1", "c0_look2"], "teamset": "forum72", "units": 60},
         "p": {"groups": ["p", "p_c0_look1", "p_c0_look2"], "panel": "home72"},
         "i_f": {"groups": ["i_f"], "panel": "forum72"}, "r72": {"groups": ["r72"], "panel": "r72"},
         "goal_a": {"groups": ["home"], "teamset": "home72", "units": 7},
         "contrast": {"on": ON8[:5], "off": D1, "groups": ["primary", "contrast_h", "d1"],
                      "panels": ["heldout72_hpfix", "forum72"], "units": 30},
         "guards": {"path": f"{NO_RUNS}/guards.json", "sha256": "0" * 64},       # unreadable: every guard not read
         "void": {"k9b_dir": "k9b", "runs": {lab: f"{NO_RUNS}/rb_{lab}" for lab in ON8 + OFF8 + D1}, "pin": dict(PINS),
                  "dose": {"reference": f"{NO_RUNS}/rb_c0_s3501", "rel_tol": 1e-6}}}
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


# ------------------------------------------------------------------------------------------------ the guards (AM-X60T-3)
def _gf_cell(row: str, col: str, k: int, *, pairs: int = 500, sha: Optional[Dict[str, str]] = None) -> Any:
    """One ledger cell of ``pairs`` mirrored pairs scoring 50 + k / 10 pp (``k`` pairs moved from index 1 to 3 of a
    symmetric pentanomial)."""
    from agents.training.eval_ledger.cells import Cell

    sh = sha or SHA
    q = pairs // 10
    pc = (q, 2 * q - k * pairs // 500, pairs - 6 * q, 2 * q + k * pairs // 500, q)
    return Cell(request_id="x60t_gf_look2", family="x60t_gf", opened="t0", player=sh[row], opponent=sh[col],
                player_id=row, opponent_id=col, rows=1, w=0, l=0, d=0, aborted=0, pairs=pc, verdict="OK", reasons=())


def _gf(delta_pp: float = 0.0, *, seed: int = 3, cells: Any = None) -> Dict[str, Any]:
    """A ``g_f`` block through the builder (``x60_guards.gf_block_from_cells``): ON × OFF at Δ ≈ ``delta_pp``."""
    from main.reads import x60_guards as XG

    rng = np.random.default_rng(seed)
    st = {"models": {lab: {"seed": SEED[lab]} for lab in ON8 + OFF8}, "decision": {"params": {"on": ON8, "off": OFF8}}}
    ms = {lab: {"sha256": SHA[lab], "source": f"/x/{lab}.zip"} for lab in ON8 + OFF8}
    if cells is None:
        cells = [_gf_cell(r, c, int(round(10 * (delta_pp + rng.normal(0, 1.0))))) for r in ON8 for c in OFF8]
    return XG.gf_block_from_cells(cells, st, ms, family="x60t_gf", request="x60t_gf_look2")


def _r2(on: Sequence[int] = (30, 200), off: Sequence[int] = (31, 200)) -> Dict[str, Any]:
    return {"on": list(on), "off": list(off), "stat": "pooled",
            "readers": {"on": {"run": "rb_x35_ge_x60t_s3501"}, "off": {"run": "rb_x35_ge_c0_s3501"}},
            "targets": {"on": {"sha256": SHA[ON8[0]]}, "off": {"sha256": SHA[OFF8[0]]}}}


def _gc(b1: float = 0.0, b2: float = 0.0) -> Dict[str, Any]:
    from main.x35 import gc as GC

    eight = list(range(3501, 3509))
    rec = GC.constructed_record("production", eight, [b1] * 8, [b2] * 8)
    rec["checkpoint_sha256"] = {**{f"ON_s{s}": SHA[ON8[i]] for i, s in enumerate(eight)},
                                **{f"OFF_s{s}": SHA[OFF8[i]] for i, s in enumerate(eight)}}
    return rec


def _artefact(tmp_path: Any, p: Dict[str, Any], *, g_f: Any = "pass", r2: Any = "pass", g_c: Any = "pass",
              schema: str = X.GUARDS_SCHEMA, status: str = X.PLAYED) -> None:
    from utils.hashing import sha256_file

    doc = {"schema": schema, "status": status, "g_f": _gf() if g_f == "pass" else g_f,
           "r2": _r2() if r2 == "pass" else r2, "g_c": _gc() if g_c == "pass" else g_c}
    art = tmp_path / "guards.json"
    art.write_text(json.dumps(doc))
    p["guards"] = {"path": str(art), "sha256": sha256_file(art)}


def _status(dec: Dict[str, Any]) -> Dict[str, str]:
    return {n: g["status"] for n, g in dec["guards"].items()}


def test_a_passing_gain_reads_adopt_only_with_every_guard_read_from_a_filled_artefact(tmp_path):
    """§11: ADOPT-CANDIDATE needs G-F / G-E / G-C PASS, read from the filled artefact the builder writes; the
    not-played stub, an unregistered or a tampered artefact reads INCONCLUSIVE."""
    cells, p, spec = _look2(home_wins=95)
    dec = _decide(cells, p, spec)
    assert dec["decision"]["verdict"] == X.INCONCLUSIVE and set(_status(dec).values()) == {"INCONCLUSIVE"}
    _artefact(tmp_path, p)
    dec = _decide(cells, p, spec)
    assert _status(dec) == {"G-F": "PASS", "G-E": "PASS", "G-C": "PASS"}
    assert dec["decision"]["verdict"] == X.ADOPT and dec["guards"]["G-F"]["detail"]["size"] == 8
    p["guards"]["sha256"] = "0" * 64
    assert "sha256" in _decide(cells, p, spec)["guards"]["G-F"]["reasons"][0]
    _artefact(tmp_path, p, g_f=None, r2=None, g_c=None, status=X.NOT_PLAYED)          # the stub's shape
    dec = _decide(cells, p, spec)
    assert set(_status(dec).values()) == {"INCONCLUSIVE"} and dec["decision"]["verdict"] == X.INCONCLUSIVE


@pytest.mark.parametrize("schema,status,blocks,needle", [
    ("gen3_x60t_guards_v0", X.PLAYED, True, "schema"),
    (X.GUARDS_SCHEMA, "DONE", True, "status"),
    (X.GUARDS_SCHEMA, X.NOT_PLAYED, True, "carries a guard input"),
])
def test_an_artefact_of_another_schema_or_status_reads_every_guard_inconclusive(tmp_path, schema, status, blocks,
                                                                                needle):
    cells, p, spec = _look2(home_wins=95)
    _artefact(tmp_path, p, schema=schema, status=status)
    dec = _decide(cells, p, spec)
    assert set(_status(dec).values()) == {"INCONCLUSIVE"} and dec["decision"]["verdict"] == X.INCONCLUSIVE
    assert all(needle in g["reasons"][0] for g in dec["guards"].values())


def _gf_mut(kind: str) -> Dict[str, Any]:
    g = _gf()
    if kind == "three_by_three":                       # a look-1-sized matrix: the old reader took any square one
        keep_r, keep_c = ON8[:3], OFF8[:3]
        g["cells"] = [c for c in g["cells"] if c["row"] in keep_r and c["col"] in keep_c]
        g["rows"], g["cols"] = keep_r, keep_c
        g["h_pp"] = [row[:3] for row in g["h_pp"][:3]]
    elif kind == "other_final":                        # a cell played by a checkpoint that is not the read's final
        g["cells"][9]["player_sha256"] = "f" * 64
    elif kind == "seed_order":
        g["rows"] = list(reversed(g["rows"]))
    elif kind == "short_cell":                         # 490 pairs, its h_pp entry consistent: only the floor catches it
        short = _gf_cell(ON8[2], OFF8[5], 0, pairs=490)
        cell = next(c for c in g["cells"] if (c["row"], c["col"]) == (ON8[2], OFF8[5]))
        cell["pairs"] = list(short.pairs)
        g["h_pp"][2][5] = 100.0 * short.score
    elif kind == "doctored_h":                         # h_pp no longer the cells' pair scores
        g["h_pp"][0][0] += 5.0
    elif kind == "missing_cell":
        g["cells"] = g["cells"][1:]
    elif kind == "min_pairs":
        g["min_pairs"] = 1000
    elif kind == "not_ok":
        g["cells"][4]["verdict"] = "INCONCLUSIVE"
    return g


@pytest.mark.parametrize("kind", ["three_by_three", "other_final", "seed_order", "short_cell", "doctored_h",
                                  "missing_cell", "min_pairs", "not_ok"])
def test_g_f_is_inconclusive_unless_it_is_this_looks_matrix_of_this_reads_finals(tmp_path, kind):
    cells, p, spec = _look2(home_wins=95)
    _artefact(tmp_path, p, g_f=_gf_mut(kind))
    dec = _decide(cells, p, spec)
    assert _status(dec)["G-F"] == "INCONCLUSIVE" and dec["decision"]["verdict"] == X.INCONCLUSIVE, kind


def test_g_f_fails_on_a_detected_head_to_head_loss_and_the_gain_is_a_trade(tmp_path):
    cells, p, spec = _look2(home_wins=95)
    _artefact(tmp_path, p, g_f=_gf(-6.0))
    dec = _decide(cells, p, spec)
    g = dec["guards"]["G-F"]
    assert g["status"] == "FAIL" and g["detail"]["ci95_pp"][1] < -2.0
    assert dec["decision"]["verdict"] == X.TRADE and "G-F FAIL" in dec["decision"]["reasons"][0]
    _artefact(tmp_path, p, g_f=_gf(-1.0))                  # a small, undetected loss PASSes
    assert _status(_decide(cells, p, spec))["G-F"] == "PASS"


@pytest.mark.parametrize("kind", ["reader", "target", "stat", "no_targets"])
def test_g_e_is_inconclusive_unless_its_readers_target_this_reads_seed_3501_finals(tmp_path, kind):
    cells, p, spec = _look2(home_wins=95)
    r2 = _r2()
    if kind == "reader":
        r2["readers"]["on"]["run"] = "rb_x35_ge_d1wide_s3501"
    elif kind == "target":
        r2["targets"]["off"]["sha256"] = SHA[OFF8[1]]        # C0 s3502's final, not s3501's
    elif kind == "stat":
        r2["stat"] = "endpoint"
    else:
        del r2["targets"]
    _artefact(tmp_path, p, r2=r2)
    dec = _decide(cells, p, spec)
    assert _status(dec)["G-E"] == "INCONCLUSIVE" and dec["decision"]["verdict"] == X.INCONCLUSIVE, kind


def test_g_e_fails_when_on_is_detected_more_exploitable(tmp_path):
    cells, p, spec = _look2(home_wins=95)
    _artefact(tmp_path, p, r2=_r2(on=(110, 200), off=(40, 200)))
    dec = _decide(cells, p, spec)
    assert _status(dec)["G-E"] == "FAIL" and dec["decision"]["verdict"] == X.TRADE


@pytest.mark.parametrize("kind", ["other_on", "other_off", "no_shas", "five_pairs"])
def test_g_c_is_inconclusive_unless_every_pair_reads_this_reads_finals(tmp_path, kind):
    from main.x35 import gc as GC

    cells, p, spec = _look2(home_wins=95)
    rec = _gc()
    if kind == "other_on":
        rec["checkpoint_sha256"]["ON_s3504"] = "e" * 64
    elif kind == "other_off":
        rec["checkpoint_sha256"]["OFF_s3508"] = SHA[OFF8[0]]
    elif kind == "no_shas":
        del rec["checkpoint_sha256"]
    else:                                              # an X35 five-pair record does not fit an 8 v 8 read
        rec = GC.constructed_record("production", list(range(3501, 3506)), [0.0] * 5, [0.0] * 5)
    _artefact(tmp_path, p, g_c=rec)
    dec = _decide(cells, p, spec)
    assert _status(dec)["G-C"] == "INCONCLUSIVE" and dec["decision"]["verdict"] == X.INCONCLUSIVE, kind


def test_g_c_fails_on_a_collapse_and_the_gain_is_a_trade(tmp_path):
    cells, p, spec = _look2(home_wins=95)
    _artefact(tmp_path, p, g_c=_gc(b1=-0.2))
    dec = _decide(cells, p, spec)
    assert _status(dec)["G-C"] == "FAIL" and dec["decision"]["verdict"] == X.TRADE


def test_a_guard_reads_inconclusive_when_the_state_records_no_sha_for_a_final(tmp_path):
    cells, p, spec = _look2(home_wins=95)
    cells["models"][OFF8[0]].pop("sha256")
    _artefact(tmp_path, p)
    dec = _decide(cells, p, spec)
    assert set(_status(dec).values()) == {"INCONCLUSIVE"}
    assert any("no sha256" in w for w in dec["guards"]["G-E"]["reasons"])


# ------------------------------------------------------------------------------------------------ REPORTED pooled (§12)
def _reported_cells() -> Tuple[Dict, Dict, Any]:
    cells, p, spec = _look2()
    h_on_late = {lab: {"heldout72_hpfix": (75, 140)} for lab in ON8[5:]}
    h_off_late = {lab: {"heldout72_hpfix": (60, 140)} for lab in OFF8[5:]}
    cells["anchor"]["h"] = _agroup(["heldout72_hpfix"], {**h_on_late, **h_off_late})
    cells["anchor"]["h_c0a"] = _agroup(["heldout72_hpfix"], {lab: {"heldout72_hpfix": (63, 140)} for lab in OFF8[:5]},
                                      imported_from="/x61p/look1")
    p["reported"] = ["h", "contrast_h", "h_c0a"]
    spec.anchor["h"] = SimpleNamespace(name="h", teamsets=("heldout72_hpfix",), models=tuple(ON8[5:] + OFF8[5:]),
                                       unit_seeds=tuple(range(7)), per_game=False, imports=None)
    spec.anchor["h_c0a"] = SimpleNamespace(name="h_c0a", teamsets=("heldout72_hpfix",), models=tuple(OFF8[:5]),
                                           unit_seeds=tuple(range(7)), per_game=False,
                                           imports=SimpleNamespace(models=tuple(OFF8[:5])))
    return cells, p, spec


def test_the_reported_delta_h_pools_units_0_6_across_its_groups_in_code():
    """FINDING 28 (AM-X60T-3): Δ_H pools ON s3506-s3508 (``h``) with ON s3501-s3505's units 0-6 of the 30-unit
    ``contrast_h`` group, against C0's ``h`` and imported groups; units 7-29 are not read."""
    cells, p, spec = _reported_cells()
    dec = _decide(cells, p, spec)
    h = dec["reported"]["pooled"]["heldout72_hpfix"]
    # contrast_h: 320 wins of 600 over 30 units, spread as _agroup does (11 or 10 wins a unit) -> units 0-6 hold 77
    want_on_w = 5 * sum(320 // 30 + (u < 320 % 30) for u in range(7)) + 3 * 75
    assert h["status"] == "OK" and h["name"] == "Δ_H" and h["units"] == 7
    assert (h["on"]["wins"], h["on"]["n"]) == (want_on_w, 8 * 140)
    assert (h["off"]["wins"], h["off"]["n"]) == (5 * 63 + 3 * 60, 8 * 140)
    assert h["delta"] == pytest.approx(want_on_w / 1120 - (5 * 63 + 3 * 60) / 1120, abs=1e-15)
    assert h["groups"] == ["h", "contrast_h", "h_c0a"]


@pytest.mark.parametrize("break_", ["regime", "short"])
def test_a_split_or_short_pooled_delta_is_reported_incomplete_and_decides_nothing(break_):
    cells, p, spec = _reported_cells()
    if break_ == "regime":
        cells["anchor"]["h_c0a"]["regime_ids"]["heldout72_hpfix"] = "q" * 16
    else:
        cells["anchor"]["h"]["units"] = [u for u in cells["anchor"]["h"]["units"]
                                         if not (u["label"] == ON8[7] and u["unit_seed"] == 6)]
    dec = _decide(cells, p, spec)
    h = dec["reported"]["pooled"]["heldout72_hpfix"]
    assert h["status"] == "INCOMPLETE" and h["reasons"]
    assert dec["decision"]["verdict"] == _decide(*_reported_cells())["decision"]["verdict"]


def test_look1_carries_no_pooled_reported_block_and_no_finals():
    cells, p, spec = _look1([612, 640, 598])
    doc = RU.inputs("x60", cells, p, spec)
    assert "reported_pooled" not in doc and "finals" not in doc


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
    p = {"look": 1, "on": ON8[:3], "off": OFF8[:3],
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


def test_the_stage_identity_is_final_only_at_a_stop_or_a_classed_look_2():
    """AM-X60T-2: ``stage_keys`` (the record's top-level ``arm`` / ``look`` / ``delta_f``) on each branch of decide."""
    def keys(dec: Dict[str, Any], spec: Any) -> Dict[str, Any]:
        models = {lab: {"arm": m.arm} for lab, m in spec.models.items()}
        return X.stage_keys(models, json.loads(json.dumps(dec)))

    cells, p, spec = _look1([640, 650, 630])
    assert keys(_decide(cells, p, spec), spec) == {"arm": "X60T", "look": 1,
                                                    X.STAGE_PRIMARY_KEY: {"final": False, "class": X.CONTINUE}}
    for f_on, cls in (([600, 620, 590], X.NUG), ([560, 570, 550], X.LOSS)):
        cells, p, spec = _look1(f_on)
        assert keys(_decide(cells, p, spec), spec)[X.STAGE_PRIMARY_KEY] == {"final": True, "class": cls}
    cells, p, spec = _look2()
    dec = _decide(cells, p, spec)
    assert keys(dec, spec)[X.STAGE_PRIMARY_KEY] == {"final": True, "class": X.GAIN} and keys(dec, spec)["look"] == 2
    cells, p, spec = _look2()
    cells["anchor"]["primary"]["unplayed"] = 3                               # Δ_F INCONCLUSIVE: no class, not final
    dec = _decide(cells, p, spec)
    assert dec["decision"]["verdict"] == X.INCONCLUSIVE
    assert keys(dec, spec)[X.STAGE_PRIMARY_KEY] == {"final": False, "class": None}


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
    (lambda p: p.update(look1_seeds=[3506, 3507, 3508]), "unknown 'look1_seeds'"),
    (lambda p: p.update(guards={"path": "x"}), "guards:"),
    (lambda p: p.pop("guards"), "guards: required at look 2"),
    (lambda p: p.pop("void"), "void: required at look 2"),
    (lambda p: p["void"].update(pin=PINS["x60t"]), "a pin per arm"),
    (lambda p: p["void"]["pin"].pop("d1"), "no pin for 'd1_s3501'"),
    (lambda p: p["void"]["runs"].pop(OFF8[6]), "'c0_s3507' has no run"),
    (lambda p: p.update(extra=1), "unknown 'extra'"),
])
def test_the_spec_load_check_refuses(mutate, needle):
    _, p, spec = _look2()
    mutate(p)
    probs = RU.RULES["x60"].check_params(p, models=spec.models, anchor=spec.anchor, s=spec.s)
    assert any(needle in x for x in probs), probs


def _add_model(spec: Any, lab: str, arm: str, seed: int, run_seed: Optional[int] = None) -> None:
    spec.models[lab] = SimpleNamespace(arm=arm, seed=seed, load={},
                                       ref=SimpleNamespace(kind="final", run=f"{NO_RUNS}/rb_x_s{run_seed or seed}"))


def test_look2_refuses_any_seed_but_s3501_s3508_on_either_side():
    _, p, spec = _look2()
    _add_model(spec, "on_s3509", "x60t", 3509)
    p["on"] = ON8[:7] + ["on_s3509"]
    probs = X.check(p, models=spec.models, anchor=spec.anchor, s=spec.s)
    assert any("on: seeds" in x and "REVIEW I M3" in x for x in probs), probs
    _, p, spec = _look2()
    _add_model(spec, "c0_s3510", "c0", 3510)
    p["off"] = OFF8[:7] + ["c0_s3510"]
    assert any("off: seeds" in x for x in X.check(p, models=spec.models, anchor=spec.anchor, s=spec.s))


def test_look1_refuses_any_triplet_but_s3501_s3503():
    _, p, spec = _look1([640, 650, 630])
    p["on"] = ON8[5:8]
    spec.anchor["primary"].models = tuple(ON8[5:8] + OFF8[:3])
    probs = X.check(p, models=spec.models, anchor=spec.anchor, s=spec.s)
    assert any("on: seeds [3506, 3507, 3508] are not the registered [3501, 3502, 3503]" in x for x in probs), probs


def test_the_contrast_refuses_any_five_but_s3501_s3505_on_both_arms():
    _, p, spec = _look2()
    p["contrast"]["on"] = ON8[3:8]
    spec.anchor["contrast_h"].models = tuple(ON8[3:8])
    assert any("contrast.on: seeds" in x for x in X.check(p, models=spec.models, anchor=spec.anchor, s=spec.s))
    _, p, spec = _look2()
    spec.models[D1[0]] = SimpleNamespace(arm="d1", seed=3509, load={},
                                         ref=SimpleNamespace(kind="final", run=f"{NO_RUNS}/rb_d1_s3509"))
    assert any("contrast.off: seeds" in x for x in X.check(p, models=spec.models, anchor=spec.anchor, s=spec.s))


def test_a_labels_seed_is_its_run_dirs():
    _, p, spec = _look2()
    spec.models[OFF8[0]].ref = SimpleNamespace(kind="pinned", path=f"{NO_RUNS}/rb_x35_c0_s3502/final_model.zip")
    probs = X.check(p, models=spec.models, anchor=spec.anchor, s=spec.s)
    assert any("c0_s3501: its run" in x and "names seed 3502" in x for x in probs), probs
    spec.models[OFF8[0]].ref = SimpleNamespace(kind="pinned", path=f"{NO_RUNS}/rb_x35_c0_s3501/checkpoints/c.zip")
    assert X.check(p, models=spec.models, anchor=spec.anchor, s=spec.s) == []


def test_a_run_recording_another_seed_is_inconclusive(tmp_path):
    cells, p, spec = _look2()
    run = tmp_path / "rb_x60t_s3501"
    run.mkdir()
    (run / "metadata.json").write_text(json.dumps({"cli_args": {"seed": 3502}}))
    spec.models[ON8[0]].ref = SimpleNamespace(kind="final", run=str(run))
    dec = _decide(cells, p, spec)
    assert dec["decision"]["verdict"] == X.INCONCLUSIVE and "REVIEW I M3" in dec["decision"]["reasons"][0]
    (run / "metadata.json").write_text(json.dumps({"cli_args": {"seed": 3501}}))
    assert _decide(cells, p, spec)["decision"]["verdict"] == X.TRADE         # the fixture's GAIN, goal (a) failing
    rec = X.recorded_seeds(spec, ON8 + OFF8 + D1)
    assert rec[ON8[0]]["recorded"] == 3501 and all(r["ok"] for r in rec.values())


def test_the_welch_guard_tie_band_takes_the_wider_interval(monkeypatch):
    on = [[(w, 1200)] for w in (560, 700, 600, 660, 520, 720, 610, 640)]
    off = [[(w, 1200)] for w in (590, 610, 600, 605, 600, 615, 600, 610)]
    lb = X.x60_interval(on, off)["sigma2_lb95"]
    for offset, near in ((5e-13, True), (-5e-13, True), (2e-12, False), (-2e-12, False)):
        monkeypatch.setattr(X, "SIGMA_F", (lb - offset) ** 0.5)     # the floor² sits `offset` below the LB
        s = X.x60_interval(on, off)
        assert s["guard_near"] is near
        if near:
            assert s["halfwidth"] == max(s["halfwidth_registered"], s["halfwidth_welch"])
            assert s["welch_used"] == (s["halfwidth_welch"] > s["halfwidth_registered"])
        else:
            assert s["welch_used"] is (offset > 0)


def test_the_void_gate_pins_each_run_at_its_arms_commit(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(RU, "void_pin_guard", lambda rd, want: seen.setdefault(str(rd).rsplit("/", 1)[-1], want)
                        and {"pass": True, "want": want, "why": ""})
    _, p, spec = _look2()
    v = dict(p["void"], k9b_dir=str(tmp_path))
    v.pop("dose")
    gate = RU.void_gate(SimpleNamespace(params={"void": v}, models=spec.models))
    assert {seen[f"rb_{lab}"] for lab in ON8} == {PINS["x60t"]}
    assert {seen[f"rb_{lab}"] for lab in OFF8} == {PINS["c0"]} and {seen[f"rb_{lab}"] for lab in D1} == {PINS["d1"]}
    assert all(row["pin"]["want"] == PINS[spec.models[lab].arm] for lab, row in gate.items())


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


SIDE_PIN = "7fe15726c77cfa823aee78110444bf25c3c7370e"


def look1_draft(side_pin: str = SIDE_PIN) -> Dict[str, Any]:
    """X60-T look 1's spec as DRAFTED (the coordinator's file is the registration): ON s3501-s3503 by the settled
    ``final`` rule (§13.1), C0 s3501-s3503 pinned and IMPORTED from X61P look 1, the determinism guard's one re-played
    C0 unit, the VOID gate with a pin PER ARM (ON at the side pin, C0 at f4b25238d; REVIEW I m4) gating ON and C0."""
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
            "look": 1, "on": sorted(on), "off": sorted(c0),
            "primary": {"groups": ["primary"], "teamset": "forum72", "units": 60},
            "determinism_guard": {"group": "determinism", "against": "primary"},
            "void": {"k9b_dir": "designs/research_state/measurements/x35_breakers_stage_2026-10-17/k9b",
                     "runs": {**{lab: f"models/rb_{lab}" for lab in sorted(on)},
                              **{lab: f"models/rb_x35_{lab}" for lab in sorted(c0)}},
                     "pin": {"x60t": side_pin, "c0": "f4b25238d"},
                     "dose": {"reference": "models/rb_x35_c0_s3501", "rel_tol": 1e-6}}}}}


def test_the_look1_draft_spec_parses_with_a_pin_per_arm():
    from main.reads import spec as SP

    got = SP.parse(look1_draft())
    assert got.rule == "x60" and got.anchor["primary"].imports.models == ("c0_s3501", "c0_s3502", "c0_s3503")
    assert got.params["void"]["pin"] == {"x60t": SIDE_PIN, "c0": "f4b25238d"} and X.reported(got.params) == []
    pend = SP.parse(look1_draft(side_pin="PENDING"), allow_pending=True)
    assert pend.pending == ("spec.decision.params.void.pin.x60t",)
    with pytest.raises(SP.SpecError, match="PENDING"):
        SP.parse(look1_draft(side_pin="PENDING"))


#: AM-X60T-1: X60-T's PLAY PIN (de15dfc7b + rule x60, its void-check support and R72's registration; tag x60t-play-*).
PLAY_PIN = "f564c333f9437f6a636e9daf890535c2c300aeb2"
LOOK1_SPEC = repo_path("designs/research_state/measurements/x60t_h_teams_2026-11-06/read_spec_look1.json")


def test_the_registered_look1_spec_is_the_draft_at_the_play_pin():
    """AM-X60T-1 registers look 1's spec: the draft above, the play pin as its ``pin`` and its own note, nothing else."""
    from main.reads import spec as SP

    doc = json.loads(LOOK1_SPEC.read_text())
    assert doc["pin"] == PLAY_PIN and {k: v for k, v in doc.items() if k not in ("pin", "note")} == \
        {k: v for k, v in look1_draft().items() if k != "note"}
    got = SP.load(LOOK1_SPEC)
    assert got.rule == "x60" and got.pin == PLAY_PIN and got.params["look"] == 1


def test_main_reconstructs_the_play_pins_engine_and_sides():
    """AM-X60T-1: the pin predates the engine / side stamps, so main names its units' engine and sides from these rows;
    without them every X60 group's import refuses (an UNKNOWN engine or side)."""
    from main.reads import engine as EN
    from main.reads import side_identity as SI

    assert EN.legacy_engine(PLAY_PIN) == EN.E3 == EN.legacy_engine("de15dfc7b6d19d6f0e4a3724f269e26e7aa22957")
    assert SI.LEGACY_COMMIT_MODE[PLAY_PIN] == "eval"
    cell = {"opponent_kind": "metamon", "our_side": "model",
            "server_version": f"ws_frontend[in-process]@{PLAY_PIN[:9]}+rust:x"}
    assert SI.unit_sides(cell) == {"peer_code": "their=legacy", "policy_mode": "eval"}
