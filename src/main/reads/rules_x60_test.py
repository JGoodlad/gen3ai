"""``main.reads`` rule ``x60`` AT X60-T'S PLAY PIN (de15dfc7b + AM-X60T-1): main's ``rules_x60_test`` less every test of
``inputs`` / ``decide`` (they run from a main checkout, PREREG §4, and need modules this pin predates) and of the
interval against ``power.py`` (not in this tree); main holds those. Kept: §11's verdict over every class combination,
the spec-load check, the seed sets, the tie band; added: the look-1 spec parses here, and the run-dir VOID guards are
refused here. Synthetic cells, no game."""
from __future__ import annotations

import itertools
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from main.reads import rules as RU
from main.reads import rules_x60 as X
from main.x35 import d1 as D

ENGINE = "port=0d3d7629f16a0eaf/91;env=393631b886965b0c/126"
SIDES = {"peer_code": "b817d3216c418f75", "policy_mode": "eval"}


# ------------------------------------------------------------------------------------------------ the interval
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
        "guards": {"rederived": "count", "choices": "require"}}
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


def _look1(f_on: Sequence[int]) -> Tuple[Dict, Dict, Any]:
    cells = {"anchor": {"primary": _agroup(["forum72"], {
        **{lab: {"forum72": (w, 1200)} for lab, w in zip(ON8[:3], f_on)},
        **{lab: {"forum72": (w, 1200)} for lab, w in zip(OFF8[:3], F_OFF[:3])}})}, "s": {},
        "guards": {"choices": "require"}}
    p = {"look": 1, "on": ON8[:3], "off": OFF8[:3],
         "primary": {"groups": ["primary"], "teamset": "forum72", "units": 60}}
    spec = _spec({"primary": (["forum72"], ON8[:3] + OFF8[:3], 60, True)}, {})
    return cells, p, spec


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


def test_look1_refuses_the_look2_meters_and_the_contrast():
    _, p, spec = _look1([640, 650, 630])
    p["goal_a"] = {"groups": ["primary"], "teamset": "home72", "units": 7}
    p["contrast"] = {"on": ON8[:3], "off": D1, "groups": ["primary"], "panels": ["forum72"], "units": 30}
    probs = RU.RULES["x60"].check_params(p, models=spec.models, anchor=spec.anchor, s=spec.s)
    assert any("futility only" in x for x in probs) and any("only after look 2" in x for x in probs)


# ------------------------------------------------------------------------------------------------ the dose VOID guard


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


def test_the_look1_spec_parses_at_the_play_pin():
    from main.reads import spec as SP

    got = SP.parse(look1_draft())
    assert got.rule == "x60" and got.anchor["primary"].imports.models == ("c0_s3501", "c0_s3502", "c0_s3503")
    assert got.params["void"]["pin"] == {"x60t": SIDE_PIN, "c0": "f4b25238d"} and X.reported(got.params) == []
    bad = look1_draft(side_pin="PENDING")
    with pytest.raises(SP.SpecError, match="void.pin"):
        SP.parse(bad)


def test_the_run_dir_void_guards_are_read_only_from_a_main_checkout():
    from main.reads import spec as SP

    spec = SP.parse(look1_draft())
    with pytest.raises(RU.RuleInputError, match="main checkout"):
        RU.void_gate(spec)
    doc = look1_draft()
    for k in ("pin", "dose"):
        doc["decision"]["params"]["void"].pop(k)
    gate = RU.void_gate(SP.parse(doc))                  # K9(b) alone is this pin's own gate: read here
    assert gate is not None and set(gate) == {"x60t_s3501", "x60t_s3502", "x60t_s3503", "c0_s3501", "c0_s3502",
                                              "c0_s3503"}
