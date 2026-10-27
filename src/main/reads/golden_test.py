"""GOLDEN: three SHIPPED reads, re-decided (and, on the owner's box, re-assembled from their archived ledgers and unit
dirs) by ``main.reads``, reproduce their RECORDED verdicts and intervals EXACTLY (float equality, no tolerance):

* SPRT look 1 (``x35_breakers_stage_2026-10-17/reads/sprt_look1.json``: R1 NULL, final), rule ``x35_r1``;
* D1 look 2 (``reads/d1_look2.json``: UNDECIDED, final), rule ``d1``;
* x26c (``x26c_continuation_2026-10-25/read/verdict.json``: goal (b) not met; FLAT), rule ``anchor_trajectory``.

The COMMITTED half (always run) feeds each read's committed input / unit rows to the rule. The ARCHIVE half reads
``~/gen3ai_archive`` (``$GEN3AI_READS_ARCHIVE`` overrides) READ-ONLY and skips, naming why, where it is absent (it
is never committed: the ledgers and unit dirs are GBs); there it also proves every regime id is recorded -- the ids
the READMEs checked by hand (x26c FINDING 4, g1 FINDING 3).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest

from main.reads import assemble as AS
from main.reads import record as REC
from main.reads import rules as RU
from main.reads.spec import parse
from utils.paths import repo_path

MEAS = "designs/research_state/measurements"
STAGE = repo_path(MEAS, "x35_breakers_stage_2026-10-17")
X26C = repo_path(MEAS, "x26c_continuation_2026-10-25", "read")
REFPOP = f"{MEAS}/x35_breakers_stage_2026-10-17/reference_population_production.json"
PANELS = ["heldout72_hpfix", "forum72", "p72"]
G1_RESULTS = {"path": f"{MEAS}/x35_breakers_stage_2026-10-17/g1/results.json",
              "sha256": "393b7d572bb03c25afc089e3c7f9ca121c79aab9e1d436380fbbac96d9cf6786", "arm": "c0"}
K9B = f"{MEAS}/x26c_continuation_2026-10-25/read/k9b_at_end.json"
POOL_REGIME, HOME72_REGIME = "adfdbee824c9eb0a", "d4de0fd7c16a2658"
#: The env core SPRT look 1's pool rows stamp (``compute.core_stamp`` at eb99ab362; E1, G1's too).
SPRT_S_ENGINE = "env=7c53280566d666ae/125"
#: the Kakuna unfamiliar-panel group's id (main.reads.assemble.group_regime_id): the SAME for D1's and x26c's units,
#: played at different commits on the same committed configs (x26c FINDING 1)
KAKUNA_UNFAMILIAR_REGIME = "09f297883701a35a"


def _archive() -> Path:
    return Path(os.environ.get("GEN3AI_READS_ARCHIVE") or Path.home() / "gen3ai_archive")


def _need(*parts: str) -> Path:
    p = _archive().joinpath(*parts)
    if not p.exists():
        pytest.skip(f"{p}: the archived read is absent (never committed; it exists on the owner's box only)")
    return p


def _json(p: Path) -> Any:
    return json.loads(Path(p).read_text())


def _refpop_content() -> str:
    return str(_json(repo_path(REFPOP))["content_sha256"])


def _pinned(path: str, sha: str) -> Dict[str, Any]:
    return {"kind": "pinned", "path": path, "sha256": sha}


# ------------------------------------------------------------------------------------------------ the specs
def sprt_spec(shas: Dict[str, str]) -> Any:
    rec = _json(STAGE / "reads/sprt_look1.json")
    return parse({
        "schema": "gen3_read_spec_v1", "read": "x35_sprt_look1_2026-10-24",
        "prereg": f"{MEAS}/x35_breakers_stage_2026-10-17/STAGED_PLAN.md",
        "models": {f"sprt_s{s}": {"arm": "sprt", "seed": s, "ref": _pinned(f"/archive/sprt_s{s}.zip", shas[str(s)])}
                   for s in (3501, 3502, 3503)},
        "s": {"pool": {"refpop": REFPOP, "refpop_content_sha256": _refpop_content(), "panel": None, "pairs": 500,
                       "batch_pairs": 100, "family": "x35_sprt_read_2026-10-24",
                       "models": ["sprt_s3501", "sprt_s3502", "sprt_s3503"], "regime_id": POOL_REGIME}},
        "decision": {"rule": "x35_r1", "params": {
            "arm": "SPRT", "look": 1, "on": "sprt", "off": "c0", "s_group": "pool", "import_off": G1_RESULTS,
            "dose": _json(STAGE / "sprt_read/read_look1_input.json")["dose"],
            "void": {"k9b_dir": f"{MEAS}/x35_breakers_stage_2026-10-17/k9b",
                     "runs": {f"sprt_s{s}": f"models/rb_x35_sprt_s{s}" for s in (3501, 3502, 3503)}},
            "guards_not_read": {"guards": ["G-F", "G-C", "G-K"], "why": rec["guards"]["G-F"]["reasons"][0],
                                "absent": {"G-E": rec["guards"]["G-E"]["reasons"][0]}}}}})


def d1_spec(shas: Dict[str, str]) -> Any:
    rec = _json(STAGE / "reads/d1_look2.json")
    labels = [f"{a}_s{s}" for a in ("c0", "d1") for s in range(3501, 3506)]
    return parse({
        "schema": "gen3_read_spec_v1", "read": "x35_d1_look2_2026-10-24",
        "models": {lab: {"arm": lab[:2].upper(), "seed": int(lab[-4:]),
                         "ref": _pinned(f"/archive/{lab}.zip", shas.get(lab, "0" * 64))} for lab in labels},
        "anchor": {"h": {"opponent": "metamon:Kakuna", "teamsets": PANELS, "unit_seeds": "0-6", "models": labels}},
        "s": {"p": {"refpop": REFPOP, "refpop_content_sha256": _refpop_content(), "panel": "home72", "pairs": 500,
                    "batch_pairs": 100, "family": "x35_d1_p_2026-10-23", "models": labels,
                    "regime_id": HOME72_REGIME}},
        "decision": {"rule": "d1", "params": {
            "arm": "X38", "look": 2, "on": "D1", "off": "C0", "anchor_group": "h", "s_group": "p",
            "guards_not_read": {"guards": ["G-F", "G-E", "G-C"], "why": rec["guards"]["G-F"]["reasons"][0]}}}})


def x26c_spec(shas: Dict[str, str]) -> Any:
    return parse({
        "schema": "gen3_read_spec_v1", "read": "x26c_continuation_2026-10-25",
        "prereg": f"{MEAS}/x26c_continuation_2026-10-25/PREREG.md", "pin": "e05c45eeb",
        "models": {"pstar": {"arm": "pstar", "seed": 2601, "ref": _pinned(
                       "models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip", shas["pstar"])},
                   "x26c_480m": {"arm": "x26c_480m", "seed": 2601, "ref": {
                       "kind": "final", "run": "models/rb_x26c_s2601", "min_step": 480_000_000, "quiet_s": 300}}},
        "anchor": {"unfamiliar": {"opponent": "metamon:Kakuna", "teamsets": PANELS, "unit_seeds": "0-19",
                                  "models": ["pstar", "x26c_480m"]}},
        "s": {"pool": {"refpop": REFPOP, "refpop_content_sha256": _refpop_content(), "panel": None, "pairs": 500,
                       "batch_pairs": 100, "family": "x26c_s_pool_2026-10-25", "models": ["x26c_480m"],
                       "regime_id": POOL_REGIME}},
        "decision": {"rule": "anchor_trajectory", "params": {
            "on": "x26c_480m", "off": "pstar", "anchor_group": "unfamiliar", "goal_bar": 0.50, "min_unit_frac": 0.90,
            "subset_below": 7, "s1": {"group": "pool", "label": "x26c_480m"},
            "s_baseline": {"s_pp": 58.23, "se_pp": 0.55, "source": "x26_drift_preread results.json, 380M (dbc49cc1)"},
            "k9b": K9B, "verdict_text": {"met": "GOAL (b) MET", "not_met": "goal (b) not met"}}}})


def _x26c_matches(dec: Dict[str, Any]) -> None:
    """``anchor_trajectory``'s generic keys against x26c's own (``s_480m`` = ``s_pp`` ...), each exactly."""
    want = _json(X26C / "verdict.json")
    assert dec["kakuna"] == want["kakuna"]
    d, w = dec["decision"], want["decision"]
    assert (d["verdict"], d["trajectory"], d["goal_met"], d["d3_revisit"]) == (
        w["verdict"], w["trajectory"], w["goal_b_met"], w["d3_revisit"]) == ("goal (b) not met; FLAT", "FLAT", False,
                                                                           False)
    for mine, theirs in (("s_pp", "s_480m"), ("se_pp", "se"), ("d_vs_baseline", "d_vs_380m"), ("lo", "lo"),
                         ("hi", "hi"), ("rising", "rising")):
        assert d["s1"][mine] == w["s1"][theirs], mine


# ------------------------------------------------------------------------------------------------ committed half
def test_sprt_look1_redecides_exactly_from_its_committed_input():
    inp = _json(STAGE / "sprt_read/read_look1_input.json")
    rec = _json(STAGE / "reads/sprt_look1.json")
    spec = sprt_spec({s: "0" * 64 for s in ("3501", "3502", "3503")})
    cells = {"s": {"pool": {"nodes": {f"sprt_s{s}": {"s_pp": v} for s, v in inp["r1"]["on"].items()},
                            "engine": SPRT_S_ENGINE}}}
    doc = RU.inputs("x35_r1", cells, spec.params, spec)
    assert doc == inp, "the rule's input from the cells (C0 imported from G1's committed results) is the read's"
    v = RU.decide("x35_r1", doc, spec.params)
    for k in ("arm", "look", "verdict", "r1", "r2", "reasons", "guards"):
        assert v[k] == rec[k], k
    assert (v["verdict"], v["r1"]["class"], v["r1"]["final"]) == ("NULL", "NULL", True)


def test_d1_look2_redecides_exactly_from_its_committed_input():
    inp = _json(STAGE / "d1_read/read_look2_input.json")
    rec = _json(STAGE / "reads/d1_look2.json")
    spec = d1_spec({})
    v = RU.decide("d1", inp, spec.params)
    for k in ("arm", "look", "verdict", "h", "p", "reasons", "guards"):
        assert v[k] == rec[k], k
    assert v["verdict"] == "UNDECIDED" and v["h"]["final"] is True


def test_x26c_redecides_exactly_from_its_committed_units():
    units = [json.loads(ln) for ln in (X26C / "kakuna_units.jsonl").read_text().splitlines()]
    s1 = next(n for n in _json(X26C / "s_results.json")["nodes"] if n["label"] == "x26c_480m_pool")
    spec = x26c_spec({"pstar": "814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed"})
    group = AS.anchor_group([dict(u, regime_id=None) for u in units], labels=["pstar", "x26c_480m"],
                            teamsets=tuple(PANELS), unit_seeds=tuple(range(20)))
    assert group["inconclusive"] and group["regime_id"] is None, "the committed rows carry no regime: never conclusive"
    cells = {"anchor": {"unfamiliar": group},
             "s": {"pool": {"nodes": {"x26c_480m": {"s_pp": s1["s_pp"], "se_pp": s1["se_pp"], "inconclusive": []}}}}}
    doc = RU.inputs("anchor_trajectory", cells, spec.params, spec)
    assert doc["k9b_verdict"] == "PASS" and doc["units_planned"] == 60
    _x26c_matches(RU.decide("anchor_trajectory", doc, spec.params))


# ------------------------------------------------------------------------------------------------ archive half
def _members() -> Dict[str, str]:
    return AS.members_of(repo_path(REFPOP), verify=False, want_content=_refpop_content())


def _code(commit: str) -> Dict[str, Any]:
    return {"commit": commit, "dirty": False, "pin": None, "root": "(the read's own checkout)"}


def test_sprt_look1_reassembles_from_its_ledgers_exactly(tmp_path):
    st = _json(_need("x35_sprt_read", "state.json"))
    shas = {lab: v["sha256"] for lab, v in st["nodes"].items()}
    spec = sprt_spec({lab[-4:]: sha for lab, sha in shas.items()})
    grp = AS.s_group(_need("x35_sprt_read", "ledger"), "x35_sprt_read_2026-10-24",
                     {lab: f"sprt_{lab}_b100" for lab in shas}, shas, _members(), want_regime=POOL_REGIME,
                     refpop_content_sha256=_refpop_content())
    assert grp["regime_id"] == POOL_REGIME and not grp["inconclusive"] and grp["engine"] == SPRT_S_ENGINE
    # the imported control IS G1's ledger, re-assembled
    g1 = _json(_need("x35_g1", "state.json"))
    g1shas = {lab: v["sha256"] for lab, v in g1["nodes"].items() if lab.startswith("c0_")}
    g1grp = AS.s_group(_need("x35_g1", "ledger"), "x35_g1_2026-10-22", {lab: f"g1_{lab}_b100" for lab in g1shas},
                       g1shas, _members(), want_regime=POOL_REGIME)
    assert {lab[-4:]: n["s_pp"] for lab, n in g1grp["nodes"].items()} == RU.imported_s(G1_RESULTS)
    cells = {"read": spec.read, "models": {lab: {"arm": "sprt", "seed": int(lab[-4:]), "sha256": sha, "step": None,
                                                 "source": st["nodes"][lab]["source"]} for lab, sha in shas.items()},
             "anchor": {}, "s": {"pool": grp}}
    doc = RU.inputs("x35_r1", cells, spec.params, spec)
    assert doc == _json(STAGE / "sprt_read/read_look1_input.json")
    v = RU.decide("x35_r1", doc, spec.params)
    from main.reads.models import spec_path

    if not all(spec_path(r).is_dir() for r in spec.params["void"]["runs"].values()):
        pytest.skip("the SPRT run dirs are absent (the VOID gate reads them; the owner's box only)")
    gate = RU.void_gate(spec)
    assert RU.void_failures(gate) == {}, "SPRT look 1 was read, so its registered VOID gate passed"
    rec = REC.build(spec, cells, doc, v, _code("eb99ab362"), void=gate)
    want = _json(STAGE / "reads/sprt_look1.json")
    for k in ("arm", "look", "verdict", "r1", "r2", "reasons", "guards"):
        assert rec[k] == want[k], k
    assert rec["regimes"] == {"anchor": {}, "s": {"pool": POOL_REGIME}}
    # the record loads where the X35 stage reads it
    from main.x35 import batch as B

    REC.write(rec, tmp_path / "reads" / "sprt_look1.json")
    assert B.read_record(str(tmp_path), "sprt", 1)["verdict"] == "NULL"


def test_d1_look2_reassembles_from_its_unit_dirs_and_ledger_exactly():
    rows = []
    for st in ("h_a", "h_b", "h_c", "h_d", "h_e"):
        rows += AS.anchor_units(_need("x35_d1_read", st))
    pst = _json(_need("x35_d1_read", "p", "state.json"))
    shas = {lab: v["sha256"] for lab, v in pst["nodes"].items()}
    spec = d1_spec(shas)
    labels = list(spec.anchor["h"].models)
    h = AS.anchor_group(rows, labels=labels, teamsets=tuple(PANELS), unit_seeds=tuple(range(7)))
    assert h["regime_id"] == KAKUNA_UNFAMILIAR_REGIME and not h["inconclusive"] and h["unplayed"] == 0
    p = AS.s_group(_need("x35_d1_read", "p", "ledger"), "x35_d1_p_2026-10-23",
                   {lab: "d1p_look2_b100" if int(lab[-4:]) >= 3504 else "d1p_look1_b100" for lab in shas}, shas,
                   _members(), panel="home72", want_regime=HOME72_REGIME)
    assert p["regime_id"] == HOME72_REGIME and not p["inconclusive"]
    cells = {"read": spec.read, "models": {lab: {"arm": lab[:2].upper(), "seed": int(lab[-4:]), "sha256": sha,
                                                 "step": None, "source": "archive"} for lab, sha in shas.items()},
             "anchor": {"h": h}, "s": {"p": p}}
    doc = RU.inputs("d1", cells, spec.params, spec)
    inp = _json(STAGE / "d1_read/read_look2_input.json")
    assert doc["h"] == inp["h"] and doc["p"] == inp["p"]
    v = RU.decide("d1", doc, spec.params)
    rec = REC.build(spec, cells, doc, v, _code("5e389ddac"))
    want = _json(STAGE / "reads/d1_look2.json")
    for k in ("arm", "look", "verdict", "h", "p", "reasons", "guards"):
        assert rec[k] == want[k], k
    assert rec["regimes"] == {"anchor": {"h": KAKUNA_UNFAMILIAR_REGIME}, "s": {"p": HOME72_REGIME}}


def test_x26c_reassembles_from_its_unit_dirs_and_ledger_exactly():
    rows = AS.anchor_units(_need("x26c_read", "kakuna_pstar")) + AS.anchor_units(_need("x26c_read", "kakuna_480m"))
    committed = [json.loads(ln) for ln in (X26C / "kakuna_units.jsonl").read_text().splitlines()]
    keys = sorted(committed[0])
    assert sorted(({k: r[k] for k in keys} for r in rows), key=lambda r: r["unit"]) == sorted(
        committed, key=lambda r: r["unit"]), "the archived unit dirs ARE the committed rows"
    finals = {f["label"]: f for f in _json(X26C / "finals.json")}
    spec = x26c_spec({"pstar": finals["pstar"]["sha256"]})
    g = AS.anchor_group(rows, labels=["pstar", "x26c_480m"], teamsets=tuple(PANELS), unit_seeds=tuple(range(20)),
                        shas={lab: f["sha256"] for lab, f in finals.items()})
    assert g["regime_id"] == KAKUNA_UNFAMILIAR_REGIME and not g["inconclusive"] and g["unplayed"] == 0
    st = _json(_need("x26c_read", "s", "state.json"))
    res = {n["label"]: n for n in _json(X26C / "s_results.json")["nodes"]}
    pool = AS.s_group(_need("x26c_read", "s", "ledger"), "x26c_s_pool_2026-10-25",
                      {"x26c_480m": "x26c_x26c_480m_pool_b100"}, {"x26c_480m": st["nodes"]["x26c_480m_pool"]["sha256"]},
                      _members(), want_regime=POOL_REGIME)
    home = AS.s_group(_need("x26c_read", "s", "ledger"), "x26c_s_home72_2026-10-25",
                      {lab: f"x26c_{lab}_b100" for lab in ("x26_380m_home72", "x26c_480m_home72")},
                      {lab: st["nodes"][lab]["sha256"] for lab in ("x26_380m_home72", "x26c_480m_home72")},
                      _members(), panel="home72", want_regime=HOME72_REGIME)
    for grp, lab, node in ((pool, "x26c_480m", "x26c_480m_pool"), (home, "x26_380m_home72", "x26_380m_home72"),
                           (home, "x26c_480m_home72", "x26c_480m_home72")):
        n = grp["nodes"][lab]
        for k in ("s_pp", "se_pp", "ci95_pp", "per_member_pp", "per_member_se_pp", "n_pairs", "inconclusive"):
            assert n[k] == res[node][k], (node, k)
        assert res[node]["regime_ids"] == [""], "the shipped results recorded no regime (x26c FINDING 4)"
    assert (pool["regime_id"], home["regime_id"]) == (POOL_REGIME, HOME72_REGIME)
    cells = {"read": spec.read, "models": {lab: {"arm": lab, "seed": 2601, "sha256": f["sha256"], "step": f["step"],
                                                 "source": f["source"]} for lab, f in finals.items()},
             "anchor": {"unfamiliar": g}, "s": {"pool": pool}}
    doc = RU.inputs("anchor_trajectory", cells, spec.params, spec)
    dec = RU.decide("anchor_trajectory", doc, spec.params)
    _x26c_matches(dec)
    rec = REC.build(spec, cells, doc, dec, _code("e05c45eeb"))
    assert rec["verdict"] == "goal (b) not met; FLAT"
    assert rec["regimes"] == {"anchor": {"unfamiliar": KAKUNA_UNFAMILIAR_REGIME}, "s": {"pool": POOL_REGIME}}
