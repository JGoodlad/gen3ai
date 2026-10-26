"""``main.reads.imports``: a group imports an EARLIER read's counted cells instead of re-playing them (N1-P FINDING 6:
look 2's spec re-played every look-1 final). The import is sha-checked (the earlier ``spec.json``, the bytes it played)
and regime-matched (its units pool with this state's own only under one regime id); the imported labels are never
planned or played here; the record names the import. Anchor units through an injected unit runner, S rows through the
pipeline's fake heavy slot: no game is played."""
from __future__ import annotations

import hashlib
import json
import time

import pytest

from main.reads import api, layout as LY
from main.reads import runner as RN
from main.reads.imports import ReadImportError
from main.reads.models import sha256_file
from main.reads.pipeline_test import CODE as S_CODE, FakeSlot, _refpop, _spec as _s_spec

CODE = {"commit": "b" * 40, "dirty": False, "root": "/tmp/pin"}


def _cell(opponent_version="kakuna-ckpt34"):
    return {"opponent": "metamon:Kakuna", "opponent_version": opponent_version, "our_regime": "greedy",
            "their_regime": "greedy", "battle_format": "gen3ou", "teamset": "home"}


def _played(wins, cell):
    played: list = []

    def unit(st, u):
        d = st / "cells" / u.name
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps({
            "status": "OK", "n": 20, "wins": wins[u.label], "losses": 20 - wins[u.label], "ties": 0,
            "regime_verified_decisions": True, "peer_clean": True, "team_source_asymmetry": False, "cell": cell}))
        played.append(u.name)
        return 0
    unit.played = played                                     # type: ignore[attr-defined]
    return unit


def _models(tmp_path):
    out = {}
    for lab in ("off", "on"):
        z = tmp_path / "m" / lab / "final_model.zip"
        z.parent.mkdir(parents=True, exist_ok=True)
        (z.parent / "model_config.json").write_text("{}")
        z.write_bytes(lab.encode())
        out[lab] = {"arm": lab, "seed": 1, "ref": {"kind": "pinned", "path": str(z), "sha256": sha256_file(z)}}
    return out


def _anchor_spec(tmp_path, name, imp=None):
    doc = {"schema": "gen3_read_spec_v1", "read": name, "models": _models(tmp_path),
           "anchor": {"u": {"opponent": "metamon:Kakuna", "teamsets": ["heldout72_hpfix"], "unit_seeds": "0-1",
                            "models": ["off", "on"]}},
           "decision": {"rule": "anchor_trajectory", "params": {"on": "on", "off": "off", "anchor_group": "u",
                                                                "goal_bar": 0.5, "min_unit_frac": 0.9}}}
    if imp is not None:
        doc["anchor"]["u"]["import"] = imp
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(doc))
    return p


def _look1(tmp_path, monkeypatch, cell=None):
    monkeypatch.setattr(RN, "checkout_code", lambda root=None: CODE)
    st = tmp_path / "look1"
    RN.plan(_anchor_spec(tmp_path, "look1_read"), st, code=CODE, step_of=lambda _p: 1)
    unit = _played({"off": 8, "on": 12}, cell or _cell())
    assert RN.run(st, "anchor", time.time() + 3600, run_unit=unit, sleep=lambda s: None) == 0
    return st, hashlib.sha256(LY.spec_file(st).read_bytes()).hexdigest()


def test_an_imported_label_is_never_played_and_its_counted_units_enter_the_group(tmp_path, monkeypatch):
    st1, sha1 = _look1(tmp_path, monkeypatch)
    imp = {"state": str(st1), "spec_sha256": sha1, "group": "u", "models": ["off"]}
    st2 = tmp_path / "look2"
    RN.plan(_anchor_spec(tmp_path, "look2_read", imp), st2, code=CODE, step_of=lambda _p: 1)
    assert not (LY.anchor_state(st2, "u", "off") / "plan.json").exists(), "an imported label is never planned"
    unit = _played({"off": 0, "on": 13}, _cell())
    assert RN.run(st2, "anchor", time.time() + 3600, run_unit=unit, sleep=lambda s: None) == 0
    assert unit.played == ["on__heldout72_hpfix__00", "on__heldout72_hpfix__01"]
    assert RN.status(st2)["anchor"]["u"]["off"].startswith("imported from")
    rec = api.record(st2)
    grp = json.loads(LY.cells_file(st2).read_text())["anchor"]["u"]
    assert grp["cells"]["off"]["heldout72_hpfix"] == {"wins": 16, "ties": 0, "n": 40, "units": 2, "planned": 2,
                                                       "skipped": []}, "look 1's counted units, not re-played"
    assert grp["cells"]["on"]["heldout72_hpfix"]["wins"] == 26
    assert rec["cells_inconclusive"] == [] and rec["regimes"]["anchor"]["u"] == grp["regime_id"]
    assert rec["imports"]["anchor.u"]["spec_sha256"] == sha1 and rec["imports"]["anchor.u"]["rows"] == 2
    assert rec["imports"]["anchor.u"]["code"]["commit"] == CODE["commit"]


def test_an_import_is_refused_on_another_spec_other_bytes_or_another_cell(tmp_path, monkeypatch):
    st1, sha1 = _look1(tmp_path, monkeypatch)
    with pytest.raises(ReadImportError, match="is not the registered"):
        RN.plan(_anchor_spec(tmp_path, "look2_read", {"state": str(st1), "spec_sha256": "0" * 64, "group": "u",
                                                      "models": ["off"]}), tmp_path / "s_a", code=CODE, step_of=lambda _p: 1)
    with pytest.raises(ReadImportError, match="no anchor group 'v'"):
        RN.plan(_anchor_spec(tmp_path, "look2_read", {"state": str(st1), "spec_sha256": sha1, "group": "v",
                                                      "models": ["off"]}), tmp_path / "s_b", code=CODE, step_of=lambda _p: 1)
    # the earlier state played OTHER bytes than this read resolves: refused at assemble
    imp = {"state": str(st1), "spec_sha256": sha1, "group": "u", "models": ["off"]}
    st2 = tmp_path / "look2"
    RN.plan(_anchor_spec(tmp_path, "look2_read", imp), st2, code=CODE, step_of=lambda _p: 1)
    m = json.loads(LY.models_file(st1).read_text())
    m["off"]["sha256"] = "f" * 64
    LY.models_file(st1).write_text(json.dumps(m))
    with pytest.raises(ReadImportError, match="the earlier state played sha ffffffffffff"):
        api.assemble(st2)


def test_imported_units_of_another_regime_make_the_group_inconclusive(tmp_path, monkeypatch):
    st1, sha1 = _look1(tmp_path, monkeypatch, cell=_cell("kakuna-ckpt28"))       # look 1 faced another opponent
    imp = {"state": str(st1), "spec_sha256": sha1, "group": "u", "models": ["off"]}
    st2 = tmp_path / "look2"
    RN.plan(_anchor_spec(tmp_path, "look2_read", imp), st2, code=CODE, step_of=lambda _p: 1)
    RN.run(st2, "anchor", time.time() + 3600, run_unit=_played({"on": 13}, _cell()), sleep=lambda s: None)
    cells = api.assemble(st2)
    assert any("span 2 regimes" in w for w in cells["anchor"]["u"]["inconclusive"])
    assert cells["anchor"]["u"]["regime_id"] is None
    with pytest.raises(api.RecordError, match="cells not conclusive"):
        api.record(st2)


def test_an_s_group_imports_the_earlier_reads_nodes_and_plays_only_its_own(tmp_path, no_git):
    refpop, rdoc = _refpop(tmp_path)
    spec1 = _s_spec(tmp_path, refpop, rdoc["content_sha256"])
    st1 = tmp_path / "look1"
    RN.plan(spec1, st1, code=S_CODE, step_of=lambda _p: 300_000_000)
    assert RN.run(st1, "s", time.time() + 7200, run_s=FakeSlot(st1, rdoc), sleep=lambda s: None) == 0
    look1 = api.record(st1)
    doc = json.loads(spec1.read_text())
    c0 = sorted(lab for lab in doc["models"] if lab.startswith("c0_"))
    doc["read"] = "demo_s_look2_2026-10-26"
    doc["s"]["pool"]["family"] = "demo_s_look2_2026-10-26"
    doc["s"]["pool"]["import"] = {"state": str(st1), "group": "pool", "models": c0,
                                  "spec_sha256": hashlib.sha256(LY.spec_file(st1).read_bytes()).hexdigest()}
    spec2 = tmp_path / "look2_spec.json"
    spec2.write_text(json.dumps(doc))
    st2 = tmp_path / "look2"
    RN.plan(spec2, st2, code=S_CODE, step_of=lambda _p: 300_000_000)
    slot = FakeSlot(st2, rdoc)
    assert RN.run(st2, "s", time.time() + 7200, run_s=slot, sleep=lambda s: None) == 0
    engines = {a[i + 1].split(":")[0] for a in slot.calls for i, x in enumerate(a) if x == "--engine"}
    assert engines == {"sprt_s3501", "sprt_s3502", "sprt_s3503"}, "the imported C0 finals are not re-played"
    rec = api.record(st2)
    assert rec["cells_inconclusive"] == [] and rec["regimes"]["s"]["pool"] == look1["regimes"]["s"]["pool"]
    assert rec["r1"] == look1["r1"], "the same rows, the same decision"
    assert rec["imports"]["s.pool"]["family"] == "demo_s_2026-10-25"
    assert rec["imports"]["s.pool"]["ledger_digest"] == look1["ledgers"]["pool"]["digest"]
    # the same cell is required: an earlier group on another panel is refused before any play
    doc["s"]["pool"]["panel"] = "home72"
    spec3 = tmp_path / "look3_spec.json"
    spec3.write_text(json.dumps(doc))
    with pytest.raises(ReadImportError, match="panel None is not 'home72'"):
        RN.plan(spec3, tmp_path / "look3", code=S_CODE, step_of=lambda _p: 300_000_000)


def test_an_earlier_spec_is_imported_even_if_todays_rule_would_refuse_its_params(tmp_path, monkeypatch):
    """An earlier read's spec was valid under its own day's rules (e.g. an SPRT spec from before the VOID gate was
    required): the import checks the CELL, never re-judges the earlier decision's params."""
    st1, _sha = _look1(tmp_path, monkeypatch)
    doc = json.loads(LY.spec_file(st1).read_text())
    doc["decision"]["params"]["retired_param"] = 1          # a key today's rule refuses as unknown
    LY.spec_file(st1).write_text(json.dumps(doc))
    sha1 = hashlib.sha256(LY.spec_file(st1).read_bytes()).hexdigest()
    imp = {"state": str(st1), "spec_sha256": sha1, "group": "u", "models": ["off"]}
    RN.plan(_anchor_spec(tmp_path, "look2_read", imp), tmp_path / "look2", code=CODE, step_of=lambda _p: 1)
