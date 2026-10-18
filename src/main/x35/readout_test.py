"""The X35 readout (``main.x35.readout``) on CONSTRUCTED inputs: the whole self-check passes check by check, and the
edges the self-check does not name. Deterministic: the only draws are the self-check's private seeded generator."""
from __future__ import annotations

import json

import pytest

from main.x35 import readout as R
from main.x35 import selfcheck as SC


@pytest.mark.parametrize("check", SC.CHECKS, ids=lambda c: c.__name__)
def test_every_self_check_passes(check):
    name, ok, detail = check()
    assert ok, f"{name}: {detail}"


def test_the_self_check_entry_point_exits_zero(capsys):
    assert R._main(["self-check"]) == 0
    assert f"{len(SC.CHECKS)} / {len(SC.CHECKS)} PASS" in capsys.readouterr().out


def test_an_unknown_arm_is_refused():
    r1 = R.decide_r1(1, {}, {})
    with pytest.raises(ValueError, match="not one of"):
        R.decide("C0", r1, R.decide_r2(None), [])


def test_a_guard_bound_on_its_boundary_is_not_a_fail_and_is_recorded():
    # an additive 3 x 3 matrix shifted so the 95 % upper bound sits on −2.0 (within float error)
    base = [[50.0 + a - b for b in (0.5, -0.5, 0.0)] for a in (1.0, -1.0, 0.0)]
    hi = R.guard_f(base, 1).detail["ci95_pp"][1]
    at = [[x + (-2.0 - hi) for x in row] for row in base]
    g = R.guard_f(at, 1)
    assert g.status == R.PASS and g.near
    past = [[x - 1e-6 for x in row] for row in at]
    assert R.guard_f(past, 1).status == R.FAIL


def test_the_dose_confound_is_reported_never_decides():
    doc = {"arm": "A1", "look": 1,
           "r1": {"on": {"1": 60.0, "2": 61.0, "3": 59.0}, "off": {"1": 50.0, "2": 51.0, "3": 49.0}},
           "dose": {"on": [2.0], "off": [1.0]}}
    v = R.read(doc)
    assert v.reported["dose"]["confound"] is True
    assert v.verdict == R.INCONCLUSIVE  # GAIN, but no guard was read: the dose did not decide it
    assert "INCONCLUSIVE" in v.reasons[0]


def test_a_malformed_r2_input_is_inconclusive_not_a_crash():
    doc = {"arm": "A3", "look": 1,
           "r1": {"on": {"1": 50.1, "2": 50.2, "3": 50.0}, "off": {"1": 50.0, "2": 50.1, "3": 49.9}},
           "r2": {"on": [500, 400], "off": [200, 400]}}
    v = R.read(doc)
    assert v.verdict == R.INCONCLUSIVE and any("R2 input" in r for r in v.reasons)


def test_the_decide_cli_writes_the_verdict_json(tmp_path, capsys):
    p = tmp_path / "read.json"
    p.write_text(json.dumps({"arm": "A2", "look": 1, "r1": {"on": {"1": 40.0, "2": 41.0, "3": 39.0},
                                                           "off": {"1": 50.0, "2": 51.0, "3": 49.0}}}))
    assert R._main(["decide", str(p)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] == R.LOSS and out["r1"]["class"] == R.LOSS
