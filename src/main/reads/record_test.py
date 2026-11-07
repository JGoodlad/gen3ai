"""``main.reads.record``: a record without a regime id, a resolved model, a ledger digest or a playing commit is
refused; an X35-stage record keeps its decision's keys at the top level."""
from __future__ import annotations

import copy

import pytest

from main.reads import record as REC


VOID_PASS = {"sprt_s3501": {"run": "models/rb_x35_sprt_s3501", "k9b": "PASS", "batch2": {"verdict": "PASS"},
                             "pass": True}}


def _rec(**over):
    r = {"schema": REC.SCHEMA, "read": "demo", "rule": "anchor_trajectory", "spec_sha256": "s" * 64, "prereg": None,
         "verdict": "goal not met; FLAT", "created": "2026-10-25T00:00:00+0000",
         "models": {"a": {"arm": "a", "seed": 1, "sha256": "a" * 64, "step": 1, "source": "/x"}},
         "regimes": {"anchor": {"u": "0123456789abcdef"}, "s": {"pool": "adfdbee824c9eb0a"}},
         "ledgers": {"pool": {"family": "f", "digest": "d" * 16, "rows": 3}},
         "code": {"commit": "e05c45eeb", "dirty": False}, "cells_inconclusive": [], "inputs": {},
         "decision": {"decision": {"verdict": "goal not met; FLAT"}}}
    r.update(over)
    return r


def test_a_complete_record_checks_and_round_trips(tmp_path):
    p = REC.write(_rec(), tmp_path / "record.json")
    assert REC.load(p)["verdict"] == "goal not met; FLAT"
    assert REC.summary_line(_rec()).startswith("demo: goal not met; FLAT (rule anchor_trajectory; regimes "
                                               "anchor.u=0123456789abcdef, s.pool=adfdbee824c9eb0a")


@pytest.mark.parametrize("over,needle", [
    ({"regimes": {"anchor": {"u": ""}, "s": {}}}, "regime id ''"),          # x26c FINDING 4: never again
    ({"regimes": {"anchor": {}, "s": {"pool": None}}}, "regime id None"),
    ({"regimes": {"anchor": {}, "s": {}}}, "no cell group"),
    ({"models": {"a": {"sha256": None}}}, "never resolved"),
    ({"ledgers": {"pool": {"family": "f", "digest": None}}}, "no ledger digest"),
    ({"code": {"commit": ""}}, "no playing commit"),
    ({"verdict": "GOAL MET; CLIMBING"}, "is not the decision's"),
    ({"rule": "vibes"}, "not registered"),
    ({"schema": "v0"}, "schema"),
    ({"cells_inconclusive": ["s.pool: regime x is not the registered y"]}, "cells not conclusive"),
])
def test_an_incomplete_record_is_refused(tmp_path, over, needle):
    r = _rec(**over)
    assert any(needle in p for p in REC.check(r)), REC.check(r)
    with pytest.raises(REC.RecordError):
        REC.write(r, tmp_path / "r.json")


def test_an_x60_record_carries_its_stage_identity_and_check_re_derives_it():
    """AM-X60T-2: rule x60's decision keeps its own shape; the record carries ``arm`` / ``look`` / ``delta_f`` at the top
    level (``Rule.stage_keys``) so ``main.x35.batch.read_record`` loads it, and check re-derives every one of them."""
    from main.reads import rules as RU

    models = {"x60t_s3501": {"arm": "x60t", "seed": 3501, "sha256": "a" * 64, "step": 1, "source": "/x"},
              "c0_s3501": {"arm": "c0", "seed": 3501, "sha256": "b" * 64, "step": 1, "source": "/y"}}
    dec = {"schema": "gen3_read_x60_v1", "look": 1, "sides": {"on": ["x60t_s3501"], "off": ["c0_s3501"]},
           "decision": {"verdict": "CONTINUE", "delta_f_class": "CONTINUE", "stop_at_look1": False}}
    keys = RU.RULES["x60"].stage_keys(models, dec)
    assert keys == {"arm": "X60T", "look": 1, "delta_f": {"final": False, "class": "CONTINUE"}}
    r = _rec(rule="x60", verdict="CONTINUE", models=models, decision=dec, **keys)
    assert REC.check(r) == []
    for k, v in (("arm", "X61P"), ("look", 2), ("delta_f", {"final": True, "class": "CONTINUE"})):
        bad = dict(copy.deepcopy(r), **{k: v})
        assert any(f"top-level {k!r}" in p for p in REC.check(bad)), k
    missing = copy.deepcopy(r)
    missing.pop("arm")
    assert any("top-level 'arm'" in p for p in REC.check(missing))
    two_arms = copy.deepcopy(r)
    two_arms["decision"]["sides"]["on"] = ["x60t_s3501", "c0_s3501"]
    assert any("cannot be derived" in p for p in REC.check(two_arms))


def test_an_x35_record_must_carry_its_decisions_arm_and_look():
    dec = {"arm": "SPRT", "look": 1, "verdict": "NULL", "r1": {"class": "NULL"}}
    r = _rec(rule="x35_r1", verdict="NULL", decision=dec, arm="SPRT", look=1, void=VOID_PASS)
    assert REC.check(r) == []
    bad = copy.deepcopy(r)
    bad["look"] = 2
    assert any("top-level 'look'" in p for p in REC.check(bad))
