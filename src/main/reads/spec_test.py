"""``main.reads.spec``: a valid spec parses; every malformed field is a named refusal, never a default."""
from __future__ import annotations

import copy
import json

import pytest

from main.reads.spec import SpecError, load, parse

SHA = "a" * 64
RC = "c" * 64


def base() -> dict:
    return {
        "schema": "gen3_read_spec_v1", "read": "demo_read_2026-10-25",
        "models": {"ctl": {"arm": "c0", "seed": 3501, "ref": {"kind": "pinned", "path": "/x/ctl.zip", "sha256": SHA}},
                   "arm": {"arm": "a1", "seed": 3501, "ref": {"kind": "final", "run": "models/rb_a1", "min_step": 10}}},
        "anchor": {"u": {"opponent": "metamon:Kakuna", "teamsets": ["heldout72_hpfix", "forum72", "p72"],
                         "unit_seeds": "0-6", "models": ["ctl", "arm"]}},
        "s": {"pool": {"refpop": "r.json", "refpop_content_sha256": RC, "panel": None, "pairs": 500,
                       "batch_pairs": 100, "family": "demo_s", "models": ["ctl", "arm"]}},
        "decision": {"rule": "anchor_trajectory", "params": {"on": "arm", "off": "ctl", "anchor_group": "u",
                                                             "goal_bar": 0.5, "min_unit_frac": 0.9}}}


def test_a_valid_spec_parses_with_every_field(tmp_path):
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(base()))
    s = load(p)
    assert s.read == "demo_read_2026-10-25" and s.rule == "anchor_trajectory" and len(s.sha256) == 64
    assert s.models["arm"].ref.kind == "final" and s.models["arm"].ref.quiet_s == 300
    assert s.anchor["u"].unit_seeds == tuple(range(7)) and s.s["pool"].request("arm") == "pool_arm_b100"


def _with(path, value):
    d = copy.deepcopy(base())
    cur = d
    for k in path[:-1]:
        cur = cur[k]
    if value is KeyError:
        del cur[path[-1]]
    else:
        cur[path[-1]] = value
    return d


@pytest.mark.parametrize("path,value,needle", [
    (("schema",), "v0", "schema"),
    (("modles",), {}, "unknown field"),
    (("models", "ctl", "ref", "sha256"), "abc", "64 lowercase hex"),
    (("models", "ctl", "ref", "kind"), "latest", "none of"),
    (("models", "arm", "ref", "quiet_s"), 5, ">= 60"),
    (("models", "arm", "ref", "min_step"), KeyError, "min_step"),
    (("models", "arm", "ref", "fallback"), "checkpoint", "unknown field"),   # x26c FINDING 2: no checkpoint fallback
    (("models", "Bad-Label"), {"arm": "a", "seed": 1, "ref": {"kind": "pinned", "path": "p", "sha256": SHA}}, "label"),
    (("models", "ctl", "seed"), True, "not int"),
    (("anchor", "u", "opponent"), "metamon:Elsewhere", "kakuna runner"),
    (("anchor", "u", "teamsets"), ["nowhere"], "team set"),
    (("anchor", "u", "unit_seeds"), "3-1", "bad range"),
    (("anchor", "u", "models"), ["ghost"], "not a declared model"),
    (("s", "pool", "pairs"), 450, "multiple"),
    (("s", "pool", "refpop_content_sha256"), "cef0960e", "64 lowercase hex"),
    (("s", "pool", "regime_id"), "ADFDBEE824C9EB0A", "16 lowercase hex"),
    (("decision", "rule"), "vibes", "none of"),
    (("decision", "params", "anchor_group"), "nope", "not an anchor group"),
    (("pin",), "main", "commit sha"),
])
def test_a_malformed_spec_is_a_named_refusal(path, value, needle):
    with pytest.raises(SpecError, match=needle):
        parse(_with(path, value))


def test_two_s_groups_never_share_a_family():
    d = base()
    d["s"]["home"] = dict(d["s"]["pool"], panel="home72")
    with pytest.raises(SpecError, match="one family per S group"):
        parse(d)


def test_a_spec_with_nothing_to_play_refuses():
    d = base()
    d["anchor"], d["s"] = {}, {}
    with pytest.raises(SpecError, match="nothing to play"):
        parse(d)


def test_a_model_load_option_parses_and_a_bad_one_refuses():
    d = copy.deepcopy(base())
    d["models"]["arm"]["load"] = {"belief_prior_mix": 0.25}
    s = parse(d)
    assert s.models["arm"].load == {"belief_prior_mix": 0.25} and s.models["ctl"].load == {}
    for bad in ({"belief_prior_mix": 0}, {"belief_prior_mix": 1.5}, {"belief_prior_mix": True},
                {"belief_prior_mix": "x"}, {"temperature": 1.0}, []):
        d["models"]["arm"]["load"] = bad
        with pytest.raises(SpecError):
            parse(d)
