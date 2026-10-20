"""The per-process verdict memo of ``validate_teams_locally`` (backlog: ``Teambuilder.__init__`` spawned a Node
validator per construction). Node is replaced by a fake ``Popen`` that counts spawns and records what it was asked."""
import json
import subprocess
import types

import pytest

from utils.bridge import team_validator as tv


def _patch_popen(monkeypatch, popen):
    """Swap ONLY the module's ``subprocess.Popen`` (a global patch would also break ``subprocess.run``)."""
    monkeypatch.setattr(tv, "subprocess", types.SimpleNamespace(
        Popen=popen, run=subprocess.run, SubprocessError=subprocess.SubprocessError, PIPE=subprocess.PIPE))


def _forget():
    getattr(tv, "_MEMO", {}).clear()


class _FakeNode:
    """Stands in for ``subprocess.Popen(['node', validate_team.js])``."""
    spawns: list = []

    def __init__(self, argv, **kw):
        self.argv = argv

    def communicate(self, input=None):
        req = json.loads(input)
        _FakeNode.spawns.append(list(req["team"]))
        return json.dumps([{"valid": "BAD" not in t, "errors": [] if "BAD" not in t else ["illegal"]}
                           for t in req["team"]]), ""


@pytest.fixture(autouse=True)
def fake_node(monkeypatch):
    _FakeNode.spawns = []
    _patch_popen(monkeypatch, _FakeNode)
    _forget()
    yield _FakeNode
    _forget()


def test_a_repeat_call_spawns_no_second_node(fake_node):
    first = tv.validate_teams_locally("gen3ou", ["A", "B BAD"])
    second = tv.validate_teams_locally("gen3ou", ["A", "B BAD"])
    assert first == second == [{"valid": True, "errors": []}, {"valid": False, "errors": ["illegal"]}]
    assert fake_node.spawns == [["A", "B BAD"]]


def test_only_the_misses_go_to_node_and_the_order_is_kept(fake_node):
    tv.validate_teams_locally("gen3ou", ["A"])
    out = tv.validate_teams_locally("gen3ou", ["C BAD", "A", "C BAD", "D"])
    assert [r["valid"] for r in out] == [False, True, False, True]
    assert fake_node.spawns == [["A"], ["C BAD", "D"]]      # the duplicate is validated once


def test_the_format_is_part_of_the_key(fake_node):
    tv.validate_teams_locally("gen3ou", ["A"])
    tv.validate_teams_locally("gen3uu", ["A"])
    assert len(fake_node.spawns) == 2


def test_a_new_validator_version_invalidates_the_memo(fake_node, monkeypatch):
    tv.validate_teams_locally("gen3ou", ["A"])
    monkeypatch.setattr(tv, "validator_version", lambda: "another-showdown-commit")
    tv.validate_teams_locally("gen3ou", ["A"])
    assert len(fake_node.spawns) == 2


def test_a_caller_editing_a_result_cannot_poison_the_memo(fake_node):
    out = tv.validate_teams_locally("gen3ou", ["A"])
    out[0]["valid"] = False
    out[0]["errors"].append("scribble")
    assert tv.validate_teams_locally("gen3ou", ["A"]) == [{"valid": True, "errors": []}]


def test_an_infrastructure_failure_is_never_memoised(monkeypatch):
    class _Dead:
        def __init__(self, *a, **k): pass
        def communicate(self, input=None): return "", "Error: Cannot find module"
    _patch_popen(monkeypatch, _Dead)
    bad = tv.validate_teams_locally("gen3ou", ["A"])
    assert bad[0]["valid"] is False and "Node execution error" in bad[0]["errors"][0]
    _FakeNode.spawns = []
    _patch_popen(monkeypatch, _FakeNode)
    assert tv.validate_teams_locally("gen3ou", ["A"]) == [{"valid": True, "errors": []}]   # asked again, answered
    assert _FakeNode.spawns == [["A"]]


def test_a_whole_request_error_becomes_one_invalid_result_per_team_and_is_not_memoised(monkeypatch):
    class _Odd:
        def __init__(self, *a, **k): pass
        def communicate(self, input=None): return json.dumps({"valid": False, "errors": ["boom"]}), ""
    _patch_popen(monkeypatch, _Odd)
    out = tv.validate_teams_locally("gen3ou", ["A", "B"])
    assert [r["valid"] for r in out] == [False, False]
    assert not getattr(tv, "_MEMO", None)


def test_an_empty_list_spawns_nothing(fake_node):
    assert tv.validate_teams_locally("gen3ou", []) == []
    assert fake_node.spawns == []


def test_the_real_validator_version_names_the_submodule_commit():
    tv.validator_version.cache_clear()
    v = tv.validator_version()
    assert v.split("|")[0] != "no-commit" and len(v.split("|")[0]) == 40
