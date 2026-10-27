"""``main.reads.repin``: a running read adopts another playing commit ONLY when the tracked play path is byte-identical.
A refusal on any play-path difference writes nothing; a scheduling-only difference is accepted and APPENDED to the
state (``repins.jsonl``), ``code.json`` stays as ``plan`` wrote it (a supervisor of older code keeps working), the pin
guard accepts the new commit and still refuses any other, and the record lists the repin."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from main.reads import api, layout as LY
from main.reads import playpath as PP
from main.reads import repin as RP
from main.reads import runner as RN
from main.reads.playpath_test import base_of, commit_on, mini_repo  # noqa: F401  (the fixture)
from main.reads.record_test import _rec
from main.reads.runner_test import CODE, _spec, _step, _zip

SCHEDULING_ONLY = {"src/main/reads/runner.py": "SCHED = 2\n", "designs/tools/reads.md": "# new\n",
                   "src/main/reads/anchor_pool.py": "POOL = 2\n", "src/main/anchors/cli_test.py": "TEST = 2\n"}


@pytest.fixture
def planned(tmp_path, run_archive_env, mini_repo, monkeypatch):      # noqa: F811
    """A planned read whose ``code.json`` names the mini repo's base commit; the default repo is the mini repo."""
    base = base_of(mini_repo)
    spec = _spec(tmp_path, run_archive_env)
    _zip(run_archive_env / "rb_new", age=3600)
    state = tmp_path / "state"
    RN.plan(spec, state, code=dict(CODE, commit=base), step_of=_step)
    monkeypatch.setattr(PP, "default_repo", lambda: mini_repo)
    return state, mini_repo, base


def _lines(state: Path):
    p = LY.repins_file(state)
    return p.read_text().splitlines() if p.exists() else []


@pytest.mark.parametrize("name,files,needle", [
    ("the Rust bridge", {"src/rust_sim/src/bridge.rs": "// changed\n"}, "changed src/rust_sim/src/bridge.rs"),
    ("a team file", {"data/teams/a.txt": "team b\n"}, "changed data/teams/a.txt"),
    ("a module reached only by a lazy import", {"src/utils/late.py": "x = 2\n"}, "changed src/utils/late.py"),
    ("the unit launcher", {"src/main/x35/kakuna.py": "def lazy():\n    return 2\n"}, "changed src/main/x35/kakuna.py"),
    ("mem_cap", {"scripts/ops/mem_cap.sh": "#!/bin/sh\n# x\n"}, "changed scripts/ops/mem_cap.sh"),
    ("a new file under data/", {"data/teams/b.txt": "b\n"}, "added data/teams/b.txt"),
    ("a play-path file deleted", {"src/utils/sibling.py": None}, "removed src/utils/sibling.py"),
    ("an import that widens the closure", {"src/main/x35/kakuna.py": "from utils.unrelated import U\n"},
     "added src/utils/unrelated.py"),
])
def test_a_repin_is_refused_on_any_play_path_difference_and_writes_nothing(planned, name, files, needle):
    state, repo, base = planned
    # the scheduling change rides along: it must not excuse the play-path one
    new = commit_on(repo, base, dict(SCHEDULING_ONLY, **files))
    before = LY.code_file(state).read_bytes()
    with pytest.raises(RP.RepinError, match=r"play path differs") as e:
        RP.repin(state, new, by="reader", why="scheduling fix", repo=repo)
    assert needle in str(e.value), (name, str(e.value))
    assert base[:12] in str(e.value) and new[:12] in str(e.value)
    assert _lines(state) == [] and LY.code_file(state).read_bytes() == before, "a refusal writes nothing"
    with pytest.raises(RN.RunError, match="changed under the read"):
        RN._code_holds(state, dict(CODE, commit=new))


def test_a_scheduling_only_difference_is_accepted_and_appended(planned):
    state, repo, base = planned
    new = commit_on(repo, base, SCHEDULING_ONLY)
    code_before = LY.code_file(state).read_bytes()
    now = time.time()
    rec = RP.repin(state, new, by="N1-P reader", why="--concurrency 2 on the anchor lane", repo=repo, now=now)
    want = PP.digest(PP.play_files(new, repo))
    assert (rec["schema"], rec["from_commit"], rec["to_commit"]) == (RP.SCHEMA, base, new)
    assert rec["playpath_digest"] == want == PP.digest(PP.play_files(base, repo)), "byte-identical play path"
    assert rec["playpath_files"] == len(PP.play_files(new, repo)) and rec["playpath_schema"] == PP.SCHEMA
    assert (rec["by"], rec["why"], rec["t"]) == ("N1-P reader", "--concurrency 2 on the anchor lane", now)
    assert rec["time"] and rec["root"] == str(repo)
    assert [json.loads(x) for x in _lines(state)] == [rec]
    assert LY.code_file(state).read_bytes() == code_before, \
        "code.json is untouched: a supervisor of older code, still on the old checkout, keeps working"
    assert json.loads(code_before)["commit"] == base
    assert RP.accepted_commits(state) == [base, new]


def test_the_pin_guard_accepts_the_plan_commit_and_a_repinned_one_and_nothing_else(planned):
    state, repo, base = planned
    new = commit_on(repo, base, SCHEDULING_ONLY)
    other = commit_on(repo, base, {"src/main/reads/runner.py": "SCHED = 3\n"})           # play-path identical too
    play_other = commit_on(repo, base, {"data/teams/a.txt": "team b\n"})
    with pytest.raises(RN.RunError, match="changed under the read"):
        RN._code_holds(state, dict(CODE, commit=new))                    # before the repin
    RP.repin(state, new, by="r", why="w", repo=repo)
    RN._code_holds(state, dict(CODE, commit=base))
    RN._code_holds(state, dict(CODE, commit=new))
    for bad in (other, play_other):                                       # not repinned: still refused
        with pytest.raises(RN.RunError, match="changed under the read"):
            RN._code_holds(state, dict(CODE, commit=bad))
    with pytest.raises(RN.RunError, match="changed under the read"):      # a repin does not excuse a dirty tree
        RN._code_holds(state, dict(CODE, commit=new, dirty=True))
    with pytest.raises(RN.RunError, match=f"repinned to {new[:12]}"):
        RN._code_holds(state, dict(CODE, commit=other))


def test_repins_append_and_never_overwrite_and_a_repeat_is_a_no_op(planned):
    state, repo, base = planned
    first = commit_on(repo, base, SCHEDULING_ONLY)
    second = commit_on(repo, base, {"src/main/reads/runner.py": "SCHED = 3\n"})
    RP.repin(state, first, by="a", why="one", repo=repo)
    line1 = _lines(state)[0]
    again = RP.repin(state, first, by="b", why="a second time", repo=repo)
    assert again["already"] is True and again["by"] == "a", "the first record stands"
    assert RP.repin(state, base, by="b", why="the plan commit", repo=repo)["to_commit"] == base
    assert len(_lines(state)) == 1
    RP.repin(state, second, by="c", why="two", repo=repo)
    lines = _lines(state)
    assert len(lines) == 2 and lines[0] == line1, "the first line is byte-for-byte what it was"
    assert [json.loads(x)["to_commit"] for x in lines] == [first, second]
    assert [json.loads(x)["from_commit"] for x in lines] == [base, base], "always compared with code.json's commit"
    assert RP.accepted_commits(state) == [base, first, second]


def test_the_refusals_before_the_play_path_is_looked_at(planned, tmp_path):
    state, repo, base = planned
    new = commit_on(repo, base, SCHEDULING_ONLY)
    for by, why, needle in (("", "w", "--by"), ("  ", "w", "--by"), ("r", "", "--why")):
        with pytest.raises(RP.RepinError, match=needle):
            RP.repin(state, new, by=by, why=why, repo=repo)
    with pytest.raises(RP.RepinError, match="names no commit"):
        RP.repin(state, "nosuchcommit", by="r", why="w", repo=repo)
    with pytest.raises(RP.RepinError, match="no plan"):
        RP.repin(tmp_path / "unplanned", new, by="r", why="w", repo=repo)
    spec = json.loads(LY.spec_file(state).read_text())
    spec["pin"] = "e05c45eeb"
    LY.spec_file(state).write_text(json.dumps(spec))
    with pytest.raises(RP.RepinError, match="registers pin e05c45eeb"):
        RP.repin(state, new, by="r", why="w", repo=repo)
    assert _lines(state) == []


def test_a_damaged_repins_file_stops_the_guard_instead_of_being_skipped(planned):
    state, repo, base = planned
    LY.repins_file(state).write_text("{not json\n")
    with pytest.raises(RP.RepinError, match="not JSON"):
        RP.load_repins(state)
    with pytest.raises(RN.RunError, match="cannot be checked"):
        RN._code_holds(state, dict(CODE, commit=base))
    LY.repins_file(state).write_text(json.dumps({"to_commit": "x"}) + "\n")
    with pytest.raises(RP.RepinError, match="lacks"):
        RP.load_repins(state)


def test_status_and_the_record_list_the_repins(planned):
    state, repo, base = planned
    new = commit_on(repo, base, SCHEDULING_ONLY)
    assert RN.status(state)["repins"] == []
    rec = RP.repin(state, new, by="N1-P reader", why="--concurrency 2", repo=repo)
    assert RN.status(state)["repins"] == [{"to_commit": new, "by": "N1-P reader", "time": rec["time"]}]
    assert api.load_repins(state) == [rec]
    from main.reads import record as REC

    assert REC.check(_rec(repins=[rec])) == [], "a record carrying a repin checks"
    assert REC.check(_rec()) == [], "and one from before repins existed still does"
    bad = dict(rec, why="")
    assert any("repins[0]" in p and "why" in p for p in REC.check(_rec(repins=[bad])))


def test_the_cli_repins_with_exit_0_and_refuses_with_exit_2(planned, capsys):
    from main.reads.cli import main

    state, repo, base = planned
    ok = commit_on(repo, base, SCHEDULING_ONLY)
    bad = commit_on(repo, base, {"data/teams/a.txt": "team b\n"})
    argv = ["repin", "--state", str(state), "--by", "r", "--why", "w"]
    assert main(argv + ["--to", bad]) == 2
    assert "REFUSED: RepinError" in capsys.readouterr().err and _lines(state) == []
    assert main(argv + ["--to", ok]) == 0
    assert json.loads(capsys.readouterr().out)["to_commit"] == ok
    assert main(["status", "--state", str(state)]) == 0
    assert json.loads(capsys.readouterr().out)["repins"][0]["to_commit"] == ok


# ---- the history rewrite: the recorded commits of a state, a spec and a repin resolve through `utils.git_sha` ----------
def test_a_state_planned_at_an_old_sha_still_holds_at_its_new_sha_and_a_repin_to_it_is_already(planned, tmp_path,
                                                                                               monkeypatch):
    from utils.git_sha_testkit import fake_old, use_map

    state, repo, base = planned
    old = fake_old("a")
    use_map(monkeypatch, tmp_path, {old: base})
    cf = LY.code_file(state)
    cf.write_text(json.dumps(dict(json.loads(cf.read_text()), commit=old)))       # code.json as a pre-rewrite plan wrote it
    RN._code_holds(state, dict(CODE, commit=base))                                 # the playing checkout is the NEW sha
    with pytest.raises(RN.RunError, match="changed under the read"):
        RN._code_holds(state, dict(CODE, commit=fake_old("c")))
    assert RP.repin(state, base, by="reader", why="same commit", repo=repo)["already"] is True
    assert _lines(state) == []
    new = commit_on(repo, base, SCHEDULING_ONLY)
    rec = RP.repin(state, new, by="reader", why="scheduling", repo=repo)           # `old` resolves as the from-commit
    assert (rec["from_commit"], rec["to_commit"]) == (base, new)


def test_a_spec_pin_registered_at_an_old_sha_matches_the_playing_checkout_at_the_new_sha(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from utils.git_sha_testkit import fake_old, use_map

    new = "ab" * 20
    use_map(monkeypatch, tmp_path, {fake_old("a"): new})
    spec = SimpleNamespace(pin=fake_old("a")[:12])
    RN.check_code(spec, {"commit": new, "dirty": False, "root": str(tmp_path)})
    with pytest.raises(RN.RunError, match="not the spec's pin"):
        RN.check_code(spec, {"commit": "cd" * 20, "dirty": False, "root": str(tmp_path)})
