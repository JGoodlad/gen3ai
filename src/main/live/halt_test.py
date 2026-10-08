"""T28 — the live-play HALT marker: recorded, refused past, cleared only by a qualifying commit.

Each test names its own marker through ``$GEN3AI_LIVE_HALT_FILE``; the default path is REFUSED under pytest
(``HaltPathUnsealed``) so a planted bad line can never halt the owner's real play.
"""
import json
import subprocess

import pytest

from main.exit_codes import TrainExitCode, exit_code_for
from main.live import halt as H


@pytest.fixture
def marker(tmp_path, monkeypatch):
    p = tmp_path / "state" / "live_play_halt.json"
    monkeypatch.setenv(H.HALT_ENV, str(p))
    return p


def test_the_default_marker_is_sealed_under_pytest(monkeypatch):
    monkeypatch.delenv(H.HALT_ENV, raising=False)
    with pytest.raises(H.HaltPathUnsealed):
        H.halt_path(write=True)
    with pytest.raises(H.HaltPathUnsealed):
        H.record_halt(reason="r", entry_point="t")
    assert not H.halt_path().exists()  # a READ sees no halt, never the owner's real file
    H.refuse_if_halted("t")


def test_the_exit_code_is_distinct_and_mapped_by_name():
    assert int(TrainExitCode.FATAL_LIVE_PARSE) == 7
    assert len({int(c) for c in TrainExitCode}) == len(TrainExitCode)
    assert exit_code_for(H.LiveParseHalt("x")) == 7
    try:
        try:
            raise H.LiveParseHalt("inner")
        except H.LiveParseHalt as e:
            raise RuntimeError("wrapped") from e
    except RuntimeError as outer:
        assert exit_code_for(outer) == 7


def test_no_marker_starts_and_a_marker_refuses(marker):
    H.refuse_if_halted("t")  # nothing recorded: starts
    H.record_halt(reason="unknown keyword |zzz|", entry_point="t", battle_id="battle-gen3ou-1",
                  offending_lines=["|zzz|p1a: X"], stream=["|t:|1", "|zzz|p1a: X"])
    with pytest.raises(H.HaltActive, match="HALTED"):
        H.refuse_if_halted("main.play")
    rec = json.loads(marker.read_text())
    assert rec["schema"] == H.SCHEMA
    assert rec["battle_id"] == "battle-gen3ou-1"
    assert rec["offending_lines"] == ["|zzz|p1a: X"]
    assert rec["stream_tail"][-1] == "|zzz|p1a: X"
    assert rec["commit"]


def test_a_second_halt_keeps_the_first(marker):
    H.record_halt(reason="first", entry_point="t")
    H.record_halt(reason="second", entry_point="t")
    rec = H.read_halt()
    assert rec["reason"] == "first" and rec["later"][0]["reason"] == "second"


def test_an_unreadable_marker_still_refuses(marker):
    marker.parent.mkdir(parents=True)
    marker.write_text("{not json")
    with pytest.raises(H.HaltActive):
        H.refuse_if_halted("t")


def test_exit_on_halt_records_and_exits_FATAL_LIVE_PARSE(marker):
    exc = H.LiveParseHalt("encoder raised", battle_id="b1", offending_lines=["|-activate|p2a: Y|move: Z"])
    with pytest.raises(SystemExit) as ei:
        H.exit_on_halt(exc, entry_point="t")
    assert ei.value.code == 7
    assert H.read_halt()["reason"] == "encoder raised"


def _repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    g = lambda *a: subprocess.run(["git", *a], cwd=r, check=True, capture_output=True, text=True)  # noqa: E731
    g("init", "-q")
    g("config", "user.email", "t@t")
    g("config", "user.name", "t")
    (r / "a.py").write_text("x = 1\n")
    g("add", ".")
    g("commit", "-qm", "code only")
    code_only = g("rev-parse", "HEAD").stdout.strip()
    (r / "a_test.py").write_text("def test_x(): pass\n")
    g("add", ".")
    g("commit", "-qm", "the fix + its regression test")
    with_test = g("rev-parse", "HEAD").stdout.strip()
    g("checkout", "-qb", "side", code_only)
    (r / "b_test.py").write_text("def test_y(): pass\n")
    g("add", ".")
    g("commit", "-qm", "off the line")
    off_line = g("rev-parse", "HEAD").stdout.strip()
    g("checkout", "-q", "-")
    return r, code_only, with_test, off_line


def test_the_clear_needs_an_ancestor_commit_that_touches_a_test(marker, tmp_path):
    r, code_only, with_test, off_line = _repo(tmp_path)
    H.record_halt(reason="r", entry_point="t")
    with pytest.raises(H.ClearRefused, match="no test file"):
        H.clear_halt(code_only, cwd=str(r))
    with pytest.raises(H.ClearRefused, match="not an ancestor"):
        H.clear_halt(off_line, cwd=str(r))
    with pytest.raises(H.ClearRefused, match="not a commit"):
        H.clear_halt("deadbeef" * 5, cwd=str(r))
    assert H.read_halt() is not None  # every refusal leaves the halt in place
    entry = H.clear_halt(with_test, note="root cause", cwd=str(r))
    assert entry["fixed_by"]["commit"] == with_test and entry["fixed_by"]["tests"] == ["a_test.py"]
    assert H.read_halt() is None
    H.refuse_if_halted("t")
    hist = [json.loads(l) for l in H.history_path().read_text().splitlines()]
    assert hist[-1]["fixed_by"]["commit"] == with_test and hist[-1]["halt"]["reason"] == "r"


def test_the_clear_refuses_when_nothing_is_halted(marker, tmp_path):
    with pytest.raises(H.ClearRefused, match="no active halt"):
        H.clear_halt("HEAD")


@pytest.mark.parametrize("path,ok", [("src/main/live/halt_test.py", True),
                                     ("src/rust_sim/tests/live_reader_test.rs", True),
                                     ("src/main/live/halt.py", False),
                                     ("src/rust_sim/src/bin/live_reader.rs", False)])
def test_is_test_path(path, ok):
    assert H.is_test_path(path) is ok


def test_the_cli_status_reports_the_halt(marker, capsys):
    assert H.main(["status"]) == 0
    H.record_halt(reason="r", entry_point="t")
    assert H.main(["status"]) == 1
    assert "HALTED" in capsys.readouterr().out
