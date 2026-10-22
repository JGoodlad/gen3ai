"""``slow_tier_prune``: rows whose test is no longer collected as `slow` go, every other row is byte-for-byte kept,
and a failed or unparseable collection touches nothing. Fakes only (a planted status file, a canned collector)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils import slow_tier_prune as P
from utils import slow_tier_status as ST

KEEP_PASS = "src/a_test.py::test_kept"
KEEP_FAIL = "src/a_test.py::test_kept_but_red[x]"
GONE_PASS = "src/old_test.py::test_deleted"
GONE_FAIL = "src/a_test.py::test_renamed_away"


def row(status: str) -> dict:
    return {"status": status, "commit": "c" * 40, "at": "2026-10-01T00:00:00Z", "duration_s": 1.0, "detail": ""}


@pytest.fixture
def status_file(tmp_path: Path) -> Path:
    p = tmp_path / "slow_tier_status.json"
    doc = {"schema": ST.SCHEMA, "last_written": "2026-10-01T00:00:00Z",
           "tests": {n: row(s) for n, s in [(KEEP_PASS, "pass"), (KEEP_FAIL, "fail"), (GONE_PASS, "pass"), (GONE_FAIL, "fail")]}}
    p.write_text(json.dumps(doc, indent=1) + "\n")
    return p


COLLECTED = {KEEP_PASS, KEEP_FAIL, "src/new_test.py::test_brand_new"}


def test_prunable_is_exactly_the_rows_outside_the_collected_set(status_file: Path) -> None:
    doc = json.loads(status_file.read_text())
    assert ST.prunable(doc, COLLECTED) == sorted([GONE_PASS, GONE_FAIL])
    assert ST.prunable(doc, set(doc["tests"])) == []


def test_check_mode_lists_and_touches_nothing(status_file: Path, capsys) -> None:
    before = status_file.read_bytes()
    rc = P.main([], collect=lambda: set(COLLECTED), path=status_file)
    out = capsys.readouterr().out
    assert rc == 1 and status_file.read_bytes() == before
    assert "WOULD prune 2 row(s)" in out and "src/old_test.py" in out and "fail" in out


def test_write_removes_only_the_gone_rows_and_is_then_idempotent(status_file: Path, capsys) -> None:
    kept_before = {n: r for n, r in json.loads(status_file.read_text())["tests"].items() if n in COLLECTED}
    assert P.main(["--write"], collect=lambda: set(COLLECTED), path=status_file) == 0
    doc = json.loads(status_file.read_text())
    assert doc["tests"] == kept_before, "kept rows are untouched, a recorded red included"
    assert doc["schema"] == ST.SCHEMA
    assert "pruned 2 row(s)" in capsys.readouterr().out
    assert P.main([], collect=lambda: set(COLLECTED), path=status_file) == 0          # nothing left: check passes


def test_a_failed_collection_refuses_and_touches_nothing(status_file: Path, capsys) -> None:
    before = status_file.read_bytes()

    def boom() -> set:
        raise P.CollectionError("collect exploded")

    assert P.main(["--write"], collect=boom, path=status_file) == 2
    assert status_file.read_bytes() == before and "REFUSING" in capsys.readouterr().err


def test_an_empty_collected_set_is_never_read_as_every_test_gone(status_file: Path) -> None:
    before = status_file.read_bytes()
    with pytest.raises(ValueError, match="EMPTY"):
        ST.prune_rows(set(), status_file)
    assert status_file.read_bytes() == before


def test_collect_output_parses_and_its_count_must_match_the_summary() -> None:
    out = "\x1b[32msrc/a_test.py::test_x[a-b]\x1b[0m\nsrc/a_test.py::test_y\n\n618/13357 tests collected (12739 deselected) in 25s\n"
    ok = "src/a_test.py::test_x[a-b]\nsrc/a_test.py::test_y\n\n2/13357 tests collected (13355 deselected) in 25s\n"
    assert P.parse_collect_output(ok) == {"src/a_test.py::test_x[a-b]", "src/a_test.py::test_y"}
    with pytest.raises(P.CollectionError, match="618 collected"):
        P.parse_collect_output(out)
    with pytest.raises(P.CollectionError, match="summary"):
        P.parse_collect_output("src/a_test.py::test_y\n")


def test_collect_slow_ids_refuses_a_nonzero_pytest_and_an_empty_collection() -> None:
    bad = lambda *a, **k: SimpleNamespace(returncode=2, stdout="ERROR collecting", stderr="boom")           # noqa: E731
    with pytest.raises(P.CollectionError, match="exited 2"):
        P.collect_slow_ids(run=bad)
    empty = lambda *a, **k: SimpleNamespace(returncode=0, stdout="0/9 tests collected (9 deselected)\n", stderr="")  # noqa: E731
    with pytest.raises(P.CollectionError, match="empty"):
        P.collect_slow_ids(run=empty)
    seen = {}

    def good(argv, **k):
        seen.update(argv=argv, env=k["env"])
        return SimpleNamespace(returncode=0, stdout="src/a_test.py::test_y\n\n1/9 tests collected (8 deselected)\n", stderr="")

    assert P.collect_slow_ids(run=good) == {"src/a_test.py::test_y"}
    assert "slow" in seen["argv"] and seen["env"]["CUDA_VISIBLE_DEVICES"] == ""
