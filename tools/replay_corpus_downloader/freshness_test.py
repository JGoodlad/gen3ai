"""The corpus freshness check: pure directory listing, no network, no write."""
import datetime as dt

import pytest

from tools.replay_corpus_downloader import freshness as F

D = dt.date


def _corpus(tmp_path, days, junk=()):
    for d in days:
        (tmp_path / d).mkdir()
        (tmp_path / d / f"battle-gen3ou-{d}.log").write_text("x")
    for j in junk:
        (tmp_path / j).mkdir()
    return tmp_path


def test_a_corpus_whose_newest_folder_is_today_or_yesterday_is_fresh(tmp_path):
    out = _corpus(tmp_path, ["2026-10-18", "2026-10-19"])
    code, msg = F.check(out, today=D(2026, 10, 20))
    assert code == F.EXIT_FRESH and "2026-10-19" in msg and "1 days old" in msg


def test_the_limit_is_inclusive_and_one_day_past_it_is_stale(tmp_path):
    out = _corpus(tmp_path, ["2026-10-17"])
    assert F.check(out, max_age_days=3, today=D(2026, 10, 20))[0] == F.EXIT_FRESH
    code, msg = F.check(out, max_age_days=3, today=D(2026, 10, 21))
    assert code == F.EXIT_STALE and msg.startswith("STALE") and "4 days old" in msg


def test_the_incident_that_motivated_it_reads_stale(tmp_path):
    """2026-09-23 newest, read on 2026-10-20: 27 days (F-X50-13)."""
    out = _corpus(tmp_path, ["2026-09-22", "2026-09-23"])
    code, msg = F.check(out, today=D(2026, 10, 20))
    assert code == F.EXIT_STALE and "27 days old" in msg


def test_an_absent_or_empty_corpus_is_exit_2_not_fresh(tmp_path):
    assert F.check(tmp_path / "nope", today=D(2026, 10, 20))[0] == F.EXIT_MISSING
    assert F.check(tmp_path, today=D(2026, 10, 20))[0] == F.EXIT_MISSING


def test_non_date_folders_and_stray_files_are_ignored(tmp_path):
    out = _corpus(tmp_path, ["2026-10-19"], junk=["2026-13-45", "scratch", "2026-10-25.bak"])
    (out / "2026-10-30").write_text("a FILE named like a date")
    assert F.date_folders(out) == [D(2026, 10, 19)]


def test_gap_runs_are_the_missing_calendar_days_inside_the_corpus():
    days = [D(2026, 6, 14), D(2026, 6, 15), D(2026, 6, 21), D(2026, 6, 22), D(2026, 6, 23), D(2026, 8, 2), D(2026, 8, 18)]
    assert F.gap_days(days) == [(D(2026, 6, 16), D(2026, 6, 20)), (D(2026, 6, 24), D(2026, 8, 1)), (D(2026, 8, 3), D(2026, 8, 17))]
    assert F.gap_days(days[:2]) == []


def test_main_exit_code_and_stream(tmp_path, capsys):
    out = _corpus(tmp_path, ["2000-01-01"])
    assert F.main(["--out", str(out)]) == F.EXIT_STALE
    cap = capsys.readouterr()
    assert cap.out == "" and cap.err.startswith("STALE")
    today = dt.datetime.now(F.CORPUS_TZ).date().isoformat()
    fresh = _corpus(tmp_path / "f", [today]) if (tmp_path / "f").mkdir() is None else None
    assert F.main(["--out", str(fresh)]) == F.EXIT_FRESH
    assert capsys.readouterr().out.startswith("fresh")


def test_the_check_writes_nothing(tmp_path):
    out = _corpus(tmp_path, ["2026-10-19"])
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    F.check(out, today=D(2026, 10, 20))
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
