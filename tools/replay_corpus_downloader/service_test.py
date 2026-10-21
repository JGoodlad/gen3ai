"""The backfill / daily drivers, on a FAKE archive (no network)."""
import datetime as _dt
import json

import pytest

from tools.replay_corpus_downloader import freshness as F
from tools.replay_corpus_downloader import service as V
from tools.replay_corpus_downloader import sync as S
from tools.replay_corpus_downloader.replay_corpus_downloader_test import FakeServer, _row


def _archive(days):
    """Five replays a day on each LA date in ``days``, newest first, each log starting at its upload time."""
    rows = []
    for d in days:
        t0 = S.la_midnight_ts(d)
        rows += [_row(f"{d}-{j}", t0 + 3600 * (j + 1)) for j in range(5)]
    rows.sort(key=lambda r: -r["uploadtime"])
    return rows, {r["id"]: f"|t:|{r['uploadtime']}\n" for r in rows}


def test_backfill_runs_each_phase_to_exhaustion_in_chunks_then_marks_done(tmp_path):
    rows, logs = _archive(["2026-06-02", "2026-06-01", "2026-05-20", "2026-05-19", "2026-05-18"])
    phases = (V.Phase("recent", "2026-06-01", None), V.Phase("hole", "2026-05-19", "2026-05-20"))
    lines = []
    total = V.backfill(tmp_path, "gen3ou", FakeServer(rows, logs), chunk=4, phases=phases, log=lines.append)
    assert total == 15
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_dir()) == ["2026-05-19", "2026-06-01", "2026-06-02"]
    assert sum("recent chunk" in m for m in lines) == 3 and sum("hole chunk" in m for m in lines) == 2
    assert json.loads((tmp_path / V.DONE_MARKER).read_text())["written_this_run"] == 15
    assert S.existing_ids(tmp_path) and F.date_folders(tmp_path)[-1].isoformat() == "2026-06-02", \
        "the marker in the corpus root is invisible to the id scan and the freshness alarm"


def test_daily_pulls_from_the_newest_folder_minus_a_day(tmp_path):
    (tmp_path / "2026-06-01").mkdir()
    assert V.daily_since(tmp_path) == "2026-05-31"
    assert V.daily_since(tmp_path / "none") is None


def test_daily_defers_to_a_running_backfill_and_still_raises_the_stale_alarm(tmp_path, monkeypatch, capsys):
    (tmp_path / "2026-01-01").mkdir()
    lock = tmp_path / "dl.lock"
    monkeypatch.setattr(S, "lock_path", lambda: lock)
    monkeypatch.setattr(S, "checked_fetch", lambda ap, a: (lambda url: pytest.fail("no request while locked")))
    monkeypatch.setenv("JOURNAL_STREAM", "8:1")
    with S.single_instance(lock):
        rc = V.main(["daily", "--out", str(tmp_path), "--proxy", "socks5h://127.0.0.1:1080"])
    err = capsys.readouterr().err
    assert rc == F.EXIT_STALE
    assert "no pull" in err and "<3>[freshness] REPLAY CORPUS STALE" in err


def test_daily_on_a_fresh_corpus_exits_0(tmp_path, monkeypatch, capsys):
    today = _dt.datetime.now(S.CORPUS_TZ).date().isoformat()
    (tmp_path / today).mkdir()
    monkeypatch.setattr(S, "lock_path", lambda: tmp_path / "dl.lock")
    monkeypatch.setattr(S, "checked_fetch", lambda ap, a: FakeServer([], {}))
    assert V.main(["daily", "--out", str(tmp_path), "--proxy", "socks5h://127.0.0.1:1080"]) == 0
    assert "[freshness] fresh" in capsys.readouterr().out


def test_backfill_refuses_a_relative_out(capsys):
    with pytest.raises(SystemExit):
        V.main(["backfill", "--out", "replays/showdown/gen3ou", "--proxy", "socks5h://127.0.0.1:1080"])
    assert "ABSOLUTE" in capsys.readouterr().err
