"""The replay-corpus downloader, driven by a FAKE fetcher (no network, ever).

The layout and the date convention are the EXISTING corpus's (measured 2026-10-08 on 100 sampled files of
`<main>/replays/showdown/gen3ou`: the folder is the America/Los_Angeles date of the `|t:|` line).
"""
import json

import pytest

from tools.replay_corpus_downloader import sync as S

# 1779150458 = 2026-05-19 00:07:38 UTC = 2026-05-18 17:07:38 PDT -> folder 2026-05-18 (a UTC folder would say 05-19)
LOG = "|init|battle\n|title|A vs. B\n|j|☆A\n|t:|1779150458\n|player|p1|A|x|1300\n"


class FakeServer:
    """`search.json` pages + `<id>.log` bodies; records every URL asked for."""

    def __init__(self, rows, logs, inclusive=True):
        self.rows, self.logs, self.asked, self.inclusive = rows, logs, [], inclusive

    def __call__(self, url):
        self.asked.append(url)
        if "/search.json" in url:
            before = int(url.split("before=")[1]) if "before=" in url else None
            rows = [r for r in self.rows if before is None
                    or (r["uploadtime"] <= before if self.inclusive else r["uploadtime"] < before)]
            return json.dumps(rows[:S.PAGE_SIZE])
        rid = url.rsplit("/", 1)[1][:-len(".log")]
        if rid not in self.logs:
            raise S.FetchError(f"404 {url}")
        return self.logs[rid]


def _row(i, t, **kw):
    return {"id": f"gen3ou-{i}", "uploadtime": t, "format": "gen3ou", "private": 0, **kw}


def test_folder_date_is_the_los_angeles_date_of_the_battle_start():
    assert S.folder_date(LOG, uploadtime=9999999999) == "2026-05-18"     # |t:| wins over uploadtime
    assert S.folder_date("|init|battle\n", uploadtime=1779150458) == "2026-05-18"
    assert S.folder_date("|t:|1779174000\n", None) == "2026-05-19"       # 07:00 UTC = 00:00 PDT -> next day
    with pytest.raises(ValueError):
        S.folder_date("|init|battle\n", None)


def test_the_file_name_is_the_corpus_name():
    assert S.file_name("gen3ou-2612838147") == "battle-gen3ou-2612838147.log"


def test_sync_writes_the_corpus_layout_skips_private_and_present(tmp_path):
    rows = [_row(3, 1779150600), _row(2, 1779150500, private=1), _row(1, 1779150400)]
    srv = FakeServer(rows, {"gen3ou-3": LOG, "gen3ou-1": LOG})
    present = tmp_path / "2026-05-18"
    present.mkdir()
    (present / "battle-gen3ou-1.log").write_text("already here")
    written, skipped = S.sync(tmp_path, "gen3ou", srv, log=lambda m: None)
    assert (written, skipped) == (1, 2)
    assert (tmp_path / "2026-05-18" / "battle-gen3ou-3.log").read_text(encoding="utf-8") == LOG
    assert (present / "battle-gen3ou-1.log").read_text() == "already here"      # never overwritten
    assert not list(tmp_path.rglob("*.part"))
    assert not any("gen3ou-2.log" in u for u in srv.asked), "a private replay is not fetched"
    # a re-run fetches nothing: resumable
    srv.asked.clear()
    assert S.sync(tmp_path, "gen3ou", srv, log=lambda m: None) == (0, 3)
    assert all("/search.json" in u for u in srv.asked)


def test_existing_ids_looks_in_every_date_folder(tmp_path):
    for day, rid in (("2026-05-18", "gen3ou-1"), ("2026-06-01", "gen3ou-2")):
        (tmp_path / day).mkdir()
        (tmp_path / day / f"battle-{rid}.log").write_text("x")
    (tmp_path / "stray.txt").write_text("x")
    assert S.existing_ids(tmp_path) == {"gen3ou-1", "gen3ou-2"}
    assert S.existing_ids(tmp_path / "nope") == set()


def test_pagination_follows_before_and_stops_at_since(tmp_path):
    n = S.PAGE_SIZE + 10                                              # forces a second page
    rows = [_row(i, 2_000_000_000 - i) for i in range(n)]            # newest first
    for inclusive in (True, False):                                  # the API's `before` semantics are unverified
        srv = FakeServer(rows, {r["id"]: LOG for r in rows}, inclusive=inclusive)
        got = [r["id"] for r in S.search_pages("gen3ou", srv)]
        assert got == [r["id"] for r in rows], f"every row exactly once across the page seam (inclusive={inclusive})"
        assert sum("/search.json" in u for u in srv.asked) == 2 and "before=" in srv.asked[1]
    srv = FakeServer(rows, {r["id"]: LOG for r in rows})
    # --since: rows older than the cut-off end the walk
    since_row_ts = rows[5]["uploadtime"]
    only = [r["id"] for r in S.search_pages("gen3ou", srv, since_ts=since_row_ts)]
    assert only == [r["id"] for r in rows[:6]]


def test_max_replays_bounds_the_writes_and_list_only_writes_nothing(tmp_path):
    rows = [_row(i, 1779150000 + 100 - i) for i in range(10)]
    srv = FakeServer(rows, {r["id"]: LOG for r in rows})
    assert S.sync(tmp_path, "gen3ou", srv, max_replays=3, log=lambda m: None)[0] == 3
    assert len(list(tmp_path.rglob("*.log"))) == 3
    out2 = tmp_path / "listed"
    srv2 = FakeServer(rows, {})
    assert S.sync(out2, "gen3ou", srv2, max_replays=4, list_only=True, log=lambda m: None)[0] == 4
    assert not out2.exists() and not any(u.endswith(".log") for u in srv2.asked)


def test_a_persistent_failure_stops_the_run_instead_of_skipping(tmp_path):
    srv = FakeServer([_row(1, 1779150400)], {})                      # the .log 404s
    with pytest.raises(S.FetchError):
        S.sync(tmp_path, "gen3ou", srv, log=lambda m: None)


def test_main_refuses_an_impolite_sleep(capsys):
    with pytest.raises(SystemExit):
        S.main(["--out", "x", "--sleep", "0.1"])
    assert "refused" in capsys.readouterr().err


def test_the_real_fetcher_makes_no_request_until_called():
    S.make_http_fetch(sleep_s=1.0)     # construction touches no network
