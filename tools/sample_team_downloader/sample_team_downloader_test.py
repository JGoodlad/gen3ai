"""Gates for ``tools/sample_team_downloader/sync.py`` — all offline, on saved HTML.

``fixtures/first_post_2026-09-23.html`` is the thread's real first post as fetched on 2026-09-23
(scripts stripped). The synthetic posts below are the two shapes the old ordered-zip pairing got
silently wrong: a pokepast link that is not a team, and a team whose title line is missing.
"""
import json
import os

import pytest
from bs4 import BeautifulSoup

from tools.sample_team_downloader import sync
from tools.sample_team_downloader.sync import (
    SyncError, check_count, extract_metadata, find_first_post, is_tool_entry, load_owned_manifest,
    make_entry, sync_teams)

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_POST = os.path.join(HERE, "fixtures", "first_post_2026-09-23.html")
SPRITES = '<img class="smilie" alt=":skarmory:"><img class="smilie" alt=":blissey:">'


def _post(body: str):
    return BeautifulSoup(f'<div class="message-inner"><div class="bbWrapper">{body}</div></div>',
                         "lxml").select_one(".bbWrapper")


def _team(pid: str, title: str) -> str:
    return f'<a href="https://pokepast.es/{pid}">{SPRITES}</a><br/>\n{title}<br/>\n'


# ── the real post ───────────────────────────────────────────────────────────────────────────────

def _real_teams():
    with open(REAL_POST, encoding="utf-8") as fh:
        return extract_metadata(find_first_post(fh.read()))


def test_the_real_post_parses_to_the_committed_manifest_exactly():
    """The committed ``data/teams/sample/teams.json`` IS this tool's output on the saved post —
    so the manifest cannot drift from what the tool would write, and the folder holds exactly
    Smogon's set (no promoted or hand-added entry could be in it)."""
    from utils.paths import repo_path
    committed = json.loads(repo_path("data", "teams", "sample", "teams.json").read_text())
    assert [make_entry(t) for t in _real_teams()] == committed
    assert len(committed) == 32
    files = sorted(p for p in os.listdir(repo_path("data", "teams", "sample")) if p.endswith(".txt"))
    assert files == sorted(f"{e['id']}.txt" for e in committed), \
        "data/teams/sample/ holds a .txt the manifest does not list (or vice versa)"


def test_the_real_post_pairs_each_title_with_its_own_link():
    teams = {t["url"].split("/")[-1]: t for t in _real_teams()}
    crl = teams["04e417ef9822abe9"]            # the paste the thread swapped in for 0972146213a667c9
    assert (crl["name"], crl["author"], crl["category"]) == ("Curse RestLax ForreDol", "ABR", "Stall")
    big5 = teams["f6229d2c867e21d6"]
    assert (big5["name"], big5["author"], big5["category"]) == \
        ("Big 5 + Starmie (Beerlover)", "UD", "Balance")
    # a co-authored title keeps BOTH authors (the old next-sibling read kept only the first)
    assert teams["8e768980fc8f3b5f"]["author"] == "Shitrock enjoyer and giraffefromholland"
    # a title followed directly by a spoiler block (no <br/> between) still ends at the block
    assert teams["d4e74946b54f1a4b"]["name"] == "Zapdos + AeroBi Spikes Offense"
    assert teams["d4e74946b54f1a4b"]["author"] == "Shitrock enjoyer"


# ── structural pairing ──────────────────────────────────────────────────────────────────────────

def test_a_non_team_pokepaste_link_is_ignored_and_does_not_shift_later_titles():
    """The ordered zip's failure mode: one extra link and every later team is misnamed."""
    post = _post(
        "<b>Balance</b><br/>"
        + _team("aaaa", "Big 5 + Starmie – by UD")
        + 'Try <a href="https://pokepast.es/variant1">this variant</a> too.<br/>'   # text link
        + '<div class="bbCodeSpoiler"><div class="bbCodeSpoiler-content">'
          f'<a href="https://pokepast.es/spoiled">{SPRITES}</a><br/>A – by B<br/></div></div>'
        + "<b>Offense</b><br/>"
        + _team("bbbb", "Superman TSS – by <a href='/members/x.1/'>ADV Community</a>"))
    got = extract_metadata(post)
    assert [(t["url"], t["name"], t["author"], t["category"]) for t in got] == [
        ("https://pokepast.es/aaaa", "Big 5 + Starmie", "UD", "Balance"),
        ("https://pokepast.es/bbbb", "Superman TSS", "ADV Community", "Offense"),
    ]


def test_a_team_link_with_no_title_line_is_refused():
    post = _post("<b>Stall</b><br/>"
                 + _team("aaaa", "Triple Natural Cure – by ABR")
                 + f'<a href="https://pokepast.es/bbbb">{SPRITES}</a><br/>\n<br/>\n'   # no title
                 + _team("cccc", "Quad Band – by M Dragon"))
    with pytest.raises(SyncError, match="pokepast.es/bbbb"):
        extract_metadata(post)


def test_a_title_line_without_an_author_is_refused():
    post = _post(_team("aaaa", "Just A Name"))
    with pytest.raises(SyncError, match="Title – by Author"):
        extract_metadata(post)


def test_a_repeated_link_is_kept_once():
    post = _post(_team("aaaa", "X – by Y") + _team("aaaa", "X – by Y"))
    assert [t["url"] for t in extract_metadata(post)] == ["https://pokepast.es/aaaa"]


# ── manifest ownership + the count pin ──────────────────────────────────────────────────────────

def _entry(pid="0123456789abcdef"):
    return make_entry({"url": f"https://pokepast.es/{pid}", "name": "N", "author": "A",
                       "category": "Balance"})


def test_tool_entries_are_recognised_and_promoted_ones_are_not():
    assert is_tool_entry(_entry())
    promoted = {"id": "8bdb5796b9", "name": "x", "format": "gen3ou", "category": "balance",
                "valid": True, "errors": [], "file": "teams/sample/8bdb5796b9.txt",
                "source": "https://pokepast.es/c78460887076e254",
                "promoted": {"from": "data/teams/others/yak_attack/c78460887076e254.txt"}}
    assert not is_tool_entry(promoted)
    assert not is_tool_entry({**_entry(), "file": "teams/promoted/0123456789abcdef.txt"})


def test_a_manifest_holding_a_foreign_entry_is_refused(tmp_path):
    p = tmp_path / "teams.json"
    p.write_text(json.dumps([_entry(), {"id": "hand", "file": "teams/sample/hand.txt"}]))
    with pytest.raises(SyncError, match="did not write"):
        load_owned_manifest(str(p))
    p.write_text(json.dumps([_entry()]))
    assert load_owned_manifest(str(p)) == [_entry()]
    assert load_owned_manifest(str(tmp_path / "missing.json")) == []


def test_a_changed_team_count_needs_the_flag():
    with pytest.raises(SyncError, match="--accept-count-change"):
        check_count(31, 32, accept=False)
    check_count(31, 32, accept=True)
    check_count(32, 32, accept=False)
    check_count(5, 0, accept=False)          # a first sync has no committed count to hold


# ── the whole sync, on a synthetic tree ─────────────────────────────────────────────────────────

def _tree(tmp_path, ids):
    root = tmp_path / "root"
    (root / "data" / "teams" / "sample").mkdir(parents=True)
    rows = []
    for pid in ids:
        (root / "data" / "teams" / "sample" / f"{pid}.txt").write_text(f"team {pid}\n")
        rows.append(make_entry({"url": f"https://pokepast.es/{pid}", "name": f"Name {pid}",
                                "author": "A", "category": "Stall"}))
    (root / "data" / "teams" / "sample" / "teams.json").write_text(json.dumps(rows, indent=2))
    return str(root)


def _html(pairs):
    return ('<div class="message-inner"><div class="bbWrapper"><b>Stall</b><br/>'
            + "".join(_team(pid, f"{name} – by A") for pid, name in pairs) + "</div></div>")


def _snapshot(root):
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            out[os.path.relpath(p, root)] = open(p, "rb").read()
    return out


def test_a_replaced_paste_is_retired_to_superseded_with_a_relocation(tmp_path):
    root = _tree(tmp_path, ["old1", "keep"])
    html = _html([("new1", "Name old1"), ("keep", "Name keep")])
    rc = sync_teams([], fetch_thread=lambda: html, fetch_team=lambda url: f"fetched {url}\n",
                    root=root)
    assert rc == 0
    teams = os.path.join(root, "data", "teams")
    assert sorted(os.listdir(os.path.join(teams, "sample"))) == ["keep.txt", "new1.txt", "teams.json"]
    manifest = json.load(open(os.path.join(teams, "sample", "teams.json")))
    assert [e["id"] for e in manifest] == ["new1", "keep"]
    assert all(is_tool_entry(e) for e in manifest)
    # the dropped paste keeps its bytes, moves to superseded/, and its old path resolves
    assert open(os.path.join(teams, "superseded", "old1.txt")).read() == "team old1\n"
    sup = json.load(open(os.path.join(teams, "superseded", "teams.json")))
    assert [(e["id"], e["file"], e["superseded"]["replaced_by"]) for e in sup] == \
        [("old1", "teams/superseded/old1.txt", "new1")]
    reloc = json.load(open(os.path.join(teams, "relocations.json")))
    assert reloc["moved"] == {"teams/sample/old1.txt": "teams/superseded/old1.txt"}


def test_a_count_change_or_a_failed_download_writes_nothing(tmp_path):
    root = _tree(tmp_path, ["a", "b"])
    before = _snapshot(root)
    fetch = lambda url: "x\n"  # noqa: E731
    assert sync_teams([], fetch_thread=lambda: _html([("a", "A")]), fetch_team=fetch, root=root) == 1
    assert _snapshot(root) == before

    def flaky(url):
        if url.endswith("/b"):
            raise SyncError("HTTP 503")
        return "x\n"
    assert sync_teams([], fetch_thread=lambda: _html([("a", "A"), ("b", "B")]), fetch_team=flaky,
                      root=root) == 1
    assert _snapshot(root) == before
    # --accept-count-change lets a shrunken post through (and retires the dropped paste)
    assert sync_teams(["--accept-count-change"], fetch_thread=lambda: _html([("a", "A")]),
                      fetch_team=fetch, root=root) == 0
    assert [e["id"] for e in json.load(open(os.path.join(root, "data/teams/sample/teams.json")))] \
        == ["a"]


def test_a_foreign_manifest_entry_blocks_the_sync(tmp_path):
    root = _tree(tmp_path, ["a"])
    path = os.path.join(root, "data", "teams", "sample", "teams.json")
    rows = json.load(open(path)) + [{"id": "p", "file": "teams/sample/p.txt", "promoted": {}}]
    open(path, "w").write(json.dumps(rows))
    before = _snapshot(root)
    assert sync_teams([], fetch_thread=lambda: _html([("a", "A")]),
                      fetch_team=lambda url: "x\n", root=root) == 1
    assert _snapshot(root) == before


def test_a_no_op_sync_is_byte_identical(tmp_path):
    root = _tree(tmp_path, ["a", "b"])        # the fetched text below equals what is on disk
    before = _snapshot(root)
    assert sync_teams([], fetch_thread=lambda: _html([("a", "Name a"), ("b", "Name b")]),
                      fetch_team=lambda url: f"team {url.split('/')[-1]}\n", root=root) == 0
    assert _snapshot(root) == before


def test_dry_run_writes_nothing(tmp_path):
    root = _tree(tmp_path, ["a"])
    before = _snapshot(root)
    assert sync_teams(["--dry-run"], fetch_thread=lambda: _html([("z", "Z")]),
                      fetch_team=lambda url: pytest.fail("dry-run must not download"),
                      root=root) == 0
    assert _snapshot(root) == before


def test_the_module_writes_to_the_curated_folder_only():
    assert sync.OUTPUT_DIR == "data/teams/sample"
    assert sync.SUPERSEDED_DIR == "data/teams/superseded"
