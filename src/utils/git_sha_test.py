"""``utils.git_sha``: map first, git second; ambiguity is reported, never silently picked."""
from __future__ import annotations

import pytest

from utils import git_sha as G
from utils.history_rewrite.synthetic_repo_testkit import make_repo

OLD1 = "1234567" + "a" * 33
OLD2 = "1234567" + "b" * 33          # shares the 7-hex prefix with OLD1
NEW1, NEW2 = "f" * 40, "e" * 40


@pytest.fixture()
def repo(tmp_path):
    return make_repo(tmp_path)


def _map(tmp_path, rows):
    p = tmp_path / "map.tsv"
    p.write_text("# header\n" + "".join(f"{o}\t{n}\n" for o, n in rows))
    return p


def test_an_old_sha_resolves_to_the_new_one_by_any_prefix_length(repo, tmp_path):
    m = _map(tmp_path, [(OLD1, NEW1)])
    for tok in (OLD1, OLD1[:12], OLD1[:7], OLD1[:7].upper()):
        r = G.resolve(tok, repo=repo["path"], map_path=m)
        assert (r.status, r.full, r.candidates) == (G.MAPPED, NEW1, (OLD1,))


def test_a_prefix_of_two_rewritten_commits_is_ambiguous_and_names_both(repo, tmp_path):
    m = _map(tmp_path, [(OLD1, NEW1), (OLD2, NEW2)])
    r = G.resolve(OLD1[:7], repo=repo["path"], map_path=m)
    assert r.status == G.AMBIGUOUS and set(r.candidates) == {OLD1, OLD2} and r.full is None
    assert G.resolve(OLD1[:9], repo=repo["path"], map_path=m).status == G.MAPPED      # cite more digits and it is exact


def test_a_commit_the_rewrite_did_not_touch_resolves_through_git(repo, tmp_path):
    m = _map(tmp_path, [(OLD1, NEW1)])
    r = G.resolve(repo["root"][:8], repo=repo["path"], map_path=m)
    assert (r.status, r.full) == (G.GIT, repo["root"])


def test_the_map_wins_over_git_even_while_the_old_commit_still_exists_locally(repo, tmp_path):
    """After the rewrite the OLD commit lingers in the object database (a worktree pin, the reflog): git must not answer."""
    m = _map(tmp_path, [(repo["intro"], NEW1)])
    r = G.resolve(repo["intro"][:8], repo=repo["path"], map_path=m)
    assert (r.status, r.full) == (G.MAPPED, NEW1)


def test_a_token_naming_a_rewritten_commit_and_a_different_live_one_is_refused(tmp_path):
    r = make_repo(tmp_path)
    live = r["cite"]
    fake_old = live[:7] + "0" * 33      # an old sha whose 7-hex prefix is also a LIVE commit's
    m = _map(tmp_path, [(fake_old, NEW1)])
    res = G.resolve(live[:7], repo=r["path"], map_path=m)
    assert res.status == G.AMBIGUOUS and "different live" in res.note
    assert G.resolve(live[:10], repo=r["path"], map_path=m).status == G.GIT          # more digits: unambiguous again


def test_unknown_and_non_hex_tokens(repo, tmp_path):
    m = _map(tmp_path, [])
    assert G.resolve("deadbee", repo=repo["path"], map_path=m).status == G.UNKNOWN
    assert G.resolve("not-a-sha", repo=repo["path"], map_path=m).status == G.UNKNOWN
    with pytest.raises(G.ShaError, match="unknown"):
        G.new_sha("deadbee", repo=repo["path"], map_path=m)


def test_new_sha_returns_the_full_current_sha(repo, tmp_path):
    assert G.new_sha(OLD1[:8], repo=repo["path"], map_path=_map(tmp_path, [(OLD1, NEW1)])) == NEW1


def test_old_sha_reverse_lookup(tmp_path):
    m = _map(tmp_path, [(OLD1, NEW1), (OLD2, NEW2)])
    assert G.old_sha(NEW2[:8], map_path=m) == [OLD2]
    assert G.old_sha("0" * 8, map_path=m) == []


def test_no_map_file_is_the_empty_map_and_the_shipped_default_parses():
    assert G.load_map("/nonexistent/map.tsv") == {}
    assert G.MAP_PATH.exists() and isinstance(G.load_map(G.MAP_PATH), dict)   # header-only until the rewrite lands, then its rows


def test_a_malformed_map_is_refused_not_ignored(tmp_path):
    p = tmp_path / "bad.tsv"
    p.write_text("abc\tdef\n")
    with pytest.raises(ValueError, match="malformed"):
        G.load_map(p)


def test_cli_exit_codes_and_output(repo, tmp_path, capsys):
    m = str(_map(tmp_path, [(OLD1, NEW1), (OLD2, NEW2)]))
    assert G.main(["--map", m, "--repo", repo["path"], OLD1[:12], repo["root"][:8]]) == 0
    out = capsys.readouterr().out
    assert f"-> {NEW1}  (mapped)" in out and f"-> {repo['root']}  (git)" in out
    assert G.main(["--map", m, "--repo", repo["path"], "deadbee"]) == 1
    assert G.main(["--map", m, "--repo", repo["path"], OLD1[:7]]) == 2
