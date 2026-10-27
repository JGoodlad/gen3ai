"""End to end on a SYNTHETIC repository: rules -> clone -> real ``git filter-repo`` -> map -> verify -> citation rewrite.

Needs the pinned ``git_filter_repo.py`` (cached under ``~/.cache/gen3ai/history_rewrite/filter_repo/`` by the first
run; fetched from PyPI with a sha256 check when absent, and the module SKIPS when it is offline).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from utils.history_rewrite import cite as C
from utils.history_rewrite import names as N
from utils.history_rewrite import rewrite as W
from utils.history_rewrite import shamap as S
from utils.history_rewrite import verify as V
from utils.history_rewrite.gitobjs import commit_oids, run_git
from utils.history_rewrite.rules import build_rules
from utils.history_rewrite.synthetic_repo_testkit import ALL_NAMES, NAME_D_ID, git, make_repo

pytestmark = pytest.mark.integration

CACHE = os.path.expanduser("~/.cache/gen3ai/history_rewrite/filter_repo")


@pytest.fixture(scope="module")
def filter_repo() -> str:
    try:
        return str(W.fetch_filter_repo(CACHE))
    except Exception as e:  # offline: nothing to run, say so
        pytest.skip(f"git-filter-repo is not cached and cannot be downloaded: {type(e).__name__}")


@pytest.fixture(scope="module")
def rewritten(tmp_path_factory, filter_repo):
    tmp = tmp_path_factory.mktemp("e2e")
    src = make_repo(tmp)
    ids = N.derive_identities(src["path"], src["intro"])
    audit = N.audit_history(src["path"], ids, before=src["intro"])
    scope = {N.RESULTS_PATH, "prose.md"}              # the "account-data files" of this tiny repo
    rules = build_rules(audit, ids, scope)
    rules_path, cb_path = tmp / "rules.txt", tmp / "callback.py"
    rules_path.write_bytes(rules.render(ids))
    cb_path.write_text(rules.render_callback(ids))
    clone = W.run_rewrite(src["path"], str(tmp / "work"), str(rules_path), str(cb_path), filter_repo,
                          introducing=src["intro"], log=lambda m: None)
    full = S.read_commit_map(tmp / "work" / "commit-map")
    return {"src": src, "ids": ids, "rules": rules, "clone": clone, "full": full, "pairs": S.changed_only(full), "tmp": tmp}


def test_the_names_are_gone_everywhere_except_the_waived_identity_in_its_waived_file(rewritten):
    r = rewritten
    assert set(r["rules"].waived) == {[i.idx for i in r["ids"] if i.tid == NAME_D_ID][0]}      # only sidewalker is scoped
    check = V.check_names(r["clone"], r["ids"], r["rules"].waived)
    assert check.ok, check.line() + str(check.items)
    assert "waived" in " ".join(check.items)
    assert NAME_D_ID in run_git(r["clone"], "show", "refs/heads/side:side.txt").decode()       # left alone, by design


def test_results_json_history_keeps_distinct_keys_and_valid_json(rewritten):
    r = rewritten
    new_intro = r["pairs"][r["src"]["intro"]]
    doc = json.loads(run_git(r["clone"], "show", f"{new_intro}:{N.RESULTS_PATH}"))
    flagged = list(doc["accounts"]["flagged"])
    assert len(flagged) == 3 and len(set(flagged)) == 3 and all(k.startswith("redacted-account-") for k in flagged)
    blob = run_git(r["clone"], "show", f"{new_intro}:prose.md").decode()
    assert "redacted-account-" in blob and not any(n.lower() in blob.lower() for n in ALL_NAMES if n != NAME_D_ID)


def test_only_descendants_of_the_introducing_commit_changed_and_the_graph_is_intact(rewritten):
    r = rewritten
    s = r["src"]
    assert set(r["pairs"]) == {s["intro"], s["cite"], s["msg"], s["pin"]}      # the pin too: a keep-ref carried it
    assert s["root"] not in r["pairs"] and s["side"] not in r["pairs"]                           # untouched SHAs
    assert s["side2"] not in r["pairs"], "a message that cites a rewritten commit must not change the citing commit's sha"
    for chk in (V.check_changed_set(s["path"], r["pairs"], r["full"], s["intro"]),
                V.check_graph(r["clone"], s["path"], r["pairs"], r["full"]),
                V.check_tip_trees(r["clone"], s["path"], S.read_ref_map(r["tmp"] / "work" / "ref-map")),
                V.check_variants(r["clone"], s["path"], r["ids"], (s["intro"],))):
        assert chk.ok, chk.line() + str(chk.items)


def test_a_tag_is_rewritten_and_a_commit_message_keeps_the_sha_it_cited(rewritten):
    r = rewritten
    new_intro = r["pairs"][r["src"]["intro"]]
    tag = run_git(r["clone"], "cat-file", "-p", "refs/tags/v-intro").decode()
    assert new_intro in tag and not any(n.lower() in tag.lower() for n in ALL_NAMES)
    s_intro7 = r["src"]["intro"][:7]
    msg = run_git(r["clone"], "log", "-1", "--format=%s", r["pairs"][r["src"]["cite"]]).decode()
    assert s_intro7 in msg and new_intro[:7] not in msg            # --preserve-commit-hashes: a message keeps the sha it cited


def test_the_citation_rewrite_points_at_the_new_commits_and_resolves(rewritten):
    r = rewritten
    clone, s = r["clone"], r["src"]
    rw = C.Rewriter(r["pairs"], C.CommitIndex(commit_oids(s["path"])), C.CommitIndex(commit_oids(clone)))
    res = C.rewrite_tree(clone, rw, apply=True)
    assert C.verify_edits(res, rw, clone) == []
    new_intro = r["pairs"][s["intro"]]
    doc = Path(clone, "docs", "cites.md").read_text()
    assert s["root"][:7] in doc                                    # root did not change: its citation stands
    assert new_intro[:7] in doc and new_intro in doc
    assert doc.count(s["intro"][:7]) == 2                           # only the two identifier-embedded ones still carry the old prefix
    assert f"pin_{s['intro'][:8]}" in doc and f"x-{s['intro'][:8]}" in doc          # identifier-embedded: kept
    assert res.kinds["embedded"] == 2 and res.kinds["rewritten"] == 2
    assert C.rewrite_tree(clone, rw, apply=False).rewritten_tokens == 0


def test_run_rewrite_refuses_a_work_dir_inside_the_source_or_a_used_one(tmp_path, filter_repo):
    src = make_repo(tmp_path)
    with pytest.raises(ValueError, match="separate directory"):
        W.run_rewrite(src["path"], os.path.join(src["path"], "work"), "r", "c", filter_repo, introducing=src["intro"])
    used = tmp_path / "used"
    used.mkdir()
    (used / "f").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        W.run_rewrite(src["path"], str(used), "r", "c", filter_repo, introducing=src["intro"])
    assert git(src["path"], "rev-parse", "main") == src["msg"]       # the source was never touched


def test_unreachable_descendants_are_found_and_carried_through_the_rewrite(rewritten):
    r = rewritten
    s = r["src"]
    keep = run_git(r["clone"], "for-each-ref", "--format=%(objectname) %(refname)", "refs/history-rewrite/keep/").decode().split()
    assert keep == [r["pairs"][s["pin"]], f"refs/history-rewrite/keep/{s['pin']}"]
    clone2 = str(r["tmp"] / "c2")
    git(r["tmp"], "clone", "-q", "--no-local", s["path"], clone2)
    assert W.unreachable_descendants(s["path"], clone2, s["intro"]) == [s["pin"]]
