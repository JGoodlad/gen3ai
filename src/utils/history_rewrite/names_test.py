"""Deriving, matching, auditing and redacting account names: on INVENTED names, with the invariant that none is ever printed."""
from __future__ import annotations

import json
import os
import stat

import pytest

from utils.history_rewrite import names as N
from utils.history_rewrite.synthetic_repo_testkit import ALL_NAMES, NAME_A_DISPLAY, NAME_A_ID, git, make_repo


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    return make_repo(tmp_path_factory.mktemp("names"))


def _ident(idx: int, tid: str, *forms: str, tier: str = "flagged") -> N.Identity:
    return N.Identity(idx, tid, tuple(forms), tier)


def test_toid_is_showdowns_lowercase_alphanumerics():
    assert N.toid("Quibble Fnord_7!") == "quibblefnord7"


def test_derive_collects_ids_display_names_and_tiers_in_a_stable_order(repo):
    ids = N.derive_identities(repo["path"], repo["intro"])
    assert [i.tid for i in ids] == ["quibblefnord", "sidewalker", "wumpushunter", "zorblaxpilot"]  # sorted, 1-based
    assert [i.idx for i in ids] == [1, 2, 3, 4]
    by = {i.tid: i for i in ids}
    assert by["zorblaxpilot"].tier == "flagged" and by["wumpushunter"].tier == "top25"  # only in top25
    assert set(by["quibblefnord"].forms) == {"quibblefnord", "Quibble Fnord"}


def test_derive_refuses_a_form_too_short_to_match_safely(tmp_path):
    r = make_repo(tmp_path, "short")
    path = os.path.join(r["path"], N.RESULTS_PATH)
    open(path, "w").write(json.dumps({"accounts": {"flagged": {}, "top25": [{"name": "Wu"}]}}))
    git(r["path"], "commit", "-q", "-am", "short name")
    with pytest.raises(ValueError, match="too short"):
        N.derive_identities(r["path"], "HEAD")


def test_names_file_is_private_and_refused_inside_the_repo(repo, tmp_path):
    ids = N.derive_identities(repo["path"], repo["intro"])
    out = tmp_path / "private" / "names.json"
    N.save_identities(out, ids, repo["path"])
    assert stat.S_IMODE(os.stat(out).st_mode) == 0o600
    assert [i.tid for i in N.load_identities(out)] == [i.tid for i in ids]
    assert [i.tid for i in N.load_identities(out, tiers=["top25"])] == ["wumpushunter"]
    with pytest.raises(ValueError, match="outside the repo"):
        N.save_identities(os.path.join(repo["path"], "names.json"), ids, repo["path"])


def test_a_gitignored_path_inside_the_repo_is_accepted(tmp_path):
    r = make_repo(tmp_path, "ign")
    open(os.path.join(r["path"], ".gitignore"), "w").write("secret/\n")
    N.assert_out_of_repo(os.path.join(r["path"], "secret", "n.json"), r["path"])


@pytest.mark.parametrize("text,expect", [
    (b"see zorblaxpilot here", True),
    (b"see ZORBLAXPILOT here", True),               # case-insensitive
    (b'"zorblaxpilot":', True),                      # quotes are boundaries
    (b"xzorblaxpilot", False),                       # touches a letter on the left
    (b"zorblaxpilot2", False),                       # touches a digit on the right
    (b"acct_zorblaxpilot_1", True),                  # an underscore is not alphanumeric
])
def test_matcher_boundaries(text, expect):
    m = N.NameMatcher([_ident(1, "zorblaxpilot", "zorblaxpilot")])
    ok = [h.boundary_ok for h in m.scan(text)]
    assert (True in ok) is expect and bool(ok)  # always FOUND; boundary_ok says whether it stands alone


def test_matcher_prefers_the_longest_form_and_falls_back_to_a_shorter_one_on_a_failed_boundary():
    m = N.NameMatcher([_ident(1, "quibble", "quibble"), _ident(2, "quibblefnord", "quibblefnord", "Quibble Fnord")])
    assert [(h.idx, h.spelling) for h in m.scan(b"a Quibble Fnord b")] == [(2, b"Quibble Fnord")]
    # "quibble fnordx": the long display form touches an x, so the shorter form that stands alone is the hit
    hits = m.scan(b"say quibble fnordx now")
    assert [(h.idx, h.boundary_ok) for h in hits] == [(1, True)]


def test_matcher_reports_the_exact_spelling_and_flags_a_form_inside_a_longer_token():
    m = N.NameMatcher([_ident(1, "zorblaxpilot", "zorblaxpilot")])
    hits = m.scan(b"ZorblaxPilot and prezorblaxpilotx")
    assert [(h.spelling, h.boundary_ok) for h in hits] == [(b"ZorblaxPilot", True), (b"zorblaxpilot", False)]


def test_redact_replaces_every_form_with_its_index():
    ids = [_ident(7, "quibblefnord", "quibblefnord", "Quibble Fnord")]
    assert N.redact("path/Quibble Fnord/quibblefnord.json", ids) == "path/#07/#07.json"


def test_audit_finds_blobs_messages_tags_paths_and_refs_and_reports_only_indices(repo, capsys):
    ids = N.derive_identities(repo["path"], repo["intro"])
    a = N.audit_history(repo["path"], ids, before=repo["intro"])
    by = {i.tid: i.idx for i in ids}
    assert a.per_identity[by["zorblaxpilot"]].blobs and a.per_identity[by["wumpushunter"]].blobs
    assert a.commit_messages[by["quibblefnord"]] == 1                      # the `msg` commit's message
    assert a.tag_messages[by["quibblefnord"]] == 1                         # the annotated tag's message
    assert a.per_identity[by["zorblaxpilot"]].pre_existing_blobs == 0      # arrived with the introducing commit
    assert a.total_hits > 0
    text = "\n".join(a.summary_lines(ids))
    assert not any(n.lower() in text.lower() for n in ALL_NAMES), "an audit summary must never print a name"


def test_audit_counts_a_ref_name_and_a_path_name_that_hold_a_name(tmp_path):
    r = make_repo(tmp_path, "refs")
    git(r["path"], "branch", f"feature-{NAME_A_ID}", r["root"])
    os.makedirs(os.path.join(r["path"], NAME_A_DISPLAY))
    open(os.path.join(r["path"], NAME_A_DISPLAY, "f.txt"), "w").write("x\n")
    git(r["path"], "add", "-A")
    git(r["path"], "commit", "-q", "-m", "dir named after an account")
    ids = N.derive_identities(r["path"], r["intro"])
    a = N.audit_history(r["path"], ids, before=r["intro"])
    idx = {i.tid: i.idx for i in ids}["zorblaxpilot"]
    assert a.ref_names[idx] == 1 and a.paths[idx] >= 1


def test_a_hit_blob_that_predates_the_introducing_commit_is_counted_as_preexisting(tmp_path):
    r = make_repo(tmp_path, "pre", root_mentions=True)  # the name is already in the ROOT, before the introducing commit
    ids = N.derive_identities(r["path"], r["intro"])
    a = N.audit_history(r["path"], ids, before=r["intro"])
    assert a.per_identity[{i.tid: i.idx for i in ids}["zorblaxpilot"]].pre_existing_blobs >= 1
    assert "PRE-DATE the introducing commit: 1" in "\n".join(a.summary_lines(ids))


def test_audit_summary_through_a_failing_git_call_does_not_leak(tmp_path):
    with pytest.raises(RuntimeError):
        N.audit_history(str(tmp_path / "not-a-repo"), [_ident(1, "zorblaxpilot", "zorblaxpilot")])


def test_cli_names_prints_counts_not_names(repo, tmp_path, capsys):
    from utils.history_rewrite.__main__ import main
    out = tmp_path / "n.json"
    rc = main(["names", "--source", repo["path"], "--introducing", repo["intro"], "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0 and "derived 4 identities" in text and "FINDING: 1 identities" in text
    assert not any(n.lower() in text.lower() for n in ALL_NAMES)


def test_matcher_handles_binary_and_empty_input():
    m = N.NameMatcher([_ident(1, "zorblaxpilot", "zorblaxpilot")])
    assert m.scan(b"") == []
    assert m.scan(bytes(range(256)) * 4) == []
