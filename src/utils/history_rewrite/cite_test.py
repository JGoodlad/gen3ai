"""The citation rewriter: rewrite ONLY an unambiguous prefix of a rewritten commit; count and leave everything else."""
from __future__ import annotations

import hashlib
import os
import subprocess

import pytest

from utils.history_rewrite import cite as C
from utils.history_rewrite.gitobjs import commit_oids
from utils.history_rewrite.synthetic_repo_testkit import git, make_repo

OLD = "abcdef1" + "0" * 33            # rewritten
NEW = "9876543" + "1" * 33
UNC = "bcdef12" + "2" * 33            # a commit the rewrite did not change
AMB_A = "cafe123" + "a" * 33          # two old commits sharing 7 hex, one rewritten
AMB_B = "cafe123" + "b" * 33
AMB_NEW = "5555555" + "c" * 33
UA, UB = "d00d123" + "a" * 33, "d00d123" + "b" * 33   # two UNCHANGED commits sharing 7 hex
COLLIDE = "9876543" + "9" * 33        # a NEW commit sharing NEW's 7-hex prefix
MAP = {OLD: NEW, AMB_A: AMB_NEW}


def _rw() -> C.Rewriter:
    old = C.CommitIndex([OLD, UNC, AMB_A, AMB_B, UA, UB])
    new = C.CommitIndex([NEW, UNC, AMB_NEW, UA, UB])
    return C.Rewriter(MAP, old, new)


def test_decisions_cover_every_class():
    rw = _rw()
    assert rw.decide(OLD[:7]) == C.Decision("rewritten", NEW[:7], OLD)
    assert rw.decide(OLD).new == NEW                                   # a full sha
    assert rw.decide("1234567").kind == "not_a_commit"
    assert rw.decide(UNC[:9]).kind == "unchanged_commit"
    assert rw.decide(AMB_A[:7]).kind == "ambiguous_changed" and rw.decide(AMB_A[:8]).kind == "rewritten"
    assert rw.decide(UA[:7]).kind == "ambiguous_unchanged"


def test_a_new_prefix_that_collides_among_the_new_commits_is_lengthened_to_a_unique_one():
    old = C.CommitIndex([OLD])
    new = C.CommitIndex([NEW, COLLIDE])
    d = C.Rewriter({OLD: NEW}, old, new).decide(OLD[:7])
    assert d.kind == "lengthened" and d.new == NEW[:8] and len(new.prefix(d.new)) == 1


def test_rewriting_keeps_the_length_and_touches_only_tokens_that_stand_alone():
    rw = _rw()
    text = (f"at {OLD[:7]}, again {OLD[:12]}; full {OLD}; paren ({OLD[:8]}) range {OLD[:7]}..{UNC[:7]} "
            f"embedded x{OLD[:9]}y and long {OLD[:9]}zz and short {OLD[:6]}\n").encode()
    out, kinds, _ = rw.rewrite(text)
    s = out.decode()
    assert f"at {NEW[:7]}, again {NEW[:12]}; full {NEW};" in s and f"({NEW[:8]})" in s
    assert f"x{OLD[:9]}y" in s and f"{OLD[:9]}zz" in s and f"short {OLD[:6]}" in s        # glued to letters / too short
    assert len(out) == len(text)                                                       # byte-length neutral
    assert kinds["rewritten"] == 5 and kinds["unchanged_commit"] == 1


@pytest.mark.parametrize("ctx", ["pin_{t}", "{t}_run", "tag-x-{t}", "{t}-d8hhaf0y"])
def test_a_token_touching_an_underscore_or_hyphen_is_an_identifier_and_is_left_alone(ctx):
    rw = _rw()
    text = ctx.format(t=OLD[:8]).encode()
    out, kinds, _ = rw.rewrite(text)
    assert out == text and kinds["embedded"] == 1 and kinds["rewritten"] == 0


def test_an_ambiguous_token_is_refused_listed_with_its_offset_and_left_alone():
    rw = _rw()
    text = f"line one\ncite {AMB_A[:7]} here\n".encode()
    out, kinds, refusals = rw.rewrite(text)
    assert out == text and kinds["ambiguous_changed"] == 1
    (off, tok, d), = refusals
    assert tok == AMB_A[:7] and text[off:off + 7].decode() == tok and set(d.candidates) == {AMB_A, AMB_B}


def test_shortest_unique_and_prefix_lookup():
    idx = C.CommitIndex([NEW, COLLIDE, UNC])
    assert idx.prefix("9876543") == sorted([NEW, COLLIDE])
    assert idx.shortest_unique(NEW, 7) == NEW[:8]
    assert idx.shortest_unique(UNC, 7) == UNC[:7]


def test_frozen_paths():
    assert C.is_frozen("designs/research_state/ledger.md")
    assert C.is_frozen("designs/research_state/claude_md_archive/a/b.md")
    assert C.is_frozen("designs/ops/history_rewrite_sha_map.tsv")
    assert not C.is_frozen("designs/research_state/ledger.md.bak.txt")
    assert not C.is_frozen("designs/endstate/README.md")


def _tree(tmp_path, files: dict[str, bytes]) -> str:
    root = tmp_path / "tree"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    return str(root)


def test_rewrite_tree_writes_rewritable_files_and_only_counts_frozen_pinned_and_binary(tmp_path):
    pinned_body = f"a doc that cites {OLD[:8]}\n".encode()
    files = {
        "docs/a.md": f"see {OLD[:8]} and {OLD}\n".encode(),
        "designs/research_state/ledger.md": f"old entry {OLD[:8]}\n".encode(),            # frozen
        "docs/pinned.md": pinned_body,                                                   # its sha256 is cited by prereg.md
        "docs/prereg.md": f"frozen sha256 {hashlib.sha256(pinned_body).hexdigest()}\n".encode(),
        "bin/blob.dat": b"\0\0" + OLD[:8].encode(),
        "docs/clean.md": b"nothing to cite here\n",
    }
    tree = _tree(tmp_path, files)
    rw = _rw()
    dry = C.rewrite_tree(tree, rw, apply=False)
    assert not dry.written and open(f"{tree}/docs/a.md").read().count(OLD[:8]) == 2          # nothing written
    assert set(dry.per_file) == {"docs/a.md"} and "designs/research_state/ledger.md" in dry.frozen
    assert "docs/pinned.md" in dry.pinned and dry.pinned_by["docs/pinned.md"] == ["docs/prereg.md"]
    assert dry.files_binary == 1 and dry.rewritten_tokens == 2

    res = C.rewrite_tree(tree, rw, apply=True)
    assert open(f"{tree}/docs/a.md").read() == f"see {NEW[:8]} and {NEW}\n"
    assert open(f"{tree}/designs/research_state/ledger.md").read() == f"old entry {OLD[:8]}\n"   # untouched
    assert open(f"{tree}/docs/pinned.md", "rb").read() == pinned_body                          # untouched
    assert res.files_changed == 1 and res.edits == {OLD[:8]: NEW[:8], OLD: NEW}
    again = C.rewrite_tree(tree, rw, apply=False)
    assert again.rewritten_tokens == 0, "idempotent: the new tokens are not old commits"


def test_verify_edits_checks_each_new_token_against_gits_own_resolution(tmp_path):
    r = make_repo(tmp_path)
    old_fake = "abcdef1" + "0" * 33
    new_real = r["intro"]
    old_idx, new_idx = C.CommitIndex([old_fake]), C.CommitIndex(commit_oids(r["path"]))
    rw = C.Rewriter({old_fake: new_real}, old_idx, new_idx)
    tree = _tree(tmp_path, {"d.md": f"cites {old_fake[:9]} and {old_fake}\n".encode()})
    res = C.rewrite_tree(tree, rw, apply=True)
    assert C.verify_edits(res, rw, r["path"]) == []
    res.edits[old_fake[:9]] = r["root"][:9]                     # a wrong replacement must be caught
    assert any("git says" in p for p in C.verify_edits(res, rw, r["path"]))


def test_tracked_files_skips_symlinks(tmp_path):
    tree = _tree(tmp_path, {"a.md": b"x\n"})
    os.symlink("a.md", os.path.join(tree, "link.md"))
    git(tree, "add", "-A")
    assert [p for p, _ in C.tracked_files(tree)] == ["a.md"]
