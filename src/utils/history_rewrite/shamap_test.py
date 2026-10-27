"""filter-repo's commit-map -> the in-repo TSV: parsing, validation (no pruned commit, a clean bijection), round trip."""
from __future__ import annotations

import pytest

from utils.history_rewrite import shamap as S

A, B, C, D, E = ("a" * 40, "b" * 40, "c" * 40, "d" * 40, "e" * 40)


def _write_cm(path, rows):
    path.write_text("old                                      new\n" + "".join(f"{o} {n}\n" for o, n in rows))


def test_commit_map_parses_every_row_and_changed_only_drops_the_unchanged(tmp_path):
    p = tmp_path / "commit-map"
    _write_cm(p, [(A, A), (B, C), (D, E)])
    full = S.read_commit_map(p)
    assert full == {A: A, B: C, D: E}
    assert S.changed_only(full) == {B: C, D: E}


def test_ref_map_rows(tmp_path):
    p = tmp_path / "ref-map"
    p.write_text(f"old new ref\n{A} {B} refs/heads/main\n{C} {D} refs/tags/x\n")
    assert S.read_ref_map(p) == [(A, B, "refs/heads/main"), (C, D, "refs/tags/x")]


@pytest.mark.parametrize("pairs,msg", [
    ({A: "0" * 40}, "PRUNED"),
    ({A: A}, "unchanged"),
    ({A: "zz"}, "not 40-hex"),
    ({A: C, B: C}, "not a bijection"),
    ({A: B, B: C}, "also a new id"),
])
def test_validate_refuses_what_would_make_a_citation_unmappable(pairs, msg):
    with pytest.raises(ValueError, match=msg):
        S.validate(pairs)


def test_tsv_round_trip_keeps_comments_out_and_rows_sorted(tmp_path):
    p = tmp_path / "m.tsv"
    S.write_tsv(p, {D: E, B: C}, ["two rows", "# already a comment"])
    text = p.read_text()
    assert text.splitlines()[:2] == ["# two rows", "# already a comment"]
    assert S.read_tsv(p) == {B: C, D: E}
    assert text.splitlines()[2].startswith(B)


def test_an_absent_or_header_only_tsv_is_the_empty_map(tmp_path):
    assert S.read_tsv(tmp_path / "nope.tsv") == {}
    (tmp_path / "h.tsv").write_text(S.HEADER_STATUS_NONE + "\n")
    assert S.read_tsv(tmp_path / "h.tsv") == {}


def test_a_malformed_row_is_refused(tmp_path):
    (tmp_path / "bad.tsv").write_text("abc\tdef\n")
    with pytest.raises(ValueError, match="malformed"):
        S.read_tsv(tmp_path / "bad.tsv")
