"""Unit tests for the manual-review annotation store (pure — no Textual)."""

import json
import os

from main.prober.review import ReviewStore


def test_flag_note_roundtrip_and_persist(tmp_path):
    run = str(tmp_path)
    s = ReviewStore(run)
    assert s.flag("b1", 3) is False and s.note("b1", 3) == ""
    assert s.toggle_flag("b1", 3) is True
    s.set("b1", 5, note="explosion at full HP")
    assert s.annotated_invs("b1") == [3, 5]
    # fresh store reads the persisted file
    s2 = ReviewStore(run)
    assert s2.flag("b1", 3) is True and s2.note("b1", 5) == "explosion at full HP"


def test_prune_empties(tmp_path):
    s = ReviewStore(str(tmp_path))
    s.toggle_flag("b1", 3)
    s.set("b1", 3, flag=False)            # unflag + no note → entry pruned
    assert s.annotated_invs("b1") == []
    assert json.load(open(os.path.join(str(tmp_path), ReviewStore.FILENAME))) == {}


def test_note_survives_unflag(tmp_path):
    s = ReviewStore(str(tmp_path))
    s.set("b1", 2, flag=True, note="why?")
    s.set("b1", 2, flag=False)           # note keeps the decision annotated
    assert s.annotated_invs("b1") == [2] and s.note("b1", 2) == "why?"


def test_all_annotations_and_export(tmp_path):
    s = ReviewStore(str(tmp_path))
    s.set("aaa", 1, flag=True)
    s.set("bbb", 0, note="weird switch")
    assert s.all_annotations() == [("aaa", 1, True, ""), ("bbb", 0, False, "weird switch")]
    path = s.export_markdown()
    body = open(path).read()
    assert "weird switch" in body and "## aaa" in body and "## bbb" in body


def test_no_run_dir_is_inert(tmp_path):
    s = ReviewStore(None)                 # e.g. no traces — must not crash
    assert s.toggle_flag("b", 0) is True
    assert s.export_markdown() is None
