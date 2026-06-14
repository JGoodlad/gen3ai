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


def test_notes_are_a_timestamped_append_log(tmp_path):
    run = str(tmp_path)
    stamps = iter(["2026-06-14 12:00", "2026-06-14 12:05"])
    s = ReviewStore(run, clock=lambda: next(stamps))
    s.add_note("b1", 2, "first thought")
    s.add_note("b1", 2, "  on reflection, fine  ")          # appends (does NOT overwrite); trims
    s.add_note("b1", 2, "   ")                               # whitespace → no-op
    assert s.notes("b1", 2) == [("2026-06-14 12:00", "first thought"),
                                ("2026-06-14 12:05", "on reflection, fine")]
    assert s.note("b1", 2) == "on reflection, fine"          # latest text (glyph / back-compat)
    # persists with timestamps
    assert ReviewStore(run).notes("b1", 2)[0] == ("2026-06-14 12:00", "first thought")


def test_legacy_string_note_reads_as_one_entry(tmp_path):
    run = str(tmp_path)
    import json
    with open(os.path.join(run, ReviewStore.FILENAME), "w") as fh:
        json.dump({"b1": {"4": {"flag": True, "note": "old-style note"}}}, fh)
    s = ReviewStore(run)
    assert s.notes("b1", 4) == [("", "old-style note")]      # empty ts for legacy
    assert s.has_annotation("b1", 4) and s.annotated_invs("b1") == [4]


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
    s = ReviewStore(str(tmp_path), clock=lambda: "2026-06-14 09:00")
    s.set("aaa", 1, flag=True)
    s.set("bbb", 0, note="weird switch")
    assert s.all_annotations() == [("aaa", 1, True, []),
                                   ("bbb", 0, False, [("2026-06-14 09:00", "weird switch")])]
    path = s.export_markdown()
    body = open(path).read()
    assert "weird switch" in body and "## aaa" in body and "## bbb" in body
    assert "2026-06-14 09:00" in body   # the timestamp is exported with the note


def test_no_run_dir_is_inert(tmp_path):
    s = ReviewStore(None)                 # e.g. no traces — must not crash
    assert s.toggle_flag("b", 0) is True
    assert s.export_markdown() is None
