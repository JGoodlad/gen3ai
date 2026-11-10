"""The human-data switch's grammar, canonical text, hash, surfaces and record (``agents.info_levers.human_data``)."""
import hashlib

import pytest

from agents.info_levers import human_data as HD


@pytest.mark.parametrize("text, canonical", [
    (None, "off"), ("", "off"), ("off", "off"), (" off ", "off"),
    ("teams:hr_spec_v1.r3@0.5", "teams:hr_spec_v1.r3@0.5"),
    ("teams:hr_spec_v1.r3@.50", "teams:hr_spec_v1.r3@0.5"),
    ("teams:hr_spec_v1.r3@0.5000", "teams:hr_spec_v1.r3@0.5"),
    ("teams:hr_spec_v1.r3@1", "teams:hr_spec_v1.r3@1"),
    ("teams:hr_spec_v1.r3@1.0", "teams:hr_spec_v1.r3@1"),
    ("teams:hr_spec_v1.r3@0", "teams:hr_spec_v1.r3@0"),
    ("teams:hr_spec_v1.r3@0.0625", "teams:hr_spec_v1.r3@0.0625"),
    ("init:hclone_t@1", "init:hclone_t@1"),
    ("init:hclone_t@1.0", "init:hclone_t@1"),
    ("init:hclone_t@0", "init:hclone_t@0"),
    ("opp:hclone_t@.2", "opp:hclone_t@0.2"),
    ("opp:hclone_t@0.2000", "opp:hclone_t@0.2"),
])
def test_canonical_text(text, canonical):
    assert HD.parse(text).canonical == canonical


def test_the_hash_is_the_documented_one():
    c = HD.parse("teams:hr_spec_v1.r3@0.50")
    want = hashlib.sha256(b"human_data_v1:teams:hr_spec_v1.r3@0.5").hexdigest()[:16]
    assert c.hash == want == HD.clause_hash("teams:hr_spec_v1.r3@0.5")
    assert HD.parse("off").hash == hashlib.sha256(b"human_data_v1:off").hexdigest()[:16]
    assert HD.parse("teams:hr_spec_v1.r3@0.5").hash != HD.parse("teams:hr_spec_v1.r3@0.25").hash


@pytest.mark.parametrize("text", [
    "on", "teams", "teams:hr_x", "teams:@0.5", "teams:hr_x@", "teams:hr_x@abc", "teams:hr_x@-0.1",
    "teams:hr_x@1.5", "teams:hr_x@nan", "teams:hr_x@inf", "judge:hom@1", "teams:hr x@0.5",
    "teams:hr_x@0.00001", "teams:hr_x@0.5:x", "TEAMS:hr_x@0.5",
    # the two heads the 2026-11-10 grammar dropped (X59-S's read closed bc; kl was never routed)
    "bc:hr_spec_v1.r2@0.25", "kl:hom_v1@0.1",
    # init is all or nothing (0 or 1); an opponent share is a probability
    "init:hclone_t@0.5", "init:hclone_t@2", "opp:hclone_t@1.5", "opp:hclone_t@-0.1", "opp:@0.2", "init:hclone t@1",
])
def test_bad_clauses_are_refused(text):
    with pytest.raises(HD.HumanDataError):
        HD.parse(text)


def test_the_grammar_has_three_heads_and_only_training_honours_them():
    """Fails on revert of the 2026-11-10 grammar (``bc`` / ``kl`` back in, or ``init`` / ``opp`` out)."""
    assert HD.LEVERS == ("teams", "init", "opp") and HD.NOT_BUILT == frozenset()
    assert HD.ROWSET_LEVERS | HD.MODEL_LEVERS == set(HD.LEVERS) and not HD.ROWSET_LEVERS & HD.MODEL_LEVERS
    for text in ("teams:hr_x.r1@0.5", "init:hclone_t@1", "opp:hclone_t@0.2"):
        c = HD.parse(text)
        assert HD.honoured_or_refuse(c, "train") is c
        assert (c.rowset is None) != (c.model is None)           # a clause names a rowset OR a model
        for surface in ("eval", "play"):
            with pytest.raises(HD.HumanDataError):
                HD.honoured_or_refuse(c, surface)
    assert HD.parse("init:hclone_t@1").model == "hclone_t" and HD.parse("teams:hr_x.r1@0.5").rowset == "hr_x.r1"
    for surface in ("train", "eval", "play"):
        assert HD.honoured_or_refuse(HD.parse("off"), surface).is_off
    with pytest.raises(HD.HumanDataError, match="unknown human-data surface"):
        HD.honoured_or_refuse(HD.parse("off"), "search")
    for head in HD.LEVERS:
        assert f"{head}:<" in HD.GRAMMAR


def test_record_and_stamp():
    off = HD.record(HD.parse(None))
    assert off == {"schema": "human_data_v1", "label": "off", "canonical": "off", "hash": HD.clause_hash("off"),
                   "lever": "off"}
    assert not HD.is_on(off) and not HD.is_on(None)
    assert HD.stamp_line(off).startswith("[HUMAN-DATA] lever=off rowset=- rows_sha=- ")
    corpus = {"data_sha256": "d" * 64, "manifest_sha256": "m" * 64, "n": 7}
    on = HD.record(HD.parse("teams:hr_x.r1@0.5"), corpus)
    assert HD.is_on(on) and on["rowset"] == "hr_x.r1" and on["model"] is None and on["param"] == 0.5
    assert on["label"] == "teams:hr_x.r1" and on["corpus"] == corpus
    line = HD.stamp_line(on)
    assert line.startswith(f"[HUMAN-DATA] lever=teams rowset=hr_x.r1 rows_sha={'d' * 64} ")
    assert "NULL-DOSE" not in line
    assert "NULL-DOSE" in HD.stamp_line(HD.record(HD.parse("teams:hr_x.r1@0"), corpus))


def test_a_model_clause_records_the_model_and_stamps_its_rows_and_zip():
    """The stamp's first three fields stay ``lever= rowset= rows_sha=`` for a model lever: the rowset and the rows
    manifest are the ones the model was trained on. Fails on revert of the ``human_model`` branch of the record."""
    hm = {"id": "hclone_t", "rowset": "hr_x.r1", "rows_manifest_sha256": "r" * 64, "checkpoint_sha256": "z" * 64}
    rec = HD.record(HD.parse("opp:hclone_t@0.2"), None, hm)
    assert rec["model"] == "hclone_t" and rec["rowset"] == "hr_x.r1" and rec["human_model"] == hm
    assert rec["label"] == "opp:hclone_t" and rec["param"] == 0.2 and not HD.is_null_dose(rec)
    line = HD.stamp_line(rec)
    assert line.startswith(f"[HUMAN-DATA] lever=opp rowset=hr_x.r1 rows_sha={'r' * 64} ")
    assert f"model=hclone_t zip_sha={'z' * 64}" in line and "NULL-DOSE" not in line
    null = HD.record(HD.parse("init:hclone_t@0"), None, hm)
    assert HD.is_null_dose(null) and "NULL-DOSE" in HD.stamp_line(null)
    assert not HD.is_null_dose(HD.record(HD.parse("off"))) and not HD.is_null_dose(None)
