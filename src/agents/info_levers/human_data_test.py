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
    ("bc:hr_spec_v1.r2@0.25", "bc:hr_spec_v1.r2@0.25"),
    ("bc:hr_spec_v1.r2@2.5", "bc:hr_spec_v1.r2@2.5"),
    ("kl:hom_v1@0.1", "kl:hom_v1@0.1"),
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
])
def test_bad_clauses_are_refused(text):
    with pytest.raises(HD.HumanDataError):
        HD.parse(text)


def test_the_grammar_keeps_four_heads_and_only_teams_trains():
    assert HD.LEVERS == ("bc", "teams", "kl")
    assert HD.honoured_or_refuse(HD.parse("teams:hr_x.r1@0.5"), "train").lever == "teams"
    assert HD.honoured_or_refuse(HD.parse("off"), "train").is_off
    for text in ("bc:hr_x.r1@0.5", "kl:hom@0.5"):
        with pytest.raises(HD.HumanDataError, match="NOT BUILT"):
            HD.honoured_or_refuse(HD.parse(text), "train")
    for surface in ("eval", "play"):
        assert HD.honoured_or_refuse(HD.parse("off"), surface).is_off
        for text in ("teams:hr_x.r1@0.5", "bc:hr_x.r1@0.5", "kl:hom@0.5"):
            with pytest.raises(HD.HumanDataError):
                HD.honoured_or_refuse(HD.parse(text), surface)
    with pytest.raises(HD.HumanDataError, match="unknown human-data surface"):
        HD.honoured_or_refuse(HD.parse("off"), "search")


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
    assert HD.record(HD.parse("kl:hom@1"))["model"] == "hom"
