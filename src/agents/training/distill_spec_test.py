"""--distill-teacher spec parsing: multi-team teacher groups + legacy pair back-compat."""
import pytest

from agents.training.distill_spec import parse_distill_teacher_spec as P


def test_single_teacher_single_team():
    assert P("models/T1:a.txt") == [("models/T1", ["a.txt"])]


def test_one_teacher_many_teams():
    # the multi-team exploiter case: ONE frozen model piloting a whole z-cluster
    assert P("models/T1:a.txt,b.txt,c.txt") == [("models/T1", ["a.txt", "b.txt", "c.txt"])]


def test_semicolon_separates_teachers():
    assert P("models/T1:a.txt,b.txt;models/T2:c.txt") == [
        ("models/T1", ["a.txt", "b.txt"]),
        ("models/T2", ["c.txt"]),
    ]


def test_legacy_comma_pairs_still_parse():
    # BACK-COMPAT: a comma segment that itself contains ':' starts a NEW teacher, so the old
    # 'T1:a,T2:b' pair form is unambiguous (a team path never contains a colon).
    assert P("models/T1:a.txt,models/T2:b.txt") == [
        ("models/T1", ["a.txt"]),
        ("models/T2", ["b.txt"]),
    ]


def test_whitespace_and_empty_segments_tolerated():
    assert P("  models/T1 : a.txt , b.txt ; models/T2 : c.txt ; ") == [
        ("models/T1", ["a.txt", "b.txt"]),
        ("models/T2", ["c.txt"]),
    ]


def test_order_is_preserved_teacher_id_is_position():
    # the teacher's 1-indexed position IS its teacher-id in the env's distill_mask
    got = P("A:1.txt;B:2.txt;C:3.txt")
    assert [t for t, _ in got] == ["A", "B", "C"]


def test_team_without_a_teacher_raises():
    with pytest.raises(ValueError, match="no teacher"):
        P("a.txt,b.txt")


def test_teacher_without_a_team_raises():
    with pytest.raises(ValueError, match="no team file"):
        P("models/T1:;models/T2:a.txt")


def test_empty_spec_is_empty():
    assert P("") == [] and P(None) == []


# ── the '*' / 'auto' wildcard: resolve a teacher's teams from ITS OWN provenance ────────────────

def _fake_resolver(teacher):
    return {"models/T1": ["a.txt", "b.txt", "c.txt"], "models/GEN": []}.get(teacher, [])


def test_wildcard_expands_from_provenance():
    assert P("models/T1:*", resolve_wildcard=_fake_resolver) == [("models/T1", ["a.txt", "b.txt", "c.txt"])]


def test_auto_is_the_globproof_spelling():
    assert P("models/T1:auto", resolve_wildcard=_fake_resolver) == P("models/T1:*", resolve_wildcard=_fake_resolver)


def test_wildcard_mixes_with_explicit_teachers():
    got = P("models/T1:*;models/T2:x.txt", resolve_wildcard=_fake_resolver)
    assert got == [("models/T1", ["a.txt", "b.txt", "c.txt"]), ("models/T2", ["x.txt"])]


def test_wildcard_on_a_generalist_raises():
    # a run that recorded NO trainee teams can't be expanded — fail loud rather than distil nothing
    with pytest.raises(ValueError, match="recorded NO trainee teams"):
        P("models/GEN:*", resolve_wildcard=_fake_resolver)


def test_wildcard_without_resolver_raises():
    with pytest.raises(ValueError, match="no resolver"):
        P("models/T1:*")


def test_wildcard_cannot_mix_with_explicit_teams():
    with pytest.raises(ValueError, match="mixes the"):
        P("models/T1:*,extra.txt", resolve_wildcard=_fake_resolver)
