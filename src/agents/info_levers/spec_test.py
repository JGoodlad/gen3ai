"""The information-lever SPEC: parsing, the canonical text the Rust core reads, the presets (the X32 levels and the
hidden-info study's arms), the hash, and each consumer's declared support."""
import json

import pytest

from agents.info_levers import consumers as C
from agents.info_levers import spec as S


def test_the_x32_levels_keep_their_names_as_labels_and_their_exact_canonical_text():
    assert S.parse("off").is_off and S.parse("off").label == "off" and S.parse("off").canonical == "off"
    assert S.parse("species").canonical == "species:unseen" and S.parse("species").label == "species"
    full = S.parse("full")
    assert full.canonical == "species:unseen,moves:both,item:both,ability:both,spread:both"
    assert full.label == "full"
    # the same levers written as clauses ARE the level (label and equality), whatever the spelling
    assert S.parse("spread,ability,item:both,moves:seen,moves:unseen,species:both").label == "full"
    assert S.same("full", "species, moves, item, ability, spread")
    assert not S.same("full", "species,moves,item,ability")


@pytest.mark.parametrize("text, want", [
    ("moves:seen@0.95", "moves:seen@0.95"),
    ("moves:seen@.5", "moves:seen@0.5"),
    ("moves:seen@0.5000", "moves:seen@0.5"),
    ("moves:seen@1", "moves:seen"),
    ("moves:seen@1.0~truth", "moves:seen"),
    ("moves:seen@0.9~placebo", "moves:seen@0.9~placebo"),
    ("moves@0.25", "moves:both@0.25"),
    ("moves:seen@0.25,moves:unseen@0.5", "moves:seen@0.25,moves:unseen@0.5"),
    ("moves:unseen,moves:seen", "moves:both"),
    ("item:seen~map,species", "species:unseen,item:seen~map"),
    ("hp", "hp:seen"), ("counters:both", "counters:seen"), ("species:both", "species:unseen"),
    ("moves:seen@0", "off"), ("moves:seen@0.0001", "moves:seen@0.0001"),
])
def test_canonical_text(text, want):
    s = S.parse(text)
    assert s.canonical == want
    assert S.parse(s.canonical) == s, "the canonical text parses back to the same levers"
    assert S.parse(s.canonical).canonical == want


@pytest.mark.parametrize("bad", [
    "moves:seen@1.5", "moves:seen@-0.1", "moves:seen@0.12345", "moves:seen@x", "move:seen", "moves:sean",
    "moves:seen~oracle", "hp:unseen", "species:seen", "", "moves,,item", "moves:seen,moves:seen@0.5",
    "TRUE", "TRUE@2",
])
def test_bad_text_is_refused_by_name(bad):
    with pytest.raises(S.InfoRevealError):
        S.parse(bad)


def test_the_hash_is_over_the_canonical_text_and_stable():
    assert S.parse("full").hash == S.parse("species,moves,item,ability,spread").hash
    assert S.parse("off").hash != S.parse("species").hash
    # pinned: the hash a run records must not move under a refactor
    assert {n: S.parse(n).hash for n in ("off", "species", "full", "F+M")} == {
        "off": "80d7064adfd9afc7", "species": "677b8948e6f4a7ab", "full": "57bfe6fa79c65a86",
        "F+M": "9ccb7440397eebca"}
    rec = S.parse("moves:seen@0.95").record()
    assert rec["label"] == rec["canonical"] == "moves:seen@0.95" and rec["levers"] == [
        {"class": "moves", "part": "seen", "mass": 0.95, "source": "truth"}]
    assert json.loads(S.dumps("species"))["label"] == "species"


def test_the_study_presets_compose_from_its_five_class_partition():
    P = S.PRESETS
    assert P["F"] == P["off"] and P["O"] == P["oracle"]
    assert P["F+M"].canonical == "moves:seen"
    assert P["F+U"].canonical == "species:unseen,moves:unseen,item:unseen,ability:unseen,spread:unseen"
    assert P["F+Usp"] == P["species"]
    assert P["F+S"].canonical == "spread:seen,hp:seen"
    assert P["O-M"].canonical == "species:unseen,moves:unseen,item:both,ability:both,spread:both,hp:seen,counters:seen"
    assert P["O-U"].canonical == "moves:seen,item:seen,ability:seen,spread:seen,hp:seen,counters:seen"
    assert P["pilot"].canonical == "item:seen,ability:seen,spread:seen,hp:seen,counters:seen"
    # the five classes partition the hidden facts except `counters` (O alone tells it)
    union = S.InfoReveal()
    for x in S.STUDY_CLASSES.values():
        union = union.plus(x)
    assert P["O"].minus(union).canonical == "counters:seen"
    assert P["D90"].canonical == "moves:seen@0.9" and P["W100"].canonical == "moves:seen~placebo"
    assert P["MAP90"].canonical == "moves:seen@0.9~map"
    assert S.preset("TRUE@0.95").canonical == "moves:seen@0.95"
    assert S.preset("TRUEHYP@1.0").canonical == "moves:both"
    assert S.preset("SELFMAP@0.8").canonical == "moves:seen@0.8~map"
    assert S.parse("PLACEBO@0.95").canonical == "moves:seen@0.95~placebo"
    with pytest.raises(S.InfoRevealError):
        S.preset("nope")


def test_lever_queries():
    s = S.parse("moves:seen@0.9~placebo,item")
    lv = s.lever("moves", "seen")
    assert lv is not None and lv.mass == 0.9 and lv.source == "placebo"
    assert s.lever("moves", "unseen") is None and s.lever("item", "unseen") is not None
    assert s.classes() == ("moves", "item") and s.sources() == ("truth", "placebo")
    assert list(S.facts())[0] == ("species", "unseen") and len(list(S.facts())) == 11


def test_the_observation_consumer_refuses_what_the_encoder_cannot_state():
    for ok in ("off", "species", "full", "moves:seen@0.5", "species,moves:unseen", "spread:seen"):
        C.OBS.validate(ok)
    for bad, word in [("hp", "exact"), ("counters", "duration"), ("moves:seen~placebo", "source"),
                      ("item:seen~map", "source"), ("moves:unseen", "species:unseen"), ("oracle", "exact")]:
        with pytest.raises(S.InfoRevealError, match=word):
            C.OBS.validate(bad)
    assert C.OBS.residual_of("off") == ()


def test_the_search_consumer_honours_every_class_and_names_its_residual():
    for name in S.PRESETS:
        C.SEARCH.validate(name)
    assert C.SEARCH.residual_of("off") == (("counters", "seen"),)
    assert C.SEARCH.residual_of("oracle") == ()
    with pytest.raises(S.InfoRevealError, match="source"):
        C.SEARCH.validate("hp~placebo")


def test_the_injection_consumer():
    for ok in ("TRUE@0.95", "PLACEBO@0.95", "SELFMAP@1.0", "TRUEHYP@0.95", "item:seen@0.9~placebo", "spread:seen"):
        C.INJECT.validate(ok)
    for bad in ("ability:seen", "species", "hp", "moves:unseen~placebo", "spread:seen~map"):
        with pytest.raises(S.InfoRevealError):
            C.INJECT.validate(bad)
