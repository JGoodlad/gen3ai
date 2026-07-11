"""Team-archetype classifier tests — known-team pins + feature extraction + artifact shape."""
import pytest

from agents.training.team_archetypes import (
    ARCHETYPE_CLASSES, classify, classify_team, extract_features, pace_score, style_tags,
    team_sha,
)

_STALL_MON = """{name} @ Leftovers
Ability: {ability}
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- {m1}
- {m2}
- {m3}
- {m4}
"""


def _team(*mons):
    return "\n".join(mons)


@pytest.fixture(scope="module")
def tss():
    with open("data/teams/specialist/tss_starmie.txt", encoding="utf-8") as f:
        return f.read()


def test_tss_features_and_tags(tss):
    f = extract_features(tss)
    assert f.n_mons == 6 and f.has_sand and f.n_spikes == 1 and f.has_spin and f.has_ghost
    assert f.n_phaze >= 1 and f.n_recovery >= 1
    tags = style_tags(f)
    for t in ("sand", "spikes", "spin", "spinblock", "phaze", "status_heavy"):
        assert t in tags, tags
    assert "trap" not in tags
    # canonical TSS is bulky ATTRITION — mid-spectrum by design, never hyper_offense
    assert classify(f) in ("balance", "semi_stall", "offense")


def test_full_stall_classifies_stall():
    team = _team(
        _STALL_MON.format(name="Blissey", ability="Natural Cure",
                          m1="Soft-Boiled", m2="Toxic", m3="Seismic Toss", m4="Protect"),
        _STALL_MON.format(name="Skarmory", ability="Keen Eye",
                          m1="Spikes", m2="Rest", m3="Whirlwind", m4="Toxic"),
        _STALL_MON.format(name="Milotic", ability="Marvel Scale",
                          m1="Recover", m2="Surf", m3="Toxic", m4="Refresh"),
        _STALL_MON.format(name="Weezing", ability="Levitate",
                          m1="Will-O-Wisp", m2="Pain Split", m3="Sludge Bomb", m4="Rest"),
        _STALL_MON.format(name="Dusclops", ability="Pressure",
                          m1="Rest", m2="Night Shade", m3="Will-O-Wisp", m4="Seismic Toss"),
        _STALL_MON.format(name="Flygon", ability="Levitate",
                          m1="Earthquake", m2="Toxic", m3="Protect", m4="Rest"),
    )
    f = extract_features(team)
    assert f.n_recovery >= 5 and f.n_setup == 0
    assert classify(f) == "stall"
    assert pace_score(f) < 0


def test_hyper_offense_classifies_fast_and_setup():
    team = _team(
        _STALL_MON.format(name="Tyranitar", ability="Sand Stream",
                          m1="Dragon Dance", m2="Rock Slide", m3="Earthquake", m4="Hidden Power Bug"),
        _STALL_MON.format(name="Salamence", ability="Intimidate",
                          m1="Dragon Dance", m2="Hidden Power Flying", m3="Earthquake", m4="Rock Slide"),
        _STALL_MON.format(name="Metagross", ability="Clear Body",
                          m1="Agility", m2="Meteor Mash", m3="Earthquake", m4="Explosion"),
        _STALL_MON.format(name="Aerodactyl", ability="Rock Head",
                          m1="Rock Slide", m2="Earthquake", m3="Hidden Power Flying", m4="Double-Edge"),
        _STALL_MON.format(name="Gengar", ability="Levitate",
                          m1="Thunderbolt", m2="Ice Punch", m3="Explosion", m4="Fire Punch"),
        _STALL_MON.format(name="Jolteon", ability="Volt Absorb",
                          m1="Thunderbolt", m2="Hidden Power Ice", m3="Baton Pass", m4="Substitute"),
    )
    f = extract_features(team)
    assert f.n_setup == 3 and f.n_boom == 2 and f.n_fast >= 3
    assert classify(f) == "hyper_offense"
    assert "setup_heavy" in style_tags(f) and "boom" in style_tags(f)


def test_trap_tag_on_magneton_dugtrio():
    team = _team(
        _STALL_MON.format(name="Magneton", ability="Magnet Pull",
                          m1="Thunderbolt", m2="Hidden Power Fire", m3="Substitute", m4="Toxic"),
        _STALL_MON.format(name="Dugtrio", ability="Arena Trap",
                          m1="Earthquake", m2="Rock Slide", m3="Hidden Power Bug", m4="Aerial Ace"),
        _STALL_MON.format(name="Blissey", ability="Natural Cure",
                          m1="Soft-Boiled", m2="Toxic", m3="Ice Beam", m4="Protect"),
    )
    f = extract_features(team)
    assert f.n_trappers == 2
    assert "trap" in style_tags(f)


def test_classify_team_record_shape(tss):
    rec = classify_team(tss)
    assert rec["archetype"] in ARCHETYPE_CLASSES
    assert isinstance(rec["pace_score"], float) and isinstance(rec["tags"], list)
    assert len(rec["species"]) == 6 and rec["features"]["n_mons"] == 6


def test_team_sha_is_strip_normalized(tss):
    # The artifact keys hash the STRIPPED export (what TeamLoader feeds the samplers). The
    # MatchupSpec pin_sha of the same team file hashes RAW bytes (trailing newline) and DIFFERS —
    # the documented convention gap; joining a pin requires stripping first.
    import hashlib
    assert team_sha(tss) == hashlib.sha1(tss.strip().encode()).hexdigest()[:10]
    assert team_sha(tss) == team_sha(tss + "\n\n")          # normalization
    raw_pin_sha = hashlib.sha1(tss.encode()).hexdigest()[:10]
    assert raw_pin_sha == "4c01c7bbbb" and team_sha(tss) != raw_pin_sha


def test_unparseable_team_fails_loud():
    with pytest.raises(ValueError):
        extract_features("")
