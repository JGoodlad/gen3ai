"""F-1b-2, the team-memory exclusion: a battle where some side shows >= 4 mons fitting one heldout72 team is dropped,
both seats, before anything is read. Each test fails when its rule is reverted."""
from __future__ import annotations

import collections

import pytest

from main.x50_r1 import memory as M
from utils.paths import repo_path

MANIFEST = repo_path("designs", "research_state", "measurements", "heldout72_hpfix_kakuna_2026-10-18",
                     "heldout72_hpfix_manifest.json")
HEAD = ["|player|p1|alice||1600", "|player|p2|bob||1550", "|teamsize|p1|6", "|teamsize|p2|6", "|start",
        "|switch|p1a: Skarmory|Skarmory, M|100/100"]
#: four of team_000389's mons (Medicham, Gyarados, Blastoise, Metagross), each revealing facts its set holds
FITS = ["|switch|p2a: Medicham|Medicham, M|100/100", "|move|p2a: Medicham|Brick Break|p1a: Skarmory",
        "|-enditem|p2a: Medicham|Lum Berry|[eat]",
        "|switch|p2a: Gyarados|Gyarados, F|100/100", "|-ability|p2a: Gyarados|Intimidate|boost",
        "|move|p2a: Gyarados|Hidden Power|p1a: Skarmory",
        "|switch|p2a: Blastoise|Blastoise, M|100/100", "|move|p2a: Blastoise|Rapid Spin|p1a: Skarmory",
        "|switch|p2a: Metagross|Metagross|100/100", "|move|p2a: Metagross|Meteor Mash|p1a: Skarmory"]


@pytest.fixture(scope="module")
def matcher():
    return M.HeldoutMatcher(MANIFEST)


def test_four_fitting_mons_of_one_heldout_team_match(matcher):
    assert len(matcher.teams) == 72
    assert ("p2", "team_000389", 4) in matcher.matches(HEAD + FITS)
    assert not any(seat == "p1" for seat, _, _ in matcher.matches(HEAD + FITS))


def test_a_revealed_fact_off_the_set_breaks_the_fit(matcher):
    off_move = [ln.replace("Meteor Mash", "Agility") for ln in FITS]
    assert not any(t == "team_000389" for _, t, _ in matcher.matches(HEAD + off_move))
    off_item = [ln.replace("Lum Berry", "Leftovers") for ln in FITS]
    assert not any(t == "team_000389" for _, t, _ in matcher.matches(HEAD + off_item))
    three = FITS[:-2]
    assert matcher.side_fits(HEAD + three, "p2")["team_000389"] == 3
    assert not any(t == "team_000389" for _, t, _ in matcher.matches(HEAD + three))


def test_the_extractor_drops_a_matching_battle_both_seats_before_reading(tmp_path, monkeypatch):
    from main.x50_r1 import run

    log = tmp_path / "battle-gen3ou-1.log"
    log.write_text("\n".join(HEAD + FITS + ["|turn|1"]) + "\n")
    monkeypatch.setattr(run, "EXCLUDE_TEAMS", str(MANIFEST))
    monkeypatch.setattr(run, "_MATCHER", None)
    out = run.human_job((0, str(log)))
    assert out["parts"] == [] and out["counts"]["excluded:team_memory"] == 1
    assert out["counts"]["team_memory:team_000389"] == 1 and out["memory"][0] == str(log)
    assert isinstance(out["counts"], collections.Counter)


def test_training_dates_are_an_inclusive_range(tmp_path):
    from main.x50_r1 import run

    (tmp_path / run.MANIFEST).write_text("2026-05-17/a.log\n2026-05-18/b.log\n2026-09-20/c.log\n2026-09-21/d.log\n")
    got = [p.rsplit("/", 2)[1] for p in run.human_paths(tmp_path, ["2026-05-18..2026-09-20"])]
    assert got == ["2026-05-18", "2026-09-20"]
