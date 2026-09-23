"""gen3_untaught_teacher_guard_v1 — a pinned trainee team must not be a member of the UNTAUGHT 8.

WHY THIS TEST EXISTS. The untaught 8 is the campaign's PRIMARY endpoint: off-slice competence,
measured on teams the model was not taught. On 2026-09-20 two of fifteen approved 5-team teacher
sets were untaught-8 members, and what caught it was a shard log happening to print the team ids
it was about to play — luck, one step before ~14 GPU-hours of contaminated teachers. Across the
run archive, 5 of 89 runs with a pinned trainee source would fire this guard, so the collision
class is real and recurrent rather than a one-off.

Every case below FAILS on a revert of the guard.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from agents.training.matchup_spec import (
    UNTAUGHT_MANIFEST,
    MatchupSpec,
    TeamSource,
    untaught_team_shas,
    validate_trainee_not_untaught,
)
from utils.paths import repo_path


def _untaught_exports(n: int = 2) -> "list[tuple[str, str]]":
    """(path, export) for the first `n` untaught-8 teams, straight from the committed manifest."""
    rels = json.loads(repo_path(UNTAUGHT_MANIFEST).read_text())["untaught"][:n]
    return [(rel, repo_path(rel).read_text()) for rel in rels]


def _spec(trainee: TeamSource) -> MatchupSpec:
    return MatchupSpec(trainee_teams=trainee, opponent_teams=TeamSource(kind="pool"))


def test_manifest_shas_are_read_and_nonempty():
    shas = untaught_team_shas()
    assert len(shas) == 8, f"the untaught slice is 8 teams, read {len(shas)}"
    assert all(len(s) == 10 for s in shas), "fingerprints are sha1[:10], as the sample gate uses"


def test_single_pin_on_an_untaught_team_raises():
    (rel, export), = _untaught_exports(1)
    with pytest.raises(ValueError) as e:
        validate_trainee_not_untaught(_spec(
            TeamSource(kind="pinned", pin_str=export, pin_file=rel)))
    assert "UNTAUGHT 8" in str(e.value) and rel in str(e.value)


def test_multi_pin_names_every_offending_member():
    pairs = _untaught_exports(2)
    with pytest.raises(ValueError) as e:
        validate_trainee_not_untaught(_spec(TeamSource(
            kind="pin_multi",
            pin_strs=tuple(x for _, x in pairs),
            pin_files=tuple(p for p, _ in pairs))))
    msg = str(e.value)
    assert msg.startswith("2 pinned trainee team(s)")
    for rel, _ in pairs:
        assert rel in msg, f"{rel} not named in the refusal"


def test_a_renamed_copy_is_still_caught():
    """Matching is by CONTENT sha, so the guard is not defeated by copying the file elsewhere."""
    (_, export), = _untaught_exports(1)
    with pytest.raises(ValueError) as e:
        validate_trainee_not_untaught(_spec(TeamSource(
            kind="pinned", pin_str=export, pin_file="data/teams/sample/not_the_same_name.txt")))
    assert "UNTAUGHT 8" in str(e.value)


def test_whitespace_differences_do_not_evade_the_guard():
    (_, export), = _untaught_exports(1)
    with pytest.raises(ValueError):
        validate_trainee_not_untaught(_spec(
            TeamSource(kind="pinned", pin_str="\n\n" + export + "\n  \n", pin_file="x.txt")))


def test_a_team_outside_the_slice_passes():
    """The era-1 stall pin is a real teacher team and is NOT in the manifest — it must pass."""
    clean = repo_path("data/teams/sample/6d56d06b9fca508e.txt").read_text()
    assert hashlib.sha1(clean.strip().encode()).hexdigest()[:10] not in untaught_team_shas()
    validate_trainee_not_untaught(_spec(
        TeamSource(kind="pinned", pin_str=clean, pin_file="data/teams/sample/6d56d06b9fca508e.txt")))


@pytest.mark.parametrize("kind", ["pool", "default_biased"])
def test_an_unpinned_trainee_is_out_of_scope(kind):
    validate_trainee_not_untaught(_spec(TeamSource(kind=kind)))


def test_the_three_era1_teacher_pins_are_all_clean():
    """Nothing already banked is affected — stated in the ledger, asserted here."""
    shas = untaught_team_shas()
    for h in ("f6229d2c867e21d6", "9eb3abdc52876a63", "6d56d06b9fca508e"):
        export = repo_path(f"data/teams/sample/{h}.txt").read_text()
        assert hashlib.sha1(export.strip().encode()).hexdigest()[:10] not in shas, h


def test_the_two_substituted_era2_teams_are_clean_and_the_originals_were_not():
    """The 2026-09-20 substitution, asserted both ways so a silent revert is caught."""
    shas = untaught_team_shas()

    def sha(h):
        # `0972146213a667c9` was relocated to data/teams/superseded/ (2026-09-23), byte-identical
        from utils.team_loader.relocations import resolve_team_file
        path = resolve_team_file(str(repo_path(f"data/teams/sample/{h}.txt")), quiet=True)
        return hashlib.sha1(open(path).read().strip().encode()).hexdigest()[:10]

    for was in ("9909f2e98e981ccc", "f7ba5702fe856292"):
        assert sha(was) in shas, f"{was} should still be an untaught-8 member"
    for now in ("9f27f5d3e34021a7", "0972146213a667c9"):
        assert sha(now) not in shas, f"{now} is the substitution and must be outside the slice"
