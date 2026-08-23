"""Every team the trainer can actually be dealt must be LEGAL gen3ou.

Validated against the Showdown team validator through the local Node bridge — no server, and an
INDEPENDENT oracle (the sim's own rules, not our loader's opinion of them).

🚨 This file SKIPPED ON EVERY TREE, FOREVER, until 2026-08-23. It opened on
``data/teams/teams.json``, a manifest that does not exist and has not for as long as the current
layout has (the real manifests are ``data/teams/sample/teams.json`` and
``data/teams/others/*/teams.json``), and on the miss it called ``pytest.skip`` with a message
blaming the operator — "Run sync-teams first." Three defects were stacked:

  1. the manifest path was stale        -> skipped unconditionally;
  2. every path was CWD-relative        -> would have broken off the repo root even if (1) were
                                           fixed, the relative-path sibling of the ``/home/...``
                                           literal class ``utils.paths`` closed;
  3. a missing team FILE printed a warning and ``continue``d, so even on the happy path a layout
     change would have validated ZERO teams and still reported success.

All three are the same mistake in three costumes: a branch between the runner and the assertion.
The rewrite removes the branches — it asks ``TeamLoader`` where the teams are (the same seam the
trainer asks), validates every one of them, and asserts a COUNT FLOOR so "validated nothing"
cannot read as "validated everything".

Measured 2026-08-23: 719 loaded teams, 0 invalid, **1.2 s** via the batch entry point
(``validate_teams_locally`` spawns one Node process for the whole pool; the per-team
``validate_team_locally`` costs ~0.58 s each and would have made this a `slow` test).
"""
import pytest

from utils.bridge.team_validator import validate_teams_locally
from utils.team_loader import TeamLoader

# The pool has been 719 since the yak_attack de-weighting + dedupe (root CLAUDE.md). The floor is
# deliberately well below it: this guards "the loader returned essentially nothing", not the exact
# count, which is a data fact and not this test's business.
_MIN_POOL = 600


@pytest.mark.integration
def test_every_team_the_trainer_can_be_dealt_is_legal_gen3ou():
    teams = TeamLoader().get_all_teams()

    # ASSERT THE BUILD. A zero-length pool would otherwise walk the loop below and pass.
    assert len(teams) >= _MIN_POOL, (
        f"TeamLoader returned {len(teams)} teams, expected >= {_MIN_POOL} — the pool the trainer "
        f"samples from is empty or nearly so, and the legality check below would have validated "
        f"nothing while reporting success"
    )

    results = validate_teams_locally("gen3ou", list(teams))
    assert len(results) == len(teams), (
        f"validator returned {len(results)} results for {len(teams)} teams — the batch bridge "
        f"dropped some, so a silent subset was checked"
    )

    invalid = [(i, r.get("errors", [])) for i, r in enumerate(results) if not r["valid"]]
    assert not invalid, (
        f"{len(invalid)} of {len(teams)} pool teams are ILLEGAL gen3ou — the trainer can be dealt "
        f"these:\n" + "\n".join(f"  - team #{i}: {errs[:3]}" for i, errs in invalid[:10])
    )


@pytest.mark.integration
def test_the_legality_check_can_actually_fail():
    """The vacuity guard: prove the oracle says NO to something.

    Without this, a validator bridge that returned ``valid: True`` unconditionally — a JSON shape
    change, a swallowed Node error — would make the gate above green forever. `paths_test.py`'s
    `test_the_scan_can_actually_fail` is the precedent.
    """
    # A single mon with no moves: the validator rejects this and has no way not to.
    bogus = "Swampert @ Leftovers\nAbility: Torrent\nEVs: 252 HP\n"
    [result] = validate_teams_locally("gen3ou", [bogus])
    assert not result["valid"], (
        "the team validator accepted a moveless single-mon team — the oracle is not answering, so "
        "the pool-legality gate above proves nothing"
    )
