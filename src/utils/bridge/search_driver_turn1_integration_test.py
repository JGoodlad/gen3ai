"""The FIRST decision of a battle must open on BOTH offline-driver impls
(``gen3_search_turn1_open_v1``).

The rust ``search_driver`` used to refuse turn 1 on both verb families: ``at_turn_start``
compared ``BattleState::turn``, which still reads ``0`` at the pre-commit first boundary, so
``build_to_turn`` walked the entire command log and reported *"battle never reached the start
of turn 1"*. Node opened it fine — so this was a silent, impl-specific coverage hole costing
one decision per battle (~3.35% of move decisions, and the one decision every battle has).

**Why this lives in its own `sim` module rather than in `bridge_impl_parity_test.py`.** That
file is the natural *semantic* home, but it is marked `slow` for its 12-battle series, and the
routine gate is `-m "not slow and not e2e"`. This test plays exactly ONE battle; per the cost
rule in the root `CLAUDE.md` ("cost tracks battle COUNT, not 'does it battle'"), it is `sim`
and NOT `slow`, so the regression is caught by the routine gate instead of only before a ship.

The assertions are impl-SYMMETRIC on purpose: the property is "turn 1 opens", not "rust
matches node", so node's arm would also catch a regression that broke the shared seam.
"""

import asyncio
import os
import random
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

from poke_env import AccountConfiguration

from utils.bridge.local_battle_runner import run_local_battles
from utils.bridge.reconstruction import pop_record, reroll_turn
from utils.team_loader.loader import TeamLoader

pytestmark = pytest.mark.sim

BATTLE_FORMAT = "gen3ou"
_SEED = [11, 22, 33, 44]
_IMPLS = ("node", "rust")


def _recorded_battle():
    """Play ONE seeded bridge battle and return its reconstruction record.

    Recorded on the NODE transport so the fixture is impl-neutral: a record produced by the
    implementation under test could hide a defect that is symmetric between the recorder and
    the replayer.
    """
    from poke_env.player import RandomPlayer

    random.seed(5)
    team = TeamLoader().get_all_teams()[0]
    p1, p2 = (
        RandomPlayer(battle_format=BATTLE_FORMAT, team=team, start_listening=False,
                     account_configuration=AccountConfiguration("T1OpenA", "password")),
        RandomPlayer(battle_format=BATTLE_FORMAT, team=team, start_listening=False,
                     account_configuration=AccountConfiguration("T1OpenB", "password")),
    )
    asyncio.run(run_local_battles(p1, p2, 1, battle_format=BATTLE_FORMAT,
                                  seed=_SEED, impl="node"))
    battle = next(iter(p1._battles.values()))
    # NON-VACUITY: a degenerate/aborted battle would make every assertion below pass for the
    # wrong reason — and turn >= 3 is what makes the turn-2 control a real second boundary.
    assert battle.finished and battle.turn >= 3, (
        f"degenerate fixture battle (finished={battle.finished}, turn={battle.turn})")
    record = pop_record(battle.battle_tag)
    assert record is not None, "the bridge emitted no __RECON__ — nothing to replay"
    return record


@pytest.fixture(scope="module")
def record():
    return _recorded_battle()


@pytest.mark.parametrize("impl", _IMPLS)
def test_turn_1_opens_on_both_impls(record, impl):
    """The regression: pre-fix this raised on ``impl="rust"`` with "battle never reached the
    start of turn 1", while ``impl="node"`` passed."""
    res = reroll_turn(record, 1, seeds=["7,7,7,7"], impl=impl)

    assert res.turn == 1, f"{impl}: reported turn {res.turn}, not the turn-1 boundary"
    # BOTH sides must be on a real move request — that is what makes turn 1 a joint decision
    # rather than something the driver merely failed to reject.
    assert set(res.requests) >= {"p1", "p2"}, f"{impl}: missing a choice surface: {res.requests}"
    assert res.recorded_choices.get("p1") and res.recorded_choices.get("p2"), (
        f"{impl}: turn 1 must name BOTH original picks, got {res.recorded_choices}")
    assert res.rerolls, f"{impl}: no arm was resolved at turn 1"


@pytest.mark.parametrize("impl", _IMPLS)
def test_turn_2_still_opens_on_both_impls(record, impl):
    """The control. The fix maps a pre-commit ``turn == 0`` to 1 and is the IDENTITY for every
    later turn — so a change that bought turn 1 by breaking turn 2 fails here."""
    res = reroll_turn(record, 2, seeds=["7,7,7,7"], impl=impl)
    assert res.turn == 2, f"{impl}: reported turn {res.turn}, not the turn-2 boundary"
    assert res.rerolls, f"{impl}: no arm was resolved at turn 2"


def test_turn_1_prefix_is_the_battle_start_on_both_impls(record):
    """A turn-1 prefix is the CONSTRUCTION protocol and nothing more — no turn has committed.
    Pinning it here is what stops a future "fix" from opening turn 1 by silently building past
    it, which would still satisfy ``res.turn == 1``."""
    per_impl = {}
    for impl in _IMPLS:
        res = reroll_turn(record, 1, seeds=["7,7,7,7"], impl=impl)
        joined = "\n".join(res.prefix_p1_chunks)
        assert "|turn|2" not in joined, (
            f"{impl}: the turn-1 prefix ran past turn 1 — it contains |turn|2")
        assert "|player|p1" in joined, f"{impl}: the turn-1 prefix is missing battle start"
        per_impl[impl] = res.recorded_choices

    assert per_impl["node"] == per_impl["rust"], (
        f"the two impls disagree on turn 1's original picks: {per_impl}")
