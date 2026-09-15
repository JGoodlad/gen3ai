"""THE CRN PAIRING, ON THE REAL BRIDGE (`gen3_fork_v1`, `--fork-crn`).

`sim`-marked: spawns the real in-process sim bridge (no Showdown server). It proves the claim the
fork arm's default rests on — **two branches that take identical actions produce byte-identical
protocol** — and, just as importantly, that pairing the DICE alone does not get you there.

Both players draw from a PER-INSTANCE decision stream seeded by `agents.training.fork_crn` — the
shape `RLPlayer._policy_generator` gives the trained policy, here in poke-env's `_choice_rng`. A
uniform policy stands in for the trained one on purpose: what is being tested is the PLUMBING of the
pairing, and a fixed distribution makes a divergence attributable to the stream rather than to any
weights. The torch half of the mechanism is pinned arithmetically in `fork_crn_test`.

Three cells, and the third is what makes the first two mean anything:

* same sim seed + same policy seeds  -> IDENTICAL protocol   (``dice_and_draws``)
* same sim seed + DIFFERENT policy seeds -> the streams diverge (``dice``: the `cf_q_labels`
  regime, i.e. two "paired" arms that differ in every coin either side flipped)
* DIFFERENT sim seed -> the streams diverge (the control that proves this test can SEE a change)
"""
from __future__ import annotations

import asyncio
import random

import pytest

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer

from agents.training.fork_crn import branch_policy_seed
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader.loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

pytestmark = pytest.mark.sim


def _player(name, team, seed):
    """A `RandomPlayer` whose decision stream is PER-INSTANCE and seeded.

    poke-env's `_choice_rng` (`gen3_player_choice_rng_v1`) is the same shape as `RLPlayer`'s
    `_policy_generator` (`gen3_policy_sample_rng_v1`) — a private stream a player draws every
    decision from — and it is what `fork_crn.player_seeds` seeds for a branch. The torch half of
    the mechanism is pinned arithmetically in `fork_crn_test`; what needs the REAL BRIDGE is the
    end-to-end claim, and a uniform policy makes a divergence attributable to the stream rather
    than to any weights.

    ``seed=None`` leaves the shared global `random` module, which is poke-env's documented
    unseeded behaviour and the regime a factory that pairs only the dice runs in.
    """
    p = RandomPlayer(
        battle_format="gen3ou", team=Gen3Teambuilder(team),
        account_configuration=AccountConfiguration(name, None),
        start_listening=False, start_timer_on_battle_start=False)
    if seed is not None:
        p._choice_rng = random.Random(int(seed))
    return p


def _protocol(sim_seed, p1_seed, p2_seed, tag):
    all_teams = TeamLoader().get_sample_teams() or TeamLoader().get_all_teams()
    # ONE PINNED TEAM PER SIDE. A multi-team builder draws per battle from a stream this test does
    # not control, and an unpinned team draw is exactly the kind of unpaired randomness the arm's
    # CRN exists to remove — it would make every cell differ for a reason that is not the pairing.
    t1, t2 = [all_teams[0]], [all_teams[1 % len(all_teams)]]
    sink: list = []

    async def go():
        p1 = _player(f"CrnA{tag}", t1, p1_seed)
        p2 = _player(f"CrnB{tag}", t2, p2_seed)
        await run_local_battles(p1, p2, 1, battle_format="gen3ou", seed=sim_seed,
                                chunk_sink=sink, impl="rust")

    asyncio.run(go())
    # The usernames are per-cell (poke-env keys its battles by account), so they are stripped
    # before the bytes are compared — otherwise every cell would differ for a reason that has
    # nothing to do with the pairing.
    text = "\n".join(chunk for _side, chunk in sink)
    return text.replace(f"CrnA{tag}", "P1").replace(f"CrnB{tag}", "P2")


SEED = [11, 22, 33, 44]


def test_identical_actions_give_BYTE_IDENTICAL_protocol():
    """`dice_and_draws`: one sim seed AND one policy stream per side. This is the property the
    whole arm's pairing rests on — two branches differ in exactly one thing, the action at the
    fork, and here they differ in nothing at all."""
    s1 = branch_policy_seed("fork:1", "trainee")
    s2 = branch_policy_seed("fork:1", "opponent")
    a = _protocol(SEED, s1, s2, "a")
    b = _protocol(SEED, s1, s2, "b")
    assert a and len(a) > 500, "the battle produced no protocol to compare"
    assert a == b


def test_pairing_the_DICE_alone_is_not_enough():
    """The `cf_q_labels` account, reproduced: same sim seed, unpaired policy draws, and the two
    lines are different games. That is why `--fork-crn dice_and_draws` is the default and `dice`
    is only the control."""
    s1 = branch_policy_seed("fork:1", "trainee")
    s2 = branch_policy_seed("fork:1", "opponent")
    paired = _protocol(SEED, s1, s2, "c")
    unpaired = _protocol(SEED, branch_policy_seed("fork:2", "trainee"),
                         branch_policy_seed("fork:2", "opponent"), "d")
    assert paired != unpaired


def test_the_comparison_can_see_a_change_at_all():
    """The control for the control: a different SIM seed must also diverge, or the two assertions
    above would pass on a comparison that cannot fail."""
    s1 = branch_policy_seed("fork:1", "trainee")
    s2 = branch_policy_seed("fork:1", "opponent")
    assert _protocol(SEED, s1, s2, "e") != _protocol([99, 88, 77, 66], s1, s2, "f")
