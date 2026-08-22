"""Prefix-sharing parity — ``materialize_branches`` ≡ per-arm ``materialize_decisions``, BIT-FOR-BIT.

:func:`agents.training.obs_materializer.materialize_branches` replays a decision's shared
prefix ONCE and restores a snapshot of the player's battle/tracker state per arm, instead
of replaying the whole battle from turn 1 for every arm. That is purely a PERFORMANCE
change, so the per-arm observation must not drift by a single byte — and "not by a byte"
is the only threshold this tree accepts for an obs (``obs_roundtrip_fuzz_test``,
``reroll_many_parity_fuzz_test``, ``search_clone_parity_fuzz_test`` all hold that line).

The gate is deliberately broad in the ONE dimension that matters for the snapshot: it
compares **every** arm, not a sample. The clone SHARES append-only immutable records
(``BattleEvent``, ``BattleContext``) rather than copying them — a contract, not an
inference — and the failure mode of a broken contract is that arm 2+ reads history arm 1
mutated. Only an all-arms comparison sees that.

Real battles, in-process via the local bridge (no server), the fuzz-test pattern: play a
battle, take its reconstruction record, re-roll a mid-battle decision into K arms, then
materialize those arms both ways and compare.

    pytest -m sim src/agents/training/obs_materializer_branch_integration_test.py
    python src/agents/training/obs_materializer_branch_integration_test.py [n_battles]
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time

import numpy as np
import pytest

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

import agents.training.obs_materializer as OM
from agents.training.obs_materializer import (Branch, materialize_branches,
                                              materialize_decisions, materialize_from_record)
from agents.training.obs_roundtrip_fuzz_test import RecordingFuzzPlayer
from main.prober.engine import _has_state
from utils.bridge.local_battle_runner import run_local_battles
from utils.bridge.reconstruction import ReconstructionRecord, reroll_many
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

pytestmark = pytest.mark.sim

BATTLE_FORMAT = "gen3ou"
MIN_TURN = 4          # branch late enough that the shared prefix is a real prefix


def record_one_battle(out_dir: str):
    ts = int(time.time() * 1000) % 100000
    pool = TeamLoader().get_all_teams()
    trainee = RecordingFuzzPlayer(
        out_dir=out_dir, rng_seed=ts, battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"PSt{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"PSo{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    asyncio.run(run_local_battles(trainee, opp, 1))
    prefix = trainee.trace_prefixes[0]
    record = ReconstructionRecord.load(f"{prefix}_reconstruction.json")
    with open(f"{prefix}_summary.json") as f:
        summary = json.load(f)
    with np.load(f"{prefix}_states.npz") as z:
        npz = {k: z[k] for k in z.files}
    return record, summary, npz


def _anchor(summary, npz):
    """The first mid-battle move decision that has a recorded successor state."""
    invs = summary["invocations"]
    for i, inv in enumerate(invs):
        if (inv.get("phase") == "move_selection" and int(inv["turn"]) >= MIN_TURN
                and i + 1 < len(invs) and _has_state(npz, i) and _has_state(npz, i + 1)):
            return i
    return None


def check_battle(record, summary, npz) -> int:
    """Assert the two materialization paths agree on every arm. Returns the # of arms gated."""
    anchor = _anchor(summary, npz)
    if anchor is None:
        return 0
    invs = summary["invocations"]
    turn = int(invs[anchor]["turn"])
    side = record.side_of(record.trainee_username)
    other = "p2" if side == "p1" else "p1"
    actions = np.asarray(npz["actions"], dtype=int)
    chosen = int(actions[anchor])

    trace = materialize_from_record(
        record, actions=actions, map_actions_at=anchor, stop_after_decision=anchor)
    choice_map = trace.action_choices or {}
    assert chosen in choice_map, f"{record.battle_tag}: chosen action not legal in replay"

    arms = [{f"{side}_action": ("recorded" if a == chosen else choice_map[a]),
             f"{other}_action": "recorded", "seed": "original", "label": int(a)}
            for a in choice_map]
    rr = reroll_many(record, turn, arms)
    prefix_chunks = list(rr.prefix_p1_chunks if side == "p1" else rr.prefix_p2_chunks)
    prefix_actions = [int(x) for x in actions[:anchor]]
    live = [a for a in rr.arms if not (a.outcome.get("ended") or a.outcome.get("stuck"))]
    if not live:
        return 0

    kw = dict(username=record.username(side), packed_team=record.packed_team(side), side=side,
              battle_format=record.format_id, battle_tag=record.battle_tag,
              stop_after_decision=anchor + 1)

    per_arm = [materialize_decisions(
        prefix_chunks + list(a.p1_chunks if side == "p1" else a.p2_chunks),
        actions=prefix_actions + [int(a.label)], **kw) for a in live]
    branches = [Branch(chunks=list(a.p1_chunks if side == "p1" else a.p2_chunks),
                       actions=[int(a.label)], label=int(a.label)) for a in live]
    shared = materialize_branches(prefix_chunks, branches, prefix_actions=prefix_actions, **kw)

    assert len(per_arm) == len(shared)
    n_gated = 0
    for arm, one, many in zip(live, per_arm, shared):
        assert len(one.decisions) == len(many.decisions), (
            f"{record.battle_tag} arm {arm.label}: decision COUNT differs "
            f"(per-arm {len(one.decisions)} vs shared {len(many.decisions)})")
        assert one.actions_complete == many.actions_complete
        for k, (da, db) in enumerate(zip(one.decisions, many.decisions)):
            assert da.turn == db.turn, f"{record.battle_tag} arm {arm.label} decision {k}: turn"
            assert np.array_equal(da.mask, db.mask), (
                f"{record.battle_tag} arm {arm.label} decision {k}: action MASK differs")
            assert (da.obs is None) == (db.obs is None)
            if da.obs is not None:
                assert da.obs.tobytes() == db.obs.tobytes(), (
                    f"{record.battle_tag} arm {arm.label} decision {k} (turn {da.turn}): "
                    f"prefix-shared obs is NOT bit-identical to the per-arm obs "
                    f"({int(np.count_nonzero(da.obs != db.obs))} dims differ)")
        if len(one.decisions) > anchor + 1:
            n_gated += 1
    return n_gated


def test_shared_prefix_arms_are_bit_identical_to_per_arm_replay():
    with tempfile.TemporaryDirectory(prefix="prefix_share_parity_") as out_dir:
        record, summary, npz = record_one_battle(out_dir)
        gated = check_battle(record, summary, npz)
    assert gated >= 1, ("no arm reached the successor-obs gate — every branch ended the "
                        "battle? re-run (the opponent is a RandomPlayer, so this is rare)")


def test_a_branch_whose_prefix_does_not_reach_the_decision_is_refused():
    """The one silent-wrongness hazard: a caller whose prefix_actions and prefix_chunks
    disagree would branch from the WRONG state. It must raise, not guess."""
    with pytest.raises(RuntimeError, match="prefix_chunks and prefix_actions disagree"):
        materialize_branches(
            ["|start\n"], [Branch(chunks=[], actions=[0])],
            username="ghost", packed_team="", side="p1",
            prefix_actions=[0] * 99, battle_format=BATTLE_FORMAT)


def main(n_battles: int) -> None:
    print(f"prefix-share parity fuzz — {n_battles} battles", flush=True)
    total = 0
    for _ in range(n_battles):
        with tempfile.TemporaryDirectory(prefix="prefix_share_parity_") as out_dir:
            record, summary, npz = record_one_battle(out_dir)
            gated = check_battle(record, summary, npz)
            total += gated
            print(f"  {record.battle_tag}: {gated} arms bit-identical", flush=True)
    assert total >= 1
    print(f"\nPASS — {total} arms: prefix-shared materialization is byte-identical to per-arm")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)


# --------------------------------------------------------------------------------------
# The two cases the parity gate above EXCLUDES (adversarial review, 2026-08-22)
# --------------------------------------------------------------------------------------
# `check_battle` drops every arm whose re-roll `outcome.ended` is true and bounds the replay to
# ONE decision past the branch — and `lookahead`, the only production caller, filters `ended` the
# same way. So the restore path written specifically FOR a battle-ending arm (draining
# `_battle_count_queue`, which poke-env's `|win|` handler `.get()`s, and re-populating the
# `_trackers` / `_stall_loggers` entries `_battle_finished_callback` EVICTS) had no gate at all: a
# regression there would wedge the next arm on an empty queue FOREVER, or silently hand it a
# freshly-constructed tracker. This runs exactly that shape.

def _last_decision(summary, npz):
    """The LAST mid-battle move decision — where a single re-rolled turn actually ends battles."""
    best = None
    for i, inv in enumerate(summary["invocations"]):
        if (inv.get("phase") == "move_selection" and int(inv["turn"]) >= MIN_TURN
                and _has_state(npz, i)):
            best = i
    return best


class _TripwireSnapshot(OM._PlayerSnapshot):
    """Records a fingerprint of every object `_pin_shared` decided to SHARE rather than copy."""

    def __init__(self, player):
        super().__init__(player)
        self.fingerprints = {i: repr(o) for i, o in self._pins.items()}
        _TRIPWIRES.append(self)

    def drift(self):
        return [(type(o).__name__, self.fingerprints[i], repr(o))
                for i, o in self._pins.items() if repr(o) != self.fingerprints[i]]


_TRIPWIRES: list = []


def check_ending_arms(record, summary, npz) -> int:
    """Every arm INCLUDING the ones that end the battle, an ending arm FIRST, full suffix."""
    anchor = _last_decision(summary, npz)
    if anchor is None:
        return 0
    invs = summary["invocations"]
    turn = int(invs[anchor]["turn"])
    side = record.side_of(record.trainee_username)
    other = "p2" if side == "p1" else "p1"
    actions = np.asarray(npz["actions"], dtype=int)
    chosen = int(actions[anchor])

    trace = materialize_from_record(
        record, actions=actions, map_actions_at=anchor, stop_after_decision=anchor)
    choice_map = trace.action_choices or {}
    if chosen not in choice_map:
        return 0
    rr = reroll_many(record, turn, [
        {f"{side}_action": ("recorded" if a == chosen else choice_map[a]),
         f"{other}_action": "recorded", "seed": "original", "label": int(a)} for a in choice_map])
    keep = [a for a in rr.arms if not a.outcome.get("stuck")]
    ended = [a for a in keep if a.outcome.get("ended")]
    if not ended or len(keep) < 2:
        # The shape under test is "an ending arm, then ANOTHER arm". One arm proves nothing.
        return 0
    keep = ended[:1] + [a for a in keep if a is not ended[0]]     # an ENDING arm goes FIRST

    prefix_chunks = list(rr.prefix_p1_chunks if side == "p1" else rr.prefix_p2_chunks)
    prefix_actions = [int(x) for x in actions[:anchor]]
    kw = dict(username=record.username(side), packed_team=record.packed_team(side), side=side,
              battle_format=record.format_id, battle_tag=record.battle_tag)   # NO stop_after

    per_arm = [materialize_decisions(
        prefix_chunks + list(a.p1_chunks if side == "p1" else a.p2_chunks),
        actions=prefix_actions + [int(a.label)], **kw) for a in keep]
    branches = [Branch(chunks=list(a.p1_chunks if side == "p1" else a.p2_chunks),
                       actions=[int(a.label)], label=int(a.label)) for a in keep]

    _TRIPWIRES.clear()
    original = OM._PlayerSnapshot
    OM._PlayerSnapshot = _TripwireSnapshot
    try:
        shared = materialize_branches(prefix_chunks, branches,
                                      prefix_actions=prefix_actions, **kw)
    finally:
        OM._PlayerSnapshot = original

    assert _TRIPWIRES, "materialize_branches took no snapshot"
    drift = _TRIPWIRES[-1].drift()
    assert not drift, (
        f"{record.battle_tag}: an arm MUTATED a shared 'append-only immutable' record in place — "
        f"the aliasing contract in `_immutable_record_types` is broken: {drift[:3]}")
    assert len(_TRIPWIRES[-1]._pins) > 0, "nothing was pinned — the tripwire proves nothing"

    assert len(per_arm) == len(shared)
    for arm, one, many in zip(keep, per_arm, shared):
        why = (f"{record.battle_tag} arm {arm.label} "
               f"(ended={bool(arm.outcome.get('ended'))})")
        assert len(one.decisions) == len(many.decisions), f"{why}: decision COUNT differs"
        assert one.actions_complete == many.actions_complete, f"{why}: actions_complete differs"
        for k, (da, db) in enumerate(zip(one.decisions, many.decisions)):
            assert da.turn == db.turn, f"{why} decision {k}: turn"
            assert np.array_equal(da.mask, db.mask), f"{why} decision {k}: action MASK"
            assert (da.obs is None) == (db.obs is None)
            if da.obs is not None:
                assert da.obs.tobytes() == db.obs.tobytes(), (
                    f"{why} decision {k} (turn {da.turn}): prefix-shared obs is NOT bit-identical "
                    f"({int(np.count_nonzero(da.obs != db.obs))} dims differ)")
    return len(keep)


def test_a_battle_ENDING_arm_does_not_corrupt_the_arms_after_it():
    """An arm that ends the battle runs FIRST; every later arm must still be bit-identical.

    Also asserts the SHARING CONTRACT directly: nothing `_pin_shared` decided to share instead of
    copy may be mutated in place by any arm. That is a stronger statement than obs equality — a
    corrupted shared record only shows up as an obs difference if some arm happens to read the
    field that was corrupted.
    """
    checked = 0
    for _ in range(3):        # the anchor must yield at least one ENDING arm; retry a flat battle
        with tempfile.TemporaryDirectory(prefix="branch_end_") as out_dir:
            record, summary, npz = record_one_battle(out_dir)
            checked = check_ending_arms(record, summary, npz)
        if checked:
            break
    assert checked >= 2, ("no battle produced a terminal decision with an ending arm AND a "
                          "follower — re-run (the opponent is a RandomPlayer)")
