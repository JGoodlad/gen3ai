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
