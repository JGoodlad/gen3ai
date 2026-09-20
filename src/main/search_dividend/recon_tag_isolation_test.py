"""An OFFLINE replay must never write into the LIVE reconstruction record
(``gen3_recon_tag_collision_v1``).

🚨 **THE DEFECT THIS PINS, and it is a GIGO class rather than a tidiness one.**
``main.search_dividend.record.install_choice_tap`` patches
``BattleStreamClient._write_choice`` PROCESS-WIDE and its only discriminator is the battle ROOM
(``LiveRecordBuilder.accepts_room``) — a room filter and not an on/off switch, because in the
mirror the unsearched side commits real live choices while our search holds a worker thread.
``obs_materializer._next_tag`` used to return the caller's ``battle_tag`` VERBATIM, and the search
passes ``battle_tag=record.battle_tag``. So the offline replay of the live battle ran in the LIVE
BATTLE'S ROOM, and every ``/choose default`` its replay player emits when its action list runs out
was appended to the live record as if the live player had chosen it.

**Measured on the `playoff` arm, one mirror game, `--impl rust`:** ~1 spurious ``default`` per
materialized ARM (8.6 arms/decision), so by turn 2 the trainee's script read
``[default x10, 'switch Tyranitar']``; every nested rollout replayed a prefix that was not the
battle and **64 of 66 playoffs died** on ``unresolvable choice for pN: MoveName(...)`` naming a
BENCHED mon. With the fix: **34 playoff errors and 68 failed rollouts fall to 1 and 2** on the same
cell.

⚠️ **IT WAS NEVER A RUST DEFECT**, though it was filed as one. The ten spurious ``default``s appear
byte-for-byte under ``--impl node`` too; node's bridge RE-REQUESTS on an unresolvable choice where
rust fails loud, so the node cell ran every rollout down a line nobody asked for and reported a
clean number. The loud impl was the one telling the truth.

The unit half pins the PROPERTY (a replay tag is never the caller's, and is unique); the ``sim``
half pins the CONSEQUENCE end to end, because the property could be satisfied by a tag scheme that
still collides in some other way.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

import agents.training.obs_materializer as OM
from main.search_dividend.record import (LiveRecordBuilder, install_choice_tap,
                                         set_active_builder, uninstall_choice_tap)


def test_a_replay_tag_is_never_the_callers_and_is_unique():
    a = OM._next_tag("battle-gen3ou-1", "gen3ou")
    b = OM._next_tag("battle-gen3ou-1", "gen3ou")
    c = OM._next_tag(None, "gen3ou")
    assert a != "battle-gen3ou-1", (
        "an offline replay took the LIVE battle's room, so the live record's choice tap — whose "
        "only filter is that room — recorded the replay's `/choose default` orders as the live "
        "player's. See the module docstring.")
    assert a != b, "two replays of the same battle shared a room"
    assert a.startswith("battle-gen3ou-1"), "the caller's tag must survive as a prefix (tracing)"
    # `Player._create_battle` checks `split_message[1] == self._format`, so segment 1 stays the
    # format or the replay battle is never created at all.
    for tag in (a, b, c):
        assert tag.split("-")[1] == "gen3ou", f"{tag!r} moved the format out of segment 1"


@pytest.mark.sim
@pytest.mark.integration
def test_an_offline_materialization_records_nothing_into_the_live_builder():
    """End to end: with the tap installed and a live builder ACTIVE, materializing the same
    battle offline must leave the live record untouched."""
    import agents.battle.one_sided_view_parity_fuzz_test as G

    with tempfile.TemporaryDirectory() as td:
        record, summary, npz = G._record_one_battle(td, "rust", 0)
    actions = [int(x) for x in np.asarray(npz["actions"], dtype=int)]
    invs = summary["invocations"]
    cand = [i for i, inv in enumerate(invs)
            if inv.get("phase") == "move_selection" and int(inv["turn"]) > 1]
    assert cand, "the fixture battle produced no mid-battle move selection"
    anchor = cand[len(cand) // 2]
    side = record.side_of(record.trainee_username)

    # The prefix the search would replay. Any chunk list reaching the branch decision will do —
    # what is on trial is the ROOM the replay uses, not what it decides.
    from utils.bridge.search_session import SearchSession
    with SearchSession(record, impl="rust") as ss:
        root = ss.open_root(int(invs[anchor]["turn"]))
        pfx = root.prefix_p1_chunks if side == "p1" else root.prefix_p2_chunks

    install_choice_tap()
    builder = LiveRecordBuilder(
        battle_format=record.format_id, seed=record.prng_seed,
        battle_tag=record.battle_tag, chunk_sink=[], our_side=side,
        trainee_username=record.trainee_username)
    builder.set_player("p1", record.username("p1"), record.packed_team("p1"))
    builder.set_player("p2", record.username("p2"), record.packed_team("p2"))
    set_active_builder(builder)
    try:
        # Deliberately ONE action short of the branch decision, which is exactly the state that
        # makes the replay player answer the next request with `/choose default`.
        OM.materialize_decisions(
            list(pfx), username=record.username(side), packed_team=record.packed_team(side),
            side=side, actions=actions[:anchor], battle_format=record.format_id,
            battle_tag=record.battle_tag)
    finally:
        set_active_builder(None)
        uninstall_choice_tap()

    assert builder.n_commands == 0, (
        f"the offline replay wrote {builder.n_commands} command(s) into the LIVE record — the "
        f"replay is sharing the live battle's room again. Every one of them becomes a choice the "
        f"live player never made, and every later counterfactual replays a prefix that is not the "
        f"battle. See the module docstring for what that cost.")


@pytest.fixture(autouse=True)
def _restore_tap():
    """The tap is a CLASS patch: one leaked install rides into every bridge test collected after
    this file. ``uninstall_choice_tap`` is idempotent, so an un-installed tap is a no-op."""
    yield
    set_active_builder(None)
    uninstall_choice_tap()
