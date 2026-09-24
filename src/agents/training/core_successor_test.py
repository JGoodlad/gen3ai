"""The CORE road's successor (`gen3_core_search_v1`) — its event transport and its INTEGRITY teeth.

The three search parity gates (``materializer_parity_integration_test``,
``fork_sharing_parity_integration_test``, ``one_sided_view_parity_fuzz_test``) hold the core road's
decisions and obs bytes equal to the protocol and view roads, with the integrity check on every arm.
This file pins what those gates cannot show by passing: that the integrity check FAILS — loudly,
naming the decision, the depth and the obs block — when the typed and the text path disagree.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from agents.battle.battle_event import BattleEvent, EventKind
from agents.training.core_successor import (CoreIntegrityError, _assert_same,
                                            events_from_readings)


def test_a_reading_becomes_the_battle_event_gen3battle_builds():
    r = {"seq": 7, "turn": 3, "kind": "DAMAGE", "side": "opp", "actor": "zapdos", "target": None,
         "value": {"amount": -0.25, "hp_after": 0.5, "reason": "psn"},
         "raw": ["", "-damage", "p2a: Zapdos", "50/100 tox", "[from] psn"]}
    (ev,) = events_from_readings([r])
    assert ev == BattleEvent(seq=7, turn=3, kind=EventKind.DAMAGE, side="opp", actor_species="zapdos",
                             target_species=None, value={"amount": -0.25, "hp_after": 0.5, "reason": "psn"},
                             raw=("", "-damage", "p2a: Zapdos", "50/100 tox", "[from] psn"))
    assert list(ev.value) == ["amount", "hp_after", "reason"], "the builder's key order survives"


def test_the_integrity_assertion_names_the_first_differing_obs_index():
    obs = np.zeros(8, dtype=np.float32)
    mask = np.ones(11, dtype=np.int8)
    _assert_same((obs, mask), (obs.copy(), mask.copy()), "x")
    other = obs.copy()
    other[5] = 1.0
    with pytest.raises(CoreIntegrityError, match="index 5"):
        _assert_same((obs, mask), (other, mask), "decision #3 (p1), depth 1, arm 2")
    m2 = mask.copy()
    m2[0] = 0
    with pytest.raises(CoreIntegrityError, match="masks differ"):
        _assert_same((obs, mask), (obs, m2), "x")
    with pytest.raises(CoreIntegrityError, match="one path opened a decision"):
        _assert_same((obs, mask), None, "x")


@pytest.mark.sim
@pytest.mark.integration
def test_a_tampered_text_view_fails_the_integrity_check_loudly():
    """TEETH, end to end: a real core arm expanded with the driver's integrity check on carries the
    text path's view; making that view disagree (one HP point on our active) must RAISE from the
    successor build — the failure an integrity-on search stops on."""
    import agents.battle.one_sided_view_parity_fuzz_test as G
    from agents.training.obs_materializer import open_view_fork
    from agents.observation.state_encoder import get_observation_encoder, load_mappings
    from utils.bridge.search_session import SearchSession

    with tempfile.TemporaryDirectory() as td:
        record, summary, npz = G._record_one_battle(td, "rust", 0)
    actions = np.asarray(npz["actions"], dtype=int)
    invs = summary["invocations"]
    anchor = next(i for i, inv in enumerate(invs)
                  if inv.get("phase") == "move_selection" and int(inv["turn"]) > 2)
    turn = int(invs[anchor]["turn"])
    side = record.side_of(record.trainee_username)
    other = "p2" if side == "p1" else "p1"
    history = [int(x) for x in actions[:anchor]]
    with SearchSession(record, impl="rust") as ss:
        root = ss.open_root(turn, core="typed")
        pfx = root.prefix_p1_chunks if side == "p1" else root.prefix_p2_chunks
        cmap = G._choice_map(record, side, history, pfx, anchor)
        a = sorted(cmap)[0]
        (node,) = ss.expand_many([{"node_id": root.node_id, f"{side}_action": cmap[a],
                                   f"{other}_action": "default", "seed": "1,2,3,4", "label": a}],
                                 side=side, integrity=1)
    core = node.core_p1 if side == "p1" else node.core_p2
    assert core is not None and core.get("text_view") is not None, "the arm was not integrity-checked"
    enc = get_observation_encoder(load_mappings())
    fork = open_view_fork(pfx, username=record.username(side), packed_team=record.packed_team(side),
                          side=side, prefix_actions=history, battle_format=record.format_id,
                          battle_tag=record.battle_tag, encoder=enc, road="core")[0]
    assert fork.successor(core, a, where="clean") is not None, "fixture: the clean arm encodes"
    act = next(m for m in core["text_view"]["ours"]["mons"] if m["active"])
    act["current_hp"] = act["current_hp"] - 1 if act["current_hp"] > 1 else 2
    act["hp_fraction"] = act["current_hp"] / act["max_hp"]
    with pytest.raises(CoreIntegrityError, match="INTEGRITY"):
        fork.successor(core, a, where="decision (tampered), depth 1")
