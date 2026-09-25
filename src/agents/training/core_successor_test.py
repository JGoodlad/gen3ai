"""The CORE road's successor (`gen3_core_search_v1`) — its event transport, and the refusal of the
DELETED typed shortcut and its integrity mode (program §4 M4 row).

The three search parity gates (``materializer_parity_integration_test``,
``fork_sharing_parity_integration_test``, ``one_sided_view_parity_fuzz_test``) hold the core road's
decisions and obs bytes equal to the protocol and view roads.
"""

from __future__ import annotations

import tempfile

import pytest

from agents.battle.battle_event import BattleEvent, EventKind
from agents.training.core_successor import events_from_readings


def test_a_reading_becomes_the_battle_event_gen3battle_builds():
    r = {"seq": 7, "turn": 3, "kind": "DAMAGE", "side": "opp", "actor": "zapdos", "target": None,
         "value": {"amount": -0.25, "hp_after": 0.5, "reason": "psn"},
         "raw": ["", "-damage", "p2a: Zapdos", "50/100 tox", "[from] psn"]}
    (ev,) = events_from_readings([r])
    assert ev == BattleEvent(seq=7, turn=3, kind=EventKind.DAMAGE, side="opp", actor_species="zapdos",
                             target_species=None, value={"amount": -0.25, "hp_after": 0.5, "reason": "psn"},
                             raw=("", "-damage", "p2a: Zapdos", "50/100 tox", "[from] psn"))
    assert list(ev.value) == ["amount", "hp_after", "reason"], "the builder's key order survives"


@pytest.mark.sim
@pytest.mark.integration
def test_the_driver_refuses_the_deleted_typed_shortcut_and_integrity_mode():
    """The typed shortcut and its INTEGRITY mode are DELETED (program §4 M4 row): the driver
    REFUSES a request that still asks for either — never serves it silently on the one text path
    — and the text path itself answers."""
    import agents.battle.one_sided_view_parity_fuzz_test as G
    from utils.bridge.search_session import SearchError, SearchSession

    with tempfile.TemporaryDirectory() as td:
        record, summary, _npz = G._record_one_battle(td, "rust", 0)
    turn = next(int(inv["turn"]) for inv in summary["invocations"]
                if inv.get("phase") == "move_selection" and int(inv["turn"]) > 2)
    with SearchSession(record, impl="rust") as ss:
        with pytest.raises(SearchError, match="typed"):
            ss.open_root(turn, core="typed")
        with pytest.raises(SearchError, match="DELETED"):
            ss._call({"cmd": "open_root", "record": record.to_dict(), "turn": turn, "core": "typed"})
        root = ss.open_root(turn, core="text")
        with pytest.raises(SearchError, match="integrity is DELETED"):
            ss._call({"cmd": "expand_many", "arms": [], "integrity": 1})
        assert root.node_id
