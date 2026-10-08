"""LiveView's SHAPE contract: LivePokemon carries NO past-turn fields — last_move et al. are physically
absent, so a consumer is forced to the event record for history — and its field set is pinned.

(The tests that built a LiveView from a poke-env battle went with ``LiveView.from_battle``, T27 P6 slice 6d-2;
the core's view is held against the engine by ``core_corpus_test.py`` and the cargo ``present`` tests, and
``core_view`` copies it field for field.)
"""

import dataclasses

import pytest

from agents.battle.live_view import LivePokemon


# ───────────────────────────── NO history (the boundary) ────────────────────
# NOTE: protect_counter is deliberately NOT here — it is surfaced as a CURRENT-board field
# (the present consecutive-stall counter value that sets the next Protect's odds), exactly like
# status_counter, NOT a past-turn event. See test_livepokemon_fields_are_exactly_the_minimal_set.
_FORBIDDEN_HISTORY_FIELDS = [
    "last_move", "last_cant_reason", "first_turn",
    "must_recharge", "preparing", "preparing_move", "active_turns",
]


def test_livepokemon_fields_are_exactly_the_minimal_set():
    """Pin the field set so nobody silently grows LivePokemon into a state grab-bag.
    The spread / consumed-item / status-counter fields are deliberate current-board
    additions (Phase 1 of the strict-API plan), stats / current_hp / max_hp are the
    incoming-damage belief's current-board inputs (EV-computed stats + integer HP, so the
    obs+reward belief reads the read-model instead of the raw Pokemon), and protect_counter is the
    current consecutive-stall counter (gen3_protect_odds_v1, the obs protect-success-odds source) —
    still NO past-turn fields."""
    names = {f.name for f in dataclasses.fields(LivePokemon)}
    assert names == {
        "species", "active", "fainted", "revealed", "hp_fraction", "status",
        "types", "moves", "item", "ability", "boosts", "volatiles",
        # spread block (own-side gated) + revealed consumed item + status/protect counters
        "base_stats", "ivs", "evs", "nature", "spread_known",
        "consumed_item", "status_counter", "protect_counter",
        # incoming-damage belief inputs: EV-computed stats + integer HP (current-board facts)
        "stats", "current_hp", "max_hp",
        # gen3_obs_facts_v1: what a PROTOCOL line has revealed (current knowledge, not history)
        "item_public", "ability_public",
    }


@pytest.mark.parametrize("attr", _FORBIDDEN_HISTORY_FIELDS)
def test_livepokemon_has_no_history_fields(attr):
    assert attr not in {f.name for f in dataclasses.fields(LivePokemon)}, (
        f"LivePokemon must not expose past-turn field {attr!r} — history belongs to the event record")
