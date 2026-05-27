"""Unit tests for the Hidden-Power-aware expected effectiveness helper in
`reactive.py`. The same helper is used by both the active-move multiplier path
and the 144-dim matchup matrix path, so testing it in isolation covers both.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock

from poke_env.battle.pokemon_type import PokemonType

from .reactive import _hp_expected_multiplier
from agents.training.hidden_power_tracker import HIDDEN_POWER_TYPE_ORDER


def _make_move(move_id, move_type=PokemonType.NORMAL):
    m = MagicMock()
    m.id = move_id
    m.type = move_type
    return m


def _make_mon(species, types, ability=""):
    mon = MagicMock()
    mon.species = species
    mon.type_1 = types[0]
    mon.type_2 = types[1] if len(types) > 1 else None
    mon.ability = ability
    mon.status = None
    return mon


class _FakeTracker:
    def __init__(self, probs_by_species):
        self._probs = {k: np.asarray(v, dtype=np.float32) for k, v in probs_by_species.items()}

    def get_probs(self, species):
        return self._probs.get(species)


def test_one_hot_distribution_collapses_to_exact_effectiveness():
    """When the tracker has narrowed HP to a single type, expected eff == real eff."""
    # Salamence (Dragon/Flying) is 4× to Ice
    salamence = _make_mon("salamence", [PokemonType.DRAGON, PokemonType.FLYING])
    bare_hp = _make_move("hiddenpower")
    attacker = _make_mon("zapdos", [PokemonType.ELECTRIC, PokemonType.FLYING])

    ice_idx = HIDDEN_POWER_TYPE_ORDER.index(PokemonType.ICE)
    one_hot = np.zeros(16, dtype=np.float32)
    one_hot[ice_idx] = 1.0
    tracker = _FakeTracker({"zapdos": one_hot})

    mult = _hp_expected_multiplier(bare_hp, attacker, salamence, tracker)
    assert mult == pytest.approx(4.0)


def test_two_type_distribution_gives_expected_value():
    """[0.5 Ice, 0.5 Grass] HP vs Salamence (Dragon/Flying):
    Ice  vs Dragon/Flying = 2 × 2 = 4
    Grass vs Dragon/Flying = 0.5 × 0.5 = 0.25
    Expected = 0.5 * 4 + 0.5 * 0.25 = 2.125."""
    salamence = _make_mon("salamence", [PokemonType.DRAGON, PokemonType.FLYING])
    bare_hp = _make_move("hiddenpower")
    attacker = _make_mon("zapdos", [PokemonType.ELECTRIC, PokemonType.FLYING])

    probs = np.zeros(16, dtype=np.float32)
    probs[HIDDEN_POWER_TYPE_ORDER.index(PokemonType.ICE)] = 0.5
    probs[HIDDEN_POWER_TYPE_ORDER.index(PokemonType.GRASS)] = 0.5
    tracker = _FakeTracker({"zapdos": probs})

    mult = _hp_expected_multiplier(bare_hp, attacker, salamence, tracker)
    assert mult == pytest.approx(0.5 * 4.0 + 0.5 * 0.25)


def test_uniform_distribution_equals_mean_over_all_16_types():
    """Uniform 1/16 probs => mean effectiveness across the 16 candidate types."""
    from agents.gen3_mechanics import effective_multiplier

    salamence = _make_mon("salamence", [PokemonType.DRAGON, PokemonType.FLYING])
    bare_hp = _make_move("hiddenpower")
    attacker = _make_mon("zapdos", [PokemonType.ELECTRIC, PokemonType.FLYING])

    probs = np.full(16, 1.0 / 16.0, dtype=np.float32)
    tracker = _FakeTracker({"zapdos": probs})

    expected = sum(effective_multiplier(t, salamence) for t in HIDDEN_POWER_TYPE_ORDER) / 16.0
    actual = _hp_expected_multiplier(bare_hp, attacker, salamence, tracker)
    assert actual == pytest.approx(expected)


def test_typed_hp_variant_uses_move_type_directly():
    """Own-team HP arrives with a typed id ("hiddenpowerice") and move.type set —
    fall through to the plain effective_multiplier, ignoring any tracker probs."""
    salamence = _make_mon("salamence", [PokemonType.DRAGON, PokemonType.FLYING])
    typed_hp = _make_move("hiddenpowerice", move_type=PokemonType.ICE)
    attacker = _make_mon("alakazam", [PokemonType.PSYCHIC])

    # Tracker has zero probs (own team — tracker wouldn't have an entry).
    tracker = _FakeTracker({})

    mult = _hp_expected_multiplier(typed_hp, attacker, salamence, tracker)
    assert mult == pytest.approx(4.0)


def test_no_tracker_falls_back_to_move_type():
    """When called without a tracker (e.g. inference outside training), bare HP
    uses poke-env's default move.type. Preserves legacy behaviour."""
    salamence = _make_mon("salamence", [PokemonType.DRAGON, PokemonType.FLYING])
    bare_hp = _make_move("hiddenpower", move_type=PokemonType.NORMAL)
    attacker = _make_mon("zapdos", [PokemonType.ELECTRIC, PokemonType.FLYING])

    mult = _hp_expected_multiplier(bare_hp, attacker, salamence, hp_tracker=None)
    # Normal vs Dragon/Flying = 1× (no immunity in this match-up)
    assert mult == pytest.approx(1.0)


def test_non_hp_move_unchanged():
    """A regular move (not Hidden Power) ignores the tracker entirely."""
    salamence = _make_mon("salamence", [PokemonType.DRAGON, PokemonType.FLYING])
    surf = _make_move("surf", move_type=PokemonType.WATER)
    attacker = _make_mon("starmie", [PokemonType.WATER, PokemonType.PSYCHIC])

    # Tracker exists but should be irrelevant for a typed move
    probs = np.zeros(16, dtype=np.float32)
    probs[HIDDEN_POWER_TYPE_ORDER.index(PokemonType.FIRE)] = 1.0
    tracker = _FakeTracker({"starmie": probs})

    mult = _hp_expected_multiplier(surf, attacker, salamence, tracker)
    # Water is 1× vs Dragon (resisted by Dragon? no — Dragon resists Water; multiplier 0.5)
    # Actually: Water vs Dragon = 0.5; Water vs Flying = 1. Combined = 0.5.
    assert mult == pytest.approx(0.5)


def test_levitate_blocks_ground_hp():
    """HP-Ground vs a Levitate user: real 0× immunity, not the unknown sentinel.

    With a one-hot Ground distribution, the expected multiplier is the literal 0,
    matching what the model should see as a real wall.
    """
    flygon = _make_mon("flygon", [PokemonType.GROUND, PokemonType.DRAGON], ability="levitate")
    bare_hp = _make_move("hiddenpower")
    attacker = _make_mon("ampharos", [PokemonType.ELECTRIC])

    probs = np.zeros(16, dtype=np.float32)
    probs[HIDDEN_POWER_TYPE_ORDER.index(PokemonType.GROUND)] = 1.0
    tracker = _FakeTracker({"ampharos": probs})

    mult = _hp_expected_multiplier(bare_hp, attacker, flygon, tracker)
    assert mult == pytest.approx(0.0)
