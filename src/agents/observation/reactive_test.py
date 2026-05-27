"""Unit tests for the Hidden-Power-aware expected effectiveness helper in
`reactive.py`. The same helper is used by both the active-move multiplier path
and the 144-dim matchup matrix path, so testing it in isolation covers both.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock

from poke_env.battle.pokemon_type import PokemonType

from .reactive import _expected_multiplier
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

    mult = _expected_multiplier(bare_hp, attacker, salamence, tracker, ability_priors=None)
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

    mult = _expected_multiplier(bare_hp, attacker, salamence, tracker, ability_priors=None)
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
    actual = _expected_multiplier(bare_hp, attacker, salamence, tracker, ability_priors=None)
    assert actual == pytest.approx(expected)


def test_typed_hp_variant_uses_move_type_directly():
    """Own-team HP arrives with a typed id ("hiddenpowerice") and move.type set —
    fall through to the plain effective_multiplier, ignoring any tracker probs."""
    salamence = _make_mon("salamence", [PokemonType.DRAGON, PokemonType.FLYING])
    typed_hp = _make_move("hiddenpowerice", move_type=PokemonType.ICE)
    attacker = _make_mon("alakazam", [PokemonType.PSYCHIC])

    # Tracker has zero probs (own team — tracker wouldn't have an entry).
    tracker = _FakeTracker({})

    mult = _expected_multiplier(typed_hp, attacker, salamence, tracker, ability_priors=None)
    assert mult == pytest.approx(4.0)


def test_no_tracker_falls_back_to_move_type():
    """When called without a tracker (e.g. inference outside training), bare HP
    uses poke-env's default move.type. Preserves legacy behaviour."""
    salamence = _make_mon("salamence", [PokemonType.DRAGON, PokemonType.FLYING])
    bare_hp = _make_move("hiddenpower", move_type=PokemonType.NORMAL)
    attacker = _make_mon("zapdos", [PokemonType.ELECTRIC, PokemonType.FLYING])

    mult = _expected_multiplier(bare_hp, attacker, salamence, hp_tracker=None, ability_priors=None)
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

    mult = _expected_multiplier(surf, attacker, salamence, tracker, ability_priors=None)
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

    mult = _expected_multiplier(bare_hp, attacker, flygon, tracker, ability_priors=None)
    assert mult == pytest.approx(0.0)


# ============================================================================
# Defender-side ability uncertainty
# ============================================================================

def test_unrevealed_single_ability_collapses_to_known_value():
    """Gengar's only Gen 3 ability is Levitate (1.0 in priors). Earthquake vs an
    unrevealed Gengar should expect 0× — the priors collapse to a singleton."""
    from poke_env.battle.move import Move
    # Build a Gengar with no ability set (unrevealed) — Ghost/Poison
    gengar = _make_mon("gengar", [PokemonType.GHOST, PokemonType.POISON], ability=None)
    earthquake = _make_move("earthquake", move_type=PokemonType.GROUND)
    attacker = _make_mon("tyranitar", [PokemonType.DARK, PokemonType.ROCK])

    ability_priors = {"gengar": {"levitate": 1.0}}
    mult = _expected_multiplier(
        earthquake, attacker, gengar, hp_tracker=None, ability_priors=ability_priors
    )
    # Ground vs Ghost/Poison = 0.5 × 2 = 1; Levitate overrides to 0
    assert mult == pytest.approx(0.0)


def test_unrevealed_two_ability_gives_weighted_value():
    """Snorlax priors: 0.86 Immunity + 0.14 Thick Fat. Ice Beam (Ice) vs
    pure Normal Snorlax = 1.0 type-chart; Thick Fat halves Ice to 0.5×.

    Expected = 0.86 * 1.0 + 0.14 * 0.5 = 0.93.
    """
    snorlax = _make_mon("snorlax", [PokemonType.NORMAL], ability=None)
    ice_beam = _make_move("icebeam", move_type=PokemonType.ICE)
    attacker = _make_mon("articuno", [PokemonType.ICE, PokemonType.FLYING])

    ability_priors = {"snorlax": {"immunity": 0.86, "thickfat": 0.14}}
    mult = _expected_multiplier(
        ice_beam, attacker, snorlax, hp_tracker=None, ability_priors=ability_priors
    )
    assert mult == pytest.approx(0.86 * 1.0 + 0.14 * 0.5)


def test_revealed_ability_ignores_priors():
    """Once a Snorlax's ability has been revealed as Thick Fat (set on the live
    mon), the matchup must collapse to that exact value regardless of any
    priors entry. This is the property that lets own-team and revealed-opp
    mons share the same code path."""
    snorlax = _make_mon("snorlax", [PokemonType.NORMAL], ability="thickfat")
    ice_beam = _make_move("icebeam", move_type=PokemonType.ICE)
    attacker = _make_mon("articuno", [PokemonType.ICE, PokemonType.FLYING])

    # Priors say 86/14, but live ability is set — priors should be ignored.
    ability_priors = {"snorlax": {"immunity": 0.86, "thickfat": 0.14}}
    mult = _expected_multiplier(
        ice_beam, attacker, snorlax, hp_tracker=None, ability_priors=ability_priors
    )
    # Thick Fat is now certain → 0.5×
    assert mult == pytest.approx(0.5)


def test_no_priors_falls_back_to_live_ability():
    """Without an ability_priors dict (e.g. inference outside training), the
    function must behave exactly like the pre-fix code: read `mon.ability`,
    None → no modifier, set value → applied."""
    flygon = _make_mon("flygon", [PokemonType.GROUND, PokemonType.DRAGON], ability="levitate")
    earthquake = _make_move("earthquake", move_type=PokemonType.GROUND)
    attacker = _make_mon("tyranitar", [PokemonType.DARK, PokemonType.ROCK])

    mult = _expected_multiplier(
        earthquake, attacker, flygon, hp_tracker=None, ability_priors=None
    )
    # Levitate → 0
    assert mult == pytest.approx(0.0)

    # And the same call with no live ability and no priors falls through to
    # type-chart only (Ground vs Ground/Dragon = 1× × 1× = 1×)
    flygon_no_ability = _make_mon("flygon", [PokemonType.GROUND, PokemonType.DRAGON], ability=None)
    mult = _expected_multiplier(
        earthquake, attacker, flygon_no_ability, hp_tracker=None, ability_priors=None
    )
    assert mult == pytest.approx(1.0)


def test_unknownability_sentinel_treated_as_unrevealed():
    """poke-env reports `mon.ability == "unknownability"` for opp mons whose
    ability hasn't fired (not just `None`). Both must route to priors —
    matching the gate in AbilitiesEncoder (abilities.py:75)."""
    snorlax = _make_mon("snorlax", [PokemonType.NORMAL], ability="unknownability")
    ice_beam = _make_move("icebeam", move_type=PokemonType.ICE)
    attacker = _make_mon("articuno", [PokemonType.ICE, PokemonType.FLYING])

    ability_priors = {"snorlax": {"immunity": 0.86, "thickfat": 0.14}}
    mult = _expected_multiplier(
        ice_beam, attacker, snorlax, hp_tracker=None, ability_priors=ability_priors
    )
    # Same value as the None case — the sentinel is treated as unrevealed.
    assert mult == pytest.approx(0.86 * 1.0 + 0.14 * 0.5)


def test_species_absent_from_priors_falls_through():
    """If priors are loaded but the defender's species isn't in the file, the
    function must fall through to live `mon.ability` rather than crash or
    pick an arbitrary ability."""
    weirdmon = _make_mon("unknownspecies", [PokemonType.NORMAL], ability=None)
    surf = _make_move("surf", move_type=PokemonType.WATER)
    attacker = _make_mon("starmie", [PokemonType.WATER, PokemonType.PSYCHIC])

    # Priors dict exists but doesn't cover this species
    ability_priors = {"snorlax": {"immunity": 1.0}}
    mult = _expected_multiplier(
        surf, attacker, weirdmon, hp_tracker=None, ability_priors=ability_priors
    )
    # Water vs Normal = 1× (no ability adjustment)
    assert mult == pytest.approx(1.0)


def test_ability_with_no_type_relevance_doesnt_affect_matchup():
    """Tentacruel's priors are Clear Body / Liquid Ooze — neither affects type
    effectiveness. Earthquake (Ground vs Water/Poison = 2×) should expect
    exactly 2× regardless of the 0.21/0.79 split."""
    tentacruel = _make_mon("tentacruel", [PokemonType.WATER, PokemonType.POISON], ability=None)
    earthquake = _make_move("earthquake", move_type=PokemonType.GROUND)
    attacker = _make_mon("tyranitar", [PokemonType.DARK, PokemonType.ROCK])

    ability_priors = {"tentacruel": {"clearbody": 0.21, "liquidooze": 0.79}}
    mult = _expected_multiplier(
        earthquake, attacker, tentacruel, hp_tracker=None, ability_priors=ability_priors
    )
    # Ground vs Water = 1×; Ground vs Poison = 2×; combined = 2×. Both abilities
    # leave it unchanged, so expected = 2× regardless of split.
    assert mult == pytest.approx(2.0)


def test_joint_hp_and_ability_uncertainty_composes():
    """HP attacker with [0.5 Ice, 0.5 Grass] vs unrevealed Snorlax (Normal)
    [0.86 Immunity, 0.14 Thick Fat]:

    Ice on Snorlax(Immunity)   = 1.0 × 1   = 1.0
    Ice on Snorlax(ThickFat)   = 1.0 × 0.5 = 0.5
    Grass on Snorlax(Immunity) = 1.0 × 1   = 1.0  (Grass × Normal = 1; Immunity no-op)
    Grass on Snorlax(ThickFat) = 1.0 × 1   = 1.0  (Thick Fat halves Ice/Fire only)
    """
    snorlax = _make_mon("snorlax", [PokemonType.NORMAL], ability=None)
    bare_hp = _make_move("hiddenpower")
    attacker = _make_mon("zapdos", [PokemonType.ELECTRIC, PokemonType.FLYING])

    hp_probs = np.zeros(16, dtype=np.float32)
    hp_probs[HIDDEN_POWER_TYPE_ORDER.index(PokemonType.ICE)] = 0.5
    hp_probs[HIDDEN_POWER_TYPE_ORDER.index(PokemonType.GRASS)] = 0.5
    tracker = _FakeTracker({"zapdos": hp_probs})

    ability_priors = {"snorlax": {"immunity": 0.86, "thickfat": 0.14}}
    mult = _expected_multiplier(
        bare_hp, attacker, snorlax, hp_tracker=tracker, ability_priors=ability_priors
    )
    expected = (
        0.5 * 0.86 * 1.0   # Ice × Immunity
        + 0.5 * 0.14 * 0.5  # Ice × Thick Fat (halves Ice)
        + 0.5 * 0.86 * 1.0  # Grass × Immunity
        + 0.5 * 0.14 * 1.0  # Grass × Thick Fat (no Grass effect)
    )
    assert mult == pytest.approx(expected)
