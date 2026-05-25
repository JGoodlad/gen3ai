import numpy as np
import pytest
from unittest.mock import MagicMock

from poke_env.battle.effect import Effect
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status

from agents.gen3_mechanics import (
    ABILITY_TYPE_MULTIPLIER,
    STATUS_MOVE_IMMUNITY,
    NOTABLE_EFFECTS,
    PHAZING_MOVES,
    INVULNERABLE_MOVES,
    STATUS_MOVES,
    RECOVERY_MOVES,
    HAZARD_CLEAR_MOVES,
    SETUP_MOVES,
    BOOST_STATS,
    BOOST_DIM,
    effective_multiplier,
    is_status_move_immune,
    mon_status_str,
    boosts_array,
    boosts_str,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mon(type_1=PokemonType.NORMAL, type_2=None, ability=None, status=None, effects=None):
    m = MagicMock()
    m.type_1 = type_1
    m.type_2 = type_2
    m.ability = ability
    m.status = status
    m.effects = effects or {}
    m.boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0}
    return m


# ---------------------------------------------------------------------------
# effective_multiplier
# ---------------------------------------------------------------------------

class TestEffectiveMultiplier:
    def test_levitate_blocks_ground(self):
        mon = _mon(type_1=PokemonType.GHOST, type_2=PokemonType.POISON, ability="levitate")
        assert effective_multiplier(PokemonType.GROUND, mon) == 0.0

    def test_levitate_does_not_block_other_types(self):
        mon = _mon(type_1=PokemonType.GHOST, type_2=PokemonType.POISON, ability="levitate")
        # Psychic is super-effective vs Poison (2×)
        assert effective_multiplier(PokemonType.PSYCHIC, mon) == 2.0

    def test_voltabsorb_blocks_electric(self):
        mon = _mon(type_1=PokemonType.WATER, ability="voltabsorb")
        assert effective_multiplier(PokemonType.ELECTRIC, mon) == 0.0

    def test_waterabsorb_blocks_water(self):
        mon = _mon(type_1=PokemonType.FIRE, ability="waterabsorb")
        assert effective_multiplier(PokemonType.WATER, mon) == 0.0

    def test_flashfire_blocks_fire(self):
        mon = _mon(type_1=PokemonType.STEEL, ability="flashfire")
        assert effective_multiplier(PokemonType.FIRE, mon) == 0.0

    def test_no_ability_uses_type_chart(self):
        # Ground is super-effective vs Poison (2×)
        mon = _mon(type_1=PokemonType.POISON, ability=None)
        assert effective_multiplier(PokemonType.GROUND, mon) == 2.0

    def test_none_ability_uses_type_chart(self):
        mon = _mon(type_1=PokemonType.NORMAL, ability=None)
        assert effective_multiplier(PokemonType.NORMAL, mon) == 1.0

    def test_case_insensitive_ability(self):
        mon = _mon(type_1=PokemonType.GHOST, ability="Levitate")
        assert effective_multiplier(PokemonType.GROUND, mon) == 0.0

    def test_immune_via_type_not_ability(self):
        # Ghost is immune to Normal (type chart, not ability)
        mon = _mon(type_1=PokemonType.GHOST, ability=None)
        assert effective_multiplier(PokemonType.NORMAL, mon) == 0.0


# ---------------------------------------------------------------------------
# is_status_move_immune
# ---------------------------------------------------------------------------

class TestIsStatusMoveImmune:
    def test_ground_immune_to_thunderwave(self):
        mon = _mon(type_1=PokemonType.GROUND)
        assert is_status_move_immune("thunderwave", mon) is True

    def test_steel_immune_to_toxic(self):
        mon = _mon(type_1=PokemonType.STEEL)
        assert is_status_move_immune("toxic", mon) is True

    def test_poison_immune_to_toxic(self):
        mon = _mon(type_1=PokemonType.POISON)
        assert is_status_move_immune("toxic", mon) is True

    def test_fire_immune_to_willowisp(self):
        mon = _mon(type_1=PokemonType.FIRE)
        assert is_status_move_immune("willowisp", mon) is True

    def test_already_statused_is_immune(self):
        mon = _mon(type_1=PokemonType.NORMAL, status=Status.PAR)
        assert is_status_move_immune("thunderwave", mon) is True

    def test_already_statused_blocks_any_status_move(self):
        mon = _mon(type_1=PokemonType.NORMAL, status=Status.BRN)
        assert is_status_move_immune("toxic", mon) is True

    def test_normal_type_not_immune_to_toxic(self):
        mon = _mon(type_1=PokemonType.NORMAL)
        assert is_status_move_immune("toxic", mon) is False

    def test_unknown_move_not_immune(self):
        mon = _mon(type_1=PokemonType.NORMAL)
        assert is_status_move_immune("tackle", mon) is False

    def test_sleep_move_no_type_immunity(self):
        # No type is immune to sleep moves by type alone
        mon = _mon(type_1=PokemonType.STEEL)
        assert is_status_move_immune("spore", mon) is False


# ---------------------------------------------------------------------------
# mon_status_str
# ---------------------------------------------------------------------------

class TestMonStatusStr:
    def test_none_mon_returns_none(self):
        assert mon_status_str(None) is None

    def test_clean_mon_returns_none(self):
        mon = _mon()
        assert mon_status_str(mon) is None

    def test_burn_status(self):
        mon = _mon(status=Status.BRN)
        assert mon_status_str(mon) == "BRN"

    def test_paralysis_status(self):
        mon = _mon(status=Status.PAR)
        assert mon_status_str(mon) == "PAR"

    def test_taunt_effect(self):
        mon = _mon(effects={Effect.TAUNT: 3})
        assert mon_status_str(mon) == "taunt"

    def test_confusion_effect(self):
        mon = _mon(effects={Effect.CONFUSION: 2})
        assert mon_status_str(mon) == "confusion"

    def test_combined_status_and_effect(self):
        mon = _mon(status=Status.BRN, effects={Effect.TAUNT: 2})
        result = mon_status_str(mon)
        assert result == "BRN, taunt"

    def test_only_notable_effects_shown(self):
        # An effect not in NOTABLE_EFFECTS should not appear
        from poke_env.battle.effect import Effect as E
        non_notable = next(e for e in Effect if e not in NOTABLE_EFFECTS)
        mon = _mon(effects={non_notable: 1})
        assert mon_status_str(mon) is None


# ---------------------------------------------------------------------------
# boosts_array
# ---------------------------------------------------------------------------

class TestBoostsArray:
    def test_none_mon_returns_zeros(self):
        arr = boosts_array(None)
        assert arr.shape == (BOOST_DIM,)
        assert arr.dtype == np.int8
        np.testing.assert_array_equal(arr, np.zeros(BOOST_DIM, dtype=np.int8))

    def test_all_zero_boosts(self):
        mon = _mon()
        arr = boosts_array(mon)
        np.testing.assert_array_equal(arr, np.zeros(BOOST_DIM, dtype=np.int8))

    def test_positive_boosts(self):
        mon = _mon()
        mon.boosts = {"atk": 2, "def": 0, "spa": 1, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0}
        arr = boosts_array(mon)
        assert arr[BOOST_STATS.index("atk")] == 2
        assert arr[BOOST_STATS.index("spa")] == 1
        assert arr[BOOST_STATS.index("def")] == 0

    def test_negative_boosts(self):
        mon = _mon()
        mon.boosts = {"atk": 0, "def": -1, "spa": 0, "spd": -2, "spe": 0, "accuracy": 0, "evasion": 0}
        arr = boosts_array(mon)
        assert arr[BOOST_STATS.index("def")] == -1
        assert arr[BOOST_STATS.index("spd")] == -2

    def test_missing_boosts_default_to_zero(self):
        mon = MagicMock()
        mon.boosts = {"atk": 3}  # only atk set
        arr = boosts_array(mon)
        assert arr[BOOST_STATS.index("atk")] == 3
        assert arr[BOOST_STATS.index("def")] == 0

    def test_no_boosts_attribute(self):
        mon = MagicMock(spec=[])  # no boosts attr
        arr = boosts_array(mon)
        np.testing.assert_array_equal(arr, np.zeros(BOOST_DIM, dtype=np.int8))

    def test_dtype_is_int8(self):
        arr = boosts_array(None)
        assert arr.dtype == np.int8


# ---------------------------------------------------------------------------
# boosts_str
# ---------------------------------------------------------------------------

class TestBoostsStr:
    def test_none_mon_returns_none(self):
        assert boosts_str(None) is None

    def test_all_zero_returns_none(self):
        mon = _mon()
        assert boosts_str(mon) is None

    def test_positive_boost(self):
        mon = _mon()
        mon.boosts = {"atk": 2, "def": 0, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0}
        assert boosts_str(mon) == "atk:+2"

    def test_mixed_boosts(self):
        mon = _mon()
        mon.boosts = {"atk": 2, "def": -1, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0}
        result = boosts_str(mon)
        assert "atk:+2" in result
        assert "def:-1" in result


# ---------------------------------------------------------------------------
# Move category set sanity checks
# ---------------------------------------------------------------------------

class TestMoveSets:
    def test_phazing_moves(self):
        assert "roar" in PHAZING_MOVES
        assert "whirlwind" in PHAZING_MOVES

    def test_invulnerable_moves(self):
        assert "protect" in INVULNERABLE_MOVES
        assert "detect" in INVULNERABLE_MOVES
        assert "endure" in INVULNERABLE_MOVES

    def test_setup_moves(self):
        assert "swordsdance" in SETUP_MOVES
        assert "calmmind" in SETUP_MOVES
        assert "dragondance" in SETUP_MOVES

    def test_recovery_moves(self):
        assert "recover" in RECOVERY_MOVES
        assert "rest" in RECOVERY_MOVES
        assert "softboiled" in RECOVERY_MOVES

    def test_status_moves(self):
        assert "toxic" in STATUS_MOVES
        assert "willowisp" in STATUS_MOVES
        assert "thunderwave" in STATUS_MOVES

    def test_hazard_clear_moves(self):
        assert "rapidspin" in HAZARD_CLEAR_MOVES

    def test_boost_stats_length(self):
        assert BOOST_DIM == 7
        assert len(BOOST_STATS) == 7
