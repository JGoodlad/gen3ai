"""Pin the accepted-enums seam (`agents.enums`).

`agents.enums` re-exports exactly the four spec-defined poke-env value-enums we permit
our code to borrow as keys. The critical assertion is that `Effect` is **never** among
them — it is the temporal volatile vocabulary replaced by `agents/observation/gen3_effects.py`,
and an accidental `Effect` re-export here would quietly reopen the door the strict-API
standard closed. Pinning `__all__` exactly makes that a CI failure, not a silent drift.
"""

import agents.enums as enums


def test_all_is_exactly_the_four_accepted_enums():
    assert set(enums.__all__) == {"PokemonType", "Status", "MoveCategory", "Weather"}
    # __all__ has no accidental duplicates / extras
    assert len(enums.__all__) == 4


def test_effect_is_not_re_exported():
    """`Effect` is intentionally excluded — gen3_effects.py replaces it."""
    assert "Effect" not in enums.__all__
    assert not hasattr(enums, "Effect")


def test_each_name_is_a_real_attribute():
    for name in enums.__all__:
        assert hasattr(enums, name), f"agents.enums.__all__ lists {name!r} but it is absent"


def test_members_are_the_same_objects_poke_env_uses():
    """Pure import-path indirection — the seam must not wrap or shadow the enums."""
    from poke_env.battle.move_category import MoveCategory
    from poke_env.battle.pokemon_type import PokemonType
    from poke_env.battle.status import Status
    from poke_env.battle.weather import Weather

    assert enums.PokemonType is PokemonType
    assert enums.Status is Status
    assert enums.MoveCategory is MoveCategory
    assert enums.Weather is Weather
