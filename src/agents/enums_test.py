"""Pin the owned-enums seam (`agents.enums`).

`agents.enums` DEFINES the four spec-defined value-enums our code uses as keys, with nothing outside the
standard library (P1 of the poke-env retirement, T27), and the vendored fork's four enum modules re-export
THEM. The critical assertions are that

* `Effect` is **never** among them — it is the temporal volatile vocabulary replaced by
  `agents/observation/gen3_effects.py`, and an accidental `Effect` here would quietly reopen the door the
  strict-API standard closed (pinning `__all__` exactly makes that a CI failure, not a silent drift);
* importing the seam loads NO `poke_env` module (a fresh interpreter — the in-process `sys.modules` of a test
  session is contaminated by every other test); and
* the identity the Python battle layer relies on still holds: the fork's `Pokemon.status` / `Move.category` /
  `Move.type` / battle weather ARE these classes' members, so `pokemon.status == Status.SLP` stays true.

It is also the ONE allowlisted home of every other "the fork re-exports OURS" pin P1 added (a new test file that
imports poke-env is a new importer, which the shrink-only gate refuses): `to_id_str` (`utils.showdown_id`), team
packing (`utils.team_packing`), the `live_view` unknown-item sentinel, `global_env`'s screen ids, `gen3_mechanics`'s
effect names, the eval roster's two statements, and the frozen special-type tables. Each reads the fork as the ORACLE
and dies with it (P6).
"""
import subprocess
import sys

import agents.enums as enums
from utils.paths import src_root


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


def test_importing_the_seam_loads_no_poke_env_module():
    """The seam is the root of `agents.gen3_data`'s and the trainer's closure — it must be poke-env-free."""
    code = ("import sys\n"
            "import agents.enums\n"
            "bad = sorted(k for k in sys.modules if k == 'poke_env' or k.startswith('poke_env.'))\n"
            "assert not bad, bad\n")
    env = {"PYTHONPATH": str(src_root()), "PATH": "/usr/bin:/bin"}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]


def test_the_fork_re_exports_the_same_objects():
    """The definition is OURS now; the fork's modules hand back the very same classes (identity is the
    contract the poke-env battle layer and every `agents.enums` consumer share)."""
    from poke_env.battle.move_category import MoveCategory
    from poke_env.battle.pokemon_type import PokemonType
    from poke_env.battle.status import Status
    from poke_env.battle.weather import Weather

    assert enums.PokemonType is PokemonType
    assert enums.Status is Status
    assert enums.MoveCategory is MoveCategory
    assert enums.Weather is Weather


def test_the_helper_methods_the_fork_calls_survive_the_move():
    """The fork's battle layer calls these; our own code does not, but P1 must not change the fork's behaviour."""
    PT, W = enums.PokemonType, enums.Weather
    assert PT.from_name("fire") is PT.FIRE and PT.from_name("???") is PT.THREE_QUESTION_MARKS
    chart = {"WATER": {"FIRE": 2.0}, "GRASS": {"FIRE": 0.5}}
    assert PT.FIRE.damage_multiplier(PT.WATER, type_chart=chart) == 2.0
    assert PT.FIRE.damage_multiplier(PT.WATER, PT.GRASS, type_chart=chart) == 1.0
    assert PT.THREE_QUESTION_MARKS.damage_multiplier(PT.WATER, type_chart=chart) == 1
    assert W.from_showdown_message("move: RainDance") is W.RAINDANCE
    assert W["SNOW"] is W["SNOWSCAPE"]       # the alias member
    assert str(enums.Status.SLP) == "SLP (status) object"


def test_poke_env_battle_objects_compare_equal_to_the_seam_members():
    """The load-bearing identity: a poke-env `Pokemon` hands back OUR enum members."""
    from poke_env.battle.pokemon import Pokemon

    mon = Pokemon(gen=3, species="snorlax")
    mon.status = enums.Status.SLP
    assert mon.status is enums.Status.SLP
    assert enums.PokemonType.NORMAL in mon.types


# ---- the other seams P1 moved out of the fork (one allowlisted home for every "the fork re-exports OURS" pin) ----

def test_to_id_str_is_one_function():
    """`utils.showdown_id.to_id_str` is the definition; the fork's `poke_env.data.normalize` hands back the same object."""
    from poke_env.data import to_id_str as forks
    from poke_env.data.normalize import to_id_str as forks_normalize
    from utils.showdown_id import to_id_str

    assert forks is to_id_str and forks_normalize is to_id_str
    assert to_id_str("King's Rock") == "kingsrock" and to_id_str("Mr. Mime") == "mrmime"


def test_team_packing_is_one_set_of_classes():
    """`utils.team_packing` owns the Showdown paste <-> packed machinery; the fork's teambuilder modules re-export it,
    so the fork's `Player` accepts our builders and `STATS_TO_IDX` is one table."""
    from poke_env.stats import STATS_TO_IDX as forks_stats
    from poke_env.teambuilder import ConstantTeambuilder, Teambuilder, TeambuilderPokemon
    from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder as c2
    from utils import team_packing as TP

    assert Teambuilder is TP.Teambuilder and TeambuilderPokemon is TP.TeambuilderPokemon
    assert ConstantTeambuilder is TP.ConstantTeambuilder is c2
    assert forks_stats is TP.STATS_TO_IDX


def test_the_live_view_unknown_item_sentinel_is_the_forks():
    """`live_view` spells the sentinel `Pokemon._item` starts at (it imports no poke-env); pin it to ``GenData``."""
    from agents.battle.live_view import _UNKNOWN_ITEMS, UNKNOWN_ITEM
    from poke_env.data import GenData

    assert UNKNOWN_ITEM == GenData.UNKNOWN_ITEM == GenData.from_gen(3).UNKNOWN_ITEM
    assert _UNKNOWN_ITEMS == {None, GenData.UNKNOWN_ITEM}


def test_the_global_env_screen_ids_are_the_forks_side_condition_names():
    """`global_env._SCREEN_CONDITIONS` is spelled as ids (no poke-env import): the lower-cased names of the fork's
    ``SideCondition`` members the LiveView keys by, in the encoded order."""
    from agents.observation.global_env import _SCREEN_CONDITIONS
    from poke_env.battle.side_condition import SideCondition

    assert _SCREEN_CONDITIONS == [c.name.lower() for c in (
        SideCondition.REFLECT, SideCondition.LIGHT_SCREEN, SideCondition.SAFEGUARD, SideCondition.MIST)]


def test_the_notable_effect_names_are_members_of_the_forks_effect_enum():
    """`gen3_mechanics.NOTABLE_EFFECT_NAMES` / `has_effect` compare by NAME; every name must be a real ``Effect``."""
    from agents.gen3_mechanics import NOTABLE_EFFECT_NAMES, has_effect
    from poke_env.battle.effect import Effect

    assert all(n in Effect.__members__ for n in NOTABLE_EFFECT_NAMES)
    assert has_effect({Effect.SUBSTITUTE: 1}, "SUBSTITUTE") and not has_effect({Effect.TAUNT: 1}, "SUBSTITUTE")
    assert not has_effect({}, "SUBSTITUTE")


def test_the_eval_roster_names_agree_between_the_poke_env_free_and_the_player_table():
    """`eval_schedule._EVAL_ROSTER` (poke-env-free) and `eval_roster._EVAL_OPPONENT_SPECS` (the player classes) are two
    statements of one roster; `eval_roster` asserts it at import, and this pins the names it re-serves."""
    from agents.training import eval_roster as ER
    from agents.training import eval_schedule as ES

    assert ES.eval_opponent_names() == [n for n, _c, _p in ER._EVAL_OPPONENT_SPECS] == ER.eval_opponent_names()
    assert ES.RANDOM_OPPONENT_NAME == ER.RANDOM_OPPONENT_NAME == ER.opponent_name(ER._EVAL_OPPONENT_SPECS[0][1])
    assert ER.BATTLE_FORMAT == ES.BATTLE_FORMAT and ER.EVAL_SHARD_GAMES == ES.EVAL_SHARD_GAMES


def test_the_frozen_special_types_equal_the_forks_table():
    """`rust_core_obs_layout._SPECIAL_TYPES_PRE_SPLIT` is the fork's ``Move._MOVE_CATEGORY_PER_TYPE_PRE_SPLIT`` SPECIAL
    names, spelled out so the layout generator imports no poke-env (P1 of the retirement). The fork is the oracle
    until P6 deletes it, so the two are pinned equal here."""
    from agents.observation.rust_core_obs_layout import _SPECIAL_TYPES_PRE_SPLIT
    from poke_env.battle.move import Move

    forks = sorted(t.name for t, cat in Move._MOVE_CATEGORY_PER_TYPE_PRE_SPLIT.items() if cat.name == "SPECIAL")
    assert list(_SPECIAL_TYPES_PRE_SPLIT) == forks


def test_the_data_facades_special_types_are_the_forks_real_special_types():
    """`gen3_data.moves._GEN3_SPECIAL_TYPES` (8 types) is the fork's SPECIAL set minus the typeless ``???``."""
    from agents.gen3_data import moves as movedex
    from poke_env.battle.move import Move

    forks = {t for t, cat in Move._MOVE_CATEGORY_PER_TYPE_PRE_SPLIT.items() if cat.name == "SPECIAL"}
    assert movedex._GEN3_SPECIAL_TYPES == forks - {enums.PokemonType.THREE_QUESTION_MARKS}
