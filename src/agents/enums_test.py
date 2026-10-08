"""Pin the owned-enums seam (`agents.enums`).

`agents.enums` DEFINES the four spec-defined value-enums our code uses as keys, with nothing outside the
standard library (P1 of the poke-env retirement, T27; the vendored fork that re-exported them was deleted in P6).
The critical assertions are that

* `Effect` is **never** among them — it is the temporal volatile vocabulary replaced by
  `agents/observation/gen3_effects.py`, and an accidental `Effect` here would quietly reopen that door (pinning
  `__all__` exactly makes it a CI failure, not a silent drift);
* importing the seam loads NO `poke_env` module (a fresh interpreter); and
* the tables the fork used to be the ORACLE for (the screen ids, the notable effect names, the pre-split special
  types) still read the values frozen from it at its last commit.
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


def test_the_helper_methods_survive():
    """The helper methods the deleted fork's battle layer called (kept: they are part of the enums' contract)."""
    PT, W = enums.PokemonType, enums.Weather
    assert PT.from_name("fire") is PT.FIRE and PT.from_name("???") is PT.THREE_QUESTION_MARKS
    chart = {"WATER": {"FIRE": 2.0}, "GRASS": {"FIRE": 0.5}}
    assert PT.FIRE.damage_multiplier(PT.WATER, type_chart=chart) == 2.0
    assert PT.FIRE.damage_multiplier(PT.WATER, PT.GRASS, type_chart=chart) == 1.0
    assert PT.THREE_QUESTION_MARKS.damage_multiplier(PT.WATER, type_chart=chart) == 1
    assert W.from_showdown_message("move: RainDance") is W.RAINDANCE
    assert W["SNOW"] is W["SNOWSCAPE"]       # the alias member
    assert str(enums.Status.SLP) == "SLP (status) object"


def test_to_id_str():
    """`utils.showdown_id.to_id_str` is the one id function (the fork that re-exported it is deleted, P6)."""
    from utils.showdown_id import to_id_str

    assert to_id_str("King's Rock") == "kingsrock" and to_id_str("Mr. Mime") == "mrmime"


# ---- values the deleted fork used to be the ORACLE for (P6 of the poke-env retirement): frozen as literals, captured
# ---- from the fork at its last commit (`c0950b03`), so a change to one of OUR tables still fails here.

def test_the_global_env_screen_ids():
    """`global_env._SCREEN_CONDITIONS`: the lower-cased ``SideCondition`` names, in the encoded order."""
    from agents.observation.global_env import _SCREEN_CONDITIONS

    assert _SCREEN_CONDITIONS == ["reflect", "light_screen", "safeguard", "mist"]


#: The fork's ``Move._MOVE_CATEGORY_PER_TYPE_PRE_SPLIT`` SPECIAL types (gen <= 3: the category is the TYPE's).
_FORK_SPECIAL_TYPES_PRE_SPLIT = ["DARK", "DRAGON", "ELECTRIC", "FIRE", "GRASS", "ICE", "PSYCHIC",
                                 "THREE_QUESTION_MARKS", "WATER"]


def test_the_frozen_special_types():
    """`layout.rs`'s ``SPECIAL_TYPES_PRE_SPLIT`` (Rust-owned since P6) is the pre-split SPECIAL set."""
    from agents.observation.rust_core_obs_layout_test import _str_table, _text

    assert sorted(_str_table(_text(), "SPECIAL_TYPES_PRE_SPLIT")) == _FORK_SPECIAL_TYPES_PRE_SPLIT


def test_the_data_facades_special_types():
    """`gen3_data.moves._GEN3_SPECIAL_TYPES` is that set minus the typeless ``???``."""
    from agents.gen3_data import moves as movedex

    assert sorted(t.name for t in movedex._GEN3_SPECIAL_TYPES) == [
        t for t in _FORK_SPECIAL_TYPES_PRE_SPLIT if t != "THREE_QUESTION_MARKS"]
