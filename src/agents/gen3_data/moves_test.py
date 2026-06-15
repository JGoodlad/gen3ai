import pytest
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon_type import PokemonType

from agents.gen3_data import moves as movedex
from agents.gen3_data.moves import MoveData


def test_known_move_fields():
    tb = movedex.move_data("thunderbolt")
    assert tb.id == "thunderbolt"
    assert tb.num == 85
    assert tb.base_power == 95
    assert tb.type is PokemonType.ELECTRIC
    assert tb.category is MoveCategory.SPECIAL
    assert tb.accuracy == 100
    assert tb.has_secondary is True
    assert tb.has_recoil is False
    assert tb.is_damaging is True


def test_physical_move_category_from_type():
    # Gen3 category is type-based: Ground is a physical type.
    eq = movedex.move_data("earthquake")
    assert eq.type is PokemonType.GROUND
    assert eq.category is MoveCategory.PHYSICAL
    assert eq.is_damaging is True


def test_status_move_not_damaging():
    toxic = movedex.move_data("toxic")
    assert toxic.category is MoveCategory.STATUS
    assert toxic.base_power == 0
    assert toxic.is_damaging is False


def test_curse_typeless_loads():
    # Curse stores type "???" in the data → THREE_QUESTION_MARKS; 0-power → STATUS.
    curse = movedex.move_data("curse")
    assert curse.type is PokemonType.THREE_QUESTION_MARKS
    assert curse.category is MoveCategory.STATUS
    assert curse.is_damaging is False


def test_never_miss_flag_carried():
    # swift bypasses accuracy/evasion in the data → never_miss True.
    assert movedex.move_data("swift").never_miss is True
    assert movedex.move_data("thunderbolt").never_miss is False


def test_get_unknown_returns_none():
    assert movedex.get("notarealmove") is None
    assert movedex.get(None) is None


def test_move_data_raises_on_unknown():
    with pytest.raises(KeyError):
        movedex.move_data("notarealmove")


def test_is_damaging_helper():
    assert movedex.is_damaging("surf") is True
    assert movedex.is_damaging("toxic") is False
    assert movedex.is_damaging("notarealmove") is False  # unknown → not assumed damaging
    assert movedex.is_damaging(None) is False


def test_borrowed_enums_are_the_spec_enums():
    # Discipline check: the dex files data under poke-env's value-enums as keys —
    # type/category ARE PokemonType/MoveCategory members, so callers can key their
    # own tables by them (we never call methods on them).
    eq = movedex.move_data("earthquake")
    assert isinstance(eq.type, PokemonType)
    assert isinstance(eq.category, MoveCategory)


def test_all_moves_load_with_valid_enums():
    # Every JSON entry builds cleanly into the borrowed enums — proves there's no
    # transcription gap between the data's type/category names and the enums.
    dex = movedex._dex()
    assert len(dex) >= 300
    for md in dex.values():
        assert isinstance(md, MoveData)
        assert isinstance(md.type, PokemonType)
        assert isinstance(md.category, MoveCategory)
        assert 0 <= md.accuracy <= 100


def test_frozen():
    md = movedex.move_data("tackle")
    with pytest.raises(Exception):
        md.base_power = 999  # frozen dataclass


# ============================================================================
# gen3_move_effects_v1 — action-aligned effect classification
# ============================================================================

def test_effect_flags_setup_moves():
    # Declarative self-positive boosts → is_boost.
    for mid in ("swordsdance", "dragondance", "calmmind", "tailglow", "meditate"):
        md = movedex.move_data(mid)
        assert md.is_boost is True, mid
        assert md.is_heal is False and md.status_inflicted is None


def test_effect_flags_belly_drum_callback_override():
    # Belly Drum's +6 Atk is implemented in a JS onHit callback, so it has NO
    # declarative `boosts` field — the curated override in the builder must catch it.
    bd = movedex.move_data("bellydrum")
    assert bd.is_boost is True
    assert bd.is_heal is False


def test_effect_flags_memento_is_not_boost():
    # Memento's boosts target the FOE (and the user faints) — never setup.
    assert movedex.move_data("memento").is_boost is False


def test_effect_flags_curse_resolved_live_not_static():
    # Curse's setup is type-conditional (non-Ghost user only) → resolved in the
    # encoder from the live user's type, NOT a static flag here.
    assert movedex.move_data("curse").is_boost is False


def test_effect_flags_heal_via_flags_heal():
    # flags.heal catches declarative AND callback-only heals (weather-scaled / Rest / Wish).
    for mid in ("recover", "softboiled", "moonlight", "synthesis", "morningsun",
                "rest", "wish", "swallow", "slackoff", "milkdrink"):
        assert movedex.move_data(mid).is_heal is True, mid
    # Drain attacks and Leech Seed are NOT dedicated heals (no flags.heal).
    for mid in ("gigadrain", "leechseed", "painsplit"):
        assert movedex.move_data(mid).is_heal is False, mid


def test_effect_flags_protect_phaze_hazard():
    for mid in ("protect", "detect", "endure"):
        assert movedex.move_data(mid).is_protect is True, mid
    for mid in ("roar", "whirlwind"):
        assert movedex.move_data(mid).is_phaze is True, mid
    assert movedex.move_data("spikes").is_hazard is True


def test_effect_flags_status_inflicted():
    assert movedex.move_data("toxic").status_inflicted == "tox"
    assert movedex.move_data("thunderwave").status_inflicted == "par"
    assert movedex.move_data("willowisp").status_inflicted == "brn"
    assert movedex.move_data("spore").status_inflicted == "slp"
    # Damaging moves carry status only as a secondary chance — NOT surfaced here.
    assert movedex.move_data("bodyslam").status_inflicted is None
    assert movedex.move_data("icebeam").status_inflicted is None


def test_effect_flags_status_cure():
    # gen3_status_cure_moves_v1: curated onHit cures. Refresh = self; Heal Bell / Aromatherapy = team.
    assert movedex.move_data("refresh").cures_self_status is True
    assert movedex.move_data("refresh").cures_team_status is False
    for mid in ("healbell", "aromatherapy"):
        assert movedex.move_data(mid).cures_team_status is True, mid
        assert movedex.move_data(mid).cures_self_status is False, mid
    # Rest is a heal+sleep, NOT a status-clear (it replaces the status with sleep) → neither bit.
    assert movedex.move_data("rest").cures_self_status is False
    assert movedex.move_data("rest").cures_team_status is False
    # A plain heal / attack carries neither cure bit.
    for mid in ("recover", "surf", "calmmind"):
        md = movedex.move_data(mid)
        assert md.cures_self_status is False and md.cures_team_status is False, mid


def test_status_cure_sets_are_exactly_the_curated_moves():
    """Pin the curated cure sets so a builder regression (a wrong add/drop) fails CI. The cure is
    an onHit callback with no declarative field, so these are the only gen3 moves that clear status
    without inflicting one (Refresh self; Heal Bell / Aromatherapy whole-party)."""
    assert _flagged(lambda md: md.cures_self_status) == {"refresh"}
    assert _flagged(lambda md: md.cures_team_status) == {"healbell", "aromatherapy"}
    # mutually exclusive scopes — no move is both
    assert _flagged(lambda md: md.cures_self_status and md.cures_team_status) == set()


# ----------------------------------------------------------------------------
# Data-QUALITY guards (not just wiring): the effect flags must agree with the
# project's INDEPENDENTLY-curated, gen-aware move sets, and no damaging move may
# ever carry a utility flag. These pin the data so a future builder change that
# silently mis-classifies a move fails CI. The exhaustive callback-aware audit
# against the Showdown .ts source lives in tools/ docs / the audit script; this is
# the always-on corroboration that needs no submodule.
# ----------------------------------------------------------------------------

def _flagged(pred):
    return {m for m, md in movedex._dex().items() if pred(md)}


def test_no_damaging_move_carries_a_utility_flag():
    """The 6 utility flags are scoped to 0-power moves — a damaging move is already
    distinguishable at the head by its base power + type multiplier, and Showdown carries
    status/heal/boost on damaging moves only as secondary/drain/charge side-effects (Dream
    Eater, Leech Life, Rage, Skull Bash). None of those may flip a utility bit."""
    bad = {
        m: md.base_power for m, md in movedex._dex().items()
        if md.base_power > 0 and (
            md.is_boost or md.is_heal or md.is_protect or md.is_phaze
            or md.is_hazard or md.status_inflicted is not None
            or md.cures_self_status or md.cures_team_status
        )
    }
    assert bad == {}, f"damaging moves wrongly flagged as utility: {bad}"


def test_protect_phaze_match_curated_sets_exactly():
    from agents.gen3_mechanics import INVULNERABLE_MOVES, PHAZING_MOVES
    assert _flagged(lambda md: md.is_protect) == set(INVULNERABLE_MOVES)
    assert _flagged(lambda md: md.is_phaze) == set(PHAZING_MOVES)


def test_heal_superset_of_curated_recovery():
    from agents.gen3_mechanics import RECOVERY_MOVES
    heal = _flagged(lambda md: md.is_heal)
    missing = set(RECOVERY_MOVES) - heal
    assert not missing, f"curated recovery moves not flagged is_heal: {missing}"
    # Swallow is the only legitimate extra (Stockpile-scaled heal the curated set omits).
    assert heal - set(RECOVERY_MOVES) <= {"swallow"}


def test_status_superset_of_curated_and_maps_to_major_status():
    from agents.gen3_mechanics import STATUS_MOVES
    status = _flagged(lambda md: md.status_inflicted is not None)
    missing = set(STATUS_MOVES) - status
    assert not missing, f"curated status moves not flagged: {missing}"
    assert all(movedex.move_data(m).status_inflicted in {"par", "brn", "psn", "tox", "slp", "frz"}
               for m in status)


def test_boost_covers_gen3_setup_excludes_memento_and_stockpile():
    from agents.gen3_mechanics import SETUP_MOVES
    boost = _flagged(lambda md: md.is_boost)
    # Every curated gen3 setup move (minus Curse, which is type-conditional → resolved live
    # in the encoder, not a static flag) must be flagged.
    expect = {m for m in SETUP_MOVES if m in movedex._dex()} - {"curse"}
    missing = expect - boost
    assert not missing, f"gen3 setup moves not flagged is_boost: {missing}"
    assert "bellydrum" in boost          # callback-only boost — caught by the curated override
    assert "memento" not in boost         # foe-target debuff + self-faint, NOT setup
    assert "stockpile" not in boost       # gen3 Stockpile does NOT raise Def/SpD (gen4 addition)
    assert movedex.move_data("curse").is_boost is False  # static flag stays off (live-resolved)
