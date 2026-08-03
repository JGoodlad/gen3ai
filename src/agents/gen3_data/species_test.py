import pytest

from agents.gen3_data import species
from agents.gen3_data.species import SpeciesData


def test_known_species_fields():
    tt = species.species_data("tyranitar")
    assert isinstance(tt, SpeciesData)
    assert tt.id == "tyranitar"
    assert tt.num == 248
    assert tt.name == "Tyranitar"
    assert tt.base_stats == {"atk": 134, "def": 110, "hp": 100, "spa": 95, "spd": 100, "spe": 61}


def test_get_unknown_returns_none():
    assert species.get("notamon") is None
    assert species.get(None) is None


def test_species_data_raises_on_unknown():
    with pytest.raises(KeyError):
        species.species_data("notamon")


def test_dex_covers_all_gen3():
    # 386 BASE-form species (Bulbasaur #1 .. Deoxys #386) + 33 gen-3 alternate/cosmetic
    # FORMES (gen3_species_formes_v1) = 419 rows.
    assert len(species.raw()) == 419
    assert len(species._dex()) == 419
    assert len(species.base_form_ids()) == 386


def test_raw_matches_dex():
    raw = species.raw()
    for sid, sd in species._dex().items():
        assert sd.num == raw[sid]["num"]


# --- gen3_species_formes_v1: the FORME coverage + num-shadowing guards --------------------- #
#
# The bug this pins: `build_species` used to drop EVERY non-base forme, so the data layer
# could not describe species gen 3 genuinely has. Measured, that made 6.6% of gen3
# random-battle TEAMS (~14% of battles) fail to construct in the `src/rust_sim` port with
# `unknown species "Unown N"` / `"Deoxys-Speed"` — a DATA gap, not an engine gap. A
# re-introduced base-only filter must fail HERE, not as a panic in a training run.

# The gen-3 forme families, from the resolved `Dex.mod('gen3')` (the oracle; the producer-side
# gate is `node src/rust_sim/harness/dump_gen3_mechanics.js --check`).
_DEOXYS_FORMES = ("deoxysattack", "deoxysdefense", "deoxysspeed")
_CASTFORM_FORMES = ("castformrainy", "castformsnowy", "castformsunny")
_UNOWN_FORMES = tuple(f"unown{c}" for c in "bcdefghijklmnopqrstuvwxyz") + (
    "unownexclamation", "unownquestion")


def test_gen3_alternate_formes_are_describable():
    for sid in _DEOXYS_FORMES + _CASTFORM_FORMES + _UNOWN_FORMES:
        sd = species.species_data(sid)          # raises if the data layer lost the row
        assert sd.base_species is not None, sid
        assert sd.base_stats and sd.types, sid


def test_deoxys_formes_carry_their_own_base_stats():
    # Deoxys formes DIFFER in base stats — a wrong row here would feed the model
    # plausible-but-false numbers, which is worse than the missing row was.
    assert species.species_data("deoxys").base_stats["spe"] == 150
    assert species.species_data("deoxysattack").base_stats == {
        "atk": 180, "def": 20, "hp": 50, "spa": 180, "spd": 20, "spe": 150}
    assert species.species_data("deoxysdefense").base_stats == {
        "atk": 70, "def": 160, "hp": 50, "spa": 70, "spd": 160, "spe": 90}
    assert species.species_data("deoxysspeed").base_stats == {
        "atk": 95, "def": 90, "hp": 50, "spa": 95, "spd": 90, "spe": 180}


def test_unown_cosmetic_formes_clone_the_base():
    # Unown letters are COSMETIC formes: Showdown builds them as a clone of the base with
    # only the display name changed, so stats/types/num must be IDENTICAL to Unown's.
    base = species.species_data("unown")
    for sid in _UNOWN_FORMES:
        sd = species.species_data(sid)
        assert (sd.num, sd.base_stats, sd.types) == (base.num, base.base_stats, base.types), sid
        assert sd.name != base.name, sid


def test_castform_weather_formes_are_battle_only_and_retyped():
    assert species.species_data("castform").types == ("NORMAL",)
    assert species.species_data("castformsunny").types == ("FIRE",)
    assert species.species_data("castformrainy").types == ("WATER",)
    assert species.species_data("castformsnowy").types == ("ICE",)
    for sid in _CASTFORM_FORMES:
        assert species.species_data(sid).battle_only is True, sid


def test_base_form_ids_is_one_row_per_dex_num():
    """The SAFETY contract: a forme SHARES its base's national-dex num, and `num` is what the
    obs species channel and every ``table[species.num] = …`` GPU buffer are keyed by. So the
    base-form view must be a BIJECTION onto the nums — otherwise a num-indexed builder that
    iterated ``raw()`` would be last-write-wins and a forme would silently redefine the base."""
    base_ids = species.base_form_ids()
    nums = [species.species_data(sid).num for sid in base_ids]
    assert len(nums) == len(set(nums)) == 386
    assert set(nums) == set(range(1, 387))
    # Every non-base row must (a) be excluded from the base view and (b) alias a real base.
    for sid, sd in species._dex().items():
        if sd.base_species is None:
            continue
        assert sid not in set(base_ids), sid
        assert species.species_data(sd.base_species).num == sd.num, sid
