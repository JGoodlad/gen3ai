"""THE one Hidden Power table and formula (``hidden_power.py``): pinned to the Showdown submodule, the reverse
formula checked on hand-worked spreads, and every former copy now reading the same object."""
import re

import pytest

from agents.gen3_data import hidden_power as H
from utils.paths import repo_path


def _showdown_hpivs() -> dict:
    """``fillStats(typechart.ts HPivs, 31)`` per type (the validator's fill), parsed from the pinned submodule."""
    src = repo_path("deps", "pokemon-showdown", "data", "typechart.ts").read_text()
    out = {}
    for name, body in re.findall(r"^\t(\w+): \{(.*?)^\t\},", src, re.MULTILINE | re.DOTALL):
        m = re.search(r"HPivs: \{([^}]*)\}", body)
        if m is None or name in ("normal", "fairy", "stellar"):
            continue
        ivs = {k: 31 for k in H.STATS}
        ivs.update({k: int(v) for k, v in re.findall(r"(\w+): (\d+)", m.group(1))})
        out[name.capitalize()] = ivs
    return out


def test_the_table_is_showdowns_validator_fill_for_all_16_types():
    found = _showdown_hpivs()
    assert len(found) == 16
    assert H.HP_IVS == found


def test_every_canonical_fill_round_trips_to_its_type_at_power_70():
    assert sorted(t.lower() for t in H.HP_IVS) == sorted(H.FORMULA_TYPES)
    for t in H.FORMULA_TYPES:
        for ivs in (H.ivs_of(t), H.ivs_packed(t)):
            assert H.type_of(ivs) == t and H.power_of(ivs) == 70, t


@pytest.mark.parametrize("ivs,typ,power", [
    # Hand-worked (Bulbapedia's gen-3 formula; weights 1..32 over HP, Atk, Def, Spe, SpA, SpD):
    ([31] * 6, "dark", 70),
    ([0] * 6, "fighting", 30),
    # type x = 1+8+16+32 = 57 -> floor(855 / 63) = 13 (ice); power x = 1+4+8+16+32 = 61 -> floor(2440 / 63) + 30 = 68
    # (the Atk 28 drops bit 1; src/rust_sim/harness/probe_hidden_power_bp_regression_rng.js, HP-ICE-BP68).
    ([31, 28, 30, 31, 31, 31], "ice", 68),
    ([31, 30, 31, 30, 31, 31], "grass", 70),
    ([31, 31, 31, 31, 31, 30], "ice", 70),         # Speed weighs 8: x = 1+2+4+16+32 = 55 -> floor(825 / 63) = 13
    ([31, 31, 31, 31, 30, 31], "steel", 70),       # SpD weighs 32: x = 1+2+4+8+16 = 31 -> floor(465 / 63) = 7
])
def test_the_reverse_formula_on_hand_worked_spreads(ivs, typ, power):
    assert H.type_of(ivs) == typ and H.power_of(ivs) == power
    assert H.type_of(dict(zip(H.STATS, ivs))) == typ


def test_a_missing_stat_reads_31_and_an_unknown_type_raises():
    assert H.type_of({}) == "dark" and H.power_of({"atk": 30}) == 70
    with pytest.raises(KeyError):
        H.ivs_of("fairy")


def test_iv_patterns_lead_with_the_canonical_fill_and_are_every_70_power_pattern():
    total = 0
    for t in H.FORMULA_TYPES:
        pats = H.iv_patterns(t)
        assert pats[0] == H.ivs_packed(t) and len({tuple(p) for p in pats}) == len(pats)
        assert all(H.type_of(p) == t and H.power_of(p) == 70 for p in pats)
        total += len(pats)
    assert total == 64


def test_every_former_copy_reads_the_one_table():
    # Each was an independent table / formula until 2026-10-19 (hidden_power_single_impl_test.py guards new ones).
    from agents.gen3_data import hp_repair
    from agents.info_levers import packed
    from main.anchors import team_overlap
    from utils import gen3_utils

    assert gen3_utils.GEN3_HP_IVS is H.HP_IVS
    assert hp_repair.HP_IVS is H.HP_IVS
    assert packed.HP_TYPES is H.FORMULA_TYPES
    for t in H.FORMULA_TYPES:
        assert packed.hp_ivs(t) == H.ivs_packed(t), t          # was the most-31s pattern: 8 types differed
        assert packed.hp_iv_masks(t) == H.iv_patterns(t), t
        assert packed.hp_type_of(H.ivs_packed(t)) == t
        assert team_overlap.hidden_power_type(H.ivs_of(t)) == t


@pytest.mark.parametrize("t", H.FORMULA_TYPES)
def test_packing_a_typed_hidden_power_with_default_ivs_fills_the_one_table(t):
    # utils.team_packing used poke-env's own table until 2026-10-19: 12 of its 16 rows gave ANOTHER gen-3 type.
    from utils.team_packing import TeambuilderPokemon

    mon = TeambuilderPokemon(species="Snorlax", moves=[f"Hidden Power [{t.capitalize()}]"], ivs=[31] * 6)
    packed = mon.packed
    assert mon.ivs == H.ivs_packed(t) and H.type_of(mon.ivs) == t, packed


def test_the_rust_engines_type_order_is_the_formulas():
    # The Rust reverse formula (state.rs `hidden_power_type`) carries its own copy of getHiddenPower's type list; the
    # crate's tests/hidden_power_fill_test.rs pins its output on typechart.ts's rows.
    src = repo_path("src", "rust_sim", "src", "state.rs").read_text()
    m = re.search(r"const HP_TYPES: \[&str; 16\] = \[(.*?)\];", src, re.DOTALL)
    assert m is not None
    assert tuple(re.findall(r'"(\w+)"', m.group(1))) == H.FORMULA_TYPES
