"""The committed Rust present() tables equal what poke-env renders today.

``src/rust_sim/src/present/tables.rs`` is GENERATED from poke-env's own static data — the gen-3
pokedex, the move table, the ``Effect`` lifecycle sets, ``SideCondition``, the type names and the
battle-message ignore set (``gen3_core_present_tables_v1``). ``present()`` must READ a battle the
way poke-env does, so a poke-env data change that is not regenerated fails HERE, the day it lands.
"""

from agents.battle.rust_core_present_tables import OUT, render


def test_the_committed_present_tables_are_current():
    assert OUT.read_text() == render(), (
        "src/rust_sim/src/present/tables.rs is STALE — run "
        "`python -m agents.battle.rust_core_present_tables --write` and rebuild the rust core")


def test_the_tables_are_not_vacuous():
    text = render()
    # Non-vacuity: the tables a present() rule reads must actually be populated.
    for needle in ("pub static SPECIES", "pub static MOVES", "pub static EFFECTS",
                   "pub static SIDE_CONDITIONS", '"metagross"', '"thunderbolt"'):
        assert needle in text, needle
