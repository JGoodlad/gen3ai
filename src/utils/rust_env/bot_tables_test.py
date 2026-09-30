"""The bots' generated Rust table is current, and the Python facts the port leans on still hold
(M5 Lane F, routine)."""
from __future__ import annotations

from utils.rust_env import bot_tables as BT


def test_the_committed_table_is_a_fresh_render():
    assert BT.path().read_text() == BT.render(), \
        "src/rust_env/src/bots/tables.rs is stale — run `python -m utils.rust_env.bot_tables --write`"


def test_setup_moves_target_self_as_the_enum_the_bots_compare():
    """F-LF-1 (FIXED): the setup branch of four bots once compared ``move.target == "self"`` — a
    ``Target`` ENUM against a str, never True — so no setup move was ever chosen. The bots now test
    ``move.target is Target.SELF`` and the port tests the table's NAME ``"SELF"``
    (``calc::target_is_self``). Pins both halves: the Python enum says SELF for the self-boosting
    setup moves, and the generated Rust table carries the same name. Curse's dex target is NOT
    self (it has no static boosts either), so the setup branches never pick it."""
    from poke_env.battle.move import Move
    from poke_env.battle.target import Target

    for mid in ("swordsdance", "dragondance", "calmmind", "bulkup"):
        m = Move(mid, 3)
        assert m.target is Target.SELF, mid
        assert f'MoveRow {{ id: "{mid}",' in BT.path().read_text()
        row = next(ln for ln in BT.path().read_text().splitlines() if f'id: "{mid}",' in ln)
        assert 'target: Some("SELF")' in row, (mid, row)
    assert Move("curse", 3).target is not Target.SELF


def test_the_two_type_charts_are_distinct_sources_with_equal_values():
    """Kept as TWO tables (each function reads its own); equal today."""
    from agents import gen3_mechanics as GM
    from poke_env.data import GenData

    assert GenData.from_gen(3).type_chart == GM._type_chart
