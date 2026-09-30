"""The bots' generated Rust table is current, and the Python facts the port leans on still hold
(M5 Lane F, routine)."""
from __future__ import annotations

from utils.rust_env import bot_tables as BT


def test_the_committed_table_is_a_fresh_render():
    assert BT.path().read_text() == BT.render(), \
        "src/rust_env/src/bots/tables.rs is stale — run `python -m utils.rust_env.bot_tables --write`"


def test_move_target_never_equals_the_string_self():
    """F-LF-1: the setup branch of four bots compares ``move.target == "self"``, a ``Target`` ENUM
    against a str — never True, so the Rust port hard-codes it (``calc::target_is_self_str``). If
    poke-env or a bot ever makes that comparison true, this fails and the port must follow."""
    from poke_env.battle.move import Move

    for mid in ("swordsdance", "dragondance", "calmmind", "bulkup", "curse"):
        m = Move(mid, 3)
        assert m.target is not None and (m.target == "self") is False, mid


def test_the_two_type_charts_are_distinct_sources_with_equal_values():
    """Kept as TWO tables (each function reads its own); equal today."""
    from agents import gen3_mechanics as GM
    from poke_env.data import GenData

    assert GenData.from_gen(3).type_chart == GM._type_chart
