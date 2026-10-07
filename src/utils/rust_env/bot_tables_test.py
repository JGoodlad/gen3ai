"""The Python facts the scripted-bot port leans on still hold, and the Rust table (`src/rust_env/src/bots/tables.rs`)
carries them (M5 Lane F, routine).

`tables.rs` was GENERATED from these Python objects by `utils/rust_env/bot_tables.py` until P1 of the poke-env
retirement (T27) froze it as Rust-owned source and deleted the generator (it imported poke-env). What stays here is the
half that is not a freshness check: the Python-side facts (the Target enum, Curse, the two charts) and that the Rust file
states the same ones."""
from __future__ import annotations

from utils.paths import src_path

_TABLES = src_path("rust_env", "src", "bots", "tables.rs")


def test_setup_moves_target_self_as_the_enum_the_bots_compare():
    """F-LF-1 (FIXED): the setup branch of four bots once compared ``move.target == "self"`` — a
    ``Target`` ENUM against a str, never True — so no setup move was ever chosen. The bots now test
    ``move.target is Target.SELF`` and the port tests the table's NAME ``"SELF"``
    (``calc::target_is_self``). Pins both halves: the Python enum says SELF for the self-boosting
    setup moves, and the generated Rust table carries the same name. Curse's dex target is NOT
    self and it has no static boosts, so the bots read it through ``self_setup_boosts`` instead
    (owner 2026-09-29) — pinned by ``test_curse_reaches_the_bots_only_through_self_setup_boosts``."""
    from poke_env.battle.move import Move
    from poke_env.battle.target import Target

    for mid in ("swordsdance", "dragondance", "calmmind", "bulkup"):
        m = Move(mid, 3)
        assert m.target is Target.SELF, mid
        assert f'MoveRow {{ id: "{mid}",' in _TABLES.read_text()
        row = next(ln for ln in _TABLES.read_text().splitlines() if f'id: "{mid}",' in ln)
        assert 'target: Some("SELF")' in row, (mid, row)
    assert Move("curse", 3).target is not Target.SELF


def test_curse_reaches_the_bots_only_through_self_setup_boosts():
    """Curse as setup: poke-env's ``Move("curse")`` has target NORMAL and no boosts, so the four
    setup steps read ``baselines.self_setup_boosts`` — a non-Ghost's Curse gives the generated
    ``CURSE_NON_GHOST_BOOSTS`` (+1 Atk, +1 Def, -1 Spe), a Ghost's gives None. The heuristic setup
    rule counts RAISED stages (``sum(v for v > 0) >= 2``); that equals the old plain sum for every
    other move only because no Target.SELF gen-3 move LOWERS a stat — pinned here."""
    from poke_env.battle.move import Move
    from poke_env.battle.target import Target
    from poke_env.data import GenData
    from poke_env.player.baselines import CURSE_NON_GHOST_BOOSTS

    c = Move("curse", 3)
    assert c.target is Target.NORMAL and c.boosts is None
    assert CURSE_NON_GHOST_BOOSTS == {"atk": 1, "def": 1, "spe": -1}
    assert 'CURSE_NON_GHOST_BOOSTS: &[(&str, i32)] = &[("atk", 1), ("def", 1), ("spe", -1)];' in _TABLES.read_text()
    lowering = [m for m in GenData.from_gen(3).moves
                if Move(m, 3).target is Target.SELF and Move(m, 3).boosts and min(Move(m, 3).boosts.values()) < 0]
    assert lowering == [], lowering

def test_the_two_type_charts_are_distinct_sources_with_equal_values():
    """Kept as TWO tables (each function reads its own); equal today."""
    from agents import gen3_mechanics as GM
    from poke_env.data import GenData

    assert GenData.from_gen(3).type_chart == GM._type_chart


def test_the_move_rows_equal_the_python_move_properties_the_bots_read():
    """The frozen `MOVES` table, row for row, against ``Move(id, 3)`` — what the deleted generator's freshness test
    held. `tables.rs` is hand-owned now, so a typo in one of its ~350 rows would otherwise be silent until a bot
    chose differently; this reads every row's id, base power, type, category, accuracy and expected hits (as IEEE bits),
    target and boosts back out of the Rust text and compares them with the fork's ``Move``."""
    import re
    import struct

    from poke_env.battle.move import Move
    from poke_env.data import GenData

    def bits(x) -> str:
        return "0x" + format(struct.unpack("<Q", struct.pack("<d", float(x)))[0], "016x")

    row_re = re.compile(
        r'MoveRow \{ id: "(\w+)", base_power: (\d+), typ: "(\w+)", category: "(\w+)", '
        r'accuracy: f64::from_bits\((0x[0-9a-f]+)\), expected_hits: f64::from_bits\((0x[0-9a-f]+)\), '
        r'target: (None|Some\("(\w+)"\)), boosts: (None|Some\(&\[(.*)\]\)) \},')
    rows = {m.group(1): m for m in map(row_re.search, _TABLES.read_text().splitlines()) if m}
    ids = sorted(set(GenData.from_gen(3).moves) | {"recharge", "fight"})
    assert sorted(rows) == ids, (len(rows), len(ids))
    for mid in ids:
        m, g = Move(mid, 3), rows[mid]
        assert int(g.group(2)) == int(m.base_power), mid
        assert (g.group(3), g.group(4)) == (m.type.name, m.category.name), mid
        assert (g.group(5), g.group(6)) == (bits(m.accuracy), bits(m.expected_hits)), mid
        assert g.group(8) == (None if m.target is None else m.target.name), mid
        want = None if m.boosts is None else [(k, int(v)) for k, v in m.boosts.items()]
        got = None if g.group(9) == "None" else [(k, int(v)) for k, v in re.findall(r'\("(\w+)", (-?\d+)\)', g.group(10))]
        assert got == want, (mid, got, want)
