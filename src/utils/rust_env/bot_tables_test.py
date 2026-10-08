"""The facts the scripted-bot port leans on still hold, and the Rust table (`src/rust_env/src/bots/tables.rs`) carries
them (M5 Lane F, routine).

`tables.rs` was GENERATED from poke-env's `Move` objects until P1 of the poke-env retirement (T27) froze it as
Rust-owned source; until P6 the deleted fork's `Move(id, 3)` was the ORACLE this file held every row to. The oracle is
now the SAME static data `Move` read — the frozen upstream JSON (`tools/pokemon_data_extractor/upstream/
poke_env_static/moves/gen3moves.json`) — with `Move`'s four derivations ported verbatim below (the pre-split category,
the accuracy fraction, the expected hits, the target's enum name). At the fork's last commit the ported oracle equalled
`Move` on every row (checked before the fork was deleted), so a hand-edit typo in one of the ~375 rows still fails here.
"""
from __future__ import annotations

import json
import re
import struct

from utils.paths import repo_path, src_path

_TABLES = src_path("rust_env", "src", "bots", "tables.rs")
_MOVES_JSON = repo_path("tools", "pokemon_data_extractor", "upstream", "poke_env_static", "moves", "gen3moves.json")

#: poke-env's `Move._MOVE_CATEGORY_PER_TYPE_PRE_SPLIT` SPECIAL types (gen <= 3: a damaging move's category is its TYPE's).
_SPECIAL_TYPES_PRE_SPLIT = {"DARK", "DRAGON", "ELECTRIC", "FIRE", "GRASS", "ICE", "PSYCHIC", "THREE_QUESTION_MARKS",
                            "WATER"}


def _moves() -> dict:
    return json.loads(_MOVES_JSON.read_text())


def _type_name(t: str) -> str:
    return "THREE_QUESTION_MARKS" if t == "???" else t.upper()


def _category(entry: dict) -> str:
    cat = entry["category"].upper()
    if cat in ("PHYSICAL", "SPECIAL"):
        return "SPECIAL" if _type_name(entry["type"]) in _SPECIAL_TYPES_PRE_SPLIT else "PHYSICAL"
    return cat


def _accuracy(entry: dict) -> float:
    acc = entry["accuracy"]
    return 1.0 if acc is True else acc / 100


def _expected_hits(mid: str, entry: dict) -> float:
    if mid in ("triplekick", "tripleaxel"):
        return 1 + 2 * 0.9 + 3 * 0.81
    mh = entry.get("multihit")
    if mh is None:
        return 1
    if isinstance(mh, int):
        return mh
    assert list(mh) == [2, 5], (mid, mh)
    return (2 + 3) / 3 + (4 + 5) / 6


def _target_name(entry: dict):
    t = entry.get("target")
    if t is None:
        return None
    return re.sub(r"(?<!^)(?=[A-Z])", "_", t).upper()


def _row(mid: str) -> str:
    return next(ln for ln in _TABLES.read_text().splitlines() if f'id: "{mid}",' in ln)


def test_setup_moves_target_self_as_the_name_the_bots_compare():
    """F-LF-1 (FIXED): the setup branch of four bots once compared ``move.target == "self"`` — a ``Target`` ENUM against
    a str, never True. The port tests the table's NAME ``"SELF"`` (``calc::target_is_self``): the static data says self
    for the self-boosting setup moves, and the Rust table carries the same name. Curse's dex target is NOT self."""
    moves = _moves()
    for mid in ("swordsdance", "dragondance", "calmmind", "bulkup"):
        assert _target_name(moves[mid]) == "SELF", mid
        assert 'target: Some("SELF")' in _row(mid), mid
    assert _target_name(moves["curse"]) != "SELF"


def test_curse_reaches_the_bots_only_through_self_setup_boosts():
    """Curse as setup: its dex target is NORMAL with no boosts, so the setup steps read ``self_setup_boosts`` — a
    non-Ghost's Curse gives ``CURSE_NON_GHOST_BOOSTS`` (+1 Atk, +1 Def, -1 Spe). The heuristic setup rule counts RAISED
    stages; that equals the plain sum for every other move only because no self-targeting gen-3 move LOWERS a stat."""
    moves = _moves()
    c = moves["curse"]
    assert _target_name(c) == "NORMAL" and not c.get("boosts")
    assert 'CURSE_NON_GHOST_BOOSTS: &[(&str, i32)] = &[("atk", 1), ("def", 1), ("spe", -1)];' in _TABLES.read_text()
    lowering = [m for m, e in moves.items()
                if _target_name(e) == "SELF" and e.get("boosts") and min(e["boosts"].values()) < 0]
    assert lowering == [], lowering


def test_the_two_type_charts_are_distinct_sources_with_equal_values():
    """Kept as TWO tables (each function reads its own); equal today — `gen3_mechanics`' chart against the one the
    acquisition layer computes from the frozen upstream typechart."""
    from agents import gen3_mechanics as GM
    from tools.pokemon_data_extractor import sync

    assert sync.build_type_chart(3) == GM._type_chart


def test_the_move_rows_equal_the_static_move_data_the_bots_read():
    """The frozen `MOVES` table, row for row, against the static move data through `Move`'s ported derivations: id,
    base power, type, category, accuracy and expected hits (as IEEE bits), target and boosts."""

    def bits(x) -> str:
        return "0x" + format(struct.unpack("<Q", struct.pack("<d", float(x)))[0], "016x")

    row_re = re.compile(
        r'MoveRow \{ id: "(\w+)", base_power: (\d+), typ: "(\w+)", category: "(\w+)", '
        r'accuracy: f64::from_bits\((0x[0-9a-f]+)\), expected_hits: f64::from_bits\((0x[0-9a-f]+)\), '
        r'target: (None|Some\("(\w+)"\)), boosts: (None|Some\(&\[(.*)\]\)) \},')
    rows = {m.group(1): m for m in map(row_re.search, _TABLES.read_text().splitlines()) if m}
    moves = _moves()
    assert sorted(rows) == sorted(set(moves) | {"recharge", "fight"}), (len(rows), len(moves))
    for mid, e in moves.items():
        g = rows[mid]
        assert int(g.group(2)) == int(e["basePower"]), mid
        assert (g.group(3), g.group(4)) == (_type_name(e["type"]), _category(e)), mid
        assert (g.group(5), g.group(6)) == (bits(_accuracy(e)), bits(_expected_hits(mid, e))), mid
        assert g.group(8) == _target_name(e), mid
        want = None if not e.get("boosts") else [(k, int(v)) for k, v in e["boosts"].items()]
        got = None if g.group(9) == "None" else [(k, int(v)) for k, v in re.findall(r'\("(\w+)", (-?\d+)\)', g.group(10))]
        assert got == want, (mid, got, want)
