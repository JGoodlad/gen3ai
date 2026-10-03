"""The scripted-bot inventory is COMPLETE and CURRENT (M5 Lane F, routine).

Every roster is derived from the CODE (AST — no heavy imports), never from memory: a bot class that
starts playing in any pool without a row in `bot_inventory.ROWS` fails here, and so does a row whose
``used_by`` no longer matches the pools.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import re

from utils.paths import src_path
from utils.rust_env import bot_inventory as BI


def _tree(rel):
    return ast.parse(src_path(*rel.split("/")).read_text(), filename=rel)


def _imports(tree) -> dict:
    """``name -> "module.Name"`` for every ``from module import Name`` in ``tree`` (any depth)."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                out[a.asname or a.name] = f"{node.module}.{a.name}"
    return out


def _resolve(dotted: str) -> type:
    mod, _, name = dotted.rpartition(".")
    return getattr(importlib.import_module(mod), name)


def _canonical(cls) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _classes(names, imports) -> set:
    return {_canonical(_resolve(imports[n])) for n in names}


def _list_assign(tree, target: str) -> list:
    """The Name elements of every ``<target> = [...]`` list literal."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == target for t in tgts) and isinstance(node.value, ast.List):
                found.append(node.value)
    assert found, f"no `{target} = [...]` found — the roster moved; re-derive it here"
    return found


def train_roster() -> set:
    t = _tree("main/train/matchup_setup.py")
    (lst,) = _list_assign(t, "OPPONENT_CLASSES")
    return _classes([e.id for e in lst.elts], _imports(t))


def eval_roster() -> set:
    t = _tree("agents/training/eval_roster.py")
    (lst,) = _list_assign(t, "_EVAL_OPPONENT_SPECS")
    return _classes([tup.elts[1].id for tup in lst.elts], _imports(t))


def _referenced_bots(t) -> set:
    """Every imported scripted-bot class the module REFERENCES (a list literal, an ``append`` loop, a
    call — any Load of the name), so a bot added by any construct is caught. ``RLPlayer`` and the
    other policy players are not bots."""
    from poke_env.player.player import Player

    imports = _imports(t)
    used = {n.id for n in ast.walk(t) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    out = set()
    for name in used & set(imports):
        if not imports[name].startswith(("agents.opponents", "agents.baitbot", "poke_env.player")):
            continue
        obj = _resolve(imports[name])
        if inspect.isclass(obj) and issubclass(obj, Player) and obj is not Player:
            out.add(_canonical(obj))
    return out


def warmstart_roster() -> set:
    return _referenced_bots(_tree("agents/training/warmstart.py"))


def bait_roster() -> set:
    """`make_baitbot_class` returns a SUBCLASS of ``Gen3BaitBotPlayer`` named for its dial; its row is
    the base class's."""
    t = _tree("main/train/matchup_setup.py")
    assert "make_baitbot_class" in ast.unparse(t), "the conditional BaitBot roster entry moved"
    from agents.baitbot import Gen3BaitBotPlayer, make_baitbot_class

    assert issubclass(make_baitbot_class(0.6), Gen3BaitBotPlayer)
    return {_canonical(Gen3BaitBotPlayer)}


def rosters() -> dict:
    return {"train": train_roster(), "train_bait": bait_roster(), "eval": eval_roster(),
            "warmstart": warmstart_roster()}


def test_every_roster_bot_has_a_row():
    rows = BI.by_class()
    for site, classes in rosters().items():
        missing = classes - set(rows)
        assert not missing, (f"{sorted(missing)} play in the {site!r} pool ({BI.SITES[site]}) with no row in "
                             "utils/rust_env/bot_inventory.py — inventory a bot before it plays")


def test_used_by_matches_the_rosters():
    rs = rosters()
    for row in BI.ROWS:
        actual = {site for site, classes in rs.items() if row.cls in classes}
        assert set(row.used_by) == actual, f"{row.name}: the table says {sorted(row.used_by)}, the code says {sorted(actual)}"
        assert set(row.used_by) <= set(BI.SITES), row


def test_every_player_class_in_the_bot_modules_has_a_row():
    from poke_env.player.player import Player

    rows = BI.by_class()
    for mod in ("agents.opponents", "agents.baitbot", "poke_env.player.baselines"):
        m = importlib.import_module(mod)
        for _, cls in inspect.getmembers(m, inspect.isclass):
            if cls.__module__ == mod and issubclass(cls, Player):
                assert _canonical(cls) in rows, f"{_canonical(cls)} is a bot class with no inventory row"


def test_display_names_agree_with_the_eval_table():
    t = _tree("agents/training/eval_roster.py")
    (lst,) = _list_assign(t, "_EVAL_OPPONENT_SPECS")
    imports = _imports(t)
    rows = BI.by_class()
    for tup in lst.elts:
        name, cls = tup.elts[0].value, _canonical(_resolve(imports[tup.elts[1].id]))
        assert rows[cls].name == name, (cls, rows[cls].name, name)


def test_a_ported_row_names_a_rust_bot():
    mod = src_path("rust_env", "src", "bots", "mod.rs")
    text = mod.read_text() if mod.exists() else ""
    for row in BI.ported():
        assert re.search(rf"\b{row.rust}\b", text), f"{row.name}: Kind::{row.rust} is not defined in {mod}"


def test_rng_streams_are_the_known_three():
    for row in BI.ROWS:
        assert row.rng and set(row.rng) <= {"choice", "protect", "bait"}, row


def test_the_inventory_has_teeth(monkeypatch):
    """A dropped row fails the roster check; a bot added to a pool by ANY construct (not only the
    list literal) fails the used_by check."""
    import pytest

    monkeypatch.setattr(BI, "ROWS", tuple(r for r in BI.ROWS if r.name != "staller_v2"))
    with pytest.raises(AssertionError, match="Gen3StallerV2Player"):
        test_every_roster_bot_has_a_row()
    monkeypatch.undo()

    real = _tree

    def with_extra_bot(rel):
        t = real(rel)
        if rel != "agents/training/warmstart.py":
            return t
        return ast.parse(ast.unparse(t) + "\nfrom poke_env.player.baselines import MaxBasePowerPlayer\n"
                                          "extra = []\nextra.append(MaxBasePowerPlayer)\n")

    monkeypatch.setattr(__import__(__name__, fromlist=["_tree"]), "_tree", with_extra_bot)
    with pytest.raises(AssertionError, match="max_base_power"):
        test_used_by_matches_the_rosters()
