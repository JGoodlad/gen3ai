"""THE CLASS, closed: no module in the tree imports the DELETED Python env core (deletion pass U3, 2026-10-02).

``Gen3Env`` / ``wrappers`` / ``env_factory`` / ``async_vec_env`` / ``bridge_session`` are gone; the Rust env core is
the only core. This AST scan fails any module that imports one (``trainee_spaces_test.py`` pins that the modules
themselves do not exist). It reads source and runs none of the product, so it declares the ``static`` tier; it lived
in ``trainee_spaces_test.py`` (``slow``, 13 s beside a live run) until 2026-11-17, and a text pre-filter (an import of
a module names its last component) keeps the full parse to the few files that could hold one.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from utils.paths import src_path

pytestmark = pytest.mark.static

#: The Python env core — what deletion pass R1 deleted. A module importing one fails here.
PY_CORE = ("agents.training.gen3_env", "agents.training.wrappers", "agents.training.async_vec_env",
           "main.train.env_factory", "utils.bridge.bridge_session")
_SELF = "agents/training/python_env_core_import_gate_test.py"
_LEAVES = tuple(m.rsplit(".", 1)[1] for m in PY_CORE)


def _importers(root: pathlib.Path) -> dict:
    found: dict = {}
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if rel.startswith(("poke_env/", "rust_sim/", "rust_env/")) or "/target/" in rel or rel == _SELF:
            continue
        text = p.read_text()
        if not any(leaf in text for leaf in _LEAVES):
            continue
        for n in ast.walk(ast.parse(text)):
            names = []
            if isinstance(n, ast.ImportFrom) and n.module:
                names = [n.module] + [f"{n.module}.{a.name}" for a in n.names]
            elif isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            hit = set(PY_CORE) & set(names)
            if hit:
                found.setdefault(rel, set()).update(hit)
    return found


def test_no_module_imports_the_deleted_python_env_core():
    found = _importers(src_path())
    assert not found, (f"module(s) import the deleted Python env core: {sorted(found)} — the Rust env core "
                       "is the only core (deletion pass U3); read spaces / class tables / the eval team "
                       "table from agents.training.trainee_spaces, opponent_classes, eval_teams")


def test_the_scan_can_actually_fail(tmp_path):
    """A gate that cannot fail is not a gate (the vacuity family)."""
    (tmp_path / "a.py").write_text("from agents.training import gen3_env\n")
    (tmp_path / "b.py").write_text("def f():\n    import main.train.env_factory\n")
    (tmp_path / "c.py").write_text("# agents.training.wrappers\nX = 'utils.bridge.bridge_session'\n")
    assert _importers(tmp_path) == {"a.py": {"agents.training.gen3_env"}, "b.py": {"main.train.env_factory"}}
