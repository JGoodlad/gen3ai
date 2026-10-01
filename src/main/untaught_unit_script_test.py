"""The battery's untaught-8 UNIT driver (``designs/research_state/measurements/n0_endofrun_2026-09-27/
scripts/gu_unit.py``) — the per-battle body every learner-battery / lineage untaught read runs.

It lives outside ``src/`` as a measurement script, so nothing imported it, and ``ecd2be00`` (the
ObservationDebugger deletion) removed ``untaught_meter._strip_debugger`` under it: every unit then died
at its model load, silently for the tree, loudly only for whoever ran a read (F-SZ-1, 2026-10-01).

* the STATIC test (routine, ~free): every ``alias.attr`` the script's ``main`` reads, where ``alias`` is
  a module it imports, must exist on that module — a deleted or renamed symbol FAILS here;
* the UNIT test (``sim``): one real battle through the script's own ``main`` (``CHUNK`` stubbed to 1),
  on the registered opponent and a lineage checkpoint, writing its one fsynced row. Skips without the
  run archive.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import sys

import pytest

from utils.paths import main_models_dir, repo_path

SCRIPT = repo_path("designs", "research_state", "measurements", "n0_endofrun_2026-09-27", "scripts", "gu_unit.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("gu_unit_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _module_aliases(fn: ast.FunctionDef) -> dict:
    """``alias -> dotted module`` for every ``from pkg import mod as alias`` / ``import a.b as alias``
    inside ``fn`` whose target is itself a MODULE (attributes of those are what this test resolves)."""
    out = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                full = f"{node.module}.{a.name}"
                try:
                    is_module = importlib.util.find_spec(full) is not None
                except ModuleNotFoundError:           # `from module import Class`: the parent is no package
                    is_module = False
                if is_module:
                    out[a.asname or a.name] = full
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    out[a.asname] = a.name
    return out


def test_every_module_attribute_the_unit_driver_reads_exists():
    tree = ast.parse(SCRIPT.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    aliases = _module_aliases(main)
    assert {"engine", "cli"} <= set(aliases), aliases          # the two it must read through
    missing = []
    for node in ast.walk(main):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
            mod = importlib.import_module(aliases[node.value.id])
            if not hasattr(mod, node.attr):
                missing.append(f"{aliases[node.value.id]}.{node.attr} (line {node.lineno})")
    assert not missing, f"gu_unit.py reads symbols that no longer exist: {missing}"


def test_the_static_check_has_teeth(tmp_path):
    """A planted read of a deleted symbol is caught by the same walk."""
    src = SCRIPT.read_text().replace("engine.pool_sequence(", "engine._strip_debugger(", 1)
    main = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    aliases = _module_aliases(main)
    bad = [n.attr for n in ast.walk(main) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
           and n.value.id in aliases and not hasattr(importlib.import_module(aliases[n.value.id]), n.attr)]
    assert bad == ["_strip_debugger"]


@pytest.mark.sim
def test_one_unit_plays_and_writes_its_row(tmp_path, monkeypatch):
    models = main_models_dir()
    if models is None:
        pytest.skip("no run archive (models/) on this box")
    opp = models / "ai_v14_01_base" / "snapshots" / "snapshot_000024000000.zip"
    ref = models / "ai_v14_01_base" / "checkpoints" / "checkpoint_4800000_steps.zip"
    if not (opp.exists() and ref.exists()):
        pytest.skip("the registered untaught opponent / the N0 checkpoint is not in this archive")
    mod = _load_script()
    monkeypatch.setattr(mod, "CHUNK", 1)                      # one battle: indices [0, 1)
    rows = tmp_path / "rows"
    monkeypatch.setattr(sys, "argv", ["gu_unit.py", "--label", "T", "--ref", str(ref), "--opponent", str(opp),
                                      "--team-index", "0", "--chunk", "0", "--rows", str(rows)])
    assert mod.main() == 0
    lines = (rows / "T" / "t0" / "c00.jsonl").read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["j"] == 0 and row["finished"] == 1 and row["team_index"] == 0
