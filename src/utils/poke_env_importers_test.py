"""The poke-env importer scanner (``utils.poke_env_importers``) — every import shape it must see, every one it
must not, and the ``--init`` / ``--shrink`` allowlist mechanics on a throwaway tree.

The gate that USES the scanner is ``src/poke_env_import_gate_test.py``; this file is what keeps the scanner
itself from going blind (a scanner that silently saw nothing would turn the gate vacuously green).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from utils import poke_env_importers as pei


def _hits(src: str):
    return [(h.module, h.scope, h.form) for h in pei.scan_source(src)]


@pytest.mark.parametrize("src, expected", [
    ("import poke_env\n", [("poke_env", "top", "import")]),
    ("import poke_env.battle.move as m\n", [("poke_env.battle.move", "top", "import")]),
    ("from poke_env import Player\n", [("poke_env", "top", "from")]),
    ("from poke_env.battle.pokemon import Pokemon\n", [("poke_env.battle.pokemon", "top", "from")]),
    # a LAZY import (inside a function / method / lambda-free nested def) still counts
    ("def f():\n    from poke_env.data import GenData\n", [("poke_env.data", "lazy", "from")]),
    ("class C:\n    def m(self):\n        import poke_env.player\n", [("poke_env.player", "lazy", "import")]),
    ("async def f():\n    import poke_env\n", [("poke_env", "lazy", "import")]),
    # TYPE_CHECKING-only imports count, as their own scope; the else arm is runtime
    ("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from poke_env.battle.move import Move\n",
     [("poke_env.battle.move", "type_checking", "from")]),
    ("import typing\nif typing.TYPE_CHECKING:\n    import poke_env\nelse:\n    import poke_env.player\n",
     [("poke_env", "type_checking", "import"), ("poke_env.player", "top", "import")]),
    # a try/except ImportError guard at module level is still TOP
    ("try:\n    import poke_env\nexcept ImportError:\n    poke_env = None\n", [("poke_env", "top", "import")]),
    # literal strings handed to an import-effecting call
    ("import importlib\nm = importlib.import_module('poke_env.battle.move')\n",
     [("poke_env.battle.move", "top", "dynamic")]),
    ("m = __import__('poke_env')\n", [("poke_env", "top", "dynamic")]),
    ("from unittest import mock\ndef t():\n    with mock.patch('poke_env.player.player.Player.x'):\n        pass\n",
     [("poke_env.player.player.Player.x", "lazy", "dynamic")]),
    ("def t(monkeypatch):\n    monkeypatch.setattr('poke_env.battle.move.Move.id', 1)\n",
     [("poke_env.battle.move.Move.id", "lazy", "dynamic")]),
    ("import importlib\ndef t(n):\n    return importlib.import_module(f'poke_env.{n}')\n",
     [("poke_env.", "lazy", "dynamic")]),
    # two imports in one statement
    ("import os, poke_env.data, sys\n", [("poke_env.data", "top", "import")]),
])
def test_the_scan_sees_every_import_shape(src, expected):
    assert _hits(src) == expected, src


@pytest.mark.parametrize("src", [
    "import poke_env_fork\n",                       # a different top-level package whose name STARTS with it
    "import not_poke_env\n",
    "from poke_env_fork import x\n",
    "x = 'poke_env.battle'\n",                      # a string that is not handed to an import-effecting call
    "# import poke_env\n",
    "'''import poke_env'''\n",
    "from . import poke_env\n",                      # a relative import (inside the fork itself)
    "from .poke_env import x\n",
    "def f(obj):\n    setattr(obj, 'poke_env', 1)\n",   # builtin setattr's FIRST arg is the object
    "import os\nos.path.join('poke_env', 'x')\n",
    "pokemon = None\n",
])
def test_the_scan_does_not_flag_look_alikes(src):
    assert pei.scan_source(src) == [], src


def test_a_file_it_cannot_parse_is_loud_not_silent():
    with pytest.raises(pei.ScanError):
        pei.scan_source("import poke_env\ndef (:\n", "broken.py")
    assert pei.scan_source("def (:\n", "broken.py") == []   # no mention of the package: nothing to see


def test_an_importers_scope_is_its_strongest_hit():
    hits = pei.scan_source("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import poke_env\n"
                           "def f():\n    import poke_env.data\n")
    assert pei._make_importer("src/x.py", hits).scope == "lazy"
    hits.append(pei.Hit(1, "poke_env", "top", "import"))
    assert pei._make_importer("src/x.py", hits).scope == "top"


def test_the_test_kind_is_decided_by_the_file_name():
    assert pei.is_test_path("src/a/b_test.py") and pei.is_test_path("conftest.py") and pei.is_test_path("src/c/conftest.py")
    assert not pei.is_test_path("src/a/b.py") and not pei.is_test_path("src/a/obs_build_benchmark.py")
    assert not pei.is_test_path("src/a/b_test_helpers.py")


# ------------------------------------------------------------------------------ the walk, on a throwaway tree

def _tree(tmp_path: Path) -> Path:
    files = {
        "src/agents/a.py": "from poke_env.battle.move import Move\n",
        "src/agents/a_test.py": "import poke_env\n",
        "src/agents/clean.py": "x = 1\n",
        "src/rust_sim/target/gen.py": "import poke_env\n",                           # a build dir: pruned
        "src/web/node_modules/x.py": "import poke_env\n",
        "tools/t/sync.py": "def f():\n    from poke_env.data import GenData\n",
        "scripts/s.py": "print('no import')\n",
        "designs/old/measure.py": "import poke_env\n",                              # out of scope
        "src/main/anchors/peer_scripts/metamon_side.py": "import poke_env.ps_client\n",   # the permanent one
    }
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return tmp_path


def test_the_walk_covers_src_tools_scripts_and_prunes_the_build_dirs(tmp_path):
    inv = pei.scan_repo(_tree(tmp_path))
    assert set(inv.importers) == {"src/agents/a.py", "src/agents/a_test.py", "tools/t/sync.py",
                                  "src/main/anchors/peer_scripts/metamon_side.py"}
    assert inv.importers["tools/t/sync.py"].scope == "lazy"
    assert inv.importers["src/agents/a_test.py"].is_test and not inv.importers["src/agents/a.py"].is_test
    # the permanent peer-process script is scanned but is not something the allowlist must cover
    assert set(inv.allowlistable()) == {"src/agents/a.py", "src/agents/a_test.py", "tools/t/sync.py"}


def test_a_re_vendored_fork_is_scanned_not_pruned(tmp_path):
    """The vendored fork was deleted in T27 P6, so nothing named `poke_env` is pruned any more: a copy that came back
    is a set of NEW importers here (and `poke_env_absent_gate_test` names the package outright)."""
    p = tmp_path / "src" / "poke_env" / "battle" / "move.py"
    p.parent.mkdir(parents=True)
    p.write_text("from poke_env.data import GenData\n")
    assert set(pei.scan_repo(tmp_path).allowlistable()) == {"src/poke_env/battle/move.py"}


def test_init_then_shrink_only_ever_removes(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    (root / "designs/ops").mkdir(parents=True)
    (root / "src").mkdir(exist_ok=True)
    (root / pei.GATE_REL).write_text("FROZEN_NON_TEST_COUNT = 0\nFROZEN_TEST_COUNT = 0\n")
    monkeypatch.setattr(pei, "repo_root", lambda: root)

    assert pei.main(["--init", "--when", "2026-01-01"]) == 0
    listed = pei.read_allowlist(root)
    assert listed == ["src/agents/a.py", "tools/t/sync.py", "src/agents/a_test.py"]    # non-test first
    assert "2026-01-01" in (root / pei.ALLOWLIST_REL).read_text()
    assert pei.main(["--init"]) == 2                       # refuses to regenerate over an existing list

    # retire one importer -> --shrink drops it, keeps the header, and lowers the gate's counts
    (root / "tools/t/sync.py").write_text("x = 1\n")
    assert pei.main(["--shrink"]) == 0
    assert pei.read_allowlist(root) == ["src/agents/a.py", "src/agents/a_test.py"]
    assert (root / pei.ALLOWLIST_REL).read_text().startswith("# poke-env IMPORT ALLOWLIST")
    assert (root / pei.GATE_REL).read_text() == "FROZEN_NON_TEST_COUNT = 1\nFROZEN_TEST_COUNT = 1\n"

    # a NEW importer is never added by --shrink: it refuses (growth is a human + owner decision)
    (root / "src/agents/new.py").write_text("import poke_env\n")
    before = (root / pei.ALLOWLIST_REL).read_text()
    assert pei.main(["--shrink"]) == 2
    assert (root / pei.ALLOWLIST_REL).read_text() == before


def test_shrink_refuses_a_gate_file_it_cannot_rewrite(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    (root / "designs/ops").mkdir(parents=True)
    (root / pei.GATE_REL).write_text("FROZEN_NON_TEST_COUNT = 1   # edited by hand\nFROZEN_TEST_COUNT = 1\n")
    monkeypatch.setattr(pei, "repo_root", lambda: root)
    assert pei.main(["--init"]) == 0
    (root / "tools/t/sync.py").write_text("x = 1\n")
    with pytest.raises(pei.ScanError):
        pei.main(["--shrink"])
