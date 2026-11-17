"""THE CLASS, closed: no run-dir writer creates a directory except through ``utils.run_dir``.

A run dir deleted or moved aside under a live run must stay gone; a writer that ``mkdir -p``\\ s its target
resurrects it at its old name as a half-run (2026-11-14). Units A-E routed every run-dir writer through the helpers
(``create_run_root`` at startup, ``ensure_subdir`` below an existing root, ``atomic_write(root=)``); this scan keeps
them there. It is an AST scan of the modules in :data:`RUN_DIR_WRITER_MODULES` (THE list, held only here): a call
``os.makedirs`` / ``os.mkdir`` / ``<path>.mkdir`` (``Path.mkdir``) / a bare ``makedirs`` or ``mkdir`` fails it.
A module that imports ``utils.run_dir`` is a run-dir writer by declaration and must be on the list, so a new writer
cannot opt out by being forgotten. It reads source and runs none of the product, so it declares the ``static`` tier.
The allowlist is EMPTY: fix the writer, never exempt a call.
"""
import ast

import pytest

from utils.paths import repo_root

pytestmark = pytest.mark.static

#: THE run-dir writer modules: every module that writes into a live run's dir.
RUN_DIR_WRITER_MODULES = (
    "src/agents/model/snapshot.py",
    "src/agents/training/eval_callback.py",
    "src/agents/training/eval_collect.py",
    "src/agents/training/eval_launch.py",
    "src/agents/training/eval_sharding/pool.py",
    "src/agents/training/eval_sharding/results.py",
    "src/agents/training/pool_seed.py",
    "src/agents/training/rust_eval/executor.py",
    "src/agents/training/rust_eval/traces.py",
    "src/agents/training/selfplay_callback.py",
    "src/agents/training/snapshot_ladder.py",
    "src/agents/training/snapshot_pool.py",
    "src/agents/training/sprt_promotion.py",
    "src/agents/training/tb_inherit.py",
    "src/agents/training/team_winrate_callback.py",
    "src/agents/training/train_logger.py",
    "src/agents/training/value_sidecar.py",
    "src/agents/training/weights_link.py",
    "src/main/launcher/child.py",
    "src/main/launcher/run.py",
    "src/main/train/deferred_abort.py",
    "src/main/train/lifecycle.py",
    "src/main/train/run_io.py",
    "src/main/train_rl_agent.py",
)

#: The helpers themselves: the only modules that may create a run-dir level (``atomic_io`` creates below its
#: ``root=`` through ``ensure_subdir``; without one it writes non-run files).
_HELPERS = ("src/utils/run_dir.py", "src/utils/atomic_io.py")

_CREATORS = frozenset({"makedirs", "mkdir"})


def raw_creators(source: str):
    """``(line, call)`` of every directory-creating call in ``source``: ``<anything>.makedirs(...)``,
    ``<anything>.mkdir(...)`` (``os.``, an alias, a ``Path``) or a bare ``makedirs(...)`` / ``mkdir(...)``."""
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
        if name in _CREATORS:
            yield node.lineno, ast.unparse(f)


def imports_run_dir(source: str) -> bool:
    """Does the module import ``utils.run_dir`` (``from utils.run_dir import ...``, ``import utils.run_dir``,
    ``from utils import run_dir``), at any depth?"""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module == "utils.run_dir":
            return True
        if isinstance(node, ast.ImportFrom) and node.module == "utils" and any(a.name == "run_dir"
                                                                               for a in node.names):
            return True
        if isinstance(node, ast.Import) and any(a.name == "utils.run_dir" for a in node.names):
            return True
    return False


def _read(rel: str) -> str:
    return (repo_root() / rel).read_text(encoding="utf-8")


def test_no_run_dir_writer_creates_a_directory_outside_the_run_dir_helpers():
    offenders = [f"{rel}:{line}: {call}(...)" for rel in RUN_DIR_WRITER_MODULES
                 for line, call in raw_creators(_read(rel))]
    assert not offenders, (
        "a run-dir writer creates a directory itself -- a run dir deleted under a live run would come back at its "
        "old name. Use utils.run_dir: create_run_root (startup), ensure_subdir (below an existing root), or "
        "utils.atomic_io.atomic_write(root=):\n  " + "\n  ".join(offenders))


def test_every_module_that_imports_run_dir_is_on_the_writer_list():
    """A module that imports ``utils.run_dir`` writes into a run dir; the list cannot silently miss it."""
    src = repo_root() / "src"
    missing = []
    for path in sorted(src.rglob("*.py")):
        rel = path.relative_to(repo_root()).as_posix()
        if path.name.endswith("_test.py") or "__pycache__" in path.parts or rel.startswith("src/rust_sim/") \
                or rel in _HELPERS or rel in RUN_DIR_WRITER_MODULES:
            continue
        if "run_dir" in (text := path.read_text(encoding="utf-8", errors="replace")) and imports_run_dir(text):
            missing.append(rel)
    assert not missing, f"modules import utils.run_dir but are not in RUN_DIR_WRITER_MODULES: {missing}"


def test_the_writer_list_names_only_existing_modules():
    gone = [rel for rel in RUN_DIR_WRITER_MODULES if not (repo_root() / rel).is_file()]
    assert not gone, f"RUN_DIR_WRITER_MODULES names files that no longer exist, drop them: {gone}"
    assert len(set(RUN_DIR_WRITER_MODULES)) == len(RUN_DIR_WRITER_MODULES)


def test_the_scan_can_actually_fail():
    """A gate that cannot fail is not a gate (the vacuity family)."""
    bad = ("import os, os as _os\nfrom os import makedirs\nfrom pathlib import Path\n"
           "os.makedirs('a', exist_ok=True)\n_os.mkdir('b')\nPath('c').mkdir(parents=True)\nmakedirs('d')\n")
    assert [line for line, _ in raw_creators(bad)] == [4, 5, 6, 7]
    good = ("# os.makedirs(x) in a comment\n'''os.mkdir(y) in a docstring'''\n"
            "from utils.run_dir import ensure_subdir\nensure_subdir('r', 's')\nX = 'makedirs'\n")
    assert list(raw_creators(good)) == []
    assert imports_run_dir(good) and imports_run_dir("def f():\n    from utils import run_dir\n")
    assert not imports_run_dir("import utils.paths\nrun_dir = 1\n")
