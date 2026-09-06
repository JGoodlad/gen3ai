"""Every ``__main__.py`` under ``src/main`` must guard its call behind ``if __name__ == "__main__"``.

Why this exists (2026-09-06): a session imported ``main.launcher.__main__`` to inspect its parser
and the import EXECUTED the launcher — it pinned a worktree, spawned a training child (PID
2094207), initialised 32 envs and started compiling beside a live GPU arm, and was only stopped
by the caller's tool timeout. A module named ``__main__`` is still importable, and an unguarded
top-level call runs on import. The prober's entry point had the same shape.

The check is an AST scan, so it is 0 ms and needs no subprocess: a top-level expression
statement that is a call, outside an ``if __name__ == "__main__"`` block, fails naming the file.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from utils.paths import src_path

_MAIN_FILES = sorted(src_path("main").rglob("__main__.py"))


def _is_name_guard(node: ast.If) -> bool:
    t = node.test
    return (
        isinstance(t, ast.Compare)
        and isinstance(t.left, ast.Name)
        and t.left.id == "__name__"
        and len(t.comparators) == 1
        and isinstance(t.comparators[0], ast.Constant)
        and t.comparators[0].value == "__main__"
    )


def _unguarded_calls(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(), filename=str(path))
    bad: list[int] = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            bad.append(node.lineno)
        elif isinstance(node, ast.If) and not _is_name_guard(node):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Expr) and isinstance(sub.value, ast.Call):
                    bad.append(sub.lineno)
    return bad


def test_there_are_entry_points_to_check() -> None:
    assert len(_MAIN_FILES) >= 4, _MAIN_FILES


@pytest.mark.parametrize("path", _MAIN_FILES, ids=[str(p.relative_to(src_path())) for p in _MAIN_FILES])
def test_entry_point_call_is_guarded(path: Path) -> None:
    bad = _unguarded_calls(path)
    assert not bad, (
        f"{path.relative_to(src_path())}: top-level call(s) at line(s) {bad} run on IMPORT — wrap them in "
        f"`if __name__ == \"__main__\":` (importing main.launcher.__main__ started a training run on 2026-09-06)"
    )
