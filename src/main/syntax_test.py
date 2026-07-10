"""Repo-wide syntax gate — every first-party module must parse.

Exists because a shipped edit to `train_rl_agent.py` once landed a `SyntaxError` (a keyword arg
inserted before a positional in a call) that NO test caught: the entry-point modules are never
imported by the unit suite, so the error only surfaced at launch, crashing the run at startup.
`ast.parse` is milliseconds per file and catches that whole failure class at test time."""

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent  # src/
_FILES = sorted(
    p for p in SRC.rglob("*.py")
    if "poke_env" not in p.parts        # vendored fork — parsed by its own tooling
    and "__pycache__" not in p.parts
)


@pytest.mark.parametrize("path", _FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_module_parses(path):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
