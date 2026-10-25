"""The ONE inventory of the static gates: every test file that declares ``pytest.mark.static``.

The prose that counts them (the count sentence in the root ``CLAUDE.md``, "N static gates run
inside the suite" in ``docs/DEVELOPING.md`` and ``docs/RUNNING.md``) used to be hand-counted and disagreed three
ways (2026-10-08). The marker is the declaration (a tier is DECLARED, never inferred), so this module reads
the declaration with an AST scan and ``static_gates_test.py`` holds every count and table to it.

    python -m utils.static_gates          # one path per line, then the count
    python -m utils.static_gates --paths  # the paths only — THE FAST STATIC TIER:
    pytest $(python -m utils.static_gates --paths) -q -n 2

WHY ``--paths`` (2026-10-12). ``pytest src/ -m static`` COLLECTS the whole tree (~12,000 tests, ~50 s per
process on the agent cores beside a live run) to keep 240; naming the 17 files collects only them. Measured
on cores 5-7 / 13-15 at nice 19, warm caches, ``-n 2``: **64 s by file list against 102 s by marker, the
identical 240 tests** (``--collect-only`` diffed). The two selections are equal only while no test OUTSIDE
these files carries the marker — :func:`stray_static_marks` finds any, and ``static_gates_test.py`` fails
on one.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import List

from utils.paths import src_path

NUMBER_WORDS = {
    "Ten": 10, "Eleven": 11, "Twelve": 12, "Thirteen": 13, "Fourteen": 14, "Fifteen": 15, "Sixteen": 16,
    "Seventeen": 17, "Eighteen": 18, "Nineteen": 19, "Twenty": 20, "Twenty-one": 21, "Twenty-two": 22,
    "Twenty-three": 23, "Twenty-four": 24, "Twenty-five": 25,
}


def declares_static(path: Path) -> bool:
    """Does the file's module-level ``pytestmark`` assignment include ``pytest.mark.static``?"""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "pytestmark" for t in node.targets):
            if "pytest.mark.static" in ast.unparse(node.value):
                return True
    return False


def static_gate_files() -> List[str]:
    """Repo-relative paths (``src/...``) of every test file declaring the ``static`` tier, sorted."""
    root = src_path()
    return sorted("src/" + p.relative_to(root).as_posix() for p in root.rglob("*_test.py") if declares_static(p))


def stray_static_marks() -> List[str]:
    """``path:line`` of every ``pytest.mark.static`` used OUTSIDE a module-level ``pytestmark`` (a decorator,
    a ``pytest.param(marks=...)``) — a test ``-m static`` would select that ``--paths`` would not."""
    root = src_path()
    out: List[str] = []
    for p in sorted(root.rglob("*.py")):
        text = p.read_text()
        if "mark.static" not in text:
            continue
        tree = ast.parse(text)
        module_level = {id(n) for node in tree.body if isinstance(node, ast.Assign)
                        and any(getattr(t, "id", "") == "pytestmark" for t in node.targets)
                        for n in ast.walk(node)}
        for n in ast.walk(tree):
            if (isinstance(n, ast.Attribute) and n.attr == "static" and isinstance(n.value, ast.Attribute)
                    and n.value.attr == "mark" and id(n) not in module_level):
                out.append(f"src/{p.relative_to(root).as_posix()}:{n.lineno}")
    return out


def number_word(n: int) -> str:
    for word, value in NUMBER_WORDS.items():
        if value == n:
            return word
    raise ValueError(f"no number word for {n}: extend utils.static_gates.NUMBER_WORDS")


if __name__ == "__main__":
    import sys

    files = static_gate_files()
    print("\n".join(files))
    if "--paths" not in sys.argv[1:]:
        print(f"{len(files)} static gates ({number_word(len(files))})")
