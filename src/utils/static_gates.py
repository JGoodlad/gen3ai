"""The ONE inventory of the static gates: every test file that declares ``pytest.mark.static``.

The prose that counts them ("Seventeen static gates" in the root ``CLAUDE.md``, "Seventeen static gates run
inside the suite" in ``CONTRIBUTING.md`` and ``docs/RUNNING.md``) used to be hand-counted and disagreed three
ways (2026-10-08). The marker is the declaration (a tier is DECLARED, never inferred), so this module reads
the declaration with an AST scan and ``static_gates_test.py`` holds every count and table to it.

    python -m utils.static_gates          # one path per line, then the count
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


def number_word(n: int) -> str:
    for word, value in NUMBER_WORDS.items():
        if value == n:
            return word
    raise ValueError(f"no number word for {n}: extend utils.static_gates.NUMBER_WORDS")


if __name__ == "__main__":
    files = static_gate_files()
    print("\n".join(files))
    print(f"{len(files)} static gates ({number_word(len(files))})")
