"""THE CLASS, closed: no module under ``src/agents``, ``src/main``, ``src/utils`` may hardcode a ``/home/…`` path as a VALUE.

Four tests read ``models/`` through a ``/home/goodlad/...`` literal and skipped forever on every other
machine; this is the scan that closed the class (it lived in ``utils/paths_test.py`` until 2026-11-16). It is an
AST scan, not a grep: comments are absent from the AST and docstrings are skipped, so the gate is about code
that *uses* an absolute path, never prose that mentions one. It reads source and runs none of the product, so
it declares the ``static`` tier: it runs in the static tier and in every ship set, and a targeted gate (the
static tier plus the tests covering the change) can no longer miss it. That is the point: commit ``6f92bf999``
typed the main checkout's ``models/`` path in a test, and the full gate caught it only because the full gate ran.
"""
import ast
import pathlib

import pytest

from utils.paths import repo_root, src_root

pytestmark = pytest.mark.static


#: Scope mirrors the ruff and file-size gates. `src/rust_sim` is a Rust crate whose Python is
#: harness scratch — not ours to shape.
_SCANNED_ROOTS = ("agents", "main", "utils")

#: The files allowed to hold an absolute-home literal, because their job IS scanning for them.
#: An exemption without a stated reason outlives its own fix and then misleads every reader.
_SCAN_EXEMPT = {
    "src/main/launcher/interpreter_literal_gate_test.py": "the launcher's own path-literal scanner — the "
                                                          "literal here is the regex it searches WITH",
    "src/utils/home_path_literal_gate_test.py": "this file — same reason",
    "src/main/ops/ops_scripts_home_literal_gate_test.py": "the same scan extended to scripts/ops/*.sh — a shell "
                                        "script is not a Python module, so this gate cannot see "
                                        "it; the literal there is what that scan searches WITH",
}


def _string_values(path: pathlib.Path):
    """Every string CONSTANT in `path` that is not a docstring, with its line.

    Comments never reach the AST, so prose mentioning a path is structurally out of scope; only
    a string the code actually *uses* can fail this gate.
    """
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            docstrings.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings:
            yield node.lineno, node.value


# LINCHPIN (ship gate, 2026-10-19; moved here 2026-11-16): kept in the routine gate at any cost — designs/ops/testing.md "The SHIP GATE"
@pytest.mark.linchpin
def test_no_module_hardcodes_an_absolute_home_path():
    """THE CLASS. Four tests read `models/` through a `/home/goodlad/...` literal and therefore
    skipped forever on every other machine — invisible coverage loss, because a skip that is
    supposed to happen looks exactly like a skip that is not. Fixing the four lines closes four
    instances; this closes the class.
    """
    offenders = []
    for pkg in _SCANNED_ROOTS:
        for path in sorted((src_root() / pkg).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(repo_root()).as_posix()
            if rel in _SCAN_EXEMPT:
                continue
            try:
                values = list(_string_values(path))
            except SyntaxError:
                continue  # syntax_test.py owns that failure
            for lineno, value in values:
                if "/home/" in value:
                    offenders.append(f"{rel}:{lineno}: {value[:100]!r}")
    assert not offenders, (
        "absolute home path used as a value — it is correct on exactly one machine, and where it "
        "guards a skip it makes the test vanish silently everywhere else. Resolve it with "
        "`utils.paths` (repo_path/src_path/main_models_dir):\n  " + "\n  ".join(offenders))


def test_the_scan_exemptions_are_all_still_load_bearing():
    """An exemption that stopped being needed must LEAVE the list — the c-family house rule."""
    stale = []
    for rel in _SCAN_EXEMPT:
        path = repo_root() / rel
        assert path.is_file(), f"exempt file {rel} no longer exists — drop the entry"
        if not any("/home/" in v for _, v in _string_values(path)):
            stale.append(rel)
    assert not stale, f"exemptions no longer needed, delete them: {stale}"


def test_the_scan_can_actually_fail(tmp_path):
    """A gate that cannot fail is not a gate (the vacuity family)."""
    bad = tmp_path / "bad.py"
    bad.write_text('X = "/home/someone/models"\n', encoding="utf-8")
    assert any("/home/" in v for _, v in _string_values(bad))
    good = tmp_path / "good.py"
    good.write_text('# /home/someone/models is where it lives\n"""/home/x doc."""\nX = 1\n',
                    encoding="utf-8")
    assert not any("/home/" in v for _, v in _string_values(good))
