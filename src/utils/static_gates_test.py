"""The static-gate count in the prose, and every gate's row in the tables, equal the declared inventory.

The inventory is ``utils.static_gates.static_gate_files()`` (every ``*_test.py`` declaring
``pytest.mark.static``). Before this test the counts disagreed three ways ("Eleven" in ``docs/RUNNING.md`` and
``CONTRIBUTING.md``, "eight" in ``README.md``, "Seventeen" in the root ``CLAUDE.md``) and
``tier_budget_guard_test._STATIC_GATES`` omitted two gates.
"""
from __future__ import annotations

import re

from utils.paths import repo_path
from utils.static_gates import NUMBER_WORDS, declares_static, number_word, static_gate_files

# Deliberately NOT `pytest.mark.static`: it reads six small text files, and a static-tier mark would make it
# a member of the inventory it counts.


def _text(*parts: str) -> str:
    return repo_path(*parts).read_text()


def test_the_inventory_is_not_trivially_empty():
    assert len(static_gate_files()) >= 17


def test_the_scanner_has_teeth(tmp_path):
    """A file whose ``pytestmark`` holds the marker is a gate; one that merely MENTIONS it is not."""
    yes = tmp_path / "a_gate_test.py"
    yes.write_text("import pytest\npytestmark = pytest.mark.static\n")
    also = tmp_path / "b_gate_test.py"
    also.write_text("import pytest\npytestmark = [pytest.mark.slow, pytest.mark.static]\n")
    no = tmp_path / "c_test.py"
    no.write_text("import pytest\n# pytest.mark.static is a tier\nX = 'pytest.mark.static'\n")
    assert declares_static(yes) and declares_static(also) and not declares_static(no)


def test_tier_budget_guard_declares_exactly_the_inventory():
    """``tier_budget_guard_test._STATIC_GATES`` (declared by hand) names every static gate — and only them."""
    from tier_budget_guard_test import _STATIC_GATES

    assert sorted(_STATIC_GATES) == static_gate_files()


def _counted(doc: str, pattern: str) -> int:
    m = re.search(pattern, doc)
    assert m, f"the count sentence {pattern!r} is gone"
    return NUMBER_WORDS[m.group(1)]


def test_every_prose_count_equals_the_inventory():
    n = len(static_gate_files())
    word = number_word(n)
    assert _counted(_text("CLAUDE.md"), r"\*\*(\w[\w-]*) static gates, all `static`-tier") == n, word
    assert _counted(_text("CONTRIBUTING.md"), r"\*\*(\w[\w-]*) static gates run inside the suite") == n, word
    assert _counted(_text("docs", "RUNNING.md"), r"(\w[\w-]*) static gates run inside the suite") == n, word


def test_every_gate_has_a_row_in_both_tables():
    """The root ``CLAUDE.md`` and ``CONTRIBUTING.md`` tables each open a row with the gate's path."""
    for doc in ("CLAUDE.md", "CONTRIBUTING.md"):
        text = _text(doc)
        rows = set(re.findall(r"^\| `(src/[\w/]+_test\.py)` \|", text, flags=re.M))
        missing = [g for g in static_gate_files() if g not in rows]
        assert not missing, f"{doc}'s static-gate table has no row for {missing}"


def test_readme_names_no_stale_count():
    """``README.md`` states no static-gate number at all (a hand count is what drifted)."""
    words = "|".join(w.lower() for w in NUMBER_WORDS) + "|one|two|three|four|five|six|seven|eight|nine|\\d+"
    assert not re.search(rf"\b(?:{words})\s+(?:always-on\s+)?static gates", _text("README.md"), flags=re.I)
