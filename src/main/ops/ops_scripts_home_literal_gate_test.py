"""No `scripts/ops/*.sh` holds a `/home/…` literal outside a comment.

The same rule `src/utils/home_path_literal_gate_test.py` enforces for modules, for the same reason: a literal is
correct on exactly one box and silently wrong everywhere else. A shell script is not a Python module, so that AST
scan cannot see it. The ONE exemption is `_common.sh`'s interpreter fall-back, which mirrors `scripts/land.sh`. It
reads the scripts and runs none of them, so it declares the ``static`` tier (it lived in `ops_scripts_test.py`, whose
`bash -n` and `--help` checks run the scripts, until 2026-11-17).
"""
from __future__ import annotations

import re

import pytest

from utils.paths import repo_path

pytestmark = pytest.mark.static

_OPS_DIR = repo_path("scripts", "ops")
_ALL_SCRIPTS = sorted(_OPS_DIR.glob("*.sh"))

#: `_common.sh` carries the `gen3ai_torch28` interpreter fall-back, exactly as `scripts/land.sh`
#: does — the one place a box-specific path is the right answer, because it is a FALL-BACK that
#: is tested for executability before use and superseded by `$GEN3AI_PYTHON`.
_HOME_LITERAL_EXEMPT = {"_common.sh"}


def _offenders(text: str):
    return [(i, line.strip()) for i, line in enumerate(text.splitlines(), 1)
            if "/home/" in line and not re.match(r"\s*#", line)]


def test_there_are_scripts_to_check() -> None:
    assert len(_ALL_SCRIPTS) >= 4, sorted(p.name for p in _OPS_DIR.glob("*"))


@pytest.mark.parametrize("path", [p for p in _ALL_SCRIPTS if p.name not in _HOME_LITERAL_EXEMPT],
                         ids=lambda p: p.name)
def test_no_absolute_home_literal(path) -> None:
    offenders = _offenders(path.read_text())
    assert not offenders, (
        f"{path.name}: absolute home path(s) outside a comment at {offenders} — resolve the "
        f"checkout with ops_repo_root / ops_models_dir from _common.sh instead"
    )


def test_the_exemption_is_still_load_bearing() -> None:
    """An exemption that no longer holds a literal is an exemption that misleads the next reader."""
    for name in _HOME_LITERAL_EXEMPT:
        assert _offenders((_OPS_DIR / name).read_text()), f"{name} no longer holds a /home literal — drop it"


def test_the_scan_can_actually_fail() -> None:
    """A gate that cannot fail is not a gate (the vacuity family)."""
    assert _offenders('ROOT=/home/someone/dev/gen3ai\n') == [(1, "ROOT=/home/someone/dev/gen3ai")]
    assert _offenders("  # lives in /home/someone\n") == []
