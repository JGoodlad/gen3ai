"""THE durable guard of the launcher's interpreter rule: no launcher module names a machine-specific path.

``child.py`` once hardcoded one box's conda interpreter, so a fresh clone died on its first launcher run
(``interpreter_test.py`` pins the resolution rule, ``$GEN3AI_PYTHON`` then ``sys.executable``). This line scan of
every module in ``src/main/launcher/`` is what keeps a fresh clone launchable. It reads source and runs none of the
product, so it declares the ``static`` tier (it lived in ``interpreter_test.py`` until 2026-11-17).
"""
import os
import re

import pytest

pytestmark = pytest.mark.static

_LAUNCHER_DIR = os.path.dirname(os.path.abspath(__file__))
BAD = re.compile(r"/home/\w+|/(?:opt|usr/local)/miniconda|/miniconda\d*/envs/")

#: Files whose text names the pattern on purpose, each with its reason.
_EXEMPT = {
    os.path.basename(__file__): "this file: the pattern above is what the scan searches WITH",
    "interpreter_test.py": "its docstring quotes the deleted hardcoded interpreter as the history of the rule",
}


def _offenders(text: str, name: str):
    return [f"{name}:{i}: {line.strip()}" for i, line in enumerate(text.splitlines(), 1) if BAD.search(line)]


def test_no_machine_specific_interpreter_literal_in_the_launcher_package():
    """No launcher module may name an absolute interpreter/home path."""
    offenders = []
    for name in sorted(os.listdir(_LAUNCHER_DIR)):
        if not name.endswith(".py") or name in _EXEMPT:
            continue
        with open(os.path.join(_LAUNCHER_DIR, name), encoding="utf-8") as f:
            offenders += _offenders(f.read(), name)
    assert not offenders, (
        "launcher modules must not hardcode a machine-specific path — use "
        "main.launcher.child.resolve_child_python() / $GEN3AI_PYTHON:\n  " + "\n  ".join(offenders))


def test_the_exemptions_are_still_load_bearing():
    for name in _EXEMPT:
        with open(os.path.join(_LAUNCHER_DIR, name), encoding="utf-8") as f:
            assert BAD.search(f.read()), f"{name} no longer holds the pattern — drop its exemption"


def test_the_scan_can_actually_fail():
    """A gate that cannot fail is not a gate (the vacuity family)."""
    assert _offenders('PY = "/opt/miniconda/bin/python"\n', "x.py")
    assert _offenders("p = '/x/miniconda3/envs/gen3ai/bin/python3'\n", "x.py")
    assert not _offenders("PY = sys.executable\n", "x.py")
