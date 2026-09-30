"""No code compares a poke-env ENUM to a STRING — the comparison is always False (or always True
for ``!=``), so the branch behind it is dead in silence.

**Why this gate exists (F-LF-1, 2026-09-29).** Four scripted bots — ``SimpleHeuristicsPlayer``,
``Gen3HeuristicV2Player``, ``Gen3SetupSweepPlayer`` and ``Gen3SetupSweepV2Player`` — gated their
SETUP step on ``move.target == "self"``. poke-env's ``Move.target`` is a ``Target`` ENUM, so the
test was never true: ``setup_sweep`` never set up, and every eval row and training bot mix that
carried those names measured an attacker with switch logic. Nothing raised; nothing could.

**How it checks — by TYPE, not by name.** A name-based AST scan cannot tell ``move.status``
(a ``Status`` enum) from ``live_mon.status`` (a LiveView ``str``) or ``device.type`` (torch) — the
first draft flagged 55 sites of which 5 were real. mypy's ``--strict-equality`` knows the types: it
reports ``[comparison-overlap]`` for an ``==`` / ``!=`` / ``in`` whose operands cannot overlap. It
caught all four bot sites on the pre-fix code. This gate runs it over ``agents``, ``main``,
``utils`` and ``poke_env`` (``--check-untyped-defs``, ``--follow-imports=silent``, independent of
``mypy.ini``'s strict scope) and FAILS on every overlap finding whose operand types name a poke-env
enum (the enum set is DERIVED from the vendored package, not typed here).

**The honest limit:** mypy can only see a comparison whose receiver it can type. A value that flows
through ``Any`` (an unannotated ``getattr``, an untyped container) is invisible to it. Other
non-enum overlap findings (a wrong annotation, a pytest ``approx``) are PRINTED, not failed — they
are not this class.

Cost: COLD ~24 s (a fresh cache), WARM ~0.3 s; the cache lives in the temp dir keyed by checkout.

    GEN3AI_SKIP_ENUM_STR_GATE=1 pytest src/ -q
"""
from __future__ import annotations

import enum
import hashlib
import importlib
import os
import pkgutil
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from utils.paths import src_root

pytestmark = pytest.mark.skipif(os.environ.get("GEN3AI_SKIP_ENUM_STR_GATE") == "1",
                                reason="GEN3AI_SKIP_ENUM_STR_GATE=1")

_PACKAGES = ("agents", "main", "utils", "poke_env")


def poke_env_enum_names() -> set[str]:
    """Every ``Enum`` class defined in ``poke_env.battle`` (Target, Status, MoveCategory, Weather,
    Field, SideCondition, PokemonType, Effect, PokemonGender, …) — derived, so a new one is covered."""
    import poke_env.battle as pkg

    names = set()
    for info in pkgutil.iter_modules(pkg.__path__):
        mod = importlib.import_module(f"poke_env.battle.{info.name}")
        for obj in vars(mod).values():
            if isinstance(obj, type) and issubclass(obj, enum.Enum) and obj.__module__ == mod.__name__:
                names.add(obj.__name__)
    return names


def run_mypy(targets: list[str], cwd: Path) -> list[str]:
    """``[comparison-overlap]`` lines from mypy ``--strict-equality`` over ``targets`` (argv form)."""
    key = hashlib.sha1(str(src_root()).encode()).hexdigest()[:12]
    cache = Path(tempfile.gettempdir()) / f"gen3ai_mypy_strict_eq_{key}"
    cmd = [sys.executable, "-m", "mypy", "--config-file", os.devnull, "--python-version", "3.11",
           "--ignore-missing-imports", "--follow-imports=silent", "--check-untyped-defs",
           "--strict-equality", "--no-pretty", "--no-color-output", "--hide-error-context",
           "--no-error-summary", "--cache-dir", str(cache), *targets]
    env = dict(os.environ, MYPYPATH=str(src_root()))
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    assert "No module named mypy" not in out, "mypy is not installed — this gate did not run"
    assert "errors prevented further checking" not in out, f"mypy could not check the tree:\n{out[-3000:]}"
    return [ln for ln in out.splitlines() if "[comparison-overlap]" in ln]


def enum_findings(lines: list[str], enums: set[str]) -> list[str]:
    """The overlap findings whose operand types name a poke-env enum."""
    pat = re.compile(r"\b(" + "|".join(sorted(enums)) + r")\b")
    return [ln for ln in lines if any(pat.search(t) for t in re.findall(r'type: "([^"]*)"', ln))]


def test_enum_set_is_derived_and_covers_the_bot_enums():
    names = poke_env_enum_names()
    assert {"Target", "Status", "MoveCategory", "Weather", "Field", "SideCondition",
            "PokemonType", "Effect"} <= names, names


def test_no_poke_env_enum_is_compared_to_a_string():
    lines = run_mypy([f"-p={p}" for p in _PACKAGES], cwd=src_root())
    bad = enum_findings(lines, poke_env_enum_names())
    other = [ln for ln in lines if ln not in bad]
    if other:
        print("non-enum [comparison-overlap] findings (reported, not this gate's class):\n  "
              + "\n  ".join(other))
    assert not bad, ("a poke-env ENUM is compared to a value it can never equal — the branch is "
                     "dead (F-LF-1: `move.target == \"self\"`). Compare to the enum member "
                     "(`move.target is Target.SELF`):\n  " + "\n  ".join(bad))


def test_the_gate_has_teeth(tmp_path):
    """The pre-fix bot shape, and its ``in`` / ``!=`` / Status / Weather cousins, are each caught."""
    probe = tmp_path / "enum_probe.py"
    probe.write_text(
        "from poke_env.battle.battle import Battle\n"
        "from poke_env.battle.move import Move\n"
        "from poke_env.battle.pokemon import Pokemon\n"
        "def f(battle: Battle, mon: Pokemon) -> int:\n"
        "    n = 0\n"
        "    for move in battle.available_moves:\n"
        "        if move.target == 'self':\n"            # 7: F-LF-1's exact shape
        "            n += 1\n"
        "        if move.category in ('physical', 'special'):\n"   # 9
        "            n += 1\n"
        "    if mon.status != 'slp':\n"                  # 11
        "        n += 1\n"
        "    if 'raindance' in battle.weather:\n"        # 13
        "        n += 1\n"
        "    return n\n")
    lines = run_mypy([str(probe)], cwd=src_root())
    hit = {int(ln.split(":")[1]) for ln in enum_findings(lines, poke_env_enum_names())}
    assert {7, 9, 11, 13} <= hit, lines
