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

Cost: COLD ~24 s (a fresh cache), WARM ~0.3 s. The cache lives in the temp dir, ONE dir per
checkout (mypy's cache records absolute paths, so two checkouts sharing one would thrash each other
cold). Each dir is ~60 MB on a RAM-backed tmpfs, and before 2026-09-30 nothing ever removed one — 20+
accumulated in a day (~1.2 GB). So every run touches its own dir's ``.last_used`` marker and PRUNES
its siblings: any unused for ``CACHE_MAX_AGE_DAYS``, and all but the ``CACHE_MAX_KEYS`` most recently
used. A sibling used within ``CACHE_IN_USE_GRACE_S`` is never pruned — a concurrent run in another
worktree may be reading it. ``GEN3AI_MYPY_STRICT_EQ_CACHE_ROOT`` moves the root (the tests use it).

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
import shutil
import tempfile
import time
from pathlib import Path

import pytest

from utils.paths import src_root

pytestmark = [pytest.mark.static, pytest.mark.skipif(os.environ.get("GEN3AI_SKIP_ENUM_STR_GATE") == "1",
                                reason="GEN3AI_SKIP_ENUM_STR_GATE=1")]

_PACKAGES = ("agents", "main", "utils", "poke_env")

CACHE_PREFIX = "gen3ai_mypy_strict_eq_"
CACHE_MARKER = ".last_used"
CACHE_MAX_AGE_DAYS = 3.0
CACHE_MAX_KEYS = 6
CACHE_IN_USE_GRACE_S = 3600.0


def cache_root() -> Path:
    return Path(os.environ.get("GEN3AI_MYPY_STRICT_EQ_CACHE_ROOT") or tempfile.gettempdir())


def cache_dir_for(src: Path) -> Path:
    """This checkout's mypy cache: one dir per ``src/`` path, under ``cache_root()``."""
    return cache_root() / f"{CACHE_PREFIX}{hashlib.sha1(str(src).encode()).hexdigest()[:12]}"


def _last_used(d: Path) -> float:
    try:
        return (d / CACHE_MARKER).stat().st_mtime
    except OSError:
        return d.lstat().st_mtime       # a pre-marker dir: its own mtime is the best we have


def prune_sibling_caches(own: Path, now: float | None = None) -> list[Path]:
    """Remove the OTHER checkouts' cache dirs that are stale (by age) or surplus (beyond the
    ``CACHE_MAX_KEYS`` most recently used, own included). Never one used within the grace window,
    never a symlink, never another uid's. Returns what was removed."""
    now = time.time() if now is None else now
    sibs = []
    for d in own.parent.glob(CACHE_PREFIX + "*"):
        try:
            if d == own or d.is_symlink() or not d.is_dir() or d.lstat().st_uid != os.getuid():
                continue
            sibs.append((_last_used(d), d))
        except OSError:
            continue
    sibs.sort(reverse=True)             # most recently used first
    removed = []
    for rank, (used, d) in enumerate(sibs, start=1):     # rank 0 is our own dir
        idle = now - used
        if idle < CACHE_IN_USE_GRACE_S:
            continue
        if idle > CACHE_MAX_AGE_DAYS * 86400 or rank >= CACHE_MAX_KEYS:
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d)
    return removed


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
    cache = cache_dir_for(src_root())
    cache.mkdir(parents=True, exist_ok=True)
    (cache / CACHE_MARKER).touch()
    prune_sibling_caches(cache)
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


def test_the_cache_is_bounded_and_prunes_its_siblings(tmp_path, monkeypatch):
    """Each run keeps ITS dir and bounds the rest: stale by age, surplus by count, in-use spared."""
    monkeypatch.setenv("GEN3AI_MYPY_STRICT_EQ_CACHE_ROOT", str(tmp_path))
    now = time.time()

    def sibling(name: str, idle_h: float, marker: bool = True) -> Path:
        d = tmp_path / f"{CACHE_PREFIX}{name}"
        (d / "3.11").mkdir(parents=True)
        t = now - idle_h * 3600
        if marker:
            (d / CACHE_MARKER).touch()
            os.utime(d / CACHE_MARKER, (t, t))
        os.utime(d, (t, t))
        return d

    stale = sibling("stale0000000", 24 * 10)
    legacy = sibling("legacy000000", 24 * 10, marker=False)      # a pre-marker dir
    busy = sibling("busy00000000", 0.1)                          # another worktree, right now
    recent = [sibling(f"recent{i:06d}", 2 + i) for i in range(8)]
    outside = tmp_path / "unrelated_dir"
    outside.mkdir()

    seen: dict = {}

    def fake_run(cmd, **kw):
        seen["cache"] = Path(cmd[cmd.index("--cache-dir") + 1])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_mypy(["x.py"], cwd=tmp_path)

    own = cache_dir_for(src_root())
    assert seen["cache"] == own and own.parent == tmp_path
    assert (own / CACHE_MARKER).exists()
    assert not stale.exists() and not legacy.exists()
    assert busy.exists() and outside.exists()
    left = sorted(tmp_path.glob(CACHE_PREFIX + "*"))
    assert len(left) <= CACHE_MAX_KEYS + 1, left     # the bound, + the in-use grace survivor
    assert recent[0].exists() and not recent[-1].exists()   # most recently used survive
