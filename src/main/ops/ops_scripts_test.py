"""The `scripts/ops/` shell layer — the properties an OS-level watcher has to have.

These four scripts are SOP §2's layers 1 and 2: they run under `nohup`, outlive the session that
started them, and are the record of an unattended period. Nothing in the Python suite exercises
them, so what can be checked statically is checked here, cheaply:

1. **They PARSE.** `bash -n` on every one. A shell script with a syntax error fails at the first
   line of the run it was supposed to be watching — hours after it was armed, with nothing
   watching in the meantime.
2. **They answer `--help`.** An operator reaching for a watcher at 03:00 gets its arguments from
   the script, not from a document. Exit 0, and the usage names the script.
3. **Bare invocation REFUSES.** No argument must not mean "watch the default run": the whole
   reason these were promoted is that their session copies had one run's name compiled in.
4. **No `/home/…` literal.** The same rule `src/utils/paths_test.py` enforces for modules, for
   the same reason — a literal is correct on exactly one box and silently wrong everywhere else.
   The ONE exemption is `_common.sh`'s interpreter fall-back, which mirrors `scripts/land.sh`.

Unmarked (runs in every tier) and ~free: four `bash -n` calls and four `--help` runs.
"""
from __future__ import annotations

import re
import subprocess

import pytest

from utils.paths import repo_path

_OPS_DIR = repo_path("scripts", "ops")

#: The executable entry points. `_common.sh` is SOURCED, so it has no `--help` of its own.
_ENTRY_POINTS = sorted(p for p in _OPS_DIR.glob("*.sh") if not p.name.startswith("_"))
_ALL_SCRIPTS = sorted(_OPS_DIR.glob("*.sh"))

#: `_common.sh` carries the `gen3ai_stable` interpreter fall-back, exactly as `scripts/land.sh`
#: does — the one place a box-specific path is the right answer, because it is a FALL-BACK that
#: is tested for executability before use and superseded by `$GEN3AI_PYTHON`.
_HOME_LITERAL_EXEMPT = {"_common.sh"}


def test_there_are_scripts_to_check() -> None:
    assert len(_ENTRY_POINTS) >= 4, sorted(p.name for p in _OPS_DIR.glob("*"))


@pytest.mark.parametrize("path", _ALL_SCRIPTS, ids=lambda p: p.name)
def test_the_script_parses(path) -> None:
    """`bash -n`. A watcher that cannot parse fails hours after it was armed."""
    r = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert r.returncode == 0, f"{path.name} does not parse:\n{r.stderr}"


@pytest.mark.parametrize("path", _ENTRY_POINTS, ids=lambda p: p.name)
def test_the_script_is_executable(path) -> None:
    assert path.stat().st_mode & 0o111, f"{path.name} is not executable"


@pytest.mark.parametrize("path", _ENTRY_POINTS, ids=lambda p: p.name)
def test_help_exits_zero_and_names_the_script(path) -> None:
    r = subprocess.run([str(path), "--help"], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, f"{path.name} --help exited {r.returncode}:\n{r.stderr}"
    assert path.name in r.stdout, f"{path.name} --help does not name itself:\n{r.stdout}"
    assert "USAGE" in r.stdout, f"{path.name} --help prints no USAGE section"


@pytest.mark.parametrize("path", _ENTRY_POINTS, ids=lambda p: p.name)
def test_no_argument_refuses_rather_than_defaulting_to_a_run(path) -> None:
    """THE REASON THESE WERE PROMOTED. The session copies had one arm's name compiled in; a
    bare invocation must never silently pick a run."""
    r = subprocess.run([str(path)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 2, (
        f"{path.name} with no argument exited {r.returncode}, not 2 — a watcher that "
        f"defaults to a run is the defect this promotion removed"
    )


@pytest.mark.parametrize("path", _ALL_SCRIPTS, ids=lambda p: p.name)
def test_no_absolute_home_literal(path) -> None:
    """Same class as `paths_test.py`'s module scan: a `/home/…` value is correct on one box."""
    if path.name in _HOME_LITERAL_EXEMPT:
        pytest.skip(f"{path.name}: documented interpreter fall-back, as in scripts/land.sh")
    offenders = [
        (i, line.strip())
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if "/home/" in line and not re.match(r"\s*#", line)
    ]
    assert not offenders, (
        f"{path.name}: absolute home path(s) outside a comment at {offenders} — resolve the "
        f"checkout with ops_repo_root / ops_models_dir from _common.sh instead"
    )


def test_the_exemption_is_still_load_bearing() -> None:
    """An exemption that no longer holds a literal is an exemption that misleads the next reader."""
    for name in _HOME_LITERAL_EXEMPT:
        text = (_OPS_DIR / name).read_text()
        assert "/home/" in text, f"{name} no longer holds a /home literal — drop the exemption"


@pytest.mark.parametrize("path", _ENTRY_POINTS, ids=lambda p: p.name)
def test_the_script_sources_the_shared_path_resolution(path) -> None:
    """Every entry point resolves its run through `_common.sh`, so there is ONE answer to
    "where is models/" and it matches `utils.paths.main_models_dir()`."""
    text = path.read_text()
    assert "_common.sh" in text, f"{path.name} does not source _common.sh"
    assert "ops_resolve_run" in text, f"{path.name} does not resolve its run through _common.sh"
