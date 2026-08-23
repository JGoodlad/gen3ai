"""Who wins `import agents` — and `import poke_env`? The static gate for IMPORT PRECEDENCE.

Sits beside `ruff_gate_test.py`, `file_size_gate_test.py` and `poke_env_fork_gate_test.py` at
the `src/` root because, like those, its subject is the whole tree rather than any one package.

## Why a whole test file about `sys.path` ordering

There are now THREE mechanisms that can put this repo's `src/` on the import path, and they
do not agree about precedence:

    1. `export PYTHONPATH=$PYTHONPATH:src`     -> lands in sys.path BEFORE site-packages
    2. `pip install -e .` (a `.pth` file)      -> lands AFTER site-packages
    3. the root `conftest.py`'s rootdir entry  -> gives `src.agents`, NOT `agents`

Two of those are load-bearing at the same time and must not be collapsed into one:

  * The **editable install** is the contributor surface. `pip install -e .` once, and every
    `python …` in this repo works with no incantation, from any directory.
  * The **launcher child's PYTHONPATH** is worktree ISOLATION. A resumed run is pinned to the
    git commit its checkpoint was saved on, and the pin is nothing but
    `PYTHONPATH=<pinned worktree>/src`. If an editable install ever outranked it, an old
    checkpoint would silently resume against current HEAD — the arch-drift disaster class,
    which fails as `[ModelVersion] FATAL` at best and as quietly-wrong training at worst.

They coexist *only because* (1) beats (2). That is a fact about CPython's startup order, not
about anything in this repo, so nothing here can enforce it — but this file can DETECT it
changing, which is the next best thing and costs milliseconds.

## The two orderings, proved rather than asserted

Both were measured in throwaway venvs on 2026-08-22 and are re-proved here on every run
against real `.pth` files, because an ordering claim that lives only in a comment is folklore
within a month:

| test | claim |
|---|---|
| `test_pythonpath_outranks_a_pth_file` | **Finding B.** PYTHONPATH beats an editable install — worktree isolation survives `pip install -e .` |
| `test_a_pth_file_loses_to_the_site_packages_that_holds_it` | **Finding A.** A `.pth` LOSES to a package installed in the same site-packages — which is exactly how an installed `poke-env` would silently shadow the vendored fork |

Finding A is why `poke-env` is not in `environment.yml` and why
`src/poke_env_fork_gate_test.py` exists. This file proves the *mechanism*; that file guards
the *consequence*. Neither replaces the other: remove the second copy AND keep the gate.

## What the `.pth` replicas here are, honestly

`site.addsitedir()` is the very function CPython's own `site.py` calls for each site-packages
directory: it appends the directory, then processes its `.pth` files and appends the paths
they name. So a subprocess that calls it on a temp directory reproduces the real ordering with
the real machinery — not a model of it. What it does NOT reproduce is a *venv layered over a
system site-packages*, where the venv's `.pth` paths land ahead of the system directory. That
layering is why a `--system-site-packages` venv does not exhibit Finding A even with upstream
`poke-env` installed, and it is a trap for anyone trying to reproduce this: the live conda env
has ONE site-packages, and there the `.pth` loses.

Cost: milliseconds plus three short subprocesses. Unmarked — it runs in the fast inner loop.
"""
import os
import re
import subprocess
import sys
import sysconfig
import textwrap
import tomllib
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

#: The top-level packages `pyproject.toml` must ship. `poke_env` is here because the FORK is
#: part of this project (see the hazard block in pyproject.toml); `rust_sim` is not, because it
#: is a Rust crate whose Python is harness scratch and it has no `__init__.py`.
EXPECTED_TOP_LEVEL = ("agents", "main", "utils", "poke_env")

# `src/main/launcher/child.py` -> the module whose PYTHONPATH export IS the isolation.
_CHILD_PY = _SRC_DIR / "main" / "launcher" / "child.py"


def _pyproject() -> dict:
    assert _PYPROJECT.is_file(), (
        f"pyproject.toml not found at {_PYPROJECT} — the editable install is the documented "
        "way to get `import agents` working; without this file `pip install -e .` fails and "
        "every contributor is back to remembering the PYTHONPATH incantation."
    )
    return tomllib.loads(_PYPROJECT.read_text())


# ────────────────────────────────────────────────────────── 1. the declaration is what we think

def test_pyproject_declares_the_src_layout() -> None:
    """`package-dir = {"" = "src"}` is the whole mechanism. Without it the install would put
    `src` itself on the path and `import agents` would still fail."""
    pkg_dir = _pyproject().get("tool", {}).get("setuptools", {}).get("package-dir", {})
    assert pkg_dir.get("") == "src", (
        f"pyproject.toml must map the import root to src/, got package-dir={pkg_dir!r}"
    )


def test_pyproject_ships_exactly_the_four_top_level_packages() -> None:
    """The include patterns must cover all four and must NOT reach `rust_sim`.

    Checked as patterns rather than a resolved package list on purpose: resolving would need
    setuptools' discovery machinery at test time, which is slower and would pass for the wrong
    reason if discovery changed. What matters is the DECLARATION.
    """
    find = _pyproject()["tool"]["setuptools"]["packages"]["find"]
    assert find.get("where") == ["src"], f"packages.find must search src/, got {find.get('where')!r}"
    include = find.get("include", [])
    for pkg in EXPECTED_TOP_LEVEL:
        assert f"{pkg}*" in include, (
            f"`{pkg}` is missing from packages.find include={include!r}. All four top-level "
            "packages ship — including the vendored poke_env fork, which is ours."
        )
    assert not any(p.startswith("rust_sim") for p in include), (
        f"rust_sim must not be packaged (it is a Rust crate, not a Python package): {include!r}"
    )


def test_pyproject_declares_no_runtime_dependencies() -> None:
    """ONE OWNER PER QUESTION. `environment.yml` owns what is installed; pyproject owns where
    imports look.

    If this file also declared dependencies, `pip install -e .` would be free to RESOLVE them —
    i.e. to replace a pinned wheel inside a working conda env, including the CUDA-local-version
    torch that is not on PyPI at all. An empty list makes `pip install -e .` incapable of
    installing anything: it writes a `.pth` and a `dist-info` and stops.
    """
    deps = _pyproject()["project"].get("dependencies", None)
    assert deps == [], (
        f"pyproject.toml must declare NO runtime dependencies, got {deps!r}. environment.yml "
        "is the single owner — see the header comment in pyproject.toml for why splitting that "
        "ownership is how a working environment gets mutated by an install."
    )


def test_nothing_installed_shadows_our_four_top_level_names() -> None:
    """`agents`, `main` and `utils` are GENERIC names — the fork hazard, generalised.

    `poke_env_fork_gate_test.py` guards the one collision we already know about, because
    `poke-env` is a real package someone might install. But `utils` and `main` are names a
    future dependency could plausibly claim, and the symptom would be identical: a clean
    import of somebody else's module and behaviour nobody can explain. Cheap to check for all
    four at once, so it is checked for all four.

    Asserts against THIS checkout (derived from `__file__`), not an absolute path — so a
    worktree checks itself, and the test is meaningful under PYTHONPATH and under an editable
    install alike.
    """
    import importlib.util

    offenders = []
    for name in EXPECTED_TOP_LEVEL:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            offenders.append(f"{name}: not importable at all")
            continue
        if not str(Path(spec.origin).resolve()).startswith(str(_SRC_DIR)):
            offenders.append(f"{name}: {spec.origin}")
    assert not offenders, (
        "a top-level package name is resolving OUTSIDE this checkout's src/:\n  "
        + "\n  ".join(offenders)
        + f"\n\nexpected everything under: {_SRC_DIR}\n\n"
        "TWO causes, and the message above tells you which:\n"
        "  * resolved into site-packages -> something installed has claimed one of our names. "
        "Uninstall the competitor. Do not rename around it and do not relax this assertion; "
        "the failure mode is silent, the import succeeds and the wrong code runs.\n"
        "  * resolved into a DIFFERENT checkout of this repo -> you are running THIS tree's "
        "tests against ANOTHER tree's code. The usual cause is running a git worktree's suite "
        "with no PYTHONPATH, so the main checkout's editable install answers the import. Fix "
        "with `export PYTHONPATH=$PYTHONPATH:src` — in a worktree that export is not optional."
    )


def test_every_declared_package_actually_exists() -> None:
    """Guard the guard: a rename that misses pyproject.toml would ship a package list naming
    directories that are gone, and the editable install would keep working (the `.pth` is just
    a directory) right up until someone built a wheel."""
    missing = [p for p in EXPECTED_TOP_LEVEL if not (_SRC_DIR / p / "__init__.py").is_file()]
    assert not missing, f"pyproject.toml names packages that do not exist under src/: {missing}"


# ────────────────────────────────────────── 2. the precedence chain, proved with real .pth files

def _probe_import(env_pythonpath: "str | None", sitedir: Path, module: str) -> str:
    """Resolve `module` in a subprocess that has processed `sitedir`'s `.pth` files.

    `site.addsitedir` is what CPython's own startup calls per site-packages directory, so this
    is the real mechanism rather than an imitation of it.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    if env_pythonpath is not None:
        env["PYTHONPATH"] = env_pythonpath
    code = textwrap.dedent(f"""
        import site, importlib.util, sys
        site.addsitedir({str(sitedir)!r})
        spec = importlib.util.find_spec({module!r})
        print(spec.origin if spec else "NOT-FOUND")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, cwd=str(_REPO_ROOT.parent), timeout=120)
    assert out.returncode == 0, f"probe failed:\n{out.stderr}"
    return out.stdout.strip()


def _make_pkg(root: Path, name: str, marker: str) -> Path:
    (root / name).mkdir(parents=True, exist_ok=True)
    (root / name / "__init__.py").write_text(f"MARKER = {marker!r}\n")
    return root / name / "__init__.py"


def test_pythonpath_outranks_a_pth_file(tmp_path: Path) -> None:
    """FINDING B — the property the launcher's worktree isolation rests on.

    An editable install names a directory in a `.pth`; a pinned worktree names a different one
    in PYTHONPATH. PYTHONPATH must win, or a resumed run imports current HEAD instead of the
    code its checkpoint was saved on.
    """
    installed = tmp_path / "installed"          # what the .pth points at (the "main checkout")
    pinned = tmp_path / "pinned"                # what PYTHONPATH points at (the "worktree")
    sitedir = tmp_path / "site-packages"
    sitedir.mkdir()
    _make_pkg(installed, "gen3ai_precedence_probe", "FROM-PTH")
    _make_pkg(pinned, "gen3ai_precedence_probe", "FROM-PYTHONPATH")
    (sitedir / "__editable__.probe.pth").write_text(f"{installed}\n")

    origin = _probe_import(str(pinned), sitedir, "gen3ai_precedence_probe")

    assert origin.startswith(str(pinned)), (
        "PYTHONPATH NO LONGER OUTRANKS AN EDITABLE INSTALL'S .pth.\n"
        f"  resolved to: {origin}\n"
        f"  expected under: {pinned}\n\n"
        "This breaks the launcher's worktree isolation OUTRIGHT: `child.py` pins a resumed "
        "run to its checkpoint's commit using nothing but PYTHONPATH, so if a `.pth` can beat "
        "it, an old run silently resumes on current HEAD. Do not adjust this test — find out "
        "what changed about sys.path construction, and re-derive the pin."
    )


def test_a_pth_file_loses_to_the_site_packages_that_holds_it(tmp_path: Path) -> None:
    """FINDING A — the mechanism behind the `poke_env` shadowing hazard, kept executable.

    A site-packages directory is added to `sys.path` BEFORE the paths its own `.pth` files
    name. So an installed package beats an editable install of the same name — which is
    precisely how upstream `poke-env` would silently win over the vendored fork.

    This test is the evidence for a claim that otherwise only exists as prose in
    `pyproject.toml`, `environment.yml` and `poke_env_fork_gate_test.py`. It PASSES when the
    hazard is real. If it ever fails, the hazard is gone and those three documents are wrong.
    """
    editable_src = tmp_path / "editable-src"
    sitedir = tmp_path / "site-packages"
    sitedir.mkdir()
    _make_pkg(editable_src, "gen3ai_shadow_probe", "FROM-EDITABLE-SRC")
    _make_pkg(sitedir, "gen3ai_shadow_probe", "FROM-SITE-PACKAGES")   # the "installed" copy
    (sitedir / "__editable__.probe.pth").write_text(f"{editable_src}\n")

    origin = _probe_import(None, sitedir, "gen3ai_shadow_probe")

    assert origin.startswith(str(sitedir)), (
        "A .pth path now beats the site-packages directory that holds it. That would be GOOD "
        "news — it is the hazard `poke_env_fork_gate_test.py` defends against — but three "
        f"documents assert the opposite and must be corrected together. Resolved to: {origin}"
    )


def test_a_stale_editable_install_is_not_pointing_at_a_deleted_checkout() -> None:
    """If THIS interpreter carries an editable install of gen3ai, its target must still exist.

    The failure it catches: `pip install -e .` was run from a git WORKTREE, the worktree was
    later deleted, and the `.pth` now names a path that is gone. Python skips missing `.pth`
    entries in silence, so the symptom is `ModuleNotFoundError: agents` on a machine where the
    install "succeeded" — and the natural next move is to re-export PYTHONPATH and never find
    out. `CONTRIBUTING.md` says to install from the MAIN CHECKOUT only; this is that rule with
    teeth.

    No editable install present (PYTHONPATH-only environments, CI, a fresh clone) means there
    is nothing to be stale, so the check has nothing to say — deliberately not a skip, since a
    skipped test reads like a missing one.
    """
    for site_dir in {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]}:
        for pth in Path(site_dir).glob("__editable__*gen3ai*.pth"):
            for line in pth.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith(("import ", "#")):
                    continue  # the finder-strategy form executes code; only static paths here
                assert Path(line).is_dir(), (
                    f"{pth} points at {line}, which does not exist.\n"
                    "An editable install made from a git worktree that has since been deleted. "
                    "Python ignores a missing .pth entry silently, so imports fail for a reason "
                    "the install itself never reports. Re-install from the MAIN checkout:\n"
                    "    pip uninstall gen3ai && cd <main checkout> && pip install -e ."
                )


# ─────────────────────────────────────── 3. the pin: child.py's PYTHONPATH may never be removed

def test_child_py_still_exports_pythonpath_in_the_spawn_env() -> None:
    """THE LITERAL PIN. `child.py` must keep building a PYTHONPATH from its src_dir and
    passing it into the child's env.

    A scan rather than a behaviour check *in addition to* the behaviour check below, because
    the two fail differently and both are worth having: the behavioural test catches a change
    of MEANING, this one catches a change of INTENT — someone "tidying up" an export that looks
    redundant now that an editable install exists. The 🚨 comment is part of the assertion for
    the same reason: an allowlist entry that outlives its own explanation misleads every reader
    after it (the c-family lesson in the root CLAUDE.md).
    """
    text = _CHILD_PY.read_text()
    assert re.search(r'"PYTHONPATH":\s*pythonpath', text), (
        f"{_CHILD_PY} no longer passes PYTHONPATH into the child's environment. That export is "
        "WORKTREE-ISOLATION MACHINERY, not leftover setup: it is the only thing making a "
        "resumed run import the code its checkpoint was saved on. See Finding B in "
        "designs/research_state/ledger.md and test_pythonpath_outranks_a_pth_file above."
    )
    assert "src_dir" in text and re.search(r"pythonpath\s*=.*src_dir", text), (
        f"{_CHILD_PY} no longer builds the child PYTHONPATH from src_dir — the pin must name "
        "the PINNED WORKTREE's src/, not inherit whatever the launcher happened to have."
    )
    assert "WORKTREE-ISOLATION MACHINERY" in text, (
        f"the 🚨 explanation above the PYTHONPATH construction in {_CHILD_PY} was removed. "
        "Keep it: the next person to read that line will otherwise delete it, exactly as this "
        "test's docstring predicts."
    )


def test_the_launcher_child_really_imports_from_the_src_dir_it_was_handed(tmp_path: Path) -> None:
    """END TO END through the production spawn: hand `_launch_child` a fake worktree and prove
    the child imports `agents` FROM IT — over site-packages, and over any editable install.

    This is Finding B exercised through the real code path rather than a replica. It is
    strictly stronger when an editable install is present (then the fake worktree is competing
    with a real `.pth` naming the real repo), and still meaningful without one.
    """
    from main.launcher.child import _build_child_env, _launch_child
    from main.launcher.state import LauncherState

    fake_worktree_src = tmp_path / "pinned-worktree" / "src"
    _make_pkg(fake_worktree_src, "agents", "FROM-PINNED-WORKTREE")

    probe = tmp_path / "probe.py"
    probe.write_text("import agents; print('AGENTS_AT', agents.__file__, flush=True)\n")

    state = LauncherState(interval_hours=0.0)
    state.run_dir = str(tmp_path)
    proc = _launch_child([], _build_child_env(), state, str(probe), str(fake_worktree_src))
    assert proc.wait(timeout=120) == 0

    logged = ""
    for _ in range(400):
        logged = "\n".join(state.snapshot().log_lines)
        if "AGENTS_AT" in logged:
            break
        import time as _t
        _t.sleep(0.05)

    assert "AGENTS_AT" in logged, f"probe child produced no output:\n{logged}"
    resolved = logged.split("AGENTS_AT", 1)[1].split("\n", 1)[0].strip()
    assert resolved.startswith(str(fake_worktree_src)), (
        "THE LAUNCHER'S WORKTREE PIN IS BROKEN — a spawned child imported `agents` from "
        f"{resolved} instead of the src_dir it was handed ({fake_worktree_src}).\n\n"
        "Consequence: a resumed run loads the code at current HEAD rather than the commit its "
        "checkpoint was saved on. The usual cause is that child.py's PYTHONPATH export was "
        "removed or reordered — it must come FIRST, ahead of anything inherited."
    )
