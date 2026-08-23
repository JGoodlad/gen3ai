"""Does `import poke_env` still reach OUR FORK — or has an installed copy shadowed it?

Sits beside `ruff_gate_test.py` and `strict_api_lock_test.py` at the `src/` root because, like
those, its subject is the whole tree rather than any one package.

## The hazard, stated plainly

This repo VENDORS a fork of poke-env at `src/poke_env/` — 57 modules where upstream 0.15.0 has
45 — and the battle layer depends on what the fork adds. `poke-env` is also a perfectly ordinary
PyPI package. So on any machine where both exist there are **two importable packages named
`poke_env`**, and which one wins is decided by nothing more principled than `sys.path` ORDER.

**The failure mode is silent.** If upstream wins there is no error, no warning, no version
mismatch — the code imports, runs, and produces subtly wrong battle behaviour forever. That is
the whole reason this file exists as a *test* rather than a note in a README: a hazard whose
symptom is "everything looks fine" cannot be defended by remembering.

## Why the ordering is not something to rely on

Today the fork wins because `PYTHONPATH=…:src` is exported for every command, and PYTHONPATH
entries land in `sys.path` BEFORE site-packages. That is a real mechanism, but it is a fragile
one to bet a subsystem on, and it is about to get more fragile:

- an editable install (`pip install -e .`) writes a `.pth` file, and `.pth` paths are appended
  **after** site-packages — so under an editable install with an installed `poke-env` present,
  **upstream silently wins**. Proved by experiment in a throwaway replica, 2026-08-22.
- any shell, IDE, CI job, or subprocess that forgets the export flips the ordering too.

The structural fix was to remove `poke-env` from `environment.yml` (see the DELIBERATELY ABSENT
block there): with no second copy installed, there is no ordering question to lose. This test is
what makes that removal permanent — it fails the moment someone re-adds the pin AND the ordering
goes the wrong way, which is precisely the combination that is otherwise invisible.

## What each test pins

1. `test_the_vendored_fork_wins_the_import` — the live assertion. `poke_env.__file__` must
   resolve inside THIS repo's `src/`. Works under a worktree (the path is derived from this
   file, so it is whatever checkout is running).
2. `test_the_gate_catches_a_shadowing_copy` — the revert-verification. Builds a decoy
   `poke_env` in a temp dir, makes it win in a subprocess, and asserts the diagnosis rejects
   the resulting path and NAMES the hazard. Without this, a gate that accidentally always
   passed would be indistinguishable from a gate that works.
3. `test_environment_yml_does_not_reinstall_poke_env` — the upstream half. Catches the re-add
   at the source, before any ordering question can arise.

Cost: milliseconds. Unmarked — it runs in the fast inner loop, which is where you want it.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

# `src/poke_env_fork_gate_test.py` -> `src/` -> repo root. Derived from __file__ so a worktree
# checks ITSELF rather than the main checkout (an absolute path here would make the test either
# wrong in a worktree or silently skipped on another machine).
_SRC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent
_FORK_DIR = _SRC_DIR / "poke_env"


def diagnose(poke_env_file: str | os.PathLike[str], src_dir: Path = _SRC_DIR) -> str | None:
    """Return None when `poke_env_file` is our fork, else a LOUD explanation of what shadowed it.

    Pure and path-only so both branches are testable — the failing branch is exercised by
    `test_the_gate_catches_a_shadowing_copy` against a real decoy, not merely asserted.
    """
    resolved = Path(poke_env_file).resolve()
    fork = (src_dir / "poke_env").resolve()
    try:
        resolved.relative_to(fork)
    except ValueError:
        pass
    else:
        return None

    where = "an installed site-packages copy" if "site-packages" in resolved.parts else "some other copy"
    # Built separately, not interpolated into the dedent block: a multi-line interpolation
    # carries its OWN indentation into textwrap.dedent's common-prefix calculation and
    # silently under-dedents the entire message.
    path_block = "\n".join(f"    {p}" for p in sys.path)
    return textwrap.dedent(
        f"""
        ============================================================================
        THE VENDORED poke_env FORK HAS BEEN SHADOWED.  This is a silent-wrong-results
        bug, not a packaging nit — read before "fixing" the test.
        ============================================================================

            import poke_env  ->  {resolved}
            expected under   ->  {fork}

        `import poke_env` is resolving to {where} instead of this repo's fork. The
        fork is AUTHORITATIVE: it carries modules upstream does not, and the battle
        layer depends on them. Upstream will import cleanly and behave subtly
        differently — there is no error to notice.

        Two things put a second copy on the path, and both have happened here:

          1. `poke-env` reinstalled as a dependency. It was REMOVED from
             environment.yml on purpose (see the DELIBERATELY ABSENT block there);
             nothing in the environment declares it. If it is back, uninstall it:
                 pip uninstall poke-env
          2. sys.path ORDER changed. `.pth` files from an editable install are
             appended AFTER site-packages, so an editable install plus an installed
             `poke-env` makes upstream win. PYTHONPATH entries come first, which is
             why `export PYTHONPATH=$PYTHONPATH:src` masks the problem.

        Do not "fix" this by deleting the assertion. Remove the second copy.

        sys.path as this process saw it, in order (the winner is whichever comes first):
        """
    ).strip() + "\n" + path_block + "\n" + "=" * 76


def test_the_vendored_fork_wins_the_import() -> None:
    """`import poke_env` must reach `src/poke_env/`, not an installed PyPI copy."""
    # Imported inside the test, not at module scope, on purpose: the import IS the subject, and
    # a module-scope import would make a collection error out of what should be a readable
    # assertion failure.
    import poke_env

    assert poke_env.__file__ is not None, "poke_env imported as a namespace package (no __file__)"
    problem = diagnose(poke_env.__file__)
    assert problem is None, problem


def test_the_fork_is_actually_present() -> None:
    """Guard the guard: if `src/poke_env/` vanished, test 1 would fail for the wrong reason."""
    assert (_FORK_DIR / "__init__.py").is_file(), (
        f"the vendored fork is missing at {_FORK_DIR} — the submodule/checkout is incomplete"
    )


def test_the_gate_catches_a_shadowing_copy(tmp_path: Path) -> None:
    """REVERT-VERIFICATION: build a decoy that wins the import, prove the gate rejects it.

    A gate that cannot fail is not a gate. This constructs the exact artifact a shadowing
    install produces — a `poke_env/__init__.py` outside `src/` that `import poke_env` resolves
    to — and asserts `diagnose()` both rejects it and explains it.
    """
    decoy_root = tmp_path / "fake-site-packages"
    (decoy_root / "poke_env").mkdir(parents=True)
    (decoy_root / "poke_env" / "__init__.py").write_text("SHADOW = True\n")

    # Resolve the import in a SUBPROCESS with the decoy first: this proves a shadow really can
    # win (it is not a hypothesis) and yields the genuine `__file__` the winner reports.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(decoy_root), str(_SRC_DIR)])
    proc = subprocess.run(
        [sys.executable, "-c", "import poke_env; print(poke_env.__file__)"],
        capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=120,
    )
    assert proc.returncode == 0, f"decoy import failed:\n{proc.stderr}"
    winner = proc.stdout.strip()
    assert winner.startswith(str(decoy_root)), (
        f"the decoy did not win the import ({winner}) — this test's premise broke, not the gate"
    )

    problem = diagnose(winner)
    assert problem is not None, "diagnose() ACCEPTED a shadowing copy — the live gate is inert"
    assert "SHADOWED" in problem and "pip uninstall poke-env" in problem, (
        f"the failure message does not name the hazard or the fix:\n{problem}"
    )


def test_environment_yml_does_not_reinstall_poke_env() -> None:
    """The upstream half: catch the re-added pin at the source, before ordering can matter.

    Text-scanned rather than YAML-parsed on purpose — `yaml` is not a dependency of this
    project, and adding one so a guard can read a file would be a poor trade.
    """
    env_yml = _REPO_ROOT / "environment.yml"
    assert env_yml.is_file(), f"environment.yml not found at {env_yml}"

    offenders = [
        line.strip()
        for line in env_yml.read_text().splitlines()
        # A requirement LINE, never a comment: the file explains the removal in prose that
        # necessarily mentions the name, and a naive substring search would flag its own reason.
        if line.strip().startswith("- ") and line.strip()[2:].lstrip().startswith(("poke-env", "poke_env"))
    ]
    assert not offenders, (
        "environment.yml re-adds the PyPI poke-env package: "
        f"{offenders}\n"
        "This repo VENDORS the fork at src/poke_env/ and it is authoritative. A second "
        "installed copy makes `import poke_env` depend on sys.path order — and under an "
        "editable install the INSTALLED copy wins silently. See the DELIBERATELY ABSENT "
        "block in environment.yml."
    )
