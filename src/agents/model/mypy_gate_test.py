"""Do the type-checked packages still TYPE-CHECK? The gate that makes the annotations load-bearing.

**Why this is a test and not a habit.** There is no CI on this box — the routine suite IS the
enforcement layer, and anything not in it is advisory. `mypy.ini` can declare
`disallow_untyped_defs` all it likes; until something FAILS on a violation, the config is a
statement of intent that the next hurried edit quietly falsifies. Annotations rot in exactly the
way the project has already watched other invariants rot: silently, while everything still runs.
So the obligation lives here, beside the code it constrains, and it goes red in the same 4-minute
gate everything else does.

**Scope is `src/agents/model` + `src/agents/observation`, and the rest of the tree is deliberately
out.** `mypy.ini` names both under one `files =` at one strictness tier, with
`follow_imports = silent` so mypy READS the remainder for types without reporting its errors. That
lets the checked packages be held to `disallow_untyped_defs`-grade strictness without first
annotating `training/` and `battle/`.

**The scope lives in `mypy.ini`, not here — but the gate PINS it.** This test invokes bare
`python -m mypy` with NO path argument, so it checks exactly what the config declares and a future
widening needs no edit to this file. The previous version hard-coded `src/agents/model` as an
argv, which would have kept silently checking one package after `observation` joined the config.
The mirror hazard is the reason for `_CHECKED_PACKAGES`: a `files =` shrunk by accident makes mypy
exit 0 by checking LESS, which is indistinguishable from a pass. So the declared scope is asserted
separately from the exit code — the same principle as the rest of this tree, that a green result
must not be reachable by doing nothing.

**Cost (measured 2026-08-17, this box, mypy 2.3.1 compiled):** WARM 0.28 s — mypy's incremental
cache makes a no-change run essentially free. COLD (a fresh worktree, no `.mypy_cache`) 19.6 s,
paid once; the `observation` widening added 21 source files to a 47-file scope and did not move
either figure out of its tier. Both sit under the root `conftest.py`'s 30 s unmarked-tier budget,
so this test takes NO cost marker and stays in the fast inner loop.

**A missing mypy FAILS rather than skips.** `mypy` is declared in `environment.yml` precisely so
that it is present; if it is not, the honest report is "this gate did not run", not a green tick.
That is the project's own lesson about default branches nothing tests — a linter that silently
opts out reads exactly like a linter that found nothing. The one intentional opt-out is explicit:

    GEN3AI_SKIP_MYPY_GATE=1 pytest src/ -q
"""
import configparser
import os
import subprocess
import sys

import pytest

from utils.paths import repo_root

_REPO_ROOT = str(repo_root())

# Every package `mypy.ini` is expected to hold at the zero-error bar. Adding one here without
# adding it to `mypy.ini` fails the scope assertion below — which is the point: the two must
# move together, or the gate starts reporting on a scope nobody chose.
_CHECKED_PACKAGES = ("src/agents/model", "src/agents/observation")


@pytest.mark.skipif(os.environ.get("GEN3AI_SKIP_MYPY_GATE") == "1",
                    reason="GEN3AI_SKIP_MYPY_GATE=1")
def test_checked_packages_type_check_clean():
    """`python -m mypy` (scope from `mypy.ini`) must exit 0, from the repo root.

    Run from `_REPO_ROOT` on purpose: `mypy.ini` lives there and carries the whole
    configuration (scope, strictness, the generated-file excludes), so a run from anywhere else
    would silently pick up mypy's defaults and check a different thing under the same name.
    """
    config = os.path.join(_REPO_ROOT, "mypy.ini")
    assert os.path.isfile(config), (
        f"mypy.ini is missing from {_REPO_ROOT} — without it this gate would run mypy's "
        f"DEFAULTS (no strictness, no scope, no excludes) and pass while checking almost "
        f"nothing. Restore the config rather than deleting this test."
    )

    # The scope is the config's, so verify the config still claims it. mypy exits 0 just as
    # happily on a scope of nothing, and that is the one failure this gate could not otherwise
    # see (it looks identical to "everything is clean").
    #
    # PARSE the `files =` value — do not substring-search the file. `mypy.ini`'s header comment
    # names both packages in prose, so a plain `in config_text` matches the COMMENT and reports
    # a scope that is no longer configured. That was this assertion's first implementation and
    # it passed against a deliberately shrunk `files =`; the check that cannot fail is worth
    # less than no check, because it also reads as coverage.
    parser = configparser.ConfigParser()
    parser.read(config, encoding="utf-8")
    declared = {p.strip().rstrip("/")
                for p in parser.get("mypy", "files", fallback="").split(",") if p.strip()}
    missing = [p for p in _CHECKED_PACKAGES if p not in declared]
    assert not missing, (
        f"mypy.ini no longer names {missing} in its `files =` scope (it declares {sorted(declared)}), "
        f"so those packages are "
        f"UNCHECKED and this gate would pass by checking less. Restore the scope, or — if the "
        f"narrowing is intentional — drop the package from `_CHECKED_PACKAGES` in the same "
        f"change, so the reduction is a decision on the record rather than a silent one."
    )

    proc = subprocess.run(
        [sys.executable, "-m", "mypy"],
        cwd=_REPO_ROOT, capture_output=True, text=True, timeout=900,
    )

    # `-m mypy` on a missing module exits 1 with this on stderr. Distinguish it from a real
    # finding so the failure names the CAUSE — "no module named mypy" buried under a
    # type-error banner has sent people looking for a type error that does not exist.
    if "No module named mypy" in proc.stderr:
        raise AssertionError(
            "mypy is NOT INSTALLED, so the checked packages went UNCHECKED — this is a gap in "
            "coverage, not a pass. It is pinned in environment.yml; install it with\n"
            "    pip install mypy==2.3.1\n"
            "or, if you genuinely mean to run without it, opt out explicitly:\n"
            "    GEN3AI_SKIP_MYPY_GATE=1 pytest ..."
        )

    assert proc.returncode == 0, (
        f"mypy FAILED on {list(_CHECKED_PACKAGES)} (exit {proc.returncode}). Fix the "
        f"annotations — or, if a finding is a genuine third-party/untyped-boundary false "
        f"positive, narrow it with a targeted `# type: ignore[code]` carrying a reason, never "
        f"by loosening mypy.ini wholesale (the two packages share one config, so a loosening "
        f"to clear one of them silently de-tiers the other).\n\n"
        f"--- mypy stdout ---\n{proc.stdout}\n--- mypy stderr ---\n{proc.stderr}"
    )
