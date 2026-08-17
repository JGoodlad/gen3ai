"""Does `src/agents/model` still TYPE-CHECK? The gate that makes the annotations load-bearing.

**Why this is a test and not a habit.** There is no CI on this box — the routine suite IS the
enforcement layer, and anything not in it is advisory. `mypy.ini` can declare
`disallow_untyped_defs` all it likes; until something FAILS on a violation, the config is a
statement of intent that the next hurried edit quietly falsifies. Annotations rot in exactly the
way the project has already watched other invariants rot: silently, while everything still runs.
So the obligation lives here, beside the code it constrains, and it goes red in the same 4-minute
gate everything else does.

**Scope is `src/agents/model` only, and that is deliberate.** `mypy.ini` sets `files =
src/agents/model` with `follow_imports = silent`, so mypy READS the rest of the tree for types
without reporting its errors. That lets the model package be held to
`disallow_untyped_defs`-grade strictness today without first annotating `training/`,
`observation/` and `battle/`. Widening the scope is a matter of editing `mypy.ini` — this test
runs whatever that file declares and needs no change to follow.

**Cost (measured 2026-08-17, this box, mypy 2.3.1 compiled):** WARM 0.28 s — mypy's incremental
cache makes a no-change run essentially free. COLD (a fresh worktree, no `.mypy_cache`) 19.6 s,
paid once. Both sit under the root `conftest.py`'s 30 s unmarked-tier budget, so this test takes
NO cost marker and stays in the fast inner loop.

**A missing mypy FAILS rather than skips.** `mypy` is declared in `environment.yml` precisely so
that it is present; if it is not, the honest report is "this gate did not run", not a green tick.
That is the project's own lesson about default branches nothing tests — a linter that silently
opts out reads exactly like a linter that found nothing. The one intentional opt-out is explicit:

    GEN3AI_SKIP_MYPY_GATE=1 pytest src/ -q
"""
import os
import subprocess
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_TARGET = "src/agents/model"


@pytest.mark.skipif(os.environ.get("GEN3AI_SKIP_MYPY_GATE") == "1",
                    reason="GEN3AI_SKIP_MYPY_GATE=1")
def test_model_package_type_checks_clean():
    """`python -m mypy src/agents/model` must exit 0, from the repo root.

    Run from `_REPO_ROOT` on purpose: `mypy.ini` lives there and carries the whole
    configuration (scope, strictness, the generated-file excludes), so a run from anywhere else
    would silently pick up mypy's defaults and check a different thing under the same name.
    """
    assert os.path.isfile(os.path.join(_REPO_ROOT, "mypy.ini")), (
        f"mypy.ini is missing from {_REPO_ROOT} — without it this gate would run mypy's "
        f"DEFAULTS (no strictness, no scope, no excludes) and pass while checking almost "
        f"nothing. Restore the config rather than deleting this test."
    )

    proc = subprocess.run(
        [sys.executable, "-m", "mypy", _TARGET],
        cwd=_REPO_ROOT, capture_output=True, text=True, timeout=900,
    )

    # `-m mypy` on a missing module exits 1 with this on stderr. Distinguish it from a real
    # finding so the failure names the CAUSE — "no module named mypy" buried under a
    # type-error banner has sent people looking for a type error that does not exist.
    if "No module named mypy" in proc.stderr:
        raise AssertionError(
            "mypy is NOT INSTALLED, so the model package went UNCHECKED — this is a gap in "
            "coverage, not a pass. It is pinned in environment.yml; install it with\n"
            "    pip install mypy==2.3.1\n"
            "or, if you genuinely mean to run without it, opt out explicitly:\n"
            "    GEN3AI_SKIP_MYPY_GATE=1 pytest ..."
        )

    assert proc.returncode == 0, (
        f"mypy FAILED on {_TARGET} (exit {proc.returncode}). Fix the annotations — or, if a "
        f"finding is a genuine third-party/untyped-boundary false positive, narrow it with a "
        f"targeted `# type: ignore[code]` carrying a reason, never by loosening mypy.ini "
        f"wholesale.\n\n"
        f"--- mypy stdout ---\n{proc.stdout}\n--- mypy stderr ---\n{proc.stderr}"
    )
