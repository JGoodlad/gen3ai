"""Does the tree still pass pyflakes? The static gate for the code MYPY does not cover.

Sits beside `strict_api_lock_test.py` at the `src/` root because, like that lock, its subject is
the whole tree rather than any one package.

**Why F and E9, and nothing else.** `--select F,E9` is narrow on purpose. `F` is pyflakes —
undefined names, unused imports, locals assigned and never read, f-strings that forgot their
placeholders — and `E9` is the syntax/IO error class. Every one of those is a statement that the
code is WRONG, not unfashionable. No style rule is selected, so this gate can never turn into an
argument about line length or import order, and a red run always means something real. That
matters for whether people keep it: a linter that cries about formatting gets excluded within a
month, and then the pyflakes findings ride along with it.

**Why it is a test.** Same reason as `agents/model/mypy_gate_test.py`: there is no CI on this box.
The routine suite is the only thing that runs on every change, so a check that is not in it is a
check that runs when someone remembers. `F821` (undefined name) in particular is a real runtime
`NameError` waiting on a branch nobody exercised — precisely the class of bug the unit suite is
worst at reaching.

**Complements mypy, does not overlap it.** `mypy_gate_test.py` is deep but covers ONE package
(`src/agents/model`, per `mypy.ini`). This is shallow but covers `agents/` + `main/` + `utils/`
entire. `src/poke_env` (a vendored FORK) and `src/rust_sim` (a Rust crate; its Python is harness
scratch) are excluded — neither is ours to lint — and those two excludes are mirrored in
`ruff.toml` so a bare `ruff check src/` from a shell agrees with this test.

**Known findings are per-file entries in `ruff.toml`, never a blanket exclude.** Two categories
live there and the file keeps them apart: a genuine PERMANENT pattern (the model package's
documented re-export hubs, which import names solely so third parties can import them back out),
and a TEMPORARY handoff list of ordinary dead code in files that were being edited concurrently
when this gate landed. The second list is meant to shrink to nothing.

**Cost (measured 2026-08-17): 0.10 s.** Ruff is fast enough that there is no tier question — no
cost marker, runs in the fast inner loop. Opt out explicitly:

    GEN3AI_SKIP_RUFF_GATE=1 pytest src/ -q
"""
import os
import subprocess
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Kept as one list so the docstring, `ruff.toml`'s header and the /gen3ai-ship step can all
# quote the SAME command. If you change it, change it in those places too — a gate whose
# documented command differs from the one it runs is worse than no documented command.
_RUFF_ARGV = [
    "-m", "ruff", "check",
    "src/agents", "src/main", "src/utils",
    "--select", "F,E9",
    "--exclude", "src/poke_env",
    "--exclude", "src/rust_sim",
]


@pytest.mark.skipif(os.environ.get("GEN3AI_SKIP_RUFF_GATE") == "1",
                    reason="GEN3AI_SKIP_RUFF_GATE=1")
def test_tree_passes_pyflakes():
    """The ruff command must exit 0, from the repo root.

    Run from `_REPO_ROOT` because the paths above are repo-relative AND because `ruff.toml`
    (which carries the per-file-ignores) is discovered from there.
    """
    assert os.path.isfile(os.path.join(_REPO_ROOT, "ruff.toml")), (
        f"ruff.toml is missing from {_REPO_ROOT}. It holds the per-file-ignores that keep this "
        f"gate honest — without it the run would report the documented re-export hubs as "
        f"findings, and the tempting fix is a blanket exclude. Restore the config."
    )

    proc = subprocess.run([sys.executable] + _RUFF_ARGV,
                          cwd=_REPO_ROOT, capture_output=True, text=True, timeout=600)

    if "No module named ruff" in proc.stderr:
        raise AssertionError(
            "ruff is NOT INSTALLED, so the tree went UNCHECKED — a gap in coverage, not a "
            "pass. It is pinned in environment.yml; install it with\n"
            "    pip install ruff==0.16.3\n"
            "or opt out explicitly:\n"
            "    GEN3AI_SKIP_RUFF_GATE=1 pytest ..."
        )

    assert proc.returncode == 0, (
        "ruff found pyflakes/syntax errors. Most are one-line deletions and `--fix` handles "
        "them; if a finding is a DELIBERATE pattern, add a targeted per-file entry to "
        "ruff.toml (or an inline `# noqa: <code>`) WITH THE REASON — never widen the exclude "
        "list, which would take unrelated real findings down with it.\n\n"
        f"--- ruff stdout ---\n{proc.stdout}\n--- ruff stderr ---\n{proc.stderr}"
    )
