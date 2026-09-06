"""THE MEASUREMENT-READOUT GATE — can each banked readout still RUN as committed?

WHAT IT GUARDS. A readout script under `designs/research_state/measurements/` is the only thing
standing between a banked number and an un-reproducible claim. Twice now a script was committed
while the per-team artifacts it reads were NOT — they lived in a session-scoped job directory
(`~/.claude/jobs/<id>/tmp/probes/`) that one cleanup would have destroyed:

  * the fleet ADMISSION artifacts, committed 2026-08-31 after `main.exploitability` was written
    against files that existed nowhere in the tree;
  * the 2x2 / K=6 per-team artifacts, found by the teacher-distance probe on 2026-09-05 and given
    their home beside `tc_readout.py` on 2026-09-06 — until then the committed readout behind every
    funded/unfunded/K=6 untaught and taught number could not be run at all.

Both were found by a person reading, not by a test, and in both cases every test in the tree was
green while the artifacts were one `rm -rf /tmp` from gone. So the gate asserts the one property
that fails in exactly this case and in almost no other: **every registered readout resolves all of
its declared inputs.**

WHY `--check` AND NOT A FULL RUN. These readouts bootstrap 20 000 draws and some read `models/`,
which exists only in the main checkout — a full run is neither fast nor portable. `--check`
resolves each input path and reports the missing ones without computing anything, so the gate costs
one interpreter start per script and stays honest about what it proves: that the FILES are there,
not that the numbers are right. Reproducing the numbers is the readout's own job, done by hand.

Adding a readout: append it to REGISTRY and give the script a `--check` that exits non-zero when an
input is missing. A script with no declared inputs does not belong here.

Unmarked, ~0.5 s. Skips cleanly when `designs/` is absent (a source tarball, a slimmed container).
Opt out with GEN3AI_SKIP_READOUT_GATE=1.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from utils.paths import repo_path

# (script, why it is registered) — paths relative to designs/research_state/measurements/
REGISTRY = [
    ("teacher_content_2x2_2026-09-04/tc_readout.py",
     "the funded-vs-unfunded 2x2 contrast + the frozen replicate floor"),
    ("arch_transfer_2026-09-05/teacher_distance/fold_table.py",
     "every gen-era fold's untaught delta, recomputed from per-team rows"),
]

_MEAS = repo_path("designs", "research_state", "measurements")


def _skip_reason() -> str | None:
    if os.environ.get("GEN3AI_SKIP_READOUT_GATE"):
        return "GEN3AI_SKIP_READOUT_GATE=1"
    if not _MEAS.is_dir():
        return f"no designs/ in this checkout ({_MEAS} absent)"
    return None


@pytest.mark.parametrize("rel,why", REGISTRY, ids=[r.split("/")[-1] for r, _ in REGISTRY])
def test_registered_readout_resolves_its_inputs(rel: str, why: str) -> None:
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)

    script = _MEAS / rel
    assert script.is_file(), (
        f"registered readout is MISSING: {script}\n"
        f"  it is registered because it reads: {why}\n"
        "  a readout that is deleted must leave REGISTRY in the same commit."
    )

    proc = subprocess.run(
        [sys.executable, str(script), "--check"],
        capture_output=True, text=True, timeout=120, cwd=str(script.parent),
    )
    out = (proc.stdout + proc.stderr).strip()
    assert proc.returncode == 0, (
        f"{rel} cannot run as committed — its inputs do not resolve.\n"
        f"  it reads: {why}\n"
        f"  exit {proc.returncode}\n{out}\n"
        "  Every number this readout backs is un-reproducible until the missing artifacts are\n"
        "  committed BESIDE it (or at a committed path it resolves). Do not 'fix' this by\n"
        "  copying files from a job/tmp directory at run time — that is the defect."
    )


def test_registry_is_not_silently_empty() -> None:
    """A registry that empties itself turns this whole gate into a no-op that still reports PASSED
    — the vacuous-guard class this tree keeps retiring. Two entries is the floor it shipped with."""
    assert len(REGISTRY) >= 2, "REGISTRY shrank; a readout may have been dropped rather than fixed"
