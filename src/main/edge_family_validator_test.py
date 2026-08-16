"""The CLI's `--edge-bias-families` validator must accept exactly what the MODEL implements.

Why this file exists: v79 added the `h` pair-history family to `_EDGE_FAMILIES`, wired its cell,
widened the obs for it and documented it as opt-in — and the CLI kept a hand-typed copy of the
valid set, so `--edge-bias-families ...,h` died with "unknown families ['h']". The family was
fully built and completely unreachable. A flag that cannot be passed is a feature that does not
exist, and nothing in the suite noticed.

The general rule this pins is the v78 flag-registry one: a toggle's legal VALUES live in exactly
one place, and every other surface derives them.

**On the shape of these tests.** `train_rl_agent.py` builds its parser inline inside `main()`, so
there is no parser to import and no way to reach the validator without running the entry point.
Driving it as a subprocess was tried and REJECTED: an ACCEPTED family set has no early exit, so
the positive case ran on into real training and had to be killed by a 300 s timeout — a test that
launches a trainer is worse than no test. So the accept direction is pinned STRUCTURALLY (the CLI
must derive its set from the model's table, which makes the two incapable of disagreeing) and only
the reject direction — which does exit immediately — is driven end to end.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

from agents.model.features_extractor import _EDGE_FAMILIES

_ENTRY = Path(__file__).with_name("train_rl_agent.py")
_SRC = _ENTRY.read_text()
_MARKER = "--edge-bias-families: unknown families"


def _validator_source() -> str:
    """The ~2 KB of `main()` immediately preceding the error string — where the set is chosen."""
    i = _SRC.find(_MARKER)
    assert i > 0, "the --edge-bias-families validator's error message moved; update this test"
    return _SRC[max(0, i - 2000):i]


def test_the_validator_does_not_re_type_the_family_set():
    """A literal set of family names in the CLI IS the bug, not merely a smell.

    Matches a brace-set of >=5 short quoted lowercase tokens assigned to the valid-set variable —
    the exact shape the stale copy had. Deriving from `_EDGE_FAMILIES` cannot match it.
    """
    literal = re.search(r"_valid\s*=\s*\{\s*(\"[a-z]\d?\"\s*,\s*){5,}", _validator_source())
    assert literal is None, (
        "the CLI re-types the edge-family set; derive it from _EDGE_FAMILIES instead — "
        "this is exactly how `h` shipped unreachable")


def test_the_validator_sources_the_models_table():
    """The positive direction, asserted structurally: if the CLI reads `_EDGE_FAMILIES`, then every
    implemented family is accepted BY CONSTRUCTION and a new one is covered the day it lands.

    Matched on the IMPORT rather than on a particular spelling of the use — `import _EDGE_FAMILIES`
    and `import _EDGE_FAMILIES as _valid` are the same guarantee, and pinning one of them would
    make this test fail on a rename that changed nothing.
    """
    assert re.search(r"from\s+agents\.model\.features_extractor\s+import\s+_EDGE_FAMILIES",
                     _validator_source()), (
        "expected the validator to import the model's family table rather than re-type it")


def test_h_is_in_the_model_table():
    """The regression that motivated the file, named so a revert says why it failed. Combined with
    the two tests above, this is what makes `--edge-bias-families ...,h` reachable."""
    assert "h" in _EDGE_FAMILIES, "v79 pair-history family missing from the model table"


def test_an_unimplemented_family_is_still_rejected():
    """Deriving the set must not turn the validator into a no-op — a typo has to keep failing.

    Safe to drive end to end: a REJECTED set exits at once (measured ~1 s), unlike an accepted one.
    """
    env = {**os.environ,
           "PYTHONPATH": os.environ.get("PYTHONPATH", "") + os.pathsep + str(_ENTRY.parent.parent)}
    p = subprocess.run([sys.executable, str(_ENTRY), "--edge-bias-families", "d1,notafamily"],
                       capture_output=True, text=True, timeout=120, env=env)
    out = p.stdout + p.stderr
    assert _MARKER in out, f"a bogus family was accepted:\n{out[-600:]}"
    assert "notafamily" in out
    # And the message must offer the REAL set, so a user who typos gets the current answer.
    # Matched as a QUOTED list entry: a bare `"h" in out` is satisfied by the letter h in
    # "families", so it passed even against the stale hard-coded set — a vacuous assertion.
    assert re.search(r"['\"]h['\"]", out), (
        f"the error should list the live family set (which now includes 'h'):\n{out[-400:]}")
