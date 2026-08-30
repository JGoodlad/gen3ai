"""The no-progress clock's two OPT-IN fixes touch ONE observation column — measured, not argued.

`gen3_data_obs_parity_integration_test` already proves the DEFAULT path is byte-identical to the
committed golden (both flags off ⇒ the same 991 per-decision sha256s). That is the landing-safety
claim. This file answers the second, different question: *when a flag is turned ON, what in the
observation actually moves?*

The method is the column-alignment one this tree's obs changes are held to (the v48 / v65 regen
notes in the golden test's own docstring): capture full float32 vectors for both arms over the same
deterministic 6-battle set, align column-wise, and name every cell that differs.

**MEASURED 2026-08-29** over all 991 decisions of the golden battle set:

| arm | decisions that differ | columns that ever differ |
|---|---|---|
| `--progress-decision-tense` | 49 / 991 | `[1602]` |
| `--progress-switch-freeze`  | 153 / 991 | `[1602]` |

Column 1602 is `turns_since_progress` — the clock's own scalar, and the ONLY route by which the
clock reaches the observation at all. So each fix is confined to the counter it redefines: no other
block moves, the decision COUNT does not change (the trajectory does not branch), and no dim moves.

That confinement is the point, and it is also the honest limit of the claim: the flags DO change
the obs stream, because `n` is read by the obs scalar and the reward charge alike (the Markovian
design's whole premise — a fix that moved only the reward would break the identity the clock
exists to provide). Both fixes are therefore retrain-class, which is why they ship OFF.

The battle set is `golden_obs_capture`'s: fixed teams, a deterministic rotate-the-legal-actions
policy for both players, a fixed sim seed — no RNG anywhere, so the two arms of each comparison
differ ONLY in the flag.

Run: `python -m pytest src/agents/training/progress_clock_obs_confinement_integration_test.py -q`
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import numpy as np
import pytest

from agents.observation.constants import OFFSET_REACTIVE
from agents.training import progress_clock as _pc
from agents.training.golden_obs_capture import capture_vectors

# ~5.6 s per capture on the bridge; three captures. `sim` for the same reason the golden test
# carries it — battle-backed but cheap, and it belongs in the routine gate.
pytestmark = pytest.mark.sim

# The clock's obs scalar. Named from the layout at import time rather than pinned as a literal, so
# an offset change re-points this file instead of silently making its claim false.
_CLOCK_COL = OFFSET_REACTIVE + 2   # reactive_layout["turns_since_progress"]["offset"] == 2


def _capture(**clock_flags):
    """The golden battle set, with every `ProgressClock` in the run built under `clock_flags`.

    The flags are per-run reward config in production (`ProgressClock.apply_reward_config`), and
    the capture harness deliberately owns no reward config — so the seam patched here is the
    constructor, which is the same one `EpisodeTracker` calls."""
    original = _pc.ProgressClock.__init__

    def patched(self, no_progress_penalty=0.15, **kw):
        kw.update(clock_flags)
        original(self, no_progress_penalty, **kw)

    _pc.ProgressClock.__init__ = patched          # type: ignore[method-assign]
    try:
        return np.stack([np.asarray(v, dtype=np.float32) for v in capture_vectors()])
    finally:
        _pc.ProgressClock.__init__ = original     # type: ignore[method-assign]


@pytest.fixture(scope="module")
def arms():
    return {
        "default": _capture(),
        "decision_tense": _capture(decision_tense=True),
        "switch_freeze": _capture(switch_freeze=True),
    }


def test_the_clock_column_is_where_this_file_says_it_is(arms):
    """Guards the whole file's claim: if `_CLOCK_COL` did not name the progress scalar, "only column
    1602 moved" would be a true statement about the wrong cell."""
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    assert layout["reactive_layout"]["turns_since_progress"] == {"offset": 2, "dim": 1}
    assert _CLOCK_COL == 1602, f"the reactive block moved; re-measure this file's table ({_CLOCK_COL})"
    assert layout["total_dim"] == arms["default"].shape[1]


@pytest.mark.parametrize("arm", ["decision_tense", "switch_freeze"])
def test_each_fix_changes_exactly_one_observation_column(arms, arm):
    base, other = arms["default"], arms[arm]
    assert other.shape == base.shape, "the decision count or obs dim changed — the trajectory branched"
    cols = sorted(np.nonzero(np.any(base != other, axis=0))[0].tolist())
    assert cols == [_CLOCK_COL], (
        f"{arm} moved columns {cols}; the clock reaches the obs ONLY through "
        f"turns_since_progress ({_CLOCK_COL}), so anything else is an unintended coupling")


@pytest.mark.parametrize("arm,expected_rows", [("decision_tense", 49), ("switch_freeze", 153)])
def test_each_fix_actually_bites(arms, arm, expected_rows):
    """The confinement result above is only meaningful if the flag DID something. These counts are
    the 2026-08-29 measurement on this fixed battle set; they are deterministic, so a change here
    means the fix's behaviour moved (and the docstring's table needs re-measuring)."""
    base, other = arms["default"], arms[arm]
    assert int(np.any(base != other, axis=1).sum()) == expected_rows


def test_the_default_arm_is_the_committed_golden(arms):
    """The landing-safety claim, restated here so this file fails on its own if the default path
    ever stops being the shipped one (the golden test owns the canonical version)."""
    import json
    import os
    from agents.training.golden_obs_capture import vector_hashes
    with open(os.path.join(os.path.dirname(__file__), "golden_obs_fixture.json")) as f:
        golden = json.load(f)
    assert vector_hashes(list(arms["default"])) == golden["hashes"]
