"""G-ENC (F-1b-6) on the golden trace: the reader's full read equals the rows the trainee played on, and a moved bit
is a DIFF (the gate's teeth)."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from main.x50_r1 import genc

pytestmark = pytest.mark.integration
DATA = Path(__file__).parent / "testdata"


def test_the_golden_trace_is_equal_and_a_moved_bit_is_a_diff(tmp_path):
    for f in DATA.glob("trace_loss_s0_002*"):          # the trace's own file names (testdata prefixes them)
        shutil.copy(f, tmp_path / f.name[len("trace_"):])
    summary = str(tmp_path / "loss_s0_002_summary.json")
    assert genc.one(summary)[0] == "equal"
    st = dict(np.load(tmp_path / "loss_s0_002_states.npz"))
    st["obs"] = st["obs"].copy()
    st["obs"][3, 7] = np.nextafter(st["obs"][3, 7], np.float32(np.inf))
    np.savez(tmp_path / "loss_s0_002_states.npz", **st)
    verdict, detail = genc.one(summary)
    assert verdict == "DIFF" and "row 3" in detail
