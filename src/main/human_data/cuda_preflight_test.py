"""The CUDA pre-flight's plumbing (``cuda_preflight.py``, Amendment 2 M7), CPU against CPU on a tiny rowset: every
check is computed, CPU-vs-CPU parity is exact, the resume lands on the uninterrupted step, the export round-trips."""
from __future__ import annotations

import pytest

from main.human_data import cuda_preflight as CP
from main.human_data import student as ST
from main.human_data.student_test import _tiny_rowset


@pytest.mark.slow
def test_preflight_cpu_against_cpu(tmp_path, monkeypatch, restore_torch_globals):
    rows = tmp_path / "rows"
    _tiny_rowset(rows, 1100, 4)                        # 3 batches an epoch: 9 steps
    monkeypatch.setattr(ST, "LOSS_EVERY", 1)
    res = CP.preflight(rows, tmp_path / "scratch", "cpu", None, steps=4, long=6)
    c = res["checks"]
    assert set(c) == {"init", "parity", "precision", "resume", "export", "learns", "cost"}
    assert c["init"]["pass"] and c["parity"]["pass"] and c["parity"]["param_rel_l2"] < 1e-12
    assert c["resume"]["pass"] and c["resume"]["step"] == 4 and c["resume"]["opt_devices"] == ["cpu"]
    assert c["export"]["pass"] and c["export"]["cpu_vs_device_logit_abs"] < 1e-5
    assert c["cost"]["steps_per_student"] == 9 and c["learns"]["loss"] is not None
