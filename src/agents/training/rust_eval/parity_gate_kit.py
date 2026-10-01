"""Shared by the Lane H gate's three test files (``parity_test`` / ``parity_fixed_test`` /
``parity_sampled_test``): the per-row PASS assertion (the ``built`` fixture is this directory's
``conftest.py``). Split
2026-09-30 so each file's ~40-70 s module fixture runs on its own xdist worker — the routine gate
schedules whole files (``utils.xdist_schedule``), and one 185 s file was its long pole."""
from __future__ import annotations


def assert_pass(rep):
    g = rep["games"]
    assert not g["missing"], g["missing"]
    assert not g["fatal"], g["fatal"][:3]
    assert g["max_dlogp"] <= rep["cfg"]["bar"], g["max_dlogp"]
    if not g["ties"]:
        assert rep["metrics_equal"], rep["metrics_diffs"][:5]
    t = rep["traces"]
    assert "error" not in t, t.get("error")
    assert not t["only_rust"] and not t["only_python"], (t["only_rust"][:3], t["only_python"][:3])
    assert t["checked"] > 0 and not t["decision_diffs"], t["decision_diffs"][:3]
    assert rep["pass"]


