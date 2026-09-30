"""``gates --gpu``'s GPU session RE-ENTERS a lock its caller already holds — the 2026-09-30 self-deadlock
(``flock gpu.lock python -m main.rust_core_m5 gates … --gpu`` waited 15 min at 0% CPU, because the GPU
pytest took the same lock again through a bare ``flock``).

A real pytest session on a one-test probe file, under a TEMP lock (``$GEN3AI_GPU_LOCK``) that this test
process holds through ``utils.gpu_lock``. On revert to a bare ``flock`` the session blocks on its own
ancestor and the join times out. ``integration``: spawns a pytest subprocess, ~3 s.
"""
from __future__ import annotations

import threading

import pytest

from main.rust_core_m5 import gates as G
from utils.gpu_lock import GPU_LOCK_ENV, HELD_ENV, gpu_lock

pytestmark = pytest.mark.integration


def test_the_gpu_session_reenters_a_lock_its_caller_already_holds(tmp_path, monkeypatch):
    lock = tmp_path / "gpu.lock"
    monkeypatch.setenv(GPU_LOCK_ENV, str(lock))
    monkeypatch.setattr(G, "GPU_LOCK", str(lock))      # never the box's real lock, even on a revert
    monkeypatch.delenv(HELD_ENV, raising=False)
    probe = tmp_path / "probe_gpu_marker_test.py"
    probe.write_text("import os\n\n\ndef test_runs_under_the_held_lock():\n"
                     f"    assert os.environ['{HELD_ENV}']\n"
                     "    assert os.environ['GEN3AI_TEST_ALLOW_GPU'] == '1'\n")
    out: list = []
    with gpu_lock(lock, timeout_s=5) as holder:
        t = threading.Thread(target=lambda: out.append(G._pytest([str(probe)], "not e2e", 1, gpu=True,
                                                                 log=tmp_path / "gates.log")), daemon=True)
        t.start()
        t.join(timeout=120)
        assert out, f"the GPU session blocked on a lock its own ancestor (pid {holder}) holds"
    assert list(out[0].values()) == ["pass"], (out, (tmp_path / "gates.log").read_text()[-2000:])


def test_the_gpu_argv_goes_through_the_helper():
    assert G.gpu_locked_argv(["pytest", "x"])[1:4] == ["-m", "utils.gpu_lock", "--"]
