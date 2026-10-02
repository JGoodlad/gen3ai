"""The CUDA memory ledger (`gen3_cuda_ledger_v1`): inert off CUDA; on a (stubbed) card each row is the
allocator's state after a step, its delta and the step's peak, the peak counter reset per step; the
trainer marks it at every startup step that acquires device memory, on both paths."""
from __future__ import annotations

import inspect
import json

import torch

import agents.training.cuda_ledger as CL


def test_off_cuda_the_ledger_is_inert(tmp_path):
    led = CL.start("cpu")
    assert not led.enabled and led.mark("anything") is None and led.rows == []
    led.report(str(tmp_path))
    assert not (tmp_path / "cuda_ledger.json").exists()


def test_each_row_is_the_state_after_the_step_its_delta_and_the_steps_peak(monkeypatch, tmp_path):
    MiB = CL.MiB
    state = {"alloc": 100 * MiB, "res": 128 * MiB, "peak": 110 * MiB, "resets": 0}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d=None: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d=None: state["alloc"])
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d=None: state["res"])
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d=None: state["peak"])
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (2000 * MiB, 12000 * MiB))

    def reset(d=None):
        state["resets"] += 1
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    led = CL.start("cuda:0", label="weights")
    state.update(alloc=900 * MiB, res=1100 * MiB, peak=1500 * MiB)
    led.mark("rust env core")
    r0, r1 = led.rows
    assert (r0["allocated_mib"], r0["d_allocated_mib"], r0["peak_allocated_mib"]) == (100, 100, 110)
    assert (r1["allocated_mib"], r1["d_allocated_mib"], r1["d_reserved_mib"]) == (900, 800, 972)
    assert r1["peak_allocated_mib"] == 1500 and r1["device_free_mib"] == 2000
    assert state["resets"] == 2                                  # one reset per mark
    led.report(str(tmp_path))
    rows = json.loads((tmp_path / "cuda_ledger.json").read_text())["rows"]
    assert [r["step"] for r in rows] == ["weights", "rust env core"]


def test_the_trainer_marks_every_startup_step_on_both_paths():
    from main.train import model_build
    src = inspect.getsource(model_build)
    for label in ("rust env core", "compiled regions: gate + prewarm + lock",
                  "optimizer state declared"):
        assert src.count(f'_ledger.mark("{label}') == 2, label
    assert src.count("_cuda_ledger.start(model.device)") == 2 and src.count("_ledger.report(model_dir)") == 2
