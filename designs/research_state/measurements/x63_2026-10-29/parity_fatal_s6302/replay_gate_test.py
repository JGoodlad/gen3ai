"""The replay's verdict rule and its CPU path (run explicitly: `pytest <this file>`; not a registered read)."""
from __future__ import annotations

import argparse
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_gate as rg  # noqa: E402


def _rows(spec):
    return [{"row": r, "fixture_row": r % 64, "max_dlogp": d, "action": 0, "eager_logp": -1.0} for r, d in spec]


def test_attribute_tie_class_only_when_every_over_bar_row_is_a_predicted_row():
    gaps = [0.0] * 64                   # a gap is reported, never judged: even a tie everywhere attributes nothing
    v, over = rg.attribute(_rows([(54, 5.19e-3), (118, 5.19e-3), (55, 4e-3), (3, 2e-5)]), gaps)
    assert v == rg.TIE_CLASS and [r["fixture_row"] for r in over] == [54, 54, 55]
    v, _ = rg.attribute(_rows([(54, 5.19e-3), (7, 2e-3)]), gaps)
    assert v == rg.BEYOND, "an over-bar row outside the predicted pair is a divergence"
    v, over = rg.attribute(_rows([(54, 9e-4)]), gaps)
    assert v == rg.NOT_REPRODUCED and over == []


def test_attribute_bar_boundary_and_nan():
    gaps = [1.0] * 64
    assert rg.attribute(_rows([(1, rg.BAR)]), gaps)[0] == rg.BEYOND              # AT the bar is over: the gate's `<`
    assert rg.attribute(_rows([(1, 0.999 * rg.BAR)]), gaps)[0] == rg.NOT_REPRODUCED
    assert rg.attribute(_rows([(1, float("nan"))]), gaps)[0] == rg.BEYOND        # a NaN is over


def test_row_report_reads_legal_entries_only():
    mask = torch.tensor([[True, True, False], [True, False, True]])
    e = torch.tensor([[-0.1, -2.0, float("-inf")], [-0.5, float("-inf"), -1.0]])
    s = e.clone()
    s[1, 2] += 3e-3
    s[0, 2] = 0.0                       # an illegal entry's value never counts
    rep = rg.row_report(e, s, mask)
    assert rep[0]["row"] == 1 and rep[0]["action"] == 2 and abs(rep[0]["max_dlogp"] - 3e-3) < 1e-6   # fp32 -1.0 + 3e-3, read in float64
    assert rep[1]["max_dlogp"] == 0.0


@pytest.mark.slow
def test_cpu_path_runs_on_a_perturbed_fresh_policy():
    """The whole replay minus the GPU: eager vs eager reads NOT REPRODUCED, the eager service passes, every
    fixture row gets a gap."""
    from agents.inference.service.fixtures import perturbed_fresh_policy

    torch.set_float32_matmul_precision("highest")
    args = argparse.Namespace(buckets=(8,), no_compile=True, no_service=False, backend="eager", slots=2, lanes=1,
                              trainee_slot=1, opp_cap=8)
    res = rg.replay_one(perturbed_fresh_policy(0), device=torch.device("cpu"), args=args)
    assert rg.overall(res) == rg.NOT_REPRODUCED
    assert res["service"]["status"] == "PASS", res["service"]
    assert len(res["smallest_gaps"]) == 6


def test_cuda_replay_refuses_without_a_lease(monkeypatch, capsys):
    monkeypatch.delenv("GEN3AI_GPU_LEASE_TOKEN", raising=False)
    assert rg.main(["--run-dir", "/nonexistent", "--ckpt", "x.zip", "--device", "cuda"]) == 2
    assert "GPU lease" in capsys.readouterr().out
