"""THE NEAR-MARGIN DUMP (`gen3_canary_near_margin_dump_v1`, 2026-10-25; the X26 continuation's u1000 canary,
`designs/research_state/measurements/x26c_canary_margin_2026-10-25/` F1).

At `rb_x26c_s2601` update 1000 the canary PASSED with the compiled arm 0.7124 from the fp32 reference against M 0.72 (a
1 % headroom), and a PASS dumped nothing, so the 478.31M weights were never saved and the CUDA read cannot be replayed.
Here, on CPU, with the rung's arms planted (no learner, no compile; the REAL `_r1_judge` and margin rule):

* the u1000 shape PASSES and its record carries ``policy_pt`` (the weights, loadable) and ``near_margin``;
* a PASS far from every bar dumps nothing, and every record carries ``bar_fractions``;
* the near-margin files ROTATE per run dir (the newest `NEAR_MARGIN_KEEP` survive, across a restart whose update
  counter starts again), and never touch an UNCONFIRMED / FATAL dump;
* `bar_fractions` reads each bf16 quantity against the bar it is judged at (drift, envelope, per-parameter ratio) and
  the clip margin against the margin the rung was judged at (M′ after a yardstick re-run);
* a failed write leaves the PASS standing with ``policy_pt`` None.

Every planted value sits far from 0.9 of its bar (rule 8): 0.99 / 0.95 / 0.92 vs 0.42 / 0.5 / 0.8."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch

from agents.model import compile_canary as CC
from agents.model import compile_regions as cr
from agents.model import compile_trainer as ct
from agents.model.canary_yardstick_test import ROWS, _judge, _Rung

M = cr.R1_CLIP_MARGIN
U1000 = {"compiled": 0.7124, "eager": 0.224}         # the x26c u1000 scheduled read's max |dlogp| per arm
U1000_STEPS = 478_310_831
FAR = {"compiled": 0.30, "eager": 0.20}              # 0.42 of M


@pytest.fixture
def world(monkeypatch, tmp_path):
    """A `CompileCanary` whose check runs the REAL `_r1_judge` and margin rule on planted arms (per check: the shifts
    the test lists), recording into the canary's own watch."""
    monkeypatch.setattr(CC, "_WEIGHT_DUMPS", 0)
    policy = torch.nn.Linear(3, 2)
    model = type("M", (), {})()
    model.policy, model.num_timesteps, model.behaviour_dump_dir = policy, U1000_STEPS, str(tmp_path)
    plan: Dict[str, Any] = {"checks": [], "rows": {"eager": 31, "compiled": 50}}
    said: List[str] = []

    def check(self, slice_, out, watch=None):
        shifts = plan["checks"].pop(0)
        if watch is not None:
            watch.fixture_rows = [int(x) for x in ct.fixture_index(2 * ROWS, ROWS, slice_)]
        with monkeypatch.context() as mp:
            _Rung(mp, shifts, plan["rows"])
            return [_judge(watch)]
    monkeypatch.setattr(CC.CompileCanary, "_check", check)

    def canary() -> CC.CompileCanary:
        return CC.CompileCanary(model, batch_size=2 * ROWS, every=1, saver=lambda _p: None, emit=said.append)
    return canary, model, plan, tmp_path, said


def _records(tmp) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in (tmp / CC.VERDICTS_FILE).read_text().splitlines()]


def _near_files(tmp) -> List[str]:
    return sorted(f for f in os.listdir(tmp) if f.endswith(CC.NEAR_MARGIN_SUFFIX))


# ----------------------------------------------------------------------------------- the dump on a PASS
def test_the_x26c_u1000_PASS_dumps_its_weights_and_records_the_path(world):
    """Fails on revert: a PASS wrote no ``policy_pt``, so the u1000 weights were lost."""
    canary, model, plan, tmp, said = world
    can = canary()
    plan["checks"] = [[U1000]]
    out = can.after_update()
    assert out["compile/canary_ok"] == 1.0 and not can.prev_disagreed           # the PASS stands
    (rec,) = _records(tmp)
    assert rec["verdict"] == "pass" and rec["near_margin"] == ["clip_margin"]
    assert rec["policy_pt"] == f"canary_u1_s{U1000_STEPS}{CC.NEAR_MARGIN_SUFFIX}"
    sd = torch.load(tmp / rec["policy_pt"])
    assert set(sd) == {"weight", "bias"} and torch.equal(sd["weight"], model.policy.weight.detach())
    cm = rec["bar_fractions"]["clip_margin"]
    assert cm["arm"] == "compiled" and cm["fraction"] == pytest.approx(0.7124 / M, rel=1e-3)
    assert cm["bar"] == pytest.approx(M)
    assert not (tmp / (rec["policy_pt"] + ".tmp")).exists()
    assert any("📐" in s and "clip_margin at 0.989" in s and rec["policy_pt"] in s for s in said), said
    assert CC._WEIGHT_DUMPS == 0                                                  # not the UNCONFIRMED cap


def test_a_PASS_far_from_every_bar_dumps_nothing_but_records_its_fractions(world):
    canary, _model, plan, tmp, _said = world
    plan["checks"] = [[FAR]]
    canary().after_update()
    (rec,) = _records(tmp)
    assert rec["verdict"] == "pass" and "policy_pt" not in rec and "near_margin" not in rec
    assert rec["bar_fractions"]["clip_margin"]["fraction"] == pytest.approx(0.30 / M, rel=1e-3)
    assert _near_files(tmp) == []


def test_the_near_margin_dumps_ROTATE_per_run_across_a_restart_and_spare_an_UNCONFIRMED(world):
    canary, model, plan, tmp, _said = world
    unconfirmed = tmp / "canary_u7_unconfirmed_policy.pt"
    torch.save({}, unconfirmed)
    names = []
    for proc in range(2):                                   # two processes: the update counter starts again at 1
        can = canary()
        for _ in range(3):
            model.num_timesteps += 1_000_000
            plan["checks"] = [[U1000]]
            can.after_update()
            names.append(_records(tmp)[-1]["policy_pt"])
            os.utime(tmp / names[-1], ns=(len(names) * 10**9, len(names) * 10**9))   # a deterministic write order
    assert len(set(names)) == 6                             # unique across the restart (the step is in the name)
    assert _near_files(tmp) == sorted(names[-CC.NEAR_MARGIN_KEEP:])
    assert unconfirmed.exists()
    assert [r["policy_pt"] for r in _records(tmp)] == names                # each record keeps its own file's name


def test_a_failed_write_leaves_the_PASS_standing(world, monkeypatch):
    canary, _model, plan, tmp, said = world

    def boom(*_a, **_k):
        raise OSError("disk full")
    monkeypatch.setattr(CC.torch, "save", boom)
    plan["checks"] = [[U1000]]
    assert canary().after_update()["compile/canary_ok"] == 1.0
    (rec,) = _records(tmp)
    assert rec["verdict"] == "pass" and rec["policy_pt"] is None and rec["near_margin"] == ["clip_margin"]
    assert any("NOT dumped" in s for s in said)


def test_without_a_run_directory_nothing_is_written():
    m = SimpleNamespace(policy=torch.nn.Linear(2, 2))
    assert CC._dump_near_margin(m, None, 3, 10) is None


# ----------------------------------------------------------------------------------- bar_fractions
def _bf16_watch(**g: float) -> cr.GradWatch:
    w = cr.GradWatch(("pointer_head.",))
    w.rule = "bf16"
    w.globals = dict(g)
    return w


def test_the_bf16_quantities_are_read_against_the_bars_they_are_judged_at():
    assert cr.BF16_DRIFT_BARS is not None and cr.BF16_ENVELOPE_K is not None
    lb, gb = cr.BF16_DRIFT_BARS["loss"], cr.BF16_DRIFT_BARS["grad_rel_l2"]
    kl, kg, kp = cr.BF16_ENVELOPE_K["loss"], cr.BF16_ENVELOPE_K["grad_rel_l2"], cr.BF16_ENVELOPE_K["param"]
    w = _bf16_watch(drift_loss=0.95 * lb, drift_grad=0.5 * gb,
                    envelope_loss=0.5 * kl * 0.95 * lb,                         # eager's drift above the loss floor
                    envelope_grad=0.92 * kg * cr.BF16_GRAD_FLOOR)               # eager's drift 0.34 > the floor 0.014
    w.worst = {"param": "edge_bias.c2_map.weight", "ratio": 0.92 * kp, "compiled": 1.0, "eager": 0.1}
    f = CC.bar_fractions(w)
    assert set(f) == {"drift_loss", "drift_grad", "envelope_loss", "envelope_grad", "param_ratio"}
    assert f["drift_loss"]["fraction"] == pytest.approx(0.95) and f["drift_grad"]["fraction"] == pytest.approx(0.5)
    assert f["envelope_loss"]["fraction"] == pytest.approx(0.5)
    assert f["envelope_grad"]["bar"] == pytest.approx(kg * 0.5 * gb)             # K x max(eager's drift, floor)
    assert f["param_ratio"]["fraction"] == pytest.approx(0.92) and f["param_ratio"]["param"] == "edge_bias.c2_map.weight"
    assert CC.near_margin(f) == ["drift_loss", "param_ratio"]


def test_the_envelope_bar_is_FLOORED_like_the_rule():
    assert cr.BF16_ENVELOPE_K is not None
    w = _bf16_watch(drift_loss=0.0, drift_grad=0.0, envelope_loss=0.95 * cr.BF16_ENVELOPE_K["loss"] * cr.BF16_LOSS_FLOOR,
                    envelope_grad=0.0)
    f = CC.bar_fractions(w)
    assert f["envelope_loss"]["fraction"] == pytest.approx(0.95) and CC.near_margin(f) == ["envelope_loss"]


def test_a_non_finite_quantity_reads_as_AT_its_bar():
    w = _bf16_watch(drift_loss=float("nan"), drift_grad=0.0)
    f = CC.bar_fractions(w)
    assert f["drift_loss"]["fraction"] == float("inf") and "drift_loss" in CC.near_margin(f)
    assert CC._frac(1.0, 0.0) == float("inf") and CC._frac(0.5, 1.0) == 0.5


def test_the_fp32_rule_and_an_empty_watch_read_only_what_they_filled():
    w = cr.GradWatch(())
    assert CC.bar_fractions(w) == {}
    w.rule, w.globals = "fp32", {"drift_loss": 1.0}
    assert CC.bar_fractions(w) == {}


def test_after_a_YARDSTICK_rerun_the_clip_margin_is_read_against_M_prime(world):
    """Eager 0.856 > M: re-run at M' = 1.07, where eager reads 0.8 of it and compiled 0.52 / 1.07 = 0.49 — no dump."""
    canary, _model, plan, tmp, _said = world
    plan["checks"] = [[{"eager": 0.856, "compiled": 0.52}]]
    canary().after_update()
    (rec,) = _records(tmp)
    cm = rec["bar_fractions"]["clip_margin"]
    assert cm["arm"] == "eager" and cm["bar"] == pytest.approx(1.07, rel=1e-3)
    assert cm["fraction"] == pytest.approx(0.8, rel=1e-3) and "policy_pt" not in rec


def test_the_trigger_is_its_declaration():
    assert CC.NEAR_MARGIN_FRACTION == 0.9 and CC.NEAR_MARGIN_KEEP == 3
    assert CC.near_margin({"a": {"fraction": 0.9}, "b": {"fraction": 0.8999}}) == ["a"]
