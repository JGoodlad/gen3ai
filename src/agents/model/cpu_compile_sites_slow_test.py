"""`gen3_cpu_compile_sites_v1` (2026-10-19): the REAL graph through each CPU compile site at the batches the torch 2.8
Inductor tile2d bug kills (B = 16 / 32 / 64; `designs/training/compile_flags.md` "The CPU Inductor tile2d codegen bug").

The seam (`agents.model.inductor_compile.compile_on`) and its tree scan are pinned in `inductor_compile_test.py` on
the 4-line repro; this file runs the production extractor itself, on the committed REAL parity rows
(`compile_parity_obs.npz`, 64 rows), through:

  * `compile_on` at a STATIC batch — the route of every site that compiles the extractor's forward on CPU
    (`compile_inventory.fullgraph_check`, the bf16 canary's CPU arms around it, `growth`'s probe);
  * `compile_opponents.maybe_compile_extractor` — the offline readers' opponent compile (the prober's
    counterfactual view, `snapshot`'s opponent loads, `compile_cache_benchmark`): installed at B = 1, then asked for
    B = 16 / 32 / 64. Its guard turns a late compile failure into a silent eager FALLBACK (a ~6x slowdown and one
    warning), so the test fails on that warning, not only on an exception.

R1 (the learner micro-step, train mode, with its backward) on CPU is `compiled_first_update_test`'s CPU cell
(B = 16, the K9 golden's micro-batch). All slow: each is a cold CPU Inductor compile of the end-state extractor.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from agents.model.inductor_compile import compile_on

FAILING_B = (16, 32, 64)

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(os.environ.get("GEN3AI_SKIP_COMPILE_TESTS") == "1",
                                 reason="GEN3AI_SKIP_COMPILE_TESTS=1")]


def _rows(fe, b: int):
    """``b`` REAL decision rows (the committed parity fixture), shaped like a live reader's call."""
    from agents.model.compile_parity_fixture import load_parity_rows
    from agents.model.extra_obs_keys import zero_extra_obs
    obs, mask = load_parity_rows(int(fe.layout["total_dim"]))
    assert len(obs) >= b, f"the parity fixture holds {len(obs)} rows, the test needs {b}"
    out = {"observation": torch.as_tensor(np.ascontiguousarray(obs[:b])),
           "action_mask": torch.as_tensor(mask[:b], dtype=torch.float32)}
    out.update(zero_extra_obs(fe, batch=b))
    return out


def _max_err(ref, got) -> float:
    return max(float((a - b).abs().max()) for a, b in zip(ref, got))


@pytest.fixture
def one_thread(restore_torch_globals):
    torch.set_num_threads(1)
    torch._dynamo.reset()
    try:
        yield
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("b", FAILING_B)
def test_the_production_extractor_compiles_on_cpu_at_the_failing_batch(one_thread, b):
    """A static-batch CPU compile of the production forward through the seam compiles (no g++ error) and matches
    eager on real rows."""
    from agents.model.extractor_compiles_test import _build_production_extractor
    assert not torch._dynamo.config.suppress_errors, "a suppressed backend error would pass as eager"
    fe, _ = _build_production_extractor()
    obs = _rows(fe, b)
    with torch.no_grad():
        ref = fe(obs)
        got = compile_on(fe.forward, device="cpu", dynamic=False, fullgraph=True)(obs)
    err = _max_err(ref, got)
    assert err < 1e-5, f"B={b}: compiled output diverged from eager: {err:.2e}"


@pytest.fixture
def fresh_policy():
    from main.fresh_checkpoint import build_fresh_model
    from utils.torch_state_guard import torch_globals
    with torch_globals(num_threads=1):
        model, _, _ = build_fresh_model(0)
        model.policy.set_training_mode(False)
        yield model
        fe = model.policy.features_extractor
        if "forward" in vars(fe):
            del fe.forward
        torch._dynamo.reset()


def test_the_opponent_compile_serves_the_failing_batches_compiled(monkeypatch, fresh_policy):
    """`maybe_compile_extractor` installs (B = 1, its decision-level parity PASS), then a reader's B = 16 / 32 / 64
    batches run COMPILED: no `FELL BACK to eager` warning, eager's numbers. The timing verdict is stubbed (a
    measurement, not this test's subject); the compile and the parity check are real."""
    from agents.model import compile_opponents as S
    from agents.model import opponent_parity as op
    warns = []
    monkeypatch.setattr(S, "_COMPILE_VALIDATED", False)
    monkeypatch.setattr(S, "_measure_arms", lambda e, c, o: ([10.0] * 5, [2.0] * 5))
    monkeypatch.setattr(S, "_compile_warn", warns.append)
    monkeypatch.setattr(op, "_PASSED", {})
    fe = fresh_policy.policy.features_extractor
    assert S.maybe_compile_extractor(fresh_policy, True, label="tile2d-sites") is True, warns
    assert "forward" in vars(fe), "the compile is not installed: the test would be vacuous"
    for b in FAILING_B:
        obs = _rows(fe, b)
        with torch.no_grad():
            got = fe.forward(obs)
            ref = type(fe).forward(fe, obs)
        assert not warns, f"B={b}: the opponent compile fell back to eager — {warns}"
        err = _max_err(ref, got)
        # 1e-4, not the static cell's 1e-5: past B = 1 this site's lazy compile is a DYNAMIC-batch graph (automatic
        # dynamic on the second shape), whose fp32 reduction order differs from eager's (measured 2026-10-19: 1.49e-5
        # at B = 32; the static-B cells read < 1e-5). A wrong kernel is orders of magnitude above either bar.
        assert err < 1e-4, f"B={b}: compiled output diverged from eager: {err:.2e}"
