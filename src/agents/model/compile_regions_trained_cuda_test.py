"""K8 — region R1's startup gate on TRAINED weights, CUDA, fp32 'highest', at the production
micro-batch (GPU tier: `slow`, run with `GEN3AI_TEST_ALLOW_GPU=1` under `scripts/ops/gpu_lock.sh`).

Every production run RESUMES at its interval restart (every 3 h), so R1's gate sees trained weights
there; at fp32 the sizing study's resume of arm C (`ai_v14_02_lbat_ctrl` final) FATAL'd on the
extractor gate's FRESH per-parameter bar (1e-3): 2.47e-3 on `history_events.itemtr_emb.weight`. The
matched-noise control (`designs/research_state/measurements/k6_k8/r1_noise/`) found that reading is
CUDA EAGER's own fp32 error (2.47e-3 against a float64 reference; the compiled gradient 1.7e-5).
This test runs the REAL gate on those weights and FAILS on a revert to the fresh bar for trained
weights (`compile_regions.R1_PARAM_BAR`, chosen by `weights_regime`). It reads arm C's checkpoint
from the box's `models/` archive and SKIPS where there is none."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Optional

import pytest
import torch

from agents.model import compile_control as cc
from agents.model import compile_regions as cr
from agents.model.extractor_compiles_test import _cuda_skip_reason
from utils.paths import main_models_dir

_skip_cuda = pytest.mark.skipif(_cuda_skip_reason() is not None, reason=_cuda_skip_reason() or "")


def _c_final() -> Optional[Path]:
    m = main_models_dir()
    p = None if m is None else m / "ai_v14_02_lbat_ctrl" / "final_model.zip"
    return p if p is not None and p.is_file() else None


@pytest.fixture
def clean_dynamo():
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    prev_prec = torch.get_float32_matmul_precision()
    prev_det = torch.backends.cudnn.deterministic
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev_prec)
        torch.backends.cudnn.deterministic = prev_det
        cc._reset_control_for_tests()
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.reset()


@_skip_cuda
@pytest.mark.skipif(_c_final() is None, reason="arm C's final checkpoint is not in this box's models/")
@pytest.mark.slow
def test_arm_Cs_trained_weights_pass_the_R1_gate_at_fp32_on_cuda(clean_dynamo):
    from agents.training import learner_golden as LG
    path = _c_final()
    assert path is not None
    model = LG.build_learner()
    LG.load_buffer_into(model)
    sd = torch.load(io.BytesIO(zipfile.ZipFile(path).read("policy.pth")), map_location="cpu")
    model.policy.load_state_dict(sd, strict=True)
    dev = torch.device("cuda")
    model.policy.to(dev)
    model.device = dev
    model.rollout_buffer.device = dev
    torch.set_float32_matmul_precision("highest")
    # R1's declared signature is the MODEL's batch size (`install` reads it) and the gate below judges
    # 2048 rows: they must agree. (The golden's 16 made every gate call a "ragged" one, which took a
    # silent EAGER route — this test compared eager with eager until that route was deleted,
    # `gen3_r1_no_ragged_v1`.)
    model.batch_size = 2048
    cc.control().install()
    cr.install(model)
    try:
        assert cr.weights_regime(model) == "trained"
        rules = cr.gate_regions(model, batch_size=2048, say=lambda _m: None)
        assert any(r.startswith("R1 [trained weights]") for r in rules), rules
    finally:
        cr.uninstall(model)
