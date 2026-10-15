"""K8 — region R1's startup gate on TRAINED weights, CUDA, at the production micro-batch (GPU tier: `slow`, run
with `GEN3AI_TEST_ALLOW_GPU=1` under a lease and `scripts/ops/gpu_lock.sh` — `scripts/ops/verify_compile_gate_cuda.sh`).

Every production run RESUMES at its interval restart (every 6 h), so R1's gate sees trained weights there. At fp32
the sizing study's resume of arm C (`ai_v14_02_lbat_ctrl` final) FATAL'd on the extractor gate's FRESH
per-parameter bar (1e-3): 2.47e-3 on `history_events.itemtr_emb.weight`, which the matched-noise control
(`designs/research_state/measurements/k6_k8/r1_noise/`) found to be CUDA EAGER's own fp32 error. This test runs
the REAL gate on trained weights and FAILS on a revert to the fresh bar for trained weights
(`compile_regions.R1_PARAM_BAR`, chosen by `weights_regime`).

THE FIXTURE is the `production` baseline, read BY NAME through `agents.training.baselines` (2026-10-15: arm C's
checkpoint is a pre-era architecture that no longer loads — 23 size mismatches — since the end-state adoption).
The entry's `config_overrides` declare its ride-along readouts OFF in production, and the learner golden builds
the production surface, so exactly those heads' `ridealong.*` keys are dropped; every other key loads strict.
SKIPS where the box has no `models/` archive (the baseline does not resolve).

TWO TESTS: the fp32 gate (the `--trunk-precision fp32` opt-out, this test's original intent) and the bf16 rule
(the production recipe) on the same trained weights, which also records the bf16 gate's resolution on trained
weights (`compile_regions_resolution_test`) to `$GEN3AI_GATE_RESOLUTION_OUT`."""
from __future__ import annotations

import io
import zipfile
from typing import Any, Dict, Optional, Tuple

import pytest
import torch

from agents.model import compile_control as cc
from agents.model import compile_regions as cr
from agents.model import trunk_precision as TP
from agents.model.extractor_compiles_test import _cuda_skip_reason

_skip_cuda = pytest.mark.skipif(_cuda_skip_reason() is not None, reason=_cuda_skip_reason() or "")
BASELINE = "production"
#: the ride-along readouts' policy-child prefix (`ridealong_heads`), dropped only when the entry declares them off
RIDEALONG_PREFIX = "ridealong."
_RIDEALONG_OFF = {"ridealong_adv": 0, "ridealong_ensemble": 0, "ridealong_opp": 0, "ridealong_rnd": False}


def _resolved() -> Tuple[Optional[Any], str]:
    """(the resolved `production` baseline, "") or (None, why it does not resolve on this box)."""
    from agents.training import baselines as BL
    try:
        return BL.resolve(BASELINE), ""
    except BL.BaselineError as exc:
        return None, f"the {BASELINE!r} baseline does not resolve on this box (no models/ archive?): {exc}"


def production_policy_state(resolved: Any) -> Dict[str, torch.Tensor]:
    """The `production` baseline's policy state dict for the learner golden's production surface: its `ridealong.*`
    keys dropped ONLY because the registry entry declares those readouts off (`config_overrides`); anything else
    the zip carries is loaded strict by the caller."""
    sd = torch.load(io.BytesIO(zipfile.ZipFile(resolved.zip_path).read("policy.pth")), map_location="cpu")
    over = resolved.baseline.config_overrides
    if any(k.startswith(RIDEALONG_PREFIX) for k in sd):
        off = {k: over.get(k) for k in _RIDEALONG_OFF}
        assert off == _RIDEALONG_OFF, (
            f"the {BASELINE!r} checkpoint carries ride-along heads but its entry does not declare them off in "
            f"production ({off}) — dropping their keys would be a guess")
        sd = {k: v for k, v in sd.items() if not k.startswith(RIDEALONG_PREFIX)}
    return sd


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


def _trained_cuda_learner(trunk_precision: Optional[str]) -> Any:
    from agents.training import learner_golden as LG
    resolved, why = _resolved()
    assert resolved is not None, why
    model = LG.build_learner(trunk_precision=trunk_precision)
    LG.load_buffer_into(model)
    model.policy.load_state_dict(production_policy_state(resolved), strict=True)
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
    return model


_skip_unresolved = pytest.mark.skipif(_resolved()[0] is None, reason=_resolved()[1] or "resolves")


@_skip_unresolved
@pytest.mark.integration
def test_the_production_fixture_loads_STRICT_into_the_learner_golden_and_reads_as_trained_on_cpu():
    """The CPU half of the fixture (no GPU): the GPU tests below would otherwise find a stale fixture only in a GPU
    window — exactly how arm C's pre-era checkpoint (23 size mismatches) went unnoticed until 2026-10-15. The
    `production` baseline's policy loads STRICT into the learner golden's production surface (ride-along keys
    dropped only as its entry declares), and the gate reads its weights as TRAINED at both trunk precisions."""
    from agents.training import learner_golden as LG
    resolved, why = _resolved()
    assert resolved is not None, why
    sd = production_policy_state(resolved)
    for prec in ("fp32", None):
        model = LG.build_learner(trunk_precision=prec)
        LG.load_buffer_into(model)
        model.policy.load_state_dict(sd, strict=True)
        assert TP.trunk_precision_of(model.policy) == (prec or "bf16")
        assert cr.weights_regime(model) == "trained"


@_skip_cuda
@_skip_unresolved
@pytest.mark.slow
def test_the_production_baselines_trained_weights_pass_the_R1_gate_at_fp32_on_cuda(clean_dynamo):
    model = _trained_cuda_learner("fp32")
    assert TP.trunk_precision_of(model.policy) == "fp32"     # PRECONDITION: the fp32 gate's arithmetic
    try:
        assert cr.weights_regime(model) == "trained"
        rules = cr.gate_regions(model, batch_size=2048, say=lambda _m: None)
        assert any(r.startswith("R1 [trained weights]") for r in rules), rules
        assert not any("bf16" in r for r in rules), rules
    finally:
        cr.uninstall(model)


@_skip_cuda
@_skip_unresolved
@pytest.mark.slow
def test_the_production_baselines_trained_weights_pass_the_R1_gate_under_the_bf16_rule_on_cuda(
        clean_dynamo, monkeypatch):
    """The production recipe's gate on trained weights (the weights were trained at fp32: a bf16 fork of
    production meets exactly this read at its first launch). Records the bf16 gate's resolution there, confirmed
    on the encoder parameter (0.5x passes, 2x refused)."""
    from agents.model import compile_regions_resolution_test as RT
    model = _trained_cuda_learner(None)
    assert TP.trunk_precision_of(model.policy) == "bf16"     # PRECONDITION: the production recipe is the bf16 trunk
    try:
        assert cr.weights_regime(model) == "trained"
        rules = cr.gate_regions(model, batch_size=2048, say=lambda _m: None)
        assert any(r.startswith("R1 [trained weights] bf16 rule:") for r in rules), rules
        rec = RT.confirm_resolution(model, monkeypatch, 2048, RT.REPRESENTATIVE[:1])
        rec["what"] = (f"the {BASELINE!r} baseline's trained weights (ridealong.* dropped as declared), production "
                       f"bf16 recipe, CUDA Inductor, B=2048")
        RT.write_record("bf16_trained", rec)
    finally:
        cr.uninstall(model)
