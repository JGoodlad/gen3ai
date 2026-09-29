"""M5 T2 — the CUDA GRAPH backend's gate: Inductor-compiled decision forward captured per slot x
bucket, parity-gated at startup, weights swapped into the CAPTURED graphs by in-place copy, and a
steady state that compiles, captures and allocates nothing.

GPU + `slow`: each bucket costs one Inductor compile (~75 s on torch 2.5.1, ~42 s on 2.8, measured
2026-09-29). The root conftest hides the GPU from the suite, so this SKIPS with the reason named
unless run on an idle box with ``GEN3AI_TEST_ALLOW_GPU=1``:

    GEN3AI_TEST_ALLOW_GPU=1 python3 -m pytest src/agents/inference/service/service_cuda_test.py -q

Each test gets FRESH Inductor/Triton cache dirs (owner, 2026-09-29: no shared compile caches).
"""
from __future__ import annotations

import pytest
import torch

from agents.inference.service import (
    InferenceService, LifecycleViolation, Priority, ServiceSpec, SlotGroupSpec, policy_reference,
)
from agents.inference.service.parity import fixture_rows
from agents.inference.service.fixtures import perturbed_fresh_policy as _perturbed_policy
from agents.model.extractor_compiles_test import _cuda_skip_reason

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(_cuda_skip_reason() is not None, reason=_cuda_skip_reason() or "")]


@pytest.fixture
def fresh_compile_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _ref(policy, obs, mask):
    dev = torch.device("cuda")
    return policy_reference(policy.to(dev), torch.as_tensor(obs, device=dev),
                            torch.as_tensor(mask, device=dev))


def test_graph_backend_parity_weight_swap_and_a_frozen_steady_state(fresh_compile_caches):
    a, b = _perturbed_policy(0), _perturbed_policy(1)
    spec = ServiceSpec(groups=(SlotGroupSpec("pool", 2, a),), device="cuda", backend="graph",
                       buckets=(4, 8), max_rows_per_flush=64)
    svc = InferenceService(spec).startup()
    st = svc.stats()
    assert st["graphs"] == 4 and st["state"] == "FROZEN"
    # every slot x bucket, full and padded, passed the compile gate's bars
    assert len(svc.startup_reports) == 2 * 2 * 2
    assert all(r.legal_logprob_max < 1e-3 and r.value_max < 1e-4 for r in svc.startup_reports)

    s0, s1 = svc.slot("pool", 0), svc.slot("pool", 1)
    svc.load(s1, b, "policy-B")               # into the CAPTURED graphs' storage, verified
    obs, mask = fixture_rows(svc.obs_dim, 64)
    a_cuda, b_cuda = _ref(a, obs[:11], mask[:11]), _ref(b, obs[:11], mask[:11])
    for _ in range(5):                        # steady state: many flushes, both slots, both buckets
        t0 = svc.submit(s0, obs[:11], mask[:11])                 # chunks 8 + (3 -> 4)
        t1 = svc.submit(s1, obs[:11], mask[:11], Priority.EVAL)
        t2 = svc.submit(s1, obs[:3], mask[:3])
        svc.flush()
        for t, ref in ((t0, a_cuda), (t1, b_cuda)):
            d = t.result()
            fin = torch.isfinite(ref[0])
            assert torch.equal(fin, torch.isfinite(d.logp))
            assert float((d.logp - ref[0])[fin].abs().max()) < 1e-3
            assert float((d.value - ref[1]).abs().max()) < 1e-4
        assert float((t2.result().value - b_cuda[1][:3]).abs().max()) < 1e-4
    svc.canary()
    st = svc.stats()
    assert (st["compiles_after_freeze"], st["captures_after_freeze"],
            st["cuda_segments_after_freeze"]) == (0, 0, 0)

    # a swap back to A in slot 1 is served by the same graphs
    svc.load(s1, a, "policy-A")
    d = svc.score(s1, obs[:8], mask[:8])
    ref = _ref(a, obs[:8], mask[:8])
    assert float((d.value - ref[1]).abs().max()) < 1e-4


def test_eager_cuda_steady_state_allocates_no_segment_and_a_new_one_poisons():
    a = _perturbed_policy(0)
    svc = InferenceService(ServiceSpec(groups=(SlotGroupSpec("pool", 1, a),), device="cuda",
                                       backend="eager", buckets=(4, 8))).startup()
    obs, mask = fixture_rows(svc.obs_dim, 8)
    for n in (8, 3, 5, 8, 1):
        svc.score(0, obs[:n], mask[:n])
    assert svc.stats()["cuda_segments_after_freeze"] == 0
    with pytest.raises(LifecycleViolation, match="CUDA segments"):
        with svc._frozen_guard("test"):
            hold = torch.empty(64 * 2**20, dtype=torch.float32, device="cuda")   # 256 MiB, new
            del hold
    assert svc.state == "POISONED"
