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
    # 3 slots on 2 LANES: slots 0 and 2 share lane 0 (its pool, in order), slot 1 runs on lane 1
    spec = ServiceSpec(groups=(SlotGroupSpec("pool", 3, a),), device="cuda", backend="graph",
                       buckets=(4, 8), max_rows_per_flush=64, lanes=2)
    svc = InferenceService(spec).startup()
    st = svc.stats()
    assert st["graphs"] == 6 and st["state"] == "FROZEN" and st["lanes"] == 2
    # every slot x bucket, full and padded, passed the compile gate's bars — and every slot at
    # once per bucket (the CONCURRENT gate: lanes replaying together)
    assert len(svc.startup_reports) == 3 * 2 * 2 + 3 * 2
    assert sum("CONCURRENT" in r.where for r in svc.startup_reports) == 3 * 2
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
    _assert_back_to_back_flushes_do_not_race(svc, (s0, a), (s1, b), (svc.slot("pool", 2), a))
    st = svc.stats()
    assert (st["compiles_after_freeze"], st["captures_after_freeze"],
            st["cuda_segments_after_freeze"]) == (0, 0, 0)

    # a swap back to A in slot 1 is served by the same graphs
    svc.load(s1, a, "policy-A")
    d = svc.score(s1, obs[:8], mask[:8])
    ref = _ref(a, obs[:8], mask[:8])
    assert float((d.value - ref[1]).abs().max()) < 1e-4


def _assert_back_to_back_flushes_do_not_race(svc, *slot_policies):
    """Unit 4's double buffer: flushes issued BACK TO BACK with no host sync between them (the
    results cloned on the stream, never read on the host until the end) must each serve their own
    rows. Without the per-arena event wait, flush k+2 re-packs the host arena that flush k's
    still-queued host-to-device copy reads — the rows would be the later flush's."""
    obs, mask = fixture_rows(svc.obs_dim, 64)
    kept = []
    for k in range(8):
        tickets = [(svc.submit(s, obs[k:k + 6 + j], mask[k:k + 6 + j]), pol, k, 6 + j)
                   for j, (s, pol) in enumerate(slot_policies)]
        svc.flush()
        kept += [(t.result().logp.clone(), t.result().value.clone(), pol, k0, n)
                 for t, pol, k0, n in tickets]
    torch.cuda.synchronize()
    for logp, value, pol, k0, n in kept:
        ref = _ref(pol, obs[k0:k0 + n], mask[k0:k0 + n])
        fin = torch.isfinite(ref[0])
        assert float((logp - ref[0])[fin].abs().max()) < 1e-3
        assert float((value - ref[1]).abs().max()) < 1e-4


def test_two_lanes_replaying_concurrently_serve_their_own_slots(fresh_compile_caches):
    """THE REGRESSION for the shared-capture-stream defect: two slots on two lanes, served in ONE
    flush so their graphs replay concurrently. With every graph captured on torch.cuda.graph's
    single default capture stream they share one cuBLAS workspace and the concurrent results are
    wrong (measured max|dlogp| 0.048) — startup's concurrent gate raises ParityFailure. Revert the
    per-lane capture stream in `engine._build_graphs` and this test FAILS."""
    a, b = _perturbed_policy(0), _perturbed_policy(1)
    svc = InferenceService(ServiceSpec(groups=(SlotGroupSpec("pool", 2, a),), device="cuda",
                                       backend="graph", buckets=(8,), lanes=2)).startup()
    svc.load(1, b, "policy-B")
    obs, mask = fixture_rows(svc.obs_dim, 8)
    for _ in range(20):
        t0, t1 = svc.submit(0, obs, mask), svc.submit(1, obs, mask)
        svc.flush()
        for t, pol in ((t0, a), (t1, b)):
            ref = _ref(pol, obs, mask)
            fin = torch.isfinite(ref[0])
            assert float((t.result().logp - ref[0])[fin].abs().max()) < 1e-3
            assert float((t.result().value - ref[1]).abs().max()) < 1e-4


def test_the_double_buffer_wait_stops_a_repack_before_the_queued_copy_reads_it(
        fresh_compile_caches):
    """THE REVERT-MUST-FAIL proof for unit 4's per-arena event wait (`Engine.execute`'s
    ``ev.synchronize()``). The GPU is held busy by a ~0.3 s sleep kernel queued on the caller's
    stream, so every flush's host-to-device copy is still QUEUED when the host issues the next
    flushes. Flush 2 re-packs the arena flush 0's queued copy has not read yet: with the wait it
    blocks until that copy ran; without it, flush 0 is served flush 2's rows — deterministically,
    not by timing luck. Backend ``graph``: the EAGER forward makes 24 host syncs, which block the
    host on the GPU inside every flush and hide the race (measured: the reverted wait PASSED on
    eager) — only a sync-free replay lets the host run ahead of the queued copies."""
    a = _perturbed_policy(0)
    svc = InferenceService(ServiceSpec(groups=(SlotGroupSpec("pool", 1, a),), device="cuda",
                                       backend="graph", buckets=(8,))).startup()
    obs, mask = fixture_rows(svc.obs_dim, 32)
    torch.cuda.synchronize()
    torch.cuda._sleep(int(3e8))                       # ~0.2-0.3 s of GPU time on this card
    kept = []
    for k in range(4):                                # arenas 0, 1, 0, 1 — distinct rows each
        rows = slice(8 * k, 8 * k + 8)
        t = svc.submit(0, obs[rows], mask[rows])
        svc.flush()
        kept.append((t.result().logp.clone(), t.result().value.clone(), rows))
    torch.cuda.synchronize()
    for k, (logp, value, rows) in enumerate(kept):
        ref = _ref(a, obs[rows], mask[rows])
        fin = torch.isfinite(ref[0])
        assert torch.equal(fin, torch.isfinite(logp)), f"flush {k}: served another flush's rows"
        assert float((value - ref[1]).abs().max()) < 1e-4, f"flush {k}: served another flush's rows"
        assert float((logp - ref[0])[fin].abs().max()) < 1e-3


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


def _aot_skip_reason():
    from agents.inference.service import aot

    try:
        aot.check_available()
    except Exception as exc:          # the typed refusal names the cause (torch version, headers)
        return str(exc)
    return None


@pytest.mark.skipif(_aot_skip_reason() is not None, reason=_aot_skip_reason() or "")
def test_aot_backend_one_package_per_bucket_serves_every_slot(fresh_compile_caches, tmp_path):
    """torch >= 2.8 + the CUDA headers (gen3ai_torch28): weights-as-inputs packages, so a load is
    served by the SAME package (no baked weight-derived constants — the embedded-constant package
    served a swapped policy at max|dlogp| 0.076)."""
    a, b = _perturbed_policy(0), _perturbed_policy(1)
    svc = InferenceService(ServiceSpec(groups=(SlotGroupSpec("pool", 2, a),), device="cuda",
                                       backend="aot", buckets=(8,),
                                       artifact_dir=str(tmp_path / "aot"))).startup()
    assert svc.stats()["packages"] == 1
    svc.load(1, b, "policy-B")
    obs, mask = fixture_rows(svc.obs_dim, 8)
    t0, t1 = svc.submit(0, obs[:7], mask[:7]), svc.submit(1, obs[:7], mask[:7])
    svc.flush()
    for t, pol in ((t0, a), (t1, b)):
        ref = _ref(pol, obs[:7], mask[:7])
        fin = torch.isfinite(ref[0])
        assert float((t.result().logp - ref[0])[fin].abs().max()) < 1e-3
        assert float((t.result().value - ref[1]).abs().max()) < 1e-4
    assert svc.stats()["cuda_segments_after_freeze"] == 0
