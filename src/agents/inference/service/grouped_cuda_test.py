"""X45 — the GROUPED FORWARD on the CUDA GRAPH backend (gen3_t2_grouped_forward_v1): the vmapped forward over the
stacked weights, Inductor-compiled and captured once per grouped range x bucket on the range's lane, parity-gated
at startup and on every load, weights swapped into the CAPTURED graph by in-place copy, and a steady state that
compiles, captures and allocates nothing.

GPU + `slow` (one Inductor compile per range x bucket plus the per-slot ones). The root conftest hides the GPU
from the suite, so this SKIPS with the reason named unless run with a GPU LEASE (root CLAUDE.md rule 5):

    GEN3AI_TEST_ALLOW_GPU=1 scripts/ops/gpu_lock.sh python3 -m pytest src/agents/inference/service/grouped_cuda_test.py -q

COST: FOUR cold production-forward compiles (slot 3's `decide` and the grouped range's vmapped `call_with`, at
buckets 4 and 8; the 3-replica vmapped graph is the largest) plus the parity gates. Measured: 830.7 s, peak RSS
12.6 GB, at `7244e74c` (the 2026-10-19 battery, `~/gen3ai_archive/gpu_battery_2026-10-19/cells/x45_cuda_test.*`) and
837 s alone on the card (2026-10-25, under `timeout 1800`). The test has no in-process bound (one blocking startup
compile has no progress to time); its bound is the GPU window node's: `timeout` of `scale_timeout(1800)`
(`utils.contention`, ~2.15x the measured run) and `mem_cap.sh 24`. The 2026-10-25 window's flat 600 s killed it
(rc 124): a bound, not a hang. The CPU contract is `grouped_test.py`.
"""
from __future__ import annotations

import pytest
import torch

from agents.inference.service import InferenceService, ServiceSpec, SlotGroupSpec, policy_reference
from agents.inference.service.fixtures import perturbed_fresh_policy as _perturbed_policy
from agents.inference.service.parity import fixture_rows
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


def test_grouped_graph_backend_parity_weight_swap_and_a_frozen_steady_state(fresh_compile_caches):
    pols = [_perturbed_policy(k) for k in range(4)]
    # slots 0-2 grouped (one range), slot 3 per-slot (the trainee's place); 2 lanes
    spec = ServiceSpec(groups=(SlotGroupSpec("pool", 4, pols[0]),), device="cuda", backend="graph",
                       buckets=(4, 8), max_rows_per_flush=64, lanes=2, grouped_slots=(0, 1, 2))
    svc = InferenceService(spec).startup()
    st = svc.stats()
    assert st["state"] == "FROZEN" and st["graphs"] == 2              # slot 3's two buckets; the range's are apart
    assert len(svc.engine.ggraphs) == 2                                # the range x (4, 8)
    assert all(r.legal_logprob_max < 1e-3 and r.value_max < 1e-4 for r in svc.startup_reports)
    for s in (1, 2, 3):
        svc.load(s, pols[s], f"p{s}")                                  # into the CAPTURED graphs' storage
    rows = (1, 11, 0, 5)                                               # padding, two rounds, an idle slot
    obs, mask = fixture_rows(svc.obs_dim, sum(rows))
    tickets, a = {}, 0
    for s, n in enumerate(rows):
        if n:
            tickets[s] = (svc.submit(s, obs[a:a + n], mask[a:a + n]), a, n)
        a += n
    before = svc.stats()
    svc.flush()
    dev = torch.device("cuda")
    for s, (t, a0, n) in tickets.items():
        d = t.result()
        o = torch.as_tensor(obs[a0:a0 + n], device=dev)
        m = torch.as_tensor(mask[a0:a0 + n], device=dev)
        ref_lp, ref_v = policy_reference(svc.groups[0].policies[s], o, m)
        assert float((d.logp - ref_lp)[m].abs().max()) < 1e-3, s
        assert float((d.value - ref_v).abs().max()) < 1e-4, s
    after = svc.stats()
    for k in ("compiles_after_freeze", "captures_after_freeze", "cuda_segments_after_freeze"):
        assert after[k] == before[k] == 0, k
    assert after["grouped_rounds"] > before["grouped_rounds"]
