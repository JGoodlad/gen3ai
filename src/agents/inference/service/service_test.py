"""M5 T2 — the inference service's contract on CPU (eager backend): the decision forward, the slot
model, packing/padding, priority, the declared lifecycle and the parity gate's teeth.

The CUDA graph backend's own gate is `service_cuda_test.py` (GPU, `slow`).

Weights are PERTURBED fresh production-arch policies (`fixtures.perturbed_fresh_policy` says why
a fresh one is a vacuous probe).
"""
from __future__ import annotations

import copy
import types

import numpy as np
import pytest
import torch

from agents.inference.service import (
    CallerError, DecisionModule, InferenceService, LifecycleViolation, ParityFailure, Priority,
    ServiceSpec, SlotArchMismatch, SlotGroupSpec, UnservableArchitecture, policy_reference,
)
from agents.inference.service.decision import _refuse_extra_obs_keys
from agents.inference.service.fixtures import perturbed_fresh_policy as _perturbed_policy
from agents.inference.service.parity import fixture_rows, judge


@pytest.fixture(scope="module")
def policies():
    torch.set_num_threads(2)
    return _perturbed_policy(0), _perturbed_policy(1)


def _service(template, *, n_slots=2, buckets=(2, 8), **kw):
    spec = ServiceSpec(groups=(SlotGroupSpec("pool", n_slots, template),), device="cpu",
                       backend="eager", buckets=buckets, **kw)
    return InferenceService(spec).startup()


def _ref(policy, obs, mask):
    return policy_reference(policy, torch.as_tensor(obs), torch.as_tensor(mask))


def _close(decision, ref, tol=1e-5):
    fin = torch.isfinite(ref[0])
    assert torch.equal(fin, torch.isfinite(decision.logp)), "the -inf (illegal) pattern differs"
    assert float((decision.logp - ref[0])[fin].abs().max()) < tol
    assert float((decision.value - ref[1]).abs().max()) < tol


# ---------------------------------------------------------------- the decision forward
def test_decision_module_is_bit_identical_to_the_policys_own_masked_distribution(policies):
    a, _ = policies
    obs, mask = fixture_rows(a.observation_space["observation"].shape[0], 64)
    o, m = torch.as_tensor(obs), torch.as_tensor(mask)
    with torch.no_grad():
        logp, value = DecisionModule(a).eval()(o, m)
    ref_logp, ref_value = policy_reference(a, o, m)
    assert torch.equal(logp, ref_logp) and torch.equal(value, ref_value)
    assert bool(torch.isneginf(logp[~m]).all()) and bool(torch.isfinite(logp[m]).all())
    # non-vacuous: the perturbed policy's log-probs are not the uniform -log(n_legal)
    uniform = -torch.log(m.sum(1, keepdim=True).float()).expand_as(logp)
    assert float((logp - uniform)[m].abs().max()) > 1e-2


def test_an_extra_obs_key_architecture_is_refused():
    from agents.model.extra_obs_keys import EXTRA_OBS_KEYS

    fe = types.SimpleNamespace(**{EXTRA_OBS_KEYS[0].attr: object()})
    with pytest.raises(UnservableArchitecture, match=EXTRA_OBS_KEYS[0].key):
        _refuse_extra_obs_keys(types.SimpleNamespace(features_extractor=fe))


# ---------------------------------------------------------------- the declaration
@pytest.mark.parametrize("kw, match", [
    (dict(buckets=(1, 8)), "smallest bucket"),
    (dict(buckets=(8, 2)), "ascending"),
    (dict(backend="graph"), "CUDA only"),
    (dict(verify_bucket=4), "not a declared bucket"),
    (dict(max_rows_per_flush=4), "largest bucket"),
])
def test_the_declaration_is_validated(policies, kw, match):
    spec = dict(groups=(SlotGroupSpec("pool", 1, policies[0]),), device="cpu", backend="eager",
                buckets=(2, 8))
    spec.update(kw)
    with pytest.raises(ValueError, match=match):
        ServiceSpec(**spec).validate()


# ---------------------------------------------------------------- the lifecycle
def test_nothing_is_served_before_the_freeze_and_startup_runs_once(policies):
    svc = InferenceService(ServiceSpec(groups=(SlotGroupSpec("pool", 1, policies[0]),),
                                       device="cpu", backend="eager", buckets=(2,)))
    obs, mask = fixture_rows(svc.spec.groups[0].template.observation_space["observation"].shape[0], 2)
    with pytest.raises(LifecycleViolation):
        svc.submit(0, obs, mask)
    svc.startup()
    assert svc.state == "FROZEN"
    with pytest.raises(LifecycleViolation):
        svc.startup()


def test_startup_gates_every_slot_x_bucket_full_and_padded(policies):
    svc = _service(policies[0], n_slots=2, buckets=(2, 8))
    wheres = [r.where for r in svc.startup_reports]
    for slot in (0, 1):
        for b, rows in ((2, 2), (2, 1), (8, 8), (8, 7)):
            assert f"eager group=pool slot={slot} bucket={b} rows={rows}" in wheres
    assert all(r.legal_logprob_max < 1e-3 and r.value_max < 1e-4 for r in svc.startup_reports)


def test_an_acquisition_after_the_freeze_poisons_the_service(policies):
    svc = _service(policies[0], n_slots=1, buckets=(2,))
    with pytest.raises(LifecycleViolation, match="dynamo graphs"):
        with svc._frozen_guard("test"):
            torch.compile(lambda x: x * 2 + 1, backend="eager")(torch.ones(3))
    assert svc.state == "POISONED" and svc.counters["compiles_after_freeze"] >= 1
    obs, mask = fixture_rows(svc.obs_dim, 2)
    with pytest.raises(LifecycleViolation, match="POISONED"):
        svc.submit(0, obs, mask)


def test_a_matmul_precision_change_after_the_freeze_is_a_violation(policies):
    svc = _service(policies[0], n_slots=1, buckets=(2,))
    prev = torch.get_float32_matmul_precision()
    try:
        obs, mask = fixture_rows(svc.obs_dim, 2)
        svc.submit(0, obs, mask)
        torch.set_float32_matmul_precision("high" if prev == "highest" else "highest")
        with pytest.raises(LifecycleViolation, match="precision"):
            svc.flush()
    finally:
        torch.set_float32_matmul_precision(prev)


# ---------------------------------------------------------------- serving
def test_slots_serve_their_own_weights_and_a_load_is_verified(policies):
    a, b = policies
    svc = _service(a, n_slots=2, buckets=(2, 8))
    s0, s1 = svc.slot("pool", 0), svc.slot("pool", 1)
    report = svc.load(s1, b, "policy-B")
    assert report.legal_logprob_max < 1e-3 and svc.model_id(s1) == "policy-B"
    obs, mask = fixture_rows(svc.obs_dim, 13)
    t0, t1 = svc.submit(s0, obs, mask), svc.submit(s1, obs, mask)
    svc.flush()
    _close(t0.result(), _ref(a, obs, mask))
    _close(t1.result(), _ref(b, obs, mask))
    # the two slots really hold different functions
    assert float((t0.result().value - t1.result().value).abs().max()) > 1e-3
    # a load copies IN PLACE: the stacked storage is the same tensor it was at startup
    group = svc.groups[0]
    ptrs = {k: v.data_ptr() for k, v in group.stacked.items()}
    svc.load(s0, b, "policy-B-again")
    assert {k: v.data_ptr() for k, v in group.stacked.items()} == ptrs


def test_rows_are_packed_across_tickets_chunked_and_padded(policies):
    a, _ = policies
    svc = _service(a, n_slots=1, buckets=(2, 8), max_rows_per_flush=64)
    obs, mask = fixture_rows(svc.obs_dim, 64)
    sizes = [5, 1, 11, 3]                                  # 20 rows: chunks 8 + 8 + (4 -> bucket 8)
    tickets, r = [], 0
    for n in sizes:
        tickets.append(svc.submit(0, obs[r:r + n], mask[r:r + n]))
        r += n
    assert svc.flush() == 20
    assert svc.batches_by_bucket == {2: 0, 8: 3} and svc.counters["rows_padded"] == 4
    r = 0
    for t, n in zip(tickets, sizes):
        _close(t.result(), _ref(a, obs[r:r + n], mask[r:r + n]))
        r += n


def test_a_result_is_valid_until_the_next_flush_only(policies):
    svc = _service(policies[0], n_slots=1, buckets=(2,))
    obs, mask = fixture_rows(svc.obs_dim, 2)
    t = svc.submit(0, obs, mask)
    with pytest.raises(CallerError, match="not been served"):
        t.result()
    svc.flush()
    t.result()
    svc.submit(0, obs, mask)
    svc.flush()
    with pytest.raises(CallerError, match="overwritten"):
        t.result()


def test_rollout_is_served_whole_and_filler_rides_one_batch_per_flush(policies):
    svc = _service(policies[0], n_slots=1, buckets=(2, 8), filler_batches_per_flush=1)
    obs, mask = fixture_rows(svc.obs_dim, 8)
    ev = [svc.submit(0, obs[:3], mask[:3], Priority.EVAL) for _ in range(3)]
    fill = svc.submit(0, obs[:2], mask[:2], Priority.FILLER)
    ro = [svc.submit(0, obs, mask) for _ in range(2)]
    svc.flush()
    assert all(t.done for t in ro)
    assert [t.done for t in ev] == [True, False, False] and not fill.done   # FIFO, one batch
    assert svc.pending_rows == {"ROLLOUT": 0, "EVAL": 6, "FILLER": 2}
    svc.drain()
    assert all(t.done for t in ev) and fill.done and svc.pending_rows["EVAL"] == 0


def test_capacity_and_malformed_requests_are_caller_errors(policies):
    svc = _service(policies[0], n_slots=1, buckets=(2, 8), max_rows_per_flush=16)
    obs, mask = fixture_rows(svc.obs_dim, 16)
    with pytest.raises(CallerError, match="obs must be"):
        svc.submit(0, obs[:, :10], mask)
    with pytest.raises(CallerError, match="no legal action"):
        svc.submit(0, obs[:2], np.zeros_like(mask[:2]))
    with pytest.raises(CallerError, match="not declared"):
        svc.submit(5, obs[:2], mask[:2])
    svc.submit(0, obs, mask)
    svc.submit(0, obs[:2], mask[:2])
    with pytest.raises(CallerError, match="exceed max_rows_per_flush"):
        svc.flush()


def test_a_foreign_architecture_or_forward_config_is_refused(policies):
    a, b = policies
    svc = _service(a, n_slots=1, buckets=(2,))
    shapes = copy.deepcopy(b)
    first = shapes.mlp_extractor.policy_net[0]
    shapes.mlp_extractor.policy_net[0] = torch.nn.Linear(first.in_features, first.out_features + 1)
    with pytest.raises(SlotArchMismatch, match="signature"):
        svc.load(0, shapes, "shapes")
    config = copy.deepcopy(b)
    config._critic_mode = "a-different-critic"
    with pytest.raises(SlotArchMismatch, match="FORWARD fingerprint"):
        svc.load(0, config, "config")
    assert svc.state == "FROZEN" and svc.model_id(0) == "<template>"


# ---------------------------------------------------------------- the gate's teeth
def _judge_inputs(policy):
    obs, mask = fixture_rows(policy.observation_space["observation"].shape[0], 16)
    o, m = torch.as_tensor(obs), torch.as_tensor(mask)
    logp, value = policy_reference(policy, o, m)
    return o, m, logp, value


def test_the_gate_passes_the_reference_itself(policies):
    o, m, logp, value = _judge_inputs(policies[0])
    judge(where="self", policy=policies[0], obs=o, mask=m, served=(logp, value, logp.argmax(-1)))


def test_the_gate_fails_a_log_prob_drift_a_value_drift_and_a_broken_mask(policies):
    p = policies[0]
    o, m, logp, value = _judge_inputs(p)
    with pytest.raises(ParityFailure, match="legal_logprob"):
        judge(where="t", policy=p, obs=o, mask=m,
              served=(torch.where(m, logp + 5e-3, logp), value, logp.argmax(-1)))
    with pytest.raises(ParityFailure, match="value"):
        judge(where="t", policy=p, obs=o, mask=m, served=(logp, value + 5e-4, logp.argmax(-1)))
    leaky = logp.clone()
    leaky[~m] = -1e8
    with pytest.raises(ParityFailure, match="mask contract"):
        judge(where="t", policy=p, obs=o, mask=m, served=(leaky, value, logp.argmax(-1)))


def test_the_gate_fails_a_wrong_greedy_action_on_a_decisive_row(policies):
    p = policies[0]
    o, m, logp, value = _judge_inputs(p)
    top2 = logp.topk(2, dim=-1)
    decisive = (top2.values[:, 0] - top2.values[:, 1]) > 1e-3
    assert bool(decisive.any())
    greedy = logp.argmax(-1).clone()
    row = int(torch.nonzero(decisive)[0])
    greedy[row] = top2.indices[row, 1]
    with pytest.raises(ParityFailure, match="greedy"):
        judge(where="t", policy=p, obs=o, mask=m, served=(logp, value, greedy))


# ---------------------------------------------------------------- FRESH weights (gen3_fresh_parity_probe_v1)
@pytest.fixture(scope="module")
def fresh_policy():
    """An UNPERTURBED fresh production policy — exactly what a fresh launch would serve."""
    from main.fresh_checkpoint import build_fresh_model
    torch.set_num_threads(2)
    return build_fresh_model(3)[0].policy.eval()


def _temperature_bug(real):
    """A served path that doubles the logits' sharpness: invisible on a uniform (fresh) row, wrong
    on every informative one — the shape of a miscompile a fresh-weights gate cannot see."""
    def decide(module, obs, mask):
        logp, value, _ = real(module, obs, mask)
        sharp = torch.log_softmax(torch.where(mask, 2.0 * logp, torch.full_like(logp, -1e9)), -1)
        sharp = torch.where(mask, sharp, torch.full_like(sharp, float("-inf")))
        return sharp, value, sharp.argmax(-1)
    return decide


def test_judge_REFUSES_a_vacuous_comparison_on_fresh_weights(fresh_policy):
    from agents.inference.service.spec import VacuousParity
    obs, mask = fixture_rows(fresh_policy.observation_space["observation"].shape[0], 8)
    o, m = torch.as_tensor(obs), torch.as_tensor(mask)
    logp, value = policy_reference(fresh_policy, o, m)
    with pytest.raises(VacuousParity, match="VACUOUS.*legal_logprob"):
        judge(where="fresh", policy=fresh_policy, obs=o, mask=m,
              served=(logp, value, logp.argmax(-1)))
    assert issubclass(VacuousParity, ParityFailure), "an unhandled vacuous gate must fail closed"


def test_a_FRESH_slot_is_gated_on_a_perturbation_and_its_weights_come_back_bit_exact(fresh_policy):
    before = {k: v.clone() for k, v in fresh_policy.state_dict().items()}
    svc = _service(fresh_policy, n_slots=1, buckets=(2, 8))
    wheres = [r.where for r in svc.startup_reports]
    assert any("[fresh weights, seeded perturbation]" in w for w in wheres), wheres
    slot_sd = svc.groups[0].policies[0].state_dict()
    assert all(torch.equal(before[k], slot_sd[k].cpu()) for k in before), \
        "the perturbed gate must leave the slot's weights bit-identical"
    rep = svc.load(0, fresh_policy, "fresh-again")
    assert "[fresh weights" not in rep.where, "load() must return the REAL weights' report first"


def test_a_miscompile_invisible_on_fresh_weights_is_CAUGHT_by_the_perturbed_gate(
        monkeypatch, fresh_policy):
    import agents.inference.service.service as svc_mod
    monkeypatch.setattr(svc_mod, "_decide", _temperature_bug(svc_mod._decide))
    with pytest.raises(ParityFailure, match="seeded perturbation"):
        _service(fresh_policy, n_slots=1, buckets=(2, 8))
