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
    CallerError, CopyParityFailure, DecisionModule, InferenceService, LifecycleViolation, ParityFailure, Priority,
    ServiceSpec, SlotArchMismatch, SlotGroupSpec, UnservableArchitecture, policy_reference,
)
from agents.inference.service.decision import _refuse_extra_obs_keys
from agents.inference.service.fixtures import perturbed_fresh_policy as _perturbed_policy
from agents.inference.service.parity import fixture_rows, judge
from utils.torch_state_guard import torch_globals


@pytest.fixture(scope="module")
def policies():
    with torch_globals(num_threads=2):          # restored at module end (the global-state guard)
        yield _perturbed_policy(0), _perturbed_policy(1)


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


def test_an_extra_obs_key_architecture_is_refused(monkeypatch):
    """The registry is EMPTY today (the privileged true-team route was deleted), so the refusal is
    exercised against a stand-in row: an extractor whose forward would read a Dict key beyond
    `observation` is refused at slot declaration."""
    from agents.model import extra_obs_keys as EOK

    row = EOK.ExtraObsKey(key="x_probe_key", attr="x_probe_attr", shape=(2,), flag="x_probe")
    monkeypatch.setattr(EOK, "EXTRA_OBS_KEYS", (row,))
    fe = types.SimpleNamespace(x_probe_attr=object())
    with pytest.raises(UnservableArchitecture, match="x_probe_key"):
        _refuse_extra_obs_keys(types.SimpleNamespace(features_extractor=fe))
    # and the ordinary (observation-only) extractor passes
    _refuse_extra_obs_keys(types.SimpleNamespace(features_extractor=types.SimpleNamespace(x_probe_attr=None)))


# ---------------------------------------------------------------- the declaration
@pytest.mark.parametrize("kw, match", [
    (dict(buckets=(1, 8)), "smallest bucket"),
    (dict(buckets=(8, 2)), "ascending"),
    (dict(backend="graph"), "CUDA only"),
    (dict(backend="aot"), "CUDA only"),
    (dict(lanes=2), "lanes"),
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


# ---------------------------------------------------------------- a load is tied to its source (P10 F3 / F8)
def test_a_load_whose_copy_MISSED_is_refused_and_poisons(policies, monkeypatch):
    """P10 F3 (`gen3_slot_copy_verify_v1`). The parity gate's eager reference is the slot's OWN replica
    (views into the same stacked storage), so with `copy_in` a no-op, `load(slot, B)` used to pass
    parity at |dV| 0 and record "policy-B" while serving A. The bit-exact copy check refuses it."""
    a, b = policies
    svc = _service(a, n_slots=1, buckets=(2,))
    group = svc.groups[0]
    monkeypatch.setattr(group, "copy_in", lambda slot, sd: None)
    with pytest.raises(CopyParityFailure, match="NOT bit-exact"):
        svc.load(0, b, "policy-B")
    assert svc.state == "POISONED" and svc.model_id(0) == "<template>"


def test_the_parity_gate_ALONE_is_blind_to_a_missed_copy(policies, monkeypatch):
    """The precondition of the test above: with the copy check disabled too, the no-op load PASSES the
    parity gate — so the copy check is what bites, not parity."""
    a, b = policies
    svc = _service(a, n_slots=1, buckets=(2,))
    group = svc.groups[0]
    monkeypatch.setattr(group, "copy_in", lambda slot, sd: None)
    monkeypatch.setattr(group, "verify_copy", lambda slot, sd, where: None)
    report = svc.load(0, b, "policy-B")
    assert report.value_max == 0.0 and svc.model_id(0) == "policy-B"   # "B", serving A


def test_a_load_that_copied_the_WRONG_source_is_refused(policies, monkeypatch):
    a, b = policies
    svc = _service(a, n_slots=2, buckets=(2,))
    group = svc.groups[0]
    real_copy = group.copy_in
    a_sd = group.check_loadable(a)
    monkeypatch.setattr(group, "copy_in", lambda slot, sd: real_copy(slot, a_sd))
    with pytest.raises(CopyParityFailure, match="slot 1"):
        svc.load(1, b, "policy-B")
    assert svc.state == "POISONED"


def test_a_load_is_gated_at_EVERY_bucket_the_slot_serves(policies):
    """P10 F8 (`gen3_load_gates_every_bucket_v1`): a miscompile can be weight-dependent, so a load is
    judged at every bucket the slot serves (full + partial rows each), not only the smallest. Slot 0 is
    capped at bucket 2 (2 verdicts), slot 1 serves 2 and 8 (4). Revert to the smallest bucket ⇒ slot 1
    adds 2."""
    a, b = policies
    svc = _service(a, n_slots=2, buckets=(2, 8), slot_bucket_caps=(2, 8))
    for slot, want in ((0, 2), (1, 4)):
        before = svc.parity_paths.get("real", 0)
        svc.load(slot, b, f"policy-B@{slot}")
        assert svc.parity_paths.get("real", 0) - before == want, (slot, svc.parity_paths)
    assert svc.stats()["load_seconds"] > 0.0


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
    with torch_globals(num_threads=2):
        yield build_fresh_model(3)[0].policy.eval()


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
    assert any("[perturbed seed=20260929 scale=0.05]" in w for w in wheres), wheres
    slot_sd = svc.groups[0].policies[0].state_dict()
    assert all(torch.equal(before[k], slot_sd[k].cpu()) for k in before), \
        "the perturbed gate must leave the slot's weights bit-identical"
    rep = svc.load(0, fresh_policy, "fresh-again")
    assert "[perturbed" not in rep.where and rep.path == "real (vacuity waived)", \
        "load() must return the REAL weights' report first"


def test_a_miscompile_invisible_on_fresh_weights_is_CAUGHT_by_the_perturbed_gate(
        monkeypatch, fresh_policy):
    import agents.inference.service.engine as engine_mod
    monkeypatch.setattr(engine_mod, "decide", _temperature_bug(engine_mod.decide))
    with pytest.raises(ParityFailure, match=r"perturbed seed=\d+ scale="):
        _service(fresh_policy, n_slots=1, buckets=(2, 8))

# ---------------------------------------------------------------- staging (unit 4)
def test_a_submitted_row_is_copied_so_a_caller_buffer_rewrite_cannot_reach_it(policies):
    a, _ = policies
    svc = _service(a, n_slots=1, buckets=(2, 8))
    obs, mask = fixture_rows(svc.obs_dim, 5)
    buf_o, buf_m = obs.copy(), mask.copy()            # the caller's (e.g. env core's) buffer
    t = svc.submit(0, buf_o, buf_m)
    buf_o[:] = 0.0                                    # the core steps before the flush
    buf_m[:] = True
    svc.flush()
    _close(t.result(), _ref(a, obs, mask))


def test_host_results_equal_device_results_across_flushes_of_both_buffers(policies):
    a, b = policies
    svc = _service(a, n_slots=2, buckets=(2, 8))
    svc.load(1, b, "B")
    obs, mask = fixture_rows(svc.obs_dim, 12)
    for k in range(4):                                # both halves of the double buffer, twice
        n = 3 + 2 * k
        t0, t1 = svc.submit(0, obs[:n], mask[:n]), svc.submit(1, obs[k:k + 4], mask[k:k + 4])
        svc.flush()
        for t, pol, o, m in ((t0, a, obs[:n], mask[:n]), (t1, b, obs[k:k + 4], mask[k:k + 4])):
            d = t.result()
            lp, v, g = t.host()
            assert np.array_equal(lp, d.logp.numpy()) and np.array_equal(v, d.value.numpy())
            assert np.array_equal(g, d.greedy.numpy())
            _close(d, _ref(pol, o, m))


def test_a_slot_never_holds_the_ride_along_heads(policies):
    """gen3_slot_served_state_v1: the DETACHED ride-along heads are left out of every replica (never
    copied, not stacked) — 18.2 MiB per slot at the X26 surface — and a heads-carrying weight set still
    loads (its heads are not a slot's to hold); what a slot SERVES is unchanged."""
    from agents.inference.service.slots import served_state_dict
    from agents.model.ridealong_heads import RideAlongSpec, build_ridealong
    a, b = policies
    obs_dim = int(a.observation_space["observation"].shape[0])

    def with_heads(p):
        q = copy.deepcopy(p)
        q.ridealong = build_ridealong(q.features_extractor, obs_dim=obs_dim, spec=RideAlongSpec(rnd=True, adv=2))
        assert q.ridealong is not None
        return q
    ha, hb = with_heads(a), with_heads(b)
    heads_keys = [k for k in ha.state_dict() if k.startswith("ridealong.")]
    assert heads_keys and not any(k.startswith("ridealong.") for k in served_state_dict(ha))
    svc = _service(ha, n_slots=2, buckets=(2, 8))
    group = svc.groups[0]
    assert all(getattr(r, "ridealong", None) is None for r in group.policies)
    assert not any(k.startswith("ridealong.") for k in group.stacked)
    assert ha.ridealong is not None                      # the template itself is untouched
    plain = _service(a, n_slots=2, buckets=(2, 8)).groups[0]
    assert group.storage_bytes() == plain.storage_bytes()
    s1 = svc.slot("pool", 1)
    svc.load(s1, hb, "heads-B")                          # a heads-carrying load matches the slot
    obs, mask = fixture_rows(svc.obs_dim, 13)
    t0, t1 = svc.submit(svc.slot("pool", 0), obs, mask), svc.submit(s1, obs, mask)
    svc.flush()
    _close(t0.result(), _ref(a, obs, mask))
    _close(t1.result(), _ref(b, obs, mask))


def test_per_slot_bucket_caps_chunk_rows_beyond_a_slots_largest_bucket(policies):
    """gen3_slot_bucket_caps_v1: a capped slot captures / gates / serves only its buckets <= the cap, and
    a request larger than its largest bucket is CHUNKED (served, never refused); the uncapped slot keeps
    every bucket. A bad declaration is refused."""
    a, b = policies
    svc = _service(a, n_slots=2, buckets=(2, 8, 16), slot_bucket_caps=(16, 2))
    e = svc.engine
    assert e.slot_buckets == [(2, 8, 16), (2,)]
    gated = {(r.where.split("slot=")[1].split(" ")[0], r.where.split("bucket=")[1].split(" ")[0])
             for r in svc.startup_reports if "CONCURRENT" not in r.where}
    assert ("1", "8") not in gated and ("1", "16") not in gated and ("0", "16") in gated
    assert e.chunks(13, 1) == [(2, 2)] * 6 + [(2, 1)] and e.chunks(13, 0) == [(16, 13)]
    svc.load(svc.slot("pool", 1), b, "B")
    obs, mask = fixture_rows(svc.obs_dim, 13)
    t0, t1 = svc.submit(svc.slot("pool", 0), obs, mask), svc.submit(svc.slot("pool", 1), obs, mask)
    svc.flush()
    _close(t0.result(), _ref(a, obs, mask))
    _close(t1.result(), _ref(b, obs, mask))
    for bad in ((16,), (16, 1)):
        with pytest.raises(ValueError, match="slot_bucket_caps"):
            _service(a, n_slots=2, buckets=(2, 8, 16), slot_bucket_caps=bad)


def test_the_rust_env_declares_caps_on_every_slot_but_the_trainees():
    from agents.training.rust_rollout.build import RustEnvDecl
    d = RustEnvDecl(n_envs=256, opponent_bucket_cap=64)
    assert d.resolved_buckets == (8, 64, 256)
    assert d.slot_bucket_caps(4, [2]) == (64, 64, 256, 64)
    assert RustEnvDecl(n_envs=48, opponent_bucket_cap=64).resolved_buckets == (8, 48)
    assert RustEnvDecl(n_envs=256).opponent_bucket_cap == 64                 # the declared default
    assert RustEnvDecl(n_envs=256, opponent_bucket_cap=0).slot_bucket_caps(4, [2]) == ()   # 0 = uncapped
    assert RustEnvDecl(n_envs=256, opponent_bucket_cap=0).resolved_buckets == (8, 256)
    typed = RustEnvDecl(n_envs=256, buckets=(8, 16, 32, 256), opponent_bucket_cap=64)
    assert typed.resolved_buckets == (8, 16, 32, 256) and typed.slot_bucket_caps(2, [0]) == (256, 64)
