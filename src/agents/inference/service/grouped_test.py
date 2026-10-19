"""X45 — T2's GROUPED FORWARD (gen3_t2_grouped_forward_v1, ``ServiceSpec.grouped_slots``, ``--t2-grouped-opponents``).

ONE vmapped forward over a slot group's stacked weights serves every grouped slot of a flush (``grouped.py``). These
tests pin, on CPU (eager, one lane), with DISTINCT weights in every slot (a grouped forward that read one slot's
weights for all, or the template's, would differ):

* PARITY: every grouped slot's decision equals the PER-SLOT service's on the same rows within a DECLARED
  deterministic tolerance (``TOL``; measured ~1e-6 — fp32 reassociation of a batched-weight matmul) and the
  greedy action is identical on every decisive row (top-2 margin above the near-tie band), on uneven row counts
  that need padding, an idle slot and a slot needing two ROUNDS;
* the grouped path really ran (its rounds counter, and no per-slot batch for a grouped slot) — reverting the
  feature fails these tests;
* the service's own gate has teeth on the grouped path: a grouped forward that serves the WRONG slot's weights
  is REFUSED at startup;
* a ``load`` into a grouped slot is served on the next flush (the weights are views into the stacked storage);
* no forward state (a slot-batched stash) escapes the grouped body;
* the declaration is validated, OFF is unchanged, and the trainer's flag reaches the decl, the spec and the stamp.

What CPU cannot show: the CUDA-graph backend and the GPU wall clock — the GPU window's read
(``grouped_benchmark.py``; ``designs/endstate/design_compute_efficiency.md`` §1.6).
"""
from __future__ import annotations

import argparse
import types

import numpy as np
import pytest
import torch

from agents.inference.service import InferenceService, ParityFailure, ServiceSpec, SlotGroupSpec, policy_reference
from agents.inference.service.fixtures import perturbed_fresh_policy as _perturbed_policy
from agents.inference.service.grouped import GroupedForwardError, grouped_ranges
from agents.inference.service.parity import fixture_rows
from agents.model.parity_probe import NEAR_TIE_BAND
from utils.torch_state_guard import torch_globals

#: The declared parity tolerance of a grouped decision against the per-slot one (legal log-probs and V). Measured
#: on CPU 2026-10-19: max|dlogp| <= 9.6e-7, max|dV| <= 4.8e-7 (fp32 reassociation). 1e-5 is 10x that, and 100x
#: inside the service's own gate bars (1e-3 / 1e-4).
TOL = 1e-5

#: Rows per slot of the parity flush (buckets (2, 8)): slot 0 one row (padded to 2), slot 1 thirteen (two ROUNDS:
#: 8 + 5), slot 2 none (an idle grouped slot), slot 3 five (the per-slot trainee-like slot).
ROWS = (1, 13, 0, 5)


@pytest.fixture(scope="module")
def policies():
    with torch_globals(num_threads=1):          # restored at module end (the global-state guard)
        yield tuple(_perturbed_policy(k) for k in range(3))


def _service(policies, *, grouped=(0, 1, 2), n_slots=4):
    spec = ServiceSpec(groups=(SlotGroupSpec("pool", n_slots, policies[0]),), device="cpu", backend="eager",
                       buckets=(2, 8), grouped_slots=grouped)
    svc = InferenceService(spec).startup()
    for s in range(1, n_slots):
        svc.load(s, policies[s % 3 or 1], f"p{s % 3 or 1}")         # 0, 1, 2 distinct (grouped); 3 holds p1
    return svc


@pytest.fixture(scope="module")
def pair(policies):
    return _service(policies, grouped=()), _service(policies)


def _serve(svc, rows=ROWS):
    obs, mask = fixture_rows(svc.obs_dim, sum(rows))
    out, a = {}, 0
    tickets = {}
    for s, n in enumerate(rows):
        if n:
            tickets[s] = (svc.submit(s, obs[a:a + n], mask[a:a + n]), a, n)
        a += n
    svc.flush()
    for s, (t, a0, n) in tickets.items():
        d = t.result()
        out[s] = (d.logp.clone(), d.value.clone(), d.greedy.clone(), obs[a0:a0 + n], mask[a0:a0 + n])
    return out


# ---------------------------------------------------------------- parity
def test_grouped_decisions_equal_the_per_slot_service_within_the_declared_tolerance(pair):
    off, on = pair
    r0 = on.engine.grouped_rounds
    d_off, d_on = _serve(off), _serve(on)
    assert on.engine.grouped_rounds - r0 == 2                     # slot 1's 13 rows: two rounds of the range
    assert set(d_off) == set(d_on) == {0, 1, 3}
    n_decisive = 0
    for s in d_off:
        lp0, v0, g0, _o, m = d_off[s]
        lp1, v1, g1, _, _ = d_on[s]
        mt = torch.as_tensor(m)
        assert torch.equal(torch.isneginf(lp1), ~mt) and bool(torch.isfinite(lp1[mt]).all())   # the mask contract
        assert float((lp1 - lp0)[mt].abs().max()) <= TOL, s
        assert float((v1 - v0).abs().max()) <= TOL, s
        # greedy: identical on every DECISIVE row (rows within the near-tie band are excluded, never tolerated)
        top2 = lp0.topk(2, dim=-1).values
        decisive = torch.where(torch.isfinite(top2[:, 1]), top2[:, 0] - top2[:, 1],
                               torch.full_like(top2[:, 0], float("inf"))) > NEAR_TIE_BAND
        n_decisive += int(decisive.sum())
        assert torch.equal(g1[decisive], g0[decisive]), s
    assert n_decisive >= 10                                       # the greedy check bit on most of the 19 rows
    # NON-VACUOUS: slot 0's weights on slot 1's rows decide differently (so a weight mix-up cannot pass)
    o, m = torch.as_tensor(d_off[1][3]), torch.as_tensor(d_off[1][4])
    other = policy_reference(on.groups[0].policies[0], o, m)[0]
    assert float((other - d_off[1][0])[m].abs().max()) > 1e-3


def test_the_grouped_path_serves_the_grouped_slots_and_only_them(pair):
    _off, on = pair
    seen = []
    real = on.engine._batch

    def rec(gi, i, lane, b, a, n):
        seen.append(i)
        return real(gi, i, lane, b, a, n)

    on.engine._batch = rec
    try:
        _serve(on)
    finally:
        on.engine._batch = real
    assert seen == [3]                                            # the grouped slots never run a per-slot batch
    st = on.stats()
    assert st["grouped_slots"] == [0, 1, 2] and st["grouped_rounds"] > 0
    assert on.engine.ranges == [(0, 0, 3, [0, 1, 2])]
    assert any("grouped" in r.where for r in on.startup_reports)  # the concurrent gate judged the range
    assert "grouped_slots" not in pair[0].stats()                # OFF: the stats are unchanged


def test_a_load_into_a_grouped_slot_is_served_on_the_next_flush(pair, policies):
    off, on = pair
    rows = (0, 0, 6, 0)
    before = _serve(on, rows)[2][0]
    try:
        for s in (on, off):
            s.load(2, policies[0], "p0 into slot 2")              # an in-place copy into the stacked storage
        d_on, d_off = _serve(on, rows), _serve(off, rows)
        m = torch.as_tensor(d_off[2][4])
        assert float((d_on[2][0] - d_off[2][0])[m].abs().max()) <= TOL
        assert float((d_on[2][0] - before)[m].abs().max()) > 1e-3    # the new weights, not the old ones
    finally:
        for s in (on, off):
            s.load(2, policies[2], "p2 back")


def test_the_gate_refuses_a_grouped_forward_that_serves_the_wrong_weights(policies, monkeypatch):
    """The service's own parity gate judges each grouped slot against ITS eager reference: a grouped forward that
    reads slot 0's weights for every slot must be refused (a load gates every bucket of the slot)."""
    from agents.inference.service import grouped as G

    real = G.GroupedForward._weights

    def wrong(self, group):
        batched, consts = real(self, group)
        return {k: v[:1].expand(v.shape[0], *v.shape[1:]) for k, v in batched.items()}, consts

    monkeypatch.setattr(G.GroupedForward, "_weights", wrong)
    svc = InferenceService(ServiceSpec(groups=(SlotGroupSpec("pool", 3, policies[0]),), device="cpu",
                                       backend="eager", buckets=(2, 8), grouped_slots=(0, 1, 2))).startup()
    with pytest.raises(ParityFailure):
        svc.load(1, policies[1], "p1")


def test_no_forward_state_escapes_the_grouped_body(pair):
    _off, on = pair
    _serve(on)
    gf = on.engine.gfwd[0]
    assert gf._rest, "the build found the forward's stashes"
    for owner, name, resting in gf._rest:
        assert getattr(owner, name) is resting                    # reset, not a slot-batched leftover
    for owner in (gf.base.policy, *gf.base.policy.modules()):
        for k, v in vars(owner).items():
            if torch.is_tensor(v):
                assert not torch._C._functorch.is_batchedtensor(v), (type(owner).__name__, k)


# ---------------------------------------------------------------- the declaration
def test_grouped_ranges_split_per_group_and_refuse_a_gap():
    slots = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    assert grouped_ranges(slots, (0, 1, 3, 4)) == [(0, 0, 2, [0, 1]), (1, 0, 2, [3, 4])]
    with pytest.raises(GroupedForwardError, match="contiguous"):
        grouped_ranges(slots, (0, 2))
    with pytest.raises(GroupedForwardError, match="not declared"):
        grouped_ranges(slots, (7,))


@pytest.mark.parametrize("kw, match", [
    (dict(grouped_slots=(0, 0)), "grouped_slots"),
    (dict(grouped_slots=(5,)), "grouped_slots"),
    (dict(grouped_slots=(0, 1), lead_slots=(1,)), "lead slot"),
    (dict(grouped_slots=(0, 1), slot_bucket_caps=(2, 8, 8)), "one bucket cap"),
    (dict(grouped_slots=(0, 1), device="cuda", backend="aot"), "eager' or 'graph"),
])
def test_the_grouped_declaration_is_validated(policies, kw, match):
    spec = dict(groups=(SlotGroupSpec("pool", 3, policies[0]),), device="cpu", backend="eager", buckets=(2, 8))
    spec.update(kw)
    with pytest.raises(ValueError, match=match):
        ServiceSpec(**spec).validate()
    ServiceSpec(**dict(spec, grouped_slots=(0, 1), lead_slots=(), slot_bucket_caps=(),
                       backend="eager", device="cpu")).validate()


def test_the_decl_groups_each_groups_opponent_slots_only():
    from agents.training.rust_rollout.build import RustEnvDecl

    off = RustEnvDecl(n_envs=256)
    on = RustEnvDecl(n_envs=256, grouped_opponents=True)
    assert off.grouped_opponents is False and off.grouped_slots([25], 24) == ()      # the declared default
    assert on.grouped_slots([25], 24) == tuple(range(24))                            # the pool, never the trainee
    # a stable of another architecture (its own one-slot group) stays per-slot; eval's extra slots are never grouped
    assert on.grouped_slots([24, 1, 1, 3], 25) == tuple(range(24))
    assert on.grouped_slots([2, 3], 4) == (0, 1, 2, 3)


def test_the_flag_reaches_the_decl_and_the_stamp():
    from agents.training.rust_rollout.build import RustEnvDecl
    from main.train.parser.env_core import add_env_core_flags
    from main.train.rust_env_setup import env_core_stamp, resolve_env_core_args

    p = argparse.ArgumentParser()
    add_env_core_flags(p)
    off, on = p.parse_args([]), p.parse_args(["--t2-grouped-opponents"])
    resolve_env_core_args(off)
    resolve_env_core_args(on)
    assert off.t2_grouped_opponents is False and on.t2_grouped_opponents is True
    assert p.parse_args(["--no-t2-grouped-opponents"]).t2_grouped_opponents is False
    col = types.SimpleNamespace(core=types.SimpleNamespace(stamp="s"), cfg=types.SimpleNamespace(
        trigger=types.SimpleNamespace(target=0, describe=lambda: "t")))
    t2_off = env_core_stamp(RustEnvDecl(n_envs=256), col)["t2"]
    t2_on = env_core_stamp(RustEnvDecl(n_envs=256, grouped_opponents=True), col)["t2"]
    assert "grouped_opponents" not in t2_off and t2_on.pop("grouped_opponents") is True and t2_on == t2_off


def test_the_batch_fresh_mode_adds_an_exact_zero():
    """``x + (-0.0)`` is ``x`` bit for bit — +0.0, -0.0, inf, -inf, subnormals and NaN included — the identity the
    factory mode relies on."""
    x = torch.tensor([0.0, -0.0, 1.5, -2.25, float("inf"), float("-inf"), 1e-45, float("nan")])
    y = x + torch.tensor(-0.0)
    assert torch.equal(x.view(torch.int32), y.view(torch.int32))
    assert np.array_equal(np.signbit(x.numpy()), np.signbit(y.numpy()))
