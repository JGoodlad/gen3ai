"""T25 perf lever 4 — T2's LEAD SLOTS (gen3_t2_lead_lanes_v1, ``ServiceSpec.lead_slots``, ``--t2-lead-trainee``).

The lever changes only WHEN a slot's replays are launched (lead slots first) and WHICH lane runs them (each lead
slot alone on its own lane). These tests pin, on CPU (eager, one lane):

* OFF (no lead slots) is the old plan exactly: slot ``s`` on lane ``s % lanes``, runs launched in slot order;
* ON moves the lead slots' runs to the front and gives each a dedicated lane, the rest round-robin over the others;
* ON serves BIT-IDENTICAL decisions to OFF (same offsets, same logp / V / greedy) — the packing is unchanged;
* the declaration is validated, and the trainer's flag reaches the decl, the spec and the env-core stamp.

What CPU cannot show (one lane, no streams): the wall-clock effect. That is the GPU window's paired read.
"""
from __future__ import annotations

import argparse
import types

import pytest
import torch

from agents.inference.service import InferenceService, ServiceSpec, SlotGroupSpec
from agents.inference.service.engine import Engine, launch_order, lane_plan
from agents.inference.service.fixtures import perturbed_fresh_policy as _perturbed_policy
from agents.inference.service.parity import fixture_rows
from utils.torch_state_guard import torch_globals


@pytest.fixture(scope="module")
def policies():
    with torch_globals(num_threads=2):          # restored at module end (the global-state guard)
        yield _perturbed_policy(0), _perturbed_policy(1)


# ---------------------------------------------------------------- the pure plan
def test_lane_plan_OFF_is_slot_mod_lanes():
    for n_slots, lanes in ((31, 8), (5, 3), (4, 1), (2, 8)):
        assert lane_plan(n_slots, lanes) == [s % lanes for s in range(n_slots)]
    assert lane_plan(4, 1, (2,)) == [0, 0, 0, 0]                    # one lane: nothing to dedicate


def test_lane_plan_ON_gives_each_lead_slot_its_own_lane():
    # production's layout: 24 pool slots, the trainee at 24, eval's six slots after it, 8 lanes
    plan = lane_plan(31, 8, (24,))
    assert plan[24] == 0 and [s for s in range(31) if plan[s] == 0] == [24]
    rest = [s for s in range(31) if s != 24]
    assert [plan[s] for s in rest] == [1 + r % 7 for r in range(len(rest))]
    two = lane_plan(6, 4, (5, 1))
    assert two[5] == 0 and two[1] == 1 and [two[s] for s in (0, 2, 3, 4)] == [2, 3, 2, 3]
    with pytest.raises(ValueError, match="more than"):
        lane_plan(6, 2, (0, 1))


def test_launch_order_OFF_is_slot_order_and_ON_puts_the_lead_slots_first():
    runs = [(0, 0, 0, 0), (1, 0, 1, 5), (3, 0, 3, 9), (4, 0, 4, 20)]
    ends = [5, 9, 20, 30]
    assert launch_order(runs, ends) == list(zip(runs, ends))
    assert [r[0][0] for r in launch_order(runs, ends, (3,))] == [3, 0, 1, 4]
    assert [r[0][0] for r in launch_order(runs, ends, (4, 1))] == [4, 1, 0, 3]
    assert [r for r in launch_order(runs, ends, (3,)) if r[0][0] == 3] == [((3, 0, 3, 9), 20)]   # its own end
    assert launch_order(runs, ends, (7,)) == list(zip(runs, ends))   # a lead slot with no rows: order kept


# ---------------------------------------------------------------- the service
def _service(template, *, lead=(), n_slots=3):
    spec = ServiceSpec(groups=(SlotGroupSpec("pool", n_slots, template),), device="cpu", backend="eager",
                       buckets=(2, 8), lead_slots=lead)
    return InferenceService(spec).startup()


def _serve(svc, b, rows):
    """Load ``b`` into slot 1, submit ``rows[s]`` rows to each slot, flush; the launch order of the batches."""
    svc.load(1, b, "B")
    obs, mask = fixture_rows(svc.obs_dim, sum(rows))
    order = []
    real = svc.engine._batch

    def rec(gi, i, lane, bk, a, n):
        order.append((i, lane, bk, a, n))
        return real(gi, i, lane, bk, a, n)

    svc.engine._batch = rec
    tickets, r = [], 0
    for s, n in enumerate(rows):
        tickets.append(svc.submit(s, obs[r:r + n], mask[r:r + n]))
        r += n
    svc.flush()
    return order, tickets


def test_ON_launches_the_lead_slot_first_and_serves_bit_identical_decisions(policies):
    a, b = policies
    rows = (3, 5, 11)                           # slot 2 (the lead) needs two batches: 8 + 3 -> bucket 8
    off_order, off_t = _serve(_service(a), b, rows)
    on_svc = _service(a, lead=(2,))
    on_order, on_t = _serve(on_svc, b, rows)
    assert [o[0] for o in off_order] == [0, 1, 2, 2]
    assert [o[0] for o in on_order] == [2, 2, 0, 1]
    # the same batches (slot, bucket, arena offset, rows), only launched in another order
    assert sorted(on_order) == sorted(off_order)
    for t_off, t_on in zip(off_t, on_t):
        assert t_off._off == t_on._off
        d0, d1 = t_off.result(), t_on.result()
        assert torch.equal(d0.logp, d1.logp) and torch.equal(d0.value, d1.value) and torch.equal(d0.greedy, d1.greedy)
    st = on_svc.stats()
    assert st["lead_slots"] == [2]
    assert "lead_slots" not in _service(a).stats()                    # OFF: the stats are unchanged


def test_the_engine_reads_the_lane_plan(policies):
    a, _ = policies
    svc = _service(a, lead=(2,))
    e = svc.engine
    assert e.lead == (2,) and [e.lane_of(s) for s in range(3)] == [0, 0, 0]      # CPU: one lane
    # the same plan on a CUDA-sized lane count (lane_of reads the engine's plan, never s % lanes)
    e._lane = lane_plan(3, 3, (2,))
    assert [e.lane_of(s) for s in range(3)] == [1, 2, 0]
    assert isinstance(e, Engine)


@pytest.mark.parametrize("kw, match", [
    (dict(lead_slots=(0, 0)), "lead_slots"),
    (dict(lead_slots=(3,)), "lead_slots"),
    (dict(lead_slots=(-1,)), "lead_slots"),
    (dict(device="cuda", lanes=2, lead_slots=(0, 1)), "MORE lanes"),
])
def test_the_lead_declaration_is_validated(policies, kw, match):
    spec = dict(groups=(SlotGroupSpec("pool", 3, policies[0]),), device="cpu", backend="eager", buckets=(2, 8))
    spec.update(kw)
    with pytest.raises(ValueError, match=match):
        ServiceSpec(**spec).validate()
    ok = dict(spec, lead_slots=(1,), lanes=1 if spec["device"] == "cpu" else 3)
    ServiceSpec(**ok).validate()


# ---------------------------------------------------------------- the trainer's flag
def test_the_flag_reaches_the_decl_the_spec_slots_and_the_stamp():
    from agents.training.rust_rollout.build import RustEnvDecl
    from main.train.parser.env_core import add_env_core_flags
    from main.train.rust_env_setup import env_core_stamp, resolve_env_core_args

    p = argparse.ArgumentParser()
    add_env_core_flags(p)
    off, on = p.parse_args([]), p.parse_args(["--t2-lead-trainee"])
    resolve_env_core_args(off)
    resolve_env_core_args(on)
    assert off.t2_lead_trainee is False and on.t2_lead_trainee is True
    assert p.parse_args(["--no-t2-lead-trainee"]).t2_lead_trainee is False
    assert RustEnvDecl(n_envs=256).lead_trainee is False                          # the declared default
    assert RustEnvDecl(n_envs=256).lead_slots([24]) == ()
    assert RustEnvDecl(n_envs=256, lead_trainee=True).lead_slots([24]) == (24,)
    col = types.SimpleNamespace(core=types.SimpleNamespace(stamp="s"), cfg=types.SimpleNamespace(
        trigger=types.SimpleNamespace(target=0, describe=lambda: "t")))
    t2_off = env_core_stamp(RustEnvDecl(n_envs=256), col)["t2"]
    t2_on = env_core_stamp(RustEnvDecl(n_envs=256, lead_trainee=True), col)["t2"]
    assert "lead_trainee" not in t2_off and t2_on.pop("lead_trainee") is True and t2_on == t2_off
