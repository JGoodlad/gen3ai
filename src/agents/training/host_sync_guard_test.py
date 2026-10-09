"""The learner update's HOST SYNCS are BOUNDED (T25 item 1, `gen3_batched_host_reads_v1`).

On CUDA every device->host read blocks the host until the GPU queue drains; the 2026-10-03 profile
counted ~764 per production update (`designs/research_state/measurements/bottleneck_profile_2026-10-03/`).
`instrumented_ppo/host_reads.py` defers every read nothing needs before the next optimizer step into that
step's ONE transfer. These tests count the syncs of one CPU update with `host_sync_trace.HostSyncTrace`
(the CPU twin of `torch.cuda.set_sync_debug_mode`) and hold them to the new bound, so a per-micro-batch
read cannot creep back unnoticed: each FAILS on a revert of the batching (measured: the golden plain
update read 36 times before, 6 after; the diagnostics update 102 before, 13 after).

The bound is a function of the update's SHAPE, never a tolerance: a plain update makes one transfer per
optimizer step plus two once-per-update reads (the rank probe's spectra, `--rank-tripwire`'s every-update
input, and the episode-start calibration); a diagnostics update adds the micro-batches read at once while
the grad-balance probe and the per-term noise tagger need their values, and one batched read per probe.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
import torch as th

from agents.training import learner_golden as L
from agents.training.host_sync_trace import HostSyncTrace, optimizer_host_tensors
from agents.training.instrumented_ppo.host_reads import HostReadQueue

#: The per-micro-batch sites the batching retired: a sync attributed to one of these is a regression.
_RETIRED_SITES = ("instrumented_ppo/micro_step.py", "instrumented_ppo/calibration.py",
                  "instrumented_ppo/learner_gates.py", "instrumented_ppo/noise_scale.py",
                  "instrumented_ppo/noise_scale_terms.py", "training/grad_balance.py")


def _learner(plain: bool):
    model = L.build_learner()
    L.load_buffer_into(model)
    if plain:                                   # not a diagnostics update (`diagnostics_cadence`)
        model.diagnostics_every = 10
        model._diagnostics_ran_in_process = True
        model.num_timesteps = model.n_steps * model.n_envs * 3
    return model


def _traced_update(model) -> HostSyncTrace:
    with L._one_thread():
        np.random.seed(L.UPDATE_SEED)
        th.manual_seed(L.UPDATE_SEED)
        tr = HostSyncTrace(host=optimizer_host_tensors(model.policy.optimizer))
        with tr:
            model.train()
    return tr


def _optimizer_steps(model) -> int:
    rows = model.n_steps * model.n_envs
    micros = math.ceil(rows / model.batch_size)
    return model.n_epochs * math.ceil(micros / model.grad_accum_steps)


@pytest.mark.parametrize("batch_size", [16, 8])
def test_a_plain_update_reads_once_per_optimizer_step(batch_size):
    """One transfer per optimizer step + the two once-per-update reads — and NOT one per micro-batch:
    halving the micro-batch doubles the micro-batches and moves the count only by the extra steps."""
    model = _learner(plain=True)
    model.batch_size = batch_size
    tr = _traced_update(model)
    bound = _optimizer_steps(model) + 2
    assert tr.total <= bound, f"{tr.total} host syncs > {bound}:\n{tr.table()}"
    retired = [e for e in tr.events if e.site.startswith(_RETIRED_SITES)]
    assert not retired, f"a retired per-micro-batch read is back:\n{tr.table()}"


def test_a_diagnostics_update_batches_every_probe_read():
    model = _learner(plain=False)
    tr = _traced_update(model)
    # immediate micro-batches: the per-term noise tagger reads epoch 0's first `accum` micro-batches
    bound = _optimizer_steps(model) + model.grad_accum_steps + 2 + 4
    assert tr.total <= bound, f"{tr.total} host syncs > {bound}:\n{tr.table()}"
    assert not [e for e in tr.events if e.site.startswith(_RETIRED_SITES)], tr.table()
    per_probe = [e for e in tr.events if e.site.startswith("training/batched_reads.py")]
    assert len(per_probe) <= 4, tr.table()      # grad balance, edge, cell, per-term noise: ONE each


def test_the_tracer_sees_a_device_read_and_not_a_host_one():
    """The instrument's own teeth: `.item()` of a learner-device tensor is a sync; of a `.cpu()` copy,
    or of a factory tensor made without a device, it is not."""
    dev = th.ones(3, device="cpu").sum()               # untracked => device-resident, as a param is
    with HostSyncTrace() as tr:
        dev.item()
        host = dev.cpu()
        host.item()
        th.zeros(()).item()
        th.ones(4)[th.ones(4, dtype=th.bool)]          # a host bool-mask index: no sync
        (dev * th.ones(4))[th.ones(4) > 0]             # a device bool-mask index: a sync
    ops = [e.op for e in tr.events]
    assert ops == ["item", "cpu", "__getitem__[bool]"], ops
    assert tr.host_reads == 3


def test_the_queue_makes_one_transfer_and_applies_in_push_order():
    q = HostReadQueue()
    seen = []
    q.push(th.tensor([1.0, 2.0]), lambda a: seen.append(("a", a.tolist())))
    q.push(th.tensor([[3.0]]), lambda a: seen.append(("b", a.tolist())))
    with HostSyncTrace() as tr:
        extra = q.drain([th.tensor(7.0)])
    assert extra is not None and extra.tolist() == [7.0]
    assert seen == [("a", [1.0, 2.0]), ("b", [3.0])]
    assert tr.total == 1 and q.drains == 1 and q.reads == 2 and len(q) == 0
    assert q.drain() is None and q.drains == 1          # nothing pending: no transfer at all
    with pytest.raises(TypeError, match="float32"):
        q.push(th.tensor([1.0], dtype=th.float64), lambda a: None)


def test_a_raising_callback_stops_the_drain():
    q = HostReadQueue()
    seen = []

    def boom(a):
        raise FloatingPointError("K9(c)")
    q.push(th.tensor([1.0]), boom)
    q.push(th.tensor([2.0]), lambda a: seen.append(a.tolist()))
    with pytest.raises(FloatingPointError):
        q.drain()
    assert seen == [] and len(q) == 0


_UPDATE_PATH = ("agents/training/instrumented_ppo", "agents/training/grad_balance.py",
                "agents/training/rank_metrics.py", "agents/training/batched_reads.py")


def test_the_update_path_makes_no_cuda_only_sync():
    """What the CPU tracer cannot see: an explicit CUDA synchronize, a stream / event wait, or a
    blocking ``.cuda()`` in the learner's update modules."""
    src = Path(__file__).resolve().parents[2]
    files = []
    for rel in _UPDATE_PATH:
        p = src / rel
        files += sorted(p.glob("*.py")) if p.is_dir() else [p]
    bad = []
    for f in files:
        if f.name.endswith("_test.py"):
            continue
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                    node.func.attr in ("synchronize", "wait_stream", "wait_event", "cuda"):
                bad.append(f"{f.name}:{node.lineno} .{node.func.attr}()")
    assert not bad, bad
