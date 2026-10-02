"""K8 — the device-resident micro-batch (`instrumented_ppo/device_batches.py`, gen3_device_batches_v1):
the gathered micro-batches are BIT-IDENTICAL to the host path's, drawn from the SAME permutation (the
global numpy RNG is consumed identically), across epochs and with fork rows; the device copy is made
once per update and dropped after it; and a whole update through it equals the host path bitwise."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import agents.training.instrumented_ppo.device_batches as DB


def _golden_buffer():
    from agents.training import learner_golden as LG
    model = LG.build_learner()
    LG.load_buffer_into(model)
    return model


def _batches(buffer, batch_size, epochs, seed):
    np.random.seed(seed)
    out = []
    for _ in range(epochs):
        out.append([s for s in buffer.get(batch_size)])
    return out, np.random.get_state()[1][:4].tolist()


def _same(a, b):
    for x, y in zip(a, b):
        for xa, ya in zip(x, y):
            if isinstance(xa, dict):
                assert set(xa) == set(ya)
                for k in xa:
                    assert xa[k].dtype == ya[k].dtype and xa[k].shape == ya[k].shape, k
                    assert torch.equal(xa[k].cpu(), ya[k].cpu()), k
            else:
                assert xa.dtype == ya.dtype and xa.shape == ya.shape
                assert torch.equal(xa.cpu(), ya.cpu())


def test_device_batches_are_bit_identical_to_the_host_path_and_consume_the_same_rng():
    model = _golden_buffer()
    buf = model.rollout_buffer
    host, rng_host = _batches(buf, 16, 3, seed=7)
    with DB.device_samples(buf, force=True) as g:
        assert g is not None
        dev, rng_dev = _batches(buf, 16, 3, seed=7)
        assert g.copies == 1                                  # ONE copy per update, not per epoch
    assert rng_host == rng_dev
    for eh, ed in zip(host, dev):
        assert len(eh) == len(ed)
        _same(eh, ed)
    assert "_get_samples" not in buf.__dict__                 # uninstalled: the host path again


def test_install_replaces_a_stale_gather_and_a_cpu_buffer_keeps_the_host_path():
    model = _golden_buffer()
    buf = model.rollout_buffer
    assert DB.install(buf) is None and "_get_samples" not in buf.__dict__   # CPU, not forced
    g1 = DB.install(buf, force=True, mode="resident")
    list(buf.get(16))
    g2 = DB.install(buf, force=True, mode="resident")                          # a raised update never uninstalled g1
    assert g2 is not g1 and g1.obs is None and buf._get_samples is g2
    DB.uninstall(buf)
    assert "_get_samples" not in buf.__dict__


def test_a_whole_update_through_device_batches_equals_the_host_path(monkeypatch):
    """The learner golden's update, the gather forced on (CPU): post-update parameters bit-equal to
    the host path's — the batches are the same tensors, so the program is the same."""
    from agents.training.instrumented_ppo import ppo as ppo_mod
    real_install = DB.install

    def one(force):
        calls = []

        def forced(buffer, **_k):
            calls.append(1)
            return real_install(buffer, force=force, mode="resident")
        monkeypatch.setattr(ppo_mod, "_devb_install", forced)   # what `ppo.train` calls
        model = _golden_buffer()
        with torch.random.fork_rng():
            np.random.seed(0)
            torch.manual_seed(0)
            model.train()
        assert calls == [1]
        assert "_get_samples" not in model.rollout_buffer.__dict__
        return {n: p.detach().clone() for n, p in model.policy.named_parameters()}

    a, b = one(False), one(True)
    assert set(a) == set(b)
    for n in a:
        assert torch.equal(a[n], b[n]), n


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_on_cuda_the_device_batches_equal_the_host_batches():
    model = _golden_buffer()
    buf = model.rollout_buffer
    buf.device = torch.device("cuda")
    host, _ = _batches(buf, 16, 2, seed=3)
    for mode in ("resident", "staged"):          # staged: pinned host gather + compute-stream copy
        with DB.device_samples(buf, mode=mode) as g:
            assert g is not None and g.device.type == "cuda"
            dev, _ = _batches(buf, 16, 2, seed=3)
            assert all(x.observations["observation"].is_cuda for e in dev for x in e)
        for eh, ed in zip(host, dev):
            _same(eh, ed)


_STRANDING_PROBE = r"""
import json, sys
import numpy as np, torch
import agents.training.instrumented_ppo.device_batches as DB
from agents.training import learner_golden as LG
model = LG.build_learner(); LG.load_buffer_into(model)
buf = model.rollout_buffer; buf.device = torch.device("cuda")
rows = []
for u in range(int(sys.argv[1])):
    with DB.device_samples(buf, mode="staged"):
        np.random.seed(u)
        for _ in range(2):
            for x in buf.get(16):
                (x.observations["observation"].float().sum() + x.advantages.sum()).item()
    torch.cuda.synchronize()
    streams = len({sg.get("stream", 0) for sg in torch.cuda.memory_snapshot()})   # self-contained: runs on old code
    rows.append([torch.cuda.memory_reserved(), streams])
print(json.dumps(rows))
"""


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_on_cuda_staged_updates_strand_no_cache_on_a_new_stream():
    """gen3_staged_compute_stream_v1: forty staged "updates" under the launcher's allocator mode
    (`expandable_segments:True`, a fresh process) hold ONE stream's cache and a flat reserved total.
    The code this replaced built a side stream per update: +1 stream with cache and +its in-flight
    micro-batches of reserved every update until torch's 32-stream pool wrapped (sizing arm B)."""
    import json
    import os
    import subprocess
    import sys

    from utils.paths import src_root
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
           "PYTHONPATH": str(src_root()) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    out = subprocess.run([sys.executable, "-c", _STRANDING_PROBE, "40"], env=env, capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    rows = json.loads(out.stdout.strip().splitlines()[-1])
    reserved = [r[0] for r in rows[1:]]
    streams = [r[1] for r in rows[1:]]
    assert len(set(streams)) == 1, streams                   # no stream acquired cache after update 1
    assert max(reserved) == min(reserved), reserved          # reserved flat after update 1


def test_the_gather_knows_its_device_copys_size():
    """gen3_cuda_ledger_v1: `train()` records it as `lifecycle/device_batch_mib`."""
    model = _golden_buffer()
    buf = model.rollout_buffer
    with DB.device_samples(buf, force=True) as g:
        list(buf.get(16))
        want = sum(int(v.reshape(-1, *v.shape[2:]).nbytes) for v in buf.observations.values())
        assert g.nbytes >= want > 0


# ---- gen3_device_batch_mode_v1: the STAGED mode (a prefetch thread, ~2 micro-batches on the card)
def test_staged_batches_are_bit_identical_to_the_host_path_and_consume_the_same_rng():
    model = _golden_buffer()
    buf = model.rollout_buffer
    host, rng_host = _batches(buf, 16, 3, seed=11)
    with DB.device_samples(buf, force=True, mode="staged") as g:
        assert isinstance(g, DB._StagedGather)
        staged, rng_staged = _batches(buf, 16, 3, seed=11)
        assert g.copies == sum(len(e) for e in staged)       # one staged gather per micro-batch
    assert rng_host == rng_staged
    for eh, es in zip(host, staged):
        assert len(eh) == len(es)
        _same(eh, es)
    # uninstalled: the class's own get / _get_samples again, the stager's thread gone
    assert "get" not in buf.__dict__ and "_get_samples" not in buf.__dict__
    assert "_devb_staged" not in buf.__dict__ and g.pool._shutdown


def test_staged_survives_an_early_break_and_host_mode_installs_nothing():
    model = _golden_buffer()
    buf = model.rollout_buffer
    with DB.device_samples(buf, force=True, mode="staged"):
        it = buf.get(16)
        next(it)                                             # a KL early stop breaks out mid-epoch
        it.close()
    assert "get" not in buf.__dict__
    assert DB.install(buf, force=True, mode="host") is None and "_get_samples" not in buf.__dict__
    with pytest.raises(ValueError):
        DB.install(buf, force=True, mode="nope")


def test_a_whole_update_staged_equals_the_resident_and_host_paths(monkeypatch):
    """The learner golden's update in each mode (forced on the CPU): post-update parameters bit-equal."""
    from agents.training.instrumented_ppo import ppo as ppo_mod
    real_install = DB.install

    def one(mode):
        def forced(buffer, **_k):
            return real_install(buffer, force=True, mode=mode)
        monkeypatch.setattr(ppo_mod, "_devb_install", forced)
        model = _golden_buffer()
        with torch.random.fork_rng():
            np.random.seed(0)
            torch.manual_seed(0)
            model.train()
        assert "get" not in model.rollout_buffer.__dict__
        return {n: p.detach().clone() for n, p in model.policy.named_parameters()}

    h, r, s = one("host"), one("resident"), one("staged")
    for n in h:
        assert torch.equal(h[n], r[n]) and torch.equal(h[n], s[n]), n


def test_the_trainer_defaults_to_staged():
    assert DB.DEFAULT_MODE == "staged" and set(DB.MODES) == {"resident", "staged", "host"}
