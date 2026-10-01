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
    g1 = DB.install(buf, force=True)
    list(buf.get(16))
    g2 = DB.install(buf, force=True)                          # a raised update never uninstalled g1
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
            return real_install(buffer, force=force)
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
    with DB.device_samples(buf) as g:
        assert g is not None and g.device.type == "cuda"
        dev, _ = _batches(buf, 16, 2, seed=3)
    for eh, ed in zip(host, dev):
        _same(eh, ed)
