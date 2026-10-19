"""X14 OVERLAP — the pieces that need no env core (`rust_rollout/overlap.py`; the real-core pins are
`overlap_integration_test.py`).

* the BEHAVIOUR SNAPSHOTS keep the last two loaded versions and swap one into the policy and back BIT-EXACTLY,
  also on an exception;
* the DECOUPLED-PPO rewrite: a row's ``old_log_prob`` becomes log π_prox and its advantage is scaled by
  min(π_prox / μ, cap) — 1 where π_prox = μ (plain PPO); it needs the fill's π_prox and consumes it;
* the LOOP: the worker begins right before an update — never before the first one (``warm``) nor the last — and
  an exception out of ``train()`` aborts it;
* the declaration refuses the fork arm and a non-graph T2 backend on CUDA; the flag parses, defaults OFF and is
  stamped only when ON.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np
import pytest
import torch as th

from agents.training.rust_rollout.overlap import IS_WEIGHT_CAP, BehaviourSnapshots, OverlapError, proximal_transform


class _Tiny(th.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = th.nn.Linear(3, 2)
        self.register_buffer("count", th.zeros(1, dtype=th.int64))


def test_the_snapshots_keep_the_last_two_versions_and_swap_back_bit_exactly():
    m = _Tiny()
    sl = BehaviourSnapshots(m)
    w = {}
    for v in range(3):
        with th.no_grad():
            m.lin.weight.add_(1.0)
            m.count.add_(1)
        w[v] = m.lin.weight.detach().clone()
        sl.take(m, v)
    assert sorted(sl.versions) == [1, 2] and not sl.has(0)
    live = m.lin.weight.detach().clone()
    with sl.swapped(m, 1):
        assert th.equal(m.lin.weight, w[1]) and int(m.count) == 2
    assert th.equal(m.lin.weight, live) and int(m.count) == 3
    with pytest.raises(RuntimeError, match="boom"):
        with sl.swapped(m, 1):
            raise RuntimeError("boom")
    assert th.equal(m.lin.weight, live)
    with pytest.raises(OverlapError):
        with sl.swapped(m, 0):
            pass
    assert sl.nbytes == 3 * (6 + 2) * 4 + 3 * 8


def test_the_decoupled_rewrite_is_plain_ppo_on_fresh_rows_and_weights_stale_rows():
    from agents.training.rust_rollout.consistency_test import _model
    from agents.training.train_logger import configure

    m = _model()
    m.set_logger(configure(None, []))
    buf = m.rollout_buffer
    own = buf.log_probs.copy()                       # the policy's own log-probs: every row "fresh"
    adv = np.arange(1, buf.log_probs.size + 1, dtype=np.float32).reshape(buf.log_probs.shape)
    buf.advantages[...] = adv
    # rows played by an OLDER policy: their stored mu is lower by delta, so w = exp(delta)
    flat_mu = buf.log_probs.reshape(-1)
    flat_mu[:5] -= 0.1
    flat_mu[5] -= 2.0                                # w = e^2 > the cap
    with pytest.raises(OverlapError, match="no proximal log-probs"):
        proximal_transform(m)                        # the fill computes them; none here
    m._overlap_prox = own.astype(np.float64)         # pi_prox: the weights now (the policy's own log-probs)
    out = proximal_transform(m)
    np.testing.assert_allclose(buf.log_probs, own, atol=1e-5)      # old_log_prob := log pi_prox (= own here)
    w = buf.advantages.reshape(-1) / adv.reshape(-1)
    np.testing.assert_allclose(w[:5], np.exp(0.1), rtol=1e-4)
    assert w[5] == pytest.approx(IS_WEIGHT_CAP, rel=1e-6)
    np.testing.assert_allclose(w[6:], 1.0, atol=1e-4)
    assert out["overlap/is_weight_truncated_frac"] == pytest.approx(1 / adv.size)
    assert out["overlap/is_weight_max_raw"] == pytest.approx(np.exp(2.0), rel=1e-4)
    assert 0.0 < out["overlap/is_ess"] <= 1.0 and out["overlap/behaviour_kl"] > 0.0
    assert m._overlap_prox is None                   # consumed: a second update cannot re-weight the buffer


# ------------------------------------------------------------------------------------------------ the loop
class _FakeRunner:
    """Records the loop's calls; never active (the toy collector has no worker to join)."""

    def __init__(self, log: List[str], warm: bool = True) -> None:
        self.log, self.warm, self.active = log, warm, False

    def begin(self) -> None:
        self.log.append("begin")

    def abort(self) -> None:
        self.log.append("abort")


def _toy(log: List[str], warm: bool = True) -> Any:
    from agents.training.owned_loop_test import N_ENVS, N_STEPS, _model

    m = _model()
    m._rust_collector.overlap = _FakeRunner(log, warm)
    train = m.train

    def _train() -> None:
        log.append("train")
        train()

    m.train = _train
    m._overlap_prepare = lambda: None
    return m, N_ENVS * N_STEPS


def test_the_worker_begins_before_every_update_but_the_last_and_only_once_warm():
    log: List[str] = []
    m, rows = _toy(log)
    m.learn(total_timesteps=3 * rows)
    assert log == ["begin", "train", "begin", "train", "train"], log
    cold: List[str] = []
    m2, _ = _toy(cold, warm=False)
    m2.learn(total_timesteps=2 * rows)
    assert cold == ["train", "train"], cold


def test_an_exception_out_of_the_update_aborts_the_worker():
    log: List[str] = []
    m, rows = _toy(log)

    def _boom() -> None:
        log.append("train")
        raise RuntimeError("update died")

    m.train = _boom
    with pytest.raises(RuntimeError, match="update died"):
        m.learn(total_timesteps=3 * rows)
    assert log == ["begin", "train", "abort"], log


# ------------------------------------------------------------------------------------------------ the surface
def test_the_declaration_refuses_the_fork_arm_and_a_non_graph_backend_on_cuda():
    from agents.training.rust_rollout.build import RustEnvDecl

    with pytest.raises(ValueError, match="fork arm"):
        RustEnvDecl(n_envs=4, device="cpu", backend="eager", overlap=True, fork=object())
    with pytest.raises(ValueError, match="'graph' backend"):
        RustEnvDecl(n_envs=4, device="cuda", backend="eager", overlap=True)
    assert RustEnvDecl(n_envs=4, device="cuda", backend="graph", overlap=True).overlap
    assert not RustEnvDecl(n_envs=4).overlap                      # default OFF


def test_the_flag_parses_defaults_off_and_is_stamped_only_when_on():
    from types import SimpleNamespace

    from main.train.parser import build_parser
    from main.train.rust_env_setup import env_core_stamp, resolve_env_core_args

    p = build_parser()
    on = p.parse_args(["--overlap-collect"])
    off = p.parse_args([])
    assert on.overlap_collect is True and off.overlap_collect is None
    resolve_env_core_args(off)
    assert off.overlap_collect is False

    class _Trig:
        target = 96

        def describe(self) -> str:
            return "t"

    def stamp(overlap: bool) -> dict:
        from agents.training.rust_rollout.build import RustEnvDecl

        decl = RustEnvDecl(n_envs=4, device="cpu", backend="eager", overlap=overlap)
        col = SimpleNamespace(core=SimpleNamespace(stamp="x"), cfg=SimpleNamespace(trigger=_Trig()))
        return env_core_stamp(decl, col)

    assert stamp(True)["overlap_collect"] is True and "overlap_collect" not in stamp(False)
