"""X14 OVERLAP — the pieces that need no env core (`rust_rollout/overlap.py`; the real-core pins are
`overlap_integration_test.py`).

* the BEHAVIOUR SNAPSHOTS keep the last two loaded versions and swap one into the policy and back BIT-EXACTLY,
  also on an exception;
* the DECOUPLED-PPO rewrite: a row's ``old_log_prob`` becomes log π_prox and its weight min(π_prox / μ, cap) is
  stored BESIDE its raw advantage — 1 where π_prox = μ (plain PPO); it needs the fill's π_prox and consumes it;
* NORMALISE, THEN WEIGHT (`gen3_overlap_norm_then_weight_v1`, core review A2 + C-3): through a real ``train()``,
  R1 gets w · norm(A) with its own normalisation off, which leaves NO push along μ → π_prox on an advantage that
  carries no information about the action (`_decoupled_logit_push`, also the measurement's harness); the signal
  tags and the ride-along target read the raw advantage; w = 1 is the serial update bit for bit;
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


def test_the_decoupled_rewrite_keeps_the_advantage_raw_and_stores_the_weight_beside_it():
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
    np.testing.assert_array_equal(buf.advantages, adv)             # the advantage stays RAW (A2 / C-3)
    w = buf.is_weights.reshape(-1)
    np.testing.assert_allclose(w[:5], np.exp(0.1), rtol=1e-4)
    assert w[5] == pytest.approx(IS_WEIGHT_CAP, rel=1e-6)
    np.testing.assert_allclose(w[6:], 1.0, atol=1e-4)
    assert out["overlap/is_weight_truncated_frac"] == pytest.approx(1 / adv.size)
    assert out["overlap/is_weight_max_raw"] == pytest.approx(np.exp(2.0), rel=1e-4)
    assert 0.0 < out["overlap/is_ess"] <= 1.0 and out["overlap/behaviour_kl"] > 0.0
    assert m._overlap_prox is None                   # consumed: a second update cannot re-weight the buffer


# ------------------------------------------------------------------------------- normalise, then weight (A2)
#: The bias fixture: mu (the behaviour policy) and pi_prox (the update's starting policy) over the toy's two
#: actions, TV distance 0.1; w = pi_prox / mu = (0.833, 1.25) stays below the cap.
_MU, _PROX = (0.6, 0.4), (0.5, 0.5)


def _decoupled_logit_push(mu=_MU, prox=_PROX, *, n_steps: int = 100, n_envs: int = 2, mean: float = 1.0,
                          std: float = 1.0, overlap: bool = True, batch_size: Any = None, seed: Any = None) -> dict:
    """The decoupled update's POLICY-GRADIENT push on the action logits, through a real ``train()`` (the toy
    learner, one epoch), read off what R1 is actually handed.

    The rows: with ``seed`` None, actions in EXACT proportion to ``mu`` and each action given the SAME multiset of
    advantages (mean ``mean``, std ``std``) — deterministic; with a ``seed``, actions drawn from ``mu`` and
    advantages from N(mean, std²), independently. Either way the advantage carries no information about the
    action, and the right answer is NO push. At the update's start pi = pi_prox, so the ratio is 1, the clip is
    inactive, and a micro-batch's surrogate logit gradient (ascent) is mean_i Â_i (e_{a_i} − pi_prox), with Â what
    R1 multiplies by the ratio — the advantage it is handed, normalised by R1 itself when its static says so; the
    push is the mean over the epoch's micro-batches. Only names present before the fix are used here, so the
    measurement script (`designs/research_state/measurements/x14_overlap_2026-10-19/adv_norm_bias.py`) runs this
    same harness against the pre-fix tree.

    Returns ``push`` (the logit vector), ``closed_form`` ((mean / S)(pi_prox − mu), the old order's expected push,
    S = std(w·A)), ``normalize_in_r1``, what R1 was handed (``handed``, ``actions``; the first micro-batch), the
    advantages the signal tags read (``signal_adv``) and each micro-batch's ``rollout_data.advantages``
    (``batch_adv``: what the eager tail — the ride-along ``adv`` head's target — reads)."""
    from types import SimpleNamespace

    import agents.training.instrumented_ppo.micro_step as ms
    import agents.training.instrumented_ppo.train_setup as ts
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    from agents.training.instrumented_ppo_test import _CounterDictEnv
    from agents.training.rust_rollout.testkit import ToyVecEnv, attach_vec_collector
    from agents.training.train_logger import configure

    rows = n_steps * n_envs
    mu_a, prox_a = np.asarray(mu, np.float64), np.asarray(prox, np.float64)
    if seed is None:
        counts = np.round(mu_a * rows).astype(int)
        unit = 20
        assert counts.sum() == rows and np.allclose(counts / rows, mu_a)  # the precondition: exact proportions
        assert all(c % unit == 0 for c in counts)
        z = np.linspace(-1.0, 1.0, unit)
        z = (z - z.mean()) / z.std()
        actions = np.concatenate([np.full(c, a) for a, c in enumerate(counts)])
        adv = np.concatenate([np.tile(mean + std * z, c // unit) for c in counts]).astype(np.float32)
    else:
        rng = np.random.default_rng(int(seed))
        actions = rng.choice(len(mu_a), size=rows, p=mu_a)
        adv = (mean + std * rng.standard_normal(rows)).astype(np.float32)

    env = ToyVecEnv([(lambda: _CounterDictEnv()) for _ in range(n_envs)])
    m = attach_vec_collector(InstrumentedMaskablePPO("MultiInputPolicy", env, n_steps=n_steps,
                                                     batch_size=int(batch_size or rows), n_epochs=1, ent_coef=0.0,
                                                     device="cpu", seed=0))
    m.set_logger(configure(None, []))
    buf = m.rollout_buffer
    shape = (n_steps, n_envs)
    buf.actions[...] = actions.reshape(*shape, 1)
    buf.log_probs[...] = np.log(mu_a[actions]).reshape(shape).astype(np.float32)
    buf.advantages[...] = adv.reshape(shape)
    buf.returns[...] = buf.advantages + buf.values
    buf.pos, buf.full = n_steps, True
    if overlap:
        m._rust_collector.overlap = SimpleNamespace(snapshots=None)     # runner_of(m): the decoupled update
        m.behaviour_check = "off"
        m._overlap_prox = np.log(prox_a[actions]).reshape(shape)
    seen: dict = {"calls": [], "batch_adv": []}
    real_micro, real_density = ms.micro_step, ts.advantage_density_metrics
    real_samples = type(buf)._get_samples

    def samples(self, batch_inds, env=None):                          # the micro-batch the eager tail reads
        out = real_samples(self, batch_inds, env)
        seen["batch_adv"].append(out.advantages.detach().double().numpy().copy())
        return out

    def spy(policy, obs, acts, masks, old_lp, old_v, advantages, returns, st):
        seen["calls"].append((acts.detach().reshape(-1).numpy().copy(),
                              advantages.detach().double().numpy().copy(), bool(st.normalize_advantage)))
        return real_micro(policy, obs, acts, masks, old_lp, old_v, advantages, returns, st)

    def density(a):
        seen["signal_adv"] = np.asarray(a, dtype=np.float64).copy()
        return real_density(a)

    ms.micro_step, ts.advantage_density_metrics, type(buf)._get_samples = spy, density, samples
    try:
        with th.random.fork_rng():
            np.random.seed(0)                                           # the epoch's permutation draw
            m.train()
    finally:
        ms.micro_step, ts.advantage_density_metrics, type(buf)._get_samples = real_micro, real_density, real_samples
    pushes = []
    for acts, handed, norm in seen["calls"]:
        a_hat = th.as_tensor(handed, dtype=th.float32)
        if norm and a_hat.numel() > 1:
            a_hat = (a_hat - a_hat.mean()) / (a_hat.std() + 1e-8)      # R1's own expression
        onehot = np.eye(len(mu_a))[acts.astype(int)]
        pushes.append((a_hat.double().numpy()[:, None] * (onehot - prox_a[None, :])).mean(0))
    wa = (prox_a / mu_a)[actions] * adv.astype(np.float64)
    first = seen["calls"][0]
    return dict(push=np.mean(pushes, axis=0), closed_form=(mean / wa.std(ddof=1)) * (prox_a - mu_a),
                normalize_in_r1=first[2], n_micro=len(pushes), signal_adv=seen.get("signal_adv"), raw_adv=adv,
                handed=first[1], actions=first[0], batch_adv=seen["batch_adv"])


def test_the_decoupled_update_normalises_the_raw_advantage_then_weights_it_and_leaves_no_push():
    got = _decoupled_logit_push()
    # THE BIAS: an action-independent advantage leaves no push (rounding only), where the old order pushed
    # along mu -> pi_prox by (E[A]/S)(pi_prox - mu), ~0.1 a logit here (pre-fix: [-0.0958, +0.0958])
    assert np.abs(got["closed_form"]).max() > 1e-2                    # the fixture is one the old order biases
    assert np.abs(got["push"]).max() < 1e-5, got["push"]
    assert got["normalize_in_r1"] is False and got["n_micro"] == 1     # R1's own normalisation is off: done here
    # what R1 is handed is w * norm(A), A the RAW advantage (normalised over the micro-batch, then weighted)
    from agents.training.rust_rollout.overlap import decoupled_advantages
    w = (np.asarray(_PROX) / np.asarray(_MU))[got["actions"].astype(int)]
    raw = th.as_tensor(got["raw_adv"])                                 # one micro-batch: every raw row
    want = decoupled_advantages(raw, th.ones_like(raw), True).double().numpy()
    np.testing.assert_allclose(np.sort(got["handed"] / w), np.sort(want), atol=1e-6)
    # C-3: the signal tags and the ride-along `adv` head's target (the micro-batch's own advantages) read the RAW
    # advantage, not w * A
    raw64 = np.sort(got["raw_adv"].astype(np.float64))
    np.testing.assert_array_equal(np.sort(got["signal_adv"].reshape(-1)), raw64)
    (batch_adv,) = got["batch_adv"]
    np.testing.assert_array_equal(np.sort(batch_adv), raw64)


def test_a_unit_weight_is_the_serial_updates_normalised_advantage_bit_for_bit():
    from agents.training.instrumented_ppo.micro_step import micro_step  # noqa: F401 — the expression's owner
    from agents.training.rust_rollout.overlap import decoupled_advantages

    g = th.Generator().manual_seed(3)
    a = th.randn(257, generator=g) * 3.0 + 0.7
    serial = (a - a.mean()) / (a.std() + 1e-8)                         # micro_step.py's expression
    assert th.equal(decoupled_advantages(a, th.ones_like(a), True), serial)
    assert th.equal(decoupled_advantages(a, th.ones_like(a), False), a)
    one = th.tensor([2.5])
    assert th.equal(decoupled_advantages(one, th.tensor([0.5]), True), one * 0.5)   # numel 1: never normalised


def test_the_serial_update_hands_r1_the_buffers_advantage_with_its_own_normalisation_on():
    got = _decoupled_logit_push(overlap=False)
    assert got["normalize_in_r1"] is True
    np.testing.assert_array_equal(np.sort(got["handed"]), np.sort(got["raw_adv"].astype(np.float64)))


def test_the_r1_gates_judge_a_decoupled_run_on_the_advantage_and_static_its_updates_hand_r1():
    """Under overlap R1's static turns its own normalisation off, so every startup gate that calls R1
    (`compile_regions._r1_args` / `_r1_reference_args`) must hand it the pre-normalised advantage (w = 1) — the
    inputs a production update gives it, and the static the compile declares at startup. Serial: untouched."""
    from types import SimpleNamespace

    from agents.model import compile_regions as cr
    from agents.training import learner_golden as LG
    from agents.training.rust_rollout.overlap import decoupled_advantages

    model = LG.build_learner()
    LG.load_buffer_into(model)
    b = cr.r1_batch(model, int(model.batch_size))
    serial = cr._r1_args(model, b)
    assert serial[6] is b.advantages and serial[-1].normalize_advantage is True
    model._rust_collector = SimpleNamespace(overlap=SimpleNamespace())          # a runner attached: decoupled
    on = cr._r1_args(model, b)
    assert on[-1].normalize_advantage is False
    assert th.equal(on[6], decoupled_advantages(b.advantages, th.ones_like(b.advantages), True))
    assert th.equal(cr._r1_reference_args(model, b)[6], on[6])


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
    # no fill made these buffers: an IDENTITY decoupled rewrite (w = 1), as the startup dry update does
    m._overlap_prepare = lambda: setattr(m.rollout_buffer, "is_weights", np.ones_like(m.rollout_buffer.advantages))
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
