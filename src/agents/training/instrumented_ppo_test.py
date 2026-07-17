"""Tests for InstrumentedMaskablePPO drift detection and instrumentation."""

import copy
import hashlib
import inspect

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from sb3_contrib import MaskablePPO

from agents.training import instrumented_ppo
from agents.training.instrumented_ppo import (
    InstrumentedMaskablePPO,
    _EXPECTED_UPSTREAM_TRAIN_HASH,
    _VALUE_TAIL_FRAC,
    _verify_upstream_unchanged,
)
import torch as th
import torch.nn.functional as F


class _TailStub:
    """Minimal stand-in to exercise the pure _value_loss_from_se without building a full PPO."""
    def __init__(self, w):
        self.value_tail_weight = w


def test_value_loss_w0_is_byte_identical_to_mse():
    """β=0 → plain MSE, byte-identical to upstream F.mse_loss (the default-off no-op)."""
    th.manual_seed(0)
    target, pred = th.randn(256), th.randn(256)
    se = (target - pred) ** 2
    out = InstrumentedMaskablePPO._value_loss_from_se(_TailStub(0.0), se)
    assert th.allclose(out, F.mse_loss(target, pred))


def test_value_loss_w_positive_blends_in_cvar():
    """β>0 → (1-β)·MSE + β·CVaR(worst _VALUE_TAIL_FRAC). Strictly ≥ MSE (CVaR ≥ mean), and exact."""
    th.manual_seed(1)
    se = th.rand(500)
    w = 0.5
    out = InstrumentedMaskablePPO._value_loss_from_se(_TailStub(w), se)
    k = max(1, int(_VALUE_TAIL_FRAC * se.numel()))
    expected = (1 - w) * se.mean() + w * th.topk(se, k).values.mean()
    assert th.allclose(out, expected)
    assert out.item() >= se.mean().item()   # the tail is the worst errors → blend lifts the loss


def test_value_loss_default_attr_is_zero():
    """An unconfigured InstrumentedMaskablePPO defaults to β=0 (the class attribute) → MSE."""
    assert InstrumentedMaskablePPO.value_tail_weight == 0.0


def test_subclass_inherits_from_maskable_ppo():
    assert issubclass(InstrumentedMaskablePPO, MaskablePPO)


def test_train_method_is_overridden():
    # The subclass must define its own train, not inherit from MaskablePPO.
    assert InstrumentedMaskablePPO.train is not MaskablePPO.train


def test_recorded_upstream_hash_matches_current_upstream_source():
    """The pinned _EXPECTED_UPSTREAM_TRAIN_HASH must match the live upstream
    source. If this fails, sb3_contrib was updated and the vendored override
    in instrumented_ppo.py needs to be re-synced."""
    src = inspect.getsource(MaskablePPO.train)
    actual = hashlib.sha256(src.encode("utf-8")).hexdigest()
    assert actual == _EXPECTED_UPSTREAM_TRAIN_HASH, (
        "Upstream sb3_contrib.MaskablePPO.train() has changed; re-port the "
        "override in src/agents/training/instrumented_ppo.py and update "
        "_EXPECTED_UPSTREAM_TRAIN_HASH."
    )


def test_verify_upstream_unchanged_raises_on_mismatch(monkeypatch):
    """If the pinned hash doesn't match upstream, _verify_... must raise
    with a message that names the file and both hashes."""
    monkeypatch.setattr(
        instrumented_ppo,
        "_EXPECTED_UPSTREAM_TRAIN_HASH",
        "0" * 64,  # deliberately wrong
    )
    with pytest.raises(RuntimeError) as exc:
        _verify_upstream_unchanged()
    msg = str(exc.value)
    assert "DRIFT DETECTED" in msg
    assert "0" * 64 in msg  # the expected (wrong) hash is shown
    assert "instrumented_ppo.py" in msg  # the file to fix is named
    assert "ACTION REQUIRED" in msg


def test_instrumentation_marker_present_in_override():
    """Sanity check: the +INSTRUMENTATION marker survives in the override.
    If someone removes the instrumentation by accident, this fails."""
    src = inspect.getsource(InstrumentedMaskablePPO.train)
    assert "vf_clip_fractions" in src
    assert "train/clip_fraction_vf" in src
    assert "+INSTRUMENTATION" in src


# --------------------------------------------------------------------------------------
# Gradient accumulation (--grad-accum-steps): K micro-batches of `batch_size` summed into
# ONE optimizer step == the EXACT gradient of a (batch_size·K) batch, at the memory cost of
# one micro-batch. The class attr is OFF (=1) by default → byte-identical to upstream.
# --------------------------------------------------------------------------------------


def test_grad_accum_default_is_one():
    """Unconfigured → grad_accum_steps == 1 (one optimizer step per minibatch, stock behaviour)."""
    assert InstrumentedMaskablePPO.grad_accum_steps == 1


def test_grad_accum_marker_present_in_override():
    """The +GRAD-ACCUM accumulation logic must survive in the override (the step is gated on a
    full group + a trailing-partial-group flush)."""
    src = inspect.getsource(InstrumentedMaskablePPO.train)
    assert "+GRAD-ACCUM" in src
    assert "micro_in_group" in src
    assert "(loss / accum).backward()" in src


# --------------------------------------------------------------------------------------
# Exploiter distillation (gen3_exploiter_distill_v1): masked ON-POLICY KL(π_teacher ‖ π_student).
# --------------------------------------------------------------------------------------


def test_distill_loss_identical_logits_zero_kl():
    """Student == teacher ⇒ KL 0, full agreement, full coverage."""
    logits = th.randn(4, 6)
    out = InstrumentedMaskablePPO._distill_loss(
        logits.clone(), logits.clone(), th.ones(4, 6), th.ones(4, 1))
    assert out is not None
    loss, m = out
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    assert m["agree_rate"] == pytest.approx(1.0)
    assert m["coverage"] == pytest.approx(1.0)
    assert m["n"] == 4


def test_distill_loss_masks_non_teacher_rows():
    """Only the distill_mask==1 rows contribute; loss == mean of their per-row KL."""
    student, teacher = th.randn(4, 5), th.randn(4, 5)
    amask = th.ones(4, 5)
    dmask = th.tensor([[1.0], [0.0], [1.0], [0.0]])
    loss, m = InstrumentedMaskablePPO._distill_loss(student, teacher, amask, dmask)
    assert m["n"] == 2 and m["coverage"] == pytest.approx(0.5)
    logp = F.log_softmax(student, -1)
    p = F.softmax(teacher, -1)
    kl_row = (p * (th.log(p.clamp_min(1e-9)) - logp)).sum(-1)
    assert float(loss) == pytest.approx(float((kl_row[0] + kl_row[2]) / 2), rel=1e-5)


def test_distill_loss_none_when_no_teacher_rows():
    """A minibatch with zero teacher-team states ⇒ None (no NaN-poisoning an empty subset)."""
    assert InstrumentedMaskablePPO._distill_loss(
        th.randn(3, 4), th.randn(3, 4), th.ones(3, 4), th.zeros(3, 1)) is None


def test_distill_loss_respects_illegal_mask():
    """An illegal action the teacher 'wants' must not leak into the KL (both sides masked to -inf)."""
    student, teacher = th.zeros(2, 4), th.zeros(2, 4)
    teacher[:, 3] = 50.0                                  # teacher spikes the ILLEGAL action
    amask = th.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
    loss, _ = InstrumentedMaskablePPO._distill_loss(student, teacher, amask, th.ones(2, 1))
    assert float(loss) == pytest.approx(0.0, abs=1e-4)   # both uniform over the legal {0,1,2}


def test_distill_loss_grad_flows_student_only():
    """Gradient flows through the student logits; the (detached) teacher gets none."""
    student = th.randn(3, 5, requires_grad=True)
    teacher = th.randn(3, 5)                              # no requires_grad → frozen teacher
    loss, _ = InstrumentedMaskablePPO._distill_loss(student, teacher, th.ones(3, 5), th.ones(3, 1))
    loss.backward()
    assert student.grad is not None and th.isfinite(student.grad).all()
    assert teacher.grad is None


def test_value_distill_zero_when_equal():
    """V_student == V_teacher ⇒ 0 masked MSE."""
    v = th.randn(4)
    out = InstrumentedMaskablePPO._value_distill_mse(v.clone(), v.clone(), th.ones(4, 1))
    assert float(out) == pytest.approx(0.0, abs=1e-6)


def test_value_distill_masks_non_teacher_rows():
    """Only distill_mask==1 rows contribute; loss == masked-mean SE of those rows."""
    s, t = th.tensor([1., 2., 3., 4.]), th.tensor([1., 9., 3., 9.])   # differ on rows 1 (7) and 3 (5)
    out = InstrumentedMaskablePPO._value_distill_mse(s, t, th.tensor([[1.], [0.], [1.], [0.]]))
    assert float(out) == pytest.approx(0.0, abs=1e-6)                 # kept rows 0,2 are EQUAL
    out2 = InstrumentedMaskablePPO._value_distill_mse(s, t, th.tensor([[0.], [1.], [0.], [1.]]))
    assert float(out2) == pytest.approx((49. + 25.) / 2, rel=1e-5)    # kept rows 1,3 → (7²+5²)/2


def test_value_distill_none_no_rows():
    """No teacher-team rows ⇒ None (no NaN-poisoning an empty subset)."""
    assert InstrumentedMaskablePPO._value_distill_mse(th.randn(3), th.randn(3), th.zeros(3, 1)) is None


def test_value_distill_popart_frame_scales_by_sigma():
    """Under PopArt both sides are normalized first, so the SE is in the student's normalized frame."""
    class _FakePopart:
        def normalize(self, x):
            return (x - 5.0) / 2.0                                    # sigma = 2
    s, t = th.tensor([3., 3.]), th.tensor([5., 5.])                   # real diff 2 → normalized diff 1
    out = InstrumentedMaskablePPO._value_distill_mse(s, t, th.ones(2, 1), popart=_FakePopart())
    assert float(out) == pytest.approx(1.0, rel=1e-5)                 # (2/2)² = 1


def test_value_distill_grad_student_only():
    """Gradient flows into the student value; the frozen teacher value gets none."""
    s = th.randn(3, requires_grad=True)
    t = th.randn(3)
    out = InstrumentedMaskablePPO._value_distill_mse(s, t, th.ones(3, 1))
    out.backward()
    assert s.grad is not None and th.isfinite(s.grad).all()
    assert t.grad is None


def test_distill_reuse_masked_logits_bit_identical():
    """The #3 optimization (reuse the evaluate_actions forward, whose logits are MASKED) must give a
    BIT-IDENTICAL KL to a fresh (RAW) get_distribution forward: over LEGAL actions the logits are the same
    (masking adds nothing to legal), and illegal actions contribute exactly 0 to the KL either way."""
    B, A = 4, 6
    raw_student, teacher = th.randn(B, A), th.randn(B, A)
    amask = th.ones(B, A); amask[:, 4:] = 0.0                 # actions 4,5 illegal
    # masked student = raw with illegal set to a large negative (mimics MaskableCategorical.apply_masking)
    masked_student = raw_student.clone(); masked_student[:, 4:] = -1e8
    loss_raw, m_raw = InstrumentedMaskablePPO._distill_loss(raw_student, teacher, amask, th.ones(B, 1))
    loss_masked, m_masked = InstrumentedMaskablePPO._distill_loss(masked_student, teacher, amask, th.ones(B, 1))
    assert float(loss_raw) == float(loss_masked)             # EXACT (bit-identical), not approx
    assert m_raw["agree_rate"] == m_masked["agree_rate"]


def test_distill_multi_teacher_averaging():
    """N teachers on DISJOINT team-id subsets → the combined loss is the MEAN of the per-teacher masked
    KLs (per-archetype balancing: each teacher weighted equally regardless of its state count); a teacher
    with zero states is SKIPPED (None), never zero-weighted-in."""
    B, A = 6, 5
    student, t1, t2 = th.randn(B, A), th.randn(B, A), th.randn(B, A)
    amask = th.ones(B, A)
    tid = th.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 0.0])       # rows 0-2 → teacher 1, rows 3-4 → teacher 2, row 5 none
    kl1, m1 = InstrumentedMaskablePPO._distill_loss(student, t1, amask, (tid == 1).float())
    kl2, m2 = InstrumentedMaskablePPO._distill_loss(student, t2, amask, (tid == 2).float())
    assert m1["n"] == 3 and m2["n"] == 2                  # each teacher sees only its own team's states
    assert InstrumentedMaskablePPO._distill_loss(student, t1, amask, (tid == 9).float()) is None  # skipped
    combined = th.stack([kl1, kl2]).mean()               # what the train() loop folds
    assert float(combined) == pytest.approx(float((kl1 + kl2) / 2), rel=1e-6)
    # teacher-1's KL is the masked-mean over ONLY rows 0-2
    logp = F.log_softmax(student, -1); p = F.softmax(t1, -1)
    kl_row = (p * (th.log(p.clamp_min(1e-9)) - logp)).sum(-1)
    assert float(kl1) == pytest.approx(float(kl_row[:3].mean()), rel=1e-5)


class _CounterDictEnv(gym.Env):
    """Tiny Dict-obs maskable env (mirrors Gen3Env's {observation, action_mask} space). The
    observation counts up each step so the policy sees varied inputs → non-trivial gradients.
    Defined inline (not imported) so this test is self-contained."""

    def __init__(self, ep_len=1000):
        super().__init__()
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=0.0, high=1e4, shape=(1,), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(2,), dtype=np.int8),
        })
        self.action_space = spaces.Discrete(2)
        self._ep_len = ep_len
        self._t = 0

    def action_masks(self):
        return np.ones(2, dtype=np.int8)

    def _obs(self):
        return {"observation": np.array([float(self._t % 17)], dtype=np.float32),
                "action_mask": np.ones(2, dtype=np.int8)}

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        return self._obs(), float((self._t * 7) % 5), self._t >= self._ep_len, False, {}


def _build_tiny_ppo(n_steps=8, n_envs=4):
    from stable_baselines3.common.vec_env import DummyVecEnv
    venv = DummyVecEnv([(lambda: _CounterDictEnv()) for _ in range(n_envs)])
    model = InstrumentedMaskablePPO(
        "MultiInputPolicy", venv,
        n_steps=n_steps, batch_size=4, n_epochs=1,
        normalize_advantage=False,   # per-micro-batch adv-norm is the ONE non-identity; remove it for an exact check
        ent_coef=0.0, vf_coef=0.5, device="cpu", seed=0,
    )
    return model, venv


def _train_from_init(model, init_sd, init_opt, *, batch_size, accum, seed=123):
    """Reset the policy + optimizer to the captured init, then run ONE train() with the given
    (batch_size, accum). Returns a detached snapshot of every policy parameter."""
    model.policy.load_state_dict(init_sd)
    model.policy.optimizer.load_state_dict(init_opt)
    model.batch_size = batch_size
    model.grad_accum_steps = accum
    np.random.seed(seed)   # the rollout buffer's get() permutation — identical across both runs
    th.manual_seed(seed)
    model.train()
    return {k: v.detach().clone() for k, v in model.policy.state_dict().items()}


@pytest.mark.parametrize("micro,accum,full", [(4, 4, 16), (8, 2, 16), (5, 3, 15)])
def test_grad_accum_matches_full_batch(micro, accum, full):
    """accum=K over micro-batches of size B reproduces the parameter update of accum=1 over a single
    (B·K)=`full` batch — the BIT-EXACT-gradient guarantee (with normalize_advantage off so this
    isolates the accumulation math; empirically max|Δ|~3e-8, the float32 noise floor). The 32-sample
    buffer with micro∈{4,8} divides cleanly (all groups = K equal-size micros); micro=5 (→ a size-2
    trailing group that is a single micro) exercises the partial-group rescale and is still exact."""
    model, _venv = _build_tiny_ppo(n_steps=8, n_envs=4)   # 32 transitions in the buffer
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    # Fill model.rollout_buffer with one real rollout (learn() also runs one discarded train()).
    model.learn(total_timesteps=8 * 4)

    ref = _train_from_init(model, init_sd, init_opt, batch_size=full, accum=1)
    acc = _train_from_init(model, init_sd, init_opt, batch_size=micro, accum=accum)

    for k in ref:
        assert th.allclose(ref[k], acc[k], rtol=1e-4, atol=1e-6), (
            f"param {k} diverged: max|Δ|={float((ref[k]-acc[k]).abs().max()):.2e}"
        )


def test_grad_accum_nondivisible_is_bounded():
    """A NON-divisible rollout whose remainder minibatch lands in a group with full-size micro-batches
    (32 samples, micro=6, accum=2 → minibatches [6,6,6,6,6,2], final group [6,2]) is NOT bit-exact —
    the size-2 remainder is weighted as if full-size. The deviation must stay SMALL and BOUNDED (the
    rescale keeps it tiny; if someone broke the partial-group handling it would balloon). Documents the
    'use a divisible rollout for bit-exactness' caveat and guards against a gross regression."""
    model, _venv = _build_tiny_ppo(n_steps=8, n_envs=4)
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)

    ref = _train_from_init(model, init_sd, init_opt, batch_size=12, accum=1)   # effective 6·2
    acc = _train_from_init(model, init_sd, init_opt, batch_size=6, accum=2)
    dev = max(float((ref[k] - acc[k]).abs().max()) for k in ref)
    assert dev < 5e-3, f"non-divisible mixed-group deviation {dev:.2e} exceeds the bounded tolerance"
    assert dev > 1e-6, "expected a small (non-bit-exact) deviation here — the mixed-group remainder caveat"


# --------------------------------------------------------------------------------------
# Gradient noise scale (train/noise_scale, McCandlish 2018): B_simple = tr(Σ)/|G|², measured for
# free from the two batch sizes accumulation already produces (micro vs accumulated group).
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("G2,S,b_small,b_big", [(4.0, 1000.0, 64, 256), (10.0, 50.0, 100, 1000)])
def test_noise_scale_estimate_recovers_known_values(G2, S, b_small, b_big):
    """With the EXACT expectations E‖Ĝ_B‖² = |G|² + tr(Σ)/B fed in, the two-point estimator recovers
    |G|² and tr(Σ) exactly → B_simple = tr(Σ)/|G|²."""
    g_small_sq = G2 + S / b_small
    g_big_sq = G2 + S / b_big
    tr_sigma, g2 = InstrumentedMaskablePPO._noise_scale_estimate(g_small_sq, g_big_sq, b_small, b_big)
    assert g2 == pytest.approx(G2, rel=1e-9)
    assert tr_sigma == pytest.approx(S, rel=1e-9)
    assert (tr_sigma / g2) == pytest.approx(S / G2, rel=1e-9)   # B_simple


def test_noise_scale_smaller_batch_is_noisier_sign():
    """Sanity: a noisier (smaller) batch has the larger squared-norm estimate, so tr(Σ) and |G|² both
    come out POSITIVE for a sane (g_small_sq > g_big_sq) input."""
    tr_sigma, g2 = InstrumentedMaskablePPO._noise_scale_estimate(
        g_small_sq=5.0, g_big_sq=4.25, b_small=8, b_big=32)
    assert tr_sigma > 0 and g2 > 0


def test_global_grad_sq_matches_manual():
    """_global_grad_sq == Σ‖p.grad‖² over params (and 0 when no grads)."""
    net = th.nn.Linear(3, 2)
    assert InstrumentedMaskablePPO._global_grad_sq(net.parameters()) == 0.0   # no grads yet
    net(th.ones(4, 3)).sum().backward()
    manual = sum(float(p.grad.pow(2).sum()) for p in net.parameters())
    assert InstrumentedMaskablePPO._global_grad_sq(net.parameters()) == pytest.approx(manual, rel=1e-6)


def test_noise_scale_logged_only_when_accumulating():
    """End-to-end: a real train() runs the noise-scale measurement ONLY when accum>=2 (it needs two
    batch sizes). accum=1 → path skipped (EMA stays None, nothing logged). accum=2 → EMA updated; and
    with the EMA primed positive (as it is after warmup in a real run) the scalar IS logged.
    (A single-sample estimate on this 32-sample toy can be negative — correctly gated out of logging —
    which is exactly why the smoothing EMA exists; the math itself is pinned by the pure tests above.)"""
    model, _venv = _build_tiny_ppo(n_steps=8, n_envs=4)
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)

    class _Rec:
        def __init__(self): self.keys = set()
        def record(self, k, v, *a, **kw): self.keys.add(k)
        def __getattr__(self, _n): return lambda *a, **kw: None   # dump/record_mean/etc no-ops

    # accum=1: measurement path is skipped entirely (no second batch size).
    model._noise_ema_s = model._noise_ema_g2 = None
    model._logger = _Rec()
    _train_from_init(model, init_sd, init_opt, batch_size=8, accum=1)
    assert "train/noise_scale" not in model.logger.keys
    assert model._noise_ema_g2 is None and model._noise_ema_s is None

    # accum=2 (micro=4 → 2 groups): EMA primed positive (post-warmup state) → the path runs, folds a
    # fresh sample (EMA moves), and emits the scalar + ratio.
    model._noise_ema_s, model._noise_ema_g2 = 50.0, 2.0
    model._logger = _Rec()
    _train_from_init(model, init_sd, init_opt, batch_size=4, accum=2)
    assert "train/noise_scale" in model.logger.keys
    assert "train/noise_scale_ratio" in model.logger.keys
    assert model._noise_ema_g2 != 2.0 and model._noise_ema_s != 50.0   # a sample was folded in


def _fill_correction_buffer(model, n=8, better=1, adv=0.8):
    """Populate model._correction_buffer with corrections matching the tiny env (1-dim obs, 2 actions)."""
    from agents.training.teacher.buffer import Correction, CorrectionBuffer
    buf = CorrectionBuffer(100)
    for i in range(n):
        buf.add(Correction(
            obs=np.array([float(i % 7)], dtype=np.float32),
            action_mask=np.ones(2, dtype=np.int8),
            better_action=better, advantage=adv, confirmed_value=0.7,
            step_produced=0, opponent="bot"))
    model._correction_buffer = buf


def test_search_teacher_off_is_noop():
    """search_teacher_on False (the default) ⇒ train() never touches the AWR path — byte-identical."""
    model, _ = _build_tiny_ppo()
    model.learn(total_timesteps=8 * 4)        # off by default; must not crash, no _correction_buffer needed
    assert getattr(model, "_search_teacher_on", False) is False


def test_search_teacher_awr_folds_into_train():
    """With the AWR on + a populated buffer, train() runs the extra forward + distillation term and
    PULLS the policy toward A* (the buffer's better_action) — agree-rate rises after a few updates."""
    model, _ = _build_tiny_ppo(n_steps=8, n_envs=4)
    model._search_teacher_on = True
    model.search_teacher_coef = 5.0
    model.search_teacher_value_coef = 0.0
    model.search_teacher_beta = 1.0
    model.search_teacher_batch_size = 4
    _fill_correction_buffer(model, n=16, better=1, adv=1.0)   # always teach action 1

    # the corrections' obs/mask, to measure the policy's agreement with A*=1 before/after.
    import torch as th
    obs = {"observation": th.tensor([[float(i % 7)] for i in range(16)]),
           "action_mask": th.ones((16, 2))}
    def agree():
        with th.no_grad():
            lg = model.policy.get_distribution(obs).distribution.logits
        return float((lg.argmax(-1) == 1).float().mean())

    before = agree()
    for _ in range(15):                       # several train() calls (each does one AWR-weighted update)
        model.learn(total_timesteps=8 * 4, reset_num_timesteps=False)
    after = agree()
    assert after >= before                    # the distillation moved the policy toward A* (monotone-ish)
    assert after > 0.5                         # and it now predominantly plays the taught action


def test_search_teacher_buffer_excluded_from_save(tmp_path):
    """REGRESSION: `_correction_buffer` holds a threading.Lock — without excluding it from the SB3 save,
    `model.save()` dies with 'cannot pickle _thread.lock' (it crashed the pre-train roundtrip smoke, so
    EVERY --search-teacher run). It must save cleanly AND not be persisted (re-created empty on resume,
    so checkpoints stay small)."""
    model, _ = _build_tiny_ppo(n_steps=8, n_envs=4)
    model._search_teacher_on = True
    _fill_correction_buffer(model, n=8)
    assert "_correction_buffer" in model._excluded_save_params()
    p = str(tmp_path / "m.zip")
    model.save(p)                              # must NOT raise (the bug was a lock-pickle crash right here)
    reloaded = InstrumentedMaskablePPO.load(p, device="cpu")
    assert not hasattr(reloaded, "_correction_buffer")   # transient scaffolding → kept out of the checkpoint


# --- ON-POLICY SELF-DISTILLATION (OPD) --------------------------------------------------------------

def test_opd_loss_zero_when_target_matches_student():
    """KL(π' ‖ π_student) == 0 when π' IS the student's own (masked) softmax — the loss vanishes at the
    distillation fixed point."""
    logits = th.tensor([[2.0, -1.0, 0.5, 3.0]])
    mask = th.ones((1, 4))
    p_tgt = F.softmax(logits, dim=-1)                          # π' == the student distribution
    out = InstrumentedMaskablePPO._opd_loss(logits, mask, p_tgt)
    assert out is not None
    kl, m = out
    assert float(kl) == pytest.approx(0.0, abs=1e-6)
    assert m["agree_rate"] == pytest.approx(1.0)               # modes coincide


def test_opd_loss_positive_when_target_differs():
    """A π' that disagrees with the student's softmax → strictly positive KL."""
    logits = th.tensor([[2.0, -1.0, 0.5, 3.0]])
    mask = th.ones((1, 4))
    p_tgt = th.tensor([[0.0, 0.0, 0.0, 1.0]])                  # peaked on the last action (≠ softmax)
    kl, _ = InstrumentedMaskablePPO._opd_loss(logits, mask, p_tgt)
    assert float(kl) > 0.0


def test_opd_loss_none_guards():
    """None logits / None π' / empty π' → None (the loss is skipped, never NaN-poisoned)."""
    mask = th.ones((1, 4))
    p = th.tensor([[0.25, 0.25, 0.25, 0.25]])
    assert InstrumentedMaskablePPO._opd_loss(None, mask, p) is None
    assert InstrumentedMaskablePPO._opd_loss(th.zeros((1, 4)), mask, None) is None
    assert InstrumentedMaskablePPO._opd_loss(th.zeros((0, 4)), th.zeros((0, 4)), th.zeros((0, 4))) is None


def test_opd_loss_masks_illegal_actions():
    """An ILLEGAL action (mask 0, π'=0 there) is excluded from the KL: the student log-prob is over the
    legal set only, so a huge illegal logit doesn't perturb the loss."""
    mask = th.tensor([[1.0, 1.0, 0.0]])                        # action 2 illegal
    legal_logits = th.tensor([[1.0, 0.5, -50.0]])
    huge_illegal = th.tensor([[1.0, 0.5, 999.0]])             # only the illegal logit changes
    # π' over the two legal actions == the student's masked softmax (so KL should be ~0 in BOTH cases).
    masked = legal_logits + (mask - 1.0) * 1e9
    p_tgt = F.softmax(masked, dim=-1)
    kl_a, _ = InstrumentedMaskablePPO._opd_loss(legal_logits, mask, p_tgt)
    kl_b, _ = InstrumentedMaskablePPO._opd_loss(huge_illegal, mask, p_tgt)
    assert float(kl_a) == pytest.approx(0.0, abs=1e-5)
    assert float(kl_b) == pytest.approx(0.0, abs=1e-5)         # the illegal logit was masked out


def _fill_correction_buffer_opd(model, n=16, better=1, pi_row=(0.0, 1.0)):
    """Populate model._correction_buffer with corrections that ALSO carry a π' target (the improved
    distribution), matching the tiny env (1-dim obs, 2 actions)."""
    from agents.training.teacher.buffer import Correction, CorrectionBuffer
    buf = CorrectionBuffer(100)
    for i in range(n):
        buf.add(Correction(
            obs=np.array([float(i % 7)], dtype=np.float32),
            action_mask=np.ones(2, dtype=np.int8),
            better_action=better, advantage=0.8, confirmed_value=0.7,
            step_produced=0, opponent="bot",
            pi_target=np.array(pi_row, dtype=np.float32)))
    model._correction_buffer = buf


def test_opd_off_is_noop():
    """opd_on False (the default) ⇒ train() never touches the OPD path — byte-identical."""
    model, _ = _build_tiny_ppo()
    model.learn(total_timesteps=8 * 4)         # off by default; must not crash, no π' targets needed
    assert getattr(model, "_opd_on", False) is False


def test_opd_off_byte_identical_with_populated_buffer():
    """A populated OPD buffer with opd_coef=0 (OFF) yields the SAME parameter update as no buffer at all —
    the OPD term is truly gated on the coef, not merely on the buffer's presence."""
    model, _ = _build_tiny_ppo(n_steps=8, n_envs=4)
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)          # one rollout to fill the buffer's obs space

    # Baseline update with NO OPD.
    model._opd_on = False
    model.opd_coef = 0.0
    base = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)

    # Same update, but with a populated buffer AND opd_coef 0 (OFF) → the fold is skipped.
    _fill_correction_buffer_opd(model, n=16)
    model._opd_on = True
    model.opd_coef = 0.0
    with_buf = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    for k in base:
        assert th.allclose(base[k], with_buf[k], atol=1e-7), f"OPD off perturbed {k}"


def test_opd_fold_moves_policy_toward_pi_target():
    """With OPD on + a populated buffer whose π' peaks on action 1, train() runs the extra forward + KL
    term and PULLS the policy toward π' (the taught distribution) — agreement rises after a few updates."""
    model, _ = _build_tiny_ppo(n_steps=8, n_envs=4)
    model._opd_on = True
    model.opd_coef = 5.0
    model.opd_beta = 1.0
    model.search_teacher_batch_size = 4         # the OPD fold reuses this sample size
    _fill_correction_buffer_opd(model, n=16, pi_row=(0.0, 1.0))   # π' certain on action 1

    obs = {"observation": th.tensor([[float(i % 7)] for i in range(16)]),
           "action_mask": th.ones((16, 2))}
    def agree():
        with th.no_grad():
            lg = model.policy.get_distribution(obs).distribution.logits
        return float((lg.argmax(-1) == 1).float().mean())

    before = agree()
    for _ in range(15):
        model.learn(total_timesteps=8 * 4, reset_num_timesteps=False)
    after = agree()
    assert after >= before
    assert after > 0.5                          # the KL distillation moved the policy toward π'


def test_opd_skips_awr_only_buffer():
    """A buffer with corrections that carry NO π' (an AWR-only run) → to_tensors sets pi_target None →
    the OPD loss None-guards and is skipped, even with opd_on + a non-zero coef (no crash, no update from
    OPD)."""
    model, _ = _build_tiny_ppo(n_steps=8, n_envs=4)
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)

    base = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)

    _fill_correction_buffer(model, n=16, better=1)   # π'-LESS corrections (pi_target None)
    model._opd_on = True
    model.opd_coef = 5.0
    model.search_teacher_batch_size = 4
    with_awr_only = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    for k in base:
        assert th.allclose(base[k], with_awr_only[k], atol=1e-7), f"OPD acted on a π'-less buffer at {k}"
