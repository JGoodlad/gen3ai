"""Tests for InstrumentedMaskablePPO drift detection and instrumentation."""

import copy
import hashlib
import inspect
import math

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
    assert str(instrumented_ppo._TRAIN_OVERRIDE_FILE) in msg  # the file to fix is named
    assert "ACTION REQUIRED" in msg


def test_the_drift_message_names_a_file_that_actually_holds_the_override():
    """The "port it into THIS file" pointer must name the file `train()` is really in.

    It used to be `__file__`, which was right only while the module was one file. On 2026-08-23
    `instrumented_ppo.py` became a package and the override moved to `ppo.py`, so `__file__`
    would have sent a reader to the hub — a message that is confidently wrong is worse than a
    vague one. `_TRAIN_OVERRIDE_FILE` is derived, and this pins it against the real definition
    site rather than against a string.
    """
    assert instrumented_ppo._TRAIN_OVERRIDE_FILE.is_file()
    assert (inspect.getfile(InstrumentedMaskablePPO.train)
            == str(instrumented_ppo._TRAIN_OVERRIDE_FILE))


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


# --- FitNets value-FEATURE distillation (gen3_exploiter_value_feat_distill_v1) ------------------------

def test_value_feat_distill_zero_when_aligned():
    """Perfectly-aligned value_pooled (same direction) ⇒ 0 masked cosine distance; a POSITIVE-SCALED copy
    is still 0 (cosine is scale-free — the whole point vs MSE)."""
    f = th.randn(4, 8)
    out = InstrumentedMaskablePPO._value_feat_distill(f.clone(), f.clone(), th.ones(4, 1))
    assert float(out) == pytest.approx(0.0, abs=1e-6)
    out_scaled = InstrumentedMaskablePPO._value_feat_distill(f.clone(), 7.5 * f.clone(), th.ones(4, 1))
    assert float(out_scaled) == pytest.approx(0.0, abs=1e-6)          # magnitude ignored, direction matched


def test_value_feat_distill_masks_non_teacher_rows():
    """Only distill_mask==1 rows contribute; loss == masked-mean cosine distance of those rows."""
    # rows 0,2 aligned (dist 0); rows 1,3 anti-aligned (cos −1 → dist 2)
    s = th.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    t = th.tensor([[2., 0.], [-3., 0.], [0., 5.], [0., -4.]])
    out = InstrumentedMaskablePPO._value_feat_distill(s, t, th.tensor([[1.], [0.], [1.], [0.]]))
    assert float(out) == pytest.approx(0.0, abs=1e-6)                 # kept rows 0,2 aligned
    out2 = InstrumentedMaskablePPO._value_feat_distill(s, t, th.tensor([[0.], [1.], [0.], [1.]]))
    assert float(out2) == pytest.approx(2.0, rel=1e-5)                # kept rows 1,3 anti-aligned → (2+2)/2


def test_value_feat_distill_none_no_rows():
    """No teacher-team rows / None inputs ⇒ None (no NaN-poisoning an empty subset)."""
    assert InstrumentedMaskablePPO._value_feat_distill(th.randn(3, 8), th.randn(3, 8), th.zeros(3, 1)) is None
    assert InstrumentedMaskablePPO._value_feat_distill(None, th.randn(3, 8), th.ones(3, 1)) is None
    assert InstrumentedMaskablePPO._value_feat_distill(th.randn(3, 8), None, th.ones(3, 1)) is None


def test_value_feat_distill_grad_student_only():
    """Gradient flows into the student hint; the frozen teacher hint gets none."""
    s = th.randn(3, 8, requires_grad=True)
    t = th.randn(3, 8)
    out = InstrumentedMaskablePPO._value_feat_distill(s, t, th.ones(3, 1))
    out.backward()
    assert s.grad is not None and th.isfinite(s.grad).all()
    assert t.grad is None


def test_value_feat_metric_is_published_under_the_distance_name_too():
    """THE NAMING TRAP. `_value_feat_distill` returns ``1 − cos`` — a DISTANCE — and the historical TB
    keys spell it `*_value_feat_cos`, which reads as its own opposite: a run logging 0.005 was reported
    as "the hint is near-ORTHOGONAL" when the data said cos ≈ 0.995 (near-parallel). Every site that
    records the old key must ALSO record the canonical `*_value_feat_dist`, so the honest name exists in
    TensorBoard while the old one stays alive for continuity."""
    import inspect
    src = inspect.getsource(InstrumentedMaskablePPO.train)
    per_teacher = 'f"t{_k}_value_feat_dist", f"t{_k}_value_feat_cos"'
    assert per_teacher in src, "the per-teacher site must publish t<k>_value_feat_dist beside the old key"
    assert '"value_feat_dist", "value_feat_cos"' in src, (
        "the aggregate site must publish value_feat_dist beside the old key")
    # …and the value really is a distance: aligned hints read 0, not 1.
    f = th.randn(4, 8)
    assert float(InstrumentedMaskablePPO._value_feat_distill(f.clone(), f.clone(), th.ones(4, 1))) \
        == pytest.approx(0.0, abs=1e-6)


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


def test_distill_loss_is_the_full_teacher_DISTRIBUTION_not_its_argmax():
    """The distillation target is the teacher's WHOLE softmax, not a hard action.

    The owner rule for the flywheel's delivery tick is 'always distil the full policy distribution' —
    so a teacher that is merely CONFIDENT and a teacher that is CERTAIN must produce different losses
    even though their argmax is the same. Two teachers sharing an argmax but differing off-mode give
    different KLs; a hard-CE (argmax) objective would score them identically. Pins the D-F contract
    against a future 'simplification' to top-1 distillation."""
    B, A = 4, 5
    student = th.randn(B, A)
    amask = th.ones(B, A)
    on = th.ones(B, 1)
    soft = th.zeros(B, A); soft[:, 2] = 1.0          # confident on action 2, mass elsewhere
    sharp = th.zeros(B, A); sharp[:, 2] = 8.0        # near-certain on action 2 — SAME argmax
    kl_soft, m_soft = InstrumentedMaskablePPO._distill_loss(student, soft, amask, on)
    kl_sharp, m_sharp = InstrumentedMaskablePPO._distill_loss(student, sharp, amask, on)
    assert m_soft["agree_rate"] == m_sharp["agree_rate"]        # identical argmax ⇒ a hard target ties
    assert float(kl_soft) != pytest.approx(float(kl_sharp), rel=1e-3)   # the full distribution does not
    # And it is the FORWARD KL Σ p_teacher·(log p_teacher − log p_student) over the legal set.
    p = F.softmax(soft, -1)
    expect = (p * (th.log(p) - F.log_softmax(student, -1))).sum(-1).mean()
    assert float(kl_soft) == pytest.approx(float(expect), rel=1e-5)


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


class _DistillDictEnv(_CounterDictEnv):
    """`_CounterDictEnv` plus the training-only `distill_mask` key the exploiter-distillation KL reads
    (the INTEGER teacher-id: 0 = not a teacher's team, k = teacher k). Every state is teacher 1's here,
    so the fold has full coverage and cannot be skipped by the 'no teacher-team rows' None guard."""

    def __init__(self, ep_len=1000):
        super().__init__(ep_len=ep_len)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=0.0, high=1e4, shape=(1,), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(2,), dtype=np.int8),
            "distill_mask": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })

    def _obs(self):
        o = super()._obs()
        o["distill_mask"] = np.array([1.0], dtype=np.float32)
        return o


def _build_distill_ppo(n_steps=8, n_envs=4):
    """A tiny PPO whose env emits `distill_mask`, plus a frozen TEACHER model built on the PLAIN env
    (no `distill_mask` key) — mirroring production, where the teacher is a foreign checkpoint whose
    observation space is a subset and the fold filters the obs keys down to what the teacher knows."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    venv = DummyVecEnv([(lambda: _DistillDictEnv()) for _ in range(n_envs)])
    model = InstrumentedMaskablePPO(
        "MultiInputPolicy", venv, n_steps=n_steps, batch_size=4, n_epochs=1,
        normalize_advantage=False, ent_coef=0.0, vf_coef=0.5, device="cpu", seed=0,
    )
    teacher, _ = _build_tiny_ppo(n_steps=n_steps, n_envs=n_envs)
    # Both are seeded identically, so an untouched teacher is a near-COPY of the student (KL ~1e-6) and
    # "did the KL fall?" would have nothing to measure. Perturb the teacher's action head so it holds a
    # genuinely different policy — the situation a foreign exploiter checkpoint is in.
    th.manual_seed(7)                      # the perturbation is FIXED — the test must not flap
    with th.no_grad():
        for p in teacher.policy.action_net.parameters():
            p.add_(th.randn_like(p) * 2.0)
    teacher.policy.set_training_mode(False)
    return model, teacher


def test_distill_off_byte_identical_with_teachers_attached():
    """Teachers ATTACHED but `distill_coef=0` ⇒ the same parameter update as no teachers at all.

    The revival pin for the flywheel's delivery tick: the OPD half has had this guarantee pinned since
    it shipped (`test_opd_off_byte_identical_with_populated_buffer`), the EXPLOITER-distillation half
    never did — so nothing caught a regression that made the fold depend on the teacher LIST rather
    than the coefficient. `distill_on` must be gated on the coef."""
    model, teacher = _build_distill_ppo(n_steps=8, n_envs=4)
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)          # one rollout to fill the buffer

    model._distill_teachers = []                # baseline: no teachers at all
    model.distill_coef = 0.0
    base = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)

    model._distill_teachers = [teacher]         # teachers present, coef 0 → the fold is skipped
    model.distill_coef = 0.0
    off = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    for k in base:
        assert th.equal(base[k], off[k]), f"distill_coef=0 with a teacher attached perturbed {k}"


def test_distill_on_folds_into_the_update_and_pulls_toward_the_teacher():
    """coef > 0 with a teacher attached DOES change the update (the term is live end-to-end through
    train(), not just in the pure loss helper), and repeated updates raise the student↔teacher argmax
    agreement — the ON half of the byte-identity pin above."""
    model, teacher = _build_distill_ppo(n_steps=8, n_envs=4)
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)

    model._distill_teachers = []
    model.distill_coef = 0.0
    base = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)

    model._distill_teachers = [teacher]
    model.distill_coef = 10.0
    on = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    assert any(not th.equal(base[k], on[k]) for k in base), "distill_coef>0 did not enter the update"

    s_obs = {"observation": th.tensor([[float(i % 17)] for i in range(17)]),
             "action_mask": th.ones((17, 2)),
             "distill_mask": th.ones((17, 1))}
    t_obs = {k: v for k, v in s_obs.items() if k != "distill_mask"}
    with th.no_grad():
        t_logp = F.log_softmax(teacher.policy.get_distribution(t_obs).distribution.logits, -1)

    def kl():
        # The quantity the fold minimises: forward KL(teacher ‖ student). argmax agreement is the wrong
        # readout on a 2-action toy — two random nets already agree ~everywhere, so it starts saturated.
        with th.no_grad():
            s_logp = F.log_softmax(model.policy.get_distribution(s_obs).distribution.logits, -1)
        return float((t_logp.exp() * (t_logp - s_logp)).sum(-1).mean())

    model.policy.load_state_dict(init_sd)
    model.policy.optimizer.load_state_dict(init_opt)
    before = kl()
    for _ in range(15):
        model.learn(total_timesteps=8 * 4, reset_num_timesteps=False)
    assert kl() < before, "the distillation KL did not fall over repeated updates"


# ---- gen3_distill_target_gate_v1 (--distill-target / --distill-topk / --distill-gate /
# --distill-gate-tau / --distill-beta): the ACTION-FORM distill target + the advantage gate
# (design_advantage_gated_distillation.md §3.1/§3.3/§7.3). The provenance genre is gated in
# agents/model/distill_target_gate_provenance_test.py; here is the loss itself + the fold.

def test_action_distill_at_K_full_and_no_gate_reproduces_the_kl():
    """§7.3 identity 1: K = n_actions + gate=none + Â ≡ 0 reproduces `_distill_loss`'s masked-mean
    forward KL to floating-point tolerance — the identity that makes the new path a SUPERSET of
    the old rather than a replacement (with zero advantages the AWR weight is exp(0) = 1 for every
    row, so the weighted mean IS the plain masked mean)."""
    th.manual_seed(3)
    B, A = 6, 5
    student, teacher = th.randn(B, A), th.randn(B, A)
    amask = th.ones(B, A); amask[:, 4] = 0.0                       # one illegal column (both sides)
    dmask = th.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    adv = th.zeros(B)
    acts = th.zeros(B, dtype=th.long)
    ref, ref_m = InstrumentedMaskablePPO._distill_loss(student, teacher, amask, dmask)
    out = InstrumentedMaskablePPO._gated_action_distill_loss(
        student, teacher, amask, dmask, adv, acts, top_k=A, tau=0.0, beta=1.0, gate="none")
    assert out is not None
    loss, m = out
    assert float(loss) == pytest.approx(float(ref), rel=1e-5)
    assert m["n_gated"] == ref_m["n"]                              # same rows: every on-pin one


def test_action_distill_at_K1_reproduces_the_searchteacher_ce():
    """§7.3 identity 2: K=1 (one-hot target ⇒ the KL form degenerates to −log π_S(a_T)) with the
    AWR weight reproduces `_searchteacher_loss`'s weighted CE when the 'better action' IS the
    teacher's argmax and every advantage is ≥ 0 (there |Â| = Â, as in the AWR buffer)."""
    th.manual_seed(4)
    B, A = 5, 4
    student, teacher = th.randn(B, A), th.randn(B, A)
    amask = th.ones(B, A)
    adv = th.rand(B) * 3.0                                         # ≥ 0, like a confirmed improvement
    t_argmax = teacher.argmax(-1)
    ref, _ = InstrumentedMaskablePPO._searchteacher_loss(student, amask, t_argmax, adv, beta_awr=1.3)
    out = InstrumentedMaskablePPO._gated_action_distill_loss(
        student, teacher, amask, th.ones(B), adv, th.zeros(B, dtype=th.long),
        top_k=1, tau=0.0, beta=1.3, gate="none")
    assert out is not None
    assert float(out[0]) == pytest.approx(float(ref), rel=1e-5)


def test_action_distill_hand_checked_kl_vs_topk_vs_ce():
    """The three target forms on one hand-checkable row: teacher p = (0.5, 0.3, 0.2), student
    uniform over 3 legal actions, Â = 0 (w ≡ 1).

      full KL (K=3): ln3 − H(p)                       ≈ 0.0690
      top-2 KL:      q = (0.625, 0.375, 0); Σq·ln q + ln3 ≈ 0.4370
      argmax CE (K=1): −ln(1/3) = ln 3                ≈ 1.0986

    CE > top-K > KL — the dial monotonically trades tail shape for ordering."""
    import math
    teacher = th.log(th.tensor([[0.5, 0.3, 0.2]]))
    student = th.zeros(1, 3)                                       # uniform
    amask, dmask = th.ones(1, 3), th.ones(1)
    adv, acts = th.zeros(1), th.zeros(1, dtype=th.long)

    def run(k):
        out = InstrumentedMaskablePPO._gated_action_distill_loss(
            student, teacher, amask, dmask, adv, acts, top_k=k, tau=0.0, beta=1.0, gate="none")
        assert out is not None
        return float(out[0])

    h_p = -(0.5 * math.log(0.5) + 0.3 * math.log(0.3) + 0.2 * math.log(0.2))
    assert run(3) == pytest.approx(math.log(3) - h_p, abs=1e-4)                       # ≈ 0.0690
    q1, q2 = 0.5 / 0.8, 0.3 / 0.8
    assert run(2) == pytest.approx(q1 * math.log(q1) + q2 * math.log(q2) + math.log(3), abs=1e-4)
    assert run(1) == pytest.approx(math.log(3), abs=1e-4)                             # ≈ 1.0986
    assert run(1) > run(2) > run(3)


def test_advantage_gate_selects_exactly_the_disagreeing_negative_rows():
    """§3.1: a row fires iff BOTH (teacher argmax ≠ sampled action) AND (Â < −τ) AND on-pin.
    Four rows, one per exclusion reason — only row 0 contributes, and a row above −τ contributes
    NOTHING (the loss equals the loss computed with that row absent)."""
    A = 3
    teacher = th.tensor([[0.0, 0.0, 5.0]] * 4)                     # argmax = 2 on every row
    student = th.tensor([[0.0, 2.0, 0.0]] * 4)                     # argmax = 1 (disagrees w/ teacher)
    amask = th.ones(4, A)
    dmask = th.tensor([1.0, 1.0, 1.0, 0.0])                        # row 3: off-pin
    acts = th.tensor([0, 0, 2, 0])                                 # row 2: sampled the teacher's action
    adv = th.tensor([-1.0, -0.2, -1.0, -1.0])                      # row 1: above −τ (τ = 0.5)
    out = InstrumentedMaskablePPO._gated_action_distill_loss(
        student, teacher, amask, dmask, adv, acts, top_k=1, tau=0.5, beta=1.0, gate="advantage")
    assert out is not None
    loss, m = out
    assert m["n_gated"] == 1.0 and m["gated_frac"] == pytest.approx(0.25)
    assert m["gate_agree_rate"] == 0.0                             # student argmax 1 ≠ teacher 2
    assert m["mean_gate_adv"] == pytest.approx(-1.0)
    # Row 0 alone: CE toward action 2 under the student's softmax (the single-row weighted mean —
    # the weight cancels).
    expect = float(F.cross_entropy(student[:1], th.tensor([2])))
    assert float(loss) == pytest.approx(expect, rel=1e-5)
    # The above-−τ row contributes nothing: removing it changes nothing.
    out2 = InstrumentedMaskablePPO._gated_action_distill_loss(
        student[[0, 2, 3]], teacher[[0, 2, 3]], amask[[0, 2, 3]], dmask[[0, 2, 3]],
        adv[[0, 2, 3]], acts[[0, 2, 3]], top_k=1, tau=0.5, beta=1.0, gate="advantage")
    assert float(out2[0]) == pytest.approx(float(loss), rel=1e-6)


def test_an_empty_gate_returns_none_never_nan():
    """§7.3: an empty gated subset returns None (the term is skipped), never a 0/0 NaN."""
    A = 3
    teacher = th.tensor([[0.0, 0.0, 5.0]] * 2)
    student = th.zeros(2, A)
    out = InstrumentedMaskablePPO._gated_action_distill_loss(
        student, teacher, th.ones(2, A), th.ones(2),
        th.tensor([0.5, 1.0]),                                     # every advantage POSITIVE → no row
        th.tensor([0, 0]), top_k=1, tau=0.0, beta=1.0, gate="advantage")
    assert out is None


def test_action_distill_end_to_end_folds_and_an_empty_gate_is_byte_identical():
    """The fold half (through a real train()): `--distill-target action` enters the update; and
    with an advantage gate no minibatch ever satisfies (τ = 1e9), the update is byte-identical to
    no distillation at all — the None guard end-to-end — while distill/n_gated still logs 0.0
    (a reading, not an absence)."""
    model, teacher = _build_distill_ppo(n_steps=8, n_envs=4)
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)

    model._distill_teachers = []
    model.distill_coef = 0.0
    base = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)

    model._distill_teachers = [teacher]
    model.distill_coef = 10.0
    model.distill_target = "action"
    on = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    assert any(not th.equal(base[k], on[k]) for k in base), (
        "--distill-target action did not enter the update")
    assert model.logger.name_to_value.get("distill/n_gated", 0.0) > 0.0
    assert "distill/gated_frac" in model.logger.name_to_value
    assert "distill/gate_agree_rate" in model.logger.name_to_value

    model.distill_gate = "advantage"
    model.distill_gate_tau = 1e9                                   # Â < −1e9 never holds
    gated = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    for k in base:
        assert th.equal(base[k], gated[k]), (
            f"an empty advantage gate still perturbed {k} — the None guard failed end-to-end")
    assert model.logger.name_to_value.get("distill/n_gated") == 0.0
    assert model.logger.name_to_value.get("distill/gated_frac") == 0.0


def test_distill_target_class_default_is_kl_and_dispatch_is_source_guarded():
    """The default is the untouched full-distribution path: the class attribute says 'kl' and the
    fold dispatches to the LITERAL `_distill_loss` call on that branch (byte-identity at the
    default is the C6 build-vs-enable contract, §7.3)."""
    assert InstrumentedMaskablePPO.distill_target == "kl"
    assert InstrumentedMaskablePPO.distill_gate == "none"
    assert InstrumentedMaskablePPO.distill_topk == 1
    import inspect
    src = inspect.getsource(InstrumentedMaskablePPO.train)
    assert 'if _d_target == "kl":' in src and "_gated_action_distill_loss(" in src, (
        "the target-form dispatch left the fold — the kl default must take the literal "
        "_distill_loss call")


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


def test_noise_scale_advice_bands_and_fixes():
    """The advisor's PURE decision logic: which band fires, and does the message name the fix.

    Re-homed here at v78 — it lived in `zarch_test.py`, which was deleted with the zarch family, and
    the FiLM half of the advisor went with it. The GLOBAL half still ships and still writes into the
    launcher Events panel, so it keeps a test rather than inheriting the deletion by accident.
    """
    advise = InstrumentedMaskablePPO._noise_scale_advice

    assert advise(None, 16384.0) == []                  # nothing measured yet
    assert advise(1.0, 16384.0) == []                   # in band
    assert advise(2.0, 16384.0) == []                   # boundary is EXCLUSIVE
    assert advise(0.5, 16384.0) == []

    high = advise(6.0, 16384.0)
    assert [k for k, _ in high] == ["global_high"]
    assert "--grad-accum-steps" in high[0][1] and "6" in high[0][1], (
        "a noise-limited warning must name the flag AND the multiple to raise it by — the whole "
        "point is that the reader does not have to derive the fix")

    low = advise(0.1, 16384.0)
    assert [k for k, _ in low] == ["global_low"]
    assert "OVER-BATCHED" in low[0][1] and "--grad-accum-steps" in low[0][1]


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
    model._noise_ema_n = 0
    model._logger = _Rec()
    _train_from_init(model, init_sd, init_opt, batch_size=8, accum=1)
    assert "train/noise_scale" not in model.logger.keys
    assert model._noise_ema_g2 is None and model._noise_ema_s is None

    # accum=2 (micro=4 → 2 groups): EMA primed positive (post-warmup state) → the path runs, folds a
    # fresh sample (EMA moves), and emits the scalar + ratio.
    model._noise_ema_s, model._noise_ema_g2 = 50.0, 2.0
    # POST-WARMUP means the COUNT is primed too (gen3_noise_scale_warmup_v1): the total now folds
    # through `noise_scale.debiased_ema`, whose effective decay is `1 - 1/(n+1)`, so an EMA primed
    # with a value but not a count is still on sample 1 and takes the next sample WHOLE — which on
    # this 32-row toy can be the negative estimate the emit gate correctly withholds.
    model._noise_ema_n = 500
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


# --------------------------------------------------------------------------------------
# Save-exclusion of transient CUDA-bearing state. Anything holding CUDA tensors that is pickled
# into a snapshot's data section deserializes WITHOUT map_location, so every env/eval worker that
# loads the snapshot (device="cpu") silently initializes a ~252 MiB GPU context — dozens of workers
# exhausted the card (the 2026-07-20 OOM cascade). `_film_grad_accumulator` was the case that
# taught it; it went with the FiLM generators at config v78, and the survivors are pinned here so
# the LESSON outlives the module that motivated it.
# --------------------------------------------------------------------------------------


def test_transient_cuda_bearing_state_excluded_from_save():
    """The transient train()-owned attachments must be in _excluded_save_params — a snapshot
    carrying one poisons every CPU worker that loads it with a GPU context (or, for the correction
    buffer, fails to pickle at all on its threading.Lock)."""
    excluded = InstrumentedMaskablePPO._excluded_save_params(
        InstrumentedMaskablePPO.__new__(InstrumentedMaskablePPO))
    for name in ("_correction_buffer", "_distill_teacher", "_distill_teachers"):
        assert name in excluded, f"{name} would be pickled into every checkpoint"


# --------------------------------------------------------------------------------------
# COUNTERFACTUAL win-prob grounding (gen3_cf_label_plumbing_v1) — the coefficient-zero
# byte-identity pin plus the two halves of `cf_head_only`.
#
# The tiny PPO's `CombinedExtractor` has no win-prob head (and no parameters at all), so the fold
# is stubbed with a two-module stand-in whose params ARE in the optimizer: a "trunk" the CF term
# reaches only when `cf_head_only` is False, and a "head" it always reaches. That makes the
# stop-grad a MEASURABLE property of the parameter update rather than a claim about a detach call.
# --------------------------------------------------------------------------------------
import base64 as _b64
import hashlib as _hashlib
import json as _json


class _CfStash:
    def __init__(self):
        self.value_pooled = None


def _build_cf_ppo(n_steps=8, n_envs=4):
    """A tiny PPO wearing a stubbed win-prob head, wired so the CF fold can actually run."""
    model, _ = _build_tiny_ppo(n_steps=n_steps, n_envs=n_envs)
    fe = model.policy.features_extractor
    th.manual_seed(11)                       # the stub is FIXED — the test must not flap
    trunk, head = th.nn.Linear(1, 8), th.nn.Linear(8, 1)
    fe.cf_trunk_stub = trunk                 # registered → in state_dict, comparable across runs
    fe.win_head = head
    fe.stash = _CfStash()
    _base = type(fe)

    def _forward(self, obs):
        # The CF term calls the extractor with ONLY the "observation" key (the only key the real
        # Gen3 extractor reads); the PPO update calls it with the full obs dict.
        if "action_mask" in obs:
            return _base.forward(self, obs)
        self.stash.value_pooled = th.relu(trunk(obs["observation"]))
        return self.stash.value_pooled

    # Patched on a per-instance SUBCLASS, not the instance: `_cf_winprob_term` deliberately calls
    # `type(fe).forward` (the always-eager path — see its docstring), so an instance attribute
    # would not be seen. Subclassing keeps the stub off the shared CombinedExtractor class.
    fe.__class__ = type("_CfStubExtractor", (_base,), {"forward": _forward})
    model.policy.optimizer.add_param_group(
        {"params": list(trunk.parameters()) + list(head.parameters())})
    return model


def _write_cf_labels(dirpath, n=8, obs_dim=1, label=1.0, policy_step=0):
    dirpath.mkdir(parents=True, exist_ok=True)
    with open(dirpath / "labels_test_0.jsonl", "w") as f:
        for i in range(n):
            raw = np.full(obs_dim, float(i % 3), dtype=np.float32).tobytes()
            f.write(_json.dumps({
                "schema": 1, "kind": "mc_winprob", "battle": f"b{i}", "decision_idx": i,
                "obs_sha1": _hashlib.sha1(raw).hexdigest(), "obs_npz": None,
                "obs_inline": _b64.b64encode(raw).decode(), "label": label, "n_rollouts": 8,
                "wilson_lo": 0.0, "wilson_hi": 1.0, "policy_step": policy_step,
                "opponent": "pool", "created_unix": 0.0,
            }) + "\n")


def _attach_cf_buffer(model, tmp_path, **kw):
    from agents.training.cf_label_buffer import CfLabelBuffer
    _write_cf_labels(tmp_path / "cf_labels", **kw)
    model._cf_buffer = CfLabelBuffer(tmp_path / "cf_labels", obs_dim=1, lag_bound=0)
    return model._cf_buffer


def test_cf_off_byte_identical_with_populated_buffer(tmp_path):
    """A POPULATED label buffer with `cf_winprob_coef=0` yields the SAME parameter update as no
    buffer at all — the fold is gated on the COEFFICIENT, not on the buffer's presence.

    This is the G3 gate: the whole step ships at coefficient zero, so "off is off" is the only
    thing standing between a plumbing change and a silent perturbation of a live run.
    """
    model = _build_cf_ppo()
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)

    model._cf_buffer = None
    model.cf_winprob_coef = 0.0
    base = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)

    buf = _attach_cf_buffer(model, tmp_path)
    model.cf_winprob_coef = 0.0
    off = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    for k in base:
        assert th.equal(base[k], off[k]), f"cf_winprob_coef=0 with a populated buffer perturbed {k}"
    assert len(buf) == 0, "an OFF run must not even poll the label directory"


def test_cf_off_byte_identical_when_only_the_head_is_missing(tmp_path):
    """coef > 0 + a populated buffer but NO win-prob head (`--win-prob-mode none`) must also be a
    no-op rather than a crash — the CLI refuses that combination, and the loss agrees."""
    model = _build_cf_ppo()
    del model.policy.features_extractor.win_head
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)

    model._cf_buffer = None
    model.cf_winprob_coef = 0.0
    base = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)

    _attach_cf_buffer(model, tmp_path)
    model.cf_winprob_coef = 5.0
    off = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    for k in base:
        assert th.equal(base[k], off[k]), f"a headless run folded a CF term into {k}"


def test_cf_head_only_moves_the_head_and_never_the_trunk(tmp_path):
    """`cf_head_only=True` (the DEFAULT, the design's safe R1 stage): the ground-truth BCE trains
    the win-prob head's own params and leaves everything upstream of the stop-grad bit-identical."""
    model = _build_cf_ppo()
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)

    model._cf_buffer = None
    model.cf_winprob_coef = 0.0
    base = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)

    _attach_cf_buffer(model, tmp_path)
    model.cf_winprob_coef = 5.0
    model.cf_head_only = True
    on = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)

    head_keys = [k for k in base if "win_head" in k]
    trunk_keys = [k for k in base if "cf_trunk_stub" in k]
    assert head_keys and trunk_keys
    assert any(not th.equal(base[k], on[k]) for k in head_keys), "the CF term never reached the head"
    for k in trunk_keys:
        assert th.equal(base[k], on[k]), f"head-only leaked a gradient into the trunk via {k}"


def test_cf_without_head_only_reaches_the_trunk(tmp_path):
    """`--no-cf-head-only`: the same term, now allowed to shape the shared representation."""
    model = _build_cf_ppo()
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)

    model._cf_buffer = None
    model.cf_winprob_coef = 0.0
    base = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)

    _attach_cf_buffer(model, tmp_path)
    model.cf_winprob_coef = 5.0
    model.cf_head_only = False
    on = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)

    trunk_keys = [k for k in base if "cf_trunk_stub" in k]
    assert any(not th.equal(base[k], on[k]) for k in trunk_keys), \
        "cf_head_only=False did not reach the trunk"


def test_cf_buffer_is_excluded_from_save():
    excluded = InstrumentedMaskablePPO._excluded_save_params(
        InstrumentedMaskablePPO.__new__(InstrumentedMaskablePPO))
    assert "_cf_buffer" in excluded


def test_cf_class_defaults_are_off_and_head_only():
    """The shipped defaults, asserted on the CLASS so a resume that sets nothing is safe."""
    assert InstrumentedMaskablePPO.cf_winprob_coef == 0.0
    assert InstrumentedMaskablePPO.cf_head_only is True
    # gen3_cf_binomial_likelihood_v1 / gen3_cf_evidential_head_v1: the likelihood DEFAULTS to the
    # correct one (the lever has never run in production, so there is no legacy to preserve), and
    # the evidential term defaults OFF.
    assert InstrumentedMaskablePPO.cf_label_likelihood == "binomial"
    assert InstrumentedMaskablePPO.cf_evidential_coef == 0.0
    assert InstrumentedMaskablePPO.cf_evidential_reg == 1e-3


# --------------------------------------------------------------------------------------
# THE BINOMIAL LIKELIHOOD (gen3_cf_binomial_likelihood_v1) — pure-function properties.
#
# These test `_cf_binomial_nll` directly rather than through a PPO step, because the two claims
# ("it reduces exactly to BCE at n=1" and "it weights by n") are EXACT arithmetic facts, and an
# end-to-end assertion could only ever check them approximately.
# --------------------------------------------------------------------------------------

def test_binomial_equals_bce_exactly_when_every_label_has_one_rollout():
    """The reduction that makes 'binomial' a strict GENERALISATION rather than a different loss.

    At n ≡ 1 a well-formed label is already 0 or 1 (a one-rollout Monte-Carlo estimate has no other
    values), so `round(label·n)` is the identity and Σn = B — leaving exactly the mean BCE the flat
    path computes. Bit-for-bit, not approximately: if this ever drifts, the two `--cf-label-
    likelihood` arms have stopped being comparable at their shared boundary.
    """
    logits = th.tensor([-1.3, 0.4, 2.0, -0.2, 5.0, -5.0])
    labels = th.tensor([0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    binom = InstrumentedMaskablePPO._cf_binomial_nll(logits, labels, th.ones(6))
    bce = th.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    assert th.equal(binom, bce)


def test_binomial_weights_each_row_by_its_rollout_count():
    """An R=16 label pulls exactly 4x an R=4 label — the whole point of the change.

    Measured on the GRADIENT w.r.t. the logits, where the weighting lives (d/dz_i of the normalized
    loss is (n_i/Σn)·(q_i − label_i)), so this is a claim about what the optimizer sees rather than
    about the loss value.
    """
    logits = th.tensor([0.3, 0.3], requires_grad=True)
    labels = th.tensor([1.0, 1.0])
    InstrumentedMaskablePPO._cf_binomial_nll(logits, labels, th.tensor([4.0, 16.0])).backward()
    assert float(logits.grad[1] / logits.grad[0]) == pytest.approx(4.0, rel=1e-6)


def test_binomial_is_normalized_per_rollout_so_the_coefficient_survives_a_producer_change():
    """Σ NLL / Σ n, not Σ NLL — so doubling the producer's R does not double the term and silently
    double the effective coefficient. Identical labels at R=4 and at R=16 must give the SAME loss."""
    logits = th.tensor([-0.7, 1.1, 0.0])
    labels = th.tensor([0.0, 1.0, 1.0])
    a = InstrumentedMaskablePPO._cf_binomial_nll(logits, labels, th.full((3,), 4.0))
    b = InstrumentedMaskablePPO._cf_binomial_nll(logits, labels, th.full((3,), 16.0))
    assert float(a) == pytest.approx(float(b), rel=1e-6)


def test_binomial_recovers_the_win_count_from_the_ratio():
    """`w = round(label·n)`: a 0.625 label at R=8 is FIVE wins, and the loss must be the loss of
    five wins and three losses — not of a soft target that happens to average to 0.625."""
    logit = th.tensor([0.0])                       # q = 0.5, so every term is log 2
    loss = InstrumentedMaskablePPO._cf_binomial_nll(logit, th.tensor([0.625]), th.tensor([8.0]))
    assert float(loss) == pytest.approx(math.log(2.0), rel=1e-6)   # (5+3)·log2 / 8


def test_a_missing_rollout_count_degrades_to_one_observation():
    """The buffer parses an absent `n_rollouts` as 0. It must become ONE observation, never a
    divide-by-zero and never a silently dropped row."""
    logits = th.tensor([0.4, -0.4])
    labels = th.tensor([1.0, 0.0])
    zero = InstrumentedMaskablePPO._cf_binomial_nll(logits, labels, th.zeros(2))
    one = InstrumentedMaskablePPO._cf_binomial_nll(logits, labels, th.ones(2))
    assert th.equal(zero, one)


# --------------------------------------------------------------------------------------
# THE EVIDENTIAL BETA HEAD (gen3_cf_evidential_head_v1) — the train-loop half.
#
# The head's MATH lives in `agents/model/cf_evidential_head_test.py` (checked against scipy). What
# is checked HERE is the only thing that can go wrong in the fold: whether the gradient reaches the
# right parameters and NOTHING else, measured on the actual parameter update rather than asserted
# about a `.detach()` call — the same standard the `cf_head_only` halves are held to above.
# --------------------------------------------------------------------------------------

def _attach_cf_evid_head(model, in_dim=8):
    """Give the stub extractor a real `CfEvidentialHead` sized to the stub's value_pooled.

    The class's loss/metric maths are classmethods over (α, β), so swapping `net` for a small
    Linear keeps every property under test while sidestepping the production D_MODEL width.
    """
    from agents.model.aux_value_heads import CfEvidentialHead
    th.manual_seed(23)
    head = CfEvidentialHead()
    head.net = th.nn.Sequential(th.nn.Linear(in_dim, 2))
    model.policy.features_extractor.cf_evid_head = head
    model.policy.optimizer.add_param_group({"params": list(head.parameters())})
    return head


def test_cf_evidential_off_byte_identical_with_a_populated_buffer(tmp_path):
    """ON-at-coefficient-0: the head exists in the state_dict and in the optimizer, a full label
    buffer is attached, and the parameter update is nonetheless IDENTICAL to no head at all.

    This is the shipping contract — the flag lands OFF, and a future run that builds the head
    without turning the coefficient on must be indistinguishable from one that did neither.
    """
    model = _build_cf_ppo()
    _attach_cf_evid_head(model)
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)

    model._cf_buffer = None
    model.cf_winprob_coef = 0.0
    model.cf_evidential_coef = 0.0
    base = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)

    _attach_cf_buffer(model, tmp_path)
    model.cf_winprob_coef = 0.0
    model.cf_evidential_coef = 0.0
    off = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    for k in base:
        assert th.equal(base[k], off[k]), f"cf_evidential_coef=0 perturbed {k}"


def test_cf_evidential_reaches_only_its_own_head(tmp_path):
    """A LIVE evidential coefficient moves the evidential head's params — and nothing else.

    Its input is detached UNCONDITIONALLY (there is no `head_only` switch), so the trunk stub must
    be bit-identical; and it must not touch the WIN-PROB head either, since the two readouts are
    separate consumers of the same `value_pooled`. Both halves are measured on the update.
    """
    model = _build_cf_ppo()
    _attach_cf_evid_head(model)
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)

    model._cf_buffer = None
    model.cf_winprob_coef = 0.0
    model.cf_evidential_coef = 0.0
    base = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)

    _attach_cf_buffer(model, tmp_path)
    model.cf_winprob_coef = 0.0                 # the SCALAR term stays off: attribution is clean
    model.cf_evidential_coef = 5.0
    on = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)

    evid_keys = [k for k in base if "cf_evid_head" in k]
    trunk_keys = [k for k in base if "cf_trunk_stub" in k]
    head_keys = [k for k in base if "win_head" in k]
    assert evid_keys and trunk_keys and head_keys
    assert any(not th.equal(base[k], on[k]) for k in evid_keys), \
        "the evidential term never reached its own head"
    for k in trunk_keys:
        assert th.equal(base[k], on[k]), f"the ALWAYS-DETACHED head leaked into the trunk via {k}"
    for k in head_keys:
        assert th.equal(base[k], on[k]), f"the evidential term perturbed the win-prob head via {k}"


def test_cf_evidential_without_the_head_is_a_no_op(tmp_path):
    """A live coefficient on a run whose extractor has no `cf_evid_head` must fold nothing rather
    than crash. The CLI refuses that combination; the loss agrees, belt and braces."""
    model = _build_cf_ppo()
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)

    model._cf_buffer = None
    model.cf_winprob_coef = 0.0
    model.cf_evidential_coef = 0.0
    base = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)

    _attach_cf_buffer(model, tmp_path)
    model.cf_winprob_coef = 0.0
    model.cf_evidential_coef = 5.0              # …but no head was ever built
    off = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    for k in base:
        assert th.equal(base[k], off[k]), f"a headless run folded an evidential term into {k}"


def test_both_cf_terms_share_one_sample_and_one_forward(tmp_path):
    """The two readouts must see the SAME rows off ONE extractor forward.

    Two samples would pay twice for the forward (the entire cost of the block) and would make the
    scalar and evidential terms disagree about which states they were scored on — which would
    quietly invalidate any comparison between them. Counted on the buffer's sampler.
    """
    model = _build_cf_ppo()
    _attach_cf_evid_head(model)
    buf = _attach_cf_buffer(model, tmp_path)
    model.cf_winprob_coef = 1.0
    model.cf_evidential_coef = 1.0
    calls = []
    real_sample = buf.sample
    buf.sample = lambda n: (calls.append(n) or real_sample(n))    # type: ignore[method-assign]
    fwd = []
    fe = model.policy.features_extractor
    real_forward = type(fe).forward
    type(fe).forward = lambda self, obs: (                        # type: ignore[method-assign]
        fwd.append("observation" in obs and "action_mask" not in obs) or real_forward(self, obs))
    try:
        model.learn(total_timesteps=8 * 4)
    finally:
        type(fe).forward = real_forward                           # type: ignore[method-assign]
    n_minibatches = len(calls)
    assert n_minibatches > 0, "preconditions: the CF block never ran"
    # exactly one CF-shaped forward (obs-only dict) per minibatch, for BOTH terms together
    assert sum(1 for is_cf in fwd if is_cf) == n_minibatches


# --------------------------------------------------------------------------------------
# THE CF FORWARD'S TWO GUARDS (task #28 / the review's perf notes, 2026-08-22)
#
# Both are properties of the ONE forward `_cf_sample_and_forward` runs, and both are the kind of
# thing that is invisible until it is wrong: a graph nobody consumes costs memory and time with no
# symptom, and a debugger fed foreign rows reports against the wrong premise with no symptom either.
# --------------------------------------------------------------------------------------

def test_the_cf_forward_builds_no_graph_when_nothing_downstream_wants_one(tmp_path):
    """`cf_head_only` (the default) detaches, and the evidential head detaches unconditionally — so
    the extractor graph was built and immediately discarded on every minibatch of every epoch.

    Measured on the stashed tensor rather than on a timing: `value_pooled.requires_grad` is exactly
    the presence of the graph.
    """
    model = _build_cf_ppo()
    _attach_cf_evid_head(model)
    _attach_cf_buffer(model, tmp_path)
    model.cf_winprob_coef = 1.0
    model.cf_evidential_coef = 1.0
    model.cf_head_only = True
    seen = []
    real = InstrumentedMaskablePPO._cf_sample_and_forward

    def spy(self):
        ctx = real(self)
        if ctx is not None:
            seen.append(bool(ctx.value_pooled.requires_grad))
        return ctx

    model.__class__ = type("_Spy", (type(model),), {"_cf_sample_and_forward": spy})
    model.learn(total_timesteps=8 * 4)
    assert seen, "preconditions: the CF block never ran"
    assert not any(seen), "head-only built an extractor graph nothing consumes"


def test_the_cf_forward_KEEPS_the_graph_for_the_trunk_open_arm(tmp_path):
    """The one configuration that needs it: `--no-cf-head-only` with a LIVE win-prob coefficient.

    The no_grad optimisation is conditioned exactly, not assumed, because dropping the graph here
    would silently turn the trunk-open arm into head-only — an A/B whose two arms are the same run.
    """
    model = _build_cf_ppo()
    _attach_cf_buffer(model, tmp_path)
    model.cf_winprob_coef = 1.0
    model.cf_head_only = False
    seen = []
    real = InstrumentedMaskablePPO._cf_sample_and_forward

    def spy(self):
        ctx = real(self)
        if ctx is not None:
            seen.append(bool(ctx.value_pooled.requires_grad))
        return ctx

    model.__class__ = type("_Spy", (type(model),), {"_cf_sample_and_forward": spy})
    model.learn(total_timesteps=8 * 4)
    assert seen and all(seen), "the trunk-open arm lost the graph it trains through"


def test_head_only_plus_evidential_still_trains_BOTH_heads_own_params(tmp_path):
    """The no_grad guard's real risk: it must not starve a head of its OWN gradient.

    `head(value_pooled)` is applied OUTSIDE the no_grad context, so a detached input still yields a
    full gradient for the head's parameters — but "still" is a claim about where a `with` block
    ends, so it is measured on the parameter update for both consumers at once.
    """
    model = _build_cf_ppo()
    _attach_cf_evid_head(model)
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)

    model._cf_buffer = None
    model.cf_winprob_coef = 0.0
    model.cf_evidential_coef = 0.0
    base = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)

    _attach_cf_buffer(model, tmp_path)
    model.cf_winprob_coef = 5.0
    model.cf_evidential_coef = 5.0
    model.cf_head_only = True
    on = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)

    win_keys = [k for k in base if "win_head" in k]
    evid_keys = [k for k in base if "cf_evid_head" in k]
    trunk_keys = [k for k in base if "cf_trunk_stub" in k]
    assert win_keys and evid_keys and trunk_keys
    assert any(not th.equal(base[k], on[k]) for k in win_keys), \
        "no_grad starved the WIN-PROB head of its own gradient"
    assert any(not th.equal(base[k], on[k]) for k in evid_keys), \
        "no_grad starved the EVIDENTIAL head of its own gradient"
    for k in trunk_keys:
        assert th.equal(base[k], on[k]), f"head-only reached the trunk via {k}"


def test_the_observation_debugger_is_suppressed_for_the_cf_forward_and_restored(tmp_path):
    """The CF rows are RECORDED FOREIGN states — other episodes, other policy steps, read off disk.

    The debugger's premise is "this is the board we are about to act on", so those rows are not its
    business: it would report their integrity failures as though the live env had produced them.
    Suppressed for that one forward and RESTORED after, including when the forward raises.
    """
    from agents.model.features_extractor import Gen3FeaturesExtractor

    model = _build_cf_ppo()
    fe = model.policy.features_extractor
    sentinel = object()
    fe._debugger = sentinel
    # Borrow the real context manager — the stub extractor is not a Gen3FeaturesExtractor, but the
    # seam under test is exactly that method, so binding it is the honest way to exercise it.
    fe.suppress_observation_debugger = (
        Gen3FeaturesExtractor.suppress_observation_debugger.__get__(fe, type(fe)))

    inside = []
    _base_forward = type(fe).forward
    type(fe).forward = lambda self, obs: (                       # type: ignore[method-assign]
        inside.append(self._debugger) if "action_mask" not in obs else None
    ) or _base_forward(self, obs)
    try:
        _attach_cf_buffer(model, tmp_path)
        model.cf_winprob_coef = 1.0
        model.learn(total_timesteps=8 * 4)
    finally:
        type(fe).forward = _base_forward                         # type: ignore[method-assign]

    assert inside, "preconditions: the CF forward never ran"
    assert all(d is None for d in inside), "the debugger was fed recorded foreign CF rows"
    assert fe._debugger is sentinel, "the debugger was not restored after the CF forward"


def test_the_debugger_suppression_restores_even_when_the_forward_raises():
    """Exception-safe, because the alternative is losing the live obs-integrity check for the rest
    of a multi-day run over one transient failure in an aux term."""
    from agents.model.features_extractor import Gen3FeaturesExtractor

    class _Fake:
        pass

    fe = _Fake()
    fe._debugger = "the-real-debugger"
    cm = Gen3FeaturesExtractor.suppress_observation_debugger.__get__(fe, _Fake)
    with pytest.raises(RuntimeError):
        with cm() as had:
            assert had is True and fe._debugger is None
            raise RuntimeError("the forward blew up")
    assert fe._debugger == "the-real-debugger"


def test_cf_rows_sampled_reports_the_rows_the_fold_actually_ate(tmp_path):
    """Residency (`cf/buffer_fill`) and throughput are different questions, and only the second one
    goes to zero when a producer dies mid-run while its last labels are still resident."""
    model = _build_cf_ppo()
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)

    class _Rec:
        def __init__(self): self.vals = {}
        def record(self, k, v, *a, **kw): self.vals[k] = v
        def __getattr__(self, _n): return lambda *a, **kw: None

    _attach_cf_buffer(model, tmp_path, n=6)
    model.cf_winprob_coef = 1.0
    model._logger = _Rec()
    _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    vals = model.logger.vals
    assert "cf/rows_sampled" in vals
    # 3 distinct states in the fixture (obs fill cycles i % 3, and rows dedup on the obs digest),
    # eaten once per minibatch — so the total is a positive multiple of the resident count.
    assert vals["cf/rows_sampled"] > 0
    assert vals["cf/rows_sampled"] % float(len(model._cf_buffer)) == 0


# --------------------------------------------------------------------------------------
# TWIN HEADS + SHADOW CRITIC (gen3_cf_twin_heads_v1) — the owner-authorized amendment to the
# signed R1 pre-registration (ledger 2026-08-22 evening, "Three owner sign-offs" item 3).
#
# The arm's whole claim is that the three heads differ ONLY in their label stream, so the gates
# here are about ROUTING and ISOLATION rather than about loss values: which head sees which label,
# and which parameters a live coefficient is allowed to touch. A swap between B and C would leave
# every published scalar looking healthy while the factorial measured its own mirror image.
# --------------------------------------------------------------------------------------


def _attach_cf_twin_heads(model, in_dim=8):
    """Two `WinProbHead`s sized to the stub's value_pooled, plus the `last_value_pooled` property
    the on-policy mirror reads. Fixed seed — a paired test must not flap on init."""
    from agents.model.aux_value_heads import WinProbHead
    th.manual_seed(29)
    heads = []
    for name in ("cf_twin_head_b", "cf_twin_head_c"):
        h = WinProbHead()
        h.net = th.nn.Sequential(th.nn.Linear(in_dim, 1))
        setattr(model.policy.features_extractor, name, h)
        model.policy.optimizer.add_param_group({"params": list(h.parameters())})
        heads.append(h)
    return heads


def _attach_cf_shadow_head(model, in_dim=8):
    from agents.model.aux_value_heads import ShadowValueHead
    th.manual_seed(31)
    head = ShadowValueHead()
    head.net = th.nn.Sequential(th.nn.Linear(in_dim, 1))
    model.policy.features_extractor.cf_shadow_head = head
    model.policy.optimizer.add_param_group({"params": list(head.parameters())})
    return head


def _write_cf_twin_labels(dirpath, n=8, obs_dim=1, label=1.0, outcome=0.0,
                          mc_return=None, reward_sha1="", policy_step=0):
    """The v1 row with the twin streams attached. `label` (tight-MC) and `outcome` are given
    DIFFERENT values on purpose: that is what makes a B/C routing swap detectable at all."""
    dirpath.mkdir(parents=True, exist_ok=True)
    with open(dirpath / "labels_twin_0.jsonl", "w") as f:
        for i in range(n):
            raw = np.full(obs_dim, float(i % 3), dtype=np.float32).tobytes()
            row = {
                "schema": 1, "kind": "mc_winprob", "battle": f"b{i}", "decision_idx": i,
                "obs_sha1": _hashlib.sha1(raw).hexdigest(), "obs_npz": None,
                "obs_inline": _b64.b64encode(raw).decode(), "label": label, "n_rollouts": 8,
                "wilson_lo": 0.0, "wilson_hi": 1.0, "policy_step": policy_step,
                "opponent": "pool", "created_unix": 0.0, "outcome_label": outcome,
            }
            if mc_return is not None:
                row.update(mc_return=mc_return, mc_return_n=8, reward_sha1=reward_sha1)
            f.write(_json.dumps(row) + "\n")


def _attach_cf_twin_buffer(model, tmp_path, **kw):
    from agents.training.cf_label_buffer import CfLabelBuffer
    _write_cf_twin_labels(tmp_path / "cf_labels", **kw)
    model._cf_buffer = CfLabelBuffer(tmp_path / "cf_labels", obs_dim=1, lag_bound=0,
                                     reward_sha1=kw.get("reward_sha1") or None)
    return model._cf_buffer


def _cf_twin_baseline(model, tmp_path, unclipped=False):
    """`(init_sd, init_opt, base_params)` with every cf coefficient at zero.

    ``unclipped`` raises `max_grad_norm` out of the way. See
    `test_the_only_coupling_between_a_headonly_term_and_the_trunk_is_the_GLOBAL_CLIP` for why an
    isolation test that leaves it alone measures the clip rather than the detach.
    """
    if unclipped:
        model.max_grad_norm = 1e9
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)
    model._cf_buffer = None
    model.cf_winprob_coef = 0.0
    model.cf_twin_coef = 0.0
    model.cf_shadow_coef = 0.0
    base = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    return init_sd, init_opt, base


def test_twin_heads_at_coefficient_zero_are_byte_identical_to_not_having_them(tmp_path):
    """ON-at-coefficient-0 must leave EVERY parameter update bit-identical — including the twins'
    own, which is the stronger half.

    The whole twin block is gated on `cf_twin_coef`, INCLUDING the on-policy mirror. That is a
    deliberate choice: a mirror that ran at coefficient zero would train B and C on head A's loss
    alone, which is a perfectly reasonable control condition but is NOT "off", and "off is off" is
    the only thing standing between building the heads and perturbing a live run.
    """
    model = _build_cf_ppo()
    _attach_cf_twin_heads(model)
    init_sd, init_opt, base = _cf_twin_baseline(model, tmp_path)

    _attach_cf_twin_buffer(model, tmp_path)
    model.cf_twin_coef = 0.0
    off = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    for k in base:
        assert th.equal(base[k], off[k]), f"cf_twin_coef=0 with a populated buffer perturbed {k}"


def test_a_live_twin_coefficient_reaches_ONLY_the_twin_heads(tmp_path):
    """Head-only is not a mode for the twins, it is their definition in v1 — so a live coefficient
    must move `cf_twin_head_{b,c}` and leave the trunk, head A and everything else bit-identical.

    Measured on the parameter update rather than asserted about a `.detach()` call, because the
    claim the amendment stands on is "the three heads share ONE trunk", and a leaked gradient would
    make the trunk a function of the arm.

    Run with the global grad-norm clip raised out of the way, deliberately: the clip is a real but
    DIFFERENT coupling (it rescales every gradient by a factor that depends on the total norm, so
    any additional term perturbs every parameter in the last bits), and it is pinned separately by
    `test_the_only_coupling_between_a_headonly_term_and_the_trunk_is_the_GLOBAL_CLIP`. Leaving it in
    here would make this test a measurement of the clip rather than of the detach.
    """
    model = _build_cf_ppo()
    _attach_cf_twin_heads(model)
    init_sd, init_opt, base = _cf_twin_baseline(model, tmp_path, unclipped=True)

    _attach_cf_twin_buffer(model, tmp_path)
    model.cf_twin_coef = 5.0
    on = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)

    twin_keys = [k for k in base if "cf_twin_head" in k]
    other_keys = [k for k in base if "cf_twin_head" not in k]
    assert twin_keys and other_keys
    assert any(not th.equal(base[k], on[k]) for k in twin_keys), "the twin term never reached B/C"
    for k in other_keys:
        assert th.equal(base[k], on[k]), f"the twin term leaked a gradient into {k}"


def test_head_B_eats_the_OUTCOME_and_head_C_eats_the_TIGHT_MC_label(tmp_path):
    """THE ROUTING PIN — the GIGO of this whole arm.

    B's stream is the recorded SINGLE OUTCOME at n=1; C's is the TIGHT-MC ratio at n=R. A swap
    leaves every scalar looking healthy and silently measures the factorial's mirror image, so the
    routing is pinned on the loss VALUES rather than on which variable name appears where: with the
    two labels set to opposite extremes, each head's loss must equal the binomial NLL of ITS OWN
    label and must NOT equal the other's.
    """
    model = _build_cf_ppo()
    heads = _attach_cf_twin_heads(model)
    model.learn(total_timesteps=8 * 4)
    # Opposite extremes: tight-MC says "certain win", the recorded outcome says "lost".
    _attach_cf_twin_buffer(model, tmp_path, label=1.0, outcome=0.0)
    model._cf_buffer.poll(0)
    model.cf_twin_coef = 1.0
    ctx = model._cf_sample_and_forward()
    assert ctx is not None
    _term, m = model._cf_twin_terms(ctx)

    pooled = ctx.value_pooled.detach()
    b_logits = heads[0](pooled).flatten()
    c_logits = heads[1](pooled).flatten()
    ones = th.ones_like(ctx.batch.label)
    want_b = float(InstrumentedMaskablePPO._cf_binomial_nll(b_logits, ctx.batch.outcome, ones))
    want_c = float(InstrumentedMaskablePPO._cf_binomial_nll(
        c_logits, ctx.batch.label, ctx.batch.n_rollouts))
    # B against ITS label...
    assert m["b_loss"] == pytest.approx(want_b, rel=1e-6)
    assert m["c_loss"] == pytest.approx(want_c, rel=1e-6)
    # ...and NOT against the other's. With the labels at opposite extremes these are far apart, so
    # this half of the pin cannot pass by coincidence.
    crossed_b = float(InstrumentedMaskablePPO._cf_binomial_nll(
        b_logits, ctx.batch.label, ctx.batch.n_rollouts))
    assert abs(m["b_loss"] - crossed_b) > 1e-3, "head B was fed the tight-MC label"
    assert m["b_coverage"] == pytest.approx(1.0)


def test_head_B_is_weighted_as_ONE_observation_per_row(tmp_path):
    """B's rows must enter at n=1 whatever the row's `n_rollouts` says.

    Under `Σ NLL / Σ n` a row's gradient magnitude is `(q − target)/B` regardless of n, so B and C
    pull EQUALLY HARD and only the target differs — which is what makes C−B a read of label
    PRECISION rather than of effective learning rate. Feeding B the row's n=8 instead would leave
    the loss on the same per-rollout scale and the pin would have to be on the value, so it is: B's
    loss is computed against `ones`, and against `n_rollouts` it would differ.
    """
    model = _build_cf_ppo()
    heads = _attach_cf_twin_heads(model)
    model.learn(total_timesteps=8 * 4)
    # outcome 0.5 → round(0.5*1)=0 wins at n=1, but round(0.5*8)=4 wins at n=8. Different loss.
    _attach_cf_twin_buffer(model, tmp_path, label=1.0, outcome=0.5)
    model._cf_buffer.poll(0)
    model.cf_twin_coef = 1.0
    ctx = model._cf_sample_and_forward()
    _term, m = model._cf_twin_terms(ctx)
    logits = heads[0](ctx.value_pooled.detach()).flatten()
    at_one = float(InstrumentedMaskablePPO._cf_binomial_nll(
        logits, ctx.batch.outcome, th.ones_like(ctx.batch.outcome)))
    at_n = float(InstrumentedMaskablePPO._cf_binomial_nll(
        logits, ctx.batch.outcome, ctx.batch.n_rollouts))
    assert m["b_loss"] == pytest.approx(at_one, rel=1e-6)
    assert abs(at_one - at_n) > 1e-6, "preconditions: the two weightings must be distinguishable"


def test_head_B_is_skipped_and_COUNTED_when_no_row_carries_an_outcome(tmp_path):
    """The starvation case, and the one way this arm produces a confident wrong answer.

    A producer that ships no `outcome_label` trains B on nothing; B then equals A, and the C−B
    contrast silently becomes C−A while every other scalar reads healthy. So B's fold is skipped
    (never trained on a zero-filled absent label) and `b_coverage` publishes the fact.
    """
    model = _build_cf_ppo()
    _attach_cf_twin_heads(model)
    model.learn(total_timesteps=8 * 4)
    _attach_cf_buffer(model, tmp_path)          # the OLD fixture — no outcome_label at all
    model._cf_buffer.poll(0)
    model.cf_twin_coef = 1.0
    ctx = model._cf_sample_and_forward()
    _term, m = model._cf_twin_terms(ctx)
    assert m["b_coverage"] == 0.0
    assert "b_loss" not in m, "head B was folded on rows that carry no outcome label"
    assert "c_loss" in m, "head C must still train — its stream is present"
    # The headline still publishes when B starves, and it is C's fold ALONE — not C plus a zero.
    assert m["loss"] == pytest.approx(m["c_loss"], rel=1e-6)


def test_the_twin_block_publishes_a_COMBINED_headline_loss(tmp_path):
    """`train/cf_twin_loss` exists, and it is the sum of the arms that ACTUALLY folded.

    The twin block contributes ONE `loss = loss + term` to the optimizer, so it owes one
    `train/*` headline like every sibling cf term (`cf_loss`, `cf_evidential_loss`,
    `cf_shadow_loss`) — it published only a `grad_share` and no loss at all until this test.

    Summed inside the term rather than in the logger, and that is the substance of the pin: B
    skips a starved minibatch entirely, so the two arms' per-minibatch lists have DIFFERENT
    lengths and a downstream `mean(c) + mean(b)` would be the mean of no minibatch that ever
    folded. Here both arms run, so the combined value must equal `c_loss + b_loss` exactly.
    """
    model = _build_cf_ppo()
    _attach_cf_twin_heads(model)
    model.learn(total_timesteps=8 * 4)
    _attach_cf_twin_buffer(model, tmp_path, label=1.0, outcome=0.0)
    model._cf_buffer.poll(0)
    model.cf_twin_coef = 1.0
    ctx = model._cf_sample_and_forward()
    _term, m = model._cf_twin_terms(ctx)
    assert "loss" in m, "the twin block published no headline loss"
    assert m["b_coverage"] == pytest.approx(1.0), "preconditions: both arms must have folded"
    assert m["loss"] == pytest.approx(m["c_loss"] + m["b_loss"], rel=1e-6)
    # UNWEIGHTED, like every sibling `loss` key — the coefficient is a separate reading, and
    # baking it in would make the scalar move when only the dosage changed.
    assert m["loss"] != pytest.approx(model.cf_twin_coef * 0.0), "degenerate: loss is zero"


def test_the_onpolicy_mirror_uses_head_As_coefficient_and_detaches(tmp_path):
    """B and C must carry a BIT-IDENTICAL copy of head A's own loss, at `win_prob_coef`.

    If the mirror rode `cf_twin_coef` instead, B−A would confound "extra states" with "a different
    base objective" and the factorial would decompose nothing. Checked as a unit on the term, since
    the tiny PPO's obs dict carries no win-prob labels.
    """
    model = _build_cf_ppo()
    heads = _attach_cf_twin_heads(model)
    fe = model.policy.features_extractor
    pooled = th.randn(6, 8, requires_grad=True)
    fe.last_value_pooled = pooled

    class _RD:
        observations = {"win_target": th.tensor([[1.0], [0.0], [1.0], [1.0], [0.0], [1.0]]),
                        "win_mask": th.ones(6, 1)}
    model.win_prob_coef = 0.25
    term, m = model._cf_twin_onpolicy_terms(_RD())
    assert term is not None and "b_onpolicy_loss" in m and "c_onpolicy_loss" in m
    want = 0.25 * sum(
        float(InstrumentedMaskablePPO._win_prob_loss(
            h(pooled.detach()), _RD.observations["win_target"],
            _RD.observations["win_mask"])[0])
        for h in heads)
    assert float(term) == pytest.approx(want, rel=1e-6)
    # The trunk must be untouchable from here: head-only ALWAYS.
    term.backward()
    assert pooled.grad is None, "the on-policy mirror leaked a gradient into the trunk"


def test_shadow_critic_at_coefficient_zero_is_byte_identical(tmp_path):
    model = _build_cf_ppo()
    _attach_cf_shadow_head(model)
    init_sd, init_opt, base = _cf_twin_baseline(model, tmp_path)
    _attach_cf_twin_buffer(model, tmp_path, mc_return=1.5)
    model.cf_shadow_coef = 0.0
    off = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    for k in base:
        assert th.equal(base[k], off[k]), f"cf_shadow_coef=0 with a populated buffer perturbed {k}"


def test_a_live_shadow_coefficient_reaches_ONLY_the_shadow_head(tmp_path):
    """The shadow is a PROMOTION PATH, not surgery: it must be provably incapable of moving the
    critic it is being compared against."""
    model = _build_cf_ppo()
    _attach_cf_shadow_head(model)
    init_sd, init_opt, base = _cf_twin_baseline(model, tmp_path, unclipped=True)
    _attach_cf_twin_buffer(model, tmp_path, mc_return=1.5)
    model.cf_shadow_coef = 5.0
    on = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    shadow_keys = [k for k in base if "cf_shadow_head" in k]
    assert shadow_keys
    assert any(not th.equal(base[k], on[k]) for k in shadow_keys), "the shadow never trained"
    for k in base:
        if "cf_shadow_head" not in k:
            assert th.equal(base[k], on[k]), f"the shadow critic leaked a gradient into {k}"


def test_shadow_term_is_masked_and_reads_the_popart_frame(tmp_path):
    """Two facts in one: rows with no `mc_return` are EXCLUDED (not supervised toward zero, which
    is the middle of this reward's range and the most plausible-looking wrong target available),
    and the loss is computed in the PopArt-NORMALIZED frame the value loss trains in."""
    model = _build_cf_ppo()
    head = _attach_cf_shadow_head(model)
    model.learn(total_timesteps=8 * 4)
    _attach_cf_twin_buffer(model, tmp_path, mc_return=2.0)
    model._cf_buffer.poll(0)
    model.cf_shadow_coef = 1.0
    ctx = model._cf_sample_and_forward()

    class _PopArt:
        sigma, mu = 2.0, 1.0
        def normalize(self, x): return (x - self.mu) / self.sigma
        def denormalize(self, x): return x * self.sigma + self.mu

    _term, m = model._cf_shadow_term(ctx, _PopArt())
    pred = head(ctx.value_pooled.detach()).flatten()
    want = float(((pred - _PopArt().normalize(ctx.batch.mc_return)) ** 2).mean())
    assert m["loss"] == pytest.approx(want, rel=1e-6)
    assert m["coverage"] == pytest.approx(1.0)
    # `pred_mean` is the DE-normalized read — the only frame a human can interpret.
    assert m["pred_mean"] == pytest.approx(float((pred * 2.0 + 1.0).mean()), rel=1e-6)
    assert m["label_mean"] == pytest.approx(2.0, rel=1e-6)

    # And with no mc_return anywhere: no term, and a coverage of 0 that SAYS so. A FRESH directory
    # — the twin fixture above is still on disk and its rows do carry one.
    _attach_cf_buffer(model, tmp_path / "bare")
    model._cf_buffer.poll(0)
    ctx2 = model._cf_sample_and_forward()
    term2, m2 = model._cf_shadow_term(ctx2, _PopArt())
    assert term2 is None and m2["coverage"] == 0.0


def test_all_four_cf_terms_share_ONE_sample_and_ONE_forward(tmp_path):
    """Two samples would pay twice for the block's whole cost AND — far worse — make the arms
    disagree about which states they scored, so a 'paired' difference would not be paired."""
    model = _build_cf_ppo()
    _attach_cf_evid_head(model)
    _attach_cf_twin_heads(model)
    _attach_cf_shadow_head(model)
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)
    _attach_cf_twin_buffer(model, tmp_path, mc_return=1.0)
    model.cf_winprob_coef = 1.0
    model.cf_evidential_coef = 1.0
    model.cf_twin_coef = 1.0
    model.cf_shadow_coef = 1.0

    calls = []
    real = InstrumentedMaskablePPO._cf_sample_and_forward

    def spy(self):
        calls.append(1)
        return real(self)

    model.__class__ = type("_Spy4", (type(model),), {"_cf_sample_and_forward": spy})
    _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    assert calls, "preconditions: the CF block never ran"
    # Every minibatch samples exactly once no matter how many consumers are live.
    n_minibatches = len(calls)
    model.cf_evidential_coef = model.cf_twin_coef = model.cf_shadow_coef = 0.0
    calls.clear()
    _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    assert len(calls) == n_minibatches, "the number of cf forwards depends on the consumer count"


def test_the_only_coupling_between_a_headonly_term_and_the_trunk_is_the_GLOBAL_CLIP(tmp_path):
    """A HEAD-ONLY term still perturbs every other parameter — through `max_grad_norm`, not through
    the trunk. Pinned because the twin-heads arm's central claim is "identical trunk".

    `clip_grad_norm_` scales EVERY gradient by `max_norm / total_norm`, and `total_norm` is taken
    over all parameters — so adding any term with a non-zero gradient anywhere changes the factor
    applied to the policy and value gradients too. It is tiny (a last-bits effect at a sane
    coefficient) and it is shared by every aux this tree runs, but it is NOT zero, and an arm that
    claims a bit-identical trunk has to know which of the two mechanisms it is claiming.

    The demonstration is the pair: with the clip ACTIVE the updates differ; with the clip raised out
    of the way and NOTHING else changed, they are bit-identical. That difference is the proof the
    detach holds — a genuine gradient leak would survive both.
    """
    def _run(max_norm):
        model = _build_cf_ppo()
        _attach_cf_twin_heads(model)
        model.max_grad_norm = max_norm
        init_sd = copy.deepcopy(model.policy.state_dict())
        init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
        model.learn(total_timesteps=8 * 4)
        model._cf_buffer, model.cf_winprob_coef, model.cf_twin_coef = None, 0.0, 0.0
        base = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
        _attach_cf_twin_buffer(model, tmp_path / f"n{max_norm}")
        model.cf_twin_coef = 5.0
        on = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
        return [k for k in base
                if "cf_twin_head" not in k and not th.equal(base[k], on[k])]

    assert _run(0.5), "preconditions: a binding clip must couple the term to the other params"
    assert _run(1e9) == [], "a head-only term moved a non-twin parameter with the clip inactive"


# ---------------------------------------------------------------------------------------------
# gen3_policy_grad_coef_v1 (--policy-grad-coef): the policy-gradient term's own weight.
# Provenance genre (recorded / _resolve-inherited / never gated) is pinned in
# agents/model/policy_grad_coef_provenance_test.py; here is the loss-fold behavior itself.
# ---------------------------------------------------------------------------------------------

def test_policy_grad_coef_class_default_is_one():
    assert InstrumentedMaskablePPO.policy_grad_coef == 1.0


def test_policy_grad_coef_one_short_circuits_to_the_unscaled_policy_loss():
    """The byte-identity claim, pinned at the SOURCE: at the 1.0 default the fold takes the
    `policy_loss` tensor ITSELF (not `1.0 * policy_loss`, a new graph node), so the loss
    expression is literally the pre-flag `loss = policy_loss + …` — identical bits, identical
    graph, identical backward."""
    import inspect

    src = inspect.getsource(InstrumentedMaskablePPO.train)
    assert "_policy_grad_term = policy_loss if policy_grad_coef == 1.0 else policy_grad_coef * policy_loss" in src, (
        "the 1.0 short-circuit is gone — --policy-grad-coef's default is no longer structurally "
        "byte-identical to upstream")


def test_policy_grad_coef_zero_removes_exactly_the_policy_gradient():
    """policy_grad_coef=0 (with ent_coef=0, the tiny harness default): the parameters reached ONLY by the
    policy-gradient term — the action head and the mlp extractor's policy branch — are
    BIT-unchanged from init (their gradients are exactly zero, and Adam's zero-state step on a
    zero gradient is exactly zero), while the value path keeps training. That is the arm-F
    contract: every other term survives, PPO's own policy pull alone is gone."""
    model, _ = _build_tiny_ppo(n_steps=8, n_envs=4)
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)

    model.policy_grad_coef = 0.0
    after = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)

    policy_only = [k for k in after
                   if k.startswith(("action_net", "mlp_extractor.policy_net"))]
    value_side = [k for k in after
                  if k.startswith(("value_net", "mlp_extractor.value_net"))]
    assert policy_only and value_side, "policy layout drifted — fix the prefixes"
    for k in policy_only:
        assert th.equal(after[k], init_sd[k]), (
            f"policy_grad_coef=0 still moved {k} — the policy-gradient term was not exactly removed")
    assert any(not th.equal(after[k], init_sd[k]) for k in value_side), (
        "policy_grad_coef=0 froze the value path too — it must scale ONLY policy_loss")


def test_policy_grad_coef_between_zero_and_one_is_live():
    """A non-default value must actually reach the fold — 0.5 produces a different update than
    the 1.0 default on the same init/data/seed."""
    model, _ = _build_tiny_ppo(n_steps=8, n_envs=4)
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)

    base = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    model.policy_grad_coef = 0.5
    scaled = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    assert any(not th.equal(base[k], scaled[k]) for k in base), (
        "policy_grad_coef=0.5 left every parameter unchanged — the coefficient never reached the loss")


# ---------------------------------------------------------------------------------------------
# gen3_grad_distill_share_v1: `grad/distill_share` — the exploiter-distillation KL's own
# shared-trunk gradient share, the §6.2 dose meter of
# designs/ai_v10/design_advantage_gated_distillation.md. The toy policy's CombinedExtractor has
# no parameters, so `shared_trunk_parameters` is monkeypatched to the mlp extractor's — the
# probe's own math (grad_balance.py) is pinned in grad_balance_test.py; these pin the FOLD.
# ---------------------------------------------------------------------------------------------

def _patch_toy_trunk(monkeypatch, model):
    # `train_setup`, not `ppo` — the trunk is resolved once per train() in `_train_probe_setup`,
    # so that is the module whose global the probe reads.
    from agents.training.instrumented_ppo import train_setup as setup_mod
    monkeypatch.setattr(
        setup_mod, "shared_trunk_parameters",
        lambda fe, _m=model: [p for p in _m.policy.mlp_extractor.parameters()
                              if p.requires_grad])


def test_grad_distill_share_is_published_when_the_distill_term_is_live(monkeypatch):
    model, teacher = _build_distill_ppo(n_steps=8, n_envs=4)
    model._distill_teachers = [teacher]
    model.distill_coef = 5.0
    _patch_toy_trunk(monkeypatch, model)
    model.learn(total_timesteps=8 * 4)
    model.train()
    logged = model.logger.name_to_value
    assert "grad/distill_share" in logged, f"missing (got {sorted(k for k in logged if k.startswith('grad/'))})"
    assert 0.0 < logged["grad/distill_share"] < 1.0
    # Same denominator family as every other share — the property dose-matching relies on.
    assert "grad/policy_share" in logged and "grad/distill_policy_cosine" in logged


def test_grad_distill_share_absent_when_distill_is_off(monkeypatch):
    """coef 0 ⇒ no `grad/distill_share` (not logged, per the design's inactive semantics) — but
    the probe itself still runs, so its absence is the term's absence, not the probe's."""
    model, teacher = _build_distill_ppo(n_steps=8, n_envs=4)
    model._distill_teachers = [teacher]
    model.distill_coef = 0.0
    _patch_toy_trunk(monkeypatch, model)
    model.learn(total_timesteps=8 * 4)
    model.train()
    logged = model.logger.name_to_value
    assert "grad/policy_share" in logged, "the probe itself failed to run"
    assert "grad/distill_share" not in logged


def test_grad_distill_share_telemetry_does_not_change_the_update(monkeypatch):
    """TELEMETRY ONLY: the probe (including the new distill entry) is read-only autograd.grad —
    the same init/data/seed produce bit-identical parameters with the probe sampling and with it
    disabled entirely. The one property the feature must never lose."""
    from agents.training.instrumented_ppo import train_setup as setup_mod

    model, teacher = _build_distill_ppo(n_steps=8, n_envs=4)
    model._distill_teachers = [teacher]
    model.distill_coef = 5.0
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)

    _patch_toy_trunk(monkeypatch, model)
    on = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    assert "grad/distill_share" in model.logger.name_to_value, (
        "precondition: the probe (with the distill entry) must actually have sampled")

    monkeypatch.setattr(setup_mod, "shared_trunk_parameters", lambda fe: [])  # probe fully off
    off = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    for k in on:
        assert th.equal(on[k], off[k]), f"the grad probe perturbed {k} — telemetry only!"
