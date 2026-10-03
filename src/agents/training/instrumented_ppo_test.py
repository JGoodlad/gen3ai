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
    _verify_upstream_unchanged,
)
import torch as th


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
    from agents.training.rust_rollout.testkit import ToyVecEnv

    from agents.training.rust_rollout.testkit import attach_vec_collector
    venv = ToyVecEnv([(lambda: _CounterDictEnv()) for _ in range(n_envs)])
    model = InstrumentedMaskablePPO(
        "MultiInputPolicy", venv,
        n_steps=n_steps, batch_size=4, n_epochs=1,
        normalize_advantage=False,   # per-micro-batch adv-norm is the ONE non-identity; remove it for an exact check
        ent_coef=0.0, vf_coef=0.5, device="cpu", seed=0,
    )
    attach_vec_collector(model)   # the rollout comes from the toy VecEnv (the Rust collector is production's)
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


@pytest.mark.parametrize("micro,accum,full", [(4, 4, 16), (8, 2, 16), (5, 3, 15), (4, 3, 12)])
def test_grad_accum_matches_full_batch(micro, accum, full):
    """accum=K over micro-batches of size B reproduces the parameter update of accum=1 over a single
    (B·K)=`full` batch — the BIT-EXACT-gradient guarantee (with normalize_advantage off so this
    isolates the accumulation math; empirically max|Δ|~3e-8, the float32 noise floor). The 32-sample
    buffer with micro∈{4,8} divides cleanly (all groups = K equal-size micros); micro=5 (→ a size-2
    trailing group that is a single micro) exercises the partial-group rescale and is still exact.
    (4, 3, 12) is the RUST COLLECTOR's shape (M5 Lane G): every micro-batch FULL (the update target is
    a multiple of lcm(batch_size, n_envs)), the last optimizer step a SHORT group of two full micros —
    the live 98,304 / 65,536 = 1.5 case — and its step equals the unaccumulated step over the same rows,
    i.e. normalised by the step's REAL rows, with no padding and no second graph shape."""
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
    carrying one poisons every CPU worker that loads it with a GPU context."""
    excluded = InstrumentedMaskablePPO._excluded_save_params(
        InstrumentedMaskablePPO.__new__(InstrumentedMaskablePPO))
    for name in ("_capacity_state",):
        assert name in excluded, f"{name} would be pickled into every checkpoint"


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

    from agents.training.instrumented_ppo.micro_step import micro_step  # K8: fold steps 1-3a (R1)
    src = inspect.getsource(micro_step)
    assert "pg_term = policy_loss if st.policy_grad_coef == 1.0 else st.policy_grad_coef * policy_loss" in src, (
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
