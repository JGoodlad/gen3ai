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
    carrying one poisons every CPU worker that loads it with a GPU context (or, for the cf label
    buffer, fails to pickle at all on its threading.Lock)."""
    excluded = InstrumentedMaskablePPO._excluded_save_params(
        InstrumentedMaskablePPO.__new__(InstrumentedMaskablePPO))
    for name in ("_cf_buffer", "_capacity_state"):
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


def test_the_onpolicy_mirror_uses_head_As_own_weight_and_detaches(tmp_path):
    """B and C must carry a BIT-IDENTICAL copy of head A's own loss, at head A's own weight (1.0).

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
    term, m = model._cf_twin_onpolicy_terms(_RD())
    assert term is not None and "b_onpolicy_loss" in m and "c_onpolicy_loss" in m
    want = sum(
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


def test_shadow_term_is_masked_and_reads_real_units(tmp_path):
    """Two facts in one: rows with no `mc_return` are EXCLUDED (not supervised toward zero, which
    is the middle of this reward's range and the most plausible-looking wrong target available),
    and the loss is computed in real return units (PopArt, which once normalized it, is deleted)."""
    model = _build_cf_ppo()
    head = _attach_cf_shadow_head(model)
    model.learn(total_timesteps=8 * 4)
    _attach_cf_twin_buffer(model, tmp_path, mc_return=2.0)
    model._cf_buffer.poll(0)
    model.cf_shadow_coef = 1.0
    ctx = model._cf_sample_and_forward()

    _term, m = model._cf_shadow_term(ctx)
    pred = head(ctx.value_pooled.detach()).flatten()
    want = float(((pred - ctx.batch.mc_return) ** 2).mean())
    assert m["loss"] == pytest.approx(want, rel=1e-6)
    assert m["coverage"] == pytest.approx(1.0)
    assert m["pred_mean"] == pytest.approx(float(pred.mean()), rel=1e-6)
    assert m["label_mean"] == pytest.approx(2.0, rel=1e-6)

    # And with no mc_return anywhere: no term, and a coverage of 0 that SAYS so. A FRESH directory
    # — the twin fixture above is still on disk and its rows do carry one.
    _attach_cf_buffer(model, tmp_path / "bare")
    model._cf_buffer.poll(0)
    ctx2 = model._cf_sample_and_forward()
    term2, m2 = model._cf_shadow_term(ctx2)
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
