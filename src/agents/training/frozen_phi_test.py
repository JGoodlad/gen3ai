"""`agents/training/frozen_phi.py` — the ACTOR-ONLY frozen-phi construction.

THE FOUR CLAIMS, and each one is measured rather than asserted about a call:

1. **THE TELESCOPING IDENTITY.** On a scripted episode with `phi(terminal) := 0`, the per-episode
   discounted shaping sum is exactly `-coef*phi(s_0)`, and at `lambda = 1` the per-ROW advantage
   difference is exactly `-coef*phi(s_t)`. The second is the stronger form and is what
   `signal/adv_shaped_minus_unshaped_mean` publishes.
2. **THE CRITIC TARGET IS UNCHANGED**, with and without the flag, on the same buffer — bit-identical
   `returns` AND a bit-identical value loss computed from them.
3. **THE ADVANTAGE DIFFERS BY EXACTLY THE POTENTIAL DELTA** — nothing else moved.
4. **`rewards` come back exactly**, by assignment rather than by subtraction (`(a+b)-b != a` in
   float32).

The buffers here are REAL `MaskableDictRolloutBuffer`s driven by a fake model, because the claims
are about what `compute_returns_and_advantage` does with the arrays — a hand-rolled GAE in the test
would be a second implementation agreeing with itself.

Run:
    pytest src/agents/training/frozen_phi_test.py -q
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer

from agents.training.frozen_phi import (
    FROZEN_PHI_COEF,
    advantage_shaping_delta,
    apply_frozen_phi_shaping,
)
from agents.training.winprob_pbrs import (
    WinProbPbrsError,
    episode_shaping_sum,
    pbrs_shaping,
    successor_potential,
)

OBS_DIM = 3


# ──────────────────────────────────────────────────────────────────────────────────────────────
# a real buffer, a fake model, and a phi that is a FIXED FUNCTION OF STATE
# ──────────────────────────────────────────────────────────────────────────────────────────────

def _phi_of(obs: np.ndarray) -> np.ndarray:
    """The frozen potential, as a deterministic function of the observation — which is what makes
    it a POTENTIAL at all. `sigmoid(mean(obs))`, so it lands in (0, 1) like `sigmoid(logit)`."""
    x = np.asarray(obs, dtype=np.float64).reshape(-1, OBS_DIM).mean(axis=1)
    return 1.0 / (1.0 + np.exp(-x))


class _FakeFE:
    def __init__(self):
        self.last_win_prob_logits = None


class _FakePolicy:
    """`predict_values` returns a constant critic and stashes the frozen phi's LOGIT, exactly as a
    real extractor's side stash does — so `winprob_pbrs._forward_phi` reads it unchanged."""

    def __init__(self, v: float = 0.25):
        self.features_extractor = _FakeFE()
        self._v = v

    def predict_values(self, obs):
        x = obs["observation"] if isinstance(obs, dict) else obs
        x = th.as_tensor(np.asarray(x, dtype=np.float32))
        p = th.as_tensor(_phi_of(x.numpy()), dtype=th.float32)
        # logit(p) — the stash a real WinProbHead leaves.
        self.features_extractor.last_win_prob_logits = th.log(p / (1.0 - p)).reshape(-1, 1)
        return th.full((x.shape[0], 1), self._v)


class _FakeSource:
    """The FROZEN phi network. `winprob_pbrs.phi_model` returns this instead of the model, and
    `_phi_obs` filters the obs to its declared space — both exercised for real here."""

    def __init__(self, device="cpu"):
        self.policy = _FakePolicy()
        self.device = device
        self.observation_space = spaces.Dict(
            {"observation": spaces.Box(-10.0, 10.0, shape=(OBS_DIM,), dtype=np.float32)})


class _FakeModel:
    def __init__(self, buf, *, gamma=1.0, on=True, coef=FROZEN_PHI_COEF, terminal_scale=1.0):
        self.rollout_buffer = buf
        self.gamma = gamma
        self.device = "cpu"
        self.policy = _FakePolicy()
        self._frozen_phi_on = on
        self.frozen_phi_coef = coef
        self.win_prob_pbrs_terminal_scale = terminal_scale
        self._winprob_phi_source = _FakeSource()
        self._last_obs = None
        self._last_episode_starts = None


def _build_buffer(*, obs, rewards, episode_starts, values, gamma, gae_lambda):
    """A REAL `MaskableDictRolloutBuffer` filled row by row, then GAE'd — the same object and the
    same method the collector hands to `train()`."""
    n_steps, n_envs = rewards.shape
    space = spaces.Dict(
        {"observation": spaces.Box(-10.0, 10.0, shape=(OBS_DIM,), dtype=np.float32)})
    buf = MaskableDictRolloutBuffer(
        n_steps, space, spaces.Discrete(2), device="cpu",
        gamma=gamma, gae_lambda=gae_lambda, n_envs=n_envs)
    for t in range(n_steps):
        buf.add(
            {"observation": obs[t].astype(np.float32)},
            np.zeros((n_envs,), dtype=np.int64),
            rewards[t].astype(np.float32),
            episode_starts[t].astype(np.float32),
            th.as_tensor(values[t].reshape(n_envs, 1), dtype=th.float32),
            th.zeros(n_envs),
            action_masks=np.ones((n_envs, 2), dtype=np.int8),
        )
    return buf


def _scripted(*, n_steps=12, n_envs=1, ep_len=4, gamma=1.0, gae_lambda=1.0, seed=0):
    """A buffer whose episodes ALL start and end inside it — so every telescoping claim below is
    exact rather than bootstrap-approximate. `n_steps` must divide by `ep_len`."""
    assert n_steps % ep_len == 0
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(n_steps, n_envs, OBS_DIM)).astype(np.float32)
    rewards = rng.normal(size=(n_steps, n_envs)).astype(np.float32)
    values = rng.normal(size=(n_steps, n_envs)).astype(np.float32)
    episode_starts = np.zeros((n_steps, n_envs), dtype=np.float32)
    episode_starts[::ep_len] = 1.0
    buf = _build_buffer(obs=obs, rewards=rewards, episode_starts=episode_starts,
                        values=values, gamma=gamma, gae_lambda=gae_lambda)
    model = _FakeModel(buf, gamma=gamma)
    # The post-rollout state: the row AFTER the buffer is a fresh episode start, so every episode
    # in the buffer is COMPLETE and no truncation bootstrap fires.
    model._last_obs = {"observation": rng.normal(size=(n_envs, OBS_DIM)).astype(np.float32)}
    model._last_episode_starts = np.ones((n_envs,), dtype=np.float32)
    buf.compute_returns_and_advantage(
        last_values=th.zeros((n_envs, 1)), dones=np.ones((n_envs,), dtype=np.float32))
    return model, obs, rewards, episode_starts


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 1. THE TELESCOPING IDENTITY
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_the_per_episode_shaping_sum_is_exactly_minus_coef_phi_of_the_START_state():
    """`sum_k gamma^k * (gamma*phi(s_{k+1}) - phi(s_k)) = gamma^T phi(s_T) - phi(s_0) = -phi(s_0)`
    under `phi(terminal) := 0`. This is the theorem's whole content: the sum depends only on the
    endpoints, so it adds the same constant to every policy's return from a given start state."""
    _model, obs, _r, episode_starts = _scripted(gamma=1.0)
    phi = _phi_of(obs).reshape(obs.shape[0], obs.shape[1])
    phi_next = successor_potential(phi, episode_starts, np.zeros(obs.shape[1]),
                                   np.ones(obs.shape[1]))
    shaping = pbrs_shaping(phi, phi_next, 1.0, FROZEN_PHI_COEF)
    for start, _end, total in episode_shaping_sum(shaping, episode_starts, 1.0, env_index=0):
        assert total == pytest.approx(-FROZEN_PHI_COEF * phi[start, 0], abs=1e-9)


def test_at_lambda_1_the_advantage_delta_is_exactly_minus_coef_phi_of_THAT_ROW():
    """The per-ROW form, which is the stronger claim and the one the published diagnostic reads:
    at lambda = 1 the GAE sum telescopes within the episode, so `A' - A = -coef*phi(s_t)` on every
    row of a COMPLETE episode. That makes the shaped advantage the unshaped one minus a function of
    the STATE ALONE — a state-dependent baseline, i.e. a zero-bias modification of the policy
    gradient."""
    model, obs, _r, _es = _scripted(gamma=1.0, gae_lambda=1.0)
    adv_before = np.array(model.rollout_buffer.advantages, copy=True)
    out = apply_frozen_phi_shaping(model, model.rollout_buffer)
    delta = advantage_shaping_delta(model.rollout_buffer.advantages, adv_before)
    phi = _phi_of(obs).reshape(obs.shape[0], obs.shape[1])
    np.testing.assert_allclose(delta, -FROZEN_PHI_COEF * phi, atol=2e-5)
    # ...and the published mean is that quantity's mean, so a reader can check the identity live.
    assert out["signal/adv_shaped_minus_unshaped_mean"] == pytest.approx(
        float((-FROZEN_PHI_COEF * phi).mean()), abs=2e-5)
    assert out["pbrs/frozen_phi_mean"] == pytest.approx(float(phi.mean()), abs=1e-6)


def test_the_published_mean_equals_minus_coef_times_the_published_phi_mean():
    """The one-glance production audit the metrics section describes: read
    `signal/adv_shaped_minus_unshaped_mean` beside `pbrs/frozen_phi_mean` and the terminal
    convention is either holding on real episodes or it is not."""
    model, _obs, _r, _es = _scripted(gamma=1.0, gae_lambda=1.0)
    out = apply_frozen_phi_shaping(model, model.rollout_buffer)
    assert out["signal/adv_shaped_minus_unshaped_mean"] == pytest.approx(
        -out["pbrs/frozen_phi_coef"] * out["pbrs/frozen_phi_mean"], abs=2e-5)


def test_a_terminal_row_gets_phi_next_ZERO_so_no_outcome_can_leak():
    """`phi(terminal) := 0` does double duty, and this is the second job: a frozen head evaluated
    at a terminal state could be read as a prediction of the result that state just revealed.
    Forcing 0 makes the last transition's shaping `-coef*phi(s_{T-1})` — a function of the state
    the agent ACTED in, never of what happened."""
    phi = np.array([[0.2], [0.7], [0.9], [0.3]])
    episode_starts = np.array([[1.0], [0.0], [1.0], [0.0]])   # row 1 ends an episode
    nxt = successor_potential(phi, episode_starts, np.zeros(1), np.ones(1))
    assert nxt[1, 0] == 0.0, "the successor of a terminal row must be 0, not phi(s')"
    assert nxt[0, 0] == pytest.approx(0.7)
    assert nxt[3, 0] == 0.0, "the last row ends an episode here too"


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 2. THE CRITIC TARGET IS UNCHANGED — the claim that makes this ACTOR-ONLY
# ──────────────────────────────────────────────────────────────────────────────────────────────

def _value_loss(buf) -> float:
    """The scalar-MSE diagnostic `train()` computes, read straight off the buffer's own arrays —
    the same `rollout_data.returns` the minibatch loop sees."""
    r = np.asarray(buf.returns, dtype=np.float64)
    v = np.asarray(buf.values, dtype=np.float64)
    return float(((r - v) ** 2).mean())


def test_the_value_target_is_BIT_IDENTICAL_with_and_without_the_shaping():
    """THE ACTOR-ONLY CLAIM. `returns` is what the scalar-MSE diagnostic, `explained_variance` and
    `value_scale_metrics` read; under `--critic winprob` the real value loss is the head's BCE
    against `win_target`, which never touches this buffer at all. Either way the shaping must not
    reach it."""
    off, _o, _r, _e = _scripted(seed=7)
    on, _o2, _r2, _e2 = _scripted(seed=7)
    returns_before = np.array(off.rollout_buffer.returns, copy=True)
    loss_before = _value_loss(off.rollout_buffer)

    apply_frozen_phi_shaping(on, on.rollout_buffer)

    np.testing.assert_array_equal(on.rollout_buffer.returns, returns_before)
    assert _value_loss(on.rollout_buffer) == loss_before
    assert _value_loss(on.rollout_buffer) == _value_loss(off.rollout_buffer)


def test_the_REWARDS_come_back_exactly_and_by_assignment():
    """`(a + b) - b` is not `a` in float32, so the restore is an ASSIGNMENT from a snapshot. A
    subtract-back implementation drifts here in the low bits, on the stream every later consumer
    reads as the run's true reward."""
    model, _o, rewards, _e = _scripted(seed=11)
    before = np.array(model.rollout_buffer.rewards, copy=True)
    apply_frozen_phi_shaping(model, model.rollout_buffer)
    np.testing.assert_array_equal(model.rollout_buffer.rewards, before)
    np.testing.assert_array_equal(model.rollout_buffer.rewards, rewards.astype(np.float32))


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 3. THE ADVANTAGE DIFFERS BY EXACTLY THE POTENTIAL DELTA — and nothing else moved
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_the_advantage_is_the_only_array_that_moves():
    model, _o, _r, _e = _scripted(seed=3)
    buf = model.rollout_buffer
    snap = {k: np.array(getattr(buf, k), copy=True)
            for k in ("rewards", "returns", "values", "episode_starts")}
    adv_before = np.array(buf.advantages, copy=True)

    apply_frozen_phi_shaping(model, buf)

    for k, v in snap.items():
        np.testing.assert_array_equal(getattr(buf, k), v, err_msg=f"{k} moved")
    assert not np.allclose(buf.advantages, adv_before), "the advantages must have moved"


def test_the_advantage_delta_matches_a_GAE_over_the_pure_SHAPING_stream():
    """At any lambda, `A' - A` is the GAE of a stream whose reward is the shaping term and whose
    value baseline is zero — because GAE is affine in the rewards. Built with the SAME real buffer
    class rather than a hand-rolled recursion, so the check is against SB3's implementation."""
    gamma, lam = 0.97, 0.9
    model, obs, _r, episode_starts = _scripted(gamma=gamma, gae_lambda=lam, seed=5)
    adv_before = np.array(model.rollout_buffer.advantages, copy=True)
    apply_frozen_phi_shaping(model, model.rollout_buffer)
    delta = advantage_shaping_delta(model.rollout_buffer.advantages, adv_before)

    phi = _phi_of(obs).reshape(obs.shape[0], obs.shape[1])
    phi_next = successor_potential(phi, episode_starts, np.zeros(obs.shape[1]),
                                   np.ones(obs.shape[1]))
    shaping = pbrs_shaping(phi, phi_next, gamma, FROZEN_PHI_COEF)
    shaping_only = _build_buffer(obs=obs, rewards=shaping, episode_starts=episode_starts,
                                 values=np.zeros_like(shaping), gamma=gamma, gae_lambda=lam)
    shaping_only.compute_returns_and_advantage(
        last_values=th.zeros((obs.shape[1], 1)),
        dones=np.ones((obs.shape[1],), dtype=np.float32))
    np.testing.assert_allclose(delta, shaping_only.advantages, atol=2e-5)


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 4. OFF, and the refusals
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_off_returns_no_metrics_and_touches_nothing():
    model, _o, _r, _e = _scripted()
    model._frozen_phi_on = False
    snap = {k: np.array(getattr(model.rollout_buffer, k), copy=True)
            for k in ("rewards", "returns", "advantages")}
    assert apply_frozen_phi_shaping(model, model.rollout_buffer) == {}
    for k, v in snap.items():
        np.testing.assert_array_equal(getattr(model.rollout_buffer, k), v)


def test_a_MISSING_frozen_source_is_a_LOUD_refusal_not_a_live_phi():
    """The SELF-phi double counting this flag exists to avoid: with no frozen source
    `winprob_pbrs.phi_model` falls back to the live model, whose win-prob head IS the critic under
    this mode — so `gamma*phi(s') - phi(s)` would be the TD residual GAE already computes."""
    model, _o, _r, _e = _scripted()
    model._winprob_phi_source = None
    with pytest.raises(WinProbPbrsError, match="SELF"):
        apply_frozen_phi_shaping(model, model.rollout_buffer)


def test_the_coefficient_is_the_currency_matched_ONE_and_is_published():
    """Derived, not chosen: under `--critic winprob` the terminal is the win indicator at
    `--victory-value 1.0`, so V is P(win) and phi = sigmoid(logit) is already one unit of V per
    unit of V. Scaling it linearly scales the whole delta, which is what makes it a coefficient
    rather than a mode."""
    assert FROZEN_PHI_COEF == 1.0
    model, _o, _r, _e = _scripted(gamma=1.0, gae_lambda=1.0, seed=13)
    adv0 = np.array(model.rollout_buffer.advantages, copy=True)
    out = apply_frozen_phi_shaping(model, model.rollout_buffer)
    assert out["pbrs/frozen_phi_coef"] == 1.0
    d1 = advantage_shaping_delta(model.rollout_buffer.advantages, adv0)

    half, _o2, _r2, _e2 = _scripted(gamma=1.0, gae_lambda=1.0, seed=13)
    half.frozen_phi_coef = 0.5
    adv0b = np.array(half.rollout_buffer.advantages, copy=True)
    apply_frozen_phi_shaping(half, half.rollout_buffer)
    d2 = advantage_shaping_delta(half.rollout_buffer.advantages, adv0b)
    np.testing.assert_allclose(d2, 0.5 * d1, atol=2e-5)


def test_the_episode_dose_is_the_shaping_budget_as_a_fraction_of_a_WIN():
    """The sizing meter, and its denominator is a CONSTANT — the run's terminal magnitude — never
    the reward stream's own mean, which under the clean-world composition is terminal-only and
    therefore exactly zero on a rollout with no episode end."""
    model, obs, _r, episode_starts = _scripted(gamma=1.0, seed=17)
    out = apply_frozen_phi_shaping(model, model.rollout_buffer)
    phi = _phi_of(obs).reshape(obs.shape[0], obs.shape[1])
    starts = np.flatnonzero(episode_starts[:, 0] >= 0.5)
    # Every episode but the last is complete; the dose averages |-coef*phi(s_0)| over those.
    expected = float(np.mean(np.abs(-FROZEN_PHI_COEF * phi[starts[:-1], 0])))
    assert out["pbrs/frozen_phi_episode_dose"] == pytest.approx(expected, abs=1e-6)
    assert out["pbrs/frozen_phi_episode_dose_n"] == float(len(starts) - 1)


def test_the_dose_is_OMITTED_rather_than_divided_by_a_fictitious_terminal_scale():
    model, _o, _r, _e = _scripted()
    model.win_prob_pbrs_terminal_scale = 0.0
    out = apply_frozen_phi_shaping(model, model.rollout_buffer)
    assert "pbrs/frozen_phi_episode_dose" not in out
    assert "pbrs/frozen_phi_mean" in out, "the potential's own meter must still publish"


# ──────────────────────────────────────────────────────────────────────────────────────────────
# THE SEAM — the ONE place both rollout loops hand the buffer to train()
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_the_hook_is_in_collect_rollouts_which_BOTH_collectors_pass_through():
    """`InstrumentedMaskablePPO.collect_rollouts` wraps `collect_rollouts_async` AND
    `super().collect_rollouts`, so the async path is covered by construction rather than by a
    parallel implementation. Read off the source, because the property is structural.

    BOTH SEAMS LIVE IN `frozen_phi.py` and `ppo.py` carries one call each — the `distill_anchor.py`
    shape, taken because `ppo.py` sits AT the file-size ratchet's 2,000-line hard bound."""
    import inspect

    from agents.training.instrumented_ppo import InstrumentedMaskablePPO

    src = inspect.getsource(InstrumentedMaskablePPO.collect_rollouts)
    assert "collect_rollouts_async" in src and "super().collect_rollouts" in src
    assert "frozen_phi.shape_after_rollout(self, rollout_buffer, ok)" in src


def test_the_seam_gates_on_the_BOOLEAN_and_on_the_collectors_own_verdict():
    """The gate is the flag, never a coefficient comparison (the coefficient is a derived constant
    here, not a knob) — and a collector that did NOT complete leaves the buffer alone."""
    model, _o, _r, _e = _scripted()
    from agents.training import frozen_phi

    snap = np.array(model.rollout_buffer.advantages, copy=True)
    frozen_phi.shape_after_rollout(model, model.rollout_buffer, False)
    np.testing.assert_array_equal(model.rollout_buffer.advantages, snap)

    model._frozen_phi_on = False
    frozen_phi.shape_after_rollout(model, model.rollout_buffer, True)
    np.testing.assert_array_equal(model.rollout_buffer.advantages, snap)

    model._frozen_phi_on = True
    frozen_phi.shape_after_rollout(model, model.rollout_buffer, True)
    assert not np.allclose(model.rollout_buffer.advantages, snap)
    assert model._frozen_phi_metrics, "the seam must stash the metrics for the train() drain"


def test_the_metrics_reach_the_logger_under_their_OWN_prefixes():
    """They arrive already prefixed because they land in TWO groups — `pbrs/` for the potential and
    `signal/` for the telescoping term — so the drain records the keys VERBATIM."""
    import inspect

    from agents.training import frozen_phi
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO

    assert "frozen_phi.record_metrics(self, self.logger)" in inspect.getsource(
        InstrumentedMaskablePPO.train)

    class _Logger:
        def __init__(self):
            self.seen = {}

        def record(self, key, value):
            self.seen[key] = value

    class _M:
        _frozen_phi_metrics = {"pbrs/frozen_phi_mean": 0.5,
                               "signal/adv_shaped_minus_unshaped_mean": -0.5}

    log = _Logger()
    frozen_phi.record_metrics(_M(), log)
    assert log.seen == {"pbrs/frozen_phi_mean": 0.5,
                        "signal/adv_shaped_minus_unshaped_mean": -0.5}
    # A rollout that recorded nothing publishes nothing — never a defaulted zero.
    log2 = _Logger()

    class _Empty:
        _frozen_phi_metrics = None

    frozen_phi.record_metrics(_Empty(), log2)
    assert log2.seen == {}


def test_the_frozen_source_attribute_is_excluded_from_the_checkpoint():
    """A frozen FOREIGN model must never be pickled into our checkpoint — the `_distill_teacher`
    genre. The frozen-phi network rides `_winprob_phi_source`, which is already excluded; this
    pins that the SHARED attribute choice keeps that property."""
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO

    class _Bare(InstrumentedMaskablePPO):
        def __init__(self):  # noqa: D107 - not constructing a real PPO
            pass

    assert "_winprob_phi_source" in _Bare()._excluded_save_params()


# ──────────────────────────────────────────────────────────────────────────────────────────────
# BYTE-IDENTITY on the REAL PPO update — `--critic shaped`, and OFF generally
# ──────────────────────────────────────────────────────────────────────────────────────────────

def _arms():
    """The `_train_from_init` protocol `instrumented_ppo_winprob_critic_test` uses: capture the
    init BEFORE `learn()`, then replay `train()` from it once per arm. `atol=1e-7` matches those
    tests — two replays of one `train()` agree to the float32 noise floor, not bit-for-bit,
    because the optimizer state is restored rather than re-derived."""
    import copy

    from agents.training.instrumented_ppo_test import _build_tiny_ppo

    model, _ = _build_tiny_ppo(n_steps=8, n_envs=4)
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)          # one rollout to fill the buffer
    return model, init_sd, init_opt


def _same(a, b, why):
    for k in a:
        assert th.allclose(a[k], b[k], atol=1e-7), f"{why}: {k}"


def test_the_harness_is_reproducible_at_all():
    """The ANTI-VACUITY control: two replays of the same seeded `train()` from the same init must
    agree, or a byte-identity assertion below would pass by measuring nothing."""
    from agents.training.instrumented_ppo_test import _train_from_init

    model, sd, opt = _arms()
    a = _train_from_init(model, sd, opt, batch_size=4, accum=1)
    b = _train_from_init(model, sd, opt, batch_size=4, accum=1)
    _same(a, b, "the harness itself is not reproducible")


def test_the_shaped_update_is_unchanged_by_the_frozen_phi_attributes():
    """`--critic shaped` cannot reach this path at all (the flag is refused there) and OFF is the
    default everywhere, so every new conditional must reduce to the expression that was there. The
    explicit-False arm is the second half: the gate is a plain boolean, so a future "cleanup" that
    read a coefficient instead would fail here."""
    from agents.training.instrumented_ppo_test import _train_from_init

    model, sd, opt = _arms()
    baseline = _train_from_init(model, sd, opt, batch_size=4, accum=1)

    model._frozen_phi_on = False
    _same(baseline, _train_from_init(model, sd, opt, batch_size=4, accum=1),
          "an explicit False moved the update")

    model._frozen_phi_metrics = None
    model.frozen_phi_coef = 0.0
    _same(baseline, _train_from_init(model, sd, opt, batch_size=4, accum=1),
          "the frozen-phi attributes moved the update")


def test_the_metrics_ride_the_logger_and_NOT_the_loss():
    """A populated metrics dict must reach TensorBoard and leave the parameter update
    bit-identical — the shaping edits the BUFFER between collection and `train()`, so it has no
    per-minibatch existence and can have no effect inside the loop."""
    from agents.training.instrumented_ppo_test import _train_from_init

    model, sd, opt = _arms()
    baseline = _train_from_init(model, sd, opt, batch_size=4, accum=1)

    model._frozen_phi_metrics = {"pbrs/frozen_phi_mean": 0.42,
                                 "signal/adv_shaped_minus_unshaped_mean": -0.42}
    after = _train_from_init(model, sd, opt, batch_size=4, accum=1)
    _same(baseline, after, "logging a metric moved the update")
    assert model.logger.name_to_value["pbrs/frozen_phi_mean"] == pytest.approx(0.42)
    assert model.logger.name_to_value[
        "signal/adv_shaped_minus_unshaped_mean"] == pytest.approx(-0.42)
