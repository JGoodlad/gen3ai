"""Tests for the TD-consistency auxiliary (gen3_td_consistency_aux_v1).

Four properties, in the order they can silently break:
  1. coef 0.0 is BYTE-IDENTICAL to today — asserted on a REAL `train()`, and additionally by
     proving the sampler is never even reached (a broken sampler cannot perturb an off run).
  2. the residual math is exactly `V(s_t) − r_t − γ·V(s_{t+1})` on a hand-built case, in the units
     the value loss trains in (PopArt's σ, or 1.0).
  3. a pair whose successor begins a new episode is DROPPED, not zeroed.
  4. the term actually deposits gradient on the CRITIC parameters (a loss that trains nothing is
     the failure mode every aux in this tree has shipped at least once).
"""

import copy

import numpy as np
import pytest
import torch as th

from agents.training import td_aux
from agents.training.instrumented_ppo import InstrumentedMaskablePPO
from agents.training.instrumented_ppo_test import _build_tiny_ppo
from agents.training.td_aux import (
    TD_AUX_SEG_LEN,
    TD_AUX_STATES,
    sample_contiguous_pairs,
    td_aux_loss,
    td_residual,
)


# --------------------------------------------------------------------------------- the sampler


def _starts(n_steps, n_envs, boundaries=()):
    """`episode_starts` with row 0 of every env a start, plus the given (t, env) boundaries."""
    ep = np.zeros((n_steps, n_envs), dtype=np.float32)
    ep[0, :] = 1.0
    for t, e in boundaries:
        ep[t, e] = 1.0
    return ep


def test_pairs_are_contiguous_in_time_and_env_major_flat():
    """The returned rows follow `swap_and_flatten`'s ENV-MAJOR convention (row = env·n_steps + t),
    and a pair's two rows are adjacent IN TIME within one env. Getting this wrong would pair states
    from different battles while every shape check still passed."""
    n_steps, n_envs = 8, 3
    rows, pa, pb, _ = sample_contiguous_pairs(
        _starts(n_steps, n_envs), n_states=48, seg_len=4, rng=np.random.default_rng(7))
    envs, ts = rows // n_steps, rows % n_steps
    assert envs.max() < n_envs and ts.max() < n_steps
    assert np.array_equal(envs[pa], envs[pb]), "a pair must not span two env columns"
    assert np.array_equal(ts[pb], ts[pa] + 1), "a pair must be (t, t+1)"


def test_pairs_spanning_an_episode_boundary_are_dropped_not_zeroed():
    """A successor that BEGINS an episode is not a transition. It must vanish from the pair set —
    zeroing it would regress V(s_t) toward r_t at every battle end."""
    n_steps, n_envs = 8, 1
    ep = _starts(n_steps, n_envs, boundaries=[(4, 0)])
    rows, pa, pb, n_cand = sample_contiguous_pairs(
        ep, n_states=8, seg_len=8, rng=np.random.default_rng(0))
    ts = rows % n_steps
    # exactly one segment covering t=0..7 → 7 candidates, the (3,4) pair dropped.
    assert n_cand == 7 and pa.size == 6
    assert 4 not in set(ts[pb].tolist()), "the row that STARTS an episode is never a successor"
    assert set(ts[pa].tolist()) == {0, 1, 2, 4, 5, 6}

    # …and with no boundary the same draw keeps all 7.
    _, pa_clean, _, n_clean = sample_contiguous_pairs(
        _starts(n_steps, n_envs), n_states=8, seg_len=8, rng=np.random.default_rng(0))
    assert n_clean == 7 and pa_clean.size == 7


def test_an_all_boundary_column_yields_no_pairs():
    """Degenerate case (every row starts an episode): zero pairs, and the caller's loss returns
    None rather than a fabricated 0.0."""
    rows, pa, pb, n_cand = sample_contiguous_pairs(
        np.ones((8, 1), dtype=np.float32), n_states=8, seg_len=8, rng=np.random.default_rng(0))
    assert pa.size == 0 and n_cand == 7 and rows.size == 8
    out = td_aux_loss(th.zeros(8), th.zeros(8), th.as_tensor(pa), th.as_tensor(pb), 0.99)
    assert out is None


def test_segment_economy_serves_l_minus_one_pairs_per_l_forwards():
    """The 'K+1 contiguous forwards serve K pairs' economy the pre-registration asks for: with no
    boundaries, S forwarded states yield S − n_segments pairs, i.e. ~1.07 forwards per pair at the
    shipped seg_len — not the 2 a random-pair sampler would need."""
    rows, pa, _, _ = sample_contiguous_pairs(
        _starts(64, 4), n_states=TD_AUX_STATES, seg_len=TD_AUX_SEG_LEN,
        rng=np.random.default_rng(1))
    n_seg = TD_AUX_STATES // TD_AUX_SEG_LEN
    assert rows.size == TD_AUX_STATES
    assert pa.size == TD_AUX_STATES - n_seg
    assert rows.size / pa.size < 1.1


def test_sampler_rejects_a_flattened_episode_starts_array():
    """`episode_starts` is NOT in RolloutBuffer.get()'s flatten list; a 1-D array means the caller
    flattened it and every (env, t) decode below would be wrong. Fail loud."""
    with pytest.raises(ValueError, match=r"\[n_steps, n_envs\]"):
        sample_contiguous_pairs(np.zeros(32, dtype=np.float32), 8, 4, np.random.default_rng(0))


# ------------------------------------------------------------------------------ the residual math


def test_residual_math_on_a_hand_built_case():
    """δ_t = V(s_t) − r_t − γ·V(s_{t+1}), exactly, on values/rewards with known arithmetic."""
    values = th.tensor([10.0, 4.0, 1.0])
    rewards = th.tensor([3.0, 2.0, 0.0])
    gamma = 0.5
    pa, pb = th.tensor([0, 1]), th.tensor([1, 2])
    got = td_residual(values, rewards, pa, pb, gamma)
    # 10 - 3 - 0.5*4 = 5 ;  4 - 2 - 0.5*1 = 1.5
    assert th.allclose(got, th.tensor([5.0, 1.5]))
    loss, m = td_aux_loss(values, rewards, pa, pb, gamma, n_candidate=2)
    assert loss.item() == pytest.approx((25.0 + 2.25) / 2)
    assert m["resid_mean"] == pytest.approx(3.25)
    assert m["resid_rms"] == pytest.approx(((25.0 + 2.25) / 2) ** 0.5)
    assert m["n_pairs"] == 2 and m["pair_drop_frac"] == pytest.approx(0.0)


def test_scale_expresses_the_residual_in_the_value_losss_space():
    """UNITS. Under PopArt the value loss trains in normalized space, and
    `normalize(V) − normalize(r + γV′)` is exactly `(V − r − γV′)/σ` (the μ cancels) — so `scale=σ`
    is the normalized-space residual, and λ keeps rung-1's meaning. σ=1 with PopArt off."""
    from agents.model.popart import PopArtNormalizer
    pop = PopArtNormalizer()
    with th.no_grad():                      # plant a non-trivial (mu, sigma)
        pop.mu.fill_(7.0)
        pop.sigma.fill_(4.0)
    values = th.tensor([10.0, 4.0])
    rewards = th.tensor([3.0, 0.0])
    gamma = 0.5
    pa, pb = th.tensor([0]), th.tensor([1])
    raw = td_residual(values, rewards, pa, pb, gamma, scale=1.0)
    scaled = td_residual(values, rewards, pa, pb, gamma, scale=float(pop.sigma))
    target = rewards[pa] + gamma * values[pb]
    assert th.allclose(scaled, pop.normalize(values[pa]) - pop.normalize(target))
    assert th.allclose(scaled, raw / 4.0)


def test_both_residual_ends_carry_gradient():
    """The residual-gradient (Baird) form the pre-registration specifies: V(s_t) AND V(s_{t+1}) are
    both live. A detached successor would make this a semi-gradient TD and change what it measures."""
    values = th.tensor([2.0, 5.0, 1.0], requires_grad=True)
    loss, _ = td_aux_loss(values, th.zeros(3), th.tensor([0]), th.tensor([1]), 0.9)
    g, = th.autograd.grad(loss, values)
    assert g[0] != 0.0 and g[1] != 0.0, "both ends must receive gradient"
    assert g[2] == 0.0


# ------------------------------------------------------ integration: a REAL InstrumentedMaskablePPO


def _train_once(model, init_sd, init_opt, coef, seed=99):
    # DEEPCOPY at load: `Optimizer.load_state_dict` installs the passed tensors themselves (the
    # scalar `step` among them), so `optimizer.step()` would mutate the captured init in place and
    # the "same starting point" would silently drift between calls — which is exactly the kind of
    # difference a byte-identity test would then misreport as the feature's doing.
    model.policy.load_state_dict(copy.deepcopy(init_sd))
    model.policy.optimizer.load_state_dict(copy.deepcopy(init_opt))
    model.td_aux_coef = coef
    model._td_aux_rng = None
    np.random.seed(seed)
    th.manual_seed(seed)
    model.train()
    return {k: v.detach().clone() for k, v in model.policy.state_dict().items()}


def _tiny_with_rollout():
    model, _venv = _build_tiny_ppo(n_steps=8, n_envs=4)
    model.learn(total_timesteps=8 * 4)      # fills model.rollout_buffer
    return model


def test_coef_zero_is_byte_identical_to_the_current_loss(monkeypatch):
    """OFF must be EXACTLY today's training step. Proven two ways: the parameter update is
    bit-identical to a run with the attribute at its class default, and the sampler is never called
    at all (monkeypatched to raise) — so no future change to the sampler can perturb an off run."""
    model = _tiny_with_rollout()
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())

    baseline = _train_once(model, init_sd, init_opt, coef=InstrumentedMaskablePPO.td_aux_coef)

    def _boom(*a, **k):
        raise AssertionError("the TD-aux sampler must not run at coef 0")
    monkeypatch.setattr(td_aux, "sample_contiguous_pairs", _boom)
    off = _train_once(model, init_sd, init_opt, coef=0.0)

    for k in baseline:
        assert th.equal(baseline[k], off[k]), f"param {k} moved with the TD aux OFF"


def test_class_default_is_off():
    assert InstrumentedMaskablePPO.td_aux_coef == 0.0


def test_coef_positive_changes_the_update_and_logs_its_metrics():
    """ON must actually do something — and say so. A term that folds into the loss but leaves the
    parameters untouched, or that never emits a metric, is indistinguishable from OFF in every plot."""
    model = _tiny_with_rollout()
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())

    off = _train_once(model, init_sd, init_opt, coef=0.0)
    on = _train_once(model, init_sd, init_opt, coef=3.0)
    assert any(not th.equal(off[k], on[k]) for k in off), "coef 3.0 left every parameter unchanged"

    logged = model.logger.name_to_value
    for key in ("td_aux/loss", "td_aux/resid_rms", "td_aux/resid_mean",
                "td_aux/n_pairs", "td_aux/pair_drop_frac", "td_aux/scale"):
        assert key in logged, f"missing {key}"
    assert logged["td_aux/n_pairs"] > 0
    assert logged["td_aux/scale"] == 1.0          # PopArt off in the tiny harness


def test_the_term_deposits_gradient_on_the_critic():
    """The whole point is to train the CRITIC. Take the term's gradient on the value head + the
    shared trunk directly, on the real buffer, through the real `_td_aux_term`."""
    model = _tiny_with_rollout()
    model.td_aux_coef = 1.0
    model._td_aux_rng = None
    # get() flattens the buffer on the first next(); _td_aux_term requires that (and says so).
    next(iter(model.rollout_buffer.get(4)))
    term, metrics = model._td_aux_term(None)
    assert term is not None and metrics["n_pairs"] > 0
    critic = list(model.policy.value_net.parameters())
    grads = th.autograd.grad(term, critic, allow_unused=True)
    assert any(g is not None and float(g.abs().sum()) > 0.0 for g in grads), (
        "the TD term produced no gradient on value_net")


def test_the_coef_is_recorded_and_inherited_on_a_flagless_resume():
    """`training_coef` class: recorded on ModelVersion for PROVENANCE and — the half that actually
    changes behaviour — so `train_rl_agent`'s `_resolve` can read it back when a launcher restart
    forwards no `--td-aux-coef`. `_resolve` does `getattr(saved_version, name, default)`, so a coef
    that is NOT a ModelVersion field silently reverts to 0.0 on every restart."""
    import json
    from agents.model.model_version import (MODEL_CONFIG_VERSION, ModelVersion,
                                            ModelVersionError, _migrate_config)
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    pk = {"net_arch": [512, 512]}
    v = ModelVersion.from_layout_and_policy_kwargs(layout, pk, td_aux_coef=3.0)
    assert v.td_aux_coef == 3.0
    assert ModelVersion(**json.loads(v.to_json())).td_aux_coef == 3.0   # survives the round trip
    # …and it is NOT compared by check_compatible (a frozen eval/pool opponent never trains, so
    # gating it there would be a false rejection that breaks league play).
    other = ModelVersion.from_layout_and_policy_kwargs(layout, pk, td_aux_coef=0.0)
    other.check_compatible(v)          # must not raise

    # THE MIGRATION LEG IS NOW UNREACHABLE, and asserting the refusal is the honest form.
    # v96 (`gen3_critic_route_wave_v1`) bumped ARCH_SIGNATURE, which the floor contract requires
    # be matched by a floor raise in the same commit — so MIGRATION_FLOOR is 96 and EVERY
    # `if version < N` branch, including v92's `setdefault("td_aux_coef", 0.0)`, sits below it.
    # A config that lacks the field can therefore only be a pre-generation one, and the floor
    # refuses it outright rather than defaulting a field into a checkpoint whose weights this
    # code cannot load anyway. A test may not claim to cover a branch the floor makes
    # unreachable (the v90 precedent), so what is asserted is the behaviour: refusal with a
    # diagnosis. Every config at or above the floor carries `td_aux_coef` explicitly, which is
    # what makes the flagless-resume read-back above the property that still matters.
    pre_v92 = json.loads(v.to_json())
    pre_v92.pop("td_aux_coef")
    pre_v92["config_version"] = 91
    with pytest.raises(ModelVersionError, match="PRE-GENERATION|floor"):
        _migrate_config(pre_v92)
    assert MODEL_CONFIG_VERSION >= 92


def test_it_refuses_to_run_before_the_buffer_is_flattened():
    """The pair rows are in the post-`swap_and_flatten` convention; using them on an un-flattened
    buffer would silently mis-pair states with rewards on any n_envs > 1."""
    model, _venv = _build_tiny_ppo(n_steps=8, n_envs=4)
    model.learn(total_timesteps=8 * 4)
    model.rollout_buffer.generator_ready = False
    model.td_aux_coef = 1.0
    with pytest.raises(RuntimeError, match="swap_and_flatten"):
        model._td_aux_term(None)
