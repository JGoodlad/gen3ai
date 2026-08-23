"""Gates for `gen3_capacity_telemetry_v1` — the three live saturation early-warnings.

FOUR properties carry this feature, and three of them are things a docstring cannot assert:

1. **The synthetic target family is REPRODUCIBLE**, across processes and across devices. A
   deferred OFFLINE probe is meant to use the same family so the two instruments cross-validate,
   and "the same family" is only a claim if the seeds are pinned somewhere executable. These tests
   pin the exact arithmetic, not just its determinism.

2. **The canary's gradient NEVER reaches the trunk.** Measured on the actual parameter update and
   on `.grad`, on a REAL `MaskablePPO`, with a LIVE graph deliberately handed to it — the same
   standard `cf_head_only` is held to. A `.detach()` call is not evidence.

3. **The cosine probe corrupts NOTHING.** It runs two extra backwards' worth of work; if it used
   `backward()` instead of `autograd.grad` it would silently add to the accumulated gradient the
   optimizer is about to step on. `.grad` AND the optimizer's `state_dict` must be byte-identical
   across a measurement.

4. **OFF is byte- and cost-identical.** The parameter update matches an off run exactly, and no
   probe state (head, optimizer, projection, probe batch) is built at all.

Pure/CPU, milliseconds, unmarked — the fast inner loop runs all of it.
"""
import copy
import types

import numpy as np
import pytest
import torch as th

from agents.training import capacity_telemetry as cap
from agents.training.instrumented_ppo_test import (   # the shared tiny-PPO harness
    _build_tiny_ppo,
    _train_from_init,
)


# =============================================================== 1. THE SYNTHETIC TARGET FAMILY

def test_the_seed_formula_is_exactly_the_documented_one():
    """`seed(k, e) = 20260823 + k + 1_000_000*e`. Pinned literally: an offline probe that wants to
    reproduce a run's canary has to be able to read the numbers off this file."""
    assert cap.CANARY_SEED_BASE == 20260823
    assert cap.CANARY_RESEED_STRIDE == 1_000_000
    assert cap.CANARY_K == 4
    assert cap.canary_seed(0, 0) == 20260823
    assert cap.canary_seed(3, 0) == 20260826
    assert cap.canary_seed(1, 2) == 20260823 + 1 + 2_000_000


def test_a_projection_column_is_reproducible_and_device_independent():
    """Same `(k, e)` -> the same column, every time, in any process. The generator is CPU-seeded
    precisely so a CUDA run's canary curve is comparable to a laptop's replay of it."""
    a = cap.canary_projection_column(2, 0, 64)
    b = cap.canary_projection_column(2, 0, 64)
    assert th.equal(a, b)
    assert a.shape == (64,) and a.device.type == "cpu"
    # …and distinct targets / distinct reseeds are genuinely different draws.
    assert not th.equal(a, cap.canary_projection_column(3, 0, 64))
    assert not th.equal(a, cap.canary_projection_column(2, 1, 64))


def test_the_target_is_the_documented_closed_form_and_is_bounded():
    """`target_k = tanh(obs @ P[:, k] / sqrt(obs_dim))` — checked against the formula written out
    by hand, so a 'harmless' refactor of the expression fails here."""
    th.manual_seed(0)
    obs = th.randn(7, 32) * 5.0
    P = th.stack([cap.canary_projection_column(k, 0, 32) for k in range(cap.CANARY_K)], dim=1)
    got = cap.canary_targets(obs, P)
    assert got.shape == (7, cap.CANARY_K)
    for k in range(cap.CANARY_K):
        expect = th.tanh(obs @ P[:, k] / (32 ** 0.5))
        # atol rather than exact: a matrix-matrix and a matrix-vector product take different BLAS
        # kernels and so different summation orders. The FORMULA is what is pinned here.
        assert th.allclose(got[:, k], expect, atol=1e-6)
    # tanh bounds it, which is what keeps `canary_recovery` a ratio of two comparable scales.
    assert bool((got.abs() < 1.0).all())


def test_a_reset_reseeds_exactly_one_column_round_robin():
    """The reset is the instrument. One target moves; the other three must be untouched, or the
    'recovery' being measured is contaminated by three simultaneous re-fits."""
    canary = _canary(feature_dim=6, obs_dim=16, reset_steps=100)
    before = canary.projection.clone()
    assert canary.maybe_reset(0) is False, "the first call only ARMS the clock"
    assert canary.maybe_reset(50) is False
    assert canary.maybe_reset(100) is True
    assert canary.reset_target == 0 and canary.reseeds == [1, 0, 0, 0]
    assert not th.equal(canary.projection[:, 0], before[:, 0])
    for k in (1, 2, 3):
        assert th.equal(canary.projection[:, k], before[:, k]), f"column {k} moved on target 0's reset"
    assert canary.maybe_reset(200) is True and canary.reset_target == 1
    assert canary.maybe_reset(300) is True and canary.reset_target == 2
    assert canary.maybe_reset(400) is True and canary.reset_target == 3
    assert canary.maybe_reset(500) is True and canary.reset_target == 0, "not round-robin"
    assert canary.reseeds == [2, 1, 1, 1]


def _canary(*, feature_dim=6, obs_dim=16, reset_steps=1_000_000):
    th.manual_seed(7)
    return cap.PlasticityCanary(feature_dim, obs_dim, reset_steps=reset_steps)


# ============================================================================ 2. THE CANARY ITSELF

def test_the_canary_actually_fits_its_targets():
    """A probe that cannot learn measures nothing. Feed a fixed (features, obs) pair whose features
    determine the obs, and the loss must fall substantially."""
    th.manual_seed(3)
    obs = th.randn(64, 16)
    features = obs[:, :6].clone()                 # the features genuinely carry obs information
    canary = _canary()
    canary.step(features, obs)
    first = float(np.mean([v for v in canary.ema if v is not None]))
    for _ in range(300):
        canary.step(features, obs)
    last = float(np.mean([v for v in canary.ema if v is not None]))
    assert last < 0.5 * first, f"the canary head does not learn ({first:.4f} -> {last:.4f})"


def test_recovery_is_the_post_over_pre_ratio_of_the_reset_target():
    """`canary_recovery` is ONE number and it must be exactly the ratio it claims — a re-seeded
    target's current EMA over what it had settled to before the reset."""
    canary = _canary(reset_steps=1)
    th.manual_seed(5)
    obs, features = th.randn(32, 16), th.randn(32, 6)
    canary.maybe_reset(0)                          # arm
    for _ in range(50):
        canary.step(features, obs)
    pre = canary.ema[0]
    assert canary.maybe_reset(10) is True and canary.reset_target == 0
    assert canary.pre_reset_loss == pre
    canary.step(features, obs)
    m = canary.metrics(10)
    assert m["canary_recovery"] == pytest.approx(canary.ema[0] / pre)
    assert m["canary_age"] == 0.0 and m["canary_resets"] == 1.0


def test_metrics_are_empty_before_any_step():
    """ABSENT, never zero: a canary that has not run must publish no `canary_loss`, because 0.0
    on a TB chart is indistinguishable from a perfectly-fitted one."""
    assert _canary().metrics(0) == {}


def test_the_canary_gradient_never_reaches_the_trunk():
    """THE isolation gate, measured three ways on a REAL policy — not asserted about a detach call.

    The canary is handed a LIVE graph on purpose (`trunk(x)`, not `trunk(x).detach()`), because the
    detach lives inside `PlasticityCanary.step` and that is the line under test. After the step:
    every policy parameter's `.grad` is still None, every policy parameter is unchanged, and the
    canary's own parameters DID move (or the test proves nothing).
    """
    model, _ = _build_tiny_ppo()
    policy_params = list(model.policy.parameters())
    for p in policy_params:
        p.grad = None
    before = [p.detach().clone() for p in policy_params]

    th.manual_seed(1)
    trunk = th.nn.Linear(16, 6)
    obs = th.randn(24, 16)
    live_features = trunk(obs)                    # requires_grad=True, a real graph
    assert live_features.requires_grad

    canary = _canary()
    head_before = [p.detach().clone() for p in canary.head.parameters()]
    canary.step(live_features, obs)

    assert all(p.grad is None for p in policy_params), \
        "the canary wrote .grad on a POLICY parameter — its input is not detached"
    assert all(p.grad is None for p in trunk.parameters()), \
        "the canary's gradient reached the module that produced its input"
    for p, b in zip(policy_params, before):
        assert th.equal(p.detach(), b)
    assert any(not th.equal(p.detach(), b)
               for p, b in zip(canary.head.parameters(), head_before)), \
        "the canary's OWN parameters did not move — the isolation assertion is vacuous"


def test_the_canary_optimizer_holds_only_its_own_parameters():
    """The structural half of the same claim. The canary's Adam must not be able to name a policy
    parameter at all — that is what keeps it out of SB3's positional optimizer-state restore (the
    ai_v6_13 '128 vs 5' class)."""
    canary = _canary()
    owned = {id(p) for p in canary.head.parameters()}
    grouped = {id(p) for g in canary.opt.param_groups for p in g["params"]}
    assert grouped == owned and grouped


# ================================================================= 3. THE HALF-BATCH TRUNK COSINE
#
# TRUNK is `grad_balance.shared_trunk_parameters` — the existing allow-list
# ("embeddings", "pokemon_encoder", "team_transformer", "assembler"), reused rather than
# re-defined so the cosine and `grad/*_share` are talking about the same set of weights.

class _FakePolicy:
    """A policy whose `evaluate_actions` is one Linear, so a half-batch's gradient is controllable."""

    def __init__(self, in_dim=3):
        th.manual_seed(2)
        self.net = th.nn.Linear(in_dim, 1)

    def evaluate_actions(self, obs, actions, action_masks=None):
        v = self.net(obs["observation"]).flatten()
        # log_prob rides the same weights so BOTH halves of the surrogate carry trunk gradient.
        return v, v * 0.1, th.zeros_like(v)


def _fake_rollout(obs, advantages, returns, old_log_prob=None):
    n = obs.shape[0]
    return types.SimpleNamespace(
        observations={"observation": obs},
        action_masks=None,
        old_log_prob=th.zeros(n) if old_log_prob is None else old_log_prob,
        returns=returns,
        advantages=advantages,
    )


def test_two_identical_halves_agree_exactly():
    """Duplicate the same rows into both halves and the cosine must be 1.0. That is the probe's
    calibration point — anything else means the two halves are not being compared like for like."""
    model = types.SimpleNamespace(policy=_FakePolicy(), vf_coef=0.5)
    rows = th.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0], [3.0, 0.0, -1.0]])
    obs = th.cat([rows, rows])
    adv = th.tensor([1.0, -2.0, 0.5] * 2)
    ret = th.tensor([0.3, 1.2, -0.4] * 2)
    out = cap.halfbatch_trunk_cosine(
        model, _fake_rollout(obs, adv, ret), th.zeros(6, dtype=th.long), adv,
        list(model.policy.net.parameters()), clip_range=0.2)
    assert out["halfbatch_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert out["halfbatch_grad_norm_ratio"] == pytest.approx(1.0, abs=1e-6)


def test_opposed_halves_read_negative():
    """The alarm state has to be reachable, or a cosine that never goes negative proves nothing.
    Two halves with mirrored advantages pull the trunk in opposite directions."""
    model = types.SimpleNamespace(policy=_FakePolicy(), vf_coef=0.0)   # policy term only
    rows = th.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])
    obs = th.cat([rows, rows])
    adv = th.tensor([1.0, 2.0, -1.0, -2.0])
    ret = th.zeros(4)
    out = cap.halfbatch_trunk_cosine(
        model, _fake_rollout(obs, adv, ret), th.zeros(4, dtype=th.long), adv,
        list(model.policy.net.parameters()), clip_range=0.2)
    assert out["halfbatch_cosine"] < 0.0


def test_a_measurement_leaves_grad_and_optimizer_state_byte_identical():
    """THE no-corruption gate. The probe runs two extra backwards' worth of work in the middle of
    an accumulation group; `autograd.grad` is what keeps that off `.grad`. Both the accumulated
    gradient and the optimizer's own state must survive bit-for-bit."""
    model, _ = _build_tiny_ppo()
    model.learn(total_timesteps=8 * 4)
    params = list(model.policy.parameters())
    # Put a REAL accumulated gradient and REAL Adam state in place first — the interesting failure
    # is a probe that ADDS to an existing accumulation, which an all-None `.grad` would hide.
    obs = {k: th.as_tensor(v[:8]) for k, v in model.rollout_buffer.observations.items()}
    values, log_prob, _ = model.policy.evaluate_actions(
        obs, th.zeros(8, dtype=th.long), action_masks=th.ones((8, 2), dtype=th.int8))
    (values.sum() + log_prob.sum()).backward()
    model.policy.optimizer.step()
    values, log_prob, _ = model.policy.evaluate_actions(
        obs, th.zeros(8, dtype=th.long), action_masks=th.ones((8, 2), dtype=th.int8))
    (values.sum() + log_prob.sum()).backward()

    grads_before = [None if p.grad is None else p.grad.detach().clone() for p in params]
    opt_before = copy.deepcopy(model.policy.optimizer.state_dict())

    rd = types.SimpleNamespace(
        observations=obs, action_masks=th.ones((8, 2), dtype=th.int8),
        old_log_prob=th.zeros(8), returns=th.zeros(8), advantages=th.ones(8))
    out = cap.halfbatch_trunk_cosine(
        model, rd, th.zeros(8, dtype=th.long), rd.advantages, params, clip_range=0.2)
    assert "halfbatch_cosine" in out and -1.0 <= out["halfbatch_cosine"] <= 1.0

    for p, g in zip(params, grads_before):
        if g is None:
            assert p.grad is None
        else:
            assert th.equal(p.grad, g), "the cosine probe ADDED to the accumulated gradient"
    after = model.policy.optimizer.state_dict()
    assert after["param_groups"] == opt_before["param_groups"]
    for k, st in opt_before["state"].items():
        for name, v in st.items():
            new = after["state"][k][name]
            assert th.equal(v, new) if th.is_tensor(v) else v == new, \
                f"the cosine probe moved optimizer state {k}/{name}"


def test_an_unmeasurable_batch_returns_nothing_rather_than_a_fake_number():
    model = types.SimpleNamespace(policy=_FakePolicy(), vf_coef=0.5)
    rd = _fake_rollout(th.zeros(2, 3), th.zeros(2), th.zeros(2))
    assert cap.halfbatch_trunk_cosine(model, rd, th.zeros(2, dtype=th.long), th.zeros(2),
                                      [], clip_range=0.2) == {}
    assert cap.halfbatch_trunk_cosine(model, rd, th.zeros(2, dtype=th.long), th.zeros(2),
                                      list(model.policy.net.parameters()), clip_range=0.2) == {}


# ========================================================================== 4. THE FEATURE VELOCITY

def test_velocity_is_zero_and_cosine_one_for_an_unmoved_representation():
    v = th.randn(16, 8)
    m = cap.feature_velocity_metrics(v, v.clone())
    assert m["feature_velocity"] == pytest.approx(0.0, abs=1e-6)
    assert m["feature_velocity_cos"] == pytest.approx(1.0, abs=1e-5)
    assert m["feature_velocity_rel"] == pytest.approx(0.0, abs=1e-6)


def test_velocity_matches_the_hand_computation():
    prev = th.tensor([[3.0, 0.0], [0.0, 4.0]])
    cur = th.tensor([[3.0, 1.0], [0.0, 4.0]])
    m = cap.feature_velocity_metrics(cur, prev)
    assert m["feature_velocity"] == pytest.approx(0.5)          # (1.0 + 0.0) / 2
    assert m["feature_velocity_rel"] == pytest.approx(0.5 / 3.5)


def test_the_first_measurement_publishes_nothing():
    """There is no velocity without a previous point, and a 0.0 there would read as 'frozen'."""
    assert cap.feature_velocity_metrics(th.randn(4, 2), None) == {}


# ================================================================= 5. THE END-TO-END TRAIN() WIRING

class _Stash:
    def __init__(self):
        self.value_pooled = None


def _build_capacity_ppo(n_steps=8, n_envs=4):
    """A tiny PPO whose stub extractor exposes a `pokemon_encoder` (so `shared_trunk_parameters`
    finds a trunk) and the `last_value_pooled` stash the canary snapshots."""
    model, _ = _build_tiny_ppo(n_steps=n_steps, n_envs=n_envs)
    fe = model.policy.features_extractor
    th.manual_seed(11)
    fe.pokemon_encoder = th.nn.Linear(1, 6)     # a NAMED shared-trunk phase; in the state_dict
    fe.stash = _Stash()
    _base = type(fe)

    def _forward(self, obs):
        # `tanh`, not `relu`: the stub trunk must be guaranteed to CARRY GRADIENT, and a relu that
        # happens to be dead on this env's observations would make the cosine probe silently
        # measure a zero gradient and publish nothing — a green test that tests nothing.
        pooled = th.tanh(self.pokemon_encoder(obs["observation"]))
        self.stash.value_pooled = pooled
        if "action_mask" in obs:
            # Folded into the features the policy actually reads, so the PPO loss reaches
            # `pokemon_encoder` and `shared_trunk_parameters` names a trunk with a real gradient.
            return _base.forward(self, obs) + pooled.mean(dim=-1, keepdim=True)
        return pooled

    cls = type("_CapStubExtractor", (_base,), {
        "forward": _forward,
        "last_value_pooled": property(lambda self: self.stash.value_pooled),
    })
    fe.__class__ = cls
    model.policy.optimizer.add_param_group({"params": list(fe.pokemon_encoder.parameters())})
    return model


def _capacity_on(model, **kw):
    model.capacity_telemetry = True
    model.canary_reset_steps = kw.get("canary_reset_steps", 1)
    model.capacity_cosine_every = kw.get("capacity_cosine_every", 1)
    model.capacity_velocity_every = kw.get("capacity_velocity_every", 1)
    model._capacity_state = None


def test_every_scalar_reaches_the_logger():
    """The smoke: with the flag on, one `train()` publishes the whole `capacity/*` family."""
    model = _build_capacity_ppo()
    model.learn(total_timesteps=8 * 4)
    _capacity_on(model)
    model.train()
    model.train()             # the SECOND call is what gives feature_velocity a previous point
    recorded = {k for k in model.logger.name_to_value if k.startswith("capacity/")}
    for name in ("capacity/canary_loss", "capacity/canary_age", "capacity/canary_steps",
                 "capacity/canary_resets", "capacity/halfbatch_cosine",
                 "capacity/halfbatch_grad_norm_ratio", "capacity/feature_velocity",
                 "capacity/feature_velocity_cos", "capacity/feature_velocity_rel"):
        assert name in recorded, f"{name} never reached the logger (got {sorted(recorded)})"
    assert model.logger.name_to_value["capacity/canary_steps"] > 0, \
        "the canary never got a value_pooled snapshot — it would be silently measuring nothing"


def test_recovery_appears_once_a_reset_has_fired():
    model = _build_capacity_ppo()
    model.learn(total_timesteps=8 * 4)
    _capacity_on(model, canary_reset_steps=1)
    for _ in range(3):
        model.num_timesteps += 1000
        model.train()
    assert "capacity/canary_recovery" in model.logger.name_to_value
    assert model.logger.name_to_value["capacity/canary_resets"] >= 1.0


def test_off_is_byte_identical_and_holds_no_state():
    """The shipping contract. An OFF run's parameter update matches an OFF-with-nothing-attached
    run exactly, and — the COST half — nothing is built: no head, no optimizer, no projection
    matrix, no frozen probe batch."""
    model = _build_capacity_ppo()
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)

    model.capacity_telemetry = False
    model._capacity_state = None
    base = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    assert model._capacity() is None
    assert getattr(model, "_capacity_state", None) is None, "an OFF run built telemetry state"
    assert not any(k.startswith("capacity/") for k in model.logger.name_to_value)

    again = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    for k in base:
        assert th.equal(base[k], again[k])


def test_on_does_not_change_the_policy_update():
    """The stronger half of the same claim: turning the telemetry ON must leave every POLICY
    parameter update bit-identical. It folds no loss term and writes no `.grad`, so this is a
    property of the design rather than of a small coefficient."""
    model = _build_capacity_ppo()
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)

    model.capacity_telemetry = False
    model._capacity_state = None
    off = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)

    _capacity_on(model)
    on = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    assert model._capacity_state is not None and model._capacity_state.canary is not None
    for k in off:
        assert th.equal(off[k], on[k]), f"capacity telemetry perturbed the policy update at {k}"


def test_a_probe_failure_disables_the_telemetry_without_killing_the_run_or_lying(capsys):
    """A DIAGNOSTIC must never crash a 3-hour training window — and must never quietly rewrite the
    run's own provenance either. On failure it latches off, says so once on stderr, and leaves
    `capacity_telemetry` (the field `model_config.json` records) telling the truth about the launch.
    """
    model = _build_capacity_ppo()
    model.learn(total_timesteps=8 * 4)
    _capacity_on(model)
    model.train()
    assert model.logger.name_to_value["capacity/canary_loss"] >= 0.0

    def _boom(*a, **kw):
        raise RuntimeError("probe exploded")

    model._capacity_state.observe = _boom
    model.train()                                  # must NOT raise
    assert model._capacity_failed is True
    assert model._capacity() is None, "the failure did not latch — it would raise every minibatch"
    assert model.capacity_telemetry is True, \
        "the failure path rewrote the RECORDED flag; model_config.json would deny the launch"
    assert "telemetry DISABLED" in capsys.readouterr().err


def test_the_state_is_excluded_from_the_checkpoint():
    """Documented limitation, made a MEASUREMENT: the canary re-inits on resume because
    `_capacity_state` is excluded from the save. If that ever changed, an Adam optimizer and a
    frozen obs batch would start riding every checkpoint."""
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    excluded = InstrumentedMaskablePPO._excluded_save_params(
        InstrumentedMaskablePPO.__new__(InstrumentedMaskablePPO))
    assert "_capacity_state" in excluded


# ============================================================================ 6. THE FLAG SURFACE

_FLAGS = {
    "capacity_telemetry": (False, True, "--capacity-telemetry"),
    "canary_reset_steps": (1_000_000, 250_000, "--canary-reset-steps"),
    "capacity_cosine_every": (50, 7, "--capacity-cosine-every"),
    "capacity_velocity_every": (50, 9, "--capacity-velocity-every"),
}


@pytest.mark.parametrize("field", sorted(_FLAGS))
def test_the_flag_defaults_to_none_so_a_flagless_resume_inherits(field):
    from main.train.parser import build_parser
    assert getattr(build_parser().parse_args([]), field) is None


@pytest.mark.parametrize("field", sorted(_FLAGS))
def test_the_flag_has_a_resolve_line(field):
    import inspect
    import re

    from main.train.config import resolve_config

    names = set(re.findall(r"_resolve\(\s*\"([a-z0-9_]+)\"", inspect.getsource(resolve_config)))
    assert field in names


@pytest.mark.parametrize("field", sorted(_FLAGS))
def test_the_field_is_recorded_and_round_trips(field):
    import json

    from agents.model.model_version import ModelVersion
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    _default, other, _flag = _FLAGS[field]
    v = ModelVersion.from_layout_and_policy_kwargs(
        layout, {"net_arch": [512, 512]}, **{field: other})
    assert getattr(v, field) == other
    assert getattr(ModelVersion(**json.loads(v.to_json())), field) == other


def test_none_of_them_is_gated_by_check_compatible():
    """A frozen eval / pool / distill opponent runs no train step at all, so gating it on a
    train-step diagnostic would be a false rejection that breaks league play."""
    from agents.model.model_version import ModelVersion
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    pk = {"net_arch": [512, 512]}
    a = ModelVersion.from_layout_and_policy_kwargs(layout, pk)
    b = ModelVersion.from_layout_and_policy_kwargs(
        layout, pk, **{f: other for f, (_d, other, _c) in _FLAGS.items()})
    a.check_compatible(b)
    b.check_compatible(a)


def test_a_pre_v101_config_migrates_to_the_argparse_defaults():
    import json

    from agents.model.model_version import MODEL_CONFIG_VERSION, ModelVersion, _migrate_config
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    old = json.loads(ModelVersion.from_layout_and_policy_kwargs(
        layout, {"net_arch": [512, 512]}).to_json())
    for field in _FLAGS:
        old.pop(field)
    old["config_version"] = 100

    migrated = _migrate_config(old)
    assert migrated["config_version"] == MODEL_CONFIG_VERSION >= 101
    for field, (default, _other, _flag) in _FLAGS.items():
        assert migrated[field] == default, field
    ModelVersion(**migrated)


def test_the_startup_banner_only_speaks_when_the_flag_is_on():
    from agents.training.instrumented_ppo.capacity_terms import capacity_startup_banner

    assert capacity_startup_banner(types.SimpleNamespace(capacity_telemetry=False)) == ""
    line = capacity_startup_banner(types.SimpleNamespace(
        capacity_telemetry=True, canary_reset_steps=1_000_000,
        capacity_cosine_every=50, capacity_velocity_every=50))
    # The two counter-intuitive properties must be SAID, not merely documented.
    assert "NOT checkpointed" in line and "1,000,000" in line
