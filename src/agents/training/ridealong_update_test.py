"""THE ride-along guarantee, end to end (`gen3_ridealong_heads_v1`): heads ON learns what heads OFF learns.

On a REAL Gen3 extractor + dual-head policy at the production surface (win-prob critic, α/β intent
heads, every pointer cell), two models are built at the same seed — one with all four ride-along
heads, one without — each collects its rollout and runs ONE real `InstrumentedMaskablePPO.train()`.
Asserted, in order:

  1. BUILDING the heads perturbs nothing: the two policies' non-ride-along state_dicts are equal and
     the global RNG leaves construction in the same state (`fork_rng` + a private seed);
  2. the two rollouts are identical (the heads are not in the forward);
  3. after the update the policy, trunk and V parameters, the PPO optimizer's state, every
     per-update PPO scalar and the torch / numpy / python RNG states are BIT-IDENTICAL;
  4. the ON arm is not vacuous: nearly every policy parameter moved, EVERY ride-along head's
     parameters moved, and the `ridealong/*` meters were emitted.

It fails on revert: drop the `.detach()` in `RideAlongBatch.detached`, fold a ride-along loss into
PPO's `loss`, put the heads in `policy.optimizer`, or build them from the global RNG — each breaks
3 or 1 (the heads' gradient would reach the trunk, ride PPO's global grad-norm clip, or move the
stream every later draw reads).
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch as th

RIDEALONG_ON = {"ridealong_ensemble": 3, "ridealong_rnd": True, "ridealong_adv": 2,
                "ridealong_opp": 2}


def _env_cls(rows, masks):
    import gymnasium as gym
    from gymnasium import spaces

    from agents.action.constants import ACTION_SPACE_SIZE

    lab = lambda: spaces.Box(0, 10 ** 6, (1,), np.int64)   # noqa: E731

    class _Env(gym.Env):
        """The committed REAL-obs fixture rows, with every label key the heads (and the α/β
        intent and win-prob losses the production surface folds) read."""

        def __init__(self, seed):
            self.observation_space = spaces.Dict({
                "observation": spaces.Box(-1e6, 1e6, (rows.shape[1],), np.float32),
                "action_mask": spaces.Box(0, 1, (ACTION_SPACE_SIZE,), np.int8),
                "win_target": spaces.Box(0.0, 1.0, (1,), np.float32),
                "win_mask": spaces.Box(0.0, 1.0, (1,), np.float32),
                "win_margin": spaces.Box(-1.0, 1.0, (1,), np.float32),
                "opp_action_kind": lab(), "opp_action_num": lab(), "opp_switch_slot": lab(),
                "opp_switch_species": lab(), "opp_class": lab()})
            self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
            self._t, self._seed = 0, seed

        def _row(self):
            return (self._t * 5 + self._seed * 17) % len(rows)

        def _o(self):
            r = self._row()
            return {"observation": rows[r].astype(np.float32),
                    "action_mask": masks[r].astype(np.int8),
                    "win_target": np.array([float((r + self._seed) % 2)], np.float32),
                    "win_mask": np.array([1.0], np.float32),
                    "win_margin": np.array([0.0], np.float32),
                    # SWITCH (kind 1) on alternate rows: always inside α's support, so B has labels.
                    "opp_action_kind": np.array([self._t % 2], np.int64),
                    "opp_action_num": np.array([0], np.int64),
                    "opp_switch_slot": np.array([1], np.int64),
                    "opp_switch_species": np.array([0], np.int64),
                    "opp_class": np.array([self._seed % 2], np.int64)}

        def reset(self, **kw):
            self._t = 0
            return self._o(), {}

        def step(self, a):
            self._t += 1
            r = float(((int(a) + self._t + self._seed) % 5) - 2)
            return self._o(), r, self._t >= 3 + self._seed, False, {}

        def action_masks(self):
            return masks[self._row()].astype(bool)

    return _Env


def _rng_state():
    return th.get_rng_state().clone(), np.random.get_state(), random.getstate()


def _build(ridealong: dict):
    from stable_baselines3.common.vec_env import DummyVecEnv

    from agents.model.compile_parity_fixture import load_parity_rows
    from agents.model.parity_probe import perturb_
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    from main.fresh_checkpoint import _production_policy_kwargs

    _args, layout, pk = _production_policy_kwargs()
    pk = {**pk, "features_extractor_kwargs": {**pk["features_extractor_kwargs"], **ridealong}}
    rows, masks = load_parity_rows(layout["total_dim"])
    env_cls = _env_cls(rows, masks)
    th.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    model = InstrumentedMaskablePPO(
        Gen3DualHeadMaskablePolicy, DummyVecEnv([(lambda s=s: env_cls(s)) for s in range(2)]),
        n_steps=8, batch_size=4, n_epochs=2, device="cpu", seed=0, policy_kwargs=pk)
    built_rng = _rng_state()
    model.opp_intent_coef = 0.05        # production's dose: the α/β losses AND B's aligned labels
    perturb_(model.policy)              # private RNG; opens the zero-init paths (see diagnostics test)
    model.grad_accum_steps = 2
    orig = InstrumentedMaskablePPO.train
    InstrumentedMaskablePPO.train = lambda self: None
    try:
        model.learn(total_timesteps=8 * 2)
    finally:
        InstrumentedMaskablePPO.train = orig
    pre = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
    buf = {k: np.array(v, copy=True) for k, v in model.rollout_buffer.observations.items()}
    buf["__actions"] = np.array(model.rollout_buffer.actions, copy=True)
    buf["__advantages"] = np.array(model.rollout_buffer.advantages, copy=True)
    model.logger.name_to_value.clear()
    model.train()
    post = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
    opt = model.policy.optimizer.state_dict()
    opt_t = {(i, k): v.detach().clone() for i, st in opt["state"].items()
             for k, v in st.items() if th.is_tensor(v)}
    names = [n for n, _ in model.policy.named_parameters()]
    return dict(model=model, built_rng=built_rng, pre=pre, post=post, buf=buf, opt=opt_t,
                logged=dict(model.logger.name_to_value), rng=_rng_state(), names=names)


@pytest.fixture(scope="module")
def arms():
    return _build({}), _build(RIDEALONG_ON)


def _core(sd):
    return {k: v for k, v in sd.items() if not k.startswith("ridealong.")}


def test_building_the_heads_draws_nothing_from_the_global_rng(arms):
    off, on = arms
    assert th.equal(off["built_rng"][0], on["built_rng"][0]), "building the heads moved torch RNG"
    assert on["model"].policy.ridealong is not None and off["model"].policy.ridealong is None
    core_on = _core(on["pre"])
    assert core_on.keys() == off["pre"].keys()
    for k in core_on:
        assert th.equal(core_on[k], off["pre"][k]), f"building the heads changed {k}"


def test_the_heads_are_in_no_PPO_param_group_and_not_in_the_forward(arms):
    _, on = arms
    pol = on["model"].policy
    in_opt = {id(p) for g in pol.optimizer.param_groups for p in g["params"]}
    ra = list(pol.ridealong.parameters())
    assert ra and not any(id(p) in in_opt for p in ra)
    # Frozen priors / RND target are BUFFERS, never parameters.
    assert all(p.requires_grad for p in ra)
    assert any(k.startswith("ridealong.") and ".priors." in k for k in on["pre"])


def test_the_rollouts_are_identical(arms):
    off, on = arms
    assert off["buf"].keys() == on["buf"].keys()
    for k in off["buf"]:
        assert np.array_equal(off["buf"][k], on["buf"][k]), f"rollout key {k} differs"


def test_one_update_is_BIT_IDENTICAL_with_the_heads_on_and_off(arms):
    off, on = arms
    core_on = _core(on["post"])
    for k in core_on:
        assert th.equal(core_on[k], off["post"][k]), f"the ride-along heads changed {k}"
    assert off["opt"].keys() == on["opt"].keys() and off["opt"]
    for k in off["opt"]:
        assert th.equal(off["opt"][k], on["opt"][k]), f"the heads changed PPO optimizer state {k}"
    for tag, v in off["logged"].items():
        if tag.endswith("_ms"):           # wall-clock timings, not training quantities
            continue
        assert tag in on["logged"], tag
        assert on["logged"][tag] == v or (v != v and on["logged"][tag] != on["logged"][tag]), \
            f"the heads perturbed {tag}: {v} vs {on['logged'][tag]}"
    assert th.equal(off["rng"][0], on["rng"][0]), "the heads consumed the torch RNG"
    assert np.array_equal(off["rng"][1][1], on["rng"][1][1]), "the heads consumed the numpy RNG"
    assert off["rng"][2] == on["rng"][2], "the heads consumed python's random"


def test_the_ON_arm_is_not_vacuous(arms):
    off, on = arms
    names = [n for n in off["names"]]
    unmoved = [k for k in names if th.equal(off["post"][k], off["pre"][k])]
    assert len(unmoved) <= 0.15 * len(names), f"{len(unmoved)}/{len(names)} policy params unmoved"
    for head in ("ensemble", "rnd", "adv", "opp"):
        keys = [n for n in on["names"] if n.startswith(f"ridealong.{head}.")]
        assert keys, head
        moved = [k for k in keys if not th.equal(on["post"][k], on["pre"][k])]
        assert moved, f"no {head} parameter moved: the head trained nothing"
    for tag in ("ridealong/ens_loss", "ridealong/rnd_loss", "ridealong/adv_loss",
                "ridealong/opp_loss", "ridealong/ens_disagreement_mean", "ridealong/rnd_z_mean",
                "ridealong/adv_std_fed", "ridealong/opp_label_rate"):
        assert tag in on["logged"], tag
    assert not any(t.startswith("ridealong/") for t in off["logged"])
