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

THE RND VARIANTS (`gen3_ridealong_rnd_variants_v1`) ride in the ON arm (`all`), so 1–4 cover them,
and a THIRD arm — the four heads WITHOUT the variants — pins that adding the variants leaves every
other ride-along tensor, base RND included, BIT-IDENTICAL after the update: the reference the
variants are compared with is untouched by them. An identification probe that drew a random number
(say a `randperm` chimera instead of the deterministic roll) breaks the RNG assertion in 3.

THE DECLARED LIFECYCLE (K8; owner 2026-09-28): every ride-along optimizer — the four heads' and each
variant's — exists, with its Adam state allocated, BEFORE the first update (`_setup_model` →
`_ridealong_acquire`), and the update creates no optimizer, no optimizer state and no buffer
(the object identities are unchanged). There is NO lazy build path: a step whose optimizers were
not acquired RAISES `RideAlongLifecycleViolation` and builds nothing, so reintroducing a lazy build
fails `test_an_UNACQUIRED_step_RAISES_and_builds_nothing`. Pre-allocation is also pinned to step
bit-identically to torch's lazy init.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch as th

RIDEALONG_CORE = {"ridealong_ensemble": 3, "ridealong_rnd": True, "ridealong_adv": 2,
                  "ridealong_opp": 2}
RIDEALONG_ON = {**RIDEALONG_CORE, "ridealong_rnd_variants": "all"}
VARIANTS = ("fast", "decay", "small", "feat")


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
    from agents.training.rust_rollout.testkit import ToyVecEnv

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
        Gen3DualHeadMaskablePolicy, ToyVecEnv([(lambda s=s: env_cls(s)) for s in range(2)]),
        n_steps=8, batch_size=4, n_epochs=2, device="cpu", seed=0, policy_kwargs=pk)
    from agents.training.rust_rollout.testkit import attach_vec_collector
    attach_vec_collector(model)         # the rollout comes from the toy VecEnv (the Rust collector is production's)
    built_rng = _rng_state()
    # The startup acquisition, observed BEFORE any update: every optimizer and its Adam state exist.
    acquired = _acquired(model)
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
                logged=dict(model.logger.name_to_value), rng=_rng_state(), names=names,
                acquired=acquired, after=_acquired(model))


def _acquired(model) -> dict:
    """The ride-along resources by object identity: each optimizer, each Adam state tensor, each
    head buffer."""
    heads = model.policy.ridealong
    if heads is None:
        return {}
    opts = {"core": model._ridealong_opt, **{f"v_{k}": v for k, v in model._ridealong_vopts.items()}}
    state = {(name, id(p), k): id(t) for name, o in opts.items() if o is not None
             for p, st in o.state.items() for k, t in st.items() if th.is_tensor(t)}
    n_params = {name: sum(len(g["params"]) for g in o.param_groups) for name, o in opts.items()
                if o is not None}
    n_state = {name: len(o.state) for name, o in opts.items() if o is not None}
    return {"opt_ids": {k: id(v) for k, v in opts.items()}, "state": state,
            "n_params": n_params, "n_state": n_state,
            "buffers": {n: id(b) for n, b in heads.named_buffers()}}


@pytest.fixture(scope="module")
def arms():
    return _build({}), _build(RIDEALONG_ON)


@pytest.fixture(scope="module")
def core_arm():
    """The four heads WITHOUT the RND variants — the base reference's own run."""
    return _build(RIDEALONG_CORE)


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
    for head in ("ensemble", "rnd", "adv", "opp", *(f"rnd_variants.{v}" for v in VARIANTS)):
        keys = [n for n in on["names"] if n.startswith(f"ridealong.{head}.")]
        assert keys, head
        moved = [k for k in keys if not th.equal(on["post"][k], on["pre"][k])]
        assert moved, f"no {head} parameter moved: the head trained nothing"
    for tag in ("ridealong/ens_loss", "ridealong/rnd_loss", "ridealong/adv_loss",
                "ridealong/opp_loss", "ridealong/ens_disagreement_mean", "ridealong/rnd_z_mean",
                "ridealong/adv_std_fed", "ridealong/opp_label_rate",
                "ridealong/rnd_err_median", "ridealong/rnd_err_rel_spread",
                "ridealong/rnd_ident_ratio"):
        assert tag in on["logged"], tag
    for v in VARIANTS:
        for stat in ("loss", "err_mean", "z_mean", "err_median", "err_iqr", "grad_norm",
                     "disabled"):
            assert f"ridealong/rndv_{v}_{stat}" in on["logged"], (v, stat)
        assert on["logged"][f"ridealong/rndv_{v}_disabled"] == 0.0
        # the identification monitor runs for the OBSERVATION variants (feat needs a trunk forward)
        assert (f"ridealong/rndv_{v}_ident_ratio" in on["logged"]) == (v != "feat"), v
    assert not any(t.startswith("ridealong/") for t in off["logged"])


def test_the_variants_leave_the_four_heads_and_BASE_bit_identical(arms, core_arm):
    """The paired reference is intact: every ride-along tensor that is not a variant's — base RND's
    predictor, normaliser and error statistics included — is BIT-IDENTICAL after one update with
    and without the variants. (So is everything PPO owns, by the test above.)"""
    _, on = arms
    core = core_arm
    keys = [k for k in core["post"] if k.startswith("ridealong.")]
    assert any(k.startswith("ridealong.rnd.predictor.") for k in keys)
    for k in keys:
        assert th.equal(core["post"][k], on["post"][k]), f"the variants changed {k}"
    # every ride-along scalar the four heads log (base's included) reads the same with the variants
    rtags = [t for t in core["logged"] if t.startswith("ridealong/")]
    assert "ridealong/rnd_loss" in rtags and "ridealong/grad_norm" in rtags
    for tag in rtags:
        v, w = core["logged"][tag], on["logged"][tag]
        assert v == w or (v != v and w != w), f"the variants perturbed {tag}: {v} vs {w}"


def test_LIFECYCLE_every_ride_along_optimizer_is_acquired_at_startup_and_nothing_after(arms):
    """K8 / the declared lifecycle: before the first update every optimizer exists with its Adam
    state allocated for EVERY parameter; the update creates no optimizer, no state, no buffer."""
    _, on = arms
    pre, post = on["acquired"], on["after"]
    assert set(pre["opt_ids"]) == {"core", *(f"v_{v}" for v in VARIANTS)}
    for name, n in pre["n_params"].items():
        assert pre["n_state"][name] == n, f"{name}: Adam state not pre-allocated for every param"
    assert post["opt_ids"] == pre["opt_ids"], "an optimizer object was replaced during the update"
    assert post["state"] == pre["state"], "Adam state tensors were (re)created during the update"
    assert post["buffers"] == pre["buffers"], "a ride-along buffer was created or replaced"


def test_an_UNACQUIRED_step_RAISES_and_builds_nothing(arms):
    """The declared lifecycle has no lazy fallback (K6.1's freeze guard makes a post-startup
    acquisition FATAL). A learner that skipped `_ridealong_acquire` must REFUSE the step — for the
    four heads' optimizer and for every variant's — and leave no optimizer behind. Reintroducing a
    lazy build (build-on-miss) fails here."""
    from agents.training.instrumented_ppo.ridealong_terms import (RideAlongLifecycleViolation,
                                                                  RideAlongTerms)

    _, on = arms

    class _L(RideAlongTerms):
        def __init__(self, policy):
            self.policy = policy

    heads = on["model"].policy.ridealong
    lr = _L(on["model"].policy)
    with pytest.raises(RideAlongLifecycleViolation):
        lr._ridealong_optimizer(heads)
    for v in VARIANTS:
        with pytest.raises(RideAlongLifecycleViolation):
            lr._ridealong_variant_optimizer(heads, v)
    assert getattr(lr, "_ridealong_opt", None) is None
    assert not getattr(lr, "_ridealong_vopts", None)
    # acquired for OTHER heads (a swap without re-acquiring) is refused too
    lr._ridealong_acquire()
    import copy
    other = copy.deepcopy(heads)
    with pytest.raises(RideAlongLifecycleViolation):
        lr._ridealong_optimizer(other)
    with pytest.raises(RideAlongLifecycleViolation):
        lr._ridealong_variant_optimizer(other, "fast")
    # ... and the acquired ones are returned as-is, every time (read, never rebuilt)
    assert lr._ridealong_optimizer(heads) is lr._ridealong_optimizer(heads)
    assert all(lr._ridealong_variant_optimizer(heads, v) is lr._ridealong_vopts[v] for v in VARIANTS)


@pytest.mark.parametrize("fused", [False, True])
def test_PREALLOCATED_adam_steps_bit_identically_to_lazy_init(fused):
    from agents.training.instrumented_ppo.ridealong_terms import preallocate_adam_state

    dev = "cuda" if fused else "cpu"
    if fused and not th.cuda.is_available():
        pytest.skip("fused Adam needs CUDA")
    g = th.Generator().manual_seed(3)
    w0 = th.randn(7, 5, generator=g)
    grads = [th.randn(7, 5, generator=g) for _ in range(4)]
    a = th.nn.Parameter(w0.clone().to(dev))
    b = th.nn.Parameter(w0.clone().to(dev))
    oa = th.optim.Adam([a], lr=3e-3, eps=1e-5, fused=fused or None)
    ob = th.optim.Adam([b], lr=3e-3, eps=1e-5, fused=fused or None)
    preallocate_adam_state(ob)
    assert th.equal(b, a) and b.grad is None and len(ob.state[b]) >= 3
    for gr in grads:
        a.grad = gr.clone().to(dev)
        b.grad = gr.clone().to(dev)
        oa.step()
        ob.step()
    assert th.equal(a, b)
    for k, v in oa.state[a].items():
        assert th.equal(th.as_tensor(v), th.as_tensor(ob.state[b][k])), k
