"""The ride-along heads (`gen3_ridealong_heads_v1`): gradient COVERAGE — no leak — and the contracts.

THE DETACHMENT PROOF, half (a) (half (b), the bit-identical update, is
`agents/training/ridealong_update_test.py`): on a REAL production policy with all four heads, one
fixture forward, the ride-along losses built through `RideAlongBatch.detached` — the one seam the
learner uses — give NO gradient to any trunk, policy or V parameter, and a NON-ZERO gradient to every
head's own parameters (the positive control). The teeth control builds the same batch WITHOUT the
detach and shows the gradient DOES reach the trunk, so the "no leak" assertion is not vacuous; and
reverting the detach in `detached()` fails the first test.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch as th

from agents.model.ridealong_heads import (RideAlongBatch, RideAlongSpec, bootstrap_mask,
                                          freeze_to_buffers, state_hash)

ON = {"ridealong_ensemble": 3, "ridealong_rnd": True, "ridealong_adv": 2, "ridealong_opp": 2}


def _policy(ridealong=ON, seed=0):
    import gymnasium as gym

    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from main.fresh_checkpoint import _production_policy_kwargs

    _args, layout, pk = _production_policy_kwargs()
    pk = {**pk, "features_extractor_kwargs": {**pk["features_extractor_kwargs"], **ridealong}}
    obs_space = gym.spaces.Dict({
        "observation": gym.spaces.Box(-np.inf, np.inf, (layout["total_dim"],), np.float32),
        "action_mask": gym.spaces.MultiBinary(11)})
    th.manual_seed(seed)
    pol = Gen3DualHeadMaskablePolicy(obs_space, gym.spaces.Discrete(11), lambda _: 3e-4, **pk)
    return pol, layout


@pytest.fixture(scope="module")
def forward():
    from agents.model.compile_parity_fixture import load_parity_rows
    from agents.model.parity_probe import perturb_

    pol, layout = _policy()
    perturb_(pol)        # private RNG: opens the zero-init paths so every gradient is informative
    rows, masks = load_parity_rows(layout["total_dim"])
    n = 16
    obs = {"observation": th.tensor(rows[:n]), "action_mask": th.tensor(masks[:n].astype(np.float32))}
    pol.train()
    return pol, obs, th.tensor(masks[:n])


def _batch_kwargs(pol, obs, masks):
    """Run the forward the learner runs, and return the kwargs the learner hands `detached`."""
    pi_mask = masks.numpy()
    actions = th.tensor([int(np.flatnonzero(m)[0]) for m in pi_mask])
    values, _, _ = pol.evaluate_actions(obs, actions, action_masks=pi_mask)
    fe = pol.features_extractor
    dist = pol._last_pi_distribution
    n = values.shape[0]
    return dict(obs=obs["observation"], pooled=fe.last_value_pooled,
                pointer=tuple(fe.last_pointer_inputs), pi=dist.distribution.probs,
                logits=dist.distribution.logits, legal=dist.distribution.probs > 0,
                values=values, actions=actions,
                advantages=values.reshape(-1) * 0.3 - 0.1,        # graph-carrying on purpose
                win_target=(th.arange(n) % 2).float(), win_mask=th.ones(n),
                alpha_logits=fe.last_alpha_logits, alpha_seat_nums=fe.last_alpha_seat_nums,
                opp_kind=(th.arange(n) % 2).long(), opp_num=th.zeros(n, dtype=th.long))


def _core_params(pol):
    return [(n, p) for n, p in pol.named_parameters()
            if not n.startswith("ridealong.") and p.requires_grad]


def _grads(loss, params):
    return th.autograd.grad(loss, [p for _, p in params], allow_unused=True, retain_graph=True)


def test_NO_ridealong_loss_reaches_a_trunk_policy_or_V_parameter(forward):
    pol, obs, masks = forward
    b = RideAlongBatch.detached(**_batch_kwargs(pol, obs, masks))
    heads = pol.ridealong
    if heads.rnd is not None:
        heads.rnd.update_obs_stats(b.obs)
    out = heads.readout(b)
    losses = heads.losses(b, out)
    assert set(losses) == {"ens", "rnd", "adv", "opp"}, sorted(losses)
    core = _core_params(pol)
    assert len(core) > 200
    for name, loss in losses.items():
        # `grad` over a param the loss's graph never touches raises unless allow_unused; a detached
        # loss has NO path to ANY core parameter at all, so every entry is None.
        g = _grads(loss, core)
        leaked = [n for (n, _), gi in zip(core, g) if gi is not None and bool(gi.abs().sum() > 0)]
        assert not leaked, f"the ride-along {name!r} loss put a gradient on {leaked[:6]}"


def test_POSITIVE_CONTROL_every_head_trains_its_own_parameters(forward):
    pol, obs, masks = forward
    b = RideAlongBatch.detached(**_batch_kwargs(pol, obs, masks))
    heads = pol.ridealong
    out = heads.readout(b)
    losses = heads.losses(b, out)
    for name, sub in (("ens", heads.ensemble), ("rnd", heads.rnd), ("adv", heads.adv),
                      ("opp", heads.opp)):
        own = [(n, p) for n, p in sub.named_parameters()]
        g = _grads(losses[name], own)
        live = [n for (n, _), gi in zip(own, g) if gi is not None and bool(gi.abs().sum() > 0)]
        assert live, f"the {name!r} loss trains none of its own parameters"


def test_TEETH_without_the_detach_the_gradient_DOES_reach_the_trunk(forward):
    """The control that makes the no-leak test mean something: the SAME losses on a batch that was
    NOT routed through `detached()` put gradient on the trunk (the pointer tokens and value_pooled
    carry the extractor's graph). If this ever stops holding, the no-leak test above is vacuous."""
    pol, obs, masks = forward
    b = RideAlongBatch(**_batch_kwargs(pol, obs, masks))
    heads = pol.ridealong
    out = heads.readout(b)
    losses = heads.losses(b, out)
    core = _core_params(pol)
    leaked = []
    for name in ("ens", "adv", "opp"):
        g = _grads(losses[name], core)
        leaked += [n for (n, _), gi in zip(core, g) if gi is not None and bool(gi.abs().sum() > 0)]
    assert any(n.startswith("features_extractor.") for n in leaked)


def test_the_heads_are_not_in_the_forward_and_not_in_the_optimizer():
    pol, _ = _policy()
    assert pol.ridealong is not None
    in_opt = {id(p) for g in pol.optimizer.param_groups for p in g["params"]}
    assert not any(id(p) in in_opt for p in pol.ridealong.parameters())
    called = []
    hooks = [m.register_forward_hook(lambda *a: called.append(1)) for m in pol.ridealong.modules()]
    from agents.model.compile_parity_fixture import load_parity_rows
    rows, masks = load_parity_rows(pol.ridealong.obs_dim)
    with th.no_grad():
        pol.forward({"observation": th.tensor(rows[:4]),
                     "action_mask": th.tensor(masks[:4].astype(np.float32))},
                    action_masks=masks[:4])
    for h in hooks:
        h.remove()
    assert not called, "a ride-along module ran inside the policy forward"


def test_OFF_builds_nothing_and_the_state_dict_has_no_key():
    pol, _ = _policy(ridealong={})
    assert pol.ridealong is None
    assert not any(k.startswith("ridealong.") for k in pol.state_dict())


def test_the_frozen_networks_are_BUFFERS_and_ride_the_state_dict():
    pol, _ = _policy()
    ra = pol.ridealong
    params = {n for n, _ in ra.named_parameters()}
    assert not any(".priors." in n or n.startswith("rnd.target.") for n in params)
    sd = ra.state_dict()
    assert any(k.startswith("rnd.target.") for k in sd)
    assert any(".priors." in k for k in sd)
    lin = th.nn.Linear(3, 2)
    freeze_to_buffers(lin)
    assert not list(lin.parameters()) and set(dict(lin.named_buffers())) == {"weight", "bias"}
    assert lin(th.ones(1, 3)).shape == (1, 2)


def test_the_heads_are_REPRODUCIBLE_and_a_checkpoint_restores_them():
    a, _ = _policy(seed=0)
    b, _ = _policy(seed=123)          # a different GLOBAL seed: the heads' private seed is fixed
    for k, v in a.ridealong.state_dict().items():
        assert th.equal(v, b.ridealong.state_dict()[k]), k
    with th.no_grad():
        for p in a.ridealong.parameters():
            p.add_(0.01)
    b.load_state_dict(a.state_dict())
    for k, v in a.ridealong.state_dict().items():
        assert th.equal(v, b.ridealong.state_dict()[k]), k


def test_the_bootstrap_hash_is_a_per_STATE_bit_independent_of_the_minibatch():
    g = th.Generator().manual_seed(0)
    obs = th.randn(64, 50, generator=g)
    mult = th.randint(1, 2 ** 62, (50,), dtype=th.int64, generator=g) | 1
    h = state_hash(obs, mult)
    perm = th.randperm(64, generator=g)
    assert th.equal(state_hash(obs[perm], mult), h[perm])        # same row ⇒ same bits, any batch
    assert th.equal(state_hash(obs[:7], mult), h[:7])
    m = bootstrap_mask(state_hash(th.randn(4000, 50, generator=g), mult), 5, 0)
    assert m.shape == (4000, 5)
    assert th.all((m.mean(0) - 0.5).abs() < 0.04), m.mean(0)     # ~Bernoulli(0.5) per member
    # members are not copies of one another
    assert float((m[:, 0] == m[:, 1]).float().mean()) < 0.6


def test_CENTRING_A_under_pi_and_B_under_alpha(forward):
    pol, obs, masks = forward
    b = RideAlongBatch.detached(**_batch_kwargs(pol, obs, masks))
    out = pol.ridealong.readout(b)
    s_a = (out["adv"] * b.pi[:, None, :]).sum(-1)
    assert float(s_a.abs().max()) < 1e-5
    alpha = th.softmax(b.alpha_logits.float(), -1)
    s_b = (out["opp"] * alpha[:, None, :]).sum(-1)
    assert float(s_b.abs().max()) < 1e-5
    # Untrained members differ ONLY by their randomized priors — and do differ (per-action spread).
    assert float(out["adv_std"][b.legal].mean()) > 0
    assert float(out["ens_std"].mean()) > 0


def test_RND_running_stats_merge_matches_the_pooled_moments():
    from agents.model.ridealong_heads import RndNovelty

    with th.random.fork_rng(devices=[]):
        r = RndNovelty(6)
    g = th.Generator().manual_seed(1)
    xs = [th.randn(n, 6, generator=g) * 3 + 1 for n in (5, 17, 40)]
    for x in xs:
        r.update_obs_stats(x)
    allx = th.cat(xs)
    assert th.allclose(r.obs_mean, allx.mean(0), atol=1e-5)
    assert th.allclose(r.obs_var, allx.var(0, unbiased=False), atol=1e-4)
    z = r.normalise(th.full((1, 6), 1e6))
    assert float(z.max()) == 5.0                                 # clipped (Burda et al.)


def test_the_spec_and_the_dependencies():
    with pytest.raises(ValueError, match="members"):
        from agents.model.ridealong_heads import validate_spec
        validate_spec(RideAlongSpec(ensemble=13))
    assert not RideAlongSpec().any and RideAlongSpec(rnd=True).any


def test_the_version_gate_refuses_a_flip_and_the_migration_defaults_off():
    import dataclasses

    from agents.model.model_version import ModelVersionError
    from agents.model.model_version.migrations import _migrate_config
    from agents.model.snapshot import current_model_version
    from agents.observation.state_encoder import load_mappings

    base = current_model_version(load_mappings())
    for field, val in (("ridealong_ensemble", 5), ("ridealong_rnd", True), ("ridealong_adv", 5),
                       ("ridealong_opp", 5)):
        on = dataclasses.replace(base, **{field: val})
        with pytest.raises(ModelVersionError, match=field):
            on.check_compatible(base)
        with pytest.raises(ModelVersionError, match=field):
            base.check_compatible(on)
        on.check_compatible(dataclasses.replace(base, **{field: val}))
    old = {k: v for k, v in dataclasses.asdict(base).items() if not k.startswith("ridealong_")}
    old["config_version"] = 125
    new = _migrate_config(old)
    assert (new["ridealong_ensemble"], new["ridealong_rnd"], new["ridealong_adv"],
            new["ridealong_opp"], new["config_version"]) == (0, False, 0, 0, 126)
