"""The ride-along heads (`gen3_ridealong_heads_v1`): gradient COVERAGE — no leak — and the contracts.

THE DETACHMENT PROOF, half (a) (half (b), the bit-identical update, is
`agents/training/ridealong_update_test.py`): on a REAL production policy with all four heads, one
fixture forward, the ride-along losses built through `RideAlongBatch.detached` — the one seam the
learner uses — give NO gradient to any trunk, policy or V parameter, and a NON-ZERO gradient to every
head's own parameters (the positive control). The teeth control builds the same batch WITHOUT the
detach and shows the gradient DOES reach the trunk, so the "no leak" assertion is not vacuous; and
reverting the detach in `detached()` fails the first test.

B is X5's: on the production (X5) policy its columns are the FLAT opponent pointer's (`fe.last_flat_intent`),
its labels the column `flat_intent.flat_intent_targets` names (the learner's own construction,
`instrumented_ppo/ridealong_terms.py`), and it centres under the flat α.

THE RND VARIANTS (`gen3_ridealong_rnd_variants_v1`) are covered by the same three tests: every
variant's loss leaks nothing (`fast` / `decay` / `small` read the observation, `feat` reads the
detached value_pooled — the teeth control shows an UNdetached value_pooled carries `feat`'s gradient
into the trunk), and each variant's loss trains ITS OWN predictor and nothing else (no other
variant, not base: their optimizers and comparisons are separate).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch as th

from agents.model.model_version.constants import MODEL_CONFIG_VERSION
from agents.model.ridealong_heads import (RND_VARIANTS, RideAlongBatch, RideAlongSpec,
                                          block_chimera, bootstrap_mask, canonical_rnd_variants,
                                          freeze_to_buffers, obs_block_edges, parse_rnd_variants,
                                          state_hash)

ON = {"ridealong_ensemble": 3, "ridealong_rnd": True, "ridealong_adv": 2, "ridealong_opp": 2,
      "ridealong_rnd_variants": "all"}
VARIANT_LOSSES = {"rndv_fast", "rndv_decay", "rndv_small", "rndv_feat"}


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
    """Run the forward the learner runs, and return the kwargs the learner hands `detached` — B's FLAT
    pointer inputs built as `ridealong_terms` builds them (`flat_intent_targets` on the Rust intent label).
    The label: every row names the opponent's FIRST move seat (a MOVE of that num), so each row with a
    live first seat is a B label — asserted, never assumed."""
    from agents.model.flat_intent import flat_intent_targets
    from agents.model.opp_intent import INTENT_IGNORE

    pi_mask = masks.numpy()
    actions = th.tensor([int(np.flatnonzero(m)[0]) for m in pi_mask])
    values, _, _ = pol.evaluate_actions(obs, actions, action_masks=pi_mask)
    fe = pol.features_extractor
    dist = pol._last_pi_distribution
    n = values.shape[0]
    fi = fe.last_flat_intent
    assert fi is not None and fe.last_flat_intent_logits is not None, "the X5 forward stashed no flat pointer"
    kind = th.zeros(n, 1)
    num = fi.seat_nums[:, :1].float()
    flat_target, _ = flat_intent_targets(fi, kind, num, th.full((n, 1), -1.0), th.zeros(n, 1))
    assert int((flat_target != INTENT_IGNORE).sum()) >= 4, flat_target      # B has labels to train on
    return dict(obs=obs["observation"], pooled=fe.last_value_pooled,
                pointer=tuple(fe.last_pointer_inputs), pi=dist.distribution.probs,
                logits=dist.distribution.logits, legal=dist.distribution.probs > 0,
                values=values, actions=actions,
                advantages=values.reshape(-1) * 0.3 - 0.1,        # graph-carrying on purpose
                win_target=(th.arange(n) % 2).float(), win_mask=th.ones(n),
                flat_logits=fe.last_flat_intent_logits, flat_ids=fi.cand_ids, flat_target=flat_target)


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
        heads.rnd_variants["feat"].update_obs_stats(b.pooled)
    out = heads.readout(b)
    losses = heads.losses(b, out)
    assert set(losses) == {"ens", "rnd", "adv", "opp"}, sorted(losses)
    vlosses = heads.variant_losses(out)
    assert set(vlosses) == VARIANT_LOSSES, sorted(vlosses)
    losses = {**losses, **vlosses}
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
        own = [(n, p) for n, p in sub.named_parameters()
               if not n.startswith("rnd_variants.")]
        g = _grads(losses[name], own)
        live = [n for (n, _), gi in zip(own, g) if gi is not None and bool(gi.abs().sum() > 0)]
        assert live, f"the {name!r} loss trains none of its own parameters"
    # Each RND VARIANT's loss trains ITS OWN predictor — every one of its tensors — and nothing else
    # among the ride-along parameters: not base, not another variant (separate optimizers, and the
    # paired comparison means nothing if one variant's loss moved another's predictor).
    vlosses = heads.variant_losses(out)
    allp = list(heads.named_parameters())
    for vn in RND_VARIANTS:
        g = _grads(vlosses[f"rndv_{vn}"], allp)
        live = {n for (n, _), gi in zip(allp, g) if gi is not None and bool(gi.abs().sum() > 0)}
        own = {n for n, _ in allp if n.startswith(f"rnd_variants.{vn}.predictor.")}
        assert own and live == own, (vn, sorted(live ^ own)[:6])
    # base's own loss does not reach a variant either
    g = _grads(losses["rnd"], allp)
    assert not any(n.startswith("rnd_variants.") for (n, _), gi in zip(allp, g)
                   if gi is not None and bool(gi.abs().sum() > 0))


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
    # `feat` reads value_pooled, which (the ens/adv rows just showed) carries the trunk's graph when
    # undetached. Its module detaches its own input as well (`RndNovelty.inputs`), so it has TWO
    # stop-grads: even this UNdetached batch leaks nothing through it. The no-leak test's `rndv_feat`
    # row therefore fails only when BOTH are reverted; the seam alone is pinned by the rows above.
    assert b.pooled.requires_grad
    g = _grads(heads.variant_losses(out)["rndv_feat"], core)
    assert not any(gi is not None and bool(gi.abs().sum() > 0) for gi in g)


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
    assert not any(".priors." in n or ".target." in n or ".anchor." in n or n.startswith("rnd.target.")
                   for n in params)
    sd = ra.state_dict()
    assert any(k.startswith("rnd.target.") for k in sd)
    assert any(".priors." in k for k in sd)
    # the variants' frozen nets: feat's own target, decay's init anchor — buffers in the state_dict
    assert any(k.startswith("rnd_variants.feat.target.") for k in sd)
    assert any(k.startswith("rnd_variants.decay.anchor.") for k in sd)
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


def test_CENTRING_A_under_pi_and_B_under_the_flat_alpha(forward):
    pol, obs, masks = forward
    b = RideAlongBatch.detached(**_batch_kwargs(pol, obs, masks))
    out = pol.ridealong.readout(b)
    s_a = (out["adv"] * b.pi[:, None, :]).sum(-1)
    assert float(s_a.abs().max()) < 1e-5
    # B's columns ARE the flat pointer's (K seats · OTHER_move · six slots · OTHER_species), centred
    # under its distribution (a masked candidate's −inf logit gives it zero weight)
    assert out["opp"].shape[-1] == b.flat_logits.shape[-1] == pol.features_extractor.entity_topk_seats + 8
    alpha = th.softmax(b.flat_logits.float(), -1)
    s_b = (out["opp"] * alpha[:, None, :]).sum(-1)
    assert float(s_b.abs().max()) < 1e-5
    # ... and the centring is not vacuous: the raw (uncentred) members' α-mean is NOT zero
    raw = pol.ridealong.opp(b.pooled, b.flat_ids)
    assert float((raw * alpha[:, None, :]).sum(-1).abs().max()) > 1e-4
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
    # B exists only over X5's flat pointer: the blob arm's B over α's support was deleted at the version
    # break, so a B spec with no flat pointer is REFUSED (never a silent alternative head)
    with pytest.raises(ValueError, match="FLAT opponent pointer"):
        validate_spec(RideAlongSpec(opp=2))
    validate_spec(RideAlongSpec(opp=2, opp_flat_k=4))
    pol, _ = _policy()
    from agents.model.ridealong_heads import FlatOppEffectEnsemble
    assert isinstance(pol.ridealong.opp, FlatOppEffectEnsemble)
    assert pol.ridealong.spec.opp_flat_k == pol.features_extractor.entity_topk_seats


def test_the_version_gate_refuses_a_flip_and_a_pre_break_config_is_refused():
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
    on = dataclasses.replace(base, ridealong_rnd_variants="fast,decay,small,feat")
    with pytest.raises(ModelVersionError, match="ridealong_rnd_variants"):
        on.check_compatible(base)
    with pytest.raises(ModelVersionError, match="ridealong_rnd_variants"):
        dataclasses.replace(base, ridealong_rnd_variants="fast").check_compatible(on)
    # the fields are RECORDED: a config at the current version carrying them migrates verbatim
    rec = dict(dataclasses.asdict(base), ridealong_ensemble=5, ridealong_rnd=True, ridealong_adv=5,
               ridealong_opp=5, ridealong_rnd_variants="fast,decay,small,feat")
    got = _migrate_config(dict(rec))
    assert {k: got[k] for k in rec if k.startswith("ridealong_")} == \
        {k: v for k, v in rec.items() if k.startswith("ridealong_")}
    assert got["config_version"] == MODEL_CONFIG_VERSION
    # a config from before the heads existed (v125) predates the X5 version break's MIGRATION_FLOOR: the
    # migration that defaulted them OFF is unreachable, and the config is REFUSED
    old = {k: v for k, v in dataclasses.asdict(base).items() if not k.startswith("ridealong_")}
    old["config_version"] = 125
    with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
        _migrate_config(old)


# ── the RND variant ensemble (gen3_ridealong_rnd_variants_v1) ────────────────────────────────────
def test_the_variant_list_is_DECLARED_and_CANONICAL():
    assert RND_VARIANTS == ("fast", "decay", "small", "feat")
    assert parse_rnd_variants("all") == RND_VARIANTS
    assert parse_rnd_variants("feat, FAST") == ("fast", "feat")
    assert canonical_rnd_variants("decay,fast") == canonical_rnd_variants("fast,decay") == "fast,decay"
    for off in (None, "", "off", "none"):
        assert parse_rnd_variants(off) == () and canonical_rnd_variants(off) == "off"
    with pytest.raises(ValueError, match="unknown RND variant"):
        parse_rnd_variants("fast,slow")
    with pytest.raises(ValueError, match="unknown RND variant"):
        parse_rnd_variants("base")              # base is --ridealong-rnd, not a variant
    with pytest.raises(ValueError, match="repeated"):
        parse_rnd_variants("fast,fast")
    from agents.model.ridealong_heads import validate_spec
    with pytest.raises(ValueError, match="requires ridealong_rnd"):
        validate_spec(RideAlongSpec(rnd_variants=("fast",)))
    with pytest.raises(ValueError, match="canonical"):
        validate_spec(RideAlongSpec(rnd=True, rnd_variants=("feat", "fast")))


def test_the_extractor_refuses_variants_without_base_and_records_them_canonically():
    with pytest.raises(ValueError, match="requires ridealong_rnd"):
        _policy(ridealong={"ridealong_rnd_variants": "fast"})
    pol, _ = _policy(ridealong={"ridealong_rnd": True, "ridealong_rnd_variants": "feat,small"})
    assert pol.features_extractor.ridealong_rnd_variants == "small,feat"
    assert pol.ridealong.variant_names() == ("small", "feat")


def test_ADDING_variants_changes_no_other_head_and_fast_decay_START_AS_BASE(forward):
    """The reference is intact: every non-variant ride-along tensor is identical with and without
    the variants; `fast` and `decay` start from base's exact predictor (paired from step 0)."""
    without, _ = _policy(ridealong={k: v for k, v in ON.items() if k != "ridealong_rnd_variants"})
    with_, _ = _policy()
    sd_w = with_.ridealong.state_dict()
    sd_o = without.ridealong.state_dict()
    assert {k for k in sd_w if not k.startswith("rnd_variants.")} == set(sd_o)
    for k, v in sd_o.items():
        assert th.equal(v, sd_w[k]), k
    base_pred = with_.ridealong.rnd.predictor.state_dict()
    for vn in ("fast", "decay"):
        vp = with_.ridealong.rnd_variants[vn].predictor.state_dict()
        assert all(th.equal(base_pred[k], vp[k]) for k in base_pred), vn
    # ... so at init they read EXACTLY base's error, row for row (one target output, shared)
    pol, obs, masks = forward
    b = RideAlongBatch.detached(**_batch_kwargs(pol, obs, masks))
    out = with_.ridealong.readout(b)
    assert th.equal(out["rndv_fast_err"], out["rnd_err"])
    assert th.equal(out["rndv_decay_err"], out["rnd_err"])
    assert not th.equal(out["rndv_small_err"], out["rnd_err"])


def test_the_small_predictor_is_the_declared_capacity():
    pol, _ = _policy()
    n = sum(p.numel() for p in pol.ridealong.rnd_variants["small"].parameters())
    base = sum(p.numel() for p in pol.ridealong.rnd.predictor.parameters())
    # gen3_obs_facts_v1 (the X5 version break's part 3): the obs-RND predictors read the whole observation, so
    # the 84-dim OBS-FACTS append widened each first layer by 84 inputs (+84 x 32 small, +84 x 256 base).
    assert (n, base) == (93_184, 810_816)


def test_DECAY_pulls_toward_init_with_the_declared_half_life():
    from agents.model.ridealong_heads import RND_DECAY_HALF_LIFE_UPDATES

    pol, _ = _policy()
    dec = pol.ridealong.rnd_variants["decay"]
    init = {k: v.clone() for k, v in dec.predictor.state_dict().items()}
    with th.no_grad():
        for p in dec.predictor.parameters():
            p.add_(1.0)
    for _ in range(int(RND_DECAY_HALF_LIFE_UPDATES)):
        pol.ridealong.begin_update_()
    for k, v in dec.predictor.state_dict().items():
        assert th.allclose(v - init[k], th.full_like(v, 0.5), atol=1e-5), k
    # the other variants are untouched by the pull
    fast0 = {k: v.clone() for k, v in pol.ridealong.rnd_variants["fast"].state_dict().items()}
    pol.ridealong.begin_update_()
    assert all(th.equal(v, fast0[k])
               for k, v in pol.ridealong.rnd_variants["fast"].state_dict().items())


def test_the_CHIMERA_probe_is_deterministic_and_made_of_real_blocks():
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    edges = obs_block_edges(layout, layout["total_dim"])
    assert edges[0] == 0 and edges[-1] == layout["total_dim"] and len(edges) >= 6
    assert layout["parts"]["opp_team"]["start"] in edges
    g = th.Generator().manual_seed(0)
    x = th.randn(64, layout["total_dim"], generator=g)
    c = block_chimera(x, edges)
    assert th.equal(c, block_chimera(x, edges))                  # no RNG: identical every call
    assert th.equal(c[:, edges[0]:edges[1]], x[:, edges[0]:edges[1]])   # block 0 is the row's own
    n, nb = x.shape[0], len(edges) - 1
    stride = max(1, n // nb)
    for k in range(1, nb):
        blk = c[:, edges[k]:edges[k + 1]]
        src = x[:, edges[k]:edges[k + 1]]
        # every chimera block is EXACTLY a real row's block — row (j + k·stride) mod n — never j's own
        donors = (th.arange(n) + k * stride) % n
        assert th.equal(blk, src[donors])
        assert not bool((donors == th.arange(n)).any())
    pol, _ = _policy()
    rng_before = th.get_rng_state().clone()
    from agents.model.compile_parity_fixture import load_parity_rows
    rows, _ = load_parity_rows(layout["total_dim"])
    errs = pol.ridealong.identification_errors(th.tensor(rows[:16]))
    assert set(errs) == {"rnd", "rndv_fast", "rndv_decay", "rndv_small"}
    assert all(not e.requires_grad for e in errs.values())
    assert th.equal(rng_before, th.get_rng_state()), "the identification probe drew from the RNG"
