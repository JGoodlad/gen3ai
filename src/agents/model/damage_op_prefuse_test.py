"""Unit tests for the PRE-ATTENTION unified damage operator (`damage_op_prefuse`, v50).

The claim under test: the DamageOperator runs EXACTLY ONCE per forward, BEFORE the transformer, and
its output does double duty — a zero-init residual onto our role tokens (so attention sees the
physics) and the SAME full block concatenated into both projection heads.

    beliefs ONCE (pre-attention) -> physics ONCE -> attention -> heads read that same block

It replaces the two-computation shape it supersedes: a LEAN `discrete_*` recompute inside the
between-layers refine loop (× `damage_refine_rounds`) PLUS the full post-transformer block.

Pins: the structural nature (exactly one new module), the three dependency guards, that the op is
called ONCE and the lean refine kernels not at all, that the HEAD CONCAT is preserved at full width
(ledger P1 — that path is the policy's largest measured dependency, so it must not silently shrink),
that OFF is byte-for-byte, that ON is identity-at-init on a REAL MaskablePPO-built policy (ledger M1
— a bare extractor proves nothing), that the pre-vs-post divergence is exactly ZERO at cold start and
REAL once the belief heads train, and the version/migration gate.
"""
import inspect

import gymnasium as gym
import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.action.constants import ACTION_SPACE_SIZE
from agents.model.features_extractor import Gen3FeaturesExtractor, HIDDEN_POWER_MOVE_NUM
from agents.model.model_version import (MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError,
                                        _migrate_config)
from agents.model.policy import Gen3DualHeadMaskablePolicy
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_mappings = load_mappings()
_layout = Gen3ObservationEncoder(_mappings).get_layout()

# The full belief+physics stack the toggle unifies: every consumer the op reads (move / spread /
# HP-type belief) plus the head-block producers, so the test exercises the production shape.
_BASE = dict(attend_unrevealed_opponents=True, opp_belief_slots=True,
             move_belief_mode="revealed", move_prior_fusion=True, move_belief_prefuse=True,
             move_latent=True, damage_op=True, damage_outgoing=True, damage_topk_k=5,
             spread_belief=True)
# The shape it replaces: the same stack with the between-layers refine loop + its threat channels.
_REFINE = dict(_BASE, damage_refine_rounds=2, threat_refine_outgoing=True, threat_status_refine=True)


def _make_model(**kw):
    obs_space = gym.spaces.Box(0.0, 1.0, shape=(_layout["total_dim"],), dtype=np.float32)
    return Gen3FeaturesExtractor(obs_space, layout=_layout, mappings=_mappings, **kw)


def _obs(batch=4, seed=1234):
    g = torch.Generator().manual_seed(seed)
    return {"observation": torch.rand(batch, _layout["total_dim"], generator=g)}


def _give_sentinel_a_spread_prior(*models):
    """Random-float observations decode to the species-0 UNKNOWN sentinel, whose spread prior is
    all-zero — and `believed = (mean + delta*std).clamp(min=1.0)` is then pinned at the clamp floor no
    matter what the head predicts. Copy a real species' prior (Tyranitar) onto row 0 so the spread
    head's output actually reaches the damage block; without this, a spread-head perturbation is
    invisible and any test built on it passes vacuously."""
    for m in models:
        with torch.no_grad():
            m.spread_belief.spread_prior[0] = m.spread_belief.spread_prior[248]


# --------------------------------------------------------------------------- structure / guards
def test_prefuse_adds_exactly_one_module():
    """STRUCTURAL, minimally: the only new state is the zero-init injection projection."""
    on = _make_model(**_BASE, damage_op_prefuse=True)
    off = _make_model(**_BASE, damage_op_prefuse=False)
    assert on.damage_op_prefuse is True and off.damage_op_prefuse is False
    assert set(on.state_dict()) - set(off.state_dict()) == {"prefuse_proj.weight", "prefuse_proj.bias"}
    assert not set(off.state_dict()) - set(on.state_dict())


def test_head_concat_width_is_preserved():
    """Ledger P1: the op's post-pool concat is the policy's LARGEST measured dependency (zeroing the
    whole block moves the policy by KL 0.9385). Unifying must not shrink it — the same full block
    reaches both heads, only its inputs move from post- to pre-attention."""
    on = _make_model(**_BASE, damage_op_prefuse=True)
    off = _make_model(**_BASE, damage_op_prefuse=False)
    assert on.projection_input_dim == off.projection_input_dim
    assert on.value_projection_input_dim == off.value_projection_input_dim
    with torch.no_grad():
        on.eval()(_obs())
    assert on.last_damage_block is not None
    assert on.last_damage_block.shape[1] == on.damage_op.out_dim


def test_requires_damage_op():
    kw = {k: v for k, v in _BASE.items() if k not in ("damage_op", "damage_outgoing",
                                                     "damage_topk_k")}
    with pytest.raises(ValueError, match="damage_op_prefuse"):
        _make_model(**kw, damage_op=False, damage_op_prefuse=True)


def test_requires_move_belief_prefuse():
    """The op consumes the move belief, so the belief must itself be pre-transformer — otherwise the
    op would read a posterior that does not exist yet."""
    kw = dict(_BASE)
    kw["move_belief_prefuse"] = False
    with pytest.raises(ValueError, match="move_belief_prefuse"):
        _make_model(**kw, damage_op_prefuse=True)


def test_mutually_exclusive_with_the_refine_loop():
    """Running both would restore the double computation the toggle exists to remove — fail loud
    rather than silently paying it."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        _make_model(**_REFINE, damage_op_prefuse=True)


# --------------------------------------------------------------------------- the core claim
def test_operator_runs_exactly_once_and_the_lean_kernels_never():
    """THE headline property. The refine shape calls the full op once AND the lean `discrete_incoming`
    once per round; the unified shape calls the full op once and the lean kernels zero times."""
    def _count(model):
        calls = {"full": 0, "lean": 0}
        real_full, real_lean = model.damage_op.forward, model.damage_op.discrete_incoming

        def full(*a, **kw):
            calls["full"] += 1
            return real_full(*a, **kw)

        def lean(*a, **kw):
            calls["lean"] += 1
            return real_lean(*a, **kw)
        model.damage_op.forward, model.damage_op.discrete_incoming = full, lean
        with torch.no_grad():
            model.eval()(_obs())
        return calls

    unified = _count(_make_model(**_BASE, damage_op_prefuse=True))
    refine = _count(_make_model(**_REFINE))
    assert unified == {"full": 1, "lean": 0}, unified
    assert refine["full"] == 1 and refine["lean"] == _REFINE["damage_refine_rounds"], refine


def test_physics_reaches_the_trunk_before_attention():
    """The injection must happen BEFORE the transformer runs — i.e. the op's output is already
    computed when the encoder layers are entered."""
    m = _make_model(**_BASE, damage_op_prefuse=True).eval()
    order = []
    real_op, real_tt = m.damage_op.forward, m.team_transformer.forward

    def op(*a, **kw):
        order.append("op")
        return real_op(*a, **kw)

    def tt(*a, **kw):
        order.append("transformer")
        return real_tt(*a, **kw)
    m.damage_op.forward, m.team_transformer.forward = op, tt
    with torch.no_grad():
        m(_obs())
    assert order == ["op", "transformer"], order


def test_grad_reaches_the_belief_heads_through_the_unified_path():
    """One computation, one gradient path: the op's damage gradient must still train the move /
    spread / HP-type belief heads (that gradient is the whole reason the op is differentiable)."""
    m = _make_model(**_BASE, damage_op_prefuse=True).train()
    pi, vf = m(_obs())
    (pi.sum() + vf.sum()).backward()
    for head in ("move_belief", "spread_belief", "hp_type_belief_head"):
        mod = getattr(m, head)
        grads = [p.grad for p in mod.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads), f"no gradient reached {head}"


# --------------------------------------------------------------------------- init-time behaviour
def test_off_is_byte_identical_to_the_untoggled_model():
    """OFF must reproduce the v49 forward exactly (the flag defaults off; no silent drift)."""
    torch.manual_seed(0)
    a = _make_model(**_BASE, damage_op_prefuse=False).eval()
    torch.manual_seed(0)
    b = _make_model(**_BASE).eval()                 # flag omitted entirely
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        pi_a, vf_a = a(_obs())
        pi_b, vf_b = b(_obs())
    assert torch.equal(pi_a, pi_b) and torch.equal(vf_a, vf_b)


def _shared_pair():
    """An ON/OFF pair with identical weights (ON's extra prefuse_proj is zero-init)."""
    torch.manual_seed(0)
    on = _make_model(**_BASE, damage_op_prefuse=True).eval()
    torch.manual_seed(0)
    off = _make_model(**_BASE, damage_op_prefuse=False).eval()
    off.load_state_dict({k: v for k, v in on.state_dict().items()
                         if not k.startswith("prefuse_proj.")})
    return on, off


def test_damage_block_is_identical_at_cold_start():
    """THE RISK THIS TOGGLE TAKES, bounded at its starting point. The head block is the thing that
    moves from a refined to an un-refined belief — but at init all three belief heads are zero-init,
    so every posterior the op consumes is TOKEN-INDEPENDENT (move == the Smogon prior, spread == the
    usage prior, HP type == the Smogon prior). Pre- and post-attention therefore feed the op the same
    numbers and the block is bit-identical. The divergence is created by TRAINING, not by the
    reordering — which is what makes a fresh run the honest way to evaluate it."""
    on, off = _shared_pair()
    with torch.no_grad():
        on(_obs())
        block_on = on.last_damage_block.clone()
        off(_obs())
        block_off = off.last_damage_block.clone()
    assert torch.equal(block_on, block_off)


def test_forward_differs_from_post_transformer_once_the_beliefs_train():
    """The flip side: it IS a real computational change — and this test also documents WHICH inputs
    actually move. The op requires `move_belief_prefuse`, so the move belief is pre-transformer in
    BOTH arms and its posterior is bit-identical either way.

    gen3_typed_hp_belief_v1 NARROWED that further: the HP-type head moved next to the move head (both
    now read the same tokens at the same point, so the two halves of `P(HP_t) = presence · P(t)` can
    never come from differently-refined states), which makes the HP-type posterior bit-identical across
    the toggle too. **The SPREAD posterior is now the only re-sourced input** — so perturb THAT head."""
    on, off = _shared_pair()
    g = torch.Generator().manual_seed(5)
    _give_sentinel_a_spread_prior(on, off)
    obs = _obs()
    with torch.no_grad():
        # ONE perturbation tensor copied into BOTH models — drawing two samples would make the test
        # pass on the weight difference alone, proving nothing about the toggle. This head's READ is
        # what the toggle re-sources (role tokens vs refined tokens).
        # Scaled small on purpose: `believed = (mean + delta*std).clamp(min=1.0)`, so a large delta
        # saturates BOTH arms onto the `clamp(min=1.0)` floor and the test would pass vacuously.
        delta = (torch.rand(on.spread_belief.stat_head.weight.shape, generator=g) * 2.0 - 1.0) * 1e-4
        on.spread_belief.stat_head.weight.copy_(delta)
        off.spread_belief.stat_head.weight.copy_(delta)
        assert torch.equal(on.spread_belief.stat_head.weight, off.spread_belief.stat_head.weight)
    with torch.no_grad():
        pi_on, vf_on = on(obs)
        pi_off, vf_off = off(obs)
    # The move + HP-type posteriors are now bit-identical across the toggle (both pre-transformer);
    # pin that, so a future change that re-splits them fails HERE rather than silently.
    assert torch.equal(on.last_move_belief_logits, off.last_move_belief_logits)
    assert torch.equal(on.last_hp_type_logits, off.last_hp_type_logits)
    assert (on.last_spread_belief > 1.0).any(), "spread saturated on the clamp floor — vacuous"
    assert not torch.allclose(on.last_spread_belief, off.last_spread_belief, atol=1e-6)
    assert not torch.allclose(on.last_damage_block, off.last_damage_block, atol=1e-6)
    assert not torch.allclose(pi_on, pi_off, atol=1e-6)
    assert torch.isfinite(pi_on).all() and torch.isfinite(vf_on).all()


# --------------------------------------------------------------------------- identity-at-init (M1)
class _Env(gym.Env):
    def __init__(self, dim):
        self.observation_space = gym.spaces.Dict(
            {"observation": gym.spaces.Box(0.0, 1.0, (dim,), np.float32)})
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)
        self._dim = dim

    def reset(self, **kw):
        return {"observation": np.zeros(self._dim, np.float32)}, {}

    def step(self, a):
        return {"observation": np.zeros(self._dim, np.float32)}, 0.0, True, False, {}

    def action_masks(self):
        return np.ones(ACTION_SPACE_SIZE, bool)


def _build_real_policy(**overrides):
    """Build through the SAME path training uses. SB3's ActorCriticPolicy._build() orthogonally
    re-inits every Linear in the extractor (ledger M1), so a zero-init asserted on a bare extractor
    proves nothing about a real run."""
    enc = Gen3ObservationEncoder(load_mappings())
    ek = enc.get_features_extractor_kwargs()
    sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    kw = {**ek, **{k: v for k, v in {**_BASE, **overrides}.items() if k in sig}}
    torch.manual_seed(0)
    model = MaskablePPO(
        Gen3DualHeadMaskablePolicy, DummyVecEnv([lambda: _Env(enc.dimension)]),
        n_steps=16, batch_size=16, n_epochs=1, device="cpu",
        policy_kwargs={"features_extractor_class": Gen3FeaturesExtractor,
                       "features_extractor_kwargs": kw,
                       "net_arch": dict(pi=[64], vf=[64])},
    )
    return model


def test_prefuse_proj_is_zero_after_a_real_policy_build():
    """The injection is documented identity-at-init; SB3 would clobber that without the M1 guard."""
    fe = _build_real_policy(damage_op_prefuse=True).policy.features_extractor
    assert "prefuse_proj" in fe._identity_init_zeroed, (
        "prefuse_proj was not captured by the identity-init snapshot — it must be zeroed BEFORE "
        "__init__ returns for restore_identity_init to protect it."
    )
    assert float(fe.prefuse_proj.weight.abs().max()) == 0.0
    assert float(fe.prefuse_proj.bias.abs().max()) == 0.0


def test_real_policy_forward_runs_and_the_block_reaches_the_heads():
    """End-to-end through the production construction path: the unified forward works and the head
    projections are sized including the full damage block."""
    model = _build_real_policy(damage_op_prefuse=True)
    fe = model.policy.features_extractor
    obs = {"observation": torch.zeros(2, fe.layout["total_dim"])}
    with torch.no_grad():
        pi, vf = fe(obs)
    assert pi.shape == vf.shape == (2, fe.projection_dim)
    assert fe.projection_input_dim >= fe.damage_op.out_dim


# --------------------------------------------------------------------------- versioning
def _mv(prefuse):
    policy_kwargs = {"features_extractor_kwargs": {"damage_op_prefuse": prefuse}}
    return ModelVersion.from_layout_and_policy_kwargs(_layout, policy_kwargs)


def test_version_gate_and_migration():
    assert MODEL_CONFIG_VERSION >= 50
    on, off = _mv(True), _mv(False)
    with pytest.raises(ModelVersionError, match="damage_op_prefuse"):
        on.check_compatible(off)
    with pytest.raises(ModelVersionError, match="damage_op_prefuse"):
        off.check_compatible(on)
    on.check_compatible(_mv(True))                  # like-for-like passes
    migrated = _migrate_config({"config_version": 49})
    assert migrated["damage_op_prefuse"] is False
    assert migrated["config_version"] >= 50
