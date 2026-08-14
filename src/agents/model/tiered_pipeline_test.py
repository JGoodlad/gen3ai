"""The UNCONDITIONAL pre-transformer belief + physics stack (`gen3_tiered_pipeline_v1`).

Supersedes `move_belief_prefuse_test.py` (v32) and `damage_op_prefuse_test.py` (v50), both of which
were A/B tests on flags that no longer exist. The claim under test is now unconditional:

    beliefs ONCE (T0, pre-attention) -> physics ONCE (T1) -> attention -> the readouts (T2/T3)

Pins: the move-belief posterior is computed and stashed BEFORE the transformer; the DamageOperator
runs exactly once and the lean `discrete_incoming` kernel never; the physics reaches the trunk ahead
of attention through a zero-init `prefuse_proj` that survives SB3's ortho-init (ledger M1); gradient
still reaches all three belief heads; and the deleted flags are gone from the constructor, the
version record and the CLI — with the v71 migration REFUSING, rather than quietly popping, a config
that recorded a placement this codebase can no longer run.

The ORDER itself (a T0 leg may not run after T1, a T2 tensor may not reach T0) is asserted
separately and more generally in `tier_contract_test.py`.
"""
import inspect

import gymnasium as gym
import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.action.constants import ACTION_SPACE_SIZE
from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.model.model_version import (MODEL_CONFIG_VERSION, ModelVersionError, _migrate_config)
from agents.model.policy import Gen3DualHeadMaskablePolicy
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_mappings = load_mappings()
_layout = Gen3ObservationEncoder(_mappings).get_layout()

# The full belief+physics stack: every consumer the op reads (move / spread / HP-type belief) plus
# the head-block producers, so the test exercises the production shape.
_BASE = dict(attend_unrevealed_opponents=True, opp_belief_slots=True,
             move_belief_mode="revealed", move_prior_fusion=True,
             move_latent=True, damage_op=True, damage_outgoing=True, damage_topk_k=5,
             damage_matrices_incoming=True, spread_belief=True)

#: The three flags step 3/4 deleted. They must not merely be ignored — a caller that still passes
#: one has a config this codebase cannot honour, and must be told so.
_DELETED_FLAGS = ("move_belief_prefuse", "damage_op_prefuse", "damage_reattend")


def _make_model(**kw):
    obs_space = gym.spaces.Box(0.0, 1.0, shape=(_layout["total_dim"],), dtype=np.float32)
    return Gen3FeaturesExtractor(obs_space, layout=_layout, mappings=_mappings, **kw)


def _obs(batch=4, seed=1234):
    g = torch.Generator().manual_seed(seed)
    return {"observation": torch.rand(batch, _layout["total_dim"], generator=g)}


# --------------------------------------------------------------------------- the flags are GONE
@pytest.mark.parametrize("flag", _DELETED_FLAGS)
def test_the_deleted_placement_flags_are_not_constructor_arguments(flag):
    """Silently ignoring an unknown kwarg is how a config quietly means something else."""
    assert flag not in inspect.signature(Gen3FeaturesExtractor.__init__).parameters
    with pytest.raises(TypeError, match=flag):
        _make_model(**_BASE, **{flag: True})


def test_prefuse_proj_exists_exactly_when_the_operator_does():
    """The injection projection used to be gated on the flag; it is now gated on the op itself, which
    is what keeps the production state_dict unchanged across the deletion."""
    with_op = _make_model(**_BASE)
    assert with_op.prefuse_proj is not None
    assert {"prefuse_proj.weight", "prefuse_proj.bias"} <= set(with_op.state_dict())
    no_op = _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed")
    assert no_op.prefuse_proj is None
    assert not [k for k in no_op.state_dict() if k.startswith("prefuse_proj.")]


def test_entity_seats_now_require_only_the_operator():
    """The E4 gate used to name `damage_op_prefuse`; the tiered order supplies that guarantee, so the
    requirement collapses to `damage_op` + `move_latent` and must still FAIL LOUD without them."""
    with pytest.raises(ValueError, match="entity_topk_seats"):
        _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed",
                    move_latent=True, damage_op=False, entity_topk_seats=5)


# --------------------------------------------------------------------------- T0: the move belief
def test_move_belief_posterior_is_computed_before_the_transformer():
    """THE step-3 property, asserted directly: the move logits exist by the time attention starts."""
    m = _make_model(**_BASE).eval()
    order = []
    real_logits, real_tt = m.move_belief.move_logits, m.team_transformer.forward

    def logits(*a, **kw):
        order.append("move_belief")
        return real_logits(*a, **kw)

    def tt(*a, **kw):
        order.append("transformer")
        assert m.last_move_belief_logits is not None, (
            "the transformer ran before the move belief was stashed")
        return real_tt(*a, **kw)

    m.move_belief.move_logits, m.team_transformer.forward = logits, tt
    with torch.no_grad():
        m(_obs())
    assert order == ["move_belief", "transformer"], order
    assert order.count("move_belief") == 1, "the move belief must be computed exactly once"


def test_move_belief_logits_are_stashed_for_the_downstream_consumers():
    """The op + the BCE aux loss + the prober all read `last_move_belief_logits`."""
    m = _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed").eval()
    with torch.no_grad():
        m(_obs(batch=2))
    assert m.last_move_belief_logits is not None
    assert m.last_move_belief_logits.shape[0] == 2


@pytest.mark.parametrize("mode", ["revealed", "unrevealed", "both"])
def test_every_move_belief_mode_reinjects_finitely(mode):
    """The reinjection mask differs per mode; each must survive the single placement."""
    m = _make_model(attend_unrevealed_opponents=True, move_belief_mode=mode).eval()
    with torch.no_grad():
        pi, vf = m(_obs(batch=2))
    assert torch.isfinite(pi).all() and torch.isfinite(vf).all(), mode


def test_grad_reaches_the_move_head():
    m = _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed").train()
    pi, vf = m(_obs(batch=6))
    (pi.sum() + vf.sum()).backward()
    g = m.move_belief.move_head.weight.grad
    assert g is not None and g.abs().sum() > 0


# --------------------------------------------------------------------------- T1: the operator
def test_operator_runs_exactly_once_and_the_lean_kernels_never():
    """THE headline property: one full op call per forward, and zero calls to the lean
    `discrete_incoming` kernel (whose only caller was the deleted between-layers refine loop)."""
    model = _make_model(**_BASE)
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
    assert calls == {"full": 1, "lean": 0}, calls


def test_physics_reaches_the_trunk_before_attention():
    m = _make_model(**_BASE).eval()
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


def test_the_full_block_is_emitted_at_full_width():
    """Ledger P1: the op block is the policy's largest measured dependency. Collapsing the two
    placements into one must not shrink it."""
    m = _make_model(**_BASE).eval()
    with torch.no_grad():
        m(_obs())
    assert m.last_damage_block is not None
    assert m.last_damage_block.shape[1] == m.damage_op.out_dim


def test_grad_reaches_all_three_belief_heads():
    """One computation, one gradient path: the op's damage gradient must still train the move /
    spread / HP-type belief heads (that gradient is the whole reason the op is differentiable)."""
    m = _make_model(**_BASE).train()
    pi, vf = m(_obs())
    (pi.sum() + vf.sum()).backward()
    for head in ("move_belief", "spread_belief", "hp_type_belief_head"):
        mod = getattr(m, head)
        grads = [p.grad for p in mod.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads), f"no gradient reached {head}"


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
    return MaskablePPO(
        Gen3DualHeadMaskablePolicy, DummyVecEnv([lambda: _Env(enc.dimension)]),
        n_steps=16, batch_size=16, n_epochs=1, device="cpu",
        policy_kwargs={"features_extractor_class": Gen3FeaturesExtractor,
                       "features_extractor_kwargs": kw,
                       "net_arch": dict(pi=[64], vf=[64])},
    )


def test_prefuse_proj_is_zero_after_a_real_policy_build():
    """The injection is documented identity-at-init; SB3 would clobber that without the M1 guard."""
    fe = _build_real_policy().policy.features_extractor
    assert "prefuse_proj" in fe._identity_init_zeroed, (
        "prefuse_proj was not captured by the identity-init snapshot — it must be zeroed BEFORE "
        "__init__ returns for restore_identity_init to protect it.")
    assert float(fe.prefuse_proj.weight.abs().max()) == 0.0
    assert float(fe.prefuse_proj.bias.abs().max()) == 0.0


def test_real_policy_forward_runs_and_the_block_reaches_the_heads():
    """End-to-end through the production construction path."""
    model = _build_real_policy()
    fe = model.policy.features_extractor
    obs = {"observation": torch.zeros(2, fe.layout["total_dim"])}
    with torch.no_grad():
        pi, vf = fe(obs)
    assert pi.shape == vf.shape == (2, fe.projection_dim)
    # gen3_no_concat_v1: the block no longer widens the projections; the op reaches the heads
    # via the pointer cells / prefuse injection / the vf-only seed window instead.
    assert fe.assembler.seed_readout is not None
    assert fe.assembler.seed_readout.last_outputs is not None, "the seed window must have run"
    assert fe.projection_input_dim < fe.damage_op.out_dim + 471, \
        "pi regained op width — the concat came back"


# --------------------------------------------------------------------------- versioning
def test_config_version_records_the_deletion():
    assert MODEL_CONFIG_VERSION >= 71


@pytest.mark.parametrize("flag", _DELETED_FLAGS)
def test_the_deleted_flags_are_not_model_version_fields(flag):
    from agents.model.model_version import ModelVersion
    assert flag not in {f.name for f in __import__("dataclasses").fields(ModelVersion)}


@pytest.mark.parametrize("flag,bad", [("move_belief_prefuse", False),
                                      ("damage_op_prefuse", False),
                                      ("damage_reattend", True)])
def test_any_config_recording_a_deleted_placement_flag_is_below_the_floor(flag, bad):
    """The v71 REFUSE-not-pop branch is pre-floor since gen3_ctx_dedup_v1 raised
    MIGRATION_FLOOR: every config old enough to carry one of the deleted placement flags is a
    pre-generation checkpoint, so the blanket floor refusal subsumes the per-flag judgment —
    a post-ordering checkpoint still cannot load into the pre-ordering forward, it just gets
    the generation-level message. Either value of the flag is refused alike."""
    with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
        _migrate_config({"config_version": 70, flag: bad})
    with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
        _migrate_config({"config_version": 70, flag: not bad})


def test_a_current_config_never_carries_the_deleted_flags():
    """The fields must not resurface in a current ModelVersion — the forward they selected
    between no longer exists."""
    import dataclasses as _dc
    from agents.model.model_version import ModelVersion
    fields = {f.name for f in _dc.fields(ModelVersion)}
    for flag in _DELETED_FLAGS:
        assert flag not in fields
