"""`DenseAuxHead` — the DENSE AUXILIARY readout's model half (`gen3_dense_aux_v1`).

Two claims carry the arm and both are asserted at a LARGE random weight rather than at init, for
the reason the true-team build recorded: a zero-init module is trivially invisible, and the
properties have to hold for the TRAINED network the arm is actually read on.

1. **The head's OUTPUT never enters pi or vf.** It is not called by the forward at all, so pi/vf
   are bit-identical for an arbitrary perturbation of this module — a strictly stronger statement
   than the win head's, which is in the forward.
2. **Its gradient DOES reach the shared trunk**, because the training term feeds it a live
   `value_pooled`. That is the arm rather than a leak: the whole point of a dense auxiliary target
   is gradient along the per-entity axes one pooled bit cannot carry. Asserted in both directions —
   a policy-only loss leaves the head with no gradient, and an aux-only loss moves a trunk
   parameter.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.action.constants import ACTION_SPACE_SIZE
from agents.model.arch_constants import D_MODEL
from agents.model.dense_aux_head import (
    DENSE_AUX_DIM_OUT, DENSE_AUX_HP, DENSE_AUX_LAYOUT, DENSE_AUX_SLOTS, DENSE_AUX_SURVIVAL,
    DENSE_AUX_TURNS, DenseAuxHead,
)
from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.model.model_version import MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError
from agents.model.policy import Gen3DualHeadMaskablePolicy
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings


def _build(seed=7, **kw):
    m = load_mappings()
    layout = Gen3ObservationEncoder(m).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    if kw.get("dense_aux"):
        # The head DECLARES `requires=("win_prob_mode",)` and the constructor enforces it: the
        # dense targets exist to shape the trunk the win-prob critic reads.
        kw.setdefault("win_prob_mode", "shaping")
    torch.manual_seed(seed)
    return Gen3FeaturesExtractor(space, layout=layout, mappings=m, **kw), layout


# ── the layout ───────────────────────────────────────────────────────────────────────────────────

def test_the_layout_is_contiguous_and_complete():
    """A hand-written slice table that leaves a gap would silently stop scoring an output; the loss
    iterates exactly these blocks, so an unlisted column would be built, forwarded and ignored."""
    spans = sorted(DENSE_AUX_LAYOUT.values())
    assert spans[0][0] == 0 and spans[-1][1] == DENSE_AUX_DIM_OUT
    for (_a, b), (c, _d) in zip(spans, spans[1:]):
        assert b == c, "the dense-aux output layout has a gap or an overlap"
    assert DENSE_AUX_SLOTS == 12 and DENSE_AUX_DIM_OUT == 25


# ── the head itself ──────────────────────────────────────────────────────────────────────────────

def test_zero_init_means_p_is_one_half_everywhere():
    """The honest state of knowledge for a head that has seen no label — and the property that
    keeps the trunk untouched until the output layer itself has moved off zero."""
    h = DenseAuxHead()
    assert float(h.net[2].weight.abs().max()) == 0.0
    assert float(h.net[2].bias.abs().max()) == 0.0
    out = h(torch.randn(4, D_MODEL))
    assert out.shape == (4, DENSE_AUX_DIM_OUT)
    assert float(out.abs().max()) == 0.0
    assert torch.equal(torch.sigmoid(out), torch.full_like(out, 0.5))


def test_a_wrong_shaped_input_raises_naming_its_only_input():
    h = DenseAuxHead()
    with pytest.raises(ValueError, match="value_pooled"):
        h(torch.randn(2, 3, D_MODEL))


# ── the arm-defining properties ──────────────────────────────────────────────────────────────────

def test_pi_and_vf_are_bit_identical_at_an_arbitrary_weight():
    """Stronger than the win head's contract: the forward never CALLS this head, so no weight in
    it can move either output. Asserted against a large random perturbation, not at init."""
    fe, layout = _build(dense_aux=True)
    fe.eval()
    obs = {"observation": torch.rand(4, layout["total_dim"])}
    with torch.no_grad():
        pi_a, vf_a = fe(obs)
        for p in fe.dense_aux_head.parameters():
            p.normal_(std=3.0)
        pi_b, vf_b = fe(obs)
    assert torch.equal(pi_a, pi_b), "the dense-aux head reached the POLICY features"
    assert torch.equal(vf_a, vf_b), "the dense-aux head reached the VALUE features"


def test_a_policy_only_loss_leaves_the_head_with_no_gradient():
    fe, layout = _build(dense_aux=True)
    fe.train()
    pi, _vf = fe({"observation": torch.rand(3, layout["total_dim"])})
    pi.sum().backward()
    for name, p in fe.dense_aux_head.named_parameters():
        assert p.grad is None or float(p.grad.abs().max()) == 0.0, name


def test_the_aux_loss_DOES_reach_the_shared_trunk():
    """The other direction, and the one that makes it an arm rather than a diagnostic. The training
    term applies the head to a LIVE `value_pooled`, so its gradient must reach a trunk parameter —
    a detached input here would make the whole build a no-op wearing a coefficient."""
    fe, layout = _build(dense_aux=True)
    fe.train()
    with torch.no_grad():                       # off zero-init, or W_out = 0 kills the chain rule
        fe.dense_aux_head.net[2].weight.normal_(std=0.5)
    fe({"observation": torch.rand(3, layout["total_dim"])})
    pooled = fe.last_value_pooled
    assert pooled is not None and pooled.requires_grad
    fe.dense_aux_head(pooled).sum().backward()
    moved = [n for n, p in fe.named_parameters()
             if not n.startswith("dense_aux_head") and p.grad is not None
             and float(p.grad.abs().max()) > 0.0]
    assert moved, "the dense-aux loss reached no shared-trunk parameter — the input was detached"


def test_off_builds_nothing_and_on_is_bit_identical_to_off():
    """The append-never-insert claim: the head is built LAST, so ON at init must leave every
    earlier module's RNG draw — and therefore the whole forward — untouched."""
    # The BASELINE must carry the same `win_prob_mode`, or the comparison would be measuring the
    # win head's own RNG draw rather than this one's absence.
    fe_off, layout = _build(seed=11, win_prob_mode="shaping")
    fe_on, _ = _build(seed=11, dense_aux=True)
    assert fe_off.dense_aux_head is None and fe_off.dense_aux is False
    assert fe_on.dense_aux_head is not None and fe_on.dense_aux is True
    obs = {"observation": torch.rand(2, layout["total_dim"])}
    fe_off.eval(); fe_on.eval()
    with torch.no_grad():
        pi_off, vf_off = fe_off(obs)
        pi_on, vf_on = fe_on(obs)
    assert torch.equal(pi_off, pi_on)
    assert torch.equal(vf_off, vf_on)


def test_the_zero_init_survives_sb3s_ortho_pass_on_a_real_build():
    """`restore_identity_init` captures its set BY OBSERVATION, so this head should be covered
    automatically — and "should be picked up" is exactly what the M1 bug was made of."""
    enc = Gen3ObservationEncoder(load_mappings())
    dim = enc.dimension

    class _Env(gym.Env):
        observation_space = gym.spaces.Dict(
            {"observation": gym.spaces.Box(0.0, 1.0, (dim,), np.float32)})
        action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)

        def reset(self, **kw):
            return {"observation": np.zeros(dim, np.float32)}, {}

        def step(self, a):
            return {"observation": np.zeros(dim, np.float32)}, 0.0, True, False, {}

        def action_masks(self):
            return np.ones(ACTION_SPACE_SIZE, bool)

    torch.manual_seed(0)
    model = MaskablePPO(
        Gen3DualHeadMaskablePolicy, DummyVecEnv([lambda: _Env()]),
        n_steps=16, batch_size=16, n_epochs=1, device="cpu",
        policy_kwargs={
            "features_extractor_class": Gen3FeaturesExtractor,
            "features_extractor_kwargs": {
                **enc.get_features_extractor_kwargs(), "dense_aux": True,
                "win_prob_mode": "shaping"},
            "net_arch": dict(pi=[64], vf=[64])},
    )
    fe = model.policy.features_extractor
    assert not fe.dense_aux_head.net[2].weight.any(), \
        "SB3's ortho-init clobbered the dense-aux output layer — restore_identity_init missed it"
    with torch.no_grad():
        probs = torch.sigmoid(fe.dense_aux_head(torch.randn(2, D_MODEL)))
    assert torch.equal(probs, torch.full_like(probs, 0.5))


# ── the v117 version gate ────────────────────────────────────────────────────────────────────────

def _ver(*, dense_aux=False, coef=0.0):
    mappings = load_mappings()
    ek = Gen3ObservationEncoder(mappings).get_features_extractor_kwargs()
    ek["dense_aux"] = dense_aux
    # BOTH sides carry the win head, always: this file is about the `dense_aux` gate, and letting
    # `win_prob_mode` differ would make every comparison below fail on the OTHER gate first.
    ek["win_prob_mode"] = "shaping"
    pk = {"features_extractor_class": Gen3FeaturesExtractor, "features_extractor_kwargs": ek,
          "net_arch": [512, 512]}
    return ModelVersion.from_layout_and_policy_kwargs(
        ek["layout"], pk, win_prob_dense_aux=coef)


def test_version_records_both_halves():
    v = _ver(dense_aux=True, coef=1.0)
    assert v.dense_aux is True and v.win_prob_dense_aux == 1.0
    assert v.config_version == MODEL_CONFIG_VERSION
    off = _ver()
    assert off.dense_aux is False and off.win_prob_dense_aux == 0.0


def test_check_compatible_REJECTS_a_flipped_head_but_allows_a_RE_DOSE():
    """The BUILD is fixed for a run's lifetime — its params are the state_dict delta and nothing
    downstream would fail on a mismatch. The DOSE is not: re-dosing an existing head is a
    legitimate resume, which is exactly why the two are separate recorded fields."""
    on, off = _ver(dense_aux=True, coef=1.0), _ver()
    with pytest.raises(ModelVersionError, match="dense_aux mismatch"):
        on.check_compatible(off)
    with pytest.raises(ModelVersionError, match="dense_aux mismatch"):
        off.check_compatible(on)
    _ver(dense_aux=True, coef=0.5).check_compatible(_ver(dense_aux=True, coef=2.0))


def test_a_pre_v117_config_migrates_to_off():
    """Not a guess about an old run: neither field existed, so no run could have set either."""
    from agents.model.model_version.migrations import _migrate_config
    data = _migrate_config({"config_version": 116})
    assert data["dense_aux"] is False and data["win_prob_dense_aux"] == 0.0
    assert data["config_version"] == MODEL_CONFIG_VERSION


def test_the_survival_and_hp_blocks_are_ordered_ours_then_theirs():
    """The label builder and the loss both index `[:6]` as ours and `[6:]` as theirs; a reordered
    layout would silently swap which side every `aux_*_own` meter describes."""
    assert DENSE_AUX_SURVIVAL == (0, 12)
    assert DENSE_AUX_HP == (12, 24)
    assert DENSE_AUX_TURNS == (24, 25)
