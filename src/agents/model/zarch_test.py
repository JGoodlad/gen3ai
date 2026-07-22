"""z_arch team-archetype latent + head FiLM (gen3_zarch_film_v1, v44) — module build,
byte-identical-off, identity-at-init, team-static invariance, detached-read gradient isolation,
the recon/VICReg aux math, and the v44 version gate."""

import numpy as np
import pytest
import torch
from gymnasium import spaces

from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.observation.constants import (
    POKEMON_FULL_DIM,
    POKEMON_HP_OFFSET,
    POKEMON_CONDITION_OFFSET,
    POKEMON_SPREAD_OFFSET,
)
from agents.model.features_extractor import Gen3FeaturesExtractor, TEAM_SIZE
from agents.model.model_version import ModelVersion, ModelVersionError, _migrate_config

ZDIM = 16  # small latent for test speed (any positive value is legal)


@pytest.fixture(scope="module")
def ek_and_space():
    mappings = load_mappings()
    ek = Gen3ObservationEncoder(mappings).get_features_extractor_kwargs()
    total = ek["layout"]["total_dim"]
    space = spaces.Dict({
        "observation": spaces.Box(-np.inf, np.inf, (total,), np.float32),
        "action_mask": spaces.Box(0, 1, (11,), np.int8),
    })
    return ek, space, total


def _build(ek, space, mode="off", dim=0):
    return Gen3FeaturesExtractor(space, **{**ek, "zarch_film": mode, "zarch_dim": dim})


def _varied_obs(total: int, batch: int = 3, seed: int = 0) -> torch.Tensor:
    """A zeros obs with FLOAT-only regions randomized (per-mon HP + spread for all 12 slots) —
    raw randn would corrupt the float-encoded categorical-ID slots (embedding index error)."""
    g = torch.Generator().manual_seed(seed)
    obs = torch.zeros(batch, total)
    for slot in range(12):
        base = slot * POKEMON_FULL_DIM
        obs[:, base + POKEMON_HP_OFFSET] = torch.rand(batch, generator=g)
        s = base + POKEMON_SPREAD_OFFSET
        obs[:, s:s + 18] = torch.rand(batch, 18, generator=g)
    return obs


# ---------------------------------------------------------------- build / validation

def test_invalid_mode_raises(ek_and_space):
    ek, space, _ = ek_and_space
    with pytest.raises(ValueError):
        _build(ek, space, "bogus", ZDIM)


def test_on_requires_positive_dim(ek_and_space):
    ek, space, _ = ek_and_space
    with pytest.raises(ValueError):
        _build(ek, space, "heads", 0)


def test_off_requires_zero_dim(ek_and_space):
    ek, space, _ = ek_and_space
    with pytest.raises(ValueError):
        _build(ek, space, "off", ZDIM)


def test_off_builds_no_modules(ek_and_space):
    ek, space, total = ek_and_space
    f = _build(ek, space)
    assert f.zarch_encoder is None and f.film_pi is None and f.film_vf is None
    f({"observation": torch.zeros(2, total)})
    assert f.last_zarch is None and f.last_zarch_recon_logits is None


# ---------------------------------------------------------------- identity-at-init

def test_identity_at_init_forward_equals_baseline(ek_and_space):
    """ON with the OFF extractor's shared weights loaded reproduces the OFF forward EXACTLY:
    the FiLM generators are zero-init (Δγ=Δβ=0) so the modulation is the identity, and the
    projection widths are unchanged (FiLM is applied post-projection)."""
    ek, space, total = ek_and_space
    f_off = _build(ek, space)
    f_on = _build(ek, space, "heads", ZDIM)
    missing, unexpected = f_on.load_state_dict(f_off.state_dict(), strict=False)
    assert not unexpected                        # every OFF key exists (same shapes) in ON
    assert all(k.startswith(("zarch_encoder.", "film_pi.", "film_vf.")) for k in missing)
    obs = {"observation": _varied_obs(total)}
    with torch.no_grad():
        pi_off, vf_off = f_off(obs)
        pi_on, vf_on = f_on(obs)
    assert torch.equal(pi_off, pi_on)
    assert torch.equal(vf_off, vf_on)


def test_nonzero_generator_modulates_its_head_only(ek_and_space):
    """A non-zero pi generator changes the pi features but leaves vf untouched (separate
    per-head generators — the policy-vs-value routing is independent)."""
    ek, space, total = ek_and_space
    f = _build(ek, space, "heads", ZDIM)
    obs = {"observation": _varied_obs(total, batch=2)}
    with torch.no_grad():
        pi0, vf0 = f(obs)
        f.film_pi.bias.fill_(0.5)                # Δβ ≠ 0 on the policy head only
        pi1, vf1 = f(obs)
    assert not torch.equal(pi0, pi1)
    assert torch.equal(vf0, vf1)


# ---------------------------------------------------------------- team-static invariance

def test_z_is_invariant_to_dynamic_state(ek_and_space):
    """z_arch reads ONLY the invariant per-mon facts: perturbing our active mon's HP + status
    condition leaves z bit-identical; perturbing the spread block (invariant input) moves it."""
    ek, space, total = ek_and_space
    f = _build(ek, space, "heads", ZDIM)
    base = torch.zeros(1, total)
    dyn = base.clone()
    dyn[0, POKEMON_HP_OFFSET] = 0.37                       # our slot-0 HP fraction
    dyn[0, POKEMON_CONDITION_OFFSET + 1] = 1.0             # our slot-0 burned
    stat = base.clone()
    stat[0, POKEMON_SPREAD_OFFSET] = 0.9                   # our slot-0 spread (invariant input)
    with torch.no_grad():
        f({"observation": base}); z_base = f.last_zarch.clone()
        f({"observation": dyn}); z_dyn = f.last_zarch.clone()
        f({"observation": stat}); z_stat = f.last_zarch.clone()
    assert torch.equal(z_base, z_dyn)
    assert not torch.equal(z_base, z_stat)


def test_z_is_permutation_invariant(ek_and_space):
    """Swapping two of OUR team slots (whole per-mon blocks) leaves z bit-identical (a team is
    a SET — the DeepSets mean is order-free)."""
    ek, space, total = ek_and_space
    f = _build(ek, space, "heads", ZDIM)
    obs = torch.zeros(1, total)
    obs[0, :6 * POKEMON_FULL_DIM] = torch.rand(6 * POKEMON_FULL_DIM)
    swapped = obs.clone()
    a = slice(0, POKEMON_FULL_DIM)
    b = slice(POKEMON_FULL_DIM, 2 * POKEMON_FULL_DIM)
    swapped[0, a], swapped[0, b] = obs[0, b], obs[0, a]
    # NB the raw per-mon block holds float-encoded categorical ids; rand() floors to id 0 on
    # .long(), which is fine — the swap symmetry is what's under test, not the ids' meaning.
    with torch.no_grad():
        f({"observation": obs}); z1 = f.last_zarch.clone()
        f({"observation": swapped}); z2 = f.last_zarch.clone()
    assert torch.allclose(z1, z2, atol=1e-6)


# ---------------------------------------------------------------- gradient isolation

def test_recon_gradient_touches_only_zarch_params(ek_and_space):
    """The detached-read guarantee: the recon BCE's backward reaches the ZArchEncoder's own
    params and NOTHING else — not the embeddings, not the encoder/transformer trunk."""
    ek, space, total = ek_and_space
    f = _build(ek, space, "heads", ZDIM)
    f.zero_grad()
    f.forward_internal({"observation": _varied_obs(total, batch=2)})
    assert f.last_zarch_recon_logits is not None           # grad-enabled path stashes it
    f.last_zarch_recon_logits.sum().backward()
    zarch_grads = [p.grad is not None and p.grad.abs().sum() > 0
                   for p in f.zarch_encoder.parameters()]
    assert any(zarch_grads)
    for mod in (f.embeddings, f.pokemon_encoder, f.team_transformer, f.cls_pool):
        assert all(p.grad is None or p.grad.abs().sum() == 0 for p in mod.parameters())


def test_stash_shapes_and_grad_gating(ek_and_space):
    ek, space, total = ek_and_space
    f = _build(ek, space, "heads", ZDIM)
    layout = ek["layout"]
    f.forward_internal({"observation": torch.zeros(4, total)})
    assert f.last_zarch.shape == (4, ZDIM)
    assert f.last_zarch_recon_logits.shape == (4, layout["max_species"])
    assert f.last_zarch_species_ids.shape == (4, TEAM_SIZE)
    with torch.no_grad():                                  # rollout path: z yes, recon stash no
        f.forward_internal({"observation": torch.zeros(4, total)})
    assert f.last_zarch is not None
    assert f.last_zarch_recon_logits is None and f.last_zarch_species_ids is None


# ---------------------------------------------------------------- the aux-loss math

def _zarch_loss(*a):
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    return InstrumentedMaskablePPO._zarch_loss(*a)


def test_zarch_loss_none_guards():
    z = torch.randn(4, ZDIM)
    assert _zarch_loss(None, None, None) is None
    assert _zarch_loss(z[:1], torch.randn(1, 50), torch.ones(1, 6).long()) is None  # 1-row batch


def test_zarch_loss_recon_and_topk_acc():
    """Perfect logits (huge on the true species, tiny elsewhere) → near-zero BCE + topk acc 1;
    the pad row 0 never counts as a positive."""
    B, S = 4, 50
    ids = torch.randint(1, S, (B, 6))
    logits = torch.full((B, S), -20.0)
    logits.scatter_(1, ids, 20.0)
    recon, vicreg, m = _zarch_loss(torch.randn(B, ZDIM), logits, ids)
    assert m["recon_bce"] < 1e-3 and m["recon_topk_acc"] == 1.0
    # A pad id (0) is zeroed out of the target — confident logit 0 on it must be PENALIZED.
    ids_pad = ids.clone(); ids_pad[:, 0] = 0
    logits_pad = torch.full((B, S), -20.0); logits_pad.scatter_(1, ids_pad, 20.0)
    recon_pad, _, _ = _zarch_loss(torch.randn(B, ZDIM), logits_pad, ids_pad)
    assert recon_pad > recon


def test_zarch_loss_vicreg_floor():
    """Identical z rows across the batch (collapse) → the per-dim hinge saturates at 1;
    high-variance z → ~0."""
    B, S = 8, 50
    ids = torch.randint(1, S, (B, 6))
    logits = torch.zeros(B, S)
    z_collapsed = torch.ones(B, ZDIM)
    _, vic_c, m_c = _zarch_loss(z_collapsed, logits, ids)
    assert float(vic_c) == pytest.approx(1.0) and m_c["std"] == pytest.approx(0.0)
    z_diverse = torch.randn(B, ZDIM) * 5.0
    _, vic_d, _ = _zarch_loss(z_diverse, logits, ids)
    assert float(vic_d) < float(vic_c)


def test_zarch_participation_ratio():
    """The live LUT-vs-style dial: ≈dims on an isotropic cloud, ≈1 on a rank-1 cloud,
    None on degenerate batches (tiny/constant)."""
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO as P
    torch.manual_seed(0)
    iso = torch.randn(5000, ZDIM)
    assert P._zarch_participation_ratio(iso) > 0.8 * ZDIM
    rank1 = torch.randn(5000, 1) * torch.randn(1, ZDIM) + 1e-4 * torch.randn(5000, ZDIM)
    assert P._zarch_participation_ratio(rank1) < 2.0
    assert P._zarch_participation_ratio(None) is None
    assert P._zarch_participation_ratio(torch.randn(2, ZDIM)) is None          # too few rows
    assert P._zarch_participation_ratio(torch.ones(64, ZDIM)) is None          # zero variance


def test_zarch_loss_grad_flows():
    B, S = 4, 50
    z = torch.randn(B, ZDIM, requires_grad=True)
    head = torch.nn.Linear(ZDIM, S)
    recon, vicreg, _ = _zarch_loss(z, head(z), torch.randint(1, S, (B, 6)))
    (recon + vicreg).backward()
    assert z.grad is not None and z.grad.abs().sum() > 0


# ---------------------------------------------------------------- v44 version gate

def _mv(ek, mode, dim):
    policy_kwargs = {"features_extractor_kwargs": {**ek, "zarch_film": mode, "zarch_dim": dim}}
    return ModelVersion.from_layout_and_policy_kwargs(ek["layout"], policy_kwargs)


def test_version_gate_mode_and_dim(ek_and_space):
    ek, _, _ = ek_and_space
    on = _mv(ek, "heads", ZDIM)
    off = _mv(ek, "off", 0)
    with pytest.raises(ModelVersionError, match="zarch_film"):
        on.check_compatible(off)
    with pytest.raises(ModelVersionError, match="zarch_film"):
        off.check_compatible(on)
    other_dim = _mv(ek, "heads", ZDIM * 2)
    with pytest.raises(ModelVersionError, match="zarch_dim"):
        on.check_compatible(other_dim)
    on.check_compatible(_mv(ek, "heads", ZDIM))            # like-for-like passes


def test_migration_defaults_off():
    """A pre-v44 config migrates to zarch off/0 with zero coefs (old models had no modules)."""
    data = {"config_version": 43}
    out = _migrate_config(dict(data))
    assert out["zarch_film"] == "off" and out["zarch_dim"] == 0
    assert out["zarch_recon_coef"] == 0.0 and out["zarch_vicreg_coef"] == 0.0
    # the migration chain always advances to the latest version (v45+ append value_from_dist etc.)
    from agents.model.model_version import MODEL_CONFIG_VERSION
    assert out["config_version"] == MODEL_CONFIG_VERSION


# ---------------------------------------------------------------- per-group grad accumulation

def test_group_grad_accumulator_gates_and_averages():
    """--film-grad-accum-steps mechanics: k=1 passthrough (grads untouched); k=3 → two gated steps
    set grads to None (optimizer skips), the third applies the AVERAGE of all three captures."""
    from agents.training.instrumented_ppo import _GroupGradAccumulator
    p = torch.nn.Parameter(torch.zeros(4))

    # k=1: pure passthrough — grad object untouched, always applies.
    p.grad = torch.ones(4)
    acc = _GroupGradAccumulator([p])
    assert acc.gate(1) is True
    assert torch.equal(p.grad, torch.ones(4))

    # k=3: capture g1,g2 (grad → None), apply mean(g1,g2,g3) on the third.
    acc = _GroupGradAccumulator([p])
    p.grad = torch.full((4,), 1.0)
    assert acc.gate(3) is False and p.grad is None
    p.grad = torch.full((4,), 2.0)
    assert acc.gate(3) is False and p.grad is None
    p.grad = torch.full((4,), 6.0)
    assert acc.gate(3) is True
    assert torch.allclose(p.grad, torch.full((4,), 3.0))   # mean(1, 2, 6)
    # buffer reset: the next cycle starts fresh.
    p.grad = torch.full((4,), 10.0)
    assert acc.gate(3) is False and p.grad is None


def test_noise_scale_advice():
    """The NSR advisor's pure logic: healthy → no warnings; each out-of-band case names its fix;
    a film ratio COVERED by the configured --film-grad-accum-steps warns nothing."""
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO as P
    b = 32768.0
    assert P._noise_scale_advice(1.1, None, 1, b) == []                      # healthy global
    hi = P._noise_scale_advice(2.6, None, 1, b)
    assert len(hi) == 1 and hi[0][0] == "global_high" and "--grad-accum-steps" in hi[0][1]
    lo = P._noise_scale_advice(0.3, None, 1, b)
    assert len(lo) == 1 and lo[0][0] == "global_low" and "lower --grad-accum-steps" in lo[0][1]
    # film ratio 3.4 covered by film accum 4 (applied 0.85) → no warning.
    assert P._noise_scale_advice(1.1, 3.4, 4, b) == []
    # film ratio 8 with accum 1 → warn, recommending ~ceil(ratio).
    fw = P._noise_scale_advice(1.1, 8.2, 1, b)
    assert len(fw) == 1 and fw[0][0] == "film_high" and "--film-grad-accum-steps ~9" in fw[0][1]
    assert P._noise_scale_advice(None, None, 1, b) == []                     # nothing measured
