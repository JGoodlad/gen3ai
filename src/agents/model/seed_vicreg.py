"""VICReg variance+covariance floor on the multi-seed critic readout (gen3_seed_vicreg_v1, v62).

The pre-registered trigger in `seed_diagnostics.py` FIRED on gen-5 (ai_v9_06, 2026-08-10):
`value_seeds/out_effective_rank` = 1.0 and `value_seeds/out_cos` = 1.000 sustained from step 196k
through 15M+ — the k=4 seed outputs of `MultiSeedValueReadout` are identical, so the critic pays
for k reads and receives one. This module is the wiring that trigger pre-registered: a VICReg-style
regularizer on the seed OUTPUTS (activations, not the queries — the queries already differ at
cos 0.13-0.33; it is the attention patterns that collapsed), enabled with
`--value-seed-vicreg-coef` (resume-immutable, the vf_coef class — see
`model_version.check_value_seed_vicreg`).

⚠️ **BOTH TERMS ARE SCALE-RELATIVE, AND THAT IS THE WHOLE DESIGN.** The first cut used ABSOLUTE
targets (a std hinge at γ=1.0, a raw squared-covariance penalty) and it MEASURABLY FAILED on the
first 2M steps of gen-6: `out_effective_rank` stayed pinned at exactly 1.000 while the hinge sat
saturated at 0.997. The diagnosis, measured on the live architecture:

    seed-output RMS 0.207 · cross-seed std 0.0015 · kv-row RMS 0.246 · across-row spread 0.141

γ=1.0 is ~7× the ENTIRE signal's RMS. Structurally each seed's output is a CONVEX COMBINATION of
the same six kv rows, so the achievable cross-seed std is bounded by the row spread (~0.14/dim) —
the target was unreachable except by `kv_proj` inflating its own scale ~10×, which fights every
downstream norm. A saturated hinge still has gradient (slope −1), so the term pushed constantly
and moved the spread ~5× in 2M steps while multiplicity never budged. **An absolute target on a
layer whose scale is learned is a bug, not a tuning choice.**

So the targets are expressed as FRACTIONS of the readout's own scale:

- **Variance** — per-dim cross-seed std divided by that dim's overall RMS (batch+seed, DETACHED),
  hinged up to `SEED_VICREG_GAMMA_REL`. Reads as "the seeds must differ by ≥25% of the feature's
  own scale", which is invariant to whatever magnitude the layer learns and sits comfortably
  inside the ~0.68 structural ceiling implied by the row spread above.
- **Covariance** — the cross-seed CORRELATION matrix (not raw covariance) of the batch-centered
  outputs; the penalty is the mean squared off-diagonal, so it lives in [0,1] and is comparable
  across runs. This is what kills the cheat the variance term alone admits — seeds = one shared
  batch-varying vector + per-seed constant offsets satisfies the variance hinge while carrying
  ZERO extra information; only decorrelating the batch-varying parts forces the seeds to READ
  different things. `seed_vicreg_test.py::test_constant_offset_cheat_is_caught_by_covariance_term`
  is the is-it-actually-wired proof (the z_arch precedent: its covariance term was specified but
  never wired, and ≈2/3 of the code's energy collapsed into one shared direction).

**The one degenerate path a relative target opens, and how it is watched.** The denominator is
detached, so the gradient can only push the numerator UP — but nothing here stops the readout from
SHRINKING its overall magnitude over many steps, which would raise the ratio without adding
information. That is why `value_seeds/out_rms` is logged: if the ratio improves while `out_rms`
falls, the regularizer is being gamed rather than satisfied. Read the two together.

Pure torch, no side effects. The loss site lives in `instrumented_ppo.train()` (folded into the
PPO loss before backward, per minibatch, on the extractor's `last_outputs` stash); coef=0.0 skips
everything (byte-identical).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

# Cross-seed std must reach this FRACTION of each dim's own RMS (see the module docstring for why
# an absolute target was wrong). The structural ceiling implied by the measured kv-row spread is
# ~0.68, so 0.25 is a real ask that stays reachable without inflating the layer's scale.
SEED_VICREG_GAMMA_REL = 0.25
SEED_VICREG_VAR_WEIGHT = 1.0   # variance-term weight inside the (already coef-scaled) loss
SEED_VICREG_COV_WEIGHT = 1.0   # covariance-term weight inside the (already coef-scaled) loss
_EPS = 1e-6


def seed_vicreg_loss(outputs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    """VICReg variance+covariance loss on the seed readout activations, SCALE-RELATIVE.

    outputs: ``[B, k, D]`` — the k per-seed readout vectors for a batch, WITH grad (the
    live `MultiSeedValueReadout.last_outputs` stash from this minibatch's forward).

    Returns ``(loss, metrics)`` where loss keeps the graph and metrics are detached floats:
    ``value_seeds/vicreg_{var_term,cov_term}`` plus ``value_seeds/{out_std_rel,out_rms}`` —
    the achieved fraction and the magnitude watchdog (see the docstring's degenerate-path note).
    """
    assert outputs.dim() == 3, f"outputs must be [B, k, D], got {tuple(outputs.shape)}"
    batch, k, dim = outputs.shape
    assert k >= 2, f"VICReg over seeds needs k >= 2, got k={k}"

    # --- VARIANCE: cross-seed std as a FRACTION of each dim's own scale. The denominator is
    # DETACHED so the gradient can only widen the seeds, never shrink the feature to cheat.
    scale = outputs.detach().pow(2).mean(dim=(0, 1)).sqrt().clamp_min(_EPS)   # [D]
    std_rel = outputs.std(dim=1, unbiased=False) / scale                      # [B, D]
    var_term = torch.relu(SEED_VICREG_GAMMA_REL - std_rel).mean()

    # --- COVARIANCE: cross-seed CORRELATION of the batch-centered outputs. Batch-centering kills
    # the constant per-seed offsets, so this term sees exactly the information content the
    # variance term is blind to; normalising to a correlation puts it in [0,1] (scale-free).
    centered = outputs - outputs.mean(dim=0, keepdim=True)                    # [B, k, D]
    cov = torch.einsum("bid,bjd->ij", centered, centered) / (batch * dim)     # [k, k]
    dstd = cov.diagonal().clamp_min(_EPS).sqrt()                              # [k]
    corr = cov / torch.outer(dstd, dstd)                                      # [k, k], unit diag
    off_diag = corr - torch.diag(torch.diag(corr))
    cov_term = off_diag.pow(2).sum() / (k * (k - 1))

    loss = SEED_VICREG_VAR_WEIGHT * var_term + SEED_VICREG_COV_WEIGHT * cov_term
    return loss, {
        "value_seeds/vicreg_var_term": float(var_term.detach()),
        "value_seeds/vicreg_cov_term": float(cov_term.detach()),
        # The achieved fraction (target = SEED_VICREG_GAMMA_REL) and the magnitude watchdog:
        # a rising ratio with a FALLING out_rms means the term is being gamed, not satisfied.
        "value_seeds/out_std_rel": float(std_rel.mean().detach()),
        "value_seeds/out_rms": float(outputs.detach().pow(2).mean().sqrt()),
    }


def assert_seed_vicreg_wirable(policy) -> None:
    """FAIL LOUD at startup when `--value-seed-vicreg-coef > 0` but the policy has no multi-seed
    readout to regularize (damage_op off → `ProjectionAssembler.seed_readout` is None) —
    never a silent no-op. Raises RuntimeError with the fix."""
    sr = getattr(getattr(getattr(policy, "features_extractor", None),
                         "assembler", None), "seed_readout", None)
    if sr is None:
        raise RuntimeError(
            "--value-seed-vicreg-coef > 0 but the extractor has NO multi-seed value readout "
            "(ProjectionAssembler.seed_readout is None — the config runs without the damage op, "
            "so there are no per-our-mon rows to read). The coefficient would be a silent no-op. "
            "Fix: drop --value-seed-vicreg-coef, or enable the damage-op config the readout requires."
        )
