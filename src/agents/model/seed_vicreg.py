"""VICReg variance+covariance floor on the multi-seed critic readout (gen3_seed_vicreg_v1, v62).

The pre-registered trigger in `seed_diagnostics.py` FIRED on gen-5 (ai_v9_06, 2026-08-10):
`value_seeds/out_effective_rank` = 1.0 and `value_seeds/out_cos` = 1.000 sustained from step 196k through
15M+ — the k=4 seed outputs of `MultiSeedValueReadout` are identical, so the critic pays for
k reads and receives one. This module is the wiring that trigger pre-registered: a VICReg-style
regularizer on the seed OUTPUTS (activations, not the queries — the queries already differ at
cos 0.33; it is the attention patterns that collapsed), enabled with `--value-seed-vicreg-coef` on
the NEXT run (resume-immutable, the vf_coef class — see `model_version.check_value_seed_vicreg`).

Two terms, BOTH mandatory (the z_arch precedent: its VICReg covariance term was specified but
never wired, and ≈2/3 of the code's energy collapsed into one shared direction — discovered
post-hoc by a probe):

- **Variance** (per sample, across seeds): hinge each per-dim std-across-seeds up to
  `SEED_VICREG_GAMMA`. Directly pushes the k outputs apart for the same board.
- **Covariance** (across the batch, between seeds): penalize off-diagonal entries of the
  cross-seed covariance of the batch-centered outputs. This is what kills the cheat the
  variance term alone admits — seeds = one shared batch-varying vector + per-seed constant
  offsets satisfies the variance hinge while carrying ZERO extra information; only
  decorrelating the batch-varying parts forces the seeds to READ different things.
  `seed_vicreg_test.py::test_constant_offset_cheat_is_caught_by_covariance_term` is the
  is-it-actually-wired proof.

Pure torch, no side effects. The loss site lives in `instrumented_ppo.train()` (folded into
the PPO loss before backward, per minibatch, on the extractor's `last_outputs` stash);
coef=0.0 skips everything (byte-identical).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

SEED_VICREG_GAMMA = 1.0        # std hinge target per dim (the VICReg γ)
SEED_VICREG_VAR_WEIGHT = 1.0   # variance-term weight inside the (already coef-scaled) loss
SEED_VICREG_COV_WEIGHT = 1.0   # covariance-term weight inside the (already coef-scaled) loss


def seed_vicreg_loss(outputs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    """VICReg variance+covariance loss on the seed readout activations.

    outputs: ``[B, k, D]`` — the k per-seed readout vectors for a batch, WITH grad (the
    live `MultiSeedValueReadout.last_outputs` stash from this minibatch's forward).

    Returns ``(loss, metrics)`` where loss keeps the graph and metrics are detached floats:
    ``{"value_seeds/vicreg_var_term": ..., "value_seeds/vicreg_cov_term": ...}``.
    """
    assert outputs.dim() == 3, f"outputs must be [B, k, D], got {tuple(outputs.shape)}"
    batch, k, dim = outputs.shape
    assert k >= 2, f"VICReg over seeds needs k >= 2, got k={k}"

    # Variance: per-sample std across the seed axis, hinged up to gamma per dim.
    std_k = outputs.std(dim=1, unbiased=False)                          # [B, D]
    var_term = torch.relu(SEED_VICREG_GAMMA - std_k).mean()

    # Covariance: batch-center each seed, then penalize cross-seed covariance of the
    # batch-VARYING parts. Constant per-seed offsets vanish under the centering, so this
    # term sees exactly the information content the variance term is blind to.
    centered = outputs - outputs.mean(dim=0, keepdim=True)              # [B, k, D]
    cov = torch.einsum("bid,bjd->ij", centered, centered) / (batch * dim)  # [k, k]
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_term = off_diag.pow(2).sum() / (k * (k - 1))

    loss = SEED_VICREG_VAR_WEIGHT * var_term + SEED_VICREG_COV_WEIGHT * cov_term
    return loss, {
        "value_seeds/vicreg_var_term": float(var_term.detach()),
        "value_seeds/vicreg_cov_term": float(cov_term.detach()),
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
