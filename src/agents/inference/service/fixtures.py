"""Policies the service's tests and benchmark serve when no checkpoint is named.

A FRESH production-arch policy is a vacuous parity probe: its pointer head is zero-initialised, so
its logits are constant per row and any log-prob comparison passes (measured 2026-09-29: an AOT
miscompile read max|dlogp| 0.0 on a fresh policy and 0.68 on a real one). Perturbing every
parameter by seeded noise makes the logits depend on the whole forward. The perturbation is the
shared one (`agents.model.parity_probe`, gen3_fresh_parity_probe_v1) — seed ``1000 + seed``, scale
0.05, drawn on CPU from a private generator, i.e. bit-identical to this fixture's original noise.
"""
from __future__ import annotations

from typing import Any

from agents.model.parity_probe import PERTURB_SCALE, perturb_


def perturbed_fresh_policy(seed: int, scale: float = PERTURB_SCALE) -> Any:
    """A seeded, untrained production-arch policy (CPU, eval) with every parameter perturbed."""
    from main.fresh_checkpoint import build_fresh_model

    model, _, _ = build_fresh_model(seed)
    return perturb_(model.policy.eval(), seed=1000 + seed, scale=scale)
