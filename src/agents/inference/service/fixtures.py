"""Policies the service's tests and benchmark serve when no checkpoint is named.

A FRESH production-arch policy is a vacuous parity probe: its pointer head is zero-initialised, so
its logits are constant per row and any log-prob comparison passes (measured 2026-09-29: an AOT
miscompile read max|dlogp| 0.0 on a fresh policy and 0.68 on a real one). Perturbing every
parameter by seeded noise makes the logits depend on the whole forward.
"""
from __future__ import annotations

from typing import Any

import torch


def perturbed_fresh_policy(seed: int, scale: float = 0.05) -> Any:
    """A seeded, untrained production-arch policy (CPU, eval) with every parameter perturbed."""
    from main.fresh_checkpoint import build_fresh_model

    model, _, _ = build_fresh_model(seed)
    policy = model.policy.eval()
    g = torch.Generator().manual_seed(1000 + seed)
    with torch.no_grad():
        for p in policy.parameters():
            p.add_(torch.randn(p.shape, generator=g) * scale)
    return policy
