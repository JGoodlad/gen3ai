"""gen3_gate_grad_coverage_v1 — the per-parameter gradient rule every train-graph compile gate applies
(`compile_trainer.train_verdict` with per-parameter sizes; region R1's arms carry them).

THE DEFECT CLASS (2026-09-29): the GLOBAL gradient cosine is dominated by the largest gradients, so
dropping the whole gradient of a small path (the pointer head's move cells) left it at 1.0000. The
per-parameter relative error is what resolves such a path. (The extractor-only gate's probe loss and
its zero-gradient coverage guard were deleted with that gate, K1 2026-10-02.)

CPU, routine tier, pure.
"""
from __future__ import annotations

import pytest
import torch

from agents.model.compile_trainer import (_MAX_PARAM_GRAD_REL, _PARAM_GRAD_FLOOR,
                                          CompileTrainerError, _cos, train_verdict)


def test_per_param_rule_catches_a_small_path_the_global_cosine_cannot():
    """One big parameter dominates the norm; a 10% error on a small one leaves the global cosine
    above 0.9999 and must still fail."""
    big = torch.full((1000,), 10.0)
    small = torch.linspace(0.1, 1.0, 10)
    e = torch.cat([big, small])
    c = torch.cat([big, small * 1.1])
    assert _cos(c, e) >= 0.9999
    f = torch.randn(3, 4, generator=torch.Generator().manual_seed(0))
    sizes = torch.tensor([1000, 10])

    def arm(grad):
        return {"features": f, "grad": grad, "grad_sizes": sizes}
    with pytest.raises(CompileTrainerError, match="per-param.*FAILED"):
        train_verdict(eager=arm(e), compiled=arm(c), precision="highest", param_names=["B", "S"])
    assert "per-param" in train_verdict(eager=arm(e), compiled=arm(e.clone()),
                                        precision="highest")
    assert 0 < _PARAM_GRAD_FLOOR < 1 and 0 < _MAX_PARAM_GRAD_REL < 0.1


def test_cosine_is_float64_and_never_reads_above_one():
    g = torch.randn(10_000_000, generator=torch.Generator().manual_seed(0)) * 1e3
    assert abs(_cos(g, g.clone()) - 1.0) < 1e-12
