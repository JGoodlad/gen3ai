"""The op's PRINCIPLED reductions (gen3_op_reduction_principled_v1, architecture audit F6b) on the PRODUCTION
architecture built through a REAL SB3 policy (the construction path training uses — the identity_init_test rule),
over REAL observation rows (the compile parity fixture: 64 rows from reproducible battles). They are the op's ONLY
reductions since the legacy deletion's U1 deleted `--op-reduction` and its per-channel hard maxima
(`designs/ops/legacy_deletion_manifest_2026-10-12.md` D4); `op_reduction_test.py` pins the arithmetic.

Each test names what a revert breaks:

* `op_worst_proj` is built with the op, zero-init and bias-free (fails on an RNG-drawing or non-zero init);
* the noisy-OR row is a probability, 0 on a fainted mon, and REACHES the policy: the gradient reaches
  `op_worst_proj`, and once it is nonzero the row moves the output (fails if the route is dropped).
"""
from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import pytest
import torch

from agents.model.compile_parity_fixture import load_parity_rows
from agents.model.identity_init_test import _build_real_policy
from agents.model.op_reduction import OP_WORST_DIM
from utils.paths import repo_path


def _production_toggles() -> Dict[str, Any]:
    from main.train.arch_arms import legacy_production_config
    with open(repo_path("designs", "production_config.json")) as fh:
        cfg = legacy_production_config(json.load(fh))  # the LEGACY base: this test's subject is the pre-adoption network (end state adopted 2026-10-12)
    return {k: v for k, v in cfg.items() if not isinstance(v, (dict, list))}


@pytest.fixture(scope="module")
def pair() -> Tuple[Any, torch.Tensor]:
    pr, enc = _build_real_policy(**_production_toggles())
    obs, _mask = load_parity_rows(enc.dimension)
    return pr.policy, torch.as_tensor(obs[:16])


def test_op_worst_proj_is_built_with_the_op_zero_init_and_bias_free(pair):
    pr, _ = pair
    proj = pr.features_extractor.op_worst_proj
    assert proj is not None and not hasattr(pr.features_extractor.damage_op, "op_principled")
    assert proj.bias is None and tuple(proj.weight.shape) == (128, OP_WORST_DIM)
    assert torch.count_nonzero(proj.weight) == 0, "op_worst_proj is zero-init (identity at init)"


def test_the_worst_case_row_is_a_probability_gated_and_reaches_the_policy(pair):
    pr, x = pair
    fe = pr.features_extractor
    pr.zero_grad()
    pi, vf = fe({"observation": x})
    w = fe.damage_op.last_worst_rows
    assert w is not None and tuple(w.shape) == (x.shape[0], 6, OP_WORST_DIM)
    assert bool(((w >= 0) & (w <= 1)).all()) and bool((w > 0).any())
    # the op's own gate: our defender alive x an opponent active present (P(first) is NOT a proxy for it since the
    # speed rule became the only one: an alive mon can be certain to move second, P(first) == 0 exactly)
    with torch.no_grad():
        ctx = fe.unpack({"observation": x})
    has_opp = ctx.hp_and_active[:, 6:12, -1].any(dim=1)
    alive = (ctx.hp_and_active[:, :6, 0] > 0) & has_opp[:, None]
    assert bool((~alive).any()), "PRECONDITION: some gated (fainted / no-opponent) row"
    assert torch.equal(w[~alive], torch.zeros_like(w[~alive]))
    (pi.square().sum() + vf.square().sum()).backward()
    g = fe.op_worst_proj.weight.grad
    assert g is not None and torch.count_nonzero(g) > 0, "the gradient reaches op_worst_proj"
    pr.zero_grad()
    with torch.no_grad():
        base = fe({"observation": x})[0].clone()
        fe.op_worst_proj.weight.copy_(torch.randn(fe.op_worst_proj.weight.shape,
                                                  generator=torch.Generator().manual_seed(3)) * 0.1)
        try:
            moved = fe({"observation": x})[0]
        finally:
            fe.op_worst_proj.weight.zero_()
    assert not torch.equal(base, moved), "the noisy-OR row reaches the policy once op_worst_proj is nonzero"
