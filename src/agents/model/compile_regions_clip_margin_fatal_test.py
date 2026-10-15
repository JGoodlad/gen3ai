"""THE T36 FIX AT THE FATAL'S OWN WEIGHTS (`gen3_r1_clip_margin_v1`): `rb_toy_ent_s4001`'s
`final_model_canary_fatal.zip` (read-only, from the run archive), CPU, B = 64 untiled (the fixture's 64 rows once each,
so every clip decision is exact; the canary's B = 2,048 tiles the same rows and reads the same mean gradient).

* THE NULL (the diagnostic's read, through PRODUCTION's own relocation): 32 seeded weight dithers (sigma 2^-9
  relative, `main.ops.diag_bf16_canary.dithered`), each read against its OWN fp32 reference. On the golden's raw rows
  two HEALTHY eager bf16 arms read the canary's per-parameter ratio above K = 12 on a pointer-head parameter in ~21 %
  of the 992 ordered pairs (max ~42x; the diagnostic's B = 2,048 read 18 %, max 31x) — the clip-boundary lottery. On
  the rows `compile_regions.relocate_clip_boundary` produces: 0 of 992 (max ~3.7x), and every dither passes the margin
  rule. Fails on revert (no relocation: the relocated reads ARE the raw ones).
* The FATAL's two boundary rows (4: A < 0, 0.0045 from its bound; 35: A > 0, 0.0070) are among the relocated rows.
* TEETH: a planted 10 % backward miscompile on `pointer_head.move_proj.weight` is refused at the fp32 rule on these
  weights' relocated rows.

Slow (~2-4 min on three cores); skipped where the run archive does not hold the checkpoint (a worktree without
`models/`, CI)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pytest
import torch

from agents.model import compile_control as cc
from agents.model import compile_regions as cr
from agents.model import compile_trainer as ct
from agents.model import trunk_precision as TP
from agents.model.compile_gate_probe import per_param_grad_errors

pytestmark = pytest.mark.slow

RUN = "rb_toy_ent_s4001"
ZIP = "final_model_canary_fatal.zip"
ROWS = 64
SEEDS = 32
HEAD = "pointer_head."


def _zip() -> Path:
    from utils.paths import main_models_dir
    m = main_models_dir()
    p = (m / RUN / ZIP) if m is not None else None
    if p is None or not p.is_file():
        pytest.skip(f"the run archive does not hold {RUN}/{ZIP}")
    return p


@pytest.fixture
def one_thread():
    was = torch.get_num_threads()
    torch.set_num_threads(1)                    # deterministic CPU reductions, and the CPU-play rule
    try:
        yield
    finally:
        torch.set_num_threads(was)


def _learner(precision: str):
    from main.ops import diag_bf16_canary as D
    m = D.build_learner(_zip(), "cpu", precision=precision, batch=ROWS)
    assert TP.trunk_precision_of(m.policy) == precision
    assert ROWS == D.n_fixture_rows(), "PRECONDITION: B = the fixture's row count (each row exactly once)"
    return m


def _head_errors(arm: Dict[str, torch.Tensor], ref: Dict[str, torch.Tensor], head: List[int]) -> Dict[int, float]:
    sizes = [int(x) for x in ref["grad_sizes"].tolist()]
    err = dict(per_param_grad_errors(arm["grad"], ref["grad"], sizes, floor_frac=ct._PARAM_GRAD_FLOOR))
    return {i: err[i] for i in head if i in err}


def _pairs(per_seed: List[Dict[int, float]], floor: float, k: float):
    """(pairs over K, pairs, the largest ratio): the canary's per-parameter statistic between two HEALTHY eager bf16
    arms i != j — d_i / max(d_j, floor) — over the pointer-head parameters both judge."""
    over, n, top = 0, 0, 0.0
    for i, a in enumerate(per_seed):
        for j, b in enumerate(per_seed):
            if i == j:
                continue
            r = max(a[p] / max(b[p], floor) for p in a if p in b)
            n, top, over = n + 1, max(top, r), over + int(r > k)
    return over, n, top


def test_the_dithered_null_at_the_FATALs_weights_drops_from_about_a_fifth_to_ZERO(one_thread):
    from agents.training.instrumented_ppo.micro_step import micro_step
    from main.ops.diag_bf16_canary import dithered
    model = _learner("bf16")
    names = [n for n, _ in ct.grad_parameters(model, model.policy.features_extractor)]
    head = [i for i, n in enumerate(names) if n.startswith(HEAD)]
    assert cr.weights_regime(model) == "trained" and not set(cr.unmoved_parameters(model)) & {names[i] for i in head}
    floor = cr.R1_PARAM_BAR["trained"]
    assert cr.BF16_ENVELOPE_K is not None
    k = cr.BF16_ENVELOPE_K["param"]
    raw: List[Dict[int, float]] = []
    rel: List[Dict[int, float]] = []
    flipped = 0
    for s in range(SEEDS):
        with dithered([model], seed=1 + s):
            args = cr._r1_args(model, cr.r1_batch(model, ROWS, 0))
            e32, ebf = cr._r1_reference(model, args), cr._r1_arm(model, micro_step, args)
            rargs, fix = cr.relocate_clip_boundary(model, args)
            e32r, ebfr = cr._r1_reference(model, rargs), cr._r1_arm(model, micro_step, rargs)
            cr._clip_margin_rule(fix, {"eager bf16": ebfr, "fp32 reference": e32r})     # raises if broken
        act = cr.clip_active(e32["log_prob"], args[4], fix.advantages, fix.clip_range)
        flipped += int(bool((cr.clip_active(ebf["log_prob"], args[4], fix.advantages, fix.clip_range) != act).any()))
        raw.append(_head_errors(ebf, e32, head))
        rel.append(_head_errors(ebfr, e32r, head))
    over_raw, n, top_raw = _pairs(raw, floor, k)
    over_rel, n_rel, top_rel = _pairs(rel, floor, k)
    assert n == n_rel == SEEDS * (SEEDS - 1)
    # PRECONDITION (the defect, at these weights): the raw rows' null reads above K often, carried by clip flips
    assert flipped >= 4 and over_raw >= 0.05 * n, (flipped, over_raw, top_raw)
    # THE FIX: not one healthy pair above K, and the largest ratio far below it (read 3.65 at B = 64)
    assert over_rel == 0 and top_rel <= 0.5 * k, (over_rel, top_rel, over_raw, top_raw)


def test_the_FATALs_two_boundary_rows_are_RELOCATED(one_thread):
    model = _learner("bf16")
    args = cr._r1_args(model, cr.r1_batch(model, ROWS, 0))
    _, fix = cr.relocate_clip_boundary(model, args)
    moved = torch.nonzero(fix.moved).flatten().tolist()
    assert 4 in moved and 35 in moved, moved
    assert fix.nearest_unmoved > cr.R1_CLIP_MARGIN


def test_a_planted_10pct_POINTER_HEAD_miscompile_is_refused_at_the_fp32_rule_on_these_weights(one_thread, monkeypatch):
    from agents.model import compile_regions_resolution_test as RT
    model = _learner("fp32")
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    try:
        cc.control().install()
        cr.install(model, backend="eager")
        rules = cr.gate_regions(model, batch_size=ROWS, say=lambda _m: None)          # honest: passes
        assert any("clip margin:" in r for r in rules), rules
        with monkeypatch.context() as mp:
            RT.plant(mp, "pointer_head.move_proj.weight", 0.10)
            with pytest.raises(ct.CompileTrainerError,
                               match=r"DISAGREES with eager.*pointer_head\.move_proj\.weight"):
                cr.gate_regions(model, batch_size=ROWS, say=lambda _m: None)
    finally:
        cr.uninstall(model)
        cc._reset_control_for_tests()
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.reset()
