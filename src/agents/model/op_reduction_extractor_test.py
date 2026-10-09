"""`--op-reduction {max,principled}` (gen3_op_reduction_principled_v1, architecture audit F6b) on the PRODUCTION
architecture built through a REAL SB3 policy (the construction path training uses — the identity_init_test rule),
over REAL observation rows (the compile parity fixture: 64 rows from reproducible battles).

Each test names what a revert breaks:

* the ONE-LEVER init: a `principled` build adds exactly `op_worst_proj` (zero-init, bias-free) and every other
  initial parameter byte equals the `max` build's at the same seed (fails on an RNG-drawing init, a module built
  out of order, a non-zero projection);
* every replaced kernel is WIRED to the mode: the incoming per-mon row's reduced channels, the C1b / C2 / C3 / D4
  edge kernels, the E5 tail seats and the Pursuit cell all read differently under `principled` on the same
  weights, while the quantities that are NOT a reduction over their believed moves (P(we act first), the
  outgoing block, the D1 / D2 kernels over OUR moves) stay bit-identical (fails if a site keeps its max, or if
  the switch leaks into a reduction over our own choices);
* the noisy-OR row is a probability, 0 on a fainted mon, and REACHES the policy: the gradient reaches
  `op_worst_proj`, and once it is nonzero the row moves the output (fails if the route is dropped).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import pytest
import torch

from agents.model.compile_parity_fixture import load_parity_rows
from agents.model.identity_init_test import _build_real_policy
from agents.model.op_reduction import OP_WORST_DIM
from utils.paths import repo_path

_KERNELS = ("pairwise_boost_incoming", "pairwise_status_consequence", "pairwise_recovery",
            "pairwise_bench_incoming", "pairwise_entry", "pairwise_outgoing", "pairwise_bench_outgoing")
#: kernels that reduce over THEIR believed moves (replaced) vs over OUR moves (kept: we choose, a max is our
#: best response)
_REPLACED = ("pairwise_boost_incoming", "pairwise_status_consequence", "pairwise_recovery",
             "pairwise_bench_incoming", "pairwise_entry")
_KEPT = ("pairwise_outgoing", "pairwise_bench_outgoing")


def _production_toggles() -> Dict[str, Any]:
    with open(repo_path("designs", "production_config.json")) as fh:
        cfg = json.load(fh)
    return {k: v for k, v in cfg.items() if not isinstance(v, (dict, list))}


@pytest.fixture(scope="module")
def pair() -> Tuple[Any, Any, torch.Tensor]:
    tog = _production_toggles()
    tog.pop("op_reduction", None)
    mx, enc = _build_real_policy(op_reduction="max", **tog)
    pr, _ = _build_real_policy(op_reduction="principled", **tog)
    obs, _mask = load_parity_rows(enc.dimension)
    return mx.policy, pr.policy, torch.as_tensor(obs[:16])


def _record(fe: Any, x: torch.Tensor, calls: Any = None) -> Tuple[Dict[str, List[torch.Tensor]], Any, Any]:
    """One forward with every listed op kernel's outputs recorded (instance shadows, removed after); ``calls``
    (a dict) also receives each kernel's FIRST call's arguments."""
    op = fe.damage_op
    rec: Dict[str, List[torch.Tensor]] = {k: [] for k in _KERNELS}

    def wrap(name: str, f: Any) -> Any:
        def g(*a: Any, **kw: Any) -> Any:
            if calls is not None and name not in calls:
                calls[name] = (a, kw)
            out = f(*a, **kw)
            rec[name].append(out if isinstance(out, torch.Tensor) else torch.cat([t.flatten() for t in out]))
            return out
        return g

    for k in _KERNELS:
        setattr(op, k, wrap(k, getattr(op, k)))
    try:
        with torch.no_grad():
            pi, vf = fe({"observation": x})
    finally:
        for k in _KERNELS:
            delattr(op, k)
    return rec, (pi, vf), op.last_raw_tensors


def test_max_builds_nothing_new(pair):
    mx, pr, _ = pair
    assert mx.features_extractor.op_worst_proj is None
    assert mx.features_extractor.damage_op.op_principled is False
    assert pr.features_extractor.damage_op.op_principled is True


def test_the_one_lever_init_every_other_parameter_equals_the_max_build(pair):
    mx, pr, _ = pair
    sm, sp = mx.state_dict(), pr.state_dict()
    extra = sorted(set(sp) - set(sm))
    assert extra and all(k.endswith("op_worst_proj.weight") for k in extra), extra
    assert set(sm) <= set(sp)
    moved = [k for k in sm if not torch.equal(sm[k], sp[k])]
    assert not moved, f"a principled build shifted shared initial weights: {moved[:5]}"
    proj = pr.features_extractor.op_worst_proj
    assert proj.bias is None and tuple(proj.weight.shape) == (128, OP_WORST_DIM)
    assert torch.count_nonzero(proj.weight) == 0, "op_worst_proj is zero-init (identity at init)"


def test_every_replaced_site_reads_the_mode_and_every_kept_one_does_not(pair):
    mx, pr, x = pair
    rm, _, tm = _record(mx.features_extractor, x)
    rp, _, tp = _record(pr.features_extractor, x)
    for k in _KERNELS:
        assert rm[k] and len(rm[k]) == len(rp[k]), (k, len(rm[k]), len(rp[k]))
    for k in _REPLACED:
        if k == "pairwise_boost_incoming":
            continue        # zero on every fixture row (no defensive setup move): its own test below
        assert any(not torch.equal(a, b) for a, b in zip(rm[k], rp[k])), f"{k} still reads its max"
    for k in _KEPT:
        assert all(torch.equal(a, b) for a, b in zip(rm[k], rp[k])), f"{k} (over OUR moves) moved"
    # the incoming row: every reduced channel moves; P(we act first) (col 10) is not a reduction — bit-equal
    im, ip = tm.incoming_rows, tp.incoming_rows
    assert torch.equal(im[..., 10], ip[..., 10])
    for c in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11):
        assert not torch.equal(im[..., c], ip[..., c]), f"incoming channel {c} still reads its max"
    assert not torch.equal(tm.cb_high, tp.cb_high) and not torch.equal(tm.cb_pko, tp.cb_pko)
    # the outgoing block (our 4 moves vs their active) is not a reduction over their moves — bit-equal
    assert torch.equal(tm.out_per_move, tp.out_per_move) and torch.equal(tm.status_p_land, tp.status_p_land)


def test_the_e5_tail_seats_read_the_mode(pair):
    mx, pr, x = pair
    seats: List[torch.Tensor] = []
    for pol in (mx, pr):
        proj = pol.features_extractor.entity_seats.tail_proj
        h = proj.register_forward_pre_hook(lambda _m, a: seats.append(a[0].detach().clone()))
        try:
            with torch.no_grad():
                pol.features_extractor({"observation": x})
        finally:
            h.remove()
    assert len(seats) == 2
    # cols 1-2 are worst_phys / worst_spec (the reduction); 0 (tail mass) and 3 (revealed) are not
    assert torch.equal(seats[0][..., 0], seats[1][..., 0]) and torch.equal(seats[0][..., 3], seats[1][..., 3])
    assert not torch.equal(seats[0][..., 1:3], seats[1][..., 1:3])


def test_the_worst_case_row_is_a_probability_gated_and_reaches_the_policy(pair):
    _, pr, x = pair
    fe = pr.features_extractor
    pr.zero_grad()
    pi, vf = fe({"observation": x})
    w = fe.damage_op.last_worst_rows
    assert w is not None and tuple(w.shape) == (x.shape[0], 6, OP_WORST_DIM)
    assert bool(((w >= 0) & (w <= 1)).all()) and bool((w > 0).any())
    alive = fe.damage_op.last_raw_tensors.incoming_rows[..., 10] > 0      # P(first) is gated by alive too
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


def test_c1b_reads_the_mode_on_a_row_with_a_defensive_setup_move(pair):
    """C1b (the INCOMING delta of our def / spd setup) is zero on every fixture row — no fixture active holds a
    defensive boost — so plant Amnesia in our active's first request slot and re-run the kernel on the
    forward's own context (each build's own op stash, X5 roster included)."""
    import dataclasses

    from agents import gen3_data
    mx, pr, x = pair
    amnesia = int(gen3_data.moves.get("amnesia").num)
    outs = []
    for pol in (mx, pr):
        calls: Dict[str, Any] = {}
        _record(pol.features_extractor, x, calls)
        (ctx, *rest), kw = calls["pairwise_boost_incoming"]
        ids = ctx.our_active_req_move_ids.clone()
        ids[:, 0] = amnesia
        with torch.no_grad():
            outs.append(pol.features_extractor.damage_op.pairwise_boost_incoming(
                dataclasses.replace(ctx, our_active_req_move_ids=ids), *rest, **kw))
    assert bool((outs[0][:, 0] != 0).any()), "PRECONDITION: Amnesia's incoming delta is live"
    assert not torch.equal(outs[0], outs[1]), "C1b still reads its max"
