"""THE X5 VERSION BREAK, parts 4 and 5 (config v144, `gen3_x5_version_break_v1`; `designs/endstate/design_arch_audit.md`
§9.4) — the damage operator's two deliberate BEHAVIOUR changes.

Each test FAILS ON REVERT of its piece:

* Part 4, the slot-tied ``out_gain`` — the gain holds ONE scalar per distinct (block region, channel), counted
  independently of the code's own key map (production 138 → 99; the render arm too); permuting the REQUEST SLOTS of
  a real raw block permutes the gained block identically (planted non-uniform gains, a real production op); after a
  backward every request slot's copy of a channel accumulates into ONE parameter; the compile gates' per-parameter
  rule still reads (and names) ``damage_op.out_gain`` at its tied size.
* Part 5, the pre-gain read — with planted gains ≠ 1 on the outspeed, high-roll and flinch channels,
  ``intent_conditional``'s inputs and its raw cell are unchanged (at the revert they move with the gain); in BOTH
  ``--speed-physics`` modes its P(first) IS the op's live pre-gain view (``last_raw_tensors``), never the post-gain
  one, and so is the move-resolution family's.
"""
from __future__ import annotations

import inspect
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

from agents.model.damage_op_layout import (_DMG_CB, _DMG_IMX_CELL, _DMG_IMX_HEADER, _DMG_OMX_CELL, _DMG_OUT_PER_MOVE,
                                           _DMG_PER_MON, _N_OUT_SECONDARY)
from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.observation.constants import TEAM_SIZE
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from utils.paths import repo_path, src_path


def _prod_kwargs(**over: Any) -> Dict[str, Any]:
    from agents.model.damage_tables import sanitize_historical_move_floor
    cfg = json.load(open(repo_path("designs", "production_config.json")))
    sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    kw = {k: v for k, v in cfg.items() if k in sig}
    sanitize_historical_move_floor(kw)
    kw.update(over)
    return kw


def _build(**over: Any) -> Gen3FeaturesExtractor:
    import gymnasium as gym
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    torch.manual_seed(0)
    return Gen3FeaturesExtractor(space, layout=layout, mappings=mappings, **_prod_kwargs(**over)).eval()


@pytest.fixture(scope="module")
def obs() -> torch.Tensor:
    return torch.as_tensor(np.load(src_path("agents", "model", "compile_parity_obs.npz"))["obs"])


@pytest.fixture(scope="module")
def fe() -> Gen3FeaturesExtractor:
    return _build()


def _positions(op: Any) -> Any:
    """The op's own typed views over a block whose value IS its flat position."""
    return op.tensors_from_block(torch.arange(op.out_dim, dtype=torch.float32)[None])


def _slot_groups(op: Any) -> List[torch.Tensor]:
    """Every request-slot-indexed region of the production block as ``[4, F]`` flat positions (slot-major)."""
    t = _positions(op)
    return [t.out_per_move[0].long(), t.out_secondary[0].long(),
            t.status_p_land[0].long()[:, None], t.status_known[0].long()[:, None]]


def _gain_index(op: Any, pos: int) -> int:
    """Which ``out_gain`` entry scales flat position ``pos`` — the tie's column, or (at the revert) ``pos`` itself."""
    tie = getattr(op, "_out_gain_tie", None)
    return int(tie[:, pos].argmax()) if tie is not None else pos


def _plant_random_gains(op: Any, seed: int = 3) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        op.out_gain.copy_(0.5 + 1.5 * torch.rand(op.out_gain.shape, generator=g))


def _raw_block(fe: Gen3FeaturesExtractor, obs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        fe({"observation": obs})
    raw = fe.damage_op.last_raw_block
    assert raw is not None and bool((raw != 0).any())
    return raw.clone()


# ------------------------------------------------------------------------------------------------ PART 4
def _expected_distinct(*, renders_k: int) -> int:
    """Counted from the LAYOUT CONSTANTS, not from the code's key map: the untied mon-axis regions in full, every
    request-slot / move-seat region once."""
    n = TEAM_SIZE * _DMG_PER_MON + _DMG_CB                     # incoming rows (per our team slot) + the CB tail
    n += _DMG_OUT_PER_MOVE + 1 + _N_OUT_SECONDARY + 2          # out_move, p_outspeed, secondaries, p_land / known
    if renders_k:
        n += TEAM_SIZE * _DMG_OMX_CELL + TEAM_SIZE             # outgoing matrix: per their mon, + revealed bits
        n += _DMG_IMX_HEADER + TEAM_SIZE * _DMG_IMX_CELL       # incoming matrix: one header, per our mon cells
    return n


def test_the_production_gain_holds_one_scalar_per_distinct_region_channel(fe: Gen3FeaturesExtractor) -> None:
    op = fe.damage_op
    assert op.drop_renders and op.outgoing, "PRECONDITION: the production op (lean forward, outgoing on)"
    assert op.out_dim == 138
    assert op.out_gain.numel() == _expected_distinct(renders_k=0) == 99
    assert len(op.out_gain_keys) == op.out_gain.numel() == len(set(op.out_gain_keys))


def test_the_render_arm_ties_the_matrices_across_moves_and_seats() -> None:
    from agents.model.damage_op import DamageOperator
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    op = DamageOperator(layout, outgoing=True, topk_k=6, matrices_outgoing=True, matrices_incoming=True)
    assert op.matrices_incoming_k == 6
    assert op.out_gain.numel() == _expected_distinct(renders_k=6) == 222
    assert op.out_dim == 138 + 126 + 6 * _DMG_IMX_HEADER + TEAM_SIZE * 6 * _DMG_IMX_CELL


def test_the_tied_init_is_the_per_slot_init(fe: Gen3FeaturesExtractor) -> None:
    """The expanded gain at init is the per-position init it replaced: the roll channels ÷1.5 / ÷3, the rest 1."""
    full = fe.damage_op.expanded_out_gain().detach()
    for grp in _slot_groups(fe.damage_op):
        assert bool((full[grp] == full[grp[:1]]).all())
    t = _positions(fe.damage_op)
    mv = full[t.out_per_move[0].long()]
    assert torch.equal(mv[0], torch.tensor([1 / 1.5, 1 / 1.5, 1 / 3.0, 1.0]))
    assert float(full[int(t.out_p_outspeed)]) == 1.0


def test_permuting_request_slots_of_a_real_raw_block_permutes_the_gained_block(
        fe: Gen3FeaturesExtractor, obs: torch.Tensor) -> None:
    op = fe.damage_op
    raw = _raw_block(fe, obs)
    _plant_random_gains(op)
    try:
        for sigma in ([3, 2, 1, 0], [1, 2, 3, 0], [0, 2, 1, 3]):
            perm = torch.arange(op.out_dim)
            for grp in _slot_groups(op):
                perm[grp.reshape(-1)] = grp[sigma].reshape(-1)
            with torch.no_grad():
                assert torch.equal(op.apply_out_gain(raw[:, perm]), op.apply_out_gain(raw)[:, perm]), sigma
    finally:
        fe_fresh = _build()
        with torch.no_grad():
            op.out_gain.copy_(fe_fresh.damage_op.out_gain)


def test_every_request_slot_copy_of_a_channel_accumulates_into_one_parameter(
        fe: Gen3FeaturesExtractor, obs: torch.Tensor) -> None:
    op = fe.damage_op
    raw = _raw_block(fe, obs)
    w = torch.randn(raw.shape, generator=torch.Generator().manual_seed(5))
    op.out_gain.grad = None
    (op.apply_out_gain(raw) * w).sum().backward()
    grad = op.out_gain.grad.detach().clone()
    op.out_gain.grad = None
    contrib = (w * raw).sum(0)                                                  # dL/dgain per FLAT position
    for grp in _slot_groups(op):
        for f in range(grp.shape[1]):
            cols = {_gain_index(op, int(p)) for p in grp[:, f]}
            assert len(cols) == 1, f"the 4 request-slot copies of one channel map to {len(cols)} parameters"
            j = cols.pop()
            torch.testing.assert_close(grad[j], contrib[grp[:, f]].sum(), rtol=1e-5, atol=1e-6)
    # and the full gradient is exactly the per-(region, channel) sum of the flat contributions
    torch.testing.assert_close(grad, op._out_gain_tie @ contrib, rtol=1e-5, atol=1e-6)


def test_the_per_parameter_compile_rule_reads_and_names_the_tied_gain(
        fe: Gen3FeaturesExtractor, obs: torch.Tensor) -> None:
    """`compile_gate_probe.grad_parameters` / `compile_trainer._param_verdict` (the rule whose motivating defect read
    0.10 on `damage_op.out_gain`): the tied parameter is in the gradient set at its TIED size, and a compiled arm that
    drops its gradient (the pointer move cells' path) FAILS the rule naming it."""
    from agents.model.compile_gate_probe import grad_parameters
    from agents.model.compile_trainer import CompileTrainerError, _param_verdict
    params = grad_parameters(object(), fe)
    names = [n for n, _ in params]
    assert "damage_op.out_gain" in names
    sizes = [int(p.numel()) for _, p in params]
    assert sizes[names.index("damage_op.out_gain")] == 99
    fe.train()
    try:
        for p in fe.parameters():
            p.grad = None
        pi, vf = fe({"observation": obs[:16]})
        (pi.square().mean() + vf.square().mean() + fe.last_damage_block.square().mean()).backward()
        eager = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for _, p in params])
    finally:
        fe.eval()
        for p in fe.parameters():
            p.grad = None
    i = names.index("damage_op.out_gain")
    off = sum(sizes[:i])
    assert float(eager[off:off + sizes[i]].abs().max()) > 0, "PRECONDITION: the gain carries a gradient"
    compiled = eager.clone()
    compiled[off:off + sizes[i]] = 0.0
    arm = {"grad_sizes": torch.tensor(sizes)}
    with pytest.raises(CompileTrainerError, match="damage_op.out_gain"):
        _param_verdict({**arm, "grad": compiled}, {**arm, "grad": eager}, names, bar=0.01)
    assert _param_verdict({**arm, "grad": eager}, {**arm, "grad": eager.clone()}, names, bar=0.01)


# ------------------------------------------------------------------------------------------------ PART 5
def _capture_intent_conditional(fe: Gen3FeaturesExtractor, x: torch.Tensor) -> Tuple[Tuple[Any, ...], torch.Tensor]:
    ic = fe.intent_conditional
    cap: Dict[str, Any] = {}
    orig = ic.forward

    def hook(*a: Any, **k: Any) -> Any:
        cap["args"] = tuple(t.detach().clone() if isinstance(t, torch.Tensor) else t for t in a)
        return orig(*a, **k)
    h = ic.proj.register_forward_pre_hook(lambda m, inp: cap.__setitem__("raw", inp[0].detach().clone()))
    ic.forward = hook
    try:
        with torch.no_grad():
            fe({"observation": x})
    finally:
        ic.forward = orig
        h.remove()
    return cap["args"], cap["raw"]


def test_intent_conditional_reads_the_pre_gain_values(obs: torch.Tensor) -> None:
    from agents.model.extractor_forward import _OUT_SEC_FLINCH_COL
    fe = _build()
    op = fe.damage_op
    assert fe.intent_conditional is not None and not op.speed_physics, "PRECONDITION: production (intent_conditional on)"
    args0, raw0 = _capture_intent_conditional(fe, obs)
    gained0 = fe.last_damage_block.detach().clone()
    t = _positions(op)
    plant = {int(t.out_p_outspeed): 3.0}
    plant.update({int(p): 2.5 for p in t.out_per_move[0, :, 1]})                # our moves' high roll
    plant.update({int(p): 4.0 for p in t.out_secondary[0, :, _OUT_SEC_FLINCH_COL]})
    with torch.no_grad():
        for pos, f in plant.items():
            op.out_gain[_gain_index(op, pos)] = f
    args1, raw1 = _capture_intent_conditional(fe, obs)
    gained1 = fe.last_damage_block.detach()
    # teeth: the plant moved the post-gain block on every planted channel that carries a value
    sp = int(t.out_p_outspeed)
    assert bool((gained1[:, sp] != gained0[:, sp]).any()), "the plant did not reach the gained block"
    assert bool((args0[6] != 0).any()) and bool((args0[5] != 0).any()) and bool((args0[7] != 0).any())
    for i, name in ((5, "out_high"), (6, "p_outspeed"), (7, "sec_flinch")):
        assert torch.equal(args0[i], args1[i]), f"intent_conditional's {name} moved with the learned gain"
    assert torch.equal(raw0, raw1), "intent_conditional's raw cell moved with the learned gain"


@pytest.mark.parametrize("mode", ["off", "on"])
def test_both_speed_modes_read_p_first_from_the_live_pre_gain_view(obs: torch.Tensor, mode: str) -> None:
    """ONE read path: the tensor `intent_conditional` receives as P(first) IS the op's `last_raw_tensors` view (the
    same storage — never the post-gain `last_tensors` one, never a detached copy), in BOTH speed modes."""
    fe = _build(speed_physics=mode).train()
    op = fe.damage_op
    ic = fe.intent_conditional
    cap: Dict[str, Any] = {}
    orig = ic.forward

    def hook(*a: Any, **k: Any) -> Any:
        cap["args"] = a
        return orig(*a, **k)
    ic.forward = hook
    try:
        fe({"observation": obs[:16]})
    finally:
        ic.forward = orig
    p_first, out_high, flinch = cap["args"][6], cap["args"][5], cap["args"][7]
    rt, pt = op.last_raw_tensors, op.last_tensors
    assert p_first.data_ptr() == rt.out_p_outspeed.data_ptr() != pt.out_p_outspeed.data_ptr()
    assert out_high.untyped_storage().data_ptr() == rt.flat.untyped_storage().data_ptr()
    assert flinch.untyped_storage().data_ptr() == rt.flat.untyped_storage().data_ptr()
    assert p_first.requires_grad, "the pre-gain read is LIVE (the op's upstream gradient route is kept)"


def test_the_move_resolution_family_reads_the_same_live_pre_gain_view(obs: torch.Tensor) -> None:
    fe = _build(move_resolution="on").train()
    cell = fe.move_resolution_cell
    cap: Dict[str, Any] = {}
    orig = cell.forward

    def hook(ops: Any) -> Any:
        cap["ops"] = ops
        return orig(ops)
    cell.forward = hook
    try:
        fe({"observation": obs[:16]})
    finally:
        cell.forward = orig
    p_out = cap["ops"].p_out
    rt = fe.damage_op.last_raw_tensors
    assert p_out.data_ptr() == rt.out_p_outspeed.data_ptr()
    assert p_out.requires_grad
