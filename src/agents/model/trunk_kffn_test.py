"""B-KFFN (`agents.model.trunk_kffn`, `gen3_trunk_ffn_by_kind_v1`, config v161): per-token-kind FFN experts in every trunk
round, and the fork's ``--grow trunk_ffn_by_kind=on`` (`design_growth_builds_kffn_int.md` §1.8, tests K-1 … K-12; the
launch side and the pool, K-10 / K-12 / K-13, are in `main/train/grow_fork_test.py`; K-14 in
`main/probe_battery/probe_battery_test.py`).

Every network is a PRODUCTION build through SB3 (`main.fresh_checkpoint.build_fresh_model`, the production construction
path), on the CPU, one torch thread, fp32 eager unless a test says otherwise; the rows are the 64 REAL rows of the
compile-parity fixture, always as ONE batch (F-GB-5: the split FFN equals the shared one bit for bit at batch >= 8 on
this box, not at batch 1-2). Each test fails on the revert its docstring names.
"""
from __future__ import annotations

import copy
import dataclasses
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

from agents.model import growth as G
from agents.model import trunk_kffn as K
from utils.torch_state_guard import torch_globals

FIELDS = {"trunk_ffn_by_kind": {"parent": "off", "child": "on"}}
#: 4 new kinds x (128x256 + 256 + 256x128 + 128) per round x production's 3 rounds (the spec's §1.7, F-GB-8).
EXPERT_ELEMENTS = 4 * 65_920 * 3
#: production fresh policy: 2,609,183 parameters (MEASURED 2026-10-19 at b79b66bb and here); kffn adds the experts.
PRODUCTION_PARAMS = 2_609_183


def _args(**over: Any) -> Any:
    from main.train.production_args import production_args

    a = copy.deepcopy(production_args())
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _build(seed: int = 0, **over: Any) -> Any:
    from main.fresh_checkpoint import build_fresh_model

    model, _a, _pk = build_fresh_model(seed, args=_args(**over))
    return model


@pytest.fixture(scope="module")
def rows() -> Tuple[np.ndarray, np.ndarray]:
    from agents.model.growth_test import _rows
    return _rows()


def _parent(obs: np.ndarray, mask: np.ndarray, **over: Any) -> Any:
    """A production-shaped parent after ONE real AdamW step (`growth_test._step`): non-trivial moments everywhere."""
    from agents.model.growth_test import _step

    model = _build(**over)
    _step(model, obs, mask)
    return model


def _record(model: Any, obs: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
    model.policy.set_training_mode(False)
    return G.forward_record(model.policy, obs, mask)


def _same(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> bool:
    return bool(np.array_equal(a["logits"], b["logits"]) and np.array_equal(a["values"], b["values"]))


def _rounds(model: Any) -> List[Any]:
    return K.trunk_rounds(model.policy.features_extractor.team_transformer)


# ---------------------------------------------------------------------------------------------------- K-1, K-5
def test_k1_off_builds_nothing_and_moves_no_byte_or_rng_draw() -> None:
    """K-1. ``off`` = the build WITHOUT the field (kwarg absent): the same keys, every initial byte, and the same global
    RNG state after the build. Fails if an expert is built, or the stream moves, when off."""
    import main.fresh_checkpoint as FC

    real = FC._production_policy_kwargs

    def _without(args: Any = None) -> Any:              # the same surface, the kwarg ABSENT (a pre-v161 zip's kwargs)
        a_, layout, pk_ = real(args)
        pk_ = {**pk_, "features_extractor_kwargs": {k: v for k, v in pk_["features_extractor_kwargs"].items()
                                                    if k != "trunk_ffn_by_kind"}}
        return a_, layout, pk_

    with torch_globals(num_threads=1):
        a = _args()
        assert a.trunk_ffn_by_kind == "off"
        m_off, _a, pk = FC.build_fresh_model(0, args=a)
        rng_off = torch.get_rng_state()
        FC._production_policy_kwargs = _without
        try:
            m_abs, _b, pk_abs = FC.build_fresh_model(0, args=copy.deepcopy(a))
        finally:
            FC._production_policy_kwargs = real
        rng_abs = torch.get_rng_state()
    assert pk["features_extractor_kwargs"]["trunk_ffn_by_kind"] == "off"
    assert "trunk_ffn_by_kind" not in pk_abs["features_extractor_kwargs"]
    sd, sd_abs = m_off.policy.state_dict(), m_abs.policy.state_dict()
    assert list(sd) == list(sd_abs) and not any(".kind_ffn." in k for k in sd)
    assert all(torch.equal(sd[k], sd_abs[k]) for k in sd)
    assert torch.equal(rng_off, rng_abs)
    assert sum(p.numel() for p in m_off.policy.parameters()) == PRODUCTION_PARAMS
    assert all(r.kind_ffn is None and r.ffn_segments is None for r in _rounds(m_off))


def test_k5_a_fresh_on_build_moves_no_other_initial_byte() -> None:
    """K-5. Every non-expert parameter of a fresh ``on`` build equals the ``off`` build's, byte for byte, and the
    experts are `IsolatedLinear`s. Fails if the experts are plain Linears (SB3's orthogonal re-init draws them from the
    global stream) or are built outside ``fork_rng``."""
    from agents.model.hypothesis_set import IsolatedLinear

    with torch_globals(num_threads=1):
        m_off, m_on = _build(0), _build(0, trunk_ffn_by_kind="on")
    sd_off, sd_on = m_off.policy.state_dict(), m_on.policy.state_dict()
    assert [k for k in sd_on if k in sd_off] == list(sd_off)            # the old keys, in their order
    assert all(torch.equal(sd_off[k], sd_on[k]) for k in sd_off)
    new = [k for k in sd_on if k not in sd_off]
    assert new and all(".kind_ffn." in k for k in new)
    assert sum(p.numel() for p in m_on.policy.parameters()) == PRODUCTION_PARAMS + EXPERT_ELEMENTS
    for r in _rounds(m_on):
        assert list(r.kind_ffn) == list(K.EXPERT_KINDS)
        for e in r.kind_ffn.values():
            assert isinstance(e.linear1, IsolatedLinear) and isinstance(e.linear2, IsolatedLinear)
        # the BiasedEncoderLayers' experts are SB3's init (orthogonal, nonzero); the identity round's linear2 is zero
        zero_out = not isinstance(r, type(_rounds(m_on)[0]))
        for e in r.kind_ffn.values():
            assert bool((e.linear2.weight != 0).any()) is not zero_out


# ---------------------------------------------------------------------------------------------------- K-2
def test_k2_the_segment_table_is_the_seat_roles_of_a_real_production_forward(rows) -> None:
    """K-2. The table read off the BUILT seat modules against the token kinds the REAL forward hands the trunk: the
    team / board token blocks, and the type of every extra seat as `EntityMoveSeats.seat_types`, the OTHER_species
    splice and the event block write it (OUR_MOVE / THEIR_THREAT — the E5 tail included — / THEIR_TEAM / HISTORY).
    Position 31 (OTHER_species, after the entity seats) is ``mon``; the E5 tail (19 + K - 4 … 30) is ``threat``."""
    from agents.model.extractor_ctx import (TOKEN_TYPE_HISTORY, TOKEN_TYPE_OUR_MOVE, TOKEN_TYPE_THEIR_TEAM,
                                            TOKEN_TYPE_THEIR_THREAT)
    from agents.observation.constants import TEAM_SIZE

    obs, mask = rows
    with torch_globals(num_threads=1):
        model = _build(0, trunk_ffn_by_kind="on")
        fe = model.policy.features_extractor
        tt = fe.team_transformer
        seen: Dict[str, Any] = {}

        def pre(_m: Any, args: Any, kwargs: Any) -> None:
            seen["extra"] = kwargs["extra"]

        h = tt.register_forward_pre_hook(pre, with_kwargs=True)
        try:
            _record(model, obs, mask)
        finally:
            h.remove()
    _tok, types, _pad = seen["extra"]
    by_type = {TOKEN_TYPE_OUR_MOVE: K.KIND_MOVE, TOKEN_TYPE_THEIR_THREAT: K.KIND_THREAT,
               TOKEN_TYPE_THEIR_TEAM: K.KIND_MON, TOKEN_TYPE_HISTORY: K.KIND_EVENT}
    want = ([K.KIND_MON] * (2 * TEAM_SIZE) + [K.KIND_BOARD] * tt.n_board_tokens
            + [by_type[int(t)] for t in types.tolist()])
    table = tt.transformer_layers[0].ffn_segments
    got = [k for s, e, k in table for _ in range(s, e)]
    assert got == want and table[-1][1] == len(want) == 64
    assert table == ((0, 12, K.KIND_MON), (12, 15, K.KIND_BOARD), (15, 19, K.KIND_MOVE), (19, 31, K.KIND_THREAT),
                     (31, 32, K.KIND_MON), (32, 64, K.KIND_EVENT))
    # the E5 tail (the last 6 entity seats) is a THREAT kind; position 31 is OTHER_species
    es = fe.entity_seats
    tail = range(tt._total_tokens + 4 + es.topk_seats, tt._total_tokens + es.n_seats)
    assert len(tail) == 6 and all(got[i] == K.KIND_THREAT for i in tail)
    assert got[tt._total_tokens + es.n_seats] == K.KIND_MON
    assert all(r.ffn_segments is table for r in _rounds(model))


def test_k2_the_table_follows_the_seat_counts_never_a_literal() -> None:
    """K-2's other half: the same function on a network with other seat counts moves every boundary; a missing kind is
    a typed refusal naming it."""
    import types as T

    def fe(k: int, tail: bool, n_ev: int, static: bool = True, other: bool = True) -> Any:
        tt = T.SimpleNamespace(_their_token_slice=slice(6, 12), _total_tokens=15 if static else 13,
                               static_board=static)
        es = T.SimpleNamespace(topk_seats=k, tail_seats=tail, n_seats=4 + k + (6 if tail else 0))
        return T.SimpleNamespace(team_transformer=tt, entity_seats=es, history_events=T.SimpleNamespace(n=n_ev),
                                 hypothesis_builder=object() if other else None)

    assert K.ffn_kind_segments(fe(5, True, 16)) == (
        (0, 12, 0), (12, 15, 1), (15, 19, 2), (19, 30, 3), (30, 31, 0), (31, 47, 4))
    for bad, why in ((fe(6, True, 32, static=False), "board"), (fe(6, False, 32), "threat"),
                     (fe(0, False, 32), "threat"), (fe(6, True, 32, other=False), "OTHER_species")):
        with pytest.raises(ValueError, match=why):
            K.ffn_kind_segments(bad)
    nf = fe(6, True, 32)
    nf.history_events = None
    with pytest.raises(ValueError, match="event"):
        K.ffn_kind_segments(nf)


# ---------------------------------------------------------------------------------------------------- K-3, K-4
def _scramble_trunk(model: Any, seed: int) -> None:
    """Non-trivial trunk weights (every round Linear, the identity round's zero outputs included), seeded."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for r in _rounds(model):
            for lin in (r.in_proj, r.out_proj, r.linear1, r.linear2):
                lin.weight.add_(0.05 * torch.randn(lin.weight.shape, generator=g))
                lin.bias.add_(0.05 * torch.randn(lin.bias.shape, generator=g))


def test_k3_tied_experts_are_the_shared_ffn_bit_for_bit(rows) -> None:
    """K-3. An ``on`` network whose every expert is a copy of its round's shared FFN gives the SAME logits and values as
    the ``off`` network with the same weights, bit for bit, on the 64 rows as one batch (every round non-trivial, the
    identity round's outputs included). Fails if a segment drops, duplicates or reorders tokens."""
    obs, mask = rows
    with torch_globals(num_threads=1):
        m_off, m_on = _build(0), _build(0, trunk_ffn_by_kind="on")
        _scramble_trunk(m_off, 7)
        _scramble_trunk(m_on, 7)
        for r in _rounds(m_on):
            K.copy_shared_into_experts(r)
        ref, got = _record(m_off, obs, mask), _record(m_on, obs, mask)
        # ...and an UNTIED expert moves the output (the comparison has teeth)
        with torch.no_grad():
            _rounds(m_on)[1].kind_ffn["event"].linear2.bias.add_(0.1)
        moved = _record(m_on, obs, mask)
    assert _same(ref, got)
    assert not _same(ref, moved)


@pytest.mark.parametrize("kind", K.KINDS)
def test_k4_perturbing_one_expert_moves_exactly_its_kinds_tokens(rows, kind: str) -> None:
    """K-4. A random shift of expert ``kind``'s output bias changes, at its round's output, EXACTLY the positions the
    table gives that kind (every row). Fails on any mis-routing between kinds."""
    obs, mask = rows
    with torch_globals(num_threads=1):
        model = _build(0, trunk_ffn_by_kind="on")
        r0 = _rounds(model)[0]
        out: List[torch.Tensor] = []
        h = r0.register_forward_hook(lambda _m, _a, o: out.append(o.detach().clone()))
        try:
            _record(model, obs[:16], mask[:16])
            mod = r0 if kind == "mon" else r0.kind_ffn[kind]
            with torch.no_grad():
                mod.linear2.bias.add_(torch.randn(mod.linear2.bias.shape, generator=torch.Generator().manual_seed(3)))
            _record(model, obs[:16], mask[:16])
        finally:
            h.remove()
    assert len(out) % 2 == 0 and len(out) >= 2                 # one call per extractor forward (policy, value)
    changed = (out[0] != out[len(out) // 2]).any(dim=2)                                  # [B, n]
    k = K.KINDS.index(kind)
    want = torch.tensor([kk == k for s, e, kk in r0.ffn_segments for _ in range(s, e)])
    assert bool(changed.all(dim=0).eq(want).all()) and bool((changed.any(dim=0) == want).all())


# ---------------------------------------------------------------------------------------------------- K-6
def test_k6_an_identity_round_with_experts_is_the_identity() -> None:
    """K-6 (first half). A fresh ``on`` build's extra round (`IdentityInitRound`) returns its input bit for bit: its
    experts' ``linear2`` start at zero like its own. Fails if the extra rounds' experts are not zero-init."""
    with torch_globals(num_threads=1):
        model = _build(0, trunk_ffn_by_kind="on")
        r = model.policy.features_extractor.team_transformer.extra_rounds[0]
        x = torch.randn(4, r.ffn_segments[-1][1], r.norm1.normalized_shape[0], generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            y = r(x, None)
    assert r.kind_ffn is not None and torch.equal(x, y)


def test_k6_growing_depth_on_a_kffn_parent_stays_bit_identical(rows) -> None:
    """K-6 (second half). ``--grow trunk_layers=4`` on a TRAINED kffn parent (depth 3): the new round carries zero
    experts and the network's table, the gate passes, and the grown state dict is a fresh depth-4 ``on`` build's."""
    obs, mask = rows
    with torch_globals(num_threads=1):
        model = _parent(obs, mask, trunk_ffn_by_kind="on")
        before = _record(model, obs, mask)
        report = G.apply_growth(model, {"trunk_layers": {"parent": 3, "child": 4}}, rows=(obs, mask))
        after = _record(model, obs, mask)
        fresh = _build(0, trunk_ffn_by_kind="on", trunk_layers=4)
    assert report["function_check"]["bit_identical"] and _same(before, after)
    new = model.policy.features_extractor.team_transformer.extra_rounds[1]
    assert new.ffn_segments == _rounds(model)[0].ffn_segments
    assert all(float(e.linear2.weight.detach().abs().max()) == 0.0 for e in new.kind_ffn.values())
    sd, sd4 = model.policy.state_dict(), fresh.policy.state_dict()
    assert list(sd) == list(sd4) and all(sd[k].shape == sd4[k].shape for k in sd)
    assert report["new_parameters"]["elements"] == 65_920 * 5 + 2 * 128 * 2 + 128 * 384 + 384 + 128 * 128 + 128


# ---------------------------------------------------------------------------------------------------- K-7, K-8
def test_k7_the_grow_fork_is_the_parents_function_and_copies_the_optimizer_state(rows) -> None:
    """K-7. A production parent after one AdamW step, ``off -> on``: the gate's CPU builds AND the live child are the
    parent's function bit for bit; the grown state dict loads strictly into (and IS, key for key in order) a fresh
    ``on`` build; every parent parameter's AdamW state object is carried; every expert parameter holds an independent
    CLONE of its shared source's state (ruling Q-K3). Fails on revert of the copy, the key layout or the optimizer rule."""
    obs, mask = rows
    with torch_globals(num_threads=1):
        model = _parent(obs, mask)
        old_opt = model.policy.optimizer
        old_params = list(old_opt.param_groups[0]["params"])
        old_state = {id(p): old_opt.state.get(p, {}) for p in old_params}
        before = _record(model, obs, mask)
        keys_before = list(model.policy.state_dict())
        report = G.apply_growth(model, FIELDS, rows=(obs, mask))
        after = _record(model, obs, mask)
        fresh = _build(0, trunk_ffn_by_kind="on")
    fc = report["function_check"]
    assert fc["bit_identical"] and fc["max_abs_d_logits"] == 0.0 and fc["max_abs_d_value"] == 0.0 and fc["rows"] == 64
    assert _same(before, after)
    sd, sdf = model.policy.state_dict(), fresh.policy.state_dict()
    assert list(sd) == list(sdf) and all(sd[k].shape == sdf[k].shape for k in sd)
    assert [k for k in sd if k not in keys_before] == [k for k in sdf if ".kind_ffn." in k]
    assert report["new_parameters"] == {"tensors": 48, "elements": EXPERT_ELEMENTS}
    opt = model.policy.optimizer
    with_state = [p for p in old_params if old_state[id(p)]]
    assert report["optimizer_state"] == {"carried": len(with_state), "no_state": len(old_params) - len(with_state),
                                         "fresh": 0, "copied": 48}
    assert len(with_state) > 200 and all(opt.state[p] is old_state[id(p)] for p in with_state)
    for r in _rounds(model):
        for e in r.kind_ffn.values():
            for lin in ("linear1", "linear2"):
                for attr in ("weight", "bias"):
                    p, src = getattr(getattr(e, lin), attr), getattr(getattr(r, lin), attr)
                    assert torch.equal(p, src)
                    st, s0 = opt.state[p], opt.state[src]
                    assert set(st) == set(s0) and all(torch.equal(st[k], s0[k]) for k in s0)
                    assert all(st[k].data_ptr() != s0[k].data_ptr() for k in s0)            # a clone, not an alias
    fe = model.policy.features_extractor
    assert fe.trunk_ffn_by_kind == "on" and model.policy_kwargs["features_extractor_kwargs"]["trunk_ffn_by_kind"] == "on"
    # the group's ORDER is a fresh `on` build's (what a later restart's by-name remap reads)
    names = {id(p): n for n, p in model.policy.named_parameters()}
    fresh_names = {id(p): n for n, p in fresh.policy.named_parameters()}
    assert [names[id(p)] for p in opt.param_groups[0]["params"]] == \
        [fresh_names[id(p)] for p in fresh.policy.optimizer.param_groups[0]["params"]]
    # and it trains as ONE trajectory: one more step gives every expert and its shared source step 2
    from agents.model.growth_test import _step
    with torch_globals(num_threads=1):
        _step(model, obs, mask)
    for r in _rounds(model):
        assert all(float(opt.state[p]["step"]) == 2.0 for p in [*r.kind_ffn.parameters(), *r.linear1.parameters(),
                                                                  *r.linear2.parameters()])


def test_k8_a_non_copy_expert_init_is_refused_by_the_gate(rows, monkeypatch: pytest.MonkeyPatch) -> None:
    """K-8. Experts left at their fresh init (the copy monkeypatched away) are NOT the parent's function: refused."""
    obs, mask = rows

    def _no_copy(r: Any) -> List[Tuple[torch.nn.Parameter, torch.nn.Parameter]]:
        return [(getattr(getattr(e, lin), a), getattr(getattr(r, lin), a))
                for e in r.kind_ffn.values() for lin in ("linear1", "linear2") for a in ("weight", "bias")]

    with torch_globals(num_threads=1):
        model = _parent(obs, mask)
        monkeypatch.setattr(K, "copy_shared_into_experts", _no_copy)
        with pytest.raises(G.GrowthError, match="NOT its parent's function"):
            G.apply_growth(model, FIELDS, rows=(obs, mask))


def test_k7_depth_and_experts_at_one_fork_apply_depth_first(rows) -> None:
    """Several fields at one fork apply in ``GROWABLE``'s order: depth FIRST, so the new round's experts are copies of
    its own (zero) FFN (its ``linear1`` included) rather than fresh experts; the result is a fresh depth-4 ``on``
    build's key set and the parent's function."""
    obs, mask = rows
    with torch_globals(num_threads=1):
        model = _parent(obs, mask)
        before = _record(model, obs, mask)
        report = G.apply_growth(model, {"trunk_ffn_by_kind": FIELDS["trunk_ffn_by_kind"],
                                        "trunk_layers": {"parent": 3, "child": 4}}, rows=(obs, mask))
        fresh = _build(0, trunk_ffn_by_kind="on", trunk_layers=4)
    assert report["function_check"]["bit_identical"] and _same(before, _record(model, obs, mask))
    new = model.policy.features_extractor.team_transformer.extra_rounds[1]
    assert all(torch.equal(e.linear1.weight, new.linear1.weight) for e in new.kind_ffn.values())
    assert list(model.policy.state_dict()) == list(fresh.policy.state_dict())
    # the new round's experts copy a parameter with no state: they start fresh; every other expert is copied
    assert report["optimizer_state"]["copied"] == 48 and report["optimizer_state"]["fresh"] == 16 + 12


# ---------------------------------------------------------------------------------------------------- K-9
def test_k9_under_bf16_the_experts_run_in_bf16_and_the_round_returns_fp32(rows) -> None:
    """K-9. `set_trunk_precision(policy, "bf16")` swaps the round classes; the experts live INSIDE the round, so their
    matmuls run under the round's autocast (bf16 out) and the round returns fp32. Fails if the experts are built as
    sibling modules outside the round class."""
    from agents.model.trunk_precision import set_trunk_precision

    obs, mask = rows
    dtypes: Dict[str, Any] = {}
    with torch_globals(num_threads=1):
        model = _build(0, trunk_ffn_by_kind="on")
        assert set_trunk_precision(model.policy, "bf16") == 3
        hooks = []
        for i, r in enumerate(_rounds(model)):
            hooks.append(r.register_forward_hook(lambda _m, _a, o, i=i: dtypes.__setitem__(f"round{i}", o.dtype)))
            for k, e in r.kind_ffn.items():
                for lin in ("linear1", "linear2"):
                    hooks.append(getattr(e, lin).register_forward_hook(
                        lambda _m, _a, o, key=f"{i}.{k}.{lin}": dtypes.__setitem__(key, o.dtype)))
            hooks.append(r.linear1.register_forward_hook(lambda _m, _a, o, i=i: dtypes.__setitem__(f"{i}.mon", o.dtype)))
        try:
            _record(model, obs, mask)
        finally:
            for h in hooks:
                h.remove()
            set_trunk_precision(model.policy, "fp32")
    assert len(dtypes) == 3 * (1 + 4 * 2 + 1)
    assert all(v == torch.float32 for k, v in dtypes.items() if k.startswith("round"))
    assert all(v == torch.bfloat16 for k, v in dtypes.items() if not k.startswith("round"))


# ---------------------------------------------------------------------------------------------------- K-10 (model side)
def test_k10_on_without_the_five_kinds_and_on_to_off_are_refused(rows) -> None:
    """K-10 (the model side). The toy argv with ``on``: a typed constructor refusal naming the missing kind; a growth
    ``off -> on`` of the toy: a typed `GrowthError`; ``on -> off``: refused; a bad value: refused."""
    from agents.model.adopted_levers_test import _toy_args
    from main.fresh_checkpoint import build_fresh_model

    toy = copy.deepcopy(_toy_args())
    toy.trunk_ffn_by_kind = "on"
    with torch_globals(num_threads=1), pytest.raises(ValueError, match="trunk_ffn_by_kind='on' requires"):
        build_fresh_model(0, args=toy)
    with pytest.raises(G.GrowthError, match="not function-preserving"):
        G.validate_step("trunk_ffn_by_kind", "on", "off")
    with pytest.raises(G.GrowthError, match="must be one of"):
        G.parse_value("trunk_ffn_by_kind", "sideways")
    assert G.parse_value("trunk_ffn_by_kind", " on ") == "on"
    from agents.model.growth_test import _parent as toy_parent
    obs, mask = rows
    with torch_globals(num_threads=1):
        model, _o, _m = toy_parent()
        with pytest.raises(G.GrowthError, match="needs all five token kinds"):
            G.apply_growth(model, FIELDS, rows=(obs, mask))


# ---------------------------------------------------------------------------------------------------- K-12 (model side)
def test_k12_versioning_migrates_to_off_and_the_mismatch_names_grow() -> None:
    """K-12 (the model side; the restart round trip is `grow_fork_test`). A pre-v161 config migrates to ``off``; a
    mismatched load names ``--grow trunk_ffn_by_kind=on``; a declared growth passes `check_compatible_grown`; the
    recorded version of a built ``on`` network says ``on``."""
    from agents.model.model_version import ModelVersion, ModelVersionError
    from agents.model.model_version.migrations import _migrate_config
    from agents.model.snapshot import recorded_model_version

    with torch_globals(num_threads=1):
        off = recorded_model_version(_build(0))
        on = recorded_model_version(_build(0, trunk_ffn_by_kind="on"))
    assert off.trunk_ffn_by_kind == "off" and on.trunk_ffn_by_kind == "on"
    old = dataclasses.asdict(off)
    old.pop("trunk_ffn_by_kind")
    old["config_version"] = 160
    migrated = _migrate_config(dict(old))
    assert migrated["trunk_ffn_by_kind"] == "off" and migrated["config_version"] >= 161
    with pytest.raises(ModelVersionError, match="(?s)trunk_ffn_by_kind mismatch.*--grow trunk_ffn_by_kind=on"):
        on.check_compatible(off)
    on.check_compatible_grown(off, FIELDS)
    with pytest.raises(ModelVersionError, match="trunk_ffn_by_kind mismatch"):
        off.check_compatible(on)
    assert isinstance(ModelVersion, type)


# ---------------------------------------------------------------------------------------------------- K-11 (slow)
#: Compile noise on the CPU (Inductor reorders fp32 arithmetic), MEASURED 2026-10-20 by a matched control on these 64 rows
#: under this test's exact procedure (seeded FFN perturbation, fullgraph, static batch, outputs of magnitude ~5): the OFF
#: production extractor reads 1.08e-5 (grad) / 1.13e-5 (no_grad), the kffn one 1.30e-5 / 1.35e-5 — the shared FFN's own
#: noise, ~1.2x. `cpu_compile_sites_slow_test`'s 1e-4 is the bar (a wrong kernel or a dropped segment is O(1e-1) or more,
#: K-3's teeth); 1e-5 failed production itself.
COMPILE_ATOL = 1e-4


@pytest.mark.slow  # a cold CPU Inductor compile (fullgraph) of the production kffn extractor + its backward
@pytest.mark.skipif(os.environ.get("GEN3AI_SKIP_COMPILE_TESTS") == "1", reason="GEN3AI_SKIP_COMPILE_TESTS=1")
def test_k11_the_kffn_extractor_compiles_fullgraph_and_matches_eager(rows) -> None:
    """K-11. The ``on`` production extractor (its experts DIVERGED: each perturbed) compiles with ``fullgraph=True``
    (a graph break raises) at the gate's batch, and the compiled forward matches eager within compile noise, and its
    gradient on every expert parameter within R1's fresh per-parameter bar (`compile_regions.R1_PARAM_BAR`)."""
    from agents.model.compile_regions import R1_PARAM_BAR
    from agents.model.extra_obs_keys import zero_extra_obs
    from agents.model.inductor_compile import compile_on

    obs, mask = rows
    with torch_globals(num_threads=1):
        try:
            torch._dynamo.reset()
            model = _build(0, trunk_ffn_by_kind="on")
            g = torch.Generator().manual_seed(11)
            with torch.no_grad():
                for r in _rounds(model):
                    for e in r.kind_ffn.values():
                        for lin in (e.linear1, e.linear2):
                            lin.weight.add_(0.02 * torch.randn(lin.weight.shape, generator=g))
                            lin.bias.add_(0.02 * torch.randn(lin.bias.shape, generator=g))
            fe = model.policy.features_extractor
            fe.train(False)
            o = {"observation": torch.as_tensor(np.ascontiguousarray(obs)),
                 "action_mask": torch.as_tensor(mask, dtype=torch.float32)}
            o.update(zero_extra_obs(fe, batch=len(obs)))
            experts = [p for r in _rounds(model) for p in r.kind_ffn.parameters()]

            def run(fn: Any) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
                for p in experts:
                    p.grad = None
                pi, vf = fn(o)
                (pi.square().mean() + vf.square().mean()).backward()
                return [pi.detach(), vf.detach()], [p.grad.detach().clone() for p in experts]

            eager_out, eager_g = run(fe.forward)
            compiled = compile_on(fe.forward, device="cpu", dynamic=False, fullgraph=True)
            comp_out, comp_g = run(compiled)
        finally:
            torch._dynamo.reset()
    d = max(float((a - b).abs().max()) for a, b in zip(eager_out, comp_out))
    assert d <= COMPILE_ATOL, f"compiled kffn extractor is {d:.3g} from eager (bar {COMPILE_ATOL})"
    rel = max(float((a - b).norm() / max(float(a.norm()), 1e-12)) for a, b in zip(eager_g, comp_g))
    assert rel <= R1_PARAM_BAR["fresh"], f"expert gradient compiled vs eager rel {rel:.3g}"
